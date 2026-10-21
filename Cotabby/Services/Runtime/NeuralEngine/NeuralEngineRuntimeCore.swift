import CoreAILanguageModels
import CoreAIShared
import Foundation
import Logging
import Tokenizers

/// File overview:
/// Owns Apple's Core AI `StaticShapeEngine` for the Neural Engine suggestion engine: loads one
/// exported model bundle, keeps the autocomplete KV cache warm across keystrokes, and runs the
/// token-by-token decode with Cotabby's own sampler. It is the Neural Engine counterpart of
/// `LlamaRuntimeCore` and follows the same contract (prompt in, raw continuation out, cumulative
/// partials during the decode), so `NeuralEngineSuggestionEngine` mirrors the llama wrapper.
///
/// Why an actor with an explicit gate: Core AI's engine is a mutable class whose `inference` and
/// `reset(to:)` are `async`. An actor alone would not serialize them, because it is re-entrant at
/// every `await`; a superseding keystroke could then rewind the cache in the middle of another
/// request's decode. `acquire()`/`release()` hand the engine to one operation at a time, in
/// arrival order, and every operation holds it across its awaits. The actor's own executor keeps
/// sampling and tokenization off the main actor.
///
/// Lifetime: created once by `NeuralEngineRuntimeManager` and kept for the app's life; models are
/// loaded and unloaded inside it.
actor NeuralEngineRuntimeCore {
    /// Everything that belongs to one loaded bundle. Replaced as a whole on model change.
    private struct LoadedModel {
        let option: NeuralEngineModelOption
        let engine: StaticShapeEngine
        let tokenizer: any Tokenizer
        let vocabulary: NeuralEngineVocabulary
        let maxContextLength: Int
    }

    private struct PreparedPrompt {
        let tokens: [Int32]
        let healingPrefix: [UInt8]
    }

    struct LoadReport: Sendable {
        /// False when Core AI had to specialize (compile) the model for this Mac first, which takes
        /// minutes once per model and OS build; later loads come from its on-disk cache.
        let wasPrepared: Bool
        let seconds: Double
    }

    private var loaded: LoadedModel?
    /// Exactly the tokens whose keys and values the engine's cache holds, in order. Always equal in
    /// length to the engine's `processedTokenCount`; reset to empty when that is unknown.
    private var processedTokens: [Int32] = []

    private var isBusy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Matches the llama path's fixed default sampler seed: the same context gives the same
    /// suggestion run to run unless a request overrides the seed.
    private static let defaultSeed: UInt32 = 0x00C0_FFEE

    // MARK: - Serialization

    private func acquire() async {
        guard isBusy else {
            isBusy = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    /// Hands the engine straight to the next waiter (it stays busy) or marks it free.
    private func release() {
        if waiters.isEmpty {
            isBusy = false
        } else {
            waiters.removeFirst().resume()
        }
    }

    // MARK: - Model lifecycle

    /// Whether Core AI already holds a specialized build of this bundle for the current OS, i.e.
    /// whether `load` will be quick. Pure file inspection; safe to call while another model runs.
    nonisolated static func isPrepared(_ option: NeuralEngineModelOption) -> Bool {
        guard let bundle = try? LanguageModelBundle(at: option.bundleURL),
              let modelURL = try? bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main) else {
            return false
        }
        return PreparedModel.isCached(at: modelURL)
    }

    func load(_ option: NeuralEngineModelOption) async throws -> LoadReport {
        await acquire()
        defer { release() }
        if let loaded, loaded.option.id == option.id {
            return LoadReport(wasPrepared: true, seconds: 0)
        }
        loaded = nil
        processedTokens = []

        let start = Date()
        let bundle = try LanguageModelBundle(at: option.bundleURL)
        let modelURL = try bundle.modelBundle.requireModelURL(for: ModelBundle.ComponentKey.main)
        let wasPrepared = PreparedModel.isCached(at: modelURL)
        CotabbyLogger.runtime.info(
            "Loading Neural Engine model",
            metadata: [
                "engine": .string("neural_engine"),
                "model": .string(option.displayName),
                "bundle_path": .string(option.bundleURL.path),
                "specialization_cached": .stringConvertible(wasPrepared)
            ]
        )
        // The tokenizer must come from the bundle's own folder. `LanguageModelBundle.loadTokenizer`
        // falls back to downloading from Hugging Face, and Cotabby never sends anything off the Mac
        // for this engine.
        guard let tokenizerDirectory = bundle.tokenizerPath else {
            throw LlamaRuntimeError.unavailable("\(option.displayName) has no embedded tokenizer.")
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let tokenizerJSON = try Data(contentsOf: tokenizerDirectory.appendingPathComponent("tokenizer.json"))
        var extraEnds: Set<Int32> = []
        if let eos = tokenizer.eosTokenId { extraEnds.insert(Int32(eos)) }
        let vocabulary = try NeuralEngineVocabulary(
            tokenizerJSON: tokenizerJSON,
            vocabularySize: bundle.vocabSize,
            additionalEndOfGeneration: extraEnds
        )

        // The slow part on a first load: Core AI specializes every static shape for the Neural
        // Engine (minutes per model, once per OS build) and caches the result on disk.
        let engine = try await StaticShapeEngine(
            configuration: ModelConfig(bundle: bundle),
            modelURL: modelURL,
            tensorData: bundle.tensorData
        )

        loaded = LoadedModel(
            option: option,
            engine: engine,
            tokenizer: tokenizer,
            vocabulary: vocabulary,
            maxContextLength: bundle.maxContextLength
        )
        let seconds = Date().timeIntervalSince(start)
        CotabbyLogger.runtime.info(
            "Neural Engine model loaded",
            metadata: [
                "engine": .string("neural_engine"),
                "model": .string(option.displayName),
                "load_seconds": .string(String(format: "%.2f", seconds)),
                "vocab_size": .stringConvertible(vocabulary.size),
                "context_window_tokens": .stringConvertible(bundle.maxContextLength),
                "prepends_bos": .stringConvertible(vocabulary.prependedSpecialToken != nil)
            ]
        )
        return LoadReport(wasPrepared: wasPrepared, seconds: seconds)
    }

    /// Releases the engine and its cache. Waits for any in-flight operation first.
    func unload() async {
        await acquire()
        defer { release() }
        if let loaded {
            CotabbyLogger.runtime.info(
                "Neural Engine model unloaded",
                metadata: ["engine": .string("neural_engine"), "model": .string(loaded.option.displayName)]
            )
        }
        loaded = nil
        processedTokens = []
    }

    /// Rewinds the cache to empty. Only needed after a failure leaves its contents in doubt:
    /// ordinary context changes are safe without it because reuse compares exact token prefixes.
    func discardCache() async {
        await acquire()
        defer { release() }
        await rewindToStart()
    }

    // MARK: - Autocomplete

    /// Decodes `prompt` into the cache without sampling, so the first suggestion in a newly focused
    /// field only pays for the typed delta (the Neural Engine half of prewarm-on-focus).
    func prefill(prompt: String, options: LlamaGenerationOptions) async throws {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let model = try requireModel()
        let prepared = try preparedPrompt(prompt, options: options, model: model)
        try await restoreCache(for: prepared.tokens, model: model)
        try Task.checkCancellation()
        _ = try await advance(to: prepared.tokens, model: model)
    }

    /// Samples a short continuation of `prompt`. `onPartialRawText` receives the cumulative raw
    /// completion after each visible token, on this actor's executor. Cancellation is checked
    /// between tokens; a cancelled decode returns what it has, and the caller decides to drop it.
    func generate(
        prompt: String,
        options: LlamaGenerationOptions,
        onPartialRawText: (@Sendable (String) -> Void)? = nil
    ) async throws -> LlamaGenerationOutput {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        let model = try requireModel()
        let prepared = try preparedPrompt(prompt, options: options, model: model)
        let reusedBefore = processedTokens.count
        try await restoreCache(for: prepared.tokens, model: model)
        let reused = processedTokens.count
        CotabbyLogger.runtime.debug(
            "Neural Engine decode start",
            metadata: [
                "engine": .string("neural_engine"),
                "prompt_tokens": .stringConvertible(prepared.tokens.count),
                "reused_tokens": .stringConvertible(reused),
                "cached_tokens_before": .stringConvertible(reusedBefore),
                "healing_bytes": .stringConvertible(prepared.healingPrefix.count),
                "max_tokens": .stringConvertible(options.maxPredictionTokens)
            ]
        )
        return try await decode(prepared, options: options, model: model, onPartialRawText: onPartialRawText)
    }

    // MARK: - Decode

    private func decode(
        _ prepared: PreparedPrompt,
        options: LlamaGenerationOptions,
        model: LoadedModel,
        onPartialRawText: (@Sendable (String) -> Void)?
    ) async throws -> LlamaGenerationOutput {
        var sequence = prepared.tokens
        var logits = try await advance(to: sequence, model: model)

        let parameters = NeuralEngineSampler.Parameters(
            temperature: options.temperature,
            topK: options.topK,
            topP: options.topP,
            minP: options.minP,
            repetitionPenalty: options.repetitionPenalty
        )
        let computeLogprob = options.confidenceFloor > -.infinity
        var generator = SplitMix64(seed: UInt64(options.seed ?? Self.defaultSeed))
        var buffer = TokenHealingBuffer(replayedPrefix: prepared.healingPrefix)
        var outstandingReplay = prepared.healingPrefix[...]
        var isFirstToken = true

        var generatedText = ""
        var tokensGenerated = 0
        var replayTokens = 0
        var sumLogprob = 0.0
        var stopReason = "budget_exhausted"

        for _ in 0 ..< options.maxPredictionTokens + prepared.healingPrefix.count {
            if Task.isCancelled {
                stopReason = "cancelled"
                break
            }

            // Same constraint order as the native engine: the healing replay wins; otherwise a
            // mid-word caret forbids whitespace-led tokens for the first token only.
            let constraint: NeuralEngineSampler.Constraint
            if !outstandingReplay.isEmpty {
                let allowed = model.vocabulary.healing.matchingTokens(
                    prefix: Array(outstandingReplay), singleLine: options.singleLine)
                guard !allowed.isEmpty else {
                    generatedText = ""
                    stopReason = "healing_prefix_unmatched"
                    break
                }
                constraint = .healing(allowed: allowed)
            } else if isFirstToken, options.forceWordContinuation {
                constraint = .continueWord
            } else {
                constraint = .none
            }

            guard let draw = NeuralEngineSampler.sample(
                logits: logits,
                vocabulary: model.vocabulary,
                constraint: constraint,
                singleLine: options.singleLine,
                history: sequence[...],
                parameters: parameters,
                computeLogprob: computeLogprob,
                using: &generator
            ) else {
                stopReason = "no_candidate"
                break
            }
            isFirstToken = false

            if model.vocabulary.endOfGeneration.contains(draw.token) {
                stopReason = "eos"
                break
            }
            // The model's own top choice was "stop" even though the sampler drew something else:
            // end here, the anti-rambling stop the sentence classifier cannot express.
            if options.stopAtArgmaxEOG, draw.argmaxIsEndOfGeneration {
                stopReason = "argmax_eog"
                break
            }

            let piece = model.vocabulary.piece(draw.token)
            outstandingReplay = outstandingReplay.dropFirst(min(outstandingReplay.count, piece.count))
            let replayWasComplete = buffer.replayComplete
            let partial = buffer.append(tokenBytes: piece)
            if buffer.hasReplayMismatch {
                generatedText = ""
                stopReason = "healing_prefix_mismatch"
                break
            }
            sequence.append(draw.token)

            if !replayWasComplete && !buffer.hasVisibleBytes {
                // Still replaying bytes the user already typed: nothing visible yet, and the
                // output-token budget is not charged.
                replayTokens += 1
                logits = try await advance(to: sequence, model: model)
                continue
            }
            generatedText = buffer.text
            tokensGenerated += 1
            sumLogprob += draw.logprob ?? 0
            // Cumulative text, not the delta, so late or reordered deliveries are harmless.
            if let partial { onPartialRawText?(partial) }

            if let earlyStop = DecodeStopPolicy.verdict(
                accumulated: generatedText,
                tokensGenerated: tokensGenerated,
                minimumTokens: options.sentenceStopMinimumTokens,
                minimumWords: options.sentenceStopMinimumWords
            ) {
                stopReason = earlyStop.rawValue
                break
            }
            if tokensGenerated >= options.maxPredictionTokens { break }
            // Feed the sampled token back only when another one is wanted: like the native
            // engine, the last sampled token is never decoded.
            logits = try await advance(to: sequence, model: model)
        }

        CotabbyLogger.runtime.debug(
            "Neural Engine decode end",
            metadata: [
                "engine": .string("neural_engine"),
                "tokens_generated": .stringConvertible(tokensGenerated),
                "replay_tokens": .stringConvertible(replayTokens),
                "chars_generated": .stringConvertible(generatedText.count),
                "stop_reason": .string(stopReason)
            ]
        )

        let averageLogprob = computeLogprob && tokensGenerated > 0 ? sumLogprob / Double(tokensGenerated) : nil
        if let averageLogprob,
           ConfidenceSuppressionPolicy.shouldSuppress(averageLogprob: averageLogprob, floor: options.confidenceFloor) {
            return LlamaGenerationOutput(text: "", averageLogprob: averageLogprob, suppressedByLowConfidence: true)
        }
        return LlamaGenerationOutput(text: generatedText, averageLogprob: averageLogprob, suppressedByLowConfidence: false)
    }

    // MARK: - Cache

    /// Rewinds the engine to the longest reusable prefix of `tokens`. Clamped to the engine's own
    /// processed count: `reset(to:)` traps on a position it has not processed.
    private func restoreCache(for tokens: [Int32], model: LoadedModel) async throws {
        let reusable = min(
            NeuralEnginePrefixReuse.reusableTokenCount(processed: processedTokens, next: tokens),
            model.engine.processedTokenCount
        )
        do {
            try await model.engine.reset(to: reusable)
            processedTokens = Array(processedTokens.prefix(reusable))
        } catch {
            await rewindToStart()
            throw error
        }
    }

    /// Runs the engine over the tokens of `sequence` it has not processed yet and returns the last
    /// position's logits. `sequence` must extend `processedTokens`.
    private func advance(to sequence: [Int32], model: LoadedModel) async throws -> [Float] {
        do {
            let (logits, _) = try await model.engine.inference(
                inputTokens: sequence,
                // Core AI always runs its own CPU sampler after a step; greedy keeps that an argmax.
                // Cotabby samples from the returned logits with its own rules instead.
                samplingConfig: .greedy,
                returnsLogits: true
            )
            processedTokens = sequence
            guard let logits else {
                throw LlamaRuntimeError.generationFailed("The Neural Engine returned no logits.")
            }
            return logits.map { Float($0) }
        } catch {
            // After a failed step the cache contents are unknown; the next request starts cold.
            await rewindToStart()
            throw error
        }
    }

    private func rewindToStart() async {
        processedTokens = []
        try? await loaded?.engine.reset(to: 0)
    }

    // MARK: - Prompt

    private func requireModel() throws -> LoadedModel {
        guard let loaded else {
            throw LlamaRuntimeError.unavailable("No Neural Engine model is loaded.")
        }
        return loaded
    }

    /// Tokenize, prepend the template's special token, heal the word at the caret, and keep the
    /// prompt inside the context window with room for the output and the healing replay.
    private func preparedPrompt(
        _ prompt: String,
        options: LlamaGenerationOptions,
        model: LoadedModel
    ) throws -> PreparedPrompt {
        var tokens = model.tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) }
        if let special = model.vocabulary.prependedSpecialToken {
            tokens.insert(special, at: 0)
        }
        guard !tokens.isEmpty else {
            throw LlamaRuntimeError.generationFailed("Tokenization returned no prompt tokens.")
        }
        // A caret inside a word keeps the unhealed prompt so the whitespace constraint applies.
        let healing = options.forceWordContinuation
            ? TokenHealingPlan.unhealed(tokens: tokens)
            : TokenHealingPlan(
                prompt: prompt, tokens: tokens, singleLine: options.singleLine,
                piece: { model.vocabulary.piece($0) }
            )
        let maxPromptTokens = max(1, model.maxContextLength - options.maxPredictionTokens - healing.replayBytes.count)
        return PreparedPrompt(
            tokens: Array(healing.promptTokens.suffix(maxPromptTokens)),
            healingPrefix: healing.replayBytes
        )
    }
}
