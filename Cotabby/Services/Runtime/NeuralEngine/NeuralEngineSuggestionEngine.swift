import Foundation
import Logging

/// File overview:
/// The Neural Engine suggestion engine: turns a `SuggestionRequest` into a normalized ghost-text
/// suggestion using a base model that runs on Apple's Neural Engine through Core AI, instead of
/// llama.cpp on the GPU. Same contract and shape as `LlamaSuggestionEngine` on purpose: the prompt
/// (`BaseCompletionPromptRenderer`), the request-to-options mapping, the output normalizer, token
/// healing and the stop policy are all shared, so the two engines differ only in where the model
/// runs and the user sees the same kind of suggestion from either.
///
/// What it does not have, compared with llama: a byte-prefix cache hint (the core compares exact
/// tokens, which is already cheap), a per-request KV reset on context changes (exact-prefix reuse
/// makes stale state impossible), and any fallback to another engine (by design: the point of
/// this engine is that nothing runs on the GPU).
///
/// Owned by `SuggestionEngineRouter` for the app's life; collaborates with
/// `NeuralEngineRuntimeManager`, which owns model state.
@MainActor
final class NeuralEngineSuggestionEngine {
    private let runtimeManager: NeuralEngineRuntimeManager
    /// The focus-time prefill in flight. A real request cancels it so it never waits behind a
    /// warmup for a prompt the user has already typed past.
    private var inflightPrewarmTask: Task<Void, Never>?

    init(runtimeManager: NeuralEngineRuntimeManager) {
        self.runtimeManager = runtimeManager
    }

    /// Best-effort prefill of the newly focused field's prompt; failures cost nothing the cold
    /// path would not have paid anyway, so they are only logged.
    func prewarm(for request: SuggestionRequest) async {
        inflightPrewarmTask?.cancel()
        guard runtimeManager.isReady else { return }
        let options = LlamaSuggestionEngine.makeGenerationOptions(for: request)
        let task = Task { [runtimeManager] in
            do {
                try await runtimeManager.prefill(prompt: request.prompt, options: options)
            } catch {
                CotabbyLogger.suggestion.debug(
                    "Neural Engine prewarm skipped: \(error.localizedDescription)",
                    metadata: ["request_id": .string(request.requestID), "engine": .string("neural_engine")]
                )
            }
        }
        inflightPrewarmTask = task
        await task.value
    }

    func generateSuggestion(for request: SuggestionRequest) async throws -> SuggestionResult {
        try await generateSuggestion(for: request, onPartial: nil)
    }

    /// Cumulative raw partials from the core are normalized and delivered on the main actor while
    /// the decode runs; the returned result stays the authoritative completion.
    func generateSuggestion(
        for request: SuggestionRequest,
        onPartial: (@MainActor (SuggestionResult) -> Void)?
    ) async throws -> SuggestionResult {
        let baseMetadata: Logger.Metadata = [
            "request_id": .string(request.requestID),
            "engine": .string("neural_engine")
        ]
        do {
            inflightPrewarmTask?.cancel()
            inflightPrewarmTask = nil

            let startTime = Date()
            let options = LlamaSuggestionEngine.makeGenerationOptions(for: request)
            var onPartialRawText: (@Sendable (String) -> Void)?
            if let onPartial {
                onPartialRawText = { raw in
                    // Called on the core actor; each hop is an independent task, so a shorter
                    // cumulative can land after a longer one. The coordinator's monotonic render
                    // policy makes that harmless, exactly as for llama.
                    Task { @MainActor in
                        let normalized = SuggestionTextNormalizer.normalizeDetailed(raw, for: request).text
                        guard !normalized.isEmpty else { return }
                        onPartial(SuggestionResult(
                            generation: request.generation,
                            rawText: raw,
                            text: normalized,
                            latency: Date().timeIntervalSince(startTime),
                            spacingIsExact: true
                        ))
                    }
                }
            }
            let output = try await runtimeManager.generate(
                prompt: request.prompt,
                options: options,
                onPartialRawText: onPartialRawText
            )
            try Task.checkCancellation()

            let rawSuggestion = output.text
            let normalization = output.suppressedByLowConfidence
                ? SuggestionNormalizationResult(text: "", suppression: .lowConfidence)
                : SuggestionTextNormalizer.normalizeDetailed(rawSuggestion, for: request)
            let latency = Date().timeIntervalSince(startTime)
            let suppressionReason = normalization.suppression?.rawValue ?? "none"
            let resultMetadata = baseMetadata.merging([
                "raw_chars": .stringConvertible(rawSuggestion.count),
                "normalized_chars": .stringConvertible(normalization.text.count),
                "suppression_reason": .string(suppressionReason),
                "latency_ms": .stringConvertible(Int(latency * 1000)),
                "model": .string(runtimeManager.currentModelName ?? "unknown")
            ]) { _, new in new }
            CotabbyLogger.suggestion.debug("Neural Engine generated", metadata: resultMetadata)
            CotabbyLogger.llmIO.debug(
                "neural engine generation",
                metadata: resultMetadata.merging([
                    "prompt": .string(request.prompt),
                    "completion_raw": .string(rawSuggestion),
                    "completion_normalized": .string(normalization.text),
                    "prompt_bytes": .stringConvertible(request.prompt.utf8.count),
                    "max_tokens": .stringConvertible(request.maxPredictionTokens)
                ]) { _, new in new }
            )
            return SuggestionResult(
                generation: request.generation,
                rawText: rawSuggestion,
                text: normalization.text,
                latency: latency,
                suppressionReason: normalization.suppression?.rawValue,
                // Base model: its leading space (or none) is the model's own word boundary.
                spacingIsExact: true
            )
        } catch is CancellationError {
            throw SuggestionClientError.cancelled
        } catch LlamaRuntimeError.cancelled {
            // Superseded by a newer keystroke. Not a failure: keep the cache for the next request.
            throw SuggestionClientError.cancelled
        } catch let error as LlamaRuntimeError {
            if case .unavailable = error {
                // Not loaded yet (or failed to load): nothing in the cache to distrust.
                throw SuggestionClientError.unavailable(error.localizedDescription)
            }
            CotabbyLogger.suggestion.error(
                "Neural Engine runtime error, resetting cache: \(error.localizedDescription)",
                metadata: baseMetadata
            )
            await runtimeManager.discardCache()
            throw SuggestionClientError.generationFailed(error.localizedDescription)
        } catch let error as SuggestionClientError {
            throw error
        } catch {
            CotabbyLogger.suggestion.error(
                "Unexpected Neural Engine error, resetting cache: \(error.localizedDescription)",
                metadata: baseMetadata
            )
            await runtimeManager.discardCache()
            throw SuggestionClientError.generationFailed(error.localizedDescription)
        }
    }

    /// A context change only stops a stale warmup. The core's cache needs no reset: it reuses
    /// exactly the tokens a new prompt shares with it, so a different field simply shares fewer.
    func resetCachedGenerationContext() async {
        inflightPrewarmTask?.cancel()
        inflightPrewarmTask = nil
    }
}

extension NeuralEngineSuggestionEngine: SuggestionGenerating {}
