import Foundation

/// Cotabby's llama.cpp sampling chain, reimplemented in Swift for the Neural Engine runtime.
///
/// Core AI's `StaticShapeEngine` hands back one raw logits row per step. Everything the llama path
/// does after that inside CotabbyInference (`buildSampler`, `maskCompletionPrefix`,
/// `maskNewWordStarts`, `argmaxIsEOG`) happens here, in the same order, so both engines turn the
/// same distribution into the same kind of suggestion:
///
/// 1. Constraint: during token healing only tokens that agree with the replayed bytes survive;
///    otherwise, mid-word, tokens that start a new word are removed for the first token.
/// 2. "Does the model want to stop?": whether the constrained row's most likely token ends
///    generation, before any mask or penalty. The runtime stops on it (anti-rambling).
/// 3. Static masks: non-printable tokens always, line breaks in single-line fields.
/// 4. Repetition penalty over the last 64 tokens (prompt included, like llama.cpp's sampler,
///    which accepts every decoded prompt token).
/// 5. Temperature, top-k, min-p, top-p, then a seeded draw; or argmax when temperature is 0.
///
/// A pure function over values: the caller owns the logits, history and random generator, so
/// tests can pin every stage without a model, and the runtime stays the only stateful piece.
/// It works on one copy of the row and stamps masks in from the vocabulary's precomputed id lists,
/// so a step costs one pass over the vocabulary (well under a millisecond for 150k logits).
nonisolated enum NeuralEngineSampler {
    struct Parameters: Equatable, Sendable {
        var temperature: Double
        var topK: Int
        var topP: Double
        var minP: Double
        var repetitionPenalty: Double
        /// llama.cpp's `penalty_last_n`, fixed at 64 in CotabbyInference.
        var penaltyWindow: Int = 64
    }

    enum Constraint: Equatable, Sendable {
        case none
        /// Mid-word caret without healing: the first token must not start with whitespace.
        case continueWord
        /// Healing replay: only these tokens agree with the outstanding prefix bytes.
        case healing(allowed: [Int32])
    }

    struct Draw: Equatable, Sendable {
        let token: Int32
        /// The constrained row's argmax ends generation. Never set while healing, like the native
        /// engine: a forced replay says nothing about whether the model wants to stop.
        let argmaxIsEndOfGeneration: Bool
        /// Log-probability of `token` under the constrained raw row; nil unless requested.
        let logprob: Double?
    }

    /// Returns nil when no token survives the constraint and masks, which the runtime treats as
    /// the end of this completion (the native engine reports it as EOS).
    static func sample<G: RandomNumberGenerator>(
        logits: [Float],
        vocabulary: NeuralEngineVocabulary,
        constraint: Constraint,
        singleLine: Bool,
        history: ArraySlice<Int32>,
        parameters: Parameters,
        computeLogprob: Bool,
        using generator: inout G
    ) -> Draw? {
        let count = min(logits.count, vocabulary.size)
        guard count > 0 else { return nil }

        // 1. Constraint, applied to a working copy of the raw row (the native engine mutates its
        // logits row the same way before sampling).
        var row: [Float]
        var isHealing = false
        switch constraint {
        case .healing(let allowed):
            isHealing = true
            row = [Float](repeating: -.infinity, count: count)
            for token in allowed where token >= 0 && Int(token) < count {
                row[Int(token)] = logits[Int(token)]
            }
        case .continueWord:
            row = Array(logits.prefix(count))
            stamp(vocabulary.newWordIDs, into: &row)
        case .none:
            row = Array(logits.prefix(count))
        }

        // 2. Argmax and (optionally) the normalizer of the constrained raw row.
        var argmaxIsEnd = false
        if !isHealing, let best = argmax(row) {
            argmaxIsEnd = vocabulary.endOfGeneration.contains(Int32(best))
        }
        let logNormalizer = computeLogprob ? logSumExp(row) : nil
        let constrained = computeLogprob ? row : []

        // 3. Static masks.
        stamp(vocabulary.nonprintableIDs, into: &row)
        if singleLine { stamp(vocabulary.lineBreakIDs, into: &row) }

        // 4. Repetition penalty: each distinct token in the window once, like llama.cpp's counts.
        if parameters.repetitionPenalty > 1.0, parameters.penaltyWindow > 0 {
            let penalty = Float(parameters.repetitionPenalty)
            var seen = Set<Int32>()
            for token in history.suffix(parameters.penaltyWindow)
            where token >= 0 && Int(token) < count && seen.insert(token).inserted {
                let logit = row[Int(token)]
                guard logit.isFinite else { continue }
                row[Int(token)] = logit <= 0 ? logit * penalty : logit / penalty
            }
        }

        // 5. Selection.
        let chosen: Int?
        if parameters.temperature <= 0 {
            chosen = argmax(row)
        } else {
            chosen = stochastic(row, parameters: parameters, using: &generator)
        }
        guard let chosen else { return nil }

        var logprob: Double?
        if let logNormalizer, logNormalizer.isFinite {
            logprob = Double(constrained[chosen]) - logNormalizer
        }
        return Draw(token: Int32(chosen), argmaxIsEndOfGeneration: argmaxIsEnd, logprob: logprob)
    }

    private static func stamp(_ ids: [Int32], into row: inout [Float]) {
        let count = row.count
        row.withUnsafeMutableBufferPointer { buffer in
            for id in ids where Int(id) < count {
                buffer[Int(id)] = -.infinity
            }
        }
    }

    /// Index of the largest finite value, or nil when every value is masked.
    private static func argmax(_ row: [Float]) -> Int? {
        row.withUnsafeBufferPointer { buffer in
            var best = -Float.infinity
            var bestIndex: Int?
            for index in buffer.indices where buffer[index] > best {
                best = buffer[index]
                bestIndex = index
            }
            return bestIndex
        }
    }

    private static func logSumExp(_ row: [Float]) -> Double {
        guard let best = argmax(row) else { return -.infinity }
        let maximum = row[best]
        var sum = 0.0
        for value in row where value.isFinite {
            sum += exp(Double(value - maximum))
        }
        return Double(maximum) + log(sum)
    }

    private static func stochastic<G: RandomNumberGenerator>(
        _ row: [Float],
        parameters: Parameters,
        using generator: inout G
    ) -> Int? {
        // Temperature scaling preserves order, so top-k selects on raw values and scales after.
        // A bounded descending buffer keeps selection linear in the vocabulary for the usual k of 20;
        // k <= 0 disables top-k (every unmasked token stays a candidate), as in llama.cpp.
        let limit = parameters.topK > 0 ? parameters.topK : Int.max
        var top: [(id: Int, value: Float)] = []
        row.withUnsafeBufferPointer { buffer in
            for index in buffer.indices {
                let value = buffer[index]
                guard value.isFinite else { continue }
                if top.count == limit {
                    guard value > top[top.count - 1].value else { continue }
                    top.removeLast()
                }
                let position = top.firstIndex { $0.value < value } ?? top.count
                top.insert((index, value), at: position)
            }
        }
        guard !top.isEmpty else { return nil }

        let temperature = Float(parameters.temperature)
        var scaled = top.map { (id: $0.id, logit: $0.value / temperature) }

        // min-p (min_keep 1): keep logits within log(p) of the best one.
        if parameters.minP > 0, parameters.minP < 1 {
            let threshold = scaled[0].logit + Float(log(parameters.minP))
            scaled = Array(scaled.prefix { $0.logit >= threshold })
        }

        var probabilities = softmax(scaled.map(\.logit))

        // top-p (min_keep 1): smallest prefix whose cumulative probability reaches p.
        if parameters.topP > 0, parameters.topP < 1 {
            var cumulative = 0.0
            var keep = scaled.count
            for (index, probability) in probabilities.enumerated() {
                cumulative += probability
                if cumulative >= parameters.topP {
                    keep = index + 1
                    break
                }
            }
            scaled = Array(scaled.prefix(keep))
            probabilities = softmax(scaled.map(\.logit))
        }

        let draw = Double.random(in: 0 ..< 1, using: &generator)
        var cumulative = 0.0
        for (index, probability) in probabilities.enumerated() {
            cumulative += probability
            if draw < cumulative { return scaled[index].id }
        }
        return scaled[scaled.count - 1].id
    }

    private static func softmax(_ logits: [Float]) -> [Double] {
        guard let maximum = logits.max() else { return [] }
        let exponentials = logits.map { exp(Double($0 - maximum)) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }
}

/// Small, fast, seedable generator so the same context yields the same suggestion run to run, the
/// guarantee llama's fixed sampler seed gives the other engine. Reseeded for every request.
nonisolated struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var mixed = state
        mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
        mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
        return mixed ^ (mixed >> 31)
    }
}
