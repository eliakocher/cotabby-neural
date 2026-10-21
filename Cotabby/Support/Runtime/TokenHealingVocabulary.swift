import Foundation

/// Answers "which vocabulary tokens may be sampled while a healed prefix is still being replayed?"
///
/// Token healing (`TokenHealingPlan`) removes the half-typed word from the prompt so the model can
/// re-spell it with its own best token ("hello i want to as" can become " ask" instead of " as" +
/// "k"). While those removed bytes are replayed, every sampled token must agree with them. The
/// llama path enforces that inside CotabbyInference (`TokenHealing.cpp`); the Neural Engine path
/// samples in Swift, so this type is the byte-for-byte port of that rule. Keeping it pure and
/// separate from the sampler lets tests pin the matching table without a model.
///
/// Built once per model load from every printable token's bytes, then read-only. Entries are
/// sorted by their bytes so one binary search finds the contiguous range of tokens that start with
/// a prefix, instead of rescanning 150k vocabulary strings on every keystroke.
nonisolated struct TokenHealingVocabulary: Sendable {
    struct Entry: Sendable {
        let token: Int32
        let piece: [UInt8]
        let containsLineBreak: Bool
    }

    private let entries: [Entry]

    /// `pieces` maps token ids to their exact bytes; empty pieces (special or unused tokens) are
    /// skipped because they can never replay editor text.
    init(pieces: [(token: Int32, bytes: [UInt8])]) {
        entries = pieces
            .filter { !$0.bytes.isEmpty }
            .map { Entry(token: $0.token, piece: $0.bytes, containsLineBreak: $0.bytes.contains(10) || $0.bytes.contains(13)) }
            .sorted { $0.piece.lexicographicallyPrecedes($1.piece) }
    }

    /// Tokens that may come next while `prefix` (the bytes still to replay) is outstanding.
    ///
    /// Preference order matches the native engine. First, tokens that cover the whole remaining
    /// prefix (and may add new text after it): a short token such as " i" can have high probability
    /// only because it starts many words that contradict the typed " int", so it must not compete
    /// with " intelligence". Only when no token covers the prefix (byte-fallback spellings) are
    /// shorter exact fragments allowed; the next step applies the same rule to what remains.
    /// Line-break tokens are dropped for single-line fields before choosing a branch, so a covering
    /// token the sampler would forbid anyway cannot hide a valid shorter spelling.
    func matchingTokens(prefix: [UInt8], singleLine: Bool) -> [Int32] {
        guard !prefix.isEmpty else { return [] }
        var matches: [Int32] = []

        var index = lowerBound(prefix)
        while index < entries.count, entries[index].piece.starts(with: prefix) {
            let entry = entries[index]
            if !singleLine || !entry.containsLineBreak { matches.append(entry.token) }
            index += 1
        }
        if !matches.isEmpty { return matches }

        for length in 1 ..< prefix.count {
            let fragment = Array(prefix.prefix(length))
            var fragmentIndex = lowerBound(fragment)
            while fragmentIndex < entries.count, entries[fragmentIndex].piece == fragment {
                let entry = entries[fragmentIndex]
                if !singleLine || !entry.containsLineBreak { matches.append(entry.token) }
                fragmentIndex += 1
            }
        }
        return matches
    }

    /// First entry whose bytes are not ordered before `key` (unsigned byte comparison, like the
    /// native `std::string` ordering this ports).
    private func lowerBound(_ key: [UInt8]) -> Int {
        var low = 0
        var high = entries.count
        while low < high {
            let middle = (low + high) / 2
            if entries[middle].piece.lexicographicallyPrecedes(key) {
                low = middle + 1
            } else {
                high = middle
            }
        }
        return low
    }
}
