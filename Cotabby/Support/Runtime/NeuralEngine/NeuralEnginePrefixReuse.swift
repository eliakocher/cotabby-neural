import Foundation

/// Decides how much of the Neural Engine's KV cache the next prompt may keep.
///
/// Autocomplete re-asks the model on every keystroke with a prompt that usually extends the last
/// one by a token or two. Core AI's `StaticShapeEngine.reset(to:)` keeps the first N cached
/// positions, so only the delta is recomputed; that is what makes a warm keystroke cost one step
/// instead of a whole prefill. The rule mirrors the llama path's `reusableTokenCount`:
///
/// - keep the longest shared token prefix with what the cache actually holds (`processed`, which
///   can include generated tokens fed back during the previous decode, so typing exactly what was
///   suggested reuses those too);
/// - but always recompute at least the prompt's last token, because its logits row is what the
///   first new token is sampled from and the engine only returns logits for tokens it runs.
///
/// Pure so the boundary cases (empty cache, identical prompt, backspace) are pinned by tests;
/// `NeuralEngineRuntimeCore` additionally clamps to the engine's own processed count before
/// calling `reset(to:)`, whose precondition would otherwise crash on an out-of-range position.
nonisolated enum NeuralEnginePrefixReuse {
    static func reusableTokenCount(processed: [Int32], next: [Int32]) -> Int {
        guard next.count > 1 else { return 0 }
        var shared = 0
        let limit = min(processed.count, next.count)
        while shared < limit, processed[shared] == next[shared] {
            shared += 1
        }
        return min(shared, next.count - 1)
    }
}
