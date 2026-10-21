import XCTest
@testable import Cotabby

/// Pins how much of the Neural Engine KV cache a new prompt keeps. Too little costs a full prefill
/// per keystroke; too much would skip the run that produces the first token's logits.
final class NeuralEnginePrefixReuseTests: XCTestCase {
    func testEmptyCacheReusesNothing() {
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [], next: [1, 2, 3]), 0)
    }

    func testTypedDeltaReusesTheWholePreviousPrompt() {
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [1, 2, 3], next: [1, 2, 3, 4]), 3)
    }

    func testIdenticalPromptStillRerunsItsLastToken() {
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [1, 2, 3], next: [1, 2, 3]), 2)
    }

    func testBackspaceOrRetokenizationKeepsOnlyTheSharedPrefix() {
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [1, 2, 3, 4], next: [1, 2, 9]), 2)
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [7, 2, 3], next: [1, 2, 3]), 0)
    }

    func testGeneratedTokensLeftInTheCacheAreReusedWhenTheUserTypesThem() {
        // Prompt [1, 2] plus fed-back generated tokens [5, 6]; the user accepted them.
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [1, 2, 5, 6], next: [1, 2, 5, 6, 8]), 4)
    }

    func testSingleTokenPromptReusesNothing() {
        XCTAssertEqual(NeuralEnginePrefixReuse.reusableTokenCount(processed: [1], next: [1]), 0)
    }
}
