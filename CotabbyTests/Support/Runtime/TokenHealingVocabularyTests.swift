import XCTest
@testable import Cotabby

/// Pins the Swift port of CotabbyInference's `TokenHealingVocabulary::matchingTokens`, which
/// decides which tokens may be sampled while a half-typed word is being replayed.
final class TokenHealingVocabularyTests: XCTestCase {
    private func index(_ pieces: [String]) -> TokenHealingVocabulary {
        TokenHealingVocabulary(pieces: pieces.enumerated().map { (Int32($0.offset), Array($0.element.utf8)) })
    }

    func testTokensCoveringTheWholePrefixArePreferred() {
        let vocabulary = index([" i", " int", " intelligence", " into", "ell"])
        // " i" also starts " int..." words, but it does not cover the typed " int" and must not compete.
        XCTAssertEqual(Set(vocabulary.matchingTokens(prefix: Array(" int".utf8), singleLine: false)), [1, 2, 3])
    }

    func testFragmentsAreAllowedOnlyWhenNothingCoversThePrefix() {
        let vocabulary = index([" a", " as", "k", " b"])
        // Nothing starts with " ask": exact shorter pieces of it may replay, the rest follows next step.
        XCTAssertEqual(Set(vocabulary.matchingTokens(prefix: Array(" ask".utf8), singleLine: false)), [0, 1])
    }

    func testSingleLineDropsLineBreakTokensBeforeChoosingABranch() {
        let vocabulary = index([" hi\n", " h"])
        XCTAssertEqual(vocabulary.matchingTokens(prefix: Array(" hi".utf8), singleLine: false), [0])
        // The covering token contains a newline; in a single-line field the shorter spelling must win.
        XCTAssertEqual(vocabulary.matchingTokens(prefix: Array(" hi".utf8), singleLine: true), [1])
    }

    func testEmptyPrefixAndEmptyPiecesMatchNothing() {
        let vocabulary = index(["", "a"])
        XCTAssertEqual(vocabulary.matchingTokens(prefix: [], singleLine: false), [])
        XCTAssertEqual(vocabulary.matchingTokens(prefix: Array("a".utf8), singleLine: false), [1])
    }

    func testOrderingIsByUnsignedBytes() {
        // "é" (0xC3 0xA9) sorts after ASCII; a signed comparison would misplace it and break the range.
        let vocabulary = index(["z", "é", "éa", "a"])
        XCTAssertEqual(Set(vocabulary.matchingTokens(prefix: Array("é".utf8), singleLine: false)), [1, 2])
    }
}
