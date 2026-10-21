import XCTest
@testable import Cotabby

/// Pins how a Hugging Face `tokenizer.json` becomes the token classes the Neural Engine sampler
/// relies on. Each wrong row here would surface as a visible bug: a chat marker in ghost text, a
/// newline in a single-line field, or a model fed a prompt without the BOS it was trained with.
final class NeuralEngineVocabularyTests: XCTestCase {
    private typealias V = NeuralEngineTestVocabulary

    func testByteLevelPiecesDecodeToExactBytes() {
        let vocabulary = V.make()
        XCTAssertEqual(vocabulary.piece(V.world), Array(" world".utf8))
        XCTAssertEqual(vocabulary.piece(V.newline), [10])
        XCTAssertEqual(vocabulary.piece(V.eAcute), Array("é".utf8))
        XCTAssertEqual(vocabulary.piece(V.endOfText), [], "Special tokens render no text, like llama's control tokens")
        XCTAssertEqual(vocabulary.piece(99), [], "Out-of-range ids have no bytes")
    }

    func testNonprintableCoversSpecialUnassignedAndMarkerTokensButNotEndOfText() {
        let vocabulary = V.make()
        XCTAssertTrue(vocabulary.isNonprintable[Int(V.beginOfText)])
        XCTAssertTrue(vocabulary.isNonprintable[Int(V.unassigned)])
        XCTAssertTrue(vocabulary.isNonprintable[Int(V.markerInVocab)], "A marker shipped as plain text is still scaffolding")
        XCTAssertFalse(vocabulary.isNonprintable[Int(V.endOfText)], "End-of-text must stay sampleable so the stop fires")
        XCTAssertFalse(vocabulary.isNonprintable[Int(V.hello)])
        XCTAssertEqual(Set(vocabulary.nonprintableIDs), [V.markerInVocab, V.beginOfText, V.unassigned])
    }

    func testEndOfGenerationComesFromKnownSpecialMarkersPlusCallerExtras() throws {
        XCTAssertEqual(V.make().endOfGeneration, [V.endOfText])
        let withExtra = try NeuralEngineVocabulary(
            tokenizerJSON: V.json(), vocabularySize: V.size, additionalEndOfGeneration: [V.ing]
        )
        XCTAssertEqual(withExtra.endOfGeneration, [V.endOfText, V.ing])
        XCTAssertFalse(withExtra.isNonprintable[Int(V.ing)])
    }

    func testWhitespaceAndLineBreakClasses() {
        let vocabulary = V.make()
        XCTAssertEqual(Set(vocabulary.newWordIDs), [V.world, V.the, V.newline, V.int, V.intelligence])
        XCTAssertEqual(vocabulary.lineBreakIDs, [V.newline])
    }

    func testTemplateBOSIsDetectedInsideASequencePostProcessor() {
        XCTAssertEqual(V.make(prependsBOS: true).prependedSpecialToken, V.beginOfText)
        XCTAssertNil(V.make(prependsBOS: false).prependedSpecialToken, "Qwen3-style tokenizers add nothing")
    }

    func testHealingIndexExcludesSpecialAndMarkerTokens() {
        let vocabulary = V.make()
        XCTAssertEqual(Set(vocabulary.healing.matchingTokens(prefix: Array(" in".utf8), singleLine: true)), [V.int, V.intelligence])
        XCTAssertEqual(vocabulary.healing.matchingTokens(prefix: Array("<|im".utf8), singleLine: true), [])
    }

    func testNonByteLevelTokenizersAreRejected() {
        XCTAssertThrowsError(
            try NeuralEngineVocabulary(tokenizerJSON: V.json(decoder: "Metaspace"), vocabularySize: V.size)
        ) { error in
            guard case NeuralEngineVocabulary.LoadError.unsupportedTokenizer = error else {
                return XCTFail("Expected unsupportedTokenizer, got \(error)")
            }
        }
    }

    func testByteLevelDecodingCoversTheShiftedControlRange() {
        // Byte 0x00 is the first shifted byte (U+0100) and 0xAD (soft hyphen) the last (U+0143).
        XCTAssertEqual(NeuralEngineVocabulary.byteLevelBytes("\u{0100}"), [0x00])
        XCTAssertEqual(NeuralEngineVocabulary.byteLevelBytes("\u{0143}"), [0xAD])
        XCTAssertEqual(NeuralEngineVocabulary.byteLevelBytes("A~"), Array("A~".utf8))
    }
}
