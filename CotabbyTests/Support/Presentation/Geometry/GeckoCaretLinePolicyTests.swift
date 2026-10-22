import CoreGraphics
import XCTest
@testable import Cotabby

/// Covers `GeckoCaretLinePolicy`: a character- or line-derived caret moves onto the caret box's line
/// only when both describe the same x and end on different lines, and it keeps its own height.
final class GeckoCaretLinePolicyTests: XCTestCase {
    /// The answers Zen gave on Gemini's composer, converted to Cocoa coordinates: the caret after
    /// the previous character 64 pt above the caret box, both at x 2651.5.
    private let measuredCaretBox = CGRect(x: 2651.5, y: 727, width: 2, height: 21.5)
    private let measuredCharacterCaret = CGRect(x: 2651.5, y: 791, width: 2, height: 21.5)

    func test_shiftedCharacterCaret_movesOntoTheCaretBoxLine() {
        XCTAssertEqual(
            GeckoCaretLinePolicy.correctedCaret(caretBox: measuredCaretBox, candidate: measuredCharacterCaret),
            measuredCaretBox
        )
    }

    func test_tallCaretBox_keepsTheCharacterHeight() {
        // Later the same caret's box came back 53.5 pt tall, still ending on the line's bottom.
        // Taking it whole sized the ghost's font from 53.5 pt.
        let tallCaretBox = CGRect(x: 2651.5, y: 727, width: 2, height: 53.5)
        XCTAssertEqual(
            GeckoCaretLinePolicy.correctedCaret(caretBox: tallCaretBox, candidate: measuredCharacterCaret),
            CGRect(x: 2651.5, y: 727, width: 2, height: 21.5)
        )
    }

    func test_shiftedLineBox_movesOntoTheCaretBoxLine() {
        // The line route builds its candidate at the caret box's own x, with the line's y and height.
        let lineCaret = CGRect(x: measuredCaretBox.minX, y: 790, width: 2, height: 24)
        XCTAssertEqual(
            GeckoCaretLinePolicy.correctedCaret(caretBox: measuredCaretBox, candidate: lineCaret),
            CGRect(x: 2651.5, y: 727, width: 2, height: 24)
        )
    }

    func test_agreeingAnswers_keepTheCandidate() {
        // Same line, a couple of points of rounding apart: nothing to repair.
        let candidate = measuredCaretBox.offsetBy(dx: 1, dy: 3)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: measuredCaretBox, candidate: candidate))
    }

    func test_tallCaretBoxOnTheSameLine_keepsTheCandidate() {
        // A taller box that ends where the character line ends describes the same line.
        let tallCaretBox = CGRect(x: 2651.5, y: 791, width: 2, height: 53.5)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: tallCaretBox, candidate: measuredCharacterCaret))
    }

    func test_exactlyHalfALineApart_keepsTheCandidate() {
        let candidate = measuredCaretBox.offsetBy(dx: 0, dy: measuredCaretBox.height / 2)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: measuredCaretBox, candidate: candidate))
    }

    func test_caretBoxAtAnotherX_keepsTheCandidate() {
        // At a soft wrap the caret box can sit at the start of the next line: a different position,
        // where the character answer is the right one.
        let nextLineStart = CGRect(x: 2629.5, y: 703, width: 2, height: 21.5)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: nextLineStart, candidate: measuredCharacterCaret))
    }

    func test_emptyHeights_keepTheCandidate() {
        let flatBox = CGRect(x: 2651.5, y: 727, width: 2, height: 0)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: flatBox, candidate: measuredCharacterCaret))
        let flatCandidate = CGRect(x: 2651.5, y: 791, width: 2, height: 0)
        XCTAssertNil(GeckoCaretLinePolicy.correctedCaret(caretBox: measuredCaretBox, candidate: flatCandidate))
    }
}
