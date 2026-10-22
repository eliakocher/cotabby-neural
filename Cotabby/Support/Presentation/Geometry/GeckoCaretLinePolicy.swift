import CoreGraphics

/// Decides which visual line the caret is on in a Gecko (Firefox-engine) host when its
/// Accessibility answers disagree.
///
/// Gecko answers two kinds of geometry query for the same caret: the zero-length
/// `AXBoundsForRange` at the caret (the caret box), and ranges that cover characters (the character
/// before the caret, or the caret's whole line). On some pages the character ranges, and the
/// field's own `AXFrame`, come back shifted vertically while the caret box stays on the line the
/// page painted. Measured in Zen on Gemini's centred composer: the characters and the field were
/// reported 64 pt above the rendered text, unchanged minutes later. Because the field frame carries
/// the same shift, the resolver's distance-from-field check cannot catch it; only the two answers
/// disagreeing reveals it.
///
/// Each answer is right about one thing. The caret box's bottom edge sits on the rendered line's
/// bottom, but its height wanders: the same caret on the same text came back 21.5 pt tall and later
/// 53.5 pt tall, both ending at the line's bottom. The character answer is shifted but keeps the
/// real line height. The repaired caret takes its line from the caret box's bottom and its height
/// from the character answer; taking the caret box whole sized the ghost's font from 53.5 pt.
///
/// The repair runs only when both answers describe the same caret position: their x must agree (a
/// caret box at another x is a different visual position, such as the start of the next soft-wrapped
/// line, where the character answer is the right one) and their bottom edges must be more than half
/// a line apart. Agreeing answers leave the candidate untouched, so a page whose answers already
/// match sees no change.
///
/// Gecko only: TextKit reports its zero-length caret one line too high at every position (see
/// `AXTextGeometryResolver.zeroLengthCaretResult`), so the same preference would move every
/// TextEdit ghost onto the wrong line.
nonisolated enum GeckoCaretLinePolicy {
    /// How far apart the two answers' x may be and still describe the same caret. Gecko puts the
    /// caret box exactly on the previous character's trailing edge; a few points absorb rounding.
    static let horizontalTolerance: CGFloat = 3

    /// The repaired caret to use instead of `candidate` (a caret built from character or line
    /// bounds), or nil when `candidate` should stand. Both rects are in Cocoa coordinates, where
    /// `minY` is the bottom edge.
    static func correctedCaret(caretBox: CGRect, candidate: CGRect) -> CGRect? {
        guard caretBox.height > 0, candidate.height > 0,
              abs(caretBox.minX - candidate.minX) <= horizontalTolerance,
              abs(caretBox.minY - candidate.minY) > candidate.height / 2 else {
            return nil
        }
        return CGRect(x: caretBox.minX, y: caretBox.minY, width: caretBox.width, height: candidate.height)
    }
}
