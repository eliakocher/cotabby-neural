import XCTest
@testable import Cotabby

/// Pins each stage of the Swift port of Cotabby's llama sampling chain. The Neural Engine engine
/// relies on it for everything llama.cpp does natively: masks, the mid-word and healing
/// constraints, the "model wants to stop" signal, the repetition penalty and the filters.
final class NeuralEngineSamplerTests: XCTestCase {
    private typealias V = NeuralEngineTestVocabulary
    private let vocabulary = V.make()
    private let greedy = NeuralEngineSampler.Parameters(
        temperature: 0, topK: 20, topP: 1, minP: 0, repetitionPenalty: 1
    )

    /// Logits with `peaks` set and everything else very unlikely.
    private func logits(_ peaks: [Int32: Float]) -> [Float] {
        var row = [Float](repeating: -10, count: V.size)
        for (token, value) in peaks { row[Int(token)] = value }
        return row
    }

    private func draw(
        _ row: [Float],
        constraint: NeuralEngineSampler.Constraint = .none,
        singleLine: Bool = false,
        history: [Int32] = [],
        parameters: NeuralEngineSampler.Parameters? = nil,
        computeLogprob: Bool = false,
        seed: UInt64 = 1
    ) -> NeuralEngineSampler.Draw? {
        var generator = SplitMix64(seed: seed)
        return NeuralEngineSampler.sample(
            logits: row,
            vocabulary: vocabulary,
            constraint: constraint,
            singleLine: singleLine,
            history: history[...],
            parameters: parameters ?? greedy,
            computeLogprob: computeLogprob,
            using: &generator
        )
    }

    func testGreedyPicksTheLargestLogit() {
        XCTAssertEqual(draw(logits([V.hello: 1, V.world: 3]))?.token, V.world)
    }

    func testNonprintableTokensAreNeverSampled() {
        let row = logits([V.markerInVocab: 9, V.beginOfText: 8, V.unassigned: 7, V.hello: 1])
        XCTAssertEqual(draw(row)?.token, V.hello)
    }

    func testSingleLineMasksLineBreaks() {
        let row = logits([V.newline: 5, V.the: 2])
        XCTAssertEqual(draw(row, singleLine: false)?.token, V.newline)
        XCTAssertEqual(draw(row, singleLine: true)?.token, V.the)
    }

    func testContinueWordForbidsWhitespaceLedTokens() {
        let row = logits([V.world: 5, V.ing: 2])
        XCTAssertEqual(draw(row, constraint: .continueWord)?.token, V.ing)
    }

    func testHealingAllowsOnlyTheMatchingTokens() {
        let row = logits([V.hello: 9, V.int: 2, V.intelligence: 4])
        let allowed = vocabulary.healing.matchingTokens(prefix: Array(" int".utf8), singleLine: true)
        XCTAssertEqual(draw(row, constraint: .healing(allowed: allowed))?.token, V.intelligence)
    }

    func testNoSurvivingCandidateReturnsNil() {
        XCTAssertNil(draw(logits([V.hello: 1]), constraint: .healing(allowed: [])))
        XCTAssertNil(draw(logits([V.markerInVocab: 1]), constraint: .healing(allowed: [V.markerInVocab])))
    }

    func testArgmaxEndOfTextIsReportedAndSampleable() {
        let result = draw(logits([V.endOfText: 6, V.hello: 5]))
        XCTAssertEqual(result?.token, V.endOfText)
        XCTAssertEqual(result?.argmaxIsEndOfGeneration, true)
    }

    func testArgmaxIsMeasuredBeforeTheMasksButAfterTheConstraint() {
        // A non-printable argmax is not "the model wants to stop".
        XCTAssertEqual(draw(logits([V.markerInVocab: 9, V.hello: 1]))?.argmaxIsEndOfGeneration, false)
        // While healing, the replay decides; a stop is never reported (native parity).
        let healing = draw(
            logits([V.endOfText: 9, V.intelligence: 1]),
            constraint: .healing(allowed: [V.intelligence])
        )
        XCTAssertEqual(healing?.token, V.intelligence)
        XCTAssertEqual(healing?.argmaxIsEndOfGeneration, false)
    }

    func testRepetitionPenaltyDividesPositiveAndMultipliesNegativeLogits() {
        var parameters = greedy
        parameters.repetitionPenalty = 1.2
        // 2.0 / 1.2 = 1.67 < 1.9, so the repeated token loses.
        XCTAssertEqual(draw(logits([V.hello: 2.0, V.world: 1.9]), history: [V.hello], parameters: parameters)?.token, V.world)
        // -1.0 * 1.2 = -1.2 < -1.1.
        XCTAssertEqual(draw(logits([V.hello: -1.0, V.world: -1.1]), history: [V.hello], parameters: parameters)?.token, V.world)
        // Outside the 64-token window the penalty no longer applies.
        let distant = [V.hello] + [Int32](repeating: V.the, count: 64)
        XCTAssertEqual(draw(logits([V.hello: 2.0, V.world: 1.9]), history: distant, parameters: parameters)?.token, V.hello)
    }

    func testTopKOfOneIsDeterministic() {
        let parameters = NeuralEngineSampler.Parameters(temperature: 1, topK: 1, topP: 1, minP: 0, repetitionPenalty: 1)
        for seed in 1 ... 20 {
            XCTAssertEqual(draw(logits([V.hello: 1.0, V.world: 0.9]), parameters: parameters, seed: UInt64(seed))?.token, V.hello)
        }
    }

    func testMinPAndTopPCutTheTail() {
        // exp(-2) = 0.135 of the best: below min-p 0.5, so only the best survives.
        let minP = NeuralEngineSampler.Parameters(temperature: 1, topK: 0, topP: 1, minP: 0.5, repetitionPenalty: 1)
        // softmax([2, 0]) = [0.88, 0.12]: the first token alone reaches top-p 0.7.
        let topP = NeuralEngineSampler.Parameters(temperature: 1, topK: 0, topP: 0.7, minP: 0, repetitionPenalty: 1)
        for seed in 1 ... 20 {
            XCTAssertEqual(draw(logits([V.hello: 2, V.world: 0]), parameters: minP, seed: UInt64(seed))?.token, V.hello)
            XCTAssertEqual(draw(logits([V.hello: 2, V.world: 0]), parameters: topP, seed: UInt64(seed))?.token, V.hello)
        }
    }

    func testSameSeedGivesTheSameDraw() {
        let parameters = NeuralEngineSampler.Parameters(temperature: 1, topK: 0, topP: 1, minP: 0, repetitionPenalty: 1)
        let row = logits([V.hello: 1, V.world: 1, V.the: 1, V.ing: 1])
        let first = (1 ... 10).map { draw(row, parameters: parameters, seed: UInt64($0))?.token }
        let second = (1 ... 10).map { draw(row, parameters: parameters, seed: UInt64($0))?.token }
        XCTAssertEqual(first, second)
        XCTAssertGreaterThan(Set(first).count, 1, "Different seeds should explore the tied tokens")
    }

    func testLogprobIsComputedOverTheConstrainedRow() throws {
        let row = [Float](repeating: 0, count: V.size)
        let unconstrained = try XCTUnwrap(draw(row, computeLogprob: true)?.logprob)
        XCTAssertEqual(unconstrained, log(1.0 / Double(V.size)), accuracy: 1e-6)
        let healing = try XCTUnwrap(
            draw(row, constraint: .healing(allowed: [V.int, V.intelligence]), computeLogprob: true)?.logprob
        )
        XCTAssertEqual(healing, log(0.5), accuracy: 1e-6)
        XCTAssertNil(draw(row)?.logprob)
    }
}
