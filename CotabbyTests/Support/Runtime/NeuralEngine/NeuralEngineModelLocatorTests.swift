import XCTest
@testable import Cotabby

/// Pins which folders count as loadable Neural Engine bundles. The folder is user-managed, so
/// discovery must accept exactly the exported shape and skip anything else without failing.
final class NeuralEngineModelLocatorTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("NeuralEngineModelLocatorTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    /// Writes a bundle like `coreai.llm.export` produces; flags drop one required piece.
    @discardableResult
    private func makeBundle(
        _ name: String,
        huggingFaceID: String?,
        kind: String = "llm",
        includeAsset: Bool = true,
        includeTokenizer: Bool = true
    ) throws -> URL {
        let bundle = directory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        var metadata: [String: Any] = ["kind": kind, "assets": ["main": "\(name).aimodel"]]
        if let huggingFaceID { metadata["source"] = ["hf_model_id": huggingFaceID] }
        try JSONSerialization.data(withJSONObject: metadata).write(to: bundle.appendingPathComponent("metadata.json"))
        if includeAsset {
            try FileManager.default.createDirectory(
                at: bundle.appendingPathComponent("\(name).aimodel"), withIntermediateDirectories: true)
        }
        if includeTokenizer {
            let tokenizer = bundle.appendingPathComponent("tokenizer", isDirectory: true)
            try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
        }
        return bundle
    }

    func testValidBundlesAreListedByDisplayName() throws {
        try makeBundle("qwen3_0_6b_static", huggingFaceID: "Qwen/Qwen3-0.6B-Base")
        try makeBundle("llama_3_2_1b_static", huggingFaceID: "unsloth/Llama-3.2-1B")

        let models = NeuralEngineModelLocator(directoryURL: directory).availableModels()

        XCTAssertEqual(models.map(\.displayName), ["Llama-3.2-1B", "Qwen3-0.6B-Base"])
        XCTAssertEqual(models.map(\.id), ["llama_3_2_1b_static", "qwen3_0_6b_static"])
    }

    func testIncompleteOrForeignFoldersAreSkipped() throws {
        try makeBundle("no_tokenizer", huggingFaceID: "a/b", includeTokenizer: false)
        try makeBundle("no_asset", huggingFaceID: "a/b", includeAsset: false)
        try makeBundle("diffusion", huggingFaceID: "a/b", kind: "diffusion")
        try Data("stray".utf8).write(to: directory.appendingPathComponent("notes.txt"))

        XCTAssertEqual(NeuralEngineModelLocator(directoryURL: directory).availableModels(), [])
    }

    func testDisplayNameFallsBackToTheFolderName() throws {
        try makeBundle("my_export", huggingFaceID: nil)
        XCTAssertEqual(NeuralEngineModelLocator(directoryURL: directory).availableModels().map(\.displayName), ["my_export"])
    }

    func testMissingFolderListsNothing() {
        let missing = directory.appendingPathComponent("absent", isDirectory: true)
        XCTAssertEqual(NeuralEngineModelLocator(directoryURL: missing).availableModels(), [])
    }
}
