import Foundation

/// One Core AI model bundle the Neural Engine engine can load.
///
/// A bundle is a directory produced by Apple's `coreai.llm.export --platform iOS` (static shapes,
/// which Core AI specializes for the Neural Engine): `metadata.json`, a `tokenizer/` folder, and
/// the `.aimodel` itself. The directory name is the stable selection key persisted in settings;
/// the display name comes from the Hugging Face id the bundle was exported from.
nonisolated struct NeuralEngineModelOption: Identifiable, Equatable, Hashable, Sendable {
    let id: String
    let displayName: String
    let bundleURL: URL
}

/// Finds Neural Engine model bundles in Cotabby's Application Support folder.
///
/// Mirrors `BundledRuntimeLocator`'s layout so each app identity keeps its own models:
/// `~/Library/Application Support/<CFBundleName>/NeuralEngine/<bundle>/` next to `LlamaRuntime/`
/// (so "Cotabby Dev" and "Cotabby" never share a folder). There is no download catalog yet; bundles
/// are exported offline and dropped into this folder, which is why discovery only validates shape
/// and never fetches anything. Value type with no state beyond the folder URL, so tests point it
/// at a temporary directory.
nonisolated struct NeuralEngineModelLocator {
    static let folderName = "NeuralEngine"

    let directoryURL: URL

    init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    /// `~/Library/Application Support/<CFBundleName>/NeuralEngine`.
    static func userModelsDirectoryURL(bundle: Bundle = .main) -> URL {
        let appSupportRoot =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent("Library/Application Support", isDirectory: true)
        let appFolderName = (bundle.object(forInfoDictionaryKey: "CFBundleName") as? String) ?? "Cotabby"
        return appSupportRoot
            .appendingPathComponent(appFolderName, isDirectory: true)
            .appendingPathComponent(folderName, isDirectory: true)
    }

    /// Valid language-model bundles, sorted by display name. Unreadable or foreign folders are
    /// skipped silently: the folder is user-managed, and one stray directory must not hide the rest.
    func availableModels(fileManager: FileManager = .default) -> [NeuralEngineModelOption] {
        guard let children = try? fileManager.contentsOfDirectory(
            at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
        ) else {
            return []
        }
        return children
            .compactMap { Self.option(at: $0, fileManager: fileManager) }
            .sorted { $0.displayName.localizedStandardCompare($1.displayName) == .orderedAscending }
    }

    /// Reads one bundle's `metadata.json`. Requires `kind: "llm"`, the `main` asset on disk, and a
    /// `tokenizer/tokenizer.json` (the Neural Engine runtime reads its byte table from it).
    static func option(at bundleURL: URL, fileManager: FileManager = .default) -> NeuralEngineModelOption? {
        let metadataURL = bundleURL.appendingPathComponent("metadata.json")
        guard let data = fileManager.contents(atPath: metadataURL.path),
              let metadata = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              metadata["kind"] as? String == "llm",
              let assets = metadata["assets"] as? [String: Any],
              let mainAsset = assets["main"] as? String,
              fileManager.fileExists(atPath: bundleURL.appendingPathComponent(mainAsset).path),
              fileManager.fileExists(atPath: bundleURL.appendingPathComponent("tokenizer/tokenizer.json").path) else {
            return nil
        }
        let source = metadata["source"] as? [String: Any]
        let huggingFaceID = (source?["hf_model_id"] as? String)
            ?? ((metadata["language"] as? [String: Any])?["tokenizer"] as? String)
        let displayName = huggingFaceID.flatMap { $0.split(separator: "/").last.map(String.init) }
            ?? bundleURL.lastPathComponent
        return NeuralEngineModelOption(id: bundleURL.lastPathComponent, displayName: displayName, bundleURL: bundleURL)
    }
}
