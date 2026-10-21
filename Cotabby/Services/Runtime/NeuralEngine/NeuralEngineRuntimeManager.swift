import AppKit
import Combine
import Foundation
import Logging

/// File overview:
/// The UI-facing side of the Neural Engine runtime: publishes load state and the discovered model
/// bundles, remembers which bundle the user picked, and forwards generation to
/// `NeuralEngineRuntimeCore`. It plays the role `LlamaRuntimeManager` plus `RuntimeBootstrapModel`
/// play for llama, folded into one object because this engine has no downloads, no fallback role
/// and no per-power-source model.
///
/// `@MainActor` because SwiftUI observes it. All model work happens inside the core actor; this
/// object only owns state transitions. Loads and unloads run through one serial chain so a quick
/// engine toggle (unload, then load) can never complete in the wrong order.
///
/// Lifetime: built once by `CotabbyAppEnvironment`; `AppDelegate` starts it while the Neural
/// Engine is the selected engine and stops it otherwise, so the model only occupies memory and
/// the Neural Engine while it is in use.
@MainActor
final class NeuralEngineRuntimeManager: ObservableObject {
    @Published private(set) var state: RuntimeBootstrapState = .idle
    @Published private(set) var availableModels: [NeuralEngineModelOption] = []
    /// Persisted bundle directory name; nil until the user picks one (the first bundle is used).
    @Published private(set) var selectedModelID: String?
    /// True while the current load is the one-time Neural Engine specialization (minutes, no
    /// suggestions meanwhile) rather than a load from the compile cache (seconds). A separate flag
    /// instead of parsing `state`'s message, so the menu bar and Home can say "preparing" without
    /// depending on the wording of the Settings status line.
    @Published private(set) var isPreparing = false

    static let selectedModelDefaultsKey = "cotabbyNeuralEngineSelectedModel"

    private let core = NeuralEngineRuntimeCore()
    private let locator: NeuralEngineModelLocator
    private let defaults: UserDefaults
    private var lifecycleTail: Task<Void, Never>?
    /// The bundle the core holds (or is loading) as far as this manager has asked for.
    private var requestedModelID: String?
    private var wantsModelLoaded = false

    init(
        locator: NeuralEngineModelLocator = NeuralEngineModelLocator(
            directoryURL: NeuralEngineModelLocator.userModelsDirectoryURL()
        ),
        defaults: UserDefaults = .standard
    ) {
        self.locator = locator
        self.defaults = defaults
        selectedModelID = defaults.string(forKey: Self.selectedModelDefaultsKey)
        refreshAvailableModels()
    }

    /// The bundle that is (or will be) loaded: the user's pick when it is still on disk, else the
    /// first discovered bundle.
    var selectedModel: NeuralEngineModelOption? {
        availableModels.first { $0.id == selectedModelID } ?? availableModels.first
    }

    /// Label for performance metrics and logs.
    var currentModelName: String? {
        selectedModel?.displayName
    }

    var modelsDirectoryURL: URL {
        locator.directoryURL
    }

    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    func refreshAvailableModels() {
        availableModels = locator.availableModels()
    }

    /// Re-scans the folder after the user drops in a bundle, and loads it right away when this
    /// engine is in use and was waiting for a model.
    func rescanModels() {
        refreshAvailableModels()
        if wantsModelLoaded {
            startIfNeeded()
        }
    }

    /// Creates the drop-in folder when missing and opens it in Finder.
    func openModelsDirectory() {
        try? FileManager.default.createDirectory(at: modelsDirectoryURL, withIntermediateDirectories: true)
        NSWorkspace.shared.open(modelsDirectoryURL)
    }

    /// Loads the selected bundle unless it is already loaded or loading.
    func startIfNeeded() {
        wantsModelLoaded = true
        refreshAvailableModels()
        guard let model = selectedModel else {
            requestedModelID = nil
            isPreparing = false
            state = .failed(
                "No Neural Engine model found. Put an exported Core AI model folder in \(modelsDirectoryURL.path)."
            )
            return
        }
        guard requestedModelID != model.id || state.failureDetail != nil else { return }
        load(model)
    }

    /// Unloads the model so it no longer holds memory while another engine is selected.
    func stop() {
        wantsModelLoaded = false
        guard requestedModelID != nil || state != .idle else { return }
        requestedModelID = nil
        state = .idle
        isPreparing = false
        enqueueLifecycle { [core] in
            await core.unload()
        }
    }

    /// Persists the pick and, when the engine is in use, swaps the loaded model.
    func selectModel(id: String) {
        guard availableModels.contains(where: { $0.id == id }) else { return }
        selectedModelID = id
        defaults.set(id, forKey: Self.selectedModelDefaultsKey)
        if wantsModelLoaded {
            startIfNeeded()
        }
    }

    // MARK: - Generation

    func generate(
        prompt: String,
        options: LlamaGenerationOptions,
        onPartialRawText: (@Sendable (String) -> Void)?
    ) async throws -> LlamaGenerationOutput {
        try requireReady()
        do {
            let output = try await core.generate(prompt: prompt, options: options, onPartialRawText: onPartialRawText)
            // The core returns its partial text when cancelled between tokens; surface the cancel
            // so a superseded request never reaches the screen.
            try Task.checkCancellation()
            return output
        } catch is CancellationError {
            throw LlamaRuntimeError.cancelled
        }
    }

    func prefill(prompt: String, options: LlamaGenerationOptions) async throws {
        try requireReady()
        try await core.prefill(prompt: prompt, options: options)
    }

    /// Only for error recovery; see `NeuralEngineRuntimeCore.discardCache`.
    func discardCache() async {
        await core.discardCache()
    }

    // MARK: - Private

    /// While the model is not ready, requests fail fast instead of queueing behind a load that can
    /// take minutes on first use.
    private func requireReady() throws {
        guard isReady else {
            throw LlamaRuntimeError.unavailable(
                state.failureDetail ?? (wantsModelLoaded ? state.summary : "The Neural Engine model is not loaded.")
            )
        }
    }

    private func load(_ model: NeuralEngineModelOption) {
        requestedModelID = model.id
        let wasPrepared = NeuralEngineRuntimeCore.isPrepared(model)
        isPreparing = !wasPrepared
        state = .loading(
            wasPrepared
                ? "Loading \(model.displayName) on the Neural Engine…"
                : "Preparing \(model.displayName) for the Neural Engine. This one-time setup takes several minutes…"
        )
        enqueueLifecycle { [weak self, core] in
            do {
                let report = try await core.load(model)
                guard let self, self.requestedModelID == model.id else { return }
                self.isPreparing = false
                self.state = .ready("\(model.displayName) on the Neural Engine")
                CotabbyLogger.runtime.info(
                    "Neural Engine ready",
                    metadata: [
                        "engine": .string("neural_engine"),
                        "model": .string(model.displayName),
                        "was_prepared": .stringConvertible(report.wasPrepared),
                        "seconds": .string(String(format: "%.1f", report.seconds))
                    ]
                )
            } catch {
                guard let self, self.requestedModelID == model.id else { return }
                self.isPreparing = false
                self.state = .failed("Could not load \(model.displayName): \(error.localizedDescription)")
                CotabbyLogger.runtime.error(
                    "Neural Engine load failed",
                    metadata: ["engine": .string("neural_engine"), "error": .string(error.localizedDescription)]
                )
            }
        }
    }

    /// Runs lifecycle work strictly in submission order on the main actor's task chain.
    private func enqueueLifecycle(_ operation: @escaping @MainActor () async -> Void) {
        let previous = lifecycleTail
        lifecycleTail = Task {
            await previous?.value
            await operation()
        }
    }
}
