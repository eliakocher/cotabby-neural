import SwiftUI

/// Neural Engine runtime status, model bundle picker, and the drop-in models folder.
/// These members are internal because Swift extensions in separate files cannot share lexical `private` access;
/// the owning view itself remains module-internal.
extension EngineAndModelPaneView {
// MARK: - Neural Engine

    @ViewBuilder
    var neuralEngineSections: some View {
        Section("Runtime") {
            LabeledContent {
                Text(neuralEngineRuntime.state.summary)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.trailing)
                    .fixedSize(horizontal: false, vertical: true)
            } label: {
                SettingsRowLabel(
                    title: "Model Status",
                    description: "Whether the model is loaded on the Neural Engine. The first load of " +
                        "each model is a one-time setup that takes several minutes; later loads are instant.",
                    systemImage: "info.circle"
                )
            }
            .settingsItem(.modelStatus)
        }

        Section("Models") {
            Text("The Neural Engine runs Core AI model bundles exported from a base model, " +
                "quietly and without using the GPU. Put each bundle folder in the folder below.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if neuralEngineRuntime.availableModels.isEmpty {
                Text("No Neural Engine models found.")
                    .foregroundStyle(.secondary)
            } else {
                Picker(selection: neuralEngineModelBinding) {
                    ForEach(neuralEngineRuntime.availableModels) { model in
                        Text(model.displayName).tag(model.id)
                    }
                } label: {
                    SettingsRowLabel(
                        title: "Selected Model",
                        description: "Which model runs on the Neural Engine. Larger models write better " +
                            "but answer more slowly.",
                        systemImage: "shippingbox"
                    )
                }
                .settingsItem(.selectedModel)
            }

            LabeledContent {
                VStack(alignment: .trailing, spacing: 8) {
                    Text(neuralEngineRuntime.modelsDirectoryURL.path)
                        .font(.callout.monospaced())
                        .textSelection(.enabled)
                        .multilineTextAlignment(.trailing)

                    HStack(spacing: 8) {
                        Button("Open Folder") {
                            neuralEngineRuntime.openModelsDirectory()
                        }

                        Button("Refresh") {
                            neuralEngineRuntime.rescanModels()
                        }
                    }
                }
            } label: {
                SettingsRowLabel(
                    title: "Models Folder",
                    description: "Where Neural Engine model bundles are stored on this Mac.",
                    systemImage: "folder"
                )
            }
            .settingsItem(.modelsFolder)
        }
    }

    var neuralEngineModelBinding: Binding<String> {
        Binding(
            get: { neuralEngineRuntime.selectedModel?.id ?? "" },
            set: { neuralEngineRuntime.selectModel(id: $0) }
        )
    }
}
