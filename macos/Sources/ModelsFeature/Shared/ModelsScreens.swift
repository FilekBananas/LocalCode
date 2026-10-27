import SwiftUI
import LocalCodeKit

/// The pieces of ModelsFeature that normally live inside a sheet or a popover,
/// exposed as plain views.
///
/// The three sections present them themselves, so nothing in the app has to use
/// this — it exists so they can be hosted anywhere they make sense (the agent's
/// own header could reuse the model picker) and so the render harness that
/// reviews this module can put each one in a window and look at it. All of them
/// read `ModelsStore` from the environment, exactly like the sections do.
public enum ModelsScreens {

    /// "Ustawienia ładowania" for one local model.
    public static func loadSettings(for model: LocalModelDTO) -> some View {
        LoadSettingsSheet(model: model)
    }

    /// The download queue that hangs off the toolbar indicator.
    public static var downloads: some View {
        DownloadsPopover()
    }

    /// The searchable local-model list behind `ModelLoaderBar`.
    public static func modelPicker(purpose: ModelPurpose,
                                   onOpenSettings: @escaping (LocalModelDTO) -> Void = { _ in }) -> some View {
        ModelPickerPopover(purpose: purpose, onOpenSettings: onOpenSettings)
    }

    /// The Hugging Face token editor a gated repository links to.
    public static var huggingFaceToken: some View {
        HuggingFaceTokenSheet()
    }
}

extension ModelsScreens {

    /// The whole store section ("Odkrywaj").
    public static var store: some View { ModelStoreView() }

    /// "Szczegóły / Zaawansowane" for one catalog entry.
    public static func catalogDetail(_ entry: CatalogEntry) -> some View {
        CatalogDetailSheet(entry: entry)
    }

    /// One store card on its own.
    public static func catalogCard(_ entry: CatalogEntry,
                                   onInstall: @escaping () -> Void = {}) -> some View {
        CatalogCard(entry: entry, onInstall: onInstall)
    }
}
