import SwiftUI
import UniformTypeIdentifiers
import AppKit
import LocalCodeKit

/// "Moje modele" — everything on this Mac that LocalCode can run.
///
/// The list deliberately mixes the three places models come from (LocalCode's
/// own directory, an LM Studio installation and the Hugging Face cache) and
/// labels each row with its origin, because that is what decides whether the
/// app is allowed to delete the files.
public struct MyModelsView: View {
    @Environment(ModelsStore.self) private var store

    @State private var settingsTarget: LocalModelDTO?
    @State private var confirmation: ConfirmRequest?

    public init() {}

    public var body: some View {
        @Bindable var store = store

        VStack(spacing: 0) {
            ScreenHeader(title: "Moje modele", subtitle: subtitle) {
                HStack(spacing: 10) {
                    StyledTextField(placeholder: "Filtruj…",
                                    text: $store.libraryFilter,
                                    systemImage: "line.3.horizontal.decrease")
                        .frame(width: 220)
                    SecondaryButton(title: "Wgraj z dysku",
                                    systemImage: "square.and.arrow.down",
                                    isLoading: store.isImporting,
                                    height: 32,
                                    action: chooseModelToImport)
                        .help("Dodaj model, który już masz: plik .gguf albo katalog MLX")
                    SecondaryButton(title: "Przeszukaj ponownie",
                                    systemImage: "arrow.clockwise",
                                    isLoading: store.isRescanning,
                                    height: 32) {
                        Task { await store.rescanLibrary() }
                    }
                }
            }

            StoreMessages()

            directoryBar

            if store.library.isEmpty {
                EmptyState(systemImage: "shippingbox",
                           title: "Nie masz jeszcze żadnych modeli",
                           message: "Pobierz pierwszy model w sekcji „Odkrywaj”. Modele z LM Studio i cache Hugging Face są wykrywane automatycznie.",
                           actionTitle: "Przejdź do Odkrywaj") {
                    ShellBus.shared.request(.show(.discover))
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.filteredLibrary.isEmpty {
                EmptyState(systemImage: "magnifyingglass",
                           title: "Brak modeli pasujących do filtra",
                           message: "Wyczyść pole filtrowania, aby zobaczyć wszystkie \(store.library.count).")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 8) {
                        ForEach(store.filteredLibrary) { model in
                            LocalModelRow(model: model,
                                          onSettings: { settingsTarget = model },
                                          onDelete: { requestDelete(model) })
                        }
                    }
                    .padding(20)
                }
            }
        }
        .background(Theme.bg)
        // Sheets and popovers are separate hosting contexts; the store is
        // handed over explicitly so they never depend on how the presenting
        // view happened to receive it.
        .sheet(item: $settingsTarget) { model in
            LoadSettingsSheet(model: model).environment(store)
        }
        .confirmDialog($confirmation)
    }

    private var subtitle: String {
        let total = store.library.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let count = Format.plural(store.library.count, "model", "modele", "modeli")
        let loaded = store.runtime.liveInstances.count
        var text = "\(count) · \(Format.bytes(total)) na dysku"
        if loaded > 0 {
            text += " · \(Format.plural(loaded, "załadowana instancja", "załadowane instancje", "załadowanych instancji"))"
        }
        return text
    }

    // MARK: - Models directory

    private var directoryBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "folder")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgSubtle)
            Text("Katalog modeli")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.fgMuted)
            Text(store.settings.modelsDir.isEmpty ? "—" : store.settings.modelsDir)
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
                .truncationMode(.head)
                .textSelection(.enabled)
            Spacer(minLength: 8)
            Button("Pokaż w Finderze") {
                guard !store.settings.modelsDir.isEmpty else { return }
                NSWorkspace.shared.activateFileViewerSelecting(
                    [URL(fileURLWithPath: store.settings.modelsDir)])
            }
            .buttonStyle(.link)
            .font(.system(size: 12))
            SecondaryButton(title: "Zmień…", height: 28, action: chooseDirectory)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(Theme.surface2)
        .overlay(alignment: .bottom) { Hairline() }
    }

    /// "Wgraj z dysku": a model the person already downloaded elsewhere.
    ///
    /// Both kinds are selectable in one panel because people do not think in
    /// terms of "a file for GGUF, a folder for MLX" — the server works out
    /// which it is, and says so plainly when it is neither.
    private func chooseModelToImport() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Dodaj"
        panel.message = "Wskaż plik .gguf albo katalog z modelem MLX."
        panel.allowedContentTypes = [.data, .folder]
        guard panel.runModal() == .OK, let url = panel.url else { return }

        let copy = NSAlert()
        copy.messageText = "Skopiować czy przenieść?"
        copy.informativeText = """
        Kopia zostawia plik tam, gdzie jest — model będzie zajmował miejsce dwa razy.
        Przeniesienie oszczędza miejsce, ale usuwa oryginał z jego obecnej lokalizacji.
        """
        copy.addButton(withTitle: "Skopiuj")
        copy.addButton(withTitle: "Przenieś")
        copy.addButton(withTitle: "Anuluj")
        let answer = copy.runModal()
        guard answer != .alertThirdButtonReturn else { return }
        let move = answer == .alertSecondButtonReturn
        Task { await store.importModel(at: url.path, move: move) }
    }

    private func chooseDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Wybierz"
        panel.message = "Wskaż katalog, w którym LocalCode ma trzymać pobrane modele."
        if !store.settings.modelsDir.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: store.settings.modelsDir)
        }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await store.setModelsDirectory(url.path) }
    }

    // MARK: - Delete

    private func requestDelete(_ model: LocalModelDTO) {
        confirmation = ConfirmRequest(
            title: "Usunąć „\(model.displayName)”?",
            message: "Pliki modelu (\(Format.bytes(model.sizeBytes))) zostaną trwale skasowane z dysku. Tej operacji nie można cofnąć.",
            confirmTitle: "Usuń",
            confirm: { Task { await store.deleteModel(key: model.key) } })
    }
}

// MARK: - One local model

/// One row of "Moje modele": identity, technical facts, and every action that
/// applies to a model that is already on disk.
struct LocalModelRow: View {
    let model: LocalModelDTO
    let onSettings: () -> Void
    let onDelete: () -> Void

    @Environment(ModelsStore.self) private var store
    @State private var isHovered = false

    private var instances: [RuntimeInstance] { store.instances(of: model.key) }
    private var isLoaded: Bool { !instances.isEmpty }
    private var isBusy: Bool { store.isLoading(model.key) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(model.displayName)
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundStyle(Theme.fg)
                            .lineLimit(1)
                        if model.remote {
                            // A borrowed model: it runs in a data centre, costs
                            // money per request and takes none of this Mac's RAM.
                            HStack(spacing: 4) {
                                Image(systemName: "cloud")
                                    .font(.system(size: 10, weight: .semibold))
                                Text(model.providerLabel ?? "zdalnie")
                                    .font(.system(size: 11, weight: .medium))
                            }
                            .foregroundStyle(Theme.accent)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 2)
                            .background(Theme.accent.opacity(0.14))
                            .clipShape(Capsule())
                            .help("Model działa u dostawcy, nie na tym Macu")
                        } else {
                            FormatBadge(format: model.format)
                        }
                        if let quant = model.quant {
                            MetaBadge(text: quant)
                        }
                        CapabilityIcons(capabilities: model.capabilities, hasVision: model.hasVision)
                    }

                    FlowLayout(spacing: 14, lineSpacing: 5) {
                        if let params = ModelFormat.params(model.paramsB) {
                            fact("cube", params, help: "Liczba parametrów")
                        }
                        fact("externaldrive", Format.bytes(model.sizeBytes), help: "Rozmiar na dysku")
                        if let arch = model.arch, !arch.isEmpty {
                            fact("cpu", arch, help: "Architektura")
                        }
                        if let context = ModelFormat.context(model.contextLength) {
                            fact("text.alignleft", context, help: "Natywny kontekst")
                        }
                        fact("shippingbox", model.sourceLabel, help: "Skąd pochodzi model")
                        if let defaults = model.loadDefaults, !defaults.isEmpty {
                            fact("slider.horizontal.3", defaultsSummary(defaults),
                                 help: "Zapisane ustawienia ładowania")
                        }
                    }

                    roleTags
                }

                Spacer(minLength: 0)

                HStack(spacing: 8) {
                    stateChip
                    actions
                }
                .fixedSize()
            }

            if isLoaded {
                loadedInstances
            }
        }
        .padding(12)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(isHovered ? Theme.borderStrong : Theme.border, lineWidth: 1)
        )
        .onHover { isHovered = $0 }
    }

    // MARK: Pieces

    @ViewBuilder
    private var stateChip: some View {
        if let instance = instances.first {
            HStack(spacing: 5) {
                Circle()
                    .fill(color(for: instance.state))
                    .frame(width: 7, height: 7)
                Text(instance.state.label)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(color(for: instance.state))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color(for: instance.state).opacity(0.12))
            .clipShape(Capsule())
        } else if isBusy {
            ProgressView().controlSize(.small)
        }
    }

    private func color(for state: RuntimeInstance.State) -> Color {
        switch state {
        case .ready: Theme.success
        case .busy: Theme.accent
        case .loading, .stopping: Theme.warning
        case .error: Theme.danger
        }
    }

    private var loadedInstances: some View {
        VStack(spacing: 4) {
            ForEach(instances) { instance in
                HStack(spacing: 10) {
                    Image(systemName: "bolt.horizontal.circle")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                    Text("kontekst \(Format.tokens(instance.nCtx))")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgMuted)
                    Text("sloty \(instance.busySlots)/\(instance.nParallel)")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgMuted)
                    if instance.memoryBytes > 0 {
                        Text(Format.bytes(instance.memoryBytes))
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgMuted)
                    }
                    if let tps = ModelFormat.tps(instance.genTps) {
                        Text("\(tps) tok/s")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgMuted)
                    }
                    if let error = instance.error {
                        Text(error)
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.danger)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Theme.surface2)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
    }

    private var hasRoles: Bool {
        store.settings.chat.defaultModel == model.key
            || store.settings.agent.defaultModel == model.key
            || store.settings.chat.embeddingModel == model.key
    }

    @ViewBuilder
    private var roleTags: some View {
        if hasRoles {
            HStack(spacing: 6) {
                if store.settings.chat.defaultModel == model.key {
                    Chip(title: "Domyślny w czacie", systemImage: "bubble.left", isSelected: true)
                }
                if store.settings.agent.defaultModel == model.key {
                    Chip(title: "Domyślny w agencie",
                         systemImage: "chevron.left.forwardslash.chevron.right",
                         isSelected: true)
                }
                if store.settings.chat.embeddingModel == model.key {
                    Chip(title: "Embeddingi",
                         systemImage: "point.3.connected.trianglepath.dotted",
                         isSelected: true)
                }
            }
            .padding(.top, 2)
        }
    }

    private var actions: some View {
        HStack(spacing: 8) {
            FavoriteStarButton(key: model.key, size: 28, fontSize: 13)

            if isLoaded {
                SecondaryButton(title: "Wyładuj", systemImage: "eject", height: 30) {
                    Task { await store.unload(key: model.key) }
                }
            } else {
                PrimaryButton(title: "Załaduj",
                              systemImage: "play.fill",
                              isLoading: isBusy,
                              fullWidth: false,
                              height: 30) {
                    Task { await store.load(key: model.key, config: model.loadDefaults) }
                }
            }

            Menu {
                Button("Ustawienia ładowania…", action: onSettings)
                Button("Pokaż w Finderze") { store.revealInFinder(model) }
                Button(FavoriteModels.shared.actionTitle(for: model.key)) {
                    FavoriteModels.shared.toggle(model.key)
                }
                Divider()
                if model.isEmbedding {
                    Button("Ustaw jako model embeddingów") {
                        Task { await store.setEmbeddingModel(model.key) }
                    }
                    .disabled(store.settings.chat.embeddingModel == model.key)
                } else {
                    Button("Ustaw jako domyślny model czatu") {
                        Task { await store.setDefaultModel(model.key, for: .chat) }
                    }
                    .disabled(store.settings.chat.defaultModel == model.key)
                    Button("Ustaw jako domyślny model agenta") {
                        Task { await store.setDefaultModel(model.key, for: .agent) }
                    }
                    .disabled(store.settings.agent.defaultModel == model.key)
                }
                if isLoaded {
                    Divider()
                    Button("Załaduj drugą instancję") {
                        Task { await store.load(key: model.key,
                                                config: model.loadDefaults,
                                                newInstance: true) }
                    }
                }
                Divider()
                Button("Usuń z dysku…", role: .destructive, action: onDelete)
                    .disabled(!model.isDeletable)
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fgMuted)
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .menuIndicator(.hidden)
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help(model.isDeletable
                  ? "Więcej działań"
                  : "Więcej działań — modeli spoza katalogu LocalCode nie można usunąć z aplikacji")
        }
    }

    private func fact(_ symbol: String, _ text: String, help: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: symbol).font(.system(size: 10))
            Text(text).font(.system(size: 11.5))
        }
        .foregroundStyle(Theme.fgMuted)
        .help(help)
    }

    private func defaultsSummary(_ config: LoadConfigDTO) -> String {
        var parts: [String] = []
        if let ctx = config.nCtx { parts.append(Format.tokens(ctx)) }
        if let parallel = config.nParallel, parallel > 1 { parts.append("\(parallel) sloty") }
        if let kv = config.kvCacheType, kv != "f16" { parts.append("KV \(kv)") }
        return parts.isEmpty ? "własne ustawienia" : parts.joined(separator: " · ")
    }
}
