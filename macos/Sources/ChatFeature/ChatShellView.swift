import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit

/// The whole "Czat" section: a header bar, the conversation sidebar and the
/// transcript.
///
/// The web app hung the title, the model picker and the per-conversation menu
/// off the window's own toolbar. Here the window toolbar belongs to the app
/// shell (it is shared by six sections), so the chat draws its own bar — which
/// also keeps the sidebar toggle, the token readout and the export menu
/// together with the thing they act on.
public struct ChatShellView: View {
    @Environment(ChatStore.self) private var store

    @State private var isSidebarVisible = true
    @State private var isConfirmingDelete = false
    @State private var isExporting = false
    @State private var exportDocument: ExportDocument?
    @State private var exportType: UTType = .plainText
    @State private var exportFilename = "rozmowa.md"
    @State private var bus = ShellBus.shared

    public init() {}

    private var conversation: Conversation? { store.activeConversation }

    // MARK: - Body

    public var body: some View {
        @Bindable var bindableStore = store

        return VStack(spacing: 0) {
            header
            Hairline()
            HStack(spacing: 0) {
                if isSidebarVisible {
                    SidebarView(onOpenConversation: { _ in })
                        .frame(width: 280)
                        .transition(.move(edge: .leading))
                    Hairline(vertical: true)
                }
                ChatView()
                    .banner($bindableStore.errorMessage)
                    .overlay(alignment: .top) { noticeOverlay }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Theme.bg)
        .alert("Usunąć rozmowę?", isPresented: $isConfirmingDelete) {
            Button("Anuluj", role: .cancel) {}
            Button("Usuń", role: .destructive) {
                if let id = conversation?.id {
                    Task { await store.deleteConversation(id) }
                }
            }
        } message: {
            Text("Rozmowa „\(conversation?.title ?? "")” wraz z wiadomościami i załącznikami zostanie trwale usunięta. Tej operacji nie można cofnąć.")
        }
        .fileExporter(isPresented: $isExporting,
                      document: exportDocument,
                      contentType: exportType,
                      defaultFilename: exportFilename) { result in
            if case let .failure(error) = result {
                store.errorMessage = error.localizedDescription
            }
        }
        .onChange(of: bus.tick) { _, _ in handle(bus.latest) }
    }

    /// The transient sentence: what "Skróć rozmowę" did, or what the server
    /// had to give up to make the last turn fit.
    @ViewBuilder
    private var noticeOverlay: some View {
        if let notice = store.notice {
            Banner(.info, message: notice, onDismiss: { store.dismissNotice() })
                .frame(maxWidth: 520)
                .padding(.top, 10)
                .padding(.horizontal, 16)
                .transition(.move(edge: .top).combined(with: .opacity))
                .animation(.easeOut(duration: 0.18), value: store.notice)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            IconButton(systemName: isSidebarVisible ? "sidebar.left" : "sidebar.leading",
                       label: "Pokaż lub ukryj listę rozmów",
                       size: 28,
                       fontSize: 14) {
                toggleSidebar()
            }

            VStack(alignment: .leading, spacing: 1) {
                Text(conversation?.title ?? "Nowa rozmowa")
                    .font(.system(size: 13.5, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                }
            }

            Spacer(minLength: 12)

            if store.isStreamingInBackground {
                HStack(spacing: 5) {
                    ProgressView().controlSize(.small).scaleEffect(0.65).frame(width: 12, height: 12)
                    Text("Odpowiedź w innej rozmowie…")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                }
            }

            EffortPicker(rawLevels: store.effortLevels,
                         selection: store.effectiveEffort,
                         showsLabel: false) { level in
                Task { await store.setEffort(level) }
            }

            if !store.contextUsage.isEmpty {
                ChatContextGauge(usage: store.contextUsage)
            }

            ModelMenu(models: store.models,
                      providers: store.providers,
                      currentProvider: store.currentProvider,
                      currentModel: store.currentModel,
                      style: .bar) { provider, model in
                Task { await store.selectModel(provider: provider, model: model) }
            }

            Menu {
                conversationMenuItems
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(.system(size: 14))
                    .foregroundStyle(Theme.fgMuted)
                    .frame(width: 28, height: 28)
                    .contentShape(Rectangle())
            }
            .menuIndicator(.hidden)
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Więcej działań")
        }
        .padding(.horizontal, 10)
        .frame(height: 42)
        .background(Theme.surface)
    }

    private var subtitle: String? {
        var parts: [String] = []
        if let name = conversation?.folderName, !name.isEmpty { parts.append(name) }
        if let tokens = TokenReadout.text(for: conversation) { parts.append(tokens) }
        return parts.isEmpty ? nil : parts.joined(separator: "  ·  ")
    }

    @ViewBuilder
    private var conversationMenuItems: some View {
        Button {
            store.newConversation(folderId: nil)
        } label: {
            Label("Nowa rozmowa", systemImage: "square.and.pencil")
        }
        Button {
            ShellBus.shared.request(.search)
        } label: {
            Label("Szukaj w rozmowach…", systemImage: "magnifyingglass")
        }

        Divider()

        Button {
            togglePin()
        } label: {
            Label(conversation?.pinned == true ? "Odepnij" : "Przypnij",
                  systemImage: conversation?.pinned == true ? "pin.slash" : "pin")
        }
        .disabled(conversation == nil)

        Button {
            runCompaction()
        } label: {
            Label("Skróć rozmowę", systemImage: "arrow.down.right.and.arrow.up.left")
        }
        .disabled(conversation == nil || store.isCompacting || store.isBusy)
        .help("Podsumuj starsze tury i zwolnij miejsce w kontekście modelu")

        Divider()

        Button {
            runExport(format: "md")
        } label: {
            Label("Eksportuj (Markdown)", systemImage: "square.and.arrow.down")
        }
        .disabled(conversation == nil)

        Button {
            runExport(format: "json")
        } label: {
            Label("Eksportuj (JSON)", systemImage: "curlybraces")
        }
        .disabled(conversation == nil)

        Divider()

        Button(role: .destructive) {
            isConfirmingDelete = true
        } label: {
            Label("Usuń rozmowę", systemImage: "trash")
        }
        .disabled(conversation == nil)
    }

    // MARK: - Actions

    private func handle(_ action: ShellBus.Action?) {
        guard let action else { return }
        switch action {
        case .newConversation:
            store.newConversation(folderId: nil)
        case .toggleSidebar:
            toggleSidebar()
        case .stopStreaming:
            store.stopStreaming()
        case .togglePin:
            togglePin()
        case .deleteConversation:
            if conversation != nil { isConfirmingDelete = true }
        case let .export(format):
            runExport(format: format)
        case .refresh:
            Task { await store.refreshAll() }
        case .search, .show:
            break                       // handled by the sidebar / the app shell
        }
    }

    private func toggleSidebar() {
        withAnimation(.easeOut(duration: 0.18)) {
            isSidebarVisible.toggle()
        }
    }

    private func togglePin() {
        guard let id = conversation?.id else { return }
        Task { await store.togglePin(id) }
    }

    /// "Skróć rozmowę". The answer is shown as a transient note, because a
    /// conversation that was already short reports that nothing changed and the
    /// user is otherwise left wondering whether the button worked.
    private func runCompaction() {
        guard conversation != nil else { return }
        Task {
            guard let result = await store.compactConversation() else { return }
            store.announce(result.note)
        }
    }

    private func runExport(format: String) {
        guard let conversation else { return }
        let id = conversation.id
        let title = conversation.title
        Task {
            guard let data = await store.exportConversation(id, format: format) else { return }
            exportDocument = ExportDocument(data: data)
            exportType = format == "json" ? .json : ExportDocument.markdown
            exportFilename = ExportDocument.filename(for: title, format: format)
            isExporting = true
        }
    }
}
