import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit

/// History of every conversation that does not belong to a project.
///
/// Rendered as rows of the sidebar's `List`. Pinned conversations get their own
/// section on top; the rest are grouped by `Format.bucket` in
/// `Format.bucketOrder`.
struct ConversationList: View {
    var onOpenConversation: ((String?) -> Void)?

    @Environment(ChatStore.self) private var store

    /// Rows are cheap, but a very long history still gets a "show older" cut.
    @State private var limit = ConversationList.pageSize

    private static let pageSize = 300

    private struct Bucket: Identifiable {
        let id: String
        let items: [Conversation]
    }

    // MARK: - Data

    private var root: [Conversation] {
        store.conversations.filter { $0.folderId == nil && !$0.archived }
    }

    private var pinned: [Conversation] {
        root.filter(\.pinned).sorted { $0.lastMessageAt > $1.lastMessageAt }
    }

    private var unpinned: [Conversation] {
        root.filter { !$0.pinned }.sorted { $0.lastMessageAt > $1.lastMessageAt }
    }

    private var visible: [Conversation] {
        Array(unpinned.prefix(limit))
    }

    private var hiddenCount: Int {
        max(0, unpinned.count - visible.count)
    }

    private var buckets: [Bucket] {
        let grouped = Dictionary(grouping: visible) { Format.bucket($0.lastMessageAt) }
        return Format.bucketOrder.compactMap { key in
            guard let items = grouped[key], !items.isEmpty else { return nil }
            return Bucket(id: key, items: items)
        }
    }

    // MARK: - Body

    var body: some View {
        if root.isEmpty {
            emptyRow
        } else {
            if !pinned.isEmpty {
                SidebarSectionLabel(title: "Przypięte", systemImage: "pin.fill")
                    .sidebarListRow()
                ForEach(pinned) { conversation in
                    ConversationRowView(conversation: conversation,
                                        onOpenConversation: onOpenConversation)
                        .sidebarListRow()
                }
            }

            ForEach(buckets) { bucket in
                SidebarSectionLabel(title: bucket.id)
                    .sidebarListRow()
                ForEach(bucket.items) { conversation in
                    ConversationRowView(conversation: conversation,
                                        onOpenConversation: onOpenConversation)
                        .sidebarListRow()
                }
            }

            if hiddenCount > 0 {
                showMoreRow
            }
        }
    }

    private var emptyRow: some View {
        VStack(spacing: 4) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 18, weight: .light))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.bottom, 2)
            Text("Brak rozmów.")
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
            Text("Zacznij nową rozmowę — pojawi się na tej liście.")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 26)
        .padding(.horizontal, 10)
        .sidebarListRow()
    }

    private var showMoreRow: some View {
        Button {
            limit += ConversationList.pageSize
        } label: {
            Text("Pokaż starsze (\(hiddenCount))")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .padding(.top, 4)
        .sidebarListRow()
    }
}

// MARK: - One conversation

/// A single row of the sidebar. Used both at the top level and nested inside an
/// expanded project, which is what `nested` indents for.
struct ConversationRowView: View {
    let conversation: Conversation
    var nested: Bool = false
    var onOpenConversation: ((String?) -> Void)?

    @Environment(ChatStore.self) private var store

    @State private var isHovered = false
    @State private var isRenaming = false
    @State private var draftTitle = ""
    @State private var isMoving = false
    @State private var isConfirmingDelete = false
    @State private var isExporting = false
    @State private var exportDocument: ExportDocument?
    @State private var exportType: UTType = .plainText
    @State private var exportFilename = "rozmowa.md"

    private var isActive: Bool { store.activeConversation?.id == conversation.id }

    private var preview: String {
        conversation.preview?
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    // MARK: Body

    var body: some View {
        Button(action: open) {
            HStack(spacing: 6) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(conversation.title)
                        .font(.system(size: 13, weight: isActive ? .medium : .regular))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    if !preview.isEmpty {
                        Text(preview)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgSubtle)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
                if conversation.pinned {
                    Image(systemName: "pin.fill")
                        .font(.system(size: 9))
                        .foregroundStyle(Theme.fgSubtle)
                }
            }
            .padding(.leading, nested ? 26 : 10)
            .padding(.trailing, 8)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .sidebarRowChrome(isActive: isActive, isHovered: isHovered)
        .overlay(alignment: .leading) {
            // Guide line tying a nested row to the project above it.
            if nested {
                Rectangle()
                    .fill(Theme.border)
                    .frame(width: 1)
                    .padding(.leading, 13)
                    .padding(.vertical, 1)
            }
        }
        .onHover { isHovered = $0 }
        .help(conversation.title)
        .contextMenu { menuItems }
        .alert("Zmień nazwę rozmowy", isPresented: $isRenaming) {
            TextField("Nazwa rozmowy", text: $draftTitle)
            Button("Anuluj", role: .cancel) {}
            Button("Zapisz", action: commitRename)
        } message: {
            Text("Własna nazwa nie zostanie nadpisana przez automatyczny tytuł.")
        }
        .alert("Usunąć rozmowę?", isPresented: $isConfirmingDelete) {
            Button("Anuluj", role: .cancel) {}
            Button("Usuń", role: .destructive) {
                Task { await store.deleteConversation(conversation.id) }
            }
        } message: {
            Text("Rozmowa „\(conversation.title)” wraz z wiadomościami i załącznikami zostanie trwale usunięta. Tej operacji nie można cofnąć.")
        }
        .sheet(isPresented: $isMoving) {
            MoveToFolderSheet(conversation: conversation)
                .environment(store)
        }
        .fileExporter(isPresented: $isExporting,
                      document: exportDocument,
                      contentType: exportType,
                      defaultFilename: exportFilename) { result in
            if case let .failure(error) = result {
                store.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: Menu

    @ViewBuilder
    private var menuItems: some View {
        Button(action: startRename) {
            Label("Zmień nazwę", systemImage: "pencil")
        }
        Button {
            Task { await store.togglePin(conversation.id) }
        } label: {
            Label(conversation.pinned ? "Odepnij" : "Przypnij",
                  systemImage: conversation.pinned ? "pin.slash" : "pin")
        }
        Button {
            isMoving = true
        } label: {
            Label("Przenieś do projektu", systemImage: "folder")
        }

        Divider()

        Button {
            export(format: "md")
        } label: {
            Label("Eksportuj (Markdown)", systemImage: "square.and.arrow.down")
        }
        Button {
            export(format: "json")
        } label: {
            Label("Eksportuj (JSON)", systemImage: "curlybraces")
        }
        Button(action: archive) {
            Label("Archiwizuj", systemImage: "archivebox")
        }

        Divider()

        Button(role: .destructive) {
            isConfirmingDelete = true
        } label: {
            Label("Usuń", systemImage: "trash")
        }
    }

    // MARK: Actions

    private func open() {
        let id = conversation.id
        Task { await store.openConversation(id) }
        onOpenConversation?(id)
    }

    private func startRename() {
        draftTitle = conversation.title
        isRenaming = true
    }

    private func commitRename() {
        let next = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !next.isEmpty, next != conversation.title else { return }
        Task { await store.updateConversation(conversation.id, ConversationInput(title: next)) }
    }

    private func archive() {
        Task { await store.updateConversation(conversation.id, ConversationInput(archived: true)) }
    }

    private func export(format: String) {
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

// MARK: - Move to project

/// Small sheet that reassigns a conversation to a project (or takes it out).
struct MoveToFolderSheet: View {
    let conversation: Conversation

    @Environment(ChatStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var isSaving = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Przenieś do projektu")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Rozmowa przejmie instrukcje, pamięć i pliki wybranego projektu.")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                IconButton(systemName: "xmark", label: "Zamknij") { dismiss() }
            }
            .padding(16)

            Hairline()

            ScrollView {
                VStack(spacing: 4) {
                    row(id: nil,
                        name: "Bez projektu (kontekst globalny)",
                        color: nil,
                        isolated: false)
                    ForEach(store.folders) { folder in
                        row(id: folder.id,
                            name: folder.name,
                            color: folder.color,
                            isolated: folder.memoryMode == .isolated)
                    }
                }
                .padding(12)
            }
        }
        .background(Theme.bg)
        .frame(minWidth: 380, idealWidth: 420, minHeight: 320, idealHeight: 420)
        .disabled(isSaving)
    }

    private func row(id: String?, name: String, color: String?, isolated: Bool) -> some View {
        let isCurrent = conversation.folderId == id
        return Button {
            move(to: id)
        } label: {
            HStack(spacing: 10) {
                Circle()
                    .fill(Format.hexColor(color) ?? Theme.fgSubtle)
                    .frame(width: 10, height: 10)
                Text(name)
                    .font(.system(size: 13.5))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                if isolated {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.fgSubtle)
                }
                Spacer(minLength: 0)
                if isCurrent {
                    Image(systemName: "checkmark")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Theme.accent)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isCurrent ? Theme.accentSoft : Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(isCurrent ? Theme.accent : Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
    }

    private func move(to folderId: String?) {
        guard folderId != conversation.folderId else {
            dismiss()
            return
        }
        isSaving = true
        let id = conversation.id
        Task {
            if let folderId {
                await store.updateConversation(id, ConversationInput(folderId: folderId))
            } else {
                await store.detachConversation(id)
            }
            isSaving = false
            dismiss()
        }
    }
}
