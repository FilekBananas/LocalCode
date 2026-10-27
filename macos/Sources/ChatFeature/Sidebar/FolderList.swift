import SwiftUI
import LocalCodeKit

/// The eight project colours, kept in the same order as the web app so a
/// project looks identical wherever it is opened.
enum ProjectColorPalette {
    static let hexes = [
        "#c96442", "#4078f2", "#2f855a", "#805ad5",
        "#d69e2e", "#e53e3e", "#319795", "#718096",
    ]

    static var `default`: String { hexes[0] }
}

/// The "Projekty" section of the sidebar: one row per project, expandable to
/// the conversations it holds.
struct FolderList: View {
    var onOpenConversation: ((String?) -> Void)?

    @Environment(ChatStore.self) private var store

    @State private var expanded: Set<String> = []
    @State private var editedFolder: Folder?
    @State private var isCreating = false
    @State private var deletionTarget: Folder?

    private var activeFolderId: String? { store.activeConversation?.folderId }

    // MARK: - Body

    var body: some View {
        SidebarSectionLabel(title: "Projekty",
                            actionIcon: "plus",
                            actionLabel: "Nowy projekt") { isCreating = true }
            .sidebarListRow()
            .sheet(isPresented: $isCreating) {
                FolderEditor().environment(store)
            }
            .sheet(item: $editedFolder) { folder in
                FolderEditor(folder: folder).environment(store)
            }
            .confirmationDialog(deletionTitle,
                                isPresented: Binding(get: { deletionTarget != nil },
                                                     set: { if !$0 { deletionTarget = nil } }),
                                titleVisibility: .visible) {
                Button("Usuń projekt, zachowaj rozmowy", role: .destructive) {
                    delete(deleteChats: false)
                }
                Button("Usuń projekt razem z rozmowami", role: .destructive) {
                    delete(deleteChats: true)
                }
                Button("Anuluj", role: .cancel) { deletionTarget = nil }
            } message: {
                Text("Instrukcje, pamięć i pliki projektu zostaną trwale usunięte. Tej operacji nie można cofnąć.")
            }
            // Reveal the project holding the open conversation, once — it stays
            // collapsible afterwards.
            .onAppear { revealActiveFolder() }
            .onChange(of: activeFolderId) { _, _ in revealActiveFolder() }

        if store.folders.isEmpty {
            emptyRow
        } else {
            ForEach(store.folders) { folder in
                FolderRowView(folder: folder,
                              isExpanded: expanded.contains(folder.id),
                              isActive: activeFolderId == folder.id,
                              onToggle: { toggle(folder.id) },
                              onEdit: { editedFolder = folder },
                              onDelete: { deletionTarget = folder },
                              onNewConversation: { startConversation(in: folder.id) })
                    .sidebarListRow()

                if expanded.contains(folder.id) {
                    if folder.memoryMode == .isolated {
                        isolatedHint
                    }
                    let items = conversations(in: folder.id)
                    if items.isEmpty {
                        emptyFolderRow(folder)
                    } else {
                        ForEach(items) { conversation in
                            ConversationRowView(conversation: conversation,
                                                nested: true,
                                                onOpenConversation: onOpenConversation)
                                .sidebarListRow()
                        }
                    }
                }
            }
        }
    }

    // MARK: - Rows

    private var emptyRow: some View {
        Button {
            isCreating = true
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "folder.badge.plus")
                    .font(.system(size: 14))
                Text("Utwórz projekt, aby nadać rozmowom wspólne instrukcje i wiedzę.")
                    .font(.system(size: 12))
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .foregroundStyle(Theme.fgSubtle)
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .foregroundStyle(Theme.border)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .padding(.top, 2)
        .sidebarListRow()
    }

    private var isolatedHint: some View {
        HStack(spacing: 6) {
            Image(systemName: "lock.fill")
                .font(.system(size: 9))
            Text("Pamięć izolowana — bez kontekstu globalnego")
                .font(.system(size: 11))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .foregroundStyle(Theme.fgSubtle)
        .padding(.leading, 26)
        .padding(.trailing, 8)
        .padding(.vertical, 3)
        .sidebarListRow()
    }

    private func emptyFolderRow(_ folder: Folder) -> some View {
        Button {
            startConversation(in: folder.id)
        } label: {
            Text("Brak rozmów — zacznij nową")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.leading, 32)
                .padding(.trailing, 8)
                .padding(.vertical, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .sidebarListRow()
    }

    // MARK: - Data

    private func conversations(in folderId: String) -> [Conversation] {
        store.conversations
            .filter { $0.folderId == folderId && !$0.archived }
            .sorted { left, right in
                if left.pinned != right.pinned { return left.pinned }
                return left.lastMessageAt > right.lastMessageAt
            }
    }

    private var deletionTitle: String {
        guard let deletionTarget else { return "Usunąć projekt?" }
        return "Usunąć projekt „\(deletionTarget.name)”?"
    }

    // MARK: - Actions

    private func revealActiveFolder() {
        guard let activeFolderId else { return }
        expanded.insert(activeFolderId)
    }

    private func toggle(_ id: String) {
        if expanded.contains(id) {
            expanded.remove(id)
        } else {
            expanded.insert(id)
        }
    }

    private func startConversation(in folderId: String) {
        expanded.insert(folderId)
        store.newConversation(folderId: folderId)
        onOpenConversation?(nil)
    }

    private func delete(deleteChats: Bool) {
        guard let folder = deletionTarget else { return }
        deletionTarget = nil
        Task { await store.deleteFolder(folder.id, deleteChats: deleteChats) }
    }
}

// MARK: - One project

/// Header row of a project: colour dot, name, isolation lock and the count of
/// conversations inside it.
struct FolderRowView: View {
    let folder: Folder
    let isExpanded: Bool
    let isActive: Bool
    let onToggle: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void
    let onNewConversation: () -> Void

    @Environment(ChatStore.self) private var store

    @State private var isHovered = false

    private var isolated: Bool { folder.memoryMode == .isolated }

    private var count: Int {
        folder.conversationCount ?? store.conversations.filter {
            $0.folderId == folder.id && !$0.archived
        }.count
    }

    private static let isolatedHint = "Pamięć izolowana — bez kontekstu globalnego"

    var body: some View {
        Button(action: onToggle) {
            HStack(spacing: 7) {
                Image(systemName: "chevron.right")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.fgSubtle)
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    .animation(.easeOut(duration: 0.15), value: isExpanded)
                Circle()
                    .fill(Format.hexColor(folder.color) ?? Theme.accent)
                    .frame(width: 9, height: 9)
                Text(folder.name)
                    .font(.system(size: 13, weight: isActive ? .medium : .regular))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                if isolated {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 10))
                        .foregroundStyle(Theme.fgSubtle)
                        .help(Self.isolatedHint)
                        .accessibilityLabel(Text(Self.isolatedHint))
                }
                Spacer(minLength: 0)
                Text("\(count)")
                    .font(.system(size: 11, weight: .medium))
                    .monospacedDigit()
                    .foregroundStyle(Theme.fgSubtle)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .sidebarRowChrome(isActive: isActive, isHovered: isHovered)
        .onHover { isHovered = $0 }
        .help(folder.description ?? folder.name)
        .accessibilityHint(Text(isExpanded ? "Zwiń projekt" : "Rozwiń projekt"))
        .contextMenu {
            Button(action: onEdit) {
                Label("Edytuj projekt", systemImage: "pencil")
            }
            Button(action: onNewConversation) {
                Label("Nowa rozmowa w projekcie", systemImage: "plus")
            }
            Divider()
            Button(role: .destructive, action: onDelete) {
                Label("Usuń projekt", systemImage: "trash")
            }
        }
    }
}
