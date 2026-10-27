import SwiftUI
import LocalCodeKit

/// The conversation sidebar of the "Czat" section: primary actions, projects,
/// history.
///
/// It owns nothing but its own presentation state — everything it shows comes
/// from `ChatStore`, and every action goes back through it. The account menu
/// the web app had at the bottom is gone with accounts; its place is taken by a
/// shortcut into the app's settings.
struct SidebarView: View {
    /// Called after the selection changed. `nil` means "a brand new conversation".
    var onOpenConversation: ((String?) -> Void)?

    @Environment(ChatStore.self) private var store

    @State private var isSearching = false

    var body: some View {
        VStack(spacing: 0) {
            actions
            content
            Hairline()
            footer
        }
        .background(Theme.sidebar)
        .sheet(isPresented: $isSearching) {
            SearchView { hit in
                onOpenConversation?(hit.conversationId)
            }
            .environment(store)
        }
        .onChange(of: ShellBus.shared.tick) { _, _ in
            if case .search = ShellBus.shared.latest { isSearching = true }
        }
    }

    // MARK: - Primary actions

    private var actions: some View {
        VStack(spacing: 8) {
            PrimaryButton(title: "Nowa rozmowa", systemImage: "plus", height: 38) {
                startNewConversation()
            }

            searchButton
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
        .padding(.bottom, 8)
    }

    private var searchButton: some View {
        Button {
            isSearching = true
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 13, weight: .medium))
                Text("Szukaj w rozmowach…")
                    .font(.system(size: 13))
                    .lineLimit(1)
                Spacer(minLength: 0)
                Text("⌘K")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.fgSubtle)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(Theme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
            }
            .foregroundStyle(Theme.fgSubtle)
            .padding(.horizontal, 10)
            .frame(height: 36)
            .frame(maxWidth: .infinity)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .keyboardShortcut("k", modifiers: .command)
        .accessibilityLabel(Text("Szukaj w rozmowach"))
    }

    // MARK: - Scrolling content

    private var content: some View {
        List {
            FolderList(onOpenConversation: onOpenConversation)
            ConversationList(onOpenConversation: onOpenConversation)
        }
        .listStyle(.plain)
        .scrollContentBackground(.hidden)
        .background(Theme.sidebar)
        .environment(\.defaultMinListRowHeight, 26)
        .frame(maxHeight: .infinity)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 8) {
            Text(store.isLoaded
                 ? Format.plural(store.conversations.filter { !$0.archived }.count,
                                 "rozmowa", "rozmowy", "rozmów")
                 : "Wczytywanie…")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
            Spacer(minLength: 0)
            IconButton(systemName: "gearshape",
                       label: "Ustawienia czatu",
                       size: 26,
                       fontSize: 13) {
                ShellBus.shared.request(.show(.settings))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
    }

    // MARK: - Actions

    private func startNewConversation() {
        store.newConversation(folderId: nil)
        onOpenConversation?(nil)
    }
}

// MARK: - Sidebar row chrome

/// Small-caps label that introduces a group of rows, with an optional trailing
/// icon button (used by "Projekty").
struct SidebarSectionLabel: View {
    let title: String
    var systemImage: String?
    var actionIcon: String?
    var actionLabel: String?
    var action: (() -> Void)?

    var body: some View {
        HStack(spacing: 5) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Theme.fgSubtle)
            }
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .textCase(.uppercase)
                .kerning(0.6)
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
            Spacer(minLength: 0)
            if let actionIcon, let action {
                IconButton(systemName: actionIcon,
                           label: actionLabel,
                           tint: Theme.fgSubtle,
                           size: 22,
                           fontSize: 12,
                           action: action)
            }
        }
        .padding(.horizontal, 8)
        .padding(.top, 12)
        .padding(.bottom, 2)
        .accessibilityElement(children: .contain)
    }
}

extension View {
    /// Uniform insets for one row of the sidebar list.
    func sidebarListRow() -> some View {
        self
            .listRowInsets(EdgeInsets(top: 1, leading: 6, bottom: 1, trailing: 6))
            .listRowSeparator(.hidden)
            .listRowBackground(Color.clear)
    }

    /// Selected / hovered background for a tappable sidebar row.
    func sidebarRowChrome(isActive: Bool, isHovered: Bool) -> some View {
        self
            .background(isActive ? Theme.accentSoft : (isHovered ? Theme.surface2 : Color.clear))
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(alignment: .leading) {
                if isActive {
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(Theme.accent)
                        .frame(width: 3)
                        .padding(.vertical, 5)
                }
            }
    }
}
