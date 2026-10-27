import SwiftUI
import LocalCodeKit

/// Left column of the "Kod" section: the project folders the agent was pointed
/// at, and the sessions of the selected one.
struct ProjectSidebar: View {
    @Environment(AgentStore.self) private var store

    @State private var sessionPendingDeletion: AgentSessionDTO?

    var body: some View {
        VStack(spacing: 0) {
            actions
            Hairline()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    projectsSection
                    sessionsSection
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 8)
            }
        }
        .frame(width: 268)
        .background(Theme.sidebar)
        .overlay(alignment: .trailing) { Hairline(vertical: true) }
        .alert("Usunąć sesję?", isPresented: Binding(
            get: { sessionPendingDeletion != nil },
            set: { if !$0 { sessionPendingDeletion = nil } }
        )) {
            Button("Anuluj", role: .cancel) { sessionPendingDeletion = nil }
            Button("Usuń", role: .destructive) {
                if let target = sessionPendingDeletion {
                    Task { await store.deleteSession(target) }
                }
                sessionPendingDeletion = nil
            }
        } message: {
            Text("Sesja „\(sessionPendingDeletion?.displayTitle ?? "")” wraz z całym zapisem rozmowy zostanie trwale usunięta.")
        }
    }

    // MARK: - Actions

    private var actions: some View {
        VStack(spacing: 8) {
            PrimaryButton(title: "Nowa sesja",
                          systemImage: "plus",
                          isEnabled: store.hasProject,
                          height: 38) {
                Task { await store.newSession() }
            }
            SecondaryButton(title: "Otwórz folder…",
                            systemImage: "folder.badge.plus",
                            fullWidth: true,
                            height: 34) {
                store.openProjectFolder()
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    // MARK: - Projects

    @ViewBuilder
    private var projectsSection: some View {
        CodeSidebarSectionLabel(title: "Projekty")
        if store.projects.isEmpty {
            sidebarHint("Otwórz folder z kodem — agent będzie w nim czytał i zmieniał pliki.")
        } else {
            ForEach(store.projects) { project in
                ProjectRow(project: project,
                           isActive: store.selectedProject?.path == project.path,
                           onOpen: { Task { await store.select(project: project) } },
                           onRemove: { store.remove(project: project) })
            }
        }
    }

    // MARK: - Sessions

    @ViewBuilder
    private var sessionsSection: some View {
        CodeSidebarSectionLabel(title: "Sesje", systemImage: "clock.arrow.circlepath")
            .padding(.top, 10)
        if !store.hasProject {
            sidebarHint("Wybierz projekt, aby zobaczyć jego sesje.")
        } else if store.isLoadingSessions && store.sessions.isEmpty {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small).scaleEffect(0.8)
                Text("Wczytuję sesje…")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
        } else if store.sessions.isEmpty {
            sidebarHint("Brak sesji w tym folderze. Zacznij nową sesję.")
        } else {
            ForEach(store.sessions) { session in
                SessionRow(session: session,
                           isActive: store.session?.id == session.id,
                           modelLabel: modelLabel(for: session),
                           onOpen: { Task { await store.open(session: session) } },
                           onDelete: { sessionPendingDeletion = session })
            }
        }
    }

    private func modelLabel(for session: AgentSessionDTO) -> String {
        guard let key = session.modelKey, !key.isEmpty else { return "bez modelu" }
        if let model = store.models.first(where: { $0.key == key }) {
            return RemoteModelKey.name(model.displayName, key: key)
        }
        return AgentPaths.name(of: key)
    }

    private func sidebarHint(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11.5))
            .foregroundStyle(Theme.fgSubtle)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
    }
}

// MARK: - Rows

private struct ProjectRow: View {
    let project: ProjectRef
    let isActive: Bool
    let onOpen: () -> Void
    let onRemove: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: onOpen) {
            HStack(spacing: 8) {
                Image(systemName: project.exists ? "folder.fill" : "questionmark.folder")
                    .font(.system(size: 12))
                    .foregroundStyle(isActive ? Theme.accent : Theme.fgSubtle)
                VStack(alignment: .leading, spacing: 1) {
                    Text(project.name)
                        .font(.system(size: 13, weight: isActive ? .semibold : .regular))
                        .foregroundStyle(project.exists ? Theme.fg : Theme.fgSubtle)
                        .lineLimit(1)
                    Text(project.parent)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                        .truncationMode(.head)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .codeSidebarRowChrome(isActive: isActive, isHovered: isHovered)
        .onHover { isHovered = $0 }
        .help(project.shortPath)
        .contextMenu {
            Button {
                NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: project.path)
            } label: {
                Label("Pokaż w Finderze", systemImage: "folder")
            }
            Button(role: .destructive, action: onRemove) {
                Label("Usuń z listy", systemImage: "xmark.circle")
            }
        }
    }
}

private struct SessionRow: View {
    let session: AgentSessionDTO
    let isActive: Bool
    let modelLabel: String
    let onOpen: () -> Void
    let onDelete: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: onOpen) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(session.displayTitle)
                        .font(.system(size: 13, weight: isActive ? .medium : .regular))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    HStack(spacing: 5) {
                        Text(Format.relative(session.updatedAt))
                        Text("·")
                        Text(modelLabel).lineLimit(1).truncationMode(.middle)
                    }
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                }
                Spacer(minLength: 0)
                if session.busy {
                    ProgressView().controlSize(.small).scaleEffect(0.6).frame(width: 12, height: 12)
                } else if isHovered {
                    IconButton(systemName: "trash", label: "Usuń sesję", size: 22, fontSize: 11) {
                        onDelete()
                    }
                }
            }
            .padding(.leading, 10)
            .padding(.trailing, 6)
            .padding(.vertical, 6)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .codeSidebarRowChrome(isActive: isActive, isHovered: isHovered)
        .onHover { isHovered = $0 }
        .help(session.displayTitle)
        .contextMenu {
            Button(role: .destructive, action: onDelete) {
                Label("Usuń sesję", systemImage: "trash")
            }
        }
    }
}

// MARK: - Shared chrome

struct CodeSidebarSectionLabel: View {
    let title: String
    var systemImage: String?

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
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
    }
}

extension View {
    /// Selected / hovered background of a row in the agent sidebar.
    func codeSidebarRowChrome(isActive: Bool, isHovered: Bool) -> some View {
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
