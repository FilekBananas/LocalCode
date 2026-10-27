import SwiftUI
import LocalCodeKit

/// The "Kod" section: the coding agent.
///
/// Left column — the project folders and their sessions. Right column — the
/// header (where, with what model, under which permission mode, how much
/// context is left), the transcript and the composer. Everything it shows comes
/// from `AgentStore`, and every action goes back through it.
public struct CodeRootView: View {
    @Environment(AgentStore.self) private var store

    @State private var draft = ""
    @State private var focusRequest = 0
    @State private var pendingImages: [AgentImage] = []

    public init() {}

    public var body: some View {
        @Bindable var store = store
        return HStack(spacing: 0) {
            ProjectSidebar()
            VStack(spacing: 0) {
                AgentHeaderBar()
                connectionStrip
                content
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .banner($store.errorMessage)
        }
        .background(Theme.bg)
        .tint(Theme.accent)
        .task {
            await store.start()
            // Follows the model picked in the ModelLoaderBar above this view;
            // cancelled automatically when the section goes away.
            await store.watchModelSelection()
        }
        .onChange(of: ShellBus.shared.tick) { _, _ in
            handleShellAction(ShellBus.shared.latest)
        }
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if !store.hasProject {
            EmptyState(systemImage: "folder.badge.plus",
                       title: "Wybierz folder projektu",
                       message: "Agent pracuje w jednym folderze: czyta z niego pliki, zmienia je i uruchamia w nim polecenia.",
                       actionTitle: "Otwórz folder…") {
                store.openProjectFolder()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
        } else if store.session == nil {
            EmptyState(systemImage: "chevron.left.forwardslash.chevron.right",
                       title: "Brak otwartej sesji",
                       message: "Zacznij nową sesję w folderze \(store.selectedProject?.shortPath ?? "") albo wybierz jedną z listy po lewej.",
                       actionTitle: "Nowa sesja") {
                Task { await store.newSession() }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.bg)
        } else {
            AgentTranscript()
            composerBar
        }
    }

    private var composerBar: some View {
        VStack(spacing: 0) {
            Hairline()
            AgentComposer(text: $draft,
                          isBusy: store.isBusy,
                          isEnabled: store.canSend,
                          hasSession: store.session != nil,
                          queuedCount: store.queuedMessages.count,
                          blockedReason: blockedReason,
                          commands: store.slashCommands,
                          focusRequest: focusRequest,
                          images: $pendingImages,
                          allowsImages: true,
                          seesImages: store.currentModelSeesImages,
                          onSend: { store.send($0, images: $1) },
                          onStop: { store.interrupt() })
                .frame(maxWidth: 860)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 16)
                .padding(.top, 10)
                .padding(.bottom, 12)
        }
        .background(Theme.bg)
    }

    private var blockedReason: String? {
        if store.pendingPermission != nil { return "Agent czeka na Twoją decyzję." }
        if case .reconnecting = store.connectionState { return "Brak połączenia z sesją." }
        return nil
    }

    // MARK: - Connection

    @ViewBuilder
    private var connectionStrip: some View {
        if let notice = store.connectionNotice {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small).scaleEffect(0.65).frame(width: 12, height: 12)
                Text(notice)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .frame(height: 26)
            .background(Theme.warningSoft)
            .overlay(alignment: .bottom) { Hairline() }
        }
    }

    // MARK: - Shell actions

    private func handleShellAction(_ action: ShellBus.Action?) {
        switch action {
        case .refresh:
            Task { await store.refresh() }
        case .stopStreaming:
            // Esc: refuse the pending question first, otherwise stop the turn.
            if store.pendingPermission != nil {
                store.answerPermission("deny")
            } else if store.isBusy {
                store.interrupt()
            }
        default:
            break
        }
    }
}
