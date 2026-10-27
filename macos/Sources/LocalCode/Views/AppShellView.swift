import SwiftUI
import LocalCodeKit
import ChatFeature
import ModelsFeature
import CodeFeature
import ImagesFeature
import SpeechFeature
import VideoFeature

/// The window once the server is up: the navigation rail and the section it
/// selected.
///
/// The three feature stores are created here, once, and injected into the whole
/// tree with `.environment(...)`. That is what lets the model browser, the chat
/// and the agent share one `APIClient` — and therefore one server, one runtime
/// and one copy of every loaded model.
struct AppShellView: View {
    let controller: ServerController

    @State private var section: AppSection = .chat
    @State private var chatStore: ChatStore
    @State private var modelsStore: ModelsStore
    @State private var agentStore: AgentStore
    @State private var imagesStore: ImagesStore
    @State private var speechStore: SpeechStore
    @State private var videoStore: VideoStore
    /// Pomocnik: the coding agent's store in help bot mode, so its panel can
    /// reuse the transcript, the permission cards and the composer.
    @State private var helpbotStore: AgentStore
    @State private var showHelpbot = false
    @State private var helpbotWidth: CGFloat = 380
    @State private var bus = ShellBus.shared

    init(controller: ServerController) {
        self.controller = controller
        _chatStore = State(initialValue: ChatStore(api: controller.api))
        _modelsStore = State(initialValue: ModelsStore(api: controller.api))
        _agentStore = State(initialValue: AgentStore(api: controller.api))
        _imagesStore = State(initialValue: ImagesStore(api: controller.api))
        _speechStore = State(initialValue: SpeechStore(api: controller.api))
        _videoStore = State(initialValue: VideoStore(api: controller.api))
        _helpbotStore = State(initialValue: AgentStore(api: controller.api, helpbot: true))
    }

    var body: some View {
        HStack(spacing: 0) {
            NavigationRail(selection: $section)
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            // Next to the open section rather than instead of it: the help bot
            // loads models and changes settings, and the point is to watch that
            // land on the screen it belongs to. Its own store is injected here,
            // so it overrides the coding agent's for this subtree only.
            //
            // A plain column, not `.inspector`: the inspector is a full-height
            // column that reaches into the title bar, and next to it the open
            // section lost its top — the store's header and search field slid
            // under the toolbar.
            if showHelpbot {
                HelpbotResizeHandle(width: $helpbotWidth)
                HelpbotPanel()
                    .environment(helpbotStore)
                    .frame(width: helpbotWidth)
                    .transition(.move(edge: .trailing))
            }
        }
        .animation(.easeOut(duration: 0.18), value: showHelpbot)
        .background(Theme.bg)
        .tint(Theme.accent)
        .navigationTitle(section.title)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                DownloadsIndicator()
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showHelpbot.toggle()
                } label: {
                    Label("Pomocnik", systemImage: "sparkles")
                }
                .help("Pomocnik — znajdzie, pobierze i ustawi modele oraz aplikację za Ciebie (⌘⇧H)")
                .keyboardShortcut("h", modifiers: [.command, .shift])
            }
        }
        // The stores are injected *outside* `.toolbar` on purpose: toolbar
        // content is hosted in the window's title bar, not inside this view, so
        // it only sees environment values applied above the toolbar modifier.
        // With `.environment` below it, `DownloadsIndicator` would find no
        // `ModelsStore`.
        .environment(chatStore)
        .environment(modelsStore)
        .environment(agentStore)
        .environment(imagesStore)
        .environment(speechStore)
        .environment(videoStore)
        .task {
            // "Ulubione" are shared by every picker; the list itself arrives
            // with the settings `ModelsStore` reads below.
            FavoriteModels.shared.attach(controller.api)
            await modelsStore.start()
            await chatStore.start()
            // `/api/bootstrap` does not carry the application settings, so the
            // chat's default reasoning level is handed over by the shell, which
            // reads them through `ModelsStore`.
            chatStore.defaultEffort = modelsStore.settings.chat.reasoningEffort
        }
        .onChange(of: modelsStore.settings.chat.reasoningEffort) { _, value in
            chatStore.defaultEffort = value
        }
        .onChange(of: bus.tick) { _, _ in
            if case let .show(target) = bus.latest { section = target }
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var content: some View {
        switch section {
        case .chat:
            VStack(spacing: 0) {
                ModelLoaderBar()
                ChatShellView()
            }
        case .code:
            VStack(spacing: 0) {
                ModelLoaderBar()
                CodeRootView()
            }
        case .images:
            ImagesView()
        case .speech:
            SpeechView()
        case .video:
            VideoView()
        case .discover:
            ModelStoreView()
        case .myModels:
            MyModelsView()
        case .runtime:
            RuntimeView()
        case .settings:
            ChatSettingsView(extraTitle: "Serwer i modele", agent: AnyView(AgentSettingsPanel())) {
                ServerSettingsPanel(controller: controller)
                RemoteProvidersPanel()
                ReasoningDefaultsPanel()
                CerberPanel()
            }
        }
    }
}

/// The hairline between a section and Pomocnik; drag it to resize the panel.
private struct HelpbotResizeHandle: View {
    @Binding var width: CGFloat
    @State private var startWidth: CGFloat?

    private static let range: ClosedRange<CGFloat> = 320...560

    var body: some View {
        Hairline(vertical: true)
            .overlay {
                Color.clear
                    .frame(width: 8)
                    .contentShape(Rectangle())
                    .onHover { inside in
                        if inside { NSCursor.resizeLeftRight.push() } else { NSCursor.pop() }
                    }
                    .gesture(
                        DragGesture(minimumDistance: 1, coordinateSpace: .global)
                            .onChanged { value in
                                let start = startWidth ?? width
                                startWidth = start
                                width = min(max(start - value.translation.width, Self.range.lowerBound),
                                            Self.range.upperBound)
                            }
                            .onEnded { _ in startWidth = nil }
                    )
            }
    }
}
