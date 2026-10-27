import Foundation
import AppKit
import Observation
import LocalCodeKit

/// Everything the "Kod" section knows: the opened project folders, the agent
/// sessions of the selected folder, the transcript of the open session, the
/// live socket and the permission question the agent is waiting on.
///
/// One instance is created by the app shell and injected with
/// `.environment(store)`; every view in this module reads it with
/// `@Environment(AgentStore.self)`.
///
/// Threading: everything is `@MainActor`. The socket delivers already-decoded
/// events on the main actor too, so a turn is a plain sequence of mutations —
/// no locks, no queues, and the transcript can never be observed half-updated.
@MainActor
@Observable
public final class AgentStore {

    // MARK: Dependencies

    @ObservationIgnored public let api: APIClient
    @ObservationIgnored private var socket: AgentSocket!
    @ObservationIgnored private var builder = TranscriptBuilder()
    /// Pictures sent and not yet echoed by a `turn_start`: (text, thumbnails).
    @ObservationIgnored private var pendingImages: [(text: String, thumbnails: [Data])] = []

    // MARK: Projects and sessions

    public private(set) var projects: [ProjectRef] = []
    public private(set) var selectedProject: ProjectRef?
    public private(set) var sessions: [AgentSessionDTO] = []
    public private(set) var session: AgentSessionDTO?
    public private(set) var isLoadingSessions = false
    public private(set) var isLoadingTranscript = false

    // MARK: Transcript and turn state

    public private(set) var entries: [TranscriptEntry] = []
    public private(set) var todos: [TodoItem] = []
    /// `/goal` of the open session (`nil`: none).
    public private(set) var goal: AgentGoal?
    public private(set) var context = ContextStats()
    /// Questions waiting for an answer, oldest first. Usually one; a helper
    /// working in the background can ask while the agent asks too.
    public private(set) var pendingPermissions: [PermissionRequestDTO] = []
    /// The question on screen: the oldest one waiting.
    public var pendingPermission: PermissionRequestDTO? { pendingPermissions.first }
    /// Helpers working in the background right now (their short names).
    public private(set) var backgroundHelpers: Set<String> = []
    /// „Ładowanie modelu…”, „Myślę…” — shown at the end of the transcript.
    public private(set) var statusMessage: String?
    public private(set) var isBusy = false
    /// Messages sent while a turn was running and parked by the server.
    public private(set) var queuedMessages: [String] = []

    // MARK: Models and connection

    public private(set) var models: [AgentModelOption] = []

    // MARK: Help bot

    /// This store drives Pomocnik rather than the coding agent: one
    /// conversation in the help bot's own directory, no project list, and no
    /// following of the shell's model picker. The server recognises the
    /// session by that directory and gives it the help bot's prompt and tools.
    public let isHelpbot: Bool
    /// Whether the help bot's model is on disk, downloading, or missing.
    public private(set) var helpbot: HelpbotStatusDTO?
    public private(set) var isInstallingHelpbot = false
    /// The slash commands this server offers a graphical client, and the text
    /// behind `/init` — both read from `GET /api/agent/commands` so the app and
    /// the terminal cannot drift apart (see `localcode/tui/commands.py`).
    public private(set) var slashCommands: [AgentSlashCommand] = []
    @ObservationIgnored private var initPrompt: String?
    public private(set) var connectionState: AgentConnectionState = .idle
    public private(set) var isReady = false
    public var errorMessage: String?

    /// Model picked in the shell's `ModelLoaderBar` before any session existed.
    @ObservationIgnored private var preferredModelKey: String?
    @ObservationIgnored private var defaultMode: AgentMode = .ask
    @ObservationIgnored private var attentionRequest: Int?
    @ObservationIgnored private var lastKnownDefaultModel: String?
    @ObservationIgnored private var isCompacting = false
    @ObservationIgnored private var compactWatchdog: Task<Void, Never>?
    @ObservationIgnored private var modelObserver: NSObjectProtocol?

    /// Posted by whoever owns the model picker (the shell's `ModelLoaderBar`)
    /// when the user selects a different model; `userInfo["modelKey"]` is the
    /// model key. Sending it is optional — the agent header has its own picker.
    public static let selectedModelDidChange = Notification.Name("LocalCode.selectedModelDidChange")

    // MARK: - Life cycle

    public init(api: APIClient, helpbot: Bool = false) {
        self.api = api
        self.isHelpbot = helpbot
        self.socket = AgentSocket(api: api)
        socket.onEvent = { [weak self] event in self?.handle(event) }
        socket.onStateChange = { [weak self] state in self?.connectionChanged(to: state) }
        // The help bot runs on its own model, always: following the shell's
        // model picker would quietly turn it into whatever model was picked.
        guard !helpbot else { return }
        modelObserver = NotificationCenter.default.addObserver(
            forName: Self.selectedModelDidChange, object: nil, queue: .main
        ) { [weak self] note in
            guard let key = note.userInfo?["modelKey"] as? String else { return }
            MainActor.assumeIsolated { self?.setModel(key) }
        }
    }

    deinit {
        if let modelObserver { NotificationCenter.default.removeObserver(modelObserver) }
    }

    /// Called by the shell once the server answers `/api/health`.
    public func start() async {
        guard !isReady else { return }
        projects = RecentProjects.load()
        await refreshModels()
        await loadDefaults()
        await loadCommands()
        if let first = projects.first(where: { $0.exists }) ?? projects.first {
            await select(project: first, openLatest: true)
        }
        isReady = true
    }

    /// ⌘R / manual refresh: the model list and the session list.
    public func refresh() async {
        await refreshModels()
        await refreshSessions()
    }

    private func refreshModels() async {
        do {
            models = try await api.agentModels().filter { !$0.isEmbedding }
        } catch {
            // Swallowing this used to show up much later and in the wrong
            // place: the picker was simply empty, and the reason only surfaced
            // as a "model_required" error after the first message was sent.
            models = []
            errorMessage = "Nie udało się wczytać listy modeli: " + APIError.describe(error)
        }
    }

    private func loadCommands() async {
        guard let served = try? await api.agentCommands() else { return }
        slashCommands = served.commands
        initPrompt = served.initPrompt
    }

    private func loadDefaults() async {
        guard let defaults = try? await api.agentDefaults() else { return }
        lastKnownDefaultModel = defaults.modelKey
        if preferredModelKey == nil { preferredModelKey = defaults.modelKey }
        if let mode = defaults.mode.flatMap(AgentMode.init(wireValue:)) { defaultMode = mode }
    }

    /// Follows the model picked in the shell's `ModelLoaderBar`.
    ///
    /// That bar edits `agent.defaultModel` on the server (it is shared with the
    /// terminal app and survives a restart) and tells nobody, and `CodeFeature`
    /// must not depend on `ModelsFeature` to hear about it. So while the "Kod"
    /// section is on screen this polls the setting and forwards a *change* to
    /// the open session — a change only ever happens when someone picked a new
    /// model. A model chosen in this screen's own header is not overwritten,
    /// because the polled value did not change.
    ///
    /// Runs until the caller's task is cancelled (the section disappears).
    public func watchModelSelection(interval: Duration = .seconds(3)) async {
        while !Task.isCancelled {
            try? await Task.sleep(for: interval)
            guard !Task.isCancelled else { return }
            guard let defaults = try? await api.agentDefaults() else { continue }
            let key = defaults.modelKey
            guard key != lastKnownDefaultModel else { continue }
            lastKnownDefaultModel = key
            guard let key, !key.isEmpty else { continue }
            setModel(key)
        }
    }

    // MARK: - Projects

    /// „Otwórz folder…”
    public func openProjectFolder() {
        guard let path = RecentProjects.choose() else { return }
        projects = RecentProjects.adding(path, to: projects)
        Task { await select(project: ProjectRef(path: path), openLatest: true) }
    }

    public func select(project: ProjectRef, openLatest: Bool = true) async {
        if selectedProject?.path != project.path {
            closeSession()
        }
        selectedProject = project
        projects = RecentProjects.adding(project.path, to: projects)
        guard project.exists else {
            // A folder can be renamed, unmounted or deleted between two
            // launches. Saying so here beats an empty screen whose buttons all
            // fail one click later.
            sessions = []
            errorMessage = "Folder „\(project.shortPath)” już nie istnieje. "
                + "Wybierz inny albo usuń go z listy."
            return
        }
        await refreshSessions()
        guard openLatest, session == nil else { return }
        if let newest = sessions.first {
            await open(session: newest)
        } else {
            // "Pick a folder and it works": a folder opened for the first time
            // has no session yet, and making the user click "Nowa sesja" to get
            // one is a step that carries no decision.
            await newSession()
        }
    }

    public func remove(project: ProjectRef) {
        projects = RecentProjects.removing(project, from: projects)
        guard selectedProject?.path == project.path else { return }
        closeSession()
        selectedProject = nil
        sessions = []
    }

    // MARK: - Sessions

    public func refreshSessions() async {
        guard let project = selectedProject else {
            sessions = []
            return
        }
        isLoadingSessions = true
        defer { isLoadingSessions = false }
        do {
            sessions = try await api.agentSessions(cwd: project.path)
        } catch {
            sessions = []
            errorMessage = APIError.describe(error)
        }
    }

    // MARK: - Help bot

    /// What the help bot panel calls instead of `start()`.
    public func startHelpbot() async {
        guard isHelpbot else { return }
        await loadCommands()
        // /init writes LOCALCODE.md into a project and /cwd names one; the help
        // bot has neither, so offering them would only invite a confused answer.
        slashCommands.removeAll { ["init", "cwd"].contains($0.name) }
        await refreshHelpbotStatus()
        if helpbot?.state == .ready, session == nil {
            await openHelpbot(new: false)
        }
    }

    public func refreshHelpbotStatus() async {
        do {
            helpbot = try await api.helpbotStatus()
        } catch {
            errorMessage = APIError.describe(error)
        }
    }

    /// „Pobierz pomocnika” — the server checks the free disk space first.
    public func installHelpbot() async {
        isInstallingHelpbot = true
        defer { isInstallingHelpbot = false }
        do {
            helpbot = try await api.installHelpbot()
        } catch {
            errorMessage = APIError.describe(error)
        }
    }

    /// The latest help bot conversation, or a fresh one.
    public func openHelpbot(new: Bool) async {
        do {
            let dto = try await api.helpbotSession(new: new)
            // Set directly, never through `select(project:)`: that would add
            // the help bot's directory to the user's recent projects.
            selectedProject = ProjectRef(path: dto.cwd)
            if !sessions.contains(where: { $0.id == dto.id }) { sessions.insert(dto, at: 0) }
            await open(session: dto)
        } catch {
            errorMessage = APIError.describe(error)
        }
    }

    /// Creates a session in the selected folder and opens it.
    public func newSession() async {
        if isHelpbot {
            await openHelpbot(new: true)
            return
        }
        guard let project = selectedProject else { return }
        guard project.exists else {
            errorMessage = "Folder „\(project.shortPath)” już nie istnieje."
            return
        }
        do {
            let created = try await api.createAgentSession(cwd: project.path,
                                                           modelKey: currentModelKey,
                                                           mode: (currentMode ?? defaultMode).wireValue)
            sessions.insert(created, at: 0)
            await open(session: created)
        } catch {
            errorMessage = APIError.describe(error)
        }
    }

    public func open(session target: AgentSessionDTO) async {
        if session?.id == target.id, connectionState.isConnected { return }
        closeSession()
        session = target
        isLoadingTranscript = true
        // The stored transcript first (instant), then the socket — which
        // replays `session` + `history` on connect and keeps it up to date.
        do {
            let snapshot = try await api.agentSession(id: target.id)
            if session?.id == target.id {
                session = snapshot.session
                builder.load(items: snapshot.items)
                entries = builder.entries
                todos = snapshot.todos
                isBusy = snapshot.session.busy
                context = ContextStats(used: snapshot.session.contextUsed,
                                       limit: snapshot.session.contextLimit)
            }
        } catch {
            errorMessage = APIError.describe(error)
            // The transcript comes over REST; if the server does not know this
            // session, the socket would only retry a 404 until it gave up.
            isLoadingTranscript = false
            closeSession()
            await refreshSessions()
            return
        }
        isLoadingTranscript = false
        guard session?.id == target.id else { return }
        socket.connect(sessionId: target.id)
    }

    public func deleteSession(_ target: AgentSessionDTO) async {
        do {
            try await api.deleteAgentSession(id: target.id)
        } catch {
            errorMessage = APIError.describe(error)
            return
        }
        sessions.removeAll { $0.id == target.id }
        if session?.id == target.id {
            closeSession()
            if let newest = sessions.first { await open(session: newest) }
        }
    }

    /// Drops the open session without touching the server.
    public func closeSession() {
        endCompaction()
        socket.disconnect()
        session = nil
        builder.reset()
        entries = []
        todos = []
        context = ContextStats()
        pendingPermissions = []
        backgroundHelpers = []
        statusMessage = nil
        isBusy = false
        queuedMessages = []
    }

    // MARK: - Commands

    /// Sends the composer's text: a slash command, or a message for the agent.
    public func send(_ raw: String, images: [AgentImage] = []) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || !images.isEmpty, session != nil else { return }
        if images.isEmpty, text.hasPrefix("/"), handleSlashCommand(text) { return }
        if isBusy { queuedMessages.append(text.isEmpty ? "[obraz]" : text) }
        if images.isEmpty {
            socket.send(.user(text))
        } else {
            pendingImages.append((text, images.map(\.thumbnail)))
            socket.send(.userWithImages(text, images))
        }
    }

    /// Whether the session's model takes pictures (`nil`: the model is unknown here).
    public var currentModelSeesImages: Bool? {
        guard let key = currentModelKey, let model = models.first(where: { $0.key == key }) else { return nil }
        return model.seesImages
    }

    public func interrupt() {
        guard session != nil else { return }
        // The queue stays: the server sends the first queued message as soon
        // as the stopped turn has wound down.
        socket.send(.interrupt)
    }

    /// „Kompaktuj” — summarise the history now.
    ///
    /// The server builds the agent lazily, on the first message of a session,
    /// and `/compact` before that is a silent no-op — so a session that has not
    /// generated anything yet (no context report) is answered here instead of
    /// leaving a spinner running for nothing. Otherwise the status line is
    /// closed by the `context` update that always follows a compaction (a
    /// `compaction` event only arrives when something was actually dropped),
    /// with a watchdog in case the server says nothing at all.
    public func compact() {
        guard session != nil else { return }
        guard context.limit > 0 else {
            note("Kompaktowanie będzie możliwe po pierwszej wiadomości w tej sesji.")
            return
        }
        isCompacting = true
        statusMessage = "Kompaktuję kontekst…"
        socket.send(.compact)
        compactWatchdog?.cancel()
        compactWatchdog = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 90_000_000_000)
            guard !Task.isCancelled, let self, self.isCompacting else { return }
            self.endCompaction()
            self.note("Serwer nie odpowiedział na żądanie kompaktowania.")
        }
    }

    /// Closes the „Kompaktuję kontekst…” line, whatever ended it.
    private func endCompaction() {
        compactWatchdog?.cancel()
        compactWatchdog = nil
        guard isCompacting else { return }
        isCompacting = false
        statusMessage = nil
    }

    public func clearTranscript() {
        guard session != nil else { return }
        socket.send(.clear)
    }

    /// `/goal wyczyść` — stop working towards the goal.
    public func clearGoal() {
        guard session != nil else { return }
        socket.send(.clearGoal)
    }

    /// Changes the reasoning effort of the open session.
    ///
    /// The server answers with a `session` broadcast, which is what actually
    /// updates the control — the optimistic write below only stops the segment
    /// from flickering back for the length of one round trip.
    public func setEffort(_ level: String) {
        guard session != nil, session?.effort != level else { return }
        session?.effort = level
        socket.send(.setEffort(level))
    }

    public func setMode(_ mode: AgentMode) {
        defaultMode = mode
        guard session != nil else { return }
        session?.mode = mode.wireValue
        socket.send(.setMode(mode.wireValue))
    }

    /// Points the agent at another model. Also the hook the shell's
    /// `ModelLoaderBar` calls (directly or through `selectedModelDidChange`).
    public func setModel(_ key: String) {
        guard !key.isEmpty else { return }
        preferredModelKey = key
        guard session != nil, session?.modelKey != key else { return }
        session?.modelKey = key
        socket.send(.setModel(key))
    }

    public func answerPermission(_ decision: String) {
        guard let pending = pendingPermission else { return }
        pendingPermissions.removeFirst()
        cancelAttention()
        socket.send(.permission(requestId: pending.requestId, decision: decision))
    }

    /// `true` when the text was a slash command and has been handled.
    private func handleSlashCommand(_ text: String) -> Bool {
        let parts = text.dropFirst().split(separator: " ", maxSplits: 1).map(String.init)
        guard let command = parts.first?.lowercased() else { return false }
        let argument = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : ""
        switch command {
        case "goal":
            let words: Set<String> = ["wyczyść", "wyczysc", "clear", "off", "stop", "usuń", "usun", "koniec"]
            if argument.isEmpty {
                note(goal.map(\.noticeText) ?? "Brak celu. Ustaw go: /goal opis tego, co ma być zrobione.")
            } else if words.contains(argument.lowercased()) {
                clearGoal()
            } else {
                if isBusy { queuedMessages.append(argument) }
                socket.send(.setGoal(argument))
            }
            return true
        case "compact":
            compact()
            return true
        case "clear":
            clearTranscript()
            return true
        case "mode":
            if let mode = AgentMode.allCases.first(where: {
                $0.wireValue == argument || $0.label.lowercased() == argument.lowercased()
            }) {
                setMode(mode)
                note("Tryb uprawnień: \(mode.label).")
            } else {
                note("Użycie: /mode " + AgentMode.allCases.map(\.wireValue).joined(separator: " | "))
            }
            return true
        case "model":
            if argument.isEmpty {
                note("Użycie: /model <klucz modelu>. Modele: "
                     + models.prefix(5).map(\.key).joined(separator: ", ") + "…")
            } else if let model = models.first(where: { $0.key == argument })
                        ?? models.first(where: { $0.key.localizedCaseInsensitiveContains(argument) }) {
                setModel(model.key)
                note("Model sesji: \(RemoteModelKey.name(model.displayName, key: model.key)).")
            } else {
                note("Nie znaleziono modelu „\(argument)”.")
            }
            return true
        case "think":
            let levels = ["off", "low", "medium", "high"]
            if argument.lowercased() == Ultracode.wireValue {
                // By name only: the cycle below never lands on it, because
                // every helper is another model turn (a paid one, remotely).
                setEffort(Ultracode.wireValue)
                note("Poziom myślenia: ultracode. \(EffortHelp.ultracode)")
            } else if levels.contains(argument.lowercased()) {
                setEffort(argument.lowercased())
                note("Poziom myślenia: \(argument.lowercased()).")
            } else {
                // No argument cycles, exactly as the terminal does.
                let current = session?.effort ?? "medium"
                let next = levels[((levels.firstIndex(of: current) ?? 2) + 1) % levels.count]
                setEffort(next)
                note("Poziom myślenia: \(next).")
            }
            return true
        case "context":
            note(contextBreakdown())
            return true
        case "stats":
            note(lastTurnSummary())
            return true
        case "cwd":
            note("Katalog roboczy: " + (session?.cwd ?? selectedProject?.path ?? "—"))
            return true
        case "init":
            guard let prompt = initPrompt else {
                note("Nie udało się wczytać polecenia /init z serwera.")
                return true
            }
            send(prompt)
            return true
        case "help":
            note(helpText())
            return true
        case "resume":
            resume(matching: argument)
            return true
        default:
            // Anything else used to be sent to the model as an ordinary
            // message, so a typo became a turn: the model dutifully tried to
            // make sense of "/complact".
            let known = slashCommands.map { "/" + $0.name }.sorted().joined(separator: ", ")
            note("Nieznane polecenie „/\(command)”. Dostępne: \(known).")
            return true
        }
    }

    /// „/resume [id albo fragment tytułu]” — the same thing the terminal does.
    ///
    /// Without an argument it lists this folder's sessions, newest first; with
    /// one it opens the session whose id starts with it, or whose title contains
    /// it. The sidebar already does this with a click, but a command the server
    /// says the app offers has to work here too.
    private func resume(matching argument: String) {
        let others = sessions.filter { $0.id != session?.id }
        guard !argument.isEmpty else {
            guard !others.isEmpty else {
                note("W tym folderze nie ma innych sesji.")
                return
            }
            let lines = others.prefix(8).map { "• \($0.id.prefix(8)) — \($0.title.isEmpty ? "bez tytułu" : $0.title)" }
            note((["Sesje w tym folderze (/resume <id>):"] + lines).joined(separator: "\n"))
            return
        }
        let needle = argument.lowercased()
        guard let target = others.first(where: { $0.id.lowercased().hasPrefix(needle) })
                ?? others.first(where: { $0.title.lowercased().contains(needle) }) else {
            note("Nie znalazłem sesji „\(argument)”. Wpisz /resume bez argumentu, żeby zobaczyć listę.")
            return
        }
        Task { await open(session: target) }
    }

    /// „/context” — where the window is going, in the same breakdown the gauge shows.
    private func contextBreakdown() -> String {
        guard context.limit > 0 else { return "Kontekst: jeszcze nic nie policzono." }
        let percent = Int((Double(context.used) / Double(context.limit) * 100).rounded())
        return "Kontekst: \(context.used) / \(context.limit) tokenów (\(percent)%) — "
            + "prompt \(context.system), narzędzia \(context.tools), "
            + "rozmowa \(context.history), rezerwa \(context.reserve)."
    }

    /// „/stats” — the last turn, from the entry the transcript already holds.
    private func lastTurnSummary() -> String {
        guard let stats = entries.reversed().first(where: { $0.kind == .stats })?.stats else {
            return "Jeszcze nie było tury w tej sesji."
        }
        var parts: [String] = []
        if let steps = stats.steps { parts.append("kroków: \(steps)") }
        if let prompt = stats.promptTokens { parts.append("prompt: \(prompt)") }
        if let out = stats.completionTokens { parts.append("odpowiedź: \(out)") }
        if let tps = stats.genTps { parts.append(String(format: "%.1f tok/s", tps)) }
        if let ms = stats.elapsedMs ?? stats.totalMs {
            parts.append(String(format: "%.1f s", ms / 1000))
        }
        return parts.isEmpty ? "Ostatnia tura nie zgłosiła statystyk."
                             : "Ostatnia tura — " + parts.joined(separator: ", ") + "."
    }

    private func helpText() -> String {
        let lines = slashCommands
            .sorted { $0.name < $1.name }
            .map { "/\($0.name) — \($0.summary)" }
        return (["Polecenia:"] + lines).joined(separator: "\n")
    }

    private func note(_ text: String) {
        builder.appendNotice(text)
        entries = builder.entries
    }

    // MARK: - Events

    private func handle(_ event: AgentEvent) {
        switch event {
        case let .session(dto):
            session = dto
            isBusy = dto.busy
            goal = dto.goal
            if dto.contextLimit > 0, context.limit == 0 {
                context = ContextStats(used: dto.contextUsed, limit: dto.contextLimit)
            }
            if let index = sessions.firstIndex(where: { $0.id == dto.id }) {
                sessions[index] = dto
            } else if selectedProject?.path == dto.cwd {
                sessions.insert(dto, at: 0)
            }

        case let .context(stats):
            context = stats
            endCompaction()

        case let .todos(items):
            todos = items
            _ = builder.apply(event)

        case let .goal(update):
            goal = update.state == "cleared" ? nil : update
            if builder.apply(event) { entries = builder.entries }

        case let .permissionRequest(request):
            if !pendingPermissions.contains(where: { $0.requestId == request.requestId }) {
                pendingPermissions.append(request)
            }
            requestAttention()

        case let .permissionResolved(requestId, _):
            pendingPermissions.removeAll { $0.requestId == requestId }
            _ = builder.apply(event)

        case let .status(message, phase):
            switch phase {
            case .tool:
                _ = builder.apply(event)
            case .queued:
                statusMessage = nil
            default:
                statusMessage = message.isEmpty ? nil : message
            }

        case let .turnStart(_, userText):
            isBusy = true
            statusMessage = nil
            if !queuedMessages.isEmpty { queuedMessages.removeFirst() }
            _ = builder.apply(event)
            // The server echoes the text (with @files expanded below it), not
            // the pictures; the thumbnails kept at send time go on the bubble.
            if let index = pendingImages.firstIndex(where: { userText.hasPrefix($0.text) }) {
                builder.attachImages(pendingImages[index].thumbnails)
                pendingImages.remove(at: index)
                entries = builder.entries
            }

        case .userMessage:
            // The agent took a queued message mid-turn: it leaves the queue
            // and becomes a bubble where the agent read it.
            statusMessage = nil
            if !queuedMessages.isEmpty { queuedMessages.removeFirst() }
            if builder.apply(event) { entries = builder.entries }

        case .textDelta, .reasoningDelta:
            statusMessage = nil
            _ = builder.apply(event)

        case .toolCall, .toolResult, .compaction:
            endCompaction()
            statusMessage = nil
            _ = builder.apply(event)

        case .autoReview:
            // The verdict goes on its tool card; "Recenzent ocenia akcję…" is over.
            statusMessage = nil
            if builder.apply(event) { entries = builder.entries }

        case let .subagentStart(payload) where payload.background:
            backgroundHelpers.insert(payload.agentId)
            if builder.apply(event) { entries = builder.entries }

        case let .subagentEnd(payload) where payload.background:
            backgroundHelpers.remove(payload.agentId)
            if builder.apply(event) { entries = builder.entries }

        case .subagentStart, .subagentStep, .subagentEnd:
            // The helper's progress belongs to the `delegate` card; the
            // transcript builder nests it there. Nothing about the turn changes
            // — a helper finishing is not the turn finishing.
            if builder.apply(event) { entries = builder.entries }

        case let .turnEnd(_, reason, _):
            isBusy = false
            endCompaction()
            statusMessage = nil
            if pendingPermission != nil { cancelAttention() }
            pendingPermissions = []
            backgroundHelpers = []
            _ = builder.apply(event)
            if reason != "interrupted" { requestAttention() }

        case let .error(message, _):
            errorMessage = message
            endCompaction()
            statusMessage = nil
            _ = builder.apply(event)

        case let .ack(of, status):
            if of == "user", status == "queued" { isBusy = true }

        case .history:
            _ = builder.apply(event)
            queuedMessages = []

        case .unknown:
            break
        }
        entries = builder.entries
    }

    private func connectionChanged(to state: AgentConnectionState) {
        connectionState = state
        switch state {
        case .reconnecting:
            builder.failRunningTools(summary: "Połączenie przerwane")
            entries = builder.entries
            statusMessage = nil
        case .connected:
            errorMessage = nil
        default:
            break
        }
    }

    /// Bounces the dock icon when something needs the user and the window is
    /// not in front — a permission question must not wait unnoticed.
    private func requestAttention() {
        guard !NSApplication.shared.isActive else { return }
        cancelAttention()
        attentionRequest = NSApplication.shared.requestUserAttention(.criticalRequest)
    }

    /// Stops the dock icon bouncing once the question has been answered.
    private func cancelAttention() {
        guard let request = attentionRequest else { return }
        attentionRequest = nil
        NSApplication.shared.cancelUserAttentionRequest(request)
    }

    // MARK: - Derived state

    public var currentMode: AgentMode? {
        session.flatMap { AgentMode(wireValue: $0.mode) }
    }

    public var activeMode: AgentMode { currentMode ?? defaultMode }

    public var currentModelKey: String? { session?.modelKey ?? preferredModelKey }

    /// Human name of the session's model (falls back to the raw key).
    public var currentModelLabel: String {
        guard let key = currentModelKey, !key.isEmpty else { return "Nie wybrano modelu" }
        return RemoteModelKey.name(models.first { $0.key == key }?.displayName ?? key, key: key)
    }

    public var canSend: Bool { session != nil && pendingPermission == nil }

    public var hasProject: Bool { selectedProject != nil }

    /// What the connection banner shows, or `nil` when everything is fine.
    public var connectionNotice: String? { connectionState.label }
}

#if DEBUG
extension AgentStore {
    /// Pomocnik mid-conversation, for `swift run ScreenShots` — the panel is
    /// drawn off-screen next to another section and looked at, because nobody
    /// can click through this window where it is built. Debug only.
    public static func helpbotPreview(api: APIClient, turns: Int = 4) -> AgentStore {
        let store = AgentStore(api: api, helpbot: true)
        let status = #"{"model":"mlx-community/Qwen3-4B-Instruct-2507-4bit","displayName":"Qwen3 4B Instruct","#
            + #""state":"ready","loaded":true,"cwd":"/tmp/helpbot","autoDownload":true}"#
        store.helpbot = try? JSONDecoder().decode(HelpbotStatusDTO.self, from: Data(status.utf8))
        store.session = AgentSessionDTO(id: "preview", cwd: "/tmp/helpbot", title: "Pomocnik")
        var items: [[String: Any]] = []
        for turn in 0..<turns {
            items.append(["kind": "user", "id": "u\(turn)", "text": "a nie ma większego/lepszego?"])
            items.append(["kind": "tool", "id": "t\(turn)", "toolName": "model_search",
                          "title": "Szukam modeli wideo", "ok": true, "summary": "12 wyników"])
            items.append(["kind": "assistant", "id": "a\(turn)", "text": """
                Nie ma modeli większych niż **MiniMax H3** (11,7 GB) z dobrym zaufaniem dla \
                generowania wideo. Wszystkie inne są mniejsze lub mają niższe oceny.

                Jeśli chcesz więcej możliwości — możesz też spróbować **Wan 2.1 I2V** (8,1 GB), \
                który animuje obrazy, ale nie z opisu tekstu.

                Chcesz, żebym załadował **MiniMax H3** do pamięci? (To zajmie ~12 GB RAM.)
                """])
        }
        if let data = try? JSONSerialization.data(withJSONObject: items),
           let decoded = try? JSONDecoder().decode([AgentItemDTO].self, from: data) {
            store.builder.load(items: decoded)
            store.entries = store.builder.entries
        }
        return store
    }
}
#endif
