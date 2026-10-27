import Foundation
import SwiftUI
import AppKit
import LocalCodeKit

/// What a chosen model is going to be used for.
///
/// The server keeps one default per purpose (`app-settings.chat.defaultModel`
/// and `app-settings.agent.defaultModel`), and `ModelLoaderBar` edits exactly
/// one of them.
public enum ModelPurpose: String, CaseIterable, Identifiable, Sendable {
    case chat
    case agent

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .chat: "Model czatu"
        case .agent: "Model agenta"
        }
    }

    public var shortTitle: String {
        switch self {
        case .chat: "czat"
        case .agent: "agent"
        }
    }
}

private struct ModelPurposeKey: EnvironmentKey {
    static let defaultValue: ModelPurpose = .chat
}

extension EnvironmentValues {
    /// Which default `ModelLoaderBar()` edits when it is created without an
    /// explicit purpose. The shell renders the bar above both "Czat" and "Kod";
    /// set this to `.agent` on the Kod branch and the same bar follows.
    public var modelPurpose: ModelPurpose {
        get { self[ModelPurposeKey.self] }
        set { self[ModelPurposeKey.self] = newValue }
    }
}

/// Everything the three model screens share: the local library, the Hugging
/// Face browser, the download queue, the runtime and the app settings.
///
/// The shell creates exactly one of these with the app's single `APIClient` and
/// injects it with `.environment(...)`, so "Odkrywaj", "Moje modele",
/// "Runtime", the toolbar's download indicator and the model bar above the chat
/// all read the same state and never disagree about what is loaded.
///
/// Live updates come from `/api/events`: one SSE connection carries runtime
/// snapshots, download progress and "the library changed". The connection is
/// re-opened with a capped exponential backoff, because the server can be
/// restarted under a running app.
@MainActor
@Observable
public final class ModelsStore {

    // MARK: Dependencies

    /// The one client the whole app shares. Feature views reach the API
    /// through the store, never through the environment.
    @ObservationIgnored public let api: APIClient

    // MARK: Server state

    public private(set) var library: [LocalModelDTO] = []
    public private(set) var runtime = RuntimeStatus()
    public private(set) var downloads: [DownloadJob] = []
    public private(set) var settings = AppConfig() {
        // Every settings answer carries the stars; the shared list follows it
        // so a change made by Pomocnik or the terminal shows up in the pickers.
        didSet { FavoriteModels.shared.update(from: settings.favoriteModels) }
    }

    // MARK: Catalog ("Sklep z modelami")

    /// The store rows for the category and query currently selected.
    public private(set) var catalog: [CatalogEntry] = []
    /// The server answered from its own copy — no network.
    public private(set) var isCatalogOffline = false
    public private(set) var isLoadingCatalog = false
    /// Selected category tab; `nil` is "Wszystkie".
    public var catalogCategory: String?
    public var catalogSearch = ""
    /// Show only what is already on disk.
    public var catalogInstalledOnly = false
    /// Show only the laboratories' own repositories, not community conversions.
    public var catalogOfficialOnly = false
    /// Which ordering the store is showing (server vocabulary, see `SORTS`).
    public var catalogSort = "polecane"
    /// Hide everything below this trust score (0 = show everything).
    public var catalogMinTrust = 0
    /// How many models matched before the page limit — "pokazuję 200 z 4312".
    public private(set) var catalogTotal = 0
    /// Catalog ids with an install request in flight, so the button on that one
    /// card can spin without freezing the whole grid.
    public private(set) var installingIds: Set<String> = []
    /// The entry open in the "Szczegóły" sheet, with its full variant list.
    public var catalogDetail: CatalogEntry?
    public private(set) var isLoadingCatalogDetail = false
    /// "Zaawansowane": the raw Hugging Face browser, reachable from the store.
    public var isShowingRawHub = false

    /// The curated "Polecane" list shown when the search field is empty.
    public private(set) var curated: [HubModel] = []
    public private(set) var searchResults: [HubModel] = []
    public private(set) var detail: HubModelDetail?

    // MARK: Browser state (kept here so switching sections does not lose it)

    public var searchText = ""
    public var formatFilter: HubFormatFilter = .all
    public var sortOrder: HubSortOrder = .trending
    /// The repository shown in the detail pane of "Odkrywaj".
    public private(set) var selectedRepo: String?
    /// Filter field of "Moje modele".
    public var libraryFilter = ""

    // MARK: Progress / errors

    public private(set) var isLoadingLibrary = false
    public private(set) var isSearching = false
    public private(set) var isLoadingDetail = false
    public private(set) var isRescanning = false
    /// True until the first `/api/hub/curated` answer, success or failure.
    public private(set) var isLoadingCurated = true
    /// Model keys with a load or unload request in flight.
    public private(set) var busyKeys: Set<String> = []
    /// True once the first round of requests came back.
    public private(set) var isReady = false
    /// `/api/events` is connected right now.
    public private(set) var isLive = false

    /// Shown in the LocalCodeKit banner style at the top of every screen.
    public var errorMessage: String?
    /// A short confirmation ("Ustawiono model czatu…") that fades by itself.
    public private(set) var statusMessage: String?

    // MARK: Private

    @ObservationIgnored private var eventsTask: Task<Void, Never>?
    @ObservationIgnored private var searchTask: Task<Void, Never>?
    @ObservationIgnored private var catalogTask: Task<Void, Never>?
    @ObservationIgnored private var catalogDetailTask: Task<Void, Never>?
    @ObservationIgnored private var detailTask: Task<Void, Never>?
    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var dismissedJobs: Set<String> = []
    @ObservationIgnored private var didStart = false

    public init(api: APIClient) {
        self.api = api
    }

    deinit {
        eventsTask?.cancel()
        searchTask?.cancel()
        detailTask?.cancel()
        statusTask?.cancel()
        catalogTask?.cancel()
        catalogDetailTask?.cancel()
    }

    // MARK: - Lifecycle

    /// Called by the shell once the server answers `/api/health`.
    public func start() async {
        guard !didStart else { return }
        didStart = true
        await refreshAll()
        isReady = true
        subscribeToEvents()
        Task { await self.loadCurated() }
        reloadCatalog()
    }

    /// Re-reads everything the server owns. Cheap enough for ⌘R.
    ///
    /// Sequential on purpose: all four handlers are local and fast, and every
    /// one of them writes main-actor state, so overlapping them would only add
    /// ordering questions for no measurable gain.
    public func refreshAll() async {
        await refreshLibrary()
        await refreshRuntime()
        await refreshDownloads()
        await refreshSettings()
    }

    public func refreshLibrary() async {
        isLoadingLibrary = true
        defer { isLoadingLibrary = false }
        do {
            library = try await api.library()
        } catch {
            report(error)
        }
    }

    public func refreshRuntime() async {
        let configured = await api.isConfigured
        serverBaseURL = configured ? await api.baseURL : nil
        do {
            runtime = try await api.runtimeStatus()
        } catch {
            report(error)
        }
    }

    public func refreshDownloads() async {
        do {
            downloads = try await api.downloadJobs()
        } catch {
            report(error)
        }
    }

    public func refreshSettings() async {
        do {
            settings = try await api.appSettings()
        } catch {
            report(error)
        }
    }

    /// `POST /api/library/rescan` — after the user moved the models directory
    /// or dropped files in by hand.
    public func rescanLibrary() async {
        isRescanning = true
        defer { isRescanning = false }
        do {
            library = try await api.rescanLibrary()
            announce("Przeszukano katalog modeli.")
        } catch {
            report(error)
        }
    }

    // MARK: - Events

    /// Subscribes to `/api/events` and keeps the subscription alive.
    private func subscribeToEvents() {
        eventsTask?.cancel()
        eventsTask = Task { [weak self] in
            var attempt = 0
            while !Task.isCancelled {
                guard let self else { return }
                let connected = await self.consumeEvents()
                if Task.isCancelled { return }
                // A clean end of stream still means the server went away.
                attempt = connected ? 0 : min(attempt + 1, 5)
                let delay = Self.backoff[min(attempt, Self.backoff.count - 1)]
                try? await Task.sleep(nanoseconds: delay)
            }
        }
    }

    /// Seconds to wait before re-opening the event stream.
    private static let backoff: [UInt64] = [
        500_000_000, 1_000_000_000, 2_000_000_000,
        4_000_000_000, 8_000_000_000, 15_000_000_000,
    ]

    /// Reads frames until the stream ends. Returns true if anything arrived,
    /// which is what tells the caller to reset the backoff.
    private func consumeEvents() async -> Bool {
        var received = false
        do {
            let stream = await api.sseStream(path: "api/events")
            for try await frame in stream {
                if Task.isCancelled { break }
                received = true
                isLive = true
                apply(frame)
            }
        } catch {
            // A dropped event stream is normal (server restart, sleep); the
            // banner is reserved for actions the user just asked for.
            isLive = false
            return received
        }
        isLive = false
        return received
    }

    private func apply(_ frame: SSEFrame) {
        guard let event = frame.decode(ServerEvent.self) else { return }
        switch event.type {
        case "runtime":
            if let status = event.runtime { runtime = status }
        case "download":
            if let job = event.job { merge(job) }
        case "library":
            Task { await self.refreshLibrary() }
        default:
            break   // unknown frame types are ignored on purpose
        }
    }

    private func merge(_ job: DownloadJob) {
        if let index = downloads.firstIndex(where: { $0.id == job.id }) {
            downloads[index] = job
        } else {
            downloads.append(job)
        }
        if job.state == .done {
            dismissedJobs.remove(job.id)
            Task { await self.refreshLibrary() }
            // "Zainstaluj" has to turn into "Zainstalowany" by itself.
            reloadCatalog()
        }
    }

    // MARK: - Hub

    public func loadCurated() async {
        guard curated.isEmpty else { return }
        isLoadingCurated = true
        defer { isLoadingCurated = false }
        do {
            curated = try await api.hubCurated()
        } catch {
            report(error)
        }
    }

    /// Runs the search the browser controls currently describe, after a short
    /// pause so typing does not fire a request per keystroke.
    public func scheduleSearch(debounce: Bool = true) {
        searchTask?.cancel()
        let text = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            searchResults = []
            isSearching = false
            return
        }
        isSearching = true
        searchTask = Task { [weak self] in
            if debounce {
                try? await Task.sleep(nanoseconds: 350_000_000)
                if Task.isCancelled { return }
            }
            guard let self else { return }
            await self.runSearch(text)
        }
    }

    private func runSearch(_ text: String) async {
        do {
            let hits = try await api.hubSearch(query: text,
                                               format: formatFilter.apiValue,
                                               sort: sortOrder.apiValue,
                                               limit: 60)
            guard !Task.isCancelled else { return }
            searchResults = hits
            isSearching = false
        } catch {
            guard !Task.isCancelled else { return }
            isSearching = false
            report(error)
        }
    }

    /// Opens (or re-opens) the detail pane for a repository.
    public func selectRepo(_ repo: String?) {
        guard selectedRepo != repo || (repo != nil && detail == nil) else { return }
        selectedRepo = repo
        detail = nil
        detailTask?.cancel()
        guard let repo else { return }
        isLoadingDetail = true
        detailTask = Task { [weak self] in
            guard let self else { return }
            do {
                let value = try await self.api.hubModel(repo: repo)
                guard !Task.isCancelled, self.selectedRepo == repo else { return }
                self.detail = value
                self.isLoadingDetail = false
            } catch {
                guard !Task.isCancelled, self.selectedRepo == repo else { return }
                self.isLoadingDetail = false
                self.report(error)
            }
        }
    }

    /// The models shown in "Odkrywaj" right now.
    public var browseResults: [HubModel] {
        searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? filteredCurated
            : searchResults
    }

    /// The curated list, narrowed by the format control. Curated entries have
    /// no download counters, so the sort control does not apply to them.
    public var filteredCurated: [HubModel] {
        curated.filter { formatFilter.accepts($0.format) }
    }

    /// Curated models grouped into the sections the "Polecane" list shows.
    public var curatedGroups: [CuratedGroup] {
        let grouped = Dictionary(grouping: filteredCurated) { $0.category ?? "other" }
        return CuratedGroup.order.compactMap { category in
            guard let models = grouped[category], !models.isEmpty else { return nil }
            return CuratedGroup(id: category, models: models)
        }
    }

    // MARK: - Catalog

    /// Re-runs the catalog query the controls currently describe.
    ///
    /// Debounced, because the search field calls it on every keystroke and the
    /// first call of a session may go out to Hugging Face.
    public func reloadCatalog(debounce: Bool = false) {
        catalogTask?.cancel()
        let category = catalogCategory
        let text = catalogSearch.trimmingCharacters(in: .whitespacesAndNewlines)
        let installedOnly = catalogInstalledOnly
        let officialOnly = catalogOfficialOnly
        let minTrust = catalogMinTrust
        let sort = catalogSort
        isLoadingCatalog = true
        catalogTask = Task { [weak self] in
            if debounce {
                try? await Task.sleep(nanoseconds: 300_000_000)
                if Task.isCancelled { return }
            }
            guard let self else { return }
            do {
                let page = try await self.api.catalog(category: category,
                                                      query: text.isEmpty ? nil : text,
                                                      installedOnly: installedOnly,
                                                      minTrust: minTrust > 0 ? minTrust : nil,
                                                      officialOnly: officialOnly,
                                                      sort: sort)
                guard !Task.isCancelled else { return }
                self.catalog = page.entries
                self.catalogTotal = page.total
                self.isCatalogOffline = page.isOffline
                self.isLoadingCatalog = false
                await self.refreshPastedEntry(text)
            } catch {
                guard !Task.isCancelled else { return }
                self.isLoadingCatalog = false
                self.report(error)
            }
        }
    }

    /// The row for a repository the person pasted into the search field.
    ///
    /// Anything on Hugging Face can be installed, not only what we catalogued;
    /// paste "org/model" or the model's link and it appears above the results,
    /// marked as coming from outside our catalogue.
    public private(set) var pastedEntry: CatalogEntry?

    /// True when the text looks like a repository id or a Hugging Face link.
    public static func looksLikeRepoReference(_ text: String) -> Bool {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !value.contains(" ") else { return false }
        if value.lowercased().contains("huggingface.co") || value.lowercased().hasPrefix("hf.co") {
            return true
        }
        let parts = value.split(separator: "/")
        return parts.count == 2 && parts.allSatisfy { !$0.isEmpty }
    }

    private func refreshPastedEntry(_ text: String) async {
        guard Self.looksLikeRepoReference(text) else {
            if pastedEntry != nil { pastedEntry = nil }
            return
        }
        let found = try? await api.catalogLookup(ref: text)
        guard !Task.isCancelled else { return }
        // Only worth a card of its own when the listing does not already have it.
        pastedEntry = (found?.external == true) ? found : nil
    }

    /// Whether the first-run download has already been considered this launch.
    @ObservationIgnored private var didOfferStarter = false

    /// On a fresh install, fetch one small model without being asked.
    ///
    /// An empty library means the app cannot answer a single question, and
    /// picking a first model out of six thousand is not a good first task.
    /// The download is visible and cancellable like any other, and the whole
    /// behaviour is one setting away from off.
    public func installStarterModelIfNeeded() {
        guard !didOfferStarter else { return }
        guard settings.store.autoInstallStarter,
              let starter = settings.store.starterModel, !starter.isEmpty,
              library.isEmpty, downloads.isEmpty else { return }
        didOfferStarter = true
        Task {
            do {
                let entry = try await api.catalogLookup(ref: starter)
                announce("Pobieram pierwszy model — „\(entry.name)”, żeby było czym zacząć.")
                await install(entry)
            } catch {
                // A fresh install without a network is not an error worth a
                // banner: the store explains itself well enough.
                didOfferStarter = true
            }
        }
    }

    /// Whether the assistant may manage models by itself.
    public var agentManagesModels: Bool { settings.agent.manageModels ?? false }

    public func setAgentManagesModels(_ enabled: Bool) {
        guard enabled != agentManagesModels else { return }
        Task {
            await patchSettings(JSONObject(["agent": .object(["manageModels": .bool(enabled)])]),
                                note: enabled
                                    ? "Asystent może teraz sam pobierać i włączać modele."
                                    : "Asystent nie zarządza już modelami.")
        }
    }

    /// Whether the agent may hand a search to a helper with its own context.
    public var subagentsEnabled: Bool { settings.agent.subagents ?? false }

    public func setSubagentsEnabled(_ enabled: Bool) {
        guard enabled != subagentsEnabled else { return }
        Task {
            await patchSettings(JSONObject(["agent": .object(["subagents": .bool(enabled)])]),
                                note: enabled
                                    ? "Agent może teraz zlecać wyszukiwanie pomocnikowi."
                                    : "Agent pracuje sam, bez pomocników.")
        }
    }

    /// Auto mode's reviewer (`agent.autoReviewerModel`); `nil` = the agent's own model.
    public var autoReviewerModel: String? {
        guard let key = settings.agent.autoReviewerModel, !key.isEmpty else { return nil }
        return key
    }

    public func setAutoReviewerModel(_ key: String?) {
        guard key != autoReviewerModel else { return }
        let name = key.map { model(for: $0)?.pickerName ?? $0 }
        Task {
            await patchSettings(JSONObject(["agent": .object(["autoReviewerModel": key.map { .string($0) } ?? .null])]),
                                note: name.map { "Recenzent trybu Auto: \($0)." }
                                    ?? "W trybie Auto akcje ocenia ten sam model, na którym pracuje agent.")
        }
    }

    /// Auto mode's local guard (`agent.autoGuardModel`); `nil` = automatic.
    public var autoGuardModel: String? {
        guard let key = settings.agent.autoGuardModel, !key.isEmpty else { return nil }
        return key
    }

    /// Sets the guard. Local models only — the server refuses a remote key too.
    public func setAutoGuardModel(_ key: String?) {
        guard key != autoGuardModel else { return }
        if let key, RemoteModelKey.isRemote(key) { return }
        let name = key.map { model(for: $0)?.pickerName ?? $0 }
        Task {
            await patchSettings(JSONObject(["agent": .object(["autoGuardModel": key.map { .string($0) } ?? .null])]),
                                note: name.map { "Strażnik lokalny trybu Auto: \($0)." }
                                    ?? "Strażnik lokalny trybu Auto wybierany automatycznie.")
        }
    }

    /// Whether the agent learns from the user's answers on permission cards
    /// (`agent.autoPrecedentsEnabled`); on unless switched off.
    public var autoPrecedentsEnabled: Bool { settings.agent.autoPrecedentsEnabled ?? true }

    public func setAutoPrecedentsEnabled(_ enabled: Bool) {
        guard enabled != autoPrecedentsEnabled else { return }
        Task {
            await patchSettings(JSONObject(["agent": .object(["autoPrecedentsEnabled": .bool(enabled)])]),
                                note: enabled
                                    ? "Recenzent i strażnik uczą się z Twoich decyzji."
                                    : "Twoje decyzje nie są już zapamiętywane ani używane.")
        }
    }

    /// Forgets every learned decision, in every project.
    public func clearAutoPrecedents() async {
        do {
            let removed = try await api.clearAutoPrecedents()
            announce(removed == 0 ? "Nie było wyuczonych decyzji."
                                  : "Zapomniano wyuczone decyzje (\(removed)).")
        } catch {
            report(error)
        }
    }

    /// Tool-use steps the agent may take per message (`agent.maxSteps`); 0 = no limit.
    public var agentMaxSteps: Int { settings.agent.maxSteps ?? 40 }

    public func setAgentMaxSteps(_ steps: Int) async {
        let steps = max(0, steps)
        guard steps != agentMaxSteps else { return }
        await patchSettings(JSONObject(["agent": .object(["maxSteps": .int(steps)])]),
                            note: steps == 0 ? "Agent pracuje bez limitu kroków."
                                             : "Limit kroków agenta: \(steps) na wiadomość.")
    }

    // MARK: - Keeping a model in memory

    /// Models the runtime keeps loaded (``runtime.pinnedModels``).
    public var pinnedModels: [String] { runtime.settings.pinnedModels }

    public func isPinned(_ key: String) -> Bool { pinnedModels.contains(key) }

    /// Pin or unpin one model. A pinned model is never unloaded on idle and is
    /// never evicted to make room — the point is that it answers without a load.
    public func setPinned(_ key: String, _ pinned: Bool) async {
        var keys = pinnedModels
        if pinned {
            guard !keys.contains(key) else { return }
            keys.append(key)
        } else {
            guard keys.contains(key) else { return }
            keys.removeAll { $0 == key }
        }
        await patchRuntime(JSONObject(["pinnedModels": .array(keys.map { .string($0) })]))
        announce(pinned ? "Model zostanie utrzymany w pamięci." : "Model nie jest już trzymany w pamięci.")
    }

    // MARK: - Security (containment and scanning)

    /// `auto` | `strict` | `off` — whether engine processes run in the macOS sandbox.
    public var sandboxMode: String { settings.security.sandbox }

    /// Whether a downloaded model's files are scanned before it may be loaded.
    public var scanDownloads: Bool { settings.security.scanDownloads }

    public func setSandboxMode(_ mode: String) async {
        guard mode != sandboxMode else { return }
        let note: String
        switch mode {
        case "strict": note = "Modele będą uruchamiane tylko w piaskownicy."
        case "off": note = "Piaskownica wyłączona — modele mają zwykłe uprawnienia procesu."
        default: note = "Piaskownica włączona, z awaryjnym uruchomieniem bez niej."
        }
        await patchSettings(JSONObject(["security": .object(["sandbox": .string(mode)])]), note: note)
    }

    public func setScanDownloads(_ enabled: Bool) async {
        guard enabled != scanDownloads else { return }
        await patchSettings(JSONObject(["security": .object(["scanDownloads": .bool(enabled)])]),
                            note: enabled
                                ? "Pobrane modele będą sprawdzane przed pierwszym użyciem."
                                : "Wyłączono sprawdzanie pobranych modeli.")
    }

    // MARK: - Cerber

    public var cerber: CerberAppConfig { settings.cerber }
    /// The last "Sprawdź połączenie" answer, shown under the switch.
    public private(set) var cerberCheck: CerberCheckDTO?
    public private(set) var isCheckingCerber = false

    public func setCerberEnabled(_ enabled: Bool) async {
        guard enabled != cerber.enabled else { return }
        await patchSettings(JSONObject(["cerber": .object(["enabled": .bool(enabled)])]),
                            note: enabled
                                ? "Cerber sprawdzi każdy model przed pobraniem."
                                : "Wyłączono sprawdzanie modeli Cerberem.")
        if enabled { await checkCerber() }
    }

    public func setCerberURL(_ url: String) async {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != cerber.url else { return }
        await patchSettings(JSONObject(["cerber": .object(["url": .string(trimmed)])]),
                            note: "Zapisano adres Cerbera.")
    }

    public func saveCerberToken(_ token: String?) async {
        do {
            settings = try await api.setCerberToken(token)
            announce(settings.cerber.tokenSet ? "Zapisano token Cerbera." : "Usunięto token Cerbera.")
            await checkCerber()
        } catch {
            report(error)
        }
    }

    public func checkCerber() async {
        isCheckingCerber = true
        defer { isCheckingCerber = false }
        cerberCheck = try? await api.checkCerber()
    }

    /// The cloud providers and whether each has a usable key.
    public private(set) var remoteProviders: [RemoteProviderDTO] = []

    public func refreshRemoteProviders() async {
        remoteProviders = (try? await api.remoteProviders()) ?? remoteProviders
    }

    /// Verify a key with the provider and save it when it works.
    public func saveRemoteKey(_ provider: String, key: String) async {
        do {
            remoteProviders = try await api.setRemoteKey(provider, key: key)
            announce("Klucz zapisany — modele tego dostawcy są już na listach.")
            await refreshLibrary()
        } catch {
            report(error)
        }
    }

    public func setRemoteKey(_ provider: String, key: String?) {
        Task {
            do {
                remoteProviders = try await api.setRemoteKey(provider, key: key)
                announce(key == nil ? "Klucz usunięty." : "Klucz zapisany.")
                await refreshLibrary()
            } catch {
                report(error)
            }
        }
    }

    public func setRemoteEnabled(_ provider: String, _ enabled: Bool) {
        Task {
            do {
                remoteProviders = try await api.setRemoteEnabled(provider, enabled: enabled)
                await refreshLibrary()
            } catch {
                report(error)
            }
        }
    }

    /// Standard / Flex / Priority for one provider. Shown at once; the server's
    /// answer (or a refresh after an error) puts the real value back.
    public func setRemoteServiceTier(_ provider: String, _ tier: String) {
        if let index = remoteProviders.firstIndex(where: { $0.id == provider }) {
            remoteProviders[index].serviceTier = tier
        }
        Task {
            do {
                remoteProviders = try await api.setRemoteServiceTier(provider, tier: tier)
            } catch {
                report(error)
                await refreshRemoteProviders()
            }
        }
    }

    /// True while a model from disk is being copied into the library.
    public private(set) var isImporting = false

    /// Add a model the person already has: a `.gguf` file or an MLX directory.
    public func importModel(at path: String, move: Bool) async {
        guard !isImporting else { return }
        isImporting = true
        defer { isImporting = false }
        do {
            let model = try await api.importModel(path: path, move: move)
            await refreshLibrary()
            announce("Dodano „\(model.displayName)” do biblioteki.")
        } catch {
            report(error)
        }
    }

    /// The bytes of a publisher's logo, or `nil` when the server has none.
    /// A missing logo is a normal answer here, never an error banner.
    public func publisherLogoData(_ slug: String) async -> Data? {
        try? await api.publisherLogo(slug: slug)
    }

    /// Opens the "Szczegóły" sheet. The row is shown immediately and replaced
    /// by the full entry (every variant, every file) when it arrives.
    public func openCatalogDetail(_ entry: CatalogEntry) {
        catalogDetail = entry
        catalogDetailTask?.cancel()
        // Always refetch: a list row carries only the recommended variant, so
        // trusting a non-empty `variants` here would show one quantisation and
        // hide the other six the repository actually offers.
        isLoadingCatalogDetail = true
        catalogDetailTask = Task { [weak self] in
            guard let self else { return }
            do {
                let full = try await self.api.catalogEntry(id: entry.id)
                guard !Task.isCancelled, self.catalogDetail?.id == entry.id else { return }
                self.catalogDetail = entry.merging(detail: full)
                self.isLoadingCatalogDetail = false
            } catch {
                guard !Task.isCancelled else { return }
                self.isLoadingCatalogDetail = false
                // The row on screen is already enough to act on, so a failed
                // detail fetch is not worth a banner over the whole screen.
            }
        }
    }

    public func closeCatalogDetail() {
        catalogDetailTask?.cancel()
        catalogDetail = nil
        isLoadingCatalogDetail = false
    }

    /// The single "Zainstaluj" action. The server picks the variant that suits
    /// this Mac, so the app deliberately sends nothing but the id.
    public func install(_ entry: CatalogEntry) async {
        guard !installingIds.contains(entry.id) else { return }
        installingIds.insert(entry.id)
        defer { installingIds.remove(entry.id) }
        do {
            let job = try await api.installFromCatalog(id: entry.id)
            merge(job)
            announce("Instaluję „\(entry.name)” — postęp widać na kafelku modelu.")
        } catch {
            report(error)
        }
    }

    public func isInstalling(_ entry: CatalogEntry) -> Bool {
        if installingIds.contains(entry.id) { return true }
        // Once the job exists it is the queue, not this set, that knows.
        return job(for: entry) != nil
    }

    /// Every quantisation a store entry offers, each with its memory verdict —
    /// what the "Zainstaluj" chooser lists (`GET /api/catalog/{id}`).
    public func catalogVariants(for entry: CatalogEntry) async throws -> CatalogEntry {
        try await api.catalogEntry(id: entry.id)
    }

    /// Install one named variant (the "Zainstaluj" chooser and Szczegóły).
    public func install(_ entry: CatalogEntry, variant: CatalogVariant) async {
        guard !installingIds.contains(entry.id) else { return }
        installingIds.insert(entry.id)
        defer { installingIds.remove(entry.id) }
        do {
            _ = try await api.installFromCatalog(id: entry.id, variantId: variant.id)
            announce("Pobieram „\(entry.name)” w wersji \(variant.label).")
            await refreshDownloads()
        } catch {
            report(error)
        }
    }

    /// The running download for a store row, so its card can show the progress
    /// where the person clicked, instead of only in the toolbar ring.
    public func job(for entry: CatalogEntry) -> DownloadJob? {
        downloads.first { $0.state.isActive && $0.repo == entry.source.repo }
    }

    /// The categories the store offers, from `app-settings.catalog`.
    public var catalogCategories: [CatalogCategory] { settings.catalog.categories }
    public var catalogSorts: [CatalogSortOption] { settings.catalog.sorts }
    public var catalogQualities: [CatalogSortOption] { settings.catalog.qualities }

    /// "Modele pobierane z Hugging Face" — the footer under the list.
    public var catalogSourceNote: String { settings.catalog.sourceNote }

    public var catalogOfflineNote: String { settings.catalog.offlineNote }

    /// Says why the app just jumped to Ustawienia instead of downloading.
    public func announceTokenNeeded(for name: String) {
        announce("„\(name)” wymaga tokenu Hugging Face — wklej go poniżej i spróbuj ponownie.")
    }

    // MARK: - Downloads

    public func download(repo: String, variantId: String?) async {
        do {
            let job = try await api.startDownload(repo: repo, variantId: variantId)
            merge(job)
            announce("Pobieranie „\(job.label)” rozpoczęte.")
        } catch {
            report(error)
        }
    }

    public func cancelDownload(_ id: String) async {
        do {
            try await api.cancelDownload(id: id)
            await refreshDownloads()
        } catch {
            report(error)
        }
    }

    /// Hides a finished or failed job from the popover without touching the server.
    public func dismissJob(_ id: String) {
        dismissedJobs.insert(id)
        downloads.removeAll { $0.id == id }
    }

    /// Jobs still worth showing: everything running plus what finished recently.
    public var visibleJobs: [DownloadJob] {
        downloads
            .filter { !dismissedJobs.contains($0.id) }
            .sorted { ($0.createdAt ?? 0) > ($1.createdAt ?? 0) }
    }

    public var activeJobs: [DownloadJob] {
        downloads.filter { $0.state.isActive }
    }

    /// Combined progress of every running job, for the toolbar ring.
    public var activeProgress: Double? {
        let jobs = activeJobs
        guard !jobs.isEmpty else { return nil }
        let total = jobs.reduce(Int64(0)) { $0 + max($1.totalBytes, 0) }
        guard total > 0 else { return 0 }
        let done = jobs.reduce(Int64(0)) { $0 + max($1.doneBytes, 0) }
        return min(1, Double(done) / Double(total))
    }

    // MARK: - Runtime

    public func load(key: String, config: LoadConfigDTO? = nil, newInstance: Bool = false) async {
        busyKeys.insert(key)
        defer { busyKeys.remove(key) }
        do {
            _ = try await api.loadModel(key: key, config: config, newInstance: newInstance)
            await refreshRuntime()
        } catch {
            report(error)
        }
    }

    public func unload(instanceId: String) async {
        let key = runtime.instances.first { $0.id == instanceId }?.modelKey
        if let key { busyKeys.insert(key) }
        defer { if let key { busyKeys.remove(key) } }
        do {
            runtime = try await api.unloadInstance(id: instanceId)
        } catch {
            report(error)
        }
    }

    /// Unloads every instance of a model — what the eject button in the bar does.
    public func unload(key: String) async {
        for instance in runtime.instances where instance.modelKey == key {
            await unload(instanceId: instance.id)
        }
    }

    public func unloadAll() async {
        do {
            runtime = try await api.unloadAllInstances()
            announce("Zwolniono wszystkie modele.")
        } catch {
            report(error)
        }
    }

    /// One memory estimate. Returns `nil` instead of raising a banner: the
    /// load sheet asks for this on every slider tick.
    public func estimate(key: String, config: LoadConfigDTO) async -> MemoryEstimate? {
        try? await api.estimateMemory(key: key, config: config)
    }

    /// A partial `PATCH /api/runtime/settings`.
    public func patchRuntime(_ patch: JSONObject) async {
        do {
            runtime = try await api.patchRuntimeSettings(patch)
        } catch {
            report(error)
        }
    }

    // MARK: - Library actions

    public func saveLoadDefaults(key: String, config: LoadConfigDTO?) async {
        do {
            let updated = try await api.setLoadDefaults(key: key, defaults: config)
            replace(updated)
            announce("Zapisano ustawienia ładowania.")
        } catch {
            report(error)
        }
    }

    public func deleteModel(key: String) async {
        do {
            try await api.deleteLocalModel(key: key)
            library.removeAll { $0.key == key }
            announce("Model usunięty z dysku.")
        } catch {
            report(error)
        }
    }

    private func replace(_ model: LocalModelDTO) {
        if let index = library.firstIndex(where: { $0.key == model.key }) {
            library[index] = model
        }
    }

    /// Reveals the model's file (or directory) in Finder.
    public func revealInFinder(_ model: LocalModelDTO) {
        guard !model.path.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
    }

    // MARK: - Application settings

    public func setDefaultModel(_ key: String?, for purpose: ModelPurpose) async {
        let section = purpose == .chat ? "chat" : "agent"
        let patch = JSONObject([section: .object(["defaultModel": .stringOrNull(key)])])
        await patchSettings(patch, note: key == nil
                            ? "Wyczyszczono domyślny model (\(purpose.shortTitle))."
                            : "Ustawiono domyślny model (\(purpose.shortTitle)).")
    }

    /// `app-settings.{chat,agent}.reasoningEffort`.
    public func setReasoningEffort(_ level: String, for purpose: ModelPurpose) async {
        let section = purpose == .chat ? "chat" : "agent"
        let patch = JSONObject([section: .object(["reasoningEffort": .string(level)])])
        await patchSettings(patch, note: "Zmieniono domyślny poziom myślenia (\(purpose.shortTitle)).")
    }

    public func setEmbeddingModel(_ key: String?) async {
        let patch = JSONObject(["chat": .object(["embeddingModel": .stringOrNull(key)])])
        await patchSettings(patch, note: key == nil
                            ? "Wyłączono model embeddingów."
                            : "Ustawiono model embeddingów.")
    }

    public func setModelsDirectory(_ path: String) async {
        await patchSettings(JSONObject(["modelsDir": .string(path)]),
                            note: "Zmieniono katalog modeli.")
        await refreshLibrary()
    }

    public func setOpenAIServer(enabled: Bool) async {
        await patchSettings(JSONObject(["server": .object(["openaiApiEnabled": .bool(enabled)])]),
                            note: enabled ? "Serwer zgodny z OpenAI włączony."
                                          : "Serwer zgodny z OpenAI wyłączony.")
    }

    public func setHuggingFaceToken(_ token: String?) async {
        do {
            settings = try await api.setHuggingFaceToken(token)
            announce(settings.hfTokenSet ? "Zapisano token Hugging Face." : "Usunięto token Hugging Face.")
            // A token can unlock a gated repository that is on screen.
            if let repo = selectedRepo {
                let current = repo
                selectedRepo = nil
                selectRepo(current)
            }
        } catch {
            report(error)
        }
    }

    /// Which quantisation level new downloads use (`settings.store.quality`).
    public var storeQuality: String { settings.store.quality }
    public var storePreferredFormat: String { settings.store.preferFormat }

    /// Change the level every later install downloads at.
    public func setStoreQuality(_ quality: String) {
        guard quality != settings.store.quality else { return }
        Task {
            await patchSettings(JSONObject(["store": .object(["quality": .string(quality)])]),
                                note: "Nowe pobierania w jakości: \(quality).")
            reloadCatalog()
        }
    }

    /// Which packaging wins when a model exists as both MLX and GGUF.
    public func setStorePreferredFormat(_ format: String) {
        guard format != settings.store.preferFormat else { return }
        Task {
            await patchSettings(JSONObject(["store": .object(["preferFormat": .string(format)])]),
                                note: format == "mlx" ? "Preferuję wersje MLX."
                                    : (format == "gguf" ? "Preferuję wersje GGUF."
                                                        : "Bez preferencji formatu."))
            reloadCatalog()
        }
    }

    private func patchSettings(_ patch: JSONObject, note: String) async {
        do {
            settings = try await api.patchAppSettings(patch)
            announce(note)
        } catch {
            report(error)
        }
    }

    // MARK: - Derived helpers

    public func model(for key: String?) -> LocalModelDTO? {
        guard let key else { return nil }
        return library.first { $0.key == key }
    }

    /// Instances of one model, whatever their state.
    public func instances(of key: String) -> [RuntimeInstance] {
        runtime.instances.filter { $0.modelKey == key }
    }

    public func isLoaded(_ key: String) -> Bool {
        runtime.instances.contains { $0.modelKey == key && $0.state != .error }
    }

    public func isLoading(_ key: String) -> Bool {
        busyKeys.contains(key)
            || runtime.instances.contains { $0.modelKey == key && $0.state == .loading }
    }

    /// Everything that can answer a prompt (i.e. not an embedding model).
    public var chatModels: [LocalModelDTO] {
        library.filter { !$0.isEmbedding }
    }

    public var embeddingModels: [LocalModelDTO] {
        library.filter(\.isEmbedding)
    }

    public func defaultModelKey(for purpose: ModelPurpose) -> String? {
        switch purpose {
        case .chat: settings.chat.defaultModel
        case .agent: settings.agent.defaultModel
        }
    }

    /// "Moje modele", narrowed by the filter field and sorted by name, "Ulubione" first.
    public var filteredLibrary: [LocalModelDTO] {
        let needle = libraryFilter.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let models = needle.isEmpty ? library : library.filter { model in
            model.displayName.lowercased().contains(needle)
                || model.key.lowercased().contains(needle)
                || (model.arch ?? "").lowercased().contains(needle)
                || (model.quant ?? "").lowercased().contains(needle)
        }
        let byName = models.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
        return FavoriteModels.shared.ordered(byName, key: \.key)
    }

    /// Where the server this app talks to really listens — the app starts its
    /// own on a free port, so `settings.server.port` is only right for a
    /// server started with `localcode serve`.
    public private(set) var serverBaseURL: URL?

    /// The OpenAI-compatible base URL the runtime panel offers to copy.
    public var openAIBaseURL: String {
        if let url = serverBaseURL,
           var components = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            components.path = "/v1"
            components.query = nil
            if let text = components.string { return text }
        }
        let port = settings.server.port ?? 8421
        return "http://127.0.0.1:\(port)/v1"
    }

    // MARK: - Messages

    /// Turns any error into the Polish sentence the banner shows.
    private func report(_ error: Error) {
        if error is CancellationError { return }
        errorMessage = APIError.describe(error)
    }

    private func announce(_ text: String) {
        statusMessage = text
        statusTask?.cancel()
        statusTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            guard !Task.isCancelled else { return }
            self?.statusMessage = nil
        }
    }
}

// MARK: - Browser controls

/// The "Wszystkie / GGUF / MLX" control above the search results.
public enum HubFormatFilter: String, CaseIterable, Identifiable, Sendable {
    case all, gguf, mlx

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .all: "Wszystkie"
        case .gguf: "GGUF"
        case .mlx: "MLX"
        }
    }

    public var apiValue: String { self == .all ? "any" : rawValue }

    public func accepts(_ format: String) -> Bool {
        self == .all || format.lowercased() == rawValue
    }
}

/// The sort menu of "Odkrywaj".
public enum HubSortOrder: String, CaseIterable, Identifiable, Sendable {
    case trending, downloads, likes, recent

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .trending: "Popularne teraz"
        case .downloads: "Pobrania"
        case .likes: "Polubienia"
        case .recent: "Najnowsze"
        }
    }

    public var systemImage: String {
        switch self {
        case .trending: "flame"
        case .downloads: "arrow.down.circle"
        case .likes: "heart"
        case .recent: "clock"
        }
    }

    public var apiValue: String { rawValue }
}

/// One section of the "Polecane" list.
public struct CuratedGroup: Identifiable, Sendable {
    public let id: String
    public let models: [HubModel]

    static let order = ["code", "chat", "reasoning", "vision", "embedding", "other"]

    public var title: String {
        switch id {
        case "code": "Do kodowania"
        case "chat": "Do rozmowy"
        case "reasoning": "Rozumowanie"
        case "vision": "Wizja"
        case "embedding": "Embeddingi"
        default: "Inne"
        }
    }

    public var systemImage: String {
        switch id {
        case "code": "chevron.left.forwardslash.chevron.right"
        case "chat": "bubble.left.and.bubble.right"
        case "reasoning": "brain"
        case "vision": "eye"
        case "embedding": "point.3.connected.trianglepath.dotted"
        default: "square.grid.2x2"
        }
    }
}
