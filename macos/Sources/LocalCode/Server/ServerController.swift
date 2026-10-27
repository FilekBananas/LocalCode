import Foundation
import AppKit
import LocalCodeKit

/// Owns the connection to the Python core and the window's top-level state.
///
/// On launch it first tries to **attach** to a server that is already running
/// (the terminal app writes `~/.localcode/server.json`), so the TUI and the Mac
/// app share one runtime and a model is never loaded twice. Only if that fails
/// does it **spawn** its own `localcode serve` on an ephemeral port.
///
/// Failure is a first-class state here, not an alert: the server is a separate
/// program that may simply not be installed yet, so the startup screen shows
/// what was tried, the tail of the log, and offers to point at the executable
/// by hand.
@MainActor
@Observable
public final class ServerController {

    public enum Phase: Equatable {
        case connecting
        case ready
        case failed
    }

    public private(set) var phase: Phase = .connecting
    /// One short line under the spinner.
    public private(set) var statusText = "Szukam serwera LocalCode…"
    /// Polish sentence for the failure screen.
    public private(set) var errorMessage: String?
    /// The tail of the server log, shown when it failed to start.
    public private(set) var logLines: [String] = []
    /// True when this app started the server (and therefore must stop it).
    public private(set) var ownsServer = false
    /// Where the running server listens; `nil` until it does.
    public private(set) var connection: ServerConnection?
    /// Server version from `/api/health`, for the settings page.
    public private(set) var serverVersion: String?
    /// Which executable was used, for the settings page.
    public private(set) var executablePath: String?

    public let api = APIClient()

    private var launcher: ServerProcessLauncher?
    private var startTask: Task<Void, Never>?

    public init() {}

    // MARK: - Lifecycle

    /// Attach, or spawn. Safe to call again after a failure.
    public func start() {
        startTask?.cancel()
        phase = .connecting
        errorMessage = nil
        logLines = []
        statusText = "Szukam serwera LocalCode…"

        startTask = Task { [weak self] in
            guard let self else { return }
            await self.connect()
        }
    }

    /// Stops a server this app started; leaves an attached one alone.
    public func shutdown() {
        startTask?.cancel()
        startTask = nil
        guard ownsServer else { return }
        launcher?.terminate()
        launcher = nil
        ownsServer = false
    }

    // MARK: - Connecting

    private func connect() async {
        if let attached = await attachToRunningServer() {
            await finish(with: attached, owns: false, executable: nil)
            return
        }

        statusText = "Uruchamiam serwer LocalCode…"

        guard let executable = ServerExecutable.resolve() else {
            fail(message: """
                 Nie znaleziono programu „localcode”. Zainstaluj środowisko LocalCode \
                 albo wskaż plik wykonywalny ręcznie.
                 """,
                 lines: [])
            return
        }

        let launcher = ServerProcessLauncher()
        self.launcher = launcher
        do {
            let connection = try await launcher.launch(executablePath: executable.path)
            await finish(with: connection, owns: true, executable: executable.path)
        } catch is CancellationError {
            return
        } catch {
            launcher.terminate()
            fail(message: APIError.describe(error), lines: launcher.logLines)
        }
    }

    /// Reads `~/.localcode/server.json` and checks that the server behind it is
    /// both alive and willing to accept its token.
    private func attachToRunningServer() async -> ServerConnection? {
        guard let info = ServerInfoFile.read(), info.port > 0 else { return nil }
        statusText = "Łączę z uruchomionym serwerem…"

        let candidate = ServerConnection(port: info.port, token: info.token, ownsProcess: false)
        await api.configure(candidate)

        guard let health = try? await api.health(), health.ok else {
            await api.configure(.unconfigured)
            return nil
        }
        serverVersion = health.version

        guard await probeAuthenticated() else {
            // The file is stale: something else owns that port, or the token
            // was rotated. Start our own rather than fight over it.
            await api.configure(.unconfigured)
            serverVersion = nil
            return nil
        }
        return candidate
    }

    /// One authenticated call, to prove the token is accepted.
    ///
    /// `/api/app-settings` is the LocalCode-specific endpoint; `/api/settings`
    /// is the chat one. A 404 means the server simply does not have that route,
    /// which is not an authentication problem — only 401/403 is.
    private func probeAuthenticated() async -> Bool {
        var sawMissingRoute = false
        for path in ["api/app-settings", "api/settings"] {
            do {
                let request = await api.makeRequest(path)
                _ = try await api.sendData(request)
                return true
            } catch let error as APIError {
                if error.isUnauthorized { return false }
                if error.status == 404 { sawMissingRoute = true; continue }
                return false
            } catch {
                return false
            }
        }
        // Health answered and nothing rejected the token: good enough.
        return sawMissingRoute
    }

    private func finish(with connection: ServerConnection, owns: Bool, executable: String?) async {
        await api.configure(connection)
        self.connection = connection
        self.ownsServer = owns
        self.executablePath = executable
        if serverVersion == nil {
            serverVersion = (try? await api.health())?.version
        }
        statusText = owns ? "Serwer uruchomiony." : "Połączono z uruchomionym serwerem."
        errorMessage = nil
        logLines = []
        phase = .ready
    }

    private func fail(message: String, lines: [String]) {
        errorMessage = message
        logLines = lines
        statusText = "Nie udało się połączyć."
        phase = .failed
    }

    // MARK: - Manual recovery

    /// Lets the user point at `localcode` by hand and retry.
    public func chooseExecutable() {
        let panel = NSOpenPanel()
        panel.title = "Wskaż program „localcode”"
        panel.message = "Wybierz plik wykonywalny localcode (np. .venv/bin/localcode)."
        panel.prompt = "Wybierz"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.showsHiddenFiles = true
        panel.treatsFilePackagesAsDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard FileManager.default.isExecutableFile(atPath: url.path) else {
            errorMessage = "Wskazany plik nie jest programem wykonywalnym."
            return
        }
        UserDefaults.standard.set(url.path, forKey: ServerExecutable.userDefaultsKey)
        start()
    }

    /// Forgets a hand-picked path so discovery starts from scratch again.
    public func forgetChosenExecutable() {
        UserDefaults.standard.removeObject(forKey: ServerExecutable.userDefaultsKey)
    }

    public var chosenExecutable: String? {
        UserDefaults.standard.string(forKey: ServerExecutable.userDefaultsKey)
    }

    /// Where discovery looked, for the failure screen — named locations one per
    /// line, `PATH` collapsed into one.
    public var searchedPaths: [String] {
        ServerExecutable.searchedSummary()
    }
}
