import Foundation
import AppKit
import LocalCodeKit

/// Finding, launching and watching the Python side of LocalCode.
///
/// The app never talks to models itself: everything goes through one
/// `localcode serve` process that owns the runtime, the model library and the
/// chat database. Two ways to reach it:
///
///  * **attach** — the terminal app writes `~/.localcode/server.json` while it
///    runs, so if a `localcode` TUI is open the Mac app joins the very same
///    process. That matters more than it sounds: a 4-bit 8 B model is several
///    gigabytes of unified memory, and loading it twice on a 32 GB machine is
///    the difference between working and swapping.
///  * **spawn** — otherwise start a private server on an ephemeral port with a
///    fresh token, and kill it when the app quits.

// MARK: - Executable discovery

/// Where the `localcode` executable might be, in the order they are tried.
enum ServerExecutable {
    /// A path the user picked by hand (save panel) wins over everything.
    static let userDefaultsKey = "serverPath"

    /// Candidates, most specific first. `reason` is shown in the error screen.
    static func candidates() -> [(path: String, reason: String)] {
        var result: [(String, String)] = []

        if let custom = UserDefaults.standard.string(forKey: userDefaultsKey), !custom.isEmpty {
            result.append((custom, "ustawienie „serverPath”"))
        }
        if let bundled = Bundle.main.object(forInfoDictionaryKey: "LCServerPath") as? String, !bundled.isEmpty {
            result.append((bundled, "Info.plist (LCServerPath)"))
        }
        result.append((NSString(string: "~/.localcode/venv/bin/localcode").expandingTildeInPath,
                       "środowisko LocalCode (~/.localcode/venv)"))
        for repository in repositoryVenvs() {
            result.append((repository, "repozytorium projektu (.venv)"))
        }
        for directory in searchPath() {
            result.append(((directory as NSString).appendingPathComponent("localcode"), "PATH"))
        }
        return result.map { (path: $0.0, reason: $0.1) }
    }

    /// A development build lives at `<repo>/macos/dist/LocalCode.app`, so the
    /// checkout's own `.venv` is found by walking up from the bundle — a copy in
    /// `/Applications` simply finds nothing here and falls through to `PATH`.
    private static func repositoryVenvs() -> [String] {
        let manager = FileManager.default
        var url = Bundle.main.bundleURL
        var found: [String] = []
        for _ in 0..<4 {
            url.deleteLastPathComponent()
            let candidate = url.appendingPathComponent(".venv/bin/localcode").path
            if manager.isExecutableFile(atPath: candidate) {
                found.append(candidate)
            }
        }
        return found
    }

    /// A GUI app inherits almost no `PATH`, so the usual install locations are
    /// appended by hand.
    private static func searchPath() -> [String] {
        let fromEnvironment = (ProcessInfo.processInfo.environment["PATH"] ?? "")
            .split(separator: ":")
            .map(String.init)
        let common = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
                      NSString(string: "~/.local/bin").expandingTildeInPath]
        var seen = Set<String>()
        return (fromEnvironment + common).filter { seen.insert($0).inserted }
    }

    /// The first candidate that exists and is executable.
    static func resolve() -> (path: String, reason: String)? {
        let manager = FileManager.default
        for candidate in candidates() where manager.isExecutableFile(atPath: candidate.path) {
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue else { continue }
            return candidate
        }
        return nil
    }

    /// What the error screen lists: the named locations one per line, and the
    /// `PATH` directories collapsed into a single line. Printing all twenty-odd
    /// `PATH` entries buries the two that anyone would actually fix.
    static func searchedSummary() -> [String] {
        let all = candidates()
        var lines = all.filter { $0.reason != "PATH" }.map { "\($0.path)  —  \($0.reason)" }
        let fromPath = all.filter { $0.reason == "PATH" }.count
        if fromPath > 0 {
            lines.append("…oraz \(Format.plural(fromPath, "katalog", "katalogi", "katalogów")) z PATH")
        }
        return lines
    }
}

// MARK: - Output collector

/// Collects a child process's stdout/stderr from the pipe's background queue.
///
/// `@unchecked Sendable` with an explicit lock: `readabilityHandler` runs on a
/// queue of URLSession's choosing, and the UI polls from the main actor.
final class ServerOutputCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()
    private var lines: [String] = []
    private var foundPort: Int?

    private static let maxLines = 100

    /// The port from `LOCALCODE_READY port=<p>`, once it has been printed.
    var readyPort: Int? {
        lock.lock(); defer { lock.unlock() }
        return foundPort
    }

    /// The tail of the log, for the error screen.
    var recentLines: [String] {
        lock.lock(); defer { lock.unlock() }
        return lines
    }

    func append(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        buffer.append(data)
        while let newline = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[buffer.startIndex..<newline]
            buffer.removeSubrange(buffer.startIndex...newline)
            record(String(decoding: lineData, as: UTF8.self))
        }
        // A server that dies before its first newline still leaves a message.
        if buffer.count > 8192 {
            record(String(decoding: buffer, as: UTF8.self))
            buffer.removeAll()
        }
    }

    /// Flushes whatever never ended in a newline.
    func finish() {
        lock.lock(); defer { lock.unlock() }
        guard !buffer.isEmpty else { return }
        record(String(decoding: buffer, as: UTF8.self))
        buffer.removeAll()
    }

    /// Caller already holds the lock.
    private func record(_ raw: String) {
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty else { return }
        lines.append(line)
        if lines.count > Self.maxLines { lines.removeFirst(lines.count - Self.maxLines) }
        if foundPort == nil, let port = Self.parseReady(line) { foundPort = port }
    }

    /// `LOCALCODE_READY port=54321`
    static func parseReady(_ line: String) -> Int? {
        guard line.contains("LOCALCODE_READY") else { return nil }
        guard let range = line.range(of: "port=") else { return nil }
        let digits = line[range.upperBound...].prefix { $0.isNumber }
        guard !digits.isEmpty, let port = Int(digits), port > 0 else { return nil }
        return port
    }
}

/// One bit, written by `Process.terminationHandler` on a private queue and read
/// from the launch loop.
final class ProcessExitFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var isSet: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }

    func mark() {
        lock.lock(); defer { lock.unlock() }
        value = true
    }
}

// MARK: - Launcher

/// Starts `localcode serve` and waits for its readiness line.
final class ServerProcessLauncher {
    enum LaunchError: LocalizedError {
        case notFound
        case cannotLaunch(String)
        case exited(code: Int32)
        case timedOut

        var errorDescription: String? {
            switch self {
            case .notFound:
                return "Nie znaleziono programu „localcode”."
            case let .cannotLaunch(reason):
                return "Nie udało się uruchomić serwera LocalCode: \(reason)"
            case let .exited(code):
                return "Serwer LocalCode zakończył się z kodem \(code) zanim zdążył wystartować."
            case .timedOut:
                return "Serwer LocalCode nie zgłosił gotowości w ciągu 90 sekund."
            }
        }
    }

    private(set) var process: Process?
    private let collector = ServerOutputCollector()

    /// The tail of the server log, for the error screen.
    var logLines: [String] { collector.recentLines }

    /// Launches the server and returns the connection once it is listening.
    ///
    /// The child is told our pid so it exits if this app is force-quit, and it
    /// is terminated explicitly on a normal quit — belt and braces, because a
    /// stray server would keep several GB of model weights resident.
    func launch(executablePath: String) async throws -> ServerConnection {
        guard FileManager.default.isExecutableFile(atPath: executablePath) else {
            throw LaunchError.notFound
        }

        let token = UUID().uuidString
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executablePath)
        process.arguments = ["serve",
                             "--port", "0",
                             "--token", token,
                             "--parent-pid", String(ProcessInfo.processInfo.processIdentifier)]

        var environment = ProcessInfo.processInfo.environment
        // Without this the readiness line sits in Python's block buffer,
        // because stdout is a pipe rather than a terminal.
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        process.standardInput = FileHandle.nullDevice

        let collector = self.collector
        // An empty read means the write end closed. Without clearing the
        // handler there, the queue spins on EOF forever — which is exactly the
        // case for a server that dies on startup, the one case that has to work.
        let reader: @Sendable (FileHandle) -> Void = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            collector.append(data)
        }
        stdout.fileHandleForReading.readabilityHandler = reader
        stderr.fileHandleForReading.readabilityHandler = reader

        let exited = ProcessExitFlag()
        process.terminationHandler = { _ in exited.mark() }

        do {
            try process.run()
        } catch {
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            throw LaunchError.cannotLaunch(error.localizedDescription)
        }
        self.process = process

        // Poll rather than block: the readiness line, an early exit and the
        // timeout are three different outcomes and all three must be reachable.
        let deadline = Date().addingTimeInterval(90)
        while Date() < deadline {
            if let port = collector.readyPort {
                return ServerConnection(port: port, token: token, ownsProcess: true)
            }
            if exited.isSet || !process.isRunning {
                // Give the pipes a moment to deliver the last of the traceback.
                try? await Task.sleep(nanoseconds: 200_000_000)
                collector.finish()
                if let port = collector.readyPort {
                    return ServerConnection(port: port, token: token, ownsProcess: true)
                }
                throw LaunchError.exited(code: process.terminationStatus)
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
            if Task.isCancelled {
                terminate()
                throw CancellationError()
            }
        }
        terminate()
        throw LaunchError.timedOut
    }

    /// Stops the server this app started. Safe to call more than once.
    func terminate() {
        guard let process, process.isRunning else { return }
        process.terminate()
        // Give aiohttp a moment to close its sockets, then insist.
        let deadline = Date().addingTimeInterval(3)
        while process.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
        (process.standardOutput as? Pipe)?.fileHandleForReading.readabilityHandler = nil
        (process.standardError as? Pipe)?.fileHandleForReading.readabilityHandler = nil
    }
}
