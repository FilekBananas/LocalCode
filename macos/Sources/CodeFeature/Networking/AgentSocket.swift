import Foundation
import LocalCodeKit

/// The live half of the agent protocol: one WebSocket per open session.
///
/// The socket is the only way a turn is driven — REST is used for the session
/// list and for the stored transcript, everything else (messages, permissions,
/// interrupt, mode, model, compaction) travels here.
///
/// Three things matter for a local server that can be restarted under the app:
///
/// * **Reconnect.** A dropped connection is retried with exponential backoff
///   (0.5 s → 15 s). The server replays `session` + `history` on every connect,
///   so a reconnect restores the transcript by itself.
/// * **Keepalive.** A ping every 20 s notices a server that died without
///   closing the socket, instead of leaving the UI waiting forever.
/// * **An outbox.** Messages sent while the socket is still connecting (the
///   very common "type, then hit Enter immediately" case) are queued and
///   flushed once it opens, instead of being dropped.
@MainActor
final class AgentSocket {

    typealias State = AgentConnectionState

    private(set) var state: State = .idle {
        didSet {
            guard state != oldValue else { return }
            onStateChange?(state)
        }
    }

    /// Called for every decoded frame, on the main actor.
    var onEvent: ((AgentEvent) -> Void)?
    var onStateChange: ((State) -> Void)?

    /// Why the last connection died — shown once retrying stops being quiet.
    private(set) var lastError: String?

    private let api: APIClient
    private var sessionId: String?
    private var task: URLSessionWebSocketTask?
    private var receiveLoop: Task<Void, Never>?
    private var pingLoop: Task<Void, Never>?
    private var connectTask: Task<Void, Never>?
    private var outbox: [AgentClientMessage] = []
    private var attempt = 0
    private var stopped = true

    private let maxBackoff: Double = 15
    private let pingInterval: UInt64 = 20_000_000_000
    /// After this many failed attempts the session is treated as gone.
    ///
    /// A session deleted on disk answers 404 for ever, so retrying it is a loop
    /// with no exit: the header said "Ponawiam połączenie… (próba 97)" and the
    /// composer stayed disabled until the app was restarted. Eight attempts span
    /// about a minute of backoff, which is long enough to ride out a server that
    /// is merely restarting.
    private let maxAttempts = 8

    init(api: APIClient) {
        self.api = api
    }

    // MARK: - Lifecycle

    /// Connects to `sessionId`, replacing any session already attached.
    func connect(sessionId: String) {
        if self.sessionId == sessionId, state == .connected || state == .connecting { return }
        teardown()
        self.sessionId = sessionId
        stopped = false
        attempt = 0
        openSocket()
    }

    /// Closes the socket and stops reconnecting.
    func disconnect() {
        stopped = true
        sessionId = nil
        outbox.removeAll()
        teardown()
        state = .idle
    }

    // MARK: - Sending

    func send(_ message: AgentClientMessage) {
        guard let task, state == .connected else {
            outbox.append(message)
            return
        }
        deliver(message, on: task)
    }

    private func deliver(_ message: AgentClientMessage, on task: URLSessionWebSocketTask) {
        task.send(.string(message.json)) { [weak self] error in
            guard let error else { return }
            Task { @MainActor in
                self?.handleFailure(error, while: "wysyłanie")
            }
        }
    }

    private func flushOutbox(on task: URLSessionWebSocketTask) {
        let pending = outbox
        outbox.removeAll()
        for message in pending { deliver(message, on: task) }
    }

    // MARK: - Connection

    private func openSocket() {
        guard let sessionId, !stopped else { return }
        state = attempt == 0 ? .connecting : .reconnecting(attempt: attempt)
        connectTask?.cancel()
        connectTask = Task { [weak self] in
            guard let self else { return }
            let socket = await self.api.webSocketTask(path: "api/agent/sessions/\(sessionId)/ws")
            guard !Task.isCancelled, !self.stopped, self.sessionId == sessionId else {
                socket.cancel(with: .goingAway, reason: nil)
                return
            }
            self.task = socket
            socket.resume()
            self.state = .connected
            self.attempt = 0
            self.startReceiving(on: socket)
            self.startPinging(on: socket)
            self.flushOutbox(on: socket)
        }
    }

    private func startReceiving(on socket: URLSessionWebSocketTask) {
        receiveLoop?.cancel()
        receiveLoop = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    let message = try await socket.receive()
                    guard let self, self.task === socket else { return }
                    self.handle(message)
                } catch {
                    guard let self, self.task === socket, !Task.isCancelled else { return }
                    self.handleFailure(error, while: "odbieranie")
                    return
                }
            }
        }
    }

    private func startPinging(on socket: URLSessionWebSocketTask) {
        pingLoop?.cancel()
        pingLoop = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: self?.pingInterval ?? 20_000_000_000)
                guard !Task.isCancelled, let self, self.task === socket else { return }
                socket.sendPing { [weak self] error in
                    guard error != nil else { return }
                    Task { @MainActor in
                        guard let self, self.task === socket else { return }
                        self.handleFailure(APIError(message: "Serwer przestał odpowiadać."),
                                           while: "ping")
                    }
                }
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        let event: AgentEvent?
        switch message {
        case let .string(text):
            event = AgentEventDecoder.decode(text)
        case let .data(data):
            event = AgentEventDecoder.decode(data)
        @unknown default:
            event = nil
        }
        guard let event else { return }
        onEvent?(event)
    }

    // MARK: - Failure and retry

    private func handleFailure(_ error: Error, while phase: String) {
        guard !stopped else { return }
        teardownTasks()
        guard sessionId != nil else {
            state = .idle
            return
        }
        lastError = "\(APIError.describe(error)) (\(phase))"
        attempt += 1
        guard attempt <= maxAttempts else {
            stopped = true
            state = .failed("Nie udało się połączyć z sesją. Otwórz ją ponownie "
                            + "albo zacznij nową.")
            return
        }
        let delay = min(maxBackoff, 0.5 * pow(2, Double(attempt - 1)))
        state = .reconnecting(attempt: attempt)
        connectTask?.cancel()
        connectTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard !Task.isCancelled, let self, !self.stopped else { return }
            self.openSocket()
        }
    }

    private func teardownTasks() {
        receiveLoop?.cancel()
        receiveLoop = nil
        pingLoop?.cancel()
        pingLoop = nil
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    private func teardown() {
        connectTask?.cancel()
        connectTask = nil
        teardownTasks()
    }
}

/// Where the session's WebSocket stands. Published by `AgentStore`, so it is
/// part of the module's public surface.
public enum AgentConnectionState: Equatable, Sendable {
    case idle
    case connecting
    case connected
    /// Waiting before the next attempt; `attempt` counts from 1.
    case reconnecting(attempt: Int)
    case failed(String)

    public var isConnected: Bool { self == .connected }

    /// Short Polish label for the header ("Łączę…", "Ponawiam połączenie…").
    public var label: String? {
        switch self {
        case .idle, .connected: nil
        case .connecting: "Łączę z agentem…"
        case let .reconnecting(attempt): "Ponawiam połączenie… (próba \(attempt))"
        case let .failed(message): message
        }
    }
}
