import Foundation

/// Where the local LocalCode server listens and how to authenticate with it.
///
/// The app either attaches to a server that is already running (the terminal
/// app writes `~/.localcode/server.json`) or spawns its own on an ephemeral
/// port; either way the result is one of these values, handed to `APIClient`
/// before any feature module makes a request.
public struct ServerConnection: Sendable, Equatable, Codable {
    public var baseURL: URL
    public var token: String?
    /// Set when this process started the server and therefore owns it.
    public var ownsProcess: Bool

    public init(baseURL: URL, token: String?, ownsProcess: Bool = false) {
        self.baseURL = baseURL
        self.token = token
        self.ownsProcess = ownsProcess
    }

    /// `http://127.0.0.1:<port>`
    public init(port: Int, token: String?, ownsProcess: Bool = false) {
        self.baseURL = URL(string: "http://127.0.0.1:\(port)")!
        self.token = token
        self.ownsProcess = ownsProcess
    }

    /// Placeholder used before the server answers; requests against it fail
    /// fast with a connection error rather than crashing on a nil URL.
    public static let unconfigured = ServerConnection(port: 0, token: nil)

    public var isConfigured: Bool { baseURL.port != 0 }

    /// `ws://127.0.0.1:<port>/<path>` for the agent WebSocket.
    public func webSocketURL(path: String) -> URL {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        components?.scheme = baseURL.scheme == "https" ? "wss" : "ws"
        components?.path = "/" + path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return components?.url ?? baseURL
    }
}

/// The contents of `~/.localcode/server.json`, written by `localcode serve`.
public struct ServerInfoFile: Codable, Sendable {
    public let port: Int
    public let token: String?
    public let pid: Int?

    public init(port: Int, token: String?, pid: Int?) {
        self.port = port
        self.token = token
        self.pid = pid
    }

    /// `~/.localcode/server.json`, honouring `LOCALCODE_HOME` like the Python side.
    public static var url: URL {
        let raw = ProcessInfo.processInfo.environment["LOCALCODE_HOME"] ?? "~/.localcode"
        return URL(fileURLWithPath: (raw as NSString).expandingTildeInPath)
            .appendingPathComponent("server.json")
    }

    /// Reads the file, or `nil` when it is missing or unreadable.
    public static func read() -> ServerInfoFile? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ServerInfoFile.self, from: data)
    }
}
