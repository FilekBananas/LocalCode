import Foundation

/// Thrown for anything the caller may want to show the user verbatim.
///
/// The Python core already produces Polish sentences (`{"error": "...",
/// "code": "..."}`), so `message` goes straight into a `Banner`.
public struct APIError: LocalizedError, Sendable {
    public let message: String
    public let status: Int?
    public let code: String?

    public init(message: String, status: Int? = nil, code: String? = nil) {
        self.message = message
        self.status = status
        self.code = code
    }

    public var errorDescription: String? { message }

    /// The token is wrong — the app has to reconnect to the server.
    public var isUnauthorized: Bool { status == 401 || status == 403 }

    /// Turns any error into a Polish sentence fit for a banner.
    public static func describe(_ error: Error) -> String {
        if let api = error as? APIError { return api.message }
        if error is CancellationError { return "Przerwano." }
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain {
            switch nsError.code {
            case NSURLErrorCannotConnectToHost, NSURLErrorNetworkConnectionLost,
                 NSURLErrorNotConnectedToInternet:
                return "Brak połączenia z lokalnym serwerem LocalCode."
            case NSURLErrorTimedOut:
                return "Serwer nie odpowiedział na czas."
            case NSURLErrorCancelled:
                return "Przerwano."
            default:
                return "Nie udało się połączyć z serwerem LocalCode."
            }
        }
        return error.localizedDescription
    }
}

/// One frame of a `text/event-stream` response.
public struct SSEFrame: Sendable {
    /// The `event:` field, when the server sends one.
    public let event: String?
    /// The joined `data:` lines.
    public let data: String

    public init(event: String?, data: String) {
        self.event = event
        self.data = data
    }

    /// Decodes the payload, returning `nil` for a frame this client does not
    /// understand — a new frame type must never break an older build.
    public func decode<T: Decodable>(_ type: T.Type, using decoder: JSONDecoder = JSONDecoder()) -> T? {
        guard let bytes = data.data(using: .utf8) else { return nil }
        return try? decoder.decode(type, from: bytes)
    }
}

/// REST + SSE + WebSocket client for the local LocalCode server.
///
/// It is an actor because the connection (base URL and bearer token) can change
/// at runtime — the app may attach to a server that was already running, fail,
/// and then spawn its own — and because a chat stream, a download poll and a
/// library refresh all run at once.
///
/// The request plumbing is **public on purpose**. `ChatFeature` uses the chat
/// endpoints declared here; `ModelsFeature` and `CodeFeature` add their own
/// endpoints in extensions of this actor, using `makeRequest`, `send`,
/// `sendVoid`, `encodeBody`, `query` and `sseStream` exactly as the chat
/// methods below do. Nothing in a feature module needs to build a `URLRequest`
/// from scratch or re-implement error mapping.
public actor APIClient {

    // MARK: Connection

    private var connectionValue: ServerConnection

    public init(connection: ServerConnection = .unconfigured) {
        self.connectionValue = connection
    }

    public var connection: ServerConnection { connectionValue }
    public var baseURL: URL { connectionValue.baseURL }
    public var token: String? { connectionValue.token }
    public var isConfigured: Bool { connectionValue.isConfigured }

    public func configure(_ connection: ServerConnection) {
        connectionValue = connection
    }

    public func setToken(_ value: String?) {
        connectionValue.token = value
    }

    // MARK: Session

    private let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        // One answer from a small local model can stream for several minutes.
        config.timeoutIntervalForResource = 3600
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        return URLSession(configuration: config)
    }()

    private let decoder = JSONDecoder()
    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = []
        return encoder
    }()

    // MARK: - Request plumbing (public: feature modules extend this actor)

    /// Builds an authenticated request against the local server.
    ///
    /// `path` is relative and without a leading slash, e.g. `"api/library"`.
    public func makeRequest(_ path: String,
                            method: String = "GET",
                            body: Data? = nil,
                            contentType: String? = "application/json",
                            accept: String = "application/json") -> URLRequest {
        var request = URLRequest(url: url(for: path))
        request.httpMethod = method
        request.setValue(accept, forHTTPHeaderField: "Accept")
        if let contentType, body != nil {
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        if let token = connectionValue.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body
        return request
    }

    /// Absolute URL for a relative path, keeping any query string intact.
    /// `URL.appendingPathComponent` would percent-escape the `?`.
    public func url(for path: String) -> URL {
        let trimmed = path.hasPrefix("/") ? String(path.dropFirst()) : path
        return URL(string: trimmed, relativeTo: baseURL)?.absoluteURL
            ?? baseURL.appendingPathComponent(trimmed)
    }

    /// Performs the request and decodes the JSON body.
    @discardableResult
    public func send<T: Decodable>(_ request: URLRequest, as type: T.Type) async throws -> T {
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError(message: "Nie udało się odczytać odpowiedzi serwera.",
                           status: nil,
                           code: "decode_error")
        }
    }

    /// Performs the request and discards the body (but not the status code).
    public func sendVoid(_ request: URLRequest) async throws {
        let (data, response) = try await session.data(for: request)
        try check(data, response)
    }

    /// Performs the request and returns the raw bytes — exports, attachment
    /// images, README markdown.
    public func sendData(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        return data
    }

    /// Performs the request and hands back the body together with one response
    /// header. Needed where the header carries meaning of its own — the model
    /// catalog answers `X-Catalog-Offline: 1` when it could not reach the
    /// network and served what it already had.
    public func sendData(_ request: URLRequest,
                         header: String) async throws -> (data: Data, header: String?) {
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        let value = (response as? HTTPURLResponse)?.value(forHTTPHeaderField: header)
        return (data, value)
    }

    /// The same, for the calls that need more than one header — the catalog
    /// sends both `X-Catalog-Offline` and `X-Catalog-Total`.
    public func sendData(_ request: URLRequest,
                         headers names: [String]) async throws -> (data: Data,
                                                                   headers: [String: String]) {
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        let http = response as? HTTPURLResponse
        var found: [String: String] = [:]
        for name in names {
            if let value = http?.value(forHTTPHeaderField: name) { found[name] = value }
        }
        return (data, found)
    }

    public func encodeBody<T: Encodable>(_ value: T) throws -> Data {
        try encoder.encode(value)
    }

    /// Appends a query string, dropping nil and empty values.
    public nonisolated func query(_ path: String, _ items: [String: String?]) -> String {
        let pairs = items
            .sorted { $0.key < $1.key }
            .compactMap { key, value -> String? in
                guard let value, !value.isEmpty else { return nil }
                let encoded = value.addingPercentEncoding(withAllowedCharacters: .lcQueryAllowed) ?? value
                return "\(key)=\(encoded)"
            }
        return pairs.isEmpty ? path : "\(path)?\(pairs.joined(separator: "&"))"
    }

    /// Percent-escapes one path segment. Model keys contain `/`, so a key in a
    /// URL path must be escaped rather than appended.
    public nonisolated func pathSegment(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: .lcPathSegmentAllowed) ?? value
    }

    private func check(_ data: Data, _ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            let parsed = try? decoder.decode(APIErrorBody.self, from: data)
            throw APIError(message: parsed?.error ?? Self.defaultMessage(for: http.statusCode),
                           status: http.statusCode,
                           code: parsed?.code)
        }
    }

    private nonisolated static func defaultMessage(for status: Int) -> String {
        switch status {
        case 401, 403: return "Serwer odrzucił token dostępu. Uruchom aplikację ponownie."
        case 404: return "Nie znaleziono zasobu na serwerze."
        case 409: return "Operacja koliduje z bieżącym stanem serwera."
        case 413: return "Plik jest za duży."
        case 500...599: return "Serwer LocalCode zgłosił błąd (\(status))."
        default: return "Serwer zwrócił błąd \(status)."
        }
    }

    // MARK: - SSE

    /// Streams a `text/event-stream` response frame by frame.
    ///
    /// Cancelling the consuming task aborts the HTTP request, which the server
    /// treats as "the client went away" — for `/api/chat` that means the
    /// partial answer is stored with status `stopped`.
    ///
    /// Used by `ChatFeature` for `/api/chat` and by `ModelsFeature` for the
    /// `/api/events` firehose (downloads, runtime, library changes).
    public func sseStream(_ request: URLRequest) -> AsyncThrowingStream<SSEFrame, Error> {
        var streamRequest = request
        streamRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        streamRequest.setValue("no-store", forHTTPHeaderField: "Cache-Control")
        // Built here (actor-isolated), consumed outside: the stream closure must
        // not capture the actor or Swift concurrency flags a possible race.
        return Self.makeSSEStream(streamRequest, session: session)
    }

    /// Convenience: an SSE stream for a path, with the bearer token attached.
    public func sseStream(path: String) -> AsyncThrowingStream<SSEFrame, Error> {
        sseStream(makeRequest(path, accept: "text/event-stream"))
    }

    private nonisolated static func makeSSEStream(_ request: URLRequest,
                                                  session: URLSession) -> AsyncThrowingStream<SSEFrame, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: request)
                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        // A stream that fails before its first frame answers with
                        // the ordinary `{"error", "code"}` body (see
                        // `server/routes_chat.py`), and that Polish sentence is
                        // the only one that says *why* — "model not downloaded",
                        // "not a chat model". Read it instead of the status code.
                        var body = Data()
                        for try await byte in bytes where body.count < 64 * 1024 {
                            body.append(byte)
                        }
                        let parsed = try? JSONDecoder().decode(APIErrorBody.self, from: body)
                        throw APIError(message: parsed?.error ?? defaultMessage(for: http.statusCode),
                                       status: http.statusCode,
                                       code: parsed?.code)
                    }
                    var event: String?
                    var payload: [String] = []

                    func flush() {
                        guard !payload.isEmpty else {
                            event = nil
                            return
                        }
                        continuation.yield(SSEFrame(event: event, data: payload.joined(separator: "\n")))
                        payload = []
                        event = nil
                    }

                    func handle(_ line: String) {
                        if line.isEmpty {
                            flush()
                        } else if line.hasPrefix(":") {
                            return                         // comment / keep-alive
                        } else if line.hasPrefix("data:") {
                            var value = String(line.dropFirst(5))
                            if value.hasPrefix(" ") { value.removeFirst() }
                            payload.append(value)
                        } else if line.hasPrefix("event:") {
                            event = String(line.dropFirst(6)).trimmingCharacters(in: .whitespaces)
                        }
                        // `id:` and `retry:` are not used by this server.
                    }

                    // Split the bytes by hand rather than with `bytes.lines`:
                    // `AsyncLineSequence` swallows empty lines, and an empty
                    // line is exactly how `text/event-stream` ends a frame, so
                    // with it no frame is ever dispatched until the connection
                    // closes. (Verified against `GET /api/events`: the raw byte
                    // stream contains the blank lines, `bytes.lines` does not.)
                    var buffer: [UInt8] = []
                    buffer.reserveCapacity(4096)
                    for try await byte in bytes {
                        guard byte != UInt8(ascii: "\n") else {
                            if buffer.last == UInt8(ascii: "\r") { buffer.removeLast() }
                            handle(String(decoding: buffer, as: UTF8.self))
                            buffer.removeAll(keepingCapacity: true)
                            continue
                        }
                        buffer.append(byte)
                    }
                    if !buffer.isEmpty {
                        handle(String(decoding: buffer, as: UTF8.self))
                    }
                    flush()
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    let nsError = error as NSError
                    if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled {
                        continuation.finish()
                    } else {
                        continuation.finish(throwing: error)
                    }
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - WebSocket

    /// An authenticated WebSocket task for `path` (e.g. an agent session).
    ///
    /// The token travels in the `Authorization` header; the server also accepts
    /// `?token=` for clients that cannot set headers.
    public func webSocketTask(path: String) -> URLSessionWebSocketTask {
        var request = URLRequest(url: connectionValue.webSocketURL(path: path))
        if let token = connectionValue.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return session.webSocketTask(with: request)
    }

    // MARK: - Health

    /// `GET /api/health` — the only endpoint that needs no token.
    public struct Health: Codable, Sendable {
        public let ok: Bool
        public let version: String?
    }

    public func health() async throws -> Health {
        var request = makeRequest("api/health")
        request.timeoutInterval = 4
        return try await send(request, as: Health.self)
    }

    // MARK: - Bootstrap, conversations, messages

    public func bootstrap() async throws -> Bootstrap {
        try await send(makeRequest("api/bootstrap"), as: Bootstrap.self)
    }

    public func conversations(folderId: String? = nil) async throws -> [Conversation] {
        try await send(makeRequest(query("api/conversations", ["folderId": folderId])),
                       as: [Conversation].self)
    }

    public func conversation(id: String) async throws -> ConversationDetail {
        try await send(makeRequest("api/conversations/\(pathSegment(id))"), as: ConversationDetail.self)
    }

    public func createConversation(_ input: ConversationInput) async throws -> Conversation {
        try await send(makeRequest("api/conversations", method: "POST", body: try encodeBody(input)),
                       as: Conversation.self)
    }

    public func updateConversation(id: String, _ input: ConversationInput) async throws -> Conversation {
        try await send(makeRequest("api/conversations/\(pathSegment(id))",
                                   method: "PATCH",
                                   body: try encodeBody(input)),
                       as: Conversation.self)
    }

    /// Detaching needs an explicit JSON `null`, which `ConversationInput` drops.
    public func detachConversationFromFolder(id: String) async throws -> Conversation {
        let body = Data(#"{"folderId": null}"#.utf8)
        return try await send(makeRequest("api/conversations/\(pathSegment(id))",
                                          method: "PATCH",
                                          body: body),
                              as: Conversation.self)
    }

    public func deleteConversation(id: String) async throws {
        try await sendVoid(makeRequest("api/conversations/\(pathSegment(id))", method: "DELETE"))
    }

    public func deleteMessage(id: String) async throws {
        try await sendVoid(makeRequest("api/messages/\(pathSegment(id))", method: "DELETE"))
    }

    /// `POST /api/conversations/{id}/compact` — summarise older turns and free
    /// context. `changed` is false when there was nothing worth dropping.
    @discardableResult
    public func compactConversation(id: String) async throws -> ConversationCompaction {
        var request = makeRequest("api/conversations/\(pathSegment(id))/compact", method: "POST")
        // Compaction asks the model for a summary, which takes a local turn.
        request.timeoutInterval = 300
        return try await send(request, as: ConversationCompaction.self)
    }

    /// `md` or `json`, returned as raw bytes for a save panel.
    public func exportConversation(id: String, format: String) async throws -> Data {
        let path = query("api/conversations/\(pathSegment(id))/export", ["format": format])
        return try await sendData(makeRequest(path, accept: "*/*"))
    }

    // MARK: - Folders

    public func folders() async throws -> [Folder] {
        try await send(makeRequest("api/folders"), as: [Folder].self)
    }

    public func createFolder(_ input: FolderInput) async throws -> Folder {
        try await send(makeRequest("api/folders", method: "POST", body: try encodeBody(input)), as: Folder.self)
    }

    public func updateFolder(id: String, _ input: FolderInput) async throws -> Folder {
        try await send(makeRequest("api/folders/\(pathSegment(id))",
                                   method: "PATCH",
                                   body: try encodeBody(input)),
                       as: Folder.self)
    }

    public func deleteFolder(id: String, deleteChats: Bool) async throws {
        let path = query("api/folders/\(pathSegment(id))", ["deleteChats": deleteChats ? "1" : nil])
        try await sendVoid(makeRequest(path, method: "DELETE"))
    }

    // MARK: - Memory

    public func memories(scope: MemoryScope, folderId: String?) async throws -> [MemoryEntry] {
        let path = query("api/memories", ["scope": scope.rawValue, "folderId": folderId])
        return try await send(makeRequest(path), as: [MemoryEntry].self)
    }

    public func createMemory(_ input: MemoryInput) async throws -> MemoryEntry {
        try await send(makeRequest("api/memories", method: "POST", body: try encodeBody(input)),
                       as: MemoryEntry.self)
    }

    public func updateMemory(id: String, _ input: MemoryInput) async throws -> MemoryEntry {
        try await send(makeRequest("api/memories/\(pathSegment(id))",
                                   method: "PATCH",
                                   body: try encodeBody(input)),
                       as: MemoryEntry.self)
    }

    public func deleteMemory(id: String) async throws {
        try await sendVoid(makeRequest("api/memories/\(pathSegment(id))", method: "DELETE"))
    }

    // MARK: - Knowledge (RAG)

    public func knowledge(scope: MemoryScope, folderId: String?) async throws -> [KbDocument] {
        let path = query("api/knowledge", ["scope": scope.rawValue, "folderId": folderId])
        return try await send(makeRequest(path), as: [KbDocument].self)
    }

    public func uploadKnowledge(files: [MultipartFile],
                                scope: MemoryScope,
                                folderId: String?) async throws -> [KbDocument] {
        var fields: [String: String] = ["scope": scope.rawValue]
        if let folderId { fields["folderId"] = folderId }
        let (body, contentType) = Multipart.build(fields: fields, files: files, fieldName: "files")
        return try await send(makeRequest("api/knowledge",
                                          method: "POST",
                                          body: body,
                                          contentType: contentType),
                              as: [KbDocument].self)
    }

    public func deleteKnowledge(id: String) async throws {
        try await sendVoid(makeRequest("api/knowledge/\(pathSegment(id))", method: "DELETE"))
    }

    // MARK: - Attachments

    public func upload(files: [MultipartFile]) async throws -> [Attachment] {
        let (body, contentType) = Multipart.build(fields: [:], files: files, fieldName: "files")
        return try await send(makeRequest("api/upload",
                                          method: "POST",
                                          body: body,
                                          contentType: contentType),
                              as: [Attachment].self)
    }

    /// Attachment bytes need the bearer header, so `AsyncImage(url:)` cannot
    /// load them.
    public func attachmentData(id: String) async throws -> Data {
        try await sendData(makeRequest("api/attachments/\(pathSegment(id))", accept: "*/*"))
    }

    // MARK: - Settings, providers, models, search

    public func settings() async throws -> AppSettings {
        try await send(makeRequest("api/settings"), as: AppSettings.self)
    }

    public func patchSettings(_ settings: AppSettings) async throws -> AppSettings {
        try await send(makeRequest("api/settings", method: "PATCH", body: try encodeBody(settings)),
                       as: AppSettings.self)
    }

    public func providers() async throws -> [ProviderInfo] {
        try await send(makeRequest("api/providers"), as: ProvidersResponse.self).providers
    }

    public func models() async throws -> [ModelInfo] {
        try await send(makeRequest("api/models"), as: [ModelInfo].self)
    }

    public func search(_ text: String) async throws -> [SearchHit] {
        try await send(makeRequest(query("api/search", ["q": text])), as: [SearchHit].self)
    }

    // MARK: - Chat streaming

    /// Streams one assistant turn.
    ///
    /// The returned sequence finishes when the server closes the stream;
    /// cancelling the surrounding task aborts the request.
    public func chat(_ request: ChatSendRequest) throws -> AsyncThrowingStream<ChatStreamEvent, Error> {
        let urlRequest = makeRequest("api/chat",
                                     method: "POST",
                                     body: try encodeBody(request),
                                     accept: "text/event-stream")
        let frames = sseStream(urlRequest)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await frame in frames {
                        if let event = ChatSSEDecoder.decode(frame) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

// MARK: - Chat SSE decoding

/// Turns one `data:` payload into a typed event. Unknown types are skipped
/// rather than treated as errors, so the server can add frames freely.
public enum ChatSSEDecoder {
    private struct Frame: Decodable {
        let type: String
        let conversationId: String?
        let userMessage: Message?
        let assistantMessageId: String?
        let title: String?
        let citations: [Citation]?
        let delta: String?
        let message: MessageOrText?
        let tokensIn: Int?
        let tokensOut: Int?
        let costUsd: Double?
    }

    /// `message` is a `Message` on `done` and a plain string on `error` and
    /// `status`, so it is decoded as either.
    private enum MessageOrText: Decodable {
        case message(Message)
        case text(String)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else {
                self = .message(try container.decode(Message.self))
            }
        }

        var asMessage: Message? {
            if case let .message(value) = self { return value }
            return nil
        }

        var asText: String? {
            if case let .text(value) = self { return value }
            return nil
        }
    }

    public static func decode(_ frame: SSEFrame) -> ChatStreamEvent? {
        guard let parsed = frame.decode(Frame.self) else { return nil }
        switch parsed.type {
        case "start":
            guard let id = parsed.conversationId,
                  let user = parsed.userMessage,
                  let assistant = parsed.assistantMessageId else { return nil }
            return .start(conversationId: id, userMessage: user, assistantMessageId: assistant)
        case "title":
            guard let id = parsed.conversationId, let title = parsed.title else { return nil }
            return .title(conversationId: id, title: title)
        case "citations":
            return .citations(parsed.citations ?? [])
        case "status":
            // New in LocalCode: "Ładowanie modelu…" while the engine starts.
            guard let text = parsed.message?.asText ?? parsed.title, !text.isEmpty else { return nil }
            return .status(text)
        case "reasoning":
            return .reasoning(parsed.delta ?? "")
        case "text":
            return .text(parsed.delta ?? "")
        case "usage":
            return .usage(tokensIn: parsed.tokensIn, tokensOut: parsed.tokensOut, costUsd: parsed.costUsd)
        case "compaction":
            // The counters sit at the top level of the frame, not in a nested
            // object, so the whole payload is decoded a second time.
            guard let data = frame.data.data(using: .utf8),
                  let value = try? JSONDecoder().decode(ChatCompaction.self, from: data) else { return nil }
            return .compaction(value)
        case "context":
            guard let data = frame.data.data(using: .utf8),
                  let value = try? JSONDecoder().decode(ChatContextUsage.self, from: data) else { return nil }
            return .context(value)
        case "done":
            guard let message = parsed.message?.asMessage else { return nil }
            return .done(message)
        case "error":
            return .failure(parsed.message?.asText ?? "Nieoczekiwany błąd serwera.")
        default:
            return nil
        }
    }
}

// MARK: - Character sets

extension CharacterSet {
    /// Query values: everything `urlQueryAllowed` permits, minus the separators
    /// that would end the value early.
    static let lcQueryAllowed: CharacterSet = {
        var set = CharacterSet.urlQueryAllowed
        set.remove(charactersIn: "&=+?#")
        return set
    }()

    /// One path segment: model keys contain `/`, which must be escaped.
    static let lcPathSegmentAllowed: CharacterSet = {
        var set = CharacterSet.urlPathAllowed
        set.remove(charactersIn: "/?#")
        return set
    }()
}
