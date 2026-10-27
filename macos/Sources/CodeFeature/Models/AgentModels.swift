import Foundation

/// The wire types of the coding-agent API (`/api/agent/*`).
///
/// Every field the server may omit is optional or has a default: a client that
/// refuses to decode a frame is a client that goes blank in the middle of a
/// turn, and the agent protocol is still growing. Decoding is deliberately
/// tolerant everywhere in this file.
///
/// Keys are camelCase and timestamps are milliseconds, exactly as
/// `docs/ARCHITECTURE.md` describes.

// MARK: - Session

/// `AgentSession` — one coding session in one project directory.
public struct AgentSessionDTO: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var cwd: String
    public var modelKey: String?
    public var mode: String
    public var title: String
    public var createdAt: Double
    public var updatedAt: Double
    public var contextUsed: Int
    public var contextLimit: Int
    public var busy: Bool
    /// Canonical reasoning effort of this session (`off|low|medium|high|ultracode`).
    public var effort: String
    /// What the session's model can express: `[]`, `["off","on"]` or
    /// `["low","medium","high"]`. Empty hides the control.
    public var effortLevels: [String]
    /// `/goal` of this session, when one is set.
    public var goal: AgentGoal?

    public init(id: String,
                cwd: String,
                modelKey: String? = nil,
                mode: String = AgentMode.ask.wireValue,
                title: String = "",
                createdAt: Double = 0,
                updatedAt: Double = 0,
                contextUsed: Int = 0,
                contextLimit: Int = 0,
                busy: Bool = false,
                effort: String = "medium",
                effortLevels: [String] = []) {
        self.id = id
        self.cwd = cwd
        self.modelKey = modelKey
        self.mode = mode
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.contextUsed = contextUsed
        self.contextLimit = contextLimit
        self.busy = busy
        self.effort = effort
        self.effortLevels = effortLevels
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(String.self, forKey: .id)) ?? ""
        cwd = (try? c.decode(String.self, forKey: .cwd)) ?? ""
        modelKey = try? c.decodeIfPresent(String.self, forKey: .modelKey)
        mode = (try? c.decode(String.self, forKey: .mode)) ?? AgentMode.ask.wireValue
        title = (try? c.decode(String.self, forKey: .title)) ?? ""
        createdAt = (try? c.decode(Double.self, forKey: .createdAt)) ?? 0
        updatedAt = (try? c.decode(Double.self, forKey: .updatedAt)) ?? createdAt
        contextUsed = (try? c.decode(Int.self, forKey: .contextUsed)) ?? 0
        contextLimit = (try? c.decode(Int.self, forKey: .contextLimit)) ?? 0
        busy = (try? c.decode(Bool.self, forKey: .busy)) ?? false
        effort = (try? c.decode(String.self, forKey: .effort)) ?? "medium"
        effortLevels = (try? c.decode([String].self, forKey: .effortLevels)) ?? []
        goal = try? c.decodeIfPresent(AgentGoal.self, forKey: .goal)
    }

    /// Title for the sidebar — sessions get their title from the first message,
    /// so a brand new one has none yet.
    public var displayTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "Nowa sesja" : trimmed
    }

    /// `~/Projects/app` instead of `/Users/me/Projects/app`.
    public var shortCwd: String { AgentPaths.abbreviate(cwd) }
}

/// `/goal`: what the agent keeps working towards, and how far it got.
///
/// `state` is `active` (set, or sent back to work — `turns` counts those),
/// `done`, `blocked` (it needs the user), `limit` (out of attempts),
/// `stalled` (the same answer again and again) or `cleared`. The whole run is one turn; Stop ends it.
public struct AgentGoal: Codable, Hashable, Sendable {
    public var text: String
    public var state: String
    public var turns: Int
    public var maxTurns: Int
    public var summary: String

    public init(text: String, state: String, turns: Int = 0, maxTurns: Int = 0, summary: String = "") {
        self.text = text
        self.state = state
        self.turns = turns
        self.maxTurns = maxTurns
        self.summary = summary
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        state = (try? c.decode(String.self, forKey: .state)) ?? "active"
        turns = (try? c.decode(Int.self, forKey: .turns)) ?? 0
        maxTurns = (try? c.decode(Int.self, forKey: .maxTurns)) ?? 0
        summary = (try? c.decode(String.self, forKey: .summary)) ?? ""
    }

    public var isActive: Bool { state == "active" }

    /// The same sentence the terminal prints (`agent/tools/goal.py:notice_text`).
    public var noticeText: String {
        let detail = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        switch state {
        case "active":
            return turns == 0 ? "Cel: \(text)" : "Kontynuuję pracę nad celem (\(turns)/\(maxTurns))."
        case "done":
            return "Cel osiągnięty" + (detail.isEmpty ? "." : ": \(detail)")
        case "blocked":
            return "Cel czeka na Twoją decyzję" + (detail.isEmpty ? "." : ": \(detail)")
        case "limit":
            return "Cel nieosiągnięty po \(maxTurns) podejściach — napisz, co dalej, albo /goal wyczyść."
        case "stalled":
            return "Agent stoi w miejscu — powtarza tę samą odpowiedź. Napisz, co dalej, albo /goal wyczyść."
        case "cleared":
            return "Cel usunięty."
        default:
            return text
        }
    }
}

/// Permission mode. The wire values are the Python `MODES` tuple.
public enum AgentMode: String, CaseIterable, Identifiable, Sendable {
    case ask
    case acceptEdits
    case auto
    case yolo
    case plan

    public var id: String { rawValue }

    public var wireValue: String {
        switch self {
        case .ask: "default"
        case .acceptEdits: "accept_edits"
        case .auto: "auto"
        case .yolo: "yolo"
        case .plan: "plan"
        }
    }

    public init?(wireValue: String) {
        switch wireValue {
        case "default": self = .ask
        case "accept_edits": self = .acceptEdits
        case "auto": self = .auto
        case "yolo": self = .yolo
        case "plan": self = .plan
        default: return nil
        }
    }

    public var label: String {
        switch self {
        case .ask: "Pytaj"
        case .acceptEdits: "Akceptuj edycje"
        case .auto: "Auto"
        case .yolo: "YOLO"
        case .plan: "Plan"
        }
    }

    public var hint: String {
        switch self {
        case .ask: "Pyta o zapis plików i polecenia powłoki."
        case .acceptEdits: "Zmiany w plikach bez pytania, polecenia nadal wymagają zgody."
        case .auto: "Model krytycznie ocenia każdą akcję i mówi dlaczego. O niebezpieczne i spoza projektu pyta Ciebie."
        case .yolo: "Wszystko bez pytania — poza poleceniami uznanymi za niebezpieczne."
        case .plan: "Tylko odczyt: agent planuje, niczego nie zmienia."
        }
    }

    public var systemImage: String {
        switch self {
        case .ask: "hand.raised"
        case .acceptEdits: "pencil.circle"
        case .auto: "checkmark.shield"
        case .yolo: "bolt.fill"
        case .plan: "list.bullet.clipboard"
        }
    }
}

// MARK: - Transcript items (REST history)

/// `AgentItem` — one line of the stored transcript, as
/// `GET /api/agent/sessions/{id}` returns it.
public struct AgentItemDTO: Decodable, Sendable {
    public var kind: String
    public var id: String
    public var text: String?
    public var toolName: String?
    public var title: String?
    public var arguments: JSONValue?
    public var ok: Bool?
    public var summary: String?
    public var output: String?
    public var diff: String?
    public var createdAt: Double
    /// Set by the server for notices it records itself (`status`, `compaction`, `error`).
    public var noticeType: String?
    /// Auto mode: what the reviewer said about this tool call.
    public var review: AutoReviewInfo?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = (try? c.decode(String.self, forKey: .kind)) ?? "notice"
        id = (try? c.decode(String.self, forKey: .id)) ?? UUID().uuidString
        text = try? c.decodeIfPresent(String.self, forKey: .text)
        toolName = try? c.decodeIfPresent(String.self, forKey: .toolName)
        title = try? c.decodeIfPresent(String.self, forKey: .title)
        arguments = try? c.decodeIfPresent(JSONValue.self, forKey: .arguments)
        ok = try? c.decodeIfPresent(Bool.self, forKey: .ok)
        summary = try? c.decodeIfPresent(String.self, forKey: .summary)
        output = try? c.decodeIfPresent(String.self, forKey: .output)
        diff = try? c.decodeIfPresent(String.self, forKey: .diff)
        createdAt = (try? c.decode(Double.self, forKey: .createdAt)) ?? 0
        noticeType = try? c.decodeIfPresent(String.self, forKey: .noticeType)
        review = try? c.decodeIfPresent(AutoReviewInfo.self, forKey: .review)
    }

    private enum CodingKeys: String, CodingKey {
        case kind, id, text, toolName, title, arguments, ok, summary, output, diff, createdAt, noticeType
        case review
    }
}

/// Auto mode's verdict on one tool call (`auto_review`, `review` on a stored
/// tool item and on a permission request). `verdict` is the reviewer's: `allow`
/// (TAK), `deny` (NIE — the user is asked and may overrule it) or `ask` (no
/// clear answer — the user is asked). A call runs on its own only when the
/// reviewer and, for a command, the local guard both said TAK.
public struct AutoReviewInfo: Decodable, Hashable, Sendable {
    public var verdict: String
    public var reason: String
    /// Model key of the reviewer that answered.
    public var model: String
    /// The context-free local guard's opinion (`agent/guard.py`), when it
    /// checked the call — shell commands and anything outside the project.
    public var guardOpinion: GuardOpinion?

    public init(verdict: String, reason: String, model: String = "",
                guardOpinion: GuardOpinion? = nil) {
        self.verdict = verdict
        self.reason = reason
        self.model = model
        self.guardOpinion = guardOpinion
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        verdict = (try? c.decode(String.self, forKey: .verdict)) ?? "ask"
        reason = (try? c.decode(String.self, forKey: .reason)) ?? ""
        model = (try? c.decode(String.self, forKey: .model)) ?? ""
        guardOpinion = try? c.decodeIfPresent(GuardOpinion.self, forKey: .guardOpinion)
    }

    private enum CodingKeys: String, CodingKey {
        case verdict, reason, model
        case guardOpinion = "guard"
    }

    /// On a permission card a TAK never decides — the user does.
    public var requestLabel: String {
        isAllowed ? "Recenzent: TAK (tylko opinia)" : (isDenied ? "Recenzent: NIE" : "Recenzent nie zdecydował")
    }

    public var isAllowed: Bool { verdict == "allow" }
    public var isDenied: Bool { verdict == "deny" }

    /// „Auto: TAK — powód” — the line under the tool card.
    public var line: String {
        let word = isAllowed ? "TAK" : (isDenied ? "NIE" : "pytam Ciebie")
        let detail = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        return detail.isEmpty ? "Auto: \(word)" : "Auto: \(word) — \(detail)"
    }
}

/// The local guard's opinion on one call: a small model on this Mac that sees
/// only the command — not the conversation — so nothing in the chat, the files
/// or a remote model's answer can talk it round. Fail-closed: anything but TAK
/// sends the call to the user.
public struct GuardOpinion: Decodable, Hashable, Sendable {
    public var verdict: String
    public var reason: String
    public var model: String

    public init(verdict: String, reason: String, model: String = "") {
        self.verdict = verdict
        self.reason = reason
        self.model = model
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        verdict = (try? c.decode(String.self, forKey: .verdict)) ?? "ask"
        reason = (try? c.decode(String.self, forKey: .reason)) ?? ""
        model = (try? c.decode(String.self, forKey: .model)) ?? ""
    }

    private enum CodingKeys: String, CodingKey { case verdict, reason, model }

    public var isAllowed: Bool { verdict == "allow" }
    public var isDenied: Bool { verdict == "deny" }

    /// „Strażnik lokalny: NIE — powód”.
    public var line: String {
        let word = isAllowed ? "TAK" : (isDenied ? "NIE" : "bez decyzji")
        let detail = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        return detail.isEmpty ? "Strażnik lokalny: \(word)" : "Strażnik lokalny: \(word) — \(detail)"
    }
}

/// `{session, items, todos}` — the body of `GET /api/agent/sessions/{id}`.
public struct AgentSessionSnapshot: Decodable, Sendable {
    public var session: AgentSessionDTO
    public var items: [AgentItemDTO]
    public var todos: [TodoItem]

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        session = try c.decode(AgentSessionDTO.self, forKey: .session)
        items = (try? c.decode([AgentItemDTO].self, forKey: .items)) ?? []
        todos = (try? c.decode([TodoItem].self, forKey: .todos)) ?? []
    }

    private enum CodingKeys: String, CodingKey { case session, items, todos }
}

// MARK: - Small payloads

/// One entry of the agent's plan (`todo_write`).
public struct TodoItem: Decodable, Hashable, Sendable, Identifiable {
    public enum Status: String, Sendable {
        case pending
        case inProgress
        case done
    }

    public var content: String
    public var status: Status

    public var id: String { "\(status.rawValue)|\(content)" }

    public init(content: String, status: Status) {
        self.content = content
        self.status = status
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        content = (try? c.decode(String.self, forKey: .content)) ?? ""
        switch (try? c.decode(String.self, forKey: .status)) ?? "pending" {
        case "in_progress", "inProgress": status = .inProgress
        case "done", "completed": status = .done
        default: status = .pending
        }
    }

    private enum CodingKeys: String, CodingKey { case content, status }
}

/// `GenStats` totals of a finished turn (`turn_end.stats`).
public struct TurnStats: Decodable, Hashable, Sendable {
    public var promptTokens: Int?
    public var cachedTokens: Int?
    public var completionTokens: Int?
    public var promptTps: Double?
    public var genTps: Double?
    public var totalMs: Double?
    public var steps: Int?
    public var elapsedMs: Double?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        promptTokens = try? c.decodeIfPresent(Int.self, forKey: .promptTokens)
        cachedTokens = try? c.decodeIfPresent(Int.self, forKey: .cachedTokens)
        completionTokens = try? c.decodeIfPresent(Int.self, forKey: .completionTokens)
        promptTps = try? c.decodeIfPresent(Double.self, forKey: .promptTps)
        genTps = try? c.decodeIfPresent(Double.self, forKey: .genTps)
        totalMs = try? c.decodeIfPresent(Double.self, forKey: .totalMs)
        steps = try? c.decodeIfPresent(Int.self, forKey: .steps)
        elapsedMs = try? c.decodeIfPresent(Double.self, forKey: .elapsedMs)
    }

    private enum CodingKeys: String, CodingKey {
        case promptTokens, cachedTokens, completionTokens, promptTps, genTps, totalMs, steps, elapsedMs
    }
}

/// `context` — everything the context gauge needs.
public struct ContextStats: Decodable, Hashable, Sendable {
    public var used: Int
    public var limit: Int
    public var system: Int
    public var tools: Int
    public var history: Int
    public var reserve: Int

    public init(used: Int = 0, limit: Int = 0, system: Int = 0,
                tools: Int = 0, history: Int = 0, reserve: Int = 0) {
        self.used = used
        self.limit = limit
        self.system = system
        self.tools = tools
        self.history = history
        self.reserve = reserve
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        used = (try? c.decode(Int.self, forKey: .used)) ?? 0
        limit = (try? c.decode(Int.self, forKey: .limit)) ?? 0
        system = (try? c.decode(Int.self, forKey: .system)) ?? 0
        tools = (try? c.decode(Int.self, forKey: .tools)) ?? 0
        history = (try? c.decode(Int.self, forKey: .history)) ?? 0
        reserve = (try? c.decode(Int.self, forKey: .reserve)) ?? 0
    }

    private enum CodingKeys: String, CodingKey { case used, limit, system, tools, history, reserve }

    /// `0…1`, clamped — the gauge never draws past its own track.
    public var fraction: Double {
        guard limit > 0 else { return 0 }
        return min(1, max(0, Double(used) / Double(limit)))
    }

    public var percent: Int { Int((fraction * 100).rounded()) }

    public var isEmpty: Bool { limit <= 0 }
}

/// `permission_request` — the agent is waiting for a yes or a no.
public struct PermissionRequestDTO: Decodable, Identifiable, Hashable, Sendable {
    public var requestId: String
    public var tool: String
    public var title: String
    public var preview: String
    public var options: [String]
    /// Auto mode: what the reviewer (and the local guard) said before the
    /// user is asked; `nil` outside auto mode.
    public var review: AutoReviewInfo?
    /// The helper agent asking ("h2"); `nil` when it is the agent itself.
    public var helper: String?

    public var id: String { requestId }

    public init(requestId: String, tool: String, title: String, preview: String,
                options: [String] = ["allow", "deny", "always"], review: AutoReviewInfo? = nil) {
        self.requestId = requestId
        self.tool = tool
        self.title = title
        self.preview = preview
        self.options = options
        self.review = review
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        requestId = (try? c.decode(String.self, forKey: .requestId)) ?? ""
        tool = (try? c.decode(String.self, forKey: .tool)) ?? ""
        title = (try? c.decode(String.self, forKey: .title)) ?? "Operacja"
        preview = (try? c.decode(String.self, forKey: .preview)) ?? ""
        options = (try? c.decode([String].self, forKey: .options)) ?? ["allow", "deny", "always"]
        review = try? c.decodeIfPresent(AutoReviewInfo.self, forKey: .review)
        helper = try? c.decodeIfPresent(String.self, forKey: .helper)
    }

    private enum CodingKeys: String, CodingKey { case requestId, tool, title, preview, options, review, helper }

    /// A shell command's preview is "Co robi: …", a blank line, then the
    /// command: the first part is prose for the card, the rest is the command.
    public var previewParts: (does: String?, body: String) {
        let prefix = "Co robi: "
        guard preview.hasPrefix(prefix), let gap = preview.range(of: "\n\n") else {
            return (nil, preview)
        }
        let does = String(preview[preview.startIndex..<gap.lowerBound])
        return (does, String(preview[gap.upperBound...]))
    }

    public var allowsAlways: Bool { options.contains("always") }

    /// A diff preview gets the red/green treatment, anything else is a command.
    public var previewIsDiff: Bool { DiffLine.looksLikeDiff(preview) }
}

/// A local model that can run the agent (`GET /api/library`).
public struct AgentModelOption: Decodable, Identifiable, Hashable, Sendable {
    public var key: String
    public var displayName: String
    public var format: String
    public var contextLength: Int?
    public var sizeBytes: Int
    public var capabilities: [String]
    public var loaded: Bool
    /// A local GGUF model with its vision projector.
    public var hasVision: Bool

    public var id: String { key }

    /// Takes pictures in LocalCode — the same rule as the terminal's
    /// `model_sees_images`: a remote model says so in its capabilities, a GGUF
    /// one needs its projector, MLX vision models are not wired to the engine.
    public var seesImages: Bool {
        if capabilities.contains("remote") { return capabilities.contains("vision") }
        if format.lowercased() == "mlx" { return false }
        return hasVision || capabilities.contains("vision")
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = (try? c.decode(String.self, forKey: .key)) ?? ""
        displayName = (try? c.decode(String.self, forKey: .displayName)) ?? key
        format = (try? c.decode(String.self, forKey: .format)) ?? ""
        contextLength = try? c.decodeIfPresent(Int.self, forKey: .contextLength)
        sizeBytes = (try? c.decode(Int.self, forKey: .sizeBytes)) ?? 0
        capabilities = (try? c.decode([String].self, forKey: .capabilities)) ?? []
        loaded = (try? c.decode(Bool.self, forKey: .loaded)) ?? false
        hasVision = (try? c.decode(Bool.self, forKey: .hasVision)) ?? false
    }

    private enum CodingKeys: String, CodingKey {
        case key, displayName, format, contextLength, sizeBytes, capabilities, loaded, hasVision
    }

    /// Embedding models cannot answer, so they never appear in the picker.
    public var isEmbedding: Bool { capabilities.contains("embedding") }
}

// MARK: - Paths

/// Path formatting shared by the sidebar, the header and the tool cards.
public enum AgentPaths {
    public static func abbreviate(_ path: String) -> String {
        let home = NSHomeDirectory()
        if path == home { return "~" }
        if path.hasPrefix(home + "/") { return "~" + path.dropFirst(home.count) }
        return path
    }

    public static func name(of path: String) -> String {
        let name = (path as NSString).lastPathComponent
        return name.isEmpty ? path : name
    }

    /// `~/Documents/localcode` → `~/Documents` (what the sidebar shows under the name).
    public static func parent(of path: String) -> String {
        abbreviate((path as NSString).deletingLastPathComponent)
    }
}

// MARK: - JSON values

/// Any JSON the server sends inside `arguments`.
///
/// Tool arguments are free-form by design (every tool has its own schema), so
/// they are kept as parsed JSON and rendered on demand instead of being forced
/// into a Swift type per tool.
public enum JSONValue: Codable, Hashable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case let .bool(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .string(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        }
    }

    public var objectValue: [String: JSONValue]? {
        if case let .object(value) = self { return value }
        return nil
    }

    public var stringValue: String? {
        if case let .string(value) = self { return value }
        return nil
    }

    /// One-line rendering, short enough for a tool card subtitle.
    public var inlineDescription: String {
        switch self {
        case .null:
            return "—"
        case let .bool(value):
            return value ? "tak" : "nie"
        case let .number(value):
            return value == value.rounded() ? String(Int(value)) : String(value)
        case let .string(value):
            return value.replacingOccurrences(of: "\n", with: " ⏎ ")
        case let .array(values):
            return values.map(\.inlineDescription).joined(separator: ", ")
        case let .object(values):
            return values.sorted { $0.key < $1.key }
                .map { "\($0.key): \($0.value.inlineDescription)" }
                .joined(separator: ", ")
        }
    }
}


/// One slash command, as `GET /api/agent/commands` serves it.
///
/// The table itself lives on the server (`localcode/tui/commands.py`) and the
/// terminal reads the very same rows. Keeping a second copy here is how the two
/// interfaces drifted to four commands against twenty-three.
public struct AgentSlashCommand: Decodable, Identifiable, Hashable, Sendable {
    public var name: String
    public var summary: String
    public var usage: String?
    public var aliases: [String]
    public var values: [String]

    public var id: String { name }
    public var takesArgument: Bool { !values.isEmpty || (usage?.contains("[") ?? false) }

    public init(name: String, summary: String, usage: String? = nil,
                aliases: [String] = [], values: [String] = []) {
        self.name = name
        self.summary = summary
        self.usage = usage
        self.aliases = aliases
        self.values = values
    }
}

/// `GET /api/agent/commands`.
public struct AgentCommandSet: Decodable, Sendable {
    public var commands: [AgentSlashCommand]
    public var initPrompt: String
}

// MARK: - Help bot

/// `GET /api/helpbot` — whether Pomocnik is ready, coming, or missing.
///
/// The help bot is the agent loop with another prompt and other tools (models,
/// downloads, every setting), so its conversation is an ordinary agent session;
/// this is only about whether its model is on disk yet.
public struct HelpbotStatusDTO: Decodable, Hashable, Sendable {
    public enum State: String, Decodable, Sendable { case ready, downloading, missing }

    public var model: String
    public var modelKey: String?
    public var displayName: String?
    public var sizeBytes: Int?
    public var state: State
    public var loaded: Bool
    public var download: HelpbotDownloadDTO?
    public var cwd: String
    public var autoDownload: Bool
    /// Why the download did not start (e.g. not enough room on the disk).
    public var error: String?
}

/// The help bot model's download job, as the download manager reports it.
public struct HelpbotDownloadDTO: Decodable, Hashable, Sendable {
    public var state: String?
    public var doneBytes: Int?
    public var totalBytes: Int?
    public var error: String?

    /// 0…1, or `nil` before the size is known.
    public var fraction: Double? {
        guard let total = totalBytes, total > 0 else { return nil }
        return min(1, Double(doneBytes ?? 0) / Double(total))
    }
}
