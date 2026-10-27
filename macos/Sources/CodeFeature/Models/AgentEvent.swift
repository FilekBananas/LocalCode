import Foundation

/// One frame of the agent WebSocket (`GET /api/agent/sessions/{id}/ws`).
///
/// The server sends the `localcode/events.py` dictionaries plus four frames of
/// its own (`session`, `history`, `ack`, `permission_resolved`). Anything this
/// build does not know becomes `.unknown`, which the store ignores — a new
/// event type on the Python side must never break an older app.
public enum AgentEvent: Sendable {
    case session(AgentSessionDTO)
    case history([AgentItemDTO])
    case turnStart(turnId: String, userText: String)
    /// A queued message the running turn took between two steps.
    case userMessage(String)
    case textDelta(String)
    case reasoningDelta(String)
    case toolCall(ToolCallPayload)
    case toolResult(ToolResultPayload)
    case permissionRequest(PermissionRequestDTO)
    case permissionResolved(requestId: String, decision: String)
    /// Auto mode: the reviewer model's verdict on one tool call, and why.
    case autoReview(callId: String, review: AutoReviewInfo)
    case context(ContextStats)
    case compaction(before: Int, after: Int, strategy: String)
    case status(message: String, phase: StatusPhase)
    case todos([TodoItem])
    /// `/goal` set, sent back to work, reached, blocked, out of attempts or cleared.
    case goal(AgentGoal)
    case turnEnd(turnId: String, reason: String, stats: TurnStats?)
    /// A helper agent working on one delegated task (`delegate`). Its own
    /// events never travel: its `turn_end` would end this turn mid-work.
    case subagentStart(SubagentStartPayload)
    case subagentStep(SubagentStepPayload)
    case subagentEnd(SubagentEndPayload)
    case error(message: String, code: String?)
    case ack(of: String, status: String)
    case unknown(String)

    /// `status.phase`: `loading` and `thinking` are shown as a status line,
    /// `tool` carries live output of the running tool, `queued` reports a
    /// message parked behind the current turn.
    public enum StatusPhase: String, Sendable {
        case loading
        case thinking
        case tool
        case queued
        case info

        public init(wireValue: String?) {
            self = StatusPhase(rawValue: wireValue ?? "info") ?? .info
        }
    }

    public struct ToolCallPayload: Sendable, Hashable {
        public var callId: String
        public var name: String
        public var title: String
        public var arguments: JSONValue?

        public init(callId: String, name: String, title: String, arguments: JSONValue?) {
            self.callId = callId
            self.name = name
            self.title = title
            self.arguments = arguments
        }
    }

    public struct ToolResultPayload: Sendable, Hashable {
        public var callId: String
        public var name: String
        public var ok: Bool
        public var summary: String
        public var output: String
        public var diff: String?

        public init(callId: String, name: String, ok: Bool, summary: String, output: String, diff: String?) {
            self.callId = callId
            self.name = name
            self.ok = ok
            self.summary = summary
            self.output = output
            self.diff = diff
        }
    }
}

/// Turns raw WebSocket text into `AgentEvent`s.
///
/// Kept free of SwiftUI and of the store so it can be replayed against the
/// captured fixtures in `macos/Tests/Fixtures/` without a window.
public enum AgentEventDecoder {

    public static func decode(_ text: String) -> AgentEvent? {
        guard let data = text.data(using: .utf8) else { return nil }
        return decode(data)
    }

    public static func decode(_ data: Data) -> AgentEvent? {
        let decoder = JSONDecoder()
        guard let envelope = try? decoder.decode(Envelope.self, from: data) else { return nil }
        switch envelope.type {
        case "session":
            guard let session = envelope.session else { return .unknown(envelope.type) }
            return .session(session)

        case "history":
            return .history(envelope.items.flatMap { decodeItems($0, using: decoder) } ?? [])

        case "turn_start":
            return .turnStart(turnId: envelope.turnId ?? "", userText: envelope.userText ?? "")

        case "user_message":
            return .userMessage(envelope.text ?? "")

        case "goal":
            return .goal(AgentGoal(text: envelope.text ?? "", state: envelope.state ?? "active",
                                   turns: envelope.turns ?? 0, maxTurns: envelope.maxTurns ?? 0,
                                   summary: envelope.summary ?? ""))

        case "text_delta":
            return .textDelta(envelope.text ?? "")

        case "reasoning_delta":
            return .reasoningDelta(envelope.text ?? "")

        case "tool_call":
            return .toolCall(.init(callId: envelope.callId ?? UUID().uuidString,
                                   name: envelope.name ?? "",
                                   title: envelope.title ?? "",
                                   arguments: envelope.arguments))

        case "tool_result":
            return .toolResult(.init(callId: envelope.callId ?? "",
                                     name: envelope.name ?? "",
                                     ok: envelope.ok ?? true,
                                     summary: envelope.summary ?? "",
                                     output: envelope.output ?? "",
                                     diff: envelope.diff))

        case "subagent_start":
            return .subagentStart(.init(agentId: envelope.agentId ?? "",
                                        parentCallId: envelope.parentCallId ?? "",
                                        title: envelope.title ?? "",
                                        model: envelope.model ?? "",
                                        remote: envelope.remote ?? false,
                                        background: envelope.background ?? false))

        case "subagent_step":
            return .subagentStep(.init(parentCallId: envelope.parentCallId ?? "",
                                       title: envelope.title ?? envelope.name ?? "",
                                       ok: envelope.ok ?? true,
                                       summary: envelope.summary ?? ""))

        case "subagent_end":
            return .subagentEnd(.init(agentId: envelope.agentId ?? "",
                                      parentCallId: envelope.parentCallId ?? "",
                                      ok: envelope.ok ?? true,
                                      steps: envelope.steps ?? 0,
                                      background: envelope.background ?? false,
                                      report: envelope.report ?? ""))

        case "permission_request":
            guard let request = try? decoder.decode(PermissionRequestDTO.self, from: data) else {
                return .unknown(envelope.type)
            }
            return .permissionRequest(request)

        case "permission_resolved":
            return .permissionResolved(requestId: envelope.requestId ?? "",
                                       decision: envelope.decision ?? "")

        case "auto_review":
            return .autoReview(callId: envelope.callId ?? "",
                               review: AutoReviewInfo(verdict: envelope.verdict ?? "ask",
                                                      reason: envelope.reason ?? "",
                                                      model: envelope.reviewerModel ?? "",
                                                      guardOpinion: envelope.guardOpinion))

        case "context":
            guard let stats = try? decoder.decode(ContextStats.self, from: data) else {
                return .unknown(envelope.type)
            }
            return .context(stats)

        case "compaction":
            return .compaction(before: envelope.before ?? 0,
                               after: envelope.after ?? 0,
                               strategy: envelope.strategy ?? "")

        case "status":
            return .status(message: envelope.message ?? "",
                           phase: AgentEvent.StatusPhase(wireValue: envelope.phase))

        case "todos":
            return .todos(envelope.items.flatMap { decodeTodos($0, using: decoder) } ?? [])

        case "turn_end":
            return .turnEnd(turnId: envelope.turnId ?? "",
                            reason: envelope.reason ?? "done",
                            stats: envelope.stats)

        case "error":
            return .error(message: envelope.message ?? "Agent zgłosił nieznany błąd.",
                          code: envelope.code)

        case "ack":
            return .ack(of: envelope.of ?? "", status: envelope.status ?? "")

        default:
            return .unknown(envelope.type)
        }
    }

    // `items` means two different things (history items and todo entries), so
    // it is decoded lazily from the raw JSON once the type is known.
    private static func decodeItems(_ value: JSONValue, using decoder: JSONDecoder) -> [AgentItemDTO]? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return try? decoder.decode([AgentItemDTO].self, from: data)
    }

    private static func decodeTodos(_ value: JSONValue, using decoder: JSONDecoder) -> [TodoItem]? {
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return try? decoder.decode([TodoItem].self, from: data)
    }

    /// Every field any known frame may carry; unknown keys are simply not read.
    private struct Envelope: Decodable {
        var type: String
        var text: String?
        var turnId: String?
        var userText: String?
        var callId: String?
        var name: String?
        var title: String?
        var arguments: JSONValue?
        var ok: Bool?
        var summary: String?
        var output: String?
        var diff: String?
        var requestId: String?
        var decision: String?
        var before: Int?
        var after: Int?
        var strategy: String?
        var message: String?
        var phase: String?
        var reason: String?
        var code: String?
        var of: String?
        var status: String?
        var items: JSONValue?
        var stats: TurnStats?
        var session: AgentSessionDTO?
        var agentId: String?
        var parentCallId: String?
        var model: String?
        var remote: Bool?
        var steps: Int?
        var state: String?
        var turns: Int?
        var maxTurns: Int?
        var verdict: String?
        var reviewerModel: String?
        var guardOpinion: GuardOpinion?
        var background: Bool?
        var report: String?

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            type = (try? c.decode(String.self, forKey: .type)) ?? ""
            text = try? c.decodeIfPresent(String.self, forKey: .text)
            turnId = try? c.decodeIfPresent(String.self, forKey: .turnId)
            userText = try? c.decodeIfPresent(String.self, forKey: .userText)
            callId = try? c.decodeIfPresent(String.self, forKey: .callId)
            name = try? c.decodeIfPresent(String.self, forKey: .name)
            title = try? c.decodeIfPresent(String.self, forKey: .title)
            arguments = try? c.decodeIfPresent(JSONValue.self, forKey: .arguments)
            ok = try? c.decodeIfPresent(Bool.self, forKey: .ok)
            summary = try? c.decodeIfPresent(String.self, forKey: .summary)
            output = try? c.decodeIfPresent(String.self, forKey: .output)
            diff = try? c.decodeIfPresent(String.self, forKey: .diff)
            requestId = try? c.decodeIfPresent(String.self, forKey: .requestId)
            decision = try? c.decodeIfPresent(String.self, forKey: .decision)
            before = try? c.decodeIfPresent(Int.self, forKey: .before)
            after = try? c.decodeIfPresent(Int.self, forKey: .after)
            strategy = try? c.decodeIfPresent(String.self, forKey: .strategy)
            message = try? c.decodeIfPresent(String.self, forKey: .message)
            phase = try? c.decodeIfPresent(String.self, forKey: .phase)
            reason = try? c.decodeIfPresent(String.self, forKey: .reason)
            code = try? c.decodeIfPresent(String.self, forKey: .code)
            of = try? c.decodeIfPresent(String.self, forKey: .of)
            status = try? c.decodeIfPresent(String.self, forKey: .status)
            items = try? c.decodeIfPresent(JSONValue.self, forKey: .items)
            stats = try? c.decodeIfPresent(TurnStats.self, forKey: .stats)
            session = try? c.decodeIfPresent(AgentSessionDTO.self, forKey: .session)
            agentId = try? c.decodeIfPresent(String.self, forKey: .agentId)
            parentCallId = try? c.decodeIfPresent(String.self, forKey: .parentCallId)
            model = try? c.decodeIfPresent(String.self, forKey: .model)
            remote = try? c.decodeIfPresent(Bool.self, forKey: .remote)
            steps = try? c.decodeIfPresent(Int.self, forKey: .steps)
            state = try? c.decodeIfPresent(String.self, forKey: .state)
            turns = try? c.decodeIfPresent(Int.self, forKey: .turns)
            maxTurns = try? c.decodeIfPresent(Int.self, forKey: .maxTurns)
            verdict = try? c.decodeIfPresent(String.self, forKey: .verdict)
            reviewerModel = try? c.decodeIfPresent(String.self, forKey: .reviewerModel)
            guardOpinion = try? c.decodeIfPresent(GuardOpinion.self, forKey: .guardOpinion)
            background = try? c.decodeIfPresent(Bool.self, forKey: .background)
            report = try? c.decodeIfPresent(String.self, forKey: .report)
        }

        private enum CodingKeys: String, CodingKey {
            case type, text, turnId, userText, callId, name, title, arguments, ok, summary, output,
                 diff, requestId, decision, before, after, strategy, message, phase, reason, code,
                 of, status, items, stats, session, agentId, parentCallId, model, remote, steps,
                 state, turns, maxTurns, verdict, reviewerModel, background, report
            case guardOpinion = "guard"
        }
    }
}

// MARK: - Client → server

/// What the app sends into the socket.
public enum AgentClientMessage: Sendable {
    case user(String)
    /// A message with pictures for a vision model.
    case userWithImages(String, [AgentImage])
    case permission(requestId: String, decision: String)
    case interrupt
    case setMode(String)
    case setModel(String)
    case setEffort(String)
    case compact
    case clear
    /// `/goal <cel>`: the server sets it and starts the turn.
    case setGoal(String)
    case clearGoal

    /// JSON text, ready for `URLSessionWebSocketTask.send`.
    public var json: String {
        let payload: [String: Any]
        switch self {
        case let .user(text):
            payload = ["type": "user", "text": text]
        case let .userWithImages(text, images):
            payload = ["type": "user", "text": text, "images": images.map(\.payload)]
        case let .permission(requestId, decision):
            payload = ["type": "permission", "requestId": requestId, "decision": decision]
        case .interrupt:
            payload = ["type": "interrupt"]
        case let .setMode(mode):
            payload = ["type": "setMode", "mode": mode]
        case let .setModel(key):
            payload = ["type": "setModel", "modelKey": key]
        case let .setEffort(effort):
            payload = ["type": "setEffort", "effort": effort]
        case .compact:
            payload = ["type": "compact"]
        case .clear:
            payload = ["type": "clear"]
        case let .setGoal(text):
            payload = ["type": "setGoal", "text": text]
        case .clearGoal:
            payload = ["type": "clearGoal"]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let text = String(data: data, encoding: .utf8) else {
            return "{\"type\":\"interrupt\"}"
        }
        return text
    }
}


// MARK: - Helper agents

/// `subagent_start` — a helper agent took on one delegated task.
public struct SubagentStartPayload: Sendable, Hashable {
    public var agentId: String
    /// The `delegate` call this helper belongs to, so its steps can be shown
    /// under that card instead of loose in the transcript.
    public var parentCallId: String
    public var title: String
    public var model: String
    /// The helper runs behind somebody's API key — worth saying, because it
    /// reads project files.
    public var remote: Bool
    /// `delegate(wait=false)`: the agent works on; the report comes later.
    public var background: Bool = false
}

/// `subagent_end` — the helper stopped; a background one brings its report.
public struct SubagentEndPayload: Sendable, Hashable {
    public var agentId: String
    public var parentCallId: String
    public var ok: Bool
    public var steps: Int
    public var background: Bool = false
    public var report: String = ""
}

/// `subagent_step` — one finished tool call of the helper, never one token.
public struct SubagentStepPayload: Sendable, Hashable {
    public var parentCallId: String
    public var title: String
    public var ok: Bool
    public var summary: String
}
