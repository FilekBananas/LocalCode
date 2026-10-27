import Foundation

/// One block of the agent transcript.
///
/// The transcript is a flat list on purpose: the reader scrolls through user
/// messages, streamed answers, thinking, tool cards, plans and notices in the
/// order they happened, exactly like the terminal agent prints them.
public struct TranscriptEntry: Identifiable, Equatable {
    public enum Kind: String, Equatable {
        case user
        case assistant
        case reasoning
        case tool
        case notice
        case compaction
        case todos
        case stats
        case error
    }

    public let id: String
    public var kind: Kind
    public var text: String
    public var createdAt: Double
    public var tool: ToolState?
    public var todos: [TodoItem]
    public var compaction: CompactionInfo?
    public var stats: TurnStats?
    /// `turn_end.reason`: `done` | `interrupted` | `max_steps` | `error`.
    public var reason: String?
    /// Thumbnails (PNG) of the pictures sent with a user message.
    public var images: [Data] = []

    public init(id: String,
                kind: Kind,
                text: String = "",
                createdAt: Double = Date().timeIntervalSince1970 * 1000,
                tool: ToolState? = nil,
                todos: [TodoItem] = [],
                compaction: CompactionInfo? = nil,
                stats: TurnStats? = nil,
                reason: String? = nil) {
        self.id = id
        self.kind = kind
        self.text = text
        self.createdAt = createdAt
        self.tool = tool
        self.todos = todos
        self.compaction = compaction
        self.stats = stats
        self.reason = reason
    }
}

/// A tool call and — once it arrives — its result.
public struct ToolState: Equatable {
    public var callId: String
    public var name: String
    public var title: String
    public var arguments: JSONValue?
    /// `nil` while the tool is still running.
    public var ok: Bool?
    public var summary: String?
    public var output: String?
    public var diff: String?
    /// Lines the tool printed while it was running (`status` frames, `phase: "tool"`).
    public var liveOutput: [String] = []
    /// Steps of the helper agent this `delegate` call started — one line per
    /// finished tool call, shown nested under the card rather than loose in the
    /// transcript, because they are the call's detail and not the conversation's.
    public var helperSteps: [String] = []
    /// The helper is still working.
    public var helperRunning: Bool = false
    /// The helper runs behind somebody's API key.
    public var helperRemote: Bool = false
    /// The helper works in the background (`delegate(wait=false)`): the card
    /// returned at once, the report comes when the helper finishes.
    public var helperBackground: Bool = false
    /// The helper's short name ("h2") — the one on its permission cards.
    public var helperLabel: String = ""
    /// A background helper's report, once it is done.
    public var helperReport: String?
    /// Auto mode: the reviewer's verdict on this call, and why.
    public var review: AutoReviewInfo?

    public init(callId: String,
                name: String,
                title: String,
                arguments: JSONValue? = nil,
                ok: Bool? = nil,
                summary: String? = nil,
                output: String? = nil,
                diff: String? = nil,
                liveOutput: [String] = []) {
        self.callId = callId
        self.name = name
        self.title = title
        self.arguments = arguments
        self.ok = ok
        self.summary = summary
        self.output = output
        self.diff = diff
        self.liveOutput = liveOutput
    }

    public var isRunning: Bool { ok == nil }

    /// Card title: the server sends a Polish one, otherwise fall back to the
    /// tool's own name and its main argument.
    public var displayTitle: String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        let tool = ToolPresentation.of(name)
        let argument = mainArgument
        return argument.isEmpty ? tool.label : "\(tool.label): \(argument)"
    }

    /// The argument worth showing next to the title (path, pattern, command).
    public var mainArgument: String {
        guard let object = arguments?.objectValue else { return "" }
        for key in ["path", "file_path", "command", "pattern", "query", "dir", "directory"] {
            if let value = object[key]?.inlineDescription, !value.isEmpty { return value }
        }
        return object.sorted { $0.key < $1.key }.first?.value.inlineDescription ?? ""
    }

    /// All arguments on one line, for the expanded body of the card. Values are
    /// clipped: `write_file` passes the whole file in `content`.
    public var argumentSummary: String {
        guard let object = arguments?.objectValue, !object.isEmpty else { return "" }
        let line = object.sorted { $0.key < $1.key }
            .map { "\($0.key): \(Self.clip($0.value.inlineDescription, 140))" }
            .joined(separator: "  ·  ")
        return Self.clip(line, 400)
    }

    private static func clip(_ text: String, _ limit: Int) -> String {
        text.count <= limit ? text : String(text.prefix(limit)) + "…"
    }

    /// Text shown when the card is expanded: the live output while it runs,
    /// the final output afterwards.
    public var bodyText: String {
        if let output, !output.isEmpty { return output }
        return liveOutput.joined(separator: "\n")
    }
}

public struct CompactionInfo: Equatable, Sendable {
    public var before: Int
    public var after: Int
    public var strategy: String

    public init(before: Int, after: Int, strategy: String) {
        self.before = before
        self.after = after
        self.strategy = strategy
    }

    /// „Kontekst skompaktowany: 12,4k → 5,1k tokenów”
    public var message: String {
        "Kontekst skompaktowany: \(TokenFormat.compact(before)) → \(TokenFormat.compact(after)) tokenów"
    }

    public var strategyLabel: String {
        switch strategy {
        case "mask_tool_results": "ukryto wyniki narzędzi"
        case "summarize": "podsumowano starsze tury"
        case "drop_oldest": "usunięto najstarsze wiadomości"
        default: strategy
        }
    }

    /// Recovers a compaction from the notice the agent stores in the session
    /// file: `Kompaktowanie kontekstu: summarize (7829 → 4020 tokenów).`
    ///
    /// Live turns deliver a `compaction` event, but a reloaded transcript only
    /// has that sentence — and both must look the same in the UI.
    public static func parse(notice: String) -> CompactionInfo? {
        guard notice.hasPrefix("Kompaktowanie kontekstu:") else { return nil }
        let pattern = #"Kompaktowanie kontekstu:\s*([a-z_]+)\s*\((\d+)\s*→\s*(\d+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: notice, range: NSRange(notice.startIndex..., in: notice)),
              match.numberOfRanges == 4,
              let strategyRange = Range(match.range(at: 1), in: notice),
              let beforeRange = Range(match.range(at: 2), in: notice),
              let afterRange = Range(match.range(at: 3), in: notice),
              let before = Int(notice[beforeRange]),
              let after = Int(notice[afterRange]) else { return nil }
        return CompactionInfo(before: before, after: after, strategy: String(notice[strategyRange]))
    }
}

/// Token counts in the Polish decimal notation (`12,4k`).
public enum TokenFormat {
    public static func compact(_ value: Int) -> String {
        if value >= 1_000_000 {
            return String(format: "%.1fM", Double(value) / 1_000_000).replacingOccurrences(of: ".", with: ",")
        }
        if value >= 1_000 {
            return String(format: "%.1fk", Double(value) / 1_000).replacingOccurrences(of: ".", with: ",")
        }
        return "\(value)"
    }
}

// MARK: - Tool presentation

/// Icon and Polish name of every tool in the agent's belt.
public struct ToolPresentation: Sendable {
    public var systemImage: String
    public var label: String

    public static func of(_ name: String) -> ToolPresentation {
        switch name {
        case "read_file": ToolPresentation(systemImage: "doc.text", label: "Odczyt pliku")
        case "write_file": ToolPresentation(systemImage: "square.and.pencil", label: "Zapis pliku")
        case "edit_file": ToolPresentation(systemImage: "pencil", label: "Edycja pliku")
        case "list_dir": ToolPresentation(systemImage: "folder", label: "Lista plików")
        case "glob": ToolPresentation(systemImage: "doc.text.magnifyingglass", label: "Szukanie plików")
        case "grep": ToolPresentation(systemImage: "magnifyingglass", label: "Szukanie w plikach")
        case "bash": ToolPresentation(systemImage: "terminal", label: "Polecenie powłoki")
        case "todo_write": ToolPresentation(systemImage: "checklist", label: "Plan zadań")
        case "delegate": ToolPresentation(systemImage: "person.2", label: "Pomocnik")
        // The help bot's tools: a delete must not look like a harmless read.
        case "model_search": ToolPresentation(systemImage: "magnifyingglass", label: "Szukanie modeli")
        case "model_list": ToolPresentation(systemImage: "shippingbox", label: "Modele na dysku")
        case "model_install": ToolPresentation(systemImage: "arrow.down.circle", label: "Pobieranie")
        case "model_load": ToolPresentation(systemImage: "memorychip", label: "Ładowanie")
        case "model_unload": ToolPresentation(systemImage: "eject", label: "Wyładowanie")
        case "model_delete": ToolPresentation(systemImage: "trash", label: "Usuwanie")
        case "model_settings": ToolPresentation(systemImage: "slider.horizontal.3", label: "Ustawienia modelu")
        case "settings": ToolPresentation(systemImage: "gearshape", label: "Ustawienia")
        case "downloads": ToolPresentation(systemImage: "arrow.down.to.line", label: "Pobierania")
        case "doctor": ToolPresentation(systemImage: "stethoscope", label: "Diagnostyka")
        default: ToolPresentation(systemImage: "wrench.and.screwdriver", label: "Narzędzie")
        }
    }
}

// MARK: - Diffs

/// One line of a unified diff, classified for colouring.
public struct DiffLine: Identifiable, Equatable {
    public enum Kind: Equatable { case added, removed, context, hunk, meta }

    public let id: Int
    public let kind: Kind
    public let text: String

    public init(id: Int, kind: Kind, text: String) {
        self.id = id
        self.kind = kind
        self.text = text
    }

    /// Splits a unified diff into classified lines.
    public static func parse(_ diff: String) -> [DiffLine] {
        diff.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated()
            .map { index, raw in
                let line = String(raw)
                let kind: Kind
                if line.hasPrefix("+++") || line.hasPrefix("---") {
                    kind = .meta
                } else if line.hasPrefix("@@") {
                    kind = .hunk
                } else if line.hasPrefix("+") {
                    kind = .added
                } else if line.hasPrefix("-") {
                    kind = .removed
                } else {
                    kind = .context
                }
                return DiffLine(id: index, kind: kind, text: line)
            }
    }

    /// Cheap check used to decide whether a preview is a diff or a command.
    public static func looksLikeDiff(_ text: String) -> Bool {
        text.hasPrefix("---") || text.hasPrefix("+++") || text.contains("\n@@") || text.hasPrefix("@@")
    }

    /// `+12 / −3` for the header of a diff block.
    public static func counts(_ lines: [DiffLine]) -> (added: Int, removed: Int) {
        (lines.filter { $0.kind == .added }.count, lines.filter { $0.kind == .removed }.count)
    }
}

// MARK: - Reducer

/// Folds the REST history and the live events into the transcript.
///
/// Separate from the store so the same reduction can be replayed against the
/// captured fixtures without SwiftUI.
public struct TranscriptBuilder {
    public private(set) var entries: [TranscriptEntry] = []

    /// Index of the assistant/reasoning block currently being streamed.
    private var assistantIndex: Int?
    private var reasoningIndex: Int?
    /// Cap on the live output kept per running tool — a `find /` would
    /// otherwise grow without bound.
    private let liveOutputLimit = 400

    public init() {}

    public mutating func reset() {
        entries.removeAll()
        assistantIndex = nil
        reasoningIndex = nil
    }

    /// Replaces the transcript with the stored history of a session.
    public mutating func load(items: [AgentItemDTO]) {
        reset()
        for item in items {
            switch item.kind {
            case "user":
                entries.append(TranscriptEntry(id: item.id, kind: .user,
                                               text: item.text ?? "", createdAt: item.createdAt))
            case "assistant":
                entries.append(TranscriptEntry(id: item.id, kind: .assistant,
                                               text: item.text ?? "", createdAt: item.createdAt))
            case "reasoning":
                entries.append(TranscriptEntry(id: item.id, kind: .reasoning,
                                               text: item.text ?? "", createdAt: item.createdAt))
            case "tool":
                let tool = ToolState(callId: item.id,
                                     name: item.toolName ?? "",
                                     title: item.title ?? "",
                                     arguments: item.arguments,
                                     ok: item.ok,
                                     summary: item.summary,
                                     output: item.output,
                                     diff: item.diff)
                var reviewed = tool
                reviewed.review = item.review
                entries.append(TranscriptEntry(id: item.id, kind: .tool,
                                               createdAt: item.createdAt, tool: reviewed))
            default:
                let text = item.text ?? ""
                guard !text.isEmpty else { continue }
                if let compaction = CompactionInfo.parse(notice: text) {
                    entries.append(TranscriptEntry(id: item.id, kind: .compaction,
                                                   createdAt: item.createdAt, compaction: compaction))
                    continue
                }
                entries.append(TranscriptEntry(id: item.id,
                                               kind: item.noticeType == "error" ? .error : .notice,
                                               text: text, createdAt: item.createdAt))
            }
        }
    }

    /// Applies one live event. Returns `true` when the transcript changed.
    @discardableResult
    public mutating func apply(_ event: AgentEvent) -> Bool {
        switch event {
        case let .turnStart(_, userText), let .userMessage(userText):
            closeStreaming()
            entries.append(TranscriptEntry(id: "user-\(UUID().uuidString)", kind: .user, text: userText))
            return true

        case let .textDelta(delta):
            guard !delta.isEmpty else { return false }
            if let index = assistantIndex, entries.indices.contains(index) {
                entries[index].text += delta
            } else {
                entries.append(TranscriptEntry(id: "assistant-\(UUID().uuidString)",
                                               kind: .assistant, text: delta))
                assistantIndex = entries.count - 1
            }
            return true

        case let .reasoningDelta(delta):
            guard !delta.isEmpty else { return false }
            if let index = reasoningIndex, entries.indices.contains(index) {
                entries[index].text += delta
            } else {
                entries.append(TranscriptEntry(id: "reasoning-\(UUID().uuidString)",
                                               kind: .reasoning, text: delta))
                reasoningIndex = entries.count - 1
            }
            return true

        case let .toolCall(call):
            closeStreaming()
            let tool = ToolState(callId: call.callId, name: call.name,
                                 title: call.title, arguments: call.arguments)
            entries.append(TranscriptEntry(id: "tool-\(call.callId)", kind: .tool, tool: tool))
            return true

        case let .toolResult(result):
            closeStreaming()
            if let index = lastToolIndex(callId: result.callId) {
                entries[index].tool?.ok = result.ok
                entries[index].tool?.summary = result.summary
                entries[index].tool?.output = result.output
                entries[index].tool?.diff = result.diff
                if entries[index].tool?.name.isEmpty ?? false { entries[index].tool?.name = result.name }
            } else {
                let tool = ToolState(callId: result.callId, name: result.name, title: "",
                                     ok: result.ok, summary: result.summary,
                                     output: result.output, diff: result.diff)
                entries.append(TranscriptEntry(id: "tool-\(result.callId)", kind: .tool, tool: tool))
            }
            return true

        case let .status(message, phase):
            // Live tool output belongs to the card of the running tool.
            guard phase == .tool, !message.isEmpty, let index = runningToolIndex() else { return false }
            entries[index].tool?.liveOutput.append(message)
            if let count = entries[index].tool?.liveOutput.count, count > liveOutputLimit {
                entries[index].tool?.liveOutput.removeFirst(count - liveOutputLimit)
            }
            return true

        case let .goal(goal):
            closeStreaming()
            entries.append(TranscriptEntry(id: "goal-\(UUID().uuidString)",
                                           kind: ["limit", "stalled"].contains(goal.state) ? .error : .notice,
                                           text: goal.noticeText))
            return true

        case let .compaction(before, after, strategy):
            closeStreaming()
            entries.append(TranscriptEntry(id: "compaction-\(UUID().uuidString)", kind: .compaction,
                                           compaction: CompactionInfo(before: before, after: after,
                                                                      strategy: strategy)))
            return true

        case let .todos(items):
            guard !items.isEmpty else { return false }
            if let last = entries.indices.last, entries[last].kind == .todos {
                guard entries[last].todos != items else { return false }
                entries[last].todos = items
            } else {
                entries.append(TranscriptEntry(id: "todos-\(UUID().uuidString)", kind: .todos, todos: items))
            }
            return true

        case let .subagentStart(payload):
            guard let index = lastToolIndex(callId: payload.parentCallId) else { return false }
            entries[index].tool?.helperRunning = true
            entries[index].tool?.helperRemote = payload.remote
            entries[index].tool?.helperBackground = payload.background
            entries[index].tool?.helperLabel = payload.agentId
            return true

        case let .subagentStep(payload):
            guard let index = lastToolIndex(callId: payload.parentCallId) else { return false }
            entries[index].tool?.helperSteps.append(
                "\(payload.ok ? "✓" : "✗") \(payload.title) — \(payload.summary)")
            return true

        case let .autoReview(callId, review):
            guard let index = lastToolIndex(callId: callId) else { return false }
            entries[index].tool?.review = review
            return true

        case let .subagentEnd(payload):
            guard let index = lastToolIndex(callId: payload.parentCallId) else { return false }
            entries[index].tool?.helperRunning = false
            if !payload.report.isEmpty {
                entries[index].tool?.helperReport = payload.report
            }
            return true

        case let .turnEnd(_, reason, stats):
            closeStreaming()
            entries.append(TranscriptEntry(id: "stats-\(UUID().uuidString)", kind: .stats,
                                           stats: stats, reason: reason))
            return true

        case let .error(message, _):
            closeStreaming()
            entries.append(TranscriptEntry(id: "error-\(UUID().uuidString)", kind: .error, text: message))
            return true

        case let .permissionResolved(_, decision):
            // The card itself is rendered from the store's pending request; the
            // transcript only keeps the trace of what was decided.
            let text: String
            switch decision {
            case "allow": text = "Zezwolono na operację."
            case "always": text = "Zezwolono na operację i zapamiętano decyzję."
            case "deny": text = "Odmówiono wykonania operacji."
            default: return false
            }
            entries.append(TranscriptEntry(id: "decision-\(UUID().uuidString)", kind: .notice, text: text))
            return true

        case let .history(items):
            load(items: items)
            return true

        case .session, .context, .permissionRequest, .ack, .unknown:
            return false
        }
    }

    /// Appends a notice the app itself produced (mode switch, reconnect…).
    public mutating func appendNotice(_ text: String) {
        entries.append(TranscriptEntry(id: "notice-\(UUID().uuidString)", kind: .notice, text: text))
    }

    /// Marks every unfinished tool card as failed — used when the connection
    /// drops mid-turn so no card spins forever.
    public mutating func failRunningTools(summary: String) {
        for index in entries.indices where entries[index].tool?.isRunning == true {
            entries[index].tool?.ok = false
            entries[index].tool?.summary = summary
        }
    }

    private mutating func closeStreaming() {
        assistantIndex = nil
        reasoningIndex = nil
    }

    private func lastToolIndex(callId: String) -> Int? {
        entries.lastIndex { $0.kind == .tool && $0.tool?.callId == callId }
    }

    private func runningToolIndex() -> Int? {
        entries.lastIndex { $0.kind == .tool && $0.tool?.isRunning == true }
    }
}

extension TranscriptBuilder {
    /// Puts the thumbnails of the pictures just sent on the newest user message.
    public mutating func attachImages(_ thumbnails: [Data]) {
        guard !thumbnails.isEmpty, let index = entries.lastIndex(where: { $0.kind == .user }) else { return }
        entries[index].images = thumbnails
    }
}

