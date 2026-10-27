import Foundation
import Observation
import LocalCodeKit

/// The one source of truth for every chat screen.
///
/// Ported from the web chat app's `AppStore` with the account layer removed:
/// there is no sign-in, no session restore and no 401 handling, because the
/// server runs on this machine and the app is handed its token at launch.
/// Everything the chat UI renders lives here; views never touch `APIClient`
/// directly, which keeps optimistic updates and their rollbacks in one place.
///
/// Two things are new relative to the web app, both because the model is local:
///
///  * `statusMessage` — loading a GGUF/MLX model takes seconds, and the server
///    says what it is doing ("Ładowanie modelu…") before the first token
///    exists. It is shown inside the assistant bubble and cleared by the first
///    piece of text.
///  * cost is always zero, so every cost readout in the UI hides itself.
@MainActor
@Observable
public final class ChatStore {

    // MARK: - Dependencies

    @ObservationIgnored public let api: APIClient

    public init(api: APIClient) {
        self.api = api
    }

    // MARK: - Data

    public var settings: AppSettings = AppSettings()
    public var folders: [Folder] = []
    public var conversations: [Conversation] = []
    public var models: [ModelInfo] = []
    public var providers: [ProviderInfo] = [.local]
    public var activeConversation: Conversation?
    public var messages: [Message] = []

    // MARK: - Transient state

    /// False until the first `/api/bootstrap` answers.
    public var isLoaded = false
    /// True only while the *currently open* conversation is receiving a stream.
    public var isStreaming = false
    /// Id of the assistant message being filled in, when it belongs to the open conversation.
    public var streamingMessageId: String?
    /// What the server is doing before the first token ("Ładowanie modelu…").
    public var statusMessage: String?
    /// Shown as a banner; the UI clears it.
    public var errorMessage: String?
    /// Project the next new conversation will be created in.
    public var draftFolderId: String?
    /// True while a conversation's history is being fetched.
    public var isLoadingMessages = false
    /// Provider the server will use by default (from `/api/bootstrap`).
    public var activeProvider = "local"
    /// How much of the model's window the last turn took, from the `context`
    /// frame. Drives the gauge in the chat header; empty before the first turn.
    public var contextUsage = ChatContextUsage()
    /// Compactions that happened in the open conversation, keyed by the message
    /// they arrived before, so the transcript can show a notice in the right
    /// place. Cleared with the conversation.
    public private(set) var compactions: [ChatCompaction] = []
    /// True while `POST /api/conversations/{id}/compact` is running.
    public var isCompacting = false
    /// A short sentence that fades by itself: the result of "Skróć rozmowę", or
    /// a `status` frame that reported something went wrong rather than
    /// something being in progress.
    public private(set) var notice: String?

    // MARK: - Stream bookkeeping (not observed)

    /// Context of one streaming turn, so late events from an aborted turn can be ignored.
    private struct StreamContext: Sendable {
        let token: UUID
        let pendingUserMessageId: String?
        let startsNewConversation: Bool
        let folderId: String?
    }

    @ObservationIgnored private var streamTask: Task<Void, Never>?
    @ObservationIgnored private var streamToken: UUID?
    @ObservationIgnored private var streamRunning = false
    @ObservationIgnored private var streamConversationId: String?
    @ObservationIgnored private var streamAssistantId: String?
    @ObservationIgnored private var streamContent = ""
    @ObservationIgnored private var streamReasoning = ""
    /// Guards against an older `openConversation` finishing after a newer one.
    @ObservationIgnored private var openToken = UUID()
    @ObservationIgnored private var didStart = false

    // MARK: - Derived helpers

    /// Provider that the next turn will use.
    public var currentProvider: String {
        let fromConversation = activeConversation?.provider ?? ""
        if !fromConversation.isEmpty { return fromConversation }
        if !settings.defaults.provider.isEmpty { return settings.defaults.provider }
        return activeProvider
    }

    /// Model key that the next turn will use.
    public var currentModel: String {
        let fromConversation = activeConversation?.model ?? ""
        if !fromConversation.isEmpty { return fromConversation }
        return settings.defaults.model
    }

    public var currentModelInfo: ModelInfo? {
        models.first { $0.id == currentModel && $0.provider == currentProvider }
            ?? models.first { $0.id == currentModel }
    }

    /// True while a stream is running for a conversation other than the open one.
    public var isStreamingInBackground: Bool {
        streamRunning && streamConversationId != activeConversation?.id
    }

    /// The store refuses a new turn while *any* conversation is streaming.
    public var isBusy: Bool { isStreaming || isStreamingInBackground }

    public var activeFolder: Folder? {
        guard let id = activeConversation?.folderId ?? draftFolderId else { return nil }
        return folders.first { $0.id == id }
    }

    /// Pretty label for a model key, falling back to the key itself.
    public func label(forModel model: String?, provider: String? = nil) -> String? {
        guard let model, !model.isEmpty else { return nil }
        let match = models.first { $0.id == model && $0.provider == (provider ?? $0.provider) }
            ?? models.first { $0.id == model }
        return match?.label ?? model
    }

    // MARK: - Loading

    /// Called once by the shell after the server is reachable.
    public func start() async {
        guard !didStart else { return }
        didStart = true
        await loadBootstrap()
        isLoaded = true
    }

    public func refreshAll() async {
        await loadBootstrap()
        if let id = activeConversation?.id, !streamRunning {
            await reloadMessages(of: id)
        }
    }

    private func loadBootstrap() async {
        do {
            let data = try await api.bootstrap()
            settings = data.settings
            folders = data.folders.sorted(by: Self.folderOrder)
            conversations = data.conversations
            sortConversationList()
            providers = data.providers.isEmpty ? [.local] : data.providers
            models = data.models
            activeProvider = data.activeProvider
            // A brand-new install has no default model; take the first one the
            // server offers so the composer is usable straight away.
            if settings.defaults.model.isEmpty, let first = models.first {
                settings.defaults.model = first.id
                settings.defaults.provider = first.provider
            }
            errorMessage = nil
        } catch {
            report(error)
        }
    }

    /// Re-reads the model list — the models section may have loaded or deleted
    /// something since bootstrap.
    public func refreshModels() async {
        do {
            models = try await api.models()
        } catch {
            report(error)
        }
    }

    public func refreshConversations() async {
        do {
            let list = try await api.conversations()
            conversations = list
            sortConversationList()
            if let active = activeConversation, let fresh = list.first(where: { $0.id == active.id }) {
                activeConversation = fresh
            }
        } catch {
            report(error)
        }
    }

    // MARK: - Conversation navigation

    /// Opens a conversation, or clears the view when `id` is nil.
    public func openConversation(_ id: String?) async {
        let token = UUID()
        openToken = token

        guard let id else {
            activeConversation = nil
            messages = []
            isLoadingMessages = false
            resetContextState()
            syncStreamingFlags()
            return
        }

        if activeConversation?.id != id {
            messages = []
            resetContextState()
            activeConversation = conversations.first { $0.id == id }
        }
        syncStreamingFlags()
        isLoadingMessages = true

        do {
            let detail = try await api.conversation(id: id)
            guard openToken == token else { return }
            upsertConversation(detail.conversation)
            activeConversation = conversations.first { $0.id == id } ?? detail.conversation
            messages = detail.messages
            restoreStreamingPlaceholder()
            syncStreamingFlags()
            isLoadingMessages = false
        } catch {
            guard openToken == token else { return }
            isLoadingMessages = false
            report(error)
        }
    }

    /// The context gauge and the compaction notices belong to one conversation
    /// and must not follow the user to the next one.
    private func resetContextState() {
        contextUsage = ChatContextUsage()
        compactions = []
    }

    /// Clears the composer target so the next `send` creates a conversation.
    public func newConversation(folderId: String?) {
        openToken = UUID()
        activeConversation = nil
        messages = []
        resetContextState()
        draftFolderId = folderId
        errorMessage = nil
        isLoadingMessages = false
        syncStreamingFlags()
    }

    private func reloadMessages(of id: String) async {
        do {
            let detail = try await api.conversation(id: id)
            guard activeConversation?.id == id else { return }
            upsertConversation(detail.conversation)
            messages = detail.messages
            restoreStreamingPlaceholder()
        } catch {
            report(error)
        }
    }

    // MARK: - Sending

    /// Sends one turn. `attachments` only renders the user's own bubble before
    /// the server echoes the stored message back.
    public func send(text content: String, attachments: [Attachment] = []) async {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        let ids = attachments.map(\.id)
        guard !trimmed.isEmpty || !ids.isEmpty else { return }
        guard !streamRunning else { return }

        errorMessage = nil
        let conversationId = activeConversation?.id
        let folderId = activeConversation?.folderId ?? draftFolderId
        let localId = "local-\(UUID().uuidString)"

        messages.append(
            Message(id: localId,
                    conversationId: conversationId ?? "",
                    role: .user,
                    content: trimmed,
                    status: .complete,
                    seq: (messages.last?.seq ?? 0) + 1,
                    createdAt: Date().timeIntervalSince1970 * 1000,
                    attachments: attachments)
        )

        let request = ChatSendRequest(conversationId: conversationId,
                                      folderId: conversationId == nil ? folderId : nil,
                                      content: trimmed,
                                      attachmentIds: ids.isEmpty ? nil : ids,
                                      provider: nilIfEmpty(currentProvider),
                                      model: nilIfEmpty(currentModel),
                                      reasoningEffort: conversationId == nil ? pendingEffort : nil)
        let context = StreamContext(token: UUID(),
                                    pendingUserMessageId: localId,
                                    startsNewConversation: conversationId == nil,
                                    folderId: folderId)
        await runStream(request, context: context)
    }

    /// Re-runs the last user turn.
    public func regenerate() async {
        guard !streamRunning, let conversation = activeConversation else { return }
        guard let lastUser = messages.last(where: { $0.role == .user }) else { return }
        errorMessage = nil

        if let index = messages.lastIndex(where: { $0.id == lastUser.id }), index + 1 < messages.count {
            messages.removeSubrange((index + 1)...)
        }

        let request = ChatSendRequest(conversationId: conversation.id,
                                      content: lastUser.content,
                                      provider: nilIfEmpty(currentProvider),
                                      model: nilIfEmpty(currentModel),
                                      regenerate: true)
        let context = StreamContext(token: UUID(),
                                    pendingUserMessageId: lastUser.id,
                                    startsNewConversation: false,
                                    folderId: conversation.folderId)
        await runStream(request, context: context)
    }

    /// Rewrites a user message and streams a fresh answer for it.
    public func edit(messageId: String, text newText: String) async {
        guard !streamRunning, let conversation = activeConversation else { return }
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        errorMessage = nil

        if let index = messages.firstIndex(where: { $0.id == messageId }) {
            messages[index].content = trimmed
            if index + 1 < messages.count {
                messages.removeSubrange((index + 1)...)
            }
        }

        let request = ChatSendRequest(conversationId: conversation.id,
                                      content: trimmed,
                                      provider: nilIfEmpty(currentProvider),
                                      model: nilIfEmpty(currentModel),
                                      editMessageId: messageId)
        let context = StreamContext(token: UUID(),
                                    pendingUserMessageId: messageId,
                                    startsNewConversation: false,
                                    folderId: conversation.folderId)
        await runStream(request, context: context)
    }

    /// Aborts the request; the server stores the partial answer as `stopped`.
    public func stopStreaming() {
        guard streamRunning else { return }
        streamRunning = false
        isStreaming = false
        statusMessage = nil
        cancelStream()
    }

    private func cancelStream() {
        streamTask?.cancel()
        streamTask = nil
    }

    // MARK: - Stream plumbing

    private func runStream(_ request: ChatSendRequest, context: StreamContext) async {
        streamToken = context.token
        streamRunning = true
        streamConversationId = request.conversationId
        streamAssistantId = nil
        streamContent = ""
        streamReasoning = ""
        isStreaming = true
        streamingMessageId = nil
        statusMessage = nil

        let task = Task { [self] in
            do {
                let stream = try await api.chat(request)
                for try await event in stream {
                    if Task.isCancelled { break }
                    reduce(event, context: context)
                }
                finishStream(context: context, cancelled: Task.isCancelled)
            } catch is CancellationError {
                finishStream(context: context, cancelled: true)
            } catch {
                failStream(context: context, error: error)
            }
        }
        streamTask = task
        await task.value
    }

    /// True when the running stream belongs to the conversation on screen.
    private var streamMatchesActiveConversation: Bool {
        streamConversationId != nil && streamConversationId == activeConversation?.id
    }

    private func reduce(_ event: ChatStreamEvent, context: StreamContext) {
        guard context.token == streamToken else { return }

        switch event {
        case let .start(conversationId, userMessage, assistantMessageId):
            streamConversationId = conversationId
            streamAssistantId = assistantMessageId
            adoptConversation(id: conversationId, context: context)
            if streamMatchesActiveConversation {
                applyStart(userMessage: userMessage, assistantId: assistantMessageId, context: context)
            }
            syncStreamingFlags()

        case let .title(conversationId, title):
            if let index = conversations.firstIndex(where: { $0.id == conversationId }) {
                conversations[index].title = title
            }
            if activeConversation?.id == conversationId {
                activeConversation?.title = title
            }

        case let .citations(citations):
            mutateStreamingMessage { $0.citations = citations }

        case let .status(text):
            guard !text.isEmpty else {
                statusMessage = nil
                return
            }
            // The server uses one frame for two things. "Ładowanie modelu…",
            // "Kompaktuję rozmowę…" and "Przeszukiwanie bazy wiedzy…" say what
            // is happening right now and belong in the assistant bubble, and
            // they all end in an ellipsis. Anything else is a report about what
            // had to be given up ("Nie udało się streścić rozmowy — …",
            // "Kontekst modelu (…): pominięto …") and belongs on screen as a
            // notice the reader can actually act on.
            if text.hasSuffix("…") {
                statusMessage = text
            } else {
                announce(text)
            }

        case let .reasoning(delta):
            guard !delta.isEmpty else { return }
            statusMessage = nil
            streamReasoning += delta
            mutateStreamingMessage { $0.reasoning = ($0.reasoning ?? "") + delta }

        case let .text(delta):
            guard !delta.isEmpty else { return }
            statusMessage = nil
            streamContent += delta
            mutateStreamingMessage { $0.content += delta }

        case let .usage(tokensIn, tokensOut, costUsd):
            mutateStreamingMessage {
                $0.tokensIn = tokensIn ?? $0.tokensIn
                $0.tokensOut = tokensOut ?? $0.tokensOut
                $0.costUsd = costUsd ?? $0.costUsd
            }

        case let .compaction(info):
            // The server only sends this when it actually dropped something.
            guard streamMatchesActiveConversation else { return }
            compactions.append(info)

        case let .context(usage):
            guard streamMatchesActiveConversation else { return }
            contextUsage = usage

        case let .done(message):
            statusMessage = nil
            applyDone(message)

        case let .failure(reason):
            statusMessage = nil
            errorMessage = reason
            mutateStreamingMessage {
                $0.status = .error
                $0.error = reason
            }
        }
    }

    /// A brand-new conversation exists only after the `start` frame.
    private func adoptConversation(id: String, context: StreamContext) {
        if !conversations.contains(where: { $0.id == id }) {
            let now = Date().timeIntervalSince1970 * 1000
            let conversation = Conversation(id: id,
                                            folderId: context.folderId,
                                            title: "Nowa rozmowa",
                                            provider: nilIfEmpty(currentProvider),
                                            model: nilIfEmpty(currentModel),
                                            createdAt: now,
                                            updatedAt: now,
                                            lastMessageAt: now,
                                            folderName: folders.first { $0.id == context.folderId }?.name)
            conversations.append(conversation)
            sortConversationList()
        }
        // Only take over the screen if the user has not navigated away meanwhile.
        if context.startsNewConversation, activeConversation == nil {
            activeConversation = conversations.first { $0.id == id }
            draftFolderId = nil
        }
    }

    private func applyStart(userMessage: Message, assistantId: String, context: StreamContext) {
        var anchor: Int
        if let index = messages.firstIndex(where: { $0.id == userMessage.id }) {
            messages[index] = userMessage
            anchor = index
        } else if let pending = context.pendingUserMessageId,
                  let index = messages.firstIndex(where: { $0.id == pending }) {
            messages[index] = userMessage
            anchor = index
        } else {
            messages.append(userMessage)
            anchor = messages.count - 1
        }
        // Drop the stale answer when regenerating or editing.
        if anchor + 1 < messages.count {
            messages.removeSubrange((anchor + 1)...)
        }
        messages.append(
            Message(id: assistantId,
                    conversationId: userMessage.conversationId,
                    role: .assistant,
                    content: "",
                    provider: nilIfEmpty(currentProvider),
                    model: nilIfEmpty(currentModel),
                    status: .streaming,
                    seq: userMessage.seq + 1,
                    createdAt: Date().timeIntervalSince1970 * 1000)
        )
    }

    private func applyDone(_ message: Message) {
        streamContent = message.content
        if let reasoning = message.reasoning { streamReasoning = reasoning }

        if streamMatchesActiveConversation {
            if let index = messages.firstIndex(where: { $0.id == message.id }) {
                messages[index] = message
            } else if let assistantId = streamAssistantId,
                      let index = messages.firstIndex(where: { $0.id == assistantId }) {
                messages[index] = message
            } else {
                messages.append(message)
            }
        }
        streamAssistantId = message.id

        if let index = conversations.firstIndex(where: { $0.id == message.conversationId }) {
            conversations[index].lastMessageAt = message.createdAt
            conversations[index].updatedAt = message.createdAt
            conversations[index].preview = String(message.content.prefix(160))
            conversations[index].tokensIn += message.tokensIn ?? 0
            conversations[index].tokensOut += message.tokensOut ?? 0
            conversations[index].costUsd += message.costUsd ?? 0
            let updated = conversations[index]
            sortConversationList()
            if activeConversation?.id == updated.id { activeConversation = updated }
        }
    }

    private func finishStream(context: StreamContext, cancelled: Bool) {
        guard context.token == streamToken else { return }
        streamRunning = false
        isStreaming = false
        streamingMessageId = nil
        statusMessage = nil
        streamTask = nil

        if let assistantId = streamAssistantId,
           let index = messages.firstIndex(where: { $0.id == assistantId }),
           messages[index].status == .streaming {
            messages[index].status = cancelled ? .stopped : .complete
        }

        let conversationId = streamConversationId
        Task { [weak self] in
            guard let self else { return }
            await self.refreshConversations()
            if cancelled, let conversationId, self.activeConversation?.id == conversationId, !self.streamRunning {
                await self.reloadMessages(of: conversationId)
            }
        }
    }

    private func failStream(context: StreamContext, error: Error) {
        guard context.token == streamToken else { return }
        streamRunning = false
        isStreaming = false
        streamingMessageId = nil
        statusMessage = nil
        streamTask = nil

        let reason = APIError.describe(error)
        if let assistantId = streamAssistantId,
           let index = messages.firstIndex(where: { $0.id == assistantId }) {
            messages[index].status = .error
            messages[index].error = reason
        } else if let pending = context.pendingUserMessageId,
                  let index = messages.firstIndex(where: { $0.id == pending }) {
            messages[index].status = .error
            messages[index].error = reason
        }
        report(error)
    }

    private func mutateStreamingMessage(_ change: (inout Message) -> Void) {
        guard streamMatchesActiveConversation,
              let assistantId = streamAssistantId,
              let index = messages.firstIndex(where: { $0.id == assistantId })
        else { return }
        change(&messages[index])
    }

    /// Keeps the two published streaming flags scoped to the open conversation.
    private func syncStreamingFlags() {
        let mine = streamRunning && streamMatchesActiveConversation
        isStreaming = mine
        streamingMessageId = mine ? streamAssistantId : nil
        if !mine { statusMessage = nil }
    }

    /// Re-attaches the in-flight answer after the history was re-fetched.
    private func restoreStreamingPlaceholder() {
        guard streamRunning, streamMatchesActiveConversation, let assistantId = streamAssistantId else { return }
        if let index = messages.firstIndex(where: { $0.id == assistantId }) {
            messages[index].content = streamContent
            if !streamReasoning.isEmpty { messages[index].reasoning = streamReasoning }
            messages[index].status = .streaming
        } else {
            messages.append(
                Message(id: assistantId,
                        conversationId: activeConversation?.id ?? "",
                        role: .assistant,
                        content: streamContent,
                        reasoning: streamReasoning.isEmpty ? nil : streamReasoning,
                        provider: nilIfEmpty(currentProvider),
                        model: nilIfEmpty(currentModel),
                        status: .streaming,
                        seq: (messages.last?.seq ?? 0) + 1,
                        createdAt: Date().timeIntervalSince1970 * 1000)
            )
        }
    }

    // MARK: - Messages

    public func deleteMessage(_ id: String) async {
        do {
            try await api.deleteMessage(id: id)
            messages.removeAll { $0.id == id }
        } catch {
            report(error)
        }
    }

    // MARK: - Transient notices

    @ObservationIgnored private var noticeTask: Task<Void, Never>?

    /// Shows a sentence for a few seconds and then clears it.
    public func announce(_ text: String) {
        guard !text.isEmpty else { return }
        notice = text
        noticeTask?.cancel()
        noticeTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 6_000_000_000)
            guard !Task.isCancelled else { return }
            self?.notice = nil
        }
    }

    public func dismissNotice() {
        noticeTask?.cancel()
        notice = nil
    }

    // MARK: - Conversations

    public func updateConversation(_ id: String, _ input: ConversationInput) async {
        do {
            let updated = try await api.updateConversation(id: id, input)
            upsertConversation(updated)
        } catch {
            report(error)
        }
    }

    // MARK: - Reasoning effort

    /// The levels the model of the open conversation can express. Empty means
    /// the model has no reasoning switch and the control stays hidden.
    public var effortLevels: [String] { currentModelInfo?.effortLevels ?? [] }

    /// What the "Myślenie" control shows: the conversation's own level, or the
    /// global chat default when it has none of its own.
    public var effectiveEffort: String? {
        activeConversation?.reasoningEffort ?? defaultEffort
    }

    /// `app-settings.chat.reasoningEffort`, pushed in by the shell — `ChatStore`
    /// reads `/api/bootstrap`, which does not carry the application settings.
    public var defaultEffort: String?

    /// Sets the level for the open conversation. Without one open, the choice
    /// is remembered and applied to the conversation the next turn creates.
    public func setEffort(_ level: String) async {
        guard let id = activeConversation?.id else {
            pendingEffort = level
            return
        }
        activeConversation?.reasoningEffort = level
        if let index = conversations.firstIndex(where: { $0.id == id }) {
            conversations[index].reasoningEffort = level
        }
        await updateConversation(id, ConversationInput(reasoningEffort: level))
    }

    /// Chosen before the conversation existed; sent with its first turn.
    @ObservationIgnored private var pendingEffort: String?

    // MARK: - Compaction

    /// `POST /api/conversations/{id}/compact` — the "Skróć rozmowę" action.
    ///
    /// The server answers with what it did; a conversation that was already
    /// short enough reports `changed: false`, and the UI says so instead of
    /// pretending the button did something.
    @discardableResult
    public func compactConversation() async -> ConversationCompaction? {
        guard let id = activeConversation?.id, !isCompacting, !isBusy else { return nil }
        isCompacting = true
        defer { isCompacting = false }
        do {
            let result = try await api.compactConversation(id: id)
            if result.changed {
                compactions.append(ChatCompaction(before: result.before,
                                                  after: result.after,
                                                  strategy: result.strategy))
                await reloadMessages(of: id)
            }
            return result
        } catch {
            report(error)
            return nil
        }
    }

    /// Takes a conversation out of its project; needs an explicit JSON null.
    public func detachConversation(_ id: String) async {
        do {
            let updated = try await api.detachConversationFromFolder(id: id)
            upsertConversation(updated)
        } catch {
            report(error)
        }
    }

    public func deleteConversation(_ id: String) async {
        do {
            try await api.deleteConversation(id: id)
            conversations.removeAll { $0.id == id }
            if streamConversationId == id { stopStreaming() }
            if activeConversation?.id == id {
                activeConversation = nil
                messages = []
                syncStreamingFlags()
            }
        } catch {
            report(error)
        }
    }

    public func togglePin(_ id: String) async {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let value = !conversations[index].pinned
        conversations[index].pinned = value
        if activeConversation?.id == id { activeConversation?.pinned = value }
        sortConversationList()
        do {
            let updated = try await api.updateConversation(id: id, ConversationInput(pinned: value))
            upsertConversation(updated)
        } catch {
            if let revert = conversations.firstIndex(where: { $0.id == id }) {
                conversations[revert].pinned = !value
            }
            if activeConversation?.id == id { activeConversation?.pinned = !value }
            sortConversationList()
            report(error)
        }
    }

    public func exportConversation(_ id: String, format: String) async -> Data? {
        do {
            return try await api.exportConversation(id: id, format: format)
        } catch {
            report(error)
            return nil
        }
    }

    // MARK: - Folders

    @discardableResult
    public func createFolder(_ input: FolderInput) async -> Folder? {
        do {
            let folder = try await api.createFolder(input)
            folders.append(folder)
            folders.sort(by: Self.folderOrder)
            return folder
        } catch {
            report(error)
            return nil
        }
    }

    @discardableResult
    public func updateFolder(_ id: String, _ input: FolderInput) async -> Folder? {
        do {
            let folder = try await api.updateFolder(id: id, input)
            if let index = folders.firstIndex(where: { $0.id == id }) {
                // The PATCH response omits the counters the list endpoint adds.
                var merged = folder
                merged.conversationCount = folder.conversationCount ?? folders[index].conversationCount
                merged.documentCount = folder.documentCount ?? folders[index].documentCount
                merged.memoryCount = folder.memoryCount ?? folders[index].memoryCount
                folders[index] = merged
            } else {
                folders.append(folder)
            }
            folders.sort(by: Self.folderOrder)
            for index in conversations.indices where conversations[index].folderId == id {
                conversations[index].folderName = folder.name
            }
            if activeConversation?.folderId == id { activeConversation?.folderName = folder.name }
            return folder
        } catch {
            report(error)
            return nil
        }
    }

    public func deleteFolder(_ id: String, deleteChats: Bool) async {
        do {
            try await api.deleteFolder(id: id, deleteChats: deleteChats)
            folders.removeAll { $0.id == id }
            if deleteChats {
                let removed = Set(conversations.filter { $0.folderId == id }.map(\.id))
                conversations.removeAll { removed.contains($0.id) }
                if let active = activeConversation, removed.contains(active.id) {
                    activeConversation = nil
                    messages = []
                    syncStreamingFlags()
                }
            } else {
                for index in conversations.indices where conversations[index].folderId == id {
                    conversations[index].folderId = nil
                    conversations[index].folderName = nil
                }
                if activeConversation?.folderId == id {
                    activeConversation?.folderId = nil
                    activeConversation?.folderName = nil
                }
            }
            if draftFolderId == id { draftFolderId = nil }
        } catch {
            report(error)
        }
    }

    // MARK: - Settings and models

    public func patchSettings(_ newSettings: AppSettings) async {
        let previous = settings
        settings = newSettings
        do {
            settings = try await api.patchSettings(newSettings)
        } catch {
            settings = previous
            report(error)
        }
    }

    /// Applies the pick to the open conversation and remembers it for new ones.
    public func selectModel(provider: String, model: String) async {
        guard !provider.isEmpty, !model.isEmpty else { return }
        if let conversation = activeConversation {
            await updateConversation(conversation.id, ConversationInput(provider: provider, model: model))
        }
        guard settings.defaults.provider != provider || settings.defaults.model != model else { return }
        var next = settings
        next.defaults.provider = provider
        next.defaults.model = model
        await patchSettings(next)
    }

    // MARK: - Bookkeeping helpers

    private func upsertConversation(_ conversation: Conversation) {
        var incoming = conversation
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            let old = conversations[index]
            // Fields only present on the list endpoint survive a detail refresh.
            incoming.preview = incoming.preview ?? old.preview
            incoming.messageCount = incoming.messageCount ?? old.messageCount
            incoming.folderName = incoming.folderName ?? old.folderName
            conversations[index] = incoming
        } else {
            if incoming.folderName == nil, let folderId = incoming.folderId {
                incoming.folderName = folders.first { $0.id == folderId }?.name
            }
            conversations.append(incoming)
        }
        sortConversationList()
        if activeConversation?.id == incoming.id { activeConversation = incoming }
    }

    private func sortConversationList() {
        conversations.sort { lhs, rhs in
            if lhs.pinned != rhs.pinned { return lhs.pinned }
            return lhs.lastMessageAt > rhs.lastMessageAt
        }
    }

    private nonisolated static func folderOrder(_ lhs: Folder, _ rhs: Folder) -> Bool {
        lhs.sortOrder == rhs.sortOrder ? lhs.name < rhs.name : lhs.sortOrder < rhs.sortOrder
    }

    private func nilIfEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }

    /// One place that decides whether an error reaches the banner.
    private func report(_ error: Error) {
        if error is CancellationError { return }
        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled { return }
        errorMessage = APIError.describe(error)
    }
}
