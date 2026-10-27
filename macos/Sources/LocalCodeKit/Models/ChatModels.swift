import Foundation

/// The DTOs of LocalCode's chat API.
///
/// They mirror the JSON the Python server emits one field at a time, so no
/// `CodingKeys` are needed anywhere: the wire format is camelCase and
/// timestamps are milliseconds since the epoch, exactly like the web chat app
/// this UX was ported from.
///
/// Two deliberate differences from that app:
///
///  * there are no accounts, so no `User`, no session and no auth responses —
///    LocalCode talks to a server that runs on this machine, authenticated by a
///    bearer token generated at launch;
///  * "providers" collapse to the single local provider, and prices are always
///    `nil`, so every cost readout in the UI hides itself.
///
/// Decoding is deliberately forgiving for the payloads that carry settings and
/// capabilities (`Bootstrap`, `AppSettings`, `ProviderInfo`, `ModelInfo`): a
/// server that gains or loses a field must not take the window down with it.

// MARK: - Folders (projects)

public enum MemoryMode: String, Codable, Sendable, CaseIterable {
    case shared
    case isolated

    public var title: String {
        switch self {
        case .shared: "Pamięć wspólna"
        case .isolated: "Pamięć izolowana"
        }
    }

    public var explanation: String {
        switch self {
        case .shared:
            "Rozmowy w tym projekcie widzą kontekst globalny (prompt, informacje o Tobie, pamięć, pliki) ORAZ dodatkowo kontekst tego projektu."
        case .isolated:
            "Rozmowy w tym projekcie NIE widzą niczego z głównego kontekstu. Działają wyłącznie na instrukcjach, danych i plikach tego projektu."
        }
    }
}

public struct Folder: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var name: String
    public var description: String?
    public var color: String?
    public var icon: String?
    public var systemPrompt: String?
    public var extraInfo: String?
    public var memoryMode: MemoryMode
    public var sortOrder: Int
    public var createdAt: Double
    public var updatedAt: Double
    /// Present on `/api/folders`, absent elsewhere.
    public var conversationCount: Int?
    public var documentCount: Int?
    public var memoryCount: Int?

    public init(id: String,
                name: String,
                description: String? = nil,
                color: String? = nil,
                icon: String? = nil,
                systemPrompt: String? = nil,
                extraInfo: String? = nil,
                memoryMode: MemoryMode = .shared,
                sortOrder: Int = 0,
                createdAt: Double = 0,
                updatedAt: Double = 0,
                conversationCount: Int? = nil,
                documentCount: Int? = nil,
                memoryCount: Int? = nil) {
        self.id = id
        self.name = name
        self.description = description
        self.color = color
        self.icon = icon
        self.systemPrompt = systemPrompt
        self.extraInfo = extraInfo
        self.memoryMode = memoryMode
        self.sortOrder = sortOrder
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.conversationCount = conversationCount
        self.documentCount = documentCount
        self.memoryCount = memoryCount
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "Projekt"
        description = try c.decodeIfPresent(String.self, forKey: .description)
        color = try c.decodeIfPresent(String.self, forKey: .color)
        icon = try c.decodeIfPresent(String.self, forKey: .icon)
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        extraInfo = try c.decodeIfPresent(String.self, forKey: .extraInfo)
        memoryMode = try c.decodeIfPresent(MemoryMode.self, forKey: .memoryMode) ?? .shared
        sortOrder = try c.decodeIfPresent(Int.self, forKey: .sortOrder) ?? 0
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt) ?? createdAt
        conversationCount = try c.decodeIfPresent(Int.self, forKey: .conversationCount)
        documentCount = try c.decodeIfPresent(Int.self, forKey: .documentCount)
        memoryCount = try c.decodeIfPresent(Int.self, forKey: .memoryCount)
    }
}

/// PATCH/POST body for a project. Nil fields are dropped by the encoder, which
/// the server reads as "leave unchanged".
public struct FolderInput: Codable, Sendable {
    public var name: String?
    public var description: String?
    public var color: String?
    public var systemPrompt: String?
    public var extraInfo: String?
    public var memoryMode: MemoryMode?

    public init(name: String? = nil,
                description: String? = nil,
                color: String? = nil,
                systemPrompt: String? = nil,
                extraInfo: String? = nil,
                memoryMode: MemoryMode? = nil) {
        self.name = name
        self.description = description
        self.color = color
        self.systemPrompt = systemPrompt
        self.extraInfo = extraInfo
        self.memoryMode = memoryMode
    }
}

// MARK: - Conversations and messages

public struct Conversation: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var folderId: String?
    public var title: String
    public var titleLocked: Bool
    public var provider: String?
    public var model: String?
    public var systemPrompt: String?
    public var temperature: Double?
    public var maxTokens: Int?
    public var pinned: Bool
    public var archived: Bool
    public var tokensIn: Int
    public var tokensOut: Int
    public var costUsd: Double
    public var createdAt: Double
    public var updatedAt: Double
    public var lastMessageAt: Double
    /// Only on the list endpoint.
    public var preview: String?
    public var messageCount: Int?
    public var folderName: String?
    /// Canonical reasoning effort for this conversation (`off|low|medium|high`).
    /// `nil` means "use the global default from `app-settings.chat`".
    public var reasoningEffort: String?

    public init(id: String,
                folderId: String? = nil,
                title: String = "Nowa rozmowa",
                titleLocked: Bool = false,
                provider: String? = nil,
                model: String? = nil,
                systemPrompt: String? = nil,
                temperature: Double? = nil,
                maxTokens: Int? = nil,
                pinned: Bool = false,
                archived: Bool = false,
                tokensIn: Int = 0,
                tokensOut: Int = 0,
                costUsd: Double = 0,
                createdAt: Double = 0,
                updatedAt: Double = 0,
                lastMessageAt: Double = 0,
                preview: String? = nil,
                messageCount: Int? = nil,
                folderName: String? = nil,
                reasoningEffort: String? = nil) {
        self.id = id
        self.folderId = folderId
        self.title = title
        self.titleLocked = titleLocked
        self.provider = provider
        self.model = model
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.pinned = pinned
        self.archived = archived
        self.tokensIn = tokensIn
        self.tokensOut = tokensOut
        self.costUsd = costUsd
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.lastMessageAt = lastMessageAt
        self.preview = preview
        self.messageCount = messageCount
        self.folderName = folderName
        self.reasoningEffort = reasoningEffort
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        folderId = try c.decodeIfPresent(String.self, forKey: .folderId)
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? "Nowa rozmowa"
        titleLocked = try c.decodeIfPresent(Bool.self, forKey: .titleLocked) ?? false
        provider = try c.decodeIfPresent(String.self, forKey: .provider)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
        pinned = try c.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        archived = try c.decodeIfPresent(Bool.self, forKey: .archived) ?? false
        tokensIn = try c.decodeIfPresent(Int.self, forKey: .tokensIn) ?? 0
        tokensOut = try c.decodeIfPresent(Int.self, forKey: .tokensOut) ?? 0
        costUsd = try c.decodeIfPresent(Double.self, forKey: .costUsd) ?? 0
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt) ?? createdAt
        lastMessageAt = try c.decodeIfPresent(Double.self, forKey: .lastMessageAt) ?? updatedAt
        preview = try c.decodeIfPresent(String.self, forKey: .preview)
        messageCount = try c.decodeIfPresent(Int.self, forKey: .messageCount)
        folderName = try c.decodeIfPresent(String.self, forKey: .folderName)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
    }
}

public struct ConversationInput: Codable, Sendable {
    public var folderId: String?
    public var title: String?
    public var provider: String?
    public var model: String?
    public var systemPrompt: String?
    public var temperature: Double?
    public var maxTokens: Int?
    public var pinned: Bool?
    public var archived: Bool?
    public var reasoningEffort: String?

    public init(folderId: String? = nil,
                title: String? = nil,
                provider: String? = nil,
                model: String? = nil,
                systemPrompt: String? = nil,
                temperature: Double? = nil,
                maxTokens: Int? = nil,
                pinned: Bool? = nil,
                archived: Bool? = nil,
                reasoningEffort: String? = nil) {
        self.folderId = folderId
        self.title = title
        self.provider = provider
        self.model = model
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.pinned = pinned
        self.archived = archived
        self.reasoningEffort = reasoningEffort
    }
}

public enum MessageRole: String, Codable, Sendable {
    case user, assistant, system
}

public enum MessageStatus: String, Codable, Sendable {
    case streaming, complete, error, stopped
}

/// One RAG hit the answer was grounded in.
public struct Citation: Codable, Hashable, Sendable {
    public let documentId: String
    public let documentName: String
    public let chunkIndex: Int
    public let snippet: String
    public let score: Double

    public init(documentId: String, documentName: String, chunkIndex: Int, snippet: String, score: Double) {
        self.documentId = documentId
        self.documentName = documentName
        self.chunkIndex = chunkIndex
        self.snippet = snippet
        self.score = score
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        documentId = try c.decodeIfPresent(String.self, forKey: .documentId) ?? UUID().uuidString
        documentName = try c.decodeIfPresent(String.self, forKey: .documentName) ?? "Dokument"
        chunkIndex = try c.decodeIfPresent(Int.self, forKey: .chunkIndex) ?? 0
        snippet = try c.decodeIfPresent(String.self, forKey: .snippet) ?? ""
        score = try c.decodeIfPresent(Double.self, forKey: .score) ?? 0
    }
}

public struct Attachment: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var messageId: String?
    public var conversationId: String?
    public var name: String
    public var mime: String
    public var size: Int
    public var kind: String
    public var textContent: String?
    public var createdAt: Double

    public var isImage: Bool { kind == "image" || mime.hasPrefix("image/") }

    public init(id: String,
                messageId: String? = nil,
                conversationId: String? = nil,
                name: String,
                mime: String,
                size: Int,
                kind: String,
                textContent: String? = nil,
                createdAt: Double = 0) {
        self.id = id
        self.messageId = messageId
        self.conversationId = conversationId
        self.name = name
        self.mime = mime
        self.size = size
        self.kind = kind
        self.textContent = textContent
        self.createdAt = createdAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        messageId = try c.decodeIfPresent(String.self, forKey: .messageId)
        conversationId = try c.decodeIfPresent(String.self, forKey: .conversationId)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "plik"
        mime = try c.decodeIfPresent(String.self, forKey: .mime) ?? "application/octet-stream"
        size = try c.decodeIfPresent(Int.self, forKey: .size) ?? 0
        kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? "file"
        textContent = try c.decodeIfPresent(String.self, forKey: .textContent)
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
    }
}

public struct Message: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var conversationId: String
    public var role: MessageRole
    public var content: String
    public var reasoning: String?
    public var provider: String?
    public var model: String?
    public var status: MessageStatus
    public var error: String?
    public var tokensIn: Int?
    public var tokensOut: Int?
    public var costUsd: Double?
    public var latencyMs: Int?
    public var seq: Int
    public var createdAt: Double
    public var attachments: [Attachment]
    public var citations: [Citation]

    public init(id: String,
                conversationId: String,
                role: MessageRole,
                content: String,
                reasoning: String? = nil,
                provider: String? = nil,
                model: String? = nil,
                status: MessageStatus = .complete,
                error: String? = nil,
                tokensIn: Int? = nil,
                tokensOut: Int? = nil,
                costUsd: Double? = nil,
                latencyMs: Int? = nil,
                seq: Int = 0,
                createdAt: Double = 0,
                attachments: [Attachment] = [],
                citations: [Citation] = []) {
        self.id = id
        self.conversationId = conversationId
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.provider = provider
        self.model = model
        self.status = status
        self.error = error
        self.tokensIn = tokensIn
        self.tokensOut = tokensOut
        self.costUsd = costUsd
        self.latencyMs = latencyMs
        self.seq = seq
        self.createdAt = createdAt
        self.attachments = attachments
        self.citations = citations
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        conversationId = try c.decodeIfPresent(String.self, forKey: .conversationId) ?? ""
        role = try c.decodeIfPresent(MessageRole.self, forKey: .role) ?? .assistant
        content = try c.decodeIfPresent(String.self, forKey: .content) ?? ""
        reasoning = try c.decodeIfPresent(String.self, forKey: .reasoning)
        provider = try c.decodeIfPresent(String.self, forKey: .provider)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        status = try c.decodeIfPresent(MessageStatus.self, forKey: .status) ?? .complete
        error = try c.decodeIfPresent(String.self, forKey: .error)
        tokensIn = try c.decodeIfPresent(Int.self, forKey: .tokensIn)
        tokensOut = try c.decodeIfPresent(Int.self, forKey: .tokensOut)
        costUsd = try c.decodeIfPresent(Double.self, forKey: .costUsd)
        latencyMs = try c.decodeIfPresent(Int.self, forKey: .latencyMs)
        seq = try c.decodeIfPresent(Int.self, forKey: .seq) ?? 0
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        attachments = try c.decodeIfPresent([Attachment].self, forKey: .attachments) ?? []
        citations = try c.decodeIfPresent([Citation].self, forKey: .citations) ?? []
    }
}

public struct ConversationDetail: Codable, Sendable {
    public let conversation: Conversation
    public let messages: [Message]

    public init(conversation: Conversation, messages: [Message]) {
        self.conversation = conversation
        self.messages = messages
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        conversation = try c.decode(Conversation.self, forKey: .conversation)
        messages = try c.decodeIfPresent([Message].self, forKey: .messages) ?? []
    }
}

// MARK: - Memory and knowledge

public enum MemoryScope: String, Codable, Sendable { case global, folder }

public enum MemoryKind: String, Codable, Sendable {
    case fact, preference, note

    public var title: String {
        switch self {
        case .fact: "Fakt"
        case .preference: "Preferencja"
        case .note: "Notatka"
        }
    }
}

public struct MemoryEntry: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var scope: MemoryScope
    public var folderId: String?
    public var kind: MemoryKind
    public var content: String
    public var source: String
    public var pinned: Bool
    public var createdAt: Double
    public var updatedAt: Double

    public init(id: String,
                scope: MemoryScope,
                folderId: String? = nil,
                kind: MemoryKind,
                content: String,
                source: String = "user",
                pinned: Bool = false,
                createdAt: Double = 0,
                updatedAt: Double = 0) {
        self.id = id
        self.scope = scope
        self.folderId = folderId
        self.kind = kind
        self.content = content
        self.source = source
        self.pinned = pinned
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        scope = try c.decodeIfPresent(MemoryScope.self, forKey: .scope) ?? .global
        folderId = try c.decodeIfPresent(String.self, forKey: .folderId)
        kind = try c.decodeIfPresent(MemoryKind.self, forKey: .kind) ?? .note
        content = try c.decodeIfPresent(String.self, forKey: .content) ?? ""
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? "user"
        pinned = try c.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt) ?? createdAt
    }
}

public struct MemoryInput: Codable, Sendable {
    public var scope: MemoryScope?
    public var folderId: String?
    public var kind: MemoryKind?
    public var content: String?
    public var pinned: Bool?

    public init(scope: MemoryScope? = nil,
                folderId: String? = nil,
                kind: MemoryKind? = nil,
                content: String? = nil,
                pinned: Bool? = nil) {
        self.scope = scope
        self.folderId = folderId
        self.kind = kind
        self.content = content
        self.pinned = pinned
    }
}

public struct KbDocument: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var scope: MemoryScope
    public var folderId: String?
    public var name: String
    public var mime: String
    public var size: Int
    public var status: String
    public var error: String?
    public var chunkCount: Int
    public var createdAt: Double
    public var updatedAt: Double

    public init(id: String,
                scope: MemoryScope,
                folderId: String? = nil,
                name: String,
                mime: String,
                size: Int,
                status: String,
                error: String? = nil,
                chunkCount: Int = 0,
                createdAt: Double = 0,
                updatedAt: Double = 0) {
        self.id = id
        self.scope = scope
        self.folderId = folderId
        self.name = name
        self.mime = mime
        self.size = size
        self.status = status
        self.error = error
        self.chunkCount = chunkCount
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        scope = try c.decodeIfPresent(MemoryScope.self, forKey: .scope) ?? .global
        folderId = try c.decodeIfPresent(String.self, forKey: .folderId)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "dokument"
        mime = try c.decodeIfPresent(String.self, forKey: .mime) ?? "application/octet-stream"
        size = try c.decodeIfPresent(Int.self, forKey: .size) ?? 0
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? "ready"
        error = try c.decodeIfPresent(String.self, forKey: .error)
        chunkCount = try c.decodeIfPresent(Int.self, forKey: .chunkCount) ?? 0
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt) ?? createdAt
    }
}

// MARK: - Settings

public struct UserProfile: Codable, Hashable, Sendable {
    public var displayName: String
    public var about: String
    public var role: String
    public var responseStyle: String
    public var customInstructions: String
    public var enabled: Bool

    public init(displayName: String = "",
                about: String = "",
                role: String = "",
                responseStyle: String = "",
                customInstructions: String = "",
                enabled: Bool = true) {
        self.displayName = displayName
        self.about = about
        self.role = role
        self.responseStyle = responseStyle
        self.customInstructions = customInstructions
        self.enabled = enabled
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        displayName = try c.decodeIfPresent(String.self, forKey: .displayName) ?? ""
        about = try c.decodeIfPresent(String.self, forKey: .about) ?? ""
        role = try c.decodeIfPresent(String.self, forKey: .role) ?? ""
        responseStyle = try c.decodeIfPresent(String.self, forKey: .responseStyle) ?? ""
        customInstructions = try c.decodeIfPresent(String.self, forKey: .customInstructions) ?? ""
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
    }
}

public struct GenerationDefaults: Codable, Hashable, Sendable {
    public var provider: String
    public var model: String
    public var temperature: Double
    public var maxTokens: Int
    public var ragTopK: Int
    public var autoTitle: Bool
    public var showReasoning: Bool
    public var historyLimit: Int

    public init(provider: String = "local",
                model: String = "",
                temperature: Double = 0.7,
                maxTokens: Int = 2048,
                ragTopK: Int = 4,
                autoTitle: Bool = true,
                showReasoning: Bool = true,
                historyLimit: Int = 20) {
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.ragTopK = ragTopK
        self.autoTitle = autoTitle
        self.showReasoning = showReasoning
        self.historyLimit = historyLimit
    }

    public init(from decoder: Decoder) throws {
        let fallback = GenerationDefaults()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        provider = try c.decodeIfPresent(String.self, forKey: .provider) ?? fallback.provider
        model = try c.decodeIfPresent(String.self, forKey: .model) ?? fallback.model
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? fallback.temperature
        maxTokens = try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? fallback.maxTokens
        ragTopK = try c.decodeIfPresent(Int.self, forKey: .ragTopK) ?? fallback.ragTopK
        autoTitle = try c.decodeIfPresent(Bool.self, forKey: .autoTitle) ?? fallback.autoTitle
        showReasoning = try c.decodeIfPresent(Bool.self, forKey: .showReasoning) ?? fallback.showReasoning
        historyLimit = try c.decodeIfPresent(Int.self, forKey: .historyLimit) ?? fallback.historyLimit
    }
}

public struct AppSettings: Codable, Hashable, Sendable {
    public var globalSystemPrompt: String
    public var profile: UserProfile
    public var defaults: GenerationDefaults
    public var memoryEnabled: Bool
    public var theme: String

    public init(globalSystemPrompt: String = "",
                profile: UserProfile = UserProfile(),
                defaults: GenerationDefaults = GenerationDefaults(),
                memoryEnabled: Bool = true,
                theme: String = "system") {
        self.globalSystemPrompt = globalSystemPrompt
        self.profile = profile
        self.defaults = defaults
        self.memoryEnabled = memoryEnabled
        self.theme = theme
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        globalSystemPrompt = try c.decodeIfPresent(String.self, forKey: .globalSystemPrompt) ?? ""
        profile = try c.decodeIfPresent(UserProfile.self, forKey: .profile) ?? UserProfile()
        defaults = try c.decodeIfPresent(GenerationDefaults.self, forKey: .defaults) ?? GenerationDefaults()
        memoryEnabled = try c.decodeIfPresent(Bool.self, forKey: .memoryEnabled) ?? true
        theme = try c.decodeIfPresent(String.self, forKey: .theme) ?? "system"
    }
}

// MARK: - Providers and models

/// In LocalCode there is exactly one provider — the machine this app runs on.
/// The type survives because the chat API is shared with the web app, and
/// because a second provider (a remote LocalCode, say) is a plausible future.
public struct ProviderInfo: Codable, Identifiable, Hashable, Sendable {
    public var provider: String
    public var label: String
    public var description: String

    public var id: String { provider }

    public init(provider: String, label: String, description: String = "") {
        self.provider = provider
        self.label = label
        self.description = description
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        provider = try c.decodeIfPresent(String.self, forKey: .provider) ?? "local"
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? "Modele lokalne"
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
    }

    public static let local = ProviderInfo(provider: "local",
                                           label: "Modele lokalne",
                                           description: "Modele uruchamiane na tym komputerze.")
}

public struct ProvidersResponse: Codable, Sendable {
    public let providers: [ProviderInfo]

    public init(providers: [ProviderInfo]) { self.providers = providers }
}

/// Models reached over somebody's API key: `remote:<provider>:<model id>`.
///
/// Every picker names them "<model> (<provider>)" — the same model can run on
/// this Mac and in a data centre, and only one of the two is free.
public enum RemoteModelKey {
    /// `"remote:deepinfra:meta-llama/…"` → `"deepinfra"`; `nil` for a local key.
    public static func provider(of key: String) -> String? {
        let parts = key.split(separator: ":", maxSplits: 2)
        guard parts.count == 3, parts[0] == "remote", !parts[2].isEmpty else { return nil }
        return String(parts[1])
    }

    public static func isRemote(_ key: String) -> Bool { provider(of: key) != nil }

    /// "DeepInfra", "Google Gemini"… for a provider id.
    public static func label(for provider: String) -> String {
        switch provider {
        case "openai": "OpenAI"
        case "anthropic": "Anthropic"
        case "gemini": "Google Gemini"
        case "deepinfra": "DeepInfra"
        case "unlid": "Unlid"
        default: provider.capitalized
        }
    }

    /// The provider's name for a key, or `nil` for a local model.
    public static func providerLabel(of key: String) -> String? {
        provider(of: key).map(label(for:))
    }

    /// "Llama 3.3 70B (DeepInfra)"; a local model keeps its name. Idempotent —
    /// the chat API already sends remote labels with the suffix.
    public static func name(_ name: String, key: String) -> String {
        guard let label = providerLabel(of: key) else { return name }
        let suffix = " (\(label))"
        return name.hasSuffix(suffix) ? name : name + suffix
    }

    /// The name without the provider suffix, for rows already grouped by it.
    public static func bareName(_ name: String, key: String) -> String {
        guard let label = providerLabel(of: key) else { return name }
        let suffix = " (\(label))"
        return name.hasSuffix(suffix) ? String(name.dropLast(suffix.count)) : name
    }
}

/// A chat-capable model the server can load. `id` is `LocalModel.key`.
public struct ModelInfo: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public var provider: String
    public var label: String
    public var contextWindow: Int?
    public var maxOutput: Int?
    public var supportsVision: Bool
    public var supportsReasoning: Bool
    /// Reasoning levels this model can express, as `GET /api/models` reports
    /// them: `[]` (no control), `["off","on"]` or `["low","medium","high"]`.
    public var effortLevels: [String]
    /// Always `nil` for local models — every cost readout hides itself.
    public var inputPricePerM: Double?
    public var outputPricePerM: Double?
    public var description: String?

    public init(id: String,
                provider: String = "local",
                label: String,
                contextWindow: Int? = nil,
                maxOutput: Int? = nil,
                supportsVision: Bool = false,
                supportsReasoning: Bool = false,
                effortLevels: [String] = [],
                inputPricePerM: Double? = nil,
                outputPricePerM: Double? = nil,
                description: String? = nil) {
        self.id = id
        self.provider = provider
        self.label = label
        self.contextWindow = contextWindow
        self.maxOutput = maxOutput
        self.supportsVision = supportsVision
        self.supportsReasoning = supportsReasoning
        self.effortLevels = effortLevels
        self.inputPricePerM = inputPricePerM
        self.outputPricePerM = outputPricePerM
        self.description = description
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        provider = try c.decodeIfPresent(String.self, forKey: .provider) ?? "local"
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? id
        contextWindow = try c.decodeIfPresent(Int.self, forKey: .contextWindow)
        maxOutput = try c.decodeIfPresent(Int.self, forKey: .maxOutput)
        supportsVision = try c.decodeIfPresent(Bool.self, forKey: .supportsVision) ?? false
        supportsReasoning = try c.decodeIfPresent(Bool.self, forKey: .supportsReasoning) ?? false
        effortLevels = try c.decodeIfPresent([String].self, forKey: .effortLevels) ?? []
        inputPricePerM = try c.decodeIfPresent(Double.self, forKey: .inputPricePerM)
        outputPricePerM = try c.decodeIfPresent(Double.self, forKey: .outputPricePerM)
        description = try c.decodeIfPresent(String.self, forKey: .description)
    }
}

// MARK: - Search

public struct SearchHit: Codable, Identifiable, Hashable, Sendable {
    public let conversationId: String
    public let conversationTitle: String
    public let messageId: String
    public let role: MessageRole
    public let snippet: String
    public let createdAt: Double
    public let folderId: String?
    public let folderName: String?

    public var id: String { messageId }

    public init(conversationId: String,
                conversationTitle: String,
                messageId: String,
                role: MessageRole,
                snippet: String,
                createdAt: Double,
                folderId: String? = nil,
                folderName: String? = nil) {
        self.conversationId = conversationId
        self.conversationTitle = conversationTitle
        self.messageId = messageId
        self.role = role
        self.snippet = snippet
        self.createdAt = createdAt
        self.folderId = folderId
        self.folderName = folderName
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        conversationId = try c.decode(String.self, forKey: .conversationId)
        conversationTitle = try c.decodeIfPresent(String.self, forKey: .conversationTitle) ?? "Rozmowa"
        messageId = try c.decode(String.self, forKey: .messageId)
        role = try c.decodeIfPresent(MessageRole.self, forKey: .role) ?? .user
        snippet = try c.decodeIfPresent(String.self, forKey: .snippet) ?? ""
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt) ?? 0
        folderId = try c.decodeIfPresent(String.self, forKey: .folderId)
        folderName = try c.decodeIfPresent(String.self, forKey: .folderName)
    }
}

// MARK: - Bootstrap

/// Everything the chat UI needs in one round trip.
public struct Bootstrap: Codable, Sendable {
    public let settings: AppSettings
    public let folders: [Folder]
    public let conversations: [Conversation]
    public let providers: [ProviderInfo]
    public let models: [ModelInfo]
    public let activeProvider: String

    public init(settings: AppSettings = AppSettings(),
                folders: [Folder] = [],
                conversations: [Conversation] = [],
                providers: [ProviderInfo] = [.local],
                models: [ModelInfo] = [],
                activeProvider: String = "local") {
        self.settings = settings
        self.folders = folders
        self.conversations = conversations
        self.providers = providers
        self.models = models
        self.activeProvider = activeProvider
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        settings = try c.decodeIfPresent(AppSettings.self, forKey: .settings) ?? AppSettings()
        folders = try c.decodeIfPresent([Folder].self, forKey: .folders) ?? []
        conversations = try c.decodeIfPresent([Conversation].self, forKey: .conversations) ?? []
        let decodedProviders = try c.decodeIfPresent([ProviderInfo].self, forKey: .providers) ?? []
        providers = decodedProviders.isEmpty ? [.local] : decodedProviders
        models = try c.decodeIfPresent([ModelInfo].self, forKey: .models) ?? []
        activeProvider = try c.decodeIfPresent(String.self, forKey: .activeProvider) ?? "local"
    }
}

// MARK: - Chat streaming

public struct ChatSendRequest: Codable, Sendable {
    public var conversationId: String?
    public var folderId: String?
    public var content: String
    public var attachmentIds: [String]?
    public var provider: String?
    public var model: String?
    public var editMessageId: String?
    public var regenerate: Bool?
    /// Overrides the conversation's own level for this one turn.
    public var reasoningEffort: String?

    public init(conversationId: String? = nil,
                folderId: String? = nil,
                content: String,
                attachmentIds: [String]? = nil,
                provider: String? = nil,
                model: String? = nil,
                editMessageId: String? = nil,
                regenerate: Bool? = nil,
                reasoningEffort: String? = nil) {
        self.conversationId = conversationId
        self.folderId = folderId
        self.content = content
        self.attachmentIds = attachmentIds
        self.provider = provider
        self.model = model
        self.editMessageId = editMessageId
        self.regenerate = regenerate
        self.reasoningEffort = reasoningEffort
    }
}

/// One decoded frame of the `/api/chat` SSE stream.
///
/// `status` is new in LocalCode: loading a local model takes seconds, and the
/// server announces what it is doing ("Ładowanie modelu…") before the first
/// token exists. Older clients simply ignore the frame.
public enum ChatStreamEvent: Sendable {
    case start(conversationId: String, userMessage: Message, assistantMessageId: String)
    case title(conversationId: String, title: String)
    case citations([Citation])
    case status(String)
    case reasoning(String)
    case text(String)
    case usage(tokensIn: Int?, tokensOut: Int?, costUsd: Double?)
    /// Older turns were summarised away to make room for this one.
    case compaction(ChatCompaction)
    /// How much of the window the turn that is starting takes.
    case context(ChatContextUsage)
    case done(Message)
    case failure(String)
}

// MARK: - Context and compaction

/// `{"type":"compaction","before","after","strategy"}` — the chat had to give
/// up part of its history to fit the next turn into the model's window.
public struct ChatCompaction: Decodable, Hashable, Sendable, Identifiable {
    public var before: Int
    public var after: Int
    public var strategy: String
    /// Assigned by the client: the server does not number these.
    public var id = UUID()

    public init(before: Int, after: Int, strategy: String) {
        self.before = before
        self.after = after
        self.strategy = strategy
    }

    private enum CodingKeys: String, CodingKey { case before, after, strategy }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        before = (try? c.decode(Int.self, forKey: .before)) ?? 0
        after = (try? c.decode(Int.self, forKey: .after)) ?? 0
        strategy = (try? c.decode(String.self, forKey: .strategy)) ?? ""
    }

    /// `Kontekst skompaktowany: 12,4k → 5,1k tokenów`
    public var headline: String {
        "Kontekst skompaktowany: \(ChatTokenFormat.compact(before)) → \(ChatTokenFormat.compact(after)) tokenów"
    }

    public var strategyLabel: String {
        switch strategy {
        case "mask_tool_results": "ukryto wyniki narzędzi"
        case "summarize": "podsumowano starsze tury"
        case "drop_oldest": "usunięto najstarsze wiadomości"
        default: strategy
        }
    }

    /// How many tokens the compaction actually freed.
    public var saved: Int { max(0, before - after) }
}

/// `{"type":"context","used","limit","system","tools","history","reserve","summary"}`
/// — the breakdown the Python context manager reports before each turn.
public struct ChatContextUsage: Decodable, Hashable, Sendable {
    public var used: Int
    public var limit: Int
    public var system: Int
    public var tools: Int
    public var history: Int
    public var reserve: Int
    public var summary: Int

    public init(used: Int = 0, limit: Int = 0, system: Int = 0,
                tools: Int = 0, history: Int = 0, reserve: Int = 0, summary: Int = 0) {
        self.used = used
        self.limit = limit
        self.system = system
        self.tools = tools
        self.history = history
        self.reserve = reserve
        self.summary = summary
    }

    private enum CodingKeys: String, CodingKey {
        case used, limit, system, tools, history, reserve, summary
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        used = (try? c.decode(Int.self, forKey: .used)) ?? 0
        limit = (try? c.decode(Int.self, forKey: .limit)) ?? 0
        system = (try? c.decode(Int.self, forKey: .system)) ?? 0
        tools = (try? c.decode(Int.self, forKey: .tools)) ?? 0
        history = (try? c.decode(Int.self, forKey: .history)) ?? 0
        reserve = (try? c.decode(Int.self, forKey: .reserve)) ?? 0
        summary = (try? c.decode(Int.self, forKey: .summary)) ?? 0
    }

    public var isEmpty: Bool { limit <= 0 }

    public var fraction: Double {
        guard limit > 0 else { return 0 }
        return min(1, max(0, Double(used) / Double(limit)))
    }

    public var percent: Int { Int((fraction * 100).rounded()) }
}

/// `1,2k` / `3,4M`, with the Polish decimal comma. The agent has the same
/// helper; the chat cannot reach it, because `CodeFeature` sits beside it.
public enum ChatTokenFormat {
    public static func compact(_ value: Int) -> String {
        if value >= 1_000_000 {
            return String(format: "%.1fM", Double(value) / 1_000_000).replacingOccurrences(of: ".", with: ",")
        }
        if value >= 1_000 {
            return String(format: "%.1fk", Double(value) / 1_000).replacingOccurrences(of: ".", with: ",")
        }
        return "\(value)"
    }

    /// `12 345` with Polish grouping, for the gauge tooltip.
    public static func decimal(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "pl_PL")
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }
}

public struct APIErrorBody: Codable, Sendable {
    public let error: String
    public let code: String?

    public init(error: String, code: String? = nil) {
        self.error = error
        self.code = code
    }
}

/// The body of `POST /api/conversations/{id}/compact`.
///
/// `changed` is false when the history was already short enough to leave alone,
/// which the UI says out loud rather than pretending something happened.
public struct ConversationCompaction: Decodable, Hashable, Sendable {
    public var summary: String
    public var before: Int
    public var after: Int
    public var changed: Bool
    public var strategy: String
    public var messages: Int

    public init(summary: String = "", before: Int = 0, after: Int = 0,
                changed: Bool = false, strategy: String = "", messages: Int = 0) {
        self.summary = summary
        self.before = before
        self.after = after
        self.changed = changed
        self.strategy = strategy
        self.messages = messages
    }

    private enum CodingKeys: String, CodingKey {
        case summary, before, after, changed, strategy, messages
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        summary = (try? c.decode(String.self, forKey: .summary)) ?? ""
        before = (try? c.decode(Int.self, forKey: .before)) ?? 0
        after = (try? c.decode(Int.self, forKey: .after)) ?? 0
        changed = (try? c.decode(Bool.self, forKey: .changed)) ?? false
        strategy = (try? c.decode(String.self, forKey: .strategy)) ?? ""
        messages = (try? c.decode(Int.self, forKey: .messages)) ?? 0
    }

    /// The sentence the chat shows after the user asked for a compaction.
    public var note: String {
        guard changed else { return "Rozmowa jest wystarczająco krótka — nie było czego skracać." }
        return "Skrócono rozmowę: \(ChatTokenFormat.compact(before)) → \(ChatTokenFormat.compact(after)) tokenów."
    }
}
