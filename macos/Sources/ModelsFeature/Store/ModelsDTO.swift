import Foundation
import LocalCodeKit

/// The wire types of the LocalCode model API (`/api/hub/*`, `/api/downloads`,
/// `/api/library`, `/api/runtime`, `/api/app-settings`, `/api/events`).
///
/// Every struct here was written against the payloads the running server
/// actually produces (captured under `macos/Tests/Fixtures/`), not against the
/// prose in `docs/ARCHITECTURE.md` — where the two disagreed the real bytes
/// won. Two consequences are worth remembering:
///
/// * `LocalModelDTO.modifiedAt` is in **seconds**, while every other timestamp
///   in the API (and in `Format.relative`) is in **milliseconds**.
/// * Several fields are `null` for curated entries that never touched the Hub
///   (`downloads`, `likes`, `updatedAt`), so almost everything is optional.
///
/// The decoders are written by hand rather than synthesised: a field the server
/// adds or drops later must degrade to a default, never break a whole screen.

// MARK: - Free-form JSON

/// A JSON value the client can build by hand.
///
/// `PATCH /api/runtime/settings` and `PATCH /api/app-settings` distinguish
/// "leave this alone" (key absent) from "back to automatic" (key present and
/// `null`), which no `Encodable` struct of optionals can express. This does.
public enum JSONValue: Encodable, Equatable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case object([String: JSONValue])
    case array([JSONValue])

    public static func intOrNull(_ value: Int?) -> JSONValue { value.map(JSONValue.int) ?? .null }
    public static func doubleOrNull(_ value: Double?) -> JSONValue { value.map(JSONValue.double) ?? .null }
    public static func stringOrNull(_ value: String?) -> JSONValue { value.map(JSONValue.string) ?? .null }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .null: try container.encodeNil()
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        }
    }
}

/// A JSON object body, e.g. `["ramLimitGb": .null]`.
public struct JSONObject: Encodable, Equatable, Sendable, ExpressibleByDictionaryLiteral {
    public var fields: [String: JSONValue]

    public init(_ fields: [String: JSONValue] = [:]) { self.fields = fields }

    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        fields = Dictionary(elements) { _, last in last }
    }

    public var isEmpty: Bool { fields.isEmpty }

    public subscript(key: String) -> JSONValue? {
        get { fields[key] }
        set { fields[key] = newValue }
    }

    public func encode(to encoder: Encoder) throws {
        try fields.encode(to: encoder)
    }
}

// MARK: - Load configuration

/// `LoadConfigDTO` — how one instance of a model should be started.
///
/// Every field is optional: the server fills a missing one from the global
/// runtime settings, which is exactly what "Auto" means in the UI.
public struct LoadConfigDTO: Codable, Hashable, Sendable {
    public var nCtx: Int?
    public var nParallel: Int?
    public var gpuLayers: Int?
    public var flashAttn: String?
    public var kvCacheType: String?
    public var embedding: Bool?
    /// Reasoning level this model should default to (`off`…`max`).
    ///
    /// Not a load parameter — the process is started the same way either
    /// way — but it belongs to the model rather than to a conversation, so the
    /// server stores it with the other per-model settings and the agent and the
    /// chat apply it per request.
    public var effort: String?

    public init(nCtx: Int? = nil,
                nParallel: Int? = nil,
                gpuLayers: Int? = nil,
                flashAttn: String? = nil,
                kvCacheType: String? = nil,
                embedding: Bool? = nil,
                effort: String? = nil) {
        self.nCtx = nCtx
        self.nParallel = nParallel
        self.gpuLayers = gpuLayers
        self.flashAttn = flashAttn
        self.kvCacheType = kvCacheType
        self.embedding = embedding
        self.effort = effort
    }

    public var isEmpty: Bool {
        nCtx == nil && nParallel == nil && gpuLayers == nil
            && flashAttn == nil && kvCacheType == nil && embedding == nil && effort == nil
    }

    private enum CodingKeys: String, CodingKey {
        case nCtx, nParallel, gpuLayers, flashAttn, kvCacheType, embedding
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nCtx = try c.decodeIfPresent(Int.self, forKey: .nCtx)
        nParallel = try c.decodeIfPresent(Int.self, forKey: .nParallel)
        gpuLayers = try c.decodeIfPresent(Int.self, forKey: .gpuLayers)
        flashAttn = try c.decodeIfPresent(String.self, forKey: .flashAttn)
        kvCacheType = try c.decodeIfPresent(String.self, forKey: .kvCacheType)
        embedding = try c.decodeIfPresent(Bool.self, forKey: .embedding)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(nCtx, forKey: .nCtx)
        try c.encodeIfPresent(nParallel, forKey: .nParallel)
        try c.encodeIfPresent(gpuLayers, forKey: .gpuLayers)
        try c.encodeIfPresent(flashAttn, forKey: .flashAttn)
        try c.encodeIfPresent(kvCacheType, forKey: .kvCacheType)
        try c.encodeIfPresent(embedding, forKey: .embedding)
    }
}

// MARK: - Memory estimate

/// `MemoryEstimate` — what one instance would cost, and whether it fits.
public struct MemoryEstimate: Decodable, Hashable, Sendable {
    /// `fits` as the three states the UI paints.
    public enum Fit: String, Sendable {
        case yes, tight, no

        public var label: String {
            switch self {
            case .yes: "Zmieści się"
            case .tight: "Na styk"
            case .no: "Za duży"
            }
        }

        public var systemImage: String {
            switch self {
            case .yes: "checkmark.circle.fill"
            case .tight: "exclamationmark.circle.fill"
            case .no: "xmark.circle.fill"
            }
        }
    }

    public var weightsBytes: Int64
    public var kvCacheBytes: Int64
    public var overheadBytes: Int64
    public var totalBytes: Int64
    public var nCtx: Int
    public var nParallel: Int
    public var fitsRaw: String
    public var message: String

    public var fits: Fit { Fit(rawValue: fitsRaw) ?? .tight }

    private enum CodingKeys: String, CodingKey {
        case weightsBytes, kvCacheBytes, overheadBytes, totalBytes, nCtx, nParallel
        case fitsRaw = "fits"
        case message
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        weightsBytes = try c.decodeIfPresent(Int64.self, forKey: .weightsBytes) ?? 0
        kvCacheBytes = try c.decodeIfPresent(Int64.self, forKey: .kvCacheBytes) ?? 0
        overheadBytes = try c.decodeIfPresent(Int64.self, forKey: .overheadBytes) ?? 0
        totalBytes = try c.decodeIfPresent(Int64.self, forKey: .totalBytes) ?? 0
        nCtx = try c.decodeIfPresent(Int.self, forKey: .nCtx) ?? 0
        nParallel = try c.decodeIfPresent(Int.self, forKey: .nParallel) ?? 1
        fitsRaw = try c.decodeIfPresent(String.self, forKey: .fitsRaw) ?? "tight"
        message = try c.decodeIfPresent(String.self, forKey: .message) ?? ""
    }
}

// MARK: - Library

/// One model on disk (`GET /api/library`).
public struct LocalModelDTO: Decodable, Identifiable, Hashable, Sendable {
    public var key: String
    public var format: String
    public var path: String
    public var displayName: String
    public var sizeBytes: Int64
    public var repo: String?
    public var quant: String?
    /// `localcode` | `lmstudio` | `hf-cache` | `custom`
    public var source: String
    public var capabilities: [String]
    public var arch: String?
    public var paramsB: Double?
    public var contextLength: Int?
    /// Unix **seconds** — unlike every other timestamp in this API.
    public var modifiedAt: Double?
    public var hasVision: Bool
    public var loaded: Bool
    public var loadDefaults: LoadConfigDTO?
    /// True for a model reached over somebody's API key: it runs in a data
    /// centre, costs money per request and takes none of this Mac's memory.
    public var remote: Bool

    public var id: String { key }

    public var isEmbedding: Bool { capabilities.contains("embedding") }

    /// "OpenAI", "Anthropic"… — the provider behind a remote model.
    public var providerLabel: String? {
        remote ? RemoteModelKey.providerLabel(of: key) : nil
    }

    /// The name every picker shows: "Llama 3.3 70B (DeepInfra)" for a remote
    /// model, the plain name for one on disk.
    public var pickerName: String { RemoteModelKey.name(displayName, key: key) }
    public var supportsTools: Bool { capabilities.contains("tools") }
    public var supportsThinking: Bool { capabilities.contains("thinking") }

    /// Where the file came from, in Polish.
    public var sourceLabel: String {
        switch source {
        case "localcode": "LocalCode"
        case "lmstudio": "LM Studio"
        case "hf-cache": "cache HF"
        case "custom": "własny katalog"
        default: source
        }
    }

    /// Only models LocalCode downloaded itself may be deleted from the app.
    public var isDeletable: Bool { source == "localcode" }

    public var formatLabel: String { format.uppercased() }

    private enum CodingKeys: String, CodingKey {
        case key, format, path, displayName, sizeBytes, repo, quant, source
        case capabilities, arch, paramsB, contextLength, modifiedAt, hasVision
        case loaded, loadDefaults, remote
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = try c.decode(String.self, forKey: .key)
        format = try c.decodeIfPresent(String.self, forKey: .format) ?? "gguf"
        path = try c.decodeIfPresent(String.self, forKey: .path) ?? ""
        displayName = try c.decodeIfPresent(String.self, forKey: .displayName) ?? key
        sizeBytes = try c.decodeIfPresent(Int64.self, forKey: .sizeBytes) ?? 0
        repo = try c.decodeIfPresent(String.self, forKey: .repo)
        quant = try c.decodeIfPresent(String.self, forKey: .quant)
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? "localcode"
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        arch = try c.decodeIfPresent(String.self, forKey: .arch)
        paramsB = try c.decodeIfPresent(Double.self, forKey: .paramsB)
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength)
        modifiedAt = try c.decodeIfPresent(Double.self, forKey: .modifiedAt)
        hasVision = try c.decodeIfPresent(Bool.self, forKey: .hasVision) ?? false
        loaded = try c.decodeIfPresent(Bool.self, forKey: .loaded) ?? false
        loadDefaults = try c.decodeIfPresent(LoadConfigDTO.self, forKey: .loadDefaults)
        remote = try c.decodeIfPresent(Bool.self, forKey: .remote) ?? (source == "remote")
    }
}

// MARK: - Hub

/// One repository on Hugging Face (`/api/hub/search`, `/api/hub/curated`).
public struct HubModel: Decodable, Identifiable, Hashable, Sendable {
    public var id: String
    public var author: String
    public var name: String
    public var format: String
    public var downloads: Int?
    public var likes: Int?
    /// Milliseconds since epoch.
    public var updatedAt: Double?
    public var tags: [String]
    public var pipelineTag: String?
    public var paramsB: Double?
    public var activeParamsB: Double?
    public var gated: Bool
    public var curated: Bool
    public var description: String?
    public var arch: String?
    public var contextLength: Int?
    public var trendingScore: Double?
    /// Curated entries only: `code` | `chat` | `reasoning` | `vision` | `embedding`.
    public var category: String?
    public var minRamGb: Int?
    public var recommendedQuant: String?
    /// The same score the store shows, so the advanced browser — where the
    /// unvetted repositories actually live — is not the one view without it.
    public var trust: CatalogTrust
    public var publisher: CatalogPublisher

    public var formatLabel: String { format.uppercased() }

    private enum CodingKeys: String, CodingKey {
        case id, author, name, format, downloads, likes, updatedAt, tags, pipelineTag
        case paramsB, activeParamsB, gated, curated, description, arch, contextLength
        case trendingScore, category, minRamGb, recommendedQuant, trust, publisher
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        author = try c.decodeIfPresent(String.self, forKey: .author)
            ?? String(id.split(separator: "/").first ?? "")
        name = try c.decodeIfPresent(String.self, forKey: .name)
            ?? String(id.split(separator: "/").last ?? "")
        format = try c.decodeIfPresent(String.self, forKey: .format) ?? "gguf"
        downloads = try c.decodeIfPresent(Int.self, forKey: .downloads)
        likes = try c.decodeIfPresent(Int.self, forKey: .likes)
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        pipelineTag = try c.decodeIfPresent(String.self, forKey: .pipelineTag)
        paramsB = try c.decodeIfPresent(Double.self, forKey: .paramsB)
        activeParamsB = try c.decodeIfPresent(Double.self, forKey: .activeParamsB)
        gated = try c.decodeIfPresent(Bool.self, forKey: .gated) ?? false
        curated = try c.decodeIfPresent(Bool.self, forKey: .curated) ?? false
        description = try c.decodeIfPresent(String.self, forKey: .description)
        arch = try c.decodeIfPresent(String.self, forKey: .arch)
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength)
        trendingScore = try c.decodeIfPresent(Double.self, forKey: .trendingScore)
        category = try c.decodeIfPresent(String.self, forKey: .category)
        minRamGb = try c.decodeIfPresent(Int.self, forKey: .minRamGb)
        recommendedQuant = try c.decodeIfPresent(String.self, forKey: .recommendedQuant)
        trust = (try? c.decode(CatalogTrust.self, forKey: .trust)) ?? CatalogTrust()
        publisher = (try? c.decode(CatalogPublisher.self, forKey: .publisher)) ?? CatalogPublisher()
    }
}

/// One downloadable quantisation of a repository.
public struct HubVariant: Decodable, Identifiable, Hashable, Sendable {
    public var id: String
    public var label: String
    public var quant: String?
    public var files: [String]
    public var sizeBytes: Int64
    public var modelBytes: Int64?
    public var isVisionProjector: Bool
    public var complete: Bool
    public var estimate: MemoryEstimate?
    public var recommended: Bool

    private enum CodingKeys: String, CodingKey {
        case id, label, quant, files, sizeBytes, modelBytes, isVisionProjector
        case complete, estimate, recommended
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? id
        quant = try c.decodeIfPresent(String.self, forKey: .quant)
        files = try c.decodeIfPresent([String].self, forKey: .files) ?? []
        sizeBytes = try c.decodeIfPresent(Int64.self, forKey: .sizeBytes) ?? 0
        modelBytes = try c.decodeIfPresent(Int64.self, forKey: .modelBytes)
        isVisionProjector = try c.decodeIfPresent(Bool.self, forKey: .isVisionProjector) ?? false
        complete = try c.decodeIfPresent(Bool.self, forKey: .complete) ?? true
        estimate = try c.decodeIfPresent(MemoryEstimate.self, forKey: .estimate)
        recommended = try c.decodeIfPresent(Bool.self, forKey: .recommended) ?? false
    }
}

/// One file in a repository listing.
public struct HubFile: Decodable, Identifiable, Hashable, Sendable {
    public var name: String
    public var size: Int64?

    public var id: String { name }

    private enum CodingKeys: String, CodingKey { case name, size }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        size = try c.decodeIfPresent(Int64.self, forKey: .size)
    }
}

/// `GET /api/hub/model?repo=` — the detail pane's whole payload.
public struct HubModelDetail: Decodable, Hashable, Sendable {
    public var model: HubModel
    public var readme: String?
    public var contextLength: Int?
    public var arch: String?
    public var capabilities: [String]
    public var isEmbedding: Bool
    public var variants: [HubVariant]
    public var files: [HubFile]
    /// A finished Polish sentence when the repository is gated or unreachable.
    public var accessError: String?
    public var sha: String?

    private enum CodingKeys: String, CodingKey {
        case model, readme, contextLength, arch, capabilities, isEmbedding
        case variants, files, accessError, sha
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(HubModel.self, forKey: .model)
        readme = try c.decodeIfPresent(String.self, forKey: .readme)
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength)
        arch = try c.decodeIfPresent(String.self, forKey: .arch)
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        isEmbedding = try c.decodeIfPresent(Bool.self, forKey: .isEmbedding) ?? false
        variants = try c.decodeIfPresent([HubVariant].self, forKey: .variants) ?? []
        files = try c.decodeIfPresent([HubFile].self, forKey: .files) ?? []
        accessError = try c.decodeIfPresent(String.self, forKey: .accessError)
        sha = try c.decodeIfPresent(String.self, forKey: .sha)
    }
}

// MARK: - Downloads

/// One download job (`GET /api/downloads`, `{"type":"download"}` events).
public struct DownloadJob: Decodable, Identifiable, Hashable, Sendable {
    public enum State: String, Sendable {
        case queued, downloading, done, error, cancelled

        public var label: String {
            switch self {
            case .queued: "W kolejce"
            case .downloading: "Pobieranie"
            case .done: "Gotowe"
            case .error: "Błąd"
            case .cancelled: "Anulowano"
            }
        }

        public var isActive: Bool { self == .queued || self == .downloading }
    }

    public var id: String
    public var repo: String
    public var files: [String]
    public var label: String
    public var format: String
    public var totalBytes: Int64
    public var doneBytes: Int64
    public var progress: Double
    public var speedBps: Double
    public var etaSeconds: Double?
    public var stateRaw: String
    /// `download` | `verify`
    public var stage: String
    public var error: String?
    public var errorCode: String?
    public var currentFile: String?
    public var createdAt: Double?
    public var startedAt: Double?
    public var finishedAt: Double?
    public var modelKey: String?
    public var dest: String?

    public var state: State { State(rawValue: stateRaw) ?? .queued }
    public var isVerifying: Bool { (stage == "verify" || stage == "cerber") && state == .downloading }
    /// Cerber is scanning the repository; nothing has been downloaded yet.
    public var isCerberChecking: Bool { stage == "cerber" && state == .downloading }

    private enum CodingKeys: String, CodingKey {
        case id, repo, files, label, format, totalBytes, doneBytes, progress
        case speedBps, etaSeconds, stage, error, errorCode, currentFile
        case createdAt, startedAt, finishedAt, modelKey, dest
        case stateRaw = "state"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        repo = try c.decodeIfPresent(String.self, forKey: .repo) ?? ""
        files = try c.decodeIfPresent([String].self, forKey: .files) ?? []
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? repo
        format = try c.decodeIfPresent(String.self, forKey: .format) ?? "gguf"
        totalBytes = try c.decodeIfPresent(Int64.self, forKey: .totalBytes) ?? 0
        doneBytes = try c.decodeIfPresent(Int64.self, forKey: .doneBytes) ?? 0
        progress = try c.decodeIfPresent(Double.self, forKey: .progress) ?? 0
        speedBps = try c.decodeIfPresent(Double.self, forKey: .speedBps) ?? 0
        etaSeconds = try c.decodeIfPresent(Double.self, forKey: .etaSeconds)
        stateRaw = try c.decodeIfPresent(String.self, forKey: .stateRaw) ?? "queued"
        stage = try c.decodeIfPresent(String.self, forKey: .stage) ?? "download"
        error = try c.decodeIfPresent(String.self, forKey: .error)
        errorCode = try c.decodeIfPresent(String.self, forKey: .errorCode)
        currentFile = try c.decodeIfPresent(String.self, forKey: .currentFile)
        createdAt = try c.decodeIfPresent(Double.self, forKey: .createdAt)
        startedAt = try c.decodeIfPresent(Double.self, forKey: .startedAt)
        finishedAt = try c.decodeIfPresent(Double.self, forKey: .finishedAt)
        modelKey = try c.decodeIfPresent(String.self, forKey: .modelKey)
        dest = try c.decodeIfPresent(String.self, forKey: .dest)
    }
}

// MARK: - Runtime

/// One loaded (or loading, or failed) model instance.
public struct RuntimeInstance: Decodable, Identifiable, Hashable, Sendable {
    public enum State: String, Sendable {
        case loading, ready, busy, error, stopping

        public var label: String {
            switch self {
            case .loading: "Ładowanie"
            case .ready: "Gotowy"
            case .busy: "Zajęty"
            case .error: "Błąd"
            case .stopping: "Zwalnianie"
            }
        }
    }

    public var id: String
    public var modelKey: String
    public var displayName: String
    public var format: String
    public var stateRaw: String
    public var nCtx: Int
    public var nParallel: Int
    public var busySlots: Int
    public var memoryBytes: Int64
    public var estimatedBytes: Int64
    /// Milliseconds since epoch.
    public var loadedAt: Double?
    public var lastUsedAt: Double?
    public var pid: Int?
    public var port: Int?
    public var genTps: Double?
    public var error: String?

    public var state: State { State(rawValue: stateRaw) ?? .loading }

    private enum CodingKeys: String, CodingKey {
        case id, modelKey, displayName, format, nCtx, nParallel, busySlots
        case memoryBytes, estimatedBytes, loadedAt, lastUsedAt, pid, port, genTps, error
        case stateRaw = "state"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        modelKey = try c.decodeIfPresent(String.self, forKey: .modelKey) ?? ""
        displayName = try c.decodeIfPresent(String.self, forKey: .displayName) ?? modelKey
        format = try c.decodeIfPresent(String.self, forKey: .format) ?? "gguf"
        stateRaw = try c.decodeIfPresent(String.self, forKey: .stateRaw) ?? "loading"
        nCtx = try c.decodeIfPresent(Int.self, forKey: .nCtx) ?? 0
        nParallel = try c.decodeIfPresent(Int.self, forKey: .nParallel) ?? 1
        busySlots = try c.decodeIfPresent(Int.self, forKey: .busySlots) ?? 0
        memoryBytes = try c.decodeIfPresent(Int64.self, forKey: .memoryBytes) ?? 0
        estimatedBytes = try c.decodeIfPresent(Int64.self, forKey: .estimatedBytes) ?? 0
        loadedAt = try c.decodeIfPresent(Double.self, forKey: .loadedAt)
        lastUsedAt = try c.decodeIfPresent(Double.self, forKey: .lastUsedAt)
        pid = try c.decodeIfPresent(Int.self, forKey: .pid)
        port = try c.decodeIfPresent(Int.self, forKey: .port)
        genTps = try c.decodeIfPresent(Double.self, forKey: .genTps)
        error = try c.decodeIfPresent(String.self, forKey: .error)
    }
}

/// Physical memory as the server sees it.
public struct SystemMemory: Decodable, Hashable, Sendable {
    public var totalBytes: Int64
    public var usedBytes: Int64
    public var availableBytes: Int64

    public init(totalBytes: Int64 = 0, usedBytes: Int64 = 0, availableBytes: Int64 = 0) {
        self.totalBytes = totalBytes
        self.usedBytes = usedBytes
        self.availableBytes = availableBytes
    }

    private enum CodingKeys: String, CodingKey { case totalBytes, usedBytes, availableBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        totalBytes = try c.decodeIfPresent(Int64.self, forKey: .totalBytes) ?? 0
        usedBytes = try c.decodeIfPresent(Int64.self, forKey: .usedBytes) ?? 0
        availableBytes = try c.decodeIfPresent(Int64.self, forKey: .availableBytes) ?? 0
    }
}

/// How much of the RAM budget the loaded models are holding.
public struct MemoryBudget: Decodable, Hashable, Sendable {
    public var limitBytes: Int64
    public var usedBytes: Int64

    public init(limitBytes: Int64 = 0, usedBytes: Int64 = 0) {
        self.limitBytes = limitBytes
        self.usedBytes = usedBytes
    }

    private enum CodingKeys: String, CodingKey { case limitBytes, usedBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        limitBytes = try c.decodeIfPresent(Int64.self, forKey: .limitBytes) ?? 0
        usedBytes = try c.decodeIfPresent(Int64.self, forKey: .usedBytes) ?? 0
    }
}

/// The global runtime settings (`RuntimeStatus.settings`).
public struct RuntimeSettings: Decodable, Hashable, Sendable {
    public var maxLoadedModels: Int
    public var maxInstancesPerModel: Int
    public var defaultParallel: Int
    /// `nil` = automatic (70 % of physical memory).
    public var ramLimitGb: Double?
    /// `0` = never unload on idle.
    public var idleTtlMinutes: Int
    /// `nil` = derive from the model and the free memory.
    public var defaultCtx: Int?
    public var flashAttn: String
    public var kvCacheType: String
    public var gpuLayers: Int
    /// `lru` = unload the least recently used model, `none` = refuse the request.
    public var evictPolicy: String
    /// Models kept in memory: never unloaded on idle, never evicted for room.
    public var pinnedModels: [String]

    public init(maxLoadedModels: Int = 2,
                maxInstancesPerModel: Int = 1,
                defaultParallel: Int = 1,
                ramLimitGb: Double? = nil,
                idleTtlMinutes: Int = 30,
                defaultCtx: Int? = nil,
                flashAttn: String = "auto",
                kvCacheType: String = "f16",
                gpuLayers: Int = -1,
                evictPolicy: String = "lru",
                pinnedModels: [String] = []) {
        self.maxLoadedModels = maxLoadedModels
        self.maxInstancesPerModel = maxInstancesPerModel
        self.defaultParallel = defaultParallel
        self.ramLimitGb = ramLimitGb
        self.idleTtlMinutes = idleTtlMinutes
        self.defaultCtx = defaultCtx
        self.flashAttn = flashAttn
        self.kvCacheType = kvCacheType
        self.gpuLayers = gpuLayers
        self.evictPolicy = evictPolicy
        self.pinnedModels = pinnedModels
    }

    private enum CodingKeys: String, CodingKey {
        case maxLoadedModels, maxInstancesPerModel, defaultParallel, ramLimitGb
        case idleTtlMinutes, defaultCtx, flashAttn, kvCacheType, gpuLayers, evictPolicy
        case pinnedModels
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        maxLoadedModels = try c.decodeIfPresent(Int.self, forKey: .maxLoadedModels) ?? 2
        maxInstancesPerModel = try c.decodeIfPresent(Int.self, forKey: .maxInstancesPerModel) ?? 1
        defaultParallel = try c.decodeIfPresent(Int.self, forKey: .defaultParallel) ?? 1
        ramLimitGb = try c.decodeIfPresent(Double.self, forKey: .ramLimitGb)
        idleTtlMinutes = try c.decodeIfPresent(Int.self, forKey: .idleTtlMinutes) ?? 30
        defaultCtx = try c.decodeIfPresent(Int.self, forKey: .defaultCtx)
        flashAttn = try c.decodeIfPresent(String.self, forKey: .flashAttn) ?? "auto"
        kvCacheType = try c.decodeIfPresent(String.self, forKey: .kvCacheType) ?? "f16"
        gpuLayers = try c.decodeIfPresent(Int.self, forKey: .gpuLayers) ?? -1
        evictPolicy = try c.decodeIfPresent(String.self, forKey: .evictPolicy) ?? "lru"
        pinnedModels = try c.decodeIfPresent([String].self, forKey: .pinnedModels) ?? []
    }
}

/// `GET /api/runtime` and the `{"type":"runtime"}` event body.
public struct RuntimeStatus: Decodable, Hashable, Sendable {
    public var instances: [RuntimeInstance]
    public var system: SystemMemory
    public var budget: MemoryBudget
    public var settings: RuntimeSettings

    public init(instances: [RuntimeInstance] = [],
                system: SystemMemory = SystemMemory(),
                budget: MemoryBudget = MemoryBudget(),
                settings: RuntimeSettings = RuntimeSettings()) {
        self.instances = instances
        self.system = system
        self.budget = budget
        self.settings = settings
    }

    private enum CodingKeys: String, CodingKey { case instances, system, budget, settings }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        instances = try c.decodeIfPresent([RuntimeInstance].self, forKey: .instances) ?? []
        system = try c.decodeIfPresent(SystemMemory.self, forKey: .system) ?? SystemMemory()
        budget = try c.decodeIfPresent(MemoryBudget.self, forKey: .budget) ?? MemoryBudget()
        settings = try c.decodeIfPresent(RuntimeSettings.self, forKey: .settings) ?? RuntimeSettings()
    }

    /// Instances that are not error placeholders, newest first.
    public var liveInstances: [RuntimeInstance] {
        instances.filter { $0.state != .error }
    }
}

// MARK: - Application settings

/// `GET /api/app-settings` → `agent`.
public struct AgentAppConfig: Decodable, Hashable, Sendable {
    public var defaultModel: String?
    public var permissionMode: String?
    public var maxSteps: Int?
    public var thinking: Bool?
    public var temperature: Double?
    public var maxOutputTokens: Int?
    public var autoCompactThreshold: Double?
    public var showReasoning: Bool?
    /// Whether the assistant may find, download and load models itself
    /// ("uruchom mi Qwen3 Coder"). Off unless somebody turns it on.
    public var manageModels: Bool?
    /// Default reasoning effort for new agent sessions (`off|low|medium|high`).
    public var reasoningEffort: String?
    /// Whether the agent may hand a search to a helper with its own context.
    public var subagents: Bool?
    /// Which model the helper runs on; `nil` = the one the agent is on. Only
    /// the user sets this — a helper reads project files, so letting a model
    /// pick a remote one would post them to somebody else's API.
    public var subagentModel: String?
    /// Auto mode's reviewer; `nil` = the model the agent runs on. Only the
    /// user sets this — the reviewer reads the conversation.
    public var autoReviewerModel: String?
    /// Auto mode's context-free local guard; `nil` = automatic (the agent's
    /// model when local, else the smallest local one). Never a remote key.
    public var autoGuardModel: String?
    /// Whether the user's answers on permission cards are remembered per
    /// project and shown to auto mode's reviewer and guard (`nil` = on).
    public var autoPrecedentsEnabled: Bool?

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case defaultModel, permissionMode, maxSteps, thinking, temperature
        case maxOutputTokens, autoCompactThreshold, showReasoning, reasoningEffort
        case manageModels, subagents, subagentModel, autoReviewerModel, autoGuardModel
        case autoPrecedentsEnabled
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        defaultModel = try c.decodeIfPresent(String.self, forKey: .defaultModel)
        permissionMode = try c.decodeIfPresent(String.self, forKey: .permissionMode)
        maxSteps = try c.decodeIfPresent(Int.self, forKey: .maxSteps)
        thinking = try c.decodeIfPresent(Bool.self, forKey: .thinking)
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        maxOutputTokens = try c.decodeIfPresent(Int.self, forKey: .maxOutputTokens)
        autoCompactThreshold = try c.decodeIfPresent(Double.self, forKey: .autoCompactThreshold)
        showReasoning = try c.decodeIfPresent(Bool.self, forKey: .showReasoning)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
        manageModels = try c.decodeIfPresent(Bool.self, forKey: .manageModels)
        subagents = try c.decodeIfPresent(Bool.self, forKey: .subagents)
        subagentModel = try c.decodeIfPresent(String.self, forKey: .subagentModel)
        autoReviewerModel = try c.decodeIfPresent(String.self, forKey: .autoReviewerModel)
        autoGuardModel = try c.decodeIfPresent(String.self, forKey: .autoGuardModel)
        autoPrecedentsEnabled = try c.decodeIfPresent(Bool.self, forKey: .autoPrecedentsEnabled)
    }
}

/// `GET /api/app-settings` → `chat`.
public struct ChatAppConfig: Decodable, Hashable, Sendable {
    public var defaultModel: String?
    public var embeddingModel: String?
    /// Default reasoning effort for conversations that have none of their own.
    public var reasoningEffort: String?

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case defaultModel, embeddingModel, reasoningEffort
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        defaultModel = try c.decodeIfPresent(String.self, forKey: .defaultModel)
        embeddingModel = try c.decodeIfPresent(String.self, forKey: .embeddingModel)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
    }
}

/// `GET /api/app-settings` → `server`.
public struct ServerAppConfig: Decodable, Hashable, Sendable {
    public var host: String?
    public var port: Int?
    public var openaiApiEnabled: Bool
    public var openaiApiRequiresToken: Bool
    public var corsOrigins: [String]

    public init() {
        openaiApiEnabled = false
        openaiApiRequiresToken = false
        corsOrigins = []
    }

    private enum CodingKeys: String, CodingKey {
        case host, port, openaiApiEnabled, openaiApiRequiresToken, corsOrigins
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        host = try c.decodeIfPresent(String.self, forKey: .host)
        port = try c.decodeIfPresent(Int.self, forKey: .port)
        openaiApiEnabled = try c.decodeIfPresent(Bool.self, forKey: .openaiApiEnabled) ?? false
        openaiApiRequiresToken = try c.decodeIfPresent(Bool.self, forKey: .openaiApiRequiresToken) ?? false
        corsOrigins = try c.decodeIfPresent([String].self, forKey: .corsOrigins) ?? []
    }
}

/// `GET /api/app-settings` — where models live, the HF token, and the defaults
/// the chat and the agent pick up.
public struct AppConfig: Decodable, Hashable, Sendable {
    public var modelsDir: String
    public var hfTokenSet: Bool
    public var language: String?
    public var scanLmstudio: Bool
    public var scanHfCache: Bool
    /// "Ulubione": starred model keys, listed first in every picker. Kept in
    /// sync with `FavoriteModels.shared` by the store.
    public var favoriteModels: [String]
    public var version: String?
    public var agent: AgentAppConfig
    public var chat: ChatAppConfig
    public var server: ServerAppConfig
    public var catalog: CatalogAppConfig
    /// What the store downloads by default: which packaging wins and at which
    /// quantisation level.
    public var store: StoreAppConfig
    /// How a downloaded model is contained and checked.
    public var security: SecurityAppConfig
    /// Cerber — the user's antivirus — checking every download first.
    public var cerber: CerberAppConfig

    public init() {
        modelsDir = ""
        hfTokenSet = false
        scanLmstudio = true
        scanHfCache = true
        favoriteModels = []
        agent = AgentAppConfig()
        chat = ChatAppConfig()
        server = ServerAppConfig()
        catalog = CatalogAppConfig()
        store = StoreAppConfig()
        security = SecurityAppConfig()
        cerber = CerberAppConfig()
    }

    private enum CodingKeys: String, CodingKey {
        case modelsDir, hfTokenSet, language, scanLmstudio, scanHfCache, favoriteModels, version
        case agent, chat, server, catalog, store, security, cerber
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelsDir = try c.decodeIfPresent(String.self, forKey: .modelsDir) ?? ""
        hfTokenSet = try c.decodeIfPresent(Bool.self, forKey: .hfTokenSet) ?? false
        language = try c.decodeIfPresent(String.self, forKey: .language)
        scanLmstudio = try c.decodeIfPresent(Bool.self, forKey: .scanLmstudio) ?? true
        scanHfCache = try c.decodeIfPresent(Bool.self, forKey: .scanHfCache) ?? true
        favoriteModels = (try? c.decodeIfPresent([String].self, forKey: .favoriteModels)) ?? []
        version = try c.decodeIfPresent(String.self, forKey: .version)
        agent = try c.decodeIfPresent(AgentAppConfig.self, forKey: .agent) ?? AgentAppConfig()
        chat = try c.decodeIfPresent(ChatAppConfig.self, forKey: .chat) ?? ChatAppConfig()
        server = try c.decodeIfPresent(ServerAppConfig.self, forKey: .server) ?? ServerAppConfig()
        catalog = try c.decodeIfPresent(CatalogAppConfig.self, forKey: .catalog) ?? CatalogAppConfig()
        store = try c.decodeIfPresent(StoreAppConfig.self, forKey: .store) ?? StoreAppConfig()
        security = try c.decodeIfPresent(SecurityAppConfig.self, forKey: .security) ?? SecurityAppConfig()
        cerber = (try? c.decodeIfPresent(CerberAppConfig.self, forKey: .cerber)) ?? CerberAppConfig()
    }
}

/// `GET /api/app-settings` → `cerber`. The token itself never comes back —
/// only whether one is saved.
public struct CerberAppConfig: Decodable, Hashable, Sendable {
    public var enabled: Bool
    public var url: String
    public var tokenSet: Bool
    /// Cerber.app is in /Applications (or ~/Applications).
    public var installed: Bool

    public init() {
        enabled = false
        url = "http://127.0.0.1:8765"
        tokenSet = false
        installed = false
    }

    private enum CodingKeys: String, CodingKey { case enabled, url, tokenSet, installed }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        enabled = (try? c.decode(Bool.self, forKey: .enabled)) ?? false
        url = (try? c.decode(String.self, forKey: .url)) ?? "http://127.0.0.1:8765"
        tokenSet = (try? c.decode(Bool.self, forKey: .tokenSet)) ?? false
        installed = (try? c.decode(Bool.self, forKey: .installed)) ?? false
    }
}

/// `POST /api/cerber/check`.
public struct CerberCheckDTO: Decodable, Hashable, Sendable {
    public var ok: Bool
    public var message: String
    public var installed: Bool

    private enum CodingKeys: String, CodingKey { case ok, message, installed }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        ok = (try? c.decode(Bool.self, forKey: .ok)) ?? false
        message = (try? c.decode(String.self, forKey: .message)) ?? ""
        installed = (try? c.decode(Bool.self, forKey: .installed)) ?? false
    }
}

/// `GET /api/app-settings` → `security` — how a downloaded model is contained.
public struct SecurityAppConfig: Decodable, Hashable, Sendable {
    /// `auto` | `strict` | `off` — whether engine processes run in the macOS
    /// sandbox (no network, no access to the user's files).
    public var sandbox: String
    /// Scan a model's files for executable content after downloading it.
    public var scanDownloads: Bool

    public init() {
        sandbox = "auto"
        scanDownloads = true
    }

    private enum CodingKeys: String, CodingKey {
        case sandbox, scanDownloads
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sandbox = try c.decodeIfPresent(String.self, forKey: .sandbox) ?? "auto"
        scanDownloads = try c.decodeIfPresent(Bool.self, forKey: .scanDownloads) ?? true
    }
}

/// `GET /api/app-settings` → `store`.
public struct StoreAppConfig: Decodable, Hashable, Sendable {
    /// `4bit` … `max` — how heavily a download may be quantised.
    public var quality: String
    /// `mlx` | `gguf` | `auto` — which packaging wins when both exist.
    public var preferFormat: String
    /// The model a fresh install downloads by itself, so the app can answer a
    /// question before anyone has chosen anything.
    public var starterModel: String?
    public var autoInstallStarter: Bool

    public init() {
        quality = "4bit"
        preferFormat = "mlx"
        starterModel = nil
        autoInstallStarter = true
    }

    private enum CodingKeys: String, CodingKey {
        case quality, preferFormat, starterModel, autoInstallStarter
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quality = (try? c.decode(String.self, forKey: .quality)) ?? "4bit"
        preferFormat = (try? c.decode(String.self, forKey: .preferFormat)) ?? "mlx"
        starterModel = try? c.decodeIfPresent(String.self, forKey: .starterModel)
        autoInstallStarter = (try? c.decode(Bool.self, forKey: .autoInstallStarter)) ?? true
    }
}

/// `GET /api/remote/providers` — one cloud provider whose models can be borrowed.
public struct RemoteProviderDTO: Decodable, Identifiable, Hashable, Sendable {
    public let id: String
    public var label: String
    /// A key is saved for this provider (the key itself never leaves the server).
    public var hasKey: Bool
    public var enabled: Bool
    public var baseUrl: String
    /// Where to get a key, for the link next to an empty field.
    public var keyUrl: String
    /// How many models the key can use; `-1` when the provider is not answering.
    public var modelCount: Int
    /// `standard`, `flex` or `priority` — what every request to this provider asks for.
    public var serviceTier: String
    /// The tiers this provider offers; empty when it has none (Anthropic, Gemini).
    public var serviceTiers: [String]

    private enum CodingKeys: String, CodingKey {
        case id, label, hasKey, enabled, baseUrl, keyUrl, modelCount, serviceTier, serviceTiers
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        label = (try? c.decode(String.self, forKey: .label)) ?? id
        hasKey = (try? c.decode(Bool.self, forKey: .hasKey)) ?? false
        enabled = (try? c.decode(Bool.self, forKey: .enabled)) ?? true
        baseUrl = (try? c.decode(String.self, forKey: .baseUrl)) ?? ""
        keyUrl = (try? c.decode(String.self, forKey: .keyUrl)) ?? "https://huggingface.co"
        modelCount = (try? c.decode(Int.self, forKey: .modelCount)) ?? 0
        serviceTier = (try? c.decode(String.self, forKey: .serviceTier)) ?? "standard"
        serviceTiers = (try? c.decode([String].self, forKey: .serviceTiers)) ?? []
    }
}

// MARK: - Events

/// One frame of `/api/events`, decoded far enough to route it.
///
/// A `runtime` frame carries the whole `RuntimeStatus` inline (not nested), a
/// `download` frame carries `job`, and `library` carries nothing at all.
public struct ServerEvent: Decodable, Sendable {
    public var type: String
    public var job: DownloadJob?
    public var runtime: RuntimeStatus?

    private enum CodingKeys: String, CodingKey { case type, job }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        type = try c.decodeIfPresent(String.self, forKey: .type) ?? ""
        job = try c.decodeIfPresent(DownloadJob.self, forKey: .job)
        runtime = type == "runtime" ? try? RuntimeStatus(from: decoder) : nil
    }
}
