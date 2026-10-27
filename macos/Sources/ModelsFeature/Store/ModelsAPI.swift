import Foundation
import LocalCodeKit

/// The model, download, runtime and app-settings endpoints, added to the one
/// `APIClient` actor the whole app shares.
///
/// `APIClient` publishes its request plumbing precisely so a feature module can
/// do this: nothing below builds a `URLRequest` by hand or re-implements the
/// Polish error mapping — `makeRequest` + `send` already do both.
extension APIClient {

    // MARK: - Hub

    /// `GET /api/hub/search` — `format` is `any|gguf|mlx`,
    /// `sort` is `trending|downloads|likes|recent`.
    public func hubSearch(query text: String,
                          format: String = "any",
                          sort: String = "trending",
                          limit: Int = 40) async throws -> [HubModel] {
        let path = query("api/hub/search", [
            "q": text,
            "format": format,
            "sort": sort,
            "limit": String(limit),
        ])
        var request = makeRequest(path)
        // Hugging Face can be slow; the default 60 s is plenty but a search
        // should give up sooner than a download would.
        request.timeoutInterval = 30
        return try await send(request, as: [HubModel].self)
    }

    /// `GET /api/hub/curated` — the "Polecane" list, grouped by `category`.
    public func hubCurated() async throws -> [HubModel] {
        var request = makeRequest("api/hub/curated")
        request.timeoutInterval = 30
        return try await send(request, as: [HubModel].self)
    }

    /// `GET /api/hub/model?repo=` — header, README, variants and memory estimates.
    public func hubModel(repo: String) async throws -> HubModelDetail {
        var request = makeRequest(query("api/hub/model", ["repo": repo]))
        request.timeoutInterval = 45
        return try await send(request, as: HubModelDetail.self)
    }

    // MARK: - Downloads

    /// `GET /api/downloads`.
    public func downloadJobs() async throws -> [DownloadJob] {
        try await send(makeRequest("api/downloads"), as: [DownloadJob].self)
    }

    /// `POST /api/downloads {repo, variantId?, files?}`.
    public func startDownload(repo: String,
                              variantId: String? = nil,
                              files: [String]? = nil) async throws -> DownloadJob {
        var body = JSONObject(["repo": .string(repo)])
        if let variantId { body["variantId"] = .string(variantId) }
        if let files, !files.isEmpty { body["files"] = .array(files.map(JSONValue.string)) }
        let data = try encodeBody(body)
        return try await send(makeRequest("api/downloads", method: "POST", body: data),
                              as: DownloadJob.self)
    }

    /// `DELETE /api/downloads/{id}` — cancels a queued or running job.
    public func cancelDownload(id: String) async throws {
        try await sendVoid(makeRequest("api/downloads/\(pathSegment(id))", method: "DELETE"))
    }

    // MARK: - Library

    /// `GET /api/library` — every model on disk.
    public func library() async throws -> [LocalModelDTO] {
        try await send(makeRequest("api/library"), as: [LocalModelDTO].self)
    }

    /// `POST /api/library/rescan` — re-reads the models directory.
    public func rescanLibrary() async throws -> [LocalModelDTO] {
        var request = makeRequest("api/library/rescan", method: "POST")
        // A rescan parses GGUF headers of every file it finds.
        request.timeoutInterval = 180
        return try await send(request, as: [LocalModelDTO].self)
    }

    /// `PATCH /api/library/{key} {loadDefaults}` — `nil` clears the defaults.
    ///
    /// Model keys contain `/`, so the key is one percent-escaped path segment.
    @discardableResult
    public func setLoadDefaults(key: String, defaults: LoadConfigDTO?) async throws -> LocalModelDTO {
        let body = JSONObject(["loadDefaults": defaults.map(Self.jsonValue(for:)) ?? .null])
        let data = try encodeBody(body)
        return try await send(makeRequest("api/library/\(pathSegment(key))", method: "PATCH", body: data),
                              as: LocalModelDTO.self)
    }

    /// `DELETE /api/library/{key}` — only models LocalCode downloaded itself.
    public func deleteLocalModel(key: String) async throws {
        try await sendVoid(makeRequest("api/library/\(pathSegment(key))", method: "DELETE"))
    }

    // MARK: - Runtime

    /// `GET /api/runtime`.
    public func runtimeStatus() async throws -> RuntimeStatus {
        try await send(makeRequest("api/runtime"), as: RuntimeStatus.self)
    }

    /// `PATCH /api/runtime/settings` — a partial patch; an explicit `null`
    /// means "automatic" for `ramLimitGb` and `defaultCtx`.
    @discardableResult
    public func patchRuntimeSettings(_ patch: JSONObject) async throws -> RuntimeStatus {
        let data = try encodeBody(patch)
        return try await send(makeRequest("api/runtime/settings", method: "PATCH", body: data),
                              as: RuntimeStatus.self)
    }

    /// `POST /api/runtime/load` — returns as soon as the instance exists; its
    /// state is usually `loading` and turns `ready` in an `/api/events` frame.
    @discardableResult
    public func loadModel(key: String,
                          config: LoadConfigDTO? = nil,
                          newInstance: Bool = false) async throws -> RuntimeInstance {
        var body = JSONObject(["modelKey": .string(key)])
        if let config, !config.isEmpty { body["config"] = Self.jsonValue(for: config) }
        if newInstance { body["newInstance"] = .bool(true) }
        let data = try encodeBody(body)
        var request = makeRequest("api/runtime/load", method: "POST", body: data)
        request.timeoutInterval = 120
        return try await send(request, as: RuntimeInstance.self)
    }

    /// `POST /api/runtime/unload {instanceId}`.
    @discardableResult
    public func unloadInstance(id: String) async throws -> RuntimeStatus {
        let data = try encodeBody(JSONObject(["instanceId": .string(id)]))
        var request = makeRequest("api/runtime/unload", method: "POST", body: data)
        request.timeoutInterval = 120
        return try await send(request, as: RuntimeStatus.self)
    }

    /// `POST /api/runtime/unload {all: true}`.
    @discardableResult
    public func unloadAllInstances() async throws -> RuntimeStatus {
        let data = try encodeBody(JSONObject(["all": .bool(true)]))
        var request = makeRequest("api/runtime/unload", method: "POST", body: data)
        request.timeoutInterval = 120
        return try await send(request, as: RuntimeStatus.self)
    }

    /// `POST /api/runtime/estimate` — drives the live memory readout while the
    /// context slider moves.
    public func estimateMemory(key: String, config: LoadConfigDTO? = nil) async throws -> MemoryEstimate {
        var body = JSONObject(["modelKey": .string(key)])
        if let config, !config.isEmpty { body["config"] = Self.jsonValue(for: config) }
        let data = try encodeBody(body)
        var request = makeRequest("api/runtime/estimate", method: "POST", body: data)
        request.timeoutInterval = 20
        return try await send(request, as: MemoryEstimate.self)
    }

    // MARK: - Application settings

    /// `POST /api/library/import` — add a model the person already has on disk.
    public func importModel(path: String, name: String? = nil,
                            move: Bool = false) async throws -> LocalModelDTO {
        var fields: [String: JSONValue] = ["path": .string(path), "move": .bool(move)]
        if let name, !name.isEmpty { fields["name"] = .string(name) }
        var request = makeRequest("api/library/import", method: "POST",
                                  body: try encodeBody(JSONObject(fields)))
        request.timeoutInterval = 600      // copying tens of gigabytes takes a while
        return try await send(request, as: LocalModelDTO.self)
    }

    /// `GET /api/remote/providers`.
    public func remoteProviders() async throws -> [RemoteProviderDTO] {
        try await send(makeRequest("api/remote/providers"), as: [RemoteProviderDTO].self)
    }

    /// `PUT /api/remote/{provider}/key` — the server verifies before saving.
    @discardableResult
    public func setRemoteKey(_ provider: String, key: String?) async throws -> [RemoteProviderDTO] {
        let body = try encodeBody(JSONObject(["key": key.map { .string($0) } ?? .null]))
        var request = makeRequest("api/remote/\(provider)/key", method: "PUT", body: body)
        request.timeoutInterval = 60        // it calls the provider before answering
        return try await send(request, as: [RemoteProviderDTO].self)
    }

    /// `PATCH /api/remote/{provider}` — switch a saved provider off or on.
    @discardableResult
    public func setRemoteEnabled(_ provider: String,
                                 enabled: Bool) async throws -> [RemoteProviderDTO] {
        let body = try encodeBody(JSONObject(["enabled": .bool(enabled)]))
        return try await send(makeRequest("api/remote/\(provider)", method: "PATCH", body: body),
                              as: [RemoteProviderDTO].self)
    }

    /// `PATCH /api/remote/{provider}` — Standard / Flex / Priority for every request.
    @discardableResult
    public func setRemoteServiceTier(_ provider: String,
                                     tier: String) async throws -> [RemoteProviderDTO] {
        let body = try encodeBody(JSONObject(["serviceTier": .string(tier)]))
        return try await send(makeRequest("api/remote/\(provider)", method: "PATCH", body: body),
                              as: [RemoteProviderDTO].self)
    }

    /// `GET /api/app-settings`.
    public func appSettings() async throws -> AppConfig {
        try await send(makeRequest("api/app-settings"), as: AppConfig.self)
    }

    /// `PATCH /api/app-settings` — a partial patch, nested objects included.
    @discardableResult
    public func patchAppSettings(_ patch: JSONObject) async throws -> AppConfig {
        let data = try encodeBody(patch)
        return try await send(makeRequest("api/app-settings", method: "PATCH", body: data),
                              as: AppConfig.self)
    }

    /// `DELETE /api/agent/auto-precedents` — forgets what the agent learned from
    /// the user's answers on permission cards, in every project; returns how many.
    @discardableResult
    public func clearAutoPrecedents() async throws -> Int {
        try await send(makeRequest("api/agent/auto-precedents", method: "DELETE"),
                       as: RemovedCount.self).removed
    }

    /// `PUT /api/app-settings/hf-token {token|null}` — `nil` forgets the token.
    @discardableResult
    public func setHuggingFaceToken(_ token: String?) async throws -> AppConfig {
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines)
        let value: JSONValue = (trimmed?.isEmpty == false) ? .string(trimmed!) : .null
        let data = try encodeBody(JSONObject(["token": value]))
        return try await send(makeRequest("api/app-settings/hf-token", method: "PUT", body: data),
                              as: AppConfig.self)
    }

    /// `PUT /api/app-settings/cerber-token {token|null}` — refused only when Cerber rejects it.
    @discardableResult
    public func setCerberToken(_ token: String?) async throws -> AppConfig {
        let trimmed = token?.trimmingCharacters(in: .whitespacesAndNewlines)
        let value: JSONValue = (trimmed?.isEmpty == false) ? .string(trimmed!) : .null
        let data = try encodeBody(JSONObject(["token": value]))
        return try await send(makeRequest("api/app-settings/cerber-token", method: "PUT", body: data),
                              as: AppConfig.self)
    }

    /// `POST /api/cerber/check` — is Cerber open, with its API on, accepting our token.
    public func checkCerber() async throws -> CerberCheckDTO {
        try await send(makeRequest("api/cerber/check", method: "POST"), as: CerberCheckDTO.self)
    }

    // MARK: - Helpers

    /// `LoadConfigDTO` as a JSON object, omitting the fields left on "Auto".
    static func jsonValue(for config: LoadConfigDTO) -> JSONValue {
        var fields: [String: JSONValue] = [:]
        if let value = config.nCtx { fields["nCtx"] = .int(value) }
        if let value = config.nParallel { fields["nParallel"] = .int(value) }
        if let value = config.gpuLayers { fields["gpuLayers"] = .int(value) }
        if let value = config.flashAttn { fields["flashAttn"] = .string(value) }
        if let value = config.kvCacheType { fields["kvCacheType"] = .string(value) }
        if let value = config.embedding { fields["embedding"] = .bool(value) }
        return .object(fields)
    }
}

/// `{removed}` of a clearing `DELETE`.
private struct RemovedCount: Decodable {
    let removed: Int
}
