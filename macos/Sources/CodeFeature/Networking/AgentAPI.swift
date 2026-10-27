import Foundation
import LocalCodeKit

/// The coding-agent endpoints, added to the one shared `APIClient` exactly the
/// way `LocalCodeKit` intends feature modules to extend it.
extension APIClient {

    /// `POST /api/agent/sessions {cwd, modelKey?, mode?}`
    public func createAgentSession(cwd: String,
                                   modelKey: String?,
                                   mode: String) async throws -> AgentSessionDTO {
        let body = try encodeBody(CreateAgentSessionBody(cwd: cwd, modelKey: modelKey, mode: mode))
        return try await send(makeRequest("api/agent/sessions", method: "POST", body: body),
                              as: AgentSessionDTO.self)
    }

    /// `GET /api/helpbot`
    public func helpbotStatus() async throws -> HelpbotStatusDTO {
        try await send(makeRequest("api/helpbot"), as: HelpbotStatusDTO.self)
    }

    /// `POST /api/helpbot/install` — starts downloading the help bot's model.
    public func installHelpbot() async throws -> HelpbotStatusDTO {
        try await send(makeRequest("api/helpbot/install", method: "POST"), as: HelpbotStatusDTO.self)
    }

    /// `POST /api/helpbot/session {new}` — the latest help bot conversation, or a fresh one.
    public func helpbotSession(new: Bool) async throws -> AgentSessionDTO {
        let body = try encodeBody(HelpbotSessionBody(new: new))
        return try await send(makeRequest("api/helpbot/session", method: "POST", body: body),
                              as: AgentSessionDTO.self)
    }

    /// `GET /api/agent/sessions[?cwd=]` — newest first, as the server sorts them.
    public func agentSessions(cwd: String? = nil) async throws -> [AgentSessionDTO] {
        let path = query("api/agent/sessions", ["cwd": cwd])
        return try await send(makeRequest(path), as: [AgentSessionDTO].self)
    }

    /// `GET /api/agent/sessions/{id}` → session header, transcript and plan.
    public func agentSession(id: String) async throws -> AgentSessionSnapshot {
        try await send(makeRequest("api/agent/sessions/\(pathSegment(id))"), as: AgentSessionSnapshot.self)
    }

    /// `DELETE /api/agent/sessions/{id}`
    public func deleteAgentSession(id: String) async throws {
        try await sendVoid(makeRequest("api/agent/sessions/\(pathSegment(id))", method: "DELETE"))
    }

    /// `GET /api/library` — the models the agent can be pointed at.
    public func agentModels() async throws -> [AgentModelOption] {
        try await send(makeRequest("api/library"), as: [AgentModelOption].self)
    }

    /// `GET /api/agent/commands` — the slash commands and the `/init` prompt.
    public func agentCommands() async throws -> AgentCommandSet {
        try await send(makeRequest("api/agent/commands"), as: AgentCommandSet.self)
    }

    /// `GET /api/app-settings` — only the agent defaults are read here.
    public func agentDefaults() async throws -> AgentAppDefaults {
        try await send(makeRequest("api/app-settings"), as: AgentAppDefaults.self)
    }
}

private struct CreateAgentSessionBody: Encodable {
    let cwd: String
    let modelKey: String?
    let mode: String
}

/// The `agent` block of `GET /api/app-settings`: what a new session starts with
/// when the user has not picked anything yet.
public struct AgentAppDefaults: Decodable, Sendable {
    public var modelKey: String?
    public var mode: String?

    public init(modelKey: String? = nil, mode: String? = nil) {
        self.modelKey = modelKey
        self.mode = mode
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        guard let agent = try? root.nestedContainer(keyedBy: AgentKeys.self, forKey: .agent) else {
            modelKey = nil
            mode = nil
            return
        }
        modelKey = try? agent.decodeIfPresent(String.self, forKey: .defaultModel)
        mode = try? agent.decodeIfPresent(String.self, forKey: .permissionMode)
    }

    private enum RootKeys: String, CodingKey { case agent }
    private enum AgentKeys: String, CodingKey { case defaultModel, permissionMode }
}

private struct HelpbotSessionBody: Encodable {
    let new: Bool
}
