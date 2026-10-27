import Foundation
import AppKit

/// A project folder the agent works in.
public struct ProjectRef: Identifiable, Hashable, Sendable {
    public var path: String

    public var id: String { path }
    public var name: String { AgentPaths.name(of: path) }
    public var parent: String { AgentPaths.parent(of: path) }
    public var shortPath: String { AgentPaths.abbreviate(path) }

    public init(path: String) {
        self.path = path
    }

    /// A folder can be renamed or unmounted between two launches.
    public var exists: Bool {
        var isDirectory: ObjCBool = false
        let found = FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        return found && isDirectory.boolValue
    }
}

/// The list of opened project folders, persisted in `UserDefaults`.
///
/// Plain paths are enough: the app is deliberately not sandboxed (it spawns the
/// Python server and reads models from anywhere on disk), so there is nothing
/// to resolve and no security-scoped bookmark to keep alive.
enum RecentProjects {
    private static let key = "codeFeature.recentProjects"
    private static let limit = 20

    static func load() -> [ProjectRef] {
        let stored = UserDefaults.standard.stringArray(forKey: key) ?? []
        var seen = Set<String>()
        return stored.compactMap { path in
            let normalized = normalize(path)
            guard seen.insert(normalized).inserted else { return nil }
            return ProjectRef(path: normalized)
        }
    }

    static func save(_ projects: [ProjectRef]) {
        UserDefaults.standard.set(Array(projects.prefix(limit)).map(\.path), forKey: key)
    }

    /// Adds (or lifts to the top) one folder and returns the new list.
    static func adding(_ path: String, to projects: [ProjectRef]) -> [ProjectRef] {
        let normalized = normalize(path)
        var next = projects.filter { $0.path != normalized }
        next.insert(ProjectRef(path: normalized), at: 0)
        let trimmed = Array(next.prefix(limit))
        save(trimmed)
        return trimmed
    }

    static func removing(_ project: ProjectRef, from projects: [ProjectRef]) -> [ProjectRef] {
        let next = projects.filter { $0.path != project.path }
        save(next)
        return next
    }

    private static func normalize(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    /// „Otwórz folder…” — the folder picker, or `nil` when the user cancels.
    @MainActor
    static func choose() -> String? {
        let panel = NSOpenPanel()
        panel.title = "Wybierz folder projektu"
        panel.prompt = "Otwórz"
        panel.message = "Agent będzie pracował w tym folderze."
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        return url.standardizedFileURL.path
    }
}
