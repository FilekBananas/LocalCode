import SwiftUI
import UniformTypeIdentifiers

/// The sections of the navigation rail.
///
/// It lives in the kit rather than in the executable so a feature module can
/// ask the shell to switch sections — "Pobrano model, pokaż Moje modele" — via
/// `ShellBus` without depending on the app target.
public enum AppSection: String, CaseIterable, Identifiable, Hashable, Sendable {
    case chat
    case code
    case images
    case speech
    case video
    case discover
    case myModels
    case runtime
    case settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .chat: "Czat"
        case .code: "Kod"
        case .images: "Obrazy"
        case .speech: "Mowa"
        case .video: "Wideo"
        case .discover: "Odkrywaj"
        case .myModels: "Moje modele"
        case .runtime: "Runtime"
        case .settings: "Ustawienia"
        }
    }

    public var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .code: "chevron.left.forwardslash.chevron.right"
        case .images: "photo.on.rectangle.angled"
        case .speech: "waveform"
        case .video: "film"
        case .discover: "magnifyingglass"
        case .myModels: "shippingbox"
        case .runtime: "cpu"
        case .settings: "gearshape"
        }
    }

    /// Tooltip in the rail.
    public var hint: String {
        switch self {
        case .chat: "Rozmowa z modelem lokalnym"
        case .code: "Agent programistyczny"
        case .images: "Generowanie obrazów lokalnie"
        case .speech: "Dyktowanie i czytanie na głos"
        case .video: "Krótkie klipy wideo z opisu"
        case .discover: "Sklep z modelami"
        case .myModels: "Modele pobrane na ten komputer"
        case .runtime: "Załadowane modele i pamięć"
        case .settings: "Ustawienia aplikacji"
        }
    }
}

/// One-way channel from places that cannot reach the shell's `@State` — the
/// macOS menu bar above all — to the shell that owns the window.
///
/// `tick` is a counter rather than a nil-able value so the same action can be
/// requested twice in a row and still be observed twice.
@MainActor
@Observable
public final class ShellBus {
    public static let shared = ShellBus()

    public enum Action: Equatable, Sendable {
        case newConversation
        case search
        case toggleSidebar
        case stopStreaming
        case togglePin
        case deleteConversation
        case export(String)
        case show(AppSection)
        case refresh
    }

    public private(set) var latest: Action?
    public private(set) var tick = 0

    public init() {}

    public func request(_ action: Action) {
        latest = action
        tick &+= 1
    }
}

/// Carries bytes the server produced straight into `fileExporter`.
public struct ExportDocument: FileDocument {
    public static let markdown = UTType(filenameExtension: "md") ?? .plainText

    public static var readableContentTypes: [UTType] { [markdown, .plainText, .json] }

    public var data: Data

    public init(data: Data) { self.data = data }

    public init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    public func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }

    /// A file name a save panel accepts, derived from a conversation title.
    public static func filename(for title: String, format: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " -_"))
        let cleaned = title.unicodeScalars
            .map { allowed.contains($0) ? Character($0) : "-" }
            .reduce(into: "") { $0.append($1) }
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let base = cleaned.isEmpty ? "rozmowa" : String(cleaned.prefix(60))
        return "\(base).\(format)"
    }
}
