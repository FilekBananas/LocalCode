//
//  The documentation that ships inside the app.
//
//  `docs/UZYCIE.md` (the user guide) and `docs/API.md` (the local HTTP API) are
//  copied into `LocalCode.app/Contents/Resources/Docs/` by
//  `scripts/build-app.sh`, so the viewer works offline and always matches the
//  build. Everything here except the folder panel is a pure function of its
//  input, which keeps the parsing and the naming rules checkable without a UI.
//
import AppKit
import Foundation

// MARK: - Pages

/// One of the two bundled documents.
public enum DocumentationPage: String, CaseIterable, Identifiable, Hashable, Sendable {
    case guide
    case api

    public var id: String { rawValue }

    /// Segment title in the viewer.
    public var title: String {
        switch self {
        case .guide: "Jak używać"
        case .api: "API (localhost)"
        }
    }

    /// The file's name in `docs/` and in the bundle's `Docs/` folder.
    public var fileName: String {
        switch self {
        case .guide: "UZYCIE.md"
        case .api: "API.md"
        }
    }

    /// The name the file gets when the user downloads it.
    public var exportName: String {
        switch self {
        case .guide: "LocalCode-dokumentacja.md"
        case .api: "LocalCode-API.md"
        }
    }

    /// The page a relative link such as `API.md` or `./UZYCIE.md` points at.
    public init?(linkPath: String) {
        let name = (linkPath as NSString).lastPathComponent.lowercased()
        guard let page = Self.allCases.first(where: { $0.fileName.lowercased() == name }) else {
            return nil
        }
        self = page
    }
}

/// The id of the documentation window, shared by the scene that declares it
/// and every place that opens it (Help menu, Settings).
public enum DocumentationWindow {
    public static let id = "localcode-documentation"
}

// MARK: - Loading

/// Finds the bundled documents.
public enum DocumentationLibrary {
    public static let unavailableMessage = "Dokumentacja nie jest dołączona do tej kompilacji."

    /// Where `page` lives: the app bundle first, then — for `swift run` and
    /// other unbundled builds — the repository's `docs/` folder.
    public static func url(for page: DocumentationPage) -> URL? {
        let resource = (page.fileName as NSString).deletingPathExtension
        if let bundled = Bundle.main.url(forResource: resource, withExtension: "md", subdirectory: "Docs") {
            return bundled
        }
        guard let folder = repositoryDocs else { return nil }
        let url = folder.appendingPathComponent(page.fileName)
        return FileManager.default.isReadableFile(atPath: url.path) ? url : nil
    }

    /// True when both documents can be read.
    public static var isAvailable: Bool {
        DocumentationPage.allCases.allSatisfy { url(for: $0) != nil }
    }

    public static func text(for page: DocumentationPage) -> String? {
        guard let url = url(for: page) else { return nil }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// `<repo>/docs`, found by walking up from the executable: a debug build
    /// runs from `macos/.build/<triple>/debug/LocalCode`, four levels below the
    /// repository.
    private static let repositoryDocs: URL? = {
        guard var folder = Bundle.main.executableURL?
            .resolvingSymlinksInPath()
            .deletingLastPathComponent() else { return nil }
        for _ in 0..<8 {
            let candidate = folder.appendingPathComponent("docs", isDirectory: true)
            let marker = candidate.appendingPathComponent(DocumentationPage.guide.fileName)
            if FileManager.default.fileExists(atPath: marker.path) { return candidate }
            let parent = folder.deletingLastPathComponent()
            if parent.path == folder.path { break }
            folder = parent
        }
        return nil
    }()
}

// MARK: - Markdown

/// A block of a bundled document, with the anchor its heading answers to.
public struct DocumentationBlock: Identifiable, Hashable, Sendable {
    public let id: Int
    public let block: MarkdownBlock
    /// GitHub-style anchor (`pierwsze-uruchomienie`) for headings, else nil.
    public let anchor: String?

    public init(id: Int, block: MarkdownBlock, anchor: String?) {
        self.id = id
        self.block = block
        self.anchor = anchor
    }
}

/// An entry of the viewer's table of contents.
public struct DocumentationHeading: Identifiable, Hashable, Sendable {
    public let anchor: String
    public let level: Int
    /// Plain text: inline markup already removed.
    public let title: String

    public var id: String { anchor }

    public init(anchor: String, level: Int, title: String) {
        self.anchor = anchor
        self.level = level
        self.title = title
    }
}

/// Where a link inside a document leads.
public enum DocumentationLink: Equatable, Sendable {
    /// `http(s)` and `mailto`: the system opens it.
    case external(URL)
    /// `#anchor` in the open document.
    case anchor(String)
    /// The other bundled document, optionally at a heading.
    case page(DocumentationPage, anchor: String?)
    /// A relative file the app does not ship (README.md, IMAGES.md…), by name.
    case missing(String)
}

/// Turns a bundled document into blocks the viewer can render and jump to.
///
/// The block grammar itself is `MarkdownParser`, the same one the chat uses;
/// this layer only adds what a document needs and a chat message does not:
/// heading anchors and soft line wrapping.
public enum DocumentationMarkdown {

    public static func blocks(_ markdown: String) -> [DocumentationBlock] {
        var seen: [String: Int] = [:]
        return MarkdownParser.parse(markdown).enumerated().map { index, block in
            if case let .heading(_, text) = block {
                // GitHub numbers repeated headings: `czat`, `czat-1`, `czat-2`.
                let base = slug(text)
                let count = seen[base, default: 0]
                seen[base] = count + 1
                return DocumentationBlock(id: index, block: block,
                                          anchor: count == 0 ? base : "\(base)-\(count)")
            }
            return DocumentationBlock(id: index, block: unwrapped(block), anchor: nil)
        }
    }

    /// Headings for the table of contents (`##` and `###` by default; the
    /// single `#` is the document title).
    public static func outline(_ blocks: [DocumentationBlock],
                               levels: ClosedRange<Int> = 2...3) -> [DocumentationHeading] {
        blocks.compactMap { item in
            guard case let .heading(level, text) = item.block, levels.contains(level),
                  let anchor = item.anchor else { return nil }
            return DocumentationHeading(anchor: anchor, level: level, title: plainText(text))
        }
    }

    /// The anchor GitHub gives a heading: lowercase, every character that is
    /// not a letter, digit, mark, `_`, `-` or space dropped, spaces turned
    /// into hyphens. `Kod: agent w Twoim projekcie` → `kod-agent-w-twoim-projekcie`.
    public static func slug(_ heading: String) -> String {
        var result = String.UnicodeScalarView()
        for scalar in plainText(heading).lowercased().unicodeScalars {
            switch scalar.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
                 .nonspacingMark, .spacingMark, .enclosingMark,
                 .decimalNumber, .letterNumber, .otherNumber,
                 .connectorPunctuation:
                result.append(scalar)
            default:
                if scalar == " " { result.append("-") } else if scalar == "-" { result.append(scalar) }
            }
        }
        return normalizedAnchor(String(result))
    }

    /// Inline markdown without its markup: `Endpointy `/api/*`` → `Endpointy /api/*`.
    public static func plainText(_ inline: String) -> String {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        options.failurePolicy = .returnPartiallyParsedIfPossible
        guard let parsed = try? AttributedString(markdown: inline, options: options) else { return inline }
        return String(parsed.characters)
    }

    /// Classifies a link the inline parser produced. `nil` for links that lead
    /// nowhere useful (`file:`, an empty `#`).
    public static func link(_ url: URL) -> DocumentationLink? {
        if let scheme = url.scheme?.lowercased() {
            return ["http", "https", "mailto"].contains(scheme) ? .external(url) : nil
        }
        let parts = url.relativeString.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)
        let rawPath = String(parts.first ?? "")
        let path = rawPath.removingPercentEncoding ?? rawPath
        let anchor = parts.count > 1 ? String(parts[1]) : ""
        let fragment = anchor.isEmpty ? nil : normalizedAnchor(anchor.removingPercentEncoding ?? anchor)

        if path.isEmpty { return fragment.map { .anchor($0) } }
        if let page = DocumentationPage(linkPath: path) { return .page(page, anchor: fragment) }
        return .missing((path as NSString).lastPathComponent)
    }

    // MARK: Helpers

    /// The documents are wrapped at ~80 columns. `MarkdownParser` keeps those
    /// line breaks (right for chat, where a newline is meant), so a document
    /// joins them back into flowing text, as any Markdown renderer would.
    static func unwrapped(_ block: MarkdownBlock) -> MarkdownBlock {
        switch block {
        case let .paragraph(text): .paragraph(unwrap(text))
        case let .quote(text): .quote(unwrap(text))
        case let .bulletList(items): .bulletList(items.map(unwrap))
        case let .numberedList(items): .numberedList(items.map(unwrap))
        default: block
        }
    }

    static func unwrap(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ")
    }

    /// Anchors compare lowercased and in composed Unicode form, so `myślenia`
    /// typed either way finds the heading.
    static func normalizedAnchor(_ anchor: String) -> String {
        anchor.lowercased().precomposedStringWithCanonicalMapping
    }
}

// MARK: - Download

/// Saves both documents into a folder the user picks.
public enum DocumentationExport {

    /// What a finished save reports to the screen that started it.
    public struct Saved: Sendable {
        public let folder: URL
        public let files: [URL]

        /// `Zapisano w „Pobrane”: LocalCode-dokumentacja.md, LocalCode-API.md.`
        public var message: String {
            let folderName = FileManager.default.displayName(atPath: folder.path)
            let names = files.map(\.lastPathComponent).joined(separator: ", ")
            return "Zapisano w „\(folderName)”: \(names)."
        }
    }

    public enum Failure: LocalizedError {
        case unavailable
        case write(String, Error)

        public var errorDescription: String? {
            switch self {
            case .unavailable:
                DocumentationLibrary.unavailableMessage
            case let .write(name, error):
                "Nie udało się zapisać „\(name)”: \(error.localizedDescription)"
            }
        }
    }

    /// Asks for a folder (Downloads by default), writes both files without
    /// overwriting anything, and shows them in Finder. `nil` when the user
    /// cancels the panel.
    @MainActor
    public static func chooseFolderAndSave() -> Result<Saved, Failure>? {
        guard DocumentationLibrary.isAvailable else { return .failure(.unavailable) }

        let panel = NSOpenPanel()
        panel.title = "Pobierz dokumentację"
        panel.message = "Wybierz folder na dokumentację LocalCode (dwa pliki Markdown)."
        panel.prompt = "Zapisz tutaj"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
        guard panel.runModal() == .OK, let folder = panel.url else { return nil }

        do {
            let files = try save(to: folder)
            NSWorkspace.shared.activateFileViewerSelecting(files)
            return .success(Saved(folder: folder, files: files))
        } catch let failure as Failure {
            return .failure(failure)
        } catch {
            return .failure(.write(folder.lastPathComponent, error))
        }
    }

    /// The Help menu's variant: Finder showing the files is the confirmation,
    /// and a failure gets an alert, because no screen is there to show it.
    @MainActor
    public static func runFromMenu() {
        guard case let .failure(failure) = chooseFolderAndSave() else { return }
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Nie udało się pobrać dokumentacji"
        alert.informativeText = failure.localizedDescription
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    /// Copies both documents into `folder` under their export names.
    public static func save(to folder: URL, fileManager: FileManager = .default) throws -> [URL] {
        var written: [URL] = []
        for page in DocumentationPage.allCases {
            guard let source = DocumentationLibrary.url(for: page) else { throw Failure.unavailable }
            let target = availableURL(for: page.exportName, in: folder) {
                fileManager.fileExists(atPath: $0.path)
            }
            do {
                let data = try Data(contentsOf: source)
                // `.withoutOverwriting` closes the gap between the check above
                // and the write: a file that appeared meanwhile fails the save
                // instead of being replaced.
                try data.write(to: target, options: .withoutOverwriting)
            } catch {
                throw Failure.write(target.lastPathComponent, error)
            }
            written.append(target)
        }
        return written
    }

    /// `name` in `folder`, or `name (2)`, `name (3)`… — the first one that
    /// does not exist yet. Finder numbers duplicates the same way.
    public static func availableURL(for name: String, in folder: URL, exists: (URL) -> Bool) -> URL {
        let first = folder.appendingPathComponent(name)
        guard exists(first) else { return first }
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        var number = 2
        while true {
            let candidate = folder.appendingPathComponent(
                ext.isEmpty ? "\(base) (\(number))" : "\(base) (\(number)).\(ext)")
            if !exists(candidate) { return candidate }
            number += 1
        }
    }
}
