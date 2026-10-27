//
//  Fenced code block: language header, copy button, monospaced body that
//  scrolls horizontally instead of wrapping.
//
import SwiftUI
import AppKit

/// Fenced code block: a header bar with the language and a copy button, and a
/// horizontally scrollable monospaced body.
public struct CodeBlockView: View {
    public let language: String?
    public let content: String
    public var fontSize: CGFloat

    public init(language: String?, content: String, fontSize: CGFloat = 13) {
        self.language = language
        self.content = content
        self.fontSize = fontSize
    }

    @State private var copied = false
    @State private var copyToken = 0

    private var languageLabel: String {
        guard let language, !language.isEmpty else { return "kod" }
        return CodeLanguage.displayName(language)
    }

    private var trimmedContent: String {
        content.trimmingCharacters(in: CharacterSet.newlines)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().overlay(Theme.border)
            ScrollView(.horizontal, showsIndicators: false) {
                Text(trimmedContent.isEmpty ? " " : trimmedContent)
                    .font(.system(size: fontSize, design: .monospaced))
                    .foregroundStyle(Theme.fg)
                    .textSelection(.enabled)
                    .lineSpacing(3)
                    // Code must never wrap or be squeezed: it scrolls instead.
                    .fixedSize()
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .task(id: copyToken) {
            guard copied else { return }
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            if !Task.isCancelled { copied = false }
        }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text(languageLabel)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.fgMuted)
                .lineLimit(1)

            Spacer(minLength: 8)

            Button {
                Pasteboard.copy(trimmedContent)
                copied = true
                copyToken &+= 1
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10, weight: .semibold))
                    Text(copied ? "Skopiowano" : "Kopiuj")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(copied ? Theme.success : Theme.fgMuted)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
            .accessibilityLabel(Text(copied ? "Skopiowano" : "Kopiuj kod"))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Theme.surface2)
    }
}

/// Pretty names for the fence info strings models usually emit.
public enum CodeLanguage {
    private static let names: [String: String] = [
        "bash": "Bash", "c": "C", "cpp": "C++", "c++": "C++", "cs": "C#", "csharp": "C#",
        "css": "CSS", "diff": "Diff", "dockerfile": "Dockerfile", "go": "Go",
        "graphql": "GraphQL", "html": "HTML", "http": "HTTP", "java": "Java",
        "javascript": "JavaScript", "js": "JavaScript", "json": "JSON", "jsx": "JSX",
        "kotlin": "Kotlin", "less": "Less", "lua": "Lua", "makefile": "Makefile",
        "markdown": "Markdown", "md": "Markdown", "objc": "Objective-C", "php": "PHP",
        "plaintext": "Tekst", "text": "Tekst", "txt": "Tekst", "powershell": "PowerShell",
        "python": "Python", "py": "Python", "r": "R", "ruby": "Ruby", "rb": "Ruby",
        "rust": "Rust", "rs": "Rust", "scss": "SCSS", "sh": "Shell", "shell": "Shell",
        "sql": "SQL", "swift": "Swift", "toml": "TOML", "ts": "TypeScript",
        "tsx": "TSX", "typescript": "TypeScript", "xml": "XML", "yaml": "YAML", "yml": "YAML",
        "zsh": "Zsh"
    ]

    public static func displayName(_ raw: String) -> String {
        let key = raw.lowercased()
        return names[key] ?? raw
    }
}

/// The one place either platform's clipboard is touched.
public enum Pasteboard {
    public static func copy(_ string: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
    }
}
