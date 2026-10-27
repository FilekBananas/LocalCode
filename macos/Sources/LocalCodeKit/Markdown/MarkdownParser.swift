//
//  Markdown block parser.
//
//  Local models stream answers token by token, so this parser runs again on
//  every delta: it has to be cheap and it has to survive half-written syntax —
//  above all an unterminated code fence, which is what most partial answers
//  look like. Only inline spans are handed to `AttributedString(markdown:)`;
//  fenced code, tables and quotes are block syntax it cannot express.
//
import Foundation
import SwiftUI

/// One rendered chunk of a message. The chat streams text in, so the parser is
/// re-run on every token; it must therefore be cheap and tolerant of half
/// finished syntax (above all: unterminated fenced code blocks).
public enum MarkdownBlock: Hashable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bulletList([String])
    case numberedList([String])
    case code(language: String?, content: String)
    case quote(String)
    case table(header: [String], rows: [[String]])
    case divider
}

/// A hand written block parser. `AttributedString(markdown:)` cannot do fenced
/// code, tables or quotes, so only *inline* spans are handed to it.
public enum MarkdownParser {

    // MARK: - Block parsing

    public static func parse(_ text: String) -> [MarkdownBlock] {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let lines = normalized.components(separatedBy: "\n")

        var blocks: [MarkdownBlock] = []
        var index = 0

        while index < lines.count {
            let raw = lines[index]
            let line = raw.trimmingCharacters(in: .whitespaces)

            if line.isEmpty {
                index += 1
                continue
            }

            // Fenced code. An unterminated fence runs to the end of the text,
            // which is what streaming responses look like most of the time.
            if let fence = Fence(line) {
                index += 1
                var body: [String] = []
                while index < lines.count {
                    if fence.closes(lines[index].trimmingCharacters(in: .whitespaces)) {
                        index += 1
                        break
                    }
                    body.append(lines[index])
                    index += 1
                }
                blocks.append(.code(language: fence.language, content: body.joined(separator: "\n")))
                continue
            }

            // ATX heading
            if let heading = headingBlock(line) {
                blocks.append(heading)
                index += 1
                continue
            }

            // Thematic break
            if isDivider(line) {
                blocks.append(.divider)
                index += 1
                continue
            }

            // GFM table: a header row followed by a separator row.
            if line.contains("|"), index + 1 < lines.count,
               isTableSeparator(lines[index + 1]) {
                let header = splitRow(line)
                var rows: [[String]] = []
                index += 2
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    if candidate.isEmpty || !candidate.contains("|") { break }
                    rows.append(padded(splitRow(candidate), to: header.count))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            // Block quote, with lazy continuation lines.
            if line.hasPrefix(">") {
                var parts: [String] = []
                while index < lines.count {
                    let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                    if candidate.hasPrefix(">") {
                        var content = String(candidate.dropFirst())
                        if content.hasPrefix(" ") { content.removeFirst() }
                        parts.append(content)
                        index += 1
                    } else if !candidate.isEmpty && !startsBlock(lines[index]) {
                        parts.append(candidate)
                        index += 1
                    } else {
                        break
                    }
                }
                let quote = parts.joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                blocks.append(.quote(quote))
                continue
            }

            // Bullet list
            if let first = bulletItem(raw) {
                var items = [first]
                index += 1
                collect(lines: lines, index: &index, items: &items, item: bulletItem)
                blocks.append(.bulletList(items))
                continue
            }

            // Numbered list
            if let first = numberedItem(raw) {
                var items = [first]
                index += 1
                collect(lines: lines, index: &index, items: &items, item: numberedItem)
                blocks.append(.numberedList(items))
                continue
            }

            // Paragraph: everything up to a blank line or the next block start.
            var parts = [line]
            index += 1
            while index < lines.count {
                let candidate = lines[index].trimmingCharacters(in: .whitespaces)
                if candidate.isEmpty || startsBlock(lines[index]) { break }
                if candidate.contains("|"), index + 1 < lines.count,
                   isTableSeparator(lines[index + 1]) { break }
                parts.append(candidate)
                index += 1
            }
            blocks.append(.paragraph(parts.joined(separator: "\n")))
        }

        return blocks
    }

    // MARK: - Inline spans

    /// Converts inline markdown (bold, italic, code, links, strikethrough) into
    /// an `AttributedString`. Block syntax is deliberately left untouched.
    public static func inline(_ text: String, textStyle: Font.TextStyle = .body) -> AttributedString {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        options.failurePolicy = .returnPartiallyParsedIfPossible
        options.allowsExtendedAttributes = true

        var attributed: AttributedString
        do {
            attributed = try AttributedString(markdown: text, options: options)
        } catch {
            return AttributedString(text)
        }

        // Style code spans and links; bold/italic are handled by SwiftUI itself.
        var codeRanges: [Range<AttributedString.Index>] = []
        var linkRanges: [Range<AttributedString.Index>] = []
        for run in attributed.runs {
            if run.inlinePresentationIntent?.contains(.code) == true {
                codeRanges.append(run.range)
            }
            if run.link != nil {
                linkRanges.append(run.range)
            }
        }
        for range in codeRanges {
            attributed[range].font = .system(textStyle, design: .monospaced)
            attributed[range].foregroundColor = Theme.accent
            attributed[range].backgroundColor = Theme.codeBg
        }
        for range in linkRanges {
            attributed[range].foregroundColor = Theme.accent
            attributed[range].underlineStyle = .single
        }
        return attributed
    }

    /// Plain text of a message, used for search snippets and copy-to-clipboard.
    public static func plainText(_ text: String) -> String {
        parse(text).map { block in
            switch block {
            case let .heading(_, text): return text
            case let .paragraph(text): return text
            case let .bulletList(items): return items.joined(separator: "\n")
            case let .numberedList(items): return items.joined(separator: "\n")
            case let .code(_, content): return content
            case let .quote(text): return text
            case let .table(header, rows):
                return ([header] + rows).map { $0.joined(separator: " ") }.joined(separator: "\n")
            case .divider: return ""
            }
        }
        .filter { !$0.isEmpty }
        .joined(separator: "\n\n")
    }

    // MARK: - Helpers

    /// Opening/closing fence bookkeeping (``` and ~~~, three or more markers).
    private struct Fence {
        let marker: Character
        let count: Int
        let language: String?

        init?(_ trimmed: String) {
            guard let first = trimmed.first, first == "`" || first == "~" else { return nil }
            let count = trimmed.prefix { $0 == first }.count
            guard count >= 3 else { return nil }
            let info = String(trimmed.dropFirst(count)).trimmingCharacters(in: .whitespaces)
            // An info string on a backtick fence may not contain a backtick.
            if first == "`" && info.contains("`") { return nil }
            self.marker = first
            self.count = count
            let token = info.split(separator: " ").first.map(String.init) ?? ""
            self.language = token.isEmpty ? nil : token.lowercased()
        }

        func closes(_ trimmed: String) -> Bool {
            guard let first = trimmed.first, first == marker else { return false }
            let run = trimmed.prefix { $0 == marker }.count
            guard run >= count else { return false }
            return trimmed.dropFirst(run).trimmingCharacters(in: .whitespaces).isEmpty
        }
    }

    private static func headingBlock(_ line: String) -> MarkdownBlock? {
        guard line.hasPrefix("#") else { return nil }
        let level = line.prefix { $0 == "#" }.count
        guard level <= 6 else { return nil }
        let rest = String(line.dropFirst(level))
        guard rest.isEmpty || rest.hasPrefix(" ") || rest.hasPrefix("\t") else { return nil }
        var text = rest.trimmingCharacters(in: .whitespaces)
        // Strip an optional closing "###" sequence, but keep things like "C#".
        if text.hasSuffix("#") {
            var stripped = text
            while stripped.hasSuffix("#") { stripped.removeLast() }
            if stripped.isEmpty || stripped.hasSuffix(" ") {
                text = stripped.trimmingCharacters(in: .whitespaces)
            }
        }
        return .heading(level: level, text: text)
    }

    private static func isDivider(_ trimmed: String) -> Bool {
        let stripped = trimmed.filter { $0 != " " && $0 != "\t" }
        guard stripped.count >= 3 else { return false }
        return stripped.allSatisfy { $0 == "-" }
            || stripped.allSatisfy { $0 == "*" }
            || stripped.allSatisfy { $0 == "_" }
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("-"), trimmed.contains("|") else { return false }
        guard trimmed.allSatisfy({ $0 == "-" || $0 == ":" || $0 == "|" || $0 == " " || $0 == "\t" })
        else { return false }
        let cells = splitRow(trimmed)
        guard !cells.isEmpty else { return false }
        return cells.allSatisfy { cell in
            let body = cell.trimmingCharacters(in: .whitespaces)
            guard !body.isEmpty else { return false }
            let dashes = body.drop { $0 == ":" }.prefix { $0 == "-" }
            guard !dashes.isEmpty else { return false }
            let tail = body.drop { $0 == ":" }.drop { $0 == "-" }
            return tail.isEmpty || tail == ":"
        }
    }

    /// Splits `| a | b |` into cells, honouring the `\|` escape.
    private static func splitRow(_ line: String) -> [String] {
        var body = Substring(line.trimmingCharacters(in: .whitespaces))
        if body.hasPrefix("|") { body = body.dropFirst() }
        if body.hasSuffix("|") { body = body.dropLast() }

        var cells: [String] = []
        var current = ""
        var escaped = false
        for character in body {
            if escaped {
                if character != "|" { current.append("\\") }
                current.append(character)
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "|" {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(character)
            }
        }
        if escaped { current.append("\\") }
        cells.append(current.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func padded(_ cells: [String], to count: Int) -> [String] {
        if cells.count == count { return cells }
        if cells.count > count { return Array(cells.prefix(count)) }
        return cells + Array(repeating: "", count: count - cells.count)
    }

    private static func bulletItem(_ line: String) -> String? {
        let body = line.drop { $0 == " " || $0 == "\t" }
        guard let marker = body.first, marker == "-" || marker == "*" || marker == "+" else { return nil }
        if isDivider(String(body)) { return nil }
        let rest = body.dropFirst()
        guard let next = rest.first, next == " " || next == "\t" else { return nil }
        var item = rest.trimmingCharacters(in: .whitespaces)
        // Task list markers render as symbols instead of raw brackets.
        if item.hasPrefix("[ ] ") { item = "☐ " + String(item.dropFirst(4)) }
        else if item.lowercased().hasPrefix("[x] ") { item = "☑ " + String(item.dropFirst(4)) }
        return item
    }

    private static func numberedItem(_ line: String) -> String? {
        let body = line.drop { $0 == " " || $0 == "\t" }
        let digits = body.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 9 else { return nil }
        let afterDigits = body.dropFirst(digits.count)
        guard let delimiter = afterDigits.first, delimiter == "." || delimiter == ")" else { return nil }
        let rest = afterDigits.dropFirst()
        guard let next = rest.first, next == " " || next == "\t" else { return nil }
        return rest.trimmingCharacters(in: .whitespaces)
    }

    /// Consumes the remaining items of a list, folding wrapped lines into the
    /// item they continue.
    private static func collect(lines: [String],
                                index: inout Int,
                                items: inout [String],
                                item: (String) -> String?) {
        while index < lines.count {
            let raw = lines[index]
            let trimmed = raw.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                // A single blank line between items keeps the list together.
                if index + 1 < lines.count, item(lines[index + 1]) != nil {
                    index += 1
                    continue
                }
                return
            }
            if let next = item(raw) {
                items.append(next)
                index += 1
                continue
            }
            if startsBlock(raw) { return }
            if !items.isEmpty {
                items[items.count - 1] += "\n" + trimmed
            }
            index += 1
        }
    }

    /// True when the line opens a block other than a paragraph.
    private static func startsBlock(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return false }
        if Fence(trimmed) != nil { return true }
        if headingBlock(trimmed) != nil { return true }
        if isDivider(trimmed) { return true }
        if trimmed.hasPrefix(">") { return true }
        if bulletItem(line) != nil { return true }
        if numberedItem(line) != nil { return true }
        return false
    }
}
