//
//  Renders parsed markdown with the typography of the web chat app.
//
import SwiftUI

/// Renders parsed markdown with the same typography as the web app.
public struct MarkdownView: View {
    public let blocks: [MarkdownBlock]
    public var baseSize: CGFloat = 15
    public var spacing: CGFloat = 12

    public init(_ text: String, baseSize: CGFloat = 15, spacing: CGFloat = 12) {
        self.blocks = MarkdownParser.parse(text)
        self.baseSize = baseSize
        self.spacing = spacing
    }

    public init(blocks: [MarkdownBlock], baseSize: CGFloat = 15, spacing: CGFloat = 12) {
        self.blocks = blocks
        self.baseSize = baseSize
        self.spacing = spacing
    }

    /// ~1.5 line height, like the site.
    private var lineSpacing: CGFloat { (baseSize * 0.5).rounded() }

    public var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .tint(Theme.accent)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case let .heading(level, text):
            Text(MarkdownParser.inline(text))
                .font(.system(size: headingSize(level), weight: .semibold))
                .foregroundStyle(level >= 5 ? Theme.fgMuted : Theme.fg)
                .lineSpacing(3)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, level <= 2 ? 4 : 0)
                .frame(maxWidth: .infinity, alignment: .leading)

        case let .paragraph(text):
            paragraphText(text)

        case let .bulletList(items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("•")
                            .font(.system(size: baseSize, weight: .semibold))
                            .foregroundStyle(Theme.fgMuted)
                            .frame(width: 10, alignment: .leading)
                        paragraphText(item)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case let .numberedList(items):
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\(index + 1).")
                            .font(.system(size: baseSize, weight: .medium))
                            .monospacedDigit()
                            .foregroundStyle(Theme.fgMuted)
                            .frame(minWidth: 18, alignment: .trailing)
                        paragraphText(item)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case let .code(language, content):
            CodeBlockView(language: language, content: content, fontSize: baseSize - 2)

        case let .quote(text):
            HStack(alignment: .top, spacing: 0) {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(Theme.accent.opacity(0.55))
                    .frame(width: 3)
                paragraphText(text, color: Theme.fgMuted, italic: true)
                    .padding(.leading, 12)
                    .padding(.vertical, 2)
            }
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)

        case let .table(header, rows):
            TableBlockView(header: header, rows: rows, baseSize: baseSize)

        case .divider:
            Rectangle()
                .fill(Theme.border)
                .frame(height: 1)
                .padding(.vertical, 2)
        }
    }

    private func paragraphText(_ text: String,
                               color: Color = Theme.fg,
                               italic: Bool = false) -> some View {
        Text(MarkdownParser.inline(text))
            .font(.system(size: baseSize).italic(italic))
            .foregroundStyle(color)
            .lineSpacing(lineSpacing)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: baseSize + 8
        case 2: baseSize + 5
        case 3: baseSize + 3
        case 4: baseSize + 1
        default: baseSize
        }
    }
}

private extension Font {
    func italic(_ isItalic: Bool) -> Font { isItalic ? self.italic() : self }
}

// MARK: - Table

/// GFM table with a bordered grid that scrolls horizontally when it is wider
/// than the message column.
private struct TableBlockView: View {
    let header: [String]
    let rows: [[String]]
    let baseSize: CGFloat

    private var columnCount: Int { max(header.count, rows.map(\.count).max() ?? 0) }

    var body: some View {
        ScrollView(.horizontal, showsIndicators: true) {
            Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(0..<columnCount, id: \.self) { column in
                        cell(text: value(header, column), isHeader: true, isLast: column == columnCount - 1)
                    }
                }
                .background(Theme.surface2)

                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    Divider().overlay(Theme.border).gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(0..<columnCount, id: \.self) { column in
                            cell(text: value(row, column), isHeader: false, isLast: column == columnCount - 1)
                        }
                    }
                }
            }
            // The border hugs the table instead of the scroll viewport.
            .fixedSize(horizontal: true, vertical: true)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
            .padding(1)
        }
    }

    private func value(_ cells: [String], _ index: Int) -> String {
        index < cells.count ? cells[index] : ""
    }

    private func cell(text: String, isHeader: Bool, isLast: Bool) -> some View {
        Text(MarkdownParser.inline(text))
            .font(.system(size: baseSize - 1, weight: isHeader ? .semibold : .regular))
            .foregroundStyle(isHeader ? Theme.fg : Theme.fgMuted)
            .textSelection(.enabled)
            .lineLimit(nil)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .frame(minWidth: 56, alignment: .leading)
            .overlay(alignment: .trailing) {
                if !isLast {
                    Rectangle().fill(Theme.border).frame(width: 1)
                }
            }
    }
}
