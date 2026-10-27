//
//  The documentation window: the bundled user guide and API reference,
//  rendered natively, with a table of contents and a download button.
//
import SwiftUI
import AppKit

// MARK: - State

/// What the documentation window shows. Shared, so the window keeps its page
/// when it is closed and opened again from the Help menu or Settings.
@MainActor
@Observable
public final class DocumentationModel {
    public static let shared = DocumentationModel()

    /// A parsed document, kept for the life of the app.
    public struct Document {
        public let blocks: [DocumentationBlock]
        public let outline: [DocumentationHeading]
    }

    /// Where the open page should scroll next. `nonce` makes the same anchor
    /// requested twice scroll twice.
    public struct ScrollTarget: Equatable {
        public let page: DocumentationPage
        /// `nil` scrolls to the top.
        public let anchor: String?
        let nonce: Int
    }

    /// A short confirmation or error under the document.
    public struct Notice: Equatable, Identifiable {
        public let id = UUID()
        public let kind: BannerKind
        public let message: String

        public init(kind: BannerKind, message: String) {
            self.kind = kind
            self.message = message
        }
    }

    public private(set) var page: DocumentationPage = .guide
    public private(set) var target: ScrollTarget?
    public var notice: Notice?

    @ObservationIgnored private var cache: [DocumentationPage: Document] = [:]
    @ObservationIgnored private var nonce = 0

    public init() {}

    /// The parsed page, or nil when this build carries no documentation.
    public func document(for page: DocumentationPage) -> Document? {
        if let cached = cache[page] { return cached }
        guard let text = DocumentationLibrary.text(for: page) else { return nil }
        let blocks = DocumentationMarkdown.blocks(text)
        let document = Document(blocks: blocks, outline: DocumentationMarkdown.outline(blocks))
        cache[page] = document
        return document
    }

    /// Switches to `page` and scrolls to `anchor`, or to the top.
    public func show(_ page: DocumentationPage, anchor: String? = nil) {
        self.page = page
        nonce &+= 1
        target = ScrollTarget(page: page, anchor: anchor, nonce: nonce)
    }

    /// A click on a link inside a document.
    public func open(_ url: URL) -> OpenURLAction.Result {
        switch DocumentationMarkdown.link(url) {
        case let .external(external)?:
            return .systemAction(external)
        case let .anchor(anchor)?:
            show(page, anchor: anchor)
            return .handled
        case let .page(other, anchor)?:
            show(other, anchor: anchor)
            return .handled
        case let .missing(name)?:
            notice = Notice(kind: .info,
                            message: "„\(name)” nie jest dołączony do aplikacji — znajdziesz go w repozytorium LocalCode.")
            return .handled
        case nil:
            return .discarded
        }
    }

    /// "Pobierz…": both files into a folder the user picks.
    public func download() {
        switch DocumentationExport.chooseFolderAndSave() {
        case let .success(saved)?:
            notice = Notice(kind: .success, message: saved.message)
        case let .failure(failure)?:
            notice = Notice(kind: .error, message: failure.localizedDescription)
        case nil:
            break
        }
    }
}

// MARK: - Window content

/// The documentation window's content: table of contents on the left, the
/// document on the right, page switch and download in the toolbar.
public struct DocumentationView: View {
    @State private var model = DocumentationModel.shared

    public init() {}

    public var body: some View {
        let document = model.document(for: model.page)

        HStack(spacing: 0) {
            if let document {
                DocumentationOutline(headings: document.outline) { anchor in
                    model.show(model.page, anchor: anchor)
                }
                .id(model.page)
                Hairline(vertical: true)
                DocumentationPageView(page: model.page, document: document, target: model.target)
                    .id(model.page)
            } else {
                EmptyState(systemImage: "book.closed",
                           title: "Brak dokumentacji",
                           message: DocumentationLibrary.unavailableMessage)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .overlay(alignment: .bottom) {
            if let notice = model.notice {
                Banner(notice.kind, message: notice.message, onDismiss: { model.notice = nil })
                    // The banner's tint is translucent; over text it needs a floor.
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .frame(maxWidth: 560)
                    .shadow(color: .black.opacity(0.12), radius: 10, y: 3)
                    .padding(16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .animation(.easeOut(duration: 0.18), value: model.notice)
        .background(Theme.bg)
        .tint(Theme.accent)
        // Every link in every Text below lands here instead of in the browser.
        .environment(\.openURL, OpenURLAction { model.open($0) })
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("Dokument", selection: Binding(get: { model.page },
                                                      set: { model.show($0) })) {
                    ForEach(DocumentationPage.allCases) { page in
                        Text(page.title).tag(page)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 260)
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    model.download()
                } label: {
                    Label("Pobierz…", systemImage: "arrow.down.circle")
                        .labelStyle(.titleAndIcon)
                }
                .help("Zapisz oba pliki Markdown w wybranym folderze")
                .disabled(document == nil)
            }
        }
        .task(id: model.notice?.id) {
            guard model.notice != nil else { return }
            try? await Task.sleep(nanoseconds: 6_000_000_000)
            if !Task.isCancelled { model.notice = nil }
        }
    }
}

// MARK: - Table of contents

private struct DocumentationOutline: View {
    let headings: [DocumentationHeading]
    let onSelect: (String) -> Void

    @State private var hovered: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 1) {
                Text("Spis treści")
                    .font(.system(size: 11, weight: .semibold))
                    .textCase(.uppercase)
                    .kerning(0.6)
                    .foregroundStyle(Theme.fgSubtle)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)

                ForEach(headings) { heading in
                    Button {
                        onSelect(heading.anchor)
                    } label: {
                        Text(heading.title)
                            .font(.system(size: heading.level <= 2 ? 12.5 : 12,
                                          weight: heading.level <= 2 ? .medium : .regular))
                            .foregroundStyle(heading.level <= 2 ? Theme.fg : Theme.fgMuted)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                            .padding(.leading, heading.level <= 2 ? 10 : 22)
                            .padding(.trailing, 8)
                            .padding(.vertical, 5)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(hovered == heading.anchor ? Theme.surface2 : Color.clear)
                            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(PressableButtonStyle())
                    .onHover { inside in
                        if inside { hovered = heading.anchor } else if hovered == heading.anchor { hovered = nil }
                    }
                    .padding(.top, heading.level <= 2 ? 4 : 0)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 14)
        }
        .frame(width: 236)
        .background(Theme.sidebar)
    }
}

// MARK: - Document

private struct DocumentationPageView: View {
    let page: DocumentationPage
    let document: DocumentationModel.Document
    let target: DocumentationModel.ScrollTarget?

    private static let topID = "documentation-top"

    /// False on the page's first target: a page that has only just appeared
    /// (a link into the other document) is still measuring its text.
    @State private var hasAppeared = false

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Color.clear.frame(height: 0).id(Self.topID)
                    ForEach(document.blocks) { item in
                        DocumentationBlockView(item: item)
                    }
                }
                .frame(maxWidth: 860, alignment: .leading)
                .padding(.horizontal, 36)
                .padding(.top, 26)
                .padding(.bottom, 64)
                .frame(maxWidth: .infinity)
            }
            .background(Theme.bg)
            // Runs when the page appears (a link to the other document) and
            // whenever a new target arrives (an anchor link, the outline).
            .task(id: target) {
                let isFresh = !hasAppeared
                hasAppeared = true
                guard let target, target.page == page else { return }
                let id = target.anchor ?? Self.topID
                if !isFresh {
                    withAnimation(.easeOut(duration: 0.25)) {
                        proxy.scrollTo(id, anchor: .top)
                    }
                    return
                }
                // A fresh page settles its layout over a few passes, and a
                // jump made too early lands short of the heading; repeating it
                // while the page appears costs nothing visible.
                for delay: UInt64 in [50_000_000, 200_000_000, 450_000_000] {
                    try? await Task.sleep(nanoseconds: delay)
                    guard !Task.isCancelled else { return }
                    proxy.scrollTo(id, anchor: .top)
                }
            }
        }
    }
}

private struct DocumentationBlockView: View {
    let item: DocumentationBlock

    private static let baseSize: CGFloat = 14

    var body: some View {
        switch item.block {
        case let .heading(level, text):
            if let anchor = item.anchor {
                heading(level: level, text: text).id(anchor)
            } else {
                heading(level: level, text: text)
            }
        case let .code(language, content):
            CodeBlockView(language: language, content: content, fontSize: 12.5)
        case let .table(header, rows):
            DocumentationTable(header: header, rows: rows)
        case .divider:
            Hairline().padding(.vertical, 8)
        default:
            MarkdownView(blocks: [item.block], baseSize: Self.baseSize, spacing: 0)
        }
    }

    private func heading(level: Int, text: String) -> some View {
        let size: CGFloat = switch level {
        case 1: 27
        case 2: 21
        case 3: 16.5
        default: 14.5
        }
        return VStack(alignment: .leading, spacing: 10) {
            Text(Self.headingText(text, size: size))
                .font(.system(size: size, weight: level == 1 ? .bold : .semibold))
                .foregroundStyle(level >= 4 ? Theme.fgMuted : Theme.fg)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            if level == 1 { Hairline() }
        }
        .padding(.top, level == 2 ? 10 : (level == 3 ? 6 : 0))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// `MarkdownParser.inline` sizes code spans for body text; in a heading
    /// (`### POST /v1/chat/completions`) they follow the heading instead.
    private static func headingText(_ text: String, size: CGFloat) -> AttributedString {
        var attributed = MarkdownParser.inline(text)
        let codeRanges = attributed.runs
            .filter { $0.inlinePresentationIntent?.contains(.code) == true }
            .map(\.range)
        for range in codeRanges {
            attributed[range].font = .system(size: size * 0.9, weight: .semibold, design: .monospaced)
        }
        return attributed
    }
}

// MARK: - Table

/// A table that wraps its long cells instead of scrolling sideways: the API
/// reference has descriptions a few hundred characters long next to one-word
/// methods. Columns with short content keep their natural width and the rest
/// share what is left, the way an `HStack` shares space between texts.
private struct DocumentationTable: View {
    let header: [String]
    let rows: [[String]]

    private var columnCount: Int { max(header.count, rows.map(\.count).max() ?? 0) }

    var body: some View {
        Grid(alignment: .topLeading, horizontalSpacing: 0, verticalSpacing: 0) {
            GridRow {
                ForEach(0..<columnCount, id: \.self) { column in
                    cell(value(header, column), isHeader: true)
                }
            }
            Rectangle()
                .fill(Theme.borderStrong)
                .frame(height: 1)
                .gridCellUnsizedAxes(.horizontal)

            ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                if index > 0 {
                    Rectangle()
                        .fill(Theme.border)
                        .frame(height: 1)
                        .gridCellUnsizedAxes(.horizontal)
                }
                GridRow {
                    ForEach(0..<columnCount, id: \.self) { column in
                        cell(value(row, column), isHeader: false)
                    }
                }
            }
        }
        .padding(.vertical, 2)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func value(_ cells: [String], _ index: Int) -> String {
        index < cells.count ? cells[index] : ""
    }

    private func cell(_ text: String, isHeader: Bool) -> some View {
        Text(MarkdownParser.inline(text))
            .font(.system(size: 13, weight: isHeader ? .semibold : .regular))
            .foregroundStyle(Theme.fg)
            .lineSpacing(3)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
    }
}
