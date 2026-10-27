import SwiftUI
import LocalCodeKit

/// Full-text search across every message, grouped by conversation (⌘K).
///
/// The server returns snippets with literal `<mark>…</mark>` markers around the
/// matches. They are never rendered as markup: the string is split on the
/// markers and the pieces are styled in an `AttributedString`.
struct SearchView: View {
    /// Called with the hit the user picked, after the conversation is opened.
    var onSelect: ((SearchHit) -> Void)?

    @Environment(ChatStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var hits: [SearchHit] = []
    @State private var isLoading = false
    @State private var errorText: String?
    @State private var searchTask: Task<Void, Never>?
    @FocusState private var isFieldFocused: Bool

    private static let debounce: UInt64 = 250_000_000

    private var trimmedQuery: String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            field
            Hairline()
            results
            Hairline()
            footer
        }
        .background(Theme.bg)
        .frame(minWidth: 620, idealWidth: 720, minHeight: 460, idealHeight: 560)
        .onChange(of: query) { _, newValue in schedule(newValue) }
        .onDisappear { searchTask?.cancel() }
        .task {
            try? await Task.sleep(nanoseconds: 120_000_000)
            isFieldFocused = true
        }
    }

    // MARK: - Field

    private var field: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Theme.fgSubtle)

            TextField("Szukaj we wszystkich rozmowach…", text: $query)
                .textFieldStyle(.plain)
                .font(.system(size: 15))
                .foregroundStyle(Theme.fg)
                .focused($isFieldFocused)
                .autocorrectionDisabled(true)
                .onSubmit { openFirst() }

            if isLoading {
                ProgressView().controlSize(.small)
            }

            IconButton(systemName: "xmark", label: "Zamknij") { dismiss() }
        }
        .padding(.horizontal, 14)
        .frame(height: 52)
    }

    // MARK: - Results

    @ViewBuilder
    private var results: some View {
        ScrollView {
            if let errorText {
                Banner(.error, message: errorText)
                    .padding(16)
            } else if trimmedQuery.isEmpty {
                EmptyState(systemImage: "magnifyingglass",
                           title: "Zacznij pisać",
                           message: "Przeszukam treść wszystkich rozmów — pytania i odpowiedzi, także w projektach.")
                    .padding(.vertical, 26)
            } else if hits.isEmpty {
                if isLoading {
                    Text("Szukam…")
                        .font(.system(size: 13))
                        .foregroundStyle(Theme.fgSubtle)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                } else {
                    EmptyState(systemImage: "bubble.left.and.bubble.right",
                               title: "Brak wyników dla „\(trimmedQuery)”",
                               message: "Spróbuj innego słowa kluczowego lub krótszej frazy.")
                        .padding(.vertical, 26)
                }
            } else {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(groups) { group in
                        VStack(alignment: .leading, spacing: 4) {
                            groupHeader(group)
                            ForEach(group.hits) { hit in
                                hitRow(hit)
                            }
                        }
                    }
                }
                .padding(12)
            }
        }
        .frame(maxHeight: .infinity)
    }

    private func groupHeader(_ group: HitGroup) -> some View {
        HStack(spacing: 8) {
            Text(group.title)
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .lineLimit(1)
            Spacer(minLength: 0)
            if let folderName = group.folderName {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .font(.system(size: 9))
                    Text(folderName)
                        .font(.system(size: 11))
                        .lineLimit(1)
                }
                .foregroundStyle(Theme.fgSubtle)
            }
        }
        .padding(.horizontal, 6)
        .padding(.top, 4)
    }

    private func hitRow(_ hit: SearchHit) -> some View {
        Button {
            open(hit)
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Text(roleLabel(hit.role))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.fgSubtle)
                    .frame(width: 58, alignment: .leading)
                Text(SearchSnippet.highlighted(hit.snippet))
                    .font(.system(size: 13))
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text(Format.relative(hit.createdAt))
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                    .lineLimit(1)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
    }

    private var footer: some View {
        HStack(spacing: 14) {
            Text("Wpisz frazę, aby przeszukać wszystkie rozmowy.")
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
            Spacer(minLength: 0)
            if !hits.isEmpty {
                Text(Format.plural(hits.count, "wynik", "wyniki", "wyników"))
                    .font(.system(size: 11))
                    .monospacedDigit()
                    .foregroundStyle(Theme.fgSubtle)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }

    // MARK: - Grouping

    private struct HitGroup: Identifiable {
        let id: String
        let title: String
        let folderName: String?
        var hits: [SearchHit]
    }

    /// Keeps the server's ranking: the first hit of a conversation decides the
    /// position of its whole group.
    private var groups: [HitGroup] {
        var order: [String] = []
        var byConversation: [String: HitGroup] = [:]
        for hit in hits {
            if var existing = byConversation[hit.conversationId] {
                existing.hits.append(hit)
                byConversation[hit.conversationId] = existing
            } else {
                order.append(hit.conversationId)
                byConversation[hit.conversationId] = HitGroup(id: hit.conversationId,
                                                              title: hit.conversationTitle,
                                                              folderName: hit.folderName,
                                                              hits: [hit])
            }
        }
        return order.compactMap { byConversation[$0] }
    }

    private func roleLabel(_ role: MessageRole) -> String {
        switch role {
        case .user: "Ty"
        case .assistant: "Asystent"
        case .system: "System"
        }
    }

    // MARK: - Searching

    private func schedule(_ text: String) {
        searchTask?.cancel()
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            hits = []
            errorText = nil
            isLoading = false
            return
        }
        isLoading = true
        searchTask = Task {
            try? await Task.sleep(nanoseconds: Self.debounce)
            guard !Task.isCancelled else { return }
            do {
                let result = try await store.api.search(trimmed)
                guard !Task.isCancelled else { return }
                hits = result
                errorText = nil
            } catch {
                guard !Task.isCancelled else { return }
                hits = []
                errorText = APIError.describe(error)
            }
            isLoading = false
        }
    }

    private func openFirst() {
        guard let first = hits.first else { return }
        open(first)
    }

    private func open(_ hit: SearchHit) {
        let id = hit.conversationId
        Task { await store.openConversation(id) }
        onSelect?(hit)
        dismiss()
    }
}

// MARK: - Snippet highlighting

/// Turns the server's `<mark>`-annotated snippet into styled text. The markers
/// are treated as plain delimiters, never as markup.
enum SearchSnippet {
    private static let open = "<mark>"
    private static let close = "</mark>"

    static func highlighted(_ snippet: String) -> AttributedString {
        var result = AttributedString()
        var rest = Substring(snippet)

        while let openRange = rest.range(of: open) {
            result += plain(String(rest[rest.startIndex..<openRange.lowerBound]))
            let after = rest[openRange.upperBound...]
            guard let closeRange = after.range(of: close) else {
                result += plain(String(after))
                return result
            }
            result += marked(String(after[after.startIndex..<closeRange.lowerBound]))
            rest = after[closeRange.upperBound...]
        }

        result += plain(String(rest))
        return result
    }

    private static func plain(_ text: String) -> AttributedString {
        guard !text.isEmpty else { return AttributedString() }
        var piece = AttributedString(text)
        piece.foregroundColor = Theme.fgMuted
        return piece
    }

    private static func marked(_ text: String) -> AttributedString {
        guard !text.isEmpty else { return AttributedString() }
        var piece = AttributedString(text)
        piece.foregroundColor = Theme.accent
        piece.backgroundColor = Theme.accentSoft
        piece.font = .system(size: 13, weight: .semibold)
        return piece
    }
}
