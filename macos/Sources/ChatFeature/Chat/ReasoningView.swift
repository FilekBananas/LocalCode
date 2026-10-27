import SwiftUI
import LocalCodeKit

/// Collapsible panel holding one assistant turn's chain of thought.
///
/// It is open while the tokens are still arriving and folds itself away once
/// the turn finishes, unless the reader has taken over with the chevron — then
/// their choice wins for the rest of the session.
struct ReasoningView: View {
    let text: String
    var isStreaming: Bool = false

    /// `nil` means "follow the stream"; once the reader taps, it sticks.
    @State private var manualExpansion: Bool?

    private var isExpanded: Bool { manualExpansion ?? isStreaming }

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if isExpanded, !trimmed.isEmpty {
                Hairline()
                reasoningBody(trimmed)
            }
        }
        .background(Theme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .animation(.easeOut(duration: 0.18), value: isExpanded)
        .accessibilityElement(children: .contain)
    }

    // MARK: - Header

    private var header: some View {
        Button {
            manualExpansion = !isExpanded
        } label: {
            HStack(spacing: 7) {
                Image(systemName: "brain")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.accent)

                Text(isStreaming ? "Myślę…" : "Rozumowanie")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.fgMuted)

                if isStreaming {
                    ProgressView()
                        .controlSize(.small)
                        .tint(Theme.fgSubtle)
                        .scaleEffect(0.7)
                        .frame(width: 12, height: 12)
                }

                Spacer(minLength: 8)

                if !isExpanded, !trimmed.isEmpty {
                    Text(summary)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }

                Image(systemName: "chevron.down")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.fgSubtle)
                    .rotationEffect(.degrees(isExpanded ? 0 : -90))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle(pressedOpacity: 0.6))
        .accessibilityLabel(Text(isExpanded ? "Zwiń rozumowanie" : "Rozwiń rozumowanie"))
    }

    /// First line of the thought, used as the collapsed teaser.
    private var summary: String {
        let firstLine = trimmed
            .split(separator: "\n", omittingEmptySubsequences: true)
            .first
            .map(String.init) ?? trimmed
        return firstLine.count > 90 ? String(firstLine.prefix(90)) + "…" : firstLine
    }

    // MARK: - Body

    private func reasoningBody(_ content: String) -> some View {
        Text(content)
            .font(.system(size: 13))
            .foregroundStyle(Theme.fgMuted)
            .lineSpacing(4)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
    }
}
