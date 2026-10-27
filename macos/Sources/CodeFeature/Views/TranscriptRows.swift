import SwiftUI
import AppKit
import LocalCodeKit

/// The user's own message.
struct UserMessageRow: View {
    let text: String
    var images: [Data] = []

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "person.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Theme.fgMuted)
                .frame(width: 20, height: 20)
                .background(Theme.surface2)
                .clipShape(Circle())
            VStack(alignment: .leading, spacing: 8) {
                if !images.isEmpty {
                    HStack(spacing: 6) {
                        ForEach(Array(images.enumerated()), id: \.offset) { _, data in
                            if let image = NSImage(data: data) {
                                Image(nsImage: image)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 72, height: 72)
                                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                        .strokeBorder(Theme.border, lineWidth: 1))
                            }
                        }
                    }
                }
                if !text.isEmpty {
                    Text(text)
                        .font(.system(size: 14))
                        .foregroundStyle(Theme.fg)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 9)
        .padding(.horizontal, 12)
        .background(Theme.bubbleUser)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

/// The streamed answer, rendered as markdown.
struct AssistantMessageRow: View {
    let text: String
    let isStreaming: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            MarkdownView(text, baseSize: 14, spacing: 10)
                .textSelection(.enabled)
            if isStreaming {
                TypingDots()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.leading, 2)
    }
}

/// „Myślenie…” — collapsed by default, because it is long and rarely the point.
struct ReasoningRow: View {
    let text: String
    let isStreaming: Bool

    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "brain")
                        .font(.system(size: 10, weight: .semibold))
                    Text(isStreaming ? "Myślenie…" : "Myślenie")
                        .font(.system(size: 12, weight: .medium))
                    Text("(\(Format.plural(lineCount, "linia", "linie", "linii")))")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                    Spacer(minLength: 0)
                }
                .foregroundStyle(Theme.fgMuted)
                .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle())

            if isExpanded {
                Text(text)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Theme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
        .padding(.vertical, 2)
    }

    private var lineCount: Int {
        max(1, text.split(separator: "\n", omittingEmptySubsequences: false).count)
    }
}

/// Server notices: mode switches, denied operations, cleared conversations.
struct NoticeRow: View {
    let text: String
    var systemImage: String = "info.circle"
    var tint: Color = Theme.fgSubtle

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: systemImage)
                .font(.system(size: 10, weight: .semibold))
            Text(text)
                .font(.system(size: 11.5))
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .foregroundStyle(tint)
        .padding(.vertical, 3)
        .padding(.horizontal, 2)
    }
}

/// „Kontekst skompaktowany: 12,4k → 5,1k tokenów”.
struct CompactionRow: View {
    let info: CompactionInfo

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "arrow.down.right.and.arrow.up.left")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Theme.warning)
            Text(info.message)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(Theme.fg)
            Text("· \(info.strategyLabel)")
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgSubtle)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Theme.warningSoft)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
    }
}

/// The line that closes a turn: steps, tokens per second, duration.
struct TurnStatsRow: View {
    let stats: TurnStats?
    let reason: String?

    private var parts: [String] {
        var out: [String] = []
        if let steps = stats?.steps { out.append(Format.plural(steps, "krok", "kroki", "kroków")) }
        if let tps = stats?.genTps, tps > 0 {
            out.append(String(format: "%.1f tok/s", tps).replacingOccurrences(of: ".", with: ","))
        }
        if let tokens = stats?.completionTokens, tokens > 0 {
            out.append("\(TokenFormat.compact(tokens)) tokenów odpowiedzi")
        }
        if let ms = stats?.elapsedMs ?? stats?.totalMs, ms > 0 {
            out.append(Format.duration(ms / 1000))
        }
        return out
    }

    private var reasonLabel: String? {
        switch reason {
        case "interrupted": "przerwano"
        case "max_steps": "przerwano po limicie kroków"
        case "error": "zakończono błędem"
        default: nil
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: reason == "done" ? "checkmark.circle" : "exclamationmark.circle")
                .font(.system(size: 10))
            if let reasonLabel {
                Text(reasonLabel)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.warning)
            }
            Text(parts.joined(separator: " · "))
                .font(.system(size: 11))
                .monospacedDigit()
            Spacer(minLength: 0)
        }
        .foregroundStyle(Theme.fgSubtle)
        .padding(.top, 2)
        .padding(.bottom, 6)
    }
}

/// An error the agent reported mid-turn.
struct TranscriptErrorRow: View {
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.danger)
            Text(text)
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fg)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(Theme.dangerSoft)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

/// „Ładowanie modelu…” while the turn has not produced anything yet.
struct StatusRow: View {
    let message: String

    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 12, height: 12)
            Text(message)
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }
}

/// Three dots that breathe while the answer is still streaming.
struct TypingDots: View {
    @State private var phase = 0.0

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Theme.fgSubtle)
                    .frame(width: 4, height: 4)
                    .opacity(opacity(for: index))
            }
        }
        .onAppear {
            withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) { phase = 3 }
        }
        .accessibilityHidden(true)
    }

    private func opacity(for index: Int) -> Double {
        let distance = abs(phase - Double(index))
        return 0.3 + 0.7 * max(0, 1 - min(distance, 1))
    }
}
