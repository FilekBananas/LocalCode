import SwiftUI
import LocalCodeKit

/// How much of the model's window the next turn will take, in the chat header.
///
/// The agent shows the same number in the same shape (`CodeFeature`'s
/// `ContextGauge`); the chat needs its own copy because the two feature modules
/// sit beside each other and neither may import the other. The numbers come
/// from the `context` frame of `/api/chat`, which reports the same four parts
/// the Python context manager splits the window into.
struct ChatContextGauge: View {
    let usage: ChatContextUsage

    private var tint: Color {
        switch usage.fraction {
        case ..<0.6: Theme.success
        case ..<0.85: Theme.warning
        default: Theme.danger
        }
    }

    private var tooltip: String {
        guard !usage.isEmpty else {
            return "Kontekst policzy się po pierwszej wiadomości."
        }
        var lines = [
            "Kontekst: \(ChatTokenFormat.decimal(usage.used)) / \(ChatTokenFormat.decimal(usage.limit)) tokenów (\(usage.percent)%)",
            "Prompt systemowy: \(ChatTokenFormat.decimal(usage.system))",
            "Narzędzia: \(ChatTokenFormat.decimal(usage.tools))",
            "Historia rozmowy: \(ChatTokenFormat.decimal(usage.history))",
            "Rezerwa na odpowiedź: \(ChatTokenFormat.decimal(usage.reserve))",
        ]
        if usage.summary > 0 {
            lines.append("Podsumowanie starszych tur: \(ChatTokenFormat.decimal(usage.summary))")
        }
        return lines.joined(separator: "\n")
    }

    var body: some View {
        HStack(spacing: 7) {
            Image(systemName: "gauge.with.dots.needle.33percent")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.fgSubtle)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text(usage.isEmpty
                         ? "—"
                         : "\(ChatTokenFormat.compact(usage.used)) / \(ChatTokenFormat.compact(usage.limit))")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                    if !usage.isEmpty {
                        Text("\(usage.percent)%")
                            .font(.system(size: 10.5))
                            .foregroundStyle(tint)
                            .monospacedDigit()
                    }
                }
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Theme.surface2)
                        .frame(height: 4)
                    Capsule()
                        .fill(tint)
                        .frame(width: max(usage.isEmpty ? 0 : 3, 84 * usage.fraction), height: 4)
                        .animation(.easeOut(duration: 0.25), value: usage.fraction)
                }
                .frame(width: 84)
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 4)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .help(tooltip)
        .accessibilityLabel(Text("Wykorzystanie kontekstu"))
        .accessibilityValue(Text(usage.isEmpty ? "brak danych" : "\(usage.percent) procent"))
    }
}

/// The inline notice the transcript shows where a compaction happened.
///
/// It is not a message — nothing the model said was lost, the *history* was
/// shortened — so it renders as a quiet rule across the transcript rather than
/// as another bubble.
struct CompactionNotice: View {
    let compaction: ChatCompaction

    var body: some View {
        HStack(spacing: 8) {
            line
            HStack(spacing: 6) {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                VStack(alignment: .leading, spacing: 1) {
                    Text(compaction.headline)
                        .font(.system(size: 11.5, weight: .medium))
                        .foregroundStyle(Theme.fgMuted)
                    if !compaction.strategyLabel.isEmpty {
                        Text(compaction.strategyLabel)
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Theme.accentSoft)
            .clipShape(Capsule())
            .overlay(Capsule().strokeBorder(Theme.accent.opacity(0.25), lineWidth: 1))
            // The rules either side are the flexible part; the pill keeps its
            // sentence on one line instead of breaking it after the arrow.
            .fixedSize(horizontal: true, vertical: false)
            .layoutPriority(1)
            line
        }
        .frame(maxWidth: .infinity)
        .help("""
        Rozmowa przekroczyła okno kontekstu modelu, więc starsze tury zostały \
        zastąpione podsumowaniem. Zaoszczędzono \
        \(ChatTokenFormat.compact(compaction.saved)) tokenów.
        """)
    }

    private var line: some View {
        Rectangle()
            .fill(Theme.border)
            .frame(height: 1)
            .frame(maxWidth: .infinity)
    }
}

// MARK: - Hosting

/// The context gauge and the compaction notice as standalone views.
///
/// `ChatShellView` and `ChatView` place them themselves, so nothing in the app
/// needs this — it exists so both can be hosted anywhere they make sense and so
/// the render harness that reviews this module can put each one in a window and
/// look at it, in light and in dark, without driving a whole conversation.
public enum ChatScreens {

    /// How much of the model's window the next turn will take.
    public static func contextGauge(_ usage: ChatContextUsage) -> some View {
        ChatContextGauge(usage: usage)
    }

    /// The inline "Kontekst skompaktowany: … → …" rule.
    public static func compactionNotice(_ compaction: ChatCompaction) -> some View {
        CompactionNotice(compaction: compaction)
    }
}
