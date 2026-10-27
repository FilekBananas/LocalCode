import SwiftUI
import LocalCodeKit

/// How much of the model's window the next prompt will take.
///
/// This is the one number that decides whether a local model can still do the
/// job, so it sits in the header of every session: a bar, the tokens, the
/// percentage, and a tooltip that breaks the number down into the four parts
/// the Python context manager reports (system, tools, history, reserve).
struct ContextGauge: View {
    let stats: ContextStats

    private var tint: Color {
        switch stats.fraction {
        case ..<0.6: Theme.success
        case ..<0.85: Theme.warning
        default: Theme.danger
        }
    }

    private var tooltip: String {
        guard !stats.isEmpty else {
            return "Kontekst policzy się po pierwszej wiadomości."
        }
        return """
        Kontekst: \(decimal(stats.used)) / \(decimal(stats.limit)) tokenów (\(stats.percent)%)
        Prompt systemowy: \(decimal(stats.system))
        Narzędzia: \(decimal(stats.tools))
        Historia rozmowy: \(decimal(stats.history))
        Rezerwa na odpowiedź: \(decimal(stats.reserve))
        """
    }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "gauge.with.dots.needle.33percent")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.fgSubtle)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 4) {
                    Text(stats.isEmpty ? "—" : "\(TokenFormat.compact(stats.used)) / \(TokenFormat.compact(stats.limit))")
                        .font(.system(size: 11.5, weight: .medium, design: .rounded))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                    if !stats.isEmpty {
                        Text("\(stats.percent)%")
                            .font(.system(size: 11))
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
                        .frame(width: max(stats.isEmpty ? 0 : 3, 96 * stats.fraction), height: 4)
                        .animation(.easeOut(duration: 0.25), value: stats.fraction)
                }
                .frame(width: 96)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .help(tooltip)
        .accessibilityLabel(Text("Wykorzystanie kontekstu"))
        .accessibilityValue(Text(stats.isEmpty ? "brak danych" : "\(stats.percent) procent"))
    }

    private func decimal(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "pl_PL")
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }
}
