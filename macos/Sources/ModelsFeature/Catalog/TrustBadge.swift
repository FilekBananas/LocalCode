import SwiftUI
import LocalCodeKit

/// How much is known about a model file, drawn as one capsule.
///
/// The store lists thousands of models nobody on this project has run, so every
/// card has to answer two questions before the download button means anything:
/// *who made this file* and *what do we actually know about it*. The number is
/// computed server-side by `localcode.hub.trust` from the publisher, the file
/// formats inside the repository, the licence, how many people use it and when
/// it was last touched — never from anything the model itself claims.
///
/// It is not a malware verdict, and the wording never suggests one: the tooltip
/// lists the sentences behind the number, and `TrustExplainer` shows all of them
/// in the details sheet.
struct TrustBadge: View {
    let trust: CatalogTrust
    let publisher: CatalogPublisher
    /// Compact cards show the number alone; the details sheet spells it out.
    var showsLabel: Bool = false

    var body: some View {
        if let score = trust.score {
            HStack(spacing: 4) {
                Image(systemName: publisher.systemImage)
                    .font(.system(size: 9.5, weight: .semibold))
                Text("\(score)")
                    .font(.system(size: 11, weight: .semibold))
                    .monospacedDigit()
                if showsLabel, !trust.label.isEmpty {
                    Text(trust.label)
                        .font(.system(size: 11, weight: .medium))
                }
            }
            .foregroundStyle(TrustPalette.tint(trust.level))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .background(TrustPalette.tint(trust.level).opacity(0.14))
            .clipShape(Capsule())
            .help(tooltip)
        }
    }

    private var tooltip: String {
        var lines: [String] = []
        if let score = trust.score {
            lines.append("Zaufanie \(score)/100 — \(trust.label)")
        }
        if !publisher.detail.isEmpty { lines.append(publisher.detail) }
        lines.append(contentsOf: trust.warnings.prefix(2))
        lines.append(contentsOf: trust.reasons.prefix(2))
        return lines.joined(separator: "\n")
    }
}

/// Colours shared by the badge and the explainer.
enum TrustPalette {
    static func tint(_ level: String) -> Color {
        switch level {
        case "high": Theme.success
        case "good": Theme.color(0x2F855A, 0x68D391)
        case "medium": Theme.warning
        case "low": Theme.danger
        case "rejected": Theme.danger
        default: Theme.fgMuted
        }
    }
}

/// The full answer to "why this number?", for the details sheet.
struct TrustExplainer: View {
    let trust: CatalogTrust
    let publisher: CatalogPublisher

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                TrustBadge(trust: trust, publisher: publisher, showsLabel: true)
                if !publisher.label.isEmpty {
                    MetaBadge(text: publisher.label)
                }
                Spacer(minLength: 0)
            }
            if !publisher.detail.isEmpty {
                Text(publisher.detail)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ForEach(trust.warnings, id: \.self) { warning in
                line(warning, symbol: "exclamationmark.triangle.fill", tint: Theme.warning)
            }
            ForEach(trust.reasons, id: \.self) { reason in
                line(reason, symbol: "checkmark", tint: Theme.fgSubtle)
            }
            Text("Ocena 0-100 liczona z pochodzenia pliku, licencji, formatu wag i popularności. "
                 + "To nie jest skan antywirusowy.")
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)
        }
    }

    private func line(_ text: String, symbol: String, tint: Color) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 12, alignment: .center)
                .padding(.top, 2)
            Text(text)
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }
}
