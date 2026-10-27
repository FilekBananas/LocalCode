import SwiftUI
import LocalCodeKit

/// The agent is asking for permission: the change it wants to make, and the
/// three answers the protocol accepts.
///
/// It is rendered inline at the end of the transcript, where the reader is
/// already looking, rather than in a modal sheet that would hide the diff it is
/// asking about. Return allows, Escape refuses, ⌥Return remembers the answer
/// for the rest of the session.
struct PermissionCardView: View {
    let request: PermissionRequestDTO
    let onDecision: (String) -> Void

    private var presentation: ToolPresentation { ToolPresentation.of(request.tool) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "hand.raised.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                Text("Zgoda wymagana")
                    .font(.system(size: 12, weight: .semibold))
                    .textCase(.uppercase)
                    .kerning(0.5)
                    .foregroundStyle(Theme.accent)
                if let helper = request.helper, !helper.isEmpty {
                    // Who asks: a helper agent, not the agent itself.
                    HStack(spacing: 4) {
                        Image(systemName: "person.2")
                            .font(.system(size: 10))
                        Text("Pomocnik \(helper)")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(Theme.fgMuted)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(Theme.fgSubtle.opacity(0.12))
                    .clipShape(Capsule())
                    .help("To polecenie chce wykonać pomocnik \(helper), nie sam agent.")
                }
                Spacer(minLength: 0)
                HStack(spacing: 5) {
                    Image(systemName: Format.symbol(presentation.systemImage, fallback: "wrench.and.screwdriver"))
                        .font(.system(size: 10))
                    Text(presentation.label)
                        .font(.system(size: 11))
                }
                .foregroundStyle(Theme.fgSubtle)
            }

            Text(request.title)
                .font(.system(size: 13.5, weight: .medium))
                .foregroundStyle(Theme.fg)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            if let review = request.review {
                reviewBlock(review)
            }

            if let does = request.previewParts.does {
                // The agent's own words on what the command does — plain text,
                // above the command itself.
                Text(does)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fg)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if !request.previewParts.body.isEmpty {
                if request.previewIsDiff {
                    DiffView(diff: request.preview, lineLimit: 120)
                } else {
                    Text(request.previewParts.body)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(9)
                        .background(Theme.codeBg)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1)
                        )
                }
            }

            // Never truncated: this is the only way to approve a delete or a
            // download, and in the help bot's narrow panel the buttons used to
            // shrink to "Z…" and "O…". The shortcut hint goes first, then the
            // row turns into a column.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    buttons
                    Spacer(minLength: 8)
                    shortcutHint
                }
                HStack(spacing: 8) {
                    buttons
                    Spacer(minLength: 0)
                }
                VStack(alignment: .leading, spacing: 6) {
                    buttons
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.accentSoft)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Theme.accent.opacity(0.45), lineWidth: 1.5)
        )
    }

    /// Auto mode: what the reviewer — and the local guard, for a command —
    /// said before the question came to the user. Normal text, not code: it
    /// starts with "Co robi: …", for someone who does not read shell.
    private func reviewBlock(_ review: AutoReviewInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            opinionRow(symbol: review.isAllowed ? "checkmark.shield.fill"
                       : (review.isDenied ? "xmark.shield.fill" : "questionmark.circle.fill"),
                       tint: review.isAllowed ? Theme.success : (review.isDenied ? Theme.danger : Theme.warning),
                       label: review.requestLabel,
                       reason: review.reason,
                       help: review.model.isEmpty ? "Recenzent trybu Auto" : "Recenzent trybu Auto: \(review.model)")
            if let opinion = review.guardOpinion {
                opinionRow(symbol: "lock.shield.fill",
                           tint: opinion.isAllowed ? Theme.success : (opinion.isDenied ? Theme.danger : Theme.warning),
                           label: opinion.isAllowed ? "Strażnik lokalny: TAK"
                                : (opinion.isDenied ? "Strażnik lokalny: NIE" : "Strażnik lokalny nie zdecydował"),
                           reason: opinion.reason,
                           help: "Lokalny model bez dostępu do rozmowy"
                                + (opinion.model.isEmpty ? "" : ": \(opinion.model)"))
            }
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private func opinionRow(symbol: String, tint: Color, label: String, reason: String,
                            help: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: symbol)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(tint)
                Text(label)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(tint)
            }
            let detail = reason.trimmingCharacters(in: .whitespacesAndNewlines)
            if !detail.isEmpty {
                Text(detail)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fg)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .help(help)
    }

    @ViewBuilder
    private var buttons: some View {
        PrimaryButton(title: "Zezwól", systemImage: "return", fullWidth: false, height: 32) {
            onDecision("allow")
        }
        .keyboardShortcut(.return, modifiers: [])
        .fixedSize()

        if request.allowsAlways {
            SecondaryButton(title: "Zawsze", systemImage: "infinity", height: 32) {
                onDecision("always")
            }
            .keyboardShortcut(.return, modifiers: .option)
            .help("Nie pytaj o to ponownie w tej sesji (⌥⏎)")
            .fixedSize()
        }

        DangerButton(title: "Odmów", systemImage: "xmark", filled: false, height: 32) {
            onDecision("deny")
        }
        .keyboardShortcut(.escape, modifiers: [])
        .fixedSize()
    }

    private var shortcutHint: some View {
        Text(request.allowsAlways ? "⏎ zezwól · ⌥⏎ zawsze · Esc odmów" : "⏎ zezwól · Esc odmów")
            .font(.system(size: 10.5))
            .foregroundStyle(Theme.fgSubtle)
            .lineLimit(1)
            .fixedSize()
    }
}
