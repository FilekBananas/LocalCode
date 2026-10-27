import SwiftUI
import LocalCodeKit

/// One tool call: what the agent did, whether it worked, and — on demand —
/// exactly what came back.
///
/// The card is deliberately quiet: an icon, the Polish title the server already
/// produced, a one-line summary and a state dot. The full output is one click
/// away and stays out of the way of reading the conversation. While a command
/// is running its live output is shown, which is what makes a `pytest` feel
/// alive instead of frozen.
struct ToolCardView: View {
    let tool: ToolState

    @State private var manualExpansion: Bool?
    @State private var copied = false

    private var presentation: ToolPresentation { ToolPresentation.of(tool.name) }

    /// A finished card is one line — except when it changed a file, because the
    /// diff is the whole point, or while a command is still printing.
    private var isExpanded: Bool {
        if let manualExpansion { return manualExpansion }
        if tool.isRunning { return !tool.liveOutput.isEmpty || !tool.helperSteps.isEmpty }
        if tool.helperBackground && tool.helperRunning { return !tool.helperSteps.isEmpty }
        return !(tool.diff ?? "").isEmpty || !tool.helperSteps.isEmpty
    }

    private var hasBody: Bool {
        !tool.bodyText.isEmpty || !(tool.diff ?? "").isEmpty || !tool.argumentSummary.isEmpty
            || !tool.helperSteps.isEmpty || tool.helperReport != nil
    }

    private var stateTint: Color {
        guard let ok = tool.ok else { return Theme.fgSubtle }
        return ok ? Theme.success : Theme.danger
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            if let review = tool.review {
                reviewLine(review)
                if let opinion = review.guardOpinion {
                    guardLine(opinion)
                }
            }
            if isExpanded {
                Hairline()
                expandedBody(for: tool)
            }
        }
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(tool.ok == false ? Theme.danger.opacity(0.4) : Theme.border, lineWidth: 1)
        )
    }

    // MARK: - Header

    private var header: some View {
        Button {
            guard hasBody else { return }
            withAnimation(.easeOut(duration: 0.15)) { manualExpansion = !isExpanded }
        } label: {
            HStack(alignment: .center, spacing: 9) {
                Image(systemName: Format.symbol(presentation.systemImage, fallback: "wrench.and.screwdriver"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(stateTint)
                    .frame(width: 20, height: 20)
                    .background(stateTint.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

                VStack(alignment: .leading, spacing: 1) {
                    Text(tool.displayTitle)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let summary = summaryLine {
                        Text(summary)
                            .font(.system(size: 11))
                            .foregroundStyle(tool.ok == false ? Theme.danger : Theme.fgSubtle)
                            .lineLimit(1)
                    }
                }

                Spacer(minLength: 6)

                if let counts = diffCounts {
                    HStack(spacing: 5) {
                        Text("+\(counts.added)")
                            .foregroundStyle(Theme.success)
                        Text("−\(counts.removed)")
                            .foregroundStyle(Theme.danger)
                    }
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                }

                stateIndicator

                if hasBody {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(Theme.fgSubtle)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .help(tool.argumentSummary.isEmpty ? tool.displayTitle : tool.argumentSummary)
    }

    /// Auto mode's verdict, always visible: a green check or a red x and the
    /// reviewer's reason (a question mark when it could not decide and the
    /// user was asked instead).
    private func reviewLine(_ review: AutoReviewInfo) -> some View {
        let tint = review.isAllowed ? Theme.success : (review.isDenied ? Theme.danger : Theme.warning)
        let symbol = review.isAllowed ? "checkmark.circle.fill"
            : (review.isDenied ? "xmark.circle.fill" : "questionmark.circle.fill")
        return HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(tint)
            Text(review.line)
                .font(.system(size: 11.5))
                .foregroundStyle(review.isDenied ? Theme.danger : Theme.fgMuted)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 10)
        .padding(.bottom, 8)
        .help(review.model.isEmpty ? "Recenzent trybu Auto" : "Recenzent trybu Auto: \(review.model)")
    }

    /// The local guard's opinion under the reviewer's (commands only).
    private func guardLine(_ opinion: GuardOpinion) -> some View {
        let tint = opinion.isAllowed ? Theme.success : (opinion.isDenied ? Theme.danger : Theme.warning)
        return HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 10.5, weight: .semibold))
                .foregroundStyle(tint)
            Text(opinion.line)
                .font(.system(size: 11.5))
                .foregroundStyle(opinion.isDenied ? Theme.danger : Theme.fgMuted)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 10)
        .padding(.bottom, 8)
        .help(opinion.model.isEmpty ? "Strażnik lokalny (bez kontekstu rozmowy)"
              : "Strażnik lokalny (bez kontekstu rozmowy): \(opinion.model)")
    }

    private var summaryLine: String? {
        if tool.helperBackground && !tool.helperRunning {
            return tool.helperReport == nil ? "Pomocnik w tle zatrzymany" : "Pomocnik skończył w tle"
        }
        if let summary = tool.summary, !summary.isEmpty { return summary }
        if tool.isRunning { return "Pracuję…" }
        let argument = tool.mainArgument
        return argument.isEmpty ? nil : argument
    }

    @ViewBuilder
    private var stateIndicator: some View {
        if tool.helperBackground && tool.helperRunning {
            // The call returned at once; the helper is still at it.
            HStack(spacing: 4) {
                ProgressView().controlSize(.small).scaleEffect(0.55).frame(width: 12, height: 12)
                Text("w tle")
                    .font(.system(size: 10, weight: .medium))
            }
            .foregroundStyle(Theme.accent)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(Theme.accent.opacity(0.12))
            .clipShape(Capsule())
            .help("Pomocnik pracuje w tle — agent nie czeka; raport dostanie, gdy pomocnik skończy.")
        } else if tool.isRunning {
            ProgressView().controlSize(.small).scaleEffect(0.65).frame(width: 14, height: 14)
        } else {
            Image(systemName: tool.ok == true ? "checkmark.circle.fill" : "xmark.octagon.fill")
                .font(.system(size: 12))
                .foregroundStyle(stateTint)
        }
    }

    // MARK: - Body

    @ViewBuilder
    /// What a helper agent did, nested under the `delegate` call that started it.
    ///
    /// Indented and dimmed on purpose: these are the call's detail, not events
    /// of the conversation — the answer the helper came back with is in the
    /// card's own output.
    private func helperBlock(for tool: ToolState) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 5) {
                Image(systemName: tool.helperRemote ? "cloud" : "person.2")
                    .font(.system(size: 10))
                Text(helperHeadline(tool))
                    .font(.system(size: 11, weight: .medium))
                if tool.helperRemote {
                    Text("zdalny")
                        .font(.system(size: 10))
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(Theme.fgSubtle.opacity(0.12))
                        .clipShape(Capsule())
                }
            }
            .foregroundStyle(Theme.fgMuted)

            ForEach(Array(tool.helperSteps.enumerated()), id: \.offset) { _, step in
                Text(step)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.fgSubtle)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let report = tool.helperReport, !report.isEmpty {
                // What the agent was handed when the background helper finished.
                Text(report)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fg)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 3)
            }
        }
        .padding(.leading, 10)
        .overlay(alignment: .leading) {
            Rectangle().fill(Theme.border).frame(width: 1)
        }
    }

    private func helperHeadline(_ tool: ToolState) -> String {
        let name = tool.helperLabel.isEmpty ? "Pomocnik" : "Pomocnik \(tool.helperLabel)"
        if tool.helperRunning {
            return tool.helperBackground ? "\(name) pracuje w tle…" : "\(name) pracuje…"
        }
        if tool.helperBackground {
            return tool.helperReport == nil ? "\(name) (w tle) — zatrzymany" : "\(name) skończył w tle — raport:"
        }
        return name
    }

    private func expandedBody(for tool: ToolState) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if showsArguments {
                Text(tool.argumentSummary)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(Theme.fgMuted)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            if let diff = tool.diff, !diff.isEmpty {
                DiffView(diff: diff)
            }

            if !tool.helperSteps.isEmpty || tool.helperRunning || tool.helperReport != nil {
                helperBlock(for: tool)
            }

            let output = tool.bodyText
            if !output.isEmpty, showsOutput(output) {
                outputBlock(output)
            }
        }
        .padding(10)
    }

    /// Arguments are worth repeating only when the title does not already say
    /// it all ("Polecenie: ls -la" plus "command: ls -la" is noise).
    private var showsArguments: Bool {
        guard !tool.argumentSummary.isEmpty else { return false }
        let single = (tool.arguments?.objectValue?.count ?? 0) <= 1
        if single, !tool.mainArgument.isEmpty, tool.displayTitle.contains(tool.mainArgument) {
            return false
        }
        return true
    }

    /// A short "Created report.md (1 line)." next to its own diff is noise too.
    private func showsOutput(_ output: String) -> Bool {
        guard let diff = tool.diff, !diff.isEmpty else { return true }
        return output.count > 200
    }

    private func outputBlock(_ output: String) -> some View {
        let lines = output.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let visible = Array(lines.prefix(Self.lineLimit))
        let hidden = lines.count - visible.count
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("Wyjście")
                    .font(.system(size: 10.5, weight: .semibold))
                    .textCase(.uppercase)
                    .kerning(0.5)
                    .foregroundStyle(Theme.fgSubtle)
                Spacer(minLength: 0)
                Button {
                    Pasteboard.copy(output)
                    copied = true
                    Task {
                        try? await Task.sleep(nanoseconds: 1_500_000_000)
                        copied = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 9, weight: .semibold))
                        Text(copied ? "Skopiowano" : "Kopiuj")
                            .font(.system(size: 10.5))
                    }
                    .foregroundStyle(copied ? Theme.success : Theme.fgSubtle)
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle())
            }

            Text(visible.joined(separator: "\n"))
                .font(.system(size: 11.5, design: .monospaced))
                .foregroundStyle(Theme.fg)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

            if hidden > 0 {
                Text("… oraz \(Format.plural(hidden, "kolejna linia", "kolejne linie", "kolejnych linii")) (skopiuj, aby zobaczyć całość)")
                    .font(.system(size: 10.5))
                    .foregroundStyle(Theme.fgSubtle)
            }
        }
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }

    private var diffCounts: (added: Int, removed: Int)? {
        guard let diff = tool.diff, !diff.isEmpty else { return nil }
        let counts = DiffLine.counts(DiffLine.parse(diff))
        return counts.added == 0 && counts.removed == 0 ? nil : counts
    }

    private static let lineLimit = 200
}

// MARK: - Diff

/// A unified diff with the usual red/green line backgrounds.
struct DiffView: View {
    let diff: String
    var lineLimit: Int = 260

    private var lines: [DiffLine] { DiffLine.parse(diff) }

    var body: some View {
        let all = lines
        let visible = Array(all.prefix(lineLimit))
        let hidden = all.count - visible.count
        VStack(alignment: .leading, spacing: 0) {
            ForEach(visible) { line in
                Text(line.text.isEmpty ? " " : line.text)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(foreground(for: line.kind))
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 1)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(background(for: line.kind))
            }
            if hidden > 0 {
                Text("… oraz \(Format.plural(hidden, "kolejna linia", "kolejne linie", "kolejnych linii")) różnicy")
                    .font(.system(size: 10.5))
                    .foregroundStyle(Theme.fgSubtle)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
            }
        }
        .padding(.vertical, 4)
        .background(Theme.codeBg)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }

    private func background(for kind: DiffLine.Kind) -> Color {
        switch kind {
        case .added: Theme.success.opacity(0.16)
        case .removed: Theme.danger.opacity(0.14)
        case .hunk: Theme.accent.opacity(0.08)
        case .meta, .context: Color.clear
        }
    }

    private func foreground(for kind: DiffLine.Kind) -> Color {
        switch kind {
        case .added: Theme.success
        case .removed: Theme.danger
        case .hunk: Theme.accent
        case .meta: Theme.fgSubtle
        case .context: Theme.fgMuted
        }
    }
}
