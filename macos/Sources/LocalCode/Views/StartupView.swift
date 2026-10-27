import SwiftUI
import LocalCodeKit

/// What the window shows before the Python core answers.
///
/// The failure half matters more than the success half: the server is a
/// separate program, and "it did not start" has to be actionable — what was
/// tried, what the process printed, and a way to point at the right binary.
struct StartupView: View {
    let controller: ServerController

    var body: some View {
        Group {
            if controller.phase == .failed {
                failure
            } else {
                progress
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }

    // MARK: - Connecting

    private var progress: some View {
        VStack(spacing: 16) {
            mark
            Text("LocalCode")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Theme.fg)
            ProgressView()
                .controlSize(.small)
            Text(controller.statusText)
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        // Centred by filling the window rather than by spacers: a `Spacer`
        // above and below a flexible sibling makes the three of them fight for
        // the same height, and the loser can end up with none of it.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var mark: some View {
        RoundedRectangle(cornerRadius: 16, style: .continuous)
            .fill(Theme.accent)
            .frame(width: 60, height: 60)
            .overlay(
                Image(systemName: "cpu")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(Theme.accentFg)
            )
    }

    // MARK: - Failure

    private var failure: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 22))
                        .foregroundStyle(Theme.danger)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Nie udało się uruchomić serwera LocalCode")
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(Theme.fg)
                        Text("Aplikacja potrzebuje działającego procesu „localcode serve”, żeby wczytywać modele i prowadzić rozmowy.")
                            .font(.system(size: 12.5))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                if let message = controller.errorMessage {
                    Banner(.error, message: message)
                }

                Card(padding: 14) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Sprawdzone lokalizacje")
                            .font(.system(size: 12, weight: .semibold))
                            .textCase(.uppercase)
                            .kerning(0.6)
                            .foregroundStyle(Theme.fgMuted)
                        ForEach(controller.searchedPaths, id: \.self) { path in
                            Text(path)
                                .font(.system(size: 11.5, design: .monospaced))
                                .foregroundStyle(Theme.fgSubtle)
                                .textSelection(.enabled)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }

                if !controller.logLines.isEmpty {
                    logPanel
                }

                HStack(spacing: 10) {
                    PrimaryButton(title: "Spróbuj ponownie",
                                  systemImage: "arrow.clockwise",
                                  fullWidth: false,
                                  height: 38) {
                        controller.start()
                    }
                    SecondaryButton(title: "Wskaż plik „localcode”…",
                                    systemImage: "folder",
                                    height: 38) {
                        controller.chooseExecutable()
                    }
                    if controller.chosenExecutable != nil {
                        SecondaryButton(title: "Zapomnij wybraną ścieżkę", height: 38) {
                            controller.forgetChosenExecutable()
                            controller.start()
                        }
                    }
                    Spacer(minLength: 0)
                }
            }
            .frame(maxWidth: 720, alignment: .leading)
            .frame(maxWidth: .infinity)
            .padding(28)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Ostatnie linie z serwera")
                .font(.system(size: 12, weight: .semibold))
                .textCase(.uppercase)
                .kerning(0.6)
                .foregroundStyle(Theme.fgMuted)

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(controller.logLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(Theme.fg)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(12)
            }
            .frame(maxHeight: 220)
            .background(Theme.codeBg)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )

            Button {
                Pasteboard.copy(controller.logLines.joined(separator: "\n"))
            } label: {
                Label("Kopiuj log", systemImage: "doc.on.doc")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .buttonStyle(PressableButtonStyle())
        }
    }
}
