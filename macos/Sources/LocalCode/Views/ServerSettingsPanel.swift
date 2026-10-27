import SwiftUI
import LocalCodeKit

/// The "Serwer" page of the settings section.
///
/// It exists mostly so the two failure modes are visible without reading a log:
/// which `localcode` binary is running, and whether this app owns that process
/// or joined one the terminal already had open.
struct ServerSettingsPanel: View {
    let controller: ServerController

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Serwer LocalCode",
                          subtitle: "Cała praca — modele, rozmowy, agent — dzieje się w lokalnym procesie „localcode serve”. Aplikacja nigdy nie wysyła danych poza ten komputer.")

            Card {
                VStack(alignment: .leading, spacing: 10) {
                    row("Stan", value: stateText, tint: stateTint)
                    Hairline()
                    row("Adres", value: controller.connection?.baseURL.absoluteString ?? "—")
                    Hairline()
                    row("Wersja", value: controller.serverVersion ?? "—")
                    Hairline()
                    row("Proces", value: processText)
                }
            }

            if !controller.ownsServer, controller.phase == .ready {
                Banner(.info,
                       message: "Aplikacja dołączyła do serwera, który już działał (np. uruchomionego z terminala). Dzięki temu modele nie są ładowane dwa razy, a zamknięcie okna nie zatrzyma serwera.")
            }

            SectionHeader("Plik wykonywalny")

            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Text(controller.chosenExecutable
                         ?? controller.executablePath
                         ?? "wykryty automatycznie")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 10) {
                        SecondaryButton(title: "Wskaż inny…", systemImage: "folder", height: 34) {
                            controller.chooseExecutable()
                        }
                        if controller.chosenExecutable != nil {
                            SecondaryButton(title: "Wykryj automatycznie", height: 34) {
                                controller.forgetChosenExecutable()
                                controller.start()
                            }
                        }
                        Spacer(minLength: 0)
                    }

                    Text("Zmiana ścieżki ponownie łączy z serwerem. Jeśli aplikacja sama uruchomiła serwer, zostanie on zatrzymany dopiero przy zamknięciu okna.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgSubtle)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if !controller.logLines.isEmpty {
                SectionHeader("Log startowy")
                Card(padding: 12, background: Theme.codeBg) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(controller.logLines.suffix(20).enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.fgMuted)
                                .textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
            }
        }
    }

    private var stateText: String {
        switch controller.phase {
        case .connecting: "łączenie…"
        case .ready: "połączony"
        case .failed: "błąd połączenia"
        }
    }

    private var processText: String {
        guard controller.phase == .ready else { return "—" }
        return controller.ownsServer ? "uruchomiony przez aplikację" : "współdzielony z terminalem"
    }

    private var stateTint: Color {
        switch controller.phase {
        case .connecting: Theme.fgMuted
        case .ready: Theme.success
        case .failed: Theme.danger
        }
    }

    private func row(_ label: String, value: String, tint: Color = Theme.fg) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
                .frame(width: 90, alignment: .leading)
            Text(value)
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(tint)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }
}
