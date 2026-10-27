import SwiftUI
import AppKit
import LocalCodeKit

/// "Cerber" — the user's antivirus checking every model before it downloads.
///
/// Cerber.app has a local API (Integracje → API (localhost), off by default,
/// alive only while the app is open) with a token on every request. With the
/// switch on, a download first asks Cerber to scan the repository and starts
/// only on a clean verdict; a closed Cerber stops the download with a message
/// instead of letting it through unchecked. The token is pasted here once and
/// never comes back from the server.
public struct CerberPanel: View {
    @Environment(ModelsStore.self) private var store

    @State private var token = ""
    @State private var url = ""
    @State private var isEditingToken = false
    @State private var isSaving = false

    public init() {}

    private var cerber: CerberAppConfig { store.cerber }

    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Cerber — sprawdzanie przed pobraniem",
                          subtitle: "Jeśli masz Cerbera, LocalCode poprosi go o przeskanowanie każdego modelu, zanim pobierze choćby bajt. Model uznany za podejrzany albo złośliwy nie zostanie pobrany. Skan trwa zwykle kilka minut — Cerber sam pobiera i rozkłada repozytorium w piaskownicy.")
            Card {
                VStack(alignment: .leading, spacing: 16) {
                    if !cerber.installed {
                        Label("Nie znaleziono aplikacji Cerber w folderze Aplikacje.",
                              systemImage: "exclamationmark.triangle")
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.warning)
                    }
                    steps
                    Hairline()
                    tokenRow
                    addressRow
                    Hairline()
                    toggle
                    status
                }
            }
        }
        .task {
            url = cerber.url
            if cerber.tokenSet { await store.checkCerber() }
        }
    }

    // MARK: - Parts

    private var steps: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Jak połączyć")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Theme.fg)
            Text("1. Otwórz Cerbera → Integracje → API (localhost) i włącz API.\n"
                 + "2. Skopiuj token, który pokaże Cerber.\n"
                 + "3. Wklej go poniżej i włącz sprawdzanie. Cerber musi być otwarty, gdy pobierasz model.")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
            if cerber.installed {
                Button("Otwórz Cerbera") {
                    NSWorkspace.shared.open(URL(fileURLWithPath: "/Applications/Cerber.app"))
                }
                .buttonStyle(.link)
                .font(.system(size: 12))
            }
        }
    }

    @ViewBuilder
    private var tokenRow: some View {
        if cerber.tokenSet, !isEditingToken {
            HStack(spacing: 10) {
                Label("Token zapisany", systemImage: "key.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                Spacer(minLength: 0)
                SecondaryButton(title: "Zmień", height: 28) { isEditingToken = true }
                SecondaryButton(title: "Usuń", height: 28) {
                    Task { await store.saveCerberToken(nil) }
                }
            }
        } else {
            HStack(spacing: 10) {
                SecureField("Token API Cerbera", text: $token)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 380)
                PrimaryButton(title: "Zapisz",
                              isLoading: isSaving,
                              isEnabled: !token.trimmingCharacters(in: .whitespaces).isEmpty && !isSaving,
                              fullWidth: false,
                              height: 30) {
                    isSaving = true
                    Task {
                        await store.saveCerberToken(token)
                        isSaving = false
                        isEditingToken = false
                        token = ""
                    }
                }
            }
        }
    }

    private var addressRow: some View {
        HStack(spacing: 10) {
            Text("Adres API")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
            TextField("http://127.0.0.1:8765", text: $url)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 240)
                .onSubmit { Task { await store.setCerberURL(url) } }
            if url.trimmingCharacters(in: .whitespaces) != cerber.url {
                SecondaryButton(title: "Zapisz adres", height: 28) {
                    Task { await store.setCerberURL(url) }
                }
            }
        }
    }

    private var toggle: some View {
        Toggle(isOn: Binding(get: { cerber.enabled },
                             set: { on in Task { await store.setCerberEnabled(on) } })) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Sprawdzaj modele Cerberem przed pobraniem")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.fg)
                Text(cerber.tokenSet
                     ? "Każde pobranie — w aplikacji, w terminalu i przez Pomocnika — najpierw czeka na werdykt Cerbera."
                     : "Najpierw wklej token z Cerbera.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .toggleStyle(.switch)
        .disabled(!cerber.tokenSet && !cerber.enabled)
    }

    private var status: some View {
        HStack(spacing: 8) {
            if store.isCheckingCerber {
                ProgressView().controlSize(.small)
                Text("Sprawdzam połączenie…")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
            } else if let check = store.cerberCheck {
                Image(systemName: check.ok ? "checkmark.shield.fill" : "xmark.shield")
                    .foregroundStyle(check.ok ? Theme.success : Theme.danger)
                Text(check.message)
                    .font(.system(size: 11.5))
                    .foregroundStyle(check.ok ? Theme.fgMuted : Theme.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            SecondaryButton(title: "Sprawdź połączenie", height: 28) {
                Task { await store.checkCerber() }
            }
            .disabled(!cerber.tokenSet)
        }
    }
}
