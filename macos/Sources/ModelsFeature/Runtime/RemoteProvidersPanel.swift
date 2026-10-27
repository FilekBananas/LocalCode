import SwiftUI
import LocalCodeKit

/// "Modele zdalne" — the providers whose models can be borrowed.
///
/// LocalCode is a local-first app and this screen does not change that: nothing
/// here does anything until somebody pastes a key, every borrowed model is
/// marked as remote wherever it appears, and a key can be switched off without
/// being deleted. The key is verified against the provider *before* it is
/// saved, because finding out that a key is wrong three screens later is the
/// kind of thing this app should not do.
public struct RemoteProvidersPanel: View {
    @Environment(ModelsStore.self) private var store

    public init() {}

    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Modele zdalne",
                          subtitle: "Wklej klucz API, a modele danego dostawcy pojawią się na listach obok lokalnych — z ikoną chmury. Nic nie wychodzi z tego Maca, dopóki nie wybierzesz takiego modelu.")

            ForEach(store.remoteProviders) { provider in
                RemoteProviderRow(provider: provider)
            }

            if store.remoteProviders.isEmpty {
                Card { Text("Ładuję listę dostawców…").foregroundStyle(Theme.fgMuted) }
            }
        }
        .task { await store.refreshRemoteProviders() }
    }
}

private struct RemoteProviderRow: View {
    let provider: RemoteProviderDTO

    @Environment(ModelsStore.self) private var store
    @State private var key = ""
    @State private var isSaving = false
    @State private var isEditing = false

    var body: some View {
        Card(padding: 14) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Text(provider.label)
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    if provider.hasKey {
                        MetaBadge(text: statusText)
                    }
                    Spacer(minLength: 0)
                    if provider.hasKey {
                        Toggle("", isOn: Binding(get: { provider.enabled },
                                                 set: { store.setRemoteEnabled(provider.id, $0) }))
                            .labelsHidden()
                            .toggleStyle(.switch)
                            .help(provider.enabled ? "Wyłącz bez kasowania klucza" : "Włącz z powrotem")
                    }
                }

                if provider.hasKey, !isEditing {
                    HStack(spacing: 10) {
                        Text("Klucz zapisany")
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.fgMuted)
                        Spacer(minLength: 0)
                        SecondaryButton(title: "Zmień", height: 28) { isEditing = true }
                        SecondaryButton(title: "Usuń", height: 28) {
                            store.setRemoteKey(provider.id, key: nil)
                        }
                    }
                } else {
                    HStack(spacing: 10) {
                        SecureField("Klucz API", text: $key)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 380)
                        PrimaryButton(title: "Sprawdź i zapisz",
                                      isLoading: isSaving,
                                      isEnabled: !key.trimmingCharacters(in: .whitespaces).isEmpty
                                                 && !isSaving,
                                      height: 30) {
                            isSaving = true
                            Task {
                                await store.saveRemoteKey(provider.id, key: key)
                                isSaving = false
                                isEditing = false
                                key = ""
                            }
                        }
                        Link("Skąd wziąć klucz?", destination: URL(string: provider.keyUrl)!)
                            .font(.system(size: 11.5))
                    }
                }

                if provider.hasKey, !provider.serviceTiers.isEmpty {
                    tierPicker
                }
            }
        }
    }

    /// Standard / Flex / Priority — what every request to this provider asks for.
    private var tierPicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Text("Tryb")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                Picker("Tryb", selection: Binding(
                    get: { provider.serviceTier },
                    set: { store.setRemoteServiceTier(provider.id, $0) })) {
                    ForEach(provider.serviceTiers, id: \.self) { tier in
                        Text(Self.tierName(tier)).tag(tier)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(maxWidth: 280)
                Spacer(minLength: 0)
            }
            Text(tierHint)
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    static func tierName(_ tier: String) -> String {
        switch tier {
        case "flex": "Flex"
        case "priority": "Priority"
        case "standard": "Standard"
        default: tier.capitalized
        }
    }

    private var tierHint: String {
        let hint = switch provider.serviceTier {
        case "flex": "Taniej (DeepInfra ok. 0,8× ceny), ale odpowiedź może czekać do 10 minut, gdy brakuje mocy."
        case "priority": "Szybciej przy dużym obciążeniu, drożej (DeepInfra ok. 1,5× ceny)."
        default: "Domyślnie."
        }
        guard provider.id == "unlid" else { return hint }
        return hint + " Unlid przekazuje to ustawienie do dostawcy modelu — czy zadziała, zależy od modelu."
    }

    private var statusText: String {
        if !provider.enabled { return "wyłączony" }
        if provider.modelCount < 0 { return "nie odpowiada" }
        // Format.plural already puts the number in front ("23 modele").
        return Format.plural(provider.modelCount, "model", "modele", "modeli")
    }
}
