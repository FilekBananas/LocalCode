import SwiftUI
import LocalCodeKit

/// The default reasoning effort for the chat and for the agent.
///
/// These are `app-settings.chat.reasoningEffort` and
/// `app-settings.agent.reasoningEffort`: what a conversation or a session uses
/// when it has not been given a level of its own. A model that cannot reason at
/// all ignores them, and a model with only a boolean switch reads anything
/// other than "Wyłączone" as "on" — which is why the control here offers the
/// full four levels rather than pretending to know which model will be used.
public struct ReasoningDefaultsPanel: View {
    @Environment(ModelsStore.self) private var store

    /// Every canonical level the server accepts, strongest last.
    private static let levels: [EffortLevel] = [.off, .low, .medium, .high]

    public init() {}

    /// Containment and scanning of downloaded models.
    ///
    /// A model file is data, but the programs that read it parse megabytes of
    /// somebody else's structure — so the engine runs with no network and no
    /// access to the user's files, and the files are checked for executable
    /// content before the model may load. Both are on by default; the switches
    /// exist because debugging sometimes needs them off.
    private var securityCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Bezpieczeństwo modeli",
                          subtitle: "Model z internetu to obcy plik. LocalCode uruchamia go bez dostępu do sieci i do Twoich plików, a przed pierwszym użyciem sprawdza, czy nie zawiera kodu.")
            Card {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Piaskownica")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(Theme.fg)
                        Text("Proces silnika nie wychodzi w internet i nie widzi Twoich dokumentów, kluczy ani Keychaina.")
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                        Picker("Piaskownica", selection: Binding(
                            get: { store.sandboxMode },
                            set: { mode in Task { await store.setSandboxMode(mode) } })) {
                            Text("Włączona").tag("auto")
                            Text("Wymagana").tag("strict")
                            Text("Wyłączona").tag("off")
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        Text(sandboxHint)
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Hairline()
                    Toggle(isOn: Binding(get: { store.scanDownloads },
                                         set: { on in Task { await store.setScanDownloads(on) } })) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Sprawdzaj pobrane modele")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(Theme.fg)
                            Text("Szuka w plikach modelu kodu, który mógłby się wykonać przy wczytywaniu. Model uznany za groźny nie zostanie wczytany bez Twojej wyraźnej zgody.")
                                .font(.system(size: 11.5))
                                .foregroundStyle(Theme.fgMuted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)  // switch on the card's right edge
                    }
                    .toggleStyle(.switch)
                }
            }
        }
    }

    private var sandboxHint: String {
        switch store.sandboxMode {
        case "strict": return "Model, którego nie da się uruchomić w piaskownicy, nie zostanie wczytany."
        case "off": return "Bez izolacji — używaj tylko do diagnozowania problemów z ładowaniem."
        default: return "Gdy model nie chce się uruchomić w piaskownicy, LocalCode startuje go bez niej i zapisuje to w stanie środowiska."
        }
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Myślenie",
                          subtitle: "Domyślny poziom rozumowania dla nowych rozmów i sesji agenta. Pojedynczą rozmowę możesz zawsze przestawić w jej własnym nagłówku.")

            Card {
                Toggle(isOn: Binding(get: { store.agentManagesModels },
                                     set: { store.setAgentManagesModels($0) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Asystent zarządza modelami")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(Theme.fg)
                        Text("Pozwala powiedzieć „pobierz mi dobry model do kodowania” — "
                             + "asystent go znajdzie, pobierze i przełączy na niego rozmowę. "
                             + "W trybie Kod każde pobranie pyta o zgodę; w czacie wykonuje się "
                             + "od razu, a pobieranie widać na liście i można je przerwać.")
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)  // switch on the card's right edge
                }
                .toggleStyle(.switch)
                .padding(2)
            }

            Card {
                Toggle(isOn: Binding(get: { store.subagentsEnabled },
                                     set: { store.setSubagentsEnabled($0) })) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Agent może zlecać pracę pomocnikowi")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(Theme.fg)
                        Text("Szukanie po projekcie, uruchomienie kilku przebiegów czy analiza "
                             + "kilku logów zajmuje dziesiątki kroków, których agent już nigdy nie "
                             + "potrzebuje. Pomocnik robi to w osobnym kontekście — także w tle, "
                             + "gdy agent pracuje dalej — i oddaje krótki raport. Czyta pliki i "
                             + "uruchamia polecenia (każde pyta o zgodę jak polecenia agenta), "
                             + "ale nie edytuje plików.")
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)  // switch on the card's right edge
                }
                .toggleStyle(.switch)
                .padding(2)
            }

            securityCard

            Card {
                VStack(alignment: .leading, spacing: 16) {
                    row(title: "Czat",
                        description: "Używany, gdy rozmowa nie ma własnego ustawienia.",
                        selection: store.settings.chat.reasoningEffort) { level in
                        Task { await store.setReasoningEffort(level, for: .chat) }
                    }
                    Hairline()
                    row(title: "Agent (Kod)",
                        description: "Startowy poziom nowych sesji agenta programistycznego.",
                        selection: store.settings.agent.reasoningEffort,
                        offersUltracode: true) { level in
                        Task { await store.setReasoningEffort(level, for: .agent) }
                    }
                }
            }

            Text(EffortHelp.text)
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)

            Text("""
            Modele bez trybu myślenia (np. Qwen3 Instruct) ignorują to ustawienie. \
            Modele z przełącznikiem włącz/wyłącz (Qwen3, GLM, DeepSeek) traktują \
            „Niskie”, „Średnie” i „Wysokie” tak samo — jako włączone.
            """)
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func row(title: String,
                     description: String,
                     selection: String?,
                     offersUltracode: Bool = false,
                     onChange: @escaping (String) -> Void) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text(description)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)
            EffortPicker(levels: Self.levels,
                         selection: selection ?? "medium",
                         showsLabel: false,
                         offersUltracode: offersUltracode,
                         onChange: onChange)
        }
    }
}
