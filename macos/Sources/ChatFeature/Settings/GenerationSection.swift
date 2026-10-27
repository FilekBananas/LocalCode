import SwiftUI
import LocalCodeKit

/// Defaults every new conversation starts with: which local model answers and
/// how it is asked to answer.
///
/// The cost line the web app had is gone — a local model costs nothing, so
/// every price readout would print a zero. What matters instead is the context
/// window, which is why it sits right under the picker.
struct GenerationSection: View {
    let draft: SettingsDraft

    @Environment(ChatStore.self) private var store

    private struct Option: Identifiable, Hashable {
        let value: String
        let label: String
        var id: String { value }
    }

    private var defaults: GenerationDefaults { draft.value.defaults }

    private var modelOptions: [Option] {
        var options = store.models
            .map { Option(value: $0.id, label: $0.label.isEmpty ? $0.id : $0.label) }
            .sorted { $0.label < $1.label }
        if !defaults.model.isEmpty, !options.contains(where: { $0.value == defaults.model }) {
            options.insert(Option(value: defaults.model, label: defaults.model), at: 0)
        }
        return options
    }

    private var selectedModel: ModelInfo? {
        store.models.first { $0.id == defaults.model }
    }

    /// The default answer limit must leave room for the question; warn when the
    /// two no longer fit in the model's window.
    private var outputExceedsContext: Bool {
        guard let context = selectedModel?.contextWindow, context > 0 else { return false }
        return defaults.maxTokens > context / 2
    }

    /// The slider's last stop, one step past the largest limit, means "bez limitu" (0).
    private static let unlimitedStop = 16_512

    private var maxTokensBinding: Binding<Double> {
        Binding(get: { Double(defaults.maxTokens == 0 ? Self.unlimitedStop : defaults.maxTokens) },
                set: { value in
                    let tokens = Int(value.rounded())
                    draft.set(\.defaults.maxTokens, tokens >= Self.unlimitedStop ? 0 : tokens)
                })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            SectionHeader("Domyślny model",
                          subtitle: "Używany przez nowe rozmowy. W otwartej rozmowie model zmienisz w jej nagłówku albo w polu wpisywania.")

            if store.models.isEmpty {
                Banner(.warning,
                       message: "Nie masz jeszcze żadnego modelu. Pobierz model w sekcji „Odkrywaj”.",
                       actionTitle: "Przejdź do Odkrywaj",
                       action: { ShellBus.shared.request(.show(.discover)) })
            }

            LabeledField("Model") {
                Picker("Model", selection: draft.binding(\.defaults.model)) {
                    ForEach(modelOptions) { option in
                        Text(option.label).tag(option.value)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(modelOptions.isEmpty)
            }

            if let model = selectedModel {
                modelSummary(model)
            }

            Hairline()

            SettingsSlider(title: "Temperatura",
                           value: draft.binding(\.defaults.temperature),
                           range: 0...2,
                           step: 0.1,
                           valueLabel: String(format: "%.1f", defaults.temperature),
                           description: "Niżej — odpowiedzi przewidywalne i powtarzalne. Wyżej — bardziej twórcze. Dla małych modeli 0,3–0,7 zwykle działa najlepiej.")

            SettingsSlider(title: "Maksymalna długość odpowiedzi",
                           value: maxTokensBinding,
                           range: 256...Double(Self.unlimitedStop),
                           step: 128,
                           valueLabel: defaults.maxTokens == 0 ? "bez limitu" : "\(defaults.maxTokens) tok.",
                           description: "Górny limit tokenów jednej odpowiedzi. Przesuń do końca, aby wyłączyć limit — model pisze, aż skończy albo zabraknie miejsca w oknie kontekstu.")

            if outputExceedsContext {
                Banner(.warning,
                       message: "Limit odpowiedzi zajmuje ponad połowę okna kontekstu tego modelu — na pytanie i historię zostanie bardzo mało miejsca.")
            }

            SettingsSlider(title: "Fragmenty z plików w prompcie",
                           value: draft.intBinding(\.defaults.ragTopK),
                           range: 0...12,
                           step: 1,
                           valueLabel: defaults.ragTopK == 0 ? "wyłączone" : "\(defaults.ragTopK)",
                           description: "Ile najlepiej pasujących fragmentów bazy wiedzy dokleić do pytania. 0 wyłącza wyszukiwanie w plikach.")

            SettingsSlider(title: "Limit historii wysyłanej do modelu",
                           value: draft.intBinding(\.defaults.historyLimit),
                           range: 0...50,
                           step: 1,
                           valueLabel: defaults.historyLimit == 0 ? "bez limitu" : "\(defaults.historyLimit) wiad.",
                           description: "Ile wcześniejszych wiadomości trafia do zapytania. Przy małym oknie kontekstu 10–20 to rozsądny wybór.")

            Hairline()

            SettingsToggleRow("Automatyczne tytuły rozmów",
                              description: "Po pierwszej wymianie zdań model nadaje rozmowie nazwę.",
                              isOn: draft.binding(\.defaults.autoTitle))

            SettingsToggleRow("Pokazuj rozumowanie modelu",
                              description: "Zwijany blok z tokiem myślenia, gdy model go zwraca. Rozumowanie nigdy nie wraca do modelu w kolejnej turze.",
                              isOn: draft.binding(\.defaults.showReasoning))

            HStack(alignment: .top, spacing: 7) {
                Image(systemName: "books.vertical")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
                Text("Pliki do wyszukiwania dodasz w zakładce „Wiedza” (globalnie) albo w ustawieniach projektu.")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Selected model

    private func modelSummary(_ model: ModelInfo) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if let description = model.description, !description.isEmpty {
                Text(description)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    if let context = model.contextWindow {
                        Chip(title: "kontekst \(Format.tokens(context))", isSelected: true)
                    }
                    if let output = model.maxOutput {
                        Chip(title: "wyjście \(Format.tokens(output))")
                    }
                    if model.supportsVision {
                        Chip(title: "obrazy", systemImage: "photo")
                    }
                    if model.supportsReasoning {
                        Chip(title: "rozumowanie", systemImage: "brain")
                    }
                }
                .padding(.vertical, 1)
            }
            // Local models are free; the price line only appears if a future
            // remote provider ever reports one.
            if let input = model.inputPricePerM, let output = model.outputPricePerM,
               input > 0 || output > 0 {
                Text("Cena: \(String(format: "$%.2f", input)) / mln tokenów wejścia, \(String(format: "$%.2f", output)) / mln wyjścia.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgSubtle)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}
