import SwiftUI
import LocalCodeKit

/// The coding agent's limits and auto mode's reviewer.
///
/// `app-settings.agent.maxSteps` is how many tool calls the agent may make for
/// one message (0 = until it is done or stopped), and
/// `app-settings.agent.autoReviewerModel` is the model that judges every action
/// in the "Auto" permission mode — `nil` means the model the agent itself runs
/// on, which is also the cheapest choice: it reuses the agent's prompt cache.
public struct AgentSettingsPanel: View {
    @Environment(ModelsStore.self) private var store

    /// The slider's position while it is being dragged; committed on release.
    @State private var stepsDraft: Double?
    /// "Wyczyść wyuczone decyzje" is running.
    @State private var clearing = false

    private static let maxSteps: Double = 200

    public init() {}

    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Agent",
                          subtitle: "Ile agent może zrobić na jedną wiadomość i kto ocenia jego akcje w trybie Auto.")

            Card {
                VStack(alignment: .leading, spacing: 16) {
                    stepsRow
                    Hairline()
                    reviewerRow
                    Hairline()
                    guardRow
                    Hairline()
                    learningRow
                }
            }
        }
    }

    // MARK: - Steps

    private var stepsValue: Double { stepsDraft ?? Double(store.agentMaxSteps) }

    private var stepsLabel: String {
        let steps = Int(stepsValue)
        return steps == 0 ? "bez limitu" : "\(steps)"
    }

    private var stepsRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Maks. liczba kroków agenta")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.fg)
                Spacer(minLength: 8)
                Text(stepsLabel)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.fgMuted)
                    .monospacedDigit()
            }
            Slider(value: Binding(get: { stepsValue }, set: { stepsDraft = $0.rounded() }),
                   in: 0...Self.maxSteps, step: 5,
                   onEditingChanged: { editing in
                       guard !editing, let draft = stepsDraft else { return }
                       Task {
                           await store.setAgentMaxSteps(Int(draft))
                           stepsDraft = nil
                       }
                   })
            .tint(Theme.accent)
            Text("Ile kroków (wywołań narzędzi) agent może zrobić na jedną wiadomość. "
                 + "Bez limitu — pracuje aż skończy albo go zatrzymasz.")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Auto mode's reviewer

    private var reviewerRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Recenzent w trybie Auto")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.fg)
                    Text("Model, który krytycznie ocenia każdą akcję agenta i mówi, dlaczego się zgadza albo nie.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                reviewerMenu
            }
            Text(reviewerHint)
                .font(.system(size: 11))
                .foregroundStyle(isRemoteReviewer ? Theme.warning : Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var isRemoteReviewer: Bool {
        store.autoReviewerModel.map(RemoteModelKey.isRemote) ?? false
    }

    private var reviewerHint: String {
        if let key = store.autoReviewerModel, RemoteModelKey.isRemote(key) {
            let provider = RemoteModelKey.providerLabel(of: key) ?? "dostawcy"
            return "Recenzent zdalny: rozmowa i każda oceniana akcja trafiają do \(provider), "
                + "a każda ocena to osobne, płatne zapytanie."
        }
        if store.autoReviewerModel != nil {
            return "Inny model lokalny musi zmieścić się w pamięci obok modelu agenta. "
                + "Gdy nie da się go uruchomić, ocenia model agenta."
        }
        return "Domyślnie ocenia ten sam model, na którym pracuje agent — przy modelu zdalnym ten sam "
            + "model u tego samego dostawcy. Wybierz model zdalny, a oceniane akcje wraz z rozmową "
            + "trafią do jego dostawcy, każda jako osobne, płatne zapytanie."
    }

    private var reviewerLabel: String {
        guard let key = store.autoReviewerModel else { return "Ten sam model co agent" }
        return store.model(for: key)?.pickerName ?? key
    }

    private var reviewerMenu: some View {
        Menu {
            Button {
                store.setAutoReviewerModel(nil)
            } label: {
                if store.autoReviewerModel == nil {
                    Label("Ten sam model co agent", systemImage: "checkmark")
                } else {
                    Text("Ten sam model co agent")
                }
            }
            let models = store.chatModels
            let local = models.filter { !$0.remote }
                .sorted { $0.pickerName.localizedCaseInsensitiveCompare($1.pickerName) == .orderedAscending }
            let remote = Dictionary(grouping: models.filter(\.remote)) {
                RemoteModelKey.provider(of: $0.key) ?? ""
            }
            if !local.isEmpty {
                Divider()
                Section("Na tym Macu") {
                    ForEach(local) { model in reviewerButton(model, title: model.pickerName) }
                }
            }
            if !remote.isEmpty {
                Divider()
                ForEach(remote.keys.sorted(), id: \.self) { provider in
                    let group = (remote[provider] ?? []).sorted {
                        $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
                    }
                    Menu("\(RemoteModelKey.label(for: provider)) (\(group.count))") {
                        ForEach(group) { model in reviewerButton(model, title: model.displayName) }
                    }
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: isRemoteReviewer ? "cloud" : "checkmark.shield")
                    .font(.system(size: 10, weight: .semibold))
                Text(reviewerLabel)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundStyle(Theme.fg)
            .frame(maxWidth: 240)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Theme.surface2)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
        .help("Model oceniający akcje agenta w trybie Auto")
    }

    // MARK: - Auto mode's local guard

    private var guardRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .center, spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Strażnik lokalny (bez kontekstu)")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.fg)
                    Text("Każde polecenie w trybie Auto sprawdza też lokalny model, który nie widzi "
                         + "rozmowy — więc nie da się go zmanipulować treścią plików ani czatu. "
                         + "Zawsze lokalnie, nic nie wychodzi z Maca.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 8)
                guardMenu
            }
            Text("Polecenie wykona się samo tylko wtedy, gdy zgodzą się recenzent i strażnik. "
                 + "Gdy strażnik odmówi albo nie ma lokalnego modelu, decyzja należy do Ciebie.")
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var guardLabel: String {
        guard let key = store.autoGuardModel else { return "Automatycznie (najmniejszy model lokalny)" }
        return store.model(for: key)?.pickerName ?? key
    }

    private var guardMenu: some View {
        Menu {
            Button {
                store.setAutoGuardModel(nil)
            } label: {
                if store.autoGuardModel == nil {
                    Label("Automatycznie (najmniejszy model lokalny)", systemImage: "checkmark")
                } else {
                    Text("Automatycznie (najmniejszy model lokalny)")
                }
            }
            let local = store.chatModels.filter { !$0.remote }
                .sorted { $0.sizeBytes < $1.sizeBytes }
            if !local.isEmpty {
                Divider()
                Section("Na tym Macu") {
                    ForEach(local) { model in
                        Button {
                            store.setAutoGuardModel(model.key)
                        } label: {
                            if model.key == store.autoGuardModel {
                                Label(model.pickerName, systemImage: "checkmark")
                            } else {
                                Text(model.pickerName)
                            }
                        }
                    }
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "lock.shield")
                    .font(.system(size: 10, weight: .semibold))
                Text(guardLabel)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundStyle(Theme.fg)
            .frame(maxWidth: 240)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Theme.surface2)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
        .help("Lokalny model sprawdzający polecenia bez kontekstu rozmowy")
    }

    // MARK: - Learning from the user's answers

    /// `app-settings.agent.autoPrecedentsEnabled` and `DELETE /api/agent/auto-precedents`.
    private var learningRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: Binding(get: { store.autoPrecedentsEnabled },
                                 set: { store.setAutoPrecedentsEnabled($0) })) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Ucz się z moich decyzji")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.fg)
                    Text("Gdy zatwierdzisz coś, co Auto odrzuciło, przy podobnych poleceniach recenzent "
                         + "i strażnik będą łagodniejsi (i ostrzejsi, gdy odmówisz). Zapisywane tylko "
                         + "na tym Macu, osobno dla każdego projektu.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)  // switch on the card's right edge
            }
            .toggleStyle(.switch)
            Button {
                clearing = true
                Task {
                    await store.clearAutoPrecedents()
                    clearing = false
                }
            } label: {
                Text("Wyczyść wyuczone decyzje")
                    .font(.system(size: 12, weight: .medium))
            }
            .disabled(clearing)
            .help("Zapomina wszystkie zapamiętane zgody i odmowy, we wszystkich projektach")
        }
    }

    private func reviewerButton(_ model: LocalModelDTO, title: String) -> some View {
        Button {
            store.setAutoReviewerModel(model.key)
        } label: {
            if model.key == store.autoReviewerModel {
                Label(title, systemImage: "checkmark")
            } else {
                Text(title)
            }
        }
    }
}
