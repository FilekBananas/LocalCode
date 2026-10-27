import SwiftUI
import LocalCodeKit

/// The strip above "Czat" and "Kod": which model answers here, whether it is in
/// memory, and one click to change either.
///
/// LM Studio puts this selector in the title bar and it is the right shape for
/// a local app — the answer to "why is this slow" is almost always "the model
/// is still loading", and that has to be visible without opening a panel.
///
/// The bar edits one server-side default (`chat.defaultModel` or
/// `agent.defaultModel`), so the choice survives a restart and is shared with
/// the terminal app. Which of the two it edits comes from `purpose`, or from
/// the `\.modelPurpose` environment value when the caller passes none — the
/// shell renders `ModelLoaderBar()` above both sections.
public struct ModelLoaderBar: View {
    private let explicitPurpose: ModelPurpose?

    @Environment(ModelsStore.self) private var store
    @Environment(\.modelPurpose) private var environmentPurpose

    @State private var isPickerOpen = false
    @State private var settingsTarget: LocalModelDTO?

    public init() {
        explicitPurpose = nil
    }

    public init(purpose: ModelPurpose) {
        explicitPurpose = purpose
    }

    private var purpose: ModelPurpose { explicitPurpose ?? environmentPurpose }

    private var selectedKey: String? { store.defaultModelKey(for: purpose) }
    private var selected: LocalModelDTO? { store.model(for: selectedKey) }
    private var instances: [RuntimeInstance] { selectedKey.map { store.instances(of: $0) } ?? [] }

    public var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "cpu")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(Theme.fgSubtle)

            Text(purpose.title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.fgSubtle)

            selector

            if let instance = instances.first {
                state(instance)
            } else if let selected, store.isLoading(selected.key) {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 12)
                    Text("Ładowanie…").font(.system(size: 11)).foregroundStyle(Theme.warning)
                }
            }

            Spacer(minLength: 0)

            if let selected, !selected.remote {
                if instances.isEmpty {
                    Button {
                        Task { await store.load(key: selected.key, config: selected.loadDefaults) }
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "play.fill").font(.system(size: 9, weight: .semibold))
                            Text("Załaduj").font(.system(size: 11, weight: .medium))
                        }
                        .foregroundStyle(Theme.fgMuted)
                    }
                    .buttonStyle(PressableButtonStyle())
                    .help("Załaduj model do pamięci teraz")
                } else {
                    IconButton(systemName: "eject", label: "Wyładuj model", size: 22, fontSize: 11) {
                        Task { await store.unload(key: selected.key) }
                    }
                }
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 32)
        .background(Theme.surface2)
        .overlay(alignment: .bottom) { Hairline() }
        .sheet(item: $settingsTarget) { model in
            LoadSettingsSheet(model: model).environment(store)
        }
    }

    // MARK: - Selector

    private var selector: some View {
        Button {
            isPickerOpen.toggle()
        } label: {
            HStack(spacing: 5) {
                Text(selected?.pickerName ?? (selectedKey ?? "Wybierz model"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(selected == nil ? Theme.fgSubtle : Theme.fg)
                    .lineLimit(1)
                if let selected, !selected.remote {
                    FormatBadge(format: selected.format)
                    if let context = ModelFormat.context(selected.contextLength) {
                        Text(context)
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(Theme.fgSubtle)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .help("Model używany w tej sekcji")
        .popover(isPresented: $isPickerOpen, arrowEdge: .bottom) {
            ModelPickerPopover(purpose: purpose) { model in
                isPickerOpen = false
                settingsTarget = model
            }
            .environment(store)
        }
    }

    private func state(_ instance: RuntimeInstance) -> some View {
        HStack(spacing: 5) {
            if instance.state == .loading {
                ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 12)
            } else {
                Circle()
                    .fill(instance.state == .error ? Theme.danger
                          : (instance.state == .busy ? Theme.accent : Theme.success))
                    .frame(width: 6, height: 6)
            }
            Text(instance.state == .ready
                 ? "gotowy · \(Format.tokens(instance.nCtx))"
                 : instance.state.label.lowercased())
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgMuted)
            if let tps = ModelFormat.tps(instance.genTps) {
                Text("\(tps) tok/s")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                    .monospacedDigit()
            }
        }
        .help(instance.error ?? "Instancja modelu w pamięci")
    }
}

// MARK: - Picker

/// The searchable list of local models behind the bar.
struct ModelPickerPopover: View {
    let purpose: ModelPurpose
    let onOpenSettings: (LocalModelDTO) -> Void

    @Environment(ModelsStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var filter = ""

    private var models: [LocalModelDTO] {
        let needle = filter.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let base = store.chatModels
        let matched = needle.isEmpty ? base : base.filter {
            $0.pickerName.lowercased().contains(needle) || $0.key.lowercased().contains(needle)
        }
        // "Ulubione" first — the user said so. Then loaded models — they
        // answer instantly, which is the only thing that matters when picking
        // one mid-conversation. Models behind an API key go last: they cost
        // money per request.
        let favorites = FavoriteModels.shared
        return matched.sorted { lhs, rhs in
            let lhsFavorite = favorites.contains(lhs.key)
            let rhsFavorite = favorites.contains(rhs.key)
            if lhsFavorite != rhsFavorite { return lhsFavorite }
            if lhs.remote != rhs.remote { return !lhs.remote }
            let lhsLoaded = store.isLoaded(lhs.key)
            let rhsLoaded = store.isLoaded(rhs.key)
            if lhsLoaded != rhsLoaded { return lhsLoaded }
            return lhs.pickerName.localizedCaseInsensitiveCompare(rhs.pickerName) == .orderedAscending
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            StyledTextField(placeholder: "Szukaj modelu…", text: $filter, systemImage: "magnifyingglass")
                .padding(10)

            Hairline()

            if models.isEmpty {
                VStack(spacing: 8) {
                    Text("Brak modeli na dysku")
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Button("Pobierz pierwszy model") {
                        dismiss()
                        ShellBus.shared.request(.show(.discover))
                    }
                    .buttonStyle(.link)
                    .font(.system(size: 12))
                }
                .frame(maxWidth: .infinity)
                .padding(24)
            } else {
                ScrollView {
                    LazyVStack(spacing: 2) {
                        ForEach(models) { model in
                            row(model)
                        }
                    }
                    .padding(6)
                }
                .frame(maxHeight: 340)
            }

            Hairline()

            HStack(spacing: 8) {
                Button("Załaduj z ustawieniami…") {
                    // A model behind an API key is never loaded here, so the
                    // load settings are for the local one picked (or the first).
                    let picked = store.model(for: store.defaultModelKey(for: purpose))
                    guard let model = (picked?.remote == false ? picked : nil)
                            ?? models.first(where: { !$0.remote }) else { return }
                    onOpenSettings(model)
                }
                .buttonStyle(.link)
                .font(.system(size: 12))
                .disabled(!models.contains { !$0.remote })

                Spacer(minLength: 0)

                Button("Moje modele") {
                    dismiss()
                    ShellBus.shared.request(.show(.myModels))
                }
                .buttonStyle(.link)
                .font(.system(size: 12))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .frame(width: 400)
        .background(Theme.surface)
    }

    private func row(_ model: LocalModelDTO) -> some View {
        let isSelected = store.defaultModelKey(for: purpose) == model.key
        let isLoaded = store.isLoaded(model.key)
        let fit = QuickFit.verdict(for: model, runtime: store.runtime)

        // The star sits beside the row's button, not inside it: a button in a
        // button's label swallows the outer click on macOS.
        return HStack(spacing: 2) {
            Button {
                Task { await store.setDefaultModel(model.key, for: purpose) }
                dismiss()
            } label: {
                rowLabel(model, isSelected: isSelected, isLoaded: isLoaded, fit: fit)
            }
            .buttonStyle(PressableButtonStyle())

            FavoriteStarButton(key: model.key)
                .padding(.trailing, 4)
        }
        .rowChrome(isActive: isSelected)
        .contextMenu {
            Button(FavoriteModels.shared.actionTitle(for: model.key)) {
                FavoriteModels.shared.toggle(model.key)
            }
        }
    }

    private func rowLabel(_ model: LocalModelDTO, isSelected: Bool, isLoaded: Bool,
                          fit: MemoryEstimate.Fit) -> some View {
        HStack(spacing: 8) {
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .font(.system(size: 11))
                .foregroundStyle(isSelected ? Theme.accent : Theme.fgSubtle)

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(model.pickerName)
                        .font(.system(size: 12.5, weight: isSelected ? .semibold : .regular))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    if isLoaded {
                        Circle().fill(Theme.success).frame(width: 6, height: 6)
                            .help("Załadowany w pamięci")
                    }
                }
                HStack(spacing: 8) {
                    if model.remote {
                        Text("zdalnie · \(model.providerLabel ?? "API")")
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.fgSubtle)
                    } else {
                        FormatBadge(format: model.format)
                        if let quant = model.quant {
                            Text(quant).font(.system(size: 10.5)).foregroundStyle(Theme.fgSubtle)
                        }
                        Text(Format.bytes(model.sizeBytes))
                            .font(.system(size: 10.5))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                    if let context = ModelFormat.context(model.contextLength) {
                        Text(context).font(.system(size: 10.5)).foregroundStyle(Theme.fgSubtle)
                    }
                }
            }

            Spacer(minLength: 0)

            if model.remote {
                Image(systemName: "cloud")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                    .help("Działa u dostawcy — nie zajmuje pamięci tego Maca, płatne za zapytanie")
            } else if isLoaded {
                Text("w pamięci")
                    .font(.system(size: 10.5, weight: .medium))
                    .foregroundStyle(Theme.success)
            } else {
                FitBadge(fit: fit)
                    .help(fitHelp(fit))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .contentShape(Rectangle())
    }

    private func fitHelp(_ fit: MemoryEstimate.Fit) -> String {
        switch fit {
        case .yes: "Zmieści się w wolnej pamięci"
        case .tight: "Zmieści się na styk — inne aplikacje mogą zwolnić"
        case .no: "Nie zmieści się w obecnym budżecie pamięci"
        }
    }
}
