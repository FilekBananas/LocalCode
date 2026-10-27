import SwiftUI
import LocalCodeKit
import ModelsFeature

/// "Obrazy" — local text-to-image.
///
/// Two panes, like the rest of the app: what to generate on the left, what has
/// been generated on the right. Nothing here leaves the machine; the model runs
/// in its own worker process and shares the same RAM budget as the language
/// models, which is why the picker says out loud whether the chosen model still
/// fits.
public struct ImagesView: View {
    @Environment(ImagesStore.self) private var store
    @Environment(ModelsStore.self) private var models

    public init() {}

    public var body: some View {
        @Bindable var store = store

        VStack(spacing: 0) {
            header
            messages
            HStack(spacing: 0) {
                ImageComposer()
                    .frame(width: 380)
                Hairline(vertical: true)
                ImageGallery()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Theme.bg)
        .sheet(item: $store.detail) { image in
            ImageDetailSheet(image: image)
        }
        .task { await store.start() }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Obrazy")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("Generowanie obrazów na tym Macu — nic nie opuszcza komputera")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            if store.isLoadingGallery || store.isLoadingModels {
                ProgressView().controlSize(.small)
            }
            IconButton(systemName: "arrow.clockwise", label: "Odśwież") {
                Task { await store.refresh() }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
    }

    @ViewBuilder
    private var messages: some View {
        @Bindable var store = store

        if store.errorMessage != nil || store.statusMessage != nil {
            VStack(spacing: 8) {
                if let message = store.errorMessage {
                    Banner(.error, message: message, onDismiss: { store.errorMessage = nil })
                }
                if let note = store.statusMessage {
                    Banner(.success, message: note)
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .animation(.easeOut(duration: 0.18), value: store.errorMessage)
            .animation(.easeOut(duration: 0.18), value: store.statusMessage)
        }
    }
}

// MARK: - Composer

/// The left pane: which model, what to draw, and in what shape.
struct ImageComposer: View {
    @Environment(ImagesStore.self) private var store
    @Environment(ModelsStore.self) private var models

    @State private var showsAdvanced = false

    var body: some View {
        @Bindable var store = store

        return ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                modelSection
                promptSection
                sizeSection
                advancedSection
                generateSection
            }
            .padding(16)
        }
        .background(Theme.sidebar)
    }

    // MARK: Model

    @ViewBuilder
    private var modelSection: some View {
        if store.models.isEmpty {
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Brak modeli graficznych")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Żeby generować obrazy, zainstaluj model z kategorii „Obrazy” w sklepie z modelami. Najlżejszy (Stable Diffusion 1.5) zajmuje ok. 1,8 GB.")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                    PrimaryButton(title: "Otwórz sklep z modelami",
                                  systemImage: "shippingbox",
                                  fullWidth: true,
                                  height: 36) {
                        ShellBus.shared.request(.show(.discover))
                    }
                }
            }
        } else {
            LabeledField("Model") {
                VStack(alignment: .leading, spacing: 7) {
                    // `.menuStyle(.borderlessButton)` draws the label itself and
                    // quietly drops trailing content it does not feel like
                    // laying out, so the box and the fit badge live outside the
                    // Menu and only the name stays inside it.
                    Menu {
                        ForEach(store.models) { model in
                            Button {
                                store.selectModel(model.key)
                            } label: {
                                if model.key == store.selectedModelKey {
                                    Label(menuTitle(model), systemImage: "checkmark")
                                } else {
                                    Text(menuTitle(model))
                                }
                            }
                        }
                        Divider()
                        Button("Zainstaluj kolejny model…") { ShellBus.shared.request(.show(.discover)) }
                    } label: {
                        HStack(spacing: 7) {
                            Image(systemName: "photo")
                                .font(.system(size: 11, weight: .semibold))
                                .foregroundStyle(Theme.fgMuted)
                            Text(store.selectedModel?.displayName ?? "Wybierz model")
                                .font(.system(size: 13))
                                .foregroundStyle(Theme.fg)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                    .menuStyle(.borderlessButton)
                    .padding(.horizontal, 11)
                    .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .strokeBorder(Theme.border, lineWidth: 1)
                    )

                    if let model = store.selectedModel {
                        HStack(spacing: 7) {
                            fitBadge(model)
                            Text(Format.bytes(model.sizeBytes))
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgSubtle)
                            if model.loaded {
                                Text("w pamięci")
                                    .font(.system(size: 11))
                                    .foregroundStyle(Theme.success)
                            }
                            Spacer(minLength: 0)
                        }
                    }
                }
            }
        }
    }

    private func menuTitle(_ model: ImageModelDTO) -> String {
        var parts = [model.displayName, Format.bytes(model.sizeBytes)]
        if model.loaded { parts.append("załadowany") }
        return parts.joined(separator: " · ")
    }

    private func fitBadge(_ model: ImageModelDTO) -> some View {
        let budget = models.runtime.budget
        let available = max(0, budget.limitBytes - budget.usedBytes)
        let fit = store.fitHint(for: model, availableBytes: available)
        return HStack(spacing: 3) {
            Image(systemName: fit.systemImage).font(.system(size: 9, weight: .semibold))
            Text(fit.label).font(.system(size: 10.5, weight: .semibold))
        }
        .foregroundStyle(fit.tint)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(fit.tint.opacity(0.14))
        .clipShape(Capsule())
        .help("Model potrzebuje ok. \(Format.bytes(model.memoryCost)). LocalCode w razie potrzeby zwolni pamięć po modelach językowych.")
    }

    // MARK: Prompt

    private var promptSection: some View {
        @Bindable var store = store

        return VStack(alignment: .leading, spacing: 10) {
            LabeledField("Opis obrazu") {
                StyledTextEditor(placeholder: "a cat wearing a hat, sitting on a roof, watercolour",
                                 text: $store.prompt,
                                 minHeight: 96,
                                 maxHeight: 180)
            }

            // Learned the hard way: a Polish prompt silently loses whole
            // clauses ("w kapeluszu", "akwarela" both vanished). The app does
            // not translate behind the user's back — it says so instead.
            HStack(spacing: 5) {
                Image(systemName: "character.bubble")
                    .font(.system(size: 10))
                Text("Modele graficzne najlepiej rozumieją angielski — polski opis bywa częściowo pomijany.")
                    .font(.system(size: 11))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(Theme.fgSubtle)

            if store.showsNegativePrompt {
                LabeledField("Czego unikać", description: "Elementy, których model ma nie rysować.") {
                    StyledTextEditor(placeholder: "blurry, text, watermark",
                                     text: $store.negativePrompt,
                                     minHeight: 56,
                                     maxHeight: 110)
                }
                Button("Ukryj „czego unikać”") { store.showsNegativePrompt = false }
                    .buttonStyle(.plain)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.accent)
            } else {
                Button("Dodaj „czego unikać”") { store.showsNegativePrompt = true }
                    .buttonStyle(.plain)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.accent)
            }
        }
    }

    // MARK: Size

    private var sizeSection: some View {
        @Bindable var store = store

        return LabeledField("Rozmiar", description: sizeDescription) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    ForEach(ImageSize.allCases) { preset in
                        Chip(title: preset.label,
                             isSelected: store.size == preset,
                             onTap: { store.size = preset })
                            .help(preset.hint)
                    }
                    Spacer(minLength: 0)
                }
                if store.size == .custom {
                    HStack(spacing: 8) {
                        numberField("Szerokość", value: $store.customWidth)
                        numberField("Wysokość", value: $store.customHeight)
                    }
                }
            }
        }
    }

    private var sizeDescription: String {
        let size = store.resolvedSize
        return "\(size.width) × \(size.height) px. Większy obraz to dłuższe generowanie."
    }

    private func numberField(_ label: String, value: Binding<Int>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgMuted)
            TextField("", value: value, format: .number)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .foregroundStyle(Theme.fg)
                .padding(.horizontal, 10)
                .frame(minHeight: 32)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
        }
    }

    // MARK: Advanced

    private var advancedSection: some View {
        @Bindable var store = store

        return VStack(alignment: .leading, spacing: 10) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { showsAdvanced.toggle() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: showsAdvanced ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .bold))
                    Text("Ustawienia zaawansowane")
                        .font(.system(size: 12, weight: .medium))
                    Spacer(minLength: 0)
                }
                .foregroundStyle(Theme.fgMuted)
                .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle())

            if showsAdvanced {
                VStack(alignment: .leading, spacing: 14) {
                    stepsControl
                    guidanceControl
                    seedControl
                }
                .padding(12)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
            }
        }
    }

    private var stepsControl: some View {
        @Bindable var store = store
        let value = store.steps ?? store.selectedModel?.defaults.steps ?? 20

        return LabeledField("Kroki",
                            description: "Więcej kroków = więcej detali, ale dłużej. \(defaultNote(store.selectedModel?.defaults.steps.map(String.init)))") {
            HStack(spacing: 10) {
                Slider(value: Binding(get: { Double(value) },
                                      set: { store.steps = Int($0.rounded()) }),
                       in: 1...60, step: 1)
                Text("\(value)")
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(Theme.fg)
                    .monospacedDigit()
                    .frame(width: 28, alignment: .trailing)
            }
        }
    }

    private var guidanceControl: some View {
        @Bindable var store = store
        let value = store.guidance ?? store.selectedModel?.defaults.guidance ?? 7

        return LabeledField("Zgodność z opisem (guidance)",
                            description: "Wyżej = model trzyma się opisu ściślej, ale obraz bywa sztywny.") {
            HStack(spacing: 10) {
                Slider(value: Binding(get: { value },
                                      set: { store.guidance = ($0 * 10).rounded() / 10 }),
                       in: 0...20, step: 0.5)
                Text(String(format: "%.1f", value).replacingOccurrences(of: ".", with: ","))
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(Theme.fg)
                    .monospacedDigit()
                    .frame(width: 34, alignment: .trailing)
            }
        }
    }

    private var seedControl: some View {
        @Bindable var store = store

        return LabeledField("Ziarno (seed)",
                            description: "To samo ziarno i ten sam opis dają ten sam obraz.") {
            HStack(spacing: 8) {
                if let seed = store.seed {
                    Text("\(seed)")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                        .padding(.horizontal, 10)
                        .frame(minHeight: 30)
                        .background(Theme.surface2)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .textSelection(.enabled)
                } else {
                    Text("losowy")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .padding(.horizontal, 10)
                        .frame(minHeight: 30)
                        .background(Theme.surface2)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                }
                Spacer(minLength: 0)
                if store.seed != nil {
                    SecondaryButton(title: "Losowy", height: 30) { store.randomiseSeed() }
                }
                if store.lastSeed != nil, store.seed != store.lastSeed {
                    SecondaryButton(title: "Użyj ponownie", height: 30) { store.reuseLastSeed() }
                        .help("Wstaw ziarno ostatnio wygenerowanego obrazu")
                }
            }
        }
    }

    private func defaultNote(_ value: String?) -> String {
        guard let value else { return "" }
        return "Ten model domyślnie używa \(value)."
    }

    // MARK: Generate

    private var generateSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.isGenerating {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(store.generationStatus ?? store.progressLabel ?? "Generuję…")
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.fgMuted)
                            .lineLimit(2)
                        Spacer(minLength: 0)
                    }
                    if let progress = store.progress {
                        ProgressView(value: progress)
                            .progressViewStyle(.linear)
                            .tint(Theme.accent)
                    }
                    SecondaryButton(title: "Przerwij", systemImage: "stop.circle", fullWidth: true, height: 36) {
                        store.cancelGeneration()
                    }
                }
            } else {
                PrimaryButton(title: "Generuj",
                              systemImage: "sparkles",
                              isEnabled: store.canGenerate,
                              fullWidth: true,
                              height: 42) {
                    store.generate()
                }
                .help(store.canGenerate
                      ? "Wygeneruj obraz na tym Macu"
                      : "Wybierz model i wpisz opis obrazu")
            }
        }
    }
}
