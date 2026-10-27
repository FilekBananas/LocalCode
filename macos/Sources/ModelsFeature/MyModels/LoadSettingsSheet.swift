import SwiftUI
import LocalCodeKit

/// "Ustawienia ładowania" for one model.
///
/// The context slider is the whole point: on a 32 GB Mac the KV cache decides
/// whether a 24 B model loads at all, so every move of the slider asks
/// `POST /api/runtime/estimate` and the sheet answers with the real number and
/// the server's own Polish verdict. The slider never goes past the context the
/// model was trained for — a larger window is not a setting, it is a mistake.
struct LoadSettingsSheet: View {
    let model: LocalModelDTO

    @Environment(ModelsStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var config: LoadConfigDTO
    @State private var estimate: MemoryEstimate?
    @State private var isEstimating = false
    @State private var isSaving = false
    @State private var estimateTask: Task<Void, Never>?

    /// The context values the slider can stop on, capped by the model's own
    /// trained window.
    private let steps: [Int]

    init(model: LocalModelDTO) {
        self.model = model
        let native = model.contextLength ?? 32_768
        let ladder = [512, 1024, 2048, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768,
                      49_152, 65_536, 98_304, 131_072, 196_608, 262_144, 393_216,
                      524_288, 786_432, 1_048_576]
        var values = ladder.filter { $0 <= native }
        if values.isEmpty { values = [min(native, 512)] }
        if values.last != native { values.append(native) }
        steps = values

        var initial = model.loadDefaults ?? LoadConfigDTO()
        if initial.nCtx == nil {
            // What the server would pick anyway: a window small enough to be
            // safe on this machine, never larger than the model supports.
            initial.nCtx = values.last(where: { $0 <= 16_384 }) ?? values[0]
        }
        if initial.nParallel == nil { initial.nParallel = 1 }
        if initial.kvCacheType == nil { initial.kvCacheType = model.format == "mlx" ? "f16" : "f16" }
        if initial.flashAttn == nil { initial.flashAttn = "auto" }
        if initial.gpuLayers == nil { initial.gpuLayers = -1 }
        _config = State(initialValue: initial)
    }

    private var isGGUF: Bool { model.format.lowercased() == "gguf" }

    private var kvOptions: [String] {
        isGGUF ? ["f16", "q8_0", "q4_0"] : ["f16", "8bit", "4bit"]
    }

    private var contextIndex: Binding<Double> {
        Binding(
            get: {
                let value = config.nCtx ?? steps[0]
                let index = steps.firstIndex { $0 >= value } ?? (steps.count - 1)
                return Double(index)
            },
            set: { newValue in
                let index = max(0, min(steps.count - 1, Int(newValue.rounded())))
                config.nCtx = steps[index]
            })
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Hairline()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    contextSection
                    memorySection
                    Hairline()
                    slotsSection
                    pinnedSection
                    if model.supportsThinking { effortSection }
                    if isGGUF { gpuSection }
                    cacheSection
                }
                .padding(20)
            }

            Hairline()
            footer
        }
        .frame(width: 560, height: 640)
        .background(Theme.bg)
        .onChange(of: config) { _, _ in scheduleEstimate() }
        .task { scheduleEstimate(immediately: true) }
        .onDisappear { estimateTask?.cancel() }
    }

    // MARK: - Header / footer

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Ustawienia ładowania")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Spacer(minLength: 0)
                FormatBadge(format: model.format)
            }
            Text(model.displayName)
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
                .lineLimit(1)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Button("Przywróć domyślne") {
                config = LoadConfigDTO(nCtx: steps.last(where: { $0 <= 16_384 }) ?? steps[0],
                                       nParallel: 1,
                                       gpuLayers: -1,
                                       flashAttn: "auto",
                                       kvCacheType: "f16")
            }
            .buttonStyle(.link)
            .font(.system(size: 12))

            Spacer(minLength: 0)

            SecondaryButton(title: "Anuluj", height: 34) { dismiss() }

            SecondaryButton(title: "Zapisz", isLoading: isSaving, height: 34) {
                save(thenLoad: false)
            }

            PrimaryButton(title: "Zapisz i załaduj",
                          systemImage: "play.fill",
                          isLoading: isSaving,
                          fullWidth: false,
                          height: 34) {
                save(thenLoad: true)
            }
        }
        .padding(16)
        .background(Theme.surface)
    }

    // MARK: - Sections

    private var contextSection: some View {
        LabeledField("Długość kontekstu",
                     description: "Maksimum tego modelu to \(Format.tokens(model.contextLength ?? steps.last ?? 0)) tokenów. Każdy token kosztuje pamięć w cache KV.") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(Format.tokens(config.nCtx ?? 0))
                        .font(.system(size: 22, weight: .semibold, design: .rounded))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                    Text("tokenów")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 0)
                    if isEstimating {
                        ProgressView().controlSize(.small)
                    }
                }
                Slider(value: contextIndex, in: 0...Double(max(1, steps.count - 1)), step: 1)
                HStack {
                    Text(Format.tokens(steps.first ?? 0))
                    Spacer(minLength: 0)
                    Text(Format.tokens(steps.last ?? 0))
                }
                .font(.system(size: 10))
                .foregroundStyle(Theme.fgSubtle)
            }
        }
    }

    @ViewBuilder
    private var memorySection: some View {
        if let estimate {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 10) {
                    Text(Format.bytes(estimate.totalBytes))
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                    FitBadge(fit: estimate.fits)
                    Spacer(minLength: 0)
                }
                memoryBreakdown(estimate)
                if !estimate.message.isEmpty {
                    Text(estimate.message)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(FitBadge.tint(estimate.fits).opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                    .strokeBorder(FitBadge.tint(estimate.fits).opacity(0.3), lineWidth: 1)
            )
        } else {
            Text("Szacuję zapotrzebowanie na pamięć…")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgSubtle)
        }
    }

    private func memoryBreakdown(_ estimate: MemoryEstimate) -> some View {
        let total = max(Double(estimate.totalBytes), 1)
        let parts: [(String, Int64, Color)] = [
            ("Wagi", estimate.weightsBytes, Theme.accent),
            ("Kontekst", estimate.kvCacheBytes, Theme.color(0x2B6CB0, 0x63B3ED)),
            ("Bufory", estimate.overheadBytes, Theme.fgSubtle),
        ]
        return VStack(alignment: .leading, spacing: 6) {
            GeometryReader { geometry in
                HStack(spacing: 2) {
                    ForEach(parts, id: \.0) { part in
                        Capsule()
                            .fill(part.2)
                            .frame(width: max(2, geometry.size.width * Double(part.1) / total))
                    }
                    Spacer(minLength: 0)
                }
            }
            .frame(height: 6)

            HStack(spacing: 14) {
                ForEach(parts, id: \.0) { part in
                    HStack(spacing: 5) {
                        Circle().fill(part.2).frame(width: 7, height: 7)
                        Text("\(part.0) \(Format.bytes(part.1))")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgMuted)
                    }
                }
                Spacer(minLength: 0)
            }
        }
    }

    private var slotsSection: some View {
        LabeledField("Sloty równoległe",
                     description: "Ile zapytań ta instancja obsłuży naraz. Każdy slot dostaje własną część cache KV, więc pamięć rośnie liniowo.") {
            HStack(spacing: 12) {
                Stepper(value: Binding(get: { config.nParallel ?? 1 },
                                       set: { config.nParallel = $0 }),
                        in: 1...8) {
                    Text("\(config.nParallel ?? 1)")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                        .frame(width: 26, alignment: .leading)
                }
                Spacer(minLength: 0)
            }
        }
    }

    /// Keep this model in memory instead of letting it be unloaded.
    ///
    /// ``runtime.pinnedModels``: the idle timer skips it and it is never evicted
    /// to make room for another model. Worth it for the small assistant that
    /// finds and switches models on request — it is only useful if it answers
    /// straight away — and wasteful for anything large.
    private var pinnedSection: some View {
        LabeledField("Trzymaj w pamięci",
                     description: "Model zostaje wczytany, dopóki tego nie wyłączysz: nie zwolni go licznik bezczynności ani robienie miejsca dla innego modelu. Przydaje się przy małym modelu-asystencie, przy dużym zajmuje pamięć na stałe.") {
            Toggle("Nie zwalniaj tego modelu", isOn: Binding(
                get: { store.isPinned(model.key) },
                set: { on in Task { await store.setPinned(model.key, on) } }))
                .toggleStyle(.switch)
                .font(.system(size: 13))
        }
    }

    /// Reasoning level for this model, independent of the global default.
    ///
    /// Only shown for models whose chat template has a thinking control —
    /// offering it for a model that cannot think would be a dead switch.
    private var effortSection: some View {
        LabeledField("Poziom myślenia",
                     description: "Ile ten model ma myśleć, zanim odpowie. Zastępuje ustawienie ogólne — przydaje się, gdy jeden model myśli świetnie, a inny traci na tym minuty.") {
            Picker("Poziom myślenia", selection: Binding(
                get: { config.effort ?? "" },
                set: { config.effort = $0.isEmpty ? nil : $0 })) {
                Text("Jak w ustawieniach").tag("")
                Text("Bez myślenia").tag("off")
                Text("Niski").tag("low")
                Text("Średni").tag("medium")
                Text("Wysoki").tag("high")
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(maxWidth: 220, alignment: .leading)
        }
    }

    private var gpuSection: some View {
        LabeledField("Warstwy na GPU",
                     description: "Na Apple Silicon wszystko mieści się w pamięci zunifikowanej — zmniejszaj tylko, gdy model nie chce się załadować.") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Wszystkie warstwy na GPU", isOn: Binding(
                    get: { (config.gpuLayers ?? -1) < 0 },
                    set: { config.gpuLayers = $0 ? -1 : 24 }))
                    .toggleStyle(.switch)
                    .font(.system(size: 13))

                if (config.gpuLayers ?? -1) >= 0 {
                    Stepper(value: Binding(get: { config.gpuLayers ?? 0 },
                                           set: { config.gpuLayers = $0 }),
                            in: 0...200, step: 1) {
                        Text("\(config.gpuLayers ?? 0) warstw")
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.fg)
                            .monospacedDigit()
                    }
                }

                Picker("Flash attention", selection: Binding(
                    get: { config.flashAttn ?? "auto" },
                    set: { config.flashAttn = $0 })) {
                    Text("Automatycznie").tag("auto")
                    Text("Włączone").tag("on")
                    Text("Wyłączone").tag("off")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
        }
    }

    private var cacheSection: some View {
        LabeledField("Typ cache KV",
                     description: "Kwantyzacja cache oszczędza pamięć kosztem odrobiny jakości. \(isGGUF ? "q8_0 jest bezpieczne, q4_0 agresywne." : "8bit jest bezpieczne, 4bit agresywne.")") {
            Picker("", selection: Binding(get: { config.kvCacheType ?? "f16" },
                                          set: { config.kvCacheType = $0 })) {
                ForEach(kvOptions, id: \.self) { option in
                    Text(option).tag(option)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    // MARK: - Actions

    private func scheduleEstimate(immediately: Bool = false) {
        estimateTask?.cancel()
        isEstimating = true
        let current = config
        estimateTask = Task {
            if !immediately {
                try? await Task.sleep(nanoseconds: 220_000_000)
                if Task.isCancelled { return }
            }
            let value = await store.estimate(key: model.key, config: current)
            if Task.isCancelled { return }
            estimate = value
            isEstimating = false
        }
    }

    private func save(thenLoad: Bool) {
        isSaving = true
        let payload = config
        Task {
            await store.saveLoadDefaults(key: model.key, config: payload)
            if thenLoad {
                await store.unload(key: model.key)
                await store.load(key: model.key, config: payload)
            }
            isSaving = false
            dismiss()
        }
    }
}
