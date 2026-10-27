import SwiftUI
import LocalCodeKit

/// The rules the loader follows: how much memory it may take, how many models
/// it keeps warm, and what it does when the next one does not fit.
///
/// Every control writes straight through `PATCH /api/runtime/settings` and
/// adopts the `RuntimeStatus` that comes back, so the panel can never drift
/// from the server. The RAM slider is the one exception: it would fire a
/// request per pixel, so it commits shortly after the drag stops.
struct RuntimeSettingsPanel: View {
    @Environment(ModelsStore.self) private var store

    @State private var ramDraft: Double = 0
    @State private var isEditingRam = false
    @State private var ramTask: Task<Void, Never>?

    private var settings: RuntimeSettings { store.runtime.settings }

    /// Physical memory in GB — the ceiling the server itself enforces.
    private var physicalGb: Double {
        let total = store.runtime.system.totalBytes
        return total > 0 ? Double(total) / 1_073_741_824 : 32
    }

    private var isAutoRam: Bool { settings.ramLimitGb == nil }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader("Zasady ładowania",
                              subtitle: "Zmiany zapisują się od razu i obowiązują przy następnym ładowaniu")

                ramLimit

                Hairline()

                HStack(alignment: .top, spacing: 24) {
                    counter(title: "Maks. załadowanych modeli",
                            description: "Ile różnych modeli może być jednocześnie w pamięci.",
                            value: settings.maxLoadedModels,
                            range: 1...8,
                            key: "maxLoadedModels")

                    counter(title: "Instancje jednego modelu",
                            description: "Kopie tego samego modelu — każda ma własny cache KV.",
                            value: settings.maxInstancesPerModel,
                            range: 1...4,
                            key: "maxInstancesPerModel")
                }

                HStack(alignment: .top, spacing: 24) {
                    counter(title: "Równoległe zapytania na instancję",
                            description: "Domyślna liczba slotów przy ładowaniu modelu.",
                            value: settings.defaultParallel,
                            range: 1...8,
                            key: "defaultParallel")

                    idleTtl
                }

                Hairline()

                HStack(alignment: .top, spacing: 24) {
                    defaultContext
                    kvCache
                }

                HStack(alignment: .top, spacing: 24) {
                    flashAttention
                    evictPolicy
                }
            }
        }
        .onAppear { syncRam() }
        .onChange(of: settings.ramLimitGb) { _, _ in
            if !isEditingRam { syncRam() }
        }
        .onDisappear { ramTask?.cancel() }
    }

    // MARK: - RAM budget

    private var ramLimit: some View {
        LabeledField("Limit RAM dla modeli",
                     description: "Ile pamięci LocalCode może zająć modelami. „Auto” to 70 % pamięci fizycznej.") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Toggle("Auto", isOn: Binding(
                        get: { isAutoRam },
                        set: { auto in
                            ramTask?.cancel()
                            let patch: JSONObject = auto
                                ? ["ramLimitGb": .null]
                                : ["ramLimitGb": .double((ramDraft > 0 ? ramDraft : physicalGb * 0.7).rounded())]
                            Task { await store.patchRuntime(patch) }
                        }))
                        .toggleStyle(.switch)
                        .font(.system(size: 13))

                    Text(isAutoRam
                         ? "\(Format.bytes(store.runtime.budget.limitBytes)) (automatycznie)"
                         : String(format: "%.0f GB", ramDraft))
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()

                    Spacer(minLength: 0)
                }

                if !isAutoRam {
                    Slider(value: Binding(get: { ramDraft }, set: { value in
                        ramDraft = value.rounded()
                        isEditingRam = true
                        commitRam()
                    }), in: 1...max(2, physicalGb.rounded()), step: 1)

                    HStack {
                        Text("1 GB")
                        Spacer(minLength: 0)
                        Text(String(format: "%.0f GB", physicalGb.rounded()))
                    }
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.fgSubtle)
                }
            }
        }
    }

    private func syncRam() {
        ramDraft = settings.ramLimitGb ?? (physicalGb * 0.7).rounded()
    }

    private func commitRam() {
        ramTask?.cancel()
        let value = ramDraft
        ramTask = Task {
            try? await Task.sleep(nanoseconds: 400_000_000)
            if Task.isCancelled { return }
            await store.patchRuntime(["ramLimitGb": .double(value)])
            isEditingRam = false
        }
    }

    // MARK: - Simple counters

    private func counter(title: String,
                         description: String,
                         value: Int,
                         range: ClosedRange<Int>,
                         key: String) -> some View {
        LabeledField(title, description: description) {
            HStack(spacing: 10) {
                Stepper(value: Binding(get: { value },
                                       set: { newValue in
                                           Task { await store.patchRuntime([key: .int(newValue)]) }
                                       }),
                        in: range) {
                    Text("\(value)")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .monospacedDigit()
                        .frame(width: 24, alignment: .leading)
                }
                Spacer(minLength: 0)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Menus

    private static let idleOptions = [0, 5, 15, 30, 60]

    private var idleTtl: some View {
        LabeledField("Automatyczne wyładowanie po bezczynności",
                     description: "Model bez zapytań przez ten czas zwalnia pamięć sam.") {
            Picker("", selection: Binding(
                get: { Self.idleOptions.contains(settings.idleTtlMinutes) ? settings.idleTtlMinutes : 30 },
                set: { value in Task { await store.patchRuntime(["idleTtlMinutes": .int(value)]) } })) {
                ForEach(Self.idleOptions, id: \.self) { minutes in
                    Text(minutes == 0 ? "Nigdy" : "\(minutes) min").tag(minutes)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 160, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private static let ctxOptions = [4096, 8192, 16_384, 32_768, 65_536, 131_072]

    private var defaultContext: some View {
        LabeledField("Domyślny kontekst",
                     description: "Używany, gdy model nie ma własnych ustawień ładowania.") {
            Picker("", selection: Binding(
                get: { settings.defaultCtx ?? 0 },
                set: { value in
                    Task { await store.patchRuntime(["defaultCtx": value == 0 ? .null : .int(value)]) }
                })) {
                Text("Auto").tag(0)
                ForEach(Self.ctxOptions, id: \.self) { value in
                    Text(Format.tokens(value)).tag(value)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 160, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private static let kvOptions = ["f16", "bf16", "q8_0", "q4_0", "8bit", "4bit"]

    private var kvCache: some View {
        LabeledField("Typ cache KV",
                     description: "GGUF: f16/q8_0/q4_0. MLX: f16/8bit/4bit.") {
            Picker("", selection: Binding(
                get: { settings.kvCacheType },
                set: { value in Task { await store.patchRuntime(["kvCacheType": .string(value)]) } })) {
                ForEach(Self.kvOptions, id: \.self) { option in
                    Text(option).tag(option)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 160, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var flashAttention: some View {
        LabeledField("Flash attention",
                     description: "Szybsza uwaga w llama.cpp; „Automatycznie” zostawia decyzję silnikowi.") {
            Picker("", selection: Binding(
                get: { settings.flashAttn },
                set: { value in Task { await store.patchRuntime(["flashAttn": .string(value)]) } })) {
                Text("Automatycznie").tag("auto")
                Text("Włączone").tag("on")
                Text("Wyłączone").tag("off")
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 260, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var evictPolicy: some View {
        LabeledField("Gdy brakuje pamięci",
                     description: "Co zrobić, gdy nowy model nie mieści się w budżecie.") {
            Picker("", selection: Binding(
                get: { settings.evictPolicy },
                set: { value in Task { await store.patchRuntime(["evictPolicy": .string(value)]) } })) {
                Text("Zwolnij najdawniej używany").tag("lru")
                Text("Odmów załadowania").tag("none")
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 260, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
