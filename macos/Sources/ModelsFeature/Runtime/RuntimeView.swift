import SwiftUI
import AppKit
import LocalCodeKit

/// "Runtime" — what is loaded, what it costs, and the rules the loader follows.
///
/// A local model app lives or dies on memory pressure, so this screen puts the
/// two bars that matter at the top: physical RAM as macOS sees it, and the
/// budget LocalCode gives itself, split into one segment per loaded instance.
/// Everything below is the policy that produced those bars.
public struct RuntimeView: View {
    @Environment(ModelsStore.self) private var store

    @State private var confirmation: ConfirmRequest?

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            ScreenHeader(title: "Runtime", subtitle: subtitle) {
                HStack(spacing: 10) {
                    if store.isLive {
                        HStack(spacing: 5) {
                            Circle().fill(Theme.success).frame(width: 6, height: 6)
                            Text("na żywo").font(.system(size: 11)).foregroundStyle(Theme.fgMuted)
                        }
                        .help("Połączono ze strumieniem zdarzeń serwera")
                    }
                    SecondaryButton(title: "Odśwież", systemImage: "arrow.clockwise", height: 32) {
                        Task { await store.refreshRuntime() }
                    }
                    DangerButton(title: "Wyładuj wszystkie",
                                 systemImage: "eject",
                                 isEnabled: !store.runtime.liveInstances.isEmpty,
                                 filled: false,
                                 height: 32) {
                        confirmation = ConfirmRequest(
                            title: "Zwolnić wszystkie modele?",
                            message: "Każda załadowana instancja zostanie zamknięta, a zajmowana przez nią pamięć oddana systemowi.",
                            confirmTitle: "Wyładuj",
                            confirm: { Task { await store.unloadAll() } })
                    }
                }
            }

            StoreMessages()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    MemoryPanel()
                    InstancesPanel()
                    RuntimeSettingsPanel()
                    OpenAIServerPanel()
                }
                .padding(20)
                .frame(maxWidth: 1180, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
        }
        .background(Theme.bg)
        .confirmDialog($confirmation)
        .task { await store.refreshRuntime() }
    }

    private var subtitle: String {
        let instances = store.runtime.liveInstances
        guard !instances.isEmpty else { return "Żaden model nie jest teraz załadowany" }
        let used = instances.reduce(Int64(0)) { $0 + max($1.memoryBytes, $1.estimatedBytes) }
        return "\(Format.plural(instances.count, "instancja", "instancje", "instancji")) · \(Format.bytes(used)) pamięci"
    }
}

// MARK: - Memory

/// The two bars: physical memory, and the budget split per instance.
struct MemoryPanel: View {
    @Environment(ModelsStore.self) private var store

    private var system: SystemMemory { store.runtime.system }
    private var budget: MemoryBudget { store.runtime.budget }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 16) {
                SectionHeader("Pamięć")

                bar(title: "Pamięć fizyczna",
                    total: system.totalBytes,
                    segments: [Segment(id: "used",
                                       label: "Zajęte",
                                       bytes: system.usedBytes,
                                       color: Theme.accent)],
                    trailing: "wolne \(Format.bytes(system.availableBytes)) z \(Format.bytes(system.totalBytes))")

                bar(title: "Budżet modeli",
                    total: budget.limitBytes,
                    segments: instanceSegments,
                    trailing: budget.limitBytes > 0
                        ? "\(Format.bytes(budget.usedBytes)) z \(Format.bytes(budget.limitBytes))"
                        : "—")
            }
        }
    }

    private struct Segment: Identifiable {
        let id: String
        let label: String
        let bytes: Int64
        let color: Color
    }

    private static let palette: [Color] = [
        Theme.accent,
        Theme.color(0x2B6CB0, 0x63B3ED),
        Theme.color(0x2F855A, 0x68D391),
        Theme.color(0x805AD5, 0xB794F4),
        Theme.color(0xB7791F, 0xECC94B),
    ]

    private var instanceSegments: [Segment] {
        store.runtime.liveInstances.enumerated().map { index, instance in
            Segment(id: instance.id,
                    label: instance.displayName,
                    bytes: max(instance.memoryBytes, instance.estimatedBytes),
                    color: Self.palette[index % Self.palette.count])
        }
    }

    private func bar(title: String,
                     total: Int64,
                     segments: [Segment],
                     trailing: String) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text(title)
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Spacer(minLength: 0)
                Text(trailing)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .monospacedDigit()
            }

            GeometryReader { geometry in
                let width = geometry.size.width
                let denominator = Double(max(total, 1))
                HStack(spacing: 1) {
                    ForEach(segments) { segment in
                        segment.color
                            .frame(width: max(0, min(width, width * Double(segment.bytes) / denominator)))
                            .help("\(segment.label) — \(Format.bytes(segment.bytes))")
                    }
                    Spacer(minLength: 0)
                }
                .frame(width: width, alignment: .leading)
                .background(Theme.surface2)
                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
            }
            .frame(height: 12)

            if segments.count > 1 || (segments.count == 1 && segments[0].id != "used") {
                FlowLayout(spacing: 12, lineSpacing: 6) {
                    ForEach(segments) { segment in
                        HStack(spacing: 5) {
                            Circle().fill(segment.color).frame(width: 7, height: 7)
                            Text(segment.label)
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgMuted)
                                .lineLimit(1)
                            Text(Format.bytes(segment.bytes))
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgSubtle)
                        }
                    }
                }
            } else if segments.isEmpty {
                Text("Budżet jest w całości wolny.")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
            }
        }
    }
}

// MARK: - Instances

/// The table of running instances, live from `/api/events`.
struct InstancesPanel: View {
    @Environment(ModelsStore.self) private var store

    var body: some View {
        Card(padding: 0) {
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    SectionHeader("Instancje",
                                  subtitle: "Każda instancja to osobny proces silnika")
                }
                .padding(16)

                if store.runtime.instances.isEmpty {
                    Text("Nic nie jest załadowane. Model ładuje się sam przy pierwszym zapytaniu albo ręcznie w „Moje modele”.")
                        .font(.system(size: 12.5))
                        .foregroundStyle(Theme.fgMuted)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)
                } else {
                    headerRow
                    ForEach(store.runtime.instances) { instance in
                        Hairline()
                        InstanceRow(instance: instance)
                    }
                }
            }
        }
    }

    private var headerRow: some View {
        HStack(spacing: 10) {
            Text("Model").frame(maxWidth: .infinity, alignment: .leading)
            Text("Stan").frame(width: InstanceRow.stateWidth, alignment: .leading)
            Text("Kontekst").frame(width: InstanceRow.numberWidth, alignment: .trailing)
            Text("Sloty").frame(width: InstanceRow.slotWidth, alignment: .trailing)
            Text("Pamięć").frame(width: InstanceRow.numberWidth, alignment: .trailing)
            Text("Tok/s").frame(width: InstanceRow.tpsWidth, alignment: .trailing)
            Text("Załadowany").frame(width: InstanceRow.timeWidth, alignment: .trailing)
            Spacer().frame(width: InstanceRow.actionWidth)
        }
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(Theme.fgSubtle)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Theme.surface2)
    }
}

/// One row of the instances table.
struct InstanceRow: View {
    let instance: RuntimeInstance

    @Environment(ModelsStore.self) private var store

    static let stateWidth: CGFloat = 96
    static let numberWidth: CGFloat = 78
    static let slotWidth: CGFloat = 52
    static let tpsWidth: CGFloat = 56
    static let timeWidth: CGFloat = 96
    static let actionWidth: CGFloat = 92

    private var tint: Color {
        switch instance.state {
        case .ready: Theme.success
        case .busy: Theme.accent
        case .loading, .stopping: Theme.warning
        case .error: Theme.danger
        }
    }

    /// What the instance actually holds: the measured resident size when the
    /// server has one, otherwise the estimate it reserved the budget with.
    /// A measured value far below the estimate means macOS has paged weights
    /// out, not that the model shrank — so the reservation is the honest number.
    private var reservedBytes: Int64 {
        max(instance.memoryBytes, instance.estimatedBytes)
    }

    private var memoryHelp: String {
        guard instance.memoryBytes > 0, instance.memoryBytes != instance.estimatedBytes else {
            return "Zarezerwowana pamięć"
        }
        return "Zarezerwowane \(Format.bytes(instance.estimatedBytes)) · zmierzone teraz \(Format.bytes(instance.memoryBytes))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                HStack(spacing: 7) {
                    FormatBadge(format: instance.format)
                    Text(instance.displayName)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                HStack(spacing: 5) {
                    if instance.state == .loading {
                        ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 10)
                    } else {
                        Circle().fill(tint).frame(width: 7, height: 7)
                    }
                    Text(instance.state.label)
                        .font(.system(size: 11.5, weight: .semibold))
                        .foregroundStyle(tint)
                }
                .frame(width: Self.stateWidth, alignment: .leading)

                Text(instance.nCtx > 0 ? Format.tokens(instance.nCtx) : "—")
                    .frame(width: Self.numberWidth, alignment: .trailing)

                Text("\(instance.busySlots)/\(instance.nParallel)")
                    .frame(width: Self.slotWidth, alignment: .trailing)

                Text(reservedBytes > 0 ? Format.bytes(reservedBytes) : "—")
                    .frame(width: Self.numberWidth, alignment: .trailing)
                    .help(memoryHelp)

                Text(ModelFormat.tps(instance.genTps) ?? "—")
                    .frame(width: Self.tpsWidth, alignment: .trailing)

                Text(instance.loadedAt.map { Format.relative($0) } ?? "—")
                    .frame(width: Self.timeWidth, alignment: .trailing)

                SecondaryButton(title: instance.state == .error ? "Ukryj" : "Wyładuj", height: 26) {
                    Task { await store.unload(instanceId: instance.id) }
                }
                .frame(width: Self.actionWidth, alignment: .trailing)
            }
            .font(.system(size: 12))
            .foregroundStyle(Theme.fgMuted)
            .monospacedDigit()

            if let error = instance.error, !error.isEmpty {
                Text(error)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.danger)
                    .fixedSize(horizontal: false, vertical: true)
            } else if let pid = instance.pid, let port = instance.port {
                Text("proces \(pid) · port \(port)")
                    .font(.system(size: 10.5))
                    .foregroundStyle(Theme.fgSubtle)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }
}

// MARK: - OpenAI-compatible server

/// The local OpenAI-compatible endpoint, the way LM Studio presents it.
struct OpenAIServerPanel: View {
    @Environment(ModelsStore.self) private var store

    @State private var didCopy = false

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader("Serwer zgodny z OpenAI",
                              subtitle: "Inne aplikacje mogą używać załadowanych modeli przez to API. Aplikacja wybiera port przy każdym starcie — stały adres daje polecenie „localcode serve” uruchomione w terminalu przed otwarciem aplikacji (opis: docs/API.md).")

                HStack(spacing: 10) {
                    Text(store.openAIBaseURL)
                        .font(.system(size: 13, design: .monospaced))
                        .foregroundStyle(Theme.fg)
                        .textSelection(.enabled)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Theme.codeBg)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1)
                        )

                    SecondaryButton(title: didCopy ? "Skopiowano" : "Kopiuj",
                                    systemImage: didCopy ? "checkmark" : "doc.on.doc",
                                    height: 32) {
                        Pasteboard.copy(store.openAIBaseURL)
                        didCopy = true
                        Task {
                            try? await Task.sleep(nanoseconds: 1_500_000_000)
                            didCopy = false
                        }
                    }

                    Spacer(minLength: 0)

                    Toggle("Włączony", isOn: Binding(
                        get: { store.settings.server.openaiApiEnabled },
                        set: { value in Task { await store.setOpenAIServer(enabled: value) } }))
                        .toggleStyle(.switch)
                        .font(.system(size: 13))
                }

                Text(store.settings.server.openaiApiRequiresToken
                     ? "Klienci muszą podać token dostępu LocalCode jako klucz API."
                     : "Bez klucza API — endpoint słucha tylko na 127.0.0.1.")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
        }
    }
}
