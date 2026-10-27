import SwiftUI
import LocalCodeKit

/// The small pieces all three model screens share: the page header, the badges
/// that describe a model at a glance, and the number formatting the rows use.
///
/// They live here rather than in LocalCodeKit because they are about *models* —
/// a quantisation tag, a "will it fit" verdict, a tools/vision capability icon —
/// and nothing in the chat or the agent needs them.

// MARK: - Page chrome

/// Title bar of a full section: heading, one line of context, and controls.
struct ScreenHeader<Trailing: View>: View {
    let title: String
    var subtitle: String?
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            trailing
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
    }
}

extension ScreenHeader where Trailing == EmptyView {
    init(title: String, subtitle: String? = nil) {
        self.init(title: title, subtitle: subtitle) { EmptyView() }
    }
}

/// The error banner + the transient confirmation, in the place every model
/// screen puts them: directly under the header, above the content.
struct StoreMessages: View {
    @Environment(ModelsStore.self) private var store

    var body: some View {
        @Bindable var store = store
        VStack(spacing: 8) {
            if let message = store.errorMessage {
                Banner(.error, message: message, onDismiss: { store.errorMessage = nil })
            }
            if let note = store.statusMessage {
                Banner(.success, message: note)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, store.errorMessage == nil && store.statusMessage == nil ? 0 : 12)
        .animation(.easeOut(duration: 0.18), value: store.errorMessage)
        .animation(.easeOut(duration: 0.18), value: store.statusMessage)
    }
}

// MARK: - Badges

/// `GGUF` / `MLX`, coloured so the two runtimes are telling apart at a glance.
struct FormatBadge: View {
    let format: String

    private var tint: Color {
        format.lowercased() == "mlx"
            ? Theme.color(0x2B6CB0, 0x63B3ED)
            : Theme.color(0x805AD5, 0xB794F4)
    }

    var body: some View {
        Text(format.uppercased())
            .font(.system(size: 10, weight: .bold))
            .kerning(0.4)
            .foregroundStyle(tint)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

/// Neutral tag: a quantisation, an architecture, a context length.
struct MetaBadge: View {
    let text: String
    var systemImage: String?
    var tint: Color = Theme.fgMuted

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage).font(.system(size: 9, weight: .semibold))
            }
            Text(text).font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Theme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}

/// "Zmieści się" / "Na styk" / "Za duży" with the matching colour.
struct FitBadge: View {
    let fit: MemoryEstimate.Fit
    var compact = false

    static func tint(_ fit: MemoryEstimate.Fit) -> Color {
        switch fit {
        case .yes: Theme.success
        case .tight: Theme.warning
        case .no: Theme.danger
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: fit.systemImage)
                .font(.system(size: compact ? 9 : 10, weight: .semibold))
            if !compact {
                Text(fit.label).font(.system(size: 11, weight: .semibold))
            }
        }
        .foregroundStyle(Self.tint(fit))
        .padding(.horizontal, compact ? 4 : 7)
        .padding(.vertical, 2)
        .background(Self.tint(fit).opacity(0.14))
        .clipShape(Capsule())
    }
}

/// Tools / thinking / vision / embedding, as icons with tooltips.
struct CapabilityIcons: View {
    let capabilities: [String]
    var hasVision = false
    var size: CGFloat = 11

    private struct Item: Identifiable {
        let id: String
        let symbol: String
        let help: String
    }

    private var items: [Item] {
        var out: [Item] = []
        if capabilities.contains("tools") {
            out.append(Item(id: "tools", symbol: "wrench.and.screwdriver", help: "Wywoływanie narzędzi"))
        }
        if capabilities.contains("thinking") {
            out.append(Item(id: "thinking", symbol: "brain", help: "Tryb myślenia"))
        }
        if hasVision || capabilities.contains("vision") {
            out.append(Item(id: "vision", symbol: "eye", help: "Rozumie obrazy"))
        }
        if capabilities.contains("embedding") {
            out.append(Item(id: "embedding", symbol: "point.3.connected.trianglepath.dotted",
                            help: "Model embeddingów"))
        }
        if capabilities.contains("stt") {
            out.append(Item(id: "stt", symbol: "waveform", help: "Rozpoznawanie mowy"))
        }
        if capabilities.contains("tts") {
            out.append(Item(id: "tts", symbol: "speaker.wave.2", help: "Synteza mowy"))
        }
        return out
    }

    var body: some View {
        HStack(spacing: 6) {
            ForEach(items) { item in
                Image(systemName: item.symbol)
                    .font(.system(size: size, weight: .medium))
                    .foregroundStyle(Theme.fgMuted)
                    .help(item.help)
                    .accessibilityLabel(Text(item.help))
            }
        }
    }
}

// MARK: - Formatting

/// Number and label formatting that only the model screens need.
enum ModelFormat {

    /// `4,0 B` parameters — the unit every model card in the wild uses.
    static func params(_ value: Double?) -> String? {
        guard let value, value > 0 else { return nil }
        if value < 1 { return String(format: "%.0f M", value * 1000) }
        return String(format: "%.1f B", value).replacingOccurrences(of: ".", with: ",")
    }

    /// Download or like counters: `171 tys.`, `1,4 mln`.
    static func count(_ value: Int?) -> String? {
        guard let value, value > 0 else { return nil }
        if value >= 1_000_000 {
            return String(format: "%.1f mln", Double(value) / 1_000_000)
                .replacingOccurrences(of: ".", with: ",")
        }
        if value >= 1_000 {
            return String(format: "%.0f tys.", Double(value) / 1_000)
        }
        return "\(value)"
    }

    /// Context window in the `k` unit the whole app speaks.
    static func context(_ value: Int?) -> String? {
        guard let value, value > 0 else { return nil }
        return Format.tokens(value)
    }

    /// Generation speed with a Polish decimal comma: `31,5`.
    static func tps(_ value: Double?) -> String? {
        guard let value, value > 0 else { return nil }
        return String(format: "%.1f", value).replacingOccurrences(of: ".", with: ",")
    }

    /// Whole gigabytes with the Polish unit: `24 GB`.
    static func wholeGigabytes(_ value: Double) -> String {
        "\(Int(value.rounded())) GB"
    }

    /// Download speed: `29,4 MB/s`.
    static func speed(_ bytesPerSecond: Double) -> String {
        guard bytesPerSecond > 1 else { return "—" }
        return "\(Format.bytes(Int64(bytesPerSecond)))/s"
    }

    /// ETA in the download row.
    static func eta(_ seconds: Double?) -> String? {
        guard let seconds, seconds > 0, seconds.isFinite else { return nil }
        return "pozostało \(Format.duration(seconds))"
    }

    /// Idle unload menu labels.
    static func idle(_ minutes: Int) -> String {
        minutes <= 0 ? "Nigdy" : "\(minutes) min"
    }
}

// MARK: - Quick fit

/// A local, approximate answer to "will this model fit right now".
///
/// The authoritative number comes from `POST /api/runtime/estimate`, which
/// reads the model's real head/layer geometry — but the model picker shows
/// thirty rows at once and cannot ask the server thirty times per keystroke.
/// This is the cheap version: the weights on disk plus a flat allowance for a
/// modest context and the engine's own buffers, against whichever is smaller,
/// the remaining RAM budget or the memory macOS says is free.
enum QuickFit {
    /// Engine buffers, tokenizer, and a small default context.
    private static let allowance: Int64 = 640 * 1024 * 1024

    static func verdict(for model: LocalModelDTO, runtime: RuntimeStatus) -> MemoryEstimate.Fit {
        let needed = Int64(Double(model.sizeBytes) * 1.08) + allowance
        let budgetLeft = max(0, runtime.budget.limitBytes - runtime.budget.usedBytes)
        let free = runtime.budget.limitBytes > 0
            ? min(budgetLeft, runtime.system.availableBytes + runtime.budget.usedBytes)
            : runtime.system.availableBytes
        guard free > 0 else { return .tight }
        if needed <= Int64(Double(free) * 0.75) { return .yes }
        if needed <= free { return .tight }
        return .no
    }
}

// MARK: - Row chrome

/// Selection / hover chrome for a list row, matching the chat sidebar.
struct RowChrome: ViewModifier {
    var isActive: Bool
    var isHovered: Bool

    func body(content: Content) -> some View {
        content
            .background(isActive ? Theme.accentSoft : (isHovered ? Theme.surface2 : Color.clear))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(isActive ? Theme.accent.opacity(0.35) : Color.clear, lineWidth: 1)
            )
    }
}

extension View {
    func rowChrome(isActive: Bool, isHovered: Bool = false) -> some View {
        modifier(RowChrome(isActive: isActive, isHovered: isHovered))
    }
}

/// A `Picker` in the segmented style the browser controls use.
struct SegmentedControl<T: Hashable & Identifiable>: View {
    let items: [T]
    @Binding var selection: T
    let title: (T) -> String

    var body: some View {
        HStack(spacing: 2) {
            ForEach(items) { item in
                let isActive = item == selection
                Button {
                    selection = item
                } label: {
                    Text(title(item))
                        .font(.system(size: 12, weight: isActive ? .semibold : .regular))
                        .foregroundStyle(isActive ? Theme.fg : Theme.fgMuted)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(isActive ? Theme.surface : Color.clear)
                        .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                        .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle())
            }
        }
        .padding(2)
        .background(Theme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}
