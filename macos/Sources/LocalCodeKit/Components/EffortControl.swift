import SwiftUI

/// The "Myślenie" control and the level arithmetic behind it.
///
/// A model either has no reasoning switch at all (`effortLevels == []`, and the
/// control hides itself), or it has one of exactly two shapes:
///
///  * `["off", "on"]` — a boolean thinking switch (Qwen3, GLM, DeepSeek V3.1,
///    Granite): thinking can be turned off, but not dialled down.
///  * `["low", "medium", "high"]` — a real effort dial (gpt-oss), which cannot
///    be turned off at all.
///
/// The server stores a *canonical* level, always one of `off|low|medium|high`
/// (`EFFORT_CHOICES` in `runtime/templates.py`), and resolves it onto whatever
/// the model's chat template can express when it builds the prompt. `"on"` is a
/// resolved level, never a request: the server normalises an incoming `"on"`
/// to `medium` rather than storing it. So the control shows the model's own
/// levels and translates in both directions — `Włączone` is sent as `medium`,
/// and a stored `medium` on a boolean model reads back as `Włączone`.
public enum EffortLevel: String, CaseIterable, Identifiable, Hashable, Sendable {
    case off
    case on
    case low
    case medium
    case high

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .off: "Wyłączone"
        case .on: "Włączone"
        case .low: "Niskie"
        case .medium: "Średnie"
        case .high: "Wysokie"
        }
    }

    /// Strength, matching `EFFORT_ORDER` on the Python side. `on` is the single
    /// enabled level of a boolean template, so it ranks with `high`.
    public var order: Int {
        switch self {
        case .off: 0
        case .low: 1
        case .medium: 2
        case .high, .on: 3
        }
    }

    /// What goes on the wire. The server only accepts `off|low|medium|high`.
    public var wireValue: String { self == .on ? "medium" : rawValue }

    /// The levels a model reports, in the order the server listed them,
    /// ignoring anything this build does not know.
    public static func levels(_ raw: [String]) -> [EffortLevel] {
        raw.compactMap { EffortLevel(rawValue: $0.lowercased()) }
    }

    /// Which segment a stored canonical level lights up.
    ///
    /// Mirrors `resolve_effort` in `runtime/templates.py`: an exact match wins,
    /// `off` falls to the weakest level the model has, and anything else picks
    /// the closest level that is not "off".
    public static func resolve(_ stored: String?, in levels: [EffortLevel]) -> EffortLevel? {
        guard !levels.isEmpty else { return nil }
        guard let stored, let wanted = EffortLevel(rawValue: stored.lowercased()) else { return nil }
        if levels.contains(wanted) { return wanted }
        if wanted == .off { return levels[0] }
        let enabled = levels.filter { $0 != .off }
        let candidates = enabled.isEmpty ? levels : enabled
        return candidates.min { lhs, rhs in
            (abs(lhs.order - wanted.order), lhs.order) < (abs(rhs.order - wanted.order), rhs.order)
        }
    }
}

/// The tooltip every "Myślenie" control carries, in one place.
public enum EffortHelp {
    public static let text = """
    Myślenie: model najpierw rozumuje, a dopiero potem odpowiada.
    Więcej myślenia = lepsze odpowiedzi, ale wolniej i więcej zużytego kontekstu.
    """

    /// The hint on the agent's "Ultracode" segment.
    public static let ultracode = """
    Najwyższy poziom: agent rozdziela pracę na wielu pomocników równolegle i sprawdza wynik. \
    Wolniej i drożej (przy modelach zdalnych każdy pomocnik to płatne zapytania).
    """
}

/// The coding agent's level above "high" (`ULTRACODE` in `runtime/templates.py`).
///
/// Not an `EffortLevel`: no chat template has it, and the chat never offers
/// it. The model thinks at "high"; what changes is how the agent works — it
/// fans independent work out to helpers running side by side and has the
/// result reviewed. Only the agent's controls show it (`offersUltracode`).
public enum Ultracode {
    public static let wireValue = "ultracode"
    public static let label = "Ultracode"

    public static func isSelected(_ stored: String?) -> Bool {
        stored?.lowercased() == wireValue
    }
}

/// Segmented "Myślenie" control. Renders nothing when the model has no levels.
///
/// `selection` is the canonical level the server stores (`off|low|medium|high`,
/// `ultracode` for the agent, or `nil` for "use the global default");
/// `onChange` is handed the value to send, never `"on"`.
///
/// `offersUltracode` (the agent's controls only) adds an "Ultracode" segment
/// after the model's own levels. It is shown even for a model without a
/// thinking switch — a remote one, say — because it is about how the agent
/// works, not about the template; the model's place is then a single
/// "Zwykły" segment that sends `medium`.
public struct EffortPicker: View {
    public let levels: [EffortLevel]
    public let selection: String?
    public var showsLabel: Bool
    public var offersUltracode: Bool
    public let onChange: (String) -> Void

    public init(levels: [EffortLevel],
                selection: String?,
                showsLabel: Bool = true,
                offersUltracode: Bool = false,
                onChange: @escaping (String) -> Void) {
        self.levels = levels
        self.selection = selection
        self.showsLabel = showsLabel
        self.offersUltracode = offersUltracode
        self.onChange = onChange
    }

    /// Convenience for the raw `effortLevels` a DTO carries.
    public init(rawLevels: [String],
                selection: String?,
                showsLabel: Bool = true,
                offersUltracode: Bool = false,
                onChange: @escaping (String) -> Void) {
        self.init(levels: EffortLevel.levels(rawLevels),
                  selection: selection,
                  showsLabel: showsLabel,
                  offersUltracode: offersUltracode,
                  onChange: onChange)
    }

    private var ultracodeOn: Bool { offersUltracode && Ultracode.isSelected(selection) }

    private var active: EffortLevel? {
        ultracodeOn ? nil : EffortLevel.resolve(selection, in: levels)
    }

    /// The model's own segments: its levels, or nothing when it has no switch.
    private var dial: [EffortLevel] { levels.count > 1 ? levels : [] }

    public var body: some View {
        if levels.count > 1 || offersUltracode {
            HStack(spacing: 7) {
                if showsLabel {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                            .font(.system(size: 10, weight: .semibold))
                        Text("Myślenie")
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(Theme.fgMuted)
                }

                HStack(spacing: 2) {
                    ForEach(dial) { level in
                        segment(level)
                    }
                    if offersUltracode {
                        if dial.isEmpty {
                            plainSegment
                        }
                        ultracodeSegment
                    }
                }
                .padding(2)
                .fixedSize()                    // a crowded header must not wrap "Ultracode" letter by letter
                .background(Theme.surface2)
                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
            }
            .help(EffortHelp.text)
            .accessibilityElement(children: .contain)
            .accessibilityLabel(Text("Poziom myślenia"))
        }
    }

    private func segment(_ level: EffortLevel) -> some View {
        let isActive = level == active
        return Button {
            onChange(level.wireValue)
        } label: {
            Text(level.label)
                .font(.system(size: 12, weight: isActive ? .semibold : .regular))
                .foregroundStyle(isActive ? Theme.fg : Theme.fgMuted)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(isActive ? Theme.surface : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .accessibilityAddTraits(isActive ? [.isButton, .isSelected] : .isButton)
    }

    /// "Zwykły": the place of a model without a thinking switch, next to Ultracode.
    private var plainSegment: some View {
        let isActive = !ultracodeOn
        return Button {
            onChange(EffortLevel.medium.wireValue)
        } label: {
            Text("Zwykły")
                .lineLimit(1)
                .font(.system(size: 12, weight: isActive ? .semibold : .regular))
                .foregroundStyle(isActive ? Theme.fg : Theme.fgMuted)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(isActive ? Theme.surface : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .help("Agent pracuje sam, bez pomocników uruchamianych równolegle.")
        .accessibilityAddTraits(isActive ? [.isButton, .isSelected] : .isButton)
    }

    private var ultracodeSegment: some View {
        let isActive = ultracodeOn
        return Button {
            onChange(Ultracode.wireValue)
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "person.3.fill")
                    .font(.system(size: 9, weight: .semibold))
                Text(Ultracode.label)
                    .lineLimit(1)
                    .font(.system(size: 12, weight: isActive ? .semibold : .regular))
            }
            .foregroundStyle(isActive ? Theme.accent : Theme.fgMuted)
            .padding(.horizontal, 9)
            .padding(.vertical, 4)
            .background(isActive ? Theme.surface : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .help(EffortHelp.ultracode)
        .accessibilityLabel(Text(Ultracode.label))
        .accessibilityHint(Text(EffortHelp.ultracode))
        .accessibilityAddTraits(isActive ? [.isButton, .isSelected] : .isButton)
    }
}
