import SwiftUI
import AppKit

/// The palette and the formatting helpers every LocalCode view uses.
///
/// The colours come straight from the web chat app this UI was ported from, so
/// the chatbot half of LocalCode reads exactly like the product it grew out of,
/// and the model/runtime half inherits the same surfaces instead of inventing a
/// second look. Each colour is a light/dark pair resolved by AppKit at draw
/// time, which is what makes a live appearance switch work without any state.
public enum Theme {

    /// Builds a dynamic colour from a light and a dark sRGB hex value.
    public static func color(_ light: UInt32, _ dark: UInt32) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            return NSColor(localCodeRGB: isDark ? dark : light)
        })
    }

    // MARK: Surfaces

    public static let bg           = color(0xFAF9F7, 0x1B1A18)
    public static let surface      = color(0xFFFFFF, 0x232220)
    public static let surface2     = color(0xF2F0EC, 0x2B2A27)
    public static let sidebar      = color(0xF5F3EF, 0x191817)
    public static let rail         = color(0xEFEDE8, 0x141312)
    public static let border       = color(0xE6E2DC, 0x35332F)
    public static let borderStrong = color(0xD6D1C8, 0x45423D)

    // MARK: Text

    public static let fg        = color(0x1F1E1C, 0xEEECE7)
    public static let fgMuted   = color(0x6B6862, 0xA7A29A)
    public static let fgSubtle  = color(0x97938C, 0x7C776F)

    // MARK: Accents

    public static let accent    = color(0xC96442, 0xD97757)
    public static let accentFg  = color(0xFFFFFF, 0x1B1A18)
    public static let danger    = color(0xC0392B, 0xE5675A)
    public static let success   = color(0x2F855A, 0x68D391)
    public static let warning   = color(0xB7791F, 0xECC94B)
    public static let codeBg    = color(0xF6F4F0, 0x1A1917)
    public static let bubbleUser = color(0xF0EDE7, 0x2E2C29)

    public static let accentSoft  = accent.opacity(0.12)
    public static let dangerSoft  = danger.opacity(0.12)
    public static let successSoft = success.opacity(0.12)
    public static let warningSoft = warning.opacity(0.14)

    // MARK: Metrics

    public static let cornerRadius: CGFloat = 14
    public static let bubbleRadius: CGFloat = 18
}

extension NSColor {
    /// `0xRRGGBB` in sRGB — the same notation the web app's CSS uses.
    convenience init(localCodeRGB rgb: UInt32) {
        self.init(srgbRed: Double((rgb >> 16) & 0xFF) / 255,
                  green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255,
                  alpha: 1)
    }
}

/// Formatting shared by every view. All user-facing output is Polish.
public enum Format {

    /// Relative time from a millisecond timestamp (the API speaks milliseconds).
    public static func relative(_ millis: Double) -> String {
        let date = Date(timeIntervalSince1970: millis / 1000)
        if Date().timeIntervalSince(date) < 60 { return "przed chwilą" }
        let formatter = RelativeDateTimeFormatter()
        formatter.locale = Locale(identifier: "pl_PL")
        formatter.unitsStyle = .short
        return formatter.localizedString(for: date, relativeTo: Date())
    }

    public static func time(_ millis: Double) -> String {
        let date = Date(timeIntervalSince1970: millis / 1000)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "pl_PL")
        formatter.timeStyle = .short
        formatter.dateStyle = .none
        return formatter.string(from: date)
    }

    public static func dateTime(_ millis: Double) -> String {
        let date = Date(timeIntervalSince1970: millis / 1000)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "pl_PL")
        formatter.timeStyle = .short
        formatter.dateStyle = .short
        return formatter.string(from: date)
    }

    public static func bytes(_ value: Int) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file)
    }

    public static func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }

    /// Gigabytes with one decimal — the unit the runtime panel thinks in.
    public static func gigabytes(_ value: Int64) -> String {
        String(format: "%.1f GB", Double(value) / 1_073_741_824)
    }

    /// Cost in dollars. Local models are free, so `0` renders as a dash and the
    /// callers simply leave the label out.
    public static func cost(_ usd: Double?) -> String {
        guard let usd, usd > 0 else { return "—" }
        return usd < 0.01 ? String(format: "$%.4f", usd) : String(format: "$%.3f", usd)
    }

    /// Compact token count: `1.2k`, `3.4M`.
    public static func compact(_ value: Int) -> String {
        if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
        if value >= 1_000 { return String(format: "%.1fk", Double(value) / 1_000) }
        return "\(value)"
    }

    /// Context window sizes are always spoken about in `k`.
    public static func tokens(_ value: Int) -> String {
        value >= 1000 ? "\(value / 1000)k" : "\(value)"
    }

    public static func duration(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0f s", seconds) }
        let minutes = Int(seconds) / 60
        let rest = Int(seconds) % 60
        if minutes < 60 { return "\(minutes) min \(rest) s" }
        return "\(minutes / 60) h \(minutes % 60) min"
    }

    /// Polish plural form: `1 wynik`, `2 wyniki`, `5 wyników`.
    public static func plural(_ count: Int, _ one: String, _ few: String, _ many: String) -> String {
        if count == 1 { return "\(count) \(one)" }
        let mod10 = count % 10
        let mod100 = count % 100
        if (2...4).contains(mod10) && !(12...14).contains(mod100) { return "\(count) \(few)" }
        return "\(count) \(many)"
    }

    /// Groups conversations the way the sidebar on the web does.
    public static func bucket(_ millis: Double) -> String {
        let date = Date(timeIntervalSince1970: millis / 1000)
        let calendar = Calendar.current
        if calendar.isDateInToday(date) { return "Dzisiaj" }
        if calendar.isDateInYesterday(date) { return "Wczoraj" }
        let days = calendar.dateComponents([.day], from: date, to: Date()).day ?? 0
        if days <= 7 { return "Ostatnie 7 dni" }
        if days <= 30 { return "Ostatnie 30 dni" }
        return "Starsze"
    }

    public static let bucketOrder = ["Dzisiaj", "Wczoraj", "Ostatnie 7 dni", "Ostatnie 30 dni", "Starsze"]

    /// Parses a `#rrggbb` (or `#rgb`) string; `nil` for anything else so the
    /// caller can fall back to a theme colour.
    public static func hexColor(_ hex: String?) -> Color? {
        guard var value = hex?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty else {
            return nil
        }
        if value.hasPrefix("#") { value.removeFirst() }
        if value.count == 3 { value = value.map { "\($0)\($0)" }.joined() }
        guard value.count == 6, let number = UInt32(value, radix: 16) else { return nil }
        return Color(.sRGB,
                     red: Double((number >> 16) & 0xFF) / 255,
                     green: Double((number >> 8) & 0xFF) / 255,
                     blue: Double(number & 0xFF) / 255,
                     opacity: 1)
    }

    /// An SF Symbol name, or a fallback when the string is not a real symbol —
    /// server-provided icon names would otherwise render as empty boxes.
    public static func symbol(_ name: String?, fallback: String = "folder.fill") -> String {
        guard let name, !name.isEmpty else { return fallback }
        return NSImage(systemSymbolName: name, accessibilityDescription: nil) != nil ? name : fallback
    }
}
