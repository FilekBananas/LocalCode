import SwiftUI

/// The shared control vocabulary: buttons, text fields, chips, cards and the
/// small layout helpers every screen is built from.
///
/// Feature modules never style a button by hand — they pick one of these, which
/// is what keeps the chat, the model browser and the agent looking like one
/// application even though three people build them in parallel.

// MARK: - Button chrome

/// Uniform press feedback for every button in the app.
public struct PressableButtonStyle: ButtonStyle {
    public var pressedOpacity: Double

    public init(pressedOpacity: Double = 0.75) {
        self.pressedOpacity = pressedOpacity
    }

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .opacity(configuration.isPressed ? pressedOpacity : 1)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

/// The inner layout every filled/outlined button shares.
private struct ButtonBody: View {
    let title: String
    let systemImage: String?
    let isLoading: Bool
    let fullWidth: Bool
    let foreground: Color
    let height: CGFloat

    var body: some View {
        HStack(spacing: 7) {
            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .tint(foreground)
            } else if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 13, weight: .semibold))
            }
            Text(title)
                .font(.system(size: 14, weight: .semibold))
                .lineLimit(1)
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 16)
        .frame(maxWidth: fullWidth ? .infinity : nil, minHeight: height)
        .contentShape(Rectangle())
    }
}

// MARK: - Buttons

/// Accent-filled call to action.
public struct PrimaryButton: View {
    public let title: String
    public var systemImage: String?
    public var isLoading: Bool
    public var isEnabled: Bool
    public var fullWidth: Bool
    public var height: CGFloat
    public let action: () -> Void

    public init(title: String,
                systemImage: String? = nil,
                isLoading: Bool = false,
                isEnabled: Bool = true,
                fullWidth: Bool = true,
                height: CGFloat = 42,
                action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isLoading = isLoading
        self.isEnabled = isEnabled
        self.fullWidth = fullWidth
        self.height = height
        self.action = action
    }

    private var active: Bool { isEnabled && !isLoading }

    public var body: some View {
        Button(action: action) {
            ButtonBody(title: title,
                       systemImage: systemImage,
                       isLoading: isLoading,
                       fullWidth: fullWidth,
                       foreground: Theme.accentFg,
                       height: height)
                .background(Theme.accent.opacity(active ? 1 : 0.45))
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(!active)
    }
}

/// Neutral button on the surface colour, with a border.
public struct SecondaryButton: View {
    public let title: String
    public var systemImage: String?
    public var isLoading: Bool
    public var isEnabled: Bool
    public var fullWidth: Bool
    public var height: CGFloat
    public let action: () -> Void

    public init(title: String,
                systemImage: String? = nil,
                isLoading: Bool = false,
                isEnabled: Bool = true,
                fullWidth: Bool = false,
                height: CGFloat = 42,
                action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isLoading = isLoading
        self.isEnabled = isEnabled
        self.fullWidth = fullWidth
        self.height = height
        self.action = action
    }

    private var active: Bool { isEnabled && !isLoading }

    public var body: some View {
        Button(action: action) {
            ButtonBody(title: title,
                       systemImage: systemImage,
                       isLoading: isLoading,
                       fullWidth: fullWidth,
                       foreground: active ? Theme.fg : Theme.fgSubtle,
                       height: height)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(!active)
    }
}

/// Destructive action. `filled` paints the danger colour, otherwise it is a tinted outline.
public struct DangerButton: View {
    public let title: String
    public var systemImage: String?
    public var isLoading: Bool
    public var isEnabled: Bool
    public var fullWidth: Bool
    public var filled: Bool
    public var height: CGFloat
    public let action: () -> Void

    public init(title: String,
                systemImage: String? = nil,
                isLoading: Bool = false,
                isEnabled: Bool = true,
                fullWidth: Bool = false,
                filled: Bool = true,
                height: CGFloat = 42,
                action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isLoading = isLoading
        self.isEnabled = isEnabled
        self.fullWidth = fullWidth
        self.filled = filled
        self.height = height
        self.action = action
    }

    private var active: Bool { isEnabled && !isLoading }

    public var body: some View {
        Button(action: action) {
            ButtonBody(title: title,
                       systemImage: systemImage,
                       isLoading: isLoading,
                       fullWidth: fullWidth,
                       foreground: filled ? Theme.accentFg : Theme.danger,
                       height: height)
                .background(filled ? Theme.danger.opacity(active ? 1 : 0.45) : Theme.dangerSoft)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(filled ? Color.clear : Theme.danger.opacity(0.35), lineWidth: 1)
                )
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(!active)
    }
}

/// Square, borderless icon button used in toolbars and list rows.
public struct IconButton: View {
    public let systemName: String
    public var label: String?
    public var tint: Color
    public var size: CGFloat
    public var fontSize: CGFloat
    public var isActive: Bool
    public var isEnabled: Bool
    public let action: () -> Void

    public init(systemName: String,
                label: String? = nil,
                tint: Color = Theme.fgMuted,
                size: CGFloat = 30,
                fontSize: CGFloat = 14,
                isActive: Bool = false,
                isEnabled: Bool = true,
                action: @escaping () -> Void) {
        self.systemName = systemName
        self.label = label
        self.tint = tint
        self.size = size
        self.fontSize = fontSize
        self.isActive = isActive
        self.isEnabled = isEnabled
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: fontSize, weight: .medium))
                .foregroundStyle(isEnabled ? (isActive ? Theme.accent : tint) : Theme.fgSubtle)
                .frame(width: size, height: size)
                .background(isActive ? Theme.accentSoft : Color.clear)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
        .disabled(!isEnabled)
        .help(label ?? "")
        .accessibilityLabel(Text(label ?? systemName))
    }
}

// MARK: - Text input

private extension View {
    /// The rounded, bordered box shared by every text control.
    func fieldChrome(focused: Bool, invalid: Bool = false) -> some View {
        self
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(invalid ? Theme.danger
                                          : (focused ? Theme.accent : Theme.border),
                                  lineWidth: focused || invalid ? 1.5 : 1)
            )
            .animation(.easeOut(duration: 0.12), value: focused)
    }
}

/// Single-line text field styled like the web app.
public struct StyledTextField: View {
    public let placeholder: String
    @Binding public var text: String
    public var systemImage: String?
    public var isInvalid: Bool
    public var disablesAutocorrection: Bool

    @FocusState private var focused: Bool

    public init(placeholder: String,
                text: Binding<String>,
                systemImage: String? = nil,
                isInvalid: Bool = false,
                disablesAutocorrection: Bool = false) {
        self.placeholder = placeholder
        self._text = text
        self.systemImage = systemImage
        self.isInvalid = isInvalid
        self.disablesAutocorrection = disablesAutocorrection
    }

    public var body: some View {
        HStack(spacing: 8) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fgSubtle)
            }
            TextField(placeholder, text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 14))
                .foregroundStyle(Theme.fg)
                .focused($focused)
                .autocorrectionDisabled(disablesAutocorrection)
        }
        .padding(.horizontal, 12)
        .frame(minHeight: 40)
        .fieldChrome(focused: focused, invalid: isInvalid)
        .contentShape(Rectangle())
        .onTapGesture { focused = true }
    }
}

/// Multi-line editor with a placeholder, because `TextEditor` has none.
public struct StyledTextEditor: View {
    public let placeholder: String
    @Binding public var text: String
    public var minHeight: CGFloat
    public var maxHeight: CGFloat?
    public var font: Font

    @FocusState private var focused: Bool

    public init(placeholder: String,
                text: Binding<String>,
                minHeight: CGFloat = 120,
                maxHeight: CGFloat? = nil,
                font: Font = .system(size: 14)) {
        self.placeholder = placeholder
        self._text = text
        self.minHeight = minHeight
        self.maxHeight = maxHeight
        self.font = font
    }

    public var body: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $text)
                .scrollContentBackground(.hidden)
                .background(Color.clear)
                .font(font)
                .foregroundStyle(Theme.fg)
                .focused($focused)
                .padding(EdgeInsets(top: 8, leading: 8, bottom: 8, trailing: 8))
                .frame(minHeight: minHeight, maxHeight: maxHeight)

            if text.isEmpty {
                Text(placeholder)
                    .font(font)
                    .foregroundStyle(Theme.fgSubtle)
                    .padding(.horizontal, 13)
                    .padding(.vertical, 16)
                    .allowsHitTesting(false)
            }
        }
        .padding(.horizontal, 4)
        .fieldChrome(focused: focused)
    }
}

// MARK: - Layout helpers

/// Label + optional hint above an arbitrary control.
public struct LabeledField<Content: View>: View {
    public let label: String
    public var description: String?
    public var isRequired: Bool
    @ViewBuilder public let content: Content

    public init(_ label: String,
                description: String? = nil,
                isRequired: Bool = false,
                @ViewBuilder content: () -> Content) {
        self.label = label
        self.description = description
        self.isRequired = isRequired
        self.content = content()
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 3) {
                Text(label)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                if isRequired {
                    Text("*").font(.system(size: 13, weight: .semibold)).foregroundStyle(Theme.danger)
                }
            }
            if let description {
                Text(description)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            content
        }
    }
}

/// Small rounded tag. Tappable when `onTap` is set, removable when `onRemove` is set.
public struct Chip: View {
    public let title: String
    public var systemImage: String?
    public var isSelected: Bool
    public var tint: Color?
    public var onTap: (() -> Void)?
    public var onRemove: (() -> Void)?

    public init(title: String,
                systemImage: String? = nil,
                isSelected: Bool = false,
                tint: Color? = nil,
                onTap: (() -> Void)? = nil,
                onRemove: (() -> Void)? = nil) {
        self.title = title
        self.systemImage = systemImage
        self.isSelected = isSelected
        self.tint = tint
        self.onTap = onTap
        self.onRemove = onRemove
    }

    private var foreground: Color {
        if let tint { return tint }
        return isSelected ? Theme.accent : Theme.fgMuted
    }

    private var background: Color {
        isSelected ? Theme.accentSoft : Theme.surface2
    }

    public var body: some View {
        HStack(spacing: 5) {
            if let systemImage {
                Image(systemName: systemImage).font(.system(size: 10, weight: .semibold))
            }
            Text(title)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
            if let onRemove {
                Button(action: onRemove) {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
                .accessibilityLabel(Text("Usuń"))
            }
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(background)
        .clipShape(Capsule())
        .overlay(
            Capsule().strokeBorder(isSelected ? Theme.accent.opacity(0.4) : Theme.border, lineWidth: 1)
        )
        .contentShape(Capsule())
        .modifier(TapIfNeeded(onTap: onTap))
    }
}

private struct TapIfNeeded: ViewModifier {
    let onTap: (() -> Void)?

    func body(content: Content) -> some View {
        if let onTap {
            Button(action: onTap) { content }
                .buttonStyle(PressableButtonStyle())
        } else {
            content
        }
    }
}

/// Uppercase section title with an optional trailing accessory.
public struct SectionHeader<Trailing: View>: View {
    public let title: String
    public var subtitle: String?
    @ViewBuilder public let trailing: Trailing

    public init(_ title: String, subtitle: String? = nil, @ViewBuilder trailing: () -> Trailing) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = trailing()
    }

    public var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .textCase(.uppercase)
                    .kerning(0.6)
                    .foregroundStyle(Theme.fgMuted)
                if let subtitle {
                    Text(subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgSubtle)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
            trailing
        }
    }
}

extension SectionHeader where Trailing == EmptyView {
    public init(_ title: String, subtitle: String? = nil) {
        self.init(title, subtitle: subtitle) { EmptyView() }
    }
}

/// Centred placeholder for empty lists and screens.
public struct EmptyState: View {
    public let systemImage: String
    public let title: String
    public var message: String?
    public var actionTitle: String?
    public var action: (() -> Void)?

    public init(systemImage: String,
                title: String,
                message: String? = nil,
                actionTitle: String? = nil,
                action: (() -> Void)? = nil) {
        self.systemImage = systemImage
        self.title = title
        self.message = message
        self.actionTitle = actionTitle
        self.action = action
    }

    public var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.bottom, 2)
            Text(title)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .multilineTextAlignment(.center)
            if let message {
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fgMuted)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 360)
            }
            if let actionTitle, let action {
                PrimaryButton(title: actionTitle, fullWidth: false, action: action)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(28)
    }
}

/// Surface panel with the standard border and corner radius.
public struct Card<Content: View>: View {
    public var padding: CGFloat
    public var background: Color
    @ViewBuilder public let content: Content

    public init(padding: CGFloat = 16,
                background: Color = Theme.surface,
                @ViewBuilder content: () -> Content) {
        self.padding = padding
        self.background = background
        self.content = content()
    }

    public var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
    }
}

extension View {
    /// Same chrome as `Card`, for views that build their own padding.
    public func cardSurface(padding: CGFloat = 16, background: Color = Theme.surface) -> some View {
        self
            .padding(padding)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
    }

    /// Hairline separator, horizontal.
    public func hairlineTop() -> some View {
        VStack(spacing: 0) {
            Rectangle().fill(Theme.border).frame(height: 1)
            self
        }
    }
}

/// Thin horizontal rule used between blocks inside a page.
public struct Hairline: View {
    public var vertical: Bool

    public init(vertical: Bool = false) {
        self.vertical = vertical
    }

    public var body: some View {
        Rectangle()
            .fill(Theme.border)
            .frame(width: vertical ? 1 : nil, height: vertical ? nil : 1)
    }
}

/// Determinate/indeterminate progress bar with the app's accent.
public struct ProgressBar: View {
    public var value: Double?
    public var height: CGFloat
    public var tint: Color

    public init(value: Double?, height: CGFloat = 6, tint: Color = Theme.accent) {
        self.value = value
        self.height = height
        self.tint = tint
    }

    public var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(Theme.surface2)
                if let value {
                    Capsule()
                        .fill(tint)
                        .frame(width: max(0, min(1, value)) * geometry.size.width)
                        .animation(.easeOut(duration: 0.2), value: value)
                }
            }
        }
        .frame(height: height)
    }
}

// MARK: - Flow layout

/// Wrapping row of chips or thumbnails. `LazyVGrid` would force equal columns,
/// which looks wrong for labels of wildly different widths.
public struct FlowLayout: Layout {
    public var spacing: CGFloat
    public var lineSpacing: CGFloat
    public var alignment: HorizontalAlignment

    public init(spacing: CGFloat = 6, lineSpacing: CGFloat = 6, alignment: HorizontalAlignment = .leading) {
        self.spacing = spacing
        self.lineSpacing = lineSpacing
        self.alignment = alignment
    }

    public func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let rows = layout(subviews: subviews, maxWidth: maxWidth)
        let height = rows.reduce(CGFloat.zero) { $0 + $1.height } +
            CGFloat(max(0, rows.count - 1)) * lineSpacing
        let widest = rows.map(\.width).max() ?? 0
        return CGSize(width: maxWidth.isFinite ? min(widest, maxWidth) : widest, height: height)
    }

    public func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) {
        let rows = layout(subviews: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in rows {
            var x: CGFloat
            if alignment == .trailing {
                x = bounds.maxX - row.width
            } else if alignment == .center {
                x = bounds.minX + (bounds.width - row.width) / 2
            } else {
                x = bounds.minX
            }
            for item in row.items {
                subviews[item.index].place(at: CGPoint(x: x, y: y),
                                           anchor: .topLeading,
                                           proposal: ProposedViewSize(item.size))
                x += item.size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private struct Item {
        let index: Int
        let size: CGSize
    }

    private struct Row {
        var items: [Item] = []
        var width: CGFloat = 0
        var height: CGFloat = 0
    }

    private func layout(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row()
        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(ProposedViewSize(width: maxWidth, height: nil))
            let candidate = current.items.isEmpty ? size.width : current.width + spacing + size.width
            if !current.items.isEmpty, candidate > maxWidth {
                rows.append(current)
                current = Row()
            }
            let width = current.items.isEmpty ? size.width : current.width + spacing + size.width
            current.items.append(Item(index: index, size: size))
            current.width = width
            current.height = max(current.height, size.height)
        }
        if !current.items.isEmpty { rows.append(current) }
        return rows
    }
}
