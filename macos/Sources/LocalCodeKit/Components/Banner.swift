import SwiftUI

/// Inline message bars and the confirmation card.
///
/// Errors from the Python core arrive as ready-to-show Polish sentences, so the
/// UI never has to translate anything — it only has to put the sentence
/// somewhere the reader will see it, which is what `Banner` is for.

// MARK: - Banner

public enum BannerKind: Sendable, Hashable {
    case error
    case info
    case success
    case warning

    public var systemImage: String {
        switch self {
        case .error: "exclamationmark.triangle.fill"
        case .info: "info.circle.fill"
        case .success: "checkmark.circle.fill"
        case .warning: "exclamationmark.circle.fill"
        }
    }

    public var tint: Color {
        switch self {
        case .error: Theme.danger
        case .info: Theme.accent
        case .success: Theme.success
        case .warning: Theme.warning
        }
    }

    public var background: Color { tint.opacity(0.12) }

    public var defaultTitle: String {
        switch self {
        case .error: "Błąd"
        case .info: "Informacja"
        case .success: "Gotowe"
        case .warning: "Uwaga"
        }
    }
}

/// Inline message bar used for form errors, hints and confirmations.
public struct Banner: View {
    public let kind: BannerKind
    public let message: String
    public var title: String?
    public var actionTitle: String?
    public var action: (() -> Void)?
    public var onDismiss: (() -> Void)?

    public init(_ kind: BannerKind,
                message: String,
                title: String? = nil,
                actionTitle: String? = nil,
                action: (() -> Void)? = nil,
                onDismiss: (() -> Void)? = nil) {
        self.kind = kind
        self.message = message
        self.title = title
        self.actionTitle = actionTitle
        self.action = action
        self.onDismiss = onDismiss
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: kind.systemImage)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(kind.tint)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 3) {
                if let title {
                    Text(title)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                }
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(title == nil ? Theme.fg : Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)

                if let actionTitle, let action {
                    Button(action: action) {
                        Text(actionTitle)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(kind.tint)
                    }
                    .buttonStyle(PressableButtonStyle())
                    .padding(.top, 2)
                }
            }

            Spacer(minLength: 0)

            if let onDismiss {
                Button(action: onDismiss) {
                    Image(systemName: "xmark")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundStyle(Theme.fgMuted)
                        .frame(width: 22, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
                .accessibilityLabel(Text("Zamknij"))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(kind.background)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(kind.tint.opacity(0.3), lineWidth: 1)
        )
    }
}

extension View {
    /// Shows a banner above the view whenever `message` is non-nil.
    public func banner(_ message: Binding<String?>,
                       kind: BannerKind = .error,
                       dismissible: Bool = true) -> some View {
        VStack(spacing: 10) {
            if let text = message.wrappedValue {
                Banner(kind, message: text, onDismiss: dismissible ? { message.wrappedValue = nil } : nil)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
            self
        }
        .animation(.easeOut(duration: 0.18), value: message.wrappedValue)
    }
}

// MARK: - Confirm dialog

/// A pending confirmation. Present it with `.confirmDialog(_:)`.
public struct ConfirmRequest: Identifiable {
    public let id = UUID()
    public var title: String
    public var message: String?
    public var confirmTitle: String
    public var cancelTitle: String
    public var isDestructive: Bool
    public var confirm: () -> Void

    public init(title: String,
                message: String? = nil,
                confirmTitle: String = "Potwierdź",
                cancelTitle: String = "Anuluj",
                isDestructive: Bool = true,
                confirm: @escaping () -> Void) {
        self.title = title
        self.message = message
        self.confirmTitle = confirmTitle
        self.cancelTitle = cancelTitle
        self.isDestructive = isDestructive
        self.confirm = confirm
    }
}

/// The card itself. Usually presented through `.confirmDialog(_:)`.
public struct ConfirmDialog: View {
    public let title: String
    public var message: String?
    public var confirmTitle: String
    public var cancelTitle: String
    public var isDestructive: Bool
    public let onConfirm: () -> Void
    public let onCancel: () -> Void

    public init(title: String,
                message: String? = nil,
                confirmTitle: String = "Potwierdź",
                cancelTitle: String = "Anuluj",
                isDestructive: Bool = true,
                onConfirm: @escaping () -> Void,
                onCancel: @escaping () -> Void) {
        self.title = title
        self.message = message
        self.confirmTitle = confirmTitle
        self.cancelTitle = cancelTitle
        self.isDestructive = isDestructive
        self.onConfirm = onConfirm
        self.onCancel = onCancel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .fixedSize(horizontal: false, vertical: true)

            if let message {
                Text(message)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                Spacer(minLength: 0)
                SecondaryButton(title: cancelTitle, height: 38, action: onCancel)
                if isDestructive {
                    DangerButton(title: confirmTitle, height: 38, action: onConfirm)
                } else {
                    PrimaryButton(title: confirmTitle, fullWidth: false, height: 38, action: onConfirm)
                }
            }
            .padding(.top, 2)
        }
        .padding(18)
        .frame(maxWidth: 380, alignment: .leading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.22), radius: 24, y: 10)
        .padding(20)
    }
}

private struct ConfirmDialogModifier: ViewModifier {
    @Binding var request: ConfirmRequest?

    func body(content: Content) -> some View {
        content
            .overlay {
                if let request {
                    ZStack {
                        Color.black.opacity(0.35)
                            .ignoresSafeArea()
                            .contentShape(Rectangle())
                            .onTapGesture { self.request = nil }

                        ConfirmDialog(title: request.title,
                                      message: request.message,
                                      confirmTitle: request.confirmTitle,
                                      cancelTitle: request.cancelTitle,
                                      isDestructive: request.isDestructive,
                                      onConfirm: {
                                          let confirm = request.confirm
                                          self.request = nil
                                          confirm()
                                      },
                                      onCancel: { self.request = nil })
                            .transition(.scale(scale: 0.96).combined(with: .opacity))
                    }
                    .transition(.opacity)
                }
            }
            .animation(.easeOut(duration: 0.16), value: request?.id)
    }
}

extension View {
    /// Presents a confirmation card over the view while `request` is non-nil.
    public func confirmDialog(_ request: Binding<ConfirmRequest?>) -> some View {
        modifier(ConfirmDialogModifier(request: request))
    }
}
