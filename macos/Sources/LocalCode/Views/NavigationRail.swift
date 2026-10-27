import SwiftUI
import LocalCodeKit

/// The narrow rail down the left edge of the window.
///
/// Six destinations, LM Studio style: a chat, the coding agent, the model
/// browser, the local library, the runtime panel and settings. It is a rail
/// rather than a `TabView` because every section keeps its own state and its
/// own second-level sidebar, and a rail makes that hierarchy obvious.
struct NavigationRail: View {
    @Binding var selection: AppSection
    /// Shown as a dot on "Runtime" when something is loaded.
    var loadedModelCount: Int = 0

    @Environment(\.openWindow) private var openWindow

    private static let width: CGFloat = 78

    var body: some View {
        VStack(spacing: 2) {
            mark
                .padding(.top, 14)
                .padding(.bottom, 10)

            ForEach(AppSection.allCases) { section in
                if section == .settings {
                    Spacer(minLength: 8)
                    helpItem
                }
                item(section)
            }

            Spacer(minLength: 10)
        }
        .frame(width: Self.width)
        .frame(maxHeight: .infinity)
        .background(Theme.rail)
        .overlay(alignment: .trailing) { Hairline(vertical: true) }
    }

    private var mark: some View {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
            .fill(Theme.accent)
            .frame(width: 30, height: 30)
            .overlay(
                Image(systemName: "cpu")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.accentFg)
            )
            .accessibilityLabel(Text("LocalCode"))
    }

    /// Not a section: the documentation opens in its own window, so it can sit
    /// next to whatever the user is doing.
    private var helpItem: some View {
        Button {
            openWindow(id: DocumentationWindow.id)
        } label: {
            VStack(spacing: 3) {
                Image(systemName: "book")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(Theme.fgMuted)
                    .frame(width: 30, height: 24)
                Text("Pomoc")
                    .font(.system(size: 10))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle(pressedOpacity: 0.6))
        .padding(.horizontal, 6)
        .help("Dokumentacja: jak używać LocalCode i API na localhost (⌘?)")
        .accessibilityLabel(Text("Pomoc — dokumentacja"))
    }

    private func item(_ section: AppSection) -> some View {
        let isActive = selection == section
        return Button {
            selection = section
        } label: {
            VStack(spacing: 3) {
                ZStack(alignment: .topTrailing) {
                    Image(systemName: section.systemImage)
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(isActive ? Theme.accent : Theme.fgMuted)
                        .frame(width: 30, height: 24)
                    if section == .runtime, loadedModelCount > 0 {
                        Circle()
                            .fill(Theme.success)
                            .frame(width: 6, height: 6)
                            .offset(x: 2, y: -1)
                    }
                }
                Text(section.title)
                    .font(.system(size: 10, weight: isActive ? .semibold : .regular))
                    .foregroundStyle(isActive ? Theme.fg : Theme.fgMuted)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 7)
            .background(isActive ? Theme.accentSoft : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle(pressedOpacity: 0.6))
        .padding(.horizontal, 6)
        .help(section.hint)
        .accessibilityLabel(Text(section.title))
        .accessibilityAddTraits(isActive ? [.isButton, .isSelected] : .isButton)
    }
}
