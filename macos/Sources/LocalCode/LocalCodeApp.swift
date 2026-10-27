import SwiftUI
import AppKit
import LocalCodeKit

/// The LocalCode Mac app.
///
/// One window, one `ServerController`. The controller is created here and torn
/// down in `applicationWillTerminate`, because the server it may have spawned
/// holds several gigabytes of model weights and must not outlive the window
/// that started it.
@main
struct LocalCodeApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup {
            RootView(controller: delegate.controller)
        }
        .defaultSize(width: 1280, height: 840)
        .commands { AppCommands() }

        // The bundled user guide and API reference (Help menu, Settings →
        // Dokumentacja). Its own window, so it can stay open next to the app.
        Window("Dokumentacja LocalCode", id: DocumentationWindow.id) {
            DocumentationView()
                .frame(minWidth: 760, minHeight: 480)
        }
        .defaultSize(width: 1080, height: 820)
    }
}

/// Owns the controller so the termination hook can reach it, and makes the
/// process behave like a real app even when it is launched from a terminal.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor let controller = ServerController()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            controller.shutdown()
        }
    }
}

/// The only view the scene creates: the startup screen until the Python core
/// answers, then the shell.
struct RootView: View {
    let controller: ServerController

    @State private var didStart = false

    var body: some View {
        Group {
            switch controller.phase {
            case .connecting, .failed:
                StartupView(controller: controller)
            case .ready:
                AppShellView(controller: controller)
                    .transition(.opacity)
            }
        }
        .frame(minWidth: 900, minHeight: 600)
        .tint(Theme.accent)
        .background(Theme.bg)
        .animation(.easeOut(duration: 0.2), value: controller.phase)
        .task {
            guard !didStart else { return }
            didStart = true
            controller.start()
        }
    }
}

/// Menu-bar commands. They cannot reach the window's state directly, so they go
/// through `ShellBus`.
struct AppCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("Nowa rozmowa") {
                ShellBus.shared.request(.show(.chat))
                ShellBus.shared.request(.newConversation)
            }
            .keyboardShortcut("n", modifiers: .command)
        }

        CommandGroup(replacing: .sidebar) {
            Button("Pokaż lub ukryj listę rozmów") {
                ShellBus.shared.request(.toggleSidebar)
            }
            .keyboardShortcut("b", modifiers: .command)
            Divider()
        }

        CommandMenu("Rozmowa") {
            Button("Szukaj w rozmowach…") {
                ShellBus.shared.request(.show(.chat))
                ShellBus.shared.request(.search)
            }
            .keyboardShortcut("f", modifiers: .command)

            Button("Zatrzymaj generowanie") {
                ShellBus.shared.request(.stopStreaming)
            }
            .keyboardShortcut(.escape, modifiers: [])

            Divider()

            Button("Przypnij lub odepnij") {
                ShellBus.shared.request(.togglePin)
            }
            .keyboardShortcut("p", modifiers: [.command, .shift])

            Button("Eksportuj (Markdown)…") {
                ShellBus.shared.request(.export("md"))
            }
            .keyboardShortcut("e", modifiers: .command)

            Button("Eksportuj (JSON)…") {
                ShellBus.shared.request(.export("json"))
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])

            Divider()

            Button("Usuń rozmowę…") {
                ShellBus.shared.request(.deleteConversation)
            }
            .keyboardShortcut(.delete, modifiers: .command)
        }

        CommandMenu("Widok") {
            ForEach(AppSection.allCases) { section in
                Button(section.title) {
                    ShellBus.shared.request(.show(section))
                }
                .keyboardShortcut(shortcut(for: section), modifiers: .command)
            }
            Divider()
            Button("Odśwież") {
                ShellBus.shared.request(.refresh)
            }
            .keyboardShortcut("r", modifiers: .command)
        }

        // Replaces "LocalCode Help", which only says that no help exists.
        CommandGroup(replacing: .help) {
            Button("Dokumentacja LocalCode") {
                openWindow(id: DocumentationWindow.id)
            }
            .keyboardShortcut("?", modifiers: .command)

            Button("Pobierz dokumentację…") {
                DocumentationExport.runFromMenu()
            }
        }
    }

    /// ⌘1…⌘6 down the rail, in the order the rail shows them.
    private func shortcut(for section: AppSection) -> KeyEquivalent {
        let index = AppSection.allCases.firstIndex(of: section) ?? 0
        return KeyEquivalent(Character("\(index + 1)"))
    }
}
