import SwiftUI
import LocalCodeKit

/// The "Ustawienia" section: profile, system prompt, generation defaults,
/// memory, the knowledge base and the bundled documentation — plus one page
/// the app target supplies for the server itself.
///
/// Every control writes through `SettingsDraft`, which batches edits into one
/// PATCH shortly after the last keystroke; a slider drag costs one request, not
/// forty.

// MARK: - Draft

/// Local copy of `AppSettings` that saves itself shortly after the last edit.
@MainActor
@Observable
public final class SettingsDraft {
    public private(set) var value: AppSettings = AppSettings()
    public private(set) var isSaving = false
    public private(set) var didJustSave = false

    @ObservationIgnored private var store: ChatStore?
    @ObservationIgnored private var saveTask: Task<Void, Never>?
    @ObservationIgnored private var noticeTask: Task<Void, Never>?
    /// True between an edit and the PATCH that carries it.
    @ObservationIgnored private var isDirty = false

    /// Long enough to swallow typing, short enough to feel immediate.
    private static let saveDelay: UInt64 = 700_000_000
    private static let noticeDelay: UInt64 = 2_000_000_000

    public init() {}

    /// Connects the draft to the store and takes its current settings.
    public func bind(to store: ChatStore) {
        self.store = store
        if !isDirty, value != store.settings { value = store.settings }
    }

    /// Adopts settings changed elsewhere (bootstrap, model picker) unless the
    /// user has unsaved edits on screen.
    public func syncFromStore() {
        guard !isDirty, let store, value != store.settings else { return }
        value = store.settings
    }

    public func set<T>(_ keyPath: WritableKeyPath<AppSettings, T>, _ newValue: T) {
        value[keyPath: keyPath] = newValue
        isDirty = true
        scheduleSave()
    }

    public func binding<T>(_ keyPath: WritableKeyPath<AppSettings, T>) -> Binding<T> {
        Binding(get: { self.value[keyPath: keyPath] },
                set: { self.set(keyPath, $0) })
    }

    /// `Slider` works in `Double`; whole-number settings go through here.
    public func intBinding(_ keyPath: WritableKeyPath<AppSettings, Int>) -> Binding<Double> {
        Binding(get: { Double(self.value[keyPath: keyPath]) },
                set: { self.set(keyPath, Int($0.rounded())) })
    }

    /// Saves right away — called when the screen goes away.
    public func flush() {
        saveTask?.cancel()
        guard isDirty else { return }
        saveTask = Task { await self.save() }
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task {
            try? await Task.sleep(nanoseconds: Self.saveDelay)
            guard !Task.isCancelled else { return }
            await self.save()
        }
    }

    private func save() async {
        guard let store else { return }
        let sent = value
        isDirty = false
        isSaving = true
        await store.patchSettings(sent)
        isSaving = false
        // The server normalises a few fields; adopt its answer, but never over
        // something the user typed while the request was in flight.
        if !isDirty, value != store.settings { value = store.settings }
        announceSaved()
    }

    private func announceSaved() {
        didJustSave = true
        noticeTask?.cancel()
        noticeTask = Task {
            try? await Task.sleep(nanoseconds: Self.noticeDelay)
            guard !Task.isCancelled else { return }
            self.didJustSave = false
        }
    }
}

// MARK: - Screen

public struct ChatSettingsView<Extra: View>: View {
    private enum Page: String, CaseIterable, Identifiable, Hashable {
        case server, agent, profile, prompt, generation, memory, knowledge, documentation

        var id: String { rawValue }

        var title: String {
            switch self {
            case .server: "Serwer"
            case .agent: "Agent"
            case .profile: "O mnie"
            case .prompt: "Prompt systemowy"
            case .generation: "Generowanie"
            case .memory: "Pamięć"
            case .knowledge: "Wiedza"
            case .documentation: "Dokumentacja"
            }
        }

        var systemImage: String {
            switch self {
            case .server: "server.rack"
            case .agent: "chevron.left.forwardslash.chevron.right"
            case .profile: "person.text.rectangle"
            case .prompt: "wand.and.stars"
            case .generation: "slider.horizontal.3"
            case .memory: "brain.head.profile"
            case .knowledge: "books.vertical"
            case .documentation: "book"
            }
        }

        var summary: String {
            switch self {
            case .server: "Połączenie z lokalnym serwerem LocalCode"
            case .agent: "Limit kroków, tryb Auto, recenzent, strażnik"
            case .profile: "Kim jesteś i jak model ma odpowiadać"
            case .prompt: "Instrukcja dołączana do każdej rozmowy"
            case .generation: "Model, temperatura, limity kontekstu"
            case .memory: "Co model pamięta między rozmowami"
            case .knowledge: "Pliki przeszukiwane przy odpowiedziach"
            case .documentation: "Jak używać i API na localhost"
            }
        }
    }

    @Environment(ChatStore.self) private var store

    @State private var draft = SettingsDraft()
    @State private var page: Page

    private let extraTitle: String?
    private let extra: Extra
    /// The coding agent's settings — they live in a module this one does not
    /// see, so the shell hands them in. No page without them.
    private let agent: AnyView?

    public init(extraTitle: String? = nil, agent: AnyView? = nil, @ViewBuilder extra: () -> Extra) {
        self.extraTitle = extraTitle
        self.extra = extra()
        self.agent = agent
        _page = State(initialValue: extraTitle == nil ? .profile : .server)
    }

    private var pages: [Page] {
        Page.allCases.filter { item in
            switch item {
            case .server: extraTitle != nil
            case .agent: agent != nil
            default: true
            }
        }
    }

    public var body: some View {
        HStack(spacing: 0) {
            pageList
            Hairline(vertical: true)
            content
        }
        .background(Theme.bg)
        .tint(Theme.accent)
        .task { draft.bind(to: store) }
        .onChange(of: store.settings) { _, _ in draft.syncFromStore() }
        .onDisappear { draft.flush() }
    }

    // MARK: Page list

    private var pageList: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(pages) { item in
                Button {
                    page = item
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: item.systemImage)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(page == item ? Theme.accent : Theme.fgMuted)
                            .frame(width: 18)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(title(for: item))
                                .font(.system(size: 13, weight: page == item ? .semibold : .regular))
                                .foregroundStyle(Theme.fg)
                                .lineLimit(1)
                            Text(item.summary)
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgSubtle)
                                .lineLimit(1)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle())
                .sidebarRowChrome(isActive: page == item, isHovered: false)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
        .padding(.top, 12)
        .frame(width: 240)
        .background(Theme.sidebar)
    }

    private func title(for item: Page) -> String {
        item == .server ? (extraTitle ?? item.title) : item.title
    }

    // MARK: Content

    @ViewBuilder
    private var content: some View {
        switch page {
        case .server:
            SettingsPage { extra }
        case .agent:
            SettingsPage { agent ?? AnyView(EmptyView()) }
        case .profile:
            SettingsPage {
                ProfileSection(draft: draft)
                SettingsAutosaveNote(draft: draft)
            }
        case .prompt:
            SettingsPage {
                SystemPromptSection(draft: draft)
                SettingsAutosaveNote(draft: draft)
            }
        case .generation:
            SettingsPage {
                GenerationSection(draft: draft)
                SettingsAutosaveNote(draft: draft)
            }
        case .memory:
            SettingsPage {
                MemorySection(draft: draft, api: store.api)
                SettingsAutosaveNote(draft: draft)
            }
        case .knowledge:
            SettingsPage {
                KnowledgeSection(api: store.api)
            }
        case .documentation:
            SettingsPage {
                DocumentationSection()
            }
        }
    }
}

extension ChatSettingsView where Extra == EmptyView {
    public init() {
        self.init(extraTitle: nil) { EmptyView() }
    }
}

// MARK: - Shared page furniture

/// Scrollable, width-limited body shared by every settings page.
struct SettingsPage<Content: View>: View {
    var maxWidth: CGFloat = 640
    @ViewBuilder let content: Content

    init(maxWidth: CGFloat = 640, @ViewBuilder content: () -> Content) {
        self.maxWidth = maxWidth
        self.content = content()
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                content
            }
            .frame(maxWidth: maxWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.horizontal, 20)
            .padding(.top, 18)
            .padding(.bottom, 28)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }
}

/// Label + description on the left, a switch on the right.
struct SettingsToggleRow: View {
    let title: String
    var description: String?
    @Binding var isOn: Bool

    init(_ title: String, description: String? = nil, isOn: Binding<Bool>) {
        self.title = title
        self.description = description
        self._isOn = isOn
    }

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                if let description {
                    Text(description)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            Toggle("", isOn: $isOn)
                .labelsHidden()
                .toggleStyle(.switch)
                .tint(Theme.accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Slider with a title, a live value and an explanation underneath.
struct SettingsSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double = 1
    let valueLabel: String
    var description: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Spacer(minLength: 8)
                Text(valueLabel)
                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                    .foregroundStyle(Theme.fgMuted)
                    .monospacedDigit()
            }
            Slider(value: $value, in: range, step: step)
                .tint(Theme.accent)
            if let description {
                Text(description)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// Autosave state for the pages that write straight through to the server.
struct SettingsAutosaveNote: View {
    let draft: SettingsDraft

    var body: some View {
        HStack(spacing: 6) {
            if draft.isSaving {
                ProgressView().controlSize(.small)
                Text("Zapisywanie…")
            } else if draft.didJustSave {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.success)
                Text("Zapisano")
            } else {
                Image(systemName: "arrow.clockwise")
                Text("Zmiany zapisują się automatycznie.")
            }
            Spacer(minLength: 0)
        }
        .font(.system(size: 12))
        .foregroundStyle(Theme.fgSubtle)
        .animation(.easeOut(duration: 0.16), value: draft.isSaving)
        .animation(.easeOut(duration: 0.16), value: draft.didJustSave)
    }
}
