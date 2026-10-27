import SwiftUI
import LocalCodeKit

/// The conversation with the agent: messages, thinking, tool cards, plans,
/// notices and the permission question.
///
/// It follows the stream only while the reader is already at the end — the
/// moment they scroll up to study a diff the view stops moving under them and
/// offers a "back to the bottom" pill instead.
struct AgentTranscript: View {
    @Environment(AgentStore.self) private var store

    @State private var scroll = AgentScrollState()

    private let bottomAnchor = "agent.bottom.anchor"
    private let scrollSpace = "agent.scroll.space"
    private let contentWidth: CGFloat = 860

    var body: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottom) {
                    content(viewportHeight: viewport.size.height, proxy: proxy)
                    if showsScrollButton {
                        scrollDownButton(proxy)
                    }
                }
                .animation(.easeOut(duration: 0.18), value: showsScrollButton)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.bg)
    }

    // MARK: - Content

    @ViewBuilder
    private func content(viewportHeight: CGFloat, proxy: ScrollViewProxy) -> some View {
        if store.isLoadingTranscript && store.entries.isEmpty {
            ProgressView()
                .controlSize(.large)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if store.entries.isEmpty && store.pendingPermission == nil {
            emptyState(viewportHeight: viewportHeight)
        } else {
            list(viewportHeight: viewportHeight, proxy: proxy)
        }
    }

    private func emptyState(viewportHeight: CGFloat) -> some View {
        ScrollView {
            AgentWelcomeView()
                .frame(maxWidth: contentWidth)
                .frame(maxWidth: .infinity, minHeight: max(viewportHeight, 300))
                .padding(.horizontal, 16)
        }
    }

    private func list(viewportHeight: CGFloat, proxy: ScrollViewProxy) -> some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(store.entries) { entry in
                    row(for: entry)
                        .id(entry.id)
                }

                if let status = store.statusMessage, store.pendingPermission == nil {
                    StatusRow(message: status)
                }

                if let permission = store.pendingPermission {
                    PermissionCardView(request: permission) { decision in
                        store.answerPermission(decision)
                    }
                    .transition(.opacity)
                }

                Color.clear.frame(height: 1).id(bottomAnchor)
            }
            .frame(maxWidth: contentWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 20)
            .background(distanceReporter(viewportHeight: viewportHeight))
        }
        .coordinateSpace(name: scrollSpace)
        .onPreferenceChange(AgentBottomDistanceKey.self) { [scroll] distance in
            let atBottom = distance < 110
            Task { @MainActor in
                if scroll.isAtBottom != atBottom { scroll.isAtBottom = atBottom }
            }
        }
        .onChange(of: scrollSignature) { _, _ in
            guard scroll.isAtBottom else { return }
            proxy.scrollTo(bottomAnchor, anchor: .bottom)
        }
        .onChange(of: store.session?.id) { _, _ in
            scroll.isAtBottom = true
            proxy.scrollTo(bottomAnchor, anchor: .bottom)
        }
        .onAppear {
            scroll.isAtBottom = true
            Task {
                try? await Task.sleep(nanoseconds: 50_000_000)
                proxy.scrollTo(bottomAnchor, anchor: .bottom)
            }
        }
    }

    @ViewBuilder
    private func row(for entry: TranscriptEntry) -> some View {
        switch entry.kind {
        case .user:
            UserMessageRow(text: entry.text, images: entry.images)
        case .assistant:
            AssistantMessageRow(text: entry.text, isStreaming: isStreaming(entry))
        case .reasoning:
            ReasoningRow(text: entry.text, isStreaming: isStreaming(entry))
        case .tool:
            if let tool = entry.tool { ToolCardView(tool: tool) }
        case .notice:
            NoticeRow(text: entry.text)
        case .compaction:
            if let info = entry.compaction { CompactionRow(info: info) }
        case .todos:
            TodoCardView(todos: entry.todos)
        case .stats:
            TurnStatsRow(stats: entry.stats, reason: entry.reason)
        case .error:
            TranscriptErrorRow(text: entry.text)
        }
    }

    private func isStreaming(_ entry: TranscriptEntry) -> Bool {
        store.isBusy && entry.id == store.entries.last?.id
    }

    // MARK: - Scrolling

    private var showsScrollButton: Bool {
        !scroll.isAtBottom && !store.entries.isEmpty
    }

    private func scrollDownButton(_ proxy: ScrollViewProxy) -> some View {
        Button {
            scroll.isAtBottom = true
            withAnimation(.easeOut(duration: 0.22)) {
                proxy.scrollTo(bottomAnchor, anchor: .bottom)
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "arrow.down")
                    .font(.system(size: 11, weight: .bold))
                Text("Przewiń w dół")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(Theme.fg)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Theme.surface)
            .clipShape(Capsule())
            .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
            .shadow(color: Color.black.opacity(0.16), radius: 10, y: 3)
            .contentShape(Capsule())
        }
        .buttonStyle(PressableButtonStyle())
        .padding(.bottom, 14)
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .accessibilityLabel(Text("Przewiń w dół"))
    }

    private func distanceReporter(viewportHeight: CGFloat) -> some View {
        GeometryReader { geometry in
            Color.clear.preference(
                key: AgentBottomDistanceKey.self,
                value: geometry.size.height + geometry.frame(in: .named(scrollSpace)).minY - viewportHeight
            )
        }
    }

    /// Changes whenever the transcript grows or the last block gains characters.
    private var scrollSignature: String {
        let last = store.entries.last
        return [
            "\(store.entries.count)",
            last?.id ?? "-",
            "\(last?.text.count ?? 0)",
            "\(last?.tool?.liveOutput.count ?? 0)",
            "\(last?.tool?.output?.count ?? 0)",
            store.statusMessage ?? "-",
            store.pendingPermission?.requestId ?? "-"
        ].joined(separator: "|")
    }
}

/// Kept in a small reference type so the preference callback can hop to the
/// main actor without capturing the whole view.
@MainActor
@Observable
final class AgentScrollState {
    var isAtBottom = true
}

private struct AgentBottomDistanceKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

// MARK: - Empty transcript

/// What a fresh session shows: what the agent can do here, and two hints.
struct AgentWelcomeView: View {
    @Environment(AgentStore.self) private var store

    var body: some View {
        if store.isHelpbot {
            helpbotWelcome
        } else {
            agentWelcome
        }
    }

    /// The help bot's first screen: what it is for, and sentences to start with.
    /// Clicking one sends it — the people this is for do not know what to ask.
    private var helpbotWelcome: some View {
        VStack(spacing: 12) {
            Text("W czym pomóc?")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("Napisz zwykłym zdaniem, czego potrzebujesz. Zanim cokolwiek pobiorę, "
                 + "załaduję albo zmienię, pokażę to do zatwierdzenia.")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 6) {
                ForEach(Self.helpbotExamples, id: \.self) { example in
                    Button {
                        store.send(example)
                    } label: {
                        Text(example)
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.fg)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(Theme.surface)
                            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Theme.border, lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                    .disabled(!store.canSend)
                }
            }
            .padding(.top, 4)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 28)
    }

    static let helpbotExamples = [
        "Pobierz mi dobry model do kodowania, który zmieści się na tym Macu.",
        "Jakie modele mam na dysku i które są teraz w pamięci?",
        "Mam mało miejsca — co mogę usunąć?",
        "Sprawdź, czy wszystko działa jak trzeba.",
    ]

    private var agentWelcome: some View {
        VStack(spacing: 14) {
            Image(systemName: "chevron.left.forwardslash.chevron.right")
                .font(.system(size: 30, weight: .light))
                .foregroundStyle(Theme.accent)
            Text("Agent gotowy do pracy")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("Pracuje w folderze \(store.session?.shortCwd ?? "—") i może czytać pliki, szukać w kodzie, wprowadzać zmiany i uruchamiać polecenia.")
                .font(.system(size: 13))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 460)

            VStack(alignment: .leading, spacing: 6) {
                hint("Napisz, co ma zrobić — np. „dodaj testy do modułu X”.")
                hint("Tryb „\(store.activeMode.label)”: \(store.activeMode.hint)")
                hint("Wpisz /, żeby zobaczyć polecenia; @plik dołącza plik do wiadomości.")
            }
            .padding(.top, 4)
        }
        .padding(.vertical, 40)
    }

    private func hint(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: "sparkle")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.top, 3)
            Text(text)
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
