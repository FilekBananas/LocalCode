import Foundation
import SwiftUI
import LocalCodeKit

/// The conversation pane: transcript on top, composer at the bottom.
///
/// The transcript follows the stream only while the reader is already at the
/// end. The moment they scroll up to re-read something the view stops moving
/// under them and offers a "back to the bottom" pill instead.
public struct ChatView: View {
    @Environment(ChatStore.self) private var store

    @State private var draft = ""
    @State private var scroll = ChatScrollState()
    @State private var focusRequest = 0

    private let bottomAnchor = "chat.bottom.anchor"
    private let scrollSpace = "chat.scroll.space"
    private let contentWidth: CGFloat = 760

    public init() {}

    // MARK: - Body

    public var body: some View {
        VStack(spacing: 0) {
            transcript
            composerBar
        }
        .background(Theme.bg)
        .tint(Theme.accent)
    }

    // MARK: - Transcript

    private var transcript: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottom) {
                    transcriptBody(viewportHeight: viewport.size.height, proxy: proxy)

                    if showsScrollButton {
                        scrollDownButton(proxy)
                    }
                }
                .animation(.easeOut(duration: 0.18), value: showsScrollButton)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func transcriptBody(viewportHeight: CGFloat, proxy: ScrollViewProxy) -> some View {
        if store.messages.isEmpty && store.isLoadingMessages {
            ProgressView()
                .controlSize(.large)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if store.messages.isEmpty {
            welcome(viewportHeight: viewportHeight)
        } else {
            messageList(viewportHeight: viewportHeight, proxy: proxy)
        }
    }

    private func welcome(viewportHeight: CGFloat) -> some View {
        ScrollView {
            VStack(spacing: 0) {
                Spacer(minLength: 24)
                WelcomeView(displayName: greetingName,
                            folder: store.activeFolder,
                            hasModels: !store.models.isEmpty,
                            onSuggestion: { prompt in
                                draft = prompt
                                focusRequest &+= 1
                            },
                            onBrowseModels: { ShellBus.shared.request(.show(.discover)) })
                Spacer(minLength: 24)
            }
            .frame(maxWidth: contentWidth)
            .frame(maxWidth: .infinity, minHeight: max(viewportHeight, 320))
            .padding(.horizontal, 16)
        }
    }

    private func messageList(viewportHeight: CGFloat, proxy: ScrollViewProxy) -> some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 14) {
                ForEach(store.messages) { message in
                    row(for: message)
                        .id(message.id)
                }
                // Compactions happen between turns, so they render after the
                // history they shortened and before whatever comes next.
                ForEach(store.compactions) { compaction in
                    CompactionNotice(compaction: compaction)
                        .id(compaction.id)
                }
                Color.clear
                    .frame(height: 1)
                    .id(bottomAnchor)
            }
            .frame(maxWidth: contentWidth)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 18)
            .background(distanceReporter(viewportHeight: viewportHeight))
        }
        .coordinateSpace(name: scrollSpace)
        .onPreferenceChange(ChatBottomDistanceKey.self) { [scroll] distance in
            // The closure may be @Sendable, so the hop is explicit.
            let atBottom = distance < 90
            Task { @MainActor in
                if scroll.isAtBottom != atBottom { scroll.isAtBottom = atBottom }
            }
        }
        .onChange(of: scrollSignature) { _, _ in
            guard scroll.isAtBottom else { return }
            proxy.scrollTo(bottomAnchor, anchor: .bottom)
        }
        .onChange(of: store.activeConversation?.id) { _, _ in
            scroll.isAtBottom = true
        }
        .onAppear {
            scroll.isAtBottom = true
            Task {
                try? await Task.sleep(nanoseconds: 40_000_000)
                proxy.scrollTo(bottomAnchor, anchor: .bottom)
            }
        }
    }

    private func row(for message: Message) -> some View {
        let isStreamingMessage = store.isStreaming && message.id == store.streamingMessageId
        return MessageRow(
            message: message,
            api: store.api,
            isLast: message.id == store.messages.last?.id,
            isStreamingMessage: isStreamingMessage,
            isBusy: store.isBusy,
            showReasoning: store.settings.defaults.showReasoning,
            modelLabel: store.label(forModel: message.model, provider: message.provider),
            statusText: isStreamingMessage ? store.statusMessage : nil,
            onRegenerate: { Task { await store.regenerate() } },
            onEdit: { text in Task { await store.edit(messageId: message.id, text: text) } },
            onDelete: { Task { await store.deleteMessage(message.id) } },
            onRetry: { retry(message) }
        )
    }

    // MARK: - Scroll to bottom

    private var showsScrollButton: Bool {
        !scroll.isAtBottom && !store.messages.isEmpty
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

    /// Reports how far the bottom of the content is from the bottom of the viewport.
    private func distanceReporter(viewportHeight: CGFloat) -> some View {
        GeometryReader { geometry in
            Color.clear.preference(
                key: ChatBottomDistanceKey.self,
                value: geometry.size.height + geometry.frame(in: .named(scrollSpace)).minY - viewportHeight
            )
        }
    }

    /// Changes whenever the transcript grows or the last answer gains tokens.
    private var scrollSignature: String {
        guard let last = store.messages.last else { return "empty" }
        return "\(store.messages.count)|\(last.id)|\(last.content.count)|\(last.reasoning?.count ?? 0)|\(last.status.rawValue)"
    }

    // MARK: - Composer

    private var composerBar: some View {
        VStack(spacing: 0) {
            Hairline()

            Composer(text: $draft,
                     api: store.api,
                     isStreaming: store.isStreaming,
                     isBlocked: store.isStreamingInBackground,
                     models: store.models,
                     providers: store.providers,
                     currentProvider: store.currentProvider,
                     currentModel: store.currentModel,
                     focusRequest: focusRequest,
                     onSelectModel: { provider, model in
                         Task { await store.selectModel(provider: provider, model: model) }
                     },
                     onSend: { text, attachments in
                         scroll.isAtBottom = true
                         Task { await store.send(text: text, attachments: attachments) }
                     },
                     onStop: { store.stopStreaming() })
                .frame(maxWidth: contentWidth)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 16)
                .padding(.top, 10)
                .padding(.bottom, 12)
        }
        .background(Theme.bg)
    }

    // MARK: - Helpers

    private var greetingName: String? {
        let profile = store.settings.profile.displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        return profile.isEmpty ? nil : profile
    }

    /// Retry after a failed turn. When the very first message failed there is
    /// no conversation to regenerate into, so the text is simply sent again.
    private func retry(_ message: Message) {
        Task {
            if store.activeConversation != nil {
                await store.regenerate()
            } else if message.role == .user {
                let text = message.content
                store.messages.removeAll { $0.id == message.id }
                await store.send(text: text)
            }
        }
    }
}

// MARK: - Scroll state

/// Kept in a small reference type so the preference callback can hop to the
/// main actor without capturing the whole view.
@MainActor
@Observable
final class ChatScrollState {
    var isAtBottom = true
}

private struct ChatBottomDistanceKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
