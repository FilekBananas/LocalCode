import SwiftUI
import LocalCodeKit
import ModelsFeature

/// "Mowa" — dictation (speech-to-text) and reading aloud (text-to-speech).
///
/// Two tasks that share one shelf in the store but nothing else (see
/// `docs/SPEECH.md`): one turns a recording into text, the other turns typed
/// text into a recording. A segmented switch picks which composer is showing,
/// the way `LoadSettingsSheet`'s KV-cache control picks a mode — no tab bar,
/// because the rest of the app does not have one either.
public struct SpeechView: View {
    @Environment(SpeechStore.self) private var store
    @Environment(ModelsStore.self) private var models

    @State private var mode: SpeechMode = .dictate

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            header
            messages
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    switch mode {
                    case .dictate: DictationSection()
                    case .speak: SpeakSection()
                    }
                }
                .padding(20)
                .frame(maxWidth: 720, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .top)
            }
        }
        .background(Theme.bg)
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Mowa")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("Dyktowanie i czytanie na głos — lokalnie, bez wysyłania nagrań")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
            ModeSwitch(mode: $mode)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
    }

    @ViewBuilder
    private var messages: some View {
        @Bindable var store = store
        if store.errorMessage != nil || store.statusMessage != nil {
            VStack(spacing: 8) {
                if let message = store.errorMessage {
                    Banner(.error, message: message, onDismiss: { store.errorMessage = nil })
                }
                if let note = store.statusMessage {
                    Banner(.success, message: note)
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .animation(.easeOut(duration: 0.18), value: store.errorMessage)
            .animation(.easeOut(duration: 0.18), value: store.statusMessage)
        }
    }
}

enum SpeechMode: String, CaseIterable, Identifiable {
    case dictate
    case speak

    var id: String { rawValue }

    var title: String {
        switch self {
        case .dictate: "Dyktowanie"
        case .speak: "Czytanie na głos"
        }
    }
}

/// The two-way switch in the header, styled like `ModelsFeature`'s own
/// segmented control (that one is internal to its module, so this is a small
/// copy rather than a shared dependency).
private struct ModeSwitch: View {
    @Binding var mode: SpeechMode

    var body: some View {
        HStack(spacing: 2) {
            ForEach(SpeechMode.allCases) { item in
                let isActive = item == mode
                Button {
                    mode = item
                } label: {
                    Text(item.title)
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


#if DEBUG
/// The contents of "Mowa" laid out flat, for rendering to an image.
///
/// `ImageRenderer` does not lay out a `ScrollView` — there is no window to
/// scroll in — so the screen itself renders as a header above an empty box.
/// This shows the same sections in a plain stack, which is what the layout
/// check actually needs. Debug only; see `macos/Sources/ScreenShots`.
public struct SpeechScreenPreview: View {
    private let showsSpeaking: Bool

    public init(speaking: Bool = false) {
        self.showsSpeaking = speaking
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if showsSpeaking { SpeakSection() } else { DictationSection() }
        }
        .padding(20)
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(Theme.bg)
    }
}
#endif

