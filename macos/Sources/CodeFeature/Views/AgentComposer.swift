import SwiftUI
import AppKit
import UniformTypeIdentifiers
import LocalCodeKit

/// The input bar of the agent: a growing field, slash-command hints, and one
/// button that flips between "send" and "stop".
struct AgentComposer: View {
    @Binding var text: String

    let isBusy: Bool
    let isEnabled: Bool
    /// A session is open — the field is blocked for another reason (a pending
    /// permission), so the placeholder must not say "open a session".
    let hasSession: Bool
    let queuedCount: Int
    let blockedReason: String?
    /// Served by `GET /api/agent/commands`; passed in rather than read from the
    /// store so this view stays a plain function of its inputs.
    let commands: [AgentSlashCommand]
    /// What the empty field says. The coding agent's default mentions the
    /// project and `@plik`; the help bot has neither, so it passes its own.
    var idlePlaceholder: String = "Co mam zrobić w tym projekcie?  (/ — polecenia, @plik — załącz)"
    var focusRequest: Int = 0
    /// Pictures waiting to go with the next message.
    var images: Binding<[AgentImage]> = .constant([])
    /// The attach button, drag-and-drop and pasting pictures. Off for the help
    /// bot, whose small model cannot see them.
    var allowsImages: Bool = false
    /// Whether the session's model takes pictures (`nil`: not known here).
    var seesImages: Bool? = nil

    let onSend: (String, [AgentImage]) -> Void
    let onStop: () -> Void

    @FocusState private var isFocused: Bool
    @State private var isDropTargeted = false

    private var trimmed: String { text.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// Pictures attached to a model that cannot see them: nothing is sent.
    private var blockedByImages: Bool { !images.wrappedValue.isEmpty && seesImages == false }

    private var canSend: Bool {
        isEnabled && !blockedByImages && (!trimmed.isEmpty || !images.wrappedValue.isEmpty)
    }

    private var suggestions: [AgentSlashCommand] {
        guard isEnabled else { return [] }
        let value = text.trimmingCharacters(in: .whitespaces)
        guard value.hasPrefix("/"), !value.contains(" ") else { return [] }
        let prefix = value.dropFirst().lowercased()
        // The list comes from the server, which serves the same table the
        // terminal reads — so a command added once shows up in both.
        return commands
            .filter { prefix.isEmpty || $0.name.hasPrefix(prefix) }
            .sorted { $0.name < $1.name }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if !suggestions.isEmpty {
                slashHints
            }

            VStack(alignment: .leading, spacing: 8) {
                if !images.wrappedValue.isEmpty {
                    attachedImages
                }
                HStack(alignment: .bottom, spacing: 8) {
                    if allowsImages {
                        attachButton
                    } else {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(Theme.fgSubtle)
                            .padding(.bottom, 9)
                    }
                    AgentTextInput(text: $text,
                                   placeholder: placeholder,
                                   isEnabled: isEnabled,
                                   isFocused: $isFocused,
                                   onSubmit: submit,
                                   onComplete: completeSuggestion,
                                   onPasteImage: allowsImages ? pasteImages : nil)
                    sendButton
                }
                if blockedByImages {
                    Label("Ten model nie widzi obrazów. Wybierz model z wizją (np. Qwen3‑VL albo Kimi K2.6) "
                          + "albo usuń obrazki, żeby wysłać samą wiadomość.",
                          systemImage: "eye.slash")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }

                HStack(spacing: 8) {
                    if queuedCount > 0 {
                        Label(Format.plural(queuedCount, "wiadomość w kolejce",
                                            "wiadomości w kolejce", "wiadomości w kolejce"),
                              systemImage: "tray.full")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.warning)
                    }
                    if let blockedReason {
                        // The only explanation of why typing is blocked, so it
                        // gets the line: the key hint steps aside instead of
                        // both being cut to "Pomocnik czeka na Tw…".
                        Text(blockedReason)
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                    if blockedReason == nil {
                        Text("Enter wysyła · Shift+Enter nowa linia · / komendy")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                            .lineLimit(1)
                    }
                }
            }
            .padding(10)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(isDropTargeted ? Theme.accent
                                  : (isFocused ? Theme.accent.opacity(0.55) : Theme.border),
                                  lineWidth: isDropTargeted ? 2 : 1)
            )
            .animation(.easeOut(duration: 0.15), value: isFocused)
            .onDrop(of: allowsImages ? [.fileURL, .image] : [], isTargeted: $isDropTargeted) { providers in
                dropImages(providers)
            }
        }
        .onChange(of: focusRequest) { _, _ in isFocused = true }
    }

    private var placeholder: String {
        if !hasSession { return "Otwórz sesję, aby rozmawiać z agentem…" }
        if !isEnabled { return "Odpowiedz na prośbę o zgodę, aby pisać dalej…" }
        // The @ hint earns its place: nothing else in the window says that a
        // file can be attached by naming it.
        return isBusy ? "Dopisz kolejne polecenie — wyślę je po bieżącej turze…"
                      : idlePlaceholder
    }

    // MARK: - Slash hints

    private var slashHints: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(suggestions) { command in
                Button {
                    text = "/\(command.name)" + (command.takesArgument ? " " : "")
                    isFocused = true
                } label: {
                    HStack(spacing: 8) {
                        Text("/\(command.name)")
                            .font(.system(size: 12, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Theme.accent)
                        Text(command.summary)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                        if let usage = command.usage {
                            Text(usage)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundStyle(Theme.fgSubtle)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(PressableButtonStyle())
            }
        }
        .padding(.vertical, 3)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.10), radius: 8, y: 2)
    }

    private func completeSuggestion() {
        guard let first = suggestions.first else { return }
        text = "/\(first.name)" + (first.takesArgument ? " " : "")
    }

    // MARK: - Pictures

    private var attachButton: some View {
        Button { chooseImages() } label: {
            Image(systemName: "photo.badge.plus")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(Theme.fgMuted)
                .frame(width: 26, height: 26)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .padding(.bottom, 4)
        .disabled(!isEnabled)
        .help(seesImages == false
              ? "Dodaj obrazek — ten model ich nie widzi, wybierz model z wizją"
              : "Dodaj obrazek (albo przeciągnij go tutaj lub wklej ⌘V)")
        .accessibilityLabel(Text("Dodaj obrazek"))
    }

    private var attachedImages: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(images.wrappedValue) { image in
                    ZStack(alignment: .topTrailing) {
                        Group {
                            if let thumb = NSImage(data: image.thumbnail) {
                                Image(nsImage: thumb).resizable().scaledToFill()
                            } else {
                                Image(systemName: "photo").foregroundStyle(Theme.fgSubtle)
                            }
                        }
                        .frame(width: 56, height: 56)
                        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(Theme.border, lineWidth: 1))
                        .help(image.name)
                        Button {
                            images.wrappedValue.removeAll { $0.id == image.id }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 14))
                                .symbolRenderingMode(.palette)
                                .foregroundStyle(.white, Color.black.opacity(0.65))
                        }
                        .buttonStyle(.plain)
                        .offset(x: 5, y: -5)
                        .help("Usuń obrazek")
                    }
                }
            }
            .padding(.top, 6)
            .padding(.horizontal, 2)
        }
    }

    private func add(_ prepared: [AgentImage]) {
        let room = AgentImage.maxPerMessage - images.wrappedValue.count
        guard room > 0 else { return }
        images.wrappedValue.append(contentsOf: prepared.prefix(room))
    }

    private func chooseImages() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = AgentImage.fileTypes
        panel.prompt = "Dodaj"
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        Task.detached(priority: .userInitiated) {
            let prepared = urls.compactMap { AgentImage.from(url: $0) }
            await MainActor.run { add(prepared) }
        }
    }

    private func dropImages(_ providers: [NSItemProvider]) -> Bool {
        var handled = false
        for provider in providers {
            if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                handled = true
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url, AgentImage.isImage(url), let image = AgentImage.from(url: url) else { return }
                    Task { @MainActor in add([image]) }
                }
            } else if provider.canLoadObject(ofClass: NSImage.self) {
                handled = true
                _ = provider.loadObject(ofClass: NSImage.self) { object, _ in
                    guard let picture = object as? NSImage,
                          let image = AgentImage.from(image: picture, name: "obraz") else { return }
                    Task { @MainActor in add([image]) }
                }
            }
        }
        return handled
    }

    /// ⌘V with a picture (or a copied image file) on the clipboard; `false` leaves
    /// the paste to the text field.
    private func pasteImages() -> Bool {
        let board = NSPasteboard.general
        if let urls = board.readObjects(forClasses: [NSURL.self],
                                        options: [.urlReadingFileURLsOnly: true]) as? [URL] {
            let pictures = urls.filter(AgentImage.isImage)
            if !pictures.isEmpty {
                add(pictures.compactMap { AgentImage.from(url: $0) })
                return true
            }
        }
        if board.canReadObject(forClasses: [NSImage.self], options: nil),
           board.string(forType: .string) == nil,
           let picture = (board.readObjects(forClasses: [NSImage.self]) as? [NSImage])?.first,
           let image = AgentImage.from(image: picture, name: "wklejony obraz") {
            add([image])
            return true
        }
        return false
    }

    // MARK: - Send / stop

    private var sendButton: some View {
        Button {
            if isBusy { onStop() } else { submit() }
        } label: {
            Image(systemName: isBusy ? "stop.fill" : "arrow.up")
                .font(.system(size: 13, weight: .bold))
                .foregroundStyle(Theme.accentFg)
                .frame(width: 32, height: 32)
                .background(Circle().fill(isBusy ? Theme.danger
                                                 : Theme.accent.opacity(canSend ? 1 : 0.35)))
                .contentShape(Circle())
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(!isBusy && !canSend)
        .help(isBusy ? "Zatrzymaj (Esc)" : "Wyślij (⏎)")
        .accessibilityLabel(Text(isBusy ? "Zatrzymaj" : "Wyślij"))
    }

    private func submit() {
        guard canSend else { return }
        let value = trimmed
        let pictures = images.wrappedValue
        text = ""
        images.wrappedValue = []
        onSend(value, pictures)
    }
}

// MARK: - Growing text field

/// The growing input. Return is handled by hand so Shift+Return can still
/// insert a newline, and Tab completes a slash command.
///
/// `TextEditor` has no intrinsic height, so an invisible `Text` twin holding
/// the same string is measured and that measurement (clamped to one…ten lines)
/// becomes the editor's height — the same trick the chat composer uses.
private struct AgentTextInput: View {
    @Binding var text: String
    let placeholder: String
    let isEnabled: Bool
    @FocusState.Binding var isFocused: Bool
    let onSubmit: () -> Void
    let onComplete: () -> Void
    /// ⌘V: returns `true` when it took a picture off the clipboard.
    var onPasteImage: (() -> Bool)? = nil

    @State private var measuredHeight: CGFloat = 0

    private let fontSize: CGFloat = 14
    private let lineSpacing: CGFloat = 3
    private let maxLines: CGFloat = 10

    private var font: Font { .system(size: fontSize) }
    private var lineHeight: CGFloat { fontSize * 1.25 + lineSpacing }
    private var minHeight: CGFloat { 34 }
    private var maxHeight: CGFloat { maxLines * lineHeight + 16 }
    private var height: CGFloat { min(max(measuredHeight, minHeight), maxHeight) }

    /// What the invisible twin measures. A trailing newline is padded with a
    /// space so the empty line the caret sits on is actually reserved; anything
    /// else is measured as typed, or the field would always be one line too tall.
    private var measuredText: String {
        if text.isEmpty { return " " }
        return text.hasSuffix("\n") ? text + " " : text
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text(measuredText)
                .font(font)
                .lineSpacing(lineSpacing)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 5)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    GeometryReader { geometry in
                        Color.clear.preference(key: AgentComposerHeightKey.self,
                                               value: geometry.size.height)
                    }
                )
                .opacity(0)
                .accessibilityHidden(true)

            TextEditor(text: $text)
                .font(font)
                .lineSpacing(lineSpacing)
                .foregroundStyle(Theme.fg)
                .scrollContentBackground(.hidden)
                .background(Color.clear)
                .focused($isFocused)
                .disabled(!isEnabled)
                .padding(.vertical, 8)
                .onKeyPress(.return, phases: .down) { press in
                    if press.modifiers.contains(.shift) { return .ignored }
                    onSubmit()
                    return .handled
                }
                .onKeyPress(.tab, phases: .down) { _ in
                    guard text.trimmingCharacters(in: .whitespaces).hasPrefix("/") else { return .ignored }
                    onComplete()
                    return .handled
                }
                .onKeyPress(KeyEquivalent("v"), phases: .down) { press in
                    guard press.modifiers.contains(.command), let onPasteImage else { return .ignored }
                    return onPasteImage() ? .handled : .ignored
                }

            if text.isEmpty {
                Text(placeholder)
                    .font(font)
                    .foregroundStyle(Theme.fgSubtle)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 8)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: height)
        .onPreferenceChange(AgentComposerHeightKey.self) { value in
            Task { @MainActor in
                if abs(measuredHeight - value) > 0.5 { measuredHeight = value }
            }
        }
    }
}

private struct AgentComposerHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

#if DEBUG
/// The composer with two pictures attached, for `swift run ScreenShots`.
public struct AgentComposerPreview: View {
    @State private var text = "Co jest na tych zrzutach?"
    @State private var images: [AgentImage]
    private let seesImages: Bool

    public init(seesImages: Bool) {
        self.seesImages = seesImages
        func picture(_ color: NSColor, _ name: String) -> AgentImage {
            let image = NSImage(size: NSSize(width: 400, height: 300), flipped: false) { rect in
                color.setFill()
                rect.fill()
                return true
            }
            return AgentImage.from(image: image, name: name)!
        }
        _images = State(initialValue: [picture(.systemRed, "zrzut1.png"), picture(.systemBlue, "zrzut2.png")])
    }

    public var body: some View {
        AgentComposer(text: $text, isBusy: false, isEnabled: true, hasSession: true, queuedCount: 0,
                      blockedReason: nil, commands: [], images: $images, allowsImages: true,
                      seesImages: seesImages, onSend: { _, _ in }, onStop: {})
            .padding(16)
            .background(Theme.bg)
    }
}
#endif
