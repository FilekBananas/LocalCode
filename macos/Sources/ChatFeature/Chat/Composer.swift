import Foundation
import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit

/// The input bar at the bottom of the transcript: a growing text field, a file
/// picker, a compact model picker and one button that flips between "send" and
/// "stop".
///
/// Attachments are uploaded the moment they are picked, so the send button
/// stays blocked until every upload has finished — the server only accepts
/// attachment ids that already exist.
struct Composer: View {
    @Binding var text: String

    let api: APIClient
    var isStreaming: Bool = false
    /// Another conversation is streaming, so the server would refuse this turn.
    var isBlocked: Bool = false
    var models: [ModelInfo] = []
    var providers: [ProviderInfo] = []
    var currentProvider: String = ""
    var currentModel: String = ""
    /// Bumped by the caller to move the caret into the field.
    var focusRequest: Int = 0

    var onSelectModel: (String, String) -> Void = { _, _ in }
    var onSend: (String, [Attachment]) -> Void = { _, _ in }
    var onStop: () -> Void = {}

    @State private var attachments: [ComposerAttachment] = []
    @State private var showFileImporter = false
    @State private var uploadError: String?
    @FocusState private var isFocused: Bool

    private var trimmed: String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var isUploading: Bool {
        attachments.contains { $0.status == .uploading }
    }

    private var readyAttachments: [Attachment] {
        attachments.compactMap(\.uploaded)
    }

    private var canSend: Bool {
        guard !isStreaming, !isBlocked, !isUploading else { return false }
        return !trimmed.isEmpty || !readyAttachments.isEmpty
    }

    // MARK: - Body

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let uploadError {
                Banner(.error, message: uploadError, onDismiss: { self.uploadError = nil })
            }

            if !attachments.isEmpty {
                FlowLayout(spacing: 6, lineSpacing: 6) {
                    ForEach(attachments) { item in
                        ComposerAttachmentChip(item: item) { remove(item.id) }
                    }
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                IconButton(systemName: "paperclip", label: "Dołącz plik", size: 32, fontSize: 15) {
                    showFileImporter = true
                }
                ComposerTextInput(text: $text,
                                  placeholder: isStreaming ? "Napisz kolejną wiadomość…" : "Napisz wiadomość…",
                                  isFocused: $isFocused,
                                  onSubmit: submit)
                sendButton
            }

            HStack(spacing: 8) {
                ModelMenu(models: models,
                          providers: providers,
                          currentProvider: currentProvider,
                          currentModel: currentModel,
                          style: .capsule,
                          onSelect: onSelectModel)
                if isUploading {
                    HStack(spacing: 5) {
                        ProgressView().controlSize(.small).scaleEffect(0.7).frame(width: 12, height: 12)
                        Text("Wgrywam załączniki…")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                } else if isBlocked {
                    Text("Trwa odpowiedź w innej rozmowie…")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
                Text("Enter wysyła · Shift+Enter nowa linia")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                    .lineLimit(1)
            }
        }
        .padding(10)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(isFocused ? Theme.accent.opacity(0.55) : Theme.border, lineWidth: 1)
        )
        .animation(.easeOut(duration: 0.15), value: isFocused)
        .animation(.easeOut(duration: 0.15), value: attachments.count)
        .onChange(of: focusRequest) { _, _ in isFocused = true }
        .fileImporter(isPresented: $showFileImporter,
                      allowedContentTypes: Self.allowedTypes,
                      allowsMultipleSelection: true) { result in
            switch result {
            case let .success(urls):
                for url in urls { readFile(at: url) }
            case let .failure(error):
                uploadError = error.localizedDescription
            }
        }
    }

    // MARK: - Send / stop

    private var sendButton: some View {
        Button {
            if isStreaming { onStop() } else { submit() }
        } label: {
            Image(systemName: isStreaming ? "stop.fill" : "arrow.up")
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(Theme.accentFg)
                .frame(width: 34, height: 34)
                .background(
                    Circle().fill(isStreaming ? Theme.danger : Theme.accent.opacity(canSend ? 1 : 0.35))
                )
                .contentShape(Circle())
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(!isStreaming && !canSend)
        .help(isStreaming ? "Zatrzymaj" : "Wyślij")
        .accessibilityLabel(Text(isStreaming ? "Zatrzymaj" : "Wyślij"))
    }

    private func submit() {
        guard canSend else { return }
        let content = trimmed
        let files = readyAttachments
        text = ""
        attachments = []
        onSend(content, files)
    }

    // MARK: - Attachments

    private func remove(_ id: UUID) {
        attachments.removeAll { $0.id == id }
    }

    private func readFile(at url: URL) {
        do {
            let file = try MultipartFile.read(contentsOf: url)
            upload(file)
        } catch {
            uploadError = "Nie udało się odczytać pliku \(url.lastPathComponent)."
        }
    }

    private func upload(_ file: MultipartFile) {
        let localId = UUID()
        attachments.append(ComposerAttachment(id: localId,
                                              name: file.name,
                                              size: file.data.count,
                                              mime: file.mime,
                                              status: .uploading,
                                              uploaded: nil))
        Task {
            do {
                let result = try await api.upload(files: [file])
                guard let stored = result.first else {
                    mark(localId, failed: "Serwer nie zwrócił załącznika.")
                    return
                }
                guard let index = attachments.firstIndex(where: { $0.id == localId }) else { return }
                attachments[index].uploaded = stored
                attachments[index].name = stored.name
                attachments[index].size = stored.size
                attachments[index].mime = stored.mime
                attachments[index].status = .ready
            } catch {
                mark(localId, failed: APIError.describe(error))
            }
        }
    }

    private func mark(_ id: UUID, failed reason: String) {
        guard let index = attachments.firstIndex(where: { $0.id == id }) else { return }
        attachments[index].status = .failed
        uploadError = reason
    }

    private static let allowedTypes: [UTType] = [
        .image, .pdf, .plainText, .text, .sourceCode, .json,
        .commaSeparatedText, .html, .xml, .rtf, .data
    ]
}

// MARK: - Pending attachment

/// A file the composer is holding: uploading, ready to send, or failed.
struct ComposerAttachment: Identifiable {
    enum Status: Equatable { case uploading, ready, failed }

    let id: UUID
    var name: String
    var size: Int
    var mime: String
    var status: Status
    var uploaded: Attachment?
}

private struct ComposerAttachmentChip: View {
    let item: ComposerAttachment
    let onRemove: () -> Void

    private var tint: Color {
        switch item.status {
        case .failed: Theme.danger
        case .uploading: Theme.fgSubtle
        case .ready: Theme.fgMuted
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            switch item.status {
            case .uploading:
                ProgressView().controlSize(.small).scaleEffect(0.65).frame(width: 12, height: 12)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 10, weight: .semibold))
            case .ready:
                Image(systemName: AttachmentIcon.name(kind: nil, mime: item.mime, fileName: item.name))
                    .font(.system(size: 10, weight: .semibold))
            }

            Text(item.name)
                .font(.system(size: 12, weight: .medium))
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: 160, alignment: .leading)

            Text(item.status == .failed ? "błąd" : Format.bytes(item.size))
                .font(.system(size: 11))
                .foregroundStyle(item.status == .failed ? Theme.danger : Theme.fgSubtle)

            Button(action: onRemove) {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .frame(width: 16, height: 16)
                    .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
            .accessibilityLabel(Text("Usuń załącznik"))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(item.status == .failed ? Theme.dangerSoft : Theme.surface2)
        .clipShape(Capsule())
        .overlay(
            Capsule().strokeBorder(item.status == .failed ? Theme.danger.opacity(0.4) : Theme.border, lineWidth: 1)
        )
    }
}

// MARK: - Text input

/// The growing field. Return is handled by hand so Shift+Return can still
/// insert a newline.
///
/// `TextEditor` has no intrinsic height — left alone it swallows every point
/// the composer is offered, which on a tall window means an eight-line input
/// box before a single character is typed. So an invisible `Text` twin holding
/// the same string is measured, and that measurement (clamped to one…eight
/// lines) becomes the editor's height.
private struct ComposerTextInput: View {
    @Binding var text: String
    let placeholder: String
    @FocusState.Binding var isFocused: Bool
    let onSubmit: () -> Void

    @State private var measuredHeight: CGFloat = 0

    private let fontSize: CGFloat = 14.5
    private let lineSpacing: CGFloat = 3
    private let maxLines: CGFloat = 8

    private var font: Font { .system(size: fontSize) }
    private var lineHeight: CGFloat { fontSize * 1.25 + lineSpacing }
    private var minHeight: CGFloat { 34 }
    private var maxHeight: CGFloat { maxLines * lineHeight + 16 }

    private var height: CGFloat {
        min(max(measuredHeight, minHeight), maxHeight)
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            // Invisible twin that measures the text, wrapping included.
            Text(text.isEmpty ? " " : text + "\n")
                .font(font)
                .lineSpacing(lineSpacing)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 5)
                .padding(.vertical, 8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    GeometryReader { geometry in
                        Color.clear.preference(key: ComposerHeightKey.self, value: geometry.size.height)
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
                .padding(.vertical, 8)
                .onKeyPress(.return, phases: .down) { press in
                    if press.modifiers.contains(.shift) { return .ignored }
                    onSubmit()
                    return .handled
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
        .onPreferenceChange(ComposerHeightKey.self) { value in
            Task { @MainActor in
                if abs(measuredHeight - value) > 0.5 { measuredHeight = value }
            }
        }
    }
}

private struct ComposerHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
