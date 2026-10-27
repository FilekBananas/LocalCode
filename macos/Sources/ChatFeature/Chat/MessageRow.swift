import Foundation
import SwiftUI
import AppKit
import LocalCodeKit

/// One turn in the transcript.
///
/// The user's turn is a right-aligned bubble; the assistant's turn is a
/// full-width block with a model header, an optional reasoning panel, a
/// markdown body and source chips. Every action is disabled while any answer is
/// streaming, because the store refuses those calls anyway.
struct MessageRow: View {
    let message: Message
    let api: APIClient
    /// Last row of the transcript — regeneration and retry only make sense here.
    var isLast: Bool = false
    /// This exact message is the one currently receiving tokens.
    var isStreamingMessage: Bool = false
    /// Any stream is running in this conversation.
    var isBusy: Bool = false
    /// Mirrors `settings.defaults.showReasoning`.
    var showReasoning: Bool = true
    /// Pretty model name resolved by the caller.
    var modelLabel: String?
    /// What the server is doing before the first token ("Ładowanie modelu…").
    var statusText: String?

    var onRegenerate: () -> Void = {}
    var onEdit: (String) -> Void = { _ in }
    var onDelete: () -> Void = {}
    var onRetry: () -> Void = {}

    @State private var isHovering = false
    @State private var isEditing = false
    @State private var editText = ""
    @State private var didCopy = false
    @State private var copyToken = 0
    @State private var confirmDelete = false
    @State private var expandedCitation: String?
    @State private var previewedAttachment: Attachment?

    // MARK: - Capabilities

    private var isUser: Bool { message.role == .user }
    private var canEdit: Bool { isUser && !isBusy && !message.id.hasPrefix("local-") }
    private var canRegenerate: Bool { !isUser && isLast && !isBusy }
    private var canDelete: Bool { !isBusy && !message.id.hasPrefix("local-") }

    private var showActions: Bool {
        if isEditing || isStreamingMessage { return false }
        return isHovering || isLast
    }

    // MARK: - Body

    var body: some View {
        Group {
            if isUser {
                userTurn
            } else {
                assistantTurn
            }
        }
        .onHover { isHovering = $0 }
        .animation(.easeOut(duration: 0.15), value: showActions)
        .task(id: copyToken) {
            guard didCopy else { return }
            try? await Task.sleep(nanoseconds: 1_400_000_000)
            if !Task.isCancelled { didCopy = false }
        }
        .alert("Usunąć wiadomość?", isPresented: $confirmDelete) {
            Button("Anuluj", role: .cancel) {}
            Button("Usuń", role: .destructive, action: onDelete)
        } message: {
            Text("Tej operacji nie można cofnąć.")
        }
        .sheet(item: $previewedAttachment) { attachment in
            MessageAttachmentPreview(attachment: attachment, api: api)
        }
    }

    // MARK: - User turn

    private var userTurn: some View {
        HStack(alignment: .bottom, spacing: 0) {
            Spacer(minLength: 48)
            VStack(alignment: .trailing, spacing: 6) {
                if isEditing {
                    editor
                } else {
                    bubble
                    if !message.attachments.isEmpty {
                        attachmentStrip(alignment: .trailing)
                    }
                    if message.status == .error, let error = message.error {
                        errorBox(error)
                    }
                    actionBar
                }
            }
            .frame(maxWidth: 600, alignment: .trailing)
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    private var bubble: some View {
        Text(message.content)
            .font(.system(size: 15))
            .foregroundStyle(Theme.fg)
            .lineSpacing(4)
            .multilineTextAlignment(.leading)
            .textSelection(.enabled)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Theme.bubbleUser)
            .clipShape(RoundedRectangle(cornerRadius: Theme.bubbleRadius, style: .continuous))
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 8) {
            StyledTextEditor(placeholder: "Treść wiadomości", text: $editText, minHeight: 90, maxHeight: 260)
            HStack(spacing: 8) {
                SecondaryButton(title: "Anuluj", height: 34) { isEditing = false }
                PrimaryButton(title: "Zapisz i wyślij", fullWidth: false, height: 34) {
                    let text = editText
                    isEditing = false
                    onEdit(text)
                }
            }
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Assistant turn

    private var assistantTurn: some View {
        VStack(alignment: .leading, spacing: 8) {
            assistantHeader

            if showReasoning, let reasoning = message.reasoning,
               !reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                ReasoningView(text: reasoning, isStreaming: isStreamingMessage && message.content.isEmpty)
            }

            if message.content.isEmpty {
                if isStreamingMessage {
                    waitingIndicator
                        .padding(.vertical, 4)
                }
            } else {
                MarkdownView(message.content)
            }

            if message.status == .stopped {
                Text("Zatrzymano")
                    .font(.system(size: 12).italic())
                    .foregroundStyle(Theme.fgSubtle)
            }

            if message.status == .error {
                errorBox(message.error ?? "Nie udało się dokończyć odpowiedzi.")
            }

            if !message.attachments.isEmpty {
                attachmentStrip(alignment: .leading)
            }

            if !uniqueCitations.isEmpty {
                citations
            }

            actionBar
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var assistantHeader: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(Theme.accent)
                .frame(width: 7, height: 7)
            Text(modelLabel ?? message.model ?? "Asystent")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
            if message.status == .complete, message.createdAt > 0 {
                Text("·")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                Text(Format.time(message.createdAt))
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
            }
            Spacer(minLength: 0)
        }
    }

    /// Before the first token: the server's own status line when it sent one
    /// ("Ładowanie modelu…" — a cold GGUF takes seconds), otherwise three dots.
    @ViewBuilder
    private var waitingIndicator: some View {
        if let statusText, !statusText.isEmpty {
            HStack(spacing: 7) {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.7)
                    .frame(width: 12, height: 12)
                Text(statusText)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            .transition(.opacity)
        } else {
            ChatTypingIndicator()
        }
    }

    // MARK: - Citations

    private var uniqueCitations: [Citation] {
        var seen = Set<String>()
        return message.citations.filter { seen.insert($0.documentId).inserted }
    }

    private var citations: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Źródła")
                .font(.system(size: 11, weight: .semibold))
                .textCase(.uppercase)
                .kerning(0.5)
                .foregroundStyle(Theme.fgSubtle)

            FlowLayout(spacing: 6, lineSpacing: 6) {
                ForEach(uniqueCitations, id: \.documentId) { citation in
                    Chip(title: citation.documentName,
                         systemImage: "doc.text",
                         isSelected: expandedCitation == citation.documentId,
                         onTap: {
                             expandedCitation = expandedCitation == citation.documentId ? nil : citation.documentId
                         })
                }
            }

            if let id = expandedCitation,
               let citation = message.citations.first(where: { $0.documentId == id }) {
                Text(citation.snippet)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Theme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
        }
        .padding(.top, 2)
    }

    // MARK: - Attachments

    private func attachmentStrip(alignment: HorizontalAlignment) -> some View {
        FlowLayout(spacing: 8, lineSpacing: 8, alignment: alignment) {
            ForEach(message.attachments) { attachment in
                MessageAttachmentThumbnail(attachment: attachment, api: api) { tapped in
                    if tapped.isImage { previewedAttachment = tapped }
                }
            }
        }
    }

    // MARK: - Error

    private func errorBox(_ text: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.danger)
                    .padding(.top, 1)
                Text(text)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fg)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
            }
            if isLast, !isBusy {
                SecondaryButton(title: "Spróbuj ponownie", systemImage: "arrow.clockwise", height: 32, action: onRetry)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.dangerSoft)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.danger.opacity(0.45), lineWidth: 1)
        )
    }

    // MARK: - Actions

    /// The bar keeps its height even when hidden, so hovering never nudges the
    /// transcript under the reader's cursor.
    private var actionBar: some View {
        HStack(spacing: 2) {
            IconButton(systemName: didCopy ? "checkmark" : "doc.on.doc",
                       label: "Kopiuj",
                       tint: didCopy ? Theme.success : Theme.fgSubtle,
                       size: 26, fontSize: 12, isEnabled: !isBusy) {
                copyContent()
            }
            if canRegenerate {
                IconButton(systemName: "arrow.clockwise", label: "Regeneruj",
                           tint: Theme.fgSubtle, size: 26, fontSize: 12,
                           isEnabled: !isBusy, action: onRegenerate)
            }
            if isUser {
                IconButton(systemName: "pencil", label: "Edytuj",
                           tint: Theme.fgSubtle, size: 26, fontSize: 12,
                           isEnabled: canEdit) {
                    beginEditing()
                }
            }
            IconButton(systemName: "trash", label: "Usuń",
                       tint: Theme.fgSubtle, size: 26, fontSize: 12,
                       isEnabled: canDelete) {
                confirmDelete = true
            }
        }
        .opacity(showActions ? 1 : 0)
        .allowsHitTesting(showActions)
    }

    private func copyContent() {
        Pasteboard.copy(message.content)
        didCopy = true
        copyToken &+= 1
    }

    private func beginEditing() {
        editText = message.content
        isEditing = true
    }
}

// MARK: - Typing indicator

/// Three pulsing dots shown before the first token arrives.
struct ChatTypingIndicator: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Theme.fgSubtle)
                    .frame(width: 6, height: 6)
                    .opacity(phase == index ? 1 : 0.28)
            }
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 320_000_000)
                if Task.isCancelled { return }
                withAnimation(.easeInOut(duration: 0.24)) { phase = (phase + 1) % 3 }
            }
        }
        .accessibilityLabel(Text("Asystent pisze"))
    }
}

// MARK: - Attachments

/// Attachment images sit behind the bearer token, so they are fetched by hand
/// and kept in a tiny process-wide cache — a streaming answer re-renders its
/// row dozens of times a second and must not refetch each time.
@MainActor
final class AttachmentImageLoader {
    static let shared = AttachmentImageLoader()

    private var cache: [String: Image] = [:]
    private var failed: Set<String> = []

    func cached(_ id: String) -> Image? { cache[id] }

    func load(_ id: String, api: APIClient) async -> Image? {
        if let image = cache[id] { return image }
        if failed.contains(id) { return nil }
        do {
            let data = try await api.attachmentData(id: id)
            guard let nsImage = NSImage(data: data) else {
                failed.insert(id)
                return nil
            }
            let image = Image(nsImage: nsImage)
            cache[id] = image
            return image
        } catch {
            failed.insert(id)
            return nil
        }
    }
}

/// Square preview for images, a compact file card for everything else.
struct MessageAttachmentThumbnail: View {
    let attachment: Attachment
    let api: APIClient
    var onTap: (Attachment) -> Void = { _ in }

    @State private var image: Image?
    @State private var didFail = false

    private let side: CGFloat = 92

    var body: some View {
        Group {
            if attachment.isImage {
                imageTile
            } else {
                fileTile
            }
        }
        .task(id: attachment.id) {
            guard attachment.isImage else { return }
            if let cached = AttachmentImageLoader.shared.cached(attachment.id) {
                image = cached
                return
            }
            let loaded = await AttachmentImageLoader.shared.load(attachment.id, api: api)
            if loaded == nil { didFail = true } else { image = loaded }
        }
    }

    private var imageTile: some View {
        Button {
            onTap(attachment)
        } label: {
            ZStack {
                Theme.surface2
                if let image {
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else if didFail {
                    Image(systemName: "photo.badge.exclamationmark")
                        .font(.system(size: 18))
                        .foregroundStyle(Theme.fgSubtle)
                } else {
                    ProgressView().controlSize(.small)
                }
            }
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .strokeBorder(Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .accessibilityLabel(Text(attachment.name))
    }

    private var fileTile: some View {
        HStack(spacing: 8) {
            Image(systemName: AttachmentIcon.name(for: attachment))
                .font(.system(size: 15))
                .foregroundStyle(Theme.accent)
            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.name)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                Text(Format.bytes(attachment.size))
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .frame(maxWidth: 260, alignment: .leading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}

/// Full-size look at one image attachment.
struct MessageAttachmentPreview: View {
    let attachment: Attachment
    let api: APIClient
    @Environment(\.dismiss) private var dismiss

    @State private var image: Image?
    @State private var didFail = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text(attachment.name)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                Spacer(minLength: 8)
                Text(Format.bytes(attachment.size))
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
                IconButton(systemName: "xmark", label: "Zamknij") { dismiss() }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(Theme.surface)

            Hairline()

            ZStack {
                Theme.bg
                if let image {
                    ScrollView([.horizontal, .vertical]) {
                        image
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity)
                            .padding(16)
                    }
                } else if didFail {
                    EmptyState(systemImage: "photo.badge.exclamationmark",
                               title: "Nie udało się wczytać obrazu",
                               message: "Spróbuj ponownie za chwilę.")
                } else {
                    ProgressView().controlSize(.large)
                }
            }
        }
        .frame(minWidth: 420, minHeight: 360)
        .background(Theme.bg)
        .task(id: attachment.id) {
            if let cached = AttachmentImageLoader.shared.cached(attachment.id) {
                image = cached
                return
            }
            let loaded = await AttachmentImageLoader.shared.load(attachment.id, api: api)
            if loaded == nil { didFail = true } else { image = loaded }
        }
    }
}

/// SF Symbol for an attachment, picked from its mime type.
enum AttachmentIcon {
    static func name(for attachment: Attachment) -> String {
        name(kind: attachment.kind, mime: attachment.mime, fileName: attachment.name)
    }

    static func name(kind: String?, mime: String, fileName: String) -> String {
        if kind == "image" || mime.hasPrefix("image/") { return "photo" }
        if mime.contains("pdf") { return "doc.richtext" }
        if mime.hasPrefix("audio/") { return "waveform" }
        if mime.hasPrefix("video/") { return "film" }
        if mime.contains("zip") || mime.contains("compressed") { return "archivebox" }
        if mime.contains("csv") || mime.contains("sheet") || mime.contains("excel") { return "tablecells" }
        if mime.hasPrefix("text/") || mime.contains("json") || mime.contains("xml") { return "doc.text" }
        let ext = String(fileName.split(separator: ".").last ?? "").lowercased()
        if ["swift", "js", "ts", "py", "rb", "go", "rs", "java", "kt", "c", "cpp", "h"].contains(ext) {
            return "chevron.left.forwardslash.chevron.right"
        }
        return "paperclip"
    }
}
