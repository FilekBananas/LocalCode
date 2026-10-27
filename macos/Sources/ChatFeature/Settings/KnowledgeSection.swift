import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit

/// Global knowledge base (RAG). Files are split into chunks server-side and
/// pulled into the prompt when they match the question — which on a small
/// context window is the difference between an answer grounded in your files
/// and one that invents them.
struct KnowledgeSection: View {
    let api: APIClient

    /// What the server can ingest; anything else is refused with a message.
    private static let allowedTypes: [UTType] = {
        var types: [UTType] = [.plainText, .text, .pdf, .json, .commaSeparatedText, .html, .xml]
        for ext in ["md", "markdown", "docx", "yaml", "yml", "log", "rtf"] {
            if let type = UTType(filenameExtension: ext) { types.append(type) }
        }
        return types
    }()

    @State private var documents: [KbDocument] = []
    @State private var isLoading = true
    @State private var loadError: String?
    @State private var actionError: String?
    @State private var isUploading = false
    @State private var isImporting = false
    @State private var busyId: String?
    @State private var confirmRemoval: ConfirmRequest?

    private var hasProcessing: Bool { documents.contains { $0.status == "processing" } }
    private var totalChunks: Int { documents.reduce(0) { $0 + $1.chunkCount } }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionHeader("Wiedza globalna",
                          subtitle: "Pliki są dzielone na fragmenty i doklejane do promptu, gdy pasują do pytania. Widzą je rozmowy z dostępem do kontekstu globalnego.")

            if let actionError {
                Banner(.error, message: actionError, onDismiss: { self.actionError = nil })
            }

            dropArea
            list
        }
        .task { await load() }
        .task(id: hasProcessing) { await pollWhileProcessing() }
        .fileImporter(isPresented: $isImporting,
                      allowedContentTypes: Self.allowedTypes,
                      allowsMultipleSelection: true) { result in
            handleImport(result)
        }
        .confirmDialog($confirmRemoval)
    }

    // MARK: - Upload

    private var dropArea: some View {
        VStack(spacing: 8) {
            Image(systemName: "arrow.up.doc")
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(Theme.fgSubtle)
            Text("Dodaj pliki do bazy wiedzy")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("PDF, DOCX, TXT, Markdown, CSV, JSON — do 20 MB na plik.")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
            SecondaryButton(title: "Wybierz pliki", systemImage: "plus",
                            isLoading: isUploading, height: 34) {
                isImporting = true
            }
            .padding(.top, 2)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 22)
        .padding(.horizontal, 16)
        .background(Theme.surface2.opacity(0.6),
                    in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
                .foregroundStyle(Theme.borderStrong)
        )
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .failure(let error):
            actionError = APIError.describe(error)
        case .success(let urls):
            guard !urls.isEmpty else { return }
            upload(urls)
        }
    }

    private func upload(_ urls: [URL]) {
        guard !isUploading else { return }
        var files: [MultipartFile] = []
        var unreadable: [String] = []

        for url in urls {
            if let file = try? MultipartFile.read(contentsOf: url) {
                files.append(file)
            } else {
                unreadable.append(url.lastPathComponent)
            }
        }

        if !unreadable.isEmpty {
            actionError = "Nie udało się odczytać: \(unreadable.joined(separator: ", "))."
        }
        guard !files.isEmpty else { return }

        isUploading = true
        Task {
            do {
                let created = try await api.uploadKnowledge(files: files, scope: .global, folderId: nil)
                merge(created)
                let failed = created.filter { $0.status == "error" }
                if let first = failed.first {
                    actionError = first.error ?? "Nie udało się przetworzyć \(failed.count) z \(created.count) plików."
                } else {
                    actionError = nil
                }
            } catch {
                actionError = APIError.describe(error)
            }
            isUploading = false
        }
    }

    // MARK: - List

    @ViewBuilder
    private var list: some View {
        if isLoading {
            HStack {
                Spacer()
                ProgressView().controlSize(.small)
                Text("Wczytywanie plików…")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                Spacer()
            }
            .padding(.vertical, 16)
        } else if let loadError {
            Banner(.error, message: loadError, actionTitle: "Spróbuj ponownie", action: {
                Task { await load() }
            })
        } else if documents.isEmpty {
            Text("Nie dodano jeszcze żadnych plików. Bez nich model korzysta wyłącznie z instrukcji i pamięci.")
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            HStack {
                Text(Format.plural(documents.count, "dokument", "dokumenty", "dokumentów"))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.fgMuted)
                Spacer(minLength: 0)
                Text(Format.plural(totalChunks, "fragment", "fragmenty", "fragmentów"))
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
                    .monospacedDigit()
            }
            VStack(spacing: 8) {
                ForEach(documents) { document in
                    row(document)
                }
            }
        }
    }

    private func row(_ document: KbDocument) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon(for: document))
                .font(.system(size: 13))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 4) {
                Text(document.name)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                    .truncationMode(.middle)

                HStack(spacing: 6) {
                    Text(Format.bytes(document.size))
                    Text("·")
                    Text(Format.plural(document.chunkCount, "fragment", "fragmenty", "fragmentów"))
                    Text("·")
                    Text(Format.relative(document.createdAt))
                }
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)

                status(of: document)
            }

            Spacer(minLength: 4)

            IconButton(systemName: "trash", label: "Usuń plik",
                       tint: Theme.danger, isEnabled: busyId != document.id) {
                confirmRemoval = ConfirmRequest(title: "Usunąć plik z bazy wiedzy?",
                                                message: document.name,
                                                confirmTitle: "Usuń") {
                    remove(document)
                }
            }
            .opacity(busyId == document.id ? 0.5 : 1)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }

    @ViewBuilder
    private func status(of document: KbDocument) -> some View {
        switch document.status {
        case "processing":
            HStack(spacing: 5) {
                ProgressView().controlSize(.small)
                Text("Przetwarzanie…")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
            }
        case "ready":
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11))
                Text("Gotowe")
                    .font(.system(size: 11.5))
            }
            .foregroundStyle(Theme.success)
        default:
            HStack(alignment: .top, spacing: 5) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 11))
                    .padding(.top, 1)
                Text(document.error ?? "Nie udało się przetworzyć pliku.")
                    .font(.system(size: 11.5))
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(Theme.danger)
        }
    }

    private func icon(for document: KbDocument) -> String {
        let mime = document.mime.lowercased()
        let name = document.name.lowercased()
        if mime.hasPrefix("image/") { return "photo" }
        if mime.contains("pdf") || name.hasSuffix(".pdf") { return "doc.richtext" }
        if mime.contains("json") || mime.contains("xml") || mime.contains("html")
            || name.hasSuffix(".json") || name.hasSuffix(".yaml") || name.hasSuffix(".yml") {
            return "chevron.left.forwardslash.chevron.right"
        }
        if mime.contains("csv") || name.hasSuffix(".csv") { return "tablecells" }
        return "doc.text"
    }

    // MARK: - Data

    private func load() async {
        isLoading = true
        do {
            documents = try await api.knowledge(scope: .global, folderId: nil)
                .sorted { $0.createdAt > $1.createdAt }
            loadError = nil
        } catch {
            loadError = APIError.describe(error)
        }
        isLoading = false
    }

    /// Ingestion happens on the request thread, but a slow PDF can still be
    /// "processing" when the list arrives — poll until everything settles.
    private func pollWhileProcessing() async {
        guard hasProcessing else { return }
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 2_500_000_000)
            if Task.isCancelled { return }
            guard let fresh = try? await api.knowledge(scope: .global, folderId: nil) else { continue }
            documents = fresh.sorted { $0.createdAt > $1.createdAt }
            if !hasProcessing { return }
        }
    }

    private func merge(_ incoming: [KbDocument]) {
        var byId: [String: KbDocument] = [:]
        for document in documents { byId[document.id] = document }
        for document in incoming { byId[document.id] = document }
        documents = byId.values.sorted { $0.createdAt > $1.createdAt }
    }

    private func remove(_ document: KbDocument) {
        busyId = document.id
        Task {
            do {
                try await api.deleteKnowledge(id: document.id)
                documents.removeAll { $0.id == document.id }
                actionError = nil
            } catch {
                actionError = APIError.describe(error)
            }
            busyId = nil
        }
    }
}
