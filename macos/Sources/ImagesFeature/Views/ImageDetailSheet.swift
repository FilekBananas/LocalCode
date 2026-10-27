import SwiftUI
import AppKit
import UniformTypeIdentifiers
import LocalCodeKit

/// One generated image in full, with every parameter it was made with.
struct ImageDetailSheet: View {
    let image: ImageDTO

    @Environment(ImagesStore.self) private var store

    @State private var url: URL?
    @State private var isSaving = false
    @State private var saveDocument: ImageDocument?
    @State private var isConfirmingDelete = false

    /// The gallery copy, so a favourite toggle is reflected here immediately.
    private var current: ImageDTO {
        store.gallery.first { $0.id == image.id } ?? image
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Hairline()
            HStack(spacing: 0) {
                preview
                Hairline(vertical: true)
                parameters
                    .frame(width: 300)
            }
            Hairline()
            footer
        }
        .frame(width: 900, height: 640)
        .background(Theme.bg)
        .task { url = await store.url(for: image) }
        .fileExporter(isPresented: $isSaving,
                      document: saveDocument,
                      contentType: .png,
                      defaultFilename: current.suggestedFilename) { result in
            if case let .failure(error) = result {
                store.errorMessage = error.localizedDescription
            }
        }
        .alert("Usunąć obraz?", isPresented: $isConfirmingDelete) {
            Button("Anuluj", role: .cancel) {}
            Button("Usuń", role: .destructive) {
                Task { await store.delete(current) }
            }
        } message: {
            Text("Plik zostanie trwale usunięty z dysku. Tej operacji nie można cofnąć.")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(current.authoredPrompt.isEmpty ? "Obraz" : current.authoredPrompt)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(2)
                    .textSelection(.enabled)
                Text("\(current.modelName)  ·  \(current.sizeLabel)  ·  \(Format.relative(current.createdAt))")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
            }
            Spacer(minLength: 8)
            IconButton(systemName: current.favorite ? "heart.fill" : "heart",
                       label: current.favorite ? "Usuń z ulubionych" : "Dodaj do ulubionych",
                       tint: current.favorite ? Theme.danger : Theme.fgMuted) {
                Task { await store.toggleFavorite(current) }
            }
            IconButton(systemName: "xmark", label: "Zamknij") { store.detail = nil }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 13)
        .background(Theme.surface)
    }

    // MARK: - Preview

    private var preview: some View {
        ZStack {
            Theme.surface2
            AuthImage(imageId: current.id, contentMode: .fit) { failed in
                if failed {
                    EmptyState(systemImage: "exclamationmark.triangle",
                               title: "Nie udało się wczytać obrazu")
                } else {
                    ProgressView().controlSize(.large)
                }
            }
            .padding(16)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // The same drag-out as in the grid: the detail view is where people
        // most often want to pull the picture into another application.
        .draggable(dragItem) {
            Image(systemName: "photo").font(.system(size: 24)).padding(10)
        }
    }

    private var dragItem: URL {
        if let path = current.path, !path.isEmpty { return URL(fileURLWithPath: path) }
        return url ?? URL(fileURLWithPath: "/")
    }

    // MARK: - Parameters

    private var parameters: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                SectionHeader("Parametry")

                Card(padding: 12) {
                    VStack(alignment: .leading, spacing: 9) {
                        field("Model", current.modelName)
                        Hairline()
                        field("Rozmiar", current.sizeLabel)
                        Hairline()
                        field("Kroki", "\(current.steps)")
                        if let guidance = current.guidance {
                            Hairline()
                            field("Guidance",
                                  String(format: "%.1f", guidance).replacingOccurrences(of: ".", with: ","))
                        }
                        Hairline()
                        field("Ziarno", "\(current.seed)", monospaced: true)
                        Hairline()
                        field("Czas", current.durationLabel)
                        Hairline()
                        field("Plik", Format.bytes(current.sizeBytes))
                        Hairline()
                        field("Utworzono", Format.dateTime(current.createdAt))
                    }
                }

                SectionHeader("Opis")
                Card(padding: 12) {
                    Text(current.authoredPrompt.isEmpty ? "—" : current.authoredPrompt)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fg)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }

                if !current.negativePrompt.isEmpty {
                    SectionHeader("Czego unikać")
                    Card(padding: 12) {
                        Text(current.negativePrompt)
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.fgMuted)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(16)
        }
        .background(Theme.sidebar)
    }

    private func field(_ label: String, _ value: String, monospaced: Bool = false) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(label)
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgMuted)
            Spacer(minLength: 8)
            Text(value)
                .font(.system(size: 12, design: monospaced ? .monospaced : .default))
                .foregroundStyle(Theme.fg)
                .multilineTextAlignment(.trailing)
                .textSelection(.enabled)
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 9) {
            SecondaryButton(title: "Użyj ponownie", systemImage: "arrow.counterclockwise", height: 34) {
                store.reuse(current)
            }
            .help("Wczytaj wszystkie parametry tego obrazu do panelu po lewej")

            SecondaryButton(title: "Zapisz jako…", systemImage: "square.and.arrow.down", height: 34) {
                Task {
                    guard let data = await store.imageData(current) else { return }
                    saveDocument = ImageDocument(data: data)
                    isSaving = true
                }
            }

            SecondaryButton(title: "Pokaż w Finderze", systemImage: "folder", height: 34) {
                store.revealInFinder(current)
            }

            Spacer(minLength: 0)

            DangerButton(title: "Usuń", systemImage: "trash", filled: false, height: 34) {
                isConfirmingDelete = true
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(Theme.surface)
    }
}

/// Carries the PNG bytes into `fileExporter`.
struct ImageDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.png] }

    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
