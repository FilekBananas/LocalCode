import SwiftUI
import AppKit
import UniformTypeIdentifiers
import LocalCodeKit

/// The right pane of "Obrazy": everything generated so far.
struct ImageGallery: View {
    @Environment(ImagesStore.self) private var store

    private let columns = [GridItem(.adaptive(minimum: 160, maximum: 240), spacing: 12)]

    var body: some View {
        @Bindable var store = store

        return VStack(spacing: 0) {
            toolbar
            if store.gallery.isEmpty {
                VStack {
                    Spacer(minLength: 0)
                    EmptyState(systemImage: "photo.on.rectangle.angled",
                               title: store.isLoadingGallery ? "Ładuję galerię…" : "Nic tu jeszcze nie ma",
                               message: store.favoritesOnly
                                   ? "Żaden obraz nie jest oznaczony jako ulubiony."
                                   : "Wpisz opis po lewej i naciśnij „Generuj”. Gotowe obrazy pojawią się tutaj.")
                    Spacer(minLength: 0)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(store.gallery) { image in
                            ImageThumbnail(image: image)
                        }
                    }
                    .padding(16)
                }
            }
        }
        .background(Theme.bg)
    }

    private var toolbar: some View {
        @Bindable var store = store

        return HStack(spacing: 10) {
            Text(Format.plural(store.gallery.count, "obraz", "obrazy", "obrazów"))
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
            Spacer(minLength: 0)
            Chip(title: "Tylko ulubione",
                 systemImage: store.favoritesOnly ? "heart.fill" : "heart",
                 isSelected: store.favoritesOnly,
                 onTap: {
                     store.favoritesOnly.toggle()
                     Task { await store.refreshGallery() }
                 })
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
    }
}

// MARK: - Thumbnail

/// One image in the grid. Click opens the detail sheet; the image can also be
/// dragged straight out of the app into any other application.
struct ImageThumbnail: View {
    let image: ImageDTO

    @Environment(ImagesStore.self) private var store
    @State private var isHovered = false
    @State private var url: URL?

    var body: some View {
        Button {
            store.detail = image
        } label: {
            ZStack(alignment: .topTrailing) {
                thumbnail
                if image.favorite {
                    Image(systemName: "heart.fill")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.white)
                        .shadow(color: .black.opacity(0.5), radius: 2)
                        .padding(7)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .onHover { isHovered = $0 }
        .help(image.authoredPrompt)
        .task { url = await store.url(for: image) }
        // Dragging the image out of the app: the file URL is what every other
        // application expects, and the gallery files are already on disk.
        .draggable(dragItem) {
            Image(systemName: "photo")
                .font(.system(size: 24))
                .padding(10)
        }
        .contextMenu {
            Button("Użyj ponownie") { store.reuse(image) }
            Button(image.favorite ? "Usuń z ulubionych" : "Dodaj do ulubionych") {
                Task { await store.toggleFavorite(image) }
            }
            Button("Pokaż w Finderze") { store.revealInFinder(image) }
            Divider()
            Button("Usuń", role: .destructive) { Task { await store.delete(image) } }
        }
    }

    /// A file URL when the server told us where the file is, otherwise the HTTP
    /// URL, which still drops into a browser or a note.
    private var dragItem: URL {
        if let path = image.path, !path.isEmpty { return URL(fileURLWithPath: path) }
        return url ?? URL(fileURLWithPath: "/")
    }

    private var thumbnail: some View {
        GeometryReader { geometry in
            AuthImage(imageId: image.id, contentMode: .fill) { failed in
                placeholder(systemImage: failed ? "exclamationmark.triangle" : "photo")
            }
            .frame(width: geometry.size.width, height: geometry.size.width)
            .clipped()
        }
        .aspectRatio(1, contentMode: .fit)
        .background(Theme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(isHovered ? Theme.accent.opacity(0.5) : Theme.border, lineWidth: 1)
        )
        .animation(.easeOut(duration: 0.12), value: isHovered)
    }

    private func placeholder(systemImage: String) -> some View {
        ZStack {
            Theme.surface2
            Image(systemName: systemImage)
                .font(.system(size: 20, weight: .light))
                .foregroundStyle(Theme.fgSubtle)
        }
    }
}
