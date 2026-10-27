import SwiftUI
import AppKit
import LocalCodeKit

/// A gallery image, loaded through the authenticated API client.
///
/// `AsyncImage` cannot be used here: every LocalCode endpoint — including
/// `GET /api/images/{id}/file` — requires the bearer token, and `AsyncImage`
/// takes a bare URL with no way to attach a header, so it renders a broken
/// image for every picture in the gallery. This loads the bytes the same way
/// the rest of the app loads anything else, and keeps them in a small process
/// cache so scrolling the grid does not refetch a PNG per row.
struct AuthImage<Placeholder: View>: View {
    let imageId: String
    var contentMode: ContentMode = .fill
    @ViewBuilder let placeholder: (Bool) -> Placeholder

    @Environment(ImagesStore.self) private var store
    @State private var image: NSImage?
    @State private var didFail = false

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            } else {
                placeholder(didFail)
            }
        }
        .task(id: imageId) { await load() }
    }

    private func load() async {
        if let cached = ImageCache.shared.image(for: imageId) {
            image = cached
            return
        }
        guard let data = await store.imageData(byId: imageId), let loaded = NSImage(data: data) else {
            didFail = true
            return
        }
        ImageCache.shared.store(loaded, for: imageId)
        image = loaded
    }
}

/// A small in-memory cache of decoded gallery images.
///
/// `NSCache` counts by object, so the limit is a picture count rather than a
/// byte budget: generous enough for a full screen of thumbnails, small enough
/// that a long gallery cannot grow without bound.
final class ImageCache: @unchecked Sendable {
    static let shared = ImageCache()

    private let cache = NSCache<NSString, NSImage>()

    private init() {
        cache.countLimit = 120
    }

    func image(for id: String) -> NSImage? {
        cache.object(forKey: id as NSString)
    }

    func store(_ image: NSImage, for id: String) {
        cache.setObject(image, forKey: id as NSString)
    }
}
