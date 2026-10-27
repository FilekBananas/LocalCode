import SwiftUI
import AppKit
import LocalCodeKit

/// The mark of the laboratory that made a model — DeepSeek's whale, Google's G.
///
/// It follows the *weights*, not the uploader: a community conversion of a
/// DeepSeek model still shows DeepSeek, because that is whose model it is. A
/// model nobody can attribute shows nothing at all; there is no placeholder
/// letter, because an invented mark would say something we do not know.
///
/// The images are the organisations' own Hugging Face avatars, shipped with the
/// server (`scripts/fetch_logos.py`) and served from `GET /api/logos/{slug}`,
/// so this never reaches out to the network by itself.
struct PublisherLogo: View {
    let slug: String?
    var size: CGFloat = 16
    /// A ring in the card's colour, so the logo reads on any background.
    var ringed: Bool = false

    @Environment(ModelsStore.self) private var store
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let slug, !slug.isEmpty {
                content
                    .task(id: slug) { await load(slug) }
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if let image {
            Image(nsImage: image)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
                .frame(width: size, height: size)
                .clipShape(RoundedRectangle(cornerRadius: size * 0.26, style: .continuous))
                .overlay {
                    if ringed {
                        RoundedRectangle(cornerRadius: size * 0.26, style: .continuous)
                            .strokeBorder(Theme.surface, lineWidth: 1.5)
                    }
                }
        } else {
            // Reserve the space so a row does not jump when the logo arrives.
            Color.clear.frame(width: size, height: size)
        }
    }

    private func load(_ slug: String) async {
        if let cached = LogoCache.shared.image(for: slug) {
            image = cached
            return
        }
        guard let data = await store.publisherLogoData(slug),
              let loaded = NSImage(data: data) else { return }
        LogoCache.shared.store(loaded, for: slug)
        image = loaded
    }
}

/// Decoded logos, kept for the life of the process.
///
/// There are about a hundred of them and each is a 64×64 PNG, so the whole set
/// costs less than one screenshot; scrolling the store must never refetch one.
final class LogoCache: @unchecked Sendable {
    static let shared = LogoCache()

    private let cache = NSCache<NSString, NSImage>()

    private init() {
        cache.countLimit = 200
    }

    func image(for slug: String) -> NSImage? {
        cache.object(forKey: slug as NSString)
    }

    func store(_ image: NSImage, for slug: String) {
        cache.setObject(image, forKey: slug as NSString)
    }
}
