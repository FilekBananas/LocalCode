import AppKit
import Foundation
import UniformTypeIdentifiers

/// A picture attached to an agent message, ready to send to a vision model.
///
/// The same rules as the terminal (`localcode/tui/images_in.py`): a small PNG,
/// JPEG, GIF or WebP goes as it is; anything else — a camera photo, HEIC, a
/// 5K screenshot — is redrawn as a JPEG at most `maxSide` pixels on its long
/// side, the size vision models scale down to anyway.
public struct AgentImage: Identifiable, Hashable, Sendable {
    public let id: UUID
    public let name: String
    public let mime: String
    /// Base64 of the bytes sent to the model.
    public let data: String
    /// A small PNG for the composer chips and the transcript.
    public let thumbnail: Data

    public static let maxSide: CGFloat = 1600
    static let maxRawBytes = 3 * 1024 * 1024
    public static let maxPerMessage = 6

    /// `{name, mime, data}` — what the `user` message carries over the socket.
    var payload: [String: Any] { ["name": name, "mime": mime, "data": data] }

    public init(id: UUID = UUID(), name: String, mime: String, data: String, thumbnail: Data) {
        self.id = id
        self.name = name
        self.mime = mime
        self.data = data
        self.thumbnail = thumbnail
    }

    /// File extensions the composer accepts.
    public static let fileTypes: [UTType] = [.png, .jpeg, .gif, .webP, .heic, .heif, .bmp, .tiff]

    public static func isImage(_ url: URL) -> Bool {
        guard let type = UTType(filenameExtension: url.pathExtension.lowercased()) else { return false }
        return type.conforms(to: .image)
    }

    /// Prepare a file. `nil` when it cannot be read as a picture.
    public static func from(url: URL) -> AgentImage? {
        let passThrough: [String: String] = ["png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                                             "gif": "image/gif", "webp": "image/webp"]
        guard let raw = try? Data(contentsOf: url), let image = NSImage(data: raw) else { return nil }
        let ext = url.pathExtension.lowercased()
        if let mime = passThrough[ext], raw.count <= maxRawBytes, longSide(of: image) <= maxSide * 1.5 {
            return AgentImage(name: url.lastPathComponent, mime: mime, data: raw.base64EncodedString(),
                              thumbnail: thumbnail(of: image))
        }
        return from(image: image, name: url.lastPathComponent)
    }

    /// Prepare a picture from the clipboard or a drag (no file behind it).
    public static func from(image: NSImage, name: String) -> AgentImage? {
        guard let jpeg = jpegData(of: image, maxSide: maxSide) else { return nil }
        let base = (name as NSString).deletingPathExtension
        return AgentImage(name: base.isEmpty ? "obraz.jpg" : "\(base).jpg", mime: "image/jpeg",
                          data: jpeg.base64EncodedString(), thumbnail: thumbnail(of: image))
    }

    // MARK: - Drawing

    /// Pixel size; a vector or drawn image (pasted from a browser, a PDF page)
    /// reports 0 pixels, and then its point size is what there is.
    private static func pixelSize(of image: NSImage) -> CGSize {
        let rep = image.representations.max { $0.pixelsWide < $1.pixelsWide }
        let width = (rep?.pixelsWide ?? 0) > 0 ? CGFloat(rep!.pixelsWide) : image.size.width
        let height = (rep?.pixelsHigh ?? 0) > 0 ? CGFloat(rep!.pixelsHigh) : image.size.height
        return CGSize(width: width, height: height)
    }

    private static func longSide(of image: NSImage) -> CGFloat {
        let size = pixelSize(of: image)
        return max(size.width, size.height)
    }

    private static func redraw(_ image: NSImage, maxSide: CGFloat) -> NSBitmapImageRep? {
        let size = pixelSize(of: image)
        var width = size.width
        var height = size.height
        guard width > 0, height > 0 else { return nil }
        let scale = min(1, maxSide / max(width, height))
        width = (width * scale).rounded()
        height = (height * scale).rounded()
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(width), pixelsHigh: Int(height),
                                            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return nil }
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        // A white ground: JPEG has no transparency, and a transparent PNG
        // would otherwise turn black.
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
        return bitmap
    }

    static func jpegData(of image: NSImage, maxSide: CGFloat) -> Data? {
        redraw(image, maxSide: maxSide)?.representation(using: .jpeg, properties: [.compressionFactor: 0.85])
    }

    static func thumbnail(of image: NSImage) -> Data {
        redraw(image, maxSide: 160)?.representation(using: .png, properties: [:]) ?? Data()
    }
}
