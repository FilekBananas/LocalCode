import Foundation
import UniformTypeIdentifiers

/// One file in a `multipart/form-data` upload.
public struct MultipartFile: Sendable {
    public let name: String
    public let mime: String
    public let data: Data

    public init(name: String, mime: String, data: Data) {
        self.name = name
        self.mime = mime
        self.data = data
    }

    /// Reads a file the user picked, opening the security scope first — panels
    /// hand back URLs outside the app's own container.
    public static func read(contentsOf url: URL) throws -> MultipartFile {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url)
        return MultipartFile(name: url.lastPathComponent, mime: mimeType(for: url), data: data)
    }

    public static func mimeType(for url: URL) -> String {
        UTType(filenameExtension: url.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
    }
}

/// Minimal `multipart/form-data` builder — the app only ever uploads files plus
/// a couple of scalar fields, so a dependency would be overkill.
public enum Multipart {
    public static func build(fields: [String: String],
                             files: [MultipartFile],
                             fieldName: String) -> (Data, String) {
        let boundary = "----localcode-\(UUID().uuidString)"
        var body = Data()

        func append(_ string: String) { body.append(Data(string.utf8)) }

        for (key, value) in fields.sorted(by: { $0.key < $1.key }) {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n")
            append("\(value)\r\n")
        }
        for file in files {
            let safeName = file.name.replacingOccurrences(of: "\"", with: "_")
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(safeName)\"\r\n")
            append("Content-Type: \(file.mime)\r\n\r\n")
            body.append(file.data)
            append("\r\n")
        }
        append("--\(boundary)--\r\n")
        return (body, "multipart/form-data; boundary=\(boundary)")
    }
}
