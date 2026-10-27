import Foundation
import LocalCodeKit
import ModelsFeature

/// The `/api/images/*` endpoints, added to the shared `APIClient` actor.
extension APIClient {

    /// `GET /api/images/models` — only models that can generate images.
    public func imageModels() async throws -> [ImageModelDTO] {
        try await send(makeRequest("api/images/models"), as: [ImageModelDTO].self)
    }

    /// `GET /api/images?limit&offset&favorite` — the gallery, newest first.
    public func images(limit: Int = 60,
                       offset: Int = 0,
                       favoritesOnly: Bool = false) async throws -> [ImageDTO] {
        let path = query("api/images", [
            "limit": String(limit),
            "offset": offset > 0 ? String(offset) : nil,
            "favorite": favoritesOnly ? "1" : nil,
        ])
        return try await send(makeRequest(path), as: [ImageDTO].self)
    }

    /// `GET /api/images/{id}/file` — the PNG bytes.
    public func imageData(id: String) async throws -> Data {
        try await sendData(makeRequest("api/images/\(pathSegment(id))/file", accept: "image/png"))
    }

    /// `PATCH /api/images/{id} {favorite}`.
    @discardableResult
    public func setImageFavorite(id: String, favorite: Bool) async throws -> ImageDTO {
        let body = try encodeBody(JSONObject(["favorite": .bool(favorite)]))
        return try await send(makeRequest("api/images/\(pathSegment(id))", method: "PATCH", body: body),
                              as: ImageDTO.self)
    }

    /// `DELETE /api/images/{id}`.
    public func deleteImage(id: String) async throws {
        try await sendVoid(makeRequest("api/images/\(pathSegment(id))", method: "DELETE"))
    }

    /// `POST /api/images/generate` — SSE.
    ///
    /// Generation is long (a minute or more on a small Mac) and the connection
    /// stays open the whole time, which is exactly what the client's 1-hour
    /// resource timeout is for. Cancelling the consuming task aborts the job.
    public func generateImage(_ request: ImageRequest) throws -> AsyncThrowingStream<ImageStreamEvent, Error> {
        let urlRequest = makeRequest("api/images/generate",
                                     method: "POST",
                                     body: try encodeBody(request),
                                     accept: "text/event-stream")
        let frames = sseStream(urlRequest)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await frame in frames {
                        if let event = ImageSSEDecoder.decode(frame) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Absolute URL of a stored image, for `AsyncImage` and for drag-and-drop.
    public func imageURL(_ image: ImageDTO) -> URL {
        url(for: image.url.hasPrefix("/") ? String(image.url.dropFirst()) : image.url)
    }
}
