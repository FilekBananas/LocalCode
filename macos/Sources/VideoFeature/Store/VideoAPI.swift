import Foundation
import LocalCodeKit

/// The `/api/video/*` endpoints, added to the shared `APIClient` actor.
///
/// The finished clip stays on disk and comes back as a path: a few-second video
/// is megabytes, the app and the server share the machine, and the player wants
/// a file URL anyway (see `docs/VIDEO.md`).
extension APIClient {

    /// `GET /api/video/models` — video-looking models on disk, runnable or not.
    public func videoModels() async throws -> [VideoModelDTO] {
        try await send(makeRequest("api/video/models"), as: [VideoModelDTO].self)
    }

    /// `POST /api/video/estimate` — how long this shape is expected to take.
    ///
    /// Asked before anything is committed to, which is the point: a minute of
    /// waiting should be a decision, not a surprise.
    public func estimateVideo(_ shape: VideoShape) async throws -> VideoEstimate {
        try await send(makeRequest("api/video/estimate", method: "POST",
                                   body: try encodeBody(shape)),
                       as: VideoEstimate.self)
    }

    /// `POST /api/video/generate` — SSE.
    ///
    /// Minutes, not seconds, so the connection stays open the whole time and the
    /// client's long resource timeout is what keeps it there. Cancelling the
    /// consuming task cancels the job on the server too.
    public func generateVideo(_ request: VideoRequest) throws -> AsyncThrowingStream<VideoStreamEvent, Error> {
        let urlRequest = makeRequest("api/video/generate",
                                     method: "POST",
                                     body: try encodeBody(request),
                                     accept: "text/event-stream")
        let frames = sseStream(urlRequest)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await frame in frames {
                        if let event = VideoSSEDecoder.decode(frame) {
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
}
