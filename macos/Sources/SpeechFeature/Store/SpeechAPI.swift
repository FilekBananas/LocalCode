import Foundation
import LocalCodeKit

/// The `/api/speech/*` endpoints, added to the shared `APIClient` actor.
///
/// Audio goes by **path on this machine**, never by upload: the app and the
/// server run side by side, recordings can be hundreds of megabytes, and both
/// sides already have the same disk (see `docs/SPEECH.md`).
extension APIClient {

    /// `GET /api/speech/voices?modelKey=` — the voices a synthesis model offers.
    public func speechVoices(modelKey: String?) async throws -> [SpeechVoice] {
        let path = query("api/speech/voices", ["modelKey": modelKey])
        return try await send(makeRequest(path), as: [SpeechVoice].self)
    }

    /// `POST /api/speech/transcribe` — a recording turned into text.
    ///
    /// The model is loaded, used and released for this one call (see
    /// `localcode/speech/runtime.py`), which on a slow Mac and a long
    /// recording can take a while — well past the client's default 60 s.
    public func transcribe(_ request: TranscribeRequest) async throws -> Transcript {
        var urlRequest = makeRequest("api/speech/transcribe", method: "POST", body: try encodeBody(request))
        urlRequest.timeoutInterval = 1800
        return try await send(urlRequest, as: Transcript.self)
    }

    /// `POST /api/speech/speak` — text turned into a wave file on disk.
    public func speak(_ request: SpeakRequest) async throws -> VoiceResult {
        var urlRequest = makeRequest("api/speech/speak", method: "POST", body: try encodeBody(request))
        urlRequest.timeoutInterval = 600
        return try await send(urlRequest, as: VoiceResult.self)
    }
}
