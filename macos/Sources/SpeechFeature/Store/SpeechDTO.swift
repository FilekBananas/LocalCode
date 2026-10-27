import Foundation
import LocalCodeKit

/// The wire types of `/api/speech/*` (see `docs/SPEECH.md`).

// MARK: - Transcription

/// One timed chunk of a transcript — what the `.srt` writer needs.
public struct TranscriptSegment: Decodable, Hashable, Sendable {
    public var start: Double
    public var end: Double
    public var text: String

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        start = (try? c.decode(Double.self, forKey: .start)) ?? 0
        end = (try? c.decode(Double.self, forKey: .end)) ?? 0
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
    }

    private enum CodingKeys: String, CodingKey { case start, end, text }
}

/// `POST /api/speech/transcribe` → a recording turned into text.
public struct Transcript: Decodable, Hashable, Sendable {
    public var text: String
    public var language: String?
    public var durationSeconds: Double?
    public var segments: [TranscriptSegment]
    public var modelKey: String
    public var elapsedMs: Int
    /// Present only when the request asked for `srt: true`.
    public var srt: String?

    private enum CodingKeys: String, CodingKey {
        case text, language, durationSeconds, segments, modelKey, elapsedMs, srt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        language = try? c.decodeIfPresent(String.self, forKey: .language)
        durationSeconds = try? c.decodeIfPresent(Double.self, forKey: .durationSeconds)
        segments = (try? c.decode([TranscriptSegment].self, forKey: .segments)) ?? []
        modelKey = (try? c.decode(String.self, forKey: .modelKey)) ?? ""
        elapsedMs = (try? c.decode(Int.self, forKey: .elapsedMs)) ?? 0
        srt = try? c.decodeIfPresent(String.self, forKey: .srt)
    }
}

/// The body of `POST /api/speech/transcribe`.
public struct TranscribeRequest: Encodable, Sendable {
    public var path: String
    public var modelKey: String?
    public var language: String?
    public var segments: Bool
    public var srt: Bool

    public init(path: String,
                modelKey: String? = nil,
                language: String? = nil,
                segments: Bool = true,
                srt: Bool = true) {
        self.path = path
        self.modelKey = modelKey
        self.language = language
        self.segments = segments
        self.srt = srt
    }
}

// MARK: - Synthesis

/// One voice a synthesis model offers (`GET /api/speech/voices`).
public struct SpeechVoice: Decodable, Identifiable, Hashable, Sendable {
    public var id: String
    public var label: String
    public var language: String?

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(String.self, forKey: .id)) ?? ""
        label = (try? c.decode(String.self, forKey: .label)) ?? id
        language = try? c.decodeIfPresent(String.self, forKey: .language)
    }

    private enum CodingKeys: String, CodingKey { case id, label, language }
}

/// `POST /api/speech/speak` → the wave file it wrote to disk.
public struct VoiceResult: Decodable, Hashable, Sendable {
    public var path: String
    public var text: String
    public var voice: String?
    public var durationSeconds: Double?
    public var sampleRate: Int
    public var modelKey: String
    public var elapsedMs: Int
    public var sizeBytes: Int64

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        path = (try? c.decode(String.self, forKey: .path)) ?? ""
        text = (try? c.decode(String.self, forKey: .text)) ?? ""
        voice = try? c.decodeIfPresent(String.self, forKey: .voice)
        durationSeconds = try? c.decodeIfPresent(Double.self, forKey: .durationSeconds)
        sampleRate = (try? c.decode(Int.self, forKey: .sampleRate)) ?? 24_000
        modelKey = (try? c.decode(String.self, forKey: .modelKey)) ?? ""
        elapsedMs = (try? c.decode(Int.self, forKey: .elapsedMs)) ?? 0
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
    }

    private enum CodingKeys: String, CodingKey {
        case path, text, voice, durationSeconds, sampleRate, modelKey, elapsedMs, sizeBytes
    }

    /// The file name the server chose — the natural default for a save panel.
    public var suggestedFilename: String {
        let name = (path as NSString).lastPathComponent
        return name.isEmpty ? "mowa.wav" : name
    }
}

/// The body of `POST /api/speech/speak`.
public struct SpeakRequest: Encodable, Sendable {
    public var text: String
    public var modelKey: String?
    public var voice: String?
    public var speed: Double
    public var output: String?

    public init(text: String,
                modelKey: String? = nil,
                voice: String? = nil,
                speed: Double = 1.0,
                output: String? = nil) {
        self.text = text
        self.modelKey = modelKey
        self.voice = voice
        self.speed = speed
        self.output = output
    }
}


#if DEBUG
extension Transcript {
    /// The transcript of a real recording, as data — for rendering the screen.
    public static func preview() -> Transcript {
        let json = """
        {"text": "Banany rosną w ciepłych krajach i są bardzo zdrowe.",
         "language": "pl", "durationSeconds": 2.8,
         "segments": [{"start": 0.0, "end": 2.8,
                       "text": "Banany rosną w ciepłych krajach i są bardzo zdrowe."}],
         "modelKey": "mlx-community/whisper-small-mlx", "elapsedMs": 1840,
         "srt": "1\\n00:00:00,000 --> 00:00:02,800\\nBanany rosną w ciepłych krajach i są bardzo zdrowe.\\n"}
        """
        return try! JSONDecoder().decode(Transcript.self, from: Data(json.utf8))
    }
}

extension VoiceResult {
    /// The wave file Kokoro actually produced here, as data.
    public static func preview() -> VoiceResult {
        let json = """
        {"path": "/Users/ty/.localcode/data/speech/mowa-20260929-112400.wav",
         "text": "Bananas grow in warm countries.", "voice": "af_heart",
         "durationSeconds": 2.52, "sampleRate": 24000,
         "modelKey": "mlx-community/Kokoro-82M-bf16", "elapsedMs": 9120, "sizeBytes": 121244}
        """
        return try! JSONDecoder().decode(VoiceResult.self, from: Data(json.utf8))
    }
}
#endif
