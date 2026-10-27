import Foundation
import LocalCodeKit

/// The wire types of `/api/video/*` (see `docs/VIDEO.md`).
///
/// Video is the slowest thing this app does — a second of footage costs about a
/// minute of computing on an M-series Mac — so every type here carries what the
/// screen needs to keep somebody informed while they wait: the shape that was
/// asked for, how far along the denoising is, and how long it actually took.

// MARK: - Models

/// One entry of `GET /api/video/models`: a video-looking model on disk.
public struct VideoModelDTO: Decodable, Hashable, Sendable, Identifiable {
    public var key: String
    public var displayName: String
    public var format: String
    public var sizeBytes: Int64
    /// `ltx-2b` (runnable), `ltx-other`, `other-video`, `unknown`.
    public var videoKind: String
    /// False when LocalCode recognises the model but cannot run it.
    public var supported: Bool
    /// Why it cannot be run, in Polish, when `supported` is false.
    public var reason: String?

    public var id: String { key }

    private enum CodingKeys: String, CodingKey {
        case key, displayName, format, sizeBytes, videoKind, supported, reason
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = (try? c.decode(String.self, forKey: .key)) ?? ""
        displayName = (try? c.decode(String.self, forKey: .displayName)) ?? key
        format = (try? c.decode(String.self, forKey: .format)) ?? ""
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
        videoKind = (try? c.decode(String.self, forKey: .videoKind)) ?? "unknown"
        supported = (try? c.decode(Bool.self, forKey: .supported)) ?? false
        reason = try? c.decodeIfPresent(String.self, forKey: .reason)
    }

    public init(key: String, displayName: String, format: String = "gguf",
                sizeBytes: Int64 = 0, videoKind: String = "ltx-2b",
                supported: Bool = true, reason: String? = nil) {
        self.key = key
        self.displayName = displayName
        self.format = format
        self.sizeBytes = sizeBytes
        self.videoKind = videoKind
        self.supported = supported
        self.reason = reason
    }
}

// MARK: - The shape of a request

/// Width, height, frames, fps and steps — the five numbers that decide both the
/// clip and how long it takes. Sent to `/api/video/estimate` before anyone
/// commits to waiting, and to `/api/video/generate` to do the work.
public struct VideoShape: Encodable, Hashable, Sendable {
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var steps: Int

    public init(width: Int = 704, height: Int = 480, numFrames: Int = 41,
                fps: Int = 24, steps: Int = 30) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.steps = steps
    }

    /// Seconds of footage this shape produces.
    public var seconds: Double { Double(numFrames) / Double(max(1, fps)) }
}

/// `POST /api/video/estimate` → how long this shape is expected to take.
public struct VideoEstimate: Decodable, Hashable, Sendable {
    public var seconds: Double
    public var message: String
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var steps: Int

    private enum CodingKeys: String, CodingKey {
        case seconds, message, width, height, numFrames, fps, steps
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        seconds = (try? c.decode(Double.self, forKey: .seconds)) ?? 0
        message = (try? c.decode(String.self, forKey: .message)) ?? ""
        width = (try? c.decode(Int.self, forKey: .width)) ?? 0
        height = (try? c.decode(Int.self, forKey: .height)) ?? 0
        numFrames = (try? c.decode(Int.self, forKey: .numFrames)) ?? 0
        fps = (try? c.decode(Int.self, forKey: .fps)) ?? 24
        steps = (try? c.decode(Int.self, forKey: .steps)) ?? 0
    }
}

/// The body of `POST /api/video/generate`.
public struct VideoRequest: Encodable, Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var modelKey: String?
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var steps: Int
    public var guidance: Double?
    public var seed: Int?
    /// Above the default frame cap the server needs this said out loud.
    public var force: Bool
    /// A file name inside the video directory — never a path (the server refuses one).
    public var output: String?

    public init(prompt: String, shape: VideoShape, modelKey: String? = nil,
                negativePrompt: String? = nil, guidance: Double? = nil,
                seed: Int? = nil, force: Bool = false, output: String? = nil) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.modelKey = modelKey
        self.width = shape.width
        self.height = shape.height
        self.numFrames = shape.numFrames
        self.fps = shape.fps
        self.steps = shape.steps
        self.guidance = guidance
        self.seed = seed
        self.force = force
        self.output = output
    }
}

// MARK: - What comes back

/// `POST /api/video/generate` → the finished clip.
public struct VideoDTO: Decodable, Hashable, Sendable, Identifiable {
    public var id: String
    public var path: String
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var seed: Int
    public var steps: Int
    public var guidance: Double
    public var modelKey: String
    public var prompt: String
    public var elapsedMs: Int
    public var sizeBytes: Int64

    /// Seconds of footage in the clip.
    public var seconds: Double { Double(numFrames) / Double(max(1, fps)) }

    private enum CodingKeys: String, CodingKey {
        case id, path, width, height, numFrames, fps, seed, steps, guidance
        case modelKey, prompt, elapsedMs, sizeBytes
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decode(String.self, forKey: .id)) ?? UUID().uuidString
        path = (try? c.decode(String.self, forKey: .path)) ?? ""
        width = (try? c.decode(Int.self, forKey: .width)) ?? 0
        height = (try? c.decode(Int.self, forKey: .height)) ?? 0
        numFrames = (try? c.decode(Int.self, forKey: .numFrames)) ?? 0
        fps = (try? c.decode(Int.self, forKey: .fps)) ?? 24
        seed = (try? c.decode(Int.self, forKey: .seed)) ?? 0
        steps = (try? c.decode(Int.self, forKey: .steps)) ?? 0
        guidance = (try? c.decode(Double.self, forKey: .guidance)) ?? 0
        modelKey = (try? c.decode(String.self, forKey: .modelKey)) ?? ""
        prompt = (try? c.decode(String.self, forKey: .prompt)) ?? ""
        elapsedMs = (try? c.decode(Int.self, forKey: .elapsedMs)) ?? 0
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
    }
}

#if DEBUG
extension VideoEstimate {
    /// A hand-made estimate, for rendering the screen without a server.
    public static func preview(seconds: Double, message: String) -> VideoEstimate {
        let json = """
        {"seconds": \(seconds), "message": "\(message)", "width": 704, "height": 480,
         "numFrames": 25, "fps": 24, "steps": 8}
        """
        return try! JSONDecoder().decode(VideoEstimate.self, from: Data(json.utf8))
    }
}

extension VideoDTO {
    /// The first clip this engine actually produced, as data.
    public static func preview() -> VideoDTO {
        let json = """
        {"id": "klip-1", "path": "/Users/ty/.localcode/data/video/wideo-20260929-122600.mp4",
         "width": 704, "height": 480, "numFrames": 25, "fps": 24, "seed": 7, "steps": 8,
         "guidance": 1.0, "modelKey": "city96/LTX-Video-gguf/ltx-video-2b-v0.9-Q4_K_M",
         "prompt": "a ripe yellow banana on a wooden table, daylight",
         "elapsedMs": 86700, "sizeBytes": 18040}
        """
        return try! JSONDecoder().decode(VideoDTO.self, from: Data(json.utf8))
    }
}
#endif

/// What one SSE frame of `/api/video/generate` means to the screen.
public enum VideoStreamEvent: Hashable, Sendable {
    case status(String)
    case progress(stage: String, step: Int, steps: Int, message: String, fraction: Double)
    case video(VideoDTO)
    case done
    case failure(String)
}

/// Decodes the `/api/video/generate` SSE frames, skipping anything this build
/// does not know so the server can add frame types freely.
public enum VideoSSEDecoder {
    private struct Frame: Decodable {
        let type: String
        let message: String?
        let stage: String?
        let step: Int?
        let steps: Int?
        let fraction: Double?
        let video: VideoDTO?
    }

    public static func decode(_ frame: SSEFrame) -> VideoStreamEvent? {
        guard let parsed = frame.decode(Frame.self) else { return nil }
        switch parsed.type {
        case "status":
            guard let message = parsed.message, !message.isEmpty else { return nil }
            return .status(message)
        case "progress":
            let steps = max(1, parsed.steps ?? 1)
            let step = parsed.step ?? 0
            return .progress(stage: parsed.stage ?? "",
                             step: step,
                             steps: steps,
                             message: parsed.message ?? "",
                             fraction: parsed.fraction ?? Double(step) / Double(steps))
        case "video":
            guard let video = parsed.video else { return nil }
            return .video(video)
        case "done":
            return .done
        case "error":
            return .failure(parsed.message ?? "Nie udało się wygenerować wideo.")
        default:
            return nil
        }
    }
}
