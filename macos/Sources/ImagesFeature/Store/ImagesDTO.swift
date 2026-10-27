import Foundation
import LocalCodeKit

/// The wire types of `/api/images/*` (see `docs/IMAGES.md`).

// MARK: - Models

/// One image model on disk, as `GET /api/images/models` returns it: a library
/// entry plus the defaults its family expects and an estimate of what it will
/// cost in memory.
public struct ImageModelDTO: Decodable, Identifiable, Hashable, Sendable {
    public let key: String
    public var format: String
    public var path: String
    public var displayName: String
    public var sizeBytes: Int64
    public var repo: String?
    public var quant: String?
    /// `flux`, `flux2`, `qwen-image`, `sd1`, `sdxl`, `sd3`, `unknown`.
    public var imageFamily: String
    public var loaded: Bool
    public var defaults: ImageDefaults
    /// What loading this model is expected to need, weights plus working set.
    public var estimatedBytes: Int64?

    public var id: String { key }

    private enum CodingKeys: String, CodingKey {
        case key, format, path, displayName, sizeBytes, repo, quant
        case imageFamily, loaded, defaults, estimatedBytes
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = try c.decode(String.self, forKey: .key)
        format = (try? c.decode(String.self, forKey: .format)) ?? ""
        path = (try? c.decode(String.self, forKey: .path)) ?? ""
        displayName = (try? c.decode(String.self, forKey: .displayName)) ?? key
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
        repo = try? c.decodeIfPresent(String.self, forKey: .repo)
        quant = try? c.decodeIfPresent(String.self, forKey: .quant)
        imageFamily = (try? c.decode(String.self, forKey: .imageFamily)) ?? "unknown"
        loaded = (try? c.decode(Bool.self, forKey: .loaded)) ?? false
        defaults = (try? c.decode(ImageDefaults.self, forKey: .defaults)) ?? ImageDefaults()
        estimatedBytes = try? c.decodeIfPresent(Int64.self, forKey: .estimatedBytes)
    }

    /// What this model needs in memory — the estimate when the server has one,
    /// otherwise the weights with a modest allowance for the working set.
    public var memoryCost: Int64 {
        estimatedBytes ?? Int64(Double(sizeBytes) * 1.3)
    }
}

/// `defaults` of an image model: what its family expects when the user does not
/// say otherwise. FLUX schnell wants 4 steps, SD 1.5 wants 20.
public struct ImageDefaults: Decodable, Hashable, Sendable {
    public var steps: Int?
    public var guidance: Double?
    /// `["1024x1024", "1024x768", "768x1024"]`.
    public var sizes: [String]

    public init(steps: Int? = nil, guidance: Double? = nil, sizes: [String] = []) {
        self.steps = steps
        self.guidance = guidance
        self.sizes = sizes
    }

    private enum CodingKeys: String, CodingKey { case steps, guidance, sizes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        steps = try? c.decodeIfPresent(Int.self, forKey: .steps)
        guidance = try? c.decodeIfPresent(Double.self, forKey: .guidance)
        sizes = (try? c.decode([String].self, forKey: .sizes)) ?? []
    }
}

// MARK: - Gallery

/// One generated image, with every parameter it was made with.
public struct ImageDTO: Decodable, Identifiable, Hashable, Sendable {
    public let id: String
    /// Server-relative: `/api/images/{id}/file`.
    public var url: String
    /// Absolute path on this machine — "Pokaż w Finderze" needs it.
    public var path: String?
    public var prompt: String
    /// What the user typed, when the prompt was translated to English before
    /// generating; empty when it went to the model unchanged.
    public var originalPrompt: String
    public var negativePrompt: String
    public var modelKey: String
    public var modelName: String
    public var width: Int
    public var height: Int
    public var seed: Int
    public var steps: Int
    public var guidance: Double?
    public var favorite: Bool
    public var sizeBytes: Int64
    public var createdAt: Double
    public var elapsedMs: Int

    private enum CodingKeys: String, CodingKey {
        case id, url, path, prompt, originalPrompt, negativePrompt, modelKey, modelName
        case width, height, seed, steps, guidance, favorite, sizeBytes, createdAt, elapsedMs
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        url = (try? c.decode(String.self, forKey: .url)) ?? "/api/images/\(id)/file"
        path = try? c.decodeIfPresent(String.self, forKey: .path)
        prompt = (try? c.decode(String.self, forKey: .prompt)) ?? ""
        originalPrompt = (try? c.decode(String.self, forKey: .originalPrompt)) ?? ""
        negativePrompt = (try? c.decode(String.self, forKey: .negativePrompt)) ?? ""
        modelKey = (try? c.decode(String.self, forKey: .modelKey)) ?? ""
        modelName = (try? c.decode(String.self, forKey: .modelName)) ?? modelKey
        width = (try? c.decode(Int.self, forKey: .width)) ?? 0
        height = (try? c.decode(Int.self, forKey: .height)) ?? 0
        seed = (try? c.decode(Int.self, forKey: .seed)) ?? 0
        steps = (try? c.decode(Int.self, forKey: .steps)) ?? 0
        guidance = try? c.decodeIfPresent(Double.self, forKey: .guidance)
        favorite = (try? c.decode(Bool.self, forKey: .favorite)) ?? false
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
        createdAt = (try? c.decode(Double.self, forKey: .createdAt)) ?? 0
        elapsedMs = (try? c.decode(Int.self, forKey: .elapsedMs)) ?? 0
    }

    public var sizeLabel: String { "\(width) × \(height)" }

    /// The prompt to show and to reuse: the user's own words win over the
    /// translation that was actually sent to the model.
    public var authoredPrompt: String { originalPrompt.isEmpty ? prompt : originalPrompt }

    /// How long the generation took, as the detail sheet prints it.
    public var durationLabel: String {
        Format.duration(Double(elapsedMs) / 1000)
    }

    /// A file name a save panel accepts.
    public var suggestedFilename: String {
        let words = prompt
            .lowercased()
            .unicodeScalars
            .map { CharacterSet.alphanumerics.contains($0) ? Character($0) : " " }
            .reduce(into: "") { $0.append($1) }
            .split(separator: " ")
            .prefix(6)
            .joined(separator: "-")
        return (words.isEmpty ? "obraz" : String(words.prefix(60))) + ".png"
    }
}

// MARK: - Generation

/// The body of `POST /api/images/generate`.
public struct ImageRequest: Encodable, Sendable {
    public var modelKey: String
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int?
    public var guidance: Double?
    public var seed: Int?
    public var count: Int?

    public init(modelKey: String,
                prompt: String,
                negativePrompt: String? = nil,
                width: Int,
                height: Int,
                steps: Int? = nil,
                guidance: Double? = nil,
                seed: Int? = nil,
                count: Int? = nil) {
        self.modelKey = modelKey
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.count = count
    }
}

/// One frame of the generation stream.
public enum ImageStreamEvent: Sendable {
    /// "Przygotowuję model …" — a worker is starting, which takes a while.
    case status(String)
    case progress(index: Int, step: Int, steps: Int)
    case image(ImageDTO)
    case done
    case failure(String)
}

/// Decodes the `/api/images/generate` SSE frames, skipping anything this build
/// does not know so the server can add frame types freely.
public enum ImageSSEDecoder {
    private struct Frame: Decodable {
        let type: String
        let message: String?
        let index: Int?
        let step: Int?
        let steps: Int?
        let image: ImageDTO?
    }

    public static func decode(_ frame: SSEFrame) -> ImageStreamEvent? {
        guard let parsed = frame.decode(Frame.self) else { return nil }
        switch parsed.type {
        case "status":
            guard let message = parsed.message, !message.isEmpty else { return nil }
            return .status(message)
        case "progress":
            return .progress(index: parsed.index ?? 0,
                             step: parsed.step ?? 0,
                             steps: max(1, parsed.steps ?? 1))
        case "image":
            guard let image = parsed.image else { return nil }
            return .image(image)
        case "done":
            return .done
        case "error":
            return .failure(parsed.message ?? "Nie udało się wygenerować obrazu.")
        default:
            return nil
        }
    }
}
