import Foundation
import SwiftUI
import AppKit
import AVKit
import UniformTypeIdentifiers
import LocalCodeKit

/// Everything the "Wideo" section knows: which models can run, what a clip of a
/// given shape will cost in time, and one generation at a time.
///
/// The screen is built around the waiting. A second of footage is about a
/// minute of computing on this hardware, so the estimate is fetched *before*
/// anything starts, the progress of every denoising step is shown while it
/// runs, and the result says how long it actually took — which is also how the
/// estimate in `localcode/video/base.py` was calibrated.
@MainActor
@Observable
public final class VideoStore {

    @ObservationIgnored public let api: APIClient

    public init(api: APIClient) {
        self.api = api
    }

    deinit {
        generateTask?.cancel()
        estimateTask?.cancel()
    }

    // MARK: - Models

    public private(set) var models: [VideoModelDTO] = []
    public var modelKey: String?

    /// Models this runtime can actually run — the only ones worth offering.
    public var runnableModels: [VideoModelDTO] { models.filter(\.supported) }

    /// Recognised video models that cannot be run, with the reason; shown as a
    /// footnote so "I have a video model, why can't I use it" has an answer.
    public var unsupportedModels: [VideoModelDTO] { models.filter { !$0.supported } }

    public var selectedModel: VideoModelDTO? {
        guard let key = modelKey else { return runnableModels.first }
        return models.first { $0.key == key } ?? runnableModels.first
    }

    public func refreshModels() async {
        do {
            models = try await api.videoModels()
            if modelKey == nil || !models.contains(where: { $0.key == modelKey && $0.supported }) {
                modelKey = runnableModels.first?.key
            }
        } catch {
            report(error)
        }
    }

    // MARK: - The shape of the clip

    public var prompt = ""
    public var negativePrompt = ""
    public var shape = VideoShape()
    public var seedText = ""

    /// Sizes worth offering: LTX wants multiples of 32 and gets slower with
    /// every pixel, so this is a short list, not a free-form field.
    public static let sizes: [(label: String, width: Int, height: Int)] = [
        ("512 × 320 (najszybciej)", 512, 320),
        ("704 × 480 (domyślnie)", 704, 480),
        ("768 × 512 (najładniej)", 768, 512),
    ]

    public var canGenerate: Bool {
        !isGenerating && selectedModel != nil && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    // MARK: - The estimate

    public private(set) var estimate: VideoEstimate?
    @ObservationIgnored private var estimateTask: Task<Void, Never>?

    /// Ask the server what this shape will cost. Debounced, because it is
    /// attached to sliders.
    public func scheduleEstimate(immediately: Bool = false) {
        estimateTask?.cancel()
        let shape = self.shape
        estimateTask = Task { [weak self] in
            if !immediately {
                try? await Task.sleep(for: .milliseconds(350))
                if Task.isCancelled { return }
            }
            guard let self else { return }
            do {
                let value = try await self.api.estimateVideo(shape)
                if !Task.isCancelled { self.estimate = value }
            } catch {
                // An estimate that cannot be fetched is not worth an error banner.
                if !Task.isCancelled { self.estimate = nil }
            }
        }
    }

    // MARK: - Generating

    public private(set) var isGenerating = false
    public private(set) var statusMessage = ""
    public private(set) var progressFraction = 0.0
    public private(set) var progressLabel = ""
    public private(set) var startedAt: Date?
    public private(set) var clips: [VideoDTO] = []
    /// Settable from the view so a banner can dismiss itself, like `SpeechStore`.
    public var errorMessage: String?

    @ObservationIgnored private var generateTask: Task<Void, Never>?

    public func generate() {
        guard canGenerate, let model = selectedModel else { return }
        let request = VideoRequest(
            prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
            shape: shape,
            modelKey: model.key,
            negativePrompt: negativePrompt.isEmpty ? nil : negativePrompt,
            seed: Int(seedText.trimmingCharacters(in: .whitespaces)),
            force: shape.numFrames > 81)
        errorMessage = nil
        statusMessage = ""
        progressFraction = 0
        progressLabel = ""
        isGenerating = true
        startedAt = Date()
        generateTask = Task { [weak self] in
            guard let self else { return }
            defer {
                self.isGenerating = false
                self.startedAt = nil
            }
            do {
                for try await event in try await self.api.generateVideo(request) {
                    if Task.isCancelled { return }
                    self.apply(event)
                }
            } catch is CancellationError {
                return
            } catch {
                self.report(error)
            }
        }
    }

    /// Stop the run; the server cancels the job when the stream closes.
    public func cancel() {
        generateTask?.cancel()
        generateTask = nil
        isGenerating = false
        startedAt = nil
        statusMessage = ""
    }

    func apply(_ event: VideoStreamEvent) {
        switch event {
        case .status(let message):
            statusMessage = message
        case .progress(_, let step, let steps, let message, let fraction):
            progressFraction = min(1, max(0, fraction))
            progressLabel = message.isEmpty ? "Krok \(step)/\(steps)" : message
        case .video(let clip):
            clips.insert(clip, at: 0)
            progressFraction = 1
        case .done:
            statusMessage = ""
            progressLabel = ""
        case .failure(let message):
            errorMessage = message
        }
    }

    // MARK: - The finished clip

    /// A file URL the player and the Finder can both use.
    public func url(for clip: VideoDTO) -> URL { URL(fileURLWithPath: clip.path) }

    /// The first frame of a clip, for the list.
    ///
    /// A list of `VideoPlayer`s would mean one `AVPlayerView` per clip — heavy,
    /// and impossible to render off-screen (`AVPlayerView` cannot even be
    /// demangled without a window, which is how this was found). A still frame
    /// is what a list wants anyway; the player opens on demand.
    public func poster(for clip: VideoDTO) async -> NSImage? {
        let url = url(for: clip)
        return await Task.detached(priority: .utility) { () -> NSImage? in
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 1024, height: 1024)
            guard let cgImage = try? await generator.image(at: .zero).image else { return nil }
            return NSImage(cgImage: cgImage, size: .zero)
        }.value
    }

    public func reveal(_ clip: VideoDTO) {
        NSWorkspace.shared.activateFileViewerSelecting([url(for: clip)])
    }

    /// Copy the clip somewhere the user chooses.
    public func save(_ clip: VideoDTO) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = URL(fileURLWithPath: clip.path).lastPathComponent
        guard panel.runModal() == .OK, let target = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.copyItem(at: url(for: clip), to: target)
        } catch {
            errorMessage = "Nie udało się zapisać pliku: \(error.localizedDescription)"
        }
    }

    // MARK: - Rendering the screen without a server

#if DEBUG
    /// A store filled with representative state, so the section can be rendered
    /// off-screen (`ImageRenderer`) and looked at.
    ///
    /// This exists because nobody can click through this window in the
    /// environment it is built in — there is no accessibility permission for UI
    /// scripting — and "it compiles" says nothing about whether a label is
    /// clipped or a card is 40 points too tall. Debug only; the shipped app is
    /// built in release.
    public static func preview(api: APIClient,
                               generating: Bool = false,
                               withClip: Bool = true) -> VideoStore {
        let store = VideoStore(api: api)
        store.models = [
            VideoModelDTO(key: "city96/LTX-Video-gguf/ltx-video-2b-v0.9-Q4_K_M",
                          displayName: "ltx video 2b v0.9 · Q4_K_M", sizeBytes: 1_416_032_768),
            VideoModelDTO(key: "city96/LTX-Video-0.9.6-distilled-gguf/ltxv-2b-0.9.6-distilled-04-25-Q4_K_M",
                          displayName: "ltxv 2b 0.9.6 distilled · Q4_K_M",
                          sizeBytes: 1_330_049_536, videoKind: "ltx-other", supported: false,
                          reason: "To inny wariant LTX niż ten, który LocalCode potrafi uruchomić — "
                                + "dla wersji 0.9.6 nie ma potoku w formacie diffusers."),
        ]
        store.modelKey = store.models[0].key
        store.prompt = "a ripe yellow banana on a wooden table, daylight"
        store.shape = VideoShape(width: 704, height: 480, numFrames: 25, fps: 24, steps: 8)
        store.estimate = VideoEstimate.preview(
            seconds: 86.8,
            message: "25 klatek (1,0 s materiału przy 24 kl/s), 704x480, 8 kroków — "
                   + "szacunkowo ~1 min liczenia.")
        if generating {
            store.isGenerating = true
            store.startedAt = Date(timeIntervalSinceNow: -47)
            store.statusMessage = "Przygotowuję ltx video 2b v0.9."
            store.progressFraction = 0.375
            store.progressLabel = "Odszumianie: krok 3/8"
        }
        if withClip {
            store.clips = [VideoDTO.preview()]
        }
        return store
    }
#endif

    // MARK: - Errors

    private func report(_ error: Error) {
        if error is CancellationError { return }
        errorMessage = APIError.describe(error)
    }
}
