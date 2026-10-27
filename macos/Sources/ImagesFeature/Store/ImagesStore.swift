import Foundation
import SwiftUI
import AppKit
import LocalCodeKit

/// Everything the "Obrazy" section knows: which image models exist, the
/// parameters of the next generation, the job that is running, and the gallery.
///
/// One of these is created by the shell and injected with `.environment(...)`,
/// like the other feature stores, so it shares the app's single `APIClient` —
/// and therefore the same server, the same RAM budget and the same runtime the
/// language models are loaded into.
@MainActor
@Observable
public final class ImagesStore {

    // MARK: Dependencies

    @ObservationIgnored public let api: APIClient

    public init(api: APIClient) {
        self.api = api
    }

    deinit {
        generateTask?.cancel()
    }

    // MARK: Server state

    public private(set) var models: [ImageModelDTO] = []
    public private(set) var gallery: [ImageDTO] = []
    public private(set) var isLoadingModels = false
    public private(set) var isLoadingGallery = false
    /// True once the first round of requests came back.
    public private(set) var isReady = false

    // MARK: Composer state

    public var selectedModelKey: String?
    public var prompt = ""
    public var negativePrompt = ""
    public var showsNegativePrompt = false
    public var size: ImageSize = .square
    /// Used when `size == .custom`.
    public var customWidth = 1024
    public var customHeight = 1024
    /// `nil` means "use the model's own default".
    public var steps: Int?
    public var guidance: Double?
    /// `nil` is the "losowy" state; the seed of the last result can be put back
    /// here with "użyj ponownie".
    public var seed: Int?

    // MARK: Generation state

    public private(set) var isGenerating = false
    /// "Przygotowuję model …" while the worker subprocess starts.
    public private(set) var generationStatus: String?
    /// 0…1, or nil before the first progress frame.
    public private(set) var progress: Double?
    public private(set) var progressLabel: String?

    // MARK: Gallery state

    public var favoritesOnly = false
    /// The image open in the detail sheet.
    public var detail: ImageDTO?

    // MARK: Messages

    public var errorMessage: String?
    public private(set) var statusMessage: String?

    @ObservationIgnored private var generateTask: Task<Void, Never>?
    @ObservationIgnored private var statusTask: Task<Void, Never>?
    @ObservationIgnored private var didStart = false

    // MARK: - Lifecycle

    public func start() async {
        guard !didStart else { return }
        didStart = true
        await refresh()
        isReady = true
    }

    public func refresh() async {
        await refreshModels()
        await refreshGallery()
    }

    public func refreshModels() async {
        isLoadingModels = true
        defer { isLoadingModels = false }
        do {
            models = try await api.imageModels()
            if selectedModelKey == nil || !models.contains(where: { $0.key == selectedModelKey }) {
                // Prefer something already in memory, then the smallest model:
                // on a Mac with a shared RAM budget, the cheapest thing that
                // works is the right default.
                selectedModelKey = (models.first { $0.loaded } ?? models.min { $0.sizeBytes < $1.sizeBytes })?.key
                applyModelDefaults()
            }
        } catch {
            report(error)
        }
    }

    public func refreshGallery() async {
        isLoadingGallery = true
        defer { isLoadingGallery = false }
        do {
            gallery = try await api.images(limit: 120, favoritesOnly: favoritesOnly)
        } catch {
            report(error)
        }
    }

    // MARK: - Model selection

    public var selectedModel: ImageModelDTO? {
        models.first { $0.key == selectedModelKey }
    }

    public func selectModel(_ key: String) {
        guard selectedModelKey != key else { return }
        selectedModelKey = key
        applyModelDefaults()
    }

    /// Steps, guidance and the pixel size follow the model unless the user set
    /// them by hand; FLUX schnell at SD 1.5's 20 steps is five times slower for
    /// nothing, and SD 1.5 at FLUX's 1024² duplicates the subject.
    private func applyModelDefaults() {
        guard let model = selectedModel else { return }
        steps = model.defaults.steps
        guidance = model.defaults.guidance
        adoptModelSize(model)
    }

    /// Move to a resolution this model was trained for, when it names any and
    /// the current one is not among them. The server lists the native size
    /// first (`localcode/images/base.py`), so that is the one to fall back to.
    private func adoptModelSize(_ model: ImageModelDTO) {
        let offered = model.defaults.sizes.compactMap(Self.dimensions(ofSize:))
        guard !offered.isEmpty else { return }
        let current = resolvedSize
        guard !offered.contains(where: { $0 == current }) else { return }
        guard let wanted = offered.first else { return }
        if let preset = ImageSize.matching(width: wanted.width, height: wanted.height) {
            size = preset
        } else {
            size = .custom
            customWidth = wanted.width
            customHeight = wanted.height
        }
    }

    /// `"512x768"` → `(512, 768)`.
    private static func dimensions(ofSize text: String) -> (width: Int, height: Int)? {
        let parts = text.lowercased().split(separator: "x")
        guard parts.count == 2, let width = Int(parts[0]), let height = Int(parts[1]) else { return nil }
        return (width, height)
    }

    /// Whether the chosen model fits in what is left of the memory budget.
    /// The runtime evicts language models to make room, so this is a hint about
    /// *speed*, not a hard verdict — hence the wording.
    public func fitHint(for model: ImageModelDTO, availableBytes: Int64) -> ImageFit {
        guard availableBytes > 0 else { return .unknown }
        let needed = model.memoryCost
        if needed <= Int64(Double(availableBytes) * 0.7) { return .fits }
        if needed <= availableBytes { return .tight }
        return .tooBig
    }

    // MARK: - Generation

    public var canGenerate: Bool {
        !isGenerating
            && selectedModelKey != nil
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The pixel size the next generation will use.
    public var resolvedSize: (width: Int, height: Int) {
        switch size {
        case .custom:
            return (clampDimension(customWidth), clampDimension(customHeight))
        default:
            return size.dimensions
        }
    }

    /// Diffusion models work in multiples of 64; anything else is rounded for
    /// the user rather than rejected.
    private func clampDimension(_ value: Int) -> Int {
        let rounded = Int((Double(value) / 64).rounded()) * 64
        return min(2048, max(256, rounded))
    }

    public func generate() {
        guard canGenerate, let modelKey = selectedModelKey else { return }
        let dimensions = resolvedSize
        let request = ImageRequest(modelKey: modelKey,
                                   prompt: prompt.trimmingCharacters(in: .whitespacesAndNewlines),
                                   negativePrompt: showsNegativePrompt
                                       ? negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                                       : nil,
                                   width: dimensions.width,
                                   height: dimensions.height,
                                   steps: steps,
                                   guidance: guidance,
                                   seed: seed)

        errorMessage = nil
        isGenerating = true
        progress = nil
        progressLabel = nil
        generationStatus = "Przygotowuję…"

        generateTask?.cancel()
        generateTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = try await self.api.generateImage(request)
                for try await event in stream {
                    if Task.isCancelled { break }
                    self.apply(event)
                }
            } catch is CancellationError {
                // The user pressed "Przerwij"; nothing to report.
            } catch {
                self.report(error)
            }
            self.finishGeneration()
        }
    }

    public func cancelGeneration() {
        generateTask?.cancel()
        generateTask = nil
        finishGeneration()
        announce("Przerwano generowanie.")
    }

    private func apply(_ event: ImageStreamEvent) {
        switch event {
        case let .status(message):
            generationStatus = message

        case let .progress(_, step, steps):
            generationStatus = nil
            progress = min(1, Double(step) / Double(max(1, steps)))
            progressLabel = "krok \(step) z \(steps)"

        case let .image(image):
            gallery.insert(image, at: 0)
            // The seed the server actually used, so "użyj ponownie" can repeat
            // exactly this picture.
            lastSeed = image.seed

        case .done:
            progress = 1

        case let .failure(message):
            errorMessage = message
        }
    }

    private func finishGeneration() {
        isGenerating = false
        generationStatus = nil
        progress = nil
        progressLabel = nil
        generateTask = nil
    }

    /// The seed of the most recent result, offered by "użyj ponownie".
    public private(set) var lastSeed: Int?

    /// Puts the last result's seed into the composer, so the next run repeats it.
    public func reuseLastSeed() {
        guard let lastSeed else { return }
        seed = lastSeed
    }

    public func randomiseSeed() {
        seed = nil
    }

    // MARK: - Reuse

    /// "Użyj ponownie": loads every parameter of an image back into the composer.
    public func reuse(_ image: ImageDTO) {
        prompt = image.authoredPrompt
        negativePrompt = image.negativePrompt
        showsNegativePrompt = !image.negativePrompt.isEmpty
        if models.contains(where: { $0.key == image.modelKey }) {
            selectedModelKey = image.modelKey
        }
        steps = image.steps > 0 ? image.steps : nil
        guidance = image.guidance
        seed = image.seed
        if let match = ImageSize.matching(width: image.width, height: image.height) {
            size = match
        } else {
            size = .custom
            customWidth = image.width
            customHeight = image.height
        }
        detail = nil
        announce("Ustawienia obrazu wczytane — możesz je zmienić i wygenerować ponownie.")
    }

    // MARK: - Gallery actions

    public func toggleFavorite(_ image: ImageDTO) async {
        let target = !image.favorite
        replace(image.id) { $0.favorite = target }
        do {
            let updated = try await api.setImageFavorite(id: image.id, favorite: target)
            replace(updated.id) { $0 = updated }
            if favoritesOnly, !updated.favorite {
                gallery.removeAll { $0.id == updated.id }
            }
        } catch {
            replace(image.id) { $0.favorite = !target }
            report(error)
        }
    }

    public func delete(_ image: ImageDTO) async {
        do {
            try await api.deleteImage(id: image.id)
            gallery.removeAll { $0.id == image.id }
            if detail?.id == image.id { detail = nil }
            announce("Obraz usunięty.")
        } catch {
            report(error)
        }
    }

    /// "Pokaż w Finderze". The file is on this machine, so this is a real
    /// reveal rather than a download.
    public func revealInFinder(_ image: ImageDTO) {
        guard let path = image.path, !path.isEmpty else {
            errorMessage = "Serwer nie podał ścieżki pliku dla tego obrazu."
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    /// The PNG bytes of one image, by id. Used by the thumbnails, which must
    /// go through the authenticated client rather than a bare URL.
    /// Returns nil quietly: a picture that fails to load shows its own
    /// placeholder and does not deserve a banner across the screen.
    public func imageData(byId id: String) async -> Data? {
        try? await api.imageData(id: id)
    }

    /// "Zapisz jako…" — the bytes for a save panel.
    public func imageData(_ image: ImageDTO) async -> Data? {
        do {
            return try await api.imageData(id: image.id)
        } catch {
            report(error)
            return nil
        }
    }

    public func url(for image: ImageDTO) async -> URL {
        await api.imageURL(image)
    }

    private func replace(_ id: String, _ mutate: (inout ImageDTO) -> Void) {
        if let index = gallery.firstIndex(where: { $0.id == id }) {
            mutate(&gallery[index])
        }
        if detail?.id == id, var value = detail {
            mutate(&value)
            detail = value
        }
    }

    // MARK: - Messages

    private func report(_ error: Error) {
        if error is CancellationError { return }
        errorMessage = APIError.describe(error)
    }

    public func announce(_ text: String) {
        statusMessage = text
        statusTask?.cancel()
        statusTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard !Task.isCancelled else { return }
            self?.statusMessage = nil
        }
    }
}

// MARK: - Size presets

/// The three shapes people actually ask for, plus a way out.
public enum ImageSize: String, CaseIterable, Identifiable, Hashable, Sendable {
    case square
    case portrait
    case landscape
    case custom

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .square: "Kwadrat"
        case .portrait: "Pion"
        case .landscape: "Poziom"
        case .custom: "Własny"
        }
    }

    public var dimensions: (width: Int, height: Int) {
        switch self {
        case .square: (1024, 1024)
        case .portrait: (768, 1024)
        case .landscape: (1024, 768)
        case .custom: (1024, 1024)
        }
    }

    public var hint: String {
        switch self {
        case .custom: "Własny rozmiar"
        default: "\(dimensions.width) × \(dimensions.height)"
        }
    }

    /// Which preset an existing image matches, if any.
    public static func matching(width: Int, height: Int) -> ImageSize? {
        allCases.first { preset in
            preset != .custom
                && preset.dimensions.width == width
                && preset.dimensions.height == height
        }
    }
}

/// The memory hint beside an image model in the picker.
public enum ImageFit: Sendable {
    case fits
    case tight
    case tooBig
    case unknown

    public var label: String {
        switch self {
        case .fits: "Zmieści się"
        case .tight: "Na styk"
        case .tooBig: "Za duży"
        case .unknown: "Nie wiadomo"
        }
    }

    public var systemImage: String {
        switch self {
        case .fits: "checkmark.circle.fill"
        case .tight: "exclamationmark.circle.fill"
        case .tooBig: "xmark.circle.fill"
        case .unknown: "questionmark.circle"
        }
    }

    public var tint: Color {
        switch self {
        case .fits: Theme.success
        case .tight: Theme.warning
        case .tooBig: Theme.danger
        case .unknown: Theme.fgMuted
        }
    }
}
