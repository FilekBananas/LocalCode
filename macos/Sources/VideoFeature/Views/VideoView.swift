import SwiftUI
import AVKit
import LocalCodeKit
import ModelsFeature

/// "Wideo" — a few seconds of footage from a sentence, computed on this Mac.
///
/// The screen is shaped by the cost: a second of video is roughly a minute of
/// computing here, so the estimate is on screen *before* the button is pressed,
/// the denoising steps are visible while it runs, and the finished clip says
/// how long it really took. See `docs/VIDEO.md`.
public struct VideoView: View {
    @Environment(VideoStore.self) private var store

    /// The clip the player sheet is showing, if any.
    @State private var playing: VideoDTO?

    /// False renders only the scrollable contents, without the header and
    /// without the `ScrollView` — that is what `ImageRenderer` can lay out
    /// (see `macos/Sources/ScreenShots`).
    private let showsChrome: Bool

    public init() { self.showsChrome = true }

    #if DEBUG
    public init(showsChrome: Bool) { self.showsChrome = showsChrome }
    #endif

    public var body: some View {
        @Bindable var store = store
        if !showsChrome {
            content.background(Theme.bg)
        } else {
        VStack(spacing: 0) {
            header
            if let message = store.errorMessage {
                Banner(.error, message: message, onDismiss: { store.errorMessage = nil })
                    .padding(.horizontal, 20)
                    .padding(.top, 12)
            }
            ScrollView { content }
        }
        .background(Theme.bg)
        .task {
            await store.refreshModels()
            store.scheduleEstimate(immediately: true)
        }
        .onChange(of: store.shape) { _, _ in store.scheduleEstimate() }
        .sheet(item: $playing) { clip in
            VStack(spacing: 0) {
                VideoPlayer(player: AVPlayer(url: store.url(for: clip)))
                    .frame(minWidth: 640, minHeight: 420)
                HStack {
                    Spacer(minLength: 0)
                    Button("Zamknij") { playing = nil }
                        .buttonStyle(.bordered)
                }
                .padding(12)
            }
            .background(Theme.bg)
        }
        }
    }

    /// The first frame of a clip, with a play badge over it.
    private struct ClipPoster: View {
        @Environment(VideoStore.self) private var store
        let clip: VideoDTO
        @State private var image: NSImage?

        var body: some View {
            ZStack {
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    Rectangle()
                        .fill(Theme.bg)
                        .aspectRatio(CGFloat(max(1, clip.width)) / CGFloat(max(1, clip.height)),
                                     contentMode: .fit)
                        .overlay {
                            Image(systemName: "film")
                                .font(.system(size: 22))
                                .foregroundStyle(Theme.fgMuted)
                        }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: 280)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .task(id: clip.id) { image = await store.poster(for: clip) }
        }
    }

    /// Everything below the header. Named, because it is also what the layout
    /// check renders on its own — `ImageRenderer` cannot lay out a `ScrollView`.
    var content: some View {
        VStack(alignment: .leading, spacing: 18) {
            if store.runnableModels.isEmpty {
                noModelCard
            } else {
                composer
                if store.isGenerating { progressCard }
                if !store.clips.isEmpty { results }
            }
            if !store.unsupportedModels.isEmpty { unsupportedNote }
        }
        .padding(20)
        .frame(maxWidth: 720, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Wideo")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("Kilka sekund materiału z opisu — liczone na tym Macu, bez wysyłania niczego")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 8)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
    }

    // MARK: - No model

    private var noModelCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                Text("Nie masz modelu do generowania wideo")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Theme.fg)
                Text("Pobierz go w sklepie z modelami — kategoria „Wideo”. LocalCode uruchamia "
                     + "dziś LTX-Video 2B w wersji 0.9; nowsze warianty tej rodziny nie mają "
                     + "jeszcze potoku, do którego dałoby się je podłączyć.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Composer

    @ViewBuilder
    private var composer: some View {
        @Bindable var store = store
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Opis", subtitle: "Po angielsku model rozumie znacznie więcej niż po polsku") {
                EmptyView()
            }
            Card {
                VStack(alignment: .leading, spacing: 14) {
                    // The shared editor, like "Tekst do przeczytania" in Mowa: it
                    // brings the placeholder, the focus ring and — the reason this
                    // is not a hand-rolled TextEditor — a maximum height. Without
                    // one the editor grew to fill the card.
                    StyledTextEditor(placeholder: "a ripe yellow banana on a wooden table, daylight",
                                     text: $store.prompt,
                                     minHeight: 84,
                                     maxHeight: 140,
                                     font: .system(size: 13))

                    LabeledField("Model") {
                        Picker("Model", selection: Binding(
                            get: { store.selectedModel?.key ?? "" },
                            set: { store.modelKey = $0 })) {
                            ForEach(store.runnableModels) { model in
                                Text(model.displayName).tag(model.key)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 320, alignment: .leading)
                    }

                    LabeledField("Rozmiar",
                                 description: "Każdy piksel kosztuje czas — najmniejszy rozmiar liczy się mniej więcej o połowę krócej.") {
                        Picker("Rozmiar", selection: Binding(
                            get: { "\(store.shape.width)×\(store.shape.height)" },
                            set: { label in
                                if let size = VideoStore.sizes.first(where: { "\($0.width)×\($0.height)" == label }) {
                                    store.shape.width = size.width
                                    store.shape.height = size.height
                                }
                            })) {
                            ForEach(VideoStore.sizes, id: \.label) { size in
                                Text(size.label).tag("\(size.width)×\(size.height)")
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                    }

                    HStack(alignment: .top, spacing: 24) {
                        LabeledField("Klatki",
                                     description: "\(Self.seconds(store.shape.seconds)) s materiału przy \(store.shape.fps) kl/s") {
                            Stepper(value: $store.shape.numFrames, in: 9...161, step: 8) {
                                Text("\(store.shape.numFrames)")
                                    .font(.system(size: 13, weight: .semibold))
                                    .monospacedDigit()
                                    .foregroundStyle(Theme.fg)
                            }
                        }
                        LabeledField("Kroki",
                                     description: "Więcej kroków to więcej detalu i proporcjonalnie więcej czasu.") {
                            Stepper(value: $store.shape.steps, in: 4...60, step: 2) {
                                Text("\(store.shape.steps)")
                                    .font(.system(size: 13, weight: .semibold))
                                    .monospacedDigit()
                                    .foregroundStyle(Theme.fg)
                            }
                        }
                    }

                    if let estimate = store.estimate {
                        Text(estimate.message)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    HStack(spacing: 10) {
                        Button("Generuj") { store.generate() }
                            .buttonStyle(.borderedProminent)
                            .disabled(!store.canGenerate)
                        if store.isGenerating {
                            Button("Przerwij") { store.cancel() }
                                .buttonStyle(.bordered)
                        }
                        Spacer(minLength: 0)
                    }
                }
            }
        }
    }

    // MARK: - While it runs

    private var progressCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                if !store.statusMessage.isEmpty {
                    Text(store.statusMessage)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fg)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ProgressView(value: store.progressFraction)
                    .progressViewStyle(.linear)
                HStack {
                    Text(store.progressLabel.isEmpty ? "Przygotowuję…" : store.progressLabel)
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 8)
                    if let started = store.startedAt {
                        TimelineView(.periodic(from: .now, by: 1)) { context in
                            Text(Self.elapsed(from: started, to: context.date))
                                .font(.system(size: 11.5))
                                .monospacedDigit()
                                .foregroundStyle(Theme.fgMuted)
                        }
                    }
                }
            }
        }
    }

    static func elapsed(from start: Date, to now: Date) -> String {
        let seconds = max(0, Int(now.timeIntervalSince(start)))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    // MARK: - Results

    private var results: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Gotowe klipy", subtitle: "Zapisane w katalogu wideo LocalCode") {
                EmptyView()
            }
            ForEach(store.clips) { clip in
                Card {
                    VStack(alignment: .leading, spacing: 10) {
                        ClipPoster(clip: clip)
                        Text(clip.prompt)
                            .font(.system(size: 12))
                            .foregroundStyle(Theme.fg)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(Self.summary(of: clip))
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgMuted)
                        HStack(spacing: 10) {
                            Button("Odtwórz") { playing = clip }
                                .buttonStyle(.borderedProminent)
                            Button("Zapisz jako…") { store.save(clip) }
                                .buttonStyle(.bordered)
                            Button("Pokaż w Finderze") { store.reveal(clip) }
                                .buttonStyle(.bordered)
                            Spacer(minLength: 0)
                        }
                    }
                }
            }
        }
    }

    /// One decimal place, with the separator Polish actually uses.
    ///
    /// `String(format:)` is locale-independent and writes "1.0"; everything else
    /// on this screen says "1,0".
    static func seconds(_ value: Double) -> String {
        String(format: "%.1f", value).replacingOccurrences(of: ".", with: ",")
    }

    static func summary(of clip: VideoDTO) -> String {
        let elapsed = Double(clip.elapsedMs) / 1000
        let took = elapsed >= 90
            ? "\(seconds(elapsed / 60)) min"
            : String(format: "%.0f s", elapsed)
        return "\(clip.width)×\(clip.height) · \(clip.numFrames) klatek "
            + "(\(seconds(clip.seconds)) s) · \(clip.steps) kroków · ziarno \(clip.seed) · liczone \(took)"
    }

    // MARK: - Models we cannot run

    private var unsupportedNote: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                Text("Modele wideo, których nie uruchomimy")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.fg)
                ForEach(store.unsupportedModels) { model in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.displayName)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fg)
                        if let reason = model.reason {
                            Text(reason)
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgMuted)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            }
        }
    }
}


#if DEBUG
/// The contents of "Wideo" laid out flat, for rendering to an image
/// (see `SpeechScreenPreview` for why). Debug only.
public struct VideoScreenPreview: View {
    public init() {}
    public var body: some View {
        VideoView(showsChrome: false)
    }
}
#endif

