import Foundation
import SwiftUI
import AppKit
import AVFoundation
import UniformTypeIdentifiers
import LocalCodeKit

/// Everything the "Mowa" section knows: the dictation composer (pick or
/// record audio, transcribe, save subtitles) and the reading composer (type
/// text, pick a voice, speak it, save the wave file).
///
/// Unlike a chat or an image model, a speech model is never "loaded" from this
/// screen's point of view — `localcode/speech/runtime.py` loads it, uses it and
/// releases it inside one HTTP call — so this store, unlike `ImagesStore`,
/// keeps no server-side gallery of its own and needs no `/api/events` stream.
@MainActor
@Observable
public final class SpeechStore {

    // MARK: Dependencies

    @ObservationIgnored public let api: APIClient

    public init(api: APIClient) {
        self.api = api
    }

    deinit {
        recordingTimerTask?.cancel()
        transcribeTask?.cancel()
        speakTask?.cancel()
        playbackTask?.cancel()
    }

    // MARK: - Dictation (speech → text)

    public var sttModelKey: String?
    /// The file that will be transcribed: a picked file, or the last recording.
    public private(set) var audioPath: String?
    /// What the source picker shows — a file name, or "Nagranie (0:12)".
    public private(set) var audioSourceLabel: String?
    /// ISO language code, e.g. "pl"; empty means "wykryj automatycznie".
    public var language = ""

    public private(set) var isTranscribing = false
    public private(set) var transcript: Transcript?

    public private(set) var isRecording = false
    public private(set) var recordingSeconds = 0

    @ObservationIgnored private var transcribeTask: Task<Void, Never>?
    @ObservationIgnored private var recorder: AVAudioRecorder?
    @ObservationIgnored private var recordingURL: URL?
    @ObservationIgnored private var recordingTimerTask: Task<Void, Never>?

    public var canTranscribe: Bool {
        !isTranscribing && !isRecording && sttModelKey != nil && audioPath != nil
    }

    /// Extensions `localcode/speech/base.py` (`AUDIO_SUFFIXES`) accepts.
    static let audioExtensions = ["wav", "mp3", "m4a", "aac", "flac", "ogg", "opus",
                                  "webm", "mp4", "mov", "mkv", "aiff", "aif", "caf"]
    static let audioContentTypes: [UTType] = audioExtensions.compactMap { UTType(filenameExtension: $0) }

    /// "Wybierz plik…": any recording already on disk.
    public func chooseAudioFile() {
        if isRecording { stopRecording() }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Wybierz"
        panel.message = "Wskaż nagranie do transkrypcji."
        panel.allowedContentTypes = Self.audioContentTypes
        guard panel.runModal() == .OK, let url = panel.url else { return }
        audioPath = url.path
        audioSourceLabel = url.lastPathComponent
        transcript = nil
    }

    public func clearAudioSource() {
        if isRecording { stopRecording() }
        audioPath = nil
        audioSourceLabel = nil
        transcript = nil
    }

    /// "Nagraj z mikrofonu" / "Zatrzymaj nagrywanie".
    public func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            Task { await startRecording() }
        }
    }

    private func startRecording() async {
        errorMessage = nil
        guard await Self.requestMicrophoneAccess() else {
            errorMessage = "LocalCode nie ma dostępu do mikrofonu. Włącz go w Ustawieniach "
                + "systemowych → Prywatność i bezpieczeństwo → Mikrofon."
            return
        }
        // 16 kHz mono PCM: exactly what the Whisper backend wants, so there is
        // nothing for it to resample before transcribing.
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false,
        ]
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("localcode-dyktowanie-\(UUID().uuidString).wav")
        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            guard recorder.record() else {
                errorMessage = "Nie udało się uruchomić nagrywania."
                return
            }
            self.recorder = recorder
            self.recordingURL = url
            transcript = nil
            isRecording = true
            recordingSeconds = 0
            recordingTimerTask?.cancel()
            recordingTimerTask = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    guard let self, !Task.isCancelled else { return }
                    self.recordingSeconds += 1
                }
            }
        } catch {
            errorMessage = "Nie udało się uruchomić nagrywania: \(error.localizedDescription)"
        }
    }

    private func stopRecording() {
        recordingTimerTask?.cancel()
        recordingTimerTask = nil
        recorder?.stop()
        recorder = nil
        isRecording = false
        guard let url = recordingURL else { return }
        audioPath = url.path
        audioSourceLabel = "Nagranie z mikrofonu (\(Format.duration(Double(recordingSeconds))))"
    }

    private static func requestMicrophoneAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    continuation.resume(returning: granted)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    public func transcribe() {
        guard let path = audioPath, let modelKey = sttModelKey else { return }
        errorMessage = nil
        isTranscribing = true
        transcript = nil
        let trimmedLanguage = language.trimmingCharacters(in: .whitespaces)
        let request = TranscribeRequest(path: path,
                                        modelKey: modelKey,
                                        language: trimmedLanguage.isEmpty ? nil : trimmedLanguage,
                                        segments: true,
                                        srt: true)
        transcribeTask?.cancel()
        transcribeTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.api.transcribe(request)
                self.transcript = result
                self.announce("Transkrypcja gotowa.")
            } catch is CancellationError {
                // "Anuluj" — nothing to report.
            } catch {
                self.report(error)
            }
            self.isTranscribing = false
        }
    }

    public func cancelTranscribe() {
        transcribeTask?.cancel()
        transcribeTask = nil
        isTranscribing = false
    }

    public func copyTranscript() {
        guard let text = transcript?.text, !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        announce("Skopiowano tekst.")
    }

    /// A file name for the `.srt` save panel, based on the audio source.
    public var srtSuggestedFilename: String {
        let base = (audioSourceLabel.map { ($0 as NSString).deletingPathExtension }) ?? "transkrypcja"
        return ExportDocument.filename(for: base, format: "srt")
    }

    // MARK: - Reading aloud (text → speech)

    public var ttsModelKey: String?
    public var text = ""
    public var selectedVoice: String?
    public var speed: Double = 1.0

    public private(set) var voices: [SpeechVoice] = []
    public private(set) var isLoadingVoices = false
    public private(set) var isSpeaking = false
    public private(set) var speakResult: VoiceResult?

    @ObservationIgnored private var speakTask: Task<Void, Never>?

    public var canSpeak: Bool {
        !isSpeaking && ttsModelKey != nil && selectedVoice != nil
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// Picking a model swaps its voice list; the server validates the model
    /// actually does text-to-speech and answers with a Polish error otherwise.
    public func selectTTSModel(_ key: String) {
        guard ttsModelKey != key else { return }
        ttsModelKey = key
        selectedVoice = nil
        voices = []
        Task { await self.loadVoices() }
    }

    public func loadVoices() async {
        guard let modelKey = ttsModelKey else { return }
        isLoadingVoices = true
        defer { isLoadingVoices = false }
        do {
            let list = try await api.speechVoices(modelKey: modelKey)
            voices = list
            if selectedVoice == nil || !list.contains(where: { $0.id == selectedVoice }) {
                selectedVoice = list.first?.id
            }
        } catch {
            report(error)
        }
    }

    public func speak() {
        guard canSpeak, let modelKey = ttsModelKey else { return }
        errorMessage = nil
        isSpeaking = true
        speakResult = nil
        stopPlayback()
        let request = SpeakRequest(text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                                   modelKey: modelKey,
                                   voice: selectedVoice,
                                   speed: speed)
        speakTask?.cancel()
        speakTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await self.api.speak(request)
                self.speakResult = result
                self.announce("Nagranie gotowe.")
            } catch is CancellationError {
                // "Anuluj" — nothing to report.
            } catch {
                self.report(error)
            }
            self.isSpeaking = false
        }
    }

    public func cancelSpeak() {
        speakTask?.cancel()
        speakTask = nil
        isSpeaking = false
    }

    /// The bytes of the last result, for "Zapisz jako…" — the file already
    /// sits on this machine (the server wrote it), so this is a plain read,
    /// not a network round trip.
    public func speakResultData() -> Data? {
        guard let path = speakResult?.path else { return nil }
        return try? Data(contentsOf: URL(fileURLWithPath: path))
    }

    public func revealSpeakResultInFinder() {
        guard let path = speakResult?.path, !path.isEmpty else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    // MARK: Playback

    public private(set) var isPlaying = false

    @ObservationIgnored private var player: AVAudioPlayer?
    @ObservationIgnored private var playbackTask: Task<Void, Never>?

    public func togglePlayback() {
        if isPlaying {
            stopPlayback()
            return
        }
        guard let path = speakResult?.path, !path.isEmpty else { return }
        do {
            let player = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: path))
            guard player.play() else { return }
            self.player = player
            isPlaying = true
            playbackTask?.cancel()
            playbackTask = Task { [weak self] in
                while true {
                    try? await Task.sleep(nanoseconds: 200_000_000)
                    guard let self, !Task.isCancelled else { return }
                    if self.player?.isPlaying != true {
                        self.isPlaying = false
                        self.player = nil
                        return
                    }
                }
            }
        } catch {
            report(error)
        }
    }

    public func stopPlayback() {
        playbackTask?.cancel()
        playbackTask = nil
        player?.stop()
        player = nil
        isPlaying = false
    }

    // MARK: - Messages

    public var errorMessage: String?
    public private(set) var statusMessage: String?
    @ObservationIgnored private var statusTask: Task<Void, Never>?

#if DEBUG
    /// A store filled with representative state, so the section can be rendered
    /// off-screen (`ImageRenderer`) and looked at — see
    /// `macos/Sources/ScreenShots`. Debug only; the shipped app is release.
    public static func preview(api: APIClient, transcribed: Bool = true,
                               spoken: Bool = true) -> SpeechStore {
        let store = SpeechStore(api: api)
        store.sttModelKey = "mlx-community/whisper-small-mlx"
        store.ttsModelKey = "mlx-community/Kokoro-82M-bf16"
        store.language = "pl"
        store.audioPath = "/Users/ty/Nagrania/rozmowa.wav"
        store.audioSourceLabel = "rozmowa.wav"
        store.text = "Banany rosną w ciepłych krajach i są bardzo zdrowe."
        store.voices = Self.previewVoices()
        store.selectedVoice = "af_heart"
        if transcribed { store.transcript = Transcript.preview() }
        if spoken { store.speakResult = VoiceResult.preview() }
        return store
    }

    private static func previewVoices() -> [SpeechVoice] {
        let json = """
        [{"id": "af_heart", "label": "Heart (kobiecy, US)", "language": "en-US"},
         {"id": "am_michael", "label": "Michael (męski, US)", "language": "en-US"}]
        """
        return (try? JSONDecoder().decode([SpeechVoice].self, from: Data(json.utf8))) ?? []
    }
#endif

    private func report(_ error: Error) {
        if error is CancellationError { return }
        errorMessage = APIError.describe(error)
    }

    private func announce(_ text: String) {
        statusMessage = text
        statusTask?.cancel()
        statusTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            guard !Task.isCancelled else { return }
            self?.statusMessage = nil
        }
    }
}
