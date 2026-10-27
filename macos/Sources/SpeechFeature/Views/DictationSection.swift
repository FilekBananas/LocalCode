import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit
import ModelsFeature

/// The "Dyktowanie" composer: pick a model, pick or record audio, transcribe,
/// read the result and save it as `.srt`.
struct DictationSection: View {
    @Environment(SpeechStore.self) private var store
    @Environment(ModelsStore.self) private var models

    @State private var isSavingSRT = false
    @State private var srtDocument: SpeechTextDocument?

    private var sttModels: [LocalModelDTO] {
        models.library.filter { $0.capabilities.contains("stt") }
    }

    var body: some View {
        @Bindable var store = store

        VStack(alignment: .leading, spacing: 16) {
            modelSection
            sourceSection
            languageSection
            transcribeSection
            resultSection
        }
        .onChange(of: sttModels, initial: true) { _, list in
            guard store.sttModelKey == nil || !list.contains(where: { $0.key == store.sttModelKey }) else { return }
            store.sttModelKey = (list.first { $0.loaded } ?? list.first)?.key
        }
        .fileExporter(isPresented: $isSavingSRT,
                     document: srtDocument,
                     contentType: UTType(filenameExtension: "srt") ?? .plainText,
                     defaultFilename: store.srtSuggestedFilename) { result in
            if case let .failure(error) = result {
                store.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: Model

    @ViewBuilder
    private var modelSection: some View {
        if sttModels.isEmpty {
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Brak modeli do rozpoznawania mowy")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Zainstaluj model rozpoznawania mowy (np. Whisper) z kategorii „Mowa” w sklepie z modelami.")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                    PrimaryButton(title: "Otwórz sklep z modelami",
                                  systemImage: "shippingbox",
                                  fullWidth: true,
                                  height: 36) {
                        ShellBus.shared.request(.show(.discover))
                    }
                }
            }
        } else {
            LabeledField("Model rozpoznawania mowy") {
                ModelPickerMenu(models: sttModels,
                                selection: store.sttModelKey,
                                icon: "waveform") { key in
                    store.sttModelKey = key
                }
            }
        }
    }

    // MARK: Source

    private var sourceSection: some View {
        @Bindable var store = store

        return LabeledField("Nagranie", description: "Plik z dysku albo nagranie z mikrofonu.") {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    SecondaryButton(title: "Wybierz plik…", systemImage: "folder", height: 32) {
                        store.chooseAudioFile()
                    }
                    SecondaryButton(title: store.isRecording ? "Zatrzymaj nagrywanie" : "Nagraj z mikrofonu",
                                    systemImage: store.isRecording ? "stop.circle" : "mic",
                                    height: 32) {
                        store.toggleRecording()
                    }
                    if store.isRecording {
                        HStack(spacing: 5) {
                            Circle().fill(Theme.danger).frame(width: 6, height: 6)
                            Text(Format.duration(Double(store.recordingSeconds)))
                                .font(.system(size: 12, design: .rounded))
                                .foregroundStyle(Theme.danger)
                                .monospacedDigit()
                        }
                    }
                    Spacer(minLength: 0)
                }

                if let source = store.audioSourceLabel {
                    Chip(title: source, systemImage: "waveform", isSelected: true,
                        onRemove: { store.clearAudioSource() })
                }
            }
        }
    }

    // MARK: Language

    private var languageSection: some View {
        @Bindable var store = store

        return LabeledField("Język (opcjonalnie)",
                            description: "Zostaw puste, aby wykryć język automatycznie.") {
            StyledTextField(placeholder: "np. pl, en", text: $store.language)
                .frame(width: 160)
        }
    }

    // MARK: Transcribe

    private var transcribeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.isTranscribing {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Rozpoznaję mowę…")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 0)
                    SecondaryButton(title: "Anuluj", height: 32) { store.cancelTranscribe() }
                }
            } else {
                PrimaryButton(title: "Transkrybuj",
                              systemImage: "waveform.badge.magnifyingglass",
                              isEnabled: store.canTranscribe,
                              fullWidth: true,
                              height: 42) {
                    store.transcribe()
                }
                .help(store.canTranscribe
                      ? "Rozpoznaj tekst z nagrania na tym Macu"
                      : "Wybierz model i nagranie")
            }
        }
    }

    // MARK: Result

    @ViewBuilder
    private var resultSection: some View {
        if let transcript = store.transcript {
            VStack(alignment: .leading, spacing: 10) {
                SectionHeader("Transkrypcja")

                HStack(spacing: 10) {
                    if let language = transcript.language, !language.isEmpty {
                        MetaBadgeView(text: language, systemImage: "globe")
                    }
                    if let duration = transcript.durationSeconds {
                        MetaBadgeView(text: Format.duration(duration), systemImage: "clock")
                    }
                    MetaBadgeView(text: "\(transcript.elapsedMs) ms", systemImage: "bolt")
                    Spacer(minLength: 0)
                }

                Card(padding: 12) {
                    ScrollView {
                        Text(transcript.text.isEmpty ? "(Pusty wynik — nagranie mogło nie zawierać mowy.)"
                                                     : transcript.text)
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.fg)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 280)
                }

                HStack(spacing: 9) {
                    SecondaryButton(title: "Kopiuj tekst", systemImage: "doc.on.doc", height: 32) {
                        store.copyTranscript()
                    }
                    SecondaryButton(title: "Zapisz napisy (.srt)…",
                                    systemImage: "captions.bubble",
                                    isEnabled: !transcript.segments.isEmpty,
                                    height: 32) {
                        srtDocument = SpeechTextDocument(text: transcript.srt ?? "")
                        isSavingSRT = true
                    }
                    .help(transcript.segments.isEmpty
                          ? "Ta transkrypcja nie ma podziału na fragmenty czasowe"
                          : "Zapisz napisy .srt obok nagrania")
                    Spacer(minLength: 0)
                }
            }
        }
    }
}
