import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit
import ModelsFeature

/// The "Czytanie na głos" composer: pick a model and a voice, type text,
/// speak it, listen to the result and save it as `.wav`.
struct SpeakSection: View {
    @Environment(SpeechStore.self) private var store
    @Environment(ModelsStore.self) private var models

    @State private var isSavingWav = false
    @State private var wavDocument: SpeechAudioDocument?

    private var ttsModels: [LocalModelDTO] {
        models.library.filter { $0.capabilities.contains("tts") }
    }

    var body: some View {
        @Bindable var store = store

        VStack(alignment: .leading, spacing: 16) {
            modelSection
            textSection
            voiceSection
            speedSection
            speakSection
            resultSection
        }
        .onChange(of: ttsModels, initial: true) { _, list in
            guard store.ttsModelKey == nil || !list.contains(where: { $0.key == store.ttsModelKey }) else { return }
            if let key = (list.first { $0.loaded } ?? list.first)?.key {
                store.selectTTSModel(key)
            }
        }
        .fileExporter(isPresented: $isSavingWav,
                     document: wavDocument,
                     contentType: UTType(filenameExtension: "wav") ?? .audio,
                     defaultFilename: store.speakResult?.suggestedFilename ?? "mowa.wav") { result in
            if case let .failure(error) = result {
                store.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: Model

    @ViewBuilder
    private var modelSection: some View {
        if ttsModels.isEmpty {
            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Brak modeli syntezy mowy")
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Zainstaluj model czytający na głos (np. Kokoro) z kategorii „Mowa” w sklepie z modelami.")
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
            LabeledField("Model syntezy mowy") {
                ModelPickerMenu(models: ttsModels,
                                selection: store.ttsModelKey,
                                icon: "speaker.wave.2") { key in
                    store.selectTTSModel(key)
                }
            }
        }
    }

    // MARK: Text

    private var textSection: some View {
        @Bindable var store = store

        return LabeledField("Tekst do przeczytania") {
            StyledTextEditor(placeholder: "Wpisz tekst, który LocalCode ma przeczytać na głos…",
                             text: $store.text,
                             minHeight: 120,
                             maxHeight: 220)
        }
    }

    // MARK: Voice

    @ViewBuilder
    private var voiceSection: some View {
        LabeledField("Głos") {
            if store.isLoadingVoices {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Wczytuję głosy…")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                }
            } else if store.voices.isEmpty {
                Text("Ten model nie zgłosił żadnego głosu.")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
            } else {
                Menu {
                    ForEach(store.voices) { voice in
                        Button {
                            store.selectedVoice = voice.id
                        } label: {
                            if voice.id == store.selectedVoice {
                                Label(voiceTitle(voice), systemImage: "checkmark")
                            } else {
                                Text(voiceTitle(voice))
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 7) {
                        Image(systemName: "person.wave.2")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(Theme.fgMuted)
                        Text(selectedVoiceTitle)
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.fg)
                            .lineLimit(1)
                    }
                }
                .menuStyle(.borderlessButton)
                .padding(.horizontal, 11)
                .frame(maxWidth: .infinity, minHeight: 40, alignment: .leading)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
            }
        }
    }

    private var selectedVoiceTitle: String {
        guard let selected = store.voices.first(where: { $0.id == store.selectedVoice }) else {
            return "Wybierz głos"
        }
        return voiceTitle(selected)
    }

    private func voiceTitle(_ voice: SpeechVoice) -> String {
        guard let language = voice.language, !language.isEmpty else { return voice.label }
        return "\(voice.label) · \(language)"
    }

    // MARK: Speed

    private var speedSection: some View {
        @Bindable var store = store

        return LabeledField("Szybkość mówienia",
                            description: "1,0× to naturalne tempo modelu.") {
            HStack(spacing: 10) {
                Slider(value: $store.speed, in: 0.5...2.0, step: 0.05)
                Text(String(format: "%.2f×", store.speed).replacingOccurrences(of: ".", with: ","))
                    .font(.system(size: 12, design: .rounded))
                    .foregroundStyle(Theme.fg)
                    .monospacedDigit()
                    .frame(width: 46, alignment: .trailing)
            }
        }
    }

    // MARK: Speak

    private var speakSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            if store.isSpeaking {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Czytam na głos…")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 0)
                    SecondaryButton(title: "Anuluj", height: 32) { store.cancelSpeak() }
                }
            } else {
                PrimaryButton(title: "Czytaj na głos",
                              systemImage: "waveform",
                              isEnabled: store.canSpeak,
                              fullWidth: true,
                              height: 42) {
                    store.speak()
                }
                .help(store.canSpeak
                      ? "Przeczytaj ten tekst na głos na tym Macu"
                      : "Wybierz model, głos i wpisz tekst")
            }
        }
    }

    // MARK: Result

    @ViewBuilder
    private var resultSection: some View {
        if let result = store.speakResult {
            VStack(alignment: .leading, spacing: 10) {
                SectionHeader("Nagranie")

                HStack(spacing: 10) {
                    if let voice = result.voice {
                        MetaBadgeView(text: voice, systemImage: "person.wave.2")
                    }
                    if let duration = result.durationSeconds {
                        MetaBadgeView(text: Format.duration(duration), systemImage: "clock")
                    }
                    MetaBadgeView(text: Format.bytes(result.sizeBytes), systemImage: "doc")
                    Spacer(minLength: 0)
                }

                HStack(spacing: 9) {
                    SecondaryButton(title: store.isPlaying ? "Zatrzymaj" : "Odtwórz",
                                    systemImage: store.isPlaying ? "stop.fill" : "play.fill",
                                    height: 34) {
                        store.togglePlayback()
                    }
                    SecondaryButton(title: "Zapisz jako…", systemImage: "square.and.arrow.down", height: 34) {
                        guard let data = store.speakResultData() else {
                            store.errorMessage = "Nie udało się odczytać zapisanego nagrania."
                            return
                        }
                        wavDocument = SpeechAudioDocument(data: data)
                        isSavingWav = true
                    }
                    SecondaryButton(title: "Pokaż w Finderze", systemImage: "folder", height: 34) {
                        store.revealSpeakResultInFinder()
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }
}
