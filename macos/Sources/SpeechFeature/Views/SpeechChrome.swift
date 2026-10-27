import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit
import ModelsFeature

/// Small pieces `DictationSection` and `SpeakSection` share: the model picker
/// (same menu-in-a-box `ImagesFeature`'s composer uses, since a speech model is
/// picked exactly the same way an image model is) and a neutral metadata badge.
/// `ModelsFeature` has both already, but as `internal` types only it can see.

/// A model picker: a bordered menu box with the size and "w pamięci" hint
/// underneath, exactly like `ImagesFeature`'s own model picker.
struct ModelPickerMenu: View {
    let models: [LocalModelDTO]
    let selection: String?
    var icon = "cpu"
    let onSelect: (String) -> Void

    private var selectedModel: LocalModelDTO? {
        models.first { $0.key == selection }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Menu {
                ForEach(models) { model in
                    Button {
                        onSelect(model.key)
                    } label: {
                        if model.key == selection {
                            Label(menuTitle(model), systemImage: "checkmark")
                        } else {
                            Text(menuTitle(model))
                        }
                    }
                }
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: icon)
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(Theme.fgMuted)
                    Text(selectedModel?.displayName ?? "Wybierz model")
                        .font(.system(size: 13))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                        .truncationMode(.middle)
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

            if let model = selectedModel {
                HStack(spacing: 7) {
                    Text(Format.bytes(model.sizeBytes))
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                    if model.loaded {
                        Text("w pamięci")
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.success)
                    }
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private func menuTitle(_ model: LocalModelDTO) -> String {
        var parts = [model.displayName, Format.bytes(model.sizeBytes)]
        if model.loaded { parts.append("załadowany") }
        return parts.joined(separator: " · ")
    }
}

/// Neutral metadata tag — a language, a duration, an elapsed time.
struct MetaBadgeView: View {
    let text: String
    var systemImage: String?

    var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage).font(.system(size: 9, weight: .semibold))
            }
            Text(text).font(.system(size: 11, weight: .medium))
        }
        .foregroundStyle(Theme.fgMuted)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Theme.surface2)
        .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 5, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}

// MARK: - File export

/// Carries the `.srt` text into `fileExporter`.
struct SpeechTextDocument: FileDocument {
    static var readableContentTypes: [UTType] { [UTType(filenameExtension: "srt") ?? .plainText] }

    var text: String

    init(text: String) { self.text = text }

    init(configuration: ReadConfiguration) throws {
        text = String(data: configuration.file.regularFileContents ?? Data(), encoding: .utf8) ?? ""
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

/// Carries the `.wav` bytes into `fileExporter`.
struct SpeechAudioDocument: FileDocument {
    static var readableContentTypes: [UTType] { [UTType(filenameExtension: "wav") ?? .audio] }

    var data: Data

    init(data: Data) { self.data = data }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
