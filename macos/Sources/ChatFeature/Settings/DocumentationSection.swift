import SwiftUI
import LocalCodeKit

/// "Dokumentacja": the user guide and the API reference that ship with the
/// app — open them in their own window or save both files to a folder.
struct DocumentationSection: View {
    @Environment(\.openWindow) private var openWindow

    @State private var notice: DocumentationModel.Notice?

    private let isAvailable = DocumentationLibrary.isAvailable

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SectionHeader("Dokumentacja",
                          subtitle: "Przewodnik po aplikacji i opis lokalnego API. Oba pliki są częścią aplikacji, więc działają bez internetu.")

            Card {
                VStack(alignment: .leading, spacing: 12) {
                    row(title: DocumentationPage.guide.title,
                        detail: "Czat, agent w zakładce Kod, pobieranie modeli i ustawienia — zadanie po zadaniu.")
                    Hairline()
                    row(title: DocumentationPage.api.title,
                        detail: "API.md opisuje serwer na localhost: API zgodne z OpenAI (/v1), endpointy /api/*, strumienie SSE i WebSocket agenta — z przykładami w curl, Pythonie i JavaScripcie.")

                    HStack(spacing: 10) {
                        PrimaryButton(title: "Otwórz dokumentację",
                                      systemImage: "book",
                                      isEnabled: isAvailable,
                                      fullWidth: false,
                                      height: 34) {
                            openWindow(id: DocumentationWindow.id)
                        }
                        SecondaryButton(title: "Pobierz…",
                                        systemImage: "arrow.down.circle",
                                        isEnabled: isAvailable,
                                        height: 34) {
                            download()
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.top, 4)

                    Text("Pobieranie zapisuje LocalCode-dokumentacja.md i LocalCode-API.md w wybranym folderze (domyślnie Pobrane) i niczego nie nadpisuje.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgSubtle)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if !isAvailable {
                Banner(.warning, message: DocumentationLibrary.unavailableMessage)
            } else if let notice {
                Banner(notice.kind, message: notice.message, onDismiss: { self.notice = nil })
            }
        }
    }

    private func row(title: String, detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text(detail)
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func download() {
        switch DocumentationExport.chooseFolderAndSave() {
        case let .success(saved)?:
            notice = .init(kind: .success, message: saved.message)
        case let .failure(failure)?:
            notice = .init(kind: .error, message: failure.localizedDescription)
        case nil:
            break
        }
    }
}
