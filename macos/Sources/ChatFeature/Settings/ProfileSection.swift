import SwiftUI
import LocalCodeKit

/// "O mnie": the profile the model receives with every conversation that has
/// access to the global context.
struct ProfileSection: View {
    let draft: SettingsDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Te dane trafiają do każdej rozmowy, która ma dostęp do kontekstu globalnego — czyli do rozmów bez projektu i do projektów z pamięcią wspólną. Projekty z pamięcią izolowaną ich nie widzą.")
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )

            SettingsToggleRow("Używaj tych informacji w rozmowach",
                              description: "Wyłączenie zostawia dane zapisane, ale model ich nie dostaje — i zwalnia miejsce w oknie kontekstu.",
                              isOn: draft.binding(\.profile.enabled))

            Hairline()

            LabeledField("Jak się do Ciebie zwracać") {
                StyledTextField(placeholder: "np. Anna", text: draft.binding(\.profile.displayName))
            }

            LabeledField("Czym się zajmujesz") {
                StyledTextField(placeholder: "np. programista frontendowy",
                                text: draft.binding(\.profile.role))
            }

            LabeledField("O mnie", description: "Kontekst, który przyda się w większości rozmów.") {
                StyledTextEditor(placeholder: "np. Pracuję nad aplikacją w Next.js. Znam TypeScript, uczę się Rusta.",
                                 text: draft.binding(\.profile.about),
                                 minHeight: 110)
            }

            LabeledField("Styl odpowiedzi",
                         description: "Ton, długość, format. Podpowiedzi: „bez wstępów”, „zawsze podawaj przykład kodu”, „używaj list”.") {
                StyledTextEditor(placeholder: "np. Odpowiadaj zwięźle. Najpierw wniosek, potem uzasadnienie.",
                                 text: draft.binding(\.profile.responseStyle),
                                 minHeight: 110)
            }

            LabeledField("Instrukcje własne",
                         description: "Wszystko, czego nie da się zmieścić powyżej.") {
                StyledTextEditor(placeholder: "np. Przy kodzie zawsze dopisz krótką notkę o wydajności.",
                                 text: draft.binding(\.profile.customInstructions),
                                 minHeight: 110)
            }

            if !draft.value.profile.enabled {
                Banner(.info, message: "Profil jest wyłączony — te dane są zapisane, ale nie trafiają do modelu.")
            }
        }
    }
}

/// The main system prompt, with a character counter and a starter template.
struct SystemPromptSection: View {
    let draft: SettingsDraft

    /// A short starter: local models follow a handful of firm rules much better
    /// than a long essay, and every character here costs context.
    private static let example = """
    Jesteś moim asystentem do pracy i nauki.

    Zasady:
    - Odpowiadaj po polsku, konkretnie i bez zbędnych wstępów.
    - Gdy czegoś nie wiesz, powiedz to wprost zamiast zgadywać.
    - Kod podawaj w całości, gotowy do uruchomienia.
    """

    private var characterCount: Int { draft.value.globalSystemPrompt.count }

    /// Rough rule of thumb: ~4 characters per token for Polish text.
    private var approximateTokens: Int { max(0, characterCount / 4) }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader("Prompt systemowy",
                          subtitle: "Główna instrukcja dla modelu. Dostają ją rozmowy bez projektu i projekty z pamięcią wspólną.")

            Banner(.info,
                   message: "Modele lokalne mają małe okno kontekstu — każdy znak promptu zabiera miejsce na rozmowę. Trzymaj instrukcję krótką i konkretną.")

            StyledTextEditor(placeholder: "Opisz, jak model ma się zachowywać w każdej rozmowie…",
                             text: draft.binding(\.globalSystemPrompt),
                             minHeight: 260,
                             font: .system(size: 13, design: .monospaced))

            HStack(spacing: 10) {
                SecondaryButton(title: "Wstaw przykład", systemImage: "wand.and.stars", height: 34) {
                    insertExample()
                }
                if !draft.value.globalSystemPrompt.isEmpty {
                    SecondaryButton(title: "Wyczyść", systemImage: "trash", height: 34) {
                        draft.set(\.globalSystemPrompt, "")
                    }
                }
                Spacer(minLength: 0)
                Text("\(characterCount) znaków · ok. \(approximateTokens) tok.")
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(Theme.fgSubtle)
                    .monospacedDigit()
            }
        }
    }

    private func insertExample() {
        let current = draft.value.globalSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let next = current.isEmpty ? Self.example : "\(current)\n\n\(Self.example)"
        draft.set(\.globalSystemPrompt, next)
    }
}
