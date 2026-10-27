import Foundation
import SwiftUI
import LocalCodeKit

/// What the chat area shows before the first message: a greeting that follows
/// the clock, the active project's card when one is selected, and four starters
/// that drop straight into the composer.
struct WelcomeView: View {
    /// Preferred name from the profile.
    var displayName: String?
    /// Project the next message will belong to, if any.
    var folder: Folder?
    /// Shown when the library is empty — there is nothing to talk to yet.
    var hasModels: Bool = true
    var onSuggestion: (String) -> Void = { _ in }
    var onBrowseModels: () -> Void = {}

    var body: some View {
        VStack(spacing: 20) {
            greeting

            if !hasModels {
                noModelsCard
            }

            if let folder {
                projectCard(folder)
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 230, maximum: 320), spacing: 10)], spacing: 10) {
                ForEach(Suggestion.all(for: folder)) { suggestion in
                    ChatSuggestionCard(suggestion: suggestion) {
                        onSuggestion(suggestion.prompt)
                    }
                }
            }
            .frame(maxWidth: 640)
            .opacity(hasModels ? 1 : 0.45)
            .disabled(!hasModels)
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 8)
    }

    // MARK: - Greeting

    private var greeting: some View {
        VStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(Theme.accent)

            Text(headline)
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .multilineTextAlignment(.center)

            Text(folder == nil
                 ? "W czym mogę dziś pomóc?"
                 : "Piszesz w projekcie — kontekst zostanie dobrany automatycznie.")
                .font(.system(size: 14))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var headline: String {
        let name = displayName?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let first = name.split(separator: " ").first.map(String.init) ?? name
        return first.isEmpty ? Self.timeGreeting() : "\(Self.timeGreeting()), \(first)"
    }

    /// Polish greeting that follows the clock.
    static func timeGreeting(at date: Date = Date()) -> String {
        switch Calendar.current.component(.hour, from: date) {
        case 5..<12: "Dzień dobry"
        case 12..<18: "Miłego popołudnia"
        case 18..<23: "Dobry wieczór"
        default: "Dobrej nocy"
        }
    }

    // MARK: - Cards

    private var noModelsCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "shippingbox")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.accent)
                    Text("Nie masz jeszcze żadnego modelu")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Spacer(minLength: 0)
                }
                Text("Pobierz model w formacie GGUF albo MLX, a rozmowa ruszy od razu. Na 32 GB pamięci dobrze sprawdzają się modele 4–8 B w kwantyzacji 4-bitowej.")
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
                PrimaryButton(title: "Przejdź do Odkrywaj",
                              systemImage: "magnifyingglass",
                              fullWidth: false,
                              height: 36,
                              action: onBrowseModels)
            }
        }
        .frame(maxWidth: 640)
    }

    private func projectCard(_ folder: Folder) -> some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: Format.symbol(folder.icon))
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Format.hexColor(folder.color) ?? Theme.accent)
                    Text(folder.name)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(2)
                    Spacer(minLength: 6)
                    Chip(title: folder.memoryMode.title,
                         systemImage: folder.memoryMode == .shared ? "arrow.triangle.branch" : "lock",
                         isSelected: folder.memoryMode == .isolated)
                }

                if let description = folder.description?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !description.isEmpty {
                    Text(description)
                        .font(.system(size: 13))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Text(folder.memoryMode.explanation)
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgSubtle)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: 640)
    }
}

// MARK: - Suggestions

/// One starter card. `prompt` is what lands in the composer.
struct Suggestion: Identifiable {
    var id: String { title }
    let icon: String
    let title: String
    let subtitle: String
    let prompt: String

    static func all(for folder: Folder?) -> [Suggestion] {
        guard let folder else { return general }
        return [
            Suggestion(icon: "questionmark.bubble",
                       title: "Zapytaj o projekt",
                       subtitle: "Na podstawie plików i notatek",
                       prompt: "Na podstawie materiałów projektu „\(folder.name)” wyjaśnij mi "),
            Suggestion(icon: "text.alignleft",
                       title: "Podsumuj materiały",
                       subtitle: "Najważniejsze wnioski",
                       prompt: "Podsumuj najważniejsze wnioski z materiałów tego projektu."),
            general[1],
            general[3]
        ]
    }

    static let general: [Suggestion] = [
        Suggestion(icon: "lightbulb",
                   title: "Wyjaśnij zagadnienie",
                   subtitle: "Prosto i na przykładach",
                   prompt: "Wyjaśnij mi prosto, krok po kroku: "),
        Suggestion(icon: "chevron.left.forwardslash.chevron.right",
                   title: "Napisz kod",
                   subtitle: "Funkcja, skrypt albo poprawka",
                   prompt: "Napisz kod, który "),
        Suggestion(icon: "doc.text.magnifyingglass",
                   title: "Streść tekst",
                   subtitle: "Wyciąg w punktach",
                   prompt: "Streść poniższy tekst w punktach:\n\n"),
        Suggestion(icon: "checklist",
                   title: "Zaplanuj działanie",
                   subtitle: "Rozpisz kroki i terminy",
                   prompt: "Pomóż mi zaplanować ")
    ]
}

private struct ChatSuggestionCard: View {
    let suggestion: Suggestion
    let action: () -> Void

    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: suggestion.icon)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(Theme.accent)
                    .frame(width: 20, alignment: .leading)
                VStack(alignment: .leading, spacing: 2) {
                    Text(suggestion.title)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .multilineTextAlignment(.leading)
                    Text(suggestion.subtitle)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isHovering ? Theme.surface2 : Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(isHovering ? Theme.borderStrong : Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .onHover { isHovering = $0 }
        .animation(.easeOut(duration: 0.12), value: isHovering)
    }
}
