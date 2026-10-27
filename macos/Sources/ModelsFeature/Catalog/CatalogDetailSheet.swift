import SwiftUI
import AppKit
import LocalCodeKit

/// "Szczegóły / Zaawansowane" for one store entry.
///
/// This is where everything the card refused to show lives: the repository id,
/// every quantisation with its size and its files, and the same model in the
/// other runtime format. Nothing here is required to install a model — the
/// server picks the variant — but a person who wants to know exactly what is
/// about to land on their disk can find out.
struct CatalogDetailSheet: View {
    let entry: CatalogEntry

    @Environment(ModelsStore.self) private var store

    /// The freshly fetched entry replaces the row the card handed over.
    private var current: CatalogEntry { store.catalogDetail ?? entry }

    var body: some View {
        VStack(spacing: 0) {
            header
            Hairline()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    summary
                    fitSection
                    sourceSection
                    variantsSection
                    if !current.alternatives.isEmpty { alternativesSection }
                }
                .padding(20)
            }
            Hairline()
            footer
        }
        .frame(width: 620, height: 600)
        .background(Theme.bg)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                Text(current.name)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text(current.technicalSummary)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
            }
            Spacer(minLength: 8)
            if store.isLoadingCatalogDetail {
                ProgressView().controlSize(.small)
            }
            IconButton(systemName: "xmark", label: "Zamknij") { store.closeCatalogDetail() }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(Theme.surface)
    }

    // MARK: - Sections

    private var summary: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(current.tagline)
                .font(.system(size: 13))
                .foregroundStyle(Theme.fg)
                .fixedSize(horizontal: false, vertical: true)
            if !current.badges.isEmpty {
                HStack(spacing: 6) {
                    ForEach(current.badges, id: \.self) { MetaBadge(text: $0) }
                    Spacer(minLength: 0)
                }
            }
        }
    }

    private var fitSection: some View {
        Card {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 7) {
                    Image(systemName: current.fit.systemImage)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(fitTint)
                    Text(current.fit.label)
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Spacer(minLength: 0)
                    if !current.speedHint.isEmpty {
                        MetaBadge(text: current.speedHint, systemImage: "speedometer")
                    }
                }
                if !current.fitDetail.isEmpty {
                    Text(current.fitDetail)
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var fitTint: Color {
        switch current.fit {
        case .fits: Theme.success
        case .tight: Theme.warning
        case .tooBig: Theme.danger
        case .unknown: Theme.fgMuted
        }
    }

    private var sourceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader("Źródło")
            if current.trust.excluded, let exclusion = current.trust.exclusion {
                Banner(.error, message: exclusion)
            }
            if current.trust.score != nil {
                Card(padding: 12) {
                    TrustExplainer(trust: current.trust, publisher: current.publisher)
                }
            }
            Card(padding: 12) {
                HStack(spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(current.source.repo)
                            .font(.system(size: 12.5, design: .monospaced))
                            .foregroundStyle(Theme.fg)
                            .textSelection(.enabled)
                            .lineLimit(2)
                        Text(current.source.provider)
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                    Spacer(minLength: 0)
                    if let upstream = current.publisher.upstream, !upstream.isEmpty,
                       upstream.lowercased() != current.source.repo.lowercased() {
                        IconButton(systemName: "arrow.up.right.square",
                                   label: "Otwórz model źródłowy: \(upstream)") {
                            guard let url = URL(string: "https://huggingface.co/\(upstream)") else { return }
                            NSWorkspace.shared.open(url)
                        }
                    }
                    IconButton(systemName: "doc.on.doc", label: "Kopiuj identyfikator") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(current.source.repo, forType: .string)
                    }
                    IconButton(systemName: "safari", label: "Otwórz na Hugging Face") {
                        guard let url = URL(string: "https://huggingface.co/\(current.source.repo)") else { return }
                        NSWorkspace.shared.open(url)
                    }
                }
            }
            if current.requiresToken {
                Banner(.warning,
                       message: "Ten model wymaga tokenu Hugging Face i zaakceptowania licencji na stronie modelu.",
                       actionTitle: "Otwórz Ustawienia") {
                    store.closeCatalogDetail()
                    ShellBus.shared.request(.show(.settings))
                }
            }
        }
    }

    @ViewBuilder
    private var variantsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader("Warianty",
                          subtitle: "LocalCode sam wybiera wariant pod pamięć tego Maca — poniżej wszystko, co oferuje repozytorium.")
            if current.variants.isEmpty {
                Card(padding: 12) {
                    Text(store.isLoadingCatalogDetail
                         ? "Ładuję listę wariantów…"
                         : "Repozytorium nie zgłasza osobnych wariantów.")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                }
            } else {
                VStack(spacing: 6) {
                    ForEach(current.variants) { variant in
                        variantRow(variant)
                    }
                }
            }
        }
    }

    private func variantRow(_ variant: CatalogVariant) -> some View {
        let isRecommended = current.isRecommended(variant)
        return Card(padding: 11) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 7) {
                    Text(variant.label)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    if isRecommended {
                        Text("zalecany")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(Theme.accent)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(Theme.accentSoft)
                            .clipShape(Capsule())
                    }
                    Spacer(minLength: 0)
                    Text(Format.bytes(variant.sizeBytes))
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                        .monospacedDigit()
                    if !current.installed {
                        // Any quantisation, not only the recommended one: the
                        // store's pick is a default, not a decision.
                        SecondaryButton(title: isRecommended ? "Pobierz" : "Pobierz tę",
                                        height: 26) {
                            Task { await store.install(current, variant: variant) }
                        }
                        .help("Pobierz dokładnie tę wersję (\(variant.label))")
                    }
                }
                if !variant.files.isEmpty {
                    Text(variant.files.joined(separator: ", "))
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var alternativesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader("Ten sam model w innym formacie")
            VStack(spacing: 6) {
                ForEach(current.alternatives, id: \.repo) { source in
                    Card(padding: 11) {
                        HStack(spacing: 8) {
                            Text(source.repo)
                                .font(.system(size: 11.5, design: .monospaced))
                                .foregroundStyle(Theme.fgMuted)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .textSelection(.enabled)
                            Spacer(minLength: 0)
                            Text(source.provider)
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgSubtle)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 10) {
            Text(current.installed
                 ? "Model jest już zainstalowany."
                 : current.trust.excluded
                 ? "Model został odrzucony i nie może być zainstalowany."
                 : "Instalacja pobiera wariant dobrany do pamięci tego Maca.")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
            Spacer(minLength: 8)
            SecondaryButton(title: "Zamknij", height: 34) { store.closeCatalogDetail() }
            if !current.installed && !current.trust.excluded {
                PrimaryButton(title: "Zainstaluj",
                              systemImage: "arrow.down.circle",
                              isLoading: store.isInstalling(current),
                              isEnabled: !store.isInstalling(current),
                              fullWidth: false,
                              height: 34) {
                    let target = current
                    store.closeCatalogDetail()
                    Task { await store.install(target) }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Theme.surface)
    }
}
