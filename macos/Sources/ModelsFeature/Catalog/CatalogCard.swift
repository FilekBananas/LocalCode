import SwiftUI
import LocalCodeKit

/// One model in the store.
///
/// Everything on the card comes from the row the list endpoint returned — the
/// name, what it is for, the size, the memory verdict and its explanation — so
/// a screen of thirty cards costs exactly one request.
struct CatalogCard: View {
    let entry: CatalogEntry
    let onInstall: () -> Void

    @Environment(ModelsStore.self) private var store
    @State private var isHovered = false

    private var isInstalling: Bool { store.isInstalling(entry) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            tagline
            Spacer(minLength: 0)
            badges
            footer
        }
        .padding(14)
        .frame(maxWidth: .infinity, minHeight: 208, alignment: .topLeading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(isHovered ? Theme.borderStrong : Theme.border, lineWidth: 1)
        )
        .onHover { isHovered = $0 }
        .animation(.easeOut(duration: 0.12), value: isHovered)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top, spacing: 10) {
            icon
            VStack(alignment: .leading, spacing: 2) {
                Text(entry.name)
                    .font(.system(size: 14.5, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 6) {
                    Text(entry.sizeLabel)
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                    if let quant = entry.quant, !quant.isEmpty {
                        // Which file you actually get: the store picks one, and
                        // not saying which turns "install" into a guess.
                        Text(quant)
                            .font(.system(size: 10.5, weight: .medium, design: .monospaced))
                            .foregroundStyle(Theme.fgSubtle)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Theme.surface2)
                            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                            .help(entry.quants.count > 1
                                  ? "Pobierze się ta wersja. Inne (\(entry.quants.joined(separator: ", "))) wybierzesz w „Szczegóły”."
                                  : "Wersja pliku, która się pobierze")
                    }
                    if !entry.paramsLabel.isEmpty {
                        Text("·").foregroundStyle(Theme.fgSubtle)
                        Text(entry.paramsLabel)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .lineLimit(1)
                            .help(entry.activeParamsB != nil
                                  ? "Model MoE: w pamięci mieści się całość, ale na każdy token pracuje tylko część — stąd szybkość większa, niż sugeruje rozmiar."
                                  : "Liczba parametrów modelu")
                    } else if !entry.speedHint.isEmpty {
                        Text("·").foregroundStyle(Theme.fgSubtle)
                        Text(entry.speedHint)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                            .lineLimit(1)
                    }
                }
            }
            Spacer(minLength: 0)
            if entry.requiresToken {
                Image(systemName: "lock.fill")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(Theme.warning)
                    .help("Wymaga tokenu Hugging Face — model na licencji wymagającej akceptacji")
            }
        }
    }

    /// The category icon (what the model is for) with the laboratory's mark in
    /// the corner (whose model it is). Two different questions, two marks — the
    /// logo never replaces the category symbol.
    private var icon: some View {
        RoundedRectangle(cornerRadius: 9, style: .continuous)
            .fill(categoryTint.opacity(0.14))
            .frame(width: 34, height: 34)
            .overlay(
                Image(systemName: categorySymbol)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(categoryTint)
            )
            .overlay(alignment: .bottomTrailing) {
                PublisherLogo(slug: entry.publisher.logo, size: 16, ringed: true)
                    .offset(x: 5, y: 5)
                    .help(entry.publisher.publisher.isEmpty
                          ? "" : "Model od: \(entry.publisher.publisher)")
            }
    }

    private var categorySymbol: String {
        CatalogCategory(id: entry.category, label: entry.category).systemImage
    }

    private var categoryTint: Color {
        switch entry.category {
        case "kodowanie": Theme.color(0x2B6CB0, 0x63B3ED)
        case "czat": Theme.accent
        case "rozumowanie": Theme.color(0x805AD5, 0xB794F4)
        case "obrazy": Theme.color(0xB83280, 0xF687B3)
        case "wizja": Theme.color(0x2C7A7B, 0x4FD1C5)
        case "embedding": Theme.color(0x975A16, 0xD69E2E)
        default: Theme.fgMuted
        }
    }

    @ViewBuilder
    private var tagline: some View {
        Text(entry.tagline)
            .font(.system(size: 12.5))
            .foregroundStyle(Theme.fgMuted)
            .lineLimit(entry.watchOut == nil ? 3 : 2)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
        if let watchOut = entry.watchOut, !watchOut.isEmpty {
            HStack(alignment: .top, spacing: 4) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 9.5, weight: .semibold))
                    .padding(.top, 2)
                Text(watchOut)
                    .font(.system(size: 11.5))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .foregroundStyle(Theme.warning)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Badges

    private var badges: some View {
        HStack(spacing: 6) {
            fitBadge
            TrustBadge(trust: entry.trust, publisher: entry.publisher)
            ForEach(entry.badges.prefix(1), id: \.self) { badge in
                MetaBadge(text: badge)
            }
            Spacer(minLength: 0)
        }
    }

    private var fitBadge: some View {
        HStack(spacing: 4) {
            Image(systemName: entry.fit.systemImage)
                .font(.system(size: 10, weight: .semibold))
            Text(entry.fit.label)
                .font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(fitTint)
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(fitTint.opacity(0.14))
        .clipShape(Capsule())
        .help(entry.fitDetail.isEmpty ? entry.fit.label : entry.fitDetail)
    }

    private var fitTint: Color {
        switch entry.fit {
        case .fits: Theme.success
        case .tight: Theme.warning
        case .tooBig: Theme.danger
        case .unknown: Theme.fgMuted
        }
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 8) {
            installControl
            Button {
                store.openCatalogDetail(entry)
            } label: {
                Text("Szczegóły")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(Theme.fgMuted)
                    .padding(.horizontal, 10)
                    .frame(minHeight: 32)
                    .background(Theme.surface2)
                    .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .strokeBorder(Theme.border, lineWidth: 1)
                    )
                    .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle())
            .help("Repozytorium, warianty i pliki")
        }
    }

    @ViewBuilder
    private var installControl: some View {
        if entry.installed {
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11, weight: .semibold))
                Text("Zainstalowany")
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(Theme.success)
            .padding(.horizontal, 11)
            .frame(minHeight: 32)
            .background(Theme.successSoft)
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .help(entry.installedKey.map { "Na dysku jako \($0)" } ?? "Model jest już na dysku")
        } else if entry.trust.excluded {
            HStack(spacing: 5) {
                Image(systemName: "xmark.shield")
                    .font(.system(size: 11, weight: .semibold))
                Text("Odrzucony")
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundStyle(Theme.danger)
            .padding(.horizontal, 11)
            .frame(minHeight: 32)
            .background(Theme.danger.opacity(0.14))
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .help(entry.trust.exclusion ?? "Model został odrzucony")
        } else if let job = store.job(for: entry) {
            // The person clicked here, so this is where the download reports
            // back — a ring in the window toolbar is easy to miss entirely.
            installProgress(job)
        } else {
            PrimaryButton(title: isInstalling ? "Instaluję…" : "Zainstaluj",
                          systemImage: isInstalling ? nil : "arrow.down.circle",
                          isLoading: isInstalling,
                          isEnabled: !isInstalling,
                          fullWidth: false,
                          height: 32,
                          action: onInstall)
                .help(entry.fitDetail.isEmpty
                      ? "Pobierz i zainstaluj ten model"
                      : "Pobierz i zainstaluj. \(entry.fitDetail)")
        }
    }

    /// Percentage, bar, speed and time left, in the width of the button.
    private func installProgress(_ job: DownloadJob) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(job.isCerberChecking ? "Cerber sprawdza…"
                     : job.isVerifying ? "Sprawdzam…" : "\(Int((job.progress * 100).rounded()))%")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .monospacedDigit()
                if job.totalBytes > 0 {
                    Text("\(Format.bytes(job.doneBytes)) / \(Format.bytes(job.totalBytes))")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgMuted)
                        .monospacedDigit()
                }
                Spacer(minLength: 0)
                if let eta = ModelFormat.eta(job.etaSeconds) {
                    Text(eta)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                        .monospacedDigit()
                }
                Button {
                    Task { await store.cancelDownload(job.id) }
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgSubtle)
                }
                .buttonStyle(.plain)
                .help("Przerwij pobieranie")
            }
            ProgressBar(value: job.totalBytes > 0 && !job.isVerifying ? job.progress : nil,
                        height: 5)
            if !job.isVerifying, job.speedBps > 1 {
                Text(ModelFormat.speed(job.speedBps))
                    .font(.system(size: 10.5))
                    .foregroundStyle(Theme.fgSubtle)
                    .monospacedDigit()
            }
        }
        .frame(minHeight: 32)
    }
}
