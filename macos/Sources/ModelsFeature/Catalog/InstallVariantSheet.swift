import SwiftUI
import LocalCodeKit

/// "Zainstaluj": which version of the model to download.
///
/// A repository usually offers the same model at several quantisations — 2 to
/// 16 bits per weight, a few gigabytes apart. The store's own choice (best
/// quality that fits this Mac) is preselected, so Enter keeps the old one-click
/// behaviour; the list is there for the person with 64 GB who wants 8-bit, or
/// the one with 16 GB who would rather take 3-bit than nothing.
struct InstallVariantSheet: View {
    let entry: CatalogEntry
    let onClose: () -> Void

    @Environment(ModelsStore.self) private var store

    @State private var variants: [CatalogVariant] = []
    @State private var selectedId: String?
    @State private var isLoading = true
    @State private var loadFailed = false

    private var selected: CatalogVariant? { variants.first { $0.id == selectedId } }

    var body: some View {
        VStack(spacing: 0) {
            header
            Hairline()
            content
            Hairline()
            footer
        }
        .frame(width: 560, height: 520)
        .background(Theme.bg)
        .task { await load() }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Zainstaluj \(entry.name)")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .lineLimit(2)
            Text("Wybierz wersję. Mniej bitów to mniej pamięci i szybsze działanie, "
                 + "więcej bitów — wierniejsze odpowiedzi.")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(20)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if isLoading {
            VStack(spacing: 10) {
                ProgressView().controlSize(.small)
                Text("Sprawdzam dostępne wersje…")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if variants.isEmpty {
            VStack(spacing: 8) {
                Image(systemName: loadFailed ? "wifi.exclamationmark" : "shippingbox")
                    .font(.system(size: 22, weight: .light))
                    .foregroundStyle(Theme.fgSubtle)
                Text(loadFailed
                     ? "Nie udało się pobrać listy wersji."
                     : "To repozytorium ma tylko jedną wersję.")
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(Theme.fg)
                Text("Możesz zainstalować wersję dobraną do tego Maca.")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(variants) { variant in
                        row(variant)
                    }
                }
                .padding(12)
            }
        }
    }

    private func row(_ variant: CatalogVariant) -> some View {
        let isSelected = variant.id == selectedId
        return Button {
            selectedId = variant.id
        } label: {
            HStack(spacing: 10) {
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .font(.system(size: 13))
                    .foregroundStyle(isSelected ? Theme.accent : Theme.fgSubtle)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(variant.label)
                            .font(.system(size: 12.5, weight: .semibold, design: .monospaced))
                            .foregroundStyle(Theme.fg)
                        if variant.recommended {
                            Text("polecana dla tego Maca")
                                .font(.system(size: 10.5, weight: .semibold))
                                .foregroundStyle(Theme.accent)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Theme.accentSoft)
                                .clipShape(Capsule())
                        }
                    }
                    if !variant.qualityHint.isEmpty {
                        Text(variant.qualityHint)
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgMuted)
                    }
                }
                Spacer(minLength: 8)
                Text(variant.sizeBytes > 0 ? Format.bytes(variant.sizeBytes) : "—")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .monospacedDigit()
                if let fit = variant.fit {
                    fitBadge(fit, detail: variant.fitDetail)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(isSelected ? Theme.accentSoft.opacity(0.6) : Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(isSelected ? Theme.accent.opacity(0.6) : Theme.border, lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
    }

    private func fitBadge(_ fit: CatalogFit, detail: String) -> some View {
        let tint: Color = switch fit {
        case .fits: Theme.success
        case .tight: Theme.warning
        case .tooBig: Theme.danger
        case .unknown: Theme.fgMuted
        }
        return HStack(spacing: 4) {
            Image(systemName: fit.systemImage).font(.system(size: 10, weight: .semibold))
            Text(fit.label).font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 7)
        .padding(.vertical, 2)
        .background(tint.opacity(0.14))
        .clipShape(Capsule())
        .fixedSize()
        .help(detail.isEmpty ? fit.label : detail)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 10) {
            Text(footnote)
                .font(.system(size: 11.5))
                .foregroundStyle(selected?.fit == .tooBig ? Theme.danger : Theme.fgSubtle)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            SecondaryButton(title: "Anuluj", height: 34) { onClose() }
                .keyboardShortcut(.cancelAction)
            PrimaryButton(title: selected?.fit == .tooBig ? "Zainstaluj mimo to" : "Zainstaluj",
                          systemImage: "arrow.down.circle",
                          isEnabled: !isLoading,
                          fullWidth: false,
                          height: 34) { install() }
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Theme.surface)
    }

    private var footnote: String {
        guard let selected else { return "Pobierze wersję dobraną do pamięci tego Maca." }
        if selected.fit == .tooBig {
            return "Ta wersja raczej się nie zmieści — model może się nie załadować albo działać bardzo wolno."
        }
        return selected.fitDetail.isEmpty
            ? "Pobierze \(selected.label)\(selected.sizeBytes > 0 ? " (\(Format.bytes(selected.sizeBytes)))" : "")."
            : selected.fitDetail
    }

    // MARK: - Actions

    private func load() async {
        do {
            let full = try await store.catalogVariants(for: entry)
            variants = full.variants
            selectedId = full.variants.first(where: \.recommended)?.id
                ?? full.recommendedVariant?.id
                ?? full.variants.first?.id
        } catch {
            loadFailed = true
        }
        isLoading = false
    }

    private func install() {
        let target = entry
        let variant = selected
        onClose()
        Task {
            if let variant, variants.count > 1 {
                await store.install(target, variant: variant)
            } else {
                await store.install(target)
            }
        }
    }
}
