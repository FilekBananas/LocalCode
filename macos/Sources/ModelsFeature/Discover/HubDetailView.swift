import SwiftUI
import LocalCodeKit

/// The right-hand pane of "Odkrywaj": one repository in full.
///
/// The variants table is the point of the screen. A repository publishes a
/// dozen quantisations of the same weights and the only question that matters
/// on a 32 GB Mac is *which of them fits* — so every row carries the size on
/// disk, the memory one instance would need and a coloured verdict, with the
/// recommended variant starred.
struct HubDetailView: View {
    @Environment(ModelsStore.self) private var store

    @State private var isReadmeExpanded = true
    @State private var tokenSheet = false

    private var detail: HubModelDetail? { store.detail }

    var body: some View {
        Group {
            if store.selectedRepo == nil {
                EmptyState(systemImage: "square.stack.3d.up",
                           title: "Wybierz model",
                           message: "Kliknij model po lewej, aby zobaczyć jego opis, warianty kwantyzacji i szacowaną pamięć.")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let detail {
                content(detail)
            } else if store.isLoadingDetail {
                VStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text("Wczytuję informacje o modelu…")
                        .font(.system(size: 12))
                        .foregroundStyle(Theme.fgMuted)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                EmptyState(systemImage: "exclamationmark.triangle",
                           title: "Nie udało się wczytać modelu",
                           message: "Sprawdź połączenie z internetem i spróbuj ponownie.")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Theme.bg)
        .sheet(isPresented: $tokenSheet) { HuggingFaceTokenSheet().environment(store) }
    }

    // MARK: - Content

    private func content(_ detail: HubModelDetail) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header(detail)

                if let message = detail.accessError {
                    Banner(.warning,
                           message: message,
                           title: "Model z ograniczonym dostępem",
                           actionTitle: store.settings.hfTokenSet
                               ? "Zmień token Hugging Face…"
                               : "Wpisz token Hugging Face…",
                           action: { tokenSheet = true })
                }

                variants(detail)

                if let readme = detail.readme, !readme.isEmpty {
                    readmeSection(readme)
                }

                if !detail.files.isEmpty {
                    filesSection(detail.files)
                }
            }
            .padding(20)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func header(_ detail: HubModelDetail) -> some View {
        let model = detail.model
        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(model.name)
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .textSelection(.enabled)
                if model.gated {
                    Image(systemName: "lock.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(Theme.warning)
                }
                Spacer(minLength: 0)
                Button {
                    if let url = URL(string: "https://huggingface.co/\(model.id)") {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Hugging Face").font(.system(size: 12, weight: .medium))
                        Image(systemName: "arrow.up.right").font(.system(size: 9, weight: .semibold))
                    }
                    .foregroundStyle(Theme.fgMuted)
                }
                .buttonStyle(PressableButtonStyle())
                .help("Otwórz stronę modelu w przeglądarce")
            }

            Text(model.id)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(Theme.fgSubtle)
                .textSelection(.enabled)

            if let description = model.description, !description.isEmpty {
                Text(description)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            FlowLayout(spacing: 6, lineSpacing: 6) {
                FormatBadge(format: model.format)
                if let params = ModelFormat.params(model.paramsB) {
                    MetaBadge(text: params, systemImage: "cube")
                }
                if let active = ModelFormat.params(model.activeParamsB) {
                    MetaBadge(text: "aktywne \(active)", systemImage: "bolt")
                }
                if let context = ModelFormat.context(detail.contextLength ?? model.contextLength) {
                    MetaBadge(text: "kontekst \(context)", systemImage: "text.alignleft")
                }
                if let arch = detail.arch ?? model.arch, !arch.isEmpty {
                    MetaBadge(text: arch, systemImage: "cpu")
                }
                if let downloads = ModelFormat.count(model.downloads) {
                    MetaBadge(text: downloads, systemImage: "arrow.down.circle")
                }
                if let likes = ModelFormat.count(model.likes) {
                    MetaBadge(text: likes, systemImage: "heart")
                }
                if let updated = model.updatedAt {
                    MetaBadge(text: Format.relative(updated), systemImage: "clock")
                }
                if detail.isEmbedding {
                    MetaBadge(text: "embeddingi",
                              systemImage: "point.3.connected.trianglepath.dotted",
                              tint: Theme.accent)
                }
            }
        }
    }

    // MARK: - Variants

    private func variants(_ detail: HubModelDetail) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader("Warianty", subtitle: variantsSubtitle(detail))

            if detail.variants.isEmpty {
                Text("To repozytorium nie zawiera plików, które LocalCode potrafi uruchomić.")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .padding(.vertical, 8)
            } else {
                VStack(spacing: 0) {
                    variantHeaderRow
                    Hairline()
                    ForEach(Array(detail.variants.enumerated()), id: \.element.id) { index, variant in
                        if index > 0 { Hairline() }
                        VariantRow(repo: detail.model.id, variant: variant)
                    }
                }
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                        .strokeBorder(Theme.border, lineWidth: 1)
                )
            }
        }
    }

    private func variantsSubtitle(_ detail: HubModelDetail) -> String {
        let count = detail.variants.count
        let word = Format.plural(count, "wariant", "warianty", "wariantów")
        return "\(word) · szacowana pamięć dotyczy jednej instancji z domyślnym kontekstem"
    }

    private var variantHeaderRow: some View {
        HStack(spacing: 10) {
            Text("Kwantyzacja")
                .frame(width: VariantRow.quantWidth, alignment: .leading)
            Text("Rozmiar")
                .frame(width: VariantRow.sizeWidth, alignment: .trailing)
            Text("Pamięć")
                .frame(width: VariantRow.memoryWidth, alignment: .trailing)
            Text("Dopasowanie")
                .frame(width: VariantRow.fitWidth, alignment: .leading)
            Spacer(minLength: 0)
        }
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(Theme.fgSubtle)
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Theme.surface2)
    }

    // MARK: - README and files

    private func readmeSection(_ readme: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.16)) { isReadmeExpanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isReadmeExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(Theme.fgSubtle)
                    Text("Opis (README)")
                        .font(.system(size: 12, weight: .semibold))
                        .textCase(.uppercase)
                        .kerning(0.6)
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(PressableButtonStyle())

            if isReadmeExpanded {
                MarkdownView(readme, baseSize: 13.5, spacing: 10)
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                            .strokeBorder(Theme.border, lineWidth: 1)
                    )
            }
        }
    }

    private func filesSection(_ files: [HubFile]) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(files) { file in
                    HStack(spacing: 8) {
                        Text(file.name)
                            .font(.system(size: 11.5, design: .monospaced))
                            .foregroundStyle(Theme.fgMuted)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                        if let size = file.size {
                            Text(Format.bytes(size))
                                .font(.system(size: 11))
                                .foregroundStyle(Theme.fgSubtle)
                        }
                    }
                }
            }
            .padding(.top, 8)
        } label: {
            Text("Pliki w repozytorium (\(files.count))")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Theme.fgMuted)
        }
        .padding(14)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }
}

// MARK: - One variant

/// A single quantisation: what it costs on disk, what it costs in RAM, and the
/// button that fetches it.
struct VariantRow: View {
    let repo: String
    let variant: HubVariant

    @Environment(ModelsStore.self) private var store
    @State private var isHovered = false

    static let quantWidth: CGFloat = 170
    static let sizeWidth: CGFloat = 80
    static let memoryWidth: CGFloat = 90
    static let fitWidth: CGFloat = 120

    /// The download job (if any) that is fetching exactly this variant.
    private var job: DownloadJob? {
        store.downloads.first {
            $0.repo == repo && $0.files.contains(where: variant.files.contains) && $0.state.isActive
        }
    }

    /// True when this variant is already on disk.
    private var isInstalled: Bool {
        store.library.contains { model in
            guard model.repo == repo else { return false }
            guard let quant = variant.quant else { return false }
            return model.quant?.caseInsensitiveCompare(quant) == .orderedSame
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            HStack(spacing: 5) {
                if variant.recommended {
                    Image(systemName: "star.fill")
                        .font(.system(size: 9))
                        .foregroundStyle(Theme.accent)
                        .help("Zalecany wariant")
                }
                Text(variant.label)
                    .font(.system(size: 12.5, weight: variant.recommended ? .semibold : .regular))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                if variant.isVisionProjector {
                    MetaBadge(text: "projektor", systemImage: "eye")
                }
            }
            .frame(width: Self.quantWidth, alignment: .leading)

            Text(variant.sizeBytes > 0 ? Format.bytes(variant.sizeBytes) : "—")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .frame(width: Self.sizeWidth, alignment: .trailing)

            Text(variant.estimate.map { Format.bytes($0.totalBytes) } ?? "—")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .frame(width: Self.memoryWidth, alignment: .trailing)

            Group {
                if let estimate = variant.estimate {
                    FitBadge(fit: estimate.fits)
                        .help(estimate.message)
                } else {
                    Text("—").font(.system(size: 12)).foregroundStyle(Theme.fgSubtle)
                }
            }
            .frame(width: Self.fitWidth, alignment: .leading)

            Spacer(minLength: 8)

            action
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(isHovered ? Theme.surface2 : Color.clear)
        .onHover { isHovered = $0 }
    }

    @ViewBuilder
    private var action: some View {
        if let job {
            HStack(spacing: 8) {
                ProgressBar(value: job.progress, height: 4)
                    .frame(width: 70)
                Text("\(Int((job.progress * 100).rounded())) %")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.fgMuted)
                    .monospacedDigit()
                IconButton(systemName: "xmark.circle.fill", label: "Anuluj pobieranie", size: 22, fontSize: 12) {
                    Task { await store.cancelDownload(job.id) }
                }
            }
        } else if isInstalled {
            HStack(spacing: 5) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 11, weight: .semibold))
                Text("Pobrany").font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(Theme.success)
        } else {
            SecondaryButton(title: "Pobierz", systemImage: "arrow.down.circle", height: 28) {
                Task { await store.download(repo: repo, variantId: variant.id) }
            }
        }
    }
}

// MARK: - Hugging Face token

/// Editor for the Hugging Face token, reachable from wherever a gated model
/// blocks the user. The token itself is never read back from the server — the
/// API only reports whether one is stored.
struct HuggingFaceTokenSheet: View {
    @Environment(ModelsStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var token = ""
    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Token Hugging Face")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(Theme.fg)

            Text("""
                 Token jest potrzebny do modeli z ograniczonym dostępem — takich, \
                 na które trzeba najpierw zaakceptować licencję na huggingface.co. \
                 Utwórz token typu „read” w ustawieniach swojego konta i wklej go poniżej.
                 """)
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)

            LabeledField("Token", description: store.settings.hfTokenSet
                         ? "Token jest już zapisany. Wpisanie nowego go zastąpi."
                         : "Token nie jest jeszcze zapisany.") {
                StyledTextField(placeholder: "hf_…", text: $token, systemImage: "key",
                                disablesAutocorrection: true)
            }

            HStack(spacing: 10) {
                Button {
                    if let url = URL(string: "https://huggingface.co/settings/tokens") {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Text("Otwórz stronę tokenów").font(.system(size: 12, weight: .medium))
                        Image(systemName: "arrow.up.right").font(.system(size: 9, weight: .semibold))
                    }
                    .foregroundStyle(Theme.accent)
                }
                .buttonStyle(PressableButtonStyle())

                Spacer(minLength: 0)

                if store.settings.hfTokenSet {
                    DangerButton(title: "Usuń token", filled: false, height: 34) {
                        save(nil)
                    }
                }
                SecondaryButton(title: "Anuluj", height: 34) { dismiss() }
                PrimaryButton(title: "Zapisz", isLoading: isSaving, fullWidth: false, height: 34) {
                    save(token)
                }
            }
        }
        .padding(20)
        .frame(width: 460)
        .background(Theme.bg)
    }

    private func save(_ value: String?) {
        isSaving = true
        Task {
            await store.setHuggingFaceToken(value)
            isSaving = false
            dismiss()
        }
    }
}
