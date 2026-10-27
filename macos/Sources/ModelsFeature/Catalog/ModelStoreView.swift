import SwiftUI
import LocalCodeKit

/// "Odkrywaj" — the in-app model store.
///
/// Installing a model is supposed to feel like installing an app, not like
/// operating a package manager: a card says what the model is *for*, whether it
/// fits in this Mac's memory, and has exactly one button. The repository id,
/// the quantisation and the file list are real and they are all still there —
/// behind "Szczegóły", where they belong for the person who wants them.
///
/// The raw Hugging Face browser is not gone either: "Zaawansowane" opens it,
/// because a store can only ever show a curated slice of 400 000 repositories.
public struct ModelStoreView: View {
    @Environment(ModelsStore.self) private var store

    @State private var installChoice: CatalogEntry?

    public init() {}

    public var body: some View {
        @Bindable var store = store

        VStack(spacing: 0) {
            ScreenHeader(title: "Sklep z modelami", subtitle: subtitle) {
                HStack(spacing: 8) {
                    if store.isLoadingCatalog {
                        ProgressView().controlSize(.small)
                    }
                    Button {
                        store.isShowingRawHub = true
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: "slider.horizontal.3")
                                .font(.system(size: 10, weight: .semibold))
                            Text("Zaawansowane")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .foregroundStyle(Theme.fgMuted)
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Theme.surface2)
                        .clipShape(Capsule())
                        .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
                        .contentShape(Capsule())
                    }
                    .buttonStyle(PressableButtonStyle())
                    .help("Pełne przeglądanie Hugging Face: repozytoria, kwantyzacje, pojedyncze pliki")
                }
            }

            StoreMessages()
            controls
            content
        }
        .background(Theme.bg)
        .sheet(item: $store.catalogDetail) { entry in
            CatalogDetailSheet(entry: entry)
        }
        .sheet(isPresented: $store.isShowingRawHub) {
            advancedSheet
        }
        .sheet(item: $installChoice) { entry in
            InstallVariantSheet(entry: entry) { installChoice = nil }
                .environment(store)
        }
        .task {
            if store.catalog.isEmpty { store.reloadCatalog() }
            store.installStarterModelIfNeeded()
        }
    }

    /// The header line says how much there is to choose from: the store ships
    /// with thousands of models, and a person who reads "modele dobrane do tego
    /// Maca" alone has no way of knowing that.
    private var subtitle: String {
        let plain = "Modele dobrane do tego Maca — jedno kliknięcie, żeby zainstalować"
        // Only for the unfiltered listing: with a search active the number is
        // the size of the result, and "296 modeli dobranych do tego Maca" would
        // be a different sentence than the one the reader expects.
        guard store.catalogSearch.isEmpty, store.catalogCategory == nil,
              !store.catalogInstalledOnly, store.catalogTotal > 0 else { return plain }
        return "\(store.catalogTotal) modeli do wyboru — jedno kliknięcie, żeby zainstalować"
    }

    // MARK: - Controls

    private var controls: some View {
        VStack(spacing: 10) {
            // One row when the window is wide; with Pomocnik open next to the
            // store the filters move under the search field instead of
            // squeezing it to a few letters.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    searchField
                    filterToggles
                    sortMenu
                    qualityMenu
                    Spacer(minLength: 0)
                    pageCount
                }
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 10) {
                        searchField
                        sortMenu
                        qualityMenu
                        Spacer(minLength: 0)
                        pageCount
                    }
                    HStack(spacing: 10) {
                        filterToggles
                        Spacer(minLength: 0)
                    }
                }
                VStack(alignment: .leading, spacing: 8) {
                    searchField
                    HStack(spacing: 10) {
                        sortMenu
                        qualityMenu
                        Spacer(minLength: 0)
                        pageCount
                    }
                    ScrollView(.horizontal, showsIndicators: false) {
                        filterToggles
                    }
                }
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    Chip(title: "Wszystkie",
                         systemImage: "square.grid.2x2",
                         isSelected: store.catalogCategory == nil,
                         onTap: { store.catalogCategory = nil })
                    ForEach(store.catalogCategories) { category in
                        Chip(title: category.label,
                             systemImage: category.systemImage,
                             isSelected: store.catalogCategory == category.id,
                             onTap: { store.catalogCategory = category.id })
                    }
                }
                .padding(.horizontal, 1)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
        .onChange(of: store.catalogSearch) { _, _ in store.reloadCatalog(debounce: true) }
        .onChange(of: store.catalogCategory) { _, _ in store.reloadCatalog() }
        .onChange(of: store.catalogInstalledOnly) { _, _ in store.reloadCatalog() }
        .onChange(of: store.catalogOfficialOnly) { _, _ in store.reloadCatalog() }
        .onChange(of: store.catalogMinTrust) { _, _ in store.reloadCatalog() }
        .onChange(of: store.catalogSort) { _, _ in store.reloadCatalog() }
    }

    private var searchField: some View {
        @Bindable var store = store
        return StyledTextField(placeholder: "Szukaj w sklepie albo wklej link do modelu…",
                               text: $store.catalogSearch,
                               systemImage: "magnifyingglass",
                               disablesAutocorrection: true)
            .frame(minWidth: 240, maxWidth: 360)
    }

    private var filterToggles: some View {
        @Bindable var store = store
        return HStack(spacing: 10) {
            Toggle(isOn: $store.catalogInstalledOnly) {
                Text("Tylko zainstalowane")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .toggleStyle(.checkbox)

            Toggle(isOn: $store.catalogOfficialOnly) {
                Text("Tylko od wydawcy")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .toggleStyle(.checkbox)
            .help("Pokaż wyłącznie repozytoria pracowni, która wytrenowała model "
                  + "— bez konwersji przygotowanych przez inne osoby")

            Toggle(isOn: Binding(get: { store.catalogMinTrust > 0 },
                                 set: { store.catalogMinTrust = $0 ? 60 : 0 })) {
                Text("Tylko sprawdzone")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
            }
            .toggleStyle(.checkbox)
            .help("Ukryj modele o ocenie zaufania poniżej 60/100")
        }
        .fixedSize()
    }

    @ViewBuilder
    private var pageCount: some View {
        if store.catalogTotal > store.catalog.count {
            Text("\(store.catalog.count) z \(store.catalogTotal)")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize()
                .help("Sklep pokazuje pierwszą stronę wyników — zawęź wyszukiwanie, "
                      + "żeby zobaczyć resztę")
        }
    }

    /// "Sortuj": the orderings the server offers, named by the server.
    private var sortMenu: some View {
        @Bindable var store = store

        return Menu {
            ForEach(store.catalogSorts) { option in
                Button {
                    store.catalogSort = option.id
                } label: {
                    Label(option.label, systemImage: option.systemImage)
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: currentSort?.systemImage ?? "arrow.up.arrow.down")
                    .font(.system(size: 10, weight: .semibold))
                Text(currentSort?.label ?? "Sortuj")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(Theme.fgMuted)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Kolejność modeli na liście")
    }

    /// "Jakość": how heavily the next download may be quantised, and which
    /// packaging wins a tie. Four-bit MLX is the default, because that is what
    /// makes the large models fit on this Mac at all.
    private var qualityMenu: some View {
        Menu {
            Section("Jakość pobieranych modeli") {
                ForEach(store.catalogQualities) { option in
                    Button {
                        store.setStoreQuality(option.id)
                    } label: {
                        Label(option.label,
                              systemImage: option.id == store.storeQuality ? "checkmark" : "circle")
                    }
                }
            }
            Section("Format") {
                ForEach(Self.formatOptions, id: \.id) { option in
                    Button {
                        store.setStorePreferredFormat(option.id)
                    } label: {
                        Label(option.label,
                              systemImage: option.id == store.storePreferredFormat
                                  ? "checkmark" : "circle")
                    }
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "dial.medium")
                    .font(.system(size: 10, weight: .semibold))
                Text(qualityLabel)
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundStyle(Theme.fgMuted)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Poziom kwantyzacji i format, w jakim pobierane są modele")
    }

    private static let formatOptions: [CatalogSortOption] = [
        CatalogSortOption(id: "mlx", label: "MLX — szybszy na Apple Silicon"),
        CatalogSortOption(id: "gguf", label: "GGUF — szersza oferta"),
        CatalogSortOption(id: "auto", label: "Bez preferencji"),
    ]

    private var qualityLabel: String {
        let quality = store.storeQuality
        let short = quality == "max" ? "pełna" : quality.replacingOccurrences(of: "bit", with: "-bit")
        let format = store.storePreferredFormat
        return format == "auto" ? short : "\(format.uppercased()) \(short)"
    }

    private var currentSort: CatalogSortOption? {
        store.catalogSorts.first { $0.id == store.catalogSort }
    }

    /// A repository somebody pasted, offered above the results.
    @ViewBuilder
    private var pastedRow: some View {
        if let entry = store.pastedEntry {
            VStack(alignment: .leading, spacing: 8) {
                Text("Z Hugging Face, spoza naszego katalogu")
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(Theme.fgSubtle)
                CatalogCard(entry: entry) { requestInstall(entry) }
                    .frame(maxWidth: 460, alignment: .leading)
            }
            .padding(.bottom, 4)
        }
    }

    // MARK: - Grid

    @ViewBuilder
    private var content: some View {
        if store.catalog.isEmpty, store.pastedEntry != nil {
            ScrollView {
                pastedRow
                    .padding(20)
            }
        } else if store.catalog.isEmpty {
            VStack {
                Spacer(minLength: 0)
                if store.isLoadingCatalog {
                    EmptyState(systemImage: "shippingbox",
                               title: "Ładuję katalog…",
                               message: "Sprawdzam, które modele zmieszczą się na tym Macu.")
                } else {
                    EmptyState(systemImage: "shippingbox",
                               title: "Nic tu nie ma",
                               message: store.catalogSearch.isEmpty
                                   ? "Zmień kategorię albo zajrzyj do „Zaawansowane”, żeby przeszukać całe Hugging Face."
                                   : "Żaden model nie pasuje do „\(store.catalogSearch)”. Spróbuj innej nazwy albo otwórz „Zaawansowane”.",
                               actionTitle: "Zaawansowane",
                               action: { store.isShowingRawHub = true })
                }
                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if store.isCatalogOffline, !store.catalogOfflineNote.isEmpty {
                        Banner(.warning, message: store.catalogOfflineNote)
                    }

                    pastedRow

                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 320, maximum: 460), spacing: 14)],
                              spacing: 14) {
                        ForEach(store.catalog) { entry in
                            CatalogCard(entry: entry) { requestInstall(entry) }
                        }
                    }

                    if !store.catalogSourceNote.isEmpty {
                        HStack(spacing: 5) {
                            Image(systemName: "info.circle")
                                .font(.system(size: 10))
                            Text(store.catalogSourceNote)
                                .font(.system(size: 11.5))
                        }
                        .foregroundStyle(Theme.fgSubtle)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.top, 4)
                    }
                }
                .padding(20)
            }
        }
    }

    /// One button, three outcomes: a gated model sends the user to the token
    /// field, a model that will not fit asks first, everything else installs.
    private func requestInstall(_ entry: CatalogEntry) {
        if entry.requiresToken, !store.settings.hfTokenSet {
            store.isShowingRawHub = false
            store.announceTokenNeeded(for: entry.name)
            ShellBus.shared.request(.show(.settings))
            return
        }
        // The chooser says, per version, whether it fits — so the "may not
        // fit" question is asked there, next to the smaller versions that do.
        installChoice = entry
    }

    // MARK: - Advanced

    private var advancedSheet: some View {
        @Bindable var store = store

        return VStack(spacing: 0) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Zaawansowane: przeglądaj Hugging Face")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Text("Wszystkie repozytoria, kwantyzacje i pojedyncze pliki — bez doboru pod ten komputer.")
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                }
                Spacer(minLength: 8)
                SecondaryButton(title: "Gotowe", height: 32) { store.isShowingRawHub = false }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Theme.surface)
            .overlay(alignment: .bottom) { Hairline() }

            DiscoverView()
        }
        .frame(minWidth: 980, minHeight: 640)
        .background(Theme.bg)
    }
}
