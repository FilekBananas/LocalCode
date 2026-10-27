import SwiftUI
import LocalCodeKit

/// "Odkrywaj" — the Hugging Face browser.
///
/// Two panes, like LM Studio's model search: the query and its results on the
/// left, one repository in full on the right. With an empty query the left pane
/// shows the curated "Polecane" list grouped by what the model is *for*, which
/// is the only honest way to start when the alternative is 400 000 repositories.
public struct DiscoverView: View {
    @Environment(ModelsStore.self) private var store

    public init() {}

    public var body: some View {
        @Bindable var store = store

        VStack(spacing: 0) {
            ScreenHeader(title: "Odkrywaj",
                         subtitle: "Modele GGUF i MLX z Hugging Face, dopasowane do tego Maca") {
                if store.isSearching {
                    ProgressView().controlSize(.small)
                }
            }

            StoreMessages()

            HStack(spacing: 0) {
                resultsPane
                    .frame(width: 400)
                Hairline(vertical: true)
                HubDetailView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Theme.bg)
        .task { await store.loadCurated() }
    }

    // MARK: - Left pane

    private var resultsPane: some View {
        @Bindable var store = store

        return VStack(spacing: 0) {
            VStack(spacing: 10) {
                StyledTextField(placeholder: "Szukaj modeli na Hugging Face…",
                                text: $store.searchText,
                                systemImage: "magnifyingglass",
                                disablesAutocorrection: true)

                HStack(spacing: 8) {
                    SegmentedControl(items: HubFormatFilter.allCases,
                                     selection: $store.formatFilter) { $0.title }
                    Spacer(minLength: 0)
                    sortMenu
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Theme.surface)
            .overlay(alignment: .bottom) { Hairline() }

            list
        }
        .background(Theme.sidebar)
        .onChange(of: store.searchText) { _, _ in store.scheduleSearch() }
        .onChange(of: store.formatFilter) { _, _ in store.scheduleSearch(debounce: false) }
        .onChange(of: store.sortOrder) { _, _ in store.scheduleSearch(debounce: false) }
    }

    private var sortMenu: some View {
        @Bindable var store = store

        return Menu {
            ForEach(HubSortOrder.allCases) { order in
                Button {
                    store.sortOrder = order
                } label: {
                    if order == store.sortOrder {
                        Label(order.title, systemImage: "checkmark")
                    } else {
                        Text(order.title)
                    }
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: store.sortOrder.systemImage)
                    .font(.system(size: 10, weight: .semibold))
                Text(store.sortOrder.title)
                    .font(.system(size: 12, weight: .medium))
                Image(systemName: "chevron.down")
                    .font(.system(size: 9, weight: .bold))
            }
            .foregroundStyle(Theme.fgMuted)
        }
        .menuIndicator(.hidden)
        .menuStyle(.borderlessButton)
        .fixedSize()
        .tint(Theme.fgMuted)
        .help("Kolejność wyników wyszukiwania")
        .disabled(store.searchText.trimmingCharacters(in: .whitespaces).isEmpty)
    }

    @ViewBuilder
    private var list: some View {
        let isBrowsingCurated = store.searchText.trimmingCharacters(in: .whitespaces).isEmpty

        if isBrowsingCurated {
            if store.curated.isEmpty {
                loadingOrEmpty(title: store.isLoadingCurated
                                   ? "Ładuję polecane modele…"
                                   : "Nie udało się pobrać listy polecanych",
                               message: store.isLoadingCurated
                                   ? "Lista startowa pobierana jest z serwera LocalCode."
                                   : "Sprawdź połączenie z internetem albo wpisz nazwę modelu w polu wyszukiwania.")
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14, pinnedViews: [.sectionHeaders]) {
                        ForEach(store.curatedGroups) { group in
                            Section {
                                VStack(spacing: 4) {
                                    ForEach(group.models) { model in
                                        HubModelRow(model: model)
                                    }
                                }
                                .padding(.horizontal, 8)
                            } header: {
                                groupHeader(group)
                            }
                        }
                    }
                    .padding(.vertical, 10)
                }
            }
        } else if store.searchResults.isEmpty {
            loadingOrEmpty(title: store.isSearching ? "Szukam…" : "Brak wyników",
                           message: store.isSearching
                               ? nil
                               : "Spróbuj innej nazwy albo zmień filtr formatu.")
        } else {
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(store.searchResults) { model in
                        HubModelRow(model: model)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 10)
            }
        }
    }

    private func groupHeader(_ group: CuratedGroup) -> some View {
        HStack(spacing: 7) {
            Image(systemName: group.systemImage)
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(Theme.fgSubtle)
            Text(group.title)
                .font(.system(size: 11, weight: .semibold))
                .textCase(.uppercase)
                .kerning(0.6)
                .foregroundStyle(Theme.fgMuted)
            Spacer(minLength: 0)
            Text("\(group.models.count)")
                .font(.system(size: 11))
                .foregroundStyle(Theme.fgSubtle)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 6)
        .background(Theme.sidebar)
    }

    private func loadingOrEmpty(title: String, message: String?) -> some View {
        VStack {
            Spacer(minLength: 0)
            EmptyState(systemImage: "sparkle.magnifyingglass", title: title, message: message)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Result row

/// One repository in the result list: who made it, what shape it has, and the
/// three numbers that decide whether it is worth opening.
struct HubModelRow: View {
    let model: HubModel

    @Environment(ModelsStore.self) private var store
    @State private var isHovered = false

    private var isActive: Bool { store.selectedRepo == model.id }

    var body: some View {
        Button {
            store.selectRepo(model.id)
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text(model.name)
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    if model.gated {
                        Image(systemName: "lock.fill")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(Theme.warning)
                            .help("Model wymaga zaakceptowania licencji na Hugging Face")
                    }
                    Spacer(minLength: 0)
                    TrustBadge(trust: model.trust, publisher: model.publisher)
                    FormatBadge(format: model.format)
                }

                Text(model.author)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)

                if model.curated, let description = model.description, !description.isEmpty {
                    Text(description)
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgMuted)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }

                HStack(spacing: 10) {
                    if let params = ModelFormat.params(model.paramsB) {
                        stat("cube", params, help: "Liczba parametrów")
                    }
                    if let downloads = ModelFormat.count(model.downloads) {
                        stat("arrow.down.circle", downloads, help: "Pobrania w ostatnim miesiącu")
                    }
                    if let likes = ModelFormat.count(model.likes) {
                        stat("heart", likes, help: "Polubienia")
                    }
                    if let updated = model.updatedAt {
                        stat("clock", Format.relative(updated), help: "Ostatnia zmiana")
                    }
                    if model.curated, let ram = model.minRamGb {
                        stat("memorychip", "\(ram) GB", help: "Zalecana pamięć")
                    }
                    Spacer(minLength: 0)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .rowChrome(isActive: isActive, isHovered: isHovered)
        .onHover { isHovered = $0 }
    }

    private func stat(_ symbol: String, _ text: String, help: String) -> some View {
        HStack(spacing: 3) {
            Image(systemName: symbol).font(.system(size: 9))
            Text(text).font(.system(size: 11))
        }
        .foregroundStyle(Theme.fgSubtle)
        .help(help)
    }
}
