import SwiftUI

/// "Ulubione": the models the user starred, listed first in every picker.
///
/// The list lives on the server (`favoriteModels` in `GET /api/app-settings`)
/// so the terminal app sees the same stars. It is kept here, in the kit, rather
/// than in one feature's store because the chat menu, the loader bar, the
/// agent's menu and "Moje modele" all live in different modules and must agree.
///
/// Keys are `LocalModel.key`s and `remote:provider:id` keys alike; a key whose
/// model is not installed right now is kept — it does no harm in a picker that
/// never shows it.
@MainActor
@Observable
public final class FavoriteModels {
    public static let shared = FavoriteModels()

    /// In the order they were starred.
    public private(set) var keys: [String] = []

    @ObservationIgnored private var api: APIClient?

    public init() {}

    /// The shell hands over the shared client once the server is up.
    public func attach(_ api: APIClient) {
        self.api = api
    }

    public func contains(_ key: String) -> Bool {
        keys.contains(key)
    }

    /// `items` with the favourites moved to the front; both groups keep the
    /// order the caller gave them (loaded first, by name — whatever it was).
    public func ordered<T>(_ items: [T], key: (T) -> String) -> [T] {
        guard !keys.isEmpty else { return items }
        let starred = Set(keys)
        return items.filter { starred.contains(key($0)) } + items.filter { !starred.contains(key($0)) }
    }

    /// Takes the list from an app-settings answer some other store received.
    public func update(from keys: [String]?) {
        guard let keys, keys != self.keys else { return }
        self.keys = keys
    }

    public func refresh() async {
        guard let api else { return }
        if let keys = try? await api.favoriteModels() { update(from: keys) }
    }

    public func toggle(_ key: String) {
        set(key, favorite: !contains(key))
    }

    /// Stars or unstars one model. The star changes at once; the server's
    /// answer then replaces the whole list (it may carry a star the terminal
    /// added meanwhile), and a failure puts the old list back.
    public func set(_ key: String, favorite: Bool) {
        let before = keys
        if favorite {
            if !keys.contains(key) { keys.append(key) }
        } else {
            keys.removeAll { $0 == key }
        }
        guard let api else { return }
        Task {
            do {
                keys = try await api.setFavoriteModel(key, favorite: favorite)
            } catch {
                keys = before
            }
        }
    }

    /// "Dodaj do ulubionych" / "Usuń z ulubionych" for a menu item or tooltip.
    public func actionTitle(for key: String) -> String {
        contains(key) ? "Usuń z ulubionych" : "Dodaj do ulubionych"
    }
}

/// The part of `GET /api/app-settings` this kit needs; absent on an older server.
private struct FavoriteModelsDTO: Decodable, Sendable {
    let favoriteModels: [String]?
}

private struct FavoriteModelPatch: Encodable, Sendable {
    let key: String
    let favorite: Bool
}

extension APIClient {
    /// `GET /api/app-settings` → `favoriteModels` (empty on a server without them).
    public func favoriteModels() async throws -> [String] {
        try await send(makeRequest("api/app-settings"), as: FavoriteModelsDTO.self).favoriteModels ?? []
    }

    /// `PUT /api/app-settings/favorite-models {key, favorite}` → the new list.
    public func setFavoriteModel(_ key: String, favorite: Bool) async throws -> [String] {
        let data = try encodeBody(FavoriteModelPatch(key: key, favorite: favorite))
        let request = makeRequest("api/app-settings/favorite-models", method: "PUT", body: data)
        return try await send(request, as: FavoriteModelsDTO.self).favoriteModels ?? []
    }
}

/// The star next to a model: filled and golden when it is a favourite.
public struct FavoriteStarButton: View {
    public let key: String
    public var size: CGFloat
    public var fontSize: CGFloat

    @State private var favorites = FavoriteModels.shared

    public init(key: String, size: CGFloat = 24, fontSize: CGFloat = 12) {
        self.key = key
        self.size = size
        self.fontSize = fontSize
    }

    public var body: some View {
        let isFavorite = favorites.contains(key)
        Button {
            favorites.toggle(key)
        } label: {
            Image(systemName: isFavorite ? "star.fill" : "star")
                .font(.system(size: fontSize, weight: .medium))
                .foregroundStyle(isFavorite ? Theme.warning : Theme.fgSubtle)
                .frame(width: size, height: size)
                .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle(pressedOpacity: 0.5))
        .help(favorites.actionTitle(for: key))
        .accessibilityLabel(Text(favorites.actionTitle(for: key)))
    }
}
