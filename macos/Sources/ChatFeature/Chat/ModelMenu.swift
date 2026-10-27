import SwiftUI
import LocalCodeKit

/// The model picker, in the two shapes the chat needs: a capsule in the
/// composer and a plain title-bar button.
///
/// "Ulubione" come first, local and remote together; then models on this Mac,
/// flat; models behind an API key follow in one submenu per provider
/// ("DeepInfra (87)") — a single provider can list a hundred models, and they
/// cost money, so they never crowd the local ones.
struct ModelMenu: View {
    enum Style { case capsule, bar }

    let models: [ModelInfo]
    let providers: [ProviderInfo]
    let currentProvider: String
    let currentModel: String
    var style: Style = .capsule
    let onSelect: (String, String) -> Void

    private struct ProviderGroup: Identifiable {
        let id: String
        let label: String
        let models: [ModelInfo]
    }

    private var favorites: FavoriteModels { FavoriteModels.shared }

    /// In the order the server listed them; each one appears only here.
    private var favoriteModels: [ModelInfo] {
        models.filter { favorites.contains($0.id) }
    }

    private var localModels: [ModelInfo] {
        models.filter { !RemoteModelKey.isRemote($0.id) && !favorites.contains($0.id) }
    }

    private var remoteGroups: [ProviderGroup] {
        let remote = models.filter { RemoteModelKey.isRemote($0.id) && !favorites.contains($0.id) }
        let byProvider = Dictionary(grouping: remote) { RemoteModelKey.provider(of: $0.id) ?? "" }
        return byProvider.keys.sorted().map { provider in
            ProviderGroup(id: provider,
                          label: RemoteModelKey.label(for: provider),
                          models: byProvider[provider]?.sorted {
                              $0.label.localizedCaseInsensitiveCompare($1.label) == .orderedAscending
                          } ?? [])
        }
    }

    private var currentLabel: String {
        if let match = models.first(where: { $0.id == currentModel && $0.provider == currentProvider })
            ?? models.first(where: { $0.id == currentModel }) {
            return RemoteModelKey.name(match.label, key: match.id)
        }
        return currentModel.isEmpty ? "Wybierz model" : currentModel
    }

    var body: some View {
        Menu {
            if models.isEmpty {
                Text("Brak modeli — pobierz model w sekcji „Odkrywaj”.")
            } else {
                if !favoriteModels.isEmpty {
                    Section("Ulubione") {
                        ForEach(favoriteModels) { model in
                            item(model)
                        }
                    }
                    if !localModels.isEmpty || !remoteGroups.isEmpty { Divider() }
                }
                ForEach(localModels) { model in
                    item(model)
                }
                if !remoteGroups.isEmpty {
                    if !localModels.isEmpty { Divider() }
                    ForEach(remoteGroups) { group in
                        Menu("\(group.label) (\(group.models.count))") {
                            ForEach(group.models) { model in
                                item(model, grouped: true)
                            }
                        }
                    }
                }
                if !currentModel.isEmpty {
                    // A menu item cannot carry a star of its own, so the
                    // model in use is starred from here.
                    Divider()
                    let isFavorite = favorites.contains(currentModel)
                    Button {
                        favorites.toggle(currentModel)
                    } label: {
                        Label(favorites.actionTitle(for: currentModel),
                              systemImage: isFavorite ? "star.slash" : "star")
                    }
                }
            }
        } label: {
            label
        }
        .menuIndicator(.hidden)
        .fixedSize()
        .menuStyle(.borderlessButton)
        // A borderless menu paints its label with the tint; without this the
        // picker shouts in accent orange next to muted chrome.
        .tint(Theme.fgMuted)
        .help("Model używany w tej rozmowie")
        .accessibilityLabel(Text("Model: \(currentLabel)"))
    }

    private func item(_ model: ModelInfo, grouped: Bool = false) -> some View {
        Button {
            onSelect(model.provider, model.id)
        } label: {
            if model.id == currentModel {
                Label(menuTitle(model, grouped: grouped), systemImage: "checkmark")
            } else if favorites.contains(model.id) {
                Label(menuTitle(model, grouped: grouped), systemImage: "star.fill")
            } else {
                Text(menuTitle(model, grouped: grouped))
            }
        }
    }

    /// Small models live and die by their context window, so it is part of the
    /// name in the menu rather than hidden in a tooltip. Inside a provider's
    /// submenu the "(DeepInfra)" suffix would only repeat the submenu's title.
    private func menuTitle(_ model: ModelInfo, grouped: Bool) -> String {
        let name = grouped ? RemoteModelKey.bareName(model.label, key: model.id)
                           : RemoteModelKey.name(model.label, key: model.id)
        guard let context = model.contextWindow, context > 0 else { return name }
        return "\(name)  ·  \(Format.tokens(context))"
    }

    @ViewBuilder
    private var label: some View {
        switch style {
        case .capsule:
            HStack(spacing: 4) {
                Image(systemName: "cpu")
                    .font(.system(size: 10, weight: .semibold))
                Text(currentLabel)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 8, weight: .semibold))
            }
            .foregroundStyle(Theme.fgMuted)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Theme.surface2)
            .clipShape(Capsule())
            .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
            .contentShape(Capsule())

        case .bar:
            HStack(spacing: 5) {
                Image(systemName: "cpu")
                    .font(.system(size: 11, weight: .semibold))
                Text(currentLabel)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .semibold))
            }
            .foregroundStyle(Theme.fgMuted)
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
    }
}

/// Compact "↑ in ↓ out" readout. Cost is left out entirely when it is zero,
/// which for local models is always.
enum TokenReadout {
    static func text(for conversation: Conversation?) -> String? {
        guard let conversation else { return nil }
        let total = conversation.tokensIn + conversation.tokensOut
        guard total > 0 else { return nil }
        var parts = ["↑ \(Format.compact(conversation.tokensIn))",
                     "↓ \(Format.compact(conversation.tokensOut))"]
        if conversation.costUsd > 0 { parts.append(Format.cost(conversation.costUsd)) }
        return parts.joined(separator: "  ")
    }
}
