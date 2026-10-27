import SwiftUI
import AppKit
import LocalCodeKit

/// The bar above the transcript: where the agent works, with what model, under
/// which permission mode, and how much context is left.
struct AgentHeaderBar: View {
    @Environment(AgentStore.self) private var store

    @State private var isConfirmingClear = false

    var body: some View {
        HStack(spacing: 10) {
            location
            Spacer(minLength: 8)
            if let goal = store.goal, goal.isActive || goal.state == "blocked" {
                goalChip(goal)
            }
            modelMenu
            modeMenu
            EffortPicker(rawLevels: store.session?.effortLevels ?? [],
                         selection: store.session?.effort,
                         showsLabel: false,
                         offersUltracode: true) { level in
                store.setEffort(level)
            }
            if !store.backgroundHelpers.isEmpty {
                backgroundChip(store.backgroundHelpers.count)
            }
            ContextGauge(stats: store.context)
            actions
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(Theme.surface)
        .overlay(alignment: .bottom) { Hairline() }
        .alert("Wyczyścić rozmowę?", isPresented: $isConfirmingClear) {
            Button("Anuluj", role: .cancel) {}
            Button("Wyczyść", role: .destructive) { store.clearTranscript() }
        } message: {
            Text("Agent zapomni całą historię tej sesji i zacznie od pustego kontekstu. Pliki na dysku pozostaną bez zmian.")
        }
    }

    // MARK: - Background helpers

    /// "Pomocnicy w tle: N" while helpers started with `wait=false` work: the
    /// agent goes on meanwhile, and their reports reach it as they finish.
    private func backgroundChip(_ count: Int) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "person.2.fill")
                .font(.system(size: 10, weight: .semibold))
            Text("Pomocnicy w tle: \(count)")
                .font(.system(size: 11.5, weight: .medium))
                .monospacedDigit()
        }
        .foregroundStyle(Theme.accent)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Theme.accent.opacity(0.12))
        .clipShape(Capsule())
        .help("Agent pracuje dalej; raport każdego pomocnika dostanie, gdy ten skończy.")
    }

    // MARK: - Goal

    /// `/goal` at a glance: what the agent is working towards and how many
    /// times it has been sent back to it. The x clears it (`/goal wyczyść`).
    private func goalChip(_ goal: AgentGoal) -> some View {
        HStack(spacing: 5) {
            Image(systemName: goal.state == "blocked" ? "questionmark.circle" : "scope")
                .font(.system(size: 10, weight: .semibold))
            Text(goal.text)
                .font(.system(size: 11.5, weight: .medium))
                .lineLimit(1)
                .truncationMode(.tail)
            if goal.turns > 0, goal.maxTurns > 0 {
                Text("\(goal.turns)/\(goal.maxTurns)")
                    .font(.system(size: 10.5))
                    .monospacedDigit()
                    .foregroundStyle(Theme.fgSubtle)
            }
            Button {
                store.clearGoal()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 8, weight: .bold))
            }
            .buttonStyle(.plain)
            .help("Usuń cel (/goal wyczyść)")
        }
        .foregroundStyle(goal.state == "blocked" ? Theme.warning : Theme.accent)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background((goal.state == "blocked" ? Theme.warning : Theme.accent).opacity(0.12))
        .clipShape(Capsule())
        .frame(maxWidth: 260)
        .help(goal.state == "blocked" ? goal.noticeText
              : "Cel: \(goal.text) — agent pracuje, dopóki go nie osiągnie (Stop przerywa).")
    }

    // MARK: - Where

    private var location: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(spacing: 6) {
                Image(systemName: "folder.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
                Text(store.session?.shortCwd ?? store.selectedProject?.shortPath ?? "Brak projektu")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                    .lineLimit(1)
                    .truncationMode(.head)
            }
            Text(store.session?.displayTitle ?? "Brak otwartej sesji")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgSubtle)
                .lineLimit(1)
        }
        .help(store.session?.cwd ?? "")
        .onTapGesture {
            guard let path = store.session?.cwd ?? store.selectedProject?.path else { return }
            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: path)
        }
    }

    // MARK: - Model

    private var modelMenu: some View {
        Menu {
            // "Ulubione" first, local and remote together, each listed only there.
            let favorites = FavoriteModels.shared
            let starred = store.models.filter { favorites.contains($0.key) }
            let others = store.models.filter { !favorites.contains($0.key) }
            let local = others.filter { !RemoteModelKey.isRemote($0.key) }
            let remote = Dictionary(grouping: others.filter { RemoteModelKey.isRemote($0.key) }) {
                RemoteModelKey.provider(of: $0.key) ?? ""
            }
            if store.models.isEmpty {
                Text("Brak modeli lokalnych")
            }
            if !starred.isEmpty {
                Section("Ulubione") {
                    ForEach(starred) { model in
                        modelButton(model, grouped: false)
                    }
                }
                if !others.isEmpty { Divider() }
            }
            ForEach(local) { model in
                modelButton(model, grouped: false)
            }
            if !remote.isEmpty {
                Divider()
                ForEach(remote.keys.sorted(), id: \.self) { provider in
                    let models = (remote[provider] ?? []).sorted {
                        $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
                    }
                    Menu("\(RemoteModelKey.label(for: provider)) (\(models.count))") {
                        ForEach(models) { model in
                            modelButton(model, grouped: true)
                        }
                    }
                }
            }
            Divider()
            if let current = store.currentModelKey {
                // A menu item cannot carry a star of its own, so the model in
                // use is starred from here.
                Button {
                    favorites.toggle(current)
                } label: {
                    Label(favorites.actionTitle(for: current),
                          systemImage: favorites.contains(current) ? "star.slash" : "star")
                }
            }
            Button("Odśwież listę modeli") { Task { await store.refresh() } }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: "cpu")
                    .font(.system(size: 10, weight: .semibold))
                Text(store.currentModelLabel)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundStyle(Theme.fg)
            .frame(maxWidth: 220)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(Theme.surface2)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
        .help("Model używany w tej sesji")
    }

    private func modelButton(_ model: AgentModelOption, grouped: Bool) -> some View {
        Button {
            store.setModel(model.key)
        } label: {
            if model.key == store.currentModelKey {
                Label(modelTitle(model, grouped: grouped), systemImage: "checkmark")
            } else if FavoriteModels.shared.contains(model.key) {
                Label(modelTitle(model, grouped: grouped), systemImage: "star.fill")
            } else {
                Text(modelTitle(model, grouped: grouped))
            }
        }
    }

    private func modelTitle(_ model: AgentModelOption, grouped: Bool) -> String {
        var parts = [grouped ? model.displayName : RemoteModelKey.name(model.displayName, key: model.key)]
        if let context = model.contextLength, context > 0 { parts.append("\(Format.tokens(context)) kontekstu") }
        if model.seesImages { parts.append("widzi obrazy") }
        if model.loaded { parts.append("załadowany") }
        return parts.joined(separator: " · ")
    }

    // MARK: - Mode

    private var modeMenu: some View {
        Menu {
            ForEach(AgentMode.allCases) { mode in
                Button {
                    store.setMode(mode)
                } label: {
                    if mode == store.activeMode {
                        Label("\(mode.label) — \(mode.hint)", systemImage: "checkmark")
                    } else {
                        Text("\(mode.label) — \(mode.hint)")
                    }
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: store.activeMode.systemImage)
                    .font(.system(size: 10, weight: .semibold))
                Text(store.activeMode.label)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                Image(systemName: "chevron.down")
                    .font(.system(size: 8, weight: .bold))
            }
            .foregroundStyle(modeTint)
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(modeTint.opacity(0.12))
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(modeTint.opacity(0.35), lineWidth: 1))
        .help("Tryb uprawnień: \(store.activeMode.hint)")
    }

    private var modeTint: Color {
        switch store.activeMode {
        case .ask: Theme.accent
        case .acceptEdits: Theme.warning
        case .auto: Color.teal
        case .yolo: Theme.danger
        case .plan: Theme.success
        }
    }

    // MARK: - Actions

    private var actions: some View {
        HStack(spacing: 6) {
            headerButton(title: "Kompaktuj",
                         systemImage: "arrow.down.right.and.arrow.up.left",
                         help: "Podsumuj starsze tury i zwolnij kontekst") {
                store.compact()
            }
            headerButton(title: "Wyczyść",
                         systemImage: "eraser",
                         help: "Zacznij od pustego kontekstu") {
                isConfirmingClear = true
            }
        }
    }

    private func headerButton(title: String,
                              systemImage: String,
                              help: String,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: systemImage)
                    .font(.system(size: 10, weight: .semibold))
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
            }
            .foregroundStyle(store.session == nil ? Theme.fgSubtle : Theme.fgMuted)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Theme.surface2)
            .clipShape(Capsule())
            .overlay(Capsule().strokeBorder(Theme.border, lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(PressableButtonStyle())
        .disabled(store.session == nil)
        .help(help)
    }
}
