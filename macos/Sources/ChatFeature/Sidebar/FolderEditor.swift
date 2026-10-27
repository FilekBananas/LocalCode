import SwiftUI
import UniformTypeIdentifiers
import LocalCodeKit

/// Creates or edits a project: identity, colour, memory mode, instructions —
/// plus its memory and files once the project exists.
///
/// The memory mode is the decision this dialog exists for: it is what tells the
/// server whether a conversation in this project may also see the global
/// prompt, profile, memory and knowledge base, or only the project's own. With
/// a small context window that choice is not cosmetic — it decides how much of
/// the window is left for the actual conversation.
struct FolderEditor: View {
    private enum Tab: String, Hashable, CaseIterable {
        case settings, memory, knowledge

        var title: String {
            switch self {
            case .settings: "Ustawienia"
            case .memory: "Pamięć projektu"
            case .knowledge: "Pliki projektu"
            }
        }
    }

    let folder: Folder?

    @Environment(ChatStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var tab: Tab = .settings
    @State private var name: String
    @State private var summary: String
    @State private var color: String
    @State private var systemPrompt: String
    @State private var extraInfo: String
    @State private var memoryMode: MemoryMode
    @State private var isSaving = false
    @State private var errorText: String?
    @State private var deleteChats = false
    @State private var isConfirmingDelete = false

    private var isEditing: Bool { folder != nil }

    init(folder: Folder? = nil) {
        self.folder = folder
        _name = State(initialValue: folder?.name ?? "")
        _summary = State(initialValue: folder?.description ?? "")
        _color = State(initialValue: folder?.color ?? ProjectColorPalette.default)
        _systemPrompt = State(initialValue: folder?.systemPrompt ?? "")
        _extraInfo = State(initialValue: folder?.extraInfo ?? "")
        _memoryMode = State(initialValue: folder?.memoryMode ?? .shared)
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            header
            Hairline()

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    if isEditing {
                        Picker("Sekcja", selection: $tab) {
                            ForEach(Tab.allCases, id: \.self) { item in
                                Text(item.title).tag(item)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                    }

                    switch tab {
                    case .settings:
                        settingsTab
                    case .memory:
                        if let folder {
                            FolderMemorySection(folderId: folder.id, api: store.api)
                        }
                    case .knowledge:
                        if let folder {
                            FolderFilesSection(folderId: folder.id, api: store.api)
                        }
                    }
                }
                .padding(18)
            }
            .banner($errorText)

            Hairline()
            footer
        }
        .background(Theme.bg)
        .frame(minWidth: 560, idealWidth: 640, minHeight: 520, idealHeight: 700)
        .alert("Usunąć projekt?", isPresented: $isConfirmingDelete) {
            Button("Anuluj", role: .cancel) {}
            Button("Usuń", role: .destructive, action: deleteFolder)
        } message: {
            Text(deleteChats
                 ? "Projekt, jego pamięć, pliki oraz wszystkie rozmowy w nim zostaną trwale usunięte."
                 : "Projekt, jego pamięć i pliki zostaną trwale usunięte. Rozmowy zostaną przeniesione poza projekt.")
        }
    }

    // MARK: - Chrome

    private var header: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(isEditing ? "Edytuj projekt" : "Nowy projekt")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("Projekt zbiera rozmowy, instrukcje, pamięć i pliki w jednym miejscu.")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            IconButton(systemName: "xmark", label: "Zamknij") { dismiss() }
        }
        .padding(16)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Spacer(minLength: 0)
            SecondaryButton(title: "Anuluj", height: 38) { dismiss() }
            PrimaryButton(title: "Zapisz",
                          isLoading: isSaving,
                          isEnabled: !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                          fullWidth: false,
                          height: 38,
                          action: save)
        }
        .padding(16)
    }

    // MARK: - Settings tab

    @ViewBuilder
    private var settingsTab: some View {
        LabeledField("Nazwa projektu", isRequired: true) {
            StyledTextField(placeholder: "np. Praca dyplomowa", text: $name)
        }

        LabeledField("Opis") {
            StyledTextField(placeholder: "Krótko, o czym jest ten projekt", text: $summary)
        }

        LabeledField("Kolor", description: "Wyróżnia projekt na liście rozmów.") {
            colorPicker
        }

        memorySection

        LabeledField("Instrukcje dla tego projektu",
                     description: "Prompt systemowy doklejany do każdej rozmowy w projekcie. Przy małym oknie kontekstu trzymaj go krótko.") {
            StyledTextEditor(placeholder: "Jesteś asystentem prowadzącym ten projekt. Trzymaj się ustalonej terminologii i odpowiadaj po polsku.",
                             text: $systemPrompt,
                             minHeight: 110)
        }

        LabeledField("Dodatkowe informacje i dane",
                     description: "Stałe fakty, słowniczek, ograniczenia — wszystko, co model ma mieć pod ręką.") {
            StyledTextEditor(placeholder: "np. nazwy produktów, skróty firmowe, styl dokumentacji",
                             text: $extraInfo,
                             minHeight: 110)
        }

        if isEditing {
            dangerZone
        }
    }

    private var colorPicker: some View {
        HStack(spacing: 10) {
            ForEach(ProjectColorPalette.hexes, id: \.self) { hex in
                let isSelected = hex.caseInsensitiveCompare(color) == .orderedSame
                Button {
                    color = hex
                } label: {
                    Circle()
                        .fill(Format.hexColor(hex) ?? Theme.accent)
                        .frame(width: 26, height: 26)
                        .overlay {
                            if isSelected {
                                Image(systemName: "checkmark")
                                    .font(.system(size: 11, weight: .bold))
                                    .foregroundStyle(.white)
                            }
                        }
                        .overlay(
                            Circle()
                                .strokeBorder(Theme.accent, lineWidth: isSelected ? 2 : 0)
                                .padding(-3)
                        )
                        .contentShape(Circle())
                }
                .buttonStyle(PressableButtonStyle())
                .accessibilityLabel(Text("Kolor \(hex)"))
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
    }

    // MARK: Memory mode

    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Tryb pamięci")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Theme.fg)
            Text("Decyduje, czy rozmowy w tym projekcie mają dostęp do Twojego kontekstu globalnego.")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .fixedSize(horizontal: false, vertical: true)

            HStack(alignment: .top, spacing: 12) {
                ForEach(MemoryMode.allCases, id: \.self) { mode in
                    MemoryModeCard(mode: mode, isSelected: memoryMode == mode) { memoryMode = mode }
                }
            }
            .padding(.top, 2)
        }
    }

    // MARK: Danger zone

    private var dangerZone: some View {
        Card(background: Theme.surface) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Usuń projekt")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("Instrukcje, pamięć i pliki tego projektu znikną bezpowrotnie.")
                    .font(.system(size: 12))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)

                Toggle(isOn: $deleteChats) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Usuń także rozmowy w tym projekcie")
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.fg)
                        Text(deleteChats
                             ? "Zostanie usuniętych \(folder?.conversationCount ?? 0) rozmów."
                             : "Rozmowy zostaną przeniesione poza projekt.")
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                }
                .toggleStyle(.switch)
                .tint(Theme.danger)

                DangerButton(title: "Usuń projekt", systemImage: "trash", height: 38) {
                    isConfirmingDelete = true
                }
            }
        }
    }

    // MARK: - Actions

    private func save() {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else {
            errorText = "Podaj nazwę projektu."
            return
        }
        // Empty strings clear a field: `FolderInput`'s nil values are dropped by
        // the encoder, which the server reads as "leave unchanged".
        let input = FolderInput(name: trimmedName,
                                description: summary.trimmingCharacters(in: .whitespacesAndNewlines),
                                color: color,
                                systemPrompt: systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines),
                                extraInfo: extraInfo.trimmingCharacters(in: .whitespacesAndNewlines),
                                memoryMode: memoryMode)

        isSaving = true
        errorText = nil
        Task {
            let saved: Folder?
            if let folder {
                saved = await store.updateFolder(folder.id, input)
            } else {
                saved = await store.createFolder(input)
            }
            isSaving = false
            if saved == nil {
                errorText = store.errorMessage ?? "Nie udało się zapisać projektu."
            } else {
                dismiss()
            }
        }
    }

    private func deleteFolder() {
        guard let folder else { return }
        let removeChats = deleteChats
        Task {
            await store.deleteFolder(folder.id, deleteChats: removeChats)
            dismiss()
        }
    }
}

// MARK: - Memory mode card

/// One of the two big, clickable cards that pick the project's memory mode.
struct MemoryModeCard: View {
    let mode: MemoryMode
    let isSelected: Bool
    let action: () -> Void

    private var systemImage: String {
        mode == .shared ? "arrow.triangle.branch" : "lock.fill"
    }

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: systemImage)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(isSelected ? Theme.accent : Theme.fgMuted)
                    Text(mode.title)
                        .font(.system(size: 13.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                    Spacer(minLength: 0)
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 15))
                            .foregroundStyle(Theme.accent)
                    }
                }
                Text(mode.explanation)
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(isSelected ? Theme.accentSoft : Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                    .strokeBorder(isSelected ? Theme.accent : Theme.border,
                                  lineWidth: isSelected ? 2 : 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(PressableButtonStyle())
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

// MARK: - Project memory

/// Long-term memory scoped to one project.
struct FolderMemorySection: View {
    let folderId: String
    let api: APIClient

    @State private var entries: [MemoryEntry] = []
    @State private var isLoading = true
    @State private var isSaving = false
    @State private var draft = ""
    @State private var kind: MemoryKind = .fact
    @State private var errorText: String?

    private let kinds: [MemoryKind] = [.fact, .preference, .note]

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Pamięć widoczna tylko dla rozmów w tym projekcie.")
                .font(.system(size: 12.5))
                .foregroundStyle(Theme.fgMuted)

            if let errorText {
                Banner(.error, message: errorText)
            }

            Card {
                VStack(alignment: .leading, spacing: 10) {
                    Picker("Rodzaj", selection: $kind) {
                        ForEach(kinds, id: \.self) { item in
                            Text(item.title).tag(item)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    StyledTextEditor(placeholder: "np. Piszę pracę o sieciach neuronowych, termin obrony to czerwiec.",
                                     text: $draft,
                                     minHeight: 70)

                    HStack {
                        Spacer(minLength: 0)
                        PrimaryButton(title: "Dodaj do pamięci",
                                      systemImage: "plus",
                                      isLoading: isSaving,
                                      isEnabled: !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                                      fullWidth: false,
                                      height: 36,
                                      action: add)
                    }
                }
            }

            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
            } else if entries.isEmpty {
                EmptyState(systemImage: "brain",
                           title: "Pamięć projektu jest pusta",
                           message: "Dodaj fakty, preferencje i notatki, które model ma pamiętać w każdej rozmowie tego projektu.")
            } else {
                VStack(spacing: 8) {
                    ForEach(entries) { entry in
                        row(entry)
                    }
                }
            }
        }
        .task { await load() }
    }

    private func row(_ entry: MemoryEntry) -> some View {
        Card(padding: 12) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Chip(title: entry.kind.title)
                    if entry.pinned {
                        Image(systemName: "pin.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(Theme.accent)
                    }
                    Spacer(minLength: 0)
                    Text(Format.relative(entry.updatedAt))
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgSubtle)
                    IconButton(systemName: entry.pinned ? "pin.slash" : "pin",
                               label: entry.pinned ? "Odepnij" : "Przypnij",
                               size: 26,
                               fontSize: 12) { togglePin(entry) }
                    IconButton(systemName: "trash",
                               label: "Usuń",
                               tint: Theme.danger,
                               size: 26,
                               fontSize: 12) { remove(entry) }
                }
                Text(entry.content)
                    .font(.system(size: 13))
                    .foregroundStyle(Theme.fg)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Actions

    private func load() async {
        isLoading = true
        do {
            entries = try await api.memories(scope: .folder, folderId: folderId)
            errorText = nil
        } catch {
            errorText = APIError.describe(error)
        }
        isLoading = false
    }

    private func add() {
        let content = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }
        isSaving = true
        Task {
            do {
                let entry = try await api.createMemory(
                    MemoryInput(scope: .folder, folderId: folderId, kind: kind, content: content)
                )
                entries.insert(entry, at: 0)
                draft = ""
                errorText = nil
            } catch {
                errorText = APIError.describe(error)
            }
            isSaving = false
        }
    }

    private func togglePin(_ entry: MemoryEntry) {
        Task {
            do {
                let updated = try await api.updateMemory(id: entry.id, MemoryInput(pinned: !entry.pinned))
                if let index = entries.firstIndex(where: { $0.id == entry.id }) {
                    entries[index] = updated
                }
            } catch {
                errorText = APIError.describe(error)
            }
        }
    }

    private func remove(_ entry: MemoryEntry) {
        Task {
            do {
                try await api.deleteMemory(id: entry.id)
                entries.removeAll { $0.id == entry.id }
            } catch {
                errorText = APIError.describe(error)
            }
        }
    }
}

// MARK: - Project files

/// Knowledge-base documents scoped to one project.
struct FolderFilesSection: View {
    let folderId: String
    let api: APIClient

    @State private var documents: [KbDocument] = []
    @State private var isLoading = true
    @State private var isUploading = false
    @State private var isImporting = false
    @State private var errorText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                Text("Pliki przeszukiwane tylko w rozmowach tego projektu.")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
                SecondaryButton(title: "Dodaj pliki",
                                systemImage: "plus",
                                isLoading: isUploading,
                                height: 34) { isImporting = true }
            }

            if let errorText {
                Banner(.error, message: errorText)
            }

            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 18)
            } else if documents.isEmpty {
                EmptyState(systemImage: "doc.text.magnifyingglass",
                           title: "Brak plików projektu",
                           message: "Dodaj dokumenty, aby model mógł z nich korzystać w rozmowach tego projektu.")
            } else {
                VStack(spacing: 8) {
                    ForEach(documents) { document in
                        row(document)
                    }
                }
            }
        }
        .task { await load() }
        .fileImporter(isPresented: $isImporting,
                      allowedContentTypes: [.item],
                      allowsMultipleSelection: true) { result in
            switch result {
            case let .success(urls):
                upload(urls)
            case let .failure(error):
                errorText = error.localizedDescription
            }
        }
    }

    private func row(_ document: KbDocument) -> some View {
        Card(padding: 12) {
            HStack(spacing: 10) {
                Image(systemName: "doc.text")
                    .font(.system(size: 15))
                    .foregroundStyle(Theme.fgMuted)
                VStack(alignment: .leading, spacing: 2) {
                    Text(document.name)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    Text(detail(document))
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
                IconButton(systemName: "trash",
                           label: "Usuń plik",
                           tint: Theme.danger,
                           size: 26,
                           fontSize: 12) { remove(document) }
            }
        }
    }

    private func detail(_ document: KbDocument) -> String {
        var parts = [Format.bytes(document.size), statusLabel(document.status)]
        if document.chunkCount > 0 { parts.append("\(document.chunkCount) fragm.") }
        if let error = document.error, !error.isEmpty { parts.append(error) }
        return parts.joined(separator: " · ")
    }

    private func statusLabel(_ status: String) -> String {
        switch status {
        case "ready": "Gotowy"
        case "processing": "Przetwarzanie…"
        case "error": "Błąd"
        default: status
        }
    }

    // MARK: Actions

    private func load() async {
        isLoading = true
        do {
            documents = try await api.knowledge(scope: .folder, folderId: folderId)
            errorText = nil
        } catch {
            errorText = APIError.describe(error)
        }
        isLoading = false
    }

    private func upload(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        isUploading = true
        Task {
            do {
                let files = try urls.map { try MultipartFile.read(contentsOf: $0) }
                let created = try await api.uploadKnowledge(files: files, scope: .folder, folderId: folderId)
                documents.insert(contentsOf: created, at: 0)
                errorText = nil
            } catch {
                errorText = APIError.describe(error)
            }
            isUploading = false
        }
    }

    private func remove(_ document: KbDocument) {
        Task {
            do {
                try await api.deleteKnowledge(id: document.id)
                documents.removeAll { $0.id == document.id }
            } catch {
                errorText = APIError.describe(error)
            }
        }
    }
}
