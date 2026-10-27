import SwiftUI
import LocalCodeKit

/// Global long-term memory: the master switch plus the list of remembered
/// entries. Project memory is edited in the project's own settings.
struct MemorySection: View {
    let draft: SettingsDraft
    let api: APIClient

    private static let kinds: [MemoryKind] = [.fact, .preference, .note]

    @State private var items: [MemoryEntry] = []
    @State private var isLoading = true
    @State private var loadError: String?

    @State private var newContent = ""
    @State private var newKind: MemoryKind = .fact
    @State private var isCreating = false

    @State private var editingId: String?
    @State private var editingText = ""
    @State private var busyId: String?
    @State private var actionError: String?
    @State private var confirmRemoval: ConfirmRequest?

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsToggleRow("Pamięć włączona",
                              description: "Po wyłączeniu żadne wspomnienia — globalne ani projektowe — nie trafiają do promptu.",
                              isOn: draft.binding(\.memoryEnabled))

            Hairline()

            SectionHeader("Pamięć globalna",
                          subtitle: "Te wpisy trafiają do każdej rozmowy z dostępem do kontekstu globalnego: do rozmów bez projektu i do projektów z pamięcią wspólną.")

            if !draft.value.memoryEnabled {
                Banner(.info, message: "Pamięć jest wyłączona — wpisy są zapisane, ale model ich nie dostaje.")
            }

            if let actionError {
                Banner(.error, message: actionError, onDismiss: { self.actionError = nil })
            }

            addForm
            list
        }
        .task { await load() }
        .confirmDialog($confirmRemoval)
    }

    // MARK: - Add form

    private var addForm: some View {
        VStack(alignment: .leading, spacing: 10) {
            StyledTextEditor(placeholder: "Co model ma zapamiętać na stałe?",
                             text: $newContent,
                             minHeight: 68)

            HStack(spacing: 10) {
                Picker("Rodzaj wpisu", selection: $newKind) {
                    ForEach(Self.kinds, id: \.self) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: 170, alignment: .leading)

                Spacer(minLength: 4)

                PrimaryButton(title: "Dodaj",
                              systemImage: "plus",
                              isLoading: isCreating,
                              isEnabled: !newContent.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                              fullWidth: false,
                              height: 34,
                              action: create)
            }

            Text(hint(for: newKind))
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgSubtle)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .background(Theme.surface2, in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }

    // MARK: - List

    @ViewBuilder
    private var list: some View {
        if isLoading {
            HStack {
                Spacer()
                ProgressView().controlSize(.small)
                Text("Wczytywanie pamięci…")
                    .font(.system(size: 12.5))
                    .foregroundStyle(Theme.fgMuted)
                Spacer()
            }
            .padding(.vertical, 20)
        } else if let loadError {
            Banner(.error, message: loadError, actionTitle: "Spróbuj ponownie", action: {
                Task { await load() }
            })
        } else if items.isEmpty {
            EmptyState(systemImage: "brain.head.profile",
                       title: "Pamięć jest pusta",
                       message: "Zapisuj tu rzeczy, które model ma pamiętać między rozmowami: fakty o Tobie i Twojej pracy, preferencje co do stylu odpowiedzi albo notatki, do których trzeba wracać.")
        } else {
            VStack(spacing: 8) {
                ForEach(items) { entry in
                    row(entry)
                }
            }
        }
    }

    private func row(_ entry: MemoryEntry) -> some View {
        let isEditing = editingId == entry.id
        let isBusy = busyId == entry.id

        return HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon(for: entry.kind))
                .font(.system(size: 13))
                .foregroundStyle(Theme.fgSubtle)
                .padding(.top, 2)

            VStack(alignment: .leading, spacing: 6) {
                if isEditing {
                    StyledTextEditor(placeholder: "Treść wpisu", text: $editingText, minHeight: 80)
                } else {
                    Text(entry.content)
                        .font(.system(size: 13))
                        .foregroundStyle(Theme.fg)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }

                HStack(spacing: 8) {
                    Chip(title: entry.kind.title,
                         systemImage: entry.pinned ? "pin.fill" : nil,
                         isSelected: entry.pinned)
                    Text(Format.relative(entry.updatedAt))
                        .font(.system(size: 11.5))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                    if entry.source == "auto" {
                        Text("· zapisane automatycznie")
                            .font(.system(size: 11.5))
                            .foregroundStyle(Theme.fgSubtle)
                            .lineLimit(1)
                    }
                }
            }

            Spacer(minLength: 4)

            HStack(spacing: 2) {
                if isEditing {
                    IconButton(systemName: "checkmark", label: "Zapisz zmiany",
                               tint: Theme.success, isEnabled: !isBusy) {
                        saveEdit(entry)
                    }
                    IconButton(systemName: "xmark", label: "Anuluj edycję", isEnabled: !isBusy) {
                        editingId = nil
                    }
                } else {
                    IconButton(systemName: entry.pinned ? "pin.slash" : "pin",
                               label: entry.pinned ? "Odepnij" : "Przypnij na górze",
                               isActive: entry.pinned, isEnabled: !isBusy) {
                        togglePin(entry)
                    }
                    IconButton(systemName: "pencil", label: "Edytuj wpis", isEnabled: !isBusy) {
                        editingId = entry.id
                        editingText = entry.content
                    }
                    IconButton(systemName: "trash", label: "Usuń wpis",
                               tint: Theme.danger, isEnabled: !isBusy) {
                        confirmRemoval = ConfirmRequest(title: "Usunąć wpis z pamięci?",
                                                        message: entry.content,
                                                        confirmTitle: "Usuń") {
                            remove(entry)
                        }
                    }
                }
            }
            .opacity(isBusy ? 0.5 : 1)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(entry.pinned ? Theme.accentSoft : Theme.surface,
                    in: RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.cornerRadius, style: .continuous)
                .strokeBorder(entry.pinned ? Theme.accent.opacity(0.4) : Theme.border, lineWidth: 1)
        )
    }

    // MARK: - Metadata

    private func icon(for kind: MemoryKind) -> String {
        switch kind {
        case .fact: "lightbulb"
        case .preference: "sparkles"
        case .note: "note.text"
        }
    }

    private func hint(for kind: MemoryKind) -> String {
        switch kind {
        case .fact: "Trwała informacja, np. „Pracuję w firmie X”."
        case .preference: "Sposób pracy, np. „Odpowiadaj zwięźle, bez wstępów”."
        case .note: "Dowolna notatka do zapamiętania na później."
        }
    }

    // MARK: - Data

    private func sorted(_ list: [MemoryEntry]) -> [MemoryEntry] {
        list.sorted { lhs, rhs in
            lhs.pinned == rhs.pinned ? lhs.updatedAt > rhs.updatedAt : lhs.pinned
        }
    }

    private func load() async {
        isLoading = true
        do {
            items = sorted(try await api.memories(scope: .global, folderId: nil))
            loadError = nil
        } catch {
            loadError = APIError.describe(error)
        }
        isLoading = false
    }

    private func create() {
        let content = newContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty, !isCreating else { return }
        isCreating = true
        let kind = newKind
        Task {
            do {
                let created = try await api.createMemory(
                    MemoryInput(scope: .global, folderId: nil, kind: kind, content: content)
                )
                items = sorted(items + [created])
                newContent = ""
                actionError = nil
            } catch {
                actionError = APIError.describe(error)
            }
            isCreating = false
        }
    }

    private func saveEdit(_ entry: MemoryEntry) {
        let content = editingText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }
        guard content != entry.content else {
            editingId = nil
            return
        }
        busyId = entry.id
        Task {
            do {
                let updated = try await api.updateMemory(id: entry.id, MemoryInput(content: content))
                replace(updated)
                editingId = nil
                actionError = nil
            } catch {
                actionError = APIError.describe(error)
            }
            busyId = nil
        }
    }

    private func togglePin(_ entry: MemoryEntry) {
        busyId = entry.id
        Task {
            do {
                let updated = try await api.updateMemory(id: entry.id, MemoryInput(pinned: !entry.pinned))
                replace(updated)
                actionError = nil
            } catch {
                actionError = APIError.describe(error)
            }
            busyId = nil
        }
    }

    private func remove(_ entry: MemoryEntry) {
        busyId = entry.id
        Task {
            do {
                try await api.deleteMemory(id: entry.id)
                items.removeAll { $0.id == entry.id }
                if editingId == entry.id { editingId = nil }
                actionError = nil
            } catch {
                actionError = APIError.describe(error)
            }
            busyId = nil
        }
    }

    private func replace(_ entry: MemoryEntry) {
        if let index = items.firstIndex(where: { $0.id == entry.id }) {
            items[index] = entry
        } else {
            items.append(entry)
        }
        items = sorted(items)
    }
}
