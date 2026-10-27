import SwiftUI
import LocalCodeKit

/// The agent's plan (`todo_write`): what it intends to do and where it is.
struct TodoCardView: View {
    let todos: [TodoItem]

    private var doneCount: Int { todos.filter { $0.status == .done }.count }

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Image(systemName: "checklist")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Theme.accent)
                Text("Plan")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text("\(doneCount)/\(todos.count)")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Theme.fgSubtle)
                    .monospacedDigit()
                Spacer(minLength: 0)
            }

            ForEach(todos) { item in
                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: symbol(for: item.status))
                        .font(.system(size: 12))
                        .foregroundStyle(tint(for: item.status))
                        .frame(width: 14)
                    Text(item.content)
                        .font(.system(size: 12.5, weight: item.status == .inProgress ? .medium : .regular))
                        .foregroundStyle(item.status == .done ? Theme.fgSubtle : Theme.fg)
                        .strikethrough(item.status == .done, color: Theme.fgSubtle)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(11)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous)
                .strokeBorder(Theme.border, lineWidth: 1)
        )
    }

    private func symbol(for status: TodoItem.Status) -> String {
        switch status {
        case .pending: "square"
        case .inProgress: "square.and.pencil"
        case .done: "checkmark.square.fill"
        }
    }

    private func tint(for status: TodoItem.Status) -> Color {
        switch status {
        case .pending: Theme.fgSubtle
        case .inProgress: Theme.accent
        case .done: Theme.success
        }
    }
}
