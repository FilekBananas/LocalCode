import SwiftUI
import LocalCodeKit

/// The window toolbar's download button.
///
/// It disappears completely when nothing has been downloaded this session —
/// a toolbar item that is always there but usually says "0" is noise. While
/// jobs run it is a progress ring; the popover behind it is the queue.
public struct DownloadsIndicator: View {
    /// Optional on purpose. This view is put in the window toolbar, and toolbar
    /// content is hosted in the title bar rather than inside the view it is
    /// attached to: it only sees environment values applied *outside* the
    /// `.toolbar` modifier. Reading the store non-optionally would turn that
    /// ordering mistake into a launch crash, so a missing store simply means
    /// "no indicator".
    @Environment(ModelsStore.self) private var store: ModelsStore?

    @State private var isPresented = false

    public init() {}

    public var body: some View {
        if let store, !store.visibleJobs.isEmpty {
            Button {
                isPresented.toggle()
            } label: {
                ring(store)
            }
            .buttonStyle(PressableButtonStyle())
            .help(helpText(store))
            .accessibilityLabel(Text("Pobierane modele"))
            .popover(isPresented: $isPresented, arrowEdge: .bottom) {
                // Explicit: the popover is a separate hosting context too.
                DownloadsPopover().environment(store)
            }
        }
    }

    private func helpText(_ store: ModelsStore) -> String {
        let active = store.activeJobs.count
        guard active > 0 else { return "Ostatnie pobierania" }
        let percent = store.activeProgress.map { " — \(Int(($0 * 100).rounded()))%" } ?? ""
        return "\(Format.plural(active, "pobieranie", "pobierania", "pobierań")) w toku\(percent)"
    }

    private func ring(_ store: ModelsStore) -> some View {
        ZStack {
            Circle()
                .stroke(Theme.border, lineWidth: 2)
            if let progress = store.activeProgress {
                Circle()
                    .trim(from: 0, to: max(0.02, progress))
                    .stroke(Theme.accent, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.25), value: progress)
                // The number people want here is how far along it is; the job
                // count only matters when there is more than one.
                Text(store.activeJobs.count > 1
                     ? "\(store.activeJobs.count)"
                     : "\(Int((progress * 100).rounded()))")
                    .font(.system(size: 8.5, weight: .bold))
                    .foregroundStyle(Theme.fg)
                    .monospacedDigit()
            } else {
                Image(systemName: "checkmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(Theme.success)
            }
        }
        .frame(width: 18, height: 18)
        .padding(3)
        .contentShape(Rectangle())
    }
}

/// The queue itself.
struct DownloadsPopover: View {
    @Environment(ModelsStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Pobierane")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Spacer(minLength: 0)
                Text(Format.plural(store.visibleJobs.count, "zadanie", "zadania", "zadań"))
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgSubtle)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            Hairline()

            ScrollView {
                VStack(spacing: 0) {
                    ForEach(Array(store.visibleJobs.enumerated()), id: \.element.id) { index, job in
                        if index > 0 { Hairline() }
                        DownloadJobRow(job: job)
                    }
                }
            }
            // A popover is sized by its content, so the list has to name its own
            // height: roughly one row each, capped so a long queue scrolls.
            .frame(height: min(340, store.visibleJobs.reduce(CGFloat(0)) {
                $0 + ($1.state.isActive ? 92 : 74)
            }))
        }
        .frame(width: 380)
        .background(Theme.surface)
    }
}

/// One job: what it is, how fast it goes, and the one action that applies.
struct DownloadJobRow: View {
    let job: DownloadJob

    @Environment(ModelsStore.self) private var store

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(job.label)
                        .font(.system(size: 12.5, weight: .semibold))
                        .foregroundStyle(Theme.fg)
                        .lineLimit(1)
                    Text(job.repo)
                        .font(.system(size: 10.5))
                        .foregroundStyle(Theme.fgSubtle)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer(minLength: 0)
                FormatBadge(format: job.format)
                action
            }

            if job.state.isActive {
                ProgressBar(value: job.totalBytes > 0 ? job.progress : nil, height: 5)
                HStack(spacing: 8) {
                    Text(statusText)
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgMuted)
                        .monospacedDigit()
                    Spacer(minLength: 0)
                    if !job.isVerifying, job.speedBps > 1 {
                        Text(ModelFormat.speed(job.speedBps))
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                            .monospacedDigit()
                    }
                    if let eta = ModelFormat.eta(job.etaSeconds) {
                        Text(eta)
                            .font(.system(size: 11))
                            .foregroundStyle(Theme.fgSubtle)
                    }
                }
            } else if let error = job.error, !error.isEmpty {
                Text(error)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.danger)
                    .fixedSize(horizontal: false, vertical: true)
            } else if job.state == .done {
                HStack(spacing: 8) {
                    Text("\(Format.bytes(job.totalBytes)) · gotowe")
                        .font(.system(size: 11))
                        .foregroundStyle(Theme.fgMuted)
                    Spacer(minLength: 0)
                    Button("Pokaż w Moich modelach") {
                        ShellBus.shared.request(.show(.myModels))
                    }
                    .buttonStyle(.link)
                    .font(.system(size: 11.5))
                }
            } else {
                Text(job.state.label)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgMuted)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    /// The line under the bar: what stage the job is in, in bytes when it has
    /// them. A queued job has no total yet — "Zero KB" would be nonsense there.
    private var statusText: String {
        if job.isCerberChecking { return "Cerber sprawdza model przed pobraniem (zwykle kilka minut)…" }
        if job.isVerifying { return "Sprawdzanie sumy kontrolnej…" }
        guard job.totalBytes > 0 else {
            return job.state == .queued ? "W kolejce…" : "Przygotowywanie…"
        }
        return "\(Format.bytes(job.doneBytes)) z \(Format.bytes(job.totalBytes))"
    }

    @ViewBuilder
    private var action: some View {
        if job.state.isActive {
            IconButton(systemName: "xmark.circle",
                       label: "Anuluj pobieranie",
                       size: 24,
                       fontSize: 13) {
                Task { await store.cancelDownload(job.id) }
            }
        } else {
            IconButton(systemName: "xmark",
                       label: "Usuń z listy",
                       size: 24,
                       fontSize: 11) {
                store.dismissJob(job.id)
            }
        }
    }
}
