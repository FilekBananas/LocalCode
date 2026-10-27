import SwiftUI
import LocalCodeKit

/// Pomocnik — the help bot, in a panel next to whatever section is open.
///
/// Next to, not instead of: the help bot changes settings and loads models, and
/// the point is to see the change land on the screen it belongs to. The
/// conversation itself is an agent session, so this panel is the Code section's
/// transcript, permission cards and composer around a store in help bot mode
/// (`AgentStore(api:helpbot: true)`), injected by the shell.
public struct HelpbotPanel: View {
    @Environment(AgentStore.self) private var store

    @State private var draft = ""
    @State private var focusRequest = 0

    public init() {}

    public var body: some View {
        @Bindable var store = store
        return VStack(spacing: 0) {
            header
            Hairline()
            content
        }
        .frame(minWidth: 320, idealWidth: 400)
        .background(Theme.bg)
        .banner($store.errorMessage)
        .task { await store.startHelpbot() }
        .task(id: store.helpbot?.state) {
            // While the model downloads, follow it; once it is there, open the
            // conversation without making anyone click again.
            while store.helpbot?.state == .downloading, !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                await store.refreshHelpbotStatus()
            }
            if store.helpbot?.state == .ready, store.session == nil {
                await store.openHelpbot(new: false)
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "sparkles")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Theme.accent)
            VStack(alignment: .leading, spacing: 1) {
                Text("Pomocnik")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.fg)
                Text(subtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.fgMuted)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
            if store.session != nil {
                Button {
                    Task { await store.newSession() }
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("Nowa rozmowa z pomocnikiem")
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 48)
    }

    private var subtitle: String {
        guard let status = store.helpbot else { return "Sprawdzam…" }
        switch status.state {
        case .ready:
            let name = status.displayName ?? status.model
            return status.loaded ? "\(name) · w pamięci" : name
        case .downloading:
            if let fraction = status.download?.fraction {
                return "Jeszcze niegotowy · \(Int((fraction * 100).rounded()))\u{00A0}%"
            }
            return "Jeszcze niegotowy"
        case .missing:
            return "Model pomocnika nie jest pobrany"
        }
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        switch store.helpbot?.state {
        case .none:
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        case .missing?:
            missing
        case .downloading?:
            downloading
        case .ready?:
            if store.session == nil {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                AgentTranscript()
                composer
            }
        }
    }

    private var missing: some View {
        VStack(spacing: 12) {
            Image(systemName: "sparkles")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(Theme.accent)
            Text("Pomocnik ustawi wszystko za Ciebie")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(Theme.fg)
                .multilineTextAlignment(.center)
            Text("Mały model (\(sizeLabel)), który znajduje, pobiera, ustawia i ładuje modele "
                 + "oraz zmienia ustawienia aplikacji — wystarczy napisać, czego potrzebujesz. "
                 + "Każdą zmianę najpierw pokazuje do zatwierdzenia.")
                .font(.system(size: 12))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let error = store.helpbot?.error {
                Text(error)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.danger)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            PrimaryButton(title: store.isInstallingHelpbot ? "Zaczynam…"
                                : (store.helpbot?.error != nil ? "Spróbuj ponownie" : "Pobierz pomocnika"),
                          systemImage: store.helpbot?.error != nil ? "arrow.clockwise" : "arrow.down.circle") {
                Task { await store.installHelpbot() }
            }
            .disabled(store.isInstallingHelpbot)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var downloading: some View {
        VStack(spacing: 12) {
            Text("Pobieram pomocnika")
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Theme.fg)
            if let fraction = store.helpbot?.download?.fraction {
                ProgressView(value: fraction)
                    .tint(Theme.accent)
                    .frame(maxWidth: 240)
                Text("\(Int((fraction * 100).rounded())) % z \(sizeLabel)")
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.fgMuted)
            } else {
                ProgressView().controlSize(.small)
            }
            if let error = store.helpbot?.download?.error {
                Text(error)
                    .font(.system(size: 11.5))
                    .foregroundStyle(Theme.danger)
                    .multilineTextAlignment(.center)
            }
            Text("To jednorazowe. Możesz w tym czasie korzystać z\u{00A0}reszty aplikacji.")
                .font(.system(size: 11.5))
                .foregroundStyle(Theme.fgMuted)
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var composer: some View {
        VStack(spacing: 0) {
            Hairline()
            AgentComposer(text: $draft,
                          isBusy: store.isBusy,
                          isEnabled: store.canSend,
                          hasSession: store.session != nil,
                          queuedCount: store.queuedMessages.count,
                          blockedReason: store.pendingPermission != nil
                              ? "Czeka na Twoją decyzję." : nil,
                          commands: store.slashCommands,
                          idlePlaceholder: "Napisz, czego potrzebujesz…",
                          focusRequest: focusRequest,
                          onSend: { text, _ in store.send(text) },
                          onStop: { store.interrupt() })
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 10)
        }
    }

    private var sizeLabel: String {
        guard let bytes = store.helpbot?.sizeBytes ?? store.helpbot?.download?.totalBytes,
              bytes > 0 else { return "ok. 2,3 GB" }
        return Format.bytes(Int64(bytes))
    }
}
