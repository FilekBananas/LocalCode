import AppKit
import SwiftUI
import LocalCodeKit
import ModelsFeature
import CodeFeature
import SpeechFeature
import VideoFeature

/// Renders the app's screens to PNG files, without a window and without a server.
///
/// It exists because the environment this app is built in cannot click through
/// it: macOS refuses UI scripting to the terminal here (`osascript` gets
/// −25211) and screen recording is off, so "does the layout hold" had no answer
/// beyond "it compiles". `ImageRenderer` draws a SwiftUI view into a bitmap
/// in-process, which needs no permission at all.
///
///     swift run ScreenShots /katalog/wyjściowy
///
/// Each screen is rendered at the window's real width with the stores filled by
/// their `preview(api:)` factories (debug-only), so what lands in the PNG is the
/// same view hierarchy the app shows.
#if DEBUG

@MainActor
func render<V: View>(_ view: V, width: CGFloat, height: CGFloat, to url: URL) throws {
    // A real window, never shown. `ImageRenderer` cannot lay out a `ScrollView`
    // and draws every AppKit-backed control (TextEditor, Picker, Stepper) as a
    // yellow placeholder, because there is no window for them to live in. An
    // off-screen `NSWindow` gives them one: layout runs for real, the scroll
    // view has a document view, and `cacheDisplay` copies what the view itself
    // draws — no screen recording permission involved, nothing appears on the
    // display.
    let hosting = NSHostingView(rootView:
        view
            .frame(width: width, height: height)
            .environment(\.colorScheme, .dark)
    )
    hosting.frame = NSRect(x: 0, y: 0, width: width, height: height)

    let window = NSWindow(contentRect: hosting.frame,
                          styleMask: [.borderless],
                          backing: .buffered,
                          defer: false)
    window.contentView = hosting
    window.isReleasedWhenClosed = false
    // Off-screen: far outside any display, and never ordered front.
    window.setFrameOrigin(NSPoint(x: -10_000, y: -10_000))
    window.displayIfNeeded()

    hosting.layoutSubtreeIfNeeded()
    // One pass through the run loop so SwiftUI commits its layout and the
    // scroll views size their document views.
    RunLoop.main.run(until: Date().addingTimeInterval(0.35))
    hosting.layoutSubtreeIfNeeded()
    hosting.displayIfNeeded()

    guard let bitmap = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else {
        throw RenderError.failed(url.lastPathComponent)
    }
    bitmap.size = hosting.bounds.size
    hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw RenderError.failed(url.lastPathComponent)
    }
    try png.write(to: url)
    window.close()
    FileHandle.standardOutput.write(Data("zapisano \(url.path) (\(png.count / 1024) kB)\n".utf8))
}

enum RenderError: Error, CustomStringConvertible {
    case failed(String)
    var description: String {
        switch self {
        case .failed(let name): "nie udało się wyrenderować \(name)"
        }
    }
}

@MainActor
func main() throws {
    // AppKit needs an application object before any view is instantiated;
    // `.accessory` keeps it out of the Dock and off the screen.
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)

    let arguments = CommandLine.arguments
    let directory = URL(fileURLWithPath: arguments.count > 1 ? arguments[1] : ".")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    // No server is running; every call fails and the preview state stands in.
    let api = APIClient()
    let models = ModelsStore(api: api)

    // The screens' chrome (header, mode switch) and, separately, their contents:
    // ImageRenderer cannot lay out a ScrollView, so the scrollable part is
    // rendered through the *ScreenPreview views (see SpeechScreenPreview).
    let speech = SpeechStore.preview(api: api)
    try render(SpeechView().environment(speech).environment(models),
               width: 1040, height: 220, to: directory.appending(path: "mowa-naglowek.png"))
    try render(SpeechScreenPreview().environment(speech).environment(models),
               width: 1040, height: 940, to: directory.appending(path: "mowa-dyktowanie.png"))
    try render(SpeechScreenPreview(speaking: true).environment(speech).environment(models),
               width: 1040, height: 940, to: directory.appending(path: "mowa-lektor.png"))

    let video = VideoStore.preview(api: api)
    try render(VideoScreenPreview().environment(video).environment(models),
               width: 1040, height: 1280, to: directory.appending(path: "wideo.png"))

    let generating = VideoStore.preview(api: api, generating: true, withClip: false)
    try render(VideoScreenPreview().environment(generating).environment(models),
               width: 1040, height: 940, to: directory.appending(path: "wideo-w-trakcie.png"))

    // Pomocnik next to "Odkrywaj", mid-conversation, at the default window size.
    let helpbot = AgentStore.helpbotPreview(api: api, turns: 6)
    try render(HelpbotShellPreview().environment(models).environment(helpbot),
               width: 1280, height: 812, to: directory.appending(path: "pomocnik-obok-sklepu.png"))

    try render(CerberPanel().padding(24).environment(models).frame(maxHeight: .infinity, alignment: .top)
                   .background(Theme.bg),
               width: 860, height: 640, to: directory.appending(path: "cerber-panel.png"))
    try render(AgentComposerPreview(seesImages: true), width: 820, height: 240,
               to: directory.appending(path: "kod-obrazki.png"))
    try render(AgentComposerPreview(seesImages: false), width: 820, height: 260,
               to: directory.appending(path: "kod-obrazki-bez-wizji.png"))
}

/// The shell's layout around Pomocnik (AppShellView): a rail stand-in,
/// "Odkrywaj", and the help bot as a plain trailing column. Not `.inspector` —
/// next to one, the section lost its header and search field.
struct HelpbotShellPreview: View {
    @Environment(AgentStore.self) private var helpbot

    var body: some View {
        HStack(spacing: 0) {
            Color(white: 0.12).frame(width: 78)
            ModelStoreView().frame(maxWidth: .infinity, maxHeight: .infinity)
            Hairline(vertical: true)
            HelpbotPanel().environment(helpbot).frame(width: 380)
        }
    }
}

MainActor.assumeIsolated {
    do {
        try main()
    } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
}

#else

// The preview state this tool renders (`VideoStore.preview(api:)` and friends)
// exists only in debug builds, so in release there is nothing to draw.
FileHandle.standardError.write(Data(
    "ScreenShots renderuje ekrany tylko w kompilacji debug: swift run ScreenShots <katalog>\n".utf8))
exit(2)

#endif
