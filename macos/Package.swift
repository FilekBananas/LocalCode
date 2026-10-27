// swift-tools-version: 6.0
//
// LocalCode — the macOS app.
//
// Four modules, no Xcode project:
//
//   LocalCodeKit   design system, shared components, markdown, networking core
//                  and the chat DTOs. Everything other modules need is `public`.
//   ChatFeature    the chatbot UX (ported from the ai_ux Apple client).
//   ModelsFeature  Odkrywaj (sklep z modelami) / Moje modele / Runtime.
//   ImagesFeature  "Obrazy": local text-to-image and the gallery.
//   SpeechFeature  "Mowa": dictation (speech-to-text) and reading aloud (text-to-speech).
//   VideoFeature   "Wideo": a few seconds of footage from a sentence (docs/VIDEO.md).
//   CodeFeature    the coding agent UI (owned by another engineer).
//   LocalCode      the executable: app entry, navigation rail, server process.
//
// Built with `swift build` (Swift 6.3 toolchain, macOS 26 SDK). The language
// mode stays at Swift 5 on purpose: the UI is `@MainActor` SwiftUI code ported
// from an app written before strict concurrency, and Swift 6 mode would turn a
// large amount of harmless main-actor hopping into build errors for no gain.

import PackageDescription

let package = Package(
    name: "LocalCode",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "LocalCode", targets: ["LocalCode"]),
        .executable(name: "ScreenShots", targets: ["ScreenShots"]),
        .library(name: "LocalCodeKit", targets: ["LocalCodeKit"]),
        .library(name: "ChatFeature", targets: ["ChatFeature"]),
        .library(name: "ModelsFeature", targets: ["ModelsFeature"]),
        .library(name: "CodeFeature", targets: ["CodeFeature"]),
        .library(name: "ImagesFeature", targets: ["ImagesFeature"]),
        .library(name: "SpeechFeature", targets: ["SpeechFeature"]),
        .library(name: "VideoFeature", targets: ["VideoFeature"]),
    ],
    targets: [
        .target(name: "LocalCodeKit"),
        .target(name: "ChatFeature", dependencies: ["LocalCodeKit"]),
        .target(name: "ModelsFeature", dependencies: ["LocalCodeKit"]),
        .target(name: "CodeFeature", dependencies: ["LocalCodeKit"]),
        .target(name: "ImagesFeature", dependencies: ["LocalCodeKit", "ModelsFeature"]),
        .target(name: "SpeechFeature", dependencies: ["LocalCodeKit", "ModelsFeature"]),
        .target(name: "VideoFeature", dependencies: ["LocalCodeKit", "ModelsFeature"]),
        // A tool, not part of the app: renders screens to PNG so a layout can be
        // looked at where nobody can click (see Sources/ScreenShots/main.swift).
        .executableTarget(name: "ScreenShots",
                          dependencies: ["LocalCodeKit", "ModelsFeature", "CodeFeature",
                                         "SpeechFeature", "VideoFeature"]),
        .executableTarget(
            name: "LocalCode",
            dependencies: ["LocalCodeKit", "ChatFeature", "ModelsFeature", "CodeFeature",
                           "ImagesFeature", "SpeechFeature", "VideoFeature"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
