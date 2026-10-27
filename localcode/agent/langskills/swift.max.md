**Build system.** SwiftPM: `Package.swift` defines targets/products/dependencies; `swift build`, `swift test`, `swift run <executable>`, `swift package resolve`. App projects use Xcode (`.xcodeproj`/`.xcworkspace`) built with `xcodebuild -scheme X build test` or the IDE. Match whichever the repo uses.

**Layout.** SwiftPM: `Sources/<Target>/`, `Tests/<Target>Tests/`. Dependencies are added to `Package.swift` `dependencies` and each target's `dependencies`.

**Idioms.** `let` unless mutation is needed. Value types first: `struct` and `enum` with associated values; `class` only for reference semantics/identity/Obj-C interop; mark classes `final` when not subclassed. Optionals: unwrap with `if let`/`guard let`, `??`, optional chaining `?.`; avoid force-unwrap `!` and force-cast `as!` except where a failure is a real programmer error. Protocols + protocol extensions for shared behavior; generics with constraints. Prefer `map`/`filter`/`compactMap`/`reduce`.

**Concurrency.** Modern code uses `async`/`await`, `Task`, `actor` for shared mutable state, and `@MainActor` for UI. Respect existing GCD (`DispatchQueue`) usage if that's the style.

**Errors.** Throwing functions (`throws`/`try`/`do-catch`), typed via `Error` conformances; `try?`/`try!` sparingly; `Result<Success, Failure>` at async boundaries. `defer` for cleanup.

**Style.** Swift API Design Guidelines: lowerCamelCase members, UpperCamelCase types, argument labels that read as phrases, no Hungarian/`get` prefixes. Format with `swift-format` or SwiftLint if configured.

**Testing.** XCTest: subclass `XCTestCase`, methods `testX()`, `XCTAssertEqual`/`XCTAssertThrowsError`, `measure {}` for perf. Newer code uses swift-testing: `import Testing`, `@Test func …`, `#expect(...)`, `#require(...)`. Run `swift test` (`--filter` to select).

**Pitfalls.** Retain cycles in closures/delegates — use `[weak self]`/`unowned` and `weak` delegates; force-unwraps crashing in production; blocking the main thread; `Codable` key mismatches; integer overflow (use `&+` only intentionally); mutating `struct` captured by value.
