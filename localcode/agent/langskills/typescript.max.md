**Toolchain.** TypeScript compiles to JS: `tsc` builds per `tsconfig.json`; `tsc --noEmit` typechecks only. Many projects run it through a bundler (Vite, esbuild, tsup, Next) or `ts-node`/`tsx` for scripts. Use the project's `npm run build`/`typecheck`. Read `tsconfig.json`: `strict`, `target`, `module`, `moduleResolution`, `paths`.

**Typing discipline.** Annotate exported/public function parameters and return types; let inference cover local variables. Prefer `interface` for object shapes you extend, `type` for unions/intersections/aliases. Use `unknown` (then narrow) instead of `any`; discriminated unions + exhaustive `switch` (with a `never` default) for variants; generics to avoid duplication; `readonly`, `as const`, utility types (`Partial`, `Pick`, `Omit`, `Record`, `ReturnType`). Narrow with type guards (`typeof`, `in`, custom `x is T`).

**Avoid.** `any` (defeats the point), non-null assertion `!` unless truly safe, `@ts-ignore`/`@ts-expect-error` to mask real errors, enums when a union of string literals is clearer, `as` casts that lie about runtime shape.

**Strict mode.** Under `strict` you must handle `null`/`undefined` (`?.`, `??`, guards), no implicit `any`, correct `this` types. Keep it on.

**Deps & modules.** Same package-manager rules as JavaScript (lockfile decides). Ship/consume type definitions (`@types/*` for untyped libs). ESM vs CJS follows `package.json` `"type"` and tsconfig `module`.

**Testing.** Vitest or Jest (`ts-jest`/`babel-jest`) are typical; files `*.test.ts`/`*.spec.ts`. Run `npm test`. Type-level expectations can be asserted with `expectTypeOf`/`tsd`.

**Pitfalls.** Trusting `as` casts; structural typing surprises (excess-property checks only on literals); `Promise` not awaited; `Date`/timezones; declaration-file drift from the implementation; mixing ESM/CJS interop (`esModuleInterop`).
