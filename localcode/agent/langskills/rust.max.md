**Cargo.** `Cargo.toml` (manifest) + `Cargo.lock`. `cargo build` (`--release` for optimized), `cargo test`, `cargo run`, `cargo check` (fast typecheck), `cargo clippy` (lints — treat warnings seriously), `cargo fmt` (canonical format). Deps: `cargo add <crate>` or edit `[dependencies]`; features via `--features`. Workspaces group crates under a top-level `Cargo.toml`.

**Ownership.** The core model: each value has one owner; borrows are `&T` (shared) or `&mut T` (exclusive, non-aliasing); lifetimes keep references valid. Prefer borrowing over moving/cloning; `clone()` is fine when it clarifies more than it costs. Use `Rc`/`Arc` for shared ownership, `RefCell`/`Mutex` for interior mutability — reach for them deliberately, not to dodge the borrow checker.

**Errors.** Fallible functions return `Result<T, E>`; propagate with `?`. Define error types or use `thiserror` (libraries) / `anyhow` (applications) if already in the project. `Option<T>` replaces null; combinators `map`/`and_then`/`unwrap_or`/`ok_or`. `unwrap()`/`expect()` only when a failure is a genuine bug or in tests; avoid `panic!` in library code.

**Idioms.** `enum` + `match` (exhaustive) for state; pattern matching and `if let`/`let else`; iterator chains (`map`/`filter`/`collect`) over manual loops; traits for shared behavior, generics with trait bounds; `impl Trait`; `derive` for `Debug`/`Clone`/`PartialEq`/`Default`/`Serialize`. Prefer `&str`/`&[T]` params, owned `String`/`Vec` returns. `unsafe` only with a clear invariant comment.

**Testing.** Unit tests in-module under `#[cfg(test)] mod tests { use super::*; #[test] fn ... }`; integration tests in `tests/`; doc-tests in `///` examples run by `cargo test`. `assert!`/`assert_eq!`, `#[should_panic]`, `?` in tests returning `Result`. Run `cargo test <filter>`.

**Pitfalls.** Fighting the borrow checker with `clone()`/`unsafe` instead of restructuring; `.unwrap()` in production paths; integer overflow (debug panics, release wraps — use checked/saturating ops); blocking in async; holding a `MutexGuard` across `.await`; moving out of a borrow.
