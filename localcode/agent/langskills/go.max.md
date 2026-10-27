**Tooling.** Modules via `go.mod`/`go.sum`. `go build ./...`, `go test ./...`, `go vet ./...`, `go run .`. Format is non-negotiable: `gofmt`/`go fmt ./...` (or `goimports`). Deps: `go get pkg@version`, then `go mod tidy` to prune. Lint with `golangci-lint` if present. `go test -race ./...` to catch data races.

**Layout.** Package per directory; `package main` + `func main()` for binaries, often under `cmd/<name>/`. Internal-only code under `internal/`. Keep package names short, lowercase, no underscores. Exported identifiers start Uppercase; everything else is package-private.

**Errors.** Errors are values — return `(T, error)` and check every one: `if err != nil { return err }`. Add context with `fmt.Errorf("doing X: %w", err)` (the `%w` keeps the chain for `errors.Is`/`errors.As`). Use `panic` only for truly unrecoverable states; `recover` only at boundaries. Sentinel errors via `errors.New`/`var ErrX`.

**Idioms.** Rely on zero values (a `nil` slice/map-read is usable). Accept interfaces, return concrete types; keep interfaces small (often one method). `defer` for unlocks/closes. Use `context.Context` as the first parameter for cancellation/deadlines. Concurrency: goroutines + channels, or `sync.Mutex`/`sync.WaitGroup`/`errgroup`; never share memory without synchronization.

**Style.** Effective Go / the standard library are the reference. No getters named `GetX` (just `X`). Document exported identifiers with a comment starting with the name. Group imports (stdlib, then third-party).

**Testing.** `*_test.go` in the same package (or `_test` package for black-box). `func TestX(t *testing.T)`, table-driven subtests with `t.Run`. Benchmarks `BenchmarkX(b *testing.B)`, fuzz `FuzzX`. `t.Helper()`, `t.Cleanup()`. Run `go test ./... -run X -v`.

**Pitfalls.** Loop-variable capture in goroutines (fixed in Go 1.22+, but be explicit); `nil` map writes panic (make it first); ignoring errors; forgetting to close resources; shadowing with `:=`; appending to a shared slice; goroutine leaks when a channel never receives.
