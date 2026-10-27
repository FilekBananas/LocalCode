**Header & safety.** `#!/usr/bin/env bash` then `set -euo pipefail` (`-e` exit on error, `-u` error on unset variable, `-o pipefail` catch failures mid-pipe). Set `IFS=$'\n\t'` when word-splitting matters. For a portable `/bin/sh` script, avoid bashisms entirely (no `[[`, arrays, `local` is non-POSIX). Lint everything with `shellcheck` — fix its warnings.

**Quoting & expansion.** Quote every expansion: `"$var"`, `"$@"` (not `$*`), `"${array[@]}"`, `"$(cmd)"`. Unquoted expansions word-split and glob. Use `${var:-default}`, `${var:?msg}`, `${var%suffix}`/`${var#prefix}` parameter expansion instead of calling `sed`/`cut` for simple edits. `$(...)` for substitution, never backticks. `(( ))` for arithmetic.

**Control & functions.** `[[ ... ]]` for tests (supports `&&`, `==`, `=~`), `[ ... ]`/`test` only for POSIX. Functions declare locals with `local`; return status via `return N`, output via stdout. Check exit codes (`if cmd; then`), and whether a command exists (`command -v tool`).

**Files & robustness.** Create temp files/dirs with `mktemp`/`mktemp -d` and remove them in a `trap 'rm -rf "$tmp"' EXIT`. Never parse `ls`; iterate with globs (`for f in ./*.txt`) or `find ... -print0 | while IFS= read -r -d '' f`. Use `read -r` (raw, no backslash mangling). Beware filenames with spaces/newlines.

**Style.** Lowercase variable names for locals, UPPERCASE for exported/env; descriptive names; small functions; a `usage()` and arg parsing (`getopts` or a `while`/`case`). Prefer readability; drop into `awk`/`python` when logic gets complex.

**Testing.** Syntax check `bash -n script.sh`; trace with `bash -x` or `set -x`. bats-core for real test suites (`@test "..." { ... }`). Run the script against representative inputs and edge cases (empty input, spaces, missing files).

**Pitfalls.** Unquoted variables (word-splitting/globbing); `cd` without checking it succeeded; `rm -rf "$dir/"` when `$dir` is empty; relying on `$?` after other commands; masking errors in pipelines (without pipefail); non-portable `echo -e`/`echo -n` (use `printf`); `set -e` not triggering inside `if`/`||`/command substitution.
