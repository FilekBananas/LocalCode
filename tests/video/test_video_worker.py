"""``localcode/video/worker.py`` — the stdin/stdout JSON protocol, run in-process.

``worker.main()`` is called directly (stdin/stdout faked) rather than through
a real subprocess: the argument-validation and missing-dependency paths never
touch ``torch``/``diffusers`` before failing, so there is nothing subprocess
isolation buys these tests. The real subprocess plumbing (spawn, stdin/stdout
pump, process-group cleanup) is exercised separately in
``test_video_runtime.py`` with a tiny stand-in worker script.

Nothing here downloads or loads a real model — see ``docs/VIDEO.md`` for what
*is* and is not verified against actual LTX-Video weights.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from localcode.video import worker
from localcode.video.base import INSTALL_HINT


def _run(monkeypatch: pytest.MonkeyPatch, payload: Any) -> tuple[int, list[dict[str, Any]]]:
    """Feed ``payload`` (a dict, dumped to JSON, or a raw string) to ``worker.main()``.

    Returns ``(exit_code, stdout JSON lines)``.
    """
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    with pytest.raises(SystemExit) as excinfo:
        worker.main()
    lines = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    return int(excinfo.value.code or 0), lines


def _payload(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"\x00fake")
    base: dict[str, Any] = {
        "modelPath": str(model), "outputPath": str(tmp_path / "out.mp4"),
        "prompt": "kot w kapeluszu", "width": 704, "height": 480, "numFrames": 41,
        "fps": 24, "steps": 30, "guidance": 3.0, "seed": 7,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Argument / payload validation
# --------------------------------------------------------------------------- #


def test_a_non_json_payload_is_reported_as_bad_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    code, lines = _run(monkeypatch, "to nie jest JSON")
    assert code == 1
    assert lines[-1]["type"] == "error"
    assert lines[-1]["code"] == "bad_payload"


def test_a_json_array_payload_is_rejected_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_read_payload`` requires a JSON *object* — a bare array is not one."""
    code, lines = _run(monkeypatch, "[1, 2, 3]")
    assert code == 1
    assert lines[-1]["code"] == "bad_payload"


def test_an_empty_stdin_is_bad_payload_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty stdin parses to ``{}`` (see ``_read_payload``), which is then
    missing ``modelPath`` — reported, not a traceback."""
    code, lines = _run(monkeypatch, "")
    assert code == 1
    assert lines[-1]["code"] == "model_not_found"


def test_a_missing_model_file_is_reported_in_polish(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    payload = _payload(tmp_path, modelPath=str(tmp_path / "brak.gguf"))
    code, lines = _run(monkeypatch, payload)
    assert code == 1
    error = lines[-1]
    assert error["code"] == "model_not_found"
    assert "Nie znaleziono pliku modelu" in error["message"]


def test_a_missing_output_path_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = _payload(tmp_path, outputPath="")
    code, lines = _run(monkeypatch, payload)
    assert code == 1
    assert lines[-1]["code"] == "missing_output"


# --------------------------------------------------------------------------- #
# The missing-dependency path
# --------------------------------------------------------------------------- #


def test_reports_the_missing_video_extra_in_polish(monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path) -> None:
    """Forces the ``import diffusers`` inside ``main()`` to fail deterministically
    (``sys.modules[name] = None`` makes Python raise ``ImportError`` for that
    exact name), independent of whether the ``video`` extra actually happens to
    be installed in the environment running the tests — see ``docs/VIDEO.md``
    for what this codebase has verified about that import path (nothing: it
    has never been run against the real ``diffusers``/``torch`` stack here)."""
    monkeypatch.setitem(sys.modules, "diffusers", None)
    payload = _payload(tmp_path)
    code, lines = _run(monkeypatch, payload)

    assert code == 1
    progress = [l for l in lines if l["type"] == "progress"]
    assert progress, "a 'loading' progress line should be emitted before the import is attempted"
    error = lines[-1]
    assert error["type"] == "error"
    assert error["code"] == "missing_extra"
    assert error["message"].startswith(INSTALL_HINT)
    assert "localcode[video]" in error["message"]
    assert "diffusers" in error["message"]            # "brakujący moduł: diffusers"
