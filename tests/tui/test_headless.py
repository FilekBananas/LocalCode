"""``localcode -p`` — the scriptable, non-interactive turn."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from localcode.events import PERMISSION_DISMISSED
from localcode.tui.backends import BackendError
from localcode.tui.headless import run_headless
from tui_fakes import ScriptedBackend, echo_script, failing_script, permission_script


async def run(backend: ScriptedBackend, prompt: str = "cześć", **kwargs) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = await run_headless(backend, prompt, out=out, err=err, **kwargs)
    return code, out.getvalue(), err.getvalue()


async def test_text_output_contains_only_the_answer(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=echo_script)
    code, out, err = await run(backend)
    assert code == 0
    assert out == "Cześć! Zrobione.\n"
    assert "● Odczyt src/app.py" in err
    assert "⎿ 200 linii" in err


async def test_the_prompt_reaches_the_backend(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=echo_script)
    await run(backend, "napraw testy")
    assert backend.sent == ["napraw testy"]


async def test_at_references_are_expanded(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("print(1)\n", encoding="utf-8")
    backend = ScriptedBackend(tmp_path, script=echo_script)
    await run(backend, "zobacz @a.py")
    assert '<file path="a.py">' in backend.sent[0]
    assert "print(1)" in backend.sent[0]


async def test_json_output(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=echo_script)
    code, out, _ = await run(backend, output_format="json")
    payload = json.loads(out)
    assert code == 0
    assert payload["type"] == "result"
    assert payload["subtype"] == "success"
    assert payload["isError"] is False
    assert payload["result"] == "Cześć! Zrobione."
    assert payload["reason"] == "done"
    assert payload["stats"]["steps"] == 1
    assert payload["contextUsed"] == 512 and payload["contextLimit"] == 8192
    assert payload["modelKey"] == backend.model_key
    assert any(e["type"] == "tool_call" for e in payload["events"])


async def test_stream_json_output(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=echo_script)
    code, out, _ = await run(backend, output_format="stream-json")
    lines = [json.loads(line) for line in out.strip().splitlines()]
    assert code == 0
    assert lines[0]["type"] == "turn_start"
    assert [line["type"] for line in lines].count("text_delta") == 2
    assert lines[-1]["type"] == "result"
    assert lines[-1]["result"] == "Cześć! Zrobione."


async def test_failure_sets_the_exit_code(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=failing_script)
    code, out, err = await run(backend)
    assert code == 1
    assert "Model nie odpowiada." in err
    assert out.strip() == ""


async def test_failure_in_json(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=failing_script)
    code, out, _ = await run(backend, output_format="json")
    payload = json.loads(out)
    assert code == 1
    assert payload["isError"] is True
    assert payload["subtype"] == "error"
    assert payload["errors"] == ["Model nie odpowiada."]


async def test_permission_requests_are_denied_with_an_explanation(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=permission_script)
    code, _, err = await run(backend)
    # Refused as "dismissed": nobody said no, so the agent learns nothing from it.
    assert backend.decisions == [PERMISSION_DISMISSED]
    assert "Odmówiono zgody na bash" in err
    assert "--mode accept_edits" in err
    # no answer plus a refused tool call is not a success
    assert code == 1


async def test_unknown_output_format(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path)
    with pytest.raises(BackendError):
        await run_headless(backend, "x", output_format="yaml")
