"""Auto mode in the terminal: the verdicts above the question, and ``/think ultracode``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import localcode.tui.app as app_mod
from localcode.tui import render
from localcode.tui.app import LocalCodeTui
from localcode.tui.headless import run_headless
from tui_fakes import ScriptedBackend

REVIEW = {"verdict": "deny", "reason": "Co robi: Usuwa katalog build. Powód: Nikt nie prosił.",
          "model": "test/model",
          "guard": {"verdict": "deny", "reason": "Co robi: Usuwa pliki. Powód: Niszczy dane.",
                    "model": "local/tiny"}}


def review_script(*, with_event: bool = True, review: dict[str, Any] | None = REVIEW):
    async def script(backend: ScriptedBackend, text: str) -> None:
        backend.emit({"type": "turn_start", "turnId": "t", "userText": text})
        backend.emit({"type": "tool_call", "callId": "c1", "name": "bash",
                      "arguments": {"command": "rm -r build"}, "title": "Polecenie: rm -r build"})
        if with_event and review is not None:
            backend.emit({"type": "auto_review", "callId": "c1", "tool": "bash",
                          "verdict": review["verdict"], "reason": review["reason"],
                          "reviewerModel": review["model"], "guard": review.get("guard")})
        request = {"type": "permission_request", "requestId": "r1", "tool": "bash",
                   "title": "Polecenie: rm -r build",
                   "preview": "Co robi: Sprząta\n\nrm -r build", "options": ["allow", "deny"]}
        if review is not None:
            request["review"] = review
        backend.emit(request)
        decision = await backend.wait_permission("r1")
        backend.decisions.append(decision)
        backend.emit({"type": "tool_result", "callId": "c1", "name": "bash", "ok": False,
                      "summary": "Odrzucone", "output": ""})
        backend.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {"steps": 1}})

    return script


def test_the_review_lines_name_both_gates() -> None:
    text = render.permission_review_line(REVIEW).plain
    first, second = text.splitlines()
    assert first.startswith("  ✗ Recenzent: NIE — Co robi: Usuwa katalog build.")
    assert second.startswith("  ✗ Strażnik lokalny: NIE — Co robi: Usuwa pliki.")
    advisory = render.permission_review_line({"verdict": "allow", "reason": "Ok.", "model": "m"})
    assert advisory.plain == "  ✓ Recenzent: TAK (tylko opinia) — Ok."
    assert render.permission_review_line({"verdict": "ask", "reason": "", "model": ""}).plain == \
        "  ? Recenzent nie zdecydował"
    assert render.permission_review_line(None) is None


async def test_the_question_shows_the_review_once(tmp_path: Path, console: Console,
                                                  isolated_home: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ScriptedBackend(tmp_path, script=review_script())

    async def deny(options, **kwargs):
        return "deny"

    monkeypatch.setattr(app_mod, "choose", deny)
    tui = LocalCodeTui(backend, console)
    await tui.turn("posprzątaj")
    out = console.export_text()
    assert backend.decisions == ["deny"]
    assert out.count("Usuwa katalog build") == 1, "printed above the question, not twice"
    assert "Recenzent: NIE" in out and "Strażnik lokalny: NIE" in out
    assert out.index("Recenzent: NIE") < out.index("rm -r build", out.index("Recenzent: NIE"))


async def test_a_verdict_with_no_question_after_it_still_shows(tmp_path: Path, console: Console,
                                                               isolated_home: Path) -> None:
    async def script(backend: ScriptedBackend, text: str) -> None:
        backend.emit({"type": "turn_start", "turnId": "t", "userText": text})
        backend.emit({"type": "auto_review", "callId": "c1", "tool": "bash", "verdict": "deny",
                      "reason": "Nie teraz.", "reviewerModel": "m"})
        backend.emit({"type": "tool_result", "callId": "c1", "name": "bash", "ok": False,
                      "summary": "Odrzucone przez recenzenta", "output": ""})
        backend.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {}})

    tui = LocalCodeTui(ScriptedBackend(tmp_path, script=script), console)
    await tui.turn("x")
    assert "Auto: NIE — Nie teraz." in console.export_text()


async def test_headless_prints_an_advisory_review_above_the_refusal(tmp_path: Path) -> None:
    import io

    advisory = {"verdict": "allow", "reason": "Co robi: Usuwa build.", "model": "m"}
    backend = ScriptedBackend(tmp_path, script=review_script(with_event=False, review=advisory))
    err = io.StringIO()
    await run_headless(backend, "x", out=io.StringIO(), err=err, expand_files=False)
    text = err.getvalue()
    assert "Recenzent: TAK (tylko opinia) — Co robi: Usuwa build." in text
    assert text.index("Recenzent: TAK") < text.index("Odmówiono zgody")


async def test_think_ultracode_is_for_the_agent(tmp_path: Path, console: Console,
                                                 isolated_home: Path) -> None:
    tui = LocalCodeTui(ScriptedBackend(tmp_path), console)
    await tui.dispatch("/think ultracode")
    assert tui.backend.effort == "ultracode"
    assert "wielu pomocników" in console.export_text(clear=False)
    await tui.dispatch("/think")                    # the cycle never lands on ultracode
    assert tui.backend.effort == "off"
    tui.chat_mode = True
    await tui.dispatch("/think ultracode")
    assert tui.backend.effort == "off"
    assert "poziom agenta" in console.export_text(clear=False)
