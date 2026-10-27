"""Background helpers in the terminal: their lines, the toolbar count, headless output."""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from localcode.tui.app import LocalCodeTui
from localcode.tui.headless import run_headless
from tui_fakes import ScriptedBackend


async def background_script(backend: ScriptedBackend, text: str) -> None:
    backend.emit({"type": "turn_start", "turnId": "t", "userText": text})
    backend.emit({"type": "tool_call", "callId": "c1", "name": "delegate",
                  "arguments": {"task": "uruchom mecz A"}, "title": "Pomocnik: uruchom mecz A"})
    backend.emit({"type": "tool_result", "callId": "c1", "name": "delegate", "ok": True,
                  "summary": "Pomocnik h1 pracuje w tle: uruchom mecz A", "output": ""})
    backend.emit({"type": "subagent_start", "agentId": "h1", "parentCallId": "c1",
                  "title": "uruchom mecz A", "model": "m", "remote": False, "background": True})
    backend.emit({"type": "auto_review", "callId": "x", "tool": "bash", "verdict": "allow",
                  "reason": "Uruchamia mecz.", "reviewerModel": "m", "helper": "h1"})
    backend.emit({"type": "subagent_step", "agentId": "h1", "parentCallId": "c1", "name": "bash",
                  "title": "Polecenie: cambc run", "ok": True, "summary": "Kod wyjścia 0"})
    backend.emit({"type": "subagent_end", "agentId": "h1", "parentCallId": "c1", "ok": True,
                  "steps": 1, "elapsedMs": 900, "background": True,
                  "report": "Mecz A: wygrana 3:1\nszczegóły…"})
    backend.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {"steps": 2}})


async def test_background_helper_lines_and_toolbar(tmp_path: Path, console: Console,
                                                   isolated_home: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=background_script)
    tui = LocalCodeTui(backend, console)
    tui.background_helpers.add("h9")                 # the count shows while any run
    assert any("pomocnicy w tle: 1" in text for _style, text in tui._toolbar())
    tui.background_helpers.clear()
    await tui.turn("przetestuj bota")
    out = console.export_text()
    assert "pomocnik h1 (w tle): uruchom mecz A" in out
    assert "Auto (pomocnik h1): TAK" in out
    assert "pomocnik h1 skończył w tle (1 krok) — Mecz A: wygrana 3:1" in out
    assert not tui.background_helpers, "cleared when the turn ends"


async def test_headless_prints_background_helpers(tmp_path: Path) -> None:
    backend = ScriptedBackend(tmp_path, script=background_script)
    err = io.StringIO()
    await run_headless(backend, "x", out=io.StringIO(), err=err, expand_files=False)
    text = err.getvalue()
    assert "pomocnik h1 w tle: uruchom mecz A" in text
    assert "pomocnik h1 skończył w tle — Mecz A: wygrana 3:1" in text
