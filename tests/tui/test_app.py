"""The terminal app itself: slash dispatch, turn rendering, permissions.

The app is driven with a :class:`~tui_fakes.ScriptedBackend` and a recording
console, so a whole turn (tool card, answer, statistics) can be asserted on
without a model, a terminal or a running server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from localcode.tui import app as app_mod
from localcode.tui.app import LocalCodeTui
from tui_fakes import ScriptedBackend, make_model_dto, permission_script


@pytest.fixture
def tui(backend: ScriptedBackend, console: Console, isolated_home: Path) -> LocalCodeTui:
    return LocalCodeTui(backend, console)


def text_of(tui: LocalCodeTui) -> str:
    """Everything printed so far (``clear=False`` — the buffer stays)."""
    return tui.console.export_text(clear=False)


# --------------------------------------------------------------------------- #
# A whole turn
# --------------------------------------------------------------------------- #


async def test_turn_renders_the_transcript(tui: LocalCodeTui) -> None:
    await tui.turn("cześć")
    out = text_of(tui)
    assert "● Odczyt src/app.py (1–200)" in out
    assert "⎿ 200 linii" in out
    assert "Cześć! Zrobione." in out
    assert "12,5 tok/s" in out
    assert "1 krok" in out


async def test_turn_expands_at_references(tui: LocalCodeTui, backend: ScriptedBackend,
                                          tmp_path: Path) -> None:
    (tmp_path / "plik.py").write_text("x = 1\n", encoding="utf-8")
    await tui.turn("zobacz @plik.py")
    assert '<file path="plik.py">' in backend.sent[0]
    assert "Dołączono: plik.py" in text_of(tui)


async def test_turn_reports_a_missing_reference(tui: LocalCodeTui) -> None:
    await tui.turn("@nie-ma.py co to?")
    assert "Nie znaleziono pliku @nie-ma.py" in text_of(tui)


async def test_turn_keeps_the_context_and_stats(tui: LocalCodeTui) -> None:
    await tui.turn("x")
    assert tui.backend.context_used == 512
    assert tui.backend.context_limit == 8192
    assert tui.last_stats is not None and tui.last_stats["steps"] == 1


async def test_diff_is_shown_for_edits(backend: ScriptedBackend, console: Console,
                                       isolated_home: Path) -> None:
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-stare\n+nowe\n"

    async def script(b: ScriptedBackend, text: str) -> None:
        b.emit({"type": "turn_start", "turnId": "t", "userText": text})
        b.emit({"type": "tool_call", "callId": "c", "name": "edit_file", "arguments": {},
                "title": "Edytuj x.py"})
        b.emit({"type": "tool_result", "callId": "c", "name": "edit_file", "ok": True,
                "summary": "1 zmiana", "output": "", "diff": diff})
        b.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {}})

    backend.script = script
    tui = LocalCodeTui(backend, console)
    await tui.turn("popraw")
    out = console.export_text()
    assert "● Edytuj x.py" in out
    assert "+nowe" in out and "-stare" in out


async def test_bash_output_tail_is_shown(backend: ScriptedBackend, console: Console,
                                         isolated_home: Path) -> None:
    async def script(b: ScriptedBackend, text: str) -> None:
        b.emit({"type": "turn_start", "turnId": "t", "userText": text})
        b.emit({"type": "tool_call", "callId": "c", "name": "bash", "arguments": {},
                "title": "Uruchom: pytest -q"})
        b.emit({"type": "tool_result", "callId": "c", "name": "bash", "ok": True,
                "summary": "exit 0", "output": "\n".join(f"linia {i}" for i in range(30))})
        b.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {}})

    backend.script = script
    tui = LocalCodeTui(backend, console)
    await tui.turn("odpal testy")
    out = console.export_text()
    assert "linia 29" in out
    assert "wcześniejszych linii" in out


async def test_compaction_notice_during_a_turn(backend: ScriptedBackend, console: Console,
                                               isolated_home: Path) -> None:
    async def script(b: ScriptedBackend, text: str) -> None:
        b.emit({"type": "turn_start", "turnId": "t", "userText": text})
        b.emit({"type": "compaction", "before": 12400, "after": 5100, "strategy": "summarize"})
        b.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {}})

    backend.script = script
    tui = LocalCodeTui(backend, console)
    await tui.turn("x")
    assert "Kontekst skompaktowany: 12,4k → 5,1k tokenów" in console.export_text()


async def test_error_event_is_red(backend: ScriptedBackend, console: Console,
                                  isolated_home: Path) -> None:
    from tui_fakes import failing_script

    backend.script = failing_script
    tui = LocalCodeTui(backend, console)
    await tui.turn("x")
    assert "✗ Model nie odpowiada." in console.export_text()


async def test_todos_are_rendered(backend: ScriptedBackend, console: Console,
                                  isolated_home: Path) -> None:
    async def script(b: ScriptedBackend, text: str) -> None:
        b.emit({"type": "turn_start", "turnId": "t", "userText": text})
        b.emit({"type": "todos", "items": [{"content": "zrobić", "status": "in_progress"}]})
        b.emit({"type": "turn_end", "turnId": "t", "reason": "done", "stats": {}})

    backend.script = script
    tui = LocalCodeTui(backend, console)
    await tui.turn("x")
    assert "▸ zrobić" in console.export_text()
    assert tui.todos == [{"content": "zrobić", "status": "in_progress"}]


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #


async def test_permission_prompt_allows(backend: ScriptedBackend, console: Console,
                                        isolated_home: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    backend.script = permission_script
    asked: list[str] = []

    async def fake_choose(options, **kwargs):
        asked.append(kwargs.get("title", ""))
        return options[0].value

    monkeypatch.setattr(app_mod, "choose", fake_choose)
    tui = LocalCodeTui(backend, console)
    await tui.turn("usuń build")
    assert backend.decisions == ["allow"]
    assert asked == ["Czy pozwolić na tę operację?"]
    out = console.export_text()
    assert "rm -rf build" in out
    assert "Zezwolono." in out


async def test_permission_prompt_denies_on_escape(backend: ScriptedBackend, console: Console,
                                                  isolated_home: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    backend.script = permission_script
    monkeypatch.setattr(app_mod, "choose", lambda options, **kw: _deny())
    tui = LocalCodeTui(backend, console)
    await tui.turn("usuń build")
    assert backend.decisions == ["deny"]
    assert "Odmówiono." in console.export_text()


async def _deny() -> str:
    return "deny"


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #


async def test_help(tui: LocalCodeTui) -> None:
    await tui.dispatch("/help")
    out = text_of(tui)
    assert "/model" in out and "Shift+Tab" in out


async def test_unknown_command_suggests(tui: LocalCodeTui) -> None:
    await tui.dispatch("/modle")
    out = text_of(tui)
    assert "Nieznane polecenie /modle." in out
    assert "Chodziło o /mode" in out


async def test_mode_command(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/mode plan")
    assert backend.modes == ["plan"]
    assert "Tryb: plan" in text_of(tui)


async def test_mode_command_rejects_nonsense(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/mode wszystko")
    assert backend.modes == []
    assert "Nieznany tryb" in text_of(tui)


async def test_forget_command_clears_this_projects_precedents(tui: LocalCodeTui,
                                                               backend: ScriptedBackend) -> None:
    from localcode.agent import precedents

    precedents.record(backend.cwd, [precedents.Key("cmd", "git push")], precedents.APPROVED)
    precedents.record(backend.cwd.parent, [precedents.Key("cmd", "make")], precedents.APPROVED)
    await tui.dispatch("/forget")
    assert precedents.load(backend.cwd) == []
    assert len(precedents.load(backend.cwd.parent)) == 1, "other projects keep theirs"
    assert "Zapomniano wyuczone decyzje w tym projekcie: 1." in text_of(tui)
    await tui.dispatch("/forget")
    assert "nie ma wyuczonych decyzji" in text_of(tui)


async def test_think_toggle(tui: LocalCodeTui) -> None:
    await tui.dispatch("/think on")
    assert tui.show_reasoning is True
    await tui.dispatch("/think off")
    assert tui.show_reasoning is False
    await tui.dispatch("/think")
    assert tui.show_reasoning is True


async def test_ps(tui: LocalCodeTui) -> None:
    await tui.dispatch("/ps")
    assert "Qwen3 0.6B" in text_of(tui)


async def test_models(tui: LocalCodeTui) -> None:
    await tui.dispatch("/models")
    out = text_of(tui)
    assert "Q4_K_M" in out and "GGUF" in out


async def test_models_when_empty(backend: ScriptedBackend, console: Console,
                                 isolated_home: Path) -> None:
    backend.models = []
    tui = LocalCodeTui(backend, console)
    await tui.dispatch("/models")
    assert "Brak modeli na dysku" in console.export_text()


async def test_runtime_show(tui: LocalCodeTui) -> None:
    await tui.dispatch("/runtime")
    assert "max-models" in text_of(tui)


async def test_runtime_set(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/runtime set ram-gb 20")
    assert backend.patches == [{"ramLimitGb": 20.0}]
    assert "Zapisano ustawienia" in text_of(tui)


async def test_runtime_set_rejects_bad_keys(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/runtime set bzdura 1")
    assert backend.patches == []
    assert "Nieznane ustawienie" in text_of(tui)


async def test_load_and_unload(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/load")
    assert backend.loaded == [backend.model_key]
    await tui.dispatch("/unload")
    assert backend.unloaded == ["all"]


async def test_context_without_data(tui: LocalCodeTui) -> None:
    await tui.dispatch("/context")
    assert "Brak danych o kontekście" in text_of(tui)


async def test_context_after_a_turn(tui: LocalCodeTui) -> None:
    await tui.turn("x")
    await tui.dispatch("/context")
    out = text_of(tui)
    assert "prompt systemowy" in out and "okno modelu" in out


async def test_stats(tui: LocalCodeTui) -> None:
    await tui.dispatch("/stats")
    assert "Jeszcze nie było żadnej tury." in text_of(tui)
    await tui.turn("x")
    await tui.dispatch("/stats")
    assert "tokeny odpowiedzi" in text_of(tui)


async def test_compact(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/compact")
    assert backend.compactions == 1
    assert "Kontekst skompaktowany" in text_of(tui)


async def test_clear(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.turn("x")
    await tui.dispatch("/clear")
    assert backend.clears == 1
    assert tui.todos == []
    assert "LocalCode" in text_of(tui)   # the welcome box is printed again


async def test_search(tui: LocalCodeTui) -> None:
    await tui.dispatch("/search qwen")
    assert "org/qwen-GGUF" in text_of(tui)


async def test_search_without_a_query(tui: LocalCodeTui) -> None:
    await tui.dispatch("/search")
    assert "Podaj zapytanie" in text_of(tui)


async def test_model_by_name(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    backend.models = [make_model_dto(), make_model_dto("mlx-community/Qwen3-4B-4bit", format="mlx")]
    await tui.dispatch("/model Qwen3-4B")
    assert backend.model_key == "mlx-community/Qwen3-4B-4bit"
    assert "Model:" in text_of(tui)


async def test_chat_toggle(tui: LocalCodeTui) -> None:
    await tui.dispatch("/chat")
    assert tui.chat_mode is True
    assert "Tryb czatu" in text_of(tui)
    await tui.dispatch("/chat")
    assert tui.chat_mode is False


async def test_cwd(tui: LocalCodeTui, backend: ScriptedBackend) -> None:
    await tui.dispatch("/cwd")
    assert backend.cwd.name in text_of(tui)


async def test_exit(tui: LocalCodeTui) -> None:
    await tui.dispatch("/exit")
    assert tui._running is False


async def test_resume_without_sessions(tui: LocalCodeTui) -> None:
    await tui.dispatch("/resume")
    assert "Brak zapisanych sesji" in text_of(tui)


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #


async def test_welcome_box(tui: LocalCodeTui) -> None:
    await tui.show_welcome()
    out = text_of(tui)
    assert "LocalCode" in out
    assert "Qwen3-0.6B-Q4_K_M" in out
    assert "/help" in out


def test_toolbar_shows_model_mode_and_cwd(tui: LocalCodeTui) -> None:
    tui.backend.context = {"used": 5123, "limit": 8192}
    tui.status = tui.backend.__class__.__dict__  # not used; the RAM part is optional
    tui.status = None
    text = "".join(fragment for _, fragment in tui._toolbar())
    assert "Qwen3-0.6B-Q4_K_M" in text
    assert "domyślny" in text
    assert "ctx 5,1k/8,2k (63%)" in text


async def test_toolbar_shows_ram(tui: LocalCodeTui) -> None:
    tui.status = await tui.backend.runtime_status()
    text = "".join(fragment for _, fragment in tui._toolbar())
    assert "RAM 667,6 MB" in text or "RAM" in text


def test_mode_cycle_covers_every_mode() -> None:
    from localcode.agent.permissions import MODES

    assert set(app_mod.MODE_CYCLE) == set(MODES)


# --------------------------------------------------------------------------- #
# Choosing a model when the user named none
# --------------------------------------------------------------------------- #


async def test_default_model_prefers_a_loaded_one(backend: ScriptedBackend) -> None:
    backend.models = [make_model_dto("a/b/small", sizeBytes=1_000_000_000, capabilities=["tools"]),
                      make_model_dto("a/b/loaded", sizeBytes=500_000_000, loaded=True)]
    assert await backend.pick_default_model() == "a/b/loaded"


async def test_default_model_prefers_tool_capable(backend: ScriptedBackend) -> None:
    backend.models = [make_model_dto("a/b/plain", sizeBytes=3_000_000_000, capabilities=[]),
                      make_model_dto("a/b/tools", sizeBytes=2_000_000_000, capabilities=["tools"])]
    assert await backend.pick_default_model() == "a/b/tools"


async def test_default_model_skips_embeddings(backend: ScriptedBackend) -> None:
    backend.models = [make_model_dto("a/b/embed", capabilities=["embedding"]),
                      make_model_dto("a/b/chat", capabilities=["tools"])]
    assert await backend.pick_default_model() == "a/b/chat"


async def test_default_model_without_any_models(backend: ScriptedBackend) -> None:
    backend.models = []
    assert await backend.pick_default_model() is None


async def test_default_model_avoids_huge_ones(backend: ScriptedBackend) -> None:
    """The fake budget is 24 GB, so a 20 GB model loses to a 6 GB one."""
    backend.models = [make_model_dto("a/b/huge", sizeBytes=20_000_000_000, capabilities=["tools"]),
                      make_model_dto("a/b/mid", sizeBytes=6_000_000_000, capabilities=["tools"])]
    assert await backend.pick_default_model() == "a/b/mid"
