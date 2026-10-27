"""Snapshots of everything the TUI draws (``Console(record=True)``)."""

from __future__ import annotations

import pytest
from rich.console import Console

from localcode.tui import render


def draw(console: Console, renderable) -> str:
    console.print(renderable)
    return console.export_text()


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0"), (1, "1"), (999, "999"), (1000, "1k"), (5123, "5,1k"), (8000, "8k"),
     (8192, "8,2k"), (12_400, "12,4k"), (1_200_000, "1,2M")],
)
def test_fmt_tokens(value: int, expected: str) -> None:
    assert render.fmt_tokens(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(0, "0 B"), (512, "512 B"), (2048, "2 KB"), (1_610_612_736, "1,5 GB")],
)
def test_fmt_bytes(value: int, expected: str) -> None:
    assert render.fmt_bytes(value) == expected


def test_fmt_duration() -> None:
    assert render.fmt_duration(1234) == "1,2 s"
    assert render.fmt_duration(95_000) == "1 min 35 s"


def test_short_path_uses_home_and_cwd(tmp_path) -> None:
    assert render.short_path(tmp_path / "a" / "b.py", cwd=tmp_path) == "a/b.py"
    assert render.short_path("/very/long/" + "x" * 80, limit=20).startswith("…")


# --------------------------------------------------------------------------- #
# Tool cards
# --------------------------------------------------------------------------- #


def test_tool_card(console: Console) -> None:
    console.print(render.tool_call_line("Odczyt src/app.py (1–200)", "read_file"))
    console.print(render.tool_result_line("200 linii"))
    out = console.export_text()
    assert "● Odczyt src/app.py (1–200)" in out
    assert "⎿ 200 linii" in out


def test_failed_tool_card(console: Console) -> None:
    console.print(render.tool_result_line("exit 1: brak pliku", ok=False))
    assert "⎿ exit 1: brak pliku" in console.export_text()


def test_output_tail_truncates_and_indents(console: Console) -> None:
    text = "\n".join(f"linia {i}" for i in range(1, 21))
    out = draw(console, render.output_tail(text, lines=5))
    assert "… 15 wcześniejszych linii" in out
    assert "linia 20" in out
    assert "linia 5" not in out


def test_output_tail_of_empty_output_is_none() -> None:
    assert render.output_tail("   \n ") is None


DIFF = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,4 +1,4 @@
 def main():
-    print("stare")
+    print("nowe")
     return 0
"""


def test_diff_is_rendered_line_by_line(console: Console) -> None:
    out = draw(console, render.diff_renderable(DIFF))
    assert '-    print("stare")' in out
    assert '+    print("nowe")' in out
    assert "@@ -1,4 +1,4 @@" in out


def test_diff_colours_additions_green() -> None:
    colour = Console(record=True, width=80, color_system="truecolor", theme=render.THEME)
    colour.print(render.diff_renderable(DIFF))
    exported = colour.export_text(styles=True)
    assert "\x1b[" in exported  # the diff is styled at all
    assert exported.count("\x1b[") >= 4


def test_long_diff_is_cut(console: Console) -> None:
    big = "\n".join(f"+linia {i}" for i in range(100))
    out = draw(console, render.diff_renderable(big, max_lines=10))
    assert "pominięto 90 linii różnic" in out


# --------------------------------------------------------------------------- #
# Context gauge and notices
# --------------------------------------------------------------------------- #


def test_context_gauge(console: Console) -> None:
    assert draw(console, render.context_gauge(5123, 8192)).strip() == "ctx 5,1k/8,2k (63%)"


def test_context_gauge_without_limit(console: Console) -> None:
    assert "ctx —" in draw(console, render.context_gauge(0, 0))


def test_context_breakdown_table(console: Console) -> None:
    out = draw(console, render.context_breakdown(
        {"used": 5123, "limit": 8192, "system": 800, "tools": 600, "history": 3723,
         "reserve": 1024}))
    assert "prompt systemowy" in out
    assert "okno modelu" in out
    assert "8,2k" in out


def test_context_breakdown_without_data(console: Console) -> None:
    assert "Brak danych o kontekście" in draw(console, render.context_breakdown(None))


def test_compaction_notice(console: Console) -> None:
    out = draw(console, render.compaction_line(12400, 5100, "summarize"))
    assert "Kontekst skompaktowany: 12,4k → 5,1k tokenów" in out
    assert "podsumowano starsze tury" in out


def test_turn_stats_line(console: Console) -> None:
    out = draw(console, render.turn_stats_line(
        {"genTps": 12.5, "steps": 3, "elapsedMs": 4100, "completionTokens": 420,
         "cachedTokens": 1200}))
    assert "12,5 tok/s" in out
    assert "3 kroki" in out
    assert "4,1 s" in out
    assert "z cache" in out


def test_turn_stats_line_for_interrupted_turn(console: Console) -> None:
    assert "przerwano" in draw(console, render.turn_stats_line({"steps": 1}, reason="interrupted"))


def test_reasoning_summary_plural(console: Console) -> None:
    assert "1 token)" in draw(console, render.reasoning_summary(1))
    assert "3 tokeny)" in draw(Console(record=True, width=80, color_system=None),
                               render.reasoning_summary(3))
    assert "412 tokenów)" in draw(Console(record=True, width=80, color_system=None),
                                  render.reasoning_summary(412))


def test_todo_panel(console: Console) -> None:
    out = draw(console, render.todo_panel([
        {"content": "przeczytać kod", "status": "done"},
        {"content": "napisać test", "status": "in_progress"},
        {"content": "poprawić", "status": "pending"},
    ]))
    assert "Lista zadań" in out
    assert "✔ przeczytać kod" in out
    assert "▸ napisać test" in out
    assert "○ poprawić" in out


# --------------------------------------------------------------------------- #
# Permission prompt
# --------------------------------------------------------------------------- #


BASH_REQUEST = {"type": "permission_request", "requestId": "r1", "tool": "bash",
                "title": "Uruchom: pytest -q", "preview": "pytest -q"}


def test_permission_panel_shows_the_command(console: Console) -> None:
    out = draw(console, render.permission_panel(BASH_REQUEST))
    assert "Uruchom: pytest -q" in out
    assert "pytest -q" in out


def test_permission_panel_shows_a_diff(console: Console) -> None:
    request = {"tool": "edit_file", "title": "Edytuj src/app.py", "preview": DIFF}
    out = draw(console, render.permission_panel(request))
    assert '+    print("nowe")' in out


def test_permission_choices_are_claude_code_shaped() -> None:
    choices = render.permission_choices(BASH_REQUEST)
    assert [c.decision for c in choices] == ["allow", "always", "deny"]
    assert choices[0].label == "Tak"
    assert choices[1].label.startswith("Tak, i nie pytaj więcej o polecenia: pytest")
    assert choices[2].label == "Nie (Esc)"


def test_permission_question_marks_the_selected_row(console: Console) -> None:
    out = draw(console, render.permission_question(BASH_REQUEST, selected=1))
    assert "1. Tak" in out
    assert "› 2." in out
    assert "3. Nie (Esc)" in out


def test_always_label_for_a_write_tool() -> None:
    assert render.always_label("write_file", "") == "nie pytaj więcej o narzędzie write_file"


def test_always_label_uses_the_command_prefix() -> None:
    assert "git commit" in render.always_label("bash", "git commit -m 'x'")


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #


def test_welcome_panel(console: Console, tmp_path) -> None:
    model = {"key": "unsloth/Qwen3-4B/Q4", "displayName": "Qwen3 4B", "quant": "Q4_K_M",
             "format": "gguf", "contextLength": 32768}
    out = draw(console, render.welcome_panel(model=model, cwd=str(tmp_path), mode="default",
                                             n_ctx=8192))
    assert "LocalCode" in out
    assert "Qwen3 4B" in out
    assert "Q4_K_M" in out
    assert "8,2k kontekstu" in out
    assert "Domyślny" in out or "domyślny" in out


def test_welcome_panel_mentions_the_server_when_attached(console: Console, tmp_path) -> None:
    out = draw(console, render.welcome_panel(model=None, cwd=str(tmp_path), mode="plan",
                                             backend="remote"))
    assert "serwer LocalCode" in out
    assert "brak modelu" in out


def test_models_table(console: Console) -> None:
    from tui_fakes import make_model_dto

    out = draw(console, render.models_table([make_model_dto(loaded=True)],
                                            budget_bytes=24_000_000_000))
    assert "Qwen3-0.6B-Q4_K_M" in out
    assert "GGUF" in out
    assert "załadowany" in out


def test_instances_table_empty(console: Console) -> None:
    assert "Żaden model nie jest załadowany." in draw(console, render.instances_table({}))


def test_runtime_panel(console: Console) -> None:
    status = {
        "instances": [],
        "system": {"totalBytes": 34_359_738_368, "usedBytes": 8_589_934_592,
                   "availableBytes": 25_769_803_776},
        "budget": {"limitBytes": 24_000_000_000, "usedBytes": 0},
        "settings": {"maxLoadedModels": 2, "maxInstancesPerModel": 1, "defaultParallel": 1,
                     "ramLimitGb": None, "idleTtlMinutes": 30, "defaultCtx": None,
                     "flashAttn": "auto", "kvCacheType": "f16", "gpuLayers": -1,
                     "evictPolicy": "lru"},
    }
    out = draw(console, render.runtime_panel(status))
    assert "budżet modeli" in out
    assert "max-models" in out
    assert "auto" in out


def test_hub_table(console: Console) -> None:
    out = draw(console, render.hub_table([
        {"id": "unsloth/Qwen3-4B-GGUF", "format": "gguf", "paramsB": 4.0, "downloads": 12345,
         "description": "Szybki model"}]))
    assert "unsloth/Qwen3-4B-GGUF" in out
    assert "12 345" in out


def test_mode_label() -> None:
    assert render.mode_label("accept_edits") == "akceptuj edycje"
    assert "pyta o polecenia" in render.mode_label("accept_edits", long=True)


def test_model_line_does_not_repeat_the_quantisation() -> None:
    """The library's display name usually already ends with the quantisation."""
    line = render.model_line({"displayName": "Qwen3 0.6B · Q4_K_M", "quant": "Q4_K_M",
                              "format": "gguf", "contextLength": 40960})
    assert line == "Qwen3 0.6B · Q4_K_M · GGUF · 41k kontekstu"


def test_model_line_adds_a_missing_quantisation() -> None:
    assert render.model_line({"displayName": "Qwen3 0.6B", "quant": "Q4_K_M", "format": "gguf"}) \
        == "Qwen3 0.6B · Q4_K_M · GGUF"


def test_welcome_panel_in_chat_mode(console: Console, tmp_path) -> None:
    out = draw(console, render.welcome_panel(model=None, cwd=str(tmp_path), mode="default",
                                             chat=True))
    assert "czat — bez narzędzi" in out
    assert "/chat — powrót do agenta" in out
    assert "Shift+Tab" not in out


@pytest.mark.parametrize("value, expected", [(0.596, "0,6"), (4, "4"), (None, "—"), (12.25, "12,25")])
def test_fmt_number(value, expected: str) -> None:
    assert render.fmt_number(value) == expected


def test_auto_review_line_says_yes_or_no_and_why() -> None:
    yes = render.auto_review_line("allow", "Użytkownik prosił o testy.")
    assert yes.plain == "  ✓ Auto: TAK — Użytkownik prosił o testy."
    no = render.auto_review_line("deny", "Usuwa pliki.")
    assert no.plain == "  ✗ Auto: NIE — Usuwa pliki."
    assert "lc.err" in str(no.style) or any("lc.err" in str(s.style) for s in no.spans)
    assert render.auto_review_line("ask", "").plain == "  ? Auto: pytam Ciebie"


def test_auto_mode_has_a_label() -> None:
    assert render.mode_label("auto") == "auto"
    assert "ocenia" in render.mode_label("auto", long=True)
