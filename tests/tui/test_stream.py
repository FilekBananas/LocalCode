"""The live display of a turn: paragraph flushing and the collapsed reasoning."""

from __future__ import annotations

import pytest
from rich.console import Console

from localcode.tui import render
from localcode.tui.stream import TurnView, split_flushable


@pytest.mark.parametrize(
    "text, done, rest",
    [
        ("", "", ""),
        ("jeszcze piszę", "", "jeszcze piszę"),
        ("akapit\n\ndrugi", "akapit\n", "drugi"),
        ("a\n\nb\n\nc", "a\n\nb\n", "c"),
        ("linia\nbez pustej", "", "linia\nbez pustej"),
    ],
)
def test_split_flushable(text: str, done: str, rest: str) -> None:
    assert split_flushable(text) == (done, rest)


def test_code_fences_are_not_split() -> None:
    text = "opis\n\n```python\ndef f():\n\n    return 1\n"
    done, rest = split_flushable(text)
    assert done == "opis\n"
    assert rest.startswith("```python")
    assert "\n\n" in rest          # the blank line inside the block stays pending


def test_a_closed_fence_can_be_flushed() -> None:
    text = "```python\nx = 1\n```\n\nreszta"
    done, rest = split_flushable(text)
    assert done == "```python\nx = 1\n```\n"
    assert rest == "reszta"


def test_the_last_line_is_never_final() -> None:
    assert split_flushable("akapit\n\n")[1] == ""
    assert split_flushable("akapit\n")[0] == ""


# --------------------------------------------------------------------------- #
# TurnView
# --------------------------------------------------------------------------- #


def test_markdown_is_printed_paragraph_by_paragraph(console: Console) -> None:
    with TurnView(console) as view:
        view.feed_text("# Nagłówek\n\n")
        mid = console.export_text(clear=False)
        view.feed_text("Treść odpowiedzi.")
    out = console.export_text(clear=False)
    assert "Nagłówek" in mid
    assert "Treść odpowiedzi." not in mid      # still pending
    assert "Treść odpowiedzi." in out          # flushed by finish()
    assert view.text == "# Nagłówek\n\nTreść odpowiedzi."


def test_reasoning_collapses_to_one_line(console: Console) -> None:
    with TurnView(console) as view:
        view.feed_reasoning("x" * 400)
        view.feed_text("odpowiedź")
    out = console.export_text()
    assert "Myślenie" in out
    assert "xxxx" not in out


def test_reasoning_is_shown_when_asked(console: Console) -> None:
    with TurnView(console, show_reasoning=True) as view:
        view.feed_reasoning("rozumowanie modelu")
        view.feed_text("odpowiedź")
    out = console.export_text()
    assert "rozumowanie modelu" in out
    assert "Myślenie" in out


def test_printing_above_the_live_region(console: Console) -> None:
    with TurnView(console) as view:
        view.print(render.tool_call_line("Odczyt a.py"))
        view.feed_text("gotowe")
    out = console.export_text()
    assert out.index("Odczyt a.py") < out.index("gotowe")


def test_status_does_not_survive_the_turn(console: Console) -> None:
    view = TurnView(console)
    with view:
        view.set_status("Ładowanie modelu…", "loading")
        view.feed_text("już")
    assert view._status is None


def test_paused_view_can_be_resumed(console: Console) -> None:
    with TurnView(console) as view:
        view.feed_text("przed\n\n")
        with view.paused():
            console.print("pytanie")
        view.feed_text("po")
    out = console.export_text()
    assert out.index("przed") < out.index("pytanie") < out.index("po")
