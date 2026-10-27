"""The inline picker: long lists scroll and can be narrowed by typing.

Regression: with the API-key providers' models in it, the `/model` list grew
past the terminal's height and pushed its own first rows off the screen — the
local models could be neither seen nor picked.
"""

from __future__ import annotations

import asyncio
from unittest import mock

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localcode.tui.prompts import Choice, choose, matches_filter, render_choices, visible_window


def text_of(fragments) -> str:
    return "".join(part[1] for part in fragments)


def test_a_short_list_is_drawn_whole() -> None:
    assert visible_window(5, 2, 10) == (0, 5)


def test_the_window_follows_the_selection() -> None:
    assert visible_window(100, 0, 10) == (0, 10)
    assert visible_window(100, 50, 10) == (45, 55)
    assert visible_window(100, 99, 10) == (90, 100)


def test_a_long_list_says_how_much_is_hidden() -> None:
    choices = [Choice(f"model {n}") for n in range(40)]
    out = text_of(render_choices(choices, 20, max_rows=10))
    assert "↑ jeszcze 15 wyżej" in out
    assert "↓ jeszcze 15 niżej" in out
    assert "model 20" in out and "model 0" not in out
    assert out.count("\n") == 12


def test_rows_are_clipped_to_one_line() -> None:
    choices = [Choice("x" * 300, hint="y" * 300)]
    for line in text_of(render_choices(choices, 0, width=80)).splitlines():
        assert len(line) <= 80


def test_the_filter_matches_every_word_anywhere() -> None:
    row = Choice("Qwen3 32B · Q4_K_M", hint="18,6 GB · GGUF")
    assert matches_filter(row, "qwen 32")
    assert matches_filter(row, "gguf")
    assert not matches_filter(row, "qwen 70")


def run_choose(keys: str, choices: list[Choice], *, lines: int = 12) -> object:
    async def go() -> object:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            return await choose(choices, title="Wybierz model")

    size = mock.Mock(lines=lines, columns=100)
    with mock.patch("localcode.tui.prompts.shutil.get_terminal_size", return_value=size):
        return asyncio.run(go())


def test_typing_narrows_a_long_list_and_enter_picks_the_match() -> None:
    choices = [Choice(f"remote model {n} (DeepInfra)", value=f"r{n}") for n in range(60)]
    choices.insert(0, Choice("Qwen3 4B Instruct", value="local"))
    assert run_choose("qwen\r", choices) == "local"


def test_arrows_reach_rows_below_the_window() -> None:
    choices = [Choice(f"model {n}", value=n) for n in range(60)]
    down = "\x1b[B" * 30
    assert run_choose(down + "\r", choices) == 30


def test_a_short_list_keeps_number_keys() -> None:
    choices = [Choice(f"model {n}", value=n) for n in range(4)]
    assert run_choose("3", choices, lines=40) == 2
