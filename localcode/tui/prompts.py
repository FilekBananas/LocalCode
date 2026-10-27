"""Small interactive widgets built on prompt_toolkit.

Two shapes cover everything the TUI needs:

* :func:`choose` — the Claude Code permission question and every "pick one of
  these" list (models, sessions, quantisations). Arrow keys or the number keys,
  ``Enter`` accepts, ``Esc`` cancels.
* :func:`ask_text` — a one-line answer (used by the first-run wizard).

They render *inline*, below whatever rich has already printed, and erase
themselves when they are done, so the transcript above them never moves.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any, Sequence

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

__all__ = ["Choice", "CHOICE_STYLE", "ask_text", "choose", "matches_filter", "render_choices",
           "visible_window"]

#: Style of the inline pickers (kept close to the rich theme in ``render.py``).
CHOICE_STYLE = Style.from_dict(
    {
        "lc.question": "#ffcc66 bold",
        "lc.selected": "bold",
        "lc.marker": "#d97757 bold",
        "lc.option": "",
        "lc.dim": "#808080",
        "lc.hint": "#808080 italic",
    }
)


@dataclass
class Choice:
    """One row of :func:`choose`."""

    label: str
    value: Any = None
    hint: str = ""

    def resolved(self, index: int) -> Any:
        return self.value if self.value is not None else index


def _as_choices(options: Sequence[Choice | str | tuple[str, Any]]) -> list[Choice]:
    out: list[Choice] = []
    for option in options:
        if isinstance(option, Choice):
            out.append(option)
        elif isinstance(option, tuple):
            out.append(Choice(str(option[0]), option[1] if len(option) > 1 else None))
        else:
            out.append(Choice(str(option)))
    return out


def _clip(text: str, width: int | None) -> str:
    if width is None or width <= 0 or len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def visible_window(count: int, selected: int, max_rows: int | None) -> tuple[int, int]:
    """``(start, end)`` of the rows to draw so that ``selected`` stays on screen."""
    if max_rows is None or count <= max_rows:
        return 0, count
    rows = max(1, max_rows)
    start = min(max(0, selected - rows // 2), count - rows)
    return start, start + rows


def render_choices(choices: Sequence[Choice], selected: int, *, numbered: bool = True,
                   max_rows: int | None = None, width: int | None = None) -> StyleAndTextTuples:
    """The formatted-text fragments of the option list (also used by the tests).

    ``max_rows`` turns the list into a scrolling window around the selection:
    a list longer than the terminal used to push its own top off the screen,
    so the first models could be neither seen nor picked. ``width`` clips each
    row to one line for the same reason — a wrapped row is two rows tall.
    """
    fragments: StyleAndTextTuples = []
    start, end = visible_window(len(choices), selected, max_rows)
    if start > 0:
        fragments.append(("class:lc.dim", f"   ↑ jeszcze {start} wyżej\n"))
    for index in range(start, end):
        choice = choices[index]
        active = index == selected
        fragments.append(("class:lc.marker" if active else "", " ❯ " if active else "   "))
        prefix = f"{index + 1}. " if numbered and index < 9 else ""
        label = f"{prefix}{choice.label}"
        room = None if width is None else width - 3
        fragments.append(("class:lc.selected" if active else "class:lc.option", _clip(label, room)))
        if choice.hint and (room is None or len(label) + 4 < room):
            fragments.append(("class:lc.dim",
                              _clip(f"  {choice.hint}", None if room is None else room - len(label))))
        fragments.append(("", "\n"))
    if end < len(choices):
        fragments.append(("class:lc.dim", f"   ↓ jeszcze {len(choices) - end} niżej\n"))
    return fragments


def matches_filter(choice: Choice, needle: str) -> bool:
    """Every word of ``needle`` appears in the row (label or hint), any case."""
    haystack = f"{choice.label} {choice.hint}".lower()
    return all(word in haystack for word in needle.lower().split())


async def choose(
    options: Sequence[Choice | str | tuple[str, Any]],
    *,
    title: str = "",
    hint: str = "↑/↓ wybór · Enter zatwierdź · Esc anuluj",
    selected: int = 0,
    numbered: bool = True,
    escape_value: Any = None,
) -> Any:
    """Ask the user to pick one option; returns its value (``escape_value`` on Esc).

    ``Choice.value`` is returned when set, otherwise the option's index. A list
    taller than the terminal scrolls, and typing narrows it ("qwen 32").
    """
    choices = _as_choices(options)
    if not choices:
        return escape_value
    size = shutil.get_terminal_size((100, 30))
    # Title, hint and the two "more above / below" lines stay on screen too.
    max_rows = max(5, size.lines - 5 - (1 if title else 0))
    searchable = len(choices) > max_rows
    state: dict[str, Any] = {"index": max(0, min(selected, len(choices) - 1)), "filter": ""}

    def shown() -> list[int]:
        needle = state["filter"].strip()
        if not needle:
            return list(range(len(choices)))
        return [i for i, choice in enumerate(choices) if matches_filter(choice, needle)]

    def position() -> tuple[list[int], int]:
        rows = shown()
        if not rows:
            return rows, 0
        if state["index"] not in rows:
            state["index"] = rows[0]
        return rows, rows.index(state["index"])

    def get_text() -> AnyFormattedText:
        fragments: StyleAndTextTuples = []
        if title:
            fragments.append(("class:lc.question", title + "\n"))
        rows, at = position()
        if searchable:
            fragments.append(("class:lc.dim", "   szukaj: "))
            fragments.append(("class:lc.selected", state["filter"] or ""))
            fragments.append(("class:lc.dim", f"   ({len(rows)} z {len(choices)})\n"))
        if rows:
            subset = [choices[i] for i in rows]
            fragments.extend(render_choices(subset, at, numbered=numbered and not searchable,
                                            max_rows=max_rows - (1 if searchable else 0),
                                            width=size.columns))
        else:
            fragments.append(("class:lc.dim", "   nic nie pasuje — Backspace cofa\n"))
        if hint:
            extra = " · pisz, żeby zawęzić" if searchable else ""
            fragments.append(("class:lc.hint", hint + extra))
        return fragments

    keys = KeyBindings()

    def step(delta: int) -> None:
        rows, at = position()
        if rows:
            state["index"] = rows[(at + delta) % len(rows)]

    @keys.add("up")
    @keys.add("c-p")
    def _up(event: Any) -> None:
        step(-1)

    @keys.add("down")
    @keys.add("c-n")
    def _down(event: Any) -> None:
        step(1)

    @keys.add("pageup")
    def _page_up(event: Any) -> None:
        step(-max(1, max_rows - 1))

    @keys.add("pagedown")
    def _page_down(event: Any) -> None:
        step(max(1, max_rows - 1))

    @keys.add("home")
    def _home(event: Any) -> None:
        rows = shown()
        if rows:
            state["index"] = rows[0]

    @keys.add("end")
    def _end(event: Any) -> None:
        rows = shown()
        if rows:
            state["index"] = rows[-1]

    @keys.add("enter")
    @keys.add("c-m")
    def _accept(event: Any) -> None:
        rows, _ = position()
        if rows:
            event.app.exit(result=choices[state["index"]].resolved(state["index"]))

    @keys.add("escape", eager=True)
    @keys.add("c-c")
    @keys.add("c-d")
    def _cancel(event: Any) -> None:
        if searchable and state["filter"]:
            state["filter"] = ""
            return
        event.app.exit(result=escape_value)

    if searchable:
        @keys.add("<any>")
        def _type(event: Any) -> None:
            text = event.data or ""
            if text.isprintable():
                state["filter"] += text

        @keys.add("backspace")
        @keys.add("c-h")
        def _erase(event: Any) -> None:
            state["filter"] = state["filter"][:-1]
    else:
        @keys.add("k")
        def _k(event: Any) -> None:
            step(-1)

        @keys.add("j")
        def _j(event: Any) -> None:
            step(1)

        for number in range(1, min(len(choices), 9) + 1):
            @keys.add(str(number), eager=True)
            def _pick(event: Any, number: int = number) -> None:
                state["index"] = number - 1
                event.app.exit(result=choices[number - 1].resolved(number - 1))

    window = Window(FormattedTextControl(get_text, focusable=True), dont_extend_height=True,
                    always_hide_cursor=True)
    app: Application[Any] = Application(
        layout=Layout(HSplit([window])),
        key_bindings=keys,
        style=CHOICE_STYLE,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    return await app.run_async()


async def ask_text(message: str, *, default: str = "") -> str | None:
    """One-line text answer; ``None`` when the user pressed Esc or Ctrl+C."""
    from prompt_toolkit.shortcuts import PromptSession

    session: PromptSession[str] = PromptSession()
    keys = KeyBindings()

    @keys.add("escape", eager=True)
    def _cancel(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    try:
        return await session.prompt_async(message, default=default, key_bindings=keys)
    except (KeyboardInterrupt, EOFError):
        return None
