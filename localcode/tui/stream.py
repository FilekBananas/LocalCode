"""Streaming output of one agent turn.

``rich`` allows exactly one live display at a time, so a turn owns exactly one
:class:`rich.live.Live` — :class:`TurnView`.  Everything final (tool cards,
finished paragraphs, diffs, notices) is *printed* through it and scrolls away
normally; only the volatile part (a spinner, the collapsed reasoning counter,
the paragraph the model is still writing) lives in the live region, which is
erased when the turn ends.  That is what keeps very long answers readable: the
screen never redraws thousands of lines, and markdown is rendered paragraph by
paragraph as soon as a paragraph is complete.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from localcode.tui import render

__all__ = ["TurnView", "split_flushable"]

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
#: The live preview never grows beyond this many lines.
MAX_PREVIEW_LINES = 12


def split_flushable(text: str) -> tuple[str, str]:
    """Split ``text`` into "complete paragraphs" and "still being written".

    A blank line ends a paragraph — unless it is inside a fenced code block, in
    which case the whole block waits until its closing fence arrives.  The last
    line is never treated as final: more of it may still be coming.
    """
    if not text:
        return "", ""
    lines = text.split("\n")
    in_fence = False
    boundary = -1
    for index, line in enumerate(lines[:-1]):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence and not line.strip():
            boundary = index
    if boundary < 0:
        return "", text
    return "\n".join(lines[: boundary + 1]), "\n".join(lines[boundary + 1:])


class TurnView:
    """The single live display of a running turn."""

    def __init__(self, console: Console, *, show_reasoning: bool = False,
                 refresh_per_second: float = 10.0) -> None:
        self.console = console
        self.show_reasoning = show_reasoning
        self.refresh_per_second = refresh_per_second
        self._live: Live | None = None
        self._spinner = Spinner("dots", text=Text("", style="lc.dim"), style="lc.accent")
        self._status: str | None = None
        self._pending = ""          # assistant text not yet printed
        self._text = ""             # everything the assistant wrote this turn
        self._reasoning = ""
        self._reasoning_tokens = 0
        self._reasoning_open = False

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> "TurnView":
        self._live = Live(get_renderable=self._renderable, console=self.console,
                          refresh_per_second=self.refresh_per_second, transient=True,
                          vertical_overflow="crop")
        self._live.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()

    def finish(self) -> None:
        """Flush what is left and stop the live region."""
        self.end_assistant()
        live, self._live = self._live, None
        if live is not None:
            live.stop()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Give the terminal back (a prompt_toolkit widget needs the cursor)."""
        live = self._live
        if live is not None:
            self.end_assistant()
            live.stop()
        try:
            yield
        finally:
            if live is not None and self._live is live:
                live.start(refresh=True)

    # ---------------------------------------------------------------- output
    def print(self, renderable: RenderableType | None) -> None:
        """Print something final above the live region."""
        if renderable is None:
            return
        self.console.print(renderable)

    def set_status(self, message: str | None, phase: str = "info") -> None:
        self._status = message
        if message:
            self._spinner.update(text=Text(message, style="lc.dim"))
        self._refresh()

    # ------------------------------------------------------------- assistant
    def feed_text(self, delta: str) -> None:
        if not delta:
            return
        self._status = None
        self.close_reasoning()
        self._text += delta
        self._pending += delta
        done, rest = split_flushable(self._pending)
        if done.strip():
            self.print(Markdown(done.strip("\n")))
            self._pending = rest
        # the live region redraws itself `refresh_per_second` times a second;
        # refreshing per token would repaint the screen at the generation rate

    def end_assistant(self) -> None:
        """Print whatever is still pending as markdown (end of a step)."""
        self.close_reasoning()
        pending, self._pending = self._pending, ""
        if pending.strip():
            self.print(Markdown(pending.strip("\n")))

    @property
    def text(self) -> str:
        return self._text

    # ------------------------------------------------------------- reasoning
    def feed_reasoning(self, delta: str) -> None:
        if not delta:
            return
        self._status = None
        self._reasoning += delta
        self._reasoning_open = True
        self._reasoning_tokens = max(1, len(self._reasoning) // 4)

    def close_reasoning(self) -> None:
        """Collapse the reasoning block into its one-line summary."""
        if not self._reasoning_open:
            return
        self._reasoning_open = False
        tokens = self._reasoning_tokens
        if self.show_reasoning and self._reasoning.strip():
            self.print(render.reasoning_summary(tokens))
            self.print(Text(self._reasoning.strip(), style="lc.dim"))
        elif self._reasoning.strip():
            self.print(render.reasoning_summary(tokens))
        self._reasoning = ""
        self._reasoning_tokens = 0

    def reset_step(self) -> None:
        """A tool call interrupts the answer: flush text, keep the turn alive."""
        self.end_assistant()

    # ---------------------------------------------------------------- render
    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _renderable(self) -> RenderableType:
        rows: list[RenderableType] = []
        if self._reasoning_open:
            if self.show_reasoning:
                rows.append(Text(_tail(self._reasoning, MAX_PREVIEW_LINES), style="lc.dim"))
            else:
                rows.append(render.reasoning_summary(self._reasoning_tokens, streaming=True))
        if self._pending.strip():
            rows.append(Text(_tail(self._pending, MAX_PREVIEW_LINES)))
        if self._status:
            rows.append(self._spinner)
        if not rows:
            return Text("")
        return Group(*rows)


def _tail(text: str, lines: int) -> str:
    rows = text.splitlines()
    if len(rows) <= lines:
        return text.rstrip("\n")
    return "…\n" + "\n".join(rows[-lines:])


def make_console(**kwargs: Any) -> Console:
    """A console with the LocalCode theme (one place for tests to copy)."""
    kwargs.setdefault("theme", render.THEME)
    kwargs.setdefault("highlight", False)
    kwargs.setdefault("soft_wrap", False)
    return Console(**kwargs)
