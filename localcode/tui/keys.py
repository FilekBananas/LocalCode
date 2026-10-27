"""Reading the keyboard while a turn is running.

The prompt is gone once a message has been sent, but Esc and Ctrl+C must still
interrupt the agent.  :class:`InterruptWatcher` puts the terminal in raw mode
and attaches prompt_toolkit's input reader to the event loop for exactly that
period; output redirection is untouched, so ``rich`` keeps printing normally
(raw mode leaves ``ONLCR`` alone, so a bare ``\\n`` still moves to column 0).

It is a no-op when stdin is not a terminal (``localcode -p``, CI, pipes).
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any, Callable, Iterator

from prompt_toolkit.keys import Keys

log = logging.getLogger(__name__)

__all__ = ["InterruptWatcher"]

#: Keys that stop the running turn.
INTERRUPT_KEYS = frozenset({Keys.Escape, Keys.ControlC})


class InterruptWatcher:
    """Calls ``on_interrupt`` when the user presses Esc or Ctrl+C.

    Use it as a context manager around the event loop of one turn::

        with InterruptWatcher(agent_interrupt) as watcher:
            ...
            with watcher.paused():       # while another widget owns the keyboard
                answer = await choose(...)
    """

    def __init__(self, on_interrupt: Callable[[], None], *, input: Any = None) -> None:
        self.on_interrupt = on_interrupt
        self.presses = 0
        self._input = input
        self._stack: contextlib.ExitStack | None = None
        self._active = False

    # ------------------------------------------------------------- lifecycle
    def _make_input(self) -> Any:
        if self._input is not None:
            return self._input
        if not (sys.stdin and sys.stdin.isatty()):
            return None
        try:
            from prompt_toolkit.input import create_input

            self._input = create_input(always_prefer_tty=True)
        except Exception:  # noqa: BLE001 - no terminal, no interrupts
            log.debug("nie udało się otworzyć wejścia terminala", exc_info=True)
            self._input = None
        return self._input

    def start(self) -> None:
        if self._active:
            return
        source = self._make_input()
        if source is None:
            return
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(source.raw_mode())
            stack.enter_context(source.attach(self._on_ready))
        except Exception:  # noqa: BLE001 - never break a turn over key handling
            log.debug("nie udało się nasłuchiwać klawiatury", exc_info=True)
            stack.close()
            return
        self._stack = stack
        self._active = True

    def stop(self) -> None:
        stack, self._stack = self._stack, None
        self._active = False
        if stack is not None:
            with contextlib.suppress(Exception):
                stack.close()

    def __enter__(self) -> "InterruptWatcher":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Hand the keyboard to another widget (the permission prompt)."""
        was_active = self._active
        self.stop()
        try:
            yield
        finally:
            if was_active:
                self.start()

    # ------------------------------------------------------------------ keys
    def _on_ready(self) -> None:
        source = self._input
        if source is None:  # pragma: no cover - detached in the meantime
            return
        try:
            presses = list(source.read_keys())
            presses += list(source.flush_keys())
        except Exception:  # noqa: BLE001 - a closed terminal
            return
        for press in presses:
            if press.key in INTERRUPT_KEYS:
                self.presses += 1
                try:
                    self.on_interrupt()
                except Exception:  # noqa: BLE001 - reported by the caller instead
                    log.warning("obsługa przerwania nie powiodła się", exc_info=True)
                return
