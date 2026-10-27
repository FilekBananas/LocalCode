"""The LocalCode terminal application (prompt_toolkit + rich).

Layout of the package::

    backends.py   one interface, two implementations: everything in this
                  process (``LocalBackend``) or attached to a running
                  ``localcode serve`` (``RemoteBackend``)
    app.py        the interactive REPL: prompt, transcript, slash commands
    headless.py   ``localcode -p`` — one turn, machine-readable output
    chatmode.py   the plain chatbot (``localcode chat`` / ``/chat``)
    render.py     every rich renderable (tool cards, diffs, gauges, tables)
    stream.py     the single live display of a running turn
    commands.py   the slash-command table, ``/help`` and ``/runtime set``
    completion.py ``/`` and ``@`` completion plus ``@file`` expansion
    prompts.py    inline pickers (permissions, models, sessions)
    keys.py       Esc / Ctrl+C while the agent is working
    models_ui.py  choosing, downloading and inspecting models

Imports are kept lazy here so ``import localcode.tui`` stays cheap for the CLI,
which only needs one of these modules per command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["LocalCodeTui", "make_console", "open_backend", "run_headless", "run_tui"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from localcode.tui.app import LocalCodeTui, run_tui
    from localcode.tui.backends import open_backend
    from localcode.tui.headless import run_headless
    from localcode.tui.stream import make_console

_LAZY = {
    "LocalCodeTui": "localcode.tui.app",
    "run_tui": "localcode.tui.app",
    "open_backend": "localcode.tui.backends",
    "run_headless": "localcode.tui.headless",
    "make_console": "localcode.tui.stream",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)
