"""Fixtures for the terminal-app tests.

``tests/server`` and ``tests/chat`` are put on ``sys.path`` so the live-server
test can reuse the HTTP layer's fakes (a real ``Core`` with a fake library,
runtime and agent) instead of inventing a second set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from rich.console import Console

from localcode.tui.render import THEME

_HERE = Path(__file__).resolve().parent
for _sibling in ("server", "chat"):
    _path = _HERE.parent / _sibling
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tui_fakes import ScriptedBackend  # noqa: E402


@pytest.fixture
def console() -> Console:
    """A recording console with the app's theme, a wide page and no colours."""
    return Console(record=True, width=100, color_system=None, theme=THEME, highlight=False,
                   legacy_windows=False, force_terminal=False)


@pytest.fixture
def backend(tmp_path: Path) -> ScriptedBackend:
    return ScriptedBackend(tmp_path)
