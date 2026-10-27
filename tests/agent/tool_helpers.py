"""Helpers shared by the agent tool tests (``tests/agent/test_tools_*.py``).

pytest puts each test directory on ``sys.path``, so these are imported as
``from tool_helpers import ...`` (the same trick ``tests/chat`` uses).
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

from localcode.agent.tools import get_tool
from localcode.agent.tools.base import ToolContext, ToolLimits, ToolResult


def make_ctx(root: Path, **limits: Any) -> ToolContext:
    """A tool context rooted at ``root``; keyword arguments override the limits."""
    return ToolContext(cwd=Path(root), limits=ToolLimits(**limits))


def write_file(path: Path, text: str, *, newline: str = "\n", encoding: str = "utf-8") -> Path:
    """Write ``text`` verbatim (no newline translation surprises) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.replace("\n", newline) if newline != "\n" else text
    path.write_bytes(data.encode(encoding))
    return path


async def run_tool(name: str, args: Any, ctx: ToolContext) -> ToolResult:
    """Run the registered tool ``name`` the way the agent loop would."""
    tool = get_tool(name)
    assert tool is not None, f"unknown tool {name!r}"
    return await tool.run(args, ctx)


async def permission_of(name: str, args: Any, ctx: ToolContext):
    tool = get_tool(name)
    assert tool is not None, f"unknown tool {name!r}"
    return await tool.permission(args, ctx)


class Progress:
    """Recording ``ctx.progress`` callback."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    async def __call__(self, line: str) -> None:
        self.lines.append(line)


def numbered(output: str) -> list[tuple[int, str]]:
    """``[(line number, text), …]`` parsed out of a ``read_file`` style block."""
    out: list[tuple[int, str]] = []
    for line in output.splitlines():
        head, tab, rest = line.partition("\t")
        if tab and head.strip().isdigit():
            out.append((int(head.strip()), rest))
    return out


def pid_alive(pid: int) -> bool:
    """True while ``pid`` exists (a zombie counts as alive until it is reaped)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # pragma: no cover - not ours, but it exists
        return True
    return True


async def wait_until_gone(pid: int, timeout: float = 5.0) -> bool:
    """Poll until ``pid`` disappears; returns False if it is still there."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not pid_alive(pid):
            return True
        await asyncio.sleep(0.05)
    return not pid_alive(pid)


def kill_pid(pid: int) -> None:
    """Best-effort cleanup for a test that left something running."""
    with_suppress = (ProcessLookupError, PermissionError, OSError)
    try:
        os.kill(pid, signal.SIGKILL)
    except with_suppress:
        pass
