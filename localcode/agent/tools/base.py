"""Tool contract for the coding agent.

Tools are deliberately few and small: every tool schema is paid for in context
tokens on every request, and small local models pick the right tool more
reliably from a short menu.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

# "read" tools never need permission; "write"/"exec" depend on the permission mode.
ToolKind = Literal["read", "write", "exec", "meta"]


@dataclass
class ToolLimits:
    """Output limits derived from the model's context size (see agent/context.py)."""

    max_output_chars: int = 12_000      # hard cap for any single tool result fed to the model
    read_default_lines: int = 200       # read_file default window
    grep_max_matches: int = 60
    list_max_entries: int = 200
    bash_timeout_s: int = 120


@dataclass
class ToolContext:
    cwd: Path                                            # project root; relative paths resolve here
    limits: ToolLimits = field(default_factory=ToolLimits)
    # set by the agent loop: files read in this session -> mtime, used to warn on stale edits
    read_files: dict[str, float] = field(default_factory=dict)
    # optional callback for long-running tools to stream progress lines to the UI
    progress: Callable[[str], Awaitable[None]] | None = None
    # shared agent state (todo list lives here)
    state: dict[str, Any] = field(default_factory=dict)
    # the tool call this run belongs to, when the loop runs several at once
    # (ultracode's parallel ``delegate``); ``state["current_call_id"]`` otherwise
    call_id: str | None = None


@dataclass
class ToolResult:
    ok: bool
    output: str                       # text for the MODEL (already truncated to limits)
    summary: str                      # one line for the UI
    display: str | None = None        # optional richer text for the UI (defaults to output)
    diff: str | None = None           # unified diff for file changes


@dataclass
class PermissionInfo:
    title: str                        # "Edytuj src/app.py", "Uruchom: npm test"
    preview: str                      # diff or command


class Tool(abc.ABC):
    name: str = ""
    kind: ToolKind = "read"
    description: str = ""             # ONE or two short sentences; schemas cost tokens
    parameters: dict[str, Any] = {}   # JSON schema (type=object)

    def schema(self) -> dict[str, Any]:
        """OpenAI-style tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def title(self, args: dict[str, Any]) -> str:
        """Short label for the UI, e.g. ``Read src/app.py``."""
        return self.name

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        """Return what the user must approve, or None if no approval is needed."""
        return None

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...
