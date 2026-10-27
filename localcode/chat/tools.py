"""The only tools the chatbot gets: the ones that manage models.

The chatbot is not an agent and does not want to be one — it has no file access,
no shell, and every schema it carries costs context on a model that may have
eight thousand tokens.  But there is one thing a person says to a chat window
and reasonably expects to happen: *"pobierz mi jakiś dobry model do kodowania"*.
Without tools the answer was a well-written essay about CodeLlama and nothing
downloaded, which is worse than saying "I can't".

So the chat gets exactly four tools — search, list, install, load
(:mod:`localcode.agent.tools.models`) — and only when the setting is on and the
model's template can call tools at all.  Everything else it does with words.

The loop is deliberately short: at most :data:`MAX_ROUNDS` tool rounds per
answer.  A chatbot that keeps calling tools instead of answering is a bug, not a
feature.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterable

from localcode.agent.tools.base import ToolContext, ToolLimits
from localcode.agent.tools.models import MODEL_TOOL_CLASSES, STATE_KEY

log = logging.getLogger(__name__)

#: How many times one answer may stop to use a tool.
MAX_ROUNDS = 3
#: Tool output is fed back to a small model, so it stays small.
MAX_TOOL_CHARS = 2_000


class ChatToolRunner:
    """Model-management tools for one chat turn."""

    def __init__(self, ops: Any, *, cwd: Any = None, enabled: bool = True,
                 max_rounds: int = MAX_ROUNDS) -> None:
        from pathlib import Path

        self.ops = ops
        self.enabled = bool(enabled and ops is not None)
        self.max_rounds = max_rounds
        self._tools = {cls.name: cls() for cls in MODEL_TOOL_CLASSES}
        self._ctx = ToolContext(cwd=Path(cwd or "."), limits=ToolLimits(), state={STATE_KEY: ops})

    # what the model is told it can do ----------------------------------------
    def schemas(self) -> list[dict[str, Any]] | None:
        """Tool schemas for the prompt, or ``None`` when the chat has no tools."""
        if not self.enabled:
            return None
        return [tool.schema() for tool in self._tools.values()]

    def supported_by(self, engine: Any) -> bool:
        """True when this model's template can be given tools at all."""
        info = getattr(engine, "template_info", None)
        return bool(self.enabled and info is not None and info.supports_tools)

    # what the model actually asked for ---------------------------------------
    def parse(self, text: str, engine: Any) -> list[Any]:
        """Tool calls in a finished answer (empty when there are none)."""
        if not self.enabled or not text:
            return []
        from localcode.runtime.toolcalls import parse_output

        info = getattr(engine, "template_info", None)
        fmt = getattr(info, "tool_format", None)
        try:
            parsed = parse_output(text, tool_format=fmt, known_tools=self.schemas() or None)
        except Exception:  # noqa: BLE001 - a malformed call must not kill the turn
            log.debug("could not parse tool calls out of a chat answer", exc_info=True)
            return []
        return [call for call in parsed.tool_calls if call.name in self._tools]

    def content_without_calls(self, text: str, engine: Any) -> str:
        """The answer with the tool-call blocks removed (what the user reads)."""
        if not self.enabled or not text:
            return text
        from localcode.runtime.toolcalls import parse_output

        info = getattr(engine, "template_info", None)
        try:
            return parse_output(text, tool_format=getattr(info, "tool_format", None)).content
        except Exception:  # noqa: BLE001
            return text

    # running them -------------------------------------------------------------
    async def run(self, calls: Iterable[Any]) -> AsyncIterator[dict[str, Any]]:
        """Execute calls, yielding one frame per call for the UI and the history.

        Each frame carries ``message`` — the ``role: "tool"`` entry that goes
        back to the model — plus the fields the app needs to draw a row.
        """
        for call in calls:
            tool = self._tools.get(call.name)
            if tool is None:
                continue
            yield {"type": "tool", "state": "running", "name": call.name,
                   "title": tool.title(call.arguments or {})}
            try:
                result = await tool.run(call.arguments or {}, self._ctx)
                output, summary, ok = result.output, result.summary, result.ok
            except Exception as exc:  # noqa: BLE001 - the model gets to read the failure
                output, summary, ok = f"Narzędzie zgłosiło błąd: {exc}", "błąd narzędzia", False
            if len(output) > MAX_TOOL_CHARS:
                output = output[:MAX_TOOL_CHARS].rstrip() + "\n[…]"
            yield {
                "type": "tool", "state": "done", "name": call.name, "ok": ok,
                "title": tool.title(call.arguments or {}), "summary": summary,
                "message": {"role": "tool", "tool_call_id": call.id, "name": call.name,
                            "content": output},
            }

    @staticmethod
    def assistant_message(text: str, calls: Iterable[Any]) -> dict[str, Any]:
        """The assistant turn that asked for the tools, for the next prompt."""
        return {
            "role": "assistant",
            "content": text or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name,
                                         "arguments": json.dumps(c.arguments or {},
                                                                 ensure_ascii=False)}}
                           for c in calls],
        }
