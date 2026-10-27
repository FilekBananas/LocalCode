"""``goal_done``: how the agent says a ``/goal`` is reached.

``/goal <cel>`` keeps the agent working across steps until the goal is met —
after every final answer the loop checks whether the goal was declared done
and, if not, sends the model back to work (``Agent._goal_continue``). The model
needs one unambiguous way to end that loop, and a tool call is the signal small
models produce most reliably: a phrase in the answer gets paraphrased, a tool
name does not.

``blocked=true`` is the other honest ending: the goal needs a decision, a key,
or a file only the user has. Without it the only way out would be to claim
success, or to burn every remaining turn trying.

The tool is offered only while a goal is active (see ``Agent._prepare_tools``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from localcode.agent.tools.base import ToolContext, ToolResult
from localcode.agent.tools.fs import CodingTool, short

__all__ = ["GOAL_STATES", "STATE_KEY", "GoalDoneTool", "GoalState", "call_in_text", "notice_text"]

STATE_KEY = "goal"
#: ``active`` → the loop continues; the others end it. ``stalled``: the model
#: kept giving the same answer without doing anything.
GOAL_STATES = ("active", "done", "blocked", "limit", "stalled", "cleared")

#: ``goal_done{"summary": …}`` / ``goal_done(summary="…", blocked=true)`` written
#: as plain text. Small models know they should call it and then *write* the
#: call into the answer; a real Qwen3-4B run burned all twenty continuations
#: re-declaring a goal it had reached after four steps.
#: Only call syntax counts — "goal_done" right before "{" or "(" — so a sentence
#: that merely mentions the tool ("wywołam goal_done, gdy…") declares nothing.
_TEXT_CALL = re.compile(r"\bgoal_done\s*(?P<open>[({])")


@dataclass
class GoalState:
    """The session's goal and how far the loop has gone with it."""

    text: str
    max_turns: int = 20
    turns: int = 0               # automatic continuations so far
    state: str = "active"
    summary: str = ""
    #: The "reached" / "blocked" notice went out (it is sent once).
    announced: bool = False
    #: The last automatic continuation's answer, and how many times in a row
    #: the model answered the same without calling a tool.
    last_answer: str = ""
    repeats: int = 0

    @property
    def active(self) -> bool:
        return self.state == "active"

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "state": self.state, "turns": self.turns,
                "maxTurns": self.max_turns, "summary": self.summary}


def call_in_text(text: str) -> tuple[str, bool] | None:
    """``(summary, blocked)`` when an answer *writes* a goal_done call instead of making it."""
    text = text or ""
    match = _TEXT_CALL.search(text)
    if match is None:
        return None
    rest = text[match.start("open"):]
    args: dict[str, Any] = {}
    brace = rest.find("{")
    if brace != -1 and rest[:brace].strip() in ("", "("):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(rest[brace:])
            if isinstance(parsed, dict):
                args = parsed
        except ValueError:
            args = {}
    if not args and rest.startswith("("):
        body = rest[1:rest.find(")")] if ")" in rest else rest[1:]
        for key, quoted, single, bare in re.findall(
                r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([\w.]+))', body):
            args[key] = quoted or single or bare
        if not args and body.strip():
            args["summary"] = body.strip().strip("\"'")
    summary = str(args.get("summary") or args.get("result") or args.get("message") or "").strip()
    return summary, _truthy(args.get("blocked") or args.get("stuck"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "tak", "y")


class GoalDoneTool(CodingTool):
    name = "goal_done"
    kind = "meta"
    description = ("Call ONLY when the active goal is fully achieved and verified "
                   "(or blocked=true when it cannot be finished without the user). "
                   "Ends the goal loop.")
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string",
                        "description": "What was done and how it was verified (1-3 sentences)"},
            "blocked": {"type": "boolean",
                        "description": "true = cannot finish without the user; say what is needed"},
        },
        "required": ["summary"],
    }
    aliases = {"result": "summary", "message": "summary", "text": "summary", "reason": "summary",
               "stuck": "blocked"}
    primary = "summary"
    usage = 'goal_done({"summary": "all tests pass; added the endpoint and its docs"})'

    def make_title(self, args: dict[str, Any]) -> str:
        return "Cel wymaga Twojej decyzji" if _truthy(args.get("blocked")) else "Cel osiągnięty"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        goal = ctx.state.get(STATE_KEY)
        if not isinstance(goal, GoalState) or not goal.active:
            return ToolResult(ok=False, output="There is no active goal; just answer the user.",
                              summary="Brak aktywnego celu")
        summary = str(args.get("summary") or "").strip()
        blocked = _truthy(args.get("blocked"))
        goal.state = "blocked" if blocked else "done"
        goal.summary = summary
        if blocked:
            return ToolResult(ok=True,
                              output="Goal paused. Tell the user exactly what you need from them.",
                              summary=short(summary, 120) or "Potrzebna decyzja użytkownika")
        return ToolResult(ok=True,
                          output="Goal marked as achieved. Give the user a short final summary.",
                          summary=short(summary, 120) or "Cel osiągnięty")


def notice_text(payload: dict[str, Any]) -> str:
    """One Polish line for a ``goal`` event — the app's notice and the terminal's line."""
    text = str(payload.get("text") or "")
    state = str(payload.get("state") or "")
    turns, limit = int(payload.get("turns") or 0), int(payload.get("maxTurns") or 0)
    summary = str(payload.get("summary") or "").strip()
    if state == "active":
        return f"Cel: {text}" if turns == 0 else f"Kontynuuję pracę nad celem ({turns}/{limit})."
    if state == "done":
        return "Cel osiągnięty" + (f": {summary}" if summary else ".")
    if state == "blocked":
        return "Cel czeka na Twoją decyzję" + (f": {summary}" if summary else ".")
    if state == "limit":
        return f"Cel nieosiągnięty po {limit} podejściach — napisz, co dalej, albo /goal wyczyść."
    if state == "stalled":
        return "Agent stoi w miejscu — powtarza tę samą odpowiedź. Napisz, co dalej, albo /goal wyczyść."
    if state == "cleared":
        return "Cel usunięty."
    return text
