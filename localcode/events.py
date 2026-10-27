"""Events emitted by the coding agent (and consumed by the TUI, the server's
WebSocket and the Swift app). Each event serialises to a flat JSON object with a
``type`` field; keys are camelCase so the Swift side can decode them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Event:
    type: ClassVar[str] = "event"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        for k, v in self.__dict__.items():
            parts = k.split("_")
            out[parts[0] + "".join(p.title() for p in parts[1:])] = v
        return out


@dataclass
class TurnStart(Event):
    type: ClassVar[str] = "turn_start"
    turn_id: str
    user_text: str


@dataclass
class UserMessage(Event):
    """A message the user typed while the turn was running, handed to the agent
    between two steps (it is in the history from here on)."""

    type: ClassVar[str] = "user_message"
    text: str


@dataclass
class GoalUpdate(Event):
    """``/goal``: set, continued after a final answer, reached, blocked or cleared.

    ``state`` is ``active`` (set, or the loop sent the model back to work —
    ``turns`` counts those), ``done``, ``blocked`` (needs the user), ``limit``
    (``max_turns`` used up), ``stalled`` (the same answer, no tools, again and
    again) or ``cleared``. ``text`` is the goal itself.
    """

    type: ClassVar[str] = "goal"
    text: str
    state: str
    turns: int = 0
    max_turns: int = 0
    summary: str = ""


@dataclass
class TextDelta(Event):
    type: ClassVar[str] = "text_delta"
    text: str


@dataclass
class ReasoningDelta(Event):
    type: ClassVar[str] = "reasoning_delta"
    text: str


@dataclass
class ToolCallEvent(Event):
    type: ClassVar[str] = "tool_call"
    call_id: str
    name: str
    arguments: dict[str, Any]
    title: str                     # short human label, e.g. "Read src/app.py (1-200)"


@dataclass
class ToolResultEvent(Event):
    type: ClassVar[str] = "tool_result"
    call_id: str
    name: str
    ok: bool
    summary: str                   # one line, e.g. "200 linii" / "exit 0"
    output: str                    # what the UI shows (already truncated for display)
    diff: str | None = None        # unified diff for edits/writes


# --------------------------------------------------------------------------- #
# A helper agent working on one delegated task
# --------------------------------------------------------------------------- #
# A helper's own events are never forwarded raw: its ``turn_end`` would end the
# parent's turn in both clients, and its tool calls would land in the parent's
# transcript as if the parent had made them. These three wrap it instead, and
# every one of them carries ``parent_call_id`` — the ``delegate`` call they
# belong to — so a client can nest them under that card.


@dataclass
class SubagentStart(Event):
    type: ClassVar[str] = "subagent_start"
    agent_id: str                  # short name, "h2" — also on its permission cards
    parent_call_id: str
    title: str                     # the delegated task, shortened
    model: str
    remote: bool = False           # the helper runs behind somebody's API key
    #: ``delegate(wait=false)``: the call already returned and the agent works
    #: on; the report reaches it as a note when the helper finishes.
    background: bool = False


@dataclass
class SubagentStep(Event):
    type: ClassVar[str] = "subagent_step"
    agent_id: str
    parent_call_id: str
    name: str                      # the tool the helper used
    title: str
    ok: bool
    summary: str
    #: One per finished tool call, never per token: a helper doing a dozen reads
    #: is a dozen events, which a client can show without flooding anything.


@dataclass
class SubagentEnd(Event):
    type: ClassVar[str] = "subagent_end"
    agent_id: str
    parent_call_id: str
    ok: bool
    steps: int
    elapsed_ms: int
    message: str = ""              # why it stopped, when it did not finish
    background: bool = False
    #: A background helper's report (what the agent is handed at its next
    #: step), so the card can show it; empty for a helper the call waited for.
    report: str = ""


#: What an approver answers when nobody decided — no client attached, the turn
#: stopped, the session closed, a headless run. The call does not run, exactly
#: as after "deny", but it was not the user's answer, so nothing is learned
#: from it (``agent/precedents.py``).
PERMISSION_DISMISSED = "dismissed"


@dataclass
class PermissionRequest(Event):
    type: ClassVar[str] = "permission_request"
    request_id: str
    tool: str
    title: str
    preview: str                   # command to run / diff to apply
    options: list[str] = field(default_factory=lambda: ["allow", "deny", "always"])
    #: Auto mode: what the reviewer said before the user is asked —
    #: ``{verdict: "deny"|"allow"|"ask", reason, model}``. ``deny``: it said NIE
    #: and the user may overrule it; ``allow``: only an opinion (a call that
    #: needs a human, which a TAK cannot approve); ``ask``: no clear answer.
    #: ``reason`` starts with "Co robi: …" when the reviewer said what the
    #: action does. ``guard`` (optional, same shape) is the context-free local
    #: guard's opinion. Left out of the JSON when there was no review.
    review: dict[str, Any] | None = None
    #: The helper agent asking ("h2"; the title then starts "Pomocnik h2: "),
    #: empty for the agent itself. Left out of the JSON when empty.
    helper: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if out.get("review") is None:
            out.pop("review", None)
        if not out.get("helper"):
            out.pop("helper", None)
        return out


@dataclass
class AutoReview(Event):
    """Auto mode: what the reviewer model said about one tool call, and why.

    ``verdict`` is ``allow`` (the call runs), ``deny`` (the user is asked right
    after with a ``permission_request`` carrying this review, and may overrule
    it; with nobody to ask the call is refused and the agent gets ``reason``)
    or ``ask`` (no usable answer — the user is asked the same way, and
    ``reason`` says why).
    ``reason`` is Polish and always shown. The same three fields are stored on
    the tool's display item as ``review: {verdict, reason, model}``.
    """

    type: ClassVar[str] = "auto_review"
    call_id: str
    tool: str
    verdict: str
    reason: str
    reviewer_model: str = ""       # model key of the reviewer that answered
    #: The context-free local guard (``agent/guard.py``) on the same call:
    #: ``{verdict, reason, model}``, or ``None`` when the call is not one it
    #: checks (not a shell command, not outside the project). The call runs on
    #: its own only when both said ``allow``. Left out of the JSON when None.
    guard: dict[str, Any] | None = None
    #: The helper agent whose call this was ("h2"); empty for the agent itself.
    helper: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = super().to_dict()
        if out.get("guard") is None:
            out.pop("guard", None)
        if not out.get("helper"):
            out.pop("helper", None)
        return out


@dataclass
class ContextUpdate(Event):
    type: ClassVar[str] = "context"
    used: int                      # tokens in the next prompt
    limit: int                     # usable context per slot
    system: int = 0
    tools: int = 0
    history: int = 0
    reserve: int = 0               # kept free for the answer


@dataclass
class Compaction(Event):
    type: ClassVar[str] = "compaction"
    before: int
    after: int
    strategy: str                  # "mask_tool_results" | "summarize" | "drop_oldest"


@dataclass
class Status(Event):
    type: ClassVar[str] = "status"
    message: str                   # "Ładowanie modelu…", "Myślę…"
    phase: str = "info"            # loading | thinking | tool | info


@dataclass
class TodoUpdate(Event):
    type: ClassVar[str] = "todos"
    items: list[dict[str, Any]]    # [{content, status: pending|in_progress|done}]


@dataclass
class TurnEnd(Event):
    type: ClassVar[str] = "turn_end"
    turn_id: str
    reason: str                    # "done" | "interrupted" | "max_steps" | "error"
    stats: dict[str, Any] = field(default_factory=dict)  # GenStats.to_dict() totals + steps


@dataclass
class ErrorEvent(Event):
    type: ClassVar[str] = "error"
    message: str
