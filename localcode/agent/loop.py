"""The coding-agent loop: one user turn → model steps → tool calls → answer.

``Agent`` is the piece that makes LocalCode feel like Claude Code. It owns the
model history of one session and drives, for every user turn, the classic
agent loop::

    TurnStart
      ├─ step 1: acquire a slot → fit the context → stream → parse
      │           ├─ no tool calls  → done
      │           └─ tool calls     → permission → run → tool results
      ├─ step 2: …
      └─ TurnEnd(done | interrupted | max_steps | error)

Design decisions that matter on a 4B model with a 4k–8k window:

* **The lease is held only while the model generates.** Tools (a ``pytest``
  run can take a minute) run *after* the slot is released, so other parts of
  the app — the chatbot, a second agent — are not blocked by our tool calls.
  ``RuntimeManager`` loads the model just in time on the first step; a
  ``Status('Ładowanie modelu…')`` event is emitted when it is not loaded yet.
* **The prompt prefix never moves.** The system prompt is built once per
  session (from ``prompts.build_system_prompt``) and reused byte-identically on
  every step, history is append-only, and ``ContextManager.fit`` only rewrites
  it when it must compact. That keeps the engine's KV prefix cache warm, which
  is the difference between a 2-second and a 20-second step.
* **Everything the model writes is fed back.** Unknown tools, unparsable calls,
  denied permissions, repeated calls and empty replies all produce a precise,
  English, corrective message in the transcript instead of silently ending the
  turn — small models recover from those surprisingly well.
* **Nothing is lost on interrupt.** ``interrupt()`` cancels the generation or
  the running tool, keeps the partial text and closes every issued tool call
  with ``[interrupted by user]`` so the stored history stays a valid
  conversation that the model can continue from.

Events (``localcode.events``) are pushed to ``on_event`` as dataclass
instances; ``event.to_dict()`` is the camelCase JSON the WebSocket and the
Swift app expect (``dict_events()`` wraps a dict-consuming callback). The
callback may be sync or async. Permission questions go to ``ask_permission``,
which returns ``"allow" | "deny" | "always"`` (or ``"dismissed"`` when nobody
answered: refused like "deny", but not learned from — ``agent/precedents.py``). In ``auto`` mode a reviewer model
answers them first (``agent/autoreview.py``, event ``AutoReview``); only what
it cannot settle, and what must never be a model's call, reaches the user.

Session persistence is optional: pass ``session=SessionStore.create(...)`` (or
a resumed one, whose messages are adopted) and every message, display item and
history rewrite is appended to its JSONL file.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import logging
import os
import secrets
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from localcode.agent import autoreview, guard, precedents, prompts
from localcode.agent.context import ContextManager
from localcode.agent.permissions import OUTSIDE_PROJECT_FLAG, PermissionPolicy, normalize_mode
from localcode.agent.profile import LOCAL as LOCAL_PROFILE
from localcode.agent.profile import profile_for
from localcode.agent.session import AgentSession, SessionStore, now_ms
from localcode.agent import langskills
from localcode.agent.skills import SkillLibrary
from localcode.agent.tools.goal import STATE_KEY as GOAL_STATE_KEY
from localcode.agent.tools.goal import GoalDoneTool, GoalState, call_in_text
from localcode.agent.tools.shell import CONFINE_STATE_KEY, command_from_args
from localcode.agent.tools.skill import STATE_KEY as SKILLS_STATE_KEY
from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolLimits, ToolResult
from localcode.config import Settings
from localcode.events import (
    PERMISSION_DISMISSED,
    AutoReview,
    Compaction,
    ContextUpdate,
    ErrorEvent,
    Event,
    GoalUpdate,
    PermissionRequest,
    ReasoningDelta,
    Status,
    TextDelta,
    TodoUpdate,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
    TurnStart,
    UserMessage,
)
from localcode.runtime.base import EngineError, GenStats, SamplingParams
from localcode.runtime.degenerate import LoopGuard
from localcode.runtime.formatter import ChatFormatter
from localcode.runtime.templates import ULTRACODE, engine_effort, normalize_agent_effort, normalize_effort
from localcode.runtime.toolcalls import ParsedOutput, ToolCall, format_hint

log = logging.getLogger(__name__)

__all__ = [
    "Agent",
    "AgentBusyError",
    "INTERRUPTED_RESULT",
    "dict_events",
]

#: Tool result written for calls that never ran because the user interrupted.
INTERRUPTED_RESULT = "[interrupted by user]"

_DENIED_RESULT = (
    "The user denied this tool call. Do not retry it. Choose a different approach, "
    "or ask the user what to do instead."
)
_PLAN_MODE_RESULT = (
    "Error: plan mode is read-only, so {name} cannot be used. Do not retry it. "
    "Finish investigating with the read-only tools and reply with a plan."
)
_NO_APPROVER_RESULT = (
    "This tool call needs the user's approval, but this session cannot ask for it. "
    "Do not retry it; continue with the tools that do not need approval."
)
_UNKNOWN_TOOL_RESULT = (
    "Error: there is no tool named {name!r}. Available tools: {available}. "
    "Call one of them with the arguments from its schema."
)
_BAD_ARGS_RESULT = (
    "Error: the arguments of {name} must be a JSON object, got {got}. "
    "Call the tool again with a JSON object of arguments."
)

_MAX_MALFORMED_RETRIES = 2
_REPEAT_LIMIT = 3
#: Ultracode turns ``delegate`` on even when Settings has it off — but not
#: below this window: there the helpers' reports would crowd out the work they
#: were meant to spare, and the small prompt tiers have no room to orchestrate.
ULTRACODE_MIN_CTX = 16384
#: Ultracode: after this many steps in a row with tool calls but no
#: ``delegate``, the agent is reminded once that helpers exist — and not again
#: before as many more such steps (at most :data:`ULTRACODE_MAX_NUDGES` a turn).
ULTRACODE_NUDGE_STEPS = 4
ULTRACODE_MAX_NUDGES = 5
_TODO_TOOLS = frozenset({"todo_write", "todo", "todos", "update_todos"})
_TODO_STATUS = {
    "done": "done", "completed": "done", "complete": "done", "finished": "done",
    "in_progress": "in_progress", "in-progress": "in_progress", "active": "in_progress",
    "doing": "in_progress", "pending": "pending", "todo": "pending", "open": "pending",
    "cancelled": "cancelled", "canceled": "cancelled",
}

_INTERJECTION = ("[The user sent this while you were working. Take it into account from now on; "
                 "if it changes the task, change course.]\n{text}")
_ERR_MAX_STEPS = (
    "Agent wykonał maksymalną liczbę kroków ({n}) bez zakończenia zadania. "
    "Napisz, co zrobić dalej, albo zwiększ limit: Ustawienia → Agent → "
    "Maks. liczba kroków agenta (0 = bez limitu)."
)
_ERR_OVERFLOW = (
    "Rozmowa nie mieści się w oknie kontekstu modelu ({n_ctx} tokenów) nawet po kompaktowaniu. "
    "Użyj /compact, zacznij nową sesję albo wczytaj model z większym kontekstem."
)
_ERR_MALFORMED = (
    "Model nie potrafi poprawnie sformułować wywołania narzędzia. "
    "Spróbuj ponownie, uprość polecenie albo użyj większego modelu."
)
_ERR_EMPTY = (
    "Model zwrócił pustą odpowiedź. Spróbuj ponownie lub użyj innego modelu."
)
_ERR_LOOPING = (
    "Model się zapętlił — powtarzał „{fragment}” i nie napisał odpowiedzi, więc "
    "przerwałem generowanie. Zwykle znaczy to, że model jest za mocno "
    "skwantyzowany (1–2 bity, nazwy w stylu IQ1_S czy Q2_K): wybierz wersję Q4 "
    "albo mniejszy model w pełniejszej jakości."
)
_ERR_INTERNAL = "Nieoczekiwany błąd agenta: {detail}"
_ERR_NO_TOOLS = (
    "Narzędzia agenta są niedostępne (localcode.agent.tools nie udostępnia default_tools()). "
    "Zaktualizuj instalację LocalCode."
)


class AgentBusyError(RuntimeError):
    """A turn is already running (message is Polish, safe to show)."""


class _Interrupted(Exception):
    """Internal: the user pressed Esc."""


class _ReviewerUnavailable(Exception):
    """Internal: the auto-mode reviewer model could not be acquired."""


def dict_events(callback: Callable[[dict[str, Any]], Any]) -> Callable[[Event], Any]:
    """Adapt a callback that wants wire JSON to the ``on_event(Event)`` contract."""

    def emit(event: Event) -> Any:
        return callback(event.to_dict())

    return emit


def _context_overflow_error() -> type[BaseException]:
    """``runtime.http_engine.ContextOverflowError`` when that module is importable."""
    try:
        from localcode.runtime.http_engine import ContextOverflowError
    except Exception:  # pragma: no cover - aiohttp missing

        class _NeverRaised(Exception):
            pass

        return _NeverRaised
    return ContextOverflowError


async def _aclose(agen: Any) -> None:
    close = getattr(agen, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:  # noqa: BLE001 - closing must never mask the real error
        pass


def _args_signature(name: str, args: Any) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str handles it
        blob = repr(args)
    return f"{name}\x00{blob}"


def _short(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


#: "Make me something" — in Polish as the users write it, and in English.
_MAKE_REQUEST = re.compile(
    r"\b(napisz|napisać|stw[oó]rz|stworzyć|zr[oó]b|zrobić|utw[oó]rz|wygeneruj|przygotuj|dodaj|"
    r"zbuduj|write|create|make|build|generate|add)\b", re.IGNORECASE)
#: …unless it asks to be *shown* something, where code in the chat is the answer.
_SHOW_REQUEST = re.compile(
    r"\b(poka[zż]|przyk[lł]ad|wyja[sś]nij|wyt[lł]umacz|jak (?:napisa[cć]|zrobi[cć])|"
    r"show|example|explain|how (?:do|to|would))\b", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)```", re.DOTALL)
#: Commands are instructions, not a file somebody forgot to save.
_SHELL_LANGUAGES = frozenset({"bash", "sh", "shell", "zsh", "console", "terminal", "cmd", "powershell"})
_MIN_FILE_LINES = 12


#: What one picture costs in the context, as far as budgeting is concerned.
IMAGE_TOKENS = 1024


def _countable(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Messages with text-only content, and how many pictures were taken out."""
    pictures = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                pictures += 1
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        out.append({**message, "content": "\n".join(texts)})
    return out, pictures


@dataclass
class _Turn:
    """Mutable state of the turn currently running."""

    id: str
    user_text: str
    started: float
    steps: int = 0
    total: GenStats = field(default_factory=GenStats)
    gen_seconds: float = 0.0
    text: str = ""            # visible text of the step being streamed
    reasoning: str = ""
    raw: str = ""             # untouched model output of the step
    malformed: int = 0
    empty_nudges: int = 0
    unsaved_nudges: int = 0
    wrote_files: bool = False    # a write_file/edit_file succeeded in this turn
    overflow_retries: int = 0
    looping: str | None = None   # the fragment the model got stuck repeating
    loading_announced: bool = False
    pending_calls: list[ToolCall] = field(default_factory=list)
    answered_calls: set[str] = field(default_factory=set)
    #: ``turn.steps`` when the current ``/goal`` continuation began: the step
    #: limit is per stretch of work, not for the whole goal.
    segment_start: int = 0
    #: Ultracode: steps with tool calls since the last ``delegate``, and how
    #: many times this turn the agent was reminded of helpers.
    quiet_steps: int = 0
    nudges: int = 0


@dataclass
class _Prepared:
    """One tool call past its checks: either a result already, or ready to run.

    ``result`` is set when the call will not run (unknown tool, denied …) and
    once it has run (``ran``). ``review`` is auto mode's verdict as it is
    stored on the tool item.
    """

    title: str
    tool: Tool | None = None
    args: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    review: dict[str, Any] | None = None
    ran: bool = False


class Agent:
    """One coding-agent session: history, tools, permissions and the turn loop.

    ``runtime`` is a :class:`localcode.runtime.manager.RuntimeManager` (anything
    with ``acquire(key, purpose=…)`` works), ``model_key`` a
    ``LocalModel.key``, ``cwd`` the project directory, ``settings`` the app
    settings (``settings.agent`` is used), ``on_event`` receives
    :class:`localcode.events.Event` instances and ``ask_permission`` answers
    :class:`~localcode.events.PermissionRequest` with
    ``"allow" | "deny" | "always"``.

    ``session`` is an optional :class:`~localcode.agent.session.SessionStore`;
    a resumed store's messages become this agent's history.
    ``tools`` overrides the tool set (a sequence of tools or a callable
    ``(mode, tiny) -> tools``) — the default comes from
    ``localcode.agent.tools.tools_for_mode``.
    """

    def __init__(
        self,
        runtime: Any,
        model_key: str,
        cwd: str | os.PathLike[str],
        settings: Settings,
        on_event: Callable[[Event], Any] | None = None,
        ask_permission: Callable[[PermissionRequest], Any] | None = None,
        *,
        session: SessionStore | None = None,
        tools: Sequence[Tool] | Callable[..., Sequence[Tool]] | None = None,
        policy: PermissionPolicy | None = None,
        system_suffix: str | None = None,
        system_prompt: str | None = None,
        tool_state: dict[str, Any] | None = None,
        allow_always: bool = True,
        helper: str | None = None,
        cache_key: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.cwd = Path(cwd).expanduser()
        self.on_event = on_event
        self.ask_permission = ask_permission
        self.session = session

        self._model_key = model_key
        self._manage_models = bool(getattr(settings.agent, "manage_models", False))
        self._subagents = bool(getattr(settings.agent, "subagents", False))
        #: Appended to the system prompt. Used for a helper agent's role, which
        #: must sit in the prompt rather than in the first user message — the
        #: conversation can be compacted, the prompt cannot.
        self._system_suffix = system_suffix
        #: Replaces the coding agent's system prompt entirely. The help bot is
        #: this same loop — permissions, compaction, the event stream — with a
        #: different job, and the coding prompt (project layout, git, "you edit
        #: files in this repository") would only mislead it.
        self._system_prompt = system_prompt
        #: Extra entries for ``ToolContext.state``: the facade a custom tool set
        #: works through. Seeded with the session's own and restored after
        #: ``clear()`` for the same reason those are.
        self._extra_tool_state = dict(tool_state or {})
        #: ``/goal``: what the agent keeps working towards (``tools/goal.py``).
        self._goal: GoalState | None = None
        #: Language of the user's latest message that could be told ("pl"/"en").
        self._user_language: str | None = None
        #: Asked before every automatic ``/goal`` continuation; ``True`` hands
        #: the floor back (the app has a message from the user waiting).
        self.goal_should_yield: Callable[[], bool] | None = None
        #: Hands over what the user typed while a turn runs (the app's queue).
        #: Read before every step after the first, so the agent sees it at the
        #: next step — not after a turn that may have no step limit at all.
        self.take_interjections: Callable[[], list[str]] | None = None
        #: ``cache_key``: given to a helper (its parent's key plus its name), so
        #: each helper's steps share a provider-side cache of their own.
        self._own_cache_key: str | None = cache_key or None
        #: Offer "always allow" on permission questions. The help bot turns it
        #: off: one click on "Zawsze" for a delete would let every later delete
        #: in that session through unasked, and "everything that changes
        #: something asks" is the one promise that makes its full power safe.
        self._allow_always = allow_always
        #: This agent is helper "h2" of another one (``tools/subagent.py``):
        #: its permission questions and auto-mode verdicts carry the name, so
        #: the user knows who asks.
        self._helper = helper or None
        #: Which squeeze the current model gets (see ``agent/profile.py``).
        self._profile = LOCAL_PROFILE
        #: Skills available here (``agent/skills.py``); the tool reads them.
        self._skills = SkillLibrary(self.cwd)
        self._tool_ctx_skills_ready = False
        self._mode = normalize_mode(
            (session.session.mode if session is not None else None) or settings.agent.permission_mode
        )
        self._policy = policy or PermissionPolicy(self._mode, cwd=str(self.cwd))
        self._policy.set_mode(self._mode)
        self._tools_override = tools

        self._history: list[dict[str, Any]] = list(session.messages) if session is not None else []
        self._tool_ctx = ToolContext(cwd=self.cwd, limits=ToolLimits(), state={})
        self._tool_ctx.progress = self._tool_progress
        self._seed_tool_state(settings, runtime)

        self._system: str | None = None
        self._system_key: tuple[Any, ...] | None = None
        self._tools: list[Tool] = []
        self._schemas: list[dict[str, Any]] = []
        self._env: dict[str, Any] | None = None
        self._instructions: str | None = None
        self._instructions_ctx: int | None = None
        #: First user message of the session, kept so language detection can also
        #: use files the task names; set once, so the prompt is still rebuilt at
        #: most once (agent/langskills.py).
        self._task_hint: str | None = None

        # Per-session reasoning-effort override; None = follow settings.
        self._effort: str | None = None

        self._formatter: ChatFormatter | None = None
        self._formatter_for: Any = None
        #: Formatter of the auto-mode reviewer when it is not the agent's model.
        self._review_fmt: tuple[Any, ChatFormatter] | None = None
        #: …and of the local guard (``agent/guard.py``), likewise.
        self._guard_fmt: tuple[Any, ChatFormatter] | None = None
        self._ctx: ContextManager | None = None
        self._ctx_n_ctx: int | None = None

        self._busy = False
        self._interrupt_event = asyncio.Event()
        self._current: asyncio.Task[Any] | None = None
        #: Ultracode's helpers running side by side; ``interrupt()`` cancels
        #: them all, not just the one the loop happens to be waiting for.
        self._helper_tasks: set[asyncio.Task[Any]] = set()
        self._small_window_logged: int | None = None
        self._pending_notice: str | None = None
        self._last_call_sig: str | None = None
        self._repeats = 0
        self._last_context: ContextUpdate | None = None
        self._overflow_error = _context_overflow_error()

        if session is not None:
            session.set_model(model_key)
            session.set_mode(self._mode)

    # ------------------------------------------------------------------ state

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The conversation as the model sees it (system prompt first, when built)."""
        system = [{"role": "system", "content": self._system}] if self._system is not None else []
        return system + [dict(m) for m in self._history]

    @property
    def history(self) -> list[dict[str, Any]]:
        """Model history without the system prompt (what the session file stores)."""
        return list(self._history)

    @property
    def items(self) -> list[dict[str, Any]]:
        """Display items of this session (empty without a session store)."""
        return list(self.session.items) if self.session is not None else []

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def context(self) -> ContextUpdate | None:
        """Breakdown of the last prompt (tokens used/limit/system/tools/history)."""
        return self._last_context

    @property
    def context_used(self) -> int:
        return self._last_context.used if self._last_context else 0

    @property
    def context_limit(self) -> int:
        if self._last_context:
            return self._last_context.limit
        return self._ctx_n_ctx or 0

    @property
    def tools(self) -> list[Tool]:
        """Tools offered in the current mode (empty until the first step)."""
        return list(self._tools)

    @property
    def policy(self) -> PermissionPolicy:
        return self._policy

    def session_dto(self) -> dict[str, Any]:
        """``AgentSession`` DTO for the HTTP API, with live counters."""
        session = self.session.session if self.session is not None else AgentSession(
            id="-", cwd=str(self.cwd), model_key=self._model_key, mode=self._mode
        )
        session.model_key = self._model_key
        session.mode = self._mode
        session.context_used = self.context_used
        session.context_limit = self.context_limit
        session.busy = self._busy
        return session.to_dict()

    def __repr__(self) -> str:
        return (f"Agent(model={self._model_key!r}, mode={self._mode!r}, cwd={str(self.cwd)!r}, "
                f"messages={len(self._history)}, busy={self._busy})")

    # ------------------------------------------------------------- commands

    def set_mode(self, mode: str) -> None:
        """Switch the permission mode (raises ``ValueError`` for unknown modes).

        The system prompt is rebuilt (plan mode offers fewer tools and different
        rules) and the model is told about the switch in the next user message.
        """
        new = normalize_mode(mode)
        if new == self._mode:
            return
        self._mode = new
        self._policy.set_mode(new)
        self._system = self._system_key = None
        notice = prompts.mode_notice(new)
        self._pending_notice = notice
        if self.session is not None:
            self.session.set_mode(new)
            self.session.append_item(self._item("notice", text=notice))

    def set_effort(self, level: str | None) -> None:
        """Override the reasoning effort for this session (``None`` = settings).

        Raises ``ValueError`` for a level outside ``off|low|medium|high|ultracode``.
        The level is only a *request*: each model resolves it down to what its
        chat template can express, and models without a control ignore it.
        ``ultracode`` thinks at "high" and changes how the agent works (see
        :attr:`ultracode`), so the prompt and the menu are rebuilt.
        """
        if level is None:
            new = None
        else:
            new = normalize_agent_effort(level)
            if new is None:
                raise ValueError(f"Nieznany poziom myślenia: {level}")
        if new == self._effort:
            return
        self._effort = new
        self._ctx = self._ctx_n_ctx = None  # the counter captured the old level

    @property
    def effort(self) -> str:
        """The requested reasoning effort (``ultracode`` included).

        Session override first (``/effort``), then this model's own default from
        the library ("Ustawienia ładowania…" — some models are good at "high" and
        some waste minutes there), then the global setting. What the template
        sees is :meth:`_engine_effort`, never ``ultracode``.
        """
        return (self._effort
                or self._model_effort()
                or normalize_agent_effort(self.settings.agent.reasoning_effort, "medium")
                or "medium")

    @property
    def ultracode(self) -> bool:
        """Effort "ultracode": think at "high" and orchestrate parallel helpers.

        ``delegate`` is offered (above :data:`ULTRACODE_MIN_CTX`), the prompt
        gets the orchestrator section, the ``delegate`` calls of one step run
        side by side and helpers get twice the steps.
        """
        return self.effort == ULTRACODE

    def _engine_effort(self, thinking: bool) -> str:
        """The level a template may see: ``off`` without thinking, never ``ultracode``."""
        return (engine_effort(self.effort) or "medium") if thinking else "off"

    def _model_effort(self) -> str | None:
        """The per-model reasoning level stored in the library, if any."""
        getter = getattr(getattr(self.runtime, "library", None), "load_defaults", None)
        if getter is None:
            return None
        try:
            defaults = getter(self._model_key) or {}
        except Exception:  # noqa: BLE001 - a missing model must not break the turn
            log.debug("load_defaults(%r) failed", self._model_key, exc_info=True)
            return None
        return normalize_effort(defaults.get("effort")) if defaults.get("effort") else None

    @property
    def effort_levels(self) -> list[str]:
        """Levels the current model can express; empty = hide the control."""
        return self._formatter.effort_levels() if self._formatter is not None else []

    def set_model(self, key: str) -> None:
        """Use another model from the next step on (context size may change)."""
        if not key or key == self._model_key:
            return
        self._model_key = key
        self._system = self._system_key = None
        self._formatter = self._formatter_for = None
        self._ctx = self._ctx_n_ctx = None
        if self.session is not None:
            self.session.set_model(key)
            self.session.append_item(self._item("notice", text=f"Model: {key}"))

    def clear(self) -> None:
        """Forget the conversation (the session file records the reset)."""
        self._history = []
        # A goal belongs to the conversation that pursued it.
        self._goal = None
        self._last_call_sig, self._repeats = None, 0
        self._last_context = None
        self._pending_notice = None
        self._tool_ctx.read_files.clear()
        # ``state`` holds two different things: scratch a tool left behind for
        # the next call (which the new conversation must not inherit) and the
        # session's own facades — the skill library and the model-management
        # one. Clearing the lot took those with it, so after ``/clear`` the
        # ``skill`` tool answered "there are no skills in this session" and the
        # model tools refused politely, for the rest of the session.
        self._tool_ctx.state.clear()
        self._seed_tool_state(self.settings, self.runtime)
        if self.session is not None:
            self.session.replace_history([], reason="clear")
            self.session.append_item(self._item("notice", text="Rozmowa wyczyszczona."))

    def _seed_tool_state(self, settings: Any, runtime: Any) -> None:
        """Put back the facades the tools reach for through ``ToolContext.state``.

        Called from ``__init__`` and again after ``clear()`` — they belong to
        the session, not to the conversation that was just thrown away.
        """
        self._tool_ctx.state[SKILLS_STATE_KEY] = self._skills
        self._tool_ctx.state[CONFINE_STATE_KEY] = bool(
            getattr(getattr(settings, "agent", None), "confine_to_project", True))
        self._tool_ctx.state.update(self._extra_tool_state)
        if getattr(self, "_goal", None) is not None:
            self._tool_ctx.state[GOAL_STATE_KEY] = self._goal
        if self._subagents:
            from localcode.agent.tools.subagent import STATE_KEY as SUBAGENT_KEY
            from localcode.agent.tools.subagent import SubagentHost

            self._tool_ctx.state[SUBAGENT_KEY] = SubagentHost(
                runtime=runtime, settings=settings, cwd=self.cwd,
                model_key=self._model_key, emit=self._emit)
        if self._manage_models:
            # The facade the model-management tools work through; without it
            # those tools refuse politely instead of reaching for the runtime.
            from localcode.agent.model_ops import LocalModelOps
            from localcode.agent.tools.models import STATE_KEY

            self._tool_ctx.state[STATE_KEY] = LocalModelOps(
                settings, runtime=runtime, on_switch=self._switch_model)

    def _sync_subagent_host(self) -> None:
        """Keep the helper host in step with the menu and the effort.

        Settings' ``subagents`` registers it up front (:meth:`_seed_tool_state`);
        ultracode offers ``delegate`` without that, so the host is made here
        the first time the menu has the tool — after ``/clear`` too. It is told
        before every step whether ultracode is on: the effort can change
        between turns, and the helpers' steps and the gate follow it.
        """
        from localcode.agent.tools.subagent import STATE_KEY as SUBAGENT_KEY
        from localcode.agent.tools.subagent import SubagentHost

        host = self._tool_ctx.state.get(SUBAGENT_KEY)
        if host is None:
            if not any(t.name == "delegate" for t in self._tools):
                return
            host = SubagentHost(runtime=self.runtime, settings=self.settings, cwd=self.cwd,
                                model_key=self._model_key, emit=self._emit)
            self._tool_ctx.state[SUBAGENT_KEY] = host
        setter = getattr(host, "set_ultracode", None)
        if callable(setter):
            setter(self.ultracode, self._ultracode_parallel())
        binder = getattr(host, "bind", None)
        if callable(binder):
            # A helper's commands are asked about like the agent's own.
            binder(ask_permission=self.ask_permission, policy=self._policy, mode=self._mode,
                   allow_always=self._allow_always, cache_key=self._cache_key)

    def _delegate_on(self, n_ctx: int) -> bool:
        """``delegate`` on the menu: Settings say so, or ultracode on a window that takes it."""
        if self._subagents:
            return True
        if not self.ultracode:
            return False
        if n_ctx < ULTRACODE_MIN_CTX:
            if self._small_window_logged != n_ctx:
                self._small_window_logged = n_ctx
                log.info("ultracode: delegate stays off for %s — its window (%d tokens) is below %d",
                         self._model_key, n_ctx, ULTRACODE_MIN_CTX)
            return False
        return True

    def interrupt(self) -> None:
        """Stop the running turn as soon as possible (safe to call any time)."""
        if not self._busy:
            return
        self._interrupt_event.set()
        task = self._current
        if task is not None and not task.done():
            task.cancel()
        for helper in list(self._helper_tasks):
            if not helper.done():
                helper.cancel()
        host = self._subagent_host()
        if host is not None:
            host.cancel_background()

    def _subagent_host(self) -> Any:
        """The helper host when it can run background helpers (``tools/subagent.py``)."""
        from localcode.agent.tools.subagent import STATE_KEY as SUBAGENT_KEY

        host = self._tool_ctx.state.get(SUBAGENT_KEY)
        return host if host is not None and hasattr(host, "take_finished") else None

    async def compact(self) -> None:
        """Summarise the conversation now (``/compact``), even if it still fits."""
        if self._busy:
            raise AgentBusyError("Agent jest zajęty — poczekaj na zakończenie tury.")
        if not self._history:
            return
        self._busy = True
        self._interrupt_event.clear()
        try:
            await self._run_cancellable(self._compact_once())
        except _Interrupted:
            await self._emit(ErrorEvent(message="Kompaktowanie przerwane."))
        except EngineError as exc:
            await self._emit(ErrorEvent(message=str(exc)))
        finally:
            self._busy = False
            self._current = None

    async def _compact_once(self, *, aggressive: bool = False) -> None:
        """Rewrite the history under a fresh lease (``/compact`` or after an overflow).

        The summariser is the model itself, so this needs a slot of its own —
        the one the failed step used has already been released.
        """
        async with _lease(self.runtime, self._model_key, purpose="agent") as lease:
            engine = lease.engine
            formatter = self._get_formatter(engine)
            await self._prepare_tools(engine)
            ctx = self._get_ctx(engine, formatter)
            ctx.summarizer = self._make_summarizer(engine, formatter)
            messages = self._prompt_messages()
            if aggressive:
                fitted, compactions = await ctx.on_overflow(messages, self._schemas)
            else:
                fitted, compactions = await ctx.compact(messages, self._schemas)
            await self._apply_fit(fitted, compactions,
                                  reason="overflow" if aggressive else "compaction")
            usage = await ctx.usage(self._prompt_messages(), self._schemas)
            self._last_context = usage
            await self._emit(usage)

    # ------------------------------------------------------------------ goal

    @property
    def goal(self) -> dict[str, Any] | None:
        """The ``/goal`` of this session (``None`` when there is none)."""
        return self._goal.to_dict() if self._goal is not None else None

    async def set_goal(self, text: str) -> dict[str, Any]:
        """``/goal <cel>``: work until ``text`` is achieved (the caller then sends it as a turn)."""
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("Cel nie może być pusty.")
        max_turns = max(1, int(getattr(self.settings.agent, "goal_max_turns", 20) or 20))
        self._goal = GoalState(text=text, max_turns=max_turns)
        self._tool_ctx.state[GOAL_STATE_KEY] = self._goal
        self._system = self._system_key = None
        self._note(f"Cel: {text}")
        await self._emit_goal()
        return self._goal.to_dict()

    async def clear_goal(self) -> None:
        """``/goal wyczyść``: stop working towards the goal."""
        if self._goal is None:
            return
        self._goal.state = "cleared"
        await self._emit_goal()
        self._note("Cel usunięty.")
        self._goal = None
        self._tool_ctx.state.pop(GOAL_STATE_KEY, None)
        self._system = self._system_key = None

    async def _emit_goal(self) -> None:
        goal = self._goal
        if goal is None:
            return
        await self._emit(GoalUpdate(text=goal.text, state=goal.state, turns=goal.turns,
                                    max_turns=goal.max_turns, summary=goal.summary))

    async def _goal_continue(self, turn: _Turn) -> bool:
        """After a final answer (or a spent step budget): back to work on the goal?

        ``True`` means the nudge is in the history and the turn goes on — one
        turn from the outside, so every client renders it with the code it
        already has, and Esc stops the whole run. ``False`` ends the turn: no
        goal, the goal was declared done or blocked, the user interrupted or has
        a message waiting, or the continuations ran out.
        """
        goal = self._goal
        if goal is None:
            return False
        if goal.active and turn.text:
            written = call_in_text(turn.text)
            if written is not None:
                # The model declared the goal reached (or blocked) in words
                # rather than with the tool — take it at its word, as the tool
                # would have.
                goal.summary, blocked = written
                goal.state = "blocked" if blocked else "done"
        if not goal.active:
            if goal.state in ("done", "blocked") and not goal.announced:
                goal.announced = True
                self._note("Cel osiągnięty." if goal.state == "done"
                           else "Cel czeka na Twoją decyzję.")
                await self._emit_goal()
            return False
        if self._interrupt_event.is_set():
            return False
        if self.goal_should_yield is not None:
            try:
                if self.goal_should_yield():
                    return False
            except Exception:  # noqa: BLE001 - a host callback must not end the turn badly
                log.debug("goal_should_yield failed", exc_info=True)
        if goal.turns >= goal.max_turns:
            goal.state = "limit"
            self._note(f"Cel nie został osiągnięty po {goal.max_turns} podejściach — "
                       "agent zatrzymał się. Napisz, co dalej, albo /goal wyczyść.")
            await self._emit_goal()
            return False
        # A stretch that was only an answer — no tool call — repeating the
        # previous one is a model that has nothing left to try; sending it back
        # again only costs minutes.
        answer = " ".join((turn.text or "").split())[:200]
        idle = turn.steps - turn.segment_start <= 1
        if goal.turns and idle and answer and answer == goal.last_answer:
            goal.repeats += 1
        else:
            goal.repeats = 0
        goal.last_answer = answer
        if goal.repeats >= 2:
            goal.state = "stalled"
            self._note("Agent stoi w miejscu — powtarza tę samą odpowiedź. "
                       "Napisz, co dalej, albo /goal wyczyść.")
            await self._emit_goal()
            return False
        goal.turns += 1
        turn.segment_start = turn.steps
        turn.text = ""
        turn.unsaved_nudges = 0
        self._append_user(prompts.GOAL_CONTINUE.format(goal=goal.text, n=goal.turns,
                                                       max=goal.max_turns))
        self._note(f"Kontynuuję pracę nad celem ({goal.turns}/{goal.max_turns}).")
        await self._emit_goal()
        return True

    # ------------------------------------------------------------------ turn

    async def run_turn(self, user_text: str, images: Sequence[dict[str, Any]] | None = None) -> str:
        """Run one user turn to completion, streaming events; returns the answer."""
        if self._busy:
            raise AgentBusyError("Agent jest zajęty — poczekaj na zakończenie tury.")
        self._busy = True
        self._interrupt_event.clear()
        self._last_call_sig, self._repeats = None, 0
        # The helper budget is per turn: six on one question is already a model
        # that has decided delegating is the work.
        host = self._tool_ctx.state.get("subagents")
        if host is not None:
            host.reset_turn()
        turn = _Turn(id=secrets.token_hex(6), user_text=user_text, started=time.monotonic())
        reason = "done"
        answer = ""
        try:
            await self._emit(TurnStart(turn_id=turn.id, user_text=user_text))
            language = prompts.detect_language(user_text)
            if language is not None:
                self._user_language = language
            if self._task_hint is None and user_text.strip():
                self._task_hint = user_text
            self._start_user_message(user_text, images)
            max_steps = int(self.settings.agent.max_steps or 0)       # 0 = no limit
            while True:
                if self._interrupt_event.is_set():
                    reason = "interrupted"
                    break
                if max_steps > 0 and turn.steps - turn.segment_start >= max_steps:
                    # Out of steps while working on a goal is not the end of
                    # it — that is exactly the stretch /goal exists for.
                    if await self._goal_continue(turn):
                        continue
                    reason = "max_steps"
                    await self._emit(ErrorEvent(message=_ERR_MAX_STEPS.format(n=max_steps)))
                    self._note(_ERR_MAX_STEPS.format(n=max_steps))
                    break
                if turn.steps > 0:
                    await self._take_interjections()
                    self._take_helper_reports()
                    self._nudge_to_delegate(turn)
                turn.steps += 1
                try:
                    parsed = await self._run_cancellable(self._generate_step(turn))
                except _Interrupted:
                    self._finish_interrupted_generation(turn)
                    reason = "interrupted"
                    break
                except self._overflow_error as exc:  # type: ignore[misc]
                    try:
                        recovered = await self._recover_overflow(turn, exc)
                    except _Interrupted:
                        reason = "interrupted"
                        break
                    if not recovered:
                        reason = "error"
                        break
                    continue
                except EngineError as exc:
                    log.warning("agent step failed: %s", exc)
                    await self._emit(ErrorEvent(message=str(exc) or _ERR_INTERNAL.format(detail="engine")))
                    self._note(str(exc))
                    reason = "error"
                    break

                outcome = await self._handle_output(turn, parsed)
                if outcome == "continue":
                    continue
                if outcome == "interrupted":
                    reason = "interrupted"
                    break
                if outcome == "error":
                    reason = "error"
                    break
                answer = turn.text
                reason = "done"
                try:
                    # Helpers still working in the background: their reports
                    # are part of this turn, never dropped.
                    if await self._wait_for_helpers():
                        continue
                except _Interrupted:
                    reason = "interrupted"
                    break
                if await self._goal_continue(turn):
                    continue
                break
        except asyncio.CancelledError:
            reason = "interrupted"
            raise
        except Exception as exc:  # never leave the caller without a TurnEnd
            log.exception("agent turn crashed")
            await self._emit(ErrorEvent(message=_ERR_INTERNAL.format(detail=f"{type(exc).__name__}: {exc}")))
            reason = "error"
        finally:
            await self._stop_background_helpers()
            self._busy = False
            self._current = None
            if reason == "interrupted":
                self._note("Przerwano przez użytkownika.")
            self._forget_images()
            await self._emit(TurnEnd(turn_id=turn.id, reason=reason, stats=self._turn_stats(turn)))
        return answer

    def _turn_stats(self, turn: _Turn) -> dict[str, Any]:
        total = turn.total
        seconds = turn.gen_seconds or (total.total_ms / 1000.0 if total.total_ms else 0.0)
        gen_tps = total.completion_tokens / seconds if seconds > 0 else 0.0
        stats = total.to_dict()
        stats["genTps"] = round(gen_tps, 1)
        stats["steps"] = turn.steps
        stats["elapsedMs"] = round((time.monotonic() - turn.started) * 1000)
        return stats

    # --------------------------------------------------------- generation

    async def _generate_step(self, turn: _Turn) -> ParsedOutput:
        """One model step: fit the context, stream the answer, parse it.

        The runtime slot is held only for the duration of this coroutine; tool
        calls run after it returns.
        """
        if not self._model_loaded():
            turn.loading_announced = True
            await self._emit(Status(message="Ładowanie modelu…", phase="loading"))
        async with _lease(self.runtime, self._model_key, purpose="agent") as lease:
            engine = lease.engine
            if turn.loading_announced:
                turn.loading_announced = False
                await self._emit(Status(message="Myślę…", phase="thinking"))
            formatter = self._get_formatter(engine)
            await self._prepare_tools(engine)
            self._sync_subagent_host()
            ctx = self._get_ctx(engine, formatter)
            ctx.summarizer = self._make_summarizer(engine, formatter)
            self._tool_ctx.limits = ctx.limits()

            fitted, compactions = await ctx.fit(self._prompt_messages(), self._schemas)
            await self._apply_fit(fitted, compactions)
            usage = await ctx.usage(fitted, self._schemas)
            self._last_context = usage
            await self._emit(usage)

            thinking = bool(self.settings.agent.thinking)
            messages, images = self._media_for(engine, self._with_language_reminder(fitted))
            prompt = await formatter.render(messages, self._schemas or None, thinking=thinking,
                                            effort=self._engine_effort(thinking))
            stats = await self._stream(lease, formatter, prompt, turn, images=images)
            if stats is not None:
                self._add_stats(turn, stats)
                lease.record_stats(stats)
            return formatter.parse(turn.raw, prompt, self._schemas or None)

    async def _stream(self, lease: Any, formatter: ChatFormatter, prompt: str, turn: _Turn,
                      images: Sequence[str] = ()) -> GenStats | None:
        """Stream one completion, emitting text/reasoning deltas as they arrive."""
        params = SamplingParams(
            max_tokens=self._max_output_tokens(),
            temperature=float(self.settings.agent.temperature),
            stop=formatter.stop_strings(),
            images=list(images),
            cache_key=self._cache_key,
        )
        splitter = formatter.splitter(prompt, self._schemas or None)
        guard = LoopGuard()
        turn.raw = turn.text = turn.reasoning = ""
        turn.looping = None
        stats: GenStats | None = None
        started = time.monotonic()
        agen = lease.engine.complete(prompt, params)
        try:
            async for chunk in agen:
                if chunk.text:
                    turn.raw += chunk.text
                    await self._emit_pieces(turn, splitter.feed(chunk.text))
                    if guard.feed(chunk.text):
                        # a model repeating itself will not stop on its own: it
                        # would fill the whole output budget with the same line
                        turn.looping = guard.describe()
                        log.info("stopped a looping generation (%r)", turn.looping)
                        break
                if chunk.done:
                    stats = chunk.stats
        finally:
            turn.gen_seconds += time.monotonic() - started
            await _aclose(agen)
        await self._emit_pieces(turn, splitter.finish())
        if turn.looping is None and guard.finish():
            turn.looping = guard.describe()
        return stats

    async def _emit_pieces(self, turn: _Turn, pieces: Iterable[tuple[str, str]]) -> None:
        for kind, piece in pieces:
            if not piece:
                continue
            if kind == "reasoning":
                turn.reasoning += piece
                await self._emit(ReasoningDelta(text=piece))
            else:
                turn.text += piece
                await self._emit(TextDelta(text=piece))

    def _max_output_tokens(self) -> int:
        configured = self.settings.agent.max_output_tokens
        ctx = self._ctx
        budget = ctx.output_budget(self._last_context.used) if ctx and self._last_context else 0
        if configured:
            wanted = int(configured)
        elif ctx is not None:
            wanted = max(ctx.reserve_output, 1024)
        else:
            wanted = 1024
        return max(64, min(wanted, budget) if budget > 0 else wanted)

    def _add_stats(self, turn: _Turn, stats: GenStats) -> None:
        total = turn.total
        total.prompt_tokens += stats.prompt_tokens
        total.cached_tokens += stats.cached_tokens
        total.completion_tokens += stats.completion_tokens
        total.total_ms += stats.total_ms
        total.prompt_tps = stats.prompt_tps or total.prompt_tps
        total.gen_tps = stats.gen_tps or total.gen_tps

    def _make_summarizer(self, engine: Any, formatter: ChatFormatter) -> Callable[[list[dict[str, Any]]], Any]:
        """Summarizer for ``ContextManager``: the same engine, no tools, no thinking."""
        max_tokens = self._ctx.summary_tokens if self._ctx is not None else 600

        async def summarize(prompt_messages: list[dict[str, Any]]) -> str:
            prompt = await formatter.render(prompt_messages, None, thinking=False)
            params = SamplingParams(max_tokens=max_tokens, temperature=0.2, stop=formatter.stop_strings())
            parts: list[str] = []
            agen = engine.complete(prompt, params)
            try:
                async for chunk in agen:
                    if chunk.text:
                        parts.append(chunk.text)
            finally:
                await _aclose(agen)
            return formatter.parse("".join(parts), prompt, None).content

        return summarize

    # ------------------------------------------------------------ step result

    async def _handle_output(self, turn: _Turn, parsed: ParsedOutput) -> str:
        """Append the assistant message and run its tool calls.

        Returns ``"continue"`` (next step), ``"done"``, ``"interrupted"`` or
        ``"error"``.
        """
        calls = list(parsed.tool_calls)
        content = parsed.content.strip()
        if turn.looping and not calls:
            return await self._handle_looping(turn, parsed)
        if not calls and parsed.errors:
            # a tool call we could not read is not an empty reply: correct it first
            return await self._handle_malformed(turn, parsed)
        if not calls and not content and not parsed.reasoning.strip():
            return await self._handle_empty(turn)

        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning:
            message["reasoning"] = parsed.reasoning
        if calls:
            message["tool_calls"] = [c.to_message() for c in calls]
        items: list[dict[str, Any]] = []
        if parsed.reasoning.strip():
            items.append(self._item("reasoning", text=parsed.reasoning))
        if content:
            items.append(self._item("assistant", text=parsed.content))
        self._append(message, items=items)

        if not calls and self._pasted_instead_of_saving(turn, content):
            # The answer carries a whole file that was never written: the model
            # treated "make me X" as a chat request. Said once, it saves it.
            turn.unsaved_nudges += 1
            self._append_user(prompts.UNSAVED_CODE_NUDGE)
            return "continue"
        if not calls:
            return "done"

        turn.pending_calls = calls
        turn.answered_calls = set()
        note = parsed.errors[0] if parsed.errors else None
        turn.quiet_steps = 0 if any(self._is_delegate(c) for c in calls) else turn.quiet_steps + 1
        if self._fans_out(calls):
            await self._run_fan_out(turn, calls, note)
        else:
            for index, call in enumerate(calls):
                if self._interrupt_event.is_set():
                    break
                try:
                    await self._run_call(turn, call, extra=note if index == len(calls) - 1 else None)
                except _Interrupted:
                    break
        if self._interrupt_event.is_set() or len(turn.answered_calls) < len(calls):
            self._close_pending_calls(turn)
            return "interrupted"
        return "continue"

    def _pasted_instead_of_saving(self, turn: _Turn, content: str) -> bool:
        """The answer pastes a whole file the user asked to have *made*.

        Seen in a real session with a 4B model: "napisz mi stronę o bananach"
        got the HTML in the chat, a claim that index.html had been written, and
        not one tool call all session — so "localhost" stayed empty. Narrow on
        purpose, every condition must hold: write_file is on the menu, nothing
        was written this turn, the request asks for something to be made rather
        than shown, and the answer contains a block long enough to be a file
        that is not shell commands. Once per turn.
        """
        if turn.unsaved_nudges >= 1 or turn.wrote_files or self._mode == "plan":
            return False
        if not any(t.name == "write_file" for t in self._tools):
            return False
        if not _MAKE_REQUEST.search(turn.user_text) or _SHOW_REQUEST.search(turn.user_text):
            return False
        for language, body in _CODE_BLOCK.findall(content):
            if language.lower() in _SHELL_LANGUAGES:
                continue
            if sum(1 for line in body.splitlines() if line.strip()) >= _MIN_FILE_LINES:
                return True
        return False

    async def _handle_looping(self, turn: _Turn, parsed: ParsedOutput) -> str:
        """The model repeated itself instead of answering: say so and stop.

        Retrying is pointless — the same prompt produces the same loop — and the
        cause is almost always the model itself, so the message names the fix.
        """
        message = _ERR_LOOPING.format(fragment=turn.looping)
        await self._emit(ErrorEvent(message=message))
        self._note(message)
        return "error"

    async def _handle_empty(self, turn: _Turn) -> str:
        if turn.empty_nudges >= 1:
            await self._emit(ErrorEvent(message=_ERR_EMPTY))
            self._note(_ERR_EMPTY)
            return "error"
        turn.empty_nudges += 1
        self._append_user(prompts.EMPTY_RESPONSE_NUDGE)
        return "continue"

    async def _handle_malformed(self, turn: _Turn, parsed: ParsedOutput) -> str:
        """The model tried to call a tool but we could not parse it."""
        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning:
            message["reasoning"] = parsed.reasoning
        items = [self._item("assistant", text=parsed.content)] if parsed.content.strip() else []
        self._append(message, items=items)
        if turn.malformed >= _MAX_MALFORMED_RETRIES:
            await self._emit(ErrorEvent(message=_ERR_MALFORMED))
            self._note(_ERR_MALFORMED + " " + _short(parsed.errors[0] if parsed.errors else "", 160))
            return "error"
        turn.malformed += 1
        fmt = self._formatter.parse_format(self._schemas or None) if self._formatter else None
        # The parser usually knows exactly what went wrong; pass that on.
        detail = (parsed.errors[0].splitlines()[0] if parsed.errors else "")
        self._append_user(prompts.malformed_call_hint(format_hint(fmt), detail=detail))
        return "continue"

    # -------------------------------------------------------------- tool call

    async def _run_call(self, turn: _Turn, call: ToolCall, *, extra: str | None = None) -> None:
        """Permission-check and execute one tool call, recording both events."""
        prepared = await self._prepare_call(turn, call)
        if prepared.result is None and prepared.tool is not None:
            # Which call this is, for a tool that emits events of its own: the
            # helper agent tags every one of its events with the `delegate` call
            # they belong to, so a client can nest them under that card.
            self._tool_ctx.state["current_call_id"] = call.id
            prepared.result = await self._run_cancellable(prepared.tool.run(prepared.args, self._tool_ctx))
            prepared.ran = True
        await self._complete_call(turn, call, prepared, extra=extra)

    async def _prepare_call(self, turn: _Turn, call: ToolCall) -> _Prepared:
        """Everything before a call runs: the event, the checks, the permission.

        Returns a :class:`_Prepared` whose ``result`` is already set when the
        call must not run. Auto mode's reviewer and the user's question happen
        here, so a call that comes back without a result may run at once.
        """
        tool = self._tool_by_name(call.name)
        args = call.arguments if isinstance(call.arguments, dict) else {}
        title = tool.title(args) if tool is not None else call.name
        await self._emit(ToolCallEvent(call_id=call.id, name=call.name, arguments=args, title=title))

        if tool is None:
            available = ", ".join(t.name for t in self._tools) or "brak"
            return _Prepared(title, result=ToolResult(
                ok=False, summary="Nieznane narzędzie",
                output=_UNKNOWN_TOOL_RESULT.format(name=call.name, available=available)))
        if not isinstance(call.arguments, dict):
            return _Prepared(title, result=ToolResult(
                ok=False, summary="Złe argumenty",
                output=_BAD_ARGS_RESULT.format(name=call.name, got=type(call.arguments).__name__)))

        signature = _args_signature(call.name, args)
        if signature == self._last_call_sig:
            self._repeats += 1
        else:
            self._last_call_sig, self._repeats = signature, 1
        if self._repeats >= _REPEAT_LIMIT:
            return _Prepared(title, result=ToolResult(
                ok=False, summary="Powtórzone wywołanie", output=prompts.REPEATED_CALL_NUDGE))

        decision, info = await self._decide(tool, args)
        if decision == "deny":
            return _Prepared(title, result=ToolResult(
                ok=False, summary="Odrzucono", output=self._denied_text(tool)))
        review: dict[str, Any] | None = None      # what the user's question carries
        stored: dict[str, Any] | None = None      # what the tool item keeps
        if decision == "ask" and self._mode == "auto":
            # Auto mode: the reviewer model — and, for commands, the local
            # guard that sees no context — answer instead of the user. A call
            # that needs a human (catastrophic, data to a provider) is still
            # judged, but only for the card: no TAK approves it.
            advisory = self._needs_human(tool, args, info)
            review = await self._auto_review(turn, call, tool, args, info, title, advisory=advisory)
            stored = None if advisory else review
            if autoreview.approved(review) and not advisory:
                decision = "allow"
        if decision == "ask":
            # A NIE is not the last word: the user sees it on the card, with
            # what the action would do, and may still allow it.
            answer = await self._ask(tool, args, info, title, review=review)
            if answer in ("deny", PERMISSION_DISMISSED):
                if answer == "deny":
                    self._learn(tool, args, info, precedents.DECLINED)
                refusal = autoreview.refusal_text(review, user_declined=True)
                result = (ToolResult(ok=False, summary="Odrzucone", output=refusal) if refusal else
                          ToolResult(ok=False, summary="Odrzucone przez użytkownika",
                                     output=_DENIED_RESULT))
                return _Prepared(title, result=result, review=stored)
            if answer == "unavailable":
                # Nobody to ask: a NIE stands, as the reviewer's always did.
                refusal = autoreview.refusal_text(review, user_declined=False)
                by = ("Odrzucone przez recenzenta" if review is not None and review["verdict"] == "deny"
                      else "Odrzucone przez strażnika")
                result = (ToolResult(ok=False, summary=by, output=refusal) if refusal else
                          ToolResult(ok=False, summary="Brak zgody", output=_NO_APPROVER_RESULT))
                return _Prepared(title, result=result, review=stored)
            if answer == "always" and not self._allow_always:
                answer = "allow"           # a client that offered it anyway gets a one-off yes
            if answer == "always":
                self._policy.remember_always(tool, args)
            self._learn(tool, args, info, precedents.APPROVED)
        return _Prepared(title, tool=tool, args=args, review=stored)

    async def _complete_call(self, turn: _Turn, call: ToolCall, prepared: _Prepared, *,
                             extra: str | None = None) -> None:
        """Record a prepared call's result (and its todo list, for ``todo_write``)."""
        result = prepared.result
        assert result is not None
        if prepared.ran and extra:
            result.output = f"{result.output}\n\nNote: {extra}"
        await self._finish_call(turn, call, prepared.title, result, review=prepared.review)
        tool = prepared.tool
        if prepared.ran and tool is not None and tool.name in _TODO_TOOLS and result.ok:
            todos = self._todo_items(prepared.args)
            if todos is not None:
                await self._emit(TodoUpdate(items=todos))

    # ------------------------------------------------------- ultracode fan-out

    def _is_delegate(self, call: ToolCall) -> bool:
        tool = self._tool_by_name(call.name)
        return tool is not None and tool.name == "delegate" and isinstance(call.arguments, dict)

    def _fans_out(self, calls: Sequence[ToolCall]) -> bool:
        """Ultracode, and this step hands out more than one ``delegate``."""
        return self.ultracode and sum(1 for c in calls if self._is_delegate(c)) > 1

    def _ultracode_parallel(self) -> int:
        return max(1, int(getattr(self.settings.agent, "ultracode_parallel", 6) or 6))

    async def _run_fan_out(self, turn: _Turn, calls: Sequence[ToolCall], note: str | None) -> None:
        """Ultracode: the step's ``delegate`` calls run side by side, the rest in order.

        Every helper starts at once (at most ``ultracode_parallel`` work at the
        same time, the others wait for a place); then the calls are taken in
        the order the model wrote them — a helper's result is waited for when
        its turn comes, any other tool runs then, one after another as always.
        So the history is the same whatever finished first. Each helper gets a
        context of its own carrying its call id, which is how its
        ``subagent_*`` events stay apart from its neighbours'. ``interrupt()``
        cancels every helper; whatever never answered is closed by the caller.
        """
        gate = asyncio.Semaphore(self._ultracode_parallel())
        started: dict[str, tuple[_Prepared, asyncio.Task[Any] | None]] = {}
        helpers: list[asyncio.Task[Any]] = []
        last = len(calls) - 1
        try:
            for call in calls:
                if self._interrupt_event.is_set():
                    return
                if not self._is_delegate(call) or call.id in started:
                    continue
                prepared = await self._prepare_call(turn, call)
                task: asyncio.Task[Any] | None = None
                if prepared.result is None and prepared.tool is not None:
                    ctx = replace(self._tool_ctx, call_id=call.id)
                    task = asyncio.ensure_future(_gated(gate, prepared.tool, prepared.args, ctx))
                    self._helper_tasks.add(task)
                    helpers.append(task)
                started[call.id] = (prepared, task)
            for index, call in enumerate(calls):
                if self._interrupt_event.is_set():
                    return
                extra = note if index == last else None
                entry = started.pop(call.id, None)
                if entry is None:
                    await self._run_call(turn, call, extra=extra)
                    continue
                prepared, task = entry
                if task is not None:
                    prepared.result = await self._run_cancellable(task)
                    prepared.ran = True
                await self._complete_call(turn, call, prepared, extra=extra)
        except _Interrupted:
            return
        finally:
            await self._stop_helpers(helpers)

    async def _stop_helpers(self, tasks: Sequence[asyncio.Task[Any]]) -> None:
        """Cancel helpers whose results nobody will read, and wait until they are gone.

        Waiting matters: a helper that is cancelled still reports its end
        (``subagent_end``) and frees its place, and the turn must not end
        before it has.
        """
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            self._helper_tasks.discard(task)

    async def _finish_call(self, turn: _Turn, call: ToolCall, title: str, result: ToolResult, *,
                           review: dict[str, Any] | None = None) -> None:
        """Emit the result, append the tool message and its display item.

        ``review`` (auto mode) is stored on the item, so the verdict and its
        reason are still there when the session is opened again.
        """
        if result.ok and call.name in ("write_file", "edit_file"):
            turn.wrote_files = True
        display = result.display if result.display is not None else result.output
        await self._emit(ToolResultEvent(
            call_id=call.id, name=call.name, ok=result.ok,
            summary=result.summary, output=display, diff=result.diff))
        ctx = self._ctx
        content = ctx.truncate_tool_output(result.output) if ctx is not None else result.output
        if not content.strip():
            # an empty tool message reads like a failure to a small model
            content = "(the tool produced no output)"
        item = self._item(
            "tool", id=call.id, toolName=call.name, title=title,
            arguments=call.arguments if isinstance(call.arguments, dict) else {},
            ok=result.ok, summary=result.summary, output=display, diff=result.diff, review=review,
        )
        self._append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content},
            items=[item],
        )
        turn.answered_calls.add(call.id)

    def _denied_text(self, tool: Tool) -> str:
        if self._mode == "plan":
            return _PLAN_MODE_RESULT.format(name=tool.name)
        return _DENIED_RESULT

    async def _decide(self, tool: Tool, args: dict[str, Any]) -> tuple[str, PermissionInfo | None]:
        info: PermissionInfo | None = None
        try:
            info = await tool.permission(args, self._tool_ctx)
        except Exception:  # a broken preview must not block the call
            log.warning("permission() of %s failed", tool.name, exc_info=True)
        try:
            decision = self._policy.decide(tool, args, info)
        except Exception:  # pragma: no cover - be conservative
            log.warning("permission policy failed for %s", tool.name, exc_info=True)
            decision = "ask"
        return decision, info

    # ------------------------------------------------------------- auto mode

    def _needs_human(self, tool: Tool, args: dict[str, Any], info: PermissionInfo | None) -> bool:
        """Flagged or catastrophic calls are the user's in auto mode too."""
        try:
            return self._policy.needs_human(tool, args, info)
        except Exception:  # pragma: no cover - be conservative
            log.warning("needs_human failed for %s", tool.name, exc_info=True)
            return True

    def _precedent_keys(self, tool: Tool, args: dict[str, Any],
                        info: PermissionInfo | None) -> list[precedents.Key] | None:
        """The keys the user's answers on this call are learned under (``agent/precedents.py``).

        ``None`` when nothing is learned or used for it: "Ucz się z moich
        decyzji" is off, or the call is neither a shell command nor reaching
        outside the project. ``[]``: in scope, but nothing our parser could
        name (``python -c …``, a redirection) — such a call is not recorded.
        """
        if not getattr(self.settings.agent, "auto_precedents_enabled", True):
            return None
        if tool.kind != "exec" and OUTSIDE_PROJECT_FLAG not in str(getattr(info, "title", "") or ""):
            return None
        try:
            return precedents.call_keys(tool, args, self._tool_ctx)
        except Exception:  # noqa: BLE001 - a key that cannot be computed is just not learned
            log.warning("precedent keys failed for %s", tool.name, exc_info=True)
            return []

    def _learn(self, tool: Tool, args: dict[str, Any], info: PermissionInfo | None,
               decision: str) -> None:
        """Remember the user's answer on a permission card, in any mode.

        A yes to a call that needs a human (catastrophic, data to a provider)
        is not learned: it must not make anything else look approved. A
        "dismissed" card (nobody answered) never gets here.
        """
        keys = self._precedent_keys(tool, args, info)
        if not keys:
            return
        if decision == precedents.APPROVED and self._needs_human(tool, args, info):
            return
        try:
            precedents.record(self.cwd, keys, decision)
        except Exception:  # noqa: BLE001 - memory is a convenience, the call goes on
            log.warning("recording a precedent failed", exc_info=True)

    def _precedent_notes(self, tool: Tool, args: dict[str, Any],
                         info: PermissionInfo | None) -> tuple[list[str], list[str]]:
        """``(reviewer lines, guard lines)``: the user's past answers to show for this call.

        The reviewer, which has the conversation, gets only the matching
        ones; the guard, which has nothing else, gets those first and then
        the most recent others — what this user approves and declines here.
        """
        keys = self._precedent_keys(tool, args, info)
        if keys is None:
            return [], []
        try:
            entries = precedents.load(self.cwd)
        except Exception:  # noqa: BLE001 - no memory, a plain review
            log.warning("reading precedents failed", exc_info=True)
            return [], []
        return precedents.reviewer_lines(entries, keys), precedents.guard_lines(entries, keys)

    def _reviewer_keys(self) -> list[str]:
        """Who reviews in auto mode: the model from Settings, then the agent's own."""
        chosen = str(getattr(self.settings.agent, "auto_reviewer_model", None) or "").strip()
        return ([chosen] if chosen and chosen != self._model_key else []) + [self._model_key]

    async def _auto_review(self, turn: _Turn, call: ToolCall, tool: Tool, args: dict[str, Any],
                           info: PermissionInfo | None, title: str, *,
                           advisory: bool = False) -> dict[str, Any] | None:
        """Auto mode: let the reviewer model judge one call (``agent/autoreview.py``).

        Returns ``{verdict, reason, model}``: ``allow``/``deny``, or ``ask`` when
        there is no usable answer (unclear, a tool call, an engine error, a
        timeout, no reviewer could be acquired) — the user decides then and
        ``reason`` says why. A reviewer from Settings that cannot be acquired
        gives way to the agent's own model. Emits :class:`AutoReview`; raises
        ``_Interrupted`` when the user stops the turn meanwhile.

        ``advisory`` (a call that needs a human anyway): the verdict is only an
        opinion for the permission card, so no :class:`AutoReview` is emitted —
        a client would read its TAK as "it ran" — and a review that gave no
        answer returns ``None``: the user is asked as before, without one.

        A shell command, or a call reaching outside the project, is also put
        to the local guard (``agent/guard.py``) at the same time; its opinion
        is ``review["guard"]``, and the call runs on its own only when both
        said TAK (:func:`autoreview.approved`).

        The user's earlier answers on similar calls (``agent/precedents.py``)
        go into both prompts — never for an ``advisory`` call, and they only
        make a gate more or less willing: both TAKs are still needed.
        """
        await self._emit(Status(message="Recenzent ocenia akcję…", phase="thinking"))
        noted, history = ([], []) if advisory else self._precedent_notes(tool, args, info)
        reviewing = self._reviewer_verdict(turn, call, tool, args, info, title, noted)
        opinion: dict[str, Any] | None = None
        if guard.in_scope(tool, info, OUTSIDE_PROJECT_FLAG):
            (verdict, reason, model), opinion = await self._run_cancellable(
                _both(reviewing, self._guard_verdict(turn, tool, args, history)))
        else:
            verdict, reason, model = await self._run_cancellable(reviewing)
        review: dict[str, Any] = {"verdict": verdict, "reason": reason, "model": model}
        if opinion is not None:
            review["guard"] = opinion
        if advisory:
            answered = verdict != "ask" or (opinion is not None and opinion["verdict"] != "ask")
            return review if answered else None
        await self._emit(AutoReview(call_id=call.id, tool=tool.name, verdict=verdict,
                                    reason=reason, reviewer_model=model, guard=opinion,
                                    helper=self._helper or ""))
        return review

    async def _reviewer_verdict(self, turn: _Turn, call: ToolCall, tool: Tool, args: dict[str, Any],
                                info: PermissionInfo | None, title: str,
                                noted: Sequence[str] = ()) -> tuple[str, str, str]:
        """``(verdict, reason, model)`` from the first reviewer that can be acquired."""
        verdict, reason, model = "ask", autoreview.REASON_NO_MODEL, ""
        for key in self._reviewer_keys():
            try:
                verdict, reason = await self._review_with(turn, key, call, tool, args, info, title,
                                                          noted)
            except _ReviewerUnavailable as exc:
                log.info("auto-mode reviewer %s unavailable: %s", key, exc)
                continue
            model = key
            break
        return verdict, reason, model

    async def _guard_verdict(self, turn: _Turn, tool: Tool, args: dict[str, Any],
                             history: Sequence[str] = ()) -> dict[str, Any]:
        """The local guard's ``{verdict, reason, model}`` on one call — fail-closed.

        Anything short of a clear answer (no local model, an engine that will
        not start, an error, a timeout, an unclear reply) is ``ask``, which
        sends the call to the user.
        """
        command = command_from_args(args) if tool.name == "bash" else None
        action = guard.action_text(args, command)
        shell_cwd = self._tool_ctx.state.get("bash_cwd") if self._tool_ctx is not None else None
        facts = guard.risk_facts(command=command, action=action, project=str(self.cwd),
                                 cwd=str(shell_cwd) if shell_cwd else None,
                                 outside=tool.kind != "exec")
        if not facts:
            # Nothing risky by our own checks: the reviewer decides alone, and
            # the guard is left for the rare call that is — so a refusal from
            # it is worth reading, not something to click through.
            return {"verdict": "allow", "reason": guard.REASON_NO_RISK, "model": guard.RULES_MODEL}
        key = guard.pick_guard_model(self.settings, self.runtime, self._model_key)
        if key is None:
            log.info("auto-mode guard: no local model for %s", self._model_key)
            return {"verdict": "ask", "reason": guard.REASON_NO_GUARD, "model": ""}
        holder = _lease(self.runtime, key, purpose="agent")
        try:
            lease = await holder.__aenter__()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - unknown model, does not fit, will not start
            log.info("auto-mode guard %s unavailable: %s", key, exc)
            return {"verdict": "ask", "model": key, "reason": guard.REASON_ERROR.format(
                detail=_short(str(exc) or type(exc).__name__, 120))}
        try:
            verdict, reason = await asyncio.wait_for(
                self._guard_on(lease, turn, tool.name, action, history,
                               task=guard.task_text(turn.user_text),
                               claim=guard.claim_text(args.get("description"), action),
                               facts=facts),
                guard.GUARD_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            verdict, reason = "ask", guard.REASON_TIMEOUT.format(seconds=int(guard.GUARD_TIMEOUT))
        except Exception as exc:  # noqa: BLE001 - an engine error must not approve anything
            log.warning("auto-mode guard on %s failed: %s", key, exc)
            verdict, reason = "ask", guard.REASON_ERROR.format(
                detail=_short(str(exc) or type(exc).__name__, 120))
        finally:
            await holder.__aexit__(None, None, None)
        return {"verdict": verdict, "reason": reason, "model": key}

    async def _guard_on(self, lease: Any, turn: _Turn, tool_name: str, action: str,
                        history: Sequence[str] = (), *, task: str = "",
                        claim: str = "", facts: Sequence[str] = ()) -> tuple[str, str]:
        """Render the guard's two messages, generate on ``lease``, parse the verdict.

        No conversation, no tools, thinking off where the template allows it —
        the prompt is :func:`guard.guard_messages` and nothing else; ``history``
        is the user's past answers (keys our parser computed), never model text.
        """
        engine = lease.engine
        formatter = (self._formatter if self._formatter is not None and self._formatter_for is engine
                     else self._guard_formatter(engine))
        messages = guard.guard_messages(tool=tool_name, action=action, cwd=str(self.cwd),
                                        history=history, task=task, claim=claim, facts=facts)
        levels = formatter.effort_levels()
        budget = (guard.GUARD_THINKING_TOKENS if levels and "off" not in levels
                  else guard.GUARD_TOKENS)
        answer = await self._guard_generate(lease, turn, formatter, messages, budget)
        found = autoreview.parse_verdict(answer)
        if found is None and answer.strip():
            # A small model often explains itself and never writes the word, or
            # runs out of tokens just before it: ask once more, for the word only.
            again = [*messages, {"role": "assistant", "content": answer},
                     {"role": "user", "content": guard.ONE_WORD}]
            word = await self._guard_generate(lease, turn, formatter, again, guard.ONE_WORD_TOKENS)
            found = autoreview.parse_verdict(word)
            if found is not None:
                reason = autoreview.parse_reason(answer)
                found = (found[0], reason or found[1])
        if found is None:
            return "ask", guard.REASON_UNCLEAR
        verdict, reason = found
        return verdict, reason or guard.REASON_MISSING

    async def _guard_generate(self, lease: Any, turn: _Turn, formatter: ChatFormatter,
                              messages: list[dict[str, Any]], budget: int) -> str:
        """One guard completion, thinking off where the template allows it; the visible text."""
        engine = lease.engine
        prompt = await formatter.render(messages, None, thinking=False, effort="off")
        if formatter.starts_in_reasoning(prompt):
            budget = max(budget, guard.GUARD_THINKING_TOKENS)
        params = SamplingParams(max_tokens=budget, temperature=0.0, stop=formatter.stop_strings())
        parts: list[str] = []
        stats: GenStats | None = None
        agen = engine.complete(prompt, params)
        try:
            async for chunk in agen:
                if chunk.text:
                    parts.append(chunk.text)
                if chunk.done:
                    stats = chunk.stats
        finally:
            await _aclose(agen)
        if stats is not None:
            self._add_stats(turn, stats)
            lease.record_stats(stats)
        return formatter.parse("".join(parts), prompt, None).content or ""

    def _guard_formatter(self, engine: Any) -> ChatFormatter:
        cached = self._guard_fmt
        if cached is None or cached[0] is not engine:
            cached = (engine, ChatFormatter(engine))
            self._guard_fmt = cached
        return cached[1]

    async def _review_with(self, turn: _Turn, key: str, call: ToolCall, tool: Tool,
                           args: dict[str, Any], info: PermissionInfo | None,
                           title: str, noted: Sequence[str] = ()) -> tuple[str, str]:
        """One review on model ``key``: ``(verdict, reason)``.

        Raises ``_ReviewerUnavailable`` only when the model cannot be acquired;
        anything that goes wrong once it is in hand is a verdict of ``ask``.
        """
        own = key == self._model_key
        if not own and not key.startswith("remote:"):
            checker = getattr(self.runtime, "is_loaded", None)
            with contextlib.suppress(Exception):
                if callable(checker) and not checker(key):
                    await self._emit(Status(message="Ładowanie modelu recenzenta…", phase="loading"))
        holder = _lease(self.runtime, key, purpose="agent")
        try:
            lease = await holder.__aenter__()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # unknown model, does not fit, will not start
            raise _ReviewerUnavailable(str(exc)) from exc
        try:
            return await asyncio.wait_for(
                self._review_on(lease, own, turn, call, tool, args, info, title, noted),
                autoreview.REVIEW_TIMEOUT)
        except (asyncio.TimeoutError, TimeoutError):
            return "ask", autoreview.REASON_TIMEOUT.format(seconds=int(autoreview.REVIEW_TIMEOUT))
        except Exception as exc:  # noqa: BLE001 - an engine error must not decide for the user
            log.warning("auto-mode review on %s failed: %s", key, exc)
            return "ask", autoreview.REASON_ERROR.format(
                detail=_short(str(exc) or type(exc).__name__, 120))
        finally:
            await holder.__aexit__(None, None, None)

    async def _review_on(self, lease: Any, own: bool, turn: _Turn, call: ToolCall, tool: Tool,
                         args: dict[str, Any], info: PermissionInfo | None,
                         title: str, noted: Sequence[str] = ()) -> tuple[str, str]:
        """Render the review prompt, generate the verdict on ``lease``, parse it.

        ``noted``: the user's past answers that match this call (``agent/precedents.py``).
        """
        engine = lease.engine
        formatter = self._get_formatter(engine) if own else self._reviewer_formatter(engine)
        remote = getattr(engine, "format", "") == "remote"
        before, assistant, after = autoreview.split_at_call(self._prompt_messages(), call.id)
        # The step's own tool list keeps the prefix the engine has cached. A
        # different local reviewer does without (the schemas cost its window); a
        # remote one keeps them: providers refuse tool calls in the history of a
        # request that defines no tools.
        tools = (self._schemas or None) if (own or remote) else None
        levels = formatter.effort_levels()
        budget = (autoreview.REVIEW_THINKING_TOKENS if remote or (levels and "off" not in levels)
                  else autoreview.REVIEW_TOKENS)
        n_ctx = int(getattr(engine, "n_ctx", 0) or 8192)
        text = autoreview.review_message(
            tool=tool.name,
            title=(info.title if info is not None else title),
            preview=(info.preview if info is not None else ""),
            arguments=args,
            assistant_text=str((assistant or {}).get("content") or ""),
            others=autoreview.sibling_lines(assistant, after, call.id),
            n_ctx=n_ctx,
            precedents=noted,
        )
        if not own:
            before = autoreview.fit_window(before, n_ctx, budget + len(text) // 3 + 64)
        prefix, images = self._media_for(engine, self._with_language_reminder(before))
        thinking, effort = await self._review_effort(formatter, prefix, tools, own)
        if thinking:
            budget = autoreview.REVIEW_THINKING_TOKENS
        messages = [*prefix, {"role": "user", "content": text}]
        prompt = await formatter.render(messages, tools, thinking=thinking, effort=effort)
        if formatter.starts_in_reasoning(prompt):
            budget = autoreview.REVIEW_THINKING_TOKENS
        params = SamplingParams(max_tokens=budget, temperature=0.0,
                                stop=formatter.stop_strings(), images=list(images),
                                cache_key=self._cache_key)    # same prefix as the step: same cache
        parts: list[str] = []
        stats: GenStats | None = None
        agen = engine.complete(prompt, params)
        try:
            async for chunk in agen:
                if chunk.text:
                    parts.append(chunk.text)
                if chunk.done:
                    stats = chunk.stats
        finally:
            await _aclose(agen)
        if stats is not None:
            self._add_stats(turn, stats)
            lease.record_stats(stats)
        parsed = formatter.parse("".join(parts), prompt, tools)
        if parsed.tool_calls:
            return "ask", autoreview.REASON_TOOL_CALL
        found = autoreview.parse_verdict(parsed.content)
        if found is None:
            return "ask", autoreview.REASON_UNCLEAR
        verdict, reason = found
        return verdict, reason or autoreview.REASON_MISSING

    async def _review_effort(self, formatter: ChatFormatter, prefix: list[dict[str, Any]],
                             tools: list[dict[str, Any]] | None, own: bool) -> tuple[bool, str]:
        """``(thinking, effort)`` for a review: off, unless that costs the prefix cache.

        Qwen3's switch only touches the generation prompt, but gpt-oss writes
        its level into the system header — reviewing with "off" there would
        re-read the whole conversation. The agent's own model then reviews
        with the step's setting (and a bigger output budget).
        """
        step_thinking = bool(self.settings.agent.thinking)
        step_effort = self._engine_effort(step_thinking)
        if not own or step_effort == "off" or not formatter.effort_levels():
            return False, "off"
        try:
            kept = await formatter.render(prefix, tools, thinking=step_thinking, effort=step_effort,
                                          add_generation_prompt=False)
            off = await formatter.render(prefix, tools, thinking=False, effort="off",
                                         add_generation_prompt=False)
        except Exception:  # noqa: BLE001 - a template that cannot end on a user turn
            return False, "off"
        return (False, "off") if kept == off else (step_thinking, step_effort)

    def _reviewer_formatter(self, engine: Any) -> ChatFormatter:
        cached = self._review_fmt
        if cached is None or cached[0] is not engine:
            cached = (engine, ChatFormatter(engine))
            self._review_fmt = cached
        return cached[1]

    async def _ask(self, tool: Tool, args: dict[str, Any], info: PermissionInfo | None, title: str,
                   *, review: dict[str, Any] | None = None) -> str:
        """Ask the user; ``review`` (auto mode) is what the reviewer said, shown on the card.

        A helper's question says whose it is: "Pomocnik h2: …" and ``helper``.
        """
        if self.ask_permission is None:
            return "unavailable"
        shown = info.title if info is not None else title
        if self._helper:
            shown = f"Pomocnik {self._helper}: {shown}"
        request = PermissionRequest(
            request_id=secrets.token_hex(6),
            tool=tool.name,
            title=shown,
            preview=(info.preview if info is not None else _short(json.dumps(args, ensure_ascii=False, default=str), 400)),
            options=(["allow", "deny", "always"] if self._allow_always else ["allow", "deny"]),
            review=dict(review) if review is not None else None,
            helper=self._helper or "",
        )
        await self._emit(request)
        answer = await self._run_cancellable(self._call_approver(request))
        return answer

    async def _call_approver(self, request: PermissionRequest) -> str:
        assert self.ask_permission is not None
        try:
            answer: Any = self.ask_permission(request)
            if inspect.isawaitable(answer):
                answer = await answer
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ask_permission failed")
            return PERMISSION_DISMISSED          # refused, but not by the user
        if answer is True:
            return "allow"
        if answer is False or answer is None:
            return "deny"
        text = str(answer).strip().lower()
        if text == PERMISSION_DISMISSED:
            return PERMISSION_DISMISSED
        if text in ("always", "always_allow", "a", "zawsze"):
            return "always"
        if text in ("allow", "yes", "y", "ok", "tak", "accept", "approve"):
            return "allow"
        return "deny"

    def _todo_items(self, args: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw = self._tool_ctx.state.get("todos")
        if not isinstance(raw, list):
            raw = args.get("todos") or args.get("items") or args.get("todo")
        if not isinstance(raw, list):
            return None
        items: list[dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, dict):
                content = str(entry.get("content") or entry.get("text") or entry.get("task") or "").strip()
                status = _TODO_STATUS.get(str(entry.get("status") or "").strip().lower(), "pending")
            else:
                content, status = str(entry).strip(), "pending"
            if content:
                items.append({"content": content, "status": status})
        return items

    # ------------------------------------------------------------- interrupt

    def _finish_interrupted_generation(self, turn: _Turn) -> None:
        """Keep whatever the model already wrote when the generation was cut."""
        text = turn.text.strip()
        if not text and not turn.reasoning.strip():
            return
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if turn.reasoning:
            message["reasoning"] = turn.reasoning
        items: list[dict[str, Any]] = []
        if turn.reasoning.strip():
            items.append(self._item("reasoning", text=turn.reasoning))
        if text:
            items.append(self._item("assistant", text=turn.text))
        self._append(message, items=items)

    def _close_pending_calls(self, turn: _Turn) -> None:
        """Answer every issued-but-unfinished call so the history stays valid."""
        for call in turn.pending_calls:
            if call.id in turn.answered_calls:
                continue
            self._append(
                {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": INTERRUPTED_RESULT},
                items=[self._item("tool", id=call.id, toolName=call.name, title=call.name,
                                  arguments=call.arguments if isinstance(call.arguments, dict) else {},
                                  ok=False, summary="Przerwano", output=INTERRUPTED_RESULT)],
            )
            turn.answered_calls.add(call.id)
        turn.pending_calls = []

    async def _run_cancellable(self, coro: Any) -> Any:
        """Await ``coro`` in a task that :meth:`interrupt` can cancel."""
        task = asyncio.ensure_future(coro)
        self._current = task
        try:
            return await task
        except asyncio.CancelledError:
            if self._interrupt_event.is_set():
                raise _Interrupted from None
            raise
        finally:
            if self._current is task:
                self._current = None

    # ---------------------------------------------------------------- context

    async def _recover_overflow(self, turn: _Turn, exc: BaseException) -> bool:
        """The engine rejected the prompt: compact hard and retry the step once.

        Raises ``_Interrupted`` when the user stops the agent while it is
        summarising.
        """
        ctx = self._ctx
        if turn.overflow_retries >= 1 or ctx is None:
            n_ctx = ctx.n_ctx if ctx is not None else 0
            await self._emit(ErrorEvent(message=_ERR_OVERFLOW.format(n_ctx=n_ctx)))
            self._note(_ERR_OVERFLOW.format(n_ctx=n_ctx))
            return False
        turn.overflow_retries += 1
        turn.steps = max(0, turn.steps - 1)  # the failed attempt does not count
        log.info("context overflow reported by the engine (%s); compacting", exc)
        try:
            await self._run_cancellable(self._compact_once(aggressive=True))
        except _Interrupted:
            raise
        except Exception:
            log.exception("compaction after overflow failed")
            await self._emit(ErrorEvent(message=_ERR_OVERFLOW.format(n_ctx=ctx.n_ctx)))
            return False
        return True

    async def _apply_fit(self, fitted: list[dict[str, Any]], compactions: Sequence[Compaction],
                         *, reason: str = "compaction") -> None:
        body = fitted[1:] if fitted and fitted[0].get("role") == "system" else list(fitted)
        changed = bool(compactions) or len(body) != len(self._history) or any(
            a is not b for a, b in zip(body, self._history)
        )
        self._history = body
        if not changed:
            return
        if self.session is not None:
            self.session.replace_history(body, reason=reason)
        for event in compactions:
            await self._emit(event)
            self._note(f"Kompaktowanie kontekstu: {event.strategy} ({event.before} → {event.after} tokenów).")

    def _with_language_reminder(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The rendered messages, with "(Odpowiedz po polsku.)" after the latest user message.

        Only for small local models, and only in what is sent — the history and
        the session file keep the user's own words.
        """
        reminder = prompts.LANGUAGE_REMINDERS.get(self._user_language or "")
        if not reminder or self._profile.is_strong:
            return messages
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            out = list(messages)
            if isinstance(content, str):
                out[index] = {**message, "content": f"{content}\n\n{reminder}"}
            elif isinstance(content, list):
                out[index] = {**message, "content": [*content, {"type": "text", "text": reminder}]}
            else:
                return messages
            return out
        return messages

    @staticmethod
    def _media_for(engine: Any, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        """Messages as this engine can take them, and the images that go along.

        A model behind an API key that reads images gets the OpenAI content
        parts unchanged. A local vision model (llama.cpp with its projector)
        gets each image as the server's media marker in the text plus the
        base64 data next to the prompt. Anything else gets a note that the
        image was left out — never the bytes as text.
        """
        from localcode.remote.providers import data_url

        if not any(isinstance(m.get("content"), list) for m in messages):
            return messages, []
        sees = bool(getattr(engine, "supports_images", False))
        if sees and getattr(engine, "format", "") == "remote":
            return messages, []
        marker = getattr(engine, "media_marker", None) if sees else None
        out: list[dict[str, Any]] = []
        images: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                out.append(message)
                continue
            pieces: list[str] = []
            for part in content:
                url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) and \
                    isinstance(part.get("image_url"), dict) else None
                image = data_url(str(url)) if url else None
                if image is not None:
                    if marker:
                        pieces.append(marker)
                        images.append(image[1])
                    else:
                        pieces.append("[Obraz pominięty — ten model nie widzi obrazów.]")
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    pieces.append(part["text"])
                elif isinstance(part, str):
                    pieces.append(part)
            out.append({**message, "content": "\n".join(pieces)})
        return out, images

    def _forget_images(self) -> None:
        """After the turn, keep only a note where an image was.

        Megabytes of base64 would otherwise be sent again on every later step
        (and to every later compaction), and written to the session file.
        """
        changed = False
        for index, message in enumerate(self._history):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            texts = [p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)]
            shown = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
            if not shown:
                continue
            note = "[obraz pokazany w tej wiadomości]" if shown == 1 else f"[{shown} obrazy pokazane w tej wiadomości]"
            self._history[index] = {**message, "content": "\n".join([*texts, note]).strip()}
            changed = True
        if changed and self.session is not None:
            with contextlib.suppress(Exception):
                self.session.replace_history(self._history, reason="images")

    def _prompt_messages(self) -> list[dict[str, Any]]:
        system = {"role": "system", "content": self._system or ""}
        return [system] + self._history

    def _get_formatter(self, engine: Any) -> ChatFormatter:
        if self._formatter is None or self._formatter_for is not engine:
            self._formatter = ChatFormatter(engine)
            self._formatter_for = engine
        return self._formatter

    def _get_ctx(self, engine: Any, formatter: ChatFormatter) -> ContextManager:
        n_ctx = int(engine.n_ctx)
        if self._ctx is None or self._ctx_n_ctx != n_ctx:
            thinking = bool(self.settings.agent.thinking)
            effort = self._engine_effort(thinking)

            def counter(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
                # Chat templates take text: a message with pictures is counted
                # as its words plus a flat cost per picture (Qwen-VL wants at
                # least ~1k tokens for one), never fed to the template as a list.
                flat, pictures = _countable(messages)
                counted = formatter.count(flat, tools, thinking=thinking, effort=effort)
                if pictures and isinstance(counted, int):
                    return counted + pictures * IMAGE_TOKENS
                return counted

            profile = profile_for(getattr(engine, "model", None), n_ctx)
            self._profile = profile
            # A strong model waits longer before compacting: its window is huge
            # and every compaction of a remote model is a paid request.
            threshold = (profile.compact_threshold if profile.is_strong
                         else float(self.settings.agent.auto_compact_threshold))
            self._ctx = ContextManager(
                n_ctx,
                counter=counter,
                compact_threshold=threshold,
                reserve_output=profile.reserve_output if profile.is_strong else None,
            )
            self._ctx_n_ctx = n_ctx
        return self._ctx

    def _model_loaded(self) -> bool:
        checker = getattr(self.runtime, "is_loaded", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(self._model_key))
        except Exception:  # pragma: no cover - defensive
            return False

    # ------------------------------------------------------------ tools/prompt

    async def _prepare_tools(self, engine: Any) -> None:
        """Pick the tools and build the system prompt once per session/mode/model."""
        n_ctx = int(engine.n_ctx)
        native = bool(getattr(engine, "template_info", None) and engine.template_info.supports_tools)
        goal_text = self._goal.text if self._goal is not None and self._goal.active else None
        language = self._user_language
        ultracode = self.ultracode
        delegate = self._delegate_on(n_ctx)
        key = (n_ctx, self._mode, native, self._model_key, len(self._skills.all()),
               self._system_suffix, self._system_prompt, goal_text, language, self._task_hint,
               ultracode, delegate)
        if self._system is not None and self._system_key == key:
            return
        # Which squeeze applies: a local model keeps the short menu below 8k,
        # a model behind an API key never gets it (agent/profile.py).
        profile = profile_for(getattr(engine, "model", None), n_ctx)
        self._profile = profile
        # Skills come in two sizes (agent/skills.py). The menu below and every
        # later `skill` call must agree on which one this model gets, so the
        # library is told once, here, where both the profile and the window
        # are known. The prompt cache key already carries n_ctx and the model.
        self._skills.use_for(strong=profile.is_strong, n_ctx=n_ctx)
        has_skills = bool(self._skills.all())
        self._tools = list(self._resolve_tools(self._mode, profile.tiny_tools, n_ctx,
                                               skills=has_skills, subagents=delegate))
        if goal_text is not None and not any(t.name == GoalDoneTool.name for t in self._tools):
            # Only while a goal is open: a tool nobody may call is context a
            # small model pays for on every step.
            self._tools.append(GoalDoneTool())
        self._schemas = [t.schema() for t in self._tools]
        if self._env is None:
            self._env = await prompts.get_env_async(self.cwd)
        if self._instructions_ctx != n_ctx:
            self._instructions = await asyncio.to_thread(
                prompts.load_project_instructions, self.cwd, prompts.project_instructions_budget(n_ctx)
            )
            self._instructions_ctx = n_ctx
        goal_rule = prompts.GOAL_RULE.format(goal=goal_text) if goal_text is not None else None
        language_line = prompts.LANGUAGE_LINES.get(language or "")
        if self._system_prompt is not None:
            self._system = self._system_prompt
            for extra in (goal_rule, language_line):
                if extra:
                    self._system = f"{self._system}\n\n{extra}"
            self._system_key = key
            return
        # Per-language coding notes for the languages this project uses: a remote/
        # strong model gets the thorough "max" text, a local model the terse "mini"
        # (agent/langskills.py). A small window takes fewer of them.
        lang_notes = await asyncio.to_thread(
            langskills.build_section,
            str(self.cwd),
            strong=profile.is_strong,
            extra_text=self._task_hint or "",
            cap=3 if profile.is_strong else 2,
        )
        self._system = prompts.build_system_prompt(
            cwd=str(self.cwd),
            n_ctx=n_ctx,
            mode=self._mode,
            native_tools=native,
            tool_names=[t.name for t in self._tools],
            project_instructions=self._instructions,
            env=self._env,
            skills=self._skills.menu() if has_skills else None,
            lang_notes=lang_notes,
            ultracode=ultracode,
        )
        if self._system_suffix:
            # A helper's role belongs in the prompt, not in the first user
            # message: the conversation can be compacted away, the prompt
            # cannot. Appended rather than woven in, so the prefix every model
            # caches stays byte-identical between steps.
            self._system = f"{self._system}\n\n{self._system_suffix}"
        for extra in (goal_rule, language_line):
            if extra:
                # Last, after everything else: the end of the prompt is what a
                # small model remembers when it starts to write.
                self._system = f"{self._system}\n\n{extra}"
        self._system_key = key

    def _resolve_tools(self, mode: str, tiny: bool, n_ctx: int, *,
                       skills: bool = False, subagents: bool | None = None) -> Sequence[Tool]:
        if subagents is None:
            subagents = self._subagents
        override = self._tools_override
        if override is not None:
            if callable(override):
                return self._call_tool_factory(override, mode, tiny, n_ctx,
                                               self._manage_models, skills, subagents)
            return list(override)
        try:
            from localcode.agent import tools as tools_pkg
        except ImportError as exc:  # pragma: no cover - package always present
            raise EngineError(_ERR_NO_TOOLS) from exc
        factory = getattr(tools_pkg, "tools_for_mode", None) or getattr(tools_pkg, "default_tools", None)
        if factory is None:
            raise EngineError(_ERR_NO_TOOLS)
        return self._call_tool_factory(factory, mode, tiny, n_ctx, self._manage_models, skills,
                                       subagents)

    def _switch_model(self, key: str) -> None:
        """Continue the conversation on the model the assistant just loaded."""
        if not key or key == self._model_key:
            return
        self._model_key = key
        # The system prompt is built per model (template, tools, context size),
        # so it has to be rebuilt rather than reused.
        self._system = None
        self._system_key = None
        # Recorded like /model, so the app's model menu and a resumed session
        # show the model the conversation is actually on.
        if self.session is not None:
            with contextlib.suppress(Exception):
                self.session.set_model(key)

    @staticmethod
    def _call_tool_factory(factory: Callable[..., Sequence[Tool]], mode: str, tiny: bool,
                           n_ctx: int, manage_models: bool = False,
                           skills: bool = False, subagents: bool = False) -> Sequence[Tool]:
        """Call ``tools_for_mode``/``default_tools`` with whatever it accepts."""
        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            params = {}
        kwargs: dict[str, Any] = {}
        if "tiny" in params:
            kwargs["tiny"] = tiny
        if "n_ctx" in params:
            kwargs["n_ctx"] = n_ctx
        if "manage_models" in params:
            kwargs["manage_models"] = manage_models
        if "skills" in params:
            kwargs["skills"] = skills
        if "subagents" in params:
            kwargs["subagents"] = subagents
        takes_mode = any(name not in ("tiny", "n_ctx") for name in params)
        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((mode,) if takes_mode else (), kwargs), ((mode,), {}), ((), {})
        ]
        last: TypeError | None = None
        for args, kw in attempts:
            try:
                return list(factory(*args, **kw))
            except TypeError as exc:
                last = exc
        raise EngineError(f"Nie można pobrać listy narzędzi agenta: {last}")

    def _tool_by_name(self, name: str) -> Tool | None:
        """Find the tool the model meant (exact name, sloppy casing, known alias).

        Only tools offered in this mode can be reached, so an alias can never
        smuggle ``bash`` into plan mode.
        """
        for tool in self._tools:
            if tool.name == name:
                return tool
        key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_").rsplit(".", 1)[-1]
        if not key:
            return None
        try:
            from localcode.agent.tools import TOOL_ALIASES
        except Exception:  # pragma: no cover - tools package incomplete
            TOOL_ALIASES = {}
        target = TOOL_ALIASES.get(key, key)
        for tool in self._tools:
            if tool.name in (key, target):
                return tool
        return None

    async def _tool_progress(self, text: str) -> None:
        await self._emit(Status(message=_short(text, 160), phase="tool"))

    # ------------------------------------------------------------- history IO

    def _start_user_message(self, user_text: str,
                            images: Sequence[dict[str, Any]] | None = None) -> None:
        text = user_text
        if self._pending_notice:
            text = f"{self._pending_notice}\n\n{user_text}" if user_text.strip() else self._pending_notice
            self._pending_notice = None
        if self.session is not None:
            self.session.title_from(user_text)
        content: Any = text
        names: list[str] = []
        if images:
            # Images first, then the question about them — the order vision
            # models are trained on.
            parts: list[dict[str, Any]] = []
            for image in images:
                data = str(image.get("data") or "")
                if not data:
                    continue
                mime = str(image.get("mime") or "image/png")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
                names.append(str(image.get("name") or "obraz"))
            if parts:
                # Small models otherwise go looking for the file with read_file.
                seen = ", ".join(names)
                many = len(names) > 1
                note = (f"(The attached image{'s' if many else ''} {seen} {'are' if many else 'is'} "
                        f"shown above — look at {'them' if many else 'it'} directly; do not open "
                        f"{'them' if many else 'it'} with tools.)")
                content = [*parts, {"type": "text", "text": f"{text}\n\n{note}"}]
        shown = user_text + "".join(f"\n[obraz: {name}]" for name in names)
        self._append({"role": "user", "content": content},
                     items=[self._item("user", text=shown)])

    def _append_user(self, text: str) -> None:
        """Add an agent-written user message, merging it into the previous one."""
        if self._history and self._history[-1].get("role") == "user":
            previous = self._history[-1]
            merged = dict(previous)
            if isinstance(previous.get("content"), list):
                merged["content"] = [*previous["content"], {"type": "text", "text": text}]
            else:
                merged["content"] = f"{previous.get('content') or ''}\n\n{text}".strip()
            self._history[-1] = merged
            if self.session is not None:
                self.session.replace_history(self._history, reason="nudge")
            return
        self._append({"role": "user", "content": text})

    def _append(self, message: dict[str, Any], *, items: Sequence[dict[str, Any]] = ()) -> None:
        self._history.append(message)
        if self.session is not None:
            self.session.append_many(messages=[message], items=items)

    @property
    def _cache_key(self) -> str:
        """One provider-side prompt cache per session (``SamplingParams.cache_key``)."""
        if self.session is not None and getattr(self.session, "id", None):
            return f"localcode-{self.session.id}"
        if self._own_cache_key is None:
            self._own_cache_key = f"localcode-{secrets.token_hex(8)}"
        return self._own_cache_key

    async def _take_interjections(self) -> None:
        """Put the messages the user typed meanwhile into the history, as users' turns."""
        if self.take_interjections is None:
            return
        try:
            texts = [str(t).strip() for t in self.take_interjections() or ()]
        except Exception:  # noqa: BLE001 - a broken hook must not end the turn
            log.warning("take_interjections failed", exc_info=True)
            return
        for text in filter(None, texts):
            self._append({"role": "user", "content": _INTERJECTION.format(text=text)},
                         items=[self._item("user", text=text)])
            await self._emit(UserMessage(text=text))

    def _take_helper_reports(self) -> bool:
        """Hand the agent the reports of background helpers that finished meanwhile.

        Each becomes a user-role note in the history (merged into one user
        message when several land together) and a notice on the transcript.
        ``True`` when at least one report was handed over.
        """
        from localcode.agent.tools.subagent import report_note

        host = self._subagent_host()
        if host is None:
            return False
        handed = False
        for job in host.take_finished():
            note = report_note(job)
            if note is None:
                continue
            self._append_user(note)
            first = note.split("\n", 2)[1] if note.count("\n") >= 1 else note
            self._note(f"Pomocnik {job.label} skończył w tle: {_short(first, 160)}")
            handed = True
        return handed

    async def _wait_for_helpers(self) -> bool:
        """The model answered, but background helpers still work: wait, then go on.

        Waits (interruptibly) for the first of them to finish, hands over what
        has finished and returns ``True`` — the turn continues with the
        reports, and comes back here if more are still running. ``False``
        when there was nothing to wait for. Raises ``_Interrupted``.
        """
        host = self._subagent_host()
        if host is None or not host.has_background:
            return False
        running = len(host.running)
        if running:
            await self._emit(Status(message=f"Czekam na pomocników w tle ({running})…",
                                    phase="thinking"))
            await self._run_cancellable(host.wait_any())
        return self._take_helper_reports()

    async def _stop_background_helpers(self) -> None:
        """The turn is over: stop any background helper and say what was lost."""
        host = self._subagent_host()
        if host is None:
            return
        lost = await host.stop_background()
        if lost:
            names = ", ".join(job.label for job in lost)
            self._note(f"Pomocnicy w tle zatrzymani, zanim oddali raport: {names}.")

    def _nudge_to_delegate(self, turn: _Turn) -> None:
        """Ultracode: a working agent that has not used a helper for a while is reminded, once.

        Only after :data:`ULTRACODE_NUDGE_STEPS` steps in a row with tool calls
        and no ``delegate``, never while helpers already work in the
        background, and at most :data:`ULTRACODE_MAX_NUDGES` times a turn.
        """
        if not self.ultracode or turn.quiet_steps < ULTRACODE_NUDGE_STEPS:
            return
        if turn.nudges >= ULTRACODE_MAX_NUDGES or not any(t.name == "delegate" for t in self._tools):
            return
        host = self._subagent_host()
        if host is not None and host.running:
            return
        self._append_user(prompts.ULTRACODE_NUDGE.format(n=turn.quiet_steps))
        turn.quiet_steps = 0
        turn.nudges += 1

    def _note(self, text: str) -> None:
        """Record a notice display item (never sent to the model)."""
        if self.session is not None and text:
            self.session.append_item(self._item("notice", text=text))

    @staticmethod
    def _item(kind: str, **fields: Any) -> dict[str, Any]:
        item: dict[str, Any] = {"kind": kind, "id": fields.pop("id", None) or secrets.token_hex(8),
                                "createdAt": now_ms()}
        for key, value in fields.items():
            if value is not None:
                item[key] = value
        return item

    async def _emit(self, event: Event) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("agent event callback failed for %s", type(event).__name__)


async def _both(first: Any, second: Any) -> tuple[Any, Any]:
    """Await two coroutines side by side; cancelling this cancels both."""
    a, b = await asyncio.gather(first, second)
    return a, b


async def _gated(gate: asyncio.Semaphore, tool: Tool, args: dict[str, Any],
                 ctx: ToolContext) -> ToolResult:
    """Run ``tool`` once ``gate`` has a place (ultracode's parallel helpers).

    The coroutine is made inside the gate, so a helper cancelled while still
    waiting leaves no never-awaited coroutine behind.
    """
    async with gate:
        return await tool.run(args, ctx)


class _lease:
    """``async with _lease(runtime, key, purpose=…) as lease`` for any runtime shape.

    ``RuntimeManager.acquire`` returns a ``PendingLease`` (awaitable *and* async
    context manager); test doubles sometimes return a plain coroutine or an
    object with ``release()``. All three work here.
    """

    __slots__ = ("_runtime", "_key", "_purpose", "_n_ctx", "_obj", "_lease")

    def __init__(self, runtime: Any, model_key: str, *, purpose: str = "agent",
                 n_ctx: int | None = None) -> None:
        self._runtime = runtime
        self._key = model_key
        self._purpose = purpose
        self._n_ctx = n_ctx
        self._obj: Any = None
        self._lease: Any = None

    async def __aenter__(self) -> Any:
        kwargs: dict[str, Any] = {"purpose": self._purpose}
        if self._n_ctx is not None:
            kwargs["n_ctx"] = self._n_ctx
        obj = self._runtime.acquire(self._key, **kwargs)
        if inspect.isawaitable(obj) and not hasattr(obj, "__aenter__"):
            obj = await obj
        self._obj = obj
        if hasattr(obj, "__aenter__"):
            self._lease = await obj.__aenter__()
        else:
            self._lease = obj
        return self._lease

    async def __aexit__(self, *exc: Any) -> None:
        obj, lease = self._obj, self._lease
        self._obj = self._lease = None
        try:
            if obj is not None and hasattr(obj, "__aexit__"):
                await obj.__aexit__(None, None, None)
            elif lease is not None:
                release = getattr(lease, "release", None)
                if callable(release):
                    result = release()
                    if inspect.isawaitable(result):
                        await result
        except Exception:  # pragma: no cover - releasing must not mask errors
            log.warning("releasing the runtime lease failed", exc_info=True)
