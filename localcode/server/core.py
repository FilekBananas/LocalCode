"""Process-wide singletons of the LocalCode server.

:class:`Core` owns everything that must exist exactly once in the server
process and wires the components together:

``library``   :class:`~localcode.hub.library.ModelLibrary` — models on disk
``hub``       :class:`~localcode.hub.search.HubClient` — Hugging Face search
``downloads`` ``hub.download.DownloadManager`` — background downloads
``runtime``   :class:`~localcode.runtime.manager.RuntimeManager` — engines
``images``    ``images.runtime.ImageRuntime`` — image models (optional extra)
``gallery``   :class:`~localcode.images.gallery.ImageGallery` — generated images
``repo``/``chat``/``chat_api`` — the chatbot (SQLite + ai_ux REST handlers)
``agents``    :class:`AgentHub` — coding-agent sessions and their WebSocket clients
``events``    :class:`EventBus` — fan-out for ``/api/events`` (SSE)

Everything that changes shared state publishes an event on the bus:
``{"type": "runtime", …status}`` from the runtime manager,
``{"type": "download", "job": {…}}`` from the download manager and
``{"type": "library"}`` whenever the model list changed (a finished download,
a rescan, a deletion).

The modules written by other engineers (``hub.download``, ``agent.loop``) are
imported lazily so the server starts — and its tests run — before they land;
their absence degrades into a Polish 503 on the endpoints that need them.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import inspect
import logging
import time
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from localcode import config as lc_config
from localcode.agent.tools.goal import notice_text as goal_notice_text
from localcode.security.cerber import cerber_installed
from localcode.chat.api import ChatApi
from localcode.chat.db import ChatDB
from localcode.chat.repo import ApiError, ChatRepo
from localcode.chat.service import ChatService
from localcode.config import VERSION, Settings, install_command
from localcode.events import PERMISSION_DISMISSED
from localcode.hub.library import ModelLibrary
from localcode.hub.search import HubClient
from localcode.hub.types import LocalModel
from localcode.runtime.manager import RuntimeManager
from localcode.runtime.templates import (
    AGENT_EFFORT_CHOICES,
    EFFORT_CHOICES,
    model_effort_levels,
    normalize_agent_effort,
    normalize_effort,
)

try:  # written in parallel; the agent endpoints degrade gracefully without it
    from localcode.agent import session as agent_sessions
except Exception:  # noqa: BLE001 - pragma: no cover
    agent_sessions = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

__all__ = ["AgentHub", "AgentSessionState", "Core", "EventBus", "Subscription", "now_ms"]

#: Events kept per SSE subscriber before the slowest ones start losing the oldest.
EVENT_QUEUE_SIZE = 256
#: Decisions a client may send for a permission request.
PERMISSION_DECISIONS = ("allow", "deny", "always")
#: Agent modes accepted by ``setMode`` / ``POST /api/agent/sessions``.
#: The model a provider's key brings in as the default (see ``set_remote_key``).
PREFERRED_REMOTE_MODELS: dict[str, str] = {
    "deepinfra": "remote:deepinfra:deepseek-ai/DeepSeek-V4.1-Flash",
}
AGENT_MODES = ("default", "accept_edits", "auto", "yolo", "plan")


def now_ms() -> int:
    return int(time.time() * 1000)


async def maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def as_dict(event: Any) -> dict[str, Any]:
    """Normalise an ``events.Event`` (or a plain dict) into a JSON-able dict."""
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    if isinstance(event, dict):
        return dict(event)
    raise TypeError(f"nieobsługiwany typ zdarzenia: {type(event)!r}")


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #


class Subscription:
    """One SSE listener. Slow listeners lose the oldest events, never block."""

    def __init__(self, bus: "EventBus", maxsize: int = EVENT_QUEUE_SIZE) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize)
        self.dropped = 0

    def offer(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(event)

    def close(self) -> None:
        self._bus.unsubscribe(self)
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)

    async def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Next event, or ``None`` on close / timeout (the caller sends a heartbeat)."""
        if timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event


class EventBus:
    """Fan-out of server events to every ``/api/events`` listener.

    :meth:`publish` is synchronous and safe to call from any thread: when it is
    called off the event loop the delivery is bounced through
    ``loop.call_soon_threadsafe``.
    """

    def __init__(self, maxsize: int = EVENT_QUEUE_SIZE) -> None:
        self._subs: set[Subscription] = set()
        self._maxsize = maxsize
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def bind(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()

    @property
    def subscribers(self) -> int:
        return len(self._subs)

    def subscribe(self) -> Subscription:
        sub = Subscription(self, self._maxsize)
        self._subs.add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    def publish(self, event: Any) -> None:
        if self._closed:
            return
        payload = as_dict(event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._deliver, payload)
            return
        self._deliver(payload)

    def _deliver(self, payload: dict[str, Any]) -> None:
        for sub in list(self._subs):
            sub.offer(payload)

    def close(self) -> None:
        self._closed = True
        for sub in list(self._subs):
            sub.close()
        self._subs.clear()


# --------------------------------------------------------------------------- #
# Agent sessions
# --------------------------------------------------------------------------- #


class AgentClient:
    """What :class:`AgentSessionState` needs from an attached WebSocket."""

    def send_event(self, payload: dict[str, Any]) -> None:  # pragma: no cover - protocol
        raise NotImplementedError


class _MemorySession:
    """Stand-in for ``agent.session.AgentSession`` when that module is unavailable."""

    def __init__(self, session_id: str, cwd: str, model_key: str | None, mode: str) -> None:
        self.id = session_id
        self.cwd = cwd
        self.model_key = model_key
        self.mode = mode
        self.title = ""
        self.created_at = now_ms()
        self.updated_at = self.created_at
        self.context_used = 0
        self.context_limit = 0
        self.busy = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "cwd": self.cwd, "modelKey": self.model_key, "mode": self.mode,
                "title": self.title, "createdAt": self.created_at, "updatedAt": self.updated_at,
                "contextUsed": self.context_used, "contextLimit": self.context_limit,
                "busy": self.busy}


class _MemoryStore:
    """Minimal :class:`~localcode.agent.session.SessionStore` replacement (no disk)."""

    def __init__(self, cwd: str, model_key: str | None, mode: str) -> None:
        self.session = _MemorySession(uuid.uuid4().hex, cwd, model_key, mode)
        self.messages: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []

    def set_model(self, model_key: str | None) -> None:
        self.session.model_key = model_key

    def set_mode(self, mode: str) -> None:
        self.session.mode = mode

    def set_title(self, title: str) -> None:
        self.session.title = title

    def title_from(self, user_text: str) -> None:
        if not self.session.title:
            self.session.title = _short_title(user_text)

    def delete(self) -> bool:
        return True


class AgentSessionState:
    """One coding-agent session: its :class:`Agent`, transcript and clients.

    The transcript and the header live in a
    :class:`~localcode.agent.session.SessionStore` (append-only JSONL), which is
    what makes sessions survive a server restart; this class adds what only the
    server needs: the attached WebSocket clients, the queue of user messages
    that arrive while a turn runs, and the pending permission questions.

    Several clients (terminal TUI and the Mac app) may watch one session; every
    event is broadcast to all of them, a permission request is answered by
    whichever client replies first and is re-sent to clients that connect while
    it is still pending.
    """

    def __init__(self, hub: "AgentHub", store: Any) -> None:
        self.hub = hub
        self.store = store
        self.todos: list[dict[str, Any]] = []
        self.clients: set[AgentClient] = set()
        self.pending: dict[str, asyncio.Future[str]] = {}
        self.pending_events: dict[str, dict[str, Any]] = {}
        #: Text, or (text, images) for a message that carries pictures.
        self.queued: collections.deque[Any] = collections.deque()
        self.agent: Any | None = None
        self.context_used = 0
        self.context_limit = 0
        self._busy = False
        self._turn_task: asyncio.Task[Any] | None = None
        self._items: list[dict[str, Any]] = list(getattr(store, "items", ()))
        self._items_from = 0
        self._assistant: dict[str, Any] | None = None
        self._reasoning: dict[str, Any] | None = None
        # Per-session reasoning-effort override; None = follow the settings.
        self._effort: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def id(self) -> str:
        return self.store.session.id

    @property
    def cwd(self) -> str:
        return self.store.session.cwd

    @property
    def model_key(self) -> str | None:
        return self.store.session.model_key

    @property
    def mode(self) -> str:
        return self.store.session.mode

    @property
    def busy(self) -> bool:
        return self._busy or bool(getattr(self.agent, "busy", False))

    @property
    def effort(self) -> str:
        """Requested reasoning effort: session override, else the settings."""
        if self._effort is not None:
            return self._effort
        agent_effort = getattr(self.agent, "effort", None)
        if isinstance(agent_effort, str) and agent_effort:
            return agent_effort
        settings = getattr(self.hub.core, "settings", None)
        return normalize_agent_effort(getattr(getattr(settings, "agent", None), "reasoning_effort", None),
                                      "medium") or "medium"

    @property
    def effort_levels(self) -> list[str]:
        """Levels this session's model can express; empty = hide the control."""
        levels = getattr(self.agent, "effort_levels", None)
        if levels:
            return list(levels)
        model = self.hub.core.resolve_model(self.model_key or "", remote=True)
        return model_effort_levels(model) if model is not None else []

    @property
    def updated_at(self) -> int:
        return self.store.session.updated_at

    @property
    def _agent_owns_store(self) -> bool:
        """True when the agent itself writes the transcript to our store."""
        return self.agent is not None and getattr(self.agent, "session", None) is self.store

    @property
    def items(self) -> list[dict[str, Any]]:
        """``AgentItem`` transcript (the store's, or ours for storeless agents)."""
        source = self.store.items if self._agent_owns_store else self._items
        return list(source[self._items_from:])

    def dto(self) -> dict[str, Any]:
        """``AgentSession`` DTO (camelCase, see docs/ARCHITECTURE.md)."""
        session = self.store.session
        session.context_used = self.context_used
        session.context_limit = self.context_limit
        session.busy = self.busy
        dto = session.to_dict()
        dto["effort"] = self.effort
        dto["effortLevels"] = self.effort_levels
        dto["goal"] = getattr(self.agent, "goal", None) if self.agent is not None else None
        return dto

    def snapshot(self) -> dict[str, Any]:
        """``{session, items, todos}`` for ``GET /api/agent/sessions/{id}``."""
        return {"session": self.dto(), "items": self.items, "todos": list(self.todos)}

    def touch(self) -> None:
        self.store.session.updated_at = now_ms()

    # --------------------------------------------------------------- clients
    def attach(self, client: AgentClient) -> None:
        """Register a client and hand it the session, its history and open questions."""
        self.clients.add(client)
        client.send_event({"type": "session", "session": self.dto()})
        client.send_event({"type": "history", "items": self.items})
        if self.todos:
            client.send_event({"type": "todos", "items": list(self.todos)})
        for payload in list(self.pending_events.values()):
            client.send_event(payload)

    def detach(self, client: AgentClient) -> None:
        self.clients.discard(client)

    def broadcast(self, payload: dict[str, Any]) -> None:
        for client in list(self.clients):
            try:
                client.send_event(payload)
            except Exception:  # noqa: BLE001 - a broken client must not break the agent
                log.warning("nie udało się wysłać zdarzenia do klienta sesji %s", self.id, exc_info=True)

    def broadcast_session(self) -> None:
        self.broadcast({"type": "session", "session": self.dto()})

    def notice(self, message: str, *, phase: str = "info") -> None:
        """Emit (and record) a server-side notice, e.g. a queued user message."""
        self.on_event({"type": "status", "message": message, "phase": phase})

    # ---------------------------------------------------------------- events
    def on_event(self, event: Any) -> None:
        """``Agent.on_event`` — update the transcript and fan out to all clients."""
        try:
            payload = as_dict(event)
        except TypeError:
            log.warning("pominięto nieznane zdarzenie agenta: %r", event)
            return
        self._reduce(payload)
        self.broadcast(payload)

    def _item(self, kind: str, **fields_: Any) -> dict[str, Any]:
        """Record a display item — only for agents that do not persist their own."""
        item = {"kind": kind, "id": uuid.uuid4().hex, "createdAt": now_ms(), **fields_}
        if not self._agent_owns_store:
            self._items.append(item)
        return item

    def _reduce(self, payload: dict[str, Any]) -> None:  # noqa: C901 - a flat dispatch table
        kind = payload.get("type")
        self.touch()
        if kind == "turn_start":
            self._busy = True
            self._assistant = self._reasoning = None
            text = str(payload.get("userText") or "")
            self._item("user", text=text)
            if text.strip() and not self.store.session.title:
                self.store.session.title = _short_title(text)
            self.broadcast_session()
        elif kind == "text_delta":
            if self._assistant is None:
                self._assistant = self._item("assistant", text="")
            self._assistant["text"] = self._assistant.get("text", "") + str(payload.get("text") or "")
        elif kind == "reasoning_delta":
            if self._reasoning is None:
                self._reasoning = self._item("reasoning", text="")
            self._reasoning["text"] = self._reasoning.get("text", "") + str(payload.get("text") or "")
        elif kind == "tool_call":
            self._assistant = self._reasoning = None
            self._item("tool", id=str(payload.get("callId") or uuid.uuid4().hex),
                       toolName=payload.get("name"), title=payload.get("title"),
                       arguments=payload.get("arguments") or {})
        elif kind == "tool_result":
            self._assistant = self._reasoning = None
            if not self._agent_owns_store:
                call_id = str(payload.get("callId") or "")
                item = next((i for i in reversed(self._items)
                             if i["kind"] == "tool" and i.get("id") == call_id), None)
                if item is None:
                    item = self._item("tool", id=call_id or uuid.uuid4().hex, toolName=payload.get("name"))
                item.update(ok=bool(payload.get("ok")), summary=payload.get("summary"),
                            output=payload.get("output"), diff=payload.get("diff"))
        elif kind == "auto_review":
            # The verdict belongs to its tool card (an agent with a session
            # file stores it there itself, as ``review``).
            if not self._agent_owns_store:
                call_id = str(payload.get("callId") or "")
                item = next((i for i in reversed(self._items)
                             if i["kind"] == "tool" and i.get("id") == call_id), None)
                if item is not None:
                    item["review"] = {"verdict": payload.get("verdict"), "reason": payload.get("reason"),
                                      "model": payload.get("reviewerModel")}
                    if isinstance(payload.get("guard"), dict):
                        item["review"]["guard"] = payload["guard"]
        elif kind == "permission_request":
            request_id = str(payload.get("requestId") or "")
            if request_id:
                self.pending_events[request_id] = payload
        elif kind == "context":
            self.context_used = int(payload.get("used") or 0)
            self.context_limit = int(payload.get("limit") or 0)
        elif kind == "todos":
            self.todos = list(payload.get("items") or [])
        elif kind in ("status", "compaction", "error"):
            self._item("notice", text=_notice_text(payload), noticeType=kind)
        elif kind == "goal":
            self._item("notice", text=goal_notice_text(payload), noticeType="goal")
            self.broadcast_session()
        elif kind == "turn_end":
            self._busy = False
            self._assistant = self._reasoning = None
            self.broadcast_session()

    # ------------------------------------------------------------------ turns
    async def ensure_agent(self) -> Any:
        if self.agent is None:
            self.agent = await self.hub.build_agent(self)
            if self._effort is not None:
                with contextlib.suppress(Exception):
                    await maybe_await(self.agent.set_effort(self._effort))
            # /goal keeps the agent going on its own; a message the user typed
            # meanwhile waits in ``queued`` and should not wait for twenty more
            # rounds of it.
            with contextlib.suppress(Exception):
                self.agent.goal_should_yield = lambda: bool(self.queued)
            # …and outside /goal the agent reads it at its next step.
            with contextlib.suppress(Exception):
                self.agent.take_interjections = self._take_queued_texts
        return self.agent

    def _take_queued_texts(self) -> list[str]:
        """The queued messages the running turn can take now: text ones, from the
        front. A message with pictures waits for its own turn, and so does all
        that came after it — the order the user typed them in is kept."""
        texts: list[str] = []
        while self.queued and isinstance(self.queued[0], str):
            texts.append(self.queued.popleft())
        return texts

    async def start_goal(self, goal: str, message: str | None = None) -> str:
        """``/goal <cel>``: set the goal, then send it (or ``message``) as a turn."""
        goal = (goal or "").strip()
        if not goal:
            raise ApiError(400, "Cel nie może być pusty.", "empty_goal")
        agent = await self.ensure_agent()
        if not hasattr(agent, "set_goal"):
            raise ApiError(409, "Ten agent nie obsługuje /goal.", "goal_unsupported")
        await maybe_await(agent.set_goal(goal))
        self.broadcast_session()
        return await self.send_user(message or goal)

    async def clear_goal(self) -> None:
        if self.agent is not None and hasattr(self.agent, "clear_goal"):
            await maybe_await(self.agent.clear_goal())
        self.broadcast_session()

    async def send_user(self, text: str, images: list[dict[str, Any]] | None = None) -> str:
        """Start a turn, or queue the message when one is already running."""
        text = (text or "").strip()
        if not text and not images:
            raise ApiError(400, "Wiadomość nie może być pusta.", "empty_message")
        text = self._expand_attachments(text)
        if self.busy or self._turn_task is not None:
            self.queued.append((text, images) if images else text)
            self.notice(f"Wiadomość w kolejce ({len(self.queued)}). Zostanie wysłana po bieżącej odpowiedzi.",
                        phase="queued")
            return "queued"
        await self._start_turn(text, images)
        return "started"

    def _expand_attachments(self, text: str) -> str:
        """Inline every ``@ścieżka`` the client left for us.

        The terminal does this itself before sending, because it shows what got
        attached and what could not be found. A graphical client has no such
        step, so doing it here is what gives both interfaces the same feature
        from one implementation.

        Idempotent on purpose: expansion keeps the original text verbatim and
        appends ``<file …>`` blocks below it, so a message the terminal already
        expanded still carries its ``@`` tokens. Seeing a block means the work
        is done — expanding again would send every file twice.
        """
        if "<file " in text or "@" not in text:
            return text
        try:
            from localcode.tui.completion import expand_at_references  # noqa: PLC0415
        except ImportError:                                            # pragma: no cover
            return text
        try:
            expanded, attachments, warnings = expand_at_references(text, self.cwd)
        except Exception:                                              # noqa: BLE001
            log.warning("nie udało się rozwinąć załączników", exc_info=True)
            return text
        for warning in warnings:
            self.notice(warning)
        if attachments:
            names = ", ".join(a.path for a in attachments)
            self.notice(f"Dołączono: {names}")
        return expanded

    async def _start_turn(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        agent = await self.ensure_agent()
        self._busy = True
        self.touch()
        self.broadcast_session()
        self._turn_task = asyncio.create_task(self._run_turn(agent, text, images),
                                              name=f"agent-turn-{self.id}")

    async def _run_turn(self, agent: Any, text: str, images: list[dict[str, Any]] | None = None) -> None:
        try:
            if images:
                await maybe_await(agent.run_turn(text, images=images))
            else:
                await maybe_await(agent.run_turn(text))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to every client
            log.warning("tura agenta %s zakończona błędem", self.id, exc_info=True)
            self.on_event({"type": "error", "message": str(exc) or "Nieoczekiwany błąd agenta."})
        finally:
            self._turn_task = None
            self._busy = False
            self.touch()
            self.broadcast_session()
            self._fail_pending()
            if self.queued:
                nxt = self.queued.popleft()
                text, images = nxt if isinstance(nxt, tuple) else (nxt, None)
                with contextlib.suppress(Exception):
                    await self._start_turn(text, images)

    # ------------------------------------------------------------ permissions
    async def ask_permission(self, *args: Any, **kwargs: Any) -> str:
        """``Agent.ask_permission`` — ask every client, take the first answer."""
        payload = _permission_payload(args, kwargs)
        request_id = payload["requestId"]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self.pending[request_id] = future
        if request_id not in self.pending_events:
            # The agent did not emit the event itself — publish it now.
            self.on_event(payload)
        if not self.clients:
            self.pending.pop(request_id, None)
            self.pending_events.pop(request_id, None)
            return PERMISSION_DISMISSED          # refused, but nobody decided
        try:
            return await future
        except asyncio.CancelledError:
            raise
        finally:
            self.pending.pop(request_id, None)
            self.pending_events.pop(request_id, None)

    def resolve_permission(self, request_id: str, decision: str) -> bool:
        """Answer a pending request; ``False`` when it is unknown or already answered."""
        future = self.pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(decision)
        self.pending_events.pop(request_id, None)
        self.broadcast({"type": "permission_resolved", "requestId": request_id, "decision": decision})
        return True

    def _fail_pending(self) -> None:
        """Refuse every unanswered permission request (turn over / session closed).

        As ``dismissed``, not ``deny``: the call does not run either way, but
        nobody said no, so the agent learns nothing from it.
        """
        for request_id, future in list(self.pending.items()):
            if not future.done():
                future.set_result(PERMISSION_DISMISSED)
            self.pending.pop(request_id, None)
            self.pending_events.pop(request_id, None)

    # -------------------------------------------------------------- commands
    async def interrupt(self) -> None:
        """Stop the running turn: the agent's own ``interrupt`` first, cancel as a fallback.

        The queue survives: stopping the answer is not taking back what the user
        typed meanwhile — the first queued message starts the next turn as soon
        as this one has wound down (``_run_turn``'s ``finally``).
        """
        self._fail_pending()
        interrupt = getattr(self.agent, "interrupt", None) if self.agent is not None else None
        if callable(interrupt):
            with contextlib.suppress(Exception):
                await maybe_await(interrupt())
            return
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()

    async def set_mode(self, mode: str) -> None:
        if mode not in AGENT_MODES:
            raise ApiError(400, f"Nieznany tryb „{mode}”. Dozwolone: {', '.join(AGENT_MODES)}.", "invalid_mode")
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await maybe_await(self.agent.set_mode(mode))
        if self.store.session.mode != mode:
            with contextlib.suppress(Exception):
                self.store.set_mode(mode)
        self.touch()
        self.broadcast_session()

    async def set_effort(self, effort: str) -> None:
        """Override the reasoning effort for this session (``off|low|medium|high|ultracode``).

        ``ultracode`` is the agent's own level (``runtime/templates.py``): the
        model thinks at "high" and the agent fans work out to parallel helpers.
        """
        level = normalize_agent_effort(effort)
        if level is None:
            raise ApiError(400, f"Nieznany poziom myślenia „{effort}”. "
                                f"Dozwolone: {', '.join(AGENT_EFFORT_CHOICES)}.", "invalid_effort")
        self._effort = level
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await maybe_await(self.agent.set_effort(level))
        self.touch()
        self.broadcast_session()

    async def set_model(self, model_key: str) -> None:
        model = self.hub.core.resolve_model(model_key, remote=True)
        if model is None:
            raise ApiError(404, f"Nie znaleziono modelu „{model_key}”.", "model_not_found")
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await maybe_await(self.agent.set_model(model.key))
        if self.store.session.model_key != model.key:
            with contextlib.suppress(Exception):
                self.store.set_model(model.key)
        self.touch()
        self.broadcast_session()

    async def compact(self) -> None:
        if self.agent is None:
            return
        await maybe_await(self.agent.compact())

    async def clear(self) -> None:
        if self.agent is not None:
            with contextlib.suppress(Exception):
                await maybe_await(self.agent.clear())
        self._items.clear()
        self._items_from = len(getattr(self.store, "items", ()))
        self.todos.clear()
        self.context_used = 0
        self._assistant = self._reasoning = None
        self.touch()
        self.broadcast({"type": "history", "items": []})
        self.broadcast_session()

    async def aclose(self) -> None:
        self.queued.clear()
        self._fail_pending()
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.agent is not None:
            for name in ("aclose", "close", "shutdown"):
                fn = getattr(self.agent, name, None)
                if callable(fn):
                    with contextlib.suppress(Exception):
                        await maybe_await(fn())
                    break
        self.agent = None
        self.clients.clear()


def _short_title(text: str, limit: int = 60) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"


def _notice_text(payload: dict[str, Any]) -> str:
    if payload.get("type") == "compaction":
        return (f"Skrócono kontekst ({payload.get('strategy')}): "
                f"{payload.get('before')} → {payload.get('after')} tokenów.")
    return str(payload.get("message") or "")


def _permission_payload(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build a ``permission_request`` event from whatever the agent passed."""
    candidate: Any = None
    for value in (*args, *kwargs.values()):
        if hasattr(value, "to_dict") or isinstance(value, dict) or hasattr(value, "request_id"):
            candidate = value
            break
    payload: dict[str, Any]
    if candidate is None:
        payload = {}
    else:
        try:
            payload = as_dict(candidate)
        except TypeError:
            payload = {k: getattr(candidate, k, None)
                       for k in ("request_id", "tool", "title", "preview", "options", "review", "helper")}
            payload = {_camel(k): v for k, v in payload.items() if v is not None}
    payload["type"] = "permission_request"
    payload.setdefault("requestId", uuid.uuid4().hex)
    payload.setdefault("tool", kwargs.get("tool") or "")
    payload.setdefault("title", kwargs.get("title") or str(payload.get("tool") or "Operacja"))
    payload.setdefault("preview", kwargs.get("preview") or "")
    payload.setdefault("options", list(PERMISSION_DECISIONS))
    return payload


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p.title() for p in rest)


class AgentHub:
    """Registry of live :class:`AgentSessionState` objects on top of the session files.

    Sessions are stored as JSONL under ``~/.localcode/sessions/`` by
    :mod:`localcode.agent.session`, so listing shows sessions from earlier runs
    too and opening one of them resumes its history.
    """

    def __init__(self, core: "Core", agent_factory: Callable[[AgentSessionState], Any] | None = None,
                 *, sessions_root: str | Path | None = None) -> None:
        self.core = core
        self.agent_factory = agent_factory
        self.sessions_root = sessions_root
        self._sessions: dict[str, AgentSessionState] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------- listing
    def list(self, cwd: str | None = None) -> list[dict[str, Any]]:
        """Live sessions merged with the stored ones, newest first."""
        wanted = _resolved_cwd(cwd) if cwd else None
        live = {s.id: s for s in self._sessions.values()}
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for stored in self._stored_sessions(wanted):
            seen.add(stored.id)
            state = live.get(stored.id)
            out.append(state.dto() if state is not None else stored.to_dict())
        for state in live.values():
            if state.id not in seen and (wanted is None or state.cwd == wanted):
                out.append(state.dto())
        out.sort(key=lambda d: d.get("updatedAt") or 0, reverse=True)
        return out

    def _stored_sessions(self, cwd: str | None) -> list[Any]:
        if agent_sessions is None:
            return []
        try:
            return agent_sessions.list_sessions(cwd, root=self.sessions_root)
        except Exception:  # noqa: BLE001 - a broken session file must not break the list
            log.warning("nie udało się odczytać zapisanych sesji", exc_info=True)
            return []

    # ------------------------------------------------------------- lookup
    def get(self, session_id: str) -> AgentSessionState | None:
        state = self._sessions.get(session_id)
        if state is not None:
            return state
        store = self._resume(session_id)
        if store is None:
            return None
        state = AgentSessionState(self, store)
        self._sessions[state.id] = state
        return state

    def require(self, session_id: str) -> AgentSessionState:
        state = self.get(session_id)
        if state is None:
            raise ApiError(404, "Nie znaleziono sesji agenta.", "not_found")
        return state

    def _resume(self, session_id: str) -> Any | None:
        if agent_sessions is None:
            return None
        try:
            return agent_sessions.SessionStore.resume(session_id, root=self.sessions_root)
        except Exception:  # noqa: BLE001 - unknown or unreadable session
            return None

    # ------------------------------------------------------------- lifecycle
    def create(self, cwd: str | None, model_key: str | None = None, mode: str | None = None) -> AgentSessionState:
        path = Path(cwd or Path.cwd()).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            raise ApiError(400, f"Katalog „{path}” nie istnieje.", "bad_cwd") from None
        if not resolved.is_dir():
            raise ApiError(400, f"Ścieżka „{resolved}” nie jest katalogiem.", "bad_cwd")
        chosen_mode = mode or self.core.settings.agent.permission_mode or "default"
        if chosen_mode not in AGENT_MODES:
            raise ApiError(400, f"Nieznany tryb „{chosen_mode}”. Dozwolone: {', '.join(AGENT_MODES)}.",
                           "invalid_mode")
        key = model_key or self.core.settings.agent.default_model
        if key:
            model = self.core.resolve_model(key, remote=True)
            if model is None:
                raise ApiError(404, f"Nie znaleziono modelu „{key}”.", "model_not_found")
            key = model.key
        store = self._new_store(str(resolved), key, chosen_mode)
        state = AgentSessionState(self, store)
        self._sessions[state.id] = state
        return state

    def helpbot_session(self, *, new: bool = False) -> AgentSessionState:
        """The help bot's conversation: the latest one, or a fresh one.

        A help bot session is any session working in ``helpbot_dir()`` — see
        :mod:`localcode.helpbot` — so "the latest" survives restarts for free.
        """
        from localcode import helpbot  # noqa: PLC0415

        cwd = str(helpbot.helpbot_dir())
        if not new:
            for row in self.list(cwd):
                state = self.get(str(row.get("id") or ""))
                if state is not None:
                    return state
        key = helpbot.status(self.core).get("modelKey")
        if not key:
            raise ApiError(409, str(helpbot.HelpbotMissing(helpbot.model_key(self.core.settings))),
                           "helpbot_missing")
        return self.create(cwd, model_key=key, mode="default")

    def _new_store(self, cwd: str, model_key: str | None, mode: str) -> Any:
        if agent_sessions is None:
            return _MemoryStore(cwd, model_key, mode)
        try:
            return agent_sessions.SessionStore.create(cwd, model_key=model_key, mode=mode,
                                                      root=self.sessions_root)
        except Exception as exc:  # noqa: BLE001 - fall back to an unsaved session
            log.warning("nie udało się utworzyć pliku sesji: %s", exc)
            return _MemoryStore(cwd, model_key, mode)

    async def delete(self, session_id: str) -> bool:
        state = self._sessions.pop(session_id, None)
        if state is not None:
            await state.aclose()
            with contextlib.suppress(Exception):
                state.store.delete()
            return True
        if agent_sessions is None:
            return False
        try:
            return bool(agent_sessions.delete(session_id, root=self.sessions_root))
        except Exception:  # noqa: BLE001
            return False

    # ---------------------------------------------------------------- agents
    async def build_agent(self, session: AgentSessionState) -> Any:
        if self.agent_factory is not None:
            return await maybe_await(self.agent_factory(session))
        return await maybe_await(self._default_agent(session))

    def _default_agent(self, session: AgentSessionState) -> Any:
        try:
            from localcode.agent.loop import Agent  # noqa: PLC0415 - optional, written in parallel
        except ImportError as exc:  # pragma: no cover - only before agent/loop.py lands
            raise ApiError(503, "Agent kodujący nie jest jeszcze dostępny w tej instalacji.",
                           "agent_unavailable") from exc
        store = session.store if not isinstance(session.store, _MemoryStore) else None
        from localcode import helpbot  # noqa: PLC0415

        if helpbot.is_helpbot_cwd(session.cwd):
            # Same loop, other job: the help bot's prompt, tools and facade.
            try:
                return helpbot.make_agent(self.core, on_event=session.on_event,
                                          ask_permission=session.ask_permission, session=store)
            except helpbot.HelpbotMissing as exc:
                raise ApiError(409, str(exc), "helpbot_missing") from exc
        model_key = session.model_key or self.core.settings.agent.default_model
        if not model_key:
            raise ApiError(400, "Nie wybrano modelu dla sesji agenta.", "model_required")
        try:
            return Agent(self.core.runtime, model_key, session.cwd, self.core.settings,
                         session.on_event, session.ask_permission, session=store)
        except TypeError:  # a differently shaped Agent (tests, future versions)
            return Agent(self.core.runtime, model_key, session.cwd, self.core.settings,
                         session.on_event, session.ask_permission)

    async def shutdown(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        for session in sessions:
            with contextlib.suppress(Exception):
                await session.aclose()


def _resolved_cwd(cwd: str) -> str:
    try:
        return str(Path(cwd).expanduser().resolve())
    except OSError:
        return str(Path(cwd).expanduser())


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #


class Core:
    """Everything the HTTP layer talks to, created once per server process."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        db_path: str | Path | None = None,
        library: Any | None = None,
        hub: Any | None = None,
        downloads: Any | None = None,
        runtime: Any | None = None,
        images: Any | None = None,
        gallery: Any | None = None,
        engine_factory: Any | None = None,
        agent_factory: Callable[[AgentSessionState], Any] | None = None,
        sessions_root: str | Path | None = None,
        version: str = VERSION,
    ) -> None:
        self.settings = settings if settings is not None else lc_config.load_settings()
        self.version = version
        self.started_at = now_ms()
        self.events = EventBus()
        self.library: Any = library if library is not None else ModelLibrary(self.settings)
        self.hub: Any = hub if hub is not None else HubClient(self.settings)
        self.runtime: Any = runtime if runtime is not None else RuntimeManager(
            self.settings, self.library, on_event=self.events.publish, engine_factory=engine_factory)
        self._downloads: Any | None = downloads
        self._downloads_built = downloads is not None
        self._downloads_notify_complete = False
        self._gallery: Any | None = gallery
        self._images: Any | None = images
        self._images_built = images is not None
        self.db = ChatDB(db_path)
        self.repo = ChatRepo(self.db)
        self.chat = ChatService(self.repo, self.runtime, self.library,
                                settings_provider=lambda: self.settings)
        self.chat_api = ChatApi(self.repo, self.chat)
        self.agents = AgentHub(self, agent_factory, sessions_root=sessions_root)
        self._started = False
        self._closed = False

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        """Open the database, scan the library, run chat housekeeping. Idempotent."""
        if self._started:
            return
        self._started = True
        self.events.bind()
        await asyncio.to_thread(lc_config.ensure_dirs, self.settings)
        await asyncio.to_thread(self.db.open)
        try:
            await asyncio.to_thread(self.library.scan)
        except Exception:  # noqa: BLE001 - an unreadable model must not stop the server
            log.warning("skanowanie biblioteki modeli nie powiodło się", exc_info=True)
        with contextlib.suppress(Exception):
            await self.chat_api.startup()
        # Touch the download manager so queued jobs (if any) are restored early.
        self.downloads  # noqa: B018 - property with a side effect (lazy construction)

    async def shutdown(self) -> None:
        """Stop every engine, cancel downloads, close agent sessions and the DB."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self.agents.shutdown()
        if self._downloads is not None:
            await self._call_downloads("shutdown", optional=True)
        with contextlib.suppress(Exception):
            await self.chat.aclose()
        if self._images is not None:
            with contextlib.suppress(Exception):
                await maybe_await(self._images.shutdown())
        if self._gallery is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._gallery.close)
        with contextlib.suppress(Exception):
            await self.runtime.shutdown()
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.repo.close)
        self.events.close()

    # ------------------------------------------------------------------ events
    def broadcast(self, event: Any) -> None:
        self.events.publish(event)

    def broadcast_library(self) -> None:
        self.events.publish({"type": "library"})

    # --------------------------------------------------------------- downloads
    @property
    def downloads(self) -> Any | None:
        """The :class:`DownloadManager`, or ``None`` when ``hub.download`` is missing."""
        if not self._downloads_built:
            self._downloads_built = True
            self._downloads = self._build_downloads()
        return self._downloads

    def require_downloads(self) -> Any:
        manager = self.downloads
        if manager is None:
            raise ApiError(503, "Moduł pobierania modeli nie jest jeszcze dostępny.", "downloads_unavailable")
        return manager

    def _build_downloads(self) -> Any | None:
        try:
            from localcode.hub.download import DownloadManager  # noqa: PLC0415 - written in parallel
        except ImportError:
            log.info("localcode.hub.download jeszcze nie istnieje — pobieranie modeli wyłączone")
            return None
        available = {"settings": self.settings, "on_event": self._on_download_event,
                     "on_complete": self._on_download_complete,
                     "library": self.library, "hub": self.hub}
        try:
            params = inspect.signature(DownloadManager).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            params = {}
        kwargs = {name: value for name, value in available.items() if name in params}
        self._downloads_notify_complete = "on_complete" in kwargs
        try:
            if kwargs:
                return DownloadManager(**kwargs)
            return DownloadManager(self.settings, self._on_download_event)
        except Exception:  # noqa: BLE001 - never block server start on this
            log.warning("nie udało się utworzyć menedżera pobierania", exc_info=True)
            return None

    def _on_download_event(self, event: Any) -> None:
        """Forward a download event; rescan the library when a job finishes.

        The rescan normally comes from the manager's ``on_complete`` hook; the
        state check here is the fallback for managers without that hook.
        """
        try:
            payload = as_dict(event)
        except TypeError:
            return
        payload.setdefault("type", "download")
        self.events.publish(payload)
        if self._downloads_notify_complete:
            return
        job = payload.get("job") if isinstance(payload.get("job"), dict) else None
        if (job or {}).get("state") == "done":
            self._schedule_rescan()

    def _on_download_complete(self, job: dict[str, Any]) -> Any:
        """``DownloadManager.on_complete`` — rescan so the new model shows up."""
        log.info("pobrano model %s (%s)", (job or {}).get("modelKey"), (job or {}).get("repo"))
        return self.rescan_library()

    def _schedule_rescan(self) -> None:
        loop = getattr(self.events, "_loop", None)
        coro = self.rescan_library()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if loop is None:
                coro.close()
                return
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro))
            return
        asyncio.ensure_future(coro)

    async def rescan_library(self) -> list[LocalModel]:
        """Rescan the models directory and tell every client the library changed."""
        try:
            models = await asyncio.to_thread(self.library.scan)
        except Exception:  # noqa: BLE001 - reported, never fatal
            log.warning("skanowanie biblioteki nie powiodło się", exc_info=True)
            return list(self.library.list())
        self.broadcast_library()
        return list(models)

    async def _call_downloads(self, name: str, *args: Any, optional: bool = False, **kwargs: Any) -> Any:
        manager = self.downloads if optional else self.require_downloads()
        if manager is None:
            return None
        fn = getattr(manager, name, None)
        if fn is None:
            if optional:
                return None
            raise ApiError(503, f"Menedżer pobierania nie obsługuje operacji „{name}”.", "downloads_unavailable")
        try:
            return await maybe_await(fn(*args, **kwargs))
        except Exception:
            if optional:
                log.warning("download manager.%s zakończone błędem", name, exc_info=True)
                return None
            raise

    # ------------------------------------------------------------------ images
    @property
    def gallery(self) -> Any:
        """The :class:`~localcode.images.gallery.ImageGallery` (SQLite + files).

        Browsing the gallery never needs the optional image extra, so this is
        always available; it is opened on first use.
        """
        if self._gallery is None:
            from localcode.images.gallery import ImageGallery  # noqa: PLC0415 - lazy

            self._gallery = ImageGallery(self.settings).open()
        return self._gallery

    @property
    def images(self) -> Any | None:
        """The ``ImageRuntime``, or ``None`` without ``localcode[images]``."""
        if not self._images_built:
            self._images_built = True
            self._images = self._build_images()
        return self._images

    def require_images(self) -> Any:
        runtime = self.images
        if runtime is None:
            raise ApiError(503, "Generowanie obrazów nie jest dostępne w tej instalacji. "
                                f"Zainstaluj dodatek: {install_command('images')}",
                           "images_unavailable")
        return runtime

    def _build_images(self) -> Any | None:
        try:
            from localcode.images.runtime import ImageRuntime  # noqa: PLC0415 - optional extra
        except ImportError:
            log.info("localcode.images.runtime niedostępny — generowanie obrazów wyłączone")
            return None
        available = {"settings": self.settings, "library": self.library,
                     "on_event": self._on_image_event, "runtime": self.runtime}
        try:
            params = inspect.signature(ImageRuntime).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            params = {}
        kwargs = {name: value for name, value in available.items() if name in params}
        if "gallery" in params:            # only build the gallery when it is wanted
            kwargs["gallery"] = self.gallery
        try:
            if kwargs:
                return ImageRuntime(**kwargs)
            return ImageRuntime(self.settings, self.library, self._on_image_event)
        except Exception:  # noqa: BLE001 - never block server start on this
            log.warning("nie udało się utworzyć środowiska obrazów", exc_info=True)
            return None

    def _on_image_event(self, event: Any) -> None:
        """Forward an image-runtime event; a bare status signal is expanded here."""
        try:
            payload = as_dict(event)
        except TypeError:
            return
        kind = payload.get("type")
        if kind in (None, "", "image_runtime") or (kind == "runtime" and "instances" not in payload):
            self.events.publish({"type": "runtime", **self.runtime_status()})
            return
        self.events.publish(payload)

    def runtime_status(self) -> dict[str, Any]:
        """``RuntimeManager.status()`` with the image instances merged in.

        Image models share the RAM budget with the LLMs, so they belong in the
        same list; their DTOs carry ``"kind": "image"`` (see ``docs/IMAGES.md``).
        """
        runtime = self._images
        if runtime is None:
            return self.runtime.status()
        merged = getattr(runtime, "runtime_status", None)
        if callable(merged):
            try:
                return dict(merged())
            except Exception:  # noqa: BLE001 - a broken image runtime must not break status
                log.warning("nie udało się odczytać stanu środowiska obrazów", exc_info=True)
        status = self.runtime.status()
        try:
            extra = list(runtime.status() or [])
        except Exception:  # noqa: BLE001
            log.warning("nie udało się odczytać instancji obrazów", exc_info=True)
            return status
        status["instances"] = list(status.get("instances") or []) + extra
        return status

    # ----------------------------------------------------------------- models
    def resolve_model(self, query: str | None, *, remote: bool = False) -> LocalModel | None:
        """A model by exact key, then by the library's fuzzy ``resolve``.

        ``remote=True`` also accepts ``remote:<provider>:<model>`` for a
        provider with a saved key — the agent session paths pass it. Loading,
        estimating and tuning do not: a model in somebody's data centre is not
        loaded here, and tuning one would bill the user for it.
        """
        if not query:
            return None
        from localcode.remote.providers import parse_key

        parsed = parse_key(query)
        if parsed is not None:
            if not remote:
                return None
            registry = getattr(self.runtime, "remote", None)
            try:
                if registry is None or not registry.configured(parsed[0]):
                    return None
            except Exception:  # noqa: BLE001 - a broken registry means "not available"
                return None
            from localcode.remote.engine import remote_model

            return remote_model(query)
        model = self.library.get(query)
        if model is not None:
            return model
        resolve = getattr(self.library, "resolve", None)
        return resolve(query) if callable(resolve) else None

    def library_dtos(self) -> list[dict[str, Any]]:
        """``LocalModel.to_dict()`` + ``loaded`` + ``loadDefaults`` for every model."""
        try:
            loaded = set(self.runtime.loaded_keys())
        except Exception:  # noqa: BLE001 - status must never fail on this
            loaded = set()
        out: list[dict[str, Any]] = []
        for model in self.library.list():
            dto = model.to_dict()
            dto["loaded"] = model.key in loaded
            dto["effortLevels"] = model_effort_levels(model)
            try:
                dto["loadDefaults"] = self.library.load_defaults(model.key)
            except Exception:  # noqa: BLE001
                dto["loadDefaults"] = None
            out.append(dto)
        return out

    # --------------------------------------------------------------- settings
    def app_settings(self) -> dict[str, Any]:
        """``GET /api/app-settings`` DTO (the HF token is never sent back)."""
        s = self.settings
        return {
            "modelsDir": s.models_dir,
            "hfTokenSet": bool(s.hf_token),
            "language": s.language,
            "scanLmstudio": s.scan_lmstudio,
            "scanHfCache": s.scan_hf_cache,
            "favoriteModels": list(s.favorite_models),
            "version": self.version,
            "agent": {
                "defaultModel": s.agent.default_model,
                "permissionMode": s.agent.permission_mode,
                "maxSteps": s.agent.max_steps,
                "thinking": s.agent.thinking,
                "temperature": s.agent.temperature,
                "maxOutputTokens": s.agent.max_output_tokens,
                "autoCompactThreshold": s.agent.auto_compact_threshold,
                "showReasoning": s.agent.show_reasoning,
                "manageModels": s.agent.manage_models,
                "reasoningEffort": s.agent.reasoning_effort,
                "subagents": s.agent.subagents,
                "subagentModel": s.agent.subagent_model,
                "subagentMaxSteps": s.agent.subagent_max_steps,
                "ultracodeParallel": s.agent.ultracode_parallel,
                "autoReviewerModel": s.agent.auto_reviewer_model,
                "autoGuardModel": s.agent.auto_guard_model,
                "autoPrecedentsEnabled": s.agent.auto_precedents_enabled,
            },
            "chat": {
                "defaultModel": s.chat.default_model,
                "embeddingModel": s.chat.embedding_model,
                "reasoningEffort": s.chat.reasoning_effort,
            },
            "store": {
                "quality": s.store.quality,
                "preferFormat": s.store.prefer_format,
                "starterModel": s.store.starter_model,
                "autoInstallStarter": s.store.auto_install_starter,
            },
            "security": {
                "sandbox": s.security.sandbox,
                "scanDownloads": s.security.scan_downloads,
            },
            "helpbot": {
                "model": s.helpbot.model,
                "autoDownload": s.helpbot.auto_download,
            },
            "cerber": {
                "enabled": s.cerber.enabled,
                "url": s.cerber.url,
                # The token itself never leaves the server, like every key.
                "tokenSet": bool(s.cerber.token),
                "installed": cerber_installed(),
                "block": list(s.cerber.block),
            },
            "server": {
                "host": s.server.host,
                "port": s.server.port,
                "openaiApiEnabled": s.server.openai_api_enabled,
                "openaiApiRequiresToken": s.server.openai_api_requires_token,
                "corsOrigins": list(s.server.cors_origins),
            },
        }

    async def update_app_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate a camelCase patch, persist it and apply it in place."""
        snake = _app_settings_patch(patch)
        await self._persist(snake)
        if "models_dir" in snake:
            await self.rescan_library()
        return self.app_settings()

    async def set_favorite_model(self, key: Any, favorite: Any) -> dict[str, Any]:
        """Star (or unstar) one model; the others keep their place in the list.

        One key at a time rather than the whole list, so the Mac app and the
        terminal starring different models at once do not undo each other.
        A key that is not installed is still accepted: a remote model's list
        may be down right now, and a deleted model's star harms nobody.
        """
        if not isinstance(key, str) or not key.strip():
            raise ApiError(400, "Oczekiwano klucza modelu w polu „key”.", "missing_key")
        if not isinstance(favorite, bool):
            raise ApiError(400, "Pole „favorite” musi być wartością logiczną.", "validation_error")
        try:
            merged = await asyncio.to_thread(lc_config.set_favorite_model, key.strip(), favorite)
        except OSError as exc:
            raise ApiError(500, f"Nie udało się zapisać ustawień: {exc.strerror or exc}.", "settings_write") from exc
        _copy_dataclass(merged, self.settings)
        return self.app_settings()

    async def set_remote_key(self, provider: str, key: str | None, *,
                             base_url: str | None = None) -> None:
        """Save (or clear) one provider's API key.

        A provider with a model we recommend (``PREFERRED_REMOTE_MODELS``)
        makes it the default for new agent sessions and chats — unless the
        user already chose a remote default, which stays.
        """
        patch: dict[str, Any] = {"api_key": key}
        if base_url is not None:
            patch["base_url"] = base_url or None
        update: dict[str, Any] = {"remote": {provider: patch}}
        preferred = PREFERRED_REMOTE_MODELS.get(provider)
        if key and preferred:
            settings = self.settings
            if not str(settings.agent.default_model or "").startswith("remote:"):
                update["agent"] = {"default_model": preferred}
            if not str(getattr(settings.chat, "default_model", "") or "").startswith("remote:"):
                update["chat"] = {"default_model": preferred}
        await self._persist(update)

    async def set_remote_enabled(self, provider: str, enabled: bool) -> None:
        await self.set_remote_options(provider, enabled=enabled)

    async def set_remote_options(self, provider: str, *, enabled: bool | None = None,
                                 service_tier: str | None = None) -> None:
        """Switch a provider on/off and/or set its service tier, in one write.

        ``None`` leaves a field as it is. The tier is expected validated
        (``routes_remote``); one this version does not know loads as standard.
        """
        patch: dict[str, Any] = {}
        if enabled is not None:
            patch["enabled"] = bool(enabled)
        if service_tier is not None:
            patch["service_tier"] = service_tier
        if patch:
            await self._persist({"remote": {provider: patch}})

    async def remote_library_dtos(self) -> list[dict[str, Any]]:
        """Borrowed models as library rows (empty when no key is configured)."""
        registry = getattr(self.runtime, "remote", None)
        if registry is None or not registry.providers:
            return []
        try:
            models = await registry.local_models()
        except Exception:  # noqa: BLE001 - a provider being down is not an error here
            return []
        return [{**m.to_dict(), "remote": True, "loaded": False} for m in models]

    async def set_cerber_token(self, token: str | None) -> dict[str, Any]:
        """Save (or clear) Cerber's API token; a wrong token is refused if Cerber can tell.

        Cerber may be closed right now — then the token is saved anyway (the
        check runs on every download), and only an explicit "wrong token" from
        Cerber itself stops it.
        """
        if token is not None and not isinstance(token, str):
            raise ApiError(400, "Token musi być tekstem albo null.", "invalid_token")
        value = (token or "").strip() or None
        if value:
            from localcode.security.cerber import CerberClient, CerberError

            try:
                await CerberClient(self.settings.cerber.url, value).info()
            except CerberError as exc:
                if exc.code == "cerber_bad_token":
                    raise ApiError(400, exc.message, exc.code) from exc
        await self._persist({"cerber": {"token": value}})
        return self.app_settings()

    async def check_cerber(self) -> dict[str, Any]:
        from localcode.security.cerber import check_connection

        return await check_connection(self.settings)

    async def set_hf_token(self, token: str | None) -> dict[str, Any]:
        if token is not None and not isinstance(token, str):
            raise ApiError(400, "Token musi być tekstem albo null.", "invalid_token")
        value = (token or "").strip() or None
        await self._persist({"hf_token": value})
        return self.app_settings()

    async def _persist(self, patch: dict[str, Any]) -> None:
        try:
            merged = await asyncio.to_thread(lc_config.update_settings, patch)
        except OSError as exc:
            raise ApiError(500, f"Nie udało się zapisać ustawień: {exc.strerror or exc}.", "settings_write") from exc
        _copy_dataclass(merged, self.settings)


def _copy_dataclass(src: Any, dst: Any) -> None:
    """Copy every field of ``src`` into ``dst`` in place (nested dataclasses too).

    The ``Settings`` object is shared by reference with the runtime manager, the
    library and the hub client, so settings changes must mutate it rather than
    replace it.
    """
    for f in fields(src):
        value = getattr(src, f.name)
        current = getattr(dst, f.name)
        if is_dataclass(current) and is_dataclass(value):
            _copy_dataclass(value, current)
        else:
            setattr(dst, f.name, value)


_AGENT_KEYS = {
    "defaultModel": ("default_model", str, True),
    "permissionMode": ("permission_mode", "mode", False),
    "maxSteps": ("max_steps", int, False),
    "thinking": ("thinking", bool, False),
    "temperature": ("temperature", float, False),
    "maxOutputTokens": ("max_output_tokens", int, True),
    "autoCompactThreshold": ("auto_compact_threshold", float, False),
    "showReasoning": ("show_reasoning", bool, False),
    "manageModels": ("manage_models", bool, False),
    "reasoningEffort": ("reasoning_effort", "agent_effort", False),
    "subagents": ("subagents", bool, False),
    # Only the user picks the helper's model: a model that could choose it would
    # be able to post this project's files to somebody else's API.
    "subagentModel": ("subagent_model", str, True),
    "subagentMaxSteps": ("subagent_max_steps", int, False),
    "ultracodeParallel": ("ultracode_parallel", int, False),
    # Auto mode's reviewer; null = the agent's own model. Only the user picks
    # it, for the same reason as the helper's: it reads the conversation.
    "autoReviewerModel": ("auto_reviewer_model", str, True),
    # Auto mode's context-free guard (agent/guard.py); null = automatic. Local
    # models only: it exists so that a remote model cannot approve its own
    # commands, and it must not send them anywhere either.
    "autoGuardModel": ("auto_guard_model", "local_model", True),
    # Learn from the user's answers on permission cards (agent/precedents.py).
    "autoPrecedentsEnabled": ("auto_precedents_enabled", bool, False),
}
_CHAT_KEYS = {
    "defaultModel": ("default_model", str, True),
    "embeddingModel": ("embedding_model", str, True),
    "reasoningEffort": ("reasoning_effort", "effort", False),
}
_STORE_KEYS = {
    "quality": ("quality", "quality", False),
    "preferFormat": ("prefer_format", "format", False),
    "starterModel": ("starter_model", str, True),
    "autoInstallStarter": ("auto_install_starter", bool, False),
}
_SERVER_KEYS = {
    "openaiApiEnabled": ("openai_api_enabled", bool, False),
    "openaiApiRequiresToken": ("openai_api_requires_token", bool, False),
    "corsOrigins": ("cors_origins", "origins", False),
    "port": ("port", int, False),
}
_SECURITY_KEYS = {
    "sandbox": ("sandbox", "sandbox_mode", False),
    "scanDownloads": ("scan_downloads", bool, False),
}
_HELPBOT_KEYS = {
    "model": ("model", str, False),
    "autoDownload": ("auto_download", bool, False),
}
_CERBER_KEYS = {
    "enabled": ("enabled", bool, False),
    "url": ("url", str, False),
}
_TOP_KEYS = {
    "modelsDir": ("models_dir", str, False),
    "language": ("language", str, False),
    "scanLmstudio": ("scan_lmstudio", bool, False),
    "scanHfCache": ("scan_hf_cache", bool, False),
    "favoriteModels": ("favorite_models", "keys", False),
}


def _app_settings_patch(patch: Any) -> dict[str, Any]:
    """camelCase ``/api/app-settings`` patch → snake_case ``config`` patch."""
    if not isinstance(patch, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    out: dict[str, Any] = {}
    _section(patch, _TOP_KEYS, out, "")
    for name, table in (("agent", _AGENT_KEYS), ("chat", _CHAT_KEYS),
                        ("store", _STORE_KEYS), ("server", _SERVER_KEYS),
                        ("security", _SECURITY_KEYS), ("helpbot", _HELPBOT_KEYS),
                        ("cerber", _CERBER_KEYS)):
        section = patch.get(name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ApiError(400, f"Sekcja „{name}” musi być obiektem.", "validation_error")
        nested: dict[str, Any] = {}
        _section(section, table, nested, f"{name}.")
        if nested:
            out[name] = nested
    if isinstance(patch.get("cerber"), dict) and "token" in patch["cerber"]:
        raise ApiError(400, "Token Cerbera zmienia się przez PUT /api/app-settings/cerber-token.",
                       "use_cerber_token_endpoint")
    if "url" in (out.get("cerber") or {}):
        _check_cerber_url(out["cerber"]["url"])
    if "hfToken" in patch:
        raise ApiError(400, "Token Hugging Face zmienia się przez PUT /api/app-settings/hf-token.",
                       "use_hf_token_endpoint")
    if not out:
        raise ApiError(400, "Brak znanych ustawień do zmiany.", "empty_patch")
    return out


def _check_cerber_url(url: str) -> None:
    """Cerber's API is loopback-only; anything else is a token sent somewhere it should not go."""
    from urllib.parse import urlsplit

    parts = urlsplit(str(url or ""))
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost"):
        raise ApiError(400, "Adres Cerbera musi być lokalny: http://127.0.0.1:PORT albo http://localhost:PORT.",
                       "invalid_cerber_url")


def _section(src: dict[str, Any], table: dict[str, tuple[str, Any, bool]], out: dict[str, Any],
             prefix: str) -> None:
    for camel, (snake, kind, nullable) in table.items():
        if camel not in src:
            continue
        value = src[camel]
        path = prefix + camel
        if value is None:
            if not nullable:
                raise ApiError(400, f"Ustawienie „{path}” nie może być puste.", "validation_error")
            out[snake] = None
            continue
        if kind is bool:
            if not isinstance(value, bool):
                raise ApiError(400, f"Ustawienie „{path}” musi być wartością logiczną.", "validation_error")
            out[snake] = value
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ApiError(400, f"Ustawienie „{path}” musi być liczbą całkowitą.", "validation_error")
            out[snake] = int(value)
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ApiError(400, f"Ustawienie „{path}” musi być liczbą.", "validation_error")
            out[snake] = float(value)
        elif kind is str:
            if not isinstance(value, str):
                raise ApiError(400, f"Ustawienie „{path}” musi być tekstem.", "validation_error")
            out[snake] = value
        elif kind == "effort":
            level = normalize_effort(value)
            if level is None:
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    f"{', '.join(EFFORT_CHOICES)}.", "validation_error")
            out[snake] = level
        elif kind == "local_model":
            if not isinstance(value, str) or not value.strip():
                raise ApiError(400, f"Ustawienie „{path}” musi być kluczem modelu.", "validation_error")
            if value.strip().startswith("remote:"):
                raise ApiError(400, "Strażnik trybu Auto musi być modelem lokalnym — model zdalny "
                                    "wysyłałby sprawdzane polecenia poza tego Maca.", "validation_error")
            out[snake] = value.strip()
        elif kind == "agent_effort":
            # The agent's default may be ultracode; the chat's maps it to "high".
            level = normalize_agent_effort(value)
            if level is None:
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    f"{', '.join(AGENT_EFFORT_CHOICES)}.", "validation_error")
            out[snake] = level
        elif kind == "quality":
            from localcode.hub.catalog import QUALITY_LEVELS

            if value not in QUALITY_LEVELS:
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    f"{', '.join(QUALITY_LEVELS)}.", "validation_error")
            out[snake] = value
        elif kind == "format":
            if value not in ("mlx", "gguf", "auto"):
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    "mlx, gguf, auto.", "validation_error")
            out[snake] = value
        elif kind == "mode":
            if value not in AGENT_MODES:
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    f"{', '.join(AGENT_MODES)}.", "validation_error")
            out[snake] = value
        elif kind == "sandbox_mode":
            from localcode.security.sandbox import MODES as SANDBOX_MODES

            if value not in SANDBOX_MODES:
                raise ApiError(400, f"Ustawienie „{path}” musi być jedną z wartości: "
                                    f"{', '.join(SANDBOX_MODES)}.", "validation_error")
            out[snake] = value
        elif kind == "keys":
            if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
                raise ApiError(400, f"Ustawienie „{path}” musi być listą kluczy modeli.", "validation_error")
            # Duplicates dropped, first occurrence wins: the order is the user's.
            out[snake] = list(dict.fromkeys(v.strip() for v in value))
        elif kind == "origins":
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ApiError(400, f"Ustawienie „{path}” musi być listą adresów.", "validation_error")
            out[snake] = list(value)
