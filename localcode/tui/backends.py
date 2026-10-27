"""Two ways to drive the coding agent from the terminal — one interface.

``LocalBackend``
    Everything runs in this process: a :class:`localcode.server.core.Core`
    (library, hub, downloads, runtime manager, chat) plus a
    :class:`localcode.agent.loop.Agent` with a JSONL session store.

``RemoteBackend``
    ``~/.localcode/server.json`` points at a running ``localcode serve`` (the
    Mac app starts one), so the models it already holds in memory are reused
    instead of being loaded a second time.  The agent runs *in the server*; we
    talk to it over ``POST /api/agent/sessions`` + the session WebSocket and
    reach the rest of the app through the documented HTTP API.

Both push the very same event stream — the camelCase dicts of
:mod:`localcode.events` — into :attr:`Backend.events`, and both answer
permission requests with :meth:`Backend.answer_permission`, so the TUI never
knows which one it is talking to.

    backend = await open_backend(cwd=".", prefer="auto")
    await backend.send_user("napraw testy")
    while True:
        event = await backend.events.get()
        ...

``prefer`` is ``"auto"`` (remote when a healthy server answers, else local),
``"local"`` (``--local``) or ``"remote"`` (``--attach``, an error when no
server is running).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from localcode import config as lc_config
from localcode.config import Settings, load_settings
from localcode.events import PERMISSION_DISMISSED
from localcode.runtime.templates import AGENT_EFFORT_CHOICES, normalize_agent_effort

log = logging.getLogger(__name__)

__all__ = [
    "Backend",
    "BackendError",
    "LocalBackend",
    "RemoteBackend",
    "ServerInfo",
    "discover_server",
    "open_backend",
]

#: How long the health check of a possibly dead server may take.
HEALTH_TIMEOUT_S = 2.0
#: Events buffered before the reader starts waiting for the UI.
EVENT_QUEUE_SIZE = 4096


class BackendError(RuntimeError):
    """Backend could not be created or used (message is Polish, safe to show)."""


# --------------------------------------------------------------------------- #
# Server discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServerInfo:
    """Contents of ``~/.localcode/server.json`` plus what ``/api/health`` said."""

    host: str
    port: int
    token: str
    pid: int | None = None
    version: str | None = None

    @property
    def base_url(self) -> str:
        host = self.host or "127.0.0.1"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def ws_url(self, path: str) -> str:
        return self.base_url.replace("http://", "ws://", 1) + path


def read_server_info(path: Path | None = None) -> ServerInfo | None:
    """Parse ``server.json`` (``None`` when it is missing or unusable)."""
    target = path or lc_config.server_info_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("port"):
        return None
    try:
        port = int(data["port"])
    except (TypeError, ValueError):
        return None
    pid = data.get("pid")
    return ServerInfo(
        host=str(data.get("host") or "127.0.0.1"),
        port=port,
        token=str(data.get("token") or ""),
        pid=int(pid) if isinstance(pid, int) else None,
        version=str(data["version"]) if data.get("version") else None,
    )


async def discover_server(*, path: Path | None = None, timeout: float = HEALTH_TIMEOUT_S) -> ServerInfo | None:
    """``server.json`` + a successful ``GET /api/health``, or ``None``."""
    info = read_server_info(path)
    if info is None:
        return None
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is a hard dependency
        return None
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(f"{info.base_url}/api/health") as response:
                if response.status != 200:
                    return None
                payload = await response.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    version = payload.get("version")
    return ServerInfo(info.host, info.port, info.token, info.pid,
                      str(version) if version else info.version)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class Backend:
    """What the TUI and the CLI need from "the app", local or remote."""

    kind: str = "local"

    def __init__(self, cwd: str | os.PathLike[str], settings: Settings | None = None) -> None:
        self.cwd = Path(cwd).expanduser().resolve()
        self.settings = settings if settings is not None else load_settings()
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue(EVENT_QUEUE_SIZE)
        self.model_key: str | None = None
        self.mode: str = self.settings.agent.permission_mode
        # Requested reasoning effort ("off"|"low"|"medium"|"high"|"ultracode");
        # models whose chat template has no control ignore the thinking part.
        self.effort: str = normalize_agent_effort(self.settings.agent.reasoning_effort,
                                                  "medium") or "medium"
        self.session_id: str | None = None
        self.context: dict[str, Any] | None = None
        self.busy: bool = False
        #: The session's ``/goal`` as the last ``goal`` event described it.
        self.goal: dict[str, Any] | None = None
        self._models: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError

    async def __aenter__(self) -> "Backend":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------ the agent
    async def send_user(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        """Send a message; ``images`` = ``[{name, mime, data (base64)}]`` for a vision model."""
        raise NotImplementedError

    async def interrupt(self) -> None:
        raise NotImplementedError

    async def set_mode(self, mode: str) -> None:
        raise NotImplementedError

    async def set_model(self, key: str) -> None:
        raise NotImplementedError

    async def set_effort(self, level: str) -> None:
        """Set the reasoning effort for this session (``off|low|medium|high|ultracode``)."""
        value = normalize_agent_effort(level)
        if value is None:
            raise BackendError(f"Nieznany poziom myślenia „{level}”. "
                               f"Dozwolone: {', '.join(AGENT_EFFORT_CHOICES)}.")
        self.effort = value

    async def effort_levels(self) -> list[str]:
        """Levels the current model supports; empty = hide the control."""
        dto = await self.model()
        levels = (dto or {}).get("effortLevels")
        return [str(x) for x in levels] if isinstance(levels, list) else []

    async def compact(self) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def start_goal(self, goal: str, message: str | None = None) -> None:
        """``/goal <cel>``: set the goal and start working on it (``message`` = the turn's text)."""
        raise NotImplementedError

    async def clear_goal(self) -> None:
        raise NotImplementedError

    async def answer_permission(self, request_id: str, decision: str) -> None:
        raise NotImplementedError

    async def sessions(self, *, limit: int = 30) -> list[dict[str, Any]]:
        return []

    async def resume_session(self, session_id: str) -> list[dict[str, Any]]:
        raise BackendError("Wznawianie sesji nie jest dostępne w tym trybie.")

    # ------------------------------------------------------------- the model
    async def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def model(self, key: str | None = None) -> dict[str, Any] | None:
        """DTO of ``key`` (default: the current model) from the cached library."""
        wanted = key or self.model_key
        if not wanted:
            return None
        models = self._models if self._models is not None else await self.list_models()
        exact = next((m for m in models if m.get("key") == wanted), None)
        if exact is not None:
            return exact
        lowered = wanted.lower()
        hits = [m for m in models if lowered in str(m.get("key", "")).lower()]
        return hits[0] if len(hits) == 1 else None

    async def pick_default_model(self) -> str | None:
        """The model to use when the user named none.

        Preference order: something already in memory (no load time at all),
        then a model that can call tools, then the biggest one that leaves
        room for its KV cache in the RAM budget.
        """
        # Never a remote model on its own: every request to one is billed.
        models = [m for m in await self.list_models()
                  if "embedding" not in (m.get("capabilities") or []) and not m.get("remote")]
        if not models:
            return None
        loaded = [m for m in models if m.get("loaded")]
        if loaded:
            return str(loaded[0]["key"])
        budget = 0
        try:
            status = await self.runtime_status()
            budget = int((status.get("budget") or {}).get("limitBytes") or 0)
        except Exception:  # noqa: BLE001 - fall back to "any model" below
            budget = 0
        comfortable = [m for m in models
                       if not budget or int(m.get("sizeBytes") or 0) <= budget * 0.5]
        with_tools = [m for m in (comfortable or models) if "tools" in (m.get("capabilities") or [])]
        candidates = with_tools or comfortable or models
        candidates.sort(key=lambda m: int(m.get("sizeBytes") or 0), reverse=True)
        return str(candidates[0]["key"])

    async def resolve_model_key(self, query: str) -> str | None:
        """Fuzzy model lookup shared by ``-m``, ``/model`` and ``load``."""
        if not query:
            return None
        models = await self.list_models()
        for model in models:
            if model.get("key") == query:
                return str(model["key"])
        lowered = query.lower()
        hits = [m for m in models if lowered in str(m.get("key", "")).lower()
                or lowered in str(m.get("displayName", "")).lower()]
        if len(hits) == 1:
            return str(hits[0]["key"])
        if hits:
            starts = [m for m in hits if str(m.get("key", "")).lower().split("/")[-1].startswith(lowered)]
            if len(starts) == 1:
                return str(starts[0]["key"])
        return None

    async def runtime_status(self) -> dict[str, Any]:
        raise NotImplementedError

    async def load_model(self, key: str, *, n_ctx: int | None = None,
                         n_parallel: int | None = None, new_instance: bool = False) -> dict[str, Any]:
        raise NotImplementedError

    async def unload(self, instance_id: str | None = None, *, everything: bool = False) -> None:
        raise NotImplementedError

    async def update_runtime_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def estimate(self, key: str, *, n_ctx: int | None = None) -> dict[str, Any]:
        """``MemoryEstimate`` DTO for loading ``key`` right now."""
        raise NotImplementedError

    async def tune_runtime(self, *, model_key: str | None = None, use_model: bool = True,
                           apply: bool = False) -> dict[str, Any]:
        """Recommended runtime settings for this machine (and ``model_key``, if given).

        ``{"recommendation": TuneProposal DTO, "status"?: RuntimeStatus}`` — see
        ``localcode/runtime/tuner.py``. ``status`` is present only when ``apply``
        wrote the proposal through the validated settings path.
        """
        raise NotImplementedError

    async def delete_model(self, key: str) -> None:
        raise NotImplementedError

    async def favorite_models(self) -> list[str]:
        """The models starred as "Ulubione" (in the app or here) — listed first by ``/model``."""
        return list(self.settings.favorite_models)

    # ------------------------------------------------------------ Hugging Face
    async def hub_search(self, query: str, *, fmt: str = "any", limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def hub_curated(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def hub_details(self, repo: str) -> dict[str, Any]:
        raise NotImplementedError

    async def start_download(self, repo: str, *, variant_id: str | None = None,
                             files: Sequence[str] | None = None) -> dict[str, Any]:
        raise NotImplementedError

    async def downloads(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def cancel_download(self, job_id: str) -> None:
        raise NotImplementedError

    # --------------------------------------------------------------- helpers
    def _publish(self, event: Any) -> None:
        """Queue one event (dict or ``events.Event``), dropping the oldest if full."""
        payload: dict[str, Any]
        to_dict = getattr(event, "to_dict", None)
        if callable(to_dict):
            payload = dict(to_dict())
        elif isinstance(event, dict):
            payload = dict(event)
        else:  # pragma: no cover - defensive
            return
        self._absorb(payload)
        try:
            self.events.put_nowait(payload)
        except asyncio.QueueFull:  # pragma: no cover - only a runaway producer
            with contextlib.suppress(asyncio.QueueEmpty):
                self.events.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.events.put_nowait(payload)

    def _absorb(self, payload: dict[str, Any]) -> None:
        """Keep the backend's own view (context, busy, mode) in sync."""
        kind = payload.get("type")
        if kind == "context":
            self.context = dict(payload)
        elif kind == "turn_start":
            self.busy = True
        elif kind == "turn_end":
            self.busy = False
        elif kind == "session":
            session = payload.get("session") or {}
            self.session_id = session.get("id") or self.session_id
            if session.get("modelKey"):
                self.model_key = str(session["modelKey"])
            if session.get("mode"):
                self.mode = str(session["mode"])
            if session.get("contextLimit"):
                self.context = {"type": "context", "used": session.get("contextUsed") or 0,
                                "limit": session.get("contextLimit") or 0}

    @property
    def context_used(self) -> int:
        return int((self.context or {}).get("used") or 0)

    @property
    def context_limit(self) -> int:
        return int((self.context or {}).get("limit") or 0)

    def error(self, message: str) -> None:
        """Inject an error event (used when a command fails outside a turn)."""
        self._publish({"type": "error", "message": message})


# --------------------------------------------------------------------------- #
# Local
# --------------------------------------------------------------------------- #


class LocalBackend(Backend):
    """Runs ``Core`` and the agent loop inside the terminal process."""

    kind = "local"

    def __init__(self, cwd: str | os.PathLike[str], settings: Settings | None = None, *,
                 model_key: str | None = None, mode: str | None = None,
                 n_ctx: int | None = None, session_id: str | None = None,
                 store_session: bool = True, ask_permission: bool = True) -> None:
        super().__init__(cwd, settings)
        self.model_key = model_key
        if mode:
            self.mode = mode
        self.n_ctx = n_ctx
        self._core: Any = None
        self._agent: Any = None
        self._store: Any = None
        self._resume_id = session_id
        self._store_session = store_session
        self._ask = ask_permission
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._early_answers: dict[str, str] = {}
        self._turn: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------- lifecycle
    @property
    def core(self) -> Any:
        if self._core is None:
            raise BackendError("Rdzeń LocalCode nie został uruchomiony.")
        return self._core

    async def start(self) -> None:
        from localcode.server.core import Core

        if self.n_ctx:
            self.settings.runtime.default_ctx = int(self.n_ctx)
        self._core = Core(self.settings)
        await self._core.start()
        if self.model_key:
            resolved = await self.resolve_model_key(self.model_key)
            if resolved is None:
                raise BackendError(f"Nie znaleziono modelu „{self.model_key}”. "
                                   "Sprawdź `localcode models list`.")
            self.model_key = resolved
        elif self.settings.agent.default_model:
            self.model_key = await self.resolve_model_key(self.settings.agent.default_model)
        if self.model_key is None:
            self.model_key = await self.pick_default_model()

    async def aclose(self) -> None:
        task = self._turn
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        for future in self._pending.values():
            if not future.done():
                future.set_result(PERMISSION_DISMISSED)
        self._pending.clear()
        if self._core is not None:
            with contextlib.suppress(Exception):
                await self._core.shutdown()
        self._core = None
        self._agent = None

    # ----------------------------------------------------------------- agent
    async def _ensure_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        if not self.model_key:
            raise BackendError(
                "Nie wybrano modelu. Pobierz model (`localcode models curated`) "
                "albo wskaż go opcją -m.")
        from localcode.agent.loop import Agent
        from localcode.agent.session import SessionStore

        store = self._store
        # A resumed session is always opened (that is the point of -c/--resume);
        # a new one only when the transcript should be written at all.
        if store is None and (self._resume_id or self._store_session):
            if self._resume_id:
                store = await asyncio.to_thread(SessionStore.resume, self._resume_id)
            else:
                store = await asyncio.to_thread(
                    SessionStore.create, str(self.cwd), model_key=self.model_key, mode=self.mode)
            self._store = store
            self.session_id = store.session.id
        from localcode import helpbot

        if helpbot.is_helpbot_cwd(self.cwd):
            # The help bot's directory means the help bot — the same rule the
            # server uses, so `localcode pomoc --local` and the app agree.
            self._agent = helpbot.make_agent(
                self.core, on_event=self._publish,
                ask_permission=self._ask_permission if self._ask else None, session=store)
        else:
            self._agent = Agent(
                self.core.runtime,
                self.model_key,
                str(self.cwd),
                self.settings,
                on_event=self._publish,
                ask_permission=self._ask_permission if self._ask else None,
                session=store,
            )
        if self._agent.mode != self.mode:
            self._agent.set_mode(self.mode)
        if self._agent.effort != self.effort:
            self._agent.set_effort(self.effort)
        return self._agent

    def _ask_permission(self, request: Any) -> "asyncio.Future[str]":
        request_id = str(getattr(request, "request_id", "") or "")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        if not request_id:  # pragma: no cover - the agent always sets one
            future.set_result(PERMISSION_DISMISSED)
            return future
        early = self._early_answers.pop(request_id, None)
        if early is not None:
            future.set_result(early)
            return future
        self._pending[request_id] = future
        return future

    async def answer_permission(self, request_id: str, decision: str) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(decision)
        elif future is None and request_id:
            # The question is published before the agent starts waiting for
            # it; an answer that arrives in between (a script, a fast client)
            # is kept for the moment the wait begins instead of being lost.
            self._early_answers[request_id] = decision

    async def send_user(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        agent = await self._ensure_agent()
        if self._turn is not None and not self._turn.done():
            raise BackendError("Agent jest zajęty — poczekaj na zakończenie odpowiedzi.")
        self.busy = True
        self._turn = asyncio.create_task(self._run_turn(agent, text, images), name="localcode-turn")

    async def _run_turn(self, agent: Any, text: str, images: list[dict[str, Any]] | None = None) -> None:
        try:
            if images:
                await agent.run_turn(text, images=images)
            else:
                await agent.run_turn(text)
            # The assistant may have moved the conversation to another model
            # (model_load, local or remote); the status line follows it.
            switched = getattr(agent, "model_key", None)
            if switched and switched != self.model_key:
                self.model_key = str(switched)
                self._publish({"type": "session", "session": self._session_dto()})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - shown to the user, never fatal
            log.warning("tura agenta zakończona błędem", exc_info=True)
            self._publish({"type": "error", "message": str(exc) or "Nieoczekiwany błąd agenta."})
            self._publish({"type": "turn_end", "turnId": "", "reason": "error", "stats": {}})
        finally:
            self.busy = False
            for request_id, future in list(self._pending.items()):
                if not future.done():
                    future.set_result(PERMISSION_DISMISSED)
                self._pending.pop(request_id, None)

    async def wait_idle(self) -> None:
        """Await the running turn (headless mode)."""
        task = self._turn
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def interrupt(self) -> None:
        if self._agent is not None:
            self._agent.interrupt()
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_result(PERMISSION_DISMISSED)
            self._pending.pop(request_id, None)

    async def set_mode(self, mode: str) -> None:
        from localcode.agent.permissions import normalize_mode

        self.mode = normalize_mode(mode)
        if self._agent is not None:
            self._agent.set_mode(self.mode)
        self._publish({"type": "session", "session": self._session_dto()})

    async def set_model(self, key: str) -> None:
        resolved = await self.resolve_model_key(key)
        if resolved is None:
            raise BackendError(f"Nie znaleziono modelu „{key}”.")
        self.model_key = resolved
        if self._agent is not None:
            self._agent.set_model(resolved)
        self._publish({"type": "session", "session": self._session_dto()})

    async def set_effort(self, level: str) -> None:
        await Backend.set_effort(self, level)
        if self._agent is not None:
            self._agent.set_effort(self.effort)
        self._publish({"type": "session", "session": self._session_dto()})

    def _session_dto(self) -> dict[str, Any]:
        return {
            "id": self.session_id or "-",
            "cwd": str(self.cwd),
            "modelKey": self.model_key,
            "mode": self.mode,
            "effort": self.effort,
            "contextUsed": self.context_used,
            "contextLimit": self.context_limit,
            "busy": self.busy,
        }

    async def compact(self) -> None:
        agent = await self._ensure_agent()
        await agent.compact()

    async def clear(self) -> None:
        if self._agent is not None:
            self._agent.clear()
        self.context = None
        self.goal = None

    async def start_goal(self, goal: str, message: str | None = None) -> None:
        agent = await self._ensure_agent()
        if self._turn is not None and not self._turn.done():
            raise BackendError("Agent jest zajęty — poczekaj na zakończenie odpowiedzi.")
        self.goal = await agent.set_goal(goal)
        await self.send_user(message or goal)

    async def clear_goal(self) -> None:
        if self._agent is not None:
            await self._agent.clear_goal()
        self.goal = None

    # -------------------------------------------------------------- sessions
    async def sessions(self, *, limit: int = 30) -> list[dict[str, Any]]:
        from localcode.agent import session as session_mod

        found = await asyncio.to_thread(session_mod.list_sessions, str(self.cwd), limit=limit)
        if not found:
            found = await asyncio.to_thread(session_mod.list_sessions, None, limit=limit)
        return [s.to_dict() for s in found]

    async def resume_session(self, session_id: str) -> list[dict[str, Any]]:
        from localcode.agent.session import SessionError, SessionStore

        try:
            store = await asyncio.to_thread(SessionStore.resume, session_id)
        except SessionError as exc:
            raise BackendError(str(exc)) from exc
        self._store = store
        self._agent = None
        self.session_id = store.session.id
        if store.session.model_key:
            self.model_key = store.session.model_key
        if store.session.mode:
            self.mode = store.session.mode
        await self._ensure_agent()
        self._publish({"type": "session", "session": self._session_dto()})
        return list(store.items)

    # --------------------------------------------------------------- library
    async def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh:
            await self.core.rescan_library()
        # The same list the app's pickers show: the disk, then the models
        # behind the API keys saved in Ustawienia → Modele zdalne.
        self._models = [*self.core.library_dtos(), *await self.core.remote_library_dtos()]
        return self._models

    async def delete_model(self, key: str) -> None:
        await asyncio.to_thread(self.core.library.delete, key)
        await self.core.rescan_library()
        self._models = None

    async def favorite_models(self) -> list[str]:
        # From the file: the Mac app may have starred a model since this started.
        with contextlib.suppress(Exception):
            fresh = await asyncio.to_thread(load_settings)
            self.settings.favorite_models = list(fresh.favorite_models)
        return list(self.settings.favorite_models)

    async def runtime_status(self) -> dict[str, Any]:
        return self.core.runtime.status()

    async def load_model(self, key: str, *, n_ctx: int | None = None,
                         n_parallel: int | None = None, new_instance: bool = False) -> dict[str, Any]:
        from localcode.runtime.base import LoadConfig

        resolved = await self.resolve_model_key(key)
        if resolved is None:
            raise BackendError(f"Nie znaleziono modelu „{key}”.")
        config: LoadConfig | None = None
        if n_ctx or n_parallel:
            runtime = self.settings.runtime
            config = LoadConfig(
                n_ctx=int(n_ctx or runtime.default_ctx or 8192),
                n_parallel=int(n_parallel or runtime.default_parallel),
                gpu_layers=runtime.gpu_layers,
                flash_attn=runtime.flash_attn,
                kv_cache_type=runtime.kv_cache_type,
            )
        return await self.core.runtime.load(resolved, config, new_instance=new_instance)

    async def unload(self, instance_id: str | None = None, *, everything: bool = False) -> None:
        if everything or not instance_id:
            await self.core.runtime.unload_all()
            return
        await self.core.runtime.unload(instance_id)

    async def update_runtime_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        return await self.core.runtime.update_settings(patch)

    async def estimate(self, key: str, *, n_ctx: int | None = None) -> dict[str, Any]:
        from localcode.runtime.base import LoadConfig

        resolved = await self.resolve_model_key(key) or key
        config = LoadConfig(n_ctx=int(n_ctx)) if n_ctx else None
        return self.core.runtime.estimate(resolved, config).to_dict()

    async def tune_runtime(self, *, model_key: str | None = None, use_model: bool = True,
                           apply: bool = False) -> dict[str, Any]:
        from localcode.runtime import tuner

        resolved = await self.resolve_model_key(model_key) if model_key else None
        model = self.core.resolve_model(resolved or model_key) if model_key else None
        facts = tuner.gather_machine_facts()
        proposal = tuner.recommend(facts, model)
        if model is not None and use_model:
            proposal = await tuner.refine_with_model(self.core.runtime, model.key, proposal)
        dto = proposal.to_dict()
        result: dict[str, Any] = {"recommendation": dto}
        if apply:
            result["status"] = await self.core.runtime.update_settings(dto["patch"])
        return result

    # -------------------------------------------------------------------- hub
    async def hub_search(self, query: str, *, fmt: str = "any", limit: int = 20) -> list[dict[str, Any]]:
        return await self.core.hub.asearch(query, format=fmt, limit=limit)

    async def hub_curated(self) -> list[dict[str, Any]]:
        return await self.core.hub.acurated()

    async def hub_details(self, repo: str) -> dict[str, Any]:
        return await self.core.hub.adetails(repo)

    async def start_download(self, repo: str, *, variant_id: str | None = None,
                             files: Sequence[str] | None = None) -> dict[str, Any]:
        manager = self.core.require_downloads()
        wanted: Any = list(files) if files else None
        if wanted is None and variant_id:
            details = await self.hub_details(repo)
            variant = next((v for v in details.get("variants", []) if v.get("id") == variant_id), None)
            if variant is None:
                raise BackendError(f"Wariant „{variant_id}” nie istnieje w repozytorium {repo}.")
            wanted = list(variant.get("files") or [])
        return manager.start(repo, wanted)

    async def downloads(self) -> list[dict[str, Any]]:
        manager = self.core.downloads
        return list(manager.list()) if manager is not None else []

    async def cancel_download(self, job_id: str) -> None:
        manager = self.core.require_downloads()
        await asyncio.to_thread(manager.cancel, job_id)


# --------------------------------------------------------------------------- #
# Remote
# --------------------------------------------------------------------------- #


class RemoteBackend(Backend):
    """Attaches to a running ``localcode serve`` over HTTP + WebSocket."""

    kind = "remote"

    def __init__(self, info: ServerInfo, cwd: str | os.PathLike[str],
                 settings: Settings | None = None, *, model_key: str | None = None,
                 mode: str | None = None, session_id: str | None = None,
                 attach_agent: bool = True) -> None:
        super().__init__(cwd, settings)
        self.info = info
        self.model_key = model_key
        if mode:
            self.mode = mode
        self.attach_agent = attach_agent
        self._resume_id = session_id
        self._session: Any = None
        self._ws: Any = None
        self._reader: asyncio.Task[Any] | None = None
        self._owns_session = False

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        import aiohttp

        headers = {"Authorization": f"Bearer {self.info.token}"} if self.info.token else {}
        self._session = aiohttp.ClientSession(
            base_url=self.info.base_url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None))
        if self.model_key:
            resolved = await self.resolve_model_key(self.model_key)
            if resolved is None:
                raise BackendError(f"Serwer nie zna modelu „{self.model_key}”.")
            self.model_key = resolved
        if self.model_key is None and self.settings.agent.default_model:
            self.model_key = await self.resolve_model_key(self.settings.agent.default_model)
        if self.model_key is None:
            self.model_key = await self.pick_default_model()
        if self._resume_id:
            await self._attach(self._resume_id)
            return
        if self.attach_agent:
            await self._create_session()

    async def _create_session(self) -> None:
        """Ask the server for an agent session and attach to its WebSocket."""
        body: dict[str, Any] = {"cwd": str(self.cwd), "mode": self.mode}
        if self.model_key:
            body["modelKey"] = self.model_key
        session = await self._request("POST", "/api/agent/sessions", json=body)
        self.session_id = str(session["id"])
        self._owns_session = True
        self.model_key = session.get("modelKey") or self.model_key
        self.mode = session.get("mode") or self.mode
        await self._connect_ws()

    async def _require_session(self) -> None:
        """Create the agent session on first use (``localcode ps`` never needs one)."""
        if self._ws is None or self._ws.closed:
            if self.session_id and self._ws is not None:
                raise BackendError("Połączenie z sesją agenta jest zamknięte.")
            await self._create_session()

    async def _attach(self, session_id: str) -> None:
        snapshot = await self._request("GET", f"/api/agent/sessions/{quote(session_id)}")
        session = snapshot.get("session") or snapshot
        self.session_id = str(session["id"])
        self.model_key = session.get("modelKey") or self.model_key
        self.mode = session.get("mode") or self.mode
        await self._connect_ws()

    async def _connect_ws(self) -> None:
        path = f"/api/agent/sessions/{quote(str(self.session_id))}/ws"
        if self.info.token:
            path += f"?token={quote(self.info.token)}"
        try:
            self._ws = await self._session.ws_connect(path, heartbeat=30.0)
        except Exception as exc:  # noqa: BLE001 - reported as a Polish message
            raise BackendError(f"Nie udało się połączyć z sesją agenta na serwerze: {exc}") from exc
        self._reader = asyncio.create_task(self._read_ws(), name="localcode-ws-reader")

    async def _read_ws(self) -> None:
        import aiohttp

        ws = self._ws
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except ValueError:
                        continue
                    if isinstance(payload, dict):
                        self._publish(payload)
                elif message.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE,
                                      aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a dropped connection is reported, not fatal
            log.warning("połączenie WebSocket z serwerem przerwane", exc_info=True)
        finally:
            if self.busy:
                self._publish({"type": "turn_end", "turnId": "", "reason": "error", "stats": {}})
            self._publish({"type": "error",
                           "message": "Połączenie z serwerem LocalCode zostało przerwane. "
                                      "Uruchom ponownie `localcode` albo użyj --local."})

    async def aclose(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None and not reader.done():
            reader.cancel()
            with contextlib.suppress(BaseException):
                await reader
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        if self._owns_session and self.session_id and self._session is not None:
            # sessions we created live in the server's memory: clean them up
            with contextlib.suppress(Exception):
                await self._request("DELETE", f"/api/agent/sessions/{quote(self.session_id)}")
            self._owns_session = False
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ HTTP
    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import aiohttp

        if self._session is None:
            raise BackendError("Brak połączenia z serwerem LocalCode.")
        try:
            async with self._session.request(method, path, **kwargs) as response:
                text = await response.text()
                payload: Any = None
                if text:
                    try:
                        payload = json.loads(text)
                    except ValueError:
                        payload = None
                if response.status >= 400:
                    message = (payload or {}).get("error") if isinstance(payload, dict) else None
                    raise BackendError(message or f"Serwer LocalCode zwrócił błąd {response.status}.")
                return payload
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise BackendError(f"Błąd połączenia z serwerem LocalCode: {exc}") from exc

    async def _send_ws(self, payload: dict[str, Any]) -> None:
        if self._ws is None or self._ws.closed:
            raise BackendError("Połączenie z sesją agenta jest zamknięte.")
        await self._ws.send_str(json.dumps(payload, ensure_ascii=False))

    # ----------------------------------------------------------------- agent
    async def send_user(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        await self._require_session()
        self.busy = True
        payload: dict[str, Any] = {"type": "user", "text": text}
        if images:
            payload["images"] = images
        await self._send_ws(payload)

    async def interrupt(self) -> None:
        if self._ws is not None:
            await self._send_ws({"type": "interrupt"})

    async def set_mode(self, mode: str) -> None:
        from localcode.agent.permissions import normalize_mode

        self.mode = normalize_mode(mode)
        if self._ws is not None:
            await self._send_ws({"type": "setMode", "mode": self.mode})

    async def set_model(self, key: str) -> None:
        resolved = await self.resolve_model_key(key)
        if resolved is None:
            raise BackendError(f"Serwer nie zna modelu „{key}”.")
        self.model_key = resolved
        if self._ws is not None:
            await self._send_ws({"type": "setModel", "modelKey": resolved})

    async def set_effort(self, level: str) -> None:
        await Backend.set_effort(self, level)
        if self._ws is not None:
            await self._send_ws({"type": "setEffort", "effort": self.effort})

    async def compact(self) -> None:
        await self._require_session()
        await self._send_ws({"type": "compact"})

    async def clear(self) -> None:
        self.context = None
        self.goal = None
        if self._ws is not None:
            await self._send_ws({"type": "clear"})

    async def start_goal(self, goal: str, message: str | None = None) -> None:
        await self._require_session()
        self.busy = True
        payload: dict[str, Any] = {"type": "setGoal", "text": goal}
        if message and message != goal:
            payload["message"] = message
        await self._send_ws(payload)

    async def clear_goal(self) -> None:
        self.goal = None
        if self._ws is not None:
            await self._send_ws({"type": "clearGoal"})

    async def answer_permission(self, request_id: str, decision: str) -> None:
        await self._send_ws({"type": "permission", "requestId": request_id, "decision": decision})

    # -------------------------------------------------------------- sessions
    async def sessions(self, *, limit: int = 30) -> list[dict[str, Any]]:
        rows = await self._request("GET", "/api/agent/sessions")
        return list(rows or [])[:limit]

    async def resume_session(self, session_id: str) -> list[dict[str, Any]]:
        snapshot = await self._request("GET", f"/api/agent/sessions/{quote(session_id)}")
        reader = self._reader
        self._reader = None
        if reader is not None and not reader.done():
            reader.cancel()
            with contextlib.suppress(BaseException):
                await reader
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        session = snapshot.get("session") or {}
        self.session_id = str(session.get("id") or session_id)
        self.model_key = session.get("modelKey") or self.model_key
        self.mode = session.get("mode") or self.mode
        await self._connect_ws()
        return list(snapshot.get("items") or [])

    # --------------------------------------------------------------- library
    async def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        path = "/api/library/rescan" if refresh else "/api/library"
        rows = await self._request("POST" if refresh else "GET", path)
        self._models = list(rows or [])
        return self._models

    async def delete_model(self, key: str) -> None:
        await self._request("DELETE", f"/api/library/{quote(key, safe='')}")
        self._models = None

    async def favorite_models(self) -> list[str]:
        try:
            body = await self._request("GET", "/api/app-settings")
        except BackendError:
            return list(self.settings.favorite_models)
        keys = (body or {}).get("favoriteModels")
        return [str(k) for k in keys] if isinstance(keys, list) else []

    async def runtime_status(self) -> dict[str, Any]:
        return await self._request("GET", "/api/runtime")

    async def load_model(self, key: str, *, n_ctx: int | None = None,
                         n_parallel: int | None = None, new_instance: bool = False) -> dict[str, Any]:
        resolved = await self.resolve_model_key(key)
        if resolved is None:
            raise BackendError(f"Serwer nie zna modelu „{key}”.")
        body: dict[str, Any] = {"modelKey": resolved, "newInstance": new_instance}
        config: dict[str, Any] = {}
        if n_ctx:
            config["nCtx"] = int(n_ctx)
        if n_parallel:
            config["nParallel"] = int(n_parallel)
        if config:
            body["config"] = config
        return await self._request("POST", "/api/runtime/load", json=body)

    async def unload(self, instance_id: str | None = None, *, everything: bool = False) -> None:
        body = {"all": True} if everything or not instance_id else {"instanceId": instance_id}
        await self._request("POST", "/api/runtime/unload", json=body)

    async def update_runtime_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        return await self._request("PATCH", "/api/runtime/settings", json=patch)

    async def estimate(self, key: str, *, n_ctx: int | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"modelKey": key}
        if n_ctx:
            body["config"] = {"nCtx": int(n_ctx)}
        return await self._request("POST", "/api/runtime/estimate", json=body)

    async def tune_runtime(self, *, model_key: str | None = None, use_model: bool = True,
                           apply: bool = False) -> dict[str, Any]:
        resolved = (await self.resolve_model_key(model_key)) if model_key else None
        body: dict[str, Any] = {"useModel": use_model, "apply": apply}
        wanted = resolved or model_key
        if wanted:
            body["modelKey"] = wanted
        return await self._request("POST", "/api/runtime/tune", json=body)

    # -------------------------------------------------------------------- hub
    async def hub_search(self, query: str, *, fmt: str = "any", limit: int = 20) -> list[dict[str, Any]]:
        params = {"q": query, "format": fmt, "limit": str(limit)}
        return list(await self._request("GET", "/api/hub/search", params=params) or [])

    async def hub_curated(self) -> list[dict[str, Any]]:
        return list(await self._request("GET", "/api/hub/curated") or [])

    async def hub_details(self, repo: str) -> dict[str, Any]:
        return await self._request("GET", "/api/hub/model", params={"repo": repo})

    async def start_download(self, repo: str, *, variant_id: str | None = None,
                             files: Sequence[str] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"repo": repo}
        if variant_id:
            body["variantId"] = variant_id
        if files:
            body["files"] = list(files)
        return await self._request("POST", "/api/downloads", json=body)

    async def downloads(self) -> list[dict[str, Any]]:
        return list(await self._request("GET", "/api/downloads") or [])

    async def cancel_download(self, job_id: str) -> None:
        await self._request("DELETE", f"/api/downloads/{quote(job_id, safe='')}")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


async def open_backend(
    *,
    cwd: str | os.PathLike[str] = ".",
    prefer: str = "auto",
    settings: Settings | None = None,
    model_key: str | None = None,
    mode: str | None = None,
    n_ctx: int | None = None,
    session_id: str | None = None,
    store_session: bool = True,
    ask_permission: bool = True,
    attach_agent: bool = True,
    server: ServerInfo | None = None,
) -> Backend:
    """Create and start the right backend.

    ``prefer="auto"`` attaches to a healthy server when there is one (so the
    Mac app's loaded models are reused), otherwise everything runs locally.
    ``attach_agent=False`` is for commands that only look at models and memory
    (``localcode ps``): no agent session is created on either side.
    """
    settings = settings if settings is not None else load_settings()
    choice = (prefer or "auto").lower()
    if choice not in ("auto", "local", "remote"):
        raise BackendError(f"Nieznany tryb połączenia „{prefer}”. Użyj --local albo --attach.")
    info = server
    if choice in ("auto", "remote") and info is None:
        info = await discover_server()
    if choice == "remote" and info is None:
        raise BackendError(
            "Nie znaleziono działającego serwera LocalCode. Uruchom aplikację na Maca "
            "albo `localcode serve`, ewentualnie dodaj --local.")
    backend: Backend
    if info is not None and choice in ("auto", "remote"):
        backend = RemoteBackend(info, cwd, settings, model_key=model_key, mode=mode,
                                session_id=session_id, attach_agent=attach_agent)
    else:
        backend = LocalBackend(cwd, settings, model_key=model_key, mode=mode, n_ctx=n_ctx,
                               session_id=session_id, store_session=store_session,
                               ask_permission=ask_permission)
    try:
        await backend.start()
    except BackendError:
        await backend.aclose()
        raise
    except Exception as exc:  # noqa: BLE001 - always a Polish message for the user
        await backend.aclose()
        raise BackendError(f"Nie udało się uruchomić LocalCode: {exc}") from exc
    return backend
