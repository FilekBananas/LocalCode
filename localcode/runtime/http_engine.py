"""Shared base for engines that run a model in a subprocess speaking the
llama-server HTTP protocol.

``LlamaServerEngine`` (GGUF, Homebrew ``llama-server``) and ``MlxEngine``
(``python -m localcode.runtime.mlx_worker``) both spawn a child process that
serves the same small HTTP surface on 127.0.0.1, so everything except "which
command to run" lives here:

Process lifecycle
    * a free port is picked, the child is spawned in its **own process group**
      (``start_new_session=True``) with stdout+stderr appended to
      ``logs_dir()/<log name>.log`` (the previous log is kept as ``.log.1``);
    * ``GET /health`` is polled until it answers 200 (503 = still loading); the
      wait is bounded by ``startup_timeout_base_s + startup_timeout_per_gb_s *
      model GB``. If the child exits first, ``EngineStartError`` is raised with
      a Polish explanation and the last ~30 log lines. A lost port race
      ("couldn't bind") is retried transparently, and subclasses may request a
      retry after adjusting their arguments (``_on_start_failure``);
    * ``stop()`` closes the HTTP session first (aborting in-flight streams —
      llama-server does NOT exit on SIGTERM while a stream is open), then sends
      SIGTERM to the whole group, waits ``stop_grace_s`` and escalates to
      SIGKILL. It is idempotent, shielded from cancellation and safe to call
      while ``start()`` is still waiting for the model to load;
    * every live process group is registered for a synchronous emergency kill
      at interpreter exit (``kill_all_engine_processes``). This registry of
      child PIDs is the only module-level state and exists solely so that a
      crashing/exiting core never leaves multi-GB model processes behind.
      A `kill -9` of the core runs no ``atexit`` hook, so the child must also
      be able to notice on its own that its parent is gone: ``mlx_worker`` and
      the image workers get ``--parent-pid``, and engines whose binary has no
      such option set ``parent_guard = True`` to be spawned under the tiny
      supervisor shim ``_PARENT_GUARD_SRC`` (see there).

HTTP client
    One ``aiohttp.ClientSession`` per engine with no total timeout (streams can
    legitimately run for minutes); short calls get a per-request timeout.
    ``complete()`` parses the SSE stream incrementally with its own decoder —
    llama-server's final frame echoes the whole prompt, which easily exceeds
    aiohttp's ``readline()`` limit on long agent prompts.

    Cancellation: closing the consumer (``aclose()`` or cancelling the task)
    closes the HTTP response; llama-server notices the disconnect and frees
    the slot within ~50 ms (verified against build 10964). Consumers that
    ``break`` out of ``async for`` should wrap the iterator in
    ``contextlib.aclosing`` so this happens immediately rather than at GC.

    Keep-alive: llama-server closes the connection after every SSE stream and
    after 5 s of idle time (``Keep-Alive: timeout=5, max=100``), so a pooled
    connection is regularly dead by the time the next request writes to it.
    A finished stream therefore never returns its connection to the pool, and
    a ``ServerDisconnectedError`` — which means the request produced no
    response at all — is retried once on a fresh connection.

Protocol facts (llama-server build 10964, verified by probing)
    * ``-c N -np P`` with explicit ``--no-kv-unified`` gives every slot
      ``N/P`` tokens (rounded up to a multiple of 256, capped at the model's
      training context). The true per-slot value is read back from ``/props``
      (``default_generation_settings.n_ctx``) and exposed as ``n_ctx``.
    * Final SSE frame: ``stop: true``, ``stop_type`` (``eos`` | ``word`` |
      ``limit``), ``truncated``, ``tokens_evaluated`` (= all prompt tokens),
      ``tokens_predicted``, ``tokens_cached`` (= tokens in the slot after the
      request, *not* the prefix-cache hit!) and ``timings`` with ``cache_n``
      (prompt tokens reused from the cache), ``prompt_n``, ``prompt_ms``,
      ``prompt_per_second``, ``predicted_n``, ``predicted_ms``,
      ``predicted_per_second``.
    * Context overflow: HTTP 400 ``{"error": {"type":
      "exceed_context_size_error", "n_prompt_tokens", "n_ctx", ...}}`` →
      ``ContextOverflowError``. Too-long embedding input: HTTP 500 "input (N
      tokens) is too large to process" → ``ContextOverflowError`` as well.
    * ``/apply-template`` accepts ``tools``, ``chat_template_kwargs`` and
      ``add_generation_prompt``; a trailing assistant message is treated as a
      prefill (continued, not closed).
    * ``/completion`` adds BOS itself when the model wants one, so a templated
      prompt that already starts with the BOS text would get two; the engine
      strips the textual BOS in that case.
    * ``/v1/embeddings`` (OpenAI shape) works in ``--embeddings`` mode;
      ``/embedding`` is kept as a fallback. Non-embedding servers answer 501.
    * ``/health`` needs no API key; every other endpoint requires the bearer
      key when the subclass sets ``_api_key``.
"""

from __future__ import annotations

import abc
import asyncio
import atexit
import inspect
import json
import logging
import math
import os
import re
import shlex
import signal
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import aiohttp

from localcode.config import logs_dir
from localcode.hub.types import LocalModel
from localcode.security import sandbox
from localcode.security.sandbox import SandboxPlan
from localcode.runtime.base import (
    CompletionChunk,
    Engine,
    EngineError,
    GenStats,
    LoadConfig,
    SamplingParams,
    TemplateInfo,
)

__all__ = [
    "HttpEngine",
    "EngineHttpError",
    "ContextOverflowError",
    "EngineUnavailableError",
    "EngineStartError",
    "SSEDecoder",
    "SSEEvent",
    "finish_reason_from_frame",
    "stats_from_frame",
    "kill_all_engine_processes",
]

log = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_BIND_RE = re.compile(r"couldn't bind|address already in use", re.IGNORECASE)
_CTX_MSG_RE = re.compile(
    r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)", re.IGNORECASE
)
_EMBED_TOO_LARGE_RE = re.compile(
    r"input \((\d+) tokens\) is too large to process.*?current batch size: (\d+)", re.IGNORECASE | re.DOTALL
)

# Sentinels used to probe what the engine's template renderer can do.
_PROBE_TOOL = "lc_probe_tool"
_PROBE_RESULT = "lc-probe-result-7f3a"
_PROBE_USER = "lc-probe-user-7f3a"
_PROBE_SYSTEM = "lc-probe-system-7f3a"
_THINK_MARKERS_RE = re.compile(r"<think>|<\|channel\|>analysis|\[THINK\]|<seed:think>|<\|begin_of_thought\|>")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class EngineHttpError(EngineError):
    """The engine process answered with an HTTP error status."""

    code = "engine_http_error"

    def __init__(
        self,
        message: str,
        *,
        status: int,
        server_message: str = "",
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.server_message = server_message
        self.error_type = error_type


class ContextOverflowError(EngineHttpError):
    """The prompt does not fit the per-slot context. Callers should compact the
    conversation and retry; ``n_prompt_tokens``/``n_ctx`` are filled when the
    engine reports them."""

    code = "context_overflow"

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        server_message: str = "",
        error_type: str | None = None,
        n_prompt_tokens: int | None = None,
        n_ctx: int | None = None,
    ) -> None:
        super().__init__(message, status=status, server_message=server_message, error_type=error_type)
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx


class EngineUnavailableError(EngineError):
    """The engine process is not running / crashed / cannot be reached.
    The runtime manager should treat the instance as dead and reload it."""

    code = "engine_unavailable"


class EngineStartError(EngineError):
    """The engine process could not be started; ``log_tail`` holds the last log lines."""

    code = "engine_start_failed"

    def __init__(self, message: str, *, log_tail: str = "", exit_code: int | None = None) -> None:
        super().__init__(message)
        self.log_tail = log_tail
        self.exit_code = exit_code


# --------------------------------------------------------------------------- #
# SSE decoding and final-frame parsing
# --------------------------------------------------------------------------- #


@dataclass
class SSEEvent:
    event: str  # "message" for data events, "error" for llama-server's `error:` lines
    data: str


class SSEDecoder:
    """Incremental ``text/event-stream`` decoder without a line-length limit.

    Handles ``\\n`` and ``\\r\\n`` line endings, multi-line ``data:`` fields,
    comments (``: ping``), the ``event:`` field and llama-server's
    non-standard ``error: {...}`` lines. Events are dispatched on a blank
    line; ``finish()`` flushes an event not terminated by one.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._data: list[str] = []
        self._event = ""

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buf += chunk
        if b"\n" not in chunk:
            return []
        *lines, rest = bytes(self._buf).split(b"\n")
        self._buf = bytearray(rest)
        out: list[SSEEvent] = []
        for raw in lines:
            ev = self._line(raw)
            if ev is not None:
                out.append(ev)
        return out

    def finish(self) -> list[SSEEvent]:
        out: list[SSEEvent] = []
        if self._buf:
            raw, self._buf = bytes(self._buf), bytearray()
            ev = self._line(raw)
            if ev is not None:
                out.append(ev)
        ev = self._dispatch()
        if ev is not None:
            out.append(ev)
        return out

    def _line(self, raw: bytes) -> SSEEvent | None:
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        line = raw.decode("utf-8", errors="replace")
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return None
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "error":
            self._event = "error"
            self._data.append(value)
        return None

    def _dispatch(self) -> SSEEvent | None:
        if not self._data:
            self._event = ""
            return None
        ev = SSEEvent(self._event or "message", "\n".join(self._data))
        self._data = []
        self._event = ""
        return ev


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def finish_reason_from_frame(frame: dict[str, Any]) -> str:
    """Map a final frame to ``"stop"`` or ``"length"``.

    llama-server sends ``stop_type`` (``eos``/``word`` → stop, ``limit`` →
    length, also used when the slot context filled up, with ``truncated``);
    the MLX worker may send an OpenAI-style ``finish_reason`` instead.
    """
    fr = frame.get("finish_reason")
    if fr in ("stop", "length"):
        return str(fr)
    if frame.get("stop_type") == "limit":
        return "length"
    return "stop"


def stats_from_frame(frame: dict[str, Any], wall_ms: float = 0.0) -> GenStats:
    """Build ``GenStats`` from a final frame.

    ``cached_tokens`` comes from ``timings.cache_n`` (llama-server's
    ``tokens_cached`` counts the whole slot after the request and is only used
    as a fallback, clamped to the prompt size, for workers without timings).
    """
    t = frame.get("timings") if isinstance(frame.get("timings"), dict) else {}
    cache_n = _num(t.get("cache_n"))
    prompt_n = _num(t.get("prompt_n"))
    evaluated = _num(frame.get("tokens_evaluated"))
    if evaluated is None:
        evaluated = (cache_n or 0) + (prompt_n or 0)
    prompt_tokens = int(evaluated)
    if cache_n is not None:
        cached = int(cache_n)
    else:
        cached = int(_num(frame.get("tokens_cached")) or 0)
    cached = max(0, min(cached, prompt_tokens))
    predicted = _num(frame.get("tokens_predicted"))
    if predicted is None:
        predicted = _num(t.get("predicted_n")) or 0
    prompt_ms = _num(t.get("prompt_ms"))
    predicted_ms = _num(t.get("predicted_ms"))
    total_ms = (prompt_ms or 0.0) + (predicted_ms or 0.0)
    if prompt_ms is None and predicted_ms is None:
        total_ms = wall_ms
    return GenStats(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached,
        completion_tokens=int(predicted),
        prompt_tps=_num(t.get("prompt_per_second")) or 0.0,
        gen_tps=_num(t.get("predicted_per_second")) or 0.0,
        total_ms=total_ms,
    )


# --------------------------------------------------------------------------- #
# Process-group helpers and the emergency-kill registry
# --------------------------------------------------------------------------- #

_LIVE_GROUPS: dict[int, str] = {}  # pgid -> description (see module docstring)
_ATEXIT_REGISTERED = False


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to a process group; False when nothing was there."""
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            os.kill(pgid, sig)
            return True
        except OSError:
            return False


# Tiny supervisor wrapped around third-party binaries that have no "die with my
# parent" option of their own (llama-server). It is spawned instead of the
# binary, in the engine's place, and re-execs nothing: it starts the real binary
# as its only child (same new session/process group, same stdio) and polls
# ``os.getppid()``. When the engine process dies — including `kill -9`, where no
# handler or atexit hook of ours can run — the supervisor is reparented to
# launchd, notices within a second and takes the binary down with it. Using
# getppid() instead of a pid passed in argv is immune to pid reuse, and the
# whole thing is passed as source on the command line so no extra module has to
# be importable in the child (startup cost: one bare `python -c`, no imports of
# aiohttp/mlx).
_PARENT_GUARD_SRC = r"""
import os, signal, subprocess, sys, time
argv, ppid = sys.argv[1:], os.getppid()
child = subprocess.Popen(argv)
for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, lambda s, f: child.send_signal(s))
while True:
    code = child.poll()
    if code is not None:
        break
    if os.getppid() != ppid:          # parent gone: we were reparented to init
        child.terminate()
        try:
            code = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
        break
    time.sleep(0.5)
sys.exit(code if code >= 0 else 128 - code)
"""


def _guarded_command(cmd: list[str]) -> list[str]:
    """``cmd`` wrapped in the parent-death supervisor (see ``_PARENT_GUARD_SRC``).

    ``-I`` (isolated) keeps the shim to the standard library: no current
    directory on ``sys.path``, no user site, no ``PYTHON*`` variables. It is one
    less way to influence a supervisor process, and it is what lets the shim run
    inside the sandbox — otherwise Python scans the working directory, which for
    a development install sits in ``~/Documents`` and is denied.
    """
    return [sys.executable, "-I", "-c", _PARENT_GUARD_SRC, *cmd]


def _register_group(pgid: int, description: str) -> None:
    global _ATEXIT_REGISTERED
    _LIVE_GROUPS[pgid] = description
    if not _ATEXIT_REGISTERED:
        atexit.register(kill_all_engine_processes)
        _ATEXIT_REGISTERED = True


def _unregister_group(pgid: int) -> None:
    _LIVE_GROUPS.pop(pgid, None)


def kill_all_engine_processes(grace_s: float = 0.3) -> None:
    """Synchronously terminate every engine process group still alive.

    Registered with ``atexit``; also safe to call from signal handlers of the
    CLI/server. SIGTERM first, then SIGKILL after ``grace_s``.
    """
    groups = list(_LIVE_GROUPS)
    if not groups:
        return
    for pgid in groups:
        _signal_group(pgid, signal.SIGTERM)
    time.sleep(max(0.0, grace_s))
    for pgid in groups:
        _signal_group(pgid, signal.SIGKILL)
        _unregister_group(pgid)


async def _terminate_process_group(proc: asyncio.subprocess.Process, grace_s: float) -> None:
    pgid = proc.pid
    if proc.returncode is None:
        _signal_group(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_s)
        except TimeoutError:
            log.warning("engine pid %d ignored SIGTERM for %.1fs, sending SIGKILL", pgid, grace_s)
            _signal_group(pgid, signal.SIGKILL)
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except TimeoutError:  # pragma: no cover - kernel-level hang
                log.error("engine pid %d did not exit after SIGKILL", pgid)
    # The leader is gone; make sure no helper process of its group survives.
    _signal_group(pgid, signal.SIGKILL)


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "engine"


def _parse_error_body(text: str) -> tuple[str, str | None, dict[str, Any]]:
    """(message, type, error object) from a llama-server/OpenAI style error body."""
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        return text.strip()[:500], None, {}
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        msg = str(err.get("message") or err.get("detail") or "").strip()
        etype = err.get("type")
        return msg or text.strip()[:500], (str(etype) if etype else None), err
    if isinstance(err, str):
        return err.strip(), None, {}
    return text.strip()[:500], None, {}


def _int_or_none(value: Any) -> int | None:
    n = _num(value)
    return int(n) if n is not None else None


# --------------------------------------------------------------------------- #
# Template helpers (fallbacks when runtime.templates is unavailable)
# --------------------------------------------------------------------------- #


def _heuristic_template_info(template: str | None) -> TemplateInfo:
    """Minimal capability guess from the template source (used only when
    ``localcode.runtime.templates`` cannot be imported)."""
    info = TemplateInfo()
    if not template:
        return info
    info.supports_tools = bool(re.search(r"\btools\b", template))
    info.supports_tool_role = bool(re.search(r"""['"]tool['"]""", template)) or "tool_response" in template
    info.supports_system = "system" in template
    if "enable_thinking" in template:
        info.thinking_kwarg = "enable_thinking"
    info.supports_thinking = info.thinking_kwarg is not None or "<think>" in template
    if "reasoning_effort" in template:
        info.thinking_kwarg = info.thinking_kwarg or "reasoning_effort"
        info.supports_thinking = True
        info.effort_kwarg, info.effort_levels = "reasoning_effort", ["low", "medium", "high"]
    elif info.thinking_kwarg:
        info.effort_kwarg, info.effort_levels = info.thinking_kwarg, ["off", "on"]
    if "<tool_call>" in template:
        info.tool_format = "hermes"
    return info


def _inspect_template(template: str | None) -> TemplateInfo:
    try:
        from localcode.runtime.templates import inspect_template
    except ImportError:
        return _heuristic_template_info(template)
    try:
        info = inspect_template(template)
    except Exception as exc:  # noqa: BLE001 - never let detection break a load
        log.warning("inspect_template failed (%s); using heuristics", exc)
        return _heuristic_template_info(template)
    return info if isinstance(info, TemplateInfo) else _heuristic_template_info(template)


def _local_renderer() -> Callable[..., str] | None:
    try:
        from localcode.runtime.templates import render_template
    except ImportError:
        return None
    return render_template


def _call_local_renderer(
    fn: Callable[..., str],
    template: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    bos: str,
    eos: str,
    kwargs: dict[str, Any],
) -> str:
    """Call ``render_template`` whether it names the special tokens
    ``bos_token``/``eos_token`` (keyword-only) or ``bos``/``eos``."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    call_kwargs: dict[str, Any] = dict(kwargs)
    call_kwargs["add_generation_prompt"] = add_generation_prompt
    if "bos_token" in params or "bos" not in params:
        call_kwargs["bos_token"] = bos
        call_kwargs["eos_token"] = eos
    else:
        call_kwargs["bos"] = bos
        call_kwargs["eos"] = eos
    return fn(template, messages, tools, **call_kwargs)


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy messages for the template: drop reasoning (never re-sent to the
    model) and replace ``None`` content with an empty string."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d = {k: v for k, v in m.items() if k not in ("reasoning", "reasoning_content")}
        if d.get("content") is None:
            d["content"] = ""
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# HttpEngine
# --------------------------------------------------------------------------- #


class HttpEngine(Engine):
    """Base class for subprocess engines speaking the llama-server HTTP protocol.

    Subclasses implement ``_build_command(port)`` and usually override
    ``_log_name()``, ``_env()``, ``_before_spawn()`` (validation, API key) and
    ``_on_start_failure()`` / ``_failure_hint()`` (engine-specific recovery and
    error messages). ``_after_start()`` reads ``/props`` (per-slot context,
    special tokens, chat template) and fills ``template_info``; override it to
    add engine-specific steps (call ``super()._after_start()``).
    """

    host: str = "127.0.0.1"
    startup_timeout_base_s: float = 90.0     # also covers Metal init under GPU contention
    startup_timeout_per_gb_s: float = 15.0
    stop_grace_s: float = 5.0
    health_poll_s: float = 0.25
    request_timeout_s: float = 120.0         # non-streaming calls
    embed_timeout_s: float = 600.0
    embed_batch_size: int = 64
    log_tail_lines: int = 30
    max_start_attempts: int = 3
    parent_guard: bool = False        # wrap the child in _PARENT_GUARD_SRC (see below)

    def __init__(self, instance_id: str, model: LocalModel, config: LoadConfig) -> None:
        super().__init__(instance_id, model, config)
        self._proc: asyncio.subprocess.Process | None = None
        self._exit_task: asyncio.Future[int] | None = None
        self._port: int | None = None
        self._session: aiohttp.ClientSession | None = None
        self._api_key: str | None = None
        self._ready = False
        self._stopping = False
        self._stop_task: asyncio.Future[None] | None = None
        self._log_path: Path | None = None
        self._props: dict[str, Any] = {}
        self._n_ctx_slot: int | None = None
        self._n_slots: int | None = None
        self._bos_text: str | None = None
        self._eos_text: str | None = None
        self._adds_bos = False
        self._chat_template: str | None = None
        self._embed_endpoint: str | None = None
        self.server_renders_tools = False     # /apply-template renders `tools`
        self.local_tools_render = False       # tools rendered locally via runtime.templates
        self.exit_code: int | None = None
        self.started_at: float | None = None  # unix seconds when ready
        self.load_ms: float | None = None
        #: True while this instance's process runs inside the macOS sandbox
        #: (:mod:`localcode.security.sandbox`).
        self.sandboxed = False
        self._sandbox_profile: Path | None = None
        self._sandbox_off = False       # set after a sandboxed start failed in "auto"

    # ------------------------------------------------------------------ hooks
    def _sandbox_plan(self) -> "SandboxPlan | None":
        """What this engine may read and write, or ``None`` to skip the sandbox.

        A model file is data, but the code that parses it is a C++ or Python
        program handling megabytes of attacker-supplied structure. Engines that
        can state their directories return a plan here and get run with no
        outbound network and no access to the user's files.
        """
        return None
    @abc.abstractmethod
    def _build_command(self, port: int) -> list[str]:
        """argv of the server process listening on ``self.host:port``."""

    def _env(self) -> dict[str, str]:
        """Complete environment of the child process."""
        return dict(os.environ)

    def _log_name(self) -> str:
        return f"{self.format or 'engine'}-{_safe_name(self.instance_id)}"

    async def _before_spawn(self) -> None:
        """Validate the configuration before the first spawn attempt."""

    def _on_start_failure(self, log_text: str, exit_code: int | None) -> bool:
        """Called when the process died during startup. Return True after
        adjusting the configuration to retry once more."""
        return False

    def _failure_hint(self, log_text: str, exit_code: int | None) -> str | None:
        """Polish one-sentence explanation of a startup failure, if recognised."""
        return None

    # -------------------------------------------------------------- properties
    @property
    def running(self) -> bool:
        return self._ready and self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def n_ctx(self) -> int:
        """True per-slot context reported by the server (falls back to the request)."""
        return self._n_ctx_slot or self.config.n_ctx

    @property
    def n_parallel(self) -> int:
        return self._n_slots or max(1, self.config.n_parallel)

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def base_url(self) -> str | None:
        return f"http://{self.host}:{self._port}" if self._port else None

    @property
    def last_props(self) -> dict[str, Any]:
        """``/props`` as read right after start."""
        return dict(self._props)

    @property
    def modalities(self) -> dict[str, bool]:
        """Input modalities reported by ``/props`` (e.g. ``{"vision": True}``)."""
        mods = self._props.get("modalities")
        return {str(k): bool(v) for k, v in mods.items()} if isinstance(mods, dict) else {}

    @property
    def media_marker(self) -> str | None:
        """Where an image goes in the prompt. llama.cpp makes up a new one at
        every start (``<__media_…__>``) and reports it in ``/props``."""
        marker = self._props.get("media_marker")
        return str(marker) if isinstance(marker, str) and marker else None

    @property
    def supports_images(self) -> bool:
        """A vision model with its projector loaded (``--mmproj``)."""
        return bool(self.modalities.get("vision")) and self.media_marker is not None

    def log_tail(self, lines: int | None = None) -> str:
        """Last lines of the engine log (ANSI codes removed)."""
        n = lines or self.log_tail_lines
        path = self._log_path
        if path is None:
            return ""
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 96 * 1024))
                data = fh.read()
        except OSError:
            return ""
        text = _ANSI_RE.sub("", data.decode("utf-8", errors="replace"))
        rows = [r.rstrip() for r in text.splitlines() if r.strip()]
        return "\n".join(rows[-n:])

    # --------------------------------------------------------------- lifecycle
    def _startup_timeout(self) -> float:
        size = self.model.size_bytes or 0
        if not size:
            try:
                size = os.path.getsize(self.model.path)
            except OSError:
                size = 0
        if self.model.mmproj_path:
            try:
                size += os.path.getsize(self.model.mmproj_path)
            except OSError:
                pass
        return self.startup_timeout_base_s + self.startup_timeout_per_gb_s * (size / 1e9)

    def _prepare_log(self) -> Path:
        path = logs_dir() / f"{self._log_name()}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                os.replace(path, path.with_name(path.name + ".1"))
            except OSError:
                pass
        return path

    async def start(self) -> None:
        if self.running:
            return
        if self._proc is not None:
            await self.stop()
        if self._stop_task is not None and not self._stop_task.done():
            await asyncio.shield(self._stop_task)
        self._stopping = False
        self._ready = False
        self.exit_code = None
        t0 = time.monotonic()
        try:
            await self._before_spawn()
            self._log_path = self._prepare_log()
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                connector=aiohttp.TCPConnector(limit=0, force_close=False),
            )
            timeout = self._startup_timeout()
            for attempt in range(1, self.max_start_attempts + 1):
                self._port = _free_port(self.host)
                await self._spawn(self._build_command(self._port))
                if await self._wait_healthy(timeout):
                    break
                if self._stopping:
                    raise EngineError("Uruchamianie modelu zostało przerwane.")
                proc = self._proc
                code = proc.returncode if proc is not None else None
                tail = self.log_tail()
                await self._reap()
                if attempt < self.max_start_attempts and _BIND_RE.search(tail):
                    log.warning("engine %s lost the port race, retrying", self.instance_id)
                    continue
                if attempt < self.max_start_attempts and self._on_start_failure(tail, code):
                    log.warning("engine %s failed to start, retrying with adjusted settings", self.instance_id)
                    continue
                if attempt < self.max_start_attempts and self._sandbox_fallback():
                    continue
                raise EngineStartError(self._failure_message(tail, code), log_tail=tail, exit_code=code)
            await self._after_start()
            if self._stopping:
                raise EngineError("Uruchamianie modelu zostało przerwane.")
            self._ready = True
            self.started_at = time.time()
            self.load_ms = (time.monotonic() - t0) * 1000.0
            log.info(
                "engine %s ready on port %s (pid %s, n_ctx/slot %s, slots %s) in %.1fs",
                self.instance_id, self._port, self.pid, self.n_ctx, self.n_parallel, self.load_ms / 1000,
            )
        except BaseException:
            await self.stop()
            raise

    async def _spawn(self, cmd: list[str]) -> None:
        assert self._log_path is not None
        argv = _guarded_command(cmd) if self.parent_guard else cmd
        argv = self._sandboxed_command(argv)
        with open(self._log_path, "ab") as fh:
            header = f"# {datetime.now().isoformat(timespec='seconds')} {shlex.join(cmd)}\n"
            fh.write(header.encode())
            fh.flush()
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=fh,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self._env(),
                    start_new_session=True,
                )
            except OSError as exc:
                raise EngineStartError(f"Nie można uruchomić procesu silnika „{cmd[0]}”: {exc.strerror or exc}.") from exc
        _register_group(self._proc.pid, f"{self._log_name()} ({self.model.key})")
        self._exit_task = asyncio.ensure_future(self._proc.wait())

    def _sandboxed_command(self, argv: list[str]) -> list[str]:
        """``argv`` wrapped in ``sandbox-exec``, when this engine asked for it.

        The sandbox goes on the outside, so the parent-guard shim runs inside it
        too. In ``auto`` mode a sandbox the platform cannot provide is simply
        skipped; in ``strict`` mode :func:`sandbox.wrap` raises and the model
        does not load.
        """
        self.sandboxed = False
        sandbox.cleanup(self._sandbox_profile)
        self._sandbox_profile = None
        if self._sandbox_off:
            return argv
        plan = self._sandbox_plan()
        if plan is None:
            return argv
        if self.parent_guard:
            # The guard shim is this very interpreter, which in a development
            # install lives under ~/Documents — a directory the profile denies.
            # Without this the sandboxed process cannot even start Python.
            plan = SandboxPlan(read=[*plan.read, Path(sys.executable).parent, Path(sys.prefix)],
                               write=list(plan.write), local_network=plan.local_network)
        argv, self.sandboxed, self._sandbox_profile = sandbox.wrap(
            argv, plan, settings=getattr(self, "_settings", None))
        return argv

    def _sandbox_fallback(self) -> bool:
        """Give up the sandbox after a failed start (``auto`` only).

        Returns ``True`` when the caller should retry; a model the user can no
        longer load is a worse outcome than one running without isolation, and
        the status line says which happened.
        """
        if not self.sandboxed or self._sandbox_off:
            return False
        if sandbox.mode(getattr(self, "_settings", None)) != "auto":
            return False
        self._sandbox_off = True
        log.warning("engine %s did not start in the sandbox, retrying without it", self.instance_id)
        return True

    async def _reap(self) -> None:
        sandbox.cleanup(self._sandbox_profile)
        self._sandbox_profile = None
        proc, self._proc = self._proc, None
        self._exit_task = None
        if proc is None:
            return
        await _terminate_process_group(proc, 1.0)
        self.exit_code = proc.returncode
        _unregister_group(proc.pid)

    async def _health_status(self) -> int | None:
        session = self._session
        if session is None or self._port is None:
            return None
        try:
            async with session.get(
                f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                await resp.read()
                return resp.status
        except (aiohttp.ClientError, TimeoutError, OSError):
            return None

    async def _wait_healthy(self, timeout: float) -> bool:
        """True when /health answers 200; False when the process exited (or a
        stop was requested) first. Raises EngineStartError on timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            proc = self._proc
            if proc is None or proc.returncode is not None or self._stopping:
                return False
            status = await self._health_status()
            if status == 200:
                return True
            proc = self._proc
            if proc is None or proc.returncode is not None or self._stopping:
                return False
            if loop.time() >= deadline:
                tail = self.log_tail()
                raise EngineStartError(
                    f"Model „{self.model.display_name}” nie uruchomił się w ciągu {int(timeout)} s "
                    f"(status: {'ładowanie' if status == 503 else 'brak odpowiedzi'}).\n\n"
                    f"Ostatnie linie logu ({self._log_path}):\n{tail}",
                    log_tail=tail,
                )
            await self._wait_exit(self.health_poll_s)

    async def _wait_exit(self, timeout: float) -> bool:
        """Wait up to ``timeout`` for the process to exit; True if it did."""
        task = self._exit_task
        if task is None:
            return self._proc is None or self._proc.returncode is not None
        done, _ = await asyncio.wait({task}, timeout=timeout)
        return bool(done)

    def _failure_message(self, tail: str, code: int | None) -> str:
        hint = self._failure_hint(tail, code) or self._generic_failure_hint(tail, code)
        if code is None:
            status = ""
        elif code < 0:
            try:
                status = f" (zakończony sygnałem {signal.Signals(-code).name})"
            except ValueError:
                status = f" (zakończony sygnałem {-code})"
        else:
            status = f" (kod wyjścia {code})"
        msg = f"Nie udało się uruchomić modelu „{self.model.display_name}”{status}."
        if hint:
            msg += f" {hint}"
        if tail:
            msg += f"\n\nOstatnie linie logu ({self._log_path}):\n{tail}"
        return msg

    @staticmethod
    def _generic_failure_hint(tail: str, code: int | None) -> str | None:
        low = tail.lower()
        if code == -signal.SIGKILL:
            return "Proces został zabity przez system – prawdopodobnie zabrakło pamięci."
        if any(s in low for s in ("out of memory", "failed to allocate", "insufficient memory", "outofmemory")):
            return (
                "Za mało pamięci, aby uruchomić model z tymi ustawieniami. Zmniejsz kontekst lub liczbę "
                "równoległych slotów albo wyładuj inne modele."
            )
        if _BIND_RE.search(tail):
            return "Nie udało się zająć portu sieciowego dla silnika."
        return None

    async def stop(self) -> None:
        task = self._stop_task
        if task is None or task.done():
            if self._proc is None and self._session is None:
                self._ready = False
                return
            task = asyncio.ensure_future(self._do_stop())
            self._stop_task = task
        await asyncio.shield(task)

    async def _do_stop(self) -> None:
        self._stopping = True
        self._ready = False
        session, self._session = self._session, None
        if session is not None:
            # Abort in-flight streams first: llama-server ignores SIGTERM while
            # a streaming response is open.
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                log.debug("closing engine session failed", exc_info=True)
        proc = self._proc
        if proc is not None:
            await _terminate_process_group(proc, self.stop_grace_s)
            self.exit_code = proc.returncode
            _unregister_group(proc.pid)
            if self._proc is proc:
                self._proc = None
                self._exit_task = None
            log.info("engine %s stopped (pid %d, exit %s)", self.instance_id, proc.pid, proc.returncode)

    def kill_now(self) -> None:
        """Synchronous emergency kill (SIGKILL to the process group). For atexit
        or signal handlers; prefer ``await stop()`` everywhere else."""
        proc = self._proc
        self._ready = False
        self._stopping = True
        if proc is not None:
            _signal_group(proc.pid, signal.SIGKILL)
            _unregister_group(proc.pid)

    # ------------------------------------------------------------ HTTP helpers
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _require_session(self) -> aiohttp.ClientSession:
        session = self._session
        if session is None or session.closed or self._port is None:
            if self._stopping:
                raise EngineUnavailableError("Model został zatrzymany.")
            raise EngineUnavailableError("Model nie jest uruchomiony.")
        return session

    async def _connection_error(self, exc: BaseException) -> EngineError:
        if self._stopping:
            return EngineUnavailableError("Model został zatrzymany.")
        proc = self._proc
        if proc is not None and proc.returncode is None:
            # a crash is often noticed on the socket before the child watcher
            await self._wait_exit(0.5)
        if proc is None or proc.returncode is not None:
            code = proc.returncode if proc is not None else None
            self._ready = False
            tail = self.log_tail()
            hint = self._generic_failure_hint(tail, code) or ""
            msg = f"Proces modelu „{self.model.display_name}” przestał działać"
            msg += f" (kod wyjścia {code})." if code is not None else "."
            if hint:
                msg += f" {hint}"
            if tail:
                msg += f"\n\nOstatnie linie logu ({self._log_path}):\n{tail}"
            return EngineUnavailableError(msg)
        return EngineUnavailableError(
            f"Brak połączenia z procesem modelu „{self.model.display_name}” ({type(exc).__name__}: {exc})."
        )

    def _http_error(self, status: int, text: str, path: str) -> EngineError:
        msg, etype, err = _parse_error_body(text)
        m = _CTX_MSG_RE.search(msg)
        if etype == "exceed_context_size_error" or m:
            n_prompt = _int_or_none(err.get("n_prompt_tokens")) or (int(m.group(1)) if m else None)
            n_ctx = _int_or_none(err.get("n_ctx")) or (int(m.group(2)) if m else None)
            detail = f"{n_prompt} > {n_ctx} tokenów" if n_prompt and n_ctx else msg
            return ContextOverflowError(
                f"Zapytanie przekracza dostępny kontekst modelu ({detail}). Skróć rozmowę lub zwiększ kontekst.",
                status=status, server_message=msg, error_type=etype, n_prompt_tokens=n_prompt, n_ctx=n_ctx,
            )
        m = _EMBED_TOO_LARGE_RE.search(msg)
        if m:
            n_prompt, n_batch = int(m.group(1)), int(m.group(2))
            return ContextOverflowError(
                f"Tekst do embeddingu przekracza kontekst modelu ({n_prompt} > {n_batch} tokenów). "
                "Podziel go na mniejsze fragmenty.",
                status=status, server_message=msg, error_type=etype, n_prompt_tokens=n_prompt, n_ctx=n_batch,
            )
        if status in (401, 403):
            text_pl = "Proces modelu odrzucił autoryzację (nieprawidłowy klucz API)."
        elif status == 404:
            text_pl = f"Silnik nie obsługuje tej operacji ({path})."
        elif status == 501:
            text_pl = f"Silnik nie obsługuje tej operacji ({path}): {msg}"
        elif status == 503:
            text_pl = "Model jest zajęty lub wciąż się ładuje. Spróbuj ponownie za chwilę."
        elif 400 <= status < 500:
            text_pl = f"Silnik odrzucił zapytanie ({path}): {msg}"
        else:
            text_pl = f"Błąd silnika modelu (HTTP {status}, {path}): {msg}"
        return EngineHttpError(text_pl, status=status, server_message=msg, error_type=etype)

    async def _request_json(
        self, method: str, path: str, body: Any = None, *, timeout: float | None = None
    ) -> Any:
        limit = timeout if timeout is not None else self.request_timeout_s
        text = ""
        for attempt in range(2):
            session = self._require_session()
            try:
                async with session.request(
                    method,
                    f"{self.base_url}{path}",
                    json=body,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=limit, sock_connect=10),
                ) as resp:
                    text = await resp.text(errors="replace")
                    if resp.status >= 400:
                        raise self._http_error(resp.status, text, path)
                break
            except aiohttp.ServerDisconnectedError as exc:
                # A pooled keep-alive connection that the server had already
                # closed: llama-server ends the TCP connection after every SSE
                # stream and after 5 s of idle time, and the FIN often arrives
                # only while we are writing the next request. Nothing was sent
                # to the model, and every endpoint reached from here is a pure
                # read, so one retry on a fresh connection is safe.
                if attempt == 0 and not self._stopping:
                    log.debug("engine %s: stale keep-alive connection on %s, retrying", self.instance_id, path)
                    continue
                raise await self._connection_error(exc) from exc
            except TimeoutError as exc:
                raise EngineError(f"Silnik modelu nie odpowiedział w ciągu {int(limit)} s ({path}).") from exc
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                raise await self._connection_error(exc) from exc
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise EngineError(f"Silnik modelu zwrócił niepoprawną odpowiedź ({path}).") from exc

    # --------------------------------------------------------- info endpoints
    async def health(self) -> bool:
        """True when the process is alive and ``/health`` answers 200."""
        return self._proc is not None and self._proc.returncode is None and await self._health_status() == 200

    async def props(self) -> dict[str, Any]:
        """Fresh ``GET /props`` (n_ctx, total_slots, chat_template, special tokens…)."""
        data = await self._request_json("GET", "/props", timeout=30)
        return data if isinstance(data, dict) else {}

    async def slots(self) -> list[dict[str, Any]]:
        """``GET /slots`` (``id``, ``n_ctx``, ``is_processing`` per slot)."""
        data = await self._request_json("GET", "/slots", timeout=30)
        return data if isinstance(data, list) else []

    async def memory_bytes(self) -> int | None:
        """Resident memory (RSS) of the process tree.

        On Apple Silicon the model weights are mmapped and wrapped as Metal
        buffers without copying, so RSS includes weights, KV cache and compute
        buffers (verified: 4B Q4_K_M + 8k ctx → RSS ≈ file + KV). Caveats:
        mmapped pages can be evicted under pressure (RSS drops, reload is slow)
        and two instances of one file share those pages (double counted).
        ``phys_footprint`` would miss the mmapped weights, so it is not used.
        """
        pid = self.pid
        if pid is None:
            return None
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a declared dependency
            return None
        try:
            proc = psutil.Process(pid)
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return int(total)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

    # --------------------------------------------------------- after start
    async def _after_start(self) -> None:
        props = await self.props()
        self._props = props
        dgs = props.get("default_generation_settings")
        n_ctx = _int_or_none(dgs.get("n_ctx")) if isinstance(dgs, dict) else None
        n_ctx = n_ctx or _int_or_none(props.get("n_ctx"))
        self._n_slots = _int_or_none(props.get("total_slots"))
        try:
            slot_list = await self.slots()
        except EngineError:
            slot_list = []
        slot_ctx = [c for c in (_int_or_none(s.get("n_ctx")) for s in slot_list if isinstance(s, dict)) if c]
        if slot_ctx:
            n_ctx = min(slot_ctx)
            self._n_slots = self._n_slots or len(slot_ctx)
        self._n_ctx_slot = n_ctx or None
        if self._n_ctx_slot and self._n_ctx_slot < self.config.n_ctx:
            log.warning(
                "engine %s: requested n_ctx %d per slot, server provides %d (capped by the model)",
                self.instance_id, self.config.n_ctx, self._n_ctx_slot,
            )
        self._bos_text = props.get("bos_token") or None
        self._eos_text = props.get("eos_token") or None
        template = props.get("chat_template") or self.model.meta.chat_template
        self._chat_template = template or None
        if self.config.embedding:
            self.template_info = TemplateInfo()
            return
        self._adds_bos = await self._probe_adds_bos()
        self.template_info = await self._detect_template_info()

    async def _probe_adds_bos(self) -> bool:
        """Does prompt tokenization in /completion prepend BOS by itself?"""
        if not self._bos_text:
            return False
        try:
            with_special = await self._request_json(
                "POST", "/tokenize", {"content": "a", "add_special": True}, timeout=30
            )
            plain = await self._request_json("POST", "/tokenize", {"content": "a", "add_special": False}, timeout=30)
        except EngineError:
            return False
        a = with_special.get("tokens", []) if isinstance(with_special, dict) else []
        b = plain.get("tokens", []) if isinstance(plain, dict) else []
        return len(a) > len(b) and len(b) > 0 and a[-len(b):] == b

    async def _detect_template_info(self) -> TemplateInfo:
        """Combine static template inspection with probes of the engine's own
        renderer (the renderer that actually produces the prompts)."""
        template = self._chat_template
        have_templates_module = _local_renderer() is not None
        info = _inspect_template(template)
        caps = self._props.get("chat_template_caps")
        caps = caps if isinstance(caps, dict) else {}
        if "supports_system_role" in caps:
            info.supports_system = bool(caps["supports_system_role"])
        native_tools = bool(caps["supports_tools"]) if "supports_tools" in caps else info.supports_tools

        tool = {
            "type": "function",
            "function": {
                "name": _PROBE_TOOL,
                "description": "Probe.",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            },
        }
        base = [{"role": "system", "content": _PROBE_SYSTEM}, {"role": "user", "content": _PROBE_USER}]
        try:
            rendered = await self._server_apply_template(base, [tool], True, None)
            self.server_renders_tools = _PROBE_TOOL in rendered
        except EngineError as exc:
            log.info("engine %s: /apply-template with tools failed: %s", self.instance_id, exc)
            self.server_renders_tools = False
        self.local_tools_render = False
        if native_tools and self.server_renders_tools:
            info.supports_tools = True
        elif native_tools and template and have_templates_module:
            info.supports_tools = True
            self.local_tools_render = True
        else:
            info.supports_tools = False

        # Tool results: do they survive the server's rendering (and does the
        # template accept them)? With local tool rendering the static
        # inspection already answered this for the local renderer.
        if not self.local_tools_render:
            call = {"id": "call_lcprobe", "type": "function", "function": {"name": _PROBE_TOOL, "arguments": {"x": "1"}}}
            conv = [
                {"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "call_lcprobe", "name": _PROBE_TOOL, "content": _PROBE_RESULT},
            ]
            try:
                rendered = await self._server_apply_template(conv, [tool] if info.supports_tools else None, True, None)
                info.supports_tool_role = _PROBE_RESULT in rendered
            except EngineError:
                info.supports_tool_role = False

        if not have_templates_module:
            await self._probe_thinking(info, template)

        if info.tool_format is None:
            try:
                from localcode.runtime.toolcalls import detect_tool_format

                info.tool_format = detect_tool_format(self.model, template)
            except Exception:  # noqa: BLE001 - optional module / best effort
                pass
        return info

    async def _probe_thinking(self, info: TemplateInfo, template: str | None) -> None:
        conv = [{"role": "user", "content": _PROBE_USER}]
        if not template or "reasoning_effort" in template:
            try:
                low = await self._server_apply_template(conv, None, True, {"reasoning_effort": "low"})
                high = await self._server_apply_template(conv, None, True, {"reasoning_effort": "high"})
            except EngineError:
                low = high = ""
            if low and high and low != high:
                info.thinking_kwarg = "reasoning_effort"
                info.supports_thinking = True
                info.effort_kwarg = "reasoning_effort"
                info.effort_levels = ["low", "medium", "high"]
                return
        for kwarg in ("enable_thinking", "thinking"):
            if template and kwarg not in template:
                continue
            try:
                on = await self._server_apply_template(conv, None, True, {kwarg: True})
                off = await self._server_apply_template(conv, None, True, {kwarg: False})
            except EngineError:
                continue
            if on != off:
                info.thinking_kwarg = kwarg
                info.supports_thinking = True
                info.effort_kwarg = kwarg
                info.effort_levels = ["off", "on"]
                return
        try:
            default = await self._server_apply_template(conv, None, True, None)
        except EngineError:
            return
        if _THINK_MARKERS_RE.search(default.split(_PROBE_USER)[-1]):
            info.supports_thinking = True

    # --------------------------------------------------------------- inference
    async def _server_apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
        template_kwargs: dict[str, Any] | None,
    ) -> str:
        body: dict[str, Any] = {"messages": messages, "add_generation_prompt": add_generation_prompt}
        if tools:
            body["tools"] = tools
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        try:
            data = await self._request_json("POST", "/apply-template", body)
        except EngineHttpError as exc:
            # Template failures (raise_exception, bad roles…) come back as 400/500.
            if isinstance(exc, ContextOverflowError) or exc.status not in (400, 422, 500):
                raise
            raise EngineHttpError(
                f"Szablon czatu modelu nie przyjął wiadomości: {exc.server_message or exc}",
                status=exc.status, server_message=exc.server_message, error_type=exc.error_type,
            ) from exc
        prompt = data.get("prompt") if isinstance(data, dict) else None
        if not isinstance(prompt, str):
            raise EngineError("Silnik modelu zwrócił niepoprawną odpowiedź (/apply-template).")
        return prompt

    async def apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = True,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render with the model's chat template.

        Server-side (``/apply-template``) normally; when the server cannot
        render ``tools`` for a template that supports them, the template is
        rendered locally with ``runtime.templates.render_template``. Note: a
        trailing assistant message is continued (prefill), not closed.
        """
        msgs = _clean_messages(messages)
        if tools and self.local_tools_render and not self.server_renders_tools:
            fn = _local_renderer()
            if fn is not None and self._chat_template:
                try:
                    return _call_local_renderer(
                        fn, self._chat_template, msgs, tools, add_generation_prompt,
                        self._bos_text or "", self._eos_text or "", dict(template_kwargs or {}),
                    )
                except EngineError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise EngineError(f"Nie udało się zastosować szablonu czatu modelu: {exc}") from exc
        return await self._server_apply_template(msgs, tools, add_generation_prompt, template_kwargs)

    async def tokenize(self, text: str) -> list[int]:
        """Token ids of ``text`` without BOS; special-token text (e.g.
        ``<|im_end|>``) is parsed as special tokens, like prompts in /completion."""
        if not text:
            return []
        data = await self._request_json(
            "POST", "/tokenize", {"content": text, "add_special": False, "parse_special": True}
        )
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if not isinstance(tokens, list):
            raise EngineError("Silnik modelu zwrócił niepoprawną odpowiedź (/tokenize).")
        return [int(t["id"]) if isinstance(t, dict) else int(t) for t in tokens]

    def _strip_bos(self, prompt: str) -> str:
        bos = self._bos_text
        if self._adds_bos and bos and prompt.startswith(bos):
            return prompt[len(bos):]
        return prompt

    def _completion_body(self, prompt: str, params: SamplingParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": self._strip_bos(prompt),
            "stream": True,
            "cache_prompt": True,
            "n_predict": params.max_tokens if params.max_tokens and params.max_tokens > 0 else -1,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "repeat_penalty": params.repeat_penalty,
            "stop": list(params.stop),
        }
        if params.seed is not None:
            body["seed"] = params.seed
        if params.images:
            if not self.supports_images:
                raise EngineError("Ten model nie przyjmuje obrazów — wybierz model z wizją.")
            # llama.cpp's multimodal /completion: the prompt carries one marker
            # per image and the images travel next to it, base64-encoded.
            body["prompt"] = {"prompt_string": body["prompt"], "multimodal_data": list(params.images)}
        return body

    def _stream_error(self, payload: Any) -> EngineError:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                return EngineError(f"Błąd silnika modelu podczas generowania: {payload[:500]}")
        err = payload.get("error", payload) if isinstance(payload, dict) else payload
        status = _int_or_none(err.get("code")) if isinstance(err, dict) else None
        return self._http_error(status or 500, json.dumps({"error": err}), "/completion")

    async def _post_completion(self, session: aiohttp.ClientSession, body: dict[str, Any]) -> aiohttp.ClientResponse:
        """POST /completion, retrying once on a stale pooled connection.

        The server closes idle keep-alive connections (see ``_request_json``);
        a ``ServerDisconnectedError`` here means the request never produced a
        response, so no generation was started and a retry cannot duplicate one.
        """
        for attempt in range(2):
            try:
                return await session.post(
                    f"{self.base_url}/completion",
                    json=body,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                )
            except aiohttp.ServerDisconnectedError:
                if attempt == 0 and not self._stopping and not session.closed:
                    log.debug("engine %s: stale keep-alive connection on /completion, retrying", self.instance_id)
                    continue
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        """Stream a completion of an already templated prompt (see module docstring)."""
        if not self.running:
            if self._stopping:
                raise EngineUnavailableError("Model został zatrzymany.")
            raise EngineUnavailableError("Model nie jest uruchomiony.")
        session = self._require_session()
        body = self._completion_body(prompt, params)
        t0 = time.perf_counter()
        resp: aiohttp.ClientResponse | None = None
        try:
            try:
                resp = await self._post_completion(session, body)
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                raise await self._connection_error(exc) from exc
            if resp.status >= 400:
                text = await resp.text(errors="replace")
                raise self._http_error(resp.status, text, "/completion")
            decoder = SSEDecoder()
            try:
                async for raw in resp.content.iter_any():
                    for ev in decoder.feed(raw):
                        chunk = self._chunk_from_event(ev, t0)
                        if chunk is None:
                            continue
                        if chunk.done:
                            yield chunk
                            return
                        yield chunk
                for ev in decoder.finish():
                    chunk = self._chunk_from_event(ev, t0)
                    if chunk is None:
                        continue
                    if chunk.done:
                        yield chunk
                        return
                    yield chunk
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                raise await self._connection_error(exc) from exc
            except RuntimeError as exc:
                # aiohttp raises a bare RuntimeError("Connection closed.") when the
                # session was closed under an open stream (engine stopped).
                if self._stopping or session.closed or "connection closed" in str(exc).lower():
                    raise await self._connection_error(exc) from exc
                raise
            proc = self._proc
            if self._stopping or proc is None or proc.returncode is not None:
                raise await self._connection_error(EOFError("stream ended"))
            raise EngineError("Strumień odpowiedzi modelu urwał się przed końcem generowania.")
        finally:
            if resp is not None:
                # Never keep the connection of a stream: llama-server closes it
                # right after the last frame, and a pooled connection that the
                # server is closing makes the *next* request fail with
                # ServerDisconnectedError (verified against build 10964).
                # Closing it also makes the server abort the slot when the
                # consumer stopped early.
                resp.close()

    def _chunk_from_event(self, ev: SSEEvent, t0: float) -> CompletionChunk | None:
        if ev.event == "error":
            raise self._stream_error(ev.data)
        data = ev.data.strip()
        if not data or data == "[DONE]":
            return None
        try:
            frame = json.loads(data)
        except ValueError:
            log.debug("engine %s: ignoring malformed SSE data: %.200s", self.instance_id, data)
            return None
        if not isinstance(frame, dict):
            return None
        if frame.get("error"):
            raise self._stream_error(frame)
        text = frame.get("content")
        text = text if isinstance(text, str) else ""
        if frame.get("stop"):
            wall_ms = (time.perf_counter() - t0) * 1000.0
            return CompletionChunk(
                text=text, done=True, finish_reason=finish_reason_from_frame(frame),
                stats=stats_from_frame(frame, wall_ms),
            )
        return CompletionChunk(text=text) if text else None

    # -------------------------------------------------------------- embeddings
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embedding vectors (one per text, in order). Requires an instance
        started with ``LoadConfig(embedding=True)``."""
        if not texts:
            return []
        if not self.running:
            raise EngineUnavailableError("Model nie jest uruchomiony.")
        out: list[list[float]] = []
        for i in range(0, len(texts), max(1, self.embed_batch_size)):
            out.extend(await self._embed_batch(list(texts[i : i + self.embed_batch_size])))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        vectors: list[list[float]] | None = None
        if self._embed_endpoint in (None, "/v1/embeddings"):
            try:
                data = await self._request_json(
                    "POST", "/v1/embeddings", {"input": batch, "encoding_format": "float"},
                    timeout=self.embed_timeout_s,
                )
                vectors = self._parse_openai_embeddings(data)
                self._embed_endpoint = "/v1/embeddings"
            except EngineHttpError as exc:
                if exc.status == 501:
                    raise EngineError(
                        "Ten model nie jest uruchomiony w trybie embeddingów (wczytaj go jako model do embeddingów)."
                    ) from exc
                if exc.status not in (404, 405):
                    raise
                self._embed_endpoint = "/embedding"
        if vectors is None:
            try:
                data = await self._request_json(
                    "POST", "/embedding", {"content": batch}, timeout=self.embed_timeout_s
                )
            except EngineHttpError as exc:
                if exc.status in (404, 405, 501):
                    raise EngineError("Ten model nie obsługuje embeddingów.") from exc
                raise
            vectors = self._parse_native_embeddings(data)
        if len(vectors) != len(batch):
            raise EngineError(
                f"Silnik zwrócił {len(vectors)} embeddingów zamiast {len(batch)}."
            )
        dims = {len(v) for v in vectors}
        if len(dims) > 1 or 0 in dims:
            raise EngineError("Silnik zwrócił embeddingi o niespójnych wymiarach.")
        return vectors

    @staticmethod
    def _parse_openai_embeddings(data: Any) -> list[list[float]]:
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise EngineError("Silnik modelu zwrócił niepoprawną odpowiedź (/v1/embeddings).")
        items = sorted(items, key=lambda e: int(e.get("index", 0)))
        return [[float(x) for x in e.get("embedding", [])] for e in items]

    @staticmethod
    def _parse_native_embeddings(data: Any) -> list[list[float]]:
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else None
        if items is None:
            raise EngineError("Silnik modelu zwrócił niepoprawną odpowiedź (/embedding).")
        items = sorted(items, key=lambda e: int(e.get("index", 0)))
        out: list[list[float]] = []
        for e in items:
            emb = e.get("embedding", [])
            if emb and isinstance(emb[0], list):
                if len(emb) != 1:
                    raise EngineError(
                        "Model zwraca embeddingi dla każdego tokenu osobno (brak poolingu) – ustaw --pooling."
                    )
                emb = emb[0]
            out.append([float(x) for x in emb])
        return out
