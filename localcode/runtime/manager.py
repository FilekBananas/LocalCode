"""RuntimeManager: model instances, request slots and the RAM budget.

The manager is the only component that starts or stops engines.  Everything
that wants to talk to a model (agent loop, chatbot, OpenAI-compatible API,
RAG embeddings, title generation) borrows a *slot* of a running instance::

    async with runtime.acquire("unsloth/Qwen3-4B-…/Qwen3-4B-…-Q4_K_M") as lease:
        async for chunk in lease.engine.complete(prompt, params):
            ...
        lease.record_stats(chunk.stats)

``acquire()`` returns a :class:`PendingLease`, which is a full coroutine *and*
an async context manager, so all three forms work:

    lease = await runtime.acquire(key)               # release with lease.release()
    async with runtime.acquire(key) as lease: ...    # released on exit
    task = asyncio.create_task(runtime.acquire(key)) # cancellable while it waits

Nothing is loaded or reserved until it is awaited, and cancelling the awaiting
task always leaves the manager exactly as it was (a slot granted in the same
event-loop iteration is handed back to the next waiter).

Instances and slots
-------------------
An *instance* is one engine process (llama-server / mlx worker) with
``config.n_parallel`` request slots.  States: ``loading`` → ``ready`` (shown as
``busy`` while any slot is taken) → ``stopping``; failed loads are reported as
``error`` entries in :meth:`RuntimeManager.status` for ``ERROR_TTL_S`` seconds
(at least one status snapshot) and then disappear.

Acquire algorithm (decisions are synchronous and taken under one
``asyncio.Lock``; engine start/stop always runs in background tasks, so a
model that is loading never blocks requests for other models):

1. a *ready* instance of the model with a free slot and ``n_ctx >= requested``
   → take the least busy one;
2. an *idle* instance exists but its context is too small → reload it with
   the bigger context if that fits in the limits, otherwise use it anyway (the
   caller reads the real window from ``lease.n_ctx`` / ``lease.engine.n_ctx``);
3. any free slot with a smaller context → take it;
4. no free slot: when no instance of the model exists, start one (this may
   evict idle instances, see below; if it cannot fit, :class:`EngineError`
   with a Polish explanation is raised to the caller); when all instances are
   busy, no other copy is loading and ``max_instances_per_model`` allows it,
   start one more copy *only if it fits without evicting anything*; then wait
   in a FIFO queue for a slot (cancellation-safe).

Concurrent ``acquire()`` calls for an unloaded model trigger exactly one load:
loads of the same model are serialised, and extra copies are started on
demand once the first one is ready.

Limits and eviction
-------------------
Before any *required* load (first instance of a model, context upgrade,
explicit :meth:`load`) the manager enforces ``max_loaded_models`` and the RAM
budget (``ram_limit_gb`` or 70 % of physical RAM; each instance reserves
``max(estimate, measured)`` bytes, see ``runtime/memory.py``).  With
``evict_policy="lru"`` idle instances are stopped least-recently-used first;
busy and loading instances are never evicted.  With ``evict_policy="none"``
nothing is evicted automatically.  If it still does not fit, the load is
refused with an explanation of what to change (smaller context, fewer
parallel slots, unloading a named model, raising the RAM limit).  Settings
changes (:meth:`update_settings`) are enforced immediately by unloading idle
instances LRU; busy instances are unloaded as soon as they become idle.

Idle instances are unloaded after ``idle_ttl_minutes`` (0 = never) by a
reaper task started lazily on first use; the same task refreshes measured
memory (``engine.memory_bytes()`` or the process footprint).

Unloading
---------
:meth:`unload` of an idle instance stops it at once.  A busy instance goes to
``stopping`` (no new requests), the manager waits up to ``UNLOAD_GRACE_S``
seconds for its leases to be released and then stops the engine anyway;
requests still streaming at that moment end with an engine error on the
caller's side.  Waiters that can no longer be served (their model has no
other instance) receive :class:`EngineError`.  Unloading is *idempotent*: an
id this manager has already stopped is a no-op, so a second click in the Mac
app or a ``/unload <id>`` in the terminal working from a status snapshot that
is one event behind never turns into an error; only an id the manager has
never issued is refused.  :meth:`shutdown` does the same for everything with
a ``SHUTDOWN_GRACE_S`` grace period.

Events
------
``on_event({"type": "runtime", **status()})`` is emitted after every state
change, coalesced to at most one event per event-loop iteration.  The
callback may be sync or async; its exceptions are logged and ignored.

All methods must be called from the event loop that owns the manager.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import itertools
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Generator, Iterable, Literal, Protocol

from localcode import config as lc_config
from localcode.config import Settings
from localcode.hub.types import LocalModel, MemoryEstimate
from localcode.runtime import memory
from localcode.security import sandbox
from localcode.runtime.base import Engine, EngineError, GenStats, LoadConfig

log = logging.getLogger(__name__)

EngineFactory = Callable[[str, LocalModel, LoadConfig], Engine]
InstanceInfo = dict[str, Any]          # camelCase instance DTO (see status())
Purpose = Literal["chat", "agent", "embedding", "title"]

# Tunables (module-level so tests and integrators can adjust them).
REAPER_INTERVAL_S = 15.0       # idle-TTL / measurement / limit enforcement tick
UNLOAD_GRACE_S = 10.0          # wait for leases of a busy instance before stopping it
SHUTDOWN_GRACE_S = 2.0
ERROR_TTL_S = 30.0             # failed loads stay visible in status() this long
MEASURE_TIMEOUT_S = 5.0
GONE_IDS_MEMORY = 128          # instance ids remembered so that unload() stays idempotent
TPS_EMA_ALPHA = 0.3
TPS_MIN_TOKENS = 4             # ignore gen_tps of tiny generations (too noisy)
AUTO_CTX_COMFORT = 16384       # auto context below this -> prefer evicting idle models

_ACTIVE_STATES = ("loading", "ready")

#: Stops with no replacement instance started in the same step.  Waiters of a
#: model whose last instance goes away this way must be told, not left in the
#: queue forever ("evict"/"reload" always start a replacement right after).
_FINAL_STOP_REASONS = ("unload", "shutdown", "idle ttl", "limits", "make_room")

_CLOSED_MESSAGE = "Środowisko uruchomieniowe modeli zostało zamknięte."


class ModelLibraryLike(Protocol):
    """What the manager needs from ``hub.library.ModelLibrary`` (duck-typed)."""

    def get(self, key: str) -> LocalModel | None: ...

    def load_defaults(self, key: str) -> dict[str, Any] | None: ...


class SettingsError(ValueError):
    """Invalid runtime settings patch (message is Polish, safe to show)."""


# --------------------------------------------------------------------------- #
# Instance record and lease
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class _Instance:
    id: str
    model: LocalModel
    config: LoadConfig
    embedding: bool
    estimated_bytes: int
    engine: Engine | None = None
    state: str = "loading"                  # loading | ready | stopping | error
    busy: int = 0
    loaded_at: float | None = None          # wall clock (s)
    last_used: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.time)
    measured_bytes: int | None = None
    gen_tps: float | None = None
    error: str | None = None
    failed_at: float | None = None          # monotonic, error records only
    shown: bool = False                     # error record included in a status snapshot
    load_task: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[Any] | None = None
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    drained: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def slots(self) -> int:
        return max(1, int(self.config.n_parallel or 1))

    @property
    def free_slots(self) -> int:
        return self.slots - self.busy

    @property
    def n_ctx(self) -> int:
        if self.engine is not None and self.state in ("ready", "stopping"):
            try:
                value = int(self.engine.n_ctx)
                if value > 0:
                    return value
            except Exception:  # engine property failing must not break the manager
                log.debug("engine.n_ctx failed for %s", self.id, exc_info=True)
        return int(self.config.n_ctx)

    @property
    def reserved_bytes(self) -> int:
        return max(self.estimated_bytes, self.measured_bytes or 0)

    @property
    def display_state(self) -> str:
        if self.state == "ready" and self.busy > 0:
            return "busy"
        return self.state

    def touch(self) -> None:
        self.last_used = time.monotonic()
        self.last_used_at = time.time()


@dataclass(eq=False)
class _Waiter:
    """One ``acquire()`` call parked in the FIFO queue of ``(model key, embedding)``."""

    model: LocalModel
    n_ctx: int | None
    purpose: str
    embedding: bool
    future: asyncio.Future[_Instance]

    @property
    def queue_key(self) -> tuple[str, bool]:
        return (self.model.key, self.embedding)


class Lease:
    """One borrowed request slot of a running instance.

    Release it exactly once (``release()``, ``await aclose()`` or leaving
    ``async with``); further releases are no-ops.  ``engine`` stays valid
    until the lease is released.
    """

    __slots__ = ("_manager", "_inst", "purpose", "_released")

    def __init__(self, manager: RuntimeManager, inst: _Instance, purpose: str) -> None:
        self._manager = manager
        self._inst = inst
        self.purpose = purpose
        self._released = False

    @property
    def engine(self) -> Engine:
        engine = self._inst.engine
        assert engine is not None, "lease on an instance without an engine"
        return engine

    @property
    def instance_id(self) -> str:
        return self._inst.id

    @property
    def model_key(self) -> str:
        return self._inst.model.key

    @property
    def model(self) -> LocalModel:
        return self._inst.model

    @property
    def config(self) -> LoadConfig:
        return self._inst.config

    @property
    def n_ctx(self) -> int:
        """Real context window per slot (may be smaller than requested)."""
        return self._inst.n_ctx

    @property
    def released(self) -> bool:
        return self._released

    def record_stats(self, stats: GenStats | None) -> None:
        """Feed generation statistics (updates the instance's tokens/s EMA)."""
        if stats is not None:
            self._manager._record_stats(self._inst, stats)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager._release_slot(self._inst)

    async def aclose(self) -> None:
        self.release()

    async def __aenter__(self) -> Lease:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return f"<Lease {self._inst.id} ({self.purpose}, {state})>"


class PendingLease(collections.abc.Coroutine[Any, Any, "Lease"]):
    """Result of :meth:`RuntimeManager.acquire`: awaitable *and* async context manager.

    ``lease = await runtime.acquire(key)`` or
    ``async with runtime.acquire(key) as lease: ...``.  Nothing happens until it
    is awaited or entered.

    It implements the full coroutine protocol (``send``/``throw``/``close``) and
    is registered as :class:`collections.abc.Coroutine`, so that
    ``asyncio.create_task(runtime.acquire(key))`` — the natural way to acquire a
    slot in the background and cancel it later — works exactly like with a plain
    coroutine.  The underlying coroutine is created lazily on first use, so a
    ``PendingLease`` that is never awaited neither acquires anything nor warns
    about a coroutine that was never awaited.
    """

    __slots__ = ("_factory", "_coro", "_lease", "__weakref__")

    def __init__(self, factory: Callable[[], Coroutine[Any, Any, Lease]]) -> None:
        self._factory = factory
        self._coro: Coroutine[Any, Any, Lease] | None = None
        self._lease: Lease | None = None

    # coroutine protocol ---------------------------------------------------
    def _coroutine(self) -> Coroutine[Any, Any, Lease]:
        if self._coro is None:
            self._coro = self._factory()
        return self._coro

    def __await__(self) -> Generator[Any, None, Lease]:
        return self._coroutine().__await__()

    def send(self, value: Any) -> Any:
        return self._coroutine().send(value)

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        return self._coroutine().throw(*args, **kwargs)

    def close(self) -> None:
        if self._coro is not None:
            self._coro.close()

    # async context manager ------------------------------------------------
    async def __aenter__(self) -> Lease:
        self._lease = await self
        return self._lease

    async def __aexit__(self, *exc: object) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.release()

    def __repr__(self) -> str:
        return f"<PendingLease {'started' if self._coro is not None else 'pending'}>"


# --------------------------------------------------------------------------- #
# Settings validation
# --------------------------------------------------------------------------- #

_SETTING_KEYS: dict[str, str] = {
    "maxLoadedModels": "max_loaded_models",
    "maxInstancesPerModel": "max_instances_per_model",
    "defaultParallel": "default_parallel",
    "ramLimitGb": "ram_limit_gb",
    "idleTtlMinutes": "idle_ttl_minutes",
    "defaultCtx": "default_ctx",
    "flashAttn": "flash_attn",
    "kvCacheType": "kv_cache_type",
    "gpuLayers": "gpu_layers",
    "evictPolicy": "evict_policy",
    "llamaServerPath": "llama_server_path",
    "pythonPath": "python_path",
    "pinnedModels": "pinned_models",
}
_SNAKE_TO_CAMEL = {v: k for k, v in _SETTING_KEYS.items()}

_INT_RANGES: dict[str, tuple[int, int]] = {
    "max_loaded_models": (1, 16),
    "max_instances_per_model": (1, 8),
    "default_parallel": (1, 32),
    "idle_ttl_minutes": (0, 7 * 24 * 60),
    "gpu_layers": (-1, 999),
}
_CHOICES: dict[str, tuple[str, ...]] = {
    "flash_attn": ("auto", "on", "off"),
    "kv_cache_type": ("f16", "bf16", "q8_0", "q4_0", "8bit", "4bit"),
    "evict_policy": ("lru", "none"),
}
CTX_RANGE = (512, 1_048_576)


def _as_int(camel: str, value: Any, lo: int, hi: int) -> int:
    ok = False
    number = 0
    if isinstance(value, bool):
        ok = False
    elif isinstance(value, int):
        number, ok = value, True
    elif isinstance(value, float) and value.is_integer():
        number, ok = int(value), True
    elif isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        number, ok = int(value), True
    if not ok or not lo <= number <= hi:
        raise SettingsError(
            f"Nieprawidłowa wartość „{camel}”: podaj liczbę całkowitą od {lo} do {hi}.")
    return number


def _as_path(camel: str, value: Any) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise SettingsError(f"Nieprawidłowa wartość „{camel}”: oczekiwano ścieżki do pliku.")
    raw = value.strip()
    path = os.path.expanduser(raw)
    if os.sep not in raw and not raw.startswith("~"):
        found = shutil.which(raw)
        if found:
            return found
    if not os.path.isfile(path):
        raise SettingsError(f"Nie znaleziono pliku „{raw}” (ustawienie „{camel}”).")
    if not os.access(path, os.X_OK):
        raise SettingsError(f"Plik „{raw}” nie jest wykonywalny (ustawienie „{camel}”).")
    return path


def validate_settings_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate a camelCase (or snake_case) runtime settings patch.

    Returns the snake_case fields to store; raises :class:`SettingsError`
    with a Polish message on the first invalid entry.
    """
    if not isinstance(patch, dict):
        raise SettingsError("Nieprawidłowe ustawienia: oczekiwano obiektu JSON.")
    out: dict[str, Any] = {}
    for key, value in patch.items():
        name = _SETTING_KEYS.get(key) or (key if key in _SNAKE_TO_CAMEL else None)
        if name is None:
            raise SettingsError(f"Nieznane ustawienie środowiska modeli: „{key}”.")
        camel = _SNAKE_TO_CAMEL[name]
        if name == "pinned_models":
            if value is None or value == "":
                out[name] = []
                continue
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise SettingsError("„pinnedModels” musi być listą kluczy modeli.")
            out[name] = [v for v in dict.fromkeys(v.strip() for v in value) if v]
            continue
        if name in _INT_RANGES:
            out[name] = _as_int(camel, value, *_INT_RANGES[name])
        elif name == "ram_limit_gb":
            if value is None or value == "":
                out[name] = None
                continue
            physical_gb = memory.system_memory()["total"] / memory.GiB
            try:
                if isinstance(value, bool):
                    raise ValueError
                gb = float(value)
            except (TypeError, ValueError):
                gb = float("nan")
            if not (math.isfinite(gb) and 1.0 <= gb <= physical_gb + 1e-6):
                raise SettingsError(
                    f"Nieprawidłowy limit RAM: podaj od 1 do {physical_gb:.0f} GB "
                    "albo zostaw puste (automatycznie 70% pamięci).")
            out[name] = round(gb, 2)
        elif name == "default_ctx":
            if value is None or value == "" or value == 0:
                out[name] = None
                continue
            out[name] = _as_int(camel, value, *CTX_RANGE)
        elif name in _CHOICES:
            text = str(value).strip().lower() if isinstance(value, str) else None
            if text not in _CHOICES[name]:
                allowed = ", ".join(_CHOICES[name])
                raise SettingsError(f"Nieprawidłowa wartość „{camel}”: dozwolone {allowed}.")
            out[name] = text
        else:  # llama_server_path / python_path
            out[name] = _as_path(camel, value)
    return out


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


def _check_can_generate(model: LocalModel, purpose: str) -> None:
    """Refuse a model that cannot produce text before the engine is even asked.

    An embedding model answers `/completion` with a raw llama-server 500 ("the
    current context does not logits computation") and an image model is not a
    language model at all; both need to be reported in the user's language.
    """
    if purpose != "embedding" and model.is_embedding:
        raise EngineError(
            f"Model „{model.key}” to model osadzeń (embeddingi) — służy do wyszukiwania "
            f"w dokumentach, nie do generowania tekstu. Wybierz model językowy.")
    if model.is_image:
        raise EngineError(
            f"Model „{model.key}” to model obrazów — nie prowadzi rozmowy. "
            f"Użyj go w „localcode image”, a do rozmowy wybierz model językowy.")


class RuntimeManager:
    """Owns every running model instance; see the module docstring."""

    def __init__(
        self,
        settings: Settings,
        library: ModelLibraryLike,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.settings = settings
        self.library = library
        self.on_event = on_event
        self._engine_factory: EngineFactory = engine_factory or self._default_engine_factory
        self._instances: dict[str, _Instance] = {}
        self._errors: list[_Instance] = []
        self._gone: collections.deque[str] = collections.deque(maxlen=GONE_IDS_MEMORY)
        self._waiters: dict[tuple[str, bool], collections.deque[_Waiter]] = {}
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)
        self._reaper: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._notify_pending = False
        self._closed = False
        #: model keys explicitly approved past a "dangerous" security scan verdict
        #: (see LoadConfig.allow_dangerous / _check_security_scan); sticky for the
        #: process lifetime so a later reload/scale-up of the same model doesn't
        #: ask again, but never persisted to disk.
        self._dangerous_overridden: set[str] = set()
        #: Models reached over an API key.  Built lazily and only when a remote
        #: key is actually used, so a machine with no keys never touches it.
        self._remote: Any | None = None

    # ------------------------------------------------------------------ public

    @property
    def runtime(self) -> Any:
        return self.settings.runtime

    def acquire(self, model_key: str, *, n_ctx: int | None = None,
                purpose: Purpose | str = "chat") -> PendingLease:
        """Borrow one request slot of ``model_key``, loading it just in time.

        ``n_ctx`` is the context per slot the caller would like; the lease may
        end up with a smaller window (``lease.n_ctx``) when a bigger one does
        not fit.  ``purpose``: ``chat`` | ``agent`` | ``embedding`` (loads the
        instance in embedding mode) | ``title`` (never triggers a reload or
        an extra copy).  Raises :class:`EngineError` (Polish message) when the
        model is unknown, cannot fit, or fails to start.
        """
        return PendingLease(lambda: self._acquire(model_key, n_ctx, str(purpose)))

    async def load(self, model_key: str, config: LoadConfig | None = None, *,
                   new_instance: bool = False, wait: bool = True) -> InstanceInfo:
        """Explicitly load a model (UI "Załaduj").

        Without ``new_instance`` an existing instance with the same config is
        reused; an idle instance with a different config is replaced.  With
        ``wait=False`` returns right after scheduling (state ``loading``) –
        progress and failures are then reported through events/status.
        """
        self._check_open()
        self._ensure_reaper()
        async with self._lock:
            model = self._resolve_model(model_key)
            embedding = model.is_embedding or bool(config is not None and config.embedding)
            same = self._instances_for(model.key, embedding)
            cfg = self._build_config(model, n_ctx=None, embedding=embedding, explicit=config)
            inst: _Instance | None = None
            if not new_instance and same:
                inst = next((i for i in same if config is None or _same_config(i.config, cfg)), None)
                if inst is not None:
                    inst.touch()           # an explicit load restarts the idle TTL
                if inst is None:
                    idle = sorted((i for i in same if self._is_idle(i)), key=lambda i: i.last_used)
                    if idle:
                        target = idle[0]
                        victims = self._plan_admission(model, cfg, exclude={target.id})
                        self._begin_stop(target, grace=0.0, reason="reload")
                        inst = self._start_instance(model, cfg, embedding, [*victims, target])
                    elif len(same) < self._max_instances():
                        victims = self._plan_admission(model, cfg)
                        inst = self._start_instance(model, cfg, embedding, victims)
                    else:
                        raise EngineError(
                            f"Model {model.display_name} jest teraz używany z inną konfiguracją, "
                            f"a osiągnięto limit instancji na model ({self._max_instances()}). "
                            "Poczekaj na zakończenie odpowiedzi, wyładuj go albo zwiększ "
                            "„Instancje na model” w ustawieniach Runtime.")
            else:
                if len(same) >= self._max_instances():
                    raise EngineError(
                        f"Osiągnięto limit instancji na model ({self._max_instances()}). "
                        "Zwiększ „Instancje na model” w ustawieniach Runtime, aby uruchomić "
                        "kolejną kopię.")
                victims = self._plan_admission(model, cfg)
                inst = self._start_instance(model, cfg, embedding, victims)
        if wait and inst.state == "loading":
            await inst.ready_event.wait()
            if inst.state == "error":
                raise EngineError(inst.error or "Nie udało się załadować modelu.")
            if inst.id not in self._instances or inst.state != "ready":
                raise EngineError(f"Ładowanie modelu {model.display_name} zostało przerwane.")
        return self._instance_dto(inst)

    async def unload(self, instance_id: str) -> None:
        """Unload one instance (see "Unloading" in the module docstring)."""
        async with self._lock:
            inst = self._instances.get(instance_id)
            if inst is None:
                before = len(self._errors)
                self._errors = [e for e in self._errors if e.id != instance_id]
                if len(self._errors) != before:
                    self._notify()
                    return
                if instance_id in self._gone:
                    # Already unloaded (a second click, a stale row in the UI, a
                    # retried request): unloading is idempotent, not an error.
                    return
                raise EngineError(f"Nie ma załadowanej instancji „{instance_id}”.")
            task = self._begin_stop(inst, grace=UNLOAD_GRACE_S, reason="unload")
        if task is not None:
            await asyncio.wait({task})

    async def unload_all(self) -> None:
        """Unload every instance (busy ones after the grace period)."""
        async with self._lock:
            tasks = [self._begin_stop(i, grace=UNLOAD_GRACE_S, reason="unload")
                     for i in list(self._instances.values())]
            self._errors.clear()
            self._notify()
        pending = {t for t in tasks if t is not None}
        if pending:
            await asyncio.wait(pending)

    async def make_room(self, n_bytes: int, *, exclude: Iterable[str] = ()) -> bool:
        """Free ``n_bytes`` of the shared RAM budget for a consumer outside this
        manager (the image runtime, ``localcode/images/runtime.py``).

        Idle instances are stopped least-recently-used first until the budget has
        room for ``n_bytes`` on top of what is still loaded; busy and loading
        instances are never touched, and ``evict_policy="none"`` evicts nothing.
        ``exclude`` names instances (by id or model key) that must be kept.
        Returns True when the room is available (or already was), False when even
        after evicting everything that could go the request does not fit.
        """
        need = max(0, int(n_bytes))
        keep = {str(x) for x in exclude}
        tasks: set[asyncio.Task[Any]] = set()
        async with self._lock:
            budget = self._budget()
            active = self._active()
            used = sum(i.reserved_bytes for i in active)
            if used + need > budget and self.runtime.evict_policy == "lru":
                victims = sorted((i for i in active
                                  if self._is_idle(i) and i.id not in keep and i.model.key not in keep),
                                 key=lambda i: i.last_used)
                for victim in victims:
                    if used + need <= budget:
                        break
                    task = self._begin_stop(victim, grace=0.0, reason="make_room")
                    if task is not None:
                        tasks.add(task)
                    used -= victim.reserved_bytes
            fits = used + need <= budget
        if tasks:
            await asyncio.wait(tasks)
        if not fits:
            log.info("make_room(%d) could not free enough of the %d byte budget", need, self._budget())
        return fits

    async def shutdown(self) -> None:
        """Stop the reaper, fail all waiters and stop every engine. Idempotent."""
        self._closed = True
        if self._reaper is not None:
            self._reaper.cancel()
            await asyncio.wait({self._reaper})
            self._reaper = None
        async with self._lock:
            for queue in self._waiters.values():
                for w in queue:
                    if not w.future.done():
                        w.future.set_exception(EngineError(_CLOSED_MESSAGE))
            self._waiters.clear()
            tasks = [self._begin_stop(i, grace=SHUTDOWN_GRACE_S, reason="shutdown")
                     for i in list(self._instances.values())]
        pending = {t for t in tasks if t is not None}
        if pending:
            await asyncio.wait(pending)
        others = {t for t in self._bg_tasks if not t.done()}
        for t in others:
            t.cancel()
        if others:
            await asyncio.wait(others)
        self._notify()

    def estimate(self, model_key: str, config: LoadConfig | None = None) -> MemoryEstimate:
        """Memory estimate for loading ``model_key`` now (UI "Szacowanie")."""
        model = self._resolve_model(model_key)
        embedding = model.is_embedding or bool(config is not None and config.embedding)
        cfg = self._build_config(model, n_ctx=None, embedding=embedding, explicit=config)
        total = memory.breakdown(model, cfg).total
        victims, _ = self._eviction_plan(total, exclude=set(), allow_evict=True)
        active = self._active()
        in_use = sum(i.reserved_bytes for i in active) - sum(v.reserved_bytes for v in victims)
        reclaimable = sum(v.measured_bytes or v.estimated_bytes for v in victims)
        est = memory.estimate(model, cfg, budget=self._budget(), in_use=in_use,
                              reclaimable=reclaimable)
        if victims and est.fits != "no":
            names = ", ".join(v.model.display_name for v in victims)
            est.message = f"{est.message} Zostanie wyładowany: {names}."
        return est

    async def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate, persist and apply a runtime settings patch; returns status()."""
        changes = validate_settings_patch(patch)
        if changes:
            try:
                await asyncio.to_thread(lc_config.update_settings, {"runtime": dict(changes)})
            except OSError as exc:
                raise SettingsError(f"Nie udało się zapisać ustawień: {exc}") from exc
            async with self._lock:
                for name, value in changes.items():
                    setattr(self.runtime, name, value)
                tasks = self._enforce_limits()
                self._notify()
            if tasks:
                await asyncio.wait(tasks)
        return self.status()

    def status(self) -> dict[str, Any]:
        """RuntimeStatus DTO (camelCase, see docs/ARCHITECTURE.md HTTP API)."""
        self._purge_errors()
        instances = [self._instance_dto(i) for i in self._instances.values()]
        for err in self._errors:
            instances.append(self._instance_dto(err))
            err.shown = True
        sysmem = memory.system_memory()
        rt = self.runtime
        return {
            "instances": instances,
            "system": {
                "totalBytes": sysmem["total"],
                "usedBytes": sysmem["used"],
                "availableBytes": sysmem["available"],
            },
            "budget": {
                "limitBytes": self._budget(),
                "usedBytes": sum(i.reserved_bytes for i in self._instances.values()),
            },
            "settings": {
                "maxLoadedModels": rt.max_loaded_models,
                "maxInstancesPerModel": rt.max_instances_per_model,
                "defaultParallel": rt.default_parallel,
                "ramLimitGb": rt.ram_limit_gb,
                "idleTtlMinutes": rt.idle_ttl_minutes,
                "pinnedModels": list(getattr(rt, "pinned_models", None) or []),
                "defaultCtx": rt.default_ctx,
                "flashAttn": rt.flash_attn,
                "kvCacheType": rt.kv_cache_type,
                "gpuLayers": rt.gpu_layers,
                "evictPolicy": rt.evict_policy,
            },
        }

    def instances(self) -> list[InstanceInfo]:
        """DTOs of the current (non-error) instances."""
        return [self._instance_dto(i) for i in self._instances.values()]

    def is_loaded(self, model_key: str) -> bool:
        """True when an instance of the model is loading or ready."""
        return any(i.model.key == model_key and i.state in _ACTIVE_STATES
                   for i in self._instances.values())

    def loaded_keys(self) -> set[str]:
        return {i.model.key for i in self._instances.values() if i.state in _ACTIVE_STATES}

    # ------------------------------------------------------------ acquisition

    @property
    def remote(self) -> Any:
        """The registry of remote providers (:mod:`localcode.remote`)."""
        if self._remote is None:
            from localcode.remote import RemoteRegistry

            self._remote = RemoteRegistry(self.settings)
        return self._remote

    async def _acquire(self, model_key: str, n_ctx: int | None, purpose: str) -> Any:
        self._check_open()
        self._ensure_reaper()
        # A remote model has no instance, no slot and no memory to account for:
        # it is a lease over an HTTP client (see localcode/remote/engine.py).
        from localcode.remote import is_remote_key

        if is_remote_key(model_key):
            return await self.remote.lease(model_key, n_ctx=n_ctx, purpose=purpose)
        if n_ctx is not None and int(n_ctx) <= 0:
            n_ctx = None
        async with self._lock:
            model = self._resolve_model(model_key)
            _check_can_generate(model, purpose)
            decision = self._decide(model, int(n_ctx) if n_ctx else None, purpose)
        if isinstance(decision, Lease):
            return decision
        waiter = decision
        try:
            inst = await waiter.future
        except asyncio.CancelledError:
            self._abandon_waiter(waiter)
            raise
        return Lease(self, inst, purpose)

    def _decide(self, model: LocalModel, n_ctx: int | None, purpose: str) -> Lease | _Waiter:
        key = model.key
        embedding = purpose == "embedding" or model.is_embedding
        native = _native_ctx(model)
        if n_ctx and native:
            n_ctx = min(n_ctx, native)
        same = self._instances_for(key, embedding)

        inst = self._pick_free(same, n_ctx, strict=True)
        if inst is not None:
            return self._occupy(inst, purpose)

        loading = any(i.state == "loading" for i in same)
        if n_ctx and not loading and purpose != "title":
            small_idle = [i for i in same if self._is_idle(i) and i.n_ctx < n_ctx]
            if small_idle:
                target = max(small_idle, key=lambda i: i.n_ctx)
                try:
                    cfg = self._build_config(model, n_ctx=n_ctx, embedding=embedding,
                                             exclude={target.id})
                    if cfg.n_ctx <= target.n_ctx:
                        return self._occupy(target, purpose)
                    victims = self._plan_admission(model, cfg, exclude={target.id})
                except EngineError as exc:
                    log.info("context upgrade of %s not possible: %s", target.id, exc)
                    return self._occupy(target, purpose)
                self._begin_stop(target, grace=0.0, reason="reload")
                self._start_instance(model, cfg, embedding, [*victims, target])
                return self._enqueue(model, n_ctx, purpose, embedding)

        inst = self._pick_free(same, n_ctx, strict=False)
        if inst is not None:
            return self._occupy(inst, purpose)

        if not same:
            cfg = self._build_config(model, n_ctx=n_ctx, embedding=embedding)
            victims = self._plan_admission(model, cfg)       # raises EngineError to the caller
            self._start_instance(model, cfg, embedding, victims)
        elif not loading and purpose != "title":
            self._try_scale_up(model, embedding, n_ctx)
        return self._enqueue(model, n_ctx, purpose, embedding)

    def _enqueue(self, model: LocalModel, n_ctx: int | None, purpose: str, embedding: bool) -> _Waiter:
        waiter = _Waiter(model=model, n_ctx=n_ctx, purpose=purpose, embedding=embedding,
                         future=asyncio.get_running_loop().create_future())
        self._waiters.setdefault(waiter.queue_key, collections.deque()).append(waiter)
        return waiter

    def _drop_waiter(self, waiter: _Waiter) -> None:
        queue = self._waiters.get(waiter.queue_key)
        if queue is None:
            return
        try:
            queue.remove(waiter)
        except ValueError:
            return
        if not queue:
            del self._waiters[waiter.queue_key]

    def _abandon_waiter(self, waiter: _Waiter) -> None:
        """The caller gave up (cancellation): leave the manager exactly as it was.

        A slot that was already granted goes back to the instance (and from
        there to the next waiter); a queued waiter is removed and its future
        cancelled so that nothing later tries to serve it, and a failure that
        arrived at the same moment is consumed instead of being reported as an
        unretrieved exception.
        """
        fut = waiter.future
        self._drop_waiter(waiter)          # no-op once _dispatch() has popped it
        if not fut.done():
            fut.cancel()
            return
        if fut.cancelled():
            return
        if fut.exception() is None:
            self._release_slot(fut.result())

    def _occupy(self, inst: _Instance, purpose: str) -> Lease:
        inst.busy += 1
        inst.drained.clear()
        inst.touch()
        self._notify()
        return Lease(self, inst, purpose)

    def _pick_free(self, candidates: list[_Instance], n_ctx: int | None, *, strict: bool) -> _Instance | None:
        free = [i for i in candidates if i.state == "ready" and i.free_slots > 0
                and (not strict or not n_ctx or i.n_ctx >= n_ctx)]
        if not free:
            return None
        return min(free, key=lambda i: (i.busy / i.slots, i.busy, -i.n_ctx))

    def _release_slot(self, inst: _Instance) -> None:
        inst.busy = max(0, inst.busy - 1)
        inst.touch()
        if inst.busy == 0:
            inst.drained.set()
        if inst.state == "ready":
            self._dispatch(inst.model.key, inst.embedding)
            if inst.busy == 0 and self._over_limits():
                self._enforce_limits()
        self._notify()

    def _dispatch(self, key: str, embedding: bool) -> None:
        """Hand free slots to waiters (FIFO); start another copy if demand remains."""
        qkey = (key, embedding)
        queue = self._waiters.get(qkey)
        if not queue:
            return
        same = self._instances_for(key, embedding)
        while queue:
            waiter = queue[0]
            if waiter.future.done():
                queue.popleft()
                continue
            inst = (self._pick_free(same, waiter.n_ctx, strict=True)
                    or self._pick_free(same, waiter.n_ctx, strict=False))
            if inst is None:
                break
            queue.popleft()
            inst.busy += 1
            inst.drained.clear()
            inst.touch()
            waiter.future.set_result(inst)
        if not queue:
            self._waiters.pop(qkey, None)
            return
        head = queue[0]
        if head.purpose != "title" and not any(i.state == "loading" for i in same):
            self._try_scale_up(head.model, embedding, head.n_ctx)

    def _fail_orphaned_waiters(self, key: str, embedding: bool, error: EngineError) -> None:
        """Fail waiters whose model has no loading/ready instance left."""
        if self._instances_for(key, embedding):
            self._dispatch(key, embedding)
            return
        queue = self._waiters.pop((key, embedding), None)
        for waiter in queue or ():
            if not waiter.future.done():
                waiter.future.set_exception(error)

    def _record_stats(self, inst: _Instance, stats: GenStats) -> None:
        if stats.gen_tps and stats.gen_tps > 0 and stats.completion_tokens >= TPS_MIN_TOKENS:
            tps = float(stats.gen_tps)
            inst.gen_tps = tps if inst.gen_tps is None else (
                TPS_EMA_ALPHA * tps + (1 - TPS_EMA_ALPHA) * inst.gen_tps)
            self._notify()
        inst.touch()

    # ---------------------------------------------------------------- loading

    def _check_security_scan(self, model: LocalModel, cfg: LoadConfig) -> None:
        """Refuse to spawn an engine for a model whose scan (``security/scan.py``) says "dangerous".

        Runs once per new engine process (called from ``_start_instance``), not
        on every request against an already-running instance. ``scan=None``
        (not scanned yet, or the scanner itself failed) is not treated as
        dangerous — only an explicit "dangerous" verdict blocks, so a library
        that predates this feature or a scan error never bricks a working
        setup. An override is sticky per model key for the process lifetime
        (see ``self._dangerous_overridden``), so scaling up or reloading with
        a bigger context after an explicit "load anyway" doesn't ask again.
        """
        if cfg.allow_dangerous:
            self._dangerous_overridden.add(model.key)
            return
        if model.key in self._dangerous_overridden:
            return
        scan = model.scan
        if not isinstance(scan, dict) or scan.get("verdict") != "dangerous":
            return
        summary = str(scan.get("summary") or "Skanowanie plików modelu wykryło niebezpieczne elementy.")
        raise EngineError(
            f"Model „{model.display_name}” nie został wczytany ze względów bezpieczeństwa: {summary} "
            "Jeśli mimo to chcesz go uruchomić, wczytaj go ręcznie w panelu Runtime z opcją "
            "„wczytaj mimo to” (allowDangerous).")

    def _start_instance(self, model: LocalModel, cfg: LoadConfig, embedding: bool,
                        victims: list[_Instance]) -> _Instance:
        self._check_security_scan(model, cfg)
        for victim in victims:
            if victim.state != "stopping":
                self._begin_stop(victim, grace=0.0, reason="evict")
        wait_for = list(victims)
        wait_for += [i for i in self._instances.values()
                     if i.state == "stopping" and i.model.key == model.key and i not in wait_for]
        inst = _Instance(
            id=self._new_id(model),
            model=model,
            config=cfg,
            embedding=embedding,
            estimated_bytes=memory.breakdown(model, cfg).total,
        )
        self._instances[inst.id] = inst
        self._gone.append(inst.id)
        inst.load_task = self._spawn(self._run_load(inst, wait_for))
        self._notify()
        return inst

    async def _run_load(self, inst: _Instance, wait_for: list[_Instance]) -> None:
        key, embedding = inst.model.key, inst.embedding
        try:
            stops = {t for t in (v.stop_task for v in wait_for) if t is not None and not t.done()}
            if stops:
                await asyncio.wait(stops)
            engine = self._engine_factory(inst.id, inst.model, inst.config)
            inst.engine = engine
            await engine.start()
        except asyncio.CancelledError:
            await self._stop_engine_quietly(inst)
            self._instances.pop(inst.id, None)
            inst.state = "stopping"
            self._fail_orphaned_waiters(key, embedding, EngineError(
                f"Ładowanie modelu {inst.model.display_name} zostało przerwane."))
            self._notify()
            inst.ready_event.set()
            raise
        except Exception as exc:  # noqa: BLE001 - every failure is reported to the user
            error = exc if isinstance(exc, EngineError) else EngineError(
                f"Nie udało się uruchomić modelu {inst.model.display_name}: {exc}")
            if not isinstance(exc, EngineError):
                log.exception("engine start failed for %s", inst.id)
            await self._stop_engine_quietly(inst)
            self._instances.pop(inst.id, None)
            inst.state = "error"
            inst.error = str(error) or f"Nie udało się uruchomić modelu {inst.model.display_name}."
            inst.failed_at = time.monotonic()
            inst.busy = 0
            self._errors.append(inst)
            self._fail_orphaned_waiters(key, embedding, EngineError(inst.error))
            self._notify()
            inst.ready_event.set()
            return
        if inst.state == "stopping":         # unloaded while start() was finishing
            await self._stop_engine_quietly(inst)
            self._instances.pop(inst.id, None)
            self._notify()
            inst.ready_event.set()
            return
        inst.state = "ready"
        inst.loaded_at = time.time()
        inst.touch()
        self._dispatch(key, embedding)
        self._notify()
        inst.ready_event.set()
        await self._measure(inst)

    async def _stop_engine_quietly(self, inst: _Instance) -> None:
        engine = inst.engine
        if engine is None:
            return
        try:
            await engine.stop()
        except Exception:  # noqa: BLE001
            log.warning("stopping engine of %s failed", inst.id, exc_info=True)

    def _try_scale_up(self, model: LocalModel, embedding: bool, n_ctx: int | None) -> bool:
        """Start one more copy of a busy model if it fits without evicting anything."""
        if self._closed:
            return False
        same = self._instances_for(model.key, embedding)
        if not same or len(same) >= self._max_instances() or any(i.state == "loading" for i in same):
            return False
        template = max(same, key=lambda i: i.last_used)
        cfg = _copy_config(template.config)
        if n_ctx and n_ctx > cfg.n_ctx:
            cfg.n_ctx = n_ctx
        try:
            self._plan_admission(model, cfg, allow_evict=False)
        except EngineError as exc:
            log.info("not starting another copy of %s: %s", model.key, exc)
            return False
        self._start_instance(model, cfg, embedding, [])
        return True

    # --------------------------------------------------------------- stopping

    def _begin_stop(self, inst: _Instance, *, grace: float, reason: str) -> asyncio.Task[Any] | None:
        """Mark ``inst`` as stopping and schedule its engine shutdown (sync)."""
        if inst.state == "stopping":
            return inst.stop_task
        if inst.state == "error":
            return None
        log.info("stopping instance %s (%s)", inst.id, reason)
        previous = inst.state
        inst.state = "stopping"
        if previous == "loading" and inst.load_task is not None and not inst.load_task.done():
            inst.load_task.cancel()
            inst.stop_task = inst.load_task
        else:
            if inst.busy == 0:
                inst.drained.set()
            inst.stop_task = self._spawn(self._finish_stop(inst, grace))
        if reason in _FINAL_STOP_REASONS and not self._instances_for(inst.model.key, inst.embedding):
            message = (_CLOSED_MESSAGE if reason == "shutdown"
                       else f"Model {inst.model.display_name} został wyładowany.")
            self._fail_orphaned_waiters(inst.model.key, inst.embedding, EngineError(message))
        self._notify()
        return inst.stop_task

    async def _finish_stop(self, inst: _Instance, grace: float) -> None:
        try:
            if inst.busy > 0 and grace > 0:
                try:
                    await asyncio.wait_for(inst.drained.wait(), grace)
                except asyncio.TimeoutError:
                    log.warning("instance %s still busy after %.0fs, stopping anyway", inst.id, grace)
            await self._stop_engine_quietly(inst)
        finally:
            if self._instances.get(inst.id) is inst:
                del self._instances[inst.id]
            self._notify()

    # --------------------------------------------------------- limits/budget

    def _budget(self) -> int:
        return memory.default_ram_budget(self.runtime)

    def _max_instances(self) -> int:
        return max(1, int(self.runtime.max_instances_per_model or 1))

    def _max_loaded(self) -> int:
        return max(1, int(self.runtime.max_loaded_models or 1))

    def _active(self, exclude: set[str] | frozenset[str] = frozenset()) -> list[_Instance]:
        return [i for i in self._instances.values() if i.state in _ACTIVE_STATES and i.id not in exclude]

    def _instances_for(self, key: str, embedding: bool) -> list[_Instance]:
        return [i for i in self._instances.values()
                if i.model.key == key and i.embedding == embedding and i.state in _ACTIVE_STATES]

    @staticmethod
    def _is_idle(inst: _Instance) -> bool:
        return inst.state == "ready" and inst.busy == 0

    def _is_pinned(self, inst: _Instance) -> bool:
        """True for a model the user asked to keep loaded (``runtime.pinned_models``).

        A pinned instance is skipped by the idle reaper and is never chosen as an
        eviction victim — the point of pinning is that it answers without a
        thirty-second load, which an eviction would undo.
        """
        pinned = getattr(self.runtime, "pinned_models", None) or ()
        return inst.model.key in set(pinned)

    def _eviction_plan(self, new_bytes: int, *, exclude: set[str],
                       allow_evict: bool) -> tuple[list[_Instance], bool]:
        """LRU idle instances to stop so that one more instance fits; (victims, fits)."""
        active = self._active(exclude)
        budget = self._budget()
        count = len(active)
        used = sum(i.reserved_bytes for i in active)
        victims: list[_Instance] = []
        if allow_evict and self.runtime.evict_policy == "lru":
            for cand in sorted((i for i in active if self._is_idle(i) and not self._is_pinned(i)),
                               key=lambda i: i.last_used):
                if count + 1 <= self._max_loaded() and used + new_bytes <= budget:
                    break
                victims.append(cand)
                count -= 1
                used -= cand.reserved_bytes
        fits = count + 1 <= self._max_loaded() and used + new_bytes <= budget
        return victims, fits

    def _plan_admission(self, model: LocalModel, cfg: LoadConfig, *,
                        exclude: set[str] | None = None, allow_evict: bool = True) -> list[_Instance]:
        """Instances to evict before starting ``model`` with ``cfg``; raises if impossible."""
        exclude = exclude or set()
        new_bytes = memory.breakdown(model, cfg).total
        victims, fits = self._eviction_plan(new_bytes, exclude=exclude, allow_evict=allow_evict)
        if not fits:
            raise EngineError(self._refusal_message(model, cfg, new_bytes, exclude, allow_evict))
        return victims

    def _refusal_message(self, model: LocalModel, cfg: LoadConfig, new_bytes: int,
                         exclude: set[str], allow_evict: bool) -> str:
        active = self._active(exclude)
        budget = self._budget()
        can_evict = allow_evict and self.runtime.evict_policy == "lru"
        pinned = [i for i in active if not (can_evict and self._is_idle(i))]
        idle = [i for i in active if self._is_idle(i)]
        name = model.display_name
        parts = memory.breakdown(model, cfg)
        detail = (f"wagi {memory.format_gb(parts.weights)} + kontekst "
                  f"{memory.format_gb(parts.kv_cache)} + bufory {memory.format_gb(parts.overhead)}")

        def names(items: list[_Instance]) -> str:
            return ", ".join(f"{i.model.display_name} ({memory.format_gb(i.reserved_bytes)})" for i in items)

        if len(pinned) + 1 > self._max_loaded():
            text = (f"Nie można załadować modelu {name}: osiągnięto limit załadowanych modeli "
                    f"({self._max_loaded()}).")
            if can_evict:
                text += (f" Wszystkie załadowane modele są teraz zajęte: {names(pinned)}. "
                         "Poczekaj na zakończenie odpowiedzi, wyładuj któryś model")
            else:
                text += (" Automatyczne wyładowywanie jest wyłączone – wyładuj ręcznie któryś "
                         f"z modeli: {names(active)},")
            return text + " albo zwiększ „Maks. załadowanych modeli” w ustawieniach Runtime."

        room = max(0, budget - sum(i.reserved_bytes for i in pinned))
        if new_bytes > budget:
            text = (f"Model {name} potrzebuje ~{memory.format_gb(new_bytes)} ({detail}), "
                    f"a cały limit RAM to {memory.format_gb(budget)}.")
        else:
            text = (f"Za mało pamięci w limicie RAM na model {name}: potrzeba "
                    f"~{memory.format_gb(new_bytes)} ({detail}), a wolne jest "
                    f"{memory.format_gb(room)} z {memory.format_gb(budget)}.")
            if pinned:
                text += f" Zajęte przez: {names(pinned)}."
        hints: list[str] = []
        ctx_fit = memory.max_ctx_that_fits(
            model, room, cfg.n_parallel, cfg.kv_cache_type, flash_attn=cfg.flash_attn,
            embedding=cfg.embedding, extra_args=cfg.extra_args)
        if 0 < ctx_fit < cfg.n_ctx:
            hints.append(f"zmniejsz kontekst (np. do {memory.format_tokens(ctx_fit)} tokenów)")
        if cfg.n_parallel > 1:
            hints.append("zmniejsz liczbę równoległych żądań")
        if pinned:
            hints.append(f"wyładuj {', '.join(i.model.display_name for i in pinned)}")
        elif idle and not can_evict:
            hints.append(f"wyładuj {', '.join(i.model.display_name for i in idle)}")
        hints.append("podnieś limit RAM w ustawieniach Runtime")
        return f"{text} Co możesz zrobić: {', '.join(hints)}."

    def _over_limits(self) -> bool:
        active = self._active()
        if len(active) > self._max_loaded():
            return True
        if sum(i.reserved_bytes for i in active) > self._budget():
            return True
        groups = collections.Counter((i.model.key, i.embedding) for i in active)
        return any(n > self._max_instances() for n in groups.values())

    def _enforce_limits(self) -> set[asyncio.Task[Any]]:
        """Unload idle instances (LRU) until every limit holds; busy ones are left alone."""
        tasks: set[asyncio.Task[Any]] = set()

        def stop(inst: _Instance) -> None:
            task = self._begin_stop(inst, grace=0.0, reason="limits")
            if task is not None:
                tasks.add(task)

        groups: dict[tuple[str, bool], list[_Instance]] = collections.defaultdict(list)
        for inst in self._active():
            groups[(inst.model.key, inst.embedding)].append(inst)
        for group in groups.values():
            excess = len(group) - self._max_instances()
            for inst in sorted((i for i in group if self._is_idle(i)), key=lambda i: i.last_used):
                if excess <= 0:
                    break
                stop(inst)
                excess -= 1
        budget = self._budget()
        for inst in sorted((i for i in self._active() if self._is_idle(i)), key=lambda i: i.last_used):
            active = self._active()
            if len(active) <= self._max_loaded() and sum(i.reserved_bytes for i in active) <= budget:
                break
            stop(inst)
        return tasks

    # ------------------------------------------------------------- configs

    def _resolve_model(self, model_key: str) -> LocalModel:
        if not model_key:
            raise EngineError("Nie wybrano modelu.")
        try:
            model = self.library.get(model_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("library.get(%r) failed", model_key, exc_info=True)
            raise EngineError(f"Nie udało się odczytać modelu „{model_key}” z biblioteki: {exc}") from exc
        if model is None:
            raise EngineError(
                f"Nie znaleziono modelu „{model_key}” w bibliotece. Pobierz go lub odśwież listę modeli.")
        return model

    def _load_defaults(self, key: str) -> dict[str, Any] | None:
        getter = getattr(self.library, "load_defaults", None)
        if getter is None:
            return None
        try:
            value = getter(key)
        except Exception:  # noqa: BLE001
            log.warning("library.load_defaults(%r) failed", key, exc_info=True)
            return None
        return value if isinstance(value, dict) else None

    def _build_config(self, model: LocalModel, *, n_ctx: int | None, embedding: bool,
                      explicit: LoadConfig | None = None,
                      exclude: set[str] | None = None) -> LoadConfig:
        """explicit LoadConfig > library load defaults > runtime settings; n_ctx None = auto."""
        rt = self.runtime
        fmt = model.format
        if explicit is not None:
            cfg = _copy_config(explicit)
            cfg.embedding = cfg.embedding or embedding
            cfg.n_parallel = max(1, int(cfg.n_parallel or 1))
            cfg.kv_cache_type = memory.normalize_kv_type(cfg.kv_cache_type, fmt)
            if not cfg.n_ctx or cfg.n_ctx <= 0:
                cfg.n_ctx = self._auto_ctx(model, cfg, exclude or set())
            return cfg
        base = LoadConfig(
            n_ctx=0,
            n_parallel=max(1, int(rt.default_parallel or 1)),
            gpu_layers=int(rt.gpu_layers),
            flash_attn=str(rt.flash_attn or "auto"),
            kv_cache_type=str(rt.kv_cache_type or "f16"),
            embedding=embedding,
        )
        cfg = base
        configured_ctx: int | None = None
        defaults = self._load_defaults(model.key)
        if defaults:
            clean = {k: v for k, v in defaults.items() if v is not None}
            try:
                cfg = LoadConfig.from_dict(clean, base)
            except (TypeError, ValueError):
                log.warning("invalid load defaults for %s: %r", model.key, defaults)
                cfg = base
            try:
                configured_ctx = int(clean["nCtx"]) if clean.get("nCtx") else None
            except (TypeError, ValueError):
                configured_ctx = None
        cfg = _copy_config(cfg)
        cfg.embedding = cfg.embedding or embedding
        cfg.n_parallel = max(1, min(64, int(cfg.n_parallel or 1)))
        cfg.kv_cache_type = memory.normalize_kv_type(cfg.kv_cache_type, fmt)
        requested = int(n_ctx) if n_ctx else 0
        native = _native_ctx(model)
        if requested and native:
            requested = min(requested, native)
        wanted = max(requested, configured_ctx or 0)
        cfg.n_ctx = wanted if wanted > 0 else self._auto_ctx(model, cfg, exclude or set())
        return cfg

    def _auto_ctx(self, model: LocalModel, cfg: LoadConfig, exclude: set[str]) -> int:
        """Automatic context: keep other models loaded when a comfortable window still fits."""
        budget = self._budget()
        active = self._active(exclude)
        kwargs = dict(n_parallel=cfg.n_parallel, kv_cache_type=cfg.kv_cache_type,
                      flash_attn=cfg.flash_attn, embedding=cfg.embedding, extra_args=cfg.extra_args)
        keep_all = budget - sum(i.reserved_bytes for i in active)
        ctx_keep = memory.auto_ctx(model, self.runtime, keep_all, **kwargs)
        want = min(memory.native_context(model), int(self.runtime.default_ctx or memory.DEFAULT_CTX_CAP))
        if ctx_keep >= min(want, AUTO_CTX_COMFORT) or self.runtime.evict_policy != "lru":
            return ctx_keep
        pinned = budget - sum(i.reserved_bytes for i in active if not self._is_idle(i))
        return max(ctx_keep, memory.auto_ctx(model, self.runtime, pinned, **kwargs))

    # ------------------------------------------------------------- engines

    def _default_engine_factory(self, instance_id: str, model: LocalModel, config: LoadConfig) -> Engine:
        """Import the engine for the model format lazily (modules are optional at import time)."""
        try:
            if model.format == "gguf":
                from localcode.runtime.llama_engine import LlamaServerEngine as engine_cls
            elif model.format == "mlx":
                from localcode.runtime.mlx_engine import MlxEngine as engine_cls  # type: ignore[no-redef]
            else:
                raise EngineError(f"Nieobsługiwany format modelu: {model.format}.")
        except ImportError as exc:
            raise EngineError(
                f"Silnik dla formatu {model.format.upper()} jest niedostępny: {exc}") from exc
        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(engine_cls).parameters
        except (TypeError, ValueError):
            params = {}  # type: ignore[assignment]
        if "settings" in params:
            kwargs["settings"] = self.settings
        return engine_cls(instance_id, model, config, **kwargs)

    def _new_id(self, model: LocalModel) -> str:
        tail = model.key.rstrip("/").rsplit("/", 1)[-1] or model.display_name or "model"
        slug = re.sub(r"[^a-z0-9._-]+", "-", tail.lower()).strip("-.")[:40] or "model"
        return f"{slug}-{next(self._ids)}"

    async def _measure(self, inst: _Instance) -> None:
        engine = inst.engine
        if engine is None or inst.state not in ("ready", "stopping"):
            return
        value: int | None = None
        try:
            value = await asyncio.wait_for(engine.memory_bytes(), MEASURE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("memory_bytes() failed for %s", inst.id, exc_info=True)
        if not value:
            try:
                value = memory.process_memory_bytes(engine.pid)
            except Exception:  # noqa: BLE001
                value = None
        if value and value > 0:
            old = inst.measured_bytes
            if old is None or abs(value - old) > max(32 * memory.MiB, 0.03 * old):
                inst.measured_bytes = int(value)
                self._notify()

    # --------------------------------------------------------------- reaper

    def _ensure_reaper(self) -> None:
        if self._closed or (self._reaper is not None and not self._reaper.done()):
            return
        self._reaper = asyncio.get_running_loop().create_task(self._reaper_loop())

    async def _reaper_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(max(0.01, float(REAPER_INTERVAL_S)))
            try:
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("runtime reaper tick failed")

    async def _reap_once(self) -> None:
        ttl_s = float(self.runtime.idle_ttl_minutes or 0) * 60.0
        now = time.monotonic()
        tasks: set[asyncio.Task[Any]] = set()
        async with self._lock:
            if ttl_s > 0:
                for inst in list(self._instances.values()):
                    if self._is_pinned(inst):
                        continue        # pinned: kept loaded on purpose
                    if self._is_idle(inst) and now - inst.last_used >= ttl_s:
                        task = self._begin_stop(inst, grace=0.0, reason="idle ttl")
                        if task is not None:
                            tasks.add(task)
        for inst in [i for i in self._instances.values() if i.state == "ready"]:
            await self._measure(inst)
        async with self._lock:
            if self._over_limits():
                tasks |= self._enforce_limits()
        if self._purge_errors():
            self._notify()
        if tasks:
            await asyncio.wait(tasks)

    def _purge_errors(self) -> bool:
        now = time.monotonic()
        keep = [e for e in self._errors
                if not e.shown or e.failed_at is None or now - e.failed_at < ERROR_TTL_S]
        changed = len(keep) != len(self._errors)
        self._errors = keep
        return changed

    # --------------------------------------------------------------- events

    def _notify(self) -> None:
        if self.on_event is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._emit()
            return
        if not self._notify_pending:
            self._notify_pending = True
            loop.call_soon(self._emit)

    def _emit(self) -> None:
        self._notify_pending = False
        callback = self.on_event
        if callback is None:
            return
        try:
            result = callback({"type": "runtime", **self.status()})
            if inspect.isawaitable(result):
                self._spawn(result)
        except Exception:  # noqa: BLE001
            log.exception("runtime on_event callback failed")

    def _spawn(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(awaitable)
        self._bg_tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._bg_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("runtime background task failed", exc_info=task.exception())

    # ---------------------------------------------------------------- misc

    def _check_open(self) -> None:
        if self._closed:
            raise EngineError(_CLOSED_MESSAGE)

    def _instance_dto(self, inst: _Instance) -> InstanceInfo:
        engine = inst.engine
        pid = port = None
        if engine is not None and inst.state != "error":
            try:
                pid, port = engine.pid, engine.port
            except Exception:  # noqa: BLE001
                pid = port = None
        return {
            "id": inst.id,
            "modelKey": inst.model.key,
            "displayName": inst.model.display_name,
            "format": inst.model.format,
            "state": inst.display_state,
            "nCtx": inst.n_ctx,
            "nParallel": inst.slots,
            "busySlots": inst.busy,
            "memoryBytes": inst.measured_bytes or inst.estimated_bytes,
            "estimatedBytes": inst.estimated_bytes,
            "loadedAt": int(inst.loaded_at * 1000) if inst.loaded_at else None,
            "lastUsedAt": int(inst.last_used_at * 1000),
            "pid": pid,
            "port": port,
            "genTps": round(inst.gen_tps, 1) if inst.gen_tps is not None else None,
            "error": inst.error,
            "sandboxed": bool(getattr(engine, "sandboxed", False)) if engine is not None else False,
            "isolation": sandbox.describe(
                bool(getattr(engine, "sandboxed", False)) if engine is not None else False,
                self.settings),
        }


def _native_ctx(model: LocalModel) -> int | None:
    """Trained context length when the metadata knows it."""
    ctx = model.meta.context_length if model.meta is not None else None
    return int(ctx) if ctx and ctx > 0 else None


def _copy_config(cfg: LoadConfig) -> LoadConfig:
    return LoadConfig(
        n_ctx=int(cfg.n_ctx or 0),
        n_parallel=int(cfg.n_parallel or 1),
        gpu_layers=int(cfg.gpu_layers),
        flash_attn=str(cfg.flash_attn),
        kv_cache_type=str(cfg.kv_cache_type),
        embedding=bool(cfg.embedding),
        extra_args=list(cfg.extra_args or []),
        allow_dangerous=bool(cfg.allow_dangerous),
    )


def _same_config(a: LoadConfig, b: LoadConfig) -> bool:
    return (a.n_ctx == b.n_ctx and a.n_parallel == b.n_parallel and a.gpu_layers == b.gpu_layers
            and a.flash_attn == b.flash_attn and a.kv_cache_type == b.kv_cache_type
            and a.embedding == b.embedding and list(a.extra_args) == list(b.extra_args))


__all__ = [
    "EngineFactory",
    "InstanceInfo",
    "Lease",
    "ModelLibraryLike",
    "PendingLease",
    "RuntimeManager",
    "SettingsError",
    "validate_settings_patch",
]
