"""ImageRuntime: loaded image models, the shared RAM budget and generation jobs.

The image runtime is the ``RuntimeManager`` of the image side, deliberately much
smaller because diffusion models have no context window and no request slots:

* at most ``settings.images.max_loaded`` models are kept in memory (default 1);
  loading another one evicts the least recently used *idle* image instance;
* generation is **serialised per instance** — one image job at a time — and
  cancellable (:meth:`ImageRuntime.cancel`);
* before loading, the estimated memory (``images/memory.py``) is checked against
  the shared budget and :meth:`RuntimeManager.make_room` is asked to evict idle
  *language* instances, so both runtimes stay inside one RAM limit;
* every state change emits ``{"type": "runtime", …}`` with the image instances
  merged into the language runtime's status (that is what ``/api/runtime``
  serves), and every denoising step emits ``{"type": "image_progress", …}``;
* idle instances are unloaded after ``settings.images.idle_ttl_minutes``.

Generated PNGs land in ``settings.images.output_dir`` under ``<yyyy-mm>/`` (the
layout ``ImageGallery`` indexes); the runtime only writes files and returns
:class:`~localcode.images.base.ImageResult` objects.

All methods must be called from the event loop that owns the runtime.
"""

from __future__ import annotations

from collections import deque

import asyncio
import contextlib
import inspect
import itertools
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from localcode.config import Settings, home
from localcode.hub.types import LocalModel, MemoryEstimate
from localcode.images import memory as image_memory
from localcode.images.base import (
    ImageEngine,
    ImageEngineError,
    ImageGenerationCancelled,
    ImageParams,
    ImageProgress,
    ImageResult,
    default_steps_for,
    family_defaults,
    new_image_id,
)
from localcode.runtime import memory as rt_memory

log = logging.getLogger(__name__)

__all__ = ["ImageRuntime", "ImageRuntimeError", "is_image_model"]

REAPER_INTERVAL_S = 15.0
MEASURE_TIMEOUT_S = 5.0
ERROR_TTL_S = 30.0
STOP_GRACE_S = 5.0

EngineFactory = Callable[[str, LocalModel], ImageEngine]

_CLOSED_MESSAGE = "Generowanie obrazów zostało zamknięte."


class ImageRuntimeError(ImageEngineError):
    """Runtime-level failure (unknown model, no room, closed runtime)."""

    code = "image_runtime_error"


def is_image_model(model: LocalModel | None) -> bool:
    """True for a model the image runtime can load."""
    if model is None:
        return False
    if "image" in (model.capabilities or []):
        return True
    family = getattr(model.meta, "image_family", None)
    return bool(family)


# --------------------------------------------------------------------------- #
# Instances and jobs
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class _Job:
    id: str
    model_key: str
    params: ImageParams
    started_at: float = field(default_factory=time.time)
    index: int = 0
    step: int = 0
    steps: int = 0
    count: int = 1
    cancelled: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modelKey": self.model_key,
            "prompt": self.params.prompt,
            "index": self.index,
            "count": self.count,
            "step": self.step,
            "steps": self.steps,
            "startedAt": int(self.started_at * 1000),
            "cancelled": self.cancelled,
            "message": self.message,
        }


@dataclass(eq=False)
class _Instance:
    id: str
    model: LocalModel
    engine: ImageEngine | None = None
    state: str = "loading"                   # loading | ready | busy | stopping | error
    estimated_bytes: int = 0
    measured_bytes: int | None = None
    loaded_at: float | None = None
    last_used: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.time)
    error: str | None = None
    failed_at: float | None = None
    shown: bool = False
    job: _Job | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def reserved_bytes(self) -> int:
        return max(self.estimated_bytes, self.measured_bytes or 0)

    @property
    def display_state(self) -> str:
        if self.state == "ready" and self.job is not None:
            return "busy"
        return self.state

    def touch(self) -> None:
        self.last_used = time.monotonic()
        self.last_used_at = time.time()


# --------------------------------------------------------------------------- #
# ImageRuntime
# --------------------------------------------------------------------------- #


class ImageRuntime:
    """Owns every loaded image model (see the module docstring)."""

    def __init__(
        self,
        settings: Settings,
        library: Any,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        runtime: Any | None = None,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.settings = settings
        self.library = library
        self.on_event = on_event
        self.manager = runtime                      # RuntimeManager | None
        self._engine_factory = engine_factory or self._default_engine_factory
        self._instances: dict[str, _Instance] = {}
        #: ids this runtime retired itself — unloading one again is a no-op
        #: (the UI may click twice or act on a stale status snapshot)
        self._gone: deque[str] = deque(maxlen=32)
        self._errors: list[_Instance] = []
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)
        self._reaper: asyncio.Task[None] | None = None
        self._bg: set[asyncio.Task[Any]] = set()
        self._closed = False

    # ------------------------------------------------------------------ config
    @property
    def config(self) -> Any:
        """``settings.images`` (falls back to defaults when absent)."""
        images = getattr(self.settings, "images", None)
        if images is None:
            from localcode.config import ImageSettings

            images = ImageSettings()
            try:
                self.settings.images = images  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - settings may be frozen in tests
                pass
        return images

    def _max_loaded(self) -> int:
        return max(1, int(getattr(self.config, "max_loaded", 1) or 1))

    def output_dir(self) -> Path:
        """Root directory for generated images (``settings.images.output_dir``)."""
        raw = str(getattr(self.config, "output_dir", "") or "data/images")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = home() / path
        return path

    def _default_size(self) -> tuple[int, int]:
        raw = str(getattr(self.config, "default_size", "") or "1024x1024")
        m = re.match(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", raw)
        if not m:
            return 1024, 1024
        return int(m.group(1)), int(m.group(2))

    # ------------------------------------------------------------------ public
    def models(self) -> list[LocalModel]:
        """Image-capable models in the library."""
        try:
            models = self.library.list()
        except Exception:  # noqa: BLE001 - a broken library must not break the UI
            log.warning("cannot list the model library", exc_info=True)
            return []
        return [m for m in models if is_image_model(m)]

    def model_info(self, model: LocalModel) -> dict[str, Any]:
        """``LocalModel`` DTO enriched with the image fields the app needs."""
        family = str(getattr(model.meta, "image_family", None) or "unknown")
        defaults = family_defaults(family)
        steps = self._model_steps(model, family)
        dto = model.to_dict()
        dto.update({
            "imageFamily": family,
            "loaded": self.is_loaded(model.key),
            "defaults": {
                "steps": steps,
                "guidance": defaults["guidance"],
                "sizes": list(defaults["sizes"]),
            },
            "estimatedBytes": image_memory.estimate_bytes(model),
        })
        return dto

    def is_loaded(self, model_key: str) -> bool:
        return any(i.model.key == model_key and i.state in ("loading", "ready")
                   for i in self._instances.values())

    def estimate(self, model_key: str, *, width: int | None = None,
                 height: int | None = None) -> MemoryEstimate:
        """Memory estimate for loading ``model_key`` right now."""
        model = self._resolve(model_key)
        w, h = self._default_size()
        budget = self._budget()
        in_use, reclaimable = self._usage()
        return image_memory.estimate(
            model, width=int(width or w), height=int(height or h),
            budget=budget, in_use=in_use, reclaimable=reclaimable,
        )

    async def load(self, model_key: str) -> dict[str, Any]:
        """Explicitly load an image model (UI "Załaduj"); returns its DTO."""
        self._check_open()
        model = self._resolve(model_key)
        inst = await self._get_instance(model)
        return self._instance_dto(inst)


    async def _translate_prompt(self, params: ImageParams) -> ImageParams:
        """Translate a non-English prompt with the loaded chat model (best effort)."""
        if self.manager is None:
            return params
        try:
            from localcode.images.prompt import prepare_prompt
        except Exception:                       # pragma: no cover - import guard
            return params
        text, translated = await prepare_prompt(params.prompt, manager=self.manager,
                                                translate=params.translate)
        if not translated or not text:
            return params
        log.info("image prompt translated: %r → %r", params.prompt[:60], text[:60])
        return replace(params, prompt=text, original_prompt=params.prompt)

    async def generate(
        self,
        model_key: str,
        params: ImageParams,
        *,
        on_progress: Callable[[ImageProgress], Any] | None = None,
        job_id: str | None = None,
    ) -> list[ImageResult]:
        """Generate ``params.count`` images with ``model_key``.

        Loads the model when needed (evicting idle models to fit the RAM
        budget), serialises jobs per instance, reports progress through
        ``on_progress`` and ``image_progress`` events and returns the finished
        images.  Raises :class:`ImageGenerationCancelled` when cancelled and
        :class:`ImageEngineError` (Polish message) on any other failure.
        """
        self._check_open()
        model = self._resolve(model_key)
        params = await self._translate_prompt(params)
        prepared = self._prepare_params(model, params)
        inst = await self._get_instance(model, width=prepared.width, height=prepared.height)
        job = _Job(id=job_id or new_image_id(), model_key=model.key, params=prepared,
                   count=prepared.count, steps=prepared.steps or 0)
        async with inst.lock:
            if inst.state not in ("ready",) or inst.engine is None:
                raise ImageRuntimeError(
                    f"Model obrazów „{model.display_name}” nie jest gotowy.")
            inst.job = job
            inst.touch()
            self._notify()
            try:
                return await self._run_job(inst, job, on_progress)
            finally:
                inst.job = None
                inst.touch()
                self._notify()

    async def cancel(self, job_id: str | None = None, *, instance_id: str | None = None) -> bool:
        """Cancel the running job (optionally only ``job_id``/``instance_id``).

        Returns True when something was cancelled.  Backends that cannot abort a
        running generation (stable-diffusion.cpp) are stopped instead, so the
        model is reloaded on the next request.
        """
        cancelled = False
        for inst in list(self._instances.values()):
            job = inst.job
            if job is None:
                continue
            if job_id is not None and job.id != job_id:
                continue
            if instance_id is not None and inst.id != instance_id:
                continue
            job.cancelled = True
            cancelled = True
            engine = inst.engine
            if engine is not None:
                try:
                    soft = await engine.cancel()
                except ImageEngineError:
                    soft = False
                if not soft:
                    # The process had to be killed: drop the instance.
                    await self._forget(inst, reason="cancel")
        if cancelled:
            self._notify()
        return cancelled

    def status(self) -> list[dict[str, Any]]:
        """Image instance DTOs (``kind: "image"``) for ``/api/runtime``."""
        self._purge_errors()
        out = [self._instance_dto(i) for i in self._instances.values()]
        for err in self._errors:
            out.append(self._instance_dto(err))
            err.shown = True
        return out

    def runtime_status(self) -> dict[str, Any]:
        """The full ``RuntimeStatus`` DTO with image instances merged in.

        With a :class:`RuntimeManager` this is its ``status()`` plus the image
        instances; standalone it is a complete status of its own, so the server
        can serve ``/api/runtime`` either way.
        """
        payload: dict[str, Any] = {}
        if self.manager is not None:
            try:
                payload = dict(self.manager.status())
            except Exception:  # noqa: BLE001 - never let the language side break this
                log.debug("RuntimeManager.status() failed", exc_info=True)
                payload = {}
        images = self.status()
        payload["instances"] = list(payload.get("instances") or []) + images
        if "system" not in payload:
            sysmem = rt_memory.system_memory()
            payload["system"] = {
                "totalBytes": sysmem["total"],
                "usedBytes": sysmem["used"],
                "availableBytes": sysmem["available"],
            }
        budget = payload.get("budget") if isinstance(payload.get("budget"), dict) else None
        used_images = sum(i.reserved_bytes for i in self._instances.values())
        if budget is None:
            payload["budget"] = {"limitBytes": self._budget(), "usedBytes": used_images}
        else:
            budget["usedBytes"] = int(budget.get("usedBytes") or 0) + used_images
        payload["images"] = {
            "instances": images,
            "settings": {
                "maxLoaded": self._max_loaded(),
                "idleTtlMinutes": int(getattr(self.config, "idle_ttl_minutes", 10) or 0),
                "defaultModel": getattr(self.config, "default_model", None),
                "defaultSize": getattr(self.config, "default_size", "1024x1024"),
                "defaultSteps": getattr(self.config, "default_steps", None),
                "defaultGuidance": getattr(self.config, "default_guidance", None),
                "outputDir": str(self.output_dir()),
                "keepMetadata": bool(getattr(self.config, "keep_metadata", True)),
            },
        }
        return payload

    async def unload(self, instance_id: str) -> None:
        """Unload one image instance (waits for a running job to be cancelled)."""
        inst = self._instances.get(instance_id)
        if inst is None:
            before = len(self._errors)
            self._errors = [e for e in self._errors if e.id != instance_id]
            if len(self._errors) != before:
                self._notify()
                return
            if instance_id in self._gone:
                return                      # already unloaded: the intent is fulfilled
            raise ImageRuntimeError(f"Nie ma załadowanego modelu obrazów „{instance_id}”.")
        if inst.job is not None:
            await self.cancel(instance_id=instance_id)
        await self._forget(inst, reason="unload")
        self._notify()

    async def unload_all(self) -> None:
        for inst in list(self._instances.values()):
            with contextlib.suppress(ImageEngineError):
                await self.unload(inst.id)

    async def shutdown(self) -> None:
        """Stop the reaper and every worker. Idempotent."""
        self._closed = True
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for inst in list(self._instances.values()):
            await self._forget(inst, reason="shutdown")
        self._instances.clear()
        self._errors.clear()
        tasks = {t for t in self._bg if not t.done()}
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.wait(tasks)
        self._notify()

    # ------------------------------------------------------------- generation
    async def _run_job(self, inst: _Instance, job: _Job,
                       on_progress: Callable[[ImageProgress], Any] | None) -> list[ImageResult]:
        engine = inst.engine
        assert engine is not None
        params = job.params
        ids = [new_image_id() for _ in range(max(1, params.count))]
        out_dir = self.output_dir() / datetime.now().strftime("%Y-%m")
        results: list[ImageResult] = []
        self._emit({"type": "image_progress", **self._progress_event(inst, job,
                                                                    ImageProgress(count=job.count,
                                                                                  message="Generowanie…"))})
        stream = engine.generate(params, output_dir=out_dir, ids=ids)
        try:
            async with contextlib.aclosing(stream):  # type: ignore[arg-type]
                async for item in stream:
                    if isinstance(item, ImageProgress):
                        job.index, job.step, job.steps = item.index, item.step, item.steps
                        job.message = item.message
                        self._emit({"type": "image_progress", **self._progress_event(inst, job, item)})
                        await _call(on_progress, item)
                    elif isinstance(item, ImageResult):
                        item.model_key = inst.model.key
                        item.model_name = inst.model.display_name
                        results.append(item)
                        inst.touch()
        except ImageGenerationCancelled:
            raise
        except ImageEngineError as exc:
            if job.cancelled:
                raise ImageGenerationCancelled() from exc
            await self._mark_error(inst, str(exc))
            raise
        if job.cancelled and not results:
            raise ImageGenerationCancelled()
        return results

    def _prepare_params(self, model: LocalModel, params: ImageParams) -> ImageParams:
        """Fill model/settings defaults into a request."""
        family = str(getattr(model.meta, "image_family", None) or "unknown")
        defaults = family_defaults(family)
        width, height = self._default_size()
        prepared = ImageParams(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            width=params.width or width,
            height=params.height or height,
            steps=params.steps or self._model_steps(model, family),
            guidance=params.guidance if params.guidance is not None
            else _first_not_none(getattr(self.config, "default_guidance", None), defaults["guidance"]),
            seed=params.seed,
            count=params.count,
            init_image=params.init_image,
            strength=params.strength,
            translate=params.translate,
            original_prompt=params.original_prompt,
        )
        multiple = 64 if model.format == "gguf" else 16
        return prepared.normalized(multiple=multiple)

    def _model_steps(self, model: LocalModel, family: str) -> int:
        configured = getattr(self.config, "default_steps", None)
        if configured:
            return int(configured)
        defaults = self._load_defaults(model.key)
        if isinstance(defaults, dict) and defaults.get("steps"):
            with contextlib.suppress(TypeError, ValueError):
                return int(defaults["steps"])
        return default_steps_for(family, model.key)

    # ---------------------------------------------------------------- loading
    async def _get_instance(self, model: LocalModel, *, width: int | None = None,
                            height: int | None = None) -> _Instance:
        """A ready instance of ``model``, loading (and evicting) as needed."""
        self._ensure_reaper()
        async with self._lock:
            existing = next((i for i in self._instances.values()
                             if i.model.key == model.key and i.state in ("loading", "ready")), None)
            if existing is not None:
                inst = existing
            else:
                w, h = self._default_size()
                estimated = image_memory.estimate_bytes(
                    model, width=int(width or w), height=int(height or h))
                await self._make_room(model, estimated)
                inst = _Instance(id=self._new_id(model), model=model, estimated_bytes=estimated)
                self._instances[inst.id] = inst
                self._notify()
                self._spawn(self._load_instance(inst))
        await inst.ready_event.wait()
        inst.touch()          # a caller is taking this instance: restart the idle TTL
        if inst.state == "error" or inst.engine is None:
            raise ImageRuntimeError(inst.error or
                                    f"Nie udało się wczytać modelu obrazów „{model.display_name}”.")
        if inst.id not in self._instances:
            raise ImageRuntimeError(
                f"Wczytywanie modelu „{model.display_name}” zostało przerwane.")
        return inst

    async def _load_instance(self, inst: _Instance) -> None:
        engine: ImageEngine | None = None
        try:
            engine = self._engine_factory(inst.id, inst.model)
            inst.engine = engine
            await engine.start()
            inst.state = "ready"
            inst.loaded_at = time.time()
            inst.touch()
            log.info("image model %s ready (instance %s)", inst.model.key, inst.id)
        except asyncio.CancelledError:
            inst.state = "error"
            inst.error = "Wczytywanie modelu obrazów zostało przerwane."
            with contextlib.suppress(Exception):
                if engine is not None:
                    await engine.stop()
            # Nobody will ever use this instance again: drop it from the table so
            # that it stops counting against max_loaded_models and the RAM budget.
            self._instances.pop(inst.id, None)
            inst.engine = None
            raise
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            inst.state = "error"
            inst.error = str(exc) or f"Nie udało się wczytać modelu „{inst.model.display_name}”."
            inst.failed_at = time.monotonic()
            with contextlib.suppress(Exception):
                if engine is not None:
                    await engine.stop()
            self._instances.pop(inst.id, None)
            self._errors.append(inst)
            log.warning("loading image model %s failed: %s", inst.model.key, exc)
        finally:
            inst.ready_event.set()
            self._notify()

    async def _make_room(self, model: LocalModel, needed: int) -> None:
        """Evict idle image instances (and ask the language runtime for room)."""
        while len(self._instances) >= self._max_loaded():
            victim = self._lru_idle()
            if victim is None:
                raise ImageRuntimeError(
                    f"Nie można wczytać modelu „{model.display_name}”: inny model obrazów jest "
                    "teraz używany. Poczekaj na zakończenie generowania albo go wyładuj.")
            await self._forget(victim, reason="limits")
        budget = self._budget()
        used_images = sum(i.reserved_bytes for i in self._instances.values())
        if self.manager is not None:
            ok = await self._manager_make_room(needed + used_images)
            if not ok:
                raise ImageRuntimeError(self._no_room_message(model, needed, budget))
            return
        if needed + used_images > budget:
            raise ImageRuntimeError(self._no_room_message(model, needed, budget))

    async def _manager_make_room(self, needed: int) -> bool:
        make_room = getattr(self.manager, "make_room", None)
        if make_room is None:
            log.debug("RuntimeManager has no make_room(); skipping language-side eviction")
            return True
        try:
            result = make_room(int(needed))
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except Exception:  # noqa: BLE001 - never fail a load because of the LLM side
            log.warning("RuntimeManager.make_room() failed", exc_info=True)
            return True

    def _no_room_message(self, model: LocalModel, needed: int, budget: int) -> str:
        return (
            f"Za mało pamięci na model obrazów „{model.display_name}”: potrzeba ok. "
            f"{image_memory.format_gb(needed)} z limitu {image_memory.format_gb(budget)}. "
            "Wyładuj inne modele, wybierz mniejszy model obrazów albo podnieś limit RAM "
            "w ustawieniach Runtime."
        )

    def _lru_idle(self) -> _Instance | None:
        idle = [i for i in self._instances.values() if i.job is None and i.state in ("ready", "error")]
        return min(idle, key=lambda i: i.last_used) if idle else None

    async def _forget(self, inst: _Instance, *, reason: str) -> None:
        """Stop an instance and drop it from the table."""
        if self._instances.get(inst.id) is inst:
            inst.state = "stopping"
            self._notify()
        engine, inst.engine = inst.engine, None
        if engine is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(engine.stop(), timeout=STOP_GRACE_S * 3)
        self._instances.pop(inst.id, None)
        self._gone.append(inst.id)
        inst.state = "stopping"
        inst.ready_event.set()
        log.info("image instance %s stopped (%s)", inst.id, reason)

    async def _mark_error(self, inst: _Instance, message: str) -> None:
        inst.error = message
        inst.failed_at = time.monotonic()
        if inst.engine is not None and not inst.engine.running:
            await self._forget(inst, reason="engine died")
            inst.state = "error"
            self._errors.append(inst)
        self._notify()

    # ----------------------------------------------------------------- budget
    def _budget(self) -> int:
        return rt_memory.default_ram_budget(self.settings)

    def _usage(self) -> tuple[int, int]:
        """(reserved by every instance, reclaimable by evicting idle ones)."""
        in_use = sum(i.reserved_bytes for i in self._instances.values())
        reclaimable = sum(i.reserved_bytes for i in self._instances.values() if i.job is None)
        if self.manager is not None:
            try:
                status = self.manager.status()
                budget = status.get("budget") if isinstance(status, dict) else None
                if isinstance(budget, dict):
                    in_use += int(budget.get("usedBytes") or 0)
                for dto in (status.get("instances") or []) if isinstance(status, dict) else []:
                    if isinstance(dto, dict) and dto.get("state") == "ready":
                        reclaimable += int(dto.get("memoryBytes") or 0)
            except Exception:  # noqa: BLE001
                log.debug("cannot read RuntimeManager status", exc_info=True)
        return in_use, reclaimable

    # ---------------------------------------------------------------- factory
    def _default_engine_factory(self, instance_id: str, model: LocalModel) -> ImageEngine:
        defaults = self._load_defaults(model.key) or {}
        python_path = getattr(getattr(self.settings, "runtime", None), "python_path", None)
        keep_metadata = bool(getattr(self.config, "keep_metadata", True))
        steps = defaults.get("steps") or getattr(self.config, "default_steps", None)
        if model.format == "mlx":
            from localcode.images.mflux_engine import MfluxEngine

            return MfluxEngine(
                instance_id, model, settings=self.settings, python_path=python_path,
                base_model=defaults.get("baseModel"), quantize=defaults.get("quantize"),
                default_steps=steps, embed_metadata=keep_metadata,
            )
        if model.format == "gguf":
            from localcode.images.sdcpp_engine import SdcppEngine

            components = dict(getattr(model.meta, "image_components", None) or {})
            components.update({k: v for k, v in (defaults.get("components") or {}).items() if v})
            return SdcppEngine(
                instance_id, model, settings=self.settings, python_path=python_path,
                components=components, default_steps=steps,
                threads=defaults.get("threads"), flash_attn=bool(defaults.get("flashAttn")),
                embed_metadata=keep_metadata,
            )
        raise ImageRuntimeError(
            f"Format „{model.format}” nie jest obsługiwany przez generator obrazów.")

    def _load_defaults(self, key: str) -> dict[str, Any] | None:
        getter = getattr(self.library, "load_defaults", None)
        if getter is None:
            return None
        try:
            value = getter(key)
        except Exception:  # noqa: BLE001
            return None
        return value if isinstance(value, dict) else None

    # ----------------------------------------------------------------- reaper
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
                log.exception("image runtime reaper tick failed")

    async def _reap_once(self) -> None:
        ttl_s = float(getattr(self.config, "idle_ttl_minutes", 10) or 0) * 60.0
        now = time.monotonic()
        if ttl_s > 0:
            for inst in list(self._instances.values()):
                if inst.job is None and inst.state == "ready" and now - inst.last_used >= ttl_s:
                    log.info("unloading idle image model %s", inst.model.key)
                    await self._forget(inst, reason="idle ttl")
                    self._notify()
        for inst in list(self._instances.values()):
            if inst.state == "ready":
                await self._measure(inst)
        if self._purge_errors():
            self._notify()

    async def _measure(self, inst: _Instance) -> None:
        engine = inst.engine
        if engine is None:
            return
        value: int | None = None
        try:
            value = await asyncio.wait_for(engine.memory_bytes(), MEASURE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.debug("memory_bytes() failed for %s", inst.id, exc_info=True)
        if value and value > 0:
            old = inst.measured_bytes
            if old is None or abs(value - old) > max(32 * image_memory.MiB, 0.03 * old):
                inst.measured_bytes = int(value)
                self._notify()

    def _purge_errors(self) -> bool:
        now = time.monotonic()
        keep = [e for e in self._errors
                if not e.shown or e.failed_at is None or now - e.failed_at < ERROR_TTL_S]
        changed = len(keep) != len(self._errors)
        self._errors = keep
        return changed

    # ----------------------------------------------------------------- events
    def _instance_dto(self, inst: _Instance) -> dict[str, Any]:
        engine = inst.engine
        pid = port = None
        if engine is not None and inst.state != "error":
            try:
                pid, port = engine.pid, engine.port
            except Exception:  # noqa: BLE001
                pid = port = None
        return {
            "id": inst.id,
            "kind": "image",
            "modelKey": inst.model.key,
            "displayName": inst.model.display_name,
            "format": inst.model.format,
            "backend": getattr(engine, "backend", None) or ("mflux" if inst.model.format == "mlx" else "sdcpp"),
            "imageFamily": str(getattr(inst.model.meta, "image_family", None) or "unknown"),
            "state": inst.display_state,
            "nCtx": 0,
            "nParallel": 1,
            "busySlots": 1 if inst.job is not None else 0,
            "memoryBytes": inst.measured_bytes or inst.estimated_bytes,
            "estimatedBytes": inst.estimated_bytes,
            "loadedAt": int(inst.loaded_at * 1000) if inst.loaded_at else None,
            "lastUsedAt": int(inst.last_used_at * 1000),
            "pid": pid,
            "port": port,
            "genTps": None,
            "error": inst.error,
            "job": inst.job.to_dict() if inst.job is not None else None,
        }

    def _progress_event(self, inst: _Instance, job: _Job, progress: ImageProgress) -> dict[str, Any]:
        return {
            "instanceId": inst.id,
            "modelKey": inst.model.key,
            "jobId": job.id,
            **progress.to_dict(),
        }

    def _notify(self) -> None:
        self._emit({"type": "runtime", **self.runtime_status()})

    def _emit(self, payload: dict[str, Any]) -> None:
        callback = self.on_event
        if callback is None:
            return
        try:
            result = callback(payload)
            if inspect.isawaitable(result):
                self._spawn(result)
        except Exception:  # noqa: BLE001
            log.exception("image runtime on_event callback failed")

    # ------------------------------------------------------------------ misc
    def _spawn(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(awaitable)
        self._bg.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._bg.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("image runtime background task failed", exc_info=task.exception())

    def _new_id(self, model: LocalModel) -> str:
        tail = model.key.rstrip("/").rsplit("/", 1)[-1] or model.display_name or "image"
        slug = re.sub(r"[^a-z0-9._-]+", "-", tail.lower()).strip("-.")[:40] or "image"
        return f"img-{slug}-{next(self._ids)}"

    def _resolve(self, model_key: str) -> LocalModel:
        if not model_key:
            default = getattr(self.config, "default_model", None)
            if not default:
                raise ImageRuntimeError("Nie wybrano modelu do generowania obrazów.")
            model_key = str(default)
        model = None
        with contextlib.suppress(Exception):
            model = self.library.get(model_key)
        if model is None:
            with contextlib.suppress(Exception):
                model = self.library.resolve(model_key)
        if model is None:
            raise ImageRuntimeError(f"Nie znaleziono modelu „{model_key}” w bibliotece.")
        if not is_image_model(model):
            raise ImageRuntimeError(
                f"Model „{model.display_name}” nie generuje obrazów. Wybierz model z etykietą „Obrazy”.")
        return model

    def _check_open(self) -> None:
        if self._closed:
            raise ImageRuntimeError(_CLOSED_MESSAGE)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _call(callback: Callable[[ImageProgress], Any] | None, progress: ImageProgress) -> None:
    if callback is None:
        return
    try:
        result = callback(progress)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - a bad UI callback must not kill a job
        log.debug("image progress callback failed", exc_info=True)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
