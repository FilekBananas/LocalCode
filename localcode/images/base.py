"""Contract of an image-generation engine plus the shared worker/HTTP plumbing.

An :class:`ImageEngine` is *one loaded image model* living in its own worker
subprocess.  The worker serves a tiny HTTP surface on 127.0.0.1 that mirrors the
shape of the language engines (``runtime/http_engine.py``), so the runtime can
drive mflux and stable-diffusion.cpp through one client:

=====================  ======================================================
``GET  /health``       ``503 {"error": {...}}`` while the model loads, then
                       ``200 {"status": "ok"}``.
``GET  /props``        backend, model path, image family, generation defaults
                       (steps/guidance/size) and what the model supports.
``POST /generate``     SSE.  Body is :meth:`ImageParams.to_payload` plus
                       ``outputs`` (absolute PNG paths, one per image).
                       Frames::

                           {"type":"progress","index":0,"step":3,"steps":4}
                           {"type":"image", ...ImageResult fields...}
                           {"type":"done"}
                           {"type":"error","message":"…","cancelled":false}

``POST /cancel``       ``{"ok": true, "hard": false}``.  ``hard`` means the
                       backend cannot stop mid-generation, so the caller must
                       stop the process to make the cancellation effective
                       (stable-diffusion.cpp has no abort callback).
``GET  /memory``       ``{"bytes": N, "peakBytes": N}`` — what the backend's
                       allocator reports, used for the RAM budget.
=====================  ======================================================

Process handling (own process group, log file, SIGTERM→SIGKILL, emergency kill
at interpreter exit) is *shared* with the language engines: this module imports
the helpers from ``runtime/http_engine.py`` instead of duplicating them, so a
crashing core never leaves multi-gigabyte image workers behind.  That module is
not modified by this one.

All user-facing messages are Polish; code and comments are English.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import math
import os
import random
import shlex
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

from localcode.config import install_command, logs_dir
from localcode.hub.types import LocalModel
from localcode.security import sandbox
from localcode.security.sandbox import SandboxPlan
from localcode.runtime.http_engine import (  # shared process plumbing (not modified here)
    _ANSI_RE,
    SSEDecoder,
    _free_port,
    _register_group,
    _safe_name,
    _terminate_process_group,
    _unregister_group,
)

log = logging.getLogger(__name__)

__all__ = [
    "IMAGE_FAMILIES",
    "ImageEngine",
    "ImageEngineError",
    "ImageEngineStartError",
    "ImageEngineUnavailableError",
    "ImageGenerationCancelled",
    "ImageParams",
    "ImageProgress",
    "ImageResult",
    "WorkerImageEngine",
    "backend_missing_message",
    "default_steps_for",
    "family_defaults",
    "new_image_id",
]

#: Image families understood across the runtime (``ModelMeta.image_family``).
IMAGE_FAMILIES = ("flux", "flux2", "qwen-image", "sd1", "sdxl", "sd3", "unknown")

MAX_DIMENSION = 2048
MIN_DIMENSION = 128
MAX_STEPS = 150
MAX_COUNT = 8
SEED_MAX = 2**31 - 1


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ImageEngineError(RuntimeError):
    """User-presentable image-engine failure (Polish message, safe to show)."""

    code = "image_engine_error"


class ImageEngineStartError(ImageEngineError):
    """The worker process could not be started; ``log_tail`` holds the last log lines."""

    code = "image_engine_start_failed"

    def __init__(self, message: str, *, log_tail: str = "", exit_code: int | None = None) -> None:
        super().__init__(message)
        self.log_tail = log_tail
        self.exit_code = exit_code


class ImageEngineUnavailableError(ImageEngineError):
    """The worker is not running / crashed / cannot be reached."""

    code = "image_engine_unavailable"


class ImageGenerationCancelled(ImageEngineError):
    """Generation was cancelled by the user."""

    code = "image_cancelled"

    def __init__(self, message: str = "Generowanie obrazu zostało przerwane.") -> None:
        super().__init__(message)


def backend_missing_message(backend: str, extra: str = "images") -> str:
    """Polish explanation shown when the optional image dependency is missing."""
    packages = {
        "mflux": "mflux",
        "sdcpp": "stable-diffusion-cpp-python",
    }
    package = packages.get(backend, backend)
    return (
        f"Generowanie obrazów wymaga pakietu „{package}”, którego nie ma w tym środowisku. "
        f"Zainstaluj dodatek: {install_command(extra)}"
    )


# --------------------------------------------------------------------------- #
# Parameters / progress / results
# --------------------------------------------------------------------------- #


def new_image_id() -> str:
    """Short, sortable, unique id of one generated image."""
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:8]}"


@dataclass
class ImageParams:
    """One generation request (see ``docs/IMAGES.md``)."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None            # None = model default (FLUX schnell 4, SD 20…)
    guidance: float | None = None
    seed: int | None = None             # None = random, echoed back in the result
    count: int = 1
    init_image: str | None = None       # img2img source, when the model supports it
    strength: float = 0.6
    # Diffusion models only really understand English (see images/prompt.py):
    # None = translate when the prompt does not look English, True/False force it.
    translate: bool | None = None
    original_prompt: str = ""            # filled in when the prompt was translated

    # ------------------------------------------------------------------ helpers
    def normalized(self, *, multiple: int = 16, max_dimension: int = MAX_DIMENSION) -> "ImageParams":
        """A validated copy with sizes rounded to ``multiple`` and sane limits.

        Raises :class:`ImageEngineError` (Polish) for values that cannot be fixed
        silently: an empty prompt, a negative count, a missing ``init_image``.
        """
        prompt = (self.prompt or "").strip()
        if not prompt:
            raise ImageEngineError("Podaj opis obrazu (prompt).")
        if self.init_image and not Path(self.init_image).expanduser().is_file():
            raise ImageEngineError(f"Nie znaleziono obrazu wejściowego: {self.init_image}")
        width = _round_to(self.width, multiple, max_dimension)
        height = _round_to(self.height, multiple, max_dimension)
        steps = self.steps
        if steps is not None:
            steps = max(1, min(int(steps), MAX_STEPS))
        guidance = None if self.guidance is None else float(self.guidance)
        seed = None if self.seed is None else int(self.seed) % (SEED_MAX + 1)
        count = max(1, min(int(self.count or 1), MAX_COUNT))
        strength = min(1.0, max(0.0, float(self.strength)))
        return ImageParams(
            prompt=prompt,
            negative_prompt=(self.negative_prompt or "").strip(),
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            count=count,
            init_image=str(Path(self.init_image).expanduser()) if self.init_image else None,
            strength=strength,
            translate=self.translate,
            original_prompt=self.original_prompt,
        )

    def seeds(self) -> list[int]:
        """One seed per requested image (random ones when ``seed`` is None)."""
        if self.seed is None:
            return [random.randint(0, SEED_MAX) for _ in range(max(1, self.count))]
        return [(int(self.seed) + i) % (SEED_MAX + 1) for i in range(max(1, self.count))]

    def to_payload(self) -> dict[str, Any]:
        """JSON body for the worker's ``POST /generate`` (camelCase)."""
        return {
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance": self.guidance,
            "seed": self.seed,
            "count": self.count,
            "initImage": self.init_image,
            "strength": self.strength,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "ImageParams":
        """Inverse of :meth:`to_payload` (used inside the workers)."""
        return cls(
            prompt=str(data.get("prompt") or ""),
            negative_prompt=str(data.get("negativePrompt") or ""),
            width=int(data.get("width") or 1024),
            height=int(data.get("height") or 1024),
            steps=_opt_int(data.get("steps")),
            guidance=_opt_float(data.get("guidance")),
            seed=_opt_int(data.get("seed")),
            count=int(data.get("count") or 1),
            init_image=data.get("initImage") or None,
            strength=float(data.get("strength") if data.get("strength") is not None else 0.6),
        )


@dataclass
class ImageProgress:
    """One progress tick of a running generation."""

    index: int = 0                       # which image of ``count`` is being made
    step: int = 0                        # finished denoising steps (0 = just started)
    steps: int = 0                       # total steps, 0 when the backend cannot tell
    count: int = 1
    preview_path: str | None = None
    elapsed_ms: int = 0
    message: str = ""                    # Polish status line, e.g. "Dekodowanie obrazu…"

    @property
    def fraction(self) -> float | None:
        """Progress in 0…1, or None for indeterminate progress."""
        if self.steps <= 0:
            return None
        done = (self.index + min(self.step / self.steps, 1.0)) / max(1, self.count)
        return max(0.0, min(1.0, done))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "step": self.step,
            "steps": self.steps,
            "count": self.count,
            "previewPath": self.preview_path,
            "elapsedMs": self.elapsed_ms,
            "message": self.message,
            "fraction": self.fraction,
        }


@dataclass
class ImageResult:
    """One finished image on disk."""

    id: str
    path: str
    width: int
    height: int
    seed: int
    model_key: str
    prompt: str
    steps: int
    guidance: float | None
    elapsed_ms: int
    created_at: float
    negative_prompt: str = ""
    index: int = 0
    size_bytes: int = 0
    model_name: str = ""
    #: the prompt the user typed, when it was translated to English first
    original_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        """camelCase DTO (the gallery/server add ``url`` and ``favorite``)."""
        return {
            "originalPrompt": self.original_prompt,
            "id": self.id,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "modelKey": self.model_key,
            "modelName": self.model_name,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "steps": self.steps,
            "guidance": self.guidance,
            "elapsedMs": self.elapsed_ms,
            "createdAt": int(self.created_at * 1000),
            "sizeBytes": self.size_bytes,
            "index": self.index,
        }


# --------------------------------------------------------------------------- #
# Family defaults
# --------------------------------------------------------------------------- #

# steps / guidance / sizes that produce a good image without the user tuning
# anything.  Verified against the model cards and mflux's own defaults table
# (``mflux.cli.defaults.MODEL_INFERENCE_STEPS``, September 2026).
_FAMILY_DEFAULTS: dict[str, dict[str, Any]] = {
    "flux": {"steps": 4, "guidance": None, "sizes": ["1024x1024", "1024x768", "768x1024"]},
    "flux2": {"steps": 4, "guidance": 1.0, "sizes": ["1024x1024", "1024x768", "768x1024"]},
    "qwen-image": {"steps": 20, "guidance": 4.0, "sizes": ["1024x1024", "1024x768", "768x1024"]},
    "sd1": {"steps": 20, "guidance": 7.0, "sizes": ["512x512", "512x768", "768x512"]},
    "sdxl": {"steps": 25, "guidance": 7.0, "sizes": ["1024x1024", "1024x768", "768x1024"]},
    "sd3": {"steps": 28, "guidance": 4.5, "sizes": ["1024x1024", "1024x768", "768x1024"]},
    "unknown": {"steps": 20, "guidance": 5.0, "sizes": ["1024x1024", "512x512"]},
}


def family_defaults(family: str | None) -> dict[str, Any]:
    """Default steps/guidance/sizes for an image family (never raises)."""
    base = _FAMILY_DEFAULTS.get((family or "unknown").lower(), _FAMILY_DEFAULTS["unknown"])
    return dict(base)


def default_steps_for(family: str | None, model_key: str = "") -> int:
    """Default step count, refined by well-known names (schnell/turbo/lightning = 4)."""
    name = (model_key or "").lower()
    if any(tag in name for tag in ("schnell", "turbo", "lightning", "-lcm", "klein")):
        return 4
    return int(family_defaults(family)["steps"])


# --------------------------------------------------------------------------- #
# Engine contract
# --------------------------------------------------------------------------- #


class ImageEngine(abc.ABC):
    """One loaded image model (see ``docs/IMAGES.md``).

    Generation is **not** concurrent: one job per engine at a time.  The runtime
    serialises calls; an engine only has to make cancellation work.
    """

    backend: str = ""
    format: str = ""

    def __init__(self, instance_id: str, model: LocalModel) -> None:
        self.instance_id = instance_id
        self.model = model

    # lifecycle ------------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None:
        """Spawn the worker and wait until the model is loaded.

        Raises :class:`ImageEngineStartError` with a Polish message (and the tail
        of the worker log) when that fails."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Terminate the worker (SIGTERM, then SIGKILL). Idempotent."""

    @property
    @abc.abstractmethod
    def running(self) -> bool: ...

    @property
    def pid(self) -> int | None:
        return None

    @property
    def port(self) -> int | None:
        return None

    @property
    def props(self) -> dict[str, Any]:
        """What the worker reported after loading (family, defaults, support flags)."""
        return {}

    # inference ------------------------------------------------------------
    @abc.abstractmethod
    def generate(
        self,
        params: ImageParams,
        *,
        output_dir: str | os.PathLike[str] | None = None,
        ids: Sequence[str] | None = None,
    ) -> AsyncIterator[ImageProgress | ImageResult]:
        """Stream progress and finished images for ``params``.

        ``output_dir``/``ids`` let the caller (the runtime, which owns the
        gallery layout) decide where the PNGs land; engines fall back to a
        temporary directory and generated ids."""

    @abc.abstractmethod
    async def cancel(self) -> bool:
        """Ask the engine to stop the running generation.

        Returns True when the backend could cancel by itself; False when the
        process had to be (or must be) stopped instead."""

    async def memory_bytes(self) -> int | None:
        """Measured memory of the instance, if the engine can tell."""
        return None


# --------------------------------------------------------------------------- #
# Worker/HTTP implementation shared by the backends
# --------------------------------------------------------------------------- #


class WorkerImageEngine(ImageEngine):
    """Base class for engines that run a backend in a worker subprocess.

    Subclasses implement :meth:`_worker_module` and :meth:`_worker_args`; the
    process lifecycle, HTTP client, SSE parsing and cancellation live here.
    """

    host = "127.0.0.1"
    startup_timeout_base_s = 120.0          # image models load slowly (big VAE/T5 weights)
    startup_timeout_per_gb_s = 60.0
    stop_grace_s = 5.0
    health_poll_s = 0.3
    request_timeout_s = 60.0
    log_tail_lines = 30

    def __init__(
        self,
        instance_id: str,
        model: LocalModel,
        *,
        settings: Any | None = None,
        python_path: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(instance_id, model)
        self.settings = settings
        self._python = python_path or _python_from_settings(settings) or sys.executable
        self.sandboxed = False
        self._sandbox_profile: Path | None = None
        self._sandbox_off = False
        self._env_extra = dict(env or {})
        self._proc: asyncio.subprocess.Process | None = None
        self._exit_task: asyncio.Future[int] | None = None
        self._port: int | None = None
        self._session: Any | None = None       # aiohttp.ClientSession
        self._ready = False
        self._stopping = False
        self._stop_task: asyncio.Future[None] | None = None
        self._log_path: Path | None = None
        self._props: dict[str, Any] = {}
        self.exit_code: int | None = None
        self.started_at: float | None = None
        self.load_ms: float | None = None

    # ------------------------------------------------------------------ hooks
    @abc.abstractmethod
    def _worker_module(self) -> str:
        """``localcode.images.<module>`` run with ``python -m``."""

    @abc.abstractmethod
    def _worker_args(self) -> list[str]:
        """Backend-specific arguments (model path, components, …)."""

    def _log_name(self) -> str:
        return f"image-{self.backend or 'engine'}-{_safe_name(self.instance_id)}"

    def _build_command(self, port: int) -> list[str]:
        return [
            self._python, "-m", self._worker_module(),
            "--host", self.host, "--port", str(port),
            "--parent-pid", str(os.getpid()),
            *self._worker_args(),
        ]

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
    def base_url(self) -> str | None:
        return f"http://{self.host}:{self._port}" if self._port else None

    @property
    def props(self) -> dict[str, Any]:
        return dict(self._props)

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def log_tail(self, lines: int | None = None) -> str:
        """Last lines of the worker log (ANSI codes removed)."""
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
        size = float(self.model.size_bytes or 0)
        if not size:
            try:
                p = Path(self.model.path)
                size = float(p.stat().st_size) if p.is_file() else 0.0
            except OSError:
                size = 0.0
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

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self._env_extra)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        return env

    async def start(self) -> None:
        if self.running:
            return
        if self._proc is not None:
            await self.stop()
        self._stopping = False
        self._ready = False
        self.exit_code = None
        t0 = time.monotonic()
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - aiohttp is a hard dependency
            raise ImageEngineError("Brak pakietu aiohttp wymaganego przez silnik obrazów.") from exc
        try:
            self._log_path = self._prepare_log()
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                connector=aiohttp.TCPConnector(limit=0, force_close=False),
            )
            self._port = _free_port(self.host)
            await self._spawn(self._build_command(self._port))
            if not await self._wait_healthy(self._startup_timeout()):
                proc = self._proc
                code = proc.returncode if proc is not None else None
                tail = self.log_tail()
                if self.sandboxed and sandbox.mode(self.settings) == "auto":
                    # Isolation is worth having, but not at the price of a model
                    # the user can no longer run; the status says which happened.
                    log.warning("image engine %s did not start in the sandbox, retrying without it",
                                self.instance_id)
                    self._sandbox_off = True
                    await self._kill_worker()
                    self._port = _free_port(self.host)
                    await self._spawn(self._build_command(self._port))
                    if not await self._wait_healthy(self._startup_timeout()):
                        tail = self.log_tail()
                        raise ImageEngineStartError(self._failure_message(tail, code),
                                                    log_tail=tail, exit_code=code)
                else:
                    raise ImageEngineStartError(self._failure_message(tail, code),
                                                log_tail=tail, exit_code=code)
            self._props = await self._get_props()
            self._ready = True
            self.started_at = time.time()
            self.load_ms = (time.monotonic() - t0) * 1000.0
            log.info(
                "image engine %s ready on port %s (pid %s) in %.1fs",
                self.instance_id, self._port, self.pid, self.load_ms / 1000,
            )
        except BaseException:
            await self.stop()
            raise

    def _sandbox_plan(self) -> "SandboxPlan":
        """What the image worker may touch (:mod:`localcode.security.sandbox`).

        Image models are the pickle-heaviest family in the store, and the worker
        is Python: besides the weights it has to read its own interpreter and
        LocalCode's source, which in a development install sit under
        ``~/Documents``. It writes only where every engine writes — its own
        directories and the caches — so generated images go back over HTTP.
        """
        import localcode

        python = Path(self._python).expanduser()
        read = [Path(self.model.path).expanduser().parent, python.parent, Path(sys.prefix),
                Path(localcode.__file__).resolve().parent.parent]
        return SandboxPlan(read=read, write=[])

    def _sandboxed_command(self, cmd: list[str]) -> list[str]:
        """``cmd`` wrapped in ``sandbox-exec``, honouring ``security.sandbox``."""
        sandbox.cleanup(self._sandbox_profile)
        self._sandbox_profile = None
        self.sandboxed = False
        if self._sandbox_off:
            return cmd
        cmd, self.sandboxed, self._sandbox_profile = sandbox.wrap(
            cmd, self._sandbox_plan(), settings=self.settings)
        return cmd

    async def _spawn(self, cmd: list[str]) -> None:
        assert self._log_path is not None
        cmd = self._sandboxed_command(cmd)
        with open(self._log_path, "ab") as fh:
            header = f"# {datetime.now().isoformat(timespec='seconds')} {shlex.join(cmd)}\n"
            fh.write(header.encode())
            fh.flush()
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=fh,
                    stderr=asyncio.subprocess.STDOUT,
                    env=self._env(),
                    start_new_session=True,
                )
            except OSError as exc:
                raise ImageEngineStartError(
                    f"Nie można uruchomić procesu generatora obrazów „{cmd[0]}”: {exc.strerror or exc}."
                ) from exc
        _register_group(self._proc.pid, f"{self._log_name()} ({self.model.key})")
        self._exit_task = asyncio.ensure_future(self._proc.wait())

    async def _health_status(self) -> int | None:
        session = self._session
        if session is None or self._port is None:
            return None
        import aiohttp

        try:
            async with session.get(
                f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                await resp.read()
                return resp.status
        except (aiohttp.ClientError, TimeoutError, OSError):
            return None

    async def _wait_healthy(self, timeout: float) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            proc = self._proc
            if proc is None or proc.returncode is not None or self._stopping:
                return False
            if await self._health_status() == 200:
                return True
            proc = self._proc
            if proc is None or proc.returncode is not None or self._stopping:
                return False
            if loop.time() >= deadline:
                tail = self.log_tail()
                raise ImageEngineStartError(
                    f"Model obrazów „{self.model.display_name}” nie wczytał się w ciągu {int(timeout)} s.\n\n"
                    f"Ostatnie linie logu ({self._log_path}):\n{tail}",
                    log_tail=tail,
                )
            await self._wait_exit(self.health_poll_s)

    async def _wait_exit(self, timeout: float) -> bool:
        task = self._exit_task
        if task is None:
            return self._proc is None or self._proc.returncode is not None
        done, _ = await asyncio.wait({task}, timeout=timeout)
        return bool(done)

    def _failure_message(self, tail: str, code: int | None) -> str:
        hint = _failure_hint(tail, code)
        status = ""
        if code is not None:
            status = f" (kod wyjścia {code})" if code >= 0 else f" (sygnał {-code})"
        msg = f"Nie udało się wczytać modelu obrazów „{self.model.display_name}”{status}."
        if hint:
            msg += f" {hint}"
        if tail:
            msg += f"\n\nOstatnie linie logu ({self._log_path}):\n{tail}"
        return msg

    async def _kill_worker(self) -> None:
        """Stop just the process, keeping the session (used by the sandbox retry)."""
        proc, self._proc = self._proc, None
        self._exit_task = None
        if proc is None:
            return
        await _terminate_process_group(proc, 1.0)
        self.exit_code = proc.returncode
        _unregister_group(proc.pid)

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
        sandbox.cleanup(self._sandbox_profile)
        self._sandbox_profile = None
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.close()
            except Exception:  # noqa: BLE001
                log.debug("closing image engine session failed", exc_info=True)
        proc = self._proc
        if proc is not None:
            await _terminate_process_group(proc, self.stop_grace_s)
            self.exit_code = proc.returncode
            _unregister_group(proc.pid)
            if self._proc is proc:
                self._proc = None
                self._exit_task = None
            log.info("image engine %s stopped (pid %d, exit %s)", self.instance_id, proc.pid, proc.returncode)

    # ------------------------------------------------------------ HTTP helpers
    def _require_session(self) -> Any:
        session = self._session
        if session is None or getattr(session, "closed", False) or self._port is None:
            raise ImageEngineUnavailableError(
                "Model obrazów został zatrzymany." if self._stopping else "Model obrazów nie jest uruchomiony."
            )
        return session

    async def _request_json(self, method: str, path: str, body: Any = None,
                            *, timeout: float | None = None) -> Any:
        import aiohttp

        limit = timeout if timeout is not None else self.request_timeout_s
        session = self._require_session()
        try:
            async with session.request(
                method, f"{self.base_url}{path}", json=body,
                timeout=aiohttp.ClientTimeout(total=limit, sock_connect=10),
            ) as resp:
                text = await resp.text(errors="replace")
                if resp.status >= 400:
                    raise ImageEngineError(_error_message(text, path, resp.status))
        except TimeoutError as exc:
            raise ImageEngineError(f"Generator obrazów nie odpowiedział w ciągu {int(limit)} s ({path}).") from exc
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
            raise await self._connection_error(exc) from exc
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise ImageEngineError(f"Generator obrazów zwrócił niepoprawną odpowiedź ({path}).") from exc

    async def _connection_error(self, exc: BaseException) -> ImageEngineError:
        if self._stopping:
            return ImageEngineUnavailableError("Model obrazów został zatrzymany.")
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await self._wait_exit(0.5)
        if proc is None or proc.returncode is not None:
            code = proc.returncode if proc is not None else None
            self._ready = False
            tail = self.log_tail()
            hint = _failure_hint(tail, code) or ""
            msg = f"Proces generatora obrazów „{self.model.display_name}” przestał działać"
            msg += f" (kod wyjścia {code})." if code is not None else "."
            if hint:
                msg += f" {hint}"
            if tail:
                msg += f"\n\nOstatnie linie logu ({self._log_path}):\n{tail}"
            return ImageEngineUnavailableError(msg)
        return ImageEngineUnavailableError(
            f"Brak połączenia z procesem generatora obrazów ({type(exc).__name__}: {exc})."
        )

    async def _get_props(self) -> dict[str, Any]:
        data = await self._request_json("GET", "/props", timeout=30)
        return data if isinstance(data, dict) else {}

    async def memory_bytes(self) -> int | None:
        """Backend-reported memory, falling back to the process footprint."""
        value: int | None = None
        if self.running:
            try:
                data = await self._request_json("GET", "/memory", timeout=10)
                if isinstance(data, dict):
                    raw = data.get("bytes")
                    value = int(raw) if isinstance(raw, (int, float)) and raw > 0 else None
            except ImageEngineError:
                value = None
        if value:
            return value
        from localcode.runtime import memory as rt_memory

        return rt_memory.process_memory_bytes(self.pid)

    # ---------------------------------------------------------------- generate
    async def generate(  # type: ignore[override]
        self,
        params: ImageParams,
        *,
        output_dir: str | os.PathLike[str] | None = None,
        ids: Sequence[str] | None = None,
    ) -> AsyncIterator[ImageProgress | ImageResult]:
        """Stream progress frames and finished images (see the module docstring)."""
        import aiohttp

        if not self.running:
            raise ImageEngineUnavailableError("Model obrazów nie jest uruchomiony.")
        normalized = params.normalized(multiple=self.size_multiple())
        out_dir = Path(output_dir).expanduser() if output_dir else Path(_default_output_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        image_ids = list(ids or [])
        while len(image_ids) < normalized.count:
            image_ids.append(new_image_id())
        outputs = [str(out_dir / f"{image_id}.png") for image_id in image_ids[: normalized.count]]
        body = dict(normalized.to_payload())
        body["outputs"] = outputs
        session = self._require_session()
        resp: Any = None
        t0 = time.perf_counter()
        try:
            try:
                resp = await session.post(
                    f"{self.base_url}/generate", json=body,
                    timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                )
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                raise await self._connection_error(exc) from exc
            if resp.status >= 400:
                text = await resp.text(errors="replace")
                raise ImageEngineError(_error_message(text, "/generate", resp.status))
            decoder = SSEDecoder()
            done = False
            try:
                async for raw in resp.content.iter_any():
                    for event in decoder.feed(raw):
                        item = self._frame(event.data, normalized, image_ids, t0)
                        if item is None:
                            continue
                        if item is _DONE:
                            done = True
                            return
                        yield item  # type: ignore[misc]
                for event in decoder.finish():
                    item = self._frame(event.data, normalized, image_ids, t0)
                    if item is None:
                        continue
                    if item is _DONE:
                        done = True
                        return
                    yield item  # type: ignore[misc]
            except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as exc:
                raise await self._connection_error(exc) from exc
            except RuntimeError as exc:
                if self._stopping or session.closed or "connection closed" in str(exc).lower():
                    raise await self._connection_error(exc) from exc
                raise
            if not done:
                raise await self._connection_error(EOFError("stream ended"))
        finally:
            if resp is not None:
                resp.close()

    def _frame(self, data: str, params: ImageParams, ids: Sequence[str],
               t0: float) -> ImageProgress | ImageResult | object | None:
        payload = data.strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            frame = json.loads(payload)
        except ValueError:
            log.debug("image engine %s: malformed SSE frame %.200s", self.instance_id, payload)
            return None
        if not isinstance(frame, dict):
            return None
        kind = frame.get("type")
        if kind == "progress":
            return ImageProgress(
                index=int(frame.get("index") or 0),
                step=int(frame.get("step") or 0),
                steps=int(frame.get("steps") or 0),
                count=int(frame.get("count") or params.count),
                preview_path=frame.get("previewPath") or None,
                elapsed_ms=int(frame.get("elapsedMs") or (time.perf_counter() - t0) * 1000),
                message=str(frame.get("message") or ""),
            )
        if kind == "image":
            index = int(frame.get("index") or 0)
            image_id = str(frame.get("id") or (ids[index] if index < len(ids) else new_image_id()))
            path = str(frame.get("path") or "")
            size = 0
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            return ImageResult(
                id=image_id,
                path=path,
                width=int(frame.get("width") or params.width),
                height=int(frame.get("height") or params.height),
                seed=int(frame.get("seed") or 0),
                model_key=self.model.key,
                model_name=self.model.display_name,
                prompt=params.prompt,
                original_prompt=getattr(params, "original_prompt", "") or "",
                negative_prompt=params.negative_prompt,
                steps=int(frame.get("steps") or params.steps or 0),
                guidance=_opt_float(frame.get("guidance")),
                elapsed_ms=int(frame.get("elapsedMs") or (time.perf_counter() - t0) * 1000),
                created_at=time.time(),
                index=index,
                size_bytes=size,
            )
        if kind == "error":
            message = str(frame.get("message") or "Nie udało się wygenerować obrazu.")
            if frame.get("cancelled"):
                raise ImageGenerationCancelled(message)
            raise ImageEngineError(message)
        if kind == "done":
            return _DONE
        return None

    async def cancel(self) -> bool:
        """POST /cancel; stops the process when the backend cannot abort itself."""
        if not self.running:
            return True
        try:
            data = await self._request_json("POST", "/cancel", {}, timeout=10)
        except ImageEngineError:
            data = None
        hard = bool(isinstance(data, dict) and data.get("hard"))
        if hard or data is None:
            await self.stop()
            return not hard
        return True

    # ------------------------------------------------------------------ tuning
    def size_multiple(self) -> int:
        """Pixel grid the backend requires (mflux 16, stable-diffusion.cpp 64)."""
        return 16


# Sentinel for the terminating ``done`` frame.
_DONE = object()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _round_to(value: Any, multiple: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = 1024
    n = max(MIN_DIMENSION, min(n, maximum))
    step = max(1, int(multiple))
    return max(step, int(math.floor(n / step) * step))


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _python_from_settings(settings: Any) -> str | None:
    runtime = getattr(settings, "runtime", None)
    path = getattr(runtime, "python_path", None)
    return str(path) if path else None


def _default_output_dir() -> str:
    from localcode.config import data_dir

    return str(data_dir() / "images")


def _failure_hint(tail: str, code: int | None) -> str | None:
    """Polish one-sentence explanation of a recognised worker failure."""
    low = (tail or "").lower()
    import signal as _signal

    if code == -_signal.SIGKILL:
        return "Proces został zabity przez system – prawdopodobnie zabrakło pamięci."
    if "modulenotfounderror: no module named 'mflux'" in low:
        return backend_missing_message("mflux")
    if "modulenotfounderror: no module named 'stable_diffusion_cpp'" in low:
        return backend_missing_message("sdcpp")
    if any(s in low for s in ("out of memory", "failed to allocate", "insufficient memory", "std::bad_alloc")):
        return ("Za mało pamięci na ten model obrazów. Wyładuj inne modele albo wybierz mniejszy "
                "(np. wersję 4-bitową lub SD 1.5).")
    if "no weights location" in low or "filenotfounderror" in low:
        return "Pliki modelu są niekompletne – pobierz model jeszcze raz."
    return None


def _error_message(text: str, path: str, status: int) -> str:
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        data = {}
    detail = ""
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
        elif isinstance(err, str):
            detail = err
    detail = detail or (text or "").strip()[:300]
    if status == 503:
        return "Model obrazów wciąż się wczytuje. Spróbuj ponownie za chwilę."
    if status == 409:
        return detail or "Generator obrazów jest zajęty innym zadaniem."
    return f"Błąd generatora obrazów (HTTP {status}, {path}): {detail}"


@dataclass
class WorkerDefaults:
    """Generation defaults a worker reports in ``/props``."""

    steps: int = 20
    guidance: float | None = None
    sizes: list[str] = field(default_factory=lambda: ["1024x1024"])
    max_size: int = MAX_DIMENSION
    size_multiple: int = 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "guidance": self.guidance,
            "sizes": list(self.sizes),
            "maxSize": self.max_size,
            "sizeMultiple": self.size_multiple,
        }


# --------------------------------------------------------------------------- #
# Worker side of the protocol (used by mflux_worker.py / sdcpp_worker.py)
# --------------------------------------------------------------------------- #


class CancelledJob(Exception):
    """Raised inside a backend loop when the client asked to cancel."""


class ImageWorkerServer(abc.ABC):
    """aiohttp server that runs ONE image backend in this process.

    The model is loaded and used exclusively from a single worker thread (both
    MLX and ggml dislike being driven from several threads); the event loop only
    moves JSON around.  Subclasses implement :meth:`load`, :meth:`props_payload`,
    :meth:`run_job` and (optionally) :meth:`memory_payload` / :meth:`unload`.
    """

    backend = ""
    #: True when the backend cannot abort a running generation by itself, so the
    #: engine has to stop the process to make a cancel effective.
    hard_cancel = False

    def __init__(self, host: str = "127.0.0.1", port: int = 0, *, parent_pid: int | None = None) -> None:
        self.host = host
        self.port = port
        self.parent_pid = parent_pid
        self._ready = False
        self._load_error: str | None = None
        self._busy = False
        self._cancel = False
        self._pool: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ hooks
    @abc.abstractmethod
    def load(self) -> None:
        """Load the model (runs in the worker thread; raise to fail the start)."""

    @abc.abstractmethod
    def props_payload(self) -> dict[str, Any]:
        """``GET /props`` body."""

    @abc.abstractmethod
    def run_job(self, params: ImageParams, outputs: list[str], emit: Any) -> None:
        """Generate every image of ``params`` (worker thread).

        ``emit(frame_dict)`` is thread-safe and pushes an SSE frame.  Raise
        :class:`CancelledJob` when :meth:`cancelled` becomes true."""

    def memory_payload(self) -> dict[str, Any]:
        return {}

    def unload(self) -> None:
        """Free the model (best effort; called on shutdown)."""

    # --------------------------------------------------------------- utilities
    def cancelled(self) -> bool:
        return self._cancel

    def _emitter(self, queue: "asyncio.Queue[dict[str, Any]]") -> Any:
        loop = self._loop
        assert loop is not None

        def emit(frame: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        return emit

    # ------------------------------------------------------------------ server
    def build_app(self) -> Any:
        from aiohttp import web

        app = web.Application(handler_args={"handler_cancellation": True})
        app.router.add_get("/health", self._health)
        app.router.add_get("/props", self._props)
        app.router.add_get("/memory", self._memory)
        app.router.add_post("/generate", self._generate)
        app.router.add_post("/cancel", self._cancel_handler)
        return app

    async def _health(self, request: Any) -> Any:
        from aiohttp import web

        if self._load_error:
            return web.json_response(
                {"error": {"code": 500, "message": self._load_error, "type": "load_error"}}, status=500)
        if not self._ready:
            return web.json_response(
                {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}, status=503)
        return web.json_response({"status": "ok"})

    def _guard(self) -> Any | None:
        from aiohttp import web

        if self._load_error:
            return web.json_response({"error": {"message": self._load_error}}, status=500)
        if not self._ready:
            return web.json_response(
                {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}, status=503)
        return None

    async def _props(self, request: Any) -> Any:
        from aiohttp import web

        blocked = self._guard()
        if blocked is not None:
            return blocked
        payload = dict(self.props_payload())
        payload.setdefault("backend", self.backend)
        payload.setdefault("hardCancel", self.hard_cancel)
        return web.json_response(payload)

    async def _memory(self, request: Any) -> Any:
        from aiohttp import web

        payload = dict(self.memory_payload())
        if "bytes" not in payload:
            from localcode.runtime import memory as rt_memory

            payload["bytes"] = rt_memory.process_memory_bytes(os.getpid()) or 0
        return web.json_response(payload)

    async def _cancel_handler(self, request: Any) -> Any:
        from aiohttp import web

        self._cancel = True
        return web.json_response({"ok": True, "hard": self.hard_cancel, "busy": self._busy})

    async def _generate(self, request: Any) -> Any:
        from aiohttp import web

        blocked = self._guard()
        if blocked is not None:
            return blocked
        if self._busy:
            return web.json_response(
                {"error": {"message": "Generator obrazów jest zajęty innym zadaniem."}}, status=409)
        try:
            body = await request.json()
        except (ValueError, TypeError):
            return web.json_response({"error": {"message": "Niepoprawne dane zapytania."}}, status=400)
        if not isinstance(body, dict):
            return web.json_response({"error": {"message": "Niepoprawne dane zapytania."}}, status=400)
        params = ImageParams.from_payload(body)
        outputs = [str(p) for p in (body.get("outputs") or [])]
        while len(outputs) < max(1, params.count):
            outputs.append(str(Path(_default_output_dir()) / f"{new_image_id()}.png"))
        for out in outputs:
            Path(out).parent.mkdir(parents=True, exist_ok=True)

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                     "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
        await response.prepare(request)
        self._busy = True
        self._cancel = False
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        self._loop = loop
        emit = self._emitter(queue)

        def job() -> dict[str, Any]:
            try:
                self.run_job(params, outputs, emit)
                return {"type": "done"}
            except CancelledJob as exc:
                return {"type": "error", "cancelled": True,
                        "message": str(exc) or "Generowanie obrazu zostało przerwane."}
            except MemoryError:
                return {"type": "error", "message": "Zabrakło pamięci podczas generowania obrazu."}
            except BaseException as exc:  # noqa: BLE001 - reported to the client
                log.exception("image generation failed")
                return {"type": "error", "message": f"{type(exc).__name__}: {exc}"}

        task = loop.run_in_executor(self._pool, job)
        try:
            while True:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    await _write_sse(response, getter.result())
                    continue
                getter.cancel()
                while not queue.empty():
                    await _write_sse(response, queue.get_nowait())
                await _write_sse(response, await task)
                break
        except (asyncio.CancelledError, ConnectionResetError):
            # The client went away: cancel and let the job finish in the background.
            self._cancel = True
            raise
        finally:
            self._busy = False
            self._cancel = False
        with contextlib_suppress():
            await response.write_eof()
        return response

    # ---------------------------------------------------------------- run loop
    async def serve(self) -> None:
        """Bind the port, load the model, then serve until stopped."""
        from concurrent.futures import ThreadPoolExecutor

        from aiohttp import web

        self._loop = asyncio.get_running_loop()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.backend}-worker")
        app = self.build_app()
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        port = self.port
        for socket_ in getattr(getattr(site, "_server", None), "sockets", None) or []:
            port = socket_.getsockname()[1]
        print(f"IMAGE_WORKER_READY port={port} backend={self.backend}", flush=True)
        stop = asyncio.Event()
        _install_signal_handlers(stop)
        watchdog = asyncio.ensure_future(_watch_parent(self.parent_pid, stop))
        try:
            try:
                await self._loop.run_in_executor(self._pool, self.load)
                self._ready = True
                print(f"IMAGE_WORKER_LOADED backend={self.backend}", flush=True)
            except BaseException as exc:  # noqa: BLE001 - reported through /health
                self._load_error = f"{type(exc).__name__}: {exc}"
                log.exception("loading the image model failed")
                raise
            await stop.wait()
        finally:
            watchdog.cancel()
            with contextlib_suppress():
                await runner.cleanup()
            with contextlib_suppress():
                await self._loop.run_in_executor(self._pool, self.unload)
            self._pool.shutdown(wait=False, cancel_futures=True)

    def main(self) -> int:
        """Entry point used by ``python -m localcode.images.<backend>_worker``."""
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:  # pragma: no cover - signal path
            return 0
        except BaseException as exc:  # noqa: BLE001
            print(f"IMAGE_WORKER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0


async def _write_sse(response: Any, frame: dict[str, Any]) -> None:
    await response.write(b"data: " + json.dumps(frame, ensure_ascii=False).encode("utf-8") + b"\n\n")


async def _watch_parent(parent_pid: int | None, stop: "asyncio.Event") -> None:
    """Exit when the parent process dies (never leave an orphan model in RAM)."""
    if not parent_pid or parent_pid <= 1:
        return
    while not stop.is_set():
        await asyncio.sleep(2.0)
        try:
            os.kill(int(parent_pid), 0)
        except (ProcessLookupError, PermissionError, OSError):
            log.warning("parent process %s is gone, shutting the image worker down", parent_pid)
            stop.set()
            return


def _install_signal_handlers(stop: "asyncio.Event") -> None:
    import signal as _signal

    loop = asyncio.get_running_loop()
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):  # pragma: no cover
            pass


def contextlib_suppress() -> Any:
    """``contextlib.suppress(Exception)`` without importing contextlib everywhere."""
    import contextlib

    return contextlib.suppress(Exception)
