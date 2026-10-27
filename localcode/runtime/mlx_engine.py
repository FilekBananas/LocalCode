"""MLX engine: ``python -m localcode.runtime.mlx_worker`` in a subprocess.

The worker (``runtime/mlx_worker.py``) serves the subset of llama-server's HTTP
protocol that ``HttpEngine`` speaks, so this module is the MLX twin of
``runtime/llama_engine.py``: it only decides *how the worker is launched* and
*how its failures are explained*; process supervision, the SSE client,
``/props`` reading, template capability detection and cancellation all live in
``HttpEngine``.  Design decisions (verified against mlx-lm 0.31.3 / mlx 0.32):

* **Interpreter.**  ``settings.runtime.python_path or sys.executable``.  The
  worker must be able to ``import localcode`` and ``import mlx_lm``; when the
  configured interpreter is a different environment we still put LocalCode's
  own source root on its ``PYTHONPATH`` so only ``mlx-lm`` has to be installed
  there.  A missing ``mlx_lm``/``localcode`` is recognised in the worker log and
  reported in Polish instead of a bare traceback.
* **Per-slot context.**  ``--ctx`` is the context *per slot* and ``--parallel``
  the number of slots, exactly like ``LoadConfig`` — no arithmetic needed (the
  GGUF engine has to multiply because llama.cpp's ``-c`` is the total).  The
  effective value is still read back from ``/props``/``/slots`` by the base
  class.
* **KV cache quantisation.**  ``LoadConfig.kv_cache_type`` is a single setting
  shared by both formats, so the llama.cpp spellings are accepted as well:
  ``8bit``/``q8_0`` → ``--kv-bits 8``, ``4bit``/``q4_0`` → ``--kv-bits 4``,
  anything else → no flag (f16).  The mapping goes through
  ``runtime.memory.normalize_kv_type`` so the engine and the memory estimate
  always agree.  mlx-lm cannot batch a quantised KV cache: with ``--kv-bits``
  the worker serves its slots sequentially (it says so in its log), which is a
  throughput trade-off, never a correctness one.
* **Memory limit.**  ``--memory-limit-mb`` is passed when the caller gives one
  explicitly or when the user configured ``runtime.ram_limit_gb``.  MLX treats
  it as a guideline: the buffer cache is released when the process approaches
  the limit, and only an allocation that cannot be satisfied by RAM *and* swap
  raises.  It is therefore a guard against one worker eating the whole machine,
  not the budget itself — the budget is enforced by ``RuntimeManager``.
* **Lifetime.**  ``--parent-pid`` makes the worker exit on its own if this
  process disappears without being able to kill it (``HttpEngine`` registers
  the process group for the emergency kill, but a ``SIGKILL``-ed core cannot
  run any handler and a multi-GB MLX worker must not survive it).
* **Memory reporting.**  ``memory_bytes()`` prefers the worker's ``/memory``
  endpoint (``mx.get_active_memory() + mx.get_cache_memory()`` — the bytes MLX
  holds from the OS, which is what the runtime budget cares about) and falls
  back to the RSS of the process tree.  Measured on an M5 with
  Qwen3-0.6B-4bit and 1k×2 slots: MLX active+cache ≈ 0.34 GB, RSS ≈ 0.66 GB —
  the difference is the Python interpreter plus the mlx/tokenizer libraries,
  which is just as resident, so the larger of the two is reported.
* **Embeddings.**  The worker is text-only (it answers 501); ``embed()`` fails
  fast with a Polish message instead of doing a round trip.

The worker has no API key: it binds ``127.0.0.1`` only and, unlike
llama-server, does not accept credentials, so ``_api_key`` stays ``None``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from localcode.config import Settings, load_settings
from localcode.hub.types import LocalModel
from localcode.runtime import memory as memory_mod
from localcode.runtime.base import EngineError, LoadConfig
from localcode.security.sandbox import SandboxPlan
from localcode.runtime.http_engine import EngineStartError, HttpEngine, _safe_name

__all__ = ["MlxEngine", "WORKER_MODULE", "kv_bits_for", "mlx_source_root"]

log = logging.getLogger(__name__)

#: Module the worker process runs (``python -m WORKER_MODULE``).
WORKER_MODULE = "localcode.runtime.mlx_worker"

#: MLX weight files (``mlx_lm`` loads safetensors, older repos ship .npz);
#: only used to size the startup timeout when the library did not measure them.
_WEIGHT_SUFFIXES = (".safetensors", ".npz")

_MIN_MEMORY_LIMIT_MB = 256          # the worker rejects anything smaller
_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def kv_bits_for(kv_cache_type: str | None) -> int | None:
    """``--kv-bits`` for a ``LoadConfig.kv_cache_type`` (``None`` = f16 cache).

    Accepts both vocabularies (``8bit``/``q8_0``, ``4bit``/``q4_0``, …) through
    ``runtime.memory.normalize_kv_type``, so a single user setting drives the
    GGUF and the MLX engine consistently.
    """
    kind = memory_mod.normalize_kv_type(kv_cache_type, "mlx")
    return {"8bit": 8, "4bit": 4}.get(kind)


def mlx_source_root() -> str:
    """Directory that contains the ``localcode`` package (for ``PYTHONPATH``)."""
    return str(Path(__file__).resolve().parent.parent.parent)


class MlxEngine(HttpEngine):
    """One MLX worker process serving one MLX model (see module docstring)."""

    format = "mlx"

    def __init__(
        self,
        instance_id: str,
        model: LocalModel,
        config: LoadConfig,
        *,
        settings: Settings | None = None,
        python_path: str | None = None,
        memory_limit_mb: int | None = None,
    ) -> None:
        super().__init__(instance_id, model, config)
        self._settings = settings
        self._python = python_path
        self._memory_limit_mb = memory_limit_mb
        self.command: list[str] = []      # argv of the last spawn (diagnostics)

    # ----------------------------------------------------------- properties
    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def python_path(self) -> str:
        """Interpreter used for the worker (settings → ``sys.executable``)."""
        return self._python or self.settings.runtime.python_path or sys.executable

    @property
    def kv_bits(self) -> int | None:
        """KV quantisation requested for this instance (``None`` = f16)."""
        return kv_bits_for(self.config.kv_cache_type)

    @property
    def memory_limit_mb(self) -> int | None:
        """``mx.set_memory_limit`` value passed to the worker, if any."""
        if self._memory_limit_mb is not None:
            value = int(self._memory_limit_mb)
        else:
            limit_gb = self.settings.runtime.ram_limit_gb
            if not limit_gb or float(limit_gb) <= 0:
                return None
            value = int(float(limit_gb) * 1024)
        return max(_MIN_MEMORY_LIMIT_MB, value)

    @property
    def batching(self) -> bool:
        """True when the worker batches its slots (no KV quantisation)."""
        return bool(self._props.get("batching"))

    @property
    def worker_kv_bits(self) -> int | None:
        """KV quantisation the worker actually ended up using."""
        value = self._props.get("kv_bits")
        return int(value) if isinstance(value, int) else None

    # --------------------------------------------------------------- hooks
    def _log_name(self) -> str:
        return f"mlx-{_safe_name(self.instance_id)}"

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        root = mlx_source_root()
        existing = env.get("PYTHONPATH", "")
        parts = [p for p in existing.split(os.pathsep) if p]
        if root not in parts:
            parts.insert(0, root)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    async def _before_spawn(self) -> None:
        if self.model.format != "mlx":
            raise EngineStartError(
                f"Model „{self.model.display_name}” nie jest w formacie MLX – ten silnik go nie uruchomi."
            )
        path = Path(self.model.path).expanduser()
        if not path.is_dir():
            raise EngineStartError(
                f"Katalog modelu MLX nie istnieje: {self.model.path}. Pobierz model ponownie."
            )
        if not (path / "config.json").is_file():
            raise EngineStartError(
                f"Katalog „{self.model.path}” nie wygląda na model MLX – brakuje pliku config.json."
            )
        python = self.python_path
        if not (os.path.isfile(python) and os.access(python, os.X_OK)):
            raise EngineStartError(
                f"Nie znaleziono interpretera Pythona dla modeli MLX: {python}. "
                "Wskaż poprawną ścieżkę w ustawieniach (Runtime → ścieżka Pythona)."
            )
        if int(self.config.n_ctx or 0) < 16:
            raise EngineStartError(
                f"Kontekst modelu musi mieć co najmniej 16 tokenów (podano {self.config.n_ctx})."
            )
        if self.config.embedding:
            raise EngineStartError(
                "Silnik MLX nie obsługuje trybu embeddingów – użyj modelu GGUF do embeddingów "
                "(np. nomic-embed-text)."
            )

    def _per_slot_ctx(self) -> int:
        """Context per slot, capped at what the model was trained for.

        llama.cpp caps ``-c`` at ``n_ctx_train`` on its own; mlx-lm would
        happily allocate a KV cache beyond the trained window (more memory,
        worse answers), so the cap is applied here and the effective value is
        read back from ``/props`` like everywhere else.
        """
        want = int(self.config.n_ctx)
        native = self.model.meta.context_length
        if native and want > int(native):
            log.warning(
                "engine %s: requested n_ctx %d per slot, the model was trained for %d",
                self.instance_id, want, int(native),
            )
            return int(native)
        return want

    def _sandbox_plan(self) -> SandboxPlan | None:
        """The MLX worker reads its weights, its interpreter and its packages.

        Unlike ``llama-server`` this is Python: it has to read the interpreter's
        prefix and LocalCode's own source, which in a development install sit
        under ``~/Documents`` — a directory the profile otherwise denies. The
        plan re-opens exactly those paths and nothing else.
        """
        import localcode

        python = Path(self.python_path).expanduser()
        read = [Path(self.model.path).expanduser().parent, python.parent,
                Path(sys.prefix), Path(localcode.__file__).resolve().parent.parent]
        return SandboxPlan(read=read, write=[])

    def _build_command(self, port: int) -> list[str]:
        cfg = self.config
        cmd = [
            self.python_path,
            "-m", WORKER_MODULE,
            "--model", str(Path(self.model.path).expanduser()),
            "--host", self.host,
            "--port", str(port),
            "--parallel", str(max(1, int(cfg.n_parallel or 1))),
            "--ctx", str(self._per_slot_ctx()),
        ]
        kv_bits = self.kv_bits
        if kv_bits is not None:
            cmd += ["--kv-bits", str(kv_bits)]
        limit = self.memory_limit_mb
        if limit is not None:
            cmd += ["--memory-limit-mb", str(limit)]
        cmd += ["--parent-pid", str(os.getpid())]
        cmd += [str(a) for a in cfg.extra_args]
        self.command = list(cmd)
        return cmd

    def _startup_timeout(self) -> float:
        """Same budget as the base class, but the size of an MLX model is the
        size of its weight files, not of the directory inode."""
        size = self.model.size_bytes or 0
        if not size:
            size = self._weights_bytes()
        return self.startup_timeout_base_s + self.startup_timeout_per_gb_s * (size / 1e9)

    def _weights_bytes(self) -> int:
        try:
            entries = list(Path(self.model.path).expanduser().iterdir())
        except OSError:
            return 0
        total = 0
        for entry in entries:
            if entry.suffix.lower() in _WEIGHT_SUFFIXES:
                try:
                    total += entry.stat().st_size
                except OSError:
                    continue
        return total

    def _failure_hint(self, log_text: str, exit_code: int | None) -> str | None:
        m = _MODULE_RE.search(log_text)
        if m:
            module = m.group(1).split(".")[0]
            if module in ("mlx", "mlx_lm"):
                return (
                    f"Interpreter {self.python_path} nie ma zainstalowanego pakietu mlx-lm "
                    "(„pip install mlx-lm”). Modele MLX działają tylko na Apple Silicon."
                )
            if module == "localcode":
                return (
                    f"Interpreter {self.python_path} nie widzi pakietu localcode – wskaż w ustawieniach "
                    "Pythona ze środowiska, w którym zainstalowano LocalCode."
                )
            return f"Brakuje modułu Pythona „{m.group(1)}” w interpreterze {self.python_path}."
        low = log_text.lower()
        if "błąd konfiguracji" in low or exit_code == 2:
            m = re.search(r"Błąd konfiguracji: (.+)", log_text)
            if m:
                return f"Worker MLX odrzucił parametry uruchomienia: {m.group(1).strip()}"
        m = re.search(r"Błąd ładowania modelu: (.+)", log_text)
        if m:
            return m.group(1).strip()
        if "architektura tego modelu nie jest obsługiwana" in low or "model type" in low:
            return (
                "Zainstalowana wersja mlx-lm nie obsługuje architektury tego modelu – "
                "zaktualizuj ją („pip install -U mlx-lm”)."
            )
        if "config.json" in low:
            return "Katalog modelu nie zawiera kompletnych plików MLX (config.json / *.safetensors)."
        if "metal" in low and ("not available" in low or "unavailable" in low):
            return "MLX nie znalazł GPU Metal – modele MLX wymagają Maca z Apple Silicon."
        return None

    # ----------------------------------------------------------- after start
    async def _after_start(self) -> None:
        await super()._after_start()
        requested, actual = self.kv_bits, self.worker_kv_bits
        if requested is not None and actual != requested:
            log.warning(
                "engine %s: requested a %d-bit KV cache, the worker uses %s",
                self.instance_id, requested, actual or "f16",
            )
        log.info(
            "engine %s: MLX worker ready (batching=%s, kv_bits=%s, %s)",
            self.instance_id, self.batching, actual, self._props.get("build_info"),
        )

    # ------------------------------------------------------------ inference
    async def worker_memory(self) -> dict[str, object]:
        """``GET /memory``: MLX allocator statistics of the worker process."""
        data = await self._request_json("GET", "/memory", timeout=30)
        return data if isinstance(data, dict) else {}

    async def memory_bytes(self) -> int | None:
        """Bytes this instance holds (see module docstring)."""
        rss = await super().memory_bytes()
        try:
            stats = await self.worker_memory()
        except EngineError:
            return rss
        except Exception:  # noqa: BLE001 - diagnostics must never break a status call
            log.debug("engine %s: /memory failed", self.instance_id, exc_info=True)
            return rss
        mlx_bytes = 0
        for key in ("active", "cache"):
            value = stats.get(key)
            if isinstance(value, (int, float)) and value > 0:
                mlx_bytes += int(value)
        if not mlx_bytes:
            return rss
        return max(mlx_bytes, rss or 0)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EngineError(
            "Silnik MLX nie obsługuje embeddingów – użyj modelu GGUF do embeddingów "
            "(np. nomic-embed-text)."
        )
