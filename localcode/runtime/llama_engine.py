"""GGUF engine: Homebrew ``llama-server`` in a subprocess.

All process management and the HTTP/SSE client live in ``HttpEngine``; this
module only decides *how* llama-server is launched and how its failures are
explained. Decisions (verified against llama-server build 10964):

* **Per-slot context.** ``LoadConfig.n_ctx`` is the context each slot needs.
  We pass ``-c n_ctx*n_parallel -np n_parallel --no-kv-unified``: with split
  KV every slot owns exactly ``n_ctx`` tokens, so one long conversation can
  never starve a parallel request, and the KV size matches
  ``runtime.memory`` (``… · n_ctx · n_parallel``). ``--kv-unified`` would give
  every slot the full ``-c`` but let them compete for one pool;
  ``--kv-unified-per-slot N`` is equivalent to our split setup. llama.cpp
  rounds the per-slot size up to a multiple of 256 and caps it at the model's
  training context, so ``engine.n_ctx`` is read back from ``/props``.
* ``-ngl 999`` for "all layers" (works on every build), ``--fit off`` so the
  server never silently shrinks what the runtime manager budgeted,
  ``--no-ui``, ``--jinja``, ``--alias <instance id>``, ``--log-colors off``.
* **Prompt cache in RAM.** llama-server keeps idle slot states in a host-RAM
  prompt cache that defaults to 8 GiB — outside any budget. We size it to two
  full slot states (bounded 256 MiB … 4 GiB, 0 for embedding servers); see
  ``prompt_cache_mib``.
* **KV cache types.** ``f16|q8_0|q4_0`` (and the other llama.cpp types;
  MLX-style ``8bit``/``4bit`` are mapped). A quantized V cache requires flash
  attention: with ``flash_attn="off"`` V stays f16 (K is still quantized); with
  ``"auto"`` we keep V quantized and, if the server refuses ("quantized V cache
  requires flash_attn"), restart once with an f16 V cache.
* **Embeddings.** ``--embeddings`` with ``-b``/``-ub`` ≥ the per-slot context
  (each input must fit one physical batch); the model's own pooling is used
  unless ``extra_args`` has ``--pooling``.
* **Security.** llama-server allows any CORS origin; a random API key is
  passed through ``LLAMA_API_KEY`` (not argv, so it does not show in ``ps``)
  and every request but ``/health`` must carry it. Inherited ``LLAMA_ARG_*``
  variables are removed so the user's shell cannot silently change the setup.
* Optional flags are only passed when ``llama-server --help`` lists them, so
  older Homebrew builds keep working. ``extra_args`` come last and therefore
  override our defaults.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
import shutil
from pathlib import Path

from localcode.config import Settings, load_settings
from localcode.hub.types import LocalModel
from localcode.runtime.base import LoadConfig
from localcode.security.sandbox import SandboxPlan
from localcode.runtime.http_engine import EngineStartError, HttpEngine, _safe_name

__all__ = [
    "LlamaServerEngine",
    "find_llama_server",
    "llama_server_flags",
    "normalize_kv_cache_type",
    "normalize_flash_attn",
    "LLAMA_SERVER_CANDIDATES",
]

log = logging.getLogger(__name__)

LLAMA_SERVER_CANDIDATES: tuple[str, ...] = ("/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server")

KV_CACHE_TYPES = frozenset({"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"})
_KV_ALIASES = {
    "8bit": "q8_0", "8": "q8_0", "q8": "q8_0",
    "4bit": "q4_0", "4": "q4_0", "q4": "q4_0",
    "fp16": "f16", "float16": "f16", "half": "f16", "": "f16", "none": "f16", "default": "f16",
    "fp32": "f32", "float32": "f32",
}
# bytes per element (ggml block sizes: q8_0 = 34 B / 32, q4_0 = 18 B / 32, …)
_KV_BYTES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
    "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}
_UNQUANTIZED = frozenset({"f32", "f16", "bf16"})

PROMPT_CACHE_MIN_MIB = 256
PROMPT_CACHE_MAX_MIB = 4096
PROMPT_CACHE_DEFAULT_MIB = 1024

_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_FLAGS_CACHE: dict[tuple[str, float], frozenset[str]] = {}  # memo of `--help` parsing per binary


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_llama_server(settings: Settings | None = None) -> str | None:
    """Locate ``llama-server``: the configured path (file or directory), then
    ``PATH``, then the Homebrew prefixes. ``None`` when not installed."""
    s = settings if settings is not None else load_settings()
    configured = s.runtime.llama_server_path
    if configured:
        p = Path(configured).expanduser()
        if p.is_dir():
            p = p / "llama-server"
        if _is_executable(p):
            return str(p)
        log.warning("configured llama_server_path %r is not an executable; searching PATH", configured)
    found = shutil.which("llama-server")
    if found:
        return found
    for candidate in LLAMA_SERVER_CANDIDATES:
        if _is_executable(Path(candidate)):
            return candidate
    return None


async def llama_server_flags(binary: str) -> frozenset[str] | None:
    """Option names listed by ``<binary> --help`` (memoised per binary+mtime).
    ``None`` when the help text cannot be read (then every flag is assumed)."""
    try:
        real = os.path.realpath(binary)
        key = (real, os.path.getmtime(real))
    except OSError:
        return None
    cached = _FLAGS_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "--help",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return None
    flags = frozenset(_FLAG_RE.findall(out.decode("utf-8", errors="replace")))
    if "--port" not in flags:  # not a llama-server help text
        return None
    _FLAGS_CACHE[key] = flags
    return flags


def normalize_kv_cache_type(value: str | None) -> str:
    """llama.cpp cache type for a user/setting value (unknown → ``f16``)."""
    v = (value or "").strip().lower()
    v = _KV_ALIASES.get(v, v)
    if v not in KV_CACHE_TYPES:
        log.warning("unknown kv_cache_type %r, using f16", value)
        return "f16"
    return v


def normalize_flash_attn(value: object) -> str:
    """``on`` | ``off`` | ``auto`` from settings values (also bool-like strings)."""
    if isinstance(value, bool):
        return "on" if value else "off"
    v = str(value or "").strip().lower()
    if v in ("on", "true", "1", "yes", "enabled"):
        return "on"
    if v in ("off", "false", "0", "no", "disabled"):
        return "off"
    return "auto"


class LlamaServerEngine(HttpEngine):
    """One ``llama-server`` process serving a GGUF model (see module docstring)."""

    format = "gguf"
    # llama-server has no "exit when my parent dies" flag (and no way to add
    # one), so it is spawned under the supervisor shim from ``http_engine``.
    # Without it a `kill -9` of the core leaves a multi-GB server running
    # forever; ``mlx_worker`` solves the same problem with ``--parent-pid``.
    parent_guard = True

    def __init__(
        self,
        instance_id: str,
        model: LocalModel,
        config: LoadConfig,
        *,
        settings: Settings | None = None,
        binary: str | None = None,
    ) -> None:
        super().__init__(instance_id, model, config)
        self._settings = settings
        self._binary = binary
        self._flags: frozenset[str] | None = None
        self._v_f16_fallback = False
        self.command: list[str] = []   # argv of the last spawn (diagnostics)

    # ----------------------------------------------------------- properties
    @property
    def binary(self) -> str | None:
        return self._binary

    @property
    def prompt_cache_mib(self) -> int:
        """Host RAM llama-server may use for its idle-slot prompt cache."""
        return self._cache_ram_mib()

    # --------------------------------------------------------------- hooks
    def _log_name(self) -> str:
        return f"llama-{_safe_name(self.instance_id)}"

    def _env(self) -> dict[str, str]:
        env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("LLAMA_ARG_") and k not in ("LLAMA_API_KEY", "LLAMA_API_KEY_FILE")
        }
        if self._api_key:
            env["LLAMA_API_KEY"] = self._api_key
        return env

    async def _before_spawn(self) -> None:
        if self.model.format != "gguf":
            raise EngineStartError(
                f"Model „{self.model.display_name}” nie jest w formacie GGUF – llama-server go nie uruchomi."
            )
        binary = self._binary or find_llama_server(self._settings if self._settings is not None else load_settings())
        if not binary:
            raise EngineStartError(
                "Nie znaleziono programu llama-server. Zainstaluj go poleceniem „brew install llama.cpp” "
                "albo wskaż jego ścieżkę w ustawieniach (Runtime → ścieżka llama-server)."
            )
        self._binary = binary
        if not os.path.isfile(self.model.path):
            raise EngineStartError(f"Plik modelu nie istnieje: {self.model.path}")
        if self.model.mmproj_path and not os.path.isfile(self.model.mmproj_path):
            raise EngineStartError(f"Plik projektora wizji (mmproj) nie istnieje: {self.model.mmproj_path}")
        self._flags = await llama_server_flags(binary)
        self._api_key = secrets.token_urlsafe(24)

    def _has(self, flag: str) -> bool:
        return self._flags is None or flag in self._flags

    def _kv_types(self) -> tuple[str, str]:
        k = v = normalize_kv_cache_type(self.config.kv_cache_type)
        if v not in _UNQUANTIZED and (normalize_flash_attn(self.config.flash_attn) == "off" or self._v_f16_fallback):
            v = "f16"
        return k, v

    def _per_slot_ctx(self) -> int:
        return int(self.config.n_ctx) if self.config.n_ctx and self.config.n_ctx > 0 else 0

    def _cache_ram_mib(self) -> int:
        if self.config.embedding:
            return 0
        meta = self.model.meta
        per_slot = self._per_slot_ctx() or meta.context_length or 8192
        layers = meta.n_full_attn_layers or meta.n_layers
        kv_heads = meta.n_kv_heads or meta.n_heads
        head_dim = meta.head_dim
        if not head_dim and meta.hidden_size and meta.n_heads:
            head_dim = meta.hidden_size // meta.n_heads
        if not (layers and kv_heads and head_dim):
            return PROMPT_CACHE_DEFAULT_MIB
        k, v = self._kv_types()
        per_token = layers * kv_heads * head_dim * (_KV_BYTES[k] + _KV_BYTES[v])
        mib = 2 * per_token * per_slot / (1024 * 1024)
        return int(min(PROMPT_CACHE_MAX_MIB, max(PROMPT_CACHE_MIN_MIB, mib)))

    def _sandbox_plan(self) -> SandboxPlan | None:
        """``llama-server`` reads its weights and answers on localhost.

        It needs no write access of its own — the log is an inherited file
        descriptor, and the Metal shader cache is added by
        :meth:`SandboxPlan.with_defaults`.
        """
        read = [Path(self.model.path).expanduser().parent]
        if self.model.mmproj_path:
            read.append(Path(self.model.mmproj_path).expanduser().parent)
        if self._binary:
            read.append(Path(self._binary).expanduser().parent)
        return SandboxPlan(read=read, write=[])

    def _build_command(self, port: int) -> list[str]:
        if not self._binary:
            raise EngineStartError("Nie znaleziono programu llama-server.")
        cfg = self.config
        n_par = max(1, int(cfg.n_parallel or 1))
        per_slot = self._per_slot_ctx()
        cmd = [
            self._binary,
            "-m", self.model.path,
            "--host", self.host,
            "--port", str(port),
            "-c", str(per_slot * n_par),
            "-np", str(n_par),
            "-ngl", "999" if cfg.gpu_layers is None or cfg.gpu_layers < 0 else str(int(cfg.gpu_layers)),
            "-fa", normalize_flash_attn(cfg.flash_attn),
        ]
        if self._has("--jinja"):
            cmd.append("--jinja")
        if self._flags is None or "--no-ui" in self._flags:
            cmd.append("--no-ui")
        elif "--no-webui" in self._flags:
            cmd.append("--no-webui")
        if self._has("--alias"):
            cmd += ["--alias", self.instance_id]
        if self._has("--fit"):
            cmd += ["--fit", "off"]
        if self._has("--no-kv-unified"):
            cmd.append("--no-kv-unified")
        if self._has("--cache-ram"):
            cmd += ["--cache-ram", str(self._cache_ram_mib())]
        if self._has("--log-colors"):
            cmd += ["--log-colors", "off"]
        k, v = self._kv_types()
        if k != "f16":
            cmd += ["--cache-type-k", k]
        if v != "f16":
            cmd += ["--cache-type-v", v]
        if self.model.mmproj_path:
            cmd += ["--mmproj", self.model.mmproj_path]
        if cfg.embedding:
            batch = max(512, per_slot or self.model.meta.context_length or 8192)
            cmd += ["--embeddings", "-b", str(batch), "-ub", str(batch)]
        cmd += [str(a) for a in cfg.extra_args]
        self.command = list(cmd)
        return cmd

    def _on_start_failure(self, log_text: str, exit_code: int | None) -> bool:
        if "quantized V cache requires flash_attn" in log_text and not self._v_f16_fallback:
            _, v = self._kv_types()
            if v not in _UNQUANTIZED:
                log.warning(
                    "engine %s: flash attention unavailable, restarting with an f16 V cache", self.instance_id
                )
                self._v_f16_fallback = True
                return True
        return False

    def _failure_hint(self, log_text: str, exit_code: int | None) -> str | None:
        m = re.search(r"unknown model architecture: '([^']+)'", log_text)
        if m:
            return (
                f"Ta wersja llama.cpp nie obsługuje architektury „{m.group(1)}”. "
                "Zaktualizuj llama.cpp (brew upgrade llama.cpp)."
            )
        m = re.search(r"(?:invalid argument|unknown argument|error while handling argument)[^\n]*", log_text, re.I)
        if m:
            return (
                f"llama-server odrzucił parametry uruchomienia ({m.group(0).strip()}). "
                "Sprawdź dodatkowe argumenty w ustawieniach modelu."
            )
        low = log_text.lower()
        if self.model.mmproj_path and any(s in low for s in ("mmproj", "mtmd", "clip_model_load", "multimodal")) and (
            "failed" in low or "error" in low
        ):
            return "Nie udało się wczytać projektora wizji (mmproj) – plik może nie pasować do modelu."
        if any(s in low for s in ("out of memory", "failed to allocate", "insufficient memory", "outofmemory")):
            return None  # the generic memory hint applies
        if any(s in low for s in ("failed to load model", "gguf_init_from_file", "failed to read magic", "invalid magic")):
            return (
                "Nie udało się wczytać pliku modelu – plik może być uszkodzony, niekompletny "
                "albo w nieobsługiwanym formacie."
            )
        if "failed to create context" in low or "failed to create_context" in low:
            return (
                "Nie udało się utworzyć kontekstu modelu – zwykle oznacza to brak pamięci dla wybranego "
                "kontekstu i liczby slotów albo niezgodne ustawienia (typ cache KV, flash attention)."
            )
        return None
