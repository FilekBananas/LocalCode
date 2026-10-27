"""Memory estimation for local model instances and system memory probing.

Why this module exists
======================
LocalCode lets the user decide how many model instances may run, how many
parallel requests each instance serves and how much RAM all of them may take
(LM Studio-like guardrails).  Every one of those decisions needs a *reliable
up-front estimate* of what an instance will cost, because on Apple Silicon the
GPU shares the system RAM: over-committing does not fail cleanly, it makes the
whole Mac swap.  The estimate is also what the UI shows next to every
downloadable variant ("Zmieści się: ~5,2 GB …") and what ``auto_ctx`` uses to
choose the largest context window that fits.

Model of an instance
====================
``total = weights + kv_cache + overhead``

* **weights** – bytes of the weight files, read from disk (GGUF: all shards of
  the model; MLX: sum of ``*.safetensors``), falling back to
  ``LocalModel.size_bytes`` for models that are not on disk (hub variants).
  llama.cpp memory-maps the GGUF and Metal wraps the mapping without copying,
  MLX copies the safetensors into Metal buffers: either way ≈ file size.
* **kv_cache** – ``2 · layers · n_kv_heads · head_dim · bytes(type)`` per token
  and per sequence, multiplied by ``n_ctx · n_parallel`` (llama-server is
  started with ``-c n_ctx·n_parallel -np n_parallel``; MLX keeps one cache per
  sequence).  Refinements:

  - ``head_dim`` falls back to ``hidden_size / n_heads`` (then 128).
  - sliding-window (SWA) layers keep at most ``window + n_ubatch`` tokens per
    sequence (llama.cpp's iSWA cache; MLX's ``RotatingKVCache``) unless
    ``--swa-full`` is passed in ``LoadConfig.extra_args``.  Which layers are
    SWA comes from ``ModelMeta.n_full_attn_layers`` or, when absent, from the
    well-known per-architecture patterns (gemma2/gpt-oss 1:1, gemma3 5:1, …);
    unknown architectures with a window are treated as full attention
    (conservative).
  - hybrid / linear-attention layers (``n_full_attn_layers`` set without a
    window: Qwen3-Next, Granite-H, Jamba, LFM2 …) have no growing KV cache;
    they get a constant recurrent-state allowance instead.
  - encoder-only embedding models (BERT family) have no KV cache at all.
  - bytes per element: f16/bf16 = 2, q8_0/8bit = 34/32 = 1.0625,
    q4_0/4bit = 18/32 = 0.5625 (MLX groups of 64 with bf16 scale+bias give
    the same ratios), unknown types count as f16.

* **overhead** – everything else, calibrated per backend (see below).

Calibration (Apple M5, 32 GB, macOS 26, llama-server b10964, mlx 0.32.2 /
mlx-lm 0.31.3, Qwen3 test models, September 2026)
------------------------------------------------------------------------------
llama-server (``-lv 4`` prints every buffer; RSS measured after a 2k-token
prompt; weights appear in RSS but not in ``phys_footprint`` because they are a
clean file mapping):

=========================  =====  =====  ========  ==========  ==========  =====
config                     model  KV     compute   est. total  measured    error
=========================  =====  =====  ========  ==========  ==========  =====
0.6B Q4_K_M, 4096×1, FA    373    448    58        989         970 RSS     +2 %
0.6B Q4_K_M, 32768×1, FA   373    3584   226       4293        4104 RSS    +5 %
0.6B Q4_K_M, 8192×4, FA    373    3584   209       4294        4109 RSS    +4 %
0.6B Q4_K_M, 32768, no FA  373    3584   1180      5189        5137 bufs   +1 %
0.6B, 16384, q8_0 KV       373    952    122       1565        1478 RSS    +6 %
4B Q4_K_M, 4096×1, FA      2376   576    90        3150        3090 RSS    +2 %
4B Q4_K_M, 4096×2, FA      2376   1152   125       3750        3680 RSS    +2 %
4B Q4_K_M, 16384×1, FA     2376   2304   162       4950        4760 RSS    +4 %
nomic-embed Q8_0, 2048     139    0      25        286         281 RSS     +2 %
=========================  =====  =====  ========  ==========  ==========  =====

(all MiB).  The KV formula matches llama.cpp's own ``llama_kv_cache: size``
line exactly.  The Metal+CPU compute buffers fit ``~32 MiB + 1.5 % of the
weights + ~6 KiB per context token`` with flash attention (the default
``auto`` enables it on Metal) and ``n_heads · n_ubatch · 4 B`` per context
token without it (the materialised KQ matrix).  The output buffer is
``vocab · 4 B`` per sequence and the process itself (binary, Metal context,
shader library) ~100 MiB.

MLX (``mx.get_peak_memory`` during prefill + process ``phys_footprint``):

=============================  =====  =====  ==========  =========  =====
config                         model  KV     est. total  peak+proc  error
=============================  =====  =====  ==========  =========  =====
0.6B 4bit, 2000 tokens         320    219    1435        1374       +4 %
0.6B 4bit, 32000 tokens        320    3500   4717        4600       +3 %
4B 4bit, 16000 tokens          2158   2250   5323        5199       +2 %
0.6B 4bit, 16000 tok, 8bit KV  320    930    4396        4400        0 %
=============================  =====  =====  ==========  =========  =====

The Python worker (interpreter, MLX, tokenizer, HTTP server) costs ~0.3 GB.
Prefill needs a transient ~0.5–0.66 GB scratch for a 152k vocabulary that did
not scale with model size or with the batch size of ``BatchGenerator`` (1 vs 4
prompts), so it is modelled as ``vocab · 4 KiB`` (≥ 256 MiB).  Quantized KV
caches in mlx-lm go through ``quantized_scaled_dot_product_attention`` which
materialises the attention scores of a whole prefill chunk
(``n_heads · 2048 · n_ctx · ~4.5 B``: fp32 scores plus the softmax copy,
measured 2.3 GB at 16k context for the 0.6B model).  That term is added for
MLX 8bit/4bit KV, which is why quantizing the MLX cache only pays off for
short prompts (a worker using a smaller prefill step for quantized caches
should lower ``MLX_PREFILL_STEP``).

Estimates are therefore slightly conservative (0…+6 %), which is the right
side to err on for a RAM budget.  Not modelled: llama-server's host-RAM prompt
cache (``--cache-ram``, 8 GiB by default!) unless the value is passed in
``LoadConfig.extra_args``, and context checkpoints of SWA/hybrid models.

System memory
=============
``system_memory()`` mirrors Activity Monitor on macOS: *used* = app memory
(anonymous − purgeable) + wired + compressor, *available* = total − used, so
file cache, purgeable and speculative pages count as reclaimable.  It is read
with ``host_statistics64`` through ctypes (microseconds, no subprocess) and
falls back to ``psutil.virtual_memory()`` elsewhere.
"""

from __future__ import annotations

import ctypes
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from localcode.hub.types import LocalModel, MemoryEstimate, ModelMeta
from localcode.runtime.base import LoadConfig

try:  # psutil is a declared dependency; keep the module importable without it
    import psutil
except ImportError:  # pragma: no cover - exercised only on broken installs
    psutil = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

KiB = 1024
MiB = 1024 * KiB
GiB = 1024 * MiB

# --------------------------------------------------------------------------- #
# Tunables (calibrated, see the module docstring)
# --------------------------------------------------------------------------- #

AUTO_BUDGET_FRACTION = 0.70        # ram_limit_gb=None -> 70 % of physical RAM
PHYSICAL_HARD_FRACTION = 0.90      # a single instance above this never fits
TIGHT_FRACTION = 0.90              # > 90 % of the remaining budget -> "tight"

DEFAULT_NATIVE_CTX = 8192          # unknown model context length
DEFAULT_CTX_CAP = 32768            # RuntimeSettings.default_ctx=None cap
MIN_USEFUL_CTX = 2048              # auto_ctx never goes below this (if the model allows)
CTX_GRANULARITY = 1024

DEFAULT_VOCAB = 152_000            # unknown vocabulary size (Qwen-sized, conservative)
DEFAULT_N_HEADS = 32
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128

GGUF_PROCESS_BASE = 100 * MiB      # llama-server binary, Metal context, shader library
GGUF_COMPUTE_BASE = 32 * MiB
GGUF_COMPUTE_WEIGHTS_FRACTION = 0.015
GGUF_FA_BYTES_PER_CTX_TOKEN = 6 * KiB   # at n_ubatch=512, flash attention on
GGUF_DEFAULT_UBATCH = 512
GGUF_KV_PAD = 256                  # llama.cpp pads the KV cell count
VISION_COMPUTE_BYTES = 256 * MiB   # mmproj (CLIP) compute buffers, rough

MLX_PROCESS_BASE = 300 * MiB       # python + mlx + tokenizer + aiohttp worker
MLX_SCRATCH_PER_VOCAB = 4 * KiB    # prefill scratch ≈ vocab · 4 KiB (0.58 GiB @ 152k)
MLX_MIN_SCRATCH = 256 * MiB
MLX_WEIGHTS_FRACTION = 0.01
MLX_PREFILL_STEP = 2048            # mlx-lm default prefill_step_size
MLX_SCORE_BYTES = 4.5              # quantized-KV attention: fp32 scores + softmax copy (measured)

RECURRENT_STATE_PER_LAYER = 4 * MiB  # per sequence, linear-attention / SSM layers

# llama.cpp SWA pattern: every Nth layer keeps full attention (hparams.set_swa_pattern)
SWA_FULL_EVERY: dict[str, int] = {
    "gemma2": 2,
    "gemma3": 6,
    "gemma3n": 5,
    "gemma_embedding": 6,
    "cohere2": 4,
    "gpt_oss": 2,
    "llama4": 4,
    "exaone4": 4,
}

# Encoder-only (non-causal) embedding architectures: llama.cpp allocates no KV cache.
ENCODER_ARCHS = frozenset({
    "bert", "nomic_bert", "nomic_bert_moe", "jina_bert_v2", "jina_bert_v3",
    "modern_bert", "neo_bert", "eurobert", "t5encoder", "roberta", "xlm_roberta",
})

_KV_ELEMENT_BYTES: dict[str, float] = {
    "f32": 4.0, "fp32": 4.0, "float32": 4.0,
    "f16": 2.0, "fp16": 2.0, "float16": 2.0,
    "bf16": 2.0, "bfloat16": 2.0,
    "q8_0": 34 / 32, "8bit": 68 / 64,
    "q6_k": 210 / 256, "6bit": 52 / 64,
    "q5_0": 22 / 32, "q5_1": 24 / 32, "5bit": 44 / 64,
    "q4_0": 18 / 32, "q4_1": 20 / 32, "iq4_nl": 18 / 32, "4bit": 36 / 64,
    "3bit": 28 / 64, "2bit": 20 / 64,
}


# --------------------------------------------------------------------------- #
# Formatting helpers (Polish UI)
# --------------------------------------------------------------------------- #


def format_gb(n_bytes: float) -> str:
    """``5.23 GiB -> "5,2 GB"`` (binary GB, like Activity Monitor; Polish comma)."""
    gb = max(0.0, float(n_bytes)) / GiB
    if 0 < gb < 0.1:
        return "<0,1 GB"
    if gb >= 100:
        return f"{gb:.0f} GB"
    return f"{gb:.1f} GB".replace(".", ",")


def format_tokens(n: int) -> str:
    """``16384 -> "16 384"`` (Polish thousands separator)."""
    return f"{int(n):,}".replace(",", " ")


# --------------------------------------------------------------------------- #
# System / process memory
# --------------------------------------------------------------------------- #


class _VMStatistics64(ctypes.Structure):
    """``struct vm_statistics64`` (mach/vm_statistics.h, rev1 layout)."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),        # includes speculative pages
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),  # file-backed
        ("internal_page_count", ctypes.c_uint32),  # anonymous
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _RUsageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` (sys/resource.h)."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
        )
    ]


_HOST_VM_INFO64 = 4
_RUSAGE_INFO_V2 = 2


class _Darwin:
    """Lazily bound libSystem entry points (None when unavailable)."""

    _lib: Any = None
    _host: int | None = None
    _page_size: int | None = None
    _failed = False

    @classmethod
    def lib(cls) -> Any:
        if cls._lib is None and not cls._failed:
            try:
                lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                lib.mach_host_self.restype = ctypes.c_uint32
                lib.mach_host_self.argtypes = []
                lib.host_statistics64.restype = ctypes.c_int
                lib.host_statistics64.argtypes = [
                    ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                lib.host_page_size.restype = ctypes.c_int
                lib.host_page_size.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_size_t)]
                lib.proc_pid_rusage.restype = ctypes.c_int
                lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                cls._lib = lib
            except (OSError, AttributeError) as exc:
                log.debug("libSystem unavailable: %s", exc)
                cls._failed = True
        return cls._lib

    @classmethod
    def host(cls) -> tuple[Any, int, int] | None:
        lib = cls.lib()
        if lib is None:
            return None
        if cls._host is None:
            host = int(lib.mach_host_self())
            size = ctypes.c_size_t()
            if lib.host_page_size(host, ctypes.byref(size)) != 0 or size.value <= 0:
                cls._failed = True
                return None
            cls._host, cls._page_size = host, int(size.value)
        assert cls._page_size is not None
        return lib, cls._host, cls._page_size


def _physical_total() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().total)
    try:  # pragma: no cover - psutil is always installed in practice
        import os

        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):  # pragma: no cover
        return 16 * GiB


def _darwin_vm_statistics() -> tuple[_VMStatistics64, int] | None:
    if sys.platform != "darwin":
        return None
    bound = _Darwin.host()
    if bound is None:
        return None
    lib, host, page_size = bound
    stats = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(_VMStatistics64) // 4)
    if lib.host_statistics64(host, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)) != 0:
        return None
    return stats, page_size


def system_memory() -> dict[str, int]:
    """Physical memory snapshot in bytes: ``{"total", "available", "used"}``.

    ``used + available == total``.  On macOS *used* is Activity Monitor's
    "Memory Used" (app + wired + compressed) and *available* everything the
    kernel can reclaim without swapping (file cache, purgeable, speculative).
    """
    total = _physical_total()
    darwin = _darwin_vm_statistics()
    if darwin is not None:
        s, page = darwin
        app = max(0, int(s.internal_page_count) - int(s.purgeable_count))
        used = (app + int(s.wire_count) + int(s.compressor_page_count)) * page
        used = min(max(used, 0), total)
        return {"total": total, "available": total - used, "used": used}
    if psutil is not None:
        vm = psutil.virtual_memory()
        available = int(min(vm.available, total))
        return {"total": total, "available": available, "used": total - available}
    return {"total": total, "available": total // 2, "used": total - total // 2}  # pragma: no cover


def process_memory_bytes(pid: int | None) -> int | None:
    """Best measurement of what a model process really occupies.

    ``max(phys_footprint, rss)``: MLX allocations live in Metal buffers that
    only ``phys_footprint`` sees (RSS drops to ~50 MB), while llama-server's
    memory-mapped weights are clean file pages that only RSS sees.  Returns
    None when the process does not exist or cannot be inspected.
    """
    if not pid or pid <= 0:
        return None
    values: list[int] = []
    if sys.platform == "darwin":
        lib = _Darwin.lib()
        if lib is not None:
            info = _RUsageInfoV2()
            if lib.proc_pid_rusage(int(pid), _RUSAGE_INFO_V2, ctypes.byref(info)) == 0:
                values.append(int(info.ri_phys_footprint))
    if psutil is not None:
        try:
            values.append(int(psutil.Process(int(pid)).memory_info().rss))
        except (psutil.Error, OSError, ValueError):
            pass
    values = [v for v in values if v > 0]
    return max(values) if values else None


def default_ram_budget(settings: Any) -> int:
    """RAM budget for all instances in bytes.

    Accepts ``Settings`` or ``RuntimeSettings``.  ``ram_limit_gb`` (binary GB)
    wins, clamped to physical RAM; ``None`` means 70 % of physical RAM (which
    also keeps us under Metal's recommended working set of ~75-78 %).
    """
    runtime = getattr(settings, "runtime", settings)
    total = system_memory()["total"]
    limit = getattr(runtime, "ram_limit_gb", None)
    if limit is not None:
        try:
            value = float(limit)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return int(min(value * GiB, total))
    return int(total * AUTO_BUDGET_FRACTION)


# --------------------------------------------------------------------------- #
# KV cache math
# --------------------------------------------------------------------------- #


def normalize_arch(arch: str | None) -> str:
    """``"gemma3_text"``/``"gpt-oss"`` -> ``"gemma3"``/``"gpt_oss"``."""
    a = (arch or "").strip().lower().replace("-", "_")
    for suffix in ("_text", "forcausallm"):
        if a.endswith(suffix):
            a = a[: -len(suffix)]
    return a


def normalize_kv_type(kv_cache_type: str | None, fmt: str) -> str:
    """Map the global setting onto the engine's vocabulary.

    GGUF understands ``f16|bf16|q8_0|q4_0|…``, MLX ``f16|8bit|4bit``; the
    runtime setting may hold either spelling, so ``q8_0 <-> 8bit`` and
    ``q4_0 <-> 4bit`` are translated for the target format.
    """
    kv = (kv_cache_type or "f16").strip().lower()
    if fmt == "mlx":
        mapping = {"q8_0": "8bit", "q4_0": "4bit", "q4_1": "4bit", "iq4_nl": "4bit",
                   "q5_0": "8bit", "q5_1": "8bit", "q6_k": "8bit",
                   "bf16": "f16", "fp16": "f16", "float16": "f16", "bfloat16": "f16"}
        kv = mapping.get(kv, kv)
        return kv if kv in ("f16", "8bit", "4bit") else "f16"
    mapping = {"8bit": "q8_0", "4bit": "q4_0", "6bit": "q8_0", "5bit": "q8_0",
               "fp16": "f16", "float16": "f16", "bfloat16": "bf16"}
    kv = mapping.get(kv, kv)
    return kv if kv in _KV_ELEMENT_BYTES else "f16"


def kv_element_bytes(kv_cache_type: str | None) -> float:
    """Bytes per cached K or V element (quant blocks included)."""
    return _KV_ELEMENT_BYTES.get((kv_cache_type or "f16").strip().lower(), 2.0)


@dataclass(frozen=True)
class AttentionLayout:
    """How the layers of a model use the KV cache."""

    n_layers: int
    full_layers: int        # KV grows with the whole context
    swa_layers: int         # KV capped at ``window``
    linear_layers: int      # recurrent / linear attention: constant state
    window: int | None
    n_kv_heads: int
    head_dim: int


def _default_layers(meta: ModelMeta, weights_bytes: int | None) -> int:
    params = meta.params_b
    if not params and weights_bytes:
        params = weights_bytes / 0.6e9      # ~4.8 bits per weight on average
    if not params:
        return 36
    for limit, layers in ((1.0, 28), (5.0, 36), (10.0, 40), (20.0, 48), (40.0, 64)):
        if params < limit:
            return layers
    return 80


def attention_layout(meta: ModelMeta | None, *, weights_bytes: int | None = None) -> AttentionLayout:
    """Resolve layer counts, KV heads and head size with conservative defaults."""
    meta = meta or ModelMeta()
    n_layers = meta.n_layers if meta.n_layers and meta.n_layers > 0 else _default_layers(meta, weights_bytes)
    n_heads = meta.n_heads if meta.n_heads and meta.n_heads > 0 else None
    n_kv = meta.n_kv_heads if meta.n_kv_heads and meta.n_kv_heads > 0 else (n_heads or DEFAULT_KV_HEADS)
    if meta.head_dim and meta.head_dim > 0:
        head_dim = meta.head_dim
    elif meta.hidden_size and n_heads:
        head_dim = max(1, meta.hidden_size // n_heads)
    else:
        head_dim = DEFAULT_HEAD_DIM

    window = meta.sliding_window if meta.sliding_window and meta.sliding_window > 0 else None
    n_full = meta.n_full_attn_layers
    if n_full is not None:
        n_full = max(0, min(int(n_full), n_layers))

    if window is not None:
        if n_full is None:
            every = SWA_FULL_EVERY.get(normalize_arch(meta.arch))
            if every is None:                       # unknown SWA layout: assume full attention
                return AttentionLayout(n_layers, n_layers, 0, 0, window, n_kv, head_dim)
            n_full = max(1, n_layers // every)
        return AttentionLayout(n_layers, n_full, n_layers - n_full, 0, window, n_kv, head_dim)
    if n_full is not None:                          # hybrid: the rest is linear attention / SSM
        return AttentionLayout(n_layers, n_full, 0, n_layers - n_full, None, n_kv, head_dim)
    return AttentionLayout(n_layers, n_layers, 0, 0, None, n_kv, head_dim)


def kv_bytes_per_token(
    meta: ModelMeta | None,
    kv_cache_type: str | None = "f16",
    fmt: str = "gguf",
    *,
    swa_full: bool = False,
    weights_bytes: int | None = None,
) -> float:
    """KV bytes added by one more token of context for ONE sequence.

    Counts only the layers whose cache grows with the context: full-attention
    layers (plus SWA layers when ``swa_full``); SWA layers are capped at their
    window and linear-attention layers keep a constant state, see
    :func:`kv_cache_bytes` for the complete size.
    """
    layout = attention_layout(meta, weights_bytes=weights_bytes)
    layers = layout.full_layers + (layout.swa_layers if swa_full else 0)
    elem = kv_element_bytes(normalize_kv_type(kv_cache_type, fmt))
    return 2.0 * layers * layout.n_kv_heads * layout.head_dim * elem


def kv_cache_bytes(
    meta: ModelMeta | None,
    n_ctx: int,
    n_parallel: int = 1,
    kv_cache_type: str | None = "f16",
    fmt: str = "gguf",
    *,
    swa_full: bool = False,
    weights_bytes: int | None = None,
    n_ubatch: int = GGUF_DEFAULT_UBATCH,
) -> int:
    """Total KV cache bytes for ``n_parallel`` sequences of ``n_ctx`` tokens."""
    n_ctx = max(0, int(n_ctx))
    n_parallel = max(1, int(n_parallel))
    layout = attention_layout(meta, weights_bytes=weights_bytes)
    elem = kv_element_bytes(normalize_kv_type(kv_cache_type, fmt))
    per_layer_token = 2.0 * layout.n_kv_heads * layout.head_dim * elem
    full_tokens = n_ctx * n_parallel
    if fmt == "gguf" and full_tokens:
        full_tokens = int(math.ceil(full_tokens / GGUF_KV_PAD) * GGUF_KV_PAD)
    total = layout.full_layers * per_layer_token * full_tokens
    if layout.swa_layers:
        if swa_full or layout.window is None:
            swa_tokens = full_tokens
        else:
            cap = layout.window + (n_ubatch if fmt == "gguf" else 0)
            swa_tokens = min(n_ctx, cap) * n_parallel
        total += layout.swa_layers * per_layer_token * swa_tokens
    return int(total)


# --------------------------------------------------------------------------- #
# Component estimates
# --------------------------------------------------------------------------- #

_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$", re.IGNORECASE)


def model_weights_bytes(model: LocalModel) -> int:
    """Bytes of the weights as they will be loaded (all shards / safetensors)."""
    try:
        path = Path(model.path).expanduser() if model.path else None
        if path is not None:
            if model.format == "mlx" and path.is_dir():
                total = sum(f.stat().st_size for f in path.glob("*.safetensors") if f.is_file())
                if total > 0:
                    return total
            elif model.format == "gguf" and path.is_file():
                m = _SHARD_RE.match(path.name)
                if m is None:
                    return path.stat().st_size
                n = int(m.group("n"))
                total = 0
                for i in range(1, n + 1):
                    shard = path.with_name(f"{m.group('stem')}-{i:05d}-of-{m.group('n')}.gguf")
                    if shard.is_file():
                        total += shard.stat().st_size
                if total > 0:
                    return total
    except OSError as exc:
        log.debug("cannot stat weights of %s: %s", model.key, exc)
    return max(0, int(model.size_bytes or 0))


def _mmproj_bytes(model: LocalModel) -> int:
    if model.format != "gguf" or not model.mmproj_path:
        return 0
    try:
        return Path(model.mmproj_path).expanduser().stat().st_size
    except OSError:
        return 0


def _arg_value(args: Iterable[str], *names: str) -> str | None:
    items = list(args or [])
    for i, item in enumerate(items):
        for name in names:
            if item == name and i + 1 < len(items):
                return items[i + 1]
            if item.startswith(name + "="):
                return item.split("=", 1)[1]
    return None


def _is_encoder_only(model: LocalModel, config: LoadConfig) -> bool:
    arch = normalize_arch(model.meta.arch if model.meta else None)
    return (config.embedding or model.is_embedding) and arch in ENCODER_ARCHS


@dataclass(frozen=True)
class Breakdown:
    """Estimated bytes of one instance, split by purpose."""

    weights: int
    kv_cache: int
    overhead: int

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.overhead


def _overhead_gguf(model: LocalModel, meta: ModelMeta, config: LoadConfig, weights: int,
                   n_ctx: int, n_parallel: int, linear_layers: int) -> int:
    extra = config.extra_args or []
    try:
        ubatch = int(_arg_value(extra, "-ub", "--ubatch-size") or GGUF_DEFAULT_UBATCH)
    except ValueError:
        ubatch = GGUF_DEFAULT_UBATCH
    ubatch = max(1, ubatch)
    ctx_total = n_ctx * n_parallel
    if (config.flash_attn or "auto").lower() == "off":
        heads = meta.n_heads or DEFAULT_N_HEADS
        per_token = heads * ubatch * 4 + 2 * KiB * ubatch / GGUF_DEFAULT_UBATCH
    else:
        per_token = GGUF_FA_BYTES_PER_CTX_TOKEN * ubatch / GGUF_DEFAULT_UBATCH
    compute = GGUF_COMPUTE_BASE + GGUF_COMPUTE_WEIGHTS_FRACTION * weights + per_token * ctx_total
    output = (meta.vocab_size or DEFAULT_VOCAB) * 4 * n_parallel
    total = GGUF_PROCESS_BASE + compute + output
    mmproj = _mmproj_bytes(model)
    if mmproj:
        total += mmproj + VISION_COMPUTE_BYTES
    try:
        cache_ram_mib = int(_arg_value(extra, "-cram", "--cache-ram") or 0)
    except ValueError:
        cache_ram_mib = 0
    if cache_ram_mib < 0:              # -1 = unlimited: count llama.cpp's default limit
        cache_ram_mib = 8192
    total += cache_ram_mib * MiB
    total += linear_layers * RECURRENT_STATE_PER_LAYER * n_parallel
    return int(total)


def _overhead_mlx(meta: ModelMeta, kv_type: str, weights: int, n_ctx: int, n_parallel: int,
                  linear_layers: int) -> int:
    vocab = meta.vocab_size or DEFAULT_VOCAB
    scratch = max(MLX_MIN_SCRATCH, vocab * MLX_SCRATCH_PER_VOCAB)
    total = MLX_PROCESS_BASE + scratch + MLX_WEIGHTS_FRACTION * weights
    if kv_type in ("8bit", "4bit"):
        heads = meta.n_heads or DEFAULT_N_HEADS
        total += heads * min(n_ctx, MLX_PREFILL_STEP) * n_ctx * MLX_SCORE_BYTES
    total += linear_layers * RECURRENT_STATE_PER_LAYER * n_parallel
    return int(total)


def breakdown(model: LocalModel, config: LoadConfig, *, weights_bytes: int | None = None) -> Breakdown:
    """Weights / KV cache / overhead bytes for ``model`` loaded with ``config``."""
    meta = model.meta or ModelMeta()
    fmt = model.format
    weights = model_weights_bytes(model) if weights_bytes is None else weights_bytes
    n_ctx = max(1, int(config.n_ctx or 1))
    n_parallel = max(1, int(config.n_parallel or 1))
    kv_type = normalize_kv_type(config.kv_cache_type, fmt)
    swa_full = "--swa-full" in (config.extra_args or [])
    layout = attention_layout(meta, weights_bytes=weights)
    if _is_encoder_only(model, config):
        kv = 0
    else:
        try:
            ubatch = int(_arg_value(config.extra_args or [], "-ub", "--ubatch-size") or GGUF_DEFAULT_UBATCH)
        except ValueError:
            ubatch = GGUF_DEFAULT_UBATCH
        kv = kv_cache_bytes(meta, n_ctx, n_parallel, kv_type, fmt, swa_full=swa_full,
                            weights_bytes=weights, n_ubatch=ubatch)
    if fmt == "mlx":
        overhead = _overhead_mlx(meta, kv_type, weights, n_ctx, n_parallel, layout.linear_layers)
    else:
        overhead = _overhead_gguf(model, meta, config, weights, n_ctx, n_parallel, layout.linear_layers)
    return Breakdown(weights=int(weights), kv_cache=int(kv), overhead=int(overhead))


def _with_ctx(config: LoadConfig, n_ctx: int, n_parallel: int | None = None) -> LoadConfig:
    return LoadConfig(
        n_ctx=n_ctx,
        n_parallel=config.n_parallel if n_parallel is None else n_parallel,
        gpu_layers=config.gpu_layers,
        flash_attn=config.flash_attn,
        kv_cache_type=config.kv_cache_type,
        embedding=config.embedding,
        extra_args=list(config.extra_args or []),
    )


# --------------------------------------------------------------------------- #
# Public estimate API
# --------------------------------------------------------------------------- #


def native_context(model: LocalModel) -> int:
    """Trained context length of the model, or a conservative default."""
    ctx = model.meta.context_length if model.meta else None
    return int(ctx) if ctx and ctx > 0 else DEFAULT_NATIVE_CTX


def _max_ctx_for(model: LocalModel, config: LoadConfig, budget_bytes: int, *, cap: int | None = None) -> int:
    native = native_context(model)
    upper = min(native, cap) if cap else native
    weights = model_weights_bytes(model)

    def fits(ctx: int) -> bool:
        return breakdown(model, _with_ctx(config, ctx), weights_bytes=weights).total <= budget_bytes

    if upper <= 0:
        return 0
    if fits(upper):
        return upper
    lo, hi = 0, upper // CTX_GRANULARITY       # search k·1024, k in [1, hi]
    if hi >= 1 and upper % CTX_GRANULARITY == 0:
        hi -= 1                                # `upper` itself already failed
    best = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid >= 1 and fits(mid * CTX_GRANULARITY):
            lo = mid
        else:
            hi = mid - 1
    if lo >= 1:
        best = lo * CTX_GRANULARITY
    return best


def max_ctx_that_fits(
    model: LocalModel,
    budget_bytes: int,
    n_parallel: int = 1,
    kv_cache_type: str = "f16",
    *,
    flash_attn: str = "auto",
    embedding: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Largest per-slot context whose whole instance fits in ``budget_bytes``.

    Returns the native context length when it fits entirely, otherwise the
    largest multiple of 1024 that fits, or 0 when not even 1024 tokens fit.
    """
    config = LoadConfig(n_ctx=CTX_GRANULARITY, n_parallel=max(1, int(n_parallel)),
                        flash_attn=flash_attn, kv_cache_type=kv_cache_type,
                        embedding=embedding, extra_args=list(extra_args or []))
    return _max_ctx_for(model, config, int(budget_bytes))


def auto_ctx(
    model: LocalModel,
    settings: Any,
    budget_left: int | None = None,
    *,
    n_parallel: int | None = None,
    kv_cache_type: str | None = None,
    flash_attn: str | None = None,
    embedding: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """Context per slot to use when the user did not pick one.

    ``min(native context or 8192, settings.default_ctx or 32768, what fits in
    budget_left)``, but never below 2048 when the model supports that much –
    if even that does not fit, the admission check reports it properly.
    ``settings`` may be ``Settings`` or ``RuntimeSettings``.
    """
    runtime = getattr(settings, "runtime", settings)
    native = native_context(model)
    cap = getattr(runtime, "default_ctx", None) or DEFAULT_CTX_CAP
    want = max(1, min(native, int(cap)))
    if budget_left is None:
        budget_left = default_ram_budget(runtime)
    config = LoadConfig(
        n_ctx=want,
        n_parallel=max(1, int(n_parallel or getattr(runtime, "default_parallel", 1) or 1)),
        flash_attn=flash_attn or getattr(runtime, "flash_attn", "auto"),
        kv_cache_type=kv_cache_type or getattr(runtime, "kv_cache_type", "f16"),
        embedding=embedding,
        extra_args=list(extra_args or []),
    )
    fit = _max_ctx_for(model, config, int(budget_left), cap=want)
    ctx = min(want, fit) if fit > 0 else 0
    return max(ctx, min(MIN_USEFUL_CTX, native))


def estimate(
    model: LocalModel,
    config: LoadConfig,
    *,
    budget: int | None = None,
    in_use: int = 0,
    reclaimable: int = 0,
) -> MemoryEstimate:
    """Estimate one instance and judge whether it fits.

    ``budget``: RAM budget in bytes (default: 70 % of physical RAM);
    ``in_use``: bytes already taken inside the budget by instances that stay
    loaded; ``reclaimable``: bytes that would be freed first (instances to be
    evicted) and therefore count as available system memory.

    ``fits``: ``"no"`` when the instance exceeds the remaining budget or 90 %
    of physical RAM; ``"tight"`` when it uses more than 90 % of the remaining
    budget or more than the memory currently available (macOS would have to
    compress/swap other apps); ``"yes"`` otherwise.
    """
    parts = breakdown(model, config)
    total = parts.total
    sysmem = system_memory()
    physical = sysmem["total"]
    if budget is None:
        budget = int(physical * AUTO_BUDGET_FRACTION)
    room = max(0, int(budget) - max(0, int(in_use)))
    available = sysmem["available"] + max(0, int(reclaimable))
    n_ctx = max(1, int(config.n_ctx or 1))
    n_parallel = max(1, int(config.n_parallel or 1))

    detail = (f"wagi {format_gb(parts.weights)} + kontekst {format_gb(parts.kv_cache)}"
              f" + bufory {format_gb(parts.overhead)}")
    if total > physical * PHYSICAL_HARD_FRACTION:
        fits = "no"
        message = (f"Nie zmieści się: potrzeba ~{format_gb(total)} ({detail}), a ten Mac ma "
                   f"{format_gb(physical)} RAM. {_suggestions(model, config, room, parts)}")
    elif total > room:
        fits = "no"
        limit_info = f"w limicie RAM zostało {format_gb(room)}"
        if in_use > 0:
            limit_info += f" z {format_gb(budget)} (reszta zajęta przez inne modele)"
        else:
            limit_info += f" (limit {format_gb(budget)})"
        message = (f"Nie zmieści się: potrzeba ~{format_gb(total)} ({detail}), a {limit_info}. "
                   f"{_suggestions(model, config, room, parts)}")
    elif total > available:
        fits = "tight"
        message = (f"Na styk: ~{format_gb(total)} ({detail}). Wolne jest teraz tylko "
                   f"~{format_gb(available)} – macOS zwolni pamięć innych aplikacji, co może "
                   f"spowolnić komputer.")
    elif total > room * TIGHT_FRACTION:
        fits = "tight"
        message = (f"Na styk: ~{format_gb(total)} ({detail}) – to prawie cały dostępny limit RAM "
                   f"({format_gb(room)}).")
    else:
        fits = "yes"
        message = f"Zmieści się: ~{format_gb(total)} ({detail})."
    return MemoryEstimate(
        weights_bytes=parts.weights,
        kv_cache_bytes=parts.kv_cache,
        overhead_bytes=parts.overhead,
        total_bytes=total,
        n_ctx=n_ctx,
        n_parallel=n_parallel,
        fits=fits,  # type: ignore[arg-type]
        message=message.strip(),
    )


def _suggestions(model: LocalModel, config: LoadConfig, room: int, parts: Breakdown) -> str:
    """Polish hints on what to change so that the instance fits in ``room``."""
    n_parallel = max(1, int(config.n_parallel or 1))
    fixed = breakdown(model, _with_ctx(config, 1, 1), weights_bytes=parts.weights).total
    if fixed > room:
        return ("Wybierz mniejszy model lub mocniejszą kwantyzację, wyładuj inne modele "
                "albo podnieś limit RAM w ustawieniach Runtime.")
    hints: list[str] = []
    ctx_fit = _max_ctx_for(model, config, room)
    if ctx_fit >= min(MIN_USEFUL_CTX, native_context(model)):
        hints.append(f"zmniejsz kontekst do {format_tokens(ctx_fit)} tokenów")
    if n_parallel > 1:
        single_fit = _max_ctx_for(model, _with_ctx(config, config.n_ctx, 1), room)
        if single_fit >= min(int(config.n_ctx or 1), native_context(model)):
            hints.append("ogranicz liczbę równoległych żądań do 1")
        else:
            hints.append("ogranicz liczbę równoległych żądań")
    hints.append("wyładuj inne modele albo podnieś limit RAM")
    text = ", ".join(hints)
    return text[0].upper() + text[1:] + "."


__all__ = [
    "AttentionLayout",
    "Breakdown",
    "attention_layout",
    "auto_ctx",
    "breakdown",
    "default_ram_budget",
    "estimate",
    "format_gb",
    "format_tokens",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "kv_element_bytes",
    "max_ctx_that_fits",
    "model_weights_bytes",
    "native_context",
    "normalize_kv_type",
    "process_memory_bytes",
    "system_memory",
]
