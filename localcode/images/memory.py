"""Memory estimates for image models.

Diffusion models have no KV cache, so the estimate is much simpler than the
language one (``runtime/memory.py``), but it still has to be *reliable*: image
weights are large, the GPU shares the system RAM on Apple Silicon and the image
runtime asks :meth:`RuntimeManager.make_room` to evict language models with this
number.

``total = weights · WEIGHT_FACTOR + BASE_OVERHEAD + pixels · PIXEL_BYTES``

* **weights** — the bytes actually loaded: the checkpoint plus every companion
  file recorded in ``ModelMeta.image_components`` (CLIP/T5/VAE), or every
  ``*.safetensors`` under an MLX diffusion directory (they live in
  ``transformer/``, ``vae/``, ``text_encoder/`` sub-directories, so the
  language-side helper, which only globs the top level, cannot be reused).
* **WEIGHT_FACTOR = 1.10** — dequantisation scratch and the allocator's slack.
* **BASE_OVERHEAD = 512 MiB** — backend, tokenizer, Metal pipelines.
* **PIXEL_BYTES = 1536 B/px** — activations of the denoiser and, above all, the
  VAE decoder, which grow with the output area.

Calibration (Apple M5, 32 GB, macOS 26, September 2026): SD 1.5 Q4_0 (1.57 GB
file) generating 512×512 with stable-diffusion.cpp/Metal peaked at 2.10 GB
resident (``ru_maxrss``); the formula gives 1.57·1.10 + 0.50 + 0.38 = 2.61 GB,
i.e. ~25 % on the safe side, which is what we want for a budget check.  The same
shape holds for mflux: a 4-bit FLUX.2-klein-4B (4.6 GB of weights) estimates at
~7.1 GB for 1024×1024.

The real number is measured after loading (``GET /memory`` of the worker, or the
process footprint) and replaces the estimate in the runtime's accounting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from localcode.hub.types import LocalModel, MemoryEstimate
from localcode.runtime import memory as rt_memory

log = logging.getLogger(__name__)

__all__ = [
    "BASE_OVERHEAD",
    "PIXEL_BYTES",
    "WEIGHT_FACTOR",
    "estimate",
    "estimate_bytes",
    "format_gb",
    "weights_bytes",
]

GiB = rt_memory.GiB
MiB = rt_memory.MiB

WEIGHT_FACTOR = 1.10
BASE_OVERHEAD = 512 * MiB
PIXEL_BYTES = 1536
DEFAULT_PIXELS = 1024 * 1024

format_gb = rt_memory.format_gb


def _component_paths(meta: Any) -> list[str]:
    components: Mapping[str, Any] | None = getattr(meta, "image_components", None)
    if not isinstance(components, Mapping):
        return []
    return [str(v) for v in components.values() if v]


def _file_size(path: str | Path) -> int:
    try:
        st = Path(path).expanduser().stat()
    except OSError:
        return 0
    return int(st.st_size)


def _dir_weights(directory: Path) -> int:
    """Sum of every weight file under an MLX diffusion checkpoint."""
    total = 0
    try:
        for pattern in ("*.safetensors", "*/*.safetensors", "*/*/*.safetensors"):
            for f in directory.glob(pattern):
                if f.is_file() and not f.name.startswith("."):
                    total += _file_size(f)
    except OSError as exc:  # pragma: no cover - unreadable directory
        log.debug("cannot size %s: %s", directory, exc)
    return total


def weights_bytes(model: LocalModel, *, components: Iterable[str] | None = None) -> int:
    """Bytes of everything the backend loads for ``model``."""
    total = 0
    path = Path(model.path).expanduser() if model.path else None
    if path is not None:
        if path.is_dir():
            total += _dir_weights(path)
        elif path.is_file():
            total += _file_size(path)
    extra = list(components) if components is not None else _component_paths(model.meta)
    seen = {str(path)} if path is not None else set()
    for item in extra:
        if str(Path(item).expanduser()) in seen:
            continue
        seen.add(str(Path(item).expanduser()))
        total += _file_size(item)
    if total <= 0:
        total = max(0, int(model.size_bytes or 0))
    return total


def estimate_bytes(
    model: LocalModel,
    *,
    width: int = 1024,
    height: int = 1024,
    count: int = 1,
    components: Iterable[str] | None = None,
) -> int:
    """Peak bytes one instance of ``model`` is expected to occupy."""
    weights = weights_bytes(model, components=components)
    pixels = max(1, int(width) * int(height))
    # Images of a batch are generated one after another, so the working set does
    # not scale with ``count``; only a small share is kept per finished image.
    work = pixels * PIXEL_BYTES + max(0, int(count) - 1) * pixels * 4
    return int(weights * WEIGHT_FACTOR + BASE_OVERHEAD + work)


def estimate(
    model: LocalModel,
    *,
    width: int = 1024,
    height: int = 1024,
    budget: int | None = None,
    in_use: int = 0,
    reclaimable: int = 0,
    components: Iterable[str] | None = None,
) -> MemoryEstimate:
    """``MemoryEstimate`` DTO for the UI ("Zmieści się: …", Polish message).

    ``in_use`` is what other instances (language *and* image) already reserve in
    the shared budget, ``reclaimable`` how much of that could be freed by
    evicting idle ones.
    """
    weights = weights_bytes(model, components=components)
    total = estimate_bytes(model, width=width, height=height, components=components)
    overhead = max(0, total - weights)
    limit = int(budget) if budget is not None else rt_memory.default_ram_budget(None)
    free_now = max(0, limit - max(0, int(in_use)))
    free_after = max(0, free_now + max(0, int(reclaimable)))
    if total <= free_now:
        fits: str = "yes"
        message = (f"Model obrazów zajmie ok. {format_gb(total)} "
                   f"(wolne w limicie: {format_gb(free_now)}).")
    elif total <= free_after:
        fits = "tight"
        message = (f"Model obrazów zajmie ok. {format_gb(total)} — zmieści się dopiero po "
                   f"wyładowaniu bezczynnych modeli (wolne teraz: {format_gb(free_now)}).")
    elif total > limit:
        fits = "no"
        message = (f"Model obrazów potrzebuje ok. {format_gb(total)}, a cały limit RAM to "
                   f"{format_gb(limit)}. Wybierz mniejszy model albo podnieś limit w ustawieniach.")
    else:
        fits = "no"
        message = (f"Za mało pamięci: model obrazów potrzebuje ok. {format_gb(total)}, "
                   f"a po zwolnieniu bezczynnych modeli dostępne byłoby {format_gb(free_after)}.")
    return MemoryEstimate(
        weights_bytes=int(weights),
        kv_cache_bytes=0,
        overhead_bytes=int(overhead),
        total_bytes=int(total),
        n_ctx=0,
        n_parallel=1,
        fits=fits,  # type: ignore[arg-type]
        message=message,
    )
