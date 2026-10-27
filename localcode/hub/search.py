"""Hugging Face Hub client: search, model details (variants + memory fit), staff picks.

Design
------
* **Search** uses one ``HfApi.list_models`` call per format with the tag filter
  (``filter="gguf"`` / ``filter="mlx"``) and ``expand=[...]`` so downloads, likes,
  last-modified, tags, gating and the ``gguf`` / ``safetensors`` summaries (total
  parameters, architecture, context length) come back in the same request
  (verified against huggingface_hub 1.32: ``ExpandModelProperty_T`` /
  ``ModelSort_T``).  Non-LLM repos (diffusion, speech, classifiers…) are dropped
  by pipeline tag, so we over-fetch 2× and trim.  ``format="any"`` runs both
  queries concurrently and merges them by the requested sort key.
* **Details** = ``model_info(files_metadata=True)`` (file sizes, LFS hashes) +
  architecture facts for memory estimates:
    - GGUF: the first 512 KB of one model file via an HTTP Range request, parsed
      with :func:`localcode.hub.metadata.read_gguf_header` in truncation-tolerant
      mode (architecture keys precede the tokenizer arrays in every GGUF writer);
      fallback: the ``gguf`` summary of ``model_info``.
    - MLX: ``config.json`` (a few KB).
  GGUF files are grouped into variants by quantisation (split shards and
  per-quant subfolders included); ``mmproj`` projectors are listed as their own
  variants and added automatically to the download of every model variant of a
  vision repo.  An MLX repo is a single variant.  Each variant gets a
  ``MemoryEstimate`` from ``runtime.memory.estimate`` (imported lazily; a local
  size-based estimator is used when that module is unavailable) and one variant
  is marked ``recommended``.
* Everything is blocking I/O; the ``a*`` methods wrap it in ``asyncio.to_thread``
  for the aiohttp server.  Errors are :class:`HubError` with Polish messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Iterable

from localcode.config import Settings
from localcode.hub.curated import CURATED_REPOS, curated_entry, curated_models, to_hub_model
from localcode.hub.metadata import (
    MetadataError,
    base_quant,
    gguf_stem,
    is_auxiliary_gguf,
    is_mmproj,
    meta_from_gguf_kv,
    meta_from_hf_config,
    mlx_quant_label,
    parse_quant,
    parse_size_label,
    read_gguf_header,
    split_shard,
)
from localcode.hub.types import LocalModel, MemoryEstimate, ModelMeta

log = logging.getLogger(__name__)

__all__ = [
    "HubClient",
    "HubError",
    "hub_error_from",
    "group_gguf_variants",
    "mlx_variant",
    "pick_recommended",
    "readme_excerpt",
    "fallback_estimate",
    "DEFAULT_ESTIMATE_CTX",
]

FORMATS = ("any", "gguf", "mlx")
SORTS = {"trending": "trending_score", "downloads": "downloads", "likes": "likes", "recent": "last_modified"}
EXPAND = [
    "author", "downloads", "downloadsAllTime", "likes", "lastModified", "tags", "gated", "gguf", "safetensors",
    "pipeline_tag", "library_name", "trendingScore",
]
# pipeline tags that can never be served by llama-server / mlx-lm
EXCLUDED_PIPELINES = frozenset({
    "text-to-image", "image-to-image", "text-to-video", "image-to-video", "text-to-speech",
    "text-to-audio", "automatic-speech-recognition", "audio-to-audio", "audio-classification",
    "image-classification", "object-detection", "image-segmentation", "depth-estimation",
    "zero-shot-image-classification", "mask-generation", "unconditional-image-generation",
    "video-classification", "image-feature-extraction", "reinforcement-learning", "robotics",
    "tabular-classification", "tabular-regression", "time-series-forecasting", "text-to-3d",
    "image-to-3d", "keypoint-detection", "token-classification", "text-classification",
    "zero-shot-classification", "voice-activity-detection", "graph-ml",
})
_HIDDEN_TAG_PREFIXES = ("base_model:", "license:", "region:", "arxiv:", "dataset:", "doi:", "deploy:", "endpoints_compatible", "autotrain_compatible")
_MEDIA_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".mp4", ".mov", ".pdf")
PREFERRED_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")  # best quality first

#: Diffusion models do not survive low-bit quantisation the way language models
#: do: Stable Diffusion 1.5 at Q4_0 renders a blurred colour field instead of the
#: prompt, while the same checkpoint at Q8_0 renders it correctly. Image models
#: are therefore never recommended below Q6_K, even when a smaller file would
#: fit more comfortably in memory.
IMAGE_PREFERRED_QUANTS = ("Q8_0", "Q6_K", "F16", "BF16")
DEFAULT_ESTIMATE_CTX = 16384
GGUF_HEAD_BYTES = 512 * 1024
README_LIMIT = 4000
DETAILS_TTL_S = 600.0
HTTP_TIMEOUT_S = 20.0

GATED_MSG = (
    "Model „{repo}” wymaga zaakceptowania licencji na Hugging Face. Otwórz https://huggingface.co/{repo}, "
    "zaloguj się i zaakceptuj warunki, a następnie wpisz swój token Hugging Face w ustawieniach LocalCode."
)


class HubError(Exception):
    """Hub failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "hub_error", status: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def hub_error_from(exc: BaseException, repo: str | None = None, *, token_set: bool = False) -> HubError:
    """Map huggingface_hub / httpx exceptions to :class:`HubError`."""
    if isinstance(exc, HubError):
        return exc
    import httpx
    from huggingface_hub.errors import (
        DisabledRepoError,
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    name = repo or "?"
    if isinstance(exc, GatedRepoError):
        return HubError(GATED_MSG.format(repo=name), "gated", 403)
    if isinstance(exc, RepositoryNotFoundError):
        extra = " albo token Hugging Face jest nieprawidłowy" if token_set else " (jeśli jest prywatne, ustaw token Hugging Face)"
        return HubError(f"Nie znaleziono repozytorium „{name}” na Hugging Face{extra}.", "not_found", 404)
    if isinstance(exc, DisabledRepoError):
        return HubError(f"Repozytorium „{name}” zostało wyłączone przez Hugging Face.", "disabled", 410)
    if isinstance(exc, (EntryNotFoundError, RevisionNotFoundError)):
        return HubError(f"Nie znaleziono pliku w repozytorium „{name}”.", "file_not_found", 404)
    if isinstance(exc, OfflineModeIsEnabled):
        return HubError("Tryb offline Hugging Face jest włączony (HF_HUB_OFFLINE) — wyszukiwanie i pobieranie są niedostępne.", "offline", 503)
    if isinstance(exc, HfHubHTTPError):
        status = exc.response.status_code if getattr(exc, "response", None) is not None else 0
        if status == 401:
            return HubError("Hugging Face odrzucił token (401). Sprawdź token w ustawieniach LocalCode.", "unauthorized", 401)
        if status == 403:
            return HubError(GATED_MSG.format(repo=name), "gated", 403)
        if status == 429:
            return HubError("Przekroczono limit zapytań do Hugging Face. Odczekaj chwilę lub ustaw token HF, aby podnieść limit.", "rate_limited", 429)
        if status >= 500:
            return HubError("Hugging Face jest chwilowo niedostępny. Spróbuj ponownie za chwilę.", "unavailable", 503)
        return HubError(f"Błąd Hugging Face (HTTP {status}).", "http_error", 502)
    if isinstance(exc, httpx.TimeoutException):
        return HubError("Przekroczono czas oczekiwania na odpowiedź Hugging Face.", "timeout", 504)
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return HubError("Brak połączenia z Hugging Face. Sprawdź połączenie z internetem.", "network", 503)
    if isinstance(exc, OSError):
        return HubError(f"Błąd sieci lub dysku podczas komunikacji z Hugging Face: {exc}.", "network", 503)
    return HubError(f"Nieoczekiwany błąd komunikacji z Hugging Face: {exc}.", "hub_error", 502)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without network)
# --------------------------------------------------------------------------- #


def _ms(dt: Any) -> int | None:
    if isinstance(dt, datetime):
        return int(dt.timestamp() * 1000)
    return None


def _params_from(gguf: Any, safetensors: Any, name: str) -> tuple[float | None, float | None]:
    label_total, label_active = parse_size_label(name)
    total = None
    if isinstance(gguf, dict) and isinstance(gguf.get("total"), (int, float)) and gguf["total"] > 0:
        total = gguf["total"] / 1e9
    elif safetensors is not None and getattr(safetensors, "total", None):
        total = safetensors.total / 1e9
    if total and label_total and not (0.5 <= total / label_total <= 2.0):
        total = label_total  # packed/quantised counts can be off (e.g. some MLX uploads)
    if total is None:
        total = label_total
    if total is not None:
        total = round(total, 2) if total >= 1 else round(total, 3)
    return total, label_active


def _description(tags: Iterable[str]) -> str | None:
    kinds = {"quantized": "Kwantyzacja modelu", "finetune": "Dostrojenie modelu", "adapter": "Adapter do modelu",
             "merge": "Połączenie modeli"}
    plain = None
    for t in tags:
        if not t.startswith("base_model:"):
            continue
        rest = t[len("base_model:"):]
        kind, _, base = rest.partition(":")
        if base and kind in kinds:
            return f"{kinds[kind]} {base}"
        if not base and plain is None:
            plain = rest
    return f"Na bazie modelu {plain}" if plain else None


def _visible_tags(tags: Iterable[str] | None) -> list[str]:
    return [t for t in (tags or []) if not t.startswith(_HIDDEN_TAG_PREFIXES)][:15]


def readme_excerpt(text: str | None, limit: int = README_LIMIT) -> str | None:
    """README without YAML front matter, trimmed to ``limit`` chars at a paragraph/line break."""
    if not text:
        return None
    t = text.lstrip("﻿")
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + 4:]
            t = t[t.find("\n") + 1:] if "\n" in t else ""
    t = re.sub(r"<!--[\s\S]*?-->", "", t).strip()
    if len(t) <= limit:
        return t or None
    cut = t.rfind("\n\n", 0, limit)
    if cut < limit * 0.6:
        cut = t.rfind("\n", 0, limit)
    if cut < limit * 0.6:
        cut = limit
    return t[:cut].rstrip() + "\n\n…"


def _projector_subject(name: str) -> str:
    base = os.path.basename(name)
    stem = base[:-5] if base.lower().endswith(".gguf") else base
    q = parse_quant(base)
    if q:
        stem = re.sub(rf"[-._]?{re.escape(q)}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?i)[-._]?mmproj[-._]?", "-", stem)
    return stem.strip("-_. ").lower()


def _default_projector(projectors: list[dict[str, Any]], model_stem: str) -> dict[str, Any] | None:
    if not projectors:
        return None
    stem_l = model_stem.lower()
    named = [p for p in projectors if _projector_subject(p["files"][0]) and _projector_subject(p["files"][0]) in stem_l]
    generic = [p for p in projectors if not _projector_subject(p["files"][0])]
    pool = named or generic or projectors
    order = {"F16": 0, "BF16": 1, "Q8_0": 2, "F32": 4}
    return sorted(pool, key=lambda p: (order.get((p["quant"] or "").upper(), 3), p["sizeBytes"]))[0]


def group_gguf_variants(files: Iterable[tuple[str, int]]) -> list[dict[str, Any]]:
    """Group GGUF repo files into download variants.

    Returns HubVariant dicts (without ``estimate``/``recommended``): model
    variants sorted by size, then projector variants (``isVisionProjector``).
    Model variants of a repo with projectors get the default projector appended
    to ``files`` and ``sizeBytes``; ``modelBytes`` keeps the weights-only size.
    """
    groups: dict[tuple[str, str], list[tuple[int, int, str, int]]] = {}
    projectors: list[dict[str, Any]] = []
    for name, size in files:
        if not name.lower().endswith(".gguf"):
            continue
        size = int(size or 0)
        directory = os.path.dirname(name)
        if is_auxiliary_gguf(name) and not is_mmproj(name):
            continue                       # draft heads, adapters, text encoders…
        if is_mmproj(name):
            projectors.append({
                "id": gguf_stem(name) if not directory else f"{directory}/{gguf_stem(name)}",
                "label": f"Projektor wizji {parse_quant(name) or ''}".strip(),
                "quant": parse_quant(name),
                "files": [name],
                "sizeBytes": size,
                "modelBytes": size,
                "isVisionProjector": True,
                "complete": True,
            })
            continue
        shard = split_shard(name)
        stem = gguf_stem(name)
        idx, count = (shard[1], shard[2]) if shard else (1, 1)
        groups.setdefault((directory, stem), []).append((idx, count, name, size))
    variants: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for (directory, stem), parts in sorted(groups.items()):
        parts.sort()
        count = parts[0][1]
        complete = count == 1 or ({p[0] for p in parts} == set(range(1, count + 1)))
        names = [p[2] for p in parts]
        quant = parse_quant(names[0]) or (parse_quant(directory + ".gguf") if directory else None)
        vid = stem if stem not in seen_ids else f"{directory}/{stem}"
        seen_ids.add(vid)
        size = sum(p[3] for p in parts)
        variants.append({
            "id": vid,
            "label": quant or stem,
            "quant": quant,
            "files": names,
            "sizeBytes": size,
            "modelBytes": size,
            "isVisionProjector": False,
            "complete": complete,
        })
    variants.sort(key=lambda v: (v["sizeBytes"], v["id"]))
    if projectors:
        for v in variants:
            proj = _default_projector(projectors, gguf_stem(v["files"][0]))
            if proj:
                v["files"] = v["files"] + proj["files"]
                v["sizeBytes"] += proj["sizeBytes"]
                v["projector"] = proj["files"][0]
    projectors.sort(key=lambda p: p["sizeBytes"])
    return variants + projectors


def _whole_repo_variant(repo: str, files: Iterable[tuple[str, int]],
                        quant: str | None) -> dict[str, Any]:
    """Every file that is part of the model, for layouts we do not itemise.

    Used by the speech models: there is nothing to choose between inside such a
    repository, so the variant is the repository — minus the readme, the images
    and the git plumbing, which would only cost bandwidth.
    """
    keep = [(name, size) for name, size in files
            if not name.startswith(".")
            and not name.lower().endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".webp",
                                           ".svg", ".mp4", ".pdf"))]
    total = sum(size for _, size in keep)
    return {
        "id": repo.split("/")[-1],
        "label": quant or "MLX",
        "quant": quant,
        "files": [name for name, _ in keep],
        "sizeBytes": total,
        "modelBytes": total,
        "isVisionProjector": False,
        "complete": bool(keep),
    }


def mlx_variant(repo: str, files: Iterable[tuple[str, int]], quant: str | None) -> dict[str, Any]:
    """The single download variant of an MLX repo (all top-level files except media)."""
    chosen = [
        (n, int(s or 0)) for n, s in files
        if "/" not in n and n != ".gitattributes" and not n.lower().endswith(_MEDIA_EXT)
    ]
    size = sum(s for _, s in chosen)
    weights = sum(s for n, s in chosen if n.endswith(".safetensors"))
    return {
        "id": repo.split("/")[-1],
        "label": quant or "MLX",
        "quant": quant,
        "files": [n for n, _ in chosen],
        "sizeBytes": size,
        "modelBytes": weights or size,
        "isVisionProjector": False,
        "complete": weights > 0,
    }


def _pick_image_recommended(models: list[dict[str, Any]]) -> str | None:
    """Recommended variant for a diffusion model (quality floor: Q6_K)."""

    def fits(v: dict[str, Any]) -> str:
        return ((v.get("estimate") or {}).get("fits")) or "no"

    usable = [v for v in models if base_quant(v.get("quant")) in IMAGE_PREFERRED_QUANTS]
    if not usable:                       # repo without a high-quality variant: take the biggest
        usable = sorted(models, key=lambda v: v["sizeBytes"], reverse=True)[:1]
    for state in ("yes", "tight"):
        for q in IMAGE_PREFERRED_QUANTS:
            matches = [v for v in usable if base_quant(v.get("quant")) == q and fits(v) == state]
            if matches:
                choice = min(matches, key=lambda v: v["sizeBytes"])
                choice["recommended"] = True
                return str(choice.get("id") or "")
    if usable:
        choice = min(usable, key=lambda v: v["sizeBytes"])
        choice["recommended"] = True
        return str(choice.get("id") or "")
    return None


def pick_recommended(variants: list[dict[str, Any]], *, image: bool = False) -> str | None:
    """Mark and return the recommended variant id.

    Best quality among Q8_0 > Q6_K > Q5_K_M > Q4_K_M whose estimate fits "yes"
    (Unsloth "UD-" dynamic quants preferred at equal level); otherwise the largest
    quantised variant that fits "yes"; otherwise the smallest that fits "tight".

    ``image=True`` switches to :data:`IMAGE_PREFERRED_QUANTS` and never falls back
    to a quantisation below Q6_K — a too-small diffusion quant produces garbage
    rather than a slightly worse picture, so a tight fit beats a useless file.
    """
    for v in variants:
        v["recommended"] = False
    models = [v for v in variants if not v.get("isVisionProjector") and v.get("complete", True)]
    if image:
        return _pick_image_recommended(models)

    def fits(v: dict[str, Any]) -> str:
        return ((v.get("estimate") or {}).get("fits")) or "no"

    choice: dict[str, Any] | None = None
    for q in PREFERRED_QUANTS:
        matches = [v for v in models if base_quant(v.get("quant")) == q and fits(v) == "yes"]
        if matches:
            matches.sort(key=lambda v: (not str(v.get("quant") or "").upper().startswith("UD-"), v["sizeBytes"]))
            choice = matches[0]
            break
    if choice is None:
        full = {"F32", "F16", "BF16"}
        yes = [v for v in models if fits(v) == "yes" and (base_quant(v.get("quant")) not in full)]
        if not yes:
            yes = [v for v in models if fits(v) == "yes"]
        if yes:
            choice = max(yes, key=lambda v: v["sizeBytes"])
    if choice is None:
        tight = [v for v in models if fits(v) == "tight"]
        if tight:
            choice = min(tight, key=lambda v: v["sizeBytes"])
    if choice is None:
        return None
    choice["recommended"] = True
    return choice["id"]


# --------------------------------------------------------------------------- #
# Memory estimates
# --------------------------------------------------------------------------- #


def _fmt_gb(n: float) -> str:
    return f"{n / 1e9:.1f}".replace(".", ",") + " GB"


def _system_memory() -> tuple[int, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:  # pragma: no cover - psutil is a dependency, but be safe
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            total = 16 * 1024**3
        return total, total // 2


_KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "8bit": 1.0625, "q4_0": 0.5625, "4bit": 0.5625}


def fallback_estimate(
    meta: ModelMeta,
    weights_bytes: int,
    *,
    n_ctx: int,
    n_parallel: int = 1,
    kv_cache_type: str = "f16",
    ram_limit_gb: float | None = None,
) -> MemoryEstimate:
    """Size-based estimate used when ``runtime.memory`` is not available."""
    bpe = _KV_BYTES.get(kv_cache_type, 2.0)
    if meta.n_layers and meta.n_kv_heads is not None and meta.head_dim:
        n_full = meta.n_full_attn_layers if meta.n_full_attn_layers is not None else meta.n_layers
        per_token_layer = 2 * meta.n_kv_heads * meta.head_dim * bpe
        kv = per_token_layer * n_full * n_ctx * n_parallel
        if meta.sliding_window and meta.n_layers > n_full:
            kv += per_token_layer * (meta.n_layers - n_full) * min(meta.sliding_window, n_ctx) * n_parallel
    else:
        kv = weights_bytes * 0.1 * (n_ctx * n_parallel / 8192)
    kv = int(kv)
    overhead = int(0.4e9 + 0.05 * weights_bytes)
    total = weights_bytes + kv + overhead
    physical, _available = _system_memory()
    budget = int(ram_limit_gb * 1e9) if ram_limit_gb else int(physical * 0.70)
    if total <= budget * 0.85:
        fits, msg = "yes", f"Zmieści się: ok. {_fmt_gb(total)} (limit {_fmt_gb(budget)})."
    elif total <= budget:
        fits, msg = "tight", f"Na styk: ok. {_fmt_gb(total)} przy limicie {_fmt_gb(budget)} — zamknij inne aplikacje lub zmniejsz kontekst."
    else:
        fits, msg = "no", f"Za duży: potrzeba ok. {_fmt_gb(total)}, a limit pamięci modeli to {_fmt_gb(budget)}."
    return MemoryEstimate(
        weights_bytes=int(weights_bytes), kv_cache_bytes=kv, overhead_bytes=overhead, total_bytes=total,
        n_ctx=n_ctx, n_parallel=n_parallel, fits=fits, message=msg,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class HubClient:
    """Search and inspect models on the Hugging Face Hub (see module docstring)."""

    def __init__(self, settings: Settings, *, api: Any | None = None) -> None:
        self.settings = settings
        self._api = api
        self._details_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._cache_lock = threading.Lock()

    # plumbing -----------------------------------------------------------------------
    @property
    def token(self) -> str | None:
        t = self.settings.hf_token
        return t.strip() if isinstance(t, str) and t.strip() else None

    @property
    def api(self) -> Any:
        if self._api is not None:
            return self._api
        from huggingface_hub import HfApi

        return HfApi(token=self.token, library_name="localcode")

    def _err(self, exc: BaseException, repo: str | None = None) -> HubError:
        return hub_error_from(exc, repo, token_set=self.token is not None)

    def _http_get(self, repo: str, filename: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        """GET a repo file (optionally a byte range) without touching any cache."""
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        headers = build_hf_headers(token=self.token, library_name="localcode")
        if byte_range:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        resp = get_session().get(hf_hub_url(repo, filename), headers=headers, follow_redirects=True, timeout=HTTP_TIMEOUT_S)
        hf_raise_for_status(resp)
        return resp.content

    # search ---------------------------------------------------------------------------
    def search(self, query: str = "", format: str = "any", sort: str = "trending", limit: int = 30) -> list[dict[str, Any]]:
        """HubModel dicts matching ``query`` (empty query = top models of the format)."""
        if format not in FORMATS:
            raise HubError(f"Nieznany format „{format}” (dozwolone: any, gguf, mlx).", "invalid", 400)
        if sort not in SORTS:
            raise HubError(f"Nieznane sortowanie „{sort}” (dozwolone: trending, downloads, likes, recent).", "invalid", 400)
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            raise HubError("Parametr limit musi być liczbą.", "invalid", 400) from None
        query = (query or "").strip()
        formats = ("gguf", "mlx") if format == "any" else (format,)
        fetch = min(limit * 2, 100)
        results: dict[str, list[dict[str, Any]]] = {}
        errors: list[BaseException] = []

        def run(fmt: str) -> None:
            try:
                results[fmt] = self._list(fmt, query, SORTS[sort], fetch)
            except BaseException as exc:  # noqa: BLE001 - re-raised below as HubError
                errors.append(exc)

        if len(formats) == 1:
            run(formats[0])
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(run, formats))
        if errors and not results:
            raise self._err(errors[0])
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fmt in formats:
            for m in results.get(fmt, []):
                if m["id"] not in seen:
                    seen.add(m["id"])
                    merged.append(m)
        if len(formats) > 1:
            key = {"trending": "trendingScore", "downloads": "downloads", "likes": "likes", "recent": "updatedAt"}[sort]
            merged.sort(key=lambda m: (m.get(key) or 0), reverse=True)
        return merged[:limit]

    def _list(self, fmt: str, query: str, sort_key: str, fetch: int) -> list[dict[str, Any]]:
        items = self.api.list_models(
            search=query or None, filter=fmt, sort=sort_key, limit=fetch, expand=EXPAND,
        )
        out: list[dict[str, Any]] = []
        for info in items:
            model = self._hub_model(info, fmt)
            if model is not None:
                out.append(model)
        return out

    @staticmethod
    def _hub_model(info: Any, fmt: str, *, strict: bool = True) -> dict[str, Any] | None:
        repo = str(info.id)
        pipeline = getattr(info, "pipeline_tag", None)
        gguf = getattr(info, "gguf", None)
        st = getattr(info, "safetensors", None)
        if strict:
            if pipeline in EXCLUDED_PIPELINES:
                return None
            if fmt == "gguf" and isinstance(gguf, dict) and gguf and not gguf.get("architecture") and pipeline is None:
                return None  # e.g. diffusion / whisper GGUF conversions without a pipeline tag
        author, _, name = repo.partition("/")
        params, active = _params_from(gguf, st, name)
        tags = list(getattr(info, "tags", None) or [])
        entry = curated_entry(repo)
        gated = getattr(info, "gated", False)
        from localcode.hub import store_data          # local: avoids an import cycle

        trust = store_data.trust_for(repo, hub={
            "author": getattr(info, "author", None) or author,
            "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None),
            "tags": tags,
            "updatedAt": getattr(info, "last_modified", None),
            "gated": gated,
        }, curated=repo in CURATED_REPOS)
        return {
            "id": repo,
            "author": getattr(info, "author", None) or author,
            "name": name,
            "format": fmt,
            "downloads": getattr(info, "downloads", None),
            "downloadsAllTime": getattr(info, "downloads_all_time", None),
            "likes": getattr(info, "likes", None),
            "updatedAt": _ms(getattr(info, "last_modified", None)),
            "tags": _visible_tags(tags),
            "pipelineTag": pipeline,
            "paramsB": params,
            "activeParamsB": active,
            "gated": bool(gated) and gated is not False,
            "curated": repo in CURATED_REPOS,
            "description": (entry or {}).get("note") or _description(tags),
            "arch": gguf.get("architecture") if isinstance(gguf, dict) else None,
            "contextLength": gguf.get("context_length") if isinstance(gguf, dict) else None,
            "trendingScore": getattr(info, "trending_score", None),
            "trust": trust.to_dict(),
            "publisher": trust.provenance.to_dict() if trust.provenance else {},
        }

    async def asearch(self, query: str = "", format: str = "any", sort: str = "trending", limit: int = 30) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, format, sort, limit)

    # curated --------------------------------------------------------------------------
    def curated(self) -> list[dict[str, Any]]:
        """Staff picks as HubModel dicts (offline, no network)."""
        return [to_hub_model(m) for m in curated_models()]

    async def acurated(self) -> list[dict[str, Any]]:
        return self.curated()

    # details --------------------------------------------------------------------------
    def details(self, repo: str, *, refresh: bool = False) -> dict[str, Any]:
        """HubModelDetail for ``repo`` (cached for 10 minutes)."""
        repo = (repo or "").strip().strip("/")
        if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
            raise HubError(f"Nieprawidłowy identyfikator repozytorium „{repo}” (oczekiwano organizacja/nazwa).", "invalid", 400)
        if not refresh:
            with self._cache_lock:
                hit = self._details_cache.get(repo)
            if hit and time.monotonic() - hit[0] < DETAILS_TTL_S:
                return json.loads(json.dumps(hit[1]))
        try:
            info = self.api.model_info(repo, files_metadata=True)
        except Exception as exc:  # noqa: BLE001 - mapped to HubError
            raise self._err(exc, repo) from exc
        detail = self._build_details(repo, info)
        with self._cache_lock:
            self._details_cache[repo] = (time.monotonic(), detail)
        return json.loads(json.dumps(detail))

    async def adetails(self, repo: str, *, refresh: bool = False) -> dict[str, Any]:
        return await asyncio.to_thread(self.details, repo, refresh=refresh)

    def variant(self, repo: str, variant_id: str) -> dict[str, Any]:
        """One variant of ``details(repo)`` (for ``POST /api/downloads {repo, variantId}``)."""
        d = self.details(repo)
        for v in d["variants"]:
            if v["id"] == variant_id:
                return v
        raise HubError(f"Repozytorium „{repo}” nie ma wariantu „{variant_id}”.", "variant_not_found", 404)

    def _build_details(self, repo: str, info: Any) -> dict[str, Any]:
        siblings = list(getattr(info, "siblings", None) or [])
        files = [(s.rfilename, int(s.size if s.size is not None else (s.lfs.size if s.lfs else 0) or 0)) for s in siblings]
        names = [n for n, _ in files]
        has_gguf = any(n.lower().endswith(".gguf") and not is_mmproj(n) for n in names)
        has_st = any(n.endswith(".safetensors") for n in names)
        # Speech models are the exception to "GGUF or safetensors": the MLX
        # ports of Whisper and Parakeet ship their weights as ``weights.npz``
        # next to a config, and refusing them here is what made every model in
        # the "Mowa" shelf listable but not installable.
        has_npz = any(n.endswith(".npz") for n in names)
        if not has_gguf and not has_st and not has_npz:
            raise HubError(
                f"Repozytorium „{repo}” nie zawiera modelu w formacie GGUF ani MLX (safetensors).", "unsupported", 422
            )
        fmt = "gguf" if has_gguf else "mlx"
        access_error: str | None = None
        config: dict[str, Any] | None = None
        meta: ModelMeta
        if fmt == "gguf":
            meta, access_error = self._remote_gguf_meta(repo, info, files)
        else:
            meta, config, access_error = self._remote_mlx_meta(repo, info)
        readme = None
        if "README.md" in names:
            try:
                readme = readme_excerpt(self._http_get(repo, "README.md").decode("utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001 - README is optional
                err = self._err(exc, repo)
                if err.code == "gated":
                    access_error = err.message
                log.debug("README of %s unavailable: %s", repo, exc)

        if fmt == "gguf":
            variants = group_gguf_variants(files)
        else:
            variants = [mlx_variant(repo, files, (config and mlx_quant_label(config)) or meta.quant)]
            if not variants[0].get("files"):
                # A plain MLX directory (``weights.npz`` + config + tokenizer):
                # the whole repository is the model, minus the documentation.
                variants = [_whole_repo_variant(repo, files, meta.quant)]
        for v in variants:
            vmeta = ModelMeta(**{**meta.__dict__})
            vmeta.quant = v.get("quant") or meta.quant
            if v["isVisionProjector"]:
                vmeta = ModelMeta(arch="clip", quant=v.get("quant"), n_layers=0, n_full_attn_layers=0)
            key = f"{repo}/{v['id']}" if fmt == "gguf" else repo
            v["estimate"] = self.estimate(vmeta, int(v["sizeBytes"]), fmt, key=key, name=v["label"])
        is_image = bool(meta.is_image) or any(
            tag in {"text-to-image", "image-generation", "diffusers"}
            for tag in (getattr(info, "tags", None) or []))
        pick_recommended(variants, image=is_image)
        model = self._hub_model(info, fmt, strict=False) or {}
        if meta.params_b and not model.get("paramsB"):
            model["paramsB"] = meta.params_b
        if meta.active_params_b and not model.get("activeParamsB"):
            model["activeParamsB"] = meta.active_params_b
        model["arch"] = meta.arch or model.get("arch")
        model["contextLength"] = meta.context_length or model.get("contextLength")
        caps: list[str] = []
        if meta.is_embedding:
            caps = ["embedding"]
        else:
            from localcode.hub.metadata import template_supports_thinking, template_supports_tools

            tmpl = self._remote_template(info, config)
            if template_supports_tools(tmpl):
                caps.append("tools")
            if template_supports_thinking(tmpl, repo):
                caps.append("thinking")
            if any(v["isVisionProjector"] for v in variants) or (config and isinstance(config.get("vision_config"), dict)):
                caps.append("vision")
        return {
            "model": model,
            "readme": readme,
            "contextLength": meta.context_length,
            "arch": meta.arch,
            "capabilities": caps,
            "isEmbedding": meta.is_embedding,
            "variants": variants,
            "files": [{"name": n, "size": s} for n, s in files],
            "accessError": access_error,
            "sha": getattr(info, "sha", None),
        }

    def _remote_template(self, info: Any, config: dict[str, Any] | None) -> str | None:
        gguf = getattr(info, "gguf", None)
        if isinstance(gguf, dict) and isinstance(gguf.get("chat_template"), str):
            return gguf["chat_template"]
        card = getattr(info, "config", None)
        if isinstance(card, dict):
            tok = card.get("tokenizer_config")
            if isinstance(tok, dict) and isinstance(tok.get("chat_template"), str):
                return tok["chat_template"]
        return None

    def _remote_gguf_meta(self, repo: str, info: Any, files: list[tuple[str, int]]) -> tuple[ModelMeta, str | None]:
        gguf = getattr(info, "gguf", None) if isinstance(getattr(info, "gguf", None), dict) else {}
        candidates = []
        for name, size in files:
            if not name.lower().endswith(".gguf") or is_mmproj(name):
                continue
            shard = split_shard(name)
            if shard and shard[1] != 1:
                continue
            candidates.append((size or 0, name))
        candidates.sort()
        access_error = None
        meta: ModelMeta | None = None
        if candidates:
            name = candidates[0][1]
            for nbytes in (GGUF_HEAD_BYTES, 8 * GGUF_HEAD_BYTES):
                try:
                    blob = self._http_get(repo, name, byte_range=(0, nbytes - 1))
                    header = read_gguf_header(blob, want_tensors=False, allow_truncated=True)
                except MetadataError as exc:
                    log.debug("remote GGUF header of %s/%s unreadable: %s", repo, name, exc)
                    break
                except Exception as exc:  # noqa: BLE001 - optional enrichment
                    err = self._err(exc, repo)
                    if err.code == "gated":
                        access_error = err.message
                    log.debug("remote GGUF header of %s/%s unavailable: %s", repo, name, exc)
                    break
                arch = header.kv.get("general.architecture")
                if arch and f"{arch}.block_count" in header.kv:
                    meta = meta_from_gguf_kv(header.kv, filename=name)
                    break
                if not header.truncated:
                    meta = meta_from_gguf_kv(header.kv, filename=name)
                    break
        if meta is None:
            meta = ModelMeta(arch=gguf.get("architecture"), context_length=gguf.get("context_length"))
        total = gguf.get("total")
        if isinstance(total, (int, float)) and total > 0:
            meta.params_b = round(total / 1e9, 2) if total >= 1e9 else round(total / 1e9, 3)
        if meta.active_params_b is None:
            _, active = parse_size_label(repo.split("/")[-1])
            meta.active_params_b = active
        if meta.context_length is None and isinstance(gguf.get("context_length"), int):
            meta.context_length = gguf["context_length"]
        return meta, access_error

    def _remote_mlx_meta(self, repo: str, info: Any) -> tuple[ModelMeta, dict[str, Any] | None, str | None]:
        config: dict[str, Any] | None = None
        access_error = None
        try:
            config = json.loads(self._http_get(repo, "config.json").decode("utf-8"))
            if not isinstance(config, dict):
                config = None
        except Exception as exc:  # noqa: BLE001 - optional enrichment
            err = self._err(exc, repo)
            if err.code == "gated":
                access_error = err.message
            log.debug("config.json of %s unavailable: %s", repo, exc)
        st = getattr(info, "safetensors", None)
        n_params = int(st.total) if st is not None and getattr(st, "total", None) else None
        name = repo.split("/")[-1]
        label_total, _ = parse_size_label(name)
        if n_params and label_total and not (0.5 <= n_params / 1e9 / label_total <= 2.0):
            n_params = int(label_total * 1e9)
        if config is None:
            return ModelMeta(params_b=round(n_params / 1e9, 2) if n_params else label_total), None, access_error
        meta = meta_from_hf_config(config, n_params=n_params, name=name)
        return meta, config, access_error

    # estimates ------------------------------------------------------------------------
    def estimate(self, meta: ModelMeta, size_bytes: int, fmt: str, *, key: str = "", name: str = "") -> dict[str, Any]:
        """MemoryEstimate dict for a not-yet-downloaded variant."""
        rt = self.settings.runtime
        n_ctx = min(meta.context_length or 8192, rt.default_ctx or DEFAULT_ESTIMATE_CTX)
        if meta.is_embedding:
            n_ctx = min(meta.context_length or 2048, 8192)
        n_parallel = max(1, int(rt.default_parallel or 1))
        kv_type = rt.kv_cache_type
        if fmt == "mlx":
            kv_type = {"q8_0": "8bit", "q4_0": "4bit"}.get(kv_type, kv_type)
        try:
            from localcode.runtime import memory  # written concurrently; optional
            from localcode.runtime.base import LoadConfig

            model = LocalModel(
                key=key or name or "hub", format=fmt, path="", display_name=name or key,  # type: ignore[arg-type]
                size_bytes=int(size_bytes), quant=meta.quant, meta=meta,
            )
            config = LoadConfig(
                n_ctx=n_ctx, n_parallel=n_parallel, gpu_layers=rt.gpu_layers, flash_attn=rt.flash_attn,
                kv_cache_type=kv_type, embedding=meta.is_embedding,
            )
            est = memory.estimate(model, config)
            if isinstance(est, MemoryEstimate):
                return est.to_dict()
        except Exception as exc:  # noqa: BLE001 - ImportError or an incomplete module: fall back
            log.debug("runtime.memory.estimate unavailable (%s); using fallback", exc)
        return fallback_estimate(
            meta, int(size_bytes), n_ctx=n_ctx, n_parallel=n_parallel, kv_cache_type=kv_type,
            ram_limit_gb=rt.ram_limit_gb,
        ).to_dict()
