"""Local model library: every GGUF/MLX model on disk that LocalCode can load.

Sources (in priority order — on a key collision the first one wins):

1. ``settings.models_dir`` (``~/.localcode/models``) — LocalCode's own downloads,
   laid out LM Studio-style as ``<org>/<repo>/<files>``; ``source="localcode"``.
2. LM Studio's models folder (``downloadsFolder`` from ``~/.lmstudio/settings.json``,
   default ``~/.lmstudio/models``) when ``settings.scan_lmstudio``; ``source="lmstudio"``.
3. The Hugging Face hub cache (``huggingface_hub.scan_cache_dir``) when
   ``settings.scan_hf_cache``; ``source="hf-cache"``.

Discovery rules

* GGUF: every ``*.gguf`` except projectors.  Split models
  (``name-00001-of-00003.gguf``) become one model (path = shard 1, size = sum)
  and are skipped until all shards exist.  ``mmproj*.gguf`` files are attached
  to the sibling model(s) as ``mmproj_path`` and add the ``vision`` capability.
* MLX: a directory with ``config.json`` and complete ``*.safetensors`` weights
  that is recognisably MLX (``quantization`` block written by mlx-lm, ``mlx`` in
  the README front matter, or "mlx" in the repo name).  A stale
  ``model.safetensors.index.json`` that references none of the present files is
  tolerated (mlx-lm globs the weights); a partially present shard set is not.
* Keys follow :class:`~localcode.hub.types.LocalModel`: ``org/repo`` (MLX),
  ``org/repo/<gguf stem>`` (GGUF), ``local/<dir>[/<stem>]`` outside the layout.

Metadata parsing costs 10–500 ms per model, so results are cached in
``data_dir()/library-cache.json`` keyed by path and fingerprinted by
size + mtime; a rescan of an unchanged library only stats files.  Per-model load
defaults (the ``LoadConfigDTO`` the UI edits) live in
``data_dir()/model-defaults.json``.

The class is thread-safe: scans are serialised and swap the result in
atomically, readers never see a half-built list.  Blocking work (``scan``,
``delete``) has ``async`` wrappers for the aiohttp server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

from localcode.config import Settings, data_dir
from localcode.hub.metadata import (
    MetadataError,
    base_quant,
    detect_capabilities,
    gguf_stem,
    image_component_kind,
    is_image_companion,
    is_image_model_dir,
    is_mmproj,
    parse_quant,
    read_gguf_meta,
    read_hf_chat_template,
    read_image_dir_meta,
    read_mlx_config,
    read_mlx_meta,
    split_shard,
)
from localcode.hub.types import LocalModel, ModelMeta
from localcode.security.scan import scan_path as _security_scan_path

log = logging.getLogger(__name__)

__all__ = [
    "ModelLibrary",
    "LibraryError",
    "model_key",
    "display_name_for",
    "lmstudio_models_dirs",
    "is_mlx_model_dir",
    "image_companions",
    "mlx_weights_complete",
    "prune_empty_dirs",
]

# 3: modele wideo (LTX-Video, Wan, HunyuanVideo) przestały być rozpoznawane jako
#    modele obrazów — bez podbicia wersji stare wpisy trzymałyby je na półce
#    „Obrazy”, gdzie backend obrazów wywraca się na nich przy ładowaniu.
CACHE_VERSION = 3
SOURCE_ORDER = {"localcode": 0, "lmstudio": 1, "hf-cache": 2, "custom": 3}
_IGNORED_ENTRIES = {".cache", ".DS_Store", ".gitattributes", ".huggingface", ".locks"}
_QUANT_PREFERENCE = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "Q4_K_XL", "Q5_K_XL", "4bit", "8bit", "6bit", "5bit"]
#: Per-model settings stored next to the model. All but ``effort`` are fields of
#: :class:`~localcode.runtime.base.LoadConfig` and shape the process that is
#: started; ``effort`` is not a load parameter at all — it is the reasoning level
#: this particular model should default to (some models think well at "high" and
#: waste minutes at it, others need it), applied per request by the agent and the
#: chat. It lives here because "settings for this model" is where a user looks
#: for it.
_LOAD_DEFAULT_KEYS = {"nCtx", "nParallel", "gpuLayers", "flashAttn", "kvCacheType", "embedding",
                      "effort"}
#: Values ``effort`` may take; "off" means "do not think", ``None`` clears it.
EFFORT_VALUES = ("off", "minimal", "low", "medium", "high", "max")


class LibraryError(Exception):
    """User-facing library error (Polish message) with a machine-readable code."""

    def __init__(self, message: str, code: str = "library_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# --------------------------------------------------------------------------- #
# Naming helpers (shared with hub.download)
# --------------------------------------------------------------------------- #


def model_key(repo_id: str, filename: str | None = None) -> str:
    """Library key of a model downloaded from ``repo_id``.

    ``model_key("unsloth/Qwen3-0.6B-GGUF", "Q4/Qwen3-0.6B-Q4_K_M-00001-of-00002.gguf")``
    → ``"unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"``; MLX (no filename) → ``repo_id``.
    """
    repo_id = repo_id.strip("/")
    if filename and filename.lower().endswith(".gguf"):
        return f"{repo_id}/{gguf_stem(filename)}"
    return repo_id


_TRAILING_MARKER_RE = re.compile(
    r"^(?:\d+bit|mlx|gguf|bf16|fp16|f16|f32|fp32|mxfp4|nvfp4|mxfp8|q\d(?:_[0-9a-z]+)*|ud|i1|int[48]|awq|gptq|dwq|imatrix)$",
    re.IGNORECASE,
)


def display_name_for(name: str, quant: str | None) -> str:
    """``"Qwen3-4B-Instruct-2507-Q4_K_M"``, ``"Q4_K_M"`` → ``"Qwen3 4B Instruct 2507 · Q4_K_M"``."""
    base = name
    if quant:
        base = re.sub(rf"[-._ ]?{re.escape(quant)}$", "", base, flags=re.IGNORECASE)
        bq = base_quant(quant)
        if bq and bq != quant:
            base = re.sub(rf"[-._ ]?{re.escape(bq)}$", "", base, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"[-_ ]+", base) if t]
    while len(tokens) > 1 and _TRAILING_MARKER_RE.match(tokens[-1]):
        tokens.pop()
    pretty = " ".join(tokens).strip() or name
    return f"{pretty} · {quant}" if quant else pretty


def _front_matter(readme: Path) -> str:
    try:
        with open(readme, encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
    except OSError:
        return ""
    if not head.startswith("---"):
        return ""
    end = head.find("\n---", 3)
    return head[3:end] if end > 0 else ""


def is_mlx_model_dir(model_dir: Path, config: dict[str, Any], repo_id: str | None = None) -> bool:
    """Whether a ``config.json`` + weights directory is an MLX checkpoint."""
    if {"n_mels", "n_audio_layer"} <= set(config):
        return True                      # a Whisper-shaped config is MLX here
    q = config.get("quantization")
    if isinstance(q, dict) and ("bits" in q or "group_size" in q or "mode" in q):
        return True
    name = f"{repo_id or ''} {model_dir.name}".lower()
    if "mlx" in name:
        return True
    fm = _front_matter(model_dir / "README.md").lower()
    if re.search(r"^library_name:\s*['\"]?mlx", fm, re.MULTILINE) or re.search(r"^\s*-\s*['\"]?mlx['\"]?\s*$", fm, re.MULTILINE):
        return True
    return False


def mlx_weights_complete(model_dir: Path) -> tuple[bool, list[Path]]:
    """``(complete, weight files)`` for an MLX directory (see module docstring)."""
    present = sorted(p for p in model_dir.glob("*.safetensors") if not p.name.startswith("."))
    if not present:
        # The MLX ports of the speech models (Whisper, Parakeet) ship a single
        # ``weights.npz`` instead. There is nothing to be incomplete about:
        # either the file is there or the directory is not a model.
        npz = sorted(p for p in model_dir.glob("*.npz") if not p.name.startswith("."))
        return (True, npz) if npz else (False, [])
    try:
        index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        referenced = set((index.get("weight_map") or {}).values()) if isinstance(index, dict) else set()
    except (OSError, ValueError):
        referenced = set()
    names = {p.name for p in present}
    if referenced and referenced & names and not referenced <= names:
        return False, present
    return True, present


def _has_partial_downloads(d: Path) -> bool:
    part_dir = d / ".cache" / "localcode"
    if not part_dir.is_dir():
        return False
    try:
        return any(p.name.endswith(".part") for p in part_dir.rglob("*"))
    except OSError:
        return True


def prune_empty_dirs(start: Path, root: Path) -> None:
    """Remove ``start`` and its parents (strictly below ``root``) when they contain no
    model files and no sub-directories — only bookkeeping (``.cache``, README, config…).

    A directory holding in-progress LocalCode downloads (``.cache/localcode/*.part``)
    is never removed.
    """
    try:
        root = root.resolve()
        d = start.resolve()
    except OSError:
        return
    while d != root and d.is_relative_to(root):
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        if _has_partial_downloads(d):
            return
        meaningful = [e for e in entries if e.name not in _IGNORED_ENTRIES]
        if any(e.is_dir() or e.name.lower().endswith((".gguf", ".safetensors")) for e in meaningful):
            return
        shutil.rmtree(d, ignore_errors=True)
        d = d.parent


def _safe_dir_name(name: str) -> str:
    """A directory name that is safe to create under ``models_dir``."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip()).strip("-.")
    return cleaned[:80] or "model"


def _shares_subject(projector: str, stem: str) -> bool:
    """True when an ``mmproj`` file belongs to the model called ``stem``."""
    subject = re.sub(r"(?i)^mmproj[-_.]*", "", gguf_stem(projector))
    base = stem.lower()
    return not subject or subject.lower() in base or base in subject.lower()


def lmstudio_models_dirs() -> list[Path]:
    """LM Studio's model folders that exist on this machine."""
    home = Path.home()
    candidates: list[Path] = []
    try:
        s = json.loads((home / ".lmstudio" / "settings.json").read_text(encoding="utf-8"))
        folder = s.get("downloadsFolder") if isinstance(s, dict) else None
        if isinstance(folder, str) and folder:
            candidates.append(Path(folder).expanduser())
    except (OSError, ValueError):
        pass
    candidates += [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        try:
            r = str(c.resolve())
        except OSError:
            continue
        if r not in seen and c.is_dir():
            seen.add(r)
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class _Found:
    key: str
    format: str
    path: Path                       # gguf shard 1 / mlx dir
    files: list[Path]                # weight files
    repo_id: str | None
    source: str
    name: str                        # stem (gguf) or directory name (mlx)
    quant_hint: str | None = None
    mmproj: Path | None = None
    listing: list[str] = field(default_factory=list)
    size: int = 0
    mtime: float = 0.0
    fingerprint_files: list[Path] = field(default_factory=list)
    image_dir: bool = False                      # MLX/diffusers diffusion checkpoint
    companions: dict[str, str] = field(default_factory=dict)  # clip_l/clip_g/t5xxl/vae -> path


def _visible(p: Path) -> bool:
    return not p.name.startswith(".")


def _stat(p: Path) -> os.stat_result | None:
    try:
        return p.stat()
    except OSError:
        return None


def _projector_subject(p: Path) -> str:
    """``mmproj-Qwen3VL-8B-Instruct-F16.gguf`` → ``qwen3vl-8b-instruct``; ``mmproj-F16.gguf`` → ``""``."""
    stem = p.name[:-5] if p.name.lower().endswith(".gguf") else p.name
    q = parse_quant(p.name)
    if q:
        stem = re.sub(rf"[-._]?{re.escape(q)}$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?i)[-._]?mmproj[-._]?", "-", stem)
    return stem.strip("-_. ").lower()


def _pick_mmproj(candidates: list[Path], model_stem: str) -> Path | None:
    """Projector for a model: named after it > generic (``mmproj-F16``) > any; then F16 > BF16 > Q8_0."""
    if not candidates:
        return None
    stem_l = model_stem.lower()
    subjects = [(c, _projector_subject(c)) for c in candidates]
    named = [c for c, s in subjects if s and s in stem_l]
    generic = [c for c, s in subjects if not s]
    pool = named or generic or candidates

    def rank(p: Path) -> tuple[int, int]:
        q = (parse_quant(p.name) or "").upper()
        order = {"F16": 0, "BF16": 1, "Q8_0": 2, "F32": 4}.get(q, 3)
        st = _stat(p)
        return order, st.st_size if st else 0

    return sorted(pool, key=rank)[0]


def image_companions(entries: Iterable[Path]) -> dict[str, str]:
    """``{"clip_l": path, "t5xxl": path, "vae": path}`` from the files of a directory.

    Split FLUX/SD3 GGUF repos ship their text encoders and autoencoder next to
    the diffusion model (``clip_l.safetensors``, ``t5xxl-Q8_0.gguf``,
    ``ae.safetensors``); stable-diffusion.cpp needs their paths and the library
    must never list them as models (see ``ModelMeta.image_components``).
    """
    found: dict[str, str] = {}
    for path in sorted(entries, key=lambda p: p.name):
        if not path.is_file() or not is_image_companion(path.name):
            continue
        if path.suffix.lower() not in (".gguf", ".safetensors", ".ckpt", ".pt"):
            continue
        kind = image_component_kind(path.name)
        if kind and kind not in found:
            found[kind] = str(path)
    return found


def _scan_gguf_dir(directory: Path, rel_prefix: str, key_prefix: str, repo_id: str | None,
                   source: str, mmproj_parent: list[Path], taken: set[str]) -> list[_Found]:
    """GGUF models directly inside ``directory`` (non-recursive).

    Companion files of image models (CLIP/T5/VAE) are collected instead of being
    listed as models of their own.
    """
    try:
        entries = [p for p in directory.iterdir() if _visible(p)]
    except OSError:
        return []
    ggufs = [p for p in entries if p.is_file() and p.name.lower().endswith(".gguf")]
    projectors = [p for p in ggufs if is_mmproj(p.name)] or mmproj_parent
    companions = image_companions(entries)
    listing = [p.name for p in entries]
    groups: dict[str, list[tuple[int, int, Path]]] = {}
    for p in ggufs:
        if is_mmproj(p.name) or is_image_companion(p.name):
            continue
        shard = split_shard(p.name)
        if shard:
            groups.setdefault(shard[0], []).append((shard[1], shard[2], p))
        else:
            groups.setdefault(gguf_stem(p.name), []).append((1, 1, p))
    found: list[_Found] = []
    for stem, parts in sorted(groups.items()):
        parts.sort()
        count = parts[0][1]
        if count > 1:
            idxs = {i for i, _, _ in parts}
            if idxs != set(range(1, count + 1)) or any(c != count for _, c, _ in parts):
                log.debug("skipping incomplete split model %s/%s", directory, stem)
                continue
        files = [p for _, _, p in parts]
        stats = [_stat(p) for p in files]
        if any(s is None or s.st_size == 0 for s in stats):
            continue
        key = f"{key_prefix}/{stem}" if key_prefix else stem
        if key in taken and rel_prefix:
            key = f"{key_prefix}/{rel_prefix}/{stem}"
        if key in taken:
            continue
        taken.add(key)
        found.append(_Found(
            key=key, format="gguf", path=files[0], files=files, repo_id=repo_id, source=source,
            name=stem, quant_hint=parse_quant(files[0].name),
            mmproj=_pick_mmproj(projectors, stem),
            listing=listing + [p.name for p in projectors if p.parent != directory],
            size=sum(s.st_size for s in stats if s), mtime=max(s.st_mtime for s in stats if s),
            fingerprint_files=list(files),
            companions=dict(companions),
        ))
    return found


def _scan_mlx_dir(directory: Path, key: str, repo_id: str | None, source: str) -> _Found | None:
    config_path = directory / "config.json"
    if not config_path.is_file():
        return None
    complete, weights = mlx_weights_complete(directory)
    if not complete:
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or not is_mlx_model_dir(directory, config, repo_id):
        return None
    fp_files = [config_path] + weights + [
        directory / n for n in ("tokenizer_config.json", "chat_template.jinja", "chat_template.json")
        if (directory / n).exists()
    ]
    stats = [_stat(p) for p in weights]
    if any(s is None for s in stats):
        return None
    fstats = [s for s in (_stat(p) for p in fp_files) if s]
    try:
        listing = [p.name for p in directory.iterdir() if _visible(p)]
    except OSError:
        listing = []
    return _Found(
        key=key, format="mlx", path=directory, files=weights, repo_id=repo_id, source=source,
        name=directory.name, listing=listing,
        size=sum(s.st_size for s in stats if s), mtime=max((s.st_mtime for s in fstats), default=0.0),
        fingerprint_files=fp_files,
    )


def _scan_image_dir(directory: Path, key: str, repo_id: str | None, source: str) -> _Found | None:
    """An MLX/diffusers diffusion checkpoint (``transformer/`` + ``vae/`` + …).

    Its weights live in sub-directories, so :func:`_scan_mlx_dir` (top-level
    ``*.safetensors``) never matches; the model is one library entry whose path
    is the directory and whose size is every weight file below it.
    """
    if not is_image_model_dir(directory):
        return None
    weights = sorted(p for p in directory.glob("*/*.safetensors") if _visible(p))
    weights += sorted(p for p in directory.glob("*/*/*.safetensors") if _visible(p))
    weights += sorted(p for p in directory.glob("*.safetensors") if _visible(p))
    stats = [(p, _stat(p)) for p in weights]
    stats = [(p, st) for p, st in stats if st is not None]
    if not stats:
        return None
    fp_files = [p for p, _ in stats][:8]
    for name in ("config.json", "model_index.json", "transformer/config.json"):
        candidate = directory / name
        if candidate.exists():
            fp_files.append(candidate)
    try:
        listing = [p.name for p in directory.iterdir() if _visible(p)]
    except OSError:
        listing = []
    return _Found(
        key=key, format="mlx", path=directory, files=[p for p, _ in stats], repo_id=repo_id,
        source=source, name=directory.name, listing=listing,
        size=sum(st.st_size for _, st in stats),
        mtime=max(st.st_mtime for _, st in stats),
        fingerprint_files=fp_files, image_dir=True,
    )


def _scan_repo_dir(repo_dir: Path, repo_id: str | None, key_base: str, source: str, taken: set[str]) -> list[_Found]:
    """One ``<org>/<repo>`` directory: MLX model and/or GGUF files (plus 1 level of subfolders)."""
    out: list[_Found] = []
    mlx = _scan_mlx_dir(repo_dir, key_base, repo_id, source) or \
        _scan_image_dir(repo_dir, key_base, repo_id, source)
    if mlx and mlx.key not in taken:
        taken.add(mlx.key)
        out.append(mlx)
    try:
        root_projectors = [p for p in repo_dir.iterdir() if _visible(p) and p.is_file() and is_mmproj(p.name)]
    except OSError:
        return out
    out += _scan_gguf_dir(repo_dir, "", key_base, repo_id, source, [], taken)
    try:
        subdirs = sorted(p for p in repo_dir.iterdir() if _visible(p) and p.is_dir())
    except OSError:
        subdirs = []
    for sub in subdirs:  # quant subfolders, e.g. "BF16/", "Q4_K_M/"
        out += _scan_gguf_dir(sub, sub.name, key_base, repo_id, source, root_projectors, taken)
    return out


def _scan_layout_root(root: Path, source: str, taken: set[str]) -> list[_Found]:
    """``<root>/<org>/<repo>/`` layout (LocalCode and LM Studio)."""
    out: list[_Found] = []
    try:
        orgs = sorted(p for p in root.iterdir() if _visible(p))
    except OSError:
        return out
    loose = _scan_gguf_dir(root, "", "local", None, source, [], taken)  # files at the root
    out += loose
    for org in orgs:
        if not org.is_dir():
            continue
        # a model directly under the root (no org level)
        mlx = _scan_mlx_dir(org, f"local/{org.name}", None, source) or \
            _scan_image_dir(org, f"local/{org.name}", None, source)
        if mlx and mlx.key not in taken:
            taken.add(mlx.key)
            out.append(mlx)
            continue
        out += _scan_gguf_dir(org, "", f"local/{org.name}", None, source, [], taken)
        try:
            repos = sorted(p for p in org.iterdir() if _visible(p) and p.is_dir())
        except OSError:
            continue
        for repo in repos:
            repo_id = f"{org.name}/{repo.name}"
            out += _scan_repo_dir(repo, repo_id, repo_id, source, taken)
    return out


def _scan_hf_cache(cache_dir: Path | None, taken: set[str]) -> list[_Found]:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:  # pragma: no cover - hard dependency
        return []
    try:
        info = scan_cache_dir(cache_dir) if cache_dir else scan_cache_dir()
    except Exception as exc:  # CacheNotFound, permission errors…
        log.debug("HF cache not scanned: %s", exc)
        return []
    out: list[_Found] = []
    for repo in sorted(info.repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model" or not repo.revisions:
            continue
        revs = sorted(repo.revisions, key=lambda r: ("main" not in r.refs, -(r.last_modified or 0)))
        snapshot = Path(revs[0].snapshot_path)
        if not snapshot.is_dir():
            continue
        out += _scan_repo_dir(snapshot, repo.repo_id, repo.repo_id, "hf-cache", taken)
    return out


# --------------------------------------------------------------------------- #
# Metadata cache
# --------------------------------------------------------------------------- #

_META_FIELDS = {f.name for f in fields(ModelMeta)}


def _meta_from_dict(d: dict[str, Any]) -> ModelMeta:
    return ModelMeta(**{k: v for k, v in d.items() if k in _META_FIELDS})


def _fingerprint(found: _Found) -> list[Any]:
    size = 0
    mtime = 0.0
    for p in found.fingerprint_files:
        st = _stat(p)
        if st:
            size += st.st_size
            mtime = max(mtime, st.st_mtime)
    return [size, round(mtime, 3), len(found.fingerprint_files)]


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# ModelLibrary
# --------------------------------------------------------------------------- #


class ModelLibrary:
    """Inventory of loadable local models (see module docstring)."""

    def __init__(
        self,
        settings: Settings,
        *,
        lmstudio_dirs: Iterable[Path] | None = None,
        hf_cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self._lmstudio_dirs = [Path(p) for p in lmstudio_dirs] if lmstudio_dirs is not None else None
        self._hf_cache_dir = hf_cache_dir
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        self._models: dict[str, LocalModel] = {}
        self._scanned = False
        self.last_scan_at: float = 0.0
        self.scan_errors: list[dict[str, str]] = []

    # paths ----------------------------------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return Path(self.settings.models_dir).expanduser()

    @staticmethod
    def cache_path() -> Path:
        return data_dir() / "library-cache.json"

    @staticmethod
    def defaults_path() -> Path:
        return data_dir() / "model-defaults.json"

    # scanning -------------------------------------------------------------------------
    def _discover(self) -> list[_Found]:
        taken: set[str] = set()
        found: list[_Found] = []
        models_dir = self.models_dir
        if models_dir.is_dir():
            found += _scan_layout_root(models_dir, "localcode", taken)
        if self.settings.scan_lmstudio:
            dirs = self._lmstudio_dirs if self._lmstudio_dirs is not None else lmstudio_models_dirs()
            try:
                own = str(models_dir.resolve())
            except OSError:
                own = str(models_dir)
            for d in dirs:
                try:
                    if str(d.resolve()) == own or not d.is_dir():
                        continue
                except OSError:
                    continue
                found += _scan_layout_root(d, "lmstudio", taken)
        if self.settings.scan_hf_cache:
            found += _scan_hf_cache(self._hf_cache_dir, taken)
        return found

    def _load_cache(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.cache_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
            return {}
        entries = raw.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _read_meta(self, f: _Found) -> tuple[ModelMeta, bool]:
        """(meta, has_vision_tower) for a discovered model."""
        if f.image_dir:
            return read_image_dir_meta(f.path, f.repo_id), False
        if f.format == "gguf":
            return read_gguf_meta(f.path), False
        meta = read_mlx_meta(f.path)
        config = read_mlx_config(f.path)
        vision = isinstance(config.get("vision_config"), dict) and bool(config.get("vision_config"))
        return meta, vision

    def _build(self, f: _Found, meta: ModelMeta, vision_tower: bool,
              scan_cache: dict[str, dict[str, Any] | None] | None = None) -> LocalModel:
        quant = f.quant_hint or meta.quant
        if meta.image_family and f.companions:
            # Companion files found next to a split GGUF checkpoint (CLIP/T5/VAE).
            merged = dict(f.companions)
            merged.update(meta.image_components or {})
            meta.image_components = merged
        caps = detect_capabilities(
            meta,
            f.listing if f.mmproj else [x for x in f.listing if not is_mmproj(x)],
            config={"vision_config": {"present": True}} if vision_tower else None,
            name=f"{f.repo_id or ''} {f.name}",
        )
        return LocalModel(
            key=f.key,
            format=f.format,  # type: ignore[arg-type]
            path=str(f.path),
            display_name=display_name_for(f.name, quant),
            size_bytes=f.size,
            repo_id=f.repo_id,
            quant=quant,
            source=f.source,
            mmproj_path=str(f.mmproj) if f.mmproj and not meta.is_embedding else None,
            capabilities=caps,
            meta=meta,
            modified_at=f.mtime,
            scan=self._security_scan(f, scan_cache if scan_cache is not None else {}),
        )

    def _security_scan(self, f: _Found, scan_cache: dict[str, dict[str, Any] | None]) -> dict[str, Any] | None:
        """Static pickle/file-policy scan of the files next to this model (``security/scan.py``).

        Scans the directory a model's weight file(s) live in (the MLX directory
        itself, or the folder containing a GGUF shard) so a stray script or
        pickle sitting next to safe-format weights is still caught. Several
        ``LocalModel``s (GGUF quant variants, a vision projector) commonly
        share one directory, so results are cached per scan run by resolved
        path. Never raises: a scanner failure must not break a library scan,
        and shows up as ``scan=None`` (not scanned yet), not a false "clean".
        Honours ``settings.security.scan_downloads`` (default on); turning it
        off just means "not scanned", the same as a scanner error.
        """
        if not self.settings.security.scan_downloads:
            return None
        target = f.path if f.path.is_dir() else f.path.parent
        cache_key = str(target)
        if cache_key in scan_cache:
            return scan_cache[cache_key]
        try:
            report = _security_scan_path(target).to_dict()
        except Exception as exc:  # noqa: BLE001 - the scanner must never block a library scan
            log.warning("security scan failed for %s: %s", target, exc)
            report = None
        scan_cache[cache_key] = report
        return report

    def scan(self) -> list[LocalModel]:
        """Rescan all sources; returns the new model list (sorted by display name)."""
        with self._scan_lock:
            started = time.monotonic()
            found = self._discover()
            cache = self._load_cache()
            new_cache: dict[str, Any] = {}
            models: dict[str, LocalModel] = {}
            errors: list[dict[str, str]] = []
            #: one security scan per directory per run, even when several models share it
            #: (GGUF quant variants, a vision projector next to its base model).
            security_scan_cache: dict[str, dict[str, Any] | None] = {}
            for f in found:
                cache_key = str(f.path)
                fp = _fingerprint(f)
                entry = cache.get(cache_key)
                meta: ModelMeta | None = None
                vision = False
                if isinstance(entry, dict) and entry.get("fp") == fp and isinstance(entry.get("meta"), dict):
                    try:
                        meta = _meta_from_dict(entry["meta"])
                        vision = bool(entry.get("vision"))
                    except TypeError:
                        meta = None
                if meta is None:
                    try:
                        meta, vision = self._read_meta(f)
                    except (MetadataError, OSError, ValueError) as exc:
                        log.warning("cannot read metadata of %s: %s", f.path, exc)
                        errors.append({"path": str(f.path), "error": str(exc)})
                        continue
                new_cache[cache_key] = {"fp": fp, "meta": asdict(meta), "vision": vision}
                model = self._build(f, meta, vision, security_scan_cache)
                models[model.key] = model
            try:
                _atomic_write_json(self.cache_path(), {"version": CACHE_VERSION, "entries": new_cache})
            except OSError as exc:
                log.warning("cannot write library cache: %s", exc)
            with self._lock:
                self._models = models
                self._scanned = True
                self.last_scan_at = time.time()
                self.scan_errors = errors
            log.debug("library scan: %d models in %.2fs", len(models), time.monotonic() - started)
            return self.list()

    async def ascan(self) -> list[LocalModel]:
        return await asyncio.to_thread(self.scan)

    def _ensure_scanned(self) -> None:
        if not self._scanned:
            self.scan()

    # queries --------------------------------------------------------------------------
    def list(self) -> list[LocalModel]:
        self._ensure_scanned()
        with self._lock:
            models = list(self._models.values())
        return sorted(models, key=lambda m: (m.display_name.lower(), SOURCE_ORDER.get(m.source, 9), m.key))

    def get(self, key: str) -> LocalModel | None:
        self._ensure_scanned()
        with self._lock:
            return self._models.get(key)

    @staticmethod
    def _rank(m: LocalModel) -> tuple[Any, ...]:
        bq = base_quant(m.quant) or ""
        pref = _QUANT_PREFERENCE.index(bq) if bq in _QUANT_PREFERENCE else len(_QUANT_PREFERENCE)
        return (m.is_embedding, SOURCE_ORDER.get(m.source, 9), pref, len(m.key), m.key)

    def resolve(self, query: str) -> LocalModel | None:
        """Find a model by key, repo id or fuzzy name; chat models win over embedding ones."""
        q = (query or "").strip()
        if not q:
            return None
        models = self.list()
        ql = q.lower()

        def hay(m: LocalModel) -> str:
            return f"{m.key} {m.display_name} {m.repo_id or ''}".lower()

        tokens = [t for t in re.split(r"[\s·]+", ql) if t]
        groups = (
            [m for m in models if m.key == q],
            [m for m in models if m.key.lower() == ql],
            [m for m in models if (m.repo_id or "").lower() == ql],
            [m for m in models if ql in m.key.lower() or ql in m.display_name.lower()],
            [m for m in models if tokens and all(t in hay(m) for t in tokens)],
        )
        for group in groups:
            if group:
                return min(group, key=self._rank)
        return None

    # importing ------------------------------------------------------------------------
    def import_path(self, source: str | os.PathLike[str], *, name: str | None = None,
                    move: bool = False) -> LocalModel:
        """Add a model the person already has on disk to the library.

        ``source`` is a single ``.gguf`` file (its split shards and any
        ``mmproj`` next to it come along) or an MLX directory (``config.json``
        plus weights).  The files are copied into ``models_dir`` under
        ``local/<name>`` so that everything LocalCode can load lives in one
        place — the same place the uninstall button is allowed to delete from.
        Pass ``move=True`` to move instead of copy (useful for a 40 GB model).
        """
        origin = Path(source).expanduser()
        try:
            origin = origin.resolve(strict=True)
        except OSError as exc:
            raise LibraryError(f"Nie znaleziono pliku „{source}”.", "not_found") from exc

        if origin.is_dir():
            kind = "mlx"
            if not (origin / "config.json").exists():
                raise LibraryError(
                    f"Katalog „{origin.name}” nie wygląda na model MLX — brakuje config.json.",
                    "not_a_model")
            if not any(p.suffix in (".safetensors", ".npz") for p in origin.iterdir()):
                raise LibraryError(
                    f"Katalog „{origin.name}” nie zawiera wag (.safetensors).", "not_a_model")
            sources = [origin]
        elif origin.suffix.lower() == ".gguf":
            kind = "gguf"
            stem = gguf_stem(origin.name)
            sources = sorted(
                p for p in origin.parent.iterdir()
                if p.is_file() and p.name.lower().endswith(".gguf")
                and (gguf_stem(p.name) == stem or (is_mmproj(p.name) and _shares_subject(p.name, stem)))
            )
        else:
            raise LibraryError(
                "Obsługiwane są pliki .gguf albo katalogi z modelem MLX.", "unsupported")

        label = _safe_dir_name(name or (origin.name if origin.is_dir() else gguf_stem(origin.name)))
        target_dir = self.models_dir / "local" / label
        if target_dir.exists() and any(target_dir.iterdir()):
            raise LibraryError(
                f"W katalogu modeli jest już „local/{label}”. Zmień nazwę albo usuń tamten model.",
                "already_exists")

        with self._lock:
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                for item in sources:
                    if item.is_dir():
                        # The directory *is* the model, so its contents land in
                        # target_dir; copying it as a child would nest the model
                        # one level too deep for the scanner to recognise.
                        if move:
                            for child in list(item.iterdir()):
                                shutil.move(str(child), target_dir / child.name)
                            item.rmdir()
                        else:
                            shutil.copytree(item, target_dir, dirs_exist_ok=True)
                    elif move:
                        shutil.move(str(item), target_dir / item.name)
                    else:
                        shutil.copy2(item, target_dir / item.name)
            except OSError as exc:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise LibraryError(
                    f"Nie udało się skopiować modelu: {exc.strerror or exc}.", "io_error") from exc

        self.scan()
        model = next((m for m in self.list()
                      if Path(m.path).resolve().is_relative_to(target_dir.resolve())), None)
        if model is None:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise LibraryError(
                "Skopiowane pliki nie wyglądają na model, który LocalCode potrafi wczytać.",
                "not_a_model")
        return model

    # deletion -------------------------------------------------------------------------
    def delete(self, key: str) -> None:
        """Delete a LocalCode-managed model from disk (refuses LM Studio / HF cache models)."""
        model = self.get(key)
        if model is None:
            raise LibraryError(f"Nie znaleziono modelu „{key}”.", "not_found")
        if model.source != "localcode":
            where = {"lmstudio": "LM Studio", "hf-cache": "cache Hugging Face"}.get(model.source, model.source)
            raise LibraryError(
                f"Model „{model.display_name}” pochodzi z {where} — LocalCode go nie usuwa. "
                f"Usuń go w tamtym miejscu ({model.path}).",
                "not_owned",
            )
        root = self.models_dir.resolve()
        target = Path(model.path)
        try:
            resolved = target.resolve()
        except OSError as exc:
            raise LibraryError(f"Nie można usunąć modelu: {exc}.", "io_error") from exc
        if not resolved.is_relative_to(root) or resolved == root:
            raise LibraryError("Odmowa usunięcia: plik modelu leży poza katalogiem modeli LocalCode.", "not_owned")
        with self._lock:
            try:
                if model.format == "mlx":
                    shutil.rmtree(resolved)
                    parent = resolved.parent
                else:
                    parent = resolved.parent
                    stem = gguf_stem(resolved.name)
                    for p in list(parent.iterdir()):
                        if p.is_file() and p.name.lower().endswith(".gguf") and not is_mmproj(p.name) and gguf_stem(p.name) == stem:
                            p.unlink()
                            self._forget_hf_metadata(parent, p.name)
                    remaining = [p for p in parent.iterdir()
                                 if p.is_file() and p.name.lower().endswith(".gguf") and not is_mmproj(p.name)]
                    if not remaining:  # projectors are useless without a model
                        for p in list(parent.iterdir()):
                            if p.is_file() and is_mmproj(p.name):
                                p.unlink()
                self._prune_empty_dirs(parent, root)
            except OSError as exc:
                raise LibraryError(f"Nie udało się usunąć plików modelu: {exc.strerror or exc}.", "io_error") from exc
            self._models.pop(key, None)
            defaults = self._read_defaults()
            if defaults.pop(key, None) is not None:
                self._write_defaults(defaults)

    async def adelete(self, key: str) -> None:
        await asyncio.to_thread(self.delete, key)

    @staticmethod
    def _forget_hf_metadata(directory: Path, filename: str) -> None:
        # walk up to the repo dir holding `.cache/huggingface/download/<file>.metadata`
        for base in (directory, directory.parent):
            meta_dir = base / ".cache" / "huggingface" / "download"
            for suffix in (".metadata", ".lock"):
                p = meta_dir / (filename + suffix)
                try:
                    p.unlink()
                except OSError:
                    pass

    @staticmethod
    def _prune_empty_dirs(start: Path, root: Path) -> None:
        prune_empty_dirs(start, root)

    # load defaults --------------------------------------------------------------------
    def _read_defaults(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.defaults_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_defaults(self, data: dict[str, Any]) -> None:
        _atomic_write_json(self.defaults_path(), data)

    def load_defaults(self, key: str) -> dict[str, Any] | None:
        """Stored ``LoadConfigDTO`` (camelCase) for a model, or ``None``."""
        with self._lock:
            value = self._read_defaults().get(key)
        return dict(value) if isinstance(value, dict) else None

    def set_load_defaults(self, key: str, defaults: dict[str, Any] | None) -> dict[str, Any] | None:
        """Persist (or with ``None``/``{}`` clear) load defaults; returns what was stored."""
        if self.get(key) is None:
            raise LibraryError(f"Nie znaleziono modelu „{key}”.", "not_found")
        clean = self._validate_defaults(defaults) if defaults else None
        with self._lock:
            data = self._read_defaults()
            if clean:
                data[key] = clean
            else:
                data.pop(key, None)
            try:
                self._write_defaults(data)
            except OSError as exc:
                raise LibraryError(f"Nie udało się zapisać ustawień modelu: {exc.strerror or exc}.", "io_error") from exc
        return clean

    @staticmethod
    def _validate_defaults(d: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(d, dict):
            raise LibraryError("Ustawienia ładowania muszą być obiektem JSON.", "invalid")
        unknown = set(d) - _LOAD_DEFAULT_KEYS
        if unknown:
            raise LibraryError(f"Nieznane ustawienia ładowania: {', '.join(sorted(unknown))}.", "invalid")
        out: dict[str, Any] = {}

        def as_int(name: str, lo: int, hi: int) -> None:
            v = d.get(name)
            if v is None:
                return
            if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
                raise LibraryError(f"Wartość „{name}” musi być liczbą całkowitą.", "invalid")
            if not lo <= int(v) <= hi:
                raise LibraryError(f"Wartość „{name}” musi być w zakresie {lo}–{hi}.", "invalid")
            out[name] = int(v)

        as_int("nCtx", 256, 4_194_304)
        as_int("nParallel", 1, 64)
        as_int("gpuLayers", -1, 100_000)
        if d.get("flashAttn") is not None:
            if d["flashAttn"] not in ("on", "off", "auto"):
                raise LibraryError("„flashAttn” musi mieć wartość on, off lub auto.", "invalid")
            out["flashAttn"] = d["flashAttn"]
        if d.get("kvCacheType") is not None:
            if d["kvCacheType"] not in ("f16", "q8_0", "q4_0", "8bit", "4bit"):
                raise LibraryError("„kvCacheType” musi mieć wartość f16, q8_0, q4_0, 8bit lub 4bit.", "invalid")
            out["kvCacheType"] = d["kvCacheType"]
        if d.get("embedding") is not None:
            if not isinstance(d["embedding"], bool):
                raise LibraryError("„embedding” musi mieć wartość true lub false.", "invalid")
            out["embedding"] = d["embedding"]
        if d.get("effort") is not None:
            if d["effort"] not in EFFORT_VALUES:
                raise LibraryError(
                    f"„effort” musi mieć jedną z wartości: {', '.join(EFFORT_VALUES)}.", "invalid")
            out["effort"] = d["effort"]
        return out

    # misc -----------------------------------------------------------------------------
    def chat_template(self, key: str) -> str | None:
        """Chat template of a model (from metadata; MLX falls back to the files)."""
        m = self.get(key)
        if m is None:
            return None
        if m.meta.chat_template:
            return m.meta.chat_template
        if m.format == "mlx":
            return read_hf_chat_template(m.path)
        return None
