"""Model descriptions shared by the hub (search/download/library) and the runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ModelFormat = Literal["gguf", "mlx"]
Capability = Literal["tools", "thinking", "vision", "embedding", "image"]
#: ``ModelMeta.image_family`` values (see ``localcode/images/``).
ImageFamily = Literal["flux", "flux2", "qwen-image", "sd1", "sdxl", "sd3", "unknown"]


@dataclass
class ModelMeta:
    """Architecture facts read from GGUF metadata or MLX ``config.json``.

    Every field is optional: unknown values fall back to conservative estimates
    in ``runtime.memory``.
    """

    arch: str | None = None               # "qwen3", "llama", "gemma3", ...
    params_b: float | None = None         # parameters in billions (total, incl. MoE experts)
    active_params_b: float | None = None  # MoE active parameters, when known
    context_length: int | None = None     # native max context (n_ctx_train / max_position_embeddings)
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None
    hidden_size: int | None = None
    sliding_window: int | None = None     # SWA window (layers that do not need full-context KV)
    n_full_attn_layers: int | None = None  # hybrid models: layers that keep a full KV cache
    vocab_size: int | None = None
    chat_template: str | None = None      # Jinja source (GGUF tokenizer.chat_template or HF template)
    bos_token: str | None = None
    eos_token: str | None = None
    quant: str | None = None              # "Q4_K_M", "4bit", "8bit", "bf16"...
    is_embedding: bool = False
    # Image models (text-to-image; empty for language models):
    image_family: str | None = None       # flux | flux2 | qwen-image | sd1 | sdxl | sd3 | unknown
    image_components: dict[str, str] = field(default_factory=dict)
    # companion files of a split GGUF setup: clip_l / clip_g / t5xxl / vae / unet -> path

    @property
    def is_image(self) -> bool:
        return bool(self.image_family)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("chat_template", None)      # large; fetched separately when needed
        return d


@dataclass
class LocalModel:
    """A model that exists on disk and can be loaded.

    ``key`` is the stable identifier used everywhere (settings, API, UI):
      * MLX:  ``"<org>/<repo>"``                       e.g. ``mlx-community/Qwen3-4B-Instruct-2507-4bit``
      * GGUF: ``"<org>/<repo>/<file stem without shard suffix>"``
              e.g. ``unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M``
    Models found outside a repo layout use ``"local/<dirname>[/<stem>]"``.
    """

    key: str
    format: ModelFormat
    path: str                               # gguf: first (or only) .gguf shard; mlx: model directory
    display_name: str
    size_bytes: int
    repo_id: str | None = None
    quant: str | None = None
    source: str = "localcode"               # localcode | lmstudio | hf-cache | custom
    mmproj_path: str | None = None          # GGUF vision projector next to the model, if any
    capabilities: list[str] = field(default_factory=list)
    meta: ModelMeta = field(default_factory=ModelMeta)
    modified_at: float = 0.0                # unix seconds
    #: localcode.security.scan.ScanReport.to_dict(), computed by ModelLibrary.scan();
    #: None means "not scanned yet" (e.g. a scan error), not "known safe".
    scan: dict[str, Any] | None = None

    @property
    def is_embedding(self) -> bool:
        return "embedding" in self.capabilities or self.meta.is_embedding

    @property
    def is_image(self) -> bool:
        """Text-to-image model (loaded by ``localcode.images``, not by the runtime)."""
        return "image" in self.capabilities or self.meta.is_image

    def to_dict(self) -> dict[str, Any]:
        """camelCase DTO for the HTTP API / Swift client."""
        return {
            "key": self.key,
            "format": self.format,
            "path": self.path,
            "displayName": self.display_name,
            "sizeBytes": self.size_bytes,
            "repo": self.repo_id,
            "quant": self.quant,
            "source": self.source,
            "capabilities": list(self.capabilities),
            "arch": self.meta.arch,
            "paramsB": self.meta.params_b,
            "contextLength": self.meta.context_length,
            "modifiedAt": self.modified_at,
            "hasVision": self.mmproj_path is not None or "vision" in self.capabilities,
            "imageFamily": self.meta.image_family,
            "scan": self.scan,
        }


@dataclass
class MemoryEstimate:
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int
    total_bytes: int
    n_ctx: int
    n_parallel: int
    fits: Literal["yes", "tight", "no"]   # vs. the runtime RAM budget and free system memory
    message: str = ""                      # Polish, user-facing explanation

    def to_dict(self) -> dict[str, Any]:
        return {
            "weightsBytes": self.weights_bytes,
            "kvCacheBytes": self.kv_cache_bytes,
            "overheadBytes": self.overhead_bytes,
            "totalBytes": self.total_bytes,
            "nCtx": self.n_ctx,
            "nParallel": self.n_parallel,
            "fits": self.fits,
            "message": self.message,
        }
