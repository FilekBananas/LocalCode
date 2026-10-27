"""Engine contract: one loaded model instance (a subprocess) behind a uniform API.

Both engines run the model in a separate process so that unloading reliably
frees memory and so that the UI/agent process never blocks on the GPU:

* ``LlamaServerEngine`` (``runtime/llama_engine.py``) — spawns Homebrew's
  ``llama-server`` for GGUF files.
* ``MlxEngine`` (``runtime/mlx_engine.py``) — spawns
  ``python -m localcode.runtime.mlx_worker`` which serves the same small HTTP
  surface on top of ``mlx_lm``.

Chat templating happens *inside* the engine (the engine has the tokenizer /
template), tool-call parsing happens *outside* (``runtime/toolcalls.py``), so
the agent and the chatbot get identical behaviour on both formats.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator

from localcode.hub.types import LocalModel


@dataclass
class LoadConfig:
    n_ctx: int                       # context PER SLOT the caller wants to use
    n_parallel: int = 1              # concurrent request slots in this instance
    gpu_layers: int = -1             # gguf only; -1 = all
    flash_attn: str = "auto"         # gguf only
    kv_cache_type: str = "f16"       # gguf: f16|q8_0|q4_0; mlx: f16|8bit|4bit
    embedding: bool = False          # start in embedding mode (GGUF embedding models)
    extra_args: list[str] = field(default_factory=list)
    #: Explicit user override for a model whose security scan came back
    #: "dangerous" (see localcode/security/scan.py and RuntimeManager._start_instance).
    #: False everywhere except an explicit "load anyway" from the UI/CLI/API.
    allow_dangerous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "nCtx": self.n_ctx,
            "nParallel": self.n_parallel,
            "gpuLayers": self.gpu_layers,
            "flashAttn": self.flash_attn,
            "kvCacheType": self.kv_cache_type,
            "embedding": self.embedding,
            "allowDangerous": self.allow_dangerous,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], base: "LoadConfig | None" = None) -> "LoadConfig":
        b = base or cls(n_ctx=8192)
        return cls(
            n_ctx=int(d.get("nCtx", b.n_ctx)),
            n_parallel=int(d.get("nParallel", b.n_parallel)),
            gpu_layers=int(d.get("gpuLayers", b.gpu_layers)),
            flash_attn=str(d.get("flashAttn", b.flash_attn)),
            kv_cache_type=str(d.get("kvCacheType", b.kv_cache_type)),
            embedding=bool(d.get("embedding", b.embedding)),
            extra_args=list(b.extra_args),
            allow_dangerous=bool(d.get("allowDangerous", b.allow_dangerous)),
        )


@dataclass
class SamplingParams:
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.05
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    #: Base64 images for the media markers in the prompt, in order (vision
    #: models only — see ``HttpEngine.media_marker``).
    images: list[str] = field(default_factory=list)
    #: Which requests share a provider-side prompt cache (one agent session,
    #: one chat): remote OpenAI-style providers route on it (DeepInfra's and
    #: OpenAI's ``prompt_cache_key``). Local engines ignore it.
    cache_key: str | None = None


@dataclass
class GenStats:
    prompt_tokens: int = 0
    cached_tokens: int = 0           # prompt tokens reused from the KV/prefix cache
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "promptTokens": self.prompt_tokens,
            "cachedTokens": self.cached_tokens,
            "completionTokens": self.completion_tokens,
            "promptTps": round(self.prompt_tps, 1),
            "genTps": round(self.gen_tps, 1),
            "totalMs": round(self.total_ms),
        }


@dataclass
class CompletionChunk:
    text: str = ""
    done: bool = False
    finish_reason: str | None = None  # "stop" | "length" | "cancelled" | "error"
    stats: GenStats | None = None     # present on the final chunk


@dataclass
class TemplateInfo:
    """What the model's chat template can do (filled by the engine after start)."""

    supports_tools: bool = False         # template renders a `tools` list natively
    supports_tool_role: bool = False     # template understands role="tool" messages
    supports_system: bool = True
    supports_thinking: bool = False      # emits/accepts <think> or similar reasoning blocks
    thinking_kwarg: str | None = None    # e.g. "enable_thinking" (Qwen3) when the template has a switch
    tool_format: str | None = None       # family hint for runtime.toolcalls (see detect_tool_format)
    stop_strings: list[str] = field(default_factory=list)
    # Reasoning-effort control, when the template exposes one. ``effort_levels``
    # is ordered weakest → strongest and holds a subset of
    # ("off", "on", "low", "medium", "high"); empty means "no control, hide the
    # UI". ``effort_kwarg`` is the template variable that carries it
    # ("reasoning_effort", "enable_thinking", "thinking_budget" …).
    effort_levels: list[str] = field(default_factory=list)
    effort_kwarg: str | None = None


class EngineError(RuntimeError):
    """User-presentable engine failure (message is Polish, safe to show)."""


class Engine(abc.ABC):
    """One running model instance. All methods are coroutine-safe; ``complete``
    may be called concurrently up to ``config.n_parallel`` times."""

    format: str = ""

    def __init__(self, instance_id: str, model: LocalModel, config: LoadConfig) -> None:
        self.instance_id = instance_id
        self.model = model
        self.config = config
        self.template_info = TemplateInfo()

    # lifecycle ------------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None:
        """Spawn the process, wait until healthy, fill ``template_info``.
        Raises ``EngineError`` with a useful message (tail of the log) on failure."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Terminate the process (SIGTERM, then SIGKILL after a grace period)."""

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
    def n_ctx(self) -> int:
        """Usable context per slot (what the context manager budgets against)."""
        return self.config.n_ctx

    # inference -----------------------------------------------------------
    @abc.abstractmethod
    async def apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = True,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render OpenAI-style messages (+ optional OpenAI tool schemas) with the
        model's own chat template. Raises ``EngineError`` if the template fails."""

    @abc.abstractmethod
    async def tokenize(self, text: str) -> list[int]:
        """Token ids for ``text`` (no BOS added)."""

    async def count_tokens(self, text: str) -> int:
        return len(await self.tokenize(text))

    @abc.abstractmethod
    def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        """Stream a raw completion of an already-templated prompt.

        Must reuse the KV/prefix cache for a shared prompt prefix. Cancelling the
        consuming task (or ``aclose()`` on the iterator) must stop generation in
        the engine promptly. The last chunk has ``done=True`` and ``stats``."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise EngineError("Ten model nie obsługuje embeddingów.")

    async def memory_bytes(self) -> int | None:
        """Measured resident memory of the instance, if the engine can tell."""
        return None
