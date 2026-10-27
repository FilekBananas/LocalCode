"""Fakes for the chat backend tests.

The runtime, formatter and stream splitter are written concurrently by other
engineers, so the tests drive :class:`ChatService` through small fakes that
honour the contracts in ``docs/ARCHITECTURE.md`` / ``localcode/runtime/base.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

from localcode.chat.repo import ChatRepo
from localcode.chat.service import BasicFormatter, ChatService, ThinkSplitter
from localcode.config import Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.base import CompletionChunk, EngineError, GenStats, TemplateInfo

CHAT_KEY = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
CHAT_KEY_2 = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
EMBED_KEY = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0"


def make_model(key: str, *, name: str | None = None, caps: list[str] | None = None, ctx: int = 32768,
               fmt: str = "gguf", embedding: bool = False) -> LocalModel:
    return LocalModel(
        key=key, format=fmt, path=f"/models/{key}", display_name=name or key.split("/")[-1],
        size_bytes=500_000_000, repo_id="/".join(key.split("/")[:2]), quant="Q4_K_M",
        capabilities=caps or [], meta=ModelMeta(arch="qwen3", params_b=0.6, context_length=ctx,
                                                is_embedding=embedding),
    )


class FakeEngine:
    """Chat-template + streaming stand-in for an ``Engine``.

    ``script`` is a list of text deltas, or a callable ``(prompt, params) -> list``.
    ``thinking_prefix``: the rendered generation prompt ends with ``<think>\\n``
    when thinking is requested (Qwen3-style templates).
    """

    format = "fake"

    def __init__(self, script: list[str] | Callable[[str, Any], list[str]] | None = None, *, n_ctx: int = 8192,
                 thinking_kwarg: str | None = "enable_thinking", thinking_prefix: bool = False,
                 stop: list[str] | None = None, delay: float = 0.0, fail: BaseException | None = None,
                 fail_times: int = 1, finish_reason: str = "stop", chars_per_token: int = 4) -> None:
        self.instance_id = "fake-1"
        self._n_ctx = n_ctx
        self.script = script if script is not None else ["Cześć", "! Jak ", "mogę pomóc?"]
        self.template_info = TemplateInfo(supports_system=True, supports_thinking=thinking_kwarg is not None,
                                          thinking_kwarg=thinking_kwarg, stop_strings=stop or ["<|im_end|>"],
                                          effort_kwarg=thinking_kwarg,
                                          effort_levels=["off", "on"] if thinking_kwarg else [])
        self.thinking_prefix = thinking_prefix
        self.delay = delay
        self.fail = fail
        self.fail_times = fail_times
        self.finish_reason = finish_reason
        self.chars_per_token = chars_per_token
        self.prompts: list[str] = []
        self.params: list[Any] = []
        self.render_kwargs: list[Any] = []
        self.closed = 0
        self.cancelled = 0
        self.embed_calls: list[list[str]] = []

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    async def apply_template(self, messages, tools=None, *, add_generation_prompt=True, template_kwargs=None):
        self.render_kwargs.append(template_kwargs)
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if self.thinking_prefix and (template_kwargs or {}).get("enable_thinking", True):
                out += "<think>\n"
        return out

    async def tokenize(self, text: str) -> list[int]:
        return list(range(await self.count_tokens(text)))

    async def count_tokens(self, text: str) -> int:
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token

    def complete(self, prompt, params) -> AsyncIterator[CompletionChunk]:
        self.prompts.append(prompt)
        self.params.append(params)
        return self._stream(prompt, params)

    async def _stream(self, prompt, params):
        try:
            if self.fail is not None and self.fail_times > 0:
                self.fail_times -= 1
                raise self.fail
            pieces = self.script(prompt, params) if callable(self.script) else list(self.script)
            produced = 0
            for piece in pieces:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if produced >= params.max_tokens:
                    break
                produced += 1
                yield CompletionChunk(text=piece)
            stats = GenStats(prompt_tokens=await self.count_tokens(prompt), completion_tokens=produced)
            yield CompletionChunk(done=True, finish_reason=self.finish_reason, stats=stats)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.closed += 1

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [bag_of_words(t) for t in texts]


VOCAB = ["kot", "pies", "ryba", "dom", "auto", "morze", "góry", "las", "python", "sqlite",
         "search_query", "search_document"]


def bag_of_words(text: str) -> list[float]:
    low = text.lower()
    vec = [float(low.count(w)) for w in VOCAB]
    if not any(vec):
        vec[-1] = 0.001
    return vec


class FakeLease:
    def __init__(self, runtime: "FakeRuntime", engine: FakeEngine) -> None:
        self.runtime = runtime
        self.engine = engine

    async def __aenter__(self) -> "FakeLease":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.runtime.active -= 1
        self.runtime.released += 1


class FakeRuntime:
    """``acquire`` is an ``async def`` returning an async-context-manager Lease."""

    def __init__(self, engines: dict[str, FakeEngine], *, acquire_delay: float = 0.0,
                 acquire_error: BaseException | None = None) -> None:
        self.engines = engines
        self.acquire_delay = acquire_delay
        self.acquire_error = acquire_error
        self.active = 0
        self.released = 0
        self.acquired: list[tuple[str, str]] = []

    async def acquire(self, model_key: str, *, n_ctx: int | None = None, purpose: str = "chat") -> FakeLease:
        if self.acquire_delay:
            await asyncio.sleep(self.acquire_delay)
        if self.acquire_error is not None:
            raise self.acquire_error
        if model_key not in self.engines:
            raise EngineError(f"Brak modelu {model_key}")
        self.acquired.append((model_key, purpose))
        self.active += 1
        return FakeLease(self, self.engines[model_key])


class FakeLibrary:
    def __init__(self, models: list[LocalModel]) -> None:
        self.models = models
        self.scans = 0

    def list(self) -> list[LocalModel]:
        return list(self.models)

    def get(self, key: str) -> LocalModel | None:
        return next((m for m in self.models if m.key == key), None)

    def resolve(self, query: str) -> LocalModel | None:
        q = query.lower()
        return next((m for m in self.models if q in m.key.lower() or q == m.display_name.lower()), None)

    def scan(self) -> list[LocalModel]:
        self.scans += 1
        return list(self.models)


async def collect(agen: AsyncIterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame async for frame in agen]


def types_of(frames: list[dict[str, Any]]) -> list[str]:
    return [f["type"] for f in frames]


def make_service(repo: ChatRepo, runtime: FakeRuntime, library: FakeLibrary, config: Settings,
                 **kwargs) -> ChatService:
    kwargs.setdefault("status_delay", 0.05)
    kwargs.setdefault("title_timeout", 2.0)
    return ChatService(repo, runtime, library, lambda: config, formatter_factory=BasicFormatter,
                       splitter_factory=lambda fmt: ThinkSplitter(fmt), **kwargs)
