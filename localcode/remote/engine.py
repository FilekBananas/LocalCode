"""A remote model behind the local :class:`~localcode.runtime.base.Engine` contract.

Everything above the runtime — the chat service, the agent loop, compaction,
the tool-call parser — talks to an ``Engine``: render a conversation into a
prompt, stream a completion of that prompt.  A cloud API works the other way
round: it takes the conversation as structured messages and answers with
structured tool calls.

The adapter is deliberately literal about that gap:

* :meth:`RemoteEngine.apply_template` does not render anything — it returns the
  conversation as JSON.  That *is* the prompt for this engine, and the callers
  never look inside a prompt.
* :meth:`RemoteEngine.complete` parses it back, calls the provider, streams the
  text, and re-encodes any tool calls in the ``<tool_call>`` form the local
  parser already understands (``runtime/toolcalls.py``), so the agent works the
  same whether the model runs on this Mac or in somebody's data centre.

Nothing here loads weights or reserves memory, so remote models never take part
in the RAM budget or the eviction policy.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from localcode.remote.providers import (
    PROVIDER_LABELS,
    RemoteClient,
    RemoteError,
    parse_key,
)
from localcode.runtime.base import (
    CompletionChunk,
    Engine,
    EngineError,
    GenStats,
    LoadConfig,
    SamplingParams,
    TemplateInfo,
)
from localcode.hub.types import LocalModel, ModelMeta

#: Context assumed when a provider does not say (large enough not to compact a
#: conversation the model would have accepted).
DEFAULT_REMOTE_CTX = 128_000
#: Rough characters per token, for the estimate that replaces a tokenizer.
CHARS_PER_TOKEN = 3.6


def remote_model(key: str, *, display_name: str = "", context_length: int | None = None,
                 description: str | None = None,
                 supports_tools: bool | None = None,
                 supports_vision: bool = False) -> LocalModel:
    """A :class:`LocalModel` row describing a model that is not on this disk."""
    parsed = parse_key(key)
    if parsed is None:
        raise EngineError(f"„{key}” nie jest modelem zdalnym.")
    provider, model_id = parsed
    return LocalModel(
        key=key,
        format="remote",
        path="",
        display_name=display_name or model_id,
        size_bytes=0,
        repo_id=model_id,
        source="remote",
        # "tools" unless the provider says otherwise: most list no capabilities
        # at all, and a model that cannot call a tool cannot drive the agent —
        # better to find out from the gateway than from a failed turn.
        capabilities=(["tools", "remote"] if supports_tools is not False else ["remote"])
        + (["vision"] if supports_vision else []),
        meta=ModelMeta(arch=provider, context_length=context_length or DEFAULT_REMOTE_CTX),
    )


class RemoteEngine(Engine):
    """One provider + one model id, wearing the local engine's interface."""

    format = "remote"

    def __init__(self, instance_id: str, model: LocalModel, config: LoadConfig,
                 client: RemoteClient) -> None:
        super().__init__(instance_id, model, config)
        self.client = client
        parsed = parse_key(model.key)
        if parsed is None:
            raise EngineError(f"„{model.key}” nie jest modelem zdalnym.")
        self.provider, self.model_id = parsed
        self._running = False
        self.template_info = TemplateInfo(
            supports_tools=True,
            supports_tool_role=True,
            supports_system=True,
            # The provider decides how much it thinks; there is no template
            # switch to offer, so the effort picker stays hidden.
            supports_thinking=False,
            tool_format="hermes",
        )

    # lifecycle ---------------------------------------------------------------
    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def supports_images(self) -> bool:
        return "vision" in (self.model.capabilities or [])

    @property
    def n_ctx(self) -> int:
        native = self.model.meta.context_length or DEFAULT_REMOTE_CTX
        wanted = self.config.n_ctx or native
        return min(int(wanted), int(native))

    # inference ---------------------------------------------------------------
    async def apply_template(self, messages: list[dict[str, Any]],
                             tools: list[dict[str, Any]] | None = None, *,
                             add_generation_prompt: bool = True,
                             template_kwargs: dict[str, Any] | None = None) -> str:
        """The conversation itself, as JSON — see the module docstring."""
        return json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)

    async def tokenize(self, text: str) -> list[int]:
        """An estimate: the providers charge for tokens but do not lend a tokenizer."""
        return [0] * max(1, int(len(text or "") / CHARS_PER_TOKEN))

    async def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        try:
            payload = json.loads(prompt or "{}")
        except ValueError as exc:
            raise EngineError("Model zdalny dostał prompt, którego nie rozumie.") from exc
        messages = payload.get("messages") or []
        tools = payload.get("tools") or None

        started = time.monotonic()
        text_chars = 0
        stats = GenStats()
        try:
            stream = self.client.stream(
                messages, model=self.model_id, tools=tools,
                max_tokens=_output_cap(params.max_tokens), temperature=params.temperature,
                stop=params.stop or None,
                **({"cache_key": params.cache_key} if getattr(params, "cache_key", None) else {}))
            async for chunk in stream:
                if chunk.text:
                    text_chars += len(chunk.text)
                    yield CompletionChunk(text=chunk.text)
                if not chunk.done:
                    continue
                tail = _tool_calls_as_text(chunk.tool_calls)
                if tail:
                    yield CompletionChunk(text=tail)
                elapsed = max(time.monotonic() - started, 1e-6)
                stats = GenStats(
                    prompt_tokens=chunk.prompt_tokens,
                    cached_tokens=chunk.cached_tokens,
                    completion_tokens=chunk.completion_tokens
                    or int(text_chars / CHARS_PER_TOKEN),
                    gen_tps=(chunk.completion_tokens or int(text_chars / CHARS_PER_TOKEN)) / elapsed,
                    total_ms=elapsed * 1000,
                )
                yield CompletionChunk(done=True, finish_reason=chunk.finish_reason or "stop",
                                      stats=stats)
        except RemoteError as exc:
            raise EngineError(exc.message) from exc

    async def memory_bytes(self) -> int | None:
        return 0        # it runs somewhere else; it costs this Mac nothing


#: Above this an output limit means "as much as it will write" (the chat's
#: "bez limitu" asks for the whole free window, up to a million tokens): no
#: provider writes that much in one answer and some refuse a limit beyond the
#: model's own, so it is left to the provider instead.
EXPLICIT_OUTPUT_CAP = 32_000


def _output_cap(max_tokens: int) -> int | None:
    return max_tokens if 0 < max_tokens <= EXPLICIT_OUTPUT_CAP else None


def _tool_calls_as_text(calls: list[dict[str, Any]]) -> str:
    """Structured tool calls → the ``<tool_call>`` text the local parser reads.

    Re-encoding rather than inventing: the provider told us the name and the
    arguments, and this is the shape ``runtime/toolcalls.py`` already parses for
    the hermes family, so one code path handles local and remote models alike.
    """
    blocks: list[str] = []
    for call in calls or []:
        name = str(call.get("name") or "").strip()
        if not name:
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                arguments = {"input": arguments}
        blocks.append("<tool_call>\n"
                      + json.dumps({"name": name, "arguments": arguments or {}},
                                   ensure_ascii=False)
                      + "\n</tool_call>")
    return ("\n" + "\n".join(blocks)) if blocks else ""


class RemoteLease:
    """What :meth:`RuntimeManager.acquire` hands back for a remote model.

    It quacks like :class:`~localcode.runtime.manager.Lease` — ``engine``,
    ``n_ctx``, ``record_stats``, ``release`` and the async context manager — but
    there is no instance, no slot and no memory to account for, so releasing it
    does nothing beyond closing the engine.
    """

    __slots__ = ("engine", "purpose", "_released")

    def __init__(self, engine: RemoteEngine, purpose: str = "chat") -> None:
        self.engine = engine
        self.purpose = purpose
        self._released = False

    @property
    def n_ctx(self) -> int:
        return self.engine.n_ctx

    @property
    def model(self) -> LocalModel:
        return self.engine.model

    @property
    def instance_id(self) -> str:
        return self.engine.instance_id

    def record_stats(self, stats: Any) -> None:
        return None

    def release(self) -> None:
        self._released = True

    async def aclose(self) -> None:
        self.release()

    async def __aenter__(self) -> "RemoteLease":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        label = PROVIDER_LABELS.get(self.engine.provider, self.engine.provider)
        return f"<RemoteLease {label}:{self.engine.model_id}>"
