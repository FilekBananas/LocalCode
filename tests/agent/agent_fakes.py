"""Test doubles for the coding-agent loop: a scripted engine, runtime and tools.

``ScriptedEngine`` replays a list of model outputs (raw text, exactly as a model
would emit it, including ``<tool_call>`` markup) and records every prompt it was
given, so tests can assert on prompt stability and on what the model saw.
``FakeRuntime`` hands it out through a lease that counts how long it is held —
which is how the tests prove that tool calls run *outside* the lease.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Sequence

from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolResult
from localcode.hub.types import LocalModel
from localcode.runtime.base import (
    CompletionChunk,
    Engine,
    EngineError,
    GenStats,
    LoadConfig,
    SamplingParams,
    TemplateInfo,
)

COMPACTION_MARKER = "You compress the history"
#: Auto mode's local guard (``agent/guard.py``) — answered from ``guard_replies``.
GUARD_MARKER = "You are a strict command safety checker"
#: What the guard says when a test did not script it: approve, so the reviewer decides.
GUARD_OK = "TAK\nCo robi: Polecenie w projekcie.\nPowód: Zwykła praca w projekcie."


def fake_model(key: str = "test/model", fmt: str = "gguf") -> LocalModel:
    return LocalModel(key=key, format=fmt, path=f"/tmp/{key.replace('/', '_')}.gguf",
                      display_name=key.split("/")[-1], size_bytes=1_000_000)


@dataclass
class Reply:
    """One scripted model answer."""

    text: str = ""
    error: BaseException | None = None      # raised instead of generating
    chunk_delay: float = 0.0                # seconds between chunks
    stall_after: int | None = None          # stop yielding after N chunks and wait forever
    completion_tokens: int = 12

    @classmethod
    def coerce(cls, item: "Reply | str | BaseException") -> "Reply":
        if isinstance(item, Reply):
            return item
        if isinstance(item, BaseException):
            return cls(error=item)
        return cls(text=str(item))


def tool_call_text(name: str, arguments: dict[str, Any], *, prefix: str = "") -> str:
    """Raw model output containing one hermes-style tool call."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{prefix}<tool_call>\n{payload}\n</tool_call>"


class ScriptedEngine(Engine):
    """An ``Engine`` that replays scripted outputs and renders a simple template."""

    format = "gguf"

    def __init__(
        self,
        replies: Sequence[Reply | str | BaseException] = (),
        *,
        n_ctx: int = 8192,
        model: LocalModel | None = None,
        instance_id: str = "scripted-1",
        supports_tools: bool = True,
        supports_tool_role: bool = True,
        supports_thinking: bool = False,
        tool_format: str = "hermes",
        chunk_size: int = 16,
        summary_text: str = ("Goal: keep working\nDone so far: earlier steps\nFiles touched: none\n"
                             "Key facts/decisions: none\nCurrent state/errors: none\nNext steps: continue\n"
                             "Todo: none"),
        chars_per_token: int = 4,
    ) -> None:
        super().__init__(instance_id, model or fake_model(), LoadConfig(n_ctx=n_ctx))
        self.template_info = TemplateInfo(
            supports_tools=supports_tools,
            supports_tool_role=supports_tool_role,
            supports_system=True,
            supports_thinking=supports_thinking,
            thinking_kwarg="enable_thinking" if supports_thinking else None,
            effort_kwarg="enable_thinking" if supports_thinking else None,
            effort_levels=["off", "on"] if supports_thinking else [],
            tool_format=tool_format,
            stop_strings=["<|end|>"],
        )
        self.replies: list[Reply] = [Reply.coerce(r) for r in replies]
        self.prompts: list[str] = []
        self.template_kwargs: list[dict[str, Any]] = []
        self.sampling: list[SamplingParams] = []
        self.summary_prompts: list[str] = []
        self.summary_text = summary_text
        #: Auto mode's guard is answered from here (``GUARD_OK`` when empty) and
        #: never takes a scripted reply — its prompts land in ``guard_prompts``.
        self.guard_replies: list[Reply] = []
        self.guard_prompts: list[str] = []
        self.guard_sampling: list[SamplingParams] = []
        self.started = False
        self.stopped = False
        self.cancelled = 0
        self._chunk_size = chunk_size
        self._chars_per_token = chars_per_token

    # lifecycle -------------------------------------------------------------
    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    @property
    def running(self) -> bool:
        return not self.stopped

    # templating ------------------------------------------------------------
    async def apply_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = True,
        template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        self.template_kwargs.append(dict(template_kwargs or {}))
        parts: list[str] = []
        if tools:
            parts.append("<|tools|>" + json.dumps(tools, ensure_ascii=False, sort_keys=True))
        for message in messages:
            role = str(message.get("role") or "user")
            if role not in ("system", "user", "assistant", "tool"):
                raise EngineError(f"Szablon nie obsługuje roli {role!r}.")
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if role == "tool":
                parts.append(f"<|tool:{message.get('name') or ''}|>\n{text}")
                continue
            block = f"<|{role}|>\n{text}"
            for call in message.get("tool_calls") or []:
                fn = call["function"]
                block += "\n" + tool_call_text(fn["name"], fn["arguments"])
            parts.append(block)
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    async def tokenize(self, text: str) -> list[int]:
        return list(range(max(1, len(text) // self._chars_per_token)))

    # inference -------------------------------------------------------------
    async def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        if COMPACTION_MARKER in prompt:
            self.summary_prompts.append(prompt)
            yield CompletionChunk(text=self.summary_text)
            yield CompletionChunk(done=True, finish_reason="stop",
                                  stats=GenStats(prompt_tokens=10, completion_tokens=20, gen_tps=20.0))
            return
        if GUARD_MARKER in prompt:
            self.guard_prompts.append(prompt)
            self.guard_sampling.append(params)
            reply = self.guard_replies.pop(0) if self.guard_replies else Reply(text=GUARD_OK)
        else:
            self.prompts.append(prompt)
            self.sampling.append(params)
            if not self.replies:
                raise AssertionError("ScriptedEngine ran out of scripted replies")
            reply = self.replies.pop(0)
        if reply.error is not None:
            raise reply.error
        chunks = [reply.text[i:i + self._chunk_size] for i in range(0, len(reply.text), self._chunk_size)] or [""]
        try:
            for index, piece in enumerate(chunks):
                if reply.stall_after is not None and index >= reply.stall_after:
                    await asyncio.Event().wait()   # never returns; the test interrupts
                if reply.chunk_delay:
                    await asyncio.sleep(reply.chunk_delay)
                yield CompletionChunk(text=piece)
            yield CompletionChunk(
                done=True,
                finish_reason="stop",
                stats=GenStats(
                    prompt_tokens=await self.count_tokens(prompt),
                    completion_tokens=reply.completion_tokens,
                    gen_tps=25.0,
                    total_ms=100.0,
                ),
            )
        except GeneratorExit:  # pragma: no cover - depends on GC timing
            self.cancelled += 1
            raise
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


class FakeLease:
    def __init__(self, runtime: "FakeRuntime", engine: Engine, purpose: str) -> None:
        self._runtime = runtime
        self.engine = engine
        self.purpose = purpose
        self.stats: list[GenStats] = []
        self.released = False

    @property
    def n_ctx(self) -> int:
        return self.engine.n_ctx

    @property
    def model_key(self) -> str:
        return self._runtime.model_key

    def record_stats(self, stats: GenStats | None) -> None:
        if stats is not None:
            self.stats.append(stats)
            self._runtime.stats.append(stats)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self._runtime.held -= 1


class _PendingLease:
    """Mirrors ``RuntimeManager.acquire``: awaitable *and* async context manager."""

    def __init__(self, factory: Callable[[], Awaitable[FakeLease]]) -> None:
        self._factory = factory
        self._lease: FakeLease | None = None

    def __await__(self):
        return self._factory().__await__()

    async def __aenter__(self) -> FakeLease:
        self._lease = await self._factory()
        return self._lease

    async def __aexit__(self, *exc: object) -> None:
        if self._lease is not None:
            self._lease.release()


class FakeRuntime:
    """Minimal ``RuntimeManager`` stand-in around one :class:`ScriptedEngine`."""

    def __init__(self, engine: Engine, *, model_key: str = "test/model", loaded: bool = False,
                 load_delay: float = 0.0, error: BaseException | None = None) -> None:
        self.engine = engine
        self.model_key = model_key
        self.loaded = loaded
        self.load_delay = load_delay
        self.error = error
        self.acquires: list[str] = []
        self.stats: list[GenStats] = []
        self.held = 0
        self.max_held = 0
        self.leases: list[FakeLease] = []

    def acquire(self, model_key: str, *, purpose: str = "chat", n_ctx: int | None = None) -> _PendingLease:
        async def factory() -> FakeLease:
            self.acquires.append(purpose)
            if self.error is not None:
                raise self.error
            if not self.loaded and self.load_delay:
                await asyncio.sleep(self.load_delay)
            self.loaded = True
            self.held += 1
            self.max_held = max(self.max_held, self.held)
            lease = FakeLease(self, self.engine, purpose)
            self.leases.append(lease)
            return lease

        return _PendingLease(factory)

    def is_loaded(self, model_key: str) -> bool:
        return self.loaded


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


Handler = Callable[[dict[str, Any], ToolContext], Any]


class FakeTool(Tool):
    """A tool whose behaviour is a plain callable (sync or async)."""

    def __init__(
        self,
        name: str,
        *,
        kind: str = "read",
        handler: Handler | None = None,
        description: str = "Test tool.",
        parameters: dict[str, Any] | None = None,
        permission_info: PermissionInfo | None = None,
        output: str = "ok",
        summary: str = "ok",
        diff: str | None = None,
    ) -> None:
        self.name = name
        self.kind = kind  # type: ignore[assignment]
        self.description = description
        self.parameters = parameters or {
            "type": "object",
            "properties": {"path": {"type": "string"}, "command": {"type": "string"},
                           "content": {"type": "string"}, "todos": {"type": "array"}},
        }
        self.calls: list[dict[str, Any]] = []
        self.started = asyncio.Event()
        self.cancelled = 0
        self._handler = handler
        self._permission = permission_info
        self._output = output
        self._summary = summary
        self._diff = diff

    def title(self, args: dict[str, Any]) -> str:
        target = args.get("path") or args.get("command") or ""
        return f"{self.name} {target}".strip()

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        return self._permission

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        self.started.set()
        try:
            if self._handler is not None:
                result = self._handler(args, ctx)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, ToolResult):
                    return result
                return ToolResult(ok=True, output=str(result), summary=self._summary, diff=self._diff)
            return ToolResult(ok=True, output=self._output, summary=self._summary, diff=self._diff)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@dataclass
class EventLog:
    """Collects agent events and makes them easy to assert on."""

    events: list[Any] = field(default_factory=list)

    def __call__(self, event: Any) -> None:
        self.events.append(event)

    @property
    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def of(self, type_name: str) -> list[Any]:
        return [e for e in self.events if e.type == type_name]

    def first(self, type_name: str) -> Any:
        found = self.of(type_name)
        assert found, f"no {type_name} event in {self.types}"
        return found[0]

    def last(self, type_name: str) -> Any:
        found = self.of(type_name)
        assert found, f"no {type_name} event in {self.types}"
        return found[-1]

    @property
    def text(self) -> str:
        return "".join(e.text for e in self.of("text_delta"))

    @property
    def reasoning(self) -> str:
        return "".join(e.text for e in self.of("reasoning_delta"))
