"""One suite, both engines: ``LlamaServerEngine`` (GGUF) and ``MlxEngine`` (MLX).

Everything the agent and the chatbot rely on must behave identically no matter
which format the user downloaded, so this module parametrises a single set of
``real_model`` tests over both engines with the two tiny Qwen3-0.6B models
(≈0.35 GB each):

* start/stop without leaving a process behind,
* ``n_ctx`` is the context **per slot** with ``n_parallel=2``,
* ``apply_template`` with and without tools (identical prompts from the GGUF
  and the MLX copy of one model),
* ``tokenize``, streaming completions and ``GenStats``,
* prefix-cache reuse, stop strings, cancellation mid-stream,
* two concurrent completions, context overflow (same error type),
* ``memory_bytes``.

Plus one smoke test per engine with the 4B models (loaded one at a time, each
stopped before the next starts) that renders a tools prompt, generates and
feeds the output to ``runtime.toolcalls.parse_output``.

Throughput of every generation is printed as ``PERF …`` lines (``pytest -s``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any, AsyncIterator

import psutil
import pytest
import pytest_asyncio

from lc_testing import model_path
from localcode.config import Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.base import CompletionChunk, EngineError, GenStats, LoadConfig, SamplingParams
from localcode.runtime.http_engine import ContextOverflowError, HttpEngine
from localcode.runtime.llama_engine import LlamaServerEngine, find_llama_server
from localcode.runtime.toolcalls import parse_output

FORMATS = ("gguf", "mlx")
# real-model tests share one event loop with the module-scoped engine fixture
module_loop = pytest.mark.asyncio(loop_scope="module")

QWEN06_META = ModelMeta(arch="qwen3", n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128, context_length=40960)
QWEN4_META = ModelMeta(arch="qwen3", n_layers=36, n_heads=32, n_kv_heads=8, head_dim=128, context_length=262144)

#: ``lc_testing`` keys of the models used here: (tiny 0.6B, small 4B).
MODEL_KEYS = {
    "gguf": ("gguf_tiny", "gguf_small"),
    "mlx": ("mlx_tiny", "mlx_small"),
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the project and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
                "required": ["path"],
            },
        },
    }
]


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #


def _skip_reason(fmt: str) -> str | None:
    if fmt == "gguf":
        return None if find_llama_server(Settings()) else "llama-server not installed"
    try:
        import mlx_lm  # noqa: F401
    except Exception:  # noqa: BLE001 - not on Apple Silicon / not installed
        return "mlx-lm not installed"
    return None


def _model(fmt: str, size: str) -> LocalModel:
    """``LocalModel`` for the tiny/small test model of one format."""
    path = model_path(MODEL_KEYS[fmt][0 if size == "tiny" else 1])
    meta = QWEN06_META if size == "tiny" else QWEN4_META
    if fmt == "gguf":
        return LocalModel(key=f"test/{path.stem}", format="gguf", path=str(path), display_name=path.stem,
                          size_bytes=path.stat().st_size, meta=meta)
    total = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    return LocalModel(key=f"test/{path.name}", format="mlx", path=str(path), display_name=path.name,
                      size_bytes=total, quant="4bit", meta=meta)


def _engine(fmt: str, size: str, instance_id: str, config: LoadConfig) -> HttpEngine:
    reason = _skip_reason(fmt)
    if reason:
        pytest.skip(reason)
    model = _model(fmt, size)
    if fmt == "gguf":
        return LlamaServerEngine(instance_id, model, config, settings=Settings())
    from localcode.runtime.mlx_engine import MlxEngine

    return MlxEngine(instance_id, model, config, settings=Settings())


@pytest.fixture(scope="module")
def lc_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("lc-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALCODE_HOME", str(home))
        yield home


@pytest_asyncio.fixture(scope="module", loop_scope="module", params=FORMATS, ids=FORMATS)
async def engine(request, lc_home) -> AsyncIterator[HttpEngine]:
    """One running tiny model per format (started once for the whole module)."""
    eng = _engine(request.param, "tiny", f"parity-{request.param}", LoadConfig(n_ctx=1024, n_parallel=2))
    await eng.start()
    try:
        yield eng
    finally:
        await eng.stop()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _collect(it: AsyncIterator[CompletionChunk]) -> list[CompletionChunk]:
    out: list[CompletionChunk] = []
    async with contextlib.aclosing(it) as agen:
        async for chunk in agen:
            out.append(chunk)
    return out


async def _generate(engine: HttpEngine, prompt: str, params: SamplingParams, label: str) -> tuple[str, GenStats]:
    chunks = await _collect(engine.complete(prompt, params))
    final = chunks[-1]
    assert final.done and final.stats is not None
    _perf(engine, label, final.stats)
    return "".join(c.text for c in chunks), final.stats


def _perf(engine: HttpEngine, label: str, stats: GenStats) -> None:
    print(
        f"PERF {engine.format:4s} {Path(engine.model.path).name:<44s} {label:<22s} "
        f"prompt {stats.prompt_tokens:5d} tok @ {stats.prompt_tps:8.1f} tok/s | "
        f"gen {stats.completion_tokens:4d} tok @ {stats.gen_tps:6.1f} tok/s"
    )


async def _prompt(engine: HttpEngine, text: str, tools: list[dict[str, Any]] | None = None) -> str:
    """User turn rendered with thinking off (both tiny models are Qwen3)."""
    kwargs = {engine.template_info.thinking_kwarg: False} if engine.template_info.thinking_kwarg else None
    return await engine.apply_template([{"role": "user", "content": text}], tools, template_kwargs=kwargs)


async def _busy_slots(engine: HttpEngine) -> int:
    return sum(1 for s in await engine.slots() if s["is_processing"])


async def _wait_idle(engine: HttpEngine, timeout: float = 5.0) -> float | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not await _busy_slots(engine):
            return time.monotonic() - t0
        await asyncio.sleep(0.02)
    return None


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
@pytest.mark.parametrize("fmt", FORMATS)
@module_loop
async def test_start_and_stop_leave_no_process(fmt, lc_home):
    eng = _engine(fmt, "tiny", f"lifecycle-{fmt}", LoadConfig(n_ctx=512))
    assert not eng.running and eng.pid is None
    await eng.start()
    pid = eng.pid
    agen = None
    try:
        assert eng.running and pid and eng.port and await eng.health()
        children = psutil.Process(pid).children(recursive=True)
        assert eng.log_path is not None and eng.log_path.exists()
        # stopping while a stream is open must not hang (llama-server ignores
        # SIGTERM until its response is closed, so the session is closed first)
        agen = eng.complete(await _prompt(eng, "Write a very long story."), SamplingParams(max_tokens=500))
        await agen.__anext__()
        t0 = time.monotonic()
        await eng.stop()
        assert time.monotonic() - t0 < 10
        with pytest.raises(EngineError):
            async for _ in agen:
                pass
    finally:
        if agen is not None:
            await agen.aclose()
        await eng.stop()
    assert not eng.running and eng.pid is None
    assert not _alive(pid)
    assert not any(_alive(c.pid) for c in children)
    await eng.stop()  # idempotent


@pytest.mark.real_model
@module_loop
async def test_per_slot_context_with_two_slots(engine):
    assert engine.running and engine.pid and engine.port
    assert engine.n_ctx == 1024                  # per slot, not the total
    assert engine.n_parallel == 2
    slots = await engine.slots()
    assert [s["n_ctx"] for s in slots] == [1024, 1024]
    assert await _busy_slots(engine) == 0
    props = await engine.props()
    assert props["default_generation_settings"]["n_ctx"] == 1024
    assert props["total_slots"] == 2
    assert "<|im_start|>" in props["chat_template"]


# --------------------------------------------------------------------------- #
# Templating and tokenisation
# --------------------------------------------------------------------------- #


#: Rendered prompts per format, compared across engines by the second run.
_RENDERED: dict[str, dict[str, str]] = {}


@pytest.mark.real_model
@module_loop
async def test_apply_template_with_and_without_tools(engine):
    info = engine.template_info
    assert info.supports_tools and info.supports_tool_role and info.supports_system
    assert info.tool_format == "hermes"

    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Read a.py"}]
    plain = await engine.apply_template(msgs)
    assert plain.startswith("<|im_start|>system\nYou are terse.")
    assert plain.endswith("<|im_start|>assistant\n")
    assert "read_file" not in plain

    with_tools = await engine.apply_template(msgs, TOOLS)
    assert "<tools>" in with_tools and '"name": "read_file"' in with_tools
    assert with_tools.endswith("<|im_start|>assistant\n")

    history = msgs + [
        {"role": "assistant", "content": "", "reasoning": "SECRET-THOUGHT",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "print(1)"},
    ]
    rendered = await engine.apply_template(history, TOOLS)
    assert '"name": "read_file"' in rendered and '"path": "a.py"' in rendered
    assert "<tool_response>\nprint(1)\n</tool_response>" in rendered
    assert "SECRET-THOUGHT" not in rendered          # reasoning is never re-sent

    # a prompt without a generation prompt is not closed with an assistant turn
    no_gen = await engine.apply_template(msgs, add_generation_prompt=False)
    assert not no_gen.endswith("<|im_start|>assistant\n")

    mine = {"plain": plain, "tools": with_tools, "history": rendered}
    for fmt, other in _RENDERED.items():
        assert other == mine, f"{fmt} and {engine.format} render the same model differently"
    _RENDERED[engine.format] = mine


@pytest.mark.real_model
@module_loop
async def test_tokenize(engine):
    assert await engine.tokenize("Hello world") == [9707, 1879]
    assert await engine.tokenize("<|im_end|>") == [151645]   # special tokens parsed, like in prompts
    assert await engine.count_tokens("") == 0
    prompt = await _prompt(engine, "Say OK.")
    assert await engine.count_tokens(prompt) > 10


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
@module_loop
async def test_streaming_completion_and_stats(engine):
    prompt = await _prompt(engine, "Count from 1 to 100, separated by commas.")
    n_prompt = await engine.count_tokens(prompt)
    chunks = await _collect(engine.complete(prompt, SamplingParams(max_tokens=32, temperature=0.0, seed=1)))
    final = chunks[-1]
    assert final.done and final.finish_reason == "length"
    assert all(not c.done and c.text for c in chunks[:-1])
    assert len(chunks) > 2                      # really streamed, not one blob
    assert "".join(c.text for c in chunks).strip()
    stats = final.stats
    _perf(engine, "stream 32 tok", stats)
    assert stats is not None
    assert stats.prompt_tokens == n_prompt
    assert stats.completion_tokens == 32
    assert stats.prompt_tps > 0 and stats.gen_tps > 0 and stats.total_ms > 0


@pytest.mark.real_model
@module_loop
async def test_prefix_cache_is_reused(engine):
    prefix = "You are a helpful assistant. " * 20
    prompt = await engine.apply_template(
        [{"role": "system", "content": prefix}, {"role": "user", "content": "Say OK."}],
        template_kwargs={engine.template_info.thinking_kwarg: False} if engine.template_info.thinking_kwarg else None,
    )
    params = SamplingParams(max_tokens=4, temperature=0.0)
    _, first = await _generate(engine, prompt, params, "cold prompt")
    _, second = await _generate(engine, prompt, params, "cached prompt")
    assert second.prompt_tokens == first.prompt_tokens
    assert second.cached_tokens > first.cached_tokens
    assert second.cached_tokens >= second.prompt_tokens - 4


@pytest.mark.real_model
@module_loop
async def test_stop_strings_are_not_emitted(engine):
    prompt = await _prompt(engine, "Count from 1 to 20, separated by commas. Answer with the numbers only.")
    text, stats = await _generate(
        engine, prompt, SamplingParams(max_tokens=64, temperature=0.0, stop=["5"]), "stop string"
    )
    assert "5" not in text
    assert stats.completion_tokens < 64          # stopped early, before the token budget
    assert text.strip()


@pytest.mark.real_model
@module_loop
async def test_cancellation_mid_stream_stops_generation(engine):
    prompt = await _prompt(engine, "Write a very long story about dragons.")
    params = SamplingParams(max_tokens=900, temperature=0.7)

    agen = engine.complete(prompt, params)
    seen = 0
    async for _ in agen:
        seen += 1
        if seen == 5:
            break
    assert await _busy_slots(engine) >= 1
    t0 = time.monotonic()
    await agen.aclose()
    assert await _wait_idle(engine) is not None, "slot still busy 5 s after aclose()"
    assert time.monotonic() - t0 < 5.0

    # cancelling the consuming task must work the same way
    async def consume() -> None:
        async for _ in engine.complete(prompt, params):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.6)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_idle(engine) is not None

    # and the engine keeps serving afterwards
    chunks = await _collect(engine.complete(prompt, SamplingParams(max_tokens=4, temperature=0.0)))
    assert chunks[-1].done and chunks[-1].stats.completion_tokens == 4


@pytest.mark.real_model
@module_loop
async def test_two_concurrent_completions(engine):
    # long answers, so both requests are certain to overlap in time
    prompts = [await _prompt(engine, q) for q in
               ("Write a long, detailed paragraph about colors.",
                "Write a long, detailed paragraph about animals.")]
    busy: list[int] = []

    async def watch() -> None:
        while True:
            busy.append(await _busy_slots(engine))
            await asyncio.sleep(0.02)

    watcher = asyncio.create_task(watch())
    t0 = time.perf_counter()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(_collect(engine.complete(p, SamplingParams(max_tokens=64, temperature=0.0)))
                             for p in prompts)),
            timeout=180,
        )
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
    wall = time.perf_counter() - t0
    total = sum(c[-1].stats.completion_tokens for c in results)
    print(f"PERF {engine.format:4s} 2 concurrent completions: {total} tok in {wall:.2f}s "
          f"({total / wall:.1f} tok/s aggregate, peak busy slots {max(busy, default=0)})")
    for chunks in results:
        assert chunks[-1].done and chunks[-1].stats.completion_tokens > 0
        assert "".join(c.text for c in chunks).strip()
    assert max(busy, default=0) == 2              # both slots really worked at once


@pytest.mark.real_model
@module_loop
async def test_requests_after_a_stream_survive_stale_keep_alive(engine):
    """Regression: llama-server closes the TCP connection after every SSE
    stream, so the pooled connection is dead when the next call writes to it
    (``ServerDisconnectedError``). Mixing streams and plain calls must work."""
    prompt = await _prompt(engine, "Say OK.")
    for _ in range(5):
        chunks = await _collect(engine.complete(prompt, SamplingParams(max_tokens=4, temperature=0.0)))
        assert chunks[-1].done
        assert await engine.count_tokens(prompt) > 0
        assert (await engine.props())["total_slots"] == 2
        assert await engine.apply_template([{"role": "user", "content": "Hi"}])


@pytest.mark.real_model
@module_loop
async def test_context_overflow_raises_the_same_error(engine):
    with pytest.raises(ContextOverflowError) as excinfo:
        await _collect(engine.complete("word " * 1500, SamplingParams(max_tokens=4)))
    err = excinfo.value
    assert err.n_ctx == 1024
    assert err.n_prompt_tokens and err.n_prompt_tokens > 1024
    assert err.status == 400 and err.error_type == "exceed_context_size_error"
    assert "przekracza dostępny kontekst" in str(err)
    # the instance stays healthy and usable
    assert engine.running and await engine.health()
    chunks = await _collect(engine.complete(await _prompt(engine, "Say OK."),
                                            SamplingParams(max_tokens=4, temperature=0.0)))
    assert chunks[-1].done


@pytest.mark.real_model
@module_loop
async def test_memory_bytes_is_reported(engine):
    measured = await engine.memory_bytes()
    assert measured is not None and measured > 0
    assert measured > engine.model.size_bytes * 0.5
    print(f"PERF {engine.format:4s} memory_bytes: {measured / 2**30:.2f} GiB")


# --------------------------------------------------------------------------- #
# 4B smoke test: a real tool call, one model at a time
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
@pytest.mark.parametrize("fmt", FORMATS)
@module_loop
async def test_small_model_produces_a_parsable_tool_call(fmt, lc_home):
    eng = _engine(fmt, "small", f"tools-{fmt}", LoadConfig(n_ctx=4096))
    await eng.start()
    pid = eng.pid
    try:
        messages = [
            {"role": "system", "content": "You are a coding agent. Use the available tools to inspect files. "
                                          "Call a tool instead of guessing the answer."},
            {"role": "user", "content": "What does the file src/main.py contain?"},
        ]
        assert eng.template_info.supports_tools
        prompt = await eng.apply_template(messages, TOOLS)
        assert "read_file" in prompt
        text, stats = await _generate(
            eng, prompt, SamplingParams(max_tokens=256, temperature=0.0, seed=7), "tool call"
        )
        parsed = parse_output(text, tool_format=eng.template_info.tool_format, known_tools=TOOLS)
        assert parsed.tool_calls, f"no tool call in: {text!r}"
        call = parsed.tool_calls[0]
        assert call.name == "read_file"
        assert isinstance(call.arguments, dict) and "main.py" in str(call.arguments.get("path", ""))
        assert not parsed.errors
        assert "<tool_call>" not in parsed.content
        assert stats.completion_tokens > 0
    finally:
        await eng.stop()
    assert pid is not None and not _alive(pid)
