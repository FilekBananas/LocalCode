"""Tests for ``localcode.runtime.llama_engine``.

Unit tests cover binary discovery, argument building (per-slot context, KV
cache types vs flash attention, embeddings, feature-flag detection) and
failure explanations. ``real_model`` tests run the Homebrew llama-server with
the tiny Qwen3-0.6B GGUF (≈0.4 GB) and the nomic embedding model (≈0.15 GB).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import stat
import time
from pathlib import Path

import psutil
import pytest
import pytest_asyncio

from lc_testing import model_path
from localcode.config import RuntimeSettings, Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime import llama_engine as le
from localcode.runtime.base import CompletionChunk, EngineError, LoadConfig, SamplingParams
from localcode.runtime.http_engine import ContextOverflowError, EngineStartError
from localcode.runtime.llama_engine import (
    LlamaServerEngine,
    find_llama_server,
    llama_server_flags,
    normalize_flash_attn,
    normalize_kv_cache_type,
)

LLAMA = find_llama_server(Settings())

QWEN_META = ModelMeta(arch="qwen3", n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128, context_length=40960)


def _model(path: str = "/models/x.gguf", *, meta: ModelMeta | None = None, mmproj: str | None = None,
           fmt: str = "gguf") -> LocalModel:
    return LocalModel(key="org/repo/x", format=fmt, path=path, display_name="Test Model", size_bytes=1000,
                      mmproj_path=mmproj, meta=meta or ModelMeta())


def _engine(config: LoadConfig | None = None, **kw) -> LlamaServerEngine:
    e = LlamaServerEngine("inst-1", kw.pop("model", None) or _model(**kw), config or LoadConfig(n_ctx=4096))
    e._binary = "/usr/bin/llama-server"
    e._flags = None  # assume a current build
    return e


def _arg(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# --------------------------------------------------------------------------- #
# Discovery and normalisation
# --------------------------------------------------------------------------- #


def _make_exe(path: Path) -> Path:
    path.write_text("#!/bin/sh\necho llama\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_llama_server_prefers_configured_path(tmp_path, monkeypatch):
    exe = _make_exe(tmp_path / "custom-llama")
    s = Settings(runtime=RuntimeSettings(llama_server_path=str(exe)))
    assert find_llama_server(s) == str(exe)
    d = tmp_path / "bin"
    d.mkdir()
    _make_exe(d / "llama-server")
    s = Settings(runtime=RuntimeSettings(llama_server_path=str(d)))
    assert find_llama_server(s) == str(d / "llama-server")


def test_find_llama_server_falls_back_to_path_and_homebrew(tmp_path, monkeypatch):
    on_path = tmp_path / "path"
    on_path.mkdir()
    _make_exe(on_path / "llama-server")
    monkeypatch.setenv("PATH", str(on_path))
    s = Settings(runtime=RuntimeSettings(llama_server_path=str(tmp_path / "missing")))
    assert find_llama_server(s) == str(on_path / "llama-server")

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    brew = _make_exe(tmp_path / "brew-llama-server")
    monkeypatch.setattr(le, "LLAMA_SERVER_CANDIDATES", (str(tmp_path / "nope"), str(brew)))
    assert find_llama_server(Settings()) == str(brew)
    monkeypatch.setattr(le, "LLAMA_SERVER_CANDIDATES", ())
    assert find_llama_server(Settings()) is None


def test_normalize_kv_and_flash_attn():
    assert normalize_kv_cache_type("Q8_0") == "q8_0"
    assert normalize_kv_cache_type("8bit") == "q8_0"
    assert normalize_kv_cache_type("4bit") == "q4_0"
    assert normalize_kv_cache_type("") == "f16"
    assert normalize_kv_cache_type(None) == "f16"
    assert normalize_kv_cache_type("q5_1") == "q5_1"
    assert normalize_kv_cache_type("q3_weird") == "f16"
    assert normalize_flash_attn("on") == "on"
    assert normalize_flash_attn(True) == "on"
    assert normalize_flash_attn("false") == "off"
    assert normalize_flash_attn("auto") == "auto"
    assert normalize_flash_attn(None) == "auto"


@pytest.mark.skipif(LLAMA is None, reason="llama-server not installed")
async def test_llama_server_flags_from_real_binary():
    flags = await llama_server_flags(LLAMA)
    assert flags is not None
    for f in ("--port", "-np", "--no-kv-unified", "--cache-ram", "--fit", "--no-ui", "--embeddings", "--alias"):
        assert f in flags
    assert await llama_server_flags(LLAMA) is flags  # memoised
    assert await llama_server_flags("/bin/echo") is None  # not a llama-server
    assert await llama_server_flags("/nonexistent/llama-server") is None


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_command_per_slot_context_and_defaults():
    e = _engine(LoadConfig(n_ctx=4096, n_parallel=2))
    cmd = e._build_command(5555)
    assert cmd[0] == "/usr/bin/llama-server"
    assert _arg(cmd, "-m") == "/models/x.gguf"
    assert _arg(cmd, "--host") == "127.0.0.1" and _arg(cmd, "--port") == "5555"
    assert _arg(cmd, "-c") == "8192" and _arg(cmd, "-np") == "2"   # n_ctx is PER SLOT
    assert "--no-kv-unified" in cmd
    assert _arg(cmd, "-ngl") == "999" and _arg(cmd, "-fa") == "auto"
    assert _arg(cmd, "--fit") == "off" and _arg(cmd, "--alias") == "inst-1"
    assert "--jinja" in cmd and "--no-ui" in cmd and "--no-webui" not in cmd
    assert _arg(cmd, "--log-colors") == "off"
    assert "--cache-type-k" not in cmd and "--cache-type-v" not in cmd
    assert "--embeddings" not in cmd and "--mmproj" not in cmd
    assert e.command == cmd


def test_command_gpu_layers_and_flash_attn():
    cmd = _engine(LoadConfig(n_ctx=1024, gpu_layers=0, flash_attn="off"))._build_command(1)
    assert _arg(cmd, "-ngl") == "0" and _arg(cmd, "-fa") == "off"
    cmd = _engine(LoadConfig(n_ctx=1024, gpu_layers=20, flash_attn="on"))._build_command(1)
    assert _arg(cmd, "-ngl") == "20" and _arg(cmd, "-fa") == "on"


def test_command_quantized_kv_cache():
    cmd = _engine(LoadConfig(n_ctx=1024, kv_cache_type="q8_0"))._build_command(1)
    assert _arg(cmd, "--cache-type-k") == "q8_0" and _arg(cmd, "--cache-type-v") == "q8_0"
    # quantized V needs flash attention: with FA off only K is quantized
    cmd = _engine(LoadConfig(n_ctx=1024, kv_cache_type="q4_0", flash_attn="off"))._build_command(1)
    assert _arg(cmd, "--cache-type-k") == "q4_0" and "--cache-type-v" not in cmd
    # MLX-style setting values are mapped
    cmd = _engine(LoadConfig(n_ctx=1024, kv_cache_type="8bit"))._build_command(1)
    assert _arg(cmd, "--cache-type-k") == "q8_0"


def test_quantized_v_failure_retries_with_f16_v():
    e = _engine(LoadConfig(n_ctx=1024, kv_cache_type="q8_0"))
    tail = "E llama_init_from_model: quantized V cache requires flash_attn to be enabled"
    assert e._on_start_failure(tail, 1) is True
    cmd = e._build_command(1)
    assert _arg(cmd, "--cache-type-k") == "q8_0" and "--cache-type-v" not in cmd
    assert e._on_start_failure(tail, 1) is False            # only once
    assert _engine()._on_start_failure(tail, 1) is False     # f16 V: nothing to fix
    assert _engine()._on_start_failure("something else", 1) is False


def test_command_embedding_mode_and_mmproj_and_extra_args():
    e = _engine(LoadConfig(n_ctx=2048, embedding=True, extra_args=["--pooling", "mean", "--cache-ram", "64"]),
                mmproj="/models/mmproj.gguf")
    cmd = e._build_command(1)
    assert "--embeddings" in cmd and _arg(cmd, "-b") == "2048" and _arg(cmd, "-ub") == "2048"
    assert _arg(cmd, "--mmproj") == "/models/mmproj.gguf"
    assert _arg(cmd, "--cache-ram") == "0"                   # our default for embedding servers …
    assert cmd[-4:] == ["--pooling", "mean", "--cache-ram", "64"]  # … overridden by extra args (last wins)


def test_command_respects_old_builds_without_new_flags():
    e = _engine(LoadConfig(n_ctx=1024))
    e._flags = frozenset({"-m", "--host", "--port", "-c", "-np", "-ngl", "-fa", "--no-webui", "--alias", "--jinja"})
    cmd = e._build_command(1)
    assert "--no-webui" in cmd and "--no-ui" not in cmd
    for flag in ("--fit", "--no-kv-unified", "--cache-ram", "--log-colors"):
        assert flag not in cmd


def test_prompt_cache_sizing():
    # Qwen3-0.6B: 28 layers · 8 kv heads · 128 dims · (2+2) bytes = 114688 B/token
    e = _engine(LoadConfig(n_ctx=4096), meta=QWEN_META)
    assert e.prompt_cache_mib == int(2 * 114688 * 4096 / 2**20)  # two full slot states = 896 MiB
    assert _engine(LoadConfig(n_ctx=256), meta=QWEN_META).prompt_cache_mib == le.PROMPT_CACHE_MIN_MIB
    assert _engine(LoadConfig(n_ctx=131072), meta=QWEN_META).prompt_cache_mib == le.PROMPT_CACHE_MAX_MIB
    assert _engine(LoadConfig(n_ctx=4096)).prompt_cache_mib == le.PROMPT_CACHE_DEFAULT_MIB  # unknown arch
    q8 = _engine(LoadConfig(n_ctx=8192, kv_cache_type="q8_0"), meta=QWEN_META).prompt_cache_mib
    assert q8 == int(2 * 28 * 8 * 128 * (34 / 32) * 2 * 8192 / 2**20)
    assert _engine(LoadConfig(n_ctx=4096, embedding=True), meta=QWEN_META).prompt_cache_mib == 0


def test_env_strips_llama_arg_variables(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_N_PARALLEL", "8")
    monkeypatch.setenv("LLAMA_API_KEY", "user-key")
    monkeypatch.setenv("SOME_OTHER", "1")
    e = _engine()
    e._api_key = "secret"
    env = e._env()
    assert "LLAMA_ARG_N_PARALLEL" not in env
    assert env["LLAMA_API_KEY"] == "secret" and env["SOME_OTHER"] == "1"
    assert "secret" not in " ".join(e._build_command(1))  # never on the command line


def test_failure_hints():
    e = _engine()
    assert "architektury „mamba9”" in e._failure_hint("llama_model_load: error loading model: unknown model architecture: 'mamba9'", 1)
    assert "odrzucił parametry" in e._failure_hint("error: invalid argument: --bogus", 1)
    assert "uszkodzony" in e._failure_hint("E load_model: failed to load model, 'x.gguf'", 1)
    assert "kontekstu modelu" in e._failure_hint("E srv load_model: failed to create_context with model", 1)
    assert e._failure_hint("all good", 0) is None
    assert e._failure_hint("ggml_metal: failed to allocate buffer", 1) is None  # generic memory hint used
    assert "pamięci" in e._generic_failure_hint("ggml_metal: failed to allocate buffer", 1)
    v = _engine(mmproj="/m/mmproj.gguf")
    assert "mmproj" in v._failure_hint("clip_model_load: failed to load mmproj", 1)


async def test_before_spawn_validation(tmp_path, monkeypatch):
    e = LlamaServerEngine("i", _model(fmt="mlx"), LoadConfig(n_ctx=512), binary="/bin/sh")
    with pytest.raises(EngineStartError, match="nie jest w formacie GGUF"):
        await e.start()
    e = LlamaServerEngine("i", _model(str(tmp_path / "missing.gguf")), LoadConfig(n_ctx=512), binary="/bin/sh")
    with pytest.raises(EngineStartError, match="Plik modelu nie istnieje"):
        await e.start()
    f = tmp_path / "m.gguf"
    f.write_bytes(b"GGUF")
    e = LlamaServerEngine("i", _model(str(f), mmproj=str(tmp_path / "nope.gguf")), LoadConfig(n_ctx=512), binary="/bin/sh")
    with pytest.raises(EngineStartError, match="mmproj"):
        await e.start()
    monkeypatch.setattr(le, "find_llama_server", lambda settings=None: None)
    e = LlamaServerEngine("i", _model(str(f)), LoadConfig(n_ctx=512), settings=Settings())
    with pytest.raises(EngineStartError, match="brew install llama.cpp"):
        await e.start()
    assert e.pid is None


# --------------------------------------------------------------------------- #
# Real llama-server with the tiny GGUF model
# --------------------------------------------------------------------------- #

requires_llama = pytest.mark.skipif(LLAMA is None, reason="llama-server not installed")
# real-model tests share one event loop with the module-scoped engine fixture
module_loop = pytest.mark.asyncio(loop_scope="module")


def _gguf_model(name: str, key: str, display: str) -> LocalModel:
    p = model_path(name)
    return LocalModel(key=key, format="gguf", path=str(p), display_name=display, size_bytes=p.stat().st_size)


@pytest.fixture(scope="module")
def lc_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("lc-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALCODE_HOME", str(home))
        yield home


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tiny(lc_home):
    if LLAMA is None:
        pytest.skip("llama-server not installed")
    model = _gguf_model("gguf_tiny", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M", "Qwen3 0.6B")
    engine = LlamaServerEngine("tiny-1", model, LoadConfig(n_ctx=1024, n_parallel=2), settings=Settings())
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


async def _collect(it) -> list[CompletionChunk]:
    out: list[CompletionChunk] = []
    async with contextlib.aclosing(it) as agen:
        async for c in agen:
            out.append(c)
    return out


def _chat(engine: LlamaServerEngine, text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_start_props_and_per_slot_ctx(tiny, lc_home):
    assert tiny.running and tiny.pid and tiny.port
    assert tiny.n_ctx == 1024                 # per slot, although -c is 2048
    assert tiny.n_parallel == 2
    assert _arg(tiny.command, "-c") == "2048" and _arg(tiny.command, "-np") == "2"
    slots = await tiny.slots()
    assert [s["n_ctx"] for s in slots] == [1024, 1024]
    props = await tiny.props()
    assert props["default_generation_settings"]["n_ctx"] == 1024
    assert "<|im_start|>" in props["chat_template"]
    assert tiny.template_info.supports_tools and tiny.server_renders_tools
    assert tiny.template_info.supports_tool_role and tiny.template_info.supports_system
    assert tiny.template_info.thinking_kwarg == "enable_thinking"
    assert tiny._adds_bos is False            # Qwen3 does not add BOS
    assert (lc_home / "logs" / "llama-tiny-1.log").exists()
    mem = await tiny.memory_bytes()
    assert mem is not None and mem > tiny.model.size_bytes * 0.5


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_api_key_required(tiny):
    import aiohttp

    async with aiohttp.ClientSession() as s:
        async with s.get(f"{tiny.base_url}/props") as r:
            assert r.status == 401
        async with s.get(f"{tiny.base_url}/health") as r:
            assert r.status == 200


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_apply_template_with_and_without_tools(tiny):
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Read a.py"}]
    plain = await tiny.apply_template(msgs)
    assert plain.startswith("<|im_start|>system\nYou are terse.") and plain.endswith("<|im_start|>assistant\n")
    assert "<tools>" not in plain
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    with_tools = await tiny.apply_template(msgs, tools)
    assert "<tools>" in with_tools and '"name": "read_file"' in with_tools
    no_think = await tiny.apply_template(msgs, template_kwargs={"enable_thinking": False})
    assert no_think.endswith("<think>\n\n</think>\n\n")
    history = msgs + [
        {"role": "assistant", "content": "", "reasoning": "SECRET-THOUGHT",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "print(1)"},
    ]
    rendered = await tiny.apply_template(history, tools)
    assert '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>' in rendered
    assert "<tool_response>\nprint(1)\n</tool_response>" in rendered
    assert "SECRET-THOUGHT" not in rendered


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_tokenize(tiny):
    assert len(await tiny.tokenize("Hello world")) == 2
    assert await tiny.tokenize("<|im_end|>") == [151645]   # special tokens parsed like in prompts
    assert await tiny.count_tokens("") == 0


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_streaming_completion_and_stats(tiny):
    prompt = await tiny.apply_template(_chat(tiny, "Count from 1 to 100, separated by commas."),
                                       template_kwargs={"enable_thinking": False})
    n_prompt = await tiny.count_tokens(prompt)
    chunks = await _collect(tiny.complete(prompt, SamplingParams(max_tokens=16, temperature=0.0, seed=1)))
    final = chunks[-1]
    assert final.done and final.finish_reason == "length"
    assert all(not c.done and c.text for c in chunks[:-1])
    assert "".join(c.text for c in chunks).strip()
    s = final.stats
    assert s is not None
    assert s.prompt_tokens == n_prompt and s.completion_tokens == 16
    assert s.gen_tps > 0 and s.prompt_tps > 0 and s.total_ms > 0

    chunks = await _collect(tiny.complete(prompt, SamplingParams(max_tokens=200, temperature=0.0, stop=["5"])))
    assert chunks[-1].finish_reason == "stop"
    assert "5" not in "".join(c.text for c in chunks)


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_prefix_cache_reports_cached_tokens(tiny):
    prefix = "You are a helpful assistant. " * 20
    prompt = await tiny.apply_template([{"role": "system", "content": prefix}, {"role": "user", "content": "Say OK."}],
                                       template_kwargs={"enable_thinking": False})
    first = (await _collect(tiny.complete(prompt, SamplingParams(max_tokens=4, temperature=0.0))))[-1].stats
    second = (await _collect(tiny.complete(prompt, SamplingParams(max_tokens=4, temperature=0.0))))[-1].stats
    assert second.prompt_tokens == first.prompt_tokens
    assert second.cached_tokens >= second.prompt_tokens - 4
    assert second.cached_tokens > first.cached_tokens


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_cancellation_stops_generation(tiny):
    prompt = await tiny.apply_template(_chat(tiny, "Write a very long story about dragons."),
                                       template_kwargs={"enable_thinking": False})
    agen = tiny.complete(prompt, SamplingParams(max_tokens=900))
    n = 0
    async for _ in agen:
        n += 1
        if n == 5:
            break
    assert any(s["is_processing"] for s in await tiny.slots())
    t0 = time.monotonic()
    await agen.aclose()
    idle_after = None
    for _ in range(100):
        if not any(s["is_processing"] for s in await tiny.slots()):
            idle_after = time.monotonic() - t0
            break
        await asyncio.sleep(0.02)
    assert idle_after is not None and idle_after < 2.0

    # cancelling the consuming task works the same way
    async def consume():
        async for _ in tiny.complete(prompt, SamplingParams(max_tokens=900)):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(100):
        if not any(s["is_processing"] for s in await tiny.slots()):
            break
        await asyncio.sleep(0.02)
    assert not any(s["is_processing"] for s in await tiny.slots())
    # and the engine keeps serving
    chunks = await _collect(tiny.complete(prompt, SamplingParams(max_tokens=3)))
    assert chunks[-1].done


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_two_concurrent_completions(tiny):
    prompts = [
        await tiny.apply_template(_chat(tiny, q), template_kwargs={"enable_thinking": False})
        for q in ("Name three colors.", "Name three animals.")
    ]
    seen_busy: list[int] = []

    async def watch():
        while True:
            seen_busy.append(sum(1 for s in await tiny.slots() if s["is_processing"]))
            await asyncio.sleep(0.02)

    watcher = asyncio.create_task(watch())
    try:
        results = await asyncio.wait_for(asyncio.gather(
            *(_collect(tiny.complete(p, SamplingParams(max_tokens=64, temperature=0.0))) for p in prompts)), 120)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
    for chunks in results:
        assert chunks[-1].done and chunks[-1].stats.completion_tokens > 0
        assert "".join(c.text for c in chunks).strip()
    assert max(seen_busy, default=0) == 2    # both slots worked at the same time


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_context_overflow(tiny):
    prompt = "word " * 1500
    with pytest.raises(ContextOverflowError) as ei:
        await _collect(tiny.complete(prompt, SamplingParams(max_tokens=4)))
    assert ei.value.n_ctx == 1024 and ei.value.n_prompt_tokens and ei.value.n_prompt_tokens > 1024
    assert "przekracza dostępny kontekst" in str(ei.value)
    # still healthy afterwards
    assert tiny.running and await tiny.health()


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_embeddings_unsupported_on_chat_model(tiny):
    with pytest.raises(EngineError, match="embeddingów"):
        await tiny.embed(["hello"])


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_quantized_kv_cache_and_stop_kills(lc_home):
    model = _gguf_model("gguf_tiny", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M", "Qwen3 0.6B")
    e = LlamaServerEngine("tiny-q8", model, LoadConfig(n_ctx=512, kv_cache_type="q8_0"), settings=Settings())
    await e.start()
    pid = e.pid
    try:
        assert _arg(e.command, "--cache-type-v") == "q8_0"
        prompt = await e.apply_template([{"role": "user", "content": "Hi"}], template_kwargs={"enable_thinking": False})
        chunks = await _collect(e.complete(prompt, SamplingParams(max_tokens=4)))
        assert chunks[-1].done
        # stop while a generation is streaming (llama-server ignores SIGTERM then)
        agen = e.complete(prompt, SamplingParams(max_tokens=500))
        await agen.__anext__()
        t0 = time.monotonic()
        await e.stop()
        assert time.monotonic() - t0 < 8
        with pytest.raises(EngineError):
            async for _ in agen:
                pass
    finally:
        await e.stop()
    assert pid is not None
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert not e.running and e.pid is None


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_bad_model_file_reports_log_tail(tmp_path, lc_home):
    bad = tmp_path / "broken.gguf"
    bad.write_bytes(b"GGUF" + os.urandom(4096))
    e = LlamaServerEngine("broken-1", _model(str(bad)), LoadConfig(n_ctx=512), settings=Settings())
    with pytest.raises(EngineStartError) as ei:
        await e.start()
    msg = str(ei.value)
    assert "Nie udało się uruchomić modelu „Test Model”" in msg
    assert "Ostatnie linie logu" in msg and "failed to load model" in msg
    assert "uszkodzony" in msg
    assert ei.value.log_tail and e.pid is None


@pytest.mark.real_model
@requires_llama
@module_loop
async def test_real_embeddings_nomic(lc_home):
    model = _gguf_model("gguf_embed", "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0", "nomic")
    e = LlamaServerEngine("embed-1", model, LoadConfig(n_ctx=2048, embedding=True), settings=Settings())
    await e.start()
    try:
        assert "--embeddings" in e.command and _arg(e.command, "--cache-ram") == "0"
        assert e.n_ctx == 2048
        texts = ["search_query: What is a cat?", "search_document: A cat is a small domesticated feline.",
                 "search_document: Quarterly revenue grew by 12 percent."]
        vecs = await e.embed(texts)
        assert len(vecs) == 3 and all(len(v) == 768 for v in vecs)
        for v in vecs:
            assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-3)

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b))

        assert cos(vecs[0], vecs[1]) > cos(vecs[0], vecs[2])
        again = await e.embed([texts[0]])
        assert cos(again[0], vecs[0]) > 0.999
        with pytest.raises(ContextOverflowError):
            await e.embed(["word " * 3000])
    finally:
        await e.stop()


# --------------------------------------------------------------------------- #
# The sandbox around the process (localcode.security.sandbox)
# --------------------------------------------------------------------------- #


def test_the_engine_declares_what_the_model_process_may_touch(tmp_path: Path) -> None:
    weights = tmp_path / "models" / "x.gguf"
    weights.parent.mkdir()
    weights.write_bytes(b"GGUF")
    engine = _engine(model=_model(str(weights)))

    plan = engine._sandbox_plan()

    assert plan is not None
    assert weights.parent in plan.read          # its own weights
    assert plan.write == []                     # llama-server writes nothing of its own
    assert plan.local_network is True           # ...but it answers on localhost


def test_the_projector_directory_is_readable_too(tmp_path: Path) -> None:
    weights = tmp_path / "w" / "x.gguf"
    proj = tmp_path / "p" / "mmproj.gguf"
    for f in (weights, proj):
        f.parent.mkdir()
        f.write_bytes(b"GGUF")
    plan = _engine(model=_model(str(weights), mmproj=str(proj)))._sandbox_plan()
    assert plan is not None and proj.parent in plan.read


@pytest.mark.skipif(not os.path.exists("/usr/bin/sandbox-exec"), reason="tylko macOS")
def test_the_spawned_command_is_wrapped_in_the_sandbox(tmp_path: Path) -> None:
    engine = _engine()
    engine._settings = Settings()

    argv = engine._sandboxed_command(["/usr/bin/llama-server", "-m", "x.gguf"])

    assert engine.sandboxed is True
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[-3:] == ["/usr/bin/llama-server", "-m", "x.gguf"]
    assert engine._sandbox_profile is not None and engine._sandbox_profile.exists()
    from localcode.security import sandbox as sb
    sb.cleanup(engine._sandbox_profile)


def test_switching_the_sandbox_off_leaves_the_command_alone() -> None:
    engine = _engine()
    settings = Settings()
    settings.security.sandbox = "off"
    engine._settings = settings

    argv = engine._sandboxed_command(["/usr/bin/llama-server"])

    assert argv == ["/usr/bin/llama-server"]
    assert engine.sandboxed is False


def test_a_failed_sandboxed_start_falls_back_once_in_auto_mode() -> None:
    """A model the user can no longer load is worse than one without isolation."""
    engine = _engine()
    engine._settings = Settings()
    engine.sandboxed = True

    assert engine._sandbox_fallback() is True      # first failure: try without it
    assert engine._sandbox_off is True
    assert engine._sandbox_fallback() is False     # ...and only once


def test_strict_mode_never_falls_back() -> None:
    engine = _engine()
    settings = Settings()
    settings.security.sandbox = "strict"
    engine._settings = settings
    engine.sandboxed = True

    assert engine._sandbox_fallback() is False
    assert engine._sandbox_off is False


def test_the_parent_guard_shim_runs_isolated() -> None:
    """``-I`` keeps the supervisor to the standard library.

    Without it Python puts the working directory on ``sys.path``; in a
    development install that is inside ``~/Documents``, which the sandbox denies,
    and the shim cannot even import its own modules.
    """
    from localcode.runtime.http_engine import _guarded_command

    argv = _guarded_command(["/usr/bin/llama-server", "-m", "x"])
    assert argv[1] == "-I" and argv[2] == "-c"
    assert argv[-2:] == ["-m", "x"]
