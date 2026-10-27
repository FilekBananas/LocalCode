"""End-to-end tests of ``RuntimeManager`` against the *real* engines.

``tests/runtime/test_manager.py`` checks the decision logic with a fake engine;
this module verifies the same manager driving actual model processes, because
that is where the interesting mistakes live (memory measured vs. estimated,
per-slot parallelism really giving two concurrent generations, eviction really
killing a subprocess).  Everything here uses the manager's *default* engine
factory, so ``LlamaServerEngine`` (Homebrew ``llama-server``) and ``MlxEngine``
(``python -m localcode.runtime.mlx_worker``) are picked by model format exactly
as in production.

Only the tiny Qwen3-0.6B models are used (≈0.37 GB GGUF, ≈0.31 GB MLX); at most
two instances run at a time, well under the ~6 GB other test suites may be using
in parallel.  Every fixture stops its processes in a ``finally``.

The MLX half is skipped with a clear reason while ``localcode/runtime/mlx_engine.py``
does not exist yet (it is written by another engineer); the check is done at call
time, so the tests start working as soon as the module lands.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
from pathlib import Path

import pytest
import pytest_asyncio

from lc_testing import TEST_MODELS, model_path
from localcode.config import Settings
from localcode.hub.metadata import detect_capabilities, read_gguf_meta, read_mlx_meta
from localcode.hub.types import LocalModel
from localcode.runtime import memory as mem
from localcode.runtime.base import CompletionChunk, EngineError, LoadConfig, SamplingParams
from localcode.runtime.llama_engine import find_llama_server
from localcode.runtime.manager import RuntimeManager

pytestmark = pytest.mark.real_model

GGUF_KEY = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
MLX_KEY = "mlx-community/Qwen3-0.6B-4bit"

LLAMA = find_llama_server(Settings())
requires_llama = pytest.mark.skipif(LLAMA is None, reason="llama-server not installed")

# real-model tests share one event loop with the module-scoped manager fixture
module_loop = pytest.mark.asyncio(loop_scope="module")

START_TIMEOUT_S = 180.0      # cold start of llama-server / the MLX worker on a busy machine


def _mlx_engine_missing() -> bool:
    try:
        return importlib.util.find_spec("localcode.runtime.mlx_engine") is None
    except (ImportError, ValueError):
        return True


def requires_mlx_engine() -> None:
    """Skip while another engineer is still writing ``runtime/mlx_engine.py``."""
    if _mlx_engine_missing():
        pytest.skip("localcode/runtime/mlx_engine.py does not exist yet (MLX engine still in progress)")
    if importlib.util.find_spec("mlx_lm") is None:
        pytest.skip("mlx-lm is not installed")


# --------------------------------------------------------------------------- #
# Library and manager fixtures
# --------------------------------------------------------------------------- #


def _gguf_model(path: Path) -> LocalModel:
    meta = read_gguf_meta(path)
    return LocalModel(
        key=GGUF_KEY, format="gguf", path=str(path), display_name="Qwen3 0.6B GGUF",
        size_bytes=path.stat().st_size, repo_id="unsloth/Qwen3-0.6B-GGUF", quant=meta.quant,
        capabilities=detect_capabilities(meta, [path.name], name=path.name), meta=meta)


def _mlx_model(path: Path) -> LocalModel:
    meta = read_mlx_meta(path)
    size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return LocalModel(
        key=MLX_KEY, format="mlx", path=str(path), display_name="Qwen3 0.6B MLX",
        size_bytes=size, repo_id=MLX_KEY, quant=meta.quant,
        capabilities=detect_capabilities(meta, name=path.name), meta=meta)


class RealLibrary:
    """``ModelLibraryLike`` over the downloaded test models (no disk scan)."""

    def __init__(self) -> None:
        self.models: dict[str, LocalModel] = {}
        gguf = TEST_MODELS["gguf_tiny"]
        if gguf.is_file():
            self.models[GGUF_KEY] = _gguf_model(gguf)
        mlx = TEST_MODELS["mlx_tiny"]
        if (mlx / "config.json").is_file():
            self.models[MLX_KEY] = _mlx_model(mlx)

    def get(self, key: str) -> LocalModel | None:
        return self.models.get(key)

    def load_defaults(self, key: str) -> dict | None:
        return None


@pytest.fixture(scope="module")
def lc_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("lc-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALCODE_HOME", str(home))
        yield home


@pytest.fixture(scope="module")
def library(lc_home) -> RealLibrary:
    return RealLibrary()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def rt(lc_home, library):
    if LLAMA is None:
        pytest.skip("llama-server not installed")
    settings = Settings()
    settings.runtime.ram_limit_gb = 6.0
    settings.runtime.default_ctx = 1024
    settings.runtime.default_parallel = 2
    settings.runtime.max_loaded_models = 2
    settings.runtime.max_instances_per_model = 1
    settings.runtime.idle_ttl_minutes = 0
    events: list[dict] = []
    manager = RuntimeManager(settings, library, on_event=events.append)
    manager.events = events            # type: ignore[attr-defined]
    try:
        yield manager
    finally:
        await manager.shutdown()


@pytest_asyncio.fixture(loop_scope="module")
async def clean_rt(rt):
    """A manager with nothing loaded and the default limits restored."""
    await rt.unload_all()
    rt.settings.runtime.ram_limit_gb = 6.0
    rt.settings.runtime.default_ctx = 1024
    rt.settings.runtime.default_parallel = 2
    rt.settings.runtime.max_loaded_models = 2
    rt.settings.runtime.max_instances_per_model = 1
    try:
        yield rt
    finally:
        await rt.unload_all()


async def _collect(iterator) -> list[CompletionChunk]:
    chunks: list[CompletionChunk] = []
    async with contextlib.aclosing(iterator) as stream:
        async for chunk in stream:
            chunks.append(chunk)
    return chunks


async def _generate(lease, text: str, max_tokens: int = 24) -> tuple[str, CompletionChunk]:
    prompt = await lease.engine.apply_template(
        [{"role": "user", "content": text}], template_kwargs={"enable_thinking": False})
    chunks = await _collect(lease.engine.complete(
        prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=7)))
    lease.record_stats(chunks[-1].stats)
    return "".join(c.text for c in chunks), chunks[-1]


def _instance(status: dict, key: str) -> dict:
    return next(i for i in status["instances"] if i["modelKey"] == key)


INSTANCE_KEYS = {"id", "modelKey", "displayName", "format", "state", "nCtx", "nParallel",
                 "busySlots", "memoryBytes", "estimatedBytes", "loadedAt", "lastUsedAt",
                 "pid", "port", "genTps", "error"}


def _check_instance_dto(inst: dict, model: LocalModel) -> None:
    assert set(inst) == INSTANCE_KEYS
    assert inst["modelKey"] == model.key and inst["format"] == model.format
    assert inst["displayName"] == model.display_name and inst["error"] is None
    assert inst["state"] in ("ready", "busy")
    assert inst["nCtx"] >= 1024 and inst["nParallel"] == 2
    assert inst["pid"] and inst["port"] and inst["loadedAt"] and inst["lastUsedAt"]
    # the process really occupies roughly what runtime/memory.py predicted
    assert inst["estimatedBytes"] > model.size_bytes
    assert model.size_bytes * 0.5 < inst["memoryBytes"] < inst["estimatedBytes"] * 3


# --------------------------------------------------------------------------- #
# Loading real engines
# --------------------------------------------------------------------------- #


@requires_llama
@module_loop
async def test_real_gguf_load_status_and_completion(clean_rt, library):
    model_path("gguf_tiny")
    info = await asyncio.wait_for(clean_rt.load(GGUF_KEY), START_TIMEOUT_S)
    assert info["state"] == "ready"
    status = clean_rt.status()
    assert set(status) == {"instances", "system", "budget", "settings"}
    _check_instance_dto(_instance(status, GGUF_KEY), library.get(GGUF_KEY))
    assert status["budget"]["limitBytes"] == int(6.0 * mem.GiB)
    assert 0 < status["budget"]["usedBytes"] <= status["budget"]["limitBytes"]
    assert status["system"]["totalBytes"] > status["system"]["availableBytes"] > 0
    assert status["settings"]["defaultParallel"] == 2 and status["settings"]["maxLoadedModels"] == 2

    async with clean_rt.acquire(GGUF_KEY) as lease:
        assert lease.model_key == GGUF_KEY and lease.n_ctx == _instance(status, GGUF_KEY)["nCtx"]
        text, final = await _generate(lease, "Say 'hello' and nothing else.")
        assert text.strip() and final.done and final.stats.completion_tokens > 0
    assert clean_rt.status()["instances"][0]["genTps"] > 0


@requires_llama
@module_loop
async def test_real_mlx_load_status_and_completion(clean_rt, library):
    requires_mlx_engine()
    model_path("mlx_tiny")
    info = await asyncio.wait_for(clean_rt.load(MLX_KEY), START_TIMEOUT_S)
    assert info["state"] == "ready" and info["format"] == "mlx"
    _check_instance_dto(_instance(clean_rt.status(), MLX_KEY), library.get(MLX_KEY))
    async with clean_rt.acquire(MLX_KEY) as lease:
        text, final = await _generate(lease, "Say 'hello' and nothing else.")
        assert text.strip() and final.done and final.stats.completion_tokens > 0


@requires_llama
@module_loop
async def test_real_two_formats_loaded_at_once(clean_rt, library):
    requires_mlx_engine()
    model_path("gguf_tiny")
    model_path("mlx_tiny")
    await asyncio.wait_for(asyncio.gather(clean_rt.load(GGUF_KEY), clean_rt.load(MLX_KEY)),
                           START_TIMEOUT_S)
    status = clean_rt.status()
    assert len(status["instances"]) == 2
    assert {i["format"] for i in status["instances"]} == {"gguf", "mlx"}
    for key in (GGUF_KEY, MLX_KEY):
        _check_instance_dto(_instance(status, key), library.get(key))
    pids = {i["pid"] for i in status["instances"]}
    ports = {i["port"] for i in status["instances"]}
    assert len(pids) == 2 and len(ports) == 2
    reserved = sum(max(i["memoryBytes"], i["estimatedBytes"]) for i in status["instances"])
    assert status["budget"]["usedBytes"] == reserved <= status["budget"]["limitBytes"]
    # both engines answer independently
    async with clean_rt.acquire(GGUF_KEY) as gguf, clean_rt.acquire(MLX_KEY) as mlx:
        assert gguf.instance_id != mlx.instance_id
        answers = await asyncio.gather(_generate(gguf, "Reply with the word ok."),
                                       _generate(mlx, "Reply with the word ok."))
    assert all(text.strip() for text, _ in answers)


# --------------------------------------------------------------------------- #
# Parallel slots
# --------------------------------------------------------------------------- #


@requires_llama
@module_loop
async def test_real_two_parallel_requests_on_one_instance(clean_rt):
    model_path("gguf_tiny")
    first = await asyncio.wait_for(clean_rt.acquire(GGUF_KEY), START_TIMEOUT_S)
    second = await clean_rt.acquire(GGUF_KEY)
    third = asyncio.create_task(clean_rt.acquire(GGUF_KEY))      # no free slot -> FIFO queue
    try:
        assert first.instance_id == second.instance_id           # one process, two slots
        inst = clean_rt.status()["instances"][0]
        assert inst["state"] == "busy" and inst["busySlots"] == 2 and inst["nParallel"] == 2
        await asyncio.sleep(0.2)
        assert not third.done()
        results = await asyncio.gather(
            _generate(first, "Count from one to ten in words."),
            _generate(second, "List three colours, comma separated."))
        assert all(text.strip() for text, _ in results)
        assert all(final.stats.completion_tokens > 0 for _, final in results)
        assert len(clean_rt.status()["instances"]) == 1           # no extra copy was started
        first.release()
        lease = await asyncio.wait_for(third, 30)
        assert lease.instance_id == second.instance_id
        lease.release()
    finally:
        if not third.done():
            third.cancel()
            with contextlib.suppress(asyncio.CancelledError, EngineError):
                await third
        second.release()
        first.release()
    assert clean_rt.status()["instances"][0]["busySlots"] == 0


# --------------------------------------------------------------------------- #
# Limits: eviction and the RAM budget
# --------------------------------------------------------------------------- #


async def _load_two_instances(rt) -> tuple[dict, dict]:
    """Two loaded instances: GGUF + MLX when available, else two GGUF copies."""
    model_path("gguf_tiny")
    older = await asyncio.wait_for(rt.load(GGUF_KEY), START_TIMEOUT_S)
    if not _mlx_engine_missing() and importlib.util.find_spec("mlx_lm") is not None:
        newer = await asyncio.wait_for(rt.load(MLX_KEY), START_TIMEOUT_S)
    else:
        rt.settings.runtime.max_instances_per_model = 2
        newer = await asyncio.wait_for(rt.load(GGUF_KEY, new_instance=True), START_TIMEOUT_S)
    return older, newer


@requires_llama
@module_loop
async def test_real_update_settings_evicts_the_idle_lru_instance(clean_rt):
    older, newer = await _load_two_instances(clean_rt)
    async with clean_rt.acquire(newer["modelKey"]) as lease:   # make `newer` the most recent
        assert lease.instance_id == newer["id"]
    assert len(clean_rt.status()["instances"]) == 2

    status = await clean_rt.update_settings({"maxLoadedModels": 1})
    assert status["settings"]["maxLoadedModels"] == 1
    for _ in range(100):                                       # the stop runs in the background
        if len(clean_rt.status()["instances"]) <= 1:
            break
        await asyncio.sleep(0.05)
    remaining = clean_rt.status()["instances"]
    assert [i["id"] for i in remaining] == [newer["id"]]        # least recently used one evicted
    assert remaining[0]["state"] == "ready"
    assert clean_rt.settings.runtime.max_loaded_models == 1
    with pytest.raises(EngineError):                           # `older` really is gone
        await clean_rt.unload(older["id"])
    await clean_rt.update_settings({"maxLoadedModels": 2})


@requires_llama
@module_loop
async def test_real_ram_limit_refuses_to_load(clean_rt, library):
    model_path("gguf_tiny")
    status = await clean_rt.update_settings({"ramLimitGb": 1})
    assert status["budget"]["limitBytes"] == int(1 * mem.GiB)
    big = LoadConfig(n_ctx=8192, n_parallel=2)
    assert mem.breakdown(library.get(GGUF_KEY), big).total > mem.GiB
    with pytest.raises(EngineError) as info:
        await clean_rt.load(GGUF_KEY, big)
    message = str(info.value)
    assert "Qwen3 0.6B GGUF" in message and "limit RAM" in message
    assert "Co możesz zrobić" in message and "zmniejsz kontekst" in message
    assert clean_rt.status()["instances"] == []                # nothing was spawned

    if not _mlx_engine_missing() and importlib.util.find_spec("mlx_lm") is not None:
        with pytest.raises(EngineError, match="limit RAM"):    # MLX needs >1 GB even at 1k ctx
            await clean_rt.load(MLX_KEY)
        assert clean_rt.status()["instances"] == []

    # the same model fits again once the limit is raised
    await clean_rt.update_settings({"ramLimitGb": 6})
    info2 = await asyncio.wait_for(clean_rt.load(GGUF_KEY), START_TIMEOUT_S)
    assert info2["state"] == "ready"


@requires_llama
@module_loop
async def test_real_unload_kills_the_process(clean_rt):
    import psutil

    model_path("gguf_tiny")
    info = await asyncio.wait_for(clean_rt.load(GGUF_KEY), START_TIMEOUT_S)
    pid = info["pid"]
    assert pid and psutil.pid_exists(pid)
    await clean_rt.unload(info["id"])
    assert clean_rt.status()["instances"] == []
    for _ in range(100):
        if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        await asyncio.sleep(0.05)
    else:                                                      # pragma: no cover - diagnostics
        pytest.fail(f"llama-server pid {pid} still running after unload")
