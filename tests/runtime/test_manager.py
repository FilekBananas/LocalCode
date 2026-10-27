"""Tests for runtime/manager.py with a fake engine and a fake library."""

from __future__ import annotations

import asyncio
import collections.abc
import json
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import pytest

from localcode import config as lc_config
from localcode.config import Settings
from localcode.hub.types import LocalModel, MemoryEstimate, ModelMeta
from localcode.runtime import manager as mgr
from localcode.runtime import memory as mem
from localcode.runtime.base import (CompletionChunk, Engine, EngineError, GenStats, LoadConfig,
                                    SamplingParams)
from localcode.runtime.manager import Lease, RuntimeManager, SettingsError

GiB = mem.GiB
MiB = mem.MiB

META = ModelMeta(arch="qwen3", context_length=32768, n_layers=28, n_heads=16, n_kv_heads=8,
                 head_dim=128, hidden_size=1024, vocab_size=151936)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class Behavior:
    delay: float = 0.0
    fail: BaseException | None = None
    n_ctx: int | None = None          # engine reports a different real context
    memory: int | None = None         # memory_bytes() result


class FakeEngine(Engine):
    format = "fake"

    def __init__(self, instance_id: str, model: LocalModel, config: LoadConfig,
                 behavior: Behavior, log: list[tuple[str, str]]) -> None:
        super().__init__(instance_id, model, config)
        self.behavior = behavior
        self.log = log
        self._running = False
        self.stopped = False
        self.start_calls = 0
        self.log.append(("create", instance_id))

    async def start(self) -> None:
        self.start_calls += 1
        if self.behavior.delay:
            await asyncio.sleep(self.behavior.delay)
        if self.behavior.fail is not None:
            raise self.behavior.fail
        self._running = True
        self.log.append(("started", self.instance_id))

    async def stop(self) -> None:
        self._running = False
        self.stopped = True
        self.log.append(("stopped", self.instance_id))

    @property
    def running(self) -> bool:
        return self._running

    @property
    def n_ctx(self) -> int:
        return self.behavior.n_ctx or self.config.n_ctx

    async def apply_template(self, messages, tools=None, *, add_generation_prompt=True,
                             template_kwargs=None) -> str:
        return "\n".join(str(m.get("content", "")) for m in messages)

    async def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    async def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk(text="ok")
        yield CompletionChunk(done=True, finish_reason="stop", stats=GenStats(completion_tokens=1))

    async def memory_bytes(self) -> int | None:
        return self.behavior.memory


class FakeFactory:
    def __init__(self) -> None:
        self.engines: list[FakeEngine] = []
        self.behaviors: dict[str, Behavior] = {}
        self.default = Behavior()
        self.log: list[tuple[str, str]] = []

    def __call__(self, instance_id: str, model: LocalModel, config: LoadConfig) -> Engine:
        engine = FakeEngine(instance_id, model, config, self.behaviors.get(model.key, self.default), self.log)
        self.engines.append(engine)
        return engine

    def of(self, key: str) -> list[FakeEngine]:
        return [e for e in self.engines if e.model.key == key]


class FakeLibrary:
    def __init__(self, models: list[LocalModel], defaults: dict[str, dict] | None = None) -> None:
        self.models = {m.key: m for m in models}
        self.defaults = defaults or {}

    def get(self, key: str) -> LocalModel | None:
        return self.models.get(key)

    def load_defaults(self, key: str) -> dict | None:
        return self.defaults.get(key)


def make_model(key: str, *, size: float = 1.0, fmt: str = "gguf", ctx: int = 32768,
               capabilities: list[str] | None = None) -> LocalModel:
    meta = ModelMeta(**{**META.__dict__, "context_length": ctx})
    return LocalModel(key=key, format=fmt, path=f"/nonexistent/{key}", display_name=key.split("/")[-1],
                      size_bytes=int(size * GiB), meta=meta, capabilities=capabilities or [])


MODELS = [make_model("org/a"), make_model("org/b"), make_model("org/c"),
          make_model("org/m", fmt="mlx"), make_model("org/e", capabilities=["embedding"], ctx=2048),
          make_model("org/img", capabilities=["image"], ctx=0)]


def est_bytes(n_ctx: int = 4096, n_parallel: int = 1, key: str = "org/a") -> int:
    model = next(m for m in MODELS if m.key == key)
    return mem.breakdown(model, LoadConfig(n_ctx=n_ctx, n_parallel=n_parallel)).total


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.runtime.ram_limit_gb = 20
    s.runtime.default_ctx = 4096
    s.runtime.max_loaded_models = 3
    s.runtime.idle_ttl_minutes = 0
    return s


@pytest.fixture
def factory() -> FakeFactory:
    return FakeFactory()


@pytest.fixture
def library() -> FakeLibrary:
    return FakeLibrary(MODELS)


@pytest.fixture
def events() -> list[dict]:
    return []


@pytest.fixture
async def rt(settings, library, factory, events):
    manager = RuntimeManager(settings, library, on_event=events.append, engine_factory=factory)
    try:
        yield manager
    finally:
        await manager.shutdown()


async def settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def instance_keys(manager: RuntimeManager) -> list[str]:
    return sorted(i["modelKey"] for i in manager.status()["instances"] if i["state"] != "error")


# --------------------------------------------------------------------------- #
# Acquire / JIT
# --------------------------------------------------------------------------- #


async def test_jit_load_on_acquire_and_release(rt, factory):
    async with rt.acquire("org/a") as lease:
        assert isinstance(lease, Lease)
        assert lease.engine.running
        assert lease.n_ctx == 4096 and lease.model_key == "org/a"
        inst = rt.status()["instances"][0]
        assert inst["state"] == "busy" and inst["busySlots"] == 1
        assert rt.is_loaded("org/a") and rt.loaded_keys() == {"org/a"}
    inst = rt.status()["instances"][0]
    assert inst["state"] == "ready" and inst["busySlots"] == 0
    # awaitable form reuses the same instance
    lease = await rt.acquire("org/a")
    assert lease.instance_id == inst["id"]
    lease.release()
    lease.release()   # idempotent
    assert rt.status()["instances"][0]["busySlots"] == 0
    assert len(factory.engines) == 1 and factory.engines[0].start_calls == 1


async def test_unknown_model_raises_polish_error(rt):
    with pytest.raises(EngineError, match="Nie znaleziono modelu"):
        await rt.acquire("org/missing")


async def test_acquire_result_behaves_like_a_coroutine(rt, factory):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # "coroutine was never awaited" would fail here
        pending = rt.acquire("org/a")
        assert isinstance(pending, collections.abc.Coroutine)
        del pending                                    # never awaited -> nothing is loaded
    await settle()
    assert rt.status()["instances"] == [] and not factory.engines
    lease = await asyncio.wait_for(asyncio.create_task(rt.acquire("org/a")), 1)
    assert lease.purpose == "chat" and not lease.released
    lease.release()
    assert lease.released


async def test_context_manager_releases_the_slot_on_error(rt):
    with pytest.raises(ZeroDivisionError):
        async with rt.acquire("org/a") as lease:
            assert lease.engine.running
            raise ZeroDivisionError
    assert rt.status()["instances"][0]["busySlots"] == 0


async def test_embedding_purpose_uses_a_separate_instance(rt, factory):
    async with rt.acquire("org/b") as chat_lease:
        async with rt.acquire("org/b", purpose="embedding") as embed_lease:
            assert chat_lease.instance_id != embed_lease.instance_id
            assert embed_lease.config.embedding and not chat_lease.config.embedding
    assert len(factory.of("org/b")) == 2
    async with rt.acquire("org/b", purpose="embedding") as again:      # reused, not reloaded
        assert again.config.embedding
    assert len(factory.of("org/b")) == 2


async def test_title_purpose_waits_instead_of_starting_a_copy(rt, settings, factory):
    settings.runtime.max_instances_per_model = 2
    held = await rt.acquire("org/a")
    waiting = asyncio.create_task(rt.acquire("org/a", purpose="title"))
    await asyncio.sleep(0.05)
    assert not waiting.done() and len(factory.of("org/a")) == 1
    held.release()
    lease = await asyncio.wait_for(waiting, 1)
    assert lease.purpose == "title" and lease.instance_id == held.instance_id
    lease.release()


async def test_title_purpose_does_not_reload_for_a_bigger_context(rt, factory):
    async with rt.acquire("org/a") as lease:
        assert lease.n_ctx == 4096
    async with rt.acquire("org/a", n_ctx=16384, purpose="title") as lease:
        assert lease.n_ctx == 4096
    assert len(factory.of("org/a")) == 1


async def test_concurrent_acquires_trigger_one_load(rt, factory):
    factory.default = Behavior(delay=0.1)
    got: list[str] = []

    async def use() -> None:
        async with rt.acquire("org/a") as lease:
            got.append(lease.instance_id)
            await asyncio.sleep(0.01)

    await asyncio.gather(*(use() for _ in range(5)))
    assert len(factory.engines) == 1
    assert factory.engines[0].start_calls == 1
    assert len(set(got)) == 1 and len(got) == 5


async def test_slot_limit_and_fifo_waiting(rt, settings):
    settings.runtime.default_parallel = 2
    first = await rt.acquire("org/a")
    second = await rt.acquire("org/a")
    assert first.instance_id == second.instance_id
    order: list[int] = []
    leases: dict[int, Lease] = {}

    async def wait_for(i: int) -> None:
        lease = await rt.acquire("org/a")
        order.append(i)
        leases[i] = lease

    tasks = []
    for i in range(3):
        tasks.append(asyncio.create_task(wait_for(i)))
        await settle()
    assert order == [] and rt.status()["instances"][0]["busySlots"] == 2
    first.release()
    await settle()
    assert order == [0]
    second.release()
    await settle()
    assert order == [0, 1]
    leases[0].release()
    await settle()
    assert order == [0, 1, 2]
    assert rt.status()["instances"][0]["busySlots"] == 2
    await asyncio.gather(*tasks)
    for lease in leases.values():
        lease.release()
    assert rt.status()["instances"][0]["busySlots"] == 0


async def test_second_instance_when_allowed(rt, settings, factory):
    settings.runtime.max_instances_per_model = 2
    one = await rt.acquire("org/a")
    two = await rt.acquire("org/a")
    assert one.instance_id != two.instance_id
    assert len(factory.of("org/a")) == 2
    third = asyncio.create_task(rt.acquire("org/a"))
    await asyncio.sleep(0.05)
    assert not third.done() and len(factory.of("org/a")) == 2   # limit reached -> waits
    two.release()
    lease3 = await asyncio.wait_for(third, 1)
    assert lease3.instance_id == two.instance_id
    lease3.release()
    one.release()


async def test_cold_start_loads_once_then_scales_up(rt, settings, factory):
    settings.runtime.max_instances_per_model = 2
    factory.default = Behavior(delay=0.05)
    hold = asyncio.Event()
    ids: list[str] = []

    async def use() -> None:
        async with rt.acquire("org/a") as lease:
            ids.append(lease.instance_id)
            await hold.wait()

    tasks = [asyncio.create_task(use()) for _ in range(2)]
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(ids) == 2:
            break
    assert len(set(ids)) == 2
    first_id = factory.engines[0].instance_id
    # the second copy was created only after the first one had started
    assert factory.log.index(("started", first_id)) < factory.log.index(("create", factory.engines[1].instance_id))
    hold.set()
    await asyncio.gather(*tasks)


async def test_scale_up_never_evicts_other_models(rt, settings, factory):
    settings.runtime.max_instances_per_model = 2
    settings.runtime.max_loaded_models = 2
    async with rt.acquire("org/b"):
        pass
    held = await rt.acquire("org/a")
    waiting = asyncio.create_task(rt.acquire("org/a"))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    assert not factory.of("org/b")[0].stopped
    assert len(factory.of("org/a")) == 1
    held.release()
    lease = await asyncio.wait_for(waiting, 1)
    assert lease.instance_id == held.instance_id
    lease.release()


# --------------------------------------------------------------------------- #
# Limits and eviction
# --------------------------------------------------------------------------- #


async def test_lru_eviction_by_max_loaded_models(rt, settings, factory):
    settings.runtime.max_loaded_models = 2
    for key in ("org/a", "org/b", "org/a"):      # b becomes least recently used
        async with rt.acquire(key):
            await asyncio.sleep(0.01)
    async with rt.acquire("org/c"):
        pass
    await settle()
    assert factory.of("org/b")[0].stopped
    assert not factory.of("org/a")[0].stopped
    assert instance_keys(rt) == ["org/a", "org/c"]


async def test_ram_budget_eviction(rt, settings, factory):
    settings.runtime.max_loaded_models = 5
    settings.runtime.ram_limit_gb = 2.5 * est_bytes() / GiB
    for key in ("org/a", "org/b"):
        async with rt.acquire(key):
            await asyncio.sleep(0.01)
    async with rt.acquire("org/c"):
        pass
    assert factory.of("org/a")[0].stopped
    assert instance_keys(rt) == ["org/b", "org/c"]
    status = rt.status()
    assert status["budget"]["usedBytes"] <= status["budget"]["limitBytes"]


async def test_ram_budget_refusal_with_polish_message(rt, settings, factory):
    settings.runtime.max_loaded_models = 5
    settings.runtime.ram_limit_gb = 2.5 * est_bytes() / GiB
    a = await rt.acquire("org/a")
    b = await rt.acquire("org/b")
    with pytest.raises(EngineError) as info:
        await rt.acquire("org/c")
    message = str(info.value)
    assert "limicie RAM" in message and "Co możesz zrobić" in message
    assert "a (" in message and "b (" in message                  # busy models are named
    assert "podnieś limit RAM" in message
    assert not factory.of("org/a")[0].stopped and not factory.of("org/b")[0].stopped
    assert not factory.of("org/c")
    a.release()
    b.release()
    # a model that can never fit the whole budget
    settings.runtime.ram_limit_gb = 0.5 * est_bytes() / GiB + 0.01
    with pytest.raises(EngineError, match="cały limit RAM"):
        await rt.acquire("org/c")


async def test_refusal_suggests_smaller_context(rt, settings):
    settings.runtime.ram_limit_gb = 1.2 * est_bytes(4096) / GiB
    with pytest.raises(EngineError) as info:
        await rt.load("org/a", LoadConfig(n_ctx=32768))
    assert "zmniejsz kontekst" in str(info.value)


async def test_busy_instance_is_never_evicted(rt, settings, factory):
    settings.runtime.max_loaded_models = 1
    held = await rt.acquire("org/a")
    with pytest.raises(EngineError, match="limit załadowanych modeli"):
        await rt.acquire("org/b")
    assert not factory.of("org/a")[0].stopped
    held.release()
    async with rt.acquire("org/b"):
        pass
    assert factory.of("org/a")[0].stopped


async def test_evict_policy_none_refuses(rt, settings, factory):
    settings.runtime.max_loaded_models = 1
    settings.runtime.evict_policy = "none"
    async with rt.acquire("org/a"):
        pass
    with pytest.raises(EngineError, match="wyłączone"):
        await rt.acquire("org/b")
    assert not factory.of("org/a")[0].stopped


async def test_loading_instances_are_not_evicted(rt, settings, factory):
    settings.runtime.max_loaded_models = 1
    factory.behaviors["org/a"] = Behavior(delay=0.2)
    task = asyncio.create_task(rt.acquire("org/a"))
    await asyncio.sleep(0.02)
    with pytest.raises(EngineError):
        await rt.acquire("org/b")
    lease = await task
    lease.release()


# --------------------------------------------------------------------------- #
# Idle TTL, failures, cancellation
# --------------------------------------------------------------------------- #


async def test_idle_ttl_reaper(rt, settings, factory, monkeypatch):
    monkeypatch.setattr(mgr, "REAPER_INTERVAL_S", 0.02)
    settings.runtime.idle_ttl_minutes = 0.1 / 60          # 100 ms
    async with rt.acquire("org/a"):
        pass
    held = await rt.acquire("org/b")
    await asyncio.sleep(0.4)
    assert factory.of("org/a")[0].stopped
    assert not factory.of("org/b")[0].stopped             # busy instances are never reaped
    assert instance_keys(rt) == ["org/b"]
    held.release()
    await asyncio.sleep(0.4)
    assert instance_keys(rt) == []


async def test_ttl_zero_never_unloads(rt, settings, factory, monkeypatch):
    monkeypatch.setattr(mgr, "REAPER_INTERVAL_S", 0.02)
    settings.runtime.idle_ttl_minutes = 0
    async with rt.acquire("org/a"):
        pass
    await asyncio.sleep(0.1)
    assert instance_keys(rt) == ["org/a"]


async def test_load_failure_propagates_to_all_waiters(rt, factory, monkeypatch, events):
    factory.behaviors["org/a"] = Behavior(delay=0.05, fail=EngineError("Plik modelu jest uszkodzony."))
    results = await asyncio.gather(*(rt.acquire("org/a") for _ in range(3)), return_exceptions=True)
    assert all(isinstance(r, EngineError) and "uszkodzony" in str(r) for r in results)
    assert len(factory.engines) == 1
    errors = [i for i in rt.status()["instances"] if i["state"] == "error"]
    assert len(errors) == 1 and "uszkodzony" in errors[0]["error"]
    await settle()
    assert any(any(i["state"] == "error" for i in e["instances"]) for e in events)
    monkeypatch.setattr(mgr, "ERROR_TTL_S", 0.0)
    assert rt.status()["instances"] == []                 # already surfaced once -> gone
    # a later attempt tries again
    factory.behaviors["org/a"] = Behavior()
    async with rt.acquire("org/a") as lease:
        assert lease.engine.running


async def test_unexpected_start_exception_is_wrapped(rt, factory):
    factory.behaviors["org/a"] = Behavior(fail=RuntimeError("socket exploded"))
    with pytest.raises(EngineError, match="Nie udało się uruchomić modelu a: socket exploded"):
        await rt.acquire("org/a")
    assert factory.engines[0].stopped


async def test_cancelled_waiter_releases_nothing(rt):
    held = await rt.acquire("org/a")
    cancelled = asyncio.create_task(rt.acquire("org/a"))
    survivor = asyncio.create_task(rt.acquire("org/a"))
    await settle()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    held.release()
    lease = await asyncio.wait_for(survivor, 1)
    assert rt.status()["instances"][0]["busySlots"] == 1
    lease.release()
    assert rt.status()["instances"][0]["busySlots"] == 0


async def test_cancel_after_slot_granted_gives_it_back(rt):
    held = await rt.acquire("org/a")
    waiter = asyncio.create_task(rt.acquire("org/a"))
    await settle()
    held.release()                 # grants the slot to `waiter` synchronously
    waiter.cancel()                # ... which is cancelled before it could resume
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert rt.status()["instances"][0]["busySlots"] == 0
    async with rt.acquire("org/a"):
        assert rt.status()["instances"][0]["busySlots"] == 1


async def test_cancel_during_load_keeps_loading(rt, factory):
    factory.default = Behavior(delay=0.1)
    task = asyncio.create_task(rt.acquire("org/a"))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.15)
    inst = rt.status()["instances"][0]
    assert inst["state"] == "ready" and inst["busySlots"] == 0
    async with rt.acquire("org/a"):
        pass
    assert len(factory.engines) == 1


# --------------------------------------------------------------------------- #
# Context size handling and config resolution
# --------------------------------------------------------------------------- #


async def test_idle_instance_reloaded_with_bigger_context(rt, factory):
    async with rt.acquire("org/a") as lease:
        assert lease.n_ctx == 4096
    async with rt.acquire("org/a", n_ctx=8192) as lease:
        assert lease.n_ctx == 8192
    engines = factory.of("org/a")
    assert len(engines) == 2 and engines[0].stopped and engines[1].config.n_ctx == 8192
    # a smaller request reuses the bigger instance
    async with rt.acquire("org/a", n_ctx=2048) as lease:
        assert lease.n_ctx == 8192
    assert len(factory.of("org/a")) == 2


async def test_busy_instance_with_small_context_is_used(rt, settings, factory):
    settings.runtime.default_parallel = 2
    held = await rt.acquire("org/a")
    async with rt.acquire("org/a", n_ctx=16384) as lease:
        assert lease.n_ctx == 4096 and lease.instance_id == held.instance_id
    held.release()
    assert len(factory.engines) == 1


async def test_context_upgrade_that_does_not_fit_uses_existing(rt, settings, factory):
    settings.runtime.ram_limit_gb = 1.2 * est_bytes(4096) / GiB
    async with rt.acquire("org/a"):
        pass
    async with rt.acquire("org/a", n_ctx=32768) as lease:
        assert lease.n_ctx == 4096
    assert len(factory.engines) == 1 and not factory.engines[0].stopped


async def test_request_is_clamped_to_native_context(rt, factory):
    async with rt.acquire("org/e", n_ctx=65536, purpose="embedding") as lease:
        assert lease.n_ctx == 2048
    async with rt.acquire("org/e", n_ctx=65536, purpose="embedding"):
        pass
    assert len(factory.engines) == 1


async def test_engine_reported_context_is_exposed(rt, factory):
    factory.behaviors["org/a"] = Behavior(n_ctx=3000)
    async with rt.acquire("org/a") as lease:
        assert lease.n_ctx == 3000 and lease.engine.n_ctx == 3000
    assert rt.status()["instances"][0]["nCtx"] == 3000


async def test_config_resolution(settings, factory):
    library = FakeLibrary(MODELS, defaults={"org/a": {"nCtx": 6144, "nParallel": 2, "kvCacheType": "q8_0",
                                                      "flashAttn": "on", "gpuLayers": None}})
    settings.runtime.kv_cache_type = "q8_0"
    manager = RuntimeManager(settings, library, engine_factory=factory)
    try:
        async with manager.acquire("org/a"):
            pass
        cfg = factory.of("org/a")[0].config
        assert (cfg.n_ctx, cfg.n_parallel, cfg.kv_cache_type, cfg.flash_attn, cfg.gpu_layers) == \
            (6144, 2, "q8_0", "on", -1)
        async with manager.acquire("org/m"):                 # MLX: kv type translated
            pass
        assert factory.of("org/m")[0].config.kv_cache_type == "8bit"
        async with manager.acquire("org/b", purpose="embedding"):
            pass
        assert factory.of("org/b")[0].config.embedding is True
        # An embedding model is only acquirable for embeddings (see
        # `_check_can_generate`); its LoadConfig gets `embedding` either way.
        async with manager.acquire("org/e", purpose="embedding"):
            pass
        assert factory.of("org/e")[0].config.embedding is True
        info = await manager.load("org/c", LoadConfig(n_ctx=2048, n_parallel=3))
        assert (info["nCtx"], info["nParallel"]) == (2048, 3)
    finally:
        await manager.shutdown()


async def test_auto_context_when_nothing_configured(settings, library, factory):
    settings.runtime.default_ctx = None
    manager = RuntimeManager(settings, library, engine_factory=factory)
    try:
        async with manager.acquire("org/a") as lease:
            assert lease.n_ctx == 32768                    # min(native 32768, default cap 32768)
        settings.runtime.ram_limit_gb = 3.0
        async with manager.acquire("org/b") as lease:
            ctx = lease.n_ctx
        assert 2048 <= ctx < 32768 and ctx % 1024 == 0
    finally:
        await manager.shutdown()


# --------------------------------------------------------------------------- #
# Explicit load / unload / shutdown
# --------------------------------------------------------------------------- #


async def test_load_and_unload(rt, settings, factory):
    info = await rt.load("org/a")
    assert info["state"] == "ready" and info["modelKey"] == "org/a"
    again = await rt.load("org/a")
    assert again["id"] == info["id"] and len(factory.engines) == 1
    with pytest.raises(EngineError, match="limit instancji"):
        await rt.load("org/a", new_instance=True)
    settings.runtime.max_instances_per_model = 2
    second = await rt.load("org/a", new_instance=True)
    assert second["id"] != info["id"]
    await rt.unload(info["id"])
    assert factory.engines[0].stopped
    assert [i["id"] for i in rt.status()["instances"]] == [second["id"]]
    with pytest.raises(EngineError, match="Nie ma załadowanej instancji"):
        await rt.unload("nope-1")
    await rt.unload_all()
    assert rt.status()["instances"] == []


async def test_load_with_different_config_replaces_idle_instance(rt, factory):
    first = await rt.load("org/a")
    replaced = await rt.load("org/a", LoadConfig(n_ctx=8192))
    assert replaced["id"] != first["id"] and replaced["nCtx"] == 8192
    assert factory.engines[0].stopped
    assert len(rt.status()["instances"]) == 1


async def test_load_without_waiting(rt, factory):
    factory.default = Behavior(delay=0.05)
    info = await rt.load("org/a", wait=False)
    assert info["state"] == "loading" and info["loadedAt"] is None
    await asyncio.sleep(0.1)
    assert rt.status()["instances"][0]["state"] == "ready"


async def test_load_failure_raises(rt, factory):
    factory.default = Behavior(fail=EngineError("Brak pamięci GPU."))
    with pytest.raises(EngineError, match="Brak pamięci GPU"):
        await rt.load("org/a")


async def test_unload_busy_waits_for_release(rt, factory, monkeypatch):
    monkeypatch.setattr(mgr, "UNLOAD_GRACE_S", 5.0)
    lease = await rt.acquire("org/a")
    task = asyncio.create_task(rt.unload(lease.instance_id))
    await asyncio.sleep(0.05)
    assert not task.done()
    assert rt.status()["instances"][0]["state"] == "stopping"
    assert not factory.engines[0].stopped
    lease.release()
    await asyncio.wait_for(task, 1)
    assert factory.engines[0].stopped and rt.status()["instances"] == []


async def test_unload_busy_after_grace_period(rt, factory, monkeypatch):
    monkeypatch.setattr(mgr, "UNLOAD_GRACE_S", 0.05)
    lease = await rt.acquire("org/a")
    started = time.monotonic()
    await rt.unload(lease.instance_id)
    assert time.monotonic() - started >= 0.04
    assert factory.engines[0].stopped
    lease.release()                                   # harmless afterwards
    assert rt.status()["instances"] == []


async def test_unload_fails_orphaned_waiters(rt, monkeypatch):
    monkeypatch.setattr(mgr, "UNLOAD_GRACE_S", 0.05)
    lease = await rt.acquire("org/a")
    waiter = asyncio.create_task(rt.acquire("org/a"))
    await settle()
    await rt.unload(lease.instance_id)
    with pytest.raises(EngineError, match="wyładowany"):
        await waiter


async def test_unload_during_load(rt, factory):
    factory.default = Behavior(delay=0.2)
    info = await rt.load("org/a", wait=False)
    waiter = asyncio.create_task(rt.acquire("org/a"))
    await asyncio.sleep(0.02)
    await rt.unload(info["id"])
    assert rt.status()["instances"] == []
    with pytest.raises(EngineError):
        await waiter


async def test_shutdown_stops_everything(settings, library, factory):
    manager = RuntimeManager(settings, library, engine_factory=factory)
    lease = await manager.acquire("org/a")
    waiter = asyncio.create_task(manager.acquire("org/a"))
    await settle()
    await manager.shutdown()
    with pytest.raises(EngineError, match="zamknięte"):
        await waiter
    assert factory.engines[0].stopped
    with pytest.raises(EngineError, match="zamknięte"):
        await manager.acquire("org/a")
    lease.release()
    await manager.shutdown()                          # idempotent


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


async def test_update_settings_persists_and_applies(isolated_home, rt, settings):
    status = await rt.update_settings({"maxLoadedModels": 4, "ramLimitGb": 12, "kvCacheType": "q8_0",
                                       "defaultCtx": None, "flashAttn": "off", "idleTtlMinutes": 5,
                                       "evictPolicy": "none", "gpuLayers": 20, "defaultParallel": 2,
                                       "maxInstancesPerModel": 2, "llamaServerPath": ""})
    assert status["settings"] == {
        "maxLoadedModels": 4, "maxInstancesPerModel": 2, "defaultParallel": 2, "ramLimitGb": 12.0,
        "idleTtlMinutes": 5, "defaultCtx": None, "flashAttn": "off", "kvCacheType": "q8_0",
        "gpuLayers": 20, "evictPolicy": "none", "pinnedModels": []}
    assert settings.runtime.max_loaded_models == 4 and settings.runtime.llama_server_path is None
    stored = json.loads((isolated_home / "settings.json").read_text())["runtime"]
    assert stored["max_loaded_models"] == 4 and stored["ram_limit_gb"] == 12.0
    assert stored["kv_cache_type"] == "q8_0" and stored["evict_policy"] == "none"
    assert lc_config.load_settings().runtime.default_parallel == 2
    # snake_case keys and executable paths are accepted too
    status = await rt.update_settings({"max_loaded_models": 2, "pythonPath": sys.executable,
                                       "ramLimitGb": None})
    assert status["settings"]["maxLoadedModels"] == 2 and status["settings"]["ramLimitGb"] is None
    assert lc_config.load_settings().runtime.python_path == sys.executable


@pytest.mark.parametrize("patch", [
    {"maxLoadedModels": 0}, {"maxLoadedModels": 1.5}, {"maxLoadedModels": True},
    {"maxInstancesPerModel": 99}, {"defaultParallel": "x"}, {"ramLimitGb": 0.2},
    {"ramLimitGb": 100000}, {"ramLimitGb": "lots"}, {"idleTtlMinutes": -1}, {"defaultCtx": 100},
    {"flashAttn": "maybe"}, {"kvCacheType": "q3_k"}, {"evictPolicy": "fifo"}, {"gpuLayers": -5},
    {"llamaServerPath": "/nonexistent/llama-server"}, {"unknownKey": 1}, {"pythonPath": 42},
])
async def test_update_settings_validation(isolated_home, rt, patch):
    with pytest.raises(SettingsError):
        await rt.update_settings(patch)
    assert not (isolated_home / "settings.json").exists()


async def test_update_settings_enforces_limits_immediately(isolated_home, rt, settings, factory):
    for key in ("org/a", "org/b", "org/c"):
        async with rt.acquire(key):
            await asyncio.sleep(0.01)
    held = await rt.acquire("org/a")                  # a is busy and most recently used
    await rt.update_settings({"maxLoadedModels": 1})
    assert factory.of("org/b")[0].stopped and factory.of("org/c")[0].stopped
    assert not factory.of("org/a")[0].stopped
    assert instance_keys(rt) == ["org/a"]
    held.release()


async def test_lowering_limits_evicts_busy_instances_once_idle(isolated_home, rt, settings, factory):
    a = await rt.acquire("org/a")
    b = await rt.acquire("org/b")
    await rt.update_settings({"maxLoadedModels": 1})
    assert instance_keys(rt) == ["org/a", "org/b"]    # both busy: nothing evicted yet
    a.release()
    await settle(10)
    assert factory.of("org/a")[0].stopped
    assert instance_keys(rt) == ["org/b"]
    b.release()


async def test_lowering_ram_limit_and_instances_per_model(isolated_home, rt, settings, factory):
    settings.runtime.max_instances_per_model = 2
    one = await rt.acquire("org/a")
    two = await rt.acquire("org/a")
    one.release()
    two.release()
    await rt.update_settings({"maxInstancesPerModel": 1})
    assert len([e for e in factory.of("org/a") if not e.stopped]) == 1
    async with rt.acquire("org/b"):
        pass
    await rt.update_settings({"ramLimitGb": max(1.0, 1.05 * est_bytes() / GiB)})
    assert len(rt.status()["instances"]) == 1


# --------------------------------------------------------------------------- #
# Status, events, stats, estimates
# --------------------------------------------------------------------------- #


async def test_status_dto_shape(rt, factory):
    factory.default = Behavior(memory=321 * MiB)
    async with rt.acquire("org/a") as lease:
        lease.record_stats(GenStats(completion_tokens=50, gen_tps=40.0))
    await settle()
    status = rt.status()
    assert set(status) == {"instances", "system", "budget", "settings"}
    assert set(status["system"]) == {"totalBytes", "usedBytes", "availableBytes"}
    assert set(status["budget"]) == {"limitBytes", "usedBytes"}
    assert set(status["settings"]) == {"maxLoadedModels", "maxInstancesPerModel", "defaultParallel",
                                       "ramLimitGb", "idleTtlMinutes", "defaultCtx", "flashAttn",
                                       "kvCacheType", "gpuLayers", "evictPolicy",
                                       # models kept in memory on purpose
                                       "pinnedModels"}
    inst = status["instances"][0]
    # "sandboxed"/"isolation": whether this process runs in the macOS sandbox and
    # one Polish line about it (localcode.security.sandbox)
    assert set(inst) == {"id", "modelKey", "displayName", "format", "state", "nCtx", "nParallel",
                         "sandboxed", "isolation",
                         "busySlots", "memoryBytes", "estimatedBytes", "loadedAt", "lastUsedAt",
                         "pid", "port", "genTps", "error"}
    assert inst["format"] == "gguf" and inst["displayName"] == "a"
    assert inst["memoryBytes"] == 321 * MiB != inst["estimatedBytes"]
    assert inst["estimatedBytes"] == est_bytes()
    now_ms = time.time() * 1000
    assert abs(inst["loadedAt"] - now_ms) < 60_000 and abs(inst["lastUsedAt"] - now_ms) < 60_000
    assert inst["genTps"] == 40.0 and inst["error"] is None
    assert status["budget"]["limitBytes"] == 20 * GiB
    # measured memory above the estimate is what the budget reserves
    factory.engines[0].behavior.memory = 5 * GiB
    await rt._measure(rt._instances[inst["id"]])
    assert rt.status()["budget"]["usedBytes"] == 5 * GiB
    json.dumps(status)


async def test_events_are_emitted(settings, library, factory):
    sync_events: list[dict] = []
    async_events: list[dict] = []

    async def async_sink(event: dict) -> None:
        async_events.append(event)

    for sink in (sync_events.append, async_sink):
        manager = RuntimeManager(settings, library, on_event=sink, engine_factory=factory)
        try:
            async with manager.acquire("org/a"):
                await settle()
            await settle()
        finally:
            await manager.shutdown()
        await settle()
    for events in (sync_events, async_events):
        assert events and all(e["type"] == "runtime" and "instances" in e for e in events)
        states = [i["state"] for e in events for i in e["instances"]]
        assert "busy" in states and "ready" in states


async def test_broken_event_callback_is_ignored(settings, library, factory):
    def boom(event: dict) -> None:
        raise RuntimeError("UI gone")

    manager = RuntimeManager(settings, library, on_event=boom, engine_factory=factory)
    try:
        async with manager.acquire("org/a"):
            await settle()
    finally:
        await manager.shutdown()


async def test_record_stats_tps_ema(rt):
    async with rt.acquire("org/a") as lease:
        lease.record_stats(GenStats(completion_tokens=100, gen_tps=50.0))
        lease.record_stats(GenStats(completion_tokens=100, gen_tps=100.0))
        lease.record_stats(GenStats(completion_tokens=1, gen_tps=1000.0))   # too short, ignored
        lease.record_stats(None)
    assert rt.status()["instances"][0]["genTps"] == pytest.approx(0.3 * 100 + 0.7 * 50, abs=0.1)


async def test_manager_estimate(rt, settings):
    est = rt.estimate("org/a")
    assert isinstance(est, MemoryEstimate)
    assert est.n_ctx == 4096 and est.total_bytes == est_bytes() and est.fits in ("yes", "tight")
    assert rt.estimate("org/a", LoadConfig(n_ctx=8192, n_parallel=2)).n_parallel == 2
    settings.runtime.max_loaded_models = 1
    async with rt.acquire("org/b"):
        pass
    est = rt.estimate("org/a")
    assert "Zostanie wyładowany: b" in est.message
    with pytest.raises(EngineError):
        rt.estimate("org/missing")


# --------------------------------------------------------------------------- #
# Default engine factory
# --------------------------------------------------------------------------- #


async def test_default_engine_factory_imports_lazily(settings, library, monkeypatch):
    created: list[dict[str, Any]] = []

    class StubLlama(FakeEngine):
        def __init__(self, instance_id, model, config, settings=None):
            super().__init__(instance_id, model, config, Behavior(), [])
            created.append({"settings": settings, "config": config})

    module = types.ModuleType("localcode.runtime.llama_engine")
    module.LlamaServerEngine = StubLlama
    monkeypatch.setitem(sys.modules, "localcode.runtime.llama_engine", module)
    monkeypatch.setitem(sys.modules, "localcode.runtime.mlx_engine", None)   # import fails
    manager = RuntimeManager(settings, library)
    try:
        async with manager.acquire("org/a") as lease:
            assert isinstance(lease.engine, StubLlama)
        assert created and created[0]["settings"] is settings
        with pytest.raises(EngineError, match="MLX jest niedostępny"):
            await manager.acquire("org/m")
    finally:
        await manager.shutdown()


# --------------------------------------------------------------------------- #
# Concurrency: unloading, cancelled waiters, orphaned waiters, idle timer
# --------------------------------------------------------------------------- #


async def test_unload_is_idempotent_for_an_instance_already_unloaded(rt, monkeypatch):
    """A second "Wyładuj" (stale row in the UI, retried request) must not fail.

    The Mac app and ``/unload <id>`` in the terminal both work from a status
    snapshot that can be one event behind, so the id they send may already be
    gone — that is a no-op, not ``409 engine_error``.
    """
    monkeypatch.setattr(mgr, "UNLOAD_GRACE_S", 0.05)
    lease = await rt.acquire("org/a")
    instance_id = lease.instance_id
    await rt.unload(instance_id)                  # busy: stopped after the grace period
    lease.release()
    await rt.unload(instance_id)                  # gone by now: still fine
    await rt.unload(instance_id)
    assert rt.status()["instances"] == []
    with pytest.raises(EngineError, match="Nie ma załadowanej instancji"):
        await rt.unload("nigdy-nie-istniala-7")   # a typo is still an error


async def test_concurrent_unloads_of_a_busy_instance_all_succeed(rt, factory, monkeypatch):
    monkeypatch.setattr(mgr, "UNLOAD_GRACE_S", 0.05)
    lease = await rt.acquire("org/a")
    results = await asyncio.gather(rt.unload(lease.instance_id), rt.unload(lease.instance_id),
                                   return_exceptions=True)
    assert results == [None, None]
    assert factory.engines[0].stopped
    lease.release()


async def test_cancelled_waiter_leaves_the_queue(rt):
    held = await rt.acquire("org/a")
    waiting = asyncio.create_task(rt.acquire("org/a"))
    await settle()
    assert sum(len(q) for q in rt._waiters.values()) == 1
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    await settle()
    assert sum(len(q) for q in rt._waiters.values()) == 0
    held.release()
    assert rt.status()["instances"][0]["busySlots"] == 0


async def test_eviction_without_a_replacement_fails_waiters(rt):
    """An instance stopped by the reaper/limits must not strand its waiters."""
    lease = await rt.acquire("org/a")
    waiting = asyncio.create_task(rt.acquire("org/a"))
    await settle()
    async with rt._lock:
        rt._begin_stop(rt._instances[lease.instance_id], grace=0.0, reason="limits")
    lease.release()
    with pytest.raises(EngineError, match="wyładowany"):
        await asyncio.wait_for(waiting, 1)


async def test_explicit_load_restarts_the_idle_timer(rt):
    info = await rt.load("org/a")
    inst = rt._instances[info["id"]]
    inst.last_used -= 1000.0                      # pretend it has been idle for a while
    stale = inst.last_used
    again = await rt.load("org/a")
    assert again["id"] == info["id"] and inst.last_used > stale


# --------------------------------------------------------------------------- #
# Models that cannot generate text
# --------------------------------------------------------------------------- #


async def test_acquiring_an_embedding_model_for_text_is_refused_in_polish(rt) -> None:
    """Regression: `localcode -m <model embeddingowy>` printed a raw engine error.

    llama-server answered `HTTP 500 … the current context does not logits
    computation`, twice, with nothing telling the user what was wrong.
    """
    with pytest.raises(EngineError) as excinfo:
        async with rt.acquire("org/e", purpose="agent"):
            pass
    assert "embedding" in str(excinfo.value).lower() or "osadze" in str(excinfo.value).lower()
    assert "500" not in str(excinfo.value)


async def test_acquiring_an_image_model_for_text_is_refused_in_polish(rt) -> None:
    with pytest.raises(EngineError) as excinfo:
        async with rt.acquire("org/img", purpose="chat"):
            pass
    assert "obraz" in str(excinfo.value).lower()


async def test_an_embedding_model_is_still_usable_for_embeddings(rt) -> None:
    async with rt.acquire("org/e", purpose="embedding") as lease:
        assert lease.engine is not None


# --------------------------------------------------------------------------- #
# Keeping a model in memory on purpose (runtime.pinned_models)
# --------------------------------------------------------------------------- #


def test_a_pinned_model_is_named_in_the_status(settings, library, factory) -> None:
    settings.runtime.pinned_models = ["org/a"]
    mgr = RuntimeManager(settings, library, engine_factory=factory)
    assert mgr.status()["settings"]["pinnedModels"] == ["org/a"]


def test_pinned_models_are_validated_as_a_list_of_keys() -> None:
    from localcode.runtime.manager import validate_settings_patch

    assert validate_settings_patch({"pinnedModels": ["a/b", " a/b ", ""]}) == {"pinned_models": ["a/b"]}
    assert validate_settings_patch({"pinnedModels": None}) == {"pinned_models": []}
    with pytest.raises(SettingsError):
        validate_settings_patch({"pinnedModels": [1, 2]})


async def test_a_pinned_model_is_not_unloaded_when_it_goes_idle(settings, library, factory) -> None:
    """The point of pinning: it answers without a load, so it must stay loaded."""
    settings.runtime.idle_ttl_minutes = 1
    settings.runtime.pinned_models = ["org/a"]
    mgr = RuntimeManager(settings, library, engine_factory=factory)
    try:
        await mgr.load("org/a", wait=True)
        await mgr.load("org/b", wait=True)
        for inst in mgr._instances.values():          # pretend both went idle long ago
            inst.last_used -= 3600

        await mgr._reap_once()
        await settle()

        alive = {i.model.key for i in mgr._instances.values()}
        assert "org/a" in alive
        assert "org/b" not in alive
    finally:
        await mgr.shutdown()


async def test_a_pinned_model_is_never_chosen_as_an_eviction_victim(settings, library, factory) -> None:
    settings.runtime.pinned_models = ["org/a"]
    mgr = RuntimeManager(settings, library, engine_factory=factory)
    try:
        await mgr.load("org/a", wait=True)
        inst = next(iter(mgr._instances.values()))
        assert mgr._is_pinned(inst) is True

        victims, fits = mgr._eviction_plan(10 ** 12, exclude=set(), allow_evict=True)

        assert [v.model.key for v in victims] == []
        assert fits is False          # ...so a model that big simply does not fit
    finally:
        await mgr.shutdown()
