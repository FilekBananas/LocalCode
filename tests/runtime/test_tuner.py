"""Tests for runtime/tuner.py: the deterministic recommender, the optional model
layer on top of it, and the HTTP route.

No real models, no network: the model layer is exercised with the same
``FakeEngine`` the chat tests use (scripted text, no process spawned).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_CHAT_TESTS = Path(__file__).resolve().parent.parent / "chat"
if str(_CHAT_TESTS) not in sys.path:
    sys.path.insert(0, str(_CHAT_TESTS))

from chat_fakes import FakeEngine, make_model  # noqa: E402

from localcode.config import Settings  # noqa: E402
from localcode.hub.types import LocalModel  # noqa: E402
from localcode.runtime import memory as mem  # noqa: E402
from localcode.runtime import tuner  # noqa: E402
from localcode.runtime.manager import validate_settings_patch  # noqa: E402
from localcode.server import routes_runtime  # noqa: E402
from localcode.server.app import (  # noqa: E402
    CORE, TOKEN, auth_middleware, error_middleware, host_middleware, origin_middleware,
    size_middleware,
)

GiB = mem.GiB

#: All ten tunable knobs the recommender covers, snake_case.
ALL_KEYS = {"ram_limit_gb", "max_loaded_models", "max_instances_per_model", "default_parallel",
           "idle_ttl_minutes", "default_ctx", "kv_cache_type", "flash_attn", "gpu_layers",
           "evict_policy"}

#: (total GB, performance cores, efficiency cores, cpu count) — plausible Apple
#: Silicon shapes at each memory size, used across the parametrized tests.
MACHINES: list[tuple[int, int, int, int]] = [
    (8, 4, 4, 8),
    (16, 4, 4, 8),
    (32, 8, 4, 12),
    (64, 10, 4, 14),
    (128, 12, 4, 16),
]


def facts_for(total_gb: int, perf: int, eff: int, cpu: int, *, apple_silicon: bool = True) -> tuner.MachineFacts:
    return tuner.MachineFacts(total_ram_bytes=total_gb * GiB, is_apple_silicon=apple_silicon,
                              performance_cores=perf, efficiency_cores=eff, cpu_count=cpu)


@pytest.fixture
def big_model() -> LocalModel:
    return make_model("org/Qwen3-4B-Instruct", name="Qwen3 4B Instruct", ctx=262144)


# --------------------------------------------------------------------------- #
# Small helper
# --------------------------------------------------------------------------- #


def test_polish_plural_helper():
    assert tuner._pl(1, "model", "modele", "modeli") == "model"
    assert tuner._pl(2, "model", "modele", "modeli") == "modele"
    assert tuner._pl(4, "model", "modele", "modeli") == "modele"
    assert tuner._pl(5, "model", "modele", "modeli") == "modeli"
    assert tuner._pl(12, "model", "modele", "modeli") == "modeli"      # 12-14 exception
    assert tuner._pl(22, "model", "modele", "modeli") == "modele"


# --------------------------------------------------------------------------- #
# Deterministic recommender
# --------------------------------------------------------------------------- #


def test_recommend_is_pure():
    facts = facts_for(*MACHINES[2])
    proposal_a = tuner.recommend(facts, None)
    proposal_b = tuner.recommend(facts, None)
    assert proposal_a == proposal_b


def test_recommend_no_model_covers_all_ten_knobs():
    facts = facts_for(32, 8, 4, 12)
    proposal = tuner.recommend(facts)
    keys = {r.key for r in proposal.recommendations}
    assert keys == ALL_KEYS
    assert proposal.model_key is None
    assert proposal.source == "deterministic"
    # every recommendation carries a non-empty Polish reason and a basis
    # ("the number it is derived from").
    for rec in proposal.recommendations:
        assert rec.reason.strip()
        assert rec.basis.strip()
        assert any(ch.isdigit() for ch in rec.basis), rec


@pytest.mark.parametrize("total_gb,perf,eff,cpu", MACHINES)
def test_recommend_sane_on_every_machine_size(total_gb, perf, eff, cpu, monkeypatch):
    facts = facts_for(total_gb, perf, eff, cpu)
    proposal = tuner.recommend(facts, None)
    patch = proposal.as_patch()
    assert set(patch) == ALL_KEYS

    assert 0 < patch["ram_limit_gb"] < total_gb
    assert 1 <= patch["max_loaded_models"]
    assert 1 <= patch["max_instances_per_model"] <= 2
    assert 1 <= patch["default_parallel"] <= 4
    assert patch["kv_cache_type"] in ("f16", "q8_0")
    assert patch["flash_attn"] == "auto"
    assert patch["gpu_layers"] == -1
    assert patch["evict_policy"] == "lru"
    assert patch["default_ctx"] >= mem.MIN_USEFUL_CTX

    # The proposal must validate against the real RAM-limit bound too: make
    # the physical machine match the fake one so the check is meaningful.
    monkeypatch.setattr(mem, "system_memory",
                        lambda: {"total": total_gb * GiB, "available": total_gb * GiB // 2,
                                 "used": total_gb * GiB // 2})
    validated = validate_settings_patch(patch)
    assert validated == patch


def test_recommend_ram_tiers_are_monotonic_in_total_ram():
    proposals = [tuner.recommend(facts_for(*m), None) for m in MACHINES]
    loaded = [p.as_patch()["max_loaded_models"] for p in proposals]
    ttl = [p.as_patch()["idle_ttl_minutes"] for p in proposals]
    ram = [p.as_patch()["ram_limit_gb"] for p in proposals]
    assert loaded == sorted(loaded)
    assert ttl == sorted(ttl)
    assert ram == sorted(ram)


def test_recommend_with_model_reuses_memory_estimator(big_model: LocalModel):
    facts = facts_for(32, 8, 4, 12)
    proposal = tuner.recommend(facts, big_model)
    patch = proposal.as_patch()

    # default_ctx must be exactly what runtime/memory.py's own admission math
    # would compute for the same per-instance budget — no second estimate.
    per_instance_budget = int(facts.total_ram_bytes * 0.70) // max(1, patch["max_loaded_models"])
    expected_fit = mem.max_ctx_that_fits(big_model, per_instance_budget, n_parallel=patch["default_parallel"],
                                         kv_cache_type=patch["kv_cache_type"], flash_attn="auto",
                                         embedding=big_model.is_embedding)
    native = mem.native_context(big_model)
    expected = max(min(mem.MIN_USEFUL_CTX, native), min(expected_fit, native))
    assert patch["default_ctx"] == expected
    assert proposal.model_key == big_model.key


def test_recommend_large_model_narrows_max_loaded_models():
    facts = facts_for(32, 8, 4, 12)
    huge = dataclasses.replace(make_model("org/Huge-70B", name="Huge 70B"), size_bytes=20 * GiB)
    baseline = tuner.recommend(facts, None).as_patch()["max_loaded_models"]
    with_model = tuner.recommend(facts, huge).as_patch()["max_loaded_models"]
    assert with_model <= baseline
    assert with_model == 1


def test_recommend_gpu_layers_cites_apple_silicon(big_model: LocalModel):
    facts_as = facts_for(32, 8, 4, 12, apple_silicon=True)
    facts_intel = facts_for(32, 8, 4, 12, apple_silicon=False)
    proposal_as = tuner.recommend(facts_as, None)
    proposal_intel = tuner.recommend(facts_intel, None)
    assert proposal_as.note is None
    assert proposal_intel.note is not None and "Apple Silicon" in proposal_intel.note
    gpu_as = next(r for r in proposal_as.recommendations if r.key == "gpu_layers")
    gpu_intel = next(r for r in proposal_intel.recommendations if r.key == "gpu_layers")
    assert gpu_as.value == gpu_intel.value == -1
    assert gpu_as.reason != gpu_intel.reason


def test_recommendation_and_proposal_to_dict_shape(big_model: LocalModel):
    facts = facts_for(32, 8, 4, 12)
    proposal = tuner.recommend(facts, big_model)
    dto = proposal.to_dict()
    assert set(dto) == {"machine", "modelKey", "source", "note", "recommendations", "patch"}
    assert dto["modelKey"] == big_model.key
    assert len(dto["recommendations"]) == 10
    for rec in dto["recommendations"]:
        assert set(rec) == {"key", "value", "reason", "basis"}
    assert set(dto["patch"]) == {tuner._CAMEL[k] for k in ALL_KEYS}


# --------------------------------------------------------------------------- #
# Model layer: refine_with_model
# --------------------------------------------------------------------------- #


class _FakeLease:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.released = False

    def release(self) -> None:
        self.released = True


class _FakeRuntime:
    """Minimal ``RuntimeLike``: only what ``refine_with_model`` needs."""

    def __init__(self, engines: dict[str, Any] | None = None) -> None:
        self.engines = engines or {}
        self.acquire_calls: list[str] = []
        self.leases: list[_FakeLease] = []

    def loaded_keys(self) -> set[str]:
        return set(self.engines)

    async def acquire(self, model_key: str, *, n_ctx: int | None = None, purpose: str = "chat") -> _FakeLease:
        self.acquire_calls.append(model_key)
        lease = _FakeLease(self.engines[model_key])
        self.leases.append(lease)
        return lease


@pytest.fixture
def base_proposal(big_model: LocalModel) -> tuner.TuneProposal:
    facts = facts_for(32, 8, 4, 12)
    return tuner.recommend(facts, big_model)


async def test_refine_with_model_not_loaded_falls_back(base_proposal: tuner.TuneProposal):
    runtime = _FakeRuntime()  # nothing loaded
    result = await tuner.refine_with_model(runtime, "org/Qwen3-4B-Instruct", base_proposal)
    assert result.source == "deterministic"
    assert result.recommendations == base_proposal.recommendations
    assert "nie jest teraz załadowany" in (result.note or "")
    assert runtime.acquire_calls == []  # never triggers a load


async def test_refine_with_model_clamps_out_of_range_answer(base_proposal: tuner.TuneProposal):
    key = "org/Qwen3-4B-Instruct"
    # ramLimitGb: valid adjustment; gpuLayers: out of range (max is 999);
    # evictPolicy: not a valid choice; unknownField: not one of the ten knobs.
    answer = ('{"ramLimitGb": 10, "gpuLayers": 999999, "evictPolicy": "sometimes", '
             '"unknownField": true}')
    engine = FakeEngine(script=[answer])
    runtime = _FakeRuntime({key: engine})

    result = await tuner.refine_with_model(runtime, key, base_proposal)

    by_key = {r.key: r for r in result.recommendations}
    assert by_key["ram_limit_gb"].value == 10.0
    assert "Model doprecyzował" in by_key["ram_limit_gb"].reason
    # rejected fields keep the deterministic value, unchanged
    original = {r.key: r for r in base_proposal.recommendations}
    assert by_key["gpu_layers"].value == original["gpu_layers"].value == -1
    assert by_key["evict_policy"].value == original["evict_policy"].value == "lru"
    assert result.source == "model"
    assert "ramLimitGb" in (result.note or "")
    assert "gpuLayers" in (result.note or "") and "evictPolicy" in (result.note or "")
    assert runtime.leases[0].released is True


async def test_refine_with_model_all_rejected_stays_deterministic(base_proposal: tuner.TuneProposal):
    key = "org/Qwen3-4B-Instruct"
    engine = FakeEngine(script=['{"gpuLayers": -999, "ramLimitGb": "dużo"}'])
    runtime = _FakeRuntime({key: engine})
    result = await tuner.refine_with_model(runtime, key, base_proposal)
    assert result.recommendations == base_proposal.recommendations
    # nothing actually changed, so the source stays "deterministic"
    assert result.source == "deterministic"
    assert "Odrzucono" in (result.note or "")


async def test_refine_with_model_empty_object_means_no_change(base_proposal: tuner.TuneProposal):
    key = "org/Qwen3-4B-Instruct"
    engine = FakeEngine(script=["{}"])
    runtime = _FakeRuntime({key: engine})
    result = await tuner.refine_with_model(runtime, key, base_proposal)
    assert result.recommendations == base_proposal.recommendations
    assert result.source == "deterministic"
    assert "nie zaproponował" in (result.note or "")


async def test_refine_with_model_non_json_answer_falls_back(base_proposal: tuner.TuneProposal):
    key = "org/Qwen3-4B-Instruct"
    engine = FakeEngine(script=["Sure, I think your settings look great overall!"])
    runtime = _FakeRuntime({key: engine})
    result = await tuner.refine_with_model(runtime, key, base_proposal)
    assert result.recommendations == base_proposal.recommendations
    assert result.source == "deterministic"
    assert result.note  # some explanation was recorded


async def test_refine_with_model_engine_failure_falls_back(base_proposal: tuner.TuneProposal):
    key = "org/Qwen3-4B-Instruct"
    engine = FakeEngine(fail=RuntimeError("boom"))
    runtime = _FakeRuntime({key: engine})
    result = await tuner.refine_with_model(runtime, key, base_proposal)
    assert result.recommendations == base_proposal.recommendations
    assert "Nie udało się zapytać modelu" in (result.note or "")


async def test_refine_with_model_accepts_snake_case_answer(base_proposal: tuner.TuneProposal):
    """The model may answer with either spelling; snake_case works too."""
    key = "org/Qwen3-4B-Instruct"
    engine = FakeEngine(script=['{"kv_cache_type": "q8_0"}'])
    runtime = _FakeRuntime({key: engine})
    result = await tuner.refine_with_model(runtime, key, base_proposal)
    kv = next(r for r in result.recommendations if r.key == "kv_cache_type")
    assert kv.value == "q8_0"


# --------------------------------------------------------------------------- #
# HTTP route: POST /api/runtime/tune
# --------------------------------------------------------------------------- #


class _RouteLibrary:
    def __init__(self, models: list[LocalModel]) -> None:
        self.models = {m.key: m for m in models}

    def get(self, key: str) -> LocalModel | None:
        return self.models.get(key)

    def resolve(self, query: str) -> LocalModel | None:
        return None


class _RouteRuntime(_FakeRuntime):
    """Adds ``update_settings``/``status`` on top of the model-layer fake."""

    def __init__(self, engines: dict[str, Any] | None = None) -> None:
        super().__init__(engines)
        self.patches: list[dict[str, Any]] = []

    async def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        self.patches.append(dict(patch))
        return {"settings": patch}


class _RouteCore:
    def __init__(self, models: list[LocalModel], engines: dict[str, Any] | None = None) -> None:
        self.settings = Settings()
        self.library = _RouteLibrary(models)
        self.runtime = _RouteRuntime(engines)

    def resolve_model(self, query: str | None) -> LocalModel | None:
        if not query:
            return None
        return self.library.get(query)


def _make_app(core: _RouteCore) -> web.Application:
    app = web.Application(middlewares=[error_middleware, host_middleware, origin_middleware,
                                       size_middleware, auth_middleware])
    app[CORE] = core
    app[TOKEN] = ""
    routes_runtime.register(app)
    return app


@pytest.fixture
async def route_client(big_model: LocalModel):
    core = _RouteCore([big_model])
    app = _make_app(core)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client, core
    finally:
        await client.close()


async def test_route_tune_deterministic_only(route_client):
    client, core = route_client
    response = await client.post("/api/runtime/tune", json={"modelKey": "org/Qwen3-4B-Instruct",
                                                             "useModel": False})
    assert response.status == 200
    body = await response.json()
    rec = body["recommendation"]
    assert rec["source"] == "deterministic"
    assert rec["modelKey"] == "org/Qwen3-4B-Instruct"
    assert len(rec["recommendations"]) == 10
    assert "status" not in body
    assert core.runtime.acquire_calls == []


async def test_route_tune_without_model_key_is_generic(route_client):
    client, core = route_client
    response = await client.post("/api/runtime/tune", json={})
    assert response.status == 200
    body = await response.json()
    assert body["recommendation"]["modelKey"] is None


async def test_route_tune_unknown_model_key_falls_back_to_generic(route_client):
    client, core = route_client
    response = await client.post("/api/runtime/tune", json={"modelKey": "nie/ma/takiego"})
    assert response.status == 200
    body = await response.json()
    assert body["recommendation"]["modelKey"] is None


async def test_route_tune_apply_writes_through_update_settings(route_client):
    client, core = route_client
    response = await client.post("/api/runtime/tune",
                                 json={"modelKey": "org/Qwen3-4B-Instruct", "useModel": False, "apply": True})
    assert response.status == 200
    body = await response.json()
    assert "status" in body
    assert len(core.runtime.patches) == 1
    applied = core.runtime.patches[0]
    assert set(applied) == {tuner._CAMEL[k] for k in ALL_KEYS}
    assert applied == body["recommendation"]["patch"]


async def test_route_tune_invalid_json_body(route_client):
    client, core = route_client
    response = await client.post("/api/runtime/tune", data="not json",
                                 headers={"Content-Type": "application/json"})
    assert response.status == 400


async def test_route_tune_uses_loaded_model_by_default(route_client, big_model: LocalModel):
    client, core = route_client
    core.runtime.engines[big_model.key] = FakeEngine(script=['{"ramLimitGb": 5}'])
    response = await client.post("/api/runtime/tune", json={"modelKey": big_model.key})
    assert response.status == 200
    body = await response.json()
    assert body["recommendation"]["source"] == "model"
    assert core.runtime.acquire_calls == [big_model.key]


async def test_route_tune_apply_with_nonsense_model_answer_writes_only_deterministic_values(
    route_client, big_model: LocalModel,
):
    """A model that answers with garbage must never reach ``update_settings``.

    Every field below is individually invalid (wrong type, out of range, not
    an allowed choice); ``refine_with_model``/``validate_settings_patch`` must
    drop all of them, so ``apply: true`` writes exactly the deterministic
    proposal — the same patch a plain, model-free tune would have applied.
    """
    client, core = route_client
    nonsense = ('{"ramLimitGb": "duzo bardzo", "maxLoadedModels": -50, '
               '"maxInstancesPerModel": 99999, "defaultParallel": "not-a-number", '
               '"idleTtlMinutes": -5, "defaultCtx": 999999999, "flashAttn": "maybe", '
               '"kvCacheType": "banana", "gpuLayers": "all-of-them", "evictPolicy": "yolo", '
               '"unknownField": "ignored"}')
    core.runtime.engines[big_model.key] = FakeEngine(script=[nonsense])

    expected = tuner.recommend(tuner.gather_machine_facts(), big_model).patch

    response = await client.post("/api/runtime/tune",
                                 json={"modelKey": big_model.key, "apply": True})
    assert response.status == 200
    body = await response.json()
    assert core.runtime.acquire_calls == [big_model.key]           # the model WAS consulted
    assert body["recommendation"]["source"] == "deterministic"      # but changed nothing
    assert len(core.runtime.patches) == 1
    applied = core.runtime.patches[0]
    assert applied == expected == body["recommendation"]["patch"]  # only vetted values written
