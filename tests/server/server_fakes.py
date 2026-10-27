"""Fakes for the server tests.

The engine/runtime fakes from ``tests/chat/chat_fakes.py`` are reused (the
server conftest puts ``tests/chat`` on ``sys.path``) and extended with the
parts of :class:`~localcode.runtime.manager.RuntimeManager` that the HTTP layer
needs (``status``, ``load``, ``unload``, ``estimate`` …), plus stand-ins for the
Hugging Face client, the download manager and the coding agent.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Awaitable

from chat_fakes import FakeEngine, make_model  # noqa: F401  (re-exported for the tests)

from localcode.events import (
    Compaction,
    ContextUpdate,
    PermissionRequest,
    Status,
    TextDelta,
    TodoUpdate,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
    TurnStart,
)
from localcode.hub.types import LocalModel, MemoryEstimate
from localcode.runtime.base import EngineError

CHAT_KEY = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
CHAT_KEY_2 = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
EMBED_KEY = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0"


# --------------------------------------------------------------------------- #
# Library / runtime
# --------------------------------------------------------------------------- #


class FakeLibrary:
    """``ModelLibrary`` stand-in with the methods the server calls."""

    def __init__(self, models: list[LocalModel]) -> None:
        self.models = list(models)
        self.scans = 0
        self.deleted: list[str] = []
        self.defaults: dict[str, dict[str, Any]] = {}

    def list(self) -> list[LocalModel]:
        return list(self.models)

    def get(self, key: str) -> LocalModel | None:
        return next((m for m in self.models if m.key == key), None)

    def resolve(self, query: str) -> LocalModel | None:
        q = (query or "").lower()
        exact = self.get(query)
        if exact is not None:
            return exact
        hits = [m for m in self.models if q in m.key.lower() or q in m.display_name.lower()]
        return hits[0] if len(hits) == 1 else None

    def scan(self) -> list[LocalModel]:
        self.scans += 1
        return list(self.models)

    def load_defaults(self, key: str) -> dict[str, Any] | None:
        return self.defaults.get(key)

    def set_load_defaults(self, key: str, defaults: dict[str, Any] | None) -> dict[str, Any] | None:
        if self.get(key) is None:
            raise LookupError(key)
        if defaults:
            self.defaults[key] = dict(defaults)
        else:
            self.defaults.pop(key, None)
        return self.defaults.get(key)

    def delete(self, key: str) -> None:
        model = self.get(key)
        if model is None:
            raise LookupError(key)
        self.models = [m for m in self.models if m.key != key]
        self.deleted.append(key)

    async def adelete(self, key: str) -> None:
        self.delete(key)


class FakeLease:
    """Lease with the surface ``openai_api`` and ``ChatService`` use."""

    def __init__(self, runtime: "FakeRuntime", engine: FakeEngine, model_key: str) -> None:
        self.runtime = runtime
        self.engine = engine
        self.model_key = model_key
        self.released = False
        self.stats: list[Any] = []

    @property
    def n_ctx(self) -> int:
        return self.engine.n_ctx

    def record_stats(self, stats: Any) -> None:
        self.stats.append(stats)

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.runtime.active -= 1
            self.runtime.released += 1

    async def aclose(self) -> None:
        self.release()

    async def __aenter__(self) -> "FakeLease":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()


class FakeRuntime:
    """``RuntimeManager`` stand-in: leases, instance list and settings."""

    def __init__(self, engines: dict[str, FakeEngine], *, on_event: Callable[[dict], Any] | None = None,
                 acquire_error: BaseException | None = None, ram_limit_gb: float | None = 8.0) -> None:
        self.engines = engines
        self.on_event = on_event
        self.acquire_error = acquire_error
        self.active = 0
        self.released = 0
        self.acquired: list[tuple[str, str]] = []
        self.instances: list[dict[str, Any]] = []
        self.unloaded: list[str] = []
        self.settings_patches: list[dict[str, Any]] = []
        self.shutdowns = 0
        self.ram_limit_gb = ram_limit_gb
        # routes_remote.py reads core.runtime.remote. With no key configured a
        # real registry never touches the network, so the HTTP layer can be
        # tested against the real thing instead of another fake.
        self.remote: Any = None

    # ---- leases
    async def acquire(self, model_key: str, *, n_ctx: int | None = None, purpose: str = "chat") -> FakeLease:
        if self.acquire_error is not None:
            raise self.acquire_error
        if model_key not in self.engines:
            raise EngineError(f"Nie znaleziono modelu „{model_key}”.")
        self.acquired.append((model_key, purpose))
        self.active += 1
        return FakeLease(self, self.engines[model_key], model_key)

    # ---- status
    def status(self) -> dict[str, Any]:
        return {
            "instances": list(self.instances),
            "system": {"totalBytes": 34_359_738_368, "usedBytes": 8_589_934_592,
                       "availableBytes": 25_769_803_776},
            "budget": {"limitBytes": 24_000_000_000,
                       "usedBytes": sum(i.get("memoryBytes", 0) for i in self.instances)},
            "settings": {"maxLoadedModels": 2, "maxInstancesPerModel": 1, "defaultParallel": 1,
                         "ramLimitGb": self.ram_limit_gb, "idleTtlMinutes": 30, "defaultCtx": None,
                         "flashAttn": "auto", "kvCacheType": "f16", "gpuLayers": -1,
                         "evictPolicy": "lru"},
        }

    def is_loaded(self, model_key: str) -> bool:
        return any(i["modelKey"] == model_key for i in self.instances)

    def loaded_keys(self) -> set[str]:
        return {i["modelKey"] for i in self.instances}

    def _emit(self) -> None:
        if self.on_event is not None:
            self.on_event({"type": "runtime", **self.status()})

    async def load(self, model_key: str, config: Any = None, *, new_instance: bool = False,
                   wait: bool = True) -> dict[str, Any]:
        if model_key not in self.engines:
            raise EngineError(f"Nie znaleziono modelu „{model_key}”.")
        info = {
            "id": f"inst-{len(self.instances) + 1}", "modelKey": model_key,
            "displayName": model_key.split("/")[-1], "format": "gguf",
            "state": "loading" if not wait else "ready",
            "nCtx": getattr(config, "n_ctx", 0) or 8192,
            "nParallel": getattr(config, "n_parallel", 1), "busySlots": 0,
            "memoryBytes": 700_000_000, "estimatedBytes": 700_000_000,
            "loadedAt": 1, "lastUsedAt": 1, "pid": 4242, "port": 51234,
            "genTps": None, "error": None,
        }
        self.instances.append(info)
        self._emit()
        return info

    async def unload(self, instance_id: str) -> None:
        if not any(i["id"] == instance_id for i in self.instances):
            raise EngineError(f"Nie ma załadowanej instancji „{instance_id}”.")
        self.instances = [i for i in self.instances if i["id"] != instance_id]
        self.unloaded.append(instance_id)
        self._emit()

    async def unload_all(self) -> None:
        self.unloaded.extend(i["id"] for i in self.instances)
        self.instances.clear()
        self._emit()

    def estimate(self, model_key: str, config: Any = None) -> MemoryEstimate:
        if model_key not in self.engines:
            raise EngineError(f"Nie znaleziono modelu „{model_key}”.")
        n_ctx = getattr(config, "n_ctx", 0) or 8192
        return MemoryEstimate(weights_bytes=500_000_000, kv_cache_bytes=100_000_000,
                              overhead_bytes=430_000_000, total_bytes=1_030_000_000,
                              n_ctx=n_ctx, n_parallel=getattr(config, "n_parallel", 1),
                              fits="yes", message="Model zmieści się w pamięci.")

    async def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        if "ramLimitGb" in patch and not isinstance(patch["ramLimitGb"], (int, float, type(None))):
            raise ValueError("Nieprawidłowa wartość „ramLimitGb”: podaj liczbę.")
        self.settings_patches.append(dict(patch))
        if "ramLimitGb" in patch:
            self.ram_limit_gb = patch["ramLimitGb"]
        self._emit()
        return self.status()

    async def shutdown(self) -> None:
        self.shutdowns += 1
        self.instances.clear()


# --------------------------------------------------------------------------- #
# Hub / downloads
# --------------------------------------------------------------------------- #


def hub_model_dto(repo: str, fmt: str = "gguf") -> dict[str, Any]:
    author, name = repo.split("/", 1)
    return {"id": repo, "author": author, "name": name, "format": fmt, "downloads": 1234, "likes": 7,
            "updatedAt": 1_700_000_000_000, "tags": ["text-generation"], "pipelineTag": "text-generation",
            "paramsB": 0.6, "gated": False, "curated": False, "description": "Testowy model"}


class FakeHub:
    """``HubClient`` stand-in (no network)."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.searches: list[tuple[str, str, str, int]] = []

    async def asearch(self, query: str = "", format: str = "any", sort: str = "trending",
                      limit: int = 30) -> list[dict[str, Any]]:
        if self.error is not None:
            raise self.error
        self.searches.append((query, format, sort, limit))
        return [hub_model_dto("unsloth/Qwen3-0.6B-GGUF")]

    def search(self, query: str = "", format: str = "any", sort: str = "trending",
               limit: int = 30) -> list[dict[str, Any]]:
        """Synchronous search (used by ``hub.catalog.Catalog``)."""
        if self.error is not None:
            raise self.error
        self.searches.append((query, format, sort, limit))
        return [hub_model_dto("unsloth/Qwen3-0.6B-GGUF")]

    async def acurated(self) -> list[dict[str, Any]]:
        return [hub_model_dto("unsloth/Qwen3-4B-Instruct-2507-GGUF")]

    async def adetails(self, repo: str, *, refresh: bool = False) -> dict[str, Any]:
        return self.details(repo)

    def details(self, repo: str, *, refresh: bool = False) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {
            "model": hub_model_dto(repo),
            "readme": "# Test", "contextLength": 32768, "arch": "qwen3",
            "capabilities": ["tools"], "isEmbedding": False,
            "variants": [
                {"id": "Qwen3-0.6B-Q4_K_M", "label": "Q4_K_M", "quant": "Q4_K_M",
                 "files": ["Qwen3-0.6B-Q4_K_M.gguf"], "sizeBytes": 500_000_000,
                 "isVisionProjector": False, "recommended": True, "estimate": {}},
                {"id": "Qwen3-0.6B-Q8_0", "label": "Q8_0", "quant": "Q8_0",
                 "files": ["Qwen3-0.6B-Q8_0.gguf"], "sizeBytes": 900_000_000,
                 "isVisionProjector": False, "recommended": False, "estimate": {}},
            ],
            "files": [{"name": "Qwen3-0.6B-Q4_K_M.gguf", "size": 500_000_000}],
            "accessError": None, "sha": "abc",
        }

    def variant(self, repo: str, variant_id: str) -> dict[str, Any]:
        for v in self.details(repo)["variants"]:
            if v["id"] == variant_id:
                return v
        raise LookupError(variant_id)


class FakeDownloads:
    """``DownloadManager`` stand-in that reports through ``on_event``."""

    def __init__(self, settings: Any = None, on_event: Callable[[dict], Any] | None = None) -> None:
        self.settings = settings
        self.on_event = on_event
        self.jobs: list[dict[str, Any]] = []
        self.shutdowns = 0

    def start(self, repo: str, files: list[str]) -> dict[str, Any]:
        job = {"id": f"job-{len(self.jobs) + 1}", "repo": repo, "files": list(files),
               "label": files[0] if files else repo, "format": "gguf",
               "totalBytes": 500_000_000, "doneBytes": 0, "speedBps": 0, "state": "queued",
               "error": None, "startedAt": 1, "finishedAt": None, "modelKey": None}
        self.jobs.append(job)
        self.emit(job)
        return job

    def emit(self, job: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event({"type": "download", "job": dict(job)})

    def finish(self, job_id: str) -> dict[str, Any]:
        job = next(j for j in self.jobs if j["id"] == job_id)
        job.update(state="done", doneBytes=job["totalBytes"], finishedAt=2)
        self.emit(job)
        return job

    def cancel(self, job_id: str) -> dict[str, Any] | bool:
        job = next((j for j in self.jobs if j["id"] == job_id), None)
        if job is None:
            return False
        job["state"] = "cancelled"
        self.emit(job)
        return job

    def list(self) -> list[dict[str, Any]]:
        return [dict(j) for j in self.jobs]

    async def shutdown(self) -> None:
        self.shutdowns += 1


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

Behaviour = Callable[["ScriptedAgent", str], Awaitable[None]]


class ScriptedAgent:
    """Coding-agent stand-in driven by a coroutine ``behaviour(agent, text)``."""

    def __init__(self, session: Any, behaviour: Behaviour | None = None) -> None:
        self.session = session
        self.behaviour = behaviour
        self.turns: list[str] = []
        self.interrupts = 0
        self.compactions = 0
        self.clears = 0
        self.modes: list[str] = []
        self.models: list[str] = []
        self.started = asyncio.Event()
        #: set by :meth:`interrupt` — behaviours that block wait on it
        self.stop = asyncio.Event()

    # ---- events
    def emit(self, event: Any) -> None:
        self.session.on_event(event)

    async def run_turn(self, text: str, images: list[dict[str, Any]] | None = None) -> None:
        self.turns.append(text)
        self.images = list(images or [])
        self.started.set()
        turn_id = uuid.uuid4().hex[:8]
        self.emit(TurnStart(turn_id=turn_id, user_text=text))
        try:
            if self.behaviour is not None:
                await self.behaviour(self, text)
            else:
                self.emit(ContextUpdate(used=120, limit=8192))
                self.emit(TextDelta(text=f"Echo: {text}"))
        finally:
            self.emit(TurnEnd(turn_id=turn_id, reason="done", stats={"steps": 1}))

    async def ask(self, tool: str = "bash", title: str = "Uruchom: ls", preview: str = "ls -la") -> str:
        request = PermissionRequest(request_id=uuid.uuid4().hex[:8], tool=tool, title=title, preview=preview)
        self.emit(request)
        return await self.session.ask_permission(request)

    def interrupt(self) -> None:
        self.interrupts += 1
        self.stop.set()

    def set_mode(self, mode: str) -> None:
        self.modes.append(mode)

    def set_model(self, model_key: str) -> None:
        self.models.append(model_key)

    async def compact(self) -> None:
        self.compactions += 1
        self.emit(Compaction(before=4000, after=1200, strategy="summarize"))

    async def clear(self) -> None:
        self.clears += 1

    # ---- /goal (the real loop is tested in tests/agent/test_goal.py)
    goal: dict[str, Any] | None = None

    async def set_goal(self, text: str) -> dict[str, Any]:
        from localcode.events import GoalUpdate

        self.goal = {"text": text, "state": "active", "turns": 0, "maxTurns": 20, "summary": ""}
        self.emit(GoalUpdate(text=text, state="active", turns=0, max_turns=20))
        return self.goal

    async def clear_goal(self) -> None:
        from localcode.events import GoalUpdate

        if self.goal is not None:
            self.emit(GoalUpdate(text=self.goal["text"], state="cleared", max_turns=20))
        self.goal = None


__all__ = [
    "CHAT_KEY", "CHAT_KEY_2", "EMBED_KEY",
    "Compaction", "ContextUpdate", "FakeDownloads", "FakeEngine", "FakeHub", "FakeLease",
    "FakeLibrary", "FakeRuntime", "PermissionRequest", "ScriptedAgent", "Status", "TextDelta",
    "TodoUpdate", "ToolCallEvent", "ToolResultEvent", "TurnEnd", "TurnStart",
    "hub_model_dto", "make_model",
]
