"""Fakes for the terminal-app tests.

:class:`ScriptedBackend` implements the :class:`~localcode.tui.backends.Backend`
contract with a coroutine instead of a model, so the headless runner, the
renderer and the slash commands can be tested without loading anything.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from localcode.config import Settings
from localcode.tui.backends import Backend

Script = Callable[["ScriptedBackend", str], Awaitable[None]]


def make_model_dto(key: str = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M", **over: Any) -> dict[str, Any]:
    dto = {
        "key": key,
        "format": "gguf",
        "path": f"/models/{key}.gguf",
        "displayName": key.split("/")[-1],
        "sizeBytes": 420_000_000,
        "repo": "/".join(key.split("/")[:2]),
        "quant": "Q4_K_M",
        "source": "localcode",
        "capabilities": ["tools", "thinking"],
        "arch": "qwen3",
        "paramsB": 0.6,
        "contextLength": 32768,
        "modifiedAt": 0.0,
        "hasVision": False,
        "loaded": False,
        "loadDefaults": None,
    }
    dto.update(over)
    return dto


async def echo_script(backend: "ScriptedBackend", text: str) -> None:
    """The default script: one tool call, a short answer, statistics."""
    backend.emit({"type": "turn_start", "turnId": "t1", "userText": text})
    backend.emit({"type": "context", "used": 512, "limit": 8192, "system": 300, "tools": 120,
                  "history": 92, "reserve": 1024})
    backend.emit({"type": "tool_call", "callId": "c1", "name": "read_file",
                  "arguments": {"path": "src/app.py"}, "title": "Odczyt src/app.py (1–200)"})
    backend.emit({"type": "tool_result", "callId": "c1", "name": "read_file", "ok": True,
                  "summary": "200 linii", "output": "print('hi')", "diff": None})
    backend.emit({"type": "text_delta", "text": "Cześć! "})
    backend.emit({"type": "text_delta", "text": "Zrobione."})
    backend.emit({"type": "turn_end", "turnId": "t1", "reason": "done",
                  "stats": {"steps": 1, "genTps": 12.5, "completionTokens": 42,
                            "promptTokens": 512, "elapsedMs": 1800}})


async def failing_script(backend: "ScriptedBackend", text: str) -> None:
    backend.emit({"type": "turn_start", "turnId": "t2", "userText": text})
    backend.emit({"type": "error", "message": "Model nie odpowiada."})
    backend.emit({"type": "turn_end", "turnId": "t2", "reason": "error", "stats": {}})


async def permission_script(backend: "ScriptedBackend", text: str) -> None:
    backend.emit({"type": "turn_start", "turnId": "t3", "userText": text})
    request = {"type": "permission_request", "requestId": "r1", "tool": "bash",
               "title": "Uruchom: rm -rf build", "preview": "rm -rf build",
               "options": ["allow", "deny", "always"]}
    backend.emit(request)
    decision = await backend.wait_permission("r1")
    backend.decisions.append(decision)
    backend.emit({"type": "tool_result", "callId": "c9", "name": "bash", "ok": decision == "allow",
                  "summary": "odmówiono" if decision != "allow" else "exit 0", "output": ""})
    backend.emit({"type": "turn_end", "turnId": "t3", "reason": "done", "stats": {"steps": 1}})


class ScriptedBackend(Backend):
    """A backend whose "model" is a coroutine over :mod:`localcode.events` dicts."""

    kind = "local"

    def __init__(self, cwd: str | Path = ".", *, script: Script | None = None,
                 models: Sequence[dict[str, Any]] | None = None,
                 settings: Settings | None = None,
                 model_key: str | None = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M") -> None:
        super().__init__(cwd, settings or Settings())
        self.script: Script = script or echo_script
        self.model_key = model_key
        self.models = list(models) if models is not None else [make_model_dto()]
        self.sent: list[str] = []
        self.decisions: list[str] = []
        self.modes: list[str] = []
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self.patches: list[dict[str, Any]] = []
        self.closed = False
        self.compactions = 0
        self.clears = 0
        self.interrupts = 0
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._task: asyncio.Task[Any] | None = None

    # ---------------------------------------------------------------- events
    def emit(self, event: dict[str, Any]) -> None:
        self._publish(event)

    async def wait_permission(self, request_id: str) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        return await future

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()

    # ----------------------------------------------------------------- agent
    async def send_user(self, text: str) -> None:
        self.sent.append(text)
        self.busy = True
        self._task = asyncio.create_task(self.script(self, text))

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self.emit({"type": "turn_end", "turnId": "x", "reason": "interrupted", "stats": {}})

    async def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.modes.append(mode)

    async def set_model(self, key: str) -> None:
        resolved = await self.resolve_model_key(key)
        self.model_key = resolved or key

    async def compact(self) -> None:
        self.compactions += 1
        self.emit({"type": "compaction", "before": 12400, "after": 5100, "strategy": "summarize"})

    async def clear(self) -> None:
        self.clears += 1
        self.context = None

    async def answer_permission(self, request_id: str, decision: str) -> None:
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(decision)

    # --------------------------------------------------------------- library
    async def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        self._models = list(self.models)
        return self._models

    async def runtime_status(self) -> dict[str, Any]:
        return {
            "instances": [
                {"id": "inst-1", "modelKey": self.model_key, "displayName": "Qwen3 0.6B",
                 "format": "gguf", "state": "ready", "nCtx": 8192, "nParallel": 1,
                 "busySlots": 0, "memoryBytes": 700_000_000, "estimatedBytes": 700_000_000,
                 "loadedAt": 1, "lastUsedAt": 2, "pid": 1, "port": 2, "genTps": 12.5,
                 "error": None},
            ],
            "system": {"totalBytes": 34_359_738_368, "usedBytes": 8_589_934_592,
                       "availableBytes": 25_769_803_776},
            "budget": {"limitBytes": 24_000_000_000, "usedBytes": 700_000_000},
            "settings": {"maxLoadedModels": 2, "maxInstancesPerModel": 1, "defaultParallel": 1,
                         "ramLimitGb": None, "idleTtlMinutes": 30, "defaultCtx": None,
                         "flashAttn": "auto", "kvCacheType": "f16", "gpuLayers": -1,
                         "evictPolicy": "lru"},
        }

    async def load_model(self, key: str, *, n_ctx: int | None = None,
                         n_parallel: int | None = None, new_instance: bool = False) -> dict[str, Any]:
        self.loaded.append(key)
        return {"id": "inst-1", "displayName": key.split("/")[-1], "nCtx": n_ctx or 8192,
                "nParallel": n_parallel or 1, "memoryBytes": 700_000_000}

    async def unload(self, instance_id: str | None = None, *, everything: bool = False) -> None:
        self.unloaded.append("all" if everything or not instance_id else instance_id)

    async def update_runtime_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        self.patches.append(dict(patch))
        return await self.runtime_status()

    async def estimate(self, key: str, *, n_ctx: int | None = None) -> dict[str, Any]:
        return {"weightsBytes": 420_000_000, "kvCacheBytes": 100_000_000,
                "overheadBytes": 50_000_000, "totalBytes": 570_000_000,
                "nCtx": n_ctx or 8192, "nParallel": 1, "fits": "yes", "message": "Zmieści się."}

    async def delete_model(self, key: str) -> None:
        self.models = [m for m in self.models if m.get("key") != key]

    # -------------------------------------------------------------------- hub
    async def hub_search(self, query: str, *, fmt: str = "any", limit: int = 20) -> list[dict[str, Any]]:
        return [{"id": f"org/{query}-GGUF", "author": "org", "name": f"{query}-GGUF",
                 "format": "gguf", "downloads": 1234, "likes": 7, "paramsB": 4.0,
                 "description": "Testowy model"}]

    async def hub_curated(self) -> list[dict[str, Any]]:
        return [{"id": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "format": "gguf", "paramsB": 4.0,
                 "minRamGb": 8, "recommendedQuant": "Q4_K_M", "category": "code",
                 "description": "Mały i szybki"}]

    async def hub_details(self, repo: str) -> dict[str, Any]:
        return {
            "model": {"id": repo, "format": "gguf", "paramsB": 4.0},
            "variants": [
                {"id": "q4", "label": "Q4_K_M", "quant": "Q4_K_M", "files": [f"{repo}-Q4.gguf"],
                 "sizeBytes": 2_500_000_000, "recommended": True,
                 "estimate": {"totalBytes": 3_000_000_000, "fits": "yes"}},
                {"id": "q8", "label": "Q8_0", "quant": "Q8_0", "files": [f"{repo}-Q8.gguf"],
                 "sizeBytes": 4_500_000_000, "recommended": False,
                 "estimate": {"totalBytes": 5_000_000_000, "fits": "tight"}},
            ],
        }

    async def start_download(self, repo: str, *, variant_id: str | None = None,
                             files: Sequence[str] | None = None) -> dict[str, Any]:
        return {"id": "job-1", "repo": repo, "label": repo, "state": "done",
                "totalBytes": 2_500_000_000, "doneBytes": 2_500_000_000,
                "modelKey": f"{repo}/{variant_id or 'q4'}"}

    async def downloads(self) -> list[dict[str, Any]]:
        return []

    async def cancel_download(self, job_id: str) -> None:
        return None
