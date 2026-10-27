"""Runtime endpoints and the ``/api/events`` stream."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp.test_utils import TestClient

from localcode.runtime.base import EngineError
from server_fakes import CHAT_KEY


async def next_record(response: Any, timeout: float = 5.0) -> str:
    """Next non-empty line of an SSE response (``data: …`` or a ``: ping`` comment)."""
    async def read() -> str:
        async for raw in response.content:
            line = raw.decode("utf-8").rstrip("\n")
            if line.strip():
                return line
        return ""
    return await asyncio.wait_for(read(), timeout)


async def next_event(response: Any, timeout: float = 5.0) -> dict[str, Any]:
    while True:
        line = await next_record(response, timeout)
        if line.startswith("data: "):
            return json.loads(line[6:])


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


async def test_runtime_status(client: TestClient) -> None:
    body = await (await client.get("/api/runtime")).json()
    assert set(body) == {"instances", "system", "budget", "settings"}
    assert body["instances"] == []
    assert set(body["system"]) == {"totalBytes", "usedBytes", "availableBytes"}
    assert set(body["settings"]) >= {"maxLoadedModels", "defaultParallel", "kvCacheType", "evictPolicy"}


async def test_runtime_settings_patch(client: TestClient, runtime) -> None:
    response = await client.patch("/api/runtime/settings", json={"ramLimitGb": 12})
    assert response.status == 200
    assert runtime.settings_patches == [{"ramLimitGb": 12}]
    assert (await response.json())["settings"]["ramLimitGb"] == 12


async def test_runtime_settings_invalid(client: TestClient) -> None:
    response = await client.patch("/api/runtime/settings", json={"ramLimitGb": "dużo"})
    assert response.status == 400
    assert (await response.json())["error"].startswith("Nieprawidłowa wartość")


async def test_runtime_load_and_unload(client: TestClient, runtime) -> None:
    loaded = await client.post("/api/runtime/load",
                               json={"modelKey": CHAT_KEY, "config": {"nCtx": 4096, "nParallel": 2}})
    assert loaded.status == 200
    instance = await loaded.json()
    assert instance["modelKey"] == CHAT_KEY
    assert instance["nCtx"] == 4096 and instance["nParallel"] == 2
    assert instance["state"] in ("loading", "ready")

    status = await (await client.post("/api/runtime/unload",
                                      json={"instanceId": instance["id"]})).json()
    assert status["instances"] == []
    assert runtime.unloaded == [instance["id"]]


async def test_runtime_unload_all(client: TestClient, runtime) -> None:
    await runtime.load(CHAT_KEY)
    response = await client.post("/api/runtime/unload", json={"all": True})
    assert response.status == 200
    assert (await response.json())["instances"] == []


async def test_runtime_load_resolves_substring(client: TestClient) -> None:
    response = await client.post("/api/runtime/load", json={"modelKey": "Qwen3-0.6B-Q4_K_M"})
    assert response.status == 200
    assert (await response.json())["modelKey"] == CHAT_KEY


async def test_runtime_load_unknown_model(client: TestClient) -> None:
    response = await client.post("/api/runtime/load", json={"modelKey": "nie/ma"})
    assert response.status == 409
    body = await response.json()
    assert body["code"] == "engine_error"
    assert "nie/ma" in body["error"]


async def test_runtime_load_requires_key(client: TestClient) -> None:
    response = await client.post("/api/runtime/load", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_model_key"


async def test_runtime_estimate(client: TestClient) -> None:
    response = await client.post("/api/runtime/estimate",
                                 json={"modelKey": CHAT_KEY, "config": {"nCtx": 16384}})
    assert response.status == 200
    body = await response.json()
    assert set(body) == {"weightsBytes", "kvCacheBytes", "overheadBytes", "totalBytes",
                         "nCtx", "nParallel", "fits", "message"}
    assert body["nCtx"] == 16384 and body["fits"] == "yes"


async def test_runtime_unload_missing_instance(client: TestClient) -> None:
    response = await client.post("/api/runtime/unload", json={"instanceId": "brak"})
    assert response.status == 409
    assert (await response.json())["code"] == "engine_error"


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


async def test_events_opens_with_runtime_snapshot(client: TestClient) -> None:
    response = await client.get("/api/events")
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    try:
        first = await next_event(response)
        assert first["type"] == "runtime"
        assert "instances" in first and "budget" in first
    finally:
        response.close()


async def test_events_forwards_runtime_download_and_library(client: TestClient, core, runtime) -> None:
    response = await client.get("/api/events")
    try:
        assert (await next_event(response))["type"] == "runtime"

        await runtime.load(CHAT_KEY)
        event = await next_event(response)
        assert event["type"] == "runtime"
        assert event["instances"][0]["modelKey"] == CHAT_KEY

        job = core.downloads_fake.start("unsloth/Qwen3-0.6B-GGUF", ["a.gguf"])
        event = await next_event(response)
        assert event["type"] == "download" and event["job"]["id"] == job["id"]

        core.broadcast_library()
        assert (await next_event(response))["type"] == "library"
    finally:
        response.close()


async def test_events_heartbeat(client: TestClient, monkeypatch) -> None:
    import localcode.server.routes_runtime as routes_runtime

    monkeypatch.setattr(routes_runtime, "EVENTS_HEARTBEAT_S", 0.05)
    response = await client.get("/api/events")
    try:
        assert (await next_event(response))["type"] == "runtime"
        line = await next_record(response, timeout=2.0)
        assert line.startswith(":")
    finally:
        response.close()


async def test_events_unsubscribes_on_disconnect(client: TestClient, core) -> None:
    response = await client.get("/api/events")
    await next_event(response)
    assert core.events.subscribers == 1
    response.close()
    for _ in range(50):
        await asyncio.sleep(0.05)
        if core.events.subscribers == 0:
            break
    assert core.events.subscribers == 0


async def test_acquire_failure_maps_to_409(client: TestClient, runtime) -> None:
    runtime.acquire_error = EngineError("Model nie mieści się w pamięci.")
    response = await client.post("/api/chat", json={"content": "Hej", "model": CHAT_KEY})
    # /api/chat reports failures inside the stream, not as an HTTP error
    assert response.status == 200
    frames = []
    async for raw in response.content:
        line = raw.decode().strip()
        if line.startswith("data: "):
            frames.append(json.loads(line[6:]))
    assert frames[-1]["type"] == "error"
    assert "pamięci" in frames[-1]["message"]
