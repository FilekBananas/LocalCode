"""Cross-cutting HTTP rules: authentication, origins, error bodies, body size."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from conftest import TOKEN
from localcode.chat.api import MAX_REQUEST_BYTES
from localcode.server.app import create_app

PROTECTED = [
    ("GET", "/api/bootstrap"),
    ("GET", "/api/conversations"),
    ("GET", "/api/library"),
    ("GET", "/api/runtime"),
    ("GET", "/api/events"),
    ("GET", "/api/app-settings"),
    ("GET", "/api/agent/sessions"),
    ("GET", "/api/hub/curated"),
]


async def test_health_needs_no_token(anon_client: TestClient) -> None:
    response = await anon_client.get("/api/health")
    assert response.status == 200
    body = await response.json()
    assert body["ok"] is True and isinstance(body["version"], str)


@pytest.mark.parametrize("method,path", PROTECTED)
async def test_api_requires_token(anon_client: TestClient, method: str, path: str) -> None:
    response = await anon_client.request(method, path)
    assert response.status == 401
    body = await response.json()
    assert body["code"] == "unauthorized"
    assert body["error"].startswith("Brak autoryzacji")


async def test_wrong_token_is_rejected(anon_client: TestClient) -> None:
    response = await anon_client.get("/api/library", headers={"Authorization": "Bearer nope"})
    assert response.status == 401


async def test_token_in_query_only_for_streams(anon_client: TestClient) -> None:
    ok = await anon_client.get(f"/api/events?token={TOKEN}")
    assert ok.status == 200
    ok.close()
    denied = await anon_client.get(f"/api/library?token={TOKEN}")
    assert denied.status == 401


async def test_authorised_request_passes(client: TestClient) -> None:
    assert (await client.get("/api/library")).status == 200


async def test_unknown_route_is_json_404(client: TestClient) -> None:
    response = await client.get("/api/nie-ma-takiego")
    assert response.status == 404
    body = await response.json()
    assert body == {"error": "Nie znaleziono zasobu.", "code": "not_found"}


async def test_method_not_allowed_is_json(client: TestClient) -> None:
    response = await client.put("/api/library")
    assert response.status == 405
    assert (await response.json())["code"] == "method_not_allowed"


async def test_invalid_json_body(client: TestClient) -> None:
    response = await client.post("/api/conversations", data="{nope",
                                 headers={"Content-Type": "application/json"})
    assert response.status == 400
    assert (await response.json())["code"] == "invalid_json"


async def test_browser_origin_is_rejected(client: TestClient) -> None:
    response = await client.get("/api/library", headers={"Origin": "http://evil.example"})
    assert response.status == 403
    body = await response.json()
    assert body["code"] == "origin_not_allowed"


async def test_allowlisted_origin_gets_cors_headers(core, client: TestClient) -> None:
    core.settings.server.cors_origins = ["http://localhost:5173"]
    response = await client.get("/api/library", headers={"Origin": "http://localhost:5173"})
    assert response.status == 200
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
    assert "Authorization" in response.headers["Access-Control-Allow-Headers"]


async def test_preflight_without_token(core, anon_client: TestClient) -> None:
    core.settings.server.cors_origins = ["http://localhost:5173"]
    response = await anon_client.options("/api/library", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    })
    assert response.status == 204
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"


async def test_oversized_body_is_refused(client: TestClient) -> None:
    response = await client.post("/api/conversations", data=b"x",
                                 headers={"Content-Length": str(MAX_REQUEST_BYTES + 1),
                                          "Content-Type": "application/json"})
    assert response.status == 413
    assert (await response.json())["code"] == "payload_too_large"


async def test_empty_token_disables_auth(core) -> None:
    client = TestClient(TestServer(create_app(core, "")))
    await client.start_server()
    try:
        assert (await client.get("/api/library")).status == 200
    finally:
        await client.close()


async def test_error_body_is_polish_and_json(client: TestClient) -> None:
    response = await client.get("/api/conversations/nie-ma")
    assert response.status == 404
    body = json.loads(await response.text())
    assert set(body) >= {"error", "code"}
    assert body["error"] == "Nie znaleziono rozmowy."


async def test_a_silent_stream_sends_keep_alive_pings(monkeypatch) -> None:
    """Without them the Mac app hung up after 60 s of silence and cancelled the job.

    Loading a video model before its first progress frame is longer than that,
    so the app said "Serwer nie odpowiedział na czas" to a server that was busy.
    """
    import asyncio

    from aiohttp import web

    from localcode.server import app as app_mod

    monkeypatch.setattr(app_mod, "SSE_HEARTBEAT_S", 0.05)

    async def slow():
        await asyncio.sleep(0.3)           # "loading the model"
        yield {"type": "done"}

    async def handler(request):
        return await app_mod.stream_sse(request, slow())

    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    app.router.add_get("/s", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        body = await (await client.get("/s")).text()
    finally:
        await client.close()
    assert ": ping" in body
    assert body.index(": ping") < body.index('"done"')
