"""Adversarial checks of the local HTTP server: Host, Origin and the token gate.

The server listens on 127.0.0.1 only, so the realistic attacker is a web page
open in the user's browser. ``origin_middleware`` stops it as long as the
browser sends an ``Origin`` — which stops being true once the page reaches the
server through a DNS name that resolves to 127.0.0.1 (DNS rebinding): the page
is then same-origin with the server and its GETs carry no ``Origin`` at all.
``host_middleware`` is what closes that hole, so it gets the tests below.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient

from conftest import TOKEN

REBIND_HOST = "rebind.attacker.example"


@pytest.mark.parametrize("path", ["/api/health", "/v1/models"])
async def test_dns_rebinding_host_is_refused(anon_client: TestClient, path: str) -> None:
    """A DNS name pointing at 127.0.0.1 must not reach the unauthenticated surface."""
    response = await anon_client.get(path, headers={"Host": REBIND_HOST})
    assert response.status == 403
    body = await response.json()
    code = body["error"]["code"] if path.startswith("/v1") else body["code"]
    assert code == "host_not_allowed"


async def test_rebinding_host_is_refused_before_auth(anon_client: TestClient) -> None:
    response = await anon_client.get("/api/library", headers={"Host": REBIND_HOST,
                                                              "Authorization": f"Bearer {TOKEN}"})
    assert response.status == 403


@pytest.mark.parametrize("host", ["127.0.0.1:8421", "localhost:8421", "[::1]:8421",
                                  "192.168.1.14:8421", "localhost"])
async def test_normal_hosts_still_work(anon_client: TestClient, host: str) -> None:
    response = await anon_client.get("/api/health", headers={"Host": host})
    assert response.status == 200


async def test_configured_host_name_is_accepted(core, anon_client: TestClient) -> None:
    core.settings.server.host = "localcode.local"
    response = await anon_client.get("/api/health", headers={"Host": "localcode.local:8421"})
    assert response.status == 200


async def test_origin_null_is_refused(anon_client: TestClient) -> None:
    """A sandboxed iframe / file:// page sends ``Origin: null``."""
    response = await anon_client.get("/api/health", headers={"Origin": "null"})
    assert response.status == 403
    assert (await response.json())["code"] == "origin_not_allowed"


async def test_browser_origin_refused_on_openai_surface(anon_client: TestClient) -> None:
    response = await anon_client.post("/v1/chat/completions", json={"model": "x", "messages": []},
                                      headers={"Origin": "http://evil.example"})
    assert response.status == 403


async def test_options_without_origin_is_no_bypass(anon_client: TestClient) -> None:
    """``OPTIONS`` skips the token check, so it must not reach a handler."""
    response = await anon_client.options("/api/library")
    assert response.status == 405


async def test_agent_ws_requires_token(anon_client: TestClient) -> None:
    from aiohttp import WSServerHandshakeError

    with pytest.raises(WSServerHandshakeError) as excinfo:
        await anon_client.ws_connect("/api/agent/sessions/whatever/ws")
    assert excinfo.value.status == 401


async def test_events_sse_requires_token(anon_client: TestClient) -> None:
    response = await anon_client.get("/api/events")
    assert response.status == 401


async def test_wrong_token_of_same_length_is_refused(anon_client: TestClient) -> None:
    """``hmac.compare_digest`` accepts only the exact token, prefixes included."""
    for bad in (TOKEN[:-1] + "x", TOKEN[:-1], TOKEN + "x", TOKEN.upper()):
        response = await anon_client.get("/api/library", headers={"Authorization": f"Bearer {bad}"})
        assert response.status == 401, bad


async def test_error_bodies_never_echo_the_token(anon_client: TestClient) -> None:
    response = await anon_client.get(f"/api/library?token={TOKEN}",
                                     headers={"Authorization": "Bearer nope"})
    assert response.status == 401
    assert TOKEN not in await response.text()
