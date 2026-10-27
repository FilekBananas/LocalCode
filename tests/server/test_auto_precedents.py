"""Precedents over HTTP: the setting, and listing / clearing what was learned."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from aiohttp.test_utils import TestClient

from localcode.agent import precedents
from localcode.agent.precedents import APPROVED, DECLINED, Key
from localcode.events import PERMISSION_DISMISSED

from server_fakes import ScriptedAgent


async def test_the_setting_is_on_by_default_and_can_be_switched_off(client: TestClient, core) -> None:
    body = await (await client.get("/api/app-settings")).json()
    assert body["agent"]["autoPrecedentsEnabled"] is True

    off = await client.patch("/api/app-settings", json={"agent": {"autoPrecedentsEnabled": False}})
    assert off.status == 200 and (await off.json())["agent"]["autoPrecedentsEnabled"] is False
    assert core.settings.agent.auto_precedents_enabled is False

    bad = await client.patch("/api/app-settings", json={"agent": {"autoPrecedentsEnabled": "tak"}})
    assert bad.status == 400


async def test_list_and_clear(client: TestClient, isolated_home: Path, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    precedents.record(a, [Key("cmd", "git push")], APPROVED, now=1)
    precedents.record(a, [Key("cmd", "git push")], APPROVED, now=2)
    precedents.record(b, [Key("cmd", "rm")], DECLINED, now=3)

    one = await (await client.get("/api/agent/auto-precedents", params={"cwd": str(a)})).json()
    assert one["enabled"] is True
    assert one["projects"] == [{"cwd": os.path.realpath(a), "precedents": [
        {"decision": "approved", "kind": "cmd", "key": "git push", "count": 2, "lastUsed": 2}]}]
    every = await (await client.get("/api/agent/auto-precedents")).json()
    assert [p["cwd"] for p in every["projects"]] == [os.path.realpath(b), os.path.realpath(a)]

    cleared = await client.delete("/api/agent/auto-precedents", params={"cwd": str(a)})
    assert cleared.status == 200 and (await cleared.json()) == {"removed": 1}
    assert precedents.load(a) == [] and len(precedents.load(b)) == 1

    everything = await client.delete("/api/agent/auto-precedents")
    assert (await everything.json()) == {"removed": 1}
    assert (await (await client.get("/api/agent/auto-precedents")).json())["projects"] == []
    assert (isolated_home / precedents.STORE_NAME).exists(), "in LOCALCODE_HOME, not the project"


async def test_the_routes_need_the_token(anon_client: TestClient) -> None:
    assert (await anon_client.get("/api/agent/auto-precedents")).status == 401
    assert (await anon_client.delete("/api/agent/auto-precedents")).status == 401


async def test_a_client_may_dismiss_a_question(client: TestClient, core, tmp_path: Path) -> None:
    """A script that cannot ask answers "dismissed": refused, but not learned from."""
    answers: list[str] = []

    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        answers.append(await agent.ask())

    core.agents.agent_factory = lambda session: ScriptedAgent(session, behaviour)
    created = await client.post("/api/agent/sessions", json={"cwd": str(tmp_path)})
    session = await created.json()
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await ws.send_json({"type": "user", "text": "zrób coś"})
        request = None
        for _ in range(50):
            message = await asyncio.wait_for(ws.receive_json(), 5)
            if message.get("type") == "permission_request":
                request = message
                break
        assert request is not None
        await ws.send_json({"type": "permission", "requestId": request["requestId"],
                            "decision": PERMISSION_DISMISSED})
        for _ in range(50):
            if answers:
                break
            await asyncio.sleep(0.01)
    finally:
        await ws.close()
    assert answers == [PERMISSION_DISMISSED]
