"""Auto mode over HTTP: the mode, the reviewer setting and the verdict on the tool card."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient

from server_fakes import ToolCallEvent, ToolResultEvent
from test_agent_ws import collect, make_session, recv, use_agent
from localcode.events import AutoReview
from localcode.server.core import AGENT_MODES


async def test_auto_is_an_agent_mode(client: TestClient, tmp_path: Path) -> None:
    assert "auto" in AGENT_MODES
    session = await make_session(client, tmp_path, mode="auto")
    assert session["mode"] == "auto"
    response = await client.patch("/api/app-settings", json={"agent": {"permissionMode": "auto"}})
    assert response.status == 200
    assert (await response.json())["agent"]["permissionMode"] == "auto"


async def test_reviewer_model_setting_round_trips(client: TestClient, core) -> None:
    body = await (await client.get("/api/app-settings")).json()
    assert body["agent"]["autoReviewerModel"] is None

    response = await client.patch("/api/app-settings",
                                  json={"agent": {"autoReviewerModel": "remote:anthropic:claude-x"}})
    assert response.status == 200
    assert (await response.json())["agent"]["autoReviewerModel"] == "remote:anthropic:claude-x"
    assert core.settings.agent.auto_reviewer_model == "remote:anthropic:claude-x"

    back = await client.patch("/api/app-settings", json={"agent": {"autoReviewerModel": None}})
    assert (await back.json())["agent"]["autoReviewerModel"] is None
    bad = await client.patch("/api/app-settings", json={"agent": {"autoReviewerModel": 7}})
    assert bad.status == 400


async def test_the_verdict_lands_on_the_tool_item(client: TestClient, core, tmp_path: Path) -> None:
    async def behaviour(agent: Any, text: str) -> None:
        agent.emit(ToolCallEvent(call_id="c1", name="bash", arguments={"command": "npm test"},
                                 title="Uruchom: npm test"))
        agent.emit(AutoReview(call_id="c1", tool="bash", verdict="deny",
                              reason="Nie prosiłeś o testy.", reviewer_model="test/model"))
        agent.emit(ToolResultEvent(call_id="c1", name="bash", ok=False,
                                   summary="Odrzucone przez recenzenta", output="…"))

    use_agent(core, behaviour)
    session = await make_session(client, tmp_path, mode="auto")
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "zrób coś"})
        review = await recv(ws, "auto_review")
        assert review == {"type": "auto_review", "callId": "c1", "tool": "bash", "verdict": "deny",
                          "reason": "Nie prosiłeś o testy.", "reviewerModel": "test/model"}
        await recv(ws, "turn_end")
    finally:
        await ws.close()

    detail = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
    tool = next(i for i in detail["items"] if i["kind"] == "tool")
    assert tool["review"] == {"verdict": "deny", "reason": "Nie prosiłeś o testy.", "model": "test/model"}


# --------------------------------------------------------------------------- #
# The local guard's setting, and what travels to the app
# --------------------------------------------------------------------------- #


async def test_guard_model_setting_is_local_only(client: TestClient, core) -> None:
    from server_fakes import CHAT_KEY

    body = await (await client.get("/api/app-settings")).json()
    assert body["agent"]["autoGuardModel"] is None

    ok = await client.patch("/api/app-settings", json={"agent": {"autoGuardModel": CHAT_KEY}})
    assert ok.status == 200 and (await ok.json())["agent"]["autoGuardModel"] == CHAT_KEY
    assert core.settings.agent.auto_guard_model == CHAT_KEY

    remote = await client.patch("/api/app-settings",
                                json={"agent": {"autoGuardModel": "remote:deepinfra:big"}})
    assert remote.status == 400
    assert "lokalnym" in (await remote.json())["error"]
    assert core.settings.agent.auto_guard_model == CHAT_KEY, "a refused value changes nothing"

    back = await client.patch("/api/app-settings", json={"agent": {"autoGuardModel": None}})
    assert (await back.json())["agent"]["autoGuardModel"] is None


async def test_the_question_carries_both_opinions(client: TestClient, core, tmp_path: Path) -> None:
    from localcode.events import PermissionRequest

    review = {"verdict": "deny", "reason": "Co robi: Usuwa build. Powód: Nikt nie prosił.",
              "model": "remote:deepinfra:big",
              "guard": {"verdict": "deny", "reason": "Niebezpieczne.", "model": "local/tiny"}}
    answers: list[str] = []

    async def behaviour(agent: Any, text: str) -> None:
        agent.emit(ToolCallEvent(call_id="c1", name="bash", arguments={"command": "rm -r build"},
                                 title="Polecenie: rm -r build"))
        agent.emit(AutoReview(call_id="c1", tool="bash", verdict="deny", reason=review["reason"],
                              reviewer_model=review["model"], guard=review["guard"]))
        request = PermissionRequest(request_id="r1", tool="bash", title="Polecenie: rm -r build",
                                    preview="rm -r build", review=review)
        agent.emit(request)
        answers.append(await agent.session.ask_permission(request))
        agent.emit(ToolResultEvent(call_id="c1", name="bash", ok=False, summary="Odrzucone",
                                   output="…"))

    use_agent(core, behaviour)
    session = await make_session(client, tmp_path, mode="auto")
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "posprzątaj"})
        event = await recv(ws, "auto_review")
        assert event["guard"] == review["guard"]
        request = await recv(ws, "permission_request")
        assert request["review"] == review
        await ws.send_json({"type": "permission", "requestId": "r1", "decision": "deny"})
        await recv(ws, "turn_end")
    finally:
        await ws.close()
    assert answers == ["deny"]

    detail = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
    tool = next(i for i in detail["items"] if i["kind"] == "tool")
    assert tool["review"]["guard"] == review["guard"]


# --------------------------------------------------------------------------- #
# Effort "ultracode" is the agent's; the chat maps it to "high"
# --------------------------------------------------------------------------- #


async def test_a_session_accepts_ultracode(client: TestClient, core, tmp_path: Path) -> None:
    use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "setEffort", "effort": "ultracode"})
        update = await recv(ws, "session", where=lambda d: d["session"]["effort"] == "ultracode")
        assert update["session"]["effort"] == "ultracode"
    finally:
        await ws.close()


async def test_settings_take_ultracode_for_the_agent_only(client: TestClient, core) -> None:
    agent = await client.patch("/api/app-settings", json={"agent": {"reasoningEffort": "ultracode"}})
    assert agent.status == 200 and (await agent.json())["agent"]["reasoningEffort"] == "ultracode"
    chat = await client.patch("/api/app-settings", json={"chat": {"reasoningEffort": "ultracode"}})
    assert chat.status == 200 and (await chat.json())["chat"]["reasoningEffort"] == "high"
    bad = await client.patch("/api/app-settings", json={"agent": {"reasoningEffort": "turbo"}})
    assert bad.status == 400
    parallel = await client.patch("/api/app-settings", json={"agent": {"ultracodeParallel": 3}})
    assert (await parallel.json())["agent"]["ultracodeParallel"] == 3
