"""Agent sessions: REST, the WebSocket protocol and multi-client behaviour."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from aiohttp.test_utils import TestClient

from localcode.events import PERMISSION_DISMISSED

from conftest import TOKEN
from server_fakes import CHAT_KEY, ScriptedAgent, TextDelta, ToolCallEvent, ToolResultEvent


def use_agent(core: Any, behaviour: Callable[..., Any] | None) -> list[ScriptedAgent]:
    """Install a scripted agent factory; returns the list it appends to."""
    created: list[ScriptedAgent] = []

    def factory(session: Any) -> ScriptedAgent:
        agent = ScriptedAgent(session, behaviour)
        created.append(agent)
        return agent

    core.agents.agent_factory = factory
    return created


async def make_session(client: TestClient, cwd: Path, **body: Any) -> dict[str, Any]:
    response = await client.post("/api/agent/sessions", json={"cwd": str(cwd), **body})
    assert response.status == 201, await response.text()
    return await response.json()


async def recv(ws: Any, kind: str, timeout: float = 5.0,
               where: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    """Read WebSocket messages until one of type ``kind`` (and matching ``where``) arrives."""
    async def loop() -> dict[str, Any]:
        while True:
            data = await ws.receive_json()
            if data.get("type") == kind and (where is None or where(data)):
                return data
    return await asyncio.wait_for(loop(), timeout)


async def collect(ws: Any, count: int, timeout: float = 5.0) -> list[dict[str, Any]]:
    async def loop() -> list[dict[str, Any]]:
        return [await ws.receive_json() for _ in range(count)]
    return await asyncio.wait_for(loop(), timeout)


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


async def test_session_crud(client: TestClient, tmp_path: Path) -> None:
    session = await make_session(client, tmp_path, modelKey=CHAT_KEY, mode="plan")
    assert set(session) == {"id", "cwd", "modelKey", "mode", "title", "createdAt", "updatedAt",
                            "contextUsed", "contextLimit", "busy", "effort", "effortLevels", "goal"}
    assert session["cwd"] == str(tmp_path.resolve())
    assert session["modelKey"] == CHAT_KEY and session["mode"] == "plan"
    assert session["busy"] is False

    listed = await (await client.get("/api/agent/sessions")).json()
    assert [s["id"] for s in listed] == [session["id"]]
    filtered = await (await client.get(f"/api/agent/sessions?cwd={tmp_path}")).json()
    assert len(filtered) == 1
    assert await (await client.get("/api/agent/sessions?cwd=/")).json() == []

    detail = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
    assert detail["session"]["id"] == session["id"] and detail["items"] == []

    assert (await client.delete(f"/api/agent/sessions/{session['id']}")).status == 204
    assert (await client.get(f"/api/agent/sessions/{session['id']}")).status == 404


async def test_session_validates_cwd_model_and_mode(client: TestClient, tmp_path: Path) -> None:
    missing = await client.post("/api/agent/sessions", json={"cwd": str(tmp_path / "nie-ma")})
    assert missing.status == 400 and (await missing.json())["code"] == "bad_cwd"

    bad_model = await client.post("/api/agent/sessions", json={"cwd": str(tmp_path), "modelKey": "x/y"})
    assert bad_model.status == 404 and (await bad_model.json())["code"] == "model_not_found"

    bad_mode = await client.post("/api/agent/sessions", json={"cwd": str(tmp_path), "mode": "turbo"})
    assert bad_mode.status == 400 and (await bad_mode.json())["code"] == "invalid_mode"


async def test_session_ws_requires_token(anon_client: TestClient, client: TestClient,
                                         tmp_path: Path) -> None:
    session = await make_session(client, tmp_path)
    response = await anon_client.get(f"/api/agent/sessions/{session['id']}/ws")
    assert response.status == 401
    ws = await anon_client.ws_connect(f"/api/agent/sessions/{session['id']}/ws?token={TOKEN}")
    try:
        assert (await recv(ws, "session"))["session"]["id"] == session["id"]
    finally:
        await ws.close()


# --------------------------------------------------------------------------- #
# WebSocket protocol
# --------------------------------------------------------------------------- #


async def test_ws_connect_sends_session_and_history(client: TestClient, core, tmp_path: Path) -> None:
    use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        first, second = await collect(ws, 2)
        assert first["type"] == "session" and first["session"]["id"] == session["id"]
        assert second == {"type": "history", "items": []}
    finally:
        await ws.close()


async def test_ws_user_turn_streams_events(client: TestClient, core, tmp_path: Path) -> None:
    agents = use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "Napraw testy"})
        ack = await recv(ws, "ack")
        assert ack["status"] == "started"
        assert (await recv(ws, "turn_start"))["userText"] == "Napraw testy"
        assert (await recv(ws, "text_delta"))["text"] == "Echo: Napraw testy"
        assert (await recv(ws, "turn_end"))["reason"] == "done"
        assert agents[0].turns == ["Napraw testy"]
    finally:
        await ws.close()

    detail = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
    kinds = [i["kind"] for i in detail["items"]]
    assert kinds == ["user", "assistant"]
    assert detail["items"][1]["text"] == "Echo: Napraw testy"
    assert detail["session"]["busy"] is False
    assert detail["session"]["contextUsed"] == 120 and detail["session"]["contextLimit"] == 8192
    assert detail["session"]["title"] == "Napraw testy"


async def test_ws_broadcasts_to_every_client(client: TestClient, core, tmp_path: Path) -> None:
    use_agent(core, None)
    session = await make_session(client, tmp_path)
    first = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    second = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(first, 2)
        await collect(second, 2)
        await first.send_json({"type": "user", "text": "Cześć"})
        assert (await recv(first, "text_delta"))["text"] == "Echo: Cześć"
        assert (await recv(second, "text_delta"))["text"] == "Echo: Cześć"
    finally:
        await first.close()
        await second.close()


async def test_ws_queues_user_message_while_busy(client: TestClient, core, tmp_path: Path) -> None:
    gate = asyncio.Event()

    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        agent.emit(TextDelta(text=f"pracuję: {text}"))
        if not gate.is_set():
            await gate.wait()

    agents = use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "pierwsza"})
        await recv(ws, "text_delta")               # the first turn now hangs on the gate

        await ws.send_json({"type": "user", "text": "druga"})
        status = await recv(ws, "status")       # broadcast to everyone …
        assert status["phase"] == "queued" and "kolejce" in status["message"]
        ack = await recv(ws, "ack", where=lambda d: d["status"] == "queued")
        assert ack["of"] == "user"              # … and acknowledged to the sender
        assert (await (await client.get(f"/api/agent/sessions/{session['id']}")).json())["session"]["busy"]

        gate.set()                                  # let the first turn finish
        second = await recv(ws, "turn_start", where=lambda d: d["userText"] == "druga")
        assert second["userText"] == "druga"
        await recv(ws, "turn_end", where=lambda d: True)
        assert agents[0].turns == ["pierwsza", "druga"]
    finally:
        gate.set()
        await ws.close()


async def test_ws_stopping_a_turn_sends_the_queued_message(client: TestClient, core, tmp_path: Path) -> None:
    """Stop is "stop answering", not "forget what I typed meanwhile"."""
    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        agent.emit(TextDelta(text=f"pracuję: {text}"))
        if text == "pierwsza":
            await agent.stop.wait()             # hangs until interrupted

    agents = use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "pierwsza"})
        await recv(ws, "text_delta")
        await ws.send_json({"type": "user", "text": "druga"})
        await recv(ws, "ack", where=lambda d: d["status"] == "queued")

        await ws.send_json({"type": "interrupt"})
        second = await recv(ws, "turn_start", where=lambda d: d["userText"] == "druga")
        assert second["userText"] == "druga"
        await recv(ws, "turn_end", where=lambda d: True)
        assert agents[0].turns == ["pierwsza", "druga"] and agents[0].interrupts == 1
    finally:
        agents[0].stop.set() if agents else None
        await ws.close()


async def test_ws_permission_first_answer_wins(client: TestClient, core, tmp_path: Path) -> None:
    decisions: list[str] = []

    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        decisions.append(await agent.ask())
        agent.emit(TextDelta(text="gotowe"))

    use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    first = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    second = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(first, 2)
        await collect(second, 2)
        await first.send_json({"type": "user", "text": "ls"})
        request = await recv(first, "permission_request")
        mirrored = await recv(second, "permission_request")
        assert request["requestId"] == mirrored["requestId"]
        assert request["options"] == ["allow", "deny", "always"]

        await second.send_json({"type": "permission", "requestId": request["requestId"],
                                "decision": "always"})
        assert (await recv(first, "text_delta"))["text"] == "gotowe"
        assert decisions == ["always"]

        # answering again is reported as stale
        await first.send_json({"type": "permission", "requestId": request["requestId"],
                               "decision": "deny"})
        ack = await recv(first, "ack")
        assert ack["of"] == "permission" and ack["status"] == "stale"
    finally:
        await first.close()
        await second.close()


async def test_pending_permission_is_resent_to_a_new_client(client: TestClient, core,
                                                            tmp_path: Path) -> None:
    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        await agent.ask(title="Zapisz plik", preview="--- a\n+++ b")

    use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    first = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(first, 2)
        await first.send_json({"type": "user", "text": "zapisz"})
        request = await recv(first, "permission_request")

        late = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
        try:
            resent = await recv(late, "permission_request")
            assert resent["requestId"] == request["requestId"]
            assert resent["title"] == "Zapisz plik"
            await late.send_json({"type": "permission", "requestId": request["requestId"],
                                  "decision": "allow"})
            assert (await recv(first, "turn_end"))["reason"] == "done"
        finally:
            await late.close()
    finally:
        await first.close()


async def test_permission_denied_when_nobody_is_attached(core, tmp_path: Path) -> None:
    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        agent.result = await agent.ask()

    use_agent(core, behaviour)
    session = core.agents.create(str(tmp_path))
    await session.send_user("zrób coś")
    for _ in range(100):
        await asyncio.sleep(0.01)
        if not session.busy:
            break
    # Refused — but as "dismissed", not the user's "deny": nothing is learned from it.
    assert session.agent.result == PERMISSION_DISMISSED


async def test_ws_commands(client: TestClient, core, tmp_path: Path) -> None:
    agents = use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "start"})
        await recv(ws, "turn_end")

        await ws.send_json({"type": "setMode", "mode": "yolo"})
        await recv(ws, "session", where=lambda d: d["session"]["mode"] == "yolo")
        assert agents[0].modes == ["yolo"]

        await ws.send_json({"type": "setModel", "modelKey": "Qwen3-0.6B-Q4_K_M"})
        await recv(ws, "session", where=lambda d: d["session"]["modelKey"] == CHAT_KEY)

        await ws.send_json({"type": "compact"})
        assert (await recv(ws, "compaction"))["strategy"] == "summarize"

        await ws.send_json({"type": "clear"})
        assert (await recv(ws, "history"))["items"] == []
        assert agents[0].clears == 1
    finally:
        await ws.close()


async def test_ws_goal_sets_it_and_starts_the_turn(client: TestClient, core, tmp_path: Path) -> None:
    agents = use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "setGoal", "text": "napraw wszystkie testy"})
        goal = await recv(ws, "goal")
        assert goal["text"] == "napraw wszystkie testy" and goal["state"] == "active"
        assert goal["maxTurns"] == 20
        start = await recv(ws, "turn_start")
        assert start["userText"] == "napraw wszystkie testy"
        await recv(ws, "turn_end")
        assert agents[0].turns == ["napraw wszystkie testy"]
        described = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
        assert described["session"]["goal"]["text"] == "napraw wszystkie testy"

        await ws.send_json({"type": "clearGoal"})
        assert (await recv(ws, "goal"))["state"] == "cleared"
        await recv(ws, "session", where=lambda d: d["session"]["goal"] is None)

        await ws.send_json({"type": "setGoal", "text": "  "})
        assert (await recv(ws, "error"))["code"] == "empty_goal"
    finally:
        await ws.close()


async def test_ws_interrupt(client: TestClient, core, tmp_path: Path) -> None:
    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        agent.emit(TextDelta(text="pracuję…"))
        await agent.stop.wait()

    agents = use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "długie zadanie"})
        await recv(ws, "text_delta")
        await ws.send_json({"type": "interrupt"})
        assert (await recv(ws, "turn_end"))["reason"] == "done"
        assert agents[0].interrupts == 1
        assert (await recv(ws, "session"))["session"]["busy"] is False
    finally:
        await ws.close()


async def test_ws_reports_bad_messages(client: TestClient, core, tmp_path: Path) -> None:
    use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_str("{nie-json")
        assert "JSON" in (await recv(ws, "error"))["message"]

        await ws.send_json({"type": "nieznany"})
        assert "nieznany" in (await recv(ws, "error"))["message"]

        await ws.send_json({"type": "user", "text": "  "})
        assert (await recv(ws, "error"))["code"] == "empty_message"

        await ws.send_json({"type": "permission", "requestId": "x", "decision": "maybe"})
        assert (await recv(ws, "error"))["code"] == "invalid_decision"
    finally:
        await ws.close()


async def test_tool_events_become_items(client: TestClient, core, tmp_path: Path) -> None:
    async def behaviour(agent: ScriptedAgent, text: str) -> None:
        agent.emit(ToolCallEvent(call_id="c1", name="bash", arguments={"command": "ls"},
                                 title="Uruchom: ls"))
        agent.emit(ToolResultEvent(call_id="c1", name="bash", ok=True, summary="exit 0",
                                   output="plik.txt", diff=None))
        agent.emit(TextDelta(text="Gotowe."))

    use_agent(core, behaviour)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "pokaż pliki"})
        await recv(ws, "turn_end")
    finally:
        await ws.close()
    items = (await (await client.get(f"/api/agent/sessions/{session['id']}")).json())["items"]
    assert [i["kind"] for i in items] == ["user", "tool", "assistant"]
    tool = items[1]
    assert tool["toolName"] == "bash" and tool["ok"] is True
    assert tool["summary"] == "exit 0" and tool["output"] == "plik.txt"
    assert tool["arguments"] == {"command": "ls"}


async def test_session_gone_closes_nothing_else(client: TestClient, core, tmp_path: Path) -> None:
    session = await make_session(client, tmp_path)
    assert (await client.delete(f"/api/agent/sessions/{session['id']}")).status == 204
    response = await client.get(f"/api/agent/sessions/{session['id']}/ws")
    assert response.status == 404


# --------------------------------------------------------------------------- #
# One command table, two interfaces
# --------------------------------------------------------------------------- #
# Kept as two lists, the terminal grew to 23 slash commands while the Mac app
# stayed at 4 — and an unknown one was sent to the model as an ordinary message.
# The app now reads the same table the terminal uses.


async def test_the_app_is_served_the_shared_slash_commands(client) -> None:
    from localcode.agent.prompts import INIT_PROMPT
    from localcode.tui.commands import COMMANDS

    response = await client.get("/api/agent/commands")
    assert response.status == 200
    body = await response.json()

    served = {c["name"] for c in body["commands"]}
    assert served == {c.name for c in COMMANDS if c.in_app}
    # The agent's own commands belong in both interfaces.
    assert {"mode", "model", "compact", "clear", "think", "context", "stats"} <= served
    # The ones the app has whole screens for, or that mean nothing in a window, do not.
    assert served.isdisjoint({"exit", "chat", "download", "runtime", "ps", "obraz"})

    assert body["initPrompt"] == INIT_PROMPT          # /init is a prompt, not a copy
    for command in body["commands"]:
        assert command["summary"]
        assert set(command) == {"name", "summary", "usage", "aliases", "values"}


# --------------------------------------------------------------------------- #
# A helper agent's events reach the clients without ending the parent's turn
# --------------------------------------------------------------------------- #


async def test_subagent_events_reach_the_client_and_leave_the_turn_running(
        core, client: TestClient, tmp_path: Path) -> None:
    """The reason they are wrapped at all.

    A helper's own `turn_end` would set the session idle and close the parent's
    turn in both the terminal and the Mac app, mid-work. The wrapped events must
    travel to the client and change nothing about the turn's state.
    """
    from localcode.events import SubagentEnd, SubagentStart, SubagentStep

    created = await make_session(client, tmp_path)
    state = core.agents.require(created["id"])
    state._busy = True
    before = len(state._items)

    for event in (
        SubagentStart(agent_id="h1", parent_call_id="c1", title="znajdź X", model="m"),
        SubagentStep(agent_id="h1", parent_call_id="c1", name="grep", title="grep X",
                     ok=True, summary="12 trafień"),
        SubagentEnd(agent_id="h1", parent_call_id="c1", ok=True, steps=1, elapsed_ms=120),
    ):
        state._reduce(event.to_dict())

    assert state._busy is True, "pomocnik nie kończy tury rodzica"
    assert len(state._items) == before, "kroki pomocnika nie są wpisami w transkrypcie"


# --------------------------------------------------------------------------- #
# @ścieżka works in both interfaces, from one implementation
# --------------------------------------------------------------------------- #


async def test_an_at_reference_from_a_graphical_client_is_expanded(
        core, client: TestClient, tmp_path: Path) -> None:
    """The terminal expanded these itself; the app sent them as plain text."""
    (tmp_path / "notatka.txt").write_text("treść pliku\n", encoding="utf-8")
    created = await make_session(client, tmp_path)
    state = core.agents.require(created["id"])

    expanded = state._expand_attachments("przeczytaj @notatka.txt")
    assert "przeczytaj @notatka.txt" in expanded          # the question is kept verbatim
    assert "treść pliku" in expanded                      # and the file travels with it
    assert '<file path="notatka.txt">' in expanded


async def test_a_message_the_terminal_already_expanded_is_left_alone(
        core, client: TestClient, tmp_path: Path) -> None:
    """Expanding twice would send every attached file to the model twice."""
    (tmp_path / "notatka.txt").write_text("treść pliku\n", encoding="utf-8")
    created = await make_session(client, tmp_path)
    state = core.agents.require(created["id"])

    once = state._expand_attachments("przeczytaj @notatka.txt")
    twice = state._expand_attachments(once)
    assert twice == once
    assert once.count("treść pliku") == 1


async def test_a_missing_attachment_is_reported_not_silently_dropped(
        core, client: TestClient, tmp_path: Path) -> None:
    created = await make_session(client, tmp_path)
    state = core.agents.require(created["id"])
    before = len(state._items)
    state._expand_attachments("przeczytaj @nie-ma-mnie.txt")
    notices = [i for i in state._items[before:] if i["kind"] == "notice"]
    assert any("nie-ma-mnie.txt" in str(n.get("text", "")) for n in notices)


# --------------------------------------------------------------------------- #
# The help bot over HTTP
# --------------------------------------------------------------------------- #


async def test_the_help_bot_reports_itself_missing_rather_than_failing(core, client: TestClient) -> None:
    core.settings.helpbot.model = "org/nie-ma-takiego"
    body = await (await client.get("/api/helpbot")).json()
    assert body["state"] == "missing"
    response = await client.post("/api/helpbot/session", json={})
    assert response.status == 409
    assert (await response.json())["code"] == "helpbot_missing"


async def test_the_help_bots_session_lives_in_its_own_directory(core, client: TestClient,
                                                                isolated_home: Path) -> None:
    """That directory is what makes the server build a help bot for it."""
    from localcode import helpbot

    body = await (await client.get("/api/helpbot")).json()
    assert body["state"] == "ready"
    first = await client.post("/api/helpbot/session", json={})
    assert first.status == 200
    session = await first.json()
    assert helpbot.is_helpbot_cwd(session["cwd"])
    again = await (await client.post("/api/helpbot/session", json={})).json()
    assert again["id"] == session["id"]                       # the latest, not a new one
    fresh = await client.post("/api/helpbot/session", json={"new": True})
    assert fresh.status == 201
    assert (await fresh.json())["id"] != session["id"]


async def test_a_session_can_switch_to_a_remote_model(client: TestClient, core, tmp_path: Path) -> None:
    """Regression: the terminal attached to the app's server picked a DeepInfra
    model with /model, and the server answered "Nie znaleziono modelu" because
    it only looked on disk — the conversation silently stayed on the local model."""
    class Registry:
        providers = ["deepinfra"]

        def configured(self, provider: str) -> bool:
            return provider == "deepinfra"

    core.runtime.remote = Registry()
    agents = use_agent(core, None)
    key = "remote:deepinfra:deepseek-ai/DeepSeek-V4.1-Flash"
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "start"})
        await recv(ws, "turn_end")
        await ws.send_json({"type": "setModel", "modelKey": key})
        changed = await recv(ws, "session", where=lambda d: d["session"]["modelKey"] == key)
        assert changed["session"]["modelKey"] == key
        assert agents[0].models == [key]
        # A provider without a key is still refused, and so is loading one into memory.
        await ws.send_json({"type": "setModel", "modelKey": "remote:openai:gpt-5"})
        assert (await recv(ws, "error"))["code"] == "model_not_found"
    finally:
        await ws.close()
    assert core.resolve_model(key) is None                    # not for load/estimate/tune
    assert core.resolve_model(key, remote=True).key == key


async def test_a_message_can_carry_pictures(client: TestClient, core, tmp_path: Path) -> None:
    agents = use_agent(core, None)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    picture = {"name": "zrzut.png", "mime": "image/png", "data": "iVBORw0KGgo="}
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "co to?", "images": [picture]})
        await recv(ws, "turn_end")
        assert agents[0].images == [picture]
        await ws.send_json({"type": "user", "text": "x", "images": [{"name": "a.exe", "mime": "application/x",
                                                                     "data": "AA=="}]})
        assert (await recv(ws, "error"))["code"] == "invalid_images"
    finally:
        await ws.close()
