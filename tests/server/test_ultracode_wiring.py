"""Choosing "Ultracode" in the Kod header reaches the running agent.

The Mac app sends ``{"type": "setEffort", "effort": "ultracode"}`` over the
session's WebSocket (EffortControl → AgentStore.setEffort). These tests run the
real agent loop behind the real server and check the far end: the agent is in
ultracode, ``delegate`` is on its menu and the orchestrator section is in the
prompt the model gets — for a fresh session, for a session in the middle of a
turn, and for a model behind an API key.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from aiohttp.test_utils import TestClient

_AGENT_TESTS = Path(__file__).resolve().parent.parent / "agent"
if str(_AGENT_TESTS) not in sys.path:
    sys.path.insert(0, str(_AGENT_TESTS))

from agent_fakes import FakeRuntime, FakeTool, ScriptedEngine, fake_model, tool_call_text  # noqa: E402
from localcode.agent.loop import Agent  # noqa: E402
from localcode.agent.tools.base import ToolContext, ToolResult  # noqa: E402
from test_agent_ws import collect, make_session, recv  # noqa: E402

SECTION = "# Ultracode (orchestrator)"
REMOTE = "remote:deepinfra:deepseek-v4.1-flash"


def install(core: Any, engine: ScriptedEngine, key: str, tools: Any = None) -> list[Agent]:
    """Build real agents for the server's sessions, on ``engine``."""
    built: list[Agent] = []

    def factory(session: Any) -> Agent:
        agent = Agent(FakeRuntime(engine, model_key=key), key, session.cwd, core.settings,
                      session.on_event, session.ask_permission, tools=tools)
        built.append(agent)
        return agent

    core.agents.agent_factory = factory
    return built


async def set_ultracode(ws: Any) -> None:
    await ws.send_json({"type": "setEffort", "effort": "ultracode"})
    await recv(ws, "session", where=lambda d: d["session"]["effort"] == "ultracode")


async def test_ultracode_from_the_header_reaches_a_remote_agent(client: TestClient, core,
                                                               tmp_path: Path) -> None:
    engine = ScriptedEngine(["Gotowe."], n_ctx=131072, model=fake_model(REMOTE, fmt="remote"))
    engine.format = "remote"
    built = install(core, engine, REMOTE)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await set_ultracode(ws)                       # before the agent even exists
        await ws.send_json({"type": "user", "text": "przetestuj bota"})
        await recv(ws, "turn_end")
    finally:
        await ws.close()

    [agent] = built
    assert agent.effort == "ultracode" and agent.ultracode
    assert "delegate" in [t.name for t in agent.tools]
    assert SECTION in engine.prompts[0]
    assert '"name": "delegate"' in engine.prompts[0], "the schema is on the menu the model sees"


async def test_ultracode_chosen_mid_turn_applies_from_the_next_step(client: TestClient, core,
                                                                    tmp_path: Path) -> None:
    release = asyncio.Event()

    async def blocking(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await release.wait()
        return ToolResult(True, "treść", "1 linia")

    engine = ScriptedEngine([tool_call_text("read_file", {"path": "bot.py"}), "Gotowe."],
                            n_ctx=65536)
    from localcode.agent.tools.subagent import SubagentTool  # noqa: PLC0415

    def menu(mode: str, tiny: bool = False, *, subagents: bool = False) -> list[Any]:
        tools: list[Any] = [FakeTool("read_file", handler=blocking)]
        if subagents:
            tools.append(SubagentTool())
        return tools

    built = install(core, engine, "test/model", tools=menu)
    session = await make_session(client, tmp_path)
    ws = await client.ws_connect(f"/api/agent/sessions/{session['id']}/ws")
    try:
        await collect(ws, 2)
        await ws.send_json({"type": "user", "text": "popraw bota"})
        await recv(ws, "tool_call")                  # step 1 is done; its tool is running
        await set_ultracode(ws)
        release.set()
        await recv(ws, "turn_end")
    finally:
        await ws.close()

    [agent] = built
    assert agent.ultracode
    assert SECTION not in engine.prompts[0]
    assert SECTION in engine.prompts[1], "the running turn's next step is an ultracode one"
    assert "delegate" in [t.name for t in agent.tools]
