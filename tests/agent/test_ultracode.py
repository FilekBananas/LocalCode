"""Effort "ultracode": the agent orchestrates parallel helpers and thinks at "high".

What is worth pinning: the name never reaches a template or an engine; the
``delegate`` calls of one step really run side by side (and no more of them
than the cap), their results land in the order the model wrote them, an
interrupt stops every one of them, and the orchestrator section is in the
prompt only when ultracode is on and ``delegate`` is on the menu.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, FakeTool, Reply, ScriptedEngine, tool_call_text
from localcode.agent import prompts
from localcode.agent.loop import INTERRUPTED_RESULT, Agent
from localcode.agent.tools.base import ToolContext, ToolResult
from localcode.agent.tools.subagent import (
    MAX_PER_TURN,
    STATE_KEY,
    ULTRACODE_MAX_PER_TURN,
    SubagentHost,
    SubagentTool,
)
from localcode.config import Settings
from localcode.runtime.templates import (
    AGENT_EFFORT_CHOICES,
    ULTRACODE,
    effort_kwargs,
    engine_effort,
    normalize_agent_effort,
    normalize_effort,
    resolve_effort,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def settings(**agent_fields: Any) -> Settings:
    s = Settings()
    s.agent = replace(s.agent, **agent_fields)
    return s


def build(replies: Any, *, tools: Any = None, n_ctx: int = 32768, supports: bool = True,
          **agent_fields: Any) -> tuple[Agent, ScriptedEngine, EventLog]:
    engine = ScriptedEngine([Reply.coerce(r) for r in replies], n_ctx=n_ctx,
                            supports_thinking=supports)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp",
                  settings(permission_mode="yolo", **agent_fields), log, None, tools=tools)
    agent.set_effort("ultracode")
    return agent, engine, log


# --------------------------------------------------------------------------- #
# The name stops at the engine boundary
# --------------------------------------------------------------------------- #


def test_the_levels_and_their_normalisation() -> None:
    assert AGENT_EFFORT_CHOICES[-1] == ULTRACODE == "ultracode"
    assert normalize_agent_effort(" UltraCode ") == "ultracode"
    assert normalize_agent_effort("high") == "high"
    assert normalize_agent_effort("bardzo") is None
    # Surfaces that only know thinking levels (the chat, /v1) get what it implies.
    assert normalize_effort("ultracode") == "high"
    assert engine_effort("ultracode") == "high" and engine_effort("low") == "low"
    assert resolve_effort("ultracode", ["low", "medium", "high"]) == "high"
    assert resolve_effort("ultracode", ["off", "on"]) == "on"
    assert effort_kwargs("reasoning_effort", "ultracode") == {"reasoning_effort": "high"}
    assert effort_kwargs("thinking_budget", "ultracode") == {"thinking_budget": 8192}


async def test_ultracode_never_reaches_the_template_or_the_sampling() -> None:
    agent, engine, _log = build(["gotowe"], tools=[])
    assert agent.effort == "ultracode" and agent.ultracode
    await agent.run_turn("cześć")
    assert engine.template_kwargs, "the prompt was rendered"
    assert all(k == {"enable_thinking": True} for k in engine.template_kwargs)
    assert "ultracode" not in repr(engine.template_kwargs)
    assert all("ultracode" not in repr(p) for p in engine.sampling)


async def test_a_dial_template_gets_high() -> None:
    engine = ScriptedEngine(["gotowe"], n_ctx=32768, supports_thinking=True)
    engine.template_info.effort_kwarg = "reasoning_effort"
    engine.template_info.effort_levels = ["low", "medium", "high"]
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(), EventLog(), None, tools=[])
    agent.set_effort("ultracode")
    await agent.run_turn("cześć")
    assert engine.template_kwargs[-1] == {"reasoning_effort": "high"}


def test_ultracode_from_the_settings_and_back() -> None:
    agent = Agent(FakeRuntime(ScriptedEngine([])), "test/model", "/tmp",
                  settings(reasoning_effort="ultracode"), EventLog(), None, tools=[])
    assert agent.effort == "ultracode"
    agent.set_effort("low")
    assert not agent.ultracode
    with pytest.raises(ValueError):
        agent.set_effort("bardzo")


# --------------------------------------------------------------------------- #
# The menu and the prompt
# --------------------------------------------------------------------------- #


def _system(engine: ScriptedEngine) -> str:
    return engine.prompts[0].split("<|user|>")[0]


async def test_ultracode_offers_delegate_and_the_orchestrator_section() -> None:
    agent, engine, _log = build(["gotowe"])            # subagents off in the settings
    await agent.run_turn("dodaj funkcję")
    assert "delegate" in [t.name for t in agent.tools]
    system = _system(engine)
    assert "# Ultracode (orchestrator)" in system
    assert "Fan out FIRST" in system and "todo_write" in system
    assert isinstance(agent._tool_ctx.state.get(STATE_KEY), SubagentHost)


async def test_without_ultracode_there_is_no_section_and_no_delegate() -> None:
    agent, engine, _log = build(["gotowe"])
    agent.set_effort("high")
    await agent.run_turn("dodaj funkcję")
    assert "delegate" not in [t.name for t in agent.tools]
    assert "Ultracode" not in _system(engine)


async def test_a_small_window_keeps_delegate_off_and_says_so(caplog) -> None:
    agent, engine, _log = build(["gotowe"], n_ctx=8192)
    with caplog.at_level(logging.INFO, logger="localcode.agent.loop"):
        await agent.run_turn("dodaj funkcję")
    assert "delegate" not in [t.name for t in agent.tools]
    assert "Ultracode" not in _system(engine)
    assert any("ultracode" in r.getMessage() and "delegate stays off" in r.getMessage()
               for r in caplog.records)


def test_the_section_follows_the_tiers_and_the_menu() -> None:
    env = {"cwd": "/tmp", "os": "macOS", "date": "2026-10-20"}
    common = dict(cwd="/tmp", mode="default", native_tools=True, project_instructions=None, env=env)
    names = ["read_file", "edit_file", "bash", "todo_write", "delegate"]
    for n_ctx in (8192, 32768, 131072):
        on = prompts.build_system_prompt(n_ctx=n_ctx, tool_names=names, ultracode=True, **common)
        off = prompts.build_system_prompt(n_ctx=n_ctx, tool_names=names, **common)
        assert "ULTRACODE" in on.upper() and "ULTRACODE" not in off.upper()
        assert on.startswith(off.split("\n\n")[0])
    without = prompts.build_system_prompt(n_ctx=32768, tool_names=names[:-1], ultracode=True, **common)
    assert "Ultracode" not in without, "no delegate, no orchestrating"


# --------------------------------------------------------------------------- #
# Running a step's delegate calls side by side
# --------------------------------------------------------------------------- #


class Probe:
    """Counts how many fake helpers run at once."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    def tool(self, *, delays: dict[str, float] | None = None, forever: bool = False) -> FakeTool:
        async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                if forever:
                    await asyncio.Event().wait()
                await asyncio.sleep((delays or {}).get(args["task"], 0.2))
            finally:
                self.running -= 1
            return ToolResult(True, f"raport: {args['task']} (call {ctx.call_id})", "pomocnik: 1 krok")

        return FakeTool("delegate", kind="meta", handler=handler,
                        parameters={"type": "object", "properties": {"task": {"type": "string"}}})


def step_with(*tasks: str, extra: str = "") -> str:
    calls = [tool_call_text("delegate", {"task": t}) for t in tasks]
    return "\n".join(calls) + (("\n" + extra) if extra else "")


async def test_delegate_calls_of_one_step_run_side_by_side_in_call_order() -> None:
    probe = Probe()
    reader = FakeTool("read_file", output="treść pliku")
    delays = {"a": 0.4, "b": 0.1, "c": 0.25}
    agent, _engine, log = build(
        [step_with("a", "b", "c", extra=tool_call_text("read_file", {"path": "x.py"})), "gotowe"],
        tools=[probe.tool(delays=delays), reader])
    started = time.perf_counter()
    await agent.run_turn("zbadaj projekt")
    elapsed = time.perf_counter() - started

    assert probe.peak == 3
    assert elapsed < 0.4 + 0.1 + 0.25 - 0.1, f"ran one after another ({elapsed:.2f} s)"
    tool_messages = [m for m in agent.history if m["role"] == "tool"]
    assert [m["name"] for m in tool_messages] == ["delegate", "delegate", "delegate", "read_file"]
    assert [m["content"].split(" (call")[0] for m in tool_messages[:3]] == [
        "raport: a", "raport: b", "raport: c"], "the history keeps the order the model wrote"
    assistant = next(m for m in agent.history if m.get("tool_calls"))
    ids = [c["id"] for c in assistant["tool_calls"]]
    assert [m["tool_call_id"] for m in tool_messages] == ids
    # Each helper ran with its own call id, not whichever one was set last.
    for message, call_id in zip(tool_messages[:3], ids):
        assert message["content"].endswith(f"(call {call_id})")
    assert log.last("turn_end").reason == "done"


async def test_the_cap_is_respected() -> None:
    probe = Probe()
    agent, _engine, _log = build([step_with("a", "b", "c", "d", "e"), "gotowe"],
                                 tools=[probe.tool()], ultracode_parallel=2)
    started = time.perf_counter()
    await agent.run_turn("zbadaj")
    elapsed = time.perf_counter() - started
    assert probe.peak == 2
    assert elapsed >= 0.55, "five helpers two at a time take three rounds"
    assert len([m for m in agent.history if m["role"] == "tool"]) == 5


async def test_without_ultracode_delegate_calls_run_one_by_one() -> None:
    probe = Probe()
    agent, _engine, _log = build([step_with("a", "b"), "gotowe"], tools=[probe.tool()])
    agent.set_effort("high")
    await agent.run_turn("zbadaj")
    assert probe.peak == 1


async def test_interrupt_cancels_every_running_helper() -> None:
    probe = Probe()
    tool = probe.tool(forever=True)
    agent, _engine, log = build([step_with("a", "b", "c"), "nie dojdzie"], tools=[tool])
    turn = asyncio.create_task(agent.run_turn("zbadaj"))
    for _ in range(200):
        if probe.running == 3:
            break
        await asyncio.sleep(0.01)
    assert probe.running == 3
    agent.interrupt()
    await asyncio.wait_for(turn, timeout=2)

    assert probe.running == 0 and tool.cancelled == 3
    assert log.last("turn_end").reason == "interrupted"
    results = [m["content"] for m in agent.history if m["role"] == "tool"]
    assert results == [INTERRUPTED_RESULT] * 3, "every issued call is answered"
    assert not agent._helper_tasks


# --------------------------------------------------------------------------- #
# The real delegate tool: events stay apart, helpers get more steps
# --------------------------------------------------------------------------- #


class SlowHelper:
    def __init__(self, task_delay: float) -> None:
        self.delay = task_delay
        self.on_event: Any = None

    async def run_turn(self, text: str) -> str:
        from localcode.events import ToolCallEvent, ToolResultEvent

        await self.on_event(ToolCallEvent(call_id="h", name="grep", arguments={}, title=f"grep {text}"))
        await asyncio.sleep(self.delay)
        await self.on_event(ToolResultEvent(call_id="h", name="grep", ok=True, summary="3 trafienia",
                                            output=""))
        return f"{text}: app.py:1 — znalezione"

    def interrupt(self) -> None:
        pass


async def test_parallel_helpers_keep_their_events_apart() -> None:
    engine = ScriptedEngine([step_with("moduł A", "moduł B"), "gotowe"], n_ctx=32768)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(permission_mode="yolo"),
                  log, None, tools=[SubagentTool()])
    agent.set_effort("ultracode")

    async def emit(event: Any) -> None:
        log(event)

    host = SubagentHost(runtime=agent.runtime, settings=agent.settings, cwd=agent.cwd,
                        model_key="test/model", emit=emit,
                        agent_factory=lambda _host, on_event: _bind(SlowHelper(0.2), on_event))
    agent._tool_ctx.state[STATE_KEY] = host
    await agent.run_turn("zbadaj oba moduły")

    ids = [c["id"] for c in next(m for m in agent.history if m.get("tool_calls"))["tool_calls"]]
    starts = log.of("subagent_start")
    assert sorted(e.parent_call_id for e in starts) == sorted(ids)
    for call_id in ids:
        steps = [e for e in log.of("subagent_step") if e.parent_call_id == call_id]
        ends = [e for e in log.of("subagent_end") if e.parent_call_id == call_id]
        assert len(steps) == 1 and len(ends) == 1 and ends[0].ok
    # Both started before either finished: they really overlapped.
    first_end = log.types.index("subagent_end")
    assert log.types[:first_end].count("subagent_start") == 2
    assert len({e.agent_id for e in starts}) == 2
    assert host.ultracode and host.max_per_turn == ULTRACODE_MAX_PER_TURN > MAX_PER_TURN


def _bind(helper: SlowHelper, on_event: Any) -> SlowHelper:
    helper.on_event = on_event
    return helper


def test_helpers_get_twice_the_steps_and_never_ultracode() -> None:
    config = settings(subagent_max_steps=12, reasoning_effort="ultracode")
    host = SubagentHost(runtime=None, settings=config, cwd="/tmp", model_key="m", emit=None)
    plain = host.helper_settings().agent
    assert plain.max_steps == 12 and plain.reasoning_effort == "high" and plain.subagents is False
    host.set_ultracode(True, 4)
    boosted = host.helper_settings().agent
    assert boosted.max_steps == 24 and boosted.reasoning_effort == "high"
    assert boosted.subagents is False, "still depth one"
    assert host._slot._value == 4
    host.set_ultracode(False)
    assert host._slot._value == 1 and host.max_per_turn == MAX_PER_TURN


async def test_the_per_turn_limit_holds_for_helpers_started_together() -> None:
    """The count is taken before the first await, so a whole step cannot slip past it."""
    host = SubagentHost(runtime=None, settings=settings(), cwd="/tmp", model_key="m", emit=None)
    assert all(host.claim() for _ in range(MAX_PER_TURN))
    assert not host.claim()
