"""Tests for ``delegate``: handing one search to a helper with its own context.

What is worth pinning here is not that a helper runs — it is what a helper is
*not* allowed to be. It cannot write, it cannot ask the user for anything, it
cannot start another helper, and its own events must never reach the client,
because a helper's ``turn_end`` would end the parent's turn in both the terminal
and the Mac app.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, ScriptedEngine, fake_model
from localcode.agent.loop import Agent
from localcode.agent.tools import tools_for_mode
from localcode.agent.tools.base import ToolContext, ToolLimits
from localcode.agent.tools.subagent import (
    HELPER_TOOLS,
    MAX_PER_TURN,
    STATE_KEY,
    SubagentHost,
    SubagentTool,
    _helper_tools,
)
from localcode.config import Settings


def settings(**agent_fields: Any) -> Settings:
    s = Settings()
    s.agent = replace(s.agent, subagents=True, **agent_fields)
    return s


class FakeHelper:
    """Stands in for the helper ``Agent``; records what it was asked."""

    def __init__(self, answer: str = "znalazłem: app.py:12 — tu jest wejście",
                 events: list[Any] = (), error: BaseException | None = None) -> None:
        self.answer = answer
        self.events = list(events)
        self.error = error
        self.task: str | None = None
        self.interrupted = 0
        self.on_event: Any = None

    async def run_turn(self, text: str) -> str:
        self.task = text
        for event in self.events:
            await self.on_event(event)
        if self.error is not None:
            raise self.error
        return self.answer

    def interrupt(self) -> None:
        self.interrupted += 1


def host_with(helper: FakeHelper, *, emitted: list[Any] | None = None,
              runtime: Any = None, config: Settings | None = None,
              model_key: str = "test/model", cwd: str = "/tmp") -> SubagentHost:
    events = emitted if emitted is not None else []

    async def emit(event: Any) -> None:
        events.append(event)

    def factory(_host: SubagentHost, on_event: Any) -> FakeHelper:
        helper.on_event = on_event
        return helper

    return SubagentHost(runtime=runtime or FakeRuntime(ScriptedEngine([])),
                        settings=config or settings(), cwd=cwd, model_key=model_key,
                        emit=emit, agent_factory=factory)


def context(host: SubagentHost | None, *, call_id: str = "call-1",
            limits: ToolLimits | None = None) -> ToolContext:
    state: dict[str, Any] = {"current_call_id": call_id}
    if host is not None:
        state[STATE_KEY] = host
    return ToolContext(cwd="/tmp", limits=limits or ToolLimits(), state=state)


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #


def test_delegate_is_offered_only_when_it_is_switched_on() -> None:
    assert "delegate" not in [t.name for t in tools_for_mode("default")]
    assert "delegate" in [t.name for t in tools_for_mode("default", subagents=True)]


def test_the_eight_ordinary_tools_are_unchanged() -> None:
    """Adding a tool behind a flag must not move the default menu."""
    from localcode.agent.tools import TOOL_NAMES

    assert "delegate" not in TOOL_NAMES


def test_the_schema_stays_small_enough_for_a_tiny_window() -> None:
    schema = SubagentTool().schema()["function"]
    assert len(schema["description"]) <= 200
    assert list(schema["parameters"]["properties"]) == ["task", "wait"]
    assert schema["parameters"]["required"] == ["task"]


@pytest.mark.parametrize("alias", ["task", "subagent", "sub_agent", "spawn_agent", "helper"])
def test_the_names_a_model_reaches_for_resolve_to_delegate(alias: str) -> None:
    from localcode.agent.tools import get_tool

    tool = get_tool(alias)
    assert tool is not None and tool.name == "delegate"


# --------------------------------------------------------------------------- #
# What a helper may do
# --------------------------------------------------------------------------- #


def test_a_helper_reads_and_runs_commands_but_never_edits() -> None:
    names = [t.name for t in _helper_tools("default", False)]
    assert set(names) == set(HELPER_TOOLS) and "bash" in names
    assert not ({"write_file", "edit_file", "todo_write", "delegate"} & set(names))
    assert "bash" not in [t.name for t in _helper_tools("plan", False)], "plan mode: no commands"


def test_a_helper_cannot_start_another_helper() -> None:
    """Depth one, on purpose: nothing about a 4B model suggests it would stop."""
    assert "delegate" not in [t.name for t in _helper_tools("default", False)]
    config = settings()
    host = host_with(FakeHelper(), config=config)
    assert host.helper_settings().agent.subagents is False


def test_a_helper_does_not_get_the_model_management_tools() -> None:
    host = host_with(FakeHelper(), config=settings(manage_models=True))
    assert host.helper_settings().agent.manage_models is False


def test_a_helper_gets_the_skill_tool_only_when_there_is_a_shelf() -> None:
    """The prompt mentions the shelf, so the tool has to be there when it does."""
    assert "skill" not in [t.name for t in _helper_tools("default", False, skills=False)]
    assert "skill" in [t.name for t in _helper_tools("default", False, skills=True)]


def test_a_tiny_window_drops_glob_for_the_helper_too() -> None:
    assert "glob" not in [t.name for t in _helper_tools("default", True)]
    assert "grep" in [t.name for t in _helper_tools("default", True)]


# --------------------------------------------------------------------------- #
# Which model, and whose decision it is
# --------------------------------------------------------------------------- #


def test_a_helper_runs_on_the_parents_model_by_default() -> None:
    host = host_with(FakeHelper(), model_key="org/Qwen3-4B")
    assert host.helper_model_key == "org/Qwen3-4B"


def test_only_the_user_can_point_a_helper_at_another_model() -> None:
    """The model must never choose: a helper reads the project's files."""
    host = host_with(FakeHelper(), model_key="org/Qwen3-4B",
                     config=settings(subagent_model="remote:unlid:glm-5.3"))
    assert host.helper_model_key == "remote:unlid:glm-5.3"
    # And there is no way to ask for one through the tool.
    assert "model" not in SubagentTool().parameters["properties"]


async def test_a_remote_helper_is_announced_as_remote() -> None:
    emitted: list[Any] = []
    host = host_with(FakeHelper(), emitted=emitted,
                     config=settings(subagent_model="remote:unlid:glm-5.3"))
    await SubagentTool().run({"task": "znajdź X"}, context(host))
    start = next(e for e in emitted if e.type == "subagent_start")
    assert start.remote is True and start.model == "remote:unlid:glm-5.3"


async def test_a_local_helper_that_does_not_fit_is_refused_before_it_starts() -> None:
    """Evicting the parent mid-turn costs a reload; saying so costs nothing."""
    class Tight:
        def estimate(self, key: str) -> Any:
            return type("Est", (), {"fits": "no"})()

    host = host_with(FakeHelper(), runtime=Tight(), model_key="org/mały",
                     config=settings(subagent_model="org/ogromny"))
    result = await SubagentTool().run({"task": "znajdź X"}, context(host))
    assert result.ok is False
    assert "memory" in result.output


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


async def test_the_parent_gets_the_report_and_not_the_helpers_transcript() -> None:
    helper = FakeHelper(answer="app.py:12 — tu jest wejście")
    result = await SubagentTool().run({"task": "gdzie jest wejście?"}, context(host_with(helper)))
    assert result.ok
    assert "app.py:12" in result.output
    assert helper.task == "gdzie jest wejście?"       # the helper sees only the task


async def test_the_report_is_cut_to_a_share_of_what_one_result_may_cost() -> None:
    helper = FakeHelper(answer="x" * 50_000)
    small = context(host_with(helper), limits=ToolLimits(max_output_chars=6_000))
    result = await SubagentTool().run({"task": "znajdź X"}, small)
    assert len(result.output) < 3_000


async def test_a_helper_that_answers_nothing_is_a_failure_not_an_empty_success() -> None:
    result = await SubagentTool().run({"task": "znajdź X"},
                                      context(host_with(FakeHelper(answer="   "))))
    assert result.ok is False


async def test_a_broken_helper_does_not_kill_the_parents_turn() -> None:
    helper = FakeHelper(error=RuntimeError("silnik padł"))
    result = await SubagentTool().run({"task": "znajdź X"}, context(host_with(helper)))
    assert result.ok is False and "silnik padł" in result.output


async def test_without_a_host_the_tool_says_so_instead_of_failing() -> None:
    result = await SubagentTool().run({"task": "znajdź X"}, context(None))
    assert result.ok is False and "not available" in result.output


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


async def test_the_helpers_own_events_are_wrapped_never_forwarded() -> None:
    """A helper's turn_end would end the parent's turn in both clients."""
    from localcode.events import TextDelta, ToolCallEvent, ToolResultEvent, TurnEnd

    helper = FakeHelper(events=[
        TextDelta(text="myślę…"),
        ToolCallEvent(call_id="h1", name="grep", arguments={}, title="grep „katalog”"),
        ToolResultEvent(call_id="h1", name="grep", ok=True, summary="12 trafień", output=""),
        TurnEnd(turn_id="t", reason="done", stats=None),
    ])
    emitted: list[Any] = []
    await SubagentTool().run({"task": "znajdź X"}, context(host_with(helper, emitted=emitted)))

    kinds = [e.type for e in emitted]
    assert kinds == ["subagent_start", "subagent_step", "subagent_end"]
    assert "turn_end" not in kinds and "text_delta" not in kinds


async def test_every_event_carries_the_call_it_belongs_to() -> None:
    from localcode.events import ToolCallEvent, ToolResultEvent

    helper = FakeHelper(events=[
        ToolCallEvent(call_id="h1", name="read_file", arguments={}, title="Read app.py"),
        ToolResultEvent(call_id="h1", name="read_file", ok=True, summary="40 linii", output=""),
    ])
    emitted: list[Any] = []
    await SubagentTool().run({"task": "znajdź X"},
                             context(host_with(helper, emitted=emitted), call_id="call-77"))
    assert {e.parent_call_id for e in emitted} == {"call-77"}
    assert len({e.agent_id for e in emitted}) == 1
    step = next(e for e in emitted if e.type == "subagent_step")
    assert step.title == "Read app.py" and step.summary == "40 linii"


async def test_the_steps_survive_in_the_stored_tool_result() -> None:
    """``display`` is kept in the transcript, so reopening a session shows them."""
    from localcode.events import ToolCallEvent, ToolResultEvent

    helper = FakeHelper(events=[
        ToolCallEvent(call_id="h1", name="grep", arguments={}, title="grep „katalog”"),
        ToolResultEvent(call_id="h1", name="grep", ok=True, summary="12 trafień", output=""),
    ])
    result = await SubagentTool().run({"task": "znajdź X"}, context(host_with(helper)))
    assert "grep „katalog”" in (result.display or "")
    assert "grep „katalog”" not in result.output        # but not in what the model reads


# --------------------------------------------------------------------------- #
# Limits and interruption
# --------------------------------------------------------------------------- #


async def test_a_turn_cannot_be_spent_on_helpers() -> None:
    host = host_with(FakeHelper())
    for _ in range(MAX_PER_TURN):
        assert (await SubagentTool().run({"task": "znajdź X"}, context(host))).ok
    refused = await SubagentTool().run({"task": "znajdź X"}, context(host))
    assert refused.ok is False and "limit" in refused.summary


async def test_interrupting_the_parent_stops_the_helper_and_is_not_swallowed() -> None:
    helper = FakeHelper(error=asyncio.CancelledError())
    emitted: list[Any] = []
    host = host_with(helper, emitted=emitted)
    with pytest.raises(asyncio.CancelledError):
        await SubagentTool().run({"task": "znajdź X"}, context(host))
    assert helper.interrupted == 1
    end = next(e for e in emitted if e.type == "subagent_end")
    assert end.ok is False
    # The slot is free again: an interrupted turn must not block the next one.
    assert host._slot._value == 1


async def test_only_one_helper_runs_at_a_time() -> None:
    """Two models in memory is the budget problem this project already has."""
    host = host_with(FakeHelper())
    await host.start()
    second = asyncio.create_task(host.start())
    await asyncio.sleep(0)
    assert not second.done()
    host.finish()
    await asyncio.wait_for(second, timeout=1)
    host.finish()


# --------------------------------------------------------------------------- #
# The session wires it up
# --------------------------------------------------------------------------- #


def test_the_session_registers_a_host_only_when_subagents_are_on() -> None:
    engine = ScriptedEngine(["gotowe"])
    off = Agent(FakeRuntime(engine), "test/model", "/tmp", Settings(), EventLog(), None)
    assert STATE_KEY not in off._tool_ctx.state

    on = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(), EventLog(), None)
    assert isinstance(on._tool_ctx.state.get(STATE_KEY), SubagentHost)
    on.clear()
    assert isinstance(on._tool_ctx.state.get(STATE_KEY), SubagentHost), "przeżywa /clear"


async def test_the_helpers_role_sits_in_the_system_prompt(tmp_path, isolated_home) -> None:
    """Not in the first user message: a conversation can be compacted away."""
    from localcode.agent.tools.subagent import HELPER_ROLE

    engine = ScriptedEngine(["gotowe"], n_ctx=8192)
    helper = Agent(FakeRuntime(engine), "test/model", tmp_path, Settings(), EventLog(), None,
                   system_suffix=HELPER_ROLE)
    await helper.run_turn("znajdź X")
    prompt = engine.prompts[0]
    assert "helper agent working on ONE task" in prompt
    assert "at most 150 words" in prompt
