"""Background helpers, helpers that run commands, and ultracode's push to delegate.

A real session (game-bot work: matches of several minutes, replays to unpack,
logs to analyse) made ~480 tool calls and four ``delegate`` calls. What is
pinned here is what should change that: a helper can go to the background and
the agent works on, its report arrives at the next step boundary (never
dropped — a turn the model ends waits for it), an interrupt stops them all;
helpers can run commands, but every command goes through the parent's policy
and approver under the helper's name; and ultracode asks for a fan-out first
and reminds an agent that has not delegated for a while.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, FakeTool, Reply, ScriptedEngine, tool_call_text
from localcode.agent import prompts
from localcode.agent.loop import Agent
from localcode.agent.tools.base import ToolContext, ToolResult
from localcode.agent.tools.subagent import STATE_KEY, SubagentHost, SubagentTool, _helper_tools
from localcode.config import Settings

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

NOTE_HEAD = "[Background helper h1 finished — its report:]"


def settings(**agent_fields: Any) -> Settings:
    s = Settings()
    s.agent = replace(s.agent, **agent_fields)
    return s


class TimedHelper:
    """Stands in for a helper agent: waits, then answers."""

    live = 0
    peak = 0

    def __init__(self, delay: float | None, answer: str) -> None:
        self.delay = delay                 # None = until cancelled
        self.answer = answer
        self.interrupted = 0
        self.on_event: Any = None

    async def run_turn(self, text: str) -> str:
        TimedHelper.live += 1
        TimedHelper.peak = max(TimedHelper.peak, TimedHelper.live)
        try:
            if self.delay is None:
                await asyncio.Event().wait()
            await asyncio.sleep(self.delay)
        finally:
            TimedHelper.live -= 1
        return f"{self.answer} ({text})"

    def interrupt(self) -> None:
        self.interrupted += 1


def ultracode_agent(replies: list[Any], helpers: list[TimedHelper], *, extra_tools=(),
                    **agent_fields: Any) -> tuple[Agent, ScriptedEngine, EventLog, SubagentHost]:
    engine = ScriptedEngine([Reply.coerce(r) for r in replies], n_ctx=32768)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp",
                  settings(permission_mode="yolo", **agent_fields), log, None,
                  tools=[SubagentTool(), *extra_tools])
    agent.set_effort("ultracode")
    queue = list(helpers)

    def factory(_host: SubagentHost, on_event: Any) -> TimedHelper:
        helper = queue.pop(0)
        helper.on_event = on_event
        return helper

    async def emit(event: Any) -> None:
        log(event)

    host = SubagentHost(runtime=agent.runtime, settings=agent.settings, cwd=agent.cwd,
                        model_key="test/model", emit=emit, agent_factory=factory)
    agent._tool_ctx.state[STATE_KEY] = host
    TimedHelper.live = TimedHelper.peak = 0
    return agent, engine, log, host


def slow_reader(delay: float) -> FakeTool:
    async def handler(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(True, f"treść {args.get('path')}", "40 linii")

    return FakeTool("read_file", handler=handler)


# --------------------------------------------------------------------------- #
# In the background
# --------------------------------------------------------------------------- #


async def test_a_background_helper_returns_at_once_and_reports_at_the_next_step() -> None:
    helper = TimedHelper(0.1, "mecz A: wygrana 3:1")
    agent, engine, log, _host = ultracode_agent(
        [tool_call_text("delegate", {"task": "uruchom mecz A"}),
         tool_call_text("read_file", {"path": "bot.py"}),
         "Gotowe."],
        [helper], extra_tools=[slow_reader(0.3)])
    await agent.run_turn("przetestuj bota")

    first = log.of("tool_result")[0]
    assert first.ok and first.summary.startswith("Pomocnik h1 pracuje w tle")
    assert "working in the background" in next(m for m in agent.history if m["role"] == "tool")["content"]
    # The agent kept working (read_file) and got the report before its next step.
    assert NOTE_HEAD not in engine.prompts[1]
    assert NOTE_HEAD in engine.prompts[2] and "mecz A: wygrana 3:1" in engine.prompts[2]
    roles = [m["role"] for m in agent.history]
    note_at = next(i for i, m in enumerate(agent.history)
                   if m["role"] == "user" and NOTE_HEAD in str(m["content"]))
    assert roles[note_at - 1] == "tool", "after the step's results, at the step boundary"
    end = log.first("subagent_end")
    assert end.background and end.ok and "wygrana 3:1" in end.report
    assert log.first("subagent_start").background and log.first("subagent_start").agent_id == "h1"
    assert log.last("turn_end").reason == "done"


async def test_a_turn_the_model_ends_waits_for_its_helpers() -> None:
    helper = TimedHelper(0.3, "log B: 12 błędów")
    agent, engine, log, host = ultracode_agent(
        [tool_call_text("delegate", {"task": "przeanalizuj log B"}),
         "Czekam na raport.",
         "Log B ma 12 błędów."],
        [helper])
    answer = await agent.run_turn("przeanalizuj logi")

    assert answer == "Log B ma 12 błędów.", "the turn went on with the report"
    assert len(engine.prompts) == 3 and NOTE_HEAD in engine.prompts[2]
    assert any(e.message.startswith("Czekam na pomocników w tle (1)") for e in log.of("status"))
    assert log.types.index("subagent_end") < log.types.index("turn_end")
    assert not host.has_background


async def test_interrupt_cancels_every_background_helper() -> None:
    helpers = [TimedHelper(None, "nigdy"), TimedHelper(None, "nigdy")]
    agent, engine, log, host = ultracode_agent(
        [tool_call_text("delegate", {"task": "mecz A"}) + "\n"
         + tool_call_text("delegate", {"task": "mecz B"}),
         "Czekam.", "nie dojdzie"],
        helpers)
    turn = asyncio.create_task(agent.run_turn("zagraj dwa mecze"))
    for _ in range(300):
        if any(e.message.startswith("Czekam na pomocników") for e in log.of("status")):
            break
        await asyncio.sleep(0.01)
    assert TimedHelper.live == 2
    agent.interrupt()
    await asyncio.wait_for(turn, timeout=2)

    assert log.last("turn_end").reason == "interrupted"
    assert TimedHelper.live == 0 and all(h.interrupted == 1 for h in helpers)
    ends = log.of("subagent_end")
    assert len(ends) == 2 and all(not e.ok and e.message == "przerwane" for e in ends)
    assert not host.running and not host.has_background


async def test_background_helpers_keep_the_parallel_cap() -> None:
    helpers = [TimedHelper(0.15, f"raport {n}") for n in range(3)]
    calls = "\n".join(tool_call_text("delegate", {"task": f"zadanie {n}"}) for n in range(3))
    agent, engine, log, _host = ultracode_agent(
        [calls, "Czekam.", "Czekam dalej.", "Czekam jeszcze.", "Koniec."], helpers,
        ultracode_parallel=2)
    await agent.run_turn("trzy zadania")

    assert TimedHelper.peak == 2
    final = engine.prompts[-1]
    for n in range(3):
        assert f"raport {n} (zadanie {n})" in final, "no report is ever dropped"
    assert [e.agent_id for e in log.of("subagent_start")] == ["h1", "h2", "h3"]


async def test_wait_true_still_waits_under_ultracode() -> None:
    helper = TimedHelper(0.05, "znalezione")
    agent, engine, log, _host = ultracode_agent(
        [tool_call_text("delegate", {"task": "znajdź X", "wait": True}), "Gotowe."], [helper])
    await agent.run_turn("szukaj")
    result = next(m for m in agent.history if m["role"] == "tool")["content"]
    assert result.startswith("znalezione (znajdź X)")
    assert not log.first("subagent_start").background


# --------------------------------------------------------------------------- #
# Helpers that run commands — through the parent's approvals, under their name
# --------------------------------------------------------------------------- #


def parent_with_real_helper(tmp_path: Path, replies: list[Any], mode: str, ask: Any = None,
                            ) -> tuple[Agent, ScriptedEngine, EventLog]:
    engine = ScriptedEngine([Reply.coerce(r) for r in replies], n_ctx=32768)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path,
                  settings(permission_mode=mode), log, ask, tools=[SubagentTool()])
    return agent, engine, log


def asker(answer: str = "deny"):
    asked: list[Any] = []

    async def ask(request: Any) -> str:
        asked.append(request)
        return answer

    return ask, asked


async def test_a_helpers_command_asks_the_user_under_its_name(tmp_path: Path) -> None:
    ask, asked = asker("deny")
    agent, engine, log = parent_with_real_helper(tmp_path, [
        tool_call_text("delegate", {"task": "utwórz znacznik", "wait": True}),
        tool_call_text("bash", {"command": "touch made-by-helper.txt"}),
        "Nie dostałem zgody.",
        "Pomocnik nie dostał zgody."], "default", ask)
    await agent.run_turn("zrób znacznik")

    [request] = asked
    assert request.title == "Pomocnik h1: Polecenie: touch made-by-helper.txt"
    assert request.helper == "h1" and request.to_dict()["helper"] == "h1"
    assert log.first("permission_request").helper == "h1", "the question reached the parent's client"
    assert not (tmp_path / "made-by-helper.txt").exists()
    step = log.first("subagent_step")
    assert step.name == "bash" and not step.ok


async def test_auto_mode_reviews_a_helpers_command_and_labels_the_verdict(tmp_path: Path) -> None:
    ask, asked = asker("deny")
    agent, engine, log = parent_with_real_helper(tmp_path, [
        tool_call_text("delegate", {"task": "utwórz znacznik", "wait": True}),
        tool_call_text("bash", {"command": "touch made-by-helper.txt"}),
        "NIE\nCo robi: Tworzy plik.\nPowód: Nie wiadomo po co.",
        "Odmówiono mi.",
        "Pomocnikowi odmówiono."], "auto", ask)
    await agent.run_turn("zrób znacznik")

    review = log.first("auto_review")
    assert review.helper == "h1" and review.verdict == "deny"
    assert review.guard is not None, "the local guard judged it too (here: its rules, no risk)"
    [request] = asked
    assert request.helper == "h1" and request.review["verdict"] == "deny"
    assert not (tmp_path / "made-by-helper.txt").exists()


async def test_yolo_lets_a_helpers_command_run(tmp_path: Path) -> None:
    agent, engine, log = parent_with_real_helper(tmp_path, [
        tool_call_text("delegate", {"task": "utwórz znacznik", "wait": True}),
        tool_call_text("bash", {"command": "touch made-by-helper.txt"}),
        "Utworzyłem made-by-helper.txt.",
        "Gotowe."], "yolo")
    await agent.run_turn("zrób znacznik")
    assert (tmp_path / "made-by-helper.txt").exists()
    assert log.of("permission_request") == []
    assert log.first("subagent_step").ok


async def test_plan_mode_gives_a_helper_no_commands(tmp_path: Path) -> None:
    assert "bash" not in [t.name for t in _helper_tools("plan", False)]
    ask, asked = asker("allow")
    agent, engine, log = parent_with_real_helper(tmp_path, [
        tool_call_text("delegate", {"task": "utwórz znacznik", "wait": True}),
        tool_call_text("bash", {"command": "touch made-by-helper.txt"}),
        "Nie mam poleceń.",
        "Plan gotowy."], "plan", ask)
    await agent.run_turn("zaplanuj")
    assert asked == [] and not (tmp_path / "made-by-helper.txt").exists()
    step = log.first("subagent_step")
    assert step.name == "bash" and not step.ok and step.summary == "Nieznane narzędzie"


def test_each_helper_has_its_own_name_and_cache_key() -> None:
    host = SubagentHost(runtime=None, settings=settings(), cwd="/tmp", model_key="m", emit=None)
    host.bind(ask_permission=None, policy=None, mode="yolo", cache_key="localcode-abc")
    first, second = host.next_label(), host.next_label()
    assert (first, second) == ("h1", "h2")
    helper = host.build_agent(on_event=None, label=second)
    assert helper._cache_key == "localcode-abc-h2"
    assert helper._helper == "h2" and helper.mode == "yolo"


# --------------------------------------------------------------------------- #
# Ultracode pushes towards helpers
# --------------------------------------------------------------------------- #


async def test_the_first_ultracode_step_asks_for_a_fan_out() -> None:
    agent, engine, _log, _host = ultracode_agent(["Gotowe."], [])
    await agent.run_turn("popraw bota")
    first = engine.prompts[0]
    assert "Fan out FIRST" in first and "background helper" in first
    assert "wait=false" in first


async def test_a_quiet_ultracode_agent_is_reminded_once_in_a_while() -> None:
    reads = [tool_call_text("read_file", {"path": f"f{n}.py"}) for n in range(9)]
    agent, engine, _log, _host = ultracode_agent([*reads, "Gotowe."], [],
                                                 extra_tools=[FakeTool("read_file")])
    await agent.run_turn("przejrzyj pliki")
    marker = "steps without helpers"
    assert marker not in engine.prompts[3], "four quiet steps first"
    assert marker in engine.prompts[4]
    assert engine.prompts[7].count(marker) == 1, "not again before four more"
    assert engine.prompts[8].count(marker) == 2
    assert prompts.ULTRACODE_NUDGE.startswith("[Ultracode")


async def test_no_reminder_without_ultracode_or_after_delegating() -> None:
    reads = [tool_call_text("read_file", {"path": f"f{n}.py"}) for n in range(6)]
    agent, engine, _log, _host = ultracode_agent([*reads, "Gotowe."], [],
                                                 extra_tools=[FakeTool("read_file")])
    agent.set_effort("high")
    await agent.run_turn("przejrzyj pliki")
    assert all("steps without helpers" not in p for p in engine.prompts)

    helper = TimedHelper(0.01, "ok")
    steps = [tool_call_text("read_file", {"path": f"g{n}.py"}) for n in range(3)]
    steps.insert(2, tool_call_text("delegate", {"task": "zbadaj moduł"}))
    agent, engine, _log, _host = ultracode_agent([*steps, "Gotowe."], [helper],
                                                 extra_tools=[FakeTool("read_file")])
    await agent.run_turn("przejrzyj")
    assert all("steps without helpers" not in p for p in engine.prompts)
