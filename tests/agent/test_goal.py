"""``/goal``: the agent keeps working until it declares the goal reached.

All against ``ScriptedEngine`` — every model step is a scripted reply.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, FakeTool, Reply, ScriptedEngine, tool_call_text
from localcode.agent.context import structure_problems
from localcode.agent.loop import Agent
from localcode.config import Settings

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def build(replies, *, tools=(), **agent_fields: Any):
    s = Settings()
    s.agent = replace(s.agent, permission_mode="yolo", **agent_fields)
    engine = ScriptedEngine(replies, n_ctx=8192)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", s, log, None, tools=list(tools))
    return agent, engine, log


def goal_states(log: EventLog) -> list[tuple[str, int]]:
    return [(e.state, e.turns) for e in log.of("goal")]


async def test_a_final_answer_without_goal_done_sends_the_agent_back_to_work() -> None:
    tool = FakeTool("bash", output="2 passed")
    agent, engine, log = build([
        "Dodałem funkcję.",                                           # stops too early
        tool_call_text("bash", {"command": "pytest"}),                # back to work
        tool_call_text("goal_done", {"summary": "testy przechodzą"}),
        "Gotowe: funkcja dodana, testy przechodzą.",
    ], tools=[tool])
    await agent.set_goal("dodaj funkcję i niech testy przechodzą")
    answer = await agent.run_turn("dodaj funkcję i niech testy przechodzą")

    assert answer == "Gotowe: funkcja dodana, testy przechodzą."
    # One turn from the outside: every client renders it as it always has.
    assert log.types.count("turn_start") == 1 and log.types.count("turn_end") == 1
    assert log.last("turn_end").reason == "done"
    assert goal_states(log) == [("active", 0), ("active", 1), ("done", 1)]
    assert tool.calls == [{"command": "pytest"}]
    nudges = [m for m in agent.messages if m["role"] == "user" and "[automatic]" in str(m["content"])]
    assert len(nudges) == 1 and "niech testy przechodzą" in nudges[0]["content"]
    assert structure_problems(agent.messages) == []
    assert agent.goal["state"] == "done"


async def test_the_goal_is_in_the_system_prompt_and_goal_done_is_offered() -> None:
    agent, engine, log = build([tool_call_text("goal_done", {"summary": "ok"}), "Zrobione."])
    await agent.set_goal("napisz README")
    await agent.run_turn("napisz README")
    first = engine.prompts[0]
    assert "ACTIVE GOAL" in first and "napisz README" in first
    assert "goal_done" in first
    # Once reached, the rule and the tool leave the prompt again.
    assert "ACTIVE GOAL" not in agent.messages[0]["content"]


async def test_without_a_goal_nothing_changes() -> None:
    agent, engine, log = build(["Cześć!"])
    await agent.run_turn("hej")
    assert log.of("goal") == []
    assert "goal_done" not in engine.prompts[0]
    assert "ACTIVE GOAL" not in agent.messages[0]["content"]


async def test_blocked_ends_the_loop_and_asks_the_user() -> None:
    agent, engine, log = build([
        tool_call_text("goal_done", {"summary": "potrzebuję klucza API", "blocked": True}),
        "Potrzebuję klucza API do serwisu.",
    ])
    await agent.set_goal("podłącz serwis")
    await agent.run_turn("podłącz serwis")
    assert goal_states(log)[-1] == ("blocked", 0)
    assert log.types.count("turn_end") == 1


async def test_the_continuations_are_limited() -> None:
    agent, engine, log = build([f"próba {i}: jeszcze nie" for i in range(10)], goal_max_turns=3)
    await agent.set_goal("niemożliwe")
    await agent.run_turn("niemożliwe")
    assert goal_states(log)[-1] == ("limit", 3)
    assert len(engine.prompts) == 4          # the first try and three continuations


async def test_a_waiting_user_message_takes_the_floor() -> None:
    agent, engine, log = build(["pierwsza odpowiedź", "druga"])
    agent.goal_should_yield = lambda: True
    await agent.set_goal("coś dużego")
    await agent.run_turn("coś dużego")
    assert len(engine.prompts) == 1
    assert agent.goal["state"] == "active"     # still the goal; the user just spoke first


async def test_running_out_of_steps_continues_instead_of_stopping() -> None:
    tool = FakeTool("read_file", output="x")
    calls = [tool_call_text("read_file", {"path": f"{i}.txt"}) for i in range(4)]
    agent, engine, log = build([*calls, tool_call_text("goal_done", {"summary": "ok"}), "Koniec."],
                               tools=[tool], max_steps=2)
    await agent.set_goal("przeczytaj wszystko")
    await agent.run_turn("przeczytaj wszystko")
    assert log.last("turn_end").reason == "done"
    assert len(tool.calls) == 4
    assert agent.goal["state"] == "done"


async def test_clearing_the_goal_stops_offering_the_tool() -> None:
    agent, engine, log = build(["a", "b"])
    await agent.set_goal("cel")
    await agent.clear_goal()
    assert agent.goal is None
    assert goal_states(log)[-1][0] == "cleared"
    await agent.run_turn("hej")
    assert "goal_done" not in engine.prompts[-1]
    assert len(engine.prompts) == 1


async def test_goal_done_without_a_goal_is_refused_politely() -> None:
    from localcode.agent.tools.base import ToolContext
    from localcode.agent.tools.goal import GoalDoneTool

    result = await GoalDoneTool().execute({"summary": "x"}, ToolContext(cwd="/tmp", state={}))
    assert not result.ok


async def test_a_goal_done_written_as_text_still_ends_the_goal() -> None:
    """Regression (real Qwen3-4B run): the model *wrote* ``goal_done{…}`` into its
    answer instead of calling the tool, and the loop sent it back twenty times."""
    agent, engine, log = build([
        'Plik hello.py działa.\n\ngoal_done{"summary": "hello.py wypisuje Cześć"}',
    ])
    await agent.set_goal("hello.py")
    await agent.run_turn("hello.py")
    assert len(engine.prompts) == 1
    assert agent.goal["state"] == "done"
    assert agent.goal["summary"] == "hello.py wypisuje Cześć"


async def test_a_blocked_call_written_as_text_counts_too() -> None:
    agent, engine, log = build(['Nie mam adresu.\ngoal_done(summary="brak adresu", blocked=true)'])
    await agent.set_goal("wyślij")
    await agent.run_turn("wyślij")
    assert agent.goal["state"] == "blocked"


async def test_merely_mentioning_the_tool_does_not_end_the_goal() -> None:
    agent, engine, log = build(["Wywołam goal_done, gdy testy przejdą.",
                                tool_call_text("goal_done", {"summary": "ok"}), "Gotowe."])
    await agent.set_goal("testy")
    await agent.run_turn("testy")
    assert len(engine.prompts) == 3
    assert agent.goal["state"] == "done"


async def test_the_same_answer_again_and_again_stops_the_loop() -> None:
    agent, engine, log = build(["Zrobione, wszystko działa."] * 10)
    await agent.set_goal("cel")
    await agent.run_turn("cel")
    assert agent.goal["state"] == "stalled"
    assert len(engine.prompts) <= 4            # not twenty
    assert log.last("goal").state == "stalled"

