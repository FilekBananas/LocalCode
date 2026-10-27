"""Tests for localcode.agent.loop (the coding-agent turn loop).

Everything runs against ``ScriptedEngine``/``FakeRuntime`` (``agent_fakes.py``):
no model is loaded, so the whole file is fast and deterministic.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import (
    EventLog,
    FakeRuntime,
    FakeTool,
    Reply,
    ScriptedEngine,
    tool_call_text,
)
from localcode.agent.context import structure_problems
from localcode.agent.loop import INTERRUPTED_RESULT, Agent, AgentBusyError, dict_events
from localcode.agent.prompts import EMPTY_RESPONSE_NUDGE, REPEATED_CALL_NUDGE
from localcode.agent.session import SessionStore
from localcode.agent.tools.base import PermissionInfo, ToolResult
from localcode.config import Settings
from localcode.runtime.base import EngineError

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def settings(**agent_fields: Any) -> Settings:
    s = Settings()
    s.agent = replace(s.agent, **agent_fields) if agent_fields else s.agent
    return s


def build(
    replies,
    *,
    tools=(),
    mode: str = "default",
    n_ctx: int = 8192,
    ask=None,
    session: SessionStore | None = None,
    engine: ScriptedEngine | None = None,
    **agent_fields: Any,
):
    engine = engine or ScriptedEngine(replies, n_ctx=n_ctx)
    runtime = FakeRuntime(engine)
    log = EventLog()
    agent = Agent(
        runtime, "test/model", "/tmp", settings(permission_mode=mode, **agent_fields),
        log, ask, tools=list(tools), session=session,
    )
    return agent, engine, runtime, log


async def wait_for(predicate, timeout: float = 2.0) -> None:
    """Wait until ``predicate()`` is true (the tests never sleep blindly)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met within the timeout")
        await asyncio.sleep(0.005)


def roles(agent: Agent) -> list[str]:
    return [m["role"] for m in agent.messages]


# --------------------------------------------------------------------------- #
# Plain turns
# --------------------------------------------------------------------------- #


async def test_plain_answer_streams_text_and_ends() -> None:
    agent, engine, runtime, log = build(["Cześć! Wszystko działa."])
    answer = await agent.run_turn("hej")

    assert answer == "Cześć! Wszystko działa."
    assert log.text == "Cześć! Wszystko działa."
    assert log.types[0] == "turn_start"
    assert log.types[-1] == "turn_end"
    assert log.last("turn_end").reason == "done"
    assert roles(agent) == ["system", "user", "assistant"]
    assert agent.messages[-1]["content"] == "Cześć! Wszystko działa."
    assert runtime.held == 0 and runtime.max_held == 1
    assert not agent.busy


async def test_turn_end_carries_stats() -> None:
    agent, engine, runtime, log = build([Reply(text="ok", completion_tokens=42)])
    await agent.run_turn("hej")

    stats = log.last("turn_end").stats
    assert stats["steps"] == 1
    assert stats["completionTokens"] == 42
    assert stats["elapsedMs"] >= 0
    assert stats["genTps"] > 0
    assert runtime.stats and runtime.stats[0].completion_tokens == 42


async def test_loading_status_then_thinking() -> None:
    agent, engine, runtime, log = build(["gotowe"])
    await agent.run_turn("hej")

    messages = [e.message for e in log.of("status")]
    assert messages[:2] == ["Ładowanie modelu…", "Myślę…"]
    assert log.of("status")[0].phase == "loading"

    log.events.clear()
    engine.replies.append(Reply(text="drugie"))
    await agent.run_turn("i jeszcze raz")
    assert not log.of("status")  # the model stays loaded


async def test_context_event_reports_the_window() -> None:
    agent, _engine, _runtime, log = build(["ok"], n_ctx=4096)
    await agent.run_turn("hej")

    update = log.first("context")
    assert update.limit == 4096
    assert update.used > 0
    assert update.system > 0
    assert update.tools == 0  # no tools registered in this test
    assert agent.context_used == update.used
    assert agent.context_limit == 4096


async def test_dict_events_adapter() -> None:
    seen: list[dict[str, Any]] = []
    engine = ScriptedEngine(["ok"])
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(),
                  dict_events(seen.append), None, tools=[])
    await agent.run_turn("hej")
    assert seen[0]["type"] == "turn_start"
    assert seen[0]["turnId"]
    assert seen[-1]["type"] == "turn_end"


# --------------------------------------------------------------------------- #
# Tool calls
# --------------------------------------------------------------------------- #


async def test_tool_call_result_then_answer() -> None:
    tool = FakeTool("read_file", output="line1\nline2", summary="2 linie")
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}), "Plik ma 2 linie."], tools=[tool]
    )
    answer = await agent.run_turn("przeczytaj a.txt")

    assert answer == "Plik ma 2 linie."
    assert tool.calls == [{"path": "a.txt"}]
    assert roles(agent) == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[3]["content"] == "line1\nline2"
    assert agent.messages[3]["tool_call_id"] == agent.messages[2]["tool_calls"][0]["id"]
    assert structure_problems(agent.messages) == []

    call = log.first("tool_call")
    result = log.first("tool_result")
    assert call.name == "read_file" and call.arguments == {"path": "a.txt"}
    assert call.title == "read_file a.txt"
    assert result.ok and result.summary == "2 linie"
    assert log.types.index("tool_call") < log.types.index("tool_result")


async def test_lease_is_released_while_the_tool_runs() -> None:
    held: list[int] = []
    runtime_box: dict[str, FakeRuntime] = {}

    def handler(args, ctx):
        held.append(runtime_box["rt"].held)
        return "done"

    tool = FakeTool("bash", kind="read", handler=handler)
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "ls"}), "Gotowe."], tools=[tool]
    )
    runtime_box["rt"] = runtime
    await agent.run_turn("pokaż pliki")

    assert held == [0], "the runtime slot must be free while a tool runs"
    assert runtime.max_held == 1
    assert len(runtime.acquires) == 2 and set(runtime.acquires) == {"agent"}


async def test_tool_output_is_truncated_for_the_model_but_not_for_the_ui() -> None:
    tool = FakeTool("read_file", output="x" * 40_000, summary="dużo")
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "big.txt"}), "Duży plik."], tools=[tool], n_ctx=4096
    )
    await agent.run_turn("przeczytaj")

    model_text = agent.messages[3]["content"]
    assert len(model_text) < 40_000
    assert "omitted" in model_text
    assert len(log.first("tool_result").output) == 40_000


async def test_empty_tool_output_is_labelled_for_the_model() -> None:
    tool = FakeTool("bash", kind="read", output="", summary="exit 0")
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "true"}), "Nic nie wypisało."], tools=[tool]
    )
    await agent.run_turn("uruchom")
    assert agent.messages[3]["content"] == "(the tool produced no output)"


async def test_unknown_tool_gets_a_corrective_result() -> None:
    tool = FakeTool("read_file")
    agent, engine, runtime, log = build(
        [tool_call_text("nonexistent_tool", {"x": 1}), "Przepraszam, użyję read_file."], tools=[tool]
    )
    await agent.run_turn("zrób coś")

    result = agent.messages[3]["content"]
    assert "no tool named" in result and "read_file" in result
    assert log.first("tool_result").ok is False


async def test_todo_update_event() -> None:
    def handler(args, ctx):
        ctx.state["todos"] = args["todos"]
        return "zapisano"

    tool = FakeTool("todo_write", kind="meta", handler=handler)
    todos = [{"content": "napraw test", "status": "in_progress"}, {"content": "uruchom pytest", "status": "pending"}]
    agent, engine, runtime, log = build(
        [tool_call_text("todo_write", {"todos": todos}), "Lista zaktualizowana."], tools=[tool]
    )
    await agent.run_turn("zrób plan")

    update = log.first("todos")
    assert update.items == todos


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #


def exec_tool() -> FakeTool:
    return FakeTool("bash", kind="exec", output="exit 0",
                    permission_info=PermissionInfo(title="Uruchom: npm test", preview="npm test"))


async def test_permission_allow() -> None:
    asked: list[Any] = []

    async def ask(request):
        asked.append(request)
        return "allow"

    tool = exec_tool()
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), "Testy przeszły."], tools=[tool], ask=ask
    )
    await agent.run_turn("uruchom testy")

    assert len(asked) == 1
    assert asked[0].tool == "bash" and asked[0].preview == "npm test"
    assert asked[0].options == ["allow", "deny", "always"]
    assert tool.calls == [{"command": "npm test"}]
    assert log.first("permission_request").title == "Uruchom: npm test"


async def test_permission_deny_tells_the_model_not_to_retry() -> None:
    async def ask(request):
        return "deny"

    tool = exec_tool()
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), "Dobrze, nie uruchamiam."], tools=[tool], ask=ask
    )
    await agent.run_turn("uruchom testy")

    assert tool.calls == []
    assert "Do not retry" in agent.messages[3]["content"]
    assert log.first("tool_result").ok is False


async def test_permission_always_is_remembered_for_the_next_call() -> None:
    asked: list[Any] = []

    async def ask(request):
        asked.append(request)
        return "always"

    tool = exec_tool()
    agent, engine, runtime, log = build(
        [
            tool_call_text("bash", {"command": "npm test"}),
            tool_call_text("bash", {"command": "npm test -- --watch=false"}),
            "Wszystko zielone.",
        ],
        tools=[tool], ask=ask,
    )
    await agent.run_turn("uruchom testy dwa razy")

    assert len(asked) == 1, "the second npm command must not ask again"
    assert len(tool.calls) == 2
    assert agent.policy.is_remembered(tool, {"command": "npm test -- --watch=false"})


async def test_permission_without_an_approver_is_refused() -> None:
    tool = exec_tool()
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), "Nie mogę tego uruchomić."], tools=[tool], ask=None
    )
    await agent.run_turn("uruchom testy")

    assert tool.calls == []
    assert "approval" in agent.messages[3]["content"]


async def test_read_tools_never_ask() -> None:
    asked: list[Any] = []

    async def ask(request):
        asked.append(request)
        return "allow"

    tool = FakeTool("read_file", kind="read")
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}), "Przeczytane."], tools=[tool], ask=ask
    )
    await agent.run_turn("czytaj")
    assert asked == [] and len(tool.calls) == 1


async def test_plan_mode_blocks_writes() -> None:
    async def ask(request):  # pragma: no cover - must never be called
        raise AssertionError("plan mode must not ask, it refuses")

    write = FakeTool("write_file", kind="write",
                     permission_info=PermissionInfo(title="Zapisz a.txt", preview="+ nowa treść"))
    read = FakeTool("read_file", kind="read", output="zawartość")
    agent, engine, runtime, log = build(
        [
            tool_call_text("write_file", {"path": "a.txt", "content": "x"}),
            tool_call_text("read_file", {"path": "a.txt"}),
            "Plan: 1. zmienić a.txt",
        ],
        tools=[write, read], mode="plan", ask=ask,
    )
    await agent.run_turn("zaplanuj zmianę")

    assert write.calls == [], "plan mode is read-only"
    assert "plan mode is read-only" in agent.messages[3]["content"]
    assert len(read.calls) == 1, "read tools still work in plan mode"
    assert log.of("permission_request") == []


async def test_yolo_mode_runs_everything() -> None:
    tool = exec_tool()
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), "Zrobione."], tools=[tool], mode="yolo"
    )
    await agent.run_turn("uruchom")
    assert len(tool.calls) == 1 and log.of("permission_request") == []


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


async def test_repeated_call_guard_stops_the_third_identical_call() -> None:
    tool = FakeTool("read_file", output="same output")
    call = tool_call_text("read_file", {"path": "a.txt"})
    agent, engine, runtime, log = build([call, call, call, "Już to mam."], tools=[tool])
    await agent.run_turn("czytaj w kółko")

    assert len(tool.calls) == 2, "the third identical call must not run"
    nudges = [m for m in agent.messages if m["role"] == "tool" and m["content"] == REPEATED_CALL_NUDGE]
    assert len(nudges) == 1
    assert log.of("tool_result")[-1].summary == "Powtórzone wywołanie"


async def test_different_arguments_reset_the_repeat_guard() -> None:
    tool = FakeTool("read_file", output="x")
    agent, engine, runtime, log = build(
        [
            tool_call_text("read_file", {"path": "a.txt"}),
            tool_call_text("read_file", {"path": "a.txt"}),
            tool_call_text("read_file", {"path": "b.txt"}),
            tool_call_text("read_file", {"path": "b.txt"}),
            "Gotowe.",
        ],
        tools=[tool],
    )
    await agent.run_turn("czytaj")
    assert len(tool.calls) == 4


async def test_malformed_call_is_corrected_and_recovered() -> None:
    tool = FakeTool("read_file", output="treść")
    agent, engine, runtime, log = build(
        [
            "<tool_call>\n{name: read_file, arguments: [oops}\n</tool_call>",
            tool_call_text("read_file", {"path": "a.txt"}),
            "Gotowe.",
        ],
        tools=[tool],
    )
    answer = await agent.run_turn("czytaj a.txt")

    assert answer == "Gotowe."
    nudges = [m for m in agent.messages if m["role"] == "user" and "could not be parsed" in str(m.get("content"))]
    assert len(nudges) == 1
    assert "<tool_call>" in nudges[0]["content"]
    assert len(tool.calls) == 1


async def test_malformed_call_gives_up_after_two_retries() -> None:
    bad = "<tool_call>\n{name: read_file, arguments: [oops}\n</tool_call>"
    agent, engine, runtime, log = build([bad, bad, bad], tools=[FakeTool("read_file")])
    await agent.run_turn("czytaj")

    assert log.last("turn_end").reason == "error"
    assert log.of("error")
    assert engine.prompts and len(engine.prompts) == 3


async def test_empty_response_is_nudged_once() -> None:
    agent, engine, runtime, log = build(["", "Już odpowiadam."], tools=[])
    answer = await agent.run_turn("hej")

    assert answer == "Już odpowiadam."
    assert any(EMPTY_RESPONSE_NUDGE in str(m.get("content")) for m in agent.messages if m["role"] == "user")
    assert log.last("turn_end").reason == "done"


async def test_two_empty_responses_end_the_turn_with_an_error() -> None:
    agent, engine, runtime, log = build(["", ""], tools=[])
    await agent.run_turn("hej")

    assert log.last("turn_end").reason == "error"
    assert "pustą odpowiedź" in log.last("error").message


async def test_max_steps_stops_the_loop() -> None:
    tool = FakeTool("read_file", output="x")
    calls = [tool_call_text("read_file", {"path": f"{i}.txt"}) for i in range(10)]
    agent, engine, runtime, log = build(calls, tools=[tool], max_steps=3)
    await agent.run_turn("czytaj wszystko")

    assert log.last("turn_end").reason == "max_steps"
    assert log.last("turn_end").stats["steps"] == 3
    assert len(tool.calls) == 3


async def test_max_steps_zero_means_no_limit() -> None:
    tool = FakeTool("read_file", output="x")
    calls = [tool_call_text("read_file", {"path": f"{i}.txt"}) for i in range(10)] + ["Gotowe."]
    agent, engine, runtime, log = build(calls, tools=[tool], max_steps=0)
    await agent.run_turn("czytaj wszystko")

    assert log.last("turn_end").reason == "done"
    assert len(tool.calls) == 10


async def test_engine_error_ends_the_turn() -> None:
    agent, engine, runtime, log = build([EngineError("Silnik padł.")], tools=[])
    await agent.run_turn("hej")

    assert log.last("turn_end").reason == "error"
    assert log.last("error").message == "Silnik padł."


async def test_busy_agent_refuses_a_second_turn() -> None:
    agent, engine, runtime, log = build([Reply(text="pracuję", stall_after=0)], tools=[])
    task = asyncio.create_task(agent.run_turn("pierwsze"))
    await wait_for(lambda: agent.busy)
    with pytest.raises(AgentBusyError):
        await agent.run_turn("drugie")
    agent.interrupt()
    await task


# --------------------------------------------------------------------------- #
# Context overflow and compaction
# --------------------------------------------------------------------------- #


def overflow_error() -> BaseException:
    from localcode.runtime.http_engine import ContextOverflowError

    return ContextOverflowError("Zapytanie przekracza dostępny kontekst modelu.",
                                n_prompt_tokens=9000, n_ctx=8192)


async def test_overflow_is_recovered_once() -> None:
    tool = FakeTool("read_file", output="dane " * 200)
    agent, engine, runtime, log = build(
        [
            tool_call_text("read_file", {"path": "a.txt"}),
            tool_call_text("read_file", {"path": "b.txt"}),
            overflow_error(),
            "Po kompaktowaniu wszystko działa.",
        ],
        tools=[tool], n_ctx=2048,
    )
    answer = await agent.run_turn("przeczytaj i podsumuj")

    assert answer == "Po kompaktowaniu wszystko działa."
    assert log.last("turn_end").reason == "done"
    assert log.of("compaction"), "on_overflow must report what it did"
    assert structure_problems(agent.messages) == []


async def test_overflow_twice_reports_a_polish_error() -> None:
    agent, engine, runtime, log = build([overflow_error(), overflow_error()], tools=[], n_ctx=2048)
    await agent.run_turn("hej")

    assert log.last("turn_end").reason == "error"
    assert "kontekst" in log.last("error").message.lower()


async def test_compaction_with_a_tiny_context_window() -> None:
    tool = FakeTool("read_file", output="\n".join(f"linia {i} z bardzo długiego pliku" for i in range(200)))
    script: list[Any] = []
    for i in range(6):
        script.append(tool_call_text("read_file", {"path": f"plik{i}.py"}))
    script.append("Przeczytałem wszystkie pliki.")
    agent, engine, runtime, log = build(script, tools=[tool], n_ctx=1024)

    await agent.run_turn("przeczytaj sześć plików i podsumuj")

    assert log.last("turn_end").reason == "done"
    compactions = log.of("compaction")
    assert compactions, "a 1k window must trigger compaction"
    assert {c.strategy for c in compactions} <= {
        "mask_tool_results", "shrink_messages", "summarize", "drop_oldest", "truncate"
    }
    assert all(c.after < c.before for c in compactions)
    assert structure_problems(agent.messages) == []
    # the summary was produced by the model through the same engine
    assert engine.summary_prompts or any(c.strategy != "summarize" for c in compactions)


async def test_compact_command_summarizes_on_demand() -> None:
    tool = FakeTool("read_file", output="x" * 2000)
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}), "Przeczytane."], tools=[tool], n_ctx=4096
    )
    await agent.run_turn("czytaj")
    before = len(agent.messages)
    log.events.clear()

    await agent.compact()

    assert engine.summary_prompts, "compact() must ask the model for a summary"
    assert len(agent.messages) <= before
    assert any("[Summary of earlier work]" in str(m.get("content")) for m in agent.messages)
    assert log.of("compaction")
    assert structure_problems(agent.messages) == []


# --------------------------------------------------------------------------- #
# Interrupts
# --------------------------------------------------------------------------- #


async def test_interrupt_during_generation_keeps_partial_text() -> None:
    agent, engine, runtime, log = build(
        [Reply(text="Zaczynam analizować projekt", stall_after=1, chunk_delay=0.001)], tools=[]
    )
    task = asyncio.create_task(agent.run_turn("analizuj"))
    await wait_for(lambda: bool(log.text))
    agent.interrupt()
    await asyncio.wait_for(task, timeout=2)

    end = log.last("turn_end")
    assert end.reason == "interrupted"
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"].startswith("Zaczynam")
    assert runtime.held == 0, "the lease must be released on interrupt"
    assert not agent.busy
    assert structure_problems(agent.messages) == []


async def test_interrupt_during_a_slow_tool_closes_the_call() -> None:
    async def slow(args, ctx):
        await asyncio.sleep(30)
        return "za późno"  # pragma: no cover

    tool = FakeTool("bash", kind="read", handler=slow)
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "pytest -q"}), "Nigdy tu nie dojdziemy."], tools=[tool]
    )
    task = asyncio.create_task(agent.run_turn("uruchom testy"))
    await wait_for(lambda: tool.started.is_set())
    agent.interrupt()
    await asyncio.wait_for(task, timeout=2)

    assert tool.cancelled == 1
    assert log.last("turn_end").reason == "interrupted"
    assert agent.messages[-1]["role"] == "tool"
    assert agent.messages[-1]["content"] == INTERRUPTED_RESULT
    assert structure_problems(agent.messages) == []
    assert runtime.held == 0


async def test_interrupt_leaves_a_resumable_history() -> None:
    tool = FakeTool("bash", kind="read", handler=lambda a, c: asyncio.sleep(30))
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "sleep 30"}), "druga tura"], tools=[tool]
    )
    task = asyncio.create_task(agent.run_turn("uruchom"))
    await wait_for(lambda: tool.started.is_set())
    agent.interrupt()
    await asyncio.wait_for(task, timeout=2)

    engine.replies.clear()
    engine.replies.append(Reply(text="Kontynuuję."))
    answer = await agent.run_turn("kontynuuj")
    assert answer == "Kontynuuję."
    assert structure_problems(agent.messages) == []


async def test_interrupt_before_any_token_ends_cleanly() -> None:
    agent, engine, runtime, log = build([Reply(text="cokolwiek", stall_after=0)], tools=[])
    task = asyncio.create_task(agent.run_turn("hej"))
    await wait_for(lambda: agent.busy and bool(engine.prompts))
    agent.interrupt()
    await asyncio.wait_for(task, timeout=2)

    assert log.last("turn_end").reason == "interrupted"
    assert roles(agent) == ["system", "user"]


# --------------------------------------------------------------------------- #
# Session state, modes and models
# --------------------------------------------------------------------------- #


async def test_system_prompt_is_byte_identical_across_steps() -> None:
    tool = FakeTool("read_file", output="x")
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}),
         tool_call_text("read_file", {"path": "b.txt"}),
         "Gotowe."],
        tools=[tool],
    )
    await agent.run_turn("czytaj dwa pliki")

    heads = {p.split("<|user|>")[0] for p in engine.prompts}
    assert len(heads) == 1, "the prompt prefix must not change between steps"
    assert all(p.startswith(engine.prompts[0][:200]) for p in engine.prompts)


async def test_set_mode_rebuilds_the_prompt_and_notifies_the_model() -> None:
    write = FakeTool("write_file", kind="write")
    agent, engine, runtime, log = build(["ok", "ok2"], tools=[write])
    await agent.run_turn("pierwsze")
    assert agent.mode == "default"

    agent.set_mode("plan")
    assert agent.mode == "plan" and agent.policy.mode == "plan"
    await agent.run_turn("drugie")

    user_messages = [m for m in agent.messages if m["role"] == "user"]
    assert "[Mode changed to plan mode" in user_messages[-1]["content"]
    assert "drugie" in user_messages[-1]["content"]
    assert "PLAN MODE" in engine.prompts[-1] or "Plan mode" in engine.prompts[-1]

    with pytest.raises(ValueError):
        agent.set_mode("nonsense")


async def test_set_model_resets_the_prompt_state() -> None:
    agent, engine, runtime, log = build(["ok"], tools=[])
    await agent.run_turn("hej")
    agent.set_model("other/model")
    assert agent.model_key == "other/model"
    assert agent.session_dto()["modelKey"] == "other/model"


async def test_clear_forgets_the_conversation() -> None:
    agent, engine, runtime, log = build(["pierwsza", "druga"], tools=[])
    await agent.run_turn("hej")
    agent.clear()
    assert agent.history == []
    await agent.run_turn("nowy temat")
    assert roles(agent) == ["system", "user", "assistant"]


async def test_session_records_messages_and_items(isolated_home, tmp_path) -> None:
    store = SessionStore.create(tmp_path, model_key="test/model", mode="default")
    tool = FakeTool("read_file", output="treść pliku", summary="1 linia", diff=None)
    agent, engine, runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}), "Plik przeczytany."],
        tools=[tool], session=store,
    )
    await agent.run_turn("przeczytaj a.txt")

    assert store.session.title == "przeczytaj a.txt"
    assert [m["role"] for m in store.messages] == ["user", "assistant", "tool", "assistant"]
    kinds = [i["kind"] for i in store.items]
    assert kinds == ["user", "tool", "assistant"]
    tool_item = store.items[1]
    assert tool_item["toolName"] == "read_file"
    assert tool_item["output"] == "treść pliku"
    assert tool_item["ok"] is True

    kinds_on_disk = [json.loads(line)["kind"] for line in
                     store.path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "replace_history" not in kinds_on_disk, "an append-only turn must not rewrite the history"

    reloaded = SessionStore.resume(store.session.id)
    assert reloaded.messages == store.messages
    assert reloaded.items == store.items


async def test_session_records_a_history_rewrite_on_compaction(isolated_home, tmp_path) -> None:
    store = SessionStore.create(tmp_path, model_key="test/model")
    tool = FakeTool("read_file", output="\n".join(f"linia {i}" for i in range(300)))
    script = [tool_call_text("read_file", {"path": f"p{i}.py"}) for i in range(5)] + ["Gotowe."]
    agent, engine, runtime, log = build(script, tools=[tool], n_ctx=1024, session=store)
    await agent.run_turn("przeczytaj pliki")

    raw = store.path.read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(line)["kind"] for line in raw]
    assert kinds[0] == "session"
    assert "replace_history" in kinds
    resumed = SessionStore.resume(store.session.id)
    assert resumed.messages == agent.history
    assert structure_problems([{"role": "system", "content": "s"}] + resumed.messages) == []


async def test_resumed_session_continues_the_conversation(isolated_home, tmp_path) -> None:
    store = SessionStore.create(tmp_path, model_key="test/model")
    agent, engine, runtime, log = build(["Pierwsza odpowiedź."], tools=[], session=store)
    await agent.run_turn("pierwsze pytanie")

    resumed_store = SessionStore.resume(store.session.id)
    agent2, engine2, runtime2, log2 = build(["Druga odpowiedź."], tools=[], session=resumed_store)
    answer = await agent2.run_turn("drugie pytanie")

    assert answer == "Druga odpowiedź."
    assert [m["role"] for m in agent2.history] == ["user", "assistant", "user", "assistant"]
    assert "pierwsze pytanie" in engine2.prompts[0]
    assert structure_problems(agent2.messages) == []


async def test_session_dto_shape(isolated_home, tmp_path) -> None:
    store = SessionStore.create(tmp_path, model_key="test/model", mode="yolo")
    agent, engine, runtime, log = build(["ok"], tools=[], session=store, mode="yolo")
    await agent.run_turn("hej")

    dto = agent.session_dto()
    assert set(dto) == {"id", "cwd", "modelKey", "mode", "title", "createdAt", "updatedAt",
                        "contextUsed", "contextLimit", "busy"}
    assert dto["mode"] == "yolo" and dto["busy"] is False
    assert dto["contextLimit"] == 8192 and dto["contextUsed"] > 0
    assert dto["title"] == "hej"


async def test_real_tool_registry_reads_and_edits_a_file(tmp_path) -> None:
    """The loop wired to the real tools (no model): read → edit → answer."""
    source = tmp_path / "calc.py"
    source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    engine = ScriptedEngine(
        [
            tool_call_text("read_file", {"path": "calc.py"}),
            tool_call_text("edit_file", {"path": "calc.py", "old_string": "return a - b",
                                         "new_string": "return a + b"}),
            "Poprawiłem dodawanie w calc.py:2.",
        ],
        n_ctx=8192,
    )
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path,
                  settings(permission_mode="yolo"), log, None)

    answer = await agent.run_turn("napraw add w calc.py")

    assert answer.startswith("Poprawiłem")
    assert source.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"
    # 8k of context is the "tiny" tier: no glob, no todo_write (see TINY_OMIT)
    assert [t.name for t in agent.tools] == [
        "read_file", "write_file", "edit_file", "list_dir", "grep", "bash"]
    diffs = [e.diff for e in log.of("tool_result") if e.diff]
    assert diffs and "+    return a + b" in diffs[0]
    assert structure_problems(agent.messages) == []


async def test_real_todo_tool_emits_a_todo_update(tmp_path) -> None:
    todos = [{"content": "przeczytać calc.py", "status": "done"},
             {"content": "naprawić add", "status": "in_progress"}]
    # 32k: the todo tool is offered from the "small" tier up (a tiny model should
    # be doing the task, not keeping a list — TINY_OMIT)
    engine = ScriptedEngine([tool_call_text("todo_write", {"todos": todos}), "Plan zapisany."], n_ctx=32768)
    log = EventLog()
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path,
                  settings(permission_mode="yolo"), log, None)
    await agent.run_turn("zrób listę zadań")

    assert log.first("todos").items == todos
    assert agent._tool_ctx.state["todos"] == todos


async def test_plan_mode_hides_the_writing_tools_from_the_registry(tmp_path) -> None:
    engine = ScriptedEngine([tool_call_text("list_dir", {"path": "."}), "Plan: 1. ..."], n_ctx=8192)
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path,
                  settings(permission_mode="plan"), EventLog(), None)
    await agent.run_turn("zaplanuj")
    names = [t.name for t in agent.tools]
    assert "edit_file" not in names and "bash" not in names and "read_file" in names


async def test_async_event_callbacks_are_awaited() -> None:
    seen: list[str] = []

    async def on_event(event):
        await asyncio.sleep(0)
        seen.append(event.type)

    engine = ScriptedEngine(["ok"])
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(), on_event, None, tools=[])
    await agent.run_turn("hej")
    assert seen[0] == "turn_start" and seen[-1] == "turn_end"


async def test_failing_event_callback_never_breaks_the_turn() -> None:
    def on_event(event):
        raise RuntimeError("UI padło")

    engine = ScriptedEngine(["ok"])
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings(), on_event, None, tools=[])
    assert await agent.run_turn("hej") == "ok"


async def test_tool_result_events_carry_diffs() -> None:
    diff = "--- a.txt\n+++ a.txt\n@@ -1 +1 @@\n-stare\n+nowe\n"
    tool = FakeTool("edit_file", kind="write", output="Zmieniono a.txt", summary="+1 -1", diff=diff)
    agent, engine, runtime, log = build(
        [tool_call_text("edit_file", {"path": "a.txt"}), "Zmiana wprowadzona."], tools=[tool], mode="yolo"
    )
    await agent.run_turn("zmień a.txt")

    assert log.first("tool_result").diff == diff


async def test_tool_crash_is_reported_to_the_model() -> None:
    def boom(args, ctx):
        raise RuntimeError("pękło")

    class Crashing(FakeTool):
        async def run(self, args, ctx):
            self.calls.append(dict(args))
            return ToolResult(ok=False, output="Error: internal error in bash: RuntimeError: pękło",
                              summary="Wewnętrzny błąd narzędzia")

    tool = Crashing("bash", kind="read")
    agent, engine, runtime, log = build(
        [tool_call_text("bash", {"command": "ls"}), "Narzędzie zawiodło."], tools=[tool]
    )
    await agent.run_turn("uruchom")
    assert "internal error" in agent.messages[3]["content"]
    assert log.first("tool_result").ok is False


# --------------------------------------------------------------------------- #
# A model that stops making sense
# --------------------------------------------------------------------------- #


async def test_a_looping_model_is_stopped_and_explained() -> None:
    """Reported by a user: a 1-bit 30B model answered ``<todo_write>`` 32 times.

    The junk is not an answer and not a tool call, so the turn has to end with an
    explanation that names the likely cause instead of printing the loop.
    """
    agent, engine, runtime, log = build([("<todo_write> " * 32).strip()],
                                        tools=[FakeTool("todo_write")])
    await agent.run_turn("zrob mi strone o bananach")

    assert log.last("turn_end").reason == "error"
    errors = log.of("error")
    assert errors and "zapętlił" in errors[0].message
    assert "skwantyzowany" in errors[0].message          # says what to change
    assert len(engine.prompts) == 1                      # no pointless retry


async def test_a_looping_model_does_not_stream_the_whole_loop() -> None:
    """The guard exists to keep the junk off the screen, not just to log it."""
    junk = "Rozumiem, że chcesz stronę o bananach. " * 40
    agent, engine, runtime, log = build([junk], tools=[])
    await agent.run_turn("zrob strone")

    streamed = "".join(e.text for e in log.of("text"))
    assert len(streamed) < len(junk) / 2


async def test_a_tool_name_written_as_a_bare_tag_gets_one_correction() -> None:
    """``<todo_write>`` alone is an attempted call: correct it, do not answer with it."""
    tool = FakeTool("todo_write", output="zapisano")
    agent, engine, runtime, log = build(
        ["Zaplanuję pracę.\n<todo_write>",
         tool_call_text("todo_write", {"todos": [{"content": "zrobić stronę", "status": "pending"}]}),
         "Gotowe."],
        tools=[tool],
    )
    answer = await agent.run_turn("zrob strone")

    assert answer == "Gotowe."
    nudges = [m for m in agent.messages
              if m["role"] == "user" and "empty tag" in str(m.get("content"))]
    assert len(nudges) == 1
    assert len(tool.calls) == 1
    assert not any("<todo_write>" in str(m.get("content"))
                   for m in agent.messages if m["role"] == "assistant")


async def test_normal_repetition_in_an_answer_is_not_treated_as_a_loop() -> None:
    """A model listing files or drawing a table repeats itself legitimately."""
    text = "Zrobione. Pliki:\n" + "".join(f"- plik_{i}.html\n" for i in range(40))
    agent, engine, runtime, log = build([text], tools=[])
    answer = await agent.run_turn("wypisz pliki")

    assert log.last("turn_end").reason == "done"
    assert answer.startswith("Zrobione.")
    assert "plik_39.html" in answer


# --------------------------------------------------------------------------- #
# Reasoning effort chosen per model
# --------------------------------------------------------------------------- #


async def test_a_models_own_effort_wins_over_the_global_setting() -> None:
    """Set in "Ustawienia ładowania…": some models earn their thinking budget.

    A property of the model, not of the session — so it beats the global default
    and loses to an explicit ``/effort``.
    """
    engine = ScriptedEngine(["Gotowe."], n_ctx=8192, supports_thinking=True)
    runtime = FakeRuntime(engine)
    runtime.library = type("Lib", (), {"load_defaults": staticmethod(lambda key: {"effort": "high"})})()
    agent = Agent(runtime, "test/model", ".", settings(reasoning_effort="low"), EventLog(), None)

    assert agent.effort == "high"

    agent.set_effort("off")                  # an explicit choice still wins
    assert agent.effort == "off"
    agent.set_effort(None)
    assert agent.effort == "high"


async def test_without_a_per_model_effort_the_global_setting_is_used() -> None:
    engine = ScriptedEngine(["Gotowe."], n_ctx=8192, supports_thinking=True)
    runtime = FakeRuntime(engine)
    runtime.library = type("Lib", (), {"load_defaults": staticmethod(lambda key: {})})()
    agent = Agent(runtime, "test/model", ".", settings(reasoning_effort="low"), EventLog(), None)
    assert agent.effort == "low"


async def test_a_library_that_raises_does_not_break_the_turn() -> None:
    def boom(key: str) -> dict[str, object]:
        raise RuntimeError("dysk zniknął")

    engine = ScriptedEngine(["Gotowe."], n_ctx=8192, supports_thinking=True)
    runtime = FakeRuntime(engine)
    runtime.library = type("Lib", (), {"load_defaults": staticmethod(boom)})()
    agent = Agent(runtime, "test/model", ".", settings(reasoning_effort="medium"), EventLog(), None)
    assert agent.effort == "medium"


# --------------------------------------------------------------------------- #
# /clear must not disarm the session's own tools
# --------------------------------------------------------------------------- #
# ``ToolContext.state`` carries two different things: scratch one tool left for
# the next call, and the facades the session hands its tools — the skill library
# and the model-management one. ``clear()`` wiped the whole dict, so after
# ``/clear`` the ``skill`` tool answered "there are no skills in this session"
# and the model tools refused politely, for the rest of the session.


def test_clear_keeps_the_skill_library_available() -> None:
    from localcode.agent.loop import SKILLS_STATE_KEY

    agent, *_ = build(["gotowe"])
    assert agent._tool_ctx.state.get(SKILLS_STATE_KEY) is not None
    agent.clear()
    assert agent._tool_ctx.state.get(SKILLS_STATE_KEY) is not None


def test_clear_keeps_the_model_tools_working() -> None:
    from localcode.agent.tools.models import STATE_KEY

    agent, *_ = build(["gotowe"], manage_models=True)
    assert agent._tool_ctx.state.get(STATE_KEY) is not None
    agent.clear()
    assert agent._tool_ctx.state.get(STATE_KEY) is not None


def test_clear_still_drops_what_a_tool_left_behind() -> None:
    """The scratch half of the state must not survive — that was the point."""
    agent, *_ = build(["gotowe"])
    agent._tool_ctx.state["notatka-po-narzedziu"] = "stare"
    agent.clear()
    assert "notatka-po-narzedziu" not in agent._tool_ctx.state


# --------------------------------------------------------------------------- #
# A skill shelf is served in the size the model can afford
# --------------------------------------------------------------------------- #
# The loop tells the library which variant applies once, where both the profile
# and the window are known. Without that line the menu and every later `skill`
# call would silently disagree, so this pins it from the outside: what actually
# reaches the model.


def _write_skill(root: Path, name: str, *, description: str, mini_description: str) -> None:
    folder = root / ".localcode" / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n"
        f"description-mini: {mini_description}\n---\n\nPełna treść skilla.\n",
        encoding="utf-8")
    (folder / "SKILL-mini.md").write_text("Krótka treść.\n", encoding="utf-8")


async def test_a_small_window_gets_the_short_skill_menu(tmp_path: Path, isolated_home) -> None:
    _write_skill(tmp_path, "testy",
                 description="Użyj, gdy trzeba uruchomić albo napisać testy w tym repozytorium.",
                 mini_description="Uruchom testy repo")
    engine = ScriptedEngine(["gotowe"], n_ctx=8192)
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, Settings(), EventLog(), None)
    await agent.run_turn("zrób coś")
    prompt = engine.prompts[0]
    assert "Uruchom testy repo" in prompt
    assert "Użyj, gdy trzeba uruchomić" not in prompt


async def test_a_large_window_gets_the_full_skill_menu(tmp_path: Path, isolated_home) -> None:
    _write_skill(tmp_path, "testy",
                 description="Użyj, gdy trzeba uruchomić albo napisać testy w tym repozytorium.",
                 mini_description="Uruchom testy repo")
    engine = ScriptedEngine(["gotowe"], n_ctx=131_072)
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, Settings(), EventLog(), None)
    await agent.run_turn("zrób coś")
    assert "Użyj, gdy trzeba uruchomić" in engine.prompts[0]


# --------------------------------------------------------------------------- #
# "Make me a page" must end with a file, not with code pasted into the chat
# --------------------------------------------------------------------------- #
# Real session, Qwen3 4B: "napisz mi stronę o bananach" got the HTML in the chat
# and a claim that index.html had been written — no tool call all session, so
# localhost stayed empty. One nudge, narrowly aimed, turns it into a write.

async def test_language_notes_are_injected_for_the_project_language(tmp_path: Path) -> None:
    """A Go project puts the Go notes into the system prompt the model sees."""
    (tmp_path / "go.mod").write_text("module example.com/x\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\nfunc main() {}\n", encoding="utf-8")
    engine = ScriptedEngine(["gotowe"], n_ctx=16_384)
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, Settings(), EventLog(), None)
    await agent.run_turn("dodaj test")
    prompt = engine.prompts[0]
    assert "# Language notes" in prompt
    assert "go test ./..." in prompt  # the real Go command, from the mini note


_PAGE = "```html\n" + "\n".join(f"<p>banan {i}</p>" for i in range(20)) + "\n```"


async def test_a_pasted_page_is_saved_after_one_nudge(tmp_path: Path) -> None:
    from agent_fakes import tool_call_text
    from localcode.agent.prompts import UNSAVED_CODE_NUDGE
    from localcode.agent.tools import tools_for_mode

    engine = ScriptedEngine([
        f"Oto strona o bananach:\n\n{_PAGE}\n\nZapisałem ją w index.html.",
        tool_call_text("write_file", {"path": "index.html", "content": "<p>banan</p>"}),
        "Zapisałem stronę w index.html.",
    ])
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, settings(permission_mode="yolo"),
                  EventLog(), None, tools=tools_for_mode("default"))
    answer = await agent.run_turn("napisz mi stronę o bananach")
    assert (tmp_path / "index.html").exists()
    assert answer == "Zapisałem stronę w index.html."
    assert UNSAVED_CODE_NUDGE.split(".")[0] in engine.prompts[1]


@pytest.mark.parametrize("request_text", [
    "pokaż mi przykład strony w HTML",       # asked to be shown
    "wyjaśnij, jak działa ten kod",          # asked to explain
    "co to jest HTML?",                      # asked nothing to be made
])
async def test_code_that_was_asked_to_be_shown_stays_in_the_chat(tmp_path: Path,
                                                                 request_text: str) -> None:
    from localcode.agent.tools import tools_for_mode

    engine = ScriptedEngine([f"Proszę:\n\n{_PAGE}"])
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, settings(permission_mode="yolo"),
                  EventLog(), None, tools=tools_for_mode("default"))
    await agent.run_turn(request_text)
    assert len(engine.prompts) == 1                         # answered once, no nudge


async def test_shell_commands_in_the_answer_are_not_mistaken_for_a_file(tmp_path: Path) -> None:
    from localcode.agent.tools import tools_for_mode

    commands = "```bash\n" + "\n".join(f"echo krok {i}" for i in range(20)) + "\n```"
    engine = ScriptedEngine([f"Uruchom:\n\n{commands}"])
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, settings(permission_mode="yolo"),
                  EventLog(), None, tools=tools_for_mode("default"))
    await agent.run_turn("napisz mi skrypt, który wypisze kroki")
    assert len(engine.prompts) == 1


async def test_the_nudge_is_said_once_not_in_a_loop(tmp_path: Path) -> None:
    from localcode.agent.tools import tools_for_mode

    engine = ScriptedEngine([f"Strona:\n\n{_PAGE}", f"Znowu strona:\n\n{_PAGE}"])
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, settings(permission_mode="yolo"),
                  EventLog(), None, tools=tools_for_mode("default"))
    answer = await agent.run_turn("zrób mi stronę o bananach")
    assert len(engine.prompts) == 2
    assert answer.startswith("Znowu strona")


async def test_after_writing_a_file_code_in_the_summary_is_fine(tmp_path: Path) -> None:
    from agent_fakes import tool_call_text
    from localcode.agent.tools import tools_for_mode

    engine = ScriptedEngine([
        tool_call_text("write_file", {"path": "index.html", "content": "<p>banan</p>"}),
        f"Zapisane. Fragment:\n\n{_PAGE}",
    ])
    agent = Agent(FakeRuntime(engine), "test/model", tmp_path, settings(permission_mode="yolo"),
                  EventLog(), None, tools=tools_for_mode("default"))
    await agent.run_turn("napisz mi stronę o bananach")
    assert len(engine.prompts) == 2                         # no third, nudging request


async def test_a_message_typed_mid_turn_reaches_the_next_step() -> None:
    tool = FakeTool("read_file", output="x")
    calls = [tool_call_text("read_file", {"path": "a.txt"}), "Dobrze, użyję b.txt."]
    agent, engine, runtime, log = build(calls, tools=[tool])
    waiting = ["Weź jednak b.txt"]
    agent.take_interjections = lambda: [waiting.pop()] if waiting else []
    await agent.run_turn("czytaj a.txt")

    assert [e.text for e in log.of("user_message")] == ["Weź jednak b.txt"]
    users = [m for m in agent._history if m.get("role") == "user"]
    assert "Weź jednak b.txt" in users[-1]["content"]
    assert agent._history.index(users[-1]) > max(i for i, m in enumerate(agent._history) if m.get("role") == "tool")
    assert log.last("turn_end").reason == "done"
