"""Auto mode: a reviewer model decides the calls that would ask the user.

Runs against ``ScriptedEngine``/``FakeRuntime``: the reviewer's answer is just
the next scripted reply, so every test says exactly what the reviewer said.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, FakeTool, Reply, ScriptedEngine, tool_call_text
from localcode.agent import autoreview
from localcode.agent import guard as guard_mod
from localcode.agent.loop import Agent
from localcode.agent.permissions import (
    DANGER_FLAG,
    MODE_LABELS,
    MODES,
    OUTSIDE_PROJECT_FLAG,
    REMOTE_FLAG,
    PermissionPolicy,
    normalize_mode,
)
from localcode.agent.session import SessionStore
from localcode.agent.tools.base import PermissionInfo
from localcode.config import Settings
from localcode.runtime.base import EngineError

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

REVIEW_MARK = "[Auto mode — safety review]"


def settings(**agent_fields: Any) -> Settings:
    s = Settings()
    s.agent = replace(s.agent, **agent_fields)
    return s


class RecordingEngine(ScriptedEngine):
    """A scripted engine that also keeps the messages and tools it was rendered with."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rendered: list[tuple[list[dict[str, Any]], Any]] = []

    async def apply_template(self, messages, tools=None, **kwargs):  # type: ignore[override]
        self.rendered.append(([dict(m) for m in messages], tools))
        return await super().apply_template(messages, tools, **kwargs)

    def review_prompts(self) -> list[str]:
        return [p for p in self.prompts if REVIEW_MARK in p]

    def review_renders(self) -> list[tuple[list[dict[str, Any]], Any]]:
        return [(m, t) for m, t in self.rendered
                if m and REVIEW_MARK in str(m[-1].get("content") or "")]


class MultiRuntime(FakeRuntime):
    """A runtime with one engine per model key; unknown keys fail to acquire.

    ``fail_on`` holds the (1-based) numbers of acquisitions that fail anyway.
    """

    def __init__(self, engines: dict[str, ScriptedEngine], *, fail_on: set[int] = frozenset()) -> None:
        super().__init__(engines["test/model"])
        self.engines = engines
        self.fail_on = set(fail_on)
        self.keys: list[str] = []

    def acquire(self, model_key: str, *, purpose: str = "chat", n_ctx: int | None = None):
        self.keys.append(model_key)
        if model_key not in self.engines or len(self.keys) in self.fail_on:
            raise EngineError(f"Nie znaleziono modelu „{model_key}”.")
        self.engine = self.engines[model_key]
        return super().acquire(model_key, purpose=purpose, n_ctx=n_ctx)


def exec_tool(title: str = "Uruchom: npm test", preview: str = "npm test") -> FakeTool:
    return FakeTool("bash", kind="exec", output="exit 0", summary="exit 0",
                    permission_info=PermissionInfo(title=title, preview=preview))


def build(replies, *, tools=(), ask=None, engine=None, session=None, **agent_fields: Any):
    engine = engine or RecordingEngine(replies)
    runtime = FakeRuntime(engine)
    log = EventLog()
    agent = Agent(runtime, "test/model", "/tmp", settings(permission_mode="auto", **agent_fields),
                  log, ask, tools=list(tools), session=session)
    return agent, engine, runtime, log


def asker(answer: str = "allow"):
    asked: list[Any] = []

    async def ask(request):
        asked.append(request)
        return answer

    return ask, asked


# --------------------------------------------------------------------------- #
# Modes and policy
# --------------------------------------------------------------------------- #


def test_auto_is_its_own_mode() -> None:
    assert normalize_mode("auto") == "auto"
    assert normalize_mode("Auto") == "auto"
    assert "auto" in MODES and MODE_LABELS["auto"].startswith("Auto")


def test_auto_decides_like_default_but_lets_project_edits_through() -> None:
    policy = PermissionPolicy("auto", cwd="/tmp/project")
    write = FakeTool("write_file", kind="write")
    info = PermissionInfo(title="Zapisz a.txt", preview="+ x")
    assert policy.decide(write, {"path": "a.txt"}, info) == "allow"
    assert policy.decide(FakeTool("read_file", kind="read"), {"path": "a.txt"}, None) == "allow"
    bash = FakeTool("bash", kind="exec")
    assert policy.decide(bash, {"command": "git status"}, None) == "allow"
    assert policy.decide(bash, {"command": "npm test"}, None) == "ask"
    policy.remember_always(bash, {"command": "npm test"})
    assert policy.decide(bash, {"command": "npm test"}, None) == "allow"


def test_needs_human_for_flagged_and_catastrophic_calls() -> None:
    policy = PermissionPolicy("auto", cwd="/tmp/project")
    bash = FakeTool("bash", kind="exec")
    write = FakeTool("write_file", kind="write")
    assert not policy.needs_human(bash, {"command": "npm test"}, PermissionInfo("Uruchom: npm test", "npm test"))
    assert policy.needs_human(bash, {"command": "rm -rf ~"}, PermissionInfo("Uruchom: rm -rf ~", "rm -rf ~"))
    for flag in (DANGER_FLAG, REMOTE_FLAG):
        assert policy.needs_human(write, {"path": "x"}, PermissionInfo(f"Zapisz x {flag}", "+ x"))
    # outside the project is the reviewer's call
    assert not policy.needs_human(write, {"path": "x"}, PermissionInfo(f"Zapisz x {OUTSIDE_PROJECT_FLAG}", "+ x"))
    assert not policy.needs_human(write, {"path": "a.txt"}, PermissionInfo("Zapisz a.txt", "+ x"))


# --------------------------------------------------------------------------- #
# Verdict parser
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text, verdict", [
    ("TAK\nPowód: Uruchamia testy, o które prosił użytkownik.", "allow"),
    ("NIE\nPowód: Usuwa katalog src bez prośby.", "deny"),
    ("**NIE**\n**Powód:** wysyła klucz do sieci.", "deny"),
    ("Werdykt: TAK\nPowód: tylko testy.", "allow"),
    ("Yes. It only lists files.", "allow"),
    ("deny", "deny"),
    ("NIE — usuwa pliki użytkownika", "deny"),
    ('{"verdict": "deny", "reason": "curl | sh"}', "deny"),
    ('```json\n{"allow": true, "reason": "bezpieczne"}\n```', "allow"),
])
def test_parse_verdict(text: str, verdict: str) -> None:
    found = autoreview.parse_verdict(text)
    assert found is not None and found[0] == verdict


@pytest.mark.parametrize("text", [
    "", "Hmm, let me think about it.", "TAK/NIE", "TAK or NIE",
    "Tak naprawdę to polecenie usuwa pliki.",
])
def test_parse_verdict_refuses_unclear_answers(text: str) -> None:
    assert autoreview.parse_verdict(text) is None


def test_parse_verdict_keeps_the_reason() -> None:
    assert autoreview.parse_verdict("NIE\nPowód: Usuwa katalog src.") == ("deny", "Usuwa katalog src.")
    assert autoreview.parse_verdict("NIE — usuwa pliki") == ("deny", "usuwa pliki")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


async def test_reviewer_yes_runs_the_tool_and_says_why() -> None:
    ask, asked = asker()
    tool = exec_tool()
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}),
         "TAK\nPowód: Użytkownik prosił o uruchomienie testów.", "Testy przeszły."],
        tools=[tool], ask=ask)
    await agent.run_turn("uruchom testy")

    assert tool.calls == [{"command": "npm test"}]
    assert asked == [] and log.of("permission_request") == []
    review = log.first("auto_review")
    assert review.verdict == "allow" and review.tool == "bash"
    assert review.reason == "Użytkownik prosił o uruchomienie testów."
    assert review.reviewer_model == "test/model"
    assert review.to_dict()["reviewerModel"] == "test/model"
    assert log.types.index("auto_review") < log.types.index("tool_result")
    assert log.first("tool_result").ok is True


NO_REPLY = ("NIE\nCo robi: Usuwa katalog build.\n"
            "Powód: Użytkownik nie prosił o usuwanie katalogu build.")


async def test_reviewer_no_goes_to_the_user_with_the_review_on_the_card() -> None:
    """NIE is not the last word: the card says what the action does and why not."""
    ask, asked = asker("allow")
    tool = exec_tool(title="Uruchom: rm -r build", preview="rm -r build")
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "rm -r build"}), NO_REPLY, "Usunięte."],
        tools=[tool], ask=ask)
    await agent.run_turn("posprzątaj")

    assert log.first("auto_review").verdict == "deny"
    assert log.types.index("auto_review") < log.types.index("permission_request")
    [request] = asked
    assert request.review["verdict"] == "deny" and request.review["model"] == "test/model"
    assert request.review["reason"].startswith("Co robi: Usuwa katalog build. Powód: ")
    assert request.to_dict()["review"]["verdict"] == "deny"
    # The user overruled the reviewer: the call ran.
    assert tool.calls == [{"command": "rm -r build"}]
    assert log.first("tool_result").ok is True


async def test_reviewer_no_and_user_no_tells_the_agent_both() -> None:
    ask, asked = asker("deny")
    tool = exec_tool(title="Uruchom: rm -r build", preview="rm -r build")
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "rm -r build"}), NO_REPLY, "Dobrze, nie usuwam."],
        tools=[tool], ask=ask)
    await agent.run_turn("posprzątaj")

    assert len(asked) == 1 and tool.calls == []
    result = log.first("tool_result")
    assert result.ok is False and result.summary == "Odrzucone"
    tool_message = next(m for m in agent.messages if m["role"] == "tool")
    assert tool_message["content"].startswith("Auto-mode reviewer rejected this call: Co robi: ")
    assert tool_message["content"].endswith("The user also declined it.")


async def test_reviewer_no_without_anyone_to_ask_still_refuses() -> None:
    tool = exec_tool(title="Uruchom: rm -r build", preview="rm -r build")
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "rm -r build"}),
         "NIE\nPowód: Użytkownik nie prosił o usuwanie katalogu build.", "Dobrze, nie usuwam."],
        tools=[tool], ask=None)
    await agent.run_turn("posprzątaj")

    assert tool.calls == [] and log.of("permission_request") == []
    result = log.first("tool_result")
    assert result.ok is False and result.summary == "Odrzucone przez recenzenta"
    tool_message = next(m for m in agent.messages if m["role"] == "tool")
    assert tool_message["content"].startswith("Auto-mode reviewer rejected this call: "
                                              "Użytkownik nie prosił o usuwanie katalogu build.")
    assert "Choose a safer way or ask the user." in tool_message["content"]


async def test_an_unclear_review_rides_on_the_question_as_ask() -> None:
    ask, asked = asker("deny")
    tool = exec_tool()
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), "Hmm, trudno powiedzieć.", "Ok."],
        tools=[tool], ask=ask)
    await agent.run_turn("uruchom testy")

    [request] = asked
    assert request.review["verdict"] == "ask"
    assert request.review["reason"] == autoreview.REASON_UNCLEAR


@pytest.mark.parametrize("review_reply", [
    "Hmm, trudno powiedzieć.",
    Reply(error=EngineError("silnik padł")),
    tool_call_text("bash", {"command": "ls"}),
])
async def test_unclear_review_falls_back_to_the_user(review_reply) -> None:
    ask, asked = asker("allow")
    tool = exec_tool()
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), review_reply, "Zrobione."],
        tools=[tool], ask=ask)
    await agent.run_turn("uruchom testy")

    review = log.first("auto_review")
    assert review.verdict == "ask"
    assert "decyzja należy do Ciebie" in review.reason
    assert len(asked) == 1 and tool.calls == [{"command": "npm test"}]
    assert log.types.index("auto_review") < log.types.index("permission_request")


async def test_a_review_that_times_out_asks_the_user(monkeypatch) -> None:
    monkeypatch.setattr(autoreview, "REVIEW_TIMEOUT", 0.05)
    ask, asked = asker("deny")
    tool = exec_tool()
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "npm test"}), Reply(text="TAK", stall_after=0), "Ok."],
        tools=[tool], ask=ask)
    await agent.run_turn("uruchom testy")

    assert log.first("auto_review").verdict == "ask"
    assert "nie odpowiedział" in log.first("auto_review").reason
    assert len(asked) == 1 and tool.calls == []


async def test_dangerous_calls_get_an_advisory_review_and_still_go_to_the_user() -> None:
    """A TAK from both gates never approves what needs a human — it is advice on the card."""
    ask, asked = asker("deny")
    tool = exec_tool(title=f"Uruchom: rm -rf ~ {DANGER_FLAG}", preview="rm -rf ~")
    agent, engine, _runtime, log = build(
        [tool_call_text("bash", {"command": "rm -rf ~"}),
         "TAK\nCo robi: Usuwa cały katalog domowy.\nPowód: Użytkownik o to prosił.", "Nie usuwam."],
        tools=[tool], ask=ask)
    await agent.run_turn("usuń wszystko")

    assert len(engine.review_prompts()) == 1, "reviewed, as advice"
    assert len(engine.guard_prompts) == 1
    [request] = asked
    assert request.review["verdict"] == "allow"
    assert request.review["reason"].startswith("Co robi: Usuwa cały katalog domowy.")
    assert request.review["guard"]["verdict"] == "allow"
    assert tool.calls == [], "the TAKs approved nothing"
    assert log.of("auto_review") == [], "no event a client would read as 'it ran'"


async def test_an_advisory_review_that_fails_leaves_a_plain_question() -> None:
    ask, asked = asker("deny")
    tool = exec_tool(title=f"Uruchom: rm -rf ~ {DANGER_FLAG}", preview="rm -rf ~")
    engine = RecordingEngine([tool_call_text("bash", {"command": "rm -rf ~"}),
                              "Hmm, nie wiem.", "Nie usuwam."])
    engine.guard_replies = [Reply(text="Może?"), Reply(text="Może?")]    # and its one-word follow-up
    agent, engine, _runtime, log = build([], tools=[tool], ask=ask, engine=engine)
    await agent.run_turn("usuń wszystko")

    [request] = asked
    assert request.review is None and "review" not in request.to_dict()
    assert tool.calls == []


async def test_reads_never_reach_the_reviewer() -> None:
    tool = FakeTool("read_file", kind="read", output="treść")
    agent, engine, _runtime, log = build(
        [tool_call_text("read_file", {"path": "a.txt"}), "Przeczytane."], tools=[tool])
    await agent.run_turn("przeczytaj a.txt")

    assert len(tool.calls) == 1 and log.of("auto_review") == []
    assert engine.review_prompts() == []


async def test_review_prompt_reuses_the_step_prefix() -> None:
    tool = exec_tool()
    agent, engine, _runtime, _log = build(
        [tool_call_text("bash", {"command": "npm test"}, prefix="Uruchamiam testy."),
         "TAK\nPowód: testy.", "Gotowe."],
        tools=[tool])
    await agent.run_turn("uruchom testy")

    [(messages, tools)] = engine.review_renders()
    last = messages[-1]
    assert last["role"] == "user" and REVIEW_MARK in last["content"]
    assert "npm test" in last["content"] and "Uruchamiam testy." in last["content"]
    assert not any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    assert tools, "the agent's own model reviews with the same tools as the step"
    step_prompt = engine.prompts[0]
    review_prompt = engine.review_prompts()[0]
    assert review_prompt.startswith(step_prompt[: -len("<|assistant|>\n")])
    params = engine.sampling[1]
    assert params.temperature == 0.0 and params.max_tokens == autoreview.REVIEW_TOKENS


async def test_second_call_of_a_step_sees_the_first_result() -> None:
    first = FakeTool("write_file", kind="write", output="Zapisano a.txt", summary="1 linia",
                     permission_info=PermissionInfo(title="Zapisz a.txt", preview="+ print(1)"))
    second = exec_tool(title="Uruchom: python a.txt", preview="python a.txt")
    calls = (tool_call_text("write_file", {"path": "a.txt", "content": "print(1)"})
             + "\n" + tool_call_text("bash", {"command": "python a.txt"}))
    agent, engine, _runtime, log = build(
        [calls, "TAK\nPowód: uruchamia nowy skrypt.", "Działa."],
        tools=[first, second])
    await agent.run_turn("napisz i uruchom a.txt")

    assert len(first.calls) == 1 and len(second.calls) == 1
    assert [e.verdict for e in log.of("auto_review")] == ["allow"]      # the edit needed no review
    messages, _tools = engine.review_renders()[0]
    assert messages[-2]["role"] != "assistant"
    assert "write_file" in messages[-1]["content"] and "Zapisano a.txt" in messages[-1]["content"]


async def test_the_verdict_is_stored_on_the_tool_item(isolated_home, tmp_path) -> None:
    store = SessionStore.create(tmp_path, model_key="test/model", mode="auto")
    tool = exec_tool()
    agent, _engine, _runtime, _log = build(
        [tool_call_text("bash", {"command": "npm test"}), "NIE\nPowód: nie teraz.", "Ok."],
        tools=[tool], session=store)
    await agent.run_turn("uruchom testy")

    item = next(i for i in store.items if i["kind"] == "tool")
    guard = item["review"].pop("guard")
    assert item["review"] == {"verdict": "deny", "reason": "nie teraz.", "model": "test/model"}
    assert guard["verdict"] == "allow" and guard["model"] == guard_mod.RULES_MODEL   # npm test: nothing risky
    reopened = SessionStore.resume(store.session.id)
    assert next(i for i in reopened.items if i["kind"] == "tool")["review"]["verdict"] == "deny"


async def test_a_reviewer_from_settings_is_used() -> None:
    agent_engine = RecordingEngine([tool_call_text("bash", {"command": "npm test"}), "Gotowe."])
    reviewer = RecordingEngine(["TAK\nPowód: testy projektu."])
    runtime = MultiRuntime({"test/model": agent_engine, "judge/model": reviewer})
    tool = exec_tool()
    log = EventLog()
    agent = Agent(runtime, "test/model", "/tmp",
                  settings(permission_mode="auto", auto_reviewer_model="judge/model"),
                  log, None, tools=[tool])
    await agent.run_turn("uruchom testy")

    assert "judge/model" in runtime.keys
    assert len(reviewer.review_prompts()) == 1 and agent_engine.review_prompts() == []
    assert "<|tools|>" not in reviewer.review_prompts()[0], "a different local reviewer gets no schemas"
    assert log.first("auto_review").reviewer_model == "judge/model"
    assert tool.calls == [{"command": "npm test"}]


async def test_an_unavailable_reviewer_gives_way_to_the_agents_model() -> None:
    engine = RecordingEngine([tool_call_text("bash", {"command": "npm test"}),
                              "TAK\nPowód: testy.", "Gotowe."])
    runtime = MultiRuntime({"test/model": engine})
    tool = exec_tool()
    log = EventLog()
    agent = Agent(runtime, "test/model", "/tmp",
                  settings(permission_mode="auto", auto_reviewer_model="gone/model"),
                  log, None, tools=[tool])
    await agent.run_turn("uruchom testy")

    assert runtime.keys.count("gone/model") == 1
    review = log.first("auto_review")
    assert review.verdict == "allow" and review.reviewer_model == "test/model"
    assert tool.calls == [{"command": "npm test"}]


async def test_without_any_reviewer_and_approver_the_call_is_refused() -> None:
    engine = RecordingEngine([tool_call_text("bash", {"command": "npm test"}), "Nie mogę."])
    runtime = MultiRuntime({"test/model": engine}, fail_on={2})   # 1 = step, 2 = review, 3 = step
    tool = exec_tool()
    log = EventLog()
    agent = Agent(runtime, "test/model", "/tmp", settings(permission_mode="auto"), log, None,
                  tools=[tool])
    await agent.run_turn("uruchom testy")

    review = log.first("auto_review")
    assert review.verdict == "ask" and review.reason == autoreview.REASON_NO_MODEL
    assert tool.calls == []
    assert "approval" in next(m for m in agent.messages if m["role"] == "tool")["content"]


async def test_mode_notice_and_prompt_mention_the_reviewer() -> None:
    agent, engine, _runtime, _log = build(["Cześć."])
    await agent.run_turn("hej")
    assert "reviewer" in agent.messages[0]["content"]
    agent.set_mode("default")
    agent.set_mode("auto")
    assert "auto" in (agent._pending_notice or "")


def _review_kwargs(engine: RecordingEngine) -> dict[str, Any]:
    index = next(i for i, (m, _t) in enumerate(engine.rendered)
                 if m and REVIEW_MARK in str(m[-1].get("content") or ""))
    return engine.template_kwargs[index]


async def test_review_turns_thinking_off_when_that_keeps_the_prefix() -> None:
    engine = RecordingEngine([tool_call_text("bash", {"command": "npm test"}),
                              "TAK\nPowód: testy.", "Gotowe."], supports_thinking=True)
    agent, engine, _runtime, _log = build([], tools=[exec_tool()], engine=engine)
    await agent.run_turn("uruchom testy")

    assert engine.template_kwargs[0] == {"enable_thinking": True}
    assert _review_kwargs(engine) == {"enable_thinking": False}
    assert engine.sampling[1].max_tokens == autoreview.REVIEW_TOKENS


class HeaderEffortEngine(RecordingEngine):
    """A template that writes the reasoning level into the prompt's first line (gpt-oss style)."""

    async def apply_template(self, messages, tools=None, **kwargs):  # type: ignore[override]
        prompt = await super().apply_template(messages, tools, **kwargs)
        level = (kwargs.get("template_kwargs") or {}).get("enable_thinking")
        return f"<|reasoning:{level}|>\n{prompt}"


async def test_review_keeps_the_steps_reasoning_when_off_would_break_the_cache() -> None:
    engine = HeaderEffortEngine([tool_call_text("bash", {"command": "npm test"}),
                                 "TAK\nPowód: testy.", "Gotowe."], supports_thinking=True)
    agent, engine, _runtime, _log = build([], tools=[exec_tool()], engine=engine)
    await agent.run_turn("uruchom testy")

    assert _review_kwargs(engine) == {"enable_thinking": True}
    step_prompt, review_prompt = engine.prompts[0], engine.review_prompts()[0]
    assert review_prompt.startswith(step_prompt[: -len("<|assistant|>\n")])
    assert engine.sampling[1].max_tokens == autoreview.REVIEW_THINKING_TOKENS


def test_the_verdict_carries_what_the_action_does() -> None:
    verdict, reason = autoreview.parse_verdict("TAK\nCo robi: Czyta linie 1–9 z pliku z zasadami.\n"
                                    "Powód: Agent potrzebuje zasad konkursu.")
    assert verdict == "allow"
    assert reason == "Co robi: Czyta linie 1–9 z pliku z zasadami. Powód: Agent potrzebuje zasad konkursu."
    assert autoreview.parse_verdict("NIE\nPowód: Usuwa cały folder.") == ("deny", "Usuwa cały folder.")


def test_the_verdict_comes_last_and_is_not_part_of_the_reason() -> None:
    """Deciding after the reasoning keeps "NIE … zatwierdzam mimo to" from happening."""
    assert autoreview.parse_verdict("Co robi: Kasuje /tmp/x.\nPowód: Katalog roboczy,\nnic z projektu.\nTAK") \
        == ("allow", "Co robi: Kasuje /tmp/x. Powód: Katalog roboczy, nic z projektu.")
    assert autoreview.parse_verdict("Bezpieczny odczyt.\nNIE") == ("deny", "Bezpieczny odczyt.")
    assert "TAK or NIE" in autoreview._RULES.splitlines()[-1]
