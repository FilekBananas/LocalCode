"""The agent answers in the language of the question.

Regression: Qwen3-0.6B answered Polish questions in English — the only rule was
one sentence at the very end of a long English system prompt.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from agent_fakes import EventLog, FakeRuntime, ScriptedEngine
from localcode.agent.loop import Agent
from localcode.agent.prompts import detect_language
from localcode.config import Settings

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.mark.parametrize("text, language", [
    ("włącz mi kimiego z deep infry", "pl"),
    ("zaladuj ty mi go", "pl"),
    ("hej", "pl"),
    ("kimi z deep infry klucz jest juz dodany", "pl"),
    ("wich model i am talking to now", "en"),
    ("what is this file for?", "en"),
    ("ok", None),
    ("", None),
])
def test_the_language_of_a_message(text: str, language: str | None) -> None:
    assert detect_language(text) == language


def build(replies: list[str]) -> tuple[Agent, ScriptedEngine]:
    s = Settings()
    s.agent = replace(s.agent, permission_mode="yolo")
    engine = ScriptedEngine(replies, n_ctx=8192)
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", s, EventLog(), None, tools=[])
    return agent, engine


async def test_a_polish_question_puts_polish_in_the_prompt_but_not_in_the_history() -> None:
    agent, engine = build(["Cześć!"])
    await agent.run_turn("hej, zrob mi strone")
    prompt = engine.prompts[0]
    assert "The user writes in Polish" in prompt
    assert "(Odpowiedz po polsku.)" in prompt                 # next to the question, for a small model
    stored = [m for m in agent.messages if m["role"] == "user"]
    assert stored[-1]["content"] == "hej, zrob mi strone"     # the user's own words are kept


async def test_switching_language_follows_the_latest_message() -> None:
    agent, engine = build(["Cześć!", "Hello!"])
    await agent.run_turn("hej, co tam")
    await agent.run_turn("which model am I talking to now?")
    assert "The user writes in English" in engine.prompts[-1]
    assert "(Odpowiedz po polsku.)" not in engine.prompts[-1]


async def test_an_unclear_message_keeps_the_language_already_known() -> None:
    agent, engine = build(["Cześć!", "Ok."])
    await agent.run_turn("hej, zrob mi to")
    await agent.run_turn("ok")
    assert "The user writes in Polish" in engine.prompts[-1]
