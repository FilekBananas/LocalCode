"""Choosing the reasoning effort end to end: agent, chatbot, API and terminal.

``tests/runtime/test_effort.py`` proves the *detection* (which model family can
express which levels). This file proves that the choice actually travels:
settings → per-session / per-conversation override → the template kwargs of the
rendered prompt, and that every surface reports ``effortLevels`` so a client can
hide the control for models that have none.

``tests/chat``, ``tests/server`` and ``tests/tui`` are put on ``sys.path`` so
their fakes can be reused instead of inventing a second set (the same trick
``tests/tui/conftest.py`` uses).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve().parent
for _sibling in ("chat", "runtime", "server", "tui"):
    _path = _HERE.parent / _sibling
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from agent_fakes import EventLog, FakeRuntime, Reply, ScriptedEngine  # noqa: E402
from localcode.agent.loop import Agent  # noqa: E402
from localcode.config import Settings  # noqa: E402
from localcode.runtime.templates import AGENT_EFFORT_CHOICES, EFFORT_CHOICES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #


def build(replies: Any = ("gotowe",), *, thinking: bool = True, supports: bool = True,
          **agent_fields: Any) -> tuple[Agent, ScriptedEngine]:
    engine = ScriptedEngine([Reply.coerce(r) for r in replies], supports_thinking=supports)
    settings = Settings()
    settings.agent = replace(settings.agent, thinking=thinking, **agent_fields)
    agent = Agent(FakeRuntime(engine), "test/model", "/tmp", settings, EventLog(), None, tools=[])
    return agent, engine


def test_default_comes_from_the_settings() -> None:
    agent, _ = build()
    assert agent.effort == "medium"
    agent, _ = build(reasoning_effort="high")
    assert agent.effort == "high"


def test_a_broken_setting_falls_back_to_medium() -> None:
    agent, _ = build(reasoning_effort="bardzo mocno")
    assert agent.effort == "medium"


def test_set_effort_overrides_the_setting() -> None:
    agent, _ = build(reasoning_effort="low")
    agent.set_effort("high")
    assert agent.effort == "high"
    agent.set_effort(None)          # back to the settings
    assert agent.effort == "low"


def test_set_effort_rejects_nonsense() -> None:
    agent, _ = build()
    with pytest.raises(ValueError):
        agent.set_effort("bardzo mocno")
    assert agent.effort == "medium"


@pytest.mark.parametrize("level,expected", [("off", False), ("low", True), ("high", True)])
async def test_the_level_reaches_the_template(level: str, expected: bool) -> None:
    agent, engine = build()
    agent.set_effort(level)
    await agent.run_turn("cześć")
    assert engine.template_kwargs
    assert all(k == {"enable_thinking": expected} for k in engine.template_kwargs)


async def test_thinking_off_in_settings_wins_over_the_level() -> None:
    agent, engine = build(thinking=False, reasoning_effort="high")
    await agent.run_turn("cześć")
    assert engine.template_kwargs[-1] == {"enable_thinking": False}


async def test_a_model_without_a_switch_gets_no_kwargs() -> None:
    agent, engine = build(supports=False, reasoning_effort="high")
    await agent.run_turn("cześć")
    assert engine.template_kwargs[-1] == {}
    assert agent.effort_levels == []


async def test_levels_are_exposed_after_the_first_turn() -> None:
    agent, _ = build()
    await agent.run_turn("cześć")
    assert agent.effort_levels == ["off", "on"]


# --------------------------------------------------------------------------- #
# Chatbot
# --------------------------------------------------------------------------- #


def test_chat_request_accepts_and_validates_the_effort() -> None:
    from localcode.chat.repo import ApiError
    from localcode.chat.service import validate_chat_request

    assert validate_chat_request({"content": "x", "reasoningEffort": "high"})["reasoningEffort"] == "high"
    assert validate_chat_request({"content": "x"})["reasoningEffort"] is None
    with pytest.raises(ApiError):
        validate_chat_request({"content": "x", "reasoningEffort": "bardzo"})


async def test_conversation_override_round_trips_through_the_db(tmp_path: Path) -> None:
    from localcode.chat.db import ChatDB
    from localcode.chat.repo import ChatRepo

    db = ChatDB(tmp_path / "chat.db")
    repo = ChatRepo(db, tmp_path / "uploads")
    try:
        conv = repo.create_conversation({"title": "Test"})
        assert conv["reasoningEffort"] is None          # default: follow the settings
        patched = repo.update_conversation(conv["id"], {"reasoningEffort": "high"})
        assert patched["reasoningEffort"] == "high"
        assert repo.get_conversation(conv["id"])["reasoningEffort"] == "high"
        assert repo.update_conversation(conv["id"], {"reasoningEffort": None})["reasoningEffort"] is None
    finally:
        repo.close()


def test_conversation_patch_validation() -> None:
    from localcode.chat.api import _conversation_input
    from localcode.chat.repo import ApiError

    assert _conversation_input({"reasoningEffort": "low"}) == {"reasoningEffort": "low"}
    assert _conversation_input({"reasoningEffort": None}) == {"reasoningEffort": None}
    with pytest.raises(ApiError):
        _conversation_input({"reasoningEffort": "bardzo"})


@pytest.mark.parametrize("override,expected", [(None, False), ("off", False), ("high", True)])
async def test_chat_service_honours_setting_and_override(tmp_path: Path, override: str | None,
                                                         expected: bool) -> None:
    """The conversation override beats ``settings.chat.reasoning_effort``."""
    from chat_fakes import (CHAT_KEY, CHAT_KEY_2, EMBED_KEY, FakeEngine, FakeLibrary, FakeRuntime,
                            collect, make_model, make_service)
    from localcode.chat.db import ChatDB
    from localcode.chat.repo import ChatRepo

    engine = FakeEngine()
    db = ChatDB(tmp_path / "chat.db")
    repo = ChatRepo(db, tmp_path / "uploads")
    runtime = FakeRuntime({CHAT_KEY: engine, CHAT_KEY_2: engine, EMBED_KEY: FakeEngine()})
    library = FakeLibrary([make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["thinking", "tools"])])
    config = Settings()
    config.chat = replace(config.chat, default_model=CHAT_KEY,
                          reasoning_effort="off" if override is None else "medium")
    service = make_service(repo, runtime, library, config)
    try:
        conv = repo.create_conversation({"title": "T", "model": CHAT_KEY})
        if override is not None:
            repo.update_conversation(conv["id"], {"reasoningEffort": override})
        await collect(service.send({"conversationId": conv["id"], "content": "cześć"}))
        # token counting always renders with thinking off, so look for the
        # prompt that actually asked the model to think
        thought = {"enable_thinking": True} in engine.render_kwargs
        assert thought is expected
    finally:
        await service.aclose()
        repo.close()


# --------------------------------------------------------------------------- #
# HTTP surfaces
# --------------------------------------------------------------------------- #


def test_app_settings_patch_accepts_the_effort() -> None:
    from localcode.chat.repo import ApiError
    from localcode.server.core import _app_settings_patch

    out = _app_settings_patch({"agent": {"reasoningEffort": "HIGH"}, "chat": {"reasoningEffort": "off"}})
    assert out == {"agent": {"reasoning_effort": "high"}, "chat": {"reasoning_effort": "off"}}
    with pytest.raises(ApiError):
        _app_settings_patch({"agent": {"reasoningEffort": "bardzo"}})


def test_model_info_reports_effort_levels() -> None:
    import template_fixtures as tf
    from localcode.chat.service import model_info
    from localcode.hub.types import LocalModel, ModelMeta

    def dto(template: str | None) -> dict[str, Any]:
        model = LocalModel(key="org/m", format="gguf", path="/m.gguf", display_name="M", size_bytes=1,
                           meta=ModelMeta(arch="qwen3", chat_template=template))
        return model_info(model)

    assert dto(tf.load("gpt-oss-20b"))["effortLevels"] == ["low", "medium", "high"]
    assert dto(tf.load("llama-3.1-8b-instruct"))["effortLevels"] == []
    assert dto(None)["effortLevels"] == []


def test_openai_reasoning_effort_is_parsed() -> None:
    from localcode.server.openai_api import _effort

    assert _effort({"reasoning_effort": "high"}) == "high"
    assert _effort({"reasoning_effort": "none"}) == "off"
    assert _effort({"reasoning_effort": "minimal"}) == "off"
    assert _effort({}) is None
    assert _effort({"reasoning_effort": "bzdura"}) is None
    # an explicit template kwarg addresses the template itself and still wins
    assert _effort({"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}) is None


# --------------------------------------------------------------------------- #
# Terminal
# --------------------------------------------------------------------------- #


@pytest.fixture
def tui(tmp_path: Path, isolated_home: Path):
    from rich.console import Console
    from tui_fakes import ScriptedBackend, make_model_dto

    from localcode.tui.app import LocalCodeTui
    from localcode.tui.render import THEME

    model = make_model_dto()
    model["effortLevels"] = ["off", "on"]
    backend = ScriptedBackend(tmp_path, models=[model], model_key=model["key"])
    console = Console(record=True, width=100, color_system=None, theme=THEME, highlight=False,
                      legacy_windows=False, force_terminal=False)
    return LocalCodeTui(backend, console)


async def test_think_sets_an_explicit_level(tui) -> None:
    await tui.dispatch("/think off")
    assert tui.backend.effort == "off" and tui.show_reasoning is False
    await tui.dispatch("/think high")
    assert tui.backend.effort == "high" and tui.show_reasoning is True
    assert "Myślenie: high" in tui.console.export_text(clear=False)


async def test_think_cycles_and_reports_the_resolved_level(tui) -> None:
    await tui.dispatch("/think off")
    for expected in ("low", "medium", "high", "off"):
        await tui.dispatch("/think")
        assert tui.backend.effort == expected
    # the model only knows off/on, so the message says what it will really use
    await tui.dispatch("/think high")
    assert "model używa: on" in tui.console.export_text(clear=False)


async def test_think_rejects_an_unknown_level(tui) -> None:
    await tui.dispatch("/think bardzo")
    assert tui.backend.effort == "medium"
    assert "Nieznany poziom" in tui.console.export_text(clear=False)


async def test_toolbar_shows_the_level(tui) -> None:
    await tui.dispatch("/think low")
    assert any("myślenie low" in text for _style, text in tui._toolbar())


def test_think_command_advertises_the_levels() -> None:
    from localcode.tui.commands import COMMANDS

    command = next(c for c in COMMANDS if c.name == "think")
    assert command.values == AGENT_EFFORT_CHOICES
