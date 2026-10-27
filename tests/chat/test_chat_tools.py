"""The chatbot's four model tools (``localcode.chat.tools``).

Asking a chat window "pobierz mi jakiś dobry model do kodowania" used to
produce an essay about CodeLlama and no download. These tests cover the loop
that makes it happen instead — and the limits that keep a chatbot a chatbot:
four tools, three rounds, nothing else.
"""

from __future__ import annotations

from typing import Any

import pytest

from localcode.agent.tools.models import ModelOps
from localcode.chat.tools import MAX_ROUNDS, ChatToolRunner
from localcode.runtime.base import TemplateInfo


class FakeOps(ModelOps):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    async def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        self.calls.append(("search", query))
        return [{"id": "unsloth/Qwen3-Coder-30B-GGUF", "name": "Qwen3 Coder 30B",
                 "sizeLabel": "ok. 18,5 GB", "fit": "Zmieści się",
                 "tagline": "Agent kodujący.", "trust": {"score": 87}, "installed": False}]

    async def installed(self) -> list[dict[str, Any]]:
        self.calls.append(("installed", None))
        return []

    async def install(self, name_or_repo: str) -> dict[str, Any]:
        self.calls.append(("install", name_or_repo))
        return {"id": "job-1", "repo": name_or_repo, "label": "Qwen3 Coder 30B", "state": "queued"}

    async def load(self, key: str) -> dict[str, Any]:
        self.calls.append(("load", key))
        return {"key": key, "displayName": key}

    async def status(self) -> dict[str, Any]:
        return {"instances": [], "budget": {}}


class FakeEngine:
    def __init__(self, *, supports_tools: bool = True, tool_format: str = "hermes") -> None:
        self.template_info = TemplateInfo(supports_tools=supports_tools, tool_format=tool_format)


def call_text(name: str, arguments: dict[str, Any]) -> str:
    import json

    return ("Zaraz sprawdzę.\n<tool_call>\n"
            + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
            + "\n</tool_call>")


# --------------------------------------------------------------------------- #
# What the chat is allowed to do
# --------------------------------------------------------------------------- #


def test_the_chat_gets_four_tools_and_no_others() -> None:
    schemas = ChatToolRunner(FakeOps()).schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == ["model_search", "model_list", "model_install", "model_load"]
    assert "bash" not in names and "write_file" not in names


def test_without_the_setting_there_are_no_tools_at_all() -> None:
    runner = ChatToolRunner(None, enabled=False)
    assert runner.schemas() is None
    assert runner.parse(call_text("model_search", {"query": "x"}), FakeEngine()) == []


def test_a_model_whose_template_cannot_call_tools_is_not_given_any() -> None:
    runner = ChatToolRunner(FakeOps())
    assert runner.supported_by(FakeEngine(supports_tools=True)) is True
    assert runner.supported_by(FakeEngine(supports_tools=False)) is False


# --------------------------------------------------------------------------- #
# Parsing what the model asked for
# --------------------------------------------------------------------------- #


def test_a_tool_call_is_recognised_in_the_answer() -> None:
    runner = ChatToolRunner(FakeOps())
    calls = runner.parse(call_text("model_search", {"query": "coder"}), FakeEngine())
    assert [c.name for c in calls] == ["model_search"]
    assert calls[0].arguments == {"query": "coder"}


def test_a_call_to_something_the_chat_does_not_have_is_ignored() -> None:
    runner = ChatToolRunner(FakeOps())
    assert runner.parse(call_text("bash", {"command": "rm -rf /"}), FakeEngine()) == []


def test_the_call_block_is_kept_out_of_what_the_user_reads() -> None:
    runner = ChatToolRunner(FakeOps())
    text = call_text("model_search", {"query": "coder"})
    assert "<tool_call>" not in runner.content_without_calls(text, FakeEngine())
    assert "Zaraz sprawdzę." in runner.content_without_calls(text, FakeEngine())


def test_a_malformed_call_does_not_break_the_turn() -> None:
    runner = ChatToolRunner(FakeOps())
    assert runner.parse("<tool_call>{ to nie jest json", FakeEngine()) == []


# --------------------------------------------------------------------------- #
# Running them
# --------------------------------------------------------------------------- #


async def test_running_a_search_reports_progress_then_the_result() -> None:
    ops = FakeOps()
    runner = ChatToolRunner(ops)
    calls = runner.parse(call_text("model_search", {"query": "coder"}), FakeEngine())

    frames = [frame async for frame in runner.run(calls)]

    assert [f["state"] for f in frames] == ["running", "done"]
    assert frames[0]["title"].startswith("Szukaj modelu")
    assert frames[1]["ok"] is True
    assert "Qwen3 Coder 30B" in frames[1]["message"]["content"]
    assert frames[1]["message"]["role"] == "tool"
    assert ops.calls == [("search", "coder")]


async def test_installing_goes_through_to_the_download_manager() -> None:
    ops = FakeOps()
    runner = ChatToolRunner(ops)
    calls = runner.parse(call_text("model_install", {"name": "Qwen3 Coder 30B"}), FakeEngine())

    frames = [frame async for frame in runner.run(calls)]

    assert frames[-1]["ok"] is True
    assert ops.calls == [("install", "Qwen3 Coder 30B")]


async def test_tool_output_stays_small_enough_for_a_small_model() -> None:
    class Chatty(FakeOps):
        async def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
            return [{"id": f"org/model-{i}", "name": f"Model {i}", "sizeLabel": "ok. 1 GB",
                     "fit": "Zmieści się", "tagline": "x" * 400, "trust": {"score": 50}}
                    for i in range(50)]

    runner = ChatToolRunner(Chatty())
    calls = runner.parse(call_text("model_search", {"query": "x", "limit": 20}), FakeEngine())
    frames = [frame async for frame in runner.run(calls)]
    assert len(frames[-1]["message"]["content"]) <= 2_100


def test_the_assistant_turn_records_what_it_asked_for() -> None:
    runner = ChatToolRunner(FakeOps())
    calls = runner.parse(call_text("model_search", {"query": "coder"}), FakeEngine())
    message = runner.assistant_message("Zaraz sprawdzę.", calls)
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["function"]["name"] == "model_search"


def test_the_loop_is_short_by_design() -> None:
    """A chatbot that keeps calling tools instead of answering is a bug."""
    assert MAX_ROUNDS <= 3
    assert ChatToolRunner(FakeOps()).max_rounds == MAX_ROUNDS
