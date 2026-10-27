"""A conversation whose model was deleted from disk keeps working on another one."""

from __future__ import annotations

from typing import Any

from chat_fakes import CHAT_KEY, CHAT_KEY_2, FakeLibrary, collect, make_service
from localcode.chat.service import _model_family

GONE_GGUF = "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M"


def test_both_builds_of_one_model_share_a_family() -> None:
    assert _model_family(GONE_GGUF) == _model_family(CHAT_KEY_2) == "qwen3-4b-instruct-2507"
    assert _model_family(CHAT_KEY) == "qwen3-0.6b"
    assert _model_family("remote:unlid:glm-5") == ""


async def test_a_deleted_model_is_replaced_by_its_other_build(service: Any, runtime: Any, repo: Any) -> None:
    frames = await collect(service.send({"conversationId": None, "content": "hej", "model": GONE_GGUF}))
    await service.drain()

    assert "error" not in [f["type"] for f in frames]
    assert runtime.acquired[0] == (CHAT_KEY_2, "chat")
    start = next(f for f in frames if f["type"] == "start")
    assert repo.get_conversation(start["conversationId"])["model"] == CHAT_KEY_2


async def test_an_existing_conversation_moves_off_its_deleted_model(service: Any, runtime: Any, repo: Any) -> None:
    conversation = repo.create_conversation({"title": "hej", "provider": "local",
                                             "model": "bartowski/Gone-7B-GGUF/Gone-7B-Q4_K_M"})
    frames = await collect(service.send({"conversationId": conversation["id"], "content": "hej",
                                         "model": "bartowski/Gone-7B-GGUF/Gone-7B-Q4_K_M"}))
    await service.drain()

    assert "error" not in [f["type"] for f in frames]
    assert runtime.acquired[0][0] == CHAT_KEY
    assert repo.get_conversation(conversation["id"])["model"] == CHAT_KEY


async def test_with_no_chat_model_left_the_error_names_the_missing_one(repo: Any, runtime: Any,
                                                                        config: Any) -> None:
    service = make_service(repo, runtime, FakeLibrary([]), config)
    frames = await collect(service.send({"conversationId": None, "content": "hej", "model": GONE_GGUF}))
    await service.drain()

    error = next(f for f in frames if f["type"] == "error")
    assert error["code"] == "model_not_found" and GONE_GGUF in error["message"]
    await service.aclose()
