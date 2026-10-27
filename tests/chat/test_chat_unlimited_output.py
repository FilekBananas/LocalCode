"""``maxTokens: 0`` ("bez limitu") lets an answer use all the window has left."""

from __future__ import annotations

from typing import Any

from chat_fakes import collect


async def test_no_limit_asks_for_the_whole_free_window(service: Any, engine: Any) -> None:
    await collect(service.send({"conversationId": None, "content": "hej", "maxTokens": 0}))
    await service.drain()

    asked = engine.params[0].max_tokens
    assert asked > engine.n_ctx // 2                 # far beyond the default reserve of a quarter
    assert asked < engine.n_ctx                      # the prompt still fits


async def test_a_set_limit_is_still_kept(service: Any, engine: Any) -> None:
    await collect(service.send({"conversationId": None, "content": "hej", "maxTokens": 300}))
    await service.drain()

    assert engine.params[0].max_tokens == 300
