"""The OpenAI-compatible ``/v1`` surface.

Most tests drive a fake engine (shapes, tools, errors, gating); the ones marked
``real_model`` run the tiny GGUF model through ``llama-server`` end to end.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from conftest import TOKEN
from lc_testing import model_path
from localcode.config import Settings
from localcode.server.app import create_app
from localcode.server.core import Core
from server_fakes import CHAT_KEY, EMBED_KEY, FakeEngine

GGUF_TINY_KEY = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
EMBED_GGUF_KEY = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0"


async def sse_chunks(response: Any) -> list[Any]:
    """``data:`` payloads of an OpenAI stream (``[DONE]`` kept as the string)."""
    out: list[Any] = []
    async for raw in response.content:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


# --------------------------------------------------------------------------- #
# Models and gating
# --------------------------------------------------------------------------- #


async def test_models_listing(client: TestClient) -> None:
    body = await (await client.get("/v1/models")).json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert CHAT_KEY in ids and EMBED_KEY in ids
    entry = next(m for m in body["data"] if m["id"] == CHAT_KEY)
    assert entry["object"] == "model" and entry["owned_by"] == "localcode"
    assert entry["type"] == "llm"
    assert next(m for m in body["data"] if m["id"] == EMBED_KEY)["type"] == "embeddings"


async def test_retrieve_model_by_substring(client: TestClient) -> None:
    body = await (await client.get("/v1/models/Qwen3-0.6B-Q4_K_M")).json()
    assert body["id"] == CHAT_KEY
    missing = await client.get("/v1/models/nie-ma-takiego")
    assert missing.status == 404
    assert (await missing.json())["error"]["code"] == "model_not_found"


async def test_v1_is_open_without_token_by_default(anon_client: TestClient) -> None:
    assert (await anon_client.get("/v1/models")).status == 200


async def test_v1_can_require_a_token(core, anon_client: TestClient) -> None:
    core.settings.server.openai_api_requires_token = True
    response = await anon_client.get("/v1/models")
    assert response.status == 401
    assert (await response.json())["error"]["code"] == "unauthorized"
    ok = await anon_client.get("/v1/models", headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status == 200


async def test_v1_can_be_disabled(core, client: TestClient) -> None:
    core.settings.server.openai_api_enabled = False
    response = await client.get("/v1/models")
    assert response.status == 404
    assert (await response.json())["error"]["code"] == "openai_api_disabled"


async def test_v1_unknown_route_uses_the_openai_error_shape(client: TestClient) -> None:
    response = await client.get("/v1/nie-ma")
    assert response.status == 404
    body = await response.json()
    assert body["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# Chat completions (fake engine)
# --------------------------------------------------------------------------- #


async def test_chat_completion_non_stream(client: TestClient, engine: FakeEngine,
                                          runtime) -> None:
    engine.script = ["Cześć", "! Jak ", "mogę pomóc?"]
    response = await client.post("/v1/chat/completions", json={
        "model": "Qwen3-0.6B-Q4_K_M",
        "messages": [{"role": "system", "content": "Bądź zwięzły."},
                     {"role": "user", "content": "Cześć"}],
        "temperature": 0.2, "max_tokens": 64,
    })
    assert response.status == 200
    body = await response.json()
    assert body["object"] == "chat.completion" and body["model"] == CHAT_KEY
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Cześć! Jak mogę pomóc?"
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert runtime.acquired == [(CHAT_KEY, "chat")]
    assert runtime.active == 0                      # the lease was released
    assert engine.params[0].temperature == pytest.approx(0.2)
    assert engine.params[0].max_tokens == 64
    assert "<|im_end|>" in engine.params[0].stop


async def test_chat_completion_reasoning_is_separate(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["<think>", "muszę pomyśleć", "</think>", "Gotowe."]
    body = await (await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY, "messages": [{"role": "user", "content": "myśl"}]})).json()
    message = body["choices"][0]["message"]
    assert message["content"] == "Gotowe."
    assert message["reasoning_content"] == "muszę pomyśleć"


async def test_chat_completion_tool_calls(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ['<tool_call>{"name": "get_weather", "arguments": {"city": "Kraków"}}</tool_call>']
    body = await (await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY,
        "messages": [{"role": "user", "content": "Pogoda w Krakowie?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Pogoda",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    })).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Kraków"}


async def test_chat_completion_accepts_tool_results(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["W Krakowie jest 20°C."]
    response = await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY,
        "messages": [
            {"role": "user", "content": "Pogoda?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Kraków"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "20"},
        ],
    })
    assert response.status == 200
    assert (await response.json())["choices"][0]["message"]["content"] == "W Krakowie jest 20°C."


async def test_chat_completion_stream(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["Ala ", "ma ", "kota"]
    response = await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY, "messages": [{"role": "user", "content": "Cześć"}],
        "stream": True, "stream_options": {"include_usage": True},
    })
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    chunks = await sse_chunks(response)
    assert chunks[-1] == "[DONE]"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks[:-1])
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks[:-1] if c["choices"])
    assert text == "Ala ma kota"
    assert any(c["choices"] and c["choices"][0]["finish_reason"] == "stop" for c in chunks[:-1])
    usage = next(c["usage"] for c in chunks[:-1] if "usage" in c)
    assert usage["completion_tokens"] > 0


async def test_stream_stops_the_engine_when_the_client_disconnects(
        client: TestClient, engine: FakeEngine, runtime) -> None:
    import asyncio

    engine.script = ["kawałek "] * 200
    engine.delay = 0.02
    response = await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY, "messages": [{"role": "user", "content": "długo"}], "stream": True})
    async for raw in response.content:                 # first chunk is enough
        if raw.decode().startswith("data: "):
            break
    response.close()
    for _ in range(100):
        await asyncio.sleep(0.05)
        if engine.closed and runtime.active == 0:
            break
    assert engine.closed >= 1                          # generation was cancelled
    assert runtime.active == 0                         # and the slot released


async def test_chat_completion_stream_tool_calls(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ['<tool_call>{"name": "ls", "arguments": {}}', "</tool_call>"]
    response = await client.post("/v1/chat/completions", json={
        "model": CHAT_KEY, "messages": [{"role": "user", "content": "pokaż"}], "stream": True,
        "tools": [{"type": "function", "function": {"name": "ls", "parameters": {"type": "object"}}}],
    })
    chunks = await sse_chunks(response)
    calls = [c for c in chunks[:-1] if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert calls and calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "ls"
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    # tool markup never leaks into the visible stream
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks[:-1] if c["choices"])
    assert "<tool_call>" not in text


async def test_chat_completion_errors(client: TestClient) -> None:
    unknown = await client.post("/v1/chat/completions",
                                json={"model": "brak", "messages": [{"role": "user", "content": "x"}]})
    assert unknown.status == 404
    assert (await unknown.json())["error"]["code"] == "model_not_found"

    no_messages = await client.post("/v1/chat/completions", json={"model": CHAT_KEY})
    assert no_messages.status == 400
    assert "messages" in (await no_messages.json())["error"]["message"]

    many = await client.post("/v1/chat/completions",
                             json={"model": CHAT_KEY, "n": 3,
                                   "messages": [{"role": "user", "content": "x"}]})
    assert many.status == 400


async def test_engine_error_is_openai_shaped(client: TestClient, runtime) -> None:
    from localcode.runtime.base import EngineError

    runtime.acquire_error = EngineError("Model nie mieści się w pamięci.")
    response = await client.post("/v1/chat/completions",
                                 json={"model": CHAT_KEY,
                                       "messages": [{"role": "user", "content": "x"}]})
    assert response.status == 503
    body = await response.json()
    assert body["error"]["type"] == "server_error"
    assert "pamięci" in body["error"]["message"]


# --------------------------------------------------------------------------- #
# Completions and embeddings (fake engine)
# --------------------------------------------------------------------------- #


async def test_text_completion(client: TestClient, engine: FakeEngine) -> None:
    engine.script = [" dalszy ciąg"]
    body = await (await client.post("/v1/completions",
                                    json={"model": CHAT_KEY, "prompt": "Był sobie",
                                          "max_tokens": 8})).json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " dalszy ciąg"
    assert engine.prompts[-1] == "Był sobie"          # no chat template


async def test_text_completion_stream(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["a", "b"]
    chunks = await sse_chunks(await client.post(
        "/v1/completions", json={"model": CHAT_KEY, "prompt": "x", "stream": True}))
    assert chunks[-1] == "[DONE]"
    assert "".join(c["choices"][0]["text"] for c in chunks[:-1]) == "ab"


async def test_embeddings(client: TestClient, runtime) -> None:
    body = await (await client.post("/v1/embeddings",
                                    json={"model": EMBED_KEY,
                                          "input": ["kot", "pies"]})).json()
    assert body["object"] == "list" and body["model"] == EMBED_KEY
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert len(body["data"][0]["embedding"]) > 3
    assert runtime.acquired == [(EMBED_KEY, "embedding")]
    assert runtime.active == 0


async def test_embeddings_validation(client: TestClient) -> None:
    response = await client.post("/v1/embeddings", json={"model": EMBED_KEY, "input": []})
    assert response.status == 400
    assert "input" in (await response.json())["error"]["message"]


# --------------------------------------------------------------------------- #
# Real model
# --------------------------------------------------------------------------- #


def _llama_server_available() -> bool:
    return bool(shutil.which("llama-server")) or Path("/opt/homebrew/bin/llama-server").is_file()


@pytest.fixture
async def real_client(tmp_path: Path, isolated_home: Path):
    """A server backed by the real library and runtime (spawns llama-server)."""
    if not _llama_server_available():
        pytest.skip("llama-server not installed")
    settings = Settings(models_dir=str(Path("~/.localcode/models").expanduser()),
                        scan_lmstudio=False, scan_hf_cache=False)
    settings.runtime.max_loaded_models = 1
    settings.runtime.default_ctx = 2048
    settings.runtime.idle_ttl_minutes = 0
    core = Core(settings, db_path=tmp_path / "chat.db")
    await core.start()
    client = TestClient(TestServer(create_app(core, TOKEN)),
                        headers={"Authorization": f"Bearer {TOKEN}"})
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()
        await core.shutdown()


@pytest.mark.real_model
async def test_real_model_chat_and_completions(real_client: TestClient) -> None:
    model_path("gguf_tiny")
    listed = await (await real_client.get("/v1/models")).json()
    assert GGUF_TINY_KEY in [m["id"] for m in listed["data"]]

    response = await real_client.post("/v1/chat/completions", json={
        "model": "Qwen3-0.6B-Q4_K_M",
        "messages": [{"role": "user", "content": "Odpowiedz jednym słowem: stolica Polski?"}],
        "max_tokens": 48, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=180)
    assert response.status == 200, await response.text()
    body = await response.json()
    assert body["model"] == GGUF_TINY_KEY
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0
    assert body["choices"][0]["finish_reason"] in ("stop", "length")

    streamed = await real_client.post("/v1/chat/completions", json={
        "model": GGUF_TINY_KEY,
        "messages": [{"role": "user", "content": "Policz od 1 do 3."}],
        "max_tokens": 32, "temperature": 0.0, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=180)
    chunks = await sse_chunks(streamed)
    assert chunks[-1] == "[DONE]"
    text = "".join(c["choices"][0]["delta"].get("content", "")
                   for c in chunks[:-1] if c["choices"])
    assert text.strip()

    raw = await real_client.post("/v1/completions", json={
        "model": GGUF_TINY_KEY, "prompt": "Stolica Polski to", "max_tokens": 12,
        "temperature": 0.0}, timeout=180)
    assert raw.status == 200
    assert (await raw.json())["choices"][0]["text"]


@pytest.mark.real_model
async def test_real_model_embeddings(real_client: TestClient) -> None:
    model_path("gguf_embed")
    response = await real_client.post("/v1/embeddings", json={
        "model": "nomic-embed-text-v1.5.Q8_0", "input": ["Ala ma kota", "Pies goni kota"]},
        timeout=180)
    assert response.status == 200, await response.text()
    body = await response.json()
    vectors = [d["embedding"] for d in body["data"]]
    assert len(vectors) == 2 and len(vectors[0]) >= 256
    assert all(isinstance(v, float) for v in vectors[0][:8])
