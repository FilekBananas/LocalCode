"""Tests for ``localcode.runtime.mlx_worker``.

Two layers:

* Fast unit tests (no model): request parsing, stop-string matching across
  token boundaries, the incremental detokenizer, message normalisation and the
  whole HTTP/SSE layer driven by a fake generation service.
* ``real_model`` tests: the worker is spawned as a subprocess on a free port
  with the tiny MLX model (Qwen3-0.6B-4bit) and exercised over HTTP exactly
  like ``HttpEngine`` does.  Key sets / renderings recorded from llama-server
  build 10964 are asserted, and an optional parity test runs llama-server
  side by side.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from lc_testing import TEST_MODELS, model_path
from localcode.runtime import mlx_worker as mw

REPO_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# llama-server (build 10964) reference data, recorded with the tiny Qwen3 GGUF
# --------------------------------------------------------------------------- #

LLAMA_STREAM_FRAME_KEYS = {
    "index", "content", "tokens", "stop", "id_slot", "tokens_predicted", "tokens_evaluated",
}
LLAMA_FINAL_KEYS = {
    "index", "content", "tokens", "id_slot", "stop", "model", "tokens_predicted", "tokens_evaluated",
    "generation_settings", "prompt", "has_new_line", "truncated", "stop_type", "stopping_word",
    "tokens_cached", "timings",
}
LLAMA_TIMINGS_KEYS = {
    "cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
}
LLAMA_PROPS_KEYS = {
    "default_generation_settings", "total_slots", "model_alias", "model_path", "modalities",
    "endpoint_slots", "endpoint_props", "endpoint_metrics", "chat_template", "chat_template_caps",
    "bos_token", "eos_token", "build_info", "is_sleeping",
}
LLAMA_TEMPLATE_CAPS_KEYS = {
    "supports_object_arguments", "supports_parallel_tool_calls", "supports_preserve_reasoning",
    "supports_reasoning_effort", "supports_string_content", "supports_system_role",
    "supports_tool_calls", "supports_tools", "supports_typed_content",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOL_CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "{\"temp\": 20}"},
]
# llama-server /apply-template output for TOOL_CONVERSATION + [WEATHER_TOOL] with
# chat_template_kwargs {"enable_thinking": false} (Qwen3 template).
LLAMA_TOOL_PROMPT = (
    "<|im_start|>system\nYou are helpful.\n\n# Tools\n\nYou may call one or more functions to assist "
    "with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": "
    "\"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": "
    "\"string\"}}, \"required\": [\"city\"]}}}\n</tools>\n\nFor each function call, return a json object "
    "with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
    "<|im_start|>user\nWhat is the weather in Paris?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n"
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n{\"temp\": 20}\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def chat_prompt(user: str, thinking: bool = False) -> str:
    """Qwen3 chat prompt for a single user turn (thinking disabled by default)."""
    tail = "" if thinking else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{tail}"


# --------------------------------------------------------------------------- #
# Unit tests: parameters, helpers
# --------------------------------------------------------------------------- #


def test_parse_completion_params_defaults_match_llama_server() -> None:
    p = mw.parse_completion_params({"prompt": "hi"})
    assert p.prompt == "hi"
    assert p.n_predict == -1
    assert (p.temperature, p.top_k, p.top_p, p.min_p) == (0.8, 40, 0.95, 0.05)
    assert p.repeat_penalty == 1.0 and p.repeat_last_n == 64
    assert p.stream is False and p.cache_prompt is True and p.seed is None
    assert p.stop == []


def test_parse_completion_params_aliases_and_normalisation() -> None:
    p = mw.parse_completion_params(
        {"prompt": ["only one"], "max_tokens": 12, "stop": "END", "seed": 42, "temperature": 0,
         "logit_bias": [[5, -1.5], [7, False]], "stream": True}
    )
    assert p.prompt == "only one"
    assert p.n_predict == 12
    assert p.stop == ["END"]
    assert p.seed == 42
    assert p.temperature == 0.0
    assert p.logit_bias[5] == -1.5 and p.logit_bias[7] == float("-inf")
    assert mw.parse_completion_params({"prompt": [1, 2, 3]}).prompt == [1, 2, 3]
    assert mw.parse_completion_params({"prompt": "x", "seed": -1}).seed is None
    assert mw.parse_completion_params({"prompt": "x", "seed": 4294967295}).seed is None
    assert mw.parse_completion_params({"prompt": "x", "n_predict": 3, "max_tokens": 9}).n_predict == 3
    assert mw.parse_completion_params({"prompt": "x", "logit_bias": {"11": 2}}).logit_bias == {11: 2.0}


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({}, "'prompt' is required"),
        ({"prompt": 5}, "type must be string"),
        ({"prompt": ["a", "b"]}, "single prompt"),
        ({"prompt": "x", "temperature": "hot"}, "Field 'temperature': type must be number, but is string"),
        ({"prompt": "x", "n_predict": 1.5}, "must be an integer"),
        ({"prompt": "x", "stream": "yes"}, "type must be boolean"),
        ({"prompt": "x", "stop": [1]}, "array of strings"),
        ({"prompt": "x", "top_k": True}, "type must be number, but is boolean"),
        ({"prompt": "x", "logit_bias": [[1]]}, "logit_bias"),
        ([1, 2], "JSON object"),
    ],
)
def test_parse_completion_params_errors(body: Any, fragment: str) -> None:
    with pytest.raises(mw.ApiError) as ei:
        mw.parse_completion_params(body)
    assert ei.value.status == 400
    assert ei.value.err_type == "invalid_request_error"
    assert fragment in ei.value.message


def test_plan_generation_mirrors_llama_server() -> None:
    assert mw.plan_generation(10, 5, 100) == (5, False)
    assert mw.plan_generation(10, -1, 100) == (90, True)
    assert mw.plan_generation(90, 50, 100) == (10, True)       # truncated at the window end
    assert mw.plan_generation(10, 0, 100) == (1, False)        # n_predict 0 still yields a token
    assert mw.plan_generation(99, 5, 100) == (1, True)


def test_error_shapes() -> None:
    err = mw.exceed_context(1101, 1024).to_json()
    assert err == {
        "error": {
            "code": 400,
            "message": "request (1101 tokens) exceeds the available context size (1024 tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 1101,
            "n_ctx": 1024,
        }
    }
    assert mw.unavailable("Loading model").to_json() == {
        "error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}
    }


def test_build_timings_keys_and_rates() -> None:
    t = mw.build_timings(3, 21, 250.0, 30, 600.0)
    assert set(t) == LLAMA_TIMINGS_KEYS
    assert t["prompt_per_second"] == pytest.approx(84.0)
    assert t["predicted_per_token_ms"] == pytest.approx(20.0)
    zero = mw.build_timings(0, 0, 0.0, 0, 0.0)
    assert zero["prompt_per_second"] == 0.0 and zero["predicted_per_second"] == 0.0


# --------------------------------------------------------------------------- #
# Unit tests: stop strings and detokenisation
# --------------------------------------------------------------------------- #


def _run_stops(stops: list[str], deltas: list[str]) -> tuple[str, str | None]:
    m = mw.StopMatcher(stops)
    out = []
    for d in deltas:
        emit, matched = m.feed(d)
        out.append(emit)
        if matched is not None:
            return "".join(out), matched
    out.append(m.flush())
    return "".join(out), None


def test_stop_matcher_across_token_boundaries() -> None:
    assert _run_stops(["5, 6"], ["1, 2, 3, 4, ", "5", ",", " ", "6", ", 7"]) == ("1, 2, 3, 4, ", "5, 6")
    assert _run_stops(["</end>"], ["abc</", "en", "d>tail"]) == ("abc", "</end>")


def test_stop_matcher_holds_back_partial_and_releases_false_alarm() -> None:
    m = mw.StopMatcher(["STOP"])
    assert m.feed("hello ST") == ("hello ", None)      # "ST" may start "STOP"
    assert m.feed("ART") == ("START", None)            # false alarm: released
    assert m.feed("S") == ("", None)
    assert m.flush() == "S"                            # generation ended: flush


def test_stop_matcher_earliest_and_longest() -> None:
    assert _run_stops(["b", "abc"], ["xxabcd"]) == ("xx", "abc")
    assert _run_stops(["cd", "b"], ["abcd"]) == ("a", "b")
    assert _run_stops(["\n\n"], ["line\n", "\nnext"]) == ("line", "\n\n")
    assert _run_stops([], ["free text"]) == ("free text", None)
    assert _run_stops(["x"], ["x"]) == ("", "x")


def _byte_decode(ids: list[int]) -> str:
    """Toy tokenizer: every token is one UTF-8 byte."""
    return bytes(ids).decode("utf-8", errors="replace")


def test_incremental_decoder_holds_incomplete_utf8() -> None:
    text = "Zażółć 😊!"
    ids = list(text.encode("utf-8"))
    dec = mw.IncrementalDecoder(_byte_decode, context=list(b"prompt: "))
    pieces = [dec.add(t) for t in ids]
    pieces.append(dec.flush())
    assert "".join(pieces) == text
    assert all("\ufffd" not in p for p in pieces)
    # the 4-byte emoji is emitted at once, not byte by byte
    assert "😊" in pieces


def test_incremental_decoder_flushes_invalid_tail() -> None:
    dec = mw.IncrementalDecoder(_byte_decode)
    assert dec.add(0xF0) == ""
    assert dec.flush() == "\ufffd"
    assert dec.flush() == ""


# --------------------------------------------------------------------------- #
# Unit tests: chat messages
# --------------------------------------------------------------------------- #


def test_normalize_messages_content_and_tool_calls() -> None:
    raw = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]},
        {"role": "tool", "content": {"ok": True}},
    ]
    out = mw.normalize_messages(raw)
    assert out[0]["content"] == "ab"
    assert out[1]["content"] == ""
    assert out[1]["tool_calls"][0]["type"] == "function"
    assert out[2]["content"] == '{"ok":true}'
    assert raw[1]["content"] is None  # caller data untouched


@pytest.mark.parametrize(
    "messages, fragment",
    [
        (None, "'messages' is required"),
        ([], "non-empty"),
        ([{"content": "x"}], "'role' is required"),
        ([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}], "Only text content"),
        ([{"role": "assistant", "tool_calls": [{"function": {}}]}], "'name' is required"),
    ],
)
def test_normalize_messages_errors(messages: Any, fragment: str) -> None:
    with pytest.raises(mw.ApiError) as ei:
        mw.normalize_messages(messages)
    assert fragment in ei.value.message


def test_convert_tool_arguments_both_ways() -> None:
    msgs = mw.normalize_messages(
        [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": {"city": "Kraków"}}},
            {"function": {"name": "g", "arguments": "{\"a\": 1}"}},
            {"function": {"name": "h", "arguments": "not json"}},
        ]}]
    )
    as_str = mw.convert_tool_arguments(msgs, as_object=False)
    args = [c["function"]["arguments"] for c in as_str[0]["tool_calls"]]
    assert args == ['{"city":"Kraków"}', '{"a": 1}', "not json"]
    as_obj = mw.convert_tool_arguments(msgs, as_object=True)
    args = [c["function"]["arguments"] for c in as_obj[0]["tool_calls"]]
    assert args == [{"city": "Kraków"}, {"a": 1}, "not json"]
    # originals untouched
    assert msgs[0]["tool_calls"][1]["function"]["arguments"] == "{\"a\": 1}"


def test_cli_rejects_missing_model_dir(tmp_path: Path) -> None:
    assert mw.main(["--model", str(tmp_path / "nope"), "--port", "0"]) == 2
    assert mw.main(["--model", str(tmp_path), "--port", "0", "--parallel", "0"]) == 2


# --------------------------------------------------------------------------- #
# HTTP layer with a fake generation service
# --------------------------------------------------------------------------- #


class FakeService:
    """Implements ``GenerationService``; "tokens" are whitespace-separated words."""

    model_path = "/models/fake"

    def __init__(self, n_ctx: int = 64, words: list[str] | None = None, delay: float = 0.0,
                 fail_first: mw.ApiError | None = None) -> None:
        self.state = "ready"
        self.n_ctx = n_ctx
        self.words = words or ["Hello", " world", "!"]
        self.delay = delay
        self.fail_first = fail_first
        self.submitted: list[mw.Sequence] = []
        self.cancelled: list[mw.Sequence] = []
        self._ids = 0

    def props(self) -> dict[str, Any]:
        return {"default_generation_settings": {"n_ctx": self.n_ctx}, "total_slots": 1}

    def encode_prompt(self, prompt: str | list[int]) -> tuple[list[int], str]:
        if isinstance(prompt, list):
            return list(prompt), " ".join(map(str, prompt))
        return [len(w) for w in prompt.split()], prompt

    def tokenize(self, content: str, add_special: bool, parse_special: bool) -> list[int]:
        return [len(w) for w in content.split()]

    def token_pieces(self, tokens: list[int]) -> list[dict[str, Any]]:
        return [{"id": t, "piece": "x" * t} for t in tokens]

    def detokenize(self, tokens: list[int]) -> str:
        return " ".join("x" * t for t in tokens)

    def apply_template(self, messages: Any, tools: Any, add_generation_prompt: bool,
                       template_kwargs: dict[str, Any]) -> str:
        msgs = mw.normalize_messages(messages)
        text = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        return text + ("<assistant>" if add_generation_prompt else "")

    def new_task_id(self) -> int:
        self._ids += 1
        return self._ids

    def submit(self, seq: mw.Sequence) -> None:
        self.submitted.append(seq)
        threading.Thread(target=self._produce, args=(seq,), daemon=True).start()

    def _produce(self, seq: mw.Sequence) -> None:
        if self.fail_first is not None:
            seq.sink.put(("error", self.fail_first))
            return
        n = 0
        for i in range(seq.max_tokens):
            if seq.cancelled:
                return
            if self.delay:
                time.sleep(self.delay)
            word = self.words[i % len(self.words)]
            n += 1
            seq.sink.put(("token", word, 100 + i, n))
        seq.finished = True
        seq.sink.put(("done", {
            "id_slot": 0, "tokens_predicted": n, "tokens_evaluated": len(seq.prompt_tokens),
            "has_new_line": False, "truncated": seq.ctx_capped, "stop_type": "limit",
            "stopping_word": "", "tokens_cached": len(seq.prompt_tokens) + n,
            "timings": mw.build_timings(0, len(seq.prompt_tokens), 1.0, n, 2.0), "content_tail": "",
        }))

    def cancel(self, seq: mw.Sequence) -> None:
        seq.cancelled = True
        self.cancelled.append(seq)

    def slots(self) -> list[dict[str, Any]]:
        return [{"id": 0, "n_ctx": self.n_ctx, "is_processing": False}]

    def memory(self) -> dict[str, Any]:
        return {"active": 1, "peak": 2, "cache": 3}


class _Http:
    def __init__(self, service: FakeService) -> None:
        self.service = service
        self.http = mw.WorkerHttp(service)
        self.server = TestServer(self.http.make_app())
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "_Http":
        await self.server.start_server()  # TestServer enables handler_cancellation
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self.session is not None
        await self.session.close()
        await self.server.close()
        self.http.close()

    def url(self, path: str) -> str:
        return str(self.server.make_url(path))


async def _read_sse(resp: aiohttp.ClientResponse) -> list[dict[str, Any]]:
    frames = []
    async for line in resp.content:
        if line.startswith(b"data: "):
            frames.append(json.loads(line[6:]))
    return frames


async def test_http_health_loading_then_ready() -> None:
    svc = FakeService()
    svc.state = "loading"
    async with _Http(svc) as h:
        for path in ("/health", "/props", "/slots"):
            async with h.session.get(h.url(path)) as r:
                assert r.status == 503
                assert await r.json() == {"error": {"code": 503, "message": "Loading model",
                                                    "type": "unavailable_error"}}
        svc.state = "ready"
        async with h.session.get(h.url("/health")) as r:
            assert r.status == 200 and await r.json() == {"status": "ok"}


async def test_http_completion_stream_frames_have_llama_shape() -> None:
    svc = FakeService()
    async with _Http(svc) as h:
        async with h.session.post(h.url("/completion"),
                                  json={"prompt": "a bb ccc", "n_predict": 3, "stream": True}) as r:
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/event-stream")
            frames = await _read_sse(r)
    assert [f["content"] for f in frames[:-1]] == ["Hello", " world", "!"]
    for f in frames[:-1]:
        assert set(f) == LLAMA_STREAM_FRAME_KEYS
        assert f["stop"] is False and f["id_slot"] == -1 and f["tokens_evaluated"] == 3
    final = frames[-1]
    assert set(final) == LLAMA_FINAL_KEYS
    assert final["stop"] is True and final["stop_type"] == "limit"
    assert set(final["timings"]) == LLAMA_TIMINGS_KEYS
    assert final["prompt"] == "a bb ccc" and final["model"] == "/models/fake"


async def test_http_completion_non_stream_accumulates_content() -> None:
    svc = FakeService()
    async with _Http(svc) as h:
        async with h.session.post(h.url("/completion"), json={"prompt": "a bb", "n_predict": 4}) as r:
            assert r.status == 200
            body = await r.json()
    assert set(body) == LLAMA_FINAL_KEYS
    assert body["content"] == "Hello world!Hello"
    assert body["tokens_predicted"] == 4


async def test_http_context_overflow_is_400_before_streaming() -> None:
    svc = FakeService(n_ctx=4)
    async with _Http(svc) as h:
        for stream in (False, True):
            async with h.session.post(h.url("/completion"),
                                      json={"prompt": "a b c d", "stream": stream}) as r:
                assert r.status == 400
                err = (await r.json())["error"]
                assert err["type"] == "exceed_context_size_error"
                assert err["n_prompt_tokens"] == 4 and err["n_ctx"] == 4
    assert svc.submitted == []


async def test_http_generation_budget_capped_by_window() -> None:
    svc = FakeService(n_ctx=10)
    async with _Http(svc) as h:
        async with h.session.post(h.url("/completion"), json={"prompt": "a b c d e f g", "n_predict": 50}) as r:
            body = await r.json()
    assert body["tokens_predicted"] == 3 and body["truncated"] is True


async def test_http_error_before_first_token_is_plain_http_error() -> None:
    svc = FakeService(fail_first=mw.unavailable("Server is shutting down"))
    async with _Http(svc) as h:
        async with h.session.post(h.url("/completion"), json={"prompt": "a", "stream": True}) as r:
            assert r.status == 503
            assert (await r.json())["error"]["type"] == "unavailable_error"


async def test_http_client_disconnect_cancels_generation() -> None:
    svc = FakeService(delay=0.02)
    async with _Http(svc) as h:
        resp = await h.session.post(h.url("/completion"),
                                    json={"prompt": "a", "n_predict": 10_000, "stream": True})
        got = 0
        async for line in resp.content:
            if line.startswith(b"data: "):
                got += 1
            if got >= 3:
                break
        resp.close()
        for _ in range(100):
            if svc.cancelled:
                break
            await asyncio.sleep(0.02)
    assert svc.cancelled and svc.cancelled[0] is svc.submitted[0]


async def test_http_empty_prompt_returns_stop_type_none() -> None:
    svc = FakeService()
    async with _Http(svc) as h:
        async with h.session.post(h.url("/completion"), json={"prompt": ""}) as r:
            body = await r.json()
    assert r.status == 200 and body["stop_type"] == "none" and body["tokens_predicted"] == 0


async def test_http_misc_endpoints_and_errors() -> None:
    svc = FakeService()
    async with _Http(svc) as h:
        s = h.session
        async with s.post(h.url("/tokenize"), json={"content": "ab cde"}) as r:
            assert await r.json() == {"tokens": [2, 3]}
        async with s.post(h.url("/tokenize"), json={"content": "ab", "with_pieces": True}) as r:
            assert await r.json() == {"tokens": [{"id": 2, "piece": "xx"}]}
        async with s.post(h.url("/tokenize"), json={}) as r:
            assert await r.json() == {"tokens": []}
        async with s.post(h.url("/detokenize"), json={"tokens": [1, 2]}) as r:
            assert await r.json() == {"content": "x xx"}
        async with s.post(h.url("/apply-template"), json={"messages": [{"role": "user", "content": "hi"}]}) as r:
            assert await r.json() == {"prompt": "<user>hi<assistant>"}
        async with s.post(h.url("/apply-template"), json={"foo": 1}) as r:
            assert r.status == 400 and (await r.json())["error"]["message"] == "'messages' is required"
        for path in ("/embedding", "/embeddings", "/v1/embeddings"):
            async with s.post(h.url(path), json={"content": "x"}) as r:
                assert r.status == 501
                assert (await r.json())["error"]["type"] == "not_supported_error"
        async with s.get(h.url("/nope")) as r:
            assert r.status == 404
            assert (await r.json())["error"]["type"] == "not_found_error"
        async with s.post(h.url("/completion"), data=b"not json",
                          headers={"Content-Type": "application/json"}) as r:
            assert r.status == 400
            assert (await r.json())["error"]["type"] == "invalid_request_error"
        async with s.get(h.url("/memory")) as r:
            assert await r.json() == {"active": 1, "peak": 2, "cache": 3}


# --------------------------------------------------------------------------- #
# Real model: worker subprocess helpers
# --------------------------------------------------------------------------- #


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_json(method: str, url: str, body: Any = None, timeout: float = 30) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw.decode(errors="replace")


class WorkerProc:
    """The MLX worker as a subprocess (stderr captured to a log file)."""

    def __init__(self, model: Path, log_path: Path, *extra: str, parallel: int = 2, ctx: int = 2048) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_path
        self._log = open(log_path, "wb")
        cmd = [sys.executable, "-m", "localcode.runtime.mlx_worker", "--model", str(model),
               "--host", "127.0.0.1", "--port", str(self.port), "--parallel", str(parallel),
               "--ctx", str(ctx), *extra]
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=self._log)
        self.health_history: list[tuple[int, Any]] = []

    def log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self, timeout: float = 180.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited with {self.proc.returncode}:\n{self.log_tail()}")
            try:
                status, body = _http_json("GET", self.url + "/health", timeout=2)
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.05)
                continue
            if not self.health_history or self.health_history[-1][0] != status:
                self.health_history.append((status, body))
            if status == 200:
                return
            time.sleep(0.05)
        raise TimeoutError(f"worker not ready after {timeout}s:\n{self.log_tail()}")

    def stop(self) -> int | None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()
        return self.proc.returncode


@pytest.fixture(scope="module")
def worker(tmp_path_factory: pytest.TempPathFactory):
    path = model_path("mlx_tiny")
    w = WorkerProc(path, tmp_path_factory.mktemp("mlxw") / "worker.log", parallel=2, ctx=2048)
    try:
        w.wait_ready()
        yield w
    finally:
        w.stop()


async def _post(session: aiohttp.ClientSession, url: str, body: dict[str, Any]) -> tuple[int, Any]:
    async with session.post(url, json=body) as r:
        return r.status, await r.json()


async def _stream(session: aiohttp.ClientSession, url: str, body: dict[str, Any],
                  events: list[tuple[str, float]] | None = None, label: str = "") -> list[dict[str, Any]]:
    frames = []
    async with session.post(url + "/completion", json={**body, "stream": True}) as r:
        assert r.status == 200, await r.text()
        async for line in r.content:
            if line.startswith(b"data: "):
                frame = json.loads(line[6:])
                frames.append(frame)
                if events is not None:
                    events.append((label, time.perf_counter()))
    return frames


# --------------------------------------------------------------------------- #
# Real model tests
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
def test_health_transitions_from_503_to_200(worker: WorkerProc) -> None:
    statuses = [s for s, _ in worker.health_history]
    assert statuses[-1] == 200
    assert worker.health_history[-1][1] == {"status": "ok"}
    if len(statuses) > 1:  # observed while loading
        assert statuses[0] == 503
        assert worker.health_history[0][1] == {
            "error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}
        }


@pytest.mark.real_model
def test_props_shape(worker: WorkerProc) -> None:
    status, props = _http_json("GET", worker.url + "/props")
    assert status == 200
    assert LLAMA_PROPS_KEYS <= set(props)
    assert props["default_generation_settings"]["n_ctx"] == 2048
    assert props["n_ctx"] == 2048
    assert props["total_slots"] == 2
    assert "<|im_start|>" in props["chat_template"]
    assert LLAMA_TEMPLATE_CAPS_KEYS <= set(props["chat_template_caps"])
    caps = props["chat_template_caps"]
    assert caps["supports_tools"] and caps["supports_tool_calls"] and caps["supports_system_role"]
    assert caps["supports_tool_role"]
    assert props["modalities"] == {"vision": False, "video": False, "audio": False}
    assert props["eos_token"] == "<|im_end|>" and props["bos_token"] == ""
    assert Path(props["model_path"]) == TEST_MODELS["mlx_tiny"].resolve()
    assert props["engine"] == "mlx" and props["batching"] is True


@pytest.mark.real_model
def test_apply_template_with_tools_matches_llama_server(worker: WorkerProc) -> None:
    body = {"messages": TOOL_CONVERSATION, "tools": [WEATHER_TOOL], "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    status, out = _http_json("POST", worker.url + "/apply-template", body)
    assert status == 200
    assert out["prompt"] == LLAMA_TOOL_PROMPT
    # arguments given as a JSON string render identically
    string_args = json.loads(json.dumps(TOOL_CONVERSATION))
    string_args[2]["tool_calls"][0]["function"]["arguments"] = "{\"city\": \"Paris\"}"
    status, out2 = _http_json("POST", worker.url + "/apply-template", {**body, "messages": string_args})
    assert status == 200 and out2["prompt"] == LLAMA_TOOL_PROMPT


@pytest.mark.real_model
def test_apply_template_prefill_and_errors_like_llama_server(worker: WorkerProc) -> None:
    url = worker.url + "/apply-template"
    _, out = _http_json("POST", url, {"messages": [{"role": "user", "content": "Hi"}]})
    assert out["prompt"] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    _, out = _http_json("POST", url, {"messages": [{"role": "user", "content": "Hi"}],
                                      "add_generation_prompt": False})
    assert out["prompt"] == "<|im_start|>user\nHi<|im_end|>\n"
    prefill = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there"}]
    _, out = _http_json("POST", url, {"messages": prefill, "add_generation_prompt": False})
    assert out["prompt"] == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello there"
    status, err = _http_json("POST", url, {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}]}]})
    assert status == 400
    assert err["error"] == {"code": 400, "message": "Cannot continue an assistant message that contains tool calls.",
                            "type": "invalid_request_error"}
    status, err = _http_json("POST", url, {"foo": 1})
    assert status == 400 and err["error"]["message"] == "'messages' is required"


@pytest.mark.real_model
def test_tokenize_matches_llama_server(worker: WorkerProc) -> None:
    url = worker.url + "/tokenize"
    assert _http_json("POST", url, {"content": "Hello world <|im_start|>"})[1] == {"tokens": [9707, 1879, 220, 151644]}
    assert _http_json("POST", url, {"content": "Hello world <|im_start|>", "add_special": True})[1] == {
        "tokens": [9707, 1879, 220, 151644]}
    assert _http_json("POST", url, {"content": "Hello world <|im_start|>", "parse_special": False})[1] == {
        "tokens": [9707, 1879, 82639, 318, 4906, 91, 29]}
    assert _http_json("POST", url, {"content": "Hi", "with_pieces": True})[1] == {"tokens": [{"id": 13048, "piece": "Hi"}]}
    assert _http_json("POST", url, {})[1] == {"tokens": []}
    assert _http_json("POST", worker.url + "/detokenize", {"tokens": [9707, 1879]})[1] == {"content": "Hello world"}


@pytest.mark.real_model
async def test_streaming_completion_shape_and_eos(worker: WorkerProc) -> None:
    body = {"prompt": chat_prompt("Say hi in three words."), "n_predict": 200, "temperature": 0,
            "cache_prompt": True}
    async with aiohttp.ClientSession() as s:
        frames = await _stream(s, worker.url, body)
        status, full = await _post(s, worker.url + "/completion", {**body, "stream": False})
    assert status == 200
    *tokens, final = frames
    assert tokens, "no token frames"
    for f in tokens:
        assert set(f) == LLAMA_STREAM_FRAME_KEYS
        assert f["stop"] is False and f["id_slot"] == -1 and len(f["tokens"]) == 1
    assert [f["tokens_predicted"] for f in tokens] == list(range(1, len(tokens) + 1))
    assert set(final) == LLAMA_FINAL_KEYS
    assert set(final["timings"]) == LLAMA_TIMINGS_KEYS
    assert final["stop"] is True and final["stop_type"] == "eos" and final["truncated"] is False
    assert final["tokens"] == [] and final["stopping_word"] == ""
    # llama-server style: the EOS token gets its own empty frame before the final one
    assert tokens[-1]["content"] == "" and tokens[-1]["tokens"] == [151645]
    n_prompt = final["tokens_evaluated"]
    assert final["tokens_predicted"] == len(tokens)
    assert final["timings"]["prompt_n"] + final["timings"]["cache_n"] == n_prompt
    assert final["timings"]["predicted_n"] == final["tokens_predicted"]
    streamed = "".join(f["content"] for f in frames)
    assert streamed.strip() and "<|im_end|>" not in streamed
    # greedy: identical text with stream=false (served from the prefix cache)
    assert set(full) == LLAMA_FINAL_KEYS
    assert full["content"] == streamed
    assert full["timings"]["cache_n"] == n_prompt - 1


@pytest.mark.real_model
async def test_stop_string_across_token_boundary_is_never_emitted(worker: WorkerProc) -> None:
    body = {"prompt": chat_prompt("Count from 1 to 20 separated by commas."), "n_predict": 80, "temperature": 0}
    async with aiohttp.ClientSession() as s:
        _, ref = await _post(s, worker.url + "/completion", body)
        text = ref["content"]
        idx = text.find("5, 6")
        stop = "5, 6" if idx > 0 else text[len(text) // 2: len(text) // 2 + 4]
        idx = text.find(stop)
        assert idx > 0
        frames = await _stream(s, worker.url, {**body, "stop": [stop, "never-ever"]})
        _, full = await _post(s, worker.url + "/completion", {**body, "stop": [stop]})
    streamed = "".join(f["content"] for f in frames)
    assert streamed == text[:idx]
    assert stop not in streamed
    final = frames[-1]
    assert final["stop_type"] == "word" and final["stopping_word"] == stop
    assert full["content"] == text[:idx] and full["stop_type"] == "word"


@pytest.mark.real_model
async def test_limit_and_window_truncation(worker: WorkerProc) -> None:
    async with aiohttp.ClientSession() as s:
        _, out = await _post(s, worker.url + "/completion",
                             {"prompt": chat_prompt("Write a long poem."), "n_predict": 5, "temperature": 0})
        assert out["stop_type"] == "limit" and out["tokens_predicted"] == 5 and out["truncated"] is False
        # prompt fits, prompt + n_predict does not: generate until the window is full
        _, out = await _post(s, worker.url + "/completion",
                             {"prompt": [9707] * 2040, "n_predict": 100, "temperature": 0, "ignore_eos": True})
    assert out["stop_type"] == "limit" and out["truncated"] is True
    assert out["tokens_predicted"] == 2048 - 2040


@pytest.mark.real_model
async def test_context_overflow_is_400(worker: WorkerProc) -> None:
    async with aiohttp.ClientSession() as s:
        for stream in (False, True):
            async with s.post(worker.url + "/completion",
                              json={"prompt": [9707] * 2048, "n_predict": 4, "stream": stream}) as r:
                assert r.status == 400
                assert r.headers["Content-Type"].startswith("application/json")
                err = (await r.json())["error"]
        assert err == {
            "code": 400,
            "message": "request (2048 tokens) exceeds the available context size (2048 tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 2048,
            "n_ctx": 2048,
        }


@pytest.mark.real_model
async def test_prefix_cache_reuses_previous_turn(worker: WorkerProc) -> None:
    system = "You are a terse assistant. " + "Always answer briefly. " * 20
    first = f"<|im_start|>system\n{system}<|im_end|>\n" + chat_prompt("Name a colour.")
    async with aiohttp.ClientSession() as s:
        _, a = await _post(s, worker.url + "/completion", {"prompt": first, "n_predict": 12, "temperature": 0})
        n_first = a["tokens_evaluated"]
        follow = first + a["content"] + "<|im_end|>\n<|im_start|>user\nAnother one.<|im_end|>\n<|im_start|>assistant\n"
        _, b = await _post(s, worker.url + "/completion", {"prompt": follow, "n_predict": 8, "temperature": 0})
        _, c = await _post(s, worker.url + "/completion",
                           {"prompt": follow, "n_predict": 8, "temperature": 0, "cache_prompt": False})
    tb = b["timings"]
    assert tb["cache_n"] >= n_first - 1, tb
    assert tb["prompt_n"] == b["tokens_evaluated"] - tb["cache_n"]
    assert tb["prompt_n"] < b["tokens_evaluated"] // 2
    assert b["tokens_cached"] >= b["tokens_evaluated"]
    assert c["timings"]["cache_n"] == 0 and c["timings"]["prompt_n"] == c["tokens_evaluated"]


@pytest.mark.real_model
async def test_parallel_requests_run_concurrently(worker: WorkerProc) -> None:
    def body(topic: str) -> dict[str, Any]:
        return {"prompt": chat_prompt(f"Write a long story about {topic}."), "n_predict": 120,
                "temperature": 0, "ignore_eos": True, "cache_prompt": False}

    async with aiohttp.ClientSession() as s:
        t = time.perf_counter()
        solo_a = await _stream(s, worker.url, body("a dragon"))
        t_a = time.perf_counter() - t
        t = time.perf_counter()
        solo_b = await _stream(s, worker.url, body("a knight"))
        t_b = time.perf_counter() - t
        events: list[tuple[str, float]] = []
        t = time.perf_counter()
        par_a, par_b = await asyncio.gather(
            _stream(s, worker.url, body("a dragon"), events, "a"),
            _stream(s, worker.url, body("a knight"), events, "b"),
        )
        t_par = time.perf_counter() - t
    for frames in (par_a, par_b):
        assert frames[-1]["stop_type"] == "limit" and frames[-1]["tokens_predicted"] == 120
    assert {par_a[-1]["id_slot"], par_b[-1]["id_slot"]} == {0, 1}
    # both streams were interleaved: each started before the other finished
    first = {lab: min(ts for l, ts in events if l == lab) for lab in "ab"}
    last = {lab: max(ts for l, ts in events if l == lab) for lab in "ab"}
    assert first["a"] < last["b"] and first["b"] < last["a"]
    assert t_par < t_a + t_b, (t_par, t_a, t_b)
    # greedy results are not changed by batching
    text = lambda frames: "".join(f["content"] for f in frames)  # noqa: E731
    assert len(text(par_a)) > 50 and len(text(par_b)) > 50
    assert text(solo_a)[:40] == text(par_a)[:40] and text(solo_b)[:40] == text(par_b)[:40]


@pytest.mark.real_model
async def test_client_disconnect_cancels_and_frees_slot(worker: WorkerProc) -> None:
    body = {"prompt": chat_prompt("Write an endless story."), "n_predict": 1500, "temperature": 0.7,
            "ignore_eos": True, "stream": True}
    async with aiohttp.ClientSession() as s:
        resp = await s.post(worker.url + "/completion", json=body)
        got = 0
        async for line in resp.content:
            if line.startswith(b"data: "):
                got += 1
            if got >= 5:
                break
        async with s.get(worker.url + "/slots") as r:
            busy = [x["is_processing"] for x in await r.json()]
        assert any(busy)
        resp.close()
        t0 = time.perf_counter()
        while True:
            async with s.get(worker.url + "/slots") as r:
                slots = await r.json()
            if not any(x["is_processing"] for x in slots):
                break
            assert time.perf_counter() - t0 < 5, slots
            await asyncio.sleep(0.05)
        # the slot is usable again right away
        _, out = await _post(s, worker.url + "/completion",
                             {"prompt": chat_prompt("Hi"), "n_predict": 4, "temperature": 0})
    assert out["tokens_predicted"] == 4


@pytest.mark.real_model
async def test_seeded_sampling_is_reproducible(worker: WorkerProc) -> None:
    body = {"prompt": chat_prompt("Invent a fantasy name."), "n_predict": 40, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "cache_prompt": False, "ignore_eos": True}
    async with aiohttp.ClientSession() as s:
        _, a = await _post(s, worker.url + "/completion", {**body, "seed": 1234})
        _, b = await _post(s, worker.url + "/completion", {**body, "seed": 1234})
        (_, c), (_, d) = await asyncio.gather(
            _post(s, worker.url + "/completion", {**body, "seed": 99}),
            _post(s, worker.url + "/completion", {**body, "seed": 99}),
        )
        _, e = await _post(s, worker.url + "/completion", {**body, "seed": 7})
    assert a["content"] == b["content"]
    assert c["content"] == d["content"]            # same seed, batched together
    assert e["content"] != a["content"]
    assert a["generation_settings"]["seed"] == 1234


@pytest.mark.real_model
async def test_sampling_parameters_and_penalties_accepted(worker: WorkerProc) -> None:
    body = {"prompt": chat_prompt("List fruits."), "n_predict": 16, "temperature": 0.7, "top_k": 20,
            "top_p": 0.9, "min_p": 0.1, "repeat_penalty": 1.3, "repeat_last_n": 32,
            "presence_penalty": 0.5, "frequency_penalty": 0.2, "logit_bias": [[151645, False]]}
    async with aiohttp.ClientSession() as s:
        status, out = await _post(s, worker.url + "/completion", body)
    assert status == 200 and out["tokens_predicted"] == 16 and out["stop_type"] == "limit"


@pytest.mark.real_model
def test_slots_memory_and_embeddings(worker: WorkerProc) -> None:
    status, slots = _http_json("GET", worker.url + "/slots")
    assert status == 200 and [s["id"] for s in slots] == [0, 1]
    assert all(s["n_ctx"] == 2048 for s in slots)
    status, mem = _http_json("GET", worker.url + "/memory")
    assert status == 200
    assert {"active", "peak", "cache"} <= set(mem)
    assert mem["active"] > 100 * 2**20 and mem["prompt_cache"]["budget"] > 0
    for path in ("/embedding", "/v1/embeddings"):
        status, err = _http_json("POST", worker.url + path, {"input": "hi"})
        assert status == 501 and err["error"]["type"] == "not_supported_error"


@pytest.mark.real_model
async def test_kv_bits_sequential_backend(tmp_path: Path) -> None:
    w = WorkerProc(model_path("mlx_tiny"), tmp_path / "kv.log", "--kv-bits", "8", parallel=2, ctx=1024)
    try:
        w.wait_ready()
        _, props = _http_json("GET", w.url + "/props")
        assert props["kv_bits"] == 8 and props["batching"] is False and props["total_slots"] == 2
        async with aiohttp.ClientSession() as s:
            body = {"prompt": chat_prompt("Count from 1 to 10."), "n_predict": 24, "temperature": 0}
            (st1, a), (st2, b) = await asyncio.gather(
                _post(s, w.url + "/completion", body),
                _post(s, w.url + "/completion", {**body, "prompt": chat_prompt("Name three animals.")}),
            )
            assert st1 == st2 == 200
            assert a["tokens_predicted"] > 0 and b["tokens_predicted"] > 0
            # quantized prefix cache is reused
            _, c = await _post(s, w.url + "/completion", body)
            assert c["timings"]["cache_n"] == c["tokens_evaluated"] - 1
            # cancellation while queued/running frees the single active slot
            resp = await s.post(w.url + "/completion", json={**body, "n_predict": 900, "ignore_eos": True,
                                                             "stream": True})
            async for line in resp.content:
                if line.startswith(b"data: "):
                    break
            resp.close()
            _, d = await _post(s, w.url + "/completion", {**body, "n_predict": 3})
            assert d["tokens_predicted"] == 3
        _, mem = _http_json("GET", w.url + "/memory")
        assert mem["kv_bytes_per_token"] > 0
    finally:
        assert w.stop() == 0, w.log_tail()


@pytest.mark.real_model
async def test_sigterm_exits_cleanly_even_while_streaming(tmp_path: Path) -> None:
    w = WorkerProc(model_path("mlx_tiny"), tmp_path / "term.log", parallel=1, ctx=1024)
    try:
        w.wait_ready()
        async with aiohttp.ClientSession() as s:
            resp = await s.post(w.url + "/completion", json={
                "prompt": chat_prompt("Write an endless story."), "n_predict": 2000, "ignore_eos": True,
                "stream": True})
            frames = 0
            async for line in resp.content:
                if line.startswith(b"data: "):
                    frames += 1
                if frames >= 3:
                    break
            t0 = time.perf_counter()
            w.proc.send_signal(signal.SIGTERM)
            tail = b""
            try:
                async for chunk in resp.content.iter_any():
                    tail += chunk
            except aiohttp.ClientError:
                pass
            resp.close()
        rc = await asyncio.to_thread(w.proc.wait, 20)
        assert rc == 0, w.log_tail()
        assert time.perf_counter() - t0 < 15
        if b"data: " in tail:  # the open stream was ended with an error frame
            last = json.loads(tail.strip().split(b"\n")[-1][6:])
            assert last.get("stop") is True or last["error"]["type"] == "unavailable_error"
    finally:
        w.stop()


@pytest.mark.real_model
def test_idle_sigterm_exit_code_zero(tmp_path: Path) -> None:
    w = WorkerProc(model_path("mlx_tiny"), tmp_path / "idle.log", parallel=1, ctx=512)
    try:
        w.wait_ready()
        w.proc.send_signal(signal.SIGTERM)
        assert w.proc.wait(timeout=20) == 0
        assert "Worker zakończony" in w.log_tail()
    finally:
        w.stop()


def test_load_failure_exits_with_polish_message(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus-model"
    bogus.mkdir()
    (bogus / "config.json").write_text(json.dumps({"model_type": "definitely_not_a_model"}))
    (bogus / "tokenizer_config.json").write_text("{}")
    w = WorkerProc(bogus, tmp_path / "fail.log", parallel=1, ctx=512)
    try:
        rc = w.proc.wait(timeout=120)
    finally:
        w.stop()
    assert rc == 1
    assert "Błąd ładowania modelu" in w.log_tail()


def test_port_in_use_exits_with_error(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        cmd = [sys.executable, "-m", "localcode.runtime.mlx_worker", "--model", str(tmp_path),
               "--port", str(port)]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, timeout=60)
    assert res.returncode == 1
    assert "Nie można nasłuchiwać" in res.stderr.decode()


# --------------------------------------------------------------------------- #
# Side-by-side parity with the real llama-server (same Qwen3-0.6B weights)
# --------------------------------------------------------------------------- #


def _llama_server() -> str | None:
    return shutil.which("llama-server") or (
        "/opt/homebrew/bin/llama-server" if os.path.exists("/opt/homebrew/bin/llama-server") else None
    )


@pytest.mark.real_model
async def test_protocol_parity_with_llama_server(worker: WorkerProc, tmp_path: Path) -> None:
    binary = _llama_server()
    if binary is None:
        pytest.skip("llama-server not installed")
    gguf = model_path("gguf_tiny")
    port = free_port()
    log = open(tmp_path / "llama.log", "wb")
    proc = subprocess.Popen(
        [binary, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-c", "2048", "-np", "1",
         "--jinja", "--no-webui"], stdout=log, stderr=subprocess.STDOUT,
    )
    llama = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while True:
            assert proc.poll() is None, "llama-server exited"
            try:
                if _http_json("GET", llama + "/health", timeout=2)[0] == 200:
                    break
            except (urllib.error.URLError, OSError):
                pass
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)

        # props: every llama-server key exists in the worker's props
        _, lp = _http_json("GET", llama + "/props")
        _, mp = _http_json("GET", worker.url + "/props")
        assert set(lp) - {"media_marker", "ui", "ui_settings", "cors_proxy_enabled", "model_ftype"} <= set(mp)
        assert set(lp["chat_template_caps"]) <= set(mp["chat_template_caps"])
        assert lp["eos_token"] == mp["eos_token"]

        # identical tokenizer and template rendering
        for body in ({"content": "Zażółć gęślą jaźń <|im_end|> 😊"},
                     {"content": "<tool_call>", "parse_special": False}):
            assert _http_json("POST", llama + "/tokenize", body) == _http_json("POST", worker.url + "/tokenize", body)
        tmpl = {"messages": TOOL_CONVERSATION, "tools": [WEATHER_TOOL],
                "chat_template_kwargs": {"enable_thinking": False}}
        assert _http_json("POST", llama + "/apply-template", tmpl) == _http_json(
            "POST", worker.url + "/apply-template", tmpl)

        # completion frames and errors have the same keys
        body = {"prompt": chat_prompt("Count from 1 to 20 separated by commas."), "n_predict": 40,
                "temperature": 0, "stop": ["7, 8"]}
        async with aiohttp.ClientSession() as s:
            lf = await _stream(s, llama, body)
            mf = await _stream(s, worker.url, body)
            assert set(lf[0]) == set(mf[0]) == LLAMA_STREAM_FRAME_KEYS
            assert set(lf[-1]) == set(mf[-1]) == LLAMA_FINAL_KEYS
            assert set(lf[-1]["timings"]) == set(mf[-1]["timings"])
            for f in (lf[-1], mf[-1]):
                assert f["stop_type"] in ("word", "eos", "limit")
            _, l_plain = await _post(s, llama + "/completion", {**body, "stream": False})
            _, m_plain = await _post(s, worker.url + "/completion", {**body, "stream": False})
            assert set(l_plain) == set(m_plain)
            overflow = {"prompt": [9707] * 2048, "n_predict": 4}
            l_status, l_err = await _post(s, llama + "/completion", overflow)
            m_status, m_err = await _post(s, worker.url + "/completion", overflow)
            assert l_status == m_status == 400
            assert l_err == m_err
            l_status, l_err = await _post(s, llama + "/v1/embeddings", {"input": "x"})
            m_status, m_err = await _post(s, worker.url + "/v1/embeddings", {"input": "x"})
            assert l_status == m_status == 501 and l_err["error"]["type"] == m_err["error"]["type"]
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()
