"""Models borrowed from OpenAI, Anthropic, Gemini and DeepInfra.

No network: every test drives the clients through a fake aiohttp session, which
is the only way to cover four providers, two wire dialects and the streaming
paths without four API keys and a bill.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from localcode.config import RemoteProvider, RemoteSettings, Settings
from localcode.remote import (
    PROVIDERS,
    RemoteClient,
    RemoteError,
    RemoteRegistry,
    is_remote_key,
    parse_key,
    probe,
)
from localcode.remote.engine import RemoteEngine, RemoteLease, remote_model
from localcode.runtime.base import LoadConfig, SamplingParams


# --------------------------------------------------------------------------- #
# A fake aiohttp
# --------------------------------------------------------------------------- #


class FakeContent:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self) -> "FakeContent":
        self._iter = iter(self._lines)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter).encode("utf-8")
        except StopIteration:
            raise StopAsyncIteration from None


class FakeResponse:
    def __init__(self, *, status: int = 200, body: Any = None, lines: list[str] | None = None,
                 headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self.content = FakeContent(lines or [])

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body or {})

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    """Records requests and replays scripted responses."""

    def __init__(self, get: FakeResponse | None = None,
                 post: "FakeResponse | list[FakeResponse] | None" = None) -> None:
        self._get = get
        self._posts = list(post) if isinstance(post, list) else None   # a script, last one repeats
        self._post = None if isinstance(post, list) else post
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, headers: dict[str, str] | None = None, **kw: Any) -> FakeResponse:
        self.requests.append(("GET", url, {"headers": headers or {}, **kw}))
        return self._get or FakeResponse(body={})

    def post(self, url: str, headers: dict[str, str] | None = None, **kw: Any) -> FakeResponse:
        self.requests.append(("POST", url, {"headers": headers or {}, **kw}))
        if self._posts is not None:
            return self._posts.pop(0) if len(self._posts) > 1 else self._posts[0]
        return self._post or FakeResponse(lines=["data: [DONE]"])

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def client(provider: str, session: FakeSession) -> RemoteClient:
    return RemoteClient(provider, "sk-test", session_factory=lambda: session)


def sse(*events: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(e)}\n" for e in events] + ["data: [DONE]\n"]


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key, parsed", [
    ("remote:openai:gpt-5", ("openai", "gpt-5")),
    ("remote:anthropic:claude-opus-5", ("anthropic", "claude-opus-5")),
    ("remote:gemini:gemini-3-pro", ("gemini", "gemini-3-pro")),
    ("remote:deepinfra:meta-llama/Llama-4-70B", ("deepinfra", "meta-llama/Llama-4-70B")),
    ("remote:unlid:glm-5.3-flash-uncensored", ("unlid", "glm-5.3-flash-uncensored")),
])
def test_a_remote_key_names_its_provider(key: str, parsed: tuple[str, str]) -> None:
    assert parse_key(key) == parsed
    assert is_remote_key(key)


@pytest.mark.parametrize("key", ["org/Qwen3-8B/q4", "remote:unknown:x", "remote:openai:", "", "x"])
def test_a_local_key_is_not_remote(key: str) -> None:
    assert parse_key(key) is None
    assert not is_remote_key(key)


# --------------------------------------------------------------------------- #
# Model lists
# --------------------------------------------------------------------------- #


async def test_openai_models_are_listed() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [
        {"id": "gpt-5", "owned_by": "openai"},
        {"id": "gpt-5-mini", "owned_by": "openai"},
    ]}))
    models = await client("openai", session).models()
    assert [m.model_id for m in models] == ["gpt-5", "gpt-5-mini"]
    assert models[0].key == "remote:openai:gpt-5"
    assert session.requests[0][1].endswith("/models")
    assert session.requests[0][2]["headers"]["Authorization"] == "Bearer sk-test"


async def test_anthropic_sends_its_own_headers() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [
        {"id": "claude-opus-5", "display_name": "Claude Opus 5"}]}))
    models = await client("anthropic", session).models()
    assert models[0].display_name == "Claude Opus 5"
    headers = session.requests[0][2]["headers"]
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"]


async def test_gemini_strips_the_models_prefix_and_skips_what_cannot_chat() -> None:
    session = FakeSession(get=FakeResponse(body={"models": [
        {"name": "models/gemini-3-pro", "inputTokenLimit": 1_000_000,
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/text-embedding-5", "supportedGenerationMethods": ["embedContent"]},
    ]}))
    models = await client("gemini", session).models()
    assert [m.model_id for m in models] == ["gemini-3-pro"]
    assert models[0].context_length == 1_000_000


async def test_the_model_list_is_cached_until_asked_to_refresh() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [{"id": "gpt-5"}]}))
    c = client("openai", session)
    await c.models()
    await c.models()
    assert len(session.requests) == 1
    await c.models(refresh=True)
    assert len(session.requests) == 2


@pytest.mark.parametrize("status, code", [(401, "bad_key"), (429, "rate_limited"),
                                          (404, "model_not_found"), (500, "provider_error")])
async def test_provider_errors_speak_polish(status: int, code: str) -> None:
    session = FakeSession(get=FakeResponse(status=status, body="nope"))
    with pytest.raises(RemoteError) as excinfo:
        await client("openai", session).models()
    assert excinfo.value.code == code
    assert excinfo.value.message


async def test_probe_checks_a_key_before_it_is_saved() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [{"id": "gpt-5"}]}))
    models = await probe("openai", "sk-test", session_factory=lambda: session)
    assert models and models[0].model_id == "gpt-5"

    with pytest.raises(RemoteError):
        await probe("openai", "   ", session_factory=lambda: session)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


async def test_openai_streaming_yields_text_then_a_final_chunk() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"choices": [{"delta": {"content": "Dzień "}}]},
        {"choices": [{"delta": {"content": "dobry"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
    )))
    chunks = [c async for c in client("openai", session).stream(
        [{"role": "user", "content": "cześć"}], model="gpt-5")]
    assert "".join(c.text for c in chunks) == "Dzień dobry"
    assert chunks[-1].done and chunks[-1].prompt_tokens == 12


async def test_anthropic_streaming_is_understood() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 9}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Cześć"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 2}},
    )))
    chunks = [c async for c in client("anthropic", session).stream(
        [{"role": "system", "content": "bądź zwięzły"}, {"role": "user", "content": "hej"}],
        model="claude-opus-5")]
    assert "".join(c.text for c in chunks) == "Cześć"
    assert chunks[-1].finish_reason == "stop"
    body = session.requests[0][2]["json"]
    # System travels separately — as a block, so it can carry a cache mark.
    assert [b["text"] for b in body["system"]] == ["bądź zwięzły"]
    assert [m["role"] for m in body["messages"]] == ["user"]


async def test_gemini_streaming_is_understood() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"candidates": [{"content": {"parts": [{"text": "Hej"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": " tam"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}},
    )))
    chunks = [c async for c in client("gemini", session).stream(
        [{"role": "user", "content": "hej"}], model="gemini-3-pro")]
    assert "".join(c.text for c in chunks) == "Hej tam"
    assert chunks[-1].prompt_tokens == 5


async def test_tool_calls_survive_the_stream() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "read_file",
                                                      "arguments": '{"path":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"a.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )))
    chunks = [c async for c in client("openai", session).stream(
        [{"role": "user", "content": "przeczytaj a.py"}], model="gpt-5",
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}])]
    calls = chunks[-1].tool_calls
    assert calls == [{"id": "call_1", "name": "read_file", "arguments": '{"path":"a.py"}'}]


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


def engine(session: FakeSession, key: str = "remote:openai:gpt-5") -> RemoteEngine:
    model = remote_model(key, display_name="GPT-5", context_length=200_000)
    return RemoteEngine("e1", model, LoadConfig(n_ctx=0), client(parse_key(key)[0], session))


async def test_the_engine_passes_the_conversation_through() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )))
    e = engine(session)
    prompt = await e.apply_template([{"role": "user", "content": "hej"}], None)
    chunks = [c async for c in e.complete(prompt, SamplingParams(max_tokens=64))]
    assert "".join(c.text for c in chunks) == "ok"
    assert session.requests[0][2]["json"]["messages"] == [{"role": "user", "content": "hej"}]


async def test_the_engine_re_encodes_tool_calls_for_the_local_parser() -> None:
    """The agent parses ``<tool_call>`` blocks out of text; a remote model
    answers with structured calls, so they are written back into that shape."""
    session = FakeSession(post=FakeResponse(lines=sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "list_dir",
                                                  "arguments": '{"path":"."}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )))
    e = engine(session)
    text = "".join([c.text async for c in
                    e.complete(await e.apply_template([{"role": "user", "content": "co tu jest"}]),
                               SamplingParams())])
    assert "<tool_call>" in text and '"list_dir"' in text

    from localcode.runtime.toolcalls import parse_output

    parsed = parse_output(text, tool_format="hermes")
    assert parsed.tool_calls and parsed.tool_calls[0].name == "list_dir"
    assert parsed.tool_calls[0].arguments == {"path": "."}


async def test_a_remote_model_costs_this_mac_nothing() -> None:
    e = engine(FakeSession())
    assert await e.memory_bytes() == 0
    assert e.n_ctx == 200_000
    assert e.model.size_bytes == 0


async def test_a_provider_error_becomes_an_engine_error() -> None:
    from localcode.runtime.base import EngineError

    session = FakeSession(post=FakeResponse(status=401, body="bad key"))
    e = engine(session)
    with pytest.raises(EngineError):
        [c async for c in e.complete(await e.apply_template([{"role": "user", "content": "x"}]),
                                     SamplingParams())]


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def settings_with(**keys: str) -> Settings:
    remote = RemoteSettings(**{name: RemoteProvider(api_key=key) for name, key in keys.items()})
    return Settings(remote=remote)


def test_a_provider_without_a_key_is_not_configured() -> None:
    registry = RemoteRegistry(Settings())
    assert registry.providers == []
    with pytest.raises(RemoteError) as excinfo:
        registry.client("openai")
    assert excinfo.value.code == "not_configured"


def test_a_disabled_provider_is_skipped() -> None:
    settings = settings_with(openai="sk-1")
    settings.remote.openai.enabled = False
    assert RemoteRegistry(settings).providers == []


async def test_the_registry_lists_every_configured_provider() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [{"id": "m1"}]}))
    registry = RemoteRegistry(settings_with(openai="sk-1", deepinfra="sk-2"),
                              client_factory=lambda p, k, b: client(p, session))
    assert registry.providers == ["openai", "deepinfra"]
    models = await registry.models()
    assert {m.provider for m in models} == {"openai", "deepinfra"}


async def test_one_broken_provider_does_not_empty_the_list() -> None:
    def factory(provider: str, key: str, base: str | None) -> RemoteClient:
        if provider == "openai":
            return client(provider, FakeSession(get=FakeResponse(status=500, body="boom")))
        return client(provider, FakeSession(get=FakeResponse(body={"data": [{"id": "ok"}]})))

    registry = RemoteRegistry(settings_with(openai="sk-1", gemini="sk-2"), client_factory=factory)
    models = await registry.models()
    assert [m.provider for m in models] == ["gemini"]


async def test_a_lease_looks_like_a_local_one() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [{"id": "gpt-5"}]}))
    registry = RemoteRegistry(settings_with(openai="sk-1"),
                              client_factory=lambda p, k, b: client(p, session))
    async with await registry.lease("remote:openai:gpt-5") as lease:
        assert isinstance(lease, RemoteLease)
        assert lease.engine.running
        assert lease.n_ctx > 0
        lease.record_stats(None)          # a no-op, but callers call it


def test_every_provider_has_a_label_and_a_key_page() -> None:
    from localcode.remote import PROVIDER_KEY_URLS, PROVIDER_LABELS

    for provider in PROVIDERS:
        assert PROVIDER_LABELS[provider]
        assert PROVIDER_KEY_URLS[provider].startswith("https://")


# --------------------------------------------------------------------------- #
# unlid.ai
# --------------------------------------------------------------------------- #
# A gateway that resells many models cannot say everything in a flat OpenAI row,
# so unlid nests it under a key named after itself. Reading only the flat fields
# gave every model the 128k fallback window and an assumed "it can call tools" —
# both wrong, and both decide how the coding agent behaves.

#: One row exactly as https://unlid.ai/docs documents GET /v1/models.
UNLID_ROW = {
    "id": "glm-5.3-flash-uncensored",
    "object": "model",
    "owned_by": "unlid",
    "unlid": {
        "status": "live",
        "available": True,
        "uncensored": True,
        "supports_tools": True,
        "pricing": {"input_usd_per_m": 0.24, "output_usd_per_m": 0.96,
                    "checked_at": "2026-09-08"},
        "context_length": 262144,
        "max_output": 8192,
    },
}


def test_unlid_is_a_provider_with_an_openai_compatible_endpoint() -> None:
    from localcode.remote import DEFAULT_BASE_URLS, PROVIDER_LABELS, PROVIDERS

    assert "unlid" in PROVIDERS
    assert PROVIDER_LABELS["unlid"] == "Unlid"
    assert DEFAULT_BASE_URLS["unlid"] == "https://api.unlid.ai/v1"


def test_unlid_speaks_openai_dialect() -> None:
    """The assumption the whole integration rests on: no special client needed.

    Streaming and tool calls are covered above for that dialect, so pinning this
    keeps unlid inside that coverage instead of quietly growing a branch.
    """
    from localcode.remote.providers import RemoteClient

    assert RemoteClient("unlid", "ul-test").dialect == "openai"


async def test_unlid_models_carry_their_real_window_and_tool_support() -> None:
    session = FakeSession(get=FakeResponse(body={"object": "list", "data": [UNLID_ROW]}))
    models = await client("unlid", session).models()
    assert [m.model_id for m in models] == ["glm-5.3-flash-uncensored"]
    assert models[0].context_length == 262_144      # from the nested block, not the fallback
    assert models[0].supports_tools is True
    # Same wire protocol as OpenAI: bearer key, /models under the base URL.
    assert session.requests[0][1].endswith("/models")
    assert session.requests[0][2]["headers"]["Authorization"] == "Bearer sk-test"


async def test_a_model_the_gateway_calls_unavailable_is_not_offered() -> None:
    """Offering a model that answers 404 is worse than not listing it."""
    retired = {"id": "stary", "object": "model",
               "unlid": {"status": "retired", "available": False}}
    session = FakeSession(get=FakeResponse(body={"data": [UNLID_ROW, retired]}))
    models = await client("unlid", session).models()
    assert [m.model_id for m in models] == ["glm-5.3-flash-uncensored"]


async def test_a_model_without_tool_support_is_not_offered_as_an_agent() -> None:
    from localcode.remote import remote_model

    row = {"id": "bez-narzedzi", "unlid": {"supports_tools": False, "context_length": 32768}}
    session = FakeSession(get=FakeResponse(body={"data": [row]}))
    model = (await client("unlid", session).models())[0]
    assert model.supports_tools is False
    assert model.to_dict()["supportsTools"] is False
    local = remote_model(model.key, context_length=model.context_length,
                         supports_tools=model.supports_tools)
    assert "tools" not in local.capabilities


async def test_a_provider_that_says_nothing_still_counts_as_tool_capable() -> None:
    """How every provider behaved before any of them reported this."""
    from localcode.remote import remote_model

    session = FakeSession(get=FakeResponse(body={"data": [{"id": "gpt-5"}]}))
    model = (await client("openai", session).models())[0]
    assert model.supports_tools is None
    assert model.to_dict()["supportsTools"] is True
    assert "tools" in remote_model(model.key).capabilities


def test_an_unlid_model_drives_the_agent_in_its_strong_profile() -> None:
    """The point of connecting a cloud at all: the agent stops squeezing."""
    from localcode.agent.profile import profile_for
    from localcode.remote import remote_model

    model = remote_model("remote:unlid:glm-5.3-flash-uncensored",
                         context_length=262_144, supports_tools=True)
    assert model.meta.context_length == 262_144
    profile = profile_for(model, model.meta.context_length)
    assert profile.is_strong
    assert not profile.tiny_tools              # full tool menu, full schemas
    assert not profile.mask_old_tool_results   # old tool output stays readable
    assert profile.compact_threshold == 0.92   # compacts late, not at 80 %
    assert profile.reserve_output == 8192


# --------------------------------------------------------------------------- #
# A generation is not a request with a length
# --------------------------------------------------------------------------- #


def test_a_listing_has_a_deadline_and_a_generation_has_none() -> None:
    """`ClientTimeout.total` covers reading the body, so one value cannot serve both.

    Sixty seconds is generous for `/models` and far too short for an answer the
    model is still writing: a remote model asked for anything substantial had
    its answer cut off mid-sentence.
    """
    import aiohttp

    from localcode.remote.providers import HTTP_TIMEOUT_S, STREAM_IDLE_TIMEOUT_S, RemoteClient

    listing = RemoteClient._timeout(aiohttp, streaming=False)
    assert listing.total == HTTP_TIMEOUT_S

    generating = RemoteClient._timeout(aiohttp, streaming=True)
    assert generating.total is None                       # no deadline on the answer
    assert generating.sock_read == STREAM_IDLE_TIMEOUT_S  # but silence still ends it
    assert generating.sock_connect == HTTP_TIMEOUT_S


async def test_a_provider_error_never_carries_our_key() -> None:
    """A proxy reflecting request headers must not put the key in the transcript."""
    session = FakeSession(get=FakeResponse(status=500, body="bad request, auth was Bearer sk-test"))
    with pytest.raises(RemoteError) as excinfo:
        await client("openai", session).models()
    assert "sk-test" not in excinfo.value.message
    assert "500" in excinfo.value.message



# --------------------------------------------------------------------------- #
# Prompt caching
# --------------------------------------------------------------------------- #
# Anthropic caches only what is marked. Unmarked, every step of an agent turn —
# a dozen a turn — paid full price for tools, system prompt and history again.


async def test_anthropic_requests_mark_tools_system_and_the_last_message() -> None:
    session = FakeSession(post=FakeResponse(lines=sse({"type": "message_stop"})))
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}},
             {"type": "function", "function": {"name": "grep", "parameters": {}}}]
    _ = [c async for c in client("anthropic", session).stream(
        [{"role": "system", "content": "jesteś agentem"},
         {"role": "user", "content": "znajdź X"},
         {"role": "assistant", "content": "szukam"},
         {"role": "user", "content": "dalej"}],
        model="claude-opus-5", tools=tools)]
    body = session.requests[0][2]["json"]
    mark = {"type": "ephemeral"}
    assert body["system"][-1]["cache_control"] == mark
    assert body["tools"][-1]["cache_control"] == mark
    assert "cache_control" not in body["tools"][0]          # one breakpoint covers them all
    assert body["messages"][-1]["content"][-1]["cache_control"] == mark
    assert all("cache_control" not in str(m) for m in body["messages"][:-1])
    marks = str(body).count("ephemeral")
    assert marks <= 4                                       # Anthropic's limit


async def test_cache_hits_are_counted_and_the_prompt_is_not_undercounted() -> None:
    """input_tokens excludes cached tokens; the prompt is all three parts."""
    session = FakeSession(post=FakeResponse(lines=sse(
        {"type": "message_start", "message": {"usage": {
            "input_tokens": 50, "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 400}}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 7}},
    )))
    chunks = [c async for c in client("anthropic", session).stream(
        [{"role": "user", "content": "hej"}], model="claude-opus-5")]
    assert chunks[-1].prompt_tokens == 9450
    assert chunks[-1].cached_tokens == 9000


async def test_openai_cache_hits_are_read_too() -> None:
    session = FakeSession(post=FakeResponse(lines=sse(
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 2000, "completion_tokens": 3,
                   "prompt_tokens_details": {"cached_tokens": 1536}}},
    )))
    chunks = [c async for c in client("openai", session).stream(
        [{"role": "user", "content": "hej"}], model="gpt-5")]
    assert chunks[-1].cached_tokens == 1536


# --------------------------------------------------------------------------- #
# Only models that write text reach the pickers
# --------------------------------------------------------------------------- #


def deepinfra_row(model_id: str, tags: list[str], ctx: int | None = None) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "owned_by": "deepinfra",
            "metadata": {"description": f"{model_id} opis", "context_length": ctx, "tags": tags}}


def test_a_gateway_s_embedding_image_and_speech_models_are_left_out() -> None:
    """Regression: DeepInfra's list put BAAI/bge, Bria and TTS models in `/model`."""
    from localcode.remote.providers import _parse_models

    data = {"data": [
        deepinfra_row("meta-llama/Llama-3.3-70B-Instruct-Turbo", ["chat"], 131072),
        deepinfra_row("Qwen/Qwen3-VL-235B", ["chat", "vlm", "vision"], 262144),
        deepinfra_row("BAAI/bge-m3", ["embed"]),
        deepinfra_row("Bria/Bria-3.2", ["image-gen"]),
        deepinfra_row("Audio8/Audio8-TTS-Preview-0.6b", ["tts"]),
        deepinfra_row("openai/whisper-large-v3", ["stt"]),
        deepinfra_row("google/veo-3.1", ["video-gen"]),
    ]}
    models = _parse_models("deepinfra", data)
    assert [m.model_id for m in models] == ["Qwen/Qwen3-VL-235B",
                                            "meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    llama = models[1]
    assert llama.context_length == 131072
    assert llama.description == "meta-llama/Llama-3.3-70B-Instruct-Turbo opis"


@pytest.mark.parametrize("model_id, text", [
    ("gpt-5", True), ("o3", True), ("chatgpt-4o-latest", True), ("claude-opus-5", True),
    ("text-embedding-3-small", False), ("dall-e-3", False), ("gpt-image-1", False),
    ("whisper-1", False), ("tts-1-hd", False), ("gpt-4o-mini-tts", False),
    ("gpt-4o-transcribe", False), ("omni-moderation-latest", False), ("sora-2", False),
    ("gpt-realtime", False),
])
def test_without_tags_the_name_decides(model_id: str, text: bool) -> None:
    from localcode.remote.providers import is_text_model

    assert is_text_model(model_id, {}, {}) is text


# --------------------------------------------------------------------------- #
# Retrying 429 / 503 / 529
# --------------------------------------------------------------------------- #


def _ok() -> FakeResponse:
    return FakeResponse(lines=sse({"choices": [{"delta": {"content": "ok"}}]},
                                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}))


def _retrying(session: FakeSession) -> tuple[RemoteClient, list[float]]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    return RemoteClient("openai", "sk-test", session_factory=lambda: session,
                        sleep=fake_sleep), slept


async def _drain(c: RemoteClient) -> str:
    return "".join([x.text async for x in c.stream([{"role": "user", "content": "hi"}],
                                                   model="gpt-5")])


@pytest.mark.parametrize("status", [429, 503, 529])
async def test_a_busy_provider_is_retried_after_the_default_delay(status: int) -> None:
    session = FakeSession(post=[FakeResponse(status=status, body="busy"), _ok()])
    c, slept = _retrying(session)
    assert await _drain(c) == "ok"
    assert slept == [2.0]
    assert len([r for r in session.requests if r[0] == "POST"]) == 2


async def test_retry_after_in_seconds_is_obeyed() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="slow", headers={"Retry-After": "7"}),
                                _ok()])
    c, slept = _retrying(session)
    assert await _drain(c) == "ok"
    assert slept == [7.0]


async def test_retry_after_as_an_http_date_is_obeyed() -> None:
    from email.utils import formatdate
    import time as _time

    date = formatdate(_time.time() + 30, usegmt=True)
    session = FakeSession(post=[FakeResponse(status=429, body="slow", headers={"Retry-After": date}),
                                _ok()])
    c, slept = _retrying(session)
    assert await _drain(c) == "ok"
    assert len(slept) == 1 and 20 <= slept[0] <= 31


async def test_an_empty_account_is_not_retried() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="You exceeded your current quota"),
                                _ok()])
    c, slept = _retrying(session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert excinfo.value.code == "rate_limited"
    assert slept == []
    assert "Ponawiałem" not in excinfo.value.message
    assert "quota" in excinfo.value.message


async def test_it_gives_up_after_four_retries() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="too fast")])   # repeats
    c, slept = _retrying(session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == [2.0, 5.0, 10.0, 20.0]
    assert excinfo.value.code == "rate_limited"
    assert "Ponawiałem 4 razy" in excinfo.value.message
    assert "too fast" in excinfo.value.message
    assert len([r for r in session.requests if r[0] == "POST"]) == 5


async def test_a_long_retry_after_is_not_waited_for() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="later",
                                             headers={"Retry-After": "120"}), _ok()])
    c, slept = _retrying(session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == []
    assert excinfo.value.code == "rate_limited"


async def test_a_server_error_is_not_retried() -> None:
    session = FakeSession(post=[FakeResponse(status=500, body="boom"), _ok()])
    c, slept = _retrying(session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == []
    assert excinfo.value.code == "provider_error"


async def test_the_key_is_redacted_from_a_retried_429_message() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="bad key sk-test-secret-123")])
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    c = RemoteClient("openai", "sk-test-secret-123", session_factory=lambda: session, sleep=fake_sleep)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert "sk-test-secret-123" not in excinfo.value.message


async def test_openai_requests_send_tool_call_arguments_as_a_json_string() -> None:
    """DeepInfra answers 422 "Input should be a valid string" to a dict."""
    session = FakeSession(post=FakeResponse(lines=["data: [DONE]"]))
    call = {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": {"path": "ą"}}}
    history = [{"role": "user", "content": "zrób bota"},
               {"role": "assistant", "content": "", "tool_calls": [call]},
               {"role": "tool", "tool_call_id": "c1", "name": "list_dir", "content": "1 pozycja"}]
    _ = [c async for c in client("openai", session).stream(history, model="m")]
    sent = session.requests[0][2]["json"]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(sent, str) and json.loads(sent) == {"path": "ą"}
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {"path": "ą"}   # not mutated


@pytest.mark.parametrize("dialect, path, expected", [
    ("openai", ("max_tokens",), None),
    ("anthropic", ("max_tokens",), 32_000),
    ("gemini", ("generationConfig", "maxOutputTokens"), None),
])
async def test_no_output_limit_is_left_to_the_provider(dialect: str, path: tuple[str, ...],
                                                       expected: int | None) -> None:
    session = FakeSession(post=FakeResponse(lines=["data: [DONE]"]))
    _ = [c async for c in client(dialect, session).stream(
        [{"role": "user", "content": "hej"}], model="m", max_tokens=None)]
    body = session.requests[0][2]["json"]
    for key in path[:-1]:
        body = body[key]
    assert body.get(path[-1]) == expected


async def test_the_engine_drops_a_limit_no_provider_would_honour() -> None:
    from localcode.remote.engine import _output_cap
    assert _output_cap(4096) == 4096
    assert _output_cap(900_000) is None


# --------------------------------------------------------------------------- #
# Service tiers (Standard / Flex / Priority)
# --------------------------------------------------------------------------- #
# DeepInfra and OpenAI take "service_tier" on a chat request; Unlid forwards it
# upstream. Flex is cheaper but may queue a request for up to ten minutes and
# then answer 429; Priority costs more. Unset is standard.


def _tiered(provider: str, tier: str, session: FakeSession) -> tuple[RemoteClient, list[float]]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    return RemoteClient(provider, "sk-test", service_tier=tier, session_factory=lambda: session,
                        sleep=fake_sleep), slept


async def _sent(provider: str, tier: str) -> dict[str, Any]:
    session = FakeSession(post=FakeResponse(lines=["data: [DONE]"]))
    c, _ = _tiered(provider, tier, session)
    _ = [x async for x in c.stream([{"role": "user", "content": "hej"}], model="m")]
    return session.requests[0][2]["json"]


@pytest.mark.parametrize("provider", ["deepinfra", "unlid", "openai"])
@pytest.mark.parametrize("tier", ["flex", "priority"])
async def test_a_chosen_tier_is_sent_with_the_request(provider: str, tier: str) -> None:
    assert (await _sent(provider, tier))["service_tier"] == tier


@pytest.mark.parametrize("provider", ["deepinfra", "unlid", "openai"])
async def test_standard_sends_no_tier_at_all(provider: str) -> None:
    assert "service_tier" not in await _sent(provider, "standard")


@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
async def test_providers_without_tiers_never_get_one(provider: str) -> None:
    body = await _sent(provider, "flex")
    assert "service_tier" not in json.dumps(body)


async def test_a_model_listing_never_carries_the_tier() -> None:
    session = FakeSession(get=FakeResponse(body={"data": [{"id": "m"}]}))
    c, _ = _tiered("deepinfra", "flex", session)
    await c.models()
    method, url, kwargs = session.requests[0]
    assert method == "GET" and "json" not in kwargs
    assert "service_tier" not in url


def test_an_unknown_tier_is_standard() -> None:
    assert RemoteClient("deepinfra", "k", service_tier="turbo").service_tier == "standard"
    assert RemoteClient("deepinfra", "k", service_tier=None).service_tier == "standard"
    assert RemoteClient("deepinfra", "k", service_tier=" Flex ").service_tier == "flex"
    assert RemoteClient("anthropic", "k", service_tier="flex").service_tier == "standard"


def test_settings_load_an_unknown_tier_as_standard(tmp_path) -> None:
    from localcode.config import load_settings

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"remote": {"deepinfra": {"api_key": "k", "service_tier": "turbo"},
                                           "unlid": {"service_tier": "priority"}}}))
    loaded = load_settings(path)
    assert loaded.remote.deepinfra.service_tier == "standard"
    assert loaded.remote.unlid.service_tier == "priority"
    assert loaded.remote.openai.service_tier == "standard"       # absent = default


def test_only_openai_dialect_providers_offer_tiers() -> None:
    from localcode.remote import SERVICE_TIERS, TIERED_PROVIDERS

    assert SERVICE_TIERS == ("standard", "flex", "priority")
    assert set(TIERED_PROVIDERS) == {"openai", "deepinfra", "unlid"}


def test_a_flex_generation_may_wait_in_the_queue() -> None:
    """``sock_read`` also covers the wait for the headers, which Flex spends queued."""
    import aiohttp

    from localcode.remote.providers import FLEX_QUEUE_TIMEOUT_S, STREAM_IDLE_TIMEOUT_S

    assert FLEX_QUEUE_TIMEOUT_S > 10 * 60
    flex = RemoteClient._timeout(aiohttp, streaming=True, queued=True)
    assert flex.total is None and flex.sock_read == FLEX_QUEUE_TIMEOUT_S
    assert RemoteClient._timeout(aiohttp, streaming=True).sock_read == STREAM_IDLE_TIMEOUT_S


async def test_a_flex_session_gets_the_queue_allowance(monkeypatch) -> None:
    import aiohttp

    from localcode.remote.providers import FLEX_QUEUE_TIMEOUT_S, STREAM_IDLE_TIMEOUT_S

    seen: list[Any] = []
    monkeypatch.setattr(aiohttp, "ClientSession", lambda timeout=None: seen.append(timeout))
    for tier in ("flex", "priority", "standard"):
        RemoteClient("deepinfra", "k", service_tier=tier)._session(streaming=True)
    RemoteClient("deepinfra", "k", service_tier="flex")._session()          # a listing
    assert [t.sock_read for t in seen[:3]] == [FLEX_QUEUE_TIMEOUT_S, STREAM_IDLE_TIMEOUT_S,
                                               STREAM_IDLE_TIMEOUT_S]
    assert seen[3].total is not None and seen[3].sock_read is None


async def test_a_flex_429_is_reported_not_retried() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="no capacity"), _ok()])
    c, slept = _tiered("deepinfra", "flex", session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == []
    assert len([r for r in session.requests if r[0] == "POST"]) == 1
    assert excinfo.value.code == "flex_unavailable"
    assert excinfo.value.message == (
        "DeepInfra nie miał wolnej mocy w trybie Flex (czekał do 10 min). "
        "Spróbuj ponownie albo przełącz na Standard w Ustawieniach.")


async def test_a_flex_429_with_a_retry_after_is_still_not_retried() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="busy", headers={"Retry-After": "1"}),
                                _ok()])
    c, slept = _tiered("unlid", "flex", session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == [] and excinfo.value.code == "flex_unavailable"
    assert "Unlid nie miał wolnej mocy w trybie Flex." in excinfo.value.message


async def test_an_empty_account_on_flex_keeps_its_own_message() -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="insufficient balance"), _ok()])
    c, slept = _tiered("deepinfra", "flex", session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == [] and excinfo.value.code == "rate_limited"


@pytest.mark.parametrize("tier", ["priority", "standard"])
async def test_priority_and_standard_still_retry_a_429(tier: str) -> None:
    session = FakeSession(post=[FakeResponse(status=429, body="busy"), _ok()])
    c, slept = _tiered("deepinfra", tier, session)
    assert await _drain(c) == "ok"
    assert slept == [2.0]


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("tier, label", [("flex", "Flex"), ("priority", "Priority")])
async def test_a_model_refusing_the_tier_says_so(status: int, tier: str, label: str) -> None:
    body = {"detail": "service_tier 'flex' is not supported for this model"}
    session = FakeSession(post=[FakeResponse(status=status, body=body), _ok()])
    c, slept = _tiered("deepinfra", tier, session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert slept == []
    assert excinfo.value.code == "tier_unsupported"
    assert excinfo.value.status == 400
    assert excinfo.value.message == (
        f"Ten model u dostawcy DeepInfra nie obsługuje trybu {label}. "
        "Przełącz DeepInfra na Standard w Ustawieniach → Modele zdalne.")


async def test_a_400_about_something_else_is_an_ordinary_error() -> None:
    session = FakeSession(post=FakeResponse(status=400, body="max_tokens too large"))
    c, _ = _tiered("deepinfra", "flex", session)
    with pytest.raises(RemoteError) as excinfo:
        await _drain(c)
    assert excinfo.value.code == "provider_error"


def test_the_registry_rebuilds_the_client_when_the_tier_changes() -> None:
    settings = settings_with(deepinfra="sk-1")
    registry = RemoteRegistry(settings)
    first = registry.client("deepinfra")
    assert first.service_tier == "standard"
    assert registry.client("deepinfra") is first                  # nothing changed: cached

    first._models = (0.0, [])                                     # a fetched list
    settings.remote.deepinfra.service_tier = "flex"
    flex = registry.client("deepinfra")
    assert flex is not first and flex.service_tier == "flex"
    assert flex._models == (0.0, [])                              # same key, same list

    settings.remote.deepinfra.base_url = "https://proxy.example/v1"
    moved = registry.client("deepinfra")
    assert moved is not flex and moved.base_url == "https://proxy.example/v1"
    assert moved.service_tier == "flex" and moved._models is None


def test_a_factory_built_client_gets_the_tier_too() -> None:
    settings = settings_with(unlid="ul-1")
    settings.remote.unlid.service_tier = "priority"
    registry = RemoteRegistry(settings, client_factory=lambda p, k, b: RemoteClient(p, k))
    assert registry.client("unlid").service_tier == "priority"


async def test_a_429_on_the_model_list_is_not_called_a_flex_queue() -> None:
    session = FakeSession(get=FakeResponse(status=429, body="slow down"))
    c, _ = _tiered("deepinfra", "flex", session)
    with pytest.raises(RemoteError) as excinfo:
        await c.models()
    assert excinfo.value.code == "rate_limited"


async def test_a_cache_key_goes_to_openai_style_providers_only() -> None:
    """DeepInfra shares its KV cache only between requests with the same prompt_cache_key."""
    for dialect, expected in (("deepinfra", "localcode-s1-x"), ("openai", "localcode-s1-x"),
                              ("anthropic", None)):
        session = FakeSession(post=FakeResponse(lines=["data: [DONE]"]))
        _ = [c async for c in client(dialect, session).stream(
            [{"role": "user", "content": "hej"}], model="m", cache_key="localcode-s1/x")]
        assert session.requests[0][2]["json"].get("prompt_cache_key") == expected, dialect
    session = FakeSession(post=FakeResponse(lines=["data: [DONE]"]))
    _ = [c async for c in client("deepinfra", session).stream([{"role": "user", "content": "hej"}], model="m")]
    assert "prompt_cache_key" not in session.requests[0][2]["json"]
