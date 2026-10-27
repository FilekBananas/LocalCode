"""Tests for ``localcode.runtime.http_engine`` without any model.

A tiny aiohttp program that mimics llama-server's HTTP surface (written to
``tmp_path`` and spawned by ``FakeEngine``) exercises the real process
lifecycle: loading phase (503), early crash with log tail, startup timeout,
SIGTERM→SIGKILL escalation, process-group kill, port-race retry, API key,
SSE streaming with a huge final frame, cancellation, error mapping and the
embedding endpoint fallback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from localcode.hub.types import LocalModel
from localcode.runtime import http_engine as he
from localcode.runtime.base import CompletionChunk, EngineError, LoadConfig, SamplingParams, TemplateInfo
from localcode.runtime.http_engine import (
    ContextOverflowError,
    EngineHttpError,
    EngineStartError,
    EngineUnavailableError,
    HttpEngine,
    SSEDecoder,
    finish_reason_from_frame,
    kill_all_engine_processes,
    stats_from_frame,
)

# --------------------------------------------------------------------------- #
# Fake llama-server
# --------------------------------------------------------------------------- #

FAKE_SERVER = r'''
import argparse, asyncio, json, os, signal, subprocess, sys, time
from pathlib import Path
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--mode", default="ok")
ap.add_argument("--state", required=True)
ap.add_argument("--load-delay", type=float, default=0.4)
ap.add_argument("--api-key", default=None)
ap.add_argument("--no-tools", action="store_true")
args = ap.parse_args()
state = Path(args.state)
state.mkdir(parents=True, exist_ok=True)
attempts = int((state / "attempts").read_text()) if (state / "attempts").exists() else 0
(state / "attempts").write_text(str(attempts + 1))

print("fake-server starting", flush=True)
if args.mode == "crash":
    for i in range(40):
        print(f"log line {i}", flush=True)
    print("load_model: failed to load model 'broken.gguf'", flush=True)
    sys.exit(3)
if args.mode == "bind_once" and attempts == 0:
    print("start: couldn't bind HTTP server socket, hostname: 127.0.0.1", flush=True)
    sys.exit(1)
if args.mode == "retry_once" and attempts == 0:
    print("special failure that the engine knows how to fix", flush=True)
    sys.exit(2)
if args.mode == "grandchild":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    (state / "grandchild.pid").write_text(str(child.pid))
if args.mode == "ignore_term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

T0 = time.monotonic()
ACTIVE = {"n": 0}

def auth(request):
    if args.api_key and request.headers.get("Authorization") != f"Bearer {args.api_key}":
        raise web.HTTPUnauthorized(text=json.dumps({"error": {"code": 401, "message": "Invalid API Key",
                                   "type": "authentication_error"}}), content_type="application/json")

async def health(request):
    if args.mode == "hang" or time.monotonic() - T0 < args.load_delay:
        return web.json_response({"error": {"code": 503, "message": "Loading model"}}, status=503)
    return web.json_response({"status": "ok"})

async def props(request):
    auth(request)
    return web.json_response({
        "default_generation_settings": {"n_ctx": 1024},
        "total_slots": 2,
        "chat_template": "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if tools %}TOOLS{% endif %}",
        "bos_token": "<s>", "eos_token": "</s>",
        "chat_template_caps": {"supports_tools": True, "supports_system_role": True},
        "modalities": {"vision": False, "audio": False},
    })

async def slots(request):
    auth(request)
    return web.json_response([
        {"id": i, "n_ctx": 1024, "is_processing": i < ACTIVE["n"]} for i in range(2)])

async def apply_template(request):
    auth(request)
    body = await request.json()
    (state / "last_template.json").write_text(json.dumps(body))
    msgs = body["messages"]
    if msgs and msgs[0].get("content") == "RAISE":
        return web.json_response({"error": {"code": 400, "message": "Conversation roles must alternate",
                                            "type": "invalid_request_error"}}, status=400)
    out = "".join(f"<|{m['role']}|>{m.get('content', '')}" for m in msgs)
    if body.get("tools") and not args.no_tools:
        out = "TOOLS:" + ",".join(t["function"]["name"] for t in body["tools"]) + "|" + out
    kw = body.get("chat_template_kwargs") or {}
    if kw.get("enable_thinking") is False:
        out += "<think></think>"
    if body.get("add_generation_prompt", True):
        out += "<|assistant|>"
    return web.json_response({"prompt": "<s>" + out})

async def tokenize(request):
    auth(request)
    body = await request.json()
    words = body["content"].split()
    toks = [100 + i for i in range(len(words))]
    if body.get("add_special"):
        toks = [1] + toks
    return web.json_response({"tokens": toks})

async def completion(request):
    auth(request)
    body = await request.json()
    (state / "last_body.json").write_text(json.dumps(body))
    prompt = body["prompt"]
    if "OVERFLOW" in prompt:
        return web.json_response({"error": {"code": 400, "message": "request (3001 tokens) exceeds the available context size (1024 tokens), try increasing it",
                                  "type": "exceed_context_size_error", "n_prompt_tokens": 3001, "n_ctx": 1024}}, status=400)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    n = body.get("n_predict", 5)
    n = 5 if n is None or n < 0 else n
    ACTIVE["n"] += 1
    try:
        for i in range(n):
            if "MIDERROR" in prompt and i == 2:
                await resp.write(b'data: {"error": {"code": 500, "message": "boom", "type": "server_error"}}\n\n')
                return resp
            if "CRASH" in prompt and i == 2:
                print("fatal: simulated crash", flush=True)
                os._exit(7)
            frame = {"index": 0, "content": f"t{i} ", "stop": False, "tokens_predicted": i + 1}
            data = b"data: " + json.dumps(frame).encode() + b"\n\n"
            if i == 1:  # split an event across writes
                await resp.write(data[:7]); await asyncio.sleep(0.01); await resp.write(data[7:])
            else:
                await resp.write(data)
            await asyncio.sleep(0.03 if "SLOW" in prompt else 0.001)
        await resp.write(b": ping\n\n")
        final = {"index": 0, "content": "", "stop": True, "stop_type": "limit" if n == body.get("n_predict") else "eos",
                 "tokens_evaluated": 12, "tokens_predicted": n, "tokens_cached": 12 + n - 1,
                 "prompt": "P" * 300000,
                 "timings": {"cache_n": 8, "prompt_n": 4, "prompt_ms": 10.0, "prompt_per_second": 400.0,
                             "predicted_n": n, "predicted_ms": 50.0, "predicted_per_second": 100.0}}
        await resp.write(b"data: " + json.dumps(final).encode() + b"\r\n\r\n")
        return resp
    except (asyncio.CancelledError, ConnectionResetError):
        with open(state / "cancelled", "a") as fh:
            fh.write("1\n")
        raise
    finally:
        ACTIVE["n"] -= 1

async def v1_embeddings(request):
    auth(request)
    if args.mode == "native_embed":
        raise web.HTTPNotFound()
    body = await request.json()
    with open(state / "embed_calls", "a") as fh:
        fh.write(json.dumps(body["input"]) + "\n")
    data = [{"index": i, "object": "embedding", "embedding": [float(len(t)), 1.0, 0.5]} for i, t in enumerate(body["input"])]
    return web.json_response({"object": "list", "data": list(reversed(data))})

async def native_embedding(request):
    auth(request)
    body = await request.json()
    return web.json_response([{"index": i, "embedding": [[float(len(t)), 2.0, 0.5]]} for i, t in enumerate(body["content"])])

app = web.Application()
app.router.add_get("/health", health)
app.router.add_get("/props", props)
app.router.add_get("/slots", slots)
app.router.add_post("/apply-template", apply_template)
app.router.add_post("/tokenize", tokenize)
app.router.add_post("/completion", completion)
app.router.add_post("/v1/embeddings", v1_embeddings)
app.router.add_post("/embedding", native_embedding)
print("listening", flush=True)
web.run_app(app, host="127.0.0.1", port=args.port, print=None, shutdown_timeout=1,
            handle_signals=args.mode != "ignore_term", handler_cancellation=True)
'''


class FakeEngine(HttpEngine):
    format = "fake"
    startup_timeout_base_s = 30.0
    stop_grace_s = 2.0

    def __init__(self, script: Path, state: Path, mode: str = "ok", *, api_key: str | None = "k-123",
                 no_tools: bool = False, config: LoadConfig | None = None) -> None:
        model = LocalModel(key="local/fake", format="gguf", path=str(script), display_name="Fake", size_bytes=0)
        super().__init__("fake-1", model, config or LoadConfig(n_ctx=2048, n_parallel=2))
        self.script = script
        self.state = state
        self.mode = mode
        self.key = api_key
        self.no_tools = no_tools
        self.retry_requests = 0

    async def _before_spawn(self) -> None:
        self._api_key = self.key

    def _build_command(self, port: int) -> list[str]:
        cmd = [sys.executable, str(self.script), "--port", str(port), "--mode", self.mode, "--state", str(self.state)]
        if self.key:
            cmd += ["--api-key", self.key]
        if self.no_tools:
            cmd.append("--no-tools")
        return cmd

    def _on_start_failure(self, log_text: str, exit_code: int | None) -> bool:
        if "engine knows how to fix" in log_text:
            self.retry_requests += 1
            return True
        return False


@pytest.fixture
def fake_script(tmp_path: Path) -> Path:
    p = tmp_path / "fake_llama_server.py"
    p.write_text(FAKE_SERVER)
    return p


@pytest.fixture
def make_engine(fake_script: Path, tmp_path: Path, isolated_home: Path):
    engines: list[FakeEngine] = []

    def factory(mode: str = "ok", **kw: Any) -> FakeEngine:
        e = FakeEngine(fake_script, tmp_path / f"state-{len(engines)}", mode, **kw)
        engines.append(e)
        return e

    yield factory
    for e in engines:
        e.kill_now()


async def _stop_all(*engines: HttpEngine) -> None:
    for e in engines:
        await e.stop()


def _pid_gone(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


async def _collect(it) -> list[CompletionChunk]:
    out = []
    async with contextlib.aclosing(it) as agen:
        async for c in agen:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_sse_decoder_handles_split_crlf_comments_and_multiline():
    d = SSEDecoder()
    evs = d.feed(b'data: {"a"')
    assert evs == []
    evs = d.feed(b': 1}\r\n\r\n: ping\n\nevent: x\ndata: line1\ndata: line2\n\nerror: {"code": 1}\n\n')
    assert [(e.event, e.data) for e in evs] == [
        ("message", '{"a": 1}'),
        ("x", "line1\nline2"),
        ("error", '{"code": 1}'),
    ]
    assert d.feed(b"data: tail-without-blank-line") == []
    assert [(e.event, e.data) for e in d.finish()] == [("message", "tail-without-blank-line")]
    assert d.finish() == []


def test_sse_decoder_has_no_line_limit_and_handles_utf8_split():
    d = SSEDecoder()
    payload = json.dumps({"content": "zażółć", "prompt": "x" * 2_000_000}, ensure_ascii=False).encode()
    line = b"data: " + payload + b"\n\n"
    cut = line.index("ż".encode()) + 1  # split inside a multi-byte character
    evs = d.feed(line[:cut]) + d.feed(line[cut:])
    assert len(evs) == 1
    assert json.loads(evs[0].data)["content"] == "zażółć"


def test_stats_from_llama_frame_uses_cache_n_not_tokens_cached():
    frame = {
        "stop": True, "stop_type": "limit", "tokens_evaluated": 21, "tokens_predicted": 10, "tokens_cached": 30,
        "timings": {"cache_n": 3, "prompt_n": 18, "prompt_ms": 32.785, "prompt_per_second": 549.03,
                    "predicted_n": 10, "predicted_ms": 52.653, "predicted_per_second": 170.93},
    }
    s = stats_from_frame(frame)
    assert (s.prompt_tokens, s.cached_tokens, s.completion_tokens) == (21, 3, 10)
    assert s.prompt_tps == pytest.approx(549.03)
    assert s.gen_tps == pytest.approx(170.93)
    assert s.total_ms == pytest.approx(32.785 + 52.653)
    assert finish_reason_from_frame(frame) == "length"


def test_stats_fallbacks_for_worker_frames():
    # MLX-worker style: finish_reason + tokens_cached as the prefix hit, no timings
    frame = {"stop": True, "finish_reason": "stop", "tokens_evaluated": 50, "tokens_cached": 40, "tokens_predicted": 7}
    s = stats_from_frame(frame, wall_ms=123.0)
    assert (s.prompt_tokens, s.cached_tokens, s.completion_tokens, s.total_ms) == (50, 40, 7, 123.0)
    assert finish_reason_from_frame(frame) == "stop"
    # tokens_cached larger than the prompt is clamped; non-finite rates become 0
    frame = {"stop": True, "stop_type": "eos", "tokens_evaluated": 5, "tokens_cached": 99,
             "timings": {"prompt_per_second": float("inf"), "predicted_per_second": None}}
    s = stats_from_frame(frame)
    assert s.cached_tokens == 5 and s.prompt_tps == 0.0 and s.gen_tps == 0.0
    assert finish_reason_from_frame({"stop_type": "word"}) == "stop"
    assert finish_reason_from_frame({"finish_reason": "length"}) == "length"
    # prompt size derived from timings when tokens_evaluated is missing
    assert stats_from_frame({"timings": {"cache_n": 4, "prompt_n": 6}}).prompt_tokens == 10


def _engine_for_errors() -> FakeEngine:
    return FakeEngine(Path("/nonexistent"), Path("/nonexistent"))


def test_http_error_mapping():
    e = _engine_for_errors()
    body = json.dumps({"error": {"code": 400, "message": "request (3001 tokens) exceeds the available context size (2048 tokens), try increasing it",
                                 "type": "exceed_context_size_error", "n_prompt_tokens": 3001, "n_ctx": 2048}})
    err = e._http_error(400, body, "/completion")
    assert isinstance(err, ContextOverflowError) and isinstance(err, EngineError)
    assert (err.n_prompt_tokens, err.n_ctx, err.status, err.code) == (3001, 2048, 400, "context_overflow")
    assert "Zapytanie przekracza dostępny kontekst modelu (3001 > 2048 tokenów)" in str(err)

    # message-only variant (e.g. a worker that does not send the type)
    err = e._http_error(400, json.dumps({"error": {"message": "request (10 tokens) exceeds the available context size (8 tokens)"}}), "/completion")
    assert isinstance(err, ContextOverflowError) and err.n_ctx == 8

    body = json.dumps({"error": {"code": 500, "message": "input (2502 tokens) is too large to process. increase the physical batch size (current batch size: 2048)", "type": "server_error"}})
    err = e._http_error(500, body, "/v1/embeddings")
    assert isinstance(err, ContextOverflowError) and (err.n_prompt_tokens, err.n_ctx) == (2502, 2048)
    assert "embeddingu" in str(err)

    for status, needle in ((401, "autoryzac"), (404, "nie obsługuje"), (503, "zajęty"), (500, "Błąd silnika"), (400, "odrzucił")):
        err = e._http_error(status, json.dumps({"error": {"message": "x", "type": "t"}}), "/p")
        assert type(err) is EngineHttpError and err.status == status and needle in str(err)
    err = e._http_error(502, "<html>bad gateway</html>", "/p")
    assert err.server_message == "<html>bad gateway</html>"


def test_clean_messages_drops_reasoning_and_none_content():
    msgs = [{"role": "assistant", "content": None, "reasoning": "secret", "reasoning_content": "x",
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": {}}}]}]
    out = he._clean_messages(msgs)
    assert out == [{"role": "assistant", "content": "", "tool_calls": msgs[0]["tool_calls"]}]
    assert "reasoning" in msgs[0]  # input untouched


def test_heuristic_template_info():
    t = "{% if tools %}<tool_call>{% endif %}{% if m.role == 'tool' %}{% endif %}system enable_thinking <think>"
    info = he._heuristic_template_info(t)
    assert info.supports_tools and info.supports_tool_role and info.supports_system
    assert info.thinking_kwarg == "enable_thinking" and info.supports_thinking and info.tool_format == "hermes"
    assert he._heuristic_template_info(None) == TemplateInfo()


def test_call_local_renderer_supports_both_signatures():
    seen: dict[str, Any] = {}

    def new_style(template, messages, tools=None, *, add_generation_prompt=True, bos_token="", eos_token="", **kw):
        seen.update(kw, agp=add_generation_prompt, bos=bos_token, eos=eos_token)
        return "new"

    def old_style(template, messages, tools, add_generation_prompt, bos, eos, **kw):
        seen.update(kw, agp=add_generation_prompt, bos=bos, eos=eos)
        return "old"

    assert he._call_local_renderer(new_style, "t", [], None, False, "<s>", "</s>", {"enable_thinking": False}) == "new"
    assert seen == {"enable_thinking": False, "agp": False, "bos": "<s>", "eos": "</s>"}
    seen.clear()
    assert he._call_local_renderer(old_style, "t", [], None, True, "B", "E", {}) == "old"
    assert seen == {"agp": True, "bos": "B", "eos": "E"}


# --------------------------------------------------------------------------- #
# Lifecycle against the fake server
# --------------------------------------------------------------------------- #


async def test_start_reads_props_and_stop_kills(make_engine, isolated_home):
    e = make_engine()
    assert not e.running and e.pid is None and e.n_ctx == 2048
    await e.start()
    try:
        assert e.running and e.pid and e.port
        assert e.n_ctx == 1024          # per-slot value read back from /props & /slots
        assert e.n_parallel == 2
        assert e.modalities == {"vision": False, "audio": False}
        assert e.last_props["eos_token"] == "</s>"
        assert e._adds_bos is True      # /tokenize add_special prepends a token
        assert e.server_renders_tools and e.template_info.supports_tools
        assert e.template_info.supports_tool_role
        assert await e.health()
        mem = await e.memory_bytes()
        assert mem and mem > 1_000_000
        log_path = e.log_path
        assert log_path is not None and log_path.parent == isolated_home / "logs"
        assert "fake-server starting" in log_path.read_text()
        assert int((e.state / "attempts").read_text()) == 1
    finally:
        pid = e.pid
        await e.stop()
    assert pid is not None and _pid_gone(pid)
    assert not e.running and e.pid is None
    assert await e.memory_bytes() is None
    await e.stop()  # idempotent
    with pytest.raises(EngineUnavailableError):
        await e.tokenize("a b")


async def test_restart_after_stop_rotates_log(make_engine):
    e = make_engine()
    await e.start()
    await e.stop()
    await e.start()
    try:
        assert e.running
        assert e.log_path is not None and e.log_path.with_name(e.log_path.name + ".1").exists()
    finally:
        await e.stop()


async def test_concurrent_stop_calls(make_engine):
    e = make_engine()
    await e.start()
    pid = e.pid
    await asyncio.gather(e.stop(), e.stop(), e.stop())
    assert _pid_gone(pid)


async def test_early_crash_raises_with_log_tail(make_engine):
    e = make_engine("crash")
    with pytest.raises(EngineStartError) as ei:
        await e.start()
    msg = str(ei.value)
    assert "Nie udało się uruchomić modelu „Fake”" in msg
    assert "kod wyjścia 3" in msg
    assert "log line 39" in msg and "failed to load model" in msg
    assert "log line 5\n" not in msg  # only the tail (~30 lines)
    assert ei.value.exit_code == 3 and "log line 39" in ei.value.log_tail
    assert e.pid is None and not e.running


async def test_startup_timeout_kills_process(make_engine):
    e = make_engine("hang")
    e.startup_timeout_base_s = 1.5
    with pytest.raises(EngineStartError) as ei:
        await e.start()
    assert "nie uruchomił się w ciągu 1 s" in str(ei.value)
    assert "ładowanie" in str(ei.value)       # /health kept answering 503
    assert e.pid is None


async def test_port_race_is_retried(make_engine):
    e = make_engine("bind_once")
    await e.start()
    try:
        assert e.running
        assert int((e.state / "attempts").read_text()) == 2
    finally:
        await e.stop()


async def test_subclass_retry_hook(make_engine):
    e = make_engine("retry_once")
    await e.start()
    try:
        assert e.retry_requests == 1 and e.running
    finally:
        await e.stop()


async def test_sigterm_ignored_escalates_to_sigkill(make_engine):
    e = make_engine("ignore_term")
    e.stop_grace_s = 0.5
    await e.start()
    pid = e.pid
    t0 = time.monotonic()
    await e.stop()
    assert _pid_gone(pid)
    assert e.exit_code == -signal.SIGKILL
    assert time.monotonic() - t0 < 5


async def test_stop_kills_whole_process_group(make_engine):
    e = make_engine("grandchild")
    await e.start()
    gpid_file = e.state / "grandchild.pid"
    gpid = int(gpid_file.read_text())
    assert psutil.pid_exists(gpid)
    await e.stop()
    for _ in range(50):
        if _pid_gone(gpid):
            break
        await asyncio.sleep(0.05)
    assert _pid_gone(gpid)


async def test_stop_during_start_aborts(make_engine):
    e = make_engine("hang")
    task = asyncio.create_task(e.start())
    for _ in range(100):
        if e.pid:
            break
        await asyncio.sleep(0.05)
    pid = e.pid
    assert pid
    await e.stop()
    with pytest.raises(EngineError, match="przerwane"):
        await asyncio.wait_for(task, 10)
    assert _pid_gone(pid)


async def test_cancelled_start_cleans_up(make_engine):
    e = make_engine("hang")
    task = asyncio.create_task(e.start())
    for _ in range(100):
        if e.pid:
            break
        await asyncio.sleep(0.05)
    pid = e.pid
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pid and _pid_gone(pid)
    assert e.pid is None


async def test_emergency_kill_all(make_engine):
    e = make_engine()
    await e.start()
    pid = e.pid
    assert pid in he._LIVE_GROUPS
    kill_all_engine_processes(grace_s=0.1)
    assert pid not in he._LIVE_GROUPS
    await e.stop()  # still safe afterwards
    assert _pid_gone(pid)


async def test_kill_now(make_engine):
    e = make_engine()
    await e.start()
    pid = e.pid
    e.kill_now()
    assert not e.running
    await e.stop()
    assert _pid_gone(pid)


# --------------------------------------------------------------------------- #
# HTTP surface against the fake server
# --------------------------------------------------------------------------- #


@pytest.fixture
async def engine(make_engine):
    e = make_engine()
    await e.start()
    try:
        yield e
    finally:
        await e.stop()


async def test_api_key_is_sent_and_required(engine, make_engine):
    assert await engine.tokenize("a b c") == [100, 101, 102]
    assert await engine.tokenize("") == []
    engine._api_key = "wrong"
    with pytest.raises(EngineHttpError) as ei:
        await engine.tokenize("a")
    assert ei.value.status == 401
    engine._api_key = "k-123"


async def test_apply_template_passes_options(engine):
    tools = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}}}]
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok", "reasoning": "hidden"}]
    out = await engine.apply_template(msgs, tools, add_generation_prompt=False, template_kwargs={"enable_thinking": False})
    assert out == "<s>TOOLS:read_file|<|user|>hi<|assistant|>ok<think></think>"
    sent = json.loads((engine.state / "last_template.json").read_text())
    assert sent["add_generation_prompt"] is False
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning" not in sent["messages"][1]
    out = await engine.apply_template([{"role": "user", "content": "x"}])
    assert out.endswith("<|assistant|>") and "TOOLS" not in out
    sent = json.loads((engine.state / "last_template.json").read_text())
    assert "tools" not in sent and "chat_template_kwargs" not in sent
    with pytest.raises(EngineHttpError, match="Szablon czatu modelu nie przyjął wiadomości: Conversation roles"):
        await engine.apply_template([{"role": "system", "content": "RAISE"}])


async def test_complete_streams_chunks_and_stats(engine):
    params = SamplingParams(max_tokens=4, temperature=0.1, top_p=0.9, top_k=20, min_p=0.01,
                            repeat_penalty=1.1, stop=["\n\n"], seed=7)
    chunks = await _collect(engine.complete("<s>hello", params))
    assert "".join(c.text for c in chunks) == "t0 t1 t2 t3 "
    final = chunks[-1]
    assert final.done and final.finish_reason == "length"
    assert final.stats is not None
    assert (final.stats.prompt_tokens, final.stats.cached_tokens, final.stats.completion_tokens) == (12, 8, 4)
    assert final.stats.gen_tps == 100.0 and final.stats.total_ms == 60.0
    assert all(not c.done for c in chunks[:-1])
    body = json.loads((engine.state / "last_body.json").read_text())
    assert body["prompt"] == "hello"  # textual BOS stripped: the server adds BOS itself
    assert body["stream"] is True and body["cache_prompt"] is True and body["n_predict"] == 4
    assert (body["temperature"], body["top_p"], body["top_k"], body["min_p"]) == (0.1, 0.9, 20, 0.01)
    assert body["repeat_penalty"] == 1.1 and body["stop"] == ["\n\n"] and body["seed"] == 7

    chunks = await _collect(engine.complete("plain", SamplingParams(max_tokens=0)))
    body = json.loads((engine.state / "last_body.json").read_text())
    assert body["n_predict"] == -1 and "seed" not in body and body["prompt"] == "plain"
    assert chunks[-1].finish_reason == "stop"


async def _wait_file(path: Path, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        await asyncio.sleep(0.02)
    return False


async def test_aclose_cancels_generation_on_server(engine):
    agen = engine.complete("SLOW story", SamplingParams(max_tokens=500))
    got = []
    async for c in agen:
        got.append(c)
        if len(got) == 3:
            break
    assert (await engine.slots())[0]["is_processing"] is True
    await agen.aclose()
    assert await _wait_file(engine.state / "cancelled")
    for _ in range(50):
        if not any(s["is_processing"] for s in await engine.slots()):
            break
        await asyncio.sleep(0.02)
    assert not any(s["is_processing"] for s in await engine.slots())


async def test_task_cancellation_cancels_generation(engine):
    async def consume():
        async for _ in engine.complete("SLOW story", SamplingParams(max_tokens=500)):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _wait_file(engine.state / "cancelled")


async def test_context_overflow(engine):
    with pytest.raises(ContextOverflowError) as ei:
        await _collect(engine.complete("OVERFLOW", SamplingParams()))
    assert ei.value.n_prompt_tokens == 3001 and ei.value.n_ctx == 1024


async def test_error_frame_mid_stream(engine):
    with pytest.raises(EngineHttpError) as ei:
        await _collect(engine.complete("MIDERROR", SamplingParams(max_tokens=10)))
    assert ei.value.status == 500 and "boom" in str(ei.value)


async def test_crash_mid_stream_reports_unavailable_with_log(engine):
    with pytest.raises(EngineUnavailableError) as ei:
        await _collect(engine.complete("CRASH", SamplingParams(max_tokens=10)))
    msg = str(ei.value)
    assert "przestał działać" in msg and "kod wyjścia 7" in msg and "simulated crash" in msg
    assert not engine.running
    with pytest.raises(EngineUnavailableError):
        await _collect(engine.complete("again", SamplingParams()))


async def test_concurrent_completions(engine):
    results = await asyncio.gather(*(_collect(engine.complete(f"q{i}", SamplingParams(max_tokens=6))) for i in range(3)))
    for chunks in results:
        assert chunks[-1].done and chunks[-1].stats.completion_tokens == 6


async def test_embeddings_openai_endpoint_batched_and_ordered(engine):
    engine.embed_batch_size = 2
    assert await engine.embed([]) == []
    vecs = await engine.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [v[0] for v in vecs] == [1.0, 2.0, 3.0, 4.0, 5.0]
    calls = (engine.state / "embed_calls").read_text().splitlines()
    assert [json.loads(c) for c in calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]


async def test_embeddings_fallback_to_native_endpoint(make_engine):
    e = make_engine("native_embed")
    await e.start()
    try:
        vecs = await e.embed(["xy", "z"])
        assert vecs == [[2.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
        assert e._embed_endpoint == "/embedding"
    finally:
        await e.stop()


async def test_local_tool_rendering_fallback(make_engine, monkeypatch):
    rendered: dict[str, Any] = {}

    def fake_render(template, messages, tools=None, *, add_generation_prompt=True, bos_token="", eos_token="", **kw):
        rendered.update(template=template, tools=tools, bos=bos_token, kw=kw)
        return "LOCAL:" + ",".join(t["function"]["name"] for t in tools or [])

    monkeypatch.setattr(he, "_local_renderer", lambda: fake_render)
    e = make_engine(no_tools=True)
    await e.start()
    try:
        assert not e.server_renders_tools
        assert e.local_tools_render and e.template_info.supports_tools
        tools = [{"type": "function", "function": {"name": "grep", "parameters": {}}}]
        out = await e.apply_template([{"role": "user", "content": "q"}], tools, template_kwargs={"enable_thinking": True})
        assert out == "LOCAL:grep"
        assert rendered["bos"] == "<s>" and rendered["kw"] == {"enable_thinking": True}
        assert "{% for m in messages %}" in rendered["template"]
        # without tools the server renders as usual
        assert (await e.apply_template([{"role": "user", "content": "q"}])).startswith("<s><|user|>q")
    finally:
        await e.stop()


async def test_no_tool_support_without_local_renderer(make_engine, monkeypatch):
    monkeypatch.setattr(he, "_local_renderer", lambda: None)
    e = make_engine(no_tools=True)
    await e.start()
    try:
        assert not e.server_renders_tools and not e.local_tools_render
        assert e.template_info.supports_tools is False
    finally:
        await e.stop()
