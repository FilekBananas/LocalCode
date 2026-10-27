"""LM Studio-style OpenAI-compatible local server (``/v1/*``).

Endpoints: ``GET /v1/models``, ``POST /v1/chat/completions`` (streaming and
non-streaming, tools included), ``POST /v1/completions``, ``POST /v1/embeddings``
and ``POST /v1/images/generations`` (local text-to-image, see ``docs/IMAGES.md``).

How a request becomes tokens:

1. ``model`` is resolved through the library — an exact ``LocalModel.key`` or
   any unique substring (``library.resolve``), so ``"qwen3-4b"`` works.
2. :meth:`RuntimeManager.acquire` loads the model just in time and lends one
   request slot for the duration of the call.
3. :class:`~localcode.runtime.formatter.ChatFormatter` renders the messages
   (and the tool schemas, natively or as a prompted block) with the model's own
   chat template.
4. The raw completion is split by :mod:`localcode.runtime.toolcalls`: visible
   text, ``reasoning_content`` and OpenAI ``tool_calls`` (arguments are handed
   back as a JSON **string**, like the real API).

``/v1/images/generations`` defaults to ``response_format: "b64_json"``, unlike
api.openai.com: the bytes are in the answer, so nothing else has to be fetched.
There is deliberately **no** unauthenticated file route — ``response_format:
"url"`` returns the absolute address of ``/api/images/{id}/file``, which sits
behind the same bearer token as the rest of ``/api/*``. Ask for it only from a
client that sends that token itself (the Mac app, the terminal, ``curl -H``);
an ``openai`` SDK will not add it when it downloads the picture for you.

Differences from api.openai.com: only ``n=1`` for text; sampling defaults are tuned for
small local models (temperature 0.3, top_p 0.95, top_k 40, min_p 0.05);
``usage`` counts come from the engine, prices do not exist. Errors use the
OpenAI shape ``{"error": {"message", "type", "code"}}`` so the ``openai``
client libraries display them properly.

Availability is controlled by ``settings.server.openai_api_enabled``;
``settings.server.openai_api_requires_token`` decides whether the bearer token
is required here as well (see ``server/app.py``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Iterable

from aiohttp import web

from localcode.chat.repo import ApiError
from localcode.runtime.base import CompletionChunk, EngineError, SamplingParams
from localcode.runtime.formatter import ChatFormatter
from localcode.runtime.templates import normalize_effort
from localcode.server.app import (
    SSE_HEADERS,
    error_payload,
    get_core,
    json_body,
    json_response,
    sse_write,
)
from localcode.server.core import Core

log = logging.getLogger(__name__)

__all__ = ["register"]

DEFAULT_MAX_TOKENS = 1024
MAX_STOP_STRINGS = 8
#: Most pictures one ``/v1/images/generations`` request may ask for.
MAX_IMAGES = 4


def register(app: web.Application) -> None:
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/embeddings", embeddings)
    app.router.add_post("/v1/images/generations", images_generations)
    # Last: ``{model:.+}`` would also swallow "images/generations".
    app.router.add_get("/v1/models/{model:.+}", retrieve_model)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class OpenAIError(Exception):
    """Error rendered in the OpenAI shape."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code

    def response(self) -> web.Response:
        body = {"error": {"message": self.message, "type": self.type, "code": self.code, "param": None}}
        return json_response(body, status=self.status)


def openai_error_response(exc: BaseException) -> web.Response:
    if isinstance(exc, OpenAIError):
        return exc.response()
    if isinstance(exc, ApiError):
        return OpenAIError(exc.status, exc.message, "invalid_request_error", exc.code).response()
    if isinstance(exc, EngineError):
        return OpenAIError(503, str(exc), "server_error", "engine_error").response()
    log.exception("błąd /v1")
    return OpenAIError(500, "Wewnętrzny błąd serwera.", "server_error", "internal_error").response()


def _guard(handler: Any) -> Any:
    """Wrap a handler so every failure is reported in the OpenAI error shape."""

    async def wrapper(request: web.Request) -> web.StreamResponse:
        try:
            return await handler(request)
        except asyncio.CancelledError:
            raise
        except web.HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to an OpenAI error body
            return openai_error_response(exc)

    wrapper.__name__ = handler.__name__
    wrapper.__doc__ = handler.__doc__
    return wrapper


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def _model_dto(core: Core, model: Any) -> dict[str, Any]:
    return {
        "id": model.key,
        "object": "model",
        "created": int(model.modified_at or 0),
        "owned_by": "localcode",
        "type": "embeddings" if model.is_embedding else "llm",
        "format": model.format,
        "quantization": model.quant,
        "context_length": model.meta.context_length,
        "capabilities": list(model.capabilities),
        "state": "loaded" if core.runtime.is_loaded(model.key) else "not-loaded",
    }


@_guard
async def list_models(request: web.Request) -> web.Response:
    """``GET /v1/models``."""
    core = get_core(request)
    models = await asyncio.to_thread(core.library.list)
    return json_response({"object": "list", "data": [_model_dto(core, m) for m in models]})


@_guard
async def retrieve_model(request: web.Request) -> web.Response:
    """``GET /v1/models/{id}``."""
    core = get_core(request)
    model = core.resolve_model(request.match_info["model"])
    if model is None:
        raise OpenAIError(404, f"Nie znaleziono modelu „{request.match_info['model']}”.",
                          "invalid_request_error", "model_not_found")
    return json_response(_model_dto(core, model))


# --------------------------------------------------------------------------- #
# Request parsing
# --------------------------------------------------------------------------- #


async def _body(request: web.Request) -> dict[str, Any]:
    body = await json_body(request)
    if not isinstance(body, dict):
        raise OpenAIError(400, "Treść żądania musi być obiektem JSON.")
    if body.get("n") not in (None, 1):
        raise OpenAIError(400, "Obsługiwana jest tylko jedna odpowiedź na żądanie (n=1).")
    return body


def _resolve(core: Core, body: dict[str, Any]) -> Any:
    name = body.get("model")
    if not isinstance(name, str) or not name.strip():
        raise OpenAIError(400, "Brak pola „model”.", "invalid_request_error", "missing_model")
    model = core.resolve_model(name.strip())
    if model is None:
        raise OpenAIError(404, f"Nie znaleziono modelu „{name}”. Sprawdź GET /v1/models.",
                          "invalid_request_error", "model_not_found")
    return model


def _number(body: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenAIError(400, f"Pole „{key}” musi być liczbą.")
    return max(lo, min(float(value), hi))


def _stops(body: dict[str, Any], extra: Iterable[str] = ()) -> list[str]:
    raw = body.get("stop")
    stops: list[str] = []
    if isinstance(raw, str):
        stops = [raw]
    elif isinstance(raw, list):
        stops = [s for s in raw if isinstance(s, str)]
    elif raw is not None:
        raise OpenAIError(400, "Pole „stop” musi być tekstem albo listą tekstów.")
    for s in extra:
        if s and s not in stops:
            stops.append(s)
    return stops[:MAX_STOP_STRINGS]


def _sampling(body: dict[str, Any], *, extra_stops: Iterable[str] = ()) -> SamplingParams:
    defaults = SamplingParams()
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    if max_tokens is None:
        limit = DEFAULT_MAX_TOKENS
    elif isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise OpenAIError(400, "Pole „max_tokens” musi być dodatnią liczbą całkowitą.")
    else:
        limit = max_tokens
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise OpenAIError(400, "Pole „seed” musi być liczbą całkowitą.")
    return SamplingParams(
        max_tokens=limit,
        temperature=_number(body, "temperature", defaults.temperature, 0.0, 2.0),
        top_p=_number(body, "top_p", defaults.top_p, 0.0, 1.0),
        top_k=int(_number(body, "top_k", defaults.top_k, 0, 1000)),
        min_p=_number(body, "min_p", defaults.min_p, 0.0, 1.0),
        repeat_penalty=_number(body, "repetition_penalty", defaults.repeat_penalty, 0.5, 2.0),
        stop=_stops(body, extra_stops),
        seed=seed,
    )


def _content_text(content: Any) -> str:
    """OpenAI content: a string, or a list of parts (only text parts are used)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text", "input_text"):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return str(content)


def _messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise OpenAIError(400, "Pole „messages” musi być niepustą listą.")
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise OpenAIError(400, "Każda wiadomość musi mieć pole „role”.")
        message: dict[str, Any] = {"role": item["role"], "content": _content_text(item.get("content"))}
        calls = item.get("tool_calls")
        if isinstance(calls, list) and calls:
            message["tool_calls"] = [_normalise_call(c) for c in calls if isinstance(c, dict)]
        for key in ("tool_call_id", "name"):
            if isinstance(item.get(key), str):
                message[key] = item[key]
        out.append(message)
    return out


def _normalise_call(call: dict[str, Any]) -> dict[str, Any]:
    """OpenAI sends ``arguments`` as a JSON string; the core uses dicts."""
    fn = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = fn.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            arguments = {"_raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {}
    return {"id": str(call.get("id") or uuid.uuid4().hex[:9]), "type": "function",
            "function": {"name": str(fn.get("name") or ""), "arguments": arguments}}


def _tools(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = body.get("tools")
    choice = body.get("tool_choice")
    if choice == "none":
        return None
    if tools is None:
        functions = body.get("functions")
        if isinstance(functions, list) and functions:
            tools = [{"type": "function", "function": f} for f in functions if isinstance(f, dict)]
        else:
            return None
    if not isinstance(tools, list) or not tools:
        return None
    clean = [t for t in tools if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    return clean or None


def _thinking(body: dict[str, Any]) -> bool:
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return bool(kwargs["enable_thinking"])
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort.lower() in ("none", "off", "minimal"):
        return False
    return True


def _effort(body: dict[str, Any]) -> str | None:
    """``reasoning_effort`` from the request (OpenAI spelling), or ``None``.

    ``"none"``/``"minimal"`` are OpenAI synonyms for "do not think"; every other
    value is normalised to ``off|low|medium|high`` and resolved per model by the
    formatter. An explicit ``chat_template_kwargs.enable_thinking`` still wins,
    because that addresses the template directly.
    """
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, dict) and "enable_thinking" in kwargs:
        return None
    effort = body.get("reasoning_effort")
    if not isinstance(effort, str):
        return None
    text = effort.strip().lower()
    if text in ("none", "minimal"):
        return "off"
    return normalize_effort(text)


# --------------------------------------------------------------------------- #
# Chat completions
# --------------------------------------------------------------------------- #


def _usage(stats: Any, fallback_prompt: int = 0) -> dict[str, int]:
    prompt = int(getattr(stats, "prompt_tokens", 0) or fallback_prompt)
    completion = int(getattr(stats, "completion_tokens", 0) or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def _tool_calls_dto(calls: Iterable[Any]) -> list[dict[str, Any]]:
    out = []
    for index, call in enumerate(calls):
        out.append({
            "id": call.id,
            "type": "function",
            "index": index,
            "function": {"name": call.name,
                         "arguments": json.dumps(call.arguments, ensure_ascii=False)},
        })
    return out


@_guard
async def chat_completions(request: web.Request) -> web.StreamResponse:
    """``POST /v1/chat/completions`` — streaming and non-streaming, with tools."""
    core = get_core(request)
    body = await _body(request)
    model = _resolve(core, body)
    messages = _messages(body)
    tools = _tools(body)
    thinking = _thinking(body)
    effort = _effort(body)
    stream = bool(body.get("stream"))
    include_usage = bool((body.get("stream_options") or {}).get("include_usage")) if stream else False

    lease = await core.runtime.acquire(model.key, purpose="chat")
    try:
        formatter = ChatFormatter(lease.engine)
        prompt = await formatter.render(messages, tools, thinking=thinking, effort=effort)
        params = _sampling(body, extra_stops=formatter.stop_strings())
        if stream:
            return await _stream_chat(request, lease, formatter, prompt, params, tools,
                                      model.key, include_usage)
        return await _complete_chat(lease, formatter, prompt, params, tools, model.key)
    finally:
        lease.release()


async def _drain(engine: Any, prompt: str, params: SamplingParams) -> tuple[str, Any, str | None]:
    """Collect a whole completion: ``(text, stats, finish_reason)``."""
    text: list[str] = []
    stats: Any = None
    finish: str | None = None
    stream = engine.complete(prompt, params)
    try:
        async for chunk in stream:
            if chunk.text:
                text.append(chunk.text)
            if chunk.done:
                stats = chunk.stats
                finish = chunk.finish_reason
    finally:
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()
    return "".join(text), stats, finish


async def _complete_chat(lease: Any, formatter: ChatFormatter, prompt: str, params: SamplingParams,
                         tools: list[dict[str, Any]] | None, model_key: str) -> web.Response:
    raw, stats, reason = await _drain(lease.engine, prompt, params)
    lease.record_stats(stats)
    parsed = formatter.parse(raw, prompt, tools)
    message: dict[str, Any] = {"role": "assistant", "content": parsed.content or ""}
    if parsed.reasoning:
        message["reasoning_content"] = parsed.reasoning
    finish = "length" if reason == "length" else "stop"
    if parsed.tool_calls:
        message["tool_calls"] = _tool_calls_dto(parsed.tool_calls)
        message["content"] = parsed.content or None
        finish = "tool_calls"
    return json_response({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_key,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": _usage(stats),
    })


async def _stream_chat(request: web.Request, lease: Any, formatter: ChatFormatter, prompt: str,
                       params: SamplingParams, tools: list[dict[str, Any]] | None,
                       model_key: str, include_usage: bool) -> web.StreamResponse:
    response = web.StreamResponse(status=200, headers=SSE_HEADERS)
    await response.prepare(request)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def envelope(delta: dict[str, Any] | None, finish: str | None = None,
                 usage: dict[str, int] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model_key,
            "choices": ([] if delta is None and finish is None
                        else [{"index": 0, "delta": delta or {}, "finish_reason": finish,
                               "logprobs": None}]),
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    splitter = formatter.splitter(prompt, tools)
    collected: list[str] = []
    stats: Any = None
    finish = "stop"
    stream: AsyncIterator[CompletionChunk] = lease.engine.complete(prompt, params)
    await _send(response, envelope({"role": "assistant", "content": ""}))
    try:
        async for chunk in stream:
            if chunk.text:
                collected.append(chunk.text)
                for kind, piece in splitter.feed(chunk.text):
                    if not piece:
                        continue
                    key = "content" if kind == "text" else "reasoning_content"
                    if not await _send(response, envelope({key: piece})):
                        return response          # client gone; `finally` stops the engine
            if chunk.done:
                stats = chunk.stats
                if chunk.finish_reason == "length":
                    finish = "length"
        for kind, piece in splitter.finish():
            if piece:
                key = "content" if kind == "text" else "reasoning_content"
                if not await _send(response, envelope({key: piece})):
                    return response
    finally:
        lease.record_stats(stats)
        await _close(stream)
    parsed = formatter.parse("".join(collected), prompt, tools)
    if parsed.tool_calls:
        finish = "tool_calls"
        await _send(response, envelope({"tool_calls": _tool_calls_dto(parsed.tool_calls)}))
    await _send(response, envelope({}, finish))
    if include_usage:
        await _send(response, envelope(None, None, _usage(stats)))
    await sse_write(response, b"data: [DONE]\n\n")
    with contextlib.suppress(Exception):
        await response.write_eof()
    return response


async def _send(response: web.StreamResponse, payload: dict[str, Any]) -> bool:
    return await sse_write(response, b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n")


async def _close(stream: Any) -> None:
    """Stop the engine generating (client disconnect, error, normal end)."""
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        with contextlib.suppress(Exception):
            await aclose()


# --------------------------------------------------------------------------- #
# Text completions
# --------------------------------------------------------------------------- #


@_guard
async def completions(request: web.Request) -> web.StreamResponse:
    """``POST /v1/completions`` — raw completion, no chat template."""
    core = get_core(request)
    body = await _body(request)
    model = _resolve(core, body)
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        if len(prompt) != 1 or not isinstance(prompt[0], str):
            raise OpenAIError(400, "Pole „prompt” musi być tekstem (obsługiwany jest jeden prompt).")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise OpenAIError(400, "Brak pola „prompt”.")
    params = _sampling(body)
    stream = bool(body.get("stream"))
    lease = await core.runtime.acquire(model.key, purpose="chat")
    try:
        if stream:
            return await _stream_completion(request, lease, prompt, params, model.key)
        text, stats, reason = await _drain(lease.engine, prompt, params)
        lease.record_stats(stats)
        finish = "length" if reason == "length" else "stop"
        return json_response({
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model.key,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}],
            "usage": _usage(stats),
        })
    finally:
        lease.release()


async def _stream_completion(request: web.Request, lease: Any, prompt: str, params: SamplingParams,
                             model_key: str) -> web.StreamResponse:
    response = web.StreamResponse(status=200, headers=SSE_HEADERS)
    await response.prepare(request)
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    stats: Any = None
    finish = "stop"
    stream = lease.engine.complete(prompt, params)

    def envelope(text: str, finish_reason: str | None) -> dict[str, Any]:
        return {"id": completion_id, "object": "text_completion", "created": created,
                "model": model_key,
                "choices": [{"index": 0, "text": text, "finish_reason": finish_reason,
                             "logprobs": None}]}

    try:
        async for chunk in stream:
            if chunk.text and not await _send(response, envelope(chunk.text, None)):
                return response
            if chunk.done:
                stats = chunk.stats
                if chunk.finish_reason == "length":
                    finish = "length"
    finally:
        lease.record_stats(stats)
        await _close(stream)
    await _send(response, envelope("", finish))
    await sse_write(response, b"data: [DONE]\n\n")
    with contextlib.suppress(Exception):
        await response.write_eof()
    return response


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


@_guard
async def embeddings(request: web.Request) -> web.Response:
    """``POST /v1/embeddings {model, input}``."""
    core = get_core(request)
    body = await _body(request)
    model = _resolve(core, body)
    raw = body.get("input")
    if isinstance(raw, str):
        texts = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(t, str) for t in raw):
        texts = list(raw)
    else:
        raise OpenAIError(400, "Pole „input” musi być tekstem albo niepustą listą tekstów.")
    async with await core.runtime.acquire(model.key, purpose="embedding") as lease:
        vectors = await lease.engine.embed(texts)
        tokens = 0
        with contextlib.suppress(Exception):
            tokens = sum(await lease.engine.count_tokens(t) for t in texts)
    return json_response({
        "object": "list",
        "model": model.key,
        "data": [{"object": "embedding", "index": i, "embedding": list(v)} for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    })


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _image_size(core: Core, body: dict[str, Any], model: Any) -> tuple[int, int]:
    """``size`` → ``(width, height)``; ``"auto"``/missing uses the settings default."""
    from localcode.images.gallery import ImageGalleryError, parse_size, settings_view

    raw = body.get("size")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "auto")):
        raw = settings_view(core.settings).default_size
    if not isinstance(raw, str):
        raise OpenAIError(400, "Pole „size” musi być tekstem w formacie SZERxWYS, np. 1024x1024.")
    try:
        return parse_size(raw)
    except ImageGalleryError as exc:
        raise OpenAIError(400, str(exc), "invalid_request_error", "invalid_size") from exc


def _image_count(body: dict[str, Any]) -> int:
    value = body.get("n")
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OpenAIError(400, "Pole „n” musi być dodatnią liczbą całkowitą.")
    if value > MAX_IMAGES:
        raise OpenAIError(400, f"Naraz można wygenerować najwyżej {MAX_IMAGES} obrazy.")
    return value


def _image_seed(body: dict[str, Any]) -> int | None:
    seed = body.get("seed")
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise OpenAIError(400, "Pole „seed” musi być liczbą całkowitą.")
    return seed


async def _image_payload(request: web.Request, record: Any, prompt: str, want_url: bool) -> dict[str, Any]:
    import base64
    from pathlib import Path

    item: dict[str, Any] = {"revised_prompt": prompt}
    if want_url:
        item["url"] = str(request.url.with_query(None).with_path(record.to_dict()["url"]))
        return item
    path = Path(record.path)
    try:
        raw = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise OpenAIError(500, f"Nie udało się odczytać wygenerowanego obrazu: {exc}",
                          "server_error", "file_unreadable") from exc
    item["b64_json"] = base64.b64encode(raw).decode("ascii")
    return item


@_guard
async def images_generations(request: web.Request) -> web.Response:
    """``POST /v1/images/generations {prompt, model?, n?, size?, response_format?, seed?}``."""
    from localcode.server.routes_images import build_params, generate_records, resolve_image_model

    core = get_core(request)
    body = await json_body(request)
    if not isinstance(body, dict):
        raise OpenAIError(400, "Treść żądania musi być obiektem JSON.")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise OpenAIError(400, "Brak pola „prompt”.", "invalid_request_error", "missing_prompt")
    prompt = prompt.strip()
    fmt = body.get("response_format") or "b64_json"
    if fmt not in ("b64_json", "url"):
        raise OpenAIError(400, "Pole „response_format” może mieć wartość „b64_json” albo „url”.")
    model = resolve_image_model(core, body.get("model") if isinstance(body.get("model"), str) else None)
    width, height = _image_size(core, body, model)
    negative = str(body.get("negative_prompt") or "")
    params = build_params(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        steps=_optional_int(body, "steps"),
        guidance=_optional_float(body, "guidance"),
        seed=_image_seed(body),
        count=_image_count(body),
    )
    try:
        records = await generate_records(core, model.key, params, model_name=model.display_name,
                                         negative_prompt=negative)
    except (OpenAIError, ApiError, asyncio.CancelledError):
        raise
    except Exception as exc:  # noqa: BLE001 - Polish message, OpenAI envelope
        status, payload = error_payload(exc)
        raise OpenAIError(status, str(payload.get("error") or exc),
                          "server_error" if status >= 500 else "invalid_request_error",
                          payload.get("code")) from exc
    data = [await _image_payload(request, record, prompt, fmt == "url") for record in records]
    return json_response({"created": int(time.time()), "data": data})


def _optional_int(body: dict[str, Any], name: str) -> int | None:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenAIError(400, f"Pole „{name}” musi być dodatnią liczbą całkowitą.")
    return value


def _optional_float(body: dict[str, Any], name: str) -> float | None:
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenAIError(400, f"Pole „{name}” musi być liczbą.")
    return float(value)
