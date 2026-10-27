"""MLX worker process: a llama-server compatible HTTP facade over ``mlx_lm``.

Usage::

    python -m localcode.runtime.mlx_worker --model <dir> --host 127.0.0.1 --port <p> \\
        --parallel N --ctx N [--kv-bits 8|4] [--memory-limit-mb M] [--parent-pid PID]

Why a separate process that speaks llama-server's protocol
----------------------------------------------------------
LocalCode runs every model instance in its own subprocess (killing the process
is the only reliable way to hand memory back to the OS).  GGUF models run in
llama.cpp's ``llama-server``; MLX models run here.  This worker serves *the same*
HTTP subset with *the same* JSON shapes (recorded from llama-server build 10964)
so one client, ``runtime/http_engine.HttpEngine``, drives both engines unchanged:

=========================  ==================================================
``GET  /health``           503 ``{"error":{"code":503,"message":"Loading model",
                           "type":"unavailable_error"}}`` while loading, then
                           200 ``{"status":"ok"}``.  Every other endpoint also
                           answers that 503 until the model is ready.
``GET  /props``            llama-server keys: ``default_generation_settings``
                           (``n_ctx`` = context PER SLOT, ``params``),
                           ``total_slots``, ``model_path``, ``modalities``,
                           ``chat_template``, ``chat_template_caps``,
                           ``bos_token``, ``eos_token``, ``build_info`` (+ a few
                           MLX extras: ``n_ctx``, ``engine``, ``batching``,
                           ``kv_bits``).
``POST /apply-template``   ``{messages, tools?, add_generation_prompt?=true,
                           chat_template_kwargs?}`` -> ``{"prompt": str}``.  A
                           trailing assistant message is *continued* (prefill),
                           exactly like llama-server.
``POST /tokenize``         ``{content, add_special?=false, parse_special?=true,
                           with_pieces?=false}`` -> ``{"tokens": [...]}``.
``POST /detokenize``       ``{tokens}`` -> ``{"content": str}``.
``POST /completion``       raw completion of an already templated prompt, SSE
                           (``stream: true``) or a single JSON object.
``GET  /slots``            per-slot status (minimal mirror of llama-server).
``GET  /memory``           MLX allocator statistics in bytes.
``POST /embedding(s)``     501 ``not_supported_error`` (MLX worker is text-only).
=========================  ==================================================

Threading model
---------------
* The asyncio loop (main thread) owns HTTP.  The socket is bound *before* the
  model is loaded, so ``/health`` can report 503 while weights load.
* ONE generation thread owns every MLX evaluation: model loading, warm-up,
  prefill, decoding and KV-cache surgery (MLX streams must not be driven from
  several threads).  The loop talks to it through a command queue
  (submit / cancel / stop); the thread answers through
  ``loop.call_soon_threadsafe`` into a per-request ``asyncio.Queue``.
* A single-worker executor runs tokenizer and chat-template work (pure CPU, no
  MLX) so the event loop never blocks on Jinja or tokenisation.

Scheduling, batching and prefix reuse
-------------------------------------
* ``--parallel N`` slots.  Requests beyond N wait in a FIFO (like llama-server).
* Batch backend: ``mlx_lm.generate.BatchGenerator`` (continuous batching, prompt
  prefill interleaved with decoding of the other slots) whenever every cache
  layer of the model supports ``merge`` and no KV quantisation is requested.
* Sequential backend: ``mlx_lm.generate.generate_step`` one request at a time.
  Used for architectures that cannot batch and for ``--kv-bits`` (mlx_lm has no
  batched quantised KV cache).  Requests are still accepted concurrently; they
  queue for the single active slot.  If KV quantisation fails for a model
  (e.g. head size not divisible by the group size) the worker logs a warning
  and falls back to an unquantised cache.
* Prefix cache: ``mlx_lm.models.cache.LRUPromptCache`` (a token trie of KV
  snapshots).  Every finished *or cancelled* request stores its KV cache keyed
  by the exact tokens it holds; a new prompt reuses the longest cached prefix
  (trimming a longer entry when the cache type allows it) and only prefills the
  remaining suffix.  Its byte budget defaults to ``ctx * parallel`` tokens worth
  of KV (what the runtime's memory estimate reserves), minus the KV currently
  held by active requests.
* Cancellation: an HTTP client disconnect cancels the aiohttp handler
  (``handler_cancellation=True``), which sends ``cancel`` to the generation
  thread; the sequence is removed from the batch at the next step (or at the
  next prefill chunk in the sequential backend) and its slot is freed.

Text handling
-------------
* Detokenisation uses an incremental "prefix window" decoder (the vLLM / TGI
  algorithm) on top of ``tokenizer.decode`` with a few prompt tokens as context.
  It reproduces ``decode(all_tokens)`` exactly, holds back incomplete UTF-8 and,
  unlike mlx_lm's streaming BPE detokenizer, does not drop the leading space of
  the first generated token (llama-server keeps it too).
* Stop strings are matched on text across token boundaries: text that could be
  the beginning of a stop string is held back and never emitted; the stop string
  itself is never emitted.  EOS tokens are detected per token id.

Protocol notes (identical to llama-server unless stated)
--------------------------------------------------------
* Context overflow: a prompt with ``n_prompt >= n_ctx`` tokens is rejected with
  HTTP 400 ``{"error":{"code":400,"message":"request (N tokens) exceeds the
  available context size (C tokens), try increasing it","type":
  "exceed_context_size_error","n_prompt_tokens":N,"n_ctx":C}}`` (also for
  ``stream: true`` - the error comes before any SSE frame).  A prompt that fits
  while ``n_prompt + n_predict`` does not is served until the window is full and
  finishes with ``stop_type: "limit"``, ``truncated: true``.
* ``tokens_cached`` is the number of tokens held in the KV cache after the
  request (llama-server's ``n_past``); the prompt tokens *reused* from the
  prefix cache are ``timings.cache_n`` and the freshly processed ones
  ``timings.prompt_n``.
* ``n_predict`` absent / negative = until EOS or the end of the window;
  ``n_predict: 0`` produces one token (llama-server does the same).
* Errors after streaming started are sent as one SSE frame
  ``data: {"error": {...}}`` and the stream ends.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from aiohttp import web

log = logging.getLogger("localcode.mlx_worker")

#: llama.cpp's "random seed" marker (LLAMA_DEFAULT_SEED).
LLAMA_DEFAULT_SEED = 0xFFFFFFFF
#: One model per worker, so the prompt cache needs a single model key.
_PROMPT_CACHE_KEY = "model"
#: Fallback template when a model ships none (llama-server falls back to ChatML too).
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + (message['content'] or '') + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
#: chat_template_kwargs that would clash with apply_chat_template's own parameters.
_RESERVED_TEMPLATE_KWARGS = frozenset(
    {
        "conversation",
        "messages",
        "tools",
        "documents",
        "chat_template",
        "add_generation_prompt",
        "continue_final_message",
        "tokenize",
        "padding",
        "truncation",
        "max_length",
        "return_tensors",
        "return_dict",
        "return_assistant_tokens_mask",
        "tokenizer_kwargs",
    }
)

_dumps = functools.partial(json.dumps, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Errors (llama-server shape)
# --------------------------------------------------------------------------- #


class ApiError(Exception):
    """An HTTP error rendered exactly like llama-server's
    ``{"error": {"code", "message", "type", ...extra}}``."""

    def __init__(self, status: int, message: str, err_type: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.extra = extra

    def to_json(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.status, "message": self.message, "type": self.err_type}
        err.update(self.extra)
        return {"error": err}

    def response(self) -> web.Response:
        return web.json_response(self.to_json(), status=self.status, dumps=_dumps)


def invalid_request(message: str) -> ApiError:
    return ApiError(400, message, "invalid_request_error")


def server_error(message: str) -> ApiError:
    return ApiError(500, message, "server_error")


def unavailable(message: str) -> ApiError:
    return ApiError(503, message, "unavailable_error")


def exceed_context(n_prompt: int, n_ctx: int) -> ApiError:
    return ApiError(
        400,
        f"request ({n_prompt} tokens) exceeds the available context size ({n_ctx} tokens), "
        "try increasing it",
        "exceed_context_size_error",
        n_prompt_tokens=n_prompt,
        n_ctx=n_ctx,
    )


class _LoadError(Exception):
    """Model loading failed; the message is Polish and meant for the user."""


# --------------------------------------------------------------------------- #
# Request parameters
# --------------------------------------------------------------------------- #


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _get_number(body: dict[str, Any], key: str, default: float) -> float:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_request(f"Field '{key}': type must be number, but is {_type_name(value)}")
    if not math.isfinite(value):
        raise invalid_request(f"Field '{key}': must be a finite number")
    return float(value)


def _get_int(body: dict[str, Any], key: str, default: int) -> int:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_request(f"Field '{key}': type must be number, but is {_type_name(value)}")
    if isinstance(value, float):
        if not value.is_integer():
            raise invalid_request(f"Field '{key}': must be an integer")
        value = int(value)
    return int(value)


def _get_bool(body: dict[str, Any], key: str, default: bool) -> bool:
    value = body.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise invalid_request(f"Field '{key}': type must be boolean, but is {_type_name(value)}")
    return value


def _parse_logit_bias(value: Any) -> dict[int, float]:
    """llama-server accepts ``[[token, bias|false], ...]`` or ``{"token": bias}``."""
    if value is None:
        return {}
    items: list[tuple[Any, Any]]
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = []
        for entry in value:
            if not isinstance(entry, list) or len(entry) != 2:
                raise invalid_request("Field 'logit_bias': expected [[token, bias], ...]")
            items.append((entry[0], entry[1]))
    else:
        raise invalid_request(f"Field 'logit_bias': type must be array or object, but is {_type_name(value)}")
    out: dict[int, float] = {}
    for tok, bias in items:
        try:
            tok_id = int(tok)
        except (TypeError, ValueError):
            raise invalid_request(f"Field 'logit_bias': invalid token id {tok!r}") from None
        if bias is False:
            out[tok_id] = -math.inf
        elif isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise invalid_request(f"Field 'logit_bias': invalid bias for token {tok_id}")
        else:
            out[tok_id] = float(bias)
    return out


@dataclass
class CompletionParams:
    """Parsed ``POST /completion`` body (llama-server names and defaults)."""

    prompt: str | list[int]
    n_predict: int = -1
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    stream: bool = False
    cache_prompt: bool = True
    ignore_eos: bool = False
    logit_bias: dict[int, float] = field(default_factory=dict)

    def generation_settings(self, max_tokens: int | None = None) -> dict[str, Any]:
        """The ``generation_settings`` object of llama-server's final response."""
        return {
            "seed": LLAMA_DEFAULT_SEED if self.seed is None else self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": list(self.stop),
            "max_tokens": self.n_predict if max_tokens is None else max_tokens,
            "n_predict": self.n_predict,
            "n_keep": 0,
            "ignore_eos": self.ignore_eos,
            "stream": self.stream,
            "logit_bias": [[k, v if math.isfinite(v) else False] for k, v in self.logit_bias.items()],
            "n_probs": 0,
            "min_keep": 0,
            "samplers": ["penalties", "top_p", "min_p", "top_k", "temperature"],
        }


def parse_completion_params(body: Any) -> CompletionParams:
    """Validate a ``/completion`` body; raises ``ApiError`` (400) on bad input."""
    if not isinstance(body, dict):
        raise invalid_request("Request body must be a JSON object")
    if "prompt" not in body:
        raise invalid_request("'prompt' is required")
    prompt = body["prompt"]
    if isinstance(prompt, list):
        if prompt and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            prompt = [int(t) for t in prompt]
        elif len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        elif not prompt:
            prompt = ""
        else:
            raise invalid_request(
                "Only a single prompt (a string or an array of token ids) is supported"
            )
    elif not isinstance(prompt, str):
        raise invalid_request(f"Field 'prompt': type must be string, but is {_type_name(prompt)}")

    if body.get("n_predict") is not None:
        n_predict = _get_int(body, "n_predict", -1)
    else:
        n_predict = _get_int(body, "max_tokens", -1)

    stop_raw = body.get("stop")
    if stop_raw is None:
        stop: list[str] = []
    elif isinstance(stop_raw, str):
        stop = [stop_raw]
    elif isinstance(stop_raw, list) and all(isinstance(s, str) for s in stop_raw):
        stop = [s for s in stop_raw if s]
    else:
        raise invalid_request("Field 'stop': type must be an array of strings")

    seed_raw = body.get("seed")
    seed: int | None = None
    if seed_raw is not None:
        seed_val = _get_int(body, "seed", -1)
        if seed_val >= 0 and seed_val != LLAMA_DEFAULT_SEED:
            seed = seed_val & 0xFFFFFFFF

    return CompletionParams(
        prompt=prompt,
        n_predict=n_predict,
        temperature=_get_number(body, "temperature", 0.8),
        top_k=_get_int(body, "top_k", 40),
        top_p=_get_number(body, "top_p", 0.95),
        min_p=_get_number(body, "min_p", 0.05),
        repeat_penalty=_get_number(body, "repeat_penalty", 1.0),
        repeat_last_n=_get_int(body, "repeat_last_n", 64),
        presence_penalty=_get_number(body, "presence_penalty", 0.0),
        frequency_penalty=_get_number(body, "frequency_penalty", 0.0),
        stop=stop,
        seed=seed,
        stream=_get_bool(body, "stream", False),
        cache_prompt=_get_bool(body, "cache_prompt", True),
        ignore_eos=_get_bool(body, "ignore_eos", False),
        logit_bias=_parse_logit_bias(body.get("logit_bias")),
    )


def plan_generation(n_prompt: int, n_predict: int, n_ctx: int) -> tuple[int, bool]:
    """Token budget of a request: ``(max_tokens, capped_by_context)``.

    Mirrors llama-server: generation stops when ``n_prompt + n_generated``
    reaches the slot context; ``n_predict < 0`` means "until EOS / window end"
    and ``n_predict == 0`` still yields one token."""
    room = n_ctx - n_prompt
    if n_predict < 0:
        return room, True
    wanted = max(1, n_predict)
    if wanted > room:
        return room, True
    return wanted, False


def build_timings(
    cache_n: int, prompt_n: int, prompt_ms: float, predicted_n: int, predicted_ms: float
) -> dict[str, Any]:
    """llama-server's ``timings`` object."""
    return {
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": prompt_ms / prompt_n if prompt_n > 0 else 0.0,
        "prompt_per_second": 1e3 * prompt_n / prompt_ms if prompt_ms > 0 and prompt_n > 0 else 0.0,
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": predicted_ms / predicted_n if predicted_n > 0 else 0.0,
        "predicted_per_second": (
            1e3 * predicted_n / predicted_ms if predicted_ms > 0 and predicted_n > 0 else 0.0
        ),
    }


# --------------------------------------------------------------------------- #
# Text helpers: stop strings and incremental detokenisation
# --------------------------------------------------------------------------- #


class StopMatcher:
    """Text-level stop strings that work across token boundaries.

    ``feed(delta)`` returns ``(text_to_emit, matched_stop_or_None)``.  Text that
    could still turn into a stop string is held back; when a stop string is
    found, only the text *before* it is returned and the matcher is done.
    ``flush()`` returns the held-back tail once generation has ended."""

    def __init__(self, stops: Iterable[str]) -> None:
        self.stops: list[str] = [s for s in dict.fromkeys(stops) if s]
        self._buf = ""

    def feed(self, text: str) -> tuple[str, str | None]:
        if not self.stops:
            return text, None
        self._buf += text
        best_idx = -1
        best_stop: str | None = None
        for stop in self.stops:
            idx = self._buf.find(stop)
            if idx == -1:
                continue
            if best_stop is None or idx < best_idx or (idx == best_idx and len(stop) > len(best_stop)):
                best_idx, best_stop = idx, stop
        if best_stop is not None:
            out = self._buf[:best_idx]
            self._buf = ""
            return out, best_stop
        hold = self._partial_suffix_len()
        cut = len(self._buf) - hold
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out, None

    def _partial_suffix_len(self) -> int:
        """Length of the longest suffix of the buffer that is a proper prefix of a stop string."""
        buf = self._buf
        longest = 0
        for stop in self.stops:
            for k in range(min(len(stop) - 1, len(buf)), longest, -1):
                if buf.endswith(stop[:k]):
                    longest = k
                    break
        return longest

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out


class IncrementalDecoder:
    """Streaming detokeniser using the "prefix window" algorithm (vLLM / TGI).

    Decodes ``tokens[prefix:]`` and ``tokens[prefix:read]`` and emits the
    difference once it no longer ends in an incomplete UTF-8 sequence.  A few
    prompt tokens are used as left context so tokenizers that treat the first
    token of a sequence specially (SentencePiece's leading space) still produce
    the exact text of ``decode(prompt + generated)[len(decode(prompt)):]``."""

    CONTEXT_TOKENS = 6
    _COMPACT_AFTER = 256

    def __init__(self, decode: Callable[[list[int]], str], context: list[int] | None = None) -> None:
        self._decode = decode
        self._tokens: list[int] = list((context or [])[-self.CONTEXT_TOKENS :])
        self._prefix = 0
        self._read = len(self._tokens)

    def add(self, token: int) -> str:
        self._tokens.append(int(token))
        return self._advance(final=False)

    def flush(self) -> str:
        return self._advance(final=True)

    def _advance(self, final: bool) -> str:
        if self._read >= len(self._tokens):
            return ""
        prefix_text = self._decode(self._tokens[self._prefix : self._read]) if self._read > self._prefix else ""
        full_text = self._decode(self._tokens[self._prefix :])
        if len(full_text) > len(prefix_text) and (final or not full_text.endswith("�")):
            delta = full_text[len(prefix_text) :]
            self._prefix = self._read
            self._read = len(self._tokens)
            if self._prefix > self._COMPACT_AFTER:
                drop = self._prefix
                self._tokens = self._tokens[drop:]
                self._prefix -= drop
                self._read -= drop
            return delta
        if final:
            # nothing printable (e.g. a lone special token that decodes to "")
            self._prefix = self._read = len(self._tokens)
        return ""


# --------------------------------------------------------------------------- #
# Chat messages
# --------------------------------------------------------------------------- #


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Validate OpenAI-style messages and make them safe for HF chat templates:
    ``content`` becomes a string (text parts are joined), tool calls are copied
    so argument conversion never mutates the caller's data."""
    if messages is None:
        raise invalid_request("'messages' is required")
    if not isinstance(messages, list) or not messages:
        raise invalid_request("'messages' must be a non-empty array")
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(messages):
        if not isinstance(raw, dict) or not isinstance(raw.get("role"), str):
            raise invalid_request(f"messages[{i}]: 'role' is required")
        msg = dict(raw)
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text") or ""))
                else:
                    raise invalid_request(
                        "Only text content is supported by this model (no image or audio input)"
                    )
            msg["content"] = "".join(texts)
        elif isinstance(content, (dict, int, float, bool)):
            msg["content"] = _dumps(content)
        elif not isinstance(content, str):
            raise invalid_request(f"messages[{i}]: 'content' must be a string")
        calls = msg.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, list):
                raise invalid_request(f"messages[{i}]: 'tool_calls' must be an array")
            copied = []
            for j, call in enumerate(calls):
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise invalid_request(f"messages[{i}].tool_calls[{j}]: 'function' is required")
                fn = dict(call["function"])
                if not isinstance(fn.get("name"), str):
                    raise invalid_request(f"messages[{i}].tool_calls[{j}]: function 'name' is required")
                if fn.get("arguments") is None:
                    fn["arguments"] = {}
                new_call = dict(call)
                new_call.setdefault("type", "function")
                new_call["function"] = fn
                copied.append(new_call)
            if copied:
                msg["tool_calls"] = copied
            else:
                msg.pop("tool_calls")
        out.append(msg)
    return out


def convert_tool_arguments(messages: list[dict[str, Any]], *, as_object: bool) -> list[dict[str, Any]]:
    """Copy of ``messages`` with every tool call's ``arguments`` as a dict
    (``as_object=True``; unparsable strings stay strings) or as a JSON string."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        calls = msg.get("tool_calls")
        if not calls:
            out.append(msg)
            continue
        new_calls = []
        for call in calls:
            fn = dict(call["function"])
            args = fn.get("arguments")
            if as_object and isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except ValueError:
                    parsed = args
                fn["arguments"] = parsed
            elif not as_object and not isinstance(args, str):
                fn["arguments"] = _dumps(args)
            new_calls.append({**call, "function": fn})
        out.append({**msg, "tool_calls": new_calls})
    return out


def _messages_have_tool_calls(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("tool_calls") for m in messages)


# --------------------------------------------------------------------------- #
# Sampling (runs on the generation thread: these build MLX graphs)
# --------------------------------------------------------------------------- #


def _identity_processor(tokens: Any, logits: Any) -> Any:
    return logits


def make_sampler(params: CompletionParams, vocab_size: int) -> Callable[[Any], Any]:
    """Sampler with llama-server semantics built from mlx_lm's filters.

    ``temperature <= 0`` is greedy; ``top_k <= 0`` / ``top_p >= 1`` /
    ``min_p <= 0`` disable the filter.  With a ``seed`` the sampler owns its own
    PRNG key (split every step), so seeded requests are reproducible even while
    batched with other requests."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p, categorical_sampling

    temp = params.temperature
    if temp <= 0:
        return lambda logprobs: mx.argmax(logprobs, axis=-1)

    methods: list[Callable[[Any], Any]] = []
    top_p, min_p, top_k = params.top_p, min(params.min_p, 1.0), params.top_k
    if 0.0 < top_p < 1.0:
        methods.append(lambda x: apply_top_p(x, top_p))
    if min_p > 0.0:
        methods.append(lambda x: apply_min_p(x, min_p, 1))
    if 0 < top_k < vocab_size:
        methods.append(lambda x: apply_top_k(x, top_k))

    if params.seed is None:

        def sampler(logprobs: Any) -> Any:
            for method in methods:
                logprobs = method(logprobs)
            return categorical_sampling(logprobs, temp)

        return sampler

    state = [mx.random.key(params.seed)]
    inv_temp = 1.0 / temp

    def seeded_sampler(logprobs: Any) -> Any:
        for method in methods:
            logprobs = method(logprobs)
        keys = mx.random.split(state[0])
        state[0] = keys[0]
        return mx.random.categorical(logprobs * inv_temp, key=keys[1])

    return seeded_sampler


def make_logits_processors(params: CompletionParams, n_ctx: int) -> list[Callable[[Any, Any], Any]]:
    """Repetition / presence / frequency penalties and logit bias (llama-server
    semantics: ``repeat_penalty == 1`` disables, ``repeat_last_n == -1`` = whole context)."""
    from mlx_lm.sample_utils import make_logits_processors as _make

    window = params.repeat_last_n
    if window < 0:
        window = n_ctx
    rep = params.repeat_penalty if params.repeat_penalty > 0 and params.repeat_penalty != 1.0 else None
    presence = params.presence_penalty or None
    frequency = params.frequency_penalty or None
    if window == 0:
        rep = presence = frequency = None
    ctx_size = max(window, 1)
    return _make(
        logit_bias=params.logit_bias or None,
        repetition_penalty=rep,
        repetition_context_size=ctx_size,
        presence_penalty=presence,
        presence_context_size=ctx_size,
        frequency_penalty=frequency,
        frequency_context_size=ctx_size,
    )


def kv_bytes_per_token(cache: Iterable[Any]) -> float:
    """Approximate KV bytes one token costs, measured from a live prompt cache."""
    total = 0.0
    for layer in cache:
        sub = getattr(layer, "caches", None)
        if sub is not None:
            total += kv_bytes_per_token(sub)
            continue
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        ks = keys if isinstance(keys, (tuple, list)) else (keys,)
        vs = values if isinstance(values, (tuple, list)) else (values,)
        try:
            length = int(ks[0].shape[-2])
        except (AttributeError, IndexError, TypeError):
            continue
        if length <= 0:
            continue
        total += (sum(a.nbytes for a in ks) + sum(a.nbytes for a in vs)) / length
    return total


# --------------------------------------------------------------------------- #
# Per-request state shared between the HTTP handler and the generation thread
# --------------------------------------------------------------------------- #


class EventSink:
    """Thread-safe bridge from the generation thread into an ``asyncio.Queue``.

    Events: ``("token", content, token_id, n_predicted)``,
    ``("done", final_dict)``, ``("error", ApiError)``."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()

    def put(self, event: tuple[Any, ...]) -> None:
        try:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)
        except RuntimeError:  # loop already closed (shutdown)
            pass


@dataclass(eq=False)
class Sequence:
    """One completion request as seen by the scheduler."""

    id_task: int
    params: CompletionParams
    prompt_tokens: list[int]
    max_tokens: int
    ctx_capped: bool
    sink: EventSink
    # scheduler state (generation thread only, read racily by /slots)
    slot: int = -1
    cancelled: bool = False
    finished: bool = False
    cached_n: int = 0
    n_predicted: int = 0
    has_new_line: bool = False
    t_submit: float = field(default_factory=time.perf_counter)
    t_admit: float = 0.0
    t_first: float = 0.0
    t_last: float = 0.0
    decoder: IncrementalDecoder | None = None
    stops: StopMatcher | None = None
    backend_ref: Any = None
    backend_final: tuple[Any, Any] | None = None


# --------------------------------------------------------------------------- #
# Backends (generation thread only)
# --------------------------------------------------------------------------- #


class _PrefillAborted(Exception):
    """Raised from the prefill progress callback to stop a cancelled prompt."""


@dataclass
class _SeqState:
    gen: Any
    cache: list[Any]
    tokens: list[int]
    rest: list[int]
    prefilled: int = 0
    started: bool = False


class _BatchBackend:
    """Continuous batching with ``mlx_lm.generate.BatchGenerator``."""

    kind = "batch"

    def __init__(self, model: Any, capacity: int, prefill_step_size: int, eos_ids: Iterable[int]) -> None:
        from mlx_lm.generate import SequenceStateMachine

        self._model = model
        self.capacity = capacity
        self._prefill_step_size = prefill_step_size
        eos = sorted(set(eos_ids))
        self._eos_sm = (
            SequenceStateMachine({"normal": [((t,), None) for t in eos]}, initial="normal")
            if eos
            else SequenceStateMachine()
        )
        self._plain_sm = SequenceStateMachine()
        self._by_uid: dict[int, Sequence] = {}
        self._bg: Any = None
        self._make()

    def _make(self) -> None:
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator

        self._bg = BatchGenerator(
            self._model,
            completion_batch_size=self.capacity,
            prefill_batch_size=self.capacity,
            prefill_step_size=self._prefill_step_size,
            stream=mx.default_stream(mx.default_device()),
        )
        self._by_uid = {}

    def admit(self, seq: Sequence, cache: Any, rest: list[int], sampler: Any, processors: list[Any]) -> None:
        # Always pass a real sampler and a non-empty processor list: mlx_lm's
        # GenerationBatch.filter() skips filtering these lists when *all*
        # entries are falsy, which misaligns them once a request with its own
        # sampler joins the batch later.
        (uid,) = self._bg.insert_segments(
            segments=[[list(rest)]],
            max_tokens=[seq.max_tokens],
            caches=[cache],
            all_tokens=[list(seq.prompt_tokens[: seq.cached_n])],
            samplers=[sampler],
            logits_processors=[processors or [_identity_processor]],
            state_machines=[self._plain_sm if seq.params.ignore_eos else self._eos_sm],
        )
        seq.backend_ref = uid
        self._by_uid[uid] = seq

    def step(self) -> list[tuple[Sequence, int | None, str | None]]:
        _, responses = self._bg.next()
        out: list[tuple[Sequence, int | None, str | None]] = []
        for r in responses:
            seq = self._by_uid.get(r.uid)
            if seq is None:
                continue
            finish = None
            if r.finish_reason is not None:
                del self._by_uid[r.uid]
                seq.backend_final = (r.prompt_cache, list(r.all_tokens or []))
                finish = "eos" if r.finish_reason == "stop" else "length"
            out.append((seq, int(r.token), finish))
        return out

    def release(self, seq: Sequence) -> tuple[Any, list[int] | None]:
        """Take ``seq`` out of the batch; returns its KV cache and the tokens it holds."""
        if seq.backend_final is not None:
            final, seq.backend_final = seq.backend_final, None
            return final
        uid = seq.backend_ref
        seq.backend_ref = None
        if uid is None or uid not in self._by_uid:
            return None, None
        del self._by_uid[uid]
        try:
            caches = self._bg.remove([uid], return_prompt_caches=True)
        except Exception:
            log.warning("Could not extract the KV cache of a removed request", exc_info=True)
            self._bg.remove([uid])
            return None, None
        cache, tokens = caches.get(uid, (None, None))
        return cache, (list(tokens) if tokens is not None else None)

    def active_cache_bytes(self) -> int:
        try:
            return int(self._bg.prompt_cache_nbytes)
        except Exception:
            return 0

    def warmup(self, tokens: list[int]) -> tuple[int, float]:
        """Compile kernels with tiny requests (one alone, then a batched pair when
        there are several slots); returns ``(vocab_size, kv_bytes_per_token)``."""
        vocab, bpt = self._warm_batch([list(tokens)])
        if self.capacity > 1 and len(tokens) > 1:
            self._warm_batch([list(tokens), list(tokens[1:])])
        return vocab, bpt

    def _warm_batch(self, prompts: list[list[int]]) -> tuple[int, float]:
        import mlx.core as mx

        n = len(prompts)
        uids = self._bg.insert_segments(
            segments=[[p] for p in prompts],
            max_tokens=[2] * n,
            samplers=[lambda x: mx.argmax(x, axis=-1)] * n,
            logits_processors=[[_identity_processor]] * n,
            state_machines=[self._plain_sm] * n,
        )
        pending = set(uids)
        vocab, bpt = 0, 0.0
        for _ in range(10_000):
            _, responses = self._bg.next()
            for r in responses:
                if r.uid not in pending:
                    continue
                vocab = int(r.logprobs.shape[-1])
                if r.finish_reason is not None:
                    pending.discard(r.uid)
                    if r.uid == uids[0]:
                        bpt = kv_bytes_per_token(r.prompt_cache or [])
            if not pending:
                return vocab, bpt
        raise RuntimeError("warm-up generation did not finish")

    def reset(self) -> None:
        self.close()
        self._make()

    def close(self) -> None:
        if self._bg is not None:
            try:
                self._bg.close()
            except Exception:
                log.debug("BatchGenerator.close failed", exc_info=True)


class _SequentialBackend:
    """One request at a time with ``mlx_lm.generate.generate_step``.

    Supports quantised KV caches (``kv_bits``) and architectures whose caches
    cannot be merged into a batch."""

    kind = "sequential"
    capacity = 1

    def __init__(
        self,
        model: Any,
        prefill_step_size: int,
        kv_bits: int | None,
        kv_group_size: int,
        eos_ids: Iterable[int],
        poll: Callable[[], None],
        stopping: Callable[[], bool],
    ) -> None:
        self._model = model
        self._prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self._kv_group_size = kv_group_size
        self._eos = frozenset(eos_ids)
        self._poll = poll
        self._stopping = stopping
        self._current: Sequence | None = None

    def _generator(self, rest: list[int], cache: list[Any], max_tokens: int, sampler: Any,
                   processors: list[Any], progress: Callable[[int, int], None] | None) -> Any:
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        return generate_step(
            mx.array(rest),
            self._model,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors or None,
            prompt_cache=cache,
            prefill_step_size=self._prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self._kv_group_size,
            quantized_kv_start=0,
            prompt_progress_callback=progress,
        )

    def admit(self, seq: Sequence, cache: Any, rest: list[int], sampler: Any, processors: list[Any]) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        if cache is None:
            cache = make_prompt_cache(self._model)
        state = _SeqState(gen=None, cache=cache, tokens=list(seq.prompt_tokens[: seq.cached_n]), rest=list(rest))

        def progress(done: int, total: int) -> None:
            state.prefilled = done
            if done >= total:
                # The prompt is complete and the first token is already being
                # fed back; aborting here would desynchronise cache and tokens.
                return
            self._poll()
            if seq.cancelled or self._stopping():
                raise _PrefillAborted()

        state.gen = self._generator(rest, cache, seq.max_tokens, sampler, processors, progress)
        seq.backend_ref = state
        self._current = seq

    def step(self) -> list[tuple[Sequence, int | None, str | None]]:
        seq = self._current
        if seq is None or seq.backend_ref is None:
            return []
        state: _SeqState = seq.backend_ref
        try:
            token, _ = next(state.gen)
        except _PrefillAborted:
            return []  # the scheduler reaps the cancelled sequence
        except StopIteration:
            return [(seq, None, "length")]
        token = int(token)
        if not state.started:
            state.tokens.extend(state.rest)
            state.started = True
        # generate_step feeds a token to the model before yielding it, so the
        # cache always holds every yielded token.
        state.tokens.append(token)
        finish = None
        if token in self._eos and not seq.params.ignore_eos:
            finish = "eos"
        elif seq.n_predicted + 1 >= seq.max_tokens:
            finish = "length"
        return [(seq, token, finish)]

    def release(self, seq: Sequence) -> tuple[Any, list[int] | None]:
        state: _SeqState | None = seq.backend_ref
        seq.backend_ref = None
        if self._current is seq:
            self._current = None
        if state is None:
            return None, None
        if state.gen is not None:
            try:
                state.gen.close()
            except Exception:
                log.debug("closing generate_step failed", exc_info=True)
        if state.started:
            tokens = state.tokens
        else:
            tokens = state.tokens + state.rest[: state.prefilled]
        return state.cache, tokens

    def active_cache_bytes(self) -> int:
        seq = self._current
        if seq is None or seq.backend_ref is None:
            return 0
        try:
            return int(sum(c.nbytes for c in seq.backend_ref.cache))
        except Exception:
            return 0

    def warmup(self, tokens: list[int]) -> tuple[int, float]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        cache = make_prompt_cache(self._model)
        gen = self._generator(list(tokens), cache, 2, lambda x: mx.argmax(x, axis=-1), [], None)
        vocab = 0
        try:
            for _, logprobs in gen:
                vocab = int(logprobs.shape[-1])
        finally:
            gen.close()
        mx.eval([c.state for c in cache])
        return vocab, kv_bytes_per_token(cache)

    def reset(self) -> None:
        self._current = None

    def close(self) -> None:
        self._current = None


# --------------------------------------------------------------------------- #
# The model service: loading, tokenizer/template work and the scheduler
# --------------------------------------------------------------------------- #


@dataclass
class WorkerConfig:
    model: str
    host: str = "127.0.0.1"
    port: int = 8080
    parallel: int = 1
    ctx: int = 8192
    kv_bits: int | None = None
    kv_group_size: int = 64
    memory_limit_mb: int | None = None
    prefill_step_size: int = 2048
    prompt_cache_size: int = 10
    prompt_cache_mb: int | None = None
    trust_remote_code: bool = False
    parent_pid: int | None = None


class GenerationService(Protocol):
    """What the HTTP layer needs (``MlxService`` in production, fakes in tests)."""

    state: str
    n_ctx: int
    model_path: str

    def props(self) -> dict[str, Any]: ...
    def encode_prompt(self, prompt: str | list[int]) -> tuple[list[int], str]: ...
    def tokenize(self, content: str, add_special: bool, parse_special: bool) -> list[int]: ...
    def token_pieces(self, tokens: list[int]) -> list[dict[str, Any]]: ...
    def detokenize(self, tokens: list[int]) -> str: ...
    def apply_template(self, messages: Any, tools: Any, add_generation_prompt: bool,
                       template_kwargs: dict[str, Any]) -> str: ...
    def new_task_id(self) -> int: ...
    def submit(self, seq: Sequence) -> None: ...
    def cancel(self, seq: Sequence) -> None: ...
    def slots(self) -> list[dict[str, Any]]: ...
    def memory(self) -> dict[str, Any]: ...


def _describe_load_error(exc: BaseException, path: str) -> str:
    text = str(exc) or exc.__class__.__name__
    if isinstance(exc, _LoadError):
        return text
    if isinstance(exc, FileNotFoundError):
        return f"Brak plików modelu MLX w katalogu {path}: {text}"
    if isinstance(exc, MemoryError) or "memory" in text.lower():
        return f"Za mało pamięci, aby załadować model MLX ({path}): {text}"
    if isinstance(exc, ValueError) and ("not supported" in text or "Model type" in text):
        try:
            from mlx_lm import __version__ as ver
        except Exception:
            ver = "?"
        return f"Architektura tego modelu nie jest obsługiwana przez zainstalowane mlx-lm {ver}: {text}"
    return f"Nie udało się załadować modelu MLX ({path}): {text}"


class MlxService:
    """Owns the model.  ``start()`` launches the generation thread, which loads
    the model and then runs the scheduler loop; everything MLX happens there."""

    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self.n_ctx = cfg.ctx
        self.model_path = str(Path(cfg.model).expanduser().resolve())
        self.state = "loading"  # loading | ready | failed | stopped
        self.load_error: str | None = None
        self.on_fatal: Callable[[str], None] | None = None

        self._commands: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._thread_main, name="mlx-generation", daemon=True)
        self._stop_requested = False
        self._task_counter = 0
        self._task_lock = threading.Lock()

        # set during loading, read-only afterwards
        self._model: Any = None
        self.tokenizer: Any = None
        self._eos_ids: list[int] = []
        self._vocab_size = 0
        self._template_src = ""
        self._fallback_template: str | None = None
        self._caps: dict[str, bool] = {}
        self._props: dict[str, Any] = {}
        self._backend: _BatchBackend | _SequentialBackend | None = None
        self._lru: Any = None
        self._cache_budget = 0
        self._bytes_per_token = 0.0
        self._wired_limit: int | None = None

        # scheduler state (generation thread)
        self._pending: deque[Sequence] = deque()
        self._active: dict[int, Sequence] = {}
        self._free_slots: list[int] = list(range(cfg.parallel))
        self._slot_table: list[Sequence | None] = [None] * cfg.parallel
        self._cache_stats: tuple[int, int] = (0, 0)
        self._stop_flag = False
        self._was_busy = False

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_requested = True
        self._commands.put(("stop",))

    def join(self, timeout: float | None = None) -> bool:
        if not self._thread.is_alive():
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _thread_main(self) -> None:
        try:
            self._load()
        except BaseException as exc:  # noqa: BLE001 - report every load failure
            msg = _describe_load_error(exc, self.model_path)
            log.error("Błąd ładowania modelu: %s", msg)
            log.debug("load failure details", exc_info=True)
            self.load_error = msg
            self.state = "failed"
            if self.on_fatal is not None:
                self.on_fatal(msg)
            return
        if self._stop_requested:
            self.state = "stopped"
            self._close_backend()
            return
        self.state = "ready"
        log.info("Model gotowy: %s", self.model_path)
        try:
            self._scheduler_loop()
        except BaseException:  # noqa: BLE001
            log.exception("Wątek generowania zakończył się błędem")
            self.state = "failed"
            self._fail_everything(server_error("Generation thread crashed"))
            if self.on_fatal is not None:
                self.on_fatal("Wątek generowania zakończył się błędem.")
        finally:
            if self.state == "ready":
                self.state = "stopped"
            self._close_backend()

    # -- loading ------------------------------------------------------------ #

    def _apply_memory_limits(self) -> None:
        import mlx.core as mx

        max_rec = None
        try:
            if mx.metal.is_available():
                max_rec = int(mx.device_info().get("max_recommended_working_set_size", 0)) or None
        except Exception:
            max_rec = None
        if self.cfg.memory_limit_mb:
            limit = int(self.cfg.memory_limit_mb) * 1024 * 1024
            mx.set_memory_limit(limit)
            mx.set_cache_limit(max(limit // 4, 64 * 1024 * 1024))
            self._wired_limit = min(limit, max_rec) if max_rec else None
            log.info("Limit pamięci MLX: %d MB", self.cfg.memory_limit_mb)
        else:
            self._wired_limit = max_rec
        self._reapply_wired_limit()

    def _reapply_wired_limit(self) -> None:
        if not self._wired_limit:
            return
        import mlx.core as mx

        try:
            mx.set_wired_limit(self._wired_limit)
        except Exception as exc:
            log.warning("Nie udało się ustawić limitu pamięci przypiętej (wired): %s", exc)

    def _load(self) -> None:
        path = Path(self.model_path)
        if not (path / "config.json").is_file():
            raise _LoadError(f"Nie znaleziono modelu MLX w katalogu {path} (brak pliku config.json).")
        t0 = time.perf_counter()
        import mlx.core as mx
        from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
        from mlx_lm.utils import load as mlx_load

        mx.random.seed(int.from_bytes(os.urandom(4), "little"))
        self._apply_memory_limits()
        tokenizer_config: dict[str, Any] = {"trust_remote_code": True} if self.cfg.trust_remote_code else {}
        model, tokenizer = mlx_load(str(path), tokenizer_config=tokenizer_config)
        self._model = model
        self.tokenizer = tokenizer
        self._eos_ids = sorted(int(t) for t in tokenizer.eos_token_ids if t is not None)
        if self._stop_requested:
            return

        # tokenizer + template (first encode may normalise tokenizer state; do it
        # here, single-threaded, before other threads use the tokenizer)
        tokenizer.encode("warm up", add_special_tokens=False)
        self._setup_template()
        self._caps = self._probe_template_caps()

        # backend selection
        batchable = all(hasattr(c, "merge") for c in make_prompt_cache(model))
        kv_bits = self.cfg.kv_bits
        warm_tokens = tokenizer.encode("Hello, world!", add_special_tokens=False) or [0]
        if kv_bits is None and batchable:
            self._backend = _BatchBackend(model, self.cfg.parallel, self.cfg.prefill_step_size, self._eos_ids)
        else:
            if kv_bits is not None and self.cfg.parallel > 1:
                log.warning(
                    "Kwantyzacja KV (%d bit) nie działa z przetwarzaniem wsadowym mlx-lm: "
                    "%d sloty będą obsługiwane po kolei.", kv_bits, self.cfg.parallel,
                )
            elif not batchable and self.cfg.parallel > 1:
                log.warning(
                    "Architektura modelu nie obsługuje przetwarzania wsadowego: "
                    "%d sloty będą obsługiwane po kolei.", self.cfg.parallel,
                )
            self._backend = self._make_sequential(kv_bits)
        try:
            vocab, bpt = self._backend.warmup(warm_tokens)
        except Exception as exc:
            if kv_bits is None:
                raise
            log.warning("Kwantyzacja KV (%d bit) nie działa dla tego modelu (%s); używam pełnej precyzji.",
                        kv_bits, exc)
            self._backend = (
                _BatchBackend(model, self.cfg.parallel, self.cfg.prefill_step_size, self._eos_ids)
                if batchable
                else self._make_sequential(None)
            )
            vocab, bpt = self._backend.warmup(warm_tokens)
        self._reapply_wired_limit()  # BatchGenerator raises the wired limit to the device maximum
        self._vocab_size = vocab or int(getattr(getattr(model, "args", None), "vocab_size", 0) or 0)
        self._bytes_per_token = bpt

        if self.cfg.prompt_cache_mb is not None:
            self._cache_budget = max(0, int(self.cfg.prompt_cache_mb)) * 1024 * 1024
        elif bpt > 0:
            self._cache_budget = max(64 * 1024 * 1024, int(bpt * self.cfg.ctx * self.cfg.parallel))
        else:
            self._cache_budget = 1024 * 1024 * 1024
        self._lru = LRUPromptCache(max_size=max(1, self.cfg.prompt_cache_size), max_bytes=self._cache_budget)
        self._props = self._build_props()
        mx.clear_cache()
        log.info(
            "Załadowano %s w %.1f s (backend=%s, sloty=%d, ctx/slot=%d, kv_bits=%s, "
            "KV %.0f B/token, bufor promptów %.0f MB, pamięć %.0f MB)",
            path.name, time.perf_counter() - t0, self._backend.kind, self.cfg.parallel, self.cfg.ctx,
            getattr(self._backend, "kv_bits", None), bpt, self._cache_budget / 2**20,
            mx.get_active_memory() / 2**20,
        )

    def _make_sequential(self, kv_bits: int | None) -> _SequentialBackend:
        return _SequentialBackend(
            self._model,
            self.cfg.prefill_step_size,
            kv_bits,
            self.cfg.kv_group_size,
            self._eos_ids,
            poll=lambda: self._drain_commands(block=False),
            stopping=lambda: self._stop_flag,
        )

    def _setup_template(self) -> None:
        tok = self.tokenizer
        src = getattr(tok, "chat_template", None)
        if isinstance(src, dict):
            src = src.get("default") or next(iter(src.values()), None)
        has_template = bool(getattr(tok, "has_chat_template", False))
        if isinstance(src, str) and src:
            self._template_src = src
        elif not has_template:
            self._template_src = CHATML_TEMPLATE
            self._fallback_template = CHATML_TEMPLATE
            log.warning("Model nie ma szablonu czatu; używam ChatML.")

    def _probe_template_caps(self) -> dict[str, bool]:
        """Detect template capabilities by test rendering (like llama.cpp's minja caps)."""
        src = self._template_src

        def render(msgs: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   add_generation_prompt: bool = False) -> str:
            try:
                return self._render(msgs, tools, add_generation_prompt, False, {})
            except Exception:
                return ""

        user = {"role": "user", "content": "Hey"}
        args = {"code": "print('Hello, World!')"}
        tool_def = {
            "type": "function",
            "function": {
                "name": "some_tool",
                "description": "Some tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string", "description": "Some argument."}},
                    "required": ["arg"],
                },
            },
        }

        def call(arguments: Any, name: str = "ipython", cid: str = "call_1___") -> dict[str, Any]:
            return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}

        def with_calls(*calls: dict[str, Any]) -> list[dict[str, Any]]:
            return [user, {"role": "assistant", "content": "", "tool_calls": list(calls)},
                    {"role": "user", "content": "Thanks"}]

        def has_args(text: str) -> bool:
            return '"code": "print(' in text or '"code":"print(' in text

        out_obj = render(with_calls(call(args)), [tool_def])
        out_str = render(with_calls(call(_dumps(args))), [tool_def])
        out_par = render(with_calls(call(args, "tool_alpha", "c1"), call(args, "tool_beta", "c2")), [tool_def])
        out_sys = render([{"role": "system", "content": "SYS_MARKER_7"}, user])
        out_tools = render([user], [tool_def], True)
        out_tool_role = render(
            [user, {"role": "assistant", "content": "", "tool_calls": [call(args)]},
             {"role": "tool", "tool_call_id": "call_1___", "name": "ipython", "content": "TOOL_MARKER_7"}],
            [tool_def], True,
        )
        obj_ok, str_ok = has_args(out_obj), has_args(out_str)
        return {
            "supports_object_arguments": obj_ok,
            "supports_parallel_tool_calls": "tool_alpha" in out_par and "tool_beta" in out_par,
            "supports_preserve_reasoning": "reasoning_content" in src,
            "supports_reasoning_effort": "reasoning_effort" in src,
            "supports_string_content": True,
            "supports_system_role": "SYS_MARKER_7" in out_sys,
            "supports_tool_calls": obj_ok or str_ok,
            "supports_tools": "some_tool" in out_tools,
            "supports_typed_content": False,
            # extras (not in llama-server)
            "supports_string_arguments": str_ok,
            "supports_tool_role": "TOOL_MARKER_7" in out_tool_role,
        }

    def _build_props(self) -> dict[str, Any]:
        tok = self.tokenizer
        from mlx_lm import __version__ as mlx_lm_version
        import mlx.core as mx

        default_params = CompletionParams(prompt="").generation_settings()
        backend = self._backend
        return {
            "default_generation_settings": {"params": default_params, "n_ctx": self.cfg.ctx},
            "total_slots": self.cfg.parallel,
            "model_alias": self.model_path,
            "model_path": self.model_path,
            "modalities": {"vision": False, "video": False, "audio": False},
            "endpoint_slots": True,
            "endpoint_props": False,
            "endpoint_metrics": False,
            "chat_template": self._template_src,
            "chat_template_caps": dict(self._caps),
            "bos_token": getattr(tok, "bos_token", None) or "",
            "eos_token": getattr(tok, "eos_token", None) or "",
            "build_info": f"mlx_lm-{mlx_lm_version}/mlx-{mx.__version__}",
            "is_sleeping": False,
            # MLX extras
            "n_ctx": self.cfg.ctx,
            "engine": "mlx",
            "batching": backend is not None and backend.kind == "batch",
            "kv_bits": getattr(backend, "kv_bits", None),
            "vocab_size": self._vocab_size,
            "eos_token_ids": list(self._eos_ids),
        }

    # -- tokenizer / template (tokenizer executor thread) ------------------- #

    def _require_tokenizer(self) -> Any:
        if self.tokenizer is None or self.state != "ready":
            raise unavailable("Loading model")
        return self.tokenizer

    def props(self) -> dict[str, Any]:
        return self._props

    def _decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def encode_prompt(self, prompt: str | list[int]) -> tuple[list[int], str]:
        tok = self._require_tokenizer()
        if isinstance(prompt, list):
            vocab = self._vocab_size
            for t in prompt:
                if t < 0 or (vocab and t >= vocab):
                    raise invalid_request(f"Invalid token id {t} in prompt (vocabulary size {vocab})")
            return list(prompt), self._decode(list(prompt))
        if not prompt:
            return [], ""
        bos = getattr(tok, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        return list(tok.encode(prompt, add_special_tokens=add_special)), prompt

    def tokenize(self, content: str, add_special: bool, parse_special: bool) -> list[int]:
        tok = self._require_tokenizer()
        if not content:
            return []
        kwargs: dict[str, Any] = {"add_special_tokens": add_special}
        if not parse_special:
            kwargs["split_special_tokens"] = True
        return [int(t) for t in tok.encode(content, **kwargs)]

    def token_pieces(self, tokens: list[int]) -> list[dict[str, Any]]:
        self._require_tokenizer()
        return [{"id": int(t), "piece": self._decode([int(t)])} for t in tokens]

    def detokenize(self, tokens: list[int]) -> str:
        self._require_tokenizer()
        return self._decode([int(t) for t in tokens])

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                add_generation_prompt: bool, continue_final: bool, kwargs: dict[str, Any]) -> str:
        extra = dict(kwargs)
        if self._fallback_template is not None:
            extra["chat_template"] = self._fallback_template
        if continue_final:
            extra["continue_final_message"] = True
        out = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **extra,
        )
        if not isinstance(out, str):
            raise TypeError(f"chat template returned {type(out).__name__}, expected str")
        return out

    def apply_template(self, messages: Any, tools: Any, add_generation_prompt: bool,
                       template_kwargs: dict[str, Any]) -> str:
        self._require_tokenizer()
        msgs = normalize_messages(messages)
        if tools is not None and not isinstance(tools, list):
            raise invalid_request("Field 'tools': type must be array")
        kwargs = {k: v for k, v in (template_kwargs or {}).items() if k not in _RESERVED_TEMPLATE_KWARGS}
        continue_final = False
        if msgs[-1]["role"] == "assistant":
            last = msgs[-1]
            if last.get("tool_calls"):
                raise invalid_request("Cannot continue an assistant message that contains tool calls.")
            if str(last.get("content") or "").strip():
                continue_final = True
                add_generation_prompt = False
            else:
                msgs = msgs[:-1]
                add_generation_prompt = True
                if not msgs:
                    raise invalid_request("'messages' must contain at least one non-empty message")
        if _messages_have_tool_calls(msgs):
            prefer_obj = self._caps.get("supports_object_arguments", True) or not self._caps.get(
                "supports_string_arguments", False
            )
            variants = [
                convert_tool_arguments(msgs, as_object=prefer_obj),
                convert_tool_arguments(msgs, as_object=not prefer_obj),
            ]
        else:
            variants = [msgs]
        last_exc: Exception | None = None
        for variant in variants:
            try:
                return self._render(variant, tools, add_generation_prompt, continue_final, kwargs)
            except Exception as exc:  # template errors come in many types
                last_exc = exc
        raise invalid_request(f"Chat template error: {last_exc}")

    # -- scheduler interface (event loop thread) ---------------------------- #

    def new_task_id(self) -> int:
        with self._task_lock:
            self._task_counter += 1
            return self._task_counter

    def submit(self, seq: Sequence) -> None:
        self._commands.put(("submit", seq))

    def cancel(self, seq: Sequence) -> None:
        seq.cancelled = True
        self._commands.put(("cancel", seq))

    def slots(self) -> list[dict[str, Any]]:
        out = []
        for i, seq in enumerate(list(self._slot_table)):
            busy = seq is not None and not seq.finished
            params = seq.params.generation_settings(seq.max_tokens) if busy and seq else {}
            out.append(
                {
                    "id": i,
                    "n_ctx": self.cfg.ctx,
                    "speculative": False,
                    "is_processing": busy,
                    "id_task": seq.id_task if busy and seq else -1,
                    "n_prompt_tokens": len(seq.prompt_tokens) if busy and seq else 0,
                    "n_prompt_tokens_processed": (len(seq.prompt_tokens) - seq.cached_n) if busy and seq else 0,
                    "n_prompt_tokens_cache": seq.cached_n if busy and seq else 0,
                    "params": params,
                    "next_token": [
                        {
                            "has_next_token": busy,
                            "has_new_line": bool(seq.has_new_line) if busy and seq else False,
                            "n_remain": (seq.max_tokens - seq.n_predicted) if busy and seq else -1,
                            "n_decoded": seq.n_predicted if busy and seq else 0,
                        }
                    ],
                }
            )
        return out

    def memory(self) -> dict[str, Any]:
        import mlx.core as mx

        entries, nbytes = self._cache_stats
        return {
            "active": int(mx.get_active_memory()),
            "peak": int(mx.get_peak_memory()),
            "cache": int(mx.get_cache_memory()),
            "memory_limit": int(self.cfg.memory_limit_mb) * 1024 * 1024 if self.cfg.memory_limit_mb else None,
            "wired_limit": self._wired_limit,
            "prompt_cache": {"entries": entries, "bytes": nbytes, "budget": self._cache_budget},
            "kv_bytes_per_token": round(self._bytes_per_token, 1),
            "pending_requests": len(self._pending),
        }

    # -- scheduler (generation thread) -------------------------------------- #

    def _scheduler_loop(self) -> None:
        while True:
            busy = bool(self._active) or bool(self._pending)
            if self._was_busy and not busy:
                self._on_idle()
            self._was_busy = busy
            self._drain_commands(block=not busy)
            if self._stop_flag:
                break
            self._reap_cancelled()
            self._admit()
            if not self._active:
                continue
            try:
                outputs = self._backend.step()  # type: ignore[union-attr]
            except Exception as exc:
                log.exception("Błąd podczas generowania")
                self._recover(exc)
                continue
            for seq, token, finish in outputs:
                self._on_token(seq, token, finish)
            self._reap_cancelled()
        self._fail_everything(unavailable("Server is shutting down"))

    def _drain_commands(self, block: bool) -> None:
        wait = block
        while True:
            try:
                cmd = self._commands.get(timeout=0.5) if wait else self._commands.get_nowait()
            except queue.Empty:
                return
            wait = False
            kind = cmd[0]
            if kind == "submit":
                seq = cmd[1]
                if self._stop_flag or self._stop_requested:
                    seq.sink.put(("error", unavailable("Server is shutting down")))
                elif not seq.cancelled:
                    self._pending.append(seq)
            elif kind == "cancel":
                seq = cmd[1]
                seq.cancelled = True
                if seq in self._pending:
                    self._pending.remove(seq)
            elif kind == "stop":
                self._stop_flag = True

    def _on_idle(self) -> None:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _admit(self) -> None:
        backend = self._backend
        assert backend is not None
        while self._pending and len(self._active) < backend.capacity and self._free_slots:
            seq = self._pending.popleft()
            if seq.cancelled:
                continue
            try:
                self._start(seq)
            except Exception as exc:
                log.exception("Nie udało się rozpocząć zadania %d", seq.id_task)
                seq.sink.put(("error", server_error(f"Failed to start generation: {exc}")))
                self._retire(seq)

    def _lookup_cache(self, tokens: list[int]) -> tuple[Any, list[int]]:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        try:
            cache, rest = self._lru.fetch_nearest_cache(_PROMPT_CACHE_KEY, tokens)
        except Exception:
            log.warning("Bufor promptów: wyszukiwanie nie powiodło się", exc_info=True)
            return None, tokens
        if cache is not None and not rest:
            # exact hit: the model must still see at least one token to produce logits
            if can_trim_prompt_cache(cache) and trim_prompt_cache(cache, 1) == 1:
                rest = tokens[-1:]
            else:
                cache, rest = None, tokens
        return cache, list(rest)

    def _start(self, seq: Sequence) -> None:
        seq.slot = self._free_slots.pop(0)
        self._slot_table[seq.slot] = seq
        self._active[seq.id_task] = seq
        seq.t_admit = time.perf_counter()
        if seq.params.cache_prompt:
            cache, rest = self._lookup_cache(seq.prompt_tokens)
        else:
            cache, rest = None, list(seq.prompt_tokens)
        seq.cached_n = len(seq.prompt_tokens) - len(rest)
        seq.decoder = IncrementalDecoder(self._decode, seq.prompt_tokens)
        seq.stops = StopMatcher(seq.params.stop)
        sampler = make_sampler(seq.params, self._vocab_size)
        processors = make_logits_processors(seq.params, self.cfg.ctx)
        self._backend.admit(seq, cache, rest, sampler, processors)  # type: ignore[union-attr]
        self._trim_prompt_cache()

    def _on_token(self, seq: Sequence, token: int | None, finish: str | None) -> None:
        if seq.finished:
            return
        if token is None:
            self._complete(seq, "limit")
            return
        now = time.perf_counter()
        if seq.n_predicted == 0:
            seq.t_first = now
        seq.n_predicted += 1
        seq.t_last = now
        assert seq.decoder is not None and seq.stops is not None
        if finish == "eos":
            seq.sink.put(("token", "", token, seq.n_predicted))
            self._complete(seq, "eos")
            return
        delta = seq.decoder.add(token)
        emit, matched = seq.stops.feed(delta)
        if "\n" in emit:
            seq.has_new_line = True
        seq.sink.put(("token", emit, token, seq.n_predicted))
        if matched is not None:
            self._complete(seq, "word", matched)
        elif finish == "length" or seq.n_predicted >= seq.max_tokens:
            self._complete(seq, "limit")

    def _complete(self, seq: Sequence, stop_type: str, stopping_word: str = "") -> None:
        assert seq.decoder is not None and seq.stops is not None
        tail = ""
        if stop_type != "word":
            out, matched = seq.stops.feed(seq.decoder.flush())
            if matched is not None:
                stop_type, stopping_word, tail = "word", matched, out
            else:
                tail = out + seq.stops.flush()
        if "\n" in tail:
            seq.has_new_line = True
        cache, tokens = self._release(seq)
        self._store_cache(seq, cache, tokens)
        now = time.perf_counter()
        prompt_end = seq.t_first or now
        prompt_ms = max(0.0, (prompt_end - seq.t_admit) * 1e3)
        predicted_ms = max(0.0, (seq.t_last - seq.t_first) * 1e3) if seq.n_predicted > 0 else 0.0
        n_prompt = len(seq.prompt_tokens)
        final = {
            "id_slot": seq.slot,
            "tokens_predicted": seq.n_predicted,
            "tokens_evaluated": n_prompt,
            "has_new_line": seq.has_new_line,
            "truncated": stop_type == "limit" and seq.ctx_capped,
            "stop_type": stop_type,
            "stopping_word": stopping_word,
            "tokens_cached": len(tokens) if tokens else n_prompt + seq.n_predicted,
            "timings": build_timings(seq.cached_n, n_prompt - seq.cached_n, prompt_ms, seq.n_predicted,
                                     predicted_ms),
            "content_tail": tail,
        }
        seq.sink.put(("done", final))
        t = final["timings"]
        log.info(
            "zadanie %d slot %d: prompt %d tok (z bufora %d) %.0f ms, wygenerowano %d tok w %.0f ms "
            "(%.1f tok/s), stop=%s", seq.id_task, seq.slot, n_prompt, seq.cached_n, prompt_ms,
            seq.n_predicted, predicted_ms, t["predicted_per_second"], stop_type,
        )
        self._retire(seq)

    def _release(self, seq: Sequence) -> tuple[Any, list[int] | None]:
        try:
            return self._backend.release(seq)  # type: ignore[union-attr]
        except Exception:
            log.exception("Nie udało się zwolnić slotu zadania %d", seq.id_task)
            self._recover(RuntimeError("slot release failed"), exclude=seq)
            return None, None

    def _store_cache(self, seq: Sequence, cache: Any, tokens: list[int] | None) -> None:
        if cache is None or not tokens or len(tokens) <= seq.cached_n or self._lru is None:
            return
        try:
            self._lru.insert_cache(_PROMPT_CACHE_KEY, list(tokens), cache, cache_type="assistant")
        except Exception:
            log.warning("Bufor promptów: zapis nie powiódł się", exc_info=True)
        self._trim_prompt_cache()

    def _trim_prompt_cache(self) -> None:
        if self._lru is None:
            return
        try:
            active = self._backend.active_cache_bytes() if self._backend is not None else 0
            self._lru.trim_to(n_bytes=max(0, self._cache_budget - active))
            self._cache_stats = (len(self._lru), int(self._lru.nbytes))
        except Exception:
            log.warning("Bufor promptów: przycinanie nie powiodło się", exc_info=True)

    def _retire(self, seq: Sequence) -> None:
        seq.finished = True
        self._active.pop(seq.id_task, None)
        if 0 <= seq.slot < len(self._slot_table):
            if self._slot_table[seq.slot] is seq:
                self._slot_table[seq.slot] = None
            if seq.slot not in self._free_slots:
                self._free_slots.append(seq.slot)
                self._free_slots.sort()

    def _reap_cancelled(self) -> None:
        for seq in [s for s in self._active.values() if s.cancelled and not s.finished]:
            cache, tokens = self._release(seq)
            self._store_cache(seq, cache, tokens)
            log.info("zadanie %d anulowane (slot %d, %d tok wygenerowanych)", seq.id_task, seq.slot,
                     seq.n_predicted)
            self._retire(seq)

    def _recover(self, exc: BaseException, exclude: Sequence | None = None) -> None:
        """A generation step failed: fail the active requests and rebuild the backend."""
        err = server_error(f"Generation failed: {exc}")
        for seq in list(self._active.values()):
            if seq is exclude:
                continue
            seq.backend_ref = None
            seq.backend_final = None
            if not seq.cancelled:
                seq.sink.put(("error", err))
            self._retire(seq)
        try:
            self._backend.reset()  # type: ignore[union-attr]
            self._reapply_wired_limit()
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            log.exception("Nie udało się odtworzyć backendu generowania")
            raise

    def _fail_everything(self, err: ApiError) -> None:
        for seq in list(self._active.values()):
            try:
                if self._backend is not None:
                    self._backend.release(seq)
            except Exception:
                pass
            if not seq.cancelled:
                seq.sink.put(("error", err))
            self._retire(seq)
        while self._pending:
            seq = self._pending.popleft()
            if not seq.cancelled:
                seq.sink.put(("error", err))
        # requests submitted after the loop stopped
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                break
            if cmd[0] == "submit" and not cmd[1].cancelled:
                cmd[1].sink.put(("error", err))

    def _close_backend(self) -> None:
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


def _sse(payload: dict[str, Any]) -> bytes:
    return b"data: " + _dumps(payload).encode("utf-8") + b"\n\n"


class WorkerHttp:
    """aiohttp handlers speaking llama-server's protocol on top of a ``GenerationService``."""

    def __init__(self, service: GenerationService) -> None:
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-tokenizer")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[self._error_middleware])
        r = app.router
        r.add_get("/health", self.health)
        r.add_get("/v1/health", self.health)
        r.add_get("/props", self.props)
        r.add_post("/apply-template", self.apply_template)
        r.add_post("/tokenize", self.tokenize)
        r.add_post("/detokenize", self.detokenize)
        r.add_post("/completion", self.completion)
        r.add_post("/completions", self.completion)
        r.add_get("/slots", self.slots)
        r.add_get("/memory", self.memory)
        for path in ("/embedding", "/embeddings", "/v1/embeddings"):
            r.add_post(path, self.embeddings)
        return app

    @web.middleware
    async def _error_middleware(self, request: web.Request, handler: Callable[..., Any]) -> web.StreamResponse:
        try:
            return await handler(request)
        except ApiError as err:
            return err.response()
        except web.HTTPNotFound:
            return ApiError(404, "File Not Found", "not_found_error").response()
        except web.HTTPMethodNotAllowed:
            return ApiError(405, "Method Not Allowed", "invalid_request_error").response()
        except web.HTTPRequestEntityTooLarge:
            return ApiError(413, "Request body is too large", "invalid_request_error").response()
        except web.HTTPException:
            raise
        except (asyncio.CancelledError, ConnectionResetError):
            raise
        except Exception as exc:
            log.exception("Nieobsłużony błąd w %s %s", request.method, request.path)
            return server_error(f"Internal error: {exc}").response()

    # -- helpers ------------------------------------------------------------ #

    def _require_ready(self) -> None:
        state = self.service.state
        if state == "ready":
            return
        if state == "loading":
            raise unavailable("Loading model")
        raise unavailable("Model is not available")

    async def _json(self, request: web.Request) -> dict[str, Any]:
        raw = await request.read()
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise invalid_request(f"Invalid JSON body: {exc}") from None
        if not isinstance(body, dict):
            raise invalid_request("Request body must be a JSON object")
        return body

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    # -- endpoints ---------------------------------------------------------- #

    async def health(self, request: web.Request) -> web.Response:
        self._require_ready()
        return web.json_response({"status": "ok"}, dumps=_dumps)

    async def props(self, request: web.Request) -> web.Response:
        self._require_ready()
        return web.json_response(self.service.props(), dumps=_dumps)

    async def apply_template(self, request: web.Request) -> web.Response:
        self._require_ready()
        body = await self._json(request)
        if "messages" not in body:
            raise invalid_request("'messages' is required")
        kwargs = body.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, dict):
            raise invalid_request("Field 'chat_template_kwargs': type must be object")
        prompt = await self._run(
            self.service.apply_template,
            body["messages"],
            body.get("tools"),
            _get_bool(body, "add_generation_prompt", True),
            kwargs,
        )
        return web.json_response({"prompt": prompt}, dumps=_dumps)

    async def tokenize(self, request: web.Request) -> web.Response:
        self._require_ready()
        body = await self._json(request)
        content = body.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise invalid_request(f"Field 'content': type must be string, but is {_type_name(content)}")
        tokens = await self._run(
            self.service.tokenize,
            content,
            _get_bool(body, "add_special", False),
            _get_bool(body, "parse_special", True),
        )
        if _get_bool(body, "with_pieces", False):
            pieces = await self._run(self.service.token_pieces, tokens)
            return web.json_response({"tokens": pieces}, dumps=_dumps)
        return web.json_response({"tokens": tokens}, dumps=_dumps)

    async def detokenize(self, request: web.Request) -> web.Response:
        self._require_ready()
        body = await self._json(request)
        tokens = body.get("tokens", [])
        if not isinstance(tokens, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens):
            raise invalid_request("Field 'tokens': type must be an array of integers")
        content = await self._run(self.service.detokenize, tokens)
        return web.json_response({"content": content}, dumps=_dumps)

    async def slots(self, request: web.Request) -> web.Response:
        self._require_ready()
        return web.json_response(self.service.slots(), dumps=_dumps)

    async def memory(self, request: web.Request) -> web.Response:
        self._require_ready()
        return web.json_response(self.service.memory(), dumps=_dumps)

    async def embeddings(self, request: web.Request) -> web.Response:
        raise ApiError(
            501,
            "This server does not support embeddings (the MLX worker serves text generation only)",
            "not_supported_error",
        )

    def _final_payload(self, final: dict[str, Any], content: str, params: CompletionParams,
                       prompt_text: str, max_tokens: int) -> dict[str, Any]:
        return {
            "index": 0,
            "content": content,
            "tokens": [],
            "id_slot": final["id_slot"],
            "stop": True,
            "model": self.service.model_path,
            "tokens_predicted": final["tokens_predicted"],
            "tokens_evaluated": final["tokens_evaluated"],
            "generation_settings": params.generation_settings(max_tokens),
            "prompt": prompt_text,
            "has_new_line": final["has_new_line"],
            "truncated": final["truncated"],
            "stop_type": final["stop_type"],
            "stopping_word": final["stopping_word"],
            "tokens_cached": final["tokens_cached"],
            "timings": final["timings"],
        }

    async def completion(self, request: web.Request) -> web.StreamResponse:
        self._require_ready()
        params = parse_completion_params(await self._json(request))
        tokens, prompt_text = await self._run(self.service.encode_prompt, params.prompt)
        n_ctx = self.service.n_ctx
        n_prompt = len(tokens)
        if n_prompt == 0:
            final = {
                "id_slot": -1, "tokens_predicted": 0, "tokens_evaluated": 0, "has_new_line": False,
                "truncated": False, "stop_type": "none", "stopping_word": "", "tokens_cached": 0,
                "timings": build_timings(0, 0, 0.0, 0, 0.0),
            }
            payload = self._final_payload(final, "", params, prompt_text, params.n_predict)
            if params.stream:
                resp = web.StreamResponse(headers=self._sse_headers())
                await resp.prepare(request)
                await resp.write(_sse(payload))
                await resp.write_eof()
                return resp
            return web.json_response(payload, dumps=_dumps)
        if n_prompt >= n_ctx:
            raise exceed_context(n_prompt, n_ctx)
        max_tokens, ctx_capped = plan_generation(n_prompt, params.n_predict, n_ctx)
        sink = EventSink(asyncio.get_running_loop())
        seq = Sequence(
            id_task=self.service.new_task_id(),
            params=params,
            prompt_tokens=tokens,
            max_tokens=max_tokens,
            ctx_capped=ctx_capped,
            sink=sink,
        )
        self.service.submit(seq)
        completed = False
        try:
            if params.stream:
                resp = await self._stream(request, seq, params, prompt_text, n_prompt, max_tokens)
            else:
                resp = await self._collect(seq, params, prompt_text, max_tokens)
            completed = True
            return resp
        finally:
            if not completed and not seq.finished:
                self.service.cancel(seq)

    @staticmethod
    def _sse_headers() -> dict[str, str]:
        return {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    async def _collect(self, seq: Sequence, params: CompletionParams, prompt_text: str,
                       max_tokens: int) -> web.Response:
        parts: list[str] = []
        while True:
            event = await seq.sink.queue.get()
            kind = event[0]
            if kind == "token":
                parts.append(event[1])
            elif kind == "done":
                final = event[1]
                parts.append(final["content_tail"])
                payload = self._final_payload(final, "".join(parts), params, prompt_text, max_tokens)
                return web.json_response(payload, dumps=_dumps)
            elif kind == "error":
                return event[1].response()

    async def _stream(self, request: web.Request, seq: Sequence, params: CompletionParams,
                      prompt_text: str, n_prompt: int, max_tokens: int) -> web.StreamResponse:
        event = await seq.sink.queue.get()
        if event[0] == "error":  # failed before the first token: plain HTTP error
            return event[1].response()
        resp = web.StreamResponse(headers=self._sse_headers())
        await resp.prepare(request)
        done = False
        try:
            while not done:
                chunks: list[bytes] = []
                while True:
                    kind = event[0]
                    if kind == "token":
                        chunks.append(
                            _sse(
                                {
                                    "index": 0,
                                    "content": event[1],
                                    "tokens": [event[2]],
                                    "stop": False,
                                    "id_slot": -1,
                                    "tokens_predicted": event[3],
                                    "tokens_evaluated": n_prompt,
                                }
                            )
                        )
                    elif kind == "done":
                        final = event[1]
                        chunks.append(
                            _sse(self._final_payload(final, final["content_tail"], params, prompt_text, max_tokens))
                        )
                        done = True
                        break
                    elif kind == "error":
                        chunks.append(_sse(event[1].to_json()))
                        done = True
                        break
                    try:
                        event = seq.sink.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await resp.write(b"".join(chunks))
                if not done:
                    event = await seq.sink.queue.get()
            await resp.write_eof()
        except ConnectionResetError:
            log.info("zadanie %d: klient rozłączył się", seq.id_task)
            if not seq.finished:
                self.service.cancel(seq)
        return resp


# --------------------------------------------------------------------------- #
# Process entry point
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m localcode.runtime.mlx_worker",
        description="LocalCode MLX worker: llama-server compatible HTTP API on top of mlx-lm.",
    )
    p.add_argument("--model", required=True, help="directory of the MLX model (config.json + *.safetensors)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--parallel", type=int, default=1, help="concurrent request slots (batch size)")
    p.add_argument("--ctx", type=int, default=8192, help="context window PER SLOT in tokens")
    p.add_argument("--kv-bits", type=int, choices=(4, 8), default=None,
                   help="quantise the KV cache (served sequentially: mlx-lm cannot batch quantised caches)")
    p.add_argument("--kv-group-size", type=int, default=64)
    p.add_argument("--memory-limit-mb", type=int, default=None,
                   help="MLX memory limit (mx.set_memory_limit); also caps the wired limit and buffer cache")
    p.add_argument("--prefill-step-size", type=int, default=2048, help="prompt tokens per prefill chunk")
    p.add_argument("--prompt-cache-size", type=int, default=10, help="max KV snapshots in the prefix cache")
    p.add_argument("--prompt-cache-mb", type=int, default=None,
                   help="byte budget of the prefix cache (default: ctx*parallel tokens of KV)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--parent-pid", type=int, default=None, help="exit when this process disappears")
    p.add_argument("--log-level", default=os.environ.get("LOCALCODE_LOG_LEVEL", "INFO"))
    return p


def _config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        model=args.model,
        host=args.host,
        port=args.port,
        parallel=args.parallel,
        ctx=args.ctx,
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        memory_limit_mb=args.memory_limit_mb,
        prefill_step_size=args.prefill_step_size,
        prompt_cache_size=args.prompt_cache_size,
        prompt_cache_mb=args.prompt_cache_mb,
        trust_remote_code=args.trust_remote_code,
        parent_pid=args.parent_pid,
    )


def _validate_config(cfg: WorkerConfig) -> str | None:
    """Polish error message for an unusable configuration, or None."""
    if not Path(cfg.model).expanduser().is_dir():
        return f"Katalog modelu nie istnieje: {cfg.model}"
    if cfg.parallel < 1:
        return "Liczba slotów (--parallel) musi być co najmniej 1."
    if cfg.ctx < 16:
        return "Kontekst (--ctx) musi mieć co najmniej 16 tokenów."
    if not 0 <= cfg.port <= 65535:
        return f"Nieprawidłowy port: {cfg.port}"
    if cfg.prefill_step_size < 1:
        return "--prefill-step-size musi być dodatni."
    if cfg.memory_limit_mb is not None and cfg.memory_limit_mb < 256:
        return "Limit pamięci (--memory-limit-mb) musi wynosić co najmniej 256 MB."
    return None


async def _watch_parent(pid: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        await asyncio.sleep(2.0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log.warning("Proces nadrzędny %d zakończył się; zamykam worker.", pid)
            stop.set()
        except PermissionError:
            pass


async def serve(cfg: WorkerConfig) -> int:
    """Run the worker until SIGTERM/SIGINT (exit code 0) or a load failure (1)."""
    loop = asyncio.get_running_loop()
    service = MlxService(cfg)
    http = WorkerHttp(service)
    runner = web.AppRunner(http.make_app(), handler_cancellation=True, access_log=None, shutdown_timeout=5.0)
    await runner.setup()
    site = web.TCPSite(runner, cfg.host, cfg.port)
    try:
        await site.start()
    except OSError as exc:
        log.error("Nie można nasłuchiwać na %s:%d: %s", cfg.host, cfg.port, exc)
        await runner.cleanup()
        http.close()
        return 1
    log.info("Nasłuchuję na http://%s:%d (ładowanie modelu %s)", cfg.host, cfg.port, service.model_path)

    stop = asyncio.Event()
    exit_code = 0

    def on_signal(name: str) -> None:
        log.info("Otrzymano %s, zamykam worker.", name)
        stop.set()

    def on_fatal(message: str) -> None:
        nonlocal exit_code
        exit_code = 1
        try:
            loop.call_soon_threadsafe(stop.set)
        except RuntimeError:
            pass

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, on_signal, sig.name)
        except (NotImplementedError, RuntimeError):
            pass
    service.on_fatal = on_fatal
    service.start()
    watchdog = asyncio.create_task(_watch_parent(cfg.parent_pid, stop)) if cfg.parent_pid else None

    await stop.wait()

    service.request_stop()
    if watchdog is not None:
        watchdog.cancel()
    try:
        await asyncio.wait_for(runner.cleanup(), timeout=10.0)
    except (asyncio.TimeoutError, TimeoutError):
        log.warning("Zamykanie serwera HTTP trwało zbyt długo.")
    joined = await loop.run_in_executor(None, service.join, 15.0)
    http.close()
    if not joined:
        # still inside mlx_lm.load (cannot be interrupted): leave without
        # tearing down MLX under a running thread
        log.warning("Wątek generowania nie zakończył się w czasie; wymuszam wyjście.")
        logging.shutdown()
        os._exit(exit_code)
    log.info("Worker zakończony.")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s mlx_worker: %(message)s",
    )
    cfg = _config_from_args(args)
    problem = _validate_config(cfg)
    if problem:
        log.error("Błąd konfiguracji: %s", problem)
        return 2
    try:
        return asyncio.run(serve(cfg))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
