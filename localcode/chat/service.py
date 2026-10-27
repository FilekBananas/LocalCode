"""Chat turn orchestration (port of ai_ux ``src/app/api/chat/route.ts``).

:meth:`ChatService.send` takes a ``ChatSendRequest`` dict and is an async
generator of SSE frames, exactly the frames the SwiftUI client decodes::

    start → [status] → [citations] → [compaction] → context
          → (reasoning | text)* → usage → done → [title]
    start → … → error                                    (failure after start)

* **Validation happens before the first frame.** Bad requests, unknown
  conversations/messages/folders raise :class:`ApiError` from the first
  ``__anext__()``; the HTTP layer should pull the first frame *before* it
  commits the SSE response and turn an ``ApiError`` into a JSON error.
* The user message (new, edited or re-used for regenerate) and a
  ``streaming`` assistant placeholder are persisted before ``start``. The
  placeholder always ends as ``complete``, ``error`` or ``stopped``: client
  disconnects arrive as task cancellation or ``aclose()`` on this generator,
  and both persist the partial answer as ``stopped``.
* Model choice: ``req.model`` > ``conversation.model`` > ``settings.defaults.model``
  > ``config.chat.default_model`` > first chat-capable local model. The model
  is JIT-loaded through ``runtime.acquire(key, purpose="chat")``; a ``status``
  frame ("Ładowanie modelu…") is sent when that takes noticeable time.
* **Compaction.** When the assembled prompt passes the automatic threshold
  (``ChatSettings.auto_compact_threshold``, falling back to the agent's), the
  oldest turns are summarised by the same model
  (:mod:`localcode.chat.compaction`), the summary is stored on the conversation
  and reused by every later turn, and a
  ``{"type": "compaction", "before", "after", "strategy": "summarize"}`` frame is
  sent before the answer. A failed summarisation is logged and the turn falls
  back to the plain trimming of :mod:`localcode.chat.context`. Every turn also
  reports ``{"type": "context", "used", "limit", "system", "tools", "history",
  "reserve"}`` — the field names of :class:`localcode.events.ContextUpdate`.
* The prompt is built by :mod:`localcode.chat.context` within the engine's real
  ``n_ctx``, rendered with ``ChatFormatter(engine).render(..., thinking=showReasoning)``
  and streamed with ``engine.complete``. Reasoning is split from the answer by
  ``runtime.toolcalls.StreamSplitter`` (fallback: a ``<think>`` splitter) and
  never re-sent to the model. A ``ContextOverflowError`` from the engine
  triggers one retry with a tighter budget.
* After ``done``: an automatic title for the first exchange (≤ 20 tokens,
  thinking off, 8 s budget, heuristic fallback) and — when memory is enabled —
  a cheap, background extraction of explicit "remember that …" requests into
  memory entries (the reference has no automatic memory extraction; this
  keeps LocalCode's version deterministic and free of extra model calls).

``costUsd`` is always 0 (local models).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

from localcode.chat.compaction import (
    DEFAULT_THRESHOLD,
    STRATEGY_SUMMARIZE,
    ChatCompactor,
    CompactionResult,
)
from localcode.chat.context import (
    TITLE_SYSTEM,
    ChatContext,
    ContextBuilder,
    ContextReport,
    TokenCounter,
    build_title_prompt,
    compose_text,
    estimate_tokens,
    resolve_scope,
)
from localcode.chat.rag import LeaseHandle, RagIndex, RuntimeEmbedder, open_lease
from localcode.chat.repo import (
    LOCAL_PROVIDER,
    ApiError,
    ChatRepo,
    collapse_ws,
    derive_title,
    truncate,
)
from localcode.config import favorites_first
from localcode.remote.providers import PROVIDER_LABELS, name_with_provider, parse_key
from localcode.runtime.base import EngineError, SamplingParams
from localcode.runtime.templates import model_effort_levels, normalize_effort, resolve_effort

log = logging.getLogger(__name__)

__all__ = [
    "ApiError",
    "ChatService",
    "ThinkSplitter",
    "BasicFormatter",
    "model_info",
    "validate_chat_request",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

FLUSH_INTERVAL_S = 0.7          #: partial answer persisted this often while streaming
TITLE_TIMEOUT_S = 8.0
TITLE_MAX_TOKENS = 20
STATUS_DELAY_S = 0.25           #: slow operations announce themselves after this delay
COMPACT_TIMEOUT_S = 180.0       #: a summary that takes longer than this is abandoned
GENERIC_ERROR = "Nieoczekiwany błąd podczas generowania odpowiedzi."
COMPACT_STATUS = "Kompaktuję rozmowę…"
COMPACT_FAILED_STATUS = ("Nie udało się streścić rozmowy — najstarsze wiadomości "
                         "zostaną pominięte w tej odpowiedzi.")
COMPACT_FAILED = ("Nie udało się skompaktować rozmowy — model jest zajęty albo zwrócił błąd. "
                  "Spróbuj ponownie za chwilę.")


def _context_overflow_error() -> type[BaseException]:
    """``runtime.http_engine.ContextOverflowError`` if that module exists."""
    try:
        from localcode.runtime.http_engine import ContextOverflowError  # type: ignore[attr-defined]
    except Exception:
        class _NeverRaised(Exception):
            pass
        return _NeverRaised
    return ContextOverflowError


# --------------------------------------------------------------------------- #
# Request validation (reference zod schema)
# --------------------------------------------------------------------------- #


def _opt_str(body: dict[str, Any], key: str, issues: list[dict[str, str]], *, max_len: int) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        issues.append({"path": key, "message": "musi być tekstem"})
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > max_len:
        issues.append({"path": key, "message": f"może mieć najwyżej {max_len} znaków"})
        return None
    return value


def validate_chat_request(body: Any) -> dict[str, Any]:
    """Normalise a ``ChatSendRequest``; raises ``ApiError(400)`` on bad input."""
    if not isinstance(body, dict):
        raise ApiError(400, "Nieprawidłowe dane żądania czatu.", "invalid_request")
    issues: list[dict[str, str]] = []
    out: dict[str, Any] = {
        "conversationId": _opt_str(body, "conversationId", issues, max_len=200),
        "folderId": _opt_str(body, "folderId", issues, max_len=200),
        "provider": _opt_str(body, "provider", issues, max_len=64),
        "model": _opt_str(body, "model", issues, max_len=500),
        "editMessageId": _opt_str(body, "editMessageId", issues, max_len=200),
    }
    content = body.get("content")
    if content is None:
        out["content"] = ""
    elif not isinstance(content, str):
        issues.append({"path": "content", "message": "musi być tekstem"})
        out["content"] = ""
    elif len(content) > 500_000:
        issues.append({"path": "content", "message": "wiadomość jest za długa (maks. 500 000 znaków)"})
        out["content"] = ""
    else:
        out["content"] = content

    ids = body.get("attachmentIds")
    attachment_ids: list[str] = []
    if ids is not None:
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            issues.append({"path": "attachmentIds", "message": "musi być listą identyfikatorów"})
        elif len(ids) > 50:
            issues.append({"path": "attachmentIds", "message": "najwyżej 50 załączników"})
        else:
            for i in ids:
                i = i.strip()
                if i and i not in attachment_ids:
                    attachment_ids.append(i)
    out["attachmentIds"] = attachment_ids

    temperature = body.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            issues.append({"path": "temperature", "message": "musi być liczbą z zakresu 0–2"})
            temperature = None
    out["temperature"] = float(temperature) if temperature is not None else None

    max_tokens = body.get("maxTokens")
    if max_tokens is not None:
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, float))
                or int(max_tokens) != max_tokens or not 0 <= max_tokens <= 1_000_000):
            issues.append({"path": "maxTokens", "message": "musi być liczbą całkowitą z zakresu 0–1 000 000 (0 = bez limitu)"})
            max_tokens = None
    out["maxTokens"] = int(max_tokens) if max_tokens is not None else None

    effort = body.get("reasoningEffort")
    if effort is not None and normalize_effort(effort) is None:
        issues.append({"path": "reasoningEffort", "message": "musi być jednym z: off, low, medium, high"})
        effort = None
    out["reasoningEffort"] = normalize_effort(effort)

    regenerate = body.get("regenerate")
    if regenerate is not None and not isinstance(regenerate, bool):
        issues.append({"path": "regenerate", "message": "musi być wartością logiczną"})
        regenerate = None
    out["regenerate"] = bool(regenerate)

    if issues:
        raise ApiError(400, "Nieprawidłowe dane żądania czatu.", "invalid_request", details=issues)
    return out


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #


def _format_size(size: int) -> str:
    if not size:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.0f} {units[idx]}" if idx == 0 else f"{value:.1f} {units[idx]}".replace(".", ",")


def is_chat_model(model: Any) -> bool:
    """True for a model the chatbot can talk to.

    Embedding models and diffusion (image) models cannot generate a reply, so
    they must never be offered — nor picked as the fallback default.
    """
    caps = set(getattr(model, "capabilities", None) or [])
    if caps & {"embedding", "image"}:
        return False
    try:
        return not bool(model.is_embedding) and not bool(model.is_image)
    except Exception:
        return True


def model_info(model: Any) -> dict[str, Any]:
    """``ModelInfo`` DTO (Models.swift) for a :class:`LocalModel`."""
    caps = list(getattr(model, "capabilities", None) or [])
    meta = getattr(model, "meta", None)
    remote = parse_key(getattr(model, "key", "") or "")
    if remote is not None:
        return _remote_model_info(model, remote[0], caps, meta)
    parts = [str(getattr(model, "format", "") or "").upper() or None, getattr(model, "quant", None),
             _format_size(int(getattr(model, "size_bytes", 0) or 0))]
    params_b = getattr(meta, "params_b", None) if meta else None
    if params_b:
        parts.insert(0, f"{params_b:g}B")
    return {
        "id": model.key,
        "provider": LOCAL_PROVIDER,
        "label": getattr(model, "display_name", None) or model.key,
        "contextWindow": getattr(meta, "context_length", None) if meta else None,
        "maxOutput": None,
        "supportsVision": getattr(model, "mmproj_path", None) is not None or "vision" in caps,
        "supportsReasoning": "thinking" in caps,
        # Empty list = the model has no reasoning-effort control; hide the UI.
        "effortLevels": model_effort_levels(model),
        "inputPricePerM": None,
        "outputPricePerM": None,
        "description": " · ".join(p for p in parts if p),
    }


def _remote_model_info(model: Any, provider: str, caps: list[str], meta: Any) -> dict[str, Any]:
    """A model behind somebody's API key, named with its provider in brackets.

    "Llama 3.3 70B (DeepInfra)": the same model name can run here and in a
    data centre, and only one of the two is free.
    """
    label = PROVIDER_LABELS.get(provider, provider)
    return {
        "id": model.key,
        "provider": LOCAL_PROVIDER,
        "label": name_with_provider(getattr(model, "display_name", None) or model.key, model.key),
        "contextWindow": getattr(meta, "context_length", None) if meta else None,
        "maxOutput": None,
        "supportsVision": "vision" in caps,
        "supportsReasoning": False,
        "effortLevels": [],
        "inputPricePerM": None,
        "outputPricePerM": None,
        "description": f"zdalnie · {label}",
    }


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


# --------------------------------------------------------------------------- #
# Reasoning splitting
# --------------------------------------------------------------------------- #

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")


def _partial_suffix(buffer: str, tags: Iterable[str]) -> int:
    """Length of the longest suffix of ``buffer`` that is a proper prefix of a tag."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(buffer)), 0, -1):
            if buffer.endswith(tag[:n]):
                best = max(best, n)
                break
    return best


class ThinkSplitter:
    """Fallback stream splitter: ``<think>…</think>`` → reasoning, rest → text.

    Same interface as ``runtime.toolcalls.StreamSplitter``: ``feed(delta)``
    returns ``[(kind, text)]``, ``finish()`` flushes. Tags split across deltas
    are handled by holding back a possible tag prefix.
    """

    def __init__(self, tool_format: str | None = None, *, starts_in_reasoning: bool = False) -> None:
        self.tool_format = tool_format
        self._in_reasoning = starts_in_reasoning
        self._buffer = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        self._buffer += delta
        out: list[tuple[str, str]] = []
        while self._buffer:
            tags = _CLOSE_TAGS if self._in_reasoning else _OPEN_TAGS
            positions = [(self._buffer.find(t), t) for t in tags if t in self._buffer]
            if positions:
                pos, tag = min(positions)
                if pos:
                    out.append(("reasoning" if self._in_reasoning else "text", self._buffer[:pos]))
                self._buffer = self._buffer[pos + len(tag):]
                self._in_reasoning = not self._in_reasoning
                continue
            keep = _partial_suffix(self._buffer, tags)
            emit = self._buffer[: len(self._buffer) - keep]
            if emit:
                out.append(("reasoning" if self._in_reasoning else "text", emit))
            self._buffer = self._buffer[len(self._buffer) - keep:]
            break
        return out

    def finish(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        out = [("reasoning" if self._in_reasoning else "text", self._buffer)]
        self._buffer = ""
        return out


class ReasoningPrefix:
    """Wraps a splitter for prompts that END inside a reasoning block (the chat
    template already emitted ``<think>``): everything up to the closing tag is
    reasoning, the rest goes through ``inner``."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._prefix = True
        self._buffer = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        if not self._prefix:
            return list(self.inner.feed(delta))
        self._buffer += delta
        positions = [(self._buffer.find(t), t) for t in _CLOSE_TAGS if t in self._buffer]
        if positions:
            pos, tag = min(positions)
            out: list[tuple[str, str]] = []
            if pos:
                out.append(("reasoning", self._buffer[:pos]))
            rest = self._buffer[pos + len(tag):]
            self._buffer = ""
            self._prefix = False
            if rest:
                out.extend(self.inner.feed(rest))
            return out
        keep = _partial_suffix(self._buffer, _CLOSE_TAGS)
        emit = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep:]
        return [("reasoning", emit)] if emit else []

    def finish(self) -> list[tuple[str, str]]:
        if self._prefix:
            out = [("reasoning", self._buffer)] if self._buffer else []
            self._buffer = ""
            self._prefix = False
            return out + list(self.inner.finish())
        return list(self.inner.finish())


# --------------------------------------------------------------------------- #
# Formatter fallback
# --------------------------------------------------------------------------- #


class BasicFormatter:
    """Minimal stand-in for ``runtime.formatter.ChatFormatter`` (used only when
    that module is unavailable): the engine's own template, reasoning stripped."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def _kwargs(self, thinking: bool, effort: str | None = None) -> dict[str, Any] | None:
        info = getattr(self.engine, "template_info", None)
        levels = list(getattr(info, "effort_levels", None) or [])
        if effort is not None and levels:
            level = resolve_effort(effort, levels)
            if level is not None:
                from localcode.runtime.templates import effort_kwargs

                return effort_kwargs(getattr(info, "effort_kwarg", None), level) or None
        kwarg = getattr(info, "thinking_kwarg", None)
        return {kwarg: bool(thinking)} if kwarg else None

    def effort_levels(self) -> list[str]:
        info = getattr(self.engine, "template_info", None)
        return list(getattr(info, "effort_levels", None) or [])

    async def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
                     thinking: bool = False, effort: str | None = None) -> str:
        clean = [{k: v for k, v in m.items() if k != "reasoning"} for m in messages]
        return await self.engine.apply_template(clean, tools, add_generation_prompt=True,
                                                template_kwargs=self._kwargs(thinking, effort))

    async def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        return await self.engine.count_tokens(await self.render(messages, tools, thinking=False))

    def stop_strings(self) -> list[str]:
        info = getattr(self.engine, "template_info", None)
        return list(getattr(info, "stop_strings", None) or [])

    def starts_in_reasoning(self, prompt: str) -> bool:
        return prompt.rstrip().endswith(_OPEN_TAGS)


# --------------------------------------------------------------------------- #
# Title sanitising (reference sanitizeTitle / cleanTitle)
# --------------------------------------------------------------------------- #


def clean_title(raw: str) -> str | None:
    """One clean headline from model output, or ``None`` when it is not a title."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).replace("\r", "").strip()
    if not text or text.startswith("```"):
        return None
    first = next((line.strip() for line in text.split("\n") if line.strip()), "")
    line = re.sub(r"^(?:#{1,6}\s*|>\s*|[*+-]\s+|\d{1,2}[.)]\s+)", "", first)
    line = re.sub(r"^(?:title|tytuł)\s*:\s*", "", line, flags=re.I)
    line = re.sub(r"[*_`#|>]", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[\"'„”«»\s-]+", "", line)
    line = re.sub(r"[\"'„”«».,:;\s]+$", "", line)
    if not line or len(line) > 80:
        return None
    return truncate(line, 60)


# --------------------------------------------------------------------------- #
# Explicit memory requests
# --------------------------------------------------------------------------- #

_MEMORY_TRIGGERS = [
    re.compile(r"^\s*(?:proszę\s*,?\s*|prosze\s*,?\s*)?zapami[eę]taj(?:\s+sobie)?\s*[,:]?\s*(?:że|ze)?\s*(?P<fact>.+)$",
               re.I | re.S),
    re.compile(r"^\s*pami[eę]taj\s*,?\s*(?:że|ze)\s+(?P<fact>.+)$", re.I | re.S),
    re.compile(r"^\s*(?:please\s*,?\s*)?remember\s*(?:that\b|:)\s*(?P<fact>.+)$", re.I | re.S),
    re.compile(r"^\s*(?:please\s*,?\s*)?keep in mind\s*(?:that\b|:)\s*(?P<fact>.+)$", re.I | re.S),
]
_PREFERENCE_RE = re.compile(
    r"\b(wol[eę]|lubi[eę]|nie lubi[eę]|preferuj[eę]|zawsze|nigdy|prefer|like|dislike|always|never)\b", re.I)
#: Format and quantisation suffixes of a repository name ("-GGUF", "-4bit", "-Q4_K_M", "-i1"…).
_BUILD_SUFFIX_RE = re.compile(
    r"[-_.](?:gguf|mlx|i1|imatrix|awq|gptq|dwq|bf16|fp16|f16|f32|int[48]|\d+bit|q\d(?:_[a-z0-9]+)*|iq\d\w*)$")


def _model_family(key: str) -> str:
    """The model behind a key, without its format: both builds of one model share it.

    ``unsloth/Qwen3-4B-Instruct-2507-GGUF/…-Q4_K_M`` and
    ``mlx-community/Qwen3-4B-Instruct-2507-4bit`` → ``qwen3-4b-instruct-2507``.
    """
    if not key or parse_key(key) is not None:
        return ""
    parts = key.split("/")
    name = (parts[1] if len(parts) > 1 else parts[0]).lower()
    while (shorter := _BUILD_SUFFIX_RE.sub("", name)) != name:
        name = shorter
    return name


def extract_memory_request(text: str) -> tuple[str, str] | None:
    """``(kind, content)`` when the message explicitly asks to remember something."""
    for pattern in _MEMORY_TRIGGERS:
        match = pattern.match(text or "")
        if not match:
            continue
        fact = match.group("fact").strip().split("\n")[0].strip()
        sentence = re.match(r"^(.+?[.!?])\s+\S", fact)
        if sentence:
            fact = sentence.group(1)
        fact = truncate(collapse_ws(fact), 400).strip(" ,;:")
        if len(fact) < 3:
            return None
        fact = fact[0].upper() + fact[1:]
        kind = "preference" if _PREFERENCE_RE.search(fact) else "fact"
        return kind, fact
    return None


# --------------------------------------------------------------------------- #
# Turn state
# --------------------------------------------------------------------------- #


@dataclass
class _Turn:
    settings: dict[str, Any]
    conversation: dict[str, Any]
    user_message: dict[str, Any]
    model: Any | None
    model_key: str | None
    model_error: ApiError | None
    temperature: float
    max_tokens: int | None
    # Requested reasoning effort ("off"|"low"|"medium"|"high"); resolved per model.
    effort: str = "medium"


@dataclass
class _Stream:
    assistant_id: str
    text: str = ""
    #: Everything the model produced, before the splitter removed the tool-call
    #: markup — that markup is exactly what the tool loop has to read.
    raw: str = ""
    reasoning: str = ""
    status: str = "complete"
    error: str | None = None
    code: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    started: float = field(default_factory=time.monotonic)
    prompt_estimate: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    finalized: dict[str, Any] | None = None


class _TurnFailure(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class ChatService:
    """Streams chat turns; see the module docstring for the frame protocol."""

    def __init__(self, repo: ChatRepo, runtime: Any, library: Any,
                 settings_provider: Callable[[], Any] | None = None, *,
                 rag: RagIndex | None = None,
                 formatter_factory: Callable[[Any], Any] | None = None,
                 splitter_factory: Callable[[str | None], Any] | None = None,
                 title_timeout: float = TITLE_TIMEOUT_S,
                 status_delay: float = STATUS_DELAY_S,
                 flush_interval: float = FLUSH_INTERVAL_S,
                 compact_timeout: float = COMPACT_TIMEOUT_S) -> None:
        if settings_provider is None:
            from localcode.config import load_settings
            settings_provider = load_settings
        self.repo = repo
        self.runtime = runtime
        self.library = library
        self.settings_provider = settings_provider
        self.rag = rag or RagIndex(repo, self._embedder)
        self._formatter_factory = formatter_factory
        self._splitter_factory = splitter_factory
        self.title_timeout = title_timeout
        self.status_delay = status_delay
        self.flush_interval = flush_interval
        self.compact_timeout = compact_timeout
        self._formatters: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()
        self._counters: "weakref.WeakKeyDictionary[Any, TokenCounter]" = weakref.WeakKeyDictionary()
        self._background: set[asyncio.Task[Any]] = set()
        self._overflow_error = _context_overflow_error()

    # ------------------------------------------------------------ config access
    def _model_effort(self, model_key: str | None) -> str | None:
        """The reasoning level stored for this model in the library, if any.

        Set per model in "Ustawienia ładowania…"; some models are worth their
        thinking budget and some waste minutes on it, and that is a property of
        the model, not of the conversation.
        """
        if not model_key:
            return None
        getter = getattr(getattr(self, "library", None), "load_defaults", None)
        if getter is None:
            return None
        try:
            value = (getter(model_key) or {}).get("effort")
        except Exception:  # noqa: BLE001 - never break a turn over a preference
            log.debug("load_defaults(%r) failed", model_key, exc_info=True)
            return None
        return value or None

    def _app_config(self) -> Any | None:
        try:
            return self.settings_provider()
        except Exception:
            log.warning("settings provider failed", exc_info=True)
            return None

    def _embedder(self) -> RuntimeEmbedder | None:
        cfg = self._app_config()
        key = getattr(getattr(cfg, "chat", None), "embedding_model", None) if cfg else None
        if not key or self.runtime is None:
            return None
        return RuntimeEmbedder(self.runtime, key)

    # ---------------------------------------------------------------- library
    async def _call_library(self, name: str, *args: Any) -> Any:
        fn = getattr(self.library, name, None)
        if fn is None:
            return None
        if inspect.iscoroutinefunction(fn):
            return await fn(*args)
        return await _maybe_await(await asyncio.to_thread(fn, *args))

    async def local_models(self) -> list[Any]:
        """All models of the library (``library.list()``)."""
        if self.library is None:
            return []
        result = await self._call_library("list")
        return list(result or [])

    async def chat_models(self) -> list[Any]:
        return [m for m in await self.local_models() if is_chat_model(m)]

    def _remote_registry(self) -> Any | None:
        registry = getattr(self.runtime, "remote", None) if self.runtime is not None else None
        try:
            return registry if registry is not None and registry.providers else None
        except Exception:  # noqa: BLE001 - a stub runtime without providers
            return None

    async def remote_models(self) -> list[Any]:
        """Models behind the API keys saved in Ustawienia → Modele zdalne.

        Offered in the picker but never chosen on their own: every request to
        one is billed, so a conversation falls back to a local model only.
        """
        registry = self._remote_registry()
        if registry is None:
            return []
        try:
            return list(await registry.local_models())
        except Exception:  # noqa: BLE001 - a provider being down hides its models, nothing more
            log.info("remote model list unavailable", exc_info=True)
            return []

    async def model_infos(self) -> list[dict[str, Any]]:
        local = [model_info(m) for m in await self.chat_models()]
        infos = local + [model_info(m) for m in await self.remote_models()]
        # "Ulubione" first, local or remote; the rest keeps its order.
        favorites = getattr(self._app_config(), "favorite_models", None)
        return favorites_first(infos, favorites if isinstance(favorites, list) else None,
                               key=lambda info: info["id"])

    async def _find_remote(self, key: str) -> Any | None:
        parsed = parse_key(key)
        registry = self._remote_registry()
        if parsed is None or registry is None:
            return None
        listed = next((m for m in await self.remote_models() if m.key == key), None)
        if listed is not None:
            return listed
        try:
            if not registry.configured(parsed[0]):
                return None
            # The provider is set up but its list did not answer (or no longer
            # names the model): keep the conversation going on the bare id.
            from localcode.remote.engine import remote_model
            return remote_model(key)
        except Exception:  # noqa: BLE001
            return None

    async def find_model(self, key: str) -> Any | None:
        if key and parse_key(key) is not None:
            # Never let the local fuzzy resolver turn "remote:…:llama-70b" into
            # a llama that happens to be on disk.
            return await self._find_remote(key)
        if not key or self.library is None:
            return None
        model = await self._call_library("get", key)
        if model is None and hasattr(self.library, "resolve"):
            model = await self._call_library("resolve", key)
        if model is None:
            model = next((m for m in await self.local_models() if m.key == key), None)
        return model

    async def default_model_key(self, settings: dict[str, Any] | None = None) -> str | None:
        """The model a new conversation would use right now (``None`` = none installed)."""
        settings = settings or await self.repo.run(self.repo.get_settings)
        models = await self.chat_models()
        keys = {m.key for m in models}
        cfg = self._app_config()
        candidates = (settings["defaults"].get("model"),
                      getattr(getattr(cfg, "chat", None), "default_model", None) if cfg else None)
        if any(parse_key(c or "") for c in candidates):
            # A remote default counts only when somebody picked it; the
            # fallback below stays local.
            keys |= {m.key for m in await self.remote_models()}
        for candidate in candidates:
            if candidate and candidate in keys:
                return candidate
        return models[0].key if models else None

    async def _resolve_model(self, requested: str | None, conversation: dict[str, Any] | None,
                             settings: dict[str, Any]) -> tuple[Any | None, str | None, ApiError | None]:
        if requested:
            model = await self.find_model(requested)
            if model is None:
                stand_in = await self._stand_in(requested, conversation, settings)
                if stand_in is not None:
                    log.info("chat model %r is gone, answering with %r", requested, stand_in.key)
                    return stand_in, stand_in.key, None
                return None, requested, ApiError(
                    404, f"Model „{requested}” nie jest dostępny na tym komputerze. Wybierz inny model "
                         "albo pobierz go w zakładce „Odkrywaj”.", "model_not_found")
            if not is_chat_model(model):
                return None, requested, ApiError(
                    400, f"Model „{model.display_name}” służy tylko do embeddingów i nie prowadzi rozmowy. "
                         "Wybierz model czatu.", "not_a_chat_model")
            return model, model.key, None
        cfg = self._app_config()
        candidates = [
            (conversation or {}).get("model"),
            settings["defaults"].get("model"),
            getattr(getattr(cfg, "chat", None), "default_model", None) if cfg else None,
        ]
        for key in candidates:
            if not key:
                continue
            model = await self.find_model(key)
            if model is not None and is_chat_model(model):
                return model, model.key, None
            log.info("chat model %r is not available, trying the next default", key)
        models = await self.chat_models()
        if models:
            return models[0], models[0].key, None
        return None, None, ApiError(
            409, "Brak lokalnych modeli czatu. Pobierz model w zakładce „Odkrywaj” i spróbuj ponownie.",
            "no_models")

    async def _stand_in(self, gone: str, conversation: dict[str, Any] | None,
                        settings: dict[str, Any]) -> Any | None:
        """A local chat model to answer with when the requested one was deleted.

        The app always sends the conversation's model, so a model removed from
        disk would otherwise break every conversation that used it. The same
        model in another format wins (a GGUF replaced by its MLX build), then
        the usual defaults, then any chat model. Never a remote one: those are
        billed, and nobody picked it for this conversation.
        """
        models = await self.chat_models()
        family = _model_family(gone)
        same = [m for m in models if family and _model_family(m.repo_id or m.key) == family]
        if same:
            return same[0]
        cfg = self._app_config()
        for key in ((conversation or {}).get("model"), settings["defaults"].get("model"),
                    getattr(getattr(cfg, "chat", None), "default_model", None) if cfg else None):
            if not key or key == gone or parse_key(key) is not None:
                continue
            model = await self.find_model(key)
            if model is not None and is_chat_model(model):
                return model
        return models[0] if models else None

    # ------------------------------------------------------------- formatting
    def _formatter(self, engine: Any) -> Any:
        try:
            cached = self._formatters.get(engine)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
        if self._formatter_factory is not None:
            formatter = self._formatter_factory(engine)
        else:
            try:
                from localcode.runtime.formatter import ChatFormatter  # type: ignore[attr-defined]
                formatter = ChatFormatter(engine)
            except ImportError:
                log.warning("runtime.formatter unavailable; using the basic chat formatter")
                formatter = BasicFormatter(engine)
        try:
            self._formatters[engine] = formatter
        except TypeError:
            pass
        return formatter

    def _counter(self, engine: Any) -> TokenCounter:
        try:
            cached = self._counters.get(engine)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
        count_fn = getattr(engine, "count_tokens", None)
        counter = TokenCounter(count_fn if callable(count_fn) else None)
        try:
            self._counters[engine] = counter
        except TypeError:
            pass
        return counter

    @staticmethod
    def _stop_strings(formatter: Any, engine: Any) -> list[str]:
        fn = getattr(formatter, "stop_strings", None)
        try:
            stops = fn() if callable(fn) else fn
        except Exception:
            stops = None
        if stops is None:
            stops = getattr(getattr(engine, "template_info", None), "stop_strings", None)
        return [s for s in (stops or []) if isinstance(s, str) and s]

    @staticmethod
    def _starts_in_reasoning(formatter: Any, prompt: str) -> bool:
        attr = getattr(formatter, "starts_in_reasoning", None)
        if callable(attr):
            try:
                return bool(attr(prompt))
            except TypeError:
                try:
                    return bool(attr())
                except Exception:
                    pass
            except Exception:
                pass
        elif isinstance(attr, bool):
            return attr
        return prompt.rstrip().endswith(_OPEN_TAGS)

    def _make_splitter(self, engine: Any, formatter: Any, prompt: str) -> Any:
        tool_format = getattr(getattr(engine, "template_info", None), "tool_format", None)
        inner: Any
        if self._splitter_factory is not None:
            inner = self._splitter_factory(tool_format)
        else:
            try:
                from localcode.runtime.toolcalls import StreamSplitter  # type: ignore[attr-defined]
                inner = StreamSplitter(tool_format)
            except ImportError:
                inner = ThinkSplitter(tool_format)
            except Exception:
                log.warning("StreamSplitter failed to initialise; using the <think> splitter", exc_info=True)
                inner = ThinkSplitter(tool_format)
        if self._starts_in_reasoning(formatter, prompt):
            return ReasoningPrefix(inner)
        return inner

    async def _render(self, formatter: Any, messages: list[dict[str, Any]], *, thinking: bool,
                      effort: str | None = None, tools: list[dict[str, Any]] | None = None) -> str:
        return await formatter.render(messages, tools, thinking=thinking, effort=effort)

    def _tool_runner(self, engine: Any) -> Any:
        """The chat's four model-management tools, or a disabled runner.

        The chatbot has no file or shell tools on purpose; these exist because
        "pobierz mi jakiś model" is a sentence people say to a chat window and
        expect to *happen* (see ``localcode/chat/tools.py``).
        """
        from localcode.chat.tools import ChatToolRunner

        settings = self.settings_provider()
        enabled = bool(getattr(settings.agent, "manage_models", False))
        if not enabled:
            return ChatToolRunner(None, enabled=False)
        from localcode.agent.model_ops import LocalModelOps

        ops = LocalModelOps(settings, runtime=self.runtime, library=self.library,
                            downloads=getattr(self, "downloads", None))
        runner = ChatToolRunner(ops, enabled=True)
        return runner if runner.supported_by(engine) else ChatToolRunner(None, enabled=False)

    # ------------------------------------------------------------- assembling
    async def _context_inputs(self, settings: dict[str, Any], conversation: dict[str, Any]) -> tuple[
            dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], Any]:
        """``(folder, global memories, folder memories, scope)`` for a conversation."""
        repo = self.repo
        folder = (await repo.run(repo.get_folder, conversation["folderId"])) if conversation.get("folderId") else None
        scope = resolve_scope(folder)
        memories_global: list[dict[str, Any]] = []
        memories_folder: list[dict[str, Any]] = []
        if settings.get("memoryEnabled"):
            if scope.include_global:
                memories_global = await repo.run(repo.list_memories, scope="global")
            if folder:
                memories_folder = await repo.run(repo.list_memories, scope="folder", folder_id=folder["id"])
        return folder, memories_global, memories_folder, scope

    async def _build_context(self, *, settings: dict[str, Any], conversation: dict[str, Any],
                             folder: dict[str, Any] | None, memories_global: list[dict[str, Any]],
                             memories_folder: list[dict[str, Any]], history: list[dict[str, Any]],
                             retrieved: list[dict[str, Any]], query: str, engine: Any, formatter: Any,
                             max_tokens: int | None, summary: str | None = None,
                             summary_upto_seq: int = 0, scale: float = 1.0) -> ChatContext:
        """One budgeted prompt for this conversation (summary block included)."""
        builder = ContextBuilder(
            settings=settings, folder=folder, conversation=conversation, history=history,
            memories_global=memories_global, memories_folder=memories_folder,
            retrieved=retrieved, query=query, n_ctx=int(engine.n_ctx), max_tokens=max_tokens,
            counter=self._counter(engine), exact_count=lambda msgs: formatter.count(msgs, None),
            summary=summary, summary_upto_seq=summary_upto_seq,
            model_tools=self._tool_runner(engine).enabled,
        )
        return await builder.build(scale=scale)

    @staticmethod
    def context_frame(ctx: ChatContext) -> dict[str, Any]:
        """``{"type": "context", …}`` — the field names of ``events.ContextUpdate``."""
        report = ctx.report
        system = min(int(report.system_tokens), int(report.prompt_tokens))
        return {
            "type": "context",
            "used": int(report.prompt_tokens),
            "limit": int(report.n_ctx),
            "system": system,
            "tools": 0,                       # the chatbot never sends tool schemas
            "history": max(0, int(report.prompt_tokens) - system),
            "reserve": int(report.reserve),
            "summary": int(report.summary_tokens),   # part of "system"; 0 when not compacted
        }

    # ---------------------------------------------------------------- preparing
    async def _prepare(self, body: dict[str, Any]) -> _Turn:
        repo = self.repo
        settings = await repo.run(repo.get_settings)
        defaults = settings["defaults"]
        content: str = body["content"]
        attachment_ids: list[str] = body["attachmentIds"]

        conversation = None
        if body["conversationId"]:
            conversation = await repo.run(repo.get_conversation, body["conversationId"])
            if conversation is None:
                raise ApiError(404, "Nie znaleziono rozmowy.", "conversation_not_found")

        model, model_key, model_error = await self._resolve_model(body["model"], conversation, settings)
        temperature = next(v for v in (body["temperature"], (conversation or {}).get("temperature"),
                                       defaults["temperature"]) if v is not None)
        max_tokens = next((v for v in (body["maxTokens"], (conversation or {}).get("maxTokens"),
                                       defaults["maxTokens"]) if v is not None), None)
        # Request > per-conversation override > this model's own default > global setting.
        effort = next((v for v in (body.get("reasoningEffort"),
                                   (conversation or {}).get("reasoningEffort"),
                                   self._model_effort(model_key),
                                   getattr(getattr(self._app_config(), "chat", None),
                                           "reasoning_effort", None))
                       if normalize_effort(v) is not None), "medium")

        if conversation is None:
            if body["regenerate"]:
                raise ApiError(400, "Nie ma czego wygenerować ponownie — rozmowa jeszcze nie istnieje.",
                               "nothing_to_regenerate")
            if body["editMessageId"]:
                raise ApiError(404, "Nie można edytować wiadomości w nieistniejącej rozmowie.",
                               "conversation_not_found")
            if not content.strip() and not attachment_ids:
                raise ApiError(400, "Wiadomość jest pusta.", "empty_message")
            if body["folderId"] and not await repo.run(repo.get_folder, body["folderId"]):
                raise ApiError(404, "Wskazany projekt nie istnieje.", "folder_not_found")
            conversation = await repo.run(repo.create_conversation, {
                "folderId": body["folderId"],
                "title": derive_title(content),
                "provider": LOCAL_PROVIDER,
                "model": model_key,
            })
        elif model_key and (conversation.get("provider") != LOCAL_PROVIDER or conversation.get("model") != model_key):
            conversation = await repo.run(repo.update_conversation, conversation["id"],
                                          {"provider": LOCAL_PROVIDER, "model": model_key}) or conversation

        if body["regenerate"]:
            last = await repo.run(repo.last_user_message, conversation["id"])
            if last is None:
                raise ApiError(400, "Rozmowa nie zawiera wiadomości użytkownika do ponownego wygenerowania.",
                               "nothing_to_regenerate")
            await repo.run(repo.remove_messages_after, conversation["id"], last["seq"])
            user_message = await repo.run(repo.get_message, last["id"]) or last
        elif body["editMessageId"]:
            target = await repo.run(repo.get_message, body["editMessageId"])
            if target is None or target["conversationId"] != conversation["id"]:
                raise ApiError(404, "Nie znaleziono wiadomości do edycji.", "message_not_found")
            if target["role"] != "user":
                raise ApiError(400, "Edytować można wyłącznie wiadomości użytkownika.", "invalid_message_role")
            if not content.strip() and not attachment_ids and not target["attachments"]:
                raise ApiError(400, "Wiadomość jest pusta.", "empty_message")
            await repo.run(repo.update_message, target["id"], {"content": content, "status": "complete", "error": None})
            if attachment_ids:
                await repo.run(repo.attach_to_message, attachment_ids, target["id"])
            await repo.run(repo.remove_messages_after, conversation["id"], target["seq"])
            user_message = await repo.run(repo.get_message, target["id"]) or target
        else:
            if not content.strip() and not attachment_ids:
                raise ApiError(400, "Wiadomość jest pusta.", "empty_message")
            user_message = await repo.run(repo.create_message, {
                "conversationId": conversation["id"],
                "role": "user",
                "content": content,
                "status": "complete",
                "attachmentIds": attachment_ids,
            })

        await repo.run(repo.touch_conversation, conversation["id"])
        return _Turn(settings=settings, conversation=conversation, user_message=user_message, model=model,
                     model_key=model_key, model_error=model_error, temperature=float(temperature),
                     max_tokens=int(max_tokens) if max_tokens is not None else None,
                     effort=normalize_effort(effort, "medium") or "medium")

    # ------------------------------------------------------------------ helpers
    async def _abandon(self, task: asyncio.Future[Any], cleanup: Callable[[Any], Awaitable[Any]] | None) -> None:
        """Cancel a helper task whose result nobody will use; release what it got."""
        if not task.done():
            task.cancel()
            if cleanup is not None:
                def _late(t: asyncio.Future[Any]) -> None:
                    if not t.cancelled() and t.exception() is None:
                        self._spawn(cleanup(t.result()))
                task.add_done_callback(_late)
            return
        if cleanup is not None and not task.cancelled() and task.exception() is None:
            try:
                await cleanup(task.result())
            except Exception:
                log.warning("cleanup after abandoned task failed", exc_info=True)

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def drain(self) -> None:
        """Wait for background work (memory extraction, embeddings)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)
        await self.rag.drain()

    async def aclose(self) -> None:
        for task in list(self._background):
            task.cancel()
        await asyncio.gather(*list(self._background), return_exceptions=True)
        await self.rag.aclose()

    async def _finalize(self, turn: _Turn, stream: _Stream) -> dict[str, Any]:
        """Persist the finished assistant message and conversation usage."""
        if stream.finalized is not None:
            return stream.finalized
        repo = self.repo
        tokens_in = stream.tokens_in if stream.tokens_in is not None else stream.prompt_estimate
        tokens_out = stream.tokens_out if stream.tokens_out is not None else estimate_tokens(stream.reasoning + stream.text)
        stream.tokens_in, stream.tokens_out = tokens_in, tokens_out
        latency = int((time.monotonic() - stream.started) * 1000)
        stored = await repo.run(repo.update_message, stream.assistant_id, {
            "content": stream.text,
            "reasoning": stream.reasoning or None,
            "status": stream.status,
            "error": stream.error if stream.status == "error" else None,
            "tokensIn": tokens_in,
            "tokensOut": tokens_out,
            "costUsd": 0.0,
            "latencyMs": latency,
        })
        if stream.status != "error" or stream.text:
            await repo.run(repo.add_usage, turn.conversation["id"], tokens_in or 0, tokens_out or 0, 0.0)
        await repo.run(repo.touch_conversation, turn.conversation["id"])
        stream.finalized = stored or {}
        return stream.finalized

    # -------------------------------------------------------------------- send
    async def send(self, req: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Run one chat turn, yielding SSE frame dicts (see module docstring)."""
        body = validate_chat_request(req)
        turn = await self._prepare(body)
        inner = self._run_turn(turn)
        try:
            async for frame in inner:
                yield frame
        finally:
            # aclose() on this generator must reach the turn (to persist "stopped").
            await inner.aclose()

    async def _run_turn(self, turn: _Turn) -> AsyncIterator[dict[str, Any]]:
        repo = self.repo
        settings = turn.settings
        conversation = turn.conversation
        user_message = turn.user_message

        folder, memories_global, memories_folder, scope = await self._context_inputs(settings, conversation)
        query = compose_text(user_message)

        assistant = await repo.run(repo.create_message, {
            "conversationId": conversation["id"],
            "role": "assistant",
            "content": "",
            "status": "streaming",
            "provider": LOCAL_PROVIDER,
            "model": turn.model_key,
        })
        stream = _Stream(assistant_id=assistant["id"])
        yield {"type": "start", "conversationId": conversation["id"], "userMessage": user_message,
               "assistantMessageId": assistant["id"]}

        wants_title = (bool(settings["defaults"].get("autoTitle")) and not conversation.get("titleLocked")
                       and await repo.run(repo.count_messages, conversation["id"]) == 2)
        title_source = (user_message.get("content") or "").strip() or query

        lease: LeaseHandle | None = None
        engine: Any = None
        formatter: Any = None
        try:
            try:
                if turn.model_error is not None:
                    raise _TurnFailure(turn.model_error.message, turn.model_error.code)
                assert turn.model is not None and turn.model_key is not None

                # 1. Knowledge retrieval (before the chat slot is taken: the
                #    embedding model may need to be loaded next to it).
                top_k = max(0, int(settings["defaults"].get("ragTopK") or 0))
                retrieved: list[dict[str, Any]] = []
                if top_k > 0 and query.strip():
                    task = asyncio.ensure_future(self.rag.retrieve(
                        query, folder_id=scope.folder_id, include_global=scope.include_global, top_k=top_k))
                    try:
                        done, _ = await asyncio.wait({task}, timeout=self.status_delay)
                        if not done:
                            yield {"type": "status", "message": "Przeszukiwanie bazy wiedzy…"}
                        retrieved = await asyncio.shield(task)
                    except BaseException:
                        await self._abandon(task, None)
                        raise

                # 2. Chat model slot (JIT load).
                acquire = asyncio.ensure_future(open_lease(self.runtime, turn.model_key, purpose="chat"))
                try:
                    done, _ = await asyncio.wait({acquire}, timeout=self.status_delay)
                    if not done:
                        name = getattr(turn.model, "display_name", None) or turn.model_key
                        yield {"type": "status", "message": f"Ładowanie modelu {name}…"}
                    lease = await asyncio.shield(acquire)
                except BaseException:
                    await self._abandon(acquire, lambda handle: handle.release())
                    raise
                engine = lease.engine
                formatter = self._formatter(engine)

                # 3. Prompt within the context window (compacting when needed), then stream.
                thinking = bool(settings["defaults"].get("showReasoning"))
                scale = 1.0
                history = await repo.run(repo.list_messages, conversation["id"])
                summary: str | None = conversation.get("summary") or None
                summary_seq = int(conversation.get("summaryUptoSeq") or 0)

                async def assemble(factor: float) -> ChatContext:
                    return await self._build_context(
                        settings=settings, conversation=conversation, folder=folder,
                        memories_global=memories_global, memories_folder=memories_folder,
                        history=history, retrieved=retrieved, query=query, engine=engine,
                        formatter=formatter, max_tokens=turn.max_tokens,
                        summary=summary, summary_upto_seq=summary_seq, scale=factor)

                for attempt in (1, 2):
                    ctx = await assemble(scale)
                    if attempt == 1:
                        compactor = self._compactor(engine, formatter, max_tokens=turn.max_tokens)
                        # The prompt is measured AFTER trimming, so in a small
                        # window it always "fits" — the oldest turns were simply
                        # dropped. Summarise them instead of losing them, which is
                        # exactly what a 2k–4k model needs most.
                        dropped = int(getattr(ctx.report, "dropped_messages", 0) or 0)
                        if compactor.needs_compaction(ctx.prompt_tokens) or dropped:
                            before = ctx.prompt_tokens
                            yield {"type": "status", "message": COMPACT_STATUS}
                            result = await self._compact_now(
                                compactor, conversation["id"], history,
                                previous_summary=summary or "", summary_upto_seq=summary_seq, force=False)
                            if result.changed:
                                summary, summary_seq = result.summary, result.upto_seq
                                ctx = await assemble(scale)
                                yield {"type": "compaction", "before": int(before),
                                       "after": int(ctx.prompt_tokens), "strategy": result.strategy}
                            elif result.error and result.error != "nothing to summarize":
                                # the status frame must not hang: say plainly that the
                                # oldest messages are being dropped instead.
                                yield {"type": "status", "message": COMPACT_FAILED_STATUS}
                    stream.prompt_estimate = ctx.prompt_tokens
                    if attempt == 1:
                        stream.citations = ctx.citations
                        await repo.run(repo.update_message, assistant["id"], {"citations": ctx.citations})
                        if ctx.citations:
                            yield {"type": "citations", "citations": ctx.citations}
                        note = ctx.report.summary_pl()
                        if note:
                            log.info("chat context trimmed: %s", ctx.report.to_dict())
                            yield {"type": "status", "message": note}
                        yield self.context_frame(ctx)
                    runner = self._tool_runner(engine)
                    schemas = runner.schemas()
                    messages = list(ctx.messages)
                    prompt = await self._render(formatter, messages, thinking=thinking,
                                                effort=turn.effort if thinking else "off",
                                                tools=schemas)
                    generation = self._generate(engine, formatter, prompt, ctx.max_output_tokens,
                                                turn.temperature, stream,
                                                cache_key=f"localcode-chat-{conversation['id']}")
                    try:
                        async for frame in generation:
                            yield frame
                        # The model may have asked for a tool instead of
                        # answering; run it and let it finish the thought.
                        for _ in range(runner.max_rounds if runner.enabled else 0):
                            calls = runner.parse(stream.raw, engine)
                            if not calls:
                                break
                            asked = stream.raw
                            stream.text = ""
                            stream.raw = ""
                            messages.append(runner.assistant_message(
                                runner.content_without_calls(asked, engine), calls))
                            async for frame in runner.run(calls):
                                message = frame.pop("message", None)
                                if message is not None:
                                    messages.append(message)
                                yield frame
                            prompt = await self._render(
                                formatter, messages, thinking=thinking,
                                effort=turn.effort if thinking else "off", tools=schemas)
                            again = self._generate(engine, formatter, prompt,
                                                   ctx.max_output_tokens, turn.temperature, stream,
                                                   cache_key=f"localcode-chat-{conversation['id']}")
                            try:
                                async for frame in again:
                                    yield frame
                            finally:
                                await again.aclose()
                        break
                    except self._overflow_error:
                        if attempt == 2 or stream.text or stream.reasoning:
                            raise _TurnFailure(
                                "Rozmowa nie mieści się w oknie kontekstu modelu. Rozpocznij nową rozmowę, "
                                "skróć wiadomość lub wczytaj model z większym kontekstem.", "context_overflow")
                        log.info("context overflow reported by the engine; retrying with a tighter budget")
                        scale = 0.7
                    finally:
                        await generation.aclose()
            except (asyncio.CancelledError, GeneratorExit):
                stream.status = "stopped"
                try:
                    await asyncio.shield(self._finalize(turn, stream))
                except Exception:
                    log.exception("could not persist the stopped answer")
                raise
            except _TurnFailure as exc:
                stream.status, stream.error, stream.code = "error", exc.message, exc.code
            except EngineError as exc:
                stream.status, stream.error, stream.code = "error", str(exc) or GENERIC_ERROR, "engine_error"
            except Exception:
                log.exception("chat turn failed")
                stream.status, stream.error, stream.code = "error", GENERIC_ERROR, "internal_error"

            stored = await self._finalize(turn, stream)
            if stream.status == "error":
                yield {"type": "error", "message": stream.error or GENERIC_ERROR, "code": stream.code or "error"}
                return
            yield {"type": "usage", "tokensIn": stream.tokens_in, "tokensOut": stream.tokens_out, "costUsd": 0}
            yield {"type": "done", "message": stored}

            # 4. Auto-title (first exchange only), still on the same slot.
            if wants_title:
                title: str | None = None
                if engine is not None and stream.status == "complete":
                    try:
                        title = await asyncio.wait_for(self._model_title(engine, formatter, title_source),
                                                       self.title_timeout)
                    except (asyncio.TimeoutError, EngineError):
                        title = None
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.warning("title generation failed", exc_info=True)
                        title = None
                final_title = title or derive_title(title_source)
                if final_title:
                    await repo.run(repo.set_title, conversation["id"], final_title)
                    yield {"type": "title", "conversationId": conversation["id"], "title": final_title}

            # 5. Explicit "remember that …" requests → memory (background, cheap).
            if settings.get("memoryEnabled") and stream.status == "complete":
                self._spawn(self._remember(user_message, folder))
        finally:
            if lease is not None:
                await asyncio.shield(lease.release())

    async def _generate(self, engine: Any, formatter: Any, prompt: str, max_tokens: int, temperature: float,
                        stream: _Stream, cache_key: str | None = None) -> AsyncIterator[dict[str, Any]]:
        splitter = self._make_splitter(engine, formatter, prompt)
        params = SamplingParams(max_tokens=max(1, int(max_tokens)), temperature=temperature,
                                stop=self._stop_strings(formatter, engine), cache_key=cache_key)
        last_flush = time.monotonic()
        finish_reason: str | None = None
        text_started = False
        iterator = engine.complete(prompt, params)

        def frames(pieces: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
            nonlocal text_started
            out: list[dict[str, Any]] = []
            for kind, piece in pieces:
                if not piece:
                    continue
                if kind == "reasoning":
                    if not stream.reasoning:
                        piece = piece.lstrip("\n")
                        if not piece:
                            continue
                    stream.reasoning += piece
                    out.append({"type": "reasoning", "delta": piece})
                elif kind == "text":
                    if not text_started:
                        piece = piece.lstrip()
                        if not piece:
                            continue
                        text_started = True
                    stream.text += piece
                    out.append({"type": "text", "delta": piece})
                else:
                    log.debug("ignoring splitter output of kind %r", kind)
            return out

        try:
            async for chunk in iterator:
                if chunk.text:
                    stream.raw += chunk.text
                    for frame in frames(splitter.feed(chunk.text)):
                        yield frame
                    now = time.monotonic()
                    if now - last_flush >= self.flush_interval:
                        last_flush = now
                        await self.repo.run(self.repo.update_message, stream.assistant_id,
                                            {"content": stream.text, "reasoning": stream.reasoning or None})
                if chunk.done:
                    finish_reason = chunk.finish_reason
                    if chunk.stats is not None:
                        stream.tokens_in = chunk.stats.prompt_tokens or None
                        stream.tokens_out = chunk.stats.completion_tokens
                    break
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await asyncio.shield(aclose())
        for frame in frames(splitter.finish()):
            yield frame
        if finish_reason == "cancelled":
            stream.status = "stopped"
        elif finish_reason == "error":
            raise _TurnFailure("Silnik modelu przerwał generowanie odpowiedzi.", "engine_error")

    # ------------------------------------------------------------ helper calls
    async def _complete_text(self, engine: Any, formatter: Any, messages: list[dict[str, Any]], *,
                             max_tokens: int, temperature: float, max_chars: int) -> str:
        """One short, tool-free, thinking-off completion; returns the answer text.

        Reasoning is dropped (the splitter separates it), the loop stops after
        ``max_chars`` so a rambling model cannot hold the slot forever.
        """
        prompt = await self._render(formatter, messages, thinking=False)
        splitter = self._make_splitter(engine, formatter, prompt)
        params = SamplingParams(max_tokens=max(1, int(max_tokens)), temperature=temperature,
                                stop=self._stop_strings(formatter, engine))
        text = ""
        iterator = engine.complete(prompt, params)
        try:
            async for chunk in iterator:
                if chunk.text:
                    text += "".join(p for kind, p in splitter.feed(chunk.text) if kind == "text")
                if chunk.done or len(text) > max_chars:
                    break
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await asyncio.shield(aclose())
        return text + "".join(p for kind, p in splitter.finish() if kind == "text")

    # -------------------------------------------------------------------- title
    async def _model_title(self, engine: Any, formatter: Any, source: str) -> str | None:
        messages = [{"role": "system", "content": TITLE_SYSTEM},
                    {"role": "user", "content": build_title_prompt(source)}]
        text = await self._complete_text(engine, formatter, messages, max_tokens=TITLE_MAX_TOKENS,
                                         temperature=0.2, max_chars=200)
        return clean_title(text)

    async def generate_title(self, conversation_id: str, *, auto: bool = False) -> dict[str, str]:
        """``POST /api/conversations/{id}/title[?auto=1]`` (reference title route)."""
        repo = self.repo
        conversation = await repo.run(repo.get_conversation, conversation_id)
        if conversation is None:
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        if conversation.get("titleLocked") and auto:
            return {"title": conversation["title"]}
        messages = await repo.run(repo.list_messages, conversation_id)
        first = next((m for m in messages if m["role"] == "user" and m["content"].strip()), None)
        if first is None:
            raise ApiError(400, "Rozmowa nie zawiera jeszcze wiadomości użytkownika, z której można utworzyć tytuł.",
                           "no_user_message")
        settings = await repo.run(repo.get_settings)
        model, model_key, error = await self._resolve_model(None, conversation, settings)
        generated: str | None = None
        if error is None and model_key and self.runtime is not None:
            try:
                async def _run() -> str | None:
                    handle = await open_lease(self.runtime, model_key, purpose="chat")
                    try:
                        return await self._model_title(handle.engine, self._formatter(handle.engine), first["content"])
                    finally:
                        await handle.release()
                generated = await asyncio.wait_for(_run(), self.title_timeout)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.info("model title generation failed; using the heuristic", exc_info=True)
        title = generated or derive_title(first["content"])
        await repo.run(repo.set_title, conversation_id, title)
        return {"title": collapse_ws(title)}

    # -------------------------------------------------------------- compaction
    def compact_threshold(self) -> float:
        """Share of the prompt budget that triggers automatic compaction."""
        cfg = self._app_config()
        for holder in ("chat", "agent"):
            value = getattr(getattr(cfg, holder, None), "auto_compact_threshold", None) if cfg else None
            try:
                if value is not None and 0.1 <= float(value) <= 1.0:
                    return float(value)
            except (TypeError, ValueError):
                log.debug("ignoring an invalid %s.auto_compact_threshold (%r)", holder, value)
        return DEFAULT_THRESHOLD

    def _compactor(self, engine: Any, formatter: Any, *, max_tokens: int | None) -> ChatCompactor:
        """A compactor summarising with the very model that answers the turn."""
        async def summarize(messages: list[dict[str, Any]], want: int) -> str:
            return await self._complete_text(engine, formatter, messages, max_tokens=want,
                                             temperature=0.2, max_chars=max(400, want * 8))

        return ChatCompactor(n_ctx=int(engine.n_ctx), max_tokens=max_tokens,
                             counter=self._counter(engine), threshold=self.compact_threshold(),
                             summarizer=summarize, timeout=self.compact_timeout,
                             language_hint=getattr(self._app_config(), "language", None),
                             model_params_b=getattr(getattr(getattr(engine, "model", None), "meta", None),
                                                    "params_b", None))

    async def _compact_now(self, compactor: ChatCompactor, conversation_id: str,
                           history: list[dict[str, Any]], *, previous_summary: str = "",
                           summary_upto_seq: int = 0, force: bool = False) -> CompactionResult:
        """Summarise and persist; a failure only logs and returns ``ok=False``."""
        result = await compactor.summarize(history, previous_summary=previous_summary,
                                           summary_upto_seq=summary_upto_seq, force=force)
        if not result.changed:
            if result.error:
                log.info("chat compaction skipped for %s: %s", conversation_id, result.error)
            return result
        if int(result.upto_seq) <= 0:
            log.warning("chat compaction for %s produced upto_seq=%s — discarding it",
                        conversation_id, result.upto_seq)
            return CompactionResult(ok=False, error="compaction covered no messages")
        await self.repo.run(self.repo.set_summary, conversation_id, result.summary,
                            result.upto_seq, result.tokens)
        log.info("chat conversation %s compacted: %d messages → %d tokens of summary%s",
                 conversation_id, result.messages, result.tokens, " (extended)" if result.extended else "")
        return result

    async def compact_conversation(self, conversation_id: str) -> dict[str, Any]:
        """``POST /api/conversations/{id}/compact`` — summarise the whole backlog now.

        Returns ``{"summary", "before", "after", "changed", "strategy", "messages"}``
        where ``before``/``after`` are the prompt sizes of the next turn (without
        retrieval, which depends on the question). Nothing to summarise is not an
        error — it answers ``changed: false`` with ``before == after``.
        """
        repo = self.repo
        conversation = await repo.run(repo.get_conversation, conversation_id)
        if conversation is None:
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        settings = await repo.run(repo.get_settings)
        history = await repo.run(repo.list_messages, conversation_id)
        model, model_key, error = await self._resolve_model(None, conversation, settings)
        if error is not None:
            raise error
        if not model_key or self.runtime is None:
            raise ApiError(409, "Brak modelu, którym można skompaktować rozmowę.", "no_models")
        max_tokens = conversation.get("maxTokens") or settings["defaults"].get("maxTokens")
        folder, memories_global, memories_folder, _scope = await self._context_inputs(settings, conversation)
        handle = await open_lease(self.runtime, model_key, purpose="chat")
        try:
            engine = handle.engine
            formatter = self._formatter(engine)

            async def measure(summary: str | None, seq: int) -> int:
                ctx = await self._build_context(
                    settings=settings, conversation=conversation, folder=folder,
                    memories_global=memories_global, memories_folder=memories_folder,
                    history=history, retrieved=[], query="", engine=engine, formatter=formatter,
                    max_tokens=max_tokens, summary=summary, summary_upto_seq=seq)
                return int(ctx.prompt_tokens)

            stored = conversation.get("summary") or ""
            before = await measure(stored or None, int(conversation.get("summaryUptoSeq") or 0))
            compactor = self._compactor(engine, formatter, max_tokens=max_tokens)
            result = await self._compact_now(compactor, conversation_id, history,
                                             previous_summary=stored,
                                             summary_upto_seq=int(conversation.get("summaryUptoSeq") or 0),
                                             force=True)
            if not result.changed:
                if result.error and result.error != "nothing to summarize":
                    raise ApiError(409, COMPACT_FAILED, "compaction_failed")
                return {"summary": stored, "before": before, "after": before, "changed": False,
                        "strategy": STRATEGY_SUMMARIZE, "messages": 0}
            after = await measure(result.summary, result.upto_seq)
            return {"summary": result.summary, "before": before, "after": after, "changed": True,
                    "strategy": result.strategy, "messages": result.messages}
        finally:
            await asyncio.shield(handle.release())

    # ------------------------------------------------------------------ memory
    async def _remember(self, user_message: dict[str, Any], folder: dict[str, Any] | None) -> dict[str, Any] | None:
        request = extract_memory_request(user_message.get("content") or "")
        if request is None:
            return None
        kind, content = request
        repo = self.repo
        if folder:
            existing = await repo.run(repo.list_memories, scope="folder", folder_id=folder["id"])
            data = {"scope": "folder", "folderId": folder["id"]}
        else:
            existing = await repo.run(repo.list_memories, scope="global")
            data = {"scope": "global"}
        needle = collapse_ws(content).lower()
        if any(collapse_ws(m["content"]).lower() == needle for m in existing):
            return None
        return await repo.run(repo.create_memory, {**data, "kind": kind, "content": content, "source": "auto"})


# Re-exported for integrators that import the report type from here.
__all__ += ["ContextReport", "extract_memory_request", "clean_title", "is_chat_model"]
