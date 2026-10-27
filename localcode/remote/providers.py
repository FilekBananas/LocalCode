"""The clouds LocalCode can borrow a model from.

OpenAI, Anthropic, Google Gemini, DeepInfra and unlid.ai.  All but two speak the
same dialect — OpenAI's ``/chat/completions`` — Anthropic speaks its own and
Gemini a third, so there are three client shapes here rather than five.

What every provider has to answer:

``models()``
    what this key can use, as :class:`RemoteModel` rows for the store and the
    model pickers;
``stream()``
    a streamed answer for a conversation, yielding text deltas and, at the end,
    whatever tool calls the model asked for.

Nothing is cached on disk and no key ever leaves the settings file: the model
list is fetched on demand and held in memory for :data:`LIST_TTL_S`, so a
revoked key stops working in minutes rather than surviving a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Literal

from localcode.config import SERVICE_TIERS, TIERED_PROVIDERS, service_tier as effective_tier

log = logging.getLogger(__name__)

#: How long a fetched model list stays fresh.
LIST_TTL_S = 300.0
HTTP_TIMEOUT_S = 60.0
#: Longest silence allowed *inside* a streamed answer.
#:
#: A generation has no useful total length — a frontier model asked for a
#: refactor can stream for minutes, and cutting it at sixty seconds killed a
#: perfectly healthy answer mid-sentence. What does mean something is a provider
#: that has stopped sending: that is this.
STREAM_IDLE_TIMEOUT_S = 120.0
#: How long a Flex request may sit in the provider's queue before its first
#: byte. DeepInfra holds one for up to ten minutes when capacity is short and
#: then answers 429; the two minutes on top are the margin that lets that 429
#: arrive instead of our own timeout cutting the wait short.
FLEX_QUEUE_TIMEOUT_S = 12 * 60.0
#: "Too many requests" / "overloaded": tried again instead of ending the turn.
#: A provider on a small plan allows a handful of requests a minute, and an
#: agent sends one per step — the third step of a task is exactly when such a
#: limit bites, and waiting a few seconds is all it takes.
RETRY_STATUSES: tuple[int, ...] = (429, 503, 529)
#: Waits between attempts when the provider does not say (``Retry-After``).
RETRY_DELAYS_S: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
#: A ``Retry-After`` longer than this is reported instead of waited out.
MAX_RETRY_WAIT_S = 60.0
#: A spent quota or an empty account: no amount of waiting helps.
_QUOTA_RE = re.compile(r"quota|insufficient|billing|balance",
                       re.IGNORECASE)
#: How a provider names the field when it refuses it ("service_tier", "service tier").
_TIER_RE = re.compile(r"service[\s_-]?tier", re.IGNORECASE)
#: Providers in the order the settings screen lists them.
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini", "deepinfra", "unlid")

PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "deepinfra": "DeepInfra",
    "unlid": "Unlid",
}
#: Where to get a key, shown next to the empty field.
PROVIDER_KEY_URLS: dict[str, str] = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
    "gemini": "https://aistudio.google.com/app/apikey",
    "deepinfra": "https://deepinfra.com/dash/api_keys",
    "unlid": "https://unlid.ai/account",
}
DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    # unlid.ai: OpenAI-compatible (/v1/models, /v1/chat/completions with tools
    # and streaming); keys start with "ul-".
    "unlid": "https://api.unlid.ai/v1",
}
#: Model-key prefix: ``remote:openai:gpt-5``.
KEY_PREFIX = "remote"
#: How the tiers are named in messages (the same words as the settings screen).
TIER_LABELS: dict[str, str] = {"standard": "Standard", "flex": "Flex", "priority": "Priority"}


class RemoteError(Exception):
    """A provider failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "remote_error", status: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass
class RemoteModel:
    """One model a provider offers."""

    provider: str
    model_id: str
    display_name: str = ""
    context_length: int | None = None
    description: str | None = None
    owned_by: str | None = None
    #: Czy model umie wołać narzędzia. ``None`` = dostawca nie mówi, więc
    #: zakładamy, że umie (tak było, zanim ktokolwiek to podawał).
    supports_tools: bool | None = None
    #: Whether it reads images (a gateway's tags, or the provider's own line-up).
    supports_vision: bool = False

    @property
    def key(self) -> str:
        return f"{KEY_PREFIX}:{self.provider}:{self.model_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "provider": self.provider,
            "providerLabel": PROVIDER_LABELS.get(self.provider, self.provider),
            "modelId": self.model_id,
            "displayName": self.display_name or self.model_id,
            "contextLength": self.context_length,
            "description": self.description,
            "supportsTools": self.supports_tools is not False,
            "remote": True,
        }


@dataclass
class RemoteChunk:
    """One piece of a streamed answer."""

    text: str = ""
    done: bool = False
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Prompt tokens the provider read from its cache (part of ``prompt_tokens``).
    cached_tokens: int = 0


def parse_key(key: str) -> tuple[str, str] | None:
    """``"remote:openai:gpt-5"`` → ``("openai", "gpt-5")``; ``None`` otherwise."""
    parts = str(key or "").split(":", 2)
    if len(parts) == 3 and parts[0] == KEY_PREFIX and parts[1] in PROVIDERS and parts[2]:
        return parts[1], parts[2]
    return None


def is_remote_key(key: str) -> bool:
    return parse_key(key) is not None


def provider_label(key: str) -> str | None:
    """``"remote:deepinfra:…"`` → ``"DeepInfra"``; ``None`` for a local key."""
    parsed = parse_key(key)
    return PROVIDER_LABELS.get(parsed[0], parsed[0]) if parsed else None


def name_with_provider(name: str, key: str) -> str:
    """"Llama 3.3 70B (DeepInfra)" for a remote key, ``name`` otherwise.

    Every picker names a remote model this way: the same model can run on this
    Mac and in a data centre, and only one of the two is free.
    """
    label = provider_label(key)
    if label is None:
        return name
    suffix = f" ({label})"
    return name if name.endswith(suffix) else name + suffix


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #


class RemoteClient:
    """HTTP for one provider.  ``dialect`` decides the request shape."""

    def __init__(self, provider: str, api_key: str, *, base_url: str | None = None,
                 service_tier: str | None = None,
                 session_factory: Any = None, sleep: Any = None) -> None:
        self.provider = provider
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
        #: standard | flex | priority — already narrowed to what this provider
        #: takes, so Anthropic and Gemini are always "standard" here.
        self.service_tier = effective_tier(provider, service_tier)
        self._session_factory = session_factory
        self._sleep = sleep or asyncio.sleep
        self._models: tuple[float, list[RemoteModel]] | None = None

    @property
    def dialect(self) -> Literal["openai", "anthropic", "gemini"]:
        if self.provider == "anthropic":
            return "anthropic"
        if self.provider == "gemini":
            return "gemini"
        return "openai"

    # plumbing ----------------------------------------------------------------
    def _session(self, *, streaming: bool = False) -> Any:
        if self._session_factory is not None:
            return self._session_factory()
        import aiohttp

        return aiohttp.ClientSession(timeout=self._timeout(
            aiohttp, streaming=streaming, queued=streaming and self.service_tier == "flex"))

    @staticmethod
    def _timeout(aiohttp: Any, *, streaming: bool, queued: bool = False) -> Any:
        """``total`` for a listing, silence-between-chunks for a generation.

        ``ClientTimeout.total`` covers reading the body, so one value cannot
        serve both: sixty seconds is generous for ``/models`` and far too short
        for an answer the model is still writing.

        ``queued`` is a Flex generation: ``sock_read`` also covers the wait for
        the response headers, and a provider queueing the request sends nothing
        at all until it has capacity — up to :data:`FLEX_QUEUE_TIMEOUT_S`. One
        aiohttp setting cannot tell that wait from silence mid-answer, so a Flex
        answer that stalls is given the same allowance.
        """
        if not streaming:
            return aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)
        return aiohttp.ClientTimeout(total=None, sock_connect=HTTP_TIMEOUT_S,
                                     sock_read=FLEX_QUEUE_TIMEOUT_S if queued
                                     else STREAM_IDLE_TIMEOUT_S)

    def _headers(self) -> dict[str, str]:
        if self.dialect == "anthropic":
            return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                    "content-type": "application/json"}
        if self.dialect == "gemini":
            return {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        return {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}

    def _error(self, status: int, body: str, *, retries: int = 0,
               generation: bool = False) -> RemoteError:
        """A Polish :class:`RemoteError` for an HTTP failure.

        ``generation``: the request carried the service tier (a listing never
        does), so a Flex 429 or a refused tier can be named as such.
        """
        label = PROVIDER_LABELS.get(self.provider, self.provider)
        if status in (401, 403):
            return RemoteError(f"{label} odrzucił klucz API. Sprawdź go w ustawieniach.",
                               "bad_key", 401)
        if generation and self._flex_unavailable(status, body):
            waited = " (czekał do 10 min)" if self.provider == "deepinfra" else ""
            return RemoteError(f"{label} nie miał wolnej mocy w trybie Flex{waited}. "
                               "Spróbuj ponownie albo przełącz na Standard w Ustawieniach.",
                               "flex_unavailable", 429)
        if generation and self._tier_refused(status, body):
            tier = TIER_LABELS.get(self.service_tier, self.service_tier)
            return RemoteError(f"Ten model u dostawcy {label} nie obsługuje trybu {tier}. "
                               f"Przełącz {label} na Standard w Ustawieniach → Modele zdalne.",
                               "tier_unsupported", 400)
        if status == 429:
            # The provider's own words say which limit it is (per minute, per
            # day, an empty account) — the one thing the user needs to know.
            excerpt = self._redact((body or "").strip().replace("\n", " "))[:200]
            tried = f" Ponawiałem {retries} razy, bez skutku." if retries else ""
            return RemoteError(f"{label} prosi o zwolnienie tempa (limit zapytań).{tried} {excerpt}".strip(),
                               "rate_limited", 429)
        if status == 404:
            return RemoteError(f"{label} nie zna tego modelu.", "model_not_found", 404)
        excerpt = self._redact((body or "").strip().replace("\n", " "))[:200]
        return RemoteError(f"{label} odpowiedział błędem {status}. {excerpt}".strip(),
                           "provider_error", 502)

    def _flex_unavailable(self, status: int, body: str) -> bool:
        """A Flex request the provider gave up queueing (DeepInfra: 429, not billed).

        A spent quota answers 429 too, and that one is not about capacity — it
        keeps the ordinary message, which quotes the provider's own words.
        """
        return (self.service_tier == "flex" and status == 429
                and not _QUOTA_RE.search(body or ""))

    def _tier_refused(self, status: int, body: str) -> bool:
        """A model that does not take the tier this provider is set to."""
        return (self.service_tier != "standard" and status in (400, 422)
                and bool(_TIER_RE.search(body or "")))

    def _redact(self, text: str) -> str:
        """The provider's error text without our key in it.

        That excerpt is shown in the app, kept in the agent's transcript and
        written to the log. Real providers rarely echo a key back, but a proxy
        in front of one may reflect the request headers — and a key that ends up
        in a transcript has left the settings file for good.
        """
        key = (self.api_key or "").strip()
        if len(key) >= 6 and key in text:          # not a one-letter "key" eating the text
            text = text.replace(key, f"{key[:3]}…{key[-2:]}")
        return text

    # models ------------------------------------------------------------------
    async def models(self, *, refresh: bool = False) -> list[RemoteModel]:
        """Everything this key can use (cached for :data:`LIST_TTL_S`)."""
        now = time.monotonic()
        if not refresh and self._models and now - self._models[0] < LIST_TTL_S:
            return list(self._models[1])
        path = "/models"
        async with self._session() as session:
            url = f"{self.base_url}{path}"
            async with session.get(url, headers=self._headers()) as response:
                body = await response.text()
                if response.status >= 400:
                    raise self._error(response.status, body)
                data = json.loads(body or "{}")
        models = _parse_models(self.provider, data)
        self._models = (now, models)
        return list(models)

    # generation --------------------------------------------------------------
    async def stream(self, messages: list[dict[str, Any]], *, model: str,
                     tools: list[dict[str, Any]] | None = None,
                     max_tokens: int | None = 1024, temperature: float = 0.3,
                     stop: Iterable[str] | None = None,
                     cache_key: str | None = None) -> AsyncIterator[RemoteChunk]:
        """Stream an answer for ``messages``; ``max_tokens=None`` = as long as the model goes."""
        url, payload = self._request(messages, model=model, tools=tools,
                                     max_tokens=max_tokens, temperature=temperature, stop=stop)
        key = _cache_key(cache_key)
        if key and self.dialect == "openai":
            # Without it a provider may send each step to another replica and
            # read nothing from cache (DeepInfra: "requests with the same
            # prompt_cache_key and model will share a KV cache").
            payload["prompt_cache_key"] = key
        attempt = 0
        async with self._session(streaming=True) as session:
            while True:
                async with session.post(url, headers=self._headers(), json=payload) as response:
                    if response.status < 400:
                        async for chunk in _read_stream(self.dialect, response):
                            yield chunk
                        return
                    body = await response.text()
                    # A Flex 429 already *was* the wait — up to ten minutes in
                    # the provider's queue. Asking again would start another.
                    wait = (None if self._flex_unavailable(response.status, body)
                            else self._retry_wait(response.status, body,
                                                  getattr(response, "headers", None), attempt))
                    if wait is None:
                        raise self._error(response.status, body, retries=attempt,
                                          generation=True)
                log.info("%s answered %s; trying again in %.0f s (attempt %d)",
                         self.provider, response.status, wait, attempt + 1)
                await self._sleep(wait)
                attempt += 1

    @staticmethod
    def _retry_wait(status: int, body: str, headers: Any, attempt: int) -> float | None:
        """Seconds to wait before trying again, or ``None`` to give up now."""
        if status not in RETRY_STATUSES or attempt >= len(RETRY_DELAYS_S):
            return None
        if _QUOTA_RE.search(body or ""):
            return None
        after = _retry_after(headers)
        if after is None:
            return RETRY_DELAYS_S[attempt]
        return after if after <= MAX_RETRY_WAIT_S else None

    def _request(self, messages: list[dict[str, Any]], *, model: str,
                 tools: list[dict[str, Any]] | None, max_tokens: int | None, temperature: float,
                 stop: Iterable[str] | None) -> tuple[str, dict[str, Any]]:
        if self.dialect == "anthropic":
            system = "\n\n".join(str(m.get("content") or "") for m in messages
                                 if m.get("role") == "system")
            payload: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens or ANTHROPIC_MAX_OUTPUT,       # required there
                "temperature": temperature,
                "stream": True,
                "messages": [_anthropic_message(m) for m in messages if m.get("role") != "system"],
            }
            if system:
                payload["system"] = [{"type": "text", "text": system, "cache_control": _CACHE}]
            if tools:
                payload["tools"] = [_anthropic_tool(t) for t in tools]
                payload["tools"][-1]["cache_control"] = _CACHE
            _mark_last_message(payload["messages"])
            if stop:
                payload["stop_sequences"] = list(stop)
            return f"{self.base_url}/messages", payload

        if self.dialect == "gemini":
            contents = [_gemini_message(m) for m in messages if m.get("role") != "system"]
            payload = {
                "contents": contents,
                "generationConfig": {"temperature": temperature},
            }
            if max_tokens:
                payload["generationConfig"]["maxOutputTokens"] = max_tokens
            system = "\n\n".join(str(m.get("content") or "") for m in messages
                                 if m.get("role") == "system")
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            if tools:
                payload["tools"] = [{"functionDeclarations": [_gemini_tool(t) for t in tools]}]
            if stop:
                payload["generationConfig"]["stopSequences"] = list(stop)
            return f"{self.base_url}/models/{model}:streamGenerateContent?alt=sse", payload

        payload = {
            "model": model,
            "messages": [_openai_message(m) for m in messages],
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = list(stop)
        if self.service_tier != "standard":
            # Unset is standard everywhere; sending "standard" would only give a
            # model that knows no tiers a reason to refuse the request.
            payload["service_tier"] = self.service_tier
        return f"{self.base_url}/chat/completions", payload


# --------------------------------------------------------------------------- #
# Wire formats
# --------------------------------------------------------------------------- #


def _model_extra(provider: str, row: dict[str, Any]) -> dict[str, Any]:
    """The nested block some gateways keep the real metadata in.

    A plain OpenAI ``/v1/models`` row says almost nothing — id, object, owner.
    Gateways that resell many models have to say more, and they do it in a
    sub-object named after themselves: unlid.ai nests the window, the output cap,
    availability and tool support under ``"unlid"``, OpenRouter-shaped ones under
    ``"top_provider"``.  Read flat first, then here — never the other way round,
    so a provider that grows a flat field later simply wins.
    """
    for name in (provider, "top_provider", "metadata"):
        block = row.get(name)
        if isinstance(block, dict):
            return block
    return {}


#: What a gateway calls a model that does not write text (DeepInfra's
#: ``metadata.tags``): such a model answers a chat request with an error, so it
#: has no place in a model picker for the chat or the agent.
_NOT_TEXT_TAGS = frozenset({"embed", "embedding", "embeddings", "image-gen", "text-to-image",
                            "tts", "text-to-speech", "stt", "speech-to-text", "asr",
                            "video-gen", "text-to-video", "rerank", "reranker", "moderation"})
_TEXT_TAGS = frozenset({"chat", "text-generation", "llm", "completion"})
#: The same, by name, for providers whose list carries no tags at all (OpenAI).
_NOT_TEXT_NAME = re.compile(
    r"(^|[/_.-])(text-embedding|embedding|embed|bge|e5|gte|dall-e|gpt-image|imagen|whisper|"
    r"tts|transcribe|realtime|omni-moderation|text-moderation|sora|veo|davinci-002|babbage-002)"
    r"([/_.-]|$|\d)", re.IGNORECASE)


def is_text_model(model_id: str, row: dict[str, Any], extra: dict[str, Any]) -> bool:
    """Whether a listed model can answer a chat request."""
    tags = {str(t).strip().lower() for t in (extra.get("tags") or row.get("tags") or [])
            if isinstance(t, str)}
    if tags:
        return bool(tags & _TEXT_TAGS) or not (tags & _NOT_TEXT_TAGS)
    return not _NOT_TEXT_NAME.search(model_id)


def _is_available(extra: dict[str, Any]) -> bool:
    """False only when the gateway says the model is explicitly not usable.

    Offering a model that answers 404 is worse than hiding it, but an unknown
    status means "no opinion", not "broken".
    """
    if extra.get("available") is False:
        return False
    status = str(extra.get("status") or "").strip().lower()
    return status in ("", "live", "active", "available", "ready")


def _first_int(*values: Any) -> int | None:
    """The first value that is a usable positive integer."""
    for value in values:
        number = _int_or_none(value)
        if number:
            return number
    return None


def _parse_models(provider: str, data: dict[str, Any]) -> list[RemoteModel]:
    """A provider's model list → :class:`RemoteModel` rows."""
    out: list[RemoteModel] = []
    rows = data.get("data") or data.get("models") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or row.get("name") or "").strip()
        if not model_id:
            continue
        if provider == "gemini":
            model_id = model_id.removeprefix("models/")
            if "generateContent" not in (row.get("supportedGenerationMethods") or
                                         ["generateContent"]):
                continue
        extra = _model_extra(provider, row)
        if not _is_available(extra):
            continue
        if not is_text_model(model_id, row, extra):
            continue
        supports_tools = row.get("supports_tools")
        if supports_tools is None:
            supports_tools = extra.get("supports_tools")
        out.append(RemoteModel(
            provider=provider,
            model_id=model_id,
            display_name=str(row.get("display_name") or row.get("displayName") or model_id),
            context_length=_first_int(row.get("context_length"),
                                      row.get("inputTokenLimit"),
                                      row.get("max_context_length"),
                                      row.get("context_window"),
                                      extra.get("context_length"),
                                      extra.get("context_window")),
            description=(str(row.get("description") or extra.get("description"))
                         if (row.get("description") or extra.get("description")) else None),
            owned_by=str(row.get("owned_by")) if row.get("owned_by") else None,
            supports_tools=bool(supports_tools) if supports_tools is not None else None,
            supports_vision=sees_images(provider, model_id, row, extra),
        ))
    out.sort(key=lambda m: m.model_id)
    return out


_VISION_NAME = re.compile(r"(gpt-4o|gpt-4\.1|gpt-5|chatgpt-4o|^o[134](-|$)|claude|gemini|vl\b|-vl-|vision|"
                          r"pixtral|llava|gemma-3|gemma-4|kimi-k2\.[5-9]|kimi-k3|qwen3\.[5-9]|qwen3-vl)",
                          re.IGNORECASE)


def sees_images(provider: str, model_id: str, row: dict[str, Any], extra: dict[str, Any]) -> bool:
    """Whether a listed model accepts images: the gateway's tags when it has them,
    the model's name otherwise (OpenAI, Anthropic and Gemini list no capabilities)."""
    tags = {str(t).strip().lower() for t in (extra.get("tags") or row.get("tags") or []) if isinstance(t, str)}
    if tags:
        return bool(tags & {"vision", "vlm", "image-to-text", "multimodal"})
    if provider in ("anthropic", "gemini"):
        return True
    return bool(_VISION_NAME.search(model_id))


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


#: Anthropic prompt caching. Unlike OpenAI's, it is opt-in: without these marks
#: every step of an agent turn paid full price for the whole conversation again —
#: a dozen steps a turn, each re-sending tools, system prompt and history. Three
#: breakpoints (of the four allowed): the tools and the system prompt, which do
#: not change during a session, and the last message, so the next step reads
#: everything up to it from the cache. A prefix below the provider's minimum is
#: simply not cached; the mark costs nothing then.
_CACHE = {"type": "ephemeral"}
#: Anthropic insists on a ``max_tokens``; this is what "no limit" asks for there
#: (every current Claude model can write at least this much in one answer).
ANTHROPIC_MAX_OUTPUT = 32_000


def _mark_last_message(messages: list[dict[str, Any]]) -> None:
    """Put a cache breakpoint on the last content block of the last message."""
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        if content:
            last["content"] = [{"type": "text", "text": content, "cache_control": _CACHE}]
    elif isinstance(content, list) and content:
        content[-1] = {**content[-1], "cache_control": _CACHE}


def _cache_key(value: str | None) -> str | None:
    """A ``prompt_cache_key``: letters, digits, ``-`` and ``_``, at most 64 characters."""
    if not value:
        return None
    clean = re.sub(r"[^A-Za-z0-9_-]", "-", str(value))[:64].strip("-")
    return clean or None


def _openai_message(message: dict[str, Any]) -> dict[str, Any]:
    """The agent keeps tool-call arguments as a dict (what local chat templates
    want); the OpenAI wire format wants them as a JSON string."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    wire = []
    for call in calls:
        function = dict(call.get("function") or {})
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            function["arguments"] = json.dumps(arguments or {}, ensure_ascii=False)
        wire.append({**call, "function": function})
    return {**message, "tool_calls": wire}


def _anthropic_message(message: dict[str, Any]) -> dict[str, Any]:
    role = "assistant" if message.get("role") == "assistant" else "user"
    if message.get("role") == "tool":
        return {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": str(message.get("tool_call_id") or ""),
            "content": str(message.get("content") or ""),
        }]}
    calls = message.get("tool_calls") or []
    if calls:
        blocks: list[dict[str, Any]] = []
        text = str(message.get("content") or "")
        if text:
            blocks.append({"type": "text", "text": text})
        for call in calls:
            function = call.get("function") or {}
            blocks.append({
                "type": "tool_use",
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "input": _as_object(function.get("arguments")),
            })
        return {"role": "assistant", "content": blocks}
    content = message.get("content")
    if isinstance(content, list):
        return {"role": role, "content": _anthropic_blocks(content)}
    return {"role": role, "content": str(content or "")}


def data_url(url: str) -> tuple[str, str] | None:
    """``data:image/png;base64,AAAA`` → ``("image/png", "AAAA")``."""
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        return None
    head, _, data = url.partition(";base64,")
    return head[5:] or "image/png", data


def _image_of(part: Any) -> tuple[str, str] | None:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
    return data_url(str(url or ""))


def _anthropic_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """OpenAI content parts → Anthropic blocks (text and base64 images)."""
    blocks: list[dict[str, Any]] = []
    for part in content:
        image = _image_of(part)
        if image is not None:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": image[0],
                                                       "data": image[1]}})
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
        elif isinstance(part, str):
            blocks.append({"type": "text", "text": part})
    return blocks


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    return {
        "name": function.get("name"),
        "description": function.get("description") or "",
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _gemini_message(message: dict[str, Any]) -> dict[str, Any]:
    role = "model" if message.get("role") == "assistant" else "user"
    if message.get("role") == "tool":
        return {"role": "user", "parts": [{"functionResponse": {
            "name": str(message.get("name") or ""),
            "response": {"content": str(message.get("content") or "")},
        }}]}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            image = _image_of(part)
            if image is not None:
                parts.append({"inline_data": {"mime_type": image[0], "data": image[1]}})
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append({"text": part["text"]})
        return {"role": role, "parts": parts or [{"text": ""}]}
    return {"role": role, "parts": [{"text": str(content or "")}]}


def _gemini_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    return {
        "name": function.get("name"),
        "description": function.get("description") or "",
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _retry_after(headers: Any) -> float | None:
    """``Retry-After`` in seconds (a number or an HTTP date), when the provider sent one."""
    try:
        raw = str((headers or {}).get("Retry-After") or "").strip()
    except Exception:  # noqa: BLE001 - an odd headers object is just "not sent"
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(raw)
        return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, IndexError):
        return None


async def _read_stream(dialect: str, response: Any) -> AsyncIterator[RemoteChunk]:
    """Server-sent events → :class:`RemoteChunk` objects."""
    calls: dict[int, dict[str, Any]] = {}
    prompt_tokens = completion_tokens = cached_tokens = 0
    finish: str | None = None

    async for raw in response.content:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except ValueError:
            continue

        if dialect == "anthropic":
            kind = event.get("type")
            if kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield RemoteChunk(text=str(delta.get("text") or ""))
                elif delta.get("type") == "input_json_delta":
                    index = int(event.get("index") or 0)
                    calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    calls[index]["arguments"] += str(delta.get("partial_json") or "")
            elif kind == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    index = int(event.get("index") or 0)
                    calls[index] = {"id": str(block.get("id") or ""),
                                    "name": str(block.get("name") or ""), "arguments": ""}
            elif kind == "message_delta":
                usage = event.get("usage") or {}
                completion_tokens = int(usage.get("output_tokens") or completion_tokens)
                finish = str((event.get("delta") or {}).get("stop_reason") or finish or "")
            elif kind == "message_start":
                usage = (event.get("message") or {}).get("usage") or {}
                # input_tokens counts only what was *not* cached; the prompt is
                # all three, and the read part is what the cache saved.
                cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
                prompt_tokens = (int(usage.get("input_tokens") or 0) + cached_tokens
                                 + int(usage.get("cache_creation_input_tokens") or 0))
            continue

        if dialect == "gemini":
            for candidate in event.get("candidates") or []:
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if part.get("text"):
                        yield RemoteChunk(text=str(part["text"]))
                    call = part.get("functionCall")
                    if call:
                        calls[len(calls)] = {"id": f"call_{len(calls)}",
                                             "name": str(call.get("name") or ""),
                                             "arguments": json.dumps(call.get("args") or {})}
                finish = candidate.get("finishReason") or finish
            usage = event.get("usageMetadata") or {}
            prompt_tokens = int(usage.get("promptTokenCount") or prompt_tokens)
            completion_tokens = int(usage.get("candidatesTokenCount") or completion_tokens)
            continue

        # OpenAI dialect (also DeepInfra)
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                yield RemoteChunk(text=str(delta["content"]))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index") or 0)
                slot = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = str(call["id"])
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = str(function["name"])
                if function.get("arguments"):
                    slot["arguments"] += str(function["arguments"])
            finish = choice.get("finish_reason") or finish
        usage = event.get("usage") or {}
        if usage:
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = int(details.get("cached_tokens") or cached_tokens)

    yield RemoteChunk(
        done=True,
        finish_reason=_normalise_finish(finish),
        tool_calls=[calls[i] for i in sorted(calls)],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )


def _normalise_finish(reason: str | None) -> str:
    value = (reason or "").lower()
    if value in ("length", "max_tokens", "max_output_tokens"):
        return "length"
    if value in ("tool_use", "tool_calls", "function_call"):
        return "stop"
    return "stop" if value in ("stop", "end_turn", "stop_sequence", "") else value


async def probe(provider: str, api_key: str, *, base_url: str | None = None,
                session_factory: Any = None) -> list[RemoteModel]:
    """Check a key by asking for the model list (what the "Sprawdź" button does)."""
    if provider not in PROVIDERS:
        raise RemoteError(f"Nieznany dostawca „{provider}”.", "unknown_provider", 400)
    if not (api_key or "").strip():
        raise RemoteError("Podaj klucz API.", "missing_key", 400)
    client = RemoteClient(provider, api_key.strip(), base_url=base_url,
                          session_factory=session_factory)
    try:
        return await asyncio.wait_for(client.models(refresh=True), timeout=HTTP_TIMEOUT_S)
    except asyncio.TimeoutError as exc:
        raise RemoteError(f"{PROVIDER_LABELS[provider]} nie odpowiedział na czas.",
                          "timeout", 504) from exc
