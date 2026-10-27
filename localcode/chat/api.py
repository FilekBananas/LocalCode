"""Framework-agnostic handlers for the ai_ux chat REST API.

Every route of ``ai_ux/src/app/api/**/route.ts`` (minus accounts/auth and
provider keys, which LocalCode does not have) is one ``async`` method of
:class:`ChatApi` taking plain Python arguments and returning a JSON-able value,
a :class:`FileResult` (downloads) or an async iterator of SSE frames
(``/api/chat``). Failures raise :class:`ApiError` ``(status, message, code)``
whose ``to_dict()`` is the ai_ux error body ``{"error", "code"}`` with a
Polish message.

:data:`ROUTES` describes the HTTP mapping declaratively (method, path, handler,
success status, path/query parameters, body kind) so the server can register
everything mechanically. Query parameters are passed RAW (``request.query.get``,
``None`` when absent) because several endpoints distinguish "absent" from
"empty" exactly like the reference (e.g. ``folderId=`` = only chats without a
project).

Differences from ai_ux: single local user (``bootstrap.user`` is ``null``),
one provider ``local`` whose models are the chat-capable models of the local
library, ``costUsd`` always 0, SQLite storage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import quote

from localcode.chat.db import new_id
from localcode.chat.extract import classify_file, display_name, extract_text, resolve_mime, safe_extension
from localcode.chat.rag import RagIndex
from localcode.chat.repo import (
    DEFAULT_CONVERSATION_TITLE,
    DEFAULT_FOLDER_NAME,
    FOLDER_COLORS,
    LOCAL_PROVIDER,
    UNSET,
    ApiError,
    ChatRepo,
)
from localcode.chat.service import ChatService
from localcode.runtime.templates import EFFORT_CHOICES, normalize_effort

log = logging.getLogger(__name__)

__all__ = ["ApiError", "ChatApi", "FileResult", "Route", "ROUTES", "UploadFile",
           "MAX_FILE_BYTES", "MAX_FILES_PER_REQUEST", "MAX_REQUEST_BYTES"]

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_REQUEST = 10
#: Whole-request budget the server should enforce from Content-Length before buffering.
MAX_REQUEST_BYTES = MAX_FILE_BYTES * MAX_FILES_PER_REQUEST + 1024 * 1024
SEARCH_LIMIT = 50

#: ``(file name, MIME type reported by the client, bytes)``
UploadFile = tuple[str, str, bytes]


# --------------------------------------------------------------------------- #
# Download payload
# --------------------------------------------------------------------------- #


@dataclass
class FileResult:
    """A file response: either in-memory ``body`` or a ``path`` on disk."""

    content_type: str
    filename: str
    disposition: str = "attachment"          # "attachment" | "inline"
    body: bytes | None = None
    path: Path | None = None
    size: int | None = None
    cache_control: str = "no-store"
    extra_headers: dict[str, str] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        """HTTP headers (Content-Type, Content-Disposition, Cache-Control, nosniff)."""
        headers = {
            "Content-Type": self.content_type,
            "Content-Disposition": content_disposition(self.filename, inline=self.disposition == "inline"),
            "Cache-Control": self.cache_control,
            "X-Content-Type-Options": "nosniff",
        }
        size = self.size if self.size is not None else (len(self.body) if self.body is not None else None)
        if size is not None:
            headers["Content-Length"] = str(size)
        headers.update(self.extra_headers)
        return headers


def content_disposition(filename: str, *, inline: bool = False) -> str:
    """RFC 5987 header with an ASCII fallback name."""
    ascii_name = re.sub(r'["\\]', "_", re.sub(r"[^\x20-\x7e]", "_", filename)) or "plik"
    kind = "inline" if inline else "attachment"
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


# --------------------------------------------------------------------------- #
# Route table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    handler: str
    status: int = 200
    #: json → JSON value; sse → async iterator of frame dicts; file → FileResult;
    kind: str = "json"
    #: ``(path parameter, handler kwarg)``
    path_params: tuple[tuple[str, str], ...] = ()
    #: ``(query parameter, handler kwarg)`` — passed raw, ``None`` when absent
    query: tuple[tuple[str, str], ...] = ()
    #: "json" → parsed JSON body as ``body``; "multipart" → ``files`` + form fields
    body: str | None = None
    #: multipart form fields ``(field, handler kwarg)``
    form: tuple[tuple[str, str], ...] = ()


_ID = (("id", "id"),)

ROUTES: tuple[Route, ...] = (
    Route("GET", "/api/bootstrap", "bootstrap"),
    Route("GET", "/api/conversations", "list_conversations",
          query=(("folderId", "folder_id"), ("archived", "archived"), ("limit", "limit"))),
    Route("POST", "/api/conversations", "create_conversation", status=201, body="json"),
    Route("GET", "/api/conversations/{id}", "get_conversation", path_params=_ID),
    Route("PATCH", "/api/conversations/{id}", "update_conversation", path_params=_ID, body="json"),
    Route("DELETE", "/api/conversations/{id}", "delete_conversation", path_params=_ID),
    Route("GET", "/api/conversations/{id}/export", "export_conversation", kind="file", path_params=_ID,
          query=(("format", "format"),)),
    Route("POST", "/api/conversations/{id}/title", "generate_title", path_params=_ID, query=(("auto", "auto"),)),
    Route("POST", "/api/conversations/{id}/compact", "compact_conversation", path_params=_ID),
    Route("DELETE", "/api/messages/{id}", "delete_message", path_params=_ID),
    Route("GET", "/api/folders", "list_folders"),
    Route("POST", "/api/folders", "create_folder", status=201, body="json"),
    Route("PATCH", "/api/folders/{id}", "update_folder", path_params=_ID, body="json"),
    Route("DELETE", "/api/folders/{id}", "delete_folder", path_params=_ID, query=(("deleteChats", "delete_chats"),)),
    Route("GET", "/api/memories", "list_memories", query=(("scope", "scope"), ("folderId", "folder_id"))),
    Route("POST", "/api/memories", "create_memory", status=201, body="json"),
    Route("PATCH", "/api/memories/{id}", "update_memory", path_params=_ID, body="json"),
    Route("DELETE", "/api/memories/{id}", "delete_memory", path_params=_ID),
    Route("GET", "/api/knowledge", "list_knowledge", query=(("scope", "scope"), ("folderId", "folder_id"))),
    Route("POST", "/api/knowledge", "upload_knowledge", status=201, body="multipart",
          form=(("scope", "scope"), ("folderId", "folder_id"))),
    Route("DELETE", "/api/knowledge/{id}", "delete_knowledge", path_params=_ID),
    Route("POST", "/api/upload", "upload_attachments", status=201, body="multipart",
          form=(("conversationId", "conversation_id"),)),
    Route("GET", "/api/attachments/{id}", "get_attachment", kind="file", path_params=_ID),
    Route("GET", "/api/settings", "get_settings"),
    Route("PATCH", "/api/settings", "patch_settings", body="json"),
    Route("GET", "/api/providers", "list_providers"),
    Route("GET", "/api/models", "list_models", query=(("refresh", "refresh"),)),
    Route("GET", "/api/search", "search", query=(("q", "q"),)),
    Route("POST", "/api/chat", "chat", kind="sse", body="json"),
)


# --------------------------------------------------------------------------- #
# Validation (ports of the reference zod schemas)
# --------------------------------------------------------------------------- #


class _Body:
    """Collects typed fields from a JSON body; unknown keys are ignored (zod strips them).
    Only keys present in the body end up in :attr:`out` (absent ≠ null)."""

    def __init__(self, body: Any) -> None:
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ApiError(400, "Treść żądania nie jest poprawnym JSON-em.", "invalid_json")
        self.body = body
        self.out: dict[str, Any] = {}
        self.issues: list[dict[str, str]] = []

    def _fail(self, key: str, message: str) -> None:
        self.issues.append({"path": key, "message": message})

    def string(self, key: str, *, max_len: int, min_len: int = 0, nullable: bool = False,
               target: dict[str, Any] | None = None, prefix: str = "") -> None:
        src = target if target is not None else self.body
        if key not in src:
            return
        value = src[key]
        path = prefix + key
        if value is None:
            if nullable:
                self._put(key, None, prefix)
            else:
                self._fail(path, "wartość jest wymagana")
            return
        if not isinstance(value, str):
            self._fail(path, "musi być tekstem")
            return
        if len(value) < min_len:
            self._fail(path, "nie może być puste" if min_len == 1 else f"musi mieć co najmniej {min_len} znaków")
            return
        if len(value) > max_len:
            self._fail(path, f"może mieć najwyżej {max_len} znaków")
            return
        self._put(key, value, prefix)

    def number(self, key: str, *, lo: float, hi: float, integer: bool = False, nullable: bool = False,
               target: dict[str, Any] | None = None, prefix: str = "") -> None:
        src = target if target is not None else self.body
        if key not in src:
            return
        value = src[key]
        path = prefix + key
        if value is None:
            if nullable:
                self._put(key, None, prefix)
            else:
                self._fail(path, "wartość jest wymagana")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._fail(path, "musi być liczbą")
            return
        if integer and int(value) != value:
            self._fail(path, "musi być liczbą całkowitą")
            return
        if not lo <= value <= hi:
            self._fail(path, f"musi mieścić się w zakresie {lo:g}–{hi:g}")
            return
        self._put(key, int(value) if integer else float(value), prefix)

    def boolean(self, key: str, *, target: dict[str, Any] | None = None, prefix: str = "") -> None:
        src = target if target is not None else self.body
        if key not in src:
            return
        value = src[key]
        if not isinstance(value, bool):
            self._fail(prefix + key, "musi być wartością logiczną (true/false)")
            return
        self._put(key, value, prefix)

    def enum(self, key: str, choices: tuple[str, ...], *, target: dict[str, Any] | None = None,
             prefix: str = "", nullable: bool = False) -> None:
        src = target if target is not None else self.body
        if key not in src:
            return
        value = src[key]
        if value is None and nullable:
            self._put(key, None, prefix)
            return
        if value not in choices:
            self._fail(prefix + key, "musi być jedną z wartości: " + ", ".join(choices))
            return
        self._put(key, value, prefix)

    def _put(self, key: str, value: Any, prefix: str) -> None:
        if prefix:
            section = prefix.rstrip(".")
            self.out.setdefault(section, {})[key] = value
        else:
            self.out[key] = value

    def done(self) -> dict[str, Any]:
        if self.issues:
            text = "; ".join(f"{i['path']} – {i['message']}" for i in self.issues[:5])
            raise ApiError(400, f"Nieprawidłowe dane wejściowe: {text}.", "validation_error", details=self.issues)
        return self.out


def _conversation_input(body: Any) -> dict[str, Any]:
    v = _Body(body)
    v.string("folderId", max_len=200, min_len=1, nullable=True)
    v.string("title", max_len=200)
    v.string("provider", max_len=64, nullable=True)
    v.string("model", max_len=500, nullable=True)
    v.string("systemPrompt", max_len=20_000, nullable=True)
    v.number("temperature", lo=0, hi=2, nullable=True)
    v.number("maxTokens", lo=0, hi=1_000_000, integer=True, nullable=True)
    v.boolean("pinned")
    v.boolean("archived")
    # null = follow settings.chat.reasoning_effort; models without an effort
    # control ignore the value entirely. A model with a plain on/off switch
    # advertises effortLevels ["off", "on"], so "on" is accepted here as well and
    # normalised to "medium" — the same mapping the agent WebSocket uses.
    _effort_field(v, "reasoningEffort")
    return v.done()



def _effort_field(v: "_Body", key: str) -> None:
    """Validate a reasoning-effort field, accepting the on/off spelling too."""
    if key not in v.body:
        return
    value = v.body[key]
    if value is None:
        v._put(key, None, "")
        return
    level = normalize_effort(value)
    if level is None:
        v._fail(key, "musi być jedną z wartości: " + ", ".join(EFFORT_CHOICES + ("on",)))
        return
    v._put(key, level, "")


def _folder_input(body: Any) -> dict[str, Any]:
    v = _Body(body)
    v.string("name", max_len=120)
    v.string("description", max_len=2_000, nullable=True)
    v.string("color", max_len=32, nullable=True)
    v.string("icon", max_len=64, nullable=True)
    v.string("systemPrompt", max_len=20_000, nullable=True)
    v.string("extraInfo", max_len=20_000, nullable=True)
    v.enum("memoryMode", ("shared", "isolated"))
    v.number("sortOrder", lo=0, hi=100_000, integer=True)
    return v.done()


def _memory_input(body: Any, *, create: bool) -> dict[str, Any]:
    v = _Body(body)
    v.enum("scope", ("global", "folder"))
    v.string("folderId", max_len=200, min_len=1, nullable=True)
    v.enum("kind", ("fact", "preference", "note"))
    if create:
        if not isinstance(v.body.get("content"), str):
            v._fail("content", "wartość jest wymagana")
        else:
            v.string("content", max_len=20_000, min_len=1)
    else:
        v.string("content", max_len=20_000)
    v.enum("source", ("user", "auto"))
    v.boolean("pinned")
    return v.done()


def _settings_patch(body: Any) -> dict[str, Any]:
    v = _Body(body)
    v.string("globalSystemPrompt", max_len=100_000)
    v.boolean("memoryEnabled")
    v.enum("theme", ("light", "dark", "system"))
    profile = v.body.get("profile")
    if profile is not None:
        if not isinstance(profile, dict):
            v._fail("profile", "musi być obiektem")
        else:
            v.string("displayName", max_len=200, target=profile, prefix="profile.")
            v.string("about", max_len=20_000, target=profile, prefix="profile.")
            v.string("role", max_len=500, target=profile, prefix="profile.")
            v.string("responseStyle", max_len=20_000, target=profile, prefix="profile.")
            v.string("customInstructions", max_len=20_000, target=profile, prefix="profile.")
            v.boolean("enabled", target=profile, prefix="profile.")
    defaults = v.body.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            v._fail("defaults", "musi być obiektem")
        else:
            v.string("provider", max_len=64, min_len=1, target=defaults, prefix="defaults.")
            # LocalCode: an empty model means "choose automatically".
            v.string("model", max_len=500, target=defaults, prefix="defaults.")
            v.number("temperature", lo=0, hi=2, target=defaults, prefix="defaults.")
            v.number("maxTokens", lo=0, hi=1_000_000, integer=True, target=defaults, prefix="defaults.")
            v.number("ragTopK", lo=0, hi=50, integer=True, target=defaults, prefix="defaults.")
            v.boolean("autoTitle", target=defaults, prefix="defaults.")
            v.boolean("showReasoning", target=defaults, prefix="defaults.")
            v.number("historyLimit", lo=0, hi=1_000, integer=True, target=defaults, prefix="defaults.")
    return v.done()


def parse_bool(value: str | None) -> bool | None:
    """Query-string boolean: 1/true/yes/on → True, 0/false/no/off → False, else None."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def parse_limit(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value.strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def read_folder_id(raw: str | None) -> Any:
    """``folderId`` absent → UNSET (any), empty/"null"/"none" → None, else the id."""
    if raw is None:
        return UNSET
    trimmed = raw.strip()
    if not trimmed or trimmed in ("null", "none"):
        return None
    return trimmed


def _read_scope(raw: str | None) -> str | None:
    scope = (raw or "").strip()
    if not scope:
        return None
    if scope not in ("global", "folder"):
        raise ApiError(400, "Zakres musi być równy „global” albo „folder”.", "invalid_scope")
    return scope


# --------------------------------------------------------------------------- #
# Formatting helpers (export)
# --------------------------------------------------------------------------- #

_PL_MONTHS = ("stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia",
              "września", "października", "listopada", "grudnia")
_PL_DIACRITICS = str.maketrans({"ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n", "ó": "o", "ś": "s",
                                "ź": "z", "ż": "z"})


def format_datetime_pl(ms: int | float) -> str:
    d = datetime.fromtimestamp(float(ms) / 1000)
    return f"{d.day} {_PL_MONTHS[d.month - 1]} {d.year} {d.hour:02d}:{d.minute:02d}"


def format_bytes(size: int) -> str:
    """Reference ``formatBytes``."""
    if not size:
        return "0 B"
    units = ("B", "KB", "MB", "GB")
    i = 0
    value = float(size)
    while value >= 1024 and i < len(units) - 1:
        value /= 1024
        i += 1
    return f"{value:.0f} {units[i]}" if i == 0 else f"{value:.1f} {units[i]}"


def format_cost(usd: float | None) -> str:
    if usd is None:
        return "—"
    if usd == 0:
        return "$0"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.3f}"


def slugify(text: str, fallback: str = "plik") -> str:
    """ASCII-safe, filesystem-safe slug for downloadable file names."""
    slug = (text or "").lower().translate(_PL_DIACRITICS)
    slug = unicodedata.normalize("NFD", slug)
    slug = "".join(ch for ch in slug if not unicodedata.combining(ch))
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")[:60].rstrip("-")
    return slug or fallback


_ROLE_HEADINGS = {"user": "## Ty", "assistant": "## Asystent", "system": "## System"}
_STATUS_NOTES = {"stopped": "_(odpowiedź przerwana przez użytkownika)_",
                 "streaming": "_(odpowiedź niedokończona)_"}


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.split("\n"))


def render_markdown(conversation: dict[str, Any], folder: dict[str, Any] | None,
                    messages: list[dict[str, Any]]) -> str:
    """Markdown export (port of the reference export route)."""
    model = f"{conversation.get('provider') or '?'} / {conversation['model']}" if conversation.get("model") else "—"
    if folder:
        mode = "izolowana" if folder.get("memoryMode") == "isolated" else "wspólna"
        folder_label = f"{folder['name']} (pamięć: {mode})"
    else:
        folder_label = "brak — rozmowa globalna"
    header = [
        f"- **Data utworzenia:** {format_datetime_pl(conversation['createdAt'])}",
        f"- **Ostatnia aktywność:** {format_datetime_pl(conversation['lastMessageAt'])}",
        f"- **Model:** {model}",
        f"- **Projekt:** {folder_label}",
        f"- **Liczba wiadomości:** {len(messages)}",
    ]
    if conversation.get("tokensIn", 0) > 0 or conversation.get("tokensOut", 0) > 0:
        header.append(f"- **Tokeny:** {conversation['tokensIn']} wejściowych / {conversation['tokensOut']} wyjściowych")
    if (conversation.get("costUsd") or 0) > 0:
        header.append(f"- **Koszt:** {format_cost(conversation['costUsd'])}")
    if (conversation.get("systemPrompt") or "").strip():
        header.append(f"- **Instrukcje rozmowy:** {' '.join(conversation['systemPrompt'].split())}")

    sections = [f"# {conversation['title']}", "\n".join(header), "---"]
    for message in messages:
        parts = [_ROLE_HEADINGS.get(message["role"], "## System"), f"_{format_datetime_pl(message['createdAt'])}_"]
        if message["attachments"]:
            listing = ", ".join(f"{a['name']} ({format_bytes(a['size'])})" for a in message["attachments"])
            parts.append(f"**Załączniki:** {listing}")
        if (message.get("reasoning") or "").strip():
            parts.append(f"**Rozumowanie:**\n\n{_quote_block(message['reasoning'].strip())}")
        content = (message.get("content") or "").strip()
        if content:
            parts.append(content)
        note = _STATUS_NOTES.get(message["status"])
        if note:
            parts.append(note)
        if message["status"] == "error" and message.get("error"):
            parts.append(f"**Błąd:** {message['error']}")
        if message["citations"]:
            sources = "\n".join(f"{i + 1}. {c['documentName']} (fragment {c['chunkIndex']})"
                                for i, c in enumerate(message["citations"]))
            parts.append(f"**Źródła:**\n\n{sources}")
        sections.append("\n\n".join(parts))
    return "\n\n".join(sections) + "\n"


# --------------------------------------------------------------------------- #
# Attachment content type safety (reference attachments route)
# --------------------------------------------------------------------------- #

_MIME_PATTERN = re.compile(r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$")
_INLINE_MIMES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/avif", "image/bmp", "image/x-icon",
    "image/vnd.microsoft.icon", "image/heic", "image/heif", "image/tiff", "application/pdf", "text/plain",
})
_NEUTRALIZED_MIMES = frozenset({
    "text/html", "application/xhtml+xml", "image/svg+xml", "text/xml", "application/xml",
    "text/javascript", "application/javascript",
})


def safe_content_type(mime: str) -> str:
    value = (mime or "").split(";")[0].strip().lower()
    if not value or not _MIME_PATTERN.match(value) or value in _NEUTRALIZED_MIMES:
        return "application/octet-stream"
    return value


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


class ChatApi:
    """All chat endpoints. ``service`` provides models and streaming."""

    PROVIDER_INFO: dict[str, Any] = {
        "provider": LOCAL_PROVIDER,
        "hasKey": False,
        "maskedKey": None,
        "baseUrl": None,
        "updatedAt": None,
        "label": "Modele lokalne",
        "description": "Modele GGUF i MLX pobrane z Hugging Face i uruchamiane na tym Macu.",
        "requiresKey": False,
        "keyHint": None,
        "defaultBaseUrl": None,
    }

    def __init__(self, repo: ChatRepo, service: ChatService, *, rag: RagIndex | None = None) -> None:
        self.repo = repo
        self.service = service
        self.rag = rag or service.rag

    async def _r(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def startup(self) -> None:
        """Housekeeping on server start: drop uploads never attached to a message (> 24 h)."""
        try:
            removed = await self._r(self.repo.purge_stale_uploads)
            if removed:
                log.info("removed %d stale pending uploads", removed)
        except Exception:
            log.warning("purging stale uploads failed", exc_info=True)

    # ------------------------------------------------------------- bootstrap
    async def bootstrap(self) -> dict[str, Any]:
        settings = await self.get_settings()
        folders, conversations = await asyncio.gather(
            self._r(self.repo.list_folders_with_stats),
            self._r(self.repo.list_conversations),
        )
        try:
            models = await self.service.model_infos()
        except Exception:
            log.warning("listing local models failed", exc_info=True)
            models = []
        return {
            "settings": settings,
            "folders": folders,
            "conversations": conversations,
            "providers": [dict(self.PROVIDER_INFO)],
            "models": models,
            "activeProvider": LOCAL_PROVIDER,
            "user": None,
        }

    # --------------------------------------------------------- conversations
    async def list_conversations(self, folder_id: str | None = None, archived: str | None = None,
                                 limit: str | None = None) -> list[dict[str, Any]]:
        folder = read_folder_id(folder_id)
        if archived is None or not archived.strip():
            archived_flag: Any = UNSET
        elif archived.strip() in ("all", "*"):
            archived_flag = None
        else:
            archived_flag = archived.strip() == "1" or archived.strip().lower() == "true"
        return await self._r(self.repo.list_conversations, folder_id=folder, archived=archived_flag,
                             limit=parse_limit(limit))

    async def create_conversation(self, body: Any = None) -> dict[str, Any]:
        data = _conversation_input(body)
        if isinstance(data.get("folderId"), str) and not await self._r(self.repo.get_folder, data["folderId"]):
            raise ApiError(400, "Wskazany projekt nie istnieje.", "folder_not_found")
        settings = await self._r(self.repo.get_settings)
        payload = {
            **data,
            "title": (data.get("title") or "").strip() or DEFAULT_CONVERSATION_TITLE,
            "provider": LOCAL_PROVIDER,
            "model": data.get("model") if data.get("model") is not None else (settings["defaults"]["model"] or None),
        }
        return await self._r(self.repo.create_conversation, payload)

    async def get_conversation(self, id: str) -> dict[str, Any]:
        conversation = await self._r(self.repo.get_conversation, id)
        if conversation is None:
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        return {"conversation": conversation, "messages": await self._r(self.repo.list_messages, id)}

    async def update_conversation(self, id: str, body: Any = None) -> dict[str, Any]:
        data = _conversation_input(body)
        if isinstance(data.get("folderId"), str) and not await self._r(self.repo.get_folder, data["folderId"]):
            raise ApiError(400, "Wskazany projekt nie istnieje.", "folder_not_found")
        if "provider" in data and data["provider"] is not None:
            data["provider"] = LOCAL_PROVIDER
        if "title" in data:
            data["titleLocked"] = True   # a manual rename pins the title
        updated = await self._r(self.repo.update_conversation, id, data)
        if updated is None:
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        return updated

    async def delete_conversation(self, id: str) -> dict[str, bool]:
        if not await self._r(self.repo.delete_conversation, id):
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        return {"ok": True}

    async def export_conversation(self, id: str, format: str | None = None) -> FileResult:
        raw = (format or "md").strip().lower()
        fmt = "md" if raw == "markdown" else raw
        if fmt not in ("md", "json"):
            raise ApiError(400, "Nieobsługiwany format eksportu. Dozwolone: md, json.", "bad_format")
        conversation = await self._r(self.repo.get_conversation, id)
        if conversation is None:
            raise ApiError(404, "Nie znaleziono rozmowy.", "not_found")
        messages = await self._r(self.repo.list_messages, id)
        folder = await self._r(self.repo.get_folder, conversation["folderId"]) if conversation["folderId"] else None
        base = slugify(conversation["title"], "rozmowa")
        if fmt == "json":
            body = json.dumps({"conversation": conversation, "messages": messages}, ensure_ascii=False, indent=2) + "\n"
            return FileResult("application/json; charset=utf-8", f"{base}.json", body=body.encode("utf-8"))
        return FileResult("text/markdown; charset=utf-8", f"{base}.md",
                          body=render_markdown(conversation, folder, messages).encode("utf-8"))

    async def generate_title(self, id: str, auto: str | None = None) -> dict[str, str]:
        return await self.service.generate_title(id, auto=parse_bool(auto) is True)

    async def compact_conversation(self, id: str) -> dict[str, Any]:
        """``POST /api/conversations/{id}/compact`` — summarise the older turns now.

        Answers ``{"summary", "before", "after", "changed", "strategy", "messages"}``;
        ``before``/``after`` are the prompt sizes in tokens, the same numbers the
        ``compaction`` SSE frame carries during a turn.
        """
        return await self.service.compact_conversation(id)

    async def delete_message(self, id: str) -> dict[str, bool]:
        if not await self._r(self.repo.delete_message, id):
            raise ApiError(404, "Nie znaleziono wiadomości.", "not_found")
        return {"ok": True}

    # --------------------------------------------------------------- folders
    async def list_folders(self) -> list[dict[str, Any]]:
        return await self._r(self.repo.list_folders_with_stats)

    async def create_folder(self, body: Any = None) -> dict[str, Any]:
        data = _folder_input(body)
        payload = {
            **data,
            "name": (data.get("name") or "").strip() or DEFAULT_FOLDER_NAME,
            "color": data["color"] if data.get("color") is not None else random.choice(FOLDER_COLORS),
            "memoryMode": data.get("memoryMode") or "shared",
        }
        return await self._r(self.repo.create_folder, payload)

    async def update_folder(self, id: str, body: Any = None) -> dict[str, Any]:
        updated = await self._r(self.repo.update_folder, id, _folder_input(body))
        if updated is None:
            raise ApiError(404, "Nie znaleziono projektu.", "not_found")
        return updated

    async def delete_folder(self, id: str, delete_chats: str | None = None) -> dict[str, bool]:
        if not await self._r(self.repo.delete_folder, id, parse_bool(delete_chats) is True):
            raise ApiError(404, "Nie znaleziono projektu.", "not_found")
        return {"ok": True}

    # -------------------------------------------------------------- memories
    async def list_memories(self, scope: str | None = None, folder_id: str | None = None) -> list[dict[str, Any]]:
        parsed_scope = _read_scope(scope)
        folder = None if parsed_scope == "global" else read_folder_id(folder_id)
        return await self._r(self.repo.list_memories, scope=parsed_scope, folder_id=folder)

    async def _check_memory_scope(self, scope: str, folder_id: str | None, patch_folder: Any) -> None:
        if scope == "global":
            if patch_folder:
                raise ApiError(400, "Wpis globalny nie może być przypisany do projektu.", "invalid_scope")
            return
        if not folder_id:
            raise ApiError(400, "Wpis w zakresie projektu wymaga wskazania projektu.", "folder_required")
        if not await self._r(self.repo.get_folder, folder_id):
            raise ApiError(404, "Wskazany projekt nie istnieje.", "folder_not_found")

    async def create_memory(self, body: Any = None) -> dict[str, Any]:
        data = _memory_input(body, create=True)
        content = data["content"].strip()
        if not content:
            raise ApiError(400, "Treść wpisu nie może być pusta.", "empty_content")
        scope = data.get("scope") or ("folder" if data.get("folderId") else "global")
        await self._check_memory_scope(scope, data.get("folderId"), data.get("folderId"))
        return await self._r(self.repo.create_memory, {
            "scope": scope,
            "folderId": data.get("folderId") if scope == "folder" else None,
            "kind": data.get("kind"),
            "content": content,
            "source": data.get("source"),
            "pinned": data.get("pinned"),
        })

    async def update_memory(self, id: str, body: Any = None) -> dict[str, Any]:
        existing = await self._r(self.repo.get_memory, id)
        if existing is None:
            raise ApiError(404, "Nie znaleziono wpisu pamięci.", "not_found")
        data = _memory_input(body, create=False)
        content = None
        if "content" in data:
            content = data["content"].strip()
            if not content:
                raise ApiError(400, "Treść wpisu nie może być pusta.", "empty_content")
        scope = data.get("scope") or existing["scope"]
        folder_id = data["folderId"] if "folderId" in data else existing["folderId"]
        await self._check_memory_scope(scope, folder_id, data.get("folderId"))
        patch: dict[str, Any] = {"scope": scope, "folderId": folder_id if scope == "folder" else None}
        for key in ("kind", "source", "pinned"):
            if key in data:
                patch[key] = data[key]
        if content is not None:
            patch["content"] = content
        updated = await self._r(self.repo.update_memory, id, patch)
        if updated is None:
            raise ApiError(404, "Nie znaleziono wpisu pamięci.", "not_found")
        return updated

    async def delete_memory(self, id: str) -> dict[str, bool]:
        if not await self._r(self.repo.delete_memory, id):
            raise ApiError(404, "Nie znaleziono wpisu pamięci.", "not_found")
        return {"ok": True}

    # ------------------------------------------------------------- knowledge
    async def list_knowledge(self, scope: str | None = None, folder_id: str | None = None) -> list[dict[str, Any]]:
        parsed_scope = _read_scope(scope)
        folder = None if parsed_scope == "global" else read_folder_id(folder_id)
        return await self._r(self.repo.list_documents, scope=parsed_scope, folder_id=folder)

    @staticmethod
    def _check_files(files: list[UploadFile] | None) -> list[UploadFile]:
        usable = [f for f in (files or []) if f and (len(f[2] or b"") > 0 or (f[0] or "").strip())]
        if not usable:
            raise ApiError(400, "Nie przesłano żadnego pliku.", "no_files")
        if len(usable) > MAX_FILES_PER_REQUEST:
            raise ApiError(413, f"Jednorazowo można wysłać maksymalnie {MAX_FILES_PER_REQUEST} plików.",
                           "too_many_files")
        for name, _mime, data in usable:
            size = len(data or b"")
            if size > MAX_FILE_BYTES:
                raise ApiError(413, f"Plik „{name}” ma {format_bytes(size)} i przekracza limit "
                                    f"{format_bytes(MAX_FILE_BYTES)} na plik.", "file_too_large")
            if size == 0:
                raise ApiError(400, f"Plik „{name}” jest pusty.", "empty_file")
        return usable

    async def upload_knowledge(self, files: list[UploadFile] | None = None, scope: str | None = None,
                               folder_id: str | None = None) -> list[dict[str, Any]]:
        raw_scope = (scope or "").strip() or "global"
        if raw_scope not in ("global", "folder"):
            raise ApiError(400, "Zakres musi być równy „global” albo „folder”.", "invalid_scope")
        folder = (folder_id or "").strip() or None
        if raw_scope == "folder":
            if not folder:
                raise ApiError(400, "Dokument w zakresie projektu wymaga wskazania projektu.", "folder_required")
            if not await self._r(self.repo.get_folder, folder):
                raise ApiError(404, "Wskazany projekt nie istnieje.", "folder_not_found")
        usable = self._check_files(files)
        documents = []
        for name, mime, data in usable:
            documents.append(await self.rag.ingest(name=name, mime=mime, data=data, scope=raw_scope,
                                                   folder_id=folder if raw_scope == "folder" else None))
        return documents

    async def delete_knowledge(self, id: str) -> dict[str, bool]:
        if not await self._r(self.repo.delete_document, id):
            raise ApiError(404, "Nie znaleziono dokumentu.", "not_found")
        return {"ok": True}

    # ----------------------------------------------------------- attachments
    async def upload_attachments(self, files: list[UploadFile] | None = None,
                                 conversation_id: str | None = None) -> list[dict[str, Any]]:
        requested = (conversation_id or "").strip() or None
        # An unknown conversation degrades to a pending upload (reference behaviour).
        conv = requested if requested and await self._r(self.repo.get_conversation, requested) else None
        usable = self._check_files(files)
        created = []
        for raw_name, raw_mime, data in usable:
            name = display_name(raw_name)
            mime = resolve_mime(raw_mime, name)
            kind = classify_file(name, mime)
            att_id = new_id()
            extension = safe_extension(name)
            try:
                relative = await self._r(self.repo.store_upload, "att", att_id, extension, data)
            except OSError as exc:
                raise ApiError(500, f"Nie udało się zapisać pliku „{name}”: {exc.strerror or exc}",
                               "storage_error") from exc
            text_content = None
            if kind != "image":
                extracted = await asyncio.to_thread(extract_text, data, name, mime)
                text_content = extracted.text if extracted.text.strip() else None
            created.append(await self._r(self.repo.create_attachment, {
                "id": att_id, "messageId": None, "conversationId": conv, "name": name, "mime": mime,
                "size": len(data), "kind": kind, "path": relative, "textContent": text_content,
            }))
        return created

    async def get_attachment(self, id: str) -> FileResult:
        found = await self._r(self.repo.attachment_file, id)
        if found is None:
            raise ApiError(404, "Nie znaleziono załącznika.", "not_found")
        attachment, path = found
        if path is None:
            raise ApiError(404, "Załącznik nie ma zapisanego pliku.", "not_found")
        try:
            stat = await asyncio.to_thread(path.stat)
        except OSError:
            raise ApiError(404, "Plik załącznika nie istnieje już na dysku.", "not_found") from None
        content_type = safe_content_type(attachment["mime"])
        return FileResult(
            content_type=content_type,
            filename=attachment["name"],
            disposition="inline" if content_type in _INLINE_MIMES else "attachment",
            path=path,
            size=stat.st_size,
            cache_control="private, max-age=31536000, immutable",
        )

    # -------------------------------------------------------------- settings
    async def get_settings(self) -> dict[str, Any]:
        settings = await self._r(self.repo.get_settings)
        if not settings["defaults"].get("model"):
            try:
                key = await self.service.default_model_key(settings)
            except Exception:
                key = None
            if key:
                settings["defaults"]["model"] = key
        settings["defaults"]["provider"] = LOCAL_PROVIDER
        return settings

    async def patch_settings(self, body: Any = None) -> dict[str, Any]:
        patch = _settings_patch(body)
        if "defaults" in patch and "provider" in patch["defaults"]:
            patch["defaults"]["provider"] = LOCAL_PROVIDER
        await self._r(self.repo.patch_settings, patch)
        return await self.get_settings()

    # ------------------------------------------------------ providers/models
    async def list_providers(self) -> dict[str, Any]:
        return {"providers": [dict(self.PROVIDER_INFO)]}

    async def list_models(self, refresh: str | None = None) -> list[dict[str, Any]]:
        if parse_bool(refresh) is True and self.service.library is not None:
            try:
                await self.service._call_library("scan")
            except Exception as exc:
                raise ApiError(502, f"Nie udało się odświeżyć listy modeli: {exc}", "models_unavailable") from exc
        try:
            return await self.service.model_infos()
        except Exception as exc:
            raise ApiError(502, "Nie udało się pobrać listy modeli.", "models_unavailable") from exc

    # ---------------------------------------------------------------- search
    async def search(self, q: str | None = None) -> list[dict[str, Any]]:
        query = (q or "").strip()
        if not query:
            return []
        return await self._r(self.repo.search_messages, query, SEARCH_LIMIT)

    # ------------------------------------------------------------------ chat
    def chat(self, body: Any = None) -> AsyncIterator[dict[str, Any]]:
        """SSE stream of frame dicts. Pull the first frame before committing the
        response: an :class:`ApiError` raised then is an ordinary JSON error."""
        return self.service.send(body)
