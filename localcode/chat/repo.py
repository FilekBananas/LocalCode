"""Chat data-access layer (port of ai_ux ``src/lib/db/repo.ts``).

Every SQL statement of the chatbot lives here. Conventions, identical to the
reference so the SwiftUI client (``Models.swift``) decodes the output as is:

* rows are snake_case, everything returned is a plain ``dict`` in camelCase
  with millisecond timestamps, JSON booleans and ``None`` for SQL ``NULL``;
* "not found" is never an exception: reads return ``None``/``[]``, writes
  return ``None``/``False``;
* messages carry ``attachments`` and ``citations`` arrays; list endpoints add
  ``preview``/``messageCount``/``folderName`` (conversations) and
  ``conversationCount``/``documentCount``/``memoryCount`` (folders);
* file paths of uploads are an implementation detail and never leave this
  module (the reference also keeps them away from the client).

LocalCode differences: no accounts (single local user, so no ``user_id``
filters), SQLite instead of PostgreSQL (FTS5 instead of ``tsvector``), and the
repository also owns the upload files so that deleting a row never leaves an
orphaned file behind.

All methods are synchronous and thread-safe (they hold the :class:`ChatDB`
lock); async callers use ``asyncio.to_thread`` or :meth:`ChatRepo.run`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from localcode.chat.db import (
    SNIPPET_CLOSE,
    SNIPPET_ELLIPSIS,
    SNIPPET_OPEN,
    ChatDB,
    fold_text,
    new_id,
    now_ms,
    restore_snippet,
    uploads_dir,
)

log = logging.getLogger(__name__)

T = TypeVar("T")


class _Unset:
    """Sentinel: "parameter not given" (distinct from ``None`` = SQL NULL)."""

    _instance: "_Unset | None" = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Any = _Unset()


class ApiError(Exception):
    """Error for the HTTP layer: status code + Polish message + machine code.

    Shared by the whole chat package (repo, service, api) so every layer can
    raise it; the server turns it into ``{"error": message, "code": code}``.
    """

    def __init__(self, status: int, message: str, code: str = "error", details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": self.message, "code": self.code}
        if self.details is not None:
            body["details"] = self.details
        return body

    def __repr__(self) -> str:
        return f"ApiError({self.status}, {self.message!r}, {self.code!r})"


# --------------------------------------------------------------------------- #
# Constants and defaults
# --------------------------------------------------------------------------- #

DEFAULT_CONVERSATION_TITLE = "Nowa rozmowa"
DEFAULT_FOLDER_NAME = "Nowy projekt"
LOCAL_PROVIDER = "local"

FOLDER_COLORS = ("#c96442", "#4078f2", "#2f855a", "#805ad5", "#d69e2e", "#e53e3e", "#319795", "#718096")

#: ``AppSettings`` defaults (reference ``DEFAULT_SETTINGS`` with the local provider).
#: ``defaults.model`` empty = "pick automatically" (config default, else first local model).
DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "globalSystemPrompt": "",
    "memoryEnabled": True,
    "theme": "system",
    "profile": {
        "displayName": "",
        "about": "",
        "role": "",
        "responseStyle": "",
        "customInstructions": "",
        "enabled": True,
    },
    "defaults": {
        "provider": LOCAL_PROVIDER,
        "model": "",
        "temperature": 0.7,
        "maxTokens": 4096,
        "ragTopK": 6,
        "autoTitle": True,
        "showReasoning": True,
        "historyLimit": 0,
    },
}

_THEMES = ("light", "dark", "system")


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #


def _b(v: Any) -> bool:
    return v is True or v == 1 or v == "1" or v == "t"


def _int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _effort_or_none(v: Any) -> str | None:
    """A per-conversation reasoning effort, or ``None`` = follow the settings."""
    from localcode.runtime.templates import normalize_effort

    return normalize_effort(v) if v is not None else None


def _memory_mode(v: Any) -> str:
    return "isolated" if v == "isolated" else "shared"


def _role(v: Any) -> str:
    return v if v in ("user", "assistant", "system") else "user"


def _status(v: Any) -> str:
    return v if v in ("streaming", "error", "stopped") else "complete"


def _att_kind(v: Any) -> str:
    return v if v in ("image", "text", "pdf") else "other"


def _kb_scope(v: Any) -> str:
    return "global" if v == "global" else "folder"


def _kb_status(v: Any) -> str:
    return v if v in ("ready", "error") else "processing"


def _mem_scope(v: Any) -> str:
    return "folder" if v == "folder" else "global"


def _mem_kind(v: Any) -> str:
    return v if v in ("preference", "note") else "fact"


def _mem_source(v: Any) -> str:
    return "auto" if v == "auto" else "user"


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def truncate(text: str, limit: int) -> str:
    """Reference ``truncate``: cut to ``limit`` chars with a trailing ellipsis."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def to_preview(text: str | None) -> str | None:
    if not text:
        return None
    clean = collapse_ws(text)
    if not clean:
        return None
    return clean if len(clean) <= 120 else clean[:119].rstrip() + "…"


def estimate_tokens(text: str) -> int:
    """Rough token estimate used for budgeting and stored chunk sizes.

    Polish text costs more tokens per character than English (about 3 chars
    per token on common local tokenizers), so the estimate errs on that side.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3))


def derive_title(text: str) -> str:
    """Offline fallback title from the first user message."""
    clean = collapse_ws(text)
    if not clean:
        return DEFAULT_CONVERSATION_TITLE
    return truncate(clean, 48)


def parse_citations(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        chunk_index = item.get("chunkIndex")
        score = item.get("score")
        out.append({
            "documentId": item.get("documentId") if isinstance(item.get("documentId"), str) else "",
            "documentName": item.get("documentName") if isinstance(item.get("documentName"), str) else "",
            "chunkIndex": int(chunk_index) if isinstance(chunk_index, (int, float)) and not isinstance(chunk_index, bool) else 0,
            "snippet": item.get("snippet") if isinstance(item.get("snippet"), str) else "",
            "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0,
        })
    return out


def serialize_citations(citations: list[dict[str, Any]] | None) -> str | None:
    if not citations:
        return None
    return json.dumps(citations, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# FTS query building
# --------------------------------------------------------------------------- #

_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)


def build_fts_query(text: str, max_tokens: int = 20) -> str | None:
    """Turn a user's words into a safe FTS5 MATCH expression.

    Port of the reference ``ftsQuery``: every token of 2+ characters becomes a
    quoted prefix query, OR-ed together. Tokens only contain word characters,
    so no input can reach the FTS5 parser as syntax. ``None`` when nothing
    usable is left.
    """
    folded = fold_text((text or "").lower())
    tokens = [t for t in _TOKEN_SPLIT.split(folded) if len(t.replace("_", "")) > 1]
    if not tokens:
        return None
    seen: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
        if len(seen) >= max_tokens:
            break
    return " OR ".join(f'"{t}"*' for t in seen)


# --------------------------------------------------------------------------- #
# Settings coercion (reference ``coerceSettings``)
# --------------------------------------------------------------------------- #


def _as_record(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _as_str(v: Any, fallback: str) -> str:
    return v if isinstance(v, str) else fallback


def _as_bool(v: Any, fallback: bool) -> bool:
    return v if isinstance(v, bool) else fallback


def _as_num(v: Any, fallback: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return fallback
    return float(v) if math.isfinite(float(v)) else fallback


def _as_int(v: Any, fallback: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        return fallback
    return int(round(v))


def coerce_settings(raw: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deep-merge an arbitrary (possibly stale/partial) object over ``base``."""
    base = base or DEFAULT_APP_SETTINGS
    rec = _as_record(raw)
    profile = _as_record(rec.get("profile"))
    defaults = _as_record(rec.get("defaults"))
    bp = base["profile"]
    bd = base["defaults"]
    theme = rec.get("theme")
    return {
        "globalSystemPrompt": _as_str(rec.get("globalSystemPrompt"), base["globalSystemPrompt"]),
        "profile": {
            "displayName": _as_str(profile.get("displayName"), bp["displayName"]),
            "about": _as_str(profile.get("about"), bp["about"]),
            "role": _as_str(profile.get("role"), bp["role"]),
            "responseStyle": _as_str(profile.get("responseStyle"), bp["responseStyle"]),
            "customInstructions": _as_str(profile.get("customInstructions"), bp["customInstructions"]),
            "enabled": _as_bool(profile.get("enabled"), bp["enabled"]),
        },
        "defaults": {
            "provider": _as_str(defaults.get("provider"), bd["provider"]) or LOCAL_PROVIDER,
            "model": _as_str(defaults.get("model"), bd["model"]),
            "temperature": _as_num(defaults.get("temperature"), bd["temperature"]),
            "maxTokens": _as_int(defaults.get("maxTokens"), bd["maxTokens"]),
            "ragTopK": _as_int(defaults.get("ragTopK"), bd["ragTopK"]),
            "autoTitle": _as_bool(defaults.get("autoTitle"), bd["autoTitle"]),
            "showReasoning": _as_bool(defaults.get("showReasoning"), bd["showReasoning"]),
            "historyLimit": _as_int(defaults.get("historyLimit"), bd["historyLimit"]),
        },
        "memoryEnabled": _as_bool(rec.get("memoryEnabled"), base["memoryEnabled"]),
        "theme": theme if theme in _THEMES else base["theme"],
    }


# --------------------------------------------------------------------------- #
# Row mappers
# --------------------------------------------------------------------------- #

_FOLDER_COLS = ("id, name, description, color, icon, system_prompt, extra_info, memory_mode, "
                "sort_order, created_at, updated_at")
_CONV_COLS = ("id, folder_id, title, title_locked, provider, model, system_prompt, temperature, "
              "max_tokens, pinned, archived, tokens_in, tokens_out, cost_usd, created_at, "
              "updated_at, last_message_at, summary, summary_upto_seq, summary_tokens, "
              "reasoning_effort")
_MSG_COLS = ("id, conversation_id, role, content, reasoning, provider, model, status, error, "
             "tokens_in, tokens_out, cost_usd, latency_ms, citations, seq, created_at")
_ATT_COLS = "id, message_id, conversation_id, name, mime, size, kind, path, text_content, created_at"
_MEM_COLS = "id, scope, folder_id, kind, content, source, pinned, created_at, updated_at"
_DOC_COLS = ("id, scope, folder_id, name, mime, size, path, status, error, chunk_count, "
             "created_at, updated_at")


def _map_folder(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "name": r["name"],
        "description": r["description"],
        "color": r["color"],
        "icon": r["icon"],
        "systemPrompt": r["system_prompt"],
        "extraInfo": r["extra_info"],
        "memoryMode": _memory_mode(r["memory_mode"]),
        "sortOrder": _int(r["sort_order"]),
        "createdAt": _int(r["created_at"]),
        "updatedAt": _int(r["updated_at"]),
    }


def _map_conversation(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "folderId": r["folder_id"],
        "title": r["title"],
        "titleLocked": _b(r["title_locked"]),
        "provider": r["provider"],
        "model": r["model"],
        "systemPrompt": r["system_prompt"],
        "temperature": _float_or_none(r["temperature"]),
        "maxTokens": _int_or_none(r["max_tokens"]),
        "pinned": _b(r["pinned"]),
        "archived": _b(r["archived"]),
        "tokensIn": _int(r["tokens_in"]),
        "tokensOut": _int(r["tokens_out"]),
        "costUsd": float(r["cost_usd"] or 0.0),
        "createdAt": _int(r["created_at"]),
        "updatedAt": _int(r["updated_at"]),
        "lastMessageAt": _int(r["last_message_at"]),
        # Compaction state (localcode.chat.compaction); no summary → None / 0 / 0.
        "summary": r["summary"] or None,
        "summaryUptoSeq": _int(r["summary_upto_seq"]),
        "summaryTokens": _int(r["summary_tokens"]),
        # None = follow settings.chat.reasoning_effort.
        "reasoningEffort": r["reasoning_effort"] or None,
    }


def _map_attachment(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "messageId": r["message_id"],
        "conversationId": r["conversation_id"],
        "name": r["name"],
        "mime": r["mime"],
        "size": _int(r["size"]),
        "kind": _att_kind(r["kind"]),
        "textContent": r["text_content"],
        "createdAt": _int(r["created_at"]),
    }


def _map_message(r: sqlite3.Row, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": r["id"],
        "conversationId": r["conversation_id"],
        "role": _role(r["role"]),
        "content": r["content"] or "",
        "reasoning": r["reasoning"],
        "provider": r["provider"],
        "model": r["model"],
        "status": _status(r["status"]),
        "error": r["error"],
        "tokensIn": _int_or_none(r["tokens_in"]),
        "tokensOut": _int_or_none(r["tokens_out"]),
        "costUsd": _float_or_none(r["cost_usd"]),
        "latencyMs": _int_or_none(r["latency_ms"]),
        "seq": _int(r["seq"]),
        "createdAt": _int(r["created_at"]),
        "attachments": attachments,
        "citations": parse_citations(r["citations"]),
    }


def _map_memory(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "scope": _mem_scope(r["scope"]),
        "folderId": r["folder_id"],
        "kind": _mem_kind(r["kind"]),
        "content": r["content"],
        "source": _mem_source(r["source"]),
        "pinned": _b(r["pinned"]),
        "createdAt": _int(r["created_at"]),
        "updatedAt": _int(r["updated_at"]),
    }


def _map_document(r: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": r["id"],
        "scope": _kb_scope(r["scope"]),
        "folderId": r["folder_id"],
        "name": r["name"],
        "mime": r["mime"],
        "size": _int(r["size"]),
        "status": _kb_status(r["status"]),
        "error": r["error"],
        "chunkCount": _int(r["chunk_count"]),
        "createdAt": _int(r["created_at"]),
        "updatedAt": _int(r["updated_at"]),
    }


def _placeholders(n: int) -> str:
    return ", ".join("?" for _ in range(n))


def _chunk_scope_sql(folder_id: str | None, include_global: bool, alias: str = "c") -> tuple[str, list[Any]] | None:
    """WHERE fragment implementing the reference retrieval scope rules.

    * isolated project (``include_global=False``) → only that folder's chunks,
    * shared project → the folder's chunks plus global ones,
    * no project → global chunks only.
    ``None`` when the scope cannot contain anything.
    """
    if not include_global:
        if not folder_id:
            return None
        return f"{alias}.folder_id = ?", [folder_id]
    if folder_id:
        return f"({alias}.scope = 'global' OR {alias}.folder_id = ?)", [folder_id]
    return f"{alias}.scope = 'global'", []


# --------------------------------------------------------------------------- #
# Repository
# --------------------------------------------------------------------------- #


class ChatRepo:
    """CRUD over :class:`ChatDB` returning ai_ux-shaped dicts."""

    def __init__(self, db: ChatDB, uploads: str | Path | None = None) -> None:
        self.db = db
        self.uploads = Path(uploads) if uploads is not None else uploads_dir()
        db.open()

    # ------------------------------------------------------------------ util
    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call a (sync) repository method in a worker thread."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------ upload files
    def upload_path(self, relative: str | None) -> Path | None:
        """Absolute path of a stored upload, never outside the upload directory."""
        if not relative:
            return None
        root = self.uploads.resolve()
        candidate = Path(relative)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if resolved != root and root not in resolved.parents:
            return None
        return resolved

    def store_upload(self, subdir: str, file_id: str, extension: str, data: bytes) -> str:
        """Write ``data`` under ``uploads/<subdir>/<id><ext>``; returns the relative path."""
        if subdir not in ("att", "kb"):
            raise ValueError(f"unknown upload dir {subdir!r}")
        relative = f"{subdir}/{file_id}{extension}"
        target = self.uploads / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return relative

    def remove_uploads(self, relatives: Iterable[str | None]) -> None:
        """Best-effort removal of stored files (an orphaned file is harmless)."""
        for rel in relatives:
            path = self.upload_path(rel)
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove upload %s", path, exc_info=True)

    # =================================================================== folders
    def list_folders(self) -> list[dict[str, Any]]:
        with self.db.read() as c:
            rows = c.execute(f"SELECT {_FOLDER_COLS} FROM folders ORDER BY sort_order ASC, created_at ASC").fetchall()
        return [_map_folder(r) for r in rows]

    def list_folders_with_stats(self) -> list[dict[str, Any]]:
        with self.db.read() as c:
            rows = c.execute(
                f"""SELECT {', '.join('f.' + col.strip() for col in _FOLDER_COLS.split(','))},
                       (SELECT COUNT(*) FROM conversations c WHERE c.folder_id = f.id) AS conversation_count,
                       (SELECT COUNT(*) FROM kb_documents d WHERE d.folder_id = f.id) AS document_count,
                       (SELECT COUNT(*) FROM memories m WHERE m.folder_id = f.id) AS memory_count
                  FROM folders f
                 ORDER BY f.sort_order ASC, f.created_at ASC"""
            ).fetchall()
        out = []
        for r in rows:
            item = _map_folder(r)
            item["conversationCount"] = _int(r["conversation_count"])
            item["documentCount"] = _int(r["document_count"])
            item["memoryCount"] = _int(r["memory_count"])
            out.append(item)
        return out

    def get_folder(self, folder_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_FOLDER_COLS} FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return _map_folder(r) if r else None

    def create_folder(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        data = data or {}
        fid = new_id()
        ts = now_ms()
        with self.db.transaction() as c:
            order = data.get("sortOrder")
            if order is None:
                order = c.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM folders").fetchone()[0]
            c.execute(
                """INSERT INTO folders (id, name, description, color, icon, system_prompt, extra_info,
                                        memory_mode, sort_order, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fid,
                    (data.get("name") or "").strip() or DEFAULT_FOLDER_NAME,
                    data.get("description"),
                    data.get("color"),
                    data.get("icon"),
                    data.get("systemPrompt"),
                    data.get("extraInfo"),
                    _memory_mode(data.get("memoryMode")),
                    int(order),
                    ts,
                    ts,
                ),
            )
        folder = self.get_folder(fid)
        assert folder is not None
        return folder

    def update_folder(self, folder_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        with self.db.transaction() as c:
            existing = c.execute("SELECT name FROM folders WHERE id = ?", (folder_id,)).fetchone()
            if not existing:
                return None
            fields: dict[str, Any] = {}
            if "name" in data and data["name"] is not None:
                fields["name"] = str(data["name"]).strip() or existing["name"]
            for key, col in (("description", "description"), ("color", "color"), ("icon", "icon"),
                             ("systemPrompt", "system_prompt"), ("extraInfo", "extra_info")):
                if key in data:
                    fields[col] = data[key]
            if data.get("memoryMode") is not None:
                fields["memory_mode"] = _memory_mode(data["memoryMode"])
            if data.get("sortOrder") is not None:
                fields["sort_order"] = int(data["sortOrder"])
            fields["updated_at"] = now_ms()
            self._apply_update(c, "folders", folder_id, fields)
        return self.get_folder(folder_id)

    def delete_folder(self, folder_id: str, delete_chats: bool) -> bool:
        """Remove a folder.

        ``delete_chats=True`` deletes its conversations (with messages and
        attachments); otherwise they become global chats (``folderId = null``).
        Folder memories and knowledge documents always go away, files included.
        """
        files: list[str | None] = []
        with self.db.transaction() as c:
            if not c.execute("SELECT 1 FROM folders WHERE id = ?", (folder_id,)).fetchone():
                return False
            files += [r[0] for r in c.execute("SELECT path FROM kb_documents WHERE folder_id = ?", (folder_id,))]
            if delete_chats:
                files += [r[0] for r in c.execute(
                    """SELECT a.path FROM attachments a
                        WHERE a.conversation_id IN (SELECT id FROM conversations WHERE folder_id = ?)
                           OR a.message_id IN (SELECT m.id FROM messages m JOIN conversations c2
                                                ON c2.id = m.conversation_id WHERE c2.folder_id = ?)""",
                    (folder_id, folder_id),
                )]
                c.execute("DELETE FROM conversations WHERE folder_id = ?", (folder_id,))
            else:
                c.execute("UPDATE conversations SET folder_id = NULL, updated_at = ? WHERE folder_id = ?",
                          (now_ms(), folder_id))
            c.execute("DELETE FROM memories WHERE folder_id = ?", (folder_id,))
            c.execute("DELETE FROM kb_documents WHERE folder_id = ?", (folder_id,))
            c.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
        self.remove_uploads(files)
        return True

    # ============================================================= conversations
    def list_conversations(self, *, folder_id: Any = UNSET, archived: bool | None | Any = UNSET,
                           limit: int | None = None) -> list[dict[str, Any]]:
        """``folder_id``: UNSET → every folder, ``None`` → only chats without a
        folder, str → that folder. ``archived``: UNSET → active only (default),
        ``None`` → both, bool → that state."""
        where: list[str] = []
        args: list[Any] = []
        if folder_id is None:
            where.append("c.folder_id IS NULL")
        elif isinstance(folder_id, str):
            where.append("c.folder_id = ?")
            args.append(folder_id)
        if archived is not None:
            where.append("c.archived = ?")
            args.append(1 if (archived is True) else 0)
        sql = f"""SELECT {', '.join('c.' + col.strip() for col in _CONV_COLS.split(','))},
                         f.name AS folder_name,
                         (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                         (SELECT m2.content FROM messages m2 WHERE m2.conversation_id = c.id
                           ORDER BY m2.seq DESC LIMIT 1) AS preview
                    FROM conversations c
                    LEFT JOIN folders f ON f.id = c.folder_id"""
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY c.pinned DESC, c.last_message_at DESC, c.created_at DESC"
        if limit and limit > 0:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self.db.read() as c:
            rows = c.execute(sql, args).fetchall()
        out = []
        for r in rows:
            item = _map_conversation(r)
            item["preview"] = to_preview(r["preview"])
            item["messageCount"] = _int(r["message_count"])
            item["folderName"] = r["folder_name"]
            out.append(item)
        return out

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_CONV_COLS} FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return _map_conversation(r) if r else None

    def create_conversation(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        data = data or {}
        cid = new_id()
        ts = now_ms()
        with self.db.transaction() as c:
            c.execute(
                """INSERT INTO conversations
                     (id, folder_id, title, title_locked, provider, model, system_prompt, temperature,
                      max_tokens, reasoning_effort, pinned, archived, tokens_in, tokens_out, cost_usd,
                      created_at, updated_at, last_message_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?)""",
                (
                    cid,
                    data.get("folderId"),
                    (data.get("title") or "").strip() or DEFAULT_CONVERSATION_TITLE,
                    1 if data.get("titleLocked") else 0,
                    data.get("provider"),
                    data.get("model"),
                    data.get("systemPrompt"),
                    _float_or_none(data.get("temperature")),
                    _int_or_none(data.get("maxTokens")),
                    _effort_or_none(data.get("reasoningEffort")),
                    1 if data.get("pinned") else 0,
                    1 if data.get("archived") else 0,
                    ts, ts, ts,
                ),
            )
        conv = self.get_conversation(cid)
        assert conv is not None
        return conv

    def update_conversation(self, conversation_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        with self.db.transaction() as c:
            existing = c.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not existing:
                return None
            fields: dict[str, Any] = {}
            if "folderId" in data:
                fields["folder_id"] = data["folderId"]
            if data.get("title") is not None:
                fields["title"] = collapse_ws(str(data["title"])) or existing["title"]
            if data.get("titleLocked") is not None:
                fields["title_locked"] = 1 if data["titleLocked"] else 0
            for key, col in (("provider", "provider"), ("model", "model"), ("systemPrompt", "system_prompt")):
                if key in data:
                    fields[col] = data[key]
            if "temperature" in data:
                fields["temperature"] = _float_or_none(data["temperature"])
            if "maxTokens" in data:
                fields["max_tokens"] = _int_or_none(data["maxTokens"])
            if "reasoningEffort" in data:
                fields["reasoning_effort"] = _effort_or_none(data["reasoningEffort"])
            if data.get("pinned") is not None:
                fields["pinned"] = 1 if data["pinned"] else 0
            if data.get("archived") is not None:
                fields["archived"] = 1 if data["archived"] else 0
            fields["updated_at"] = now_ms()
            self._apply_update(c, "conversations", conversation_id, fields)
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.db.transaction() as c:
            files = [r[0] for r in c.execute(
                """SELECT path FROM attachments WHERE conversation_id = ?
                    OR message_id IN (SELECT id FROM messages WHERE conversation_id = ?)""",
                (conversation_id, conversation_id),
            )]
            c.execute("DELETE FROM attachments WHERE conversation_id = ?", (conversation_id,))
            deleted = c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount
        if deleted:
            self.remove_uploads(files)
        return deleted > 0

    def touch_conversation(self, conversation_id: str, at: int | None = None) -> None:
        ts = at if at is not None else now_ms()
        with self.db.transaction() as c:
            c.execute("UPDATE conversations SET last_message_at = ?, updated_at = ? WHERE id = ?",
                      (ts, ts, conversation_id))

    def add_usage(self, conversation_id: str, tokens_in: int, tokens_out: int, cost_usd: float = 0.0) -> None:
        with self.db.transaction() as c:
            c.execute(
                """UPDATE conversations SET tokens_in = tokens_in + ?, tokens_out = tokens_out + ?,
                          cost_usd = cost_usd + ?, updated_at = ? WHERE id = ?""",
                (max(0, int(tokens_in or 0)), max(0, int(tokens_out or 0)),
                 cost_usd if math.isfinite(cost_usd or 0.0) else 0.0, now_ms(), conversation_id),
            )

    def set_title(self, conversation_id: str, title: str) -> None:
        clean = collapse_ws(title)
        if not clean:
            return
        with self.db.transaction() as c:
            c.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                      (clean, now_ms(), conversation_id))

    # ------------------------------------------------------------- compaction
    def set_summary(self, conversation_id: str, summary: str, upto_seq: int,
                    tokens: int = 0) -> dict[str, Any] | None:
        """Store the compaction summary covering every message up to ``upto_seq``.

        An empty ``summary`` (or ``upto_seq <= 0``) clears it, so the next turn
        sends the whole stored history again.
        """
        text = (summary or "").strip()
        seq = max(0, int(upto_seq or 0))
        if not text or seq <= 0:
            return self.clear_summary(conversation_id)
        with self.db.transaction() as c:
            if not c.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone():
                return None
            c.execute(
                "UPDATE conversations SET summary = ?, summary_upto_seq = ?, summary_tokens = ?, "
                "updated_at = ? WHERE id = ?",
                (text, seq, max(0, int(tokens or 0)), now_ms(), conversation_id),
            )
        return self.get_conversation(conversation_id)

    def clear_summary(self, conversation_id: str) -> dict[str, Any] | None:
        """Forget the summary (the conversation is sent verbatim again)."""
        with self.db.transaction() as c:
            if not c.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone():
                return None
            c.execute(
                "UPDATE conversations SET summary = NULL, summary_upto_seq = 0, summary_tokens = 0, "
                "updated_at = ? WHERE id = ?",
                (now_ms(), conversation_id),
            )
        return self.get_conversation(conversation_id)

    def invalidate_summary_from(self, conversation_id: str, seq: int) -> bool:
        """Drop the summary when it covers ``seq`` (an edited or removed message).

        ``summary_upto_seq`` must never point at or past a message whose text
        changed — the summary would then describe something the user never said.
        Returns ``True`` when a summary was actually dropped.
        """
        with self.db.transaction() as c:
            removed = c.execute(
                "UPDATE conversations SET summary = NULL, summary_upto_seq = 0, summary_tokens = 0, "
                "updated_at = ? WHERE id = ? AND summary_upto_seq >= ? AND summary_upto_seq > 0",
                (now_ms(), conversation_id, max(0, int(seq or 0))),
            ).rowcount
        return removed > 0

    # ================================================================== messages
    def _attachments_for(self, c: sqlite3.Connection, message_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        if not message_ids:
            return grouped
        for start in range(0, len(message_ids), 500):
            batch = message_ids[start:start + 500]
            rows = c.execute(
                f"SELECT {_ATT_COLS} FROM attachments WHERE message_id IN ({_placeholders(len(batch))}) "
                "ORDER BY created_at ASC, rowid ASC",
                batch,
            ).fetchall()
            for r in rows:
                grouped.setdefault(r["message_id"], []).append(_map_attachment(r))
        return grouped

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.db.read() as c:
            rows = c.execute(
                f"SELECT {_MSG_COLS} FROM messages WHERE conversation_id = ? ORDER BY seq ASC, created_at ASC",
                (conversation_id,),
            ).fetchall()
            grouped = self._attachments_for(c, [r["id"] for r in rows])
        return [_map_message(r, grouped.get(r["id"], [])) for r in rows]

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_MSG_COLS} FROM messages WHERE id = ?", (message_id,)).fetchone()
            if not r:
                return None
            grouped = self._attachments_for(c, [r["id"]])
        return _map_message(r, grouped.get(r["id"], []))

    def create_message(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a message. ``data`` keys follow the Message DTO plus
        ``attachmentIds`` (pending uploads linked to the new message)."""
        mid = data.get("id") or new_id()
        ts = data.get("createdAt") or now_ms()
        attachment_ids = [a for a in (data.get("attachmentIds") or []) if isinstance(a, str) and a]
        with self.db.transaction() as c:
            seq = data.get("seq")
            if seq is None:
                seq = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?",
                                (data["conversationId"],)).fetchone()[0]
            c.execute(
                """INSERT INTO messages (id, conversation_id, role, content, reasoning, provider, model,
                                         status, error, tokens_in, tokens_out, cost_usd, latency_ms,
                                         citations, seq, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mid,
                    data["conversationId"],
                    _role(data.get("role")),
                    data.get("content") or "",
                    data.get("reasoning"),
                    data.get("provider"),
                    data.get("model"),
                    _status(data.get("status")),
                    data.get("error"),
                    _int_or_none(data.get("tokensIn")),
                    _int_or_none(data.get("tokensOut")),
                    _float_or_none(data.get("costUsd")),
                    _int_or_none(data.get("latencyMs")),
                    serialize_citations(data.get("citations")),
                    int(seq),
                    int(ts),
                ),
            )
            self._link_attachments(c, attachment_ids, mid, data["conversationId"])
        msg = self.get_message(mid)
        assert msg is not None
        return msg

    def update_message(self, message_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        fields: dict[str, Any] = {}
        mapping = {
            "content": "content", "reasoning": "reasoning", "provider": "provider", "model": "model",
            "error": "error",
        }
        for key, col in mapping.items():
            if key in patch:
                fields[col] = patch[key] if key != "content" else (patch[key] or "")
        if "status" in patch:
            fields["status"] = _status(patch["status"])
        for key, col, conv in (("tokensIn", "tokens_in", _int_or_none), ("tokensOut", "tokens_out", _int_or_none),
                               ("costUsd", "cost_usd", _float_or_none), ("latencyMs", "latency_ms", _int_or_none)):
            if key in patch:
                fields[col] = conv(patch[key])
        if "citations" in patch:
            fields["citations"] = serialize_citations(patch["citations"])
        if fields:
            with self.db.transaction() as c:
                self._apply_update(c, "messages", message_id, fields)
        return self.get_message(message_id)

    def delete_message(self, message_id: str) -> bool:
        with self.db.transaction() as c:
            files = [r[0] for r in c.execute("SELECT path FROM attachments WHERE message_id = ?", (message_id,))]
            row = c.execute("SELECT conversation_id, seq FROM messages WHERE id = ?", (message_id,)).fetchone()
            deleted = c.execute("DELETE FROM messages WHERE id = ?", (message_id,)).rowcount
            if deleted and row is not None:
                # A message inside the summarised region is gone: the summary
                # still describes it, so it must go too.
                c.execute(
                    "UPDATE conversations SET summary = NULL, summary_upto_seq = 0, summary_tokens = 0, "
                    "updated_at = ? WHERE id = ? AND summary_upto_seq >= ? AND summary_upto_seq > 0",
                    (now_ms(), row["conversation_id"], _int(row["seq"])),
                )
        if deleted:
            self.remove_uploads(files)
        return deleted > 0

    def remove_messages_after(self, conversation_id: str, seq: int) -> int:
        """Drop everything strictly after ``seq`` (edit & regenerate).

        ``seq`` is the message being edited or re-answered, so a summary that
        covers it is no longer trustworthy and is dropped in the same
        transaction (see :meth:`invalidate_summary_from`).
        """
        with self.db.transaction() as c:
            files = [r[0] for r in c.execute(
                """SELECT a.path FROM attachments a JOIN messages m ON m.id = a.message_id
                    WHERE m.conversation_id = ? AND m.seq > ?""",
                (conversation_id, seq),
            )]
            removed = c.execute("DELETE FROM messages WHERE conversation_id = ? AND seq > ?",
                                (conversation_id, seq)).rowcount
            c.execute(
                "UPDATE conversations SET summary = NULL, summary_upto_seq = 0, summary_tokens = 0, "
                "updated_at = ? WHERE id = ? AND summary_upto_seq >= ? AND summary_upto_seq > 0",
                (now_ms(), conversation_id, max(0, int(seq or 0))),
            )
        self.remove_uploads(files)
        return removed

    def next_seq(self, conversation_id: str) -> int:
        with self.db.read() as c:
            return int(c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?",
                                 (conversation_id,)).fetchone()[0])

    def last_user_message(self, conversation_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(
                "SELECT id FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY seq DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return self.get_message(r["id"]) if r else None

    def count_messages(self, conversation_id: str) -> int:
        with self.db.read() as c:
            return int(c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                                 (conversation_id,)).fetchone()[0])

    # =============================================================== attachments
    def _link_attachments(self, c: sqlite3.Connection, ids: list[str], message_id: str, conversation_id: str) -> None:
        if not ids:
            return
        c.execute(
            f"UPDATE attachments SET message_id = ?, conversation_id = ? WHERE id IN ({_placeholders(len(ids))})",
            [message_id, conversation_id, *ids],
        )

    def create_attachment(self, data: dict[str, Any]) -> dict[str, Any]:
        aid = data.get("id") or new_id()
        with self.db.transaction() as c:
            c.execute(
                """INSERT INTO attachments (id, message_id, conversation_id, name, mime, size, kind, path,
                                            text_content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    aid,
                    data.get("messageId"),
                    data.get("conversationId"),
                    data["name"],
                    data.get("mime") or "application/octet-stream",
                    max(0, _int(data.get("size"))),
                    _att_kind(data.get("kind")),
                    data.get("path"),
                    data.get("textContent"),
                    data.get("createdAt") or now_ms(),
                ),
            )
        att = self.get_attachment(aid)
        assert att is not None
        return att

    def get_attachment(self, attachment_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_ATT_COLS} FROM attachments WHERE id = ?", (attachment_id,)).fetchone()
        return _map_attachment(r) if r else None

    def attachment_file(self, attachment_id: str) -> tuple[dict[str, Any], Path | None] | None:
        """``(attachment DTO, absolute file path or None)``; ``None`` when missing."""
        with self.db.read() as c:
            r = c.execute(f"SELECT {_ATT_COLS} FROM attachments WHERE id = ?", (attachment_id,)).fetchone()
        if not r:
            return None
        return _map_attachment(r), self.upload_path(r["path"])

    def list_attachments_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        ids = [i for i in ids if i]
        if not ids:
            return []
        with self.db.read() as c:
            rows = c.execute(
                f"SELECT {_ATT_COLS} FROM attachments WHERE id IN ({_placeholders(len(ids))}) ORDER BY created_at ASC",
                ids,
            ).fetchall()
        return [_map_attachment(r) for r in rows]

    def attach_to_message(self, ids: list[str], message_id: str) -> list[dict[str, Any]]:
        ids = [i for i in ids if i]
        if not ids:
            return []
        with self.db.transaction() as c:
            parent = c.execute("SELECT conversation_id FROM messages WHERE id = ?", (message_id,)).fetchone()
            if not parent:
                return []
            self._link_attachments(c, ids, message_id, parent["conversation_id"])
            grouped = self._attachments_for(c, [message_id])
        return grouped.get(message_id, [])

    def purge_stale_uploads(self, older_than_ms: int = 24 * 3600 * 1000) -> int:
        """Delete pending uploads (never linked to a message) older than the cutoff."""
        cutoff = now_ms() - older_than_ms
        with self.db.transaction() as c:
            rows = c.execute("SELECT id, path FROM attachments WHERE message_id IS NULL AND created_at < ?",
                             (cutoff,)).fetchall()
            if rows:
                c.execute(f"DELETE FROM attachments WHERE id IN ({_placeholders(len(rows))})",
                          [r["id"] for r in rows])
        self.remove_uploads(r["path"] for r in rows)
        return len(rows)

    # ================================================================== memories
    def list_memories(self, *, scope: str | None = None, folder_id: Any = UNSET) -> list[dict[str, Any]]:
        where: list[str] = []
        args: list[Any] = []
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if folder_id is None:
            where.append("folder_id IS NULL")
        elif isinstance(folder_id, str):
            where.append("folder_id = ?")
            args.append(folder_id)
        sql = f"SELECT {_MEM_COLS} FROM memories"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY pinned DESC, updated_at DESC"
        with self.db.read() as c:
            rows = c.execute(sql, args).fetchall()
        return [_map_memory(r) for r in rows]

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_MEM_COLS} FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _map_memory(r) if r else None

    def create_memory(self, data: dict[str, Any]) -> dict[str, Any]:
        mid = new_id()
        ts = now_ms()
        scope = data.get("scope") or ("folder" if data.get("folderId") else "global")
        folder_id = data.get("folderId") if scope == "folder" else None
        with self.db.transaction() as c:
            c.execute(
                """INSERT INTO memories (id, scope, folder_id, kind, content, source, pinned, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, _mem_scope(scope), folder_id, _mem_kind(data.get("kind")), data.get("content") or "",
                 _mem_source(data.get("source")), 1 if data.get("pinned") else 0, ts, ts),
            )
        mem = self.get_memory(mid)
        assert mem is not None
        return mem

    def update_memory(self, memory_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_memory(memory_id)
        if not existing:
            return None
        fields: dict[str, Any] = {}
        scope = _mem_scope(data["scope"]) if data.get("scope") is not None else None
        next_scope = scope or existing["scope"]
        if "folderId" in data or scope is not None:
            fields["folder_id"] = (data.get("folderId") or existing["folderId"]) if next_scope == "folder" else None
        if scope is not None:
            fields["scope"] = scope
        if data.get("kind") is not None:
            fields["kind"] = _mem_kind(data["kind"])
        if data.get("content") is not None:
            fields["content"] = data["content"]
        if data.get("source") is not None:
            fields["source"] = _mem_source(data["source"])
        if data.get("pinned") is not None:
            fields["pinned"] = 1 if data["pinned"] else 0
        fields["updated_at"] = now_ms()
        with self.db.transaction() as c:
            self._apply_update(c, "memories", memory_id, fields)
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self.db.transaction() as c:
            return c.execute("DELETE FROM memories WHERE id = ?", (memory_id,)).rowcount > 0

    # ================================================================= knowledge
    def list_documents(self, *, scope: str | None = None, folder_id: Any = UNSET) -> list[dict[str, Any]]:
        where: list[str] = []
        args: list[Any] = []
        if scope:
            where.append("scope = ?")
            args.append(scope)
        if folder_id is None:
            where.append("folder_id IS NULL")
        elif isinstance(folder_id, str):
            where.append("folder_id = ?")
            args.append(folder_id)
        sql = f"SELECT {_DOC_COLS} FROM kb_documents"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC"
        with self.db.read() as c:
            rows = c.execute(sql, args).fetchall()
        return [_map_document(r) for r in rows]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self.db.read() as c:
            r = c.execute(f"SELECT {_DOC_COLS} FROM kb_documents WHERE id = ?", (document_id,)).fetchone()
        return _map_document(r) if r else None

    def create_document(self, data: dict[str, Any]) -> dict[str, Any]:
        did = data.get("id") or new_id()
        ts = now_ms()
        scope = _kb_scope(data.get("scope") or ("folder" if data.get("folderId") else "global"))
        folder_id = data.get("folderId") if scope == "folder" else None
        with self.db.transaction() as c:
            c.execute(
                """INSERT INTO kb_documents (id, scope, folder_id, name, mime, size, path, status, error,
                                             chunk_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (did, scope, folder_id, data["name"], data.get("mime") or "application/octet-stream",
                 max(0, _int(data.get("size"))), data.get("path"), _kb_status(data.get("status")),
                 data.get("error"), max(0, _int(data.get("chunkCount"))), ts, ts),
            )
        doc = self.get_document(did)
        assert doc is not None
        return doc

    def update_document(self, document_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        fields: dict[str, Any] = {}
        for key, col in (("name", "name"), ("mime", "mime"), ("path", "path"), ("error", "error")):
            if key in patch:
                fields[col] = patch[key]
        if "size" in patch:
            fields["size"] = max(0, _int(patch["size"]))
        if "status" in patch:
            fields["status"] = _kb_status(patch["status"])
        if "chunkCount" in patch:
            fields["chunk_count"] = max(0, _int(patch["chunkCount"]))
        fields["updated_at"] = now_ms()
        with self.db.transaction() as c:
            if not c.execute("SELECT 1 FROM kb_documents WHERE id = ?", (document_id,)).fetchone():
                return None
            self._apply_update(c, "kb_documents", document_id, fields)
        return self.get_document(document_id)

    def document_path(self, document_id: str) -> Path | None:
        with self.db.read() as c:
            r = c.execute("SELECT path FROM kb_documents WHERE id = ?", (document_id,)).fetchone()
        return self.upload_path(r["path"]) if r else None

    def delete_document(self, document_id: str) -> bool:
        with self.db.transaction() as c:
            r = c.execute("SELECT path FROM kb_documents WHERE id = ?", (document_id,)).fetchone()
            if not r:
                return False
            c.execute("DELETE FROM kb_chunks WHERE document_id = ?", (document_id,))
            c.execute("DELETE FROM kb_documents WHERE id = ?", (document_id,))
        self.remove_uploads([r["path"]])
        return True

    def insert_chunks(self, document_id: str, contents: list[str], *,
                      token_estimates: list[int] | None = None) -> list[str]:
        """Replace the chunks of a document; returns the new chunk ids (in order)."""
        ids: list[str] = []
        with self.db.transaction() as c:
            doc = c.execute("SELECT scope, folder_id FROM kb_documents WHERE id = ?", (document_id,)).fetchone()
            if not doc:
                return []
            c.execute("DELETE FROM kb_chunks WHERE document_id = ?", (document_id,))
            scope = _kb_scope(doc["scope"])
            folder_id = doc["folder_id"] if scope == "folder" else None
            for idx, content in enumerate(contents):
                cid = new_id()
                estimate = token_estimates[idx] if token_estimates and idx < len(token_estimates) else estimate_tokens(content)
                c.execute(
                    """INSERT INTO kb_chunks (id, document_id, folder_id, scope, idx, content, token_estimate)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (cid, document_id, folder_id, scope, idx, content, int(estimate)),
                )
                ids.append(cid)
            c.execute("UPDATE kb_documents SET chunk_count = ?, updated_at = ? WHERE id = ?",
                      (len(ids), now_ms(), document_id))
        return ids

    def list_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with self.db.read() as c:
            rows = c.execute(
                """SELECT id, document_id, folder_id, scope, idx, content, token_estimate, embedding_model,
                          embedding_dim FROM kb_chunks WHERE document_id = ? ORDER BY idx ASC""",
                (document_id,),
            ).fetchall()
        return [{
            "id": r["id"], "documentId": r["document_id"], "folderId": r["folder_id"],
            "scope": _kb_scope(r["scope"]), "idx": _int(r["idx"]), "content": r["content"],
            "tokenEstimate": _int(r["token_estimate"]), "embeddingModel": r["embedding_model"],
            "embeddingDim": _int_or_none(r["embedding_dim"]),
        } for r in rows]

    def search_chunks(self, match: str, *, folder_id: str | None, include_global: bool,
                      limit: int) -> list[dict[str, Any]]:
        """BM25-ranked chunks for an FTS5 ``match`` expression (best first)."""
        scope = _chunk_scope_sql(folder_id, include_global)
        if scope is None or limit <= 0:
            return []
        scope_sql, scope_args = scope
        sql = f"""SELECT c.id, c.document_id, c.folder_id, c.scope, c.idx, c.content, c.token_estimate,
                         d.name AS document_name, bm25(kb_chunks_fts) AS rank
                    FROM kb_chunks_fts
                    JOIN kb_chunks c ON c.rid = kb_chunks_fts.rowid
                    JOIN kb_documents d ON d.id = c.document_id
                   WHERE kb_chunks_fts MATCH ? AND d.status = 'ready' AND {scope_sql}
                   ORDER BY rank ASC
                   LIMIT ?"""
        try:
            with self.db.read() as c:
                rows = c.execute(sql, [match, *scope_args, int(limit)]).fetchall()
        except sqlite3.OperationalError:
            log.warning("fts chunk query failed for %r", match, exc_info=True)
            return []
        return [{
            "id": r["id"], "documentId": r["document_id"], "folderId": r["folder_id"],
            "scope": _kb_scope(r["scope"]), "idx": _int(r["idx"]), "content": r["content"],
            "tokenEstimate": _int(r["token_estimate"]), "documentName": r["document_name"],
            "bm25": float(r["rank"]),
        } for r in rows]

    def embedded_chunks(self, model: str, *, folder_id: str | None, include_global: bool) -> list[dict[str, Any]]:
        """Chunks in scope that carry an embedding from ``model`` (with the raw BLOB)."""
        scope = _chunk_scope_sql(folder_id, include_global)
        if scope is None:
            return []
        scope_sql, scope_args = scope
        with self.db.read() as c:
            rows = c.execute(
                f"""SELECT c.id, c.document_id, c.folder_id, c.scope, c.idx, c.content, c.token_estimate,
                           c.embedding, c.embedding_dim, d.name AS document_name
                      FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                     WHERE d.status = 'ready' AND c.embedding_model = ? AND c.embedding IS NOT NULL
                       AND {scope_sql}""",
                [model, *scope_args],
            ).fetchall()
        return [{
            "id": r["id"], "documentId": r["document_id"], "folderId": r["folder_id"],
            "scope": _kb_scope(r["scope"]), "idx": _int(r["idx"]), "content": r["content"],
            "tokenEstimate": _int(r["token_estimate"]), "documentName": r["document_name"],
            "embedding": r["embedding"], "embeddingDim": _int(r["embedding_dim"]),
        } for r in rows]

    def chunks_missing_embedding(self, model: str, *, document_id: str | None = None,
                                 limit: int = 256) -> list[tuple[str, str]]:
        """``(chunk id, content)`` of ready chunks without an embedding from ``model``."""
        sql = """SELECT c.id, c.content FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                  WHERE d.status = 'ready' AND (c.embedding_model IS NULL OR c.embedding_model != ?)"""
        args: list[Any] = [model]
        if document_id:
            sql += " AND c.document_id = ?"
            args.append(document_id)
        sql += " ORDER BY c.rid ASC LIMIT ?"
        args.append(int(limit))
        with self.db.read() as c:
            return [(r["id"], r["content"]) for r in c.execute(sql, args)]

    def set_chunk_embeddings(self, model: str, items: list[tuple[str, bytes, int]]) -> int:
        if not items:
            return 0
        with self.db.transaction() as c:
            c.executemany(
                "UPDATE kb_chunks SET embedding = ?, embedding_dim = ?, embedding_model = ? WHERE id = ?",
                [(blob, int(dim), model, cid) for cid, blob, dim in items],
            )
        return len(items)

    # ================================================================== settings
    def get_settings(self) -> dict[str, Any]:
        with self.db.read() as c:
            r = c.execute("SELECT value FROM settings WHERE id = 1").fetchone()
        if not r:
            return coerce_settings({})
        try:
            return coerce_settings(json.loads(r["value"]))
        except (TypeError, ValueError):
            return coerce_settings({})

    def patch_settings(self, partial: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge ``partial`` over the stored settings and persist the result."""
        with self.db.transaction() as c:
            current = self.get_settings()
            merged = {
                **current,
                **{k: v for k, v in partial.items() if k not in ("profile", "defaults")},
                "profile": {**current["profile"], **_as_record(partial.get("profile"))},
                "defaults": {**current["defaults"], **_as_record(partial.get("defaults"))},
            }
            nxt = coerce_settings(merged, current)
            c.execute(
                """INSERT INTO settings (id, value, updated_at) VALUES (1, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
                (json.dumps(nxt, ensure_ascii=False), now_ms()),
            )
        return nxt

    # ==================================================================== search
    def search_messages(self, query: str, limit: int = 30) -> list[dict[str, Any]]:
        """Full-text search over message bodies, best matches first.

        Diacritics are folded on both sides (``zazolc`` finds ``Zażółć``); the
        snippet keeps the original spelling with ``<mark>…</mark>`` around hits.
        """
        match = build_fts_query(query)
        if not match:
            return []
        sql = f"""SELECT m.id AS message_id, m.conversation_id, m.role, m.content, m.created_at,
                         c.title AS conversation_title, c.folder_id, f.name AS folder_name,
                         messages_fts.content AS folded,
                         snippet(messages_fts, 0, '{SNIPPET_OPEN}', '{SNIPPET_CLOSE}', '{SNIPPET_ELLIPSIS}', 18) AS snip,
                         bm25(messages_fts) AS rank
                    FROM messages_fts
                    JOIN messages m ON m.rid = messages_fts.rowid
                    JOIN conversations c ON c.id = m.conversation_id
                    LEFT JOIN folders f ON f.id = c.folder_id
                   WHERE messages_fts MATCH ?
                   ORDER BY rank ASC, m.created_at DESC
                   LIMIT ?"""
        try:
            with self.db.read() as c:
                rows = c.execute(sql, (match, max(1, int(limit)))).fetchall()
        except sqlite3.OperationalError:
            log.warning("fts message query failed for %r", match, exc_info=True)
            return []
        return [{
            "conversationId": r["conversation_id"],
            "conversationTitle": r["conversation_title"],
            "messageId": r["message_id"],
            "role": _role(r["role"]),
            "snippet": restore_snippet(r["snip"], r["folded"] or "", r["content"] or ""),
            "createdAt": _int(r["created_at"]),
            "folderId": r["folder_id"],
            "folderName": r["folder_name"],
        } for r in rows]

    # ==================================================================== helpers
    @staticmethod
    def _apply_update(c: sqlite3.Connection, table: str, row_id: str, fields: dict[str, Any]) -> int:
        """UPDATE from a sparse field map. Table/column names are constants, never input."""
        if not fields:
            return 0
        assignments = ", ".join(f"{col} = ?" for col in fields)
        return c.execute(f"UPDATE {table} SET {assignments} WHERE id = ?", [*fields.values(), row_id]).rowcount
