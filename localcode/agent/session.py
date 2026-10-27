"""Persistence and resume of coding-agent sessions (append-only JSONL).

One session is one file::

    ~/.localcode/sessions/<slug-of-cwd>/<session id>.jsonl

The first line is the header, everything after it is appended and never
rewritten, so a crash can only ever lose the tail of a transcript:

``{"kind": "session", "id", "cwd", "modelKey", "mode", "title", "createdAt"}``
    header (line 1).
``{"kind": "message", "at": <ms>, "message": {...}}``
    one message of the **model history** (OpenAI shape, see
    ``docs/ARCHITECTURE.md``; the system prompt is *not* stored — it is rebuilt
    on resume from the current cwd, mode and model).
``{"kind": "replace_history", "at": <ms>, "messages": [...], "reason": "..."}``
    the whole history was rewritten (compaction, ``/clear``). Resume replays
    records in order, so the restored history is exactly what the model saw.
``{"kind": "item", "at": <ms>, "item": {...}}``
    one display item (``AgentItem``: user / assistant / reasoning / tool /
    notice). Display items are independent of the model history: they keep the
    reasoning and the full tool output that the model never sees again.
``{"kind": "meta", "at": <ms>, ...}``
    a header field changed later in the session (``title`` — derived from the
    first user message, ``modelKey``, ``mode``).

Reads are line-based and tolerant: an unparsable or truncated last line (a
crash during a write) is skipped, not fatal. Listing only parses the ``session``
and ``meta`` records, which are recognised by the line prefix, so it never has
to decode megabytes of transcript.

Writes are synchronous (open, append, close): one record is a single small
``write`` syscall, which is far cheaper than moving it to a thread, and keeping
the ordering trivially correct matters more here than the microseconds. Use
:meth:`SessionStore.append_many` when writing several records at once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from localcode.config import sessions_dir

log = logging.getLogger(__name__)

__all__ = [
    "AgentSession",
    "ITEM_KINDS",
    "SessionError",
    "SessionStore",
    "delete",
    "delete_session",
    "derive_title",
    "find_path",
    "list_sessions",
    "load",
    "load_session",
    "new_session_id",
    "slug_for",
]

#: Display-item kinds (``AgentItem.kind`` in the HTTP API).
ITEM_KINDS: tuple[str, ...] = ("user", "assistant", "reasoning", "tool", "notice")

_TITLE_MAX = 60
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SLUG_CLEAN = re.compile(r"[^A-Za-z0-9._-]+")
_HEADER_PREFIX = '{"kind": "session"'
_META_PREFIX = '{"kind": "meta"'


class SessionError(RuntimeError):
    """Session could not be read or written (message is Polish, safe to show)."""


def now_ms() -> int:
    return int(time.time() * 1000)


def new_session_id() -> str:
    """Time-sortable, filename-safe session id (``20260923-141503-a1b2c3``)."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def slug_for(cwd: str | os.PathLike[str]) -> str:
    """Directory name for a project: readable basename + hash of the full path."""
    path = Path(cwd).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    name = _SLUG_CLEAN.sub("-", path.name or "root").strip("-.") or "root"
    digest = hashlib.sha1(str(path).encode("utf-8", "replace")).hexdigest()[:8]
    return f"{name[:40]}-{digest}"


def derive_title(text: str, limit: int = _TITLE_MAX) -> str:
    """Session title from the first user message: its first non-empty line."""
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return ""


def _normalize_cwd(cwd: str | os.PathLike[str]) -> str:
    path = Path(cwd).expanduser()
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _root(root: str | os.PathLike[str] | None = None) -> Path:
    """Sessions directory (``LOCALCODE_HOME`` is read on every call, for tests)."""
    return Path(root).expanduser() if root is not None else sessions_dir()


# --------------------------------------------------------------------------- #
# Session header
# --------------------------------------------------------------------------- #


@dataclass
class AgentSession:
    """Header of one session plus the live counters the API reports."""

    id: str
    cwd: str
    model_key: str | None = None
    mode: str = "default"
    title: str = ""
    created_at: int = field(default_factory=now_ms)
    updated_at: int = field(default_factory=now_ms)
    #: runtime-only, filled by the agent (not stored in the header)
    context_used: int = 0
    context_limit: int = 0
    busy: bool = False
    path: Path | None = None

    def header(self) -> dict[str, Any]:
        return {
            "kind": "session",
            "id": self.id,
            "cwd": self.cwd,
            "modelKey": self.model_key,
            "mode": self.mode,
            "title": self.title,
            "createdAt": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """``AgentSession`` DTO of the HTTP API (camelCase)."""
        return {
            "id": self.id,
            "cwd": self.cwd,
            "modelKey": self.model_key,
            "mode": self.mode,
            "title": self.title,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "contextUsed": self.context_used,
            "contextLimit": self.context_limit,
            "busy": self.busy,
        }

    @classmethod
    def from_header(cls, data: dict[str, Any], *, path: Path | None = None) -> "AgentSession":
        created = data.get("createdAt")
        return cls(
            id=str(data.get("id") or ""),
            cwd=str(data.get("cwd") or ""),
            model_key=(str(data["modelKey"]) if data.get("modelKey") else None),
            mode=str(data.get("mode") or "default"),
            title=str(data.get("title") or ""),
            created_at=int(created) if isinstance(created, (int, float)) else now_ms(),
            updated_at=int(created) if isinstance(created, (int, float)) else now_ms(),
            path=path,
        )


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise SessionError(f"Nie znaleziono sesji: {path.name}") from None
    except OSError as exc:
        raise SessionError(f"Nie można odczytać sesji {path.name}: {exc.strerror or exc}") from exc
    with handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                log.warning("session %s: skipping unparsable line %d", path.name, number)
                continue
            if isinstance(record, dict):
                yield record


def _apply_meta(session: AgentSession, record: dict[str, Any]) -> None:
    if "title" in record and record["title"] is not None:
        session.title = str(record["title"])
    if "modelKey" in record:
        value = record["modelKey"]
        session.model_key = str(value) if value else None
    if "mode" in record and record["mode"]:
        session.mode = str(record["mode"])
    at = record.get("at")
    if isinstance(at, (int, float)):
        session.updated_at = max(session.updated_at, int(at))


def _read_header(path: Path) -> AgentSession | None:
    """Header + later ``meta`` changes, without decoding the transcript."""
    session: AgentSession | None = None
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with handle:
        for line in handle:
            if line.startswith(_HEADER_PREFIX):
                if session is not None:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    return None
                session = AgentSession.from_header(data, path=path)
            elif line.startswith(_META_PREFIX) and session is not None:
                try:
                    _apply_meta(session, json.loads(line))
                except ValueError:
                    continue
    if session is None or not session.id:
        return None
    try:
        session.updated_at = max(session.updated_at, int(path.stat().st_mtime * 1000))
    except OSError:
        pass
    return session


def find_path(session_id: str, *, root: str | os.PathLike[str] | None = None) -> Path | None:
    """The transcript file of ``session_id`` in any project directory."""
    if not _ID_RE.match(session_id or ""):
        return None
    base = _root(root)
    try:
        matches = sorted(base.glob(f"*/{session_id}.jsonl"))
    except OSError:
        return None
    return matches[0] if matches else None


def list_sessions(
    cwd: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    limit: int | None = None,
) -> list[AgentSession]:
    """Sessions, newest first; ``cwd`` restricts them to one project."""
    base = _root(root)
    if cwd is not None:
        target = _normalize_cwd(cwd)
        dirs = [base / slug_for(target)]
    else:
        target = None
        try:
            dirs = sorted(d for d in base.iterdir() if d.is_dir())
        except OSError:
            return []
    found: list[AgentSession] = []
    for directory in dirs:
        try:
            files = sorted(directory.glob("*.jsonl"))
        except OSError:
            continue
        for path in files:
            session = _read_header(path)
            if session is None:
                continue
            if target is not None and session.cwd and _normalize_cwd(session.cwd) != target:
                continue
            found.append(session)
    found.sort(key=lambda s: (s.updated_at, s.created_at, s.id), reverse=True)
    return found[: limit] if limit is not None and limit >= 0 else found


def load(
    session_id: str, *, root: str | os.PathLike[str] | None = None
) -> tuple[AgentSession, list[dict[str, Any]], list[dict[str, Any]]]:
    """``(session, model messages, display items)`` of a stored session."""
    path = find_path(session_id, root=root)
    if path is None:
        raise SessionError(f"Nie znaleziono sesji o identyfikatorze {session_id!r}.")
    return _load_path(path)


def _load_path(path: Path) -> tuple[AgentSession, list[dict[str, Any]], list[dict[str, Any]]]:
    session: AgentSession | None = None
    messages: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    last_at = 0
    for record in _iter_records(path):
        kind = record.get("kind")
        at = record.get("at")
        if isinstance(at, (int, float)):
            last_at = max(last_at, int(at))
        if kind == "session":
            if session is None:
                session = AgentSession.from_header(record, path=path)
            continue
        if session is None:
            continue
        if kind == "message":
            message = record.get("message")
            if isinstance(message, dict):
                messages.append(message)
        elif kind == "replace_history":
            new = record.get("messages")
            messages = [m for m in new if isinstance(m, dict)] if isinstance(new, list) else []
        elif kind == "item":
            item = record.get("item")
            if isinstance(item, dict):
                items.append(item)
        elif kind == "meta":
            _apply_meta(session, record)
    if session is None:
        raise SessionError(f"Plik sesji {path.name} jest uszkodzony (brak nagłówka).")
    session.updated_at = max(session.updated_at, last_at)
    try:
        session.updated_at = max(session.updated_at, int(path.stat().st_mtime * 1000))
    except OSError:
        pass
    return session, messages, items


def delete(session_id: str, *, root: str | os.PathLike[str] | None = None) -> bool:
    """Remove a session file (and its project directory when it becomes empty)."""
    path = find_path(session_id, root=root)
    if path is None:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SessionError(f"Nie można usunąć sesji: {exc.strerror or exc}") from exc
    try:
        next(path.parent.iterdir())
    except StopIteration:
        try:
            path.parent.rmdir()
        except OSError:
            pass
    except OSError:
        pass
    return True


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class SessionStore:
    """Append-only writer for one session; also the in-memory mirror of it.

    ``messages`` is the model history as it will be restored on resume and
    ``items`` the display transcript; both are kept in sync with the file, so a
    caller that just resumed a session can hand them straight to the agent.
    """

    def __init__(
        self,
        session: AgentSession,
        path: Path,
        *,
        messages: Sequence[dict[str, Any]] = (),
        items: Sequence[dict[str, Any]] = (),
        writable: bool = True,
    ) -> None:
        self.session = session
        self.path = path
        self.messages: list[dict[str, Any]] = list(messages)
        self.items: list[dict[str, Any]] = list(items)
        self.writable = writable
        session.path = path

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def create(
        cls,
        cwd: str | os.PathLike[str],
        *,
        model_key: str | None = None,
        mode: str = "default",
        title: str = "",
        session_id: str | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> "SessionStore":
        """Start a new session file under ``<root>/<slug of cwd>/<id>.jsonl``."""
        resolved = _normalize_cwd(cwd)
        sid = session_id or new_session_id()
        if not _ID_RE.match(sid):
            raise SessionError(f"Nieprawidłowy identyfikator sesji: {sid!r}.")
        directory = _root(root) / slug_for(resolved)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(f"Nie można utworzyć katalogu sesji: {exc.strerror or exc}") from exc
        path = directory / f"{sid}.jsonl"
        if path.exists():
            raise SessionError(f"Sesja {sid} już istnieje.")
        session = AgentSession(id=sid, cwd=resolved, model_key=model_key, mode=mode, title=title)
        store = cls(session, path)
        store._write([session.header()])
        return store

    @classmethod
    def resume(cls, session_id: str, *, root: str | os.PathLike[str] | None = None) -> "SessionStore":
        """Reopen a stored session for appending, with its history and items."""
        path = find_path(session_id, root=root)
        if path is None:
            raise SessionError(f"Nie znaleziono sesji o identyfikatorze {session_id!r}.")
        session, messages, items = _load_path(path)
        return cls(session, path, messages=messages, items=items)

    @classmethod
    def open_path(cls, path: str | os.PathLike[str]) -> "SessionStore":
        """Reopen a session by file path (used by tools that already have one)."""
        p = Path(path).expanduser()
        session, messages, items = _load_path(p)
        return cls(session, p, messages=messages, items=items)

    # ------------------------------------------------------------------ write
    def _write(self, records: Iterable[dict[str, Any]]) -> None:
        if not self.writable:
            return
        lines = []
        for record in records:
            try:
                lines.append(json.dumps(record, ensure_ascii=False, default=str))
            except (TypeError, ValueError):  # pragma: no cover - default=str covers it
                log.warning("session %s: record could not be serialised", self.session.id)
        if not lines:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError as exc:
            # Losing the transcript must never kill a running turn.
            log.error("session %s: could not append (%s); continuing without persistence",
                      self.session.id, exc)
            self.writable = False
            return
        self.session.updated_at = now_ms()

    def append_message(self, message: dict[str, Any]) -> None:
        """Record one model-history message."""
        self.messages.append(message)
        self._write([{"kind": "message", "at": now_ms(), "message": message}])

    def append_messages(self, messages: Sequence[dict[str, Any]]) -> None:
        if not messages:
            return
        at = now_ms()
        self.messages.extend(messages)
        self._write([{"kind": "message", "at": at, "message": m} for m in messages])

    def replace_history(self, messages: Sequence[dict[str, Any]], *, reason: str = "compaction") -> None:
        """Record a rewrite of the whole history (compaction, ``/clear``)."""
        self.messages = list(messages)
        self._write([{"kind": "replace_history", "at": now_ms(), "reason": reason,
                      "messages": list(messages)}])

    def append_item(self, item: dict[str, Any]) -> None:
        """Record one display item (``AgentItem``)."""
        self.items.append(item)
        self._write([{"kind": "item", "at": now_ms(), "item": item}])

    def append_many(self, *, messages: Sequence[dict[str, Any]] = (),
                    items: Sequence[dict[str, Any]] = ()) -> None:
        """Append messages and items in one file write (same order as given)."""
        at = now_ms()
        records = [{"kind": "message", "at": at, "message": m} for m in messages]
        records += [{"kind": "item", "at": at, "item": i} for i in items]
        if not records:
            return
        self.messages.extend(messages)
        self.items.extend(items)
        self._write(records)

    # ------------------------------------------------------------------- meta
    def set_title(self, title: str) -> None:
        title = (title or "").strip()
        if not title or title == self.session.title:
            return
        self.session.title = title
        self._write([{"kind": "meta", "at": now_ms(), "title": title}])

    def title_from(self, user_text: str) -> None:
        """Set the title from the first user message (does nothing afterwards)."""
        if self.session.title:
            return
        self.set_title(derive_title(user_text))

    def set_model(self, model_key: str | None) -> None:
        if model_key == self.session.model_key:
            return
        self.session.model_key = model_key
        self._write([{"kind": "meta", "at": now_ms(), "modelKey": model_key}])

    def set_mode(self, mode: str) -> None:
        if mode == self.session.mode:
            return
        self.session.mode = mode
        self._write([{"kind": "meta", "at": now_ms(), "mode": mode}])

    # ------------------------------------------------------------------- misc
    def delete(self) -> bool:
        self.writable = False
        try:
            self.path.unlink()
        except (FileNotFoundError, OSError):
            return False
        return True

    def __repr__(self) -> str:
        return (f"SessionStore(id={self.session.id!r}, path={str(self.path)!r}, "
                f"messages={len(self.messages)}, items={len(self.items)})")


#: Unambiguous aliases for callers that import them next to other ``load``/``delete``
#: helpers (``localcode.agent`` re-exports these).
load_session = load
delete_session = delete
