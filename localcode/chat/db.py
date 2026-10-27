"""SQLite storage for the chatbot (port of ai_ux ``src/lib/db/schema.ts`` + ``client.ts``).

Design
------
* One file, ``data_dir()/chat.db``, in WAL mode with foreign keys on. The chat
  runs inside a single local process, so a **single connection** guarded by a
  re-entrant lock is enough: every repository call holds the lock for the whole
  statement group, which also makes multi-statement writes atomic. Async code
  reaches the database through :meth:`ChatDB.run` (``asyncio.to_thread``), so
  the event loop never blocks on disk I/O.
* Schema changes are numbered migrations tracked by ``PRAGMA user_version``.
  A database written by a newer LocalCode is refused instead of being damaged.
* Timestamps are integer **milliseconds** (exactly like ai_ux), booleans are
  ``0/1`` integers converted by the repository.
* Full-text search uses two FTS5 tables (``messages_fts``, ``kb_chunks_fts``)
  with ``tokenize='unicode61 remove_diacritics 2'`` and are kept in sync by
  triggers. ``remove_diacritics`` only folds letters that have a Unicode
  decomposition, so Polish ``ł`` (and a few others such as ``ø``/``đ``) would
  stay unfolded and ``"zazolc"`` would never find ``"Zażółć"``. The triggers
  therefore index a *folded copy* of the text (:data:`FOLD_MAP`, applied with
  nested ``replace()`` calls so the database stays usable from any SQLite
  client) and :func:`fold_text` applies the same mapping to queries. Every
  mapping is one character to one character, so the indexed copy has exactly
  the same length as the original — :func:`restore_snippet` relies on that to
  map an FTS5 ``snippet()`` back onto the original spelling.
* ``messages`` and ``kb_chunks`` carry an ``INTEGER PRIMARY KEY`` (``rid``)
  that the FTS rows use as their ``rowid``: implicit rowids are not stable
  across ``VACUUM``, an aliased primary key is. The public identifier stays the
  text ``id`` column (UUID), which child tables reference.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from localcode.config import data_dir

log = logging.getLogger(__name__)

T = TypeVar("T")

# --------------------------------------------------------------------------- #
# Text folding shared by the FTS triggers and the query builder
# --------------------------------------------------------------------------- #

#: Letters that ``unicode61 remove_diacritics 2`` leaves alone because they have
#: no canonical decomposition. Every entry maps ONE character to ONE character.
FOLD_MAP: dict[str, str] = {
    "ł": "l", "Ł": "L",
    "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D",
    "ħ": "h", "Ħ": "H",
    "ŧ": "t", "Ŧ": "T",
    "ı": "i",
}
_FOLD_TABLE = str.maketrans(FOLD_MAP)

FTS_TOKENIZE = "unicode61 remove_diacritics 2"


def fold_text(text: str) -> str:
    """Apply :data:`FOLD_MAP` (length preserving)."""
    return text.translate(_FOLD_TABLE)


def _fold_sql(expr: str) -> str:
    """SQL expression applying :data:`FOLD_MAP` to ``expr`` with nested ``replace``."""
    out = expr
    for src, dst in FOLD_MAP.items():
        out = f"replace({out}, '{src}', '{dst}')"
    return out


# Private-use markers for snippet(); they never occur in normal text.
SNIPPET_OPEN = ""
SNIPPET_CLOSE = ""
SNIPPET_ELLIPSIS = ""


def restore_snippet(raw: str | None, folded: str, original: str,
                    open_tag: str = "<mark>", close_tag: str = "</mark>") -> str:
    """Turn an FTS5 ``snippet()`` computed over the folded copy into one that
    shows the ORIGINAL characters (``ł`` instead of ``l``) with ``<mark>`` tags.

    ``raw`` must have been produced with :data:`SNIPPET_OPEN`,
    :data:`SNIPPET_CLOSE` and :data:`SNIPPET_ELLIPSIS` as markers.
    """
    if not raw:
        return ""
    lead = raw.startswith(SNIPPET_ELLIPSIS)
    trail = raw.endswith(SNIPPET_ELLIPSIS)
    body = raw.strip(SNIPPET_ELLIPSIS)
    plain = body.replace(SNIPPET_OPEN, "").replace(SNIPPET_CLOSE, "")
    pos = folded.find(plain) if plain else -1
    out: list[str] = []
    if pos >= 0 and len(folded) == len(original):
        i = pos
        for ch in body:
            if ch == SNIPPET_OPEN:
                out.append(open_tag)
            elif ch == SNIPPET_CLOSE:
                out.append(close_tag)
            else:
                out.append(original[i])
                i += 1
        text = "".join(out)
    else:  # should not happen; fall back to the folded snippet
        text = body.replace(SNIPPET_OPEN, open_tag).replace(SNIPPET_CLOSE, close_tag)
    return ("…" if lead else "") + text + ("…" if trail else "")


# --------------------------------------------------------------------------- #
# Small helpers shared with the repository
# --------------------------------------------------------------------------- #


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return str(uuid.uuid4())


def default_db_path() -> Path:
    return data_dir() / "chat.db"


def uploads_dir() -> Path:
    return data_dir() / "uploads"


class ChatDBError(RuntimeError):
    """Storage failure with a Polish, user-presentable message."""


# --------------------------------------------------------------------------- #
# Schema and migrations
# --------------------------------------------------------------------------- #

_SCHEMA_V1 = f"""
CREATE TABLE folders (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  description   TEXT,
  color         TEXT,
  icon          TEXT,
  system_prompt TEXT,
  extra_info    TEXT,
  memory_mode   TEXT NOT NULL DEFAULT 'shared' CHECK (memory_mode IN ('shared','isolated')),
  sort_order    INTEGER NOT NULL DEFAULT 0,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

CREATE TABLE conversations (
  id              TEXT PRIMARY KEY,
  folder_id       TEXT REFERENCES folders(id) ON DELETE SET NULL,
  title           TEXT NOT NULL DEFAULT 'Nowa rozmowa',
  title_locked    INTEGER NOT NULL DEFAULT 0,
  provider        TEXT,
  model           TEXT,
  system_prompt   TEXT,
  temperature     REAL,
  max_tokens      INTEGER,
  pinned          INTEGER NOT NULL DEFAULT 0,
  archived        INTEGER NOT NULL DEFAULT 0,
  tokens_in       INTEGER NOT NULL DEFAULT 0,
  tokens_out      INTEGER NOT NULL DEFAULT 0,
  cost_usd        REAL NOT NULL DEFAULT 0,
  created_at      INTEGER NOT NULL,
  updated_at      INTEGER NOT NULL,
  last_message_at INTEGER NOT NULL
);
CREATE INDEX idx_conv_folder ON conversations(folder_id);
CREATE INDEX idx_conv_list ON conversations(archived, pinned, last_message_at);

CREATE TABLE messages (
  rid             INTEGER PRIMARY KEY,
  id              TEXT NOT NULL UNIQUE,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
  content         TEXT NOT NULL DEFAULT '',
  reasoning       TEXT,
  provider        TEXT,
  model           TEXT,
  status          TEXT NOT NULL DEFAULT 'complete'
                    CHECK (status IN ('streaming','complete','error','stopped')),
  error           TEXT,
  tokens_in       INTEGER,
  tokens_out      INTEGER,
  cost_usd        REAL,
  latency_ms      INTEGER,
  citations       TEXT,
  seq             INTEGER NOT NULL,
  created_at      INTEGER NOT NULL
);
CREATE INDEX idx_msg_conv ON messages(conversation_id, seq);

CREATE TABLE attachments (
  id              TEXT PRIMARY KEY,
  message_id      TEXT REFERENCES messages(id) ON DELETE CASCADE,
  conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  mime            TEXT NOT NULL,
  size            INTEGER NOT NULL DEFAULT 0,
  kind            TEXT NOT NULL DEFAULT 'other' CHECK (kind IN ('image','text','pdf','other')),
  path            TEXT,
  text_content    TEXT,
  created_at      INTEGER NOT NULL
);
CREATE INDEX idx_att_msg ON attachments(message_id);
CREATE INDEX idx_att_conv ON attachments(conversation_id);

CREATE TABLE memories (
  id         TEXT PRIMARY KEY,
  scope      TEXT NOT NULL DEFAULT 'global' CHECK (scope IN ('global','folder')),
  folder_id  TEXT REFERENCES folders(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL DEFAULT 'fact' CHECK (kind IN ('fact','preference','note')),
  content    TEXT NOT NULL,
  source     TEXT NOT NULL DEFAULT 'user' CHECK (source IN ('user','auto')),
  pinned     INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX idx_mem_scope ON memories(scope, folder_id);

CREATE TABLE kb_documents (
  id          TEXT PRIMARY KEY,
  scope       TEXT NOT NULL DEFAULT 'folder' CHECK (scope IN ('global','folder')),
  folder_id   TEXT REFERENCES folders(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  mime        TEXT NOT NULL,
  size        INTEGER NOT NULL DEFAULT 0,
  path        TEXT,
  status      TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing','ready','error')),
  error       TEXT,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX idx_kbdoc_folder ON kb_documents(folder_id);

CREATE TABLE kb_chunks (
  rid             INTEGER PRIMARY KEY,
  id              TEXT NOT NULL UNIQUE,
  document_id     TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
  folder_id       TEXT,
  scope           TEXT NOT NULL DEFAULT 'folder',
  idx             INTEGER NOT NULL,
  content         TEXT NOT NULL,
  token_estimate  INTEGER NOT NULL DEFAULT 0,
  embedding       BLOB,
  embedding_model TEXT,
  embedding_dim   INTEGER
);
CREATE INDEX idx_kbchunk_doc ON kb_chunks(document_id, idx);
CREATE INDEX idx_kbchunk_scope ON kb_chunks(scope, folder_id);
CREATE INDEX idx_kbchunk_emb ON kb_chunks(embedding_model);

CREATE TABLE settings (
  id         INTEGER PRIMARY KEY CHECK (id = 1),
  value      TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE VIRTUAL TABLE messages_fts USING fts5(content, tokenize='{FTS_TOKENIZE}');
CREATE VIRTUAL TABLE kb_chunks_fts USING fts5(content, tokenize='{FTS_TOKENIZE}');

CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.rid, {_fold_sql('new.content')});
END;
CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.rid;
END;
CREATE TRIGGER messages_fts_au AFTER UPDATE OF content ON messages
  WHEN old.content IS NOT new.content BEGIN
  DELETE FROM messages_fts WHERE rowid = old.rid;
  INSERT INTO messages_fts(rowid, content) VALUES (new.rid, {_fold_sql('new.content')});
END;

CREATE TRIGGER kb_chunks_fts_ai AFTER INSERT ON kb_chunks BEGIN
  INSERT INTO kb_chunks_fts(rowid, content) VALUES (new.rid, {_fold_sql('new.content')});
END;
CREATE TRIGGER kb_chunks_fts_ad AFTER DELETE ON kb_chunks BEGIN
  DELETE FROM kb_chunks_fts WHERE rowid = old.rid;
END;
CREATE TRIGGER kb_chunks_fts_au AFTER UPDATE OF content ON kb_chunks
  WHEN old.content IS NOT new.content BEGIN
  DELETE FROM kb_chunks_fts WHERE rowid = old.rid;
  INSERT INTO kb_chunks_fts(rowid, content) VALUES (new.rid, {_fold_sql('new.content')});
END;
"""

#: Conversation compaction (:mod:`localcode.chat.compaction`): the model-written
#: summary of the oldest turns, the ``seq`` it covers up to (inclusive) and its
#: size in tokens. ``summary_upto_seq = 0`` means "no summary yet".
_SCHEMA_V2 = """
ALTER TABLE conversations ADD COLUMN summary TEXT;
ALTER TABLE conversations ADD COLUMN summary_upto_seq INTEGER NOT NULL DEFAULT 0;
ALTER TABLE conversations ADD COLUMN summary_tokens INTEGER NOT NULL DEFAULT 0;
"""

#: Per-conversation reasoning-effort override ("off"|"low"|"medium"|"high");
#: NULL means "follow ``settings.chat.reasoning_effort``". Models whose chat
#: template has no effort control ignore it.
_SCHEMA_V3 = """
ALTER TABLE conversations ADD COLUMN reasoning_effort TEXT;
"""

#: Ordered migrations. Index ``i`` upgrades ``user_version`` from ``i`` to ``i + 1``.
MIGRATIONS: list[str | Callable[[sqlite3.Connection], None]] = [_SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3]

SCHEMA_VERSION = len(MIGRATIONS)


# --------------------------------------------------------------------------- #
# Connection wrapper
# --------------------------------------------------------------------------- #


class ChatDB:
    """Thread-safe single-connection SQLite wrapper.

    * :meth:`transaction` — ``BEGIN IMMEDIATE`` … ``COMMIT`` (nested calls become
      savepoints), holding the lock for the whole block.
    * :meth:`read` — holds the lock, no transaction (autocommit reads).
    * :meth:`run` — run a callable ``fn(conn, *args)`` in a worker thread under the
      lock; the way async code talks to the database.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: str = str(path) if path is not None else str(default_db_path())
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._depth = 0
        self._closed = False

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> "ChatDB":
        with self._lock:
            if self._conn is not None:
                return self
            if self._closed:
                raise ChatDBError("Baza czatu została już zamknięta.")
            if self.path != ":memory:" and not self.path.startswith("file:"):
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            try:
                conn = sqlite3.connect(
                    self.path,
                    timeout=10.0,
                    isolation_level=None,          # autocommit; transactions are explicit
                    check_same_thread=False,       # guarded by self._lock
                    uri=self.path.startswith("file:"),
                )
            except sqlite3.Error as exc:
                raise ChatDBError(f"Nie udało się otworzyć bazy czatu ({self.path}): {exc}") from exc
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 10000")
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
            try:
                self._migrate(conn)
            except Exception:
                conn.close()
                self._conn = None
                raise
            return self

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
            self._closed = True

    def __enter__(self) -> "ChatDB":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # ---------------------------------------------------------------- migrations
    @staticmethod
    def _user_version(conn: sqlite3.Connection) -> int:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = self._user_version(conn)
        if version > SCHEMA_VERSION:
            raise ChatDBError(
                "Baza czatu pochodzi z nowszej wersji LocalCode "
                f"(schemat {version}, obsługiwany {SCHEMA_VERSION}). Zaktualizuj aplikację."
            )
        for index in range(version, SCHEMA_VERSION):
            step = MIGRATIONS[index]
            try:
                conn.execute("BEGIN IMMEDIATE")
                if isinstance(step, str):
                    for statement in _split_sql(step):
                        conn.execute(statement)
                else:
                    step(conn)
                conn.execute(f"PRAGMA user_version = {index + 1}")
                conn.execute("COMMIT")
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise ChatDBError(f"Nie udało się zaktualizować schematu bazy czatu (krok {index + 1}): {exc}") from exc
            log.info("chat db migrated to schema %d", index + 1)

    @property
    def schema_version(self) -> int:
        with self.read() as conn:
            return self._user_version(conn)

    # ---------------------------------------------------------------- access
    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self.conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self.conn
            if self._depth == 0:
                conn.execute("BEGIN IMMEDIATE")
                savepoint = None
            else:
                savepoint = f"sp_{self._depth}"
                conn.execute(f"SAVEPOINT {savepoint}")
            self._depth += 1
            try:
                yield conn
            except BaseException:
                self._depth -= 1
                try:
                    if savepoint is None:
                        conn.execute("ROLLBACK")
                    else:
                        conn.execute(f"ROLLBACK TO {savepoint}")
                        conn.execute(f"RELEASE {savepoint}")
                except sqlite3.Error:
                    log.exception("chat db rollback failed")
                raise
            else:
                self._depth -= 1
                if savepoint is None:
                    conn.execute("COMMIT")
                else:
                    conn.execute(f"RELEASE {savepoint}")

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn(conn, *args, **kwargs)`` under the lock (synchronously)."""
        with self._lock:
            return fn(self.conn, *args, **kwargs)

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn(conn, *args, **kwargs)`` in a worker thread under the lock."""
        return await asyncio.to_thread(self.call, fn, *args, **kwargs)


def _split_sql(script: str) -> list[str]:
    """Split a schema script into statements, keeping trigger bodies intact."""
    statements: list[str] = []
    buf: list[str] = []
    for line in script.splitlines():
        buf.append(line)
        candidate = "\n".join(buf).strip()
        if candidate and sqlite3.complete_statement(candidate):
            statements.append(candidate)
            buf = []
    rest = "\n".join(buf).strip()
    if rest:
        statements.append(rest)
    return statements
