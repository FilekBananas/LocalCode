"""SQLite storage of the image gallery (``~/.localcode/data/images.db``).

One file, WAL mode, a **single connection** guarded by a re-entrant lock — the
same shape as :mod:`localcode.chat.db`, only much smaller: the gallery is one
table plus its indexes. Async callers reach it through :meth:`ImageDB.run`
(``asyncio.to_thread``) so the event loop never blocks on disk I/O.

Schema changes are numbered migrations tracked by ``PRAGMA user_version``;
index ``i`` of :data:`MIGRATIONS` upgrades the file from version ``i`` to
``i + 1``. A database written by a newer LocalCode is refused instead of being
damaged.

Timestamps are unix **seconds** as ``REAL`` (that is what ``ImageResult`` uses),
booleans are ``0/1`` integers, and the full generation parameters of every image
are kept as a JSON object in ``params`` so an image can always be reproduced
even after new columns are added.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from localcode.config import data_dir

log = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "ImageDB",
    "ImageDBError",
    "default_db_path",
]


class ImageDBError(RuntimeError):
    """Database could not be opened or migrated (the message is Polish)."""

    code = "images_db"
    status = 500


def default_db_path() -> Path:
    return data_dir() / "images.db"


_SCHEMA_V1 = """
CREATE TABLE images (
  id              TEXT PRIMARY KEY,
  path            TEXT NOT NULL,
  prompt          TEXT NOT NULL DEFAULT '',
  negative_prompt TEXT NOT NULL DEFAULT '',
  model_key       TEXT NOT NULL DEFAULT '',
  model_name      TEXT NOT NULL DEFAULT '',
  width           INTEGER NOT NULL DEFAULT 0,
  height          INTEGER NOT NULL DEFAULT 0,
  seed            INTEGER NOT NULL DEFAULT 0,
  steps           INTEGER NOT NULL DEFAULT 0,
  guidance        REAL,
  favorite        INTEGER NOT NULL DEFAULT 0,
  size_bytes      INTEGER NOT NULL DEFAULT 0,
  created_at      REAL NOT NULL DEFAULT 0,
  elapsed_ms      INTEGER NOT NULL DEFAULT 0,
  params          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX images_created_idx ON images(created_at DESC, id DESC);
CREATE INDEX images_favorite_idx ON images(favorite, created_at DESC);
CREATE INDEX images_model_idx ON images(model_key);
"""

#: Ordered migrations. Index ``i`` upgrades ``user_version`` from ``i`` to ``i + 1``.
MIGRATIONS: list[str | Callable[[sqlite3.Connection], None]] = [_SCHEMA_V1]

SCHEMA_VERSION = len(MIGRATIONS)


class ImageDB:
    """Thread-safe single-connection SQLite wrapper for the gallery.

    * :meth:`transaction` — ``BEGIN IMMEDIATE`` … ``COMMIT`` (nested calls become
      savepoints), holding the lock for the whole block;
    * :meth:`read` — holds the lock without a transaction (autocommit reads);
    * :meth:`run` — run ``fn(conn, *args)`` in a worker thread under the lock.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: str = str(path) if path is not None else str(default_db_path())
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._depth = 0
        self._closed = False

    # ---------------------------------------------------------------- lifecycle
    def open(self) -> "ImageDB":
        with self._lock:
            if self._conn is not None:
                return self
            if self._closed:
                raise ImageDBError("Baza galerii obrazów została już zamknięta.")
            if self.path != ":memory:" and not self.path.startswith("file:"):
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            try:
                conn = sqlite3.connect(
                    self.path,
                    timeout=10.0,
                    isolation_level=None,        # autocommit; transactions are explicit
                    check_same_thread=False,     # guarded by self._lock
                    uri=self.path.startswith("file:"),
                )
            except sqlite3.Error as exc:
                raise ImageDBError(
                    f"Nie udało się otworzyć bazy galerii obrazów ({self.path}): {exc}") from exc
            conn.row_factory = sqlite3.Row
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

    def __enter__(self) -> "ImageDB":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # --------------------------------------------------------------- migrations
    @staticmethod
    def _user_version(conn: sqlite3.Connection) -> int:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = self._user_version(conn)
        if version > SCHEMA_VERSION:
            raise ImageDBError(
                "Baza galerii obrazów pochodzi z nowszej wersji LocalCode "
                f"(schemat {version}, obsługiwany {SCHEMA_VERSION}). Zaktualizuj aplikację.")
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
                raise ImageDBError("Nie udało się zaktualizować schematu bazy galerii obrazów "
                                   f"(krok {index + 1}): {exc}") from exc
            log.info("images db migrated to schema %d", index + 1)

    @property
    def schema_version(self) -> int:
        with self.read() as conn:
            return self._user_version(conn)

    # ------------------------------------------------------------------ access
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
                savepoint: str | None = None
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
                except sqlite3.Error:  # pragma: no cover - the connection is already broken
                    log.exception("images db rollback failed")
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
    """Split a schema script into complete statements."""
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
