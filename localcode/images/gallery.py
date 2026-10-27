"""The gallery of generated images: files on disk + their parameters in SQLite.

Layout (everything under ``LOCALCODE_HOME``, default ``~/.localcode``)::

    data/images.db                 the index (see localcode.images.db)
    data/images/2026-09/<id>.png   the files, one directory per month

:class:`ImageGallery` owns both halves, so a row and its file are created and
removed together: :meth:`ImageGallery.add` moves whatever the image runtime
produced into the month directory, :meth:`ImageGallery.delete` removes the row
*and* the file.

``add`` accepts an ``ImageResult`` (see ``docs/IMAGES.md``) or a plain dict with
the same fields, which keeps this module independent of
:mod:`localcode.images.runtime` — the gallery works (and is tested) with a fake
engine, and the image extra (``pip install "localcode[images]"``) is only needed
to *generate* an image, never to browse one.

The DTO handed to the HTTP API and the Mac app is the ``ImageDTO`` of
``docs/IMAGES.md`` plus one extra field, ``path`` (the absolute path on this
machine), which is what the terminal prints after ``/obraz``. ``createdAt`` is
in **milliseconds** there, like every other LocalCode DTO (and like
``ImageResult.to_dict``), while the column and ``ImageRecord`` keep the unix
seconds that ``ImageResult`` uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from localcode import config as lc_config
from localcode.images.db import ImageDB

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SIZE",
    "ImageGallery",
    "ImageGalleryError",
    "ImageRecord",
    "ImageSettingsView",
    "file_url",
    "parse_size",
    "settings_view",
]

#: Fallback when ``settings.images`` does not exist yet (the extra is optional).
DEFAULT_SIZE = "1024x1024"
#: Largest page the gallery will return in one call.
MAX_PAGE = 200
#: Smallest / largest image edge accepted by :func:`parse_size`.
MIN_EDGE, MAX_EDGE = 128, 4096


class ImageGalleryError(RuntimeError):
    """A gallery operation failed; the message is Polish and safe to show."""

    def __init__(self, message: str, code: str = "images_error", status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


# --------------------------------------------------------------------------- #
# Settings (read defensively: config.ImageSettings is optional)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImageSettingsView:
    """``Settings.images`` with defaults for installations that predate it."""

    default_model: str | None = None
    max_loaded: int = 1
    idle_ttl_minutes: int = 10
    output_dir: str = ""
    default_steps: int | None = None
    default_size: str = DEFAULT_SIZE
    default_guidance: float | None = None
    keep_metadata: bool = True

    @property
    def size(self) -> tuple[int, int]:
        return parse_size(self.default_size or DEFAULT_SIZE)


def settings_view(settings: Any | None) -> ImageSettingsView:
    """Read ``settings.images`` without requiring it to exist."""
    section = getattr(settings, "images", None)
    if section is None:
        return ImageSettingsView()
    defaults = ImageSettingsView()

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(section, name, None)
        return fallback if value is None else value

    return ImageSettingsView(
        default_model=getattr(section, "default_model", None),
        max_loaded=int(pick("max_loaded", defaults.max_loaded)),
        idle_ttl_minutes=int(pick("idle_ttl_minutes", defaults.idle_ttl_minutes)),
        output_dir=str(pick("output_dir", defaults.output_dir)),
        default_steps=getattr(section, "default_steps", None),
        default_size=str(pick("default_size", defaults.default_size)),
        default_guidance=getattr(section, "default_guidance", None),
        keep_metadata=bool(pick("keep_metadata", defaults.keep_metadata)),
    )


def parse_size(text: str) -> tuple[int, int]:
    """``"1024x1024"`` → ``(1024, 1024)``; raises :class:`ImageGalleryError`."""
    raw = (text or "").strip().lower().replace("×", "x").replace("*", "x")
    if "x" not in raw:
        raise ImageGalleryError(f"Nieprawidłowy rozmiar „{text}”. Podaj np. 1024x1024.",
                                "invalid_size")
    left, _, right = raw.partition("x")
    try:
        width, height = int(left.strip()), int(right.strip())
    except ValueError:
        raise ImageGalleryError(f"Nieprawidłowy rozmiar „{text}”. Podaj np. 1024x1024.",
                                "invalid_size") from None
    for value in (width, height):
        if value < MIN_EDGE or value > MAX_EDGE:
            raise ImageGalleryError(
                f"Rozmiar obrazu musi mieścić się między {MIN_EDGE} a {MAX_EDGE} pikselami.",
                "invalid_size")
        if value % 8:
            raise ImageGalleryError("Szerokość i wysokość muszą być wielokrotnością 8 pikseli.",
                                    "invalid_size")
    return width, height


def file_url(image_id: str) -> str:
    """Relative URL of an image's bytes, as used in the ``ImageDTO``."""
    return f"/api/images/{image_id}/file"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class ImageRecord:
    """One row of the gallery — an image and everything it was made with."""

    id: str
    path: str
    prompt: str = ""
    negative_prompt: str = ""
    model_key: str = ""
    model_name: str = ""
    width: int = 0
    height: int = 0
    seed: int = 0
    steps: int = 0
    guidance: float | None = None
    favorite: bool = False
    size_bytes: int = 0
    created_at: float = 0.0
    elapsed_ms: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    #: what the user typed, when the prompt was translated to English first
    original_prompt: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ImageRecord":
        try:
            params = json.loads(row["params"] or "{}")
        except ValueError:  # pragma: no cover - only a hand-edited database
            params = {}
        return cls(
            id=str(row["id"]),
            path=str(row["path"]),
            prompt=str(row["prompt"] or ""),
            original_prompt=str((params or {}).get("originalPrompt") or ""),
            negative_prompt=str(row["negative_prompt"] or ""),
            model_key=str(row["model_key"] or ""),
            model_name=str(row["model_name"] or ""),
            width=int(row["width"] or 0),
            height=int(row["height"] or 0),
            seed=int(row["seed"] or 0),
            steps=int(row["steps"] or 0),
            guidance=None if row["guidance"] is None else float(row["guidance"]),
            favorite=bool(row["favorite"]),
            size_bytes=int(row["size_bytes"] or 0),
            created_at=float(row["created_at"] or 0.0),
            elapsed_ms=int(row["elapsed_ms"] or 0),
            params=params if isinstance(params, dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """``ImageDTO`` (camelCase) — see ``docs/IMAGES.md``; ``path`` is an extra."""
        return {
            "id": self.id,
            "url": file_url(self.id),
            "path": self.path,
            "prompt": self.prompt,
            "originalPrompt": self.original_prompt,
            "negativePrompt": self.negative_prompt,
            "modelKey": self.model_key,
            "modelName": self.model_name,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "steps": self.steps,
            "guidance": self.guidance,
            "favorite": self.favorite,
            "sizeBytes": self.size_bytes,
            "createdAt": int(self.created_at * 1000),
            "elapsedMs": self.elapsed_ms,
        }


def _get(source: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an ``ImageResult``-like object or a snake/camel dict."""
    if isinstance(source, dict):
        if name in source:
            return source[name]
        head, *rest = name.split("_")
        camel = head + "".join(part.title() for part in rest)
        return source.get(camel, default)
    value = getattr(source, name, None)
    return default if value is None else value


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #


class ImageGallery:
    """Index of every generated image, with its file and its parameters.

        gallery = ImageGallery(settings).open()
        record = gallery.add(result)          # moves the PNG into data/images/<yyyy-mm>/
        gallery.list(limit=20)                # newest first, paged
        gallery.delete(record.id)             # row and file

    Every method has an ``a``-prefixed coroutine twin (:meth:`aadd`,
    :meth:`alist`, …) that runs the same work in a worker thread, which is what
    the HTTP handlers use.
    """

    def __init__(self, settings: Any | None = None, *, db: ImageDB | None = None,
                 db_path: str | Path | None = None, root: str | Path | None = None) -> None:
        self.settings = settings
        self.images = settings_view(settings)
        self.db = db if db is not None else ImageDB(db_path)
        self._root = Path(root).expanduser() if root is not None else _output_dir(self.images)

    # ------------------------------------------------------------- lifecycle
    def open(self) -> "ImageGallery":
        self.db.open()
        self._root.mkdir(parents=True, exist_ok=True)
        return self

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ImageGallery":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def root(self) -> Path:
        """Directory that holds the month directories with the image files."""
        return self._root

    def path_for(self, image_id: str, created_at: float | None = None) -> Path:
        """Where an image with this id belongs: ``<root>/<yyyy-mm>/<id>.png``."""
        stamp = time.localtime(created_at if created_at else time.time())
        return self._root / time.strftime("%Y-%m", stamp) / f"{image_id}.png"

    # ------------------------------------------------------------------- add
    def add(self, result: Any, *, model_name: str = "", negative_prompt: str = "",
            params: dict[str, Any] | None = None, move: bool = True) -> ImageRecord:
        """Store one generated image (an ``ImageResult`` or an equivalent dict).

        The file is moved into ``<root>/<yyyy-mm>/<id>.png`` unless it is already
        there (``move=False`` copies instead of moving, for a file the caller
        still owns).
        """
        image_id = str(_get(result, "id", "") or uuid.uuid4().hex)
        source = Path(str(_get(result, "path", "") or "")).expanduser()
        if not str(source):
            raise ImageGalleryError("Wynik generowania nie zawiera ścieżki do pliku.",
                                    "invalid_result")
        created_at = float(_get(result, "created_at", 0.0) or time.time())
        target = self._store_file(source, image_id, created_at, move=move)
        record = ImageRecord(
            id=image_id,
            path=str(target),
            prompt=str(_get(result, "prompt", "") or ""),
            original_prompt=str(_get(result, "original_prompt", "") or ""),
            negative_prompt=str(negative_prompt or _get(result, "negative_prompt", "") or ""),
            model_key=str(_get(result, "model_key", "") or ""),
            model_name=str(model_name or _get(result, "model_name", "") or ""),
            width=int(_get(result, "width", 0) or 0),
            height=int(_get(result, "height", 0) or 0),
            seed=int(_get(result, "seed", 0) or 0),
            steps=int(_get(result, "steps", 0) or 0),
            guidance=_as_float(_get(result, "guidance", None)),
            favorite=False,
            size_bytes=_size_of(target),
            created_at=created_at,
            elapsed_ms=int(_get(result, "elapsed_ms", 0) or 0),
            params={},
        )
        record.params = _merge_params(record, params)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO images (id, path, prompt, negative_prompt, model_key,"
                " model_name, width, height, seed, steps, guidance, favorite, size_bytes,"
                " created_at, elapsed_ms, params)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.id, record.path, record.prompt, record.negative_prompt, record.model_key,
                 record.model_name, record.width, record.height, record.seed, record.steps,
                 record.guidance, int(record.favorite), record.size_bytes, record.created_at,
                 record.elapsed_ms, json.dumps(record.params, ensure_ascii=False)),
            )
        return record

    def _store_file(self, source: Path, image_id: str, created_at: float, *, move: bool) -> Path:
        target = self.path_for(image_id, created_at)
        try:
            same = source.exists() and target.exists() and source.samefile(target)
        except OSError:  # pragma: no cover - a race with an external deletion
            same = False
        if same or str(source) == str(target):
            return target
        if not source.exists():
            raise ImageGalleryError(f"Plik obrazu „{source}” nie istnieje.", "file_missing", 404)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if move:
                try:
                    os.replace(source, target)
                except OSError:                      # across filesystems
                    shutil.copy2(source, target)
                    source.unlink(missing_ok=True)
            else:
                shutil.copy2(source, target)
        except OSError as exc:
            raise ImageGalleryError(f"Nie udało się zapisać obrazu w galerii: {exc}",
                                    "write_failed", 500) from exc
        return target

    # ------------------------------------------------------------------ read
    def list(self, limit: int = 50, offset: int = 0, favorite_only: bool = False, *,
             model_key: str | None = None) -> list[ImageRecord]:
        """Newest first, paged. ``limit`` is clamped to :data:`MAX_PAGE`."""
        count = max(1, min(int(limit or 1), MAX_PAGE))
        skip = max(0, int(offset or 0))
        where: list[str] = []
        args: list[Any] = []
        if favorite_only:
            where.append("favorite = 1")
        if model_key:
            where.append("model_key = ?")
            args.append(model_key)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self.db.read() as conn:
            rows = conn.execute(
                f"SELECT * FROM images{clause} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (*args, count, skip)).fetchall()
        return [ImageRecord.from_row(row) for row in rows]

    def get(self, image_id: str) -> ImageRecord | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM images WHERE id = ?", (str(image_id),)).fetchone()
        return ImageRecord.from_row(row) if row is not None else None

    def require(self, image_id: str) -> ImageRecord:
        record = self.get(image_id)
        if record is None:
            raise ImageGalleryError(f"Nie znaleziono obrazu „{image_id}”.", "not_found", 404)
        return record

    def count(self, *, favorite_only: bool = False) -> int:
        clause = " WHERE favorite = 1" if favorite_only else ""
        with self.db.read() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM images{clause}").fetchone()[0])

    def stats(self) -> dict[str, Any]:
        """Totals for the gallery header: how many images, how much disk, which models."""
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(size_bytes), 0) AS bytes,"
                " COALESCE(SUM(favorite), 0) AS favorites,"
                " MIN(created_at) AS oldest, MAX(created_at) AS newest FROM images").fetchone()
            models = conn.execute(
                "SELECT model_key, COUNT(*) AS n FROM images GROUP BY model_key"
                " ORDER BY n DESC, model_key").fetchall()
        return {
            "count": int(row["n"] or 0),
            "favorites": int(row["favorites"] or 0),
            "sizeBytes": int(row["bytes"] or 0),
            "oldestAt": float(row["oldest"]) if row["oldest"] is not None else None,
            "newestAt": float(row["newest"]) if row["newest"] is not None else None,
            "models": [{"modelKey": str(m["model_key"] or ""), "count": int(m["n"])}
                       for m in models],
        }

    # ----------------------------------------------------------------- write
    def set_favorite(self, image_id: str, favorite: bool) -> ImageRecord:
        with self.db.transaction() as conn:
            cursor = conn.execute("UPDATE images SET favorite = ? WHERE id = ?",
                                  (1 if favorite else 0, str(image_id)))
            if cursor.rowcount == 0:
                raise ImageGalleryError(f"Nie znaleziono obrazu „{image_id}”.", "not_found", 404)
        return self.require(image_id)

    def delete(self, image_id: str) -> bool:
        """Remove the row and the file; ``False`` when the id is unknown."""
        record = self.get(image_id)
        if record is None:
            return False
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM images WHERE id = ?", (record.id,))
        _unlink(Path(record.path))
        return True

    def delete_all(self) -> int:
        """Empty the gallery (used by ``localcode images rm --all``)."""
        records = self.list(limit=MAX_PAGE, offset=0)
        removed = 0
        while records:
            for record in records:
                if self.delete(record.id):
                    removed += 1
            records = self.list(limit=MAX_PAGE, offset=0)
        return removed

    # ----------------------------------------------------------------- async
    async def aadd(self, result: Any, **kwargs: Any) -> ImageRecord:
        return await asyncio.to_thread(lambda: self.add(result, **kwargs))

    async def alist(self, limit: int = 50, offset: int = 0, favorite_only: bool = False,
                    **kwargs: Any) -> list[ImageRecord]:
        return await asyncio.to_thread(lambda: self.list(limit, offset, favorite_only, **kwargs))

    async def aget(self, image_id: str) -> ImageRecord | None:
        return await asyncio.to_thread(self.get, image_id)

    async def adelete(self, image_id: str) -> bool:
        return await asyncio.to_thread(self.delete, image_id)

    async def aset_favorite(self, image_id: str, favorite: bool) -> ImageRecord:
        return await asyncio.to_thread(self.set_favorite, image_id, favorite)

    async def astats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.stats)

    def dtos(self, records: Iterable[ImageRecord]) -> list[dict[str, Any]]:
        return [record.to_dict() for record in records]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _output_dir(images: ImageSettingsView) -> Path:
    """``settings.images.output_dir`` resolved against ``LOCALCODE_HOME``."""
    raw = (images.output_dir or "").strip()
    if not raw:
        return lc_config.data_dir() / "images"
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (lc_config.home() / path)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _size_of(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a read-only directory
        log.warning("nie udało się usunąć pliku obrazu %s", path, exc_info=True)


def _merge_params(record: ImageRecord, extra: dict[str, Any] | None) -> dict[str, Any]:
    """Everything needed to generate this image again, in camelCase."""
    params: dict[str, Any] = {
        "prompt": record.prompt,
        "originalPrompt": record.original_prompt,
        "negativePrompt": record.negative_prompt,
        "modelKey": record.model_key,
        "width": record.width,
        "height": record.height,
        "steps": record.steps,
        "guidance": record.guidance,
        "seed": record.seed,
    }
    if extra:
        params.update({k: v for k, v in extra.items() if _jsonable(v)})
    return params


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
