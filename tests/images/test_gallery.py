"""The image gallery: the SQLite index, the file layout and their lifecycle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from localcode.config import Settings, data_dir
from localcode.images.db import SCHEMA_VERSION, ImageDB, ImageDBError
from localcode.images.gallery import (
    ImageGallery,
    ImageGalleryError,
    file_url,
    parse_size,
    settings_view,
)

from images_fakes import IMAGE_KEY, PNG_1X1, make_result, write_png


@pytest.fixture
def fresh(settings: Settings) -> ImageGallery:
    """A gallery on an empty temporary home (nothing else touches it)."""
    gallery = ImageGallery(settings).open()
    try:
        yield gallery
    finally:
        gallery.close()


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


def test_database_is_created_and_migrated(settings: Settings) -> None:
    with ImageDB(data_dir() / "images.db") as db:
        assert db.schema_version == SCHEMA_VERSION
        assert Path(db.path).exists()
        with db.read() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_reopening_does_not_migrate_twice(settings: Settings) -> None:
    path = data_dir() / "images.db"
    with ImageDB(path) as first:
        first.conn.execute("INSERT INTO images (id, path) VALUES ('a', '/tmp/a.png')")
    with ImageDB(path) as second:
        assert second.schema_version == SCHEMA_VERSION
        assert second.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 1


def test_a_newer_schema_is_refused(settings: Settings) -> None:
    path = data_dir() / "images.db"
    with ImageDB(path) as db:
        db.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    with pytest.raises(ImageDBError) as excinfo:
        ImageDB(path).open()
    assert "nowszej wersji" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Adding
# --------------------------------------------------------------------------- #


def test_add_moves_the_file_into_the_month_directory(fresh: ImageGallery, tmp_path: Path) -> None:
    result = make_result(tmp_path / "engine", prompt="kot w kapeluszu")
    source = Path(result.path)
    record = fresh.add(result)

    target = Path(record.path)
    assert target.exists() and not source.exists()
    assert target.parent.name == time.strftime("%Y-%m", time.localtime(result.created_at))
    assert target.parent.parent == fresh.root
    assert target.name == f"{record.id}.png"
    assert record.size_bytes == len(PNG_1X1)


def test_add_keeps_a_file_that_is_already_in_place(fresh: ImageGallery) -> None:
    result = make_result(fresh.root, image_id="abc123")
    target = fresh.path_for("abc123", result.created_at)
    write_png(target)
    result.path = str(target)

    record = fresh.add(result)
    assert Path(record.path) == target
    assert target.exists()


def test_add_stores_every_generation_parameter(fresh: ImageGallery, tmp_path: Path) -> None:
    result = make_result(tmp_path / "e", prompt="zamek nad jeziorem", seed=7, steps=25,
                         guidance=4.5, width=832, height=1216)
    record = fresh.add(result, model_name="FLUX.1 schnell", negative_prompt="rozmycie",
                       params={"count": 2})

    assert record.params["prompt"] == "zamek nad jeziorem"
    assert record.params["seed"] == 7
    assert record.params["guidance"] == 4.5
    assert record.params["negativePrompt"] == "rozmycie"
    assert record.params["count"] == 2
    stored = fresh.get(record.id)
    assert stored is not None
    assert stored.params == record.params
    assert stored.model_name == "FLUX.1 schnell"
    assert (stored.width, stored.height, stored.steps) == (832, 1216, 25)


def test_add_accepts_a_plain_dict(fresh: ImageGallery, tmp_path: Path) -> None:
    path = write_png(tmp_path / "raw.png")
    record = fresh.add({"id": "dict-1", "path": str(path), "width": 512, "height": 512,
                        "seed": 3, "modelKey": IMAGE_KEY, "prompt": "pies", "steps": 20,
                        "elapsedMs": 900, "createdAt": time.time()})
    assert record.model_key == IMAGE_KEY
    assert record.elapsed_ms == 900
    assert record.width == 512


def test_add_without_a_file_is_a_polish_error(fresh: ImageGallery, tmp_path: Path) -> None:
    result = make_result(tmp_path / "e")
    Path(result.path).unlink()
    with pytest.raises(ImageGalleryError) as excinfo:
        fresh.add(result)
    assert "nie istnieje" in str(excinfo.value)
    assert excinfo.value.status == 404


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


def _seed(gallery: ImageGallery, tmp_path: Path, count: int) -> list[str]:
    ids: list[str] = []
    now = time.time()
    for index in range(count):
        result = make_result(tmp_path / "e", prompt=f"obraz {index}",
                             created_at=now + index, seed=index)
        ids.append(gallery.add(result).id)
    return ids


def test_list_is_newest_first_and_paged(fresh: ImageGallery, tmp_path: Path) -> None:
    ids = _seed(fresh, tmp_path, 5)

    first_page = fresh.list(limit=2)
    assert [r.id for r in first_page] == list(reversed(ids))[:2]
    second_page = fresh.list(limit=2, offset=2)
    assert [r.id for r in second_page] == list(reversed(ids))[2:4]
    assert fresh.count() == 5
    assert len(fresh.list(limit=10_000)) == 5      # limit is clamped, never fails


def test_favorites(fresh: ImageGallery, tmp_path: Path) -> None:
    ids = _seed(fresh, tmp_path, 3)
    updated = fresh.set_favorite(ids[1], True)
    assert updated.favorite is True

    only = fresh.list(favorite_only=True)
    assert [r.id for r in only] == [ids[1]]
    assert fresh.count(favorite_only=True) == 1
    assert fresh.set_favorite(ids[1], False).favorite is False
    assert fresh.list(favorite_only=True) == []


def test_set_favorite_of_an_unknown_image(fresh: ImageGallery) -> None:
    with pytest.raises(ImageGalleryError) as excinfo:
        fresh.set_favorite("nie-ma", True)
    assert excinfo.value.status == 404


def test_get_and_require(fresh: ImageGallery, tmp_path: Path) -> None:
    [image_id] = _seed(fresh, tmp_path, 1)
    assert fresh.get(image_id) is not None
    assert fresh.get("nie-ma") is None
    with pytest.raises(ImageGalleryError):
        fresh.require("nie-ma")


# --------------------------------------------------------------------------- #
# Deleting
# --------------------------------------------------------------------------- #


def test_delete_removes_the_row_and_the_file(fresh: ImageGallery, tmp_path: Path) -> None:
    [image_id] = _seed(fresh, tmp_path, 1)
    path = Path(fresh.require(image_id).path)
    assert path.exists()

    assert fresh.delete(image_id) is True
    assert not path.exists()
    assert fresh.get(image_id) is None
    assert fresh.delete(image_id) is False


def test_delete_survives_a_missing_file(fresh: ImageGallery, tmp_path: Path) -> None:
    [image_id] = _seed(fresh, tmp_path, 1)
    Path(fresh.require(image_id).path).unlink()
    assert fresh.delete(image_id) is True
    assert fresh.count() == 0


def test_delete_all_empties_the_gallery(fresh: ImageGallery, tmp_path: Path) -> None:
    _seed(fresh, tmp_path, 4)
    assert fresh.delete_all() == 4
    assert fresh.count() == 0
    assert list(fresh.root.rglob("*.png")) == []


# --------------------------------------------------------------------------- #
# Statistics and DTOs
# --------------------------------------------------------------------------- #


def test_stats(fresh: ImageGallery, tmp_path: Path) -> None:
    assert fresh.stats() == {"count": 0, "favorites": 0, "sizeBytes": 0, "oldestAt": None,
                             "newestAt": None, "models": []}
    ids = _seed(fresh, tmp_path, 3)
    fresh.set_favorite(ids[0], True)

    stats = fresh.stats()
    assert stats["count"] == 3
    assert stats["favorites"] == 1
    assert stats["sizeBytes"] == 3 * len(PNG_1X1)
    assert stats["models"] == [{"modelKey": IMAGE_KEY, "count": 3}]
    assert stats["oldestAt"] < stats["newestAt"]


def test_dto_matches_the_documented_shape(fresh: ImageGallery, tmp_path: Path) -> None:
    result = make_result(tmp_path / "e", prompt="las we mgle", guidance=3.5, steps=20)
    record = fresh.add(result, model_name="FLUX.1 schnell", negative_prompt="tekst")
    dto = record.to_dict()

    # "originalPrompt" carries what the user typed when the prompt was translated
    # to English before generating (localcode/images/prompt.py); "" otherwise.
    assert set(dto) == {"id", "url", "path", "prompt", "originalPrompt", "negativePrompt",
                        "modelKey", "modelName", "width", "height", "seed", "steps", "guidance",
                        "favorite", "sizeBytes", "createdAt", "elapsedMs"}
    assert dto["url"] == file_url(record.id) == f"/api/images/{record.id}/file"
    assert dto["negativePrompt"] == "tekst"
    assert dto["guidance"] == 3.5
    assert dto["favorite"] is False
    assert dto["createdAt"] == int(record.created_at * 1000)     # milliseconds, like every DTO
    assert json.dumps(dto)                                        # JSON-serialisable


# --------------------------------------------------------------------------- #
# Async twins, settings and helpers
# --------------------------------------------------------------------------- #


async def test_async_api(fresh: ImageGallery, tmp_path: Path) -> None:
    record = await fresh.aadd(make_result(tmp_path / "e"))
    assert [r.id for r in await fresh.alist()] == [record.id]
    assert (await fresh.aget(record.id)).id == record.id
    assert (await fresh.aset_favorite(record.id, True)).favorite is True
    assert (await fresh.astats())["count"] == 1
    assert await fresh.adelete(record.id) is True
    assert await fresh.aget(record.id) is None


def test_output_dir_follows_the_settings(settings: Settings, tmp_path: Path) -> None:
    assert ImageGallery(settings).root == data_dir() / "images"

    class WithImages:
        images = type("S", (), {"output_dir": str(tmp_path / "gdzie-indziej")})()

    gallery = ImageGallery(WithImages())
    assert gallery.root == tmp_path / "gdzie-indziej"


def test_settings_view_defaults_without_the_section(settings: Settings) -> None:
    view = settings_view(settings)
    assert view.max_loaded >= 1
    assert view.size == (1024, 1024)
    assert settings_view(None).default_model is None


@pytest.mark.parametrize("text, expected", [("1024x1024", (1024, 1024)), ("832X1216", (832, 1216)),
                                            (" 512 x 768 ", (512, 768)), ("1024×768", (1024, 768))])
def test_parse_size(text: str, expected: tuple[int, int]) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "duży", "1024", "100x100", "9999x9999", "1023x1024"])
def test_parse_size_rejects_nonsense(text: str) -> None:
    with pytest.raises(ImageGalleryError):
        parse_size(text)
