"""The HTTP surface of image generation: ``/api/images*`` and ``/v1/images/generations``."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient

from localcode.images.base import ImageEngineError
from localcode.server.core import Core

from images_fakes import IMAGE_KEY, IMAGE_KEY_2, PNG_1X1, FakeImageRuntime, make_result

PROMPT = "kot w kapeluszu na dachu"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def read_frames(response: Any, *, limit: int = 500) -> list[dict[str, Any]]:
    """Every ``data:`` record of an SSE response, as dicts."""
    frames: list[dict[str, Any]] = []
    async for raw in response.content:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        frames.append(json.loads(line[5:].strip()))
        if len(frames) >= limit or frames[-1].get("type") in ("done", "error"):
            break
    return frames


async def generate(client: TestClient, **body: Any) -> list[dict[str, Any]]:
    payload = {"prompt": PROMPT, **body}
    async with client.post("/api/images/generate", json=payload) as response:
        assert response.status == 200, await response.text()
        assert response.headers["Content-Type"].startswith("text/event-stream")
        return await read_frames(response)


def kinds(frames: list[dict[str, Any]]) -> list[str]:
    return [str(f.get("type")) for f in frames]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


async def test_models_lists_only_image_models_with_defaults(client: TestClient) -> None:
    response = await client.get("/api/images/models")
    assert response.status == 200
    models = await response.json()

    assert [m["key"] for m in models] == [IMAGE_KEY, IMAGE_KEY_2]
    flux = models[0]
    assert flux["imageFamily"] == "flux"
    assert flux["defaults"]["steps"] == 4                  # schnell is a 4-step model
    assert "1024x1024" in flux["defaults"]["sizes"]
    assert flux["loaded"] is False
    sd = models[1]
    assert sd["imageFamily"] == "sd1"
    assert sd["defaults"]["guidance"] == 7.0
    assert "512x512" in sd["defaults"]["sizes"]


async def test_every_image_route_needs_the_token(anon_client: TestClient) -> None:
    for method, path in (("GET", "/api/images/models"), ("GET", "/api/images"),
                         ("GET", "/api/images/stats"), ("POST", "/api/images/generate"),
                         ("GET", "/api/images/abc/file"), ("PATCH", "/api/images/abc"),
                         ("DELETE", "/api/images/abc")):
        response = await anon_client.request(method, path, json={})
        assert response.status == 401, f"{method} {path}"
        assert (await response.json())["code"] == "unauthorized"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


async def test_generate_streams_progress_and_stores_the_image(
        client: TestClient, core: Core, image_runtime: FakeImageRuntime) -> None:
    frames = await generate(client, modelKey=IMAGE_KEY, width=512, height=512, steps=2, seed=11)

    assert kinds(frames)[0] == "status"
    assert kinds(frames)[-1] == "done"
    progress = [f for f in frames if f["type"] == "progress"]
    assert [p["step"] for p in progress] == [1, 2]
    assert all(p["steps"] == 2 and p["index"] == 0 for p in progress)

    images = [f["image"] for f in frames if f["type"] == "image"]
    assert len(images) == 1
    image = images[0]
    assert image["prompt"] == PROMPT
    assert (image["width"], image["height"], image["seed"]) == (512, 512, 11)
    assert image["url"] == f"/api/images/{image['id']}/file"
    assert Path(image["path"]).is_file()

    stored = core.gallery.get(image["id"])
    assert stored is not None and stored.model_name == "FLUX.1 schnell"
    assert image_runtime.calls[0][0] == IMAGE_KEY
    assert image_runtime.calls[0][1].prompt == PROMPT


async def test_generate_makes_several_images(client: TestClient, core: Core) -> None:
    frames = await generate(client, count=3, steps=1)
    images = [f["image"] for f in frames if f["type"] == "image"]
    assert len(images) == 3
    assert len({i["id"] for i in images}) == 3
    assert core.gallery.count() == 3


async def test_generate_uses_the_only_image_model_when_none_is_given(
        client: TestClient, image_runtime: FakeImageRuntime) -> None:
    frames = await generate(client, steps=1)
    assert "done" in kinds(frames)
    assert image_runtime.calls[0][0] == IMAGE_KEY        # the first image model on disk


async def test_generate_rejects_an_empty_prompt(client: TestClient) -> None:
    response = await client.post("/api/images/generate", json={"prompt": "  "})
    assert response.status == 400
    body = await response.json()
    assert body["code"] == "missing_prompt"
    assert "opis obrazu" in body["error"]


async def test_generate_rejects_an_unknown_model(client: TestClient) -> None:
    response = await client.post("/api/images/generate",
                                 json={"prompt": PROMPT, "modelKey": "nie-ma-takiego"})
    assert response.status == 404
    assert (await response.json())["code"] == "model_not_found"


@pytest.mark.parametrize("body, code", [
    ({"prompt": PROMPT, "width": 1020, "height": 1024}, "invalid_size"),
    ({"prompt": PROMPT, "width": 64, "height": 64}, "invalid_size"),
    ({"prompt": PROMPT, "count": 99}, "invalid_count"),
    ({"prompt": PROMPT, "steps": "dużo"}, "invalid_field"),
])
async def test_generate_validates_the_body(client: TestClient, body: dict[str, Any],
                                           code: str) -> None:
    response = await client.post("/api/images/generate", json=body)
    assert response.status == 400
    assert (await response.json())["code"] == code


async def test_a_failing_runtime_ends_the_stream_with_a_polish_error(
        client: TestClient, image_runtime: FakeImageRuntime, core: Core) -> None:
    image_runtime.error = ImageEngineError("Zabrakło pamięci na model FLUX.")
    frames = await generate(client)

    assert kinds(frames)[-1] == "error"
    assert frames[-1]["message"] == "Zabrakło pamięci na model FLUX."
    assert core.gallery.count() == 0


async def test_generate_without_the_image_extra_is_a_503(client: TestClient, core: Core) -> None:
    core._images = None
    core._images_built = True
    response = await client.post("/api/images/generate", json={"prompt": PROMPT})
    assert response.status == 503
    body = await response.json()
    assert body["code"] == "images_unavailable"
    assert "localcode[images]" in body["error"]


async def test_a_client_disconnect_cancels_the_job(client: TestClient,
                                                   image_runtime: FakeImageRuntime) -> None:
    image_runtime.hold = asyncio.Event()          # stall inside generate()
    response = await client.post("/api/images/generate",
                                 json={"prompt": PROMPT, "steps": 4})
    assert response.status == 200
    assert (await response.content.readline()).startswith(b"data:")
    response.close()                              # the user hit Ctrl+C / closed the app

    for _ in range(200):
        if image_runtime.cancelled:
            break
        await asyncio.sleep(0.02)
    assert image_runtime.cancelled == 1
    image_runtime.hold.set()


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded(core: Core, tmp_path: Path) -> list[str]:
    """Three images already in the gallery, oldest first."""
    ids: list[str] = []
    for index in range(3):
        result = make_result(tmp_path / "seed", prompt=f"obraz {index}",
                             created_at=1_700_000_000 + index)
        ids.append(core.gallery.add(result, model_name="FLUX.1 schnell").id)
    return ids


async def test_list_is_paged_and_newest_first(client: TestClient, seeded: list[str]) -> None:
    response = await client.get("/api/images", params={"limit": "2"})
    assert response.status == 200
    rows = await response.json()
    assert [r["id"] for r in rows] == list(reversed(seeded))[:2]

    response = await client.get("/api/images", params={"limit": "2", "offset": "2"})
    assert [r["id"] for r in await response.json()] == [seeded[0]]


async def test_favorite_filter_and_patch(client: TestClient, seeded: list[str]) -> None:
    response = await client.patch(f"/api/images/{seeded[1]}", json={"favorite": True})
    assert response.status == 200
    assert (await response.json())["favorite"] is True

    response = await client.get("/api/images", params={"favorite": "1"})
    assert [r["id"] for r in await response.json()] == [seeded[1]]

    bad = await client.patch(f"/api/images/{seeded[1]}", json={"favorite": "tak"})
    assert bad.status == 400 and (await bad.json())["code"] == "invalid_field"
    missing = await client.patch(f"/api/images/{seeded[1]}", json={})
    assert missing.status == 400 and (await missing.json())["code"] == "missing_field"
    unknown = await client.patch("/api/images/nie-ma", json={"favorite": True})
    assert unknown.status == 404


async def test_file_route_serves_the_png(client: TestClient, seeded: list[str]) -> None:
    response = await client.get(f"/api/images/{seeded[0]}/file")
    assert response.status == 200
    assert response.headers["Content-Type"] == "image/png"
    assert await response.read() == PNG_1X1


async def test_file_route_reports_a_missing_file(client: TestClient, core: Core,
                                                 seeded: list[str]) -> None:
    Path(core.gallery.require(seeded[0]).path).unlink()
    response = await client.get(f"/api/images/{seeded[0]}/file")
    assert response.status == 404
    assert (await response.json())["code"] == "file_missing"

    unknown = await client.get("/api/images/nie-ma/file")
    assert unknown.status == 404 and (await unknown.json())["code"] == "not_found"


async def test_delete_removes_the_row_and_the_file(client: TestClient, core: Core,
                                                   seeded: list[str]) -> None:
    path = Path(core.gallery.require(seeded[0]).path)
    response = await client.delete(f"/api/images/{seeded[0]}")
    assert response.status == 204
    assert not path.exists()
    assert core.gallery.get(seeded[0]) is None

    again = await client.delete(f"/api/images/{seeded[0]}")
    assert again.status == 404


async def test_stats(client: TestClient, seeded: list[str]) -> None:
    response = await client.get("/api/images/stats")
    stats = await response.json()
    assert stats["count"] == 3
    assert stats["sizeBytes"] == 3 * len(PNG_1X1)
    assert stats["models"][0]["modelKey"] == IMAGE_KEY


# --------------------------------------------------------------------------- #
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------- #


async def test_openai_images_returns_base64_by_default(client: TestClient, core: Core) -> None:
    response = await client.post("/v1/images/generations",
                                 json={"prompt": PROMPT, "model": "schnell", "size": "512x512"})
    assert response.status == 200
    body = await response.json()

    assert set(body) == {"created", "data"}
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert base64.b64decode(item["b64_json"]) == PNG_1X1
    assert item["revised_prompt"] == PROMPT
    assert core.gallery.count() == 1
    assert core.gallery.list()[0].width == 512


async def test_openai_images_can_return_a_url(client: TestClient) -> None:
    response = await client.post("/v1/images/generations",
                                 json={"prompt": PROMPT, "n": 2, "response_format": "url"})
    body = await response.json()
    assert len(body["data"]) == 2
    for item in body["data"]:
        assert "b64_json" not in item
        assert "/api/images/" in item["url"] and item["url"].endswith("/file")
        assert item["url"].startswith("http://")


@pytest.mark.parametrize("body, fragment", [
    ({}, "prompt"),
    ({"prompt": PROMPT, "n": 99}, "najwyżej"),
    ({"prompt": PROMPT, "size": "duży"}, "rozmiar"),
    ({"prompt": PROMPT, "response_format": "webp"}, "response_format"),
    ({"prompt": PROMPT, "model": "nie-ma"}, "Nie znaleziono"),
])
async def test_openai_images_errors_use_the_openai_shape(client: TestClient, body: dict[str, Any],
                                                         fragment: str) -> None:
    response = await client.post("/v1/images/generations", json=body)
    assert response.status >= 400
    payload = await response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"message", "type", "code", "param"}
    assert fragment.lower() in payload["error"]["message"].lower()


async def test_openai_images_reports_a_failing_runtime(client: TestClient,
                                                       image_runtime: FakeImageRuntime) -> None:
    image_runtime.error = ImageEngineError("Silnik obrazów padł.")
    response = await client.post("/v1/images/generations", json={"prompt": PROMPT})
    assert response.status >= 400
    assert "Silnik obrazów padł." in (await response.json())["error"]["message"]


async def test_openai_models_route_still_works(client: TestClient) -> None:
    """``/v1/models/{id}`` must not swallow ``/v1/images/generations``."""
    response = await client.get(f"/v1/models/{IMAGE_KEY}")
    assert response.status == 200
    assert (await response.json())["id"] == IMAGE_KEY


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


async def test_generate_then_browse_download_and_delete(client: TestClient, core: Core) -> None:
    """The whole path with a fake engine: SSE → gallery → file → delete."""
    frames = await generate(client, steps=2, seed=5, count=2)
    images = [f["image"] for f in frames if f["type"] == "image"]
    assert len(images) == 2

    listed = await (await client.get("/api/images")).json()
    assert [r["id"] for r in listed] == [images[1]["id"], images[0]["id"]]

    file_response = await client.get(listed[0]["url"])
    assert file_response.status == 200 and await file_response.read() == PNG_1X1

    assert (await client.patch(f"/api/images/{listed[0]['id']}",
                               json={"favorite": True})).status == 200
    assert (await (await client.get("/api/images/stats")).json())["favorites"] == 1

    for row in listed:
        assert (await client.delete(f"/api/images/{row['id']}")).status == 204
    assert await (await client.get("/api/images")).json() == []
    assert list(core.gallery.root.rglob("*.png")) == []


# --------------------------------------------------------------------------- #
# Integration with the real ImageRuntime surface
# --------------------------------------------------------------------------- #


class DescribingRuntime(FakeImageRuntime):
    """A runtime that describes its own models and status, like the real one."""

    def __init__(self, root: Any) -> None:
        super().__init__(root)
        self.model_infos = 0

    def models(self) -> list[Any]:
        return [type("M", (), {"key": IMAGE_KEY})()]

    def model_info(self, model: Any) -> dict[str, Any]:
        self.model_infos += 1
        return {"key": model.key, "imageFamily": "flux", "loaded": True,
                "estimatedBytes": 4_000_000_000,
                "defaults": {"steps": 4, "guidance": None, "sizes": ["1024x1024"]}}

    def runtime_status(self) -> dict[str, Any]:
        return {"instances": [{"id": "img-1", "kind": "image", "modelKey": IMAGE_KEY}],
                "images": {"instances": [], "settings": {"maxLoaded": 1}}}


@pytest.fixture
def describing(core: Core, tmp_path: Path) -> DescribingRuntime:
    runtime = DescribingRuntime(tmp_path / "engine")
    core._images = runtime
    core._images_built = True
    return runtime


async def test_models_come_from_the_image_runtime_when_it_can_describe_them(
        client: TestClient, describing: DescribingRuntime) -> None:
    models = await (await client.get("/api/images/models")).json()
    assert describing.model_infos == 1
    assert models == [{"key": IMAGE_KEY, "imageFamily": "flux", "loaded": True,
                       "estimatedBytes": 4_000_000_000,
                       "defaults": {"steps": 4, "guidance": None, "sizes": ["1024x1024"]}}]


def test_runtime_status_prefers_the_runtimes_own_merge(core: Core,
                                                       describing: DescribingRuntime) -> None:
    status = core.runtime_status()
    assert [i["id"] for i in status["instances"]] == ["img-1"]
    assert status["images"]["settings"]["maxLoaded"] == 1


def test_runtime_status_merges_image_instances_itself(core: Core,
                                                      image_runtime: FakeImageRuntime) -> None:
    """A runtime with only ``status()`` still shows up in ``/api/runtime``."""
    image_runtime.instances = [{"id": "img-9", "kind": "image", "modelKey": IMAGE_KEY}]
    status = core.runtime_status()
    assert {i["id"] for i in status["instances"]} >= {"img-9"}
    assert "budget" in status and "system" in status


async def test_image_events_reach_the_event_bus(core: Core) -> None:
    sub = core.events.subscribe()
    try:
        core._on_image_event({"type": "image_progress", "index": 0, "step": 2, "steps": 4})
        assert (await sub.get(timeout=1.0))["type"] == "image_progress"

        core._on_image_event({"type": "runtime"})          # bare signal → full snapshot
        expanded = await sub.get(timeout=1.0)
        assert expanded["type"] == "runtime" and "instances" in expanded
    finally:
        sub.close()


# --------------------------------------------------------------------------- #
# With a real model (skipped unless one is on disk)
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
async def test_a_real_model_generates_into_the_gallery(tmp_path: Path,
                                                       isolated_home: Path) -> None:
    """The whole path with a real engine: ``ImageRuntime`` → file → gallery row.

    Uses the smallest image model in ``~/.localcode/models`` (SD 1.5 GGUF,
    ~1.5 GB) at 256×256 / 2 steps, and always stops the worker process.
    """
    from localcode.config import Settings
    from localcode.hub.library import ModelLibrary
    from localcode.images.runtime import ImageRuntime
    from localcode.server.routes_images import build_params, generate_records

    models_dir = Path("~/.localcode/models").expanduser()
    settings = Settings(models_dir=str(models_dir))
    if not models_dir.is_dir():
        pytest.skip("brak katalogu ~/.localcode/models")
    library = ModelLibrary(settings)
    available = [m for m in library.scan() if m.is_image]
    if not available:
        pytest.skip("brak modelu obrazów na dysku (np. second-state/stable-diffusion-v1-5-GGUF)")
    model = min(available, key=lambda m: m.size_bytes)

    runtime = ImageRuntime(settings, library)
    core = Core(settings, db_path=tmp_path / "chat.db", library=library, images=runtime)
    await core.start()
    try:
        params = build_params(prompt="mały czerwony kwiat", width=256, height=256, steps=2,
                              seed=1, count=1)
        records = await generate_records(core, model.key, params, model_name=model.display_name)
        assert len(records) == 1
        record = records[0]
        assert Path(record.path).is_file()
        assert Path(record.path).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert Path(record.path).parent.parent == core.gallery.root
        assert core.gallery.get(record.id) is not None
        assert record.seed == 1 and record.steps == 2
    finally:
        await core.shutdown()          # stops the worker subprocess


async def test_model_sizes_are_native_first(client: TestClient) -> None:
    """``defaults.sizes`` opens with the resolution the family was trained for.

    The Mac app adopts ``sizes.first`` when the composer sits on a resolution
    this model does not offer (``ImagesStore.adoptModelSize``), so the order is
    part of the contract: SD 1.5 must not be handed FLUX's 1024².
    """
    models = {m["key"]: m for m in await (await client.get("/api/images/models")).json()}
    assert models[IMAGE_KEY]["defaults"]["sizes"][0] == "1024x1024"
    assert models[IMAGE_KEY_2]["defaults"]["sizes"][0] == "512x512"
    for model in models.values():
        assert model["defaults"]["sizes"], "every image model must name at least one size"
