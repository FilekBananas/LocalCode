"""The HTTP surface of video generation: ``/api/video/models|estimate|generate``.

The runtime is replaced with :class:`~video_fakes.FakeVideoRuntime` (the same
shape as ``tests/server/test_routes_speech.py``'s ``StubSpeech`` and
``tests/images/test_routes_images.py``'s ``FakeImageRuntime``): what these
tests are about is the HTTP contract — request validation, status codes, the
SSE frame sequence — not a real generation, which is covered (with the real
``VideoParams``/``classify_video_model`` logic) in ``test_video_base.py`` and
``test_video_runtime.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aiohttp.test_utils import TestClient

from localcode.server.core import Core
from localcode.video.base import INSTALL_HINT, VideoError

from video_fakes import VIDEO_KEY, FakeVideoRuntime, make_result

PROMPT = "kot w kapeluszu idzie ulicą"


async def read_frames(response: Any, *, limit: int = 200) -> list[dict[str, Any]]:
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
    async with client.post("/api/video/generate", json=payload) as response:
        assert response.status == 200, await response.text()
        assert response.headers["Content-Type"].startswith("text/event-stream")
        return await read_frames(response)


def kinds(frames: list[dict[str, Any]]) -> list[str]:
    return [str(f.get("type")) for f in frames]


# --------------------------------------------------------------------------- #
# GET /api/video/models
# --------------------------------------------------------------------------- #


async def test_models_returns_whatever_the_runtime_reports(client: TestClient,
                                                            video: FakeVideoRuntime) -> None:
    video.models = [
        {"key": VIDEO_KEY, "displayName": "LTX-Video 2B", "videoKind": "ltx-2b", "supported": True},
        {"key": "other/model/x", "displayName": "Wan 2.1", "videoKind": "other-video", "supported": False},
    ]
    response = await client.get("/api/video/models")
    assert response.status == 200
    body = await response.json()
    assert [m["key"] for m in body] == [VIDEO_KEY, "other/model/x"]
    assert body[0]["supported"] is True
    assert body[1]["supported"] is False


async def test_every_video_route_needs_the_token(anon_client: TestClient) -> None:
    for method, path in (("GET", "/api/video/models"), ("POST", "/api/video/estimate"),
                         ("POST", "/api/video/generate")):
        response = await anon_client.request(method, path, json={})
        assert response.status == 401, f"{method} {path}"
        assert (await response.json())["code"] == "unauthorized"


# --------------------------------------------------------------------------- #
# POST /api/video/estimate
# --------------------------------------------------------------------------- #


async def test_estimate_does_not_require_a_prompt(client: TestClient, video: FakeVideoRuntime) -> None:
    response = await client.post("/api/video/estimate", json={})
    assert response.status == 200
    body = await response.json()
    assert body["numFrames"] == 41 and body["width"] == 704
    assert video.estimate_calls == [{"width": None, "height": None, "num_frames": None,
                                     "fps": None, "steps": None}]


async def test_estimate_threads_the_shape_fields_through(client: TestClient,
                                                          video: FakeVideoRuntime) -> None:
    response = await client.post("/api/video/estimate",
                                 json={"width": 512, "height": 384, "numFrames": 65,
                                       "fps": 12, "steps": 10})
    assert response.status == 200
    assert video.estimate_calls[-1] == {"width": 512, "height": 384, "num_frames": 65,
                                        "fps": 12, "steps": 10}


async def test_estimate_rejects_a_non_object_body(client: TestClient) -> None:
    response = await client.post("/api/video/estimate", json=[1, 2, 3])
    assert response.status == 400
    assert (await response.json())["code"] == "invalid_json"


@pytest.mark.parametrize("field", ["width", "height", "numFrames", "fps", "steps"])
async def test_estimate_rejects_a_non_integer_field(client: TestClient, field: str) -> None:
    response = await client.post("/api/video/estimate", json={field: "duży"})
    assert response.status == 400
    assert (await response.json())["code"] == "invalid_field"


async def test_estimate_propagates_a_video_error(client: TestClient, video: FakeVideoRuntime) -> None:
    video.estimate_error = VideoError("9999 klatek przekracza bezwzględny limit 161.",
                                       "frame_hard_cap")
    response = await client.post("/api/video/estimate", json={"numFrames": 9999})
    assert response.status == 400
    body = await response.json()
    assert body["code"] == "frame_hard_cap"
    assert "bezwzględny limit" in body["error"]


# --------------------------------------------------------------------------- #
# POST /api/video/generate — argument validation
# --------------------------------------------------------------------------- #


async def test_generate_rejects_an_empty_prompt(client: TestClient, video: FakeVideoRuntime) -> None:
    response = await client.post("/api/video/generate", json={"prompt": "   "})
    assert response.status == 400
    body = await response.json()
    assert body["code"] == "missing_prompt"
    assert "opis materiału wideo" in body["error"]
    assert video.generate_calls == []


async def test_generate_rejects_a_non_object_body(client: TestClient, video: FakeVideoRuntime) -> None:
    response = await client.post("/api/video/generate", json="wideo, proszę")
    assert response.status == 400
    assert (await response.json())["code"] == "invalid_json"
    assert video.generate_calls == []


@pytest.mark.parametrize("body", [
    {"prompt": PROMPT, "width": "duży"},
    {"prompt": PROMPT, "numFrames": "dużo"},
    {"prompt": PROMPT, "seed": "losowe"},
    {"prompt": PROMPT, "guidance": "mocne"},
])
async def test_generate_rejects_a_bad_numeric_field(client: TestClient, video: FakeVideoRuntime,
                                                     body: dict[str, Any]) -> None:
    response = await client.post("/api/video/generate", json=body)
    assert response.status == 400
    assert (await response.json())["code"] == "invalid_field"
    assert video.generate_calls == []


async def test_generate_rejects_a_non_boolean_force(client: TestClient, video: FakeVideoRuntime) -> None:
    response = await client.post("/api/video/generate", json={"prompt": PROMPT, "force": "tak"})
    assert response.status == 400
    body = await response.json()
    assert body["code"] == "invalid_field"
    assert "force" in body["error"]
    assert video.generate_calls == []


# --------------------------------------------------------------------------- #
# POST /api/video/generate — happy path / progress reporting
# --------------------------------------------------------------------------- #


async def test_generate_streams_progress_then_the_finished_clip(client: TestClient,
                                                                 video: FakeVideoRuntime) -> None:
    video.result = make_result(prompt=PROMPT)
    frames = await generate(client)

    assert kinds(frames) == ["progress", "progress", "progress", "progress", "video", "done"]
    progress = [f for f in frames if f["type"] == "progress"]
    assert [p["stage"] for p in progress] == ["loading", "denoising", "denoising", "decoding"]
    # every progress frame carries the full VideoProgress shape the Swift app reads
    for p in progress:
        assert set(p) >= {"stage", "step", "steps", "elapsedMs", "message", "fraction"}
    mid = progress[1]
    assert mid["step"] == 1 and mid["steps"] == 2 and mid["fraction"] == pytest.approx(0.5)

    video_frame = next(f for f in frames if f["type"] == "video")["video"]
    assert video_frame["id"] == video.result.id
    assert video_frame["prompt"] == PROMPT
    assert (video_frame["width"], video_frame["height"], video_frame["numFrames"]) == (704, 480, 41)


async def test_generate_threads_every_field_to_the_runtime(client: TestClient,
                                                            video: FakeVideoRuntime) -> None:
    video.result = make_result(prompt=PROMPT)
    # "output" is a name inside the video directory, not a path: the route refuses
    # to write anywhere else (see routes_video._safe_output)
    await generate(client, modelKey=VIDEO_KEY, negativePrompt="rozmyte", width=512, height=384,
                   numFrames=65, fps=12, steps=10, guidance=4.5, seed=99, force=True,
                   output="klip.mp4")

    prompt, kwargs = video.generate_calls[0]
    assert prompt == PROMPT
    assert kwargs["model_key"] == VIDEO_KEY
    assert kwargs["negative_prompt"] == "rozmyte"
    assert (kwargs["width"], kwargs["height"]) == (512, 384)
    assert kwargs["num_frames"] == 65
    assert kwargs["fps"] == 12
    assert kwargs["steps"] == 10
    assert kwargs["guidance"] == 4.5
    assert kwargs["seed"] == 99
    assert kwargs["force"] is True
    assert kwargs["output"] == str(video.output_dir() / "klip.mp4")


async def test_generate_defaults_are_left_to_the_runtime(client: TestClient,
                                                          video: FakeVideoRuntime) -> None:
    video.result = make_result(prompt=PROMPT)
    await generate(client)
    _prompt, kwargs = video.generate_calls[0]
    for key in ("model_key", "width", "height", "num_frames", "fps", "steps", "guidance",
               "seed", "output"):
        assert kwargs[key] is None, key
    assert kwargs["negative_prompt"] is None
    assert kwargs["force"] is False


# --------------------------------------------------------------------------- #
# POST /api/video/generate — failure paths
# --------------------------------------------------------------------------- #


async def test_generate_reports_the_frame_cap_as_an_error_frame(client: TestClient,
                                                                 video: FakeVideoRuntime) -> None:
    video.generate_error = VideoError(
        "300 klatek przekracza domyślny limit 81 (to już kilka minut liczenia na tym sprzęcie). "
        "Użyj --force (albo \"force\": true), żeby wygenerować więcej, maksymalnie 161.",
        "frame_cap_exceeded")
    frames = await generate(client, numFrames=300)
    assert kinds(frames)[-1] == "error"
    error = frames[-1]
    assert error["code"] == "frame_cap_exceeded"
    assert "przekracza domyślny limit" in error["message"]


async def test_generate_reports_the_missing_dependency_in_polish(client: TestClient,
                                                                  video: FakeVideoRuntime) -> None:
    video.generate_error = VideoError(INSTALL_HINT, "missing_extra", 501)
    frames = await generate(client)
    assert kinds(frames)[-1] == "error"
    error = frames[-1]
    assert error["code"] == "missing_extra"
    assert error["message"] == INSTALL_HINT
    assert 'localcode[video]' in error["message"]


async def test_generate_reports_no_model_installed(client: TestClient, video: FakeVideoRuntime) -> None:
    video.generate_error = VideoError(
        "Nie masz jeszcze modelu do generowania wideo (LTX-Video 2B). "
        "Zainstaluj go w sklepie (kategoria „Wideo”).", "no_model")
    frames = await generate(client)
    assert kinds(frames)[-1] == "error"
    assert frames[-1]["code"] == "no_model"
    assert "Wideo" in frames[-1]["message"]


async def test_a_client_disconnect_cancels_the_job(client: TestClient, video: FakeVideoRuntime) -> None:
    video.hold = asyncio.Event()               # stall inside generate()
    response = await client.post("/api/video/generate", json={"prompt": PROMPT})
    assert response.status == 200
    assert (await response.content.readline()).startswith(b"data:")
    response.close()                            # the user closed the app / hit Ctrl+C

    for _ in range(200):
        if video.cancelled:
            break
        await asyncio.sleep(0.02)
    assert video.cancelled == 1
    video.hold.set()
