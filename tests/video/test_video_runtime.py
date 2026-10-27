"""``localcode/video/runtime.py`` — model selection, the generation pipeline and
the worker subprocess's JSON-lines protocol.

Two layers of test double, matching the two test seams ``VideoRuntime`` ships
(see its own docstring):

* ``runner=`` replaces the whole subprocess with a plain async callable —
  used for the model-selection, frame-cap-fails-fast and error-mapping tests,
  where the subprocess itself is not what is under test.
* ``worker_cmd=`` replaces only the argv, so a *real* subprocess still runs —
  used for the protocol tests at the bottom, which exercise the actual
  ``asyncio.create_subprocess_exec`` + stdin/stdout JSON-lines pump
  (:meth:`VideoRuntime._pump`) against a tiny stand-in worker script instead
  of the real one, since the real one needs ``torch``/``diffusers`` and real
  model weights (see ``docs/VIDEO.md`` — nothing here has run against those).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from localcode.config import Settings, home
from localcode.video.base import (
    INSTALL_HINT,
    MAX_FRAMES_DEFAULT,
    VideoError,
    VideoProgress,
)
from localcode.video.runtime import VideoRuntime

from server_fakes import CHAT_KEY, FakeLibrary, make_model
from video_fakes import VIDEO_KEY, VIDEO_KEY_LTX_OTHER, VIDEO_KEY_OTHER_FAMILY, make_ltx_model

PROMPT = "kot w kapeluszu idzie ulicą"


@pytest.fixture
def settings(tmp_path: Path, isolated_home: Path) -> Settings:
    return Settings(models_dir=str(tmp_path / "models"))


class RecordingRunner:
    """The ``runner=`` test seam: records every call, returns/raises canned results."""

    def __init__(self, *, raw: dict[str, Any] | None = None,
                error: BaseException | None = None,
                on_call: Any = None) -> None:
        self.raw = raw if raw is not None else {}
        self.error = error
        self.on_call = on_call
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model_path: str, base_repo: str, params: Any,
                       output_path: str, on_progress: Any) -> dict[str, Any]:
        self.calls.append({"model_path": model_path, "base_repo": base_repo,
                           "params": params, "output_path": output_path})
        if self.on_call is not None:
            await self.on_call(on_progress)
        if self.error is not None:
            raise self.error
        return self.raw


# --------------------------------------------------------------------------- #
# list_models()
# --------------------------------------------------------------------------- #


def test_list_models_tags_kind_and_supported_and_drops_unknown(settings: Settings) -> None:
    library = FakeLibrary([
        make_model(CHAT_KEY, name="Qwen3 0.6B"),        # unknown -> dropped
        make_ltx_model(),                                 # ltx-2b -> supported
        make_model(VIDEO_KEY_LTX_OTHER, name="LTX-2 13B"),   # ltx-other -> not supported
        make_model(VIDEO_KEY_OTHER_FAMILY, name="Wan2.1 T2V"),  # other-video -> not supported
    ])
    runtime = VideoRuntime(settings, library)
    dtos = runtime.list_models()

    assert [d["key"] for d in dtos] == [VIDEO_KEY, VIDEO_KEY_LTX_OTHER, VIDEO_KEY_OTHER_FAMILY]
    by_key = {d["key"]: d for d in dtos}
    assert by_key[VIDEO_KEY]["videoKind"] == "ltx-2b" and by_key[VIDEO_KEY]["supported"] is True
    assert by_key[VIDEO_KEY_LTX_OTHER]["videoKind"] == "ltx-other"
    assert by_key[VIDEO_KEY_LTX_OTHER]["supported"] is False
    assert by_key[VIDEO_KEY_OTHER_FAMILY]["videoKind"] == "other-video"
    assert by_key[VIDEO_KEY_OTHER_FAMILY]["supported"] is False


# --------------------------------------------------------------------------- #
# estimate()
# --------------------------------------------------------------------------- #


def test_estimate_needs_no_prompt_and_no_library(settings: Settings) -> None:
    runtime = VideoRuntime(settings, library=FakeLibrary([]))
    result = runtime.estimate(width=512, height=384, num_frames=41, fps=24, steps=10)
    assert result["width"] == 512 and result["height"] == 384
    assert result["numFrames"] == 41
    assert result["seconds"] > 0
    assert "szacunkowo" in result["message"]


def test_estimate_does_not_enforce_the_frame_cap(settings: Settings) -> None:
    """The cost estimate must be able to quote a price for an over-budget
    request without requiring ``force`` — it still refuses to guess past the
    absolute hard cap (see ``test_video_base.py``)."""
    runtime = VideoRuntime(settings, library=FakeLibrary([]))
    result = runtime.estimate(num_frames=MAX_FRAMES_DEFAULT + 40)
    assert result["numFrames"] > MAX_FRAMES_DEFAULT


# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #


async def test_generate_picks_the_first_supported_model_when_none_named(
        settings: Settings, isolated_home: Path) -> None:
    library = FakeLibrary([make_model(CHAT_KEY, name="Qwen3 0.6B"), make_ltx_model()])
    runner = RecordingRunner(raw={})
    runtime = VideoRuntime(settings, library, runner=runner)
    result = await runtime.generate(PROMPT, num_frames=9, steps=1)
    assert result.model_key == VIDEO_KEY
    assert runner.calls[0]["model_path"] == "/models/" + VIDEO_KEY


async def test_generate_rejects_an_unknown_model_key(settings: Settings) -> None:
    runtime = VideoRuntime(settings, FakeLibrary([make_ltx_model()]), runner=RecordingRunner())
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, model_key="nie/ma/takiego")
    assert excinfo.value.code == "model_not_found"


async def test_generate_rejects_an_unsupported_named_model(settings: Settings) -> None:
    library = FakeLibrary([make_model(VIDEO_KEY_LTX_OTHER, name="LTX-2 13B")])
    runtime = VideoRuntime(settings, library, runner=RecordingRunner())
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, model_key=VIDEO_KEY_LTX_OTHER)
    assert excinfo.value.code == "unsupported_model"
    assert "LTX-2 13B" in str(excinfo.value)


async def test_generate_with_no_models_at_all_says_so(settings: Settings) -> None:
    runtime = VideoRuntime(settings, FakeLibrary([]), runner=RecordingRunner())
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT)
    assert excinfo.value.code == "no_model"
    assert "Wideo" in str(excinfo.value)


async def test_generate_with_only_unsupported_models_names_one_of_them(settings: Settings) -> None:
    library = FakeLibrary([make_model(VIDEO_KEY_OTHER_FAMILY, name="Wan2.1 T2V")])
    runtime = VideoRuntime(settings, library, runner=RecordingRunner())
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT)
    assert excinfo.value.code == "unsupported_model"
    assert "Wan2.1 T2V" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# generate() — happy path, progress and result mapping
# --------------------------------------------------------------------------- #


async def test_generate_reports_the_initial_estimate_before_the_runner_runs(
        settings: Settings) -> None:
    library = FakeLibrary([make_ltx_model(name="LTX-Video 2B")])
    seen: list[VideoProgress] = []

    async def during(on_progress: Any) -> None:
        on_progress(VideoProgress(stage="denoising", step=1, steps=1, message="krok 1/1"))

    runner = RecordingRunner(raw={}, on_call=during)
    runtime = VideoRuntime(settings, library, runner=runner)
    await runtime.generate(PROMPT, on_progress=seen.append, steps=1, num_frames=9)

    assert [p.stage for p in seen] == ["loading", "denoising"]
    assert "LTX-Video 2B" in seen[0].message
    assert "szacunkowo" in seen[0].message


async def test_generate_maps_the_raw_result_and_falls_back_sensibly(
        settings: Settings, tmp_path: Path) -> None:
    library = FakeLibrary([make_ltx_model()])
    target = tmp_path / "clips" / "wideo.mp4"

    async def make_file(_on_progress: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00" * 4096)             # so the size_bytes fallback has something to stat

    runner = RecordingRunner(raw={}, on_call=make_file)   # empty raw: every field must fall back
    runtime = VideoRuntime(settings, library, runner=runner)
    result = await runtime.generate(PROMPT, output=str(target), seed=42, width=512, height=384,
                                    num_frames=41, fps=24, steps=10, guidance=4.5)

    assert result.path == str(target)
    assert (result.width, result.height, result.num_frames) == (512, 384, 41)
    assert (result.fps, result.steps, result.guidance) == (24, 10, 4.5)
    assert result.seed == 42
    assert result.size_bytes == 4096                    # stat()'d since "sizeBytes" was absent
    assert result.prompt == PROMPT
    assert result.model_key == VIDEO_KEY
    assert result.estimated_seconds > 0
    assert result.elapsed_ms >= 0


async def test_generate_uses_the_default_output_path_under_home(
        settings: Settings, isolated_home: Path) -> None:
    library = FakeLibrary([make_ltx_model()])
    runner = RecordingRunner(raw={})
    runtime = VideoRuntime(settings, library, runner=runner)
    result = await runtime.generate(PROMPT, num_frames=9, steps=1)

    expected_dir = home() / "data" / "video"
    assert Path(result.path).parent == expected_dir
    assert Path(result.path).name.startswith("wideo-") and result.path.endswith(".mp4")
    assert expected_dir.is_dir()                         # generate() creates it up front


# --------------------------------------------------------------------------- #
# generate() — the frame cap fails fast (no subprocess spawned)
# --------------------------------------------------------------------------- #


async def test_generate_enforces_the_frame_cap_before_touching_the_runner(
        settings: Settings) -> None:
    library = FakeLibrary([make_ltx_model()])
    runner = RecordingRunner()
    runtime = VideoRuntime(settings, library, runner=runner)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, num_frames=MAX_FRAMES_DEFAULT + 8)
    assert excinfo.value.code == "frame_cap_exceeded"
    assert runner.calls == [], "an over-budget request must never reach the worker"


# --------------------------------------------------------------------------- #
# generate() — exception mapping
# --------------------------------------------------------------------------- #


async def test_generate_wraps_an_import_error_as_the_polish_missing_extra_error(
        settings: Settings) -> None:
    library = FakeLibrary([make_ltx_model()])
    runner = RecordingRunner(error=ImportError("No module named 'diffusers'"))
    runtime = VideoRuntime(settings, library, runner=runner)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, num_frames=9, steps=1)
    assert excinfo.value.code == "missing_extra"
    assert excinfo.value.status == 501
    assert str(excinfo.value) == INSTALL_HINT


async def test_generate_wraps_an_unexpected_failure_as_generate_failed(settings: Settings) -> None:
    library = FakeLibrary([make_ltx_model()])
    runner = RecordingRunner(error=RuntimeError("Zabrakło pamięci GPU."))
    runtime = VideoRuntime(settings, library, runner=runner)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, num_frames=9, steps=1)
    assert excinfo.value.code == "generate_failed"
    assert excinfo.value.status == 500
    assert "Zabrakło pamięci GPU." in str(excinfo.value)


async def test_generate_propagates_a_video_error_from_the_runner_unchanged(
        settings: Settings) -> None:
    library = FakeLibrary([make_ltx_model()])
    original = VideoError("Nie udało się wczytać modelu.", "load_failed", 500)
    runner = RecordingRunner(error=original)
    runtime = VideoRuntime(settings, library, runner=runner)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, num_frames=9, steps=1)
    assert excinfo.value is original


# --------------------------------------------------------------------------- #
# The real subprocess protocol (worker_cmd= — a stand-in script, no torch)
# --------------------------------------------------------------------------- #


_FAKE_WORKER_SRC = """
import json, sys

def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

raw = sys.stdin.read()
payload = json.loads(raw) if raw.strip() else {}
mode = payload.get("negativePrompt")  # reused as a control channel by the tests

if mode == "no-result":
    sys.exit(0)
if mode == "worker-error":
    emit({"type": "error", "message": "Zabrakło pamięci GPU.", "code": "generate_failed"})
    sys.exit(1)

emit({"type": "progress", "stage": "loading", "step": 0, "steps": 0, "elapsedMs": 0,
      "message": "wczytuję"})
steps = int(payload.get("steps") or 1)
for i in range(1, steps + 1):
    emit({"type": "progress", "stage": "denoising", "step": i, "steps": steps,
          "elapsedMs": i * 10, "message": "krok %d/%d" % (i, steps)})
emit({"type": "result", "path": payload.get("outputPath"), "width": payload.get("width"),
      "height": payload.get("height"), "numFrames": payload.get("numFrames"),
      "fps": payload.get("fps"), "seed": payload.get("seed"), "steps": steps,
      "guidance": payload.get("guidance"), "sizeBytes": 999})
"""


@pytest.fixture
def fake_worker_cmd(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER_SRC, encoding="utf-8")
    return [sys.executable, "-u", str(script)]


async def test_subprocess_protocol_streams_progress_and_a_result(
        settings: Settings, isolated_home: Path, fake_worker_cmd: list[str]) -> None:
    library = FakeLibrary([make_ltx_model()])
    runtime = VideoRuntime(settings, library, worker_cmd=fake_worker_cmd)
    seen: list[VideoProgress] = []

    result = await runtime.generate(PROMPT, on_progress=seen.append, steps=2, num_frames=9,
                                     width=512, height=384, fps=24, seed=555)

    # one "loading" progress from VideoRuntime.generate() itself, then the
    # real subprocess's own "loading" + 2 "denoising" lines, parsed back into
    # VideoProgress objects by VideoRuntime._pump:
    assert [p.stage for p in seen] == ["loading", "loading", "denoising", "denoising"]
    assert [p.step for p in seen[-2:]] == [1, 2]
    assert result.seed == 555
    assert (result.width, result.height, result.num_frames) == (512, 384, 9)
    assert result.size_bytes == 999
    assert result.path.endswith(".mp4")


async def test_subprocess_protocol_propagates_a_worker_reported_error(
        settings: Settings, isolated_home: Path, fake_worker_cmd: list[str]) -> None:
    library = FakeLibrary([make_ltx_model()])
    runtime = VideoRuntime(settings, library, worker_cmd=fake_worker_cmd)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, negative_prompt="worker-error", num_frames=9, steps=1)
    assert excinfo.value.code == "generate_failed"
    assert "Zabrakło pamięci GPU." in str(excinfo.value)


async def test_subprocess_protocol_reports_a_silent_exit_as_no_result(
        settings: Settings, isolated_home: Path, fake_worker_cmd: list[str]) -> None:
    library = FakeLibrary([make_ltx_model()])
    runtime = VideoRuntime(settings, library, worker_cmd=fake_worker_cmd)
    with pytest.raises(VideoError) as excinfo:
        await runtime.generate(PROMPT, negative_prompt="no-result", num_frames=9, steps=1)
    assert excinfo.value.code == "no_result"
    assert "bez wyniku" in str(excinfo.value)
