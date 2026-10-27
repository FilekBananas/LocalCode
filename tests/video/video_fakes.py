"""Fakes for the video tests.

``FakeVideoRuntime`` stands in for :class:`~localcode.video.runtime.VideoRuntime`
the same way ``images_fakes.FakeImageRuntime`` stands in for the image runtime
and ``tests/video/test_cli_video.py``'s own inline fake stands in for the CLI
tests: what the route tests are about is the HTTP contract (status codes,
error codes, the SSE frame sequence), not a real generation.

The model keys below are chosen so :func:`localcode.video.base.classify_video_model`
recognises them without any extra wiring — ``repo_id`` is derived from the key
by ``server_fakes.make_model`` (``"/".join(key.split("/")[:2])``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from localcode.video.base import VideoError, VideoProgress, VideoResult

from server_fakes import make_model

#: A checkpoint whose repo_id is one of ``KNOWN_LTX_2B_REPOS`` — classified "ltx-2b".
VIDEO_KEY = "city96/ltx-video-gguf/ltxv-2b-q8_0"
#: Same LTX family, but the 13B+ "LTX-2" line — classified "ltx-other".
VIDEO_KEY_LTX_OTHER = "Lightricks/LTX-2-13B/ltx2-fp16"
#: A different video family entirely — classified "other-video".
VIDEO_KEY_OTHER_FAMILY = "Wan-AI/Wan2.1-T2V-1.3B/wan2-t2v"
#: Nothing about the name suggests a video model — classified "unknown", so
#: ``VideoRuntime.list_models()`` leaves it out entirely.
VIDEO_KEY_UNKNOWN = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"


def make_ltx_model(key: str = VIDEO_KEY, *, name: str = "LTX-Video 2B") -> Any:
    """A ``LocalModel`` :func:`classify_video_model` recognises as ``"ltx-2b"``."""
    return make_model(key, name=name, fmt="gguf")


class FakeVideoRuntime:
    """Stands in for :class:`~localcode.video.runtime.VideoRuntime` at the route layer."""

    def __init__(self, *, models: list[dict[str, Any]] | None = None,
                 output_root: Any | None = None) -> None:
        self.models = models if models is not None else []
        #: Where the route is allowed to write (see ``routes_video._safe_output``).
        self.output_root = Path(output_root) if output_root else Path.home() / ".localcode" / "data" / "video"
        self.estimate_calls: list[dict[str, Any]] = []
        self.generate_calls: list[tuple[str, dict[str, Any]]] = []
        self.progress_seen: list[VideoProgress] = []
        #: fed to ``on_progress`` during ``generate`` — a realistic sequence.
        self.progress_events: list[VideoProgress] = [
            VideoProgress(stage="loading", step=0, steps=0, message="Wczytuję model…"),
            VideoProgress(stage="denoising", step=1, steps=2, message="Odszumianie: krok 1/2"),
            VideoProgress(stage="denoising", step=2, steps=2, message="Odszumianie: krok 2/2"),
            VideoProgress(stage="decoding", step=2, steps=2, message="Dekoduję klatki…"),
        ]
        self.estimate_error: VideoError | None = None
        self.generate_error: VideoError | None = None
        self.result: VideoResult | None = None
        self.hold: asyncio.Event | None = None    # set to stall inside generate() (disconnect test)
        self.cancelled = 0

    def output_dir(self) -> Path:
        """Mirrors ``VideoRuntime.output_dir`` — the route confines writes to it."""
        return self.output_root

    def list_models(self) -> list[dict[str, Any]]:
        return list(self.models)

    def estimate(self, **kwargs: Any) -> dict[str, Any]:
        self.estimate_calls.append(kwargs)
        if self.estimate_error:
            raise self.estimate_error
        return {
            "seconds": 90.0, "message": "41 klatek — szacunkowo ~2 min liczenia (niezmierzone).",
            "width": kwargs.get("width") or 704, "height": kwargs.get("height") or 480,
            "numFrames": kwargs.get("num_frames") or 41, "fps": kwargs.get("fps") or 24,
            "steps": kwargs.get("steps") or 30,
        }

    async def generate(self, prompt: str, *, on_progress: Any = None, **kwargs: Any) -> VideoResult:
        self.generate_calls.append((prompt, kwargs))
        try:
            for progress in self.progress_events:
                self.progress_seen.append(progress)
                if on_progress is not None:
                    on_progress(progress)
                if self.hold is not None:
                    await self.hold.wait()
                else:
                    await asyncio.sleep(0)
            if self.generate_error:
                raise self.generate_error
            assert self.result is not None, "set fake.result before calling generate()"
            return self.result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


def make_result(*, prompt: str = "kot w kapeluszu idzie ulicą", **overrides: Any) -> VideoResult:
    fields: dict[str, Any] = dict(
        id="deadbeef01", path="/tmp/wideo.mp4", width=704, height=480, num_frames=41, fps=24,
        seed=123, steps=30, guidance=3.0, model_key=VIDEO_KEY, prompt=prompt,
        negative_prompt="worst quality", elapsed_ms=125_000, size_bytes=4_000_000,
        estimated_seconds=90.0,
    )
    fields.update(overrides)
    return VideoResult(**fields)
