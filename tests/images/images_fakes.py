"""Fakes for the image tests.

:class:`FakeImageRuntime` stands in for ``localcode.images.runtime.ImageRuntime``
— it writes a real (tiny) PNG, reports progress through ``on_progress`` and can
be told to fail or to stall, so the gallery, the HTTP layer and the terminal can
be tested without mflux or stable-diffusion.cpp and without loading anything.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from localcode.hub.types import LocalModel, ModelMeta
from localcode.images.base import ImageParams, ImageProgress, ImageResult

IMAGE_KEY = "dhairyashil/FLUX.1-schnell-mflux-4bit"
IMAGE_KEY_2 = "second-state/stable-diffusion-v1-5-GGUF/stable-diffusion-v1-5-Q4_0"

#: The smallest valid PNG there is (1×1, opaque black).
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def make_image_model(key: str = IMAGE_KEY, *, name: str | None = None, family: str = "flux",
                     fmt: str = "mlx", size_bytes: int = 4_000_000_000) -> LocalModel:
    """A ``LocalModel`` that the image endpoints accept."""
    meta = ModelMeta(quant="4bit")
    meta.image_family = family                       # type: ignore[attr-defined]
    return LocalModel(
        key=key, format=fmt, path=f"/models/{key}", display_name=name or key.split("/")[-1],
        size_bytes=size_bytes, repo_id="/".join(key.split("/")[:2]), quant="4bit",
        capabilities=["image"], meta=meta)


def write_png(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(PNG_1X1)
    return target


def make_result(root: str | Path, *, image_id: str | None = None, prompt: str = "kot w kapeluszu",
                model_key: str = IMAGE_KEY, **over: Any) -> ImageResult:
    """An ``ImageResult`` whose file really exists under ``root``."""
    ident = image_id or uuid.uuid4().hex
    path = write_png(Path(root) / f"{ident}.png")
    fields: dict[str, Any] = {
        "id": ident, "path": str(path), "width": 1024, "height": 1024, "seed": 42,
        "model_key": model_key, "prompt": prompt, "steps": 4, "guidance": None,
        "elapsed_ms": 1234, "created_at": time.time(), "negative_prompt": "",
        "index": 0, "size_bytes": len(PNG_1X1), "model_name": model_key.split("/")[-1],
    }
    fields.update(over)
    return ImageResult(**fields)


class FakeImageRuntime:
    """``ImageRuntime`` stand-in: progress ticks, real files, no model."""

    def __init__(self, root: str | Path, *, steps: int = 4, delay: float = 0.0,
                 error: BaseException | None = None, status_message: str = "Ładuję model…",
                 hold: asyncio.Event | None = None) -> None:
        self.root = Path(root)
        self.steps = steps
        self.delay = delay
        self.error = error
        self.status_message = status_message
        self.hold = hold                       # set to stall inside generate()
        self.calls: list[tuple[str, ImageParams]] = []
        self.cancelled = 0
        self.unloaded: list[str] = []
        self.shutdowns = 0
        self.instances: list[dict[str, Any]] = []

    async def generate(self, model_key: str, params: ImageParams, *,
                       on_progress: Callable[[Any], None] | None = None) -> list[ImageResult]:
        self.calls.append((model_key, params))
        if self.error is not None:
            raise self.error
        count = max(1, int(getattr(params, "count", 1) or 1))
        steps = int(getattr(params, "steps", None) or self.steps)
        results: list[ImageResult] = []
        try:
            if on_progress is not None and self.status_message:
                on_progress(ImageProgress(message=self.status_message, count=count))
            for index in range(count):
                for step in range(1, steps + 1):
                    if on_progress is not None:
                        on_progress(ImageProgress(index=index, step=step, steps=steps,
                                                  count=count))
                    if self.hold is not None:
                        await self.hold.wait()
                    if self.delay:
                        await asyncio.sleep(self.delay)
                    else:
                        await asyncio.sleep(0)
                results.append(make_result(
                    self.root, prompt=getattr(params, "prompt", ""), model_key=model_key,
                    width=int(getattr(params, "width", 1024)),
                    height=int(getattr(params, "height", 1024)),
                    seed=int(getattr(params, "seed", None) or 1000 + index),
                    steps=steps, guidance=getattr(params, "guidance", None), index=index,
                    negative_prompt=getattr(params, "negative_prompt", "")))
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return results

    def status(self) -> list[dict[str, Any]]:
        return list(self.instances)

    async def unload(self, instance_id: str) -> None:
        self.unloaded.append(instance_id)

    async def unload_all(self) -> None:
        self.unloaded.append("all")

    async def shutdown(self) -> None:
        self.shutdowns += 1
