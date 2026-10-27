"""Concurrency tests for ``localcode/images/runtime.py`` with a fake engine.

Covers the two things that silently cost the user memory: an instance that a
cancelled load leaves behind in the table, and the idle reaper unloading a model
the caller has just taken.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from localcode.config import Settings
from localcode.images import runtime as imgrt
from localcode.images.base import ImageEngine, ImageParams, ImageProgress, ImageResult
from localcode.images.runtime import ImageRuntime
from tests.images.images_fakes import IMAGE_KEY, make_image_model, write_png


class FakeImageEngine(ImageEngine):
    """Loads instantly (or after ``delay``) and writes one 1×1 PNG per request."""

    backend = "fake"
    format = "mlx"

    def __init__(self, instance_id: str, model: Any, *, root: Path, delay: float = 0.0) -> None:
        super().__init__(instance_id, model)
        self.root = root
        self.delay = delay
        self.stops = 0
        self._running = False

    async def start(self) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        self._running = True

    async def stop(self) -> None:
        self.stops += 1
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def generate(self, params: ImageParams, *, output_dir=None,
                       ids=None) -> AsyncIterator[ImageProgress | ImageResult]:
        ident = (ids or ["x"])[0]
        path = write_png(Path(output_dir or self.root) / f"{ident}.png")
        yield ImageProgress(index=0, count=1, step=1, steps=1)
        yield ImageResult(id=ident, path=str(path), width=params.width, height=params.height,
                          seed=1, model_key=self.model.key, prompt=params.prompt, steps=1,
                          guidance=None, elapsed_ms=1, created_at=0.0, negative_prompt="",
                          index=0, size_bytes=100, model_name=self.model.display_name)

    async def cancel(self) -> bool:
        return True


class _Library:
    def __init__(self, models: list[Any]) -> None:
        self.models = {m.key: m for m in models}

    def get(self, key: str) -> Any:
        return self.models.get(key)

    def list(self) -> list[Any]:
        return list(self.models.values())

    def load_defaults(self, key: str) -> None:
        return None


def make_runtime(tmp_path: Path, *, delay: float = 0.0, ttl_minutes: float = 0.0) -> ImageRuntime:
    settings = Settings(models_dir=str(tmp_path / "models"))
    settings.images.output_dir = str(tmp_path / "out")
    settings.images.idle_ttl_minutes = ttl_minutes
    settings.runtime.ram_limit_gb = 20
    model = make_image_model(size_bytes=1_000_000)
    return ImageRuntime(
        settings, _Library([model]),
        engine_factory=lambda iid, m: FakeImageEngine(iid, m, root=tmp_path / "eng", delay=delay))


async def test_cancelled_load_leaves_no_instance_holding_the_budget(tmp_path: Path) -> None:
    """A cancelled load used to stay in the table forever, reserving its RAM."""
    runtime = make_runtime(tmp_path, delay=0.2)
    try:
        load = asyncio.create_task(runtime.load(IMAGE_KEY))
        await asyncio.sleep(0.02)
        next(iter(runtime._bg)).cancel()               # the background load task
        with pytest.raises(BaseException):
            await load
        await asyncio.sleep(0.05)
        assert runtime._instances == {}
        assert sum(i.reserved_bytes for i in runtime._instances.values()) == 0
    finally:
        await runtime.shutdown()


async def test_load_restarts_the_idle_timer(tmp_path: Path, monkeypatch) -> None:
    """Explicitly loading a model that is already up must not leave it a heartbeat
    away from the idle reaper."""
    monkeypatch.setattr(imgrt, "REAPER_INTERVAL_S", 0.01)
    runtime = make_runtime(tmp_path, ttl_minutes=0.0015)      # 90 ms
    try:
        info = await runtime.load(IMAGE_KEY)
        await asyncio.sleep(0.07)
        await runtime.load(IMAGE_KEY)                         # restarts the TTL
        await asyncio.sleep(0.07)
        assert info["id"] in runtime._instances
        await asyncio.sleep(0.12)
        assert runtime._instances == {}                       # and it is still reaped later
    finally:
        await runtime.shutdown()


async def test_unloading_twice_is_not_an_error(tmp_path: Path) -> None:
    """A second "Wyładuj" — or a click on a stale status snapshot — is a no-op.

    The language runtime had the same defect and answered 409 for an intent it
    had already fulfilled, which the app showed as an error banner.
    """
    runtime = make_runtime(tmp_path)
    try:
        instance = await runtime.load(IMAGE_KEY)
        await runtime.unload(instance["id"])
        await runtime.unload(instance["id"])            # must not raise

        with pytest.raises(Exception):
            await runtime.unload("nigdy-nie-istniał")
    finally:
        await runtime.shutdown()
