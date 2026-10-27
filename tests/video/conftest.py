"""Fixtures for the video tests.

Every HTTP-layer test runs against a real :class:`~localcode.server.core.Core`;
only the model machinery (library, LLM runtime, hub) and the video runtime
itself are faked — the same shape as ``tests/images/conftest.py``.
``tests/server`` and ``tests/chat`` are put on ``sys.path`` so their fakes
(``FakeLibrary``, ``FakeRuntime``, ``FakeHub``, ``make_model``) can be reused
instead of inventing a second set; ``routes_video.py`` reads the video runtime off
``core._video`` (a lazy attribute, like ``core._speech``), so the fixture
below just sets it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for _sibling in ("server", "chat"):
    _path = _HERE.parent / _sibling
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from localcode.config import Settings  # noqa: E402
from localcode.server.app import create_app  # noqa: E402
from localcode.server.core import Core  # noqa: E402

from server_fakes import CHAT_KEY, FakeEngine, FakeHub, FakeLibrary, FakeRuntime, make_model  # noqa: E402

from video_fakes import make_ltx_model  # noqa: E402

TOKEN = "test-token-12345"


@pytest.fixture
def settings(tmp_path: Path, isolated_home: Path) -> Settings:
    return Settings(models_dir=str(tmp_path / "models"))


@pytest.fixture
def library() -> FakeLibrary:
    return FakeLibrary([
        make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["tools"]),
        make_ltx_model(),
    ])


@pytest.fixture
def video(monkeypatch: pytest.MonkeyPatch):
    """A :class:`~video_fakes.FakeVideoRuntime`, injected as ``core._video``."""
    from video_fakes import FakeVideoRuntime

    return FakeVideoRuntime()


@pytest.fixture
async def core(settings: Settings, tmp_path: Path, library: FakeLibrary, video) -> Core:
    instance = Core(settings, db_path=tmp_path / "chat.db", library=library, hub=FakeHub(),
                    runtime=FakeRuntime({CHAT_KEY: FakeEngine()}))
    instance._video = video  # noqa: SLF001 - the route's own lazy attribute, same as core._speech
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
async def client(core: Core) -> TestClient:
    test_client = TestClient(TestServer(create_app(core, TOKEN)),
                             headers={"Authorization": f"Bearer {TOKEN}"})
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()


@pytest.fixture
async def anon_client(core: Core) -> TestClient:
    """Client without the Authorization header (auth enforcement tests)."""
    test_client = TestClient(TestServer(create_app(core, TOKEN)))
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()
