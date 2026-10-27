"""Fixtures for the HTTP server tests.

Every test runs against a real :class:`~localcode.server.core.Core` (real chat
database, real ``ChatApi``) whose library, runtime, hub and download manager are
fakes — so the tests exercise the HTTP layer, not the model machinery.
``tests/chat`` is put on ``sys.path`` so the chat fakes and the ``Models.swift``
shape checker can be reused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CHAT_TESTS = Path(__file__).resolve().parent.parent / "chat"
if str(_CHAT_TESTS) not in sys.path:
    sys.path.insert(0, str(_CHAT_TESTS))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from localcode.chat.api import ChatApi  # noqa: E402
from localcode.config import Settings  # noqa: E402
from localcode.server.app import create_app  # noqa: E402
from localcode.server.core import Core  # noqa: E402

from chat_fakes import make_service  # noqa: E402
from server_fakes import (  # noqa: E402
    CHAT_KEY,
    CHAT_KEY_2,
    EMBED_KEY,
    FakeDownloads,
    FakeEngine,
    FakeHub,
    FakeLibrary,
    FakeRuntime,
    ScriptedAgent,
    make_model,
)

TOKEN = "test-token-12345"


@pytest.fixture
def settings(tmp_path: Path, isolated_home: Path) -> Settings:
    return Settings(models_dir=str(tmp_path / "models"))


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def library() -> FakeLibrary:
    return FakeLibrary([
        make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["thinking", "tools"]),
        make_model(CHAT_KEY_2, name="Qwen3 4B Instruct", fmt="mlx"),
        make_model(EMBED_KEY, name="Nomic Embed", caps=["embedding"], embedding=True),
    ])


@pytest.fixture
def hub() -> FakeHub:
    return FakeHub()


@pytest.fixture
def runtime(engine: FakeEngine) -> FakeRuntime:
    return FakeRuntime({CHAT_KEY: engine, CHAT_KEY_2: engine, EMBED_KEY: FakeEngine()})


@pytest.fixture
def agent_behaviour():
    """Overridden by a test to script the agent (``async def (agent, text)``)."""
    return None


@pytest.fixture
def agents() -> list[ScriptedAgent]:
    return []


@pytest.fixture
async def core(settings: Settings, tmp_path: Path, library: FakeLibrary, hub: FakeHub,
               runtime: FakeRuntime, agent_behaviour, agents: list[ScriptedAgent]) -> Core:
    def agent_factory(session):
        agent = ScriptedAgent(session, agent_behaviour)
        agents.append(agent)
        return agent

    instance = Core(settings, db_path=tmp_path / "chat.db", library=library, hub=hub,
                    runtime=runtime, agent_factory=agent_factory)
    # The chat service gets the simple formatter/splitter the chat tests use.
    instance.chat = make_service(instance.repo, runtime, library, settings)
    instance.chat_api = ChatApi(instance.repo, instance.chat)
    instance.downloads_fake = FakeDownloads(settings, instance._on_download_event)  # type: ignore[attr-defined]
    instance._downloads = instance.downloads_fake                                    # type: ignore[attr-defined]
    instance._downloads_built = True                                                 # type: ignore[attr-defined]
    runtime.on_event = instance.events.publish
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
async def client(core: Core) -> TestClient:
    app = create_app(core, TOKEN)
    test_client = TestClient(TestServer(app), headers={"Authorization": f"Bearer {TOKEN}"})
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()


@pytest.fixture
async def anon_client(core: Core) -> TestClient:
    """Client without the Authorization header (auth enforcement tests)."""
    app = create_app(core, TOKEN)
    test_client = TestClient(TestServer(app))
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()
