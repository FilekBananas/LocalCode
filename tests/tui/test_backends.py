"""Backend selection, server discovery and ``RemoteBackend`` against a live server.

The remote tests run the real aiohttp application (``localcode.server.app``) on a
loopback port with the HTTP layer's fakes behind it, so the WebSocket protocol,
the bearer token and every REST call the terminal makes are exercised end to
end.  They skip when the server package cannot be imported.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from localcode.config import Settings
from localcode.tui.backends import (
    LocalBackend,
    RemoteBackend,
    ServerInfo,
    discover_server,
    open_backend,
    read_server_info,
)

aiohttp = pytest.importorskip("aiohttp")
pytest.importorskip("localcode.server.app")

from aiohttp.test_utils import TestServer  # noqa: E402

from localcode.server.app import create_app  # noqa: E402
from localcode.server.core import Core  # noqa: E402
from server_fakes import (  # noqa: E402
    CHAT_KEY,
    CHAT_KEY_2,
    FakeEngine,
    FakeHub,
    FakeLibrary,
    FakeRuntime,
    ScriptedAgent,
    TextDelta,
    make_model,
)

TOKEN = "tui-test-token"


# --------------------------------------------------------------------------- #
# Live server fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def behaviour():
    """Overridden by a test to script the agent (``async def (agent, text)``)."""
    return None


@pytest.fixture
async def core(tmp_path: Path, isolated_home: Path, behaviour) -> AsyncIterator[Core]:
    library = FakeLibrary([
        make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["tools", "thinking"]),
        make_model(CHAT_KEY_2, name="Qwen3 4B Instruct", fmt="mlx"),
    ])
    runtime = FakeRuntime({CHAT_KEY: FakeEngine(), CHAT_KEY_2: FakeEngine()})
    agents: list[ScriptedAgent] = []

    def agent_factory(session: Any) -> ScriptedAgent:
        agent = ScriptedAgent(session, behaviour)
        agents.append(agent)
        return agent

    instance = Core(Settings(models_dir=str(tmp_path / "models")), db_path=tmp_path / "chat.db",
                    library=library, hub=FakeHub(), runtime=runtime, agent_factory=agent_factory)
    instance.agents_created = agents  # type: ignore[attr-defined]
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
async def server(core: Core) -> AsyncIterator[ServerInfo]:
    test_server = TestServer(create_app(core, TOKEN))
    await test_server.start_server()
    try:
        yield ServerInfo("127.0.0.1", test_server.port, TOKEN, pid=1, version="0.1.0")
    finally:
        await test_server.close()


@pytest.fixture
async def remote(server: ServerInfo, tmp_path: Path) -> AsyncIterator[RemoteBackend]:
    backend = RemoteBackend(server, tmp_path, Settings(), model_key=CHAT_KEY)
    await backend.start()
    try:
        yield backend
    finally:
        await backend.aclose()


async def drain(backend: RemoteBackend, until: str, *, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Collect events until one of type ``until`` arrives."""
    seen: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        while True:
            event = await backend.events.get()
            seen.append(event)
            if event.get("type") == until:
                return seen


# --------------------------------------------------------------------------- #
# server.json discovery
# --------------------------------------------------------------------------- #


def test_read_server_info_without_a_file(isolated_home: Path) -> None:
    assert read_server_info() is None


def test_read_server_info(isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text(
        json.dumps({"port": 8421, "token": "t", "pid": 5, "version": "0.1.0",
                    "host": "127.0.0.1"}), encoding="utf-8")
    info = read_server_info()
    assert info is not None
    assert info.base_url == "http://127.0.0.1:8421"
    assert info.ws_url("/x") == "ws://127.0.0.1:8421/x"
    assert (info.token, info.pid) == ("t", 5)


def test_broken_server_json_is_ignored(isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text("{nope", encoding="utf-8")
    assert read_server_info() is None


async def test_discover_server_without_a_file(isolated_home: Path) -> None:
    assert await discover_server() is None


async def test_discover_server_finds_a_live_one(server: ServerInfo, isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text(
        json.dumps({"port": server.port, "token": TOKEN, "host": "127.0.0.1", "pid": 1}),
        encoding="utf-8")
    found = await discover_server()
    assert found is not None
    assert found.port == server.port
    assert found.token == TOKEN


async def test_discover_server_ignores_a_dead_port(isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text(
        json.dumps({"port": 1, "token": "t", "host": "127.0.0.1"}), encoding="utf-8")
    assert await discover_server(timeout=0.5) is None


# --------------------------------------------------------------------------- #
# RemoteBackend
# --------------------------------------------------------------------------- #


async def test_remote_creates_a_session(remote: RemoteBackend) -> None:
    assert remote.kind == "remote"
    assert remote.session_id
    assert remote.model_key == CHAT_KEY


async def test_remote_streams_a_turn(remote: RemoteBackend) -> None:
    await remote.send_user("cześć")
    events = await drain(remote, "turn_end")
    kinds = [e["type"] for e in events]
    assert "turn_start" in kinds
    assert "text_delta" in kinds
    text = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert text == "Echo: cześć"
    assert remote.context_limit == 8192


async def test_remote_tracks_context_from_events(remote: RemoteBackend) -> None:
    await remote.send_user("x")
    await drain(remote, "turn_end")
    assert remote.context_used == 120
    assert remote.busy is False


async def test_remote_set_mode_and_model(remote: RemoteBackend, core: Core) -> None:
    await remote.set_mode("plan")
    await remote.set_model(CHAT_KEY_2)
    await asyncio.sleep(0.15)
    session = core.agents.get(str(remote.session_id))
    assert session is not None
    assert session.mode == "plan"
    assert session.model_key == CHAT_KEY_2


async def test_remote_clear_and_compact(remote: RemoteBackend) -> None:
    await remote.send_user("x")
    await drain(remote, "turn_end")
    await remote.compact()
    await remote.clear()
    await asyncio.sleep(0.15)
    assert remote.context_used == 0


@pytest.mark.parametrize("behaviour", [None])
async def test_remote_library_and_runtime(remote: RemoteBackend) -> None:
    models = await remote.list_models()
    assert {m["key"] for m in models} == {CHAT_KEY, CHAT_KEY_2}
    assert await remote.resolve_model_key("Qwen3-0.6B") == CHAT_KEY

    status = await remote.runtime_status()
    assert status["settings"]["maxLoadedModels"] == 2

    instance = await remote.load_model(CHAT_KEY, n_ctx=4096)
    assert instance["modelKey"] == CHAT_KEY
    assert instance["nCtx"] == 4096
    await remote.unload(everything=True)
    assert (await remote.runtime_status())["instances"] == []


async def test_remote_runtime_settings(remote: RemoteBackend) -> None:
    status = await remote.update_runtime_settings({"maxLoadedModels": 3})
    assert status["settings"]["maxLoadedModels"] in (2, 3)  # the fake echoes its own defaults


async def test_remote_hub(remote: RemoteBackend) -> None:
    results = await remote.hub_search("qwen", fmt="gguf")
    assert results and results[0]["format"] == "gguf"
    assert await remote.hub_curated() is not None
    details = await remote.hub_details("unsloth/Qwen3-4B-Instruct-2507-GGUF")
    assert "variants" in details


async def test_remote_sessions_and_resume(remote: RemoteBackend) -> None:
    sessions = await remote.sessions()
    assert [s["id"] for s in sessions] == [remote.session_id]
    items = await remote.resume_session(str(remote.session_id))
    assert isinstance(items, list)


async def test_remote_reports_http_errors(remote: RemoteBackend) -> None:
    from localcode.tui.backends import BackendError

    with pytest.raises(BackendError) as excinfo:
        await remote.load_model("nie/ma/takiego")
    assert "nie zna modelu" in str(excinfo.value)


async def test_remote_rejects_a_bad_token(server: ServerInfo, tmp_path: Path) -> None:
    from localcode.tui.backends import BackendError

    bad = ServerInfo(server.host, server.port, "zly-token")
    backend = RemoteBackend(bad, tmp_path, Settings())
    with pytest.raises(BackendError):
        await backend.start()
    await backend.aclose()


# --------------------------------------------------------------------------- #
# The permission round-trip over the WebSocket
# --------------------------------------------------------------------------- #


async def _asks_for_permission(agent: ScriptedAgent, text: str) -> None:
    decision = await agent.ask(tool="bash", title="Uruchom: pytest -q", preview="pytest -q")
    agent.emit(TextDelta(text=f"decyzja={decision}"))


@pytest.mark.parametrize("behaviour", [_asks_for_permission])
async def test_permission_round_trip(remote: RemoteBackend) -> None:
    await remote.send_user("uruchom testy")
    request = None
    async with asyncio.timeout(5):
        while request is None:
            event = await remote.events.get()
            if event.get("type") == "permission_request":
                request = event
    assert request["tool"] == "bash"
    assert request["preview"] == "pytest -q"
    await remote.answer_permission(str(request["requestId"]), "allow")
    events = await drain(remote, "turn_end")
    text = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert text == "decyzja=allow"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


async def test_open_backend_requires_a_server_with_attach(isolated_home: Path,
                                                          tmp_path: Path) -> None:
    from localcode.tui.backends import BackendError

    with pytest.raises(BackendError) as excinfo:
        await open_backend(cwd=tmp_path, prefer="remote")
    assert "localcode serve" in str(excinfo.value)


async def test_open_backend_rejects_an_unknown_preference(tmp_path: Path) -> None:
    from localcode.tui.backends import BackendError

    with pytest.raises(BackendError):
        await open_backend(cwd=tmp_path, prefer="magic")


async def test_open_backend_uses_the_server_when_it_runs(server: ServerInfo, tmp_path: Path,
                                                         isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text(
        json.dumps({"port": server.port, "token": TOKEN, "host": "127.0.0.1"}), encoding="utf-8")
    backend = await open_backend(cwd=tmp_path, prefer="auto")
    try:
        assert isinstance(backend, RemoteBackend)
        assert backend.kind == "remote"
    finally:
        await backend.aclose()


async def test_open_backend_falls_back_to_local(isolated_home: Path, tmp_path: Path) -> None:
    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    backend = await open_backend(cwd=tmp_path, prefer="auto", settings=settings,
                                 store_session=False)
    try:
        assert isinstance(backend, LocalBackend)
        assert backend.kind == "local"
        assert await backend.list_models() == []       # nothing downloaded in this home
        assert backend.model_key is None
    finally:
        await backend.aclose()


async def test_local_backend_rejects_an_unknown_model(isolated_home: Path, tmp_path: Path) -> None:
    from localcode.tui.backends import BackendError

    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    with pytest.raises(BackendError) as excinfo:
        await open_backend(cwd=tmp_path, prefer="local", settings=settings,
                           model_key="nie/ma", store_session=False)
    assert "Nie znaleziono modelu" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Agent sessions are created lazily and cleaned up
# --------------------------------------------------------------------------- #


async def test_read_only_commands_create_no_agent_session(server: ServerInfo, core: Core,
                                                          tmp_path: Path) -> None:
    backend = RemoteBackend(server, tmp_path, Settings(), attach_agent=False)
    await backend.start()
    try:
        assert await backend.runtime_status() is not None
        assert backend.session_id is None
        assert len(core.agents) == 0
    finally:
        await backend.aclose()


async def test_the_session_appears_on_the_first_message(server: ServerInfo, core: Core,
                                                        tmp_path: Path) -> None:
    backend = RemoteBackend(server, tmp_path, Settings(), model_key=CHAT_KEY, attach_agent=False)
    await backend.start()
    try:
        await backend.send_user("cześć")
        await drain(backend, "turn_end")
        assert backend.session_id is not None
        assert len(core.agents) == 1
    finally:
        await backend.aclose()


async def test_closing_removes_the_session_we_created(server: ServerInfo, core: Core,
                                                      tmp_path: Path) -> None:
    backend = RemoteBackend(server, tmp_path, Settings(), model_key=CHAT_KEY)
    await backend.start()
    assert len(core.agents) == 1
    await backend.aclose()
    assert len(core.agents) == 0


# --------------------------------------------------------------------------- #
# Choosing a model when the user named none
# --------------------------------------------------------------------------- #


async def test_remote_picks_a_default_model(server: ServerInfo, tmp_path: Path) -> None:
    backend = RemoteBackend(server, tmp_path, Settings(), attach_agent=False)
    await backend.start()
    try:
        assert backend.model_key in (CHAT_KEY, CHAT_KEY_2)
    finally:
        await backend.aclose()


# --------------------------------------------------------------------------- #
# Session files (-c / --resume)
# --------------------------------------------------------------------------- #


async def test_local_backend_resumes_a_stored_session(isolated_home: Path, tmp_path: Path) -> None:
    """``-c`` must restore the history even with ``--no-session`` (no new file)."""
    from localcode.agent.session import SessionStore

    store = SessionStore.create(tmp_path, model_key=CHAT_KEY, mode="default")
    store.append_message({"role": "user", "content": "pierwsza wiadomość"})
    store.append_message({"role": "assistant", "content": "odpowiedź"})

    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    backend = LocalBackend(tmp_path, settings, session_id=store.session.id, store_session=False)
    await backend.start()
    try:
        backend.model_key = CHAT_KEY          # no library in this temp home
        agent = await backend._ensure_agent()  # noqa: SLF001 - the point of the test
        assert [m["content"] for m in agent.history] == ["pierwsza wiadomość", "odpowiedź"]
        assert backend.session_id == store.session.id
    finally:
        await backend.aclose()


async def test_local_backend_without_a_session_writes_nothing(isolated_home: Path,
                                                              tmp_path: Path) -> None:
    from localcode.agent import session as session_mod

    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    backend = LocalBackend(tmp_path, settings, store_session=False)
    await backend.start()
    try:
        backend.model_key = CHAT_KEY
        await backend._ensure_agent()  # noqa: SLF001
        assert backend.session_id is None
        assert session_mod.list_sessions(str(tmp_path)) == []
    finally:
        await backend.aclose()
