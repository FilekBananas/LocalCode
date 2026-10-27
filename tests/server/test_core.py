"""``server/core.py``: the event bus, settings persistence and session resume."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from localcode import config as lc_config
from localcode.chat.repo import ApiError
from localcode.config import Settings
from localcode.server.core import Core, EventBus
from server_fakes import CHAT_KEY, FakeHub, FakeLibrary, FakeRuntime


# --------------------------------------------------------------------------- #
# Event bus
# --------------------------------------------------------------------------- #


async def test_event_bus_fans_out() -> None:
    bus = EventBus()
    bus.bind()
    first, second = bus.subscribe(), bus.subscribe()
    bus.publish({"type": "library"})
    assert await first.get(timeout=1) == {"type": "library"}
    assert await second.get(timeout=1) == {"type": "library"}
    first.close()
    assert bus.subscribers == 1


async def test_event_bus_drops_oldest_for_slow_listeners() -> None:
    bus = EventBus(maxsize=2)
    bus.bind()
    sub = bus.subscribe()
    for i in range(5):
        bus.publish({"type": "runtime", "n": i})
    assert sub.dropped == 3
    assert (await sub.get(timeout=1))["n"] == 3        # the two newest survive
    assert (await sub.get(timeout=1))["n"] == 4


async def test_event_bus_accepts_event_objects() -> None:
    from localcode.events import Status

    bus = EventBus()
    bus.bind()
    sub = bus.subscribe()
    bus.publish(Status(message="Ładowanie modelu…", phase="loading"))
    assert await sub.get(timeout=1) == {"type": "status", "message": "Ładowanie modelu…",
                                        "phase": "loading"}


async def test_event_bus_from_another_thread() -> None:
    bus = EventBus()
    bus.bind()
    sub = bus.subscribe()
    await asyncio.to_thread(bus.publish, {"type": "library"})
    assert await sub.get(timeout=1) == {"type": "library"}


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


async def test_start_and_shutdown_are_idempotent(core: Core, runtime: FakeRuntime,
                                                 library: FakeLibrary) -> None:
    await core.start()                       # the fixture already started it
    assert library.scans >= 1
    await core.shutdown()
    await core.shutdown()
    assert runtime.shutdowns == 1
    assert core.downloads_fake.shutdowns == 1


async def test_rescan_emits_library_event(core: Core, library: FakeLibrary) -> None:
    sub = core.events.subscribe()
    await core.rescan_library()
    assert await sub.get(timeout=1) == {"type": "library"}
    sub.close()


async def test_downloads_missing_module(settings: Settings, tmp_path: Path) -> None:
    instance = Core(settings, db_path=tmp_path / "chat.db", library=FakeLibrary([]),
                    hub=FakeHub(), runtime=FakeRuntime({}))
    try:
        instance._downloads_built = True     # pretend the import failed
        instance._downloads = None
        assert instance.downloads is None
        with pytest.raises(ApiError) as excinfo:
            instance.require_downloads()
        assert excinfo.value.status == 503
    finally:
        await instance.shutdown()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


async def test_app_settings_patch_is_persisted(core: Core, isolated_home: Path) -> None:
    await core.update_app_settings({"language": "en", "agent": {"maxSteps": 12}})
    stored = json.loads((isolated_home / "settings.json").read_text())
    assert stored["language"] == "en" and stored["agent"]["max_steps"] == 12
    assert lc_config.load_settings().agent.max_steps == 12
    # the in-place update keeps the object the runtime manager holds
    assert core.settings is core.chat.settings_provider()
    assert core.settings.agent.max_steps == 12


async def test_hf_token_is_stored_with_0600(core: Core, isolated_home: Path) -> None:
    await core.set_hf_token("hf_tajne")
    path = isolated_home / "settings.json"
    assert json.loads(path.read_text())["hf_token"] == "hf_tajne"
    assert oct(path.stat().st_mode)[-3:] == "600"


async def test_resolve_model_by_substring(core: Core) -> None:
    assert core.resolve_model(CHAT_KEY).key == CHAT_KEY
    assert core.resolve_model("Qwen3-0.6B-Q4_K_M").key == CHAT_KEY
    assert core.resolve_model("nie-ma") is None
    assert core.resolve_model(None) is None


# --------------------------------------------------------------------------- #
# Agent session persistence
# --------------------------------------------------------------------------- #


async def test_sessions_survive_a_restart(core: Core, settings: Settings, tmp_path: Path,
                                          isolated_home: Path) -> None:
    session = core.agents.create(str(tmp_path), CHAT_KEY, "plan")
    session.store.set_title("Refaktoryzacja")
    session.store.append_item({"kind": "user", "id": "i1", "text": "cześć", "createdAt": 1})
    await core.shutdown()

    fresh = Core(settings, db_path=tmp_path / "chat2.db", library=core.library, hub=core.hub,
                 runtime=core.runtime)
    try:
        listed = fresh.agents.list()
        assert [s["id"] for s in listed] == [session.id]
        assert listed[0]["title"] == "Refaktoryzacja"
        assert listed[0]["mode"] == "plan" and listed[0]["modelKey"] == CHAT_KEY

        resumed = fresh.agents.require(session.id)
        assert [i["text"] for i in resumed.items] == ["cześć"]
        assert fresh.agents.list(str(tmp_path))[0]["id"] == session.id
        assert fresh.agents.list("/") == []

        assert await fresh.agents.delete(session.id) is True
        assert fresh.agents.list() == []
    finally:
        await fresh.shutdown()


async def test_session_files_land_under_localcode_home(core: Core, tmp_path: Path,
                                                       isolated_home: Path) -> None:
    session = core.agents.create(str(tmp_path))
    files = list((isolated_home / "sessions").rglob("*.jsonl"))
    assert len(files) == 1 and files[0].stem == session.id
    header = json.loads(files[0].read_text().splitlines()[0])
    assert header["kind"] == "session" and header["cwd"] == str(tmp_path.resolve())


def test_a_running_turn_takes_the_queued_text_messages_in_order() -> None:
    import collections
    from types import SimpleNamespace

    from localcode.server.core import AgentSessionState

    image = ("z obrazem", [{"data": "AAAA"}])
    state = SimpleNamespace(queued=collections.deque(["pierwsza", "druga", image, "po obrazie"]))
    assert AgentSessionState._take_queued_texts(state) == ["pierwsza", "druga"]
    # the picture waits for its own turn, and so does what came after it
    assert list(state.queued) == [image, "po obrazie"]


async def test_a_deepinfra_key_makes_deepseek_flash_the_default(core) -> None:
    from localcode.server.core import PREFERRED_REMOTE_MODELS

    await core._persist({"agent": {"default_model": "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"}})
    await core.set_remote_key("deepinfra", "sk-test")
    flash = PREFERRED_REMOTE_MODELS["deepinfra"]
    assert core.settings.agent.default_model == flash and core.settings.chat.default_model == flash
    # a remote default the user chose stays
    await core._persist({"agent": {"default_model": "remote:unlid:glm-5"}})
    await core.set_remote_key("deepinfra", "sk-test-2")
    assert core.settings.agent.default_model == "remote:unlid:glm-5"
