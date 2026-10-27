"""Pomocnik — the help bot that runs the rest of the app for the user.

It has full power on purpose (models, downloads, every setting), and runs on a
4B model. So what is worth pinning is not that the tools work, but where the
power stops: everything that changes something is shown first, a key never
enters the conversation, it cannot delete itself or unload itself through
"unload all", and loading a model for the user does not turn it into that model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

_AGENT_TESTS = Path(__file__).resolve().parent.parent / "agent"
if str(_AGENT_TESTS) not in sys.path:
    sys.path.insert(0, str(_AGENT_TESTS))

from agent_fakes import EventLog, FakeRuntime, ScriptedEngine, fake_model, tool_call_text  # noqa: E402

from localcode import helpbot  # noqa: E402
from localcode.agent.permissions import PermissionPolicy  # noqa: E402
from localcode.agent.tools.base import ToolContext  # noqa: E402
from localcode.config import Settings  # noqa: E402
from localcode.helpbot.ops import HelpbotError, HelpbotOps  # noqa: E402
from localcode.helpbot.tools import HELPBOT_TOOL_CLASSES, STATE_KEY, helpbot_tools  # noqa: E402

HELPBOT_KEY = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
BIG_KEY = "org/Ogromny-70B"


# --------------------------------------------------------------------------- #
# A Core-shaped fake: library, runtime, downloads and the settings endpoints
# --------------------------------------------------------------------------- #


class FakeLibrary:
    def __init__(self) -> None:
        self.models = {k: fake_model(k, "mlx") for k in (HELPBOT_KEY, BIG_KEY, "org/Maly-7B")}
        self.defaults: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def get(self, key: str) -> Any:
        return self.models.get(key)

    def resolve(self, key: str) -> Any:
        return next((m for k, m in self.models.items() if key and key.lower() in k.lower()), None)

    def list(self) -> list[Any]:
        return list(self.models.values())

    def load_defaults(self, key: str) -> dict[str, Any] | None:
        return self.defaults.get(key)

    def set_load_defaults(self, key: str, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value and "nCtx" in value and not isinstance(value["nCtx"], int):
            raise ValueError("Wartość „nCtx” musi być liczbą całkowitą.")
        if value:
            self.defaults[key] = dict(value)
        else:
            self.defaults.pop(key, None)
        return value

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.models.pop(key, None)


class HelpbotRuntime(FakeRuntime):
    """The agent's fake runtime plus what the help bot asks of a real one."""

    def __init__(self, engine: Any) -> None:
        super().__init__(engine, model_key=HELPBOT_KEY)
        self.instances = [{"id": "i-bot", "modelKey": HELPBOT_KEY, "displayName": "Pomocnik"},
                          {"id": "i-7b", "modelKey": "org/Maly-7B", "displayName": "Mały 7B"}]
        self.unloaded: list[str] = []
        self.loads: list[tuple[str, Any]] = []
        self.runtime_patches: list[dict[str, Any]] = []

    def status(self) -> dict[str, Any]:
        return {"instances": list(self.instances),
                "budget": {"limitBytes": 20 * 10**9, "usedBytes": 5 * 10**9},
                "settings": {"ramLimitGb": None, "maxLoadedModels": 2}}

    def estimate(self, key: str) -> Any:
        fits = "no" if key == BIG_KEY else "yes"
        message = "Za duży: 40 GB przy budżecie 20 GB." if fits == "no" else "Zmieści się."
        return type("Est", (), {"fits": fits, "message": message, "total_bytes": 1})()

    async def load(self, key: str, config: Any = None) -> dict[str, Any]:
        self.loads.append((key, config))
        return {"id": f"i-{key}", "modelKey": key}

    async def unload(self, instance_id: str) -> None:
        self.unloaded.append(instance_id)
        self.instances = [i for i in self.instances if i["id"] != instance_id]

    async def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        self.runtime_patches.append(patch)
        return self.status()


class FakeCore:
    def __init__(self, engine: Any = None) -> None:
        self.settings = Settings()
        self.library = FakeLibrary()
        self.runtime = HelpbotRuntime(engine or ScriptedEngine([]))
        self.downloads = None
        self.app_patches: list[dict[str, Any]] = []

    def app_settings(self) -> dict[str, Any]:
        return {"agent": {"maxSteps": self.settings.agent.max_steps},
                "helpbot": {"model": self.settings.helpbot.model}}

    async def update_app_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        self.app_patches.append(patch)
        if "agent" in patch and "maxSteps" in patch["agent"]:
            self.settings.agent.max_steps = int(patch["agent"]["maxSteps"])
        return self.app_settings()


@pytest.fixture
def core() -> FakeCore:
    return FakeCore()


@pytest.fixture
def ctx(core: FakeCore, tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, state={STATE_KEY: HelpbotOps(core, helpbot_key=HELPBOT_KEY)})


def tool(name: str) -> Any:
    return next(t for t in helpbot_tools() if t.name == name)


def run(name: str, args: dict[str, Any], ctx: ToolContext) -> Any:
    return asyncio.run(tool(name).run(args, ctx))


def decision(name: str, args: dict[str, Any], ctx: ToolContext) -> str:
    t = tool(name)
    info = asyncio.run(t.permission(args, ctx))
    return PermissionPolicy("default", cwd=str(ctx.cwd)).decide(t, args, info)


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #


def test_the_menu_covers_what_the_app_can_do() -> None:
    names = {cls.name for cls in HELPBOT_TOOL_CLASSES}
    assert names == {"model_search", "model_info", "model_list", "downloads", "settings",
                     "model_settings", "doctor", "model_install", "model_load", "model_unload",
                     "model_delete"}


def test_every_schema_stays_one_sentence() -> None:
    """Paid for on every request by a 4B model."""
    for t in helpbot_tools():
        assert len(t.description) <= 200, t.name


def test_the_menu_has_no_file_or_shell_tools() -> None:
    names = {t.name for t in helpbot_tools()}
    assert not names & {"read_file", "write_file", "edit_file", "bash", "delegate"}


# --------------------------------------------------------------------------- #
# Where the power stops: confirmations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, args", [
    ("settings", {}),
    ("settings", {"section": "agent"}),
    ("model_settings", {"key": HELPBOT_KEY}),
    ("downloads", {}),
    ("model_list", {}),
    ("doctor", {}),
    ("model_info", {"name": "flux klein"}),
])
def test_looking_never_asks(ctx: ToolContext, name: str, args: dict[str, Any]) -> None:
    assert decision(name, args, ctx) == "allow"


@pytest.mark.parametrize("name, args", [
    ("settings", {"changes": {"agent": {"maxSteps": 60}}}),
    ("model_settings", {"key": "org/Maly-7B", "changes": {"nCtx": 32768}}),
    ("downloads", {"cancel": "job-1"}),
    ("model_install", {"name": "qwen coder"}),
    ("model_load", {"key": "org/Maly-7B"}),
    ("model_unload", {"key": "all"}),
    ("model_delete", {"key": "org/Maly-7B"}),
])
def test_every_change_is_shown_first(ctx: ToolContext, name: str, args: dict[str, Any]) -> None:
    assert decision(name, args, ctx) == "ask"


def test_deleting_says_how_much_it_frees_before_it_happens(ctx: ToolContext) -> None:
    info = asyncio.run(tool("model_delete").permission({"key": "org/Maly-7B"}, ctx))
    assert "GB" in info.title and "cofnąć" in info.preview


# --------------------------------------------------------------------------- #
# Where the power stops: what it refuses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("changes", [
    {"remote": {"unlid": {"apiKey": "ul-sekret"}}},
    {"hfToken": "hf_sekret"},
])
def test_a_key_never_enters_the_conversation(ctx: ToolContext, changes: dict[str, Any]) -> None:
    result = run("settings", {"changes": changes}, ctx)
    assert result.ok is False
    assert "Modele zdalne" in result.output


def test_an_unknown_settings_section_is_refused_by_name(ctx: ToolContext) -> None:
    result = run("settings", {"changes": {"turbo": {"on": True}}}, ctx)
    assert result.ok is False and "turbo" in result.output


def test_it_cannot_delete_its_own_model(ctx: ToolContext, core: FakeCore) -> None:
    result = run("model_delete", {"key": HELPBOT_KEY}, ctx)
    assert result.ok is False
    assert core.library.deleted == []


def test_unload_all_leaves_the_help_bot_loaded(ctx: ToolContext, core: FakeCore) -> None:
    """It holds its model only while it writes; unloading it mid-turn just
    makes the next sentence wait, and can push out what it was asked to load."""
    result = run("model_unload", {"key": "all"}, ctx)
    assert result.ok
    assert core.runtime.unloaded == ["i-7b"]


def test_a_model_that_will_not_fit_is_not_loaded(ctx: ToolContext, core: FakeCore) -> None:
    result = run("model_load", {"key": BIG_KEY}, ctx)
    assert result.ok is False
    assert "nie zmieści" in result.output.lower()
    assert core.runtime.loads == []


# --------------------------------------------------------------------------- #
# What it does
# --------------------------------------------------------------------------- #


def test_settings_go_through_the_apps_own_code_path(ctx: ToolContext, core: FakeCore) -> None:
    result = run("settings", {"changes": {"agent": {"maxSteps": 60},
                                          "runtime": {"maxLoadedModels": 3}}}, ctx)
    assert result.ok
    assert core.app_patches == [{"agent": {"maxSteps": 60}}]
    assert core.runtime.runtime_patches == [{"maxLoadedModels": 3}]


def test_a_rejected_setting_comes_back_as_the_validators_message(ctx: ToolContext,
                                                                 core: FakeCore) -> None:
    async def refuse(patch: dict[str, Any]) -> None:
        raise ValueError("Wartość „maxSteps” musi być w zakresie 1–500.")

    core.update_app_settings = refuse  # type: ignore[method-assign]
    result = run("settings", {"changes": {"agent": {"maxSteps": 9999}}}, ctx)
    assert result.ok is False and "1–500" in result.output


def test_load_configuration_is_merged_not_replaced(ctx: ToolContext, core: FakeCore) -> None:
    core.library.defaults["org/Maly-7B"] = {"nParallel": 2}
    run("model_settings", {"key": "org/Maly-7B", "changes": {"nCtx": 32768}}, ctx)
    assert core.library.defaults["org/Maly-7B"] == {"nParallel": 2, "nCtx": 32768}
    run("model_settings", {"key": "org/Maly-7B", "changes": {"nParallel": None}}, ctx)
    assert core.library.defaults["org/Maly-7B"] == {"nCtx": 32768}


def test_loading_for_the_user_keeps_the_conversation_with_the_help_bot(ctx: ToolContext,
                                                                       core: FakeCore) -> None:
    """The agent's own model_load hands the conversation over; this one must not."""
    ops = ctx.state[STATE_KEY]
    assert ops.on_switch is None
    result = run("model_load", {"key": "org/Maly-7B", "context": 16384}, ctx)
    assert result.ok
    key, config = core.runtime.loads[0]
    assert key == "org/Maly-7B" and config.n_ctx == 16384


def test_without_the_facade_every_tool_refuses_politely(tmp_path: Path) -> None:
    bare = ToolContext(cwd=tmp_path, state={})
    for t in helpbot_tools():
        if t.name in ("model_search", "model_list", "model_install"):
            continue                           # the agent's own tools, tested there
        result = asyncio.run(t.run({"key": "x"}, bare))
        assert result.ok is False, t.name


# --------------------------------------------------------------------------- #
# The session
# --------------------------------------------------------------------------- #


def test_the_help_bots_directory_is_what_makes_a_session_its(isolated_home: Path) -> None:
    assert helpbot.is_helpbot_cwd(helpbot.helpbot_dir())
    assert not helpbot.is_helpbot_cwd(isolated_home)


async def test_the_help_bot_speaks_with_its_own_prompt_not_the_coding_one(isolated_home: Path) -> None:
    engine = ScriptedEngine(["Masz trzy modele."])
    core = FakeCore(engine)
    agent = helpbot.make_agent(core, on_event=EventLog())
    await agent.run_turn("jakie mam modele?")
    prompt = engine.prompts[0]
    assert "You are Pomocnik" in prompt
    assert "API keys" in prompt                  # the rule about keys travels with it
    assert "model_unload" in prompt and "read_file" not in prompt


async def test_the_help_bot_survives_clear(isolated_home: Path) -> None:
    core = FakeCore()
    agent = helpbot.make_agent(core, on_event=EventLog())
    agent.clear()
    assert isinstance(agent._tool_ctx.state.get(STATE_KEY), HelpbotOps)


async def test_a_full_turn_finds_the_models_and_answers(isolated_home: Path) -> None:
    engine = ScriptedEngine([tool_call_text("model_list", {}), "Masz trzy modele."])
    core = FakeCore(engine)
    log = EventLog()
    agent = helpbot.make_agent(core, on_event=log)
    answer = await agent.run_turn("jakie mam modele?")
    assert answer == "Masz trzy modele."
    results = [e for e in log.events if getattr(e, "type", "") == "tool_result"]
    assert results and results[0].ok


def test_a_missing_model_is_reported_not_crashed(isolated_home: Path) -> None:
    core = FakeCore()
    core.library.models.pop(HELPBOT_KEY)
    assert helpbot.status(core)["state"] == "missing"
    with pytest.raises(helpbot.HelpbotMissing):
        helpbot.make_agent(core)


async def test_the_download_waits_for_room_on_the_disk(isolated_home: Path, monkeypatch) -> None:
    """A 2.3 GB download on a full disk breaks more than a missing help bot does."""
    import shutil

    core = FakeCore()
    core.library.models.pop(HELPBOT_KEY)
    core.settings.models_dir = str(isolated_home)
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda _p: type("U", (), {"free": 1_000_000_000})())
    started: list[str] = []

    class Catalog:
        def __init__(self, *a: Any, **k: Any) -> None: ...
        def entries(self, **_k: Any) -> list[dict[str, Any]]:
            return [{"id": HELPBOT_KEY, "sizeBytes": 2_263_022_417}]
        async def ainstall(self, name: str) -> dict[str, Any]:
            started.append(name)
            return {}

    monkeypatch.setattr("localcode.hub.catalog.Catalog", Catalog)
    result = await helpbot.ensure_installed(core)
    assert started == []
    assert "miejsca" in result["error"]



# --------------------------------------------------------------------------- #
# The store's card, as the "Szczegóły" sheet shows it
# --------------------------------------------------------------------------- #

FLUX_CARD = {
    "id": "Runpod/FLUX.2-klein-4B-mflux-4bit", "name": "FLUX.2 klein 4B", "format": "mlx",
    "sizeLabel": "ok. 4,6 GB", "paramsLabel": "4 mld", "quant": "4bit",
    "tagline": "Najszybszy sensowny generator obrazów na Apple Silicon.",
    "fit": "Zmieści się", "fitDetail": "Potrzebuje ok. 4,6 GB z 29,0 GB dostępnych dla modeli.",
    "downloads": 92_000, "likes": 11, "installed": True,
    "trust": {"score": 86, "label": "wysokie zaufanie", "license": "Apache 2.0", "gated": False,
              "reasons": ["Znany zespół konwertujący (RunPod)."], "warnings": [],
              "provenance": {"label": "sprawdzona konwersja",
                             "detail": "Konwersja przygotowana przez RunPod."}},
    "variants": [{"label": "4bit", "sizeBytes": 3291, "recommended": True}],
}


def test_the_card_carries_what_the_store_sheet_shows() -> None:
    from localcode.helpbot.tools import model_card

    card = model_card(FLUX_CARD)
    for expected in ("FLUX.2 klein 4B", "ok. 4,6 GB", "z 29,0 GB", "86/100", "wysokie zaufanie",
                     "sprawdzona konwersja", "RunPod", "Apache 2.0", "92 tys.", "Zainstalowany: tak"):
        assert expected in card, expected
    # A variant the hub measured at 3 kB (only the top-level JSON) shows no size
    # rather than a misleading "0,0 GB".
    assert "0,0 GB" not in card


class _Catalog:
    def __init__(self, rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]) -> None:
        self.rows, self.details, self.detail_calls = rows, details, []

    def entries(self, **_: Any) -> list[dict[str, Any]]:
        return list(self.rows)

    def detail(self, entry_id: str) -> dict[str, Any]:
        self.detail_calls.append(entry_id)
        return self.details[entry_id]


async def test_a_name_picks_the_installed_then_the_most_trusted_model(core: FakeCore) -> None:
    """The store's top hit is ranked for browsing; "flux klein" means the one you have."""
    rows = [{"id": "x/flux-text-encoder", "installed": False, "trust": {"score": 55}},
            {"id": "Runpod/FLUX.2-klein-4B-mflux-4bit", "installed": True, "trust": {"score": 86}}]
    catalog = _Catalog(rows, {"Runpod/FLUX.2-klein-4B-mflux-4bit": FLUX_CARD})
    ops = HelpbotOps(core, helpbot_key=HELPBOT_KEY)
    ops._catalog = catalog
    entry = await ops.model_info("flux klein")
    assert entry["id"] == "Runpod/FLUX.2-klein-4B-mflux-4bit"
    assert catalog.detail_calls == ["Runpod/FLUX.2-klein-4B-mflux-4bit"]   # never detail("flux klein")


async def test_an_id_goes_straight_to_the_stores_detail(core: FakeCore) -> None:
    catalog = _Catalog([], {"Runpod/FLUX.2-klein-4B-mflux-4bit": FLUX_CARD})
    ops = HelpbotOps(core, helpbot_key=HELPBOT_KEY)
    ops._catalog = catalog
    entry = await ops.model_info("Runpod/FLUX.2-klein-4B-mflux-4bit")
    assert entry["name"] == "FLUX.2 klein 4B"


async def test_the_help_bot_sets_a_remote_model_as_default_instead_of_loading_it() -> None:
    """"Ustaw mi Kimi K2.5" with a DeepInfra key: there is nothing to load."""
    from localcode.helpbot.ops import HelpbotError, HelpbotOps

    class Registry:
        providers = ["deepinfra"]

        async def models(self):
            from localcode.remote.providers import RemoteModel
            return [RemoteModel("deepinfra", "moonshotai/Kimi-K2.5")]

    class Runtime:
        remote = Registry()

    class Library:
        def get(self, key):
            return None

        def resolve(self, key):
            return None

    ops = HelpbotOps.__new__(HelpbotOps)
    ops.runtime, ops._library, ops.settings = Runtime(), Library(), None
    import pytest as _pytest
    with _pytest.raises(HelpbotError) as excinfo:
        await ops.load("kimi k2.5")
    assert "defaultModel" in str(excinfo.value) and "remote:deepinfra:moonshotai/Kimi-K2.5" in str(excinfo.value)
