"""The tools that let the assistant manage models (``agent/tools/models.py``).

The point of these four tools is that "uruchom mi Qwen3 Coder" can be answered
by a small always-resident model without anyone leaving the conversation. The
risky half of that — downloading gigabytes, taking memory — must stay behind a
permission prompt and behind a setting, and a session that was never given the
facade must refuse rather than improvise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from localcode.agent.tools import get_tool, tools_for_mode
from localcode.agent.tools.base import ToolContext, ToolLimits
from localcode.agent.tools.models import (
    STATE_KEY,
    ModelInstallTool,
    ModelListTool,
    ModelLoadTool,
    ModelOps,
    ModelSearchTool,
)


class FakeOps(ModelOps):
    def __init__(self, *, fail: str | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail = fail

    async def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        self.calls.append(("search", query))
        return [{
            "id": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
            "name": "Qwen3 Coder 30B",
            "sizeLabel": "ok. 18,5 GB",
            "fit": "Na styk",
            "tagline": "Agent kodujący na 32 GB.",
            "trust": {"score": 87},
            "installed": False,
        }][:limit]

    async def installed(self) -> list[dict[str, Any]]:
        self.calls.append(("installed", None))
        return [{"key": "org/Model/q4", "displayName": "Model 8B", "format": "gguf",
                 "sizeBytes": 4_700_000_000}]

    async def install(self, name_or_repo: str, quant: str | None = None) -> dict[str, Any]:
        self.calls.append(("install", name_or_repo) if quant is None
                          else ("install", name_or_repo, quant))
        if self.fail == "install":
            raise RuntimeError("brak miejsca na dysku")
        return {"id": "job-1", "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                "label": "Qwen3 Coder 30B", "state": "queued"}

    async def load(self, key: str) -> dict[str, Any]:
        self.calls.append(("load", key))
        if self.fail == "load":
            raise RuntimeError("model nie mieści się w pamięci")
        return {"key": key, "displayName": "Model 8B"}

    async def status(self) -> dict[str, Any]:
        self.calls.append(("status", None))
        return {"instances": [{"modelKey": "org/Model/q4"}],
                "budget": {"limitBytes": 22_000_000_000, "usedBytes": 4_700_000_000}}


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    context = ToolContext(cwd=tmp_path, limits=ToolLimits(), state={})
    context.state[STATE_KEY] = FakeOps()
    return context


def ops(ctx: ToolContext) -> FakeOps:
    return ctx.state[STATE_KEY]


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #


def test_the_tools_are_absent_until_they_are_switched_on() -> None:
    names = [t.name for t in tools_for_mode("default")]
    assert "model_install" not in names
    assert "model_search" not in names


def test_switching_them_on_adds_exactly_four() -> None:
    names = [t.name for t in tools_for_mode("default", manage_models=True)]
    assert names[-4:] == ["model_search", "model_list", "model_install", "model_load"]


def test_plan_mode_keeps_only_the_harmless_two() -> None:
    """Plan mode may look at models; it may not download or load them."""
    names = [t.name for t in tools_for_mode("plan", manage_models=True)]
    assert "model_search" in names and "model_list" in names
    assert "model_install" not in names and "model_load" not in names


@pytest.mark.parametrize("alias, real", [
    ("install_model", "model_install"),
    ("download_model", "model_install"),
    ("switch_model", "model_load"),
    ("use_model", "model_load"),
    ("list_models", "model_list"),
    ("find_model", "model_search"),
])
def test_the_names_models_reach_for_are_understood(alias: str, real: str) -> None:
    tool = get_tool(alias)
    assert tool is not None and tool.name == real


# --------------------------------------------------------------------------- #
# Permission
# --------------------------------------------------------------------------- #


async def test_looking_needs_no_permission(ctx: ToolContext) -> None:
    assert await ModelSearchTool().permission({"query": "qwen"}, ctx) is None
    assert await ModelListTool().permission({}, ctx) is None


async def test_downloading_and_loading_ask_first(ctx: ToolContext) -> None:
    install = await ModelInstallTool().permission({"name": "Qwen3 Coder 30B"}, ctx)
    assert install is not None and "Qwen3 Coder 30B" in install.title

    load = await ModelLoadTool().permission({"key": "org/Model/q4"}, ctx)
    assert load is not None and "pamięć" in load.preview.lower()


# --------------------------------------------------------------------------- #
# Running them
# --------------------------------------------------------------------------- #


async def test_search_reports_what_the_person_would_want_to_know(ctx: ToolContext) -> None:
    result = await ModelSearchTool().run({"query": "qwen coder"}, ctx)
    assert result.ok
    assert "Qwen3 Coder 30B" in result.output
    assert "18,5 GB" in result.output
    assert "87/100" in result.output                 # trust travels to the model too
    assert "Na styk" in result.output                # ...and so does the memory verdict
    assert ops(ctx).calls == [("search", "qwen coder")]


async def test_search_without_a_query_says_so(ctx: ToolContext) -> None:
    result = await ModelSearchTool().run({"query": "  "}, ctx)
    assert not result.ok
    assert ops(ctx).calls == []


async def test_list_shows_what_is_on_disk_and_what_is_loaded(ctx: ToolContext) -> None:
    result = await ModelListTool().run({}, ctx)
    assert result.ok
    assert "Model 8B" in result.output
    assert "załadowany" in result.output
    assert "Pamięć dla modeli" in result.output


async def test_install_starts_a_download_and_says_where_to_watch(ctx: ToolContext) -> None:
    result = await ModelInstallTool().run({"name": "Qwen3 Coder 30B"}, ctx)
    assert result.ok
    assert "Pobieram" in result.output
    assert ops(ctx).calls == [("install", "Qwen3 Coder 30B")]


async def test_a_failed_install_is_reported_not_raised(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, limits=ToolLimits(), state={STATE_KEY: FakeOps(fail="install")})
    result = await ModelInstallTool().run({"name": "Cokolwiek"}, ctx)
    assert not result.ok
    assert "brak miejsca" in result.output


async def test_load_switches_the_conversation(ctx: ToolContext) -> None:
    result = await ModelLoadTool().run({"key": "org/Model/q4"}, ctx)
    assert result.ok
    assert "odpowiada" in result.output
    assert ops(ctx).calls == [("load", "org/Model/q4")]


async def test_a_failed_load_is_reported_not_raised(tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, limits=ToolLimits(), state={STATE_KEY: FakeOps(fail="load")})
    result = await ModelLoadTool().run({"key": "org/Model/q4"}, ctx)
    assert not result.ok
    assert "nie mieści się" in result.output


# --------------------------------------------------------------------------- #
# Without the facade
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tool, args", [
    (ModelSearchTool(), {"query": "qwen"}),
    (ModelListTool(), {}),
    (ModelInstallTool(), {"name": "qwen"}),
    (ModelLoadTool(), {"key": "org/Model/q4"}),
])
async def test_a_session_without_model_access_refuses_politely(tool, args, tmp_path: Path) -> None:
    ctx = ToolContext(cwd=tmp_path, limits=ToolLimits(), state={})
    result = await tool.run(args, ctx)
    assert not result.ok
    assert "wyłączone" in result.output


async def test_install_passes_the_version_the_user_asked_for(ctx: ToolContext) -> None:
    result = await ModelInstallTool().run({"name": "Qwen3 Coder 30B", "quant": "Q8_0"}, ctx)
    assert result.ok
    assert ops(ctx).calls[-1] == ("install", "Qwen3 Coder 30B", "Q8_0")
    asked = await ModelInstallTool().permission({"name": "Qwen3 Coder 30B", "quant": "Q8_0"}, ctx)
    assert "Q8_0" in asked.title


VARIANTS = [
    {"id": "a", "label": "Q4_K_M", "quant": "Q4_K_M"},
    {"id": "b", "label": "Q6_K", "quant": "Q6_K"},
    {"id": "c", "label": "Q8_0", "quant": "Q8_0"},
    {"id": "d", "label": "BF16", "quant": "BF16"},
]
MLX_VARIANTS = [{"id": "m4", "label": "4bit", "quant": "4bit"},
                {"id": "m8", "label": "8bit", "quant": "8bit"}]


@pytest.mark.parametrize("asked, picked", [
    ("Q8_0", "c"), ("q8_0", "c"), ("8bit", "c"), ("8-bit", "c"), ("Q6", "b"),
    ("q4_k_m", "a"), ("16", "d"), ("bf16", "d"), ("Q3_K_M", None),
])
def test_a_named_version_is_found_among_the_variants(asked: str, picked: str | None) -> None:
    from localcode.agent.model_ops import pick_variant

    found = pick_variant(VARIANTS, asked)
    assert (found or {}).get("id") == picked


def test_bit_counts_also_match_mlx_versions() -> None:
    from localcode.agent.model_ops import pick_variant

    assert pick_variant(MLX_VARIANTS, "8 bit")["id"] == "m8"
    assert pick_variant(MLX_VARIANTS, "4bit")["id"] == "m4"



# --------------------------------------------------------------------------- #
# Models behind an API key
# --------------------------------------------------------------------------- #

REMOTE_ROWS = [
    {"key": "remote:deepinfra:deepseek-ai/DeepSeek-V4-Pro", "displayName": "deepseek-ai/DeepSeek-V4-Pro (DeepInfra)",
     "provider": "DeepInfra", "remote": True},
    {"key": "remote:deepinfra:moonshotai/Kimi-K2.5", "displayName": "moonshotai/Kimi-K2.5 (DeepInfra)",
     "provider": "DeepInfra", "remote": True},
] + [{"key": f"remote:deepinfra:org/m{i}", "displayName": f"org/m{i} (DeepInfra)", "provider": "DeepInfra",
      "remote": True} for i in range(40)]


class RemoteOps(FakeOps):
    async def remote(self) -> list[dict[str, Any]]:
        return list(REMOTE_ROWS)

    async def resolve(self, text: str) -> dict[str, Any] | None:
        from localcode.agent.tools.models import matches
        hits = [m for m in REMOTE_ROWS if m["key"] == text or matches(m, text)]
        return hits[0] if hits else None

    async def load(self, key: str) -> dict[str, Any]:
        target = await self.resolve(key)
        self.calls.append(("load", key))
        return dict(target) if target else {"key": key, "displayName": "Model 8B"}


@pytest.fixture
def remote_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, limits=ToolLimits(), state={STATE_KEY: RemoteOps()})


async def test_the_list_shows_remote_models_with_their_provider(remote_ctx: ToolContext) -> None:
    """Regression: "ustaw mi DeepSeek V4 Pro" only ever saw the 24 models on disk."""
    result = await ModelListTool().run({}, remote_ctx)
    assert "Modele zdalne" in result.output
    assert "Kimi-K2.5 (DeepInfra)" in result.output
    assert "więcej" in result.output                  # 42 remote, 25 shown without a query
    assert "42 zdalnych" in result.summary


async def test_a_query_narrows_the_list(remote_ctx: ToolContext) -> None:
    result = await ModelListTool().run({"query": "deepseek v4"}, remote_ctx)
    assert "DeepSeek-V4-Pro" in result.output and "Kimi" not in result.output


async def test_switching_to_a_remote_model_always_asks(remote_ctx: ToolContext) -> None:
    from localcode.agent.permissions import REMOTE_FLAG, PermissionPolicy

    tool = ModelLoadTool()
    info = await tool.permission({"key": "deepseek v4 pro"}, remote_ctx)
    assert REMOTE_FLAG in info.title and "DeepInfra" in info.preview and "płatne" in info.preview
    for mode in ("default", "accept_edits", "yolo"):
        policy = PermissionPolicy(mode, cwd=str(remote_ctx.cwd))
        policy.remember_always(tool, {"key": "deepseek v4 pro"})
        assert policy.decide(tool, {"key": "deepseek v4 pro"}, info) == "ask"


async def test_switching_says_where_the_model_runs(remote_ctx: ToolContext) -> None:
    result = await ModelLoadTool().run({"key": "kimi k2.5"}, remote_ctx)
    assert result.ok and "zdalnym" in result.output and "płatne" in result.output


class FakeRegistry:
    providers = ["deepinfra"]

    async def models(self):
        from localcode.remote.providers import RemoteModel
        return [RemoteModel("deepinfra", "deepseek-ai/DeepSeek-V4-Pro"),
                RemoteModel("deepinfra", "deepseek-ai/DeepSeek-V4-Pro-Distill-8B"),
                RemoteModel("deepinfra", "moonshotai/Kimi-K2.5")]


class FakeRuntime:
    remote = FakeRegistry()

    async def load(self, key: str):
        raise AssertionError("a remote model must not be loaded")


class EmptyLibrary:
    def get(self, key):
        return None

    def resolve(self, key):
        return None


async def test_local_ops_switch_the_conversation_to_a_remote_model() -> None:
    from localcode.agent.model_ops import LocalModelOps

    switched: list[str] = []
    ops = LocalModelOps(None, runtime=FakeRuntime(), library=EmptyLibrary(), on_switch=switched.append)
    remote = await ops.remote()
    assert [m["displayName"] for m in remote][0] == "deepseek-ai/DeepSeek-V4-Pro (DeepInfra)"
    found = await ops.resolve("deep seek v4 pro")      # as the user typed it
    assert found["key"] == "remote:deepinfra:deepseek-ai/DeepSeek-V4-Pro"
    found = await ops.resolve("deepseek v4 pro")
    assert found["key"] == "remote:deepinfra:deepseek-ai/DeepSeek-V4-Pro"   # the shortest match
    info = await ops.load("deepseek v4 pro")
    assert info["remote"] and switched == ["remote:deepinfra:deepseek-ai/DeepSeek-V4-Pro"]


async def test_a_version_that_is_not_offered_shows_its_family_and_asks_first(remote_ctx: ToolContext) -> None:
    """Regression (real Qwen3-4B): "kimi k2.5" hid Kimi K3 and the model quietly
    switched to K2.6 without asking."""
    ops = remote_ctx.state[STATE_KEY]
    ops_rows = [{"key": "remote:deepinfra:moonshotai/Kimi-K2.6", "displayName": "moonshotai/Kimi-K2.6 (DeepInfra)",
                 "provider": "DeepInfra", "remote": True},
                {"key": "remote:deepinfra:moonshotai/Kimi-K3", "displayName": "moonshotai/Kimi-K3 (DeepInfra)",
                 "provider": "DeepInfra", "remote": True}]

    async def remote() -> list[dict[str, Any]]:
        return ops_rows

    ops.remote = remote  # type: ignore[method-assign]
    result = await ModelListTool().run({"query": "kimi k2.5"}, remote_ctx)
    assert "Podobne" in result.output and "Kimi-K3" in result.output and "Kimi-K2.6" in result.output
    assert "zapytaj" in result.output
    assert "nic nie zostało przełączone" in result.output


async def test_an_answer_that_arrives_before_the_wait_is_not_lost(tmp_path: Path) -> None:
    from localcode.tui.backends import LocalBackend

    backend = LocalBackend.__new__(LocalBackend)
    backend._pending, backend._early_answers = {}, {}

    class Request:
        request_id = "r1"

    await backend.answer_permission("r1", "allow")          # the fast client answers first
    assert await backend._ask_permission(Request()) == "allow"


@pytest.mark.parametrize("query, expected", [
    ("deep_infra_k2_6_key", "remote:deepinfra:moonshotai/Kimi-K2.6"),       # what a 0.6B model passed
    ("kimi z deep infry k2.6", "remote:deepinfra:moonshotai/Kimi-K2.6"),
    ("deep seek v4.1 flash", "remote:deepinfra:deepseek-ai/DeepSeek-V4.1-Flash"),
])
def test_a_name_with_provider_words_still_finds_the_model(query: str, expected: str) -> None:
    from localcode.agent.tools.models import matches

    rows = [{"key": "remote:deepinfra:moonshotai/Kimi-K2.6", "displayName": "moonshotai/Kimi-K2.6 (DeepInfra)"},
            {"key": "remote:deepinfra:ByteDance/Seed-1.8", "displayName": "ByteDance/Seed-1.8 (DeepInfra)"},
            {"key": "remote:deepinfra:deepseek-ai/DeepSeek-V4.1-Flash",
             "displayName": "deepseek-ai/DeepSeek-V4.1-Flash (DeepInfra)"},
            {"key": "remote:deepinfra:moonshotai/Kimi-K2-Instruct-0905",
             "displayName": "moonshotai/Kimi-K2-Instruct-0905 (DeepInfra)"}]
    assert [r["key"] for r in rows if matches(r, query)] == [expected]


async def test_the_nearest_models_share_words_with_the_request_not_the_provider() -> None:
    """Regression: "deep_infra_k2_6_key" suggested ByteDance and Gryphe — every
    model matched "deep" through "(DeepInfra)" in its name."""
    from localcode.agent.model_ops import LocalModelOps

    class Registry:
        providers = ["deepinfra"]

        async def models(self):
            from localcode.remote.providers import RemoteModel
            return [RemoteModel("deepinfra", "ByteDance/Seed-1.8"), RemoteModel("deepinfra", "Gryphe/MythoMax-L2-13b"),
                    RemoteModel("deepinfra", "moonshotai/Kimi-K2.6"), RemoteModel("deepinfra", "moonshotai/Kimi-K3")]

    class Runtime:
        remote = Registry()

    ops = LocalModelOps(None, runtime=Runtime(), library=EmptyLibrary())
    note = await ops._nearest("kimi k2.5")
    assert "Kimi-K2.6" in note and "Kimi-K3" in note
    assert "ByteDance" not in note and "Gryphe" not in note
