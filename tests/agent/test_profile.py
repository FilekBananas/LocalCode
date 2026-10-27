"""Which squeeze a model gets (``localcode.agent.profile``).

Every optimisation in this project — short tool menus, clipped tool output,
masked history, early compaction — exists for a small model with a tight
window. A model reached over an API key has neither problem, and paying for a
compaction that saves room nobody needed is the opposite of thrift.
"""

from __future__ import annotations

import pytest

from localcode.agent.context import limits_for
from localcode.agent.profile import (
    LOCAL,
    STRONG,
    ModelProfile,
    is_remote_model,
    profile_for,
)
from localcode.hub.types import LocalModel, ModelMeta
from localcode.remote.engine import remote_model


def local_model(key: str = "org/Qwen3-4B/q4") -> LocalModel:
    return LocalModel(key=key, format="gguf", path="/models/x", display_name="Qwen3 4B",
                      size_bytes=2_500_000_000, meta=ModelMeta(context_length=8192))


# --------------------------------------------------------------------------- #
# Telling them apart
# --------------------------------------------------------------------------- #


def test_a_model_on_this_disk_is_not_remote() -> None:
    assert is_remote_model(local_model()) is False


def test_a_model_behind_an_api_key_is_remote() -> None:
    assert is_remote_model(remote_model("remote:openai:gpt-5")) is True
    assert is_remote_model(remote_model("remote:anthropic:claude-opus-5")) is True


def test_nothing_is_not_remote() -> None:
    assert is_remote_model(None) is False


# --------------------------------------------------------------------------- #
# What each profile does
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_ctx, tiny", [(4096, True), (8192, True), (16384, False),
                                         (131_072, False)])
def test_a_local_model_keeps_the_short_menu_only_when_the_window_is_tight(
        n_ctx: int, tiny: bool) -> None:
    assert profile_for(local_model(), n_ctx).tiny_tools is tiny


def test_a_remote_model_never_gets_the_short_menu() -> None:
    profile = profile_for(remote_model("remote:openai:gpt-5"), 4096)
    assert profile.tiny_tools is False
    assert profile.is_strong is True


def test_a_remote_model_compacts_later_and_answers_longer() -> None:
    strong = profile_for(remote_model("remote:gemini:gemini-3-pro"), 200_000)
    local = profile_for(local_model(), 8192)
    assert strong.compact_threshold > local.compact_threshold
    assert strong.reserve_output > local.reserve_output


def test_a_remote_model_keeps_its_history_intact() -> None:
    """Masking old tool results buys context a frontier model already has."""
    assert profile_for(remote_model("remote:openai:gpt-5")).mask_old_tool_results is False
    assert profile_for(local_model(), 8192).mask_old_tool_results is True


def test_the_override_wins_over_detection() -> None:
    assert profile_for(local_model(), 8192, remote=True).is_strong is True
    assert profile_for(remote_model("remote:openai:gpt-5"), remote=False).is_strong is False


def test_the_two_profiles_are_the_only_ones() -> None:
    assert LOCAL.name == "local" and STRONG.name == "strong"
    assert isinstance(profile_for(local_model(), 8192), ModelProfile)


# --------------------------------------------------------------------------- #
# Tool limits follow the window
# --------------------------------------------------------------------------- #


def test_tool_output_grows_with_the_window() -> None:
    small = limits_for(4096)
    medium = limits_for(32768)
    huge = limits_for(200_000)
    assert small.max_output_chars < medium.max_output_chars < huge.max_output_chars
    assert small.read_default_lines < medium.read_default_lines < huge.read_default_lines


def test_a_frontier_window_is_not_clipped_to_the_old_ceiling() -> None:
    """The ladder used to stop at 24 000 characters, which for a 200k window
    meant the model kept asking for the rest of the file."""
    assert limits_for(200_000).max_output_chars >= 50_000
    assert limits_for(200_000).read_default_lines >= 1_000
