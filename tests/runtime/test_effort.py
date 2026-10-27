"""Reasoning-effort detection and plumbing (``localcode.runtime.templates``).

The levels a model can be asked for come from its *real* chat template, so the
expectations below are pinned per family against the fixtures in
``tests/fixtures/templates``:

* gpt-oss (``reasoning_effort``) — ``low|medium|high``; it cannot be switched
  off, so ``"off"`` resolves down to ``"low"``;
* Qwen3 / GLM / DeepSeek V3.1 / Granite (boolean switch) — ``off|on`` only;
* templates without any switch — no levels at all, and every surface must hide
  the control.

The ``real_model`` test at the bottom closes the loop on a real MLX model:
``off`` really suppresses the ``<think>`` block, a higher level does not.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

import template_fixtures as tf
from lc_testing import model_path
from localcode.runtime.base import TemplateInfo
from localcode.runtime.formatter import ChatFormatter
from localcode.runtime.templates import (
    EFFORT_BUDGETS,
    EFFORT_CHOICES,
    effort_kwargs,
    inspect_template,
    model_effort_levels,
    normalize_effort,
    resolve_effort,
)
from test_formatter import FakeEngine, _model

USER = [{"role": "user", "content": "Cześć"}]

#: slug -> (effort_kwarg, levels) as the real template expresses them.
EXPECTED_EFFORT: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "gpt-oss-20b": ("reasoning_effort", ("low", "medium", "high")),
    "glm-4.5-air": ("enable_thinking", ("off", "on")),
    "deepseek-v3.1": ("thinking", ("off", "on")),
    "granite-3.3-8b": ("thinking", ("off", "on")),
    "local-mlx-tiny": ("enable_thinking", ("off", "on")),
    "local-gguf-tiny": ("enable_thinking", ("off", "on")),
    # No switch at all → no control.
    "qwen3-4b-instruct-2507": (None, ()),
    "qwen3-coder-30b": (None, ()),
    "devstral-small-2507": (None, ()),
    "gemma-3-4b-it": (None, ()),
    "llama-3.1-8b-instruct": (None, ()),
    "kimi-k2": (None, ()),
    # Always-thinking models (no kwarg) also offer nothing to choose from.
    "minimax-m2": (None, ()),
    "local-mlx-small": (None, ()),
    "local-gguf-small": (None, ()),
}

SLUGS = [s.slug for s in tf.available()]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_detection_per_template(slug: str) -> None:
    kwarg, levels = EXPECTED_EFFORT[slug]
    info = inspect_template(tf.load(slug))
    assert (info.effort_kwarg, tuple(info.effort_levels)) == (kwarg, levels)


def test_every_fixture_has_an_expectation() -> None:
    assert set(SLUGS) <= set(EXPECTED_EFFORT)


def test_no_template_means_no_control() -> None:
    for empty in (None, "", "   "):
        assert inspect_template(empty).effort_levels == []


def test_template_info_defaults_are_additive() -> None:
    info = TemplateInfo()
    assert info.effort_levels == [] and info.effort_kwarg is None


def test_inspect_returns_independent_level_lists() -> None:
    a = inspect_template(tf.load("gpt-oss-20b"))
    a.effort_levels.append("MUTATED")
    assert "MUTATED" not in inspect_template(tf.load("gpt-oss-20b")).effort_levels


def test_model_effort_levels_from_metadata() -> None:
    assert model_effort_levels(_model(template=tf.load("gpt-oss-20b"))) == ["low", "medium", "high"]
    assert model_effort_levels(_model(template=tf.load("llama-3.1-8b-instruct"))) == []
    assert model_effort_levels(_model(template=None)) == []
    assert model_effort_levels(None) == []


# --------------------------------------------------------------------------- #
# Resolution and kwargs
# --------------------------------------------------------------------------- #


def test_resolve_onto_a_boolean_switch() -> None:
    levels = ["off", "on"]
    assert resolve_effort("off", levels) == "off"
    # anything but "off" still means "think" — a boolean model cannot think less
    assert resolve_effort("low", levels) == "on"
    assert resolve_effort("medium", levels) == "on"
    assert resolve_effort("high", levels) == "on"


def test_resolve_onto_gpt_oss() -> None:
    levels = ["low", "medium", "high"]
    assert resolve_effort("medium", levels) == "medium"
    assert resolve_effort("high", levels) == "high"
    # gpt-oss cannot stop reasoning: "off" becomes the weakest level it has
    assert resolve_effort("off", levels) == "low"


def test_resolve_without_control_or_request() -> None:
    assert resolve_effort("high", []) is None
    assert resolve_effort(None, ["off", "on"]) is None
    assert resolve_effort("bzdura", ["off", "on"]) is None


def test_kwargs_per_family() -> None:
    assert effort_kwargs("reasoning_effort", "high") == {"reasoning_effort": "high"}
    assert effort_kwargs("reasoning_effort", "low") == {"reasoning_effort": "low"}
    assert effort_kwargs("enable_thinking", "off") == {"enable_thinking": False}
    assert effort_kwargs("enable_thinking", "on") == {"enable_thinking": True}
    assert effort_kwargs("thinking", "off") == {"thinking": False}
    assert effort_kwargs("thinking_budget", "off") == {"thinking_budget": EFFORT_BUDGETS["off"]}
    assert effort_kwargs("thinking_budget", "high") == {"thinking_budget": EFFORT_BUDGETS["high"]}
    assert effort_kwargs(None, "high") == {} and effort_kwargs("enable_thinking", None) == {}


def test_normalize_effort() -> None:
    assert [normalize_effort(v) for v in EFFORT_CHOICES] == list(EFFORT_CHOICES)
    assert normalize_effort(" HIGH ") == "high"
    assert normalize_effort("on") == "medium"
    assert normalize_effort("nonsens") is None
    assert normalize_effort(None, "medium") == "medium"
    assert normalize_effort(7) is None


# --------------------------------------------------------------------------- #
# ChatFormatter
# --------------------------------------------------------------------------- #


def _kwargs_for(slug: str, level: str | None) -> dict:
    formatter = ChatFormatter(FakeEngine(tf.load(slug)))
    _, _, kwargs = formatter.prepare(USER, None, effort=level)
    return kwargs


def test_formatter_kwargs_gpt_oss() -> None:
    assert _kwargs_for("gpt-oss-20b", "high") == {"reasoning_effort": "high"}
    assert _kwargs_for("gpt-oss-20b", "medium") == {"reasoning_effort": "medium"}
    assert _kwargs_for("gpt-oss-20b", "off") == {"reasoning_effort": "low"}


@pytest.mark.parametrize("slug", ["local-gguf-tiny", "glm-4.5-air"])
def test_formatter_kwargs_boolean_switch(slug: str) -> None:
    kwarg = EXPECTED_EFFORT[slug][0]
    assert _kwargs_for(slug, "off") == {kwarg: False}
    assert _kwargs_for(slug, "medium") == {kwarg: True}


def test_formatter_kwargs_deepseek() -> None:
    assert _kwargs_for("deepseek-v3.1", "off") == {"thinking": False}
    assert _kwargs_for("deepseek-v3.1", "high") == {"thinking": True}


def test_formatter_without_control_passes_nothing() -> None:
    assert _kwargs_for("llama-3.1-8b-instruct", "high") == {}
    assert ChatFormatter(FakeEngine(tf.load("llama-3.1-8b-instruct"))).effort_levels() == []


def test_formatter_effort_wins_over_thinking_flag() -> None:
    formatter = ChatFormatter(FakeEngine(tf.load("local-gguf-tiny")))
    _, _, kwargs = formatter.prepare(USER, None, thinking=False, effort="high")
    assert kwargs == {"enable_thinking": True}
    # ...and without an effort the legacy flag still decides
    _, _, legacy = formatter.prepare(USER, None, thinking=False)
    assert legacy == {"enable_thinking": False}


def test_formatter_exposes_levels_and_resolution() -> None:
    formatter = ChatFormatter(FakeEngine(tf.load("gpt-oss-20b")))
    assert formatter.effort_levels() == ["low", "medium", "high"]
    assert formatter.resolve_effort("off") == "low"


async def test_render_reaches_the_prompt_and_is_cached_per_level() -> None:
    engine = FakeEngine(tf.load("gpt-oss-20b"))
    formatter = ChatFormatter(engine)
    high = await formatter.render(USER, None, effort="high")
    low = await formatter.render(USER, None, effort="low")
    assert "Reasoning: high" in high and "Reasoning: low" in low
    calls = len(engine.applied)
    assert await formatter.render(USER, None, effort="high") == high
    assert len(engine.applied) == calls  # cache key includes the level


async def test_qwen_off_writes_an_empty_think_block() -> None:
    formatter = ChatFormatter(FakeEngine(tf.load("local-gguf-tiny")))
    off = await formatter.render(USER, None, effort="off")
    on = await formatter.render(USER, None, effort="medium")
    assert off != on
    assert off.rstrip().endswith("</think>")


# --------------------------------------------------------------------------- #
# Real model
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lc_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("lc-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALCODE_HOME", str(home))
        yield home


@pytest.mark.real_model
async def test_off_suppresses_thinking_on_a_real_mlx_model(lc_home) -> None:
    """Qwen3-0.6B (MLX): ``off`` really removes the ``<think>`` reasoning."""
    pytest.importorskip("mlx_lm")
    directory = model_path("mlx_tiny")

    from localcode.config import Settings
    from localcode.hub.types import LocalModel, ModelMeta
    from localcode.runtime.base import LoadConfig, SamplingParams
    from localcode.runtime.mlx_engine import MlxEngine

    meta = ModelMeta(arch="qwen3", n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128,
                     context_length=40960)
    model = LocalModel(key="mlx-community/Qwen3-0.6B-4bit", format="mlx", path=str(directory),
                       display_name="Qwen3 0.6B", size_bytes=0, quant="4bit", meta=meta)
    engine = MlxEngine("effort-1", model, LoadConfig(n_ctx=2048), settings=Settings())
    await engine.start()
    try:
        assert engine.template_info.effort_levels == ["off", "on"]
        formatter = ChatFormatter(engine)
        messages = [{"role": "user", "content": "Ile to jest 2+2? Odpowiedz krotko."}]
        params = SamplingParams(max_tokens=96, temperature=0.0, stop=formatter.stop_strings())

        async def answer(level: str) -> str:
            prompt = await formatter.render(messages, None, effort=level)
            text = ""
            async for chunk in engine.complete(prompt, params):
                text += chunk.text or ""
            # What the template already closed does not count as the model thinking.
            return (prompt.split("</think>")[-1]) + text

        assert "<think>" not in await answer("off")
        assert "<think>" in await answer("high")
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(engine.stop(), timeout=30)
