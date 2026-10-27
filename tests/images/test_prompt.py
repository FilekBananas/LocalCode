"""Prompt preparation for image models (``localcode/images/prompt.py``).

The behaviour these tests pin down was found by generating real pictures: Stable
Diffusion 1.5 rendered "kot w kapeluszu siedzi na dachu, akwarela" as a plain
photo-like cat (no hat, no watercolour) and the English version of the same
prompt exactly as asked, so a non-English prompt is translated first.
"""

from __future__ import annotations

from typing import Any

import pytest

from localcode.images.prompt import _clean, looks_english, prepare_prompt, translate_prompt


# --------------------------------------------------------------------------- #
# Language heuristic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [
    "a cat wearing a hat sitting on a roof, watercolor",
    "cyberpunk city at night, neon reflections",
    "portrait of an old fisherman, oil painting",
    "",
    "1024x1024",
])
def test_english_prompts_are_left_alone(text: str) -> None:
    assert looks_english(text) is True


@pytest.mark.parametrize("text", [
    "kot w kapeluszu siedzi na dachu, akwarela",
    "zamek nad jeziorem o świcie",
    "ein Hund im Garten",
    "un chien dans le jardin",
    "кот в шляпе на крыше",
])
def test_other_languages_are_detected(text: str) -> None:
    assert looks_english(text) is False


# --------------------------------------------------------------------------- #
# Cleaning a model answer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw, expected", [
    ("a cat wearing a hat", "a cat wearing a hat"),
    ('"a cat wearing a hat"', "a cat wearing a hat"),
    ("Translation: a cat wearing a hat", "a cat wearing a hat"),
    ("<think>hmm, kapelusz = hat</think>a cat wearing a hat", "a cat wearing a hat"),
    ("a cat wearing a hat\n\nI kept the watercolour style.", "a cat wearing a hat"),
])
def test_clean_keeps_only_the_prompt(raw: str, expected: str) -> None:
    assert _clean(raw, "oryginał") == expected


@pytest.mark.parametrize("raw", ["", "   ", "<think>only reasoning</think>"])
def test_clean_falls_back_to_the_original(raw: str) -> None:
    assert _clean(raw, "oryginał") == "oryginał"


# --------------------------------------------------------------------------- #
# prepare_prompt with a fake runtime
# --------------------------------------------------------------------------- #


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _Engine:
    """Engine that returns a fixed translation (or raises)."""

    def __init__(self, answer: str = "a cat wearing a hat, watercolor", fail: bool = False) -> None:
        self.answer, self.fail = answer, fail
        self.template_info = type("TI", (), {"supports_tools": False, "supports_tool_role": False,
                                             "supports_system": True, "supports_thinking": False,
                                             "thinking_kwarg": None, "tool_format": None,
                                             "stop_strings": [], "effort_levels": [],
                                             "effort_kwarg": None})()
        self.model = type("M", (), {"key": "fake", "meta": type("Meta", (), {"params_b": 4.0})()})()
        self.n_ctx = 4096

    async def apply_template(self, messages: list[dict[str, Any]], tools: Any = None, **kw: Any) -> str:
        return "\n".join(str(m.get("content")) for m in messages)

    async def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    async def complete(self, prompt: str, params: Any) -> Any:
        if self.fail:
            raise RuntimeError("engine busy")
        yield _Chunk(self.answer)


class _Lease:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine

    async def __aenter__(self) -> "_Lease":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def record_stats(self, stats: Any) -> None:
        return None


class _Manager:
    def __init__(self, engine: _Engine | None = None, instances: list[dict[str, Any]] | None = None) -> None:
        self.engine = engine or _Engine()
        self._instances = instances if instances is not None else [
            {"modelKey": "loaded/model", "state": "ready"}]
        self.acquired: list[str] = []

    def status(self) -> dict[str, Any]:
        return {"instances": self._instances}

    def acquire(self, key: str, **kw: Any) -> _Lease:
        self.acquired.append(key)
        return _Lease(self.engine)


async def test_polish_prompt_is_translated() -> None:
    manager = _Manager()
    text, translated = await prepare_prompt("kot w kapeluszu siedzi na dachu, akwarela",
                                            manager=manager)
    assert translated is True
    assert text == "a cat wearing a hat, watercolor"
    assert manager.acquired == ["loaded/model"]      # reuses what is already loaded


async def test_english_prompt_is_not_touched() -> None:
    manager = _Manager()
    text, translated = await prepare_prompt("a cat wearing a hat", manager=manager)
    assert (text, translated) == ("a cat wearing a hat", False)
    assert manager.acquired == []                    # no model call at all


async def test_translation_can_be_forced_and_disabled() -> None:
    manager = _Manager()
    text, translated = await prepare_prompt("a cat", manager=manager, translate=True)
    assert translated is True and text == "a cat wearing a hat, watercolor"

    manager = _Manager()
    text, translated = await prepare_prompt("kot w kapeluszu", manager=manager, translate=False)
    assert (text, translated) == ("kot w kapeluszu", False)
    assert manager.acquired == []


async def test_a_failing_model_keeps_the_original_prompt() -> None:
    manager = _Manager(engine=_Engine(fail=True))
    text, translated = await prepare_prompt("kot w kapeluszu", manager=manager)
    assert (text, translated) == ("kot w kapeluszu", False)


async def test_without_a_runtime_nothing_happens() -> None:
    assert await prepare_prompt("kot w kapeluszu", manager=None) == ("kot w kapeluszu", False)
    assert await translate_prompt("kot", manager=None) == "kot"


# --------------------------------------------------------------------------- #
# The translated prompt must survive normalisation (it was dropped twice)
# --------------------------------------------------------------------------- #


def test_normalisation_keeps_the_translation_fields() -> None:
    from localcode.images.base import ImageParams

    params = ImageParams(prompt="a castle above a lake", translate=True,
                         original_prompt="zamek nad jeziorem", width=512, height=512)
    normalised = params.normalized()
    assert normalised.translate is True
    assert normalised.original_prompt == "zamek nad jeziorem"


def test_runtime_preparation_keeps_the_translation_fields() -> None:
    """``ImageRuntime._prepare_params`` rebuilds ImageParams field by field."""
    import inspect

    from localcode.images import runtime as runtime_module

    source = inspect.getsource(runtime_module.ImageRuntime._prepare_params)
    assert "translate=params.translate" in source
    assert "original_prompt=params.original_prompt" in source
