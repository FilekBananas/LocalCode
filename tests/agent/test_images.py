"""Pictures for vision models: the agent turn, per kind of engine.

The local path was checked against a real llama-server with Qwen3-VL-2B: the
image goes as base64 in ``multimodal_data`` next to a prompt that carries the
server's own media marker (read from ``/props`` — it changes at every start).
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace

import pytest

from agent_fakes import EventLog, FakeRuntime, ScriptedEngine
from localcode.agent.loop import Agent
from localcode.config import Settings

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

PIXEL = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
IMAGE = {"name": "zrzut.png", "mime": "image/png", "data": PIXEL}


def build(engine: ScriptedEngine) -> Agent:
    s = Settings()
    s.agent = replace(s.agent, permission_mode="yolo")
    return Agent(FakeRuntime(engine), "test/model", "/tmp", s, EventLog(), None, tools=[])


async def test_a_local_vision_model_gets_the_marker_and_the_image_beside_the_prompt() -> None:
    engine = ScriptedEngine(["Czerwony kwadrat."], n_ctx=8192)
    engine.supports_images = True
    engine.media_marker = "<__media_QA__>"
    agent = build(engine)
    await agent.run_turn("co jest na obrazku?", images=[IMAGE])
    assert "<__media_QA__>" in engine.prompts[0]
    assert engine.sampling[0].images == [PIXEL]
    assert PIXEL not in engine.prompts[0]


async def test_a_model_that_cannot_see_gets_a_note_and_no_bytes() -> None:
    engine = ScriptedEngine(["Nie widzę obrazu."], n_ctx=8192)
    agent = build(engine)
    await agent.run_turn("co jest na obrazku?", images=[IMAGE])
    assert "Obraz pominięty" in engine.prompts[0]
    assert engine.sampling[0].images == []
    assert PIXEL not in engine.prompts[0]


async def test_a_remote_vision_model_gets_openai_content_parts() -> None:
    engine = ScriptedEngine(["To czerwony kwadrat."], n_ctx=8192)
    engine.format = "remote"
    engine.supports_images = True
    agent = build(engine)
    sent: list[list] = []
    original = Agent._media_for

    def spy(eng, messages):
        out, images = original(eng, messages)
        sent.append(out)
        return out, images

    agent._media_for = spy  # type: ignore[method-assign]
    await agent.run_turn("co jest na obrazku?", images=[IMAGE])
    user = [m for m in sent[0] if m["role"] == "user"][-1]
    assert user["content"][0] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PIXEL}"}}
    assert user["content"][-1]["type"] == "text"


async def test_after_the_turn_the_history_keeps_a_note_not_the_bytes() -> None:
    engine = ScriptedEngine(["Ok."], n_ctx=8192)
    engine.supports_images = True
    engine.media_marker = "<m>"
    agent = build(engine)
    await agent.run_turn("opisz", images=[IMAGE, IMAGE])
    user = [m for m in agent.messages if m["role"] == "user"][-1]
    assert isinstance(user["content"], str)
    assert "2 obrazy pokazane" in user["content"] and PIXEL not in json.dumps(agent.messages)


def test_anthropic_and_gemini_get_images_in_their_own_shape() -> None:
    from localcode.remote.providers import _anthropic_message, _gemini_message

    message = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{PIXEL}"}},
        {"type": "text", "text": "co to?"}]}
    blocks = _anthropic_message(message)["content"]
    assert blocks[0] == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": PIXEL}}
    assert blocks[1] == {"type": "text", "text": "co to?"}
    parts = _gemini_message(message)["parts"]
    assert parts[0] == {"inline_data": {"mime_type": "image/jpeg", "data": PIXEL}}


@pytest.mark.parametrize("provider, model_id, tags, sees", [
    ("deepinfra", "Qwen/Qwen3-VL-235B", ["chat", "vlm"], True),
    ("deepinfra", "deepseek-ai/DeepSeek-V4.1-Flash", ["chat"], False),
    ("openai", "gpt-5", [], True),
    ("openai", "gpt-3.5-turbo", [], False),
    ("anthropic", "claude-opus-5", [], True),
    ("gemini", "gemini-3-pro", [], True),
])
def test_which_remote_models_see_images(provider: str, model_id: str, tags: list[str], sees: bool) -> None:
    from localcode.remote.providers import sees_images

    assert sees_images(provider, model_id, {}, {"tags": tags} if tags else {}) is sees


def test_llama_server_gets_the_prompt_with_its_images() -> None:
    """The shape checked against a real llama-server (Qwen3-VL-2B answered "Red square")."""
    from localcode.runtime.base import SamplingParams
    from localcode.runtime.llama_engine import LlamaServerEngine

    engine = LlamaServerEngine.__new__(LlamaServerEngine)
    engine._props = {"modalities": {"vision": True}, "media_marker": "<__media_X__>"}
    engine._adds_bos, engine._bos_text = False, None
    body = engine._completion_body("<__media_X__>\nco to?", SamplingParams(images=[PIXEL]))
    assert body["prompt"] == {"prompt_string": "<__media_X__>\nco to?", "multimodal_data": [PIXEL]}

    engine._props = {}
    from localcode.runtime.base import EngineError
    with pytest.raises(EngineError):
        engine._completion_body("x", SamplingParams(images=[PIXEL]))


async def test_a_message_with_a_picture_is_counted_without_breaking_the_template() -> None:
    """Regression (real run): the token counter fed list content to the chat
    template ("can only concatenate str (not list)") and fell back to a guess."""
    from localcode.agent.loop import IMAGE_TOKENS, _countable

    flat, pictures = _countable([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PIXEL}"}},
        {"type": "text", "text": "co to?"}]}])
    assert flat == [{"role": "user", "content": "co to?"}] and pictures == 1 and IMAGE_TOKENS >= 1000


async def test_the_model_is_told_the_picture_is_already_visible() -> None:
    engine = ScriptedEngine(["Kwadrat."], n_ctx=8192)
    engine.supports_images = True
    engine.media_marker = "<m>"
    agent = build(engine)
    await agent.run_turn("co to?", images=[IMAGE])
    assert "zrzut.png is shown above" in engine.prompts[0] and "do not open it with tools" in engine.prompts[0]
