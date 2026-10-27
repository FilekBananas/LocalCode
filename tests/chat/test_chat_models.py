"""Which local models the chatbot is allowed to pick."""

from __future__ import annotations

from localcode.chat.service import is_chat_model
from localcode.hub.types import LocalModel, ModelMeta


def model(key: str, caps: list[str]) -> LocalModel:
    return LocalModel(key=key, format="gguf", path=f"/models/{key}", display_name=key,
                      size_bytes=1, capabilities=caps, meta=ModelMeta(arch="x"))


def test_a_language_model_is_a_chat_model() -> None:
    assert is_chat_model(model("acme/Qwen3-4B", ["tools"]))
    assert is_chat_model(model("acme/Gemma-vision", ["vision"]))


def test_an_embedding_model_is_not_a_chat_model() -> None:
    assert not is_chat_model(model("nomic-ai/nomic-embed", ["embedding"]))


def test_an_image_model_is_not_a_chat_model() -> None:
    """Regression: `GET /api/settings` reported an image model as the chat default.

    With only a diffusion model sorting first in the library, the chatbot's
    fallback ("first chat-capable local model") picked `FLUX.2 klein` and every
    conversation started against a model that cannot produce text at all.
    """
    assert not is_chat_model(model("Runpod/FLUX.2-klein-4B-mflux-4bit", ["image"]))
