"""Models behind an API key in the chatbot's picker and turns.

Regression: the remote providers were wired into "Moje modele" and the agent,
but the chat service listed only the local library — so a model connected in
Ustawienia → Modele zdalne could not be chosen in a conversation at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from chat_fakes import CHAT_KEY, FakeEngine, FakeLibrary, FakeRuntime, collect, make_model
from localcode.chat.service import ChatService, model_info
from localcode.config import RemoteProvider, RemoteSettings, Settings
from localcode.remote import RemoteRegistry
from localcode.remote.engine import remote_model
from localcode.remote.providers import RemoteClient

REMOTE_KEY = "remote:deepinfra:meta-llama/Llama-3.3-70B-Instruct"


class FakeContent:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __aiter__(self) -> "FakeContent":
        self._iter = iter(self._lines)
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._iter).encode("utf-8")
        except StopIteration:
            raise StopAsyncIteration from None


class FakeResponse:
    def __init__(self, *, status: int = 200, body: Any = None, lines: list[str] | None = None) -> None:
        self.status = status
        self._body = body
        self.content = FakeContent(lines or [])

    async def text(self) -> str:
        return self._body if isinstance(self._body, str) else json.dumps(self._body or {})

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeSession:
    def __init__(self, models: list[str], reply: list[str], *, list_status: int = 200) -> None:
        self.models = models
        self.reply = reply
        self.list_status = list_status
        self.posts: list[dict[str, Any]] = []

    def get(self, url: str, headers: dict[str, str] | None = None, **kw: Any) -> FakeResponse:
        if self.list_status != 200:
            return FakeResponse(status=self.list_status, body="down")
        return FakeResponse(body={"data": [{"id": m} for m in self.models]})

    def post(self, url: str, headers: dict[str, str] | None = None, **kw: Any) -> FakeResponse:
        self.posts.append(kw.get("json") or {})
        events = [{"choices": [{"delta": {"content": piece}}]} for piece in self.reply]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 12, "completion_tokens": 3}})
        return FakeResponse(lines=[f"data: {json.dumps(e)}\n" for e in events] + ["data: [DONE]\n"])

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class RemoteAwareRuntime(FakeRuntime):
    """The fake runtime plus the real remote registry, like ``RuntimeManager``."""

    def __init__(self, engines: dict[str, FakeEngine], remote: RemoteRegistry) -> None:
        super().__init__(engines)
        self.remote = remote

    async def acquire(self, model_key: str, *, n_ctx: int | None = None, purpose: str = "chat") -> Any:
        if model_key.startswith("remote:"):
            self.acquired.append((model_key, purpose))
            return await self.remote.lease(model_key, n_ctx=n_ctx, purpose=purpose)
        return await super().acquire(model_key, n_ctx=n_ctx, purpose=purpose)


def make(repo: Any, session: FakeSession, *, key: str | None = "sk-test",
         favorites: list[str] | None = None) -> tuple[ChatService, RemoteAwareRuntime]:
    remote = RemoteSettings(deepinfra=RemoteProvider(api_key=key))
    settings = Settings(remote=remote, favorite_models=list(favorites or []))
    registry = RemoteRegistry(settings, client_factory=lambda p, k, b: RemoteClient(
        p, k, session_factory=lambda: session))
    runtime = RemoteAwareRuntime({CHAT_KEY: FakeEngine()}, registry)
    library = FakeLibrary([make_model(CHAT_KEY, name="Qwen3 0.6B")])
    service = ChatService(repo, runtime, library, lambda: settings, status_delay=0.05, title_timeout=2.0)
    return service, runtime


def test_a_remote_model_is_named_with_its_provider_in_brackets() -> None:
    info = model_info(remote_model(REMOTE_KEY, display_name="Llama 3.3 70B Instruct", context_length=131072))
    assert info["id"] == REMOTE_KEY
    assert info["label"] == "Llama 3.3 70B Instruct (DeepInfra)"
    assert info["contextWindow"] == 131072
    assert info["description"] == "zdalnie · DeepInfra"


@pytest.mark.parametrize("key, label", [
    ("remote:openai:gpt-5", "gpt-5 (OpenAI)"),
    ("remote:gemini:gemini-3-pro", "gemini-3-pro (Google Gemini)"),
    ("remote:anthropic:claude-opus-5", "claude-opus-5 (Anthropic)"),
    ("remote:unlid:glm-5", "glm-5 (Unlid)"),
])
def test_every_provider_has_a_readable_name(key: str, label: str) -> None:
    assert model_info(remote_model(key))["label"] == label


async def test_the_picker_lists_local_models_first_then_remote(repo: Any) -> None:
    service, _ = make(repo, FakeSession(["meta-llama/Llama-3.3-70B-Instruct"], ["ok"]))
    infos = await service.model_infos()
    assert [i["id"] for i in infos] == [CHAT_KEY, REMOTE_KEY]
    assert infos[1]["label"].endswith("(DeepInfra)")
    await service.aclose()


async def test_a_favourite_remote_model_is_listed_before_the_local_ones(repo: Any) -> None:
    service, _ = make(repo, FakeSession(["meta-llama/Llama-3.3-70B-Instruct"], ["ok"]),
                      favorites=[REMOTE_KEY, "org/not-installed/any"])
    assert [i["id"] for i in await service.model_infos()] == [REMOTE_KEY, CHAT_KEY]
    await service.aclose()


async def test_without_a_key_the_picker_is_local_only(repo: Any) -> None:
    service, _ = make(repo, FakeSession(["x"], ["ok"]), key=None)
    assert [i["id"] for i in await service.model_infos()] == [CHAT_KEY]
    await service.aclose()


async def test_a_provider_that_is_down_hides_only_its_models(repo: Any) -> None:
    service, _ = make(repo, FakeSession([], ["ok"], list_status=500))
    assert [i["id"] for i in await service.model_infos()] == [CHAT_KEY]
    await service.aclose()


async def test_a_remote_key_is_never_fuzzy_matched_to_a_local_model(repo: Any) -> None:
    service, _ = make(repo, FakeSession([], ["ok"]), key=None)
    # "qwen3" would match the local model through the library's resolver.
    assert await service.find_model("remote:deepinfra:Qwen/Qwen3-0.6B") is None
    await service.aclose()


async def test_a_conversation_runs_on_a_remote_model(repo: Any) -> None:
    session = FakeSession(["meta-llama/Llama-3.3-70B-Instruct"], ["Dzień ", "dobry!"])
    service, runtime = make(repo, session)
    frames = await collect(service.send({"conversationId": None, "content": "Cześć", "model": REMOTE_KEY}))
    await service.drain()

    assert (REMOTE_KEY, "chat") in runtime.acquired
    done = next(f for f in frames if f["type"] == "done")
    assert "Dzień dobry!" in json.dumps(done, ensure_ascii=False)
    sent = session.posts[0]
    assert sent["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert sent["messages"][-1] == {"role": "user", "content": "Cześć"}
    await service.aclose()


async def test_a_chosen_remote_default_is_kept_but_never_the_fallback(repo: Any) -> None:
    service, _ = make(repo, FakeSession(["meta-llama/Llama-3.3-70B-Instruct"], ["ok"]))
    settings = repo.get_settings()
    assert await service.default_model_key(settings) == CHAT_KEY
    settings["defaults"]["model"] = REMOTE_KEY
    assert await service.default_model_key(settings) == REMOTE_KEY
    await service.aclose()
