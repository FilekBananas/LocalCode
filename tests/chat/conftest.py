"""Fixtures for the chat backend tests (fakes live in ``chat_fakes.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chat_fakes import (CHAT_KEY, CHAT_KEY_2, EMBED_KEY, FakeEngine, FakeLibrary, FakeRuntime, make_model,
                        make_service)
from localcode.chat.api import ChatApi
from localcode.chat.db import ChatDB
from localcode.chat.repo import ChatRepo
from localcode.chat.service import ChatService
from localcode.config import Settings


@pytest.fixture
def repo(tmp_path: Path) -> ChatRepo:
    db = ChatDB(tmp_path / "chat.db")
    r = ChatRepo(db, tmp_path / "uploads")
    yield r
    r.close()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def library() -> FakeLibrary:
    return FakeLibrary([make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["thinking", "tools"]),
                        make_model(CHAT_KEY_2, name="Qwen3 4B Instruct", fmt="mlx"),
                        make_model(EMBED_KEY, name="Nomic Embed", caps=["embedding"], embedding=True)])


@pytest.fixture
def config() -> Settings:
    return Settings()


@pytest.fixture
def runtime(engine: FakeEngine) -> FakeRuntime:
    return FakeRuntime({CHAT_KEY: engine, CHAT_KEY_2: engine, EMBED_KEY: FakeEngine()})


@pytest.fixture
async def service(repo: ChatRepo, runtime: FakeRuntime, library: FakeLibrary, config: Settings) -> ChatService:
    svc = make_service(repo, runtime, library, config)
    yield svc
    await svc.aclose()


@pytest.fixture
def api(repo: ChatRepo, service: ChatService) -> ChatApi:
    return ChatApi(repo, service)
