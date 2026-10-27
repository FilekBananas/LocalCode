"""End-to-end check that ``RuntimeManager`` really refuses to load a model whose
security scan (``localcode/security/scan.py``, wired in ``localcode/hub/library.py``)
came back "dangerous", and that the explicit ``allowDangerous`` override really
lets it through.

This mirrors the fixture style of ``test_manager.py`` (a fake engine, a fake
library) but stays self-contained in its own file: the behaviour under test
(``RuntimeManager._check_security_scan``) is a small, focused slice, and a
separate file avoids growing the already-large ``test_manager.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest

from localcode.config import Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.base import CompletionChunk, Engine, EngineError, GenStats, LoadConfig, SamplingParams
from localcode.runtime.manager import RuntimeManager

META = ModelMeta(arch="qwen3", context_length=32768, n_layers=28, n_heads=16, n_kv_heads=8,
                 head_dim=128, hidden_size=1024, vocab_size=151936)
GiB = 1024 ** 3


class FakeEngine(Engine):
    format = "fake"

    def __init__(self, instance_id: str, model: LocalModel, config: LoadConfig) -> None:
        super().__init__(instance_id, model, config)
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    async def apply_template(self, messages, tools=None, *, add_generation_prompt=True,
                             template_kwargs=None) -> str:
        return ""

    async def tokenize(self, text: str) -> list[int]:
        return []

    async def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:
        yield CompletionChunk(done=True, finish_reason="stop", stats=GenStats(completion_tokens=1))


@dataclass
class FakeFactory:
    def __call__(self, instance_id: str, model: LocalModel, config: LoadConfig) -> Engine:
        return FakeEngine(instance_id, model, config)


class FakeLibrary:
    def __init__(self, models: list[LocalModel]) -> None:
        self.models = {m.key: m for m in models}

    def get(self, key: str) -> LocalModel | None:
        return self.models.get(key)

    def load_defaults(self, key: str) -> dict | None:
        return None


def make_model(key: str, *, scan: dict[str, Any] | None) -> LocalModel:
    return LocalModel(key=key, format="gguf", path=f"/nonexistent/{key}",
                      display_name=key.split("/")[-1], size_bytes=1 * GiB, meta=META, scan=scan)


DANGEROUS_SCAN = {
    "verdict": "dangerous",
    "summary": "Wykryto niebezpieczne elementy w plikach modelu (liczba: 1) — wczytanie mogłoby uruchomić kod.",
    "findings": [{"file": "model.bin", "kind": "pickle_import", "severity": "dangerous",
                 "reason": "Wczytanie pliku może wykonać „os.system”.", "detail": "os.system"}],
}
SUSPICIOUS_SCAN = {"verdict": "suspicious", "summary": "Znaleziono podejrzane elementy…", "findings": []}
CLEAN_SCAN = {"verdict": "clean", "summary": "Przeanalizowano pliki pickle…", "findings": []}


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.runtime.ram_limit_gb = 20
    s.runtime.default_ctx = 4096
    return s


@pytest.fixture
async def rt(settings):
    models = [
        make_model("org/dangerous", scan=DANGEROUS_SCAN),
        make_model("org/suspicious", scan=SUSPICIOUS_SCAN),
        make_model("org/clean", scan=CLEAN_SCAN),
        make_model("org/unscanned", scan=None),
    ]
    manager = RuntimeManager(settings, FakeLibrary(models), engine_factory=FakeFactory())
    try:
        yield manager
    finally:
        await manager.shutdown()


# --------------------------------------------------------------------------- #
# Blocking
# --------------------------------------------------------------------------- #


async def test_dangerous_verdict_blocks_loading_with_a_polish_message(rt: RuntimeManager):
    with pytest.raises(EngineError) as excinfo:
        await rt.load("org/dangerous")
    message = str(excinfo.value)
    assert "nie został wczytany ze względów bezpieczeństwa" in message
    assert "wczytanie mogłoby uruchomić kod" in message
    assert "wczytaj mimo to" in message and "allowDangerous" in message
    assert rt.status()["instances"] == []      # nothing was actually spawned


async def test_dangerous_verdict_blocks_acquire_too(rt: RuntimeManager):
    """The JIT path (``acquire``, used by chat/agent requests) is gated the same way."""
    with pytest.raises(EngineError, match="względów bezpieczeństwa"):
        async with rt.acquire("org/dangerous"):
            pass


async def test_suspicious_and_clean_and_unscanned_all_load_normally(rt: RuntimeManager):
    """Only an explicit "dangerous" verdict blocks — see ``_check_security_scan``:
    "suspicious" is a warning shown elsewhere (CLI/TUI), and ``scan=None``
    (not scanned yet, or the scanner itself failed) must never brick loading."""
    for key in ("org/suspicious", "org/clean", "org/unscanned"):
        info = await rt.load(key)
        assert info["state"] == "ready"
        await rt.unload(info["id"])


# --------------------------------------------------------------------------- #
# The explicit override
# --------------------------------------------------------------------------- #


async def test_allow_dangerous_override_lets_it_load(rt: RuntimeManager):
    info = await rt.load("org/dangerous", LoadConfig(n_ctx=4096, allow_dangerous=True))
    assert info["state"] == "ready"


async def test_override_is_sticky_for_the_process_lifetime(rt: RuntimeManager, settings):
    """A later reload of the same model (e.g. a bigger context) must not ask again."""
    first = await rt.load("org/dangerous", LoadConfig(n_ctx=4096, allow_dangerous=True))
    await rt.unload(first["id"])
    # no allow_dangerous this time — the earlier override is remembered per model key
    second = await rt.load("org/dangerous", LoadConfig(n_ctx=8192))
    assert second["state"] == "ready" and second["nCtx"] == 8192


async def test_override_does_not_leak_to_a_different_dangerous_model(rt: RuntimeManager):
    models = [
        make_model("org/dangerous", scan=DANGEROUS_SCAN),
        make_model("org/also-dangerous", scan=DANGEROUS_SCAN),
    ]
    manager = RuntimeManager(Settings(), FakeLibrary(models), engine_factory=FakeFactory())
    try:
        await manager.load("org/dangerous", LoadConfig(n_ctx=4096, allow_dangerous=True))
        with pytest.raises(EngineError, match="względów bezpieczeństwa"):
            await manager.load("org/also-dangerous")
    finally:
        await manager.shutdown()


async def test_allow_dangerous_false_still_blocks(rt: RuntimeManager):
    """Passing a config without turning the flag on is exactly the same as
    passing no config at all — the default is safe."""
    with pytest.raises(EngineError, match="względów bezpieczeństwa"):
        await rt.load("org/dangerous", LoadConfig(n_ctx=4096, allow_dangerous=False))
