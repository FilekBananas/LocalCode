"""Cerber (the user's antivirus) as a gate in front of every model download.

Driven by a fake of Cerber's local API — the real one is Cerber.app's
``cerber.api_server`` (``POST /api/huggingface`` → job → ``GET /api/jobs/{id}``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from localcode.config import CerberSettings, Settings
from localcode.security.cerber import CerberClient, CerberError, CerberGate, check_connection


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return json.dumps(self._body)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeCerber:
    """Cerber's API: one scan job that reports ``states`` in turn, then ``verdict``."""

    def __init__(self, *, verdict: str = "CLEAN", states: tuple[str, ...] = ("running", "done"),
                 token: str = "secret-token-123456", down: bool = False, flagged: list[str] | None = None) -> None:
        self.verdict = verdict
        self.states = list(states)
        self.token = token
        self.down = down
        self.flagged = flagged or []
        self.calls: list[tuple[str, str, Any]] = []

    def session(self) -> "FakeCerber":
        return self

    async def __aenter__(self) -> "FakeCerber":
        if self.down:
            raise ConnectionRefusedError("connection refused")
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def _auth(self, headers: dict[str, str]) -> bool:
        return headers.get("Authorization") == f"Bearer {self.token}"

    def post(self, url: str, headers: dict[str, str], data: str | None = None) -> FakeResponse:
        self.calls.append(("POST", url, json.loads(data or "{}")))
        if not self._auth(headers):
            return FakeResponse(401, {"error": "Brak tokenu albo token jest nieprawidłowy."})
        if url.endswith("/cancel"):
            return FakeResponse(200, {"ok": True})
        return FakeResponse(202, {"job": {"id": "job1", "state": "queued"}, "links": {}})

    def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
        self.calls.append(("GET", url, None))
        if not self._auth(headers):
            return FakeResponse(401, {"error": "token"})
        if url.endswith("/api/info"):
            return FakeResponse(200, {"patterns_total": 1234567, "version": "1.0"})
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        job: dict[str, Any] = {"id": "job1", "state": state}
        if state == "done":
            job.update({"verdict": self.verdict, "findings": len(self.flagged),
                        "summary": {"verdict": self.verdict,
                                    "results": [{"path": p, "verdict": self.verdict} for p in self.flagged]
                                    + [{"path": "README.md", "verdict": "CLEAN"}]}})
        return FakeResponse(200, {"job": job})


async def no_sleep(_: float) -> None:
    return None


def client(cerber: FakeCerber, token: str = "secret-token-123456") -> CerberClient:
    return CerberClient("http://127.0.0.1:8765", token, session_factory=cerber.session)


def gate(cerber: FakeCerber, **cfg: Any) -> CerberGate:
    settings = Settings(cerber=CerberSettings(enabled=True, token="secret-token-123456", **cfg))
    factory = lambda url, token: CerberClient(url, token, session_factory=cerber.session)  # noqa: E731
    g = CerberGate(settings, client_factory=factory)
    return g


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("localcode.security.cerber.POLL_S", 0.0)


async def test_a_clean_model_passes_and_the_scan_names_the_repository() -> None:
    cerber = FakeCerber(verdict="CLEAN")
    verdict = await client(cerber).scan_huggingface("Qwen/Qwen3-8B-GGUF")
    assert verdict.verdict == "CLEAN"
    assert cerber.calls[0] == ("POST", "http://127.0.0.1:8765/api/huggingface", {"url": "Qwen/Qwen3-8B-GGUF"})


@pytest.mark.parametrize("verdict", ["MALICIOUS", "PUA", "SUSPICIOUS"])
async def test_a_bad_verdict_blocks_the_download(verdict: str) -> None:
    cerber = FakeCerber(verdict=verdict, flagged=["pytorch_model.bin"])
    with pytest.raises(CerberError) as excinfo:
        await gate(cerber).check("evil/model")
    assert excinfo.value.code == "cerber_blocked"
    assert "pytorch_model.bin" in excinfo.value.message
    assert "README.md" not in excinfo.value.message          # clean files are not named


async def test_info_is_not_a_reason_to_block() -> None:
    assert (await gate(FakeCerber(verdict="INFO")).check("org/model")).verdict == "INFO"


async def test_a_clean_repository_is_not_scanned_twice_the_same_day() -> None:
    cerber = FakeCerber(verdict="CLEAN")
    g = gate(cerber)
    await g.check("org/model")
    posts = sum(1 for c in cerber.calls if c[0] == "POST")
    await g.check("org/model")
    assert sum(1 for c in cerber.calls if c[0] == "POST") == posts


async def test_a_closed_cerber_stops_the_download_instead_of_skipping_the_check() -> None:
    with pytest.raises(CerberError) as excinfo:
        await gate(FakeCerber(down=True)).check("org/model")
    assert excinfo.value.code == "cerber_unavailable"
    assert "Integracje → API" in excinfo.value.message


async def test_a_wrong_token_says_so() -> None:
    with pytest.raises(CerberError) as excinfo:
        await client(FakeCerber(), token="wrong-token-0000000").scan_huggingface("org/model")
    assert excinfo.value.code == "cerber_bad_token"


async def test_a_failed_scan_is_not_a_pass() -> None:
    with pytest.raises(CerberError) as excinfo:
        await client(FakeCerber(states=("running", "failed"))).scan_huggingface("org/model")
    assert excinfo.value.code == "cerber_failed"


async def test_a_cancelled_download_cancels_the_scan() -> None:
    cerber = FakeCerber(states=("running",))
    with pytest.raises(asyncio.CancelledError):
        await client(cerber).scan_huggingface("org/model", cancelled=lambda: True)
    assert any(c[1].endswith("/api/jobs/job1/cancel") for c in cerber.calls)


async def test_a_scan_that_never_ends_times_out() -> None:
    cerber = FakeCerber(states=("running",))
    with pytest.raises(CerberError) as excinfo:
        await client(cerber).scan_huggingface("org/model", timeout_s=0.0)
    assert excinfo.value.code == "cerber_timeout"


async def test_when_the_check_is_off_nothing_is_asked() -> None:
    cerber = FakeCerber()
    g = CerberGate(Settings(), client_factory=lambda u, t: CerberClient(u, t, session_factory=cerber.session))
    assert await g.check("org/model") is None and cerber.calls == []


async def test_the_connection_check_reports_without_raising() -> None:
    settings = Settings(cerber=CerberSettings(enabled=True, token="secret-token-123456"))
    ok = await check_connection(settings, client_factory=lambda u, t: client(FakeCerber(), t))
    assert ok["ok"] is True
    down = await check_connection(settings, client_factory=lambda u, t: client(FakeCerber(down=True), t))
    assert down["ok"] is False and "nie odpowiada" in down["message"]
    no_token = await check_connection(Settings())
    assert no_token["ok"] is False


async def test_the_download_manager_asks_cerber_before_fetching_anything(tmp_path) -> None:
    from localcode.hub.download import DownloadError, DownloadManager, _Job

    settings = Settings(models_dir=str(tmp_path / "models"))
    manager = DownloadManager(settings, state_path=tmp_path / "downloads.json")
    manager._cerber = gate(FakeCerber(verdict="MALICIOUS"))
    job = _Job(id="j1", repo="evil/model", dest=tmp_path / "models" / "evil", files=[], label="evil",
               format="gguf", model_key=None)
    stages: list[str] = []
    manager._emit = lambda job, force=False: stages.append(job.stage)  # type: ignore[method-assign]
    with pytest.raises(DownloadError) as excinfo:
        await manager._cerber_check(job)
    assert excinfo.value.code == "cerber_blocked"
    assert stages == ["cerber"]
