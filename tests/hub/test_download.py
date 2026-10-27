"""Tests for hub/download.py: the job queue, progress, cancellation and recovery.

The unit tests replace the download worker with ``fake_worker.py`` (same
protocol, no network) so that queueing, throttling, cancellation and partial
cleanup are deterministic.  The ``network`` tests run the real worker against
Hugging Face: a 49 MB embedding model is downloaded into an isolated
``LOCALCODE_HOME`` and a second, larger download is cancelled mid-flight.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from localcode.config import Settings
from localcode.hub import download as dl
from localcode.hub.download import DownloadError, DownloadManager, job_format, job_label, job_model_key
from localcode.hub.library import ModelLibrary

FAKE_WORKER = Path(__file__).parent / "fake_worker.py"
REPO = "unsloth/Qwen3-0.6B-GGUF"
FILE = "Qwen3-0.6B-Q4_K_M.gguf"
NOMIC_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
NOMIC_SMALL = "nomic-embed-text-v1.5.Q2_K.gguf"
NOMIC_SMALL_SIZE = 49_361_088
NOMIC_BIG = "nomic-embed-text-v1.5.Q8_0.gguf"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class _AnyFile(dict):
    """Hub metadata stub: every requested file exists, sizes stay as given."""

    def __contains__(self, key: object) -> bool:
        return True

    def __getitem__(self, key: object) -> tuple[int | None, str]:
        return None, "0" * 64


@pytest.fixture
def use_fake_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``fake_worker.py`` instead of the real one and never touch the network."""
    monkeypatch.setattr(DownloadManager, "_worker_command", lambda self: [sys.executable, str(FAKE_WORKER)])
    monkeypatch.setattr(DownloadManager, "_file_metadata", lambda self, repo: _AnyFile())


@pytest.fixture
def events() -> list[dict[str, Any]]:
    return []


@pytest.fixture
async def manager(settings: Settings, events: list[dict[str, Any]]):
    completed: list[dict[str, Any]] = []
    m = DownloadManager(settings, on_event=events.append, on_complete=completed.append)
    m.completed = completed  # type: ignore[attr-defined]  - handy for assertions
    try:
        yield m
    finally:
        await m.shutdown()


def job_events(events: list[dict[str, Any]], job_id: str) -> list[dict[str, Any]]:
    return [e["job"] for e in events if e["type"] == "download" and e["job"]["id"] == job_id]


async def wait_for(predicate, timeout: float = 20.0, interval: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within the timeout")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - not our process any more
        return True
    return True


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_job_label_and_key_for_gguf():
    files = [{"name": "Qwen3-0.6B-Q4_K_M.gguf"}]
    assert job_label(REPO, files) == "Qwen3 0.6B · Q4_K_M"
    assert job_format(files) == "gguf"
    assert job_model_key(REPO, files) == "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"


def test_job_key_uses_the_first_shard_of_a_split_model():
    files = [{"name": "Q4/x-Q4_K_M-00001-of-00002.gguf"}, {"name": "Q4/x-Q4_K_M-00002-of-00002.gguf"}]
    assert job_model_key(REPO, files) == "unsloth/Qwen3-0.6B-GGUF/x-Q4_K_M"


def test_job_key_and_label_for_mlx():
    files = [{"name": "config.json"}, {"name": "model.safetensors"}]
    repo = "mlx-community/Qwen3-0.6B-4bit"
    assert job_model_key(repo, files) == repo
    assert job_label(repo, files) == "Qwen3 0.6B · 4bit"
    assert job_format(files) == "mlx"


def test_projector_only_job_has_no_model_key():
    files = [{"name": "mmproj-F16.gguf"}]
    assert job_model_key(REPO, files) is None
    assert "projektor" in job_label(REPO, files)


def test_normalise_files_accepts_names_dicts_and_camel_case():
    out = dl._normalise_files(["a.gguf", {"name": "b.gguf", "sizeBytes": 7}, {"rfilename": "c.gguf", "size": 9,
                                                                             "sha256": "ab"}])
    assert out == [
        {"name": "a.gguf", "size": None, "sha256": None},
        {"name": "b.gguf", "size": 7, "sha256": None},
        {"name": "c.gguf", "size": 9, "sha256": "ab"},
    ]


@pytest.mark.parametrize("bad", [["../escape.gguf"], ["a/../../b.gguf"], [""], [{"name": 3}], [5]])
def test_normalise_files_rejects_dangerous_names(bad):
    with pytest.raises(DownloadError):
        dl._normalise_files(bad)


def test_normalise_files_strips_a_leading_slash():
    assert dl._normalise_files(["/Q4/model.gguf"])[0]["name"] == "Q4/model.gguf"


# --------------------------------------------------------------------------- #
# start() / DTO
# --------------------------------------------------------------------------- #


async def test_start_returns_a_queued_camel_case_dto(manager: DownloadManager, use_fake_worker):
    job = manager.start(REPO, [{"name": FILE, "size": 1000}])
    assert job["state"] == "queued"
    assert job["repo"] == REPO and job["files"] == [FILE]
    assert job["format"] == "gguf"
    assert job["totalBytes"] == 1000 and job["doneBytes"] == 0
    assert job["modelKey"] == "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
    assert job["label"] == "Qwen3 0.6B · Q4_K_M"
    assert job["dest"].endswith("models/unsloth/Qwen3-0.6B-GGUF")
    assert set(job) >= {"id", "speedBps", "etaSeconds", "error", "errorCode", "startedAt", "finishedAt",
                        "createdAt", "progress", "stage", "currentFile"}
    assert manager.list() == [job] or manager.list()[0]["id"] == job["id"]


async def test_start_rejects_bad_input(manager: DownloadManager):
    with pytest.raises(DownloadError) as exc:
        manager.start("not-a-repo", [FILE])
    assert exc.value.code == "invalid"
    assert "repozytorium" in exc.value.message.lower()
    with pytest.raises(DownloadError):
        manager.start(REPO, [])


async def test_duplicate_start_returns_the_running_job(manager: DownloadManager, use_fake_worker,
                                                       monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "20")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    first = manager.start(REPO, [{"name": FILE, "size": 4096}])
    second = manager.start(REPO, [{"name": FILE, "size": 4096}])
    assert second["id"] == first["id"]
    assert len(manager.list()) == 1


async def test_disk_space_is_checked_before_starting(manager: DownloadManager, use_fake_worker,
                                                     monkeypatch: pytest.MonkeyPatch):
    usage = shutil.disk_usage(Path.home())._replace(free=10 * 1024**2)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda _p: usage)
    with pytest.raises(DownloadError) as exc:
        manager.start(REPO, [{"name": FILE, "size": 8 * 1024**3}])
    assert exc.value.code == "disk_full"
    assert "Za mało miejsca na dysku" in exc.value.message
    assert manager.list() == []


# --------------------------------------------------------------------------- #
# Running jobs
# --------------------------------------------------------------------------- #


async def test_download_completes_and_reports_progress(manager: DownloadManager, use_fake_worker,
                                                       events, models_dir: Path,
                                                       monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "8")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.1")      # long enough for throttled progress events
    job = manager.start(REPO, [{"name": FILE, "size": 8192}])
    final = await manager.wait(job["id"], timeout=30)

    assert final["state"] == "done"
    assert final["doneBytes"] == final["totalBytes"] == 8192
    assert final["progress"] == 1.0
    assert final["error"] is None and final["errorCode"] is None
    assert final["finishedAt"] >= final["startedAt"] >= final["createdAt"]
    assert final["speedBps"] == 0.0 and final["etaSeconds"] is None

    downloaded = models_dir / REPO / FILE
    assert downloaded.is_file() and downloaded.stat().st_size == 8192
    assert not (models_dir / REPO / ".cache" / "localcode").exists() or \
        not list((models_dir / REPO / ".cache" / "localcode").glob("*.part"))

    seen = job_events(events, job["id"])
    assert [e["state"] for e in seen][0] == "queued"
    assert seen[-1]["state"] == "done"
    assert any(e["state"] == "downloading" and 0 < e["doneBytes"] < 8192 for e in seen)
    assert [e["doneBytes"] for e in seen] == sorted(e["doneBytes"] for e in seen)
    assert manager.completed and manager.completed[0]["id"] == job["id"]  # type: ignore[attr-defined]


async def test_progress_events_are_throttled(manager: DownloadManager, use_fake_worker, events,
                                             monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "60")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.02")     # ~60 worker events in ~1.2 s
    started = time.monotonic()
    job = manager.start(REPO, [{"name": FILE, "size": 60_000}])
    await manager.wait(job["id"], timeout=30)
    elapsed = time.monotonic() - started
    downloading = [e for e in job_events(events, job["id"]) if e["state"] == "downloading"]
    # ≤ 4 throttled events per second, plus the forced ones (start, file_done, verify)
    assert len(downloading) <= 4 * elapsed + 6, f"{len(downloading)} events in {elapsed:.1f}s"
    assert len(downloading) >= 3


async def test_speed_and_eta_are_reported(manager: DownloadManager, use_fake_worker, events,
                                          monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "40")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.03")
    job = manager.start(REPO, [{"name": FILE, "size": 4_000_000}])
    await manager.wait(job["id"], timeout=30)
    during = [e for e in job_events(events, job["id"]) if e["state"] == "downloading" and e["doneBytes"]]
    assert any(e["speedBps"] > 0 for e in during)
    assert any(e["etaSeconds"] is not None and e["etaSeconds"] > 0 for e in during)


async def test_only_two_jobs_run_at_once(manager: DownloadManager, use_fake_worker,
                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "10")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    jobs = [manager.start(f"org/repo{i}", [{"name": f"m{i}.gguf", "size": 1000}]) for i in range(4)]
    seen_queued_while_running = False
    for _ in range(200):
        states = [j["state"] for j in manager.list()]
        assert states.count("downloading") <= 2, states
        if states.count("downloading") == 2 and "queued" in states:
            seen_queued_while_running = True
        if all(s in ("done", "error", "cancelled") for s in states):
            break
        await asyncio.sleep(0.02)
    for job in jobs:
        assert (await manager.wait(job["id"], timeout=30))["state"] == "done"
    assert seen_queued_while_running


async def test_worker_error_is_reported_in_polish(manager: DownloadManager, use_fake_worker,
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_MODE", "error")
    job = manager.start(REPO, [{"name": FILE, "size": 4096}])
    final = await manager.wait(job["id"], timeout=30)
    assert final["state"] == "error"
    assert final["errorCode"] == "network"
    assert "Brak połączenia z Hugging Face" in final["error"]
    assert final["finishedAt"] is not None


async def test_worker_crash_is_reported(manager: DownloadManager, use_fake_worker,
                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_MODE", "crash")
    job = manager.start(REPO, [{"name": FILE, "size": 4096}])
    final = await manager.wait(job["id"], timeout=30)
    assert final["state"] == "error"
    assert final["errorCode"] == "worker_failed"
    assert "kod 3" in final["error"]


async def test_non_json_worker_output_is_ignored(manager: DownloadManager, use_fake_worker,
                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_MODE", "garbage")
    job = manager.start(REPO, [{"name": FILE, "size": 1024}])
    assert (await manager.wait(job["id"], timeout=30))["state"] == "done"


async def test_missing_sizes_are_resolved_from_the_hub(manager: DownloadManager, use_fake_worker,
                                                       monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_metadata(self, repo: str):
        calls.append(repo)
        return {FILE: (2048, "0" * 64), "README.md": (10, None)}

    monkeypatch.setattr(DownloadManager, "_file_metadata", fake_metadata)
    job = manager.start(REPO, [FILE])
    assert job["totalBytes"] == 0
    final = await manager.wait(job["id"], timeout=30)
    assert calls == [REPO]
    assert final["state"] == "done"
    assert final["totalBytes"] == 2048


async def test_unknown_file_is_reported(manager: DownloadManager, use_fake_worker,
                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DownloadManager, "_file_metadata", lambda self, repo: {"other.gguf": (1, None)})
    job = manager.start(REPO, ["missing.gguf"])
    final = await manager.wait(job["id"], timeout=30)
    assert final["state"] == "error" and final["errorCode"] == "file_not_found"
    assert "missing.gguf" in final["error"]


def _hub_exception(kind: str) -> BaseException:
    import httpx
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    request = httpx.Request("GET", "https://huggingface.co/x")
    return {
        "gated": lambda: GatedRepoError("gated", response=httpx.Response(403, request=request)),
        "unauthorized": lambda: HfHubHTTPError("401", response=httpx.Response(401, request=request)),
        "not_found": lambda: RepositoryNotFoundError("404", response=httpx.Response(404, request=request)),
        "network": lambda: httpx.ConnectError("no route to host"),
    }[kind]()


@pytest.mark.parametrize("kind,fragment", [
    ("gated", "licencji na Hugging Face"),
    ("unauthorized", "odrzucił token"),
    ("not_found", "Nie znaleziono repozytorium"),
    ("network", "Brak połączenia z Hugging Face"),
])
async def test_hub_errors_are_translated(manager: DownloadManager, use_fake_worker, kind: str, fragment: str,
                                         monkeypatch: pytest.MonkeyPatch):
    def boom(self, repo: str):
        raise _hub_exception(kind)

    monkeypatch.setattr(DownloadManager, "_file_metadata", boom)
    job = manager.start(REPO, [FILE])
    final = await manager.wait(job["id"], timeout=30)
    assert final["state"] == "error" and final["errorCode"] == kind
    assert fragment in final["error"]


async def test_hub_lookup_failures_are_survivable_when_the_sizes_are_known(
    manager: DownloadManager, use_fake_worker, monkeypatch: pytest.MonkeyPatch,
):
    """A checksum lookup that fails must not stop a download whose sizes we know."""
    def boom(self, repo: str):
        raise _hub_exception("network")

    monkeypatch.setattr(DownloadManager, "_file_metadata", boom)
    job = manager.start(REPO, [{"name": FILE, "size": 2048}])
    assert (await manager.wait(job["id"], timeout=30))["state"] == "done"


async def test_the_worker_gets_the_token_and_destination(manager: DownloadManager, use_fake_worker,
                                                         settings: Settings, models_dir: Path,
                                                         tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec_file = tmp_path / "spec.json"
    monkeypatch.setenv("LC_FAKE_SPEC_FILE", str(spec_file))
    settings.hf_token = "  hf_secret  "
    job = manager.start(REPO, [{"name": FILE, "size": 1024, "sha256": "ab" * 32}])
    await manager.wait(job["id"], timeout=30)
    spec = json.loads(spec_file.read_text())
    assert spec["repo"] == REPO
    assert spec["token"] == "hf_secret"                     # trimmed, taken from the settings
    assert spec["dest"] == str(models_dir / REPO)
    assert spec["files"] == [{"name": FILE, "size": 1024, "sha256": "ab" * 32}]
    assert spec["engine"] == "http"


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


async def test_cancel_running_job_kills_the_worker_and_removes_partials(
    manager: DownloadManager, use_fake_worker, models_dir: Path, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("LC_FAKE_PID_FILE", str(pid_file))
    monkeypatch.setenv("LC_FAKE_STEPS", "200")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    job = manager.start(REPO, [{"name": FILE, "size": 200_000}])
    part = models_dir / REPO / ".cache" / "localcode" / (FILE + ".part")
    await wait_for(lambda: part.exists() and part.stat().st_size > 0)
    pid = int(pid_file.read_text())

    final = await manager.acancel(job["id"])

    assert final["state"] == "cancelled"
    assert final["error"] and "anulowane" in final["error"].lower()
    await wait_for(lambda: not pid_alive(pid), timeout=5)
    assert not part.exists()
    assert not (models_dir / REPO / FILE).exists()
    assert manager.get(job["id"])["state"] == "cancelled"


async def test_cancel_kills_a_worker_that_ignores_sigterm(manager: DownloadManager, use_fake_worker,
                                                          tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dl, "TERMINATE_GRACE_S", 0.3)
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("LC_FAKE_PID_FILE", str(pid_file))
    monkeypatch.setenv("LC_FAKE_MODE", "ignore_term")
    monkeypatch.setenv("LC_FAKE_STEPS", "400")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    job = manager.start(REPO, [{"name": FILE, "size": 400_000}])
    await wait_for(lambda: pid_file.exists())
    pid = int(pid_file.read_text())
    final = await manager.acancel(job["id"], timeout=20)
    assert final["state"] == "cancelled"
    await wait_for(lambda: not pid_alive(pid), timeout=5)


async def test_cancel_queued_job_never_starts_it(settings: Settings, use_fake_worker, models_dir: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LC_FAKE_STEPS", "10")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    manager = DownloadManager(settings, max_concurrent=1)
    try:
        first = manager.start(REPO, [{"name": FILE, "size": 1000}])
        second = manager.start("org/other", [{"name": "other.gguf", "size": 1000}])
        assert second["state"] == "queued"
        final = await manager.acancel(second["id"])
        assert final["state"] == "cancelled"
        assert (await manager.wait(first["id"], timeout=30))["state"] == "done"
        assert not (models_dir / "org" / "other").exists()
    finally:
        await manager.shutdown()


async def test_cancelling_a_finished_job_is_a_no_op(manager: DownloadManager, use_fake_worker):
    job = manager.start(REPO, [{"name": FILE, "size": 512}])
    await manager.wait(job["id"], timeout=30)
    assert manager.cancel(job["id"])["state"] == "done"
    with pytest.raises(DownloadError) as exc:
        manager.cancel("dl-nope")
    assert exc.value.code == "not_found"


async def test_shutdown_stops_workers_but_keeps_partials(settings: Settings, use_fake_worker, models_dir: Path,
                                                         tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("LC_FAKE_PID_FILE", str(pid_file))
    monkeypatch.setenv("LC_FAKE_STEPS", "200")
    monkeypatch.setenv("LC_FAKE_DELAY", "0.05")
    manager = DownloadManager(settings)
    job = manager.start(REPO, [{"name": FILE, "size": 200_000}])
    part = models_dir / REPO / ".cache" / "localcode" / (FILE + ".part")
    await wait_for(lambda: part.exists() and part.stat().st_size > 0)
    pid = int(pid_file.read_text())
    await manager.shutdown()
    await wait_for(lambda: not pid_alive(pid), timeout=5)
    assert manager.get(job["id"])["state"] == "error"
    assert manager.get(job["id"])["errorCode"] == "interrupted"
    assert part.exists(), "partial files must survive a shutdown so the download can resume"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


async def test_jobs_are_persisted_and_reloaded(manager: DownloadManager, use_fake_worker, hub_home: Path,
                                               settings: Settings):
    job = manager.start(REPO, [{"name": FILE, "size": 1024}])
    await manager.wait(job["id"], timeout=30)
    state_file = hub_home / "data" / "downloads.json"
    assert state_file.is_file()
    raw = json.loads(state_file.read_text())
    assert raw["version"] == dl.STATE_VERSION
    assert raw["jobs"][0]["job"]["state"] == "done"
    assert raw["jobs"][0]["fileSpecs"][0]["name"] == FILE

    reloaded = DownloadManager(settings)
    assert [j["id"] for j in reloaded.list()] == [job["id"]]
    assert reloaded.list()[0]["state"] == "done"
    assert reloaded.clear_finished() == []
    assert json.loads(state_file.read_text())["jobs"] == []


async def test_a_running_job_from_a_previous_run_becomes_an_error(settings: Settings, hub_home: Path):
    state_file = hub_home / "data" / "downloads.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    dead_pid = 999_999
    state_file.write_text(json.dumps({
        "version": dl.STATE_VERSION,
        "jobs": [{
            "job": {"id": "dl-old", "repo": REPO, "files": [FILE], "label": "x", "format": "gguf",
                    "totalBytes": 100, "doneBytes": 40, "state": "downloading", "createdAt": 1,
                    "startedAt": 2, "finishedAt": None, "modelKey": None,
                    "dest": str(Path(settings.models_dir) / REPO)},
            "pid": dead_pid, "fileSpecs": [{"name": FILE, "size": 100}], "completed": [],
        }, {
            "job": {"id": "dl-queued", "repo": REPO, "files": [FILE], "label": "x", "format": "gguf",
                    "totalBytes": 100, "doneBytes": 0, "state": "queued", "createdAt": 1,
                    "modelKey": None, "dest": str(Path(settings.models_dir) / REPO)},
            "pid": None, "fileSpecs": [{"name": FILE, "size": 100}], "completed": [],
        }],
    }))
    manager = DownloadManager(settings)
    try:
        jobs = {j["id"]: j for j in manager.list()}
        assert jobs["dl-old"]["state"] == "error"
        assert jobs["dl-old"]["errorCode"] == "interrupted"
        assert "przerwane" in jobs["dl-old"]["error"]
        assert jobs["dl-queued"]["state"] == "error"
        assert jobs["dl-old"]["doneBytes"] == 40      # kept, so the UI can show what is resumable
    finally:
        await manager.shutdown()


async def test_an_orphaned_worker_is_killed_on_reload(settings: Settings, hub_home: Path):
    """A worker that somehow survived the app is killed when its job is reloaded."""
    orphan = subprocess.Popen([sys.executable, "-m", "localcode.hub.download_worker"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert pid_alive(orphan.pid)
        state_file = hub_home / "data" / "downloads.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps({
            "version": dl.STATE_VERSION,
            "jobs": [{"job": {"id": "dl-orphan", "repo": REPO, "files": [FILE], "state": "downloading",
                              "createdAt": 1, "dest": str(Path(settings.models_dir) / REPO)},
                      "pid": orphan.pid, "fileSpecs": [{"name": FILE, "size": 10}], "completed": []}],
        }))
        manager = DownloadManager(settings)
        try:
            assert manager.get("dl-orphan")["state"] == "error"
            await wait_for(lambda: orphan.poll() is not None, timeout=10)
        finally:
            await manager.shutdown()
    finally:
        if orphan.poll() is None:  # pragma: no cover - only if the kill failed
            orphan.kill()
        orphan.wait(10)


async def test_a_corrupt_state_file_is_ignored(settings: Settings, hub_home: Path):
    (hub_home / "data").mkdir(parents=True, exist_ok=True)
    (hub_home / "data" / "downloads.json").write_text("{not json")
    manager = DownloadManager(settings)
    try:
        assert manager.list() == []
    finally:
        await manager.shutdown()


# --------------------------------------------------------------------------- #
# Real downloads (network)
# --------------------------------------------------------------------------- #


@pytest.mark.network
async def test_real_download_reports_progress_and_lands_in_the_library(settings: Settings, models_dir: Path):
    events: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    manager = DownloadManager(settings, on_event=events.append, on_complete=done.append)
    try:
        job = manager.start(NOMIC_REPO, [NOMIC_SMALL])        # no size: resolved from the Hub
        final = await manager.wait(job["id"], timeout=600)
        assert final["state"] == "done", final["error"]
        assert final["totalBytes"] == NOMIC_SMALL_SIZE
        assert final["doneBytes"] == NOMIC_SMALL_SIZE
        assert final["modelKey"] == f"{NOMIC_REPO}/nomic-embed-text-v1.5.Q2_K"

        path = models_dir / NOMIC_REPO / NOMIC_SMALL
        assert path.stat().st_size == NOMIC_SMALL_SIZE
        assert not list((models_dir / NOMIC_REPO).rglob("*.part"))

        seen = job_events(events, job["id"])
        progress = [e["doneBytes"] for e in seen if e["state"] == "downloading"]
        assert progress == sorted(progress) and progress[-1] == NOMIC_SMALL_SIZE
        assert len([p for p in progress if 0 < p < NOMIC_SMALL_SIZE]) >= 2, progress
        assert any(e["speedBps"] > 0 for e in seen if e["state"] == "downloading")
        assert done and done[0]["modelKey"] == final["modelKey"]

        library = ModelLibrary(settings)
        keys = [m.key for m in library.scan()]
        assert final["modelKey"] in keys
    finally:
        await manager.shutdown()


@pytest.mark.network
async def test_real_download_can_be_cancelled_mid_flight(settings: Settings, models_dir: Path):
    manager = DownloadManager(settings)
    try:
        job = manager.start(NOMIC_REPO, [NOMIC_BIG])
        part = models_dir / NOMIC_REPO / ".cache" / "localcode" / (NOMIC_BIG + ".part")
        await wait_for(lambda: part.exists() and part.stat().st_size > 1_000_000, timeout=120)
        pid = manager._jobs[job["id"]].pid
        assert pid and pid_alive(pid)

        final = await manager.acancel(job["id"], timeout=30)

        assert final["state"] == "cancelled"
        assert 0 < final["doneBytes"] < final["totalBytes"]
        await wait_for(lambda: not pid_alive(pid), timeout=10)
        assert not part.exists()
        assert not (models_dir / NOMIC_REPO / NOMIC_BIG).exists()
    finally:
        await manager.shutdown()
