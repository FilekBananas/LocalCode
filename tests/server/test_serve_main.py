"""``localcode serve`` as a real subprocess.

Covers the contract the Mac app and the TUI depend on: the ``LOCALCODE_READY``
line, ``server.json`` (contents + 0600), token enforcement over real HTTP,
``--parent-pid`` supervision and a SIGTERM shutdown that leaves no engine
process behind.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import pytest

from lc_testing import model_path

READY_PREFIX = "LOCALCODE_READY port="
GGUF_TINY_KEY = "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"


class Server:
    """A spawned ``python -m localcode.server`` process."""

    def __init__(self, process: subprocess.Popen[str], port: int, token: str, home: Path) -> None:
        self.process = process
        self.port = port
        self.token = token
        self.home = home

    @property
    def info_path(self) -> Path:
        return self.home / "server.json"

    def info(self) -> dict[str, Any]:
        return json.loads(self.info_path.read_text())

    def get(self, path: str, *, token: str | None = "", timeout: float = 10.0) -> tuple[int, Any]:
        """GET ``path``; ``token=""`` means "use the server token", ``None`` means "no header"."""
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        used = self.token if token == "" else token
        if used:
            request.add_header("Authorization", f"Bearer {used}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def post(self, path: str, payload: dict[str, Any], *, timeout: float = 120.0) -> tuple[int, Any]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")

    def wait(self, timeout: float) -> int | None:
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


def _read_line(process: subprocess.Popen[str], timeout: float) -> str:
    box: dict[str, str] = {}

    def reader() -> None:
        assert process.stdout is not None
        box["line"] = process.stdout.readline()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout)
    return box.get("line", "")


def spawn(home: Path, *args: str, token: str = "sekret-testowy", env: dict[str, str] | None = None,
          timeout: float = 60.0) -> Server:
    environment = {**os.environ, "LOCALCODE_HOME": str(home), "PYTHONUNBUFFERED": "1", **(env or {})}
    process = subprocess.Popen(
        [sys.executable, "-m", "localcode.server", "serve", "--port", "0", "--token", token, *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
    )
    line = _read_line(process, timeout)
    if not line.startswith(READY_PREFIX):
        process.kill()
        stderr = process.stderr.read() if process.stderr else ""
        raise AssertionError(f"brak linii READY (dostano {line!r}); stderr:\n{stderr}")
    return Server(process, int(line[len(READY_PREFIX):].strip()), token, home)


def stop(server: Server, sig: int = signal.SIGTERM, timeout: float = 15.0) -> int | None:
    if server.process.poll() is None:
        server.process.send_signal(sig)
    code = server.wait(timeout)
    if code is None:
        server.process.kill()
        server.process.wait(timeout=5)
    return code


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "lc-home"
    path.mkdir()
    return path


@pytest.fixture
def server(home: Path) -> Iterator[Server]:
    instance = spawn(home)
    try:
        yield instance
    finally:
        stop(instance)


# --------------------------------------------------------------------------- #
# Startup contract
# --------------------------------------------------------------------------- #


def test_ready_line_and_server_info(server: Server) -> None:
    assert server.port > 0
    info = server.info()
    assert info["port"] == server.port
    assert info["token"] == server.token
    assert info["pid"] == server.process.pid
    assert isinstance(info["version"], str) and info["version"]
    assert isinstance(info["startedAt"], int) and info["startedAt"] > 1_600_000_000_000
    assert info["host"] == "127.0.0.1"
    mode = stat.S_IMODE(server.info_path.stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_health_and_auth_over_http(server: Server) -> None:
    status, body = server.get("/api/health", token=None)
    assert status == 200 and body["ok"] is True

    status, body = server.get("/api/library", token=None)
    assert status == 401 and body["code"] == "unauthorized"

    status, body = server.get("/api/library")
    assert status == 200 and isinstance(body, list)

    status, _ = server.get("/api/library", token="zly-token")
    assert status == 401


def test_log_file_is_written(server: Server) -> None:
    log = server.home / "logs" / "server.log"
    for _ in range(50):
        if log.is_file() and log.stat().st_size > 0:
            break
        time.sleep(0.1)
    assert log.is_file() and "nasłuchuje" in log.read_text(encoding="utf-8")


def test_stdout_holds_only_the_ready_line(home: Path) -> None:
    instance = spawn(home)
    stop(instance)
    remaining = instance.process.stdout.read() if instance.process.stdout else ""
    assert remaining.strip() == "", remaining


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #


def test_sigterm_shuts_down_and_removes_server_info(home: Path) -> None:
    instance = spawn(home)
    assert instance.info_path.is_file()
    started = time.monotonic()
    code = stop(instance, signal.SIGTERM, timeout=12.0)
    assert code == 0
    assert time.monotonic() - started < 12.0
    assert not instance.info_path.exists()


def test_sigint_shuts_down(home: Path) -> None:
    instance = spawn(home)
    assert stop(instance, signal.SIGINT, timeout=12.0) == 0


def test_server_info_of_another_pid_is_kept(home: Path) -> None:
    instance = spawn(home)
    foreign = {"port": 1, "token": "x", "pid": 999_999, "version": "0", "startedAt": 1}
    instance.info_path.write_text(json.dumps(foreign))
    stop(instance)
    assert json.loads(instance.info_path.read_text()) == foreign


def test_exits_when_parent_pid_dies(home: Path) -> None:
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        instance = spawn(home, "--parent-pid", str(parent.pid))
    except Exception:
        parent.kill()
        raise
    try:
        parent.terminate()
        parent.wait(timeout=10)
        assert instance.wait(timeout=15) == 0
        assert not instance.info_path.exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        stop(instance)


def test_two_servers_use_different_ports(home: Path, tmp_path: Path) -> None:
    other = tmp_path / "other-home"
    other.mkdir()
    first = spawn(home)
    second = spawn(other)
    try:
        assert first.port != second.port
        assert first.get("/api/health", token=None)[0] == 200
        assert second.get("/api/health", token=None)[0] == 200
    finally:
        stop(first)
        stop(second)


# --------------------------------------------------------------------------- #
# No orphaned engines
# --------------------------------------------------------------------------- #


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@pytest.mark.real_model
def test_sigterm_leaves_no_engine_process(home: Path) -> None:
    if not (shutil.which("llama-server") or Path("/opt/homebrew/bin/llama-server").is_file()):
        pytest.skip("llama-server not installed")
    model_path("gguf_tiny")
    models = str(Path("~/.localcode/models").expanduser())
    (home / "settings.json").write_text(json.dumps({
        "models_dir": models, "scan_lmstudio": False, "scan_hf_cache": False,
        "runtime": {"max_loaded_models": 1, "default_ctx": 1024, "idle_ttl_minutes": 0},
    }))
    instance = spawn(home)
    engine_pid: int | None = None
    try:
        status, _ = instance.post("/api/runtime/load", {"modelKey": GGUF_TINY_KEY,
                                                        "config": {"nCtx": 1024}})
        assert status == 200
        for _ in range(120):
            time.sleep(0.5)
            code, runtime = instance.get("/api/runtime")
            assert code == 200
            instances = runtime["instances"]
            if instances and instances[0]["state"] in ("ready", "error"):
                assert instances[0]["state"] == "ready", instances[0].get("error")
                engine_pid = instances[0]["pid"]
                break
        assert engine_pid and _alive(engine_pid)
    finally:
        assert stop(instance, signal.SIGTERM, timeout=15.0) == 0

    for _ in range(40):
        if not _alive(engine_pid or 0):
            break
        time.sleep(0.25)
    assert engine_pid is not None and not _alive(engine_pid), "llama-server został osierocony"
