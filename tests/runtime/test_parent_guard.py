"""No engine subprocess may survive a `kill -9` of the process that spawned it.

``mlx_worker`` and the image workers watch ``--parent-pid`` themselves;
``llama-server`` is a third-party binary with no such flag, so ``HttpEngine``
spawns it under the supervisor shim in ``http_engine._PARENT_GUARD_SRC``.
These tests need no model: they exercise the shim with ``/bin/sleep``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from localcode.runtime.http_engine import HttpEngine, _guarded_command
from localcode.runtime.llama_engine import LlamaServerEngine
from localcode.runtime.mlx_engine import MlxEngine


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float = 8.0) -> float | None:
    """Seconds until ``pid`` disappeared, or None if it is still alive."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if not _alive(pid):
            return time.monotonic() - t0
        time.sleep(0.1)
    return None


def test_guard_is_enabled_only_where_the_child_cannot_watch_itself() -> None:
    assert LlamaServerEngine.parent_guard is True      # third-party binary
    assert MlxEngine.parent_guard is False             # has --parent-pid
    assert HttpEngine.parent_guard is False            # opt-in


def test_guarded_command_prefixes_the_supervisor() -> None:
    cmd = _guarded_command(["/opt/homebrew/bin/llama-server", "-m", "x.gguf"])
    assert cmd[0] == sys.executable
    # -I keeps the supervisor to the standard library: no working directory on
    # sys.path (which the sandbox denies in a development install), no user site
    assert cmd[1] == "-I"
    assert cmd[2] == "-c"
    assert "getppid" in cmd[3]
    assert cmd[4:] == ["/opt/homebrew/bin/llama-server", "-m", "x.gguf"]


def test_guard_kills_the_binary_when_its_spawner_is_sigkilled(tmp_path) -> None:
    """The reproduction of the orphan bug, without a model: a python process
    spawns `sleep` exactly the way HttpEngine spawns llama-server (own session,
    so no signal reaches it), then dies with SIGKILL."""
    spawner = tmp_path / "spawner.py"
    spawner.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from localcode.runtime.http_engine import _guarded_command
        p = subprocess.Popen(_guarded_command(["/bin/sleep", "300"]), start_new_session=True)
        print(p.pid, flush=True)
        time.sleep(300)
    """))
    parent = subprocess.Popen([sys.executable, str(spawner)], stdout=subprocess.PIPE, text=True)
    try:
        shim_pid = int(parent.stdout.readline().strip())
        # The shim must really be running the binary as its own child.
        deadline = time.monotonic() + 5
        children: list[int] = []
        while time.monotonic() < deadline and not children:
            out = subprocess.run(["pgrep", "-P", str(shim_pid)], capture_output=True, text=True).stdout
            children = [int(x) for x in out.split()]
            time.sleep(0.1)
        assert children, "the supervisor did not spawn the binary"
        sleep_pid = children[0]

        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=5)

        assert _wait_gone(shim_pid) is not None, "the supervisor outlived its parent"
        assert _wait_gone(sleep_pid) is not None, "the binary was orphaned"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


def test_guard_forwards_sigterm_and_returns_the_child_exit_code() -> None:
    """``stop()`` signals the whole group; the shim must not swallow that, and a
    failing binary must still surface its exit code to ``_on_start_failure``."""
    proc = subprocess.Popen(_guarded_command(["/bin/sleep", "300"]), start_new_session=True)
    try:
        time.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) != 0          # 128 + SIGTERM from the child
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    failing = subprocess.Popen(_guarded_command(["/bin/sh", "-c", "exit 7"]))
    assert failing.wait(timeout=10) == 7


@pytest.mark.real_model
def test_real_llama_server_dies_with_its_spawner(tmp_path) -> None:
    """The original bug end to end: a core process starts a real llama-server,
    the core is SIGKILL-ed, and nothing may be left running."""
    from lc_testing import model_path

    gguf = model_path("gguf_tiny")
    if not gguf.exists():
        pytest.skip("tiny GGUF model missing")
    spawner = tmp_path / "core.py"
    spawner.write_text(textwrap.dedent(f"""
        import asyncio, os, sys
        sys.path.insert(0, {os.getcwd()!r})
        from localcode.config import Settings
        from localcode.hub.types import LocalModel
        from localcode.runtime.base import LoadConfig
        from localcode.runtime.llama_engine import LlamaServerEngine

        async def main():
            p = {str(gguf)!r}
            m = LocalModel(key="tiny", format="gguf", path=p, display_name="tiny",
                           size_bytes=os.path.getsize(p))
            e = LlamaServerEngine("guard-1", m, LoadConfig(n_ctx=512, n_parallel=1),
                                  settings=Settings())
            await e.start()
            print(e.pid, flush=True)
            await asyncio.sleep(300)

        asyncio.run(main())
    """))
    env = dict(os.environ, LOCALCODE_HOME=str(tmp_path / "home"))
    core = subprocess.Popen([sys.executable, str(spawner)], stdout=subprocess.PIPE, text=True, env=env)
    server_pid = None
    try:
        shim_pid = int(core.stdout.readline().strip())
        out = subprocess.run(["pgrep", "-P", str(shim_pid)], capture_output=True, text=True).stdout
        server_pid = int(out.split()[0])
        core.send_signal(signal.SIGKILL)
        core.wait(timeout=5)
        assert _wait_gone(shim_pid) is not None
        assert _wait_gone(server_pid) is not None, "llama-server was orphaned"
    finally:
        if core.poll() is None:
            core.kill()
            core.wait(timeout=5)
        for pid in (server_pid,):
            if pid and _alive(pid):
                os.kill(pid, signal.SIGKILL)
