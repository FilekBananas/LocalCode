"""``localcode serve`` — the server process.

    localcode serve [--host 127.0.0.1] [--port P|0] [--token T] [--parent-pid PID]

Behaviour the Mac app and the TUI rely on:

* ``--port 0`` lets the OS pick a free port; the chosen port is printed as the
  single stdout line ``LOCALCODE_READY port=<p>`` (flushed) the moment the
  socket accepts connections, and nothing else is ever written to stdout;
* the bearer token defaults to ``secrets.token_urlsafe(32)``;
* ``~/.localcode/server.json`` (mode 0600) holds
  ``{port, token, pid, version, startedAt, host}`` while the server runs and is
  removed on exit when it still describes this process;
* ``--parent-pid`` makes the server exit as soon as that process is gone, so
  closing the app never leaves a server (or a model) running;
* ``SIGTERM``/``SIGINT`` start a graceful shutdown that stops every engine
  subprocess (llama-server / mlx_worker) within ``SHUTDOWN_TIMEOUT_S``;
* everything is logged to ``~/.localcode/logs/server.log``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import logging.handlers
import os
import secrets
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Sequence

from aiohttp import web

from localcode import config as lc_config
from localcode.config import VERSION
from localcode.server.app import create_app
from localcode.server.core import Core, now_ms

log = logging.getLogger("localcode.server")

__all__ = ["serve_main", "build_parser", "run_server"]

#: Everything (HTTP shutdown + engines) must be gone within this many seconds.
SHUTDOWN_TIMEOUT_S = 10.0
#: How often the parent process is checked.
PARENT_POLL_S = 1.0
READY_PREFIX = "LOCALCODE_READY port="
LOG_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 3


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode serve",
                                     description="Serwer HTTP LocalCode (API dla aplikacji macOS i TUI).")
    parser.add_argument("--host", default=None, help="adres nasłuchu (domyślnie 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port (0 = wybierany przez system)")
    parser.add_argument("--token", default=None, help="token dostępu (domyślnie losowy)")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="zakończ, gdy ten proces przestanie istnieć")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="poziom logowania")
    parser.add_argument("--verbose", action="store_true", help="dubluj logi na stderr")
    return parser


def setup_logging(level: str = "INFO", *, verbose: bool = False) -> Path:
    """Log to ``logs/server.log``; stdout stays reserved for the READY line."""
    logs = lc_config.logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "server.log"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS,
                                                   encoding="utf-8")
    handler.setFormatter(formatter)
    root.handlers = [handler]
    if verbose:
        stderr = logging.StreamHandler(sys.stderr)
        stderr.setFormatter(formatter)
        root.addHandler(stderr)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    return path


# --------------------------------------------------------------------------- #
# server.json
# --------------------------------------------------------------------------- #


def write_server_info(port: int, token: str, host: str, *, path: Path | None = None) -> Path:
    """Write ``server.json`` with mode 0600 (it holds the access token)."""
    target = path or lc_config.server_info_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "port": port,
        "token": token,
        "pid": os.getpid(),
        "version": VERSION,
        "startedAt": now_ms(),
        "host": host,
    }
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, target)
    with contextlib.suppress(OSError):
        os.chmod(target, 0o600)
    return target


def remove_server_info(path: Path | None = None) -> None:
    """Remove ``server.json`` — but only when it still describes this process."""
    target = path or lc_config.server_info_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and data.get("pid") == os.getpid():
        with contextlib.suppress(OSError):
            target.unlink()


# --------------------------------------------------------------------------- #
# Runtime plumbing
# --------------------------------------------------------------------------- #


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


async def _watch_parent(pid: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if not _pid_alive(pid):
            log.info("proces nadrzędny %s zniknął — zamykam serwer", pid)
            stop.set()
            return
        with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
            await asyncio.wait_for(stop.wait(), PARENT_POLL_S)


def _install_signals(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        with contextlib.suppress(NotImplementedError, ValueError, AttributeError):
            loop.add_signal_handler(sig, _on_signal, sig, stop)


def _on_signal(sig: signal.Signals, stop: asyncio.Event) -> None:
    log.info("odebrano sygnał %s — zamykam serwer", getattr(sig, "name", sig))
    stop.set()


def _bound_port(runner: web.AppRunner, fallback: int) -> int:
    for address in runner.addresses:
        if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[1], int):
            return int(address[1])
    return fallback


async def run_server(
    host: str,
    port: int,
    token: str,
    *,
    parent_pid: int | None = None,
    core: Core | None = None,
    info_path: Path | None = None,
    ready: Any = None,
) -> int:
    """Serve until a signal or the parent's death; always shuts ``core`` down."""
    core = core or Core()
    stop = asyncio.Event()
    _install_signals(stop)
    await core.start()
    app = create_app(core, token)
    runner = web.AppRunner(app, handle_signals=False)
    await runner.setup()
    site = web.TCPSite(runner, host, port, reuse_address=True, shutdown_timeout=SHUTDOWN_TIMEOUT_S / 2)
    try:
        await site.start()
    except OSError as exc:
        await runner.cleanup()
        await core.shutdown()
        print(f"Nie udało się nasłuchiwać na {host}:{port}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    bound = _bound_port(runner, port)
    written = write_server_info(bound, token, host, path=info_path)
    log.info("serwer nasłuchuje na %s:%s (pid %s), info: %s", host, bound, os.getpid(), written)
    watcher: asyncio.Task[None] | None = None
    if parent_pid:
        watcher = asyncio.create_task(_watch_parent(parent_pid, stop), name="parent-watch")
    if ready is None:
        sys.stdout.write(f"{READY_PREFIX}{bound}\n")
        sys.stdout.flush()
    else:
        ready(bound)
    try:
        await stop.wait()
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(BaseException):
                await watcher
        await _shutdown(runner, core)
        remove_server_info(info_path)
    log.info("serwer zatrzymany")
    return 0


async def _shutdown(runner: web.AppRunner, core: Core | None) -> None:
    """Close the HTTP server and stop every engine within ``SHUTDOWN_TIMEOUT_S``."""
    async def close() -> None:
        with contextlib.suppress(Exception):
            await runner.cleanup()
        if core is not None:
            with contextlib.suppress(Exception):
                await core.shutdown()

    try:
        await asyncio.wait_for(close(), SHUTDOWN_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        log.error("zamykanie przekroczyło %.0f s — kończę proces", SHUTDOWN_TIMEOUT_S)


def free_port(host: str = "127.0.0.1") -> int:
    """An unused TCP port (used by tests and by ``--port 0`` callers)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def serve_main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``localcode serve``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "serve":
        args = args[1:]
    options = build_parser().parse_args(args)

    # Started from the Mac app, the server has macOS's minimal PATH and cannot
    # find ffmpeg (and friends) by name — see localcode/system_path.py.
    from localcode.system_path import ensure_tool_path

    ensure_tool_path()

    settings = lc_config.load_settings()
    host = options.host or settings.server.host or "127.0.0.1"
    port = settings.server.port if options.port is None else options.port
    token = options.token or os.environ.get("LOCALCODE_TOKEN") or secrets.token_urlsafe(32)

    setup_logging(options.log_level, verbose=options.verbose)
    core = Core(settings)
    try:
        return asyncio.run(run_server(host, int(port), token, parent_pid=options.parent_pid, core=core))
    except KeyboardInterrupt:  # pragma: no cover - asyncio.run handles SIGINT for us
        return 0
