"""``bash``: run shell commands the way a headless agent has to run them.

Every design decision here comes from one fact: nobody is sitting at a terminal.
The command gets no TTY, no stdin and no pager, it must not be able to hang the
agent, and a command that is stopped must not leave a runaway process behind.

* **Isolation.** ``/bin/bash -c <command>`` runs in its own session/process group
  (``start_new_session=True``) with ``stdin`` connected to ``/dev/null`` and
  ``PAGER=cat GIT_PAGER=cat TERM=dumb CI=1 NO_COLOR=1`` (plus
  ``GIT_TERMINAL_PROMPT=0`` so git never waits for credentials and
  ``PYTHONUNBUFFERED=1`` so progress actually streams). Because the whole tree
  shares one process group, ``killpg`` on a timeout — or when the agent turn is
  cancelled — reaps children, grandchildren and background jobs, first with
  ``SIGTERM``, then with ``SIGKILL``.
* **Confined to the project.** With ``agent.confine_to_project`` (the
  default) the command runs inside macOS' sandbox
  (:func:`localcode.security.sandbox.shell_profile`): it cannot read the rest
  of the home directory or write anywhere but the project, temporary files and
  the build caches. A command that *names* a place outside the project
  (``cat ~/.zshrc``, ``ls ..``) — or asks for it with ``outside_project`` —
  is flagged ``⚠ poza projektem``, which asks the user in every mode, and once
  allowed runs without the sandbox: the user saw exactly what it touches.
* **Interactive commands are refused, not attempted.** ``vim``, ``less``,
  ``top``, ``git commit`` without ``-m``, a bare ``python`` REPL … would either
  hang or produce garbage; :func:`interactive_reason` explains to the model what
  to run instead. The check is conservative: it only looks at the command name
  of every simple command in the line (see ``permissions.parse_shell``).
* **Bounded output.** stdout and stderr are merged (so the order the user would
  see is preserved), captured through a head+tail buffer that can never grow
  past :data:`MAX_CAPTURE_BYTES`, and finally truncated to
  ``ctx.limits.max_output_chars`` with an explicit marker. The model always sees
  the exit code, even when the output was cut.
* **Progress.** Complete output lines are forwarded to ``ctx.progress`` while the
  command runs (first :data:`PROGRESS_BURST` lines immediately, then at most one
  line per :data:`PROGRESS_INTERVAL_S`), so the UI shows a long build moving.
* **Sticky working directory.** A ``cd`` survives to the next call: the command
  is followed by a probe that prints ``$PWD`` behind a random marker, which is
  stripped from the output and remembered in ``ctx.state['bash_cwd']``. The probe
  keeps the command's own exit status, and is skipped for command lines it could
  not be appended to safely (a trailing backslash continuation).

A process the command deliberately left running in the background keeps running:
once the shell itself is gone the tool closes its end of the pipe and returns
instead of waiting for that child.

Public API: :class:`BashTool`, :func:`run_bash` (the reusable runner),
:func:`command_from_args` (used by ``agent/permissions.py`` to recover the exact
command from the model's arguments), :func:`interactive_reason`,
:func:`timeout_from_args`, :func:`build_env`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from localcode.agent.permissions import (
    DANGER_FLAG,
    OUTSIDE_PROJECT_FLAG,
    _has_flag,
    _positionals,
    _strip_wrappers,
    is_catastrophic,
    outside_paths,
    parse_shell,
)
from localcode.agent.tools.base import PermissionInfo, ToolContext, ToolResult
from localcode.agent.tools.fs import (
    CodingTool,
    ToolInputError,
    arg_bool,
    arg_int,
    arg_str,
    cwd_of,
    display_path,
    human_size,
    normalize_args,
    short,
    truncate_middle,
)

__all__ = [
    "BASH_ENV",
    "BASH_PATH",
    "COMMAND_ALIASES",
    "MAX_CAPTURE_BYTES",
    "MAX_TIMEOUT_S",
    "STATE_CWD_KEY",
    "CONFINE_STATE_KEY",
    "BashTool",
    "ShellOutcome",
    "build_env",
    "command_from_args",
    "interactive_reason",
    "run_bash",
    "timeout_from_args",
]

BASH_PATH = "/bin/bash"

MIN_TIMEOUT_S = 1
MAX_TIMEOUT_S = 600              # 10 minutes is already far more than an agent should wait
MS_THRESHOLD = 3600              # a "timeout" bigger than an hour is read as milliseconds

TERM_GRACE_S = 2.0               # SIGTERM → SIGKILL delay
KILL_WAIT_S = 3.0                # how long we wait for a killed process group
EOF_GRACE_S = 0.3                # after exit: how long a background child may hold the pipe
READ_CHUNK = 64 * 1024
MAX_CAPTURE_BYTES = 2 * 1024 * 1024   # hard cap on the bytes we keep in memory
MAX_PARTIAL_LINE = 64 * 1024          # flush a line this long even without a newline

PROGRESS_BURST = 40              # lines streamed without throttling
PROGRESS_INTERVAL_S = 0.2
PROGRESS_LINE_CHARS = 200

STATE_CWD_KEY = "bash_cwd"
#: ``ToolContext.state`` key: ``False`` when the user turned the folder sandbox off
#: (``agent.confine_to_project``); missing means on.
CONFINE_STATE_KEY = "confine_to_project"
#: Appended when the sandbox stopped a command, so the model knows why and what to do.
CONFINED_HINT = ("[LocalCode] Commands run confined to the project folder: files outside it "
                 "cannot be read and nothing outside it can be written (only temporary files "
                 "and build caches). If the task really needs something outside, run the "
                 "command again with outside_project=true — the user will be asked first.")
_CWD_MARKER_PREFIX = "__LC_CWD_"

# Environment forced on every command (merged over the parent environment).
BASH_ENV: dict[str, str] = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "TERM": "dumb",
    "CI": "1",
    "NO_COLOR": "1",
    "GIT_TERMINAL_PROMPT": "0",   # git fails instead of asking for a password
    "PYTHONUNBUFFERED": "1",      # so python output streams as progress
}


def build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment a command runs in: the parent's, with :data:`BASH_ENV` forced."""
    env = dict(os.environ if base is None else base)
    env.update(BASH_ENV)
    for stale in ("PWD", "OLDPWD"):     # ours, not the command's; bash would rewrite them anyway
        env.pop(stale, None)
    return env


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #

COMMAND_ALIASES: dict[str, str] = {
    "cmd": "command", "shell": "command", "shell_command": "command", "bash": "command",
    "bash_command": "command", "script": "command", "command_line": "command", "cmdline": "command",
    "commandline": "command", "run": "command", "exec": "command", "input": "command",
    "code": "command", "commands": "command", "line": "command", "query": "command",
    "outside": "outside_project", "outside_folder": "outside_project",
    "unsandboxed": "outside_project", "no_sandbox": "outside_project",
    "timeout_s": "timeout", "timeout_sec": "timeout", "timeout_secs": "timeout",
    "timeout_seconds": "timeout", "timeout_ms": "timeout", "timeoutms": "timeout",
    "time_limit": "timeout", "max_time": "timeout", "seconds": "timeout", "deadline": "timeout",
}


def command_from_args(args: Any) -> str:
    """The exact command line :class:`BashTool` would run for ``args``.

    ``agent/permissions.py`` calls this so that the read-only / catastrophic
    analysis sees the same string as the shell (it accepts the same forgiving
    argument shapes as the tool itself: a bare string, ``{"cmd": …}``, …).
    """
    try:
        normalized = normalize_args(args, COMMAND_ALIASES, "command")
    except ToolInputError:
        return ""
    value = normalized.get("command")
    if value is None:
        return ""
    if isinstance(value, list):
        # Exactly what ``arg_str`` (and therefore ``execute``) makes of a list:
        # one command per line. Joining with spaces would hide a second command
        # from the read-only / catastrophic analysis.
        return "\n".join(str(v) for v in value).strip()
    return str(value).strip()


def timeout_from_args(args: dict[str, Any], default: int) -> int:
    """Timeout in seconds, clamped; values over an hour are read as milliseconds."""
    raw = arg_int(args, "timeout")
    if raw is None or raw <= 0:
        value = default
    elif raw > MS_THRESHOLD:
        value = round(raw / 1000)
    else:
        value = raw
    return max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, int(value)))


# --------------------------------------------------------------------------- #
# Interactive commands
# --------------------------------------------------------------------------- #

_EDITOR_FIX = "Use read_file, edit_file or write_file to change files."
_PAGER_FIX = "Use read_file, or print the file with cat/head/tail."
_MONITOR_FIX = "Use a one-shot snapshot instead, e.g. 'ps aux' or 'top -l 1'."
_REPL_FIX = "Run a script file, or pass the code with -c."
_DB_FIX = "Pass the statement as an argument, e.g. sqlite3 app.db 'select 1'."
_REMOTE_FIX = "Run the command locally, or pass it to the remote host as an argument."
_MUX_FIX = "Run the command directly instead."
_DEBUG_FIX = "Run the program directly and read the traceback, or add logging."

_GROUPS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("vim", "vi", "nvim", "view", "vimdiff", "emacs", "nano", "pico", "ed", "micro", "joe",
      "helix", "hx", "kak", "mcedit"), "an interactive editor", _EDITOR_FIX),
    (("less", "more", "most", "pg", "bat"), "an interactive pager", _PAGER_FIX),
    (("top", "htop", "btop", "atop", "glances", "iotop", "nmon", "vtop"),
     "a full-screen monitor that never exits", _MONITOR_FIX),
    (("python", "python2", "python3", "node", "irb", "ipython", "bpython", "ghci", "julia",
      "R", "scala", "erl", "iex", "clj", "lua", "swift", "perl", "ruby", "php", "deno", "bun"),
     "an interactive REPL when it gets no script", _REPL_FIX),
    (("mysql", "psql", "sqlite3", "sqlite", "mongo", "mongosh", "redis-cli", "sqlplus"),
     "an interactive database shell when it gets no statement", _DB_FIX),
    (("ssh", "sftp", "ftp", "telnet", "kubectl-exec"), "an interactive remote session", _REMOTE_FIX),
    (("screen", "tmux", "byobu", "zellij"), "a terminal multiplexer", _MUX_FIX),
    (("gdb", "lldb", "pdb", "cgdb"), "an interactive debugger", _DEBUG_FIX),
    (("crontab", "visudo", "passwd", "su", "login", "adduser", "useradd"),
     "an interactive system tool", "Do not run it from the agent."),
    (("watch", "entr"), "a command that repeats forever", "Run the command once instead."),
    (("fzf", "dialog", "whiptail", "vipe"), "an interactive picker", "Choose the value yourself."),
)

# Names that only hang when nothing tells them what to run. Everything else on
# the lists above is refused whatever its arguments.
_REPL_NAMES = frozenset(_GROUPS[3][0])
_DB_NAMES = frozenset(_GROUPS[4][0])
_REMOTE_NAMES = frozenset(_GROUPS[5][0])
# only these read "-i" as "interactive"; for perl/ruby/sed-likes it means in-place
_INTERACTIVE_FLAG_REPLS = frozenset({"python", "python2", "python3", "ipython", "bpython",
                                     "node", "deno", "bun", "julia", "R"})
_SHELLS = frozenset({"sh", "bash", "zsh", "fish", "ksh", "dash", "csh", "tcsh"})
_SSH_VALUE_OPTS = ("-p", "-i", "-l", "-o", "-F", "-L", "-R", "-D", "-J", "-c", "-m", "-b",
                   "-E", "-Q", "-W", "-S", "-e")

_INTERACTIVE: dict[str, tuple[str, str]] = {
    name: (what, fix) for names, what, fix in _GROUPS for name in names
}


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _repl_is_interactive(name: str, args: list[str]) -> bool:
    """A REPL hangs only when nothing on the command line tells it what to execute.

    Deliberately permissive: any argument at all (a script, ``-c``, a subcommand)
    means the command does something, and a wrong guess here refuses a perfectly
    good command.
    """
    if not args:
        return True
    return name in _INTERACTIVE_FLAG_REPLS and _has_flag(args, "-i", "--interactive")


def _db_is_interactive(name: str, args: list[str]) -> bool:
    """A database client is a shell only without a statement, file or database."""
    if _has_flag(args, "-e", "--execute", "-c", "--command", "-f", "--file", "--eval"):
        return False
    positionals = _positionals(args, ("-u", "-h", "-P", "-p", "-D", "-d", "-U", "-cmd", "-init"))
    if name in ("sqlite3", "sqlite"):
        return len(positionals) < 2      # "sqlite3 app.db" alone opens the shell
    return not positionals


def _remote_is_interactive(name: str, args: list[str]) -> bool:
    """``ssh host command`` runs and exits; ``ssh host`` opens a shell."""
    if name != "ssh":
        return True
    return len(_positionals(args, _SSH_VALUE_OPTS)) < 2


def _git_subcommand(args: list[str]) -> tuple[str, list[str]]:
    """``["-C", "/tmp", "commit", "-m", "x"]`` → ``("commit", ["-m", "x"])``."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c", "--git-dir", "--work-tree", "--namespace",
                              "--exec-path", "--config-env") else 1
    return (args[i], args[i + 1:]) if i < len(args) else ("", [])


def _git_is_interactive(args: list[str]) -> str | None:
    sub, rest = _git_subcommand(args)
    if sub == "commit" and not _has_flag(rest, "-m", "--message", "-F", "--file", "-C",
                                         "--reuse-message", "--no-edit", "-t", "--template"):
        return ("'git commit' without -m opens an editor and would hang. "
                "Pass the message: git commit -m \"...\".")
    if sub in ("rebase", "add", "checkout", "restore", "stash", "reset") and \
            _has_flag(rest, "-i", "--interactive", "-p", "--patch"):
        return (f"'git {sub}' in interactive/patch mode needs a terminal. "
                "Run it without -i/-p, or make the change with edit_file.")
    if sub == "mergetool" or sub == "difftool":
        return f"'git {sub}' opens an interactive tool. Use 'git diff' and edit_file instead."
    return None


def interactive_reason(command: str) -> str | None:
    """English explanation when ``command`` would block waiting for a terminal.

    ``None`` means the command is safe to run headless. Only the command name of
    every simple command in the line is inspected, so text *inside* arguments
    (``grep vim file``) never triggers a refusal.
    """
    parsed = parse_shell(command)
    for simple in parsed.commands:
        if not simple.argv:
            continue
        if _basename(simple.argv[0]) == "command" and simple.argv[1:2] \
                and simple.argv[1] in ("-v", "-V"):
            continue                      # "command -v vim" only prints a path
        argv = _strip_wrappers(simple.argv)
        if not argv:
            continue
        name = _basename(argv[0])
        args = argv[1:]
        if name in _SHELLS and not args:
            return ("A bare shell would wait for input forever. Pass the command to run, "
                    "e.g. bash -c \"...\", or just run the command itself.")
        if name == "git":
            reason = _git_is_interactive(args)
            if reason:
                return reason
            continue
        if name == "tail" and _has_flag(args, "-f", "-F", "--follow"):
            return ("'tail -f' never ends. Use 'tail -n 200 <file>' for the last lines, "
                    "or grep the log file.")
        if name in ("docker", "podman") and args[:1] and args[0] in ("run", "exec", "attach") \
                and _has_flag(args[1:], "-i", "-it", "--interactive"):
            return (f"'{name} {args[0]} -i' attaches an interactive terminal. "
                    f"Drop -i/-it and pass the command to run.")
        if name in _REPL_NAMES and not _repl_is_interactive(name, args):
            continue
        if name in _DB_NAMES and not _db_is_interactive(name, args):
            continue
        if name in _REMOTE_NAMES and not _remote_is_interactive(name, args):
            continue
        entry = _INTERACTIVE.get(name)
        if entry:
            what, fix = entry
            return f"'{name}' is {what}; with no terminal it would hang or print nothing. {fix}"
    return None


# --------------------------------------------------------------------------- #
# Output capture
# --------------------------------------------------------------------------- #


class _Capture:
    """Keeps the head and the tail of a byte stream; the middle is counted, not stored."""

    def __init__(self, keep: int = MAX_CAPTURE_BYTES) -> None:
        keep = max(2048, int(keep))
        self.head_limit = keep // 2
        self.tail_limit = keep - self.head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.head_limit - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail += chunk
            overflow = len(self._tail) - self.tail_limit
            if overflow > 0:
                del self._tail[:overflow]

    @property
    def dropped(self) -> int:
        return max(0, self.total - len(self._head) - len(self._tail))

    def text(self) -> str:
        head = self._head.decode("utf-8", "replace")
        if not self._tail:
            return head
        tail = self._tail.decode("utf-8", "replace")
        dropped = self.dropped
        if not dropped:
            return head + tail
        return f"{head}\n[… {human_size(dropped)} of output dropped …]\n{tail}"


async def _pump(reader: asyncio.StreamReader, capture: _Capture,
                progress: Callable[[str], Awaitable[None]] | None, marker: str | None) -> None:
    """Read the merged output to EOF, capturing it and streaming whole lines."""
    buffer = bytearray()
    sent = 0
    last = 0.0

    async def emit(raw: bytes) -> None:
        nonlocal sent, last
        if progress is None:
            return
        line = raw.decode("utf-8", "replace").strip()
        if not line or (marker and marker in line):
            return
        now = time.monotonic()
        if sent >= PROGRESS_BURST and now - last < PROGRESS_INTERVAL_S:
            return
        sent += 1
        last = now
        if len(line) > PROGRESS_LINE_CHARS:
            line = line[:PROGRESS_LINE_CHARS - 1] + "…"
        with contextlib.suppress(Exception):   # a failing UI must not kill the command
            await progress(line)

    while True:
        chunk = await reader.read(READ_CHUNK)
        if not chunk:
            break
        capture.feed(chunk)
        if progress is None:
            continue
        buffer += chunk
        while True:
            nl = buffer.find(b"\n")
            if nl < 0:
                break
            line = bytes(buffer[:nl])
            del buffer[:nl + 1]
            await emit(line)
        if len(buffer) > MAX_PARTIAL_LINE:
            await emit(bytes(buffer))
            buffer.clear()
    if buffer:
        await emit(bytes(buffer))


# --------------------------------------------------------------------------- #
# Process control
# --------------------------------------------------------------------------- #


def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
    """Signal the command's whole process group (never our own)."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None
    if pgid is None or pgid == os.getpgrp():
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.send_signal(sig)
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.send_signal(sig)


async def _wait_exit(proc: asyncio.subprocess.Process, interval: float = 0.02) -> int:
    """Wait for the *process* to exit.

    Deliberately not ``proc.wait()``: asyncio resolves that only once every pipe
    is closed too, so a background grandchild holding stdout would keep it
    pending long after the shell itself is gone. ``proc.returncode`` is set the
    moment the child is reaped, which is exactly what we need.
    """
    while proc.returncode is None:
        await asyncio.sleep(interval)
    return proc.returncode


def _close_pipes(proc: asyncio.subprocess.Process) -> None:
    """Release our end of the pipes (a leftover background writer gets EPIPE)."""
    transport = getattr(proc, "_transport", None)
    if transport is None:                        # pragma: no cover - always present in CPython
        return
    for fd in (1, 2):
        try:
            pipe = transport.get_pipe_transport(fd)
        except Exception:                        # pragma: no cover - defensive
            continue
        if pipe is not None:
            with contextlib.suppress(Exception):
                pipe.close()


async def _terminate(proc: asyncio.subprocess.Process, *, graceful: bool) -> None:
    """Kill the command and everything it started. Safe to await while cancelled."""
    if proc.returncode is not None:
        return
    if graceful:
        _signal_group(proc, signal.SIGTERM)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(_wait_exit(proc), TERM_GRACE_S)
    if proc.returncode is None:
        _signal_group(proc, signal.SIGKILL)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(_wait_exit(proc), KILL_WAIT_S)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


@dataclass
class ShellOutcome:
    """Everything the tool (or another caller) needs to report one command."""

    command: str
    exit_code: int | None        # None when it was killed before reporting a status
    output: str                  # merged stdout+stderr, marker stripped, not yet clamped
    duration_s: float
    timed_out: bool
    timeout_s: int
    cwd: Path
    new_cwd: Path | None = None  # set when the command left the shell in another directory
    dropped_bytes: int = 0


#: Printed when a job started with ``&`` is already dead a second later.
BACKGROUND_DIED = ("[LocalCode] The background job exited at once, so it most likely failed. "
                   "Read its log, or run the command without & to see the error.")


def starts_background_job(command: str) -> bool:
    """``… &`` (a job left running), not ``a && b``."""
    tail = command.rstrip().rstrip(";").rstrip()
    return tail.endswith("&") and not tail.endswith("&&")


def _background_check() -> str:
    """Lines that run right after a ``… &`` command.

    ``&`` succeeds the moment the job is forked, so a server that crashed on its
    first line still looked started. A real session: the model ran
    ``nohup python3 -m http.server8000 … &`` (a dropped space), got exit 0, and
    told the user the page was up — localhost answered nothing. One second is
    enough for a typo, a missing module or a busy port to kill the job.
    """
    return ("__lc_bg=$!\n"
            "sleep 1\n"
            f"kill -0 \"$__lc_bg\" 2>/dev/null || printf '\\n%s\\n' '{BACKGROUND_DIED}'\n")


def _with_cwd_probe(command: str) -> tuple[str, str | None]:
    """``command`` plus a probe printing ``$PWD``; ``(command, None)`` when unsafe."""
    background = _background_check() if starts_background_job(command) else ""
    if command.rstrip("\n").endswith("\\"):
        return command, None          # a trailing line continuation would swallow the probe
    marker = f"{_CWD_MARKER_PREFIX}{uuid.uuid4().hex}"
    script = (f"{command}\n__lc_rc=$?\n"
              f"{background}"
              f"printf '\\n{marker}%s\\n' \"$PWD\"\n"
              f"exit $__lc_rc\n")
    return script, marker


def _split_marker(text: str, marker: str | None) -> tuple[str, str | None]:
    """Strip the trailing ``\\n<marker><pwd>\\n`` probe line; returns (output, pwd)."""
    if not marker:
        return text, None
    index = text.rfind("\n" + marker)
    if index < 0:
        if text.startswith(marker):       # the command produced no output at all
            return "", text[len(marker):].strip("\n") or None
        return text, None
    pwd = text[index + 1 + len(marker):].strip("\n")
    return text[:index], (pwd or None)


async def run_bash(command: str, *, cwd: Path | str, timeout_s: int,
                   progress: Callable[[str], Awaitable[None]] | None = None,
                   env: dict[str, str] | None = None,
                   keep_bytes: int = MAX_CAPTURE_BYTES,
                   track_cwd: bool = True,
                   confine_to: Path | str | None = None) -> ShellOutcome:
    """Run ``command`` with bash and return what happened.

    ``confine_to`` runs it inside the folder sandbox of that project
    (:func:`localcode.security.sandbox.wrap_shell`).

    The command always runs in its own process group; on timeout and on
    :class:`asyncio.CancelledError` the whole group is killed (``SIGTERM`` then
    ``SIGKILL``) before this coroutine returns or re-raises.
    """
    if track_cwd:
        script, marker = _with_cwd_probe(command)
    elif starts_background_job(command):
        script, marker = f"{command}\n__lc_rc=$?\n{_background_check()}exit $__lc_rc\n", None
    else:
        script, marker = command, None
    argv = [BASH_PATH, "-c", script]
    if confine_to is not None:
        from localcode.security.sandbox import wrap_shell

        argv = wrap_shell(argv, confine_to)
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=build_env(env),
        start_new_session=True,
        limit=READ_CHUNK * 4,
    )
    assert proc.stdout is not None
    capture = _Capture(keep_bytes)
    pump: asyncio.Task[None] = asyncio.create_task(_pump(proc.stdout, capture, progress, marker))
    exited: asyncio.Task[int] = asyncio.create_task(_wait_exit(proc))
    timed_out = False
    graceful = True
    try:
        done, _pending = await asyncio.wait({pump, exited}, timeout=timeout_s,
                                           return_when=asyncio.FIRST_COMPLETED)
        if not done:
            timed_out = True
            # kill first, then let the reader drain whatever the command managed to
            # print before it died — that tail is usually the interesting part
            await _terminate(proc, graceful=True)
            with contextlib.suppress(BaseException):
                await asyncio.wait({pump}, timeout=EOF_GRACE_S)
        elif exited in done and pump not in done:
            # the shell is gone but a background child still holds the pipe open
            await asyncio.wait({pump}, timeout=EOF_GRACE_S)
        elif pump in done and exited not in done:
            remaining = max(0.0, timeout_s - (time.monotonic() - started))
            finished, _ = await asyncio.wait({exited}, timeout=remaining)
            timed_out = not finished
    except asyncio.CancelledError:
        graceful = False
        raise
    finally:
        for task in (pump, exited):
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        await _terminate(proc, graceful=graceful)
        _close_pipes(proc)
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(proc.wait()), 1.0)

    duration = time.monotonic() - started
    text, pwd = _split_marker(capture.text(), marker)
    new_cwd: Path | None = None
    if pwd:
        candidate = Path(pwd)
        if candidate.is_dir() and os.path.realpath(candidate) != os.path.realpath(str(cwd)):
            new_cwd = candidate
    return ShellOutcome(
        command=command,
        exit_code=proc.returncode if not timed_out else None,
        output=text,
        duration_s=duration,
        timed_out=timed_out,
        timeout_s=timeout_s,
        cwd=Path(str(cwd)),
        new_cwd=new_cwd,
        dropped_bytes=capture.dropped,
    )


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

_STATUS_RESERVE = 240            # characters kept free for the status lines


class BashTool(CodingTool):
    name = "bash"
    kind = "exec"
    description = ("Run a shell command in the project directory (tests, git, build tools). "
                   "It cannot touch files outside the project. "
                   "Use read_file/edit_file for files, not cat/sed.")
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command line for bash"},
            "description": {"type": "string",
                            "description": "What it does and why it is needed, in a few words, in the user's language"},
            "timeout": {"type": "integer", "description": "Seconds before it is killed (default 120)"},
            "outside_project": {"type": "boolean",
                                "description": "Only when it must reach outside the project "
                                               "(asks the user)"},
        },
        "required": ["command"],
    }
    aliases = COMMAND_ALIASES
    primary = "command"
    usage = 'bash({"command": "npm test", "timeout": 120})'

    def make_title(self, args: dict[str, Any]) -> str:
        """``Polecenie: <what it does>`` when the model said so, else the command itself."""
        does = _description(args)
        command = arg_str(args, "command") or "?"
        return f"Polecenie: {short(does or command, 60)}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        try:
            normalized = self.args(args)
        except ToolInputError:
            return None                      # run() reports the malformed call
        command = (arg_str(normalized, "command") or "").strip()
        if not command:
            return None
        # The flags are judged on the command, never on the model's words
        # about it; the policy reads the command from the arguments too.
        title = self.make_title(normalized)
        if is_catastrophic(command, cwd_of(ctx)):
            title += f" {DANGER_FLAG}"
        does = _description(normalized)
        preview = f"Co robi: {does}\n\n{command}" if does else command
        reach = self._outside(normalized, command, ctx)
        if reach is not None:
            title += f" {OUTSIDE_PROJECT_FLAG}"
            where = ", ".join(reach) if reach else "pliki poza projektem"
            preview = (f"{preview}\n\nTo polecenie sięga poza folder projektu ({where}). "
                       "Po zgodzie uruchomi się bez ograniczenia do folderu.")
        return PermissionInfo(title=title, preview=preview)

    @staticmethod
    def _confined(ctx: ToolContext) -> bool:
        return ctx.state.get(CONFINE_STATE_KEY, True) is not False

    def _outside(self, args: dict[str, Any], command: str, ctx: ToolContext) -> list[str] | None:
        """What outside the project this call reaches, or ``None`` when it stays inside.

        An empty list means the model asked for it (``outside_project``)
        without naming a path. Always ``None`` with the folder sandbox off:
        then nothing is confined and nothing needs the extra question.
        """
        if not self._confined(ctx):
            return None
        try:
            cwd = self._cwd(ctx)
        except ToolInputError:
            cwd = cwd_of(ctx)
        reach = outside_paths(command, cwd_of(ctx), cwd)
        if reach:
            return reach
        return [] if arg_bool(args, "outside_project") else None

    # -------------------------------------------------------------- execution

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not str(args.get("command") or "").strip():
            raise ToolInputError("Missing required argument 'command' (the shell command to run).")
        command = (arg_str(args, "command", required=True, allow_empty=False) or "").strip()
        timeout_s = timeout_from_args(args, int(ctx.limits.bash_timeout_s))
        if not os.path.exists(BASH_PATH):     # pragma: no cover - macOS always has bash
            return ToolResult(ok=False, output=f"Error: {BASH_PATH} is missing on this machine.",
                              summary="Brak powłoki bash")
        reason = interactive_reason(command)
        if reason:
            return ToolResult(ok=False, output=f"Error: {reason}", summary="Polecenie interaktywne")
        cwd = self._cwd(ctx)
        # A call that reaches outside was flagged in permission(), and the
        # flag asks the user in every mode — so here it has been allowed.
        confined = self._confined(ctx) and self._outside(args, command, ctx) is None
        outcome = await run_bash(command, cwd=cwd, timeout_s=timeout_s, progress=ctx.progress,
                                 confine_to=cwd_of(ctx) if confined else None)
        if outcome.new_cwd is not None:
            ctx.state[STATE_CWD_KEY] = str(outcome.new_cwd)
        result = self._format(outcome, ctx)
        if confined and not result.ok and "Operation not permitted" in outcome.output:
            result.output += "\n" + CONFINED_HINT
        return result

    @staticmethod
    def _cwd(ctx: ToolContext) -> Path:
        """Where the command runs: the sticky directory when a previous ``cd`` set one."""
        root = cwd_of(ctx)
        sticky = ctx.state.get(STATE_CWD_KEY)
        if sticky:
            path = Path(str(sticky))
            if path.is_dir():
                return path
            ctx.state.pop(STATE_CWD_KEY, None)
        if not root.is_dir():
            raise ToolInputError(f"The project directory {root} does not exist.",
                                 summary="Brak katalogu projektu")
        return root

    def _format(self, outcome: ShellOutcome, ctx: ToolContext) -> ToolResult:
        limits = ctx.limits
        body = outcome.output.strip("\n")
        budget = max(400, int(limits.max_output_chars) - _STATUS_RESERVE)
        body = truncate_middle(
            body, budget,
            "output too long; re-run with fewer lines (grep, tail -n 50, --quiet)")
        parts: list[str] = [body if body.strip() else "(no output)"]

        if outcome.new_cwd is not None:
            parts.append(f"[cwd: {display_path(outcome.new_cwd, ctx)}]")
        if outcome.timed_out:
            parts.append(f"[timed out after {outcome.timeout_s} s and was killed with its child "
                         f"processes — no exit code]")
            summary = f"Przerwane po {outcome.timeout_s} s"
            ok = False
        elif outcome.exit_code is None:       # killed without reporting a status
            parts.append("[the command was stopped before it reported an exit code]")
            summary = "Przerwane"
            ok = False
        else:
            code = outcome.exit_code
            parts.append(f"[exit code {code}{_signal_note(code)}]")
            summary = f"Kod wyjścia {code}"
            ok = code == 0
            if outcome.duration_s >= 1.0:
                summary += f" · {outcome.duration_s:.1f} s"
        return ToolResult(ok=ok, output="\n".join(parts), summary=summary)


def _description(args: dict[str, Any]) -> str:
    """The model's few words on what a command does (``description``), one line, clipped.

    For the people reading the card only: nothing that decides anything reads
    it — the command does.
    """
    value = args.get("description")
    if not isinstance(value, str):
        return ""
    return short(value, 160).strip().rstrip(".") if value.strip() else ""


def _signal_note(code: int) -> str:
    """``" (killed by SIGKILL)"`` for a negative exit status, else an empty string."""
    if code >= 0:
        return ""
    try:
        return f" (killed by {signal.Signals(-code).name})"
    except ValueError:
        return " (killed by a signal)"
