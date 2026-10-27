"""Tests for the ``bash`` tool.

Real processes, real signals: the two properties that matter most — a timeout
kills the *whole* process tree, and so does cancelling the agent turn — cannot be
tested with a fake subprocess. Every test that starts a long-lived process
verifies in a ``finally`` block that nothing survived (``wait_until_gone``) and
kills whatever a failure may have leaked (``kill_pid``).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from localcode.agent.permissions import DANGER_FLAG
from localcode.agent.tools import get_tool
from localcode.agent.tools.base import ToolContext, ToolLimits
from localcode.agent.tools.shell import (
    BASH_ENV,
    MAX_TIMEOUT_S,
    build_env,
    command_from_args,
    interactive_reason,
    run_bash,
    timeout_from_args,
)
from tool_helpers import (
    Progress,
    kill_pid,
    make_ctx,
    permission_of,
    pid_alive,
    run_tool,
    wait_until_gone,
    write_file,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_file(tmp_path / "README.md", "# Demo\n")
    return tmp_path


async def bash(project: Path, args, ctx: ToolContext | None = None):
    return await run_tool("bash", args, ctx or make_ctx(project))


def child_pid(marker: Path, timeout: float = 5.0) -> int:
    """Read the pid a test command wrote, waiting briefly for it to appear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text().strip():
            return int(marker.read_text().strip())
        time.sleep(0.02)
    raise AssertionError(f"the command never wrote its pid to {marker}")


# --------------------------------------------------------------------------- #
# Exit codes and output
# --------------------------------------------------------------------------- #


async def test_successful_command(project: Path):
    result = await bash(project, {"command": "echo hello"})
    assert result.ok
    assert result.output == "hello\n[exit code 0]"
    assert result.summary == "Kod wyjścia 0"


async def test_failing_command_reports_the_exit_code(project: Path):
    result = await bash(project, {"command": "exit 7"})
    assert not result.ok
    assert "[exit code 7]" in result.output
    assert result.summary == "Kod wyjścia 7"


async def test_stdout_and_stderr_are_merged_in_order(project: Path):
    result = await bash(project, {"command": "echo out; echo err 1>&2; echo out2"})
    assert result.output.splitlines()[:3] == ["out", "err", "out2"]


async def test_command_without_output(project: Path):
    result = await bash(project, {"command": "true"})
    assert result.output == "(no output)\n[exit code 0]"


async def test_missing_binary_reports_bash_error_text(project: Path):
    result = await bash(project, {"command": "definitely-not-a-real-binary-xyz"})
    assert not result.ok
    assert "not found" in result.output.lower()
    assert "[exit code 127]" in result.output


async def test_signal_deaths_are_named(project: Path):
    result = await bash(project, {"command": "kill -TERM $$"})
    assert not result.ok
    assert "killed by SIGTERM" in result.output


async def test_runs_in_the_project_directory(project: Path):
    result = await bash(project, {"command": "pwd"})
    assert os.path.realpath(result.output.splitlines()[0]) == os.path.realpath(str(project))


async def test_stdin_is_closed_so_reading_it_does_not_hang(project: Path):
    result = await bash(project, {"command": "cat", "timeout": 5})
    assert result.ok
    assert "[exit code 0]" in result.output


async def test_environment_is_non_interactive(project: Path):
    result = await bash(project, {"command": "echo $PAGER/$GIT_PAGER/$TERM/$CI/$NO_COLOR"})
    assert result.output.splitlines()[0] == "cat/cat/dumb/1/1"


def test_build_env_forces_the_overrides():
    env = build_env({"TERM": "xterm-256color", "HOME": "/x"})
    assert env["TERM"] == "dumb"
    assert env["HOME"] == "/x"
    for key, value in BASH_ENV.items():
        assert env[key] == value


async def test_output_is_truncated_head_and_tail(project: Path):
    ctx = make_ctx(project, max_output_chars=1200)
    result = await bash(project, {"command": "for i in $(seq 1 400); do echo line$i; done"}, ctx)
    assert result.ok
    assert len(result.output) <= 1200
    assert "line1\n" in result.output
    assert "line400" in result.output
    assert "lines omitted" in result.output
    assert "[exit code 0]" in result.output          # the status always survives


async def test_huge_output_does_not_explode_memory(project: Path):
    ctx = make_ctx(project, max_output_chars=2000)
    result = await bash(
        project,
        {"command": "for i in $(seq 1 20000); do echo 'the quick brown fox jumps over'; done",
         "timeout": 60}, ctx)
    assert result.ok
    assert len(result.output) <= 2000


async def test_unicode_output(project: Path):
    result = await bash(project, {"command": "echo 'zażółć gęślą jaźń 🐍'"})
    assert "zażółć gęślą jaźń 🐍" in result.output


# --------------------------------------------------------------------------- #
# Working directory
# --------------------------------------------------------------------------- #


async def test_cd_sticks_to_the_next_call(project: Path):
    (project / "sub").mkdir()
    ctx = make_ctx(project)
    first = await bash(project, {"command": "cd sub && pwd"}, ctx)
    assert "[cwd: sub]" in first.output
    assert ctx.state["bash_cwd"] == str(project / "sub")
    second = await bash(project, {"command": "pwd"}, ctx)
    assert second.output.splitlines()[0].endswith("/sub")
    assert "[cwd:" not in second.output          # unchanged this time


async def test_a_deleted_sticky_directory_falls_back_to_the_project(project: Path):
    ctx = make_ctx(project)
    ctx.state["bash_cwd"] = str(project / "gone")
    result = await bash(project, {"command": "pwd"}, ctx)
    assert os.path.realpath(result.output.splitlines()[0]) == os.path.realpath(str(project))
    assert "bash_cwd" not in ctx.state


async def test_exit_status_survives_the_cwd_probe(project: Path):
    result = await bash(project, {"command": "cd / && false"})
    assert "[exit code 1]" in result.output


async def test_the_cwd_marker_never_leaks_into_the_output(project: Path):
    result = await bash(project, {"command": "echo done"})
    assert "__LC_CWD_" not in result.output
    assert result.output == "done\n[exit code 0]"


async def test_a_trailing_backslash_command_still_runs(project: Path):
    result = await bash(project, {"command": "echo a \\"})
    assert result.ok
    assert result.output.startswith("a")


# --------------------------------------------------------------------------- #
# Timeout and cancellation
# --------------------------------------------------------------------------- #


async def test_timeout_kills_the_command_and_its_children(project: Path):
    marker = project / "child.pid"
    started = time.monotonic()
    result = await bash(project, {"command": f"sleep 30 & echo $! > {marker}; sleep 30",
                                  "timeout": 1})
    elapsed = time.monotonic() - started
    pid = child_pid(marker)
    try:
        assert not result.ok
        assert elapsed < 8, "the timeout did not stop the command"
        assert "timed out after 1 s" in result.output
        assert result.summary == "Przerwane po 1 s"
        assert await wait_until_gone(pid), f"orphan process {pid} survived the timeout"
    finally:
        kill_pid(pid)


async def test_cancellation_kills_the_whole_tree(project: Path):
    marker = project / "child.pid"
    tool = get_tool("bash")
    ctx = make_ctx(project)
    task = asyncio.create_task(
        tool.run({"command": f"sleep 30 & echo $! > {marker}; sleep 30", "timeout": 60}, ctx))
    pid = await asyncio.to_thread(child_pid, marker)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    try:
        assert await wait_until_gone(pid), f"orphan process {pid} survived cancellation"
    finally:
        kill_pid(pid)


async def test_a_quick_command_is_not_affected_by_the_timeout(project: Path):
    started = time.monotonic()
    result = await bash(project, {"command": "echo fast", "timeout": 30})
    assert result.ok
    assert time.monotonic() - started < 5


async def test_a_background_child_does_not_delay_the_result(project: Path):
    """The pipe stays open; the tool must return when the shell exits."""
    marker = project / "bg.pid"
    started = time.monotonic()
    result = await bash(project, {"command": f"(sleep 20 & echo $! > {marker}); echo done",
                                  "timeout": 30})
    pid = child_pid(marker)
    try:
        assert result.ok
        assert "done" in result.output
        assert time.monotonic() - started < 5
        assert pid_alive(pid), "a deliberate background job must not be killed"
    finally:
        kill_pid(pid)
        assert await wait_until_gone(pid)


@pytest.mark.parametrize("args, expected", [
    ({}, 120),
    ({"timeout": 30}, 30),
    ({"timeout": "45"}, 45),
    ({"timeout": 0}, 120),
    ({"timeout": -5}, 120),
    ({"timeout": 99999}, 100),          # milliseconds, recognised and converted
    ({"timeout": 10_000_000}, MAX_TIMEOUT_S),
])
def test_timeout_parsing(args, expected):
    assert timeout_from_args(args, 120) == expected


async def test_the_timeout_comes_from_the_limits(project: Path):
    ctx = ToolContext(cwd=project, limits=ToolLimits(bash_timeout_s=1))
    started = time.monotonic()
    result = await bash(project, {"command": "sleep 20"}, ctx)
    assert "timed out after 1 s" in result.output
    assert time.monotonic() - started < 8


# --------------------------------------------------------------------------- #
# Progress streaming
# --------------------------------------------------------------------------- #


async def test_progress_lines_are_streamed(project: Path):
    progress = Progress()
    ctx = ToolContext(cwd=project, limits=ToolLimits(), progress=progress)
    result = await bash(project, {"command": "echo one; echo two; echo three"}, ctx)
    assert progress.lines == ["one", "two", "three"]
    assert result.ok


async def test_progress_never_leaks_the_cwd_marker(project: Path):
    progress = Progress()
    ctx = ToolContext(cwd=project, limits=ToolLimits(), progress=progress)
    await bash(project, {"command": "cd /tmp && echo moved"}, ctx)
    assert progress.lines == ["moved"]


async def test_a_failing_progress_callback_does_not_break_the_command(project: Path):
    async def broken(_line: str) -> None:
        raise RuntimeError("UI is gone")

    ctx = ToolContext(cwd=project, limits=ToolLimits(), progress=broken)
    result = await bash(project, {"command": "echo still fine"}, ctx)
    assert result.ok
    assert "still fine" in result.output


# --------------------------------------------------------------------------- #
# Interactive commands
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command, needle", [
    ("vim notes.txt", "interactive editor"),
    ("nano file", "interactive editor"),
    ("less big.log", "interactive pager"),
    ("cat x | less", "interactive pager"),
    ("top", "monitor"),
    ("htop -d 5", "monitor"),
    ("watch ls", "repeats forever"),
    ("tail -f server.log", "never ends"),
    ("python", "REPL"),
    ("node", "REPL"),
    ("git commit", "without -m"),
    ("git rebase -i HEAD~3", "interactive/patch mode"),
    ("git add -p", "interactive/patch mode"),
    ("ssh host", "remote session"),
    ("sudo vim /etc/hosts", "interactive editor"),
    ("tmux new", "multiplexer"),
    ("mysql", "database shell"),
    ("sqlite3 app.db", "database shell"),
    ("docker run -it ubuntu bash", "interactive terminal"),
    ("echo x; vim y", "interactive editor"),
    ("bash", "wait for input"),
])
def test_interactive_commands_are_detected(command: str, needle: str):
    reason = interactive_reason(command)
    assert reason is not None, command
    assert needle in reason


@pytest.mark.parametrize("command", [
    "npm test",
    "git commit -m 'fix'",
    "git commit --amend --no-edit",
    "git log --oneline -n 5",
    "python -c 'print(1)'",
    "python -m pytest -q",
    "python script.py",
    "node build.js",
    "tail -n 100 server.log",
    "grep -rn 'vim' src",
    "echo less is more",
    "sqlite3 app.db 'select 1'",
    "ls -la && cat README.md",
    "PAGER=cat git diff",
    "./scripts/build.sh",
    "command -v vim",
    "which less",
    "perl -i -pe 's/a/b/' f.txt",
    "ssh build-host 'make all'",
    "mysql -e 'select 1'",
    "psql -d app -c 'select 1'",
    "docker run ubuntu ls",
    "git -C /tmp commit -m x",
    "swift build",
    "php -r 'echo 1;'",
    "git log -p -n 2",
    "timeout 30 npm test",
    "grep -rn 'less' docs",
])
def test_normal_commands_are_not_refused(command: str):
    assert interactive_reason(command) is None, command


async def test_an_interactive_command_is_refused_before_running(project: Path):
    result = await bash(project, {"command": "vim README.md"})
    assert not result.ok
    assert "interactive editor" in result.output
    assert "edit_file" in result.output
    assert result.summary == "Polecenie interaktywne"


# --------------------------------------------------------------------------- #
# Arguments and permission
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("args, expected", [
    ({"command": "ls -la"}, "ls -la"),
    ({"cmd": "ls -la"}, "ls -la"),
    ({"shell_command": "ls"}, "ls"),
    ("ls -la", "ls -la"),
    ({"arguments": {"command": "ls"}}, "ls"),
    ('{"command": "ls"}', "ls"),
    # A list is joined the way ``arg_str`` (and therefore ``execute``) joins it:
    # one line per element. The permission analysis must see exactly that string,
    # otherwise ["ls", "rm -rf ~"] would look like a single read-only command.
    ({"command": ["git", "status"]}, "git\nstatus"),
    ({}, ""),
    (None, ""),
])
def test_command_from_args(args, expected):
    assert command_from_args(args) == expected


async def test_empty_command_is_an_error(project: Path):
    result = await bash(project, {"command": "   "})
    assert not result.ok
    assert "command" in result.output


async def test_permission_shows_the_command(project: Path):
    info = await permission_of("bash", {"command": "npm test"}, make_ctx(project))
    assert info is not None
    assert info.title == "Polecenie: npm test"
    assert info.preview == "npm test"
    assert DANGER_FLAG not in info.title


async def test_permission_flags_a_catastrophic_command(project: Path):
    info = await permission_of("bash", {"command": "rm -rf /"}, make_ctx(project))
    assert info is not None
    assert DANGER_FLAG in info.title


async def test_permission_is_none_without_a_command(project: Path):
    assert await permission_of("bash", {}, make_ctx(project)) is None


def test_title_is_polish():
    tool = get_tool("bash")
    assert tool.title({"command": "npm test"}) == "Polecenie: npm test"
    assert tool.title({"command": "x" * 200}).startswith("Polecenie: ")
    assert len(tool.title({"command": "x" * 200})) < 80


# --------------------------------------------------------------------------- #
# run_bash (the reusable runner)
# --------------------------------------------------------------------------- #


async def test_run_bash_returns_the_details(project: Path):
    outcome = await run_bash("echo hi; exit 4", cwd=project, timeout_s=10)
    assert outcome.exit_code == 4
    assert outcome.output.strip() == "hi"
    assert not outcome.timed_out
    assert outcome.duration_s >= 0
    assert outcome.new_cwd is None


async def test_run_bash_reports_a_timeout(project: Path):
    outcome = await run_bash("sleep 20", cwd=project, timeout_s=1)
    assert outcome.timed_out
    assert outcome.exit_code is None


async def test_run_bash_keeps_only_the_configured_bytes(project: Path):
    outcome = await run_bash("for i in $(seq 1 5000); do echo 0123456789; done",
                             cwd=project, timeout_s=30, keep_bytes=4096)
    assert len(outcome.output) < 12_000
    assert "of output dropped" in outcome.output
    assert outcome.dropped_bytes > 0


async def test_a_timeout_keeps_the_output_printed_before_the_kill(project: Path):
    result = await bash(project, {"command": "echo before; sleep 30", "timeout": 1})
    assert "before" in result.output
    assert "timed out after 1 s" in result.output


async def test_progress_is_throttled_for_chatty_commands(project: Path):
    from localcode.agent.tools.shell import PROGRESS_BURST

    progress = Progress()
    ctx = ToolContext(cwd=project, limits=ToolLimits(max_output_chars=2000), progress=progress)
    result = await bash(project, {"command": "for i in $(seq 1 3000); do echo line$i; done",
                                  "timeout": 60}, ctx)
    assert result.ok
    assert PROGRESS_BURST <= len(progress.lines) < 3000
    assert progress.lines[0] == "line1"


async def test_progress_clips_a_very_long_line(project: Path):
    from localcode.agent.tools.shell import PROGRESS_LINE_CHARS

    progress = Progress()
    ctx = ToolContext(cwd=project, limits=ToolLimits(max_output_chars=2000), progress=progress)
    await bash(project, {"command": "python3 -c \"print('x' * 5000)\"", "timeout": 30}, ctx)
    assert progress.lines
    assert all(len(line) <= PROGRESS_LINE_CHARS for line in progress.lines)
    assert progress.lines[0].endswith("…")


# --------------------------------------------------------------------------- #
# A job left running with `&` that dies at once is reported, not passed as fine
# --------------------------------------------------------------------------- #
# `&` succeeds the moment the job forks. A model ran
# `nohup python3 -m http.server8000 … &` (a dropped space), got exit 0, and told
# the user the page was up; localhost answered nothing.


async def test_a_background_job_that_dies_at_once_is_reported(tmp_path) -> None:
    from localcode.agent.tools.shell import BACKGROUND_DIED, run_bash

    result = await run_bash("nohup python3 -m http.server8000 > /dev/null 2>&1 &",
                            cwd=tmp_path, timeout_s=20)
    assert BACKGROUND_DIED in result.output


async def test_a_background_job_that_keeps_running_is_left_alone(tmp_path) -> None:
    from localcode.agent.tools.shell import BACKGROUND_DIED, run_bash

    result = await run_bash("sleep 3 > /dev/null 2>&1 &", cwd=tmp_path, timeout_s=20)
    assert BACKGROUND_DIED not in result.output
    assert result.exit_code == 0


@pytest.mark.parametrize("command, expected", [
    ("python3 -m http.server 8000 &", True),
    ("nohup x > log 2>&1 &", True),
    ("serve & ", True),
    ("true && echo ok", False),
    ("make || exit 1", False),
    ("echo 'a & b'", False),
])
def test_what_counts_as_a_background_job(command: str, expected: bool) -> None:
    from localcode.agent.tools.shell import starts_background_job

    assert starts_background_job(command) is expected
