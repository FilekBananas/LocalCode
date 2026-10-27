"""The agent's bash is confined to the project folder.

Three layers: ``outside_paths`` (the question the user is asked), the tool's
permission title / policy decision, and — on macOS — the real sandbox, driven
with commands that reach outside only at run time (so no flag, no question).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from localcode.agent.permissions import OUTSIDE_PROJECT_FLAG, PermissionPolicy, outside_paths
from localcode.agent.tools import get_tool
from localcode.agent.tools.shell import CONFINE_STATE_KEY, CONFINED_HINT
from localcode.security import sandbox
from tool_helpers import make_ctx, permission_of, run_tool, write_file


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_file(tmp_path / "README.md", "# Demo\n")
    (tmp_path / "src").mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# outside_paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("command", [
    "cat ~/.ssh/id_rsa",
    "ls ~/Documents",
    "grep -r x ~/",
    "cat $HOME/.aws/credentials",
    'python3 -c "open(\'/Users/someone/x\')"',
    "find / -name x",
    "ls /Volumes/X",
    "cd",
    "cat ~/.config/gh/hosts.yml",
])
def test_commands_reaching_user_data_are_flagged(command: str, project: Path) -> None:
    assert outside_paths(command, project), command


def test_a_dotdot_escape_from_a_project_under_users_is_flagged() -> None:
    # Only user data counts, so the project has to live under /Users for ``..`` to matter
    # (tmp dirs sit under /private/var, where ``..`` leads to harmless temp files).
    project = "/Users/someone/proj"          # need not exist; paths are resolved lexically
    assert outside_paths("cat ../other/x", project) == ["../other/x"]
    assert outside_paths("cat src/../README.md", project) == []


@pytest.mark.parametrize("command", [
    "npm test",
    "git log HEAD~3",
    "sed s/a/b/ f",
    "cat /etc/hosts",
    "curl https://example.com/a/b",
    "cat src/../README.md",
    "~/.cargo/bin/cargo build",
    'echo "/api/users"',
    "ls /tmp /usr/bin",
])
def test_ordinary_commands_are_not_flagged(command: str, project: Path) -> None:
    assert outside_paths(command, project) == []


def test_an_absolute_path_inside_the_project_is_not_flagged(project: Path) -> None:
    assert outside_paths(f"cat {project}/README.md", project) == []
    assert outside_paths(f"cat {os.path.realpath(project)}/src", project) == []


def test_a_cwd_outside_the_project_is_flagged(project: Path) -> None:
    other = Path.home()
    assert outside_paths("ls", project, cwd=other) == [os.path.realpath(other)]
    assert outside_paths("ls", project, cwd=project / "src") == []


def test_the_flagged_word_is_returned_as_written(project: Path) -> None:
    assert outside_paths("cat ~/.ssh/id_rsa", project) == ["~/.ssh/id_rsa"]


def test_a_command_that_cds_back_into_the_project_is_not_flagged(project: Path) -> None:
    """The shell was left in /tmp by an earlier call; this one starts with ``cd`` home."""
    elsewhere = os.path.realpath(Path.home())
    assert outside_paths(f'cd "{project}" && grep -n x src/a.py | head', project, cwd=elsewhere) == []
    assert outside_paths("grep -n x src/a.py", project, cwd=elsewhere) == [elsewhere]
    assert outside_paths("cd src && ls", project, cwd=elsewhere) == [elsewhere]
    # a scratch dir in /tmp is not somebody's data: the sandbox lets the shell write there
    assert outside_paths("rm -rf ./* && unzip -o x.zip", project, cwd="/private/tmp/scratch") == []


def test_a_quoted_project_path_with_spaces_stays_inside() -> None:
    """Cut at the space, ``"…/my bot project"`` read as the sibling ``…/my``."""
    spaced = os.path.join(Path.home(), "Documents", "my bot project")
    assert outside_paths(f'cd "{spaced}" && sed -n 1,9p "notes dir/a.md"', spaced) == []
    assert outside_paths(f"cat '{spaced}/a b.txt'", spaced) == []
    assert outside_paths('cat "$HOME/Documents/my bot project/a"', spaced) == []
    assert outside_paths(f'cat "{Path.home()}/Documents/other dir/x"', spaced) == [
        f"{Path.home()}/Documents/other dir/x"]
    # a quoted command line is still searched word by word
    assert outside_paths(f'bash -c "/usr/bin/env {Path.home()}/secret"', spaced) == [
        f"{Path.home()}/secret"]


# --------------------------------------------------------------------------- #
# Permission title and policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("args", [
    {"command": "cat ~/.zshrc"},
    {"command": "ls", "outside_project": True},
    {"command": "ls", "unsandboxed": True},
])
async def test_reaching_outside_flags_the_permission_title(args: dict, project: Path) -> None:
    info = await permission_of("bash", args, make_ctx(project))
    assert info is not None and OUTSIDE_PROJECT_FLAG in info.title
    assert "poza" in info.preview


async def test_an_ordinary_command_is_not_flagged(project: Path) -> None:
    info = await permission_of("bash", {"command": "ls"}, make_ctx(project))
    assert info is not None and OUTSIDE_PROJECT_FLAG not in info.title
    assert info.preview == "ls"


@pytest.mark.parametrize("args", [{"command": "cat ~/.zshrc"},
                                  {"command": "ls", "outside_project": True}])
async def test_with_confinement_off_nothing_is_flagged(args: dict, project: Path) -> None:
    ctx = make_ctx(project)
    ctx.state[CONFINE_STATE_KEY] = False
    info = await permission_of("bash", args, ctx)
    assert info is not None and OUTSIDE_PROJECT_FLAG not in info.title


@pytest.mark.parametrize("mode", ["yolo", "default", "accept_edits"])
async def test_a_flagged_bash_call_always_asks(mode: str, project: Path) -> None:
    tool = get_tool("bash")
    args = {"command": "cat ~/.zshrc"}
    info = await tool.permission(args, make_ctx(project))
    assert PermissionPolicy(mode, cwd=project).decide(tool, args, info) == "ask"


async def test_always_allowed_does_not_silence_the_outside_question(project: Path) -> None:
    tool = get_tool("bash")
    args = {"command": "cat ~/.zshrc"}
    info = await tool.permission(args, make_ctx(project))
    policy = PermissionPolicy("default", cwd=project, always_tools=["bash"],
                              always_commands=["cat ~/.zshrc"])
    assert policy.decide(tool, args, info) == "ask"


async def test_yolo_runs_an_ordinary_command_without_asking(project: Path) -> None:
    tool = get_tool("bash")
    args = {"command": "make build"}
    info = await tool.permission(args, make_ctx(project))
    assert PermissionPolicy("yolo", cwd=project).decide(tool, args, info) == "allow"


def test_the_bash_schema_offers_outside_project() -> None:
    props = get_tool("bash").schema()["function"]["parameters"]["properties"]
    assert props["outside_project"]["type"] == "boolean"


# --------------------------------------------------------------------------- #
# The real sandbox (macOS)
# --------------------------------------------------------------------------- #


def _sandbox_works() -> bool:
    """``sandbox-exec`` can be applied here (not when this process is sandboxed itself)."""
    import subprocess

    if not sandbox.available():
        return False
    try:
        done = subprocess.run(sandbox.wrap_shell(["/usr/bin/true"], tempfile.gettempdir()),
                              capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


needs_sandbox = pytest.mark.skipif(not _sandbox_works(),
                                   reason="sandbox-exec niedostępny lub nie da się go tu zastosować")

PROBE = Path.home() / ".lc_sandbox_probe"


@needs_sandbox
async def test_reading_the_home_folder_at_run_time_is_blocked(project: Path) -> None:
    # No outside path is *named*, so the command is not flagged and runs confined.
    command = "python3 -c \"import os; print(os.listdir(os.environ['HO'+'ME']))\""
    assert outside_paths(command, project) == []
    result = await run_tool("bash", {"command": command}, make_ctx(project))
    assert not result.ok
    assert "Operation not permitted" in result.output
    assert CONFINED_HINT in result.output


@needs_sandbox
async def test_writing_to_the_home_folder_at_run_time_is_blocked(project: Path) -> None:
    assert not PROBE.exists(), "stale probe file from an earlier run"
    command = "python3 -c \"import os; open(os.environ['HO'+'ME']+'/.lc_sandbox_probe','w')\""
    try:
        result = await run_tool("bash", {"command": command}, make_ctx(project))
        assert not result.ok
        assert "Operation not permitted" in result.output
        assert not PROBE.exists()
    finally:
        if PROBE.exists():
            PROBE.unlink()


@needs_sandbox
async def test_the_project_and_temp_files_stay_writable(project: Path) -> None:
    result = await run_tool("bash", {"command": "echo hi > out.txt && cat out.txt"},
                            make_ctx(project))
    assert result.ok, result.output
    assert (project / "out.txt").read_text() == "hi\n"

    scratch = tempfile.mkdtemp(prefix="lc_conf_")
    try:
        command = f"echo t > {scratch}/f && cat {scratch}/f"
        result = await run_tool("bash", {"command": command}, make_ctx(project))
        assert result.ok, result.output
        result = await run_tool(
            "bash", {"command": "python3 -c \"import tempfile; f=tempfile.NamedTemporaryFile(); "
                                "f.write(b'x'); print('tmp ok')\""}, make_ctx(project))
        assert result.ok and "tmp ok" in result.output, result.output
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@needs_sandbox
async def test_git_still_works_inside_the_sandbox(project: Path) -> None:
    result = await run_tool("bash", {"command": "git --version"}, make_ctx(project))
    assert result.ok and "git version" in result.output, result.output


@needs_sandbox
async def test_with_confinement_off_the_home_folder_is_readable(project: Path) -> None:
    ctx = make_ctx(project)
    ctx.state[CONFINE_STATE_KEY] = False
    command = "python3 -c \"import os; print(len(os.listdir(os.environ['HO'+'ME']))>0)\""
    result = await run_tool("bash", {"command": command}, ctx)
    assert result.ok and "True" in result.output, result.output
