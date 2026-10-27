"""Containing an engine process (``localcode.security.sandbox``).

The profile is generated, so these tests read it rather than trust it: what it
denies, what it has to keep open for the engine to start at all, and the two
mistakes that already cost a day —

* the Metal shader cache left out of the write list, which made a 0,8-second
  model load take thirty seconds (the sandbox looked thirty times slower when
  all it did was throw away a cache);
* a read allowance without the directories above it, so a sandboxed Python could
  not resolve its own path ("realpath: … Operation not permitted").

``tests/security/test_scan.py`` covers the other half of the feature, the scanner.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import pytest

from localcode.config import Settings
from localcode.security import sandbox
from localcode.security.sandbox import SandboxPlan, SandboxUnavailable, build_profile, mode, wrap

darwin_only = pytest.mark.skipif(platform.system() != "Darwin",
                                 reason="piaskownica sandbox-exec istnieje tylko na macOS")


def plan() -> SandboxPlan:
    return SandboxPlan(read=[Path.home() / ".localcode" / "models"],
                       write=[Path.home() / ".localcode"])


def settings(sandbox_mode: str) -> Settings:
    s = Settings()
    s.security.sandbox = sandbox_mode
    return s


# --------------------------------------------------------------------------- #
# What the profile says
# --------------------------------------------------------------------------- #


def test_the_model_process_has_no_way_out_to_the_network() -> None:
    profile = build_profile(plan())
    assert "(deny network*)" in profile
    # ...but it still has to answer LocalCode on localhost
    assert "(allow network-bind (local ip))" in profile
    assert "(allow network-inbound (local ip))" in profile


def test_a_plan_can_refuse_even_the_local_port() -> None:
    profile = build_profile(SandboxPlan(write=[], local_network=False))
    assert "(deny network*)" in profile
    assert "network-bind" not in profile


def test_the_private_directories_are_denied_for_reading() -> None:
    profile = build_profile(plan())
    home = Path.home()
    for private in ("Documents", "Desktop", ".ssh", "Library/Keychains"):
        assert f'(subpath "{home / private}")' in profile, private


def test_writing_is_allowed_only_where_a_model_legitimately_writes() -> None:
    profile = build_profile(plan())
    write_block = profile.split("(allow file-write*", 1)[1].split(")", 1)[0]
    assert str(Path.home() / ".localcode") in write_block
    assert str(Path.home() / "Documents") not in write_block


@darwin_only
def test_the_metal_shader_cache_is_writable() -> None:
    """Without this, llama.cpp recompiles every kernel on every start: 0,8 s → 30 s.

    The regression that produced this test was subtle: ``os.confstr`` does not
    know ``CS_DARWIN_USER_CACHE_DIR`` on macOS — it raises — so the lookup
    silently returned nothing and the cache directory never reached the profile.
    """
    cache = subprocess.run(["/usr/bin/getconf", "DARWIN_USER_CACHE_DIR"],
                           capture_output=True, text=True, check=False).stdout.strip()
    if not cache:
        pytest.skip("getconf nie zna DARWIN_USER_CACHE_DIR na tej maszynie")
    profile = build_profile(plan())
    write_block = profile.split("(allow file-write*", 1)[1].split("\n)", 1)[0]
    assert cache.rstrip("/") in write_block
    # both spellings, because the sandbox matches the path the process opens
    assert "/private/var/folders" in write_block and "\n    (subpath \"/var/folders" in write_block


def test_a_read_allowance_brings_the_directories_above_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening a file means resolving every directory on the way to it.

    A plan that needs to read something inside a denied tree (the MLX worker's
    own interpreter, in a development install) gets the parents as metadata-only
    allowances: enough to walk the path, not enough to read or list them.
    """
    private = tmp_path / "Documents"
    venv = private / "some-project" / ".venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "PRIVATE_PATHS", lambda: [private])

    profile = build_profile(SandboxPlan(read=[venv], write=[]))

    assert f'(subpath "{private}")' in profile          # still denied as a tree
    assert f'(subpath "{venv}")' in profile             # ...with this one hole
    assert "(allow file-read-metadata" in profile
    assert f'(literal "{venv.parent}")' in profile      # enough to walk to it
    read_allow = profile.split(";; ...except what this engine", 1)[1].split("(allow file-write*", 1)[0]
    assert f'(subpath "{venv.parent}")' not in read_allow


# --------------------------------------------------------------------------- #
# The three modes
# --------------------------------------------------------------------------- #


def test_off_returns_the_command_untouched() -> None:
    cmd, sandboxed, profile = wrap(["/bin/echo", "x"], plan(), settings=settings("off"))
    assert cmd == ["/bin/echo", "x"] and sandboxed is False and profile is None


def test_an_unknown_mode_falls_back_to_auto() -> None:
    for value in ("nonsense", "", None):
        s = Settings()
        s.security.sandbox = value  # type: ignore[assignment]
        assert mode(s) == "auto"


def test_settings_without_a_security_block_still_work() -> None:
    """``mode`` is called with whatever the caller has; it must not raise."""
    assert mode(None) == "auto"
    assert mode(object()) == "auto"


@darwin_only
def test_auto_wraps_the_command_and_writes_a_profile(tmp_path: Path) -> None:
    cmd, sandboxed, profile = wrap(["/bin/echo", "x"], plan(),
                                   settings=settings("auto"), profile_dir=tmp_path)
    assert sandboxed is True
    assert cmd[:2] == [sandbox.SANDBOX_EXEC, "-f"]
    assert cmd[-2:] == ["/bin/echo", "x"]
    assert profile is not None and profile.exists()
    sandbox.cleanup(profile)
    assert not profile.exists()


def test_strict_refuses_when_the_platform_has_no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(SandboxUnavailable) as exc:
        wrap(["/bin/echo", "x"], plan(), settings=settings("strict"))
    assert "ścisły" in str(exc.value)          # Polish, shown to the user


def test_auto_gives_up_quietly_when_the_platform_has_no_sandbox(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "available", lambda: False)
    cmd, sandboxed, profile = wrap(["/bin/echo", "x"], plan(), settings=settings("auto"))
    assert cmd == ["/bin/echo", "x"] and sandboxed is False and profile is None


# --------------------------------------------------------------------------- #
# What the user is told
# --------------------------------------------------------------------------- #


def test_the_status_line_says_which_of_the_four_things_happened() -> None:
    assert "piaskownicy" in sandbox.describe(True, settings("auto"))
    assert "wyłączona" in sandbox.describe(False, settings("off"))
    failed = sandbox.describe(False, settings("auto"))
    assert "bez izolacji" in failed


# --------------------------------------------------------------------------- #
# It actually contains something
# --------------------------------------------------------------------------- #


@darwin_only
def test_a_sandboxed_process_cannot_read_your_documents_or_reach_the_network(
        tmp_path: Path) -> None:
    """The point of the whole module, checked by running it."""
    secret = Path.home() / "Documents"
    if not secret.exists():
        pytest.skip("brak ~/Documents na tej maszynie")
    cmd, sandboxed, profile = wrap(
        ["/bin/sh", "-c", f'ls "{secret}" >/dev/null 2>&1 && echo WIDZI || echo NIE_WIDZI'],
        plan(), settings=settings("auto"), profile_dir=tmp_path)
    assert sandboxed
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        assert out == "NIE_WIDZI"
    finally:
        sandbox.cleanup(profile)


@darwin_only
def test_a_sandboxed_process_still_runs_normally(tmp_path: Path) -> None:
    cmd, _, profile = wrap(["/bin/echo", "działa"], plan(),
                           settings=settings("auto"), profile_dir=tmp_path)
    try:
        assert subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip() == "działa"
    finally:
        sandbox.cleanup(profile)


# --------------------------------------------------------------------------- #
# The agent's shell profile
# --------------------------------------------------------------------------- #


def test_the_shell_profile_names_the_project_in_both_spellings(tmp_path: Path) -> None:
    profile = sandbox.shell_profile(tmp_path)
    assert f'(subpath "{tmp_path}")' in profile
    real = os.path.realpath(tmp_path)
    assert f'(subpath "{real}")' in profile
    if real.startswith("/private/"):                    # /var/folders… and /private/var/folders…
        assert f'(subpath "{real[len("/private"):]}")' in profile


def test_the_shell_profile_closes_home_and_writes(tmp_path: Path) -> None:
    profile = sandbox.shell_profile(tmp_path)
    assert "(allow default)" in profile
    assert f'(deny file-read-data\n    (subpath "{Path.home()}")' in profile
    assert '(deny file-write* (subpath "/"))' in profile
    assert '(subpath "/tmp")' in profile and '(subpath "/dev")' in profile


def test_the_secrets_are_denied_after_the_read_allowance(tmp_path: Path) -> None:
    profile = sandbox.shell_profile(tmp_path)
    allow = profile.index("(allow file-read-data")
    secret = Path.home() / ".config/gh"
    deny_secret = profile.index(f'(subpath "{secret}")', allow + 1)
    assert deny_secret > profile.index(")", allow)
    assert "(deny file-read-data" in profile[profile.index(")", allow):deny_secret]
    assert profile.index('(deny file-write* (subpath "/"))') > deny_secret


def test_wrap_shell_prefixes_sandbox_exec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "available", lambda: True)
    argv = sandbox.wrap_shell(["/bin/bash", "-c", "ls"], tmp_path)
    assert argv[:2] == [sandbox.SANDBOX_EXEC, "-p"]
    assert argv[2] == sandbox.shell_profile(tmp_path)
    assert argv[3:] == ["/bin/bash", "-c", "ls"]


def test_wrap_shell_is_a_no_op_without_a_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "available", lambda: False)
    assert sandbox.wrap_shell(["/bin/bash", "-c", "ls"], tmp_path) == ["/bin/bash", "-c", "ls"]


def test_shell_home_paths_keep_secrets_apart() -> None:
    readable, secret = sandbox.shell_home_paths()
    assert Path.home() / ".cargo" in readable
    assert Path.home() / ".config/gh" in secret
