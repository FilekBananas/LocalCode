"""``ensure_tool_path``: the server finds ffmpeg even when started from Finder.

A program launched by Finder or launchd gets ``PATH=/usr/bin:/bin:/usr/sbin:/sbin``.
mlx-whisper runs ``ffmpeg`` by name, so the app's transcription failed with
"No such file or directory: 'ffmpeg'" while ``/opt/homebrew/bin/ffmpeg`` existed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localcode import system_path


@pytest.fixture
def minimal_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    return tmp_path


def test_homebrew_directories_are_appended_after_the_systems(minimal_path, monkeypatch) -> None:
    brew = minimal_path / "brew-bin"
    brew.mkdir()
    monkeypatch.setattr(system_path, "TOOL_DIRS", (str(brew),))
    added = system_path.ensure_tool_path(home=minimal_path)
    parts = os.environ["PATH"].split(os.pathsep)
    assert str(brew) in added
    assert parts[:4] == ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]   # system binaries still win
    assert parts[-1] == str(brew) or str(brew) in parts


def test_calling_it_twice_adds_nothing_twice(minimal_path, monkeypatch) -> None:
    brew = minimal_path / "brew-bin"
    brew.mkdir()
    monkeypatch.setattr(system_path, "TOOL_DIRS", (str(brew),))
    system_path.ensure_tool_path(home=minimal_path)
    assert system_path.ensure_tool_path(home=minimal_path) == []
    assert os.environ["PATH"].split(os.pathsep).count(str(brew)) == 1


def test_a_directory_that_does_not_exist_is_not_added(minimal_path, monkeypatch) -> None:
    monkeypatch.setattr(system_path, "TOOL_DIRS", (str(minimal_path / "nie-ma"),))
    system_path.ensure_tool_path(home=minimal_path)
    assert str(minimal_path / "nie-ma") not in os.environ["PATH"]


def test_without_any_ffmpeg_the_bundled_one_is_given_its_usual_name(minimal_path, monkeypatch) -> None:
    """A Mac without Homebrew: imageio-ffmpeg's binary, symlinked as plain `ffmpeg`."""
    import shutil
    import sys
    import types

    fake = minimal_path / "ffmpeg-macos-aarch64-v7.1"
    fake.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg",
                        types.SimpleNamespace(get_ffmpeg_exe=lambda: str(fake)))
    monkeypatch.setattr(system_path, "TOOL_DIRS", ())
    home = minimal_path / "lc-home"
    system_path.ensure_tool_path(home=home)
    found = shutil.which("ffmpeg")
    assert found is not None
    assert Path(found).resolve() == fake.resolve()
