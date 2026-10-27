"""Make the command-line tools the app depends on findable.

A program started from Finder or by launchd gets macOS's minimal
``PATH=/usr/bin:/bin:/usr/sbin:/sbin`` — not the one a terminal has. The Mac
app starts the server exactly that way, so anything the server looks up *by
name* is invisible even when it is installed: mlx-whisper runs ``ffmpeg`` to
decode a recording, and transcription failed with
``[Errno 2] No such file or directory: 'ffmpeg'`` on a machine where
``/opt/homebrew/bin/ffmpeg`` was right there.

:func:`ensure_tool_path` appends Homebrew's directories when they exist (after
the system's own, so a system binary still wins) and, when there is still no
``ffmpeg``, points one at the copy ``imageio-ffmpeg`` ships inside the video
extra — so a Mac without Homebrew can transcribe too.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["TOOL_DIRS", "ensure_tool_path"]

#: Where Homebrew puts binaries on Apple Silicon and on Intel.
TOOL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")


def ensure_tool_path(*, home: Path | None = None) -> list[str]:
    """Extend ``os.environ["PATH"]``; returns the directories it added.

    Safe to call more than once and from any entry point: a directory already on
    the path is never added twice.
    """
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    added: list[str] = []
    for directory in TOOL_DIRS:
        if directory not in parts and os.path.isdir(directory):
            parts.append(directory)
            added.append(directory)
    os.environ["PATH"] = os.pathsep.join(parts)

    if shutil.which("ffmpeg") is None:
        shim = _bundled_ffmpeg_dir(home)
        if shim is not None and shim not in parts:
            parts.append(shim)
            added.append(shim)
            os.environ["PATH"] = os.pathsep.join(parts)
    if added:
        log.info("PATH uzupełniony o: %s", ", ".join(added))
    return added


def _bundled_ffmpeg_dir(home: Path | None) -> str | None:
    """A directory holding ``ffmpeg`` → the binary inside ``imageio-ffmpeg``.

    That package names its binary ``ffmpeg-macos-aarch64-v7.1``; tools look for
    plain ``ffmpeg``, so a symlink under ``<LOCALCODE_HOME>/bin`` gives it the
    expected name without copying 50 MB.
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415 - only with the video extra

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not installed, or no binary for this platform
        return None
    if not exe or not os.path.isfile(exe):
        return None
    if home is None:
        from localcode.config import home as config_home  # noqa: PLC0415

        home = config_home()
    directory = Path(home) / "bin"
    link = directory / "ffmpeg"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and os.readlink(link) != exe:
            link.unlink()
        if not link.exists():
            link.symlink_to(exe)
    except OSError as exc:
        log.warning("nie udało się podpiąć ffmpeg z imageio-ffmpeg: %s", exc)
        return None
    return str(directory)
