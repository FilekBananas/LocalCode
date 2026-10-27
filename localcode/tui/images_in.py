"""Pictures in a terminal message, for a model that can see them.

Two ways to attach one, both in the message itself:

* ``@zrzut.png`` / ``@~/Desktop/zdjęcie.jpg`` — like any other ``@`` file;
* dragging a file onto the Terminal window, which types its path — with
  ``\\ `` before every space, or in quotes.

Every picture is sent as base64 next to the text (``Backend.send_user(…,
images=…)``). Photos straight from a camera are many megabytes and HEIC, which
no model API takes, so anything large or unusual is first converted with
macOS's own ``sips`` to a JPEG at most :data:`MAX_SIDE` pixels on its long side
— the size vision models scale down to anyway.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["IMAGE_EXTENSIONS", "PreparedImage", "collect_images", "is_image_path", "model_sees_images"]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff")
#: Sent as they are when small enough; everything else goes through sips.
_PASS_THROUGH = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp"}
MAX_SIDE = 1600
MAX_RAW_BYTES = 3 * 1024 * 1024
MAX_IMAGES = 6

_EXT = r"\.(?:png|jpe?g|gif|webp|heic|heif|bmp|tiff?)"
#: ``@path`` (no spaces) — the same token the text attachments use.
_AT = re.compile(r"(?<![\w@])@((?:[^\s@\"'`\\]|\\ )+" + _EXT + r")\b", re.IGNORECASE)
#: A dropped path: absolute or ``~``, spaces escaped with a backslash…
_DROPPED = re.compile(r"(?<![\w/@])((?:~|/)(?:[^\s\\'\"]|\\.)+" + _EXT + r")(?=\s|$)", re.IGNORECASE)
#: …or the whole thing in quotes.
_QUOTED = re.compile(r"(['\"])((?:~|/)[^'\"]+" + _EXT + r")\1", re.IGNORECASE)


@dataclass
class PreparedImage:
    name: str
    mime: str
    data: str            # base64
    note: str = ""       # "zmniejszono do 1600 px", for the line the terminal prints

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "mime": self.mime, "data": self.data}


def is_image_path(path: str) -> bool:
    return str(path).lower().endswith(IMAGE_EXTENSIONS)


def model_sees_images(model: dict[str, Any] | None) -> bool:
    """Whether the active model takes pictures in LocalCode.

    A remote model says so in its capabilities; a local GGUF model needs its
    vision projector (``hasVision``). MLX vision models are not wired to the MLX
    engine, so they do not count — better a clear "no" than an image the model
    silently never receives.
    """
    if not model:
        return False
    caps = set(model.get("capabilities") or [])
    if "remote" in caps or model.get("remote"):
        return "vision" in caps
    if str(model.get("format") or "").lower() == "mlx":
        return False
    return bool(model.get("hasVision")) or "vision" in caps


def _unescape(raw: str) -> str:
    return re.sub(r"\\(.)", r"\1", raw)


def _candidates(text: str) -> list[str]:
    found: list[str] = []
    for match in _AT.finditer(text):
        found.append(_unescape(match.group(1)))
    for match in _QUOTED.finditer(text):
        found.append(match.group(2))
    for match in _DROPPED.finditer(text):
        found.append(_unescape(match.group(1)))
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def _resolve(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (cwd / path)


def _convert(path: Path) -> tuple[bytes, str] | None:
    """JPEG, long side at most MAX_SIDE, via macOS sips; ``None`` without sips."""
    sips = shutil.which("sips")
    if not sips:
        return None
    with tempfile.TemporaryDirectory(prefix="localcode-img-") as tmp:
        out = Path(tmp) / "obraz.jpg"
        try:
            subprocess.run([sips, "-s", "format", "jpeg", "-s", "formatOptions", "85",
                            "-Z", str(MAX_SIDE), str(path), "--out", str(out)],
                           check=True, capture_output=True, timeout=60)
            return out.read_bytes(), "image/jpeg"
        except (OSError, subprocess.SubprocessError):
            return None


def _prepare(path: Path) -> tuple[PreparedImage | None, str | None]:
    ext = path.suffix.lower()
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, f"Nie można odczytać {path.name}: {exc.strerror or exc}."
    if ext in _PASS_THROUGH and size <= MAX_RAW_BYTES:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return None, f"Nie można odczytać {path.name}: {exc.strerror or exc}."
        return PreparedImage(path.name, _PASS_THROUGH[ext], base64.b64encode(raw).decode("ascii")), None
    converted = _convert(path)
    if converted is None:
        return None, (f"Nie udało się przygotować {path.name} — zapisz go jako PNG albo JPEG "
                      f"(do {MAX_RAW_BYTES // (1024 * 1024)} MB).")
    data, mime = converted
    return PreparedImage(path.name, mime, base64.b64encode(data).decode("ascii"),
                         note=f"zmniejszony do {MAX_SIDE} px"), None


def collect_images(text: str, cwd: str | os.PathLike[str]) -> tuple[list[PreparedImage], list[str]]:
    """The pictures a message refers to (``@obraz.png`` or a dropped path), ready to send."""
    root = Path(cwd).expanduser()
    images: list[PreparedImage] = []
    warnings: list[str] = []
    for raw in _candidates(text or ""):
        path = _resolve(raw, root)
        if not path.is_file():
            if raw.startswith(("/", "~")) or "@" + raw in (text or ""):
                warnings.append(f"Nie znaleziono obrazu {raw}.")
            continue
        if len(images) >= MAX_IMAGES:
            warnings.append(f"Pominięto {path.name}: najwyżej {MAX_IMAGES} obrazów w jednej wiadomości.")
            continue
        image, warning = _prepare(path)
        if warning:
            warnings.append(warning)
        if image is not None:
            images.append(image)
    return images, warnings
