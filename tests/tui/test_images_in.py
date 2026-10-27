"""Attaching pictures in the terminal: ``@obraz.png`` or a path dropped onto the window."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from localcode.tui.images_in import collect_images, model_sees_images

PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
       b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND"
       b"\xaeB`\x82")


def test_an_at_reference_attaches_the_picture(tmp_path: Path) -> None:
    (tmp_path / "zrzut.png").write_bytes(PNG)
    images, warnings = collect_images("co jest na @zrzut.png ?", tmp_path)
    assert [i.name for i in images] == ["zrzut.png"] and images[0].mime == "image/png" and not warnings


def test_a_path_dropped_onto_the_terminal_is_found(tmp_path: Path) -> None:
    """Terminal types a dropped file as its path, with "\\ " before each space."""
    folder = tmp_path / "Zrzuty ekranu"
    folder.mkdir()
    (folder / "Zrzut ekranu 1.png").write_bytes(PNG)
    escaped = str(folder / "Zrzut ekranu 1.png").replace(" ", "\\ ")
    images, _ = collect_images(f"opisz {escaped} prosze", tmp_path)
    assert [i.name for i in images] == ["Zrzut ekranu 1.png"]
    quoted, _ = collect_images(f"opisz '{folder / 'Zrzut ekranu 1.png'}'", tmp_path)
    assert [i.name for i in quoted] == ["Zrzut ekranu 1.png"]


def test_a_missing_picture_is_reported(tmp_path: Path) -> None:
    images, warnings = collect_images("@brak.png", tmp_path)
    assert images == [] and "Nie znaleziono" in warnings[0]


@pytest.mark.skipif(shutil.which("sips") is None, reason="sips is macOS-only")
def test_a_heic_or_a_huge_photo_is_converted_to_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "foto.png"
    source.write_bytes(PNG)
    heic = tmp_path / "foto.tiff"
    import subprocess
    subprocess.run(["sips", "-s", "format", "tiff", str(source), "--out", str(heic)], check=True,
                   capture_output=True)
    images, warnings = collect_images("@foto.tiff", tmp_path)
    assert images and images[0].mime == "image/jpeg" and "zmniejszony" in images[0].note, warnings


@pytest.mark.parametrize("model, sees", [
    ({"format": "gguf", "hasVision": True, "capabilities": ["vision"]}, True),
    ({"format": "gguf", "hasVision": False, "capabilities": ["tools"]}, False),
    ({"format": "mlx", "hasVision": True, "capabilities": ["vision"]}, False),     # not wired to MLX
    ({"format": "remote", "remote": True, "capabilities": ["tools", "remote", "vision"]}, True),
    ({"format": "remote", "remote": True, "capabilities": ["tools", "remote"]}, False),
    (None, False),
])
def test_which_active_models_take_pictures(model, sees: bool) -> None:
    assert model_sees_images(model) is sees
