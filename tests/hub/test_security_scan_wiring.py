"""End-to-end check that the security scan (``localcode/security/scan.py``) really
runs at the two points the app promises it does: right after a local import
(``ModelLibrary.import_path``) and right after a download finishes.

A download itself has no dedicated scan call anywhere — by design
(``localcode/hub/download.py``'s docstring): ``DownloadManager.on_complete`` is a
generic "a job finished" hook, and the production wiring
(``localcode/server/core.py``'s ``_on_download_complete`` → ``rescan_library`` →
``ModelLibrary.scan()``) is what turns "download done" into "scan the new
files". That wiring lives outside ``localcode/hub/`` (in ``server/core.py``,
owned elsewhere), so it is verified here by reading the code, not by spinning
up the whole server. What *is* tested here, directly, is the part that
actually does the analysis and that this module owns: ``ModelLibrary.scan()``
computing a real, correct security verdict for exactly the directory layout a
finished download or a finished import leaves on disk — including a
downloaded/imported GGUF file whose *sibling* files (not the model file
itself) carry a malicious payload, since that is the realistic shape of a
poisoned repository and the scanner scans the whole directory, not just the
recognised model file (see ``ModelLibrary._security_scan``).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from hub_fixtures import make_mlx_dir, write_gguf
from localcode.config import Settings
from localcode.hub.download import job_model_key
from localcode.hub.library import ModelLibrary


class _ReduceToSystemCall:
    """Pickling this embeds a call to ``os.system(...)`` in the opcode stream —
    the same "dangerous" shape ``tests/security/test_scan.py`` uses, kept here
    too so this file does not depend on that one."""

    def __reduce__(self):
        import os
        return (os.system, ("id",))


def dangerous_pickle_bytes() -> bytes:
    return pickle.dumps(_ReduceToSystemCall())


# --------------------------------------------------------------------------- #
# After a local import (ModelLibrary.import_path)
# --------------------------------------------------------------------------- #


def test_import_path_scans_an_mlx_directory_and_flags_a_bundled_payload(
    settings: Settings, models_dir: Path, tmp_path: Path,
):
    source = tmp_path / "external" / "some-mlx-model"
    make_mlx_dir(source)
    # A poisoned source directory: the weights look fine, but something else
    # sitting right next to them is a pickle that runs a shell command.
    (source / "convert_helper.pt").write_bytes(dangerous_pickle_bytes())

    library = ModelLibrary(settings)
    model = library.import_path(source, name="imported")

    assert model.scan is not None
    assert model.scan["verdict"] == "dangerous"
    assert any(f["detail"].endswith("system") for f in model.scan["findings"])
    # the same verdict is what a fresh scan (e.g. the app restarting) would find
    assert library.get(model.key).scan["verdict"] == "dangerous"


def test_import_path_scans_a_clean_mlx_directory_as_clean(
    settings: Settings, models_dir: Path, tmp_path: Path,
):
    source = tmp_path / "external" / "clean-mlx-model"
    make_mlx_dir(source)
    library = ModelLibrary(settings)
    model = library.import_path(source, name="clean-import")
    assert model.scan is not None and model.scan["verdict"] == "clean"


def test_import_path_of_a_gguf_file_is_scanned_too(
    settings: Settings, models_dir: Path, tmp_path: Path,
):
    """A ``.gguf`` import only ever copies ``.gguf`` files (see
    ``ModelLibrary.import_path``), so the model itself can never carry a
    pickle payload — but the scan still runs and still reports "clean" by
    construction, exactly like a freshly downloaded GGUF would."""
    source_dir = tmp_path / "external"
    gguf = write_gguf(source_dir / "some-model-Q4_K_M.gguf")
    library = ModelLibrary(settings)
    model = library.import_path(gguf)
    assert model.scan is not None and model.scan["verdict"] == "clean"


# --------------------------------------------------------------------------- #
# After a download completes (the directory layout DownloadManager leaves,
# scanned by the same ModelLibrary.scan() that ``server.core.rescan_library()``
# calls from ``DownloadManager.on_complete`` — see module docstring)
# --------------------------------------------------------------------------- #


def test_scan_after_download_flags_a_malicious_sibling_file(settings: Settings, models_dir: Path):
    repo = "someorg/Some-Model-GGUF"
    write_gguf(models_dir / repo / "Some-Model-Q4_K_M.gguf")
    # A file a poisoned repo bundled next to the (harmless) GGUF weights —
    # nothing in LocalCode ever asks to load it directly, but it sits inside
    # the same downloaded directory and must not go unnoticed.
    (models_dir / repo / "postprocess.bin").write_bytes(dangerous_pickle_bytes())

    library = ModelLibrary(settings)
    models = library.scan()

    # the model's key must be exactly what the download layer computed it
    # would be — otherwise "the scan ran" would be moot because the UI would
    # be looking at a different key than the one the scan is attached to.
    expected_key = job_model_key(repo, [{"name": "Some-Model-Q4_K_M.gguf"}])
    assert [m.key for m in models] == [expected_key]
    model = library.get(expected_key)
    assert model.scan["verdict"] == "dangerous"
    finding = model.scan["findings"][0]
    assert finding["file"] == "postprocess.bin"
    assert finding["severity"] == "dangerous"


def test_scan_after_download_of_a_clean_repo_is_clean(settings: Settings, models_dir: Path):
    repo = "someorg/Another-Model-GGUF"
    write_gguf(models_dir / repo / "Another-Model-Q4_K_M.gguf")
    (models_dir / repo / "README.md").write_text("# Another Model\n")
    (models_dir / repo / "config.json").write_text("{}")

    library = ModelLibrary(settings)
    models = library.scan()
    expected_key = job_model_key(repo, [{"name": "Another-Model-Q4_K_M.gguf"}])
    model = library.get(expected_key)
    assert model.scan["verdict"] == "clean"


def test_scan_after_download_of_an_mlx_repo_flags_its_own_dangerous_file(
    settings: Settings, models_dir: Path,
):
    repo = "mlx-community/Some-Model-4bit"
    make_mlx_dir(models_dir / repo)
    (models_dir / repo / "evil.pt").write_bytes(dangerous_pickle_bytes())

    library = ModelLibrary(settings)
    models = library.scan()
    expected_key = job_model_key(repo, [{"name": "config.json"}, {"name": "model.safetensors"}])
    assert expected_key == repo
    model = library.get(expected_key)
    assert model is not None and model.scan["verdict"] == "dangerous"


# --------------------------------------------------------------------------- #
# The scanner must never be the reason a library scan (and therefore a
# download/import) fails outright — see ``ModelLibrary._security_scan``.
# --------------------------------------------------------------------------- #


def test_a_broken_scanner_does_not_break_the_library_scan(
    settings: Settings, models_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    import localcode.hub.library as lib_module

    def boom(_path):
        raise RuntimeError("disk fell over")

    monkeypatch.setattr(lib_module, "_security_scan_path", boom)
    library = ModelLibrary(settings)
    models = library.scan()
    assert len(models) == 1
    assert models[0].scan is None       # "not scanned", never a false "clean"


# --------------------------------------------------------------------------- #
# settings.security.scan_downloads — was declared (config.py) but never
# actually read anywhere; ModelLibrary._security_scan now honours it.
# --------------------------------------------------------------------------- #


def test_scan_downloads_setting_is_honoured(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    (models_dir / "org/repo/evil.bin").write_bytes(dangerous_pickle_bytes())

    settings.security.scan_downloads = False
    library = ModelLibrary(settings)
    model = library.scan()[0]
    assert model.scan is None           # scanning turned off -> "not scanned", not a false "clean"

    settings.security.scan_downloads = True
    model = ModelLibrary(settings).scan()[0]
    assert model.scan is not None and model.scan["verdict"] == "dangerous"


def test_scan_downloads_defaults_to_on(settings: Settings, models_dir: Path):
    assert settings.security.scan_downloads is True
    write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    model = ModelLibrary(settings).scan()[0]
    assert model.scan is not None
