"""Fixtures for the hub tests (builders live in ``hub_fixtures.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from localcode.config import Settings


@pytest.fixture
def hub_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated ``LOCALCODE_HOME`` so caches/state never touch ``~/.localcode``."""
    home = tmp_path / "lc-home"
    (home / "data").mkdir(parents=True)
    monkeypatch.setenv("LOCALCODE_HOME", str(home))
    return home


@pytest.fixture
def models_dir(hub_home: Path) -> Path:
    d = hub_home / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def settings(hub_home: Path, models_dir: Path) -> Settings:
    """Settings that only see the temporary models directory."""
    return Settings(models_dir=str(models_dir), scan_lmstudio=False, scan_hf_cache=False)
