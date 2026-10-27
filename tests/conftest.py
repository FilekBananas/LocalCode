"""Fixtures shared by every test area."""

from __future__ import annotations

import os

import pytest

from lc_testing import MODELS, TEST_MODELS, model_path  # noqa: F401


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point LOCALCODE_HOME at a temp dir so tests never touch ~/.localcode."""
    home = tmp_path / "lc-home"
    home.mkdir()
    monkeypatch.setenv("LOCALCODE_HOME", str(home))
    return home


#: LOCALCODE_HOME as the test run found it: a test that has not changed it
#: (``isolated_home``) is still pointing at the real one.
_HOME_AT_START = os.environ.get("LOCALCODE_HOME")


@pytest.fixture(autouse=True)
def _precedents_off_the_real_home(tmp_path_factory, monkeypatch):
    """The agent learns from every permission answer (``agent/precedents.py``) and
    keeps it in LOCALCODE_HOME; a test that did not isolate the home must still
    not write ``~/.localcode/auto_precedents.json``."""
    from localcode.agent import precedents

    spare = tmp_path_factory.mktemp("precedents") / precedents.STORE_NAME
    real = precedents.store_path
    monkeypatch.setattr(precedents, "store_path",
                        lambda: real() if os.environ.get("LOCALCODE_HOME") != _HOME_AT_START else spare)


@pytest.fixture(autouse=True)
def _empty_model_index(tmp_path_factory, monkeypatch):
    """Hide the shipped store index (``localcode/hub/data``) from the tests.

    The index ships thousands of real models; a test that asserts "offline the
    store falls back to the staff picks" would otherwise be asserting something
    about Hugging Face's download charts.  Tests that need an index install
    their own by pointing ``store_data.DATA_PATH`` at a file they wrote.
    """
    from localcode.hub import store_data

    missing = tmp_path_factory.mktemp("no-index") / "catalog.json.gz"
    monkeypatch.setattr(store_data, "DATA_PATH", missing)
    store_data.reset_cache()
    yield
    store_data.reset_cache()


@pytest.fixture(autouse=True)
def _no_helpbot_download(monkeypatch):
    """A server started by a test must not download the help bot's 2.3 GB model.

    Tests that check the download itself call ``helpbot.ensure_installed`` with
    a fake store instead.
    """
    monkeypatch.setenv("LOCALCODE_NO_HELPBOT_DOWNLOAD", "1")
