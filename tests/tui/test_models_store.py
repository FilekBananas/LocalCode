"""Terminal rendering of the in-app model store.

``localcode models search`` / ``curated`` print the friendly catalogue (name,
what the model is for, size, whether it fits, "zainstalowany") and only show
repository ids under ``--advanced``; ``localcode models install "<nazwa>"``
resolves a friendly name, asks before installing a model that will not fit and
then shows a progress bar.  The snapshots are taken with a recording
``rich.Console`` so the exact text is asserted, not the code that produces it.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from localcode.tui import models_ui


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def entry(name: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": f"acme/{name.replace(' ', '-')}-GGUF",
        "name": name,
        "tagline": "Model do pisania i poprawiania kodu.",
        "category": "kodowanie",
        "format": "gguf",
        "sizeBytes": 18_500_000_000,
        "sizeLabel": "ok. 18,5 GB",
        "fit": "Zmieści się",
        "fitDetail": "Potrzebuje ok. 23,1 GB z 44,8 GB dostępnych dla modeli.",
        "speedHint": "Bardzo szybki",
        "installed": False,
        "installedKey": None,
        "recommendedVariant": None,
        "variants": [],
        "badges": ["narzędzia", "kontekst 256k"],
        "source": {"provider": "Hugging Face", "repo": f"acme/{name.replace(' ', '-')}-GGUF"},
        "alternatives": [],
        "requiresToken": False,
        "curated": True,
    }
    base.update(over)
    return base


ENTRIES = [
    entry("Qwen3 Coder 30B"),
    entry("Gemma 4 12B", tagline="Bardzo dobry czat po polsku, obrazy i długi kontekst.",
          category="czat", sizeLabel="ok. 7,3 GB", fit="Na styk", installed=True,
          installedKey="unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M"),
    entry("Qwen3 Coder Next 80B", sizeLabel="ok. 45,0 GB", fit="Za duży",
          fitDetail="Potrzebuje ok. 56,2 GB, a dla modeli jest tylko 44,8 GB."),
]


class FakeCatalog:
    """``Catalog`` stand-in for the terminal tests."""

    def __init__(self, entries: list[dict[str, Any]] | None = None,
                 files: list[str] | None = None) -> None:
        self._entries = entries if entries is not None else ENTRIES
        self.files = files or ["model-Q4_K_M.gguf"]
        self.notice: str | None = None
        self.offline = False
        self.install_calls: list[tuple[str, str | None]] = []

    def entries(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self._entries)

    def resolve_name(self, text: str) -> dict[str, Any] | None:
        folded = text.lower()
        return next((e for e in self._entries if folded in e["name"].lower()), None)

    def install_files(self, entry: dict[str, Any], *, variant: str | None = None) -> list[str]:
        self.install_calls.append((entry["id"], variant))
        return list(self.files)


class FakeBackend:
    """Only the three methods ``install_from_catalog`` touches."""

    def __init__(self, state: str = "done") -> None:
        self.state = state
        self.started: list[tuple[str, str | None, list[str] | None]] = []
        self.refreshed = False

    async def start_download(self, repo: str, *, variant_id: str | None = None,
                             files: list[str] | None = None) -> dict[str, Any]:
        self.started.append((repo, variant_id, files))
        return {"id": "job-1", "repo": repo, "label": "model-Q4_K_M.gguf",
                "state": self.state, "totalBytes": 1_000, "doneBytes": 1_000,
                "modelKey": "acme/Repo/model-Q4_K_M" if self.state == "done" else None,
                "error": None if self.state == "done" else "Brak miejsca na dysku."}

    async def downloads(self) -> list[dict[str, Any]]:
        return []

    async def list_models(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        self.refreshed = refresh
        return []

    async def cancel_download(self, job_id: str) -> None:
        return None


@pytest.fixture
def console() -> Console:
    """A wide recording console with the app's theme (no colours in the export)."""
    from localcode.tui.render import THEME

    return Console(record=True, width=140, color_system=None, theme=THEME,
                   highlight=False, legacy_windows=False, force_terminal=False)


def rendered(console: Console) -> str:
    return console.export_text()


# --------------------------------------------------------------------------- #
# The catalogue table
# --------------------------------------------------------------------------- #


def test_catalog_table_shows_names_not_repository_ids(console: Console) -> None:
    console.print(models_ui.catalog_table(ENTRIES))
    text = rendered(console)
    assert "Qwen3 Coder 30B" in text
    assert "Model do pisania i poprawiania kodu." in text
    assert "18,5 GB" in text
    assert "Zmieści się" in text
    assert "zainstalowany" in text
    # no repository ids, no quantisation labels in the friendly view
    assert "acme/" not in text
    assert "Q4_K_M" not in text
    assert "GGUF" not in text


def test_catalog_table_column_headers_are_polish(console: Console) -> None:
    console.print(models_ui.catalog_table(ENTRIES))
    text = rendered(console)
    for header in ("nazwa", "do czego służy", "rozmiar", "pamięć", "stan"):
        assert header in text


def test_catalog_table_advanced_adds_the_repository_column(console: Console) -> None:
    console.print(models_ui.catalog_table(ENTRIES, advanced=True))
    text = rendered(console)
    assert "repozytorium" in text
    assert "acme/Qwen3-Coder-30B-GGUF" in text


def test_catalog_table_keeps_the_repository_readable_next_to_the_trust_column(
        console: Console) -> None:
    """The trust badge must not squeeze the repository id into an ellipsis.

    It did: adding the column pushed "acme/Qwen3-Coder-30B-GGUF" past the width
    of a normal terminal, so the advanced view stopped showing the one thing it
    exists for.  In the advanced view the badge therefore drops its word.
    """
    console.print(models_ui.catalog_table(ENTRIES, advanced=True))
    text = rendered(console)
    assert "acme/Qwen3-Coder-30B-GGUF" in text
    assert "oficjalne" not in text, "w widoku zaawansowanym zostaje sama liczba"


def test_catalog_table_marks_the_model_that_does_not_fit(console: Console) -> None:
    console.print(models_ui.catalog_table(ENTRIES))
    text = rendered(console)
    assert "Za duży" in text
    assert "Na styk" in text


def test_catalog_table_shows_badges_when_a_model_is_not_installed(console: Console) -> None:
    console.print(models_ui.catalog_table([ENTRIES[0]]))
    assert "narzędzia" in rendered(console)


def test_catalog_table_handles_an_empty_catalogue(console: Console) -> None:
    console.print(models_ui.catalog_table([]))
    assert "nazwa" in rendered(console)


def test_catalog_footer_names_hugging_face_discreetly(console: Console) -> None:
    console.print(models_ui.catalog_footer())
    text = rendered(console)
    assert "Katalog modeli: Hugging Face" in text
    assert "Modele pobierane z Hugging Face" in text


def test_catalog_summary_explains_the_memory_verdict(console: Console) -> None:
    console.print(models_ui.catalog_summary(ENTRIES[2]))
    text = rendered(console)
    assert "Qwen3 Coder Next 80B" in text
    assert "Za duży" in text
    assert "Potrzebuje ok. 56,2 GB" in text


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #


async def test_install_resolves_a_friendly_name_and_downloads(console: Console) -> None:
    catalog, backend = FakeCatalog(), FakeBackend()
    key = await models_ui.install_from_catalog(catalog, backend, console, "qwen3 coder 30b")
    assert key == "acme/Repo/model-Q4_K_M"
    assert backend.started == [("acme/Qwen3-Coder-30B-GGUF", None, ["model-Q4_K_M.gguf"])]
    assert catalog.install_calls == [("acme/Qwen3-Coder-30B-GGUF", None)]
    assert backend.refreshed is True
    text = rendered(console)
    assert "Qwen3 Coder 30B" in text
    assert "Pobrano" in text


async def test_install_passes_an_explicit_variant_through(console: Console) -> None:
    catalog, backend = FakeCatalog(), FakeBackend()
    await models_ui.install_from_catalog(catalog, backend, console, "Qwen3 Coder 30B",
                                         variant="Coder-Q6_K")
    assert catalog.install_calls == [("acme/Qwen3-Coder-30B-GGUF", "Coder-Q6_K")]
    assert backend.started[0][1] == "Coder-Q6_K"


async def test_install_reports_an_unknown_name_in_polish(console: Console) -> None:
    backend = FakeBackend()
    key = await models_ui.install_from_catalog(FakeCatalog(), backend, console, "nie ma takiego")
    assert key is None
    assert backend.started == []
    text = rendered(console)
    assert "Nie znaleziono modelu „nie ma takiego” w katalogu." in text
    assert "localcode models curated" in text


async def test_install_is_a_no_op_for_a_model_already_installed(console: Console) -> None:
    backend = FakeBackend()
    key = await models_ui.install_from_catalog(FakeCatalog(), backend, console, "Gemma 4 12B")
    assert key == "unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M"
    assert backend.started == []
    assert "jest już zainstalowany" in rendered(console)


async def test_install_asks_before_a_model_that_does_not_fit(console: Console,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(models_ui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": asked.append(prompt) or "n")
    backend = FakeBackend()
    key = await models_ui.install_from_catalog(FakeCatalog(), backend, console,
                                               "Qwen3 Coder Next 80B")
    assert key is None
    assert backend.started == []
    assert asked and "nie zmieści się w pamięci" in asked[0]
    assert "Anulowano." in rendered(console)


async def test_install_proceeds_when_the_user_confirms(console: Console,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_ui.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "tak")
    backend = FakeBackend()
    assert await models_ui.install_from_catalog(FakeCatalog(), backend, console,
                                                "Qwen3 Coder Next 80B")
    assert backend.started


async def test_install_skips_the_question_with_yes(console: Console,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("nie powinno pytać"))
    backend = FakeBackend()
    assert await models_ui.install_from_catalog(FakeCatalog(), backend, console,
                                                "Qwen3 Coder Next 80B", assume_yes=True)
    assert backend.started


async def test_install_refuses_a_too_large_model_without_a_terminal(
        console: Console, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models_ui.sys.stdin, "isatty", lambda: False)
    backend = FakeBackend()
    assert await models_ui.install_from_catalog(FakeCatalog(), backend, console,
                                                "Qwen3 Coder Next 80B") is None
    assert backend.started == []
    assert "użyj -y" in rendered(console)


async def test_install_returns_none_when_the_download_fails(console: Console) -> None:
    backend = FakeBackend(state="error")
    assert await models_ui.install_from_catalog(FakeCatalog(), backend, console,
                                                "Qwen3 Coder 30B") is None
    assert "Brak miejsca na dysku." in rendered(console)


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_models_parser_knows_install_search_and_curated() -> None:
    from localcode.cli import build_models_parser

    parser = build_models_parser()
    options = parser.parse_args(["install", "Qwen3", "Coder", "--variant", "Q6_K", "-y"])
    assert options.action == "install"
    assert options.name == ["Qwen3", "Coder"]
    assert options.variant == "Q6_K"
    assert options.yes is True

    assert parser.parse_args(["search", "qwen", "--advanced"]).advanced is True
    assert parser.parse_args(["curated", "--category", "obrazy"]).category == "obrazy"
    # the advanced path is still there
    assert parser.parse_args(["download", "org/repo", "Q4_K_M"]).quant == "Q4_K_M"


# --------------------------------------------------------------------------- #
# What to do with a freshly installed model
# --------------------------------------------------------------------------- #


def test_next_step_hint_never_offers_an_embedding_model_as_the_agent_model() -> None:
    """Regression: `models install` proposed `localcode -m <embeddings>`.

    Following that advice ends in a raw llama-server error ("the current
    context does not logits computation"), because an embedding model cannot
    generate text at all.
    """
    key = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0"
    hint = models_ui.next_step_hint({"key": key, "capabilities": ["embedding"]}, key)
    assert "localcode -m" not in hint
    assert "osadz" in hint.lower() or "rag" in hint.lower()


def test_next_step_hint_points_an_image_model_at_the_image_command() -> None:
    key = "second-state/stable-diffusion-v1-5-GGUF/sd15"
    hint = models_ui.next_step_hint({"key": key, "capabilities": ["image"]}, key)
    assert "localcode image" in hint


def test_next_step_hint_offers_a_language_model_to_the_agent() -> None:
    key = "acme/Repo/model-Q4_K_M"
    for model in ({"key": key, "capabilities": ["tools"]}, None):
        assert f"localcode -m {key}" in models_ui.next_step_hint(model, key)


# --------------------------------------------------------------------------- #
# Another quantisation of a model already on disk
# --------------------------------------------------------------------------- #


def installed_entry(quant: str, installed_key: str) -> dict[str, Any]:
    """A catalogue row whose model is on disk, at ``installed_key``'s quantisation."""
    entry = dict(ENTRIES[0])
    entry.update({"installed": True, "installedKey": installed_key, "quant": quant})
    return entry


async def test_a_different_quantisation_of_an_installed_model_is_downloaded(
        console: Console) -> None:
    """Choosing "najwyższa jakość" for something already on disk used to do nothing.

    A quantisation is a different file and a different trade-off; refusing to
    fetch it made the quality choice unreachable for any installed model.
    """
    catalog = FakeCatalog([installed_entry("Q8_0", "acme/Repo/model-Q4_K_M")])
    backend = FakeBackend()

    await models_ui.install_from_catalog(catalog, backend, console, "Qwen3 Coder 30B")

    assert backend.started, "nie rozpoczęto pobierania innej kwantyzacji"
    text = rendered(console)
    assert "Masz już wersję Q4_K_M" in text
    assert "Q8_0" in text


async def test_the_same_quantisation_is_not_downloaded_again(console: Console) -> None:
    catalog = FakeCatalog([installed_entry("Q4_K_M", "acme/Repo/model-Q4_K_M")])
    backend = FakeBackend()

    key = await models_ui.install_from_catalog(catalog, backend, console, "Qwen3 Coder 30B")

    assert key == "acme/Repo/model-Q4_K_M"
    assert backend.started == []
    assert "jest już zainstalowany" in rendered(console)


async def test_an_explicit_variant_still_reports_an_installed_model(console: Console) -> None:
    """With a variant named by hand the caller knows what they asked for."""
    catalog = FakeCatalog([installed_entry("Q8_0", "acme/Repo/model-Q4_K_M")])
    backend = FakeBackend()

    await models_ui.install_from_catalog(catalog, backend, console, "Qwen3 Coder 30B",
                                         variant="Coder-Q6_K")

    assert backend.started == []
    assert "jest już zainstalowany" in rendered(console)


# --------------------------------------------------------------------------- #
# /model — "Ulubione" first
# --------------------------------------------------------------------------- #


async def test_the_model_picker_lists_favourites_first_with_a_star(console: Console, monkeypatch) -> None:
    from localcode.config import Settings
    from tui_fakes import ScriptedBackend, make_model_dto

    models = [make_model_dto("acme/A-GGUF/A"), make_model_dto("acme/B-GGUF/B"),
              make_model_dto("remote:openai:gpt-5", remote=True, displayName="gpt-5")]
    backend = ScriptedBackend(models=models, model_key="acme/A-GGUF/A",
                              settings=Settings(favorite_models=["remote:openai:gpt-5", "acme/B-GGUF/B"]))
    seen: dict[str, Any] = {}

    async def fake_choose(choices, **kwargs):
        seen["choices"], seen["selected"] = choices, kwargs.get("selected")
        return None

    monkeypatch.setattr(models_ui, "choose", fake_choose)
    await models_ui.pick_model(backend, console)
    # Favourites first (by name among themselves), then the rest.
    assert [c.value for c in seen["choices"]] == ["acme/B-GGUF/B", "remote:openai:gpt-5", "acme/A-GGUF/A"]
    assert seen["choices"][0].label.startswith("★ ")
    assert seen["choices"][2].label.startswith("• ") and "★" not in seen["choices"][2].label
    # The cursor still starts on the model in use.
    assert seen["selected"] == 2
