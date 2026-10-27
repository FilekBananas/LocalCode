"""Tests for the in-app model store (``localcode.hub.catalog``).

Everything here runs against synthetic hub data — friendly naming,
categorisation, the "does it fit in this Mac" verdict, fuzzy name resolution,
automatic variant choice on a 32 GB and an 8 GB budget, installing through a
fake ``DownloadManager``, the offline fallback and the two HTTP routes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

from localcode.config import Settings
from localcode.hub.catalog import (
    CATEGORIES,
    _steps_badge,
    FIT_NO,
    FIT_OK,
    FIT_TIGHT,
    OFFLINE_NOTE,
    SOURCE_NOTE,
    Catalog,
    CatalogError,
    catalog_entries,
    catalog_settings_dto,
    display_name,
    fmt_gb,
    speed_hint,
)
from localcode.hub.types import LocalModel, ModelMeta

GB = 10**9


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def hub_model(repo: str, *, fmt: str = "gguf", params: float | None = 7.0,
              tags: list[str] | None = None, pipeline: str = "text-generation",
              gated: bool = False, ctx: int | None = None) -> dict[str, Any]:
    author, _, name = repo.partition("/")
    return {
        "id": repo, "author": author, "name": name, "format": fmt,
        "downloads": 1000, "likes": 10, "updatedAt": None,
        "tags": tags or [], "pipelineTag": pipeline, "paramsB": params,
        "activeParamsB": None, "gated": gated, "curated": False,
        "description": None, "arch": "qwen3", "contextLength": ctx,
        "authorInfo": {"verified": False, "followers": 100, "isOrg": False},
    }


def variant(vid: str, *, quant: str, gb: float, fits: str = "yes",
            files: list[str] | None = None, recommended: bool = False,
            projector: bool = False) -> dict[str, Any]:
    size = int(gb * GB)
    return {
        "id": vid, "label": quant, "quant": quant,
        "files": files or [f"{vid}.gguf"], "sizeBytes": size, "modelBytes": size,
        "isVisionProjector": projector, "complete": True, "recommended": recommended,
        "estimate": {"fits": fits, "message": f"{quant}: {fits}"},
    }


class FakeHub:
    """``HubClient`` stand-in: scripted search results and variant lists."""

    def __init__(self, models: list[dict[str, Any]] | None = None,
                 details: dict[str, dict[str, Any]] | None = None,
                 error: BaseException | None = None) -> None:
        self.models = models or []
        self._details = details or {}
        self.error = error
        self.searches: list[str] = []
        self.detail_calls: list[str] = []

    def search(self, query: str = "", format: str = "any", sort: str = "trending",
               limit: int = 30) -> list[dict[str, Any]]:
        if self.error is not None:
            raise self.error
        self.searches.append(query)
        return list(self.models)

    def details(self, repo: str, *, refresh: bool = False) -> dict[str, Any]:
        self.detail_calls.append(repo)
        if repo not in self._details:
            raise LookupError(repo)
        return self._details[repo]


class FakeLibrary:
    def __init__(self, models: list[LocalModel] | None = None) -> None:
        self.models = list(models or [])

    def list(self) -> list[LocalModel]:
        return list(self.models)


class FakeDownloads:
    """``DownloadManager`` stand-in recording what would be downloaded."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def start(self, repo: str, files: Any = None, *, label: str | None = None) -> dict[str, Any]:
        self.calls.append((repo, list(files or [])))
        return {"id": f"job-{len(self.calls)}", "repo": repo, "files": list(files or []),
                "label": label or repo, "state": "queued"}


def local_model(key: str, *, repo: str, name: str = "Model", size: int = 2 * GB,
                caps: list[str] | None = None) -> LocalModel:
    return LocalModel(
        key=key, format="gguf", path=f"/models/{key}", display_name=name,
        size_bytes=size, repo_id=repo, quant="Q4_K_M", capabilities=caps or [],
        meta=ModelMeta(arch="qwen3", params_b=4.0, context_length=32768),
    )


@pytest.fixture
def make_catalog(settings: Settings):
    def build(*, hub: Any = None, library: Any = None, downloads: Any = None,
              budget_gb: float = 32.0) -> Catalog:
        return Catalog(settings, hub=hub or FakeHub(), library=library or FakeLibrary(),
                       downloads=downloads, budget_bytes=int(budget_gb * GB))

    return build


# --------------------------------------------------------------------------- #
# Friendly names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("repo", "expected"), [
    ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3 4B Instruct"),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", "Qwen3 4B Instruct"),
    ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5 9B"),
    ("lmstudio-community/gemma-4-12B-it-MLX-4bit", "Gemma 4 12B Instruct"),
    ("Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3 VL 8B Instruct"),
    ("ibm-granite/granite-4.1-8b-GGUF", "Granite 4.1 8B"),
    ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "Mistral 7B Instruct v0.2"),
])
def test_display_name_strips_packaging_noise(repo: str, expected: str) -> None:
    assert display_name(repo) == expected


def test_display_name_uses_the_override_table() -> None:
    assert display_name("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF") == "Qwen3 Coder 30B"
    assert display_name("ggml-org/gpt-oss-20b-GGUF") == "GPT-OSS 20B"
    assert display_name("Runpod/FLUX.2-klein-4B-mflux-4bit") == "FLUX.2 klein 4B"


def test_display_name_survives_a_repo_without_an_organisation() -> None:
    assert display_name("weird-model-GGUF") == "Weird Model"


def test_fmt_gb_uses_a_polish_decimal_comma() -> None:
    assert fmt_gb(18_500_000_000) == "18,5 GB"
    assert fmt_gb(300_000_000) == "300 MB"
    assert fmt_gb(0) == "—"


@pytest.mark.parametrize(("count", "expected"), [
    (1, "1 krok"), (2, "2 kroki"), (4, "4 kroki"), (5, "5 kroków"),
    (12, "12 kroków"), (22, "22 kroki"), (25, "25 kroków"),
])
def test_steps_badge_uses_the_polish_plural(count: int, expected: str) -> None:
    assert _steps_badge(count) == expected


def test_speed_hint_follows_the_active_parameters() -> None:
    assert speed_hint(30.5, 3.3) == "Bardzo szybki"     # MoE: active params win
    assert speed_hint(30.5, None) != "Bardzo szybki"
    assert speed_hint(None, None).startswith("Szybkość")


# --------------------------------------------------------------------------- #
# Entries: shape, categorisation, fit
# --------------------------------------------------------------------------- #


def test_entry_shape_is_complete_and_names_hugging_face_as_the_source(make_catalog) -> None:
    catalog = make_catalog(hub=FakeHub([hub_model("acme/Widget-7B-GGUF")]))
    entry = next(e for e in catalog.entries() if e["id"] == "acme/Widget-7B-GGUF")
    assert set(entry) >= {
        "id", "name", "tagline", "category", "sizeLabel", "fit", "fitDetail", "speedHint",
        "installed", "installedKey", "recommendedVariant", "variants", "badges", "source",
        "requiresToken",
    }
    assert entry["source"] == {"provider": "Hugging Face", "repo": "acme/Widget-7B-GGUF"}
    assert entry["name"] == "Widget 7B"
    assert entry["tagline"]                      # never empty — it is the store subtitle
    assert entry["category"] in CATEGORIES


def test_curated_entries_keep_their_polish_note_as_the_tagline(make_catalog) -> None:
    # One row per model, and the MLX build of this one now wins the tie, so the
    # listing is asked for every packaging (as the advanced view does).
    catalog = make_catalog()
    entry = next(e for e in catalog.entries(deduplicate=False)
                 if e["id"] == "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")
    assert entry["curated"] is True
    assert "agent kodujący" in entry["tagline"]
    assert entry["category"] == "kodowanie"


@pytest.mark.parametrize(("repo", "pipeline", "category"), [
    ("acme/Zephyr-Coder-7B-GGUF", "text-generation", "kodowanie"),
    ("acme/Zephyr-Reasoning-7B-GGUF", "text-generation", "rozumowanie"),
    ("acme/Zephyr-Chat-7B-GGUF", "text-generation", "czat"),
    ("acme/Zephyr-Embedding-GGUF", "feature-extraction", "embedding"),
    ("acme/Zephyr-VL-7B-GGUF", "image-text-to-text", "wizja"),
    ("acme/Zephyr-Paint-GGUF", "text-to-image", "obrazy"),
])
def test_category_comes_from_the_pipeline_tag_and_the_name(
        make_catalog, repo: str, pipeline: str, category: str) -> None:
    catalog = make_catalog(hub=FakeHub([hub_model(repo, pipeline=pipeline)]))
    entry = next(e for e in catalog.entries() if e["id"] == repo)
    assert entry["category"] == category


def test_entries_reject_an_unknown_category(make_catalog) -> None:
    with pytest.raises(CatalogError) as excinfo:
        make_catalog().entries(category="coding")
    assert "Nieznana kategoria" in excinfo.value.message


def test_fit_verdicts_track_the_memory_budget(make_catalog) -> None:
    models = [hub_model("acme/Tiny-1B-GGUF", params=1.0),
              hub_model("acme/Huge-120B-GGUF", params=120.0)]
    big_mac = {e["id"]: e for e in make_catalog(hub=FakeHub(models), budget_gb=32).entries()}
    small_mac = {e["id"]: e for e in make_catalog(hub=FakeHub(models), budget_gb=8).entries()}

    assert big_mac["acme/Tiny-1B-GGUF"]["fit"] == FIT_OK
    assert big_mac["acme/Huge-120B-GGUF"]["fit"] == FIT_NO
    assert small_mac["acme/Huge-120B-GGUF"]["fit"] == FIT_NO
    # the verdict is always explained in Polish, with numbers
    assert "GB" in small_mac["acme/Huge-120B-GGUF"]["fitDetail"]


def test_fit_is_tight_between_seventy_percent_and_the_budget(make_catalog) -> None:
    # 10B at ~0.58 B/param ≈ 5.8 GB of weights; ×1.25 ≈ 7.25 GB — of a 8 GB budget.
    catalog = make_catalog(hub=FakeHub([hub_model("acme/Mid-10B-GGUF", params=10.0)]),
                           budget_gb=8)
    entry = next(e for e in catalog.entries() if e["id"] == "acme/Mid-10B-GGUF")
    assert entry["fit"] == FIT_TIGHT


def test_installed_models_are_marked_and_can_be_filtered(make_catalog) -> None:
    repo = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
    library = FakeLibrary([local_model(f"{repo}/Qwen3-Coder-Q4_K_M", repo=repo)])
    catalog = make_catalog(library=library)

    installed = catalog.entries(installed=True)
    assert [e["id"] for e in installed] == [repo]
    assert installed[0]["installed"] is True
    assert installed[0]["installedKey"] == f"{repo}/Qwen3-Coder-Q4_K_M"
    assert all(e["id"] != repo for e in catalog.entries(installed=False))


def test_installed_models_are_listed_first(make_catalog) -> None:
    repo = "ggml-org/embeddinggemma-300M-GGUF"
    catalog = make_catalog(library=FakeLibrary([local_model("k", repo=repo)]))
    assert catalog.entries()[0]["id"] == repo


def test_the_same_model_in_two_packagings_is_one_row(make_catalog) -> None:
    """MLX wins the tie on Apple Silicon; GGUF stays as the alternative."""
    entries = make_catalog().entries()
    coder = [e for e in entries if e["name"] == "Qwen3 Coder 30B"]
    assert len(coder) == 1
    assert coder[0]["format"] == "mlx"
    assert any("GGUF" in alt["repo"] for alt in coder[0]["alternatives"])
    # ...unless the advanced view asks for one row per repository
    raw = make_catalog().entries(deduplicate=False)
    assert len([e for e in raw if e["name"] == "Qwen3 Coder 30B"]) == 2


def test_the_preferred_packaging_is_a_setting(settings: Settings) -> None:
    settings.store.prefer_format = "gguf"
    catalog = Catalog(settings, hub=FakeHub(), library=FakeLibrary(),
                      budget_bytes=32 * GB)
    coder = [e for e in catalog.entries() if e["name"] == "Qwen3 Coder 30B"]
    assert coder and coder[0]["format"] == "gguf"


def test_query_matches_friendly_names_without_diacritics(make_catalog) -> None:
    hits = make_catalog().entries(query="qwen3 coder")
    assert hits and all("Coder" in e["name"] for e in hits)


def test_badges_are_polish_and_include_the_context_size(make_catalog) -> None:
    catalog = make_catalog(hub=FakeHub([
        hub_model("acme/Tooler-7B-GGUF", tags=["tools"], ctx=262144),
    ]))
    entry = next(e for e in catalog.entries() if e["id"] == "acme/Tooler-7B-GGUF")
    assert "narzędzia" in entry["badges"]
    assert "kontekst 262k" in entry["badges"]


def test_image_models_get_a_step_count_badge_with_a_polish_plural(make_catalog) -> None:
    entries = {e["name"]: e for e in make_catalog().entries(category="obrazy")}
    assert entries["FLUX.1 schnell"]["badges"] == ["4 kroki"]
    assert entries["Stable Diffusion 3.5 Medium"]["badges"] == ["28 kroków"]
    # tools/thinking/vision mean nothing for a text-to-image model
    assert all("wizja" not in e["badges"] for e in entries.values())


def test_gated_repos_require_a_token_only_without_one(make_catalog, settings: Settings) -> None:
    hub = FakeHub([hub_model("acme/Gated-7B-GGUF", gated=True)])
    entry = next(e for e in make_catalog(hub=hub).entries() if e["id"] == "acme/Gated-7B-GGUF")
    assert entry["requiresToken"] is True

    settings.hf_token = "hf_secret"
    entry = next(e for e in make_catalog(hub=hub).entries() if e["id"] == "acme/Gated-7B-GGUF")
    assert entry["requiresToken"] is False


def test_module_level_helper_mirrors_the_method(make_catalog, settings: Settings) -> None:
    catalog = make_catalog()
    assert catalog_entries(settings, catalog=catalog, category="kodowanie") == \
        catalog.entries(category="kodowanie")


# --------------------------------------------------------------------------- #
# Caching and offline
# --------------------------------------------------------------------------- #


def test_search_results_are_cached_so_the_ui_feels_instant(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Widget-7B-GGUF")])
    catalog = make_catalog(hub=hub)
    catalog.entries(query="widget")
    catalog.entries(query="widget")
    assert hub.searches == ["widget"]


def test_the_cache_expires(make_catalog, monkeypatch: pytest.MonkeyPatch) -> None:
    import localcode.hub.catalog as module

    hub = FakeHub([hub_model("acme/Widget-7B-GGUF")])
    catalog = make_catalog(hub=hub)
    catalog.entries(query="widget")
    monkeypatch.setattr(module, "SEARCH_TTL_S", -1.0)
    catalog.entries(query="widget")
    assert hub.searches == ["widget", "widget"]


def test_offline_falls_back_to_the_staff_picks_with_a_polish_explanation(make_catalog) -> None:
    catalog = make_catalog(hub=FakeHub(error=OSError("brak sieci")))
    entries = catalog.entries()
    assert entries                       # never an error screen
    assert all(e["curated"] for e in entries)
    assert catalog.offline is True
    assert catalog.notice == OFFLINE_NOTE
    assert "Brak połączenia" in catalog.notice


def test_offline_still_shows_what_is_already_installed(make_catalog) -> None:
    repo = "unsloth/Qwen3.5-9B-GGUF"
    catalog = make_catalog(hub=FakeHub(error=OSError("brak sieci")),
                           library=FakeLibrary([local_model("k", repo=repo)]))
    assert catalog.entries(installed=True)[0]["id"] == repo


def test_a_broken_library_does_not_hide_the_store(make_catalog) -> None:
    class Broken:
        def list(self) -> list[LocalModel]:
            raise RuntimeError("uszkodzony katalog modeli")

    assert make_catalog(library=Broken()).entries()


# --------------------------------------------------------------------------- #
# resolve_name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("text", "expected_name"), [
    ("qwen3 coder", "Qwen3 Coder 30B"),
    ("Qwen3 Coder 30B", "Qwen3 Coder 30B"),
    ("QWEN3 CODER 30B", "Qwen3 Coder 30B"),
    ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3 Coder 30B"),
    ("gpt oss", "GPT-OSS 20B"),
    ("devstral", "Devstral Small 2 24B"),
])
def test_resolve_name_matches_names_ids_and_aliases(make_catalog, text: str,
                                                    expected_name: str) -> None:
    entry = make_catalog().resolve_name(text)
    assert entry is not None and entry["name"] == expected_name


def test_resolve_name_finds_an_image_model_by_family(make_catalog) -> None:
    entry = make_catalog().resolve_name("flux")
    assert entry is not None and entry["category"] == "obrazy"


def test_resolve_name_returns_none_for_nonsense(make_catalog) -> None:
    assert make_catalog().resolve_name("nie ma takiego modelu") is None
    assert make_catalog().resolve_name("") is None


# --------------------------------------------------------------------------- #
# Variant choice and install
# --------------------------------------------------------------------------- #


def _coder_details(*, fits_for: set[str]) -> dict[str, Any]:
    """Variants of a fictional coder repo; ``fits_for`` names the ones that fit."""
    specs = [("Coder-Q4_K_M", "Q4_K_M", 3.0), ("Coder-Q6_K", "Q6_K", 5.0),
             ("Coder-Q8_0", "Q8_0", 8.0)]
    variants = [
        variant(vid, quant=q, gb=gb, fits="yes" if vid in fits_for else "no")
        for vid, q, gb in specs
    ]
    # the hub marks the best quality that fits (see search.pick_recommended)
    best = next((v for v in reversed(variants) if v["estimate"]["fits"] == "yes"), None)
    if best is not None:
        best["recommended"] = True
    return {"model": hub_model("acme/Coder-7B-GGUF"), "variants": variants,
            "capabilities": ["tools"], "contextLength": 32768, "accessError": None}


def test_install_takes_the_configured_quality_when_it_fits(make_catalog) -> None:
    """Four-bit is the default: it is the level at which the big models fit.

    The store used to take the best quality the Mac could hold, which meant a
    7B model arrived as Q8_0 — twice the download for a difference most people
    never hear.  The level is now a setting, and this is its default.
    """
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(
                      fits_for={"Coder-Q4_K_M", "Coder-Q6_K", "Coder-Q8_0"})})
    downloads = FakeDownloads()
    job = make_catalog(hub=hub, downloads=downloads, budget_gb=32).install("Coder 7B")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q4_K_M.gguf"])]
    assert job["state"] == "queued"


def test_install_follows_a_higher_quality_setting(settings: Settings) -> None:
    settings.store.quality = "8bit"
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(
                      fits_for={"Coder-Q4_K_M", "Coder-Q6_K", "Coder-Q8_0"})})
    downloads = FakeDownloads()
    Catalog(settings, hub=hub, library=FakeLibrary(), downloads=downloads,
            budget_bytes=32 * GB).install("Coder 7B")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q8_0.gguf"])]


def test_a_quality_that_does_not_fit_gives_way_to_one_that_does(settings: Settings) -> None:
    """The setting is a preference, not an instruction to fill the memory."""
    settings.store.quality = "8bit"
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q4_K_M"})})
    downloads = FakeDownloads()
    Catalog(settings, hub=hub, library=FakeLibrary(), downloads=downloads,
            budget_bytes=8 * GB).install("Coder 7B")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q4_K_M.gguf"])]


def test_install_drops_to_a_smaller_quantisation_on_an_8_gb_mac(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q4_K_M"})})
    downloads = FakeDownloads()
    make_catalog(hub=hub, downloads=downloads, budget_gb=8).install("Coder 7B")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q4_K_M.gguf"])]


def test_install_takes_the_smallest_variant_when_nothing_fits(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for=set())})
    downloads = FakeDownloads()
    make_catalog(hub=hub, downloads=downloads, budget_gb=4).install("Coder 7B")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q4_K_M.gguf"])]


def test_install_adds_the_vision_projector(make_catalog) -> None:
    details = {
        "model": hub_model("acme/Seer-8B-GGUF"),
        "variants": [
            variant("Seer-Q4_K_M", quant="Q4_K_M", gb=4.0, recommended=True),
            variant("mmproj-F16", quant="F16", gb=0.5, files=["mmproj-F16.gguf"],
                    recommended=True, projector=True),
        ],
        "capabilities": ["vision"], "contextLength": 32768, "accessError": None,
    }
    hub = FakeHub([hub_model("acme/Seer-8B-GGUF")], {"acme/Seer-8B-GGUF": details})
    downloads = FakeDownloads()
    make_catalog(hub=hub, downloads=downloads).install("Seer 8B")
    assert downloads.calls == [("acme/Seer-8B-GGUF",
                                ["Seer-Q4_K_M.gguf", "mmproj-F16.gguf"])]


def test_install_of_an_image_model_takes_the_companion_files(make_catalog) -> None:
    downloads = FakeDownloads()
    make_catalog(downloads=downloads).install("Stable Diffusion 3.5 Medium")
    repo, files = downloads.calls[0]
    assert repo == "second-state/stable-diffusion-3.5-medium-GGUF"
    # diffusion weights are never recommended below Q6_K (SD 1.5 at Q4_0 renders a
    # blurred colour field); the T5 text encoder tolerates Q4_0 and stays small
    assert files == ["sd3.5_medium-Q8_0.gguf", "clip_l-Q8_0.gguf",
                     "clip_g-Q8_0.gguf", "t5xxl-Q4_0.gguf"]


def test_install_accepts_an_explicit_variant(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q4_K_M"})})
    downloads = FakeDownloads()
    make_catalog(hub=hub, downloads=downloads).install("Coder 7B", variant="Coder-Q6_K")
    assert downloads.calls == [("acme/Coder-7B-GGUF", ["Coder-Q6_K.gguf"])]


def test_install_reports_an_unknown_model_in_polish(make_catalog) -> None:
    with pytest.raises(CatalogError) as excinfo:
        make_catalog(downloads=FakeDownloads()).install("model którego nie ma")
    assert excinfo.value.status == 404
    assert "Nie znaleziono modelu" in excinfo.value.message


def test_install_reports_an_unknown_variant_in_polish(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q8_0"})})
    with pytest.raises(CatalogError) as excinfo:
        make_catalog(hub=hub, downloads=FakeDownloads()).install("Coder 7B", variant="nope")
    assert excinfo.value.status == 404
    assert "wariantu" in excinfo.value.message


def test_install_without_a_download_manager_explains_itself(make_catalog) -> None:
    with pytest.raises(CatalogError) as excinfo:
        make_catalog().install("Qwen3 Coder 30B")
    assert excinfo.value.status == 503
    assert "niedostępne" in excinfo.value.message


def test_detail_fills_the_variant_list_for_the_advanced_sheet(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q4_K_M"})})
    entry = make_catalog(hub=hub, budget_gb=8).detail("Coder 7B")
    assert [v["label"] for v in entry["variants"]] == ["Q4_K_M", "Q6_K", "Q8_0"]
    assert entry["recommendedVariant"]["label"] == "Q4_K_M"
    assert entry["recommendedVariant"]["files"] == ["Coder-Q4_K_M.gguf"]
    assert entry["fit"] == FIT_OK          # taken from the hub's own estimate
    assert "narzędzia" in entry["badges"]


def test_every_variant_says_whether_it_fits_for_the_install_chooser(make_catalog) -> None:
    """"Zainstaluj" lets the person pick the quantisation; each option carries
    its own memory verdict, and exactly one is the store's own choice."""
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")],
                  {"acme/Coder-7B-GGUF": _coder_details(fits_for={"Coder-Q4_K_M"})})
    entry = make_catalog(hub=hub, budget_gb=8).detail("Coder 7B")
    variants = {v["label"]: v for v in entry["variants"]}
    assert [v["label"] for v in entry["variants"] if v["recommended"]] == ["Q4_K_M"]
    assert all(v["fit"] in (FIT_OK, FIT_TIGHT, FIT_NO) and v["fitDetail"] for v in variants.values())
    sizes = sorted(variants.values(), key=lambda v: v["sizeBytes"])
    order = {FIT_OK: 0, FIT_TIGHT: 1, FIT_NO: 2}
    assert [order[v["fit"]] for v in sizes] == sorted(order[v["fit"]] for v in sizes)


def test_detail_maps_a_hub_failure_to_a_polish_message(make_catalog) -> None:
    hub = FakeHub([hub_model("acme/Coder-7B-GGUF")])
    with pytest.raises(CatalogError) as excinfo:
        make_catalog(hub=hub).detail("Coder 7B")
    assert excinfo.value.status == 502
    assert "Nie udało się pobrać informacji" in excinfo.value.message


def test_settings_dto_carries_the_provenance_note() -> None:
    dto = catalog_settings_dto()
    assert dto["sourceNote"] == SOURCE_NOTE == "Modele pobierane z Hugging Face"
    assert dto["offlineNote"] == OFFLINE_NOTE
    assert [c["id"] for c in dto["categories"]] == list(CATEGORIES)


# --------------------------------------------------------------------------- #
# HTTP routes
# --------------------------------------------------------------------------- #

# The routes are exercised against a real ``Core`` whose library, runtime, hub
# and download manager are the fakes the server tests already use.
for _extra in ("server", "chat"):  # pragma: no cover - import plumbing
    _path = str(Path(__file__).resolve().parent.parent / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from localcode.server.app import create_app  # noqa: E402
from localcode.server.core import Core  # noqa: E402

from chat_fakes import make_service  # noqa: E402
from server_fakes import (  # noqa: E402
    CHAT_KEY,
    FakeDownloads as ServerDownloads,
    FakeEngine,
    FakeHub as ServerHub,
    FakeLibrary as ServerLibrary,
    FakeRuntime,
    make_model,
)

TOKEN = "test-token-12345"
#: A curated repository, so the catalogue can report it as installed.
INSTALLED_REPO = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"


@pytest.fixture
async def core(settings: Settings, tmp_path: Path):
    """A ``Core`` with one curated model already on disk."""
    library = ServerLibrary([
        make_model(f"{INSTALLED_REPO}/Qwen3-Coder-30B-Q4_K_M", name="Qwen3 Coder 30B"),
        make_model(CHAT_KEY, name="Qwen3 0.6B"),
    ])
    runtime = FakeRuntime({m.key: FakeEngine() for m in library.models})
    instance = Core(settings, db_path=tmp_path / "chat.db", library=library,
                    hub=ServerHub(), runtime=runtime)
    instance.chat = make_service(instance.repo, runtime, library, settings)
    instance._downloads = ServerDownloads(settings, instance._on_download_event)  # noqa: SLF001
    instance._downloads_built = True                                              # noqa: SLF001
    await instance.start()
    try:
        yield instance
    finally:
        await instance.shutdown()


@pytest.fixture
async def client(core: Core):
    test_client = TestClient(TestServer(create_app(core, TOKEN)),
                             headers={"Authorization": f"Bearer {TOKEN}"})
    await test_client.start_server()
    try:
        yield test_client
    finally:
        await test_client.close()


async def test_get_api_catalog_returns_friendly_entries(client) -> None:
    response = await client.get("/api/catalog?category=kodowanie")
    assert response.status == 200
    entries = await response.json()
    assert entries and all(e["category"] == "kodowanie" for e in entries)
    assert all(e["source"]["provider"] == "Hugging Face" for e in entries)
    assert all("name" in e and "tagline" in e and "fit" in e for e in entries)


async def test_get_api_catalog_filters_by_installed_and_query(client) -> None:
    response = await client.get("/api/catalog?installed=true")
    assert response.status == 200
    entries = await response.json()
    assert entries and [e["installed"] for e in entries] == [True] * len(entries)
    assert any(e["id"] == INSTALLED_REPO for e in entries)

    response = await client.get("/api/catalog?q=qwen3%20coder")
    assert response.status == 200
    assert all("Coder" in e["name"] for e in await response.json())


async def test_get_api_catalog_pages_and_counts(client) -> None:
    """``limit`` caps the page and ``X-Catalog-Total`` says how many matched."""
    response = await client.get("/api/catalog?limit=3")
    assert response.status == 200
    entries = await response.json()
    assert len(entries) <= 3
    total = int(response.headers["X-Catalog-Total"])
    assert total >= len(entries)


async def test_get_api_catalog_carries_trust_and_publisher(client) -> None:
    response = await client.get("/api/catalog?limit=5")
    entries = await response.json()
    assert entries
    for entry in entries:
        assert 0 <= entry["trust"]["score"] <= 100
        assert entry["trust"]["label"]
        assert entry["publisher"]["kind"] in ("official", "trusted", "community", "unknown")


async def test_get_api_catalog_filters_by_trust_and_publisher(client) -> None:
    everything = await (await client.get("/api/catalog?limit=200")).json()
    official = await (await client.get("/api/catalog?official=1&limit=200")).json()
    assert all(e["publisher"]["official"] for e in official)
    assert len(official) <= len(everything)

    strict = await (await client.get("/api/catalog?minTrust=100&limit=200")).json()
    assert all(e["trust"]["score"] >= 100 for e in strict)


async def test_get_api_catalog_clamps_out_of_range_numbers(client) -> None:
    """Numeric parameters are clamped, as everywhere else in this API; only a
    value that is not a number at all is an error."""
    assert (await client.get("/api/catalog?limit=0")).status == 200
    assert len(await (await client.get("/api/catalog?limit=0")).json()) == 1
    assert (await client.get("/api/catalog?minTrust=101")).status == 200
    assert (await client.get("/api/catalog?limit=abc")).status == 400


async def test_get_api_catalog_rejects_a_bad_category(client) -> None:
    response = await client.get("/api/catalog?category=coding")
    assert response.status == 400
    assert "Nieznana kategoria" in (await response.json())["error"]


async def test_post_api_catalog_install_starts_a_download(client, core) -> None:
    response = await client.post("/api/catalog/install",
                                 json={"id": "Stable Diffusion 1.5"})
    assert response.status == 201
    job = await response.json()
    assert job["repo"] == "second-state/stable-diffusion-v1-5-GGUF"
    assert job["files"] == ["stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"]
    assert core.downloads.jobs


async def test_post_api_catalog_install_needs_an_id(client) -> None:
    response = await client.post("/api/catalog/install", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_id"


async def test_post_api_catalog_install_reports_an_unknown_model(client) -> None:
    response = await client.post("/api/catalog/install", json={"id": "nie ma takiego"})
    assert response.status == 404
    assert "Nie znaleziono modelu" in (await response.json())["error"]


async def test_app_settings_expose_the_hugging_face_note(client) -> None:
    response = await client.get("/api/app-settings")
    assert response.status == 200
    body = await response.json()
    assert body["catalog"]["sourceNote"] == "Modele pobierane z Hugging Face"
    assert body["catalog"]["offlineNote"]
    assert body["hfTokenSet"] is False          # the existing DTO is untouched


async def test_hub_routes_stay_as_the_advanced_view(client) -> None:
    response = await client.get("/api/hub/curated")
    assert response.status == 200
    assert all("id" in m for m in await response.json())


# --------------------------------------------------------------------------- #
# Against the real Hugging Face hub
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_the_real_hub_produces_friendly_entries(settings: Settings) -> None:
    """A live search still comes back as a store listing, not as repo ids."""
    catalog = Catalog(settings, budget_bytes=32 * GB)
    entries = catalog.entries(query="qwen coder")
    assert entries
    assert not catalog.offline
    for entry in entries:
        assert entry["name"] and "/" not in entry["name"]
        assert entry["tagline"]
        assert entry["category"] in CATEGORIES
        assert entry["fit"] in (FIT_OK, FIT_TIGHT, FIT_NO, "Nie wiadomo")
        assert entry["source"]["provider"] == "Hugging Face"


@pytest.mark.network
def test_the_real_hub_resolves_a_curated_name_to_installable_files(settings: Settings) -> None:
    catalog = Catalog(settings, budget_bytes=32 * GB)
    entry = catalog.resolve_name("qwen3 coder")
    assert entry is not None
    files = catalog.install_files(entry)
    assert files and all(f.lower().endswith((".gguf", ".safetensors", ".json"))
                         or "." in f for f in files)


# --------------------------------------------------------------------------- #
# Companion files must never be offered as the model itself
# --------------------------------------------------------------------------- #


def test_auxiliary_gguf_files_are_not_variants() -> None:
    """A repo shipping a speculative-decoding head must not offer it for install.

    ``ggml-org/gpt-oss-20b-GGUF`` ships an ~0.9 GB EAGLE3 draft head next to the
    12 GB weights; it used to win "smallest variant that fits" and the store then
    showed 1,1 GB on the detail sheet for a 12 GB model.
    """
    from localcode.hub.search import group_gguf_variants

    variants = group_gguf_variants([
        ("gpt-oss-20b-MXFP4.gguf", 12_110_000_000),
        ("eagle3-gpt-oss-20b-Q8_0.gguf", 921_488_000),
        ("mmproj-F16.gguf", 600_000_000),
    ])
    ids = [v["id"] for v in variants]
    assert "gpt-oss-20b-MXFP4" in ids
    assert not any("eagle3" in i for i in ids)
    # the vision projector still shows up, flagged as such
    assert [v["isVisionProjector"] for v in variants if v["id"].startswith("mmproj")] == [True]


class QueryHub(FakeHub):
    """Hub whose models only show up when they are actually searched for.

    The real Hugging Face search behaves this way: the default (query-less)
    listing is a trending sample, so a model the user found with
    ``localcode models search`` is usually not in it.
    """

    def search(self, query: str = "", format: str = "any", sort: str = "trending",
               limit: int = 30) -> list[dict[str, Any]]:
        self.searches.append(query)
        if not (query or "").strip():
            return []
        def flat(text: str) -> str:
            return re.sub(r"[^a-z0-9.]+", "", text.lower())

        needle = flat(query)
        return [m for m in self.models if needle in flat(str(m["id"]))]


def test_resolve_name_searches_the_hub_for_a_model_outside_the_default_listing(
        make_catalog) -> None:
    """`models install` must reach anything `models search` can show.

    Regression: ``localcode models search "Qwen3-1.7B"`` listed the model but
    ``localcode models install "Qwen3 1.7B"`` (and even the full repository id)
    answered "Nie znaleziono modelu … w katalogu", because resolution only
    looked at the query-less listing (staff picks + trending).
    """
    hub = QueryHub([hub_model("Qwen/Qwen3-1.7B-GGUF", params=1.7)])
    catalog = make_catalog(hub=hub)
    by_repo = catalog.resolve_name("Qwen/Qwen3-1.7B-GGUF")
    assert by_repo is not None and by_repo["id"] == "Qwen/Qwen3-1.7B-GGUF"
    by_name = catalog.resolve_name("Qwen3 1.7B")
    assert by_name is not None and by_name["id"] == "Qwen/Qwen3-1.7B-GGUF"


# --------------------------------------------------------------------------- #
# A full repository id must land on that repository
# --------------------------------------------------------------------------- #


def test_an_exact_repository_id_beats_a_fuzzy_match(settings) -> None:
    """Reproduced by installing the wrong video model.

    Every word of ``city96/LTX-Video-gguf`` also appears in
    ``city96/LTX-Video-0.9.6-distilled-gguf``, so the fuzzy pass over the default
    listing answered first and the wrong checkpoint was downloaded.
    """
    from localcode.hub.catalog import Catalog

    catalog = Catalog(settings)
    for repo in ("city96/LTX-Video-gguf", "city96/LTX-Video-0.9.6-distilled-gguf"):
        entry = catalog.resolve_name(repo)
        assert entry is not None and entry["id"] == repo, repo


def test_a_friendly_name_still_resolves_fuzzily(make_catalog) -> None:
    # Offline: an anonymous upload from a live search is out of the store now
    # (unknown creator), so the hub here offers a trusted conversion.
    hub = FakeHub([hub_model("city96/LTX-Video-gguf", pipeline="text-to-video", params=2.0),
                   hub_model("someone/ltx-video-test", pipeline="text-to-video", params=2.0)])
    entry = make_catalog(hub=hub).resolve_name("ltx video")
    assert entry is not None and entry["id"] == "city96/LTX-Video-gguf"


@pytest.mark.parametrize("listed, measured, expected", [
    (4_608_176_543, 3_291, 0),                  # FLUX.2 klein: only the top-level JSON was measured
    (4_608_176_543, 4_500_000_000, 4_500_000_000),
    (8_000_000_000, 2_200_000_000, 2_200_000_000),   # a 2-bit build of an 8-bit listing
    (0, 3_000_000_000, 3_000_000_000),          # nothing listed: the measurement is all we have
    (4_000_000_000, 0, 0),
])
def test_a_failed_size_measurement_does_not_replace_the_listed_size(listed, measured, expected) -> None:
    from localcode.hub.catalog import trustworthy_size

    assert trustworthy_size(listed, measured) == expected
