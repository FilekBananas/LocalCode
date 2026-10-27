"""The store listing over the shipped index (``Catalog.entries``).

Before the index existed the store showed sixty staff picks plus whatever a live
Hugging Face search returned, which meant an empty shop whenever the network was
down.  These tests cover the join: index rows become full catalogue entries,
they carry a trust score and a publisher, the listing can be paged and filtered,
and the things that must never fall out of it — installed models and staff picks
— stay in regardless of the filters.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from localcode.config import Settings
from localcode.hub import store_data
from localcode.hub.catalog import Catalog
from localcode.hub.types import LocalModel, ModelMeta

from tests.hub.test_catalog import FakeDownloads, FakeHub, FakeLibrary, hub_model

GB = 10**9


def index_row(repo: str, **kw: Any) -> dict[str, Any]:
    entry = {
        "repo": repo,
        "format": "gguf",
        "author": repo.partition("/")[0],
        "name": repo.partition("/")[2].replace("-", " "),
        "category": "czat",
        "tagline": "Model do rozmowy i pisania tekstów.",
        "paramsB": 8.0,
        "quant": "Q4_K_M",
        "sizeBytes": 4 * GB,
        "downloads": 1_000,
        "likes": 10,
        "license": "apache-2.0",
        "files": {"known": True, "safe": True, "pickle": [], "code": []},
        "caps": ["tools"],
        "contextLength": 32768,
        "provenance": "trusted",
        "publisher": "Qwen (Alibaba)",
        "baseModel": "Qwen/Qwen3-8B",
        "rank": 0,
        "authorInfo": {"verified": True, "followers": 100, "isOrg": False},
    }
    entry.update(kw)
    return entry


@pytest.fixture
def with_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def build(models: list[dict[str, Any]]) -> None:
        path = tmp_path / "catalog.json.gz"
        with gzip.open(path, "wb") as fh:
            fh.write(json.dumps({"version": 1, "generatedAt": "2026-09-26T00:00:00Z",
                                 "models": models}, ensure_ascii=False).encode("utf-8"))
        monkeypatch.setattr(store_data, "DATA_PATH", path)
        store_data.reset_cache()
    yield build
    store_data.reset_cache()


@pytest.fixture
def catalog(settings: Settings):
    def build(*, hub: Any = None, library: Any = None, downloads: Any = None,
              budget_gb: float = 32.0) -> Catalog:
        return Catalog(settings, hub=hub or FakeHub(), library=library or FakeLibrary(),
                       downloads=downloads, budget_bytes=int(budget_gb * GB))
    return build


def by_id(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in entries}


# --------------------------------------------------------------------------- #
# The join
# --------------------------------------------------------------------------- #


def test_index_models_appear_in_the_listing(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF", name="Qwen3 8B")])
    entries = catalog().entries()
    entry = by_id(entries)["unsloth/Qwen3-8B-GGUF"]
    assert entry["name"] == "Qwen3 8B"
    assert entry["tagline"] == "Model do rozmowy i pisania tekstów."
    assert entry["category"] == "czat"
    assert entry["sizeBytes"] == 4 * GB


def test_every_entry_carries_a_trust_score_and_a_publisher(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    entry = by_id(catalog().entries())["unsloth/Qwen3-8B-GGUF"]
    assert 0 <= entry["trust"]["score"] <= 100
    assert entry["trust"]["reasons"]
    assert entry["publisher"]["kind"] == "trusted"
    assert entry["publisher"]["publisher"] == "Qwen (Alibaba)"
    assert entry["publisher"]["official"] is False


def test_the_laboratorys_own_repository_is_marked_official(with_index, catalog) -> None:
    with_index([index_row("Qwen/Qwen3-8B-GGUF", author="Qwen", provenance="official",
                          baseModel=None)])
    entry = by_id(catalog().entries())["Qwen/Qwen3-8B-GGUF"]
    assert entry["publisher"]["official"] is True
    assert entry["publisher"]["label"] == "źródło oficjalne"


def test_a_caveat_from_the_index_reaches_the_card(with_index, catalog) -> None:
    with_index([index_row("nick/Model-Abliterated-GGUF",
                          watchOut="Model bez wbudowanych filtrów — odpowie na wszystko.")])
    entry = by_id(catalog().entries())["nick/Model-Abliterated-GGUF"]
    assert entry["watchOut"].startswith("Model bez wbudowanych filtrów")


def test_capabilities_from_the_index_become_badges(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF", caps=["tools", "thinking", "vision"])])
    entry = by_id(catalog().entries())["unsloth/Qwen3-8B-GGUF"]
    assert "narzędzia" in entry["badges"]
    assert "myślenie" in entry["badges"]
    assert any(b.startswith("kontekst") for b in entry["badges"])


# --------------------------------------------------------------------------- #
# Paging and filters
# --------------------------------------------------------------------------- #


def test_limit_caps_the_page_and_total_counts_the_rest(with_index, catalog) -> None:
    with_index([index_row(f"org/model-{i}", rank=i, downloads=1000 - i) for i in range(40)])
    store = catalog()
    page = store.entries(limit=10)
    assert len(page) == 10
    assert store.total_matches > 10


def test_min_trust_hides_the_bottom_of_the_index(with_index, catalog) -> None:
    with_index([
        index_row("Qwen/Good-GGUF", author="Qwen", provenance="official",
                  downloads=2_000_000, likes=900, baseModel=None),
        index_row("nick/Doubtful-GGUF", author="nick", provenance="community", downloads=3,
                  likes=0, license=None, baseModel=None),
    ])
    store = catalog()
    all_ids = set(by_id(store.entries()))
    filtered = set(by_id(store.entries(min_trust=60)))
    assert "nick/Doubtful-GGUF" in all_ids
    assert "nick/Doubtful-GGUF" not in filtered
    assert "Qwen/Good-GGUF" in filtered


def test_official_only_keeps_the_laboratories(with_index, catalog) -> None:
    with_index([
        index_row("Qwen/Own-GGUF", author="Qwen", provenance="official", baseModel=None),
        index_row("bartowski/Converted-GGUF", author="bartowski"),
    ])
    ids = set(by_id(catalog().entries(official_only=True)))
    assert "Qwen/Own-GGUF" in ids
    assert "bartowski/Converted-GGUF" not in ids


def test_query_matches_index_rows_without_a_network(with_index, catalog) -> None:
    hub = FakeHub(error=OSError("brak sieci"))
    with_index([
        # A name of its own: a row whose display name matches a staff pick is
        # merged into it by _deduplicate, which is a different behaviour.
        index_row("testorg/Alpaka-Coder-9B-GGUF", name="Alpaka Coder 9B", category="kodowanie"),
        index_row("testorg/Alpaka-Chat-9B-GGUF", name="Alpaka Chat 9B", publisher="Google",
                  baseModel=None),
    ])
    found = by_id(catalog(hub=hub).entries(query="alpaka coder"))
    assert "testorg/Alpaka-Coder-9B-GGUF" in found
    assert "testorg/Alpaka-Chat-9B-GGUF" not in found


def test_category_filter_uses_the_index_category(with_index, catalog) -> None:
    with_index([
        index_row("a/coder-GGUF", category="kodowanie"),
        index_row("b/chat-GGUF", category="czat"),
    ])
    # The staff picks are always part of the listing, so this asserts the
    # index rows are filtered, not that nothing else is there.
    ids = set(by_id(catalog().entries(category="kodowanie")))
    assert "a/coder-GGUF" in ids
    assert "b/chat-GGUF" not in ids


# --------------------------------------------------------------------------- #
# What must never disappear
# --------------------------------------------------------------------------- #


def test_installed_models_are_listed_even_when_the_index_omits_them(with_index, catalog) -> None:
    local = LocalModel(key="me/Private-GGUF/model", format="gguf", path="/models/x",
                       display_name="Mój model", size_bytes=3 * GB, repo_id="me/Private-GGUF",
                       quant="Q4_K_M", meta=ModelMeta(arch="qwen3", params_b=7.0))
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    entries = catalog(library=FakeLibrary([local])).entries()
    entry = by_id(entries)["me/Private-GGUF"]
    assert entry["installed"] is True
    assert entries[0]["id"] == "me/Private-GGUF", "zainstalowane idą na górę"


def test_staff_picks_stay_in_the_listing(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    entries = catalog().entries()
    assert any(e.get("curated") for e in entries)


def test_a_staff_pick_that_is_also_in_the_index_is_still_marked_verified(
        with_index, catalog) -> None:
    from localcode.hub.curated import CURATED

    pick = CURATED[0]["repo"]
    with_index([index_row(pick, author=pick.partition("/")[0])])
    entry = by_id(catalog().entries())[pick]
    assert entry["curated"] is True
    assert any("LocalCode" in r for r in entry["trust"]["reasons"])


# --------------------------------------------------------------------------- #
# Excluded models
# --------------------------------------------------------------------------- #


def test_a_row_with_pickle_files_is_not_listed(with_index, catalog) -> None:
    with_index([
        index_row("unsloth/Qwen3-8B-GGUF"),
        index_row("nick/Risky-GGUF", files={"known": True, "safe": False, "pickle": [".bin"], "code": []}),
    ])
    ids = set(by_id(catalog().entries()))
    assert "unsloth/Qwen3-8B-GGUF" in ids
    assert "nick/Risky-GGUF" not in ids


def test_a_row_from_an_unknown_creator_is_not_listed(with_index, catalog) -> None:
    with_index([
        index_row("unsloth/Qwen3-8B-GGUF"),
        index_row("anon/Mystery-GGUF", author="anon", provenance="unknown",
                  baseModel="anon/Mystery-Base", authorInfo=None),
    ])
    ids = set(by_id(catalog().entries()))
    assert "unsloth/Qwen3-8B-GGUF" in ids
    assert "anon/Mystery-GGUF" not in ids


def test_an_unknown_creator_with_verified_author_info_is_listed(with_index, catalog) -> None:
    with_index([
        index_row("unsloth/Qwen3-8B-GGUF"),
        index_row("verified-user/Model-GGUF", author="verified-user", provenance="unknown",
                  baseModel="verified-user/Model-Base",
                  authorInfo={"verified": True, "followers": 100, "isOrg": False}),
    ])
    ids = set(by_id(catalog().entries()))
    assert "unsloth/Qwen3-8B-GGUF" in ids
    assert "verified-user/Model-GGUF" in ids


def test_an_excluded_model_that_is_installed_is_still_listed(with_index, catalog) -> None:
    local = LocalModel(key="anon/Risky-GGUF/model", format="gguf", path="/models/x",
                       display_name="Ryzykowany model", size_bytes=3 * GB,
                       repo_id="anon/Risky-GGUF",
                       quant="Q4_K_M", meta=ModelMeta(arch="qwen3", params_b=7.0))
    with_index([
        index_row("unsloth/Qwen3-8B-GGUF"),
        index_row("anon/Risky-GGUF", files={"known": True, "safe": False, "pickle": [".bin"], "code": []}),
    ])
    entries = catalog(library=FakeLibrary([local])).entries()
    ids = set(by_id(entries))
    assert "unsloth/Qwen3-8B-GGUF" in ids
    assert "anon/Risky-GGUF" in ids
    entry = by_id(entries)["anon/Risky-GGUF"]
    assert entry["installed"] is True


def test_installing_an_excluded_entry_raises_error(with_index, catalog) -> None:
    from localcode.hub.catalog import CatalogError
    with_index([
        index_row("nick/Risky-GGUF", files={"known": True, "safe": False, "pickle": [".bin"], "code": []}),
    ])
    store = catalog(hub=FakeHub([]), downloads=FakeDownloads())
    with pytest.raises(CatalogError) as exc_info:
        store.install("nick/Risky-GGUF")
    assert exc_info.value.code == "rejected_model"
    assert exc_info.value.status == 409


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_among_equals_the_more_used_model_comes_first(with_index, catalog) -> None:
    with_index([
        index_row("org/rare-GGUF", name="Rzadki", downloads=100, rank=5),
        index_row("org/popular-GGUF", name="Popularny", downloads=5_000_000, rank=1),
    ])
    listing = [e["id"] for e in catalog().entries() if e["id"].startswith("org/")]
    assert listing == ["org/popular-GGUF", "org/rare-GGUF"]


def test_a_model_that_does_not_fit_ranks_below_one_that_does(with_index, catalog) -> None:
    with_index([
        index_row("org/huge-GGUF", name="Ogromny", paramsB=400.0, sizeBytes=240 * GB,
                  downloads=9_000_000, rank=0),
        index_row("org/small-GGUF", name="Mały", paramsB=8.0, sizeBytes=4 * GB,
                  downloads=1_000, rank=1),
    ])
    listing = [e["id"] for e in catalog(budget_gb=16).entries() if e["id"].startswith("org/")]
    assert listing == ["org/small-GGUF", "org/huge-GGUF"]


def test_the_laboratorys_own_models_come_before_community_remixes(with_index, catalog) -> None:
    """Regression: a search for "qwen" opened on abliterated and "uncensored"
    fine-tunes, with Qwen's own first repository in fifteenth place."""
    with_index([
        index_row("huihui-ai/Qwen3-8B-abliterated-GGUF", name="Qwen3 8B abliterated",
                  provenance="community", publisher="Qwen (Alibaba)",
                  baseModel="huihui-ai/Qwen3-8B-abliterated", downloads=900_000, rank=0),
        index_row("bartowski/Qwen3-8B-uncensored-GGUF", name="Qwen3 8B uncensored",
                  provenance="trusted", publisher="Qwen (Alibaba)", downloads=800_000, rank=1),
        index_row("Qwen/Qwen3-8B-GGUF", name="Qwen3 8B", provenance="official",
                  publisher="Qwen (Alibaba)", baseModel=None, downloads=500_000, rank=2),
        index_row("unsloth/Qwen3-14B-GGUF", name="Qwen3 14B", provenance="trusted",
                  publisher="Qwen (Alibaba)", baseModel="Qwen/Qwen3-14B",
                  sizeBytes=8 * GB, downloads=400_000, rank=3),
    ])
    ours = {"Qwen/Qwen3-8B-GGUF", "unsloth/Qwen3-14B-GGUF", "huihui-ai/Qwen3-8B-abliterated-GGUF",
            "bartowski/Qwen3-8B-uncensored-GGUF"}
    # Staff picks still lead the page; this is about the order below them.
    listing = [e["id"] for e in catalog().entries(query="qwen") if e["id"] in ours]
    assert listing[:2] == ["Qwen/Qwen3-8B-GGUF", "unsloth/Qwen3-14B-GGUF"]
    assert set(listing[2:]) == {"huihui-ai/Qwen3-8B-abliterated-GGUF",
                                "bartowski/Qwen3-8B-uncensored-GGUF"}


def test_an_unused_laboratory_experiment_does_not_jump_the_queue(with_index, catalog) -> None:
    with_index([
        index_row("Qwen/Qwen3-experiment-GGUF", name="Eksperyment", provenance="official",
                  baseModel=None, downloads=12, rank=1),
        index_row("unsloth/Qwen3-8B-GGUF", name="Qwen3 8B", downloads=300_000, rank=0),
    ])
    listing = [e["id"] for e in catalog().entries() if e["id"].endswith("-GGUF")]
    assert listing.index("unsloth/Qwen3-8B-GGUF") < listing.index("Qwen/Qwen3-experiment-GGUF")


def test_largest_first_shows_the_biggest_model_this_mac_can_hold(with_index, catalog) -> None:
    """32 GB and 72 GB Macs were shown the same few-gigabyte models as an 8 GB one."""
    with_index([
        index_row("org/small-GGUF", name="Mały", paramsB=8.0, sizeBytes=5 * GB, rank=0),
        index_row("org/mid-GGUF", name="Średni", paramsB=32.0, sizeBytes=19 * GB, rank=1),
        index_row("org/big-GGUF", name="Duży", paramsB=70.0, sizeBytes=42 * GB, rank=2),
        index_row("org/giant-GGUF", name="Gigant", paramsB=671.0, sizeBytes=400 * GB, rank=3),
    ])
    small_mac = [e["id"] for e in catalog(budget_gb=29).entries(sort="najwieksze")
                 if e["id"].startswith("org/")]
    big_mac = [e["id"] for e in catalog(budget_gb=64).entries(sort="najwieksze")
               if e["id"].startswith("org/")]
    assert small_mac[0] == "org/mid-GGUF"
    assert big_mac[0] == "org/big-GGUF"
    assert small_mac[-1] == big_mac[-1] == "org/giant-GGUF"


# --------------------------------------------------------------------------- #
# Live search still merges on top
# --------------------------------------------------------------------------- #


def test_a_live_search_result_outside_the_index_is_still_shown(with_index, catalog) -> None:
    hub = FakeHub([hub_model("brand/New-Model-GGUF")])
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    entries = by_id(catalog(hub=hub).entries(query="new"))
    assert "brand/New-Model-GGUF" in entries
    scored = entries["brand/New-Model-GGUF"]["trust"]
    assert scored["score"] is not None, "wynik z sieci też dostaje ocenę"


# --------------------------------------------------------------------------- #
# Image models from the index
# --------------------------------------------------------------------------- #


def test_an_image_model_from_the_index_installs_with_its_encoders(with_index, catalog) -> None:
    """Downloading only the diffusion weights yields a model that renders nothing.

    The staff picks always carried the full file list; index rows did not, so a
    non-curated image model would have installed as a single file and failed at
    the first prompt.
    """
    files = ["sd35_medium-Q8_0.gguf", "clip_l-Q8_0.gguf", "clip_g-Q8_0.gguf", "t5xxl-Q4_0.gguf"]
    with_index([index_row("someone/SD3.5-Medium-GGUF", category="obrazy",
                          imageFamily="sd3", imageFiles=files, caps=["image"])])
    store = catalog()
    entry = by_id(store.entries())["someone/SD3.5-Medium-GGUF"]
    assert entry["recommendedVariant"]["files"] == files
    assert store.install_files(entry) == files      # no hub request needed


def test_a_language_model_still_asks_the_hub_for_its_variants(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    hub = FakeHub(details={"unsloth/Qwen3-8B-GGUF": {"variants": [
        {"id": "q4", "quant": "Q4_K_M", "files": ["Qwen3-8B-Q4_K_M.gguf"],
         "sizeBytes": 4 * GB, "complete": True, "recommended": True},
    ]}})
    store = catalog(hub=hub)
    entry = by_id(store.entries())["unsloth/Qwen3-8B-GGUF"]
    assert store.install_files(entry) == ["Qwen3-8B-Q4_K_M.gguf"]
    assert hub.detail_calls == ["unsloth/Qwen3-8B-GGUF"]


# --------------------------------------------------------------------------- #
# Installing: where the download is actually started
# --------------------------------------------------------------------------- #


class LoopBoundDownloads:
    """A ``DownloadManager`` stand-in with the real one's constraint.

    The real manager schedules tasks, so ``start`` must be called on the event
    loop; from a worker thread it raises.  The app showed that message —
    "Menedżer pobierania wymaga działającej pętli asyncio" — instead of starting
    a download, because the route resolved *and* started the job in a thread.
    """

    def __init__(self) -> None:
        self.started: list[tuple[str, list[str]]] = []

    def start(self, repo: str, files: Any = None, *, label: str | None = None) -> dict[str, Any]:
        asyncio.get_running_loop()          # raises RuntimeError off the loop
        self.started.append((repo, list(files or [])))
        return {"id": "job-1", "repo": repo, "files": list(files or []), "state": "queued"}


async def test_installing_starts_the_download_on_the_event_loop(with_index, settings) -> None:
    hub = FakeHub(details={"unsloth/Qwen3-8B-GGUF": {"variants": [
        {"id": "q4", "quant": "Q4_K_M", "files": ["Qwen3-8B-Q4_K_M.gguf"],
         "sizeBytes": 4 * GB, "complete": True, "recommended": True},
    ]}})
    downloads = LoopBoundDownloads()
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    store = Catalog(settings, hub=hub, library=FakeLibrary(), downloads=downloads,
                    budget_bytes=32 * GB)

    job = await store.ainstall("unsloth/Qwen3-8B-GGUF")

    assert job["state"] == "queued"
    assert downloads.started == [("unsloth/Qwen3-8B-GGUF", ["Qwen3-8B-Q4_K_M.gguf"])]


async def test_the_blocking_install_cannot_be_used_from_a_thread(with_index, settings) -> None:
    """Why :meth:`Catalog.ainstall` exists at all."""
    hub = FakeHub(details={"unsloth/Qwen3-8B-GGUF": {"variants": [
        {"id": "q4", "quant": "Q4_K_M", "files": ["x.gguf"], "sizeBytes": 4 * GB,
         "complete": True, "recommended": True},
    ]}})
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    store = Catalog(settings, hub=hub, library=FakeLibrary(),
                    downloads=LoopBoundDownloads(), budget_bytes=32 * GB)

    with pytest.raises(RuntimeError):
        await asyncio.to_thread(store.install, "unsloth/Qwen3-8B-GGUF")


# --------------------------------------------------------------------------- #
# Installing anything from Hugging Face, by id or by pasted link
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text, repo", [
    ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    ("https://huggingface.co/Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
    ("https://huggingface.co/Qwen/Qwen3-8B/tree/main", "Qwen/Qwen3-8B"),
    ("hf.co/unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-4B-GGUF"),
    ("https://huggingface.co/org/model?library=true", "org/model"),
])
def test_a_pasted_reference_is_recognised(text: str, repo: str) -> None:
    from localcode.hub.catalog import parse_repo_ref

    assert parse_repo_ref(text) == repo


@pytest.mark.parametrize("text", ["qwen coder", "coder", "", "a b/c", "https://example.com/a/b"])
def test_an_ordinary_query_is_not_a_reference(text: str) -> None:
    from localcode.hub.catalog import parse_repo_ref

    assert parse_repo_ref(text) is None


def _details(repo: str) -> dict[str, Any]:
    return {"model": {"id": repo, "format": "gguf", "paramsB": 8.0, "downloads": 12,
                      "likes": 1, "tags": ["gguf"]},
            "variants": [{"id": "q4", "quant": "Q4_K_M", "files": ["m-Q4_K_M.gguf"],
                          "sizeBytes": 4 * GB, "complete": True, "recommended": True}]}


def test_any_repository_can_be_installed_even_outside_the_catalogue(with_index, catalog) -> None:
    hub = FakeHub(details={"someone/Brand-New-7B-GGUF": _details("someone/Brand-New-7B-GGUF")})
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    entry = catalog(hub=hub).resolve_name("https://huggingface.co/someone/Brand-New-7B-GGUF")

    assert entry is not None
    assert entry["id"] == "someone/Brand-New-7B-GGUF"
    assert entry["external"] is True
    assert "spoza katalogu" in entry["badges"]      # the row says what it is
    assert entry["recommendedVariant"]["files"] == ["m-Q4_K_M.gguf"]
    assert entry["trust"]["score"] is not None       # scored like everything else


def test_our_own_entry_wins_over_the_pasted_one(with_index, catalog) -> None:
    """Pasting the id of a model we do describe gives our row, not a bare one."""
    hub = FakeHub(details={"unsloth/Qwen3-8B-GGUF": _details("unsloth/Qwen3-8B-GGUF")})
    with_index([index_row("unsloth/Qwen3-8B-GGUF", tagline="Nasz opis tego modelu.")])
    entry = catalog(hub=hub).resolve_name("unsloth/Qwen3-8B-GGUF")

    assert entry is not None
    assert entry.get("external") is not True
    assert entry["tagline"] == "Nasz opis tego modelu."


def test_a_repository_that_does_not_exist_is_simply_not_found(with_index, catalog) -> None:
    with_index([index_row("unsloth/Qwen3-8B-GGUF")])
    assert catalog(hub=FakeHub()).resolve_name("kto-to/nie-istnieje") is None
