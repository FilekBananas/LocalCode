"""Tests for hub/curated.py: the staff-picks list and its ``HubModel`` projection.

The offline tests pin the shape of the list (the UI and the search client rely on
it); the ``network`` tests verify that every recommendation still exists on the
Hub and really ships the recommended quantisation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from localcode.hub.curated import (CURATED, CURATED_REPOS, curated_entry, curated_models, to_hub_model)
from localcode.hub.metadata import base_quant, is_mmproj, parse_quant

CATEGORIES = {"code", "chat", "reasoning", "vision", "embedding"}


def test_every_entry_has_the_required_fields():
    assert len(CURATED) >= 20
    for entry in CURATED:
        assert set(entry) >= {"repo", "format", "category", "paramsB", "minRamGb", "recommendedQuant", "note"}
        assert entry["repo"].count("/") == 1, entry["repo"]
        assert entry["format"] in ("gguf", "mlx")
        assert entry["category"] in CATEGORIES
        assert isinstance(entry["paramsB"], (int, float)) and entry["paramsB"] > 0
        assert entry["minRamGb"] in (8, 12, 16, 24, 32, 48, 64)
        assert entry["recommendedQuant"]
        assert entry["note"].endswith(".") and len(entry["note"]) < 200


def test_repos_are_unique_and_cover_every_tier_and_format():
    repos = [e["repo"] for e in CURATED]
    assert len(repos) == len(set(repos))
    assert CURATED_REPOS == frozenset(repos)
    assert {e["format"] for e in CURATED} == {"gguf", "mlx"}
    assert {e["category"] for e in CURATED} == CATEGORIES
    assert min(e["minRamGb"] for e in CURATED) <= 8
    assert max(e["minRamGb"] for e in CURATED) >= 32
    assert any(e["category"] == "embedding" and e["format"] == "gguf" for e in CURATED)
    assert any(e.get("activeParamsB") for e in CURATED)              # at least one MoE


def test_notes_are_polish():
    polish = [e for e in CURATED if any(ch in e["note"] for ch in "ąćęłńóśźż")]
    assert len(polish) >= 0.7 * len(CURATED)          # a few short notes need no diacritics
    for entry in CURATED:
        assert " the " not in entry["note"] and " with " not in entry["note"], entry["note"]


def test_curated_models_returns_copies():
    first = curated_models()
    first[0]["note"] = "zmienione"
    assert curated_models()[0]["note"] != "zmienione"


def test_curated_entry_lookup_is_case_insensitive():
    repo = CURATED[0]["repo"]
    assert curated_entry(repo.upper())["repo"] == repo
    assert curated_entry("nope/nope") is None


def test_to_hub_model_dto():
    entry = curated_entry("nomic-ai/nomic-embed-text-v1.5-GGUF")
    dto = to_hub_model(entry)
    assert dto["id"] == "nomic-ai/nomic-embed-text-v1.5-GGUF"
    assert dto["author"] == "nomic-ai" and dto["name"] == "nomic-embed-text-v1.5-GGUF"
    assert dto["format"] == "gguf" and dto["curated"] is True
    assert dto["pipelineTag"] == "feature-extraction"
    assert dto["downloads"] is None and dto["likes"] is None       # offline: no counters
    assert dto["description"] == entry["note"]
    assert dto["minRamGb"] == entry["minRamGb"]
    assert to_hub_model(curated_entry("unsloth/Qwen3-4B-Instruct-2507-GGUF"))["pipelineTag"] == "text-generation"


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


@pytest.mark.network
def test_every_curated_repo_exists_and_ships_the_recommended_quant(settings):
    from huggingface_hub import HfApi

    api = HfApi(library_name="localcode")

    def check(entry: dict[str, Any]) -> tuple[str, str]:
        try:
            info = api.model_info(entry["repo"], files_metadata=False)
        except Exception as exc:  # noqa: BLE001 - reported below with the repo id
            return entry["repo"], f"{type(exc).__name__}: {exc}"[:120]
        names = [s.rfilename for s in info.siblings]
        if getattr(info, "gated", False):
            return entry["repo"], "repozytorium jest gated"
        if entry["format"] == "gguf":
            quants = {base_quant(parse_quant(n)) for n in names
                      if n.lower().endswith(".gguf") and not is_mmproj(n)}
            if base_quant(entry["recommendedQuant"]) not in quants:
                return entry["repo"], f"missing {entry['recommendedQuant']} (has {sorted(q for q in quants if q)})"
        elif "config.json" not in names or not any(n.endswith(".safetensors") for n in names):
            return entry["repo"], "not an MLX checkpoint"
        return entry["repo"], ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        problems = {repo: why for repo, why in pool.map(check, CURATED) if why}
    assert not problems, problems
