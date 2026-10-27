"""Who published a model (``localcode.hub.publishers``).

The store's "źródło oficjalne" badge is only worth something if it cannot be
earned by naming a repository cleverly, so these tests pin down the three ways
provenance is established — the Hub's own ``baseModels`` link, the
``base_model:`` tags, and the model family in the name — and the cases where the
answer must stay "nie wiadomo".
"""

from __future__ import annotations

import pytest

from localcode.hub.publishers import (
    OFFICIAL_PUBLISHERS,
    TRUSTED_QUANTIZERS,
    classify,
    family_publisher,
    is_official_org,
    is_trusted_quantizer,
    publisher_display,
    upstream_repo,
)


def base(repo: str, relation: str = "quantized") -> dict[str, object]:
    return {"relation": relation, "models": [{"id": repo}]}


# --------------------------------------------------------------------------- #
# The laboratory's own repository
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("repo", [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-8B-GGUF",
    "google/gemma-4-12b-it-qat-q4_0-gguf",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "ibm-granite/granite-4.2-8b-GGUF",
])
def test_the_laboratorys_own_repository_is_official(repo: str) -> None:
    prov = classify(repo, base_models=base(repo.partition("/")[0] + "/upstream"))
    assert prov.kind == "official"
    assert prov.official is True
    assert prov.label == "źródło oficjalne"
    assert prov.lab_known is True


def test_official_repository_names_the_laboratory_in_polish() -> None:
    prov = classify("Qwen/Qwen3-8B-GGUF")
    assert prov.publisher == "Qwen (Alibaba)"
    assert "Qwen (Alibaba)" in prov.detail
    assert prov.reasons


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #


def test_a_known_packager_of_a_known_model_is_a_trusted_conversion() -> None:
    prov = classify("bartowski/Qwen3-8B-GGUF", base_models=base("Qwen/Qwen3-8B"))
    assert prov.kind == "trusted"
    assert prov.official is False
    assert prov.lab_known is True
    assert prov.packager == "bartowski"
    assert prov.publisher == "Qwen (Alibaba)"
    assert prov.upstream == "Qwen/Qwen3-8B"
    assert "Qwen/Qwen3-8B" in prov.detail


def test_a_known_packager_of_an_unknown_model_does_not_borrow_its_trust() -> None:
    """mradermacher packaging somebody's private merge is still an unknown model."""
    prov = classify("mradermacher/SomeMerge-13B-i1-GGUF",
                    base_models=base("randomperson/SomeMerge-13B"))
    assert prov.kind == "trusted"
    assert prov.lab_known is False          # ← what keeps the score down
    assert prov.packager == "mradermacher"


def test_an_unknown_author_converting_a_known_model_is_a_community_conversion() -> None:
    prov = classify("somebody/Qwen3-8B-GGUF-v2", tags=["base_model:Qwen/Qwen3-8B"])
    assert prov.kind == "community"
    assert prov.publisher == "Qwen (Alibaba)"
    assert prov.packager == "somebody"
    assert "somebody" in prov.detail


def test_nothing_known_at_all_says_so() -> None:
    prov = classify("nick/MyModel-7B-GGUF")
    assert prov.kind == "unknown"
    assert prov.official is False
    assert prov.lab_known is False
    assert prov.label == "źródło nieznane"
    assert "nick" in prov.detail


# --------------------------------------------------------------------------- #
# Where the upstream link comes from
# --------------------------------------------------------------------------- #


def test_upstream_is_read_from_the_hub_object() -> None:
    assert upstream_repo(base("Qwen/Qwen3-8B")) == "Qwen/Qwen3-8B"


def test_upstream_falls_back_to_the_tags() -> None:
    tags = ["gguf", "base_model:quantized:meta-llama/Llama-3.1-8B", "en"]
    assert upstream_repo(None, tags) == "meta-llama/Llama-3.1-8B"


def test_upstream_ignores_tags_without_an_organisation() -> None:
    assert upstream_repo(None, ["base_model:something"]) is None


# --------------------------------------------------------------------------- #
# The family fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, org", [
    ("Qwen3-8B-GGUF", "qwen"),
    ("gemma-4-12b-it", "google"),
    ("CodeLlama-13B-Instruct-GGUF", "meta-llama"),       # longest fragment wins
    ("Llama-3.1-8B-Instruct", "meta-llama"),
    ("Devstral-Small-2-24B", "mistralai"),
    ("gpt-oss-20b-GGUF", "openai"),
    ("GLM-4.7-Flash-GGUF", "zai-org"),
    ("Bielik-11B-v3.0-Instruct", "speakleash"),
    ("nomic-embed-text-v1.5-GGUF", "nomic-ai"),
    ("FLUX.1-schnell-mflux-4bit", "black-forest-labs"),
])
def test_family_publisher_reads_the_model_family(name: str, org: str) -> None:
    assert family_publisher(name) == org


def test_family_publisher_is_not_confused_by_packaging_suffixes() -> None:
    assert family_publisher("Qwen3-Coder-30B-A3B-Instruct-GGUF-Q4_K_M") == "qwen"
    assert family_publisher("MyModel-7B-GGUF") is None


def test_family_publisher_prefers_the_more_specific_family() -> None:
    # "qwen-image" must not be answered with the plain "qwen" text family
    assert family_publisher("Qwen-Image-2512-4bit") == "qwen"
    assert family_publisher("codegemma-7b-it") == "google"


# --------------------------------------------------------------------------- #
# The tables themselves
# --------------------------------------------------------------------------- #


def test_the_two_tables_do_not_overlap() -> None:
    """An organisation is either a laboratory or a packager, never both."""
    assert not (set(OFFICIAL_PUBLISHERS) & set(TRUSTED_QUANTIZERS))


def test_table_keys_are_lower_case() -> None:
    for org in [*OFFICIAL_PUBLISHERS, *TRUSTED_QUANTIZERS]:
        assert org == org.lower(), org


def test_lookups_ignore_case() -> None:
    assert is_official_org("QWEN") and is_official_org("qwen")
    assert is_trusted_quantizer("Unsloth")
    assert publisher_display("unsloth") == "Unsloth"
    assert publisher_display("nobody-at-all") == "nobody-at-all"


def test_provenance_dto_is_json_friendly() -> None:
    dto = classify("bartowski/Qwen3-8B-GGUF", base_models=base("Qwen/Qwen3-8B")).to_dict()
    assert dto["kind"] == "trusted"
    assert dto["official"] is False
    assert dto["labKnown"] is True
    assert set(dto) >= {"kind", "publisher", "packager", "upstream", "official", "label", "detail"}


# --------------------------------------------------------------------------- #
# The logo belongs to the laboratory, not to whoever repackaged its model
# --------------------------------------------------------------------------- #


def test_only_the_laboratorys_own_repository_carries_its_logo() -> None:
    """A conversion may be excellent and still is not Google's upload.

    Putting Google's G on ``bartowski/gemma-…`` would claim something nobody
    verified; the provenance badge already says where the weights came from.
    """
    own = classify("google/gemma-4-12b-it-qat-q4_0-gguf").to_dict()
    conversion = classify("bartowski/gemma-4-12b-GGUF",
                          base_models=base("google/gemma-4-12b")).to_dict()
    community = classify("ktos/gemma-4-12b-GGUF-v2",
                         tags=["base_model:google/gemma-4-12b"]).to_dict()

    assert own["logo"] == "google"
    assert conversion["logo"] is None
    assert community["logo"] is None
    # ...but all three still say whose model it is, in words
    assert "Google" in conversion["publisher"] and "Google" in community["publisher"]


def test_an_unknown_repository_has_no_logo() -> None:
    assert classify("nick/Mystery-7B").to_dict()["logo"] is None
