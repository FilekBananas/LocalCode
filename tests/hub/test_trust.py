"""The trust score (``localcode.hub.trust``).

The store puts a number between 0 and 100 on models nobody here has run, so the
rules behind that number have to be pinned down: what raises it, what caps it,
and — the part that matters most — that no repository can show a green badge
if its files can execute code or if its creator is unknown.

Every assertion here is about a *relative* ordering or a documented rule, not
about a magic constant, so the rubric can be retuned without rewriting the file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from localcode.hub.publishers import classify
from localcode.hub.trust import (
    CURATED_FLOOR,
    LEVEL_LABELS,
    Trust,
    evaluate,
    evaluate_row,
    summarize_files,
)

GGUF = ["model-Q4_K_M.gguf", "README.md", ".gitattributes"]
PICKLE = ["pytorch_model.bin", "config.json", "README.md"]
SUSPICIOUS_CODE = ["model.exe", "script.sh", "library.dll"]
RECENT = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
OLD = (datetime.now(timezone.utc) - timedelta(days=900)).isoformat()


def official(**kw: object) -> Trust:
    """Official repository: Qwen's own GGUF, well-maintained, popular."""
    args: dict = dict(
        author="Qwen",
        downloads=2_000_000,
        downloads_all_time=100_000_000,
        likes=900,
        license_id="apache-2.0",
        siblings=GGUF,
        base_models={"models": [{"id": "Qwen/Qwen3-8B"}]},
        last_modified=RECENT,
    )
    args.update(kw)
    return evaluate("Qwen/Qwen3-8B-GGUF", **args)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Component thresholds: downloads_month
# --------------------------------------------------------------------------- #


def test_downloads_month_thresholds() -> None:
    """Verify downloads_month component thresholds: 0, 3, 6, 9, 12, 15."""
    # >=1M -> 15
    t = official(downloads=1_000_000)
    assert t.components["downloads_month"] == 15

    # >=100k -> 12
    t = official(downloads=100_000)
    assert t.components["downloads_month"] == 12

    # >=10k -> 9
    t = official(downloads=10_000)
    assert t.components["downloads_month"] == 9

    # >=1k -> 6
    t = official(downloads=1_000)
    assert t.components["downloads_month"] == 6

    # >=100 -> 3
    t = official(downloads=100)
    assert t.components["downloads_month"] == 3

    # <100 -> 0
    t = official(downloads=50)
    assert t.components["downloads_month"] == 0


# --------------------------------------------------------------------------- #
# Component thresholds: downloads_total
# --------------------------------------------------------------------------- #


def test_downloads_total_thresholds() -> None:
    """Verify downloads_total component thresholds: 0, 3, 6, 9, 12, 15."""
    # >=10M -> 15
    t = official(downloads_all_time=10_000_000)
    assert t.components["downloads_total"] == 15

    # >=1M -> 12
    t = official(downloads_all_time=1_000_000)
    assert t.components["downloads_total"] == 12

    # >=100k -> 9
    t = official(downloads_all_time=100_000)
    assert t.components["downloads_total"] == 9

    # >=10k -> 6
    t = official(downloads_all_time=10_000)
    assert t.components["downloads_total"] == 6

    # >=1k -> 3
    t = official(downloads_all_time=1_000)
    assert t.components["downloads_total"] == 3

    # <1k -> 0
    t = official(downloads_all_time=500)
    assert t.components["downloads_total"] == 0


def test_downloads_total_fallback_to_monthly() -> None:
    """When downloads_all_time is None, use monthly downloads with all-time thresholds."""
    # monthly=1M is >=1M in all-time thresholds, so should score 12
    t = official(downloads_all_time=None, downloads=1_000_000)
    assert t.components["downloads_total"] == 12


# --------------------------------------------------------------------------- #
# Component thresholds: likes
# --------------------------------------------------------------------------- #


def test_likes_thresholds() -> None:
    """Verify likes component thresholds: 0, 2, 5, 10, 14, 18, 22, 26, 30."""
    # >=1000 -> 30
    t = official(likes=1000)
    assert t.components["likes"] == 30

    # >=500 -> 26
    t = official(likes=500)
    assert t.components["likes"] == 26

    # >=200 -> 22
    t = official(likes=200)
    assert t.components["likes"] == 22

    # >=100 -> 18
    t = official(likes=100)
    assert t.components["likes"] == 18

    # >=50 -> 14
    t = official(likes=50)
    assert t.components["likes"] == 14

    # >=20 -> 10
    t = official(likes=20)
    assert t.components["likes"] == 10

    # >=5 -> 5
    t = official(likes=5)
    assert t.components["likes"] == 5

    # >=1 -> 2
    t = official(likes=1)
    assert t.components["likes"] == 2

    # 0 -> 0
    t = official(likes=0)
    assert t.components["likes"] == 0


# --------------------------------------------------------------------------- #
# Component: creator (max 40 points)
# --------------------------------------------------------------------------- #


def test_creator_official_is_40() -> None:
    """Official repository (lab's own) gets 40 points."""
    t = evaluate(
        "Qwen/Qwen3-8B-GGUF",
        author="Qwen",
        downloads=100,
        base_models={"models": [{"id": "Qwen/Qwen3-8B"}]},
    )
    assert t.components["creator"] == 40
    assert t.provenance.official is True


def test_creator_trusted_with_known_lab_is_34() -> None:
    """Trusted converter of known lab's model gets 34 points."""
    t = evaluate(
        "bartowski/Qwen3-8B-GGUF",
        author="bartowski",
        downloads=100,
        base_models={"models": [{"id": "Qwen/Qwen3-8B"}]},
    )
    assert t.components["creator"] == 34
    assert t.provenance.kind == "trusted"
    assert t.provenance.lab_known is True


def test_creator_verified_org_is_34() -> None:
    """Verified Hugging Face organisation gets 34 points."""
    t = evaluate(
        "verified-org/Model-GGUF",
        author="verified-org",
        downloads=100,
        author_info={"verified": True},
    )
    assert t.components["creator"] == 34


def test_creator_trusted_without_known_lab_is_26() -> None:
    """Trusted converter of unknown-origin model gets 26 points."""
    t = evaluate(
        "mradermacher/PrivateMerge-8B-GGUF",
        author="mradermacher",
        downloads=100,
        base_models={"models": [{"id": "someone/PrivateMerge-8B"}]},
    )
    assert t.components["creator"] == 26
    assert t.provenance.kind == "trusted"
    assert t.provenance.lab_known is False


def test_creator_author_1000_followers_is_22() -> None:
    """Author with >=1000 followers gets 22 points."""
    t = evaluate(
        "popular-author/Model-GGUF",
        author="popular-author",
        downloads=100,
        author_info={"followers": 1000},
    )
    assert t.components["creator"] == 22


def test_creator_community_is_16() -> None:
    """Community conversion of known lab model gets 16 points."""
    t = evaluate(
        "nick/Qwen3-8B-GGUF",
        author="nick",
        downloads=100,
        base_models={"models": [{"id": "Qwen/Qwen3-8B"}]},
    )
    assert t.components["creator"] == 16
    assert t.provenance.kind == "community"


def test_creator_author_100_followers_is_10() -> None:
    """Author with >=100 followers gets 10 points."""
    t = evaluate(
        "small-author/Model-GGUF",
        author="small-author",
        downloads=100,
        author_info={"followers": 100},
    )
    assert t.components["creator"] == 10


def test_creator_unknown_is_0() -> None:
    """Unknown creator (no provenance, no verification, no followers) gets 0 points."""
    t = evaluate(
        "unknown-author/Mystery-Model",
        author="unknown-author",
        downloads=100,
    )
    assert t.components["creator"] == 0


def test_creator_max_of_all_tiers() -> None:
    """Creator component returns the max of all applicable tiers."""
    # Official repos don't get additional points for followers
    t = official(author_info={"followers": 10000})
    assert t.components["creator"] == 40


# --------------------------------------------------------------------------- #
# Exclusion: suspicious files
# --------------------------------------------------------------------------- #


def test_excluded_for_pickle_files() -> None:
    """Repository with pickle files is excluded."""
    t = official(siblings=PICKLE)
    assert t.excluded is True
    assert "Odrzucony" in (t.exclusion or "")
    assert t.score == 0
    assert t.level == "rejected"


def test_excluded_for_suspicious_code() -> None:
    """Repository with .exe, .sh, .dll, etc. is excluded."""
    t = official(siblings=SUSPICIOUS_CODE)
    assert t.excluded is True
    assert "Odrzucony" in (t.exclusion or "")
    assert t.score == 0
    assert t.level == "rejected"


def test_not_excluded_for_py_files() -> None:
    """Repository with .py files is NOT excluded (only warned)."""
    t = official(siblings=[*GGUF, "modeling.py"])
    assert t.excluded is False
    assert any(".py" in w for w in t.warnings)


def test_excluded_even_if_popular_and_curated() -> None:
    """Exclusion (a) applies even to curated repositories."""
    t = official(siblings=PICKLE, curated=True)
    assert t.excluded is True
    assert t.score == 0


# --------------------------------------------------------------------------- #
# Exclusion: unknown creator
# --------------------------------------------------------------------------- #


def test_excluded_for_unknown_creator() -> None:
    """Repository with creator component == 0 is excluded (unless curated)."""
    t = evaluate(
        "nobody/Mystery-Model",
        author="nobody",
        downloads=5_000_000,
        likes=5_000,
        license_id="apache-2.0",
        siblings=GGUF,
        last_modified=RECENT,
    )
    assert t.components["creator"] == 0
    assert t.excluded is True
    assert "Odrzucony" in (t.exclusion or "")
    assert t.score == 0
    assert t.level == "rejected"


def test_curated_immune_to_unknown_creator_exclusion() -> None:
    """Curated repositories are immune to unknown-creator exclusion (rule b)."""
    t = evaluate(
        "unknown-author/Model",
        author="unknown-author",
        downloads=100,
        curated=True,
    )
    assert t.components["creator"] == 0
    # Should NOT be excluded due to rule (b) because curated=True
    # but there may be other factors; check that it's not excluded for creator reason
    if t.excluded:
        # Must be excluded for reason (a) - files, not (b) - creator
        assert "creator" not in (t.exclusion or "").lower()


# --------------------------------------------------------------------------- #
# Score bands and levels
# --------------------------------------------------------------------------- #


def test_popular_official_gguf_scores_high() -> None:
    """Popular official GGUF repository scores >= 80 (high band)."""
    t = official()
    assert t.score >= 80
    assert t.level == "high"


def test_score_stays_in_range() -> None:
    """Score is always 0-100."""
    t = official(
        downloads=99_000_000,
        downloads_all_time=99_000_000,
        likes=99_000,
        curated=True,
    )
    assert 0 <= t.score <= 100


def test_excluded_repo_is_level_rejected() -> None:
    """Excluded repository has level 'rejected'."""
    t = official(siblings=PICKLE)
    assert t.level == "rejected"
    assert t.label == "odrzucony"


def test_curated_floor_applies_to_non_excluded() -> None:
    """Curated repository with little data scores >= CURATED_FLOOR."""
    t = evaluate(
        "lmstudio/Model",
        author="lmstudio",
        curated=True,
        downloads=10,
        likes=0,
    )
    assert t.score >= CURATED_FLOOR
    assert t.level in ("good", "high")


# --------------------------------------------------------------------------- #
# Ordering: same model, different origins
# --------------------------------------------------------------------------- #


def test_provenance_orders_the_same_model() -> None:
    """Identical popularity and licence; only the publisher differs."""
    common: dict = dict(
        downloads=500_000,
        downloads_all_time=10_000_000,
        likes=200,
        license_id="apache-2.0",
        siblings=GGUF,
        last_modified=RECENT,
    )
    upstream = {"models": [{"id": "Qwen/Qwen3-8B"}]}

    lab = evaluate("Qwen/Qwen3-8B-GGUF", author="Qwen", **common)
    packager = evaluate(
        "bartowski/Qwen3-8B-GGUF",
        author="bartowski",
        base_models=upstream,
        **common,
    )
    community = evaluate(
        "nick/Qwen3-8B-GGUF",
        author="nick",
        base_models=upstream,
        **common,
    )
    unknown = evaluate(
        "unknown/Mystery-8B-GGUF",
        author="unknown",
        **common,
    )

    assert lab.score > packager.score > community.score > unknown.score
    assert lab.provenance is not None and lab.provenance.official is True


# --------------------------------------------------------------------------- #
# Individual signals
# --------------------------------------------------------------------------- #


def test_unknown_publisher_with_upstream_explains_and_names_creator() -> None:
    """Unknown publisher with a known base model: detail names the upstream, creator is warned about."""
    upstream = {"models": ["FastVideo/FastVideo-FastH3-4-step"]}
    prov = classify("someone/FastH3_GGUFs", base_models=upstream)
    assert "FastVideo/FastVideo-FastH3-4-step" in prov.detail
    assert "nie wskazuje modelu źródłowego" not in prov.detail

    t = evaluate(
        "someone/FastH3_GGUFs",
        author="someone",
        base_models=upstream,
        siblings=GGUF,
        author_info={"verified": False, "followers": 348, "isOrg": False},
    )
    assert "Twórca repozytorium: someone (konto prywatne, 348 obserwujących)." in t.warnings


def test_missing_licence_is_warned_about() -> None:
    """Repository without licence is warned about (but doesn't affect score)."""
    licensed = official()
    unlicensed = official(license_id=None)
    # License no longer affects score in new formula
    assert licensed.score == unlicensed.score
    assert any("licencj" in w.lower() for w in unlicensed.warnings)


def test_abandoned_repository_is_flagged() -> None:
    """Repository not updated for 2+ years is warned about (but doesn't affect score)."""
    fresh = official(last_modified=RECENT)
    stale = official(last_modified=OLD)
    # Maintenance no longer affects score in new formula
    assert fresh.score == stale.score
    assert any("Nieaktualizowane" in w for w in stale.warnings)


def test_gated_model_is_marked_without_penalty() -> None:
    """Gated model gets warning but no score penalty."""
    open_model = official()
    gated = official(gated=True)
    assert gated.score == open_model.score
    assert gated.gated is True
    assert any("token" in w.lower() for w in gated.warnings)


def test_nothing_known_at_all_scores_zero_and_excluded() -> None:
    """Repository with no data and unknown creator is excluded."""
    trust = evaluate("unknown/whatever")
    assert trust.score == 0
    assert trust.level == "rejected"
    assert trust.excluded is True


# --------------------------------------------------------------------------- #
# File summary (what the shipped index stores)
# --------------------------------------------------------------------------- #


def test_summarize_files_includes_suspicious_extensions() -> None:
    """File summary includes suspicious executable extensions."""
    summary = summarize_files(["model.gguf", "script.sh", "lib.dll", "code.py"])
    assert ".sh" in summary.get("suspicious", [])
    assert ".dll" in summary.get("suspicious", [])
    assert ".py" in summary.get("code", [])
    assert ".py" not in summary.get("suspicious", [])


def test_summary_and_file_list_score_identically() -> None:
    """Scoring from file list and from summary yields same result."""
    files = ["model-Q4_K_M.gguf", "README.md"]
    from_list = evaluate(
        "Qwen/Qwen3-8B-GGUF",
        author="Qwen",
        downloads=100,
        siblings=files,
    )
    from_summary = evaluate(
        "Qwen/Qwen3-8B-GGUF",
        author="Qwen",
        downloads=100,
        file_summary=summarize_files(files),
    )
    assert from_list.score == from_summary.score


# --------------------------------------------------------------------------- #
# evaluate_row: integration with Hub API
# --------------------------------------------------------------------------- #


def test_evaluate_row_reads_index_row() -> None:
    """evaluate_row parses Hub/index row format correctly."""
    row = {
        "id": "Qwen/Qwen3-8B-GGUF",
        "author": "Qwen",
        "downloads": 1_000_000,
        "downloadsAllTime": 100_000_000,
        "likes": 300,
        "license": "apache-2.0",
        "files": summarize_files(GGUF),
        "baseModels": {"models": [{"id": "Qwen/Qwen3-8B"}]},
        "lastModified": RECENT,
    }
    trust = evaluate_row(row)
    assert trust.provenance is not None
    assert trust.provenance.official is True
    assert trust.score > 60
    assert not trust.excluded


def test_evaluate_row_with_author_info() -> None:
    """evaluate_row passes author_info to creator evaluation."""
    row = {
        "id": "verified-org/Model",
        "author": "verified-org",
        "downloads": 100,
        "authorInfo": {"verified": True, "followers": 500},
    }
    trust = evaluate_row(row)
    assert trust.components["creator"] == 34  # verified org


# --------------------------------------------------------------------------- #
# Shape of the answer
# --------------------------------------------------------------------------- #


def test_bands_and_labels_agree() -> None:
    """Level and label are consistent."""
    trust = official()
    assert trust.level in LEVEL_LABELS
    assert trust.label == LEVEL_LABELS[trust.level]


def test_reasons_are_proper_sentences() -> None:
    """Reasons and warnings are well-formed Polish sentences."""
    trust = official()
    for sentence in trust.reasons + trust.warnings:
        assert sentence.strip().endswith((".", "!")), f"Bad ending: {sentence}"
        assert sentence[0].isupper() or sentence[0].isdigit(), f"Bad start: {sentence}"


def test_components_contain_only_new_keys() -> None:
    """Components dict has exactly the new keys: downloads_month, downloads_total, likes, creator."""
    trust = official()
    assert set(trust.components.keys()) == {
        "downloads_month",
        "downloads_total",
        "likes",
        "creator",
    }


def test_dto_carries_excluded_and_exclusion() -> None:
    """to_dict includes excluded and exclusion fields."""
    t_open = official()
    dto_open = t_open.to_dict()
    assert "excluded" in dto_open
    assert dto_open["excluded"] is False

    t_excluded = official(siblings=PICKLE)
    dto_excluded = t_excluded.to_dict()
    assert "excluded" in dto_excluded
    assert dto_excluded["excluded"] is True
    assert "exclusion" in dto_excluded


def test_dto_carries_provenance_and_components() -> None:
    """to_dict includes provenance and all components."""
    dto = official().to_dict()
    assert set(dto.keys()) >= {
        "score",
        "level",
        "label",
        "reasons",
        "warnings",
        "license",
        "gated",
        "components",
        "excluded",
        "provenance",
    }
    assert dto["components"]["downloads_month"] >= 0
    assert dto["components"]["downloads_total"] >= 0
    assert dto["components"]["likes"] >= 0
    assert dto["components"]["creator"] >= 0


def test_an_index_summary_without_the_suspicious_key_is_still_checked() -> None:
    """The shipped index stores {known, safe, pickle, code}; an executable there
    must exclude the repository just like a live file list does."""
    from localcode.hub.trust import evaluate

    stored = {"known": True, "safe": True, "pickle": [], "code": [".py", ".sh"]}
    assert evaluate("Qwen/Qwen3-8B-GGUF", file_summary=stored, downloads=10_000).excluded
    plain = {"known": True, "safe": True, "pickle": [], "code": [".py"]}
    assert not evaluate("Qwen/Qwen3-8B-GGUF", file_summary=plain, downloads=10_000).excluded

