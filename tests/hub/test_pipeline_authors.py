"""Tests for author-overview fetching and pipeline integration.

Tests the author overview parser, the build_row integration with authorInfo,
and the attachment of author data from the authors map to catalogue entries.
Offline tests drive the pipeline with fake HTTP functions to avoid network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Author overview parsing
# --------------------------------------------------------------------------- #


def fake_hub(responses: dict[str, tuple[int, Any]]):
    calls: list[str] = []

    def getter(url: str, headers: dict[str, str]) -> tuple[int, Any]:
        calls.append(url)
        for fragment, answer in responses.items():
            if fragment in url:
                return answer
        return 404, {"error": "Sorry, we can't find the page you are looking for."}

    getter.calls = calls  # type: ignore[attr-defined]
    return getter


def test_an_organisation_overview_is_read() -> None:
    from scripts.hub_fetch import fetch_author_overview

    getter = fake_hub({"organizations/neuphonic": (200, {"isVerified": True, "numFollowers": 1234,
                                                          "plan": "team"})})
    result = fetch_author_overview("neuphonic", http_getter=getter)
    assert result == {"isOrg": True, "verified": True, "plan": "team", "followers": 1234}


def test_a_user_is_asked_after_the_organisation_endpoint_says_404() -> None:
    from scripts.hub_fetch import fetch_author_overview

    getter = fake_hub({"users/DavidAU": (200, {"numFollowers": 567, "isPro": True})})
    result = fetch_author_overview("DavidAU", http_getter=getter)
    assert result == {"isOrg": False, "verified": False, "plan": None, "followers": 567}
    assert len(getter.calls) == 2


def test_an_error_body_is_never_taken_for_an_overview() -> None:
    """Regression: with ``requests().json()`` a 404 body parsed as an organisation
    with 0 followers, so every user would have been "an organisation"."""
    from scripts.hub_fetch import fetch_author_overview

    result = fetch_author_overview("nikt", http_getter=fake_hub({}))
    assert result["missing"] is True and result["isOrg"] is False


def test_a_rate_limit_waits_and_retries() -> None:
    from scripts.hub_fetch import fetch_author_overview

    answers = iter([(429, {"error": "rate limit"}), (200, {"isVerified": False, "numFollowers": 2100})])
    slept: list[float] = []
    result = fetch_author_overview("QuantStack", http_getter=lambda u, h: next(answers),
                                   sleep=slept.append)
    assert result["followers"] == 2100 and slept == [90]


def test_still_limited_after_every_retry_is_marked_for_a_later_run() -> None:
    from scripts.hub_fetch import fetch_author_overview

    result = fetch_author_overview("x", http_getter=lambda u, h: (429, None), sleep=lambda s: None)
    assert result.get("retry") is True and "missing" not in result


# --------------------------------------------------------------------------- #
# build_row integration
# --------------------------------------------------------------------------- #


def test_build_row_includes_downloads_all_time() -> None:
    """build_row must add downloadsAllTime from raw row."""
    from scripts.build_catalog import build_row

    row = {
        "id": "unsloth/Qwen3-0.6B-GGUF",
        "format": "gguf",
        "author": "unsloth",
        "downloads": 50000,
        "downloadsAllTime": 150000,
        "likes": 123,
        "tags": ["gguf", "text-generation"],
        "gated": False,
        "private": False,
        "disabled": False,
        "siblings": [],
    }

    entry = build_row(row)
    assert entry["downloadsAllTime"] == 150000
    assert entry["downloads"] == 50000


def test_build_row_handles_missing_downloads_all_time() -> None:
    """build_row must handle missing downloadsAllTime."""
    from scripts.build_catalog import build_row

    row = {
        "id": "unsloth/Qwen3-0.6B-GGUF",
        "format": "gguf",
        "author": "unsloth",
        "downloads": 50000,
        "likes": 123,
        "tags": ["gguf", "text-generation"],
        "gated": False,
        "private": False,
        "disabled": False,
        "siblings": [],
    }

    entry = build_row(row)
    assert entry["downloadsAllTime"] is None
    assert entry["downloads"] == 50000


# --------------------------------------------------------------------------- #
# Author data attachment
# --------------------------------------------------------------------------- #


def test_select_attaches_author_info(tmp_path: Path) -> None:
    """select() attaches authorInfo from authors map to entries."""
    from scripts.build_catalog import select

    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps({
            "id": "unsloth/Qwen3-0.6B-GGUF",
            "format": "gguf",
            "author": "unsloth",
            "downloads": 50000,
            "downloadsAllTime": 150000,
            "likes": 123,
            "tags": ["gguf", "text-generation"],
            "gated": False,
            "private": False,
            "disabled": False,
            "siblings": ["Qwen3-0.6B-Q4_K_M.gguf"],
            "pipelineTag": "text-generation",
        }) + "\n"
    )

    authors_path = tmp_path / "authors.json"
    authors_path.write_text(json.dumps({
        "unsloth": {
            "isOrg": True,
            "verified": True,
            "plan": "pro",
            "followers": 1234,
        }
    }))

    rows, rejects = select(raw_path, authors_path=authors_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["repo"] == "unsloth/Qwen3-0.6B-GGUF"
    assert row["authorInfo"] == {
        "isOrg": True,
        "verified": True,
        "plan": "pro",
        "followers": 1234,
    }


def test_select_skips_missing_authors(tmp_path: Path) -> None:
    """select() does not attach authorInfo if author has 'missing' flag."""
    from scripts.build_catalog import select

    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps({
            "id": "someuser/model",
            "format": "gguf",
            "author": "someuser",
            "downloads": 100,
            "likes": 1,
            "tags": ["gguf", "text-generation"],
            "gated": False,
            "private": False,
            "disabled": False,
            "siblings": ["model-Q4_K_M.gguf"],
            "pipelineTag": "text-generation",
        }) + "\n"
    )

    authors_path = tmp_path / "authors.json"
    authors_path.write_text(json.dumps({
        "someuser": {
            "isOrg": False,
            "verified": False,
            "plan": None,
            "followers": 0,
            "missing": True,
        }
    }))

    rows, rejects = select(raw_path, authors_path=authors_path)

    assert len(rows) == 1
    row = rows[0]
    assert "authorInfo" not in row


def test_select_without_authors_path(tmp_path: Path) -> None:
    """select() works when authors_path is None."""
    from scripts.build_catalog import select

    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps({
            "id": "unsloth/Qwen3-0.6B-GGUF",
            "format": "gguf",
            "author": "unsloth",
            "downloads": 50000,
            "likes": 123,
            "tags": ["gguf", "text-generation"],
            "gated": False,
            "private": False,
            "disabled": False,
            "siblings": ["Qwen3-0.6B-Q4_K_M.gguf"],
            "pipelineTag": "text-generation",
        }) + "\n"
    )

    rows, rejects = select(raw_path, authors_path=None)

    assert len(rows) == 1
    row = rows[0]
    assert "authorInfo" not in row


# --------------------------------------------------------------------------- #
# Search live data (expanded fields)
# --------------------------------------------------------------------------- #


def test_search_expand_includes_downloads_all_time() -> None:
    """The EXPAND list must include downloadsAllTime."""
    from localcode.hub.search import EXPAND

    assert "downloadsAllTime" in EXPAND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
