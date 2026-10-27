"""The store's search (``localcode.hub.store_search``).

The queries here are the ones people actually type: a misspelt brand, a Polish
phrase describing what they want, a model family with the wrong ending. Each
test names the query it protects, because the whole point of this module is
that none of them require knowing a model's exact name.
"""

from __future__ import annotations

import pytest

from localcode.hub.store_search import (
    Intent,
    close_enough,
    fold,
    matches,
    parse,
    rank,
    score,
    tokens,
)


def model(name: str, **kw) -> dict:
    entry = {"name": name, "repo": f"org/{name.replace(' ', '-')}", "category": "czat",
             "tagline": "", "caps": [], "downloads": 1000}
    entry.update(kw)
    return entry


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_folding_removes_polish_accents() -> None:
    assert fold("Bielik Żółć") == "bielik zolc"
    assert fold("Qwen3-Coder") == "qwen3-coder"


def test_tokens_split_on_every_separator() -> None:
    assert tokens("Qwen3-Coder_30B/instruct") == ["qwen3", "coder", "30b", "instruct"]


# --------------------------------------------------------------------------- #
# Typos
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("typo, real", [
    ("deepsek", "deepseek"),
    ("gemmma", "gemma"),
    ("mistrall", "mistral"),
    ("qwn", "qwen"),
    ("codr", "coder"),
    ("embedings", "embeddings"),
])
def test_a_typo_still_finds_the_word(typo: str, real: str) -> None:
    assert close_enough(typo, real)


@pytest.mark.parametrize("query, other", [
    ("gpt", "opt"),        # three letters: a different first letter is a different word
    ("qwen", "gwen"),
    ("llama", "gemma"),
    ("bert", "bart"),
])
def test_a_different_word_is_not_a_typo(query: str, other: str) -> None:
    assert not close_enough(query, other)


def test_a_prefix_counts_as_a_match() -> None:
    assert close_enough("qwen", "qwen3")
    assert close_enough("embed", "embedding")


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query, category", [
    ("model do kodu", "kodowanie"),
    ("cos do programowania", "kodowanie"),
    ("do rysowania obrazkow", "obrazy"),
    ("generator obrazkow", "obrazy"),
    ("model do rozumowania", "rozumowanie"),
    ("do wyszukiwania w dokumentach", "embedding"),
    ("czyta zrzuty ekranu", "wizja"),
])
def test_polish_wording_picks_the_shelf(query: str, category: str) -> None:
    assert parse(query).category == category


def test_a_size_preference_is_understood() -> None:
    assert parse("maly szybki model").prefer_small is True
    assert parse("najmocniejszy model").prefer_big is True


def test_asking_for_polish_is_understood() -> None:
    assert parse("model po polsku").wants_polish is True


def test_noise_words_are_dropped() -> None:
    assert parse("szukam jakiegos modelu do kodu").words == ()


def test_english_category_words_stay_searchable() -> None:
    """"coder" is half of the model names in the store.

    Treating it as a category filter made "qwen coder" miss any repository whose
    category we had guessed wrong, so it only adds weight now.
    """
    intent = parse("qwen coder")
    assert intent.category is None
    assert intent.category_hint == "kodowanie"
    assert "coder" in intent.words


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #


def test_every_word_has_to_land_somewhere() -> None:
    entry = model("Qwen3 Coder 30B", category="kodowanie")
    assert matches(entry, "qwen coder")
    assert not matches(entry, "qwen gemma")


def test_a_name_match_outranks_a_description_match() -> None:
    named = model("Gemma 4 12B")
    described = model("Qwen3 8B", tagline="Podobny do modelu Gemma, ale mniejszy.")
    intent = parse("gemma")
    assert score(named, intent) > score(described, intent)


def test_the_category_filter_rejects_the_wrong_shelf() -> None:
    chat = model("Qwen3 8B", category="czat")
    assert score(chat, parse("model do obrazkow")) is None


def test_a_size_preference_reorders_rather_than_filters() -> None:
    small = model("Qwen3 4B", paramsB=4.0)
    big = model("Qwen3 72B", paramsB=72.0)
    intent = parse("maly qwen")
    assert score(small, intent) > score(big, intent)
    assert score(big, intent) is not None          # still an answer, just lower


def test_asking_for_tools_requires_them() -> None:
    with_tools = model("Qwen3 8B", caps=["tools"])
    without = model("Qwen3 8B")
    assert score(with_tools, parse("qwen z narzedziami")) is not None
    assert score(without, parse("qwen z narzedziami")) is None


def test_asking_for_polish_keeps_only_families_that_speak_it() -> None:
    bielik = model("Bielik 11B")
    random_model = model("Foobar 7B")
    assert score(bielik, parse("model po polsku")) is not None
    assert score(random_model, parse("model po polsku")) is None


def test_an_empty_query_matches_everything() -> None:
    assert matches(model("Qwen3 8B"), "")
    assert parse("").is_empty


def test_rank_returns_best_first() -> None:
    rows = [model("Gemma 4 12B"), model("Qwen3 8B", tagline="Alternatywa dla Gemma.")]
    ranked = rank(rows, "gemma")
    assert [entry["name"] for _, entry in ranked] == ["Gemma 4 12B", "Qwen3 8B"]
