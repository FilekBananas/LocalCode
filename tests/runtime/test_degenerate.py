"""Catching a model that has stopped saying anything new
(``localcode.runtime.degenerate``).

The case that produced this module: a 1-bit quantisation of a 30B model asked
for a web page answered ``<todo_write>`` thirty-two times.  These tests pin down
both halves of the bargain — real loops are caught early, and the shapes that
merely *look* repetitive in normal writing and code are not.
"""

from __future__ import annotations

import pytest

from localcode.runtime.degenerate import LoopGuard, repeated_unit


# --------------------------------------------------------------------------- #
# What counts as a loop
# --------------------------------------------------------------------------- #


def test_the_reported_failure_is_caught() -> None:
    assert repeated_unit("<todo_write> " * 32) is not None


def test_a_repeated_sentence_is_caught() -> None:
    assert repeated_unit("Rozumiem, że chcesz stronę o bananach. " * 10) is not None


def test_a_single_character_running_away_is_caught() -> None:
    assert repeated_unit("a" * 400) == "a"


def test_it_catches_the_loop_long_before_the_output_budget_runs_out() -> None:
    """A guard that only fires at the end saves nothing: the user reads the junk."""
    guard = LoopGuard()
    junk = "<todo_write> " * 32
    seen = 0
    for i in range(0, len(junk), 7):
        seen = i + 7
        if guard.feed(junk[i:i + 7]):
            break
    assert guard.tripped
    assert seen < len(junk) // 1.5        # stopped in the first two thirds


# --------------------------------------------------------------------------- #
# What does not count — the expensive half of the bargain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [
    "- pierwszy punkt\n- drugi punkt\n- trzeci punkt\n- czwarty punkt\n" * 3,
    "=" * 80 + "\nkod\n" + "-" * 80,
    "\n\n".join(f"## Sekcja {i}\nTreść akapitu numer {i}." for i in range(20)),
    "def f(x):\n    return x\n\n\ndef g(y):\n    return y\n" * 3,
    " " * 400,                                   # indentation is not an answer gone wrong
    "\n" * 200,
    "| kolumna | wartość |\n|---|---|\n" + "".join(f"| a{i} | {i} |\n" for i in range(30)),
])
def test_ordinary_writing_and_code_are_left_alone(text: str) -> None:
    guard = LoopGuard()
    guard.feed(text)
    guard.finish()
    assert not guard.tripped, text[:40]


def test_short_output_is_never_a_loop() -> None:
    assert repeated_unit("tak tak tak tak tak tak tak tak ") is None   # under the character floor


# --------------------------------------------------------------------------- #
# Using it in a stream
# --------------------------------------------------------------------------- #


def test_the_guard_reports_the_fragment_at_a_word_boundary() -> None:
    """The tail is cut wherever the stream stopped, the message has to read well."""
    guard = LoopGuard()
    guard.feed("<todo_write> " * 32)
    assert guard.describe() == "<todo_write>"


def test_a_long_repeated_paragraph_is_shortened_for_the_message() -> None:
    guard = LoopGuard()
    guard.feed(("Zaraz utworzę plik index.html z treścią o bananach i stylami. " * 8))
    assert len(guard.describe()) <= 41 and guard.describe().endswith("…")


def test_feeding_after_it_trips_changes_nothing() -> None:
    guard = LoopGuard()
    guard.feed("x" * 400)
    first = guard.unit
    assert guard.feed("zupełnie nowy tekst") is True
    assert guard.unit == first


def test_finish_catches_a_loop_that_ended_between_checks() -> None:
    guard = LoopGuard(check_every=10_000)     # never checks during feeding
    guard.feed("<todo_write> " * 32)
    assert not guard.tripped
    assert guard.finish() is True
