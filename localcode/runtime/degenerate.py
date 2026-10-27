"""Noticing that a model has stopped saying anything new.

A small, heavily quantised model sometimes falls into a loop: it emits the same
fragment again and again until it runs out of output tokens.  Seen in the wild
with a 1-bit quantisation of a 30B model, whose entire answer to "make me a page
about bananas" was ``<todo_write>`` thirty-two times, and with a model repeating
one Polish sentence for a thousand tokens.

Nothing downstream can repair that — the text is not an answer and not a tool
call — so the only useful thing is to stop early, keep the junk off the screen
and tell the user which knob to turn (a stronger quantisation, a bigger model).

The test is deliberately literal: the tail of the output has to be one unit
repeated exactly, at least :data:`MIN_REPEATS` times, covering at least
:data:`MIN_CHARS` characters.  Prose does not do that; a broken model does.
Ordinary repetition in real answers stays under the bar — a Markdown rule
(``---``) or a run of eighty ``=`` is far short of 200 characters, and list
items differ from each other, which breaks the period.
"""

from __future__ import annotations

#: How many identical repetitions in a row count as a loop.
MIN_REPEATS = 8
#: ...and how many characters they have to cover, so short lists are safe.
MIN_CHARS = 200
#: The longest repeated unit looked for (a repeated paragraph is still a loop).
MAX_UNIT = 200
#: How much of the tail is kept, and how often the check runs.
WINDOW = 4_000
CHECK_EVERY = 120


def repeated_unit(text: str, *, min_repeats: int = MIN_REPEATS, min_chars: int = MIN_CHARS,
                  max_unit: int = MAX_UNIT) -> str | None:
    """The fragment ``text`` ends on repeating, or ``None``.

    Alignment does not matter: inside a periodic run every window of the period
    length repeats, so the check works on a tail cut mid-unit.
    """
    n = len(text)
    if n < min_chars:
        return None
    for period in range(1, min(max_unit, n // min_repeats) + 1):
        unit = text[n - period:]
        if not unit.strip():
            continue                      # blank lines and indentation are not a loop
        repeats, cut = 1, n - period
        while cut - period >= 0 and text[cut - period:cut] == unit:
            repeats += 1
            cut -= period
        if repeats >= min_repeats and repeats * period >= min_chars:
            return unit
    return None


class LoopGuard:
    """Streaming detector: feed it deltas, stop when it says stop.

    Cheap enough to sit in the token loop — it looks at the tail, and only once
    per :data:`CHECK_EVERY` characters.
    """

    def __init__(self, *, min_repeats: int = MIN_REPEATS, min_chars: int = MIN_CHARS,
                 max_unit: int = MAX_UNIT, window: int = WINDOW,
                 check_every: int = CHECK_EVERY) -> None:
        self.min_repeats, self.min_chars, self.max_unit = min_repeats, min_chars, max_unit
        self.window, self.check_every = window, check_every
        self._tail = ""
        self._since_check = 0
        self.unit: str | None = None

    @property
    def tripped(self) -> bool:
        return self.unit is not None

    def feed(self, text: str) -> bool:
        """Add a delta; ``True`` once the output has collapsed into a loop."""
        if not text or self.tripped:
            return self.tripped
        self._tail = (self._tail + text)[-self.window:]
        self._since_check += len(text)
        if self._since_check < self.check_every:
            return False
        self._since_check = 0
        self.unit = repeated_unit(self._tail, min_repeats=self.min_repeats,
                                  min_chars=self.min_chars, max_unit=self.max_unit)
        return self.tripped

    def finish(self) -> bool:
        """Final check, for output that ended before the next scheduled one."""
        if not self.tripped:
            self.unit = repeated_unit(self._tail, min_repeats=self.min_repeats,
                                      min_chars=self.min_chars, max_unit=self.max_unit)
        return self.tripped

    def describe(self) -> str:
        """The repeated fragment, shortened for a message (Polish UI quotes it)."""
        unit = _rotate(self.unit or "").strip().replace("\n", " ")
        return unit if len(unit) <= 40 else unit[:40].rstrip() + "…"


def _rotate(unit: str) -> str:
    """The unit turned so it starts at a word boundary.

    The detector reads the tail wherever the stream happened to stop, so the
    fragment it finds can start mid-word (``_write> <todo``). Rotating it to the
    first boundary makes the quoted text recognisable (``<todo_write>``).
    """
    for i, ch in enumerate(unit):
        if i and unit[i - 1].isspace() and not ch.isspace():
            return unit[i:] + unit[:i]
    return unit
