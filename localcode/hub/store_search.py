"""Finding a model in the store without knowing its name.

The store holds thousands of models with names like
``Qwen3.8-27B-A3B-Instruct-2512``.  Nobody types that.  People type "qwen
coder", "cos do kodu", "mały model po polsku", "deepsek" — with a typo, in
Polish, describing what they want rather than naming it.

This module is the search that answers those.  No model is loaded and nothing
is embedded: the queries people actually type are short, and three cheap layers
handle them far more predictably than a neural ranker would.

1. **Normalisation** — accents folded, punctuation dropped, so "Bielik" finds
   "bielik" and "gpt-oss" finds "gpt oss".
2. **Intent** — a small Polish vocabulary maps what people *mean* onto the
   catalogue's own fields: "do kodu" → the coding category, "mały" → a size
   preference, "po polsku" → models that actually speak Polish, "obrazek" →
   image models.  The words that carried the intent are then removed from the
   query, so "mały model do kodu" does not also have to match the word "mały"
   in some tagline.
3. **Matching** — what remains is matched token by token: exact, prefix, or
   within one or two typos (:func:`close_enough`), and scored so that a hit in
   the name outranks a hit in the description.

Everything here is pure text work on data the catalogue already has, so it runs
over six thousand rows in a few milliseconds and can be tested exhaustively.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, NamedTuple

#: Words that carry an intent rather than a name, mapped onto a category.
CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "kodowanie": (
        "kod", "kodu", "kodem", "kodowanie", "kodowania", "koder", "programowanie",
        "programowania", "programista", "programowac", "python", "javascript", "typescript",
        "swift", "rust", "sql", "skrypt", "skryptow", "refaktor", "debug", "debugowanie",
        "code", "coder", "coding", "programming",
    ),
    "czat": (
        "czat", "czatu", "rozmowa", "rozmowy", "rozmow", "gadanie", "asystent", "asystenta",
        "pisanie", "tekst", "teksty", "tekstow", "chat", "assistant", "writing",
    ),
    "rozumowanie": (
        "rozumowanie", "rozumowania", "myslenie", "myslenia", "logika", "logiki", "matematyka",
        "matematyki", "analiza", "analizy", "zadania", "reasoning", "thinking", "math",
    ),
    "obrazy": (
        "obraz", "obrazy", "obrazek", "obrazki", "obrazow", "grafika", "grafiki", "rysunek",
        "rysowanie", "zdjecie", "zdjecia", "ilustracja", "plakat", "image", "picture", "art",
        "diffusion",
    ),
    "wizja": (
        "wizja", "widzi", "widzenie", "zrzut", "zrzuty", "screenshot", "ocr", "skan",
        "rozpoznawanie", "vision", "vlm",
    ),
    "embedding": (
        "embedding", "embeddingi", "wyszukiwanie", "wyszukiwania", "rag", "dokumenty",
        "dokumentow", "baza", "wiedzy", "podobienstwo", "wektor", "wektory", "retrieval",
        "search", "similarity",
    ),
    "mowa": (
        "mowa", "mowy", "glos", "glosu", "dyktowanie", "transkrypcja", "transkrypcje",
        "napisy", "lektor", "czytanie", "nagranie", "nagrania", "podcast", "audio",
        "tts", "stt", "asr", "speech", "voice", "whisper", "dictation", "subtitles",
    ),
    "wideo": (
        "wideo", "video", "film", "filmy", "filmik", "animacja", "animacje", "klip",
        "klipy", "ruch",
    ),
}

#: Words that describe a size preference rather than a model.
SMALL_WORDS = ("maly", "mala", "male", "malutki", "lekki", "lekka", "szybki", "szybka",
               "najmniejszy", "mini", "drobny", "slaby", "small", "tiny", "fast", "light")
BIG_WORDS = ("duzy", "duza", "duze", "najwiekszy", "mocny", "mocna", "najlepszy", "najmocniejszy",
             "potezny", "big", "large", "strong", "best")

#: Words asking for a model that speaks Polish, and the families that do.
POLISH_WORDS = ("polski", "polsku", "polska", "polskie", "polskim", "polskiego", "pl",
                "polish")
POLISH_FAMILIES = ("bielik", "pllum", "qra", "trurl", "polka", "eurollm", "mistral", "qwen",
                   "gemma", "llama", "aya", "command", "salamandra", "occiglot", "teuken")

#: Within the speech shelf there are two opposite jobs, and the words for them
#: are different: reading text aloud, and writing down what was said.
TTS_STEMS = ("czyta", "lektor", "glos", "mow", "wypowie", "nagra", "audiobook", "speak")
TTS_WORDS = ("tts", "text-to-speech", "voice", "narration")
STT_STEMS = ("transkryp", "napis", "dyktow", "spisa", "rozpozna", "sluchow", "subtitl")
STT_WORDS = ("stt", "asr", "speech-to-text", "whisper", "transcribe", "transcription")

#: Words asking for the agent / tool use.
TOOL_STEMS = ("narzedz", "agent", "funkcj")
TOOL_WORDS = ("tools", "function", "functions", "toolcall")
#: Words asking for a model that is safe to trust.
TRUST_WORDS = ("sprawdzony", "sprawdzone", "pewny", "bezpieczny", "zaufany", "oficjalny",
               "oficjalne", "official", "verified")
#: Noise: words that mean "a model" and match everything.
STOP_WORDS = ("model", "modelu", "modele", "modeli", "modelem", "jakis", "jakies",
              "jakiegos", "jakiego", "jakas", "jakiejs", "cos", "czegos", "do", "dla",
              "na", "po", "od", "z", "ze", "w", "i", "oraz", "albo", "lub", "ktory",
              "ktore", "the", "a", "an", "for", "of", "to", "some", "any",
              "chce", "chcialbym", "chcialabym", "potrzebuje", "szukam", "daj", "pokaz",
              "znajdz", "proponuj", "polec",
              # generic nouns for "a thing that makes X" — the category already
              # carries that meaning, and as literal words they match nothing
              "generator", "generatora", "generatory", "generowanie", "generowania",
              "tworzenie", "tworzenia", "robienie", "robienia", "narzedzie")

_WORD_RE = re.compile(r"[a-z0-9.]+")

#: Stems that stand for a category whatever the Polish ending: "obrazek",
#: "obrazkow", "obrazkami" all mean the same shelf.
CATEGORY_STEMS: dict[str, tuple[str, ...]] = {
    "kodowanie": ("kod", "koder", "programow", "programist", "skrypt", "refaktor", "debug"),
    "czat": ("czat", "rozmow", "asystent", "pisani", "tekst", "gadk"),
    "rozumowanie": ("rozumowa", "mysle", "logik", "matematy", "analiz", "wnioskow"),
    "obrazy": ("obraz", "grafik", "rysun", "rysowa", "zdjec", "ilustrac", "plakat", "malowa"),
    "wizja": ("wizj", "widzen", "zrzut", "rozpoznaw", "skanow"),
    "embedding": ("embedding", "embeding", "wyszukiw", "dokument", "wektor", "podobienstw"),
    "mowa": ("mow", "glos", "dyktow", "transkryp", "napis", "lektor", "nagran", "sluchow"),
    "wideo": ("wideo", "film", "animacj", "klip"),
}


def _category_of(word: str) -> str | None:
    """The category a Polish word asks for, by exact word or by stem."""
    for cid, vocab in CATEGORY_WORDS.items():
        if word in vocab:
            return cid
    if len(word) >= 5:
        for cid, stems in CATEGORY_STEMS.items():
            if any(word.startswith(stem) for stem in stems):
                return cid
    return None


#: Words that name a category *and* appear in model names ("Qwen3-Coder").
#: They must not filter by category — a repository whose category we guessed
#: wrong would then be unfindable by its own name — so they only add weight,
#: and the word is still matched literally.
SOFT_CATEGORY_WORDS = frozenset({
    "code", "coder", "coding", "chat", "assistant", "writing", "reasoning", "thinking",
    "math", "vision", "vlm", "image", "art", "diffusion", "embedding", "search",
    "similarity", "retrieval", "rag", "ocr",
})


class Intent(NamedTuple):
    """What a query asked for, beyond the words themselves."""

    words: tuple[str, ...]          # what is left to match literally
    category: str | None            # "kodowanie" — a filter
    prefer_small: bool
    prefer_big: bool
    wants_polish: bool
    wants_tools: bool
    wants_trusted: bool
    category_hint: str | None = None   # a preference, not a filter
    speech_direction: str | None = None   # "tts" (czytaj) | "stt" (spisz)

    @property
    def is_empty(self) -> bool:
        return not (self.words or self.category or self.category_hint or self.prefer_small
                    or self.prefer_big or self.wants_polish or self.wants_tools
                    or self.wants_trusted or self.speech_direction)


#: Letters that are not an accented form of anything, so Unicode decomposition
#: leaves them as they are — "ł" is the one that matters for Polish.
_LETTER_MAP = str.maketrans({"ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ø": "o", "Ø": "o",
                             "ß": "ss", "æ": "ae", "Æ": "ae"})


def fold(text: str) -> str:
    """Lower-case, accent-free text: ``"Ż"`` → ``"z"``, ``"ł"`` → ``"l"``."""
    decomposed = unicodedata.normalize("NFKD", (text or "").translate(_LETTER_MAP))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def tokens(text: str) -> list[str]:
    """Searchable words of a string (``"Qwen3-Coder-30B"`` → qwen3, coder, 30b)."""
    folded = fold(text).replace("_", " ").replace("-", " ").replace("/", " ")
    return _WORD_RE.findall(folded)


def close_enough(query: str, token: str) -> bool:
    """True when ``token`` is ``query`` with at most a typo or two.

    The budget grows with the length of the word, because a slip in "gpt" is a
    different word while a slip in "deepseeek" obviously is not: up to three
    letters nothing is forgiven, then one typo, then two.
    """
    if query == token:
        return True
    if len(query) >= 3 and token.startswith(query):
        return True
    budget = 0 if len(query) <= 2 else (1 if len(query) <= 7 else 2)
    if not budget or abs(len(query) - len(token)) > budget:
        return False
    # A short word is mostly its opening: "qwn" may become "qwen" and "codr"
    # "coder", but "bert" must not become "bart" — BERT and BART are two
    # different families — and "gpt" must not become "opt".
    if len(query) <= 4 and query[:2] != token[:2]:
        return False
    return _within(query, token, budget)


def _within(a: str, b: str, budget: int) -> bool:
    """Levenshtein distance ≤ ``budget``, computed only that far."""
    if abs(len(a) - len(b)) > budget:
        return False
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > budget:
            return False
        previous = current
    return previous[-1] <= budget


def parse(query: str) -> Intent:
    """Split a query into an intent and the words still to match."""
    words = [w for w in tokens(query)]
    category: str | None = None
    category_hint: str | None = None
    speech_direction: str | None = None
    prefer_small = prefer_big = wants_polish = wants_tools = wants_trusted = False
    rest: list[str] = []

    for word in words:
        # The direction is remembered, but the shelf is only decided at the end:
        # "czyta zrzuty ekranu" is about vision, and the word "czyta" must not
        # settle the question before "zrzuty" has been seen.
        if word in TTS_WORDS or (len(word) >= 4 and any(word.startswith(st) for st in TTS_STEMS)):
            speech_direction = speech_direction or "tts"
            continue
        if word in STT_WORDS or (len(word) >= 5 and any(word.startswith(st) for st in STT_STEMS)):
            speech_direction = speech_direction or "stt"
            continue
        hit = _category_of(word)
        if hit and word in SOFT_CATEGORY_WORDS:
            category_hint = category_hint or hit
            rest.append(word)                   # still a word in model names
            continue
        if hit == "czat":
            # "tekst", "pisanie", "asystent" are weak evidence — every model
            # chats — so they must not outvote a clearer signal elsewhere in
            # the query ("czytanie tekstu na głos" is speech, not chat).
            category_hint = category_hint or hit
            continue
        if hit:
            category = category or hit
            continue
        if word in SMALL_WORDS:
            prefer_small = True
            continue
        if word in BIG_WORDS:
            prefer_big = True
            continue
        if word in POLISH_WORDS:
            wants_polish = True
            continue
        if word in TOOL_WORDS or any(word.startswith(stem) for stem in TOOL_STEMS):
            wants_tools = True
            continue
        if word in TRUST_WORDS:
            wants_trusted = True
            continue
        if word in STOP_WORDS:
            continue
        rest.append(word)
    if speech_direction and category is None:
        category = "mowa"
    # A query that is nothing but a weak word ("rozmowa") has no other signal
    # to rank by, so there the hint becomes the filter.
    if category is None and category_hint and not rest:
        category, category_hint = category_hint, None
    return Intent(tuple(rest), category, prefer_small, prefer_big, wants_polish,
                  wants_tools, wants_trusted, category_hint, speech_direction)


def haystacks(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(name tokens, everything else)`` for one catalogue row.

    Callers that search the same rows repeatedly cache the result on the row
    itself (``_name_tokens`` / ``_other_tokens``); tokenising six thousand rows
    per keystroke would otherwise dominate the search.
    """
    cached = entry.get("_name_tokens")
    if cached is not None:
        return cached, entry.get("_other_tokens") or []
    name = tokens(str(entry.get("name") or "")) + tokens(str(entry.get("repo") or entry.get("id") or ""))
    other = tokens(str(entry.get("tagline") or ""))
    other += tokens(str(entry.get("publisher") or ""))
    other += tokens(str(entry.get("baseModel") or ""))
    other += tokens(str(entry.get("arch") or ""))
    other += [str(c) for c in (entry.get("caps") or ())]
    return name, other


def score(entry: dict[str, Any], intent: Intent,
          *, trust: int | None = None) -> float | None:
    """How well a row answers a query, or ``None`` when it does not.

    The number only orders results within one query; it is not comparable
    between queries and never leaves this module.
    """
    name, other = haystacks(entry)
    total = 0.0

    for word in intent.words:
        hit = 0.0
        if word in name:
            hit = 10.0
        elif any(t.startswith(word) for t in name if len(word) >= 3):
            hit = 7.0
        elif any(close_enough(word, t) for t in name):
            hit = 5.0
        elif word in other:
            hit = 3.0
        elif any(close_enough(word, t) for t in other):
            hit = 1.5
        if not hit:
            return None                      # every word has to land somewhere
        total += hit

    if intent.category:
        if entry.get("category") != intent.category:
            return None
        total += 2.0
    elif intent.category_hint and entry.get("category") == intent.category_hint:
        total += 4.0

    if intent.wants_polish:
        family = " ".join(name)
        if not any(f in family for f in POLISH_FAMILIES):
            return None
        total += 2.0

    if intent.speech_direction:
        caps = {str(c) for c in (entry.get("caps") or ())}
        if intent.speech_direction in caps:
            total += 6.0
        elif caps & {"tts", "stt"}:
            return None          # the other direction is the wrong tool entirely

    if intent.wants_tools:
        if "tools" not in (entry.get("caps") or ()):
            return None
        total += 1.0

    if intent.wants_trusted:
        if trust is not None and trust < 60:
            return None
        total += 1.0

    params = entry.get("paramsB")
    if params:
        if intent.prefer_small:
            total += 6.0 if float(params) <= 5 else (3.0 if float(params) <= 10 else 0.0)
        if intent.prefer_big:
            total += 6.0 if float(params) >= 25 else (3.0 if float(params) >= 12 else 0.0)

    # Among equally good matches, the one more people use comes first; the log
    # keeps a million downloads from drowning the text score.
    downloads = int(entry.get("downloads") or 0)
    if downloads > 0:
        total += min(4.0, len(str(downloads)) * 0.4)
    return total


def matches(entry: dict[str, Any], query: str, *, trust: int | None = None) -> bool:
    """Convenience wrapper for callers that only need a yes/no."""
    intent = parse(query)
    if intent.is_empty:
        return True
    return score(entry, intent, trust=trust) is not None


def rank(entries: Iterable[dict[str, Any]], query: str,
         *, trust_of: Any = None) -> list[tuple[float, dict[str, Any]]]:
    """Every matching row with its score, best first."""
    intent = parse(query)
    out: list[tuple[float, dict[str, Any]]] = []
    for entry in entries:
        trust = None
        if trust_of is not None:
            try:
                trust = int(trust_of(entry).score)
            except Exception:  # noqa: BLE001 - scoring must never break a search
                trust = None
        value = score(entry, intent, trust=trust)
        if value is not None:
            out.append((value, entry))
    out.sort(key=lambda pair: -pair[0])
    return out
