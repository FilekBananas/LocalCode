"""Prompt preparation for image models — in practice: translating to English.

Diffusion models are trained on English captions and simply ignore most of a
prompt written in another language. Measured on Stable Diffusion 1.5 with the
same seed and steps:

    "kot w kapeluszu siedzi na dachu, akwarela"  → a photo-like cat on a roof,
                                                   no hat, no watercolour
    "a cat wearing a hat sitting on a roof,
     watercolor"                                 → exactly that picture

So before generating, a non-English prompt is translated by the chat model that
is already loaded (one short, thinking-off call). The translation is best-effort:
if no language model is available, the call fails or takes too long, the original
prompt is used unchanged — a slightly worse picture beats an error.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["looks_english", "translate_prompt", "prepare_prompt"]

TRANSLATE_TIMEOUT_S = 25.0
MAX_PROMPT_CHARS = 1200

#: A model smaller than this does not translate, it invents: Qwen3-0.6B turned
#: "kot w kapeluszu siedzi na dachu, akwarela" into a prompt about an apple.
#: With nothing bigger on disk the prompt is left alone instead.
MIN_TRANSLATE_PARAMS_B = 2.0

_SYSTEM = (
    "You translate image-generation prompts into English. Answer with the "
    "translated prompt only: no quotes, no explanation, no preamble.\n"
    "Translate EVERY element: every object, every piece of clothing, every "
    "colour, every style word (watercolor, oil painting, 3D render, photo…), "
    "every detail of the setting. Dropping a detail is the worst mistake you can "
    "make — the picture would be missing it. Keep the order of the original, keep "
    "proper names unchanged, and repeat an already-English prompt as it is.\n"
    "Example:\n"
    "  kot w kapeluszu siedzi na dachu, akwarela\n"
    "  → a cat wearing a hat sitting on a roof, watercolor"
)

#: Any non-ASCII letter (diacritics, Cyrillic, CJK…) means "not English".
_NON_ASCII = re.compile(r"[^\x00-\x7F]")
#: Frequent short words of languages a user of this app is likely to type, for
#: prompts that are plain ASCII but still not English ("un chien dans le jardin").
_NON_ENGLISH_WORDS = re.compile(
    r"\b(w|na|z|do|nie|jest|siedzi|kot|pies|dom|drzewo|kobieta|mezczyzna|dziewczyna|chlopiec"
    r"|und|der|die|das|mit|ein|eine|nicht|auf"
    r"|le|la|les|un|une|avec|sur|dans|chien|chat"
    r"|el|los|las|con|para|sobre|perro|gato"
    r"|il|lo|gli|sul|che|cane|gatto)\b",
    re.I,
)


def looks_english(text: Any) -> bool:
    """True when ``text`` is plausibly English (and needs no translation)."""
    body = str(text or "").strip()
    if not body:
        return True
    if _NON_ASCII.search(body):
        return False
    words = re.findall(r"[a-zA-Z]{2,}", body)
    if not words:
        return True                       # digits/punctuation only: nothing to translate
    hits = len(_NON_ENGLISH_WORDS.findall(body))
    return hits < max(1, len(words) // 8 + 1)


async def translate_prompt(text: str, *, manager: Any, model_key: str | None = None,
                           timeout: float = TRANSLATE_TIMEOUT_S) -> str:
    """Translate ``text`` to English with a loaded chat model; never raises.

    ``manager`` is a :class:`~localcode.runtime.manager.RuntimeManager`. Returns
    the original text when translation is impossible.
    """
    body = str(text or "").strip()
    if not body or manager is None:
        return body
    key = model_key or _pick_model(manager)
    if not key:
        return body
    try:
        return await asyncio.wait_for(_translate(body, manager, key), timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:              # busy model, template error, timeout…
        log.info("image prompt translation skipped (%s: %s)", type(exc).__name__, exc)
        return body


async def _translate(body: str, manager: Any, key: str) -> str:
    from localcode.runtime.base import SamplingParams
    from localcode.runtime.formatter import ChatFormatter

    async with manager.acquire(key, purpose="title") as lease:
        formatter = ChatFormatter(lease.engine)
        messages = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": body[:MAX_PROMPT_CHARS]}]
        prompt = await formatter.render(messages, None, thinking=False)
        params = SamplingParams(max_tokens=160, temperature=0.0,
                                stop=formatter.stop_strings())
        out: list[str] = []
        async for chunk in lease.engine.complete(prompt, params):
            if chunk.text:
                out.append(chunk.text)
    return _clean(("".join(out)), body)


def _clean(raw: str, original: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I).strip()
    text = text.split("\n\n")[0].strip().strip('"').strip("'").strip()
    if text.lower().startswith(("translation:", "english:", "prompt:")):
        text = text.split(":", 1)[1].strip()
    if not text or len(text) > MAX_PROMPT_CHARS * 2:
        return original
    return text


def _pick_model(manager: Any) -> str | None:
    """A language model that can be trusted to translate a prompt, or ``None``.

    Order: a model already in memory, then the configured chat/agent default,
    then the smallest one on disk — but every candidate must pass
    :func:`_big_enough`, because a too-small model rewrites the prompt into
    something else entirely instead of translating it.
    """
    library = getattr(manager, "library", None)

    def usable(key: str | None) -> str | None:
        if not key:
            return None
        model = None
        if library is not None:
            try:
                model = library.get(str(key))
            except Exception:
                model = None
        if model is None:                    # unknown to the library: trust the caller
            return str(key)
        if getattr(model, "is_embedding", False) or getattr(model, "is_image", False):
            return None
        return str(key) if _big_enough(model) else None

    try:
        for inst in manager.status().get("instances", []):
            if inst.get("state") in ("ready", "busy") and inst.get("kind") != "image":
                key = usable(inst.get("modelKey"))
                if key:
                    return key
        settings = getattr(manager, "settings", None)
        for holder in ("chat", "agent"):
            key = usable(getattr(getattr(settings, holder, None), "default_model", None))
            if key:
                return key
        models = list(library.list()) if library is not None else []
        candidates = [m for m in models
                      if not getattr(m, "is_embedding", False) and not getattr(m, "is_image", False)
                      and _big_enough(m)]
        if candidates:
            return str(min(candidates, key=lambda m: getattr(m, "size_bytes", 0) or 0).key)
    except Exception:
        log.debug("could not pick a translation model", exc_info=True)
    return None


def _big_enough(model: Any) -> bool:
    """True when ``model`` can be trusted with a translation."""
    params = getattr(getattr(model, "meta", None), "params_b", None)
    if params:
        return float(params) >= MIN_TRANSLATE_PARAMS_B
    # Unknown parameter count: judge by file size (a 4-bit 2B model is ~1.2 GB).
    return (getattr(model, "size_bytes", 0) or 0) >= 1_200_000_000


async def prepare_prompt(text: str, *, manager: Any, translate: bool | None = None,
                         model_key: str | None = None) -> tuple[str, bool]:
    """Return ``(prompt, translated)`` for an image request.

    ``translate=None`` (the default) means "translate only when the prompt does
    not look English"; ``True`` forces it, ``False`` disables it.
    """
    body = str(text or "").strip()
    if translate is False or not body:
        return body, False
    if translate is None and looks_english(body):
        return body, False
    result = await translate_prompt(body, manager=manager, model_key=model_key)
    return result, bool(result and result.strip().lower() != body.strip().lower())
