"""Summarising compaction for chatbot conversations.

The coding agent keeps long sessions inside a small context window with a staged
pipeline (mask tool results → shrink → model-written summary → drop oldest, see
:mod:`localcode.agent.context`). The chatbot has no tool results, so only the
summarisation stage applies — and this module is that stage.

What happens
------------
When the prompt assembled for a turn would pass ``threshold × prompt budget``
(``ChatSettings.auto_compact_threshold``, default 0.8, falling back to the
agent's ``auto_compact_threshold``), the oldest part of the conversation is
replaced by one dense summary written **by the same model** in one short,
tool-free, thinking-off call of ~400–600 tokens:

* the most recent turns stay verbatim (:data:`RECENT_SHARE` of the prompt
  budget), and the cut always lands on a user message, so a user/assistant pair
  is never split;
* the summary never replaces the system prompt, the memories or the citations
  of the current turn — it only stands in for stored messages;
* the result is persisted on the conversation (``summary``, ``summary_upto_seq``,
  ``summary_tokens``), so it survives a restart and every later turn is
  ``system + summary + messages after summary_upto_seq``;
* compacting again **extends** the summary: the previous summary and the newly
  dropped turns go into the prompt together, and the answer replaces the old
  summary;
* when the model is busy or fails, nothing is stored and the caller simply falls
  back to the existing trimming — a failed summary never breaks a turn.

Everything here is framework-free: :class:`ChatCompactor` gets a token counter
and an ``async summarizer(messages, max_tokens) -> str`` callable, and returns a
:class:`CompactionResult`. Storing it and reporting it over SSE is
:mod:`localcode.chat.service`'s job.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from localcode.chat.context import (
    MESSAGE_OVERHEAD,
    SUMMARY_HEADER,
    SUMMARY_NOTE,
    TokenCounter,
    clean,
    plan_budget,
    summary_block,
    truncate,
    truncate_to_tokens,
)

log = logging.getLogger(__name__)

__all__ = [
    "CompactionPlan",
    "CompactionResult",
    "ChatCompactor",
    "STRATEGY_SUMMARIZE",
    "SUMMARY_HEADER",
    "SUMMARY_SHARE",
    "DEFAULT_THRESHOLD",
    "build_summary_prompt",
    "clean_summary",
    "pending_messages",
    "render_transcript",
    "summary_block",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

STRATEGY_SUMMARIZE = "summarize"   #: the model wrote the summary (mirrors events.Compaction)
STRATEGY_EXTRACT = "extract"       #: facts copied verbatim (model too small to summarise)

#: Below this many parameters a model writes summaries that are worse than no
#: summary at all: measured on Qwen3-0.6B, which turned a Polish conversation
#: into an English one and renamed the user's dog into their child. For such
#: models the oldest turns are compressed by QUOTING them instead.
MIN_SUMMARY_PARAMS_B = 1.5

DEFAULT_THRESHOLD = 0.80           #: share of the prompt budget that triggers compaction
SUMMARY_SHARE = 0.25               #: share of the prompt budget the summary block may take
MIN_SUMMARY_TOKENS = 400           #: the model is asked for at least this many tokens…
MAX_SUMMARY_TOKENS = 600           #: …and at most this many
RECENT_SHARE = 0.45                #: share of the prompt budget kept verbatim after the summary
MIN_REGION_MESSAGES = 2            #: never summarise a single message
MIN_REGION_TOKENS = 200            #: …nor a region cheaper than the summary would be
SUMMARY_TIMEOUT_S = 180.0          #: a summary that takes longer than this is abandoned

#: characters of one message kept in the transcript handed to the model
TRANSCRIPT_USER_CHARS = 2000
TRANSCRIPT_ASSISTANT_CHARS = 1200

#: Sections the model is asked for (English labels keep the summary parseable;
#: the prose inside them follows the language of the conversation).
SUMMARY_SECTIONS = ("Goal", "Facts", "Decisions", "Open questions", "Preferences")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*?)\n?```$", re.S)

#: ``async (prompt_messages, max_tokens) -> str`` — one tool-free model call.
Summarizer = Callable[[list[dict[str, Any]], int], Awaitable[str]]


# --------------------------------------------------------------------------- #
# Prompt and rendering
# --------------------------------------------------------------------------- #


def clean_summary(raw: Any) -> str:
    """Strip reasoning, code fences and our own header from a model answer.

    Returns ``""`` when nothing usable is left (too short, only reasoning, …),
    which the caller treats as a failed summarisation.
    """
    if not isinstance(raw, str):
        return ""
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:                     # unterminated reasoning: nothing usable after it
        text = text.split("<think>", 1)[0]
    text = text.replace("</think>", "").replace(SUMMARY_HEADER, "").replace(SUMMARY_NOTE, "").strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    return text if len(text) >= 20 else ""


def _attachment_note(message: dict[str, Any]) -> str:
    names = [str(a.get("name") or "plik") for a in (message.get("attachments") or [])]
    return f" [attachments: {', '.join(names)}]" if names else ""


def render_transcript(messages: Sequence[dict[str, Any]]) -> str:
    """Render stored messages as ``USER:`` / ``ASSISTANT:`` lines for the summariser.

    Attachments are named but not inlined (their text is already summarised in
    the answers) and every message is clipped, so one huge message cannot push
    the rest of the conversation out of the summarisation prompt.
    """
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        text = clean(message.get("content"))
        note = _attachment_note(message) if role == "user" else ""
        if not text and not note:
            continue
        if role == "user":
            lines.append("USER: " + truncate(text, TRANSCRIPT_USER_CHARS) + note)
        elif role == "assistant":
            lines.append("ASSISTANT: " + truncate(text, TRANSCRIPT_ASSISTANT_CHARS))
        else:
            lines.append("NOTE: " + truncate(text, 500))
    return "\n".join(lines)


def extractive_summary(region: Sequence[dict[str, Any]], *, max_items: int = 12,
                       max_chars: int = 240) -> str:
    """A summary made of the user's own words, for models too small to write one.

    Quoting beats paraphrasing here: a 0.6B model hallucinates, but the raw user
    turns keep every name, number and preference intact, and they compress well
    because the assistant's answers are dropped.
    """
    lines: list[str] = []
    for message in region:
        if str(message.get("role")) != "user":
            continue
        body = clean(message.get("content"))
        if not body:
            continue
        body = " ".join(body.split())
        if len(body) > max_chars:
            body = body[: max_chars - 1].rstrip() + "…"
        lines.append(f"- {body}")
    if not lines:
        return ""
    if len(lines) > max_items:                     # keep the oldest and the newest
        head = lines[: max_items // 2]
        tail = lines[-(max_items - len(head)):]
        lines = head + [f"- (pominięto {len(lines) - len(head) - len(tail)} wcześniejszych wiadomości)"] + tail
    return "Facts:\n" + "\n".join(lines)


#: Languages we can name explicitly in the summarisation prompt. Small models
#: ignore "use the same language as the conversation" surprisingly often (a 0.6B
#: model summarised a Polish chat in English and then kept answering in English),
#: so the language is detected here and stated as a direct order instead.
_LANGUAGE_HINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "pl": ("Polish", ("ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż",
                      " jest ", " nie ", " się ", " czy ", " jak ", " dla ")),
    "de": ("German", ("ß", " der ", " die ", " und ", " nicht ", " ist ")),
    "es": ("Spanish", ("¿", "ñ", " que ", " para ", " con ", " está ")),
    "fr": ("French", (" est ", " les ", " pour ", " avec ", " être ", "ç")),
    "it": ("Italian", (" che ", " per ", " sono ", " della ", " più ")),
    "pt": ("Portuguese", (" não ", " para ", " com ", " está ", "ção")),
    "uk": ("Ukrainian", ("ї", "є", "ґ")),
    "ru": ("Russian", ("ы", "ъ", "э", " это ", " для ")),
}


def detect_language(text: Any, *, fallback: str | None = None) -> str | None:
    """Name of the language ``text`` is written in ("Polish"), or ``fallback``.

    Deliberately crude: it only has to pick the language of a chat transcript
    well enough to tell the summariser which one to write in.
    """
    body = clean(text).lower()
    if not body:
        return _LANGUAGE_HINTS.get(str(fallback or "").lower(), (None, ()))[0]
    padded = f" {body} "
    best, best_score = None, 0
    for name, markers in _LANGUAGE_HINTS.values():
        score = sum(padded.count(marker) for marker in markers)
        if score > best_score:
            best, best_score = name, score
    if best_score >= 3:
        return best
    # Too few markers to be sure: the configured UI language wins over a guess,
    # and only plain ASCII prose with no hint at all is assumed to be English.
    hinted = _LANGUAGE_HINTS.get(str(fallback or "").lower(), (None, ()))[0]
    if hinted:
        return hinted
    return best or ("English" if body.isascii() else None)


def build_summary_prompt(transcript: str, *, previous_summary: str | None = None,
                         target_tokens: int = 500,
                         language: str | None = None) -> list[dict[str, Any]]:
    """Messages asking the model for a dense summary of the removed part.

    ``previous_summary`` is the summary being extended: the model is told to
    fold it into the new one, so nothing from the first compaction is lost.
    """
    target = max(150, int(target_tokens))
    low = max(100, min(400, target // 2))
    system = (
        "You compress the beginning of a conversation so it can continue in a small context window. "
        "After this only your summary and the most recent messages remain, so keep everything the "
        "assistant needs to answer follow-up questions.\n\n"
        + (f"WRITE THE SUMMARY IN {language.upper()}. " if language else
           "Write the summary in the SAME LANGUAGE as the conversation. ")
        + "Keep these English section labels and write the content under them:\n"
        "Goal: what the user wants from this conversation, in one or two lines.\n"
        "Facts: facts established so far — names, numbers, definitions, quoted fragments that matter.\n"
        "Decisions: what was agreed, chosen or rejected, and why.\n"
        "Open questions: what is still unanswered or was promised for later.\n"
        "Preferences: how the user wants to be answered (language, tone, format, constraints).\n\n"
        "Rules: be dense and factual, do not invent anything, keep exact names, numbers and quotes, "
        "no code blocks longer than 5 lines, write nothing after the Preferences section. "
        f"Length: about {low}-{target} tokens."
    )
    parts: list[str] = []
    previous = clean(previous_summary)
    if previous:
        parts.append("Summary of the conversation so far (still valid unless the transcript below "
                     "contradicts it — fold it into your answer):\n" + previous)
    body = clean(transcript)
    if body:
        parts.append("Transcript of the messages being removed:\n<transcript>\n" + body + "\n</transcript>")
    else:
        parts.append("Transcript of the messages being removed: (empty)")
    tail = "Write the summary now, with the sections " + ", ".join(SUMMARY_SECTIONS) + "."
    if language:
        tail += f" The prose under the labels must be written in {language}."
    parts.append(tail)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# --------------------------------------------------------------------------- #
# Selecting what to summarise
# --------------------------------------------------------------------------- #


def pending_messages(history: Sequence[dict[str, Any]], *, after_seq: int = 0,
                     current_turn: bool = True) -> list[dict[str, Any]]:
    """Stored messages that may still be summarised, oldest first.

    Skips streaming placeholders and failed messages (exactly like
    :class:`~localcode.chat.context.ContextBuilder`), everything already covered
    by the stored summary (``seq <= after_seq``) and — when ``current_turn`` is
    set — the last user message with whatever follows it: that is the question
    being answered right now and it always stays verbatim.
    """
    usable = [m for m in history
              if m.get("role") in ("user", "assistant") and m.get("status") not in ("error", "streaming")]
    if current_turn:
        for idx in range(len(usable) - 1, -1, -1):
            if usable[idx].get("role") == "user":
                usable = usable[:idx]
                break
    return [m for m in usable if int(m.get("seq") or 0) > int(after_seq or 0)]


@dataclass(frozen=True)
class CompactionPlan:
    """Which stored messages become the summary and which stay verbatim."""

    region: list[dict[str, Any]]       #: messages to summarise, oldest first
    region_tokens: int                 #: their size in the prompt
    kept: int                          #: pending messages kept verbatim after the cut
    upto_seq: int                      #: ``seq`` of the last summarised message

    def __bool__(self) -> bool:
        return bool(self.region)


def _turn_starts(items: Sequence[dict[str, Any]]) -> list[int]:
    """Indices where a user turn begins — the only places the cut may land."""
    return [i for i, m in enumerate(items) if m.get("role") == "user"]


def plan_compaction(items: Sequence[dict[str, Any]], sizes: Sequence[int], *,
                    recent_budget: int) -> CompactionPlan | None:
    """Pick the cut point: keep whole recent turns worth ``recent_budget`` tokens.

    Returns ``None`` when there is nothing worth summarising (too few messages,
    too few tokens, or every pending turn fits in the verbatim budget).
    """
    if len(items) < MIN_REGION_MESSAGES:
        return None
    starts = _turn_starts(items)
    cut = len(items)
    if recent_budget > 0 and starts:
        acc = 0
        bounds = starts + [len(items)]
        for pos in range(len(starts) - 1, -1, -1):
            start, end = bounds[pos], bounds[pos + 1]
            cost = sum(sizes[start:end])
            if acc + cost > recent_budget:
                break
            acc += cost
            cut = start
    if cut <= 0:
        return None
    region = list(items[:cut])
    if len(region) < MIN_REGION_MESSAGES:
        return None
    region_tokens = sum(sizes[:cut])
    if region_tokens < MIN_REGION_TOKENS:
        return None
    return CompactionPlan(region=region, region_tokens=region_tokens, kept=len(items) - cut,
                          upto_seq=int(region[-1].get("seq") or 0))


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class CompactionResult:
    """Outcome of one compaction attempt (``ok`` false ⇒ keep trimming instead)."""

    ok: bool = False
    summary: str = ""
    upto_seq: int = 0
    tokens: int = 0
    messages: int = 0                  #: how many stored messages the summary replaced
    region_tokens: int = 0             #: their size before the summary took their place
    extended: bool = False             #: an existing summary was folded in
    strategy: str = STRATEGY_SUMMARIZE
    error: str | None = None           #: English diagnostic when ``ok`` is false

    @property
    def changed(self) -> bool:
        return bool(self.ok and self.summary)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "uptoSeq": self.upto_seq,
            "tokens": self.tokens,
            "messages": self.messages,
            "extended": self.extended,
            "strategy": self.strategy,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# The compactor
# --------------------------------------------------------------------------- #


class ChatCompactor:
    """Decides when to compact a conversation and produces the summary.

    ``counter`` counts tokens with the loaded model (``TokenCounter``);
    ``summarizer(prompt_messages, max_tokens) -> str`` runs one tool-free,
    thinking-off completion and returns its text. Without a summarizer the
    compactor can still plan and report, but never writes a summary.
    """

    def __init__(self, *, n_ctx: int, max_tokens: int | None, counter: TokenCounter,
                 threshold: float = DEFAULT_THRESHOLD, summarizer: Summarizer | None = None,
                 timeout: float | None = SUMMARY_TIMEOUT_S,
                 language_hint: str | None = None,
                 model_params_b: float | None = None) -> None:
        budget = plan_budget(n_ctx, max_tokens)
        try:
            share = float(threshold)
        except (TypeError, ValueError):
            share = DEFAULT_THRESHOLD
        if not 0.1 <= share <= 1.0:
            share = DEFAULT_THRESHOLD
        self.n_ctx = budget.n_ctx
        self.budget = budget
        self.threshold = share
        self.prompt_budget = budget.prompt
        self.soft_limit = max(1, int(budget.prompt * share))
        self.summary_cap = max(64, int(budget.prompt * SUMMARY_SHARE))
        self.summary_tokens = max(64, min(MAX_SUMMARY_TOKENS, self.summary_cap,
                                          max(MIN_SUMMARY_TOKENS, int(budget.n_ctx * 0.06))))
        self.recent_budget = max(128, int(budget.prompt * RECENT_SHARE) - self.summary_tokens)
        self.counter = counter
        self.summarizer = summarizer
        self.timeout = timeout
        #: ``Settings.language`` ("pl"), used when the transcript is too short to
        #: detect a language from; ``language`` is the name we order the model to use.
        self.language_hint = language_hint
        #: Parameters (in billions) of the model that would write the summary.
        self.model_params_b = model_params_b
        self.language: str | None = detect_language("", fallback=language_hint)
        #: English diagnostic of the last failure (surfaced in logs, not to the user)
        self.last_error: str | None = None

    async def _replaced_tokens(self, plan: "CompactionPlan", previous: str) -> int:
        """Tokens the summary takes the place of: the region plus the old summary block."""
        prior = await self.counter.count(summary_block(previous)) if previous else 0
        return plan.region_tokens + prior

    def needs_compaction(self, prompt_tokens: int) -> bool:
        """True when a prompt of ``prompt_tokens`` crosses the automatic threshold."""
        return int(prompt_tokens) > self.soft_limit

    async def _sizes(self, items: Sequence[dict[str, Any]]) -> list[int]:
        texts = [clean(m.get("content")) for m in items]
        counts = await self.counter.count_many(texts)
        return [c + MESSAGE_OVERHEAD for c in counts]

    async def plan(self, history: Sequence[dict[str, Any]], *, summary_upto_seq: int = 0,
                   force: bool = False) -> CompactionPlan | None:
        """What would be summarised right now (``None`` = nothing worth doing)."""
        items = pending_messages(history, after_seq=summary_upto_seq, current_turn=not force)
        if not items:
            return None
        sizes = await self._sizes(items)
        return plan_compaction(items, sizes, recent_budget=0 if force else self.recent_budget)

    async def _too_big(self, tokens: int, plan: "CompactionPlan",
                       previous: str) -> CompactionResult | None:
        """Refuse a summary that is not smaller than the text it stands in for.

        Compaction must shrink the prompt; a summary longer than the removed
        turns (plus the previous summary it replaces) would grow it and trigger
        yet another compaction on the next turn.
        """
        replaced = await self._replaced_tokens(plan, previous)
        if tokens < replaced:
            return None
        self.last_error = (f"summary is larger than the messages it replaces "
                           f"({tokens} >= {replaced} tokens)")
        log.info("chat summarisation skipped: %s", self.last_error)
        return CompactionResult(ok=False, error=self.last_error, region_tokens=plan.region_tokens)

    async def _transcript_budget(self, previous_summary: str) -> int:
        scaffold = build_summary_prompt("", previous_summary=previous_summary,
                                        target_tokens=self.summary_tokens,
                                        language=self.language)
        fixed = sum(await self.counter.count_many([str(m["content"]) for m in scaffold]))
        room = self.prompt_budget - fixed - int(self.summary_tokens * 1.5) - 32
        return max(256, room)

    async def summarize(self, history: Sequence[dict[str, Any]], *, previous_summary: str = "",
                        summary_upto_seq: int = 0, force: bool = False) -> CompactionResult:
        """Summarise the oldest pending turns with the model.

        Never raises: a missing summarizer, a timeout, an engine error or an
        unusable answer all come back as ``ok=False`` so the caller can fall
        back to plain trimming.
        """
        self.last_error = None
        plan = await self.plan(history, summary_upto_seq=summary_upto_seq, force=force)
        if plan is None:
            return CompactionResult(ok=False, error="nothing to summarize")
        if plan.upto_seq <= 0:
            # A summary stored with ``summary_upto_seq = 0`` covers nothing: the next
            # turn would keep every message AND the summary, and compact again forever.
            self.last_error = "compaction would cover no messages"
            log.warning("chat compaction planned upto_seq=0; skipping it")
            return CompactionResult(ok=False, error=self.last_error)
        if self.summarizer is None:
            self.last_error = "no summarizer available"
            return CompactionResult(ok=False, error=self.last_error)

        previous = clean(previous_summary)
        if self.model_params_b is not None and self.model_params_b < MIN_SUMMARY_PARAMS_B:
            quoted = extractive_summary(plan.region)
            if quoted:
                text = (previous + "\n" + quoted) if previous else quoted
                text, _ = await truncate_to_tokens(text, self.summary_cap, self.counter)
                tokens = await self.counter.count(summary_block(text))
                too_big = await self._too_big(tokens, plan, previous)
                if too_big is not None:
                    return too_big
                return CompactionResult(
                    ok=True, summary=text, upto_seq=plan.upto_seq,
                    tokens=tokens,
                    messages=len(plan.region), region_tokens=plan.region_tokens,
                    extended=bool(previous), strategy=STRATEGY_EXTRACT,
                )
        transcript = render_transcript(plan.region)
        transcript, _ = await truncate_to_tokens(transcript, await self._transcript_budget(previous),
                                                 self.counter, mode="head_tail")
        language = detect_language(transcript, fallback=self.language_hint) or self.language
        prompt = build_summary_prompt(transcript, previous_summary=previous,
                                      target_tokens=self.summary_tokens, language=language)
        want = int(self.summary_tokens * 1.4)
        try:
            call = self.summarizer(prompt, want)
            raw = await (asyncio.wait_for(call, self.timeout) if self.timeout else call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                       # busy engine, timeout, template error, …
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("chat summarisation failed (%s); keeping the trimming fallback", self.last_error)
            return CompactionResult(ok=False, error=self.last_error)
        text = clean_summary(raw)
        if not text:
            self.last_error = "summarizer returned no usable text"
            log.info("chat summarisation produced nothing usable; keeping the trimming fallback")
            return CompactionResult(ok=False, error=self.last_error)
        text, _ = await truncate_to_tokens(text, self.summary_cap, self.counter)
        tokens = await self.counter.count(summary_block(text))
        too_big = await self._too_big(tokens, plan, previous)
        if too_big is not None:
            return too_big
        return CompactionResult(
            ok=True,
            summary=text,
            upto_seq=plan.upto_seq,
            tokens=tokens,
            messages=len(plan.region),
            region_tokens=plan.region_tokens,
            extended=bool(previous),
        )
