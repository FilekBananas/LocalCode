"""Prompt assembly for one chat turn (port of ai_ux ``src/lib/context.ts``)
with a **token budget** for small local context windows.

Scope rules — the single place where "folder = project" is enforced, exactly
as in the reference:

* no folder          → global context only,
* folder ``shared``  → global context AND project context,
* folder ``isolated``→ project context ONLY; nothing global (prompt, profile,
  memory, knowledge) may leak in. Retrieved chunks are re-filtered against
  the resolved scope before they are rendered or cited.

Block order is the reference order: base instructions → global instructions →
"about the user" profile → long-term memory → project block (with project
memory) → conversation instructions → source material. Model-facing text is
English (small models follow English instructions better); the base block
tells the model to answer in the user's language.

Budget (all numbers in tokens of the loaded model)::

    reserve  = min(maxTokens, 25% n_ctx)          kept free for the answer
    prompt   = n_ctx − reserve − margin           everything we may send
    system   ≤ 15% prompt   (instructions, profile, memories — memories first to go)
    summary  ≤ 25% prompt   (the compaction summary of the older turns)
    sources  ≤ 25% prompt   (fewer, then shorter chunks when tight)
    attach.  ≤ 30% prompt   (text of the current message's attachments)
    history  = the rest, newest first, never splitting the current turn

A conversation that was compacted (:mod:`localcode.chat.compaction`) passes its
stored ``summary`` and ``summary_upto_seq`` here: the summary is rendered as a
block right after the system instructions and the messages it covers
(``seq <= summary_upto_seq``) are left out of the history.

The caps only bite under competition: when the history fits completely, the
unused room is handed back to the capped sections in a second pass. The
result is checked with the exact counter (``ChatFormatter.count``) when one is
available; the oldest history messages are dropped until it fits. Everything
that was dropped or shortened is reported in :class:`ContextReport`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SNIPPET_CHARS = 240          #: max characters of a chunk kept in Citation.snippet
MAX_MEMORIES = 40            #: safety cap so a huge memory list cannot swallow the prompt
MAX_MEMORY_CHARS = 400
MAX_ATTACHMENT_CHARS = 20_000  #: per-attachment cap before token budgeting (reference)
MESSAGE_OVERHEAD = 5         #: approx. template tokens per message (role markers)
TINY_CONTEXT = 4096          #: at or below this n_ctx the compact base block is used

SYSTEM_SHARE = 0.15
SUMMARY_SHARE = 0.25
RAG_SHARE = 0.25
ATTACHMENT_SHARE = 0.30
CURRENT_TEXT_SHARE = 0.60    #: hard cap for the text the user typed in this turn
HISTORY_ATTACHMENT_SHARE = 0.10

TRUNCATION_MARKER = "[… truncated to fit the context window …]"

MEMORY_LABELS = {"fact": "fact", "preference": "preference", "note": "note"}

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def clean(value: str | None) -> str:
    return (value or "").strip()


def collapse(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def today_line(now: datetime | None = None) -> str:
    d = now or datetime.now()
    return f"Today's date: {_WEEKDAYS[d.weekday()]}, {d.day} {_MONTHS[d.month - 1]} {d.year}."


def estimate_tokens(text: str) -> int:
    """Fallback estimate (~3 characters per token — conservative for Polish)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3))


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


class TokenCounter:
    """Counts tokens of text pieces with the model tokenizer, with a cache.

    ``count_fn`` is typically ``engine.count_tokens``. When it fails the counter
    falls back to :func:`estimate_tokens` (and stops calling it after repeated
    failures, e.g. when the engine died), so budgeting never breaks a turn.
    """

    def __init__(self, count_fn: Callable[[str], Awaitable[int]] | None = None, *,
                 max_cache: int = 8192, concurrency: int = 8) -> None:
        self._count_fn = count_fn
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._max_cache = max_cache
        self._sem = asyncio.Semaphore(concurrency)
        self._failures = 0

    @property
    def exact(self) -> bool:
        return self._count_fn is not None and self._failures < 3

    @staticmethod
    def estimate(text: str) -> int:
        return estimate_tokens(text)

    async def count(self, text: str) -> int:
        if not text:
            return 0
        key = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=16).hexdigest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        value: int | None = None
        if self.exact:
            try:
                async with self._sem:
                    value = int(await self._count_fn(text))  # type: ignore[misc]
            except asyncio.CancelledError:
                raise
            except Exception:
                self._failures += 1
                log.debug("token counting failed; using the estimate", exc_info=True)
                value = None
        if value is None:
            return estimate_tokens(text)   # estimates are not cached (exact may recover)
        self._cache[key] = value
        if len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)
        return value

    async def count_many(self, texts: list[str]) -> list[int]:
        return list(await asyncio.gather(*(self.count(t) for t in texts)))


async def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter, *,
                             mode: str = "head", marker: str = TRUNCATION_MARKER) -> tuple[str, bool]:
    """Shorten ``text`` to at most ``max_tokens`` (``mode``: ``head`` or ``head_tail``).

    Returns ``(text, truncated)``. The cut lands on whitespace when possible
    and the marker tells the model that something was left out.
    """
    if max_tokens <= 0:
        return "", bool(text)
    total = await counter.count(text)
    if total <= max_tokens:
        return text, False
    marker_tokens = await counter.count(marker)
    budget = max(1, max_tokens - marker_tokens - 2)
    chars_per_token = len(text) / max(1, total)
    target = max(1, int(budget * chars_per_token * 0.95))
    candidate = text
    for _ in range(6):
        candidate = _cut(text, target, mode, marker)
        n = await counter.count(candidate)
        if n <= max_tokens:
            return candidate, True
        target = max(1, int(target * (max_tokens / max(1, n)) * 0.9))
    # Last resort: estimate-driven hard cut.
    return _cut(text, max(1, max_tokens * 2), mode, marker), True


def _cut(text: str, chars: int, mode: str, marker: str) -> str:
    if chars >= len(text):
        return text
    if mode == "head_tail":
        head_chars = int(chars * 0.7)
        tail_chars = max(0, chars - head_chars)
        head = _cut_head(text, head_chars)
        tail = text[len(text) - tail_chars:] if tail_chars else ""
        space = tail.find(" ")
        if 0 <= space < 40:
            tail = tail[space + 1:]
        return f"{head}\n{marker}\n{tail}".strip()
    return f"{_cut_head(text, chars)}\n{marker}"


def _cut_head(text: str, chars: int) -> str:
    head = text[:chars]
    space = max(head.rfind(" "), head.rfind("\n"))
    if space > chars * 0.8:
        head = head[:space]
    return head.rstrip()


# --------------------------------------------------------------------------- #
# Scope resolution (reference, verbatim semantics)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContextScope:
    include_global: bool
    folder_id: str | None


def resolve_scope(folder: dict[str, Any] | None) -> ContextScope:
    """Which knowledge/memory scopes a conversation may see."""
    if not folder:
        return ContextScope(True, None)
    if folder.get("memoryMode") == "isolated":
        return ContextScope(False, folder["id"])
    return ContextScope(True, folder["id"])


def filter_retrieved(chunks: list[dict[str, Any]], scope: ContextScope) -> list[dict[str, Any]]:
    """Drop chunks the scope must never see: anything global for an isolated
    project, and anything belonging to a different project always."""
    out = []
    for chunk in chunks:
        if chunk.get("scope") == "global":
            if scope.include_global:
                out.append(chunk)
            continue
        if not chunk.get("folderId"):
            continue
        if chunk["folderId"] == scope.folder_id:
            out.append(chunk)
    return out


def order_memories(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pinned first, then most recently updated; empty entries dropped; capped."""
    usable = [e for e in entries if clean(e.get("content"))]
    usable.sort(key=lambda e: (0 if e.get("pinned") else 1, -int(e.get("updatedAt") or 0)))
    return usable[:MAX_MEMORIES]


def memory_line(entry: dict[str, Any]) -> str:
    label = MEMORY_LABELS.get(entry.get("kind") or "", "note")
    return f"- [{label}] {truncate(collapse(entry.get('content')), MAX_MEMORY_CHARS)}"


def format_memories(entries: list[dict[str, Any]]) -> str:
    return "\n".join(memory_line(e) for e in order_memories(entries))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #


def base_block(now: datetime | None = None, *, compact: bool = False) -> str:
    if compact:
        return (
            "You are a helpful AI assistant running locally on the user's computer. "
            "Answer in the language the user writes in, use Markdown (code in fenced blocks with a language tag) "
            "and say so plainly when you do not know something.\n" + today_line(now)
        )
    return "\n".join([
        "You are a helpful AI assistant running locally on the user's computer.",
        "Rules for answering:",
        "- Answer in the language the user writes in — if they write in Polish, answer in Polish.",
        "- Format answers in Markdown: headings, lists, tables and bold text where they help readability.",
        "- Put code in fenced code blocks with a language tag (e.g. ts, python, sql).",
        "- Write math inline as $...$ and display math as $$...$$ on its own line.",
        "- Be specific and factual. If you do not know something or data is missing, say so instead of guessing.",
        today_line(now),
    ])


#: Told to the model only when the model tools are actually on its menu.  A
#: 4B model will not reach for a tool it was merely handed: it has to be told
#: that "pobierz mi model" is something it *does*, not something it explains.
MODEL_TOOLS_BLOCK = "\n".join([
    "You can manage the models on this computer with tools:",
    "- `model_search` — find a model in the local store (by name or by what it should do),",
    "- `model_list` — what is installed and what is loaded,",
    "- `model_install` — download and install one,",
    "- `model_load` — load an installed model and continue the conversation on it.",
    "When the user asks you to find, download, install or switch a model, CALL THE TOOL. "
    "Do not describe how they could do it themselves and do not recommend models from memory "
    "— search the store first, then answer with what you found.",
])


def global_prompt_block(settings: dict[str, Any]) -> str:
    prompt = clean(settings.get("globalSystemPrompt"))
    return f"## Global instructions\n{prompt}" if prompt else ""


def profile_block(settings: dict[str, Any]) -> str:
    profile = settings.get("profile") or {}
    if not profile or not profile.get("enabled"):
        return ""
    lines: list[str] = []
    for label, key in (("Name", "displayName"), ("Role / occupation", "role"), ("About", "about"),
                       ("Preferred response style", "responseStyle"),
                       ("Additional instructions", "customInstructions")):
        value = clean(profile.get(key))
        if value:
            lines.append(f"- {label}: {value}")
    if not lines:
        return ""
    return "## About the user\n" + "\n".join(lines)


GLOBAL_MEMORY_HEADER = ("## Long-term memory\nPersistent facts from earlier conversations. Use them when helpful; "
                        "do not mention them when they are unrelated to the question.")
FOLDER_MEMORY_HEADER = "### Project memory"
ISOLATION_NOTE = ("This conversation takes place in an isolated project space: you have no access to global "
                  "instructions, the user profile, memory or knowledge outside this project. Rely only on this "
                  "project's context and on the conversation.")


def global_memory_block(settings: dict[str, Any], memories: list[dict[str, Any]]) -> str:
    if not settings.get("memoryEnabled"):
        return ""
    listing = format_memories(memories)
    return f"{GLOBAL_MEMORY_HEADER}\n{listing}" if listing else ""


def _folder_parts(folder: dict[str, Any]) -> tuple[str, list[str], str]:
    """``(header, middle parts, isolation note or "")`` of a project block."""
    header = f"## Project: {collapse(folder.get('name')) or 'unnamed'}"
    middle: list[str] = []
    description = clean(folder.get("description"))
    if description:
        middle.append(description)
    prompt = clean(folder.get("systemPrompt"))
    if prompt:
        middle.append(f"### Project instructions\n{prompt}")
    extra = clean(folder.get("extraInfo"))
    if extra:
        middle.append(f"### Additional project information\n{extra}")
    note = ISOLATION_NOTE if folder.get("memoryMode") == "isolated" else ""
    return header, middle, note


def folder_block(folder: dict[str, Any], settings: dict[str, Any], memories_folder: list[dict[str, Any]]) -> str:
    header, middle, note = _folder_parts(folder)
    parts = [header, *middle]
    if settings.get("memoryEnabled"):
        listing = format_memories(memories_folder)
        if listing:
            parts.append(f"{FOLDER_MEMORY_HEADER}\n{listing}")
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def conversation_block(conversation: dict[str, Any] | None) -> str:
    prompt = clean((conversation or {}).get("systemPrompt"))
    return f"## Instructions for this conversation\n{prompt}" if prompt else ""


SUMMARY_HEADER = "## Summary of the earlier conversation"
SUMMARY_NOTE = (
    "The beginning of this conversation did not fit into the model's context window and was compressed "
    "into the summary below. Treat it as an established record of what was said; the messages that follow "
    "are the recent part of the same conversation, quoted verbatim. Do not mention the summary to the user "
    "unless they ask about it."
)


def summary_block(summary: str | None) -> str:
    """The block standing in for the turns replaced by a compaction summary."""
    text = clean(summary)
    return f"{SUMMARY_HEADER}\n{SUMMARY_NOTE}\n\n{text}" if text else ""


SOURCES_INTRO = (
    "## Source material\n"
    "Below are excerpts from the user's knowledge base selected for the current question. "
    "Use them when they are relevant and cite them as [1], [2]. "
    "If they do not contain the answer, say so plainly and make clear that you are answering from your own knowledge."
)


def source_entry(index: int, chunk: dict[str, Any], content: str | None = None) -> str:
    header = f"[{index}] {collapse(chunk.get('documentName')) or 'document'} (fragment {chunk.get('idx', 0)})"
    return f"{header}\n{clean(content if content is not None else chunk.get('content'))}"


def sources_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    return SOURCES_INTRO + "\n\n" + "\n\n".join(source_entry(i + 1, c) for i, c in enumerate(chunks))


def citation_for(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "documentId": chunk["documentId"],
        "documentName": chunk.get("documentName") or "",
        "chunkIndex": int(chunk.get("idx") or 0),
        "snippet": truncate(collapse(chunk.get("content")), SNIPPET_CHARS),
        "score": float(chunk.get("score") or 0.0),
    }


@dataclass
class BuiltSystem:
    system: str
    citations: list[dict[str, Any]]
    used_global: bool
    used_folder: bool
    blocks: list[str]


def build_system_prompt(*, settings: dict[str, Any], folder: dict[str, Any] | None = None,
                        conversation: dict[str, Any] | None = None, query: str = "",
                        memories_global: list[dict[str, Any]] | None = None,
                        memories_folder: list[dict[str, Any]] | None = None,
                        retrieved: list[dict[str, Any]] | None = None,
                        now: datetime | None = None, compact: bool = False,
                        model_tools: bool = False) -> BuiltSystem:
    """Unbudgeted assembly — the direct port of ``buildContext``."""
    scope = resolve_scope(folder)
    blocks: list[str] = [base_block(now, compact=compact)]
    if model_tools:
        blocks.append(MODEL_TOOLS_BLOCK)
    used_global = used_folder = False
    if scope.include_global:
        for block in (global_prompt_block(settings), profile_block(settings),
                      global_memory_block(settings, memories_global or [])):
            if block:
                blocks.append(block)
                used_global = True
    if folder:
        blocks.append(folder_block(folder, settings, memories_folder or []))
        used_folder = True
    conv = conversation_block(conversation)
    if conv:
        blocks.append(conv)
    allowed = filter_retrieved(retrieved or [], scope) if clean(query) else []
    sources = sources_block(allowed)
    if sources:
        blocks.append(sources)
        used_global = used_global or any(c.get("scope") == "global" for c in allowed)
        used_folder = used_folder or any(c.get("scope") == "folder" for c in allowed)
    return BuiltSystem("\n\n".join(blocks), [citation_for(c) for c in allowed], used_global, used_folder, blocks)


# --------------------------------------------------------------------------- #
# Title prompt (reference buildTitlePrompt, English)
# --------------------------------------------------------------------------- #

TITLE_SYSTEM = ("You generate short conversation titles. Reply with the title only — no quotes, "
                "no trailing period and no commentary.")


def build_title_prompt(first_user_message: str) -> str:
    message = truncate(collapse(first_user_message), 1000) or "(empty message)"
    return "\n".join([
        "Suggest a title for this conversation based on the user's first message.",
        "Rules:",
        "- at most 6 words,",
        "- no quotes, no trailing period, no emoji,",
        "- in the same language as the user's message,",
        "- name the topic, not the form (do not start with \"Question about\" or \"Request for\").",
        "Reply with the title only, without any commentary.",
        "",
        "User message:",
        message,
    ])


# --------------------------------------------------------------------------- #
# Message text (reference composeText)
# --------------------------------------------------------------------------- #


def precut_attachment(text: str) -> tuple[str, bool]:
    """Character cap applied before token budgeting (reference: 20 000 chars per
    attachment). Keeps the head AND the tail, so the end of a long log survives."""
    if len(text) <= MAX_ATTACHMENT_CHARS:
        return text, False
    return _cut(text, MAX_ATTACHMENT_CHARS, "head_tail", TRUNCATION_MARKER), True


def attachment_section(name: str, text: str) -> str:
    return f"--- Attachment: {name} ---\n{text}\n--- End of attachment: {name} ---"


def image_note(name: str) -> str:
    return f"[Image attachment: {name} — images are not passed to this model]"


def compose_text(message: dict[str, Any]) -> str:
    """Message text plus the extracted content of its text/pdf attachments
    (reference ``composeText``; used as the retrieval query)."""
    segments: list[str] = []
    content = clean(message.get("content"))
    if content:
        segments.append(content)
    for att in message.get("attachments") or []:
        if att.get("kind") == "image":
            continue
        text = clean(att.get("textContent"))
        if text:
            segments.append(attachment_section(att.get("name") or "file", truncate(text, MAX_ATTACHMENT_CHARS)))
    return "\n\n".join(segments)


# --------------------------------------------------------------------------- #
# Budgeted assembly
# --------------------------------------------------------------------------- #


@dataclass
class Budget:
    n_ctx: int
    reserve: int
    prompt: int
    system: int
    rag: int
    attachments: int
    summary: int = 0


def plan_budget(n_ctx: int, max_tokens: int | None, *, scale: float = 1.0) -> Budget:
    n_ctx = max(256, int(n_ctx))
    quarter = n_ctx // 4
    reserve = min(int(max_tokens), quarter) if max_tokens and max_tokens > 0 else quarter
    reserve = max(reserve, min(256, n_ctx // 8))
    margin = max(16, n_ctx // 100)
    prompt = max(64, int((n_ctx - reserve - margin) * scale))
    return Budget(
        n_ctx=n_ctx,
        reserve=reserve,
        prompt=prompt,
        system=int(prompt * SYSTEM_SHARE),
        rag=int(prompt * RAG_SHARE),
        attachments=int(prompt * ATTACHMENT_SHARE),
        summary=int(prompt * SUMMARY_SHARE),
    )


@dataclass
class ContextReport:
    n_ctx: int = 0
    budget: int = 0
    reserve: int = 0
    prompt_tokens: int = 0
    exact: bool = False
    history_total: int = 0          #: earlier usable messages before any trimming
    history_included: int = 0
    dropped_by_limit: int = 0       #: removed by the historyLimit setting
    dropped_messages: int = 0       #: removed to fit the context window
    chunks_total: int = 0
    chunks_included: int = 0
    chunks_shortened: int = 0
    memories_total: int = 0
    memories_included: int = 0
    truncated_blocks: list[str] = field(default_factory=list)
    truncated_attachments: list[str] = field(default_factory=list)
    current_truncated: bool = False
    system_tokens: int = 0          #: size of the whole system message (blocks + summary + sources)
    summary_tokens: int = 0         #: size of the compaction summary block in the prompt
    summary_truncated: bool = False  #: the stored summary did not fit and was shortened
    summarized_messages: int = 0    #: stored messages the summary stands in for
    attempts: int = 1

    @property
    def trimmed(self) -> bool:
        return bool(self.dropped_messages or self.chunks_included < self.chunks_total or self.chunks_shortened
                    or self.memories_included < self.memories_total or self.truncated_blocks
                    or self.truncated_attachments or self.current_truncated or self.summary_truncated)

    def summary_pl(self) -> str | None:
        """Short Polish note for the UI when something did not fit (else ``None``)."""
        if not self.trimmed:
            return None
        parts: list[str] = []
        if self.dropped_messages:
            parts.append(f"pominięto {self.dropped_messages} najstarszych wiadomości")
        skipped_chunks = self.chunks_total - self.chunks_included
        if skipped_chunks > 0 or self.chunks_shortened:
            parts.append(f"użyto {self.chunks_included} z {self.chunks_total} fragmentów bazy wiedzy")
        if self.memories_included < self.memories_total:
            parts.append(f"użyto {self.memories_included} z {self.memories_total} wpisów pamięci")
        if self.truncated_attachments:
            parts.append("skrócono załączniki: " + ", ".join(self.truncated_attachments))
        if self.current_truncated:
            parts.append("skrócono bieżącą wiadomość")
        if self.truncated_blocks:
            parts.append("skrócono instrukcje")
        if self.summary_truncated:
            parts.append("skrócono podsumowanie wcześniejszej rozmowy")
        return f"Kontekst modelu ({self.n_ctx} tokenów): " + "; ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trimmed"] = self.trimmed
        return d


@dataclass
class ChatContext:
    """Everything the service needs for one generation request."""

    messages: list[dict[str, Any]]      #: core OpenAI-style messages, system first
    system: str
    blocks: list[str]
    citations: list[dict[str, Any]]
    used_global: bool
    used_folder: bool
    prompt_tokens: int
    max_output_tokens: int
    report: ContextReport


@dataclass
class _SystemFit:
    blocks: list[str]
    tokens: int
    memories_included: int
    truncated: list[str]
    used_global: bool
    used_folder: bool


@dataclass
class _RagFit:
    block: str
    tokens: int
    included: list[dict[str, Any]]
    shortened: int


@dataclass
class _SummaryFit:
    block: str
    tokens: int
    truncated: bool


@dataclass
class _CurrentFit:
    text: str
    tokens: int
    truncated_attachments: list[str]
    text_truncated: bool


class ContextBuilder:
    """Budgeted port of ``buildContext`` + history assembly.

    ``history`` is the stored transcript (Message DTOs in ``seq`` order) whose
    LAST user message is the current turn; streaming placeholders and failed
    messages are ignored like in the reference.

    ``summary`` / ``summary_upto_seq`` come from a compacted conversation: the
    summary is rendered as its own block and every message with
    ``seq <= summary_upto_seq`` is left out of the history.
    """

    def __init__(self, *, settings: dict[str, Any], folder: dict[str, Any] | None,
                 conversation: dict[str, Any] | None, history: list[dict[str, Any]],
                 memories_global: list[dict[str, Any]] | None = None,
                 memories_folder: list[dict[str, Any]] | None = None,
                 retrieved: list[dict[str, Any]] | None = None, query: str = "",
                 n_ctx: int, max_tokens: int | None, counter: TokenCounter,
                 exact_count: Callable[[list[dict[str, Any]]], Awaitable[int]] | None = None,
                 summary: str | None = None, summary_upto_seq: int | None = None,
                 now: datetime | None = None, model_tools: bool = False) -> None:
        self.settings = settings
        #: The chat may manage models this turn; the system block says so, and
        #: a small model needs to be told, not merely handed the schemas.
        self.model_tools = bool(model_tools)
        self.folder = folder
        self.conversation = conversation
        self.history = history
        self.summary = clean(summary)
        self.summary_upto_seq = int(summary_upto_seq or 0) if self.summary else 0
        self.memories_global = memories_global or []
        self.memories_folder = memories_folder or []
        self.retrieved = retrieved or []
        self.query = query
        self.n_ctx = int(n_ctx)
        self.max_tokens = max_tokens
        self.counter = counter
        self.exact_count = exact_count
        self.now = now
        self.scope = resolve_scope(folder)

    # ------------------------------------------------------------------ system
    async def _fit_system(self, cap: int, compact: bool) -> _SystemFit:
        settings = self.settings
        include_global = self.scope.include_global
        memory_on = bool(settings.get("memoryEnabled"))
        base = base_block(self.now, compact=compact)
        if self.model_tools:
            base = base + "\n\n" + MODEL_TOOLS_BLOCK
        used = await self.counter.count(base)
        truncated: list[str] = []

        texts: dict[str, str] = {}

        async def take(key: str, text: str, *, keep_tail: str = "") -> None:
            nonlocal used
            if not text:
                return
            full = text + (f"\n\n{keep_tail}" if keep_tail else "")
            n = await self.counter.count(full)
            remaining = cap - used
            if n <= remaining:
                texts[key] = full
                used += n
                return
            tail_tokens = await self.counter.count(keep_tail) if keep_tail else 0
            room = remaining - tail_tokens - 4
            if room >= 48:
                short, _ = await truncate_to_tokens(text, room, self.counter)
                texts[key] = short + (f"\n\n{keep_tail}" if keep_tail else "")
                used += await self.counter.count(texts[key])
            elif keep_tail:
                # The isolation note must survive even when nothing else fits.
                texts[key] = keep_tail
                used += tail_tokens
            truncated.append(key)

        await take("conversation", conversation_block(self.conversation))
        if self.folder:
            header, middle, note = _folder_parts(self.folder)
            folder_text = "\n\n".join([header, *middle])
            await take("folder", folder_text, keep_tail=note)
        if include_global:
            await take("global_prompt", global_prompt_block(settings))
            await take("profile", profile_block(settings))

        # Memories, entry by entry: project memory first, then global memory.
        folder_mem: list[str] = []
        global_mem: list[str] = []
        total_mem = 0
        if memory_on:
            candidates: list[tuple[str, dict[str, Any]]] = []
            if self.folder:
                candidates += [("folder", m) for m in order_memories(self.memories_folder)]
            if include_global:
                candidates += [("global", m) for m in order_memories(self.memories_global)]
            total_mem = len(candidates)
            header_cost = {"folder": await self.counter.count(FOLDER_MEMORY_HEADER),
                           "global": await self.counter.count(GLOBAL_MEMORY_HEADER)}
            full = False
            for where, entry in candidates:
                if full:
                    break
                line = memory_line(entry)
                n = await self.counter.count(line) + 1
                bucket = folder_mem if where == "folder" else global_mem
                extra = header_cost[where] + 2 if not bucket else 0
                if used + n + extra > cap:
                    full = True
                    continue
                bucket.append(line)
                used += n + extra

        blocks = [base]
        used_global = used_folder = False
        if include_global:
            for key in ("global_prompt", "profile"):
                if texts.get(key):
                    blocks.append(texts[key])
                    used_global = True
            if global_mem:
                blocks.append(GLOBAL_MEMORY_HEADER + "\n" + "\n".join(global_mem))
                used_global = True
        if self.folder:
            folder_text = texts.get("folder", "")
            if folder_mem:
                mem_block = FOLDER_MEMORY_HEADER + "\n" + "\n".join(folder_mem)
                _, _, note = _folder_parts(self.folder)
                if note and folder_text.endswith(note):
                    folder_text = folder_text[: -len(note)].rstrip() + "\n\n" + mem_block + "\n\n" + note
                else:
                    folder_text = (folder_text + "\n\n" + mem_block) if folder_text else mem_block
            if folder_text:
                blocks.append(folder_text)
                used_folder = True
        if texts.get("conversation"):
            blocks.append(texts["conversation"])
        return _SystemFit(blocks, used, len(folder_mem) + len(global_mem), truncated, used_global, used_folder)

    def _memory_total(self) -> int:
        if not self.settings.get("memoryEnabled"):
            return 0
        total = 0
        if self.folder:
            total += len(order_memories(self.memories_folder))
        if self.scope.include_global:
            total += len(order_memories(self.memories_global))
        return total

    # --------------------------------------------------------------------- RAG
    def _allowed_chunks(self) -> list[dict[str, Any]]:
        return filter_retrieved(self.retrieved, self.scope) if clean(self.query) else []

    async def _fit_rag(self, cap: int) -> _RagFit:
        chunks = self._allowed_chunks()
        if not chunks:
            return _RagFit("", 0, [], 0)
        intro = await self.counter.count(SOURCES_INTRO) + 2
        if intro + 40 > cap:
            return _RagFit("", 0, [], 0)
        used = intro
        entries: list[str] = []
        included: list[dict[str, Any]] = []
        shortened = 0
        for chunk in chunks:
            entry = source_entry(len(included) + 1, chunk)
            n = await self.counter.count(entry) + 2
            if used + n <= cap:
                entries.append(entry)
                included.append(chunk)
                used += n
                continue
            room = cap - used - 2
            header_tokens = await self.counter.count(source_entry(len(included) + 1, chunk, ""))
            if room - header_tokens >= 80:
                content, _ = await truncate_to_tokens(clean(chunk.get("content")), room - header_tokens - 2,
                                                      self.counter)
                entry = source_entry(len(included) + 1, chunk, content)
                entries.append(entry)
                included.append(chunk)
                used += await self.counter.count(entry) + 2
                shortened += 1
            break
        if not included:
            return _RagFit("", 0, [], 0)
        return _RagFit(SOURCES_INTRO + "\n\n" + "\n\n".join(entries), used, included, shortened)

    # ----------------------------------------------------------------- current
    async def _fit_current(self, message: dict[str, Any], att_cap: int, text_cap: int) -> _CurrentFit:
        truncated_atts: list[str] = []
        content = clean(message.get("content"))
        text_truncated = False
        if content:
            content, text_truncated = await truncate_to_tokens(content, text_cap, self.counter, mode="head_tail")
        segments = [content] if content else []
        remaining = att_cap
        for att in message.get("attachments") or []:
            name = att.get("name") or "file"
            if att.get("kind") == "image":
                segments.append(image_note(name))
                continue
            text = clean(att.get("textContent"))
            if not text:
                continue
            text, precut = precut_attachment(text)
            wrapper = await self.counter.count(attachment_section(name, ""))
            room = remaining - wrapper
            if room < 32:
                segments.append(f"[Attachment {name} omitted: it does not fit into the context window]")
                truncated_atts.append(name)
                continue
            short, cut = await truncate_to_tokens(text, room, self.counter, mode="head_tail")
            if cut or precut:
                truncated_atts.append(name)
            section = attachment_section(name, short)
            segments.append(section)
            remaining -= await self.counter.count(section)
        text = "\n\n".join(segments)
        tokens = await self.counter.count(text) + MESSAGE_OVERHEAD
        return _CurrentFit(text, tokens, truncated_atts, text_truncated)

    async def _history_text(self, message: dict[str, Any], att_cap: int) -> str:
        segments: list[str] = []
        content = clean(message.get("content"))
        if content:
            segments.append(content)
        if message.get("role") == "user":
            for att in message.get("attachments") or []:
                name = att.get("name") or "file"
                if att.get("kind") == "image":
                    segments.append(image_note(name))
                    continue
                text = clean(att.get("textContent"))
                if text:
                    short, _ = await truncate_to_tokens(precut_attachment(text)[0], att_cap,
                                                        self.counter, mode="head_tail")
                    segments.append(attachment_section(name, short))
        return "\n\n".join(segments)

    # ----------------------------------------------------------------- summary
    async def _fit_summary(self, cap: int) -> _SummaryFit:
        """The compaction summary block, shortened to ``cap`` tokens if need be."""
        if not self.summary:
            return _SummaryFit("", 0, False)
        block = summary_block(self.summary)
        tokens = await self.counter.count(block)
        if tokens <= cap:
            return _SummaryFit(block, tokens, False)
        wrapper = await self.counter.count(summary_block("…"))
        room = cap - wrapper
        if room < 48:            # no room at all: the header alone would be noise
            return _SummaryFit("", 0, True)
        short, _ = await truncate_to_tokens(self.summary, room, self.counter, mode="head_tail")
        block = summary_block(short)
        return _SummaryFit(block, await self.counter.count(block), True)

    # -------------------------------------------------------------------- main
    def _usable_history(self) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int, int]:
        """``(earlier messages, current user message, dropped by limit, summarised)``."""
        usable = [m for m in self.history
                  if m.get("role") in ("user", "assistant") and m.get("status") not in ("error", "streaming")]
        limit = int(self.settings.get("defaults", {}).get("historyLimit") or 0)
        dropped_by_limit = 0
        if limit > 0 and len(usable) > limit:
            dropped_by_limit = len(usable) - limit
            usable = usable[-limit:]
        current = None
        for idx in range(len(usable) - 1, -1, -1):
            if usable[idx].get("role") == "user":
                current = usable[idx]
                # Anything after the current user turn (should not exist) is ignored.
                usable = usable[:idx]
                break
        summarized = 0
        if self.summary_upto_seq > 0:
            kept = [m for m in usable if int(m.get("seq") or 0) > self.summary_upto_seq]
            summarized = len(usable) - len(kept)
            usable = kept
        return usable, current, dropped_by_limit, summarized

    async def build(self, *, scale: float = 1.0) -> ChatContext:
        attempts = 0
        while True:
            attempts += 1
            ctx, over = await self._build_once(scale)
            ctx.report.attempts = attempts
            if not over or attempts >= 4:
                return ctx
            scale *= 0.8

    async def _build_once(self, scale: float) -> tuple[ChatContext, bool]:
        budget = plan_budget(self.n_ctx, self.max_tokens, scale=scale)
        compact = self.n_ctx <= TINY_CONTEXT
        earlier, current, dropped_by_limit, summarized = self._usable_history()
        report = ContextReport(n_ctx=self.n_ctx, budget=budget.prompt, reserve=budget.reserve,
                               dropped_by_limit=dropped_by_limit, memories_total=self._memory_total(),
                               chunks_total=len(self._allowed_chunks()), history_total=len(earlier),
                               summarized_messages=summarized)

        caps = {"system": budget.system, "rag": budget.rag, "attachments": budget.attachments,
                "summary": budget.summary}
        text_cap = int(budget.prompt * CURRENT_TEXT_SHARE)
        history_att_cap = max(64, int(budget.prompt * HISTORY_ATTACHMENT_SHARE))

        history_items: list[tuple[dict[str, Any], str, int]] = []
        for message in earlier:
            if message.get("role") == "assistant" and not clean(message.get("content")):
                continue
            text = await self._history_text(message, history_att_cap)
            if not text:
                continue
            history_items.append((message, text, await self.counter.count(text) + MESSAGE_OVERHEAD))

        for pass_no in (1, 2):
            system_fit = await self._fit_system(caps["system"], compact)
            summary_fit = await self._fit_summary(caps["summary"])
            rag_fit = await self._fit_rag(caps["rag"])
            current_fit = (await self._fit_current(current, caps["attachments"], text_cap)
                           if current else _CurrentFit("", 0, [], False))
            fixed = (system_fit.tokens + summary_fit.tokens + rag_fit.tokens + MESSAGE_OVERHEAD
                     + current_fit.tokens)
            remaining = budget.prompt - fixed
            kept: list[tuple[dict[str, Any], str, int]] = []
            for item in reversed(history_items):
                if item[2] > remaining:
                    break
                kept.append(item)
                remaining -= item[2]
            kept.reverse()
            exhausted = len(kept) == len(history_items)
            capped = {
                "system": bool(system_fit.truncated) or system_fit.memories_included < report.memories_total,
                "rag": len(rag_fit.included) < report.chunks_total or rag_fit.shortened > 0,
                "attachments": bool(current_fit.truncated_attachments),
                "summary": summary_fit.truncated,
            }
            if pass_no == 2 or not exhausted or remaining < 64 or not any(capped.values()):
                break
            # Hand the unused room back to the sections that were cut.
            spare = remaining
            for key in ("summary", "attachments", "rag", "system"):
                if capped[key]:
                    caps[key] += spare

        extra_blocks = ([summary_fit.block] if summary_fit.block else []) + \
                       ([rag_fit.block] if rag_fit.block else [])
        system_text = "\n\n".join(system_fit.blocks + extra_blocks)
        current_text = (current_fit.text or "(empty message)") if current is not None else None

        def trim_leading(items: list[tuple[dict[str, Any], str, int]]) -> list[tuple[dict[str, Any], str, int]]:
            # History handed to the template must start with a user turn.
            while items and items[0][0].get("role") != "user":
                items = items[1:]
            return items

        def assemble(items: list[tuple[dict[str, Any], str, int]]) -> list[dict[str, Any]]:
            msgs: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
            msgs += [{"role": m["role"], "content": text} for m, text, _ in items]
            if current_text is not None:
                msgs.append({"role": "user", "content": current_text})
            return normalize_turns(msgs)

        kept = trim_leading(kept)
        messages = assemble(kept)
        estimated = (await self.counter.count(system_text) + MESSAGE_OVERHEAD + current_fit.tokens
                     + sum(item[2] for item in kept))
        prompt_tokens = estimated
        over = False
        hard_limit = self.n_ctx - budget.reserve
        if self.exact_count is not None:
            try:
                prompt_tokens = int(await self.exact_count(messages))
                report.exact = True
                while prompt_tokens > hard_limit and kept:
                    kept = trim_leading(kept[1:])       # drop the oldest turn
                    messages = assemble(kept)
                    prompt_tokens = int(await self.exact_count(messages))
                over = prompt_tokens > hard_limit
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("exact prompt count failed; relying on the estimate", exc_info=True)
                prompt_tokens = estimated
                report.exact = False

        report.history_total = len(history_items)
        report.history_included = len(kept)
        report.dropped_messages = max(0, len(history_items) - len(kept))
        report.chunks_included = len(rag_fit.included)
        report.chunks_shortened = rag_fit.shortened
        report.memories_included = system_fit.memories_included
        report.truncated_blocks = list(system_fit.truncated)
        report.truncated_attachments = list(current_fit.truncated_attachments)
        report.current_truncated = current_fit.text_truncated
        report.summary_tokens = summary_fit.tokens
        report.summary_truncated = summary_fit.truncated
        report.system_tokens = system_fit.tokens + summary_fit.tokens + rag_fit.tokens
        report.prompt_tokens = prompt_tokens

        if self.max_tokens == 0:                  # "bez limitu": all the window has left
            wanted = self.n_ctx - prompt_tokens - 4
        else:
            wanted = int(self.max_tokens) if self.max_tokens and self.max_tokens > 0 else budget.reserve
        max_output = max(16, min(wanted, self.n_ctx - prompt_tokens - 4))

        used_global = system_fit.used_global or any(c.get("scope") == "global" for c in rag_fit.included)
        used_folder = system_fit.used_folder or any(c.get("scope") == "folder" for c in rag_fit.included)
        blocks = system_fit.blocks + extra_blocks
        ctx = ChatContext(
            messages=messages,
            system=system_text,
            blocks=blocks,
            citations=[citation_for(c) for c in rag_fit.included],
            used_global=used_global,
            used_folder=used_folder,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output,
            report=report,
        )
        return ctx, over


def normalize_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make the transcript acceptable to strict chat templates: the first turn
    after the system prompt is a user turn and roles alternate (consecutive
    same-role turns are merged)."""
    if not messages:
        return messages
    head = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[len(head):]
    while rest and rest[0].get("role") != "user":
        rest = rest[1:]
    merged: list[dict[str, Any]] = []
    for msg in rest:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1] = {**merged[-1], "content": f"{merged[-1]['content']}\n\n{msg['content']}"}
        else:
            merged.append(dict(msg))
    return head + merged


async def build_chat_context(**kwargs: Any) -> ChatContext:
    """Convenience wrapper: ``ContextBuilder(**kwargs).build()`` (``scale`` optional)."""
    scale = kwargs.pop("scale", 1.0)
    return await ContextBuilder(**kwargs).build(scale=scale)
