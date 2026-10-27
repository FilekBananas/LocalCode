"""Context-window management for the LocalCode coding agent (small local models).

Local models run with small windows (4k–32k tokens), so every request is fitted
into a budget before it is sent::

    n_ctx ─┬─ reserve_output        kept free for the answer; default clamp(20 % of n_ctx, 512, 8192)
           └─ usable = n_ctx − reserve_output
                └─ soft limit = compact_threshold · usable      (compaction trigger)

``ContextManager.fit(messages, tools)`` counts the prompt with the injected
*exact* counter (``ChatFormatter.count`` — the model's own template and
tokenizer). While the prompt is at or under the soft limit the messages are
returned untouched. Above it, progressively more destructive stages run, each
followed by a recount, and the pipeline stops as soon as the prompt is under
the soft limit again:

1. ``mask_tool_results`` — every tool result except the K most recent ones
   (K = 2 for n_ctx ≤ 8k, else 5) is replaced by a one-line stub
   (``[old output of read_file removed to save context — call the tool again if
   you need it]`` plus the first line of the original output).
2. ``shrink_messages`` — long assistant texts, long tool-call arguments (e.g. the
   ``content`` of an old ``write_file``) and long intermediate user messages
   outside the most recent units are cut down (head + tail, explicit marker).
3. ``summarize`` — everything except the system message, the original task (the
   first user message; trimmed unless it is also the latest user message) and
   the most recent units is replaced by a ``[Summary of earlier work]`` block
   written by the injected ``summarizer`` (the model itself, prompted with
   ``prompts.compaction_prompt``). If there is no summarizer, or it fails, times
   out or returns nothing useful, a deterministic extractive summary is used
   (files read/changed, commands with exit codes, searches, user messages,
   errors, the todo list).
4. ``drop_oldest`` — the oldest remaining units are dropped (atomically).
5. ``truncate`` — last resort when even the minimal context (system, task/summary,
   latest user message, final unit) exceeds ``usable``: the final tool output,
   the summary, the task, the latest user message and finally the system message
   are hard-truncated. ``fit`` never raises because of size.

Every applied stage is reported as an ``events.Compaction(before, after,
strategy)``; ``strategy`` is one of the five names above.

Invariants (property-tested in ``tests/agent/test_context.py``):

* an assistant message with ``tool_calls`` is never separated from its tool
  results (both are kept or dropped together, as one *unit*);
* the system message (index 0) is never dropped;
* the latest user message is always kept (verbatim unless stage 5 had to cut
  it); a user message written by a person is also pinned when the latest one is
  an automatic nudge from the agent loop (see ``prompts.is_nudge``);
* the output is a valid alternation for strict chat templates: after the system
  message comes a user message, no two user or two assistant messages are
  adjacent, tool messages directly follow the assistant that called them. To get
  this, a user message that would follow another user message is *merged* into
  it (separated by a blank line). In particular the summary block lives inside
  the first user message, right after the original task, instead of being a
  separate message (Gemma/Mistral templates reject consecutive user turns);
* input messages are never mutated; changed messages are new dicts and
  unchanged messages are the very same objects.

Prompt-prefix (KV cache) stability: llama-server and the MLX worker reuse the
KV cache for the longest common prefix of consecutive prompts. ``fit`` therefore
rewrites nothing while under the soft limit (append-only history), and when it
must compact it does so in *batches* (all old tool results at once, one summary
for the whole old region) so the following requests are append-only again.
Callers should adopt the returned list as their working history — re-fitting the
original, unmasked history on every step would move the masking boundary by one
message per step and invalidate the cache tail each time. Only compaction
rewrites early messages.

Token estimates: ``estimate_tokens`` is a cheap, conservative heuristic
(~3.5 characters per token for letters and spaces, 0.5 token per ASCII
punctuation mark, one token per digit and per non-ASCII character — Qwen-style
tokenizers split digits individually and Polish diacritics cost about a token
each). It is used for pre-checks and for planning inside the pipeline; every
decision about the real budget is verified with the exact counter.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import logging
import math
import re
import string
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from localcode.agent.tools.base import ToolLimits
from localcode.events import Compaction, ContextUpdate

log = logging.getLogger(__name__)

Message = dict[str, Any]
ToolSchema = dict[str, Any]
#: ``counter(messages, tools) -> int`` (normally async: ``ChatFormatter.count``).
TokenCounter = Callable[[list[Message], "list[ToolSchema] | None"], "Awaitable[int] | int"]
#: ``summarizer(prompt_messages) -> str`` — runs the model on the compaction prompt.
Summarizer = Callable[[list[Message]], Awaitable[str]]

__all__ = [
    "CHARS_PER_TOKEN",
    "ContextManager",
    "DROPPED_NOTE",
    "STRATEGIES",
    "SUMMARY_FOOTER",
    "SUMMARY_HEADER",
    "Summarizer",
    "TokenCounter",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
    "limits_for",
    "structure_problems",
    "truncate",
]

CHARS_PER_TOKEN = 3.5

SUMMARY_HEADER = "[Summary of earlier work]"
SUMMARY_FOOTER = "[End of summary]"
DROPPED_NOTE = "(Some older messages were removed to fit the context window.)"
MASK_TEMPLATE = "[old output of {name} removed to save context — call the tool again if you need it]"
_MASK_SIGNATURE = "removed to save context — call the tool again if you need it]"
_CUT_HINT = "cut to fit the context window"
_SEP = "\n\n"

STRATEGY_MASK = "mask_tool_results"
STRATEGY_SHRINK = "shrink_messages"
STRATEGY_SUMMARIZE = "summarize"
STRATEGY_DROP = "drop_oldest"
STRATEGY_TRUNCATE = "truncate"
#: Compaction strategies in the order the pipeline applies them.
STRATEGIES = (STRATEGY_MASK, STRATEGY_SHRINK, STRATEGY_SUMMARIZE, STRATEGY_DROP, STRATEGY_TRUNCATE)

_MSG_OVERHEAD = 4          # role markers / separators of one chat-template turn
_CALL_OVERHEAD = 8         # tool-call wrapper (name/arguments keys, tags)
_IMAGE_TOKENS = 256        # rough cost of one image part (vision models)
_PUNCT_DELETE = str.maketrans("", "", string.punctuation)
_DIGIT_DELETE = str.maketrans("", "", string.digits)

_READ_TOOLS = {"read_file", "view", "cat"}
_EDIT_TOOLS = {"edit_file", "write_file", "multi_edit", "apply_patch", "create_file"}
_SEARCH_TOOLS = {"grep", "glob", "list_dir", "find", "search"}
_BASH_TOOLS = {"bash", "shell", "run_command"}
_TODO_TOOLS = {"todo_write", "todo"}
_EXIT_RE = re.compile(r"(?i)\bexit(?:\s+code|\s+status)?\s*[:=]?\s*(-?\d+)")
_ERROR_START_RE = re.compile(r"(?i)^\s*(error|błąd|blad|failed|failure|exception|fatal|traceback)\b")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.S)


# --------------------------------------------------------------------------- #
# Token estimates
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate for ``text`` (see module docstring)."""
    if not text:
        return 0
    n = len(text)
    non_ascii = n - len(text.encode("ascii", "ignore"))
    digits = n - len(text.translate(_DIGIT_DELETE))
    punct = n - len(text.translate(_PUNCT_DELETE))
    other = n - non_ascii - digits - punct
    return math.ceil(digits + non_ascii + 0.5 * punct + other / CHARS_PER_TOKEN)


def _content_text(content: Any) -> str:
    """Plain text of a message ``content`` (str, OpenAI parts list or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _image_parts(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for p in content if isinstance(p, dict) and p.get("type") not in (None, "text"))


def _args_text(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(arguments)


def estimate_message_tokens(message: Message) -> int:
    """Estimated tokens of one core message (``reasoning`` is never sent, so it is ignored)."""
    content = message.get("content")
    n = _MSG_OVERHEAD + estimate_tokens(_content_text(content)) + _IMAGE_TOKENS * _image_parts(content)
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {} if isinstance(call, dict) else {}
        n += _CALL_OVERHEAD + estimate_tokens(str(fn.get("name") or "")) + estimate_tokens(_args_text(fn.get("arguments")))
    if message.get("role") == "tool":
        n += 4 + estimate_tokens(str(message.get("name") or ""))
    return n


def estimate_messages_tokens(messages: Iterable[Message], tools: Sequence[ToolSchema] | None = None) -> int:
    """Estimated prompt size of ``messages`` plus tool schemas (+ generation prompt)."""
    n = 3 + sum(estimate_message_tokens(m) for m in messages)
    if tools:
        try:
            n += estimate_tokens(json.dumps(list(tools), ensure_ascii=False))
        except (TypeError, ValueError):
            n += estimate_tokens(str(tools))
    return n


def _fit_text_to_tokens(text: str, max_tokens: int, *, head_ratio: float = 0.5) -> str:
    """Truncate ``text`` (head + tail) until ``estimate_tokens`` is within ``max_tokens``."""
    max_tokens = max(0, int(max_tokens))
    est = estimate_tokens(text)
    if est <= max_tokens:
        return text
    chars = int(len(text) * max_tokens / max(1, est))
    for _ in range(8):
        cut = truncate(text, chars, head_ratio=head_ratio)
        if estimate_tokens(cut) <= max_tokens:
            return cut
        chars = int(chars * 0.85)
    # every character costs at most one estimated token, so this always fits
    return text[:max_tokens]


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def _omitted_marker(n_lines: int, n_chars: int, hint: str | None) -> str:
    noun = "line" if n_lines == 1 else "lines"
    marker = f"[… {n_lines} {noun} ({n_chars} chars) omitted"
    if hint:
        marker += f" — {hint}"
    return marker + " …]"


def truncate(text: str, max_chars: int, *, head_ratio: float = 0.5, hint: str | None = None) -> str:
    """Line-aware head + tail truncation to at most ``max_chars`` characters.

    Whole lines are kept from the start and the end; the middle is replaced by
    ``[… N lines (M chars) omitted …]`` on its own line (with `` — hint`` before
    the closing ellipsis when ``hint`` is given). N counts the lines that are not
    shown completely, M the characters removed. A first/last line longer than
    its budget is cut inside the line. ``head_ratio`` is the share of the budget
    given to the head. The result never exceeds ``max_chars``.
    """
    if not isinstance(text, str):
        text = _content_text(text)
    max_chars = max(0, int(max_chars))
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    budget = max_chars - len(_omitted_marker(len(lines), len(text), hint)) - 2  # marker on its own line
    if budget < 8:
        return text[:max_chars]
    head_budget = int(budget * min(1.0, max(0.0, head_ratio)))

    head: list[str] = []
    used = 0
    i = 0
    while i < len(lines) and used + len(lines[i]) + 1 <= head_budget:
        head.append(lines[i])
        used += len(lines[i]) + 1
        i += 1
    head_text = "\n".join(head)
    partial_head = False
    if not head and head_budget >= 2:
        head_text = lines[0][: head_budget - 1]  # line 0 is only partly shown
        partial_head = True

    tail_budget = budget - len(head_text) - (1 if head_text else 0)
    tail: list[str] = []
    tused = 0
    j = len(lines) - 1
    lowest = i + 1 if partial_head else i
    while j >= lowest and tused + len(lines[j]) + 1 <= tail_budget:
        tail.append(lines[j])
        tused += len(lines[j]) + 1
        j -= 1
    tail.reverse()
    tail_text = "\n".join(tail)
    if not tail and j >= i and tail_budget >= 2:
        tail_text = lines[j][-(tail_budget - 1):]  # line j is only partly shown

    omitted_lines = max(1, j - i + 1)
    omitted_chars = max(1, len(text) - len(head_text) - len(tail_text))
    marker = _omitted_marker(omitted_lines, omitted_chars, hint)
    out = "\n".join(p for p in (head_text, marker, tail_text) if p)
    return out if len(out) <= max_chars else out[:max_chars]


# --------------------------------------------------------------------------- #
# Tool limits
# --------------------------------------------------------------------------- #

# (max n_ctx, max_output_chars, read_default_lines, grep_max_matches, list_max_entries)
#
# The whole ladder exists because this app was built for small local models: a
# 4k window cannot afford a 24 000-character file listing.  The top of the
# ladder is for the opposite case — a model with a six-figure window, which is
# what the remote providers offer — where clipping a file to 400 lines just
# makes the model ask for the rest.
_LIMIT_TABLE: tuple[tuple[int, int, int, int, int], ...] = (
    (4096, 3_000, 60, 20, 50),
    (8192, 6_000, 120, 40, 100),
    (16384, 10_000, 200, 60, 150),
    (32768, 16_000, 300, 100, 200),
    (131_072, 24_000, 400, 150, 300),
)
_LIMIT_LARGE = (60_000, 1_500, 400, 800)


def limits_for(n_ctx: int, usable: int | None = None) -> ToolLimits:
    """Tool output limits scaled to a context window of ``n_ctx`` tokens."""
    row = next((r[1:] for r in _LIMIT_TABLE if n_ctx <= r[0]), _LIMIT_LARGE)
    max_chars, lines, matches, entries = row
    if usable is not None:
        # never let one tool result take more than ~35 % of the usable window
        max_chars = max(800, min(max_chars, int(usable * 0.35 * CHARS_PER_TOKEN)))
    return ToolLimits(
        max_output_chars=max_chars,
        read_default_lines=lines,
        grep_max_matches=matches,
        list_max_entries=entries,
    )


# --------------------------------------------------------------------------- #
# Message structure helpers
# --------------------------------------------------------------------------- #


def _role(m: Message) -> str:
    return str(m.get("role") or "")


def _body_start(msgs: Sequence[Message]) -> int:
    return 1 if msgs and _role(msgs[0]) == "system" else 0


def _units(msgs: Sequence[Message], start: int) -> list[tuple[int, int]]:
    """Split ``msgs[start:]`` into atomic units ``[s, e)``.

    An assistant message with tool calls forms one unit with the tool messages
    that follow it; tool messages without a caller stay glued to the preceding
    unit; every other message is a unit of its own.
    """
    units: list[tuple[int, int]] = []
    i, n = start, len(msgs)
    while i < n:
        role = _role(msgs[i])
        j = i + 1
        if role == "assistant" and msgs[i].get("tool_calls"):
            while j < n and _role(msgs[j]) == "tool":
                j += 1
        elif role == "tool":
            while j < n and _role(msgs[j]) == "tool":
                j += 1
            if units:
                s, _ = units.pop()
                units.append((s, j))
                i = j
                continue
        units.append((i, j))
        i = j
    return units


def _join(*parts: Any) -> Any:
    """Join message contents with a blank line (lists of parts when any part is multimodal)."""
    items = [p for p in parts if p is not None and not (isinstance(p, (str, list)) and len(p) == 0)]
    if not items:
        return ""
    if all(isinstance(p, str) for p in items):
        return _SEP.join(items)
    out: list[Any] = []
    for p in items:
        if isinstance(p, list):
            out.extend(p)
        else:
            out.append({"type": "text", "text": p if isinstance(p, str) else str(p)})
    return out


def _split_summary_text(text: str) -> tuple[str, str | None, str | None]:
    idx = text.find(SUMMARY_HEADER)
    if idx < 0:
        return text, None, None
    task = text[:idx]
    if task.endswith(_SEP):
        task = task[: -len(_SEP)]
    after = text[idx + len(SUMMARY_HEADER):]
    if after.startswith("\n"):
        after = after[1:]
    end = after.find(SUMMARY_FOOTER)
    if end < 0:
        return task, after.strip(), None
    summary = after[:end]
    if summary.endswith("\n"):
        summary = summary[:-1]
    rest = after[end + len(SUMMARY_FOOTER):]
    if rest.startswith(_SEP):
        rest = rest[len(_SEP):]
    return task, summary, (rest or None)


def _split_head(content: Any) -> tuple[Any, str | None, Any]:
    """Split a first-user-message content into (task, summary, rest)."""
    if isinstance(content, str):
        return _split_summary_text(content)
    if isinstance(content, list):
        for k, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].startswith(SUMMARY_HEADER):
                _, summary, _ = _split_summary_text(part["text"])
                return (content[:k] or None), summary, (content[k + 1:] or None)
    return content, None, None


def _summary_block(summary: str) -> str:
    return f"{SUMMARY_HEADER}\n{summary}\n{SUMMARY_FOOTER}"


def _build_head(task: Any, summary: str | None, rest: Any) -> Any:
    return _join(task, _summary_block(summary) if summary is not None else None, rest)


def _repair(msgs: list[Message], body: int) -> list[Message]:
    """Merge adjacent user messages and adjacent plain assistant messages."""
    out = list(msgs[:body])
    for m in msgs[body:]:
        prev = out[-1] if len(out) > body else None
        role = _role(m)
        if prev is not None and role == "user" and _role(prev) == "user":
            merged = dict(prev)
            merged["content"] = _join(prev.get("content"), m.get("content"))
            out[-1] = merged
        elif prev is not None and role == "assistant" and _role(prev) == "assistant" and not prev.get("tool_calls"):
            merged = dict(m)
            merged["content"] = _join(_content_text(prev.get("content")), _content_text(m.get("content")))
            out[-1] = merged
        else:
            out.append(m)
    return out


def _call_names(msgs: Iterable[Message]) -> dict[str, str]:
    names: dict[str, str] = {}
    for m in msgs:
        for call in m.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("id"):
                names[str(call["id"])] = str((call.get("function") or {}).get("name") or "tool")
    return names


def _shrink_value(value: Any, limit: int, slack: float) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) > limit * slack:
            return truncate(value, limit, head_ratio=0.6), True
        return value, False
    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for k, v in value.items():
            out[k], ch = _shrink_value(v, limit, slack)
            changed = changed or ch
        return (out, True) if changed else (value, False)
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for v in value:
            nv, ch = _shrink_value(v, limit, slack)
            items.append(nv)
            changed = changed or ch
        return (items, True) if changed else (value, False)
    return value, False


def _shrink_calls(calls: list[Any], limit: int, slack: float = 1.0) -> tuple[list[Any], bool]:
    """Shrink long string arguments of tool calls (dict arguments only)."""
    changed = False
    out: list[Any] = []
    for call in calls:
        fn = call.get("function") if isinstance(call, dict) else None
        args = fn.get("arguments") if isinstance(fn, dict) else None
        if isinstance(args, dict):
            new_args, ch = _shrink_value(args, limit, slack)
            if ch:
                call = copy.deepcopy(call)
                call["function"]["arguments"] = new_args
                changed = True
        out.append(call)
    return out, changed


def structure_problems(messages: Sequence[Message]) -> list[str]:
    """Describe chat-template structure violations in ``messages`` (empty = valid).

    Checked: system only at index 0; the first non-system message is a user
    message; no two adjacent user or assistant messages; tool messages directly
    follow the assistant whose ``tool_calls`` contain their ``tool_call_id``;
    every tool call has a result (except calls of the very last message).
    """
    problems: list[str] = []
    body = _body_start(messages)
    for i, m in enumerate(messages):
        if _role(m) == "system" and i != 0:
            problems.append(f"#{i}: system message not at index 0")
    if len(messages) > body and _role(messages[body]) != "user":
        problems.append(f"#{body}: first non-system message is {_role(messages[body])!r}, not 'user'")
    open_ids: set[str] = set()
    caller: int | None = None
    for i in range(body, len(messages)):
        m = messages[i]
        role = _role(m)
        prev_role = _role(messages[i - 1]) if i > body else None
        if role != "tool":
            if open_ids and caller is not None:
                problems.append(f"#{caller}: tool calls without results: {sorted(open_ids)}")
            open_ids, caller = set(), None
        if role == "user" and prev_role == "user":
            problems.append(f"#{i}: two consecutive user messages")
        if role == "assistant" and prev_role == "assistant":
            problems.append(f"#{i}: two consecutive assistant messages")
        if role == "assistant" and m.get("tool_calls"):
            open_ids = {str(c.get("id")) for c in m["tool_calls"] if isinstance(c, dict)}
            caller = i
        elif role == "tool":
            if caller is None:
                problems.append(f"#{i}: tool message without a preceding assistant tool call")
            else:
                tid = str(m.get("tool_call_id"))
                if tid in open_ids:
                    open_ids.discard(tid)
                else:
                    problems.append(f"#{i}: tool_call_id {tid!r} does not match the calling assistant")
    return problems


def _clean_summary(raw: Any) -> str:
    """Strip reasoning, fences and our own markers from a model-written summary."""
    if not isinstance(raw, str):
        return ""
    text = _THINK_RE.sub("", raw)
    if "<think>" in text:  # unterminated reasoning: nothing usable after it
        text = text.split("<think>", 1)[0]
    text = text.replace("</think>", "").replace(SUMMARY_HEADER, "").replace(SUMMARY_FOOTER, "").strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    return text if len(text) >= 20 else ""


def _first_line(text: str, limit: int = 120) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def _looks_failed(text: str) -> bool:
    return bool(_ERROR_START_RE.match(text)) or "Traceback (most recent call last)" in text


def _is_masked(text: str) -> bool:
    return _MASK_SIGNATURE in text


def _nudge_checker() -> Callable[[str], bool]:
    try:
        from localcode.agent.prompts import is_nudge
    except Exception:  # pragma: no cover - prompts is part of this package
        return lambda _text: False
    return is_nudge


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #


@dataclass
class _Pass:
    """Parameters of one compaction pass."""

    target: int               # stop as soon as the prompt is at or under this
    hard_limit: int           # the final prompt must fit under this (else stage 5)
    keep_tool_results: int    # K most recent tool results kept verbatim
    recent_share: float       # max share of ``usable`` kept verbatim at the end when summarizing
    force: bool = False       # run mask/shrink/summarize even when already under ``target``
    pessimistic: bool = False  # measure as max(exact, estimate) — the counter proved unreliable


class ContextManager:
    """Keeps an agent conversation inside the model's context window.

    ``counter(messages, tools) -> int`` must return the exact prompt size in
    tokens (async or sync; ``ChatFormatter.count``). ``summarizer(prompt_messages)
    -> str`` is an optional async callable that runs the model on
    ``prompt_messages`` (built with ``prompts.compaction_prompt``) without tools,
    with thinking disabled and ``max_tokens`` ≈ ``summary_tokens``, and returns
    the text of the answer.
    """

    def __init__(
        self,
        n_ctx: int,
        counter: TokenCounter,
        *,
        reserve_output: int | None = None,
        compact_threshold: float = 0.8,
        summarizer: Summarizer | None = None,
        summarizer_timeout: float | None = 300.0,
    ) -> None:
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < 256:
            raise ValueError(f"Okno kontekstu modelu jest za małe ({n_ctx!r}); potrzeba co najmniej 256 tokenów.")
        if not callable(counter):
            raise TypeError("Licznik tokenów (counter) musi być funkcją.")
        try:
            threshold = float(compact_threshold)
        except (TypeError, ValueError):
            raise ValueError("Próg kompaktowania kontekstu musi być liczbą z zakresu 0.1–1.0.") from None
        if not 0.1 <= threshold <= 1.0:
            raise ValueError(f"Próg kompaktowania kontekstu musi być z zakresu 0.1–1.0 (podano {compact_threshold}).")
        if reserve_output is None:
            reserve = min(max(int(n_ctx * 0.2), 512), 8192, n_ctx // 2)
        else:
            reserve = int(reserve_output)
            if reserve < 0 or reserve > n_ctx - 128:
                raise ValueError(
                    f"Rezerwa na odpowiedź ({reserve_output}) musi być nieujemna i mniejsza od okna kontekstu ({n_ctx})."
                )
        self.n_ctx = n_ctx
        self.reserve_output = reserve
        self.usable = n_ctx - reserve
        self.compact_threshold = threshold
        self.soft_limit = max(1, int(threshold * self.usable))
        self.keep_tool_results = 2 if n_ctx <= 8192 else 5
        self.summary_tokens = min(max(int(n_ctx * 0.06), 250), 600)
        self.summarizer: Summarizer | None = summarizer
        self.summarizer_timeout = summarizer_timeout
        #: "model" | "fallback" | None — where the last summary came from (diagnostics)
        self.last_summary_source: str | None = None
        #: error of the last failed summarizer call, if any (diagnostics, English)
        self.last_summary_error: str | None = None
        self._counter = counter
        self._last_tools: list[ToolSchema] | None = None
        self._fixed_cache: dict[tuple[str, str], tuple[int, int]] = {}
        self._shrink_chars = min(max(n_ctx // 8, 400), 2000)
        self._task_chars = min(max(int(self.usable * 0.12 * CHARS_PER_TOKEN), 600), 6000)
        self._summary_max_tokens = int(self.summary_tokens * 1.5)
        self._is_nudge = _nudge_checker()

    # ------------------------------------------------------------------ info

    def __repr__(self) -> str:
        return (
            f"ContextManager(n_ctx={self.n_ctx}, reserve={self.reserve_output}, "
            f"usable={self.usable}, soft_limit={self.soft_limit})"
        )

    def limits(self) -> ToolLimits:
        """Tool output limits for this window (see ``limits_for``)."""
        return limits_for(self.n_ctx, self.usable)

    def truncate_tool_output(self, text: str) -> str:
        """Clip a tool result to ``limits().max_output_chars`` with a paging hint."""
        return truncate(text, self.limits().max_output_chars, hint="use offset/limit or grep to see more")

    def output_budget(self, prompt_tokens: int) -> int:
        """``max_tokens`` that still fits after a prompt of ``prompt_tokens`` tokens."""
        return max(0, self.n_ctx - int(prompt_tokens))

    def needs_compaction(self, prompt_tokens: int) -> bool:
        return int(prompt_tokens) > self.soft_limit

    # --------------------------------------------------------------- counting

    async def _count(self, messages: list[Message], tools: list[ToolSchema] | None) -> int:
        try:
            result = self._counter(messages, tools)
            if inspect.isawaitable(result):
                result = await result
            return int(result)
        except Exception as exc:  # engine gone, template error, ... -> estimate
            log.warning("token counter failed (%s); using an estimate", exc)
            return estimate_messages_tokens(messages, tools)

    async def _measure(self, messages: list[Message], tools: list[ToolSchema] | None, pessimistic: bool) -> int:
        exact = await self._count(messages, tools)
        if pessimistic:
            return max(exact, estimate_messages_tokens(messages, tools))
        return exact

    async def usage(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system_tokens: int | None = None,
    ) -> ContextUpdate:
        """Break down the next prompt: total, system prompt, tool schemas, history."""
        msgs = list(messages)
        tool_list = list(tools) if tools else []
        used = await self._count(msgs, tool_list)
        sys_msgs = msgs[:1] if msgs and _role(msgs[0]) == "system" else []
        sys_tok, tools_tok = await self._fixed_costs(sys_msgs, tool_list)
        if system_tokens is not None:
            sys_tok = int(system_tokens)
        history = max(0, used - sys_tok - tools_tok)
        return ContextUpdate(
            used=used,
            limit=self.n_ctx,
            system=sys_tok,
            tools=tools_tok,
            history=history,
            reserve=self.reserve_output,
        )

    async def _fixed_costs(self, sys_msgs: list[Message], tools: list[ToolSchema]) -> tuple[int, int]:
        """Tokens of the system message and of the tool schemas (cached by content)."""
        try:
            tools_key = json.dumps(tools, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            tools_key = str(tools)
        key = (_content_text(sys_msgs[0].get("content")) if sys_msgs else "", tools_key)
        cached = self._fixed_cache.get(key)
        if cached is not None:
            return cached
        # Some templates refuse to render without a user turn, so measure differences
        # against a tiny probe conversation.
        probe = [{"role": "user", "content": "hi"}]
        base = await self._count(probe, [])
        with_sys = await self._count(sys_msgs + probe, []) if sys_msgs else base
        with_tools = await self._count(sys_msgs + probe, tools) if tools else with_sys
        result = (max(0, with_sys - base), max(0, with_tools - with_sys))
        if len(self._fixed_cache) > 32:
            self._fixed_cache.clear()
        self._fixed_cache[key] = result
        return result

    # ------------------------------------------------------------ public API

    async def fit(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema] | None = None
    ) -> tuple[list[Message], list[Compaction]]:
        """Return ``(messages, compactions)`` fitting under the soft limit.

        Unchanged (same message objects) while under the soft limit; otherwise
        the staged pipeline described in the module docstring runs.
        """
        msgs = list(messages)
        tool_list = list(tools) if tools is not None else None
        self._last_tools = tool_list
        total = await self._measure(msgs, tool_list, False)
        if total <= self.soft_limit:
            return msgs, []
        return await self._run(
            msgs,
            tool_list,
            total,
            _Pass(
                target=self.soft_limit,
                hard_limit=self.usable,
                keep_tool_results=self.keep_tool_results,
                recent_share=0.5,
            ),
        )

    async def on_overflow(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema] | None = None
    ) -> tuple[list[Message], list[Compaction]]:
        """Aggressive pass after the engine rejected a prompt as too long.

        The counter evidently under-counted, so sizes are measured as
        max(exact, estimate), only the newest tool result is kept, a summary is
        forced and the target is 60 % of the usable window.
        """
        tool_list = list(tools) if tools is not None else self._last_tools
        msgs = list(messages)
        total = await self._measure(msgs, tool_list, True)
        return await self._run(
            msgs,
            tool_list,
            total,
            _Pass(
                target=int(self.usable * 0.6),
                hard_limit=int(self.usable * 0.85),
                keep_tool_results=1,
                recent_share=0.2,
                force=True,
                pessimistic=True,
            ),
        )

    async def compact(
        self, messages: Sequence[Message], tools: Sequence[ToolSchema] | None = None
    ) -> tuple[list[Message], list[Compaction]]:
        """Manual compaction (``/compact``): summarize old work even under the limit."""
        tool_list = list(tools) if tools is not None else self._last_tools
        msgs = list(messages)
        total = await self._measure(msgs, tool_list, False)
        return await self._run(
            msgs,
            tool_list,
            total,
            _Pass(
                target=self.soft_limit,
                hard_limit=self.usable,
                keep_tool_results=self.keep_tool_results,
                recent_share=0.0,
                force=True,
            ),
        )

    # -------------------------------------------------------------- pipeline

    async def _run(
        self, msgs: list[Message], tools: list[ToolSchema] | None, total: int, p: _Pass
    ) -> tuple[list[Message], list[Compaction]]:
        events: list[Compaction] = []

        async def attempt(candidate: list[Message] | None, strategy: str) -> None:
            nonlocal msgs, total
            if candidate is None:
                return
            after = await self._measure(candidate, tools, p.pessimistic)
            if after < total:
                events.append(Compaction(before=total, after=after, strategy=strategy))
                msgs, total = candidate, after

        await attempt(self._mask_tool_results(msgs, p.keep_tool_results), STRATEGY_MASK)
        if total <= p.target and not p.force:
            return msgs, events
        await attempt(self._shrink_old(msgs, p.keep_tool_results), STRATEGY_SHRINK)
        if total <= p.target and not p.force:
            return msgs, events
        await attempt(await self._summarize(msgs, tools, total, p), STRATEGY_SUMMARIZE)
        if total <= p.target:
            return msgs, events
        dropped = await self._drop_oldest(msgs, tools, total, p)
        if dropped is not None and dropped[1] < total:
            events.append(Compaction(before=total, after=dropped[1], strategy=STRATEGY_DROP))
            msgs, total = dropped
        if total <= p.hard_limit:
            return msgs, events
        cut = await self._hard_truncate(msgs, tools, total, p)
        if cut is not None and cut[1] < total:
            events.append(Compaction(before=total, after=cut[1], strategy=STRATEGY_TRUNCATE))
            msgs, total = cut
        if total > p.hard_limit:
            log.warning("context still over the limit after compaction: %d > %d", total, p.hard_limit)
        return msgs, events

    def _pinned_users(self, msgs: Sequence[Message]) -> tuple[int | None, set[int]]:
        """(index of the first user message, indices of user messages that must be kept)."""
        body = _body_start(msgs)
        users = [i for i in range(body, len(msgs)) if _role(msgs[i]) == "user"]
        if not users:
            return None, set()
        pinned = {users[-1]}
        for i in reversed(users):
            if not self._is_nudge(_content_text(msgs[i].get("content"))):
                pinned.add(i)
                break
        return users[0], pinned

    # stage 1 ---------------------------------------------------------------

    def _mask_tool_results(self, msgs: list[Message], keep: int) -> list[Message] | None:
        tool_idx = [i for i, m in enumerate(msgs) if _role(m) == "tool"]
        old = tool_idx[: len(tool_idx) - keep] if keep > 0 else tool_idx
        if not old:
            return None
        names = _call_names(msgs)
        out = list(msgs)
        changed = False
        for i in old:
            m = msgs[i]
            content = m.get("content")
            if not isinstance(content, str) or _is_masked(content):
                continue
            name = str(m.get("name") or names.get(str(m.get("tool_call_id"))) or "tool")
            stub = MASK_TEMPLATE.format(name=name)
            first = _first_line(content)
            if first:
                stub += "\n" + first
            if len(stub) >= len(content):
                continue
            new = dict(m)
            new["content"] = stub
            out[i] = new
            changed = True
        return out if changed else None

    # stage 2 ---------------------------------------------------------------

    def _shrink_old(self, msgs: list[Message], keep: int) -> list[Message] | None:
        body = _body_start(msgs)
        units = _units(msgs, body)
        keep_units = max(1, keep)
        if len(units) <= keep_units:
            return None
        boundary = units[-keep_units][0]
        first_user, pinned = self._pinned_users(msgs)
        limit = self._shrink_chars
        out = list(msgs)
        changed = False
        for i in range(body, boundary):
            m = msgs[i]
            role = _role(m)
            new: Message | None = None
            content = m.get("content")
            if role == "assistant":
                if isinstance(content, str) and len(content) > limit * 1.25:
                    new = dict(m)
                    new["content"] = truncate(content, limit)
                if m.get("tool_calls"):
                    calls, ch = _shrink_calls(m["tool_calls"], limit, slack=1.25)
                    if ch:
                        new = new if new is not None else dict(m)
                        new["tool_calls"] = calls
            elif role == "user" and i == first_user and i not in pinned:
                # the original task is no longer the latest request: keep its beginning
                task, summary, rest = _split_head(content)
                if isinstance(task, str) and len(task) > self._task_chars * 1.25:
                    new = dict(m)
                    new["content"] = _build_head(self._trim_task(task), summary, rest)
            elif role == "user" and i not in pinned:
                if isinstance(content, str) and len(content) > 2 * limit * 1.25:
                    new = dict(m)
                    new["content"] = truncate(content, 2 * limit, head_ratio=0.7)
            if new is not None:
                out[i] = new
                changed = True
        return out if changed else None

    # stage 3 ---------------------------------------------------------------

    async def _summarize(
        self, msgs: list[Message], tools: list[ToolSchema] | None, total: int, p: _Pass
    ) -> list[Message] | None:
        body = _body_start(msgs)
        t, pinned = self._pinned_users(msgs)
        if t is None:
            return None
        units = _units(msgs, t + 1)
        if not units:
            return None
        head_pinned = t in pinned
        pulled = {i for i in pinned if i != t}
        task, old_summary, rest = _split_head(msgs[t].get("content"))
        new_task = task if head_pinned else self._trim_task(task)
        new_rest = rest if head_pinned else None

        # planning happens in estimate space, calibrated against the exact count
        ratio = total / max(1, estimate_messages_tokens(msgs, tools))
        fixed = estimate_messages_tokens(msgs[:body], tools)
        head_est = (
            estimate_tokens(_content_text(new_task))
            + estimate_tokens(_content_text(new_rest))
            + self._summary_max_tokens
            + 16
            + sum(estimate_message_tokens(msgs[i]) for i in pulled)
        )
        available = p.target / ratio - fixed - head_est
        budget = max(0.0, min(self.usable * p.recent_share / ratio, available * 0.9))

        recent_start = len(msgs)
        acc = 0
        for k, (s, e) in enumerate(reversed(units)):
            cost = estimate_messages_tokens(msgs[s:e]) - 3
            if k == 0 or s in pulled or acc + cost <= budget:
                recent_start = s
                if s not in pulled:
                    acc += cost
            else:
                break

        region: list[Message] = []
        if rest is not None and not head_pinned:
            region.append({"role": "user", "content": rest})
        region.extend(msgs[i] for i in range(body, t))
        region.extend(msgs[i] for i in range(t + 1, recent_start) if i not in pulled)
        if not region:
            return None

        summary = await self._make_summary(region, old_summary, msgs)
        head = dict(msgs[t])
        head["content"] = _build_head(new_task, summary, new_rest)
        new = list(msgs[:body]) + [head]
        new.extend(msgs[i] for i in sorted(pulled) if i < recent_start)
        new.extend(msgs[recent_start:])
        return _repair(new, body)

    def _trim_task(self, task: Any) -> Any:
        if task is None:
            return None
        if isinstance(task, str):
            return truncate(task, self._task_chars, head_ratio=0.7)
        text = _content_text(task)
        if _image_parts(task):
            text = (text + "\n[image omitted]").strip()
        return truncate(text, self._task_chars, head_ratio=0.7)

    async def _make_summary(self, region: list[Message], previous: str | None, all_msgs: list[Message]) -> str:
        names = _call_names(all_msgs)
        text = ""
        if self.summarizer is not None:
            from localcode.agent import prompts

            scaffold = prompts.compaction_prompt("", previous_summary=previous, target_tokens=self.summary_tokens)
            budget = max(128, self.usable - estimate_messages_tokens(scaffold) - 32)
            transcript = _fit_text_to_tokens(self._render_transcript(region, names), budget, head_ratio=0.4)
            prompt_msgs = prompts.compaction_prompt(
                transcript, previous_summary=previous, target_tokens=self.summary_tokens
            )
            try:
                raw: Any = self.summarizer(prompt_msgs)
                if inspect.isawaitable(raw):
                    raw = await (asyncio.wait_for(raw, self.summarizer_timeout) if self.summarizer_timeout else raw)
                text = _clean_summary(raw)
                self.last_summary_error = None if text else "summarizer returned no usable text"
            except Exception as exc:  # timeouts, engine errors, ... -> deterministic fallback
                self.last_summary_error = f"{type(exc).__name__}: {exc}"
                log.warning("summarizer failed (%s); using the extractive summary", self.last_summary_error)
                text = ""
        if text:
            self.last_summary_source = "model"
        else:
            self.last_summary_source = "fallback"
            text = self._extractive_summary(region, previous, all_msgs, names, self._summary_max_tokens)
        return _fit_text_to_tokens(text, self._summary_max_tokens, head_ratio=0.8)

    @staticmethod
    def _render_transcript(region: list[Message], names: dict[str, str]) -> str:
        lines: list[str] = []
        for m in region:
            role = _role(m)
            text = _content_text(m.get("content")).strip()
            if role == "user":
                if text:
                    lines.append("USER: " + truncate(text, 1500, head_ratio=0.7))
            elif role == "assistant":
                if text:
                    lines.append("ASSISTANT: " + truncate(text, 1000))
                for call in m.get("tool_calls") or []:
                    fn = call.get("function") or {} if isinstance(call, dict) else {}
                    lines.append(f"CALL {fn.get('name') or '?'} {truncate(_args_text(fn.get('arguments')), 300)}")
            elif role == "tool":
                name = m.get("name") or names.get(str(m.get("tool_call_id"))) or "tool"
                lines.append(f"RESULT {name}: " + truncate(text, 600))
            elif text:
                lines.append("NOTE: " + truncate(text, 500))
        return "\n".join(lines)

    def _extractive_summary(
        self,
        region: list[Message],
        previous: str | None,
        all_msgs: list[Message],
        names: dict[str, str],
        max_tokens: int,
    ) -> str:
        """Deterministic summary built from tool calls and messages (no model).

        Lines are added in priority order (user messages, changed files, errors,
        commands, todo, earlier summary, files read, searches, notes) while they
        fit in ``max_tokens``; a line that does not fit is shortened.
        """
        results = {str(m.get("tool_call_id")): _content_text(m.get("content")) for m in region if _role(m) == "tool"}
        reads: list[str] = []
        changed: dict[str, str] = {}
        commands: list[str] = []
        searches: list[str] = []
        others: list[str] = []
        users: list[str] = []
        notes: list[str] = []
        errors: list[str] = []

        def add_unique(items: list[str], value: str) -> None:
            if value and value not in items:
                items.append(value)

        for m in region:
            role = _role(m)
            text = _content_text(m.get("content")).strip()
            if role == "user" and text:
                users.append(truncate(" ".join(text.split()), 200))
            elif role == "assistant":
                if text:
                    notes.append(truncate(" ".join(text.split()), 160))
                for call in m.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    name = str(fn.get("name") or "tool")
                    args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
                    result = results.get(str(call.get("id")), "")
                    failed = bool(result) and _looks_failed(result) and not _is_masked(result)
                    path = str(args.get("path") or args.get("file_path") or args.get("file") or "")
                    if name in _READ_TOOLS:
                        add_unique(reads, path or "?")
                    elif name in _EDIT_TOOLS:
                        label = name + (", failed" if failed else "")
                        changed[path or "?"] = label
                    elif name in _BASH_TOOLS:
                        cmd = " ".join(str(args.get("command") or args.get("cmd") or "").split())
                        code = _EXIT_RE.search(result or "")
                        status = f" → exit {code.group(1)}" if code else (" → failed" if failed else "")
                        commands.append(f"`{truncate(cmd, 120)}`{status}")
                    elif name in _SEARCH_TOOLS:
                        what = args.get("pattern") or args.get("query") or args.get("path") or ""
                        where = args.get("path") if (args.get("pattern") or args.get("query")) else None
                        label = f"{name} {json.dumps(str(what), ensure_ascii=False)}"
                        if where:
                            label += f" in {where}"
                        add_unique(searches, truncate(label, 120))
                    elif name not in _TODO_TOOLS:
                        add_unique(others, name)
                    if failed:
                        errors.append(f"{name}: {_first_line(result, 160)}")

        todos = self._latest_todos(all_msgs)

        def listed(items: list[str], limit: int, *, count_earlier: bool = True) -> str:
            # collapse repeats ("`pytest -q` → exit 1 (×3)"), keep the newest `limit` entries
            counts: dict[str, int] = {}
            for item in items:
                counts[item] = counts.get(item, 0) + 1
            order = list(dict.fromkeys(reversed(items)))[::-1]
            shown = order[-limit:]
            text = "; ".join(f"{x} (×{counts[x]})" if counts[x] > 1 else x for x in shown)
            if count_earlier and len(order) > limit:
                text += f" (+{len(order) - limit} earlier)"
            return text

        sections: list[tuple[str, str]] = [
            ("User messages", listed([f'"{u}"' for u in users], 3) if users else ""),
            # failed changes go last so they survive the "newest entries" cut
            ("Files changed", listed(
                [f"{k} ({v})" for k, v in changed.items() if not v.endswith("failed")]
                + [f"{k} ({v})" for k, v in changed.items() if v.endswith("failed")], 10) if changed else ""),
            ("Errors", listed(errors, 3) if errors else ""),
            ("Commands run", listed(commands, 6) if commands else ""),
            ("Todo", todos),
            ("Earlier summary", " ".join(previous.split()) if previous else ""),
            ("Files read", listed(reads, 10) if reads else ""),
            ("Searches", listed(searches, 5) if searches else ""),
            ("Other tools used", ", ".join(others[:8])),
            ("Last assistant notes", listed(notes, 2, count_earlier=False) if notes else ""),
        ]
        intro = "(Automatic summary: the details of these steps were removed from the context.)"
        lines = [intro]
        remaining = max_tokens - estimate_tokens(intro) - 1
        for label, body in sections:
            if not body:
                continue
            line = f"{label}: {body}"
            cost = estimate_tokens(line) + 1
            if cost > remaining:
                if remaining < 24:
                    continue
                line = _fit_text_to_tokens(line, remaining - 1, head_ratio=1.0)
                cost = estimate_tokens(line) + 1
            lines.append(line)
            remaining -= cost
        if len(lines) == 1:
            lines.append("Nothing important happened in the removed part.")
        return "\n".join(lines)

    @staticmethod
    def _latest_todos(msgs: list[Message]) -> str:
        marks = {"done": "[x]", "completed": "[x]", "in_progress": "[~]", "in-progress": "[~]", "active": "[~]"}
        for m in reversed(msgs):
            for call in reversed(m.get("tool_calls") or []):
                fn = call.get("function") or {} if isinstance(call, dict) else {}
                if fn.get("name") not in _TODO_TOOLS:
                    continue
                args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
                items = args.get("todos") or args.get("items") or []
                if not isinstance(items, list):
                    continue
                rendered = []
                for item in items[:15]:
                    if isinstance(item, dict):
                        text = str(item.get("content") or item.get("text") or item.get("task") or "").strip()
                        mark = marks.get(str(item.get("status") or "").lower(), "[ ]")
                    else:
                        text, mark = str(item).strip(), "[ ]"
                    if text:
                        rendered.append(f"{mark} {truncate(text, 100)}")
                return "; ".join(rendered) if rendered else "(empty)"
        return ""

    # stage 4 ---------------------------------------------------------------

    async def _drop_oldest(
        self, msgs: list[Message], tools: list[ToolSchema] | None, total: int, p: _Pass
    ) -> tuple[list[Message], int] | None:
        body = _body_start(msgs)
        units = _units(msgs, body)
        if len(units) < 2:
            return None
        t, pinned = self._pinned_users(msgs)
        final = units[-1]
        droppable = [u for u in units if u != final and u[0] != t and u[0] not in pinned]
        if not droppable:
            return None
        costs = [estimate_messages_tokens(msgs[s:e]) - 3 for s, e in droppable]
        head_obj = msgs[t] if t is not None else None

        def build(n: int) -> list[Message]:
            gone: set[int] = set()
            for s, e in droppable[:n]:
                gone.update(range(s, e))
            kept = [m for i, m in enumerate(msgs) if i not in gone]
            if head_obj is not None:
                pos = next(k for k, m in enumerate(kept) if m is head_obj)
                task, summary, rest = _split_head(head_obj.get("content"))
                if summary is None or DROPPED_NOTE not in summary:
                    summary = DROPPED_NOTE if not summary else f"{summary}\n{DROPPED_NOTE}"
                    head = dict(head_obj)
                    head["content"] = _build_head(task, summary, rest)
                    kept[pos] = head
            return _repair(kept, body)

        n = 0
        cur_msgs, cur_total = msgs, total
        best: tuple[list[Message], int] | None = None
        while n < len(droppable):
            # drop enough estimated tokens to cover the excess (calibrated), at least one unit
            ratio = cur_total / max(1, estimate_messages_tokens(cur_msgs, tools))
            excess = cur_total - p.target
            add = 0.0
            while n < len(droppable) and (add == 0 or add * ratio < excess):
                add += costs[n]
                n += 1
            cur_msgs = build(n)
            cur_total = await self._measure(cur_msgs, tools, p.pessimistic)
            best = (cur_msgs, cur_total)
            if cur_total <= p.target:
                break
        return best

    # stage 5 ---------------------------------------------------------------

    async def _hard_truncate(
        self, msgs: list[Message], tools: list[ToolSchema] | None, total: int, p: _Pass
    ) -> tuple[list[Message], int] | None:
        body = _body_start(msgs)
        units = _units(msgs, body)
        t, pinned = self._pinned_users(msgs)
        cands: list[tuple[int, str, int, float]] = []  # (index, kind, min chars, head ratio)
        if units:
            s, e = units[-1]
            tool_idx = sorted(
                (i for i in range(s, e) if _role(msgs[i]) == "tool"),
                key=lambda i: -len(_content_text(msgs[i].get("content"))),
            )
            cands += [(i, "content", 400, 0.5) for i in tool_idx]
            if _role(msgs[s]) == "assistant":
                cands += [(s, "content", 300, 0.5), (s, "args", 300, 0.6)]
        if t is not None:
            cands += [(t, "summary", 200, 0.7), (t, "task", 300, 0.7), (t, "rest", 400, 0.7)]
        cands += [(i, "content", 400, 0.7) for i in sorted(pinned, reverse=True) if i != t]
        if body:
            cands.append((0, "content", 1200, 0.7))

        changed = False
        for idx, kind, min_chars, head_ratio in cands:
            for _ in range(8):
                if total <= p.hard_limit:
                    break
                cur = self._field_len(msgs[idx], kind)
                if cur <= min_chars:
                    break
                ratio = total / max(1, estimate_messages_tokens(msgs, tools))
                field_text = self._field_text(msgs[idx], kind)
                chars_per_est = len(field_text) / max(1, estimate_tokens(field_text))
                cut = math.ceil((total - p.hard_limit) / ratio * chars_per_est * 1.15) + 32
                new_len = max(min_chars, cur - cut)
                candidate = self._cut_field(msgs, idx, kind, new_len, head_ratio)
                if candidate is None:
                    break
                msgs, total = candidate, await self._measure(candidate, tools, p.pessimistic)
                changed = True
            if total <= p.hard_limit:
                break
        return (msgs, total) if changed else None

    @staticmethod
    def _field_text(m: Message, kind: str) -> str:
        if kind == "content":
            return _content_text(m.get("content"))
        if kind == "args":
            return " ".join(_args_text((c.get("function") or {}).get("arguments")) for c in m.get("tool_calls") or [] if isinstance(c, dict))
        task, summary, rest = _split_head(m.get("content"))
        part = {"summary": summary, "task": task, "rest": rest}[kind]
        return _content_text(part) if part is not None else ""

    @classmethod
    def _field_len(cls, m: Message, kind: str) -> int:
        if kind == "args":
            longest = 0
            stack: list[Any] = [(c.get("function") or {}).get("arguments") for c in m.get("tool_calls") or [] if isinstance(c, dict)]
            while stack:
                v = stack.pop()
                if isinstance(v, str):
                    longest = max(longest, len(v))
                elif isinstance(v, dict):
                    stack.extend(v.values())
                elif isinstance(v, list):
                    stack.extend(v)
            return longest
        return len(cls._field_text(m, kind))

    @staticmethod
    def _cut_field(msgs: list[Message], idx: int, kind: str, new_len: int, head_ratio: float) -> list[Message] | None:
        m = msgs[idx]
        new = dict(m)
        if kind == "content":
            text = _content_text(m.get("content"))
            if len(text) <= new_len:
                return None
            new["content"] = truncate(text, new_len, head_ratio=head_ratio, hint=_CUT_HINT)
        elif kind == "args":
            calls, ch = _shrink_calls(list(m.get("tool_calls") or []), new_len)
            if not ch:
                return None
            new["tool_calls"] = calls
        else:
            task, summary, rest = _split_head(m.get("content"))
            parts = {"task": task, "summary": summary, "rest": rest}
            part = parts[kind]
            text = _content_text(part) if part is not None else ""
            if len(text) <= new_len:
                return None
            if kind == "summary":
                parts["summary"] = truncate(text, new_len, head_ratio=head_ratio)
            else:
                parts[kind] = truncate(text, new_len, head_ratio=head_ratio, hint=_CUT_HINT)
            new["content"] = _build_head(parts["task"], parts["summary"], parts["rest"])
        out = list(msgs)
        out[idx] = new
        return out
