"""Auto mode: a critical reviewer model decides the calls that would ask the user.

In ``auto`` mode the policy decides exactly as in ``default`` (reads, read-only
shell and remembered commands run freely). Every call that would show a
permission question instead goes to a reviewer — by default the model the
agent runs on, or the one picked in Settings (``agent.auto_reviewer_model``) —
which answers ``TAK``/``NIE``, what the action does and why. A ``TAK`` runs
the call; a ``NIE`` or no clear answer goes to the user, whose permission card
shows the review (``PermissionRequest.review``) and who may overrule it. Calls
the policy says need a human (``PermissionPolicy.needs_human``: data to a
provider, catastrophic) are reviewed only as advice for that card — a ``TAK``
never approves them; a call outside the project is the reviewer's to decide.

The review prompt is built to hit the engine's prefix cache: the agent's own
messages up to — not including — the assistant message holding the pending
tool calls (an assistant ``tool_calls`` message must be followed by tool
results, never by a user message), then one user message describing the call
plus the reviewer's instructions. With the agent's own model and the same tool
list that prefix is byte-identical to the step that produced the call.

This module holds the pure parts (prompt text, prefix split, verdict parser,
window fit); the loop (``Agent._auto_review``) acquires the model and runs it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

__all__ = [
    "DENIED_RESULT",
    "USER_DENIED_RESULT",
    "approved",
    "refusal_text",
    "REVIEW_TIMEOUT",
    "REVIEW_TOKENS",
    "REVIEW_THINKING_TOKENS",
    "fit_window",
    "parse_verdict",
    "precedent_text",
    "review_message",
    "split_at_call",
]

#: Output budget of one review: a verdict line and a reason line.
REVIEW_TOKENS = 300
#: …for a model that cannot stop reasoning (gpt-oss, R1-style templates) or a
#: remote one, whose hidden reasoning may count against ``max_tokens``.
REVIEW_THINKING_TOKENS = 1200
#: Seconds one review may take once the model is in hand.
REVIEW_TIMEOUT = 120.0

#: What the agent is told when the reviewer said no and nobody could be asked.
DENIED_RESULT = ("Auto-mode reviewer rejected this call: {reason}. "
                 "Choose a safer way or ask the user.")
#: …and when the user, shown the reviewer's NIE, declined it too.
USER_DENIED_RESULT = "Auto-mode reviewer rejected this call: {reason}. The user also declined it."
#: The context-free local guard's NIE, as the agent reads it (``agent/guard.py``).
GUARD_DENIED = "The local safety check rejected this call: {reason}."


def approved(review: dict[str, Any] | None) -> bool:
    """Both gates said TAK: the reviewer, and the guard when it checked the call."""
    if review is None or review.get("verdict") != "allow":
        return False
    guard = review.get("guard")
    return guard is None or (isinstance(guard, dict) and guard.get("verdict") == "allow")


def refusal_text(review: dict[str, Any] | None, *, user_declined: bool) -> str | None:
    """What the agent reads when a gate said NIE (``None`` when none did).

    The reviewer's NIE alone reads exactly as :data:`DENIED_RESULT` /
    :data:`USER_DENIED_RESULT`; the guard's is added as its own sentence.
    """
    if review is None:
        return None
    parts: list[str] = []
    if review.get("verdict") == "deny":
        reason = str(review.get("reason") or "").strip().rstrip(".")
        parts.append(f"Auto-mode reviewer rejected this call: {reason}.")
    guard = review.get("guard")
    if isinstance(guard, dict) and guard.get("verdict") == "deny":
        reason = str(guard.get("reason") or "").strip().rstrip(".")
        parts.append(GUARD_DENIED.format(reason=reason))
    if not parts:
        return None
    parts.append("The user also declined it." if user_declined
                 else "Choose a safer way or ask the user.")
    return " ".join(parts)

#: Polish reasons for falling back to the user (shown in the UI).
REASON_UNCLEAR = "Recenzent nie dał jednoznacznej odpowiedzi — decyzja należy do Ciebie."
REASON_TOOL_CALL = ("Recenzent zamiast ocenić akcję próbował wywołać narzędzie — "
                    "decyzja należy do Ciebie.")
REASON_TIMEOUT = "Recenzent nie odpowiedział w {seconds} s — decyzja należy do Ciebie."
REASON_ERROR = "Recenzent nie odpowiedział ({detail}) — decyzja należy do Ciebie."
REASON_NO_MODEL = "Nie udało się uruchomić modelu recenzenta — decyzja należy do Ciebie."
REASON_MISSING = "Recenzent nie podał powodu."

_RULES = """\
You are now a strict, skeptical security reviewer — not the assistant. Do not continue the task \
and do not call any tool.
Approve only if ALL of these hold:
- the action clearly serves what the user asked for in this conversation;
- it stays inside the project directory — or the user's request clearly needs that place \
outside it; then be extra strict: never approve reading private data (~/.ssh, keychains, browser \
profiles, other projects' .env files or tokens) or changing system files or other projects unless \
the user asked for exactly that;
- it is not destructive or irreversible beyond what the user asked for (deleting or overwriting \
files or data, git reset --hard, force push, dropping a database…);
- it does not send code, data, secrets, keys or tokens anywhere, and does not read secrets it \
has no need for;
- it does not install packages or download and run code unless the user asked for that.
If anything is unclear or only "probably fine", reject it.

Answer in exactly three lines and nothing else, in this order:
Co robi: <one plain sentence in Polish saying what the action does, for someone who cannot read \
code>
Powód: <one or two sentences in Polish weighing it against the rules above>
TAK or NIE — the last line is only that word, and it must agree with your Powód."""


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content
                         if isinstance(p, dict) and isinstance(p.get("text"), str))
    return str(content or "")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


def split_at_call(messages: Sequence[dict[str, Any]], call_id: str
                  ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    """``(before, assistant, after)`` around the assistant message holding ``call_id``.

    ``before`` is what the review prompt is built on; ``after`` holds the
    results of the calls of the same step that already ran. Without such a
    message (it was compacted away) everything is ``before``.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        ids = [str(c.get("id") or "") for c in message.get("tool_calls") or [] if isinstance(c, dict)]
        if call_id in ids:
            return list(messages[:index]), message, list(messages[index + 1:])
    return list(messages), None, []


def sibling_lines(assistant: dict[str, Any] | None, after: Sequence[dict[str, Any]],
                  call_id: str) -> list[str]:
    """One short line per other call of the same step, with its result when it ran."""
    if assistant is None:
        return []
    results = {str(m.get("tool_call_id") or ""): _content_text(m.get("content"))
               for m in after if m.get("role") == "tool"}
    lines: list[str] = []
    for call in assistant.get("tool_calls") or []:
        if not isinstance(call, dict) or str(call.get("id") or "") == call_id:
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        args = json.dumps(fn.get("arguments") or {}, ensure_ascii=False, default=str)
        line = f"{fn.get('name') or '?'} {_clip(args, 160)}"
        result = results.get(str(call.get("id") or ""))
        line += f" → ran, result: {_clip(result, 160)}" if result is not None else " → not run yet"
        lines.append(line)
    return lines


def precedent_text(lines: Sequence[str]) -> str:
    """The user's matching past answers (``agent/precedents.py``) as a short note, or ``""``.

    ``lines`` are ``approved: git push (3×)`` / ``declined: rm (1×)`` — keys our
    own parser computed from calls the user answered, never the model's text.
    """
    lines = [str(line) for line in lines if str(line).strip()]
    if not lines:
        return ""
    parts = ["The user's own earlier answers in this project on calls like this one "
             "(recorded by LocalCode from the permission cards, not written by the assistant):",
             *(f"- {line}" for line in lines)]
    if any(line.startswith("approved:") for line in lines):
        parts.append("Treat a call clearly like an approved one as acceptable, unless this one is "
                     "riskier (another target, extra destructive flags, paths outside the project, "
                     "network).")
    if any(line.startswith("declined:") for line in lines):
        parts.append("Be stricter with a call like a declined one.")
    return "\n".join(parts)


def review_message(*, tool: str, title: str, preview: str, arguments: Any,
                   assistant_text: str = "", others: Sequence[str] = (), n_ctx: int = 8192,
                   precedents: Sequence[str] = ()) -> str:
    """The user message that asks for a verdict on one proposed call.

    ``precedents``: the user's past answers matching this call, put just
    before the rules (which stay last).
    """
    cap = 3000 if n_ctx <= 16384 else 10000
    parts = [
        "[Auto mode — safety review]",
        "The assistant above has just proposed the tool call below. In auto mode it runs only "
        "if you approve it; nobody else checks it.",
        "",
        f"Tool: {tool}",
        f"Action: {title or tool}",
    ]
    if assistant_text.strip():
        parts.append(f"What the assistant wrote with it: {_clip(assistant_text, 1200)}")
    args_json = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False, default=str)
    if preview.strip():
        parts += ["Exactly what would run or change:", "```", _clip(preview, cap), "```"]
    if not preview.strip() or len(args_json) <= 600:
        parts.append(f"Arguments (JSON): {_clip(args_json, cap)}")
    if others:
        parts.append("Other tool calls in the same step:\n" + "\n".join(f"- {o}" for o in others))
    noted = precedent_text(precedents)
    if noted:
        parts += ["", noted]
    parts += ["", _RULES]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #

#: Characters per token for the estimate (deliberately pessimistic).
_CHARS_PER_TOKEN = 3.0
_IMAGE_TOKENS = 1024


def _estimate(messages: Sequence[dict[str, Any]]) -> int:
    chars = 0
    pictures = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            pictures += sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        chars += len(_content_text(content)) + 16
        if message.get("tool_calls"):
            chars += len(json.dumps(message["tool_calls"], ensure_ascii=False, default=str))
    return int(chars / _CHARS_PER_TOKEN) + pictures * _IMAGE_TOKENS


def fit_window(messages: Sequence[dict[str, Any]], n_ctx: int, reserve: int) -> list[dict[str, Any]]:
    """Drop the oldest turns (never the system prompt) until the estimate fits.

    Only for a reviewer that is not the agent's own model — that one's prefix
    already fits, and trimming it would throw the prefix cache away. A cut
    always starts at a user message, so no tool result loses its call.
    """
    msgs = list(messages)
    budget = max(256, int(n_ctx) - int(reserve))
    if _estimate(msgs) <= budget:
        return msgs
    system = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
    body = msgs[len(system):]
    starts = [i for i, m in enumerate(body) if m.get("role") == "user" and i > 0]
    for start in starts:
        candidate = system + body[start:]
        if _estimate(candidate) <= budget:
            return candidate
    return system + body[starts[-1]:] if starts else msgs


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

_YES = ("tak", "yes", "allow", "allowed", "approve", "approved", "accept", "accepted",
        "zgoda", "zezwalam", "zatwierdzam", "akceptuję", "akceptuje")
_NO = ("nie", "no", "deny", "denied", "reject", "rejected", "block", "blocked",
       "refuse", "refused", "odrzucam", "odmawiam", "odrzucone")
_WORD = "|".join(sorted(_YES + _NO, key=len, reverse=True))
#: The word alone, or followed by punctuation and the reason ("NIE — usuwa
#: pliki"). "Tak naprawdę…" and an echoed "TAK/NIE" or "TAK or NIE" are not answers.
_VERDICT_LINE = re.compile(rf"^({_WORD})(?!\w)\s*(?:[.,:;!)\]—–-]+\s*(.*))?$", re.IGNORECASE)
_LABEL = re.compile(r"^(?:werdykt|decyzja|odpowied[zź]|ocena|verdict|decision|answer)\s*[:=-]\s*",
                    re.IGNORECASE)
_REASON = re.compile(r"^(?:pow[oó]d|uzasadnienie|reason|why|because)\s*[:=-]\s*(.*)$", re.IGNORECASE)
_DOES = re.compile(r"^(?:co robi|co to robi|dzia[lł]anie|what it does|what|action)\s*[:=-]\s*(.*)$", re.IGNORECASE)
_JSON = re.compile(r"\{.*\}", re.DOTALL)
_JSON_VERDICT = ("verdict", "decision", "answer", "allow", "allowed", "approve", "approved",
                 "werdykt", "decyzja", "odpowiedz", "odpowiedź")
_JSON_REASON = ("reason", "powód", "powod", "why", "uzasadnienie", "explanation")


def _clean_line(line: str) -> str:
    line = line.strip().strip("*_`#>\"'„”“ ").strip()
    line = re.sub(r"^[-•]\s+", "", line)
    line = line.strip("*_` ").strip()
    return _LABEL.sub("", line).strip("*_` ").strip()


def _word_verdict(word: str) -> str:
    return "allow" if word.lower() in _YES else "deny"


def _from_json(text: str) -> tuple[str, str] | None:
    match = _JSON.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    lowered = {str(k).lower(): v for k, v in data.items()}
    verdict: str | None = None
    for key in _JSON_VERDICT:
        if key not in lowered:
            continue
        value = lowered[key]
        if isinstance(value, bool):
            verdict = "allow" if value else "deny"
        elif isinstance(value, str):
            hit = _VERDICT_LINE.match(_clean_line(value))
            if hit is not None:
                verdict = _word_verdict(hit.group(1))
        if verdict is not None:
            break
    if verdict is None:
        return None
    reason = next((str(lowered[k]) for k in _JSON_REASON if lowered.get(k)), "")
    return verdict, " ".join(reason.split())


_CAPS_VERDICT = re.compile(r"(?<![\w/])(TAK|NIE)(?![\w/])")


def parse_reason(text: str) -> str:
    """The "Co robi: … Powód: …" of an answer, without its verdict ("" when it has none)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    does = next((m.group(1).strip("*_` ") for m in (_DOES.match(_clean_line(x)) for x in lines) if m), "")
    why = ""
    for index, raw in enumerate(lines):
        hit = _REASON.match(_clean_line(raw))
        if hit is not None:
            tail = []
            for x in lines[index + 1:]:
                if _VERDICT_LINE.match(_clean_line(x)) or _DOES.match(_clean_line(x)):
                    break
                tail.append(_clean_line(x))
            why = " ".join([hit.group(1).strip("*_` "), *tail])
            break
    if not why:
        why = " ".join(_clean_line(x) for x in lines
                       if not _DOES.match(_clean_line(x)) and not _VERDICT_LINE.match(_clean_line(x)))
    why = _clip(" ".join(why.split()).strip(" -—–:"), 400)
    if does:
        does = _clip(" ".join(does.split()).rstrip(" ."), 300)
        return f"Co robi: {does}. Powód: {why}" if why else f"Co robi: {does}."
    return why


def parse_verdict(text: str) -> tuple[str, str] | None:
    """``("allow" | "deny", reason)`` from a reviewer's answer; ``None`` if unclear.

    Takes the asked-for two lines (``TAK`` / ``Powód: …``) and the usual
    variations: English yes/no, allow/deny, a ``Werdykt:`` label, markdown
    around the word, the reason on the verdict line (``NIE — usuwa pliki``),
    or a JSON object. An echoed ``TAK/NIE`` is not an answer.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.lstrip().startswith(("{", "```")):
        found = _from_json(text)
        if found is not None:
            return found
    lines = [ln for ln in (raw.strip() for raw in text.splitlines()) if ln]
    verdict: str | None = None
    inline = ""
    rest_from = len(lines)
    for index, raw in enumerate(lines):
        line = _clean_line(raw)
        if _REASON.match(line) or _DOES.match(line):
            continue
        hit = _VERDICT_LINE.match(line)
        if hit is None:
            continue
        verdict = _word_verdict(hit.group(1))
        inline = (hit.group(2) or "").strip()
        rest_from = index + 1
        break
    if verdict is None:
        found = _from_json(text)
        if found is not None:
            return found
        # "… więc TAK." — the verdict said in a sentence: the last word written
        # in capitals, the way the answer format asks for it.
        caps = set(_CAPS_VERDICT.findall(text))
        if len(caps) != 1:
            return None                  # none, or both ("TAK or NIE" echoed back)
        lines = [x for x in lines if not _VERDICT_LINE.match(_clean_line(x))]
        return _word_verdict(caps.pop()), parse_reason("\n".join(lines)) or ""
    does = next((m.group(1).strip("*_` ") for m in (_DOES.match(_clean_line(x)) for x in lines) if m), "")
    others = [x for x in lines if not _DOES.match(_clean_line(x))]
    reason = ""
    for index, raw in enumerate(others):
        hit = _REASON.match(_clean_line(raw))
        if hit is not None:
            follow: list[str] = []
            for x in others[index + 1:]:
                if _VERDICT_LINE.match(_clean_line(x)):
                    break                # the verdict comes last now; it is not part of the reason
                follow.append(_clean_line(x))
            reason = " ".join([hit.group(1).strip("*_` "), *follow])
            break
    if not reason:
        tail = [x for x in lines[rest_from:] if not _DOES.match(_clean_line(x))]
        reason = " ".join([inline, *(_clean_line(x) for x in tail)])
    if not reason:                       # a bare verdict on the last line: the reason came first
        reason = " ".join(_clean_line(x) for x in lines[:rest_from - 1] if not _DOES.match(_clean_line(x)))
    reason = _clip(" ".join(reason.split()).strip(" -—–:"), 400)
    if does:
        does = _clip(" ".join(does.split()).rstrip(" ."), 300)
        reason = f"Co robi: {does}. Powód: {reason}" if reason else f"Co robi: {does}."
    return verdict, reason
