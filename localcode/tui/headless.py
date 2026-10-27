"""``localcode -p "prompt"`` — one turn, no terminal UI, scriptable output.

Three output formats:

``text``
    the assistant's answer on stdout as it is generated, nothing else;
``json``
    one object at the end: ``{"type": "result", "result": …, "isError": …,
    "stats": …}``;
``stream-json``
    one JSON object per line — every agent event exactly as
    :mod:`localcode.events` produced it, then the same ``result`` object.

Notices and errors go to stderr, so ``localcode -p … > answer.md`` always gets
only the answer.  The exit code is ``0`` when the turn finished normally and
``1`` otherwise (error, interrupt, step limit), which makes it usable in
scripts and in CI.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any, TextIO

from localcode.events import PERMISSION_DISMISSED
from localcode.tui import render
from localcode.tui.backends import Backend, BackendError
from localcode.tui.completion import at_budget_chars, expand_at_references

__all__ = ["OUTPUT_FORMATS", "run_headless"]

OUTPUT_FORMATS = ("text", "json", "stream-json")


async def run_headless(
    backend: Backend,
    prompt: str,
    *,
    output_format: str = "text",
    out: TextIO | None = None,
    err: TextIO | None = None,
    expand_files: bool = True,
    effort: str | None = None,
) -> int:
    """Run one turn and print it; returns the process exit code.

    ``effort`` (``localcode -p --effort off|low|medium|high|ultracode``)
    overrides ``settings.agent.reasoning_effort`` for this run; models whose
    chat template has no effort control ignore the thinking part of it.
    """
    if output_format not in OUTPUT_FORMATS:
        raise BackendError(f"Nieznany format wyjścia „{output_format}”. "
                           f"Dostępne: {', '.join(OUTPUT_FORMATS)}.")
    if effort is not None:
        await backend.set_effort(effort)
    out = out or sys.stdout
    err = err or sys.stderr
    started = time.monotonic()

    text = prompt
    if expand_files:
        text, attachments, warnings = expand_at_references(
            prompt, backend.cwd, max_chars=at_budget_chars(backend.context_limit))
        for warning in warnings:
            print(warning, file=err)

    events: list[dict[str, Any]] = []
    answer: list[str] = []
    errors: list[str] = []
    stats: dict[str, Any] = {}
    reason = "error"
    shown_review: tuple[str, str] | None = None

    await backend.send_user(text)
    while True:
        event = await backend.events.get()
        kind = event.get("type")
        if output_format == "stream-json":
            print(json.dumps(event, ensure_ascii=False), file=out, flush=True)
        elif output_format == "json":
            events.append(event)
        if kind == "text_delta":
            piece = str(event.get("text") or "")
            answer.append(piece)
            if output_format == "text":
                out.write(piece)
                out.flush()
        elif kind == "tool_call" and output_format == "text":
            print(f"● {event.get('title') or event.get('name')}", file=err)
        elif kind == "subagent_start" and output_format == "text" and event.get("background"):
            print(f"  ▸ pomocnik {event.get('agentId') or ''} w tle: {event.get('title') or ''}", file=err)
        elif kind == "subagent_end" and output_format == "text" and event.get("background"):
            report = str(event.get("report") or "").strip()
            tail = f" — {report.splitlines()[0][:120]}" if report else ""
            print(f"  └ pomocnik {event.get('agentId') or ''} skończył w tle{tail}", file=err)
        elif kind == "tool_result" and output_format == "text":
            print(f"  ⎿ {event.get('summary')}", file=err)
        elif kind == "auto_review" and output_format == "text":
            word = {"allow": "✓ Auto: TAK", "deny": "✗ Auto: NIE"}.get(
                str(event.get("verdict") or ""), "? Auto: pytam Ciebie")
            print(f"  {word} — {event.get('reason') or ''}", file=err)
            second = render.guard_line(event.get("guard"))
            if second is not None:
                print(second.plain, file=err)
            shown_review = (str(event.get("verdict") or ""), str(event.get("reason") or ""))
        elif kind == "permission_request":
            # Nobody can answer in headless mode: deny and tell the model why.
            review = event.get("review")
            if output_format == "text" and isinstance(review, dict) and (
                    str(review.get("verdict") or ""), str(review.get("reason") or "")) != shown_review:
                # The reviewer's opinion, when the auto_review line above did not say it.
                line = render.permission_review_line(review)
                if line is not None:
                    print(line.plain, file=err)
            # "dismissed": refused, but not the user's no — nothing is learned from it.
            await backend.answer_permission(str(event.get("requestId") or ""), PERMISSION_DISMISSED)
            errors.append(f"Odmówiono zgody na {event.get('tool')} "
                          "(tryb -p nie pyta; użyj --mode accept_edits, auto albo yolo).")
            if output_format == "text":
                print(errors[-1], file=err)
        elif kind == "error":
            message = str(event.get("message") or "")
            errors.append(message)
            if output_format == "text":
                print(message, file=err)
        elif kind == "turn_end":
            reason = str(event.get("reason") or "done")
            stats = dict(event.get("stats") or {})
            break

    body = "".join(answer).strip()
    if output_format == "text":
        if body and not body.endswith("\n"):
            out.write("\n")
        out.flush()
    is_error = reason != "done" or bool(errors and not body)
    result = {
        "type": "result",
        "subtype": "error" if is_error else "success",
        "isError": is_error,
        "reason": reason,
        "result": body,
        "errors": errors,
        "sessionId": backend.session_id,
        "modelKey": backend.model_key,
        "mode": backend.mode,
        "cwd": str(backend.cwd),
        "contextUsed": backend.context_used,
        "contextLimit": backend.context_limit,
        "durationMs": round((time.monotonic() - started) * 1000),
        "stats": stats,
    }
    if output_format == "json":
        result["events"] = events
        print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
    elif output_format == "stream-json":
        print(json.dumps(result, ensure_ascii=False), file=out, flush=True)
    return 1 if is_error else 0


async def run_headless_guarded(backend: Backend, prompt: str, *, timeout: float | None = None,
                               **kwargs: Any) -> int:
    """:func:`run_headless` with an optional wall-clock limit."""
    if timeout is None:
        return await run_headless(backend, prompt, **kwargs)
    try:
        return await asyncio.wait_for(run_headless(backend, prompt, **kwargs), timeout)
    except (asyncio.TimeoutError, TimeoutError):
        await backend.interrupt()
        print(f"Przekroczono limit czasu ({timeout:.0f} s).", file=sys.stderr)
        return 1
