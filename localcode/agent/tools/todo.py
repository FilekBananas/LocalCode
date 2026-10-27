"""``todo_write``: the agent's visible plan for a multi-step task.

A small model loses the thread of a long job quickly: after three tool calls it
no longer remembers that the user also asked for tests. One short list, rewritten
whenever something changes, is the cheapest fix — it costs ~10 tokens per item
and it is re-read by the model on every request because it lives in the tool
result history, not in a side channel.

The tool therefore:

* takes the **whole list** every time (models are bad at patching a list by
  index) and stores it in ``ctx.state['todos']``;
* is extremely forgiving about shapes, because this is the tool small models
  most often malform: a bare string, one line per task, markdown checkboxes
  (``- [x] done``), ``{"task": …, "state": "doing"}``, a status spelled
  ``"in-progress"``/``"completed"``/``"TODO"``, a list of strings, or a single
  object instead of a list;
* refuses only what it cannot interpret, with a message that shows the expected
  shape;
* returns a compact rendering (``[x] / [>] / [ ]``) which is both the model's
  confirmation and the UI's text.

``kind = "meta"``, so the permission policy lets it run in every mode — including
``plan``, where writing the plan is the whole point. The agent loop turns
:func:`get_todos` into a :class:`localcode.events.TodoUpdate` event after the
call.
"""

from __future__ import annotations

import re
from typing import Any

from localcode.agent.tools.base import ToolContext, ToolResult
from localcode.agent.tools.fs import CodingTool, ToolInputError, pl_plural, short

__all__ = [
    "MAX_TODOS",
    "STATE_KEY",
    "STATUSES",
    "TodoWriteTool",
    "get_todos",
    "normalize_todos",
    "render_todos",
]

STATE_KEY = "todos"
STATUSES: tuple[str, ...] = ("pending", "in_progress", "done")

MAX_TODOS = 30
MAX_CONTENT_CHARS = 200

_STATUS_ALIASES: dict[str, str] = {
    "pending": "pending", "todo": "pending", "to_do": "pending", "open": "pending",
    "not_started": "pending", "notstarted": "pending", "new": "pending", "waiting": "pending",
    "queued": "pending", "planned": "pending", "false": "pending", "0": "pending", "": "pending",
    "in_progress": "in_progress", "inprogress": "in_progress", "in_process": "in_progress",
    "active": "in_progress", "doing": "in_progress", "working": "in_progress",
    "current": "in_progress", "started": "in_progress", "running": "in_progress",
    "wip": "in_progress", "ongoing": "in_progress",
    "done": "done", "completed": "done", "complete": "done", "finished": "done", "closed": "done",
    "ok": "done", "success": "done", "x": "done", "true": "done", "1": "done", "resolved": "done",
}

_CONTENT_KEYS = ("content", "task", "text", "title", "description", "name", "todo", "item",
                 "activeform", "active_form", "label", "summary", "step", "goal")
_STATUS_KEYS = ("status", "state", "done", "completed", "progress", "stage")

_CHECKBOX = re.compile(r"^\s*(?:[-*+]\s*)?\[\s*([ xX~>\-*/])\s*\]\s*(.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_CHECKBOX_STATUS = {"x": "done", "~": "in_progress", ">": "in_progress", "/": "in_progress",
                    "-": "in_progress", "*": "in_progress", " ": "pending"}

_MARKS = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
_PL_STATUS = {"pending": "do zrobienia", "in_progress": "w toku", "done": "zrobione"}


def normalize_status(value: Any, default: str = "pending") -> str:
    """Canonical status for whatever the model wrote; unknown text falls back to ``default``."""
    if isinstance(value, bool):
        return "done" if value else "pending"
    if value is None:
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _STATUS_ALIASES.get(key, default)


def _content_of(item: dict[str, Any]) -> str:
    lowered = {str(k).strip().lower().replace("-", "_").replace(" ", "_"): v for k, v in item.items()}
    for key in _CONTENT_KEYS:
        value = lowered.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, bool)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _status_of(item: dict[str, Any], default: str = "pending") -> str:
    lowered = {str(k).strip().lower().replace("-", "_").replace(" ", "_"): v for k, v in item.items()}
    for key in _STATUS_KEYS:
        if key in lowered:
            return normalize_status(lowered[key], default)
    return default


def _from_text(line: str) -> dict[str, str] | None:
    """One text line → a todo item (markdown checkboxes and bullets understood)."""
    text = line.strip()
    if not text:
        return None
    status = "pending"
    match = _CHECKBOX.match(text)
    if match:
        status = _CHECKBOX_STATUS.get(match.group(1).lower(), "pending")
        text = match.group(2).strip()
    else:
        text = _BULLET.sub("", text, count=1).strip()
    if not text:
        return None
    # "fix the parser — done" / "fix the parser (in progress)"
    trailer = re.search(r"[\s]*[—–-]{1,2}\s*(\w[\w\s-]{0,14})$|\(\s*([\w\s-]{1,15})\s*\)$", text)
    if trailer:
        word = (trailer.group(1) or trailer.group(2) or "").strip()
        mapped = normalize_status(word, default="")
        if mapped:
            status = mapped
            text = text[:trailer.start()].strip()
    return {"content": text, "status": status} if text else None


def _items_from(raw: Any) -> list[Any]:
    """Whatever the model sent → a flat list of candidate items."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        # {"todos": [...]} slipped through, or a single item, or {"1": "task"}
        for key in ("todos", "items", "list", "tasks", "todo_list"):
            if key in raw:
                return _items_from(raw[key])
        if any(k in {str(x).lower() for x in raw} for k in _CONTENT_KEYS):
            return [raw]
        return [v for _, v in sorted(raw.items(), key=lambda kv: str(kv[0]))]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        return [line for line in text.splitlines() if line.strip()]
    return [raw]


def normalize_todos(raw: Any) -> list[dict[str, str]]:
    """Validate and canonicalise the model's list.

    Raises :class:`ToolInputError` (with the expected shape) when nothing usable
    is in there; an empty list is valid and clears the todo list.
    """
    candidates = _items_from(raw)
    if not candidates and raw not in (None, "", [], {}):
        raise ToolInputError(
            "todos must be a list of objects like "
            '{"content": "run the tests", "status": "pending"}.',
            summary="Zła lista zadań")
    items: list[dict[str, str]] = []
    bad: list[str] = []
    for candidate in candidates:
        item: dict[str, str] | None
        if isinstance(candidate, dict):
            content = _content_of(candidate)
            if not content:
                bad.append(str(candidate)[:60])
                continue
            item = {"content": content, "status": _status_of(candidate)}
        elif isinstance(candidate, str):
            item = _from_text(candidate)
        elif isinstance(candidate, (list, tuple)) and candidate:
            # ["run the tests", "done"]
            item = {"content": str(candidate[0]).strip(),
                    "status": normalize_status(candidate[1] if len(candidate) > 1 else None)}
            if not item["content"]:
                item = None
        elif candidate is None:
            continue
        else:
            item = _from_text(str(candidate))
        if item is None:
            continue
        content = " ".join(item["content"].split())
        if not content:
            continue
        if len(content) > MAX_CONTENT_CHARS:
            content = content[: MAX_CONTENT_CHARS - 1] + "…"
        items.append({"content": content, "status": item["status"]})
    if not items and bad:
        raise ToolInputError(
            f"None of the {len(bad)} todo entries has a 'content' field "
            '(expected {"content": "run the tests", "status": "pending"}).',
            summary="Zła lista zadań")
    if len(items) > MAX_TODOS:
        items = items[:MAX_TODOS]
    return items


def get_todos(ctx: ToolContext) -> list[dict[str, str]]:
    """The current todo list (a copy), for the agent loop's ``TodoUpdate`` event."""
    stored = ctx.state.get(STATE_KEY) or []
    return [dict(item) for item in stored if isinstance(item, dict)]


def render_todos(items: list[dict[str, str]]) -> str:
    """Compact rendering shown to the model and the user."""
    if not items:
        return "Todo list is empty."
    done = sum(1 for i in items if i["status"] == "done")
    lines = [f"Todo list ({done}/{len(items)} done):"]
    lines += [f"{_MARKS.get(i['status'], '[ ]')} {i['content']}" for i in items]
    return "\n".join(lines)


class TodoWriteTool(CodingTool):
    name = "todo_write"
    kind = "meta"
    description = ("Track a multi-step job: send the whole task list every time, "
                   "with one task in_progress.")
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete list, in order",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Short task, imperative"},
                        "status": {"type": "string", "enum": list(STATUSES)},
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["todos"],
    }
    aliases = {"todo": "todos", "items": "todos", "list": "todos", "tasks": "todos",
               "task_list": "todos", "todo_list": "todos", "todolist": "todos", "plan": "todos",
               "steps": "todos", "entries": "todos", "checklist": "todos"}
    primary = "todos"
    usage = ('todo_write({"todos": [{"content": "read the config", "status": "done"}, '
             '{"content": "fix the parser", "status": "in_progress"}]})')

    def make_title(self, args: dict[str, Any]) -> str:
        try:
            items = normalize_todos(args.get("todos"))
        except ToolInputError:
            return "Lista zadań"
        if not items:
            return "Lista zadań"
        current = next((i for i in items if i["status"] == "in_progress"), None)
        if current is not None:
            return f"Lista zadań: {short(current['content'], 50)}"
        done = sum(1 for i in items if i["status"] == "done")
        return f"Lista zadań ({done}/{len(items)})"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if "todos" not in args:
            raise ToolInputError(
                "Missing required argument 'todos' (the complete task list).")
        raw = args.get("todos")
        items = normalize_todos(raw)
        ctx.state[STATE_KEY] = [dict(item) for item in items]

        if not items:
            return ToolResult(ok=True, output="Todo list cleared.", summary="Lista zadań wyczyszczona")
        parts = [render_todos(items)]
        in_progress = [i for i in items if i["status"] == "in_progress"]
        if len(in_progress) > 1:
            parts.append("Note: keep exactly one task in_progress — the one you are doing now.")
        if len(_items_from(raw)) > MAX_TODOS:
            parts.append(f"Note: only the first {MAX_TODOS} tasks were kept — "
                         "a plan this long is too detailed.")
        done = sum(1 for i in items if i["status"] == "done")
        total = len(items)
        summary = (f"Lista zadań: {done}/{total} "
                   f"{pl_plural(done, 'zrobione', 'zrobione', 'zrobionych')}")
        if in_progress:
            summary += f" · {_PL_STATUS['in_progress']}: {short(in_progress[0]['content'], 40)}"
        return ToolResult(ok=True, output="\n".join(parts), summary=summary)
