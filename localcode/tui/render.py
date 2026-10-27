"""Rendering of agent events for the terminal (rich).

Everything the TUI draws lives here as small, side-effect-free functions that
return ``rich`` renderables, so they can be snapshot-tested with
``Console(record=True)``.  The look follows Claude Code:

* a tool call is one line ``● Odczyt src/app.py (1–200)`` followed by a result
  line ``  ⎿ 200 linii``;
* file changes are shown as a coloured unified diff;
* the context gauge is ``ctx 5,1k/8k (63%)``;
* notices (compaction, errors, statistics) are single dim/red lines.

All user-facing text is Polish; numbers use the Polish decimal comma.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rich.box import ROUNDED
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from localcode.remote.providers import name_with_provider

__all__ = [
    "THEME",
    "always_label",
    "bullet",
    "compaction_line",
    "context_breakdown",
    "context_gauge",
    "diff_renderable",
    "error_line",
    "fmt_bytes",
    "fmt_duration",
    "fmt_tokens",
    "mode_label",
    "model_line",
    "notice_line",
    "output_tail",
    "permission_panel",
    "reasoning_summary",
    "short_path",
    "status_line",
    "todo_panel",
    "tool_call_line",
    "tool_result_line",
    "auto_review_line",
    "guard_line",
    "permission_review_line",
    "turn_stats_line",
    "welcome_panel",
]

#: Colours used across the TUI (Claude Code-ish: warm accent, dim chrome).
THEME = Theme(
    {
        "lc.accent": "#d97757",
        "lc.title": "bold #d97757",
        "lc.dim": "grey50",
        "lc.tool": "bold #d97757",
        "lc.arg": "grey62",
        "lc.ok": "green",
        "lc.err": "bold red",
        "lc.warn": "yellow",
        "lc.add": "green",
        "lc.del": "red",
        "lc.hunk": "cyan",
        "lc.user": "bold white",
        "lc.key": "bold cyan",
        "lc.gauge": "grey62",
        "lc.gauge.hot": "yellow",
        "lc.gauge.full": "bold red",
    }
)

#: Glyphs (kept in one place so a future ASCII fallback is a one-line change).
DOT = "●"
ELBOW = "⎿"
ARROW = "→"
THINK = "✻"

_MODE_SHORT = {
    "default": "domyślny",
    "accept_edits": "akceptuj edycje",
    "auto": "auto",
    "yolo": "yolo",
    "plan": "plan",
}
_MODE_HELP = {
    "default": "pyta o zmiany plików i polecenia",
    "accept_edits": "zmienia pliki bez pytania, pyta o polecenia",
    "auto": "model-recenzent krytycznie ocenia każdą akcję i mówi dlaczego",
    "yolo": "nic nie pyta (poza niebezpiecznymi poleceniami)",
    "plan": "tylko odczyt — bez zmian i poleceń",
}
_TODO_GLYPH = {"done": ("✔", "lc.ok"), "in_progress": ("▸", "lc.accent"),
               "cancelled": ("✗", "lc.dim"), "pending": ("○", "lc.dim")}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _pl(value: float, digits: int = 1) -> str:
    """Polish number formatting (decimal comma, no trailing ``,0``)."""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def fmt_number(value: float | int | None, digits: int = 2) -> str:
    """Plain number with a Polish decimal comma (``0.596 → "0,6"``)."""
    if value is None:
        return "—"
    return _pl(float(value), digits)


def fmt_tokens(n: int | float | None) -> str:
    """``5123 → "5,1k"``, ``8000 → "8k"``, ``1_200_000 → "1,2M"``."""
    if not n:
        return "0"
    n = int(n)
    if abs(n) < 1000:
        return str(n)
    if abs(n) < 1_000_000:
        return _pl(n / 1000) + "k"
    return _pl(n / 1_000_000) + "M"


def fmt_bytes(n: int | float | None) -> str:
    """Human size with Polish decimals (``3.2 GB → "3,2 GB"``)."""
    if not n:
        return "0 B"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{_pl(value)} {unit}"
        value /= 1024
    return f"{_pl(value)} TB"  # pragma: no cover - unreachable


def fmt_duration(ms: float | int | None) -> str:
    """``1234 → "1,2 s"``, ``95_000 → "1 min 35 s"``."""
    if not ms:
        return "0 s"
    seconds = float(ms) / 1000.0
    if seconds < 60:
        return f"{_pl(seconds)} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} min {rest} s"


def short_path(path: str | os.PathLike[str], cwd: str | os.PathLike[str] | None = None,
               limit: int = 60) -> str:
    """Path relative to ``cwd`` (or with ``~``), shortened from the left."""
    text = str(path)
    try:
        if cwd is not None:
            text = str(Path(text).relative_to(Path(cwd)))
    except (ValueError, OSError):
        pass
    home = str(Path.home())
    if text.startswith(home + os.sep):
        text = "~" + text[len(home):]
    if len(text) > limit:
        text = "…" + text[-(limit - 1):]
    return text


def mode_label(mode: str, *, long: bool = False) -> str:
    short = _MODE_SHORT.get(mode, mode)
    return f"{short} ({_MODE_HELP[mode]})" if long and mode in _MODE_HELP else short


def model_line(model: dict[str, Any] | None, *, n_ctx: int | None = None) -> str:
    """``Qwen3 4B Instruct · Q4_K_M · GGUF · 8k kontekstu`` for the welcome box."""
    if not model:
        return "brak modelu"
    name = name_with_provider(str(model.get("displayName") or model.get("key") or "model"),
                              str(model.get("key") or ""))
    parts = [name]
    quant = model.get("quant")
    # the library's display name often already ends with "· Q4_K_M"
    if quant and str(quant).lower() not in name.lower():
        parts.append(str(quant))
    fmt = model.get("format")
    if fmt:
        parts.append(str(fmt).upper())
    ctx = n_ctx or model.get("contextLength")
    if ctx:
        parts.append(f"{fmt_tokens(int(ctx))} kontekstu")
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #


def welcome_panel(*, model: dict[str, Any] | None, cwd: str, mode: str,
                  n_ctx: int | None = None, backend: str = "local", chat: bool = False,
                  tips: Sequence[str] | None = None) -> Panel:
    """The box printed when the TUI starts."""
    body = Table.grid(padding=(0, 1))
    body.add_column(style="lc.dim", justify="right", no_wrap=True)
    body.add_column(overflow="fold")
    body.add_row("model", model_line(model, n_ctx=n_ctx))
    body.add_row("katalog", short_path(cwd, limit=70))
    if chat:
        body.add_row("tryb", "czat — bez narzędzi, rozmowa zapisuje się w aplikacji")
    else:
        body.add_row("tryb", mode_label(mode, long=True))
    if backend == "remote":
        body.add_row("silnik", "serwer LocalCode (modele współdzielone z aplikacją)")
    default_tips = [
        "/help — lista poleceń, /model — wybór modelu, /context — zużycie kontekstu",
        "@plik.py dołącza plik, Shift+Tab zmienia tryb, Esc przerywa odpowiedź",
    ]
    if chat:
        default_tips = [
            "/help — lista poleceń, /model — wybór modelu, /chat — powrót do agenta",
            "Esc przerywa odpowiedź, Ctrl+D kończy",
        ]
    hints = list(tips) if tips is not None else default_tips
    lines: list[RenderableType] = [body]
    if hints:
        lines.append(Text(""))
        for hint in hints:
            lines.append(Text(hint, style="lc.dim"))
    return Panel(Group(*lines), title="LocalCode", title_align="left",
                 border_style="lc.accent", box=ROUNDED, padding=(0, 1))


def context_gauge(used: int, limit: int) -> Text:
    """``ctx 5,1k/8k (63%)`` — yellow above 75 %, red above 90 %."""
    if not limit:
        return Text("ctx —", style="lc.gauge")
    percent = max(0, min(999, round(used * 100 / limit)))
    style = "lc.gauge"
    if percent >= 90:
        style = "lc.gauge.full"
    elif percent >= 75:
        style = "lc.gauge.hot"
    return Text(f"ctx {fmt_tokens(used)}/{fmt_tokens(limit)} ({percent}%)", style=style)


def context_breakdown(ctx: dict[str, Any] | None) -> RenderableType:
    """The table shown by ``/context``."""
    if not ctx or not ctx.get("limit"):
        return Text("Brak danych o kontekście — wyślij pierwszą wiadomość.", style="lc.dim")
    limit = int(ctx.get("limit") or 0)
    used = int(ctx.get("used") or 0)
    rows = [
        ("prompt systemowy", int(ctx.get("system") or 0)),
        ("opisy narzędzi", int(ctx.get("tools") or 0)),
        ("rozmowa", int(ctx.get("history") or 0)),
        ("rezerwa na odpowiedź", int(ctx.get("reserve") or 0)),
    ]
    table = Table(box=ROUNDED, border_style="lc.dim", title="Kontekst", title_style="lc.title",
                  title_justify="left", expand=False)
    table.add_column("część", style="lc.dim")
    table.add_column("tokeny", justify="right")
    table.add_column("udział", justify="right", style="lc.dim")
    for name, value in rows:
        share = f"{round(value * 100 / limit)}%" if limit else "—"
        table.add_row(name, fmt_tokens(value), share)
    free = max(0, limit - used)
    table.add_row("[bold]razem w prompcie[/bold]", f"[bold]{fmt_tokens(used)}[/bold]",
                  f"[bold]{round(used * 100 / limit)}%[/bold]")
    table.add_row("wolne", fmt_tokens(free), f"{round(free * 100 / limit)}%")
    table.add_row("okno modelu", fmt_tokens(limit), "100%")
    return table


# --------------------------------------------------------------------------- #
# Turn rendering
# --------------------------------------------------------------------------- #


def bullet(text: str, *, style: str = "lc.tool") -> Text:
    line = Text(f"{DOT} ", style=style)
    line.append(text, style="")
    return line


def tool_call_line(title: str, name: str = "") -> Text:
    """``● Odczyt src/app.py (1–200)``."""
    return bullet(title or name or "narzędzie")


def tool_result_line(summary: str, *, ok: bool = True) -> Text:
    """``  ⎿ 200 linii`` (red when the tool failed)."""
    line = Text(f"  {ELBOW} ", style="lc.dim")
    line.append(summary or ("gotowe" if ok else "błąd"), style="lc.dim" if ok else "lc.err")
    return line


_REVIEW_MARK = {"allow": ("✓", "TAK", "lc.ok"), "deny": ("✗", "NIE", "lc.err"),
                "ask": ("?", "pytam Ciebie", "lc.warn")}


def auto_review_line(verdict: str, reason: str, helper: str = "") -> Text:
    """``  ✓ Auto: TAK — powód`` — what auto mode's reviewer said about a call.

    ``helper`` ("h2"): the call was a helper agent's — ``Auto (pomocnik h2): …``.
    """
    glyph, word, style = _REVIEW_MARK.get(verdict, _REVIEW_MARK["ask"])
    who = f" (pomocnik {helper})" if helper else ""
    line = Text(f"  {glyph} Auto{who}: {word}", style=style)
    if reason:
        line.append(f" — {reason}", style="lc.dim")
    return line


#: What the reviewer's verdict means on a permission question (the Mac app's
#: card says the same): a NIE the user may overrule, a TAK that is only an
#: opinion (the call needs a human anyway), or no decision at all.
_REQUEST_REVIEW = {"deny": ("✗", "Recenzent: NIE", "lc.err"),
                   "allow": ("✓", "Recenzent: TAK (tylko opinia)", "lc.ok"),
                   "ask": ("?", "Recenzent nie zdecydował", "lc.warn")}


_GUARD_MARK = {"allow": ("✓", "TAK", "lc.ok"), "deny": ("✗", "NIE", "lc.err"),
               "ask": ("?", "bez decyzji", "lc.warn")}


def guard_line(guard: Any) -> Text | None:
    """``  ✗ Strażnik lokalny: NIE — …`` — the context-free local check (``agent/guard.py``)."""
    if not isinstance(guard, dict):
        return None
    glyph, word, style = _GUARD_MARK.get(str(guard.get("verdict") or ""), _GUARD_MARK["ask"])
    line = Text(f"  {glyph} Strażnik lokalny: {word}", style=style)
    reason = str(guard.get("reason") or "").strip()
    if reason:
        line.append(f" — {reason}", style="lc.dim")
    return line


def permission_review_line(review: Any) -> Text | None:
    """``  ✗ Recenzent: NIE — Co robi: … Powód: …`` above a permission question.

    With the local guard's opinion as a second line. ``None`` without a review
    (any mode but auto, or no gate gave an answer on a call that needed a
    human anyway).
    """
    if not isinstance(review, dict):
        return None
    glyph, label, style = _REQUEST_REVIEW.get(str(review.get("verdict") or ""),
                                              _REQUEST_REVIEW["ask"])
    line = Text(f"  {glyph} {label}", style=style)
    reason = str(review.get("reason") or "").strip()
    if reason:
        line.append(f" — {reason}", style="lc.dim")
    second = guard_line(review.get("guard"))
    if second is not None:
        line.append("\n")
        line.append_text(second)
    return line


def output_tail(output: str, *, lines: int = 8, width: int = 0) -> Text | None:
    """The last ``lines`` lines of a tool's output, indented and dimmed."""
    text = (output or "").rstrip()
    if not text:
        return None
    rows = text.splitlines()
    hidden = max(0, len(rows) - lines)
    shown = rows[-lines:] if hidden else rows
    out = Text()
    if hidden:
        out.append(f"     … {hidden} wcześniejszych linii\n", style="lc.dim")
    for index, row in enumerate(shown):
        if width and len(row) > width:
            row = row[: width - 1] + "…"
        out.append("     " + row, style="lc.dim")
        if index < len(shown) - 1:
            out.append("\n")
    return out


_DIFF_HEADER = re.compile(r"^(diff |index |--- |\+\+\+ |new file|deleted file|similarity|rename )")


def diff_renderable(diff: str, *, max_lines: int = 60) -> Text:
    """A unified diff coloured per line and indented under the tool card."""
    out = Text()
    rows = (diff or "").splitlines()
    hidden = max(0, len(rows) - max_lines)
    if hidden:
        rows = rows[:max_lines]
    for index, row in enumerate(rows):
        if _DIFF_HEADER.match(row):
            style = "lc.dim"
        elif row.startswith("@@"):
            style = "lc.hunk"
        elif row.startswith("+"):
            style = "lc.add"
        elif row.startswith("-"):
            style = "lc.del"
        else:
            style = "lc.arg"
        out.append("     " + row, style=style)
        if index < len(rows) - 1 or hidden:
            out.append("\n")
    if hidden:
        out.append(f"     … pominięto {hidden} linii różnic", style="lc.dim")
    return out


def todo_panel(items: Sequence[dict[str, Any]]) -> RenderableType:
    """The todo list the agent maintains, as a bullet block."""
    if not items:
        return Text("Lista zadań jest pusta.", style="lc.dim")
    out = Text()
    out.append(f"{DOT} ", style="lc.tool")
    out.append("Lista zadań\n")
    for index, item in enumerate(items):
        status = str(item.get("status") or "pending")
        glyph, style = _TODO_GLYPH.get(status, _TODO_GLYPH["pending"])
        content = str(item.get("content") or "")
        prefix = f"  {ELBOW} " if index == 0 else "    "
        out.append(prefix, style="lc.dim")
        out.append(f"{glyph} ", style=style)
        out.append(content, style="lc.dim" if status in ("done", "cancelled") else "")
        if index < len(items) - 1:
            out.append("\n")
    return out


def reasoning_summary(tokens: int, *, streaming: bool = False) -> Text:
    """``✻ Myślenie… (123 tokenów)`` — the collapsed reasoning block."""
    word = "token" if tokens == 1 else ("tokeny" if 2 <= tokens % 10 <= 4 and not 12 <= tokens % 100 <= 14
                                        else "tokenów")
    suffix = "…" if streaming else ""
    return Text(f"{THINK} Myślenie{suffix} ({tokens} {word})", style="lc.dim")


def goal_line(event: dict[str, Any]) -> Text:
    """``◎ Cel: …`` / ``↻ Kontynuuję pracę nad celem (2/20).`` / ``✓ Cel osiągnięty: …``."""
    from localcode.agent.tools.goal import notice_text

    state = str(event.get("state") or "")
    mark, style = {"done": ("✓", "lc.ok"), "blocked": ("?", "lc.warn"),
                   "limit": ("!", "lc.warn"), "stalled": ("!", "lc.warn"),
                   "cleared": ("·", "lc.dim")}.get(
        state, ("↻" if int(event.get("turns") or 0) else "◎", "lc.accent"))
    text = Text(f"{mark} ", style=style)
    text.append(notice_text(event), style="lc.dim" if state == "cleared" else style)
    return text


def compaction_line(before: int, after: int, strategy: str = "") -> Text:
    """``Kontekst skompaktowany: 12,4k → 5,1k tokenów``."""
    text = Text(f"Kontekst skompaktowany: {fmt_tokens(before)} {ARROW} {fmt_tokens(after)} tokenów",
                style="lc.dim")
    if strategy:
        text.append(f" ({_STRATEGY.get(strategy, strategy)})", style="lc.dim")
    return text


_STRATEGY = {
    "mask_tool_results": "ukryto stare wyniki narzędzi",
    "summarize": "podsumowano starsze tury",
    "drop_oldest": "usunięto najstarsze wiadomości",
}


def status_line(message: str, phase: str = "info") -> Text:
    style = "lc.warn" if phase == "loading" else "lc.dim"
    return Text(f"  {message}", style=style)


def notice_line(message: str) -> Text:
    return Text(f"  {message}", style="lc.dim")


def error_line(message: str) -> Text:
    return Text(f"✗ {message}", style="lc.err")


def turn_stats_line(stats: dict[str, Any] | None, *, reason: str = "done") -> Text:
    """``12 tok/s · 3 kroki · 1,8 s · 420 tokenów`` after every turn."""
    stats = stats or {}
    parts: list[str] = []
    tps = stats.get("genTps")
    if tps:
        parts.append(f"{_pl(float(tps))} tok/s")
    steps = stats.get("steps")
    if steps:
        parts.append(f"{steps} " + ("krok" if steps == 1 else "kroki" if 2 <= steps <= 4 else "kroków"))
    elapsed = stats.get("elapsedMs")
    if elapsed:
        parts.append(fmt_duration(elapsed))
    out = stats.get("completionTokens")
    if out:
        parts.append(f"{fmt_tokens(out)} tokenów odpowiedzi")
    cached = stats.get("cachedTokens")
    if cached:
        parts.append(f"{fmt_tokens(cached)} z cache")
    prefix = {"interrupted": "przerwano · ", "max_steps": "limit kroków · ", "error": "błąd · "}.get(reason, "")
    if not parts:
        return Text(f"  {prefix.rstrip(' ·') or 'gotowe'}", style="lc.dim")
    return Text("  " + prefix + " · ".join(parts), style="lc.dim")


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #


@dataclass
class PermissionChoice:
    """One answer offered by the permission prompt."""

    label: str
    decision: str


def always_label(tool: str, preview: str) -> str:
    """Polish label of the "don't ask again" option (mirrors ``PermissionPolicy``)."""
    try:
        from localcode.agent.permissions import command_prefix, parse_shell
    except Exception:  # pragma: no cover - agent package incomplete
        return f"nie pytaj więcej o {tool}"
    if tool != "bash":
        return f"nie pytaj więcej o narzędzie {tool}"
    parsed = parse_shell(preview or "")
    prefixes: list[str] = []
    for command in parsed.commands:
        if not command.argv:
            continue
        prefix = " ".join(command_prefix(command.argv))
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    if not prefixes:
        return "nie pytaj więcej o to polecenie"
    return "nie pytaj więcej o polecenia: " + ", ".join(prefixes)


def permission_choices(event: dict[str, Any]) -> list[PermissionChoice]:
    """The three Claude Code answers, in order (``1``, ``2``, ``3``)."""
    tool = str(event.get("tool") or "")
    preview = str(event.get("preview") or "")
    choices = [
        PermissionChoice("Tak", "allow"),
        PermissionChoice(f"Tak, i {always_label(tool, preview)}", "always"),
        PermissionChoice("Nie (Esc)", "deny"),
    ]
    # The question says which answers it accepts. The help bot leaves "always"
    # out — offering it here would promise something the agent then ignores.
    offered = event.get("options")
    if isinstance(offered, list) and offered:
        choices = [c for c in choices if c.decision in offered]
    return choices


def permission_panel(event: dict[str, Any], *, max_lines: int = 40) -> Panel:
    """The box shown above the permission question: what exactly would happen."""
    tool = str(event.get("tool") or "")
    title = str(event.get("title") or tool or "Operacja")
    preview = str(event.get("preview") or "")
    body: RenderableType
    if _looks_like_diff(preview):
        body = diff_renderable(preview, max_lines=max_lines)
    elif preview:
        text = Text()
        rows = preview.splitlines() or [""]
        hidden = max(0, len(rows) - max_lines)
        for index, row in enumerate(rows[:max_lines]):
            # The command in the key colour; "Co robi: …" above it is prose.
            command_row = tool == "bash" and not row.startswith("Co robi:")
            text.append(row, style="lc.key" if command_row else "")
            if index < min(len(rows), max_lines) - 1:
                text.append("\n")
        if hidden:
            text.append(f"\n… pominięto {hidden} linii", style="lc.dim")
        body = text
    else:
        body = Text("(bez podglądu)", style="lc.dim")
    return Panel(body, title=title, title_align="left", border_style="lc.warn", box=ROUNDED,
                 padding=(0, 1))


def permission_question(event: dict[str, Any], selected: int = 0) -> Text:
    """The numbered answers, with ``›`` on the selected one (used by tests)."""
    out = Text("Czy pozwolić na tę operację?\n", style="lc.warn")
    for index, choice in enumerate(permission_choices(event)):
        marker = "›" if index == selected else " "
        style = "bold" if index == selected else "lc.dim"
        out.append(f" {marker} {index + 1}. {choice.label}", style=style)
        if index < 2:
            out.append("\n")
    return out


def _looks_like_diff(text: str) -> bool:
    return bool(re.search(r"^@@ ", text or "", re.M)) or (text or "").startswith(("--- ", "diff "))


# --------------------------------------------------------------------------- #
# Tables shared by the TUI and the CLI
# --------------------------------------------------------------------------- #


def models_table(models: Iterable[dict[str, Any]], *, budget_bytes: int | None = None) -> Table:
    """``localcode models list`` / ``/models``."""
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False)
    table.add_column("klucz", overflow="fold")
    table.add_column("format")
    table.add_column("kwant.")
    table.add_column("rozmiar", justify="right")
    table.add_column("kontekst", justify="right")
    table.add_column("stan")
    for model in models:
        size = int(model.get("sizeBytes") or 0)
        flags = []
        if model.get("loaded"):
            flags.append("[lc.ok]załadowany[/lc.ok]")
        if budget_bytes and size > budget_bytes:
            flags.append("[lc.warn]za duży[/lc.warn]")
        caps = model.get("capabilities") or []
        if "embedding" in caps:
            flags.append("embeddingi")
        if "vision" in caps or model.get("hasVision"):
            flags.append("obrazy")
        table.add_row(
            str(model.get("key") or ""),
            str(model.get("format") or "").upper(),
            str(model.get("quant") or "—"),
            fmt_bytes(size),
            fmt_tokens(model.get("contextLength")) if model.get("contextLength") else "—",
            " ".join(flags) or "—",
        )
    return table


def instances_table(status: dict[str, Any]) -> RenderableType:
    """``localcode ps`` / ``/ps``."""
    instances = list(status.get("instances") or [])
    if not instances:
        return Text("Żaden model nie jest załadowany.", style="lc.dim")
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False)
    table.add_column("id")
    table.add_column("model", overflow="fold")
    table.add_column("stan")
    table.add_column("ctx", justify="right")
    table.add_column("sloty", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("tok/s", justify="right")
    for inst in instances:
        state = str(inst.get("state") or "")
        style = {"ready": "lc.ok", "busy": "lc.accent", "loading": "lc.warn",
                 "error": "lc.err"}.get(state, "lc.dim")
        table.add_row(
            str(inst.get("id") or ""),
            str(inst.get("displayName") or inst.get("modelKey") or ""),
            f"[{style}]{state}[/{style}]",
            fmt_tokens(inst.get("nCtx")),
            f"{inst.get('busySlots', 0)}/{inst.get('nParallel', 1)}",
            fmt_bytes(inst.get("memoryBytes")),
            _pl(float(inst.get("genTps") or 0)) if inst.get("genTps") else "—",
        )
    return table


def runtime_panel(status: dict[str, Any]) -> RenderableType:
    """Memory budget + runtime settings (``/runtime``)."""
    system = status.get("system") or {}
    budget = status.get("budget") or {}
    settings = status.get("settings") or {}
    head = Table.grid(padding=(0, 2))
    head.add_column(style="lc.dim", justify="right")
    head.add_column()
    head.add_row("pamięć systemu", f"{fmt_bytes(system.get('usedBytes'))} / "
                                   f"{fmt_bytes(system.get('totalBytes'))} "
                                   f"(wolne {fmt_bytes(system.get('availableBytes'))})")
    head.add_row("budżet modeli", f"{fmt_bytes(budget.get('usedBytes'))} / "
                                  f"{fmt_bytes(budget.get('limitBytes'))}")
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False, title="Ustawienia runtime",
                  title_style="lc.title", title_justify="left")
    table.add_column("ustawienie", style="lc.dim")
    table.add_column("wartość")
    table.add_column("klucz /runtime set", style="lc.dim")
    for camel, label, cli in (
        ("maxLoadedModels", "modeli naraz", "max-models"),
        ("maxInstancesPerModel", "instancji na model", "instances"),
        ("defaultParallel", "równoległych żądań", "parallel"),
        ("ramLimitGb", "limit RAM (GB)", "ram-gb"),
        ("idleTtlMinutes", "wyładuj po (min)", "idle-ttl"),
        ("defaultCtx", "domyślny kontekst", "ctx"),
        ("kvCacheType", "typ cache KV", "kv"),
        ("flashAttn", "flash attention", "flash-attn"),
        ("evictPolicy", "polityka zwalniania", "evict"),
    ):
        value = settings.get(camel)
        table.add_row(label, "auto" if value is None else str(value), cli)
    return Group(head, Text(""), table, instances_table(status))


_TUNE_LABELS = {
    "maxLoadedModels": "modeli naraz",
    "maxInstancesPerModel": "instancji na model",
    "defaultParallel": "równoległych żądań",
    "ramLimitGb": "limit RAM (GB)",
    "idleTtlMinutes": "wyładuj po (min)",
    "defaultCtx": "domyślny kontekst",
    "kvCacheType": "typ cache KV",
    "flashAttn": "flash attention",
    "gpuLayers": "warstwy na GPU",
    "evictPolicy": "polityka zwalniania",
}


def tune_table(recommendation: dict[str, Any]) -> RenderableType:
    """Proposed runtime settings with reasons (``localcode runtime tune``)."""
    machine = recommendation.get("machine") or {}
    head = Table.grid(padding=(0, 2))
    head.add_column(style="lc.dim", justify="right")
    head.add_column()
    ram_gb = (machine.get("totalRamBytes") or 0) / (1024 ** 3)
    chip = "Apple Silicon" if machine.get("isAppleSilicon") else "nie Apple Silicon"
    head.add_row("ten komputer", f"{ram_gb:.0f} GB RAM · {chip} · "
                                 f"{machine.get('performanceCores', 0)} rdzeni wydajności")
    model_key = recommendation.get("modelKey")
    if model_key:
        head.add_row("model", str(model_key))
    if recommendation.get("source") == "model":
        head.add_row("źródło", "reguły + druga opinia modelu")
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False,
                  title="Proponowane ustawienia runtime", title_style="lc.title", title_justify="left")
    table.add_column("ustawienie", style="lc.dim")
    table.add_column("wartość")
    table.add_column("powód", overflow="fold")
    for rec in recommendation.get("recommendations") or []:
        label = _TUNE_LABELS.get(rec.get("key"), str(rec.get("key")))
        table.add_row(label, str(rec.get("value")), str(rec.get("reason") or ""))
    parts: list[RenderableType] = [head, Text(""), table]
    note = recommendation.get("note")
    if note:
        parts.append(Text(""))
        parts.append(Text(str(note), style="lc.dim"))
    return Group(*parts)


def hub_table(models: Iterable[dict[str, Any]]) -> Table:
    """Hugging Face search results (``/search``, ``localcode models search``)."""
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False)
    table.add_column("#", justify="right", style="lc.dim")
    table.add_column("repozytorium", overflow="fold")
    table.add_column("format")
    table.add_column("mld par.", justify="right")
    table.add_column("pobrania", justify="right")
    table.add_column("opis", overflow="fold", style="lc.dim")
    for index, model in enumerate(models, 1):
        params = model.get("paramsB")
        downloads = model.get("downloads")
        table.add_row(
            str(index),
            str(model.get("id") or ""),
            str(model.get("format") or "").upper(),
            _pl(float(params)) if params else "—",
            f"{int(downloads):,}".replace(",", " ") if downloads else "—",
            str(model.get("description") or "")[:70],
        )
    return table


def subagent_line(text: str, *, ok: bool = True, started: bool = False,
                  finished: bool = False, remote: bool = False) -> str:
    """One indented line about a helper agent (``delegate``).

    Indented and dimmed on purpose: a helper's steps are context for the call
    above them, not events of the conversation. Reading them should be optional.
    """
    if started:
        mark = "☁" if remote else "▸"
        return f"[dim]  {mark} {text}[/dim]"
    if finished:
        return f"[dim]  └ {text}[/dim]"
    return f"[dim]  {'✓' if ok else '✗'} {text}[/dim]"
