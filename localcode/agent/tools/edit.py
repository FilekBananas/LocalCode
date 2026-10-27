"""``edit_file``: search-and-replace editing that survives small-model mistakes.

The model sends ``old_string`` (text copied from the file) and ``new_string``.
Small models rarely copy text perfectly, so matching is a cascade; the first
stage that finds the text wins, and a stage that finds it more than once stops
the cascade with an "ambiguous" error (never guess between places):

1. **exact** substring match (must be unique unless ``replace_all``);
2. **whitespace-tolerant, line based**: lines compared without leading/trailing
   whitespace (then with internal whitespace collapsed) – handles trailing
   spaces, tabs vs spaces, CRLF and a uniformly shifted indentation.
   ``new_string`` is **re-indented** to the file's real indentation: unchanged
   lines are copied verbatim from the file, other lines are mapped through the
   indentation levels observed in the matched region (tabs stay tabs);
3. **whitespace-insensitive regex** (runs of whitespace in ``old_string`` match
   any whitespace) – handles intra-line spacing and partial-line snippets;
4. **fuzzy line window** (``difflib`` ratio ≥ 0.92 over windows of n-1..n+1
   lines, the best window must be unique) – handles a typo or a missing line.

Before matching, accidental ``cat -n`` prefixes (``"   12\\t"`` copied from
``read_file`` output) are stripped, and a JSON double-escaped variant
(literal ``\\n``) is tried. On failure the closest region of the file is shown
with line numbers so the model can copy the real text.

The whole plan is computed without touching the disk, so ``permission()`` can
show the exact diff that ``run()`` will apply.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localcode.agent.tools.base import PermissionInfo, ToolContext, ToolResult
from localcode.agent.tools.fs import (
    PATH_ALIASES,
    BinaryFileError,
    CodingTool,
    ToolInputError,
    _run_blocking,
    arg_bool,
    arg_str,
    clean_path_text,
    clip_line,
    consume_permission_skipped,
    diff_counts,
    display_path,
    file_key,
    line_number_width,
    load_text,
    mark_permission_skipped,
    not_found_error,
    suggest_paths,
    outside_flag,
    pl_plural,
    resolve_path,
    save_text,
    short,
    syntax_problem,
    unified_diff,
)

FUZZY_THRESHOLD = 0.92
FUZZY_MARGIN = 0.02          # the best fuzzy window must beat any other by this much
FUZZY_MAX_FILE_LINES = 40_000
FUZZY_MIN_CHARS = 20         # shorter old_strings are too risky to match fuzzily
TAB_WIDTH = 4
SNIPPET_CONTEXT = 2
SNIPPET_MAX_LINES = 24


class EditError(Exception):
    """The edit cannot be applied; ``str(e)`` tells the model what to do."""

    def __init__(self, message: str, summary: str) -> None:
        super().__init__(message)
        self.summary = summary


@dataclass
class EditPlan:
    new_text: str
    method: str                       # exact | whitespace | regex | fuzzy | create
    count: int                        # replaced occurrences
    start_line: int                   # first replaced line (1-based, in the new text)
    new_lines: int                    # lines of the first replacement in the new text
    old_lines: list[int] = field(default_factory=list)   # start line of every occurrence (old text)
    notes: list[str] = field(default_factory=list)       # model-facing remarks


# --------------------------------------------------------------------------- #
# Input clean-up
# --------------------------------------------------------------------------- #

_NUMBERED_TAB = re.compile(r"^\s*(\d+)(?:\t(.*))?$", re.DOTALL)
_NUMBERED_LOOSE = re.compile(r"^\s*(?:L)?(\d+)(?:(?:[:|]\s?| {1,2}|\t)(.*))?$", re.DOTALL)


def _strip_numbers(text: str, pattern: re.Pattern[str]) -> str | None:
    """Remove ``cat -n`` style prefixes when EVERY line has one and the numbers are
    consecutive (so real code that starts with digits is never touched)."""
    lines = text.split("\n")
    trailing_nl = lines and lines[-1] == ""
    body = lines[:-1] if trailing_nl else lines
    if not body:
        return None
    numbers: list[int] = []
    out: list[str] = []
    for line in body:
        m = pattern.match(line)
        if not m:
            return None
        numbers.append(int(m.group(1)))
        out.append(m.group(2) or "")
    if any(b - a != 1 for a, b in zip(numbers, numbers[1:])):
        return None
    if len(body) == 1 and pattern is _NUMBERED_LOOSE and "\t" not in body[0]:
        return None  # a lone "12: foo" is too ambiguous
    return "\n".join(out) + ("\n" if trailing_nl else "")


def strip_line_numbers(text: str) -> str | None:
    """``text`` without read_file line-number prefixes, or None if it has none."""
    return _strip_numbers(text, _NUMBERED_TAB) or _strip_numbers(text, _NUMBERED_LOOSE)


def _unescape(text: str) -> str | None:
    if "\n" in text or "\\n" not in text:
        return None
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')


@dataclass
class _Candidate:
    old: str
    new: str
    note: str | None = None


def _candidates(old: str, new: str) -> list[_Candidate]:
    out = [_Candidate(old, new)]
    stripped_old = strip_line_numbers(old)
    if stripped_old is not None:
        stripped_new = strip_line_numbers(new)
        out.append(_Candidate(stripped_old, stripped_new if stripped_new is not None else new,
                              "removed the line-number prefixes copied from read_file output"))
    unescaped_old = _unescape(old)
    if unescaped_old is not None:
        unescaped_new = _unescape(new)
        out.append(_Candidate(unescaped_old, unescaped_new if unescaped_new is not None else new,
                              "interpreted literal \\n sequences as newlines"))
    return out


# --------------------------------------------------------------------------- #
# Indentation
# --------------------------------------------------------------------------- #


def _lead(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _width(ws: str) -> int:
    w = 0
    for c in ws:
        w = (w // TAB_WIDTH + 1) * TAB_WIDTH if c == "\t" else w + 1
    return w


def _render(width: int, use_tabs: bool) -> str:
    width = max(0, width)
    if use_tabs:
        return "\t" * (width // TAB_WIDTH) + " " * (width % TAB_WIDTH)
    return " " * width


def _unit(widths: list[int], default: int) -> int:
    distinct = sorted(set(widths))
    diffs = [b - a for a, b in zip(distinct, distinct[1:]) if b - a > 0]
    return min(diffs) if diffs else default


@dataclass
class FileIndent:
    """Indentation style of a whole file: unit width and whether it indents with tabs."""

    unit: int | None
    tabs: bool | None


def detect_indent(lines: list[str]) -> FileIndent:
    """Most common positive indentation step between consecutive non-blank lines."""
    steps: Counter[int] = Counter()
    tab_lines = space_lines = 0
    prev: int | None = None
    for line in lines[:5000]:
        if not line.strip():
            continue
        lead = _lead(line)
        if lead.startswith("\t"):
            tab_lines += 1
        elif lead.startswith(" "):
            space_lines += 1
        w = _width(lead)
        if prev is not None and w > prev:
            steps[w - prev] += 1
        prev = w
    unit = steps.most_common(1)[0][0] if steps else None
    tabs = None if tab_lines == space_lines == 0 else tab_lines > space_lines
    return FileIndent(unit=unit, tabs=tabs)


def _line_span(text: str) -> int:
    """Number of lines ``text`` occupies once inserted (a trailing newline adds none)."""
    if not text:
        return 1
    return max(1, text.count("\n") + (0 if text.endswith("\n") else 1))


def reindent(new_lines: list[str], old_lines: list[str], window: list[str],
             pairs: list[tuple[int, int]], file_indent: FileIndent | None = None) -> tuple[list[str], bool]:
    """Map ``new_lines`` (written in the model's indentation) onto the file's.

    ``pairs`` are (old index, window index) of lines known to correspond. Lines of
    ``new_lines`` identical to a paired old line are copied verbatim from the file.
    Returns (lines, changed?) where changed tells whether any indentation moved.
    """
    old_to_file: dict[int, str] = {}   # model width -> file leading whitespace
    for oi, wi in pairs:
        if not old_lines[oi].strip() or not window[wi].strip():
            continue
        ow = _width(_lead(old_lines[oi]))
        old_to_file.setdefault(ow, _lead(window[wi]))
    if not old_to_file:
        return list(new_lines), False
    file_leads = [_lead(window[wi]) for _, wi in pairs if window[wi].strip()]
    if any(l.startswith("\t") for l in file_leads):
        use_tabs = True
    elif any(l.startswith(" ") for l in file_leads):
        use_tabs = False
    else:
        use_tabs = bool(file_indent and file_indent.tabs)
    model_widths = [_width(_lead(l)) for l in old_lines + new_lines if l.strip()]
    unit_model = _unit(model_widths, TAB_WIDTH)
    if use_tabs:
        unit_file = TAB_WIDTH
    else:
        fallback = file_indent.unit if file_indent and file_indent.unit else unit_model
        unit_file = _unit([_width(l) for l in file_leads], fallback)
        if file_indent and file_indent.unit and unit_file % file_indent.unit == 0:
            unit_file = file_indent.unit
    anchors = sorted(old_to_file)

    verbatim: dict[int, int] = {}
    old_index_to_window = dict(pairs)
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                wi = old_index_to_window.get(i1 + k)
                if wi is not None:
                    verbatim[j1 + k] = wi

    out: list[str] = []
    changed = False
    for j, line in enumerate(new_lines):
        if j in verbatim:
            out.append(window[verbatim[j]])
            continue
        if not line.strip():
            out.append("")
            continue
        nw = _width(_lead(line))
        if nw in old_to_file:
            lead = old_to_file[nw]
        else:
            below = [a for a in anchors if a <= nw]
            anchor = below[-1] if below else anchors[0]
            levels = (nw - anchor) / unit_model if unit_model else 0
            lead = _render(_width(old_to_file[anchor]) + round(levels * unit_file), use_tabs)
        if lead != _lead(line):
            changed = True
        out.append(lead + line.lstrip(" \t"))
    return out, changed


# --------------------------------------------------------------------------- #
# Matching stages
# --------------------------------------------------------------------------- #


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _trim_blank(lines: list[str]) -> tuple[list[str], int, int]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end], start, len(lines) - end


def _trim_like(lines: list[str], lead: int, trail: int) -> list[str]:
    """Drop up to ``lead``/``trail`` blank lines from new_string's ends (mirrors old)."""
    s, e = 0, len(lines)
    while s < min(lead, e) and not lines[s].strip():
        s += 1
    t = 0
    while t < trail and e > s and not lines[e - 1].strip():
        e -= 1
        t += 1
    return lines[s:e]


def _ambiguous(disp: str, starts: list[int]) -> EditError:
    shown = ", ".join(str(n) for n in starts[:8]) + (" …" if len(starts) > 8 else "")
    return EditError(
        f"old_string matches {len(starts)} places in {disp} (lines {shown}). Include more surrounding "
        "lines in old_string so it matches only one place, or set replace_all=true to change all of them.",
        summary=f"Tekst występuje {len(starts)} razy")


def _stage_exact(text: str, old: str, new: str, replace_all: bool, disp: str) -> EditPlan | None:
    count = text.count(old)
    if count == 0:
        return None
    starts: list[int] = []
    idx = text.find(old)
    while idx != -1:
        starts.append(idx)
        idx = text.find(old, idx + len(old))
    if len(starts) > 1 and not replace_all:
        raise _ambiguous(disp, [_line_of(text, i) for i in starts])
    first = starts[0]
    new_text = text.replace(old, new) if replace_all else text[:first] + new + text[first + len(old):]
    return EditPlan(new_text=new_text, method="exact", count=len(starts) if replace_all else 1,
                    start_line=_line_of(new_text, first), new_lines=_line_span(new),
                    old_lines=[_line_of(text, i) for i in starts])


def _key_strip(line: str) -> str:
    return line.strip()


def _key_collapse(line: str) -> str:
    return " ".join(line.split())


def _stage_lines(text: str, old: str, new: str, replace_all: bool, disp: str,
                 key: Any) -> EditPlan | None:
    file_lines = text.split("\n")
    old_trim, lead, trail = _trim_blank(old.split("\n"))
    if not old_trim:
        return None
    n = len(old_trim)
    want = [key(l) for l in old_trim]
    keys = [key(l) for l in file_lines]
    matches: list[int] = []
    i = 0
    while i <= len(file_lines) - n:
        if keys[i] == want[0] and keys[i:i + n] == want:
            matches.append(i)
            i += n
        else:
            i += 1
    if not matches:
        return None
    if len(matches) > 1 and not replace_all:
        raise _ambiguous(disp, [m + 1 for m in matches])
    new_trim = _trim_like(new.split("\n"), lead, trail)
    pairs = [(k, k) for k in range(n)]
    indent = detect_indent(file_lines)
    reindented_any = False
    first_new_len = 0
    for m in reversed(matches):
        window = file_lines[m:m + n]
        repl, changed = reindent(new_trim, old_trim, window, pairs, indent)
        reindented_any |= changed
        file_lines[m:m + n] = repl
        first_new_len = len(repl)
    notes = ["old_string matched with different whitespace"]
    if reindented_any:
        notes.append("new_string was re-indented to match the file")
    new_text = "\n".join(file_lines)
    # start line of the first match in the new text: earlier replacements don't exist (we go in order)
    return EditPlan(new_text=new_text, method="whitespace", count=len(matches) if replace_all else 1,
                    start_line=matches[0] + 1, new_lines=max(1, first_new_len),
                    old_lines=[m + 1 for m in matches], notes=notes)


def _stage_regex(text: str, old: str, new: str, replace_all: bool, disp: str) -> EditPlan | None:
    core = old.strip()
    if not core or not re.search(r"\s", core):
        return None
    pieces = re.split(r"(\s+)", core)
    pattern = "".join(r"\s+" if p.isspace() else re.escape(p) for p in pieces if p)
    try:
        rx = re.compile(pattern)
    except re.error:  # pragma: no cover - escaped input always compiles
        return None
    found = list(rx.finditer(text))
    if not found:
        return None
    if len(found) > 1 and not replace_all:
        raise _ambiguous(disp, [_line_of(text, m.start()) for m in found])
    replacement = new.strip() if old != core else new
    targets = found if replace_all else found[:1]
    out = []
    last = 0
    for m in targets:
        out.append(text[last:m.start()])
        out.append(replacement)
        last = m.end()
    out.append(text[last:])
    new_text = "".join(out)
    first = found[0].start()
    return EditPlan(new_text=new_text, method="regex", count=len(targets),
                    start_line=_line_of(new_text, first), new_lines=_line_span(replacement),
                    old_lines=[_line_of(text, m.start()) for m in found],
                    notes=["old_string matched ignoring differences in spacing"])


def _windows(file_stripped: list[str], old_stripped: list[str], sizes: list[int],
             threshold: float) -> list[tuple[float, int, int]]:
    target = "\n".join(old_stripped)
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(target)
    found: list[tuple[float, int, int]] = []
    for size in sizes:
        if size < 1 or size > len(file_stripped):
            continue
        for i in range(0, len(file_stripped) - size + 1):
            sm.set_seq1("\n".join(file_stripped[i:i + size]))
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            r = sm.ratio()
            if r >= threshold:
                found.append((r, i, size))
    return found


def _stage_fuzzy(text: str, old: str, new: str, replace_all: bool, disp: str) -> EditPlan | None:
    file_lines = text.split("\n")
    old_trim, lead, trail = _trim_blank(old.split("\n"))
    if not old_trim or len(file_lines) > FUZZY_MAX_FILE_LINES:
        return None
    if len("".join(l.strip() for l in old_trim)) < FUZZY_MIN_CHARS:
        return None
    file_stripped = [l.strip() for l in file_lines]
    old_stripped = [l.strip() for l in old_trim]
    n = len(old_trim)
    found = _windows(file_stripped, old_stripped, [n, n - 1, n + 1], FUZZY_THRESHOLD)
    if not found:
        return None
    found.sort(key=lambda t: (-t[0], t[1]))
    best_r, best_i, best_size = found[0]
    rivals = [(r, i, s) for r, i, s in found[1:]
              if (i + s <= best_i or i >= best_i + best_size) and r >= best_r - FUZZY_MARGIN]
    if rivals:
        starts = sorted({best_i + 1} | {i + 1 for _, i, _ in rivals})
        raise EditError(
            f"old_string does not match {disp} exactly and several places are similar "
            f"(lines {', '.join(str(s) for s in starts[:8])}). Copy the exact text of the one you mean "
            "from read_file output, with a few more surrounding lines.",
            summary="Kilka podobnych miejsc")
    window = file_lines[best_i:best_i + best_size]
    sm = difflib.SequenceMatcher(None, old_stripped, [l.strip() for l in window], autojunk=False)
    pairs: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace") and (i2 - i1) == (j2 - j1):
            pairs.extend((i1 + k, j1 + k) for k in range(i2 - i1))
    new_trim = _trim_like(new.split("\n"), lead, trail)
    repl, _changed = reindent(new_trim, old_trim, window, pairs, detect_indent(file_lines))
    file_lines[best_i:best_i + best_size] = repl
    notes = [f"old_string did not match exactly; applied to the most similar region, lines "
             f"{best_i + 1}–{best_i + best_size} ({round(best_r * 100)}% similar) — check the result"]
    if replace_all:
        notes.append("replace_all was ignored for this approximate match (only one place changed)")
    return EditPlan(new_text="\n".join(file_lines), method="fuzzy", count=1, start_line=best_i + 1,
                    new_lines=max(1, len(repl)), old_lines=[best_i + 1], notes=notes)


def closest_region(text: str, old: str) -> tuple[float, int, int] | None:
    """(similarity, first line index, size) of the region most similar to ``old``."""
    file_lines = text.split("\n")
    old_trim, _, _ = _trim_blank(old.split("\n"))
    if not old_trim or len(file_lines) > FUZZY_MAX_FILE_LINES or not text.strip():
        return None
    file_stripped = [l.strip() for l in file_lines]
    target = "\n".join(l.strip() for l in old_trim)
    n = min(len(old_trim), len(file_lines))
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(target)
    best: tuple[float, int, int] | None = None
    for i in range(0, len(file_lines) - n + 1):
        sm.set_seq1("\n".join(file_stripped[i:i + n]))
        floor = best[0] if best else 0.0
        if sm.real_quick_ratio() <= floor or sm.quick_ratio() <= floor:
            continue
        r = sm.ratio()
        if best is None or r > best[0]:
            best = (r, i, n)
    return best


def plan_edit(text: str, old: str, new: str, *, replace_all: bool = False, disp: str = "the file") -> EditPlan:
    """Compute the edit without touching the disk. ``text`` uses LF newlines.
    Raises :class:`EditError` with a model-facing explanation when it cannot apply."""
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")
    if old == new:
        raise EditError("old_string and new_string are identical, so there is nothing to change. "
                        "Put the new text in new_string.", summary="Brak zmian do wprowadzenia")
    candidates = _candidates(old, new)
    stages = (
        lambda c: _stage_exact(text, c.old, c.new, replace_all, disp),
        lambda c: _stage_lines(text, c.old, c.new, replace_all, disp, _key_strip),
        lambda c: _stage_lines(text, c.old, c.new, replace_all, disp, _key_collapse),
        lambda c: _stage_regex(text, c.old, c.new, replace_all, disp),
        lambda c: _stage_fuzzy(text, c.old, c.new, replace_all, disp),
    )
    for stage in stages:
        for cand in candidates:
            if cand.old == cand.new or not cand.old.strip():
                continue
            plan = stage(cand)
            if plan is not None:
                if plan.new_text == text:
                    raise EditError("The replacement leaves the file unchanged (new_string equals the text "
                                    "already there).", summary="Brak zmian do wprowadzenia")
                if cand.note:
                    plan.notes.insert(0, cand.note)
                return plan
    raise _not_found(text, candidates[1].old if len(candidates) > 1 else old, disp)


def _missing_target(path: Path, ctx: ToolContext) -> ToolInputError:
    """``edit_file`` on a file that is not there.

    The generic "use list_dir or glob to find the right path" is wrong advice
    here and small models take it literally: asked to write a new file, one 0,6B
    model tried ``edit_file``, was told to look for the path, and then listed the
    same empty directory twice and gave up. What it needed to hear is that the
    file has to be created first. A mistyped path is the other common cause, so
    the suggestions still come along.
    """
    disp = display_path(path, ctx)
    hints = suggest_paths(path, ctx)
    msg = (f"File not found: {disp}. It does not exist yet, so there is nothing to edit — "
           f"create it with write_file instead (edit_file only changes a file that already exists).")
    if hints:
        msg += " If you meant an existing file: " + ", ".join(hints) + "."
    return ToolInputError(msg, summary="Nie ma takiego pliku")


def _not_found(text: str, old: str, disp: str) -> EditError:
    region = closest_region(text, old)
    lines = text.split("\n")
    msg = f"old_string was not found in {disp}."
    if region and region[0] >= 0.5:
        ratio, i, n = region
        s = max(0, i - 1)
        e = min(len(lines), i + n + 1)
        if e - s > SNIPPET_MAX_LINES:
            e = s + SNIPPET_MAX_LINES
        width = line_number_width(e)
        snippet = "\n".join(f"{k + 1:>{width}}\t{clip_line(lines[k], 300)}" for k in range(s, e))
        msg += (f" The most similar text ({round(ratio * 100)}% similar) is at lines {i + 1}–{i + n}:\n"
                f"{snippet}\n"
                "Copy old_string exactly from the file (without the line-number prefix and TAB), then retry.")
    else:
        msg += " Nothing similar was found. Call read_file to see the current content, then copy old_string exactly."
    return EditError(msg, summary="Nie znaleziono tekstu do zamiany")


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #


@dataclass
class _Prepared:
    path: Path
    disp: str
    old_text: str | None       # None: a new file is created
    new_text: str
    encoding: str
    newline: str
    plan: EditPlan
    diff: str
    stale: bool


def _snippet(text: str, start_line: int, n_lines: int) -> str:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return "(the file is now empty)"
    s = max(1, start_line - SNIPPET_CONTEXT)
    e = min(len(lines), start_line + n_lines - 1 + SNIPPET_CONTEXT)
    cut = e - s + 1 > SNIPPET_MAX_LINES
    if cut:
        e = s + SNIPPET_MAX_LINES - 1
    width = line_number_width(e)
    body = "\n".join(f"{k:>{width}}\t{clip_line(lines[k - 1], 300)}" for k in range(s, e + 1))
    if cut:
        body += f"\n[… snippet cut; use read_file with offset={e + 1} to see more]"
    return body


class EditFileTool(CodingTool):
    name = "edit_file"
    kind = "write"
    description = ("Replace old_string with new_string in a file. Copy old_string exactly from read_file "
                   "(without line numbers); it must be unique unless replace_all is true.")
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the project"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"},
        },
        "required": ["path", "old_string", "new_string"],
    }
    aliases = {
        **PATH_ALIASES,
        "old_str": "old_string", "old": "old_string", "old_text": "old_string", "search": "old_string",
        "find": "old_string", "original": "old_string", "from": "old_string", "oldstring": "old_string",
        "search_string": "old_string", "old_code": "old_string", "original_text": "old_string",
        "target_text": "old_string", "before": "old_string",
        "new_str": "new_string", "new": "new_string", "new_text": "new_string", "replace": "new_string",
        "replacement": "new_string", "to": "new_string", "newstring": "new_string", "new_code": "new_string",
        "replace_with": "new_string", "replacement_text": "new_string", "after": "new_string",
        "all": "replace_all", "replaceall": "replace_all", "global": "replace_all",
        "replace_all_occurrences": "replace_all", "all_occurrences": "replace_all",
    }
    usage = ('edit_file({"path": "src/app.py", "old_string": "<exact current text>", '
             '"new_string": "<replacement>"})')

    def make_title(self, args: dict[str, Any]) -> str:
        return f"Edycja {short(clean_path_text(arg_str(args, 'path') or '?'), 70)}"

    def _prepare(self, args: dict[str, Any], ctx: ToolContext) -> _Prepared:
        raw = arg_str(args, "path", required=True, allow_empty=False)
        assert raw is not None
        old = arg_str(args, "old_string")
        new = arg_str(args, "new_string")
        if old is None:
            raise ToolInputError("Missing required argument 'old_string' (the exact text to replace).")
        if new is None:
            raise ToolInputError("Missing required argument 'new_string' (use \"\" to delete old_string).")
        replace_all = arg_bool(args, "replace_all")
        path = resolve_path(raw, ctx)
        disp = display_path(path, ctx)
        if path.is_dir():
            raise ToolInputError(f"{disp} is a directory. edit_file needs a file path.", summary="To jest katalog")

        if not path.exists():
            if old.strip():
                raise _missing_target(path, ctx)
            content = new.replace("\r\n", "\n")
            if content and not content.endswith("\n"):
                content += "\n"
            plan = EditPlan(new_text=content, method="create", count=1, start_line=1,
                            new_lines=max(1, content.count("\n")))
            return _Prepared(path=path, disp=disp, old_text=None, new_text=content, encoding="utf-8",
                             newline="\n", plan=plan, diff=unified_diff("", content, disp, created=True),
                             stale=False)

        tf = load_text(path)
        key = file_key(path)
        stale = key in ctx.read_files and abs(ctx.read_files[key] - tf.mtime) > 1e-6
        if not old.strip():
            if tf.text.strip():
                raise ToolInputError(
                    f"old_string is empty but {disp} is not. To insert text, put an existing nearby line in "
                    "old_string and that line plus your new text in new_string. To replace the whole file use write_file.",
                    summary="Pusty old_string")
            content = new.replace("\r\n", "\n")
            plan = EditPlan(new_text=content, method="create", count=1, start_line=1,
                            new_lines=max(1, content.count("\n")))
        else:
            try:
                plan = plan_edit(tf.text, old, new, replace_all=replace_all, disp=disp)
            except EditError as e:
                message = str(e)
                if stale:
                    message += f"\nNote: {disp} changed on disk since you last read it — read it again first."
                raise ToolInputError(message, summary=e.summary) from None
        diff = unified_diff(tf.text, plan.new_text, disp)
        return _Prepared(path=path, disp=disp, old_text=tf.text, new_text=plan.new_text, encoding=tf.encoding,
                         newline=tf.newline, plan=plan, diff=diff, stale=stale)

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        try:
            norm = self.args(args)
        except ToolInputError:
            return None
        try:
            prep = await _run_blocking(self._prepare, norm, ctx)
        except (ToolInputError, BinaryFileError, OSError):
            mark_permission_skipped(ctx, self.name, norm)
            return None
        title = f"Edycja {prep.disp}"
        if prep.old_text is None:
            title += " (nowy plik)"
        elif prep.plan.count > 1:
            title += f" ({prep.plan.count} {pl_plural(prep.plan.count, 'zmiana', 'zmiany', 'zmian')})"
        title += outside_flag(prep.path, ctx)
        return PermissionInfo(title=title, preview=prep.diff)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        skipped = consume_permission_skipped(ctx, self.name, args)
        prep = await _run_blocking(self._prepare, args, ctx)
        if skipped:
            return ToolResult(ok=False,
                              output=f"Error: {prep.disp} changed while this edit was being checked. "
                                     "Call edit_file again.",
                              summary="Plik zmienił się w międzyczasie")
        before_problem = syntax_problem(prep.path, prep.old_text) if prep.old_text is not None else None
        mtime, used_encoding = await _run_blocking(
            lambda: save_text(prep.path, prep.new_text, encoding=prep.encoding, newline=prep.newline))
        ctx.read_files[file_key(prep.path)] = mtime
        plan = prep.plan
        added, removed = diff_counts(prep.diff)

        if plan.method == "create":
            head = f"Created {prep.disp}." if prep.old_text is None else f"Wrote {prep.disp}."
            summary = "Utworzono plik" if prep.old_text is None else f"+{added} −{removed}"
        elif plan.count > 1:
            where = ", ".join(str(n) for n in plan.old_lines[:10]) + (" …" if len(plan.old_lines) > 10 else "")
            head = f"Edited {prep.disp}: replaced {plan.count} occurrences (lines {where})."
            summary = f"Zamieniono {plan.count} {pl_plural(plan.count, 'wystąpienie', 'wystąpienia', 'wystąpień')}"
        else:
            end = plan.start_line + plan.new_lines - 1
            span = f"line {plan.start_line}" if end <= plan.start_line else f"lines {plan.start_line}–{end}"
            head = f"Edited {prep.disp} ({span})."
            summary = f"+{added} −{removed}"
        parts = [head]
        for note in plan.notes:
            parts.append(f"Note: {note}.")
        if prep.stale:
            parts.append(f"Warning: {prep.disp} had changed on disk since you last read it; "
                         "the edit was applied to the current content.")
        if used_encoding != prep.encoding:
            parts.append(f"Note: saved as UTF-8 (the text did not fit the original {prep.encoding} encoding).")
        problem = syntax_problem(prep.path, prep.new_text)
        if problem and not before_problem:
            parts.append(f"Warning: the file now has a syntax error at {problem}. Fix it with another edit.")
        if plan.method != "create" or prep.old_text is not None:
            if plan.count == 1 or plan.method != "exact":
                parts.append("Result:\n" + _snippet(prep.new_text, plan.start_line, plan.new_lines))
        return ToolResult(ok=True, output="\n".join(parts), summary=summary, diff=prep.diff)
