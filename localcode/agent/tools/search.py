"""Search tools: ``glob`` (find files by name) and ``grep`` (search contents).

Both skip the usual noise (``.git``, ``node_modules``, ``.venv``, build output,
patterns from the project's ``.gitignore``) and print paths relative to the
project so the model can pass them straight to ``read_file``.

``grep`` uses a real ripgrep binary when one exists (``shutil.which("rg")`` —
note that on some machines ``rg`` is only a shell *function*, invisible to
``which``; ``LOCALCODE_RIPGREP`` can point at a binary explicitly) and otherwise a
pure-Python engine running in a worker thread: whole-file regex pre-filter, then
per-line matching, binary files skipped, a time budget, and cancellation.
Invalid regexes fall back to a literal search (the output says so), and the
grep/BRE habit ``foo\\|bar`` is read as alternation.

Output formats are compact and stable: ``path:line: text`` for matches,
``path-line- text`` for context lines, ``--`` between groups, and a footer with
the totals plus a hint when results were cut.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from localcode.agent.tools.base import ToolContext, ToolResult
from localcode.agent.tools.fs import (
    IGNORED_NAMES,
    PATH_ALIASES,
    CodingTool,
    IgnoreRules,
    ToolInputError,
    arg_bool,
    arg_int,
    arg_str,
    binary_kind,
    clean_path_text,
    cwd_of,
    display_path,
    en_plural,
    expand_braces,
    glob_to_regex,
    not_found_error,
    pl_plural,
    resolve_path,
    short,
    walk_files,
)

GREP_TIME_BUDGET_S = 20.0
GREP_MAX_FILE_BYTES = 10 * 1024 * 1024
GREP_HARD_MATCH_CAP = 10_000       # stop counting beyond this many matches
GREP_LINE_CHARS = 240
GLOB_MAX_SCAN = 200_000
MAX_CONTEXT = 10


# --------------------------------------------------------------------------- #
# glob
# --------------------------------------------------------------------------- #


def _static_prefix(pattern: str) -> tuple[str, str]:
    """Split ``src/components/**/*.tsx`` into (``src/components``, ``**/*.tsx``)."""
    parts = pattern.split("/")
    prefix: list[str] = []
    for i, part in enumerate(parts[:-1]):
        if any(c in part for c in "*?[{"):
            return "/".join(prefix), "/".join(parts[i:])
        prefix.append(part)
    return "/".join(prefix), parts[-1]


class GlobTool(CodingTool):
    name = "glob"
    kind = "read"
    description = "Find files by name pattern, e.g. '**/*.py' or 'src/**/test_*.ts'. Newest first."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob; a pattern without '/' matches at any depth"},
            "path": {"type": "string", "description": "Directory to search (default: project root)"},
        },
        "required": ["pattern"],
    }
    aliases = {**PATH_ALIASES, "glob": "pattern", "glob_pattern": "pattern", "name": "pattern",
               "file_pattern": "pattern", "query": "pattern", "filename_pattern": "pattern",
               "patterns": "pattern", "dir": "path", "directory": "path", "folder": "path",
               "search_path": "path", "root": "path", "cwd": "path"}
    primary = "pattern"
    usage = 'glob({"pattern": "**/*.py", "path": "src"})'

    def make_title(self, args: dict[str, Any]) -> str:
        pattern = arg_str(args, "pattern") or "?"
        path = clean_path_text(arg_str(args, "path") or "")
        where = f" w {short(path, 40)}" if path and path not in (".", "./") else ""
        return f"Szukanie plików: {short(pattern, 60)}{where}"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_pattern = args.get("pattern")
        if isinstance(raw_pattern, list):
            patterns = [str(p) for p in raw_pattern if str(p).strip()]
        else:
            text = arg_str(args, "pattern", required=True, allow_empty=False) or ""
            patterns = [p for p in re.split(r"[,\s]+(?![^{]*})", text.strip()) if p]
        if not patterns:
            raise ToolInputError("Missing required argument 'pattern', e.g. '**/*.py'.")
        base = resolve_path(arg_str(args, "path") or ".", ctx)
        if not base.exists():
            raise not_found_error(base, ctx, what="Directory")
        if not base.is_dir():
            raise ToolInputError(f"{display_path(base, ctx)} is a file, not a directory. "
                                 "Pass a directory as path, or use read_file for the file.",
                                 summary="To jest plik")
        return await asyncio.to_thread(self._glob, patterns, base, ctx)

    def _glob(self, patterns: list[str], base: Path, ctx: ToolContext) -> ToolResult:
        rules = IgnoreRules(cwd_of(ctx))
        matches: dict[str, float] = {}
        scanned = 0
        for pattern in patterns:
            pat = unicodedata.normalize("NFC", pattern.strip())
            root = base
            if pat.startswith("~") or os.path.isabs(pat):
                pat = os.path.expanduser(pat)
                prefix, rest = _static_prefix(pat)
                root = Path(prefix or "/")
                pat = rest
            else:
                while pat.startswith("./"):
                    pat = pat[2:]
                prefix, rest = _static_prefix(pat)
                if prefix:
                    root = Path(os.path.normpath(base / prefix))
                    pat = rest
            if not pat:
                continue
            by_name = "/" not in pat
            regex = glob_to_regex(pat)
            if not root.is_dir():
                continue
            root_s = str(root)
            for path, entry in walk_files(root, rules, max_entries=GLOB_MAX_SCAN):
                scanned += 1
                subject = entry.name if by_name else path[len(root_s) + 1:].replace(os.sep, "/")
                subject = unicodedata.normalize("NFC", subject)
                if regex.match(subject):
                    try:
                        matches[path] = entry.stat().st_mtime
                    except OSError:
                        matches[path] = 0.0
        disp_patterns = ", ".join(patterns)
        if not matches:
            where = display_path(base, ctx)
            return ToolResult(ok=True,
                              output=f"No files match {disp_patterns!r} in {where} "
                                     f"(checked {en_plural(scanned, 'file')}; ignored folders are skipped).",
                              summary="Brak plików")
        ordered = sorted(matches.items(), key=lambda kv: (-kv[1], kv[0]))
        limit = max(10, min(int(ctx.limits.list_max_entries), 500))
        shown = [display_path(Path(p), ctx) for p, _ in ordered[:limit]]
        lines = list(shown)
        total = len(ordered)
        if total > limit:
            lines.append(f"[showing the {limit} newest of {total} files — use a more specific pattern or path]")
        else:
            lines.append(f"[{en_plural(total, 'file')}, newest first]")
        return ToolResult(ok=True, output="\n".join(lines),
                          summary=f"{total} {pl_plural(total, 'plik', 'pliki', 'plików')}")


# --------------------------------------------------------------------------- #
# grep
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=4)
def _probe_ripgrep(candidate: str) -> bool:
    try:
        out = subprocess.run([candidate, "--version"], capture_output=True, timeout=5,
                             stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.lower().startswith(b"ripgrep")


def find_ripgrep() -> str | None:
    """Path of a working ripgrep binary, or None (then the Python engine is used)."""
    for candidate in (os.environ.get("LOCALCODE_RIPGREP"), shutil.which("rg")):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK) \
                and _probe_ripgrep(candidate):
            return candidate
    return None


@dataclass
class _Query:
    pattern: str                 # what the user asked for
    regex: re.Pattern[str]       # compiled Python regex (literal-escaped when invalid)
    literal: bool                # the pattern is searched literally
    note: str | None             # model-facing remark about pattern handling
    ignore_case: bool
    context: int
    includes: list[str]


@dataclass
class _GrepHit:
    path: str                    # display path
    line: int
    text: str
    is_match: bool


@dataclass
class _GrepOutcome:
    hits: list[_GrepHit] = field(default_factory=list)   # in output order, up to the cap
    total_matches: int = 0
    files_with_matches: int = 0
    files_searched: int = 0
    capped: bool = False          # stopped counting at GREP_HARD_MATCH_CAP
    timed_out: bool = False
    engine: str = "python"


def _compile_query(pattern: str, ignore_case: bool) -> tuple[re.Pattern[str], bool, str | None, str]:
    """Returns (regex, literal?, note, effective pattern)."""
    flags = re.MULTILINE | (re.IGNORECASE if ignore_case else 0)
    effective = pattern
    note = None
    if "\\|" in pattern and not re.search(r"(?<!\\)\|", pattern):
        effective = pattern.replace("\\|", "|")
        note = "read '\\|' as alternation (regex syntax is 'a|b')"
    try:
        return re.compile(effective, flags), False, note, effective
    except re.error as e:
        return (re.compile(re.escape(pattern), flags), True,
                f"invalid regex ({e.msg}), searched as literal text", pattern)


def _normalize_includes(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r"[,;\s]+(?![^{]*})", str(value))
    out: list[str] = []
    for item in items:
        g = str(item).strip().strip("'\"")
        if not g:
            continue
        if re.fullmatch(r"\.?[A-Za-z0-9_+-]{1,10}", g) and not any(c in g for c in "*?[{/"):
            g = "*." + g.lstrip(".")   # "py" / ".py" → "*.py"
        out.append(g)
    return out


def _clip_match(text: str, span: tuple[int, int] | None) -> str:
    text = text.rstrip("\r\n")
    if len(text) <= GREP_LINE_CHARS:
        return text
    if span is None:
        return text[:GREP_LINE_CHARS] + "…"
    start = max(0, span[0] - GREP_LINE_CHARS // 3)
    end = min(len(text), start + GREP_LINE_CHARS)
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


class _IncludeFilter:
    def __init__(self, includes: list[str]) -> None:
        self.name_rx = []
        self.path_rx = []
        for inc in includes:
            for g in expand_braces(inc):
                (self.path_rx if "/" in g else self.name_rx).append(glob_to_regex(g.lstrip("./") if g.startswith("./") else g))
        self.active = bool(includes)

    def ok(self, name: str, rel: str) -> bool:
        if not self.active:
            return True
        return any(r.match(name) for r in self.name_rx) or any(r.match(rel) for r in self.path_rx)


def _python_grep(q: _Query, target: Path, ctx: ToolContext, limit: int, stop: threading.Event) -> _GrepOutcome:
    out = _GrepOutcome(engine="python")
    rules = IgnoreRules(cwd_of(ctx))
    inc = _IncludeFilter(q.includes)
    deadline = time.monotonic() + GREP_TIME_BUDGET_S
    root_s = str(target)
    if target.is_file():
        files = [(str(target), target.name)]
        iterator: Any = iter(files)
    else:
        iterator = ((p, e.name) for p, e in walk_files(target, rules))
    last_group: tuple[str, int] | None = None
    for path, name in iterator:
        if stop.is_set():
            break
        if time.monotonic() > deadline:
            out.timed_out = True
            break
        rel = path[len(root_s) + 1:].replace(os.sep, "/") if path.startswith(root_s + os.sep) else name
        if not inc.ok(name, rel):
            continue
        try:
            size = os.path.getsize(path)
            if size > GREP_MAX_FILE_BYTES:
                continue
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        out.files_searched += 1
        if binary_kind(Path(path), data[:8192]):
            continue
        text = data.decode("utf-8", "replace")
        if not q.regex.search(text):
            continue
        lines = text.split("\n")
        disp = display_path(Path(path), ctx)
        matched = [i for i, line in enumerate(lines) if q.regex.search(line)]
        if not matched:
            continue
        out.files_with_matches += 1
        for i in matched:
            out.total_matches += 1
            if out.total_matches > GREP_HARD_MATCH_CAP:
                out.capped = True
                out.total_matches = GREP_HARD_MATCH_CAP
                return out
            if sum(1 for h in out.hits if h.is_match) >= limit:
                continue
            lo = max(0, i - q.context)
            hi = min(len(lines) - 1, i + q.context)
            if q.context and last_group and last_group[0] == disp and lo <= last_group[1] + 1:
                lo = last_group[1] + 1
            elif q.context and out.hits:
                out.hits.append(_GrepHit(path="", line=0, text="--", is_match=False))
            for k in range(lo, hi + 1):
                if k == i:
                    m = q.regex.search(lines[k])
                    out.hits.append(_GrepHit(disp, k + 1, _clip_match(lines[k], m.span() if m else None), True))
                elif k > i:
                    if k in matched:
                        break  # the next match renders itself
                    out.hits.append(_GrepHit(disp, k + 1, _clip_match(lines[k], None), False))
                else:
                    out.hits.append(_GrepHit(disp, k + 1, _clip_match(lines[k], None), False))
            last_group = (disp, max(hi if not any(k in matched for k in range(i + 1, hi + 1)) else i, lo))
    return out


async def _ripgrep(rg: str, q: _Query, target: Path, ctx: ToolContext, limit: int) -> _GrepOutcome | None:
    """Run ripgrep; None when it cannot handle the query (the caller falls back to Python)."""
    root = cwd_of(ctx)
    try:
        rel = target.relative_to(root).as_posix()
    except ValueError:
        rel = str(target)
    cmd = [rg, "--json", "--hidden", "--no-require-git", "--sort=path", "--no-messages",
           f"--max-filesize={GREP_MAX_FILE_BYTES}"]
    if q.ignore_case:
        cmd.append("-i")
    if q.context:
        cmd += ["-C", str(q.context)]
    for name in sorted(IGNORED_NAMES):
        cmd += ["-g", f"!{name}"]
    for inc in q.includes:
        cmd += ["-g", inc]
    if q.literal:
        cmd.append("-F")
    cmd += ["-e", q.pattern if q.literal else q.regex.pattern, "--", rel or "."]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(root), stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=1 << 24)
    out = _GrepOutcome(engine="ripgrep")
    files_with: set[str] = set()
    last: tuple[str, int] | None = None
    shown_matches = 0
    deadline = time.monotonic() + GREP_TIME_BUDGET_S
    try:
        assert proc.stdout is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out.timed_out = True
                break
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                out.timed_out = True
                break
            if not raw:
                break
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("type")
            data = ev.get("data") or {}
            if kind == "summary":
                out.files_searched = int(((data.get("stats") or {}).get("searches")) or 0)
                continue
            if kind not in ("match", "context"):
                continue
            p = _rg_text(data.get("path"))
            if p.startswith("./"):
                p = p[2:]
            disp = display_path(Path(os.path.normpath(root / p)), ctx)
            line_no = int(data.get("line_number") or 0)
            text = _rg_text(data.get("lines"))
            if kind == "match":
                out.total_matches += 1
                files_with.add(disp)
                if out.total_matches > GREP_HARD_MATCH_CAP:
                    out.capped = True
                    out.total_matches = GREP_HARD_MATCH_CAP
                    break
                if shown_matches >= limit:
                    continue
                subs = data.get("submatches") or []
                span = None
                if subs:
                    b = text.encode("utf-8", "replace")
                    span = (len(b[: subs[0].get("start", 0)].decode("utf-8", "replace")),
                            len(b[: subs[0].get("end", 0)].decode("utf-8", "replace")))
                shown_matches += 1
                if q.context and last and (last[0] != disp or line_no > last[1] + 1):
                    out.hits.append(_GrepHit("", 0, "--", False))
                out.hits.append(_GrepHit(disp, line_no, _clip_match(text, span), True))
                last = (disp, line_no)
            else:
                if shown_matches >= limit and (not last or last[0] != disp or line_no > last[1] + 1):
                    continue
                if q.context and last and (last[0] != disp or line_no > last[1] + 1):
                    out.hits.append(_GrepHit("", 0, "--", False))
                out.hits.append(_GrepHit(disp, line_no, _clip_match(text, None), False))
                last = (disp, line_no)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
    out.files_with_matches = len(files_with)
    if proc.returncode == 2 and out.total_matches == 0 and not out.capped and not out.timed_out:
        return None  # e.g. a regex feature ripgrep does not support
    # drop trailing context of a match that was not shown
    while out.hits and not out.hits[-1].is_match and out.hits[-1].text == "--":
        out.hits.pop()
    return out


def _rg_text(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    if "text" in obj:
        return str(obj["text"])
    if "bytes" in obj:
        import base64

        try:
            return base64.b64decode(obj["bytes"]).decode("utf-8", "replace")
        except ValueError:
            return ""
    return ""


class GrepTool(CodingTool):
    name = "grep"
    kind = "read"
    description = "Search file contents with a regex. Returns 'path:line: text' for each matching line."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex (plain text works too)"},
            "path": {"type": "string", "description": "File or directory (default: project root)"},
            "include": {"type": "string", "description": "File filter, e.g. '*.py' or '*.{ts,tsx}'"},
            "ignore_case": {"type": "boolean"},
            "context": {"type": "integer", "description": "Lines of context around matches"},
        },
        "required": ["pattern"],
    }
    aliases = {**PATH_ALIASES, "query": "pattern", "regex": "pattern", "search": "pattern", "text": "pattern",
               "string": "pattern", "q": "pattern", "search_pattern": "pattern", "term": "pattern",
               "search_term": "pattern", "keyword": "pattern", "expression": "pattern",
               "dir": "path", "directory": "path", "folder": "path", "search_path": "path", "root": "path",
               "glob": "include", "file_pattern": "include", "files": "include", "filter": "include",
               "include_pattern": "include", "file_glob": "include", "type": "include", "ext": "include",
               "extension": "include", "file_type": "include", "includes": "include",
               "i": "ignore_case", "case_insensitive": "ignore_case", "insensitive": "ignore_case",
               "ignorecase": "ignore_case", "nocase": "ignore_case", "icase": "ignore_case",
               "c": "context", "context_lines": "context", "around": "context", "surrounding": "context",
               "lines_of_context": "context"}
    primary = "pattern"
    usage = 'grep({"pattern": "def main", "path": "src", "include": "*.py"})'

    def make_title(self, args: dict[str, Any]) -> str:
        pattern = arg_str(args, "pattern") or "?"
        path = clean_path_text(arg_str(args, "path") or "")
        where = f" w {short(path, 40)}" if path and path not in (".", "./") else ""
        return f"Szukanie „{short(pattern, 50)}”{where}"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = arg_str(args, "pattern", required=True, allow_empty=False) or ""
        ignore_case = arg_bool(args, "ignore_case")
        context = arg_int(args, "context", default=0, minimum=0, maximum=MAX_CONTEXT) or 0
        for k in ("a", "b", "after", "before", "after_context", "before_context"):
            if k in args and not context:
                context = arg_int(args, k, default=0, minimum=0, maximum=MAX_CONTEXT) or 0
        includes = _normalize_includes(args.get("include"))
        target = resolve_path(arg_str(args, "path") or ".", ctx)
        if not target.exists():
            raise not_found_error(target, ctx, what="Path")
        regex, literal, note, effective = _compile_query(pattern, ignore_case)
        q = _Query(pattern=effective, regex=regex, literal=literal, note=note, ignore_case=ignore_case,
                   context=context, includes=includes)
        limit = max(5, int(ctx.limits.grep_max_matches))

        outcome: _GrepOutcome | None = None
        rg = find_ripgrep()
        if rg and not target.is_file() or rg and target.is_file():
            outcome = await _ripgrep(rg, q, target, ctx, limit) if rg else None
        if outcome is None:
            stop = threading.Event()
            try:
                outcome = await asyncio.to_thread(_python_grep, q, target, ctx, limit, stop)
            finally:
                stop.set()
        return self._format(q, target, outcome, limit, ctx)

    @staticmethod
    def _format(q: _Query, target: Path, o: _GrepOutcome, limit: int, ctx: ToolContext) -> ToolResult:
        where = display_path(target, ctx)
        notes = []
        if q.note:
            notes.append(f"[note: {q.note}]")
        if o.timed_out:
            notes.append(f"[search stopped after {GREP_TIME_BUDGET_S:.0f} s — results are partial; "
                         "narrow it with path or include]")
        if o.total_matches == 0:
            msg = f"No matches for {q.pattern!r} in {where}"
            if o.files_searched:
                msg += f" (searched {en_plural(o.files_searched, 'file')})"
            msg += "."
            tips = []
            if not q.ignore_case and re.search(r"[A-Za-z]", q.pattern):
                tips.append("ignore_case=true")
            if q.includes:
                tips.append("removing include")
            if tips:
                msg += " Try " + " or ".join(tips) + "."
            return ToolResult(ok=True, output="\n".join([msg, *notes]), summary="Brak wyników")
        lines = []
        for h in o.hits:
            if h.text == "--" and not h.path:
                lines.append("--")
            elif h.is_match:
                lines.append(f"{h.path}:{h.line}: {h.text}")
            else:
                lines.append(f"{h.path}-{h.line}- {h.text}")
        shown = sum(1 for h in o.hits if h.is_match)
        total_txt = f"{'more than ' if o.capped else ''}{o.total_matches}"
        files_txt = en_plural(o.files_with_matches, "file")
        if shown < o.total_matches or o.capped:
            lines.append(f"[showing {shown} of {total_txt} matches in {files_txt} — narrow the search with a more "
                         "specific pattern, path or include='*.ext']")
        else:
            lines.append(f"[{en_plural(o.total_matches, 'match', 'matches')} in {files_txt}]")
        lines.extend(notes)
        n = o.total_matches
        summary = (f"{'>' if o.capped else ''}{n} {pl_plural(n, 'wynik', 'wyniki', 'wyników')} w "
                   f"{o.files_with_matches} {pl_plural(o.files_with_matches, 'pliku', 'plikach', 'plikach')}")
        return ToolResult(ok=True, output="\n".join(lines), summary=summary)
