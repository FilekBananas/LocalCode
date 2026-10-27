"""File-system tools (``read_file``, ``write_file``, ``list_dir``) and the helpers
shared by every LocalCode coding tool.

Design for small local models
-----------------------------
* **Forgiving inputs.** Arguments go through :func:`normalize_args`: common
  aliases (``file_path``/``filepath``/``file`` → ``path``), numbers given as
  strings (``"10"``), booleans as ``"true"``, a bare string instead of an object,
  a JSON string instead of an object, and a stray wrapper (``{"arguments": {...}}``).
  Paths accept ``~``, quotes, ``file://``, ``@path`` and "``/src/app.py``" meaning
  the project's ``src/app.py`` (the model forgot the path is relative).
* **Compact, self-explanatory output.** Every result is plain text sized to
  ``ctx.limits``; truncation keeps head and tail with an explicit
  ``[… N lines omitted — how to see them …]`` marker.
* **Errors that teach.** A failing call returns ``ok=False`` with an English
  message for the model that says exactly how to fix the call (usage line,
  "did you mean" paths, …) and a short Polish ``summary`` for the UI.

Shared pieces live here because every other tool module builds on them:
:class:`CodingTool` (argument normalisation, error handling, output clamping),
path resolution, text-file IO that preserves encoding/BOM/CRLF, unified diffs,
``.gitignore`` handling and the project walker.

``ctx.read_files`` keys are absolute, normalised paths (``str(Path)``); values
are the file's ``st_mtime`` when the model last saw its content (read, write or
edit). ``edit_file`` uses them to warn about files changed on disk meanwhile.
"""

from __future__ import annotations

import abc
import ast
import asyncio
import codecs
import difflib
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from localcode.agent.permissions import OUTSIDE_PROJECT_FLAG
from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolLimits, ToolResult

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Errors and argument handling
# --------------------------------------------------------------------------- #


class ToolInputError(Exception):
    """A problem with the model's call. ``str(e)`` is the (English) message for the
    model and must say how to fix the call; ``summary`` is a short Polish UI label."""

    def __init__(self, message: str, summary: str = "Błędne wywołanie narzędzia") -> None:
        super().__init__(message)
        self.summary = summary


_WRAPPER_KEYS = ("arguments", "args", "input", "params", "parameters", "kwargs")


def normalize_args(raw: Any, aliases: dict[str, str], primary: str | None = None) -> dict[str, Any]:
    """Turn whatever the model produced into a dict keyed by canonical parameter names.

    * ``None`` → ``{}``; a JSON string → the decoded object; any other bare string
      or a list → ``{primary: value}``.
    * ``{"arguments": {...}}`` style wrappers are unwrapped.
    * Keys are lower-cased, ``-`` → ``_``, leading dashes dropped, then mapped
      through ``aliases``. The canonical key wins over an alias when both are given.
    """
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        text = raw.strip()
        decoded: Any = None
        if text[:1] in ("{", "["):
            try:
                decoded = json.loads(text)
            except ValueError:
                decoded = None
        if isinstance(decoded, dict):
            raw = decoded
        elif isinstance(decoded, list) and primary:
            return {primary: decoded}
        elif primary:
            return {primary: raw}
        else:
            raise ToolInputError("Arguments must be a JSON object.")
    if isinstance(raw, list):
        if primary:
            return {primary: raw}
        raise ToolInputError("Arguments must be a JSON object.")
    if not isinstance(raw, dict):
        if primary:
            return {primary: raw}
        raise ToolInputError("Arguments must be a JSON object.")
    if len(raw) == 1:
        (only_key, only_val), = raw.items()
        if str(only_key).lower() in _WRAPPER_KEYS and isinstance(only_val, (dict, str)):
            return normalize_args(only_val, aliases, primary)
    out: dict[str, Any] = {}
    alias_hits: dict[str, Any] = {}
    for key, value in raw.items():
        k = str(key).strip().lower().lstrip("-").replace("-", "_").replace(" ", "_")
        canonical = aliases.get(k, k)
        if canonical == k:
            out[k] = value
        elif canonical not in alias_hits or alias_hits[canonical] in (None, ""):
            alias_hits[canonical] = value
    for k, v in alias_hits.items():
        if k not in out or out[k] in (None, ""):
            out[k] = v
    return out


def arg_str(args: dict[str, Any], name: str, *, required: bool = False,
            default: str | None = None, allow_empty: bool = True) -> str | None:
    """A string argument. Numbers are stringified, lists of strings joined by newlines,
    dicts/lists serialised as JSON (a model writing ``package.json`` as an object)."""
    value = args.get(name)
    if value is None:
        if required:
            raise ToolInputError(f"Missing required argument '{name}'.")
        return default
    if isinstance(value, str):
        text = value
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        text = "\n".join(value)
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    else:
        text = str(value)
    if required and not allow_empty and not text.strip():
        raise ToolInputError(f"Argument '{name}' must not be empty.")
    return text


_INT_RE = re.compile(r"^\s*([+-]?\d+)(?:\.0*)?\b")


def arg_int(args: dict[str, Any], name: str, *, default: int | None = None,
            minimum: int | None = None, maximum: int | None = None) -> int | None:
    """An integer argument; accepts ``"10"``, ``10.0``, ``"10 lines"``. Clamped to the range."""
    value = args.get(name)
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "null")):
        return default
    if isinstance(value, bool):
        raise ToolInputError(f"Argument '{name}' must be a whole number, got {value!r}.")
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ToolInputError(f"Argument '{name}' must be a whole number, got {value!r}.")
        n = int(value)
    else:
        m = _INT_RE.match(str(value))
        if not m:
            raise ToolInputError(f"Argument '{name}' must be a whole number, got {str(value)[:40]!r}.")
        n = int(m.group(1))
    if minimum is not None and n < minimum:
        n = minimum
    if maximum is not None and n > maximum:
        n = maximum
    return n


_TRUE = {"true", "yes", "y", "1", "on", "t"}
_FALSE = {"false", "no", "n", "0", "off", "f", "", "none", "null"}


def arg_bool(args: dict[str, Any], name: str, default: bool = False) -> bool:
    value = args.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ToolInputError(f"Argument '{name}' must be true or false, got {str(value)[:40]!r}.")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def cwd_of(ctx: ToolContext) -> Path:
    """The project root as an absolute Path (ctx.cwd may be a str in callers' hands)."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(ctx.cwd))))


def clean_path_text(raw: str) -> str:
    """Strip the decorations models put around paths: quotes, backticks, file://, @."""
    p = raw.strip()
    for _ in range(2):
        if len(p) >= 2 and p[0] == p[-1] and p[0] in "'\"`":
            p = p[1:-1].strip()
    if p.startswith("file://"):
        p = p[len("file://"):]
    return p


def resolve_path(raw: str, ctx: ToolContext) -> Path:
    """Absolute, normalised path for a model-supplied path (symlinks are kept)."""
    root = cwd_of(ctx)
    text = clean_path_text(raw)
    if not text:
        raise ToolInputError("The path is empty. Pass a file path relative to the project, e.g. 'src/app.py'.")
    if "\x00" in text:
        raise ToolInputError("The path contains a NUL character.")
    if text.startswith("@") and not (root / text).exists():
        text = text[1:]
    p = Path(os.path.expanduser(text))
    if not p.is_absolute():
        return Path(os.path.normpath(root / p))
    p = Path(os.path.normpath(p))
    if not p.exists() and not text.startswith("~"):
        # "/src/app.py" when the model meant the project's src/app.py
        parts = p.parts
        if len(parts) >= 2 and not Path(parts[0], parts[1]).exists():
            candidate = Path(os.path.normpath(root.joinpath(*parts[1:])))
            if candidate.exists() or candidate.parent.exists():
                return candidate
    return p


def is_within(path: Path, root: Path) -> bool:
    """True when ``path`` (after resolving symlinks) is ``root`` or below it."""
    real = os.path.realpath(path)
    real_root = os.path.realpath(root)
    return real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep)


def display_path(path: Path, ctx: ToolContext) -> str:
    """Path as shown to the model/UI: relative to the project when inside it."""
    root = cwd_of(ctx)
    try:
        rel = Path(path).relative_to(root)
        s = rel.as_posix()
        return s if s != "." else "."
    except ValueError:
        pass
    home = str(Path.home())
    s = str(path)
    if s == home or s.startswith(home + os.sep):
        return "~" + s[len(home):]
    return s


def file_key(path: Path) -> str:
    """Key used in ``ctx.read_files``."""
    return str(Path(os.path.normpath(os.path.abspath(path))))


def short(text: str, limit: int = 60) -> str:
    """One-line, length-limited text for UI titles."""
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Text files
# --------------------------------------------------------------------------- #

BINARY_EXTENSIONS: dict[str, str] = {
    **{e: "image" for e in ("png", "jpg", "jpeg", "gif", "bmp", "ico", "icns", "webp", "tiff", "tif",
                            "heic", "psd", "avif")},
    **{e: "archive" for e in ("zip", "gz", "tgz", "bz2", "xz", "7z", "rar", "tar", "zst", "dmg", "jar",
                              "whl", "pkg", "deb", "rpm", "apk", "ipa")},
    **{e: "audio/video" for e in ("mp3", "wav", "flac", "ogg", "m4a", "aac", "mp4", "mov", "avi", "mkv",
                                  "webm")},
    **{e: "document" for e in ("pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "key",
                               "pages", "numbers")},
    **{e: "compiled/binary" for e in ("so", "dylib", "dll", "exe", "o", "a", "pyc", "pyo", "class",
                                      "wasm", "bin", "dat", "db", "sqlite", "sqlite3", "node")},
    **{e: "font" for e in ("ttf", "otf", "woff", "woff2", "eot")},
    **{e: "model weights" for e in ("gguf", "safetensors", "pt", "pth", "ckpt", "onnx", "npy", "npz",
                                    "pkl", "pickle", "h5", "mlmodel")},
}

MAX_EDIT_BYTES = 10 * 1024 * 1024      # edit/write refuse bigger files
STREAM_THRESHOLD = 8 * 1024 * 1024     # read_file streams bigger files instead of loading them


class BinaryFileError(Exception):
    """The file is not text; ``str(e)`` describes it (for the model)."""


@dataclass
class TextFile:
    text: str             # content with "\n" newlines when the file is pure CRLF
    encoding: str         # codec used to decode and to write back
    newline: str          # "\n" or "\r\n" (pure CRLF files only)
    mtime: float
    size: int


def human_size(n: int) -> str:
    """Compact size: 812B, 4.2K, 1.3M, 2.0G."""
    size = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            if unit == "B":
                return f"{int(size)}B"
            return f"{size:.1f}{unit}" if size < 100 else f"{size:.0f}{unit}"
        size /= 1024
    return f"{n}B"  # pragma: no cover


def binary_kind(path: Path, head: bytes) -> str | None:
    """A short description when the file looks binary, else None."""
    ext = path.suffix.lower().lstrip(".")
    if ext in BINARY_EXTENSIONS:
        return BINARY_EXTENSIONS[ext]
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF8)):
        return None
    if b"\x00" in head:
        return "binary"
    if head:
        control = sum(1 for b in head if b < 32 and b not in (9, 10, 12, 13, 27))
        if control / len(head) > 0.10:
            return "binary"
    return None


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode file content; returns (text, encoding). Tries UTF-8 (with BOM), UTF-16
    (with BOM), then CP-1250 (Polish Windows files), then Latin-1 (never fails)."""
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8):].decode("utf-8", "replace"), "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", "replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1250"), "cp1250"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def split_newlines(text: str) -> tuple[str, str]:
    """(text with LF newlines, newline style). Only pure-CRLF files are converted."""
    crlf = text.count("\r\n")
    if crlf and crlf == text.count("\n"):
        return text.replace("\r\n", "\n"), "\r\n"
    return text, "\n"


def load_text(path: Path, *, max_bytes: int = MAX_EDIT_BYTES) -> TextFile:
    """Read a whole text file. Raises FileNotFoundError, IsADirectoryError,
    BinaryFileError or ToolInputError (too big)."""
    st = path.stat()
    if stat.S_ISDIR(st.st_mode):
        raise IsADirectoryError(str(path))
    if st.st_size > max_bytes:
        raise ToolInputError(
            f"{path.name} is too large ({human_size(st.st_size)}) to load whole. "
            "Use read_file with offset/limit, or bash tools such as sed for big files.",
            summary="Plik jest za duży")
    data = path.read_bytes()
    kind = binary_kind(path, data[:8192])
    if kind:
        raise BinaryFileError(f"{path.name} is a {kind} file ({human_size(st.st_size)}), not text.")
    text, encoding = decode_bytes(data)
    text, newline = split_newlines(text)
    return TextFile(text=text, encoding=encoding, newline=newline, mtime=st.st_mtime, size=st.st_size)


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


_UMASK = _current_umask()


def save_text(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> tuple[float, str]:
    """Write ``text`` atomically (temp file + rename), keeping the file's mode and
    writing through symlinks. Returns (new mtime, encoding actually used): when the
    text cannot be represented in the original encoding it is written as UTF-8."""
    data_text = text.replace("\n", newline) if newline != "\n" else text
    try:
        data = data_text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        encoding = "utf-8"
        data = data_text.encode("utf-8")
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = 0o666 & ~_UMASK
    try:
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".lctmp")
    except OSError:
        # directory not writable but the file might be: write in place
        target.write_bytes(data)
        return target.stat().st_mtime, encoding
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target.stat().st_mtime, encoding


# --------------------------------------------------------------------------- #
# Formatting and truncation
# --------------------------------------------------------------------------- #


def pl_plural(n: int, one: str, few: str, many: str) -> str:
    """Polish plural: 1 linia, 2 linie, 5 linii, 22 linie, 12 linii."""
    n = abs(n)
    if n == 1:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def en_plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def clip_line(line: str, max_chars: int) -> str:
    if len(line) <= max_chars:
        return line
    return f"{line[:max_chars]}… [line clipped, {len(line)} chars]"


def line_number_width(last: int) -> int:
    return max(5, len(str(last)))


def number_lines(lines: list[str], start: int, max_line_chars: int) -> str:
    """``cat -n`` style block: right-aligned number, TAB, text."""
    width = line_number_width(start + len(lines) - 1)
    return "\n".join(f"{n:>{width}}\t{clip_line(l, max_line_chars)}" for n, l in enumerate(lines, start))


def omitted_marker(n_lines: int, hint: str) -> str:
    return f"[… {n_lines} {'line' if n_lines == 1 else 'lines'} omitted — {hint} …]"


def truncate_middle(text: str, max_chars: int, hint: str) -> str:
    """Keep whole lines from the head (~40 %) and tail (~60 %) of ``text`` with a
    marker telling the model how many lines were dropped and how to see them."""
    if len(text) <= max_chars:
        return text
    lines = text.split("\n")
    marker_reserve = 120 + len(hint)
    budget = max(200, max_chars - marker_reserve)
    head_budget = int(budget * 0.4)
    tail_budget = budget - head_budget
    head: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > head_budget:
            break
        head.append(line)
        used += len(line) + 1
    tail: list[str] = []
    used = 0
    for line in reversed(lines[len(head):]):
        if used + len(line) + 1 > tail_budget:
            break
        tail.append(line)
        used += len(line) + 1
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    if omitted <= 0:  # pragma: no cover - only when budgets exceed the text
        return text
    if not head and not tail:
        # a few gigantic lines: fall back to characters
        h = text[: head_budget]
        t = text[-tail_budget:]
        return f"{h}\n[… {len(text) - len(h) - len(t)} characters omitted — {hint} …]\n{t}"
    parts = []
    if head:
        parts.append("\n".join(head))
    parts.append(omitted_marker(omitted, hint))
    if tail:
        parts.append("\n".join(tail))
    return "\n".join(parts)


MAX_DIFF_LINES = 1500


def unified_diff(old: str, new: str, label: str, *, created: bool = False,
                 max_lines: int = MAX_DIFF_LINES) -> str:
    """git-style unified diff (``a/label`` → ``b/label``), capped at ``max_lines``."""
    old_lines = old.splitlines(keepends=True) if old else []
    new_lines = new.splitlines(keepends=True)
    out: list[str] = []
    for line in difflib.unified_diff(old_lines, new_lines,
                                     fromfile="/dev/null" if created else f"a/{label}",
                                     tofile=f"b/{label}", n=3):
        if line.endswith("\n"):
            out.append(line)
        else:
            out.append(line + "\n")
            out.append("\\ No newline at end of file\n")
    if len(out) > max_lines:
        dropped = len(out) - max_lines
        out = out[:max_lines] + [f"[… diff truncated: {dropped} more lines …]\n"]
    return "".join(out)


def diff_counts(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def syntax_problem(path: Path, text: str) -> str | None:
    """A one-line description of a syntax error for .py/.json files, else None."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            compile(text, str(path), "exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True)
        except SyntaxError as e:
            return f"line {e.lineno}: {e.msg}"
        except (ValueError, RecursionError, MemoryError):
            return None
        return None
    if suffix == ".json":
        if not text.strip():
            return None
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            return f"line {e.lineno}: {e.msg}"
    return None


# --------------------------------------------------------------------------- #
# Ignore rules and the project walker
# --------------------------------------------------------------------------- #

IGNORED_NAMES = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", "target",
    ".build", ".DS_Store", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".nox",
    ".svn", ".hg", ".idea", ".gradle", "Pods", "DerivedData", ".turbo", ".parcel-cache",
    ".nuxt", ".svelte-kit", ".angular", ".terraform", "coverage", ".coverage", ".eggs",
})


def _glob_body_to_regex(pat: str) -> str:
    """Translate a glob body (``*``, ``**``, ``?``, ``[...]``) into a regex fragment
    where ``*`` and ``?`` never cross ``/`` and ``**`` spans directories."""
    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                j = i + 2
                while j < n and pat[j] == "*":
                    j += 1
                if j < n and pat[j] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                    continue
                out.append(".*")
                i = j
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and pat[j] in "!^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = pat[i + 1:j]
                if body[:1] in ("!", "^"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "\\" and i + 1 < n:
            out.append(re.escape(pat[i + 1]))
            i += 2
            continue
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def expand_braces(pattern: str, limit: int = 64) -> list[str]:
    """``*.{ts,tsx}`` → ``['*.ts', '*.tsx']`` (nested braces supported)."""
    m = None
    depth = 0
    start = -1
    for i, c in enumerate(pattern):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                m = (start, i)
                break
    if m is None:
        return [pattern]
    s, e = m
    body = pattern[s + 1:e]
    options: list[str] = []
    depth = 0
    cur = ""
    for c in body:
        if c == "," and depth == 0:
            options.append(cur)
            cur = ""
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        cur += c
    options.append(cur)
    if len(options) == 1:
        return [pattern]
    out: list[str] = []
    for opt in options:
        for x in expand_braces(pattern[:s] + opt + pattern[e + 1:], limit):
            if x not in out:
                out.append(x)
            if len(out) >= limit:
                return out
    return out


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Full-match regex for a glob relative to a directory (supports ``**`` and braces)."""
    alternatives = [_glob_body_to_regex(p) for p in expand_braces(pattern)]
    return re.compile("^(?:" + "|".join(alternatives) + ")$", re.DOTALL)


@dataclass
class _IgnoreRule:
    regex: re.Pattern[str]
    negate: bool
    dir_only: bool


def _parse_gitignore(text: str) -> list[_IgnoreRule]:
    rules: list[_IgnoreRule] = []
    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.endswith("\\ "):
            line = line.rstrip()
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        if line.startswith("\\"):
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        anchored = "/" in line
        line = line.lstrip("/")
        body = _glob_body_to_regex(line)
        regex = "^" + body + "$" if anchored else "^(?:.*/)?" + body + "$"
        try:
            rules.append(_IgnoreRule(re.compile(regex, re.DOTALL), negate, dir_only))
        except re.error:
            continue
    return rules


class IgnoreRules:
    """``.gitignore`` of the project root (and of the enclosing git repo root, if
    the project is a sub-directory of a repo) plus the built-in ``IGNORED_NAMES``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(root))
        self._sets: list[tuple[str, list[_IgnoreRule]]] = []
        repo = self._git_root(self.root)
        bases = [repo] if repo is not None and repo != self.root else []
        bases.append(self.root)
        for base in bases:
            gi = base / ".gitignore"
            try:
                if gi.is_file() and gi.stat().st_size < 1_000_000:
                    rules = _parse_gitignore(gi.read_text(encoding="utf-8", errors="replace"))
                    if rules:
                        self._sets.append((str(base), rules))
            except OSError:
                continue

    @staticmethod
    def _git_root(start: Path) -> Path | None:
        p = start
        for _ in range(40):
            if (p / ".git").exists():
                return p
            if p.parent == p:
                return None
            p = p.parent
        return None

    def ignored(self, path: str | Path, is_dir: bool) -> bool:
        path_s = os.fspath(path)
        name = os.path.basename(path_s)
        if name in IGNORED_NAMES:
            return True
        result = False
        for base, rules in self._sets:
            if not path_s.startswith(base + os.sep):
                continue
            rel = path_s[len(base) + 1:].replace(os.sep, "/")
            for rule in rules:
                if rule.dir_only and not is_dir:
                    continue
                if rule.regex.match(rel):
                    result = not rule.negate
        return result


def walk_files(base: Path, rules: IgnoreRules, *, include_dirs: bool = False,
               max_entries: int = 200_000) -> Iterator[tuple[str, os.DirEntry[str]]]:
    """Depth-first, sorted walk yielding (absolute path, entry) for files (and dirs when
    ``include_dirs``) below ``base``; skips ignored entries and never follows directory
    symlinks. ``base`` itself is always walked even if it is an ignored directory."""
    stack = [str(base)]
    seen = 0
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: e.name)
        except (PermissionError, FileNotFoundError, NotADirectoryError, OSError):
            continue
        subdirs: list[str] = []
        for entry in entries:
            seen += 1
            if seen > max_entries:
                return
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if rules.ignored(entry.path, is_dir):
                continue
            if is_dir:
                subdirs.append(entry.path)
                if include_dirs:
                    yield entry.path, entry
            else:
                yield entry.path, entry
        stack.extend(reversed(subdirs))


def suggest_paths(target: Path, ctx: ToolContext, limit: int = 3) -> list[str]:
    """Paths the model probably meant when ``target`` does not exist."""
    out: list[str] = []
    parent = target.parent
    name = target.name
    try:
        if parent.is_dir():
            names = os.listdir(parent)
            for m in difflib.get_close_matches(name, names, n=limit, cutoff=0.6):
                out.append(display_path(parent / m, ctx))
    except OSError:
        pass
    root = cwd_of(ctx)
    if len(out) < limit and root.is_dir():
        lname = unicodedata.normalize("NFC", name).lower()
        rules = IgnoreRules(root)
        for path, entry in walk_files(root, rules, include_dirs=True, max_entries=30_000):
            if unicodedata.normalize("NFC", entry.name).lower() == lname:
                d = display_path(Path(path), ctx)
                if d not in out:
                    out.append(d)
                if len(out) >= limit:
                    break
    return out[:limit]


def not_found_error(target: Path, ctx: ToolContext, what: str = "File") -> ToolInputError:
    hints = suggest_paths(target, ctx)
    msg = f"{what} not found: {display_path(target, ctx)}."
    if hints:
        msg += " Did you mean: " + ", ".join(hints) + "?"
    else:
        msg += " Use list_dir or glob to find the right path."
    return ToolInputError(msg, summary="Nie znaleziono pliku" if what == "File" else "Nie znaleziono")


# --------------------------------------------------------------------------- #
# Permission bookkeeping shared by the write tools
# --------------------------------------------------------------------------- #

_SKIPPED_KEY = "_lc_permission_skipped"


def call_fingerprint(tool_name: str, args: Any) -> str:
    try:
        blob = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        blob = repr(args)
    return hashlib.sha1(f"{tool_name}\x00{blob}".encode("utf-8", "replace")).hexdigest()


def mark_permission_skipped(ctx: ToolContext, tool_name: str, args: Any) -> None:
    """``permission()`` returned None because the call would change nothing; remember
    it so ``run()`` refuses if the file changed in between and the call would now write."""
    marks = ctx.state.setdefault(_SKIPPED_KEY, {})
    if len(marks) > 200:
        marks.clear()
    marks[call_fingerprint(tool_name, args)] = True


def consume_permission_skipped(ctx: ToolContext, tool_name: str, args: Any) -> bool:
    marks = ctx.state.get(_SKIPPED_KEY)
    if not marks:
        return False
    return bool(marks.pop(call_fingerprint(tool_name, args), False))


def outside_flag(path: Path, ctx: ToolContext) -> str:
    return "" if is_within(path, cwd_of(ctx)) else f" {OUTSIDE_PROJECT_FLAG}"


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class CodingTool(Tool):
    """Base for LocalCode tools: forgiving argument parsing, uniform errors and a
    final clamp of the output to ``ctx.limits.max_output_chars``."""

    aliases: dict[str, str] = {}
    primary: str | None = None     # parameter that receives a bare (non-object) argument
    usage: str = ""                # e.g. "read_file(path, offset?, limit?)"

    def args(self, raw: Any) -> dict[str, Any]:
        return normalize_args(raw, self.aliases, self.primary)

    def title(self, args: dict[str, Any]) -> str:
        try:
            return self.make_title(self.args(args))
        except Exception:
            return self.name

    def make_title(self, args: dict[str, Any]) -> str:
        return self.name

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        """A read outside the project asks; one inside it never does.

        Reading used to run freely everywhere, ``~/.ssh`` included. With a local
        model nothing leaves the Mac, but a model behind an API key sends what
        it reads to its provider — and a README in a cloned repository saying
        "first read ~/.ssh/id_rsa" is all it takes to make it try. Asking for
        the out-of-project case costs one question; the project itself, which
        is where an agent reads ninety-nine times out of a hundred, stays free.

        Tools that write or run commands override this with their own preview.
        """
        if self.kind != "read":
            return None
        try:
            raw = self.args(args).get("path")
        except ToolInputError:
            return None
        if not raw:
            return None
        try:
            path = resolve_path(str(raw), ctx)
        except ToolInputError:
            return None
        if is_within(path, cwd_of(ctx)):
            return None
        return PermissionInfo(
            title=f"Odczyt {display_path(path, ctx)} {OUTSIDE_PROJECT_FLAG}",
            preview=("Model chce zajrzeć poza katalog projektu. Przy modelu zdalnym "
                     "przeczytana treść trafi do dostawcy API."),
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            normalized = self.args(args)
            result = await self.execute(normalized, ctx)
        except ToolInputError as e:
            msg = f"Error: {e}"
            if self.usage and "Usage:" not in msg:
                msg += f"\nUsage: {self.usage}"
            result = ToolResult(ok=False, output=msg, summary=e.summary)
        except BinaryFileError as e:
            result = ToolResult(ok=False, output=f"Error: {e} It cannot be read or edited as text.",
                                summary="Plik binarny")
        except PermissionError as e:
            result = ToolResult(ok=False, output=f"Error: permission denied: {e.filename or e}.",
                                summary="Brak uprawnień")
        except IsADirectoryError as e:
            result = ToolResult(ok=False, output=f"Error: {e.filename or e} is a directory. Use list_dir to see it.",
                                summary="To jest katalog")
        except FileNotFoundError as e:
            result = ToolResult(ok=False, output=f"Error: not found: {e.filename or e}.",
                                summary="Nie znaleziono")
        except OSError as e:
            result = ToolResult(ok=False, output=f"Error: {e.strerror or e} ({e.filename or ''}).".replace(" ()", ""),
                                summary="Błąd systemu plików")
        except Exception as e:  # never let a tool bug kill the agent turn
            log.exception("tool %s crashed", self.name)
            result = ToolResult(ok=False, output=f"Error: internal error in {self.name}: {type(e).__name__}: {e}",
                                summary="Wewnętrzny błąd narzędzia")
        return clamp_result(result, ctx.limits)

    @abc.abstractmethod
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


def clamp_result(result: ToolResult, limits: ToolLimits) -> ToolResult:
    max_chars = max(500, int(limits.max_output_chars))
    if len(result.output) > max_chars:
        result.output = truncate_middle(result.output, max_chars,
                                        "output too long; narrow the request (offset/limit, grep, a more specific path)")
    return result


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #

PATH_ALIASES = {
    "file_path": "path", "filepath": "path", "file": "path", "filename": "path", "file_name": "path",
    "target_file": "path", "target": "path", "fname": "path", "relative_path": "path",
    "absolute_path": "path", "pathname": "path", "abs_path": "path", "uri": "path",
}

_LINE_SUFFIX = re.compile(r"^(.*?):(\d+)(?:[-:](\d+))?$")


def _count_lines_bytes(path: Path) -> int:
    count = 0
    last = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last = chunk[-1:]
    if last and last != b"\n":
        count += 1
    return count


class ReadFileTool(CodingTool):
    name = "read_file"
    kind = "read"
    description = "Read a text file with line numbers. For big files use offset (first line) and limit."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the project"},
            "offset": {"type": "integer", "description": "First line to show (1-based; negative = from end)"},
            "limit": {"type": "integer", "description": "Number of lines"},
        },
        "required": ["path"],
    }
    aliases = {**PATH_ALIASES, "start": "offset", "start_line": "offset", "line": "offset",
               "from_line": "offset", "line_start": "offset", "begin": "offset", "startline": "offset",
               "lines": "limit", "count": "limit", "n": "limit", "max_lines": "limit", "num_lines": "limit",
               "length": "limit", "end": "end_line", "to_line": "end_line", "line_end": "end_line",
               "endline": "end_line", "last_line": "end_line"}
    primary = "path"
    usage = 'read_file({"path": "src/app.py", "offset": 1, "limit": 200})'

    def make_title(self, args: dict[str, Any]) -> str:
        path = short(clean_path_text(arg_str(args, "path") or "?"), 70)
        offset = arg_int(args, "offset")
        limit = arg_int(args, "limit")
        end = arg_int(args, "end_line")
        if offset is None and limit is None and end is None:
            return f"Odczyt {path}"
        start = offset if offset and offset > 0 else 1
        if limit and limit > 0:
            return f"Odczyt {path} ({start}–{start + limit - 1})"
        if end:
            return f"Odczyt {path} ({start}–{end})"
        return f"Odczyt {path} (od {start})"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = arg_str(args, "path", required=True, allow_empty=False)
        assert raw is not None
        offset = arg_int(args, "offset")
        limit = arg_int(args, "limit")
        end_line = arg_int(args, "end_line")
        path = resolve_path(raw, ctx)
        if not path.exists():
            m = _LINE_SUFFIX.match(clean_path_text(raw))
            if m and resolve_path(m.group(1), ctx).is_file():
                # "src/app.py:120" or "src/app.py:120-140"
                path = resolve_path(m.group(1), ctx)
                offset = int(m.group(2)) if offset is None else offset
                if m.group(3) and limit is None and end_line is None:
                    end_line = int(m.group(3))
            else:
                raise not_found_error(path, ctx)
        if path.is_dir():
            disp = display_path(path, ctx)
            raise ToolInputError(f"{disp} is a directory, not a file. Use list_dir(path='{disp}') to see its contents.",
                                 summary="To jest katalog")
        return await _run_blocking(self._read, path, offset, limit, end_line, ctx)

    def _read(self, path: Path, offset: int | None, limit: int | None, end_line: int | None,
              ctx: ToolContext) -> ToolResult:
        limits = ctx.limits
        disp = display_path(path, ctx)
        st = path.stat()
        with open(path, "rb") as fh:
            head = fh.read(8192)
        kind = binary_kind(path, head)
        if kind:
            if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic", ".bmp", ".tiff")):
                # A picture the user attached is already in the conversation;
                # reading its bytes can never show it.
                return ToolResult(ok=False,
                                  output=(f"{disp} is an image ({human_size(st.st_size)}). If the user attached it, "
                                          "you can already see it in their message — describe it from there. "
                                          "read_file cannot show images."),
                                  summary=f"Obraz ({human_size(st.st_size)})")
            return ToolResult(ok=False,
                              output=f"{disp} is a {kind} file ({human_size(st.st_size)}); it cannot be shown as text.",
                              summary=f"Plik binarny ({human_size(st.st_size)})")
        default_limit = max(1, int(limits.read_default_lines))
        max_line_chars = max(200, min(2000, int(limits.max_output_chars) // 8))
        streamed = st.st_size > STREAM_THRESHOLD
        encoding = "utf-8"
        if streamed:
            total = _count_lines_bytes(path)
        else:
            tf = load_text(path, max_bytes=STREAM_THRESHOLD + 1)
            encoding = tf.encoding
            all_lines = tf.text.split("\n")
            if all_lines and all_lines[-1] == "":
                all_lines.pop()
            total = len(all_lines)
        ctx.read_files[file_key(path)] = st.st_mtime

        if total == 0:
            return ToolResult(ok=True, output=f"File: {disp} (empty file)", summary="Pusty plik")

        start = 1 if offset is None or offset == 0 else offset
        if start < 0:
            start = max(1, total + start + 1)
        if start > total:
            return ToolResult(ok=False,
                              output=f"Error: offset {start} is past the end of {disp}, which has {total} lines. "
                                     f"Use an offset between 1 and {total}.",
                              summary="Poza końcem pliku")
        if limit is None or limit <= 0:
            limit = end_line - start + 1 if end_line and end_line >= start else default_limit
        end = min(total, start + limit - 1)

        if streamed:
            window = self._stream_window(path, start, end, head)
        else:
            window = all_lines[start - 1:end]

        # respect the character budget: stop early at a whole line
        budget = max(400, int(limits.max_output_chars) - 400)
        width = line_number_width(end)
        rendered: list[str] = []
        used = 0
        clipped = False
        for n, line in enumerate(window, start):
            if len(line) > max_line_chars:
                clipped = True
            text = f"{n:>{width}}\t{clip_line(line, max_line_chars)}"
            if rendered and used + len(text) + 1 > budget:
                break
            rendered.append(text)
            used += len(text) + 1
        shown_end = start + len(rendered) - 1

        whole = start == 1 and shown_end == total
        header = f"File: {disp} ({total} lines)"
        if not whole:
            header += f" — showing {start}–{shown_end}"
        if encoding not in ("utf-8", "utf-8-sig"):
            header += f" [encoding: {encoding}]"
        parts = [header, "\n".join(rendered)]
        if shown_end < total:
            remaining = total - shown_end
            parts.append(f"[{remaining} more {'line' if remaining == 1 else 'lines'} — "
                         f"call read_file with offset={shown_end + 1} to continue]")
        if clipped:
            parts.append(f"[lines longer than {max_line_chars} characters were clipped; use grep or bash to see them]")
        summary = f"{total} {pl_plural(total, 'linia', 'linie', 'linii')}"
        if not whole:
            summary += f" · pokazano {start}–{shown_end}"
        return ToolResult(ok=True, output="\n".join(parts), summary=summary)

    @staticmethod
    def _stream_window(path: Path, start: int, end: int, head: bytes) -> list[str]:
        try:
            head.decode("utf-8")
            enc = "utf-8"
        except UnicodeDecodeError as e:
            enc = "utf-8" if e.start > len(head) - 4 else "cp1250"
        out: list[str] = []
        with open(path, "r", encoding=enc, errors="replace", newline="") as fh:
            for n, line in enumerate(fh, 1):
                if n < start:
                    continue
                if n > end:
                    break
                out.append(line.rstrip("\r\n"))
        return out


async def _run_blocking(fn: Any, *args: Any) -> Any:
    """Run blocking file IO in a worker thread (keeps the event loop responsive)."""
    return await asyncio.to_thread(fn, *args)


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #

CONTENT_ALIASES = {"contents": "content", "text": "content", "data": "content", "code": "content",
                   "file_content": "content", "file_text": "content", "body": "content", "source": "content",
                   "new_content": "content", "string": "content", "value": "content"}

# "... rest of the code ...", "// existing code unchanged", "# (keep the rest)" — models
# write these when they mean "unchanged", which would silently delete real code.
_PH_COMMENT = re.compile(r"^(?:#+|//+|/\*+|\*|<!--|--|;+|%|\{/\*)\s*")
_PH_PHRASE = re.compile(
    r"\b(?:rest\s+of\s+(?:the\s+)?(?:code|file|content|implementation|class|function|methods?|module|component)"
    r"|(?:existing|previous|remaining|original|other|unchanged)\s+(?:code|content|methods?|functions?|"
    r"implementation|logic|imports?|tests?|lines?|fields?|properties|styles?|rules?)"
    r"|keep\s+(?:the\s+)?(?:rest|existing)|same\s+as\s+before)\b",
    re.IGNORECASE)
_PH_UNCHANGED = re.compile(r"\b(?:unchanged|as\s+before|goes\s+here|remains?|stays?|omitted|here)\b",
                           re.IGNORECASE)


def find_placeholder(text: str) -> str | None:
    """The first line that looks like an "unchanged code here" placeholder, else None."""
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) > 120:
            continue
        m = _PH_COMMENT.match(s)
        body = s[m.end():] if m else s
        body = re.sub(r"(?:\*/|-->|\*/\})\s*$", "", body).strip()
        if not _PH_PHRASE.search(body):
            continue
        has_ellipsis = "..." in body or "…" in body
        if m and (has_ellipsis or _PH_UNCHANGED.search(body)):
            return s
        if not m and body.startswith(("...", "…")):
            return s
    return None


@dataclass
class _WritePlan:
    path: Path
    old_text: str | None
    new_text: str
    encoding: str
    newline: str
    diff: str


class WriteFileTool(CodingTool):
    name = "write_file"
    kind = "write"
    description = "Create a file or replace ALL of its content. For small changes to existing files use edit_file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the project"},
            "content": {"type": "string", "description": "The complete file content"},
        },
        "required": ["path", "content"],
    }
    aliases = {**PATH_ALIASES, **CONTENT_ALIASES}
    usage = 'write_file({"path": "src/app.py", "content": "<complete file content>"})'

    def make_title(self, args: dict[str, Any]) -> str:
        return f"Zapis {short(clean_path_text(arg_str(args, 'path') or '?'), 70)}"

    def _plan(self, args: dict[str, Any], ctx: ToolContext) -> _WritePlan | None:
        """What the write would do; None when it would change nothing."""
        raw = arg_str(args, "path", required=True, allow_empty=False)
        content = arg_str(args, "content")
        if content is None:
            raise ToolInputError("Missing required argument 'content' (the complete new file content).")
        assert raw is not None
        path = resolve_path(raw, ctx)
        disp = display_path(path, ctx)
        if path.is_dir():
            raise ToolInputError(f"{disp} is a directory. Give a file path such as '{disp}/new_file.py'.",
                                 summary="To jest katalog")
        new_text = content.replace("\r\n", "\n")
        old_text: str | None = None
        encoding = "utf-8"
        newline = "\r\n" if "\r\n" in content and content.count("\r\n") == content.count("\n") else "\n"
        if path.exists():
            tf = load_text(path)
            old_text, encoding, newline = tf.text, tf.encoding, tf.newline
            placeholder = find_placeholder(new_text)
            if placeholder and len(new_text) < len(old_text) and not find_placeholder(old_text):
                raise ToolInputError(
                    f"content contains the placeholder {placeholder!r}. write_file replaces the WHOLE file, "
                    "so that would delete the existing code. Use edit_file to change part of the file, "
                    "or send the complete content.",
                    summary="Treść zawiera zaślepkę zamiast kodu")
            if old_text.endswith("\n") and new_text and not new_text.endswith("\n"):
                new_text += "\n"
            if new_text == old_text:
                return None
        elif new_text and not new_text.endswith("\n"):
            new_text += "\n"
        diff = unified_diff(old_text or "", new_text, disp, created=old_text is None)
        return _WritePlan(path=path, old_text=old_text, new_text=new_text, encoding=encoding,
                          newline=newline, diff=diff)

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        try:
            norm = self.args(args)
        except ToolInputError:
            return None  # run() will report the malformed call without touching anything
        try:
            plan = await _run_blocking(self._plan, norm, ctx)
        except (ToolInputError, BinaryFileError, OSError):
            plan = None
        if plan is None:
            mark_permission_skipped(ctx, self.name, norm)
            return None
        disp = display_path(plan.path, ctx)
        title = f"Zapis {disp}" + (" (nowy plik)" if plan.old_text is None else "") + outside_flag(plan.path, ctx)
        return PermissionInfo(title=title, preview=plan.diff)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        skipped = consume_permission_skipped(ctx, self.name, args)
        plan = await _run_blocking(self._plan, args, ctx)
        path = resolve_path(arg_str(args, "path") or "", ctx)
        disp = display_path(path, ctx)
        if plan is None:
            ctx.read_files[file_key(path)] = path.stat().st_mtime
            return ToolResult(ok=True, output=f"No changes: {disp} already has exactly this content.",
                              summary="Bez zmian")
        if skipped:
            return ToolResult(ok=False,
                              output=f"Error: {disp} changed while this call was being checked. Call write_file again.",
                              summary="Plik zmienił się w międzyczasie")
        before_problem = syntax_problem(plan.path, plan.old_text) if plan.old_text is not None else None
        mtime, used_encoding = await _run_blocking(
            lambda: save_text(plan.path, plan.new_text, encoding=plan.encoding, newline=plan.newline))
        ctx.read_files[file_key(plan.path)] = mtime
        n_lines = plan.new_text.count("\n") + (0 if plan.new_text.endswith("\n") or not plan.new_text else 1)
        added, removed = diff_counts(plan.diff)
        if plan.old_text is None:
            output = f"Created {disp} ({en_plural(n_lines, 'line')})."
            summary = f"Utworzono ({n_lines} {pl_plural(n_lines, 'linia', 'linie', 'linii')})"
        else:
            output = f"Updated {disp}: now {en_plural(n_lines, 'line')} (+{added} −{removed})."
            summary = f"Zapisano (+{added} −{removed})"
        if used_encoding != plan.encoding:
            output += f" Note: saved as UTF-8 (the text did not fit the original {plan.encoding} encoding)."
        problem = syntax_problem(plan.path, plan.new_text)
        if problem and not before_problem:
            output += f"\nWarning: the file now has a syntax error at {problem}. Fix it with edit_file."
        return ToolResult(ok=True, output=output, summary=summary, diff=plan.diff)


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #


class ListDirTool(CodingTool):
    name = "list_dir"
    kind = "read"
    description = "List a directory as a tree with file sizes (skips .git, node_modules and ignored files)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory (default: project root)"},
            "depth": {"type": "integer", "description": "Levels to show (default 2)"},
        },
    }
    aliases = {**PATH_ALIASES, "dir": "path", "directory": "path", "dir_path": "path", "folder": "path",
               "dirname": "path", "directory_path": "path", "max_depth": "depth", "levels": "depth",
               "level": "depth", "recursive_depth": "depth"}
    primary = "path"
    usage = 'list_dir({"path": "src", "depth": 2})'

    MAX_DEPTH = 6

    def make_title(self, args: dict[str, Any]) -> str:
        path = clean_path_text(arg_str(args, "path") or ".") or "."
        return f"Lista plików: {short(path, 70)}"

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = arg_str(args, "path") or "."
        if args.get("recursive") is not None and args.get("depth") is None:
            depth = self.MAX_DEPTH if arg_bool(args, "recursive") else 1
        else:
            depth = arg_int(args, "depth", default=2, minimum=1, maximum=self.MAX_DEPTH) or 2
        path = resolve_path(raw if raw.strip() else ".", ctx)
        if not path.exists():
            raise not_found_error(path, ctx, what="Directory")
        if not path.is_dir():
            disp = display_path(path, ctx)
            size = human_size(path.stat().st_size)
            raise ToolInputError(f"{disp} is a file ({size}), not a directory. Use read_file(path='{disp}') to read it.",
                                 summary="To jest plik")
        return await _run_blocking(self._list, path, depth, ctx)

    def _list(self, root: Path, depth: int, ctx: ToolContext) -> ToolResult:
        rules = IgnoreRules(cwd_of(ctx))
        budget = max(20, int(ctx.limits.list_max_entries))
        per_dir_cap = max(15, budget // 3)

        # breadth-first selection so shallow levels always fit
        children: dict[str, list[tuple[str, bool, int, str]]] = {}   # dir -> [(name, is_dir, size, link)]
        more: dict[str, int] = {}
        counts: dict[str, int] = {}                                   # unexpanded dir -> entry count
        skipped_top: list[str] = []
        shown = 0
        frontier = [(str(root), 1)]
        truncated_budget = False
        while frontier:
            next_frontier: list[tuple[str, int]] = []
            for d, level in frontier:
                entries = self._scan(d, rules, skipped_top if d == str(root) else None)
                if entries is None:
                    continue
                keep: list[tuple[str, bool, int, str]] = []
                for item in entries:
                    if shown >= budget:
                        truncated_budget = True
                        break
                    if len(keep) >= per_dir_cap:
                        break
                    keep.append(item)
                    shown += 1
                children[d] = keep
                if len(entries) > len(keep):
                    more[d] = len(entries) - len(keep)
                for name, is_dir, _size, link in keep:
                    if is_dir and not link:
                        sub = os.path.join(d, name)
                        if level < depth:
                            next_frontier.append((sub, level + 1))
                        else:
                            counts[sub] = self._count(sub, rules)
            frontier = next_frontier

        lines: list[str] = []

        def render(d: str, indent: int) -> None:
            pad = "  " * indent
            for name, is_dir, size, link in children.get(d, []):
                sub = os.path.join(d, name)
                if link:
                    lines.append(f"{pad}{name}{'/' if is_dir else ''} -> {link}")
                elif is_dir:
                    if sub in children:
                        lines.append(f"{pad}{name}/")
                        render(sub, indent + 1)
                    else:
                        n = counts.get(sub)
                        label = "" if n is None else (" (empty)" if n == 0 else f" ({n} {'entry' if n == 1 else 'entries'})")
                        lines.append(f"{pad}{name}/{label}")
                else:
                    lines.append(f"{pad}{name} ({human_size(size)})")
            if d in more:
                lines.append(f"{pad}… {more[d]} more {'entry' if more[d] == 1 else 'entries'}")

        render(str(root), 0)
        disp = display_path(root, ctx)
        header = f"Directory: {disp}" + ("" if depth == 2 else f" (depth {depth})")
        parts = [header]
        parts.append("\n".join(lines) if lines else "(empty directory)")
        if skipped_top:
            parts.append("[not listed: " + ", ".join(sorted(skipped_top)[:12]) + "]")
        if truncated_budget:
            parts.append("[listing truncated — list a subdirectory or use glob to find specific files]")
        n_entries = shown
        summary = f"{n_entries} {pl_plural(n_entries, 'pozycja', 'pozycje', 'pozycji')}"
        return ToolResult(ok=True, output="\n".join(parts), summary=summary)

    @staticmethod
    def _scan(d: str, rules: IgnoreRules, skipped: list[str] | None) -> list[tuple[str, bool, int, str]] | None:
        try:
            with os.scandir(d) as it:
                raw_entries = list(it)
        except OSError:
            return None
        items: list[tuple[str, bool, int, str]] = []
        for e in raw_entries:
            try:
                is_link = e.is_symlink()
                is_dir = e.is_dir(follow_symlinks=True)
            except OSError:
                continue
            if rules.ignored(e.path, is_dir):
                if skipped is not None and e.name not in (".DS_Store",):
                    skipped.append(e.name + ("/" if is_dir else ""))
                continue
            link = ""
            if is_link:
                try:
                    link = os.readlink(e.path)
                except OSError:
                    link = "?"
            size = 0
            if not is_dir:
                try:
                    size = e.stat(follow_symlinks=True).st_size
                except OSError:
                    size = 0
            items.append((e.name, is_dir, size, link))
        items.sort(key=lambda t: (not t[1], t[0].lower()))
        return items

    @staticmethod
    def _count(d: str, rules: IgnoreRules) -> int | None:
        try:
            with os.scandir(d) as it:
                n = 0
                for e in it:
                    try:
                        if not rules.ignored(e.path, e.is_dir(follow_symlinks=False)):
                            n += 1
                    except OSError:
                        continue
                    if n > 9999:
                        break
                return n
        except OSError:
            return None
