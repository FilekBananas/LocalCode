"""Per-language coding notes injected into the system prompt.

Unlike :mod:`localcode.agent.skills` — an on-demand shelf the model opens with
the ``skill`` tool — these are short, language-specific guidance texts the agent
*injects straight into the system prompt* for the languages a project actually
uses, so a small local model gets the idioms and the exact build/test commands
without having to ask for them or guess.

Two sizes, chosen by the model, never by the user:

* ``mini`` — for local models (small windows). A few hundred tokens: the most
  load-bearing idioms and the canonical build/test command. Small windows pay
  for the system prompt on every request, so these stay terse.
* ``max`` — for remote models (large windows). Thorough: project layout,
  dependency management, typing, error handling, testing conventions, lint and
  the common pitfalls.

The texts live next to this file as ``<lang>.mini.md`` and ``<lang>.max.md`` and
ship in the wheel (``pyproject.toml`` package-data). Everything is English
because every text the model reads from us is English.

Selection (:func:`detect_languages`) looks at the project folder — file
extensions and manifest files (``pyproject.toml``, ``package.json``,
``go.mod`` …) — and at files named in the task text, then keeps the most-used
few (``cap``). :func:`build_section` renders the chosen notes as one labelled
block for :func:`localcode.agent.prompts.build_system_prompt`.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Sequence

__all__ = [
    "LANGS",
    "LANG_LABELS",
    "detect_languages",
    "load_note",
    "build_section",
]

_HERE = Path(__file__).resolve().parent

#: Canonical language ids, in a stable order (used to break ties deterministically).
LANGS: tuple[str, ...] = (
    "python", "typescript", "javascript", "swift", "go", "rust", "java",
    "kotlin", "cpp", "c", "csharp", "php", "ruby", "bash", "sql", "htmlcss",
)

#: Human-readable headings for the prompt section.
LANG_LABELS: dict[str, str] = {
    "python": "Python",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "swift": "Swift",
    "go": "Go",
    "rust": "Rust",
    "java": "Java",
    "kotlin": "Kotlin",
    "cpp": "C++",
    "c": "C",
    "csharp": "C#",
    "php": "PHP",
    "ruby": "Ruby",
    "bash": "Bash / shell",
    "sql": "SQL",
    "htmlcss": "HTML / CSS",
}

#: File extension → language id.
_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".swift": "swift",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".c++": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".c": "c", ".h": "c",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql",
    ".html": "htmlcss", ".htm": "htmlcss", ".css": "htmlcss",
    ".scss": "htmlcss", ".sass": "htmlcss", ".less": "htmlcss",
}

#: Manifest / config filename → language id. A manifest is strong evidence that
#: a project *is* that language even when few source files are present yet.
_MANIFEST: dict[str, str] = {
    "pyproject.toml": "python", "setup.py": "python", "setup.cfg": "python",
    "requirements.txt": "python", "pipfile": "python",
    "tsconfig.json": "typescript",
    "package.json": "javascript",
    "package.swift": "swift",
    "go.mod": "go",
    "cargo.toml": "rust",
    "pom.xml": "java", "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "composer.json": "php",
    "gemfile": "ruby",
}

#: Weight a manifest adds, so a project with a manifest but few files still wins
#: over an incidental stray file of another language.
_MANIFEST_WEIGHT = 50

#: Directories never worth scanning for language detection.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", ".env",
    "__pycache__", "dist", "build", "target", ".build", "DerivedData",
    "vendor", ".tox", ".mypy_cache", ".pytest_cache", ".idea", ".gradle",
    "Pods", ".next", ".nuxt", "out", "coverage", "site-packages",
})

#: How much of a tree to walk before stopping — detection must be cheap.
_MAX_FILES = 2000
_MAX_DIRS = 400

_FILE_TOKEN = re.compile(r"[\w./@+-]+\.[A-Za-z0-9]{1,5}\b")


def _scan_counts(root: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen_files = 0
    seen_dirs = 0
    try:
        walker = os.walk(root)
    except OSError:
        return counts
    for current, dirs, files in walker:
        seen_dirs += 1
        if seen_dirs > _MAX_DIRS or seen_files > _MAX_FILES:
            break
        # prune noise in place so os.walk does not descend into it
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            seen_files += 1
            if seen_files > _MAX_FILES:
                break
            low = name.lower()
            manifest = _MANIFEST.get(low)
            if manifest is not None:
                counts[manifest] += _MANIFEST_WEIGHT
            # *.csproj / *.sln are per-project names, not fixed → match by suffix
            if low.endswith((".csproj", ".sln", ".fsproj")):
                counts["csharp"] += _MANIFEST_WEIGHT
            ext = os.path.splitext(low)[1]
            lang = _EXT.get(ext)
            if lang is not None:
                counts[lang] += 1
    return counts


def _text_counts(text: str) -> Counter[str]:
    """Languages implied by filenames mentioned in the task text."""
    counts: Counter[str] = Counter()
    if not text:
        return counts
    for token in _FILE_TOKEN.findall(text):
        low = token.lower()
        base = os.path.basename(low)
        manifest = _MANIFEST.get(base)
        if manifest is not None:
            counts[manifest] += _MANIFEST_WEIGHT
        ext = os.path.splitext(low)[1]
        lang = _EXT.get(ext)
        if lang is not None:
            counts[lang] += 5  # a named file is a strong hint, but weaker than a manifest
    return counts


def detect_languages(
    cwd: str | os.PathLike[str],
    *,
    extra_text: str = "",
    cap: int = 3,
) -> list[str]:
    """The project's main languages, most-used first, at most ``cap``.

    Combines a capped scan of the project folder (extensions + manifest files)
    with any languages named by files in ``extra_text`` (the task). Ties break
    on :data:`LANGS` order so the result is deterministic.
    """
    if cap <= 0:
        return []
    counts = _scan_counts(Path(cwd).expanduser())
    counts.update(_text_counts(extra_text))
    if not counts:
        return []
    order = {lang: i for i, lang in enumerate(LANGS)}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], order.get(kv[0], 99)))
    return [lang for lang, weight in ranked if weight > 0][:cap]


@lru_cache(maxsize=None)
def load_note(lang: str, *, strong: bool) -> str | None:
    """The ``mini`` (local) or ``max`` (remote) note for ``lang``, or ``None``."""
    suffix = "max" if strong else "mini"
    path = _HERE / f"{lang}.{suffix}.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        # fall back to the other size rather than drop the language entirely
        other = _HERE / f"{lang}.{'mini' if strong else 'max'}.md"
        try:
            text = other.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return text or None


def build_section(
    cwd: str | os.PathLike[str],
    *,
    strong: bool,
    extra_text: str = "",
    cap: int = 3,
    langs: Sequence[str] | None = None,
    char_budget: int | None = None,
) -> str | None:
    """A labelled ``# Language notes`` block for the detected languages.

    ``strong`` picks the size: a remote/strong model gets ``max``, a local one
    ``mini``. ``char_budget`` (default: unlimited for ``max``, 2000 for ``mini``)
    stops adding languages once the block would exceed it — a small window
    cannot afford three long notes.
    """
    chosen = list(langs) if langs is not None else detect_languages(cwd, extra_text=extra_text, cap=cap)
    if not chosen:
        return None
    if char_budget is None:
        char_budget = 0 if strong else 2000
    blocks: list[str] = []
    used = 0
    for lang in chosen:
        note = load_note(lang, strong=strong)
        if not note:
            continue
        block = f"## {LANG_LABELS.get(lang, lang)}\n{note}"
        if char_budget and blocks and used + len(block) > char_budget:
            break
        blocks.append(block)
        used += len(block)
    if not blocks:
        return None
    intro = ("Notes for the language(s) this project uses. Follow them alongside the "
             "project's own conventions.")
    return "# Language notes\n" + intro + "\n\n" + "\n\n".join(blocks)
