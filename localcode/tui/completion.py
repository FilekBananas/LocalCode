"""Input completion and ``@file`` expansion.

Two things happen to what the user types:

* **completion** — ``/`` at the start of a line offers slash commands (and the
  values of the ones that take a fixed argument), ``@`` anywhere offers paths
  from the project;
* **expansion** — ``@src/app.py`` stays visible in the transcript, but the text
  that reaches the model gets the file appended inside a ``<file>`` block,
  truncated to a slice of the context window so one careless ``@`` cannot blow
  up a 4k-token model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from localcode.tui.commands import COMMANDS, find_command

__all__ = [
    "AT_TOKEN",
    "LocalCodeCompleter",
    "at_budget_chars",
    "expand_at_references",
    "list_paths",
]

#: ``@some/path`` — stops at whitespace; a trailing ``,``/``.``/``)`` is not part of it.
from localcode.tui.images_in import IMAGE_EXTENSIONS  # noqa: E402

AT_TOKEN = re.compile(r"(?<![\w@])@([^\s@\"'`]+)")

#: Directories never offered by the completer nor walked for expansion.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", ".build", "DerivedData",
    ".tox", ".idea", ".gradle", "target",
})
#: Files bigger than this are never inlined whole.
MAX_INLINE_BYTES = 512 * 1024
#: Fraction of the context window one ``@`` expansion may take.
AT_CONTEXT_SHARE = 0.25
#: Rough characters per token for local tokenizers (used only for the budget).
CHARS_PER_TOKEN = 3.5
DEFAULT_AT_CHARS = 6000


def at_budget_chars(context_limit: int | None) -> int:
    """How many characters of inlined files fit in the model's window."""
    if not context_limit:
        return DEFAULT_AT_CHARS
    return max(1200, int(context_limit * AT_CONTEXT_SHARE * CHARS_PER_TOKEN))


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #


def list_paths(prefix: str, cwd: Path, *, limit: int = 60) -> list[tuple[str, bool]]:
    """``(relative path, is_dir)`` candidates for ``@<prefix>``, directories first."""
    prefix = prefix.replace("\\", "")
    head, _, tail = prefix.rpartition("/")
    base = (cwd / head).expanduser() if head else cwd
    try:
        entries = sorted(os.scandir(base), key=lambda e: (not e.is_dir(), e.name.lower()))
    except (OSError, ValueError):
        return []
    out: list[tuple[str, bool]] = []
    lowered = tail.lower()
    for entry in entries:
        name = entry.name
        if name.startswith(".") and not tail.startswith("."):
            continue
        if name in SKIP_DIRS:
            continue
        if lowered and not name.lower().startswith(lowered):
            continue
        try:
            is_dir = entry.is_dir()
        except OSError:  # pragma: no cover - race with a deleted file
            continue
        out.append((f"{head}/{name}" if head else name, is_dir))
        if len(out) >= limit:
            break
    return out


class LocalCodeCompleter(Completer):
    """Slash commands at the start of a line, project paths after ``@``."""

    def __init__(self, cwd: Path, *, chat_mode: bool = False) -> None:
        self.cwd = Path(cwd)
        self.chat_mode = chat_mode

    def get_completions(self, document: Document, complete_event: object) -> Iterator[Completion]:
        text = document.text_before_cursor
        line = text.split("\n")[-1]
        at_start = line.lstrip()
        if at_start.startswith("/") and "\n" not in text:
            yield from self._slash(at_start)
            return
        yield from self._paths(text)

    # -- slash ---------------------------------------------------------------
    def _slash(self, line: str) -> Iterator[Completion]:
        head, space, rest = line[1:].partition(" ")
        if not space:
            for command in COMMANDS:
                if self.chat_mode and not command.in_chat:
                    continue
                for name in (command.name, *command.aliases):
                    if name.startswith(head.lower()):
                        yield Completion(f"/{name}", start_position=-len(line),
                                         display=command.display, display_meta=command.help)
                        break
            return
        command = find_command(head)
        if command is None or not command.values:
            return
        word = rest.rsplit(" ", 1)[-1]
        for value in command.values:
            if value.startswith(word):
                yield Completion(value, start_position=-len(word))

    # -- paths ---------------------------------------------------------------
    def _paths(self, text: str) -> Iterator[Completion]:
        match = re.search(r"@([^\s@\"'`]*)$", text)
        if match is None:
            return
        prefix = match.group(1)
        for path, is_dir in list_paths(prefix, self.cwd):
            yield Completion(
                path + ("/" if is_dir else ""),
                start_position=-len(prefix),
                display=path + ("/" if is_dir else ""),
                display_meta="katalog" if is_dir else _meta_for(self.cwd / path),
            )


def _meta_for(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB".replace(".", ",")
    return f"{size / (1024 * 1024):.1f} MB".replace(".", ",")


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #


@dataclass
class Attachment:
    """One file (or directory listing) inlined into a user message."""

    path: str
    lines: int
    chars: int
    truncated: bool = False
    block: str = ""
    is_dir: bool = False

    @property
    def summary(self) -> str:
        suffix = " (skrócone)" if self.truncated else ""
        unit = "plików" if self.is_dir else "linii"
        return f"{self.path} — {self.lines} {unit}{suffix}"


def expand_at_references(
    text: str,
    cwd: str | os.PathLike[str],
    *,
    max_chars: int = DEFAULT_AT_CHARS,
    max_files: int = 8,
) -> tuple[str, list[Attachment], list[str]]:
    """Inline every ``@path`` of ``text``.

    Returns ``(text for the model, attachments, warnings)``. The original text
    is kept verbatim; the files are appended below it inside ``<file …>``
    blocks, together never longer than ``max_chars``.
    """
    root = Path(cwd).expanduser()
    seen: list[str] = []
    attachments: list[Attachment] = []
    warnings: list[str] = []
    budget = max(0, int(max_chars))
    for match in AT_TOKEN.finditer(text or ""):
        raw = match.group(1).rstrip(".,;:)»\"'")
        if not raw or raw in seen:
            continue
        seen.append(raw)
        if len(attachments) >= max_files:
            warnings.append(f"Pominięto @{raw}: maksymalnie {max_files} plików w jednej wiadomości.")
            continue
        path = (root / raw).expanduser() if not os.path.isabs(raw) else Path(raw)
        if path.is_dir():
            listing, count = _dir_listing(path, root)
            if listing and budget > 0:
                listing = listing[:budget]
                budget -= len(listing)
                attachments.append(Attachment(raw, count, len(listing), block=listing, is_dir=True))
            continue
        if not path.is_file():
            warnings.append(f"Nie znaleziono pliku @{raw}.")
            continue
        if raw.lower().endswith(IMAGE_EXTENSIONS):
            # A picture goes to a vision model as an image (tui/images_in.py),
            # never into the prompt as text.
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            warnings.append(f"Nie można odczytać @{raw}: {exc.strerror or exc}.")
            continue
        if size > MAX_INLINE_BYTES:
            warnings.append(f"Plik @{raw} jest za duży ({size // 1024} KB) — użyj narzędzia read_file.")
            continue
        if budget <= 0:
            warnings.append(f"Pominięto @{raw}: brak miejsca w kontekście modelu.")
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            warnings.append(f"Nie można odczytać @{raw}: {exc.strerror or exc}.")
            continue
        if "\x00" in content[:4096]:
            warnings.append(f"Pominięto @{raw}: to plik binarny.")
            continue
        share = min(len(content), budget)
        truncated = share < len(content)
        body = content[:share]
        if truncated:
            body = body.rsplit("\n", 1)[0] + "\n… [plik skrócony do budżetu kontekstu]"
        budget -= len(body)
        attachments.append(Attachment(raw, body.count("\n") + 1, len(body), truncated,
                                      block=f'<file path="{raw}">\n{body}\n</file>'))
    blocks = [a.block for a in attachments if a.block]
    if not blocks:
        return text, attachments, warnings
    return text.rstrip() + "\n\n" + "\n\n".join(blocks), attachments, warnings


def _dir_listing(path: Path, root: Path, *, limit: int = 120) -> tuple[str, int]:
    """A short recursive listing used when ``@`` points at a directory."""
    rows: list[str] = []
    for current, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            if name.startswith("."):
                continue
            full = Path(current) / name
            try:
                rows.append(str(full.relative_to(root)))
            except ValueError:
                rows.append(str(full))
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    if not rows:
        return "", 0
    try:
        label = str(path.relative_to(root))
    except ValueError:
        label = str(path)
    return f'<dir path="{label}">\n' + "\n".join(rows) + "\n</dir>", len(rows)


def summarize_attachments(attachments: Sequence[Attachment]) -> str:
    """One Polish line listing what was attached (shown under the user message)."""
    if not attachments:
        return ""
    names = ", ".join(a.summary for a in attachments)
    return f"Dołączono: {names}"


def iter_at_paths(text: str) -> Iterable[str]:
    """Every ``@path`` in ``text`` (used by tests and by the transcript)."""
    for match in AT_TOKEN.finditer(text or ""):
        yield match.group(1).rstrip(".,;:)»\"'")
