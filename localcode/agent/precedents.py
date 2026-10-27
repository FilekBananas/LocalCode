"""The user's own answers on permission cards, remembered per project.

When the user answers a permission card for a shell command or for a call that
reaches outside the project — in any mode — the answer is kept as a
*precedent*: ``approved: git push (3×)``, ``declined: rm (1×)``. Auto mode's
two gates then read them: the reviewer (``agent/autoreview.py``), which has the
conversation already, gets only the precedents that match the call; the local
guard (``agent/guard.py``), which sees no conversation at all, gets a short
history — the matching ones first, then the most recent others. That is how
"I said yes to this before" makes them more willing next time, and "I said no"
makes them stricter. A precedent never runs anything by itself: a call still
needs both gates' TAK, and calls that need a human
(``PermissionPolicy.needs_human``) get no precedents at all.

What is stored is never the model's text, only keys our own code computed:

* a shell command → one key per simple command that is not read-only, its
  :func:`~localcode.agent.permissions.command_prefix` (``git push``,
  ``npm run build``, ``python -m pytest``). A command the parser does not fully
  understand — substitution, subshell, redirection to a file, an unsafe
  variable assignment, ``eval``/``source``, code passed inline (``python -c``),
  a parse error — gives no key at all and is never recorded;
* a file tool outside the project → ``"<tool> <directory>"``, the directory
  resolved by us (real path, ``~`` form), e.g. ``write_file ~/notes``.

Keys pass a strict check (letters incl. Polish, digits, a few path characters;
no quotes, backticks, angle brackets, ``$`` or newlines; at most
:data:`KEY_LIMIT` characters) — anything else is skipped, so the lines put in
front of the gates are short, plain data.

The store is ``<LOCALCODE_HOME>/auto_precedents.json`` — the user's home, never
the project folder: a cloned repository must not be able to ship "the user
approved curl". Projects are keyed by their real path; each keeps at most
:data:`MAX_PER_PROJECT` entries (the least recently used go first); the file is
written atomically with mode 0600 (``config.write_private_json``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, NamedTuple, Sequence

from localcode import config as lc_config
from localcode.agent.permissions import (
    _DUP_REDIRECTS,
    _INTERPRETERS,
    _OUT_REDIRECTS,
    _SAFE_ASSIGNMENTS,
    _command_name,
    _simple_is_read_only,
    _strip_wrappers,
    command_prefix,
    parse_shell,
)

log = logging.getLogger(__name__)

__all__ = [
    "APPROVED",
    "DECLINED",
    "GUARD_LIMIT",
    "KEY_LIMIT",
    "MAX_PER_PROJECT",
    "REVIEWER_LIMIT",
    "STORE_NAME",
    "Key",
    "Precedent",
    "call_keys",
    "clear",
    "guard_lines",
    "listing",
    "load",
    "matching",
    "path_key",
    "project_id",
    "record",
    "reviewer_lines",
    "shell_keys",
    "store_path",
]

STORE_NAME = "auto_precedents.json"
APPROVED, DECLINED = "approved", "declined"
#: Entries one project keeps; the least recently used go first.
MAX_PER_PROJECT = 100
#: Projects the file keeps; the least recently used go first.
MAX_PROJECTS = 200
#: Matching precedents the reviewer is shown (it has the conversation too).
REVIEWER_LIMIT = 5
#: Lines of history the guard is shown: matching first, then the most recent.
GUARD_LIMIT = 8
#: Longest key that is kept; a longer one is skipped, never shortened (a
#: shorter directory would match more than the user approved).
KEY_LIMIT = 60

_LETTERS = "A-Za-z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
#: One word of a command key: no spaces, quotes, backticks, ``$``, ``<>`` or newlines.
_WORD = re.compile(rf"[{_LETTERS}._/+=:,@%~^-]+")
#: A tool name, and a directory (spaces allowed, nothing else extra).
_TOOL = re.compile(r"[A-Za-z0-9_.-]+")
_FOLDER = re.compile(rf"[{_LETTERS} ._/+=:,@%~^-]+")
#: Changing directory says nothing about what was approved.
_NAVIGATION = frozenset({"cd", "pushd", "popd"})
#: Commands whose prefix says nothing about what they run.
_OPAQUE = frozenset({"eval", "source", ".", "trap", "alias"})

_lock = threading.Lock()


class Key(NamedTuple):
    """One precedent key of a call: ``kind`` is ``"cmd"`` or ``"path"``."""

    kind: str
    text: str


@dataclass(frozen=True)
class Precedent:
    """One remembered answer: ``approved: git push (3×)``."""

    decision: str
    kind: str
    key: str
    count: int
    last: int          # ms since the epoch

    def line(self) -> str:
        return f"{self.decision}: {self.key} ({self.count}×)"

    def dto(self) -> dict[str, Any]:
        return {"decision": self.decision, "kind": self.kind, "key": self.key,
                "count": self.count, "lastUsed": self.last}


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _plain_redirects(redirects: Sequence[tuple[str, str]]) -> bool:
    """Only ``2>&1``-style duplication and ``> /dev/null`` — no file is read or written."""
    for op, target in redirects:
        if op in _DUP_REDIRECTS and (target.isdigit() or target == "-"):
            continue
        if op in _OUT_REDIRECTS and target == "/dev/null":
            continue
        return False
    return True


def _command_key(argv: list[str]) -> str | None:
    """The key of one simple command; ``""`` = nothing worth a key, ``None`` = not understood."""
    words = list(argv)
    first = _command_name(words[0]) or words[0].rsplit("/", 1)[-1]
    if first not in ("sudo", "doas"):
        words = _strip_wrappers(words)        # nohup, timeout 60, env X=1, xargs …
        if not words:
            return None
        first = _command_name(words[0]) or words[0].rsplit("/", 1)[-1]
    if first in _NAVIGATION:
        return ""
    if first in _OPAQUE:
        return None
    prefix = command_prefix(words)
    if not prefix:
        return None
    core = prefix[1:] if prefix[0] in ("sudo", "doas") else prefix
    if not core or core[0] in _OPAQUE:
        return None
    if _INTERPRETERS.match(core[0]) and (len(core) == 1 or (core[1].startswith("-") and core[1] != "-m")):
        return None                           # python -c '…', bash -c, node -e, a bare REPL
    if not all(_WORD.fullmatch(_nfc(w)) for w in prefix):
        return None
    key = _nfc(" ".join(prefix))
    return key if len(key) <= KEY_LIMIT else None


def shell_keys(command: str) -> list[str]:
    """Keys of a shell command: one per simple command that is not read-only.

    ``git add -A && git push origin main`` → ``["git add", "git push"]``. All
    or nothing: when any part is not understood (see the module docstring) the
    whole command gives ``[]`` — it is neither recorded nor matched.
    """
    parsed = parse_shell(command or "")
    if parsed.parse_error or parsed.has_substitution or parsed.has_subshell or not parsed.commands:
        return []
    keys: list[str] = []
    for sc in parsed.commands:
        if not _plain_redirects(sc.redirects):
            return []
        if any(a.split("=", 1)[0].rstrip("+") not in _SAFE_ASSIGNMENTS for a in sc.assignments):
            return []
        if not sc.argv or _simple_is_read_only(sc.argv):
            continue
        key = _command_key(sc.argv)
        if key is None:
            return []
        if key and key not in keys:
            keys.append(key)
    return keys


def _tilde(path: str) -> str:
    home = os.path.realpath(os.path.expanduser("~"))
    if path == home or path.startswith(home.rstrip(os.sep) + os.sep):
        return "~" + path[len(home):]
    return path


def _folder_of(path: str | os.PathLike[str]) -> str:
    """The real directory a target lives in (the target itself when it is a directory)."""
    real = os.path.realpath(os.path.expanduser(os.fspath(path)))
    return real if os.path.isdir(real) else os.path.dirname(real)


def path_key(tool: str, path: str | os.PathLike[str]) -> str | None:
    """``"<tool> <directory>"`` for a file tool's target; ``None`` when it fails the check."""
    tool = str(tool or "")
    folder = _nfc(_tilde(_folder_of(path)))
    key = f"{tool} {folder}"
    if not _TOOL.fullmatch(tool) or not _FOLDER.fullmatch(folder) or len(key) > KEY_LIMIT:
        return None
    return key


def _target_of(tool: Any, args: Any) -> str | None:
    """The ``path`` argument of a file tool, after the tool's own alias handling."""
    data = args if isinstance(args, dict) else {}
    normalize = getattr(tool, "args", None)
    if callable(normalize):
        try:
            data = normalize(args)
        except Exception:  # noqa: BLE001 - a malformed call has no target
            return None
    raw = data.get("path") if isinstance(data, dict) else None
    return str(raw) if raw else None


def call_keys(tool: Any, args: Any, ctx: Any) -> list[Key]:
    """The precedent keys of one call: shell keys for ``bash``, else the target's folder.

    ``ctx`` is the tool context (``ctx.cwd`` resolves a relative path exactly as
    the tool does). Other ``exec`` tools (``model_load``) have no keys.
    """
    name = str(getattr(tool, "name", "") or "")
    if str(getattr(tool, "kind", "") or "") == "exec":
        if name != "bash":
            return []
        from localcode.agent.tools.shell import command_from_args  # lazy: tools import permissions

        return [Key("cmd", k) for k in shell_keys(command_from_args(args))]
    raw = _target_of(tool, args)
    if raw is None:
        return []
    from localcode.agent.tools.fs import ToolInputError, resolve_path  # lazy, as above

    try:
        target = resolve_path(raw, ctx if hasattr(ctx, "cwd") else SimpleNamespace(cwd=ctx))
    except (ToolInputError, OSError, ValueError):
        return []
    key = path_key(name, target)
    return [Key("path", key)] if key else []


def _covers(entry: Precedent, key: Key) -> bool:
    if entry.kind != key.kind:
        return False
    if key.kind == "cmd":
        return entry.key == key.text
    tool, _, folder = entry.key.partition(" ")
    their_tool, _, their_folder = key.text.partition(" ")
    if tool != their_tool or not folder:
        return False
    root = os.path.expanduser(folder).rstrip(os.sep) or os.sep
    here = os.path.expanduser(their_folder)
    return here == root or here.startswith(root.rstrip(os.sep) + os.sep)


def matching(entries: Iterable[Precedent], keys: Sequence[Key]) -> list[Precedent]:
    """Entries that match any of the call's keys (same command prefix; same tool, folder inside)."""
    return [e for e in entries if any(_covers(e, k) for k in keys)]


def reviewer_lines(entries: Sequence[Precedent], keys: Sequence[Key]) -> list[str]:
    """The reviewer's precedents: only matching ones, newest first, at most :data:`REVIEWER_LIMIT`."""
    return [e.line() for e in matching(entries, keys)[:REVIEWER_LIMIT]]


def guard_lines(entries: Sequence[Precedent], keys: Sequence[Key]) -> list[str]:
    """The guard's history: the matching entries, then the most recent others, :data:`GUARD_LIMIT` in all."""
    hits = matching(entries, keys)
    rest = [e for e in entries if e not in hits]
    return [e.line() for e in [*hits, *rest][:GUARD_LIMIT]]


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def store_path() -> Path:
    return lc_config.home() / STORE_NAME


def project_id(cwd: str | os.PathLike[str]) -> str:
    """The real path of a project: what precedents are kept under."""
    return os.path.realpath(os.path.expanduser(os.fspath(cwd)))


def _now() -> int:
    return int(time.time() * 1000)


def _read() -> dict[str, Any]:
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": 1, "projects": {}}
    projects = data.get("projects") if isinstance(data, dict) else None
    return {"version": 1, "projects": projects if isinstance(projects, dict) else {}}


def _entry(raw: Any) -> Precedent | None:
    """A stored entry, checked again: the file is ours, but it is still only data."""
    if not isinstance(raw, dict):
        return None
    decision, kind, key = raw.get("decision"), raw.get("kind"), raw.get("key")
    if decision not in (APPROVED, DECLINED) or kind not in ("cmd", "path") or not isinstance(key, str):
        return None
    key = _nfc(key)
    if len(key) > KEY_LIMIT:
        return None
    if kind == "cmd" and not all(_WORD.fullmatch(w) for w in key.split(" ")):
        return None
    if kind == "path":
        tool, _, folder = key.partition(" ")
        if not _TOOL.fullmatch(tool) or not _FOLDER.fullmatch(folder or " "):
            return None
    try:
        count, last = max(1, int(raw.get("count") or 1)), int(raw.get("last") or 0)
    except (TypeError, ValueError):
        return None
    return Precedent(decision, kind, key, count, last)


def _entries(project: Any) -> list[Precedent]:
    raw = project.get("entries") if isinstance(project, dict) else None
    found = [e for e in (_entry(r) for r in raw or ()) if e is not None]
    return sorted(found, key=lambda e: e.last, reverse=True)


def load(cwd: str | os.PathLike[str]) -> list[Precedent]:
    """This project's precedents, most recently used first (``[]`` when the file is unreadable)."""
    with _lock:
        projects = _read()["projects"]
    return _entries(projects.get(project_id(cwd)))


def record(cwd: str | os.PathLike[str], keys: Sequence[Key], decision: str, *,
           now: int | None = None) -> None:
    """Count one answer (``approved``/``declined``) for every key of a call.

    Never raises: a store that cannot be written costs the memory, not the call.
    """
    if decision not in (APPROVED, DECLINED) or not keys:
        return
    stamp = _now() if now is None else int(now)
    project = project_id(cwd)
    try:
        with _lock:
            data = _read()
            projects = data["projects"]
            entries = {(e.decision, e.kind, e.key): e for e in _entries(projects.get(project))}
            for key in keys:
                if _entry({"decision": decision, "kind": key.kind, "key": key.text}) is None:
                    continue                  # the same check as on reading: never store junk
                old = entries.get((decision, key.kind, key.text))
                entries[(decision, key.kind, key.text)] = Precedent(
                    decision, key.kind, key.text, (old.count + 1) if old else 1, stamp)
            kept = sorted(entries.values(), key=lambda e: e.last, reverse=True)[:MAX_PER_PROJECT]
            projects[project] = {"used": stamp, "entries": [
                {"decision": e.decision, "kind": e.kind, "key": e.key, "count": e.count, "last": e.last}
                for e in kept]}
            if len(projects) > MAX_PROJECTS:
                order = sorted(projects, key=lambda p: _used(projects[p]), reverse=True)
                data["projects"] = {p: projects[p] for p in order[:MAX_PROJECTS]}
            lc_config.write_private_json(store_path(), data)
    except OSError:
        log.warning("could not save permission precedents", exc_info=True)


def _used(project: Any) -> int:
    try:
        return int(project.get("used") or 0) if isinstance(project, dict) else 0
    except (TypeError, ValueError):
        return 0


def listing(cwd: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """``[{cwd, precedents: [...]}, …]`` — one project, or every one (most recent first)."""
    with _lock:
        projects = _read()["projects"]
    names = [project_id(cwd)] if cwd else sorted(projects, key=lambda p: _used(projects[p]), reverse=True)
    out = []
    for name in names:
        entries = _entries(projects.get(name))
        if entries or cwd:
            out.append({"cwd": name, "precedents": [e.dto() for e in entries]})
    return out


def clear(cwd: str | os.PathLike[str] | None = None) -> int:
    """Forget one project's precedents (``cwd``) or everyone's; returns how many were dropped."""
    with _lock:
        data = _read()
        projects = data["projects"]
        if cwd:
            dropped = len(_entries(projects.pop(project_id(cwd), None)))
        else:
            dropped = sum(len(_entries(p)) for p in projects.values())
            data["projects"] = {}
        if dropped or (cwd is None and store_path().exists()):
            lc_config.write_private_json(store_path(), data)
    return dropped
