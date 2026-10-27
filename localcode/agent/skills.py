"""Skills: instructions the agent loads only when they are needed.

A skill is a folder with a ``SKILL.md`` inside:

    ~/.localcode/skills/publikacja/SKILL.md      ← yours, every project
    <projekt>/.localcode/skills/testy/SKILL.md   ← this project's, shared in git

    ---
    name: testy
    description: Jak uruchamiać i pisać testy w tym repo (pytest, znaczniki).
    ---

    # Testy

    Uruchom `.venv/bin/pytest tests -q`…

Why a folder and not one long system prompt: this project runs 4B models with
8k windows.  Everything in the prompt is paid for on *every* request, so the
agent is told only the names and the one-line descriptions — a few dozen tokens
— and reads the body through the ``skill`` tool at the moment it decides the
skill applies.  A skill with ten pages of instructions therefore costs nothing
until it is used.

Two places, one rule: a project skill shadows a personal one with the same name,
because the repository knows better than the machine what this repository needs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

#: Where skills live, relative to the home directory and to the project.
USER_DIR = "skills"
PROJECT_DIR = ".localcode/skills"
SKILL_FILE = "SKILL.md"
#: The short variant, next to ``SKILL.md`` and optional (body only, no header).
MINI_FILE = "SKILL-mini.md"

#: Variants. ``full`` is what a skill has always been; ``mini`` is for a model
#: whose window cannot afford it.
FULL, MINI = "full", "mini"

#: A window this size or smaller gets the mini variant. Chosen from what a tool
#: result may cost there: at 16k the per-result clamp is 10 000 characters
#: (``context.py``), so a full 20 000-character skill would be cut anyway —
#: better to send a short text written on purpose than a long one chopped.
MINI_CTX = 16_384

#: Hard limits — a skill body that does not fit the window is no use to anybody.
MAX_BODY_CHARS = 20_000
MAX_DESCRIPTION_CHARS = 200
MAX_NAME_CHARS = 48
#: The mini body. An 8k window leaves roughly 2 200 tokens for actual work once
#: the system prompt, the project instructions and the first task are in, so a
#: skill may take about a fifth of that: ~500 tokens ≈ 1 500 characters of
#: Polish (its diacritics cost a token each, so 3 chars/token, not 3.5).
MAX_MINI_CHARS = 1_500
#: The mini description, for the menu that sits in *every* request.
MAX_MINI_DESCRIPTION_CHARS = 80
#: The whole mini menu. Without a ceiling, forty skills × 200 characters would
#: spend ~2 300 tokens — a third of an 8k window — before the model does
#: anything. Over this, the menu degrades to bare names.
MAX_MINI_MENU_CHARS = 1_200

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$", re.IGNORECASE)
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


class SkillError(Exception):
    """A skill problem with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "skill_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class Skill:
    """One folder of instructions the agent can read on demand."""

    name: str
    description: str
    body: str
    path: Path
    scope: str = "user"                       # user | project
    allowed_tools: list[str] = field(default_factory=list)
    #: ``SKILL-mini.md``, when the skill ships one; empty otherwise.
    mini_body: str = ""
    #: ``description-mini`` from the header, or a shortened ``description``.
    mini_description: str = ""

    @property
    def directory(self) -> Path:
        return self.path.parent

    def body_for(self, variant: str) -> str:
        """The text a model of this size should get.

        A skill with no mini variant is not refused: the model gets the opening
        of the full one, cut at a paragraph, with a line saying where the rest
        is — it can read on with ``read_file`` if it needs to. Refusing would
        leave a small model with nothing, and the skill template puts what
        matters first precisely so that this cut stays useful.
        """
        if variant != MINI:
            return self.body
        if self.mini_body:
            return self.mini_body
        if len(self.body) <= MAX_MINI_CHARS:
            return self.body
        return (_clip_at_paragraph(self.body, MAX_MINI_CHARS)
                + f"\n\n[skrócone — ten skill nie ma {MINI_FILE}; całość: {self.path}]")

    def description_for(self, variant: str) -> str:
        return self.mini_description if variant == MINI else self.description

    def to_dict(self, *, include_body: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "path": str(self.path),
            "allowedTools": list(self.allowed_tools),
            "hasMini": bool(self.mini_body),
        }
        if include_body:
            data["body"] = self.body
        return data

    def files(self) -> list[str]:
        """Other files shipped with the skill (scripts, templates, examples)."""
        try:
            return sorted(p.name for p in self.directory.iterdir()
                          if p.is_file() and p.name not in (SKILL_FILE, MINI_FILE)
                          and not p.name.startswith("."))
        except OSError:
            return []


def _clip_at_paragraph(text: str, limit: int) -> str:
    """``text`` cut to ``limit``, preferring the last blank line before it.

    Cutting mid-sentence reads as a bug to the model as much as to a person, so
    fall back to a hard cut only when there is no paragraph break to use.
    """
    if len(text) <= limit:
        return text
    head = text[:limit]
    break_at = head.rfind("\n\n")
    return (head[:break_at] if break_at > limit // 2 else head).rstrip()


def _shorten(text: str, limit: int) -> str:
    """One line cut to ``limit`` characters, on a word boundary where possible."""
    if len(text) <= limit:
        return text
    head = text[:limit - 1]
    space = head.rfind(" ")
    return (head[:space] if space > limit // 2 else head).rstrip() + "…"


def parse_skill(text: str, path: Path, scope: str = "user",
                mini_text: str | None = None) -> Skill:
    """``SKILL.md`` → :class:`Skill`.

    The front matter is read with a deliberately small YAML subset (``key: value``
    and ``key: [a, b]``): a skill file is written by hand, and depending on a
    YAML parser for three keys would be a dependency for nothing.
    """
    match = _FRONT_MATTER_RE.match(text or "")
    if not match:
        raise SkillError(
            f"„{path.name}” nie ma nagłówka. Zacznij plik od bloku:\n"
            "---\nname: nazwa\ndescription: jedno zdanie, kiedy tego użyć\n---",
            "no_front_matter")
    meta = _front_matter(match.group(1))
    body = (match.group(2) or "").strip()

    name = str(meta.get("name") or path.parent.name).strip()
    if not _NAME_RE.match(name) or len(name) > MAX_NAME_CHARS:
        raise SkillError(
            f"Nazwa skilla „{name}” jest nieprawidłowa — użyj liter, cyfr, „-” i „_”.",
            "bad_name")
    description = " ".join(str(meta.get("description") or "").split())
    if not description:
        raise SkillError(
            f"Skill „{name}” nie ma pola description — bez niego model nie wie, kiedy go użyć.",
            "no_description")
    if len(description) > MAX_DESCRIPTION_CHARS:
        description = description[:MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
    if not body:
        raise SkillError(f"Skill „{name}” jest pusty — po nagłówku nie ma żadnych instrukcji.",
                         "empty_body")
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS].rstrip() + "\n\n[…dalsza część pominięta…]"

    # The mini description defaults to a shortened full one, so a skill written
    # before this existed still gets a usable short menu entry.
    mini_description = " ".join(str(meta.get("description-mini")
                                    or meta.get("description_mini") or "").split())
    mini_description = (_shorten(mini_description, MAX_MINI_DESCRIPTION_CHARS)
                        if mini_description
                        else _shorten(description, MAX_MINI_DESCRIPTION_CHARS))

    # ``SKILL-mini.md`` carries the body alone: one header, one source of truth
    # for name and description, so the two variants cannot contradict each other.
    mini_body = (mini_text or "").strip()
    mini_match = _FRONT_MATTER_RE.match(mini_body)
    if mini_match:
        mini_body = (mini_match.group(2) or "").strip()
    if len(mini_body) > MAX_MINI_CHARS:
        mini_body = _clip_at_paragraph(mini_body, MAX_MINI_CHARS) + "\n\n[…skrócone…]"

    tools = meta.get("allowed-tools") or meta.get("allowed_tools") or []
    return Skill(name=name, description=description, body=body, path=path, scope=scope,
                 allowed_tools=[str(t).strip() for t in tools if str(t).strip()],
                 mini_body=mini_body, mini_description=mini_description)


def _front_matter(text: str) -> dict[str, Any]:
    """The handful of keys a skill header may carry."""
    out: dict[str, Any] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if value.startswith("[") and value.endswith("]"):
            out[key] = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
        else:
            out[key] = value
    return out


def variant_for(*, strong: bool, n_ctx: int) -> str:
    """Which variant a model of this shape should be served.

    Deliberately not ``ModelProfile.tiny_tools``: that is true only up to 8k and
    false for an unknown window (``n_ctx == 0``), which would hand the longest
    text to the model we know least about. A model reached over an API key has
    a window large enough that the full text is always the better answer.
    """
    if strong:
        return FULL
    window = int(n_ctx or 0)
    return FULL if window > MINI_CTX else MINI


class SkillLibrary:
    """Every skill available here: the personal ones plus this project's."""

    def __init__(self, cwd: str | Path | None = None, *, home: Path | None = None) -> None:
        self.cwd = Path(cwd).expanduser() if cwd else Path.cwd()
        self._home = home
        self._cache: dict[str, Skill] | None = None
        self._errors: list[str] = []
        #: Which variant this session serves; set by the agent loop once it
        #: knows the model (see ``use_for``). Full until told otherwise, which
        #: is what every caller got before variants existed.
        self.variant = FULL

    @property
    def home(self) -> Path:
        if self._home is not None:
            return self._home
        from localcode.config import home as config_home

        return config_home()

    def directories(self) -> list[tuple[Path, str]]:
        """``(directory, scope)`` pairs, personal first so the project can shadow."""
        return [(self.home / USER_DIR, "user"), (self.cwd / PROJECT_DIR, "project")]

    def use_for(self, *, strong: bool, n_ctx: int) -> None:
        """Serve the variant that fits this model (see :func:`variant_for`)."""
        self.variant = variant_for(strong=strong, n_ctx=n_ctx)

    def reload(self) -> None:
        self._cache = None
        self._errors = []

    @property
    def errors(self) -> list[str]:
        """Skills that could not be read, as Polish sentences (never raised)."""
        self.all()
        return list(self._errors)

    def all(self) -> list[Skill]:
        if self._cache is None:
            found: dict[str, Skill] = {}
            errors: list[str] = []
            for directory, scope in self.directories():
                for skill in _scan(directory, scope, errors):
                    found[skill.name] = skill      # project wins: it is scanned last
            self._cache = found
            self._errors = errors
        return sorted(self._cache.values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        wanted = str(name or "").strip().lower()
        return next((s for s in self.all() if s.name.lower() == wanted), None)

    def menu(self, variant: str | None = None) -> str:
        """The handful of tokens the system prompt spends on skills.

        This sits in *every* request, so for a small window it is budgeted: the
        short descriptions first, and if forty of them still do not fit, the
        names alone. That last step is why a skill's name has to say what it is
        — sometimes it is all the model sees.
        """
        skills = self.all()
        if not skills:
            return ""
        wanted = variant or self.variant
        if wanted != MINI:
            return "\n".join(f"- {s.name}: {s.description}" for s in skills)
        lines = [f"- {s.name}: {s.description_for(MINI)}" for s in skills]
        menu = "\n".join(lines)
        if len(menu) <= MAX_MINI_MENU_CHARS:
            return menu
        return "Skille: " + ", ".join(s.name for s in skills)

    def create(self, name: str, description: str, body: str = "", *,
               scope: str = "user", mini_body: str = "",
               mini_description: str = "") -> Skill:
        """Write a new skill folder (what ``localcode skills new`` does)."""
        clean = str(name or "").strip().lower()
        if not _NAME_RE.match(clean):
            raise SkillError("Nazwa może zawierać litery, cyfry, „-” i „_”.", "bad_name")
        if not description.strip():
            raise SkillError("Podaj jedno zdanie opisu — po nim model wybiera skill.",
                             "no_description")
        base = self.home / USER_DIR if scope == "user" else self.cwd / PROJECT_DIR
        directory = base / clean
        target = directory / SKILL_FILE
        if target.exists():
            raise SkillError(f"Skill „{clean}” już istnieje: {target}", "already_exists")
        directory.mkdir(parents=True, exist_ok=True)
        header = [f"name: {clean}", f"description: {description.strip()}"]
        if mini_description.strip():
            header.append(f"description-mini: {mini_description.strip()}")
        content = ("---\n" + "\n".join(header) + "\n---\n\n"
                   + (body.strip() or f"# {clean}\n\nTu wpisz instrukcje.\n") + "\n")
        target.write_text(content, encoding="utf-8")
        if mini_body.strip():
            (directory / MINI_FILE).write_text(mini_body.strip() + "\n", encoding="utf-8")
        self.reload()
        skill = self.get(clean)
        if skill is None:  # pragma: no cover - only if the write silently failed
            raise SkillError("Nie udało się odczytać nowo utworzonego skilla.", "write_failed")
        return skill


def _scan(directory: Path, scope: str, errors: list[str]) -> Iterable[Skill]:
    try:
        entries = sorted(p for p in directory.iterdir() if p.is_dir())
    except OSError:
        return []
    out: list[Skill] = []
    for entry in entries:
        path = entry / SKILL_FILE
        if not path.is_file():
            continue
        mini = entry / MINI_FILE
        try:
            mini_text = mini.read_text(encoding="utf-8") if mini.is_file() else None
        except OSError as exc:      # unreadable mini must not cost us the skill
            mini_text = None
            errors.append(f"{mini}: nie udało się odczytać ({exc.strerror or exc}).")
        try:
            out.append(parse_skill(path.read_text(encoding="utf-8"), path, scope, mini_text))
        except SkillError as exc:
            errors.append(f"{path}: {exc.message}")
        except OSError as exc:
            errors.append(f"{path}: nie udało się odczytać ({exc.strerror or exc}).")
    return out
