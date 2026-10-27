"""The skills shipped in this repository, checked as content rather than code.

Every other test here drives the skill *machinery* with fixtures. This one reads
what is actually on the shelf (``.localcode/skills/``) and holds it to the rules
in ``docs/SKILLS.md``, because a skill is read by a model that cannot tell a
careful instruction from a careless one: a mini variant that quietly grew past
the window, or a command that appears only in the short version, costs a turn
and teaches the model to distrust the whole shelf.

The limits come from ``localcode.agent.skills`` rather than being repeated here,
so raising one is a single edit in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from localcode.agent.skills import (
    MAX_MINI_CHARS,
    MAX_MINI_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    MINI,
    MINI_FILE,
    SKILL_FILE,
    SkillLibrary,
    parse_skill,
)

REPO = Path(__file__).resolve().parents[2]
SHELF = REPO / ".localcode" / "skills"

#: A mini variant is a briefing, not a page: past this it stops being one.
MAX_MINI_LINES = 15
#: Shell commands are the part a model copies verbatim, so they are compared.
_BASH_BLOCK = re.compile(r"```(?:bash|sh|console)\n(.*?)```", re.DOTALL)
#: The library's own rule (``skills.py``), not a stricter one: a test that
#: rejects a name the loader accepts is a trap for whoever writes the next skill.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def skill_dirs() -> list[Path]:
    if not SHELF.is_dir():
        return []
    return sorted(d for d in SHELF.iterdir() if d.is_dir() and (d / SKILL_FILE).is_file())


def ids(dirs: list[Path]) -> list[str]:
    return [d.name for d in dirs]


DIRS = skill_dirs()


def commands(text: str) -> set[str]:
    """Every non-comment command line inside a shell block."""
    out: set[str] = set()
    for block in _BASH_BLOCK.findall(text):
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out


def test_the_shelf_is_not_empty() -> None:
    """A guard for the guard: a silent empty shelf would pass every test below."""
    assert DIRS, f"brak skillów w {SHELF}"


@pytest.mark.parametrize("folder", DIRS, ids=ids(DIRS))
def test_a_skill_parses_and_names_itself_clearly(folder: Path) -> None:
    mini = folder / MINI_FILE
    skill = parse_skill((folder / SKILL_FILE).read_text(encoding="utf-8"),
                        folder / SKILL_FILE, "project",
                        mini.read_text(encoding="utf-8") if mini.is_file() else None)
    assert skill.name == folder.name, "nazwa w nagłówku musi zgadzać się z katalogiem"
    assert _NAME_RE.match(skill.name), "małe litery, cyfry i myślniki"
    assert len(skill.name) <= MAX_NAME_CHARS
    # When the shelf outgrows the menu budget the names are all a small model
    # sees. "Descriptive" cannot be asserted, but a two-letter name or a trailing
    # digit ("testy2") is a reliable sign of the opposite.
    assert len(skill.name) >= 4, "nazwa za krótka, żeby cokolwiek znaczyła"
    assert not skill.name[-1].isdigit(), "nazwa z numerem nic nie mówi o treści"


@pytest.mark.parametrize("folder", DIRS, ids=ids(DIRS))
def test_a_skill_has_both_descriptions(folder: Path) -> None:
    skill = parse_skill((folder / SKILL_FILE).read_text(encoding="utf-8"),
                        folder / SKILL_FILE, "project")
    assert skill.description.endswith(".") or skill.description.endswith("…")
    assert len(skill.description_for(MINI)) <= MAX_MINI_DESCRIPTION_CHARS
    # A shortened long description ends in an ellipsis; a written one does not.
    # Requiring the second is the whole point of the field.
    assert not skill.description_for(MINI).endswith("…"), (
        f"„{folder.name}” nie ma pola description-mini — pełny opis jest ucinany")


@pytest.mark.parametrize("folder", [d for d in DIRS if (d / MINI_FILE).is_file()],
                         ids=ids([d for d in DIRS if (d / MINI_FILE).is_file()]))
def test_the_mini_variant_stays_a_briefing(folder: Path) -> None:
    text = (folder / MINI_FILE).read_text(encoding="utf-8")
    assert not text.lstrip().startswith("---"), "mini to sama treść, bez nagłówka"
    assert len(text) <= MAX_MINI_CHARS, f"{len(text)} znaków, limit {MAX_MINI_CHARS}"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) <= MAX_MINI_LINES, f"{len(lines)} niepustych linii, limit {MAX_MINI_LINES}"
    assert "|" not in text, "bez tabel — w oknie 8k to sam narzut"


@pytest.mark.parametrize("folder", [d for d in DIRS if (d / MINI_FILE).is_file()],
                         ids=ids([d for d in DIRS if (d / MINI_FILE).is_file()]))
def test_the_two_variants_agree_on_the_commands(folder: Path) -> None:
    """Mini is a subset of the full one, not a second opinion."""
    full = commands((folder / SKILL_FILE).read_text(encoding="utf-8"))
    mini = commands((folder / MINI_FILE).read_text(encoding="utf-8"))
    missing = sorted(mini - full)
    assert not missing, f"polecenia tylko w mini: {missing}"


@pytest.mark.parametrize("folder", DIRS, ids=ids(DIRS))
def test_a_skill_does_not_send_the_model_to_a_file_that_is_not_there(folder: Path) -> None:
    """The failure this repository already made once.

    A skill told the agent to run ``./scripts/deploy.sh``, which has never
    existed here. The model spends a turn on it and learns to distrust the rest.
    """
    text = (folder / SKILL_FILE).read_text(encoding="utf-8")
    if (folder / MINI_FILE).is_file():
        text += (folder / MINI_FILE).read_text(encoding="utf-8")
    referenced = set(re.findall(r"(?:^|[\s`(])(\./[\w./-]+|scripts/[\w./-]+|"
                                r"localcode/[\w./-]+|macos/[\w./-]+|docs/[\w./-]+)", text))
    missing = sorted(ref for ref in referenced
                     if not (REPO / ref.removeprefix("./")).exists())
    assert not missing, f"skill odsyła do nieistniejących ścieżek: {missing}"


def test_every_skill_on_the_shelf_loads_through_the_library() -> None:
    library = SkillLibrary(REPO)
    library.reload()
    names = {s.name for s in library.all()}
    assert names >= {d.name for d in DIRS}
    assert library.errors == [], f"skille z błędami: {library.errors}"
