"""Skills: instructions loaded on demand (``localcode.agent.skills``).

The whole point is what does *not* happen: a shelf of long procedures must cost
a few dozen tokens per request, not thousands, because the models this app runs
have eight-thousand-token windows. So the prompt gets names and one-liners, and
the body arrives only through the tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localcode.agent.skills import (
    MAX_BODY_CHARS,
    Skill,
    SkillError,
    SkillLibrary,
    parse_skill,
)
from localcode.agent.tools.base import ToolContext, ToolLimits
from localcode.agent.tools.skill import STATE_KEY, SkillTool

SKILL = """---
name: testy
description: Jak uruchamiać testy w tym repo.
---

# Testy

Uruchom `.venv/bin/pytest tests -q`.
"""


def write_skill(directory: Path, name: str, *, description: str = "Opis skilla.",
                body: str = "Instrukcje.", mini: str | None = None,
                mini_description: str | None = None) -> Path:
    folder = directory / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    header = [f"name: {name}", f"description: {description}"]
    if mini_description is not None:
        header.append(f"description-mini: {mini_description}")
    path.write_text("---\n" + "\n".join(header) + f"\n---\n\n{body}\n", encoding="utf-8")
    if mini is not None:
        (folder / "SKILL-mini.md").write_text(mini + "\n", encoding="utf-8")
    return path


@pytest.fixture
def library(tmp_path: Path) -> SkillLibrary:
    home = tmp_path / "home"
    project = tmp_path / "projekt"
    (home / "skills").mkdir(parents=True)
    (project / ".localcode" / "skills").mkdir(parents=True)
    return SkillLibrary(project, home=home)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_a_skill_file_is_a_header_and_a_body(tmp_path: Path) -> None:
    skill = parse_skill(SKILL, tmp_path / "testy" / "SKILL.md")
    assert skill.name == "testy"
    assert skill.description == "Jak uruchamiać testy w tym repo."
    assert skill.body.startswith("# Testy")


def test_a_file_without_a_header_says_what_to_write(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        parse_skill("# Testy\n\nbez nagłówka", tmp_path / "x" / "SKILL.md")
    assert "description:" in excinfo.value.message


def test_a_skill_without_a_description_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        parse_skill("---\nname: x\n---\n\ncokolwiek", tmp_path / "x" / "SKILL.md")
    assert excinfo.value.code == "no_description"


def test_an_empty_skill_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SkillError) as excinfo:
        parse_skill("---\nname: x\ndescription: coś\n---\n\n   ", tmp_path / "x" / "SKILL.md")
    assert excinfo.value.code == "empty_body"


def test_a_silly_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SkillError):
        parse_skill("---\nname: ../../etc\ndescription: coś\n---\n\nciało",
                    tmp_path / "x" / "SKILL.md")


def test_a_very_long_body_is_cut(tmp_path: Path) -> None:
    text = "---\nname: x\ndescription: coś\n---\n\n" + ("a" * (MAX_BODY_CHARS + 500))
    skill = parse_skill(text, tmp_path / "x" / "SKILL.md")
    assert len(skill.body) <= MAX_BODY_CHARS + 60
    assert "pominięta" in skill.body


def test_allowed_tools_are_read_as_a_list(tmp_path: Path) -> None:
    text = "---\nname: x\ndescription: coś\nallowed-tools: [read_file, bash]\n---\n\nciało"
    assert parse_skill(text, tmp_path / "x" / "SKILL.md").allowed_tools == ["read_file", "bash"]


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #


def test_both_places_are_searched(library: SkillLibrary) -> None:
    write_skill(library.home / "skills", "publikacja")
    write_skill(library.cwd / ".localcode" / "skills", "testy")
    assert [s.name for s in library.all()] == ["publikacja", "testy"]
    assert {s.scope for s in library.all()} == {"user", "project"}


def test_a_project_skill_shadows_a_personal_one(library: SkillLibrary) -> None:
    write_skill(library.home / "skills", "testy", description="Wersja domowa.")
    write_skill(library.cwd / ".localcode" / "skills", "testy", description="Wersja projektu.")
    skills = library.all()
    assert len(skills) == 1
    assert skills[0].description == "Wersja projektu."
    assert skills[0].scope == "project"


def test_a_broken_skill_is_reported_not_raised(library: SkillLibrary) -> None:
    folder = library.home / "skills" / "zepsuty"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("bez nagłówka", encoding="utf-8")
    write_skill(library.home / "skills", "dobry")

    assert [s.name for s in library.all()] == ["dobry"]      # the good one still loads
    assert any("zepsuty" in error for error in library.errors)


def test_the_menu_is_short(library: SkillLibrary) -> None:
    """What the system prompt pays for on every single request."""
    for name in ("testy", "publikacja", "migracje"):
        write_skill(library.home / "skills", name, description="Jedno zdanie o tym skillu.")
    menu = library.menu()
    assert menu.count("\n") == 2
    assert len(menu) < 300
    assert "Instrukcje." not in menu                          # the body stays out


def test_no_skills_means_no_menu(library: SkillLibrary) -> None:
    assert library.menu() == ""
    assert library.all() == []


def test_creating_a_skill_writes_a_folder(library: SkillLibrary) -> None:
    skill = library.create("wydanie", "Jak wydać nową wersję.", scope="user")
    assert skill.path.exists()
    assert skill.path.parent.name == "wydanie"
    assert library.get("wydanie") is not None

    with pytest.raises(SkillError) as excinfo:
        library.create("wydanie", "Drugi raz.")
    assert excinfo.value.code == "already_exists"


def test_files_next_to_the_skill_are_listed(library: SkillLibrary) -> None:
    path = write_skill(library.home / "skills", "raport")
    (path.parent / "szablon.md").write_text("szablon", encoding="utf-8")
    (path.parent / ".ukryty").write_text("x", encoding="utf-8")
    assert library.get("raport").files() == ["szablon.md"]


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #


@pytest.fixture
def ctx(library: SkillLibrary, tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, limits=ToolLimits(), state={STATE_KEY: library})


async def test_the_tool_returns_the_body(library: SkillLibrary, ctx: ToolContext) -> None:
    write_skill(library.home / "skills", "testy", body="Uruchom pytest.")
    library.reload()
    result = await SkillTool().run({"name": "testy"}, ctx)
    assert result.ok
    assert "Uruchom pytest." in result.output
    assert "Skill: testy" in result.output


async def test_the_tool_lists_the_alternatives_when_the_name_is_wrong(
        library: SkillLibrary, ctx: ToolContext) -> None:
    write_skill(library.home / "skills", "testy")
    library.reload()
    result = await SkillTool().run({"name": "test"}, ctx)
    assert not result.ok
    assert "testy" in result.output


async def test_the_tool_needs_no_permission(ctx: ToolContext) -> None:
    assert await SkillTool().permission({"name": "testy"}, ctx) is None


async def test_without_a_library_the_tool_says_so(tmp_path: Path) -> None:
    empty = ToolContext(cwd=tmp_path, limits=ToolLimits(), state={})
    result = await SkillTool().run({"name": "testy"}, empty)
    assert not result.ok


# --------------------------------------------------------------------------- #
# The menu and the prompt
# --------------------------------------------------------------------------- #


def test_the_prompt_mentions_skills_only_when_there_are_any() -> None:
    from localcode.agent.prompts import build_system_prompt

    common = dict(cwd="/tmp/projekt", n_ctx=32768, mode="default", native_tools=True,
                  tool_names=["read_file", "skill"], project_instructions=None, env={})
    without = build_system_prompt(**common)
    with_skills = build_system_prompt(**common, skills="- testy: Jak uruchamiać testy.")

    assert "skill" not in without.lower().split("# skills")[0].split("tools")[0]
    assert "# Skills" in with_skills
    assert "- testy: Jak uruchamiać testy." in with_skills
    assert len(with_skills) - len(without) < 400          # the shelf stays cheap


def test_the_tool_appears_only_with_skills() -> None:
    from localcode.agent.tools import tools_for_mode

    assert "skill" not in [t.name for t in tools_for_mode("default")]
    assert "skill" in [t.name for t in tools_for_mode("default", skills=True)]
    assert "skill" in [t.name for t in tools_for_mode("plan", skills=True)]


# --------------------------------------------------------------------------- #
# Two sizes of the same skill
# --------------------------------------------------------------------------- #
# A 4B model with an 8k window and a frontier model behind an API key cannot be
# served the same page of instructions: what is a useful briefing for one is a
# third of the other's window. So a skill may ship `SKILL-mini.md` next to
# `SKILL.md`, and the menu that sits in *every* request has a short form too.

from localcode.agent.skills import (  # noqa: E402
    FULL,
    MAX_MINI_CHARS,
    MAX_MINI_DESCRIPTION_CHARS,
    MAX_MINI_MENU_CHARS,
    MINI,
    variant_for,
)


@pytest.mark.parametrize("strong, n_ctx, expected", [
    (False, 4096, MINI),
    (False, 8192, MINI),
    (False, 16_384, MINI),
    (False, 16_385, FULL),
    (False, 131_072, FULL),
    (False, 0, MINI),          # unknown window: assume the worst, not the best
    (True, 4096, FULL),        # behind an API key the window is never the problem
])
def test_the_variant_follows_the_window_not_the_guess(strong: bool, n_ctx: int,
                                                      expected: str) -> None:
    assert variant_for(strong=strong, n_ctx=n_ctx) == expected


def test_a_skill_without_a_mini_file_still_works(library: SkillLibrary,
                                                 tmp_path: Path) -> None:
    """Every skill written before this existed has to keep working unchanged."""
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "stary", body="Krótka treść.")
    library.reload()
    skill = library.get("stary")
    assert skill is not None and skill.mini_body == ""
    assert skill.body_for(MINI) == skill.body_for(FULL) == "Krótka treść."


def test_the_mini_file_is_served_to_a_small_model(library: SkillLibrary,
                                                  tmp_path: Path) -> None:
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "testy",
                body="Długi wykład o testach.", mini="Uruchom pytest.")
    library.reload()
    skill = library.get("testy")
    assert skill is not None
    assert skill.body_for(MINI) == "Uruchom pytest."
    assert skill.body_for(FULL) == "Długi wykład o testach."
    assert skill.to_dict()["hasMini"] is True


def test_a_header_in_the_mini_file_is_ignored(library: SkillLibrary, tmp_path: Path) -> None:
    """Name and description have one source, so the two cannot contradict."""
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "testy",
                mini="---\nname: co-innego\n---\n\nSama treść.")
    library.reload()
    skill = library.get("testy")
    assert skill is not None and skill.name == "testy"
    assert skill.body_for(MINI) == "Sama treść."


def test_a_long_skill_without_a_mini_is_cut_and_says_where_the_rest_is(
        library: SkillLibrary, tmp_path: Path) -> None:
    """Refusing would leave a small model with nothing; a usable opening is better."""
    body = "\n\n".join(f"Akapit {i}. " + "x" * 200 for i in range(40))
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "dlugi", body=body)
    library.reload()
    skill = library.get("dlugi")
    assert skill is not None
    mini = skill.body_for(MINI)
    assert mini.startswith("Akapit 0.")
    assert "SKILL-mini.md" in mini and str(skill.path) in mini
    text = mini.split("[skrócone")[0].rstrip()
    assert len(text) <= MAX_MINI_CHARS
    assert len(text) < len(skill.body) / 4                   # genuinely shorter
    assert text.endswith("x")                                # cut between paragraphs


def test_an_over_long_mini_file_is_cut(library: SkillLibrary, tmp_path: Path) -> None:
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "gadatliwy",
                mini="y" * (MAX_MINI_CHARS + 500))
    library.reload()
    skill = library.get("gadatliwy")
    assert skill is not None
    assert len(skill.mini_body) <= MAX_MINI_CHARS + 20       # the marker may add a little
    assert "skrócone" in skill.mini_body


def test_the_mini_file_is_not_listed_as_a_companion_file(library: SkillLibrary,
                                                         tmp_path: Path) -> None:
    folder = tmp_path / "projekt" / ".localcode" / "skills" / "zskryptem"
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "zskryptem", mini="Krótko.")
    (folder / "sprawdz.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    library.reload()
    skill = library.get("zskryptem")
    assert skill is not None
    assert skill.files() == ["sprawdz.sh"]


def test_a_folder_with_only_a_mini_file_is_skipped(library: SkillLibrary,
                                                   tmp_path: Path) -> None:
    folder = tmp_path / "projekt" / ".localcode" / "skills" / "sierota"
    folder.mkdir(parents=True)
    (folder / "SKILL-mini.md").write_text("Sama treść.\n", encoding="utf-8")
    library.reload()
    assert library.get("sierota") is None
    assert library.errors == []          # not an error, just not a skill


# --- descriptions ---------------------------------------------------------- #


def test_the_description_has_a_short_form_too(library: SkillLibrary, tmp_path: Path) -> None:
    """The menu is paid for on every request, so it is budgeted as well."""
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "testy",
                description="Użyj, gdy trzeba uruchomić albo napisać testy w tym repozytorium.",
                mini_description="Uruchom testy repo")
    library.reload()
    skill = library.get("testy")
    assert skill is not None
    assert skill.description_for(MINI) == "Uruchom testy repo"
    assert skill.description_for(FULL).startswith("Użyj, gdy")


def test_a_missing_short_description_falls_back_to_a_shortened_one(
        library: SkillLibrary, tmp_path: Path) -> None:
    long = "Użyj tego skilla zawsze wtedy, kiedy " + "bardzo długie wyjaśnienie " * 8
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "rozwlekly", description=long)
    library.reload()
    skill = library.get("rozwlekly")
    assert skill is not None
    short = skill.description_for(MINI)
    assert len(short) <= MAX_MINI_DESCRIPTION_CHARS
    assert short.endswith("…")
    assert short.startswith("Użyj tego skilla")


def test_the_menu_uses_the_short_descriptions_for_a_small_model(
        library: SkillLibrary, tmp_path: Path) -> None:
    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "testy",
                description="Bardzo długi opis, który w oknie 8k jest luksusem.",
                mini_description="Uruchom testy")
    library.reload()
    library.use_for(strong=False, n_ctx=8192)
    assert "- testy: Uruchom testy" in library.menu()
    library.use_for(strong=True, n_ctx=8192)
    assert "Bardzo długi opis" in library.menu()


def test_a_shelf_too_big_for_the_menu_degrades_to_bare_names(
        library: SkillLibrary, tmp_path: Path) -> None:
    """Forty skills × 200 characters would spend a third of an 8k window."""
    shelf = tmp_path / "projekt" / ".localcode" / "skills"
    for i in range(40):
        write_skill(shelf, f"skill-numer-{i:02d}",
                    description="Opis o długości, która w czterdziestu sztukach nie ma prawa się zmieścić.")
    library.reload()
    library.use_for(strong=False, n_ctx=8192)
    menu = library.menu()
    assert len(menu) <= MAX_MINI_MENU_CHARS + 600          # names only, still bounded
    assert menu.startswith("Skille: ")
    assert "skill-numer-00" in menu and "skill-numer-39" in menu
    # The full menu is untouched: a big window can afford the descriptions.
    library.use_for(strong=True, n_ctx=200_000)
    assert library.menu().startswith("- skill-numer-00: Opis")


def test_the_tool_serves_the_variant_the_library_is_set_to(library: SkillLibrary,
                                                           tmp_path: Path) -> None:
    import asyncio

    write_skill(tmp_path / "projekt" / ".localcode" / "skills", "testy",
                body="Pełny wykład.", mini="Krótko.")
    library.reload()
    ctx = ToolContext(cwd=str(tmp_path), limits=ToolLimits(), state={STATE_KEY: library})

    library.use_for(strong=False, n_ctx=8192)
    out = asyncio.run(SkillTool().run({"name": "testy"}, ctx))
    assert "Krótko." in out.output and "Pełny wykład." not in out.output

    library.use_for(strong=True, n_ctx=8192)
    out = asyncio.run(SkillTool().run({"name": "testy"}, ctx))
    assert "Pełny wykład." in out.output
