"""Tests for ``edit_file`` — the matching cascade that has to survive small models.

Each stage of the cascade (exact → whitespace → spacing regex → fuzzy) gets its
own case, plus the two things that must NEVER happen: guessing between several
candidate places, and silently editing a file that changed on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localcode.agent.tools.edit import (
    EditError,
    detect_indent,
    plan_edit,
    strip_line_numbers,
)
from localcode.agent.tools.fs import file_key
from tool_helpers import make_ctx, permission_of, run_tool, write_file

SAMPLE = """\
class Service:
    def __init__(self, url):
        self.url = url

    def fetch(self, path):
        response = get(self.url + path)
        return response.json()
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_file(tmp_path / "svc.py", SAMPLE)
    return tmp_path


async def edit(project: Path, args: dict, ctx=None):
    return await run_tool("edit_file", args, ctx or make_ctx(project))


# --------------------------------------------------------------------------- #
# Stage 1 — exact
# --------------------------------------------------------------------------- #


async def test_exact_unique_replacement(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "self.url = url",
                                  "new_string": "self.url = url.rstrip('/')"})
    assert result.ok
    assert "self.url = url.rstrip('/')" in (project / "svc.py").read_text()
    assert result.output.startswith("Edited svc.py (line 3)")
    assert result.diff is not None and "+        self.url = url.rstrip('/')" in result.diff
    assert result.summary == "+1 −1"


async def test_ambiguous_match_is_refused_and_explained(project: Path):
    write_file(project / "dup.py", "x = 1\ny = 2\nx = 1\n")
    result = await edit(project, {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 9"})
    assert not result.ok
    assert "matches 2 places" in result.output
    assert "lines 1, 3" in result.output
    assert "replace_all=true" in result.output
    assert (project / "dup.py").read_text() == "x = 1\ny = 2\nx = 1\n"     # untouched


async def test_replace_all_changes_every_occurrence(project: Path):
    write_file(project / "dup.py", "x = 1\ny = 2\nx = 1\n")
    result = await edit(project, {"path": "dup.py", "old_string": "x = 1", "new_string": "x = 9",
                                  "replace_all": True})
    assert result.ok
    assert (project / "dup.py").read_text() == "x = 9\ny = 2\nx = 9\n"
    assert "replaced 2 occurrences" in result.output
    assert "wystąpienia" in result.summary


async def test_identical_strings_are_refused(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "self.url", "new_string": "self.url"})
    assert not result.ok
    assert "identical" in result.output


# --------------------------------------------------------------------------- #
# Stage 2 — whitespace tolerant, with re-indentation
# --------------------------------------------------------------------------- #


async def test_trailing_whitespace_and_tabs_still_match(project: Path):
    write_file(project / "t.py", "def f():\n\tif x:\n\t\treturn 1\n")
    result = await edit(project, {"path": "t.py",
                                  "old_string": "    if x:   \n        return 1",
                                  "new_string": "    if x:\n        return 2"})
    assert result.ok
    assert (project / "t.py").read_text() == "def f():\n\tif x:\n\t\treturn 2\n"
    assert "different whitespace" in result.output


async def test_new_string_is_reindented_to_the_file(project: Path):
    """The model writes 4-space blocks; the file uses 8 (or tabs)."""
    write_file(project / "deep.py", "class A:\n    def go(self):\n        a = 1\n        b = 2\n")
    result = await edit(project, {
        "path": "deep.py",
        "old_string": "def go(self):\n    a = 1\n    b = 2",
        "new_string": "def go(self):\n    a = 1\n    if a:\n        a = 3\n    b = 2",
    })
    assert result.ok
    assert (project / "deep.py").read_text() == (
        "class A:\n    def go(self):\n        a = 1\n        if a:\n            a = 3\n        b = 2\n")
    assert "re-indented" in result.output


async def test_line_number_prefixes_from_read_file_are_stripped(project: Path):
    old = "    5\t    def fetch(self, path):\n    6\t        response = get(self.url + path)"
    new = "    5\t    def fetch(self, path):\n    6\t        response = get(self.url + path, timeout=5)"
    result = await edit(project, {"path": "svc.py", "old_string": old, "new_string": new})
    assert result.ok
    assert "timeout=5" in (project / "svc.py").read_text()
    assert "line-number prefixes" in result.output
    assert "\t" not in (project / "svc.py").read_text()


async def test_literal_backslash_n_is_understood(project: Path):
    result = await edit(project, {"path": "svc.py",
                                  "old_string": "        self.url = url\\n",
                                  "new_string": "        self.url = url.strip()\\n"})
    assert result.ok
    assert "url.strip()" in (project / "svc.py").read_text()


# --------------------------------------------------------------------------- #
# Stage 3 — spacing-insensitive regex
# --------------------------------------------------------------------------- #


async def test_intra_line_spacing_differences_match(project: Path):
    write_file(project / "s.py", "total = a+b  +  c\n")
    result = await edit(project, {"path": "s.py", "old_string": "a+b + c",
                                  "new_string": "a + b + c"})
    assert result.ok
    assert (project / "s.py").read_text() == "total = a + b + c\n"
    assert "ignoring differences in spacing" in result.output


# --------------------------------------------------------------------------- #
# Stage 4 — fuzzy
# --------------------------------------------------------------------------- #


async def test_fuzzy_match_applies_with_a_warning(project: Path):
    result = await edit(project, {
        "path": "svc.py",
        # one word misremembered ("resp" instead of "response") on the second line
        "old_string": "    def fetch(self, path):\n        resp = get(self.url + path)\n"
                      "        return response.json()",
        "new_string": "    def fetch(self, path):\n        resp = get(self.url + path)\n"
                      "        return resp.json()",
    })
    assert result.ok
    assert "return resp.json()" in (project / "svc.py").read_text()
    assert "did not match exactly" in result.output
    assert "% similar" in result.output


async def test_fuzzy_refuses_when_several_places_are_similar(tmp_path: Path):
    block = "def handler(request):\n    data = load(request)\n    return render(data)\n"
    write_file(tmp_path / "h.py", block + "\n\n" + block.replace("handler", "handler2"))
    result = await edit(tmp_path, {
        "path": "h.py",
        "old_string": "def handlerX(request):\n    data = load(request)\n    return render(data)",
        "new_string": "def handlerX(request):\n    return render(load(request))",
    })
    assert not result.ok
    assert "several places are similar" in result.output or "not found" in result.output


async def test_short_old_strings_are_not_fuzzy_matched(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "self.rul",
                                  "new_string": "self.url"})
    assert not result.ok
    assert "was not found" in result.output


# --------------------------------------------------------------------------- #
# Failure quality
# --------------------------------------------------------------------------- #


async def test_not_found_shows_the_closest_region_with_line_numbers(project: Path):
    result = await edit(project, {
        "path": "svc.py",
        "old_string": "    def fetch(self, path, timeout=30):\n        response = post(self.url)\n"
                      "        return response.text",
        "new_string": "    pass",
    })
    assert not result.ok
    assert "old_string was not found in svc.py" in result.output
    assert "most similar text" in result.output
    assert "\t    def fetch(self, path):" in result.output       # numbered snippet
    assert result.summary == "Nie znaleziono tekstu do zamiany"


async def test_missing_arguments_are_named(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "x"})
    assert not result.ok
    assert "new_string" in result.output
    assert "Usage:" in result.output


async def test_empty_old_string_on_a_non_empty_file_is_refused(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "", "new_string": "x = 1\n"})
    assert not result.ok
    assert "write_file" in result.output
    assert result.summary == "Pusty old_string"


async def test_edit_creates_a_new_file_when_old_string_is_empty(project: Path):
    result = await edit(project, {"path": "fresh.py", "old_string": "", "new_string": "x = 1"})
    assert result.ok
    assert (project / "fresh.py").read_text() == "x = 1\n"
    assert "Created fresh.py" in result.output


async def test_edit_on_a_missing_file_with_content_is_an_error(project: Path):
    result = await edit(project, {"path": "nope.py", "old_string": "x", "new_string": "y"})
    assert not result.ok
    assert "not found" in result.output


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


async def test_stale_file_is_edited_but_flagged(project: Path):
    ctx = make_ctx(project)
    await run_tool("read_file", "svc.py", ctx)
    ctx.read_files[file_key(project / "svc.py")] -= 60        # pretend it changed since
    result = await edit(project, {"path": "svc.py", "old_string": "self.url = url",
                                  "new_string": "self.url = url + '/'"}, ctx)
    assert result.ok
    assert "had changed on disk" in result.output


async def test_stale_warning_is_added_to_a_failed_match(project: Path):
    ctx = make_ctx(project)
    await run_tool("read_file", "svc.py", ctx)
    ctx.read_files[file_key(project / "svc.py")] -= 60
    result = await edit(project, {"path": "svc.py", "old_string": "zzz not here zzz",
                                  "new_string": "y"}, ctx)
    assert not result.ok
    assert "changed on disk since you last read it" in result.output


async def test_edit_warns_about_a_new_syntax_error(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "        return response.json()",
                                  "new_string": "        return response.json("})
    assert result.ok
    assert "syntax error" in result.output


async def test_edit_keeps_crlf_and_encoding(tmp_path: Path):
    write_file(tmp_path / "dos.py", "a = 1\nb = 2\n", newline="\r\n")
    result = await edit(tmp_path, {"path": "dos.py", "old_string": "b = 2", "new_string": "b = 3"})
    assert result.ok
    assert (tmp_path / "dos.py").read_bytes() == b"a = 1\r\nb = 3\r\n"


async def test_edit_permission_previews_the_diff(project: Path):
    info = await permission_of("edit_file", {"path": "svc.py", "old_string": "self.url = url",
                                             "new_string": "self.url = url2"}, make_ctx(project))
    assert info is not None
    assert info.title == "Edycja svc.py"
    assert "-        self.url = url" in info.preview
    assert "+        self.url = url2" in info.preview


async def test_edit_permission_is_none_for_an_impossible_edit(project: Path):
    info = await permission_of("edit_file", {"path": "svc.py", "old_string": "nothing like this",
                                             "new_string": "x"}, make_ctx(project))
    assert info is None


async def test_permission_then_a_changed_file_is_refused(project: Path):
    """permission() said "nothing to approve"; the file must not be written anyway."""
    ctx = make_ctx(project)
    args = {"path": "svc.py", "old_string": "not there yet", "new_string": "x = 1"}
    assert await permission_of("edit_file", args, ctx) is None
    write_file(project / "svc.py", SAMPLE + "not there yet\n")
    result = await edit(project, args, ctx)
    assert not result.ok
    assert "changed while this edit was being checked" in result.output


async def test_edit_result_shows_the_new_text_with_line_numbers(project: Path):
    result = await edit(project, {"path": "svc.py", "old_string": "self.url = url",
                                  "new_string": "self.url = url.rstrip('/')"})
    assert "Result:" in result.output
    assert "\t        self.url = url.rstrip('/')" in result.output


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text, expected", [
    ("    1\tfirst\n    2\tsecond", "first\nsecond"),
    ("12: alpha\n13: beta", "alpha\nbeta"),
    ("1\tone", "one"),
    ("not numbered\nat all", None),
    ("    1\tfirst\n    7\tjumped", None),        # non-consecutive numbers
])
def test_strip_line_numbers(text: str, expected):
    assert strip_line_numbers(text) == expected


def test_detect_indent():
    spaces = "def f():\n    a = 1\n    if a:\n        b = 2\n".split("\n")
    assert detect_indent(spaces).unit == 4
    assert detect_indent(spaces).tabs is False
    tabs = "def f():\n\ta = 1\n\tif a:\n\t\tb = 2\n".split("\n")
    assert detect_indent(tabs).tabs is True


def test_plan_edit_is_pure():
    text = "a\nb\nc\n"
    plan = plan_edit(text, "b", "B")
    assert plan.new_text == "a\nB\nc\n"
    assert plan.method == "exact"
    assert plan.count == 1
    with pytest.raises(EditError):
        plan_edit(text, "zzz", "x")


async def test_editing_a_file_that_does_not_exist_says_to_create_it(tmp_path) -> None:
    """The generic "use list_dir or glob" sent small models wandering.

    Reproduced with a 0,6B model asked to write a new file: it tried ``edit_file``,
    was told to look for the path, listed the same empty directory twice and gave
    up without writing anything. The error has to name the tool that works.
    """
    result = await run_tool("edit_file",
                            {"path": "hello.txt", "old_string": "x", "new_string": "banan"},
                            make_ctx(tmp_path))

    assert result.ok is False
    assert "write_file" in result.output
    assert "does not exist yet" in result.output
    assert result.summary == "Nie ma takiego pliku"


async def test_a_mistyped_path_still_gets_its_suggestion(tmp_path) -> None:
    write_file(tmp_path / "hello.txt", "banan\n")

    result = await run_tool("edit_file",
                            {"path": "Hello.TXT", "old_string": "banan", "new_string": "gruszka"},
                            make_ctx(tmp_path))

    # either it resolved the case difference, or it told the model the real name
    assert result.ok is True or "hello.txt" in result.output
