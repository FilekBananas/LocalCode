"""Tests for the file-system tools: read_file, write_file, list_dir.

They run against real temporary directories (the tools do real IO in worker
threads, and half the bugs live in the OS layer), and they check the *text the
model sees*: a wrong error message costs the agent a whole turn.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from localcode.agent.permissions import OUTSIDE_PROJECT_FLAG
from localcode.agent.tools import get_tool
from localcode.agent.tools.fs import (
    IgnoreRules,
    decode_bytes,
    display_path,
    file_key,
    find_placeholder,
    human_size,
    normalize_args,
    pl_plural,
    resolve_path,
    truncate_middle,
)
from tool_helpers import make_ctx, numbered, permission_of, run_tool, write_file


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    write_file(tmp_path / "src" / "app.py", "def main():\n    return 1\n")
    write_file(tmp_path / "README.md", "# Demo\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


async def test_read_file_numbers_lines_and_reports_the_total(project: Path):
    write_file(project / "a.txt", "alpha\nbeta\ngamma\n")
    result = await run_tool("read_file", {"path": "a.txt"}, make_ctx(project))
    assert result.ok
    assert result.output.startswith("File: a.txt (3 lines)")
    assert numbered(result.output) == [(1, "alpha"), (2, "beta"), (3, "gamma")]
    assert result.summary == "3 linie"


async def test_read_file_window_with_offset_and_limit(project: Path):
    write_file(project / "big.txt", "".join(f"line{i}\n" for i in range(1, 51)))
    result = await run_tool("read_file", {"path": "big.txt", "offset": 10, "limit": 5},
                            make_ctx(project))
    assert [n for n, _ in numbered(result.output)] == [10, 11, 12, 13, 14]
    assert "showing 10–14" in result.output
    assert "offset=15 to continue" in result.output
    assert result.summary == "50 linii · pokazano 10–14"


async def test_read_file_default_window_comes_from_the_limits(project: Path):
    write_file(project / "big.txt", "".join(f"line{i}\n" for i in range(1, 400)))
    ctx = make_ctx(project, read_default_lines=12)
    result = await run_tool("read_file", "big.txt", ctx)      # bare string argument
    assert [n for n, _ in numbered(result.output)] == list(range(1, 13))
    assert "[387 more lines" in result.output


async def test_read_file_negative_offset_reads_from_the_end(project: Path):
    write_file(project / "big.txt", "".join(f"line{i}\n" for i in range(1, 21)))
    result = await run_tool("read_file", {"path": "big.txt", "offset": -3}, make_ctx(project))
    assert [n for n, _ in numbered(result.output)] == [18, 19, 20]


async def test_read_file_offset_past_the_end_explains_the_range(project: Path):
    write_file(project / "a.txt", "one\ntwo\n")
    result = await run_tool("read_file", {"path": "a.txt", "offset": 99}, make_ctx(project))
    assert not result.ok
    assert "offset 99 is past the end" in result.output
    assert "which has 2 lines" in result.output


async def test_read_file_accepts_a_path_with_a_line_suffix(project: Path):
    write_file(project / "big.txt", "".join(f"line{i}\n" for i in range(1, 31)))
    result = await run_tool("read_file", {"path": "big.txt:5-7"}, make_ctx(project))
    assert [n for n, _ in numbered(result.output)] == [5, 6, 7]


async def test_read_file_keeps_unicode(project: Path):
    text = "zażółć gęślą jaźń\nемодзи: 🐍🚀\n"
    write_file(project / "pl.txt", text)
    result = await run_tool("read_file", "pl.txt", make_ctx(project))
    assert "zażółć gęślą jaźń" in result.output
    assert "🐍🚀" in result.output


async def test_read_file_normalises_crlf_line_endings(project: Path):
    write_file(project / "dos.txt", "alpha\nbeta\n", newline="\r\n")
    assert b"\r\n" in (project / "dos.txt").read_bytes()
    result = await run_tool("read_file", "dos.txt", make_ctx(project))
    assert numbered(result.output) == [(1, "alpha"), (2, "beta")]
    assert "\r" not in result.output


async def test_read_file_reads_cp1250_and_says_so(project: Path):
    (project / "old.txt").write_bytes("zażółć\n".encode("cp1250"))
    result = await run_tool("read_file", "old.txt", make_ctx(project))
    assert result.ok
    assert "zażółć" in result.output
    assert "encoding: cp1250" in result.output


async def test_read_file_clips_very_long_lines(project: Path):
    write_file(project / "long.txt", "x" * 5000 + "\nshort\n")
    ctx = make_ctx(project, max_output_chars=4000)
    result = await run_tool("read_file", "long.txt", ctx)
    assert "[line clipped, 5000 chars]" in result.output
    assert "were clipped" in result.output
    assert len(result.output) <= 4000


async def test_read_file_refuses_binary_files(project: Path):
    (project / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00\x01\x02" * 100)
    result = await run_tool("read_file", "logo.png", make_ctx(project))
    assert not result.ok
    assert "image" in result.output
    # A picture is named as one, and the model is pointed at the attachment it
    # can already see instead of trying again (a real Qwen3-VL run did that).
    assert "obraz" in result.summary.lower()
    assert "already see it" in result.output


async def test_read_file_detects_binary_content_without_an_extension(project: Path):
    (project / "blob").write_bytes(b"abc\x00\x00\x00def" * 50)
    result = await run_tool("read_file", "blob", make_ctx(project))
    assert not result.ok
    assert "binary" in result.output


async def test_read_file_missing_file_suggests_the_real_one(project: Path):
    # (a case-only difference would resolve on macOS, so misspell it properly)
    result = await run_tool("read_file", {"path": "src/aqp.py"}, make_ctx(project))
    assert not result.ok
    assert "not found" in result.output
    assert "src/app.py" in result.output
    assert result.summary == "Nie znaleziono pliku"


async def test_read_file_on_a_directory_points_at_list_dir(project: Path):
    result = await run_tool("read_file", "src", make_ctx(project))
    assert not result.ok
    assert "is a directory" in result.output
    assert "list_dir(path='src')" in result.output


async def test_read_file_empty_file(project: Path):
    write_file(project / "empty.txt", "")
    result = await run_tool("read_file", "empty.txt", make_ctx(project))
    assert result.ok
    assert "empty file" in result.output


async def test_read_file_records_the_mtime_for_stale_checks(project: Path):
    ctx = make_ctx(project)
    await run_tool("read_file", "src/app.py", ctx)
    key = file_key(project / "src" / "app.py")
    assert ctx.read_files[key] == pytest.approx((project / "src" / "app.py").stat().st_mtime)


async def test_read_file_output_never_exceeds_the_limit(project: Path):
    write_file(project / "huge.txt", "".join(f"{i} some text here\n" for i in range(5000)))
    ctx = make_ctx(project, max_output_chars=2000, read_default_lines=4000)
    result = await run_tool("read_file", "huge.txt", ctx)
    assert len(result.output) <= 2000


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #


async def test_write_file_creates_a_file_with_a_diff(project: Path):
    ctx = make_ctx(project)
    result = await run_tool("write_file", {"path": "new.txt", "content": "hello\n"}, ctx)
    assert result.ok
    assert (project / "new.txt").read_text() == "hello\n"
    assert "Created new.txt" in result.output
    assert result.diff is not None and "+hello" in result.diff
    assert "/dev/null" in result.diff


async def test_write_file_creates_missing_directories(project: Path):
    result = await run_tool("write_file", {"path": "a/b/c/deep.txt", "content": "x"},
                            make_ctx(project))
    assert result.ok
    assert (project / "a" / "b" / "c" / "deep.txt").read_text() == "x\n"


async def test_write_file_overwrites_and_counts_the_diff(project: Path):
    write_file(project / "f.txt", "one\ntwo\nthree\n")
    result = await run_tool("write_file", {"path": "f.txt", "content": "one\nTWO\nthree\n"},
                            make_ctx(project))
    assert result.ok
    assert "+1 −1" in result.output
    assert result.summary == "Zapisano (+1 −1)"
    assert (project / "f.txt").read_text() == "one\nTWO\nthree\n"


async def test_write_file_identical_content_changes_nothing(project: Path):
    write_file(project / "f.txt", "same\n")
    before = (project / "f.txt").stat().st_mtime_ns
    result = await run_tool("write_file", {"path": "f.txt", "content": "same\n"}, make_ctx(project))
    assert result.ok
    assert "No changes" in result.output
    assert (project / "f.txt").stat().st_mtime_ns == before


async def test_write_file_adds_a_trailing_newline(project: Path):
    await run_tool("write_file", {"path": "n.txt", "content": "no newline"}, make_ctx(project))
    assert (project / "n.txt").read_text() == "no newline\n"


async def test_write_file_keeps_crlf_files_crlf(project: Path):
    write_file(project / "dos.txt", "a\nb\n", newline="\r\n")
    await run_tool("write_file", {"path": "dos.txt", "content": "a\nB\n"}, make_ctx(project))
    assert (project / "dos.txt").read_bytes() == b"a\r\nB\r\n"


async def test_write_file_rejects_placeholder_content(project: Path):
    write_file(project / "src" / "big.py", "def a():\n    pass\n\ndef b():\n    pass\n" * 5)
    result = await run_tool(
        "write_file",
        {"path": "src/big.py", "content": "def a():\n    pass\n\n# ... rest of the code ...\n"},
        make_ctx(project))
    assert not result.ok
    assert "placeholder" in result.output
    assert "edit_file" in result.output
    assert "def b()" in (project / "src" / "big.py").read_text()      # nothing was written


async def test_write_file_missing_content_argument(project: Path):
    result = await run_tool("write_file", {"path": "x.txt"}, make_ctx(project))
    assert not result.ok
    assert "content" in result.output
    assert "Usage:" in result.output


async def test_write_file_warns_about_a_new_syntax_error(project: Path):
    write_file(project / "src" / "ok.py", "x = 1\n")
    result = await run_tool("write_file", {"path": "src/ok.py", "content": "def broken(\n"},
                            make_ctx(project))
    assert result.ok
    assert "syntax error" in result.output


async def test_write_file_permission_shows_the_diff(project: Path):
    ctx = make_ctx(project)
    info = await permission_of("write_file", {"path": "src/app.py", "content": "x = 1\n"}, ctx)
    assert info is not None
    assert info.title.startswith("Zapis src/app.py")
    assert "-def main():" in info.preview


async def test_write_file_permission_flags_paths_outside_the_project(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    outside = tmp_path / "elsewhere.txt"
    info = await permission_of("write_file", {"path": str(outside), "content": "x"},
                               make_ctx(project))
    assert info is not None
    assert OUTSIDE_PROJECT_FLAG in info.title


async def test_write_file_permission_is_none_when_nothing_changes(project: Path):
    write_file(project / "f.txt", "same\n")
    ctx = make_ctx(project)
    assert await permission_of("write_file", {"path": "f.txt", "content": "same\n"}, ctx) is None


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #


async def test_list_dir_shows_a_tree_with_sizes(project: Path):
    result = await run_tool("list_dir", {}, make_ctx(project))
    assert result.ok
    assert result.output.startswith("Directory: .")
    assert "src/" in result.output
    assert "README.md (" in result.output
    assert "app.py" in result.output           # depth 2 by default


async def test_list_dir_depth_one_collapses_subdirectories(project: Path):
    result = await run_tool("list_dir", {"path": ".", "depth": 1}, make_ctx(project))
    assert "src/ (1 entry)" in result.output
    assert "app.py" not in result.output


async def test_list_dir_skips_ignored_directories(project: Path):
    (project / "node_modules" / "pkg").mkdir(parents=True)
    write_file(project / "node_modules" / "pkg" / "index.js", "x")
    (project / ".git").mkdir()
    result = await run_tool("list_dir", {}, make_ctx(project))
    assert "node_modules" not in result.output.split("[not listed:")[0]
    assert "not listed" in result.output


async def test_list_dir_respects_gitignore(project: Path):
    write_file(project / ".gitignore", "build/\n*.log\n")
    (project / "build").mkdir()
    write_file(project / "build" / "out.o", "x")
    write_file(project / "debug.log", "x")
    write_file(project / "keep.txt", "x")
    result = await run_tool("list_dir", {}, make_ctx(project))
    body = result.output.split("[not listed:")[0]
    assert "keep.txt" in body
    assert "debug.log" not in body
    assert "build/" not in body


async def test_list_dir_honours_the_entry_limit(project: Path):
    for i in range(80):
        write_file(project / f"f{i:03}.txt", "x")
    ctx = make_ctx(project, list_max_entries=20)
    result = await run_tool("list_dir", {}, ctx)
    assert result.output.count(".txt (") <= 20
    assert "truncated" in result.output or "more entries" in result.output


async def test_list_dir_on_a_file_points_at_read_file(project: Path):
    result = await run_tool("list_dir", "README.md", make_ctx(project))
    assert not result.ok
    assert "is a file" in result.output
    assert "read_file(path='README.md')" in result.output


async def test_list_dir_missing_directory(project: Path):
    result = await run_tool("list_dir", "nope", make_ctx(project))
    assert not result.ok
    assert "Directory not found" in result.output


async def test_list_dir_shows_symlinks_without_following_them(project: Path):
    (project / "link").symlink_to(project / "src")
    result = await run_tool("list_dir", {"depth": 1}, make_ctx(project))
    assert "link/ ->" in result.output


async def test_list_dir_empty_directory(project: Path):
    (project / "empty").mkdir()
    result = await run_tool("list_dir", "empty", make_ctx(project))
    assert "(empty directory)" in result.output


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw, expected", [
    ({"file_path": "a.py"}, {"path": "a.py"}),
    ({"FilePath": "a.py"}, {"path": "a.py"}),
    ({"arguments": {"path": "a.py"}}, {"path": "a.py"}),
    ('{"path": "a.py"}', {"path": "a.py"}),
    ("a.py", {"path": "a.py"}),
    (None, {}),
    ({"path": "a.py", "file": "b.py"}, {"path": "a.py"}),     # canonical key wins
])
def test_normalize_args_accepts_what_small_models_produce(raw, expected):
    assert normalize_args(raw, {"file_path": "path", "file": "path", "filepath": "path"}, "path") == expected


@pytest.mark.parametrize("raw", ["'src/app.py'", '"src/app.py"', "`src/app.py`",
                                 "file:///PROJECT/src/app.py", "@src/app.py", "/src/app.py"])
def test_resolve_path_cleans_up_model_decorations(project: Path, raw: str):
    ctx = make_ctx(project)
    raw = raw.replace("/PROJECT", str(project))
    assert resolve_path(raw, ctx) == project / "src" / "app.py"


def test_display_path_is_relative_inside_the_project(project: Path):
    ctx = make_ctx(project)
    assert display_path(project / "src" / "app.py", ctx) == "src/app.py"
    assert display_path(Path.home() / "x.txt", ctx) == "~/x.txt"


@pytest.mark.parametrize("n, expected", [(1, "linia"), (2, "linie"), (5, "linii"),
                                         (22, "linie"), (12, "linii"), (101, "linii")])
def test_polish_plurals(n: int, expected: str):
    assert pl_plural(n, "linia", "linie", "linii") == expected


def test_truncate_middle_keeps_head_and_tail():
    text = "\n".join(f"line {i}" for i in range(500))
    out = truncate_middle(text, 400, "use offset")
    assert out.startswith("line 0")
    assert out.rstrip().endswith("line 499")
    assert "lines omitted — use offset" in out
    assert len(out) <= 400 + 200


def test_decode_bytes_handles_boms_and_fallbacks():
    assert decode_bytes("x".encode("utf-8-sig")) == ("x", "utf-8-sig")
    assert decode_bytes("x".encode("utf-16")) == ("x", "utf-16")
    assert decode_bytes(b"plain") == ("plain", "utf-8")
    assert decode_bytes("ą".encode("cp1250")) == ("ą", "cp1250")


@pytest.mark.parametrize("text, found", [
    ("# ... rest of the code ...", True),
    ("// existing code unchanged", True),
    ("    # keep the rest ...", True),
    ("rest = of_the_code(x)", False),
    ("# the rest of the code is in utils.py, see there", False),
])
def test_find_placeholder(text: str, found: bool):
    assert (find_placeholder(text) is not None) is found


def test_human_size():
    assert human_size(512) == "512B"
    assert human_size(2048) == "2.0K"
    assert human_size(5 * 1024 * 1024) == "5.0M"


def test_ignore_rules_negation(tmp_path: Path):
    write_file(tmp_path / ".gitignore", "*.log\n!keep.log\n")
    rules = IgnoreRules(tmp_path)
    assert rules.ignored(str(tmp_path / "a.log"), False)
    assert not rules.ignored(str(tmp_path / "keep.log"), False)
    assert rules.ignored(str(tmp_path / "node_modules"), True)


async def test_tools_never_raise_out_of_run(project: Path):
    """``run`` must turn every failure into a ToolResult (the turn must survive)."""
    ctx = make_ctx(project)
    for name, args in [("read_file", {}), ("write_file", {"content": "x"}),
                       ("list_dir", {"path": "\x00"}), ("read_file", {"path": 12}),
                       ("edit_file", {"path": "src/app.py"}), ("grep", {}), ("glob", {}),
                       ("bash", {}), ("todo_write", {})]:
        result = await run_tool(name, args, ctx)
        assert not result.ok, name
        assert result.output.startswith("Error:"), name
        assert result.summary, name


async def test_read_only_tools_do_not_ask_for_permission(project: Path):
    ctx = make_ctx(project)
    for name, args in [("read_file", {"path": "README.md"}), ("list_dir", {}),
                       ("glob", {"pattern": "*.md"}), ("grep", {"pattern": "Demo"})]:
        assert await permission_of(name, args, ctx) is None, name


def test_every_tool_has_a_polish_title():
    ctx_args = {
        "read_file": {"path": "src/app.py"},
        "write_file": {"path": "src/app.py", "content": "x"},
        "edit_file": {"path": "src/app.py", "old_string": "a", "new_string": "b"},
        "list_dir": {"path": "src"},
        "glob": {"pattern": "*.py"},
        "grep": {"pattern": "def"},
        "bash": {"command": "npm test"},
        "todo_write": {"todos": [{"content": "x", "status": "pending"}]},
    }
    expected = {"Odczyt", "Zapis", "Edycja", "Lista", "Szukanie", "Polecenie", "Lista zadań"}
    for name, args in ctx_args.items():
        tool = get_tool(name)
        assert tool is not None
        title = tool.title(args)
        assert any(title.startswith(word) for word in expected), (name, title)


def test_file_key_is_absolute_and_normalised(project: Path):
    assert file_key(Path(os.path.join(project, "src", "..", "src", "app.py"))) == \
        str(project / "src" / "app.py")
