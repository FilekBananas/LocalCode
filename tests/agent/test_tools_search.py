"""Tests for the search tools: ``glob`` and ``grep``.

``rg`` is a shell *function* on the development machine, so ``shutil.which("rg")``
finds nothing and the pure-Python engine is what actually runs. The engine tests
therefore force ``find_ripgrep() -> None`` to stay deterministic wherever they
run, and a separate test pins the detection rule itself.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from localcode.agent.tools import search as search_mod
from localcode.agent.tools.search import _normalize_includes, _probe_ripgrep, find_ripgrep
from tool_helpers import make_ctx, run_tool, write_file


@pytest.fixture
def no_ripgrep(monkeypatch):
    """Force the pure-Python engine (the only one available on this machine)."""
    monkeypatch.setattr(search_mod, "find_ripgrep", lambda: None)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_file(tmp_path / "src" / "app.py", "import os\n\n\ndef main():\n    return 1\n")
    write_file(tmp_path / "src" / "util.py", "def helper():\n    return main()\n")
    write_file(tmp_path / "src" / "ui" / "view.tsx", "export const View = () => null;\n")
    write_file(tmp_path / "tests" / "test_app.py", "def test_main():\n    assert main() == 1\n")
    write_file(tmp_path / "README.md", "# Demo\nRun `main`.\n")
    write_file(tmp_path / ".gitignore", "ignored/\n*.log\n")
    write_file(tmp_path / "ignored" / "hidden.py", "def main():\n    return 'secret'\n")
    write_file(tmp_path / "debug.log", "def main(): pass\n")
    write_file(tmp_path / "node_modules" / "dep" / "index.js", "function main() {}\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# glob
# --------------------------------------------------------------------------- #


async def test_glob_finds_files_at_any_depth(project: Path):
    result = await run_tool("glob", "**/*.py", make_ctx(project))
    assert result.ok
    found = {line for line in result.output.splitlines() if line.endswith(".py")}
    assert found == {"src/app.py", "src/util.py", "tests/test_app.py"}
    assert "[3 files, newest first]" in result.output


async def test_glob_without_a_slash_matches_the_name_anywhere(project: Path):
    result = await run_tool("glob", {"pattern": "*.tsx"}, make_ctx(project))
    assert "src/ui/view.tsx" in result.output


async def test_glob_expands_braces(project: Path):
    result = await run_tool("glob", {"pattern": "**/*.{py,tsx}"}, make_ctx(project))
    assert "src/ui/view.tsx" in result.output
    assert "src/app.py" in result.output


async def test_glob_honours_the_path_argument(project: Path):
    result = await run_tool("glob", {"pattern": "*.py", "path": "tests"}, make_ctx(project))
    assert "tests/test_app.py" in result.output
    assert "src/app.py" not in result.output


async def test_glob_skips_ignored_and_gitignored_files(project: Path):
    result = await run_tool("glob", "**/*", make_ctx(project))
    assert "node_modules" not in result.output
    assert "ignored/hidden.py" not in result.output
    assert "debug.log" not in result.output


async def test_glob_sorts_newest_first(project: Path):
    newest = project / "src" / "newest.py"
    write_file(newest, "x = 1\n")
    os.utime(newest, (time.time() + 60, time.time() + 60))
    result = await run_tool("glob", "**/*.py", make_ctx(project))
    assert result.output.splitlines()[0] == "src/newest.py"


async def test_glob_reports_no_matches_helpfully(project: Path):
    result = await run_tool("glob", "**/*.rs", make_ctx(project))
    assert result.ok
    assert "No files match" in result.output
    assert result.summary == "Brak plików"


async def test_glob_limits_the_number_of_results(project: Path):
    for i in range(40):
        write_file(project / "many" / f"f{i:02}.py", "x\n")
    result = await run_tool("glob", "**/*.py", make_ctx(project, list_max_entries=10))
    assert "showing the 10 newest of" in result.output


async def test_glob_requires_a_pattern(project: Path):
    result = await run_tool("glob", {"path": "src"}, make_ctx(project))
    assert not result.ok
    assert "pattern" in result.output


async def test_glob_on_a_file_path_is_an_error(project: Path):
    result = await run_tool("glob", {"pattern": "*.py", "path": "README.md"}, make_ctx(project))
    assert not result.ok
    assert "is a file, not a directory" in result.output


# --------------------------------------------------------------------------- #
# grep
# --------------------------------------------------------------------------- #


async def test_grep_finds_matches_with_path_and_line(project: Path, no_ripgrep):
    result = await run_tool("grep", "def main", make_ctx(project))
    assert result.ok
    assert "src/app.py:4: def main():" in result.output
    assert "[1 match in 1 file]" in result.output
    assert result.summary == "1 wynik w 1 pliku"


async def test_grep_is_a_regex(project: Path, no_ripgrep):
    result = await run_tool("grep", {"pattern": r"def \w+\("}, make_ctx(project))
    assert "src/app.py:4:" in result.output
    assert "src/util.py:1:" in result.output


async def test_grep_ignore_case(project: Path, no_ripgrep):
    plain = await run_tool("grep", {"pattern": "DEMO"}, make_ctx(project))
    assert "No matches" in plain.output
    assert "ignore_case=true" in plain.output
    insensitive = await run_tool("grep", {"pattern": "DEMO", "ignore_case": True}, make_ctx(project))
    assert "README.md:1:" in insensitive.output


async def test_grep_include_filter(project: Path, no_ripgrep):
    result = await run_tool("grep", {"pattern": "main", "include": "*.md"}, make_ctx(project))
    assert "README.md:2:" in result.output
    assert "src/app.py" not in result.output


@pytest.mark.parametrize("include, expected", [
    ("py", ["*.py"]), (".py", ["*.py"]), ("*.py", ["*.py"]),
    ("*.ts,*.tsx", ["*.ts", "*.tsx"]), (["*.py", "*.md"], ["*.py", "*.md"]),
    ("*.{ts,tsx}", ["*.{ts,tsx}"]),
])
def test_normalize_includes(include, expected):
    assert _normalize_includes(include) == expected


async def test_grep_context_lines(project: Path, no_ripgrep):
    result = await run_tool("grep", {"pattern": "return 1", "path": "src/app.py", "context": 1},
                            make_ctx(project))
    assert "src/app.py-4- def main():" in result.output
    assert "src/app.py:5:     return 1" in result.output


async def test_grep_skips_ignored_files(project: Path, no_ripgrep):
    result = await run_tool("grep", "main", make_ctx(project))
    assert "node_modules" not in result.output
    assert "ignored/hidden.py" not in result.output
    assert "debug.log" not in result.output


async def test_grep_skips_binary_files(project: Path, no_ripgrep):
    (project / "blob.bin").write_bytes(b"main\x00\x00\x00main" * 40)
    result = await run_tool("grep", "main", make_ctx(project))
    assert "blob.bin" not in result.output


async def test_grep_invalid_regex_falls_back_to_a_literal_search(project: Path, no_ripgrep):
    write_file(project / "weird.txt", "call foo( here\n")
    result = await run_tool("grep", "foo(", make_ctx(project))
    assert result.ok
    assert "weird.txt:1:" in result.output
    assert "invalid regex" in result.output
    assert "searched as literal text" in result.output


async def test_grep_reads_bre_alternation(project: Path, no_ripgrep):
    result = await run_tool("grep", r"helper\|View", make_ctx(project))
    assert "src/util.py:1:" in result.output
    assert "src/ui/view.tsx:1:" in result.output
    assert "alternation" in result.output


async def test_grep_limits_the_matches_and_says_so(project: Path, no_ripgrep):
    write_file(project / "many.txt", "".join(f"hit {i}\n" for i in range(50)))
    result = await run_tool("grep", {"pattern": "hit", "path": "many.txt"},
                            make_ctx(project, grep_max_matches=5))
    assert result.output.count("many.txt:") == 5
    assert "showing 5 of 50 matches" in result.output


async def test_grep_clips_very_long_lines(project: Path, no_ripgrep):
    write_file(project / "long.txt", "x" * 3000 + "NEEDLE" + "y" * 3000 + "\n")
    result = await run_tool("grep", {"pattern": "NEEDLE", "path": "long.txt"}, make_ctx(project))
    assert "NEEDLE" in result.output
    assert len(max(result.output.splitlines(), key=len)) < 400


async def test_grep_on_a_single_file(project: Path, no_ripgrep):
    result = await run_tool("grep", {"pattern": "main", "path": "src/util.py"}, make_ctx(project))
    assert "src/util.py:2:" in result.output
    assert "src/app.py" not in result.output


async def test_grep_missing_path(project: Path, no_ripgrep):
    result = await run_tool("grep", {"pattern": "x", "path": "nope"}, make_ctx(project))
    assert not result.ok
    assert "Path not found" in result.output


async def test_grep_requires_a_pattern(project: Path, no_ripgrep):
    result = await run_tool("grep", {"path": "src"}, make_ctx(project))
    assert not result.ok
    assert "pattern" in result.output


async def test_grep_output_respects_the_char_limit(project: Path, no_ripgrep):
    write_file(project / "many.txt", "".join(f"needle {i} " + "z" * 100 + "\n" for i in range(200)))
    ctx = make_ctx(project, max_output_chars=1500, grep_max_matches=200)
    result = await run_tool("grep", {"pattern": "needle", "path": "many.txt"}, ctx)
    assert len(result.output) <= 1500


async def test_grep_uses_the_python_engine_when_there_is_no_ripgrep(project: Path, monkeypatch):
    """Sanity check that nothing shells out when ``rg`` is missing."""
    import asyncio

    async def fail(*args, **kwargs):      # pragma: no cover - only runs on a regression
        raise AssertionError("grep must not spawn a subprocess without ripgrep")

    monkeypatch.setattr(search_mod, "find_ripgrep", lambda: None)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    result = await run_tool("grep", "def main", make_ctx(project))
    assert result.ok
    assert "src/app.py:4:" in result.output


def test_find_ripgrep_only_accepts_a_real_binary():
    """`rg` as a shell function is invisible to ``which`` — that is the normal case here."""
    found = find_ripgrep()
    on_path = shutil.which("rg")
    if found is None:
        assert on_path is None or not _probe_ripgrep(on_path)
    else:
        assert os.path.isfile(found) and os.access(found, os.X_OK)


def test_find_ripgrep_rejects_a_fake_binary(tmp_path: Path, monkeypatch):
    fake = tmp_path / "rg"
    fake.write_text("#!/bin/sh\necho not-ripgrep\n")
    fake.chmod(0o755)
    monkeypatch.setenv("LOCALCODE_RIPGREP", str(fake))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert find_ripgrep() is None
