"""Tests for ``todo_write`` — the agent's plan, and the tool small models malform most."""

from __future__ import annotations

from pathlib import Path

import pytest

from localcode.agent.tools import get_tool
from localcode.agent.tools.todo import (
    MAX_TODOS,
    STATE_KEY,
    STATUSES,
    get_todos,
    normalize_status,
    normalize_todos,
    render_todos,
)
from tool_helpers import make_ctx, permission_of, run_tool

THREE = [
    {"content": "read the config", "status": "done"},
    {"content": "fix the parser", "status": "in_progress"},
    {"content": "run the tests", "status": "pending"},
]


async def test_writes_the_list_and_renders_it(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    result = await run_tool("todo_write", {"todos": THREE}, ctx)
    assert result.ok
    assert result.output == (
        "Todo list (1/3 done):\n"
        "[x] read the config\n"
        "[>] fix the parser\n"
        "[ ] run the tests"
    )
    assert ctx.state[STATE_KEY] == THREE
    assert get_todos(ctx) == THREE
    assert result.summary.startswith("Lista zadań: 1/3")
    assert "fix the parser" in result.summary


async def test_the_stored_list_is_a_copy(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    items = [dict(item) for item in THREE]
    await run_tool("todo_write", {"todos": items}, ctx)
    items[0]["content"] = "changed"
    assert ctx.state[STATE_KEY][0]["content"] == "read the config"
    stored = get_todos(ctx)
    stored[0]["content"] = "also changed"
    assert ctx.state[STATE_KEY][0]["content"] == "read the config"


async def test_rewriting_replaces_the_whole_list(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    await run_tool("todo_write", {"todos": THREE}, ctx)
    await run_tool("todo_write", {"todos": [{"content": "only this", "status": "pending"}]}, ctx)
    assert get_todos(ctx) == [{"content": "only this", "status": "pending"}]


async def test_an_empty_list_clears_the_todos(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    await run_tool("todo_write", {"todos": THREE}, ctx)
    result = await run_tool("todo_write", {"todos": []}, ctx)
    assert result.ok
    assert result.output == "Todo list cleared."
    assert get_todos(ctx) == []


async def test_several_in_progress_tasks_are_flagged(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    result = await run_tool("todo_write", {"todos": [
        {"content": "a", "status": "in_progress"},
        {"content": "b", "status": "in_progress"},
    ]}, ctx)
    assert result.ok
    assert "keep exactly one task in_progress" in result.output


async def test_missing_argument_is_reported(tmp_path: Path):
    result = await run_tool("todo_write", {}, make_ctx(tmp_path))
    assert not result.ok
    assert "todos" in result.output
    assert "Usage:" in result.output


async def test_unusable_entries_are_refused(tmp_path: Path):
    result = await run_tool("todo_write", {"todos": [{"foo": "bar"}]}, make_ctx(tmp_path))
    assert not result.ok
    assert "content" in result.output
    assert result.summary == "Zła lista zadań"


async def test_todo_write_needs_no_permission_even_in_plan_mode(tmp_path: Path):
    assert await permission_of("todo_write", {"todos": THREE}, make_ctx(tmp_path)) is None
    assert get_tool("todo_write").kind == "meta"


async def test_the_list_is_capped(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    result = await run_tool(
        "todo_write", {"todos": [{"content": f"task {i}", "status": "pending"} for i in range(60)]},
        ctx)
    assert result.ok
    assert len(get_todos(ctx)) == MAX_TODOS
    assert f"only the first {MAX_TODOS} tasks were kept" in result.output


async def test_long_content_is_clipped(tmp_path: Path):
    ctx = make_ctx(tmp_path)
    await run_tool("todo_write", {"todos": [{"content": "x" * 500, "status": "pending"}]}, ctx)
    assert len(get_todos(ctx)[0]["content"]) <= 200


def test_title_names_the_current_task():
    tool = get_tool("todo_write")
    assert tool.title({"todos": THREE}) == "Lista zadań: fix the parser"
    assert tool.title({"todos": [{"content": "a", "status": "done"}]}) == "Lista zadań (1/1)"
    assert tool.title({"todos": "nonsense that is not a list"}).startswith("Lista zadań")


# --------------------------------------------------------------------------- #
# Forgiving parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw, expected", [
    # the canonical shape
    ([{"content": "a", "status": "done"}], [{"content": "a", "status": "done"}]),
    # a bare list of strings
    (["a", "b"], [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
    # one string with newlines
    ("a\nb", [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]),
    # markdown checkboxes
    ("- [x] a\n- [ ] b\n- [~] c", [{"content": "a", "status": "done"},
                                   {"content": "b", "status": "pending"},
                                   {"content": "c", "status": "in_progress"}]),
    # bullets
    (["1. first", "* second"], [{"content": "first", "status": "pending"},
                                {"content": "second", "status": "pending"}]),
    # other key names
    ([{"task": "a", "state": "doing"}], [{"content": "a", "status": "in_progress"}]),
    ([{"text": "a", "completed": True}], [{"content": "a", "status": "done"}]),
    # a single object instead of a list
    ({"content": "a", "status": "pending"}, [{"content": "a", "status": "pending"}]),
    # a nested wrapper
    ({"todos": [{"content": "a", "status": "done"}]}, [{"content": "a", "status": "done"}]),
    # pairs
    ([["a", "done"]], [{"content": "a", "status": "done"}]),
    # status spellings
    ([{"content": "a", "status": "in-progress"}], [{"content": "a", "status": "in_progress"}]),
    ([{"content": "a", "status": "COMPLETED"}], [{"content": "a", "status": "done"}]),
    ([{"content": "a", "status": "TODO"}], [{"content": "a", "status": "pending"}]),
    ([{"content": "a", "status": "weird"}], [{"content": "a", "status": "pending"}]),
    # whitespace normalisation, empty entries dropped
    ([" a   b ", "", None], [{"content": "a b", "status": "pending"}]),
    # a trailing status word
    (["fix the parser — done"], [{"content": "fix the parser", "status": "done"}]),
    ([], []),
    (None, []),
])
def test_normalize_todos(raw, expected):
    assert normalize_todos(raw) == expected


@pytest.mark.parametrize("value, expected", [
    ("pending", "pending"), ("in_progress", "in_progress"), ("done", "done"),
    (True, "done"), (False, "pending"), (None, "pending"), ("wip", "in_progress"),
    ("Finished", "done"), ("", "pending"),
])
def test_normalize_status(value, expected):
    assert normalize_status(value) in STATUSES
    assert normalize_status(value) == expected


def test_render_todos():
    assert render_todos([]) == "Todo list is empty."
    assert render_todos([{"content": "a", "status": "in_progress"}]) == \
        "Todo list (0/1 done):\n[>] a"


def test_status_values_match_the_schema():
    tool = get_tool("todo_write")
    enum = tool.parameters["properties"]["todos"]["items"]["properties"]["status"]["enum"]
    assert tuple(enum) == STATUSES
