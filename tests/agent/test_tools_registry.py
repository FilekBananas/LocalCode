"""Tests for the tool registry (``localcode/agent/tools/__init__.py``).

Two things are contracts for everybody else: the set of tool *names* the model
sees, and what the schemas cost. The token budget test is the important one — a
tiny model pays for every schema on every request, so schema bloat is a bug, not
a style issue.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from localcode.agent.tools import (
    READ_ONLY_TOOLS,
    TINY_DROP_PARAMS,
    TINY_OMIT,
    TOOL_CLASSES,
    TOOL_NAMES,
    compact_parameters,
    default_tools,
    get_tool,
    tool_names,
    tool_schemas,
    tools_for_mode,
)
from localcode.agent.tools.base import Tool

EXPECTED_NAMES = ("read_file", "write_file", "edit_file", "list_dir", "glob", "grep",
                  "bash", "todo_write")

JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}

#: The estimate the context manager uses for a schema block: characters / 3.5.
CHARS_PER_TOKEN = 3.5

#: A tiny model (≤8k) must not spend more than this on the tool menu.
TINY_SCHEMA_BUDGET = 700
#: Even with every tool the menu stays modest.
FULL_SCHEMA_BUDGET = 1200


def schema_tokens(tools: list[Tool]) -> float:
    return len(json.dumps(tool_schemas(tools), ensure_ascii=False)) / CHARS_PER_TOKEN


# --------------------------------------------------------------------------- #
# The menu
# --------------------------------------------------------------------------- #


def test_the_tool_names_are_the_agreed_ones():
    assert TOOL_NAMES == EXPECTED_NAMES
    assert len(TOOL_CLASSES) == len(set(TOOL_NAMES))


def test_default_tools():
    assert tool_names(default_tools()) == list(EXPECTED_NAMES)


def test_tiny_mode_drops_glob_and_the_todo_list():
    """A tiny model gets the tools that do work, not the ones that keep notes.

    ``glob`` goes because ``grep``+``list_dir`` cover it; ``todo_write`` goes
    because a model this small trips over it — one 1-bit build answered a request
    for a web page with ``<todo_write>`` thirty-two times.
    """
    names = tool_names(default_tools(tiny=True))
    assert set(EXPECTED_NAMES) - set(names) == TINY_OMIT == {"glob", "todo_write"}
    assert names == [n for n in EXPECTED_NAMES if n not in TINY_OMIT]


def test_a_tiny_model_in_plan_mode_still_gets_the_todo_list():
    """Plan mode exists to produce a plan, so there the list is the point."""
    assert "todo_write" in tool_names(tools_for_mode("plan", tiny=True))


@pytest.mark.parametrize("mode", ["default", "accept_edits", "yolo"])
def test_writing_modes_get_every_tool(mode: str):
    assert tool_names(tools_for_mode(mode)) == list(EXPECTED_NAMES)


def test_plan_mode_is_read_only_plus_todo():
    names = tool_names(tools_for_mode("plan"))
    assert names == ["read_file", "list_dir", "glob", "grep", "todo_write"]
    assert set(names) == set(READ_ONLY_TOOLS)
    for tool in tools_for_mode("plan"):
        assert tool.kind in ("read", "meta"), tool.name


def test_plan_mode_hides_the_dangerous_tools():
    names = tool_names(tools_for_mode("plan"))
    for hidden in ("bash", "write_file", "edit_file"):
        assert hidden not in names


def test_plan_mode_tiny():
    assert tool_names(tools_for_mode("plan", tiny=True)) == \
        ["read_file", "list_dir", "grep", "todo_write"]


@pytest.mark.parametrize("alias, canonical", [
    ("default", "default"), ("normal", "default"), ("accept-edits", "accept_edits"),
    ("yolo", "yolo"), ("readonly", "plan"), ("PLAN", "plan"),
])
def test_mode_aliases_are_accepted(alias: str, canonical: str):
    assert tool_names(tools_for_mode(alias)) == tool_names(tools_for_mode(canonical))


def test_unknown_mode_raises_with_a_polish_message():
    with pytest.raises(ValueError) as excinfo:
        tools_for_mode("banana")
    assert "Nieznany tryb" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# get_tool
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_get_tool_by_name(name: str):
    tool = get_tool(name)
    assert tool is not None and tool.name == name


@pytest.mark.parametrize("asked, expected", [
    ("Bash", "bash"), ("run_command", "bash"), ("shell", "bash"), ("READ", "read_file"),
    ("read-file", "read_file"), ("str_replace_editor", "edit_file"), ("ls", "list_dir"),
    ("search", "grep"), ("find", "glob"), ("todo", "todo_write"), ("  grep  ", "grep"),
    ("functions.read_file", "read_file"),
])
def test_get_tool_recovers_from_near_misses(asked: str, expected: str):
    tool = get_tool(asked)
    assert tool is not None and tool.name == expected


@pytest.mark.parametrize("name", ["", "nope", "delete_everything", "read_fil"])
def test_get_tool_returns_none_for_unknown_names(name: str):
    assert get_tool(name) is None


def test_tools_are_independent_instances():
    a, b = get_tool("read_file"), get_tool("read_file")
    assert a is not b
    a.parameters = {"type": "object", "properties": {}}
    assert "path" in b.parameters["properties"]
    assert "path" in get_tool("read_file").parameters["properties"]


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def check_schema(schema: dict[str, Any]) -> None:
    """A structural JSON-Schema / OpenAI-function-schema check."""
    assert json.loads(json.dumps(schema, ensure_ascii=False)) == schema, "must be JSON round-trippable"
    assert set(schema) == {"type", "function"}
    assert schema["type"] == "function"
    function = schema["function"]
    assert set(function) == {"name", "description", "parameters"}
    assert isinstance(function["name"], str) and function["name"].isidentifier()
    assert len(function["name"]) <= 64
    assert isinstance(function["description"], str) and function["description"].strip()
    assert len(function["description"]) <= 200, "tool descriptions are paid for on every request"

    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert set(parameters) <= {"type", "properties", "required", "additionalProperties"}
    properties = parameters["properties"]
    assert isinstance(properties, dict) and properties
    for name, spec in properties.items():
        assert isinstance(name, str) and name and name.isidentifier(), name
        assert isinstance(spec, dict), name
        assert spec.get("type") in JSON_TYPES, (name, spec)
        assert set(spec) <= {"type", "description", "enum", "items", "properties", "required"}, name
        if "description" in spec:
            assert isinstance(spec["description"], str) and spec["description"].strip()
            assert len(spec["description"]) <= 100, name
        if spec["type"] == "array":
            items = spec.get("items")
            assert isinstance(items, dict) and items.get("type") in JSON_TYPES, name
            if items["type"] == "object":
                assert isinstance(items.get("properties"), dict) and items["properties"]
                assert set(items.get("required", [])) <= set(items["properties"])
        if "enum" in spec:
            assert isinstance(spec["enum"], list) and spec["enum"]
    required = parameters.get("required", [])
    assert isinstance(required, list)
    assert set(required) <= set(properties), "required names a parameter that does not exist"


@pytest.mark.parametrize("tiny", [False, True])
def test_every_tool_schema_is_valid(tiny: bool):
    for tool in default_tools(tiny=tiny):
        check_schema(tool.schema())


def test_tool_metadata_is_complete():
    for tool in default_tools():
        assert tool.name and tool.kind in ("read", "write", "exec", "meta"), tool.name
        assert tool.description.strip(), tool.name
        assert tool.usage and tool.name in tool.usage, tool.name


def test_kinds_match_the_permission_model():
    kinds = {t.name: t.kind for t in default_tools()}
    assert kinds == {
        "read_file": "read", "list_dir": "read", "glob": "read", "grep": "read",
        "write_file": "write", "edit_file": "write",
        "bash": "exec", "todo_write": "meta",
    }


def test_tiny_schemas_are_a_subset_of_the_full_ones():
    full = {t.name: t for t in default_tools()}
    for tool in default_tools(tiny=True):
        big = full[tool.name].parameters
        assert set(tool.parameters["properties"]) <= set(big["properties"]), tool.name
        assert tool.parameters.get("required") == big.get("required"), tool.name
        dropped = set(big["properties"]) - set(tool.parameters["properties"])
        assert dropped == set(TINY_DROP_PARAMS.get(tool.name, ())), tool.name
        for spec in tool.parameters["properties"].values():
            assert "description" not in spec


def test_compact_parameters_never_drops_a_required_field():
    parameters = {"type": "object",
                  "properties": {"a": {"type": "string", "description": "x"},
                                 "b": {"type": "integer", "description": "y"}},
                  "required": ["a"]}
    out = compact_parameters(parameters, ("a", "b"))
    assert set(out["properties"]) == {"a"}
    assert "description" not in out["properties"]["a"]
    assert parameters["properties"]["a"]["description"] == "x", "the input must not be mutated"


def test_tiny_mode_does_not_change_the_class_level_schema():
    default_tools(tiny=True)
    assert "description" in get_tool("read_file").parameters["properties"]["path"]
    assert "replace_all" in get_tool("edit_file").parameters["properties"]


# --------------------------------------------------------------------------- #
# The token budget (the reason the menu is short)
# --------------------------------------------------------------------------- #


def test_tiny_schemas_fit_a_small_window():
    cost = schema_tokens(tools_for_mode("default", tiny=True))
    assert cost < TINY_SCHEMA_BUDGET, f"tiny tool schemas cost ~{cost:.0f} tokens"


def test_full_schemas_stay_reasonable():
    cost = schema_tokens(tools_for_mode("default"))
    assert cost < FULL_SCHEMA_BUDGET, f"tool schemas cost ~{cost:.0f} tokens"


def test_tiny_is_meaningfully_cheaper():
    assert schema_tokens(tools_for_mode("default", tiny=True)) < \
        schema_tokens(tools_for_mode("default")) * 0.75


def test_plan_mode_is_the_cheapest_menu():
    assert schema_tokens(tools_for_mode("plan", tiny=True)) < \
        schema_tokens(tools_for_mode("default", tiny=True))


# --------------------------------------------------------------------------- #
# Integration points
# --------------------------------------------------------------------------- #


def test_titles_never_raise_on_junk_arguments():
    junk = [None, {}, "", [], {"path": 12}, {"weird": object()}, "not json {"]
    for tool in default_tools():
        for args in junk:
            assert isinstance(tool.title(args), str) and tool.title(args)


def test_the_system_prompt_knows_these_tool_names():
    from localcode.agent import prompts

    prompt = prompts.build_system_prompt(
        cwd="/tmp/x", n_ctx=8192, mode="default", native_tools=True,
        tool_names=list(TOOL_NAMES), project_instructions=None,
        env={"os": "macOS", "date": "2026-01-01"})
    assert "read_file" in prompt
    assert "edit_file" in prompt
    assert "bash" in prompt

    plan_prompt = prompts.build_system_prompt(
        cwd="/tmp/x", n_ctx=32768, mode="plan", native_tools=True,
        tool_names=tool_names(tools_for_mode("plan")), project_instructions=None, env={})
    assert "bash" not in plan_prompt
    assert "read_file" in plan_prompt


def test_permissions_can_look_up_every_tool_kind():
    from localcode.agent.permissions import _tool_kind

    for name in TOOL_NAMES:
        assert _tool_kind(name) == get_tool(name).kind
