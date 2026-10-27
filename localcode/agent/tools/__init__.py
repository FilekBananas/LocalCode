"""The coding agent's tool belt and its registry.

Eight tools, no more: ``read_file``, ``write_file``, ``edit_file``, ``list_dir``,
``glob``, ``grep``, ``bash`` and ``todo_write``. Every schema is re-sent on every
request, so the menu is short on purpose — a 4k-context model spends a tenth of
its window on the tool definitions alone, and a long menu also makes small models
pick the wrong tool.

Two knobs shape the menu:

``mode``
    The permission mode (``agent/permissions.py``). ``plan`` is read-only: the
    model gets the four exploration tools plus ``todo_write`` (writing the plan
    is what plan mode is for), and never sees ``bash``, ``write_file`` or
    ``edit_file`` — a tool the model cannot see is a tool it cannot try.
``tiny``
    For windows of ~8k tokens or less: ``glob`` and ``todo_write`` are dropped
    (``grep`` and ``list_dir`` cover finding files; a model this small should be
    doing the task, not keeping a list) and the remaining schemas lose their
    per-parameter prose and their rarest options (:data:`TINY_DROP_PARAMS`),
    which the tiny system prompt covers instead. That takes the default menu
    from ~1100 to under 700 estimated tokens (chars/3.5).

Everything here is cheap to build (tools are stateless), so callers get fresh
instances and may mutate them freely::

    tools = tools_for_mode(settings.agent.permission_mode, tiny=n_ctx <= 8192)
    schemas = [t.schema() for t in tools]
    tool = get_tool(call.name)
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Sequence

from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolLimits, ToolResult
from localcode.agent.tools.edit import EditFileTool
from localcode.agent.tools.fs import ListDirTool, ReadFileTool, ToolInputError, WriteFileTool
from localcode.agent.tools.search import GlobTool, GrepTool
from localcode.agent.tools.shell import BashTool
from localcode.agent.tools.models import (
    MODEL_TOOL_CLASSES,
    ModelInstallTool,
    ModelListTool,
    ModelLoadTool,
    ModelOps,
    ModelSearchTool,
)
from localcode.agent.tools.skill import SkillTool
from localcode.agent.tools.subagent import SubagentTool
from localcode.agent.tools.todo import TodoWriteTool

__all__ = [
    "BashTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ListDirTool",
    "MODEL_TOOL_CLASSES",
    "ModelInstallTool",
    "ModelListTool",
    "ModelLoadTool",
    "ModelOps",
    "ModelSearchTool",
    "PermissionInfo",
    "READ_ONLY_TOOLS",
    "ReadFileTool",
    "SkillTool",
    "SubagentTool",
    "TINY_DROP_PARAMS",
    "TINY_OMIT",
    "TOOL_ALIASES",
    "TOOL_CLASSES",
    "TOOL_NAMES",
    "TodoWriteTool",
    "Tool",
    "ToolContext",
    "ToolInputError",
    "ToolLimits",
    "ToolResult",
    "WriteFileTool",
    "default_tools",
    "get_tool",
    "compact_parameters",
    "make_tool",
    "tool_names",
    "tool_schemas",
    "tools_for_mode",
]

# Order matters: it is the order the model sees, and small models favour the
# first entries. Reading comes before writing, shell last.
TOOL_CLASSES: tuple[type[Tool], ...] = (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
    GlobTool,
    GrepTool,
    BashTool,
    TodoWriteTool,
)

TOOL_NAMES: tuple[str, ...] = tuple(cls.name for cls in TOOL_CLASSES)

#: Dropped in tiny mode — ``grep`` plus ``list_dir`` find files well enough, and
#: a bookkeeping tool is a trap for a model this small: asked for a web page, a
#: 1-bit 30B model answered with ``<todo_write>`` thirty-two times and nothing
#: else. Plan mode keeps it, because writing the plan is the whole point there.
TINY_OMIT: frozenset[str] = frozenset({"glob", "todo_write"})

#: Tools the ``plan`` mode offers (read-only exploration + the plan itself).
READ_ONLY_TOOLS: tuple[str, ...] = tuple(
    cls.name for cls in TOOL_CLASSES if cls.kind in ("read", "meta")
)

_BY_NAME: dict[str, type[Tool]] = {cls.name: cls
                                  for cls in (*TOOL_CLASSES, *MODEL_TOOL_CLASSES, SkillTool,
                                              SubagentTool)}

# Names models reach for that mean one of ours. Recovering here is much cheaper
# than a wasted turn explaining that "run_command" does not exist.
TOOL_ALIASES: dict[str, str] = {
    "read": "read_file", "readfile": "read_file", "open_file": "read_file", "cat": "read_file",
    "view_file": "read_file", "view": "read_file", "get_file": "read_file", "file_read": "read_file",
    "write": "write_file", "writefile": "write_file", "create_file": "write_file",
    "save_file": "write_file", "new_file": "write_file", "file_write": "write_file",
    "edit": "edit_file", "editfile": "edit_file", "replace": "edit_file", "patch": "edit_file",
    "str_replace": "edit_file", "str_replace_editor": "edit_file", "apply_patch": "edit_file",
    "search_replace": "edit_file", "modify_file": "edit_file", "update_file": "edit_file",
    "ls": "list_dir", "listdir": "list_dir", "list_directory": "list_dir", "list_files": "list_dir",
    "dir": "list_dir", "tree": "list_dir", "list": "list_dir",
    "find": "glob", "find_files": "glob", "file_search": "glob", "glob_file_search": "glob",
    "search_models": "model_search", "find_model": "model_search", "list_models": "model_list",
    "download_model": "model_install", "install_model": "model_install",
    "load_model": "model_load", "switch_model": "model_load", "use_model": "model_load",
    "use_skill": "skill", "load_skill": "skill", "read_skill": "skill", "skills": "skill",
    "task": "delegate", "subagent": "delegate", "sub_agent": "delegate",
    "spawn_agent": "delegate", "run_agent": "delegate", "helper": "delegate",
    "search": "grep", "search_files": "grep", "grep_search": "grep", "rg": "grep",
    "ripgrep": "grep", "code_search": "grep", "content_search": "grep",
    "shell": "bash", "sh": "bash", "run": "bash", "run_command": "bash", "run_shell": "bash",
    "execute": "bash", "exec": "bash", "terminal": "bash", "command": "bash",
    "run_terminal_cmd": "bash", "shell_command": "bash",
    "todo": "todo_write", "todos": "todo_write", "todowrite": "todo_write",
    "task_list": "todo_write", "update_todos": "todo_write", "write_todos": "todo_write",
    "plan": "todo_write",
    "goal_complete": "goal_done", "complete_goal": "goal_done", "finish_goal": "goal_done",
    "mark_goal_done": "goal_done", "goal_achieved": "goal_done", "goal_reached": "goal_done",
}


# --------------------------------------------------------------------------- #
# Schema compaction (tiny windows)
# --------------------------------------------------------------------------- #


def _strip_descriptions(node: Any) -> Any:
    """Drop prose from a JSON-schema node, keeping types, enums and requirements."""
    if isinstance(node, dict):
        out = {k: _strip_descriptions(v) for k, v in node.items() if k != "description"}
        return out
    if isinstance(node, list):
        return [_strip_descriptions(v) for v in node]
    return node


#: Optional parameters not advertised in tiny mode. The tools still accept them
#: (argument parsing is independent of the schema) — they are simply not worth a
#: line of schema in an 8k window, and the tools' error messages mention them
#: when they matter (``edit_file`` suggests ``replace_all`` on an ambiguous match).
TINY_DROP_PARAMS: dict[str, frozenset[str]] = {
    "edit_file": frozenset({"replace_all"}),
    "list_dir": frozenset({"depth"}),
    "grep": frozenset({"ignore_case", "context"}),
    "bash": frozenset({"timeout"}),
}


def compact_parameters(parameters: dict[str, Any], drop: Iterable[str] = ()) -> dict[str, Any]:
    """The tiny-mode schema: no per-property prose, no rarely used options.

    The property *names* and types stay, so the model still knows the call shape;
    what the arguments mean is explained once in the tiny system prompt instead
    of once per request in every schema. Required parameters are never dropped.
    """
    out = copy.deepcopy(parameters)
    properties = out.get("properties")
    if not isinstance(properties, dict):
        return out
    required = set(out.get("required") or ())
    omit = {name for name in drop if name not in required}
    out["properties"] = {k: _strip_descriptions(v) for k, v in properties.items() if k not in omit}
    return out


# --------------------------------------------------------------------------- #
# Building tools
# --------------------------------------------------------------------------- #


def make_tool(cls: type[Tool], *, tiny: bool = False) -> Tool:
    """One tool instance; with ``tiny`` its schema is the compact variant."""
    tool = cls()
    if tiny:
        tool.parameters = compact_parameters(cls.parameters, TINY_DROP_PARAMS.get(cls.name, ()))
    return tool


def default_tools(tiny: bool = False) -> list[Tool]:
    """Every tool the agent has (minus :data:`TINY_OMIT` when ``tiny``)."""
    return [make_tool(cls, tiny=tiny) for cls in TOOL_CLASSES
            if not (tiny and cls.name in TINY_OMIT)]


def tools_for_mode(mode: str, tiny: bool = False, *, manage_models: bool = False,
                   skills: bool = False, subagents: bool = False) -> list[Tool]:
    """The tools offered in ``mode``; ``plan`` gets read-only tools + ``todo_write``.

    ``manage_models`` adds the four model-management tools
    (:mod:`localcode.agent.tools.models`).  They are off by default because
    they download gigabytes and take memory, and because four more schemas is a
    real cost in a 4k-token window — a model that cannot see them cannot spend
    context on them either.

    Unknown modes raise ``ValueError`` with a Polish message
    (``permissions.normalize_mode``).
    """
    from localcode.agent.permissions import normalize_mode  # lazy: permissions imports us back

    normalized = normalize_mode(mode)
    tools = default_tools(tiny=tiny)
    if skills:
        # Only offered when there is at least one skill to open; an empty shelf
        # would be one more schema for a small model to trip over.
        tools.append(make_tool(SkillTool, tiny=tiny))
    if manage_models:
        tools += [make_tool(cls, tiny=tiny) for cls in MODEL_TOOL_CLASSES]
    if subagents:
        # Off by default (``settings.agent.subagents``): a helper is a second
        # model turn nobody asked for, and on a tiny window the round trip
        # rarely pays for itself.
        tools.append(make_tool(SubagentTool, tiny=tiny))
    if normalized == "plan":
        keep = set(READ_ONLY_TOOLS) | {ModelSearchTool.name, ModelListTool.name, SkillTool.name,
                                       SubagentTool.name}
        planning = [t for t in tools if t.name in keep]
        if tiny and not any(t.name == TodoWriteTool.name for t in planning):
            planning.append(make_tool(TodoWriteTool, tiny=tiny))   # tiny drops it elsewhere
        return planning
    return tools


def get_tool(name: str, *, tiny: bool = False) -> Tool | None:
    """The tool called ``name`` (aliases and sloppy casing accepted), or ``None``."""
    if not name:
        return None
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = key.rsplit(".", 1)[-1]                 # "functions.read_file"
    cls = _BY_NAME.get(key) or _BY_NAME.get(TOOL_ALIASES.get(key, ""))
    return make_tool(cls, tiny=tiny) if cls is not None else None


def tool_names(tools: Iterable[Tool]) -> list[str]:
    """Names of ``tools`` — what ``prompts.build_system_prompt`` wants."""
    return [t.name for t in tools]


def tool_schemas(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    """OpenAI-style schemas, ready for ``ChatFormatter.render``."""
    return [t.schema() for t in tools]
