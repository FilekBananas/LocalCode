"""``bash``'s optional ``description``: what the command does, on the card.

The words are the model's, so they are for the reader only: the permission
card and the tool card show them, and nothing that decides anything reads them
— the policy judges the command from the arguments.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from localcode.agent.permissions import (
    DANGER_FLAG,
    OUTSIDE_PROJECT_FLAG,
    PermissionPolicy,
    _command_of,
)
from localcode.agent.tools import tools_for_mode
from localcode.agent.tools.base import ToolContext
from localcode.agent.tools.shell import BashTool


def permission(args: dict, cwd: Path):
    return asyncio.run(BashTool().permission(args, ToolContext(cwd=cwd)))


def test_the_schema_addition_is_one_optional_string() -> None:
    schema = BashTool().schema()["function"]["parameters"]
    assert schema["properties"]["description"]["type"] == "string"
    assert "user's language" in schema["properties"]["description"]["description"]
    assert schema["required"] == ["command"]
    tiny = next(t for t in tools_for_mode("default", tiny=True) if t.name == "bash")
    assert tiny.parameters["properties"]["description"] == {"type": "string"}, "no prose when tiny"


def test_the_card_title_uses_the_description_when_there_is_one() -> None:
    tool = BashTool()
    assert tool.title({"command": "npm test", "description": "Uruchamia testy."}) == "Polecenie: Uruchamia testy"
    assert tool.title({"command": "npm test"}) == "Polecenie: npm test"
    assert tool.title({"command": "npm test", "description": "   "}) == "Polecenie: npm test"


def test_the_preview_says_what_it_does_then_the_command(tmp_path: Path) -> None:
    info = permission({"command": "npm run build", "description": "Buduje aplikację"}, tmp_path)
    assert info.preview == "Co robi: Buduje aplikację\n\nnpm run build"
    assert info.title == "Polecenie: Buduje aplikację"
    plain = permission({"command": "npm run build"}, tmp_path)
    assert plain.preview == "npm run build" and plain.title == "Polecenie: npm run build"


def test_the_outside_note_stays_after_the_command_and_the_flags_stay_on_the_title(tmp_path: Path) -> None:
    info = permission({"command": "cat ~/.ssh/id_rsa", "description": "Czyta klucz"}, tmp_path)
    does, command, note = info.preview.split("\n\n", 2)
    assert does == "Co robi: Czyta klucz" and command == "cat ~/.ssh/id_rsa"
    assert note.startswith("To polecenie sięga poza folder projektu")
    assert info.title == f"Polecenie: Czyta klucz {OUTSIDE_PROJECT_FLAG}"
    danger = permission({"command": "rm -rf ~", "description": "Sprząta"}, tmp_path)
    assert danger.title.startswith(f"Polecenie: Sprząta {DANGER_FLAG}")


def test_the_policy_reads_the_command_not_the_description(tmp_path: Path) -> None:
    tool = BashTool()
    policy = PermissionPolicy("default", cwd=str(tmp_path))
    harmless = {"command": "ls -la", "description": "Usuwa wszystko: rm -rf /"}
    info = permission(harmless, tmp_path)
    assert _command_of(tool, harmless, info) == "ls -la"
    assert policy.decide(tool, harmless, info) == "allow", "read-only stays read-only"
    assert not policy.needs_human(tool, harmless, info)

    lying = {"command": "rm -rf ~", "description": "Wyświetla listę plików"}
    info = permission(lying, tmp_path)
    assert _command_of(tool, lying, info) == "rm -rf ~"
    assert policy.decide(tool, lying, info) == "ask"
    assert policy.needs_human(tool, lying, info), "a nice description does not hide a catastrophe"
    assert PermissionPolicy("yolo", cwd=str(tmp_path)).decide(tool, lying, info) == "ask"


def test_remembered_commands_ignore_the_description(tmp_path: Path) -> None:
    tool = BashTool()
    policy = PermissionPolicy("default", cwd=str(tmp_path))
    first = {"command": "npm test", "description": "Testy"}
    policy.remember_always(tool, first)
    other = {"command": "npm test", "description": "Coś zupełnie innego"}
    assert policy.decide(tool, other, permission(other, tmp_path)) == "allow"
