"""Slash-command parsing, the help table and ``/runtime set``."""

from __future__ import annotations

import pytest

from localcode.tui import commands
from localcode.tui.app import LocalCodeTui


@pytest.mark.parametrize(
    "line, name, arg",
    [
        ("/help", "help", ""),
        ("/model", "model", ""),
        ("  /model qwen3-4b ", "model", "qwen3-4b"),
        ("/mode plan", "mode", "plan"),
        ("/runtime set ram-gb 20", "runtime", "set ram-gb 20"),
        ("/download unsloth/Qwen3-4B-GGUF Q4_K_M", "download", "unsloth/Qwen3-4B-GGUF Q4_K_M"),
        ("/q", "exit", ""),          # alias
        ("/ls", "models", ""),       # alias
        ("/?", "help", ""),
    ],
)
def test_parse_command(line: str, name: str, arg: str) -> None:
    parsed = commands.parse_command(line)
    assert parsed is not None
    assert (parsed.name, parsed.arg) == (name, arg)
    assert parsed.known


@pytest.mark.parametrize("line", ["hej", "", "  ", "/ spacja", "//komentarz", "adres/sciezka"])
def test_not_a_command(line: str) -> None:
    assert commands.parse_command(line) is None
    assert not commands.is_command(line)


def test_unknown_command_is_parsed_but_not_known() -> None:
    parsed = commands.parse_command("/nieistnieje")
    assert parsed is not None and not parsed.known
    assert parsed.name == "nieistnieje"


def test_args_splits_the_argument() -> None:
    parsed = commands.parse_command("/runtime set ram-gb 20")
    assert parsed is not None
    assert parsed.args == ["set", "ram-gb", "20"]


def test_every_command_has_a_handler_on_the_app() -> None:
    """The completer must never offer something ``dispatch`` cannot run."""
    missing = [c.name for c in commands.COMMANDS if not hasattr(LocalCodeTui, f"_cmd_{c.name}")]
    assert missing == []


def test_help_table_lists_every_command(console) -> None:
    console.print(commands.help_table())
    out = console.export_text()
    for command in commands.COMMANDS:
        assert f"/{command.name}" in out


def test_chat_help_hides_agent_only_commands(console) -> None:
    console.print(commands.help_table(chat=True))
    out = console.export_text()
    # agent-only: session resuming and project bootstrapping make no sense in chat
    # ("/mode" would match "/model", so assert on unambiguous names)
    assert "/resume" not in out
    assert "/init" not in out
    assert "/model" in out
    # the chatbot compacts its context too (docs/ARCHITECTURE.md, chat/compaction.py)
    assert "/compact" in out


def test_keys_table_documents_the_shortcuts(console) -> None:
    console.print(commands.keys_table())
    out = console.export_text()
    assert "Shift+Tab" in out
    assert "Alt+Enter" in out
    assert "Ctrl+C" in out


def test_command_names_include_aliases() -> None:
    names = commands.command_names()
    assert "exit" in names and "quit" in names and "q" in names


def test_suggest_finds_a_typo() -> None:
    assert commands.suggest("modles") == "models"
    assert commands.suggest("zzzzzz") is None


# --------------------------------------------------------------------------- #
# /runtime set
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("max-models", "3", {"maxLoadedModels": 3}),
        ("instances", "2", {"maxInstancesPerModel": 2}),
        ("parallel", "4", {"defaultParallel": 4}),
        ("ram-gb", "20", {"ramLimitGb": 20.0}),
        ("ram-gb", "20,5", {"ramLimitGb": 20.5}),
        ("ram-gb", "auto", {"ramLimitGb": None}),
        ("idle-ttl", "0", {"idleTtlMinutes": 0}),
        ("ctx", "8192", {"defaultCtx": 8192}),
        ("ctx", "auto", {"defaultCtx": None}),
        ("kv", "q8_0", {"kvCacheType": "q8_0"}),
        ("flash-attn", "on", {"flashAttn": "on"}),
        ("evict", "none", {"evictPolicy": "none"}),
    ],
)
def test_runtime_patch(key: str, value: str, expected: dict) -> None:
    assert commands.runtime_patch(key, value) == expected


def test_runtime_patch_accepts_the_api_name() -> None:
    assert commands.runtime_patch("maxLoadedModels", "2") == {"maxLoadedModels": 2}


@pytest.mark.parametrize("key, value", [("nic-takiego", "1"), ("max-models", "dużo"),
                                        ("ram-gb", "bardzo")])
def test_runtime_patch_rejects_garbage(key: str, value: str) -> None:
    with pytest.raises(ValueError):
        commands.runtime_patch(key, value)


def test_runtime_keys_match_the_manager() -> None:
    """Every key we offer must be one ``RuntimeManager`` accepts."""
    from localcode.runtime.manager import _SETTING_KEYS

    unknown = [v for v in commands.RUNTIME_SETTING_KEYS.values() if v not in _SETTING_KEYS]
    assert unknown == []


def test_every_command_served_to_the_app_is_handled_there() -> None:
    """The server tells the Mac app which commands to offer; it must handle them.

    `/resume` was served and not handled, so the app answered "Nieznane
    polecenie" to a command it had just suggested. Read from the Swift source,
    because that is where the second half of this contract lives.
    """
    import re
    from pathlib import Path

    swift = (Path(__file__).resolve().parents[2] / "macos" / "Sources" / "CodeFeature"
             / "Store" / "AgentStore.swift").read_text(encoding="utf-8")
    handler = swift[swift.index("private func handleSlashCommand"):]
    handler = handler[:handler.index("\n    }\n")]
    handled = set(re.findall(r'case "([a-z_]+)"', handler))
    served = {c.name for c in commands.COMMANDS if c.in_app}
    assert served <= handled, f"brak obsługi w aplikacji: {sorted(served - handled)}"
