"""Command-line parsing and dispatch (no models are ever loaded here)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from localcode import cli
from localcode.config import VERSION


# --------------------------------------------------------------------------- #
# Root parser
# --------------------------------------------------------------------------- #


def parse(*args: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(args))


def test_bare_invocation_has_sane_defaults() -> None:
    options = parse()
    assert options.prompt == []
    assert options.model is None
    assert options.mode is None
    assert options.ctx is None
    assert options.continue_last is False
    assert options.resume is None
    assert options.print_prompt is None
    assert options.output_format == "text"
    assert (options.local, options.attach) == (False, False)


def test_free_form_prompt() -> None:
    assert parse("napraw", "testy").prompt == ["napraw", "testy"]


def test_model_mode_and_context() -> None:
    options = parse("-m", "qwen3-4b", "--mode", "accept_edits", "--ctx", "8192")
    assert (options.model, options.mode, options.ctx) == ("qwen3-4b", "accept_edits", 8192)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse("--mode", "wszystko-wolno")


def test_continue_and_resume() -> None:
    assert parse("-c").continue_last is True
    assert parse("--resume").resume == ""
    assert parse("--resume", "20260923-141503-a1b2c3").resume == "20260923-141503-a1b2c3"


def test_headless_flags() -> None:
    options = parse("-p", "powiedz cześć", "--output-format", "json")
    assert options.print_prompt == "powiedz cześć"
    assert options.output_format == "json"


def test_headless_without_a_value_reads_the_positional() -> None:
    assert parse("-p").print_prompt == ""


def test_local_and_attach_are_exclusive() -> None:
    assert parse("--local").local is True
    assert parse("--attach").attach is True
    with pytest.raises(SystemExit):
        parse("--local", "--attach")


def test_cwd_flag() -> None:
    assert parse("-C", "/tmp").cwd == "/tmp"


def test_prefer_mapping() -> None:
    assert cli._prefer(parse()) == "auto"
    assert cli._prefer(parse("--local")) == "local"
    assert cli._prefer(parse("--attach")) == "remote"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert VERSION in capsys.readouterr().out


def test_help_lists_the_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    for name in ("models", "runtime", "serve", "doctor"):
        assert name in out


# --------------------------------------------------------------------------- #
# Subcommand parsers
# --------------------------------------------------------------------------- #


def test_models_list_is_the_default_action() -> None:
    assert cli.build_models_parser().parse_args([]).action is None


def test_models_search() -> None:
    options = cli.build_models_parser().parse_args(["search", "qwen", "coder", "--format", "gguf"])
    assert options.action == "search"
    assert options.query == ["qwen", "coder"]
    assert options.fmt == "gguf"


def test_models_download_with_quant() -> None:
    options = cli.build_models_parser().parse_args(["download", "unsloth/Qwen3-4B-GGUF", "Q4_K_M"])
    assert (options.repo, options.quant) == ("unsloth/Qwen3-4B-GGUF", "Q4_K_M")


def test_models_download_without_quant() -> None:
    assert cli.build_models_parser().parse_args(["download", "org/repo"]).quant is None


def test_models_rm_and_info() -> None:
    parser = cli.build_models_parser()
    assert parser.parse_args(["rm", "a/b/c", "-y"]).yes is True
    assert parser.parse_args(["info", "a/b"]).repo == "a/b"


def test_models_curated() -> None:
    assert cli.build_models_parser().parse_args(["curated"]).action == "curated"


def test_runtime_set() -> None:
    options = cli.build_runtime_parser().parse_args(["set", "ram-gb", "20"])
    assert (options.action, options.key, options.value) == ("set", "ram-gb", ["20"])


def test_runtime_show_is_the_default() -> None:
    assert cli.build_runtime_parser().parse_args([]).action is None


def test_subcommand_list_matches_the_dispatcher() -> None:
    """Everything in ``SUBCOMMANDS`` must be routable, and vice versa.

    The routed set is read out of the dispatcher's own source rather than
    copied here: a hand-kept copy only records what somebody remembered to
    update, which is exactly what this test exists to catch.
    """
    import inspect
    import re

    source = inspect.getsource(cli)
    # Two shapes of routing: the subcommand dispatcher (`if name == "x"`) and
    # the early branches in `main` that answer before a backend is opened
    # (`if args and args[0] == "serve"`).
    routed = set(re.findall(r'if name == "([a-z-]+)"', source))
    routed |= set(re.findall(r'args\[0\] == "([a-z-]+)"', source))
    routed |= {name for group in re.findall(r'args\[0\] in \(([^)]*)\)', source)
               for name in re.findall(r'"([a-z-]+)"', group)}
    routed -= {"--help", "-h"}          # flags, not commands
    assert routed, "nie znaleziono rozgałęzień dyspozytora"
    assert set(cli.SUBCOMMANDS) == routed


# --------------------------------------------------------------------------- #
# serve / app / doctor delegation
# --------------------------------------------------------------------------- #


def test_serve_delegates_to_the_server_module(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    import localcode.server.main as server_main

    monkeypatch.setattr(server_main, "serve_main", lambda argv: (seen.append(list(argv)), 7)[1])
    assert cli.main(["serve", "--port", "0"]) == 7
    assert seen == [["serve", "--port", "0"]]


def test_doctor_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import localcode.doctor as doctor

    monkeypatch.setattr(doctor, "main", lambda argv: 3)
    assert cli.main(["doctor"]) == 3


def test_app_without_a_bundle(monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "find_mac_app", lambda: None)
    assert cli.main(["app"]) == 1
    assert "build-app.sh" in capsys.readouterr().err


def test_app_opens_the_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "LocalCode.app"
    bundle.mkdir()
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "find_mac_app", lambda: bundle)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, check: calls.append(argv))
    assert cli.main(["app"]) == 0
    assert calls == [["open", str(bundle)]]


def test_app_locations_include_the_repo_build() -> None:
    assert any(p.name == "LocalCode.app" and "macos" in str(p) for p in cli.APP_LOCATIONS)


# --------------------------------------------------------------------------- #
# Headless plumbing
# --------------------------------------------------------------------------- #


def test_headless_without_a_prompt_is_a_usage_error(monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    assert cli.main(["-p"]) == 2
    assert "Podaj treść polecenia" in capsys.readouterr().err


def test_missing_directory_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["-C", "/nie/ma/takiego/katalogu", "-p", "x"]) == 1
    assert "nie istnieje" in capsys.readouterr().err


def test_effort_accepts_every_level_the_rest_of_the_app_accepts() -> None:
    """Regression: `--effort on` was rejected by the terminal only.

    ``normalize_effort`` (templates), ``PATCH /api/conversations`` and the Mac
    app all accept ``"on"`` — a Qwen3-style thinking switch is shown as
    "on"/"off" — so the terminal must not be the one place that refuses it.
    """
    from localcode.runtime.templates import EFFORT_CHOICES

    for level in (*EFFORT_CHOICES, "on"):
        assert parse("--effort", level).effort == level


def test_bare_images_command_can_list() -> None:
    """Regression: `localcode images` (documented in docs/IMAGES.md) crashed.

    Without the `list` subcommand argparse produced a namespace without
    `limit`/`favorite`, and the dispatcher — which defaults `action` to
    "list" — died with `AttributeError: 'Namespace' object has no attribute
    'limit'`.
    """
    options = cli.build_images_parser().parse_args([])
    assert options.action is None
    assert options.limit == 20
    assert options.favorite is False
