"""Per-language notes: detection, size selection, packaging, prompt wiring."""

from __future__ import annotations

import os

import pytest

from localcode.agent import langskills, prompts


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _write(root, rel, text="x"):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path) or root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_detects_language_from_manifest_and_extensions(tmp_path):
    root = str(tmp_path)
    _write(root, "go.mod", "module x")
    _write(root, "cmd/main.go", "package main")
    _write(root, "README.md", "# doc")
    assert langskills.detect_languages(root) == ["go"]


def test_a_manifest_outweighs_a_single_stray_file(tmp_path):
    root = str(tmp_path)
    _write(root, "Cargo.toml", "[package]")
    _write(root, "src/main.rs", "fn main() {}")
    _write(root, "helper.py", "print()")  # one incidental script
    langs = langskills.detect_languages(root)
    assert langs[0] == "rust"


def test_detects_languages_named_in_the_task_text(tmp_path):
    # empty folder, but the task mentions files
    langs = langskills.detect_languages(
        str(tmp_path), extra_text="please fix src/main.rs and update Cargo.toml"
    )
    assert langs == ["rust"]


def test_cap_limits_the_number_of_languages(tmp_path):
    root = str(tmp_path)
    _write(root, "a.py")
    _write(root, "b.py")
    _write(root, "c.js")
    _write(root, "d.go")
    _write(root, "e.rs")
    assert len(langskills.detect_languages(root, cap=2)) == 2
    assert langskills.detect_languages(root, cap=0) == []


def test_empty_project_detects_nothing(tmp_path):
    assert langskills.detect_languages(str(tmp_path)) == []


def test_noise_directories_are_not_scanned(tmp_path):
    root = str(tmp_path)
    _write(root, "main.go", "package main")
    _write(root, "node_modules/pkg/index.js", "x")  # must be ignored
    assert langskills.detect_languages(root) == ["go"]


# --------------------------------------------------------------------------- #
# Size selection (mini for local, max for remote)
# --------------------------------------------------------------------------- #


def test_note_files_are_packaged_as_resources():
    # importlib.resources sees exactly what ships in the wheel
    import importlib.resources as ir

    names = {p.name for p in ir.files("localcode.agent.langskills").iterdir()}
    for lang in langskills.LANGS:
        assert f"{lang}.mini.md" in names, lang
        assert f"{lang}.max.md" in names, lang


def test_every_language_ships_both_sizes():
    for lang in langskills.LANGS:
        mini = langskills.load_note(lang, strong=False)
        full = langskills.load_note(lang, strong=True)
        assert mini, f"{lang} has no mini note"
        assert full, f"{lang} has no max note"


def test_max_is_more_thorough_than_mini():
    for lang in langskills.LANGS:
        mini = langskills.load_note(lang, strong=False)
        full = langskills.load_note(lang, strong=True)
        assert len(full) > len(mini), lang


def test_mini_notes_stay_terse_for_small_windows():
    # a mini note is a few hundred tokens at most (~3.5 chars/token)
    for lang in langskills.LANGS:
        mini = langskills.load_note(lang, strong=False)
        assert len(mini) <= 1200, f"{lang} mini note is too long ({len(mini)} chars)"


def test_build_section_picks_size_by_model_kind(tmp_path):
    _write(str(tmp_path), "main.py", "print()")
    local = langskills.build_section(str(tmp_path), strong=False)
    remote = langskills.build_section(str(tmp_path), strong=True)
    assert local and remote
    assert "# Language notes" in local
    assert "## Python" in local
    assert len(remote) > len(local)  # max text is longer


def test_build_section_is_none_without_languages(tmp_path):
    assert langskills.build_section(str(tmp_path), strong=False) is None


def test_mini_section_budget_drops_extra_languages(tmp_path):
    root = str(tmp_path)
    for name in ("a.py", "b.go", "c.rs"):
        _write(root, name)
    # a tiny budget keeps only the first language's note
    section = langskills.build_section(root, strong=False, cap=3, char_budget=200)
    assert section.count("## ") == 1


# --------------------------------------------------------------------------- #
# System-prompt wiring
# --------------------------------------------------------------------------- #


def test_system_prompt_includes_language_notes_after_project_instructions():
    notes = langskills.build_section  # sanity: callable
    assert callable(notes)
    prompt = prompts.build_system_prompt(
        cwd="/tmp/x", n_ctx=32768, mode="default", native_tools=True,
        tool_names=["read_file", "write_file", "bash"],
        project_instructions="Use tabs here.",
        env={"os": "macOS", "date": "2026-01-01"},
        lang_notes="# Language notes\n## Python\n- use pytest",
    )
    assert "# Language notes" in prompt
    assert "## Python" in prompt
    # repo rules must still come before the generic language notes
    assert prompt.index("# Project instructions") < prompt.index("# Language notes")


def test_system_prompt_without_language_notes_is_unchanged():
    prompt = prompts.build_system_prompt(
        cwd="/tmp/x", n_ctx=32768, mode="default", native_tools=True,
        tool_names=["read_file"], project_instructions=None,
        env={"os": "macOS", "date": "2026-01-01"},
    )
    assert "# Language notes" not in prompt


# --------------------------------------------------------------------------- #
# Attached / @-mentioned files are explained (the "asks for files it has" bug)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_ctx", [4096, 16384, 131072])  # tiny / small / large tiers
def test_prompt_explains_inlined_file_blocks(n_ctx):
    prompt = prompts.build_system_prompt(
        cwd="/tmp/x", n_ctx=n_ctx, mode="default", native_tools=True,
        tool_names=["read_file", "write_file", "bash"],
        project_instructions=None, env={"os": "macOS", "date": "2026-01-01"},
    )
    assert "<file path" in prompt
    low = prompt.lower()
    # it must tell the model not to ask the user to resend an attached file
    assert "resend" in low or "paste" in low
