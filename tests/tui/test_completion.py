"""``@path`` expansion and the prompt completer."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from localcode.tui import completion


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".hidden").write_text("secret", encoding="utf-8")
    (tmp_path / "README.md").write_text("# projekt\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #


def test_expansion_inlines_the_file(project: Path) -> None:
    text, attachments, warnings = completion.expand_at_references("zobacz @src/app.py", project)
    assert warnings == []
    assert [a.path for a in attachments] == ["src/app.py"]
    assert text.startswith("zobacz @src/app.py")
    assert '<file path="src/app.py">' in text
    assert "print('hi')" in text


def test_expansion_keeps_the_original_text(project: Path) -> None:
    text, _, _ = completion.expand_at_references("@README.md a potem @src/util.py", project)
    assert text.splitlines()[0] == "@README.md a potem @src/util.py"


def test_expansion_reports_a_missing_file(project: Path) -> None:
    text, attachments, warnings = completion.expand_at_references("@nie/ma.py", project)
    assert attachments == []
    assert warnings and "nie/ma.py" in warnings[0]
    assert text == "@nie/ma.py"


def test_expansion_respects_the_budget(project: Path) -> None:
    big = project / "big.txt"
    big.write_text("x" * 5000, encoding="utf-8")
    text, attachments, _ = completion.expand_at_references("@big.txt", project, max_chars=500)
    assert attachments[0].truncated
    assert len(text) < 1200
    assert "plik skrócony do budżetu kontekstu" in text


def test_expansion_skips_binary(project: Path) -> None:
    (project / "blob.bin").write_bytes(b"\x00\x01\x02" * 100)
    _, attachments, warnings = completion.expand_at_references("@blob.bin", project)
    assert attachments == []
    assert "binarny" in warnings[0]


def test_expansion_lists_a_directory(project: Path) -> None:
    text, attachments, _ = completion.expand_at_references("@src", project)
    assert attachments and attachments[0].is_dir
    assert '<dir path="src">' in text
    assert "src/app.py" in text


def test_expansion_deduplicates(project: Path) -> None:
    _, attachments, _ = completion.expand_at_references("@src/app.py i znowu @src/app.py", project)
    assert len(attachments) == 1


def test_expansion_limits_the_number_of_files(project: Path) -> None:
    for i in range(10):
        (project / f"f{i}.txt").write_text(str(i), encoding="utf-8")
    line = " ".join(f"@f{i}.txt" for i in range(10))
    _, attachments, warnings = completion.expand_at_references(line, project, max_files=3)
    assert len(attachments) == 3
    assert any("maksymalnie 3" in w for w in warnings)


def test_email_like_text_is_not_a_reference(project: Path) -> None:
    text, attachments, warnings = completion.expand_at_references("pisz na jan@example.com", project)
    assert attachments == [] and warnings == [] and text == "pisz na jan@example.com"


def test_no_reference_leaves_the_text_alone(project: Path) -> None:
    text, attachments, warnings = completion.expand_at_references("zwykły tekst", project)
    assert (text, attachments, warnings) == ("zwykły tekst", [], [])


def test_summary_line(project: Path) -> None:
    _, attachments, _ = completion.expand_at_references("@src/app.py", project)
    assert completion.summarize_attachments(attachments).startswith("Dołączono: src/app.py — ")


def test_at_budget_scales_with_the_context() -> None:
    assert completion.at_budget_chars(None) == completion.DEFAULT_AT_CHARS
    assert completion.at_budget_chars(4096) < completion.at_budget_chars(32768)
    assert completion.at_budget_chars(1024) >= 1200


# --------------------------------------------------------------------------- #
# Completer
# --------------------------------------------------------------------------- #


def complete(completer: completion.LocalCodeCompleter, text: str) -> list[str]:
    return [c.text for c in completer.get_completions(Document(text, len(text)), None)]


def test_slash_completion(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project)
    assert "/model" in complete(completer, "/mod")
    assert "/help" in complete(completer, "/")


def test_slash_completion_hides_agent_commands_in_chat(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project, chat_mode=True)
    assert "/mode" not in complete(completer, "/m")
    assert "/resume" not in complete(completer, "/r")
    assert "/clear" in complete(completer, "/c")
    # compaction exists in both modes
    assert "/compact" in complete(completer, "/c")


def test_slash_value_completion(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project)
    assert "plan" in complete(completer, "/mode pl")


def test_path_completion(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project)
    assert "src/" in complete(completer, "zobacz @sr")
    assert "src/app.py" in complete(completer, "zobacz @src/a")


def test_path_completion_skips_noise(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project)
    offered = complete(completer, "@")
    assert "node_modules/" not in offered
    assert ".hidden" not in offered
    assert "README.md" in offered


def test_no_completion_in_plain_text(project: Path) -> None:
    completer = completion.LocalCodeCompleter(project)
    assert complete(completer, "zwykły tekst") == []


def test_list_paths_of_a_missing_directory(project: Path) -> None:
    assert completion.list_paths("nie-ma/", project) == []
