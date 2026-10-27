"""Tests for localcode.agent.session (JSONL transcripts, listing, resume)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from localcode.agent import session as sessions
from localcode.agent.session import AgentSession, SessionError, SessionStore
from localcode.config import sessions_dir


@pytest.fixture
def project(tmp_path: Path) -> Path:
    path = tmp_path / "projekt"
    path.mkdir()
    return path


def records(store: SessionStore) -> list[dict]:
    return [json.loads(line) for line in store.path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Layout and header
# --------------------------------------------------------------------------- #


def test_create_writes_the_header_in_the_project_directory(isolated_home, project: Path) -> None:
    store = SessionStore.create(project, model_key="org/model", mode="plan")

    assert store.path.parent == sessions_dir() / sessions.slug_for(project)
    assert store.path.name == f"{store.session.id}.jsonl"
    header = records(store)[0]
    assert header == {
        "kind": "session",
        "id": store.session.id,
        "cwd": str(project.resolve()),
        "modelKey": "org/model",
        "mode": "plan",
        "title": "",
        "createdAt": store.session.created_at,
    }


def test_slug_is_stable_and_readable(tmp_path: Path) -> None:
    a = sessions.slug_for(tmp_path / "moj-projekt")
    b = sessions.slug_for(str(tmp_path / "moj-projekt"))
    other = sessions.slug_for(tmp_path / "inny-projekt")
    assert a == b and a.startswith("moj-projekt-") and a != other


def test_session_dto_is_camel_case() -> None:
    dto = AgentSession(id="x", cwd="/tmp", model_key="k", mode="yolo", title="t").to_dict()
    assert set(dto) == {"id", "cwd", "modelKey", "mode", "title", "createdAt", "updatedAt",
                        "contextUsed", "contextLimit", "busy"}


def test_duplicate_id_is_refused(isolated_home, project: Path) -> None:
    store = SessionStore.create(project, session_id="abc123")
    assert store.session.id == "abc123"
    with pytest.raises(SessionError):
        SessionStore.create(project, session_id="abc123")


def test_invalid_id_is_refused(isolated_home, project: Path) -> None:
    with pytest.raises(SessionError):
        SessionStore.create(project, session_id="../../etc/passwd")


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_round_trip_of_messages_and_items(isolated_home, project: Path) -> None:
    store = SessionStore.create(project, model_key="org/model")
    store.title_from("Napraw failujący test\n(drugi wiersz)")
    store.append_message({"role": "user", "content": "Napraw failujący test"})
    store.append_item({"kind": "user", "id": "i1", "text": "Napraw failujący test", "createdAt": 1})
    store.append_many(
        messages=[{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "pytest"}}}]}],
        items=[{"kind": "tool", "id": "c1", "toolName": "bash", "ok": False, "summary": "exit 1",
                "output": "1 failed", "createdAt": 2}],
    )
    store.append_message({"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "1 failed"})

    session, messages, items = sessions.load(store.session.id)
    assert session.title == "Napraw failujący test"
    assert session.model_key == "org/model"
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages == store.messages
    assert [i["kind"] for i in items] == ["user", "tool"]
    assert items == store.items


def test_replace_history_is_what_resume_restores(isolated_home, project: Path) -> None:
    store = SessionStore.create(project)
    for i in range(4):
        store.append_message({"role": "user", "content": f"wiadomość {i}"})
    compacted = [{"role": "user", "content": "[Summary of earlier work]\nrobiliśmy X\n[End of summary]"}]
    store.replace_history(compacted)
    store.append_message({"role": "assistant", "content": "dalej"})

    _session, messages, _items = sessions.load(store.session.id)
    assert messages == compacted + [{"role": "assistant", "content": "dalej"}]

    resumed = SessionStore.resume(store.session.id)
    assert resumed.messages == messages
    kinds = [r["kind"] for r in records(store)]
    assert kinds.count("replace_history") == 1


def test_meta_records_update_the_header(isolated_home, project: Path) -> None:
    store = SessionStore.create(project, model_key="a/b", mode="default")
    store.set_mode("yolo")
    store.set_model("c/d")
    store.set_title("Nowy tytuł")

    session, _messages, _items = sessions.load(store.session.id)
    assert (session.mode, session.model_key, session.title) == ("yolo", "c/d", "Nowy tytuł")
    listed = sessions.list_sessions(project)[0]
    assert (listed.mode, listed.model_key, listed.title) == ("yolo", "c/d", "Nowy tytuł")


def test_title_is_only_derived_once(isolated_home, project: Path) -> None:
    store = SessionStore.create(project)
    store.title_from("  pierwsze pytanie  ")
    store.title_from("drugie pytanie")
    assert store.session.title == "pierwsze pytanie"


def test_long_titles_are_shortened() -> None:
    assert sessions.derive_title("x" * 200).endswith("…")
    assert len(sessions.derive_title("x" * 200)) == 60
    assert sessions.derive_title("\n\n  hej  \nkolejna") == "hej"
    assert sessions.derive_title("") == ""


def test_unicode_survives_the_round_trip(isolated_home, project: Path) -> None:
    store = SessionStore.create(project)
    text = "Zażółć gęślą jaźń — ✅ 日本語"
    store.append_message({"role": "user", "content": text})
    assert text in store.path.read_text(encoding="utf-8")  # not \u-escaped
    _s, messages, _i = sessions.load(store.session.id)
    assert messages[0]["content"] == text


# --------------------------------------------------------------------------- #
# Listing, deleting, robustness
# --------------------------------------------------------------------------- #


def test_list_sessions_newest_first_and_filtered_by_cwd(isolated_home, tmp_path: Path) -> None:
    first = tmp_path / "p1"
    second = tmp_path / "p2"
    first.mkdir()
    second.mkdir()
    a = SessionStore.create(first, title="A")
    time.sleep(0.01)
    b = SessionStore.create(first, title="B")
    time.sleep(0.01)
    c = SessionStore.create(second, title="C")

    all_ids = [s.id for s in sessions.list_sessions()]
    assert all_ids[0] == c.session.id
    assert set(all_ids) == {a.session.id, b.session.id, c.session.id}

    only_first = sessions.list_sessions(first)
    assert [s.id for s in only_first] == [b.session.id, a.session.id]
    assert all(s.cwd == str(first.resolve()) for s in only_first)
    assert sessions.list_sessions(tmp_path / "nieistniejacy") == []
    assert len(sessions.list_sessions(limit=1)) == 1


def test_list_sessions_ignores_broken_files(isolated_home, project: Path) -> None:
    good = SessionStore.create(project)
    directory = good.path.parent
    (directory / "smieci.jsonl").write_text("nie json\n", encoding="utf-8")
    (directory / "pusty.jsonl").write_text("", encoding="utf-8")

    listed = sessions.list_sessions()
    assert [s.id for s in listed] == [good.session.id]


def test_truncated_last_line_is_skipped(isolated_home, project: Path) -> None:
    store = SessionStore.create(project)
    store.append_message({"role": "user", "content": "całe"})
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "message", "at": 1, "mess')  # crash mid-write

    _session, messages, _items = sessions.load(store.session.id)
    assert [m["content"] for m in messages] == ["całe"]


def test_delete_removes_the_file_and_the_empty_directory(isolated_home, project: Path) -> None:
    store = SessionStore.create(project)
    directory = store.path.parent
    assert sessions.delete(store.session.id) is True
    assert not store.path.exists() and not directory.exists()
    assert sessions.delete(store.session.id) is False
    assert sessions.list_sessions() == []


def test_load_and_resume_of_a_missing_session_raise(isolated_home) -> None:
    with pytest.raises(SessionError):
        sessions.load("20200101-000000-aaaaaa")
    with pytest.raises(SessionError):
        SessionStore.resume("20200101-000000-aaaaaa")
    assert sessions.find_path("nope") is None


def test_write_failure_does_not_raise(isolated_home, project: Path, monkeypatch) -> None:
    store = SessionStore.create(project)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "open", boom)
    store.append_message({"role": "user", "content": "nie zapisze się"})
    assert store.writable is False
    assert store.messages[-1]["content"] == "nie zapisze się"  # the turn keeps working


def test_new_session_ids_are_unique_and_sortable() -> None:
    ids = [sessions.new_session_id() for _ in range(5)]
    assert len(set(ids)) == 5
    stamps = [i.rsplit("-", 1)[0] for i in ids]
    assert stamps == sorted(stamps), "the timestamp prefix must order sessions by age"
