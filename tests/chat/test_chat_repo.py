"""Database schema and repository: shapes, FTS, cascades, settings."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from localcode.chat.db import (
    FOLD_MAP,
    SCHEMA_VERSION,
    ChatDB,
    ChatDBError,
    fold_text,
    restore_snippet,
)
from localcode.chat.repo import (
    DEFAULT_APP_SETTINGS,
    UNSET,
    ChatRepo,
    build_fts_query,
    coerce_settings,
    derive_title,
    parse_citations,
    to_preview,
)
from swift_shapes import FALLBACK, assert_shape, load_spec, parse_models_swift, MODELS_SWIFT


# --------------------------------------------------------------------------- #
# Schema / connection
# --------------------------------------------------------------------------- #


def test_schema_created_with_wal_and_foreign_keys(tmp_path: Path) -> None:
    db = ChatDB(tmp_path / "chat.db").open()
    try:
        assert db.schema_version == SCHEMA_VERSION
        with db.read() as c:
            assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master")}
        for table in ("folders", "conversations", "messages", "attachments", "memories", "kb_documents",
                      "kb_chunks", "settings", "messages_fts", "kb_chunks_fts"):
            assert table in names
        with db.read() as c:
            sql = c.execute("SELECT sql FROM sqlite_master WHERE name = 'kb_chunks_fts'").fetchone()[0]
        assert "remove_diacritics 2" in sql
    finally:
        db.close()


def test_reopen_is_idempotent_and_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "chat.db"
    ChatDB(path).open().close()
    ChatDB(path).open().close()          # migrations do not run twice
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.commit()
    conn.close()
    with pytest.raises(ChatDBError, match="nowszej wersji"):
        ChatDB(path).open()


def test_transaction_rolls_back_and_nests(tmp_path: Path) -> None:
    db = ChatDB(tmp_path / "chat.db").open()
    with pytest.raises(RuntimeError):
        with db.transaction() as c:
            c.execute("INSERT INTO settings (id, value, updated_at) VALUES (1, '{}', 0)")
            raise RuntimeError("boom")
    with db.read() as c:
        assert c.execute("SELECT COUNT(*) FROM settings").fetchone()[0] == 0
    with db.transaction() as c:
        c.execute("INSERT INTO settings (id, value, updated_at) VALUES (1, '{}', 0)")
        with pytest.raises(ValueError):
            with db.transaction() as inner:
                inner.execute("UPDATE settings SET value = 'x'")
                raise ValueError
    with db.read() as c:
        assert c.execute("SELECT value FROM settings").fetchone()[0] == "{}"
    db.close()


async def test_db_run_uses_worker_thread(tmp_path: Path) -> None:
    db = ChatDB(tmp_path / "chat.db").open()
    main = threading.get_ident()
    ident = await db.run(lambda conn: threading.get_ident())
    assert ident != main
    db.close()


def test_repo_is_thread_safe(repo: ChatRepo) -> None:
    conv = repo.create_conversation({"title": "Wątki"})
    errors: list[BaseException] = []

    def worker(n: int) -> None:
        try:
            for i in range(20):
                repo.create_message({"conversationId": conv["id"], "role": "user", "content": f"w{n} {i}"})
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    seqs = [m["seq"] for m in repo.list_messages(conv["id"])]
    assert seqs == list(range(1, 101))


# --------------------------------------------------------------------------- #
# Folding / FTS helpers
# --------------------------------------------------------------------------- #


def test_fold_map_is_length_preserving() -> None:
    assert all(len(k) == 1 and len(v) == 1 for k, v in FOLD_MAP.items())
    text = "Łódź, żółw i ŁOŚ"
    assert len(fold_text(text)) == len(text)
    assert fold_text(text) == "Lódź, żólw i LOŚ"


def test_build_fts_query_is_safe() -> None:
    assert build_fts_query("") is None
    assert build_fts_query("a ! ?") is None
    q = build_fts_query('Zażółć "gęślą" OR NEAR(jaźń) col:x*')
    assert q is not None
    # only quoted word tokens, OR-ed prefix queries; no raw operators survive
    for part in q.split(" OR "):
        assert part.startswith('"') and part.endswith('"*')
    assert '"zażólć"*' in q            # only ł is folded here; the tokenizer folds the rest
    assert '"or"*' in q                 # operators become harmless quoted words
    assert build_fts_query("kot kot pies") == '"kot"* OR "pies"*'


def test_restore_snippet_maps_back_to_original() -> None:
    from localcode.chat.db import SNIPPET_CLOSE, SNIPPET_ELLIPSIS, SNIPPET_OPEN

    original = "Stara łódź stoi na brzegu jeziora"
    folded = fold_text(original)
    raw = f"{SNIPPET_ELLIPSIS}Stara {SNIPPET_OPEN}lódź{SNIPPET_CLOSE} stoi{SNIPPET_ELLIPSIS}"
    assert restore_snippet(raw, folded, original) == "…Stara <mark>łódź</mark> stoi…"
    assert restore_snippet(None, folded, original) == ""


# --------------------------------------------------------------------------- #
# Shapes vs Models.swift
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not MODELS_SWIFT.is_file(), reason="ai_ux Swift sources not on this machine")
def test_fallback_spec_matches_models_swift() -> None:
    spec, _ = parse_models_swift()
    for name, fields in FALLBACK.items():
        assert spec[name] == fields, name


def test_shapes_match_swift_models(repo: ChatRepo, tmp_path: Path) -> None:
    folder = repo.create_folder({"name": "Projekt", "memoryMode": "isolated", "color": "#fff"})
    assert_shape(folder, "Folder")
    folders = repo.list_folders_with_stats()
    assert_shape(folders, "[Folder]", list_item=True)

    conv = repo.create_conversation({"folderId": folder["id"], "title": "  Tytuł ", "temperature": 0.5,
                                     "maxTokens": 512, "provider": "local", "model": "m"})
    assert conv["title"] == "Tytuł"
    assert_shape(conv, "Conversation")
    listed = repo.list_conversations()
    assert_shape(listed, "[Conversation]", list_item=True)

    rel = repo.store_upload("att", "a1", ".txt", b"hello")
    att = repo.create_attachment({"id": "a1", "name": "notatka.txt", "mime": "text/plain", "size": 5,
                                  "kind": "text", "path": rel, "textContent": "hello"})
    assert "path" not in att
    assert_shape(att, "Attachment")
    msg = repo.create_message({"conversationId": conv["id"], "role": "user", "content": "Cześć",
                               "attachmentIds": ["a1"]})
    assert msg["attachments"][0]["messageId"] == msg["id"]
    reply = repo.create_message({"conversationId": conv["id"], "role": "assistant", "content": "Hej",
                                 "status": "complete", "tokensIn": 10, "tokensOut": 2, "costUsd": 0,
                                 "latencyMs": 12, "citations": [{"documentId": "d", "documentName": "n",
                                                                  "chunkIndex": 1, "snippet": "s", "score": 0.5}]})
    assert_shape(msg, "Message")
    assert_shape(reply, "Message")
    assert reply["citations"][0]["chunkIndex"] == 1
    assert_shape({"conversation": conv, "messages": repo.list_messages(conv["id"])}, "ConversationDetail")

    mem = repo.create_memory({"scope": "folder", "folderId": folder["id"], "content": "Lubi herbatę",
                              "kind": "preference"})
    assert_shape(mem, "MemoryEntry")
    doc = repo.create_document({"scope": "global", "name": "a.md", "mime": "text/markdown", "size": 3})
    assert "path" not in doc
    assert_shape(doc, "KbDocument")
    assert_shape(repo.get_settings(), "AppSettings")
    hits = repo.search_messages("cześć")
    assert hits
    assert_shape(hits, "[SearchHit]")


def test_list_conversation_fields_and_ordering(repo: ChatRepo) -> None:
    f = repo.create_folder({"name": "F"})
    a = repo.create_conversation({"title": "A", "folderId": f["id"]})
    b = repo.create_conversation({"title": "B"})
    c = repo.create_conversation({"title": "C", "archived": True})
    repo.create_message({"conversationId": a["id"], "role": "user", "content": "  pierwsza\n\n wiadomość  " * 20})
    repo.touch_conversation(a["id"], at=a["createdAt"] + 10_000)
    repo.update_conversation(b["id"], {"pinned": True})

    items = repo.list_conversations()
    assert [i["title"] for i in items] == ["B", "A"]            # pinned first, archived hidden
    a_item = items[1]
    assert a_item["messageCount"] == 1 and a_item["folderName"] == "F"
    assert a_item["preview"].startswith("pierwsza wiadomość") and len(a_item["preview"]) <= 120
    assert [i["title"] for i in repo.list_conversations(folder_id=None)] == ["B"]
    assert [i["title"] for i in repo.list_conversations(folder_id=f["id"])] == ["A"]
    assert {i["title"] for i in repo.list_conversations(archived=None)} == {"A", "B", "C"}
    assert [i["title"] for i in repo.list_conversations(archived=True)] == ["C"]
    assert len(repo.list_conversations(limit=1)) == 1
    assert repo.list_conversations(folder_id=UNSET) == repo.list_conversations()


def test_update_semantics(repo: ChatRepo) -> None:
    conv = repo.create_conversation({"title": "Stary", "systemPrompt": "x"})
    same = repo.update_conversation(conv["id"], {"title": "   "})
    assert same["title"] == "Stary"                    # blank title keeps the old one
    upd = repo.update_conversation(conv["id"], {"systemPrompt": None, "titleLocked": True, "temperature": 1.2})
    assert upd["systemPrompt"] is None and upd["titleLocked"] is True and upd["temperature"] == 1.2
    assert repo.update_conversation("missing", {"title": "x"}) is None
    folder = repo.create_folder({})
    assert folder["name"] == "Nowy projekt" and folder["memoryMode"] == "shared"
    second = repo.create_folder({"name": "Drugi"})
    assert second["sortOrder"] == folder["sortOrder"] + 1
    renamed = repo.update_folder(folder["id"], {"name": "", "memoryMode": "isolated", "description": "opis"})
    assert renamed["name"] == "Nowy projekt" and renamed["memoryMode"] == "isolated"
    assert repo.update_folder("nope", {"name": "x"}) is None


def test_message_seq_and_remove_after(repo: ChatRepo) -> None:
    conv = repo.create_conversation({})
    ids = [repo.create_message({"conversationId": conv["id"], "role": r, "content": str(i)})["id"]
           for i, r in enumerate(["user", "assistant", "user", "assistant"])]
    assert repo.next_seq(conv["id"]) == 5
    assert repo.last_user_message(conv["id"])["id"] == ids[2]
    assert repo.remove_messages_after(conv["id"], 2) == 2
    assert [m["id"] for m in repo.list_messages(conv["id"])] == ids[:2]
    assert repo.count_messages(conv["id"]) == 2
    patched = repo.update_message(ids[1], {"status": "stopped", "reasoning": "r", "citations": []})
    assert patched["status"] == "stopped" and patched["reasoning"] == "r" and patched["citations"] == []


def test_parse_citations_is_defensive() -> None:
    assert parse_citations(None) == []
    assert parse_citations("not json") == []
    assert parse_citations('{"a": 1}') == []
    out = parse_citations(json.dumps([{"documentId": 5, "chunkIndex": "x", "score": True}, 7]))
    assert out == [{"documentId": "", "documentName": "", "chunkIndex": 0, "snippet": "", "score": 0.0}]


def test_helpers() -> None:
    assert derive_title("   ") == "Nowa rozmowa"
    assert derive_title("a  b\n c") == "a b c"
    assert len(derive_title("x" * 100)) == 48 and derive_title("x" * 100).endswith("…")
    assert to_preview(None) is None and to_preview("  \n ") is None


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_folds_polish_diacritics(repo: ChatRepo) -> None:
    folder = repo.create_folder({"name": "Językowy"})
    conv = repo.create_conversation({"title": "Pangram", "folderId": folder["id"]})
    m = repo.create_message({"conversationId": conv["id"], "role": "user", "content": "Zażółć gęślą jaźń"})
    repo.create_message({"conversationId": conv["id"], "role": "assistant", "content": "Łódź i Kraków"})

    for query in ("zazolc", "Zażółć", "ZAZOLC", "gesla", "jazn", "zaz"):
        hits = repo.search_messages(query)
        assert [h["messageId"] for h in hits] == [m["id"]], query
    hit = repo.search_messages("zazolc")[0]
    assert hit["snippet"] == "<mark>Zażółć</mark> gęślą jaźń"
    assert hit["folderName"] == "Językowy" and hit["conversationTitle"] == "Pangram"
    lodz = repo.search_messages("lodz")
    assert lodz and lodz[0]["snippet"].startswith("<mark>Łódź</mark>")
    assert repo.search_messages("krakow")[0]["role"] == "assistant"
    assert repo.search_messages("") == []
    assert repo.search_messages('"; DROP TABLE messages; --') == []


def test_search_tracks_updates_and_deletes(repo: ChatRepo) -> None:
    conv = repo.create_conversation({})
    m = repo.create_message({"conversationId": conv["id"], "role": "user", "content": "pierwotna treść"})
    repo.update_message(m["id"], {"content": "zmieniona wersja"})
    assert repo.search_messages("pierwotna") == []
    assert repo.search_messages("zmieniona")
    repo.delete_conversation(conv["id"])
    assert repo.search_messages("zmieniona") == []
    with repo.db.read() as c:
        assert c.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 0


def test_long_snippet_has_ellipsis(repo: ChatRepo) -> None:
    conv = repo.create_conversation({})
    text = " ".join(["słowo"] * 60 + ["żaglówka"] + ["koniec"] * 60)
    repo.create_message({"conversationId": conv["id"], "role": "user", "content": text})
    snippet = repo.search_messages("zaglowka")[0]["snippet"]
    assert "<mark>żaglówka</mark>" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")


# --------------------------------------------------------------------------- #
# Deletion cascades (rows and files)
# --------------------------------------------------------------------------- #


def _upload(repo: ChatRepo, conv_id: str | None, name: str) -> dict:
    rel = repo.store_upload("att", name, ".txt", b"data")
    return repo.create_attachment({"id": name, "name": f"{name}.txt", "mime": "text/plain", "size": 4,
                                   "kind": "text", "path": rel, "conversationId": conv_id})


def test_delete_conversation_removes_messages_attachments_and_files(repo: ChatRepo) -> None:
    conv = repo.create_conversation({})
    att = _upload(repo, conv["id"], "f1")
    repo.create_message({"conversationId": conv["id"], "role": "user", "content": "x", "attachmentIds": [att["id"]]})
    path = repo.uploads / "att" / "f1.txt"
    assert path.exists()
    assert repo.delete_conversation(conv["id"]) is True
    assert not path.exists()
    assert repo.get_attachment(att["id"]) is None
    assert repo.delete_conversation(conv["id"]) is False


def test_delete_message_and_remove_after_delete_files(repo: ChatRepo) -> None:
    conv = repo.create_conversation({})
    u1 = repo.create_message({"conversationId": conv["id"], "role": "user", "content": "a"})
    att = _upload(repo, conv["id"], "f2")
    repo.create_message({"conversationId": conv["id"], "role": "user", "content": "b", "attachmentIds": ["f2"]})
    assert (repo.uploads / "att/f2.txt").exists()
    repo.remove_messages_after(conv["id"], u1["seq"])
    assert not (repo.uploads / "att/f2.txt").exists()
    assert repo.get_attachment(att["id"]) is None
    assert repo.delete_message(u1["id"]) is True and repo.delete_message(u1["id"]) is False


@pytest.mark.parametrize("delete_chats", [False, True])
def test_delete_folder(repo: ChatRepo, delete_chats: bool) -> None:
    folder = repo.create_folder({"name": "Do usunięcia"})
    other = repo.create_folder({"name": "Inny"})
    conv = repo.create_conversation({"folderId": folder["id"]})
    keep = repo.create_conversation({"folderId": other["id"]})
    att = _upload(repo, conv["id"], "f3")
    repo.create_message({"conversationId": conv["id"], "role": "user", "content": "x", "attachmentIds": [att["id"]]})
    repo.create_memory({"scope": "folder", "folderId": folder["id"], "content": "m"})
    repo.create_memory({"scope": "global", "content": "g"})
    rel = repo.store_upload("kb", "d1", ".md", b"# doc")
    doc = repo.create_document({"id": "d1", "scope": "folder", "folderId": folder["id"], "name": "d.md",
                                "path": rel, "status": "ready"})
    repo.insert_chunks(doc["id"], ["treść dokumentu numer jeden, wystarczająco długa"])

    assert repo.delete_folder(folder["id"], delete_chats) is True
    assert repo.get_folder(folder["id"]) is None
    assert repo.get_document(doc["id"]) is None
    assert not (repo.uploads / "kb/d1.md").exists()
    assert [m["content"] for m in repo.list_memories()] == ["g"]
    with repo.db.read() as c:
        assert c.execute("SELECT COUNT(*) FROM kb_chunks_fts").fetchone()[0] == 0
    remaining = repo.get_conversation(conv["id"])
    if delete_chats:
        assert remaining is None
        assert not (repo.uploads / "att/f3.txt").exists()
    else:
        assert remaining is not None and remaining["folderId"] is None
        assert (repo.uploads / "att/f3.txt").exists()
    assert repo.get_conversation(keep["id"])["folderId"] == other["id"]
    assert repo.delete_folder(folder["id"], delete_chats) is False


def test_upload_path_never_escapes(repo: ChatRepo) -> None:
    assert repo.upload_path("../../etc/passwd") is None
    assert repo.upload_path("/etc/passwd") is None
    assert repo.upload_path(None) is None
    assert repo.upload_path("att/x.txt") == (repo.uploads / "att/x.txt").resolve()
    with pytest.raises(ValueError):
        repo.store_upload("../x", "id", "", b"")


def test_purge_stale_uploads(repo: ChatRepo) -> None:
    att = _upload(repo, None, "old")
    with repo.db.transaction() as c:
        c.execute("UPDATE attachments SET created_at = 0 WHERE id = ?", (att["id"],))
    fresh = _upload(repo, None, "fresh")
    assert repo.purge_stale_uploads() == 1
    assert repo.get_attachment(att["id"]) is None and repo.get_attachment(fresh["id"]) is not None
    assert not (repo.uploads / "att/old.txt").exists()


# --------------------------------------------------------------------------- #
# Memories / knowledge rows
# --------------------------------------------------------------------------- #


def test_memories_filters_and_scope_update(repo: ChatRepo) -> None:
    f = repo.create_folder({})
    g = repo.create_memory({"content": "globalny", "pinned": True})
    assert g["scope"] == "global" and g["folderId"] is None and g["source"] == "user"
    m = repo.create_memory({"folderId": f["id"], "content": "projektowy", "source": "auto"})
    assert m["scope"] == "folder" and m["source"] == "auto"
    assert [x["id"] for x in repo.list_memories(scope="global")] == [g["id"]]
    assert [x["id"] for x in repo.list_memories(scope="folder", folder_id=f["id"])] == [m["id"]]
    assert [x["id"] for x in repo.list_memories(folder_id=None)] == [g["id"]]
    moved = repo.update_memory(m["id"], {"scope": "global"})
    assert moved["scope"] == "global" and moved["folderId"] is None
    back = repo.update_memory(m["id"], {"scope": "folder", "folderId": f["id"], "kind": "note"})
    assert back["folderId"] == f["id"] and back["kind"] == "note"
    assert repo.update_memory("x", {}) is None
    assert repo.delete_memory(g["id"]) and not repo.delete_memory(g["id"])


def test_chunks_and_embeddings(repo: ChatRepo) -> None:
    doc = repo.create_document({"scope": "global", "name": "a.txt", "status": "ready"})
    ids = repo.insert_chunks(doc["id"], ["pierwszy fragment tekstu o kotach", "drugi fragment o psach"])
    assert repo.get_document(doc["id"])["chunkCount"] == 2
    assert [c["idx"] for c in repo.list_chunks(doc["id"])] == [0, 1]
    missing = repo.chunks_missing_embedding("emb")
    assert [cid for cid, _ in missing] == ids
    repo.set_chunk_embeddings("emb", [(ids[0], b"\x00" * 8, 2)])
    assert [cid for cid, _ in repo.chunks_missing_embedding("emb")] == [ids[1]]
    rows = repo.embedded_chunks("emb", folder_id=None, include_global=True)
    assert [r["id"] for r in rows] == [ids[0]] and rows[0]["embeddingDim"] == 2
    # replacing chunks resets the index
    repo.insert_chunks(doc["id"], ["zupełnie nowa treść dokumentu"])
    assert len(repo.list_chunks(doc["id"])) == 1
    assert repo.search_chunks(build_fts_query("kotach"), folder_id=None, include_global=True, limit=5) == []


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_settings_defaults_and_deep_merge(repo: ChatRepo) -> None:
    s = repo.get_settings()
    assert s == coerce_settings({})
    assert s["defaults"]["provider"] == "local" and s["defaults"]["model"] == ""
    assert s["memoryEnabled"] is True and s["theme"] == "system"
    out = repo.patch_settings({"profile": {"displayName": "Filip"}, "defaults": {"ragTopK": 3}})
    assert out["profile"]["displayName"] == "Filip" and out["profile"]["enabled"] is True
    assert out["defaults"]["ragTopK"] == 3 and out["defaults"]["maxTokens"] == 4096
    out = repo.patch_settings({"defaults": {"temperature": 1.1}, "theme": "dark"})
    assert out["defaults"]["ragTopK"] == 3 and out["defaults"]["temperature"] == 1.1 and out["theme"] == "dark"
    assert repo.get_settings() == out


def test_coerce_settings_rejects_garbage() -> None:
    out = coerce_settings({"theme": "neon", "memoryEnabled": "yes", "profile": [1],
                           "defaults": {"maxTokens": 1024.4, "ragTopK": True, "temperature": float("nan"),
                                        "provider": ""}})
    assert out["theme"] == "system" and out["memoryEnabled"] is True
    assert out["profile"] == DEFAULT_APP_SETTINGS["profile"]
    assert out["defaults"]["maxTokens"] == 1024 and isinstance(out["defaults"]["maxTokens"], int)
    assert out["defaults"]["ragTopK"] == 6 and out["defaults"]["temperature"] == 0.7
    assert out["defaults"]["provider"] == "local"


def test_corrupted_settings_row_falls_back(repo: ChatRepo) -> None:
    with repo.db.transaction() as c:
        c.execute("INSERT INTO settings (id, value, updated_at) VALUES (1, 'not json', 0)")
    assert repo.get_settings() == coerce_settings({})
    spec, enums = load_spec()
    assert "AppSettings" in spec and "MemoryMode" in enums


def test_a_new_conversation_keeps_its_reasoning_effort(repo: ChatRepo) -> None:
    conversation = repo.create_conversation({"title": "x", "reasoningEffort": "high"})
    assert conversation["reasoningEffort"] == "high"
    assert repo.create_conversation({"title": "y"})["reasoningEffort"] is None
