"""The ai_ux chat API over HTTP: JSON shapes, uploads, export and the SSE turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from aiohttp import FormData
from aiohttp.test_utils import TestClient

from server_fakes import CHAT_KEY, FakeEngine
from swift_shapes import assert_shape


async def read_frames(response: Any, limit: int | None = None) -> list[dict[str, Any]]:
    """Decode ``data: {json}`` records from an open SSE response."""
    frames: list[dict[str, Any]] = []
    async for raw in response.content:
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        assert line.startswith("data: "), line
        frames.append(json.loads(line[6:]))
        if limit is not None and len(frames) >= limit:
            break
    return frames


# --------------------------------------------------------------------------- #
# Bootstrap / settings / models
# --------------------------------------------------------------------------- #


async def test_bootstrap_matches_swift(client: TestClient) -> None:
    response = await client.get("/api/bootstrap")
    assert response.status == 200
    body = await response.json()
    assert body["user"] is None
    assert body["activeProvider"] == "local"
    assert_shape(body, "Bootstrap")


async def test_models_and_providers(client: TestClient) -> None:
    models = await (await client.get("/api/models")).json()
    assert [m["id"] for m in models] == [CHAT_KEY, "mlx-community/Qwen3-4B-Instruct-2507-4bit"]
    for model in models:
        assert_shape(model, "ModelInfo")
    providers = await (await client.get("/api/providers")).json()
    assert_shape(providers, "ProvidersResponse")


async def test_settings_roundtrip(client: TestClient) -> None:
    settings = await (await client.get("/api/settings")).json()
    assert_shape(settings, "AppSettings")
    response = await client.patch("/api/settings", json={"theme": "dark",
                                                         "defaults": {"temperature": 0.4}})
    assert response.status == 200
    patched = await response.json()
    assert patched["theme"] == "dark"
    assert patched["defaults"]["temperature"] == pytest.approx(0.4)


async def test_settings_validation_error(client: TestClient) -> None:
    response = await client.patch("/api/settings", json={"theme": "neon"})
    assert response.status == 400
    body = await response.json()
    assert body["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# Conversations, folders, memories
# --------------------------------------------------------------------------- #


async def test_conversation_crud(client: TestClient) -> None:
    created = await client.post("/api/conversations", json={"title": "Test"})
    assert created.status == 201
    conversation = await created.json()
    assert_shape(conversation, "Conversation")
    cid = conversation["id"]

    listed = await (await client.get("/api/conversations")).json()
    assert [c["id"] for c in listed] == [cid]

    detail = await (await client.get(f"/api/conversations/{cid}")).json()
    assert_shape(detail, "ConversationDetail")

    patched = await (await client.patch(f"/api/conversations/{cid}", json={"title": "Inny"})).json()
    assert patched["title"] == "Inny" and patched["titleLocked"] is True

    assert (await client.delete(f"/api/conversations/{cid}")).status == 200
    assert (await client.get(f"/api/conversations/{cid}")).status == 404


async def test_folder_and_memory_crud(client: TestClient) -> None:
    folder = await (await client.post("/api/folders", json={"name": "Projekt"})).json()
    assert_shape(folder, "Folder")
    fid = folder["id"]

    memory = await (await client.post("/api/memories",
                                      json={"scope": "folder", "folderId": fid,
                                            "content": "Lubię zwięzłe odpowiedzi."})).json()
    assert_shape(memory, "MemoryEntry")
    assert memory["folderId"] == fid

    listed = await (await client.get(f"/api/memories?scope=folder&folderId={fid}")).json()
    assert [m["id"] for m in listed] == [memory["id"]]

    assert (await client.patch(f"/api/memories/{memory['id']}", json={"pinned": True})).status == 200
    assert (await client.delete(f"/api/memories/{memory['id']}")).status == 200
    assert (await client.delete(f"/api/folders/{fid}?deleteChats=1")).status == 200


async def test_memory_scope_error(client: TestClient) -> None:
    response = await client.post("/api/memories", json={"scope": "folder", "content": "x"})
    assert response.status == 400
    assert (await response.json())["code"] == "folder_required"


# --------------------------------------------------------------------------- #
# Uploads and knowledge
# --------------------------------------------------------------------------- #


def _form(name: str, content: bytes, mime: str = "text/plain", **fields: str) -> FormData:
    form = FormData()
    for key, value in fields.items():
        form.add_field(key, value)
    form.add_field("files", content, filename=name, content_type=mime)
    return form


async def test_upload_and_fetch_attachment(client: TestClient) -> None:
    response = await client.post("/api/upload", data=_form("notatka.txt", b"Ala ma kota"))
    assert response.status == 201
    attachments = await response.json()
    assert len(attachments) == 1
    assert_shape(attachments[0], "Attachment")
    assert attachments[0]["name"] == "notatka.txt"

    fetched = await client.get(f"/api/attachments/{attachments[0]['id']}")
    assert fetched.status == 200
    assert await fetched.read() == b"Ala ma kota"
    assert "notatka.txt" in fetched.headers["Content-Disposition"]


async def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = await client.post("/api/upload", data=_form("pusty.txt", b""))
    assert response.status == 400
    assert (await response.json())["code"] == "empty_file"


async def test_upload_requires_multipart(client: TestClient) -> None:
    response = await client.post("/api/upload", json={"files": []})
    assert response.status == 415
    assert (await response.json())["code"] == "unsupported_media_type"


async def test_knowledge_upload_and_delete(client: TestClient) -> None:
    response = await client.post("/api/knowledge",
                                 data=_form("wiedza.md", b"# Tytul\n\nTresc dokumentu.", "text/markdown",
                                            scope="global"))
    assert response.status == 201
    documents = await response.json()
    assert_shape(documents[0], "KbDocument")
    assert documents[0]["chunkCount"] >= 1

    listed = await (await client.get("/api/knowledge?scope=global")).json()
    assert [d["id"] for d in listed] == [documents[0]["id"]]
    assert (await client.delete(f"/api/knowledge/{documents[0]['id']}")).status == 200


async def test_knowledge_unknown_folder(client: TestClient) -> None:
    response = await client.post("/api/knowledge",
                                 data=_form("x.txt", b"tresc", "text/plain",
                                            scope="folder", folderId="brak"))
    assert response.status == 404
    assert (await response.json())["code"] == "folder_not_found"


# --------------------------------------------------------------------------- #
# Chat streaming
# --------------------------------------------------------------------------- #


async def test_chat_stream_full_turn(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["Cześć", ", ", "jak mogę pomóc?"]
    response = await client.post("/api/chat", json={"content": "Hej", "model": CHAT_KEY})
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    frames = await read_frames(response)
    kinds = [f["type"] for f in frames]
    assert kinds[0] == "start"
    assert "text" in kinds and "usage" in kinds and "done" in kinds
    start = frames[0]
    assert_shape(start["userMessage"], "Message")
    done = next(f for f in frames if f["type"] == "done")
    assert_shape(done["message"], "Message")
    assert "".join(f["delta"] for f in frames if f["type"] == "text") == "Cześć, jak mogę pomóc?"


async def test_chat_validation_error_is_json(client: TestClient) -> None:
    response = await client.post("/api/chat", json={"content": ""})
    assert response.status == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert (await response.json())["code"] == "empty_message"


async def test_chat_unknown_conversation(client: TestClient) -> None:
    response = await client.post("/api/chat", json={"content": "x", "conversationId": "brak"})
    assert response.status == 404
    assert (await response.json())["code"] == "conversation_not_found"


async def test_chat_disconnect_marks_message_stopped(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["kawałek "] * 60
    engine.delay = 0.05
    response = await client.post("/api/chat", json={"content": "Opowiedz", "model": CHAT_KEY})
    frames = await read_frames(response, limit=2)
    conversation_id = frames[0]["conversationId"]
    response.close()                       # the client hangs up — this is the "stop" button

    for _ in range(60):
        await asyncio.sleep(0.1)
        detail = await (await client.get(f"/api/conversations/{conversation_id}")).json()
        status = detail["messages"][-1]["status"]
        if status != "streaming":
            break
    assert status == "stopped"
    assert engine.closed >= 1


async def test_export_markdown_and_json(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["Odpowiedź"]
    response = await client.post("/api/chat", json={"content": "Pytanie", "model": CHAT_KEY})
    frames = await read_frames(response)
    cid = frames[0]["conversationId"]

    md = await client.get(f"/api/conversations/{cid}/export?format=md")
    assert md.status == 200
    assert md.headers["Content-Type"].startswith("text/markdown")
    text = await md.text()
    assert "## Ty" in text and "Pytanie" in text

    as_json = await client.get(f"/api/conversations/{cid}/export?format=json")
    payload = json.loads(await as_json.text())
    assert payload["conversation"]["id"] == cid
    assert len(payload["messages"]) == 2

    bad = await client.get(f"/api/conversations/{cid}/export?format=pdf")
    assert bad.status == 400 and (await bad.json())["code"] == "bad_format"


async def test_search(client: TestClient, engine: FakeEngine) -> None:
    engine.script = ["Odpowiedź o kotach"]
    response = await client.post("/api/chat", json={"content": "Opowiedz o kotach", "model": CHAT_KEY})
    await read_frames(response)
    hits = await (await client.get("/api/search?q=kotach")).json()
    assert hits and all(h["conversationId"] for h in hits)
    for hit in hits:
        assert_shape(hit, "SearchHit")


async def test_chat_validation_error_is_a_plain_json_body(client: TestClient) -> None:
    """A request that fails validation answers as an ordinary JSON error.

    ``routes_chat._sse`` pulls the first frame before committing the response so
    that this stays a JSON body instead of an ``{"type":"error"}`` frame, and
    the Mac app reads that body (``APIClient.makeSSEStream``) — the status code
    alone never says which field was wrong. An unusable *model* is different: it
    is reported inside the stream, after the user message has been stored.
    """
    response = await client.post("/api/chat", json={"content": "Cześć", "temperature": 99})
    assert response.status == 400
    assert response.content_type == "application/json"
    body = await response.json()
    assert body["code"] == "invalid_request"
    assert body["error"]
    assert any(issue["path"] == "temperature" for issue in body["details"])
