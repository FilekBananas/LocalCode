"""The JSON this server returns must decode into the Mac app's Codable structs.

The Swift sources under ``macos/Sources`` are the contract: this module parses
their ``init(from decoder:)`` bodies (which say exactly which keys are required
and what type each has) and validates live server responses against them. It
also checks that every ``"api/…"`` path the Swift client builds is a route this
server registers.

Skipped (with a clear message) while ``macos/Sources`` does not exist yet; the
ai_ux-compatible part of the chat API is checked against
``the reference Models.swift`` in ``test_routes_chat.py``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient

from server_fakes import CHAT_KEY
from swift_shapes import check_shape

MACOS = Path(__file__).resolve().parents[2] / "macos"
SOURCES = MACOS / "Sources"

pytestmark = pytest.mark.skipif(not SOURCES.is_dir(), reason="macos/Sources jeszcze nie istnieje")

#: The four spellings the DTOs use. Only a bare ``try c.decode`` is a *required*
#: key; ``decodeIfPresent`` and every ``try?`` form tolerate a missing or null
#: value, so they are recorded as optional.
_DECODE = re.compile(
    r"^\s*(\w+)\s*=\s*\(?\s*try(\??)\s+c\.(decode|decodeIfPresent)"
    r"\(\s*([\w\[\]?.]+?)\.self,\s*forKey:\s*\.(\w+)",
    re.M)
_STRUCT = re.compile(r"\bpublic struct (\w+)\b[^{\n]*\{")
_ENUM = re.compile(r"\bpublic enum (\w+): String\b[^{\n]*\{")
_CASE = re.compile(r"^[ \t]*case ([\w, =\"]+)$", re.M)
_PROPERTY = re.compile(r"^\s*public (?:let|var) (\w+): ([\w\[\]]+)(\?)?\s*(?:=|$)", re.M)
_PATH = re.compile(r'"((?:api|v1)/[^"]*)"')
_INTERPOLATION = re.compile(r"\\\([^)]*\)")


def swift_files() -> list[Path]:
    return sorted(SOURCES.rglob("*.swift"))


def _body(source: str, open_brace: int) -> str:
    """Text between ``{`` at ``open_brace`` and its matching ``}``."""
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1:index]
    return source[open_brace + 1:]


def parse_spec() -> tuple[dict[str, list[tuple[str, str, bool]]], dict[str, set[str]]]:
    """``({struct: [(field, type, optional)]}, {enum: {values}})`` from the Swift sources."""
    spec: dict[str, list[tuple[str, str, bool]]] = {}
    enums: dict[str, set[str]] = {}
    for path in swift_files():
        source = path.read_text(encoding="utf-8")
        for match in _ENUM.finditer(source):
            values: set[str] = set()
            for raw in _CASE.findall(_body(source, match.end() - 1)):
                for case in raw.split(","):
                    name, _, literal = case.partition("=")
                    value = literal.strip().strip('"') or name.strip()
                    if value:
                        values.add(value)
            if values:
                enums[match.group(1)] = values
        for match in _STRUCT.finditer(source):
            body = _body(source, match.end() - 1)
            decoded = _DECODE.findall(body)
            if decoded:
                fields = [(key, _clean_type(swift_type),
                           bool(optional) or kind == "decodeIfPresent")
                          for _, optional, kind, swift_type, key in decoded]
            else:
                fields = [(field, _clean_type(swift_type), bool(optional))
                          for field, swift_type, optional in _PROPERTY.findall(body)]
            if fields:
                spec[match.group(1)] = fields
    return spec, enums


def _clean_type(swift_type: str) -> str:
    return swift_type.rstrip("?").replace("Swift.", "")


def check(value: Any, type_name: str) -> None:
    spec, enums = parse_spec()
    assert type_name in spec, f"macos/Sources nie definiuje struktury {type_name}"
    problems = check_shape(value, type_name, spec, enums, require_optional=False)
    assert not problems, f"{type_name} nie pasuje do struktur Swift:\n" + "\n".join(problems)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def route_patterns(app: Any) -> list[re.Pattern[str]]:
    patterns = []
    for resource in app.router.resources():
        canonical = resource.canonical
        patterns.append(re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", canonical) + "$"))
    return patterns


async def test_every_swift_path_is_a_route(client: TestClient) -> None:
    patterns = route_patterns(client.app)
    missing: list[str] = []
    for path in sorted({p for f in swift_files() for p in _PATH.findall(f.read_text(encoding="utf-8"))}):
        concrete = "/" + _INTERPOLATION.sub("x", path).strip("/")
        if not any(p.match(concrete) for p in patterns):
            missing.append(path)
    assert not missing, f"ścieżki używane przez aplikację macOS bez trasy na serwerze: {missing}"


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


async def test_health_matches_swift(client: TestClient) -> None:
    check(await (await client.get("/api/health")).json(), "Health")


async def test_bootstrap_matches_swift(client: TestClient) -> None:
    body = await (await client.get("/api/bootstrap")).json()
    check(body, "Bootstrap")
    check(body["settings"], "AppSettings")
    for model in body["models"]:
        check(model, "ModelInfo")
    for provider in body["providers"]:
        check(provider, "ProviderInfo")


async def test_chat_objects_match_swift(client: TestClient, engine) -> None:
    engine.script = ["Odpowiedź"]
    folder = await (await client.post("/api/folders", json={"name": "Projekt"})).json()
    check(folder, "Folder")

    memory = await (await client.post("/api/memories",
                                      json={"content": "Pamiętaj o kotach."})).json()
    check(memory, "MemoryEntry")

    response = await client.post("/api/chat", json={"content": "Cześć", "model": CHAT_KEY})
    import json as _json

    frames = []
    async for raw in response.content:
        line = raw.decode().strip()
        if line.startswith("data: "):
            frames.append(_json.loads(line[6:]))
    start = frames[0]
    check(start["userMessage"], "Message")
    done = next(f for f in frames if f["type"] == "done")
    check(done["message"], "Message")

    detail = await (await client.get(f"/api/conversations/{start['conversationId']}")).json()
    check(detail, "ConversationDetail")
    check(detail["conversation"], "Conversation")
    for message in detail["messages"]:
        check(message, "Message")

    hits = await (await client.get("/api/search?q=Cześć")).json()
    for hit in hits:
        check(hit, "SearchHit")


async def test_attachment_matches_swift(client: TestClient) -> None:
    from aiohttp import FormData

    form = FormData()
    form.add_field("files", b"tresc", filename="plik.txt", content_type="text/plain")
    attachments = await (await client.post("/api/upload", data=form)).json()
    check(attachments[0], "Attachment")


async def test_knowledge_matches_swift(client: TestClient) -> None:
    from aiohttp import FormData

    form = FormData()
    form.add_field("scope", "global")
    form.add_field("files", b"# Dokument\n\ntresc", filename="dok.md", content_type="text/markdown")
    documents = await (await client.post("/api/knowledge", data=form)).json()
    check(documents[0], "KbDocument")


# --------------------------------------------------------------------------- #
# LocalCode payloads (library, runtime, downloads, catalogue, agent)
# --------------------------------------------------------------------------- #


async def test_app_settings_matches_swift(client: TestClient) -> None:
    for body in (
        await (await client.get("/api/app-settings")).json(),
        await (await client.patch("/api/app-settings", json={"language": "pl"})).json(),
        await (await client.put("/api/app-settings/hf-token", json={"token": None})).json(),
        await (await client.put("/api/app-settings/favorite-models",
                                json={"key": "remote:openai:gpt-5", "favorite": True})).json(),
    ):
        check(body, "AppConfig")
        check(body["agent"], "AgentAppConfig")
        check(body["chat"], "ChatAppConfig")
        check(body["server"], "ServerAppConfig")
        check(body["catalog"], "CatalogAppConfig")


async def test_remote_providers_match_swift(client: TestClient, core) -> None:  # type: ignore[no-untyped-def]
    from localcode.remote import RemoteRegistry

    core.runtime.remote = RemoteRegistry(core.settings)         # no key: no network
    for provider in await (await client.get("/api/remote/providers")).json():
        check(provider, "RemoteProviderDTO")
    patched = await client.patch("/api/remote/deepinfra", json={"serviceTier": "flex"})
    for provider in await patched.json():
        check(provider, "RemoteProviderDTO")


async def test_library_matches_swift(client: TestClient) -> None:
    for model in await (await client.get("/api/library")).json():
        check(model, "LocalModelDTO")


async def test_runtime_matches_swift(client: TestClient) -> None:
    body = await (await client.get("/api/runtime")).json()
    check(body, "RuntimeStatus")
    check(body["system"], "SystemMemory")
    check(body["budget"], "MemoryBudget")
    check(body["settings"], "RuntimeSettings")
    for instance in body["instances"]:
        check(instance, "RuntimeInstance")


async def test_runtime_estimate_matches_swift(client: TestClient) -> None:
    body = await (await client.post("/api/runtime/estimate", json={"modelKey": CHAT_KEY})).json()
    check(body, "MemoryEstimate")


async def test_catalog_matches_swift(client: TestClient) -> None:
    """The store rows, including the trust badge the cards draw.

    This endpoint had no contract test at all, which is how a field can be
    added on one side and silently ignored on the other.
    """
    entries = await (await client.get("/api/catalog?limit=5")).json()
    assert entries
    for entry in entries:
        check(entry, "CatalogEntry")
        check(entry["source"], "CatalogSource")
        check(entry["trust"], "CatalogTrust")
        check(entry["publisher"], "CatalogPublisher")
        for variant in entry["variants"]:
            check(variant, "CatalogVariant")
        if entry.get("recommendedVariant"):
            check(entry["recommendedVariant"], "CatalogVariant")


async def test_hub_matches_swift(client: TestClient) -> None:
    for model in await (await client.get("/api/hub/search?q=qwen")).json():
        check(model, "HubModel")
    detail = await (await client.get("/api/hub/model?repo=unsloth/Qwen3-0.6B-GGUF")).json()
    check(detail, "HubModelDetail")
    for variant in detail["variants"]:
        check(variant, "HubVariant")
    for entry in detail["files"]:
        check(entry, "HubFile")


async def test_downloads_match_swift(client: TestClient) -> None:
    job = await (await client.post("/api/downloads",
                                   json={"repo": "unsloth/Qwen3-0.6B-GGUF"})).json()
    check(job, "DownloadJob")
    for listed in await (await client.get("/api/downloads")).json():
        check(listed, "DownloadJob")


async def test_agent_session_matches_swift(client: TestClient, tmp_path) -> None:
    session = await (await client.post("/api/agent/sessions",
                                       json={"cwd": str(tmp_path)})).json()
    check(session, "AgentSessionDTO")
    snapshot = await (await client.get(f"/api/agent/sessions/{session['id']}")).json()
    check(snapshot, "AgentSessionSnapshot")


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #


async def test_every_route_is_documented(client: TestClient) -> None:
    """Every ``/api`` and ``/v1`` route this server registers appears in the docs.

    The docs are the contract the Mac app and the OpenAI-compatible clients are
    written against, so a route that exists only in the code is a route nobody
    outside this repository can discover.
    """
    root = MACOS.parent
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((root / "docs").glob("*.md")))
    text += (root / "README.md").read_text(encoding="utf-8")
    routes = sorted({
        re.sub(r"\{(\w+)(?::[^}]*)?\}", r"{\1}", resource.canonical)
        for resource in client.app.router.resources()
        if resource.canonical.startswith(("/api", "/v1"))
    })
    missing = [route for route in routes if route not in text]
    assert not missing, f"trasy nieopisane w docs/ ani w README.md: {missing}"


def test_image_dto_keeps_the_untranslated_prompt() -> None:
    """The Mac app must decode ``originalPrompt``.

    Prompt translation is on by default, so ``prompt`` is the English text that
    reached the model and ``originalPrompt`` is what the user actually typed
    (``localcode/images/gallery.py``). Without it "Użyj ponownie" puts the
    translation into the composer and the user's own words are unrecoverable.
    """
    spec, _ = parse_spec()
    assert "originalPrompt" in {name for name, _, _ in spec["ImageDTO"]}


# --------------------------------------------------------------------------- #
# Wideo — the screen the app got last, and the one nobody can click here
# --------------------------------------------------------------------------- #


async def test_video_models_match_swift(client: TestClient) -> None:
    """`GET /api/video/models` against `VideoModelDTO`.

    The section cannot be exercised by clicking in this environment (no
    accessibility permission for UI scripting), so the contract check *is* the
    verification: every field the screen reads has to exist in the payload.
    """
    for model in await (await client.get("/api/video/models")).json():
        check(model, "VideoModelDTO")


async def test_video_estimate_matches_swift(client: TestClient) -> None:
    body = await (await client.post("/api/video/estimate",
                                    json={"width": 704, "height": 480, "numFrames": 25,
                                          "fps": 24, "steps": 8})).json()
    check(body, "VideoEstimate")


async def test_the_video_progress_frame_matches_swift(client: TestClient) -> None:
    """The SSE frames the screen draws its progress bar from."""
    from localcode.video.base import VideoProgress

    progress = VideoProgress(stage="denoising", step=3, steps=8,
                             message="Odszumianie: krok 3/8").to_dict()
    for key in ("stage", "step", "steps", "message", "fraction"):
        assert key in progress, key
    assert 0.0 <= progress["fraction"] <= 1.0


async def test_the_finished_clip_matches_swift(client: TestClient) -> None:
    """`VideoDTO` — what the player, the summary line and "Zapisz jako…" read."""
    from localcode.video.base import VideoResult

    # The numbers are the real ones from the first clip this engine produced.
    result = VideoResult(id="klip-1", path="/tmp/klip.mp4", width=704, height=480,
                         num_frames=25, fps=24, seed=7, steps=8, guidance=1.0,
                         model_key="city96/LTX-Video-gguf/ltx-video-2b-v0.9-Q4_K_M",
                         prompt="a ripe yellow banana on a wooden table",
                         elapsed_ms=86_700, size_bytes=18_040, estimated_seconds=86.8)

    check(result.to_dict(), "VideoDTO")
