"""Hub, downloads, library and app-settings endpoints."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient

from localcode.hub.search import HubError
from server_fakes import CHAT_KEY, FakeHub

REPO = "unsloth/Qwen3-0.6B-GGUF"


# --------------------------------------------------------------------------- #
# Hub
# --------------------------------------------------------------------------- #


async def test_hub_search(client: TestClient, hub: FakeHub) -> None:
    response = await client.get("/api/hub/search?q=qwen&format=gguf&sort=likes&limit=5")
    assert response.status == 200
    body = await response.json()
    assert body[0]["id"] == REPO
    assert hub.searches == [("qwen", "gguf", "likes", 5)]


async def test_hub_search_limit_is_clamped(client: TestClient, hub: FakeHub) -> None:
    await client.get("/api/hub/search?limit=9999")
    assert hub.searches[-1][3] == 100
    bad = await client.get("/api/hub/search?limit=abc")
    assert bad.status == 400 and (await bad.json())["code"] == "invalid_param"


async def test_hub_curated_and_model(client: TestClient) -> None:
    curated = await (await client.get("/api/hub/curated")).json()
    assert curated[0]["format"] == "gguf"
    detail = await (await client.get(f"/api/hub/model?repo={REPO}")).json()
    assert detail["model"]["id"] == REPO
    assert detail["variants"][0]["recommended"] is True


async def test_hub_model_requires_repo(client: TestClient) -> None:
    response = await client.get("/api/hub/model")
    assert response.status == 400
    assert (await response.json())["code"] == "missing_repo"


async def test_hub_error_is_mapped(client: TestClient, hub: FakeHub) -> None:
    hub.error = HubError("Brak połączenia z Hugging Face.", "network", 503)
    response = await client.get("/api/hub/search")
    assert response.status == 503
    body = await response.json()
    assert body == {"error": "Brak połączenia z Hugging Face.", "code": "network"}


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


async def test_downloads_lifecycle(client: TestClient, core) -> None:
    assert await (await client.get("/api/downloads")).json() == []

    started = await client.post("/api/downloads", json={"repo": REPO, "variantId": "Qwen3-0.6B-Q8_0"})
    assert started.status == 201
    job = await started.json()
    assert job["files"] == ["Qwen3-0.6B-Q8_0.gguf"]
    assert job["state"] == "queued"

    listed = await (await client.get("/api/downloads")).json()
    assert [j["id"] for j in listed] == [job["id"]]

    cancelled = await client.delete(f"/api/downloads/{job['id']}")
    assert cancelled.status == 200
    assert (await cancelled.json())["state"] == "cancelled"


async def test_downloads_defaults_to_recommended_variant(client: TestClient) -> None:
    job = await (await client.post("/api/downloads", json={"repo": REPO})).json()
    assert job["files"] == ["Qwen3-0.6B-Q4_K_M.gguf"]


async def test_downloads_explicit_files(client: TestClient) -> None:
    job = await (await client.post("/api/downloads",
                                   json={"repo": REPO, "files": ["a.gguf", "b.gguf"]})).json()
    assert job["files"] == ["a.gguf", "b.gguf"]


async def test_downloads_requires_repo(client: TestClient) -> None:
    response = await client.post("/api/downloads", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_repo"


async def test_download_cancel_unknown(client: TestClient) -> None:
    response = await client.delete("/api/downloads/nie-ma")
    assert response.status == 404


async def test_finished_download_rescans_the_library(client: TestClient, core, library) -> None:
    job = await (await client.post("/api/downloads", json={"repo": REPO})).json()
    before = library.scans
    core.downloads_fake.finish(job["id"])
    for _ in range(50):
        if library.scans > before:
            break
        await _tick()
    assert library.scans > before


async def _tick() -> None:
    import asyncio

    await asyncio.sleep(0.02)


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #


async def test_library_list(client: TestClient) -> None:
    models = await (await client.get("/api/library")).json()
    assert {m["key"] for m in models} == {
        CHAT_KEY, "mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0"}
    first = models[0]
    for key in ("key", "format", "displayName", "sizeBytes", "capabilities", "loaded", "loadDefaults"):
        assert key in first
    assert first["loaded"] is False and first["loadDefaults"] is None


async def test_library_reports_loaded_models(client: TestClient, runtime) -> None:
    await runtime.load(CHAT_KEY)
    models = await (await client.get("/api/library")).json()
    assert [m["loaded"] for m in models if m["key"] == CHAT_KEY] == [True]


async def test_library_rescan(client: TestClient, library) -> None:
    before = library.scans
    response = await client.post("/api/library/rescan")
    assert response.status == 200
    assert library.scans == before + 1


async def test_library_load_defaults(client: TestClient) -> None:
    response = await client.patch(f"/api/library/{CHAT_KEY}",
                                  json={"loadDefaults": {"nCtx": 4096, "nParallel": 2}})
    assert response.status == 200
    body = await response.json()
    assert body["loadDefaults"] == {"nCtx": 4096, "nParallel": 2}
    models = await (await client.get("/api/library")).json()
    assert next(m for m in models if m["key"] == CHAT_KEY)["loadDefaults"]["nCtx"] == 4096


async def test_library_patch_requires_field(client: TestClient) -> None:
    response = await client.patch(f"/api/library/{CHAT_KEY}", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_field"


async def test_library_patch_unknown_model(client: TestClient) -> None:
    response = await client.patch("/api/library/org/nie-ma", json={"loadDefaults": None})
    assert response.status == 404


async def test_library_delete(client: TestClient, library) -> None:
    response = await client.delete(f"/api/library/{CHAT_KEY}")
    assert response.status == 204
    assert library.deleted == [CHAT_KEY]


# --------------------------------------------------------------------------- #
# Application settings
# --------------------------------------------------------------------------- #


async def test_app_settings_shape(client: TestClient) -> None:
    body = await (await client.get("/api/app-settings")).json()
    assert set(body) >= {"modelsDir", "hfTokenSet", "language", "agent", "chat", "server"}
    assert body["hfTokenSet"] is False
    assert set(body["agent"]) >= {"defaultModel", "permissionMode", "maxSteps", "thinking"}
    assert set(body["chat"]) == {"defaultModel", "embeddingModel", "reasoningEffort"}
    assert set(body["server"]) >= {"port", "openaiApiEnabled", "openaiApiRequiresToken"}


async def test_app_settings_patch(client: TestClient, core) -> None:
    response = await client.patch("/api/app-settings",
                                  json={"language": "en", "agent": {"permissionMode": "plan"},
                                        "chat": {"embeddingModel": None}})
    assert response.status == 200
    body = await response.json()
    assert body["language"] == "en"
    assert body["agent"]["permissionMode"] == "plan"
    assert core.settings.language == "en"
    assert core.settings.agent.permission_mode == "plan"


async def test_app_settings_patch_validation(client: TestClient) -> None:
    response = await client.patch("/api/app-settings", json={"agent": {"permissionMode": "turbo"}})
    assert response.status == 400
    assert (await response.json())["code"] == "validation_error"
    empty = await client.patch("/api/app-settings", json={"nieznane": 1})
    assert empty.status == 400 and (await empty.json())["code"] == "empty_patch"


async def test_favourite_models_start_empty_and_are_starred_one_at_a_time(client: TestClient, core) -> None:
    assert (await (await client.get("/api/app-settings")).json())["favoriteModels"] == []
    remote = "remote:openai:gpt-5"
    for key in (CHAT_KEY, remote, CHAT_KEY):
        response = await client.put("/api/app-settings/favorite-models",
                                    json={"key": key, "favorite": True})
        assert response.status == 200
    body = await response.json()
    # Starring twice does not duplicate; the order is the order of starring.
    assert body["favoriteModels"] == [CHAT_KEY, remote]
    assert "catalog" in body
    unstar = await client.put("/api/app-settings/favorite-models", json={"key": CHAT_KEY, "favorite": False})
    assert (await unstar.json())["favoriteModels"] == [remote]
    assert core.settings.favorite_models == [remote]


async def test_favourite_models_survive_a_restart_and_keep_another_writers_stars(client: TestClient,
                                                                                core) -> None:
    from localcode.config import load_settings, set_favorite_model

    await client.put("/api/app-settings/favorite-models", json={"key": CHAT_KEY, "favorite": True})
    # The terminal stars a model in the same settings.json while the server runs…
    set_favorite_model("remote:anthropic:claude-opus-5", True)
    # …and the server's next star must not drop it.
    await client.put("/api/app-settings/favorite-models", json={"key": "org/repo/x", "favorite": True})
    assert load_settings().favorite_models == [CHAT_KEY, "remote:anthropic:claude-opus-5", "org/repo/x"]


async def test_favourite_models_can_be_replaced_by_patch(client: TestClient) -> None:
    response = await client.patch("/api/app-settings",
                                  json={"favoriteModels": ["b", " a ", "b"]})
    assert response.status == 200
    assert (await response.json())["favoriteModels"] == ["b", "a"]


@pytest.mark.parametrize("body", [
    {"favorite": True},
    {"key": "", "favorite": True},
    {"key": 5, "favorite": True},
    {"key": "a/b/c", "favorite": "tak"},
    {"key": "a/b/c"},
])
async def test_a_malformed_favourite_request_is_refused(client: TestClient, body: dict) -> None:
    response = await client.put("/api/app-settings/favorite-models", json=body)
    assert response.status == 400
    assert (await response.json())["error"]


def test_favorites_first_is_a_stable_partition() -> None:
    from localcode.config import favorites_first

    rows = [{"key": k} for k in ("a", "b", "c", "d")]
    assert [r["key"] for r in favorites_first(rows, ["d", "b", "gone"])] == ["b", "d", "a", "c"]
    assert favorites_first(["x", "y"], None) == ["x", "y"]
    assert favorites_first([("y", 1), ("x", 2)], ["x"], key=lambda t: t[0]) == [("x", 2), ("y", 1)]


async def test_a_favourite_list_must_be_a_list_of_keys(client: TestClient) -> None:
    for bad in ("a/b/c", [1, 2], [""], None):
        response = await client.patch("/api/app-settings", json={"favoriteModels": bad})
        assert response.status == 400, bad


async def test_hf_token_is_write_only(client: TestClient, core) -> None:
    response = await client.put("/api/app-settings/hf-token", json={"token": "hf_secret"})
    assert response.status == 200
    body = await response.json()
    assert body["hfTokenSet"] is True
    assert "hf_secret" not in await (await client.get("/api/app-settings")).text()
    assert core.settings.hf_token == "hf_secret"

    cleared = await client.put("/api/app-settings/hf-token", json={"token": None})
    assert (await cleared.json())["hfTokenSet"] is False


async def test_hf_token_requires_field(client: TestClient) -> None:
    response = await client.put("/api/app-settings/hf-token", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_token"


async def test_cerber_token_is_write_only_and_saved_when_cerber_is_closed(client: TestClient, core,
                                                                          monkeypatch) -> None:
    """Cerber may be closed while the token is pasted — it is saved anyway; it is
    the check on every download that needs Cerber running."""
    from localcode.security import cerber as cerber_mod

    async def closed(self):
        raise cerber_mod.CerberError("Cerber nie odpowiada", "cerber_unavailable", 503)

    monkeypatch.setattr(cerber_mod.CerberClient, "info", closed)
    response = await client.put("/api/app-settings/cerber-token", json={"token": "cerber-secret-123456"})
    assert response.status == 200
    assert (await response.json())["cerber"]["tokenSet"] is True
    assert "cerber-secret-123456" not in await (await client.get("/api/app-settings")).text()
    assert core.settings.cerber.token == "cerber-secret-123456"


async def test_a_token_cerber_rejects_is_not_saved(client: TestClient, core, monkeypatch) -> None:
    from localcode.security import cerber as cerber_mod

    async def rejected(self):
        raise cerber_mod.CerberError("Cerber odrzucił token", "cerber_bad_token", 400)

    monkeypatch.setattr(cerber_mod.CerberClient, "info", rejected)
    response = await client.put("/api/app-settings/cerber-token", json={"token": "wrong-token-123456"})
    assert response.status == 400 and (await response.json())["code"] == "cerber_bad_token"
    assert not core.settings.cerber.token


async def test_cerber_is_switched_on_with_a_local_address_only(client: TestClient, core) -> None:
    on = await client.patch("/api/app-settings", json={"cerber": {"enabled": True}})
    assert on.status == 200 and (await on.json())["cerber"]["enabled"] is True
    bad = await client.patch("/api/app-settings", json={"cerber": {"url": "http://evil.example:8765"}})
    assert bad.status == 400 and (await bad.json())["code"] == "invalid_cerber_url"
    sneaky = await client.patch("/api/app-settings", json={"cerber": {"token": "x"}})
    assert sneaky.status == 400 and (await sneaky.json())["code"] == "use_cerber_token_endpoint"


async def test_the_cerber_check_answers_without_a_token(client: TestClient) -> None:
    response = await client.post("/api/cerber/check")
    assert response.status == 200
    body = await response.json()
    assert body["ok"] is False and "token" in body["message"].lower()


async def test_every_app_settings_response_carries_the_catalog(client: TestClient) -> None:
    """The store's tabs live in ``catalog``; every response must carry them.

    ``ModelsStore`` assigns the whole body to ``settings`` after a PATCH and
    after a token change (``ModelsStore.swift`` ``patchSettings`` /
    ``setHuggingFaceToken``), and ``catalogCategories`` reads
    ``settings.catalog.categories``. A response without ``catalog`` therefore
    empties the store's category bar until the next ``GET``.
    """
    get = await (await client.get("/api/app-settings")).json()
    assert get["catalog"]["categories"], "GET must list the store categories"

    patched = await (await client.patch("/api/app-settings", json={"language": "en"})).json()
    assert patched.get("catalog") == get["catalog"]

    token = await (await client.put("/api/app-settings/hf-token", json={"token": "hf_x"})).json()
    assert token.get("catalog") == get["catalog"]


# --------------------------------------------------------------------------- #
# Security settings (containment + scanning) over the API
# --------------------------------------------------------------------------- #


async def test_app_settings_report_how_a_model_is_contained(client) -> None:
    """The app needs to be able to show and change this, not just the file."""
    body = await (await client.get("/api/app-settings")).json()
    assert body["security"] == {"sandbox": "auto", "scanDownloads": True}


async def test_the_sandbox_mode_can_be_changed(client) -> None:
    resp = await client.patch("/api/app-settings", json={"security": {"sandbox": "strict"}})
    assert resp.status == 200
    assert (await resp.json())["security"]["sandbox"] == "strict"


async def test_scanning_downloads_can_be_switched_off(client) -> None:
    resp = await client.patch("/api/app-settings", json={"security": {"scanDownloads": False}})
    assert (await resp.json())["security"]["scanDownloads"] is False


async def test_a_nonsense_sandbox_mode_is_refused_in_polish(client) -> None:
    resp = await client.patch("/api/app-settings", json={"security": {"sandbox": "bardzo"}})
    assert resp.status == 400
    body = await resp.json()
    assert "auto, strict, off" in body["error"]
