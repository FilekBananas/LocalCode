"""Tests for ``/api/remote/*`` — the route the "Modele zdalne" panel reads.

The registry itself is covered in ``tests/remote``; what is covered here is the
step between it and the Mac app, because that is where a newly added provider
can silently fail to appear: the panel builds its rows from this endpoint alone.

No key is configured in these tests, so the registry never touches the network.
"""

from __future__ import annotations

from typing import Any

import pytest
from aiohttp.test_utils import TestClient

from localcode.config import RemoteProvider, Settings
from localcode.remote import DEFAULT_BASE_URLS, PROVIDERS, RemoteRegistry


@pytest.fixture
def runtime(engine, settings: Settings):  # type: ignore[no-untyped-def]
    """The standard runtime fake, plus a real (keyless) remote registry."""
    from server_fakes import CHAT_KEY, CHAT_KEY_2, EMBED_KEY, FakeEngine, FakeRuntime

    fake = FakeRuntime({CHAT_KEY: engine, CHAT_KEY_2: engine, EMBED_KEY: FakeEngine()})
    fake.remote = RemoteRegistry(settings)
    return fake


async def rows(client: TestClient) -> list[dict[str, Any]]:
    response = await client.get("/api/remote/providers")
    assert response.status == 200
    return await response.json()


async def test_every_provider_reaches_the_app(client: TestClient) -> None:
    listed = await rows(client)
    assert [row["id"] for row in listed] == list(PROVIDERS)
    for row in listed:
        assert row["label"] and row["keyUrl"].startswith("https://")
        assert row["baseUrl"] == DEFAULT_BASE_URLS[row["id"]]
        assert row["hasKey"] is False


async def test_unlid_is_offered_with_its_own_endpoint(client: TestClient) -> None:
    unlid = next(row for row in await rows(client) if row["id"] == "unlid")
    assert unlid["label"] == "Unlid"
    assert unlid["baseUrl"] == "https://api.unlid.ai/v1"
    assert unlid["keyUrl"] == "https://unlid.ai/account"


async def test_a_key_is_never_sent_back(client: TestClient, settings: Settings) -> None:
    settings.remote.unlid = RemoteProvider(api_key="ul-sekret")
    listed = await rows(client)
    unlid = next(row for row in listed if row["id"] == "unlid")
    assert unlid["hasKey"] is True
    assert "ul-sekret" not in str(listed)


async def test_an_unknown_provider_is_a_404_naming_the_known_ones(client: TestClient) -> None:
    response = await client.put("/api/remote/nieistniejacy/key", json={"key": "x"})
    assert response.status == 404
    body = await response.json()
    assert body["code"] == "unknown_provider"
    assert "unlid" in body["error"]


# --------------------------------------------------------------------------- #
# Service tiers
# --------------------------------------------------------------------------- #


async def test_tiers_are_offered_only_where_the_provider_has_them(client: TestClient) -> None:
    by_id = {row["id"]: row for row in await rows(client)}
    for name in ("openai", "deepinfra", "unlid"):
        assert by_id[name]["serviceTiers"] == ["standard", "flex", "priority"]
        assert by_id[name]["serviceTier"] == "standard"
    for name in ("anthropic", "gemini"):
        assert by_id[name]["serviceTiers"] == []
        assert by_id[name]["serviceTier"] == "standard"


async def test_a_tier_is_saved_and_shown(client: TestClient, settings: Settings) -> None:
    from localcode.config import load_settings

    response = await client.patch("/api/remote/deepinfra", json={"serviceTier": "flex"})
    assert response.status == 200
    deepinfra = next(row for row in await response.json() if row["id"] == "deepinfra")
    assert deepinfra["serviceTier"] == "flex"
    assert deepinfra["enabled"] is True                     # untouched
    assert settings.remote.deepinfra.service_tier == "flex"
    assert load_settings().remote.deepinfra.service_tier == "flex"   # settings.json

    back = await client.patch("/api/remote/deepinfra", json={"serviceTier": "Standard"})
    assert back.status == 200
    assert load_settings().remote.deepinfra.service_tier == "standard"


async def test_the_tier_and_the_switch_go_together(client: TestClient) -> None:
    from localcode.config import load_settings

    response = await client.patch("/api/remote/unlid",
                                  json={"enabled": False, "serviceTier": "priority"})
    assert response.status == 200
    stored = load_settings().remote.unlid
    assert stored.enabled is False and stored.service_tier == "priority"


@pytest.mark.parametrize("provider, body, code", [
    ("anthropic", {"serviceTier": "flex"}, "tier_unsupported"),
    ("gemini", {"serviceTier": "standard"}, "tier_unsupported"),
    ("deepinfra", {"serviceTier": "turbo"}, "invalid_service_tier"),
    ("deepinfra", {"serviceTier": 2}, "invalid_service_tier"),
    ("deepinfra", {"serviceTier": None}, "invalid_service_tier"),
    ("deepinfra", {"enabled": "tak"}, "invalid_param"),
    ("deepinfra", {}, "missing_enabled"),
])
async def test_a_bad_patch_is_a_polish_400(client: TestClient, provider: str,
                                           body: dict[str, Any], code: str) -> None:
    response = await client.patch(f"/api/remote/{provider}", json=body)
    assert response.status == 400
    answer = await response.json()
    assert answer["code"] == code
    assert answer["error"]


async def test_a_bad_tier_saves_nothing(client: TestClient) -> None:
    from localcode.config import load_settings

    response = await client.patch("/api/remote/deepinfra",
                                  json={"enabled": False, "serviceTier": "turbo"})
    assert response.status == 400
    assert "Dostępne: standard, flex, priority" in (await response.json())["error"]
    assert load_settings().remote.deepinfra.enabled is True


async def test_unsupported_provider_names_the_ones_with_tiers(client: TestClient) -> None:
    response = await client.patch("/api/remote/anthropic", json={"serviceTier": "flex"})
    message = (await response.json())["error"]
    assert message.startswith("Anthropic nie ma trybów")
    assert "DeepInfra" in message and "Unlid" in message


async def test_a_new_tier_reaches_the_next_request(client: TestClient, core: Any,
                                                   settings: Settings) -> None:
    """The registry hands out a client with the new tier — no restart, no key re-entry."""
    from localcode.config import update_settings
    from localcode.remote import RemoteClient

    class Listing:
        """A /models answer from memory: no network in these tests."""

        def get(self, *a: Any, **kw: Any) -> "Listing":
            return self

        async def text(self) -> str:
            return '{"data": [{"id": "m1"}]}'

        status = 200

        async def __aenter__(self) -> "Listing":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    def factory(provider: str, key: str, base: str | None) -> RemoteClient:
        return RemoteClient(provider, key, base_url=base, session_factory=Listing)

    update_settings({"remote": {"deepinfra": {"api_key": "di-test-key"}}})
    settings.remote.deepinfra = RemoteProvider(api_key="di-test-key")
    core.runtime.remote = RemoteRegistry(settings, client_factory=factory)
    before = core.runtime.remote.client("deepinfra")
    assert before.service_tier == "standard"

    response = await client.patch("/api/remote/deepinfra", json={"serviceTier": "priority"})
    assert response.status == 200
    deepinfra = next(row for row in await response.json() if row["id"] == "deepinfra")
    assert deepinfra["serviceTier"] == "priority" and deepinfra["modelCount"] == 1
    after = core.runtime.remote.client("deepinfra")
    assert after is not before and after.service_tier == "priority"
    assert "di-test-key" not in str(await rows(client))
