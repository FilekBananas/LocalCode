"""Remote providers over HTTP: keys in, borrowed models out.

| route | what it does |
| --- | --- |
| ``GET  /api/remote/providers`` | every provider, which have a key, how many models |
| ``PUT  /api/remote/{provider}/key`` | save or clear a key (checked before it is saved) |
| ``PATCH /api/remote/{provider}`` | switch a provider off without losing the key; pick its service tier |
| ``GET  /api/remote/models`` | every borrowed model, as library rows |

A key is never sent back: the provider list says ``hasKey``, not the key.  It is
also verified before it is stored — pasting a key that does not work and finding
out three screens later is the kind of thing this app should not do.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from localcode.remote import (
    DEFAULT_BASE_URLS,
    PROVIDER_KEY_URLS,
    PROVIDER_LABELS,
    PROVIDERS,
    SERVICE_TIERS,
    TIERED_PROVIDERS,
    RemoteError,
    probe,
)
from localcode.server.app import ApiError, get_core, json_body, json_response


def register(app: web.Application) -> None:
    app.router.add_get("/api/remote/providers", remote_providers)
    app.router.add_get("/api/remote/models", remote_models)
    app.router.add_put("/api/remote/{provider}/key", remote_set_key)
    app.router.add_patch("/api/remote/{provider}", remote_patch)


def _registry(core: Any) -> Any:
    return core.runtime.remote


def _provider(request: web.Request) -> str:
    name = str(request.match_info.get("provider") or "").strip().lower()
    if name not in PROVIDERS:
        raise ApiError(404, f"Nieznany dostawca „{name}”. Dostępni: {', '.join(PROVIDERS)}.",
                       "unknown_provider")
    return name


def _error(exc: RemoteError) -> ApiError:
    return ApiError(exc.status, exc.message, exc.code)


async def remote_providers(request: web.Request) -> web.Response:
    """``GET /api/remote/providers`` → one row per provider (never the key itself)."""
    core = get_core(request)
    remote = getattr(core.settings, "remote", None)
    registry = _registry(core)
    counts: dict[str, int] = {}
    for name in registry.providers:
        try:
            counts[name] = len(await registry.client(name).models())
        except RemoteError:
            counts[name] = -1          # configured, but not answering right now
    rows = []
    for name in PROVIDERS:
        config = getattr(remote, name, None)
        rows.append({
            "id": name,
            "label": PROVIDER_LABELS[name],
            "hasKey": bool(config and (config.api_key or "").strip()),
            "enabled": bool(config and config.enabled),
            "baseUrl": (config.base_url if config else None) or DEFAULT_BASE_URLS[name],
            "keyUrl": PROVIDER_KEY_URLS[name],
            "modelCount": counts.get(name, 0),
            # standard | flex | priority; the list is empty for a provider
            # that has no tiers (Anthropic, Gemini), so the app shows no picker.
            "serviceTier": (config.service_tier if config and name in TIERED_PROVIDERS
                            else "standard"),
            "serviceTiers": list(SERVICE_TIERS) if name in TIERED_PROVIDERS else [],
        })
    return json_response(rows)


async def remote_models(request: web.Request) -> web.Response:
    """``GET /api/remote/models?provider=&refresh=`` → library rows for borrowed models."""
    core = get_core(request)
    provider = (request.query.get("provider") or "").strip() or None
    if provider and provider not in PROVIDERS:
        raise ApiError(404, f"Nieznany dostawca „{provider}”.", "unknown_provider")
    refresh = str(request.query.get("refresh") or "").lower() in ("1", "true", "yes")
    try:
        models = await _registry(core).models(provider, refresh=refresh)
    except RemoteError as exc:
        raise _error(exc) from exc
    return json_response([m.to_dict() for m in models])


async def remote_set_key(request: web.Request) -> web.Response:
    """``PUT /api/remote/{provider}/key {key}`` — verify, then save (``null`` clears)."""
    core = get_core(request)
    provider = _provider(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    raw = body.get("key")
    if raw is not None and not isinstance(raw, str):
        raise ApiError(400, "Klucz musi być tekstem albo null.", "invalid_key")
    key = (raw or "").strip() or None
    base_url = str(body.get("baseUrl") or "").strip() or None

    if key is not None:
        try:
            await probe(provider, key, base_url=base_url)
        except RemoteError as exc:
            raise _error(exc) from exc

    await core.set_remote_key(provider, key, base_url=base_url)
    _registry(core).forget(provider)
    return await remote_providers(request)


async def remote_patch(request: web.Request) -> web.Response:
    """``PATCH /api/remote/{provider} {enabled?, serviceTier?}`` — on/off and the tier.

    Either field alone or both; both are checked before anything is saved, so a
    bad tier does not leave the provider half-changed.
    """
    core = get_core(request)
    provider = _provider(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict) or not ({"enabled", "serviceTier"} & body.keys()):
        raise ApiError(400, "Brak pola „enabled” ani „serviceTier”.", "missing_enabled")
    enabled = body.get("enabled")
    if "enabled" in body and not isinstance(enabled, bool):
        raise ApiError(400, "Pole „enabled” musi być wartością logiczną.", "invalid_param")
    tier = _service_tier(provider, body) if "serviceTier" in body else None
    await core.set_remote_options(provider, enabled=enabled, service_tier=tier)
    if "enabled" in body:
        _registry(core).forget(provider)
    # A new tier needs no forget(): the registry rebuilds a client whose tier
    # no longer matches the settings, and keeps its model list while at it.
    return await remote_providers(request)


def _service_tier(provider: str, body: dict[str, Any]) -> str:
    """``serviceTier`` from a PATCH body, or a Polish 400."""
    label = PROVIDER_LABELS[provider]
    if provider not in TIERED_PROVIDERS:
        offered = ", ".join(PROVIDER_LABELS[p] for p in TIERED_PROVIDERS)
        raise ApiError(400, f"{label} nie ma trybów Standard / Flex / Priority. "
                            f"Mają je: {offered}.", "tier_unsupported")
    raw = body.get("serviceTier")
    tier = raw.strip().lower() if isinstance(raw, str) else ""
    if tier not in SERVICE_TIERS:
        raise ApiError(400, f"Nieznany tryb „{raw}”. Dostępne: {', '.join(SERVICE_TIERS)}.",
                       "invalid_service_tier")
    return tier
