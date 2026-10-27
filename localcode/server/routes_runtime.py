"""Health, the server event stream and the runtime (loaded models) API.

| route | handler |
|---|---|
| ``GET /api/health`` | ``{"ok": true, "version": …}`` — the only unauthenticated route |
| ``GET /api/events`` | SSE: ``runtime`` / ``download`` / ``library`` events + heartbeats |
| ``GET /api/runtime`` | :meth:`RuntimeManager.status` |
| ``PATCH /api/runtime/settings`` | :meth:`RuntimeManager.update_settings` |
| ``POST /api/runtime/load`` | :meth:`RuntimeManager.load` (``wait=False``) |
| ``POST /api/runtime/unload`` | :meth:`RuntimeManager.unload` / ``unload_all`` (idempotent) |
| ``POST /api/runtime/estimate`` | :meth:`RuntimeManager.estimate` |
| ``POST /api/runtime/tune`` | :func:`~localcode.runtime.tuner.recommend` (+ optional model refinement, + apply) |

The event stream opens with the current runtime status so a client that
connects late is immediately in sync, then forwards everything the
:class:`~localcode.server.core.EventBus` publishes. A ``: ping`` comment every
:data:`~localcode.server.app.EVENTS_HEARTBEAT_S` seconds keeps the connection
alive and makes a disconnect visible to the server.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from aiohttp import web

from localcode.chat.repo import ApiError
from localcode.runtime.base import LoadConfig
from localcode.server.app import EVENTS_HEARTBEAT_S, get_core, json_body, json_response, stream_sse
from localcode.server.core import Core

__all__ = ["register"]


def register(app: web.Application) -> None:
    app.router.add_get("/api/health", health)
    app.router.add_get("/api/events", events)
    app.router.add_get("/api/runtime", runtime_status)
    app.router.add_patch("/api/runtime/settings", runtime_settings)
    app.router.add_post("/api/runtime/load", runtime_load)
    app.router.add_post("/api/runtime/unload", runtime_unload)
    app.router.add_post("/api/runtime/estimate", runtime_estimate)
    app.router.add_post("/api/runtime/tune", runtime_tune)


async def health(request: web.Request) -> web.Response:
    """``GET /api/health`` — reachable without a token (used to probe the port)."""
    core = get_core(request)
    return json_response({"ok": True, "version": core.version})


async def _event_source(core: Core) -> AsyncIterator[dict[str, Any]]:
    """Runtime snapshot first, then every published event until the client leaves."""
    sub = core.events.subscribe()
    try:
        try:
            yield {"type": "runtime", **core.runtime.status()}
        except Exception:  # noqa: BLE001 - a broken snapshot must not kill the stream
            yield {"type": "runtime", "instances": []}
        while True:
            event = await sub.get()
            if event is None:
                return
            yield event
    finally:
        sub.close()


async def events(request: web.Request) -> web.StreamResponse:
    """``GET /api/events`` — server-sent events for the whole app."""
    core = get_core(request)
    return await stream_sse(request, _event_source(core), heartbeat=EVENTS_HEARTBEAT_S)


async def runtime_status(request: web.Request) -> web.Response:
    """``GET /api/runtime`` → ``RuntimeStatus``."""
    return json_response(get_core(request).runtime.status())


async def runtime_settings(request: web.Request) -> web.Response:
    """``PATCH /api/runtime/settings {partial}`` → ``RuntimeStatus``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    return json_response(await core.runtime.update_settings(body))


def _load_config(core: Core, raw: Any) -> LoadConfig | None:
    """``LoadConfigDTO`` → :class:`LoadConfig`; missing fields fall back to settings."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ApiError(400, "Pole „config” musi być obiektem.", "invalid_config")
    if not raw:
        return None
    rt = core.settings.runtime
    base = LoadConfig(
        n_ctx=0,
        n_parallel=max(1, int(rt.default_parallel or 1)),
        gpu_layers=int(rt.gpu_layers),
        flash_attn=str(rt.flash_attn or "auto"),
        kv_cache_type=str(rt.kv_cache_type or "f16"),
    )
    try:
        return LoadConfig.from_dict(raw, base)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, f"Nieprawidłowa konfiguracja ładowania: {exc}", "invalid_config") from exc


async def runtime_load(request: web.Request) -> web.Response:
    """``POST /api/runtime/load {modelKey, config?, newInstance?}`` → the instance DTO."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    key = str(body.get("modelKey") or "").strip()
    if not key:
        raise ApiError(400, "Brak pola „modelKey”.", "missing_model_key")
    model = core.resolve_model(key)
    config = _load_config(core, body.get("config"))
    info = await core.runtime.load(model.key if model else key, config,
                                   new_instance=bool(body.get("newInstance")), wait=False)
    return json_response(info)


async def runtime_unload(request: web.Request) -> web.Response:
    """``POST /api/runtime/unload {instanceId} | {all: true}`` → ``RuntimeStatus``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    if body.get("all"):
        await core.runtime.unload_all()
    else:
        instance_id = str(body.get("instanceId") or "").strip()
        if not instance_id:
            raise ApiError(400, "Podaj „instanceId” albo {\"all\": true}.", "missing_instance_id")
        await core.runtime.unload(instance_id)
    return json_response(core.runtime.status())


async def runtime_estimate(request: web.Request) -> web.Response:
    """``POST /api/runtime/estimate {modelKey, config?}`` → ``MemoryEstimate``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    key = str(body.get("modelKey") or "").strip()
    if not key:
        raise ApiError(400, "Brak pola „modelKey”.", "missing_model_key")
    model = core.resolve_model(key)
    estimate = core.runtime.estimate(model.key if model else key, _load_config(core, body.get("config")))
    return json_response(estimate.to_dict())


async def runtime_tune(request: web.Request) -> web.Response:
    """``POST /api/runtime/tune {modelKey?, useModel?, apply?}`` → tuning recommendation.

    Always computes the deterministic proposal (``runtime.tuner.recommend``);
    with ``modelKey`` set and that model currently loaded, ``useModel``
    (default ``true``) also asks it for a second opinion — a no-op when the
    model is not loaded, never a trigger to load one. ``apply: true`` writes
    the proposal through :meth:`RuntimeManager.update_settings`, the same
    validated path ``PATCH /api/runtime/settings`` uses.
    """
    from localcode.runtime import tuner

    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    key = str(body.get("modelKey") or "").strip()
    model = core.resolve_model(key) if key else None
    facts = tuner.gather_machine_facts()
    proposal = tuner.recommend(facts, model)
    if model is not None and bool(body.get("useModel", True)):
        proposal = await tuner.refine_with_model(core.runtime, model.key, proposal)
    dto = proposal.to_dict()
    result: dict[str, Any] = {"recommendation": dto}
    if body.get("apply"):
        result["status"] = await core.runtime.update_settings(dto["patch"])
    return json_response(result)
