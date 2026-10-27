"""Hugging Face hub, downloads, the local library and the application settings.

| route | handler |
|---|---|
| ``GET /api/hub/search`` | :meth:`HubClient.asearch` |
| ``GET /api/hub/curated`` | :meth:`HubClient.acurated` |
| ``GET /api/hub/model?repo=`` | :meth:`HubClient.adetails` |
| ``GET /api/catalog`` | :meth:`Catalog.entries` (the friendly in-app store) |
| ``GET /api/catalog/{id}`` | :meth:`Catalog.detail` (variants for "Zaawansowane") |
| ``POST /api/catalog/install`` | :meth:`Catalog.install` |
| ``GET/POST /api/downloads``, ``DELETE /api/downloads/{id}`` | ``DownloadManager`` |
| ``GET /api/library``, ``POST /api/library/rescan`` | :class:`ModelLibrary` |
| ``PATCH/DELETE /api/library/{key}`` | load defaults / delete from disk |
| ``GET/PATCH /api/app-settings``, ``PUT /api/app-settings/hf-token``, ``PUT /api/app-settings/cerber-token``, ``POST /api/cerber/check`` | :class:`Core` |
| ``PUT /api/app-settings/favorite-models {key, favorite}`` | :meth:`Core.set_favorite_model` ("Ulubione") |

The model key in ``/api/library/{key}`` contains slashes
(``org/repo/file-stem``); aiohttp matches it with a ``{key:.+}`` pattern and
gives it back already percent-decoded.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aiohttp import web

from localcode.chat.repo import ApiError
from localcode.hub.library import LibraryError
from localcode.hub.catalog import (
    Catalog,
    CatalogError,
    catalog_settings_dto,
    parse_repo_ref,
)
from localcode.server.app import get_core, json_body, json_response
from localcode.server.core import Core, maybe_await

__all__ = ["register"]


def register(app: web.Application) -> None:
    app.router.add_get("/api/hub/search", hub_search)
    app.router.add_get("/api/hub/curated", hub_curated)
    app.router.add_get("/api/hub/model", hub_model)

    app.router.add_post("/api/library/import", library_import)
    app.router.add_get("/api/catalog", catalog_list)
    app.router.add_get("/api/logos/{slug}", publisher_logo)
    app.router.add_post("/api/catalog/install", catalog_install)
    # Registered before the catch-all below, which would otherwise swallow it.
    app.router.add_get("/api/catalog/lookup", catalog_lookup)
    app.router.add_get("/api/catalog/{id:.+}", catalog_detail)

    app.router.add_get("/api/downloads", downloads_list)
    app.router.add_post("/api/downloads", downloads_start)
    app.router.add_delete("/api/downloads/{id}", downloads_cancel)

    app.router.add_get("/api/library", library_list)
    app.router.add_post("/api/library/rescan", library_rescan)
    app.router.add_patch("/api/library/{key:.+}", library_patch)
    app.router.add_delete("/api/library/{key:.+}", library_delete)

    app.router.add_get("/api/app-settings", app_settings_get)
    app.router.add_patch("/api/app-settings", app_settings_patch)
    app.router.add_put("/api/app-settings/hf-token", app_settings_hf_token)
    app.router.add_put("/api/app-settings/cerber-token", app_settings_cerber_token)
    app.router.add_put("/api/app-settings/favorite-models", app_settings_favorite_model)
    app.router.add_post("/api/cerber/check", cerber_check)


# --------------------------------------------------------------------------- #
# Hub
# --------------------------------------------------------------------------- #


async def hub_search(request: web.Request) -> web.Response:
    """``GET /api/hub/search?q&format&sort&limit`` → ``[HubModel]``."""
    core = get_core(request)
    query = request.query.get("q") or ""
    fmt = request.query.get("format") or "any"
    sort = request.query.get("sort") or "trending"
    limit = _int_param(request.query.get("limit"), "limit", default=30, lo=1, hi=100)
    return json_response(await core.hub.asearch(query, format=fmt, sort=sort, limit=limit))


async def hub_curated(request: web.Request) -> web.Response:
    """``GET /api/hub/curated`` → ``[HubModel]``."""
    core = get_core(request)
    return json_response(await core.hub.acurated())


async def hub_model(request: web.Request) -> web.Response:
    """``GET /api/hub/model?repo=org/name`` → ``HubModelDetail``."""
    core = get_core(request)
    repo = (request.query.get("repo") or "").strip()
    if not repo:
        raise ApiError(400, "Brak parametru „repo”.", "missing_repo")
    refresh = (request.query.get("refresh") or "").strip() in ("1", "true", "yes")
    return json_response(await core.hub.adetails(repo, refresh=refresh))


# --------------------------------------------------------------------------- #
# Catalogue (the in-app store; /api/hub/* stays as the advanced view)
# --------------------------------------------------------------------------- #


#: Default page of the store listing.  The index ships thousands of models; an
#: app that asks for "everything" wants the first screenful, not 6000 DTOs.
CATALOG_PAGE = 200


def _catalog(core: Core) -> Catalog:
    """The per-Core catalogue (kept so its search cache survives requests)."""
    catalog = getattr(core, "_catalog", None)
    if catalog is None:
        catalog = Catalog(core.settings, hub=core.hub, library=core.library,
                          downloads=core.downloads)
        core._catalog = catalog  # noqa: SLF001 - lazy attribute on our own Core
    catalog.downloads = core.downloads
    return catalog


def _catalog_error(exc: CatalogError) -> ApiError:
    return ApiError(exc.status, exc.message, exc.code)


async def library_import(request: web.Request) -> web.Response:
    """``POST /api/library/import {path, name?, move?}`` → the new ``LocalModel``.

    Adds a model the person already has: a ``.gguf`` file or an MLX directory.
    The files are copied into the models directory so that the library owns
    them — including the right to delete them again.
    """
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    path = str(body.get("path") or "").strip()
    if not path:
        raise ApiError(400, "Brak pola „path”.", "missing_path")
    name = str(body.get("name") or "").strip() or None
    move = bool(body.get("move"))
    try:
        model = await asyncio.to_thread(core.library.import_path, path, name=name, move=move)
    except LibraryError as exc:
        raise ApiError(400, exc.message, exc.code) from exc
    await core.rescan_library()
    return json_response(model.to_dict(), status=201)


async def publisher_logo(request: web.Request) -> web.Response:
    """``GET /api/logos/{slug}`` → the laboratory's mark as a PNG.

    The files are the organisations' own Hugging Face avatars, shipped with the
    package (``scripts/fetch_logos.py``).  They never change between releases,
    so the answer is cacheable for a long time.
    """
    from localcode.hub.publishers import LOGO_DIR, available_logos

    slug = str(request.match_info["slug"]).removesuffix(".png")
    if slug not in available_logos():
        raise ApiError(404, "Nie ma logo tego wydawcy.", "logo_not_found")
    path = LOGO_DIR / f"{slug}.png"
    return web.FileResponse(path, headers={
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=604800, immutable",
    })


async def catalog_list(request: web.Request) -> web.Response:
    """``GET /api/catalog?category=&q=&installed=`` → ``[CatalogEntry]``.

    Offline the staff picks plus the installed models are returned and the
    ``X-Catalog-Offline: 1`` header is set; the Polish explanation for it is
    ``catalog.offlineNote`` in ``GET /api/app-settings``.

    ``sort`` is one of :data:`~localcode.hub.catalog.SORTS`
    (``polecane|popularne|lubiane|zaufanie|oficjalne|najmniejsze``).
    ``limit`` (default :data:`CATALOG_PAGE`) caps the page, ``minTrust`` hides
    models below a trust score and ``official=1`` keeps only the laboratories'
    own repositories.  ``X-Catalog-Total`` carries the unpaged match count.
    """
    core = get_core(request)
    catalog = _catalog(core)
    category = (request.query.get("category") or "").strip() or None
    query = (request.query.get("q") or "").strip() or None
    installed = _bool_param(request.query.get("installed"), "installed")
    limit = _int_param(request.query.get("limit"), "limit", default=CATALOG_PAGE, lo=1, hi=2000)
    min_trust = _int_param(request.query.get("minTrust"), "minTrust", default=0, lo=0, hi=100)
    official_only = bool(_bool_param(request.query.get("official"), "official"))
    sort = (request.query.get("sort") or "polecane").strip() or "polecane"
    try:
        entries = await asyncio.to_thread(
            catalog.entries, category=category, query=query, installed=installed,
            limit=limit, min_trust=min_trust, official_only=official_only, sort=sort)
    except CatalogError as exc:
        raise _catalog_error(exc) from exc
    response = json_response(entries)
    if catalog.offline:
        response.headers["X-Catalog-Offline"] = "1"
    # How many models matched before `limit` — the app shows "z 6000".
    response.headers["X-Catalog-Total"] = str(catalog.total_matches)
    return response


async def catalog_lookup(request: web.Request) -> web.Response:
    """``GET /api/catalog/lookup?ref=`` → one CatalogEntry for a pasted repository.

    ``ref`` is a repository id or a Hugging Face URL.  A model that is already
    in the catalogue is returned as the catalogue knows it (with our
    description); anything else comes back marked ``external`` and installs the
    same way, just without the parts we add.
    """
    core = get_core(request)
    ref = (request.query.get("ref") or "").strip()
    if not ref:
        raise ApiError(400, "Brak parametru „ref”.", "missing_ref")
    repo = parse_repo_ref(ref)
    if not repo:
        raise ApiError(400, "To nie wygląda na model z Hugging Face "
                            "(oczekiwano „autor/nazwa” albo adresu huggingface.co).",
                       "invalid_ref")
    catalog = _catalog(core)
    known = await asyncio.to_thread(catalog.resolve_name, repo)
    if known is not None and not known.get("external"):
        return json_response(known)
    entry = await asyncio.to_thread(catalog.remote_entry, repo)
    if entry is None:
        raise ApiError(404, f"Nie znaleziono modelu „{repo}” na Hugging Face.", "not_found")
    return json_response(entry)


async def catalog_detail(request: web.Request) -> web.Response:
    """``GET /api/catalog/{id}`` → one ``CatalogEntry`` with its variant list."""
    core = get_core(request)
    entry_id = request.match_info["id"]
    try:
        return json_response(await asyncio.to_thread(_catalog(core).detail, entry_id))
    except CatalogError as exc:
        raise _catalog_error(exc) from exc


async def catalog_install(request: web.Request) -> web.Response:
    """``POST /api/catalog/install {id, variantId?}`` → ``DownloadJob``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    entry_id = str(body.get("id") or "").strip()
    if not entry_id:
        raise ApiError(400, "Brak pola „id”.", "missing_id")
    raw_variant = body.get("variantId")
    variant = str(raw_variant).strip() if isinstance(raw_variant, str) and raw_variant.strip() else None
    core.require_downloads()
    try:
        job = await _catalog(core).ainstall(entry_id, variant=variant)
    except CatalogError as exc:
        raise _catalog_error(exc) from exc
    return json_response(job, status=201)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #


async def downloads_list(request: web.Request) -> web.Response:
    """``GET /api/downloads`` → ``[DownloadJob]`` (empty when the module is missing)."""
    core = get_core(request)
    manager = core.downloads
    if manager is None:
        return json_response([])
    jobs = await maybe_await(manager.list())
    return json_response(list(jobs or []))


async def downloads_start(request: web.Request) -> web.Response:
    """``POST /api/downloads {repo, variantId?, files?}`` → ``DownloadJob``."""
    core = get_core(request)
    manager = core.require_downloads()
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    repo = str(body.get("repo") or "").strip()
    if not repo:
        raise ApiError(400, "Brak pola „repo”.", "missing_repo")
    files = await _files_for(core, repo, body)
    job = await maybe_await(manager.start(repo, files))
    return json_response(job, status=201)


async def _files_for(core: Core, repo: str, body: dict[str, Any]) -> list[str]:
    """Explicit ``files``, else the named ``variantId``, else the recommended variant."""
    raw = body.get("files")
    if isinstance(raw, list) and raw:
        if any(not isinstance(f, str) or not f.strip() for f in raw):
            raise ApiError(400, "Pole „files” musi być listą nazw plików.", "invalid_files")
        return [f.strip() for f in raw]
    variant_id = body.get("variantId")
    if isinstance(variant_id, str) and variant_id.strip():
        variant = await asyncio.to_thread(core.hub.variant, repo, variant_id.strip())
        return list(variant["files"])
    detail = await core.hub.adetails(repo)
    variants = detail.get("variants") or []
    if not variants:
        raise ApiError(422, f"Repozytorium „{repo}” nie ma plików do pobrania.", "no_variants")
    chosen = next((v for v in variants if v.get("recommended")), variants[0])
    return list(chosen["files"])


async def downloads_cancel(request: web.Request) -> web.Response:
    """``DELETE /api/downloads/{id}`` — cancel a queued or running job."""
    core = get_core(request)
    manager = core.require_downloads()
    job_id = request.match_info["id"]
    result = await maybe_await(manager.cancel(job_id))
    if result is False:
        raise ApiError(404, "Nie znaleziono zadania pobierania.", "not_found")
    return json_response(result if isinstance(result, dict) else {"ok": True})


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #


async def library_list(request: web.Request) -> web.Response:
    """``GET /api/library`` → models on disk (+ borrowed ones) with ``loaded``.

    Remote models are appended rather than kept in a list of their own: every
    picker in the app reads this one, and a borrowed model is a model you can
    talk to — it simply carries ``remote: true`` so the UI can say where it runs.
    """
    core = get_core(request)
    rows = await asyncio.to_thread(core.library_dtos)
    return json_response([*rows, *await core.remote_library_dtos()])


async def library_rescan(request: web.Request) -> web.Response:
    """``POST /api/library/rescan`` — rescan the models directory."""
    core = get_core(request)
    await core.rescan_library()
    return json_response(await asyncio.to_thread(core.library_dtos))


async def library_patch(request: web.Request) -> web.Response:
    """``PATCH /api/library/{key} {loadDefaults}`` → the model DTO."""
    core = get_core(request)
    key = request.match_info["key"]
    model = core.library.get(key)
    if model is None:
        raise ApiError(404, f"Nie znaleziono modelu „{key}”.", "not_found")
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    if "loadDefaults" not in body:
        raise ApiError(400, "Brak pola „loadDefaults”.", "missing_field")
    defaults = body["loadDefaults"]
    if defaults is not None and not isinstance(defaults, dict):
        raise ApiError(400, "Pole „loadDefaults” musi być obiektem albo null.", "invalid_field")
    stored = await asyncio.to_thread(core.library.set_load_defaults, key, defaults)
    dto = model.to_dict()
    dto["loaded"] = core.runtime.is_loaded(key)
    dto["loadDefaults"] = stored
    return json_response(dto)


async def library_delete(request: web.Request) -> web.Response:
    """``DELETE /api/library/{key}`` → 204 (LocalCode-managed models only)."""
    core = get_core(request)
    key = request.match_info["key"]
    await core.library.adelete(key)
    core.broadcast_library()
    return web.Response(status=204)


# --------------------------------------------------------------------------- #
# Application settings
# --------------------------------------------------------------------------- #


def _app_settings_dto(core: Core) -> dict[str, Any]:
    """The application settings plus the catalogue provenance strings.

    Every route that answers with the settings goes through here: the Mac app
    replaces its whole ``settings`` object with the body of a PATCH or of a
    token change, so a response without ``catalog`` would empty the store's
    category bar until the next ``GET``.
    """
    settings = dict(core.app_settings())
    settings["catalog"] = catalog_settings_dto()
    return settings


async def app_settings_get(request: web.Request) -> web.Response:
    """``GET /api/app-settings`` (plus the catalogue provenance strings)."""
    return json_response(_app_settings_dto(get_core(request)))


async def app_settings_patch(request: web.Request) -> web.Response:
    """``PATCH /api/app-settings {partial}``."""
    core = get_core(request)
    await core.update_app_settings(await json_body(request) or {})
    return json_response(_app_settings_dto(core))


async def app_settings_hf_token(request: web.Request) -> web.Response:
    """``PUT /api/app-settings/hf-token {token|null}``."""
    core = get_core(request)
    body = await json_body(request)
    if not isinstance(body, dict) or "token" not in body:
        raise ApiError(400, "Oczekiwano obiektu {\"token\": \"…\"} albo {\"token\": null}.", "missing_token")
    await core.set_hf_token(body["token"])
    return json_response(_app_settings_dto(core))


async def app_settings_cerber_token(request: web.Request) -> web.Response:
    """``PUT /api/app-settings/cerber-token {token|null}`` — verified with Cerber when it is open."""
    core = get_core(request)
    body = await json_body(request)
    if not isinstance(body, dict) or "token" not in body:
        raise ApiError(400, "Oczekiwano obiektu {\"token\": \"…\"} albo {\"token\": null}.", "missing_token")
    await core.set_cerber_token(body["token"])
    return json_response(_app_settings_dto(core))


async def app_settings_favorite_model(request: web.Request) -> web.Response:
    """``PUT /api/app-settings/favorite-models {key, favorite}`` — star or unstar one model."""
    core = get_core(request)
    body = await json_body(request)
    if not isinstance(body, dict) or "key" not in body or "favorite" not in body:
        raise ApiError(400, "Oczekiwano obiektu {\"key\": \"…\", \"favorite\": true|false}.", "missing_key")
    await core.set_favorite_model(body["key"], body["favorite"])
    return json_response(_app_settings_dto(core))


async def cerber_check(request: web.Request) -> web.Response:
    """``POST /api/cerber/check`` → ``{ok, message, installed}`` (never raises for a closed Cerber)."""
    return json_response(await get_core(request).check_cerber())


def _bool_param(raw: str | None, name: str) -> bool | None:
    """``"1"/"true"/"yes"`` → ``True``, ``"0"/"false"/"no"`` → ``False``, missing → ``None``."""
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "tak"):
        return True
    if value in ("0", "false", "no", "nie"):
        return False
    raise ApiError(400, f"Parametr „{name}” musi być wartością logiczną.", "invalid_param")


def _int_param(raw: str | None, name: str, *, default: int, lo: int, hi: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ApiError(400, f"Parametr „{name}” musi być liczbą.", "invalid_param") from None
    return max(lo, min(value, hi))
