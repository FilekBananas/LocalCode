"""Local image generation over HTTP — models, generation (SSE) and the gallery.

| route | handler |
|---|---|
| ``GET /api/images/models`` | image models on disk + per-family defaults |
| ``POST /api/images/generate`` | SSE: ``status`` / ``progress`` / ``image`` / ``done`` / ``error`` |
| ``GET /api/images`` | :meth:`ImageGallery.list` (paged, newest first) |
| ``GET /api/images/stats`` | :meth:`ImageGallery.stats` |
| ``GET /api/images/{id}/file`` | the PNG bytes |
| ``PATCH /api/images/{id}`` | ``{favorite}`` → ``ImageDTO`` |
| ``DELETE /api/images/{id}`` | remove the row **and** the file |

Everything lives under ``/api/``, so the bearer token of
:mod:`localcode.server.app` is required exactly like on every other route.

Generation is streamed: the client sees ``status`` while the model loads,
``progress`` for every denoising step, one ``image`` frame per finished picture
(already stored in the gallery, so its ``url`` works immediately) and finally
``done``. Closing the connection cancels the job — :func:`generate_frames` runs
the runtime in a task and cancels it when the generator is closed, which is what
``stream_sse`` does on a client disconnect.

The image runtime itself (``localcode.images.runtime``) is optional: it needs
``pip install "localcode[images]"``. Without it every generating endpoint
answers 503 with a Polish explanation while the gallery keeps working.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from aiohttp import web

from localcode.chat.repo import ApiError
from localcode.images.gallery import ImageGallery, ImageGalleryError, ImageRecord, parse_size
from localcode.images.runtime import ImageRuntimeError
from localcode.server.app import (
    error_payload,
    get_core,
    json_body,
    json_response,
    stream_sse,
)
from localcode.server.core import Core

log = logging.getLogger(__name__)

__all__ = [
    "MAX_COUNT",
    "build_params",
    "family_defaults",
    "generate_frames",
    "generate_records",
    "image_model_dtos",
    "params_from_body",
    "register",
    "resolve_image_model",
]

#: Most images one request may ask for (they are generated one after another).
MAX_COUNT = 8
#: Largest gallery page.
MAX_LIMIT = 200

#: Per-family generation defaults — ``docs/IMAGES.md`` "defaults:{steps,guidance,sizes}".
#: They live in :mod:`localcode.images.base` (one table for the whole feature);
#: the copy below is only used if that module cannot be imported.
FALLBACK_DEFAULTS: dict[str, Any] = {"steps": 20, "guidance": 5.0,
                                     "sizes": ["1024x1024", "512x512"]}


def register(app: web.Application) -> None:
    # Static paths first: aiohttp resolves routes in registration order.
    app.router.add_get("/api/images/models", images_models)
    app.router.add_get("/api/images/stats", images_stats)
    app.router.add_post("/api/images/generate", images_generate)
    app.router.add_get("/api/images", images_list)
    app.router.add_get("/api/images/{id}/file", images_file)
    app.router.add_patch("/api/images/{id}", images_patch)
    app.router.add_delete("/api/images/{id}", images_delete)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def image_family(model: Any) -> str:
    """``ModelMeta.image_family`` with a conservative fallback."""
    meta = getattr(model, "meta", None)
    family = getattr(meta, "image_family", None)
    if isinstance(family, str) and family.strip():
        return family.strip()
    return "unknown"


def family_defaults(model: Any) -> dict[str, Any]:
    """Steps / guidance / size presets for one model (``images.base`` decides)."""
    family = image_family(model)
    key = str(getattr(model, "key", "") or "")
    try:
        from localcode.images import base as image_base  # noqa: PLC0415 - optional package

        defaults = image_base.family_defaults(family)
        defaults["steps"] = image_base.default_steps_for(family, key)
    except ImportError:  # pragma: no cover - only without localcode.images
        defaults = dict(FALLBACK_DEFAULTS)
    defaults["sizes"] = list(defaults.get("sizes") or [])
    return defaults


def is_image_model(model: Any) -> bool:
    """``LocalModel.is_image`` — the ``image`` capability or a known family."""
    flag = getattr(model, "is_image", None)
    if isinstance(flag, bool):
        return flag
    return "image" in (getattr(model, "capabilities", None) or [])


def image_models(core: Core) -> list[Any]:
    return [m for m in core.library.list() if is_image_model(m)]


def image_model_dtos(core: Core) -> list[dict[str, Any]]:
    """``LocalModel`` DTO + ``imageFamily`` + ``defaults`` for every image model.

    The image runtime describes its own models (it also knows what each one
    would cost in memory), so it is asked first; without the optional extra the
    list is built from the library alone and browsing still works.
    """
    runtime = core.images
    if runtime is not None and hasattr(runtime, "model_info") and hasattr(runtime, "models"):
        try:
            return [runtime.model_info(model) for model in runtime.models()]
        except Exception:  # noqa: BLE001 - fall back to the library description
            log.warning("środowisko obrazów nie opisało modeli", exc_info=True)
    out: list[dict[str, Any]] = []
    for model in image_models(core):
        dto = model.to_dict()
        dto["imageFamily"] = image_family(model)
        dto["defaults"] = family_defaults(model)
        try:
            dto["loaded"] = bool(core.runtime.is_loaded(model.key))
        except Exception:  # noqa: BLE001 - the list must not depend on the runtime
            dto["loaded"] = False
        out.append(dto)
    return out


def resolve_image_model(core: Core, query: str | None) -> Any:
    """An image model by key, by substring, or the default one.

    Without a query it is ``settings.images.default_model``, else a model that
    is already in memory (no load time at all), else the first image model in
    the library. Raises a Polish :class:`ApiError` when nothing matches, which
    the caller turns into the right shape (``/api`` JSON or an OpenAI error).
    """
    models = image_models(core)
    if not models:
        raise ApiError(404, "Nie masz żadnego modelu do generowania obrazów. "
                            "Pobierz jakiś w zakładce „Odkrywaj” albo poleceniem "
                            "`localcode models search flux`.", "no_image_models")
    wanted = (query or "").strip()
    if not wanted:
        from localcode.images.gallery import settings_view

        wanted = (settings_view(core.settings).default_model or "").strip()
    if not wanted:
        return _default_model(core, models)
    exact = next((m for m in models if m.key == wanted), None)
    if exact is not None:
        return exact
    lowered = wanted.lower()
    hits = [m for m in models
            if lowered in m.key.lower() or lowered in str(m.display_name or "").lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ApiError(404, f"Nie znaleziono modelu obrazów „{wanted}”.", "model_not_found")
    raise ApiError(400, f"Nazwa „{wanted}” pasuje do kilku modeli: "
                        f"{', '.join(m.key for m in hits[:5])}.", "ambiguous_model")


def _default_model(core: Core, models: list[Any]) -> Any:
    """Something already loaded, otherwise the first image model on disk."""
    try:
        loaded = [m for m in models if core.runtime.is_loaded(m.key)]
    except Exception:  # noqa: BLE001 - the choice must not depend on the runtime
        loaded = []
    return (loaded or models)[0]


async def images_models(request: web.Request) -> web.Response:
    """``GET /api/images/models`` → ``[LocalModel + imageFamily + defaults]``."""
    core = get_core(request)
    return json_response(await asyncio.to_thread(image_model_dtos, core))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@dataclass
class FallbackParams:
    """``ImageParams`` of ``docs/IMAGES.md`` for installations without the extra."""

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    count: int = 1
    init_image: str | None = None
    strength: float = 0.6


def _params_class() -> Any:
    """``localcode.images.base.ImageParams`` when it exists, else :class:`FallbackParams`."""
    for module, name in (("localcode.images.base", "ImageParams"),
                         ("localcode.images.runtime", "ImageParams")):
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except (ImportError, AttributeError):
            continue
    return FallbackParams


def build_params(**kwargs: Any) -> Any:
    """Build ``ImageParams``, dropping fields this installation does not know."""
    cls = _params_class()
    try:
        allowed = set(inspect.signature(cls).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        allowed = set(kwargs)
    clean = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    clean.setdefault("prompt", str(kwargs.get("prompt") or ""))
    try:
        return cls(**clean)
    except TypeError:  # pragma: no cover - a differently shaped ImageParams
        return FallbackParams(prompt=str(kwargs.get("prompt") or ""))


def _is_result(item: Any) -> bool:
    """Tell an ``ImageResult`` from an ``ImageProgress``."""
    if isinstance(item, dict):
        return bool(item.get("path")) and item.get("type") in (None, "image", "result")
    return getattr(item, "path", None) is not None and getattr(item, "step", None) is None


def _progress_frame(item: Any, index: int) -> dict[str, Any] | None:
    """Translate whatever the runtime reported into an SSE frame."""
    if isinstance(item, dict):
        kind = str(item.get("type") or "")
        if kind == "status":
            return {"type": "status", "message": str(item.get("message") or "")}
        if kind in ("progress", ""):
            if item.get("step") is None:
                return None
            return {"type": "progress", "index": int(item.get("index", index) or index),
                    "step": int(item.get("step") or 0), "steps": int(item.get("steps") or 0)}
        return None
    message = getattr(item, "message", None)
    steps = int(getattr(item, "steps", 0) or 0)
    if steps <= 0:                       # indeterminate: a status line, not a bar
        return {"type": "status", "message": str(message)} if message else None
    return {"type": "progress", "index": int(getattr(item, "index", index) or index),
            "step": int(getattr(item, "step", 0) or 0), "steps": steps}


async def _call_generate(runtime: Any, model_key: str, params: Any, on_progress: Any) -> Any:
    """``ImageRuntime.generate`` with ``on_progress`` when it is supported."""
    try:
        return await runtime.generate(model_key, params, on_progress=on_progress)
    except TypeError as exc:  # pragma: no cover - a runtime without the hook
        if "on_progress" not in str(exc):
            raise
        return await runtime.generate(model_key, params)


async def generate_frames(
    core: Core,
    model_key: str,
    params: Any,
    *,
    model_name: str = "",
    negative_prompt: str = "",
    collect: list[ImageRecord] | None = None,
    reraise: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Run one generation job, storing every picture, yielding SSE frames.

    Closing the iterator (a client disconnect) cancels the job; a failure ends
    the stream with an ``error`` frame unless ``reraise`` asks for the exception.
    """
    runtime = core.require_images()
    gallery = core.gallery
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def on_progress(*args: Any, **kwargs: Any) -> None:
        item = args[0] if args else next(iter(kwargs.values()), None)
        if item is not None:
            queue.put_nowait(item)

    label = model_name or model_key
    yield {"type": "status", "message": f"Przygotowuję model {label}…"}
    task: asyncio.Task[Any] = asyncio.create_task(
        _call_generate(runtime, model_key, params, on_progress), name="localcode-image")
    seen: set[str] = set()
    index = 0
    try:
        while True:
            if not queue.empty():
                item = queue.get_nowait()
            elif task.done():
                break
            else:
                getter: asyncio.Future[Any] = asyncio.ensure_future(queue.get())
                try:
                    await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    # Also on cancellation: when the client goes away this
                    # generator is cancelled right here, and a getter left
                    # pending surfaced as "Task was destroyed but it is pending"
                    # / "cannot reuse already awaited coroutine" in server.log.
                    if not getter.done():
                        getter.cancel()
                        with contextlib.suppress(BaseException):
                            await getter
                if getter.cancelled():
                    continue
                item = getter.result()
            if _is_result(item):
                record = await _store(gallery, item, model_name, negative_prompt, params)
                seen.add(record.id)
                if collect is not None:
                    collect.append(record)
                index += 1
                yield {"type": "image", "image": record.to_dict()}
                continue
            frame = _progress_frame(item, index)
            if frame is not None:
                yield frame
        results = await task
        for result in _as_list(results):
            record_id = str(getattr(result, "id", "") or (result.get("id") if isinstance(result, dict) else ""))
            if record_id and record_id in seen:
                continue
            record = await _store(gallery, result, model_name, negative_prompt, params)
            if collect is not None:
                collect.append(record)
            yield {"type": "image", "image": record.to_dict()}
        yield {"type": "done"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the client as a frame
        if reraise:
            raise
        # An ImageRuntimeError already carries a full Polish explanation (missing
        # backend, model too big…); a traceback on stderr would only be noise in
        # the terminal, where this logger writes straight to the user.
        log.warning("image generation failed: %s", exc,
                    exc_info=not isinstance(exc, ImageRuntimeError))
        _status, payload = error_payload(exc)
        yield {"type": "error", "message": str(payload.get("error") or exc),
               "code": payload.get("code")}
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


async def generate_records(core: Core, model_key: str, params: Any, *, model_name: str = "",
                           negative_prompt: str = "") -> list[ImageRecord]:
    """Generate and store images, returning the gallery rows (no streaming)."""
    records: list[ImageRecord] = []
    frames = generate_frames(core, model_key, params, model_name=model_name,
                             negative_prompt=negative_prompt, collect=records, reraise=True)
    async for _frame in frames:
        pass
    return records


async def _store(gallery: ImageGallery, result: Any, model_name: str, negative_prompt: str,
                 params: Any) -> ImageRecord:
    extra = {"count": int(getattr(params, "count", 1) or 1)}
    init_image = getattr(params, "init_image", None)
    if init_image:
        extra["initImage"] = str(init_image)
        extra["strength"] = float(getattr(params, "strength", 0.6) or 0.6)
    return await gallery.aadd(result, model_name=model_name, negative_prompt=negative_prompt,
                              params=extra)


def _as_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


async def images_generate(request: web.Request) -> web.StreamResponse:
    """``POST /api/images/generate`` → SSE (``status``/``progress``/``image``/``done``)."""
    core = get_core(request)
    core.require_images()          # a 503 must land before the stream is opened
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    model = resolve_image_model(core, body.get("modelKey"))
    params, negative = params_from_body(core, body, model)
    frames = generate_frames(core, model.key, params, model_name=model.display_name,
                             negative_prompt=negative)
    return await stream_sse(request, frames)


def params_from_body(core: Core, body: dict[str, Any], model: Any) -> tuple[Any, str]:
    from localcode.images.gallery import settings_view

    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ApiError(400, "Podaj opis obrazu (pole „prompt”).", "missing_prompt")
    view = settings_view(core.settings)
    defaults = family_defaults(model)
    try:
        default_w, default_h = view.size
    except ImageGalleryError:  # pragma: no cover - a hand-edited settings file
        default_w = default_h = 1024
    width = _int_field(body, "width", default_w)
    height = _int_field(body, "height", default_h)
    _validate_edges(width, height)
    steps = _int_field(body, "steps", view.default_steps or defaults.get("steps"))
    guidance = _float_field(body, "guidance", view.default_guidance if view.default_guidance
                            is not None else defaults.get("guidance"))
    seed = _int_field(body, "seed", None)
    count = _int_field(body, "count", 1) or 1
    if count < 1 or count > MAX_COUNT:
        raise ApiError(400, f"Liczba obrazów musi mieścić się w zakresie 1–{MAX_COUNT}.",
                       "invalid_count")
    negative = str(body.get("negativePrompt") or "")
    translate = body.get("translatePrompt")
    if translate is not None and not isinstance(translate, bool):
        raise ApiError(400, "Pole „translatePrompt” musi być prawdą albo fałszem.", "invalid_translate")
    params = build_params(prompt=prompt, negative_prompt=negative, width=width, height=height,
                          steps=steps, guidance=guidance, seed=seed, count=count,
                          translate=translate)
    return params, negative


def _validate_edges(width: int, height: int) -> None:
    try:
        parse_size(f"{width}x{height}")
    except ImageGalleryError as exc:
        raise ApiError(400, str(exc), "invalid_size") from exc


def _int_field(body: dict[str, Any], name: str, default: Any) -> Any:
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ApiError(400, f"Pole „{name}” musi być liczbą całkowitą.", "invalid_field")
    return int(value)


def _float_field(body: dict[str, Any], name: str, default: Any) -> Any:
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, f"Pole „{name}” musi być liczbą.", "invalid_field")
    return float(value)


# --------------------------------------------------------------------------- #
# Gallery
# --------------------------------------------------------------------------- #


async def images_list(request: web.Request) -> web.Response:
    """``GET /api/images?limit&offset&favorite`` → ``[ImageDTO]`` (newest first)."""
    core = get_core(request)
    limit = _int_param(request.query.get("limit"), "limit", default=50, lo=1, hi=MAX_LIMIT)
    offset = _int_param(request.query.get("offset"), "offset", default=0, lo=0, hi=10**9)
    favorite = _bool_param(request.query.get("favorite"))
    records = await core.gallery.alist(limit, offset, favorite)
    return json_response([record.to_dict() for record in records])


async def images_stats(request: web.Request) -> web.Response:
    """``GET /api/images/stats`` → how many images, how much disk, which models."""
    return json_response(await get_core(request).gallery.astats())


async def images_file(request: web.Request) -> web.StreamResponse:
    """``GET /api/images/{id}/file`` → the PNG bytes."""
    core = get_core(request)
    image_id = request.match_info["id"]
    record = await core.gallery.aget(image_id)
    if record is None:
        raise ApiError(404, f"Nie znaleziono obrazu „{image_id}”.", "not_found")
    path = Path(record.path)
    if not path.is_file():
        raise ApiError(404, "Plik obrazu został usunięty z dysku.", "file_missing")
    return web.FileResponse(path, headers={
        "Content-Type": "image/png",
        "Cache-Control": "private, max-age=86400",
        "Content-Disposition": f'inline; filename="{record.id}.png"',
    })


async def images_patch(request: web.Request) -> web.Response:
    """``PATCH /api/images/{id} {favorite}`` → ``ImageDTO``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    if "favorite" not in body:
        raise ApiError(400, "Brak pola „favorite”.", "missing_field")
    if not isinstance(body["favorite"], bool):
        raise ApiError(400, "Pole „favorite” musi być wartością logiczną.", "invalid_field")
    record = await core.gallery.aset_favorite(request.match_info["id"], bool(body["favorite"]))
    return json_response(record.to_dict())


async def images_delete(request: web.Request) -> web.Response:
    """``DELETE /api/images/{id}`` → 204; the file goes with the row."""
    core = get_core(request)
    image_id = request.match_info["id"]
    if not await core.gallery.adelete(image_id):
        raise ApiError(404, f"Nie znaleziono obrazu „{image_id}”.", "not_found")
    return web.Response(status=204)


def _int_param(raw: str | None, name: str, *, default: int, lo: int, hi: int) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ApiError(400, f"Parametr „{name}” musi być liczbą.", "invalid_param") from None
    return max(lo, min(value, hi))


def _bool_param(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "tak")


def dtos(records: Iterable[ImageRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]
