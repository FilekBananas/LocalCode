"""Video generation over HTTP: models, a cost estimate and generation (SSE).

| route | what it does |
| --- | --- |
| ``GET  /api/video/models`` | video-looking models on disk, supported or not |
| ``POST /api/video/estimate`` | the rough, honestly-unverified time estimate for a shape |
| ``POST /api/video/generate`` | SSE: ``status`` / ``progress`` / ``video`` / ``done`` / ``error`` |

Generation is streamed exactly like ``/api/images/generate``: the client sees
a ``status`` frame with the time estimate before anything starts, a
``progress`` frame per denoising step, then a ``video`` frame (the finished
clip's path) and ``done``. Closing the connection cancels the job — the
generation runs in a task that is cancelled when the SSE generator is closed,
which tears down the worker subprocess through
:class:`~localcode.video.runtime.VideoRuntime`'s own ``finally`` cleanup.

The video runtime (:mod:`localcode.video`) is always importable — unlike
``localcode.images``, it does not import ``torch``/``diffusers`` at module
level, only inside the worker subprocess — so there is no "module missing"
503 here. A missing ``localcode[video]`` extra instead surfaces as a normal
``error`` frame (HTTP 501 shape, Polish message naming the extra) the first
time a generation actually needs the worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from aiohttp import web

from localcode.server.app import ApiError, error_payload, get_core, json_body, json_response, stream_sse
from localcode.video.base import VideoError

log = logging.getLogger(__name__)

__all__ = ["register"]


def register(app: web.Application) -> None:
    app.router.add_get("/api/video/models", video_models)
    app.router.add_post("/api/video/estimate", video_estimate)
    app.router.add_post("/api/video/generate", video_generate)


def _runtime(core: Any) -> Any:
    """The per-Core video runtime (cheap, so it is built on first use)."""
    runtime = getattr(core, "_video", None)
    if runtime is None:
        from localcode.video import VideoRuntime

        runtime = VideoRuntime(core.settings, core.library)
        core._video = runtime  # noqa: SLF001 - lazy attribute on our own Core
    return runtime


def _error(exc: VideoError) -> ApiError:
    return ApiError(exc.status, exc.message, exc.code)


async def video_models(request: web.Request) -> web.Response:
    """``GET /api/video/models`` → ``[LocalModel + videoKind + supported]``."""
    core = get_core(request)
    return json_response(await asyncio.to_thread(_runtime(core).list_models))


async def video_estimate(request: web.Request) -> web.Response:
    """``POST /api/video/estimate {width?,height?,numFrames?,fps?,steps?}``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    kwargs = _shape_fields(body)
    try:
        estimate = _runtime(core).estimate(**kwargs)
    except VideoError as exc:
        raise _error(exc) from exc
    return json_response(estimate)


async def video_generate(request: web.Request) -> web.StreamResponse:
    """``POST /api/video/generate`` → SSE (``status``/``progress``/``video``/``done``/``error``)."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        raise ApiError(400, "Podaj opis materiału wideo (pole „prompt”).", "missing_prompt")
    kwargs = _shape_fields(body)
    kwargs["negative_prompt"] = str(body.get("negativePrompt") or "") or None
    kwargs["seed"] = _int_field(body, "seed", None)
    kwargs["guidance"] = _float_field(body, "guidance", None)
    force = body.get("force", False)
    if not isinstance(force, bool):
        raise ApiError(400, "Pole „force” musi być wartością logiczną.", "invalid_field")
    kwargs["force"] = force
    kwargs["output"] = _safe_output(_runtime(core), body.get("output"))
    kwargs["model_key"] = str(body.get("modelKey") or "").strip() or None
    frames = video_frames(core, prompt, kwargs)
    return await stream_sse(request, frames)


def _safe_output(runtime: Any, raw: Any) -> str | None:
    """A client-chosen output name, kept inside the video directory.

    The terminal may write a clip wherever the user points it — it runs as the
    user. A request over HTTP may not: the server would happily overwrite
    ``~/.zshrc`` for anyone who can reach the port. So the route accepts a name
    (or a relative path) under ``~/.localcode/data/video`` and refuses anything
    that leaves it.
    """
    if raw is None:
        return None
    name = str(raw).strip()
    if not name:
        return None
    root = Path(runtime.output_dir()).expanduser()
    candidate = (root / name).expanduser()
    if Path(name).expanduser().is_absolute() or ".." in Path(name).parts:
        raise ApiError(400, "Pole „output” musi być nazwą pliku w katalogu wideo, bez ścieżki.",
                       "invalid_output")
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        raise ApiError(400, "Pole „output” wskazuje poza katalog wideo.", "invalid_output") from None
    return str(resolved)


def _shape_fields(body: dict[str, Any]) -> dict[str, Any]:
    """``width``/``height``/``numFrames``/``fps``/``steps`` from a JSON body."""
    return {
        "width": _int_field(body, "width", None),
        "height": _int_field(body, "height", None),
        "num_frames": _int_field(body, "numFrames", None),
        "fps": _int_field(body, "fps", None),
        "steps": _int_field(body, "steps", None),
    }


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
# Generation stream
# --------------------------------------------------------------------------- #


async def video_frames(core: Any, prompt: str, kwargs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Run one generation job, yielding SSE frames; a disconnect cancels it."""
    runtime = _runtime(core)
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def on_progress(progress: Any) -> None:
        queue.put_nowait(progress)

    task: asyncio.Task[Any] = asyncio.create_task(
        runtime.generate(prompt, on_progress=on_progress, **kwargs), name="localcode-video")
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
            yield {"type": "progress", **item.to_dict()}
        result = await task
        yield {"type": "video", "video": result.to_dict()}
        yield {"type": "done"}
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the client as a frame
        log.warning("video generation failed: %s", exc, exc_info=not isinstance(exc, VideoError))
        _status, payload = error_payload(exc)
        yield {"type": "error", "message": str(payload.get("error") or exc), "code": payload.get("code")}
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
