"""The ai_ux-compatible chat API, mapped 1:1 onto :class:`localcode.chat.api.ChatApi`.

``chat/api.py`` describes every route declaratively in :data:`~localcode.chat.api.ROUTES`
(method, path, handler name, path/query parameters, body kind, success status),
so this module is a mechanical bridge: it pulls the arguments out of the aiohttp
request exactly the way the reference Next.js routes do (query parameters raw,
``None`` when absent) and turns the handler's return value into JSON, a file
download or an SSE stream.

``POST /api/chat`` is special: the first frame is pulled *before* the response
is committed, so a validation failure is still an ordinary JSON error; after
that the frames stream until the client disconnects (which is how the "stop"
button works — the generator is closed and the partial answer is stored as
``stopped``).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from aiohttp import web

from localcode.chat.api import ROUTES, FileResult, Route
from localcode.server.app import (
    file_response,
    get_core,
    json_body,
    json_response,
    read_multipart,
    stream_sse,
)

__all__ = ["register"]


def register(app: web.Application) -> None:
    """Register every route of :data:`localcode.chat.api.ROUTES`."""
    for route in ROUTES:
        app.router.add_route(route.method, route.path, _handler(route),
                             name=f"chat_{route.handler}_{route.method.lower()}")


def _handler(route: Route) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
    async def handler(request: web.Request) -> web.StreamResponse:
        core = get_core(request)
        api = core.chat_api
        kwargs = await _arguments(request, route)
        fn = getattr(api, route.handler)
        if route.kind == "sse":
            return await _sse(request, fn(**kwargs))
        result = await fn(**kwargs)
        if route.kind == "file":
            assert isinstance(result, FileResult)
            return file_response(result)
        return json_response(result, status=route.status)

    handler.__name__ = f"chat_{route.handler}"
    handler.__doc__ = f"{route.method} {route.path} → ChatApi.{route.handler}"
    return handler


async def _arguments(request: web.Request, route: Route) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name, kwarg in route.path_params:
        kwargs[kwarg] = request.match_info[name]
    for name, kwarg in route.query:
        kwargs[kwarg] = request.query.get(name)          # raw: absent ≠ empty
    if route.body == "json":
        kwargs["body"] = await json_body(request)
    elif route.body == "multipart":
        files, fields = await read_multipart(request)
        kwargs["files"] = files
        for name, kwarg in route.form:
            kwargs[kwarg] = fields.get(name)
    return kwargs


async def _empty() -> AsyncIterator[Any]:
    return
    yield  # pragma: no cover - makes this an async generator


async def _sse(request: web.Request, source: Any) -> web.StreamResponse:
    """Commit the SSE response only after the first frame validated the request.

    Until then an :class:`~localcode.chat.repo.ApiError` is an ordinary JSON
    error (the error middleware turns it into one); afterwards failures travel
    as ``{"type": "error"}`` frames inside the stream.
    """
    iterator = source.__aiter__()
    try:
        first = await iterator.__anext__()
    except StopAsyncIteration:
        return await stream_sse(request, _empty())
    return await stream_sse(request, iterator, initial=[first])
