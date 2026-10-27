"""aiohttp application: middlewares, shared helpers and route registration.

``create_app(core, token)`` builds the whole HTTP surface described in
``docs/ARCHITECTURE.md``:

* ``/api/*`` — the ai_ux-compatible chat API (``routes_chat``), the LocalCode
  hub/library/settings API (``routes_models``), runtime + events
  (``routes_runtime``), image generation and the gallery (``routes_images``)
  and the coding agent (``agent_ws``);
* ``/v1/*`` — the LM Studio-style OpenAI-compatible server (``openai_api``).

Cross-cutting rules implemented here:

* **Auth** — every ``/api/*`` route except ``/api/health`` needs
  ``Authorization: Bearer <token>``; ``/api/events`` and the agent WebSocket
  also accept ``?token=`` (an ``EventSource``/``WebSocket`` cannot set headers).
  Tokens are compared with ``hmac.compare_digest``.
* **Host** — a request addressed to a DNS name (anything but an IP literal,
  ``localhost`` or the configured host) is refused with 403: such a name can be
  made to resolve to 127.0.0.1 ("DNS rebinding"), which would make an attacker's
  page same-origin with this server and hide it from the ``Origin`` check below.
* **Origins** — a request that carries an ``Origin`` header (i.e. comes from a
  browser page) is rejected with 403 unless that origin is in
  ``settings.server.cors_origins``; allowed origins get CORS response headers
  and preflight support. Native clients (Swift app, TUI, curl) send no Origin
  and are unaffected. This is what keeps a random web page from talking to the
  local server.
* **Errors** — ``ApiError``/``HubError``/``LibraryError``/``EngineError`` and
  any ``web.HTTPException`` become ``{"error": "<polski komunikat>",
  "code": "<slug>"}`` with the right status; under ``/v1`` the same failure is
  rendered in the OpenAI shape ``{"error": {"message", "type", "code"}}``.
* **Body size** — ``Content-Length`` above ``chat.api.MAX_REQUEST_BYTES`` is
  refused before anything is buffered.

The route modules are imported inside :func:`create_app` so they can import the
helpers below without a circular import.
"""

from __future__ import annotations

import asyncio
import os
import contextlib
import hmac
import ipaddress
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Sequence

from aiohttp import hdrs, web

from localcode.chat.api import MAX_FILE_BYTES, MAX_FILES_PER_REQUEST, MAX_REQUEST_BYTES, FileResult, UploadFile
from localcode.chat.repo import ApiError
from localcode.runtime.base import EngineError
from localcode.server.core import Core

log = logging.getLogger(__name__)

__all__ = [
    "CORE", "TOKEN",
    "create_app",
    "error_payload",
    "error_response",
    "file_response",
    "get_core",
    "json_body",
    "json_response",
    "read_multipart",
    "sse_frame",
    "sse_write",
    "stream_sse",
]

#: Application keys (aiohttp's typed replacement for ``app["core"]``).
CORE: web.AppKey[Core] = web.AppKey("localcode_core", Core)
TOKEN: web.AppKey[str] = web.AppKey("localcode_token", str)

#: Query parameter accepted instead of the Authorization header (SSE/WebSocket).
TOKEN_QUERY_PATHS = ("/api/events",)

SSE_HEADERS = {
    hdrs.CONTENT_TYPE: "text/event-stream; charset=utf-8",
    hdrs.CACHE_CONTROL: "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}
#: How often the SSE writer checks whether the client is still connected.
DISCONNECT_POLL_S = 0.25
#: Heartbeat comment interval for long-lived event streams.
EVENTS_HEARTBEAT_S = 15.0

_CORS_HEADERS = "Authorization, Content-Type, Accept, X-Client-Type, X-Requested-With"
_CORS_METHODS = "GET, POST, PATCH, PUT, DELETE, OPTIONS"
_UNAUTHORIZED = {"error": "Brak autoryzacji — nieprawidłowy token dostępu.",
                 "code": "unauthorized"}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def error_payload(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """Map an exception to ``(status, {"error", "code"})`` with a Polish message."""
    if isinstance(exc, ApiError):
        return exc.status, exc.to_dict()
    status = getattr(exc, "status", None)
    code = getattr(exc, "code", None)
    if isinstance(exc, EngineError):
        return 409, {"error": str(exc) or "Błąd silnika modelu.", "code": "engine_error"}
    if isinstance(status, int) and isinstance(code, str):          # HubError
        return status, {"error": str(exc), "code": code}
    if isinstance(code, str):                                      # LibraryError / SettingsError
        return (404 if code == "not_found" else 400), {"error": str(exc), "code": code}
    if isinstance(exc, ValueError):                                # SettingsError and friends
        return 400, {"error": str(exc) or "Nieprawidłowe dane wejściowe.", "code": "invalid"}
    if isinstance(exc, web.HTTPException):
        return exc.status, {"error": _http_message(exc), "code": _http_code(exc)}
    return 500, {"error": "Wewnętrzny błąd serwera.", "code": "internal_error"}


_HTTP_MESSAGES = {
    401: ("Brak autoryzacji — nieprawidłowy token dostępu.", "unauthorized"),
    403: ("Dostęp zabroniony.", "forbidden"),
    404: ("Nie znaleziono zasobu.", "not_found"),
    405: ("Metoda nieobsługiwana dla tego zasobu.", "method_not_allowed"),
    413: ("Przesłane dane są za duże.", "payload_too_large"),
    415: ("Nieobsługiwany typ zawartości.", "unsupported_media_type"),
}


def _http_message(exc: web.HTTPException) -> str:
    known = _HTTP_MESSAGES.get(exc.status)
    if known:
        return known[0]
    text = (exc.text or "").strip()
    return text or f"Serwer zwrócił błąd {exc.status}."


def _http_code(exc: web.HTTPException) -> str:
    known = _HTTP_MESSAGES.get(exc.status)
    return known[1] if known else f"http_{exc.status}"


def json_response(data: Any, status: int = 200) -> web.Response:
    """JSON body with the exact serialisation the Swift client expects."""
    text = json.dumps(data, ensure_ascii=False, allow_nan=False)
    return web.Response(text=text, status=status, content_type="application/json", charset="utf-8")


def openai_error_body(status: int, payload: dict[str, Any]) -> dict[str, Any]:
    """``{"error": {...}}`` — the shape the ``openai`` client libraries expect."""
    return {"error": {"message": payload.get("error"),
                      "type": "server_error" if status >= 500 else "invalid_request_error",
                      "code": payload.get("code"), "param": None}}


def is_openai_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def error_response(request: web.Request, status: int, payload: dict[str, Any]) -> web.Response:
    """JSON error in the shape that fits the namespace of ``request``."""
    if is_openai_path(request.path):
        return json_response(openai_error_body(status, payload), status=status)
    return json_response(payload, status=status)


def json_error(request: web.Request, exc: BaseException) -> web.Response:
    status, payload = error_payload(exc)
    return error_response(request, status, payload)


# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #


def get_core(request: web.Request) -> Core:
    return request.app[CORE]


async def json_body(request: web.Request) -> Any:
    """Parsed JSON body (``None`` when empty); raises a Polish ``ApiError`` on garbage."""
    if not request.can_read_body:
        return None
    try:
        raw = await request.read()
    except web.HTTPException:
        raise
    except ValueError as exc:
        raise ApiError(413, "Przesłane dane są za duże.", "payload_too_large") from exc
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ApiError(400, "Treść żądania nie jest poprawnym JSON-em.", "invalid_json") from exc


async def read_multipart(request: web.Request) -> tuple[list[UploadFile], dict[str, str]]:
    """Read a ``multipart/form-data`` body into ``([(name, mime, bytes)], fields)``."""
    if not (request.content_type or "").startswith("multipart/"):
        raise ApiError(415, "Oczekiwano formularza multipart/form-data.", "unsupported_media_type")
    try:
        reader = await request.multipart()
    except (ValueError, AssertionError) as exc:
        raise ApiError(400, "Nie udało się odczytać formularza.", "invalid_multipart") from exc
    files: list[UploadFile] = []
    fields: dict[str, str] = {}
    while True:
        try:
            part = await reader.next()
        except (ValueError, AssertionError) as exc:
            raise ApiError(400, "Uszkodzony formularz multipart.", "invalid_multipart") from exc
        if part is None:
            break
        filename = getattr(part, "filename", None)
        if filename:
            if len(files) >= MAX_FILES_PER_REQUEST:
                raise ApiError(413, f"Jednorazowo można wysłać maksymalnie {MAX_FILES_PER_REQUEST} plików.",
                               "too_many_files")
            buffer = bytearray()
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                buffer.extend(chunk)
                if len(buffer) > MAX_FILE_BYTES:
                    raise ApiError(413, f"Plik „{filename}” przekracza limit "
                                        f"{MAX_FILE_BYTES // (1024 * 1024)} MB.", "file_too_large")
            mime = part.headers.get(hdrs.CONTENT_TYPE) or "application/octet-stream"
            files.append((filename, mime.split(";")[0].strip(), bytes(buffer)))
        elif getattr(part, "name", None):
            raw = await part.read(decode=True)
            fields[str(part.name)] = raw.decode("utf-8", "replace")
    return files, fields


def file_response(result: FileResult) -> web.StreamResponse:
    """Turn a :class:`~localcode.chat.api.FileResult` into an aiohttp response."""
    headers = result.headers()
    if result.body is not None:
        return web.Response(body=result.body, headers=headers)
    if result.path is None:
        raise ApiError(404, "Nie znaleziono pliku.", "not_found")
    headers.pop("Content-Length", None)          # FileResponse sets it itself
    return web.FileResponse(result.path, headers=headers)


# --------------------------------------------------------------------------- #
# Server-sent events
# --------------------------------------------------------------------------- #


async def _watch_disconnect(request: web.Request, flag: asyncio.Event) -> None:
    while True:
        transport = request.transport
        if transport is None or transport.is_closing():
            flag.set()
            return
        await asyncio.sleep(DISCONNECT_POLL_S)


def sse_frame(payload: Any) -> bytes:
    """One ``data: {json}`` SSE record."""
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"


async def sse_write(response: web.StreamResponse, data: bytes) -> bool:
    """Write raw bytes to a prepared response; ``False`` when the client is gone."""
    try:
        await response.write(data)
        return True
    except (ConnectionResetError, ConnectionError, RuntimeError, asyncio.CancelledError):
        return False


#: Seconds of silence after which a stream sends a keep-alive comment. Well under
#: the Mac app's sixty-second idle timeout (``APIClient.timeoutIntervalForRequest``).
SSE_HEARTBEAT_S = 15.0


async def stream_sse(
    request: web.Request,
    source: AsyncIterator[Any],
    *,
    initial: Iterable[Any] = (),
    heartbeat: float | None = None,
) -> web.StreamResponse:
    """Stream ``source`` as ``data: {json}`` records until it ends or the client goes away.

    A ``: ping`` comment goes out after every ``heartbeat`` seconds of silence —
    :data:`SSE_HEARTBEAT_S` unless the caller says otherwise. Without it a long
    silent stretch ended the stream from the *client's* side: the Mac app gives
    up on a connection after sixty seconds without a byte, and loading LTX-Video
    plus its text encoder, or a big model before the first token, is easily
    longer than that. The app then cancelled the request, which cancels the job
    — "Serwer nie odpowiedział na czas" on a server that was working fine. Every
    client (Swift, the terminal, the store) skips comment lines.

    A disconnect (the client cancelling the request — this is how the chat
    "stop" button works) closes ``source`` with ``aclose()`` so the producer can
    persist whatever it had.
    """
    if heartbeat is None:
        heartbeat = SSE_HEARTBEAT_S
    response = web.StreamResponse(status=200, headers=SSE_HEADERS)
    await response.prepare(request)
    iterator = source.__aiter__() if hasattr(source, "__aiter__") else source
    gone = asyncio.Event()
    watcher = asyncio.create_task(_watch_disconnect(request, gone), name="sse-disconnect")
    stop = asyncio.ensure_future(gone.wait())
    pending: asyncio.Future[Any] | None = None
    try:
        for frame in initial:
            if not await sse_write(response, sse_frame(frame)):
                return response
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending, stop}, timeout=heartbeat,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                if not await sse_write(response, b": ping\n\n"):
                    break
                continue
            if stop in done:
                break
            try:
                frame = pending.result()
            except StopAsyncIteration:
                pending = None
                break
            except Exception as exc:  # noqa: BLE001 - the response is already committed
                pending = None
                log.warning("strumień SSE %s przerwany błędem", request.path, exc_info=True)
                status, payload = error_payload(exc)
                await sse_write(response, sse_frame({"type": "error", "message": payload["error"],
                                               "code": payload["code"], "status": status}))
                break
            pending = None
            if not await sse_write(response, sse_frame(frame)):
                break
    finally:
        stop.cancel()
        watcher.cancel()
        if pending is not None:
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        with contextlib.suppress(BaseException):
            await stop
        with contextlib.suppress(BaseException):
            await watcher
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(BaseException):
                await aclose()
        with contextlib.suppress(BaseException):
            await response.write_eof()
    return response


# --------------------------------------------------------------------------- #
# Middlewares
# --------------------------------------------------------------------------- #


@web.middleware
async def error_middleware(request: web.Request, handler: Callable[..., Awaitable[Any]]) -> web.StreamResponse:
    try:
        return await handler(request)
    except asyncio.CancelledError:
        raise
    except web.HTTPException as exc:
        if exc.status < 400:
            raise
        return json_error(request, exc)
    except ApiError as exc:
        return json_error(request, exc)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a JSON error
        status, payload = error_payload(exc)
        if status >= 500:
            log.exception("nieobsłużony błąd w %s %s", request.method, request.path)
        else:
            log.info("%s %s → %s (%s)", request.method, request.path, status, payload.get("code"))
        return error_response(request, status, payload)


def _allowed_origins(request: web.Request) -> set[str]:
    core = request.app.get(CORE)
    server = getattr(getattr(core, "settings", None), "server", None)
    origins = list(getattr(server, "cors_origins", None) or [])
    return {o.strip().rstrip("/") for o in origins if isinstance(o, str) and o.strip()}


@web.middleware
async def origin_middleware(request: web.Request, handler: Callable[..., Awaitable[Any]]) -> web.StreamResponse:
    """Reject browser requests from origins that are not allowlisted; add CORS headers."""
    origin = request.headers.get(hdrs.ORIGIN)
    allowed = _allowed_origins(request)
    permitted = bool(origin) and (origin.rstrip("/") in allowed or "*" in allowed)
    if origin and not permitted:
        return json_response({"error": f"Żądania z origin „{origin}” nie są dozwolone.",
                              "code": "origin_not_allowed"}, status=403)
    if request.method == "OPTIONS" and origin:
        response: web.StreamResponse = web.Response(status=204)
    else:
        response = await handler(request)
    if origin and permitted:
        response.headers[hdrs.ACCESS_CONTROL_ALLOW_ORIGIN] = origin
        response.headers[hdrs.ACCESS_CONTROL_ALLOW_HEADERS] = _CORS_HEADERS
        response.headers[hdrs.ACCESS_CONTROL_ALLOW_METHODS] = _CORS_METHODS
        response.headers[hdrs.ACCESS_CONTROL_MAX_AGE] = "600"
        response.headers[hdrs.VARY] = "Origin"
    return response


#: Host names always accepted (besides IP literals and the configured host).
_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


def _host_name(raw: str) -> str:
    """The name part of a ``Host`` header, without the port and the IPv6 brackets."""
    host = raw.strip()
    if host.startswith("["):                       # [::1]:8421
        host = host[1:].partition("]")[0]
    else:
        host = host.rpartition(":")[0] if host.count(":") == 1 else host
    return host.strip().rstrip(".").lower()


def _host_allowed(request: web.Request, raw: str) -> bool:
    """Accept IP literals, ``localhost`` and the configured host — nothing else.

    A DNS name that resolves to 127.0.0.1 (a "DNS rebinding" page) would
    otherwise be same-origin with this server in the browser, so no ``Origin``
    header is sent and :func:`origin_middleware` never sees the attacker.
    """
    host = _host_name(raw)
    if not host or host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    core = request.app.get(CORE)
    server = getattr(getattr(core, "settings", None), "server", None)
    configured = str(getattr(server, "host", "") or "").strip().lower()
    return bool(configured) and host == configured


@web.middleware
async def host_middleware(request: web.Request, handler: Callable[..., Awaitable[Any]]) -> web.StreamResponse:
    """Refuse requests addressed to a DNS name (DNS-rebinding protection)."""
    raw = request.headers.get(hdrs.HOST)
    if raw is not None and not _host_allowed(request, raw):
        return error_response(request, 403,
                              {"error": f"Nagłówek Host „{raw}” nie jest dozwolony.",
                               "code": "host_not_allowed"})
    return await handler(request)


@web.middleware
async def size_middleware(request: web.Request, handler: Callable[..., Awaitable[Any]]) -> web.StreamResponse:
    length = request.content_length
    if length is not None and length > MAX_REQUEST_BYTES:
        return json_response({"error": f"Żądanie przekracza limit "
                                       f"{MAX_REQUEST_BYTES // (1024 * 1024)} MB.",
                              "code": "payload_too_large"}, status=413)
    return await handler(request)


def _bearer(request: web.Request) -> str | None:
    header = request.headers.get(hdrs.AUTHORIZATION, "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return header.strip() or None


def _query_token_allowed(path: str) -> bool:
    return path in TOKEN_QUERY_PATHS or path.endswith("/ws")


def _token_ok(request: web.Request, expected: str) -> bool:
    supplied = _bearer(request)
    if supplied is None and _query_token_allowed(request.path):
        supplied = request.query.get("token")
    if not supplied:
        return False
    return hmac.compare_digest(supplied, expected)


@web.middleware
async def auth_middleware(request: web.Request, handler: Callable[..., Awaitable[Any]]) -> web.StreamResponse:
    """Bearer-token gate for ``/api/*`` and (optionally) ``/v1/*``."""
    path = request.path
    if request.method == "OPTIONS":
        return await handler(request)
    token = request.app.get(TOKEN) or ""
    core = request.app.get(CORE)
    if path.startswith("/v1/"):
        server = getattr(getattr(core, "settings", None), "server", None)
        if server is not None and not server.openai_api_enabled:
            return error_response(request, 404,
                                  {"error": "Serwer zgodny z OpenAI jest wyłączony w ustawieniach.",
                                   "code": "openai_api_disabled"})
        if server is not None and server.openai_api_requires_token and token and not _token_ok(request, token):
            return error_response(request, 401, dict(_UNAUTHORIZED))
        return await handler(request)
    if path.startswith("/api/") and path != "/api/health" and token:
        if not _token_ok(request, token):
            return error_response(request, 401, dict(_UNAUTHORIZED))
    return await handler(request)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def create_app(core: Core, token: str | None = None, *, middlewares: Sequence[Any] = ()) -> web.Application:
    """Build the aiohttp application serving ``core``.

    ``token`` is the bearer token clients must present; an empty token disables
    authentication (useful for embedding the server in tests).
    """
    app = web.Application(
        client_max_size=MAX_REQUEST_BYTES,
        middlewares=[error_middleware, host_middleware, origin_middleware, size_middleware,
                     auth_middleware, *middlewares],
    )
    app[CORE] = core
    app[TOKEN] = token or ""

    from localcode.server import (
        agent_ws,
        openai_api,
        routes_chat,
        routes_images,
        routes_models,
        routes_helpbot,
        routes_remote,
        routes_runtime,
        routes_speech,
        routes_video,
    )

    routes_runtime.register(app)
    routes_models.register(app)
    routes_chat.register(app)
    routes_images.register(app)
    routes_speech.register(app)
    routes_video.register(app)
    routes_remote.register(app)
    routes_helpbot.register(app)
    agent_ws.register(app)
    openai_api.register(app)
    # CORS preflight needs no route: ``origin_middleware`` answers OPTIONS
    # requests that carry an allowlisted Origin before the handler runs.
    app.on_startup.append(_load_pinned_models)
    app.on_startup.append(_fetch_helpbot)
    app.on_shutdown.append(_on_shutdown)
    return app


async def _load_pinned_models(app: web.Application) -> None:
    """Load ``runtime.pinned_models`` in the background as the server starts.

    A model pinned to stay in memory — the small assistant that finds, downloads
    and switches models on request — is only useful if it answers immediately,
    and that means being loaded before the first question rather than after it.
    Failures are logged and ignored: a missing or too-large pinned model must not
    stop the server from starting.
    """
    core = app[CORE]
    keys = list(getattr(getattr(core.settings, "runtime", None), "pinned_models", None) or [])
    if not keys:
        return

    async def warm() -> None:
        for key in keys:
            try:
                await core.runtime.load(key, wait=True)
                log.info("pinned model %s is loaded", key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad key must not break startup
                log.warning("could not load pinned model %s: %s", key, exc)

    app["_pinned_task"] = asyncio.ensure_future(warm())


async def _fetch_helpbot(app: web.Application) -> None:
    """Download the help bot's model on first start — "it comes with the app".

    In the background and never fatal: offline, a full disk or a store error
    leave the server running and ``GET /api/helpbot`` saying why. Skipped when
    the user switched ``helpbot.auto_download`` off.
    """
    core = app[CORE]
    if not getattr(getattr(core.settings, "helpbot", None), "auto_download", False):
        return
    # Tests, CI and offline installs: a server that starts must not start a
    # 2.3 GB download as a side effect.
    if os.environ.get("LOCALCODE_NO_HELPBOT_DOWNLOAD"):
        return

    async def fetch() -> None:
        from localcode import helpbot

        try:
            result = await helpbot.ensure_installed(core, reason="startup")
            if result.get("error"):
                log.warning("helpbot: %s", result["error"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never break startup
            log.warning("helpbot: could not start the download: %s", exc)

    app["_helpbot_task"] = asyncio.ensure_future(fetch())


async def _on_shutdown(app: web.Application) -> None:
    """Close every attached agent WebSocket before the server stops."""
    from localcode.server import agent_ws

    task = app.get("_pinned_task")
    if task is not None and not task.done():
        task.cancel()
    await agent_ws.close_all(app)
