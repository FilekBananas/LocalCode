"""Coding-agent sessions: REST + the WebSocket protocol.

REST
----
``POST /api/agent/sessions {cwd, modelKey?, mode?}`` → ``AgentSession``
``GET  /api/agent/sessions[?cwd=]``                 → ``[AgentSession]``
``GET  /api/agent/sessions/{id}``                   → ``{session, items}``
``DELETE /api/agent/sessions/{id}``                 → 204
``GET  /api/agent/auto-precedents[?cwd=]``          → ``{enabled, projects: [{cwd, precedents}]}``
``DELETE /api/agent/auto-precedents[?cwd=]``        → ``{removed}`` (no ``cwd``: every project)

WebSocket ``GET /api/agent/sessions/{id}/ws``
---------------------------------------------
Several clients may watch one session (the terminal TUI and the Mac app at the
same time); every agent event is broadcast to all of them.

client → server::

    {"type":"user","text":"…"}
    {"type":"permission","requestId":"…","decision":"allow"|"deny"|"always"}
    {"type":"permission","requestId":"…","decision":"dismissed"}   nobody decided (a script
        that cannot ask): refused like "deny", but not learned from (agent/precedents.py)
    {"type":"interrupt"} | {"type":"setMode","mode":"…"} | {"type":"setModel","modelKey":"…"}
    {"type":"setEffort","effort":"off"|"low"|"medium"|"high"|"ultracode"}
    {"type":"compact"} | {"type":"clear"}
    {"type":"user","text":"…","images"?:[{"name","mime","data":"<base64>"}]}   pictures for a vision model
    {"type":"setGoal","text":"…","message"?:"…"}   /goal: set it, then start the turn
    {"type":"clearGoal"}

server → client: the ``localcode.events`` dicts (``turn_start``, ``text_delta``,
``reasoning_delta``, ``tool_call``, ``tool_result``, ``permission_request``,
``auto_review``, ``context``, ``compaction``, ``status``, ``todos``, ``goal``,
``turn_end``, ``error``)
plus::

    {"type":"session","session":AgentSession}   on connect and on every change
    {"type":"history","items":[AgentItem]}      on connect (and after "clear")
    {"type":"ack","of":"user","status":"started"|"queued"}
    {"type":"permission_resolved","requestId":"…","decision":"…"}

A user message sent while a turn is running is queued (the client gets
``status: "queued"`` plus a ``status`` event); a permission request is resolved
by whichever client answers first and is re-sent to clients that connect while
it is still pending.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from aiohttp import WSMsgType, web

from localcode.chat.repo import ApiError
from localcode.server.app import get_core, json_body, json_response
from localcode.events import PERMISSION_DISMISSED
from localcode.server.core import PERMISSION_DECISIONS, AgentClient, AgentSessionState

log = logging.getLogger(__name__)

__all__ = ["close_all", "register"]

#: Open WebSocket clients of this application (closed on shutdown).
WS_CLIENTS: web.AppKey[set["WsClient"]] = web.AppKey("localcode_ws_clients", set)
#: Events buffered per client before the slowest one starts losing the oldest.
CLIENT_QUEUE_SIZE = 1000
WS_HEARTBEAT_S = 30.0


def register(app: web.Application) -> None:
    app.setdefault(WS_CLIENTS, set())
    app.router.add_post("/api/agent/sessions", create_session)
    app.router.add_get("/api/agent/sessions", list_sessions)
    app.router.add_get("/api/agent/sessions/{id}", get_session)
    app.router.add_delete("/api/agent/sessions/{id}", delete_session)
    app.router.add_get("/api/agent/sessions/{id}/ws", session_ws)
    app.router.add_get("/api/agent/commands", agent_commands)
    app.router.add_get("/api/agent/auto-precedents", list_precedents)
    app.router.add_delete("/api/agent/auto-precedents", clear_precedents)


async def agent_commands(request: web.Request) -> web.Response:
    """``GET /api/agent/commands`` → the slash commands a graphical client offers.

    One table, two interfaces. The terminal reads it directly
    (``localcode/tui/commands.py``) and the Mac app over this route, so adding a
    command makes it appear in both — which is the point: kept as two lists they
    had drifted to four commands against twenty-three.

    ``initPrompt`` travels with them because ``/init`` *is* a prompt, and a
    second copy of it in Swift would drift the same way.
    """
    # The table lives next to the terminal app for historical reasons; it has no
    # terminal dependencies, it is plain data.
    from localcode.agent.prompts import INIT_PROMPT    # noqa: PLC0415
    from localcode.tui.commands import COMMANDS        # noqa: PLC0415

    return json_response({
        "commands": [{
            "name": c.name,
            "summary": c.help,
            "usage": c.usage or None,
            "aliases": list(c.aliases),
            "values": list(c.values),
        } for c in COMMANDS if c.in_app],
        "initPrompt": INIT_PROMPT,
    })


async def list_precedents(request: web.Request) -> web.Response:
    """``GET /api/agent/auto-precedents[?cwd=]`` → what was learned from the user's answers.

    ``{enabled, projects: [{cwd, precedents: [{decision, kind, key, count, lastUsed}]}]}``
    — one project with ``cwd`` (by its real path), else every one, most recent
    first (``agent/precedents.py``).
    """
    from localcode.agent import precedents    # noqa: PLC0415

    core = get_core(request)
    cwd = (request.query.get("cwd") or "").strip() or None
    projects = await asyncio.to_thread(precedents.listing, cwd)
    return json_response({"enabled": bool(core.settings.agent.auto_precedents_enabled),
                          "projects": projects})


async def clear_precedents(request: web.Request) -> web.Response:
    """``DELETE /api/agent/auto-precedents[?cwd=]`` → ``{removed}``: forget one project's, or all."""
    from localcode.agent import precedents    # noqa: PLC0415

    cwd = (request.query.get("cwd") or "").strip() or None
    removed = await asyncio.to_thread(precedents.clear, cwd)
    return json_response({"removed": removed})


async def close_all(app: web.Application) -> None:
    """Close every attached WebSocket (server shutdown)."""
    for client in list(app.get(WS_CLIENTS) or ()):
        with contextlib.suppress(Exception):
            await client.close()


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


async def create_session(request: web.Request) -> web.Response:
    """``POST /api/agent/sessions {cwd, modelKey?, mode?}``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    session = core.agents.create(_opt_str(body, "cwd"), _opt_str(body, "modelKey"), _opt_str(body, "mode"))
    return json_response(session.dto(), status=201)


async def list_sessions(request: web.Request) -> web.Response:
    """``GET /api/agent/sessions[?cwd=]``."""
    core = get_core(request)
    return json_response(core.agents.list(request.query.get("cwd")))


async def get_session(request: web.Request) -> web.Response:
    """``GET /api/agent/sessions/{id}`` → ``{session, items}``."""
    core = get_core(request)
    return json_response(core.agents.require(request.match_info["id"]).snapshot())


async def delete_session(request: web.Request) -> web.Response:
    """``DELETE /api/agent/sessions/{id}``."""
    core = get_core(request)
    if not await core.agents.delete(request.match_info["id"]):
        raise ApiError(404, "Nie znaleziono sesji agenta.", "not_found")
    return web.Response(status=204)


def _opt_str(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(400, f"Pole „{key}” musi być tekstem.", "validation_error")
    return value.strip() or None


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


class WsClient(AgentClient):
    """One attached WebSocket. Events are queued so the agent never blocks on IO."""

    def __init__(self, ws: web.WebSocketResponse, session: AgentSessionState) -> None:
        self.ws = ws
        self.session = session
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(CLIENT_QUEUE_SIZE)
        self.dropped = 0

    def send_event(self, payload: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(payload)

    async def writer(self) -> None:
        while True:
            payload = await self.queue.get()
            if payload is None:
                return
            if self.ws.closed:
                return
            try:
                await self.ws.send_json(payload, dumps=_dumps)
            except (ConnectionResetError, ConnectionError, RuntimeError):
                return

    def stop(self) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(None)

    async def close(self) -> None:
        self.stop()
        with contextlib.suppress(Exception):
            await self.ws.close()


async def session_ws(request: web.Request) -> web.StreamResponse:
    """``GET /api/agent/sessions/{id}/ws`` — the live agent protocol."""
    core = get_core(request)
    session = core.agents.require(request.match_info["id"])
    # Room for a few pictures in one message (aiohttp refuses anything over 4 MB by default).
    ws = web.WebSocketResponse(heartbeat=WS_HEARTBEAT_S, max_msg_size=MAX_IMAGES * MAX_IMAGE_B64 + 1024 * 1024)
    ready = ws.can_prepare(request)
    if not ready.ok:
        raise ApiError(400, "Ten zasób wymaga połączenia WebSocket.", "websocket_required")
    await ws.prepare(request)

    client = WsClient(ws, session)
    clients: set[WsClient] = request.app.setdefault(WS_CLIENTS, set())
    clients.add(client)
    writer = asyncio.create_task(client.writer(), name=f"agent-ws-writer-{session.id}")
    session.attach(client)
    try:
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                await _dispatch(session, client, message.data)
            elif message.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSING):
                break
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - a client error must not kill the session
        log.warning("połączenie WebSocket sesji %s zakończone błędem", session.id, exc_info=True)
    finally:
        session.detach(client)
        clients.discard(client)
        client.stop()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(writer, timeout=2.0)
        if not writer.done():
            writer.cancel()
        with contextlib.suppress(Exception):
            await ws.close()
    return ws


#: Pictures in one message, and the size of each (base64 characters).
MAX_IMAGES = 6
MAX_IMAGE_B64 = 28 * 1024 * 1024
_IMAGE_MIMES = ("image/png", "image/jpeg", "image/gif", "image/webp")


def _images_of(raw: Any) -> list[dict[str, Any]] | None:
    """``[{name, mime, data}]`` from a ``user`` message, validated; ``None`` when there are none."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ApiError(400, "Pole „images” musi być listą.", "invalid_images")
    if len(raw) > MAX_IMAGES:
        raise ApiError(400, f"Najwyżej {MAX_IMAGES} obrazów w jednej wiadomości.", "too_many_images")
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("data"), str) or not item["data"]:
            raise ApiError(400, "Każdy obraz to {name, mime, data (base64)}.", "invalid_images")
        mime = str(item.get("mime") or "image/png").lower()
        if mime not in _IMAGE_MIMES:
            raise ApiError(400, f"Nieobsługiwany format obrazu: {mime}.", "invalid_images")
        if len(item["data"]) > MAX_IMAGE_B64:
            raise ApiError(400, "Obraz jest za duży (najwyżej ok. 20 MB).", "image_too_large")
        out.append({"name": str(item.get("name") or "obraz"), "mime": mime, "data": item["data"]})
    return out or None


async def _dispatch(session: AgentSessionState, client: WsClient, raw: str) -> None:
    """Handle one client message; failures are reported to that client only."""
    try:
        message = json.loads(raw)
    except ValueError:
        client.send_event({"type": "error", "message": "Nieprawidłowy JSON w wiadomości WebSocket."})
        return
    if not isinstance(message, dict):
        client.send_event({"type": "error", "message": "Wiadomość musi być obiektem JSON."})
        return
    kind = message.get("type")
    try:
        if kind == "user":
            images = _images_of(message.get("images"))
            status = await session.send_user(str(message.get("text") or ""), images)
            client.send_event({"type": "ack", "of": "user", "status": status})
        elif kind == "permission":
            request_id = str(message.get("requestId") or "")
            decision = str(message.get("decision") or "")
            if decision not in PERMISSION_DECISIONS and decision != PERMISSION_DISMISSED:
                raise ApiError(400, f"Nieznana decyzja „{decision}”. Dozwolone: "
                                    f"{', '.join(PERMISSION_DECISIONS)}.", "invalid_decision")
            if not session.resolve_permission(request_id, decision):
                client.send_event({"type": "ack", "of": "permission", "status": "stale",
                                   "requestId": request_id})
        elif kind == "interrupt":
            await session.interrupt()
        elif kind == "setMode":
            await session.set_mode(str(message.get("mode") or ""))
        elif kind == "setModel":
            await session.set_model(str(message.get("modelKey") or ""))
        elif kind == "setEffort":
            await session.set_effort(str(message.get("effort") or ""))
        elif kind == "compact":
            await session.compact()
        elif kind == "clear":
            await session.clear()
        elif kind == "setGoal":
            status = await session.start_goal(str(message.get("text") or ""),
                                              str(message.get("message") or "") or None)
            client.send_event({"type": "ack", "of": "setGoal", "status": status})
        elif kind == "clearGoal":
            await session.clear_goal()
        else:
            client.send_event({"type": "error", "message": f"Nieznany typ wiadomości „{kind}”."})
    except ApiError as exc:
        client.send_event({"type": "error", "message": exc.message, "code": exc.code})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - reported to the client
        log.warning("obsługa wiadomości %r nie powiodła się", kind, exc_info=True)
        client.send_event({"type": "error", "message": str(exc) or "Nieoczekiwany błąd serwera."})
