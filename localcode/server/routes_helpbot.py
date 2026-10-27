"""The help bot over HTTP: is it there, get it, talk to it.

| route | what it does |
| --- | --- |
| ``GET  /api/helpbot`` | ready / downloading / missing, plus the model and whether it is loaded |
| ``POST /api/helpbot/install`` | start downloading the help bot's model (checks free disk space) |
| ``POST /api/helpbot/session`` | the help bot's conversation — the latest one, or ``{"new": true}`` |

The conversation itself is an ordinary agent session (``/api/agent/sessions/{id}/ws``):
the help bot is the agent loop with another prompt and other tools, so the app
talks to it with the socket, the transcript and the permission cards it already
has. What makes a session the help bot's is its working directory
(:func:`localcode.helpbot.helpbot_dir`).
"""

from __future__ import annotations

from typing import Any

from aiohttp import web

from localcode.server.app import ApiError, get_core, json_body, json_response


def register(app: web.Application) -> None:
    app.router.add_get("/api/helpbot", helpbot_status)
    app.router.add_post("/api/helpbot/install", helpbot_install)
    app.router.add_post("/api/helpbot/session", helpbot_session)


async def helpbot_status(request: web.Request) -> web.Response:
    from localcode import helpbot

    return json_response(helpbot.status(get_core(request)))


async def helpbot_install(request: web.Request) -> web.Response:
    from localcode import helpbot

    result = await helpbot.ensure_installed(get_core(request), reason="user")
    if result.get("error"):
        raise ApiError(409, str(result["error"]), "helpbot_install_failed")
    return json_response(result, status=202 if result.get("state") == "downloading" else 200)


async def helpbot_session(request: web.Request) -> web.Response:
    body: Any = await json_body(request) if request.can_read_body else {}
    fresh = bool((body or {}).get("new"))
    core = get_core(request)
    state = core.agents.helpbot_session(new=fresh)
    return json_response(state.dto(), status=201 if fresh else 200)
