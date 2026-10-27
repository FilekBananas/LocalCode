"""Speech over HTTP: transcription, synthesis and the voice list.

| route | what it does |
| --- | --- |
| ``POST /api/speech/transcribe`` | recording → text (+ timed segments, + SRT) |
| ``POST /api/speech/speak`` | text → a wave file on disk |
| ``GET  /api/speech/voices`` | the voices the synthesis model offers |

Audio arrives as a **path on this machine**, not as an upload: the app and the
terminal both run next to the server, the files are often hundreds of megabytes,
and copying them through an HTTP body to read them back off disk would be a
detour with no destination.
"""

from __future__ import annotations

from typing import Any

from aiohttp import web

from localcode.server.app import ApiError, get_core, json_body, json_response
from localcode.speech import SpeechError


def register(app: web.Application) -> None:
    app.router.add_get("/api/speech/voices", speech_voices)
    app.router.add_post("/api/speech/transcribe", speech_transcribe)
    app.router.add_post("/api/speech/speak", speech_speak)


def _runtime(core: Any) -> Any:
    """The per-Core speech runtime (cheap, so it is built on first use)."""
    runtime = getattr(core, "_speech", None)
    if runtime is None:
        from localcode.speech import SpeechRuntime

        runtime = SpeechRuntime(core.settings, core.library)
        core._speech = runtime  # noqa: SLF001 - lazy attribute on our own Core
    return runtime


def _error(exc: SpeechError) -> ApiError:
    return ApiError(exc.status, exc.message, exc.code)


async def speech_transcribe(request: web.Request) -> web.Response:
    """``POST /api/speech/transcribe {path, modelKey?, language?, segments?}``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    path = str(body.get("path") or "").strip()
    if not path:
        raise ApiError(400, "Brak pola „path” (ścieżka do nagrania).", "missing_path")
    try:
        transcript = await _runtime(core).transcribe(
            path,
            model_key=str(body.get("modelKey") or "").strip() or None,
            language=str(body.get("language") or "").strip() or None,
            segments=bool(body.get("segments", True)),
        )
    except SpeechError as exc:
        raise _error(exc) from exc
    payload = transcript.to_dict()
    if body.get("srt"):
        payload["srt"] = transcript.to_srt()
    return json_response(payload)


async def speech_speak(request: web.Request) -> web.Response:
    """``POST /api/speech/speak {text, modelKey?, voice?, speed?, output?}``."""
    core = get_core(request)
    body = await json_body(request) or {}
    if not isinstance(body, dict):
        raise ApiError(400, "Treść żądania musi być obiektem JSON.", "invalid_json")
    text = str(body.get("text") or "")
    speed = body.get("speed", 1.0)
    try:
        speed = float(speed)
    except (TypeError, ValueError):
        raise ApiError(400, "Pole „speed” musi być liczbą.", "invalid_param") from None
    if not 0.5 <= speed <= 2.0:
        raise ApiError(400, "Pole „speed” musi mieścić się w zakresie 0,5–2,0.", "invalid_param")
    try:
        result = await _runtime(core).speak(
            text,
            model_key=str(body.get("modelKey") or "").strip() or None,
            voice=str(body.get("voice") or "").strip() or None,
            speed=speed,
            output=str(body.get("output") or "").strip() or None,
        )
    except SpeechError as exc:
        raise _error(exc) from exc
    return json_response(result.to_dict(), status=201)


async def speech_voices(request: web.Request) -> web.Response:
    """``GET /api/speech/voices?modelKey=`` → ``[{id, label, language}]``."""
    core = get_core(request)
    key = (request.query.get("modelKey") or "").strip() or None
    try:
        voices = _runtime(core).voices(key)
    except SpeechError as exc:
        raise _error(exc) from exc
    return json_response([v.to_dict() for v in voices])
