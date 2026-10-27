"""The plain chatbot (no tools) — ``localcode chat`` and ``/chat`` in the TUI.

Conversations are stored in the same SQLite database as the Mac app's "Czat"
tab (``~/.localcode/data/chat.db``), so whatever is typed here shows up there
and vice versa.  Locally that means calling :class:`localcode.chat.service.ChatService`
directly; attached to a server it means ``POST /api/chat`` with the ai_ux SSE
frames — the same frames the SwiftUI client decodes.

Context economy works exactly like in the agent: the ``context`` frame feeds
``/context`` (the same breakdown table) and a ``compaction`` frame prints the
one-liner ``Kontekst skompaktowany: 12,4k → 5,1k tokenów`` whenever the older
turns were summarised — automatically during a turn, or on demand with
``/compact`` (``POST /api/conversations/{id}/compact``).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from localcode.tui import render
from localcode.tui.backends import Backend, BackendError, LocalBackend, RemoteBackend
from localcode.tui.stream import TurnView

__all__ = ["ChatSession", "chat_command", "run_chat_turn"]


class ChatSession:
    """One chatbot conversation, local or over the server API."""

    def __init__(self, backend: Backend, *, model_key: str | None = None,
                 conversation_id: str | None = None) -> None:
        self.backend = backend
        self.model_key = model_key or backend.model_key
        self.conversation_id = conversation_id
        self.title: str | None = None
        self.last_usage: dict[str, Any] | None = None
        #: the last ``context`` frame — ``{"used", "limit", "system", "history", …}``
        self.context: dict[str, Any] | None = None

    async def send(self, text: str) -> AsyncIterator[dict[str, Any]]:
        """Yield ai_ux SSE frames for one message."""
        body: dict[str, Any] = {"content": text}
        if self.conversation_id:
            body["conversationId"] = self.conversation_id
        if self.model_key:
            body["model"] = self.model_key
        if isinstance(self.backend, LocalBackend):
            stream = self.backend.core.chat.send(body)
        elif isinstance(self.backend, RemoteBackend):
            stream = self._remote(body)
        else:  # pragma: no cover - defensive
            raise BackendError("Czat nie jest dostępny w tym trybie.")
        async for frame in stream:
            self._absorb(frame)
            yield frame

    async def _remote(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        backend = self.backend
        assert isinstance(backend, RemoteBackend)
        session = backend._session  # noqa: SLF001 - one client, one session
        if session is None:
            raise BackendError("Brak połączenia z serwerem LocalCode.")
        async with session.post("/api/chat", json=body) as response:
            if response.status >= 400:
                payload = await response.json(content_type=None)
                message = (payload or {}).get("error") if isinstance(payload, dict) else None
                raise BackendError(message or f"Serwer zwrócił błąd {response.status}.")
            async for raw in response.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    frame = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(frame, dict):
                    yield frame

    async def compact(self) -> dict[str, Any]:
        """Summarise the older turns now (``/compact``).

        Returns the endpoint's ``{"summary", "before", "after", "changed", …}``.
        """
        if not self.conversation_id:
            raise BackendError("Nie ma jeszcze czego kompaktować — wyślij pierwszą wiadomość.")
        backend = self.backend
        if isinstance(backend, LocalBackend):
            return await backend.core.chat_api.compact_conversation(self.conversation_id)
        if isinstance(backend, RemoteBackend):
            session = backend._session  # noqa: SLF001 - one client, one session
            if session is None:
                raise BackendError("Brak połączenia z serwerem LocalCode.")
            async with session.post(f"/api/conversations/{self.conversation_id}/compact") as response:
                payload = await response.json(content_type=None)
                if response.status >= 400:
                    message = (payload or {}).get("error") if isinstance(payload, dict) else None
                    raise BackendError(message or f"Serwer zwrócił błąd {response.status}.")
                return dict(payload or {})
        raise BackendError("Czat nie jest dostępny w tym trybie.")

    def _absorb(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "start" and frame.get("conversationId"):
            self.conversation_id = str(frame["conversationId"])
        elif kind == "title" and frame.get("title"):
            self.title = str(frame["title"])
        elif kind == "usage":
            self.last_usage = dict(frame)
        elif kind == "context":
            self.context = dict(frame)
        elif kind == "compaction":
            # keep the gauge honest between turns
            if self.context:
                self.context = {**self.context, "used": int(frame.get("after") or 0)}


async def run_chat_turn(session: ChatSession, console: Console, text: str, *,
                        show_reasoning: bool = False) -> bool:
    """Stream one chatbot answer; ``False`` when the turn failed."""
    ok = True
    stream = session.send(text)
    with TurnView(console, show_reasoning=show_reasoning) as view:
        view.set_status("Myślę…")
        try:
            async for frame in stream:
                kind = frame.get("type")
                if kind == "text":
                    view.feed_text(str(frame.get("delta") or ""))
                elif kind == "reasoning":
                    view.feed_reasoning(str(frame.get("delta") or ""))
                elif kind == "status":
                    view.set_status(str(frame.get("message") or ""))
                elif kind == "citations":
                    citations = frame.get("citations") or []
                    if citations:
                        view.print(Text("Źródła: " + ", ".join(
                            str(c.get("title") or c.get("id")) for c in citations), style="lc.dim"))
                elif kind == "compaction":
                    view.print(render.compaction_line(int(frame.get("before") or 0),
                                                      int(frame.get("after") or 0),
                                                      str(frame.get("strategy") or "")))
                elif kind == "usage":
                    view.end_assistant()      # the answer belongs above its statistics
                    view.print(Text(
                        f"  {render.fmt_tokens(frame.get('tokensIn'))} tokenów wejścia · "
                        f"{render.fmt_tokens(frame.get('tokensOut'))} tokenów odpowiedzi",
                        style="lc.dim"))
                elif kind == "error":
                    ok = False
                    view.print(render.error_line(str(frame.get("message") or "Błąd czatu.")))
                elif kind == "done":
                    view.end_assistant()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await stream.aclose()
            view.print(Text("Przerwano.", style="lc.warn"))
            raise
        except BackendError as exc:
            ok = False
            view.print(render.error_line(str(exc)))
        finally:
            view.set_status(None)
    return ok


async def chat_command(session: ChatSession | None, console: Console, name: str) -> bool:
    """Run ``/compact`` or ``/context`` against the chatbot conversation.

    Returns ``False`` for a command this mode does not handle, so the caller can
    fall back to the agent's implementation.
    """
    if name == "context":
        console.print(render.context_breakdown((session.context if session else None)))
        return True
    if name != "compact":
        return False
    if session is None or not session.conversation_id:
        console.print(Text("Nie ma jeszcze czego kompaktować — wyślij pierwszą wiadomość.", style="lc.dim"))
        return True
    console.print(Text("Kompaktuję rozmowę…", style="lc.dim"))
    try:
        result = await session.compact()
    except BackendError as exc:
        console.print(render.error_line(str(exc)))
        return True
    before, after = int(result.get("before") or 0), int(result.get("after") or 0)
    if not result.get("changed"):
        console.print(Text("Rozmowa jest wystarczająco krótka — nie ma czego kompaktować.", style="lc.dim"))
        return True
    console.print(render.compaction_line(before, after, str(result.get("strategy") or "")))
    if session.context:
        session.context = {**session.context, "used": after}
    return True


def render_history(console: Console, messages: list[dict[str, Any]], *, limit: int = 20) -> None:
    """Print the tail of a stored conversation when it is reopened."""
    for message in messages[-limit:]:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            console.print(Text("› " + content, style="lc.user"))
        else:
            console.print(Markdown(content))
