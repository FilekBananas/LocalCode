"""Generating and browsing images from the terminal.

Everything the terminal does with images goes through an :class:`ImageClient`,
which is to ``localcode image`` what :mod:`localcode.tui.backends` is to the
agent: one interface, two implementations.

``LocalImageClient``
    The models run in this process — it talks straight to the
    :class:`~localcode.server.core.Core` held by a
    :class:`~localcode.tui.backends.LocalBackend` (its ``ImageRuntime`` and its
    :class:`~localcode.images.gallery.ImageGallery`).

``RemoteImageClient``
    A ``localcode serve`` (usually started by the Mac app) already holds the
    image model in memory, so the terminal drives it over the documented HTTP
    API and reads the generation as server-sent events.

:func:`image_client` picks the right one from a backend, so ``localcode image``,
``localcode images`` and the ``/obraz`` slash command are written once.

The rendering helpers live here too: the live progress bar, the gallery table
and :func:`inline_preview`, which draws the finished picture right in the
terminal on iTerm2/WezTerm and kitty (and quietly does nothing anywhere else).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from localcode.tui import render
from localcode.tui.backends import Backend, BackendError

log = logging.getLogger(__name__)

__all__ = [
    "ImageClient",
    "LocalImageClient",
    "RemoteImageClient",
    "generate_with_progress",
    "image_client",
    "images_table",
    "inline_preview",
    "open_in_finder",
    "preview_protocol",
    "print_results",
    "save_copy",
    "summary_line",
]

#: Largest picture (in bytes) drawn inline in the terminal.
MAX_PREVIEW_BYTES = 6 * 1024 * 1024
#: Width of the inline preview, in terminal cells.
PREVIEW_CELLS = 40
#: kitty's graphics protocol wants the payload in 4 KB chunks.
KITTY_CHUNK = 4096


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #


class ImageClient:
    """What the terminal needs from "the image side", local or remote."""

    kind = "local"

    async def models(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def generate(self, prompt: str, *, model_key: str | None = None, width: int | None = None,
                 height: int | None = None, steps: int | None = None, seed: int | None = None,
                 guidance: float | None = None, count: int = 1,
                 negative_prompt: str = "",
                 translate: bool | None = None) -> AsyncIterator[dict[str, Any]]:
        """Stream the frames of one generation (see ``docs/IMAGES.md``)."""
        raise NotImplementedError

    async def list(self, *, limit: int = 20, offset: int = 0,
                   favorite: bool = False) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get(self, image_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def delete(self, image_id: str) -> None:
        raise NotImplementedError

    async def set_favorite(self, image_id: str, favorite: bool) -> dict[str, Any]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> "ImageClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------- helpers
    async def resolve(self, image_id: str) -> dict[str, Any]:
        """A gallery entry by full id or by a unique prefix."""
        wanted = (image_id or "").strip()
        if not wanted:
            raise BackendError("Podaj identyfikator obrazu.")
        exact = await self.get(wanted)
        if exact is not None:
            return exact
        rows = await self.list(limit=200)
        hits = [r for r in rows if str(r.get("id", "")).startswith(wanted)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise BackendError(f"Nie znaleziono obrazu „{image_id}”.")
        raise BackendError(f"„{image_id}” pasuje do {len(hits)} obrazów — podaj dłuższy identyfikator.")


def _body(prompt: str, model_key: str | None, width: int | None, height: int | None,
          steps: int | None, seed: int | None, guidance: float | None, count: int,
          negative_prompt: str, translate: bool | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"prompt": prompt, "count": max(1, int(count or 1))}
    if translate is not None:
        body["translatePrompt"] = bool(translate)
    if model_key:
        body["modelKey"] = model_key
    if width:
        body["width"] = int(width)
    if height:
        body["height"] = int(height)
    if steps:
        body["steps"] = int(steps)
    if seed is not None:
        body["seed"] = int(seed)
    if guidance is not None:
        body["guidance"] = float(guidance)
    if negative_prompt:
        body["negativePrompt"] = negative_prompt
    return body


@contextlib.contextmanager
def _as_backend_error() -> Any:
    """Every server-side image failure reaches the terminal as a ``BackendError``.

    The in-process path raises what the HTTP layer would have turned into a
    JSON error (``ApiError``, ``ImageGalleryError``, ``ImageEngineError``);
    the terminal only knows :class:`BackendError`, and all of them already
    carry a Polish message.
    """
    try:
        yield
    except BackendError:
        raise
    except Exception as exc:  # noqa: BLE001 - only the ones with a message for the user
        if not isinstance(getattr(exc, "code", None), str):
            raise                        # a real bug: let it show with its traceback
        raise BackendError(str(getattr(exc, "message", "") or exc)) from exc


class LocalImageClient(ImageClient):
    """Images generated in this process, through the backend's ``Core``."""

    kind = "local"

    def __init__(self, core: Any) -> None:
        self.core = core

    async def models(self) -> list[dict[str, Any]]:
        from localcode.server.routes_images import image_model_dtos

        with _as_backend_error():
            return await asyncio.to_thread(image_model_dtos, self.core)

    async def _frames(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        from localcode.server.routes_images import (
            generate_frames,
            params_from_body,
            resolve_image_model,
        )

        with _as_backend_error():
            self.core.require_images()
            model = resolve_image_model(self.core, body.get("modelKey"))
            params, negative = params_from_body(self.core, body, model)
        async for frame in generate_frames(self.core, model.key, params,
                                           model_name=model.display_name,
                                           negative_prompt=negative):
            yield frame

    def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        return self._frames(_body(prompt, **_generate_kwargs(kwargs)))

    async def list(self, *, limit: int = 20, offset: int = 0,
                   favorite: bool = False) -> list[dict[str, Any]]:
        with _as_backend_error():
            records = await self.core.gallery.alist(limit, offset, favorite)
        return [record.to_dict() for record in records]

    async def get(self, image_id: str) -> dict[str, Any] | None:
        with _as_backend_error():
            record = await self.core.gallery.aget(image_id)
        return record.to_dict() if record is not None else None

    async def delete(self, image_id: str) -> None:
        with _as_backend_error():
            deleted = await self.core.gallery.adelete(image_id)
        if not deleted:
            raise BackendError(f"Nie znaleziono obrazu „{image_id}”.")

    async def set_favorite(self, image_id: str, favorite: bool) -> dict[str, Any]:
        with _as_backend_error():
            record = await self.core.gallery.aset_favorite(image_id, favorite)
        return record.to_dict()


class RemoteImageClient(ImageClient):
    """Images generated by a running ``localcode serve`` (HTTP + SSE)."""

    kind = "remote"

    def __init__(self, info: Any) -> None:
        self.info = info
        self._session: Any = None

    async def _open(self) -> Any:
        if self._session is None:
            import aiohttp

            headers = {"Authorization": f"Bearer {self.info.token}"} if self.info.token else {}
            self._session = aiohttp.ClientSession(
                base_url=self.info.base_url, headers=headers,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None))
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.close()
            self._session = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import aiohttp

        session = await self._open()
        try:
            async with session.request(method, path, **kwargs) as response:
                text = await response.text()
                payload: Any = None
                if text:
                    with contextlib.suppress(ValueError):
                        payload = json.loads(text)
                if response.status >= 400:
                    message = (payload or {}).get("error") if isinstance(payload, dict) else None
                    raise BackendError(message or f"Serwer LocalCode zwrócił błąd {response.status}.")
                return payload
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise BackendError(f"Błąd połączenia z serwerem LocalCode: {exc}") from exc

    async def models(self) -> list[dict[str, Any]]:
        return list(await self._request("GET", "/api/images/models") or [])

    async def _frames(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        import aiohttp

        session = await self._open()
        try:
            async with session.post("/api/images/generate", json=body) as response:
                if response.status >= 400:
                    payload: Any = None
                    with contextlib.suppress(Exception):
                        payload = await response.json(content_type=None)
                    message = (payload or {}).get("error") if isinstance(payload, dict) else None
                    raise BackendError(message or f"Serwer LocalCode zwrócił błąd {response.status}.")
                async for raw in response.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    with contextlib.suppress(ValueError):
                        frame = json.loads(line[5:].strip())
                        if isinstance(frame, dict):
                            yield frame
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise BackendError(f"Błąd połączenia z serwerem LocalCode: {exc}") from exc

    def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        return self._frames(_body(prompt, **_generate_kwargs(kwargs)))

    async def list(self, *, limit: int = 20, offset: int = 0,
                   favorite: bool = False) -> list[dict[str, Any]]:
        params = {"limit": str(limit), "offset": str(offset)}
        if favorite:
            params["favorite"] = "1"
        return list(await self._request("GET", "/api/images", params=params) or [])

    async def get(self, image_id: str) -> dict[str, Any] | None:
        rows = await self.list(limit=200)
        return next((r for r in rows if str(r.get("id")) == image_id), None)

    async def delete(self, image_id: str) -> None:
        from urllib.parse import quote

        await self._request("DELETE", f"/api/images/{quote(image_id, safe='')}")

    async def set_favorite(self, image_id: str, favorite: bool) -> dict[str, Any]:
        from urllib.parse import quote

        return await self._request("PATCH", f"/api/images/{quote(image_id, safe='')}",
                                   json={"favorite": bool(favorite)})


def _generate_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fill in the keyword arguments :func:`_body` expects."""
    return {
        "model_key": kwargs.get("model_key"),
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "steps": kwargs.get("steps"),
        "seed": kwargs.get("seed"),
        "guidance": kwargs.get("guidance"),
        "count": kwargs.get("count", 1),
        "negative_prompt": kwargs.get("negative_prompt", ""),
        "translate": kwargs.get("translate"),
    }


def image_client(backend: Backend) -> ImageClient:
    """The image client that matches ``backend`` (local core or remote server)."""
    info = getattr(backend, "info", None)
    if getattr(backend, "kind", "local") == "remote" and info is not None:
        return RemoteImageClient(info)
    core = getattr(backend, "core", None)
    if core is None:
        raise BackendError("Generowanie obrazów nie jest dostępne w tym trybie.")
    return LocalImageClient(core)


# --------------------------------------------------------------------------- #
# Generating with a progress bar
# --------------------------------------------------------------------------- #


async def generate_with_progress(client: ImageClient, console: Console, prompt: str,
                                 **kwargs: Any) -> list[dict[str, Any]]:
    """Run one generation, drawing a live progress bar; returns the ``ImageDTO``s.

    Errors reported by the server end up as a :class:`BackendError` so every
    caller (CLI, ``/obraz``) shows them the same way.
    """
    count = max(1, int(kwargs.get("count", 1) or 1))
    images: list[dict[str, Any]] = []
    error: str | None = None
    columns = (
        SpinnerColumn(style="lc.accent"),
        TextColumn("[lc.dim]{task.description}"),
        BarColumn(complete_style="lc.accent", finished_style="lc.ok"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    label = _short_prompt(prompt)
    frames = client.generate(prompt, **kwargs)
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(label, total=None)
        try:
            async for frame in frames:
                kind = str(frame.get("type") or "")
                if kind == "status":
                    progress.update(task, description=str(frame.get("message") or label))
                elif kind == "progress":
                    steps = int(frame.get("steps") or 0)
                    index = int(frame.get("index") or 0)
                    step = int(frame.get("step") or 0)
                    if steps > 0:
                        progress.update(task, total=steps * count,
                                        completed=min(index * steps + step, steps * count),
                                        description=label)
                elif kind == "image":
                    image = frame.get("image") or {}
                    if isinstance(image, dict):
                        images.append(image)
                    progress.update(task, description=f"{label} ({len(images)}/{count})")
                elif kind == "error":
                    error = str(frame.get("message") or "Nie udało się wygenerować obrazu.")
                elif kind == "done":
                    total = progress.tasks[task].total
                    if total:
                        progress.update(task, completed=total)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await frames.aclose()
            raise
    if error:
        raise BackendError(error)
    return images


def _short_prompt(prompt: str, limit: int = 42) -> str:
    text = " ".join((prompt or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summary_line(image: dict[str, Any]) -> Text:
    """``✔ 1024×1024 · ziarno 42 · 4 kroki · 3,1 s · /ścieżka/do/pliku.png``."""
    bits = [f"{image.get('width')}×{image.get('height')}"]
    if image.get("seed") is not None:
        bits.append(f"ziarno {image.get('seed')}")
    if image.get("steps"):
        bits.append(f"{image['steps']} kroków")
    if image.get("elapsedMs"):
        bits.append(render.fmt_duration(image.get("elapsedMs")))
    line = Text("✔ ", style="lc.ok")
    line.append(" · ".join(bits), style="lc.dim")
    line.append("  ")
    line.append(str(image.get("path") or ""), style="lc.key")
    return line


def images_table(images: Sequence[dict[str, Any]]) -> Table:
    """The table printed by ``localcode images list``."""
    table = Table(border_style="lc.dim", expand=False, title="Wygenerowane obrazy",
                  title_style="lc.title", title_justify="left")
    table.add_column("id", style="lc.key", no_wrap=True)
    table.add_column("kiedy", no_wrap=True)
    table.add_column("opis", overflow="ellipsis", max_width=44)
    table.add_column("model", overflow="ellipsis", max_width=22)
    table.add_column("rozmiar", justify="right", no_wrap=True)
    table.add_column("plik", justify="right", no_wrap=True)
    for image in images:
        star = "★ " if image.get("favorite") else ""
        table.add_row(
            str(image.get("id") or "")[:8],
            _stamp(image.get("createdAt")),
            star + " ".join(str(image.get("prompt") or "").split()),
            str(image.get("modelName") or image.get("modelKey") or "").split("/")[-1],
            f"{image.get('width')}×{image.get('height')}",
            render.fmt_bytes(image.get("sizeBytes")),
        )
    return table


def _stamp(created_at: Any) -> str:
    try:
        return time.strftime("%d.%m %H:%M", time.localtime(float(created_at or 0) / 1000))
    except (TypeError, ValueError, OSError):  # pragma: no cover - defensive
        return "—"


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def save_copy(image: dict[str, Any], destination: str | os.PathLike[str],
              *, index: int = 0, total: int = 1) -> Path:
    """Copy a generated image to ``destination`` (``plik-2.png`` for a series)."""
    import shutil

    target = Path(destination).expanduser()
    if total > 1:
        target = target.with_name(f"{target.stem}-{index + 1}{target.suffix or '.png'}")
    elif not target.suffix:
        target = target.with_suffix(".png")
    source = Path(str(image.get("path") or ""))
    if not source.is_file():
        raise BackendError(f"Plik obrazu „{source}” nie istnieje.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
    except OSError as exc:
        raise BackendError(f"Nie udało się zapisać pliku „{target}”: {exc}") from exc
    return target


def open_in_finder(path: str | os.PathLike[str]) -> bool:
    """``open`` the file in the system viewer; ``False`` when that is impossible."""
    target = Path(path).expanduser()
    if not target.exists():
        return False
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run([opener, str(target)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


# --------------------------------------------------------------------------- #
# Inline preview (iTerm2 / kitty)
# --------------------------------------------------------------------------- #


def preview_protocol(env: dict[str, str] | None = None) -> str | None:
    """``"iterm2"``, ``"kitty"`` or ``None`` for this terminal."""
    environ = os.environ if env is None else env
    if environ.get("TMUX"):          # passthrough needs extra wrapping; not worth it
        return None
    if environ.get("KITTY_WINDOW_ID") or "kitty" in (environ.get("TERM") or ""):
        return "kitty"
    if environ.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    if (environ.get("TERM_PROGRAM") or "") in ("iTerm.app", "WezTerm"):
        return "iterm2"
    return None


def inline_preview(console: Console, path: str | os.PathLike[str], *,
                   cells: int = PREVIEW_CELLS, env: dict[str, str] | None = None) -> bool:
    """Draw the picture in the terminal; ``False`` when it cannot be drawn."""
    protocol = preview_protocol(env)
    if protocol is None or not console.is_terminal:
        return False
    target = Path(path).expanduser()
    try:
        if target.stat().st_size > MAX_PREVIEW_BYTES:
            return False
        raw = target.read_bytes()
    except OSError:
        return False
    payload = base64.b64encode(raw).decode("ascii")
    try:
        if protocol == "iterm2":
            _write(console, f"\033]1337;File=inline=1;size={len(raw)};width={cells};"
                            f"preserveAspectRatio=1:{payload}\a\n")
        else:
            _write(console, _kitty_sequence(payload, cells))
    except OSError:  # pragma: no cover - a closed terminal
        return False
    return True


def _kitty_sequence(payload: str, cells: int) -> str:
    chunks = [payload[i:i + KITTY_CHUNK] for i in range(0, len(payload), KITTY_CHUNK)] or [""]
    out: list[str] = []
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        head = f"a=T,f=100,c={cells},m={more}" if index == 0 else f"m={more}"
        out.append(f"\033_G{head};{chunk}\033\\")
    out.append("\n")
    return "".join(out)


def _write(console: Console, text: str) -> None:
    stream = getattr(console, "file", None) or sys.stdout
    stream.write(text)
    with contextlib.suppress(Exception):
        stream.flush()


def print_results(console: Console, images: Iterable[dict[str, Any]], *,
                  preview: bool = True) -> None:
    """Print one summary line per picture, with an inline preview when possible."""
    for image in images:
        console.print(summary_line(image))
        if preview:
            with contextlib.suppress(Exception):
                inline_preview(console, str(image.get("path") or ""))
