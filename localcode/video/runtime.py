"""Video generation, one clip at a time, in a worker subprocess.

Unlike :mod:`localcode.speech` (a model loaded, used and released inside one
``asyncio.to_thread`` call) a video generation genuinely needs a *process*:
it is slow enough — minutes per second of footage on this hardware — that the
caller needs progress as it happens, and diffusion frameworks are heavy and
occasionally crash or hang on a Metal kernel; neither should be able to take
the LocalCode server down. So this module spawns
``python -u -m localcode.video.worker``, feeds it one JSON payload on stdin,
and reads one JSON object per line back from its stdout — the same "own
process group, log file, SIGTERM→SIGKILL, atexit safety net" plumbing every
other engine subprocess in this codebase uses
(:mod:`localcode.runtime.http_engine`), imported rather than duplicated.

There is no resident instance and no RAM-budget bookkeeping here (unlike
:mod:`localcode.images.runtime`): a video model is not something you keep
warm between chat turns, and LocalCode only targets one small model today
(LTX-Video 2B — see :mod:`localcode.video.base`), so the extra machinery
would not pay for itself yet.

The ``video`` extra (``torch`` + ``diffusers`` + ``transformers``) is optional;
without it every entry point explains what to install instead of failing with
an ImportError — the worker itself detects the missing import and reports it
over the same JSON protocol, so this module never has to import ``torch`` to
know that.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from localcode.config import home, logs_dir
from localcode.runtime.http_engine import (  # shared process plumbing (not modified here)
    _register_group,
    _safe_name,
    _terminate_process_group,
    _unregister_group,
)
from localcode.video.base import (
    DEFAULT_FPS,
    DEFAULT_GUIDANCE,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    INSTALL_HINT,
    VideoError,
    VideoParams,
    VideoProgress,
    VideoResult,
    classify_video_model,
    default_guidance_for,
    estimate_seconds,
    new_video_id,
    plan_message,
)

log = logging.getLogger(__name__)

__all__ = ["VideoRuntime"]

#: Grace period given to the worker after SIGTERM before SIGKILL (generation
#: can be mid-Metal-kernel; llama-server/images workers use similar values).
_STOP_GRACE_S = 5.0

Runner = Callable[[str, str, VideoParams, str, "Callable[[VideoProgress], Any] | None"],
                  Awaitable[dict[str, Any]]]


class VideoRuntime:
    """Text-to-video for one machine."""

    def __init__(self, settings: Any, library: Any = None, *,
                 runner: Runner | None = None,
                 worker_cmd: list[str] | None = None) -> None:
        self.settings = settings
        self._library = library
        self._runner = runner            # test seam: replaces the whole subprocess
        self._worker_cmd = worker_cmd     # test seam: replaces just the argv (real subprocess)

    # plumbing ----------------------------------------------------------------
    @property
    def library(self) -> Any:
        if self._library is None:
            from localcode.hub.library import ModelLibrary

            self._library = ModelLibrary(self.settings)
        return self._library

    def list_models(self) -> list[dict[str, Any]]:
        """Every model on disk that looks like a video model, supported or not.

        ``hub/library.py`` does not tag a "video" capability today (see
        ``docs/VIDEO.md``), so this is the same best-effort name/size
        recognition :func:`~localcode.video.base.classify_video_model` uses
        everywhere else in this module.
        """
        out: list[dict[str, Any]] = []
        for model in self.library.list():
            kind, _base_repo = classify_video_model(model)
            if kind == "unknown":
                continue
            dto = model.to_dict()
            dto["videoKind"] = kind
            dto["supported"] = kind == "ltx-2b"
            out.append(dto)
        return out

    def _model(self, key: str | None) -> tuple[Any, str]:
        """(model, base diffusers repo) for a supported LTX-Video 2B checkpoint."""
        if key:
            model = self.library.get(key) or self.library.resolve(key)
            if model is None:
                raise VideoError(f"Nie znaleziono modelu „{key}”.", "model_not_found", 404)
            kind, base_repo = classify_video_model(model)
            if kind != "ltx-2b":
                raise VideoError(_unsupported_message(model, kind), "unsupported_model")
            return model, base_repo or "Lightricks/LTX-Video"
        classified = [(m, *classify_video_model(m)) for m in self.library.list()]
        for model, kind, base_repo in classified:
            if kind == "ltx-2b":
                return model, base_repo or "Lightricks/LTX-Video"
        other = next((m for m, kind, _ in classified if kind in ("ltx-other", "other-video")), None)
        if other is not None:
            raise VideoError(_unsupported_message(other, classify_video_model(other)[0]),
                             "unsupported_model")
        raise VideoError(
            "Nie masz jeszcze modelu do generowania wideo (LTX-Video 2B). Zainstaluj go w "
            "sklepie (kategoria „Wideo”), np. `localcode models search ltx-video`.",
            "no_model")

    def output_dir(self) -> Path:
        """Where generated clips go by default (``~/.localcode/data/video``)."""
        return home() / "data" / "video"

    def _default_output(self) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return self.output_dir() / f"wideo-{stamp}.mp4"

    # estimate ------------------------------------------------------------------
    def estimate(self, *, width: int | None = None, height: int | None = None,
                num_frames: int | None = None, fps: int | None = None,
                steps: int | None = None) -> dict[str, Any]:
        """The rough, honestly-unverified time/shape estimate for a request.

        Does not touch the library or require a prompt — this is meant to be
        callable *before* the user has committed to anything.
        """
        params = VideoParams(
            prompt="", negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=width if width is not None else DEFAULT_WIDTH,
            height=height if height is not None else DEFAULT_HEIGHT,
            num_frames=num_frames if num_frames is not None else DEFAULT_NUM_FRAMES,
            fps=fps if fps is not None else DEFAULT_FPS,
            steps=steps if steps is not None else DEFAULT_STEPS,
            guidance=DEFAULT_GUIDANCE,
        ).normalized(require_prompt=False, enforce_frame_cap=False)
        seconds = estimate_seconds(width=params.width, height=params.height,
                                   num_frames=params.num_frames, steps=params.steps)
        return {
            "seconds": round(seconds, 1), "message": plan_message(params, seconds),
            "width": params.width, "height": params.height, "numFrames": params.num_frames,
            "fps": params.fps, "steps": params.steps,
        }

    # generation ------------------------------------------------------------------
    async def generate(self, prompt: str, *, model_key: str | None = None,
                       negative_prompt: str | None = None, width: int | None = None,
                       height: int | None = None, num_frames: int | None = None,
                       fps: int | None = None, steps: int | None = None,
                       guidance: float | None = None, seed: int | None = None,
                       force: bool = False, output: str | Path | None = None,
                       on_progress: Callable[[VideoProgress], Any] | None = None) -> VideoResult:
        """Prompt → an MP4 on disk. Raises :class:`VideoError` (Polish) on failure."""
        model, base_repo = self._model(model_key)
        params = VideoParams(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT,
            width=width if width is not None else DEFAULT_WIDTH,
            height=height if height is not None else DEFAULT_HEIGHT,
            num_frames=num_frames if num_frames is not None else DEFAULT_NUM_FRAMES,
            fps=fps if fps is not None else DEFAULT_FPS,
            steps=steps if steps is not None else DEFAULT_STEPS,
            # A distilled checkpoint wants no guidance at all — with the usual
            # value it returns noise (see DISTILLED_GUIDANCE).
            guidance=guidance if guidance is not None else default_guidance_for(model),
            seed=seed, force=force,
        ).normalized()
        target = Path(output).expanduser() if output else self._default_output()
        target.parent.mkdir(parents=True, exist_ok=True)
        estimated = estimate_seconds(width=params.width, height=params.height,
                                     num_frames=params.num_frames, steps=params.steps)
        await _call(on_progress, VideoProgress(
            stage="loading", step=0, steps=0,
            message=f"Przygotowuję {model.display_name}. {plan_message(params, estimated)}"))
        started = time.monotonic()
        runner = self._runner or self._run_worker
        try:
            raw = await runner(model.path, base_repo, params, str(target), on_progress)
        except VideoError:
            raise
        except ImportError as exc:
            raise VideoError(INSTALL_HINT, "missing_extra", 501) from exc
        except Exception as exc:  # noqa: BLE001 - one bad generation must not kill the server
            raise VideoError(f"Nie udało się wygenerować wideo: {exc}", "generate_failed", 500) from exc
        path = Path(str(raw.get("path") or target))
        return VideoResult(
            id=new_video_id(),
            path=str(path),
            width=int(raw.get("width") or params.width),
            height=int(raw.get("height") or params.height),
            num_frames=int(raw.get("numFrames") or params.num_frames),
            fps=int(raw.get("fps") or params.fps),
            seed=int(raw.get("seed") if raw.get("seed") is not None else params.seed or 0),
            steps=int(raw.get("steps") or params.steps),
            guidance=float(raw.get("guidance") if raw.get("guidance") is not None else params.guidance),
            model_key=model.key,
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            size_bytes=int(raw.get("sizeBytes") or (path.stat().st_size if path.exists() else 0)),
            estimated_seconds=estimated,
        )

    # the real subprocess --------------------------------------------------------
    async def _run_worker(self, model_path: str, base_repo: str, params: VideoParams,
                          output_path: str,
                          on_progress: Callable[[VideoProgress], Any] | None) -> dict[str, Any]:
        python = getattr(getattr(self.settings, "runtime", None), "python_path", None) or sys.executable
        cmd = list(self._worker_cmd) if self._worker_cmd else [python, "-u", "-m", "localcode.video.worker"]
        payload = dict(params.to_payload())
        payload.update({"modelPath": model_path, "baseRepo": base_repo, "outputPath": output_path})
        label = _safe_name(Path(output_path).stem)
        log_path = logs_dir() / f"video-{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_file = open(log_path, "ab")  # noqa: SIM115 - closed in finally
        except OSError as exc:
            raise VideoError(f"Nie udało się otworzyć pliku logu: {exc}", "spawn_failed", 500) from exc
        try:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=log_file, start_new_session=True)
            except OSError as exc:
                raise VideoError(f"Nie udało się uruchomić procesu generowania wideo: {exc}",
                                 "spawn_failed", 500) from exc
            _register_group(proc.pid, f"video:{label}")
            try:
                result, error = await self._pump(proc, payload, on_progress)
            finally:
                await _terminate_process_group(proc, _STOP_GRACE_S)
                _unregister_group(proc.pid)
        finally:
            log_file.close()
        if error is not None:
            raise error
        if result is None:
            tail = _log_tail(log_path)
            code = proc.returncode
            raise VideoError(
                "Proces generowania wideo zakończył się bez wyniku"
                f"{f' (kod {code})' if code else ''}."
                + (f"\n\nOstatnie linie logu ({log_path}):\n{tail}" if tail else ""),
                "no_result", 500)
        return result

    @staticmethod
    async def _pump(proc: asyncio.subprocess.Process, payload: dict[str, Any],
                    on_progress: Callable[[VideoProgress], Any] | None,
                    ) -> tuple[dict[str, Any] | None, VideoError | None]:
        """Write the payload, relay progress, collect the final result/error line."""
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
        result: dict[str, Any] | None = None
        error: VideoError | None = None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except ValueError:
                log.debug("video worker stdout (not JSON): %s", text)
                continue
            kind = obj.get("type")
            if kind == "progress":
                await _call(on_progress, VideoProgress(
                    stage=str(obj.get("stage") or ""), step=int(obj.get("step") or 0),
                    steps=int(obj.get("steps") or 0), elapsed_ms=int(obj.get("elapsedMs") or 0),
                    message=str(obj.get("message") or "")))
            elif kind == "result":
                result = obj
            elif kind == "error":
                error = VideoError(str(obj.get("message") or "Błąd generowania wideo."),
                                   str(obj.get("code") or "generate_failed"), 500)
        await proc.wait()
        return result, error


def _unsupported_message(model: Any, kind: str) -> str:
    name = str(getattr(model, "display_name", None) or getattr(model, "key", "") or "?")
    if kind == "ltx-other":
        return (f"Model „{name}” to inny wariant LTX (nie podstawowy LTX-Video 2B) — silnik "
               "LocalCode obsługuje na razie tylko LTX-Video 2B (np. city96/LTX-Video-gguf).")
    if kind == "other-video":
        return (f"Model „{name}” to inny model wideo (nie LTX-Video) — silnik LocalCode "
               "obsługuje na razie tylko LTX-Video 2B.")
    return (f"Model „{name}” nie wygląda na obsługiwany model wideo (rozpoznajemy na razie "
           "tylko LTX-Video 2B).")


def _log_tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


async def _call(callback: Callable[[VideoProgress], Any] | None, progress: VideoProgress) -> None:
    if callback is None:
        return
    try:
        result = callback(progress)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - a broken progress callback must not kill the job
        log.warning("video progress callback failed", exc_info=True)
