"""Download manager: queued, cancellable, resumable model downloads from Hugging Face.

Design
------
Every job runs in its own subprocess — ``python -m localcode.hub.download_worker``
(see that module for the stdin/stdout protocol).  A subprocess is not an
implementation detail but the whole point: ``hf_xet`` and our own HTTP engine
transfer bytes in native threads that a Python thread cannot interrupt, so the
only reliable way to stop a 20 GB download is to kill the process that runs it.
The manager therefore never downloads anything itself; it

* keeps the job list (in memory + ``data_dir()/downloads.json``),
* runs at most :data:`MAX_CONCURRENT` workers and queues the rest (FIFO),
* turns the worker's byte counters into UI-ready progress (speed EMA + ETA)
  and emits ``{"type": "download", "job": {...}}`` at most ~4×/s,
* kills the worker and removes the partial files on ``cancel()``,
* maps every failure to a Polish message (gated repo, 401, 404, network, disk).

Everything runs on the caller's asyncio loop: ``start()``/``cancel()`` are
non-blocking and return immediately, the transfer continues in a background
task that only reads lines from a pipe.  Blocking work (Hub metadata lookup,
disk statistics, deleting partial files) goes through ``asyncio.to_thread``.

States: ``queued`` → ``downloading`` → ``done`` | ``error`` | ``cancelled``.
A restart cannot adopt a running worker (the worker exits as soon as its parent
dies), so jobs found in ``queued``/``downloading`` state when the file is
reloaded become ``error`` — their partial files stay on disk and a new job for
the same files resumes them byte-exactly.

The ``modelKey`` of a job is computed from the files it downloads
(:func:`localcode.hub.library.model_key`), so the UI can select the freshly
downloaded model as soon as ``on_complete`` triggers a library rescan.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from localcode.config import Settings, data_dir
from localcode.hub.download_worker import PART_DIR
from localcode.hub.library import display_name_for, model_key, prune_empty_dirs
from localcode.hub.metadata import gguf_stem, is_mmproj, parse_quant
from localcode.hub.search import HubError, hub_error_from

log = logging.getLogger(__name__)

__all__ = ["DownloadManager", "DownloadError", "JobState", "MAX_CONCURRENT"]

MAX_CONCURRENT = 2                # workers running at the same time
EVENT_INTERVAL_S = 0.25           # ≤ 4 progress events per second and job
PERSIST_INTERVAL_S = 2.0          # downloads.json rewrites while a job is running
SPEED_MIN_SAMPLE_S = 0.25         # window of one speed sample
SPEED_ALPHA = 0.35                # EMA weight of the newest sample
DISK_MARGIN_BYTES = 512 * 1024**2  # keep this much free after the download
TERMINATE_GRACE_S = 3.0           # SIGTERM → SIGKILL delay on cancel
HISTORY_LIMIT = 50                # finished jobs kept in downloads.json
STATE_VERSION = 1

_REPO_RE = re.compile(r"[\w.\-]+/[\w.\-]+")

JobState = str  # "queued" | "downloading" | "done" | "error" | "cancelled"
ACTIVE_STATES = ("queued", "downloading")
TERMINAL_STATES = ("done", "error", "cancelled")


class DownloadError(Exception):
    """Download failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "download_error", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fmt_size(n: float) -> str:
    """Polish size for a user-facing message (comma decimal separator)."""
    if n < 1e9:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e9:.1f}".replace(".", ",") + " GB"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _existing_dir(path: Path) -> Path:
    """Nearest existing ancestor of ``path`` (for ``shutil.disk_usage``)."""
    p = path
    while not p.exists():
        if p.parent == p:
            return p
        p = p.parent
    return p


def _normalise_files(files: Any) -> list[dict[str, Any]]:
    """Accept ``["a.gguf", …]`` / ``[{"name"|"rfilename", "size"|"sizeBytes", "sha256"}, …]``."""
    if files is None:
        return []
    if isinstance(files, (str, bytes)):
        files = [files]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        if isinstance(item, str):
            name, size, sha = item, None, None
        elif isinstance(item, dict):
            name = item.get("name") or item.get("rfilename") or item.get("path")
            size = item.get("size", item.get("sizeBytes"))
            sha = item.get("sha256") or item.get("sha")
        else:
            raise DownloadError("Lista plików do pobrania ma nieprawidłowy format.", "invalid", 400)
        if not isinstance(name, str) or not name.strip():
            raise DownloadError("Lista plików do pobrania zawiera pustą nazwę pliku.", "invalid", 400)
        name = name.strip().lstrip("/")
        if ".." in Path(name).parts or Path(name).is_absolute():
            raise DownloadError(f"Nieprawidłowa nazwa pliku „{name}”.", "invalid", 400)
        if name in seen:
            continue
        seen.add(name)
        try:
            isize = int(size) if size is not None else None
        except (TypeError, ValueError):
            isize = None
        out.append({"name": name, "size": isize if (isize or 0) >= 0 else None,
                    "sha256": sha if isinstance(sha, str) and sha else None})
    return out


def job_label(repo: str, files: Sequence[dict[str, Any]]) -> str:
    """Human label of a job: ``"Qwen3 0.6B · Q4_K_M"`` / the repo name for MLX."""
    name = repo.split("/")[-1]
    models = [f["name"] for f in files if f["name"].lower().endswith(".gguf") and not is_mmproj(f["name"])]
    if models:
        stem = gguf_stem(models[0])
        return display_name_for(stem, parse_quant(models[0]))
    projectors = [f["name"] for f in files if is_mmproj(f["name"])]
    if projectors and len(projectors) == len(files):
        return f"{display_name_for(name, None)} · projektor wizji"
    tail = re.split(r"[-_ ]", name)[-1]
    quant = tail if re.fullmatch(r"(?i)\d+bit|bf16|fp16|f16|fp8|mxfp4|nvfp4|dwq", tail) else None
    return display_name_for(name, quant)


def job_format(files: Sequence[dict[str, Any]]) -> str:
    return "gguf" if any(f["name"].lower().endswith(".gguf") for f in files) else "mlx"


def job_model_key(repo: str, files: Sequence[dict[str, Any]]) -> str | None:
    """Library key the downloaded files will get (``None`` for a projector-only job)."""
    models = [f["name"] for f in files if f["name"].lower().endswith(".gguf") and not is_mmproj(f["name"])]
    if models:
        return model_key(repo, models[0])
    if any(f["name"].lower().endswith(".gguf") for f in files):
        return None  # projector only — it belongs to a model that is downloaded separately
    return model_key(repo)


@dataclass
class _Job:
    """One download job: the DTO fields plus the bookkeeping the manager needs."""

    id: str
    repo: str
    dest: Path
    files: list[dict[str, Any]]
    label: str
    format: str
    model_key: str | None
    total_bytes: int = 0
    done_bytes: int = 0
    speed_bps: float = 0.0
    eta_s: float | None = None
    state: JobState = "queued"
    stage: str = "download"                 # cerber | download | verify
    error: str | None = None
    error_code: str | None = None
    current_file: str | None = None
    created_at: int = field(default_factory=_now_ms)
    started_at: int | None = None
    finished_at: int | None = None
    pid: int | None = None
    completed: list[str] = field(default_factory=list)
    # runtime only (never persisted)
    proc: Any = None
    task: Any = None
    cancel_requested: bool = False
    finishing: bool = False
    done_event: asyncio.Event | None = None
    last_emit: float = 0.0
    sample: tuple[float, int] | None = None

    @property
    def names(self) -> list[str]:
        return [f["name"] for f in self.files]

    def to_dict(self) -> dict[str, Any]:
        """camelCase ``DownloadJob`` DTO (see docs/ARCHITECTURE.md)."""
        total = self.total_bytes
        done = min(self.done_bytes, total) if total else self.done_bytes
        if self.state == "done":
            done = total or done
        return {
            "id": self.id,
            "repo": self.repo,
            "files": self.names,
            "label": self.label,
            "format": self.format,
            "totalBytes": total,
            "doneBytes": done,
            "progress": round(done / total, 4) if total else (1.0 if self.state == "done" else 0.0),
            "speedBps": round(self.speed_bps, 1) if self.state == "downloading" else 0.0,
            "etaSeconds": round(self.eta_s, 1) if (self.state == "downloading" and self.eta_s is not None) else None,
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "errorCode": self.error_code,
            "currentFile": self.current_file,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "modelKey": self.model_key,
            "dest": str(self.dest),
        }

    def record(self) -> dict[str, Any]:
        """Persisted form: the DTO plus what is needed to resume/clean up later."""
        return {"job": self.to_dict(), "pid": self.pid, "fileSpecs": self.files,
                "completed": list(self.completed)}


class DownloadManager:
    """Queued, cancellable downloads of model files into ``settings.models_dir``.

    ``on_event(payload)`` receives ``{"type": "download", "job": DownloadJob}`` on
    every state change and (throttled to ~4×/s) while bytes arrive.
    ``on_complete(job)`` runs once per successfully finished job — the server
    uses it to rescan the library.  Both may be sync or async.
    """

    def __init__(
        self,
        settings: Settings,
        on_event: Callable[[dict[str, Any]], Any] | None = None,
        on_complete: Callable[[dict[str, Any]], Any] | None = None,
        *,
        max_concurrent: int = MAX_CONCURRENT,
        state_path: Path | None = None,
        python_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.on_event = on_event
        self.on_complete = on_complete
        self.max_concurrent = max(1, int(max_concurrent))
        self._state_path = state_path
        self._python = python_path
        self._jobs: dict[str, _Job] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_persist = 0.0
        self._closed = False
        self._cerber: Any = None
        self._load_state()

    @property
    def cerber(self) -> Any:
        """The Cerber check in front of every download (off unless the user turned it on)."""
        if self._cerber is None:
            from localcode.security.cerber import CerberGate

            self._cerber = CerberGate(self.settings)
        return self._cerber

    # ------------------------------------------------------------------ paths
    @property
    def models_dir(self) -> Path:
        return Path(self.settings.models_dir).expanduser()

    @property
    def state_path(self) -> Path:
        return self._state_path or (data_dir() / "downloads.json")

    @property
    def token(self) -> str | None:
        t = self.settings.hf_token
        return t.strip() if isinstance(t, str) and t.strip() else None

    # ------------------------------------------------------------- public API
    def start(self, repo: str, files: Any = None, *, label: str | None = None) -> dict[str, Any]:
        """Queue a download of ``files`` from ``repo``; returns the job DTO.

        ``files`` is a list of repo-relative names or of
        ``{"name", "size"?, "sha256"?}`` dicts (as produced by
        :meth:`localcode.hub.search.HubClient.variant`).  Unknown sizes and
        checksums are looked up on the Hub when the job starts.  Raises
        :class:`DownloadError` for a bad request or too little free disk space.
        """
        if self._closed:
            raise DownloadError("Menedżer pobierania został zamknięty.", "closed", 409)
        loop = self._loop()
        repo = (repo or "").strip().strip("/")
        if not _REPO_RE.fullmatch(repo):
            raise DownloadError(
                f"Nieprawidłowy identyfikator repozytorium „{repo}” (oczekiwano organizacja/nazwa).", "invalid", 400
            )
        specs = _normalise_files(files)
        if not specs:
            raise DownloadError("Nie wskazano plików do pobrania.", "invalid", 400)
        existing = self._find_active(repo, specs)
        if existing is not None:
            return existing.to_dict()
        job = _Job(
            id=f"dl-{uuid.uuid4().hex[:12]}",
            repo=repo,
            dest=self.models_dir / repo,
            files=specs,
            label=label or job_label(repo, specs),
            format=job_format(specs),
            model_key=job_model_key(repo, specs),
            total_bytes=sum(int(f["size"] or 0) for f in specs) if all(f["size"] is not None for f in specs) else 0,
            done_event=asyncio.Event(),
        )
        if job.total_bytes:
            self._check_disk(job)          # fail fast; re-checked once sizes are known
        self._jobs[job.id] = job
        self._trim_history()
        self._persist(force=True)
        self._emit(job)
        self._schedule(loop)
        return job.to_dict()

    def list(self) -> list[dict[str, Any]]:
        """All known jobs, oldest first (queue order)."""
        return [j.to_dict() for j in self._jobs.values()]

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Stop a job: kill its worker and delete its partial files.

        Returns immediately; the job reaches ``cancelled`` once the worker is
        gone (use :meth:`acancel` to await that).  Finished jobs are returned
        unchanged.
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise DownloadError(f"Nie znaleziono pobierania „{job_id}”.", "not_found", 404)
        if job.state in TERMINAL_STATES:
            return job.to_dict()
        job.cancel_requested = True
        if job.state == "queued" or job.task is None:
            if job.task is not None:
                job.task.cancel()      # it may not have started running yet
            self._spawn(self._finish_cancelled(job))
            return job.to_dict()
        proc = job.proc
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError) as exc:
                log.debug("cannot terminate worker of %s: %s", job.id, exc)
            self._spawn(self._force_kill_later(job))
        elif job.task is not None:
            job.task.cancel()          # still resolving metadata / checking disk
        return job.to_dict()

    async def acancel(self, job_id: str, *, timeout: float = 15.0) -> dict[str, Any]:
        """:meth:`cancel` + wait until the job is really in a terminal state."""
        dto = self.cancel(job_id)
        if dto["state"] in TERMINAL_STATES:
            return dto
        return await self.wait(job_id, timeout=timeout)

    async def wait(self, job_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Wait until the job finishes (done/error/cancelled) and return its DTO."""
        job = self._jobs.get(job_id)
        if job is None:
            raise DownloadError(f"Nie znaleziono pobierania „{job_id}”.", "not_found", 404)
        if job.state in TERMINAL_STATES:
            return job.to_dict()
        assert job.done_event is not None
        await asyncio.wait_for(job.done_event.wait(), timeout)
        return job.to_dict()

    def clear_finished(self) -> list[dict[str, Any]]:
        """Forget finished jobs (the UI's "wyczyść listę")."""
        for jid in [j.id for j in self._jobs.values() if j.state in TERMINAL_STATES]:
            self._jobs.pop(jid, None)
        self._persist(force=True)
        return self.list()

    async def shutdown(self) -> None:
        """Stop every worker without deleting partial files (they resume later)."""
        self._closed = True
        running = [j for j in self._jobs.values() if j.state in ACTIVE_STATES]
        for job in running:
            job.state = "error"
            job.error = "Pobieranie przerwano przy zamykaniu aplikacji. Uruchom je ponownie, aby wznowić."
            job.error_code = "interrupted"
            job.finished_at = _now_ms()
            proc = job.proc
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except (ProcessLookupError, OSError):
                    pass
            if job.done_event is not None:
                job.done_event.set()
        tasks = [j.task for j in running if j.task is not None] + list(self._tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job in running:
            proc = job.proc
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), 3.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
        self._tasks.clear()
        self._persist(force=True)

    # ------------------------------------------------------------- scheduling
    @staticmethod
    def _loop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError as exc:  # pragma: no cover - programming error
            raise DownloadError(
                "Menedżer pobierania wymaga działającej pętli asyncio.", "no_loop", 500
            ) from exc

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = self._loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _schedule(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start queued jobs while a worker slot is free (FIFO)."""
        if self._closed:
            return
        loop = loop or self._loop()
        # a job counts as running from the moment its task exists — the task only
        # switches the state to "downloading" on the next loop iteration.
        running = sum(1 for j in self._jobs.values() if j.task is not None)
        for job in self._jobs.values():
            if running >= self.max_concurrent:
                break
            if job.state != "queued" or job.task is not None or job.cancel_requested:
                continue
            running += 1
            job.task = loop.create_task(self._run(job))

    def _find_active(self, repo: str, specs: list[dict[str, Any]]) -> _Job | None:
        wanted = {f["name"] for f in specs}
        for job in self._jobs.values():
            if job.state in ACTIVE_STATES and job.repo == repo and wanted & set(job.names):
                return job
        return None

    # ------------------------------------------------------------- job runner
    async def _run(self, job: _Job) -> None:
        if job.state in TERMINAL_STATES:      # cancelled before the task started running
            job.task = None
            return
        try:
            if job.cancel_requested:
                raise asyncio.CancelledError
            job.state = "downloading"
            job.started_at = _now_ms()
            job.sample = (time.monotonic(), job.done_bytes)
            self._emit(job, force=True)
            self._persist(force=True)
            await self._cerber_check(job)
            await self._resolve_files(job)
            if job.cancel_requested:
                raise asyncio.CancelledError
            await asyncio.to_thread(self._check_disk, job)
            await self._transfer(job)
        except asyncio.CancelledError:
            await self._finish_cancelled(job)
            raise
        except DownloadError as exc:
            self._fail(job, exc.message, exc.code)
        except HubError as exc:  # pragma: no cover - mapped below in practice
            self._fail(job, exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001 - never let a job hang
            log.exception("download job %s failed", job.id)
            self._fail(job, f"Nieoczekiwany błąd pobierania: {exc}.", "internal")
        finally:
            job.task = None
            job.proc = None
            if job.state in TERMINAL_STATES and job.done_event is not None:
                job.done_event.set()
            self._persist(force=True)
            self._schedule()

    async def _cerber_check(self, job: _Job) -> None:
        """Ask Cerber about the repository before a single byte of it is fetched."""
        gate = self.cerber
        if not gate.enabled:
            return
        from localcode.security.cerber import CerberError

        job.stage = "cerber"
        self._emit(job, force=True)
        try:
            await gate.check(job.repo, cancelled=lambda: job.cancel_requested)
        except CerberError as exc:
            raise DownloadError(exc.message, exc.code, exc.status) from exc
        if job.cancel_requested:
            raise asyncio.CancelledError
        job.stage = "download"
        self._emit(job, force=True)

    async def _resolve_files(self, job: _Job) -> None:
        """Fill in missing sizes/checksums from ``model_info(files_metadata=True)``."""
        if all(f["size"] is not None and f["sha256"] for f in job.files):
            return
        sizes_known = all(f["size"] is not None for f in job.files)
        try:
            meta = await asyncio.to_thread(self._file_metadata, job.repo)
        except DownloadError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to Polish below
            err = hub_error_from(exc, job.repo, token_set=self.token is not None)
            if sizes_known:
                log.debug("file metadata of %s unavailable (%s); downloading without checksums", job.repo, exc)
                return
            raise DownloadError(err.message, err.code, err.status) from exc
        missing = [f["name"] for f in job.files if f["name"] not in meta]
        if missing:
            raise DownloadError(
                f"Repozytorium „{job.repo}” nie zawiera pliku {missing[0]}.", "file_not_found", 404
            )
        for f in job.files:
            size, sha = meta[f["name"]]
            if size is not None:
                f["size"] = int(size)
            if sha:
                f["sha256"] = sha
        total = sum(int(f["size"] or 0) for f in job.files)
        if total:
            job.total_bytes = total

    def _file_metadata(self, repo: str) -> dict[str, tuple[int | None, str | None]]:
        from huggingface_hub import HfApi

        api = HfApi(token=self.token, library_name="localcode")
        info = api.model_info(repo, files_metadata=True)
        out: dict[str, tuple[int | None, str | None]] = {}
        for sibling in getattr(info, "siblings", None) or []:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(sibling, "size", None)
            if size is None and lfs is not None:
                size = getattr(lfs, "size", None)
            out[sibling.rfilename] = (int(size) if size is not None else None,
                                      getattr(lfs, "sha256", None) if lfs is not None else None)
        return out

    def _check_disk(self, job: _Job) -> None:
        """Refuse to start when the remaining bytes would not fit on the disk."""
        needed = 0
        for f in job.files:
            size = int(f["size"] or 0)
            if not size:
                continue
            final = job.dest / f["name"]
            try:
                have = final.stat().st_size if final.is_file() else 0
            except OSError:
                have = 0
            if have != size:
                part = job.dest / PART_DIR / (f["name"] + ".part")
                try:
                    have = part.stat().st_size if part.is_file() else 0
                except OSError:
                    have = 0
                needed += max(0, size - have)
        if not needed:
            return
        try:
            usage = shutil.disk_usage(_existing_dir(job.dest))
        except OSError as exc:  # pragma: no cover - unreadable mount point
            log.debug("disk usage of %s unavailable: %s", job.dest, exc)
            return
        if usage.free < needed + DISK_MARGIN_BYTES:
            raise DownloadError(
                f"Za mało miejsca na dysku, aby pobrać „{job.label}”: potrzeba ok. {_fmt_size(needed)}, "
                f"a wolne jest {_fmt_size(usage.free)}.",
                "disk_full", 507,
            )

    # -------------------------------------------------------------- transfers
    def _worker_command(self) -> list[str]:
        python = self._python or self.settings.runtime.python_path or sys.executable
        return [python, "-m", "localcode.hub.download_worker"]

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        root = str(Path(__file__).resolve().parents[2])   # repo root holding `localcode/`
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        return env

    async def _transfer(self, job: _Job) -> None:
        job.dest.mkdir(parents=True, exist_ok=True)
        spec = {
            "repo": job.repo,
            "dest": str(job.dest),
            "files": job.files,
            "token": self.token,
            "engine": os.environ.get("LOCALCODE_DOWNLOAD_ENGINE") or "http",
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._worker_command(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._worker_env(),
                cwd=str(job.dest),
            )
        except OSError as exc:
            raise DownloadError(
                f"Nie udało się uruchomić procesu pobierania: {exc.strerror or exc}.", "spawn_failed", 500
            ) from exc
        job.proc = proc
        job.pid = proc.pid
        errors: list[str] = []
        stderr_task = self._spawn(self._drain_stderr(proc, errors))
        try:
            assert proc.stdin is not None
            proc.stdin.write((json.dumps(spec, ensure_ascii=False) + "\n").encode())
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            log.debug("worker of %s closed stdin early: %s", job.id, exc)
        saw_done = False
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    log.debug("worker of %s wrote a non-JSON line: %r", job.id, line[:200])
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "done":
                    saw_done = True
                self._apply_worker_event(job, event)
        finally:
            stderr_task.cancel()
            try:
                rc = await asyncio.wait_for(proc.wait(), 10.0)
            except asyncio.TimeoutError:  # pragma: no cover - a worker that ignores SIGTERM
                log.warning("download worker %s did not exit, killing", proc.pid)
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                rc = await proc.wait()
        if job.cancel_requested:
            raise asyncio.CancelledError
        if job.error:
            self._fail(job, job.error, job.error_code or "download_error")
            return
        if saw_done and rc == 0:
            self._succeed(job)
            return
        tail = " ".join(errors[-3:])[:400]
        detail = f" Szczegóły: {tail}" if tail else ""
        self._fail(
            job,
            f"Proces pobierania zakończył się błędem (kod {rc}).{detail}",
            "worker_failed",
        )

    @staticmethod
    async def _drain_stderr(proc: Any, sink: list[str]) -> None:
        if proc.stderr is None:  # pragma: no cover - always a pipe here
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    sink.append(text)
                    del sink[:-20]
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            log.debug("cannot read worker stderr: %s", exc)

    def _apply_worker_event(self, job: _Job, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind in ("start", "progress"):
            total = event.get("total")
            if isinstance(total, int) and total > 0:
                job.total_bytes = max(job.total_bytes, total)
            done = event.get("done")
            if isinstance(done, int) and done >= 0:
                if done >= job.done_bytes or kind == "start":
                    job.done_bytes = done
                self._sample_speed(job, done)
            if isinstance(event.get("file"), str):
                job.current_file = event["file"]
            job.stage = "download"
            self._emit(job)
            self._persist()
        elif kind == "verify":
            job.stage = "verify"
            job.current_file = event.get("file") or job.current_file
            self._emit(job, force=True)
        elif kind == "file_done":
            name = event.get("file")
            if isinstance(name, str) and name not in job.completed:
                job.completed.append(name)
            job.stage = "download"
            self._emit(job, force=True)
            self._persist(force=True)
        elif kind == "error":
            job.error = event.get("message") or "Pobieranie nie powiodło się."
            job.error_code = event.get("code") or "download_error"
        elif kind == "done":
            job.done_bytes = job.total_bytes or job.done_bytes

    def _sample_speed(self, job: _Job, done: int) -> None:
        now = time.monotonic()
        if job.sample is None:
            job.sample = (now, done)
            return
        last_t, last_b = job.sample
        dt = now - last_t
        if dt < SPEED_MIN_SAMPLE_S:
            return
        instant = max(0.0, float(done - last_b)) / dt
        job.sample = (now, done)
        job.speed_bps = instant if job.speed_bps <= 0 else SPEED_ALPHA * instant + (1 - SPEED_ALPHA) * job.speed_bps
        remaining = max(0, job.total_bytes - done) if job.total_bytes else 0
        job.eta_s = remaining / job.speed_bps if (remaining and job.speed_bps > 1024) else None

    # ------------------------------------------------------------ transitions
    def _succeed(self, job: _Job) -> None:
        job.state = "done"
        job.stage = "download"
        job.done_bytes = job.total_bytes or job.done_bytes
        job.speed_bps = 0.0
        job.eta_s = None
        job.current_file = None
        job.finished_at = _now_ms()
        job.error = job.error_code = None
        job.model_key = job.model_key or job_model_key(job.repo, job.files)
        self._emit(job, force=True)
        self._notify_complete(job)

    def _fail(self, job: _Job, message: str, code: str) -> None:
        if job.state in TERMINAL_STATES:
            return
        job.state = "error"
        job.error = message
        job.error_code = code
        job.speed_bps = 0.0
        job.eta_s = None
        job.finished_at = _now_ms()
        self._emit(job, force=True)

    async def _finish_cancelled(self, job: _Job) -> None:
        if job.state in TERMINAL_STATES or job.finishing:
            return
        job.finishing = True
        proc = job.proc
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except (asyncio.TimeoutError, ProcessLookupError):  # pragma: no cover - kill always works
                log.warning("worker %s did not exit after SIGKILL", job.pid)
        job.proc = None
        try:
            await asyncio.to_thread(self._cleanup_partials, job)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the cancel
            log.warning("cannot remove partial files of %s: %s", job.id, exc)
        job.state = "cancelled"
        job.stage = "download"
        job.speed_bps = 0.0
        job.eta_s = None
        job.current_file = None
        job.finished_at = _now_ms()
        job.error = "Pobieranie anulowane."
        job.error_code = "cancelled"
        if job.done_event is not None:
            job.done_event.set()
        self._emit(job, force=True)
        self._persist(force=True)
        self._schedule()

    async def _force_kill_later(self, job: _Job) -> None:
        """SIGKILL a worker that ignored SIGTERM; the runner does the cleanup."""
        try:
            await asyncio.sleep(TERMINATE_GRACE_S)
        except asyncio.CancelledError:
            return
        proc = job.proc
        if proc is not None and proc.returncode is None:
            log.warning("download worker %s ignored SIGTERM, killing", job.pid)
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass

    def _cleanup_partials(self, job: _Job) -> None:
        """Delete everything this job left half-written (keeps finished files)."""
        dest = job.dest
        for f in job.files:
            name = f["name"]
            size = f["size"]
            for candidate in (
                dest / PART_DIR / (name + ".part"),
                dest / ".cache" / "huggingface" / "download" / (name + ".incomplete"),
                dest / ".cache" / "huggingface" / "download" / (name + ".lock"),
                dest / ".cache" / "huggingface" / "download" / (name + ".metadata"),
            ):
                try:
                    candidate.unlink()
                except OSError:
                    pass
            final = dest / name
            try:
                if final.is_file() and size is not None and final.stat().st_size != int(size):
                    final.unlink()
            except OSError as exc:
                log.debug("cannot remove incomplete file %s: %s", final, exc)
        try:
            prune_empty_dirs(dest, self.models_dir)
        except OSError as exc:  # pragma: no cover - best effort
            log.debug("cannot prune %s: %s", dest, exc)

    # ---------------------------------------------------------------- hooks
    def _emit(self, job: _Job, *, force: bool = False) -> None:
        if self.on_event is None:
            return
        now = time.monotonic()
        if not force and now - job.last_emit < EVENT_INTERVAL_S:
            return
        job.last_emit = now
        payload = {"type": "download", "job": job.to_dict()}
        try:
            result = self.on_event(payload)
        except Exception:  # noqa: BLE001 - a broken listener must not stop a download
            log.exception("download event listener failed")
            return
        if inspect.isawaitable(result):
            self._spawn(_await(result))

    def _notify_complete(self, job: _Job) -> None:
        if self.on_complete is None:
            return
        try:
            result = self.on_complete(job.to_dict())
        except Exception:  # noqa: BLE001 - see above
            log.exception("download completion hook failed")
            return
        if inspect.isawaitable(result):
            self._spawn(_await(result))

    # ----------------------------------------------------------- persistence
    def _trim_history(self) -> None:
        finished = [j for j in self._jobs.values() if j.state in TERMINAL_STATES]
        for job in finished[: max(0, len(finished) - HISTORY_LIMIT)]:
            self._jobs.pop(job.id, None)

    def _persist(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_persist < PERSIST_INTERVAL_S:
            return
        self._last_persist = now
        data = {"version": STATE_VERSION, "jobs": [j.record() for j in self._jobs.values()]}
        try:
            _atomic_write_json(self.state_path, data)
        except OSError as exc:
            log.warning("cannot write %s: %s", self.state_path, exc)

    def _load_state(self) -> None:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
            return
        for record in raw.get("jobs") or []:
            job = self._job_from_record(record)
            if job is not None:
                self._jobs[job.id] = job
        self._trim_history()

    def _job_from_record(self, record: Any) -> _Job | None:
        if not isinstance(record, dict) or not isinstance(record.get("job"), dict):
            return None
        dto = record["job"]
        try:
            specs = _normalise_files(record.get("fileSpecs") or dto.get("files"))
            job = _Job(
                id=str(dto["id"]),
                repo=str(dto["repo"]),
                dest=Path(dto.get("dest") or (self.models_dir / str(dto["repo"]))),
                files=specs,
                label=str(dto.get("label") or job_label(str(dto["repo"]), specs)),
                format=str(dto.get("format") or job_format(specs)),
                model_key=dto.get("modelKey"),
                total_bytes=int(dto.get("totalBytes") or 0),
                done_bytes=int(dto.get("doneBytes") or 0),
                state=str(dto.get("state") or "error"),
                error=dto.get("error"),
                error_code=dto.get("errorCode"),
                created_at=int(dto.get("createdAt") or _now_ms()),
                started_at=dto.get("startedAt"),
                finished_at=dto.get("finishedAt"),
                pid=record.get("pid"),
                completed=[str(n) for n in (record.get("completed") or [])],
                done_event=asyncio.Event(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("ignoring unreadable download record: %s", exc)
            return None
        if job.state in ACTIVE_STATES:
            self._kill_orphan(job)
            job.state = "error"
            job.error = (
                "Pobieranie zostało przerwane (LocalCode został zamknięty). "
                "Uruchom je ponownie — pobieranie wznowi się od miejsca przerwania."
            )
            job.error_code = "interrupted"
            job.finished_at = job.finished_at or _now_ms()
        if job.done_event is not None:
            job.done_event.set()          # every reloaded job is in a terminal state
        return job

    @staticmethod
    def _kill_orphan(job: _Job) -> None:
        """Kill a worker that somehow outlived the app (it normally exits by itself)."""
        pid = job.pid
        if not pid:
            return
        try:
            import psutil

            proc = psutil.Process(int(pid))
            if "localcode.hub.download_worker" not in " ".join(proc.cmdline()):
                return
            proc.kill()
            log.info("killed orphaned download worker %s", pid)
        except Exception as exc:  # noqa: BLE001 - psutil raises many process errors
            log.debug("no orphaned worker %s to kill: %s", pid, exc)


async def _await(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception:  # noqa: BLE001 - listener errors are logged, never fatal
        log.exception("download listener coroutine failed")
