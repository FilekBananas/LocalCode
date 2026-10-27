"""Download worker: one subprocess per download job (``python -m localcode.hub.download_worker``).

Why a subprocess: cancelling must really stop the transfer.  hf_xet runs its
transfer in native threads that a Python thread cannot interrupt, so the
``DownloadManager`` runs every job in its own process and terminates it
(SIGTERM, then SIGKILL) on cancel.

Protocol
--------
stdin — one JSON object::

    {"repo": "org/name", "dest": "/abs/models/org/name",
     "files": [{"name": "x.gguf", "size": 123, "sha256": "…"|null}, …],
     "token": "hf_…"|null, "engine": "http"|"hf"}

stdout — JSON lines (at most ~10 progress lines per second)::

    {"event": "start", "done": <bytes already on disk>, "total": <bytes>}
    {"event": "progress", "file": "x.gguf", "done": <job bytes>, "total": <bytes>}
    {"event": "verify", "file": "x.gguf"}
    {"event": "file_done", "file": "x.gguf"}
    {"event": "done"}
    {"event": "error", "code": "gated"|"not_found"|"disk_full"|"network"|…, "message": "<Polish>"}

Engines
-------
* ``http`` (default) — our own resumable downloader: ``GET resolve/main/<file>``
  with ``Range: bytes=<have>-`` into ``<dest>/.cache/localcode/<file>.part``,
  retries with back-off, SHA-256 verification against the LFS hash, then an
  atomic rename.  Verified with huggingface_hub 1.32: the resolve endpoint
  redirects (also for Xet-backed files) to a CDN that honours Range requests.
  huggingface_hub 1.32's own ``hf_hub_download`` writes to a process-unique
  ``*.incomplete`` temp file that is deleted on failure (and hf_xet keeps no chunk
  cache by default), so it cannot resume an interrupted multi-GB file.
* ``hf`` — ``hf_hub_download(local_dir=dest, tqdm_class=…)``; progress comes from
  the tqdm hook.  Measured with huggingface_hub 1.32 + hf_xet: a Xet-backed file
  goes through ``xet_get``, which builds one ``XetDownloadProgressReporter``
  around this hook (it routes both bars into a single instance because the class
  defines ``update_transfer``).  ``update()`` then reports *reconstructed* bytes
  in a few large jumps — once per flushed block, e.g. twice for a 146 MB file —
  while ``update_transfer()`` reports *network* bytes ~10×/s; the latter is also
  slightly lower than the file size because of dedup/compression.  Hence
  ``max(n, transferred)``, clamped to ``total``.  Used when
  ``LOCALCODE_DOWNLOAD_ENGINE=hf`` or as a fallback for unexpected HTTP answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

CHUNK = 1024 * 1024
MAX_RETRIES = 6
PROGRESS_INTERVAL_S = 0.1
PART_DIR = Path(".cache") / "localcode"


class WorkerError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class _Retryable(Exception):
    pass


class _Unexpected(Exception):
    """The server answered in a way the HTTP engine does not handle (fallback to hf)."""


def emit(event: dict[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, ValueError):
        raise SystemExit(141) from None


def part_path(dest: Path, name: str) -> Path:
    return dest / PART_DIR / (name + ".part")


class Progress:
    """Aggregates per-file progress into job-level bytes and throttles output."""

    def __init__(self, total: int, completed: int) -> None:
        self.total = total
        self.completed = completed  # bytes of files already finished
        self.current = 0
        self.file = ""
        self._last = 0.0
        self._ppid = os.getppid()

    def update(self, file: str, current: int, *, force: bool = False) -> None:
        if os.getppid() != self._ppid:  # parent died: stop instead of downloading as an orphan
            raise SystemExit(0)
        self.file = file
        self.current = current
        now = time.monotonic()
        if force or now - self._last >= PROGRESS_INTERVAL_S:
            self._last = now
            emit({"event": "progress", "file": file, "done": self.completed + current, "total": self.total})

    def finish_file(self, file: str, size: int) -> None:
        self.completed += size
        self.current = 0
        emit({"event": "progress", "file": file, "done": self.completed, "total": self.total})
        emit({"event": "file_done", "file": file})


def _hash_file(path: Path, upto: int | None = None) -> "hashlib._Hash":
    h = hashlib.sha256()
    remaining = upto
    with open(path, "rb") as fh:
        while True:
            n = CHUNK * 8 if remaining is None else min(CHUNK * 8, remaining)
            if n == 0:
                break
            block = fh.read(n)
            if not block:
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
    return h


def download_http(repo: str, dest: Path, spec: dict[str, Any], token: str | None, progress: Progress) -> None:
    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    from localcode.config import VERSION

    name: str = spec["name"]
    size: int | None = spec.get("size")
    sha: str | None = spec.get("sha256")
    final = dest / name
    part = part_path(dest, name)
    part.parent.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo, name)
    session = get_session()
    hasher = hashlib.sha256() if sha else None
    hashed = 0
    attempt = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            part.unlink()
            have = 0
        if hasher is not None and hashed != have:
            hasher = _hash_file(part, have) if have else hashlib.sha256()
            hashed = have
        if size is not None and have == size:
            break
        headers = build_hf_headers(token=token, library_name="localcode", library_version=VERSION)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with session.stream(
                "GET", url, headers=headers, follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=90.0),
            ) as resp:
                if resp.status_code == 416:  # range beyond the end: our part is complete or stale
                    if size is None or have >= size:
                        break
                    part.unlink(missing_ok=True)
                    continue
                if resp.status_code >= 400:
                    resp.read()
                    try:
                        hf_raise_for_status(resp)
                    except HfHubHTTPError as exc:
                        status = resp.status_code
                        if status == 429 or status >= 500:
                            raise _Retryable(str(exc)) from exc
                        raise
                if resp.status_code not in (200, 206):
                    raise _Unexpected(f"HTTP {resp.status_code}")
                mode = "ab"
                if have and resp.status_code == 200:  # server ignored the Range header
                    have = 0
                    mode = "wb"
                    if hasher is not None:
                        hasher = hashlib.sha256()
                        hashed = 0
                if size is None:
                    length = resp.headers.get("content-length")
                    size = have + int(length) if length and length.isdigit() else None
                with open(part, mode) as fh:
                    for chunk in resp.iter_bytes(CHUNK):
                        fh.write(chunk)
                        have += len(chunk)
                        if hasher is not None:
                            hasher.update(chunk)
                            hashed += len(chunk)
                        progress.update(name, have)
            if size is None or have >= size:
                break
            raise _Retryable("connection closed before the end of the file")
        except (_Retryable, httpx.TransportError) as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise WorkerError(
                    "Połączenie z Hugging Face zostało przerwane zbyt wiele razy. Wznów pobieranie później.",
                    "network",
                ) from exc
            time.sleep(min(30.0, 1.5 ** attempt))
            continue
    have = part.stat().st_size
    if size is not None and have != size:
        raise WorkerError(f"Pobrany plik {name} ma nieprawidłowy rozmiar ({have} zamiast {size} bajtów).", "size_mismatch")
    if sha:
        emit({"event": "verify", "file": name})
        digest = (hasher if hasher is not None and hashed == have else _hash_file(part)).hexdigest()
        if digest.lower() != sha.lower():
            part.unlink(missing_ok=True)
            raise WorkerError(
                f"Suma kontrolna pliku {name} się nie zgadza — plik usunięto, spróbuj pobrać go ponownie.",
                "checksum",
            )
    os.replace(part, final)


def download_hf(repo: str, dest: Path, spec: dict[str, Any], token: str | None, progress: Progress) -> None:
    from huggingface_hub import hf_hub_download

    name: str = spec["name"]

    class Hook:
        """Minimal tqdm replacement understood by huggingface_hub (incl. the xet reporter)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.total = kwargs.get("total")
            self.n = int(kwargs.get("initial") or 0)
            self.transferred = 0

        def _report(self) -> None:
            done = max(self.n, self.transferred)
            if self.total:
                done = min(done, int(self.total))
            progress.update(name, done)

        def update(self, n: int | float = 1) -> None:
            self.n += int(n or 0)
            self._report()

        def update_transfer(self, n: int | float) -> None:
            self.transferred += int(n or 0)
            self._report()

        def set_postfix_str(self, *a: Any, **k: Any) -> None: ...
        def set_transfer_postfix_str(self, *a: Any, **k: Any) -> None: ...
        def set_description(self, *a: Any, **k: Any) -> None: ...
        def refresh(self, *a: Any, **k: Any) -> None: ...
        def reset(self, total: int | None = None) -> None:
            self.total = total
        def close(self) -> None: ...
        def __enter__(self) -> "Hook":
            return self
        def __exit__(self, *a: Any) -> None:
            self.close()

        @property
        def format_dict(self) -> dict[str, Any]:
            return {"n": self.n, "total": self.total, "rate": None}

    hf_hub_download(repo, name, local_dir=str(dest), token=token or None, tqdm_class=Hook)  # type: ignore[arg-type]


def _map_error(exc: BaseException, repo: str, token: str | None) -> WorkerError:
    if isinstance(exc, WorkerError):
        return exc
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:  # ENOSPC
        return WorkerError("Zabrakło miejsca na dysku podczas pobierania.", "disk_full")
    from localcode.hub.search import hub_error_from

    err = hub_error_from(exc, repo, token_set=bool(token))
    return WorkerError(err.message, err.code)


def run(spec: dict[str, Any]) -> int:
    repo = spec["repo"]
    dest = Path(spec["dest"])
    token = spec.get("token") or None
    engine = spec.get("engine") or os.environ.get("LOCALCODE_DOWNLOAD_ENGINE") or "http"
    files: list[dict[str, Any]] = spec["files"]
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(int(f.get("size") or 0) for f in files)
    pending: list[dict[str, Any]] = []
    completed = 0
    partial = 0
    for f in files:
        final = dest / f["name"]
        size = f.get("size")
        if final.is_file() and (size is None or final.stat().st_size == size):
            completed += int(size or final.stat().st_size)
            continue
        pending.append(f)
        p = part_path(dest, f["name"])
        if engine == "http" and p.exists():
            partial += p.stat().st_size
    progress = Progress(total, completed)
    emit({"event": "start", "done": completed + partial, "total": total})
    for f in files:
        if f not in pending:
            emit({"event": "file_done", "file": f["name"]})
    for f in pending:
        name = f["name"]
        progress.update(name, 0, force=True)
        try:
            if engine == "hf":
                download_hf(repo, dest, f, token, progress)
            else:
                try:
                    download_http(repo, dest, f, token, progress)
                except _Unexpected:
                    download_hf(repo, dest, f, token, progress)
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            err = _map_error(exc, repo, token)
            emit({"event": "error", "code": err.code, "message": err.message, "file": name})
            return 1
        final = dest / name
        progress.finish_file(name, int(f.get("size") or (final.stat().st_size if final.exists() else 0)))
    try:
        (dest / PART_DIR).rmdir()
    except OSError:
        pass
    emit({"event": "done"})
    return 0


def _install_signal_handlers() -> None:
    def stop(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def main(argv: list[str] | None = None, stdin: Callable[[], str] | None = None) -> int:
    _install_signal_handlers()
    raw = (stdin or sys.stdin.readline)()
    try:
        spec = json.loads(raw)
        if not isinstance(spec, dict) or not spec.get("repo") or not spec.get("dest") or not spec.get("files"):
            raise ValueError("missing fields")
    except ValueError as exc:
        emit({"event": "error", "code": "invalid", "message": f"Nieprawidłowe zlecenie pobierania: {exc}."})
        return 2
    return run(spec)


if __name__ == "__main__":
    raise SystemExit(main())
