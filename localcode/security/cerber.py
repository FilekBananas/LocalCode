"""Cerber — the user's antivirus for macOS — as a gate in front of every download.

Cerber.app exposes everything it can do on a local HTTP API (Integracje → API
(localhost), off by default, alive only while the app is open):

* loopback only, ``Authorization: Bearer <token>`` on every ``/api/*`` request,
  the Host header must be ``127.0.0.1:PORT`` or ``localhost:PORT``, no CORS;
* ``POST /api/huggingface {"url": "org/model"}`` queues a scan of a Hugging Face
  repository — code and configs whole, pickle weights in full or their
  executable part, safetensors/GGUF headers, chat templates — inside Cerber's
  own sandbox, and answers ``202`` with a ``job``;
* ``GET /api/jobs/{id}`` reports the job: ``state`` (queued / running / done /
  failed / cancelled), and once done a ``verdict``: CLEAN, INFO, SUSPICIOUS,
  PUA or MALICIOUS (the worst of everything it found).

:class:`CerberGate` asks before a download starts and refuses the ones whose
verdict is in ``settings.cerber.block``. It fails *closed*: a person who turned
the check on and then closed Cerber gets a clear "Cerber nie odpowiada", not a
download that quietly skipped the check. A clean verdict is remembered for a
day per repository, so the second quantisation of the same model does not wait
for a second scan.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

__all__ = [
    "CerberClient",
    "CerberError",
    "CerberGate",
    "CerberVerdict",
    "cerber_installed",
    "check_connection",
    "VERDICT_LABELS",
]

#: Polish names of Cerber's verdicts.
VERDICT_LABELS = {
    "CLEAN": "czysto",
    "INFO": "drobne uwagi",
    "SUSPICIOUS": "podejrzany",
    "PUA": "potencjalnie niechciany",
    "MALICIOUS": "złośliwy",
}
TERMINAL = ("done", "failed", "cancelled")
POLL_S = 2.0
CLEAN_TTL_S = 24 * 3600
APP_PATHS = (Path("/Applications/Cerber.app"), Path.home() / "Applications" / "Cerber.app")


class CerberError(Exception):
    """A Polish message for the person, and a code for the client."""

    def __init__(self, message: str, code: str = "cerber_error", status: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass
class CerberVerdict:
    repo: str
    verdict: str
    job_id: str = ""
    findings: int = 0
    flagged: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return VERDICT_LABELS.get(self.verdict, self.verdict.lower())


def cerber_installed() -> bool:
    """Whether Cerber.app is on this Mac (the settings panel says so either way)."""
    return any(path.exists() for path in APP_PATHS)


def _unavailable(url: str) -> CerberError:
    return CerberError(
        f"Cerber nie odpowiada pod {url}. Otwórz aplikację Cerber i włącz Integracje → API "
        "(localhost) — albo wyłącz sprawdzanie Cerberem w Ustawieniach LocalCode.",
        "cerber_unavailable", 503)


class CerberClient:
    """HTTP to one Cerber API. ``session_factory`` is injected by the tests."""

    def __init__(self, url: str, token: str | None, *, session_factory: Any = None) -> None:
        self.url = (url or "").rstrip("/")
        self.token = (token or "").strip()
        self._session_factory = session_factory

    def _session(self) -> Any:
        if self._session_factory is not None:
            return self._session_factory()
        import aiohttp

        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                "Accept": "application/json"}

    async def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        if not self.token:
            raise CerberError("Brak tokenu Cerbera — wklej go w Ustawieniach LocalCode "
                              "(token pokazuje Cerber w Integracje → API).", "cerber_no_token", 400)
        try:
            async with self._session() as session:
                call = session.post if method == "POST" else session.get
                kwargs: dict[str, Any] = {"headers": self._headers()}
                if body is not None:
                    kwargs["data"] = json.dumps(body)
                async with call(f"{self.url}{path}", **kwargs) as response:
                    status = response.status
                    text = await response.text()
        except CerberError:
            raise
        except (OSError, asyncio.TimeoutError) as exc:
            raise _unavailable(self.url) from exc
        except Exception as exc:  # noqa: BLE001 - aiohttp's client errors
            if type(exc).__name__.startswith("Client"):
                raise _unavailable(self.url) from exc
            raise
        try:
            data = json.loads(text or "null")
        except ValueError:
            data = None
        if status == 401:
            raise CerberError("Cerber odrzucił token — skopiuj aktualny z Integracje → API i wklej "
                              "go w Ustawieniach LocalCode.", "cerber_bad_token", 400)
        if status >= 400:
            detail = data.get("error") if isinstance(data, dict) else None
            raise CerberError(f"Cerber odpowiedział błędem ({status})" + (f": {detail}" if detail else "."),
                              "cerber_error", 502)
        return data

    async def info(self) -> dict[str, Any]:
        data = await self._request("GET", "/api/info")
        return data if isinstance(data, dict) else {}

    async def scan_huggingface(self, repo: str, *, cancelled: Callable[[], bool] | None = None,
                               timeout_s: float = 900,
                               sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> CerberVerdict:
        """Queue a scan of ``repo`` and wait for its verdict."""
        started = await self._request("POST", "/api/huggingface", {"url": repo})
        job = (started or {}).get("job") or {}
        job_id = str(job.get("id") or "")
        if not job_id:
            raise CerberError("Cerber nie zwrócił zadania skanowania.", "cerber_error", 502)
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        while True:
            if cancelled is not None and cancelled():
                await self._cancel(job_id)
                raise asyncio.CancelledError
            if job.get("state") in TERMINAL:
                break
            if time.monotonic() > deadline:
                await self._cancel(job_id)
                raise CerberError(f"Cerber sprawdzał „{repo}” dłużej niż {int(timeout_s // 60)} min — "
                                  "pobieranie przerwane. Spróbuj ponownie, gdy Cerber skończy inne skany.",
                                  "cerber_timeout", 504)
            await sleep(POLL_S)
            job = ((await self._request("GET", f"/api/jobs/{job_id}")) or {}).get("job") or {}
        state = job.get("state")
        if state != "done":
            detail = job.get("error") or ("anulowane w Cerberze" if state == "cancelled" else "nieudane")
            raise CerberError(f"Cerber nie dokończył sprawdzania „{repo}”: {detail}.", "cerber_failed", 502)
        summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
        # Per-file records live in ``summary.results`` (``public_result`` in
        # Cerber's gui_backend); only the ones worth naming go in the message.
        flagged = [str(f.get("path")) for f in (summary.get("results") or [])
                   if isinstance(f, dict) and f.get("path")
                   and str(f.get("verdict") or "").upper() not in ("CLEAN", "INFO")]
        return CerberVerdict(repo=repo, verdict=str(job.get("verdict") or summary.get("verdict") or "CLEAN"),
                             job_id=job_id, findings=int(job.get("findings") or 0), flagged=flagged[:5])

    async def _cancel(self, job_id: str) -> None:
        try:
            await self._request("POST", f"/api/jobs/{job_id}/cancel", {})
        except Exception:  # noqa: BLE001 - best effort; the download stops either way
            log.debug("could not cancel Cerber job %s", job_id, exc_info=True)


class CerberGate:
    """The check :class:`~localcode.hub.download.DownloadManager` runs before a download."""

    def __init__(self, settings: Any, *, client_factory: Callable[[str, str | None], CerberClient] | None = None
                 ) -> None:
        self.settings = settings
        self._client_factory = client_factory or (lambda url, token: CerberClient(url, token))
        self._clean: dict[str, float] = {}

    @property
    def config(self) -> Any:
        return getattr(self.settings, "cerber", None)

    @property
    def enabled(self) -> bool:
        return bool(self.config is not None and self.config.enabled)

    async def check(self, repo: str, *, cancelled: Callable[[], bool] | None = None) -> CerberVerdict | None:
        """``None`` when the check is off; a passing verdict; or :class:`CerberError`."""
        if not self.enabled:
            return None
        cfg = self.config
        seen = self._clean.get(repo)
        if seen is not None and time.monotonic() - seen < CLEAN_TTL_S:
            return CerberVerdict(repo=repo, verdict="CLEAN")
        client = self._client_factory(cfg.url, cfg.token)
        verdict = await client.scan_huggingface(repo, cancelled=cancelled, timeout_s=cfg.timeout_s)
        blocked = {str(v).upper() for v in (cfg.block or [])}
        if verdict.verdict.upper() in blocked:
            where = f" ({', '.join(verdict.flagged)})" if verdict.flagged else ""
            raise CerberError(
                f"Cerber zablokował pobranie „{repo}”: {verdict.label}{where}. "
                "Szczegóły są w Cerberze → Historia.", "cerber_blocked", 403)
        self._clean[repo] = time.monotonic()
        return verdict


async def check_connection(settings: Any, *, client_factory: Any = None) -> dict[str, Any]:
    """``{ok, message, installed}`` for the settings panel's "Sprawdź połączenie"."""
    cfg = getattr(settings, "cerber", None)
    installed = cerber_installed()
    if cfg is None or not (cfg.token or "").strip():
        return {"ok": False, "installed": installed,
                "message": "Wklej token z Cerbera (Integracje → API), żeby LocalCode mógł z nim rozmawiać."}
    client = (client_factory or CerberClient)(cfg.url, cfg.token)
    try:
        info = await client.info()
    except CerberError as exc:
        return {"ok": False, "installed": installed, "message": exc.message, "code": exc.code}
    return {"ok": True, "installed": installed,
            "message": "Połączono z Cerberem — modele będą sprawdzane przed pobraniem."
                       if cfg.enabled else "Połączono z Cerberem. Włącz sprawdzanie, żeby działało.",
            "info": {k: info[k] for k in list(info)[:8] if isinstance(info[k], (str, int, float, bool))}}
