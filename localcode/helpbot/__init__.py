"""Pomocnik — the help bot that runs the rest of LocalCode for the user.

A small model (``settings.helpbot.model``, Qwen3 4B Instruct by default) that is
downloaded with the app and can do what the app's screens do: find, download,
configure, load, unload and delete models, change any setting and read the
diagnostics. See :mod:`localcode.helpbot.ops` for the full list and the two
things it deliberately cannot do.

It is the coding agent's loop with a different job, not a second engine: the
same permissions (everything that changes something asks first), the same
compaction, the same event stream — so the terminal and the Mac app show it with
the views they already have. What differs is the prompt
(:mod:`localcode.helpbot.prompt`), the tools (:mod:`localcode.helpbot.tools`)
and the working directory.

**The working directory is how a session is recognised.** Every help bot
session works in ``<LOCALCODE_HOME>/helpbot``; the server builds a help bot for
any session in that directory, so a help bot conversation survives a restart
and a resume without a new field in the session format.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

from localcode import config as lc_config

log = logging.getLogger(__name__)

__all__ = [
    "HELPBOT_DIRNAME",
    "helpbot_dir",
    "is_helpbot_cwd",
    "make_agent",
    "model_key",
    "status",
    "ensure_installed",
]

HELPBOT_DIRNAME = "helpbot"
#: Room kept free on top of the model itself before an automatic download starts.
#: A full disk breaks far more than a missing help bot does.
_DOWNLOAD_HEADROOM = 2 * 1024 ** 3


def helpbot_dir() -> Path:
    """``<LOCALCODE_HOME>/helpbot`` — created on first use."""
    path = lc_config.home() / HELPBOT_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_helpbot_cwd(cwd: Any) -> bool:
    try:
        return os.path.realpath(str(cwd)) == os.path.realpath(str(lc_config.home() / HELPBOT_DIRNAME))
    except (OSError, TypeError):
        return False


def model_key(settings: Any) -> str:
    return str(getattr(getattr(settings, "helpbot", None), "model", "") or "")


def make_agent(core: Any, *, on_event: Any = None, ask_permission: Any = None,
               session: Any = None) -> Any:
    """The help bot: the agent loop with the help bot's prompt, tools and facade."""
    from localcode.agent.loop import Agent  # noqa: PLC0415 - heavy
    from localcode.helpbot.ops import HelpbotOps  # noqa: PLC0415
    from localcode.helpbot.prompt import HELPBOT_PROMPT  # noqa: PLC0415
    from localcode.helpbot.tools import STATE_KEY, helpbot_tools  # noqa: PLC0415

    key = _resolved_key(core)
    ops = HelpbotOps(core, helpbot_key=key)
    return Agent(
        core.runtime, key, str(helpbot_dir()), core.settings,
        on_event, ask_permission,
        session=session,
        tools=helpbot_tools,
        system_prompt=HELPBOT_PROMPT,
        tool_state={STATE_KEY: ops},
        allow_always=False,
    )


def _resolved_key(core: Any) -> str:
    """The configured model as the library knows it (a fragment is enough)."""
    wanted = model_key(core.settings)
    library = getattr(core, "library", None)
    model = None
    if library is not None and wanted:
        model = library.get(wanted) or library.resolve(wanted)
    if model is None:
        raise HelpbotMissing(wanted)
    return str(model.key)


class HelpbotMissing(RuntimeError):
    """The help bot's model is not on disk (yet)."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Model pomocnika ({key or 'nieustawiony'}) nie jest jeszcze pobrany. "
                         "Pobierz go: localcode pomoc --install albo przyciskiem w aplikacji.")
        self.key = key


def status(core: Any) -> dict[str, Any]:
    """``GET /api/helpbot`` — is the help bot ready, coming, or missing?"""
    wanted = model_key(core.settings)
    library = getattr(core, "library", None)
    model = (library.get(wanted) or library.resolve(wanted)) if library is not None and wanted else None
    job = _download_job(core, wanted)
    loaded = False
    runtime = getattr(core, "runtime", None)
    if model is not None and runtime is not None:
        try:
            loaded = any(i.get("modelKey") == model.key
                         for i in (runtime.status().get("instances") or []))
        except Exception:  # noqa: BLE001
            loaded = False
    failed = bool(job and job.get("error"))
    if model is not None:
        state = "ready"
    elif job and not job.get("done") and not failed:
        state = "downloading"
    else:
        # A failed download is "missing" again, with its reason — so the panel
        # offers "Spróbuj ponownie" instead of a progress bar that never moves.
        state = "missing"
    return {
        "model": wanted,
        "modelKey": model.key if model is not None else None,
        "displayName": model.display_name if model is not None else None,
        "sizeBytes": int(model.size_bytes) if model is not None else None,
        "state": state,
        "loaded": loaded,
        "download": job,
        "cwd": str(helpbot_dir()),
        "autoDownload": bool(getattr(core.settings.helpbot, "auto_download", True)),
        **({"error": f"Pobieranie się nie udało: {job.get('error')}"} if failed and model is None else {}),
    }


def _download_job(core: Any, wanted: str) -> dict[str, Any] | None:
    manager = getattr(core, "downloads", None)
    if manager is None or not wanted:
        return None
    repo = wanted.split("/", 2)
    repo_id = "/".join(repo[:2]) if len(repo) >= 2 else wanted
    try:
        jobs = [j for j in manager.list() if str(j.get("repo") or "") == repo_id]
    except Exception:  # noqa: BLE001
        return None
    return jobs[-1] if jobs else None


def room_problem(settings: Any, size_bytes: int = 0) -> str | None:
    """Why there is no room for the help bot's model, or ``None`` when there is.

    Shared by the server's automatic download and ``localcode pomoc``, so both
    refuse the same way instead of one of them filling the disk.
    """
    size = int(size_bytes or 0) or 2_500_000_000
    target = settings.models_dir if os.path.isdir(settings.models_dir) else lc_config.home()
    try:
        free = shutil.disk_usage(target).free
    except OSError:
        return None
    if free >= size + _DOWNLOAD_HEADROOM:
        return None
    return (f"Za mało miejsca na dysku na model pomocnika: wolne {_gb(free)}, potrzeba "
            f"ok. {_gb(size + _DOWNLOAD_HEADROOM)} (model + zapas). Zwolnij miejsce i spróbuj "
            f"ponownie.")


def _gb(n: int) -> str:
    """``2.3 GB`` in Polish: ``2,3 GB``."""
    return f"{n / 1e9:.1f}".replace(".", ",") + " GB"


async def ensure_installed(core: Any, *, reason: str = "auto") -> dict[str, Any]:
    """Start downloading the help bot's model when it is missing.

    Returns :func:`status`. Refuses — with ``status.error`` — when the disk does
    not have room for the model plus :data:`_DOWNLOAD_HEADROOM`: starting a
    2.3 GB download on a full disk breaks more than a missing help bot does.
    """
    current = status(core)
    if current["state"] != "missing":
        return current
    wanted = current["model"]
    try:
        from localcode.hub.catalog import Catalog  # noqa: PLC0415

        catalog = Catalog(core.settings, library=core.library, downloads=core.downloads)
        # Unknown size (offline index, renamed repo) is not "free": room_problem
        # assumes the default model's 2.3 GB rather than skip the check.
        problem = room_problem(core.settings, _expected_size(catalog, wanted))
        if problem:
            current["error"] = problem
            return current
        await catalog.ainstall(wanted)
        log.info("helpbot: pobieram %s (%s)", wanted, reason)
    except Exception as exc:  # noqa: BLE001 - offline, store error: say so, never crash startup
        current["error"] = f"Nie udało się zacząć pobierania pomocnika: {getattr(exc, 'message', None) or exc}"
        return current
    return status(core)


def _expected_size(catalog: Any, wanted: str) -> int:
    try:
        rows = catalog.entries(query=wanted.split("/")[-1], limit=5)
    except Exception:  # noqa: BLE001
        return 0
    for row in rows:
        if str(row.get("repo") or row.get("id") or "") == wanted:
            return int(row.get("sizeBytes") or 0)
    return 0
