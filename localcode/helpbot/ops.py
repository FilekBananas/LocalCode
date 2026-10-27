"""Everything the help bot may do, behind one class.

The help bot runs the rest of the app for somebody who does not want to learn
it: finding a model, downloading it, choosing how to load it, loading and
unloading, changing any setting, reading the diagnostics. Every one of those
already exists for the app's own screens — the store, the library, the runtime
manager, ``Core.update_app_settings`` — so this class adds no new behaviour. It
only gathers the same calls in one place, which is what makes "what exactly can
the help bot do to my machine" answerable by reading one file.

It is built on a :class:`~localcode.server.core.Core`, and that is deliberate:
the server has one, and so does the terminal (``LocalBackend`` runs a ``Core``
in-process). The help bot therefore changes settings through the same code path
as the app's settings screen — validation, persistence and the live update of
whatever is already running.

Two things it cannot do, on purpose:

* **set an API key.** A key typed into this conversation would be stored in its
  transcript for good; keys go through Ustawienia → Modele zdalne, which writes
  them to the settings file and nowhere else.
* **unload itself through "unload all".** The bot only holds its model while it
  is generating, so unloading it mid-turn just makes the next sentence wait for
  a reload — and, if memory is tight, push out the very model it was asked to
  load.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from localcode.agent.model_ops import LocalModelOps

log = logging.getLogger(__name__)

__all__ = ["HelpbotOps", "HelpbotError"]

#: Settings sections the help bot may change, as ``Core.app_settings`` names them
#: (plus ``runtime``, which belongs to the runtime manager).
SETTINGS_SECTIONS = ("agent", "chat", "store", "security", "server", "helpbot", "runtime")


class HelpbotError(RuntimeError):
    """A help bot action failed; the message is Polish and shown to the model."""


class HelpbotOps(LocalModelOps):
    """Model management plus settings, diagnostics and downloads, for one session.

    Loading never moves the conversation (``on_switch`` stays ``None``): the
    help bot loads a model *for the user* and must still be the help bot after.
    """

    def __init__(self, core: Any, *, helpbot_key: str | None = None) -> None:
        super().__init__(core.settings, runtime=core.runtime, library=core.library,
                         downloads=getattr(core, "downloads", None), on_switch=None)
        self.core = core
        self.helpbot_key = helpbot_key

    # -- models ------------------------------------------------------------ #

    def _model(self, key: str) -> Any:
        model = self.library.get(key) or self.library.resolve(key)
        if model is None:
            raise HelpbotError(f"Nie ma na dysku modelu „{key}”. Sprawdź listę (model_list).")
        return model

    def fit(self, key: str) -> dict[str, Any]:
        """Whether ``key`` fits in memory now, and what would be pushed out."""
        model = self._model(key)
        estimate = getattr(self.runtime, "estimate", None)
        if not callable(estimate):
            return {"fits": "unknown", "message": ""}
        est = estimate(model.key)
        return {"fits": str(getattr(est, "fits", "unknown")),
                "message": str(getattr(est, "message", "") or ""),
                "bytes": int(getattr(est, "total_bytes", 0) or getattr(est, "totalBytes", 0) or 0)}

    async def load(self, key: str, *, n_ctx: int | None = None) -> dict[str, Any]:  # type: ignore[override]
        if self.runtime is None:
            raise HelpbotError("Silnik modeli jest niedostępny w tej sesji.")
        from localcode.remote.providers import is_remote_key

        target = await self.resolve(key)
        if is_remote_key(key) or (target is not None and target.get("remote")):
            name = (target or {}).get("displayName") or key
            raise HelpbotError(
                f"„{name}” to model zdalny — nie ładuje się go do pamięci. Żeby go używać, ustaw go "
                f"jako domyślny narzędziem settings: changes "
                f"{{\"chat\": {{\"defaultModel\": \"{(target or {}).get('key') or key}\"}}}} "
                f"(Czat) albo z kluczem \"agent\" (Kod).")
        model = self._model(key)
        config = None
        if n_ctx:
            from localcode.runtime.base import LoadConfig  # noqa: PLC0415

            config = LoadConfig(n_ctx=int(n_ctx))
        info = await self.runtime.load(model.key, config) if config else await self.runtime.load(model.key)
        out = dict(info) if isinstance(info, dict) else {}
        return {**out, "key": model.key, "displayName": model.display_name}

    async def unload(self, which: str) -> list[str]:
        """Unload one model (by key or name) or every model except the help bot."""
        status = await self.status()
        instances = [i for i in (status.get("instances") or []) if i.get("id")]
        wanted = (which or "").strip()
        if wanted.lower() in ("all", "wszystko", "wszystkie", "*"):
            targets = [i for i in instances if i.get("modelKey") != self.helpbot_key]
        else:
            model = self.library.get(wanted) or self.library.resolve(wanted)
            key = model.key if model is not None else wanted
            targets = [i for i in instances if i.get("modelKey") == key]
        if not targets:
            return []
        names: list[str] = []
        for instance in targets:
            await self.runtime.unload(str(instance["id"]))
            names.append(str(instance.get("displayName") or instance.get("modelKey")))
        return names

    def _is_own_model(self, model: Any) -> bool:
        """The help bot's model, judged from the settings and not only from
        ``helpbot_key``: an empty or stale key must never make it deletable."""
        own = {self.helpbot_key} if self.helpbot_key else set()
        configured = str(getattr(getattr(self.settings, "helpbot", None), "model", "") or "")
        if configured:
            own.add(configured)
            resolved = self.library.get(configured) or self.library.resolve(configured)
            if resolved is not None:
                own.add(resolved.key)
        return model.key in own or str(getattr(model, "repo_id", "") or "") in own

    async def delete(self, key: str) -> str:
        """Remove a model's files (unloading it first); returns its display name."""
        model = self._model(key)
        if self._is_own_model(model):
            raise HelpbotError("Nie usunę własnego modelu — bez niego nie byłoby pomocnika.")
        await self.unload(model.key)
        await asyncio.to_thread(self.library.delete, model.key)
        return str(model.display_name)

    async def model_info(self, name: str) -> dict[str, Any]:
        """The store's full card for one model — the "Szczegóły" sheet's data.

        An id goes straight to :meth:`Catalog.detail`; a name goes through the
        store's own search first, because ``resolve_name("flux klein")`` picks
        a text-encoder repository over FLUX itself. Offline, the variant list
        (one hub request) is what is lost, not the card: the index row has the
        trust score, the reasons, the licence and the fit.
        """
        from localcode.hub.catalog import CatalogError  # noqa: PLC0415

        wanted = (name or "").strip()
        if not wanted:
            raise HelpbotError("Podaj nazwę albo identyfikator modelu.")
        catalog = self.catalog
        if "/" in wanted:                       # an id: straight to the store's detail
            try:
                return await asyncio.to_thread(catalog.detail, wanted)
            except CatalogError as exc:
                if getattr(exc, "code", "") != "not_found":
                    row = await self._first_row(wanted)
                    if row is not None:
                        return row              # offline: the card without variants
                    raise HelpbotError(str(exc)) from exc
        # A name never goes to detail(): it resolves names its own way, and
        # "flux klein" came back as a text-encoder conversion.
        row = await self._first_row(wanted)
        if row is None:
            raise HelpbotError(f"Nie znalazłem w sklepie modelu „{wanted}”.")
        try:
            return await asyncio.to_thread(catalog.detail, str(row.get("id")))
        except CatalogError:
            return row

    async def _first_row(self, query: str) -> dict[str, Any] | None:
        """The model a person most likely means by a name.

        The store's top hit is ranked for browsing, not for "the one I mean":
        "flux klein" put a FLUX text-encoder conversion (trust 55) above the
        FLUX.2 klein the user has installed (trust 86). So among the first few
        hits: the installed one, then the most trusted.
        """
        rows = await asyncio.to_thread(self.catalog.entries, query=query, limit=5)
        if not rows:
            return None
        best = max(rows, key=lambda r: (bool(r.get("installed")),
                                         int((r.get("trust") or {}).get("score") or 0)))
        return dict(best)

    def load_config(self, key: str) -> dict[str, Any]:
        model = self._model(key)
        return {"key": model.key, "displayName": model.display_name,
                "contextLength": getattr(model.meta, "context_length", None),
                "defaults": self.library.load_defaults(model.key) or {}}

    def set_load_config(self, key: str, changes: dict[str, Any]) -> dict[str, Any]:
        """Merge ``changes`` into the model's load defaults (``None`` clears a key)."""
        model = self._model(key)
        current = dict(self.library.load_defaults(model.key) or {})
        for name, value in (changes or {}).items():
            if value is None:
                current.pop(name, None)
            else:
                current[name] = value
        try:
            stored = self.library.set_load_defaults(model.key, current or None)
        except Exception as exc:  # noqa: BLE001 - LibraryError carries the Polish message
            raise HelpbotError(str(exc)) from exc
        return stored or {}

    # -- downloads --------------------------------------------------------- #

    def downloads(self) -> list[dict[str, Any]]:
        manager = getattr(self.core, "downloads", None)
        if manager is None:
            return []
        try:
            return list(manager.list())
        except Exception:  # noqa: BLE001 - a broken manager means "nothing to show"
            return []

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        manager = getattr(self.core, "downloads", None)
        if manager is None:
            raise HelpbotError("Menedżer pobierania nie działa w tej sesji.")
        try:
            return dict(manager.cancel(job_id))
        except Exception as exc:  # noqa: BLE001
            raise HelpbotError(f"Nie udało się przerwać pobierania: {exc}") from exc

    # -- settings ---------------------------------------------------------- #

    async def settings_snapshot(self) -> dict[str, Any]:
        app = self.core.app_settings()
        status = await self.status()
        return {**app, "runtime": dict(status.get("settings") or {})}

    async def update_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply a camelCase patch through the same code as the settings screen."""
        if not isinstance(changes, dict) or not changes:
            raise HelpbotError("Podaj, co zmienić, np. {\"agent\": {\"maxSteps\": 60}}.")
        patch = dict(changes)
        # Secrets first: answered with "unknown setting", the model would tell
        # the user the option does not exist instead of where it lives.
        if "remote" in patch or "hfToken" in patch:
            raise HelpbotError("Kluczy API i tokenów nie zmieniam w rozmowie — zostałyby w jej "
                               "zapisie. Wpisz je w Ustawienia → Modele zdalne.")
        runtime_patch = patch.pop("runtime", None)
        unknown = [k for k in patch if k not in SETTINGS_SECTIONS and k not in (
            "modelsDir", "language", "scanLmstudio", "scanHfCache")]
        if unknown:
            raise HelpbotError(f"Nie znam ustawień: {', '.join(unknown)}. "
                               f"Sekcje: {', '.join(SETTINGS_SECTIONS)}.")
        try:
            if runtime_patch:
                await self.runtime.update_settings(dict(runtime_patch))
            if patch:
                await self.core.update_app_settings(patch)
        except Exception as exc:  # noqa: BLE001 - ApiError / SettingsError, both Polish
            message = getattr(exc, "message", None) or str(exc)
            raise HelpbotError(message) from exc
        return await self.settings_snapshot()

    # -- diagnostics ------------------------------------------------------- #

    async def doctor(self) -> list[dict[str, Any]]:
        from localcode.doctor import collect  # noqa: PLC0415 - heavy, optional

        report = await collect(self.settings)
        data = report.to_dict()
        return list(data.get("checks") or [])
