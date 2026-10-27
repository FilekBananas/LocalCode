"""The real implementation of :class:`localcode.agent.tools.models.ModelOps`.

This is the only place where the assistant's model-management tools touch the
store, the library and the runtime.  Keeping it behind one small class is what
makes the tools testable with a fake and what makes "what exactly may the model
do to my machine" answerable by reading a hundred lines.

What it deliberately cannot do: delete anything, change the RAM budget, or
reach outside the catalogue — installing still goes through
:meth:`localcode.hub.catalog.Catalog.ainstall`, which resolves a name the same
way the store does and refuses what it cannot find.
"""

from __future__ import annotations

from typing import Any

from localcode.agent.tools.models import ModelOps


class LocalModelOps(ModelOps):
    """Model management for one agent session.

    ``runtime`` is the shared :class:`~localcode.runtime.manager.RuntimeManager`;
    ``on_switch`` is called with the model key after a successful load, which is
    how the conversation actually moves to the newly loaded model.
    """

    def __init__(self, settings: Any, *, runtime: Any = None, catalog: Any = None,
                 library: Any = None, downloads: Any = None, on_switch: Any = None) -> None:
        self.settings = settings
        self.runtime = runtime
        self._catalog = catalog
        self._library = library
        self._downloads = downloads
        self.on_switch = on_switch

    # plumbing ----------------------------------------------------------------
    @property
    def catalog(self) -> Any:
        if self._catalog is None:
            from localcode.hub.catalog import Catalog

            self._catalog = Catalog(self.settings, library=self._library,
                                    downloads=self._downloads)
        elif self._downloads is not None and getattr(self._catalog, "downloads", None) is None:
            self._catalog.downloads = self._downloads
        return self._catalog

    @property
    def library(self) -> Any:
        if self._library is None:
            from localcode.hub.library import ModelLibrary

            self._library = ModelLibrary(self.settings)
        return self._library

    # operations --------------------------------------------------------------
    async def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        import asyncio

        return await asyncio.to_thread(self.catalog.entries, query=query, limit=limit)

    async def installed(self) -> list[dict[str, Any]]:
        import asyncio

        models = await asyncio.to_thread(self.library.list)
        return [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in models]

    async def install(self, name_or_repo: str, quant: str | None = None) -> dict[str, Any]:
        """Install a model; ``quant`` ("Q8_0", "8bit", "8-bit"…) picks the version.

        Without it the store chooses, as its "Zainstaluj" button does.
        """
        if not (quant or "").strip():
            return await self.catalog.ainstall(name_or_repo)
        import asyncio

        entry = await asyncio.to_thread(self.catalog.detail, name_or_repo)
        variants = list(entry.get("variants") or [])
        variant = pick_variant(variants, str(quant))
        if variant is None:
            labels = ", ".join(str(v.get("label")) for v in variants) or "brak listy wersji"
            raise RuntimeError(f"„{entry.get('name') or name_or_repo}” nie ma wersji „{quant}”. "
                               f"Dostępne: {labels}.")
        return await self.catalog.ainstall(str(entry["id"]), variant=str(variant["id"]))

    async def remote(self) -> list[dict[str, Any]]:
        """Models behind the API keys in Ustawienia → Modele zdalne ("Kimi K2.5 (DeepInfra)")."""
        registry = getattr(self.runtime, "remote", None) if self.runtime is not None else None
        try:
            if registry is None or not registry.providers:
                return []
            models = await registry.models()
        except Exception:  # noqa: BLE001 - a provider being down hides its models, nothing more
            return []
        from localcode.remote.providers import PROVIDER_LABELS, name_with_provider

        return [{"key": m.key, "displayName": name_with_provider(m.display_name or m.model_id, m.key),
                 "provider": PROVIDER_LABELS.get(m.provider, m.provider), "remote": True,
                 "contextLength": m.context_length} for m in models]

    async def resolve(self, text: str) -> dict[str, Any] | None:
        """A key or a name → the model it means, on disk first, then remote."""
        from localcode.agent.tools.models import matches
        from localcode.remote.providers import is_remote_key

        text = (text or "").strip()
        if not text:
            return None
        if not is_remote_key(text):
            model = self.library.get(text) or self.library.resolve(text)
            if model is not None:
                return {"key": model.key, "displayName": model.display_name, "remote": False}
        remote = await self.remote()
        exact = next((m for m in remote if m["key"] == text), None)
        if exact is not None:
            return exact
        hits = [m for m in remote if matches(m, text)]
        # The shortest name among the matches: "deepseek v4 pro" should pick
        # "DeepSeek-V4-Pro", not "DeepSeek-V4-Pro-Distill-Something".
        return min(hits, key=lambda m: len(str(m["displayName"]))) if hits else None

    async def _nearest(self, text: str) -> str:
        """" Najbliższe: …" — the models sharing the most name words ("kimi k2.5" → Kimi-K2.6, Kimi-K3)."""
        from localcode.agent.tools.models import similarity

        scored = [(similarity(m, text), m) for m in await self.remote()]
        best = sorted((pair for pair in scored if pair[0] > 0),
                      key=lambda pair: (-pair[0], len(str(pair[1]["displayName"]))))[:6]
        names = [m["displayName"] for _, m in best]
        return f" Najbliższe: {', '.join(names)}." if names else " Sprawdź model_list."

    async def load(self, key: str) -> dict[str, Any]:
        if self.runtime is None:
            raise RuntimeError("Silnik modeli jest niedostępny w tej sesji.")
        target = await self.resolve(key)
        if target is not None and target.get("remote"):
            # Nothing to load: a lease opens on the first step. Only the
            # conversation moves.
            if callable(self.on_switch):
                result = self.on_switch(target["key"])
                if hasattr(result, "__await__"):
                    await result
            return dict(target)
        model = self.library.get(key) or self.library.resolve(key)
        if model is None:
            raise RuntimeError(f"Nie znaleziono modelu „{key}” — ani na dysku, ani u dostawców "
                               f"zdalnych.{await self._nearest(key)}")
        info = await self.runtime.load(model.key)
        if callable(self.on_switch):
            result = self.on_switch(model.key)
            if hasattr(result, "__await__"):
                await result
        return {**(info if isinstance(info, dict) else {}),
                "key": model.key, "displayName": model.display_name}

    async def status(self) -> dict[str, Any]:
        if self.runtime is None:
            return {"instances": [], "budget": {}}
        status = self.runtime.status()
        return status if isinstance(status, dict) else {"instances": [], "budget": {}}


def pick_variant(variants: list[dict[str, Any]], quant: str) -> dict[str, Any] | None:
    """The version a person (or the assistant) named: "Q8_0", "q4_k_m", "8bit", "8-bit".

    An exact label wins; a bit count ("8bit") takes the first version with that
    many bits, in the store's order — "Q8_0" for GGUF, "8bit" for MLX.
    """
    import re

    def norm(text: Any) -> str:
        return re.sub(r"[^A-Z0-9_]", "", str(text or "").upper().replace("-", ""))

    wanted = norm(quant)
    if not wanted:
        return None
    for variant in variants:
        if wanted in (norm(variant.get("label")), norm(variant.get("quant")), norm(variant.get("id"))):
            return variant
    bits = re.fullmatch(r"(?:Q)?(\d{1,2})(?:BIT|B)?", wanted)
    if bits:
        n = bits.group(1)
        for variant in variants:
            label = norm(variant.get("quant") or variant.get("label"))
            if label.startswith(f"Q{n}") or label.startswith(f"IQ{n}") or label.startswith(f"{n}BIT") \
                    or label in (f"UDQ{n}KXL",):
                return variant
        if n == "16":
            return next((v for v in variants
                         if norm(v.get("quant") or v.get("label")) in ("F16", "BF16", "FP16")), None)
    return None

