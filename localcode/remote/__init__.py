"""Models that run somewhere else: OpenAI, Anthropic, Gemini, DeepInfra.

LocalCode is a local-first app and this does not change that — nothing here
does anything until somebody pastes an API key, and a borrowed model is marked
as remote everywhere it appears.  What it buys is the awkward case: a model far
too large for this Mac, or a machine that is busy with something else.

The pieces:

* :mod:`localcode.remote.providers` — the four clients (two wire dialects) and
  the model lists they answer with;
* :mod:`localcode.remote.engine` — one of those models wearing the local
  :class:`~localcode.runtime.base.Engine` interface, so the chat, the agent and
  the compaction code do not need to know the difference;
* :class:`RemoteRegistry` — settings in, usable models out.
"""

from __future__ import annotations

from typing import Any

from localcode.remote.engine import RemoteEngine, RemoteLease, remote_model
from localcode.remote.providers import (
    sees_images,
    DEFAULT_BASE_URLS,
    PROVIDER_KEY_URLS,
    PROVIDER_LABELS,
    PROVIDERS,
    SERVICE_TIERS,
    TIERED_PROVIDERS,
    RemoteClient,
    RemoteError,
    RemoteModel,
    effective_tier,
    is_remote_key,
    parse_key,
    probe,
)
from localcode.runtime.base import LoadConfig

__all__ = [
    "DEFAULT_BASE_URLS",
    "PROVIDERS",
    "PROVIDER_KEY_URLS",
    "PROVIDER_LABELS",
    "SERVICE_TIERS",
    "TIERED_PROVIDERS",
    "RemoteClient",
    "RemoteEngine",
    "RemoteError",
    "RemoteLease",
    "RemoteModel",
    "RemoteRegistry",
    "is_remote_key",
    "parse_key",
    "probe",
    "remote_model",
]


class RemoteRegistry:
    """The configured providers, their models, and engines for them.

    One instance lives next to the runtime; it holds no keys of its own, only a
    reference to the settings, so removing a key in the UI takes effect on the
    next call rather than on the next restart.
    """

    def __init__(self, settings: Any, *, client_factory: Any = None) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._clients: dict[str, RemoteClient] = {}
        #: What each cached client was built from: key, endpoint, service tier.
        self._built_from: dict[str, tuple[str, str | None, str]] = {}

    # configuration -----------------------------------------------------------
    def _config(self, provider: str) -> Any:
        return getattr(getattr(self.settings, "remote", None), provider, None)

    def configured(self, provider: str) -> bool:
        config = self._config(provider)
        return bool(config and (config.api_key or "").strip() and config.enabled)

    @property
    def providers(self) -> list[str]:
        """Providers with a usable key, in the order the settings screen shows."""
        return [p for p in PROVIDERS if self.configured(p)]

    def client(self, provider: str) -> RemoteClient:
        config = self._config(provider)
        if not self.configured(provider):
            raise RemoteError(
                f"{PROVIDER_LABELS.get(provider, provider)} nie ma zapisanego klucza API. "
                "Dodaj go w Ustawieniach.", "not_configured", 400)
        key = (config.api_key or "").strip()
        tier = effective_tier(provider, getattr(config, "service_tier", None))
        # Any of the three changed in the settings → a new client on the next
        # call; switching to Flex must not wait for a restart to take effect.
        built_from = (key, config.base_url, tier)
        cached = self._clients.get(provider)
        if cached is not None and self._built_from.get(provider) == built_from:
            return cached
        if self._client_factory is not None:
            # The factory keeps its (provider, key, base_url) shape; the tier
            # is set on what it returns.
            client = self._client_factory(provider, key, config.base_url)
            client.service_tier = tier
        else:
            client = RemoteClient(provider, key, base_url=config.base_url, service_tier=tier)
        if cached is not None and self._built_from.get(provider, ())[:2] == built_from[:2]:
            # Only the tier moved: same key, same endpoint, same model list.
            client._models = getattr(cached, "_models", None)
        self._clients[provider] = client
        self._built_from[provider] = built_from
        return client

    def forget(self, provider: str | None = None) -> None:
        """Drop cached clients (after a key changes)."""
        if provider is None:
            self._clients.clear()
            self._built_from.clear()
        else:
            self._clients.pop(provider, None)
            self._built_from.pop(provider, None)

    # models ------------------------------------------------------------------
    async def models(self, provider: str | None = None, *,
                     refresh: bool = False) -> list[RemoteModel]:
        """Models from one provider, or from every configured one."""
        wanted = [provider] if provider else self.providers
        out: list[RemoteModel] = []
        for name in wanted:
            try:
                out.extend(await self.client(name).models(refresh=refresh))
            except RemoteError:
                if provider:
                    raise
                # One unreachable provider must not empty the whole list.
                continue
        return out

    async def local_models(self, *, refresh: bool = False) -> list[Any]:
        """The same models as library rows, for the pickers and "Moje modele"."""
        return [remote_model(m.key, display_name=m.display_name,
                             context_length=m.context_length, description=m.description,
                             supports_tools=m.supports_tools, supports_vision=m.supports_vision)
                for m in await self.models(refresh=refresh)]

    # engines -----------------------------------------------------------------
    def engine(self, key: str, *, n_ctx: int | None = None,
               known: dict[str, RemoteModel] | None = None) -> RemoteEngine:
        parsed = parse_key(key)
        if parsed is None:
            raise RemoteError(f"„{key}” nie jest modelem zdalnym.", "not_remote", 400)
        provider, model_id = parsed
        client = self.client(provider)
        described = (known or {}).get(key)
        if described is None:
            # The list fetched for the pickers knows the context window and
            # whether the model reads images; reuse it rather than guess.
            cached = getattr(client, "_models", None)
            if isinstance(cached, tuple) and len(cached) == 2:
                described = next((m for m in cached[1] if m.key == key), None)
        model = remote_model(
            key,
            display_name=described.display_name if described else model_id,
            context_length=described.context_length if described else None,
            supports_tools=described.supports_tools if described else None,
            supports_vision=(described.supports_vision if described
                             else sees_images(provider, model_id, {}, {})),
        )
        config = LoadConfig(n_ctx=n_ctx or 0)
        return RemoteEngine(f"remote-{provider}-{model_id}", model, config, client)

    async def lease(self, key: str, *, n_ctx: int | None = None,
                    purpose: str = "chat") -> RemoteLease:
        engine = self.engine(key, n_ctx=n_ctx)
        await engine.start()
        return RemoteLease(engine, purpose)
