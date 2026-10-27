"""Tools that let the assistant manage the models it runs on.

"Uruchom mi Qwen3 Coder" should be a sentence, not a trip to another screen.
With these four tools the model can look through the store, download something,
load it and hand the conversation over to it — which is how a small model that
is always resident becomes useful: it does not have to *be* good at everything,
it only has to know who is.

The tools never talk to Hugging Face or the runtime directly.  Everything goes
through a small facade the caller puts in ``ctx.state["models"]``
(:class:`ModelOps`), so the agent loop decides what the assistant is allowed to
reach, the tests can pass a fake, and a session without that facade simply gets
a clear refusal instead of a crash.

Downloading gigabytes and loading a model into memory are not read-only acts,
so ``model_install`` and ``model_load`` ask for permission exactly like
``write_file`` and ``bash`` do; the whole family is also off unless
``settings.agent.manage_models`` is on.
"""

from __future__ import annotations

import abc
from typing import Any

from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolResult

#: Where the caller leaves the facade.
STATE_KEY = "models"

_NO_OPS = ("Zarządzanie modelami jest wyłączone w tej sesji. "
           "Włącz je w ustawieniach agenta (manage_models).")


class ModelOps(abc.ABC):
    """What the assistant may do with models.

    Deliberately narrow: find, install, list, load.  There is no "delete" —
    removing somebody's 20 GB download is not a thing to do from a sentence.
    """

    @abc.abstractmethod
    async def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    async def installed(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    async def install(self, name_or_repo: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def load(self, key: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    async def status(self) -> dict[str, Any]: ...

    async def remote(self) -> list[dict[str, Any]]:
        """Models behind the API keys saved in Ustawienia → Modele zdalne."""
        return []

    async def resolve(self, text: str) -> dict[str, Any] | None:
        """``{"key", "displayName", "remote", "provider"}`` for a key or a name, if any."""
        return None


def _ops(ctx: ToolContext) -> ModelOps | None:
    ops = ctx.state.get(STATE_KEY)
    return ops if isinstance(ops, ModelOps) else None


def _fit_note(row: dict[str, Any]) -> str:
    fit = str(row.get("fit") or "")
    return f" [{fit}]" if fit and fit != "Zmieści się" else ""


class ModelSearchTool(Tool):
    name = "model_search"
    kind = "read"
    description = ("Znajdź model w sklepie LocalCode (po nazwie albo po tym, do czego ma służyć). "
                   "Zwraca nazwę, rozmiar, ocenę zaufania i czy zmieści się w pamięci.")
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "np. „qwen coder”, „mały model po polsku”, „do obrazków”"},
            "limit": {"type": "integer", "description": "ile wyników (domyślnie 8)"},
        },
        "required": ["query"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Szukaj modelu: {args.get('query', '')}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu do modeli")
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(False, "Podaj, czego mam szukać.", "pusty query")
        limit = max(1, min(int(args.get("limit") or 8), 20))
        rows = await ops.search(query, limit=limit)
        if not rows:
            return ToolResult(True, f"Nic nie znalazłem dla „{query}”.", "0 wyników")
        lines = []
        for row in rows:
            trust = (row.get("trust") or {}).get("score")
            marks = " (zainstalowany)" if row.get("installed") else ""
            lines.append(
                f"- {row.get('name')} — {row.get('sizeLabel') or '?'}"
                f"{_fit_note(row)}, zaufanie {trust if trust is not None else '?'}/100{marks}\n"
                f"  id: {row.get('id')}\n"
                f"  {row.get('tagline') or ''}".rstrip()
            )
        return ToolResult(True, "\n".join(lines), f"{len(rows)} modeli dla „{query}”")


#: Remote models listed without a query: a gateway offers a hundred, and a
#: small model reading all of them has no room left for the conversation.
REMOTE_LIST_LIMIT = 25


#: Words a person (or a small model) puts around a model's name that are not
#: part of it: "kimi z deep infry", "deep_infra_k2_6_key", "załaduj mi model …".
_NOISE = frozenset("""
deepinfra infra infry openai anthropic gemini google unlid provider dostawca dostawcy
key klucz klucza api model modelu modele the a an from od z ze na w do mi ty go ten ta to
load zaladuj załaduj wlacz włącz ustaw uruchom use switch please prosze proszę remote zdalny zdalnie
""".split())


def _words(text: str) -> list[str]:
    import re

    return re.sub(r"[^a-z0-9ąćęłńóśźż]+", " ", (text or "").lower()).split()


def query_words(query: str) -> list[str]:
    """The words of a request that can name a model: "deep_infra_k2_6_key" → ["k2", "6"]."""
    words = _words(query)
    out: list[str] = []
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if word in _NOISE or (word == "deep" and nxt in ("infra", "infry")):
            continue
        out.append(word)
    return out


def _haystack(model: dict[str, Any]) -> list[str]:
    """The model's own name — not its provider: "(DeepInfra)" must not make every
    DeepInfra model match the word "deep"."""
    import re

    key = str(model.get("key") or "")
    model_id = key.split(":", 2)[2] if key.startswith("remote:") and key.count(":") >= 2 else key
    name = re.sub(r"\s*\([^)]*\)\s*$", "", str(model.get("displayName") or ""))
    return _words(f"{name} {model_id}")


def _hit(word: str, hay: list[str]) -> bool:
    # A number must be a whole token ("6" is not in "0905"); a word may be part
    # of one ("seek" in "deepseek", "k2" in "k2").
    if word.isdigit():
        return word in hay
    return any(word in token for token in hay)


def matches(model: dict[str, Any], query: str) -> bool:
    """Every name word of ``query`` appears in the model's name (any case, any punctuation)."""
    words = query_words(query)
    if not words:
        return True            # only a provider or filler words: everything they cover
    hay = _haystack(model)
    return all(_hit(word, hay) for word in words)


def similarity(model: dict[str, Any], query: str) -> int:
    """How many name words of ``query`` the model has — for "the nearest are …"."""
    hay = _haystack(model)
    return sum(1 for word in query_words(query) if not word.isdigit() and _hit(word, hay))


class ModelListTool(Tool):
    name = "model_list"
    kind = "read"
    description = ("Wypisz modele na dysku, te załadowane do pamięci i modele zdalne "
                   "(od dostawców z kluczem API). Tylko pokazuje — niczego nie przełącza; "
                   "do przełączenia służy model_load.")
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "fragment nazwy modelu, o który prosi użytkownik; bez niego — cała lista"},
        },
    }

    def title(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        return f"Lista modeli: {query}" if query else "Lista modeli"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu do modeli")
        query = str(args.get("query") or "").strip()
        every_remote = await ops.remote()
        models = [m for m in await ops.installed() if not query or matches(m, query)]
        remote = [m for m in every_remote if not query or matches(m, query)]
        # "kimi k2" must not hide Kimi K3: the rest of the family is shown
        # apart, so the assistant can say what there is instead of guessing.
        similar = sorted((m for m in every_remote if m not in remote and similarity(m, query) > 0),
                         key=lambda m: (-similarity(m, query), len(str(m.get("displayName")))))[:10] \
            if query else []
        status = await ops.status()
        loaded = {str(i.get("modelKey")) for i in (status.get("instances") or [])}
        if not models and not remote and not similar:
            if query:
                return ToolResult(True, f"Nie ma modelu pasującego do „{query}” — ani na dysku, "
                                        "ani u dostawców zdalnych. Spróbuj model_search.", "0 modeli")
            return ToolResult(True, "Na dysku nie ma jeszcze żadnego modelu.", "0 modeli")
        lines = [
            f"- {m.get('displayName') or m.get('key')} ({m.get('format')}, "
            f"{round(float(m.get('sizeBytes') or 0) / 1e9, 1)} GB)"
            f"{' — załadowany' if m.get('key') in loaded else ''}\n  key: {m.get('key')}"
            for m in models
        ]
        if remote:
            shown = remote if query else remote[:REMOTE_LIST_LIMIT]
            lines.append("\nModele zdalne (działają u dostawcy, płatne za zapytanie; "
                         "model_load przełącza na nie rozmowę):")
            lines.extend(f"- {m.get('displayName')} — key: {m.get('key')}" for m in shown)
            if len(shown) < len(remote):
                lines.append(f"… i {len(remote) - len(shown)} więcej — zawęź: model_list(query=…).")
        if similar:
            lines.append(f"\nPodobne (nie pasują dokładnie do „{query}”):")
            lines.extend(f"- {m.get('displayName')} — key: {m.get('key')}" for m in similar)
        if query and not any(matches(m, query) for m in (*models, *remote)):
            lines.append(f"\nModelu „{query}” nie ma. Powiedz o tym użytkownikowi i zapytaj, czy chce "
                         "któryś z podobnych — nie przełączaj na inny bez jego zgody.")
        lines.append("\n(To tylko lista — nic nie zostało przełączone. Model zmienia dopiero udany "
                     "model_load.)")
        budget = status.get("budget") or {}
        if budget.get("limitBytes"):
            lines.append(f"\nPamięć dla modeli: {round(float(budget['limitBytes']) / 1e9, 1)} GB, "
                         f"zajęte {round(float(budget.get('usedBytes') or 0) / 1e9, 1)} GB.")
        summary = f"{len(models)} modeli na dysku" + (f", {len(remote)} zdalnych" if remote else "")
        return ToolResult(True, "\n".join(lines), summary)


class ModelInstallTool(Tool):
    name = "model_install"
    kind = "write"
    description = ("Pobierz i zainstaluj model ze sklepu (po nazwie albo identyfikatorze "
                   "repozytorium). Pobieranie idzie w tle. Bez „quant” sklep sam dobiera wersję.")
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "nazwa z wyszukiwarki albo „autor/model”"},
            "quant": {"type": "string",
                      "description": "wersja, gdy użytkownik o nią prosi: Q4_K_M, Q8_0, 4bit, 8bit…"},
        },
        "required": ["name"],
    }

    def title(self, args: dict[str, Any]) -> str:
        quant = str(args.get("quant") or "").strip()
        return f"Zainstaluj model: {args.get('name', '')}" + (f" ({quant})" if quant else "")

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        quant = str(args.get("quant") or "").strip()
        return PermissionInfo(
            title=f"Pobrać model „{args.get('name', '')}”" + (f" w wersji {quant}" if quant else "") + "?",
            preview="Pobranie zajmie miejsce na dysku i transfer z internetu.",
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu do modeli")
        name = str(args.get("name") or "").strip()
        if not name:
            return ToolResult(False, "Podaj, który model mam zainstalować.", "brak nazwy")
        quant = str(args.get("quant") or "").strip() or None
        try:
            job = await (ops.install(name, quant=quant) if quant else ops.install(name))
        except Exception as exc:  # noqa: BLE001 - the message is for the model to read
            return ToolResult(False, f"Nie udało się zacząć pobierania: {exc}", "błąd instalacji")
        return ToolResult(
            True,
            f"Pobieram „{job.get('label') or name}” ({job.get('repo')}). "
            "Postęp widać w aplikacji; gdy się skończy, model pojawi się na liście.",
            f"pobieram {job.get('label') or name}",
        )


class ModelLoadTool(Tool):
    name = "model_load"
    kind = "exec"
    description = ("Przełącz rozmowę na inny model: załaduj model z dysku do pamięci albo użyj "
                   "modelu zdalnego (od dostawcy z kluczem API). Podaj „key” z model_list albo nazwę. "
                   "Tylko model, o który prosi użytkownik — gdy go nie ma, zapytaj, zanim wybierzesz "
                   "inną wersję. Nie mów, że model jest ustawiony, dopóki to narzędzie się nie uda.")
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "klucz z model_list albo nazwa modelu"},
        },
        "required": ["key"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Załaduj model: {args.get('key', '')}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        ops = _ops(ctx)
        target = None
        if ops is not None and args.get("key"):
            try:
                target = await ops.resolve(str(args["key"]))
            except Exception:  # noqa: BLE001 - the question is still asked, just plainer
                target = None
        if target and target.get("remote"):
            from localcode.agent.permissions import REMOTE_FLAG

            provider = target.get("provider") or "dostawcy"
            return PermissionInfo(
                title=f"Przełączyć rozmowę na model zdalny „{target.get('displayName')}” {REMOTE_FLAG}?",
                preview=(f"Od teraz rozmowa i pliki, które agent czyta, będą wysyłane do {provider}. "
                         "Każde zapytanie jest płatne. Model nie zajmuje pamięci tego Maca."),
            )
        return PermissionInfo(
            title=f"Załadować model „{(target or {}).get('displayName') or args.get('key', '')}”?",
            preview="Model zajmie pamięć; w razie potrzeby inny zostanie wyładowany.",
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu do modeli")
        key = str(args.get("key") or "").strip()
        if not key:
            return ToolResult(False, "Podaj klucz modelu (z model_list).", "brak klucza")
        try:
            info = await ops.load(key)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Nie udało się załadować modelu: {exc}", "błąd ładowania")
        name = info.get("displayName") or key
        if info.get("remote"):
            return ToolResult(
                True,
                f"Rozmowa jest teraz na modelu zdalnym „{name}”. Działa u dostawcy — nie zajmuje "
                "pamięci tego Maca, a każde zapytanie jest płatne.",
                f"przełączono na {name}",
            )
        return ToolResult(
            True,
            f"Model „{name}” jest załadowany i od teraz odpowiada na pytania w tej rozmowie.",
            f"załadowano {name}",
        )


#: The family, in the order the model sees it.
MODEL_TOOL_CLASSES: tuple[type[Tool], ...] = (
    ModelSearchTool,
    ModelListTool,
    ModelInstallTool,
    ModelLoadTool,
)
