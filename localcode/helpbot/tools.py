"""The help bot's tools: models, downloads, settings and diagnostics.

Search, list and install are the agent's own model tools (``agent/tools/models``)
— the same store, the same descriptions. The rest is new, because the coding
agent never needed it: unloading, choosing how a model is loaded, cancelling a
download, changing any setting, deleting a model, and reading the diagnostics.

Permissions follow the agent's rules, so a 4B model with full power still cannot
act on a misheard sentence: reading anything runs freely, and everything that
downloads, loads, unloads, deletes or changes a setting is shown to the user
first. A tool that both reads and writes (``model_settings``, ``settings``,
``downloads``) asks only when it is actually asked to change something.

Descriptions are Polish, like the model tools they sit next to; every schema is
paid for on every request, so they stay one sentence each.
"""

from __future__ import annotations

import json
from typing import Any

from localcode.agent.tools.base import PermissionInfo, Tool, ToolContext, ToolResult
from localcode.agent.tools.models import ModelInstallTool, ModelListTool, ModelSearchTool

__all__ = ["HELPBOT_TOOL_CLASSES", "STATE_KEY", "helpbot_tools"]

#: Where the session leaves :class:`~localcode.helpbot.ops.HelpbotOps`. The same
#: key the model tools read, so search/list/install work unchanged.
STATE_KEY = "models"

_NO_OPS = "Pomocnik nie ma dostępu do aplikacji w tej sesji."
_GB = 1_000_000_000


def _ops(ctx: ToolContext) -> Any:
    from localcode.helpbot.ops import HelpbotOps  # noqa: PLC0415 - cycle

    ops = ctx.state.get(STATE_KEY)
    return ops if isinstance(ops, HelpbotOps) else None


def _error(exc: Exception) -> str:
    return getattr(exc, "message", None) or str(exc)


def _compact(data: Any, limit: int = 3_000) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _thousands(n: Any) -> str:
    value = int(n or 0)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".replace(".", ",") + " mln"
    if value >= 1_000:
        return f"{value // 1_000} tys."
    return str(value)


def model_card(entry: dict[str, Any]) -> str:
    """The store's detail sheet as text: what the user sees under "Szczegóły".

    Written for two readers at once — the model, which relays it, and the user,
    who gets it pasted — so it keeps the store's own Polish sentences (trust
    reasons, fit detail) verbatim instead of paraphrasing them.
    """
    trust = entry.get("trust") or {}
    provenance = trust.get("provenance") or {}
    head = " · ".join(str(x) for x in (entry.get("format", "").upper() or None, entry.get("sizeLabel"),
                                       entry.get("paramsLabel"), entry.get("quant")) if x)
    lines = [f"{entry.get('name') or entry.get('id')} — {head}".rstrip(" —")]
    if entry.get("tagline"):
        lines.append(f"Opis: {entry['tagline']}")
    if entry.get("fit"):
        lines.append(f"Pamięć: {entry.get('fitDetail') or entry['fit']}")
    if trust.get("score") is not None:
        label = trust.get("label") or ""
        kind = provenance.get("label") or ""
        lines.append(f"Zaufanie: {trust['score']}/100 ({label})" + (f" — {kind}" if kind else ""))
        if provenance.get("detail"):
            lines.append(f"  Pochodzenie: {provenance['detail']}")
        lines += [f"  + {reason}" for reason in trust.get("reasons") or []]
        lines += [f"  ! {warning}" for warning in trust.get("warnings") or []]
    if trust.get("license"):
        lines.append(f"Licencja: {trust['license']}" + (" (wymaga zgody autora)" if trust.get("gated") else ""))
    if entry.get("downloads") or entry.get("likes"):
        lines.append(f"Popularność: {_thousands(entry.get('downloads'))} pobrań, "
                     f"{_thousands(entry.get('likes'))} polubień")
    if entry.get("watchOut"):
        lines.append(f"Uwaga: {entry['watchOut']}")
    variants = entry.get("variants") or []
    if variants:
        shown = ", ".join(
            f"{v.get('label') or v.get('quant')}"
            + (f" {_gb(v.get('sizeBytes'))}" if int(v.get("sizeBytes") or 0) >= 10_000_000 else "")
            + (" (polecany)" if v.get("recommended") else "")
            for v in variants[:8])
        more = f" i {len(variants) - 8} innych" if len(variants) > 8 else ""
        lines.append(f"Warianty: {shown}{more}")
    lines.append("Zainstalowany: tak" if entry.get("installed") else "Zainstalowany: nie")
    lines.append(f"id: {entry.get('id')}")
    return "\n".join(lines)


def _gb(n: Any) -> str:
    return f"{int(n or 0) / 1e9:.1f}".replace(".", ",") + " GB"


class ModelInfoTool(Tool):
    name = "model_info"
    kind = "read"
    description = ("Pełna karta modelu ze sklepu: do czego służy, czy się zmieści, ocena zaufania "
                   "z uzasadnieniem, pochodzenie, licencja, popularność, warianty.")
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "id z model_search albo nazwa"}},
        "required": ["name"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Karta modelu: {args.get('name', '')}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        try:
            entry = await ops.model_info(str(args.get("name") or ""))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, _error(exc), "nie znaleziono")
        return ToolResult(True, model_card(entry), f"karta: {entry.get('name') or entry.get('id')}")


class ModelLoadTool(Tool):
    name = "model_load"
    kind = "exec"
    description = "Załaduj zainstalowany model do pamięci, żeby był gotowy w czacie i w Kodzie."
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "klucz z model_list"},
            "context": {"type": "integer", "description": "okno kontekstu w tokenach (opcjonalnie)"},
        },
        "required": ["key"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Załaduj model: {args.get('key', '')}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        ops = _ops(ctx)
        note = "Model zajmie pamięć."
        if ops is not None and args.get("key"):
            try:
                fit = ops.fit(str(args["key"]))
                note = fit.get("message") or note
            except Exception:  # noqa: BLE001 - the preview is best effort
                pass
        return PermissionInfo(title=f"Załadować „{args.get('key', '')}”?", preview=note)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        key = str(args.get("key") or "").strip()
        if not key:
            return ToolResult(False, "Podaj klucz modelu (z model_list).", "brak klucza")
        try:
            fit = ops.fit(key)
            if fit.get("fits") == "no":
                return ToolResult(False, f"Ten model się nie zmieści. {fit.get('message', '')} "
                                         "Zaproponuj mniejszy albo wyładowanie innych.",
                                  "nie zmieści się")
            info = await ops.load(key, n_ctx=int(args["context"]) if args.get("context") else None)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Nie udało się załadować: {_error(exc)}", "błąd ładowania")
        name = info.get("displayName") or key
        return ToolResult(True, f"Załadowano „{name}”. Jest gotowy w czacie i w sekcji Kod.",
                          f"załadowano {name}")


class ModelUnloadTool(Tool):
    name = "model_unload"
    kind = "exec"
    description = "Wyładuj model z pamięci (klucz albo „all” — wszystkie poza mną)."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string", "description": "klucz modelu albo „all”"}},
        "required": ["key"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Wyładuj: {args.get('key', '')}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        return PermissionInfo(title=f"Wyładować „{args.get('key', '')}” z pamięci?",
                              preview="Przerwie to rozmowy, które teraz używają tego modelu.")

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        try:
            names = await ops.unload(str(args.get("key") or ""))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Nie udało się wyładować: {_error(exc)}", "błąd")
        if not names:
            return ToolResult(True, "Ten model nie był załadowany.", "nic do wyładowania")
        return ToolResult(True, "Wyładowano: " + ", ".join(names) + ".",
                          f"wyładowano {len(names)}")


class ModelSettingsTool(Tool):
    name = "model_settings"
    kind = "write"
    description = ("Pokaż albo zmień, jak model się ładuje (nCtx, nParallel, flashAttn, "
                   "kvCacheType, effort). Bez „changes” tylko pokazuje.")
    parameters = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "klucz modelu"},
            "changes": {"type": "object", "description": "np. {\"nCtx\": 32768}; null usuwa"},
        },
        "required": ["key"],
    }

    def title(self, args: dict[str, Any]) -> str:
        verb = "Zmień" if args.get("changes") else "Pokaż"
        return f"{verb} ustawienia modelu: {args.get('key', '')}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        if not args.get("changes"):
            return None                                  # reading asks nothing
        return PermissionInfo(title=f"Zmienić sposób ładowania „{args.get('key', '')}”?",
                              preview=_compact(args.get("changes"), 600))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        key = str(args.get("key") or "").strip()
        changes = args.get("changes")
        try:
            if changes:
                if not isinstance(changes, dict):
                    return ToolResult(False, "„changes” musi być obiektem.", "zły format")
                stored = ops.set_load_config(key, changes)
                return ToolResult(True, f"Zapisano. Model ładuje się teraz z: {_compact(stored)}. "
                                        "Działa od następnego załadowania.", "zapisano")
            return ToolResult(True, _compact(ops.load_config(key)), "ustawienia modelu")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, _error(exc), "błąd")


class ModelDeleteTool(Tool):
    name = "model_delete"
    kind = "write"
    description = "Usuń model z dysku, żeby zwolnić miejsce. Zawsze najpierw zapytaj użytkownika."
    parameters = {
        "type": "object",
        "properties": {"key": {"type": "string", "description": "klucz z model_list"}},
        "required": ["key"],
    }

    def title(self, args: dict[str, Any]) -> str:
        return f"Usuń model: {args.get('key', '')}"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        ops = _ops(ctx)
        size = ""
        if ops is not None and args.get("key"):
            try:
                model = ops.library.get(str(args["key"])) or ops.library.resolve(str(args["key"]))
                if model is not None:
                    size = f" ({model.size_bytes / _GB:.1f} GB)".replace(".", ",")
            except Exception:  # noqa: BLE001
                pass
        return PermissionInfo(title=f"Usunąć model „{args.get('key', '')}”{size} z dysku?",
                              preview="Tego nie da się cofnąć — trzeba go będzie pobrać od nowa.")

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        try:
            name = await ops.delete(str(args.get("key") or ""))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Nie usunąłem: {_error(exc)}", "błąd usuwania")
        return ToolResult(True, f"Usunięto „{name}” z dysku.", f"usunięto {name}")


class DownloadsTool(Tool):
    name = "downloads"
    kind = "write"
    description = "Pokaż pobierania i ich postęp; z „cancel” przerwij jedno."
    parameters = {
        "type": "object",
        "properties": {"cancel": {"type": "string", "description": "id pobierania do przerwania"}},
    }

    def title(self, args: dict[str, Any]) -> str:
        return "Przerwij pobieranie" if args.get("cancel") else "Pobierania"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        if not args.get("cancel"):
            return None
        return PermissionInfo(title="Przerwać pobieranie?",
                              preview="Pobrana część zostaje — można je wznowić później.")

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        if args.get("cancel"):
            try:
                job = ops.cancel_download(str(args["cancel"]))
            except Exception as exc:  # noqa: BLE001
                return ToolResult(False, _error(exc), "błąd")
            return ToolResult(True, f"Przerwano: {job.get('label') or args['cancel']}.", "przerwano")
        jobs = ops.downloads()
        if not jobs:
            return ToolResult(True, "Nic się teraz nie pobiera.", "brak pobierań")
        lines = []
        for job in jobs[:10]:
            done, total = job.get("doneBytes") or 0, job.get("totalBytes") or 0
            share = f"{100 * done / total:.0f}%" if total else "?"
            if job.get("error"):
                share += f" — {job['error']}"
            lines.append(f"- {job.get('label') or job.get('repo')}: {job.get('state')} {share} "
                         f"(id {job.get('id')})")
        return ToolResult(True, "\n".join(lines), f"{len(jobs)} pobierań")


class SettingsTool(Tool):
    name = "settings"
    kind = "write"
    description = ("Pokaż wszystkie ustawienia aplikacji albo zmień je, podając „changes” "
                   "w tym samym układzie (np. {\"runtime\": {\"ramLimitGb\": 20}}).")
    parameters = {
        "type": "object",
        "properties": {
            "section": {"type": "string",
                        "description": "agent, chat, store, security, server, helpbot, runtime"},
            "changes": {"type": "object", "description": "co zmienić; bez tego tylko pokazuje"},
        },
    }

    def title(self, args: dict[str, Any]) -> str:
        return "Zmień ustawienia" if args.get("changes") else "Ustawienia"

    async def permission(self, args: dict[str, Any], ctx: ToolContext) -> PermissionInfo | None:
        if not args.get("changes"):
            return None
        return PermissionInfo(title="Zmienić ustawienia aplikacji?",
                              preview=_compact(args.get("changes"), 800))

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        changes = args.get("changes")
        try:
            if changes:
                if not isinstance(changes, dict):
                    return ToolResult(False, "„changes” musi być obiektem.", "zły format")
                snapshot = await ops.update_settings(changes)
                section = next(iter(changes))
                return ToolResult(True, f"Zapisano. {section}: {_compact(snapshot.get(section))}",
                                  "zapisano ustawienia")
            snapshot = await ops.settings_snapshot()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Nie zapisano: {_error(exc)}", "błąd ustawień")
        section = str(args.get("section") or "").strip()
        data = snapshot.get(section) if section else snapshot
        if section and data is None:
            return ToolResult(False, f"Nie ma sekcji „{section}”.", "brak sekcji")
        return ToolResult(True, _compact(data), "ustawienia")


class DoctorTool(Tool):
    name = "doctor"
    kind = "read"
    description = "Sprawdź instalację: silniki, dodatki, pamięć, dysk, połączenie. Pokazuje, co poprawić."
    parameters = {"type": "object", "properties": {}}

    def title(self, args: dict[str, Any]) -> str:
        return "Diagnostyka"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ops = _ops(ctx)
        if ops is None:
            return ToolResult(False, _NO_OPS, "brak dostępu")
        try:
            checks = await ops.doctor()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"Diagnostyka się nie udała: {_error(exc)}", "błąd")
        marks = {"ok": "✓", "warn": "!", "fail": "✗"}
        lines = [f"{marks.get(c.get('status'), '?')} {c.get('name')}: {c.get('detail')}"
                 + (f" → {c.get('fix')}" if c.get("fix") and c.get("status") != "ok" else "")
                 for c in checks]
        bad = sum(1 for c in checks if c.get("status") != "ok")
        return ToolResult(True, "\n".join(lines), f"{bad} do poprawy" if bad else "wszystko OK")


#: In the order the model sees them: look first, then act.
HELPBOT_TOOL_CLASSES: tuple[type[Tool], ...] = (
    ModelSearchTool,
    ModelInfoTool,
    ModelListTool,
    DownloadsTool,
    SettingsTool,
    ModelSettingsTool,
    DoctorTool,
    ModelInstallTool,
    ModelLoadTool,
    ModelUnloadTool,
    ModelDeleteTool,
)


def helpbot_tools(*_args: Any, **_kwargs: Any) -> list[Tool]:
    """The help bot's menu, whatever the window — it has no smaller version.

    Accepts and ignores the agent loop's factory arguments (mode, tiny, skills):
    a help bot without, say, ``model_unload`` on a small window would simply be
    unable to do what it was asked.
    """
    return [cls() for cls in HELPBOT_TOOL_CLASSES]
