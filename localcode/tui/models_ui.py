"""Choosing, downloading and inspecting models from the terminal.

Shared by the TUI slash commands (``/model``, ``/download``, ``/search``) and
by ``localcode models …`` so both show the same tables, the same
"does it fit in this Mac's memory" verdict and the same progress bar.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import Any, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

from localcode.hub.metadata import base_quant
from localcode.remote.providers import name_with_provider, provider_label
from localcode.tui.images_in import model_sees_images
from localcode.tui import render
from localcode.tui.backends import Backend, BackendError
from localcode.tui.prompts import Choice, choose

__all__ = [
    "FITS_LABELS",
    "catalog_footer",
    "catalog_summary",
    "catalog_table",
    "install_from_catalog",
    "next_step_hint",
    "IN_PROGRESS_STATES",
    "download_model",
    "fits_label",
    "first_run",
    "model_choices",
    "pick_model",
    "pick_variant",
    "show_model_details",
    "watch_download",
]

#: Download job states that are not finished yet.
IN_PROGRESS_STATES = ("queued", "downloading")
#: Extra memory a loaded model needs beyond its weights (KV cache + overhead).
LOAD_OVERHEAD = 1.25
#: ``MemoryEstimate.fits`` in Polish.
FITS_LABELS = {"yes": "mieści się", "tight": "ciasno", "no": "nie mieści się"}


def fits_label(size_bytes: int, budget_bytes: int) -> tuple[str, str]:
    """``("mieści się", "lc.ok")`` / ``("ciasno", …)`` / ``("za duży", …)``."""
    if not budget_bytes:
        return "—", "lc.dim"
    need = size_bytes * LOAD_OVERHEAD
    if need <= budget_bytes * 0.7:
        return "mieści się", "lc.ok"
    if need <= budget_bytes:
        return "ciasno", "lc.warn"
    return "za duży", "lc.err"


async def _budget(backend: Backend) -> int:
    try:
        status = await backend.runtime_status()
    except Exception:  # noqa: BLE001 - the picker must work without the runtime
        return 0
    budget = status.get("budget") or {}
    return int(budget.get("limitBytes") or 0)


# --------------------------------------------------------------------------- #
# Local library
# --------------------------------------------------------------------------- #


def model_choices(models: Sequence[dict[str, Any]], budget_bytes: int,
                  *, current: str | None = None,
                  favorites: Sequence[str] = ()) -> list[Choice]:
    """Rows of the ``/model`` picker: name, size, quantisation, RAM verdict."""
    out: list[Choice] = []
    for model in models:
        key = str(model.get("key"))
        mark = "• " if model.get("key") == current else ""
        if key in favorites:
            mark += "★ "
        name = name_with_provider(str(model.get("displayName") or key), key)
        if model.get("remote"):
            bits = [f"zdalnie · {provider_label(key)}"]
            if model.get("contextLength"):
                bits.append(f"{render.fmt_tokens(model['contextLength'])} ctx")
            if model_sees_images(model):
                bits.append("widzi obrazy")
            bits.append("płatne za zapytanie")
            out.append(Choice(f"{mark}{name}", value=key, hint=" · ".join(bits)))
            continue
        size = int(model.get("sizeBytes") or 0)
        verdict, _ = fits_label(size, budget_bytes)
        bits = [render.fmt_bytes(size)]
        if model.get("quant"):
            bits.append(str(model["quant"]))
        bits.append(str(model.get("format") or "").upper())
        if model.get("contextLength"):
            bits.append(f"{render.fmt_tokens(model['contextLength'])} ctx")
        bits.append(verdict)
        if model_sees_images(model):
            bits.append("widzi obrazy")
        if model.get("loaded"):
            bits.append("załadowany")
        out.append(Choice(f"{mark}{name}", value=key, hint=" · ".join(bits)))
    return out


async def pick_model(backend: Backend, console: Console, *, title: str = "Wybierz model") -> str | None:
    """Interactive model picker; returns the chosen ``LocalModel.key``."""
    models = await backend.list_models()
    if not models:
        console.print(Text("Nie masz jeszcze żadnych modeli. Użyj /download albo /search.",
                           style="lc.warn"))
        return None
    budget = await _budget(backend)
    try:
        favorites = await backend.favorite_models()
    except Exception:  # noqa: BLE001 - the stars are a nicety, the picker is not
        favorites = []
    # "Ulubione" (starred here or in the app) first; then chat before
    # embedding, local before remote, by name.
    models = sorted(models, key=lambda m: (m.get("key") not in favorites,
                                           ("embedding" in (m.get("capabilities") or [])),
                                           bool(m.get("remote")),
                                           str(m.get("displayName") or m.get("key"))))
    choices = model_choices(models, budget, current=backend.model_key, favorites=favorites)
    selected = next((i for i, m in enumerate(models) if m.get("key") == backend.model_key), 0)
    return await choose(choices, title=title, selected=selected, numbered=True)


async def show_model_details(backend: Backend, console: Console, key: str) -> None:
    """``localcode models info`` — metadata plus the memory estimate."""
    model = await backend.model(key)
    if model is None:
        raise BackendError(f"Nie znaleziono modelu „{key}”.")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="lc.dim", justify="right")
    table.add_column()
    table.add_row("klucz", str(model.get("key")))
    table.add_row("nazwa", str(model.get("displayName")))
    table.add_row("format", str(model.get("format", "")).upper())
    table.add_row("kwantyzacja", str(model.get("quant") or "—"))
    table.add_row("rozmiar", render.fmt_bytes(model.get("sizeBytes")))
    table.add_row("architektura", str(model.get("arch") or "—"))
    if model.get("paramsB"):
        table.add_row("parametry", f"{render.fmt_number(model['paramsB'])} mld")
    if model.get("contextLength"):
        table.add_row("kontekst modelu", render.fmt_tokens(model["contextLength"]))
    table.add_row("możliwości", ", ".join(model.get("capabilities") or []) or "—")
    table.add_row("ścieżka", render.short_path(str(model.get("path") or ""), limit=70))
    console.print(table)
    with contextlib.suppress(Exception):
        estimate = await backend.estimate(str(model["key"]))
        if estimate:
            console.print(Text(
                f"Pamięć: {render.fmt_bytes(estimate.get('totalBytes'))} "
                f"(wagi {render.fmt_bytes(estimate.get('weightsBytes'))}, "
                f"KV {render.fmt_bytes(estimate.get('kvCacheBytes'))}) "
                f"przy {render.fmt_tokens(estimate.get('nCtx'))} kontekstu — "
                f"{FITS_LABELS.get(str(estimate.get('fits')), '—')}", style="lc.dim"))
            if estimate.get("message"):
                console.print(Text(str(estimate["message"]), style="lc.dim"))


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #


def variant_choices(variants: Sequence[dict[str, Any]], budget_bytes: int) -> list[Choice]:
    out: list[Choice] = []
    for variant in variants:
        size = int(variant.get("sizeBytes") or 0)
        estimate = variant.get("estimate") or {}
        verdict = FITS_LABELS.get(str(estimate.get("fits") or ""),
                                  fits_label(size, budget_bytes)[0])
        bits = [render.fmt_bytes(size), verdict]
        if variant.get("recommended"):
            bits.append("zalecany")
        out.append(Choice(str(variant.get("label") or variant.get("quant") or variant.get("id")),
                          value=str(variant.get("id")), hint=" · ".join(bits)))
    return out


async def pick_variant(backend: Backend, console: Console, repo: str,
                       quant: str | None = None) -> tuple[dict[str, Any], str] | None:
    """``(details, variant id)`` — the quantisation to download from ``repo``."""
    details = await backend.hub_details(repo)
    variants = [v for v in (details.get("variants") or []) if not v.get("isVisionProjector")]
    if not variants:
        raise BackendError(f"Repozytorium {repo} nie zawiera plików, które LocalCode potrafi uruchomić.")
    if quant:
        wanted = quant.strip().lower()
        match = next((v for v in variants
                      if wanted in (str(v.get("quant") or "").lower(),
                                    str(v.get("label") or "").lower(),
                                    str(v.get("id") or "").lower())), None)
        if match is None:
            match = next((v for v in variants if wanted in str(v.get("label") or "").lower()), None)
        if match is None:
            available = ", ".join(str(v.get("quant") or v.get("label")) for v in variants)
            raise BackendError(f"Nie ma wariantu „{quant}” w {repo}. Dostępne: {available}.")
        return details, str(match["id"])
    if len(variants) == 1:
        return details, str(variants[0]["id"])
    budget = await _budget(backend)
    recommended = next((i for i, v in enumerate(variants) if v.get("recommended")), 0)
    if not (sys.stdin and sys.stdin.isatty()):
        chosen = variants[recommended]
        console.print(Text(f"Wybrano zalecany wariant: {chosen.get('label') or chosen.get('quant')}",
                           style="lc.dim"))
        return details, str(chosen["id"])
    console.print(Text(f"{repo} — wybierz kwantyzację:", style="lc.title"))
    chosen = await choose(variant_choices(variants, budget), title="", selected=recommended)
    if chosen is None:
        return None
    return details, str(chosen)


async def watch_download(backend: Backend, console: Console, job: dict[str, Any]) -> dict[str, Any]:
    """Follow one download job with a progress bar until it ends."""
    job_id = str(job.get("id") or "")
    label = str(job.get("label") or job.get("repo") or job_id)
    columns = (
        SpinnerColumn(style="lc.accent"),
        TextColumn("[lc.dim]{task.description}"),
        BarColumn(complete_style="lc.accent", finished_style="lc.ok"),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(label, total=int(job.get("totalBytes") or 0) or None)
        while True:
            state = str(job.get("state") or "")
            total = int(job.get("totalBytes") or 0)
            done = int(job.get("doneBytes") or 0)
            stage = str(job.get("stage") or "")
            description = {"cerber": f"{label} — Cerber sprawdza model…",
                           "verify": f"{label} — sprawdzam sumy kontrolne…"}.get(stage, label)
            progress.update(task, completed=done, total=total or None, description=description)
            if state not in IN_PROGRESS_STATES:
                break
            try:
                await asyncio.sleep(0.4)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await backend.cancel_download(job_id)
                raise
            jobs = await backend.downloads()
            found = next((j for j in jobs if str(j.get("id")) == job_id), None)
            if found is None:
                break
            job = found
    state = str(job.get("state") or "")
    if state == "done":
        console.print(Text(f"✔ Pobrano {label} ({render.fmt_bytes(job.get('totalBytes'))})",
                           style="lc.ok"))
    elif state == "cancelled":
        console.print(Text(f"Pobieranie {label} anulowane.", style="lc.warn"))
    else:
        console.print(render.error_line(str(job.get("error") or f"Pobieranie {label} nie powiodło się.")))
    return job


async def download_model(backend: Backend, console: Console, repo: str,
                         quant: str | None = None) -> str | None:
    """Pick a quantisation, download it and return the new ``LocalModel.key``."""
    picked = await pick_variant(backend, console, repo, quant)
    if picked is None:
        console.print(Text("Anulowano.", style="lc.dim"))
        return None
    details, variant_id = picked
    variant = next((v for v in details.get("variants", []) if str(v.get("id")) == variant_id), {})
    files = list(variant.get("files") or [])
    projector = next((v for v in details.get("variants", [])
                      if v.get("isVisionProjector") and v.get("recommended")), None)
    if projector:
        files += [f for f in (projector.get("files") or []) if f not in files]
    job = await backend.start_download(repo, variant_id=variant_id, files=files or None)
    try:
        job = await watch_download(backend, console, job)
    except asyncio.CancelledError:
        console.print(Text("Pobieranie przerwane.", style="lc.warn"))
        return None
    if str(job.get("state")) != "done":
        return None
    models = await backend.list_models(refresh=True)
    key = job.get("modelKey")
    scan = next((m.get("scan") for m in models if m.get("key") == key), None) or {}
    verdict = scan.get("verdict")
    if verdict == "dangerous":
        console.print(Text(f"⚠ {scan.get('summary')}", style="lc.err"))
    elif verdict == "suspicious":
        console.print(Text(f"⚠ {scan.get('summary')}", style="lc.warn"))
    return str(key) if key else None


async def first_run(backend: Backend, console: Console) -> str | None:
    """No models on disk: offer the curated ones that fit this Mac and fetch one."""
    console.print(Text("Nie masz jeszcze żadnego modelu.", style="lc.warn"))
    console.print(Text("Poniżej modele polecane dla tego Maca — wybierz jeden, "
                       "a LocalCode go pobierze.", style="lc.dim"))
    try:
        curated = await backend.hub_curated()
    except Exception as exc:  # noqa: BLE001 - offline is a normal case here
        console.print(render.error_line(f"Nie udało się pobrać listy modeli: {exc}"))
        console.print(Text("Sprawdź połączenie z internetem i spróbuj: localcode models curated",
                           style="lc.dim"))
        return None
    total_gb = _total_ram_gb()
    fitting = [m for m in curated if not m.get("minRamGb") or float(m["minRamGb"]) <= total_gb]
    fitting = [m for m in fitting if m.get("category") != "embedding"] or fitting
    if not fitting:
        console.print(render.error_line("Żaden z polecanych modeli nie zmieści się w pamięci."))
        return None
    fitting.sort(key=lambda m: (m.get("category") != "code", -(m.get("paramsB") or 0)))
    choices = [
        Choice(str(m.get("id")),
               value=str(m.get("id")),
               hint=" · ".join(filter(None, [
                   str(m.get("format") or "").upper(),
                   f"{m.get('paramsB')} mld" if m.get("paramsB") else "",
                   f"od {m.get('minRamGb')} GB RAM" if m.get("minRamGb") else "",
                   str(m.get("description") or "")[:60],
               ])))
        for m in fitting[:9]
    ]
    repo = await choose(choices, title="Polecane modele (Esc anuluje)")
    if not repo:
        return None
    entry = next((m for m in fitting if str(m.get("id")) == repo), {})
    return await download_model(backend, console, str(repo), entry.get("recommendedQuant"))


def _total_ram_gb() -> float:
    from localcode.runtime import memory

    try:
        return memory.system_memory()["total"] / (1024 ** 3)
    except Exception:  # noqa: BLE001 - never block the wizard
        return 16.0


# --------------------------------------------------------------------------- #
# Catalogue (the friendly in-app store; see localcode/hub/catalog.py)
# --------------------------------------------------------------------------- #

def _shorten(text: str, limit: int) -> str:
    """Cut ``text`` on a word boundary and mark it with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—-")
    return f"{cut}…"


#: Colour per catalogue verdict.
_FIT_STYLES = {"Zmieści się": "lc.ok", "Na styk": "lc.warn", "Za duży": "lc.err"}

#: Colour per trust band (``localcode.hub.trust``).
_TRUST_STYLES = {"high": "lc.ok", "good": "lc.ok", "medium": "lc.warn",
                 "low": "lc.err", "unknown": "lc.dim"}
#: Provenance kind → the word the table has room for.
_SOURCE_WORDS = {"official": "oficjalne", "trusted": "konwersja",
                 "community": "społeczność", "unknown": "nieznane"}


def trust_cell(entry: dict[str, Any], *, compact: bool = False) -> Text:
    """``"88 oficjalne"`` — the score and where the file comes from, coloured.

    ``compact`` drops the word: in the advanced view the repository id is on the
    same row and needs the width more than a second copy of the provenance does.
    """
    trust = entry.get("trust") or {}
    score = trust.get("score")
    if score is None:
        return Text("—", style="lc.dim")
    kind = str((entry.get("publisher") or {}).get("kind") or "unknown")
    text = Text(f"{int(score)}", style=_TRUST_STYLES.get(str(trust.get("level")), "lc.dim"))
    if not compact:
        text.append(f" {_SOURCE_WORDS.get(kind, '—')}", style="lc.dim")
    return text


def catalog_table(entries: Sequence[dict[str, Any]], *, advanced: bool = False) -> Table:
    """The catalogue as the user sees it: name, purpose, size, fit, state.

    Repository ids and quantisation labels only appear with ``advanced=True``.
    """
    from rich import box

    table = Table(box=box.ROUNDED, border_style="lc.dim", expand=False)
    table.add_column("#", justify="right", style="lc.dim")
    table.add_column("nazwa", overflow="fold")
    table.add_column("do czego służy", overflow="fold", style="lc.dim")
    table.add_column("rozmiar", justify="right")
    table.add_column("pamięć")
    table.add_column("zaufanie", justify="right" if advanced else "left")
    table.add_column("stan")
    if advanced:
        table.add_column("repozytorium", overflow="fold", style="lc.dim")
    for index, entry in enumerate(entries, 1):
        fit = str(entry.get("fit") or "—")
        state = "zainstalowany" if entry.get("installed") else ""
        badges = [b for b in (entry.get("badges") or []) if b]
        # Parameters live under the name rather than in a column of their own:
        # a ninth column pushed the repository id into an ellipsis on a normal
        # terminal, and this costs no width at all.
        name = Text(str(entry.get("name") or entry.get("id") or ""))
        if entry.get("paramsLabel"):
            name.append(f"\n{entry['paramsLabel']}", style="lc.dim")
        row = [
            str(index),
            name,
            _shorten(str(entry.get("tagline") or ""), 48 if advanced else 72),
            str(entry.get("sizeLabel") or "—").removeprefix("ok. "),
            Text(fit, style=_FIT_STYLES.get(fit, "lc.dim")),
            trust_cell(entry, compact=advanced),
            # In the advanced view the badges give way to the repository id:
            # somebody reading repository ids wants the id, not a second copy of
            # what the tagline already said.
            Text(state, style="lc.ok") if state
            else Text("" if advanced else " · ".join(badges[:2]), style="lc.dim"),
        ]
        if advanced:
            row.append(str((entry.get("source") or {}).get("repo") or entry.get("id") or ""))
        table.add_row(*row)
    return table


def catalog_footer() -> Text:
    """The discreet provenance line printed under the catalogue."""
    from localcode.hub.catalog import SOURCE_NOTE

    return Text(
        f"Katalog modeli: Hugging Face · {SOURCE_NOTE}\n"
        "Zaufanie 0-100 liczone z pochodzenia, licencji, formatu plików i popularności "
        "— to nie jest skan antywirusowy.",
        style="lc.dim")


def catalog_summary(entry: dict[str, Any]) -> Text:
    """One block describing the model about to be installed."""
    text = Text()
    text.append(str(entry.get("name") or entry.get("id")), style="lc.accent")
    text.append(f"\n{entry.get('tagline') or ''}\n", style="lc.dim")
    fit = str(entry.get("fit") or "—")
    if entry.get("paramsLabel"):
        text.append(f"Parametry: {entry['paramsLabel']}\n", style="lc.dim")
    text.append(f"Rozmiar: {entry.get('sizeLabel') or '—'}   ")
    text.append(fit, style=_FIT_STYLES.get(fit, "lc.dim"))
    text.append(f"\n{entry.get('fitDetail') or ''}", style="lc.dim")

    # Before a download starts is the moment to say what we know about the file.
    trust = entry.get("trust") or {}
    publisher = entry.get("publisher") or {}
    if trust.get("score") is not None:
        text.append("\nZaufanie: ")
        text.append(f"{int(trust['score'])}/100 {trust.get('label') or ''}",
                    style=_TRUST_STYLES.get(str(trust.get("level")), "lc.dim"))
        if publisher.get("detail"):
            text.append(f"\n{publisher['detail']}", style="lc.dim")
    if entry.get("watchOut"):
        text.append(f"\n{entry['watchOut']}", style="lc.warn")
    for warning in (trust.get("warnings") or [])[:2]:
        text.append(f"\n· {warning}", style="lc.warn")
    return text


def next_step_hint(model: dict[str, Any] | None, key: str) -> str:
    """What to do with a model that has just been installed.

    An embedding model cannot generate text and an image model is not a chat
    model, so neither may be offered as ``localcode -m`` — following that
    advice ends in a raw engine error.
    """
    caps = {str(c) for c in (model or {}).get("capabilities") or []}
    if "embedding" in caps:
        return ("Model osadzeń (embeddingi) — służy do wyszukiwania w dokumentach (RAG), "
                "nie do rozmowy. Wskaż go w aplikacji albo w settings.json "
                f"(chat.embedding_model = \"{key}\").")
    if "image" in caps:
        return f"Model obrazów. Użyj: localcode image \"twój opis\" -m {key}"
    return f"Użyj: localcode -m {key}"


def _quant_of(key: str) -> str | None:
    """The quantisation of an installed model, read from its key.

    A library key ends with the file stem, so ``…/ltxv-2b-…-Q4_K_M`` carries the
    answer without touching the disk.
    """
    from localcode.hub.metadata import base_quant as _base, parse_quant

    return _base(parse_quant(key.rsplit("/", 1)[-1] + ".gguf") or "") or None


async def install_from_catalog(catalog: Any, backend: Backend, console: Console, name: str,
                               *, variant: str | None = None,
                               assume_yes: bool = False) -> str | None:
    """``localcode models install "<nazwa>"`` — resolve, confirm, download.

    ``name`` is matched fuzzily against the friendly names, repository ids and
    aliases of the catalogue; the variant is chosen automatically unless
    ``variant`` names one.  Returns the new ``LocalModel.key``.
    """
    entry = catalog.resolve_name(name)
    if entry is None:
        console.print(Text(f"Nie znaleziono modelu „{name}” w katalogu.", style="lc.err"))
        console.print(Text("Zobacz dostępne: localcode models curated", style="lc.dim"))
        return None
    installed_quant = _quant_of(str(entry.get("installedKey") or ""))
    wanted_quant = base_quant(str(entry.get("quant") or "")) or None
    if entry.get("installed") and (variant or not wanted_quant or wanted_quant == installed_quant):
        console.print(Text(f"„{entry['name']}” jest już zainstalowany "
                           f"({entry.get('installedKey')}).", style="lc.ok"))
        return str(entry.get("installedKey") or "") or None
    if entry.get("installed"):
        # A different quantisation of the same repository is a different file and
        # a different trade-off: refusing to fetch it made "wybierz jakość"
        # unreachable for anything already on disk.
        console.print(Text(f"Masz już wersję {installed_quant or '—'}; pobieram "
                           f"{wanted_quant} obok niej.", style="lc.dim"))
    console.print(catalog_summary(entry))
    if entry.get("fit") == "Za duży" and not assume_yes:
        if not sys.stdin.isatty():
            console.print(Text("Model nie zmieści się w pamięci — przerwano "
                               "(użyj -y, aby pobrać mimo to).", style="lc.warn"))
            return None
        answer = input("Ten model nie zmieści się w pamięci tego Maca. Pobrać mimo to? [t/N] ")
        if answer.strip().lower() not in ("t", "tak", "y", "yes"):
            console.print(Text("Anulowano.", style="lc.dim"))
            return None
    files = await asyncio.to_thread(catalog.install_files, entry, variant=variant)
    job = await backend.start_download(str(entry["id"]), variant_id=variant, files=files or None)
    try:
        job = await watch_download(backend, console, job)
    except asyncio.CancelledError:
        console.print(Text("Pobieranie przerwane.", style="lc.warn"))
        return None
    if str(job.get("state")) != "done":
        return None
    models = await backend.list_models(refresh=True)
    key = job.get("modelKey")
    scan = next((m.get("scan") for m in models if m.get("key") == key), None) or {}
    verdict = scan.get("verdict")
    if verdict == "dangerous":
        console.print(Text(f"⚠ {scan.get('summary')}", style="lc.err"))
    elif verdict == "suspicious":
        console.print(Text(f"⚠ {scan.get('summary')}", style="lc.warn"))
    return str(key) if key else None
