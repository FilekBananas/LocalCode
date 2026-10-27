"""``localcode doctor`` — is this machine ready to run local models?

The report is a checklist: every line is ``✔`` (fine), ``!`` (works, but worth
knowing) or ``✗`` (broken), and every ``✗``/``!`` carries a concrete command to
fix it.  The process exits with ``1`` when something *essential* is missing —
an interpreter that is too old, no inference engine at all, or a models
directory that cannot be written — so the installer and CI can use it as a gate.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table
from rich.text import Text

from localcode import config as lc_config
from localcode.config import VERSION, Settings, install_command, load_settings

__all__ = ["Check", "Report", "collect", "main", "run_doctor"]

#: Oldest interpreter LocalCode is tested with.
MIN_PYTHON = (3, 11)
#: Warn below this much free disk (models are big).
MIN_FREE_DISK_GB = 10.0
#: Warn below this much unified memory.
MIN_RAM_GB = 8.0
HF_PROBE_URL = "https://huggingface.co/api/models?limit=1"
HF_TIMEOUT_S = 6.0

_MARKS = {"ok": ("✔", "lc.ok"), "warn": ("!", "lc.warn"), "fail": ("✗", "lc.err")}


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: str          # ok | warn | fail
    detail: str = ""
    fix: str = ""
    essential: bool = False

    @property
    def failed(self) -> bool:
        return self.status == "fail"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *checks: Check) -> None:
        self.checks.extend(checks)

    @property
    def ok(self) -> bool:
        """False when an *essential* check failed (the exit code)."""
        return not any(c.failed and c.essential for c in self.checks)

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if c.status in ("fail", "warn")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": VERSION,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "fix": c.fix,
                 "essential": c.essential}
                for c in self.checks
            ],
        }


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_platform() -> Check:
    system = platform.system()
    machine = platform.machine()
    release = platform.mac_ver()[0] or platform.release()
    if system != "Darwin":
        return Check("System", "fail", f"{system} {release}",
                     "LocalCode działa na macOS (Apple Silicon). Na innych systemach "
                     "uruchom llama.cpp bezpośrednio.", essential=True)
    if machine != "arm64":
        return Check("System", "warn", f"macOS {release} ({machine})",
                     "To Mac z procesorem Intel — modele MLX nie zadziałają, "
                     "zostaje tylko format GGUF (llama.cpp).")
    return Check("System", "ok", f"macOS {release} · Apple Silicon ({machine})")


def check_python() -> Check:
    version = ".".join(str(v) for v in sys.version_info[:3])
    if sys.version_info < MIN_PYTHON:
        return Check("Python", "fail", f"{version} ({sys.executable})",
                     f"Potrzebny Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} lub nowszy: "
                     "brew install python@3.12 && ./scripts/install.sh", essential=True)
    return Check("Python", "ok", f"{version} ({sys.executable})")


def check_memory() -> tuple[Check, Check]:
    from localcode.runtime import memory

    info = memory.system_memory()
    total_gb = info["total"] / memory.GiB
    free_gb = info["available"] / memory.GiB
    status = "ok" if total_gb >= MIN_RAM_GB else "warn"
    fix = ("" if status == "ok" else
           "Mniej niż 8 GB pamięci — używaj modeli do 4 mld parametrów "
           "(np. Qwen3 4B Q4_K_M) i kontekstu 4–8k.")
    ram = Check("Pamięć", status,
                f"{total_gb:.1f} GB łącznie, {free_gb:.1f} GB wolne".replace(".", ","), fix)
    budget = memory.default_ram_budget(load_settings())
    budget_gb = budget / memory.GiB
    hint = ("limit z ustawień (runtime.ram_limit_gb)"
            if load_settings().runtime.ram_limit_gb else "automatycznie 70% pamięci")
    budget_check = Check("Budżet modeli", "ok",
                         f"{budget_gb:.1f} GB — {hint}".replace(".", ","),
                         "Zmiana: localcode runtime set ram-gb 20")
    return ram, budget_check


def check_disk(settings: Settings) -> Check:
    path = Path(settings.models_dir).expanduser()
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check("Dysk", "warn", str(exc), "Sprawdź uprawnienia do katalogu modeli.")
    free_gb = usage.free / (1024 ** 3)
    status = "ok" if free_gb >= MIN_FREE_DISK_GB else "warn"
    fix = "" if status == "ok" else (
        "Mało miejsca na modele. Usuń nieużywane: localcode models rm <klucz>")
    return Check("Wolne miejsce", status, f"{free_gb:.1f} GB".replace(".", ","), fix)


def check_models_dir(settings: Settings) -> Check:
    path = Path(settings.models_dir).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check("Katalog modeli", "fail", f"{path}: {exc.strerror or exc}",
                     f"Utwórz katalog ręcznie albo zmień ścieżkę w {lc_config.settings_path()}.",
                     essential=True)
    if not os.access(path, os.W_OK):
        return Check("Katalog modeli", "fail", f"{path} (brak prawa zapisu)",
                     f"chmod u+w {path}", essential=True)
    return Check("Katalog modeli", "ok", str(path))


def check_speech() -> Check:
    """The optional speech extra (transcription and reading aloud)."""
    have: list[str] = []
    missing: list[str] = []
    for module, label in (("mlx_whisper", "mlx-whisper"), ("mlx_audio", "mlx-audio")):
        try:
            __import__(module)
            have.append(label)
        except Exception:  # noqa: BLE001 - not installed is the normal case
            missing.append(label)
    if not have:
        return Check("Mowa (dyktowanie i głos)", "warn", "brak dodatku",
                     f"Zainstaluj: {install_command('speech')}")
    if missing:
        return Check("Mowa (dyktowanie i głos)", "warn",
                     f"{', '.join(have)} — brakuje {', '.join(missing)}",
                     f"Zainstaluj całość: {install_command('speech')}")
    return Check("Mowa (dyktowanie i głos)", "ok", ", ".join(have))


def check_store_index() -> Check:
    """The shipped store catalogue (``localcode/hub/data/catalog.json.gz``).

    Without it the store still works — staff picks plus a live search — so this
    is a warning, not a failure; but an installation that lost the file would
    otherwise silently show sixty models instead of thousands.
    """
    try:
        from localcode.hub import store_data

        count = store_data.count()
        built = store_data.generated_at() or "?"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        return Check("Sklep z modelami", "warn", f"nie udało się wczytać katalogu: {exc}",
                     "Zbuduj katalog: python scripts/build_catalog.py build")
    if not count:
        return Check("Sklep z modelami", "warn", "brak wbudowanego katalogu modeli",
                     "Zbuduj katalog: python scripts/build_catalog.py build "
                     "(patrz docs/CATALOG.md)")
    return Check("Sklep z modelami", "ok", f"{count} modeli w katalogu · zbudowany {built[:10]}")


def check_llama_server(settings: Settings) -> Check:
    try:
        from localcode.runtime.llama_engine import find_llama_server
    except Exception:  # pragma: no cover - engine module missing
        find_llama_server = None  # type: ignore[assignment]
    binary = find_llama_server(settings) if find_llama_server else shutil.which("llama-server")
    if not binary:
        return Check("llama-server (GGUF)", "fail", "nie znaleziono",
                     "brew install llama.cpp   (albo ./scripts/install.sh)")
    version = _binary_version(binary)
    return Check("llama-server (GGUF)", "ok", f"{binary}{' · ' + version if version else ''}")


def _binary_version(binary: str) -> str:
    """First line of ``<binary> --version`` (llama.cpp prints it on stderr)."""
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    text = (result.stderr or result.stdout or "").strip()
    return text.splitlines()[0].strip() if text else ""


def check_mlx() -> Check:
    if platform.machine() != "arm64":
        return Check("mlx-lm (MLX)", "warn", "niedostępne na Intelu",
                     "Na tym Macu używaj modeli GGUF.")
    try:
        import mlx_lm  # noqa: PLC0415 - optional dependency, checked on purpose
    except ImportError:
        return Check("mlx-lm (MLX)", "warn", "nie zainstalowano",
                     "pip install mlx-lm   (albo ./scripts/install.sh)")
    version = getattr(mlx_lm, "__version__", "")
    try:
        import mlx.core as mx

        version += f" · mlx {getattr(mx, '__version__', '')}"
    except Exception:  # noqa: BLE001 - mlx-lm without mlx is broken but not fatal
        pass
    return Check("mlx-lm (MLX)", "ok", version or "zainstalowane")


def _module_version(name: str) -> str | None:
    """``name``'s version string when it is importable, else ``None``."""
    import importlib.metadata  # noqa: PLC0415 - only needed here

    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001 - not installed, or installed without metadata
        return None


def check_images() -> Check:
    """The optional image backends (docs/IMAGES.md) — never fatal."""
    found = [f"{name} {version}"
             for name, version in ((n, _module_version(n))
                                   for n in ("mflux", "stable-diffusion-cpp-python"))
             if version]
    if found:
        return Check("Generowanie obrazów", "ok", " · ".join(found))
    return Check("Generowanie obrazów", "warn", "brak dodatku do obrazów",
                 f"Zainstaluj, aby generować obrazy: {install_command('images')}")


def check_video() -> Check:
    """The optional video extra (docs/VIDEO.md) — never fatal.

    Odrębny dodatek, bo jako jedyny ciągnie PyTorcha (ok. 1 GB na dysku); bez
    niego cała reszta LocalCode działa, tylko ekran „Wideo" nic nie wygeneruje.
    """
    found = [f"{name} {version}"
             for name, version in ((n, _module_version(n))
                                   for n in ("diffusers", "torch"))
             if version]
    if len(found) == 2:
        return Check("Generowanie wideo", "ok", " · ".join(found))
    if found:
        return Check("Generowanie wideo", "warn", f"niepełny dodatek: {' · '.join(found)}",
                     f"Zainstaluj całość: {install_command('video')}")
    return Check("Generowanie wideo", "warn", "brak dodatku do wideo",
                 f"Zainstaluj, aby generować wideo: {install_command('video')}")


def check_engines(llama: Check, mlx: Check) -> Check:
    """At least one inference engine must work — this is the essential one."""
    if llama.status == "ok" or mlx.status == "ok":
        return Check("Silnik modeli", "ok",
                     " + ".join(n for n, c in (("GGUF", llama), ("MLX", mlx)) if c.status == "ok"))
    return Check("Silnik modeli", "fail", "brak jakiegokolwiek silnika",
                 "Zainstaluj co najmniej jeden: brew install llama.cpp  lub  pip install mlx-lm",
                 essential=True)


def check_library(settings: Settings) -> Check:
    try:
        from localcode.hub.library import ModelLibrary

        models = ModelLibrary(settings).scan()
    except Exception as exc:  # noqa: BLE001 - a broken model must not break the report
        return Check("Modele", "warn", f"nie udało się przeskanować: {exc}",
                     f"Sprawdź katalog {settings.models_dir}.")
    if not models:
        return Check("Modele", "warn", "brak modeli na dysku",
                     "Pobierz pierwszy model: localcode models curated")
    by_format: dict[str, int] = {}
    for model in models:
        by_format[model.format] = by_format.get(model.format, 0) + 1
    total = sum(m.size_bytes for m in models)
    detail = (f"{len(models)} " + _plural(len(models), "model", "modele", "modeli") + ": "
              + ", ".join(f"{count}× {fmt.upper()}" for fmt, count in sorted(by_format.items()))
              + f" · {total / (1024 ** 3):.1f} GB".replace(".", ","))
    return Check("Modele", "ok", detail)


async def check_server() -> Check:
    from localcode.tui.backends import discover_server, read_server_info

    info = read_server_info()
    if info is None:
        return Check("Serwer LocalCode", "ok", "nie działa (terminal użyje trybu lokalnego)",
                     "Uruchomienie: localcode serve  albo aplikacja LocalCode.app")
    live = await discover_server()
    if live is None:
        return Check("Serwer LocalCode", "warn",
                     f"plik {lc_config.server_info_path().name} wskazuje port {info.port}, "
                     "ale serwer nie odpowiada",
                     f"Usuń nieaktualny plik: rm {lc_config.server_info_path()}")
    return Check("Serwer LocalCode", "ok",
                 f"{live.base_url} · wersja {live.version or '?'} · pid {live.pid or '?'}")


async def check_huggingface(settings: Settings) -> Check:
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - hard dependency
        return Check("Hugging Face", "warn", "brak biblioteki aiohttp", "pip install aiohttp")
    headers = {"User-Agent": f"localcode/{VERSION}"}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"
    try:
        timeout = aiohttp.ClientTimeout(total=HF_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(HF_PROBE_URL) as response:
                status = response.status
                await response.read()
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return Check("Hugging Face", "warn", f"brak połączenia ({type(exc).__name__})",
                     "Pobieranie modeli nie zadziała. Sprawdź internet albo proxy "
                     "(HTTPS_PROXY), modele z dysku działają normalnie.")
    if status >= 400:
        return Check("Hugging Face", "warn", f"HTTP {status}",
                     "Sprawdź token: localcode models search qwen")
    token = "z tokenem" if settings.hf_token else "bez tokenu (modele publiczne)"
    return Check("Hugging Face", "ok", f"połączenie działa, {token}")


def check_settings() -> Check:
    path = lc_config.settings_path()
    if not path.exists():
        return Check("Ustawienia", "ok", f"domyślne (plik {path} powstanie przy pierwszej zmianie)")
    try:
        path.read_text(encoding="utf-8")
    except OSError as exc:
        return Check("Ustawienia", "warn", f"{path}: {exc.strerror or exc}",
                     f"Napraw uprawnienia pliku {path}.")
    return Check("Ustawienia", "ok", str(path))


def _plural(n: int, one: str, few: str, many: str) -> str:
    if n == 1:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


async def collect(settings: Settings | None = None) -> Report:
    """Run every check (the network ones concurrently)."""
    settings = settings if settings is not None else load_settings()
    report = Report()
    report.add(check_platform(), check_python())
    ram, budget = check_memory()
    report.add(ram)
    report.add(check_disk(settings))
    llama = await asyncio.to_thread(check_llama_server, settings)
    mlx = check_mlx()
    report.add(check_engines(llama, mlx), llama, mlx)
    report.add(check_images())
    report.add(check_speech())
    report.add(check_video())
    report.add(check_models_dir(settings), check_settings(), check_store_index())
    library, server, hf = await asyncio.gather(
        asyncio.to_thread(check_library, settings),
        check_server(),
        check_huggingface(settings),
    )
    report.add(library, server, hf, budget)
    return report


def render_report(report: Report, console: Console) -> None:
    """Print the checklist plus the list of things to fix."""
    table = Table.grid(padding=(0, 1))
    table.add_column(width=1)
    table.add_column(style="lc.dim", no_wrap=True)
    table.add_column(overflow="fold")
    for check in report.checks:
        mark, style = _MARKS.get(check.status, _MARKS["ok"])
        table.add_row(Text(mark, style=style), check.name, check.detail or "—")
    console.print(Text(f"LocalCode {VERSION} — diagnostyka", style="lc.title"))
    console.print(table)
    fixes = [c for c in report.checks if c.fix and c.status in ("fail", "warn")]
    if fixes:
        console.print("")
        console.print(Text("Co zrobić:", style="lc.title"))
        for check in fixes:
            mark, style = _MARKS.get(check.status, _MARKS["warn"])
            console.print(Text(f"  {mark} {check.name}: ", style=style).append(check.fix, style=""))
    console.print("")
    if report.ok:
        console.print(Text("Wszystko gotowe — uruchom `localcode`.", style="lc.ok"))
    else:
        console.print(Text("Brakuje czegoś niezbędnego (zobacz ✗ powyżej).", style="lc.err"))


async def run_doctor(console: Console | None = None, *, settings: Settings | None = None) -> int:
    """Run the checks and print the report; returns the exit code."""
    from localcode.tui.stream import make_console

    console = console or make_console()
    report = await collect(settings)
    render_report(report, console)
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """``localcode doctor`` (also usable as ``python -m localcode.doctor``)."""
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="localcode doctor",
                                     description="Sprawdza, czy ten Mac jest gotowy do "
                                                 "uruchamiania modeli lokalnie.")
    parser.add_argument("--json", action="store_true", help="wypisz wynik w formacie JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.json:
        report = asyncio.run(collect())
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.ok else 1
    return asyncio.run(run_doctor())


if __name__ == "__main__":  # pragma: no cover - manual use
    raise SystemExit(main())
