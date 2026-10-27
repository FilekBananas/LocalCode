"""``localcode`` / ``lc`` — the command line of the terminal app.

    localcode                       interaktywny agent w bieżącym katalogu
    localcode "napraw testy"        to samo, z pierwszym poleceniem
    localcode -p "…"                jedna tura bez UI (skrypty, CI)
    localcode chat                  czat bez narzędzi
    localcode models …              katalog modeli: przeglądanie i instalacja
    localcode image "opis"          wygeneruj obraz lokalnym modelem
    localcode images …              galeria wygenerowanych obrazów
    localcode video "opis"          wygeneruj krótki klip wideo (LTX-Video 2B)
    localcode ps | load | unload    co jest w pamięci
    localcode runtime …             ustawienia środowiska modeli
    localcode serve …               serwer HTTP dla aplikacji na Maca
    localcode app | doctor          aplikacja na Maca · diagnostyka

Everything goes through :mod:`localcode.tui.backends`, so every command works
the same whether the models run in this process (``--local``) or in a server
started by the Mac app (``--attach``); with neither flag the server is used
when it is running and this process takes over when it is not.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from localcode.config import VERSION, load_settings

__all__ = ["build_parser", "main"]

#: Subcommands recognised before the free-form prompt.
SUBCOMMANDS = ("chat", "models", "image", "images", "video", "transcribe", "say", "skills", "ps",
               "load", "unload", "runtime", "serve", "app", "doctor", "help", "pomoc", "helpbot")
#: Where ``localcode app`` looks for the bundle.
APP_LOCATIONS = (
    Path(__file__).resolve().parents[1] / "macos" / "dist" / "LocalCode.app",
    Path("/Applications/LocalCode.app"),
    Path("~/Applications/LocalCode.app").expanduser(),
)

_EPILOG = """\
przykłady:
  localcode                                  agent w bieżącym katalogu
  localcode -c                               wznów ostatnią sesję
  localcode -p "co robi ten projekt?"        jedna odpowiedź na stdout
  localcode -m qwen3-4b --mode accept_edits  wybrany model, edycje bez pytania
  localcode models curated                   katalog modeli na ten komputer
  localcode models install "Qwen3 Coder 30B"  instalacja po nazwie
  localcode models download unsloth/Qwen3-4B-Instruct-2507-GGUF Q4_K_M
  localcode image "kot w kapeluszu" --size 1024x1024 --open
  localcode images list
  localcode runtime set ram-gb 20
  localcode runtime tune --model qwen3-4b --apply
"""


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-C", "--cwd", metavar="KATALOG", default=None,
                        help="katalog projektu (domyślnie bieżący)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true",
                       help="uruchom modele w tym procesie, nawet gdy serwer działa")
    group.add_argument("--attach", action="store_true",
                       help="wymagaj działającego serwera LocalCode (aplikacja na Maca)")


def build_parser() -> argparse.ArgumentParser:
    """The root parser (``localcode [PROMPT] …``)."""
    parser = argparse.ArgumentParser(
        prog="localcode",
        description="LocalCode — lokalny agent kodujący i czat dla modeli GGUF/MLX.",
        epilog=_EPILOG + "\npolecenia: " + ", ".join(SUBCOMMANDS[:-1]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="pierwsza wiadomość dla agenta")
    parser.add_argument("-m", "--model", metavar="KLUCZ", default=None,
                        help="model do użycia (fragment klucza wystarczy)")
    parser.add_argument("--mode", default=None,
                        choices=["default", "accept_edits", "auto", "yolo", "plan"],
                        help="tryb uprawnień agenta")
    parser.add_argument("--ctx", type=int, default=None, metavar="N",
                        help="rozmiar okna kontekstu przy ładowaniu modelu")
    # "on" is what a model with a plain thinking switch (Qwen3) shows in the
    # app and accepts over the API; it normalises to "medium".
    parser.add_argument("--effort", default=None,
                        choices=["off", "low", "medium", "high", "on", "ultracode"],
                        help="ile model ma myśleć (tylko modele, które to obsługują); "
                             "ultracode = wysokie + praca na wielu pomocnikach równolegle")
    parser.add_argument("-c", "--continue", dest="continue_last", action="store_true",
                        help="wznów ostatnią sesję w tym katalogu")
    parser.add_argument("--resume", nargs="?", const="", default=None, metavar="ID",
                        help="wznów sesję o podanym identyfikatorze (bez ID: wybór z listy)")
    parser.add_argument("-p", "--print", dest="print_prompt", nargs="?", const="", default=None,
                        metavar="PROMPT", help="jedna tura bez interfejsu, wynik na stdout")
    parser.add_argument("--output-format", default="text",
                        choices=["text", "json", "stream-json"],
                        help="format wyjścia dla -p (domyślnie text)")
    parser.add_argument("--no-session", action="store_true",
                        help="nie zapisuj transkryptu sesji na dysku")
    parser.add_argument("-V", "--version", action="version", version=f"LocalCode {VERSION}")
    _add_common(parser)
    return parser


def build_models_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode models",
                                     description="Katalog modeli: przeglądanie, instalacja, "
                                                 "usuwanie. Modele pobierane z Hugging Face.")
    _add_common(parser)
    sub = parser.add_subparsers(dest="action")
    listing = sub.add_parser("list", help="modele zainstalowane na dysku")
    listing.add_argument("--refresh", action="store_true", help="przeskanuj katalog modeli")
    search = sub.add_parser("search", help="szukaj modelu w katalogu")
    search.add_argument("query", nargs="+", help="czego szukać")
    search.add_argument("--format", dest="fmt", default="any", choices=["any", "gguf", "mlx"])
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--advanced", action="store_true",
                        help="pokaż identyfikatory repozytoriów i kwantyzacje")
    search.add_argument("--official", action="store_true",
                        help="tylko repozytoria wydawców modeli (bez konwersji społeczności)")
    search.add_argument("--min-trust", dest="min_trust", type=int, default=0,
                        metavar="0-100", help="pomiń modele o niższej ocenie zaufania")
    search.add_argument("--sort", default="polecane",
                        choices=["polecane", "popularne", "lubiane", "zaufanie", "oficjalne",
                                 "najmniejsze"],
                        help="kolejność: popularne | lubiane | zaufanie | oficjalne | najmniejsze")
    search.add_argument("--category", default=None,
                        choices=["kodowanie", "czat", "rozumowanie", "obrazy", "wizja",
                                 "embedding"],
                        help="pokaż tylko jedną kategorię")
    info = sub.add_parser("info", help="szczegóły modelu (lokalnego albo z Hugging Face)")
    info.add_argument("repo", help="klucz modelu na dysku albo repozytorium HF")
    download = sub.add_parser("download", help="pobierz konkretny plik (zaawansowane)")
    download.add_argument("repo", help="repozytorium Hugging Face")
    download.add_argument("quant", nargs="?", default=None, help="kwantyzacja, np. Q4_K_M")
    add = sub.add_parser("add", help="dodaj własny model z dysku (.gguf albo katalog MLX)")
    add.add_argument("path", help="ścieżka do pliku .gguf albo katalogu z modelem MLX")
    add.add_argument("--name", default=None, help="nazwa katalogu w bibliotece")
    add.add_argument("--move", action="store_true",
                     help="przenieś zamiast kopiować (oszczędza miejsce)")
    remove = sub.add_parser("rm", help="usuń model z dysku")
    remove.add_argument("key", help="klucz modelu")
    remove.add_argument("-y", "--yes", action="store_true", help="bez pytania o potwierdzenie")
    curated = sub.add_parser("curated", help="katalog modeli polecanych na ten komputer")
    curated.add_argument("--advanced", action="store_true",
                         help="pokaż identyfikatory repozytoriów i kwantyzacje")
    curated.add_argument("--category", default=None,
                         choices=["kodowanie", "czat", "rozumowanie", "obrazy", "wizja", "embedding"],
                         help="pokaż tylko jedną kategorię")
    curated.add_argument("--official", action="store_true",
                         help="tylko repozytoria wydawców modeli")
    install = sub.add_parser("install", help="zainstaluj model po nazwie")
    install.add_argument("name", nargs="+", help="nazwa modelu, np. „Qwen3 Coder”")
    install.add_argument("--quality", default=None,
                         choices=["4bit", "5bit", "6bit", "8bit", "max"],
                         help="poziom kwantyzacji (domyślnie z ustawień: 4bit)")
    install.add_argument("--variant", default=None,
                         help="wariant pliku (zaawansowane; domyślnie wybierany automatycznie)")
    install.add_argument("-y", "--yes", action="store_true",
                         help="nie pytaj, nawet jeśli model się nie zmieści")
    return parser


def build_image_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localcode image",
        description="Generuje obraz lokalnym modelem (FLUX, SD, Qwen-Image).")
    parser.add_argument("prompt", nargs="+", help="opis obrazu")
    parser.add_argument("-m", "--model", metavar="KLUCZ", default=None,
                        help="model obrazu (fragment klucza wystarczy)")
    parser.add_argument("--size", default=None, metavar="SZERxWYS",
                        help="rozmiar obrazu, np. 1024x1024 (domyślnie z ustawień)")
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="liczba kroków (domyślnie zależna od modelu)")
    parser.add_argument("--guidance", type=float, default=None, metavar="X",
                        help="siła podpowiedzi (guidance)")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="ziarno losowe (to samo ziarno = ten sam obraz)")
    parser.add_argument("--negative", default="", metavar="OPIS",
                        help="czego na obrazie ma nie być")
    parser.add_argument("-n", "--count", type=int, default=1, metavar="ILE",
                        help="ile obrazów wygenerować")
    parser.add_argument("-o", "--output", default=None, metavar="PLIK",
                        help="zapisz kopię obrazu pod tą ścieżką")
    parser.add_argument("--open", dest="open_file", action="store_true",
                        help="otwórz gotowy obraz w podglądzie")
    parser.add_argument("--no-preview", action="store_true",
                        help="nie rysuj podglądu w terminalu (iTerm2/kitty)")
    translate = parser.add_mutually_exclusive_group()
    translate.add_argument("--translate", dest="translate", action="store_true", default=None,
                           help="przetłumacz opis na angielski (modele graficzne rozumieją go lepiej)")
    translate.add_argument("--no-translate", dest="translate", action="store_false",
                           help="użyj opisu dokładnie tak, jak go wpisano")
    _add_common(parser)
    return parser


def build_images_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode images",
                                     description="Galeria wygenerowanych obrazów.")
    _add_common(parser)
    # Bare `localcode images` means `localcode images list`, so the listing
    # options must exist on the namespace even without the subcommand.
    parser.set_defaults(limit=20, favorite=False)
    sub = parser.add_subparsers(dest="action")
    listing = sub.add_parser("list", help="ostatnio wygenerowane obrazy")
    listing.add_argument("--limit", type=int, default=20, help="ile pozycji pokazać")
    listing.add_argument("--favorite", action="store_true", help="tylko ulubione")
    remove = sub.add_parser("rm", help="usuń obraz (plik i wpis w galerii)")
    remove.add_argument("id", help="identyfikator obrazu (wystarczy początek)")
    remove.add_argument("-y", "--yes", action="store_true", help="bez pytania o potwierdzenie")
    opener = sub.add_parser("open", help="otwórz obraz w podglądzie")
    opener.add_argument("id", help="identyfikator obrazu (wystarczy początek)")
    return parser


def build_runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localcode runtime",
                                     description="Ile modeli naraz, ile pamięci, jaki kontekst.")
    _add_common(parser)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("show", help="pokaż ustawienia i zajętość pamięci")
    setter = sub.add_parser("set", help="zmień ustawienie")
    setter.add_argument("key", help="max-models, instances, parallel, ram-gb, idle-ttl, ctx, "
                                    "kv, flash-attn, gpu-layers, evict")
    setter.add_argument("value", nargs="+", help="nowa wartość (auto = domyślna)")
    tuner = sub.add_parser("tune", help="dobierz ustawienia do tego komputera i (opcjonalnie) modelu")
    tuner.add_argument("--apply", action="store_true", help="od razu zapisz zaproponowane ustawienia")
    tuner.add_argument("--model", metavar="KLUCZ", default=None,
                       help="model, dla którego dobrać kontekst i limity (fragment klucza wystarczy)")
    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a command line; returns the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    # `localcode` may be run by launchd or a GUI with macOS's minimal PATH too.
    from localcode.system_path import ensure_tool_path

    ensure_tool_path()
    if args and args[0] == "serve":
        from localcode.server.main import serve_main

        return serve_main(args)
    if args and args[0] in ("help", "--help", "-h") and len(args) > 1:
        return main([args[1], "--help"])
    if args and args[0] == "help":
        build_parser().print_help()
        return 0
    if args and args[0] == "app":
        return open_mac_app(args[1:])
    if args and args[0] == "doctor":
        from localcode.doctor import main as doctor_main

        return doctor_main(args[1:])
    try:
        if args and args[0] in SUBCOMMANDS:
            return asyncio.run(_run_subcommand(args[0], args[1:]))
        options = build_parser().parse_args(args)
        return asyncio.run(_run_root(options))
    except KeyboardInterrupt:
        print("\nPrzerwano.", file=sys.stderr)
        return 130
    except SystemExit as exc:  # argparse already printed its own message
        code = exc.code
        if isinstance(code, str):
            print(code, file=sys.stderr)
            return 1
        return int(code or 0)


def _prefer(options: argparse.Namespace) -> str:
    if getattr(options, "local", False):
        return "local"
    if getattr(options, "attach", False):
        return "remote"
    return "auto"


def _cwd(options: argparse.Namespace) -> Path:
    path = Path(getattr(options, "cwd", None) or ".").expanduser()
    try:
        return path.resolve(strict=True)
    except OSError:
        raise SystemExit(f"Katalog „{path}” nie istnieje.") from None


async def _open(options: argparse.Namespace, **kwargs: Any) -> Any:
    from localcode.tui.backends import open_backend

    return await open_backend(cwd=_cwd(options), prefer=_prefer(options),
                              settings=load_settings(), **kwargs)


# --------------------------------------------------------------------------- #
# localcode [PROMPT] / -p
# --------------------------------------------------------------------------- #


async def _run_root(options: argparse.Namespace) -> int:
    from localcode.tui.backends import BackendError

    headless = options.print_prompt is not None
    prompt = (options.print_prompt or " ".join(options.prompt) or "").strip()
    if headless and not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if headless and not prompt:
        print("Podaj treść polecenia: localcode -p \"…\"", file=sys.stderr)
        return 2

    session_id = await _session_to_resume(options)
    if session_id is False:
        return 1
    try:
        backend = await _open(
            options,
            model_key=options.model,
            mode=options.mode,
            n_ctx=options.ctx,
            session_id=session_id or None,
            store_session=not options.no_session and not headless,
            ask_permission=not headless,
        )
    except BackendError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        if headless:
            from localcode.tui.headless import run_headless

            if not backend.model_key:
                print("Brak modelu. Pobierz jakiś: localcode models curated", file=sys.stderr)
                return 1
            return await run_headless(backend, prompt, output_format=options.output_format,
                                      effort=options.effort)
        from localcode.tui.app import run_tui

        if options.effort:
            try:
                await backend.set_effort(options.effort)
            except Exception as exc:                      # model without a thinking switch
                print(f"Nie udało się ustawić poziomu myślenia: {exc}", file=sys.stderr)
        return await run_tui(backend, first=" ".join(options.prompt) or None)
    finally:
        await backend.aclose()


async def _session_to_resume(options: argparse.Namespace) -> str | bool | None:
    """``-c`` → the newest session here; ``--resume ID`` → that one; ``--resume`` → pick."""
    from localcode.agent import session as session_mod

    if not options.continue_last and options.resume is None:
        return None
    if options.resume:
        return options.resume
    sessions = await asyncio.to_thread(session_mod.list_sessions, str(_cwd(options)), limit=20)
    if not sessions:
        print("Brak zapisanych sesji w tym katalogu.", file=sys.stderr)
        return None
    if options.continue_last:
        return sessions[0].id
    if not sys.stdin.isatty():
        return sessions[0].id
    from localcode.tui.prompts import Choice, choose

    picked = await choose(
        [Choice(s.title or s.id, value=s.id, hint=s.model_key or "") for s in sessions],
        title="Wznów sesję")
    if picked is None:
        return False
    return str(picked)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


async def _run_subcommand(name: str, args: list[str]) -> int:
    from localcode.tui.backends import BackendError

    try:
        if name == "chat":
            return await _cmd_chat(args)
        if name == "pomoc":
            return await _cmd_helpbot(args)
        if name == "helpbot":
            return await _cmd_helpbot(args)
        if name == "models":
            return await _cmd_models(args)
        if name == "image":
            return await _cmd_image(args)
        if name == "images":
            return await _cmd_images(args)
        if name == "video":
            return await _cmd_video(args)
        if name == "transcribe":
            return await _cmd_transcribe(args)
        if name == "say":
            return await _cmd_say(args)
        if name == "skills":
            return await _cmd_skills(args)
        if name == "ps":
            return await _cmd_ps(args)
        if name == "load":
            return await _cmd_load(args)
        if name == "unload":
            return await _cmd_unload(args)
        if name == "runtime":
            return await _cmd_runtime(args)
    except BackendError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 2  # pragma: no cover - guarded by SUBCOMMANDS


def _console() -> Any:
    from localcode.tui.stream import make_console

    return make_console()


async def _cmd_chat(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcode chat",
                                     description="Czat bez narzędzi; rozmowy widać w aplikacji.")
    parser.add_argument("-m", "--model", default=None, help="model do rozmowy")
    parser.add_argument("message", nargs="*", help="pierwsza wiadomość")
    _add_common(parser)
    options = parser.parse_args(args)
    backend = await _open(options, model_key=options.model, store_session=False,
                          attach_agent=False)
    try:
        from localcode.tui.app import run_tui

        return await run_tui(backend, chat_mode=True, console=_console(),
                             first=" ".join(options.message) or None)
    finally:
        await backend.aclose()


async def _cmd_helpbot(args: list[str]) -> int:
    """``localcode pomoc`` — the help bot, in the terminal.

    The same conversation as the app's: both work in the help bot's directory,
    so a question asked here can be continued there. Without the model on disk
    it is downloaded first, with the store's own progress bar.
    """
    from localcode import helpbot
    from localcode.agent import session as session_mod
    from localcode.hub.library import ModelLibrary

    parser = argparse.ArgumentParser(
        prog="localcode pomoc",
        description="Pomocnik: mały model, który za Ciebie znajduje, pobiera, ustawia, ładuje "
                    "i usuwa modele oraz zmienia ustawienia aplikacji. Wszystko, co coś "
                    "zmienia, najpierw pokazuje do zatwierdzenia.")
    parser.add_argument("message", nargs="*", help="pierwsze pytanie, np. „pobierz mi coś do kodu”")
    parser.add_argument("--install", action="store_true", help="tylko pobierz model pomocnika")
    parser.add_argument("--new", action="store_true",
                        help="zacznij nową rozmowę (domyślnie wznawia ostatnią)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true",
                       help="uruchom pomocnika w tym procesie, nawet gdy aplikacja działa")
    group.add_argument("--attach", action="store_true",
                       help="wymagaj działającej aplikacji LocalCode")
    options = parser.parse_args(args)

    settings = load_settings()
    wanted = helpbot.model_key(settings)
    library = ModelLibrary(settings)
    model = await asyncio.to_thread(lambda: library.get(wanted) or library.resolve(wanted))
    if model is None:
        problem = helpbot.room_problem(settings)
        if problem:
            print(problem, file=sys.stderr)
            return 1
        print(f"Pobieram model pomocnika ({wanted}, ok. 2,3 GB) — tylko za pierwszym razem.")
        code = await _cmd_models(["download", wanted])
        if code != 0:
            return code
        model = await asyncio.to_thread(lambda: library.get(wanted) or library.resolve(wanted))
        if model is None:
            print("Model pomocnika się nie pobrał. Spróbuj: localcode pomoc --install", file=sys.stderr)
            return 1
    if options.install:
        print(f"Pomocnik gotowy: {model.display_name}. Zapytaj go: localcode pomoc")
        return 0

    cwd = helpbot.helpbot_dir()
    resume = None
    if not options.new:
        recent = await asyncio.to_thread(session_mod.list_sessions, cwd, limit=1)
        resume = recent[0].id if recent else None
    backend = await _open_at(options, cwd, model_key=model.key, session_id=resume,
                             mode="default", store_session=True)
    try:
        from localcode.tui.app import run_tui

        return await run_tui(backend, first=" ".join(options.message) or None)
    finally:
        await backend.aclose()


async def _open_at(options: argparse.Namespace, cwd: Any, **kwargs: Any) -> Any:
    from localcode.tui.backends import open_backend

    return await open_backend(cwd=str(cwd), prefer=_prefer(options),
                              settings=load_settings(), **kwargs)


async def _cmd_models(args: list[str]) -> int:
    from localcode.tui import models_ui, render

    options = build_models_parser().parse_args(args)
    action = options.action or "list"
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    try:
        if action == "list":
            models = await backend.list_models(refresh=bool(getattr(options, "refresh", False)))
            if not models:
                console.print("Brak modeli. Pobierz pierwszy: [lc.key]localcode models curated[/]")
                return 0
            status = await backend.runtime_status()
            console.print(render.models_table(
                models, budget_bytes=int((status.get("budget") or {}).get("limitBytes") or 0)))
            return 0
        if action == "add":
            from localcode.hub.library import LibraryError, ModelLibrary

            library = ModelLibrary(load_settings())
            try:
                model = await asyncio.to_thread(
                    library.import_path, options.path, name=options.name, move=options.move)
            except LibraryError as exc:
                console.print(f"[lc.err]{exc}[/]")
                return 1
            console.print(f"[lc.ok]Dodano[/] {model.display_name} "
                          f"({model.size_bytes / 1e9:.1f} GB) jako [lc.key]{model.key}[/]")
            scan = model.scan or {}
            if scan.get("verdict") == "dangerous":
                console.print(f"[lc.err]Uwaga:[/] {scan.get('summary')}")
            elif scan.get("verdict") == "suspicious":
                console.print(f"[lc.warn]Uwaga:[/] {scan.get('summary')}")
            await backend.list_models(refresh=True)
            return 0
        if action in ("search", "curated", "install"):
            catalog = _catalog()
        if action == "search":
            query = " ".join(options.query)
            entries = await asyncio.to_thread(
                catalog.entries, query=query, category=options.category,
                limit=options.limit, min_trust=options.min_trust,
                official_only=options.official, sort=options.sort)
            if not entries:
                console.print("Nic nie znaleziono.")
                if catalog.notice:
                    console.print(f"[lc.dim]{catalog.notice}[/]")
                return 1
            console.print(models_ui.catalog_table(entries, advanced=options.advanced))
            if catalog.total_matches > len(entries):
                console.print(f"[lc.dim]Pokazuję {len(entries)} z {catalog.total_matches} "
                              f"pasujących modeli — zawęź zapytanie albo podnieś --limit.[/]")
            if catalog.notice:
                console.print(f"[lc.dim]{catalog.notice}[/]")
            console.print("[lc.dim]Instalacja: [/][lc.key]localcode models install \"<nazwa>\"[/]")
            console.print(models_ui.catalog_footer())
            return 0
        if action == "curated":
            entries = await asyncio.to_thread(catalog.entries, category=options.category,
                                              official_only=options.official)
            entries = [e for e in entries if e.get("curated")]
            console.print(models_ui.catalog_table(entries, advanced=options.advanced))
            if catalog.notice:
                console.print(f"[lc.dim]{catalog.notice}[/]")
            console.print("[lc.dim]Instalacja: [/][lc.key]localcode models install \"<nazwa>\"[/]")
            console.print(models_ui.catalog_footer())
            return 0
        if action == "install":
            if getattr(options, "quality", None):
                catalog.settings.store.quality = options.quality
            key = await models_ui.install_from_catalog(
                catalog, backend, console, " ".join(options.name),
                variant=options.variant, assume_yes=options.yes)
            if key:
                console.print(f"[lc.ok]Gotowe.[/] "
                              f"{models_ui.next_step_hint(await backend.model(key), key)}")
                return 0
            return 1
        if action == "info":
            local = await backend.model(options.repo)
            if local is not None:
                await models_ui.show_model_details(backend, console, str(local["key"]))
                return 0
            details = await backend.hub_details(options.repo)
            _print_hub_details(console, details)
            return 0
        if action == "download":
            key = await models_ui.download_model(backend, console, options.repo, options.quant)
            if key:
                console.print(f"[lc.ok]Gotowe.[/] "
                              f"{models_ui.next_step_hint(await backend.model(key), key)}")
                return 0
            return 1
        if action == "rm":
            key = await backend.resolve_model_key(options.key)
            if key is None:
                console.print(f"[lc.err]Nie znaleziono modelu „{options.key}”.[/]")
                return 1
            if not options.yes and sys.stdin.isatty():
                answer = input(f"Usunąć {key} z dysku? [t/N] ").strip().lower()
                if answer not in ("t", "tak", "y", "yes"):
                    console.print("Anulowano.")
                    return 1
            await backend.delete_model(key)
            console.print(f"[lc.ok]Usunięto {key}.[/]")
            return 0
    finally:
        await backend.aclose()
    return 2  # pragma: no cover - argparse covers the rest


def _catalog() -> Any:
    """The in-app model catalogue for this machine (see ``localcode.hub.catalog``)."""
    from localcode.hub.catalog import Catalog

    return Catalog(load_settings())


def _print_hub_details(console: Any, details: dict[str, Any]) -> None:
    from rich.table import Table

    from localcode.tui import models_ui, render

    model = details.get("model") or {}
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="lc.dim", justify="right")
    grid.add_column(overflow="fold")
    grid.add_row("repozytorium", str(model.get("id") or ""))
    grid.add_row("format", str(model.get("format") or "").upper())
    if model.get("paramsB"):
        grid.add_row("parametry", f"{render.fmt_number(model['paramsB'])} mld")
    if details.get("arch"):
        grid.add_row("architektura", str(details["arch"]))
    if details.get("contextLength"):
        grid.add_row("kontekst", render.fmt_tokens(details["contextLength"]))
    if model.get("description"):
        grid.add_row("opis", str(model["description"]))
    console.print(grid)
    variants = [v for v in (details.get("variants") or []) if not v.get("isVisionProjector")]
    if not variants:
        return
    table = Table(border_style="lc.dim")
    table.add_column("wariant")
    table.add_column("rozmiar", justify="right")
    table.add_column("pamięć", justify="right")
    table.add_column("zmieści się")
    for variant in variants:
        estimate = variant.get("estimate") or {}
        table.add_row(
            str(variant.get("label") or variant.get("quant")) + (" ★" if variant.get("recommended") else ""),
            render.fmt_bytes(variant.get("sizeBytes")),
            render.fmt_bytes(estimate.get("totalBytes")) if estimate else "—",
            models_ui.FITS_LABELS.get(str(estimate.get("fits")), "—"),
        )
    console.print(table)


async def _cmd_skills(args: list[str]) -> int:
    """``localcode skills`` — what the agent can load, and how to add one."""
    from localcode.agent.skills import SkillError, SkillLibrary

    parser = argparse.ArgumentParser(
        prog="localcode skills",
        description="Skille: instrukcje, które agent wczytuje wtedy, kiedy ich potrzebuje.")
    _add_common(parser)
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("list", help="pokaż dostępne skille")
    show = sub.add_parser("show", help="wypisz treść skilla")
    show.add_argument("name")
    new_skill = sub.add_parser("new", help="utwórz nowy skill")
    new_skill.add_argument("name")
    new_skill.add_argument("description", nargs="+", help="jedno zdanie: kiedy go użyć")
    new_skill.add_argument("--project", action="store_true",
                           help="zapisz w projekcie (.localcode/skills) zamiast u siebie")
    options = parser.parse_args(args)

    console = _console()
    library = SkillLibrary(options.cwd or Path.cwd())
    action = options.action or "list"

    if action == "new":
        try:
            skill = library.create(options.name, " ".join(options.description),
                                   scope="project" if options.project else "user")
        except SkillError as exc:
            console.print(f"[lc.err]{exc}[/]")
            return 1
        console.print(f"[lc.ok]Utworzono skill[/] [lc.key]{skill.name}[/] → {skill.path}")
        console.print("[lc.dim]Wpisz instrukcje do tego pliku — agent wczyta je, gdy będą "
                      "potrzebne.[/]")
        return 0

    if action == "show":
        skill = library.get(options.name)
        if skill is None:
            console.print(f"[lc.err]Nie ma skilla „{options.name}”.[/]")
            return 1
        console.print(f"[lc.accent]{skill.name}[/] [lc.dim]({skill.scope} · {skill.path})[/]")
        console.print(skill.body)
        return 0

    skills = library.all()
    if not skills:
        console.print("Nie masz jeszcze żadnego skilla.")
        console.print("[lc.dim]Utwórz pierwszy: [/][lc.key]localcode skills new testy "
                      "\"Jak uruchamiać testy w tym repo\"[/]")
        return 0
    for skill in skills:
        where = "projekt" if skill.scope == "project" else "twoje"
        console.print(f"[lc.key]{skill.name}[/] [lc.dim]({where})[/]\n  {skill.description}")
    for error in library.errors:
        console.print(f"[lc.warn]Pominięto:[/] [lc.dim]{error}[/]")
    return 0


async def _cmd_transcribe(args: list[str]) -> int:
    """``localcode transcribe nagranie.m4a`` — a recording turned into text."""
    from localcode.speech import SpeechError, SpeechRuntime

    parser = argparse.ArgumentParser(prog="localcode transcribe",
                                     description="Zamień nagranie na tekst (model z kategorii „Mowa”).")
    _add_common(parser)
    parser.add_argument("audio", help="plik z nagraniem (wav, mp3, m4a, mp4…)")
    parser.add_argument("--model", default=None, help="klucz modelu; domyślnie pierwszy do mowy")
    parser.add_argument("--language", default=None, help="kod języka, np. pl (domyślnie wykryj)")
    parser.add_argument("--srt", metavar="PLIK", default=None, help="zapisz napisy do pliku .srt")
    parser.add_argument("--out", metavar="PLIK", default=None, help="zapisz tekst do pliku")
    options = parser.parse_args(args)

    console = _console()
    runtime = SpeechRuntime(load_settings())
    try:
        transcript = await runtime.transcribe(options.audio, model_key=options.model,
                                              language=options.language)
    except SpeechError as exc:
        console.print(f"[lc.err]{exc}[/]")
        return 1
    if options.out:
        Path(options.out).expanduser().write_text(transcript.text + "\n", encoding="utf-8")
        console.print(f"[lc.ok]Zapisano tekst:[/] {options.out}")
    else:
        console.print(transcript.text)
    if options.srt:
        Path(options.srt).expanduser().write_text(transcript.to_srt(), encoding="utf-8")
        console.print(f"[lc.ok]Zapisano napisy:[/] {options.srt}")
    console.print(f"[lc.dim]{transcript.model_key} · {transcript.elapsed_ms / 1000:.1f} s"
                  f"{f' · język {transcript.language}' if transcript.language else ''}[/]")
    return 0


async def _cmd_say(args: list[str]) -> int:
    """``localcode say "tekst"`` — a wave file read by a local voice."""
    from localcode.speech import SpeechError, SpeechRuntime

    parser = argparse.ArgumentParser(prog="localcode say",
                                     description="Przeczytaj tekst na głos (model z kategorii „Mowa”).")
    _add_common(parser)
    parser.add_argument("text", nargs="+", help="tekst do przeczytania")
    parser.add_argument("--model", default=None, help="klucz modelu; domyślnie pierwszy do mowy")
    parser.add_argument("--voice", default=None, help="głos, np. af_heart")
    parser.add_argument("--speed", type=float, default=1.0, help="tempo 0,5–2,0")
    parser.add_argument("--out", metavar="PLIK", default=None, help="gdzie zapisać .wav")
    parser.add_argument("--play", action="store_true", help="odtwórz od razu")
    options = parser.parse_args(args)

    console = _console()
    runtime = SpeechRuntime(load_settings())
    try:
        result = await runtime.speak(" ".join(options.text), model_key=options.model,
                                     voice=options.voice, speed=options.speed,
                                     output=options.out)
    except SpeechError as exc:
        console.print(f"[lc.err]{exc}[/]")
        return 1
    console.print(f"[lc.ok]Gotowe:[/] {result.path}"
                  f"[lc.dim] · {result.voice} · {result.elapsed_ms / 1000:.1f} s[/]")
    if options.play:
        subprocess.run(["afplay", result.path], check=False)
    return 0


async def _cmd_video(args: list[str]) -> int:
    """``localcode video "opis"`` — a short clip from a local LTX-Video 2B model."""
    from rich.markup import escape

    from localcode.video import VideoError, VideoRuntime
    from localcode.video.base import MAX_FRAMES_DEFAULT, MAX_FRAMES_HARD

    parser = argparse.ArgumentParser(
        prog="localcode video",
        description="Wygeneruj krótki klip wideo lokalnym modelem (na razie tylko LTX-Video 2B; "
                    "zobacz docs/VIDEO.md — generowanie jest wolne, liczone w minutach).")
    _add_common(parser)
    parser.add_argument("prompt", nargs="+", help="opis materiału wideo")
    parser.add_argument("-m", "--model", default=None,
                        help="klucz modelu; domyślnie pierwszy obsługiwany model wideo")
    parser.add_argument("--size", default=None, metavar="SZERxWYS",
                        help="rozdzielczość, wielokrotność 32 (domyślnie 704x480)")
    parser.add_argument("--frames", type=int, default=None, metavar="N",
                        help=f"liczba klatek (domyślnie 41; limit {MAX_FRAMES_DEFAULT} bez --force, "
                             f"maksimum {MAX_FRAMES_HARD})")
    parser.add_argument("--fps", type=int, default=None, metavar="N",
                        help="klatki na sekundę pliku wyjściowego (domyślnie 24)")
    parser.add_argument("--steps", type=int, default=None, metavar="N",
                        help="liczba kroków odszumiania (domyślnie 30)")
    parser.add_argument("--guidance", type=float, default=None, metavar="X",
                        help="siła podpowiedzi (domyślnie 3,0)")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="ziarno losowe (to samo ziarno = ten sam klip)")
    parser.add_argument("--negative", default=None, metavar="OPIS",
                        help="czego na wideo ma nie być")
    parser.add_argument("-o", "--output", default=None, metavar="PLIK",
                        help="ścieżka pliku wynikowego .mp4")
    parser.add_argument("--force", action="store_true",
                        help=f"pozwól na więcej niż {MAX_FRAMES_DEFAULT} klatek (do {MAX_FRAMES_HARD})")
    options = parser.parse_args(args)

    prompt = " ".join(options.prompt).strip()
    if not prompt:
        print('Podaj opis wideo: localcode video "kot w kapeluszu idzie ulicą"', file=sys.stderr)
        return 2
    try:
        width, height = _parse_size(options.size)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    console = _console()
    runtime = VideoRuntime(load_settings())
    try:
        preview = runtime.estimate(width=width, height=height, num_frames=options.frames,
                                   fps=options.fps, steps=options.steps)
    except VideoError as exc:
        console.print(f"[lc.err]{escape(str(exc))}[/]")
        return 1
    console.print(f"[lc.dim]{escape(preview['message'])}[/]")

    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                 BarColumn(), TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                 console=console, transient=True) as bar:
        task_id = bar.add_task("Przygotowuję…", total=None)

        def on_progress(p: Any) -> None:
            bar.update(task_id, description=escape(p.message or p.stage), completed=p.step,
                       total=p.steps or None)

        try:
            result = await runtime.generate(
                prompt, model_key=options.model, width=width, height=height,
                num_frames=options.frames, fps=options.fps, steps=options.steps,
                guidance=options.guidance, seed=options.seed, negative_prompt=options.negative,
                force=options.force, output=options.output, on_progress=on_progress)
        except VideoError as exc:
            console.print(f"[lc.err]{escape(str(exc))}[/]")
            return 1

    console.print(f"[lc.ok]Gotowe:[/] {escape(result.path)}"
                  f"[lc.dim] · {result.width}x{result.height} · {result.num_frames} klatek "
                  f"· {result.elapsed_ms / 1000:.1f} s[/]")
    return 0


async def _cmd_image(args: list[str]) -> int:
    """``localcode image "opis" [-m MODEL] [--size WxH] …``."""
    from localcode.tui import images_ui

    options = build_image_parser().parse_args(args)
    prompt = " ".join(options.prompt).strip()
    if not prompt:
        print("Podaj opis obrazu: localcode image \"kot w kapeluszu\"", file=sys.stderr)
        return 2
    try:
        width, height = _parse_size(options.size)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if options.count < 1:
        print("Liczba obrazów musi być dodatnia.", file=sys.stderr)
        return 2
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    client = images_ui.image_client(backend)
    try:
        images = await images_ui.generate_with_progress(
            client, console, prompt, model_key=options.model, width=width, height=height,
            steps=options.steps, guidance=options.guidance, seed=options.seed,
            count=options.count, negative_prompt=options.negative,
            translate=options.translate)
    finally:
        await client.aclose()
        await backend.aclose()
    if not images:
        console.print("[lc.warn]Nie powstał żaden obraz.[/]")
        return 1
    original = str(images[0].get("originalPrompt") or "")
    if original:
        console.print(f"[lc.dim]Opis przetłumaczony na angielski: "
                      f"{images[0].get('prompt', '')}[/]")
    images_ui.print_results(console, images, preview=not options.no_preview)
    if options.output:
        for index, image in enumerate(images):
            saved = images_ui.save_copy(image, options.output, index=index, total=len(images))
            console.print(f"[lc.ok]Zapisano[/] {saved}")
    if options.open_file and not images_ui.open_in_finder(str(images[0].get("path") or "")):
        console.print("[lc.warn]Nie udało się otworzyć podglądu.[/]")
    return 0


async def _cmd_images(args: list[str]) -> int:
    """``localcode images [list|rm ID|open ID]``."""
    from localcode.tui import images_ui

    options = build_images_parser().parse_args(args)
    action = options.action or "list"
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    client = images_ui.image_client(backend)
    try:
        if action == "list":
            images = await client.list(limit=max(1, options.limit), favorite=options.favorite)
            if not images:
                console.print("Galeria jest pusta. Wygeneruj obraz: "
                              "[lc.key]localcode image \"kot w kapeluszu\"[/]")
                return 0
            console.print(images_ui.images_table(images))
            return 0
        if action == "rm":
            image = await client.resolve(options.id)
            if not options.yes and sys.stdin.isatty():
                answer = input(f"Usunąć obraz {str(image.get('id'))[:8]} z dysku? [t/N] ").strip().lower()
                if answer not in ("t", "tak", "y", "yes"):
                    console.print("Anulowano.")
                    return 1
            await client.delete(str(image.get("id")))
            console.print(f"[lc.ok]Usunięto obraz {str(image.get('id'))[:8]}.[/]")
            return 0
        if action == "open":
            image = await client.resolve(options.id)
            path = str(image.get("path") or "")
            if not images_ui.open_in_finder(path):
                console.print(f"[lc.err]Nie udało się otworzyć pliku {path}.[/]")
                return 1
            console.print(f"Otwieram {path}")
            return 0
    finally:
        await client.aclose()
        await backend.aclose()
    return 2  # pragma: no cover - argparse covers the rest


def _parse_size(text: str | None) -> tuple[int | None, int | None]:
    """``"1024x768"`` → ``(1024, 768)``; ``None`` leaves the model/settings default."""
    if not text or not text.strip():
        return None, None
    raw = text.strip().lower().replace("×", "x").replace("*", "x")
    left, _, right = raw.partition("x")
    try:
        return int(left.strip()), int(right.strip())
    except ValueError:
        raise ValueError(f"Nieprawidłowy rozmiar „{text}”. Podaj np. 1024x1024.") from None


async def _cmd_ps(args: list[str]) -> int:
    from localcode.tui import render

    parser = argparse.ArgumentParser(prog="localcode ps", description="Załadowane modele.")
    _add_common(parser)
    options = parser.parse_args(args)
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    try:
        status = await backend.runtime_status()
        console.print(render.instances_table(status))
        system = status.get("system") or {}
        budget = status.get("budget") or {}
        console.print(f"[lc.dim]Pamięć: {render.fmt_bytes(budget.get('usedBytes'))} z "
                      f"{render.fmt_bytes(budget.get('limitBytes'))} budżetu · "
                      f"system: {render.fmt_bytes(system.get('availableBytes'))} wolne[/]")
        return 0
    finally:
        await backend.aclose()


async def _cmd_load(args: list[str]) -> int:
    from localcode.tui import render

    parser = argparse.ArgumentParser(prog="localcode load", description="Załaduj model do pamięci.")
    parser.add_argument("key", help="klucz modelu (fragment wystarczy)")
    parser.add_argument("--ctx", type=int, default=None, help="rozmiar kontekstu")
    parser.add_argument("--parallel", type=int, default=None, help="równoległe żądania")
    parser.add_argument("--new", action="store_true", help="dodatkowa kopia modelu")
    _add_common(parser)
    options = parser.parse_args(args)
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    try:
        with console.status(f"Ładuję {options.key}…", spinner="dots"):
            instance = await backend.load_model(options.key, n_ctx=options.ctx,
                                                n_parallel=options.parallel,
                                                new_instance=options.new)
        console.print(f"[lc.ok]{instance.get('displayName') or options.key}[/] — "
                      f"{render.fmt_tokens(instance.get('nCtx'))} kontekstu, "
                      f"{instance.get('nParallel')} sloty, "
                      f"{render.fmt_bytes(instance.get('memoryBytes'))}, id {instance.get('id')}")
        return 0
    finally:
        await backend.aclose()


async def _cmd_unload(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="localcode unload", description="Zwolnij pamięć.")
    parser.add_argument("instance", nargs="?", default=None, help="identyfikator instancji")
    parser.add_argument("--all", action="store_true", help="wszystkie instancje")
    _add_common(parser)
    options = parser.parse_args(args)
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    try:
        await backend.unload(options.instance, everything=options.all or not options.instance)
        console.print("[lc.ok]Zwolniono pamięć.[/]")
        return 0
    finally:
        await backend.aclose()


async def _cmd_runtime(args: list[str]) -> int:
    from localcode.tui import render
    from localcode.tui.commands import runtime_patch

    options = build_runtime_parser().parse_args(args)
    console = _console()
    backend = await _open(options, store_session=False, attach_agent=False)
    try:
        action = options.action or "show"
        if action == "set":
            try:
                patch = runtime_patch(options.key, " ".join(options.value))
            except ValueError as exc:
                console.print(f"[lc.err]{exc}[/]")
                return 2
            status = await backend.update_runtime_settings(patch)
            console.print("[lc.ok]Zapisano.[/]")
            console.print(render.runtime_panel(status))
            return 0
        if action == "tune":
            model_key = None
            if options.model:
                model_key = await backend.resolve_model_key(options.model)
                if model_key is None:
                    console.print(f"[lc.err]Nie znaleziono modelu „{options.model}”.[/]")
                    return 1
            result = await backend.tune_runtime(model_key=model_key, apply=options.apply)
            console.print(render.tune_table(result["recommendation"]))
            if options.apply:
                console.print("[lc.ok]Zapisano zaproponowane ustawienia.[/]")
            return 0
        status = await backend.runtime_status()
        console.print(render.runtime_panel(status))
        return 0
    finally:
        await backend.aclose()


# --------------------------------------------------------------------------- #
# localcode app
# --------------------------------------------------------------------------- #


def find_mac_app() -> Path | None:
    """The LocalCode.app bundle, if it was built or installed."""
    for candidate in APP_LOCATIONS:
        if candidate.exists():
            return candidate
    return None


def open_mac_app(args: Sequence[str] = ()) -> int:
    """``localcode app`` — open the SwiftUI application."""
    parser = argparse.ArgumentParser(prog="localcode app",
                                     description="Otwiera aplikację LocalCode na Maca.")
    parser.add_argument("--path", default=None, help="ścieżka do LocalCode.app")
    options = parser.parse_args(list(args))
    bundle = Path(options.path).expanduser() if options.path else find_mac_app()
    if bundle is None or not bundle.exists():
        print("Nie znalazłem aplikacji LocalCode.app.", file=sys.stderr)
        print("Zbuduj ją: ./macos/scripts/build-app.sh  (albo ./scripts/install.sh)",
              file=sys.stderr)
        return 1
    if sys.platform != "darwin":
        print("Aplikacja działa tylko na macOS.", file=sys.stderr)
        return 1
    try:
        subprocess.run(["open", str(bundle)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Nie udało się otworzyć aplikacji: {exc}", file=sys.stderr)
        return 1
    print(f"Otwieram {bundle}")
    return 0


if __name__ == "__main__":  # pragma: no cover - `python -m localcode.cli`
    raise SystemExit(main())
