"""Slash commands of the terminal app: the table, parsing and the help screen.

The registry is data only — :mod:`localcode.tui.app` maps a name to a handler.
Keeping it here means the completer, ``/help`` and the tests all read the same
list, so a command can never be offered but not implemented (there is a test
asserting exactly that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from rich.box import ROUNDED
from rich.table import Table

__all__ = [
    "COMMANDS",
    "KEY_BINDINGS",
    "RUNTIME_SETTING_KEYS",
    "Command",
    "ParsedCommand",
    "command_names",
    "find_command",
    "help_table",
    "is_command",
    "keys_table",
    "parse_command",
    "runtime_patch",
    "suggest",
]


@dataclass(frozen=True)
class Command:
    """One slash command as offered by the completer and ``/help``."""

    name: str
    help: str
    usage: str = ""
    aliases: tuple[str, ...] = ()
    #: also available in chatbot mode (``/chat``)
    in_chat: bool = False
    #: also offered by the Mac app's "Kod" section.
    #:
    #: Not every command travels: the app has whole screens for the model store,
    #: the runtime and images, and ``/exit`` or ``/chat`` mean nothing in a
    #: window with a sidebar. The ones marked here are the agent's own, and the
    #: app reads this table over HTTP rather than keeping a second copy — that
    #: second copy is exactly how the two interfaces drifted to 4 commands
    #: against 23.
    in_app: bool = False
    #: argument values the completer offers after the command
    values: tuple[str, ...] = field(default=())

    @property
    def display(self) -> str:
        return self.usage or f"/{self.name}"


COMMANDS: tuple[Command, ...] = (
    Command("help", "lista poleceń i skrótów klawiszowych", aliases=("?",), in_chat=True, in_app=True),
    Command("model", "wybierz model (bez argumentu: lista z rozmiarem i kwantyzacją)",
            usage="/model [klucz]", in_chat=True, in_app=True),
    Command("models", "modele zainstalowane na dysku", aliases=("ls",), in_chat=True),
    Command("search", "szukaj modeli na Hugging Face", usage="/search zapytanie", in_chat=True),
    Command("download", "pobierz model z Hugging Face", usage="/download repo [KWANT]", in_chat=True),
    Command("load", "załaduj model do pamięci", usage="/load [klucz]", in_chat=True),
    Command("unload", "wyładuj model (bez argumentu: wszystkie)", usage="/unload [id|all]", in_chat=True),
    Command("ps", "załadowane modele i zużycie pamięci", in_chat=True),
    Command("runtime", "ustawienia środowiska modeli", usage="/runtime [set klucz wartość]", in_chat=True),
    Command("obraz", "wygeneruj obraz lokalnym modelem", usage="/obraz opis obrazu",
            in_chat=True),
    Command("mode", "tryb uprawnień agenta", usage="/mode [default|accept_edits|auto|plan|yolo]",
            values=("default", "accept_edits", "auto", "plan", "yolo"), in_app=True),
    Command("think", "poziom myślenia modelu (bez argumentu: następny)",
            usage="/think [off|low|medium|high|ultracode]",
            values=("off", "low", "medium", "high", "ultracode"), in_chat=True, in_app=True),
    Command("goal", "pracuj, aż cel zostanie osiągnięty (bez argumentu: stan celu)",
            usage="/goal [cel | wyczyść]", in_app=True, values=("wyczyść",)),
    Command("compact", "skróć rozmowę teraz (podsumowanie starszych tur)", in_chat=True, in_app=True),
    Command("clear", "wyczyść rozmowę i kontekst", aliases=("new",), in_chat=True, in_app=True),
    Command("context", "co zajmuje kontekst modelu", in_chat=True, in_app=True),
    Command("resume", "wznów wcześniejszą sesję agenta", usage="/resume [id]", in_app=True),
    Command("stats", "statystyki ostatniej tury", in_chat=True, in_app=True),
    Command("init", "napisz plik LOCALCODE.md dla tego projektu", in_app=True),
    Command("chat", "przełącz między agentem a czatem", in_chat=True),
    Command("doctor", "diagnostyka instalacji", in_chat=True),
    Command("forget", "zapomnij decyzje o uprawnieniach wyuczone w tym projekcie (tryb Auto)"),
    Command("cwd", "pokaż katalog roboczy", in_chat=True, in_app=True),
    Command("exit", "zakończ", aliases=("quit", "q"), in_chat=True),
)

_BY_NAME: dict[str, Command] = {}
for _command in COMMANDS:
    _BY_NAME[_command.name] = _command
    for _alias in _command.aliases:
        _BY_NAME[_alias] = _command


@dataclass(frozen=True)
class ParsedCommand:
    """``/mode plan`` → ``name="mode"``, ``arg="plan"``, ``args=["plan"]``."""

    name: str
    arg: str = ""
    command: Command | None = None

    @property
    def args(self) -> list[str]:
        return self.arg.split()

    @property
    def known(self) -> bool:
        return self.command is not None


def is_command(text: str) -> bool:
    """True for a line that starts a slash command (``/`` + a letter)."""
    stripped = (text or "").lstrip()
    return len(stripped) > 1 and stripped[0] == "/" and not stripped[1].isspace() and stripped[1] != "/"


def parse_command(text: str) -> ParsedCommand | None:
    """Split ``"/model qwen"`` into its name and argument (``None`` when not a command)."""
    if not is_command(text):
        return None
    head, _, rest = (text or "").strip()[1:].partition(" ")
    name = head.strip().lower()
    command = _BY_NAME.get(name)
    return ParsedCommand(name=command.name if command else name, arg=rest.strip(), command=command)


def find_command(name: str) -> Command | None:
    return _BY_NAME.get((name or "").strip().lstrip("/").lower())


def command_names(*, chat_only: bool = False) -> list[str]:
    """Canonical names (plus aliases) offered by the completer."""
    out: list[str] = []
    for command in COMMANDS:
        if chat_only and not command.in_chat:
            continue
        out.append(command.name)
        out.extend(command.aliases)
    return out


#: ``localcode runtime set <klucz> <wartość>`` → the camelCase API field.
RUNTIME_SETTING_KEYS: dict[str, str] = {
    "max-models": "maxLoadedModels",
    "instances": "maxInstancesPerModel",
    "parallel": "defaultParallel",
    "ram-gb": "ramLimitGb",
    "idle-ttl": "idleTtlMinutes",
    "ctx": "defaultCtx",
    "kv": "kvCacheType",
    "flash-attn": "flashAttn",
    "gpu-layers": "gpuLayers",
    "evict": "evictPolicy",
    "llama-server": "llamaServerPath",
    "python": "pythonPath",
}

#: Values that are stored as ``None`` ("automatycznie").
RUNTIME_AUTO_VALUES = frozenset({"auto", "automatycznie", "none", "null", "-", ""})


def runtime_patch(key: str, value: str) -> dict[str, Any]:
    """``("ram-gb", "20")`` → ``{"ramLimitGb": 20.0}`` (validated server-side)."""
    name = RUNTIME_SETTING_KEYS.get(key.strip().lower())
    if name is None and key in RUNTIME_SETTING_KEYS.values():
        name = key
    if name is None:
        known = ", ".join(sorted(RUNTIME_SETTING_KEYS))
        raise ValueError(f"Nieznane ustawienie „{key}”. Dostępne: {known}.")
    raw = (value or "").strip()
    if raw.lower() in RUNTIME_AUTO_VALUES and name in ("ramLimitGb", "defaultCtx"):
        return {name: None}
    if name == "ramLimitGb":
        try:
            return {name: float(raw.replace(",", "."))}
        except ValueError:
            raise ValueError("Limit RAM podaj w gigabajtach, np. 20.") from None
    if name in ("maxLoadedModels", "maxInstancesPerModel", "defaultParallel", "idleTtlMinutes",
                "defaultCtx", "gpuLayers"):
        try:
            return {name: int(raw)}
        except ValueError:
            raise ValueError(f"Ustawienie „{key}” wymaga liczby całkowitej.") from None
    return {name: raw}


KEY_BINDINGS: tuple[tuple[str, str], ...] = (
    ("Enter", "wyślij wiadomość"),
    ("Alt+Enter / Ctrl+J", "nowa linia (albo zakończ linię znakiem \\)"),
    ("Shift+Tab", "zmień tryb: domyślny → akceptuj edycje → auto → plan → yolo"),
    ("Esc", "przerwij odpowiedź / odrzuć pytanie o zgodę"),
    ("Ctrl+C", "przerwij; dwa razy przy pustym polu — wyjście"),
    ("Ctrl+D", "wyjście"),
    ("Ctrl+R", "wyszukiwanie w historii"),
    ("@", "podpowiedzi ścieżek — plik zostanie dołączony do wiadomości"),
    ("↑ / ↓", "historia wpisanych wiadomości"),
)


def help_table(*, chat: bool = False) -> Table:
    """The table printed by ``/help``."""
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False,
                  title="Polecenia" + (" (tryb czatu)" if chat else ""),
                  title_style="lc.title", title_justify="left")
    table.add_column("polecenie", style="lc.key", no_wrap=True)
    table.add_column("opis")
    for command in COMMANDS:
        if chat and not command.in_chat:
            continue
        table.add_row(command.display, command.help)
    return table


def keys_table() -> Table:
    table = Table(box=ROUNDED, border_style="lc.dim", expand=False, title="Skróty klawiszowe",
                  title_style="lc.title", title_justify="left")
    table.add_column("klawisz", style="lc.key", no_wrap=True)
    table.add_column("działanie")
    for key, description in KEY_BINDINGS:
        table.add_row(key, description)
    return table


def suggest(name: str, candidates: Iterable[str] | None = None) -> str | None:
    """Closest known command name (for "did you mean" messages)."""
    import difflib

    pool: Sequence[str] = list(candidates) if candidates is not None else command_names()
    matches = difflib.get_close_matches((name or "").lower(), pool, n=1, cutoff=0.6)
    return matches[0] if matches else None
