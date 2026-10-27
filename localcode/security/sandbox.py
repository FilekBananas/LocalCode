"""Keeping a model inside its own process.

A model file is data, but the programs that read it — ``llama-server``, the MLX
worker, the image workers — are C and Python parsing megabytes of attacker-
supplied structure.  A bug there is the realistic way a downloaded model turns
into code running as you.  The answer is not to trust the parser but to give it
nothing worth stealing and nowhere to go:

* **no outbound network** — it can bind a local port so LocalCode can talk to
  it, and it cannot reach the internet.  A model cannot phone home, and a
  compromised loader cannot upload your files;
* **no writes outside its own directories** — the models directory, the app's
  own data directory and a temporary directory.  It cannot drop a login item,
  edit ``~/.zshrc`` or touch your documents;
* **no reads of your home directory** beyond what it needs — your Documents,
  Desktop, Keychain and SSH keys are simply not there for it.

This is macOS' own sandbox (``sandbox-exec``, the mechanism behind App Sandbox)
applied to the engine subprocess.  It is not a virtual machine: a kernel
vulnerability defeats it, and anything the engine may legitimately read (the
model files themselves) it may read.  What it removes is the whole class of
"the loader was tricked and then it did something to the rest of the machine".

``settings.security.sandbox``:

``auto`` (default)
    sandbox when the platform allows it; if the sandboxed process cannot start,
    fall back to a normal one and say so in the logs and in the runtime status.
``strict``
    sandbox or refuse to load the model at all.
``off``
    no sandbox (for debugging, or a platform without one).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
MODES = ("auto", "strict", "off")


@dataclass
class SandboxPlan:
    """What a sandboxed engine may touch."""

    read: list[Path] = field(default_factory=list)
    write: list[Path] = field(default_factory=list)
    #: Bind and accept connections on localhost (the engine's own HTTP port).
    local_network: bool = True

    def with_defaults(self) -> "SandboxPlan":
        """Add the directories every engine writes to: its own, and a temp one.

        The caches matter for speed rather than correctness: without a writable
        Metal shader cache, llama.cpp recompiles its kernels on every start and
        a one-second launch becomes twenty-six.
        """
        home = Path.home()
        write = [*self.write, Path("/private/tmp"), Path("/private/var/tmp"),
                 Path(os.environ.get("TMPDIR", "/private/tmp")),
                 home / "Library" / "Caches",
                 home / "Library" / "Saved Application State",
                 *_darwin_dirs()]
        return SandboxPlan(read=list(self.read), write=write,
                           local_network=self.local_network)


@lru_cache(maxsize=2)
def _getconf(name: str) -> str | None:
    """``getconf DARWIN_USER_CACHE_DIR`` and friends.

    Python's :func:`os.confstr` does not know these names on macOS — it raises
    ``ValueError`` — which is exactly how the Metal shader cache ended up outside
    the profile and a 0,8-second model load became thirty seconds.
    """
    try:
        out = subprocess.run(["/usr/bin/getconf", name], capture_output=True,
                             text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = (out.stdout or "").strip()
    return value or None


def _darwin_dirs() -> list[Path]:
    """macOS' per-user temp and cache directories (``/var/folders/xx/…/{T,C}``).

    Metal keeps its compiled shaders in the cache one (``…/C``).  Without write
    access there, llama.cpp recompiles every kernel on every start: measured,
    that turned a 0,8-second load into thirty seconds — the sandbox looked like
    it made inference thirty times slower, when all it did was throw away a
    cache.
    """
    values: list[str] = []
    for name in ("DARWIN_USER_TEMP_DIR", "DARWIN_USER_CACHE_DIR"):
        value = _getconf(name)
        if value:
            values.append(value)
    if not values:
        # No getconf (or a stripped-down system): derive both from TMPDIR, which
        # macOS sets to the "T" sibling of the "C" cache directory.
        tmp = os.environ.get("TMPDIR", "").rstrip("/")
        if tmp.endswith("/T"):
            values += [tmp, tmp[:-2] + "/C"]
    out: list[Path] = []
    for value in values:
        # Both spellings: the sandbox matches the path the process actually
        # opens, and macOS hands these out as /var/folders/… while the resolved
        # form is /private/var/folders/… . Allowing only one of the two leaves
        # the cache unwritable.
        value = value.rstrip("/")
        out.append(Path(value))
        if value.startswith("/private/"):
            out.append(Path(value.replace("/private", "", 1)))
        else:
            out.append(Path("/private" + value))
    return out


def available() -> bool:
    """True when this machine can sandbox a subprocess."""
    return platform.system() == "Darwin" and os.path.exists(SANDBOX_EXEC)


def mode(settings: object | None) -> str:
    raw = str(getattr(getattr(settings, "security", None), "sandbox", "auto") or "auto")
    return raw if raw in MODES else "auto"


def _quote(path: Path) -> str:
    """A path as an SBPL string literal."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_profile(plan: SandboxPlan) -> str:
    """The sandbox profile for ``plan`` (SBPL, the format ``sandbox-exec`` reads).

    Reads are allowed broadly and then *denied* where it matters, rather than
    allow-listed: an allow-list breaks the dynamic loader and the Metal stack
    long before it protects anything, while a deny-list keeps the engine
    working and still puts your documents, keys and messages out of reach.
    Writes work the other way round — nothing, except the few directories the
    engine legitimately owns.
    """
    full = plan.with_defaults()
    denied = "\n".join(f"    (subpath {_quote(p)})" for p in _existing(PRIVATE_PATHS()))
    write = "\n".join(f"    (subpath {_quote(p)})" for p in _existing(full.write))
    lines = [
        "(version 1)",
        ";; LocalCode: an engine may read its model and answer on localhost; nothing else.",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow mach-lookup)",          # Metal, IOKit and dyld need many services
        "(allow iokit-open)",           # the GPU
        "(allow ipc-posix-shm)",
        "(allow file-read*)",
    ]
    if denied:
        lines += [";; Your files are simply not there for the model.",
                  "(deny file-read*", denied, ")"]
    # SBPL is last-match-wins, so this re-opens exactly what the engine needs and
    # the deny list above happened to cover. It matters for a development
    # install: the MLX worker's own interpreter and packages can sit under
    # ~/Documents, and without this the worker cannot read Python itself.
    reads = _existing(full.read)
    allowed = "\n".join(f"    (subpath {_quote(q)})" for q in reads)
    if allowed and denied:
        lines += [";; ...except what this engine legitimately needs to read.",
                  "(allow file-read*", allowed, ")"]
        # Opening a file also means resolving every directory above it, and
        # ``realpath`` on a denied parent fails with "Operation not permitted" —
        # which is how a sandboxed Python could not even find itself. Metadata
        # only: enough to walk the path, not enough to read or list the folder.
        walk = "\n".join(f"    (literal {_quote(a)})" for a in _ancestors(reads))
        if walk:
            lines += [";; Only enough to resolve the path above them.",
                      "(allow file-read-metadata", walk, ")"]
    lines += [
        ";; It writes only where a model legitimately writes.",
        "(allow file-write*",
        write,
        '    (literal "/dev/null")',
        '    (literal "/dev/zero")',
        '    (literal "/dev/random")',
        '    (literal "/dev/urandom")',
        '    (literal "/dev/dtracehelper")',
        ")",
        ";; No way out: no internet, only an answer on localhost.",
        "(deny network*)",
    ]
    if full.local_network:
        lines += ["(allow network-bind (local ip))",
                  "(allow network-inbound (local ip))"]
    return "\n".join(lines) + "\n"


def PRIVATE_PATHS() -> list[Path]:
    """The parts of a home directory a model has no business reading.

    Not a complete list of everything private — a complete list is impossible —
    but the places where the answer to "could it read that?" has to be no:
    documents, keys, mail, messages, browser profiles and the keychain.
    """
    home = Path.home()
    return [home / name for name in (
        "Documents", "Desktop", "Downloads", "Pictures", "Movies", "Music",
        ".ssh", ".aws", ".gnupg", ".config/gh", ".kube",
        "Library/Keychains", "Library/Messages", "Library/Mail", "Library/Safari",
        "Library/Application Support/Google/Chrome",
        "Library/Application Support/Firefox",
        "Library/Containers", "Library/Cookies",
    )]


def _ancestors(paths: Iterable[Path]) -> list[Path]:
    """Every directory above ``paths``, nearest first, without duplicates."""
    seen: dict[str, Path] = {}
    for path in paths:
        for parent in Path(path).parents:
            if str(parent) in ("/", ""):
                continue
            seen.setdefault(str(parent), parent)
    return list(seen.values())


def _existing(paths: Iterable[Path]) -> list[Path]:
    """Deduplicated paths that exist, keeping the spelling they were given.

    Resolving symlinks here would collapse ``/var/folders/…`` into
    ``/private/var/folders/…``; the sandbox matches what the process opens, so
    both spellings have to survive into the profile.
    """
    seen: dict[str, Path] = {}
    for path in paths:
        expanded = Path(path).expanduser()
        try:
            if not expanded.exists():
                continue
        except OSError:
            continue
        seen.setdefault(str(expanded), expanded)
    return list(seen.values())


def wrap(command: Sequence[str], plan: SandboxPlan, *, settings: object | None = None,
         profile_dir: Path | None = None) -> tuple[list[str], bool, Path | None]:
    """``(command, sandboxed, profile file)`` — the command to actually run.

    Returns the original command unchanged when sandboxing is off or
    unavailable; in ``strict`` mode an unavailable sandbox raises instead.
    """
    chosen = mode(settings)
    if chosen == "off":
        return list(command), False, None
    if not available():
        if chosen == "strict":
            raise SandboxUnavailable(
                "Tryb ścisły wymaga piaskownicy macOS, a ta maszyna jej nie ma. "
                "Zmień „security.sandbox” na „auto” albo uruchom LocalCode na macOS.")
        return list(command), False, None

    directory = profile_dir or Path(os.environ.get("TMPDIR", "/tmp"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"localcode-{os.getpid()}-{abs(hash(tuple(command))) % 10**8}.sb"
    path.write_text(build_profile(plan), encoding="utf-8")
    return [SANDBOX_EXEC, "-f", str(path), *command], True, path


class SandboxUnavailable(RuntimeError):
    """Strict mode asked for a sandbox the platform cannot provide."""


def cleanup(path: Path | None) -> None:
    """Remove a written profile (best effort)."""
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        pass


def describe(sandboxed: bool, settings: object | None = None) -> str:
    """One Polish line for the runtime status / logs."""
    if sandboxed:
        return "Model działa w piaskownicy: bez internetu i bez dostępu do Twoich plików."
    chosen = mode(settings)
    if chosen == "off":
        return "Piaskownica wyłączona w ustawieniach — model ma zwykłe uprawnienia procesu."
    if not available():
        return "Ten system nie ma piaskownicy macOS — model działa bez izolacji."
    return "Nie udało się uruchomić modelu w piaskownicy — działa bez izolacji."


def has_sandbox_exec() -> bool:
    """Whether the binary exists at all (separate from platform checks)."""
    return bool(shutil.which(SANDBOX_EXEC) or os.path.exists(SANDBOX_EXEC))


# --------------------------------------------------------------------------- #
# The agent's shell: confined to the project folder
# --------------------------------------------------------------------------- #

#: What the agent's commands may *read* in the home directory besides the
#: project: tool chains, package caches and the configuration a build reads
#: (``~/.gitconfig``, ``~/.npmrc``).  Everything else there — documents, keys,
#: other projects, LocalCode's own settings — is not there for a command.
SHELL_HOME_READ: tuple[str, ...] = (
    ".CFUserTextEncoding", ".gitconfig", ".gitignore", ".gitignore_global", ".gitattributes",
    ".config", ".npmrc", ".yarnrc", ".yarnrc.yml", ".condarc", ".editorconfig",
    ".tool-versions", ".python-version", ".nvmrc",
    ".cargo", ".rustup", ".nvm", ".pyenv", ".rbenv", ".rvm", ".bun", ".deno", ".npm",
    ".pnpm-store", ".yarn", ".gradle", ".m2", ".sdkman", ".jenv", ".volta", ".asdf", ".local",
    ".cache", ".swiftpm", ".conda", "miniconda3", "anaconda3", "miniforge3", "mambaforge",
    ".julia", ".ghcup", ".cabal", ".stack", ".opam", ".elan", "go", ".dotnet", ".nuget",
    ".gem", ".bundle", ".composer", ".docker", ".terraform.d",
    "Library/Caches", "Library/Developer", "Library/pnpm", "Library/Python", "Library/Android",
    "Library/org.swift.swiftpm",
)
#: Inside the readable directories above, the parts that hold credentials.
SHELL_HOME_SECRET: tuple[str, ...] = (
    ".config/gh", ".config/gcloud", ".config/op", ".config/hub", ".config/rclone",
    ".config/doctl", ".docker/config.json", ".cargo/credentials", ".cargo/credentials.toml",
    ".local/share/keyrings",
)
#: Where the agent's commands may *write* outside the project: temporary files
#: and the caches every package manager and compiler keeps in the home folder.
SHELL_HOME_WRITE: tuple[str, ...] = (
    ".cache", "Library/Caches", ".npm", ".pnpm-store", "Library/pnpm", ".yarn",
    ".bun/install/cache", ".cargo/registry", ".cargo/git", ".gradle", ".m2/repository",
    "go/pkg", ".swiftpm", "Library/org.swift.swiftpm", "Library/Developer/Xcode/DerivedData",
    ".nuget/packages", ".conda/pkgs",
)


def _spellings(path: Path) -> list[Path]:
    """``path`` as given and resolved: the sandbox matches what is opened."""
    out = [Path(path)]
    try:
        real = Path(os.path.realpath(path))
    except OSError:
        real = Path(path)
    if str(real) != str(path):
        out.append(real)
    text = str(path)
    if text.startswith("/private/"):
        out.append(Path(text[len("/private"):]))
    elif text.startswith(("/tmp", "/var/")):
        out.append(Path("/private" + text))
    return out


def shell_profile(project: Path | str) -> str:
    """The sandbox for a command the agent runs: the project folder and nothing else.

    Unlike :func:`build_profile` (a model loader, which needs almost nothing)
    a build needs almost everything — compilers, the network, package caches —
    so this one starts from "allow" and closes the two things that matter:

    * **reading the home directory** is denied except the project and the
      tool chains in :data:`SHELL_HOME_READ` (minus :data:`SHELL_HOME_SECRET`).
      Only file *contents* are denied, not metadata: ``git`` and ``realpath``
      still walk up through ``~/Documents`` to the project, but ``ls ~`` and
      ``cat ~/.ssh/id_rsa`` get "Operation not permitted";
    * **writing** is denied everywhere except the project, temporary
      directories, ``/dev`` and the caches in :data:`SHELL_HOME_WRITE` — no
      ``~/.zshrc``, no login items, no global installs.

    SBPL is last-match-wins, which is what lets each allow re-open a part of
    the deny before it.
    """
    home = Path.home()
    root = Path(project).expanduser()
    projects = _spellings(root)
    homes = _spellings(home)
    reads = projects + [home / name for name in SHELL_HOME_READ]
    secrets = [home / name for name in SHELL_HOME_SECRET]
    temp = [Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path("/private/var/tmp")]
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        temp += _spellings(Path(tmpdir.rstrip("/") or "/tmp"))
    writes = projects + temp + _darwin_dirs() + [home / name for name in SHELL_HOME_WRITE]

    def block(kind: str, paths: Iterable[Path]) -> list[str]:
        seen: list[str] = []
        for path in paths:
            line = f"    (subpath {_quote(path)})"
            if line not in seen:
                seen.append(line)
        return [f"({kind}", *seen, ")"] if seen else []

    lines = [
        "(version 1)",
        ";; LocalCode: an agent's command, confined to the project folder.",
        "(allow default)",
        ";; Your home directory is closed...",
        *block("deny file-read-data", homes),
        ";; ...except the project and the tools a build reads.",
        *block("allow file-read-data", reads),
        ";; Credentials stay closed even there.",
        *block("deny file-read-data", secrets),
        ";; Writes: nowhere...",
        '(deny file-write* (subpath "/"))',
        ";; ...but the project, temporary files, /dev and the build caches.",
        *block("allow file-write*", [*writes, Path("/dev")]),
    ]
    return "\n".join(lines) + "\n"


def wrap_shell(argv: Sequence[str], project: Path | str) -> list[str]:
    """``argv`` run inside :func:`shell_profile`; unchanged where there is no sandbox."""
    if not available():
        return list(argv)
    return [SANDBOX_EXEC, "-p", shell_profile(project), *argv]


def shell_home_paths() -> tuple[list[Path], list[Path]]:
    """``(readable, secret)`` home paths of the shell sandbox, for the path check."""
    home = Path.home()
    return ([home / name for name in SHELL_HOME_READ], [home / name for name in SHELL_HOME_SECRET])
