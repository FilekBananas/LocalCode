"""Permission policy for the coding agent.

The agent loop asks the policy about every tool call::

    info = await tool.permission(args, ctx)          # None = nothing to approve
    decision = policy.decide(tool, args, info)       # "allow" | "ask" | "deny"
    if decision == "ask": ...show PermissionRequest, on "always": policy.remember_always(tool, args)

Modes (``AgentSettings.permission_mode``):

* ``default``      – read tools run freely; file writes ask; shell commands ask
                     unless they are provably read-only (``git status``, ``ls`` …).
* ``accept_edits`` – file writes inside the project run freely; shell as in default.
* ``auto``         – decided like ``default``, but every call that would ask is
                     judged by a critical reviewer model instead (``agent/autoreview.py``),
                     which says yes or no and why. Calls :meth:`PermissionPolicy.needs_human`
                     names (flagged or catastrophic) still go to the user.
* ``yolo``         – everything runs, except catastrophic commands (``rm -rf ~``,
                     ``curl … | sh`` …) which always ask.
* ``plan``         – read-only exploration: write and exec tools are denied.

Two flags inside ``PermissionInfo.title`` (set by the tools, which know how they
resolve paths) force a question in every mode but ``plan``:
``OUTSIDE_PROJECT_FLAG`` (a write outside the project directory) and
``DANGER_FLAG`` (a catastrophic command).

"Always allow" memory is per session: file-writing tools are remembered per tool,
shell commands per *command prefix* (``git commit``, ``npm test``,
``python -m pytest`` …). A compound command is auto-allowed only when every simple
command in it is read-only or remembered, and it contains no command
substitution, subshell or file redirection.

Shell analysis is deliberately conservative: anything the parser does not fully
understand is treated as "not read-only". False negatives only cost one
question; false positives could run a destructive command unasked.

This module must not import ``localcode.agent.tools`` at module level (the tools
import the flags defined here); the few helpers it needs are imported lazily.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Decision = Literal["allow", "ask", "deny"]

MODES: tuple[str, ...] = ("default", "accept_edits", "auto", "yolo", "plan")

MODE_LABELS: dict[str, str] = {
    "default": "Domyślny — pytaj o zmiany plików i polecenia",
    "accept_edits": "Akceptuj edycje — pytaj tylko o polecenia",
    "auto": "Auto — model krytycznie ocenia każdą akcję i mówi dlaczego",
    "yolo": "YOLO — wszystko bez pytania (poza niebezpiecznymi poleceniami)",
    "plan": "Plan — tylko odczyt, bez zmian i poleceń",
}

# Markers embedded in PermissionInfo.title by the tools.
OUTSIDE_PROJECT_FLAG = "⚠ poza projektem"
DANGER_FLAG = "⚠ niebezpieczne"
#: Moving the conversation to a model behind an API key: from then on what the
#: agent reads leaves this Mac, and every step is billed. Asks in every mode,
#: and "always" does not stick — like DANGER_FLAG.
REMOTE_FLAG = "⚠ dane do dostawcy"

_MODE_ALIASES = {
    "default": "default",
    "normal": "default",
    "ask": "default",
    "accept_edits": "accept_edits",
    "acceptedits": "accept_edits",
    "accept-edits": "accept_edits",
    "edits": "accept_edits",
    "auto_edit": "accept_edits",
    "yolo": "yolo",
    "bypass": "yolo",
    "bypass_permissions": "yolo",
    "auto": "auto",
    "auto_review": "auto",
    "review": "auto",
    "plan": "plan",
    "read_only": "plan",
    "readonly": "plan",
}


def normalize_mode(mode: str | None) -> str:
    """Canonical mode name; raises ``ValueError`` (Polish message) for unknown modes."""
    key = (mode or "default").strip().lower().replace(" ", "_")
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    raise ValueError(
        f"Nieznany tryb uprawnień: {mode!r}. Dostępne: {', '.join(MODES)}."
    )


# --------------------------------------------------------------------------- #
# Shell parsing
# --------------------------------------------------------------------------- #

_PUNCT = "();<>|&\n"
_SEPARATORS = frozenset({"&&", "||", ";", "|", "&", "|&", "\n", ";;", ";&", ";;&"})
_OUT_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>"})
_IN_REDIRECTS = frozenset({"<", "<<", "<<-", "<<<", "<>"})
_DUP_REDIRECTS = frozenset({">&", "<&"})
_REDIRECTS = _OUT_REDIRECTS | _IN_REDIRECTS | _DUP_REDIRECTS
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")


@dataclass
class SimpleCommand:
    """One simple command of a pipeline/list: words, leading assignments, redirections."""

    argv: list[str] = field(default_factory=list)
    assignments: list[str] = field(default_factory=list)
    redirects: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ParsedShell:
    commands: list[SimpleCommand]
    has_substitution: bool = False   # $( ), backticks, <( ), >( )
    has_subshell: bool = False       # ( ... ) grouping
    parse_error: bool = False


def _is_operator(tok: str) -> bool:
    return bool(tok) and all(c in _PUNCT for c in tok)


def parse_shell(command: str) -> ParsedShell:
    """Split a bash command line into simple commands (conservatively).

    Uses :mod:`shlex` with punctuation handling, so quoting is respected. The
    lexer drops quote information, which can only make the analysis *more*
    cautious (a quoted ``">"`` looks like a redirection).
    """
    has_sub = any(s in command for s in ("$(", "`", "<(", ">("))
    lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCT)
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return ParsedShell(commands=[], has_substitution=has_sub, parse_error=True)

    commands: list[SimpleCommand] = []
    current = SimpleCommand()
    subshell = False
    error = False
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_operator(tok):
            if tok in _SEPARATORS:
                if current.argv or current.assignments or current.redirects:
                    commands.append(current)
                current = SimpleCommand()
            elif tok in _REDIRECTS:
                # "2>" is lexed as "2", ">" — a bare fd number before the operator is not an argument
                if current.argv and current.argv[-1].isdigit():
                    current.argv.pop()
                target = tokens[i + 1] if i + 1 < len(tokens) else ""
                if not target or _is_operator(target):
                    error = True
                else:
                    i += 1
                current.redirects.append((tok, target))
            elif set(tok) <= {"(", ")"}:
                subshell = True
                if current.argv or current.assignments or current.redirects:
                    commands.append(current)
                current = SimpleCommand()
            else:
                error = True  # unknown operator combination such as "|>" or "&&&"
        else:
            if not current.argv and _ASSIGNMENT.match(tok):
                current.assignments.append(tok)
            else:
                current.argv.append(tok)
        i += 1
    if current.argv or current.assignments or current.redirects:
        commands.append(current)
    return ParsedShell(commands=commands, has_substitution=has_sub, has_subshell=subshell,
                       parse_error=error)


# Absolute paths to these directories are accepted as the "same" command as the bare name.
_SYSTEM_BIN_DIRS = ("/bin", "/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/sbin", "/usr/sbin")


def _command_name(word: str) -> str | None:
    """Name used for allowlist checks; None for project-relative executables (``./ls``)."""
    if "/" not in word:
        return word
    head, _, base = word.rpartition("/")
    if head in _SYSTEM_BIN_DIRS:
        return base
    return None


def _positionals(args: list[str], takes_value: Iterable[str] = ()) -> list[str]:
    """Non-option arguments; options listed in ``takes_value`` consume the next word."""
    out: list[str] = []
    skip = False
    after_dd = False
    values = set(takes_value)
    for a in args:
        if skip:
            skip = False
            continue
        if after_dd:
            out.append(a)
            continue
        if a == "--":
            after_dd = True
            continue
        if a.startswith("-") and a != "-":
            if a in values:
                skip = True
            continue
        out.append(a)
    return out


def _has_flag(args: list[str], *flags: str) -> bool:
    """True when any of ``flags`` is present (also as ``--flag=value`` or inside ``-abc``)."""
    for a in args:
        if a == "--":
            break
        for f in flags:
            if a == f or (f.startswith("--") and a.startswith(f + "=")):
                return True
            if len(f) == 2 and f[0] == "-" and f[1] != "-" and a.startswith("-") \
                    and not a.startswith("--") and f[1] in a[1:]:
                return True
    return False


# Commands that can be read-only. Those with a checker in _simple_is_read_only
# (find, git, sed, sort, …) are read-only only for some arguments; the rest never
# modify anything, whatever their arguments.
_READ_ONLY_CANDIDATES = frozenset({
    "ls", "pwd", "echo", "printf", "cat", "head", "tail", "wc", "stat", "which", "whereis",
    "type", "du", "df", "whoami", "id", "groups", "uname", "basename", "dirname", "realpath",
    "readlink", "grep", "egrep", "fgrep", "ag", "ack", "cut", "tr", "nl", "column", "fold", "rev",
    "diff", "cmp", "comm", "md5", "md5sum", "shasum", "sha1sum", "sha256sum", "sha512sum",
    "cksum", "od", "hexdump", "strings", "ps", "pgrep", "lsof", "true", "false", "test", "[",
    "seq", "cd", "nproc", "sw_vers", "arch", "printenv", "jq", "tokei", "cloc", "less", "more",
    "sleep", "expand", "unexpand", "look", "locale", "uptime", "vm_stat", "free", "tac",
    # read-only only for some arguments (see the checkers below)
    "file", "tree", "date", "hostname", "sort", "uniq", "xxd", "sed", "env", "command",
    "sysctl", "defaults", "find", "fd", "fdfind", "rg", "git", "npm", "pip", "pip3",
})

_VERSION_TOOLS = frozenset({
    "python", "python3", "node", "npm", "npx", "pnpm", "yarn", "bun", "deno", "pip", "pip3",
    "cargo", "rustc", "rustup", "go", "ruby", "java", "javac", "gcc", "g++", "clang", "clang++",
    "make", "cmake", "swift", "swiftc", "uv", "poetry", "tsc", "brew", "docker", "kubectl",
    "git", "perl", "php", "dotnet", "gradle", "mvn", "xcodebuild", "llama-server", "gh",
})
_VERSION_FLAGS = frozenset({"--version", "-v", "-V", "version", "-version"})

_GIT_READ_ONLY_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog",
    "cat-file", "ls-tree", "merge-base", "name-rev", "rev-list", "count-objects",
    "check-ignore", "grep", "whatchanged", "version", "help", "show-ref", "for-each-ref",
    "ls-remote", "cherry", "range-diff", "diff-tree", "diff-files", "diff-index", "annotate",
})


def _git_is_read_only(args: list[str]) -> bool:
    i = 0
    # global options before the subcommand
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-c", "--config-env", "--exec-path") or a.startswith("--exec-path="):
            return False  # config overrides can run arbitrary programs
        if a in ("-C", "--git-dir", "--work-tree", "--namespace"):
            i += 2
            continue
        if a in ("--version", "--help"):
            return True
        i += 1
    if i >= len(args):
        return True  # bare "git" prints help
    sub, rest = args[i], args[i + 1:]
    if sub in _GIT_READ_ONLY_SUBCOMMANDS:
        if _has_flag(rest, "--output", "--ext-diff", "-O", "--open-files-in-pager"):
            return False
        return True
    if sub == "branch":
        if _has_flag(rest, "-d", "-D", "-m", "-M", "-c", "-C", "-f", "-u", "--delete", "--move",
                     "--copy", "--force", "--set-upstream-to", "--unset-upstream",
                     "--edit-description", "--track", "--no-track", "--create-reflog"):
            return False
        value_opts = ("--contains", "--no-contains", "--merged", "--no-merged", "--points-at",
                      "--sort", "--format", "--color", "--abbrev")
        return not _positionals(rest, value_opts) or _has_flag(rest, "--list", "-l")
    if sub == "tag":
        if _has_flag(rest, "-d", "-a", "-s", "-f", "-m", "-F", "-u", "--delete", "--annotate",
                     "--sign", "--force", "--message", "--file", "--local-user", "-e", "--edit"):
            return False
        value_opts = ("--contains", "--no-contains", "--merged", "--no-merged", "--points-at",
                      "--sort", "--format", "--color", "-n")
        return not _positionals(rest, value_opts) or _has_flag(rest, "--list", "-l")
    if sub == "remote":
        return all(a in ("-v", "--verbose") for a in rest) or (bool(rest) and rest[0] in ("show", "get-url"))
    if sub == "stash":
        return bool(rest) and rest[0] in ("list", "show")
    if sub == "config":
        if _has_flag(rest, "--unset", "--unset-all", "--add", "--replace-all", "--rename-section",
                     "--remove-section", "-e", "--edit"):
            return False
        return _has_flag(rest, "--get", "--get-all", "--get-regexp", "--list", "-l", "--get-urlmatch")
    if sub == "reflog":
        return not rest or rest[0] == "show" or rest[0].startswith("-")
    if sub == "worktree":
        return bool(rest) and rest[0] == "list"
    if sub == "submodule":
        return bool(rest) and rest[0] == "status"
    return False


_SAFE_ASSIGNMENTS = frozenset({"LC_ALL", "LANG", "LC_CTYPE", "NO_COLOR", "TERM", "COLUMNS",
                               "LINES", "GIT_PAGER", "PAGER", "TZ", "CLICOLOR", "FORCE_COLOR"})


def _simple_is_read_only(argv: list[str]) -> bool:
    name = _command_name(argv[0])
    if name is None:
        return False
    args = argv[1:]
    if name in _VERSION_TOOLS and len(args) == 1 and args[0] in _VERSION_FLAGS:
        return True
    if name in ("python", "python3") and args in (["-V"], ["--version"], ["-VV"]):
        return True
    if name not in _READ_ONLY_CANDIDATES:
        return False
    if name == "find":
        return not any(a in ("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint",
                             "-fprint0", "-fprintf", "-fls") for a in args)
    if name in ("fd", "fdfind"):
        return not _has_flag(args, "-x", "-X", "--exec", "--exec-batch")
    if name == "rg":
        return not _has_flag(args, "--pre")
    if name == "git":
        return _git_is_read_only(args)
    if name == "sed":
        # only "sed -n '<addr>p' file…" style printing; -i and w/e/r commands can write or execute
        if _has_flag(args, "-i", "--in-place", "-f", "--file"):
            return False
        if not _has_flag(args, "-n", "--quiet", "--silent"):
            return False
        scripts = [args[i + 1] for i, a in enumerate(args[:-1]) if a in ("-e", "--expression")]
        if not scripts:
            scripts = _positionals(args)[:1]
        return bool(scripts) and all(
            re.fullmatch(r"[0-9$,;!p ]*p", s.strip()) is not None for s in scripts)
    if name == "sort":
        return not _has_flag(args, "-o", "--output")
    if name == "uniq":
        return len(_positionals(args, ("-f", "-s", "-w"))) <= 1
    if name == "xxd":
        return not _has_flag(args, "-r", "--revert") and len(_positionals(args, ("-l", "-s", "-c", "-g", "-o"))) <= 1
    if name == "tree":
        return not _has_flag(args, "-o")
    if name == "file":
        return not _has_flag(args, "-C", "--compile")
    if name == "date":
        return not _has_flag(args, "-s", "--set") and not any(re.fullmatch(r"\d{4,}", a) for a in args)
    if name == "hostname":
        return not _positionals(args)
    if name == "env":
        return not args  # "env cmd" runs a command
    if name == "command":
        return bool(args) and args[0] in ("-v", "-V")
    if name == "sysctl":
        return not _has_flag(args, "-w") and not any("=" in a for a in args)
    if name == "defaults":
        return bool(args) and args[0] in ("read", "domains", "read-type", "find")
    if name == "npm":
        if not args:
            return False
        if args[0] == "config":
            return len(args) >= 2 and args[1] in ("get", "list", "ls")
        return args[0] in ("ls", "list", "view", "outdated", "root", "prefix", "why", "explain",
                           "-v", "--version")
    if name in ("pip", "pip3"):
        return bool(args) and args[0] in ("list", "show", "freeze", "check", "--version", "-V")
    if name in ("less", "more"):
        return not any(a.startswith("+") for a in args)  # "+cmd" runs less commands
    return True


def is_read_only_command(command: str) -> bool:
    """True when every part of ``command`` is on the read-only allowlist.

    Compound commands (``&&``, ``||``, ``;``, ``|``, newlines) are checked part by
    part. Command substitution, subshells, output redirection to a file (other
    than ``/dev/null``), unsafe environment assignments and unparsable input all
    make a command *not* read-only.
    """
    if not command or not command.strip():
        return False
    parsed = parse_shell(command)
    if parsed.parse_error or parsed.has_substitution or parsed.has_subshell or not parsed.commands:
        return False
    for sc in parsed.commands:
        if not _redirects_are_safe(sc.redirects):
            return False
        for a in sc.assignments:
            if a.split("=", 1)[0].rstrip("+") not in _SAFE_ASSIGNMENTS:
                return False
        if not sc.argv:
            continue
        if not _simple_is_read_only(sc.argv):
            return False
    return True


def _redirects_are_safe(redirects: list[tuple[str, str]]) -> bool:
    for op, target in redirects:
        if op in _DUP_REDIRECTS and (target.isdigit() or target == "-"):
            continue
        if op in _IN_REDIRECTS:
            continue
        if op in _OUT_REDIRECTS and target == "/dev/null":
            continue
        return False
    return True


# --------------------------------------------------------------------------- #
# Catastrophic commands
# --------------------------------------------------------------------------- #

_PIPE_TO_INTERPRETER = re.compile(
    r"\b(?:curl|wget|fetch)\b[^|;&\n]*\|\s*(?:sudo\s+(?:-\S+\s+)*)?(?:env\s+(?:\S+=\S*\s+)*)?"
    r"(?:\S*/)?(?:(?:ba|z|da|k|fi|tc|c)?sh|python[\d.]*|perl|ruby|node)\b"
)
_SHELL_FROM_NETWORK = re.compile(
    r"\b(?:(?:ba|z|da|k)?sh|source|eval)\b[^\n]*?(?:<\(|\$\(|`)\s*(?:curl|wget)\b"
)
_DOT_FROM_NETWORK = re.compile(r"(?:^|[;&|]\s*)\.\s+<\(\s*(?:curl|wget)\b")
_FORK_BOMB = re.compile(r"([\w:.-]+)\(\)\{\1\|\1&;?\};?\1")
_DISK_DEVICE = re.compile(r"^/dev/(?:r?disk\d|sd[a-z]|nvme\d|hd[a-z]|mmcblk\d|xvd[a-z]|vd[a-z])")

_HOME_CHILDREN = frozenset({"Documents", "Desktop", "Downloads", "Library", "Pictures", "Movies",
                            "Music", "Applications", "Public", ".ssh", ".gnupg", ".config",
                            ".local", "Developer", "Projects", "src", "code"})
_SYSTEM_TOP = frozenset({"bin", "sbin", "usr", "etc", "System", "Library", "private", "var",
                         "Applications", "opt", "cores", "dev", "Volumes", "Users", "home",
                         "boot", "lib", "lib64", "root", "srv", "mnt", "tmp"})

_WRAPPERS = frozenset({"sudo", "doas", "nohup", "time", "nice", "ionice", "command", "exec",
                       "builtin", "caffeinate", "stdbuf", "timeout", "gtimeout"})


def _strip_wrappers(argv: list[str]) -> list[str]:
    """Drop ``sudo``/``nohup``/``env X=1``/``xargs`` style prefixes to find the real command."""
    out = list(argv)
    for _ in range(8):
        if not out:
            return out
        name = _command_name(out[0]) or out[0].rsplit("/", 1)[-1]
        if name in _WRAPPERS:
            out = out[1:]
            # skip the wrapper's own options (and a numeric argument, e.g. "timeout 10", "nice -n 5")
            while out and (out[0].startswith("-") or re.fullmatch(r"[\d.]+[smhd]?", out[0])):
                if out[0] in ("-u", "-g", "-n", "-p", "-s", "-k", "-o", "-i", "-e") and len(out) > 1:
                    out = out[2:]
                else:
                    out = out[1:]
            continue
        if name == "env":
            out = out[1:]
            while out and (out[0].startswith("-") or _ASSIGNMENT.match(out[0])):
                out = out[1:]
            continue
        if name == "xargs":
            out = out[1:]
            while out and out[0].startswith("-"):
                if out[0] in ("-I", "-n", "-P", "-L", "-s", "-d", "-E") and len(out) > 1:
                    out = out[2:]
                else:
                    out = out[1:]
            continue
        break
    return out


def _is_dangerous_target(target: str, cwd: str | None) -> bool:
    """Is deleting/re-permissioning ``target`` catastrophic (root, home, system dir, project)?"""
    home = os.path.expanduser("~").rstrip("/") or "/"
    t = target.strip().strip("'\"")
    if not t:
        return False
    if "--no-preserve-root" in t:
        return True
    t = t.replace("${HOME}", home).replace("$HOME", home)
    if t.startswith("~"):
        t = os.path.expanduser(t)
    # a trailing glob ("/*", "~/*", "*") deletes the directory's content
    base = t
    while base.endswith(("/*", "/.*", "/")) and len(base) > 1:
        base = base[:-2] if base.endswith("/*") else (base[:-3] if base.endswith("/.*") else base[:-1])
    if base in ("*", ".*", ""):
        base = "/" if t.startswith("/") else "."
    if base.startswith("/"):
        path = os.path.normpath(base)
    elif cwd:
        path = os.path.normpath(os.path.join(cwd, base))
    else:
        return base in (".", "..") or base.startswith("../") or base == ".git"
    if path == "/":
        return True
    parts = [p for p in path.split("/") if p]
    if len(parts) == 1:
        return True  # any top-level directory: /usr, /Users, /etc, /Volumes …
    if parts[0] in _SYSTEM_TOP and len(parts) <= 2 and parts[0] not in ("tmp", "private", "var", "Volumes"):
        return True
    if parts[0] in ("private", "var") and len(parts) <= 2:
        return True
    if path == home or home.startswith(path + "/"):
        return True  # home or one of its ancestors
    if os.path.dirname(path) == home and os.path.basename(path) in _HOME_CHILDREN:
        return True
    if len(parts) == 2 and parts[0] in ("Users", "home", "Volumes"):
        return True
    if cwd:
        root = os.path.normpath(cwd)
        if path == root or root.startswith(path + "/"):
            return True  # the whole project or one of its ancestors
        if path == os.path.join(root, ".git"):
            return True
    return False


def _catastrophic_simple(argv: list[str], cwd: str | None) -> bool:
    argv = _strip_wrappers(argv)
    if not argv:
        return False
    name = argv[0].rsplit("/", 1)[-1]
    args = argv[1:]
    if name == "rm":
        if "--no-preserve-root" in args:
            return True
        recursive = _has_flag(args, "-r", "-R", "--recursive")
        return recursive and any(_is_dangerous_target(t, cwd) for t in _positionals(args))
    if name in ("chmod", "chown", "chgrp"):
        pos = _positionals(args, ("--reference",))
        targets = pos[1:] if len(pos) > 1 else []
        recursive = _has_flag(args, "-R", "--recursive")
        if any(os.path.normpath(t) == "/" for t in targets if t.startswith("/")):
            return True
        return recursive and any(_is_dangerous_target(t, cwd) for t in targets)
    if name == "mv":
        pos = _positionals(args, ("-t", "--target-directory", "-S", "--suffix"))
        return any(_is_dangerous_target(t, cwd) for t in pos[:-1])
    if name == "dd":
        for a in args:
            if a.startswith("of=/dev/") and a[3:] not in ("/dev/null", "/dev/zero", "/dev/stdout",
                                                          "/dev/stderr", "/dev/tty"):
                return True
        return False
    if name.startswith(("mkfs", "newfs")) or name in ("fdisk", "sfdisk", "wipefs", "gpt"):
        return True
    if name == "diskutil":
        sub = args[0].lower() if args else ""
        if sub.startswith(("erase", "zerodisk", "randomdisk", "secureerase", "partitiondisk",
                           "reformat")):
            return True
        return sub == "apfs" and len(args) > 1 and args[1].lower().startswith(("delete", "erase"))
    if name in ("shutdown", "reboot", "halt", "poweroff"):
        return True
    if name == "init" and args[:1] in (["0"], ["6"]):
        return True
    if name == "systemctl" and any(a in ("poweroff", "reboot", "halt", "kexec") for a in args):
        return True
    if name == "launchctl" and args[:1] and args[0].startswith("reboot"):
        return True
    if name == "find":
        roots = []
        for a in args:
            if a.startswith("-") or a in ("(", "!", ")"):
                break
            roots.append(a)
        destructive = "-delete" in args or any(
            a in ("-exec", "-execdir") and i + 1 < len(args) and args[i + 1].rsplit("/", 1)[-1] in ("rm", "shred")
            for i, a in enumerate(args))
        if not destructive:
            return False
        if any(a in _FIND_FILTERS for a in args):
            # a targeted cleanup ("find . -name '*.pyc' -delete") is fine inside the project;
            # only machine-wide roots stay catastrophic
            return any(_is_dangerous_target(r, None) for r in roots
                       if r.startswith(("/", "~", "$HOME", "${HOME}")))
        return any(_is_dangerous_target(r, cwd) for r in roots)
    return False


_FIND_FILTERS = frozenset({"-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-newer",
                           "-mtime", "-mmin", "-empty", "-size", "-wholename", "-iwholename",
                           "-user", "-perm", "-atime", "-ctime", "-cmin", "-amin", "-links"})


# --------------------------------------------------------------------------- #
# Commands that reach outside the project
# --------------------------------------------------------------------------- #

_PATH_STOP = r"""\s'"`;|&<>(){}"""
#: A path-looking word: ``~…``, ``$HOME…``, an absolute path or a ``..`` hop,
#: starting the line or right after a space, quote, ``=``, ``(``, ``:`` or ``,``
#: — so ``python -c "open('/Users/…')"`` is caught and ``HEAD~3`` or
#: ``s/a/b/`` are not.
_OUTSIDE_WORD = re.compile(
    r"(?:^|(?<=[\s'\"=(:,;|&<>]))"
    r"(~[^" + _PATH_STOP + r"]*|\$\{?HOME\}?[^" + _PATH_STOP + r"]*|/[^" + _PATH_STOP + r"]*"
    r"|\.\.(?:/[^" + _PATH_STOP + r"]*)?)")
#: A quoted path is one word even with spaces in it: ``cd "/Users/me/my project"``.
_QUOTED_PATH = re.compile(r"'((?:~|\$\{?HOME\}?|/|\.\.)[^'\n]*)'"
                          r'|"((?:~|\$\{?HOME\}?|/|\.\.)[^"\n]*)"')


#: A ``cd``/``pushd`` at the start of a simple command, and its target.
_CD_AT = re.compile(r"(?:^|[;&|(\n]\s*)(?:cd|pushd)\s+(\"[^\"]*\"|'[^']*'|[^\s;&|)]+)")


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def outside_paths(command: str, project: str | os.PathLike[str],
                  cwd: str | os.PathLike[str] | None = None) -> list[str]:
    """The places outside ``project`` that ``command`` names, as written.

    What counts is the user's own data: anything in the home directory that is
    not the project (nor a tool chain the shell sandbox lets builds read —
    ``~/.cargo/bin/cargo``), other users' folders, other disks, and ``/`` itself
    (``find /`` walks through all of them).  System directories (``/usr``,
    ``/opt/homebrew``, ``/tmp``) do not count: reading them is harmless and
    the sandbox stops writing there anyway.

    A heuristic for the *question* the user is asked, not the protection
    itself: a path assembled at run time slips through here and is stopped by
    the sandbox (:func:`localcode.security.sandbox.shell_profile`) instead.
    """
    from localcode.security.sandbox import shell_home_paths

    root = os.path.realpath(os.path.expanduser(os.fspath(project)))
    here = os.path.realpath(os.path.expanduser(os.fspath(cwd))) if cwd else root
    home = os.path.realpath(os.path.expanduser("~"))
    readable, secret = shell_home_paths()
    tools = [os.path.realpath(p) for p in readable]
    secrets = [os.path.realpath(p) for p in secret]

    found: list[str] = []

    def outside(target: str) -> bool:
        if _inside(target, root):
            return False
        if target == "/" or _inside(target, "/Users") or _inside(target, "/Volumes"):
            if _inside(target, home) and not any(_inside(target, s) for s in secrets) \
                    and any(_inside(target, t) for t in tools):
                return False
            return True
        return False

    words: list[tuple[int, str]] = []
    heads: set[int] = set()              # where a quoted path starts: its cut-off head is not a word
    for match in _QUOTED_PATH.finditer(command or ""):
        word = match.group(1) or match.group(2)
        heads.add(match.start() + 1)
        if word != "/":                  # '/' in code is a string ("a/b".split('/')), not the disk
            words.append((match.start(), word))
    words += [(m.start(1), m.group(1)) for m in _OUTSIDE_WORD.finditer(command or "") if m.start() not in heads]
    # a ``..`` means "above where the shell is at that point": follow the ``cd``s before it
    moves: list[tuple[int, str | None]] = []
    at: str | None = here
    for move in _CD_AT.finditer(command or ""):
        at = _cd_target([move.group(1).strip("'\"")], at)
        moves.append((move.end(), at))
    for position, word in words:
        if word.startswith("//"):
            continue                     # the rest of a URL (https://…)
        where = next((cwd_ for end, cwd_ in reversed(moves) if end <= position), here) or here
        text = word
        if text.startswith("~"):
            name, _, rest = text[1:].partition("/")
            base = home if not name else os.path.join(os.path.dirname(home), name)
            text = os.path.join(base, rest) if rest else base
        elif text.startswith("$"):
            text = home + re.sub(r"^\$\{?HOME\}?", "", text)
        elif text.startswith(".."):
            text = os.path.join(where, text)
        target = os.path.realpath(os.path.normpath(text))
        if outside(target) and word not in found:
            found.append(word)
    parsed = parse_shell(command or "")
    first = parsed.commands[0].argv if parsed.commands else []
    # ``cd "<project>" && …`` puts the rest back inside, wherever the shell was left
    returns = (len(first) == 2 and first[0] in ("cd", "pushd")
               and _inside(os.path.realpath(os.path.join(here, os.path.expanduser(first[1]))), root))
    if outside(here) and not returns and here not in found:      # /tmp & co. are not user data
        found.append(here)               # a previous ``cd`` already left the project
    for sc in parsed.commands:
        if sc.argv and sc.argv[0] in ("cd", "pushd") and len(sc.argv) == 1 and "~" not in found:
            found.append("~")            # a bare ``cd`` goes home
    return found


def is_catastrophic(command: str, cwd: str | os.PathLike[str] | None = None) -> bool:
    """True for commands that can wreck the machine or the whole project.

    Covers ``rm -r`` of ``/``, ``~``, ``$HOME``, system directories, the project
    root or its ancestors; ``mkfs``/``diskutil erase…``; ``dd of=/dev/…``; writes to raw
    disk devices; shutdown/reboot; fork bombs; recursive ``chmod``/``chown`` of such
    targets; and piping downloads into a shell/interpreter (``curl … | sh``).
    """
    if not command or not command.strip():
        return False
    cwd_s = os.fspath(cwd) if cwd is not None else None
    squeezed = re.sub(r"\s+", "", command)
    if _FORK_BOMB.search(squeezed):
        return True
    if _PIPE_TO_INTERPRETER.search(command) or _SHELL_FROM_NETWORK.search(command) \
            or _DOT_FROM_NETWORK.search(command):
        return True
    parsed = parse_shell(command)
    if parsed.parse_error and not parsed.commands:
        # unparsable (e.g. unbalanced quotes): fall back to a naive split
        chunks = re.split(r"&&|\|\||[;|&\n]", command)
        commands = [SimpleCommand(argv=[w.strip("'\"") for w in c.split()]) for c in chunks]
    else:
        commands = parsed.commands
    here = cwd_s                         # follows ``cd`` inside this command line
    for sc in commands:
        for op, target in sc.redirects:
            if op in _OUT_REDIRECTS and _DISK_DEVICE.match(target):
                return True
        if not sc.argv:
            continue
        if sc.argv[0] in ("cd", "pushd"):
            here = _cd_target(sc.argv[1:], here)
            continue
        argv = sc.argv
        if here and cwd_s and here != cwd_s:
            # ``cd /tmp/x && rm -rf ./*`` empties /tmp/x, not the project
            argv = [os.path.join(here, a) if _HERE_RELATIVE.match(a) else a for a in argv]
        if _catastrophic_simple(argv, cwd_s):
            return True
    return False


#: Targets that mean "this directory" or above it — the ones a ``cd`` changes the meaning of.
_HERE_RELATIVE = re.compile(r"^(?:\.{1,2}(?:/.*)?|\*.*|\.git(?:/.*)?)$")


def _cd_target(args: list[str], here: str | None) -> str | None:
    """Where ``cd args`` lands, or ``here`` when that cannot be known (``cd $X``, ``cd -``)."""
    if not args:
        return os.path.expanduser("~")
    target = args[0]
    if len(args) > 1 or target == "-" or any(ch in target for ch in "$`"):
        return here
    target = os.path.expanduser(target)
    if not os.path.isabs(target):
        if not here:
            return here
        target = os.path.join(here, target)
    return os.path.realpath(target)


# --------------------------------------------------------------------------- #
# Command prefixes for "always allow"
# --------------------------------------------------------------------------- #

_TWO_WORD_TOOLS = frozenset({
    "git", "npm", "pnpm", "yarn", "bun", "npx", "bunx", "cargo", "go", "docker", "docker-compose",
    "podman", "kubectl", "brew", "pip", "pip3", "uv", "poetry", "make", "swift", "gh", "dotnet",
    "mvn", "gradle", "./gradlew", "deno", "terraform", "rustup", "conda", "mamba", "pipx", "just",
    "task", "composer", "bundle", "rake", "rails", "mix", "flutter", "dart", "xcrun", "hatch",
    "pdm", "tox", "nox", "turbo", "nx", "pod", "helm", "systemctl", "launchctl", "apt", "apt-get",
})
_INTERPRETERS = re.compile(r"^(?:python[\d.]*|node|ruby|perl|php|bash|sh|zsh|Rscript|julia|lua)$")


def command_prefix(argv: list[str]) -> tuple[str, ...]:
    """The part of a simple command that "always allow" remembers.

    ``git commit -m x`` → ``("git", "commit")``; ``python -m pytest -q`` →
    ``("python", "-m", "pytest")``; ``python app.py`` → ``("python", "app.py")``;
    ``ls -la`` → ``("ls",)``.
    """
    words = [w for w in argv if w]
    if not words:
        return ()
    head = words[0]
    rest = words[1:]
    name = head.rsplit("/", 1)[-1] if head.startswith(tuple(d + "/" for d in _SYSTEM_BIN_DIRS)) else head
    if name in ("sudo", "doas"):
        inner = command_prefix(_strip_wrappers(words))
        return (name, *inner) if inner else (name,)
    if _INTERPRETERS.match(name):
        if not rest:
            return (name,)
        if rest[0] == "-m" and len(rest) > 1:
            return (name, "-m", rest[1])
        return (name, rest[0])
    if name in _TWO_WORD_TOOLS:
        i = 0
        while i < len(rest) and rest[i].startswith("-"):
            # "git -C dir status", "npm --prefix x run"
            i += 2 if rest[i] in ("-C", "--prefix", "-c", "--git-dir", "--work-tree", "-f", "--file",
                                  "--manifest-path", "--cwd", "--dir", "-w", "--workspace") else 1
        if i < len(rest):
            sub = rest[i]
            if name in ("npm", "pnpm", "yarn", "bun") and sub in ("run", "run-script", "exec", "x") \
                    and i + 1 < len(rest):
                return (name, sub, rest[i + 1])
            return (name, sub)
        return (name,)
    return (name,)


def _command_of(tool: Any, args: Any, info: Any = None) -> str:
    """The exact command a shell tool will run (same normalisation as the tool).

    For ``bash`` the arguments come first: its preview is written for people
    (``Co robi: …`` above the command, a note about leaving the project below
    it), and judging that prose as shell — or the model's own words about the
    command — would decide on something that never runs. Other ``exec`` tools
    have no command argument; their preview stands in for one.
    """
    try:
        from localcode.agent.tools.shell import command_from_args  # lazy: avoids an import cycle
    except ImportError:  # pragma: no cover - shell tool module missing
        command_from_args = None  # type: ignore[assignment]
    if _tool_name(tool) == "bash" and command_from_args is not None:
        try:
            command = command_from_args(args)
        except Exception:
            command = ""
        if command:
            return command
    if info is not None and getattr(info, "preview", None):
        return str(info.preview)
    if command_from_args is not None:
        try:
            return command_from_args(args)
        except Exception:
            pass
    if isinstance(args, dict):
        return str(args.get("command") or args.get("cmd") or "")
    return str(args or "")


def _tool_name(tool: Any) -> str:
    return tool if isinstance(tool, str) else str(getattr(tool, "name", "") or "")


def _tool_kind(tool: Any) -> str:
    if isinstance(tool, str):
        try:
            from localcode.agent.tools import get_tool  # lazy
        except ImportError:  # pragma: no cover
            return "exec"
        t = get_tool(tool)
        return t.kind if t is not None else "exec"
    kind = getattr(tool, "kind", None)
    return kind if kind in ("read", "write", "exec", "meta") else "exec"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


class PermissionPolicy:
    """Decides whether a tool call runs, needs the user's approval, or is refused.

    ``cwd`` (optional) lets :func:`is_catastrophic` recognise ``rm -rf .`` of the
    project root; tools also flag such commands with ``DANGER_FLAG``.
    """

    def __init__(self, mode: str = "default", *, cwd: str | os.PathLike[str] | None = None,
                 always_tools: Iterable[str] = (), always_prefixes: Iterable[Iterable[str]] = (),
                 always_commands: Iterable[str] = ()) -> None:
        self._mode = normalize_mode(mode)
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self._always_tools: set[str] = set(always_tools)
        self._always_prefixes: set[tuple[str, ...]] = {tuple(p) for p in always_prefixes if p}
        self._always_commands: set[str] = set(always_commands)

    # mode -------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode = normalize_mode(value)

    def set_mode(self, value: str) -> None:
        self.mode = value

    # decisions --------------------------------------------------------------
    def decide(self, tool: Any, args: dict[str, Any] | None, info: Any = None) -> Decision:
        """``info`` is what ``tool.permission()`` returned; ``None`` means the tool
        reported that nothing needs approval (e.g. an edit that cannot apply)."""
        kind = _tool_kind(tool)
        title = str(getattr(info, "title", "") or "")
        if kind in ("read", "meta"):
            # Reading is free inside the project and asks outside it, in every
            # mode — see CodingTool.permission for why.
            if OUTSIDE_PROJECT_FLAG in title and _tool_name(tool) not in self._always_tools:
                return "ask"
            return "allow"
        if self._mode == "plan":
            return "deny"
        if kind == "write":
            if info is None:
                return "allow"
            if OUTSIDE_PROJECT_FLAG in title or DANGER_FLAG in title:
                return "ask"
            if self._mode in ("yolo", "accept_edits", "auto"):     # auto: edits in the project are free
                return "allow"
            return "allow" if _tool_name(tool) in self._always_tools else "ask"
        # exec (and anything unknown)
        command = _command_of(tool, args, info)
        if (DANGER_FLAG in title or OUTSIDE_PROJECT_FLAG in title or REMOTE_FLAG in title
                or is_catastrophic(command, self.cwd)):
            return "ask"
        if self._mode == "yolo":
            return "allow"
        if command and is_read_only_command(command):
            return "allow"
        if self._is_remembered_command(command):
            return "allow"
        if not command and _tool_name(tool) in self._always_tools:
            return "allow"
        return "ask"

    def needs_human(self, tool: Any, args: dict[str, Any] | None, info: Any = None) -> bool:
        """Auto mode: must this call go to the user rather than to the reviewer model?

        A move of the conversation to a provider (``REMOTE_FLAG``) and anything
        catastrophic (``DANGER_FLAG``, :func:`is_catastrophic`) are the user's
        call in every mode — a model is never the one that says yes to those.
        A call outside the project (``OUTSIDE_PROJECT_FLAG``) is the reviewer's,
        which is told to be strictest there.
        """
        title = str(getattr(info, "title", "") or "")
        if DANGER_FLAG in title or REMOTE_FLAG in title:
            return True
        if _tool_kind(tool) != "exec":
            return False
        return is_catastrophic(_command_of(tool, args, info), self.cwd)

    # memory -----------------------------------------------------------------
    def remember_always(self, tool: Any, args: dict[str, Any] | None) -> None:
        """Remember the user's "always allow" answer for this kind of call."""
        kind = _tool_kind(tool)
        name = _tool_name(tool)
        if kind != "exec":
            if name:
                self._always_tools.add(name)
            return
        command = _command_of(tool, args)
        if not command.strip():
            if name:
                self._always_tools.add(name)
            return
        parsed = parse_shell(command)
        if parsed.parse_error or parsed.has_substitution or parsed.has_subshell or not parsed.commands:
            self._always_commands.add(command.strip())
            return
        for sc in parsed.commands:
            if not sc.argv or _simple_is_read_only(sc.argv):
                continue
            prefix = command_prefix(sc.argv)
            if prefix:
                self._always_prefixes.add(prefix)

    def is_remembered(self, tool: Any, args: dict[str, Any] | None) -> bool:
        if _tool_kind(tool) != "exec":
            return _tool_name(tool) in self._always_tools
        return self._is_remembered_command(_command_of(tool, args))

    def _is_remembered_command(self, command: str) -> bool:
        cmd = command.strip()
        if not cmd:
            return False
        if cmd in self._always_commands:
            return True
        if not self._always_prefixes:
            return False
        parsed = parse_shell(cmd)
        if parsed.parse_error or parsed.has_substitution or parsed.has_subshell or not parsed.commands:
            return False
        for sc in parsed.commands:
            if not _redirects_are_safe(sc.redirects):
                return False
            if any(a.split("=", 1)[0].rstrip("+") not in _SAFE_ASSIGNMENTS for a in sc.assignments):
                return False
            if not sc.argv or _simple_is_read_only(sc.argv):
                continue
            words = tuple(sc.argv)
            if not any(words[:len(p)] == p for p in self._always_prefixes):
                return False
        return True

    def describe_always(self, tool: Any, args: dict[str, Any] | None) -> str:
        """Polish description of what "always allow" would remember (for the UI button)."""
        kind = _tool_kind(tool)
        name = _tool_name(tool)
        if kind != "exec":
            return f"Zawsze zezwalaj na {name} w tej sesji"
        command = _command_of(tool, args)
        parsed = parse_shell(command)
        if parsed.parse_error or parsed.has_substitution or parsed.has_subshell or not parsed.commands:
            return f"Zawsze zezwalaj na dokładnie to polecenie: {command.strip()}"
        prefixes = []
        for sc in parsed.commands:
            if sc.argv and not _simple_is_read_only(sc.argv):
                p = " ".join(command_prefix(sc.argv))
                if p and p not in prefixes:
                    prefixes.append(p)
        if not prefixes:
            return "Polecenie jest tylko do odczytu — nie wymaga zgody"
        return "Zawsze zezwalaj na polecenia zaczynające się od: " + ", ".join(prefixes)

    def forget_all(self) -> None:
        self._always_tools.clear()
        self._always_prefixes.clear()
        self._always_commands.clear()

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable state (mode + remembered rules) for session persistence."""
        return {
            "mode": self._mode,
            "tools": sorted(self._always_tools),
            "prefixes": sorted([list(p) for p in self._always_prefixes]),
            "commands": sorted(self._always_commands),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any], *, cwd: str | os.PathLike[str] | None = None) -> "PermissionPolicy":
        mode = data.get("mode") or "default"
        try:
            normalize_mode(mode)
        except ValueError:
            mode = "default"
        return cls(mode, cwd=cwd, always_tools=data.get("tools") or (),
                   always_prefixes=data.get("prefixes") or (),
                   always_commands=data.get("commands") or ())

    def __repr__(self) -> str:
        return (f"PermissionPolicy(mode={self._mode!r}, tools={sorted(self._always_tools)}, "
                f"prefixes={sorted(self._always_prefixes)})")
