"""Model-facing prompts of the coding agent, sized for small local models.

Everything the model reads from us is English (small models follow English
instructions best); the system prompt ends with "Respond in the language the
user writes in." so the user still gets answers in Polish (or whatever they
write in).

Three tiers keep the fixed prompt cost proportional to the window
(sizes measured with ``context.estimate_tokens``, which over-estimates English
prose by ~30 %, with the environment block but without project instructions):

* ``tiny``  (n_ctx ≤ 8192):  ≤ ~350 tokens — one terse rule list;
* ``small`` (n_ctx ≤ 32768): ≤ ~800 tokens — short sections;
* ``large`` (bigger):        ≤ ~1200 tokens — full guidance.

Only tools listed in ``tool_names`` are mentioned, so plan mode (read-only
tools) never talks about editing or running commands. When the chat template
has no native tool support (``native_tools=False``) the formatter appends a
``<tool_call>`` format block to the system message; the prompt then refers to
"the format described below" instead of the function-calling interface.

Build the system prompt once per session (or model change) and keep it: it
heads every request, so any change — even the git "dirty" flag in the
environment block — invalidates the engine's whole KV prefix cache. Mode
switches in the middle of a session can use ``mode_notice()`` in the next user
message instead of rebuilding the prompt.

Also here: environment discovery (``get_env``), project instruction files
(``LOCALCODE.md``, ``AGENTS.md``, ``CLAUDE.md`` — ``load_project_instructions``),
the compaction prompt used by ``ContextManager`` summaries, and the short nudges
the agent loop injects (repeated call, malformed call, empty response).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Literal, Sequence

from localcode.agent.context import estimate_tokens, truncate

__all__ = [
    "EMPTY_RESPONSE_NUDGE",
    "INSTRUCTION_FILES",
    "LANGUAGE_LINES",
    "LANGUAGE_REMINDERS",
    "LANGUAGE_RULE",
    "detect_language",
    "MALFORMED_CALL_HINT",
    "REPEATED_CALL_NUDGE",
    "ULTRACODE_NUDGE",
    "Tier",
    "build_system_prompt",
    "compaction_prompt",
    "get_env",
    "get_env_async",
    "is_nudge",
    "load_project_instructions",
    "malformed_call_hint",
    "mode_notice",
    "project_instructions_budget",
    "tier_for",
]

Tier = Literal["tiny", "small", "large"]

LANGUAGE_RULE = ("Always answer in the language of the user's latest message — a question in Polish "
                 "gets an answer in Polish — even though these instructions are in English.")

#: Words that only (or mostly) appear in one of the two languages LocalCode's
#: users write in. Diacritics settle Polish on their own; these decide the rest
#: ("zrob mi to", "which model is this").
_PL_WORDS = frozenset("""
jest nie sie się mi mnie mam masz jak co czy to ten ta te tak dla przez jestem czesc cześć hej
zrob zrób zrobic zrobić prosze proszę dzieki dzięki jaki jakie jaka ktory który ktora która teraz
moze może ustaw wlacz włącz zaladuj załaduj napisz pokaz pokaż dodaj sprawdz sprawdź wez weź daj
dlaczego czemu gdzie kiedy ile wszystko tylko jeszcze juz już bo albo lub tez też ze że sa są
bedzie będzie byl był mozesz możesz potrzebuje potrzebuję chce chcę napraw uruchom pobierz
""".split())
_EN_WORDS = frozenset("""
the is are was were what which who how why where when i am you your we they it this that these
please can could would should will do does did have has with from for and but not now talking
""".split())


def detect_language(text: str) -> str | None:
    """``"pl"``, ``"en"`` or ``None`` (too short or too mixed to tell)."""
    import re as _re

    if not text or not text.strip():
        return None
    if _re.search(r"[ąćęłńśźżĄĆĘŁŃŚŹŻ]", text):
        return "pl"
    words = _re.findall(r"[a-ząćęłńóśźż]+", text.lower())
    pl = sum(1 for w in words if w in _PL_WORDS)
    en = sum(1 for w in words if w in _EN_WORDS)
    if pl > en and pl >= 1:
        return "pl"
    if en > pl and en >= 2:
        return "en"
    return None


#: Appended to the system prompt once the user's language is known.
LANGUAGE_LINES = {
    "pl": "The user writes in Polish. Write every answer in Polish.",
    "en": "The user writes in English. Write every answer in English.",
}
#: Added to the user's latest message in the prompt a *small local* model sees
#: (never to the stored conversation): a 0.6B model follows a reminder next to
#: the question far more reliably than a rule at the end of a long prompt.
LANGUAGE_REMINDERS = {"pl": "(Odpowiedz po polsku.)"}
INSTRUCTION_FILES: tuple[str, ...] = ("LOCALCODE.md", "AGENTS.md", "CLAUDE.md")

_TIER_LISTING = {"tiny": 15, "small": 30, "large": 30}          # max entries shown
_TIER_LISTING_CHARS = {"tiny": 200, "small": 450, "large": 700}  # and max characters
_HIDDEN_ALLOWED = {".github", ".gitlab-ci.yml", ".env.example"}
_NOISE_ENTRIES = {
    "__pycache__", "node_modules", "venv", "env", ".DS_Store", "Thumbs.db",
    "target", ".build", "DerivedData",
}
_MAX_INSTRUCTION_BYTES = 256 * 1024


#: What ``/init`` asks the agent to do. It lives here, not in the terminal app,
#: because both interfaces send it and two copies would drift the moment one of
#: them was improved.
INIT_PROMPT = (
    "Analyse this project and write a LOCALCODE.md file in its root. Keep it under 40 lines. "
    "Cover: what the project is, how to build/run/test it (exact commands you verified from the "
    "repository, not guesses), the directory layout, and the conventions a new contributor must "
    "follow. Look at the build files, the README and the source layout first. If LOCALCODE.md "
    "already exists, improve it instead of replacing it. Answer in Polish."
)


def tier_for(n_ctx: int) -> Tier:
    """Prompt tier for a context window of ``n_ctx`` tokens."""
    if n_ctx <= 8192:
        return "tiny"
    if n_ctx <= 32768:
        return "small"
    return "large"


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #


def _tools_line(names: Sequence[str], available: set[str]) -> str:
    present = [n for n in names if n in available]
    return "/".join(present)


#: Small models often invent a project root ("/src", "/") instead of using the
#: working directory, and then spend the whole turn lost in the filesystem. One
#: explicit sentence per tier costs ~20 tokens and prevents it.
_PATHS_RULE_TINY = (
    "Paths are relative to cwd in <env> (its files are listed there). "
    "Never start a path with / and never guess a directory."
)
_PATHS_RULE = (
    "- File paths are relative to the working directory shown in Environment, whose files are listed "
    "there: start from them. Never start a path with `/` and never invent a directory."
)


def _tiny_rules(has: set[str], mode: str, native_tools: bool) -> list[str]:
    rules: list[str] = [_PATHS_RULE_TINY, _ATTACHMENTS_RULE_TINY]
    explore = _tools_line(("list_dir", "glob", "grep"), has)
    if "read_file" in has:
        lead = f"Explore with {explore}, then" if explore else "Always"
        rules.append(f"{lead} read_file before changing code. Never invent file contents or APIs.")
    elif explore:
        rules.append(f"Explore with {explore}. Never invent file contents or APIs.")
    if "edit_file" in has:
        rules.append(
            "edit_file: copy old_string exactly from read_file output (no line-number prefix), "
            "with enough lines to be unique. Small edits."
        )
    if "write_file" in has and mode != "plan":
        rules.append("Asked to write or make something? Create the file with write_file; "
                     "do not paste it into the answer.")
    step = "One step at a time."
    if "bash" in has and mode != "plan":
        step += " After changes, run the tests with bash."
    rules.append(step)
    if "bash" in has and mode != "plan":
        rules.append("bash can run anything. Serve a page in the background: "
                     "`nohup python3 -m http.server 8000 > /tmp/localcode-http.log 2>&1 &`, "
                     "then give http://localhost:8000.")
    how = "the tool-calling interface" if native_tools else "the tool call format below"
    rules.append(f"Call tools only via {how}, with valid JSON arguments. Stop calling tools when done.")
    if mode == "plan":
        rules.append(
            "PLAN MODE (read-only): investigate, then reply with a short numbered plan (files, changes, "
            "how to verify). Do not edit files or run commands that change anything."
        )
    elif "bash" in has:
        rules.append("No destructive commands (rm -rf, git reset --hard, force push) unless asked.")
    if mode == "auto":
        rules.append("A reviewer checks every change and command; if it rejects one, don't retry it.")
    rules.append("Be brief. When done, summarize in 1-3 sentences.")
    return rules


#: "napisz mi stronę o bananach" after a "hej" read to a 4B model as a chat
#: request: it pasted the HTML into its answer, claimed it had saved index.html,
#: and never called a tool in the whole session. Nothing in the prompt said that
#: asking for something to be made means making it in the project.
_DELIVER_RULE = ("When the user asks you to write, create or make something (a page, a script, "
                 "an app), create it as files in the working directory with write_file. Do not "
                 "paste whole files into your answer; say which file you created.")
#: The same session then said it "cannot run a server — there is no python3".
#: Both false: bash runs anything, and a job started with nohup … & outlives the
#: call (agent/tools/shell.py). The model simply did not know it could.
_SERVE_RULE = ("You can run programs with bash. To show a web page on localhost, start a server "
               "in the background, e.g. `nohup python3 -m http.server 8000 > /tmp/localcode-http.log "
               "2>&1 &`, check it with curl and give the user http://localhost:8000. Never claim "
               "you cannot run commands or that a tool is missing without trying it first.")
#: Files the user attaches or @-mentions are inlined into their message as
#: ``<file path="…">…</file>`` (directories as ``<dir path="…">``) by the client
#: (tui/completion.py). Without being told, small models treat the tag as a
#: reference and ask the user to paste the file, or re-read it with a tool — the
#: exact complaint behind this prompt. One sentence stops both.
_ATTACHMENTS_RULE = ('Files the user attached or @-mentioned are pasted into the conversation inside '
                     '`<file path="…">` blocks (directories as `<dir path="…">`): that is their real, '
                     'full content — use it directly. Never ask the user to paste, send or locate a '
                     'file, and do not re-read a file already shown. To see any other file, open it '
                     'yourself with a tool.')
_ATTACHMENTS_RULE_TINY = ('Attached or @-mentioned files are pasted in <file path=…> blocks — that is '
                          'their content; use it, never ask the user to resend a file.')


def _small_sections(has: set[str], mode: str, native_tools: bool) -> list[str]:
    work: list[str] = [_PATHS_RULE, f"- {_ATTACHMENTS_RULE}"]
    explore = _tools_line(("list_dir", "glob", "grep"), has)
    if explore:
        work.append(f"- Explore first: find the relevant code with {explore}.")
    if "read_file" in has:
        work.append("- Read a file with read_file before editing it. Never invent file contents, APIs or command output.")
    if "edit_file" in has:
        work.append(
            "- edit_file: copy old_string EXACTLY from read_file output, WITHOUT the line-number prefix, "
            "with enough surrounding lines to be unique. Make small, focused edits."
        )
    if "write_file" in has:
        work.append("- Use write_file only for new files or full rewrites.")
        if mode != "plan":
            work.append(f"- {_DELIVER_RULE}")
    work.append("- Work one step at a time: call a tool, check the result, then decide the next step.")
    if "bash" in has and mode != "plan":
        work.append("- After changes, run the tests or build with bash; if something fails, fix the cause.")
        work.append(f"- {_SERVE_RULE}")
    if "todo_write" in has and mode != "plan":
        work.append("- For multi-step tasks keep a short todo list with todo_write.")
    if "delegate" in has:
        work.append(f"- {_DELEGATE_RULE}")
    work.append("- Follow the project's existing style. If the request is unclear, ask a short question.")

    how = (
        "the provided tool-calling interface"
        if native_tools
        else "the tool call format described below (one call per block)"
    )
    calls = [
        f"- Call tools only through {how}, with valid JSON arguments matching the schema.",
        "- Do not repeat a call with the same arguments; use the result you already have.",
        "- Long outputs are truncated; page with offset/limit or narrow with grep.",
        "- When the task is done, stop calling tools and answer.",
    ]
    answers = [
        "- Be concise and direct. Refer to code as path:line.",
        "- When you finish, give a 1-3 sentence summary of what you did.",
    ]
    sections = [
        "# How to work\n" + "\n".join(work),
        "# Tool calls\n" + "\n".join(calls),
        "# Answers\n" + "\n".join(answers),
    ]
    if mode == "plan":
        sections.append(
            "# Plan mode (read-only)\n"
            "You may only read and search. Investigate the code, then reply with a numbered plan: the files "
            "and functions to change, what to change, and how to verify it. Do not edit files or run commands "
            "that change anything; the user approves the plan first."
        )
    safety = ["- Never run destructive commands (rm -rf, git reset --hard, git push --force) unless asked."]
    if mode != "plan":
        safety.append("- Don't commit or push unless asked. If the user denies a tool call, don't retry it.")
    if mode == "yolo":
        safety.append("- All tool calls run without approval, so be careful.")
    elif mode == "auto":
        safety.append("- A strict reviewer checks every change and command. If it rejects one, "
                      "don't retry it: take a safer way or ask the user.")
    sections.append("# Safety\n" + "\n".join(safety))
    return sections


def _large_sections(has: set[str], mode: str, native_tools: bool) -> list[str]:
    work: list[str] = [_PATHS_RULE, f"- {_ATTACHMENTS_RULE}"]
    explore = _tools_line(("list_dir", "glob", "grep"), has)
    if explore:
        work.append(f"- Understand before changing: find the relevant code with {explore}.")
    if "read_file" in has:
        work.append(
            "- Read a file with read_file before editing it, and read enough of it to know the context. "
            "Never guess file contents, function names, APIs or command output."
        )
    if "edit_file" in has:
        work.append(
            "- edit_file: copy old_string EXACTLY from the read_file output, WITHOUT the line-number prefix, "
            "including enough surrounding lines to be unique in the file. Prefer several small, focused edits "
            "over one huge edit. If an edit fails, read the file again and retry with the current text."
        )
    if "write_file" in has:
        work.append("- Use write_file only for new files or complete rewrites of small files.")
        if mode != "plan":
            work.append(f"- {_DELIVER_RULE}")
    work.append("- Work one step at a time: call a tool, look at the result, then decide the next step.")
    if "bash" in has and mode != "plan":
        work.append(
            "- Verify your work: run the tests, linter or build with bash when the project has them. "
            "If something fails, read the error, fix the cause, and run it again."
        )
        work.append(f"- {_SERVE_RULE}")
    if "todo_write" in has and mode != "plan":
        work.append("- For tasks with several steps, keep a short todo list with todo_write and update it as you go.")
    if "delegate" in has:
        work.append(f"- {_DELEGATE_RULE} A broad search whose output you will not need again is "
                    "a helper's job too.")
    work.append(
        "- Follow the project's conventions, style and libraries; don't add dependencies without asking."
    )
    work.append("- If the request is ambiguous or a decision belongs to the user, ask one short question instead of guessing.")

    how = (
        "the provided tool-calling interface"
        if native_tools
        else "the tool call format described below"
    )
    calls = [
        f"- Call tools only through {how}, with valid JSON arguments that match the tool's schema. "
        "Never describe a tool call in prose instead of making it.",
        "- Do not repeat a call with the same arguments; use the result you already have.",
        "- Tool outputs may be truncated (marked with [… omitted …]); page with offset/limit or narrow with grep. "
        "Such markers and '[old output … removed]' stubs are not file content: never write them into files.",
        "- When the task is done, stop calling tools and give your answer.",
    ]
    answers = [
        "- Be concise and direct: no filler, no repeating the question, no long explanations unless asked.",
        "- Refer to code as path:line.",
        "- When you finish, reply with a 1-3 sentence summary of what you changed and anything the user must do.",
    ]
    sections = [
        "# How to work\n" + "\n".join(work),
        "# Tool calls\n" + "\n".join(calls),
        "# Answers\n" + "\n".join(answers),
    ]
    if mode == "plan":
        sections.append(
            "# Plan mode (read-only)\n"
            "You are in plan mode: you may only read and search; do not edit files and do not run commands "
            "that change anything. Investigate the code, then reply with a concrete plan: numbered steps with "
            "the files and functions to change, what to change and how to verify it. Do not start implementing; "
            "the user will review and approve the plan first."
        )
    safety = [
        "- Never run destructive or irreversible commands (rm -rf, git reset --hard, git push --force, "
        "deleting data) unless the user explicitly asked for it.",
        "- Stay inside the project directory unless the task requires otherwise. Never reveal secrets such as "
        "API keys or tokens.",
    ]
    if mode != "plan":
        safety.append("- Don't commit, push or publish unless asked. If the user denies a tool call, don't retry it; adapt or ask.")
    if mode == "yolo":
        safety.append("- All tool calls run without approval in this mode, so be extra careful.")
    elif mode == "auto":
        safety.append("- Auto mode: a strict reviewer model checks every file change and command before it "
                      "runs, and may reject it with a reason. Don't retry a rejected call; take a safer "
                      "way or ask the user.")
    elif mode == "accept_edits":
        safety.append("- File edits are applied automatically; commands still need the user's approval.")
    sections.append("# Safety\n" + "\n".join(safety))
    return sections


#: When to hand work to a helper — concrete, because "for a broad search" was
#: read as "almost never": a real session of ~480 tool calls (game-bot work:
#: matches of several minutes each, replays to unpack, logs to analyse) used
#: ``delegate`` four times.
_DELEGATE_RULE = ("When you have 2+ independent things to run, read or analyse — several matches or "
                  "configurations to run, several logs to analyse, several modules to understand — "
                  "start background helpers for them (delegate with wait=false, one call each, all in "
                  "one step) instead of doing them one by one, and keep working meanwhile. Helpers "
                  "can read and run commands, not edit files; each sees only its task, so give it "
                  "the paths, the commands and what to report.")


def _ultracode_section(tier: Tier, has: set[str], mode: str) -> str | None:
    """Effort "ultracode": lead a team of background helpers.

    Only where ``delegate`` is on the menu. Helpers go to the background by
    default here, so several ``delegate`` calls in one step cost the agent
    nothing to wait for — which is why the section asks for them first, in
    the first step, for any task with independent parts. Steps name only the
    tools this menu has.
    """
    if "delegate" not in has:
        return None
    plan = "todo_write" in has and mode != "plan"
    verify = "bash" in has and mode != "plan"
    if tier == "tiny":
        line = ("ULTRACODE: you lead helpers. In your FIRST step start a background helper "
                "(delegate) for each independent part of the task — several calls in that one "
                "step; reports arrive as notes while you keep working.")
        if mode != "plan":
            line += (" After your changes" + (", run the tests with bash, then" if verify else ",")
                     + " send a helper to review the changed files against the request; fix what it finds.")
        return line
    steps: list[str] = [
        "Fan out FIRST: in your first step, start a background helper (delegate) for every part "
        "of the task that can be run, read or analysed on its own — several matches or "
        "configurations, several logs, several modules or questions. Put all these delegate calls "
        "in that one step (typically 3-8). Doing independent parts one by one yourself is the "
        "wrong way in this mode.",
        "Helpers work in the background (wait=false is the default here): keep working yourself; "
        "each report arrives as a note when its helper finishes. Use wait=true only when your "
        "very next step needs the answer.",
        "Helpers can read files and run commands, not edit files. Each sees only its task: give it "
        "the paths, the exact commands and what to report.",
    ]
    if plan:
        steps.append("Keep the plan in todo_write while the helpers work.")
    if mode == "plan":
        steps.append("Combine their reports into the plan; check anything they disagree on yourself.")
    else:
        steps.append("Make the changes yourself, based on what the helpers reported. New "
                     "independent work later in the task (more runs, more logs) goes to new helpers too.")
        if verify:
            steps.append("Verify: run the tests or the build and fix every failure.")
        steps.append("Before you finish, send one or more helpers to critically review the changed "
                     "files against the user's request (name the files and restate the request). "
                     "Fix what they find" + (", then verify again." if verify else "."))
    body = "\n".join(f"{n}. {text}" for n, text in enumerate(steps, 1))
    return ("# Ultracode (orchestrator)\nThe user chose the highest effort: you lead a team of "
            "helper agents and do not work alone.\n" + body)


def _env_block(tier: Tier, cwd: str, env: dict[str, Any]) -> str:
    cwd_s = str(env.get("cwd") or cwd)
    os_name = str(env.get("os") or "macOS")
    day = str(env.get("date") or date.today().isoformat())
    shell = env.get("shell")
    git = ""
    if env.get("is_git") or env.get("git_branch"):
        git = f"branch {env.get('git_branch') or 'unknown'}"
        dirty = env.get("git_dirty")
        if dirty is not None:
            git += ", uncommitted changes" if dirty else ", clean"
    elif "is_git" in env:
        git = "not a git repository"
    listing = [str(x) for x in env.get("listing") or []]
    total = max(int(env.get("listing_total") or 0), len(listing))
    shown: list[str] = []
    used = 0
    for name in listing[: _TIER_LISTING[tier]]:
        if shown and used + len(name) + 2 > _TIER_LISTING_CHARS[tier]:
            break
        shown.append(name)
        used += len(name) + 2
    more = total - len(shown)
    files = (", ".join(shown) + (f" (+{more} more)" if more > 0 else "")) if shown else ""

    if tier == "tiny":
        lines = [f"cwd: {cwd_s}", f"{os_name}, {day}" + (f", {shell}" if shell else "")]
        if git:
            lines.append(f"git: {git}")
        if files:
            lines.append(f"files in cwd: {files}")
        return "<env>\n" + "\n".join(lines) + "\n</env>"
    lines = [
        f"- Working directory: {cwd_s}",
        f"- OS: {os_name}" + (f", shell: {shell}" if shell else "") + f", date: {day}",
    ]
    if git:
        lines.append(f"- Git: {git}")
    if files:
        lines.append(f"- Files in the working directory: {files}")
    return "# Environment\n" + "\n".join(lines)


def build_system_prompt(
    *,
    cwd: str | os.PathLike[str],
    n_ctx: int,
    mode: str,
    native_tools: bool,
    tool_names: Sequence[str],
    project_instructions: str | None,
    env: dict[str, Any],
    skills: str | None = None,
    lang_notes: str | None = None,
    ultracode: bool = False,
) -> str:
    """The agent's system prompt for a window of ``n_ctx`` tokens.

    ``mode`` is the permission mode (``default``/``accept_edits``/``auto``/``yolo``/``plan``),
    ``tool_names`` the tools actually offered, ``env`` usually ``get_env(cwd)`` and
    ``project_instructions`` usually ``load_project_instructions(cwd, ...)``.
    ``ultracode`` adds the orchestrator section (only when ``delegate`` is offered).
    """
    tier = tier_for(n_ctx)
    has = set(tool_names)
    parts: list[str] = []
    if tier == "tiny":
        parts.append(
            "You are LocalCode, a coding agent in the user's project on their Mac. Use the tools to do the task."
        )
        parts.append("\n".join(f"- {r}" for r in _tiny_rules(has, mode, native_tools)))
    else:
        parts.append(
            "You are LocalCode, a coding agent running locally on the user's Mac. You help with software "
            "engineering tasks in the project: fixing bugs, adding features, refactoring, explaining code and "
            "running commands. You act only through the tools and see only what they return."
        )
        builder = _small_sections if tier == "small" else _large_sections
        parts.extend(builder(has, mode, native_tools))
    orchestrate = _ultracode_section(tier, has, mode) if ultracode else None
    if orchestrate:
        parts.append(orchestrate)
    parts.append(_env_block(tier, str(cwd), env or {}))
    if project_instructions and project_instructions.strip():
        heading = (
            "Project instructions (follow them):"
            if tier == "tiny"
            else "# Project instructions\nFollow these; they override the defaults above."
        )
        parts.append(heading + "\n" + project_instructions.strip())
    if skills and skills.strip():
        # Names and one-liners only: the bodies are read through the `skill`
        # tool when one actually applies, so a shelf of long procedures costs
        # a few dozen tokens per request instead of thousands.
        heading = ("Skills (read one with the `skill` tool before starting a task it covers):"
                   if tier == "tiny"
                   else "# Skills\nInstructions you can load on demand. When a task matches one, "
                        "call the `skill` tool with its name *before* you start.")
        parts.append(heading + "\n" + skills.strip())
    if lang_notes and lang_notes.strip():
        # Per-language guidance for the languages this project uses, already
        # rendered as a labelled block by ``langskills.build_section`` (mini for
        # local models, max for remote). Placed after project instructions so a
        # repo's own rules still win.
        parts.append(lang_notes.strip())
    parts.append(LANGUAGE_RULE)
    return "\n\n".join(parts)


def mode_notice(mode: str) -> str:
    """Short note for the next user message after the permission mode changed mid-session."""
    notices = {
        "plan": (
            "[Mode changed to plan mode: read-only. Investigate and reply with a numbered plan; "
            "do not edit files or run commands that change anything.]"
        ),
        "default": "[Mode changed: you may edit files and run commands again; the user approves each change.]",
        "accept_edits": "[Mode changed: file edits are applied automatically; commands still need approval.]",
        "auto": ("[Mode changed to auto: a reviewer model checks every file change and command and may "
                 "reject it with a reason; don't retry a rejected call.]"),
        "yolo": "[Mode changed: all tool calls run without approval. Be careful with destructive commands.]",
    }
    return notices.get(mode, f"[Mode changed to {mode}.]")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def _os_name() -> str:
    if platform.system() == "Darwin":
        version = platform.mac_ver()[0]
        return f"macOS {version}" if version else "macOS"
    return platform.system() or "unknown OS"


def _run_git(cwd: Path, args: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    env = dict(os.environ)
    env.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _listing(path: Path) -> tuple[list[str], int]:
    try:
        entries = list(os.scandir(path))
    except OSError:
        return [], 0
    dirs: list[str] = []
    files: list[str] = []
    for e in entries:
        name = e.name
        if name in _NOISE_ENTRIES or name.endswith(".egg-info"):
            continue
        if name.startswith(".") and name not in _HIDDEN_ALLOWED:
            continue
        try:
            is_dir = e.is_dir()
        except OSError:
            is_dir = False
        (dirs if is_dir else files).append(name + "/" if is_dir else name)
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    items = dirs + files
    return items, len(items)


def get_env(cwd: str | os.PathLike[str], *, max_entries: int = 30, timeout: float = 2.0) -> dict[str, Any]:
    """Facts about the working environment for the system prompt (blocking; see ``get_env_async``).

    Returns ``{cwd, os, date, shell, is_git, git_root, git_branch, git_dirty,
    listing, listing_total}``. Git calls have short timeouts; a missing ``git``
    binary, a non-git directory or a slow repository simply leave the git
    fields empty (``git_dirty`` is ``None`` when unknown).
    """
    path = Path(cwd).expanduser()
    env: dict[str, Any] = {
        "cwd": str(path),
        "os": _os_name(),
        "date": date.today().isoformat(),
        "shell": Path(os.environ.get("SHELL") or "").name or None,
        "is_git": False,
        "git_root": None,
        "git_branch": None,
        "git_dirty": None,
        "listing": [],
        "listing_total": 0,
    }
    root = _run_git(path, ["rev-parse", "--show-toplevel"], timeout)
    if root is not None and root.returncode == 0 and root.stdout.strip():
        env["is_git"] = True
        env["git_root"] = root.stdout.strip()
        branch = _run_git(path, ["symbolic-ref", "--short", "-q", "HEAD"], timeout)
        if branch is not None and branch.returncode == 0 and branch.stdout.strip():
            env["git_branch"] = branch.stdout.strip()
        else:
            sha = _run_git(path, ["rev-parse", "--short", "HEAD"], timeout)
            if sha is not None and sha.returncode == 0 and sha.stdout.strip():
                env["git_branch"] = f"detached at {sha.stdout.strip()}"
        status = _run_git(path, ["status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=dirty"], timeout)
        if status is not None and status.returncode == 0:
            env["git_dirty"] = bool(status.stdout.strip())
    items, total = _listing(path)
    env["listing"] = items[: max(0, max_entries)]
    env["listing_total"] = total
    return env


async def get_env_async(cwd: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """``get_env`` in a worker thread (it runs git subprocesses)."""
    return await asyncio.to_thread(get_env, cwd, **kwargs)


# --------------------------------------------------------------------------- #
# Project instructions
# --------------------------------------------------------------------------- #


def project_instructions_budget(n_ctx: int) -> int:
    """Token budget for project instruction files (~10 % of the window)."""
    return max(0, int(n_ctx * 0.10))


def _git_root_of(start: Path) -> Path | None:
    for d in (start, *start.parents):
        if (d / ".git").exists():
            return d
    return None


def _instruction_dirs(start: Path) -> list[Path]:
    """``start`` and its parents up to the git root (inclusive) or the home dir (exclusive)."""
    try:
        home: Path | None = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    root = _git_root_of(start)
    if root is not None and root == home and start != home:
        root = None  # a dotfiles repository in $HOME is not a project root
    dirs = [start]
    if root is not None:
        for d in start.parents if root != start else ():
            dirs.append(d)
            if d == root:
                break
    elif home is not None and home in start.parents:
        for d in start.parents:
            if d == home:
                break
            dirs.append(d)
    return dirs


def _read_text(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            data = fh.read(_MAX_INSTRUCTION_BYTES)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


def load_project_instructions(
    cwd: str | os.PathLike[str],
    max_tokens: int,
    *,
    names: Sequence[str] = INSTRUCTION_FILES,
) -> str | None:
    """Project instruction files, concatenated with headers, within ``max_tokens``.

    Looks for ``LOCALCODE.md``, ``AGENTS.md`` and ``CLAUDE.md`` in ``cwd`` and its
    parents up to the git root (or up to, but not including, the home
    directory outside git). The closest files come first and get the budget
    first; symlinked or identical files are included once. Each section starts
    with ``## <path relative to cwd>``. Returns ``None`` when nothing is found.
    """
    if max_tokens <= 0:
        return None
    try:
        start = Path(cwd).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    found: list[tuple[str, str]] = []
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for d in _instruction_dirs(start):
        for name in names:
            p = d / name
            try:
                if not p.is_file():
                    continue
                real = p.resolve()
            except OSError:
                continue
            if real in seen_paths:
                continue
            seen_paths.add(real)
            text = _read_text(p).strip()
            if not text:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            label = os.path.relpath(p, start) if d != start else name
            found.append((label, text))
    if not found:
        return None

    sections: list[str] = []
    remaining = max_tokens
    for label, text in found:
        header = f"## {label}\n"
        sep_cost = estimate_tokens("\n\n") if sections else 0
        cost = estimate_tokens(header) + estimate_tokens(text) + sep_cost
        if cost <= remaining:
            sections.append(header + text)
            remaining -= cost
            continue
        room = remaining - estimate_tokens(header) - sep_cost
        if room < 40:
            break
        chars = int(len(text) * room / max(1, estimate_tokens(text)))
        cut = truncate(text, chars, head_ratio=0.8)
        while estimate_tokens(cut) > room and chars > 50:
            chars = int(chars * 0.9)
            cut = truncate(text, chars, head_ratio=0.8)
        if estimate_tokens(cut) <= room:
            sections.append(header + cut)
        break
    return "\n\n".join(sections) if sections else None


# --------------------------------------------------------------------------- #
# Compaction prompt
# --------------------------------------------------------------------------- #

_COMPACTION_SECTIONS = (
    "Goal", "Done so far", "Files touched", "Key facts/decisions", "Current state/errors", "Next steps", "Todo",
)


def compaction_prompt(
    transcript: str = "",
    *,
    previous_summary: str | None = None,
    target_tokens: int = 500,
) -> list[dict[str, Any]]:
    """Messages asking the model for a dense structured summary of earlier work.

    ``transcript`` is the rendered part of the conversation being removed (see
    ``ContextManager``); ``previous_summary`` an older summary it replaces.
    The model must answer with plain text (no tools, no reasoning needed).
    """
    target = max(150, int(target_tokens))
    low = max(100, min(300, target // 2))
    system = (
        "You compress the history of a coding session so the work can continue in a small context window. "
        "After this, only your summary and the most recent messages will remain, so keep everything needed "
        "to continue the task.\n\n"
        "Write a dense, factual summary in English with exactly these sections:\n"
        "Goal: the user's request and requirements, in one or two lines.\n"
        "Done so far: completed steps and their results.\n"
        "Files touched: one line per file, \"path: what changed\" (files only read: \"path: read\").\n"
        "Key facts/decisions: names, APIs, commands, constraints and user preferences that matter later.\n"
        "Current state/errors: what works, what fails, the latest error messages (quoted briefly).\n"
        "Next steps: what remains to do, in order.\n"
        "Todo: the current todo list with status, or \"none\".\n\n"
        "Rules: keep exact file paths, function names, commands and error texts; do not invent anything; "
        "no code blocks longer than 3 lines; do not call tools; write nothing after the Todo section. "
        f"Length: about {low}-{target} tokens."
    )
    parts: list[str] = []
    if previous_summary and previous_summary.strip():
        parts.append("Previous summary (still valid unless the transcript says otherwise):\n" + previous_summary.strip())
    if transcript and transcript.strip():
        parts.append("Transcript of the earlier work:\n<transcript>\n" + transcript.strip() + "\n</transcript>")
    else:
        parts.append("Transcript of the earlier work: (empty)")
    parts.append("Write the summary now, with the sections " + ", ".join(_COMPACTION_SECTIONS) + ".")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# --------------------------------------------------------------------------- #
# Nudges used by the agent loop
# --------------------------------------------------------------------------- #


class _PromptTemplate(str):
    """A prompt string with ``{format_example}`` that can also be called.

    ``MALFORMED_CALL_HINT(example)`` and ``MALFORMED_CALL_HINT.format(format_example=example)``
    give the same text; without an example the format sentence is left out.
    """

    def __call__(self, format_example: str = "") -> str:
        return malformed_call_hint(format_example)


REPEATED_CALL_NUDGE = (
    "You already made this exact tool call and got the same result. Do not repeat it. "
    "Use the result you have, try a different approach, or give your final answer."
)

#: Ultracode: the agent has worked ``n`` steps without a helper (``Agent._nudge_to_delegate``).
ULTRACODE_NUDGE = (
    "[Ultracode: you have made {n} steps without helpers. If anything ahead is independent — "
    "several runs, logs, files or modules — start background helpers for it now (delegate with "
    "wait=false, one call each, in one step) and keep working while they run.]"
)

UNSAVED_CODE_NUDGE = (
    "You wrote the file's code in your answer but did not save it. Save it into the project with "
    "write_file now (pick a sensible name, e.g. index.html), then say in one sentence which file you "
    "created. Do not paste the code again."
)

EMPTY_RESPONSE_NUDGE = (
    "Your last reply was empty. Continue the task: call a tool, or give your final answer if you are done."
)

_MALFORMED_PREFIX = "Your last tool call could not be parsed."
_MALFORMED_TEXT = (
    _MALFORMED_PREFIX
    + " Call the tool again using exactly this format, with valid JSON arguments "
    "(double quotes, no comments, no trailing commas) and no other text inside the call:\n{format_example}"
)


def malformed_call_hint(format_example: str = "", detail: str = "") -> str:
    """Nudge after an unparsable tool call; ``format_example`` shows the expected syntax.

    ``detail`` is the parser's own complaint (``ParsedOutput.errors``), which is
    far more use to a small model than the generic sentence: "you wrote the tool
    name as an empty tag" tells it what to do differently, "could not be parsed"
    does not.  The prefix stays, so :func:`is_nudge` still recognises the message
    as the loop's and not the user's.
    """
    detail = (detail or "").strip()
    if detail:
        text = f"{_MALFORMED_PREFIX} {detail}"
        return f"{text}\n{format_example.strip()}" if format_example.strip() else text
    if format_example and format_example.strip():
        return _MALFORMED_TEXT.format(format_example=format_example.strip())
    return (
        _MALFORMED_PREFIX
        + " Call the tool again with valid JSON arguments (double quotes, no comments, no trailing commas)."
    )


MALFORMED_CALL_HINT = _PromptTemplate(_MALFORMED_TEXT)

_NUDGES = (REPEATED_CALL_NUDGE, EMPTY_RESPONSE_NUDGE)


def is_nudge(text: str) -> bool:
    """True for messages written by the agent loop rather than the user (nudges, mode notices)."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in _NUDGES or stripped.startswith(_MALFORMED_PREFIX):
        return True
    return stripped.startswith("[Mode changed") and stripped.endswith("]") and "\n" not in stripped


# --------------------------------------------------------------------------- #
# /goal
# --------------------------------------------------------------------------- #

#: Appended to the system prompt while a goal is active — appended, so the
#: prefix the model caches stays the same as without a goal.
GOAL_RULE = """ACTIVE GOAL (set by the user with /goal): {goal}
Keep working until this goal is fully achieved — do not stop to ask whether to continue. \
Verify the result (run the tests, the build, the script; read the file back). \
When it is achieved and verified, make a goal_done TOOL CALL with a short summary \
(a real tool call, not text in your answer). \
Do not install system-wide software (brew, global pip/npm) just to reach the goal; use the project's \
own environment. If it cannot be finished without the user (a decision, a password, a missing file \
or tool), call goal_done with blocked=true and say exactly what you need."""

#: Sent as the next user message when a step ends with a final answer while the
#: goal is still open. The user does not see it; the transcript shows a notice.
GOAL_CONTINUE = """[automatic] The goal is not marked as achieved yet: {goal}
Continue working on it (continuation {n} of {max}). Pick the next concrete step and do it with tools. \
If it is already achieved and verified, make the goal_done tool call now (a tool call, not text). \
If you are blocked, make the goal_done tool call with blocked=true."""

