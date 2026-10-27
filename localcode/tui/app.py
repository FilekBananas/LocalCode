"""The interactive terminal app: prompt, streaming transcript, slash commands.

One turn looks like this::

    ›  napraw testy                      prompt_toolkit (history, @ and / completion)
    ● Odczyt tests/test_app.py (1–200)   rich, printed above the live region
      ⎿ 200 linii
    ✻ Myślenie… (412 tokenów)            collapsed unless /think on
    Poprawiłem asercję …                 markdown, paragraph by paragraph
      12,4 tok/s · 3 kroki · 4,1 s

The prompt owns the keyboard between turns; during a turn
:class:`~localcode.tui.keys.InterruptWatcher` takes over so Esc and Ctrl+C can
stop the agent, and it steps aside whenever a permission question needs the
arrow keys.  Everything visible is Polish; the backend (local or attached to a
running server) is invisible from here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from localcode import config as lc_config
from localcode.agent.permissions import MODES
from localcode.runtime.templates import (
    AGENT_EFFORT_CHOICES,
    EFFORT_CHOICES,
    ULTRACODE,
    normalize_agent_effort,
    resolve_effort,
)
from localcode.hub.metadata import agent_model_warning, quant_warning
from localcode.tui import models_ui, render
from localcode.tui.backends import Backend, BackendError
from localcode.tui.chatmode import ChatSession, chat_command, run_chat_turn
from localcode.agent.prompts import INIT_PROMPT
from localcode.tui.commands import (
    help_table,
    is_command,
    keys_table,
    parse_command,
    runtime_patch,
    suggest,
)
from localcode.tui.images_in import collect_images, model_sees_images
from localcode.tui.completion import (
    LocalCodeCompleter,
    at_budget_chars,
    expand_at_references,
    summarize_attachments,
)
from localcode.tui.keys import InterruptWatcher
from localcode.tui.prompts import Choice, choose
from localcode.tui.stream import TurnView, make_console

log = logging.getLogger(__name__)

__all__ = ["LocalCodeTui", "run_tui"]

#: Order of the modes cycled by Shift+Tab.
MODE_CYCLE: tuple[str, ...] = ("default", "accept_edits", "auto", "plan", "yolo")
#: Ctrl+C twice within this many seconds on an empty prompt exits.
DOUBLE_CTRL_C_S = 2.0
#: How often the bottom toolbar's RAM figure is refreshed.
STATUS_REFRESH_S = 4.0
#: Tool output shown under the result line (bash and friends).
TAIL_LINES = 8
#: How long /compact waits for the events its command produces.
GOAL_CLEAR_WORDS = frozenset({"wyczyść", "wyczysc", "clear", "off", "stop", "usuń", "usun", "koniec"})
COMPACT_IDLE_S = 0.8
#: Rows kept free below the prompt for the completion menu.
MENU_LINES = 4

PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "#d97757 bold",
        "continuation": "#808080",
        "bottom-toolbar": "noreverse #808080 bg:default",
        "bottom-toolbar.model": "noreverse #d97757 bg:default",
        "bottom-toolbar.mode": "noreverse #ffcc66 bg:default",
        "bottom-toolbar.hot": "noreverse #ff8c66 bg:default",
        "completion-menu.completion": "bg:#333333 #dddddd",
        "completion-menu.completion.current": "bg:#d97757 #000000",
        "completion-menu.meta.completion": "bg:#333333 #808080",
    }
)



def _plural_steps(count: int) -> str:
    """„1 krok", „2 kroki", „5 kroków" — Polish needs all three."""
    if count == 1:
        return "krok"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "kroki"
    return "kroków"


class LocalCodeTui:
    """The terminal application. One instance per ``localcode`` process."""

    def __init__(self, backend: Backend, console: Console | None = None, *,
                 chat_mode: bool = False, history_path: Path | None = None) -> None:
        self.backend = backend
        self.console = console or make_console()
        self.chat_mode = chat_mode
        self.show_reasoning = bool(backend.settings.agent.show_reasoning)
        self.last_stats: dict[str, Any] | None = None
        self.todos: list[dict[str, Any]] = []
        self.status: dict[str, Any] | None = None
        self.chat: ChatSession | None = None
        self._running = True
        self._ctrl_c_at = 0.0
        self._turn_view: TurnView | None = None
        self._watcher: InterruptWatcher | None = None
        self._refresher: asyncio.Task[Any] | None = None
        #: Auto mode's NIE / no-answer, waiting for the question that shows it.
        self._held_review: dict[str, Any] | None = None
        #: Helpers working in the background right now ("h2", …), for the toolbar.
        self.background_helpers: set[str] = set()
        self._history_path = history_path if history_path is not None else _history_path()
        self.prompt: PromptSession[str] = PromptSession(
            history=FileHistory(str(self._history_path)) if self._history_path else InMemoryHistory(),
            style=PROMPT_STYLE,
            complete_while_typing=True,
            multiline=True,
            key_bindings=self._key_bindings(),
            completer=LocalCodeCompleter(backend.cwd, chat_mode=chat_mode),
            bottom_toolbar=self._toolbar,
            prompt_continuation=self._continuation,
            reserve_space_for_menu=MENU_LINES,
        )

    # ------------------------------------------------------------------- run
    async def run(self, first: str | None = None) -> int:
        """The main loop; ``first`` is the message given on the command line."""
        await self.show_welcome()
        if not await self._ensure_model():
            return 1
        self._refresher = asyncio.create_task(self._refresh_status(), name="localcode-status")
        try:
            if first and first.strip():
                self.console.print(Text("› " + first.strip(), style="lc.user"))
                await self._handle_line(first)
            while self._running:
                try:
                    text = await self._read()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if text is None:
                    break
                if not text.strip():
                    continue
                try:
                    await self._handle_line(text)
                except BackendError as exc:
                    self.console.print(render.error_line(str(exc)))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - never drop the user out of the app
                    log.exception("polecenie zakończone błędem")
                    self.console.print(render.error_line(f"{type(exc).__name__}: {exc}"))
        finally:
            if self._refresher is not None:
                self._refresher.cancel()
                with contextlib.suppress(BaseException):
                    await self._refresher
        self.console.print(Text("Do zobaczenia!", style="lc.dim"))
        return 0

    async def show_welcome(self) -> None:
        model = await self.backend.model()
        self.console.print(render.welcome_panel(
            model=model, cwd=str(self.backend.cwd), mode=self.backend.mode,
            n_ctx=self.backend.context_limit or None, backend=self.backend.kind,
            chat=self.chat_mode))
        for warning in self._model_warnings(model):
            self.console.print(Text(warning, style="lc.warn"))

    async def _ensure_model(self) -> bool:
        """First run: no models on disk at all → offer the curated ones."""
        if self.backend.model_key:
            return True
        models = await self.backend.list_models()
        if models:
            key = await models_ui.pick_model(self.backend, self.console,
                                             title="Wybierz model dla tej sesji")
            if key:
                await self.backend.set_model(key)
                return True
            self.console.print(Text("Bez modelu nie da się rozmawiać. Kończę.", style="lc.warn"))
            return False
        key = await models_ui.first_run(self.backend, self.console)
        if not key:
            return False
        with contextlib.suppress(BackendError):
            await self.backend.set_model(key)
        await self.show_welcome()
        return bool(self.backend.model_key)

    # ----------------------------------------------------------------- input
    async def _read(self) -> str | None:
        message: StyleAndTextTuples = [("class:prompt", "› " if not self.chat_mode else "💬 ")]
        return await self.prompt.prompt_async(message, refresh_interval=1.0)

    def _continuation(self, width: int, line_number: int, is_soft_wrap: bool) -> StyleAndTextTuples:
        return [("class:continuation", "… ".rjust(max(width, 2)))]

    def _key_bindings(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("enter")
        def _submit(event: Any) -> None:
            buffer = event.current_buffer
            text = buffer.text
            if text.rstrip("\r\n").endswith("\\"):
                stripped = text.rstrip()
                buffer.delete_before_cursor(len(text) - len(stripped) + 1)
                buffer.insert_text("\n")
                return
            if buffer.complete_state:
                completion = buffer.complete_state.current_completion
                if completion is not None:
                    buffer.apply_completion(completion)
                    return
                buffer.cancel_completion()
            buffer.validate_and_handle()

        @keys.add("escape", "enter")
        @keys.add("c-j")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        @keys.add("s-tab")
        def _cycle_mode(event: Any) -> None:
            current = self.backend.mode if self.backend.mode in MODE_CYCLE else MODE_CYCLE[0]
            nxt = MODE_CYCLE[(MODE_CYCLE.index(current) + 1) % len(MODE_CYCLE)]
            event.app.create_background_task(self._switch_mode(nxt))

        @keys.add("c-c")
        def _cancel(event: Any) -> None:
            buffer = event.current_buffer
            if buffer.text.strip():
                buffer.reset()
                self._ctrl_c_at = 0.0
                return
            now = time.monotonic()
            if now - self._ctrl_c_at < DOUBLE_CTRL_C_S:
                self._running = False
                event.app.exit(exception=EOFError, style="class:exiting")
                return
            self._ctrl_c_at = now
            run_in_terminal(lambda: self.console.print(
                Text("Naciśnij Ctrl+C jeszcze raz, aby wyjść (albo Ctrl+D).", style="lc.dim")))

        return keys

    async def _switch_mode(self, mode: str) -> None:
        try:
            await self.backend.set_mode(mode)
        except BackendError as exc:  # pragma: no cover - remote hiccup
            log.warning("nie udało się zmienić trybu: %s", exc)

    # --------------------------------------------------------------- toolbar
    def _toolbar(self) -> StyleAndTextTuples:
        backend = self.backend
        parts: StyleAndTextTuples = []
        model = (backend.model_key or "brak modelu").split("/")[-1]
        parts.append(("class:bottom-toolbar.model", f" {model}"))
        parts.append(("class:bottom-toolbar", " · "))
        parts.append(("class:bottom-toolbar.mode", render.mode_label(backend.mode)))
        effort = getattr(backend, "effort", None)
        if effort:
            parts.append(("class:bottom-toolbar", " · "))
            parts.append(("class:bottom-toolbar.mode", f"myślenie {effort}"))
        if self.background_helpers:
            parts.append(("class:bottom-toolbar", " · "))
            parts.append(("class:bottom-toolbar.mode", f"pomocnicy w tle: {len(self.background_helpers)}"))
        limit = backend.context_limit
        if limit:
            used = backend.context_used
            percent = round(used * 100 / limit)
            style = "class:bottom-toolbar.hot" if percent >= 80 else "class:bottom-toolbar"
            parts.append(("class:bottom-toolbar", " · "))
            parts.append((style, f"ctx {render.fmt_tokens(used)}/{render.fmt_tokens(limit)} ({percent}%)"))
        ram = self._ram_bytes()
        if ram:
            parts.append(("class:bottom-toolbar", f" · RAM {render.fmt_bytes(ram)}"))
        if self.chat_mode:
            parts.append(("class:bottom-toolbar.mode", " · czat"))
        parts.append(("class:bottom-toolbar", f" · {render.short_path(backend.cwd, limit=40)} "))
        return parts

    def _ram_bytes(self) -> int:
        status = self.status or {}
        budget = status.get("budget") or {}
        if budget.get("usedBytes"):
            return int(budget["usedBytes"])
        return sum(int(i.get("memoryBytes") or 0) for i in status.get("instances") or [])

    async def _refresh_status(self) -> None:
        while True:
            try:
                self.status = await self.backend.runtime_status()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the toolbar is cosmetic
                pass
            await asyncio.sleep(STATUS_REFRESH_S)

    # ----------------------------------------------------------------- lines
    async def _handle_line(self, text: str) -> None:
        if is_command(text):
            await self.dispatch(text)
            return
        if self.chat_mode:
            await self._chat_turn(text)
            return
        await self.turn(text)

    # ------------------------------------------------------------------ turn
    async def turn(self, text: str) -> None:
        """Send one user message to the agent and render the whole turn."""
        images, image_warnings = await asyncio.to_thread(collect_images, text, self.backend.cwd)
        for warning in image_warnings:
            self.console.print(Text("  " + warning, style="lc.warn"))
        if images:
            model = await self.backend.model()
            if not model_sees_images(model):
                name = (model or {}).get("displayName") or self.backend.model_key or "Ten model"
                self.console.print(Text(
                    f"  {name} nie widzi obrazów — wiadomość nie została wysłana. Przełącz się na model "
                    "z wizją (/model; lokalnie np. Qwen3-VL, zdalnie np. Kimi K2.6 albo Qwen3-VL) "
                    "i wyślij ją jeszcze raz.", style="lc.warn"))
                return
            for image in images:
                note = f" ({image.note})" if image.note else ""
                self.console.print(Text(f"  Załączono obraz: {image.name}{note}", style="lc.dim"))
        budget = at_budget_chars(self.backend.context_limit)
        expanded, attachments, warnings = expand_at_references(text, self.backend.cwd, max_chars=budget)
        for warning in warnings:
            self.console.print(Text("  " + warning, style="lc.warn"))
        if attachments:
            self.console.print(Text("  " + summarize_attachments(attachments), style="lc.dim"))
        if images:
            await self.backend.send_user(expanded, [image.payload() for image in images])
        else:
            await self.backend.send_user(expanded)
        await self.pump(stop_on_turn_end=True)

    async def pump(self, task: "asyncio.Task[Any] | None" = None, *, stop_on_turn_end: bool = False,
                   idle: float = 1.0) -> None:
        """Render events until the turn ends (or ``task`` finishes and the queue calms down)."""
        with TurnView(self.console, show_reasoning=self.show_reasoning) as view:
            self._turn_view = view
            with InterruptWatcher(self._interrupt) as watcher:
                self._watcher = watcher
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(self.backend.events.get(),
                                                           timeout=None if stop_on_turn_end else idle)
                        except (asyncio.TimeoutError, TimeoutError):
                            if task is None or task.done():
                                break
                            continue
                        if await self._render_event(event, view):
                            break
                finally:
                    self._watcher = None
                    self._turn_view = None
        if task is not None and not task.done():
            with contextlib.suppress(Exception):
                await task

    def _interrupt(self) -> None:
        """Esc / Ctrl+C during a turn."""
        view = self._turn_view
        if view is not None:
            view.set_status("Przerywam…")
        asyncio.ensure_future(self._interrupt_async())

    async def _interrupt_async(self) -> None:
        with contextlib.suppress(Exception):
            await self.backend.interrupt()

    async def _render_event(self, event: dict[str, Any], view: TurnView) -> bool:
        """Draw one event; ``True`` ends the turn."""
        kind = event.get("type")
        held = self._held_review
        if held is not None and kind != "permission_request":
            # No question followed (nobody to ask): the verdict still shows.
            self._held_review = None
            self._print_review(held, view)
        if kind == "text_delta":
            view.feed_text(str(event.get("text") or ""))
        elif kind == "reasoning_delta":
            view.feed_reasoning(str(event.get("text") or ""))
        elif kind == "tool_call":
            view.reset_step()
            view.set_status(None)
            view.print(render.tool_call_line(str(event.get("title") or ""), str(event.get("name") or "")))
        elif kind == "tool_result":
            self._render_tool_result(event, view)
        elif kind == "subagent_start":
            name = str(event.get("agentId") or "")
            if event.get("background"):
                self.background_helpers.add(name)
            where = " (w tle)" if event.get("background") else ""
            view.print(render.subagent_line(
                f"pomocnik {name}{where}: {event.get('title') or ''}", started=True,
                remote=bool(event.get("remote"))))
        elif kind == "subagent_step":
            # Indented under the delegate call, one line per finished tool call.
            view.print(render.subagent_line(
                f"{event.get('title') or event.get('name') or ''} — {event.get('summary') or ''}",
                ok=bool(event.get("ok"))))
        elif kind == "subagent_end":
            steps = int(event.get("steps") or 0)
            name = str(event.get("agentId") or "")
            self.background_helpers.discard(name)
            where = " w tle" if event.get("background") else ""
            line = f"pomocnik {name} skończył{where} ({steps} " + _plural_steps(steps) + ")"
            report = str(event.get("report") or "").strip()
            if report:
                line += " — " + report.splitlines()[0][:120]
            view.print(render.subagent_line(line, ok=bool(event.get("ok")), finished=True))
        elif kind == "auto_review":
            view.set_status(None)
            opinion = event.get("guard")
            runs = event.get("verdict") == "allow" and (
                not isinstance(opinion, dict) or opinion.get("verdict") == "allow")
            if runs:
                self._print_review(event, view)
            else:
                # Not approved by both gates: the user's question follows and
                # shows the same review right above it — printed once, there.
                # Should no question follow, it is printed with the next event.
                self._held_review = dict(event)
        elif kind == "permission_request":
            self._held_review = None
            await self._ask_permission(event, view)
        elif kind == "context":
            self.backend.context = dict(event)
        elif kind == "compaction":
            view.print(render.compaction_line(int(event.get("before") or 0),
                                              int(event.get("after") or 0),
                                              str(event.get("strategy") or "")))
        elif kind == "goal":
            view.end_assistant()
            self.backend.goal = None if event.get("state") == "cleared" else dict(event)
            view.print(render.goal_line(event))
        elif kind == "status":
            view.set_status(str(event.get("message") or ""), str(event.get("phase") or "info"))
        elif kind == "todos":
            self.todos = list(event.get("items") or [])
            view.print(render.todo_panel(self.todos))
        elif kind == "error":
            view.print(render.error_line(str(event.get("message") or "")))
        elif kind == "turn_end":
            view.end_assistant()
            view.set_status(None)
            self.background_helpers.clear()
            self.last_stats = dict(event.get("stats") or {})
            view.print(render.turn_stats_line(self.last_stats, reason=str(event.get("reason") or "done")))
            return True
        elif kind == "session":
            session = event.get("session") or {}
            if session.get("modelKey"):
                self.backend.model_key = str(session["modelKey"])
            if session.get("mode"):
                self.backend.mode = str(session["mode"])
        return False

    def _render_tool_result(self, event: dict[str, Any], view: TurnView) -> None:
        ok = bool(event.get("ok"))
        view.print(render.tool_result_line(str(event.get("summary") or ""), ok=ok))
        diff = event.get("diff")
        if diff:
            view.print(render.diff_renderable(str(diff)))
            return
        name = str(event.get("name") or "")
        output = str(event.get("output") or "")
        if not output:
            return
        if not ok or name in ("bash", "grep", "glob", "list_dir"):
            tail = render.output_tail(output, lines=TAIL_LINES, width=self.console.width - 8)
            if tail is not None:
                view.print(tail)

    @staticmethod
    def _print_review(event: dict[str, Any], view: TurnView) -> None:
        """The reviewer's line, and the local guard's under it when it checked the call."""
        view.print(render.auto_review_line(str(event.get("verdict") or ""),
                                           str(event.get("reason") or ""),
                                           str(event.get("helper") or "")))
        second = render.guard_line(event.get("guard"))
        if second is not None:
            view.print(second)

    async def _ask_permission(self, event: dict[str, Any], view: TurnView) -> None:
        request_id = str(event.get("requestId") or "")
        choices = [Choice(c.label, c.decision) for c in render.permission_choices(event)]
        watcher = self._watcher
        decision: Any = "deny"
        pause = watcher.paused() if watcher is not None else contextlib.nullcontext()
        with pause, view.paused():
            review = render.permission_review_line(event.get("review"))
            if review is not None:
                self.console.print(review)
            self.console.print(render.permission_panel(event))
            decision = await choose(choices, title="Czy pozwolić na tę operację?",
                                    escape_value="deny",
                                    hint=f"↑/↓ lub 1–{len(choices)} · Enter zatwierdź · Esc = nie")
        answer = str(decision or "deny")
        label = {"allow": "Zezwolono.", "always": "Zezwolono — nie będę więcej pytać.",
                 "deny": "Odmówiono."}.get(answer, "Odmówiono.")
        view.print(Text("  " + label, style="lc.ok" if answer != "deny" else "lc.warn"))
        await self.backend.answer_permission(request_id, answer)

    # ------------------------------------------------------------------ chat
    async def _chat_turn(self, text: str) -> None:
        if self.chat is None:
            self.chat = ChatSession(self.backend, model_key=self.backend.model_key)
        await run_chat_turn(self.chat, self.console, text, show_reasoning=self.show_reasoning)

    # -------------------------------------------------------------- commands
    async def dispatch(self, line: str) -> None:
        parsed = parse_command(line)
        if parsed is None:  # pragma: no cover - guarded by is_command
            return
        if parsed.command is None:
            hint = suggest(parsed.name)
            message = f"Nieznane polecenie /{parsed.name}."
            if hint:
                message += f" Chodziło o /{hint}?"
            self.console.print(Text(message + " Wpisz /help.", style="lc.warn"))
            return
        handler: Callable[[str], Awaitable[None]] | None = getattr(self, f"_cmd_{parsed.name}", None)
        if handler is None:  # pragma: no cover - the test suite asserts this cannot happen
            self.console.print(Text(f"Polecenie /{parsed.name} nie jest jeszcze dostępne.",
                                    style="lc.warn"))
            return
        await handler(parsed.arg)

    async def _cmd_help(self, arg: str) -> None:
        self.console.print(help_table(chat=self.chat_mode))
        self.console.print(keys_table())

    async def _cmd_cwd(self, arg: str) -> None:
        self.console.print(Text(str(self.backend.cwd), style="lc.dim"))

    async def _cmd_forget(self, arg: str) -> None:
        """``/forget`` — drop what was learned from the user's answers here (``agent/precedents.py``).

        The store is the user's (``LOCALCODE_HOME``), shared with the app's
        server on this Mac, so this works the same with either backend.
        """
        from localcode.agent import precedents  # noqa: PLC0415

        removed = await asyncio.to_thread(precedents.clear, self.backend.cwd)
        if removed:
            self.console.print(Text(f"Zapomniano wyuczone decyzje w tym projekcie: {removed}.",
                                    style="lc.ok"))
        else:
            self.console.print(Text("W tym projekcie nie ma wyuczonych decyzji.", style="lc.dim"))

    async def _cmd_exit(self, arg: str) -> None:
        self._running = False

    async def _cmd_model(self, arg: str) -> None:
        key = arg.strip()
        if not key:
            key = await models_ui.pick_model(self.backend, self.console) or ""
        if not key:
            return
        await self.backend.set_model(key)
        model = await self.backend.model()
        self.console.print(Text(f"Model: {render.model_line(model)}", style="lc.ok"))
        for warning in self._model_warnings(model):
            # A 1-bit build, or a 0,6B model in the agent, is a legitimate choice
            # on a small machine — but the user has to hear it before wondering
            # why the answers are nonsense.
            self.console.print(Text(warning, style="lc.warn"))

    def _model_warnings(self, model: dict[str, object] | None) -> list[str]:
        """Warnings worth one yellow line when this model is selected.

        The size warning is for the agent only — a 1,2B model is a perfectly
        reasonable chat partner and a poor code editor.
        """
        out = []
        quant = quant_warning(str((model or {}).get("quant") or "") or None)
        if quant:
            out.append(quant)
        if not self.chat_mode:
            params = (model or {}).get("paramsB")
            size = agent_model_warning(float(params) if isinstance(params, (int, float)) else None)
            if size:
                out.append(size)
        return out

    async def _cmd_models(self, arg: str) -> None:
        models = await self.backend.list_models(refresh=arg.strip() in ("-r", "--refresh"))
        if not models:
            self.console.print(Text("Brak modeli na dysku. Użyj /search albo /download.",
                                    style="lc.warn"))
            return
        status = self.status or await self.backend.runtime_status()
        budget = int((status.get("budget") or {}).get("limitBytes") or 0)
        self.console.print(render.models_table(models, budget_bytes=budget))

    async def _cmd_search(self, arg: str) -> None:
        query = arg.strip()
        if not query:
            self.console.print(Text("Podaj zapytanie, np. /search qwen coder", style="lc.warn"))
            return
        with self.console.status("Szukam na Hugging Face…", spinner="dots"):
            results = await self.backend.hub_search(query, limit=20)
        if not results:
            self.console.print(Text("Nic nie znaleziono.", style="lc.warn"))
            return
        self.console.print(render.hub_table(results))
        self.console.print(Text("Pobierz wybrany model: /download <repozytorium>", style="lc.dim"))

    async def _cmd_download(self, arg: str) -> None:
        parts = arg.split()
        if not parts:
            self.console.print(Text("Użycie: /download <repozytorium> [KWANT]", style="lc.warn"))
            return
        repo = parts[0]
        quant = parts[1] if len(parts) > 1 else None
        key = await models_ui.download_model(self.backend, self.console, repo, quant)
        if key and not self.backend.model_key:
            with contextlib.suppress(BackendError):
                await self.backend.set_model(key)
        elif key:
            self.console.print(Text(f"Aby użyć: /model {key}", style="lc.dim"))

    async def _cmd_load(self, arg: str) -> None:
        key = arg.strip() or self.backend.model_key
        if not key:
            self.console.print(Text("Nie wybrano modelu.", style="lc.warn"))
            return
        with self.console.status(f"Ładuję {key}…", spinner="dots"):
            instance = await self.backend.load_model(key)
        self.console.print(Text(
            f"Załadowano {instance.get('displayName') or key} "
            f"({render.fmt_tokens(instance.get('nCtx'))} kontekstu, "
            f"{render.fmt_bytes(instance.get('memoryBytes'))})", style="lc.ok"))
        self.status = await self.backend.runtime_status()

    async def _cmd_unload(self, arg: str) -> None:
        target = arg.strip()
        if target in ("", "all", "--all"):
            await self.backend.unload(everything=True)
            self.console.print(Text("Wyładowano wszystkie modele.", style="lc.ok"))
        else:
            await self.backend.unload(target)
            self.console.print(Text(f"Wyładowano instancję {target}.", style="lc.ok"))
        self.status = await self.backend.runtime_status()

    async def _cmd_ps(self, arg: str) -> None:
        self.status = await self.backend.runtime_status()
        self.console.print(render.instances_table(self.status))

    async def _cmd_runtime(self, arg: str) -> None:
        parts = arg.split()
        if parts and parts[0] == "set":
            if len(parts) < 3:
                self.console.print(Text("Użycie: /runtime set <klucz> <wartość>", style="lc.warn"))
                return
            try:
                patch = runtime_patch(parts[1], " ".join(parts[2:]))
            except ValueError as exc:
                self.console.print(render.error_line(str(exc)))
                return
            self.status = await self.backend.update_runtime_settings(patch)
            self.console.print(Text("Zapisano ustawienia środowiska.", style="lc.ok"))
        else:
            self.status = await self.backend.runtime_status()
        self.console.print(render.runtime_panel(self.status))

    async def _cmd_obraz(self, arg: str) -> None:
        """``/obraz <opis>`` — generate a picture with a local image model."""
        from localcode.tui import images_ui

        prompt = arg.strip()
        if not prompt:
            self.console.print(Text("Użycie: /obraz opis obrazu", style="lc.warn"))
            return
        client = images_ui.image_client(self.backend)
        try:
            images = await images_ui.generate_with_progress(client, self.console, prompt)
        finally:
            await client.aclose()
        if not images:
            self.console.print(Text("Nie powstał żaden obraz.", style="lc.warn"))
            return
        images_ui.print_results(self.console, images)

    async def _cmd_mode(self, arg: str) -> None:
        mode = arg.strip()
        if not mode:
            chosen = await choose(
                [Choice(render.mode_label(m, long=True), m) for m in MODE_CYCLE],
                title="Tryb uprawnień",
                selected=MODE_CYCLE.index(self.backend.mode) if self.backend.mode in MODE_CYCLE else 0)
            if chosen is None:
                return
            mode = str(chosen)
        if mode not in MODES:
            self.console.print(Text(f"Nieznany tryb „{mode}”. Dostępne: {', '.join(MODES)}.",
                                    style="lc.warn"))
            return
        await self.backend.set_mode(mode)
        self.console.print(Text(f"Tryb: {render.mode_label(mode, long=True)}", style="lc.ok"))

    async def _cmd_think(self, arg: str) -> None:
        """``/think [off|low|medium|high|ultracode]`` — how hard the model should think.

        Without an argument the level is cycled (through the thinking levels —
        ultracode costs helpers, so it is only ever chosen by name). ``on``/``off``
        still work (from when this command only toggled the reasoning display),
        and the reasoning panel follows the level: ``off`` collapses it,
        anything else shows it. Ultracode is the agent's: the chat refuses it.
        """
        value = arg.strip().lower()
        levels = await self._effort_levels()
        if value in ("on", "1", "tak", "true"):
            level = "medium"
        elif value in ("0", "nie", "false"):
            level = "off"
        elif value:
            level = normalize_agent_effort(value)
            if level is None:
                self.console.print(Text(
                    f"Nieznany poziom „{value}”. Dostępne: {', '.join(AGENT_EFFORT_CHOICES)}.",
                    style="lc.warn"))
                return
            if level == ULTRACODE and self.chat_mode:
                self.console.print(Text(
                    "Ultracode to poziom agenta (wielu pomocników naraz) — w czacie wybierz "
                    "off, low, medium albo high.", style="lc.warn"))
                return
        else:
            order = list(EFFORT_CHOICES)
            current = self.backend.effort if self.backend.effort in order else "high"
            level = order[(order.index(current) + 1) % len(order)]
        await self.backend.set_effort(level)
        self.show_reasoning = level != "off"
        self.backend.settings.agent.show_reasoning = self.show_reasoning
        if level == ULTRACODE:
            self.console.print(Text(
                "Myślenie: ultracode — agent rozdziela pracę na wielu pomocników równolegle i sprawdza "
                "wynik. Wolniej i drożej (przy modelach zdalnych każdy pomocnik to płatne zapytania).",
                style="lc.ok"))
            return
        if not levels:
            self.console.print(Text(
                f"Myślenie: {level} (ten model nie ma przełącznika — ustawienie zadziała po zmianie modelu).",
                style="lc.warn"))
            return
        resolved = resolve_effort(level, levels) or level
        suffix = "" if resolved == level else f" (model używa: {resolved})"
        self.console.print(Text(f"Myślenie: {level}{suffix}", style="lc.ok"))

    async def _effort_levels(self) -> list[str]:
        """Effort levels of the current model; never raises (the TUI must not die)."""
        try:
            return await self.backend.effort_levels()
        except Exception:  # noqa: BLE001 - a library hiccup must not break a command
            return []

    async def _cmd_compact(self, arg: str) -> None:
        if self.chat_mode and await chat_command(self.chat, self.console, "compact"):
            return
        task = asyncio.ensure_future(self.backend.compact())
        await self.pump(task, idle=COMPACT_IDLE_S)

    async def _cmd_goal(self, arg: str) -> None:
        """``/goal <cel>``: keep working until the goal is reached; Esc stops."""
        text = arg.strip()
        if not text:
            goal = self.backend.goal
            if not goal:
                self.console.print(Text("Brak celu. Ustaw go: /goal opis tego, co ma być zrobione.",
                                        style="lc.dim"))
            else:
                self.console.print(render.goal_line(goal))
            return
        if text.lower() in GOAL_CLEAR_WORDS:
            await self.backend.clear_goal()
            self.console.print(render.goal_line({"state": "cleared"}))
            return
        budget = at_budget_chars(self.backend.context_limit)
        expanded, attachments, warnings = expand_at_references(text, self.backend.cwd, max_chars=budget)
        for warning in warnings:
            self.console.print(Text("  " + warning, style="lc.warn"))
        if attachments:
            self.console.print(Text("  " + summarize_attachments(attachments), style="lc.dim"))
        await self.backend.start_goal(text, expanded)
        await self.pump(stop_on_turn_end=True)

    async def _cmd_clear(self, arg: str) -> None:
        await self.backend.clear()
        self.todos = []
        self.last_stats = None
        self.chat = None
        self.console.clear()
        await self.show_welcome()

    async def _cmd_context(self, arg: str) -> None:
        if self.chat_mode and await chat_command(self.chat, self.console, "context"):
            return
        self.console.print(render.context_breakdown(self.backend.context))
        if self.todos:
            self.console.print(render.todo_panel(self.todos))

    async def _cmd_stats(self, arg: str) -> None:
        if not self.last_stats:
            self.console.print(Text("Jeszcze nie było żadnej tury.", style="lc.dim"))
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(style="lc.dim", justify="right")
        table.add_column()
        stats = self.last_stats
        table.add_row("kroki", str(stats.get("steps") or 0))
        table.add_row("tokeny promptu", render.fmt_tokens(stats.get("promptTokens")))
        table.add_row("z cache", render.fmt_tokens(stats.get("cachedTokens")))
        table.add_row("tokeny odpowiedzi", render.fmt_tokens(stats.get("completionTokens")))
        table.add_row("prędkość", f"{stats.get('genTps') or 0} tok/s")
        table.add_row("czas", render.fmt_duration(stats.get("elapsedMs")))
        self.console.print(table)

    async def _cmd_resume(self, arg: str) -> None:
        sessions = await self.backend.sessions()
        if not sessions:
            self.console.print(Text("Brak zapisanych sesji w tym katalogu.", style="lc.dim"))
            return
        target = arg.strip()
        if not target:
            choices = [
                Choice(str(s.get("title") or s.get("id")), value=str(s.get("id")),
                       hint=" · ".join(filter(None, [
                           render.short_path(str(s.get("cwd") or ""), limit=40),
                           _stamp(s.get("updatedAt")),
                           f"{render.fmt_tokens(s.get('contextUsed'))} ctx" if s.get("contextUsed") else "",
                       ])))
                for s in sessions[:20]
            ]
            picked = await choose(choices, title="Wznów sesję")
            if picked is None:
                return
            target = str(picked)
        items = await self.backend.resume_session(target)
        self.console.print(Text(f"Wznowiono sesję {target} ({len(items)} wpisów).", style="lc.ok"))
        self._replay(items)

    def _replay(self, items: Sequence[dict[str, Any]], *, limit: int = 12) -> None:
        for item in list(items)[-limit:]:
            kind = str(item.get("kind") or "")
            text = str(item.get("text") or "")
            if kind == "user" and text:
                self.console.print(Text("› " + text.splitlines()[0][:120], style="lc.user"))
            elif kind == "assistant" and text:
                self.console.print(Markdown(text))
            elif kind == "tool":
                self.console.print(render.tool_call_line(str(item.get("title") or ""),
                                                         str(item.get("toolName") or "")))
                review = item.get("review")
                if isinstance(review, dict):
                    self.console.print(render.auto_review_line(str(review.get("verdict") or ""),
                                                               str(review.get("reason") or "")))
                    second = render.guard_line(review.get("guard"))
                    if second is not None:
                        self.console.print(second)
                if item.get("summary"):
                    self.console.print(render.tool_result_line(str(item["summary"]),
                                                               ok=bool(item.get("ok", True))))
            elif kind == "notice" and text:
                self.console.print(render.notice_line(text))

    async def _cmd_init(self, arg: str) -> None:
        self.console.print(Text("Przygotowuję LOCALCODE.md dla tego projektu…", style="lc.dim"))
        await self.turn(INIT_PROMPT)

    async def _cmd_chat(self, arg: str) -> None:
        self.chat_mode = not self.chat_mode
        self.prompt.completer = LocalCodeCompleter(self.backend.cwd, chat_mode=self.chat_mode)
        if self.chat_mode and self.chat is None:
            self.chat = ChatSession(self.backend, model_key=self.backend.model_key)
        where = "czatu (bez narzędzi, rozmowa zapisuje się w aplikacji)" if self.chat_mode \
            else "agenta kodującego"
        self.console.print(Text(f"Tryb {where}.", style="lc.ok"))

    async def _cmd_doctor(self, arg: str) -> None:
        from localcode.doctor import run_doctor

        await run_doctor(self.console)


def _stamp(ms: Any) -> str:
    if not ms:
        return ""
    return time.strftime("%d.%m %H:%M", time.localtime(int(ms) / 1000))


def _history_path() -> Path | None:
    try:
        directory = lc_config.home() / "history"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "tui.txt"
    except OSError:  # pragma: no cover - read-only home
        return None


async def run_tui(backend: Backend, *, chat_mode: bool = False, first: str | None = None,
                  console: Console | None = None) -> int:
    """Entry point used by :mod:`localcode.cli`."""
    app = LocalCodeTui(backend, console, chat_mode=chat_mode)
    return await app.run(first)
