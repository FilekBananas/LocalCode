"""``delegate``: hand one self-contained task to a helper with its own context.

The problem it solves is the window, and since helpers can run commands, the
clock too. "Which modules touch the catalogue?" costs a dozen greps and reads
whose output the agent will never need again; "run these three matches and
compare the logs" costs minutes of waiting per match. A helper does it in a
context nobody else pays for and hands back a short report — and with
``wait=false`` it does it in the background while the agent works on.

What the helper is, deliberately:

* **No file edits.** It gets ``read_file``, ``list_dir``, ``grep``, ``glob``,
  ``bash`` (not in plan mode) and ``skill`` — never ``write_file`` or
  ``edit_file``: two agents editing the same files is a conflict nobody
  resolves.
* **Asking like the agent, under its own name.** Every command goes through
  the parent's policy and approver exactly as the agent's own would — plan
  mode refuses it, yolo runs it, auto mode puts it to the reviewer and the
  local guard, default asks the user — and the question is labelled
  ("Pomocnik h2: …", ``PermissionRequest.helper``), so the user knows who asks.
* **Shallow.** A helper cannot start another helper. Nothing about a 4B model
  that loops on ``todo_write`` suggests it would stop at depth two, and each
  level costs another prompt, another lease and another model in memory.
* **On a model the user chose.** Never one the main model names. The helper
  reads project files; letting the model pick a remote one would post them to
  somebody else's API without the user ever deciding that.

**In the background** (``wait=false``, the default under effort "ultracode"):
the call returns at once ("Pomocnik h2 pracuje w tle"), the helper runs as a
task held by :class:`SubagentHost`, and when it finishes the loop hands its
report to the agent at the next step boundary as a user-role note
(:meth:`SubagentHost.take_finished`, :func:`report_note`). A turn the model
ends while helpers still run waits for them (``Agent._wait_for_helpers``);
an interrupt cancels them all.

The helper's own events do not reach the client. Its ``turn_end`` would end the
parent's turn in both the terminal and the app, and its tool calls would appear
in the parent's transcript as the parent's own. They are translated into the
three ``Subagent*`` events instead — except its permission questions and auto
mode's verdicts on them, which go out as they are, labelled.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from localcode.agent.tools.base import Tool, ToolContext, ToolResult

log = logging.getLogger(__name__)

__all__ = ["STATE_KEY", "BackgroundJob", "SubagentHost", "SubagentTool", "report_note"]

#: Where ``ToolContext.state`` keeps the host (set by the agent loop).
STATE_KEY = "subagents"

#: Tools the helper is allowed. ``glob`` is dropped for a tiny window the same
#: way the main menu drops it: ``grep`` and ``list_dir`` cover the ground.
#: ``bash`` is left out in plan mode, where the agent itself has none.
HELPER_TOOLS = ("read_file", "list_dir", "grep", "glob", "bash")

#: How many helpers one turn may start. ``_REPEAT_LIMIT`` in the loop only stops
#: *identical* calls, and a model that has discovered delegation can otherwise
#: spend a turn delegating.
MAX_PER_TURN = 6
#: …with effort "ultracode", where fanning out *is* the work: a round of
#: investigations (3–8), the reviews before the end, and a second round after
#: a fix still fit; a model delegating in circles does not get further.
ULTRACODE_MAX_PER_TURN = 20

#: Report size, as a fraction of what one tool result may cost, then clamped.
#: A helper that returns four thousand characters has not saved anyone anything.
_REPORT_SHARE = 3
_REPORT_MIN, _REPORT_MAX = 1_200, 4_000

#: Appended to the helper's system prompt. The format is what keeps the report
#: from being a second conversation.
HELPER_ROLE = (
    "You are a helper agent working on ONE task for another agent, not for a person. "
    "You can read files and, when bash is on your menu, run commands (each one may need "
    "the user's approval); you cannot write or edit files. "
    "Finish by answering in at most 150 words: the results (exact numbers, exit codes, "
    "`path:line — fact`). Do not paste code or whole logs; name where they are. "
    "If you could not find out, say so plainly."
)

#: How a background helper's report reaches the agent (a user-role note).
REPORT_NOTE = "[Background helper {label} finished — its report:]\n{report}"
FAILED_NOTE = "[Background helper {label} stopped without a report: {why}]"


@dataclass
class BackgroundJob:
    """One helper started with ``wait=false``."""

    label: str                       # "h2"
    call_id: str                     # the ``delegate`` call that started it
    task: str
    future: "asyncio.Task[ToolResult]"
    reported: bool = False


def report_note(job: BackgroundJob) -> str | None:
    """What the agent reads about a finished background helper; ``None`` when it was cancelled."""
    future = job.future
    if future.cancelled():
        return None
    error = future.exception()
    if error is not None:
        return FAILED_NOTE.format(label=job.label, why=str(error) or type(error).__name__)
    result = future.result()
    if result.ok:
        return REPORT_NOTE.format(label=job.label, report=result.output.strip())
    return FAILED_NOTE.format(label=job.label, why=result.output.strip())


def _report_cap(ctx: ToolContext) -> int:
    limit = int(getattr(ctx.limits, "max_output_chars", 0) or 0) or _REPORT_MAX
    return max(_REPORT_MIN, min(_REPORT_MAX, limit // _REPORT_SHARE))


class SubagentHost:
    """What ``delegate`` needs from the session that owns it.

    Built by the agent loop (which has the runtime, the settings and the event
    stream) and handed to the tool through ``ToolContext.state`` — the same way
    the skill library and the model-management facade travel.
    """

    def __init__(self, *, runtime: Any, settings: Any, cwd: Any, model_key: str,
                 emit: Any, agent_factory: Any = None) -> None:
        self.runtime = runtime
        self.settings = settings
        self.cwd = cwd
        self.model_key = model_key
        self.emit = emit
        #: Injected by the tests; the real one is imported lazily to keep this
        #: module importable without pulling the whole loop in.
        self._agent_factory = agent_factory
        #: One helper at a time. The loop already runs tool calls one after
        #: another, but a second one would quietly double the memory budget.
        #: Ultracode widens it (:meth:`set_ultracode`): there the user asked
        #: for helpers side by side.
        self._slot = asyncio.Semaphore(1)
        self._parallel = 1
        self._running = 0
        self._started = 0
        #: Helpers started in this session — their short names, "h1", "h2" …
        self._named = 0
        #: Helpers working (or done and not yet reported) in the background.
        self._jobs: list[BackgroundJob] = []
        #: Effort "ultracode" on the parent: helpers get twice the steps, think
        #: at "high" and go to the background unless told to wait; set by the
        #: loop before every step.
        self.ultracode = False
        #: The parent's approvals, which the helper's own calls go through
        #: (:meth:`bind`): its approver, its policy object and its mode.
        self.ask_permission: Any = None
        self.policy: Any = None
        self.mode: str | None = None
        self.allow_always = True
        #: The parent's provider-side cache key; a helper's is this plus its name.
        self.cache_key: str | None = None

    # -- what the helper runs on ------------------------------------------- #

    @property
    def helper_model_key(self) -> str:
        chosen = getattr(self.settings.agent, "subagent_model", None)
        return str(chosen or self.model_key)

    def helper_settings(self) -> Any:
        """The parent's settings, with the helper's own limits.

        A helper is never ultracode itself, whatever the settings say: it
        thinks at the level that implies and cannot fan out (no ``delegate``).
        """
        from localcode.runtime.templates import engine_effort  # noqa: PLC0415 - light, but keep lazy

        steps = int(getattr(self.settings.agent, "subagent_max_steps", 12))
        effort = getattr(self.settings.agent, "reasoning_effort", "medium")
        agent = replace(
            self.settings.agent,
            max_steps=steps * 2 if self.ultracode else steps,
            reasoning_effort="high" if self.ultracode else (engine_effort(effort) or "medium"),
            manage_models=False,       # a helper does not download or load models
            subagents=False,           # and does not start another helper
            # The parent's mode, not the settings' default: the session may
            # have been switched (and the policy object is shared).
            permission_mode=self.mode or getattr(self.settings.agent, "permission_mode", "default"),
        )
        return replace(self.settings, agent=agent)

    def bind(self, *, ask_permission: Any, policy: Any, mode: str, allow_always: bool = True,
             cache_key: str | None = None) -> None:
        """The parent's approvals and cache key; set by the loop before every step."""
        self.ask_permission = ask_permission
        self.policy = policy
        self.mode = mode
        self.allow_always = allow_always
        self.cache_key = cache_key

    def set_ultracode(self, on: bool, parallel: int = 1) -> None:
        """Effort "ultracode" on or off, with how many helpers may run at once.

        Called by the loop between steps. Not resized while a helper runs or a
        background one waits for a place: a gate swapped under it would lose
        its release.
        """
        self.ultracode = bool(on)
        wanted = max(1, int(parallel)) if on else 1
        if wanted != self._parallel and self._running == 0 and not self.running:
            self._slot = asyncio.Semaphore(wanted)
            self._parallel = wanted

    def budget_refusal(self) -> str | None:
        """Why this helper cannot start, when its model would not fit.

        Only asked for a *different* local model: the parent's own is already
        loaded, and a remote one takes no memory at all.
        """
        key = self.helper_model_key
        if key == self.model_key or str(key).startswith("remote:"):
            return None
        estimate = getattr(self.runtime, "estimate", None)
        if not callable(estimate):
            return None
        try:
            fits = getattr(estimate(key), "fits", None)
        except Exception:                                  # noqa: BLE001 - never fatal
            return None
        if str(fits) == "no":
            return (f"Helper model {key} does not fit in memory next to the current one. "
                    "Do the search yourself, or ask the user to pick a smaller helper model.")
        return None

    @property
    def max_per_turn(self) -> int:
        return ULTRACODE_MAX_PER_TURN if self.ultracode else MAX_PER_TURN

    def claim(self) -> bool:
        """Count one more helper for this turn; ``False`` when the turn is spent.

        Synchronous on purpose: with helpers running side by side, a check and
        a count separated by an ``await`` would let a whole step through.
        """
        if self.exhausted:
            return False
        self._started += 1
        return True

    async def start(self) -> None:
        await self._slot.acquire()
        self._running += 1

    def finish(self) -> None:
        self._running = max(0, self._running - 1)
        self._slot.release()

    @property
    def exhausted(self) -> bool:
        return self._started >= self.max_per_turn

    def reset_turn(self) -> None:
        self._started = 0

    def next_label(self) -> str:
        self._named += 1
        return f"h{self._named}"

    # -- background helpers ------------------------------------------------ #

    def spawn(self, label: str, call_id: str, task: str, work: Any) -> BackgroundJob:
        """Run ``work`` (the helper's coroutine) as a background job."""
        job = BackgroundJob(label=label, call_id=call_id, task=task,
                            future=asyncio.ensure_future(work))
        self._jobs.append(job)
        return job

    @property
    def running(self) -> list[BackgroundJob]:
        return [j for j in self._jobs if not j.future.done()]

    @property
    def has_background(self) -> bool:
        """A background helper is working, or done and its report not yet handed over."""
        return any(not j.reported for j in self._jobs)

    def take_finished(self) -> list[BackgroundJob]:
        """The background helpers that finished since the last call, in start order."""
        done = [j for j in self._jobs if j.future.done() and not j.reported]
        for job in done:
            job.reported = True
        self._jobs = [j for j in self._jobs if not j.reported]
        return done

    async def wait_any(self) -> None:
        """Until one background helper finishes (at once when one already has).

        Cancelling this wait cancels no helper: they belong to the turn, which
        stops them itself (:meth:`stop_background`).
        """
        if any(j.future.done() and not j.reported for j in self._jobs):
            return
        running = [j.future for j in self.running]
        if running:
            await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)

    def cancel_background(self) -> int:
        """Cancel every background helper still working; how many there were."""
        running = self.running
        for job in running:
            job.future.cancel()
        return len(running)

    async def stop_background(self) -> list[BackgroundJob]:
        """Cancel the background helpers, wait until they are gone, forget them all.

        Returns the jobs whose reports never reached the agent (the turn
        ended first) — the caller says so rather than dropping them silently.
        """
        self.cancel_background()
        futures = [j.future for j in self._jobs]
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)
        lost = [j for j in self._jobs if not j.reported]
        self._jobs = []
        return lost

    def build_agent(self, *, on_event: Any, label: str = "") -> Any:
        if self._agent_factory is not None:
            return self._agent_factory(self, on_event)
        from localcode.agent.loop import Agent               # noqa: PLC0415 - cycle

        return Agent(
            self.runtime, self.helper_model_key, self.cwd, self.helper_settings(),
            on_event,
            # The parent's approver and policy: a helper's command is asked
            # about exactly like the agent's own, labelled with its name.
            self.ask_permission,
            session=None,               # nothing of a helper is written to disk
            tools=_helper_tools,
            policy=self.policy,
            system_suffix=HELPER_ROLE,
            allow_always=self.allow_always,
            helper=label or None,
            cache_key=f"{self.cache_key}-{label}" if self.cache_key and label else None,
        )


def _helper_tools(mode: str, tiny: bool, *, skills: bool = False,
                  **_ignored: Any) -> list[Tool]:
    """The helper's menu: reading, commands (not in plan mode), the skill shelf.

    Built by name rather than by filtering the full menu, so a tool added later
    does not quietly become something a helper may call.
    """
    from localcode.agent.tools import get_tool                # noqa: PLC0415 - cycle

    wanted = [n for n in HELPER_TOOLS
              if not (tiny and n == "glob") and not (mode == "plan" and n == "bash")]
    tools = [t for t in (get_tool(n, tiny=tiny) for n in wanted) if t is not None]
    if skills:
        # The prompt tells the helper it has a skill shelf, so the tool has to
        # be there — otherwise it reaches for something that does not exist.
        skill_tool = get_tool("skill", tiny=tiny)
        if skill_tool is not None:
            tools.append(skill_tool)
    return tools


def _wants_wait(args: dict[str, Any], host: SubagentHost) -> bool:
    """``wait`` as asked; by default background under ultracode, waiting otherwise."""
    value = args.get("wait")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
        return value.strip().lower() in ("true", "yes", "1")
    return not host.ultracode


class SubagentTool(Tool):
    name = "delegate"
    kind = "meta"
    description = ("Start a helper agent on one self-contained task. It can read files and run "
                   "commands (not edit files) and returns a short report. wait=false: it works "
                   "in the background, the report arrives later.")
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What to do or find out, with all context needed — "
                               "the helper cannot see this conversation",
            },
            "wait": {
                "type": "boolean",
                "description": "false = work in the background; you keep working and get the "
                               "report as a note when it is done",
            },
        },
        "required": ["task"],
    }

    def title(self, args: dict[str, Any]) -> str:
        task = str(args.get("task") or "").strip().splitlines()[0] if args.get("task") else ""
        return f"Pomocnik: {task[:60]}" if task else "Pomocnik"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        host: SubagentHost | None = ctx.state.get(STATE_KEY)
        if host is None:
            return ToolResult(False, "Helper agents are not available in this session.",
                              "sub-agenci wyłączeni")
        task = str(args.get("task") or "").strip()
        if not task:
            return ToolResult(False, "Say what the helper should find out.", "brak zadania")
        refusal = host.budget_refusal()
        if refusal:
            return ToolResult(False, refusal, "pomocnik się nie mieści w pamięci")
        if not host.claim():
            return ToolResult(False,
                              f"No more helpers this turn (limit {host.max_per_turn}). "
                              "Do the rest yourself.", "limit pomocników")

        # Several helpers of one step run at once under ultracode, so the call
        # comes with the context the loop made for it; the shared state only
        # names the one call running.
        call_id = str(ctx.call_id or ctx.state.get("current_call_id") or "")
        label = host.next_label()
        cap = _report_cap(ctx)
        if _wants_wait(args, host):
            return await _run_helper(host, label, call_id, task, cap, progress=ctx.progress)
        host.spawn(label, call_id, task,
                   _run_helper(host, label, call_id, task, cap, background=True))
        short = task.splitlines()[0][:60]
        return ToolResult(True,
                          f"Helper {label} is working in the background on: {short}. Keep working; "
                          "its report will arrive as a note when it finishes.",
                          f"Pomocnik {label} pracuje w tle: {short}")


async def _run_helper(host: SubagentHost, label: str, call_id: str, task: str, cap: int, *,
                      background: bool = False, progress: Any = None) -> ToolResult:
    """Run one helper to its report — awaited by the call, or as a background job."""
    steps: list[str] = []
    started = time.perf_counter()

    await host.start()
    helper = None
    try:
        helper = host.build_agent(on_event=_translator(host, label, call_id, steps), label=label)
        await host.emit(_start_event(host, label, call_id, task, background))
        if progress is not None:
            # One plain status for clients that do not know the new events.
            await progress(f"Pomocnik: {task[:60]}")
        answer = await helper.run_turn(task)
    except asyncio.CancelledError:
        if helper is not None:
            helper.interrupt()
        await host.emit(_end_event(label, call_id, False, len(steps), started, "przerwane",
                                   background=background))
        raise                        # the parent's turn is ending; do not swallow it
    except Exception as exc:         # noqa: BLE001 - a helper must not kill the turn
        log.warning("helper failed: %s", exc)
        await host.emit(_end_event(label, call_id, False, len(steps), started, str(exc),
                                   background=background))
        return ToolResult(False, f"The helper failed: {exc}", "pomocnik zawiódł",
                          display="\n".join(steps) or None)
    finally:
        host.finish()

    report = (answer or "").strip()
    if not report:
        await host.emit(_end_event(label, call_id, True, len(steps), started,
                                   background=background))
        return ToolResult(False, "The helper finished without an answer.",
                          "pomocnik nic nie odpowiedział",
                          display="\n".join(steps) or None)
    if len(report) > cap:
        report = report[:cap].rstrip() + "\n[…]"
    elapsed = time.perf_counter() - started
    await host.emit(_end_event(label, call_id, True, len(steps), started, background=background,
                               report=report if background else ""))
    footer = f"\n\n[pomocnik {label}: {len(steps)} krok(ów), {elapsed:.0f} s]"
    return ToolResult(True, report + footer,
                      f"pomocnik: {len(steps)} krok(ów)",
                      display="\n".join(steps) or None)


# --------------------------------------------------------------------------- #
# Translating the helper's events
# --------------------------------------------------------------------------- #


def _translator(host: SubagentHost, agent_id: str, call_id: str, steps: list[str]) -> Any:
    """Turn the helper's own events into the three wrapped ones.

    Its permission questions and auto mode's verdicts on them go out as they
    are — already labelled by the helper (``Agent(helper=…)``): the user has
    to answer the one and should see the other. Everything else is dropped on
    purpose — see this module's docstring.
    """
    titles: dict[str, str] = {}

    async def on_event(event: Any) -> None:
        kind = getattr(event, "type", "")
        if kind in ("permission_request", "auto_review"):
            await host.emit(event)
        elif kind == "tool_call":
            titles[event.call_id] = event.title
        elif kind == "tool_result":
            title = titles.pop(event.call_id, event.name)
            steps.append(f"{'✓' if event.ok else '✗'} {title} — {event.summary}")
            from localcode.events import SubagentStep       # noqa: PLC0415 - cycle

            await host.emit(SubagentStep(agent_id=agent_id, parent_call_id=call_id,
                                         name=event.name, title=title, ok=bool(event.ok),
                                         summary=str(event.summary)))

    return on_event


def _start_event(host: SubagentHost, agent_id: str, call_id: str, task: str,
                 background: bool = False) -> Any:
    from localcode.events import SubagentStart              # noqa: PLC0415 - cycle

    key = host.helper_model_key
    return SubagentStart(agent_id=agent_id, parent_call_id=call_id,
                         title=task.splitlines()[0][:80], model=key,
                         remote=str(key).startswith("remote:"), background=background)


def _end_event(agent_id: str, call_id: str, ok: bool, steps: int, started: float,
               message: str = "", *, background: bool = False, report: str = "") -> Any:
    from localcode.events import SubagentEnd                # noqa: PLC0415 - cycle

    return SubagentEnd(agent_id=agent_id, parent_call_id=call_id, ok=ok, steps=steps,
                       elapsed_ms=int((time.perf_counter() - started) * 1000),
                       message=message, background=background, report=report)
