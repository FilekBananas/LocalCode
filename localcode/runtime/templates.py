"""Local rendering and capability detection for Hugging Face / GGUF chat templates.

Why this exists
---------------
Engines render prompts with the model's own chat template (llama-server's
``/apply-template``, ``tokenizer.apply_chat_template`` in the MLX worker).  We
still need a faithful *local* renderer for three reasons:

* fallback when an engine cannot render a template (older llama.cpp builds
  choke on some Jinja constructs; GGUF files without a server-side template),
* offline capability detection (``inspect_template``) without a running engine,
* tests that must not load a model.

Rendering mirrors ``transformers.utils.chat_template_utils`` exactly:
``ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`` with the
``loopcontrols`` extension, a no-op ``{% generation %}`` block, the
``raise_exception`` and ``strftime_now`` globals and a ``tojson`` filter that
does not HTML-escape and accepts ``ensure_ascii``/``indent``/``separators``/
``sort_keys``.  Special-token variables (``bos_token``, ``eos_token``) and any
extra keyword (``enable_thinking``, ``reasoning_effort`` …) are passed through
as template variables, exactly like ``apply_chat_template(**kwargs)``.

Capability detection
--------------------
``inspect_template`` combines three signals, because template source code is
too varied for any single one to be reliable:

1. the Jinja AST (``jinja2.meta.find_undeclared_variables``) tells which
   top-level variables the template reads (``tools``, ``enable_thinking`` …) —
   this ignores look-alikes such as Phi-4's ``message['tools']``;
2. probe renders with sentinel strings prove that a feature actually reaches
   the prompt (a system prompt that a template silently drops is not
   "supported"; a template that raises on ``role="tool"`` does not support it);
3. marker strings (``<think>``, ``<|im_end|>`` …) give the reasoning style and
   the stop strings.

Results are cached per template source; callers get fresh copies.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from typing import Any, Sequence

import jinja2
from jinja2 import meta as jinja_meta
from jinja2 import nodes as jinja_nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from localcode.runtime.base import EngineError, TemplateInfo

__all__ = [
    "TemplateRenderError",
    "THINKING_KWARGS",
    "EFFORT_KWARGS",
    "EFFORT_BUDGETS",
    "EFFORT_ORDER",
    "EFFORT_CHOICES",
    "AGENT_EFFORT_CHOICES",
    "ULTRACODE",
    "engine_effort",
    "normalize_agent_effort",
    "normalize_effort",
    "render_template",
    "inspect_template",
    "renders_tool_calls",
    "template_variables",
    "stop_strings_for_template",
    "thinking_kwargs",
    "effort_kwargs",
    "resolve_effort",
    "model_effort_levels",
]


class TemplateRenderError(EngineError):
    """The chat template could not be compiled or rendered.

    ``str(error)`` is a Polish, user-presentable message; ``detail`` keeps the
    original (English) Jinja/Python error for logs and for callers that retry
    with a different message layout.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


# --------------------------------------------------------------------------- #
# Jinja environment (transformers-compatible)
# --------------------------------------------------------------------------- #


class _GenerationBlock(Extension):
    """``{% generation %}…{% endgeneration %}`` — used by some templates to mark
    assistant tokens for training masks. Rendering-wise it is a no-op."""

    tags = {"generation"}

    def parse(self, parser: Any) -> jinja_nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja_nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

    def _render(self, caller: Any) -> str:
        return caller()


def _raise_exception(message: Any) -> None:
    raise jinja2.exceptions.TemplateError(str(message))


def _tojson(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: Any = None,
    sort_keys: bool = False,
) -> str:
    if separators is not None:
        separators = tuple(separators)
    return json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def _strftime_now(fmt: str) -> str:
    return datetime.now().strftime(fmt)


def _make_env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationBlock, loopcontrols],
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env


_ENV = _make_env()


@lru_cache(maxsize=32)
def _compile(template: str) -> jinja2.Template:
    return _ENV.from_string(template)


@lru_cache(maxsize=32)
def _undeclared(template: str) -> frozenset[str]:
    return frozenset(jinja_meta.find_undeclared_variables(_ENV.parse(template)))


def template_variables(template: str) -> frozenset[str]:
    """Top-level variables a template reads (``messages``, ``tools``,
    ``enable_thinking`` …). Empty set when the template does not even parse."""
    try:
        return _undeclared(template)
    except jinja2.TemplateError:
        return frozenset()


def _error_detail(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    if not isinstance(exc, jinja2.TemplateError):
        text = f"{type(exc).__name__}: {text}"
    return text[:500]


def render_template(
    template: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = True,
    bos_token: str = "",
    eos_token: str = "",
    **kwargs: Any,
) -> str:
    """Render ``messages`` (+ optional OpenAI tool schemas) with a chat template.

    Behaves like ``tokenizer.apply_chat_template(..., tokenize=False)``: extra
    keyword arguments become template variables. Raises ``TemplateRenderError``
    (an ``EngineError``) with a Polish message on any compile/render failure,
    including the template's own ``raise_exception`` calls.
    """
    if not template or not template.strip():
        raise TemplateRenderError("Model nie ma szablonu czatu.", "empty template")
    try:
        compiled = _compile(template)
    except jinja2.TemplateSyntaxError as exc:
        detail = f"line {exc.lineno}: {exc.message}"
        raise TemplateRenderError(f"Szablon czatu modelu jest niepoprawny ({detail}).", detail) from exc
    except jinja2.TemplateError as exc:
        detail = _error_detail(exc)
        raise TemplateRenderError(f"Szablon czatu modelu jest niepoprawny ({detail}).", detail) from exc

    variables: dict[str, Any] = {
        "bos_token": bos_token or "",
        "eos_token": eos_token or "",
        "documents": None,
    }
    variables.update(kwargs)
    variables["messages"] = messages
    variables["tools"] = tools if tools else None
    variables["add_generation_prompt"] = add_generation_prompt
    try:
        return compiled.render(**variables)
    except jinja2.TemplateError as exc:
        detail = _error_detail(exc)
        raise TemplateRenderError(f"Szablon czatu modelu zgłosił błąd: {detail}", detail) from exc
    except RecursionError as exc:  # pragma: no cover - pathological templates
        raise TemplateRenderError("Szablon czatu modelu jest zbyt złożony (rekurencja).", "recursion") from exc
    except Exception as exc:  # TypeError on str + None, KeyError, ...
        detail = _error_detail(exc)
        raise TemplateRenderError(f"Nie udało się zastosować szablonu czatu modelu: {detail}", detail) from exc


# --------------------------------------------------------------------------- #
# Capability detection
# --------------------------------------------------------------------------- #

# Template variables that switch reasoning on/off, in order of preference, with
# the (on, off) values used for probing. ``None`` for "on" means "leave the
# template default" (formatter omits the kwarg).
THINKING_KWARGS: dict[str, tuple[Any, Any]] = {
    "enable_thinking": (True, False),     # Qwen3, Qwen3.5, GLM 4.5+, Gemma 4, SmolLM3
    "thinking": (True, False),            # DeepSeek V3.1, Granite 3.x
    "reasoning_effort": ("high", "low"),  # gpt-oss (cannot be disabled, only lowered)
    "thinking_budget": (-1, 0),           # Seed-OSS
    "enable_reasoning": (True, False),
    "thinking_mode": (True, False),
    "reasoning": (True, False),
}


def thinking_kwargs(kwarg: str | None, thinking: bool) -> dict[str, Any]:
    """Template kwargs that turn reasoning on/off for a detected switch."""
    if not kwarg:
        return {}
    if kwarg == "reasoning_effort":
        return {} if thinking else {"reasoning_effort": "low"}
    if kwarg == "thinking_budget":
        return {} if thinking else {"thinking_budget": 0}
    on, off = THINKING_KWARGS.get(kwarg, (True, False))
    return {kwarg: on if thinking else off}


# --------------------------------------------------------------------------- #
# Reasoning effort
# --------------------------------------------------------------------------- #

# Canonical effort levels, weakest -> strongest. "on" is the single enabled
# level of a template that only has a boolean thinking switch (Qwen3, GLM,
# DeepSeek V3.1 ...); such a model cannot be asked for *less* reasoning, only
# for none at all.
EFFORT_ORDER: dict[str, int] = {"off": 0, "low": 1, "medium": 2, "high": 3, "on": 3}

# What a user may ask for (settings, CLI, API). "on" exists only as a resolved
# level of a boolean template, never as a request.
EFFORT_CHOICES: tuple[str, ...] = ("off", "low", "medium", "high")


#: The coding agent's level above "high" (``agent/loop.py``): the model thinks
#: at "high" and the agent works as an orchestrator, fanning independent work
#: out to parallel helpers and having the result reviewed. It is a way of
#: *working*, not a template level, so it never reaches a template or an
#: engine — :func:`engine_effort` maps it down at that boundary.
ULTRACODE = "ultracode"
_ULTRACODE_NAMES = frozenset({ULTRACODE, "ultra"})

#: What the coding agent may be asked for: the thinking levels plus ultracode.
#: The chatbot and the API keep :data:`EFFORT_CHOICES`.
AGENT_EFFORT_CHOICES: tuple[str, ...] = (*EFFORT_CHOICES, ULTRACODE)


def normalize_effort(level: Any, default: str | None = None) -> str | None:
    """A user-supplied effort as one of ``EFFORT_CHOICES``, else ``default``.

    ``"ultracode"`` reads as ``"high"`` here: a surface that only knows thinking
    levels (the chatbot, ``/v1``) gets the thinking it implies, never the name.
    """
    if isinstance(level, str):
        text = level.strip().lower()
        if text in EFFORT_CHOICES:
            return text
        if text == "on":
            return "medium"
        if text in _ULTRACODE_NAMES:
            return "high"
    return default


def normalize_agent_effort(level: Any, default: str | None = None) -> str | None:
    """Like :func:`normalize_effort`, but keeps ``"ultracode"`` (the agent's own level)."""
    if isinstance(level, str) and level.strip().lower() in _ULTRACODE_NAMES:
        return ULTRACODE
    return normalize_effort(level, default)


def engine_effort(level: str | None) -> str | None:
    """``level`` as a template may see it: ultracode thinks at ``"high"``."""
    if isinstance(level, str) and level.strip().lower() in _ULTRACODE_NAMES:
        return "high"
    return level

_BOOL_LEVELS: tuple[str, ...] = ("off", "on")
_EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")
_BUDGET_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")

# Token budgets for templates whose switch is a number of thinking tokens
# (Seed-OSS ``thinking_budget``, Gemini-style ``reasoning_budget``).
EFFORT_BUDGETS: dict[str, int] = {"off": 0, "low": 512, "medium": 2048, "high": 8192}

# Template variables that select the reasoning effort, in order of preference,
# with the levels each one can express. Probed in this order, so a template that
# has both ``reasoning_effort`` and a boolean switch offers the richer control.
EFFORT_KWARGS: dict[str, tuple[str, ...]] = {
    "reasoning_effort": _EFFORT_LEVELS,   # gpt-oss: low/medium/high, never off
    "thinking_budget": _BUDGET_LEVELS,    # Seed-OSS
    "reasoning_budget": _BUDGET_LEVELS,
    "enable_thinking": _BOOL_LEVELS,      # Qwen3, GLM 4.5+, Gemma 4, SmolLM3
    "thinking": _BOOL_LEVELS,             # DeepSeek V3.1, Granite 3.x
    "enable_reasoning": _BOOL_LEVELS,
    "thinking_mode": _BOOL_LEVELS,
    "reasoning": _BOOL_LEVELS,
}


def effort_kwargs(kwarg: str | None, level: str | None) -> dict[str, Any]:
    """Template kwargs that request ``level`` through ``kwarg``.

    ``level`` must already be one the template supports (see
    :func:`resolve_effort`); unknown levels fall back to the template default
    (no kwarg at all).
    """
    if not kwarg or not level:
        return {}
    level = engine_effort(level)
    levels = EFFORT_KWARGS.get(kwarg)
    if levels is None:
        return {}
    if levels is _BOOL_LEVELS:
        return {kwarg: level != "off"}
    if levels is _BUDGET_LEVELS:
        budget = EFFORT_BUDGETS.get(level)
        return {kwarg: budget} if budget is not None else {}
    if level not in levels:
        return {}
    return {kwarg: level}


def resolve_effort(level: str | None, levels: Sequence[str] | None) -> str | None:
    """Map a requested effort onto what a given template can actually express.

    ``"medium"`` on a Qwen3 (boolean) template becomes ``"on"``; ``"off"`` on
    gpt-oss (which cannot stop reasoning) becomes its weakest level ``"low"``.
    Returns ``None`` when the model has no control at all, or when ``level`` is
    ``None`` ("leave the template default").
    """
    if not levels:
        return None
    supported = list(levels)
    if not level:
        return None
    level = engine_effort(level.strip().lower()) or ""
    if level in supported:
        return level
    want = EFFORT_ORDER.get(level)
    if want is None:
        return None
    if level == "off":
        return supported[0]
    candidates = [lv for lv in supported if lv != "off"] or supported
    return min(candidates, key=lambda lv: (abs(EFFORT_ORDER[lv] - want), EFFORT_ORDER[lv]))


# Reasoning markers that prove a template knows about thinking blocks.
_REASONING_MARKERS = (
    "<think>", "</think>", "[THINK]", "<seed:think>", "<|channel>thought", "<|begin_of_thought|>",
    "<|START_THINKING|>", "<longcat_think>", "◁think▷", "reasoning_content",
)

# (marker present in the template source, stop strings it implies).
_STOP_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("<|im_end|>", ("<|im_end|>",)),
    ("<|eot_id|>", ("<|eot_id|>", "<|eom_id|>")),
    ("<end_of_turn>", ("<end_of_turn>",)),
    ("<turn|>", ("<turn|>",)),
    ("<|return|>", ("<|return|>", "<|call|>")),
    ("<|observation|>", ("<|user|>", "<|observation|>")),
    ("<｜end▁of▁sentence｜>", ("<｜end▁of▁sentence｜>",)),
    ("[e~[", ("[e~[",)),
    ("<|end_of_text|>", ("<|end_of_text|>",)),
    ("<seed:eos>", ("<seed:eos>",)),
    ("<|END_OF_TURN_TOKEN|>", ("<|END_OF_TURN_TOKEN|>",)),
    ("<end_function_call>", ("<start_function_response>",)),
    ("<|tool_response>", ("<|tool_response>",)),
)


def stop_strings_for_template(template: str) -> list[str]:
    """Extra stop strings implied by the template's turn delimiters."""
    out: list[str] = []
    for marker, stops in _STOP_RULES:
        if marker in template:
            for s in stops:
                if s not in out:
                    out.append(s)
    # Phi-3/4 style "<|end|>" ends a turn, but in gpt-oss harmony it only ends a
    # message inside the same turn, so it must never be a stop there.
    if "<|end|>" in template and "<|channel|>" not in template and "<|end|>" not in out:
        out.append("<|end|>")
    # Gemma 4 emits <|tool_response> right after a call; it is only a stop there.
    if "<|tool_response>" in out and "<|tool_call>" not in template:
        out.remove("<|tool_response>")
    return out


_PROBE_SYSTEM = "LCPROBE_SYSTEM_7f3a"
_PROBE_USER = "LCPROBE_USER_7f3a"
_PROBE_TOOL_NAME = "lcprobe_tool_7f3a"
_PROBE_TOOL_RESULT = "LCPROBE_RESULT_7f3a"
_PROBE_ARGUMENT = "LCPROBEARG7f3a"  # survives json.dumps unchanged
_PROBE_CALL_ID = "lcprobe01"  # 9 alphanumerics: Mistral templates demand >= 9

_PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _PROBE_TOOL_NAME,
            "description": "Probe tool.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "A path."}},
                "required": ["path"],
            },
        },
    }
]

_TOOL_ROLE_MENTION = re.compile(r"""['"](?:tool|ipython|tool_results|observation)['"]""")


def _try_render(template: str, messages: list[dict[str, Any]], **kwargs: Any) -> str | None:
    try:
        return render_template(template, messages, **kwargs)
    except TemplateRenderError:
        return None


def _probe_call(arguments: Any) -> dict[str, Any]:
    return {"id": _PROBE_CALL_ID, "type": "function", "function": {"name": _PROBE_TOOL_NAME, "arguments": arguments}}


def _probe_tool_role(template: str, with_tools: bool) -> bool:
    tools = _PROBE_TOOLS if with_tools else None
    for arguments in ({"path": "a.txt"}, json.dumps({"path": "a.txt"})):
        messages = [
            {"role": "user", "content": _PROBE_USER},
            {"role": "assistant", "content": "", "tool_calls": [_probe_call(arguments)]},
            {"role": "tool", "tool_call_id": _PROBE_CALL_ID, "name": _PROBE_TOOL_NAME, "content": _PROBE_TOOL_RESULT},
        ]
        out = _try_render(template, messages, tools=tools)
        if out is not None:
            return _PROBE_TOOL_RESULT in out
    return False


@lru_cache(maxsize=64)
def _probe_renders_tool_calls(template: str) -> bool:
    variables = template_variables(template)
    tool_sets = (_PROBE_TOOLS, None) if "tools" in variables else (None,)
    for tools in tool_sets:
        for arguments in ({"path": _PROBE_ARGUMENT}, json.dumps({"path": _PROBE_ARGUMENT})):
            messages = [
                {"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": "", "tool_calls": [_probe_call(arguments)]},
            ]
            out = _try_render(template, messages, tools=tools)
            if out is not None and _PROBE_ARGUMENT in out:
                return True
    return False


def renders_tool_calls(template: str | None) -> bool:
    """True when the template writes an assistant *tool call* back into the prompt.

    Some templates render the tool schemas but only ever print
    ``message['content']`` (Granite 3.x, Phi-4, many small fine-tunes), so
    ``tool_calls`` silently disappear from the history and the model sees a tool
    result for a call it never made. ``ChatFormatter`` uses this to decide
    whether it has to serialise past calls into the message text itself.
    """
    if not template or not template.strip():
        return False
    return _probe_renders_tool_calls(template)


def _probe_thinking_kwarg(template: str, variables: frozenset[str]) -> str | None:
    messages = [{"role": "user", "content": _PROBE_USER}]
    for kwarg, (on, off) in THINKING_KWARGS.items():
        if kwarg not in variables:
            continue
        a = _try_render(template, messages, **{kwarg: on})
        b = _try_render(template, messages, **{kwarg: off})
        if a is not None and b is not None and a != b:
            return kwarg
    return None


def _probe_effort(template: str, variables: frozenset[str]) -> tuple[str | None, list[str]]:
    """Find the template variable that selects reasoning effort, and the levels
    it can really express (levels that render identically are collapsed)."""
    messages = [{"role": "user", "content": _PROBE_USER}]
    for kwarg, levels in EFFORT_KWARGS.items():
        if kwarg not in variables:
            continue
        kept: list[str] = []
        seen: list[str] = []
        failed = False
        for level in levels:
            out = _try_render(template, messages, **effort_kwargs(kwarg, level))
            if out is None:
                failed = True
                break
            if out in seen:
                continue
            seen.append(out)
            kept.append(level)
        if failed or len(kept) < 2:
            continue
        return kwarg, kept
    return None, []


@lru_cache(maxsize=64)
def _inspect_cached(template: str) -> TemplateInfo:
    from localcode.runtime.toolcalls import REASONING_OPENERS, detect_tool_format

    variables = template_variables(template)
    parsed = bool(variables) or _try_render(template, [{"role": "user", "content": "x"}]) is not None

    # tools ---------------------------------------------------------------
    supports_tools = False
    if "tools" in variables:
        out = _try_render(template, [{"role": "user", "content": _PROBE_USER}], tools=_PROBE_TOOLS)
        if out is None:
            out = _try_render(
                template,
                [{"role": "system", "content": _PROBE_SYSTEM}, {"role": "user", "content": _PROBE_USER}],
                tools=_PROBE_TOOLS,
            )
        supports_tools = out is not None and _PROBE_TOOL_NAME in out

    # system role ----------------------------------------------------------
    out = _try_render(
        template, [{"role": "system", "content": _PROBE_SYSTEM}, {"role": "user", "content": _PROBE_USER}]
    )
    supports_system = out is not None and _PROBE_SYSTEM in out if parsed else True

    # tool role -------------------------------------------------------------
    supports_tool_role = False
    if parsed:
        mentioned = bool(_TOOL_ROLE_MENTION.search(template))
        if mentioned or supports_tools:
            supports_tool_role = _probe_tool_role(template, supports_tools)

    # thinking ---------------------------------------------------------------
    thinking_kwarg = _probe_thinking_kwarg(template, variables) if parsed else None
    effort_kwarg, effort_levels = _probe_effort(template, variables) if parsed else (None, [])
    supports_thinking = thinking_kwarg is not None or any(m in template for m in _REASONING_MARKERS)
    if not supports_thinking and parsed:
        gen = _try_render(template, [{"role": "user", "content": _PROBE_USER}])
        if gen is not None:
            tail = gen.rstrip()
            supports_thinking = any(tail.endswith(o) for o in REASONING_OPENERS)

    return TemplateInfo(
        supports_tools=supports_tools,
        supports_tool_role=supports_tool_role,
        supports_system=supports_system,
        supports_thinking=supports_thinking,
        thinking_kwarg=thinking_kwarg,
        tool_format=detect_tool_format(None, template),
        stop_strings=stop_strings_for_template(template),
        effort_levels=effort_levels,
        effort_kwarg=effort_kwarg,
    )


def inspect_template(template: str | None) -> TemplateInfo:
    """Detect what a chat template supports (see module docstring).

    ``None``/empty templates yield conservative defaults: no native tools (the
    formatter will prompt for ``<tool_call>`` JSON), system role assumed.
    """
    if not template or not template.strip():
        return TemplateInfo(tool_format="hermes")
    info = _inspect_cached(template)
    return replace(info, stop_strings=list(info.stop_strings), effort_levels=list(info.effort_levels))


def model_effort_levels(model: Any) -> list[str]:
    """Reasoning-effort levels a ``LocalModel``'s chat template can express.

    Best effort and never raising: models whose metadata carries no template
    (or whose template has no switch) get ``[]``, which every surface reads as
    "hide the effort control".
    """
    meta = getattr(model, "meta", None)
    template = getattr(meta, "chat_template", None)
    if not isinstance(template, str) or not template.strip():
        return []
    try:
        return list(inspect_template(template).effort_levels)
    except Exception:  # noqa: BLE001 - a broken template must not break a listing
        return []
