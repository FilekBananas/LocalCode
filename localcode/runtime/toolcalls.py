"""Tool-call and reasoning parsing for local models — the robustness core.

Small local models are sloppy: they forget closing tags, wrap calls in code
fences, emit Python dicts instead of JSON, put raw newlines inside JSON
strings, misspell tool names or call tools from inside their reasoning. This
module turns *any* of the common output conventions into clean
``(content, reasoning, tool_calls, errors)``.

Supported conventions
---------------------
Tool calls (``detect_tool_format`` names in brackets):

* ``<tool_call>{json}</tool_call>`` — Hermes, Qwen 2.5/3, many fine-tunes [hermes]
* ``<tool_call><function=name><parameter=p>v</parameter></function></tool_call>``
  — Qwen3-Coder, Qwen3.5, Nemotron 3 [qwen3_coder]; ``<seed:tool_call>`` wrapper [seed_oss]
* ``{"name": …, "parameters": …}`` and ``<|python_tag|>`` — Llama 3.x [llama3]
* ``[TOOL_CALLS][{…}]`` / ``[TOOL_CALLS]name[CALL_ID]id[ARGS]{…}`` — Mistral, Devstral [mistral]
* gpt-oss harmony ``<|channel|>commentary to=functions.x <|constrain|>json<|message|>{…}<|call|>`` [harmony]
* ``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`` — GLM 4.5–4.7 [glm]
* ``<｜tool▁calls▁begin｜>…`` DeepSeek V3/R1/V3.1 and V3.2 DSML [deepseek]
* ``<|tool_calls_section_begin|>…functions.x:0<|tool_call_argument_begin|>{…}`` — Kimi K2 [kimi_k2]
* ``<minimax:tool_call><invoke name=…><parameter name=…>`` — MiniMax M2 [minimax_m2]
* ``<|tool_call>call:x{k:<|"|>v<|"|>}<tool_call|>`` — Gemma 4 [gemma4];
  ``<start_function_call>call:x{k:<escape>v<escape>}`` — FunctionGemma [function_gemma];
  ````` ```tool_code ````` fences — Gemma 3
* ``[func(a=1)]`` / ``<|tool_call_start|>[…]<|tool_call_end|>`` — LFM2, Llama 4 [pythonic]
* ``<|tool_call|>[…]`` Granite/Phi-4, ``<TOOLCALL>[…]`` Nemotron, ``<|START_ACTION|>`` Command-R
* bare or fenced ````` ```json {"name": …, "arguments": …} ````` blocks (any model) [json]

Reasoning: ``<think>…</think>`` (also when the prompt already opened the block:
``starts_in_reasoning=True``), harmony ``analysis`` channel, ``[THINK]…[/THINK]``
(Magistral), ``<seed:think>``, ``<|channel>thought…<channel|>`` (Gemma 4),
``<|begin_of_thought|>``, ``<|START_THINKING|>``, ``<longcat_think>``, ``◁think▷``.

Design
------
One incremental state machine, ``_Segmenter``, is shared by the streaming API
(``StreamSplitter``) and the one-shot API (``parse_output``). It classifies the
output into ``text`` / ``reasoning`` segments and hidden tool-call *blocks*;
markers split across chunk boundaries are held back until they can be decided.
Because both APIs run the same machine, the visible stream is by construction
identical to the final parse, and chunking never changes the result. The only
exceptions are ``parse_output``'s two *retroactive* repairs, which need the
whole answer and therefore cannot happen mid-stream: an orphan ``</think>``
that turns everything before it into reasoning, and the ``_fallback_calls``
rescue of a call written as prose (ReAct ``Action:``…) when no markup was found
at all. Those answers come back with ``ParsedOutput.repaired_after_stream``
set, so a UI that streamed the text knows it has to re-render it.

Blocks are then parsed by family-specific parsers (the detected format first,
then every other family, then a generic parser), JSON is repaired by a
tolerant parser (trailing commas, single quotes, Python literals, raw newlines
and unescaped quotes inside strings, missing brackets, JSON-string arguments),
tool names are fuzzy-matched against the known tools, and arguments are
coerced to the declared JSON-schema types. Anything that looks like an
attempted call but cannot be used produces an English, model-facing entry in
``ParsedOutput.errors`` that says how to fix it.

Heuristics that protect normal prose: markers inside inline code (preceded by
a backtick) are literal; turn/stop tokens inside fenced code are literal; bare
JSON is only treated as a call at the start of the answer or of a line and
only when its shape is a call; a tool block that is never closed and does not
look like a call is given back as text at the end.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import secrets
import string
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from localcode.hub.types import LocalModel

__all__ = [
    "ToolCall",
    "ParsedOutput",
    "StreamSplitter",
    "TOOL_FORMATS",
    "REASONING_OPENERS",
    "REASONING_CLOSERS",
    "detect_tool_format",
    "parse_output",
    "prompted_tools_block",
    "format_hint",
    "render_tool_call",
    "loads_lenient",
    "new_call_id",
]


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""

    def to_message(self) -> dict[str, Any]:
        """OpenAI-style entry for an assistant message's ``tool_calls`` list."""
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class ParsedOutput:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # True when the final parse had to repair something a stream could not undo
    # (an orphan "</think>", a call written as prose): whatever a StreamSplitter
    # already showed for this answer differs from ``content``/``reasoning``.
    repaired_after_stream: bool = False


TOOL_FORMATS: tuple[str, ...] = (
    "hermes", "qwen3_coder", "seed_oss", "llama3", "mistral", "harmony", "glm", "longcat", "deepseek",
    "kimi_k2", "minimax_m2", "gemma4", "function_gemma", "pythonic", "granite", "nemotron", "command_r",
    "json",
)

_ID_ALPHABET = string.ascii_letters + string.digits


def new_call_id() -> str:
    """9 random alphanumerics — the strictest format any template demands (Mistral)."""
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(9))


# --------------------------------------------------------------------------- #
# Marker tables
# --------------------------------------------------------------------------- #

_REASONING_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("<think>", ("</think>",)),
    ("<thinking>", ("</thinking>",)),
    ("<seed:think>", ("</seed:think>",)),
    ("[THINK]", ("[/THINK]",)),
    ("<|begin_of_thought|>", ("<|end_of_thought|>",)),
    ("<|START_THINKING|>", ("<|END_THINKING|>",)),
    ("<longcat_think>", ("</longcat_think>",)),
    ("\u25c1think\u25b7", ("\u25c1/think\u25b7",)),
    ("<|channel>thought", ("<channel|>",)),
)
_HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
REASONING_OPENERS: tuple[str, ...] = tuple(o for o, _ in _REASONING_PAIRS) + (_HARMONY_ANALYSIS_OPEN,)
REASONING_CLOSERS: tuple[str, ...] = tuple(c for _, cs in _REASONING_PAIRS for c in cs)


@dataclass(frozen=True)
class _OpenerSpec:
    opener: str
    kind: str
    closers: tuple[str, ...]
    to_end: bool = False          # the block may legitimately run to the end of the output
    json_aware: bool = False      # when the payload starts with { or [, closers only count after the value
    inner_end: str | None = None  # tag that proves an unclosed XML-ish block is complete


_TOOL_OPENERS: tuple[_OpenerSpec, ...] = (
    _OpenerSpec("<tool_call>", "tool_call", ("</tool_call>",), json_aware=True),
    _OpenerSpec("<function=", "xml_function", ("</function>",)),
    _OpenerSpec("<seed:tool_call>", "seed", ("</seed:tool_call>",), inner_end="</function>"),
    _OpenerSpec("<longcat_tool_call>", "longcat", ("</longcat_tool_call>",)),
    _OpenerSpec("<minimax:tool_call>", "invoke", ("</minimax:tool_call>",), inner_end="</invoke>"),
    _OpenerSpec("<function_calls>", "invoke", ("</function_calls>",), inner_end="</invoke>"),
    _OpenerSpec("<|tool_call>", "gemma4", ("<tool_call|>",)),
    _OpenerSpec("<start_function_call>", "function_gemma", ("<end_function_call>",)),
    _OpenerSpec("<|tool_call_start|>", "lfm2", ("<|tool_call_end|>",), json_aware=True),
    _OpenerSpec("<|tool_calls_section_begin|>", "kimi", ("<|tool_calls_section_end|>",), inner_end="<|tool_call_end|>"),
    _OpenerSpec("<|tool_call_begin|>", "kimi", ("<|tool_call_end|>",)),
    _OpenerSpec("<\uff5ctool\u2581calls\u2581begin\uff5c>", "deepseek", ("<\uff5ctool\u2581calls\u2581end\uff5c>",),
                inner_end="<\uff5ctool\u2581call\u2581end\uff5c>"),
    _OpenerSpec("<\uff5ctool\u2581call\u2581begin\uff5c>", "deepseek", ("<\uff5ctool\u2581call\u2581end\uff5c>",)),
    _OpenerSpec("<\uff5cDSML\uff5cfunction_calls>", "dsml", ("</\uff5cDSML\uff5cfunction_calls>",),
                inner_end="</\uff5cDSML\uff5cinvoke>"),
    _OpenerSpec("<\uff5cDSML\uff5ctool_calls>", "dsml", ("</\uff5cDSML\uff5ctool_calls>",),
                inner_end="</\uff5cDSML\uff5cinvoke>"),
    _OpenerSpec("[TOOL_CALLS]", "mistral", (), to_end=True),
    _OpenerSpec("<|python_tag|>", "python_tag", ("<|eom_id|>", "<|eot_id|>"), to_end=True),
    _OpenerSpec("<|tool_call|>", "json_list", ("<|/tool_call|>",), to_end=True, json_aware=True),
    _OpenerSpec("<TOOLCALL>", "json_list", ("</TOOLCALL>",), json_aware=True),
    _OpenerSpec("<|START_ACTION|>", "json_list", ("<|END_ACTION|>",), json_aware=True),
)
_OPENER_BY_TEXT = {s.opener: s for s in _TOOL_OPENERS}

_HARMONY_TOKENS = ("<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|call|>", "<|return|>", "<|constrain|>")

# Turn/stop tokens that carry no content: removed wherever they leak into text.
_DROP_TOKENS = (
    "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|endoftext|>", "<\uff5cend\u2581of\u2581sentence\uff5c>",
    "<turn|>", "<|end_of_text|>", "<seed:eos>", "[e~[", "<|begin_of_solution|>", "<|end_of_solution|>",
    "<|START_RESPONSE|>", "<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>",
)

# Tokens after which the model is hallucinating a new turn or a fake tool result.
_TRUNCATE_TOKENS = (
    "<|im_start|>", "<|start_header_id|>", "<start_of_turn>", "<|turn>", "<\uff5cUser\uff5c>", "<|user|>",
    "<|observation|>", "]~b]", "<tool_response>", "<|tool_response>", "<start_function_response>",
    "<|tool_response_start|>", "<\uff5ctool\u2581outputs\u2581begin\uff5c>", "<\uff5ctool\u2581output\u2581begin\uff5c>",
    "[TOOL_RESULTS]",
)

_FENCE = "```"

_TOOLISH_KEYS = frozenset({
    "name", "function", "tool", "tool_name", "function_name", "type", "action", "recipient_name",
    "tool_calls", "function_call", "id", "call_id", "tool_call", "toolname", "command_name",
})
_NAME_KEYS = ("name", "tool_name", "function_name", "tool", "function", "action", "recipient_name",
              "toolname", "command_name")
_ARG_KEYS = ("arguments", "parameters", "args", "input", "tool_input", "action_input", "params", "kwargs",
             "argument", "tool_args", "tool_arguments", "function_arguments")
_ID_KEYS = ("id", "call_id", "tool_call_id")
_IGNORABLE_KEYS = frozenset({"id", "call_id", "tool_call_id", "type", "index"})

_JSON_FENCE_LANGS = frozenset({"json", "json5", "jsonc", "javascript", "js", "tool", "tool_json"})
_TOOL_FENCE_LANGS = frozenset({"tool_code", "tool_call", "tool_calls", "function_call", "functions", "tool_use"})
_PY_FENCE_LANGS = frozenset({"python", "py", ""})


# --------------------------------------------------------------------------- #
# Tolerant JSON
# --------------------------------------------------------------------------- #


class _JsonRepairError(ValueError):
    pass


_WS = " \t\r\n\ufeff"
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)")
_BARE_KEY_RE = re.compile(r"[A-Za-z_$][\w$.\-]*")
_NEXT_KEY_RE = re.compile(
    r"""\s*(?:"(?:[^"\\\n]|\\.){0,200}"|'(?:[^'\\\n]|\\.){0,200}'|[A-Za-z_$][\w$.\-]{0,100})\s*[:=]"""
)
_LITERALS = (
    ("true", True), ("false", False), ("null", None), ("True", True), ("False", False), ("None", None),
    ("undefined", None),
)
_ESCAPES = {'"': '"', "'": "'", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_IDENT_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_MAX_DEPTH = 200
_SURROGATE_ESCAPE_RE = re.compile(r"\\u[dD][89abcdefABCDEF]")


class _Lenient:
    """Recursive-descent JSON parser that repairs what small models get wrong."""

    def __init__(self, s: str, pos: int = 0) -> None:
        self.s = s
        self.i = pos
        self.n = len(s)
        self.truncated = False
        self.string_truncated = False  # the input ended *inside* a string literal
        self.repaired = False
        self.depth = 0

    # -- helpers ----------------------------------------------------------
    def _err(self, msg: str) -> _JsonRepairError:
        line = self.s.count("\n", 0, self.i) + 1
        col = self.i - (self.s.rfind("\n", 0, self.i) + 1) + 1
        return _JsonRepairError(f"{msg} at line {line} column {col}")

    def ws(self) -> None:
        s, n = self.s, self.n
        while self.i < n:
            c = s[self.i]
            if c in _WS:
                self.i += 1
            elif c == "/" and self.i + 1 < n and s[self.i + 1] == "/":
                j = s.find("\n", self.i)
                self.i = n if j < 0 else j + 1
                self.repaired = True
            elif c == "/" and self.i + 1 < n and s[self.i + 1] == "*":
                j = s.find("*/", self.i + 2)
                self.i = n if j < 0 else j + 2
                self.repaired = True
            else:
                break

    def _closes_string(self, j: int, ctx: str, is_key: bool) -> bool:
        s, n = self.s, self.n
        while j < n and s[j] in _WS:
            j += 1
        if j >= n:
            return True
        c = s[j]
        if is_key:
            return c in ":="
        if ctx == "object":
            if c == "}":
                return True
            if c == ",":
                k = j + 1
                while k < n and s[k] in _WS:
                    k += 1
                return k >= n or s[k] == "}" or bool(_NEXT_KEY_RE.match(s, k))
            return False
        if ctx == "array":
            return c in ",]"
        return True

    # -- values -----------------------------------------------------------
    def value(self, ctx: str) -> Any:
        self.ws()
        if self.i >= self.n:
            self.truncated = True
            return None
        s = self.s
        c = s[self.i]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c in "\"'":
            if s.startswith(c * 3, self.i):
                return self.triple(c)
            return self.string(c, ctx, False)
        m = _NUM_RE.match(s, self.i)
        if m and (m.end() >= self.n or s[m.end()] in _WS + ",}]):"):
            self.i = m.end()
            text = m.group().lstrip("+")
            try:
                if any(ch in text for ch in ".eE"):
                    return float(text)
                return int(text)
            except ValueError:  # pragma: no cover - regex guarantees a number
                return text
        for word, val in _LITERALS:
            if s.startswith(word, self.i):
                end = self.i + len(word)
                if end >= self.n or s[end] not in _IDENT_CHARS:
                    self.i = end
                    if word not in ("true", "false", "null"):
                        self.repaired = True
                    return val
        return self.bare(ctx)

    def bare(self, ctx: str) -> str:
        s, n = self.s, self.n
        start = self.i
        if ctx not in ("object", "array"):
            # A bare word at the top level is prose, not JSON. Accepting it would
            # make `loads_lenient` return arbitrary text as a "value".
            raise self._err(f"unexpected character {s[start]!r}" if start < n else "unexpected end of input")
        while self.i < n and s[self.i] not in ",}]\n":
            self.i += 1
        text = s[start:self.i].strip()
        if not text:
            raise self._err(f"unexpected character {s[start]!r}" if start < n else "unexpected end of input")
        self.repaired = True
        return text

    def triple(self, q: str) -> str:
        start = self.i + 3
        end = self.s.find(q * 3, start)
        self.repaired = True
        if end < 0:
            self.truncated = self.string_truncated = True
            self.i = self.n
            return self.s[start:]
        self.i = end + 3
        return self.s[start:end]

    def string(self, q: str, ctx: str, is_key: bool) -> str:
        s, n = self.s, self.n
        i = self.i + 1
        out: list[str] = []
        special = re.compile(r"[\\" + q + "]")
        while True:
            m = special.search(s, i)
            if m is None:
                out.append(s[i:])
                self.i = n
                self.truncated = self.string_truncated = True
                return "".join(out)
            j = m.start()
            if j > i:
                out.append(s[i:j])
            if s[j] == "\\":
                if j + 1 >= n:
                    self.i = n
                    self.truncated = self.string_truncated = True
                    return "".join(out)
                e = s[j + 1]
                if e in _ESCAPES:
                    out.append(_ESCAPES[e])
                    i = j + 2
                elif e == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[j + 2:j + 6]):
                    code = int(s[j + 2:j + 6], 16)
                    i = j + 6
                    if 0xD800 <= code < 0xDC00 and s.startswith("\\u", i) and re.fullmatch(
                        r"[dD][c-fC-F][0-9a-fA-F]{2}", s[i + 2:i + 6]
                    ):
                        low = int(s[i + 2:i + 6], 16)
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        i += 6
                    out.append("\ufffd" if 0xD800 <= code < 0xE000 else chr(code))
                elif e == "\n":
                    i = j + 2
                    self.repaired = True
                else:  # invalid escape such as \d in a regex: keep it literally
                    out.append("\\" + e)
                    i = j + 2
                    self.repaired = True
                continue
            # a quote character
            if self._closes_string(j + 1, ctx, is_key):
                self.i = j + 1
                return "".join(out)
            out.append(q)
            i = j + 1
            self.repaired = True

    def obj(self) -> dict[str, Any]:
        self.i += 1
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise self._err("nesting too deep")
        result: dict[str, Any] = {}
        s = self.s
        while True:
            self.ws()
            if self.i >= self.n:
                self.truncated = True
                break
            c = s[self.i]
            if c == "}":
                self.i += 1
                break
            if c == ",":
                self.i += 1
                self.repaired = True
                continue
            if c in "\"'":
                key = self.string(c, "object", True)
            else:
                m = _BARE_KEY_RE.match(s, self.i)
                if not m:
                    raise self._err(f"expected a property name but found {c!r}")
                key = m.group()
                self.i = m.end()
                self.repaired = True
            self.ws()
            if self.i >= self.n:
                self.truncated = True
                break
            if s[self.i] in ":=":
                self.i += 1
            else:
                raise self._err(f"expected ':' after property name {key!r}")
            result[key] = self.value("object")
            if self.truncated:
                break
            self.ws()
            if self.i >= self.n:
                self.truncated = True
                break
            c = s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == "}":
                self.i += 1
                break
            if c in "])":
                self.i += 1
                self.repaired = True
                break
            if _NEXT_KEY_RE.match(s, self.i):
                self.repaired = True
                continue
            raise self._err(f"expected ',' or '}}' but found {c!r}")
        self.depth -= 1
        return result

    def arr(self) -> list[Any]:
        self.i += 1
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise self._err("nesting too deep")
        result: list[Any] = []
        s = self.s
        while True:
            self.ws()
            if self.i >= self.n:
                self.truncated = True
                break
            c = s[self.i]
            if c == "]":
                self.i += 1
                break
            if c == ",":
                self.i += 1
                self.repaired = True
                continue
            result.append(self.value("array"))
            if self.truncated:
                break
            self.ws()
            if self.i >= self.n:
                self.truncated = True
                break
            c = s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == "]":
                self.i += 1
                break
            if c in "})":
                self.i += 1
                self.repaired = True
                break
            if c in "\"'{[-0123456789tfnTFN":
                self.repaired = True
                continue
            raise self._err(f"expected ',' or ']' but found {c!r}")
        self.depth -= 1
        return result


@dataclass
class _JsonResult:
    value: Any
    end: int
    truncated: bool
    repaired: bool
    string_truncated: bool = False   # the text ran out inside a string: data is really missing


def _parse_json_at(text: str, start: int = 0) -> _JsonResult:
    """Parse one (possibly broken) JSON value starting at ``start``; trailing text is ignored."""
    # ``json`` happily decodes a lone surrogate escape into an unencodable
    # string that would later blow up on `.encode("utf-8")` in a tool or on the
    # wire; the repairing parser replaces those with U+FFFD, so route them there.
    if not _SURROGATE_ESCAPE_RE.search(text, start):
        try:
            value, end = json.JSONDecoder().raw_decode(text, start)
            return _JsonResult(value, end, False, False)
        except (json.JSONDecodeError, ValueError):
            pass
    p = _Lenient(text, start)
    try:
        value = p.value("top")
    except RecursionError as exc:  # pragma: no cover - guarded by _MAX_DEPTH
        raise _JsonRepairError("nesting too deep") from exc
    return _JsonResult(value, p.i, p.truncated, True, p.string_truncated)


_FENCE_STRIP_RE = re.compile(r"^\s*```[\w+\-]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def loads_lenient(text: str) -> Any:
    """``json.loads`` that repairs common model mistakes; raises ``ValueError``
    with a short description when the text cannot be understood as JSON."""
    if not isinstance(text, str):
        raise ValueError("expected a string")
    if not _SURROGATE_ESCAPE_RE.search(text):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
    m = _FENCE_STRIP_RE.match(text)
    if m:
        text = m.group(1)
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty input")
    start = len(text) - len(text.lstrip())
    result = _parse_json_at(text, start)
    return result.value


class _BalanceScanner:
    """Incrementally finds where a bracketed value ends (brackets inside strings ignored)."""

    __slots__ = ("pos", "depth", "quote", "escape", "started")
    _INTERESTING = re.compile(r"[\"'\\{}\[\]()]")

    def __init__(self) -> None:
        self.pos = 0
        self.depth = 0
        self.quote: str | None = None
        self.escape = False
        self.started = False

    def scan(self, s: str) -> int:
        """Scan ``s[self.pos:]``; return the index just past the value, or -1."""
        i = self.pos
        n = len(s)
        depth, quote, esc, started = self.depth, self.quote, self.escape, self.started
        result = -1
        while i < n:
            if esc:
                esc = False
                i += 1
                continue
            m = self._INTERESTING.search(s, i)
            if m is None:
                i = n
                break
            i = m.start()
            c = s[i]
            if quote:
                if c == "\\":
                    esc = True
                elif c == quote:
                    quote = None
            elif c == "\\":
                pass
            elif c in "\"'":
                if c == "'" and 0 < i < n - 1 and s[i - 1].isalnum() and s[i + 1].isalnum():
                    pass  # apostrophe in a bare word
                else:
                    quote = c
            elif c in "{[(":
                depth += 1
                started = True
            else:
                depth -= 1
                if started and depth <= 0:
                    result = i + 1
                    i += 1
                    break
            i += 1
        self.pos, self.depth, self.quote, self.escape, self.started = i, depth, quote, esc, started
        return result


def _iter_json_values(text: str) -> Iterable[tuple[int, int, _JsonResult]]:
    """All top-level ``{…}``/``[…]`` values in ``text`` with their spans."""
    i = 0
    n = len(text)
    while i < n:
        m = re.compile(r"[{\[]").search(text, i)
        if m is None:
            return
        start = m.start()
        scanner = _BalanceScanner()
        scanner.pos = start
        end = scanner.scan(text)
        try:
            res = _parse_json_at(text, start)
        except _JsonRepairError:
            i = start + 1
            continue
        if end < 0:
            end = max(res.end, start + 1)
        yield start, max(end, res.end) if res.truncated else end, res
        i = end if end > start else start + 1


# --------------------------------------------------------------------------- #
# Known tools: name resolution and argument coercion
# --------------------------------------------------------------------------- #

_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "bash": ("shell", "run", "run_command", "execute", "execute_command", "exec", "terminal", "run_shell_command",
             "run_shell", "command", "sh", "cmd", "execute_bash", "run_bash", "bash_command", "shell_command"),
    "read_file": ("read", "cat", "open_file", "view_file", "view", "file_read", "get_file", "read_text_file",
                  "open", "readfile", "show_file"),
    "write_file": ("write", "create_file", "save_file", "file_write", "write_to_file", "writefile", "create"),
    "edit_file": ("edit", "str_replace", "replace_in_file", "apply_edit", "modify_file", "update_file",
                  "file_edit", "patch_file", "replace"),
    "list_dir": ("ls", "list_files", "list_directory", "listdir", "dir", "list", "list_folder", "list_dir_contents"),
    "grep": ("search", "search_files", "find_in_files", "rg", "ripgrep", "code_search", "search_code",
             "grep_search", "search_text"),
    "glob": ("find_files", "find", "file_search", "glob_files", "find_file"),
    "todo_write": ("todo", "todos", "update_todos", "write_todos", "todo_list", "update_todo", "set_todos"),
}
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file", "file_path", "filepath", "filename", "file_name", "target_file", "dir", "directory",
             "folder", "dir_path", "directory_path"),
    "command": ("cmd", "shell_command", "script", "bash_command", "commands"),
    "pattern": ("regex", "query", "search", "search_pattern", "glob", "expression"),
    "content": ("text", "contents", "data", "file_content", "body", "file_text", "code"),
    "old_string": ("old", "old_str", "old_text", "search_text", "find", "original"),
    "new_string": ("new", "new_str", "new_text", "replace_text", "replacement"),
}
_NAMESPACE_PREFIX_RE = re.compile(r"^(?:functions|function|default_api|tools|tool|api|repo_browser|container)[.:]")


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_ALIAS_BY_NORM = {_norm_name(a): target for target, aliases in _NAME_ALIASES.items() for a in aliases}


def _has_items(known: Any) -> bool:
    try:
        return len(known) > 0
    except TypeError:
        return True


@dataclass
class _ToolSpec:
    name: str
    properties: dict[str, Any]
    required: tuple[str, ...]


class _ToolIndex:
    def __init__(self, specs: list[_ToolSpec], *, disabled: bool = False) -> None:
        self.specs = {s.name: s for s in specs}
        self.names = [s.name for s in specs]
        self.disabled = disabled  # built from an empty list: the caller offers no tools at all
        self._norm: dict[str, str] = {}
        for s in specs:
            self._norm.setdefault(_norm_name(s.name), s.name)

    @classmethod
    def build(cls, known: Any) -> "_ToolIndex | None":
        if known is None:
            return None
        if isinstance(known, _ToolIndex):
            return known
        if isinstance(known, (str, dict)):
            known = [known]
        specs: list[_ToolSpec] = []
        for item in known:
            name: str | None = None
            params: Any = None
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                fn = item.get("function") if isinstance(item.get("function"), dict) else item
                name = fn.get("name")
                params = fn.get("parameters")
            else:
                name = getattr(item, "name", None)
                params = getattr(item, "parameters", None)
            if not isinstance(name, str) or not name:
                continue
            props = params.get("properties") if isinstance(params, dict) else None
            req = params.get("required") if isinstance(params, dict) else None
            specs.append(_ToolSpec(
                name,
                dict(props) if isinstance(props, dict) else {},
                tuple(r for r in req if isinstance(r, str)) if isinstance(req, list) else (),
            ))
        return cls(specs, disabled=not specs and not _has_items(known))

    def exact(self, name: str) -> str | None:
        """The tool called exactly ``name`` (spelling normalised), never a guess.

        Used where a wrong guess would be worse than no answer — reading
        ``<body>`` in an HTML page as a call to ``bash``, for one.
        """
        if name in self.specs:
            return name
        return self._norm.get(_norm_name(name))

    def resolve(self, name: str) -> tuple[str, bool]:
        """Return ``(best_name, is_known)``."""
        cleaned = name.strip().strip("`'\"").strip()
        if cleaned.endswith("()"):
            cleaned = cleaned[:-2]
        if cleaned in self.specs:
            return cleaned, True
        candidates = [cleaned]
        stripped = _NAMESPACE_PREFIX_RE.sub("", cleaned)
        if stripped != cleaned:
            candidates.append(stripped)
        if "." in cleaned:
            candidates.append(cleaned.rsplit(".", 1)[1])
        for cand in candidates:
            if cand in self.specs:
                return cand, True
        for cand in candidates:
            hit = self._norm.get(_norm_name(cand))
            if hit:
                return hit, True
        for cand in candidates:
            target = _ALIAS_BY_NORM.get(_norm_name(cand))
            if target is not None and target in self.specs:
                return target, True
        norm_names = list(self._norm)
        for cand in candidates:
            close = difflib.get_close_matches(_norm_name(cand), norm_names, n=1, cutoff=0.8)
            if close:
                return self._norm[close[0]], True
        return stripped or cleaned, False

    def types_of(self, tool: str, param: str) -> set[str] | None:
        spec = self.specs.get(tool)
        if spec is None:
            return None
        schema = spec.properties.get(param)
        if not isinstance(schema, dict):
            return None
        return _schema_types(schema) or None

    def fix_arg_names(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self.specs.get(tool)
        if spec is None or not spec.properties:
            return args
        props = spec.properties
        norm_props = {_norm_name(p): p for p in props}
        out: dict[str, Any] = {}
        for key, value in args.items():
            if key in props:
                out[key] = value
                continue
            target = norm_props.get(_norm_name(key))
            if target is None:
                for canonical, aliases in _ARG_ALIASES.items():
                    if canonical in props and key.lower() in aliases and canonical not in args:
                        target = canonical
                        break
            if target is not None and target not in out and target not in args:
                out[target] = value
            else:
                out[key] = value
        return out


def _schema_types(schema: dict[str, Any]) -> set[str]:
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or ():
            if isinstance(sub, dict):
                types |= _schema_types(sub)
    if not types and isinstance(schema.get("enum"), list):
        for v in schema["enum"]:
            types.add("string" if isinstance(v, str) else "number" if isinstance(v, (int, float)) else "string")
    return types


_INT_RE = re.compile(r"[-+]?\d+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_TRUE = frozenset({"true", "yes", "1", "on"})
_FALSE = frozenset({"false", "no", "0", "off"})


def _coerce_value(value: Any, types: set[str] | None, textual: bool) -> Any:
    """Convert ``value`` towards the declared JSON-schema ``types``."""
    if types is None:
        return _guess_scalar(value) if textual and isinstance(value, str) else value
    if isinstance(value, str):
        v = value.strip()
        if "string" in types:
            return value
        if "null" in types and v.lower() in ("null", "none"):
            return None
        if "integer" in types and _INT_RE.fullmatch(v):
            return int(v)
        if "number" in types and _FLOAT_RE.fullmatch(v):
            f = float(v)
            return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        if "integer" in types and _FLOAT_RE.fullmatch(v) and float(v).is_integer():
            return int(float(v))
        if "boolean" in types and v.lower() in _TRUE | _FALSE:
            return v.lower() in _TRUE
        if types & {"array", "object"}:
            try:
                parsed = loads_lenient(v)
            except ValueError:
                parsed = None
            if isinstance(parsed, list) and "array" in types:
                return parsed
            if isinstance(parsed, dict) and "object" in types:
                return parsed
            if "array" in types and v:
                return [value]
        return value
    if types == {"string"}:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, float) and "integer" in types and "number" not in types and value.is_integer():
        return int(value)
    if isinstance(value, dict) and types == {"array"}:
        return [value]
    return value


def _guess_scalar(value: str) -> Any:
    """Type a textual (XML-ish) argument when no schema is known."""
    v = value.strip()
    if not v:
        return value
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    if _INT_RE.fullmatch(v) and not (len(v) > 1 and v.lstrip("+-").startswith("0")):
        return int(v)
    if _FLOAT_RE.fullmatch(v) and any(c in v for c in ".eE"):
        try:
            return float(v)
        except ValueError:  # pragma: no cover
            return value
    if v[0] in "[{" and v[-1] in "]}":
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


# --------------------------------------------------------------------------- #
# Raw calls and their normalisation
# --------------------------------------------------------------------------- #


@dataclass
class _Raw:
    name: str | None
    arguments: Any
    raw: str
    call_id: str | None = None
    textual: bool = False
    error: str | None = None
    truncated: bool = False
    string_truncated: bool = False  # the text ran out inside a string (real data loss)
    open_tail: bool = False       # ended at the end of the text without its closing tag


def _as_args(value: Any) -> tuple[dict[str, Any] | None, str | None, bool, bool]:
    """Normalise an ``arguments`` value → (dict, error, truncated, string_truncated)."""
    if value is None:
        return {}, None, False, False
    if isinstance(value, dict):
        return value, None, False, False
    if isinstance(value, str):
        text = value.strip()
        if not text or text in ("{}", "null", "None"):
            return {}, None, False, False
        try:
            res = _parse_json_at(text, 0)
        except _JsonRepairError as exc:
            return None, f"the arguments string is not valid JSON ({exc})", False, False
        if isinstance(res.value, dict):
            return res.value, None, res.truncated, res.string_truncated
        if isinstance(res.value, str) and res.value.strip() != text:
            return _as_args(res.value)   # double-encoded JSON; the guard stops any loop
        return None, "arguments must be a JSON object", False, False
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            return value[0], None, False, False
        if not value:
            return {}, None, False, False
    return None, f"arguments must be a JSON object, not {type(value).__name__}", False, False


def _raws_from_json(value: Any, raw: str, *, strict: bool, index: _ToolIndex | None,
                    truncated: bool = False, string_truncated: bool = False) -> list[_Raw] | None:
    """Interpret a parsed JSON value as tool call(s). ``None`` = not a call shape."""
    kw = {"strict": strict, "index": index, "truncated": truncated, "string_truncated": string_truncated}
    if isinstance(value, str) and strict is False:
        try:
            inner = loads_lenient(value)
        except ValueError:
            return None
        if isinstance(inner, (dict, list)):
            return _raws_from_json(inner, raw, **kw)
        return None
    if isinstance(value, list):
        out: list[_Raw] = []
        for item in value:
            got = _raws_from_json(item, raw, **kw)
            if got is None:
                if strict:
                    return None
                continue
            out.extend(got)
        return out or None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("tool_calls"), list):
        return _raws_from_json(value["tool_calls"], raw, **kw)
    for wrapper in ("tool_call", "function_call"):   # {"tool_call": {…}} / {"function_call": {…}}
        inner = value.get(wrapper)
        if isinstance(inner, (dict, list)):
            return _raws_from_json(inner, raw, **kw)
    call_id = next((value[k] for k in _ID_KEYS if isinstance(value.get(k), str) and value.get(k)), None)
    fn = value.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        args_val = next((fn[k] for k in _ARG_KEYS if k in fn), None)
        if args_val is None:
            args_val = next((value[k] for k in _ARG_KEYS if k in value), None)
        return [_mk_raw(fn["name"], args_val, raw, call_id, truncated, string_truncated)]
    name_key = next((k for k in _NAME_KEYS if isinstance(value.get(k), str) and value.get(k).strip()), None)
    if name_key is None:
        return None
    name = value[name_key].strip()
    arg_key = next((k for k in _ARG_KEYS if k in value), None)
    extra = [k for k in value if k not in _IGNORABLE_KEYS and k != name_key and k != arg_key]
    if arg_key is not None:
        if strict and extra:
            return None
        if strict and not isinstance(value[arg_key], (dict, str, type(None), list)):
            return None
        return [_mk_raw(name, value[arg_key], raw, call_id, truncated, string_truncated)]
    # no explicit arguments: the remaining keys are the arguments ("flat" style)
    flat = {k: value[k] for k in extra}
    if strict:
        if index is None:
            return None
        resolved, known = index.resolve(name)
        if not known:
            return None
        props = index.specs[resolved].properties
        if flat and props and not all(k in props for k in flat):
            return None
        if not flat and index.specs[resolved].required:
            return None
    return [_mk_raw(name, flat, raw, call_id, truncated, string_truncated)]


def _mk_raw(name: str, args_val: Any, raw: str, call_id: str | None, truncated: bool,
            string_truncated: bool = False) -> _Raw:
    args, err, trunc, str_trunc = _as_args(args_val)
    return _Raw(name=name, arguments=args, raw=raw, call_id=call_id, error=err,
                truncated=truncated or trunc, string_truncated=string_truncated or str_trunc)


def _finalize(raws: list[_Raw], index: _ToolIndex | None, fmt: str | None,
              calls: list[ToolCall], errors: list[str], used_ids: set[str]) -> None:
    for r in raws:
        snippet = _snippet(r.raw)
        if r.truncated:
            errors.append(
                f"The tool call{_named(r.name)} was cut off before it was complete ({snippet}). "
                "Send it again in full; keep arguments short (write large files in several smaller steps)."
            )
            continue
        if r.error or not r.name or r.arguments is None:
            reason = r.error or ("the tool name is missing" if not r.name else "the arguments are invalid")
            errors.append(
                f"Could not use the tool call{_named(r.name)}: {reason} ({snippet}). "
                f"Use exactly this format:\n{format_hint(fmt)}"
            )
            continue
        name = r.name.strip()
        args = dict(r.arguments)
        if index is not None:
            name, known = index.resolve(name)
            if known:
                args = index.fix_arg_names(name, args)
                args = {k: _coerce_value(v, index.types_of(name, k), r.textual) for k, v in args.items()}
            elif r.textual:
                args = {k: _guess_scalar(v) if isinstance(v, str) else v for k, v in args.items()}
        elif r.textual:
            args = {k: _guess_scalar(v) if isinstance(v, str) else v for k, v in args.items()}
        call_id = r.call_id if isinstance(r.call_id, str) and r.call_id.strip() else None
        if call_id is None or call_id in used_ids:
            call_id = new_call_id()
            while call_id in used_ids:  # pragma: no cover - astronomically unlikely
                call_id = new_call_id()
        used_ids.add(call_id)
        calls.append(ToolCall(id=call_id, name=name, arguments=args, raw=r.raw))


def _named(name: str | None) -> str:
    return f" to '{name}'" if name else ""


def _snippet(raw: str, limit: int = 160) -> str:
    one = " ".join(raw.split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Family parsers (payload → raw calls)
# --------------------------------------------------------------------------- #


def _p_json(payload: str, index: _ToolIndex | None, *, strict: bool = False) -> list[_Raw]:
    out: list[_Raw] = []
    stripped = payload.strip()
    # "name\n{json}" / "name {json}" (GLM and some Hermes fine-tunes)
    m = re.match(r"^([A-Za-z_][\w.\-]*)\s*([{\[].*)$", stripped, re.S)
    if m and not stripped.startswith(("{", "[")):
        try:
            res = _parse_json_at(m.group(2), 0)
        except _JsonRepairError as exc:
            return [_Raw(m.group(1), None, stripped, error=f"invalid JSON arguments ({exc})")]
        return [_mk_raw(m.group(1), res.value, stripped, None, res.truncated, res.string_truncated)]
    for start, end, res in _iter_json_values(payload):
        raw = payload[start:end]
        got = _raws_from_json(res.value, raw, strict=strict, index=index, truncated=res.truncated,
                              string_truncated=res.string_truncated)
        if got:
            out.extend(got)
    return out


_XML_FUNC_RE = re.compile(
    r"<function(?:=|\s+name\s*=\s*)[\"']?([^>\"'\n]+?)[\"']?\s*>(.*?)(?:</function>|(?=<function[=\s])|\Z)", re.S
)
_XML_PARAM_RE = re.compile(
    r"<param(?:eter)?(?:=|\s+name\s*=\s*)[\"']?([^>\"'\n]+?)[\"']?\s*>(.*?)"
    r"(?:</param(?:eter)?>|(?=<param(?:eter)?[=\s])|(?=</function>)|\Z)",
    re.S,
)


def _strip_one_newline(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    elif v.startswith("\r\n"):
        v = v[2:]
    if v.endswith("\n"):
        v = v[:-1]
        if v.endswith("\r"):
            v = v[:-1]
    return v


def _p_xml_functions(payload: str) -> list[_Raw]:
    out: list[_Raw] = []
    for m in _XML_FUNC_RE.finditer(payload):
        name = m.group(1).strip()
        body = m.group(2)
        args: dict[str, Any] = {}
        open_tail = False
        for pm in _XML_PARAM_RE.finditer(body):
            args[pm.group(1).strip()] = _strip_one_newline(pm.group(2))
            open_tail = pm.end() >= len(body) and not pm.group(0).endswith(("</parameter>", "</param>"))
        closed = m.group(0).endswith("</function>")
        out.append(_Raw(name, args, m.group(0), textual=True, open_tail=not closed and (open_tail or not args)))
    return out


def _p_glm(payload: str, key_tag: str = "arg_key", value_tag: str = "arg_value") -> list[_Raw]:
    text = payload.strip()
    if f"<{key_tag}>" not in text:
        return []
    name = text.split(f"<{key_tag}>", 1)[0].strip()
    args: dict[str, Any] = {}
    pat = re.compile(
        rf"<{key_tag}>(.*?)</{key_tag}>\s*<{value_tag}>(.*?)(?:</{value_tag}>|(?=<{key_tag}>)|\Z)", re.S
    )
    open_tail = False
    for m in pat.finditer(text):
        args[m.group(1).strip()] = m.group(2).strip()
        open_tail = not m.group(0).endswith(f"</{value_tag}>")
    return [_Raw(name or None, args, payload.strip(), textual=True, open_tail=open_tail)]


_INVOKE_RE = re.compile(r"<(?:\uff5cDSML\uff5c)?invoke\s+name\s*=\s*[\"']?([^\"'>]+)[\"']?\s*>(.*?)"
                        r"(?:</(?:\uff5cDSML\uff5c)?invoke>|\Z)", re.S)
_INVOKE_PARAM_RE = re.compile(
    r"<(?:\uff5cDSML\uff5c)?parameter\s+name\s*=\s*[\"']?([^\"'>]+)[\"']?((?:\s+\w+\s*=\s*[\"'][^\"']*[\"'])*)\s*>"
    r"(.*?)(?:</(?:\uff5cDSML\uff5c)?parameter>|\Z)",
    re.S,
)


def _p_invoke(payload: str) -> list[_Raw]:
    out: list[_Raw] = []
    for m in _INVOKE_RE.finditer(payload):
        args: dict[str, Any] = {}
        typed: dict[str, Any] = {}
        for pm in _INVOKE_PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            value = _strip_one_newline(pm.group(3))
            attrs = pm.group(2) or ""
            if re.search(r"string\s*=\s*[\"']false[\"']", attrs):
                try:
                    typed[key] = json.loads(value)
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass
            args[key] = value
        closed = m.group(0).rstrip().endswith("invoke>")
        out.append(_Raw(m.group(1).strip(), {**args, **typed}, m.group(0), textual=True, open_tail=not closed))
    return out


_KIMI_CALL_RE = re.compile(
    r"(?:<\|tool_call_begin\|>)?\s*((?:functions\.)?([\w.\-]+?)(?::\d+)?)\s*<\|tool_call_argument_begin\|>(.*?)"
    r"(?:<\|tool_call_end\|>|(?=<\|tool_call_begin\|>)|\Z)",
    re.S,
)


def _p_kimi(payload: str) -> list[_Raw]:
    out: list[_Raw] = []
    for m in _KIMI_CALL_RE.finditer(payload):
        call_id = m.group(1).strip()
        name = m.group(2)
        try:
            res = _parse_json_at(m.group(3).strip() or "{}", 0)
        except _JsonRepairError as exc:
            out.append(_Raw(name, None, m.group(0), error=f"invalid JSON arguments ({exc})"))
            continue
        raw = _mk_raw(name, res.value, m.group(0), call_id if ":" in call_id else None, res.truncated,
                      res.string_truncated)
        out.append(raw)
    return out


_DS_SEP = "<\uff5ctool\u2581sep\uff5c>"
_DS_CALL_BEGIN = "<\uff5ctool\u2581call\u2581begin\uff5c>"
_DS_CALL_END = "<\uff5ctool\u2581call\u2581end\uff5c>"
_DS_CALLS_BEGIN = "<\uff5ctool\u2581calls\u2581begin\uff5c>"
_DS_CALLS_END = "<\uff5ctool\u2581calls\u2581end\uff5c>"


def _p_deepseek(payload: str) -> list[_Raw]:
    out: list[_Raw] = []
    parts = [p for p in payload.split(_DS_CALL_BEGIN) if _DS_SEP in p]
    for part in parts:
        body = part.split(_DS_CALL_END, 1)[0]
        head, _, tail = body.partition(_DS_SEP)
        head = head.strip()
        tail_s = tail.strip()
        if head in ("function", ""):
            # V3/R1: function<sep>name\n```json\n{...}\n```
            name, _, rest = tail_s.partition("\n")
            args_text = rest.strip()
        else:
            name, args_text = head, tail_s
        fence = _FENCE_STRIP_RE.match(args_text)
        if fence:
            args_text = fence.group(1)
        try:
            res = _parse_json_at(args_text or "{}", 0)
        except _JsonRepairError as exc:
            out.append(_Raw(name.strip(), None, body, error=f"invalid JSON arguments ({exc})"))
            continue
        out.append(_mk_raw(name.strip(), res.value, body, None, res.truncated, res.string_truncated))
    return out


def _gemma_args_to_json(text: str, delim: str) -> str:
    strings: list[str] = []

    def capture(m: re.Match[str]) -> str:
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = re.sub(re.escape(delim) + r"(.*?)" + re.escape(delim), capture, text, flags=re.S)
    text = re.sub(r"\x00(\d+)\x00", lambda m: json.dumps(strings[int(m.group(1))], ensure_ascii=False), text)
    return text


def _p_gemma_call(payload: str, delim: str) -> list[_Raw]:
    out: list[_Raw] = []
    for m in re.finditer(r"call:([\w.\-]+)\s*(\{)", payload):
        start = m.start(2)
        converted = _gemma_args_to_json(payload[start:], delim)
        try:
            res = _parse_json_at(converted, 0)
        except _JsonRepairError as exc:
            out.append(_Raw(m.group(1), None, payload[m.start():], error=f"invalid arguments ({exc})"))
            continue
        scanner = _BalanceScanner()
        scanner.pos = start
        end = scanner.scan(payload)
        raw = payload[m.start(): end if end > 0 else len(payload)]
        out.append(_mk_raw(m.group(1), res.value, raw, None, res.truncated, res.string_truncated))
    return out


_JSON_LITERAL_FIX = re.compile(r"(?<![\w\"'])(true|false|null)(?![\w\"'])")


def _p_pythonic(payload: str, index: _ToolIndex | None, *, require_known: bool = False) -> list[_Raw]:
    text = payload.strip()
    if not text:
        return []
    fence = _FENCE_STRIP_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    tree = None
    for candidate in (text, _JSON_LITERAL_FIX.sub(lambda m: {"true": "True", "false": "False", "null": "None"}[m.group(1)], text)):
        try:
            tree = ast.parse(candidate, mode="eval")
            text = candidate
            break
        except SyntaxError:
            try:
                tree = ast.parse(candidate, mode="exec")
                text = candidate
                break
            except SyntaxError:
                continue
    if tree is None:
        return []
    nodes: list[ast.AST] = []
    if isinstance(tree, ast.Expression):
        body = tree.body
        if isinstance(body, (ast.List, ast.Tuple)):
            nodes = list(body.elts)
        else:
            nodes = [body]
    else:
        for stmt in tree.body:  # type: ignore[attr-defined]
            if isinstance(stmt, ast.Expr):
                v = stmt.value
                nodes.extend(v.elts if isinstance(v, (ast.List, ast.Tuple)) else [v])
            else:
                return []
    out: list[_Raw] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            return []
        name = _dotted(node.func)
        if name is None:
            return []
        if name.endswith(".call"):  # Llama 3 builtin: brave_search.call(query="…")
            name = name[: -len(".call")]
        if require_known and (index is None or not index.resolve(name)[1]):
            return []
        raw = ast.get_source_segment(text, node) or name
        args: dict[str, Any] = {}
        error = None
        for kw in node.keywords:
            if kw.arg is None:
                try:
                    extra = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError, TypeError):
                    extra = None
                if isinstance(extra, dict):
                    args.update(extra)
                continue
            args[kw.arg] = _literal(kw.value, text)
        if node.args:
            order: list[str] = []
            if index is not None:
                resolved, known = index.resolve(name)
                if known:
                    spec = index.specs[resolved]
                    order = [p for p in spec.properties if p not in args]
            for pos, arg in enumerate(node.args):
                if pos < len(order):
                    args[order[pos]] = _literal(arg, text)
                else:
                    error = "positional arguments are not allowed; use name=value"
                    break
        out.append(_Raw(name, args if error is None else None, raw, error=error))
    return out


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _literal(node: ast.AST, source: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        seg = ast.get_source_segment(source, node)
        return seg if seg is not None else ""


_MISTRAL_CALL_RE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s*(?:\[CALL_ID\]\s*([\w\-]+)\s*)?\[ARGS\]\s*(.*)$", re.S)


def _p_mistral(payload: str, index: _ToolIndex | None) -> list[_Raw]:
    out: list[_Raw] = []
    for part in payload.split("[TOOL_CALLS]"):
        text = part.replace("</s>", "").strip()
        if not text:
            continue
        m = _MISTRAL_CALL_RE.match(text)
        if m:
            try:
                res = _parse_json_at(m.group(3), 0)
            except _JsonRepairError as exc:
                out.append(_Raw(m.group(1), None, text, error=f"invalid JSON arguments ({exc})"))
                continue
            out.append(_mk_raw(m.group(1), res.value, text, m.group(2), res.truncated, res.string_truncated))
            continue
        got = _p_json(text, index)
        if got:
            out.extend(got)
    return out


def _p_python_tag(payload: str, index: _ToolIndex | None) -> list[_Raw]:
    text = payload.strip()
    got = _p_json(text, index)
    if got:
        return got
    return _p_pythonic(text.replace(";", "\n"), index)


def _p_harmony(header: str, payload: str) -> list[_Raw]:
    m = re.search(r"\bto=([^\s<]+)", header)
    recipient = m.group(1) if m else ""
    name = recipient[len("functions."):] if recipient.startswith("functions.") else recipient
    raw = f"{header}<|message|>{payload}"
    text = payload.strip()
    if not text:
        return [_Raw(name, {}, raw)]
    try:
        res = _parse_json_at(text, 0)
    except _JsonRepairError as exc:
        if recipient in ("python", "container.exec"):
            return [_Raw(name, {"code": payload}, raw)]
        return [_Raw(name, None, raw, error=f"invalid JSON arguments ({exc})")]
    if not isinstance(res.value, dict) and recipient == "python":
        return [_Raw(name, {"code": payload}, raw)]
    return [_mk_raw(name, res.value, raw, None, res.truncated, res.string_truncated)]


def _p_generic(payload: str, index: _ToolIndex | None) -> list[_Raw]:
    if re.search(r"<function(?:=|\s+name\s*=)", payload):
        got = _p_xml_functions(payload)
        if got:
            return got
    if "<arg_key>" in payload:
        got = _p_glm(payload)
        if got:
            return got
    if re.search(r"invoke\s+name\s*=", payload):
        got = _p_invoke(payload)
        if got:
            return got
    if "call:" in payload:
        for delim in ('<|"|>', "<escape>"):
            if delim in payload:
                got = _p_gemma_call(payload, delim)
                if got:
                    return got
    got = _p_json(payload, index)
    if got:
        return got
    got = _p_pythonic(payload, index)
    if got:
        return got
    if "call:" in payload:
        got = _p_gemma_call(payload, '<|"|>')
        if got:
            return got
    return []


# --------------------------------------------------------------------------- #
# Blocks (what the segmenter hides) → calls
# --------------------------------------------------------------------------- #


@dataclass
class _Block:
    kind: str
    raw: str
    payload: str
    closed: bool
    in_reasoning: bool = False
    header: str = ""
    inner_end: str | None = None


_KIND_LABEL = {
    "tool_call": "<tool_call> block", "xml_function": "<function=…> block", "seed": "<seed:tool_call> block",
    "longcat": "<longcat_tool_call> block", "invoke": "<invoke> block", "gemma4": "<|tool_call> block",
    "function_gemma": "<start_function_call> block", "lfm2": "<|tool_call_start|> block", "kimi": "tool call section",
    "deepseek": "tool call block", "dsml": "DSML tool call block", "mistral": "[TOOL_CALLS] block",
    "python_tag": "<|python_tag|> block", "json_list": "tool call block", "fence_code": "```tool_code block",
    "json": "JSON tool call", "pythonic": "function-call list", "harmony": "commentary tool message",
}


def _parse_block(block: _Block, fmt: str | None, index: _ToolIndex | None) -> list[_Raw]:
    kind = block.kind
    payload = block.payload
    raws: list[_Raw] = []
    if kind == "tool_call":
        stripped = payload.strip()
        order: list[str]
        sniff_xml = bool(re.search(r"<function(?:=|\s+name\s*=)", payload))
        if sniff_xml:
            order = ["xml", "json", "glm"]
        elif "<arg_key>" in payload:
            order = ["glm", "json", "xml"]
        elif re.search(r"invoke\s+name\s*=", payload):
            order = ["invoke", "json"]
        elif stripped.startswith(("{", "[")):
            order = ["json", "xml", "glm"]
        else:
            order = ["glm", "json", "xml"] if fmt == "glm" else ["json", "glm", "xml"]
        for step in order:
            if step == "json":
                raws = _p_json(payload, index)
            elif step == "xml":
                raws = _p_xml_functions(payload)
            elif step == "glm":
                raws = _p_glm(payload)
            elif step == "invoke":
                raws = _p_invoke(payload)
            if raws:
                break
        if not raws and re.fullmatch(r"\s*[A-Za-z_][\w.\-]*\s*", payload):
            raws = [_Raw(payload.strip(), {}, block.raw)]  # GLM call without arguments
    elif kind in ("xml_function", "seed"):
        text = payload if kind == "seed" else "<function=" + payload + ("</function>" if block.closed else "")
        raws = _p_xml_functions(text)
    elif kind == "longcat":
        raws = _p_glm(payload, "longcat_arg_key", "longcat_arg_value")
    elif kind == "invoke" or kind == "dsml":
        raws = _p_invoke(payload)
    elif kind == "gemma4":
        raws = _p_gemma_call(payload, '<|"|>')
    elif kind == "function_gemma":
        raws = _p_gemma_call(payload, "<escape>")
    elif kind == "lfm2" or kind == "pythonic":
        raws = _p_pythonic(payload, index)
    elif kind == "kimi":
        raws = _p_kimi(payload)
    elif kind == "deepseek":
        raws = _p_deepseek(block.raw)
    elif kind == "mistral":
        raws = _p_mistral(payload, index)
    elif kind == "python_tag":
        raws = _p_python_tag(payload, index)
    elif kind in ("json_list", "json"):
        raws = _p_json(payload, index)
    elif kind == "fence_code":
        raws = _p_pythonic(payload, index) or _p_json(payload, index)
    elif kind == "harmony":
        raws = _p_harmony(block.header, payload)
    if not raws and kind not in ("harmony",):
        raws = _p_generic(payload, index)
    if block.closed:
        # The model wrote the closing marker, so it considered the call finished:
        # a JSON value that merely lacks its closing brackets is a slip, not a cut
        # off generation. Missing string data (an unterminated quote) still counts.
        for r in raws:
            if r.truncated and not r.string_truncated and r.name and r.arguments is not None:
                r.truncated = False
    else:
        cut = bool(block.inner_end) and block.inner_end not in payload
        for r in raws:
            if cut or r.open_tail:
                r.truncated = True
    return raws


def _looks_like_call(block: _Block, fmt: str | None, index: _ToolIndex | None) -> bool:
    try:
        raws = _parse_block(block, fmt, index)
    except Exception:  # pragma: no cover - parsers are defensive, but never break parsing
        return False
    return any(r.name for r in raws)


# --------------------------------------------------------------------------- #
# The segmenter
# --------------------------------------------------------------------------- #

_TEXT, _REASON, _TOOL, _TENT, _FENCEPEEK, _HHEAD, _HBODY, _SWALLOW_FENCE, _DONE = range(9)


def _prefix_set(markers: Iterable[str]) -> tuple[frozenset[str], int]:
    prefixes: set[str] = set()
    longest = 0
    for m in markers:
        longest = max(longest, len(m))
        for k in range(1, len(m)):
            prefixes.add(m[:k])
    return frozenset(prefixes), longest


class _MarkerSet:
    """A set of literal markers with leftmost-longest search and partial-suffix detection."""

    def __init__(self, markers: Iterable[str]) -> None:
        uniq = sorted(set(markers), key=len, reverse=True)
        self.markers = uniq
        self.regex = re.compile("|".join(re.escape(m) for m in uniq)) if uniq else None
        self.prefixes, self.longest = _prefix_set(uniq)
        self.extendable = frozenset(m for m in uniq if any(o != m and o.startswith(m) for o in uniq))

    def search(self, s: str, pos: int = 0) -> re.Match[str] | None:
        return self.regex.search(s, pos) if self.regex else None

    def partial_suffix(self, s: str) -> int:
        """Length of the longest suffix of ``s`` that is a proper prefix of a marker."""
        upto = min(self.longest - 1, len(s))
        for k in range(upto, 0, -1):
            if s[-k:] in self.prefixes:
                return k
        return 0


_REASON_OPENERS = {o: cs for o, cs in _REASONING_PAIRS}
_ALL_CLOSERS = tuple(dict.fromkeys(REASONING_CLOSERS))
_TOOL_OPENER_TEXTS = tuple(s.opener for s in _TOOL_OPENERS)
_HARMONY_BODY_END = ("<|end|>", "<|call|>", "<|return|>", "<|start|>", "<|channel|>")
_MAX_HEADER = 400   # a "<|channel|>…<|message|>" header longer than this is prose


def _text_marker_set(tools: bool, fenced: bool) -> _MarkerSet:
    markers: list[str] = [_FENCE, "\n"]
    if tools:
        markers += _TOOL_OPENER_TEXTS
    if not fenced:
        markers += list(_REASON_OPENERS) + list(_ALL_CLOSERS) + list(_HARMONY_TOKENS)
        markers += list(_DROP_TOKENS) + list(_TRUNCATE_TOKENS)
    return _MarkerSet(markers)


_TEXT_MARKERS = {(t, f): _text_marker_set(t, f) for t in (True, False) for f in (True, False)}
_HBODY_MARKERS = _MarkerSet(list(_HARMONY_BODY_END) + list(_DROP_TOKENS) + list(_TRUNCATE_TOKENS))
_HHEAD_MARKERS = _MarkerSet(["<|message|>", "<|start|>"])
_REASON_MARKER_CACHE: dict[tuple[tuple[str, ...], bool], _MarkerSet] = {}
_BRACKET_START_MARKERS = tuple(m for m in _TEXT_MARKERS[(True, False)].markers if m[0] in "{[")
_IDENT_RE = re.compile(r"[A-Za-z_][\w.\-]*")
_KEY_IDENT_RE = re.compile(r"[A-Za-z_][\w\-]*")


def _reason_markers(closers: tuple[str, ...], tools: bool) -> _MarkerSet:
    key = (closers, tools)
    ms = _REASON_MARKER_CACHE.get(key)
    if ms is None:
        markers = list(closers) + list(_DROP_TOKENS) + list(_TRUNCATE_TOKENS)
        if tools:
            markers += _TOOL_OPENER_TEXTS
        ms = _MarkerSet(markers)
        _REASON_MARKER_CACHE[key] = ms
    return ms


@dataclass
class _ToolState:
    spec: _OpenerSpec
    parent: int
    in_reasoning: bool
    raw_prefix: str
    payload: str = ""
    scanner: _BalanceScanner | None = None
    json_decided: bool = False
    json_end: int = -1
    search_from: int = 0
    fence_wrapped: bool = False


@dataclass
class _TentState:
    kind: str            # "json" | "array?" (undecided [ …) | "pythonic"
    fenced: bool
    prefix: str          # fence opener text (fenced) or ""
    dropped: str = ""    # separator swallowed before the value (given back when rejected)
    text: str = ""
    decided: bool = False
    scanner: _BalanceScanner = field(default_factory=_BalanceScanner)
    require_known: bool = False
    end: int = -1        # index just past the value once the scanner found it


_Event = tuple[str, Any]


class _Segmenter:
    """Incremental classifier: text / reasoning / hidden tool blocks (see module doc).

    Events: ``("text", s)``, ``("reasoning", s)``, ``("block", _Block)``,
    ``("reasoning_start", marker)``, ``("reasoning_end", marker)``,
    ``("orphan_close", marker)``.
    """

    def __init__(self, tool_format: str | None = None, starts_in_reasoning: bool = False,
                 index: _ToolIndex | None = None) -> None:
        self.fmt = tool_format
        self.index = index
        self.tools = index is None or not index.disabled
        self.buf = ""
        self.events: list[_Event] = []
        self.state = _TEXT
        self.reason_closers: tuple[str, ...] = ()
        self.at_start = True       # no visible text since the start / the last block
        self.line_start = True     # the next buffered char starts a line
        self.after_block = False   # the last thing emitted was a tool-call block
        self.in_fence = False      # inside an ordinary ``` code fence
        self.last_char = ""
        self.tool: _ToolState | None = None
        self.tent: _TentState | None = None
        self.h_header = ""
        self.h_body_kind = "text"
        self.h_in_tool = False
        self.h_payload = ""
        self.finished = False
        if starts_in_reasoning:
            if tool_format == "harmony":
                self.state = _HBODY
                self.h_header = "<|channel|>analysis"
                self.h_body_kind = "reasoning"
            else:
                self.state = _REASON
                self.reason_closers = _ALL_CLOSERS

    # -- public -------------------------------------------------------------
    def feed(self, delta: str) -> list[_Event]:
        if self.finished:
            return []
        if delta and self.state != _DONE:
            self.buf += delta
            self._run(final=False)
        return self._take()

    def finish(self) -> list[_Event]:
        if self.finished:
            return []
        self._run(final=True)
        self.finished = True
        return self._take()

    @property
    def in_tool(self) -> bool:
        return self.state in (_TOOL, _TENT) or (self.state == _HBODY and self.h_in_tool)

    # -- plumbing -------------------------------------------------------------
    def _take(self) -> list[_Event]:
        ev, self.events = self.events, []
        return ev

    def _emit(self, kind: str, text: str) -> None:
        if not text:
            return
        if kind == "text":
            if not text.isspace():
                self.at_start = False
                self.after_block = False
            self.line_start = text.endswith("\n")
        self.last_char = text[-1]
        self.events.append((kind, text))

    def _block(self, block: _Block) -> None:
        self.events.append(("block", block))
        self.at_start = True
        self.line_start = True
        self.after_block = True
        self.last_char = ""

    def _run(self, final: bool) -> None:
        handlers = {
            _TEXT: self._h_text, _REASON: self._h_reason, _TOOL: self._h_tool, _TENT: self._h_tent,
            _FENCEPEEK: self._h_fence, _HHEAD: self._h_hhead, _HBODY: self._h_hbody,
            _SWALLOW_FENCE: self._h_swallow_fence,
        }
        while self.state != _DONE:
            if not handlers[self.state](final):
                break
        if self.state == _DONE:
            self.buf = ""

    # -- TEXT -------------------------------------------------------------------
    def _h_text(self, final: bool) -> bool:
        buf = self.buf
        if not buf:
            return False
        if self.tools and not self.in_fence and (self.at_start or self.line_start):
            verdict = self._maybe_tentative(buf, final)
            if verdict is not None:
                return verdict
        ms = _TEXT_MARKERS[(self.tools, self.in_fence)]
        m = ms.search(buf)
        if m is None:
            keep = 0 if final else ms.partial_suffix(buf)
            self._emit("text", buf[: len(buf) - keep])
            self.buf = buf[len(buf) - keep:]
            return False
        marker = m.group()
        p = m.start()
        if not final and m.end() == len(buf) and marker in ms.extendable:
            self._emit("text", buf[:p])
            self.buf = buf[p:]
            return False
        if marker == "\n":
            self._emit("text", buf[: p + 1])
            self.buf = buf[p + 1:]
            return True
        self._emit("text", buf[:p])
        self.buf = buf[m.end():]
        if marker != _FENCE and self.last_char == "`":
            self._emit("text", marker)  # `<tool_call>` in inline code is just text
            return True
        return self._dispatch_text_marker(marker)

    def _maybe_tentative(self, buf: str, final: bool) -> bool | None:
        """Start a tentative JSON/pythonic block at the start of the answer or a line.

        Returns ``None`` when nothing applies, ``False`` to wait for more data,
        ``True`` after switching state."""
        ws = " \t\r\n" if self.at_start else " \t"
        i = 0
        n = len(buf)
        sep = 0   # end of a ";"/"," separating this value from the previous call
        while i < n and (buf[i] in ws or (self.after_block and buf[i] in ";,")):
            if buf[i] in ";,":
                sep = i + 1
            i += 1
        if i == n:
            return None if final else False
        c = buf[i]
        if c != "{" and not (c == "[" and self.at_start):
            return None
        if any(buf.startswith(mk, i) for mk in _BRACKET_START_MARKERS):
            return None
        if not final and buf[i:] in _TEXT_MARKERS[(True, False)].prefixes:
            return False
        self._emit("text", buf[sep:i])
        self.buf = buf[i:]
        self.tent = _TentState(kind="json" if c == "{" else "array?", fenced=False, prefix="",
                               dropped=buf[:sep])
        self.state = _TENT
        return True

    def _dispatch_text_marker(self, marker: str) -> bool:
        if marker in _REASON_OPENERS:
            self.state = _REASON
            self.reason_closers = _REASON_OPENERS[marker]
            self.events.append(("reasoning_start", marker))
            return True
        if marker in _OPENER_BY_TEXT:
            self._start_tool(_OPENER_BY_TEXT[marker], _TEXT, False)
            return True
        if marker == _FENCE:
            if self.in_fence:
                self._emit("text", _FENCE)
                self.in_fence = False
                return True
            self.state = _FENCEPEEK
            self.buf = _FENCE + self.buf
            return True
        if marker in _ALL_CLOSERS:
            self.events.append(("orphan_close", marker))
            return True
        if marker in ("<|start|>", "<|channel|>"):
            self.state = _HHEAD
            self.h_header = marker if marker == "<|channel|>" else ""
            return True
        if marker in _TRUNCATE_TOKENS:
            self.state = _DONE
            self.buf = ""
            return False
        return True  # stray harmony / end-of-turn tokens carry no content

    # -- REASONING ----------------------------------------------------------------
    def _h_reason(self, final: bool) -> bool:
        buf = self.buf
        if not buf:
            return False
        ms = _reason_markers(self.reason_closers, self.tools)
        m = ms.search(buf)
        if m is None:
            keep = 0 if final else ms.partial_suffix(buf)
            self._emit("reasoning", buf[: len(buf) - keep])
            self.buf = buf[len(buf) - keep:]
            return False
        marker = m.group()
        p = m.start()
        if not final and m.end() == len(buf) and marker in ms.extendable:
            self._emit("reasoning", buf[:p])
            self.buf = buf[p:]
            return False
        self._emit("reasoning", buf[:p])
        self.buf = buf[m.end():]
        if marker in self.reason_closers:
            self.state = _TEXT
            self.events.append(("reasoning_end", marker))
            self.at_start = True
            self.line_start = True
            self.last_char = ""
            return True
        if self.last_char == "`":
            self._emit("reasoning", marker)
            return True
        if marker in _OPENER_BY_TEXT:
            self._start_tool(_OPENER_BY_TEXT[marker], _REASON, True)
            return True
        if marker in _TRUNCATE_TOKENS:
            self.state = _DONE
            self.buf = ""
            return False
        return True  # end-of-turn token

    # -- TOOL ---------------------------------------------------------------------
    def _start_tool(self, spec: _OpenerSpec, parent: int, in_reasoning: bool, *, prefix: str | None = None,
                    fence_wrapped: bool = False) -> None:
        self.tool = _ToolState(spec=spec, parent=parent, in_reasoning=in_reasoning,
                               raw_prefix=spec.opener if prefix is None else prefix, fence_wrapped=fence_wrapped)
        self.state = _TOOL

    def _h_tool(self, final: bool) -> bool:
        st = self.tool
        assert st is not None
        if self.buf:
            st.payload += self.buf
            self.buf = ""
        payload = st.payload
        spec = st.spec
        if not st.json_decided:
            stripped = payload.lstrip()
            if stripped:
                st.json_decided = True
                if spec.json_aware and stripped[0] in "{[":
                    st.scanner = _BalanceScanner()
                    st.scanner.pos = len(payload) - len(stripped)
            elif not final:
                return False
        search_from = st.search_from
        if st.scanner is not None:
            if st.json_end < 0:
                end = st.scanner.scan(payload)
                if end < 0:
                    return self._tool_final(st) if final else False
                st.json_end = end
            search_from = max(search_from, st.json_end)
        best: tuple[int, str] | None = None
        for closer in spec.closers:
            q = payload.find(closer, search_from)
            if q >= 0 and (best is None or q < best[0]):
                best = (q, closer)
        if best is not None:
            q, closer = best
            self._close_tool(st, payload[:q], closer, payload[q + len(closer):])
            return True
        if final:
            return self._tool_final(st)
        longest = max((len(c) for c in spec.closers), default=1)
        st.search_from = max(search_from, len(payload) - longest + 1, 0)
        return False

    def _close_tool(self, st: _ToolState, body: str, closer: str, rest: str) -> None:
        block = _Block(kind=st.spec.kind, raw=st.raw_prefix + body + closer, payload=body, closed=True,
                       in_reasoning=st.in_reasoning, inner_end=st.spec.inner_end)
        self._finish_tool(st, block, rest)

    def _finish_tool(self, st: _ToolState, block: _Block, rest: str) -> None:
        self.tool = None
        self.buf = rest + self.buf
        if st.parent == _REASON:
            self.state = _REASON
            self.events.append(("block", block))
        else:
            self._block(block)
            self.state = _SWALLOW_FENCE if st.fence_wrapped else _TEXT

    def _tool_final(self, st: _ToolState) -> bool:
        payload = st.payload
        spec = st.spec
        if st.json_end >= 0:
            # the JSON value is complete but no closer followed: the block ends with the value
            body, rest = payload[:st.json_end], payload[st.json_end:]
            if spec.to_end:
                body, rest = payload, ""
            block = _Block(kind=spec.kind, raw=st.raw_prefix + body, payload=body, closed=spec.to_end,
                           in_reasoning=st.in_reasoning, inner_end=spec.inner_end)
            self._finish_tool(st, block, rest)
            return True
        for closer in spec.closers:  # the value never closed, but a closer exists: trust it
            q = payload.find(closer)
            if q >= 0:
                self._close_tool(st, payload[:q], closer, payload[q + len(closer):])
                return True
        block = _Block(kind=spec.kind, raw=st.raw_prefix + payload, payload=payload, closed=spec.to_end,
                       in_reasoning=st.in_reasoning, inner_end=spec.inner_end)
        if spec.to_end or _looks_like_call(block, self.fmt, self.index):
            self._finish_tool(st, block, "")
            return True
        # false alarm (e.g. "<tool_call>" mentioned in prose): give the markup back as text
        self.tool = None
        self.state = st.parent
        self._emit("reasoning" if st.parent == _REASON else "text", st.raw_prefix)
        if st.fence_wrapped:
            self.in_fence = True
        self.buf = payload + self.buf
        return True

    def _h_swallow_fence(self, final: bool) -> bool:
        buf = self.buf
        stripped = buf.lstrip()
        if not stripped:
            if final:
                self.state = _TEXT
                return True
            return False
        if stripped.startswith(_FENCE):
            self.buf = stripped[len(_FENCE):]
        elif _FENCE.startswith(stripped) and not final:
            return False
        self.state = _TEXT
        return True

    # -- FENCE (```lang + first line peek) ------------------------------------------
    def _h_fence(self, final: bool) -> bool:
        buf = self.buf  # starts with ```
        nl = buf.find("\n", len(_FENCE))
        if nl < 0:
            if final or len(buf) > 48:
                self._fence_as_text(len(_FENCE))
                return True
            return False
        lang = buf[len(_FENCE):nl].strip()
        lang_l = lang.lower()
        opener_line = buf[: nl + 1]
        body = buf[nl + 1:]
        body_s = body.lstrip()
        lead = body[: len(body) - len(body_s)]
        if not self.tools or " " in lang or len(lang) > 24:
            self._fence_as_text(nl + 1)
            return True
        if lang_l in _TOOL_FENCE_LANGS:
            self.buf = body
            self._start_tool(_OpenerSpec(opener_line, "fence_code", (_FENCE,)), _TEXT, False)
            return True
        if not body_s:
            if final:
                self._fence_as_text(nl + 1)
                return True
            return False
        for opener in _TOOL_OPENER_TEXTS:  # a tool call wrapped in an ordinary fence
            if body_s.startswith(opener):
                self.buf = body_s[len(opener):]
                self._start_tool(_OPENER_BY_TEXT[opener], _TEXT, False, prefix=opener_line + lead + opener,
                                 fence_wrapped=True)
                return True
            if opener.startswith(body_s) and not final:
                return False
        c = body_s[0]
        if c in "{[" and (lang_l in _JSON_FENCE_LANGS or lang_l == ""):
            self.buf = body_s
            self.tent = _TentState(kind="json" if c == "{" else "array?", fenced=True, prefix=opener_line + lead)
            self.state = _TENT
            return True
        if lang_l in _PY_FENCE_LANGS and self.index is not None and (c.isalpha() or c == "_"):
            m = _IDENT_RE.match(body_s)
            if m:
                after = body_s[m.end():].lstrip(" \t")
                if not after and not final:
                    return False
                if after.startswith("(") and self.index.resolve(m.group())[1]:
                    self.buf = body_s
                    self.tent = _TentState(kind="pythonic", fenced=True, prefix=opener_line + lead, decided=True,
                                           require_known=True)
                    self.state = _TENT
                    return True
        self._fence_as_text(nl + 1)
        return True

    def _fence_as_text(self, upto: int) -> None:
        self._emit("text", self.buf[:upto])
        self.buf = self.buf[upto:]
        self.in_fence = True
        self.state = _TEXT

    # -- TENTATIVE (bare / fenced JSON or pythonic list) ------------------------------
    def _h_tent(self, final: bool) -> bool:
        tt = self.tent
        assert tt is not None
        if self.buf:
            tt.text += self.buf
            self.buf = ""
        text = tt.text
        if not tt.decided:
            verdict = self._tent_decide(tt, final)
            if verdict is None:
                return False
            if verdict is False:
                return self._tent_reject(tt)
            tt.decided = True
        if tt.end < 0:
            tt.end = tt.scanner.scan(text)
        end = tt.end
        if end < 0:
            if not final:
                return False
            return self._tent_resolve(tt, len(text), truncated=True)
        if tt.fenced:
            q = text.find(_FENCE, end)
            if q < 0:
                if not final:
                    return False
                return self._tent_resolve(tt, len(text), truncated=False)
            return self._tent_resolve(tt, q, truncated=False, fence_end=q + len(_FENCE))
        return self._tent_resolve(tt, end, truncated=False)

    def _tent_decide(self, tt: _TentState, final: bool) -> bool | None:
        """Decide from the first key (or callee) whether this can be a call.
        ``None`` = need more data. Depends only on the text, never on chunking:
        ``tt.kind`` is only narrowed once the verdict is positive, otherwise the
        next chunk would restart the scan with the wrong expectations."""
        text = tt.text
        n = len(text)
        i = 1
        kind = tt.kind
        while i < n and text[i] in _WS:
            i += 1
        if kind == "array?":
            if i >= n:
                return False if final else None
            if text[i] == "{":
                kind = "json"
                i += 1
                while i < n and text[i] in _WS:
                    i += 1
            elif text[i].isalpha() or text[i] == "_":
                m = _IDENT_RE.match(text, i)
                if m is None:   # a non-ASCII word: "[łaty]" is prose, not a call
                    return False
                j = m.end()
                while j < n and text[j] in " \t":
                    j += 1
                if j >= n:
                    return False if final else None
                if text[j] != "(":
                    return False
                tt.kind = "pythonic"
                return True
            else:
                return False
        if i >= n:
            return False if final else None
        c = text[i]
        if c in "\"'":
            j = text.find(c, i + 1)
            if j < 0:
                return False if final or n - i > 80 else None
            key = text[i + 1:j]
        else:
            m = _KEY_IDENT_RE.match(text, i)
            if not m:
                return False
            if m.end() >= n:
                return False if final else None
            key = m.group()
        if key.lower() not in _TOOLISH_KEYS:
            return False
        tt.kind = kind
        return True

    def _tent_reject(self, tt: _TentState) -> bool:
        self.tent = None
        self.state = _TEXT
        if tt.fenced:
            self._emit("text", tt.dropped + tt.prefix)
            self.in_fence = True
            self.buf = tt.text + self.buf
        else:
            self._emit("text", tt.dropped + tt.text[:1])
            self.buf = tt.text[1:] + self.buf
        return True

    def _tent_resolve(self, tt: _TentState, end: int, *, truncated: bool, fence_end: int | None = None) -> bool:
        text = tt.text
        body = text[:end]
        if tt.kind == "pythonic":
            raws = _p_pythonic(body, self.index, require_known=tt.require_known)
            ok = bool(raws) and all(r.name for r in raws)
            if ok and not tt.require_known:
                if self.index is not None and self.index.specs:
                    ok = self.fmt == "pythonic" or all(self.index.resolve(r.name or "")[1] for r in raws)
                else:
                    ok = self.fmt in ("pythonic", "llama3")
        else:
            vals = [res for _, _, res in _iter_json_values(body)]
            ok = bool(vals)
            for res in vals:
                if not _raws_from_json(res.value, body, strict=True, index=self.index, truncated=res.truncated):
                    ok = False
                    break
        if not ok:
            return self._tent_reject(tt)
        self.tent = None
        stop = fence_end if fence_end is not None else end
        block = _Block(kind="pythonic" if tt.kind == "pythonic" else "json", raw=tt.prefix + text[:stop],
                       payload=body, closed=not truncated)
        self.buf = text[stop:] + self.buf
        self.state = _TEXT
        self._block(block)
        return True

    # -- HARMONY -------------------------------------------------------------------------
    def _h_hhead(self, final: bool) -> bool:
        buf = self.buf
        q = buf.find("<|message|>")
        if q < 0:
            restart = buf.find("<|start|>")
            if restart >= 0:
                self.h_header = ""
                self.buf = buf[restart + len("<|start|>"):]
                return True
            if final:  # an unfinished header at the very end carries nothing
                self.h_header = ""
                self.buf = ""
                self.state = _TEXT
                return False
            keep = _HHEAD_MARKERS.partial_suffix(buf)
            self.h_header += buf[: len(buf) - keep]
            self.buf = buf[len(buf) - keep:]
            if len(self.h_header) > _MAX_HEADER:  # not a harmony header after all
                self._reject_header("")
                return True
            return False
        if len(self.h_header) + q > _MAX_HEADER:
            # Too long to be a channel header — the same verdict the streaming
            # branch above reaches, so chunking cannot change the outcome. The
            # trailing "<|message|>" is dropped by the text handler.
            self._reject_header(buf[:q])
            self.buf = buf[q:]
            return True
        header = self.h_header + buf[:q]
        self.buf = buf[q + len("<|message|>"):]
        self.h_header = header
        chan = re.search(r"<\|channel\|>\s*([\w\-]+)", header)
        channel = chan.group(1).lower() if chan else "final"
        recipient = re.search(r"\bto=([^\s<]+)", header)
        if recipient and not recipient.group(1).startswith("assistant"):
            self.h_in_tool = True
            self.h_payload = ""
            self.h_body_kind = "tool"
        else:
            self.h_in_tool = False
            self.h_body_kind = "reasoning" if channel == "analysis" else "text"
        self.state = _HBODY
        return True

    def _reject_header(self, tail: str) -> None:
        """Give an over-long pseudo-header back as text (without its control token)."""
        header = self.h_header
        if header.startswith("<|channel|>"):
            header = header[len("<|channel|>"):]
        self._emit("text", header + tail)
        self.h_header = ""
        self.state = _TEXT

    def _h_hbody(self, final: bool) -> bool:
        buf = self.buf
        ms = _HBODY_MARKERS
        m = ms.search(buf) if buf else None
        if m is None:
            keep = 0 if final else ms.partial_suffix(buf)
            chunk = buf[: len(buf) - keep]
            self.buf = buf[len(buf) - keep:]
            if self.h_in_tool:
                self.h_payload += chunk
            else:
                self._emit(self.h_body_kind, chunk)
            if final:
                if self.h_in_tool:
                    self._harmony_tool_done(closed=False)
                self.state = _TEXT
            return False
        marker = m.group()
        p = m.start()
        if not final and m.end() == len(buf) and marker in ms.extendable:
            return False
        if self.h_in_tool:
            self.h_payload += buf[:p]
        else:
            self._emit(self.h_body_kind, buf[:p])
        self.buf = buf[m.end():]
        if marker in _DROP_TOKENS:
            return True
        if marker in _TRUNCATE_TOKENS:
            if self.h_in_tool:
                self._harmony_tool_done(closed=False)
            self.state = _DONE
            return False
        if self.h_in_tool:
            self._harmony_tool_done(closed=marker in ("<|call|>", "<|end|>", "<|return|>"))
        if marker in ("<|start|>", "<|channel|>"):
            self.state = _HHEAD
            self.h_header = marker if marker == "<|channel|>" else ""
        else:
            self.state = _TEXT
            self.at_start = True
            self.line_start = True
        return True

    def _harmony_tool_done(self, closed: bool) -> None:
        block = _Block(kind="harmony", raw=f"{self.h_header}<|message|>{self.h_payload}", payload=self.h_payload,
                       closed=closed, header=self.h_header)
        self.h_in_tool = False
        self.h_payload = ""
        self._block(block)


# --------------------------------------------------------------------------- #
# Public parsing API
# --------------------------------------------------------------------------- #


class StreamSplitter:
    """Split a streamed completion into visible ``text`` and ``reasoning`` pieces.

    Tool-call markup (including markers split across chunks) never reaches the
    visible stream. ``feed``/``finish`` return ``[(kind, text), …]`` with kind in
    ``"text" | "reasoning"``. Concatenating all ``text`` pieces gives exactly
    what ``parse_output`` returns as ``content`` (modulo surrounding
    whitespace), regardless of how the output was chunked — except when the
    final parse reports ``repaired_after_stream`` (orphan ``</think>``, prose
    fallback calls), which cannot be undone once text has been streamed; see the
    module docstring. ``known_tools`` is optional and only sharpens the
    detection of bare-JSON calls.
    """

    def __init__(self, tool_format: str | None = None, starts_in_reasoning: bool = False, *,
                 known_tools: Any = None) -> None:
        self._seg = _Segmenter(tool_format, starts_in_reasoning, _ToolIndex.build(known_tools))
        self._text_started = False
        self._reason_started = False
        self._blocks: list[_Block] = []

    def feed(self, delta: str) -> list[tuple[str, str]]:
        return self._visible(self._seg.feed(delta))

    def finish(self) -> list[tuple[str, str]]:
        return self._visible(self._seg.finish())

    @property
    def in_tool_call(self) -> bool:
        """True while the model is (probably) writing tool-call markup."""
        return self._seg.in_tool

    @property
    def tool_blocks_seen(self) -> int:
        return len(self._blocks)

    def _visible(self, events: list[_Event]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for kind, value in events:
            if kind == "block":
                self._blocks.append(value)
                continue
            if kind == "reasoning_start" and self._reason_started:
                # a second reasoning block: keep it apart, exactly like parse_output
                out.append(("reasoning", "\n\n"))
                continue
            if kind not in ("text", "reasoning"):
                continue
            text = value
            if kind == "text" and not self._text_started:
                text = text.lstrip()
                if not text:
                    continue
                self._text_started = True
            if kind == "reasoning" and not self._reason_started:
                text = text.lstrip()
                if not text:
                    continue
                self._reason_started = True
            if out and out[-1][0] == kind:
                out[-1] = (kind, out[-1][1] + text)
            else:
                out.append((kind, text))
        return out


_ACTION_RE = re.compile(r"Action:\s*`?([A-Za-z_][\w.\-]*)`?\s*\n\s*Action Input:\s*", re.S)
_FENCE_LINE_RE = re.compile(r"^[ \t]*```[^\n]*$", re.M)


def parse_output(
    text: str,
    *,
    tool_format: str | None = None,
    known_tools: Any = None,
    starts_in_reasoning: bool = False,
) -> ParsedOutput:
    """Split a complete model output into content, reasoning and tool calls.

    ``known_tools`` may be tool names, OpenAI tool schemas or ``Tool`` objects;
    with schemas, names are fuzzy-corrected and argument types coerced.
    ``errors`` lists unusable call attempts in English, ready to be sent back to
    the model.
    """
    index = _ToolIndex.build(known_tools)
    seg = _Segmenter(tool_format, starts_in_reasoning, index)
    events = seg.feed(text or "") + seg.finish()
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    blocks: list[_Block] = []
    orphan_used = False
    repaired = False          # a repair a stream could not have made (see ParsedOutput)
    for kind, value in events:
        if kind == "text":
            content_parts.append(value)
        elif kind == "reasoning":
            reasoning_parts.append(value)
        elif kind == "reasoning_start":
            if reasoning_parts and "".join(reasoning_parts).strip():
                reasoning_parts.append("\n\n")
        elif kind == "block":
            blocks.append(value)
        elif kind == "orphan_close" and not orphan_used:
            orphan_used = True
            if not "".join(reasoning_parts).strip():
                if "".join(content_parts).strip():
                    repaired = True
                reasoning_parts = content_parts
                content_parts = []

    calls: list[ToolCall] = []
    errors: list[str] = []
    used_ids: set[str] = set()
    if index is not None and index.disabled:
        blocks = []  # no tools were offered: nothing can be a call
    outside = [b for b in blocks if not b.in_reasoning]
    inside = [b for b in blocks if b.in_reasoning]
    for group in (outside, inside):
        if not group:
            continue
        for block in group:
            raws = _parse_block(block, tool_format, index)
            if not raws:
                errors.append(
                    f"Could not parse the {_KIND_LABEL.get(block.kind, 'tool call')} ({_snippet(block.raw)}). "
                    f"Use exactly this format:\n{format_hint(tool_format)}"
                )
                continue
            _finalize(raws, index, tool_format, calls, errors, used_ids)
        break  # calls drafted inside reasoning are used only when there are none outside

    content = "".join(content_parts)
    if not blocks and index is not None and index.specs:
        content, fallback = _fallback_calls(content, tool_format, index)
        if fallback:
            repaired = True
            _finalize(fallback, index, tool_format, calls, errors, used_ids)
    if not calls and index is not None and index.specs:
        content, bare = _bare_tool_tags(content, index)
        if bare:
            errors.append(
                f"You wrote the tool name as an empty tag (<{bare[0]}>) instead of calling it. "
                f"Call {bare[0]} using exactly this format, with JSON arguments:\n"
                f"{format_hint(tool_format)}"
            )
    return ParsedOutput(content=content.strip(), reasoning="".join(reasoning_parts).strip(), tool_calls=calls,
                        errors=errors, repaired_after_stream=repaired)


_BARE_TAG_RE = re.compile(r"<\s*/?\s*([A-Za-z_][\w.:-]*)\s*/?\s*>")


def _bare_tool_tags(content: str, index: _ToolIndex) -> tuple[str, list[str]]:
    """``(content without them, tool names)`` for tags like ``<todo_write>``.

    A small model that has understood *which* tool it wants but not how to call
    it writes the name as an empty tag and nothing else — the whole answer to
    "make me a page about bananas" was ``<todo_write>`` repeated. That is an
    attempted call, not prose: reported as an error the model can act on, it
    gets one corrected retry instead of the user getting markup as an answer.

    Only an exact tool name counts, and only outside code fences, so an HTML
    page full of ``<div>`` is untouched.
    """
    fenced = _fenced_spans(content)
    names: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _BARE_TAG_RE.finditer(content):
        if any(start <= m.start() < end for start, end in fenced):
            continue
        hit = index.exact(m.group(1))
        if hit is None:
            continue
        names.append(hit)
        spans.append(m.span())
    if not names:
        return content, []
    out = content
    for start, end in reversed(spans):
        out = out[:start] + out[end:]
    return out, names


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Spans of ``` … ``` code fences (an unclosed fence runs to the end)."""
    marks = list(_FENCE_LINE_RE.finditer(text))
    spans = [(marks[i].start(), marks[i + 1].end()) for i in range(0, len(marks) - 1, 2)]
    if len(marks) % 2:
        spans.append((marks[-1].start(), len(text)))
    return spans


def _fallback_calls(content: str, fmt: str | None, index: _ToolIndex) -> tuple[str, list[_Raw]]:
    """Last resort when no tool markup was found and the tools are known: JSON
    calls embedded in prose, ReAct ``Action:`` lines, a bare ``tool(arg=…)``.

    Code fences are off limits: a call-shaped object inside ``` … ``` is a code
    example (and the segmenter has already had its chance at fenced JSON), so
    rescuing it would both mangle the answer and disagree with the stream.
    """
    if not content.strip():
        return content, []
    fenced = _fenced_spans(content)

    def in_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fenced)

    # ReAct: "Action: name\nAction Input: {json}"
    m = _ACTION_RE.search(content)
    if m and not in_fence(m.start()):
        try:
            res = _parse_json_at(content, m.end())
        except _JsonRepairError:
            res = None
        if res is not None and isinstance(res.value, (dict, str)):
            name = m.group(1)
            if index.resolve(name)[1]:
                raw = content[m.start():res.end]
                return (content[: m.start()] + content[res.end:],
                        [_mk_raw(name, res.value, raw, None, res.truncated, res.string_truncated)])
    # JSON objects anywhere whose shape is a call (names must be known when tools are known)
    found: list[_Raw] = []
    spans: list[tuple[int, int]] = []
    for start, end, res in _iter_json_values(content):
        if in_fence(start):
            continue
        got = _raws_from_json(res.value, content[start:end], strict=True, index=index, truncated=res.truncated)
        if not got:
            continue
        if not all(index.resolve(r.name or "")[1] for r in got):
            continue
        found.extend(got)
        spans.append((start, end))
    if found:
        out = content
        for start, end in reversed(spans):
            out = out[:start] + out[end:]
        return out, found
    # the whole answer is a single pythonic call of a known tool: read_file(path="x")
    stripped = content.strip()
    if re.fullmatch(r"\[?[A-Za-z_][\w.]*\(.*\)\]?", stripped, re.S):
        raws = _p_pythonic(stripped, index, require_known=True)
        if raws:
            return "", raws
    return content, []


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #


def _format_from_template(tpl: str) -> str | None:
    if "<|channel|>" in tpl and ("<|start|>" in tpl or "analysis" in tpl):
        return "harmony"
    if "<minimax:tool_call>" in tpl or ("]~b]" in tpl and "<invoke name=" in tpl):
        return "minimax_m2"
    if "<|tool_call>" in tpl and "<tool_call|>" in tpl:
        return "gemma4"
    if "<start_function_call>" in tpl:
        return "function_gemma"
    if "<longcat_tool_call>" in tpl:
        return "longcat"
    if "<seed:tool_call>" in tpl:
        return "seed_oss"
    if "<arg_key>" in tpl:
        return "glm"
    if "<|tool_list_start|>" in tpl or "<|tool_call_start|>" in tpl:
        return "pythonic"
    if "<|tool_calls_section_begin|>" in tpl:
        return "kimi_k2"
    if ("dsml_token" in tpl and "DSML" in tpl) or "<\uff5ctool\u2581calls\u2581begin\uff5c>" in tpl:
        return "deepseek"
    if "[TOOL_CALLS]" in tpl:
        return "mistral"
    if "<function=" in tpl and ("<parameter=" in tpl or "<tool_call>" in tpl):
        return "qwen3_coder"
    if "<|python_tag|>" in tpl or ('"parameters": ' in tpl and "<|start_header_id|>" in tpl):
        return "llama3"
    if "<|START_ACTION|>" in tpl:
        return "command_r"
    if "<TOOLCALL>" in tpl:
        return "nemotron"
    if "<|tool_call|>" in tpl:
        return "granite"
    if "<tool_call>" in tpl:
        return "hermes"
    if "<|start_header_id|>" in tpl and "ipython" in tpl:
        return "llama3"
    return None


_NAME_RULES: tuple[tuple[str, str], ...] = (
    ("gpt-oss", "harmony"), ("gpt_oss", "harmony"), ("gptoss", "harmony"),
    ("qwen3-coder", "qwen3_coder"), ("qwen3_coder", "qwen3_coder"), ("qwen3coder", "qwen3_coder"),
    ("qwen3.5", "qwen3_coder"), ("qwen3_5", "qwen3_coder"), ("qwen35", "qwen3_coder"),
    ("seed-oss", "seed_oss"), ("seed_oss", "seed_oss"),
    ("minimax", "minimax_m2"), ("kimi", "kimi_k2"), ("glm", "glm"), ("chatglm", "glm"),
    ("deepseek", "deepseek"),
    ("functiongemma", "function_gemma"), ("function-gemma", "function_gemma"),
    ("gemma-4", "gemma4"), ("gemma4", "gemma4"), ("gemma", "hermes"),
    ("devstral", "mistral"), ("magistral", "mistral"), ("ministral", "mistral"), ("mistral", "mistral"),
    ("mixtral", "mistral"), ("codestral", "mistral"),
    ("llama-4", "pythonic"), ("llama4", "pythonic"), ("lfm2", "pythonic"), ("lfm-2", "pythonic"),
    ("llama-3", "llama3"), ("llama3", "llama3"), ("llama_3", "llama3"), ("llama", "llama3"),
    ("granite", "granite"), ("command-r", "command_r"), ("command-a", "command_r"), ("c4ai", "command_r"),
    ("hermes", "hermes"), ("qwen", "hermes"),
)


def _format_from_name(*names: str | None) -> str | None:
    hay = " ".join(n.lower() for n in names if n)
    if not hay:
        return None
    for needle, fmt in _NAME_RULES:
        if needle in hay:
            return fmt
    return None


def detect_tool_format(model: "LocalModel | None" = None, template: str | None = None) -> str:
    """Name of the tool-call convention the model will most likely use.

    Template markers win (they are what the model was trained on); a template
    that cannot render tools at all means we will *prompt* for Hermes-style
    ``<tool_call>`` JSON, so ``"hermes"`` is returned. Without a template the
    model's architecture/name decides; the final default is ``"hermes"``.
    """
    tpl = template
    if not tpl and model is not None:
        tpl = getattr(getattr(model, "meta", None), "chat_template", None)
    if tpl:
        fmt = _format_from_template(tpl)
        if fmt:
            return fmt
        from localcode.runtime.templates import template_variables

        variables = template_variables(tpl)
        if variables and "tools" not in variables:
            return "hermes"
    if model is not None:
        meta = getattr(model, "meta", None)
        fmt = _format_from_name(getattr(meta, "arch", None), getattr(model, "key", None),
                                getattr(model, "display_name", None), getattr(model, "repo_id", None))
        if fmt:
            return fmt
    return "hermes"


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

_FORMAT_HINTS: dict[str, str] = {
    "hermes": '<tool_call>\n{"name": "tool_name", "arguments": {"param": "value"}}\n</tool_call>',
    "json": '<tool_call>\n{"name": "tool_name", "arguments": {"param": "value"}}\n</tool_call>',
    "qwen3_coder": "<tool_call>\n<function=tool_name>\n<parameter=param>\nvalue\n</parameter>\n</function>\n</tool_call>",
    "seed_oss": "<seed:tool_call>\n<function=tool_name>\n<parameter=param>value</parameter>\n</function>\n</seed:tool_call>",
    "glm": "<tool_call>tool_name\n<arg_key>param</arg_key>\n<arg_value>value</arg_value>\n</tool_call>",
    "longcat": "<longcat_tool_call>tool_name\n<longcat_arg_key>param</longcat_arg_key>\n"
               "<longcat_arg_value>value</longcat_arg_value>\n</longcat_tool_call>",
    "llama3": '{"name": "tool_name", "parameters": {"param": "value"}}',
    "mistral": '[TOOL_CALLS]tool_name[ARGS]{"param": "value"}',
    "harmony": '<|channel|>commentary to=functions.tool_name <|constrain|>json<|message|>{"param": "value"}<|call|>',
    "deepseek": "<\uff5ctool\u2581calls\u2581begin\uff5c><\uff5ctool\u2581call\u2581begin\uff5c>tool_name"
                '<\uff5ctool\u2581sep\uff5c>{"param": "value"}<\uff5ctool\u2581call\u2581end\uff5c>'
                "<\uff5ctool\u2581calls\u2581end\uff5c>",
    "kimi_k2": "<|tool_calls_section_begin|><|tool_call_begin|>functions.tool_name:0<|tool_call_argument_begin|>"
               '{"param": "value"}<|tool_call_end|><|tool_calls_section_end|>',
    "minimax_m2": '<minimax:tool_call>\n<invoke name="tool_name">\n<parameter name="param">value</parameter>\n'
                  "</invoke>\n</minimax:tool_call>",
    "gemma4": '<|tool_call>call:tool_name{param:<|"|>value<|"|>}<tool_call|>',
    "function_gemma": "<start_function_call>call:tool_name{param:<escape>value<escape>}<end_function_call>",
    "pythonic": '<|tool_call_start|>[tool_name(param="value")]<|tool_call_end|>',
    "granite": '<|tool_call|>[{"name": "tool_name", "arguments": {"param": "value"}}]',
    "nemotron": '<TOOLCALL>[{"name": "tool_name", "arguments": {"param": "value"}}]</TOOLCALL>',
    "command_r": '<|START_ACTION|>[{"tool_name": "tool_name", "parameters": {"param": "value"}}]<|END_ACTION|>',
}


def format_hint(tool_format: str | None) -> str:
    """A minimal, correct example call in the given convention (for error feedback)."""
    return _FORMAT_HINTS.get(tool_format or "hermes", _FORMAT_HINTS["hermes"])


def _arg_text(value: Any) -> str:
    """One argument value as the textual (XML-ish) conventions write it."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _json_args(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)


def _gemma_args(arguments: dict[str, Any], delim: str) -> str:
    parts = [f"{k}:{delim}{v}{delim}" if isinstance(v, str) else f"{k}:{json.dumps(v, ensure_ascii=False)}"
             for k, v in arguments.items()]
    return "{" + ",".join(parts) + "}"


def render_tool_call(tool_format: str | None, name: str, arguments: dict[str, Any], index: int = 0) -> str:
    """Serialise one tool call the way the model itself would have written it.

    Needed when a chat template renders the tool *schemas* but drops
    ``tool_calls`` from the history (Granite, Phi-4 …): the formatter then writes
    the call into the assistant message text, and it must use the convention the
    model was trained on — otherwise the history teaches it the wrong format.
    ``index`` disambiguates ids in the conventions that carry one (Kimi).
    """
    fmt = tool_format or "hermes"
    args = arguments if isinstance(arguments, dict) else {}
    if fmt == "qwen3_coder" or fmt == "seed_oss":
        body = "".join(f"<parameter={k}>\n{_arg_text(v)}\n</parameter>\n" for k, v in args.items())
        inner = f"<function={name}>\n{body}</function>"
        return f"<seed:tool_call>\n{inner}\n</seed:tool_call>" if fmt == "seed_oss" else \
            f"<tool_call>\n{inner}\n</tool_call>"
    if fmt in ("glm", "longcat"):
        key, value = ("arg_key", "arg_value") if fmt == "glm" else ("longcat_arg_key", "longcat_arg_value")
        body = "".join(f"<{key}>{k}</{key}>\n<{value}>{_arg_text(v)}</{value}>\n" for k, v in args.items())
        tag = "tool_call" if fmt == "glm" else "longcat_tool_call"
        return f"<{tag}>{name}\n{body}</{tag}>"
    if fmt == "minimax_m2":
        body = "".join(f'<parameter name="{k}">{_arg_text(v)}</parameter>\n' for k, v in args.items())
        return f'<minimax:tool_call>\n<invoke name="{name}">\n{body}</invoke>\n</minimax:tool_call>'
    if fmt == "llama3":
        return json.dumps({"name": name, "parameters": args}, ensure_ascii=False)
    if fmt == "mistral":
        return f"[TOOL_CALLS]{name}[ARGS]{_json_args(args)}"
    if fmt == "harmony":
        return (f"<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>"
                f"{_json_args(args)}<|call|>")
    if fmt == "deepseek":
        return (f"{_DS_CALLS_BEGIN}{_DS_CALL_BEGIN}{name}{_DS_SEP}{_json_args(args)}"
                f"{_DS_CALL_END}{_DS_CALLS_END}")
    if fmt == "kimi_k2":
        return (f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{index}"
                f"<|tool_call_argument_begin|>{_json_args(args)}<|tool_call_end|><|tool_calls_section_end|>")
    if fmt == "gemma4":
        return f"<|tool_call>call:{name}{_gemma_args(args, '<|\"|>')}<tool_call|>"
    if fmt == "function_gemma":
        return f"<start_function_call>call:{name}{_gemma_args(args, '<escape>')}<end_function_call>"
    if fmt == "pythonic":
        inner = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"<|tool_call_start|>[{name}({inner})]<|tool_call_end|>"
    if fmt in ("granite", "nemotron", "command_r"):
        payload = json.dumps([{"name": name, "arguments": args}], ensure_ascii=False)
        if fmt == "granite":
            return f"<|tool_call|>{payload}"
        if fmt == "nemotron":
            return f"<TOOLCALL>{payload}</TOOLCALL>"
        return ("<|START_ACTION|>"
                + json.dumps([{"tool_name": name, "parameters": args}], ensure_ascii=False)
                + "<|END_ACTION|>")
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


def _type_str(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "any"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        shown = "|".join(json.dumps(v, ensure_ascii=False) for v in enum[:6])
        return shown + ("|…" if len(enum) > 6 else "")
    t = schema.get("type")
    if isinstance(t, list):
        return "|".join(str(x) for x in t if x != "null") or "any"
    if t == "array":
        return f"{_type_str(schema.get('items'))}[]" if isinstance(schema.get("items"), dict) else "array"
    if isinstance(t, str):
        return t
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs:
            return "|".join(_type_str(s) for s in subs if not (isinstance(s, dict) and s.get("type") == "null"))
    return "any"


def _one_line(text: Any, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    line = " ".join(text.strip().split("\n", 1)[0].split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def prompted_tools_block(tools: Sequence[dict[str, Any]]) -> str:
    """Compact tool instructions for templates without native tool support.

    Lists each tool as a typed signature with a one-line description (optional
    parameters marked ``?``), then the exact ``<tool_call>`` JSON format. Kept
    deliberately short: it is paid for in tokens on every request.
    """
    lines = ["# Tools", "You can call these tools:"]
    for tool in tools:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        required = set(params.get("required") or ())
        sig = ", ".join(f"{p}{'' if p in required else '?'}: {_type_str(s)}" for p, s in props.items())
        desc = _one_line(fn.get("description"), 160)
        lines.append(f"- {name}({sig})" + (f": {desc}" if desc else ""))
        pdesc = [f"{p}: {_one_line(s.get('description'), 90)}" for p, s in props.items()
                 if isinstance(s, dict) and _one_line(s.get("description"), 90)]
        if pdesc:
            lines.append("    " + "; ".join(pdesc))
    lines += [
        "",
        "To call a tool, reply with exactly:",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"param": "value"}}',
        "</tool_call>",
        "Use one block per call and valid JSON arguments. Results come back in <tool_response> tags. "
        "If no tool is needed, answer normally.",
    ]
    return "\n".join(lines)
