"""Turn core chat messages (+ tool schemas) into the exact prompt a model expects.

``ChatFormatter`` sits between the agent/chatbot and an ``Engine``:

* **Tools** — native when the template renders a ``tools`` list
  (``template_info.supports_tools``); otherwise ``prompted_tools_block`` is
  merged into the system message and past assistant calls are written back as
  the same ``<tool_call>{json}</tool_call>`` text the model was told to use.
  Templates that render the schemas but silently drop ``tool_calls`` from the
  history (Granite, Phi-4 …) get the calls written into the assistant text in
  *their own* convention (``render_tool_call``), so the model never sees a tool
  result for a call it apparently never made.
* **History normalisation** — ``reasoning`` (and ``reasoning_content`` /
  ``thinking``) is stripped: reasoning is shown to the user but never re-sent.
  ``content=None`` becomes ``""`` (many templates concatenate it), tool-call
  arguments become dicts, tool-call ids get the shape the template demands
  (Mistral: 9 alphanumerics; Kimi: ``functions.<name>:<n>``), tool results use
  role ``tool`` (with ``tool_call_id`` and ``name``) when the template knows
  that role, else they are wrapped as ``<tool_response>…</tool_response>`` user
  turns (consecutive results merged). Templates without a system role get the
  system text prepended to the first user message.
* **Thinking switch / effort** — ``render(..., effort=…)`` asks for a level
  (``off|low|medium|high``) and it is resolved down to what the template can
  express (``effort_levels``: gpt-oss ``low|medium|high``, a Qwen3-style switch
  only ``off|on``, nothing at all for templates without a switch) and passed
  through the template's own kwarg (``enable_thinking``, ``reasoning_effort``,
  ``thinking_budget`` …) as detected by ``inspect_template``. Without an
  ``effort`` the legacy ``thinking`` on/off flag still decides.
* **Resilience** — templates differ in what they accept (DeepSeek concatenates
  JSON-string arguments; Llama 3.1 and gpt-oss allow one call per message;
  Gemma 3 insists on strictly alternating roles). When rendering fails the
  formatter retries with progressively more conservative layouts and, when the
  engine itself cannot render, with the local Jinja renderer
  (``runtime/templates.py``). The first layout that works is remembered.
* **Counting** — ``count`` renders the real prompt and tokenizes it through the
  engine; results are cached in an LRU keyed by the prompt hash (and renders in
  an LRU keyed by the request), so the context manager can call it freely.
  ``count_fast`` is a cheap character heuristic for rough checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from typing import Any, Iterable, Sequence

from localcode.runtime.base import Engine, EngineError, TemplateInfo
from localcode.runtime.templates import (
    TemplateRenderError,
    effort_kwargs,
    render_template,
    renders_tool_calls,
    resolve_effort,
    thinking_kwargs,
)
from localcode.runtime.toolcalls import (
    REASONING_OPENERS,
    ParsedOutput,
    StreamSplitter,
    detect_tool_format,
    loads_lenient,
    parse_output,
    prompted_tools_block,
    render_tool_call,
)

__all__ = ["ChatFormatter", "CHARS_PER_TOKEN"]

CHARS_PER_TOKEN = 3.5

_STRIPPED_KEYS = ("reasoning", "reasoning_content", "thinking")
_B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_KIMI_ID_RE = re.compile(r"^functions\.[\w.\-]+:\d+$")

# Layout variants tried in order when a template rejects the default layout.
_ARGS_AS_STRING = "args_as_string"   # tool-call arguments as JSON text (DeepSeek V3.x)
_SPLIT_CALLS = "split_calls"         # one tool call per assistant message (Llama 3.1, gpt-oss)
_WRAP_RESULTS = "wrap_results"       # tool results as <tool_response> user turns
_MERGE_ROLES = "merge_roles"         # strict user/assistant alternation (Gemma, old Mistral)
_VARIANTS: tuple[frozenset[str], ...] = (
    frozenset(),
    frozenset({_ARGS_AS_STRING}),
    frozenset({_SPLIT_CALLS}),
    frozenset({_SPLIT_CALLS, _ARGS_AS_STRING}),
    frozenset({_WRAP_RESULTS, _MERGE_ROLES}),
    frozenset({_WRAP_RESULTS, _MERGE_ROLES, _SPLIT_CALLS, _ARGS_AS_STRING}),
)


def _b62_id(seed: str, length: int = 9) -> str:
    digest = int.from_bytes(hashlib.sha1(seed.encode("utf-8")).digest(), "big")
    out = []
    for _ in range(length):
        digest, rem = divmod(digest, 62)
        out.append(_B62[rem])
    return "".join(out)


def _as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _args_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            value = loads_lenient(text)
        except ValueError:
            return {"input": arguments}
        return value if isinstance(value, dict) else {"input": value}
    if arguments is None:
        return {}
    return {"input": arguments}


class _LRU(OrderedDict):
    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def get_item(self, key: str) -> Any:
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key: str, value: Any) -> None:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.maxsize:
            self.popitem(last=False)


class ChatFormatter:
    """Prompt rendering, token counting and parser wiring for one engine."""

    def __init__(
        self,
        engine: Engine,
        *,
        count_cache_size: int = 512,
        render_cache_size: int = 64,
        strip_leading_bos: bool | None = None,
    ) -> None:
        self.engine = engine
        self._counts = _LRU(count_cache_size)
        self._renders = _LRU(render_cache_size)
        self._variant: frozenset[str] | None = None
        self._prefer_local = False
        # llama.cpp removes a leading BOS from rendered templates because the
        # tokenizer adds it again; mirror that for local fallback renders.
        self._strip_bos = (getattr(engine, "format", "") == "gguf") if strip_leading_bos is None else strip_leading_bos

    # ------------------------------------------------------------------ info
    @property
    def info(self) -> TemplateInfo:
        return self.engine.template_info

    def template_source(self) -> str | None:
        """The model's Jinja chat template, when the engine or model exposes it."""
        for attr in ("chat_template", "template_source"):
            value = getattr(self.engine, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        meta = getattr(getattr(self.engine, "model", None), "meta", None)
        value = getattr(meta, "chat_template", None)
        return value if isinstance(value, str) and value.strip() else None

    @property
    def tool_format(self) -> str:
        """Tool-call convention of the model (native)."""
        return self.info.tool_format or detect_tool_format(getattr(self.engine, "model", None), self.template_source())

    def renders_calls(self) -> bool:
        """Whether the template writes assistant tool calls back into the prompt.

        Granite, Phi-4 and several small fine-tunes render the tool *schemas* but
        print only ``message['content']``, so a call would vanish from the
        history. When we cannot see the template source we trust the engine.
        """
        template = self.template_source()
        return True if template is None else renders_tool_calls(template)

    def parse_format(self, tools: Sequence[dict[str, Any]] | None = None) -> str:
        """Convention to parse replies with: prompted tools are always Hermes-style."""
        if tools and not self.info.supports_tools:
            return "hermes"
        return self.tool_format

    def effort_levels(self) -> list[str]:
        """Effort levels this model's template can express, weakest first.
        Empty when the model has no control — surfaces must hide it then."""
        return list(self.info.effort_levels)

    def resolve_effort(self, level: str | None) -> str | None:
        """The supported level closest to ``level`` (``None`` = template default)."""
        return resolve_effort(level, self.info.effort_levels)

    def _reasoning_kwargs(self, thinking: bool, effort: str | None) -> dict[str, Any]:
        """Template kwargs for the requested reasoning effort.

        An explicit ``effort`` wins whenever the template has an effort control;
        otherwise the legacy on/off ``thinking`` switch is used.
        """
        info = self.info
        if effort is not None and info.effort_levels:
            resolved = resolve_effort(effort, info.effort_levels)
            if resolved is not None:
                return effort_kwargs(info.effort_kwarg, resolved)
        return thinking_kwargs(info.thinking_kwarg, thinking)

    def stop_strings(self) -> list[str]:
        return list(self.info.stop_strings)

    def starts_in_reasoning(self, prompt: str) -> bool:
        """True when the rendered prompt ends inside an opened reasoning block
        (e.g. the generation prompt ends with ``<think>\\n``)."""
        tail = prompt.rstrip()
        return any(tail.endswith(opener) for opener in REASONING_OPENERS)

    def splitter(self, prompt: str, tools: Sequence[dict[str, Any]] | None = None) -> StreamSplitter:
        """A ``StreamSplitter`` configured for a reply to ``prompt``."""
        return StreamSplitter(self.parse_format(tools), self.starts_in_reasoning(prompt),
                              known_tools=list(tools) if tools is not None else None)

    def parse(self, text: str, prompt: str, tools: Sequence[dict[str, Any]] | None = None) -> ParsedOutput:
        """``parse_output`` configured for a reply to ``prompt``."""
        return parse_output(text, tool_format=self.parse_format(tools),
                            known_tools=list(tools) if tools is not None else None,
                            starts_in_reasoning=self.starts_in_reasoning(prompt))

    # --------------------------------------------------------------- render
    def prepare(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        thinking: bool = True,
        effort: str | None = None,
        variant: Iterable[str] = (),
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
        """Messages, tools and template kwargs exactly as handed to the template."""
        return self._prepare(messages, tools, thinking, effort, frozenset(variant))

    async def render(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        thinking: bool = True,
        effort: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the prompt; raises ``EngineError`` (Polish) if no layout works.

        ``effort`` (``"off" | "low" | "medium" | "high" | "on"``) wins over
        ``thinking`` when the template exposes an effort control; it is resolved
        down to what the template can express (see ``effort_levels``).
        """
        key = self._request_key(messages, tools, thinking, effort, add_generation_prompt)
        cached = self._renders.get_item(key)
        if cached is not None:
            return cached
        prompt = await self._render_uncached(messages, tools, thinking, effort, add_generation_prompt)
        self._renders.put(key, prompt)
        return prompt

    async def _render_uncached(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        thinking: bool,
        effort: str | None,
        add_generation_prompt: bool,
    ) -> str:
        order = list(_VARIANTS)
        if self._variant is not None:
            order.remove(self._variant)
            order.insert(0, self._variant)
        template = self.template_source()
        last_error: Exception | None = None
        seen: set[str] = set()
        for variant in order:
            msgs, tls, kwargs = self._prepare(messages, tools, thinking, effort, variant)
            fingerprint = json.dumps([msgs, tls], sort_keys=True, default=str, ensure_ascii=False)
            if fingerprint in seen:
                continue  # this variant changes nothing for these messages
            seen.add(fingerprint)
            if not self._prefer_local or template is None:
                try:
                    prompt = await self.engine.apply_template(
                        msgs, tls, add_generation_prompt=add_generation_prompt, template_kwargs=kwargs or None
                    )
                    self._variant = variant
                    return prompt
                except EngineError as exc:
                    last_error = exc
            if template is not None:
                try:
                    prompt = self._render_local(template, msgs, tls, add_generation_prompt, kwargs)
                except TemplateRenderError as exc:
                    last_error = exc
                    continue
                if last_error is not None and not self._prefer_local:
                    self._prefer_local = True
                self._variant = variant
                return prompt
        detail = str(last_error) if last_error else "brak szablonu"
        raise EngineError(f"Nie udało się przygotować promptu dla modelu: {detail}")

    def _render_local(
        self,
        template: str,
        msgs: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
        kwargs: dict[str, Any],
    ) -> str:
        meta = getattr(getattr(self.engine, "model", None), "meta", None)
        bos = getattr(self.engine, "bos_token", None) or getattr(meta, "bos_token", None) or ""
        eos = getattr(self.engine, "eos_token", None) or getattr(meta, "eos_token", None) or ""
        prompt = render_template(template, msgs, tools, add_generation_prompt=add_generation_prompt,
                                 bos_token=bos, eos_token=eos, **kwargs)
        if self._strip_bos and bos and prompt.startswith(bos):
            prompt = prompt[len(bos):]
        return prompt

    # --------------------------------------------------------------- counting
    async def count(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        thinking: bool = True,
        effort: str | None = None,
    ) -> int:
        """Exact token count of the prompt these messages render to (engine tokenizer)."""
        prompt = await self.render(messages, tools, thinking=thinking, effort=effort,
                                   add_generation_prompt=True)
        return await self.count_text(prompt)

    async def count_text(self, text: str) -> int:
        """Exact token count of arbitrary text, cached by content hash."""
        key = hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()
        cached = self._counts.get_item(key)
        if cached is not None:
            return cached
        n = await self.engine.count_tokens(text)
        self._counts.put(key, n)
        return n

    @staticmethod
    def count_fast(text: str) -> int:
        """Cheap estimate (~3.5 characters per token) for rough budget checks."""
        if not text:
            return 0
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def clear_caches(self) -> None:
        self._counts.clear()
        self._renders.clear()

    # --------------------------------------------------------------- internals
    @staticmethod
    def _request_key(messages: Any, tools: Any, thinking: bool, effort: str | None, agp: bool) -> str:
        payload = json.dumps([messages, tools, thinking, effort, agp], sort_keys=True,
                             default=str, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8", "surrogatepass")).hexdigest()

    def _prepare(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        thinking: bool,
        effort: str | None,
        variant: frozenset[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
        info = self.info
        fmt = self.tool_format
        native_calls = info.supports_tools and self.renders_calls()
        native_tools = bool(tools) and info.supports_tools
        tool_role = info.supports_tool_role and _WRAP_RESULTS not in variant

        msgs = self._clean(messages)
        self._normalise_ids(msgs, fmt)
        if _SPLIT_CALLS in variant or (native_calls and fmt in ("harmony", "llama3")):
            msgs = self._split_calls(msgs)
        msgs = self._tool_results(msgs, tool_role)
        for m in msgs:
            if m["role"] == "assistant" and m.get("tool_calls"):
                if native_calls:
                    if _ARGS_AS_STRING in variant:
                        for call in m["tool_calls"]:
                            fn = call["function"]
                            fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                else:
                    # The template cannot carry the call: write it into the text.
                    # Use the family's own markup when the model has native tool
                    # support (that is what it will emit), Hermes when we prompt.
                    text_fmt = fmt if info.supports_tools else "hermes"
                    blocks = "\n".join(
                        render_tool_call(text_fmt, c["function"]["name"], c["function"]["arguments"], pos)
                        for pos, c in enumerate(m["tool_calls"])
                    )
                    m["content"] = f"{m['content']}\n{blocks}" if m["content"].strip() else blocks
                    del m["tool_calls"]
        if tools and not native_tools:
            block = prompted_tools_block(list(tools))
            if msgs and msgs[0]["role"] == "system":
                base = msgs[0]["content"]
                msgs[0]["content"] = f"{base}\n\n{block}" if base.strip() else block
            else:
                msgs.insert(0, {"role": "system", "content": block})
        msgs = self._system_placement(msgs, info.supports_system, _MERGE_ROLES in variant)
        if _MERGE_ROLES in variant:
            msgs = self._merge_roles(msgs)
        tools_out = [dict(t) for t in tools] if native_tools and tools else None
        return msgs, tools_out, self._reasoning_kwargs(thinking, effort)

    @staticmethod
    def _clean(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in messages:
            m = {k: v for k, v in message.items() if k not in _STRIPPED_KEYS}
            role = m.get("role") or "user"
            if role == "developer":
                role = "system"
            m["role"] = role
            content = m.get("content")
            if content is None:
                m["content"] = ""
            elif not isinstance(content, (str, list)):
                m["content"] = str(content)
            if role == "assistant" and m.get("tool_calls"):
                calls = []
                for call in m["tool_calls"]:
                    fn = call.get("function") if isinstance(call.get("function"), dict) else call
                    calls.append({
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {"name": str(fn.get("name") or ""), "arguments": _args_dict(fn.get("arguments"))},
                    })
                m["tool_calls"] = calls
            elif "tool_calls" in m:
                del m["tool_calls"]
            if role == "tool":
                m["content"] = _as_text(m.get("content"))
            out.append(m)
        return out

    @staticmethod
    def _normalise_ids(msgs: list[dict[str, Any]], fmt: str) -> None:
        """Give every call an id in the shape the template family expects and
        keep ``tool_call_id`` of results consistent with it."""
        mapping: dict[str, str] = {}
        name_of: dict[str, str] = {}
        counter = 0
        for index, m in enumerate(msgs):
            if m["role"] == "assistant" and m.get("tool_calls"):
                for pos, call in enumerate(m["tool_calls"]):
                    old = call["id"] or f"call_{index}_{pos}"
                    name = call["function"]["name"]
                    if fmt == "mistral":
                        new = old if (len(old) == 9 and old.isalnum()) else _b62_id(old)
                    elif fmt == "kimi_k2":
                        new = old if _KIMI_ID_RE.match(old) else f"functions.{name}:{counter}"
                    else:
                        new = old
                    counter += 1
                    if call["id"]:
                        mapping[call["id"]] = new
                    call["id"] = new
                    name_of[new] = name
            elif m["role"] == "tool":
                old = str(m.get("tool_call_id") or "")
                new = mapping.get(old, old)
                if not new and name_of:
                    new = list(name_of)[-1]
                if fmt == "mistral" and new and not (len(new) == 9 and new.isalnum()):
                    new = _b62_id(new)
                m["tool_call_id"] = new
                if not m.get("name") and new in name_of:
                    m["name"] = name_of[new]

    @staticmethod
    def _split_calls(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        i = 0
        while i < len(msgs):
            m = msgs[i]
            calls = m.get("tool_calls") if m["role"] == "assistant" else None
            if not calls or len(calls) == 1:
                out.append(m)
                i += 1
                continue
            j = i + 1
            results: list[dict[str, Any]] = []
            while j < len(msgs) and msgs[j]["role"] == "tool":
                results.append(msgs[j])
                j += 1
            by_id = {r.get("tool_call_id"): r for r in results}
            used: set[int] = set()
            for pos, call in enumerate(calls):
                out.append({**m, "content": m["content"] if pos == 0 else "", "tool_calls": [call]})
                result = by_id.get(call["id"])
                if result is None and pos < len(results) and id(results[pos]) not in used:
                    result = results[pos]
                if result is not None and id(result) not in used:
                    used.add(id(result))
                    out.append(result)
            out.extend(r for r in results if id(r) not in used)
            i = j
        return out

    @staticmethod
    def _tool_results(msgs: list[dict[str, Any]], tool_role: bool) -> list[dict[str, Any]]:
        if tool_role:
            return msgs
        out: list[dict[str, Any]] = []
        for m in msgs:
            if m["role"] != "tool":
                out.append(m)
                continue
            wrapped = f"<tool_response>\n{m['content']}\n</tool_response>"
            prev = out[-1] if out else None
            if prev is not None and prev.get("_tool_results"):
                prev["content"] = f"{prev['content']}\n{wrapped}"
            else:
                out.append({"role": "user", "content": wrapped, "_tool_results": True})
        for m in out:
            m.pop("_tool_results", None)
        return out

    @staticmethod
    def _system_placement(msgs: list[dict[str, Any]], supports_system: bool, strict: bool) -> list[dict[str, Any]]:
        if not supports_system:
            system_parts = [_as_text(m["content"]) for m in msgs if m["role"] == "system"]
            rest = [m for m in msgs if m["role"] != "system"]
            text = "\n\n".join(p for p in system_parts if p.strip())
            if not text:
                return rest
            for m in rest:
                if m["role"] == "user":
                    m["content"] = f"{text}\n\n{_as_text(m['content'])}"
                    return rest
            return [{"role": "user", "content": text}] + rest
        if strict:
            # only a leading system message is safe everywhere; later ones become user notes
            out: list[dict[str, Any]] = []
            for pos, m in enumerate(msgs):
                if m["role"] == "system" and pos > 0:
                    out.append({"role": "user", "content": f"[System] {_as_text(m['content'])}"})
                else:
                    out.append(m)
            return out
        return msgs

    @staticmethod
    def _merge_roles(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in msgs:
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == m["role"]
                and m["role"] in ("user", "assistant")
                and not prev.get("tool_calls")
                and not m.get("tool_calls")
                and isinstance(prev.get("content"), str)
                and isinstance(m.get("content"), str)
            ):
                prev["content"] = f"{prev['content']}\n\n{m['content']}" if prev["content"] else m["content"]
            else:
                out.append(dict(m))
        return out
