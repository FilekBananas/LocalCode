"""Tests for ``localcode.runtime.formatter``.

``ChatFormatter`` is exercised against a ``FakeEngine`` whose ``apply_template``
is the real Jinja renderer running the real chat templates checked into
``tests/fixtures/templates`` — so "native tools vs prompted tools", "tool role
vs wrapped results", the layout fallbacks and the thinking switches are all
proven on the templates the shipped models actually use.

The ``real_model`` tests at the bottom close the loop: they render a tools
prompt with a real engine (llama-server on the tiny GGUF, and ``MlxEngine``
once it exists), let the model generate and parse the reply back into a tool
call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from typing import Any, AsyncIterator

import pytest

import template_fixtures as tf
from lc_testing import model_path
from localcode.config import Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.base import (
    CompletionChunk,
    Engine,
    EngineError,
    LoadConfig,
    SamplingParams,
    TemplateInfo,
)
from localcode.runtime.formatter import CHARS_PER_TOKEN, ChatFormatter
from localcode.runtime.templates import inspect_template, render_template
from localcode.runtime.toolcalls import parse_output

# --------------------------------------------------------------------------- #
# Fake engine
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"<\|[^|]*\|>|<[^\s<>]{1,40}>|\w+|[^\w\s]|\s+")


class FakeEngine(Engine):
    """An engine that renders with the local Jinja renderer and counts calls."""

    format = "gguf"

    def __init__(self, template: str | None, *, model: LocalModel | None = None,
                 info: TemplateInfo | None = None, broken: bool = False, n_ctx: int = 4096,
                 expose_template: bool = True) -> None:
        super().__init__("fake-1", model or _model(), LoadConfig(n_ctx=n_ctx))
        self._template = template
        self.broken = broken            # apply_template always fails (old llama.cpp)
        self.template_info = info if info is not None else inspect_template(template)
        self.applied: list[dict[str, Any]] = []
        self.tokenize_calls = 0
        if expose_template and template is not None:
            self.chat_template = template

    # -- lifecycle (unused by the formatter) -------------------------------
    async def start(self) -> None:  # pragma: no cover - not used
        return None

    async def stop(self) -> None:  # pragma: no cover - not used
        return None

    @property
    def running(self) -> bool:
        return True

    # -- what the formatter uses -------------------------------------------
    async def apply_template(self, messages, tools=None, *, add_generation_prompt=True,
                             template_kwargs=None) -> str:
        self.applied.append({"messages": messages, "tools": tools, "kwargs": dict(template_kwargs or {}),
                             "add_generation_prompt": add_generation_prompt})
        if self.broken or not self._template:
            raise EngineError("Silnik nie potrafi zastosować szablonu.")
        return render_template(self._template, messages, tools,
                               add_generation_prompt=add_generation_prompt, bos_token="<s>", eos_token="</s>",
                               **(template_kwargs or {}))

    async def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls += 1
        return [len(t) for t in _TOKEN_RE.findall(text)]

    def complete(self, prompt: str, params: SamplingParams) -> AsyncIterator[CompletionChunk]:  # pragma: no cover
        raise NotImplementedError


def _model(key: str = "org/repo/model", *, template: str | None = None, fmt: str = "gguf") -> LocalModel:
    return LocalModel(key=key, format=fmt, path="/models/m.gguf", display_name="Model", size_bytes=1,
                      meta=ModelMeta(arch="qwen3", chat_template=template))


def formatter_for(slug: str, **kw) -> ChatFormatter:
    return ChatFormatter(FakeEngine(tf.load(slug), **kw))


# --------------------------------------------------------------------------- #
# Messages and tools used throughout
# --------------------------------------------------------------------------- #

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Path."},
                                      "limit": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]

SYSTEM = "You are LocalCode."
USER = "Przeczytaj plik README.md"
ANSWER = "Plik zaczyna się nagłówkiem."
RESULT = "# LocalCode\nżółć"


def conversation() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "reasoning": "Trzeba odczytać plik.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "read_file", "arguments": {"path": "README.md"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": RESULT},
        {"role": "assistant", "content": ANSWER, "reasoning": "Podsumuję."},
        {"role": "user", "content": "Dzięki!"},
    ]


# --------------------------------------------------------------------------- #
# Native tools
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", ["qwen3-4b-instruct-2507", "local-gguf-tiny", "llama-3.1-8b-instruct",
                                  "qwen3-coder-30b", "glm-4.5-air", "kimi-k2", "devstral-small-2507",
                                  "gpt-oss-20b", "minimax-m2"])
async def test_native_tools_reach_the_template(slug):
    fmt = formatter_for(slug)
    prompt = await fmt.render(conversation(), TOOLS)
    assert "read_file" in prompt and "Read a file from disk." in prompt
    assert RESULT.splitlines()[0] in prompt
    assert USER in prompt and ANSWER in prompt
    # the schemas went through the template, not through a prompted text block
    assert fmt.engine.applied[-1]["tools"] is not None
    assert "# Tools\nYou can call these tools:" not in prompt


@pytest.mark.parametrize("slug", ["qwen3-4b-instruct-2507", "local-gguf-tiny", "llama-3.1-8b-instruct"])
async def test_without_tools_nothing_tool_ish_is_rendered(slug):
    fmt = formatter_for(slug)
    prompt = await fmt.render([{"role": "user", "content": USER}], None)
    assert "read_file" not in prompt
    assert fmt.engine.applied[-1]["tools"] is None


async def test_reasoning_is_never_sent_back():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = conversation()
    messages[2]["reasoning_content"] = "też nie"
    messages[4]["thinking"] = "ani to"
    prompt = await fmt.render(messages, TOOLS)
    for secret in ("Trzeba odczytać plik.", "Podsumuję.", "też nie", "ani to"):
        assert secret not in prompt
    sent = fmt.engine.applied[-1]["messages"]
    assert not any(k in m for m in sent for k in ("reasoning", "reasoning_content", "thinking"))


async def test_input_messages_are_not_mutated():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = conversation()
    snapshot = json.dumps(messages, sort_keys=True)
    await fmt.render(messages, TOOLS)
    assert json.dumps(messages, sort_keys=True) == snapshot


# --------------------------------------------------------------------------- #
# Prompted tools (templates without native tool support)
# --------------------------------------------------------------------------- #


async def test_prompted_tools_go_into_the_system_message():
    fmt = formatter_for("gemma-3-4b-it")
    assert fmt.info.supports_tools is False
    prompt = await fmt.render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], TOOLS)
    assert "# Tools" in prompt and "read_file(path: string, limit?: integer)" in prompt
    assert SYSTEM in prompt
    assert fmt.engine.applied[-1]["tools"] is None
    assert fmt.parse_format(TOOLS) == "hermes"


async def test_prompted_tools_without_a_system_message():
    fmt = formatter_for("gemma-3-4b-it")
    prompt = await fmt.render([{"role": "user", "content": USER}], TOOLS)
    assert "# Tools" in prompt and USER in prompt


async def test_prompted_history_uses_hermes_markup():
    fmt = formatter_for("gemma-3-4b-it")
    prompt = await fmt.render(conversation(), TOOLS)
    assert '<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>' in prompt
    assert "<tool_response>" in prompt and RESULT.splitlines()[0] in prompt


async def test_template_that_drops_tool_calls_gets_them_as_text():
    """Granite renders the schemas but never prints ``tool_calls``."""
    fmt = formatter_for("granite-3.3-8b")
    assert fmt.info.supports_tools is True
    assert fmt.renders_calls() is False
    prompt = await fmt.render(conversation(), TOOLS)
    assert '<|tool_call|>[{"name": "read_file", "arguments": {"path": "README.md"}}]' in prompt
    # and the markup is the one this family parses back
    assert parse_output('<|tool_call|>[{"name": "read_file", "arguments": {"path": "README.md"}}]',
                        tool_format=fmt.tool_format, known_tools=TOOLS).tool_calls[0].name == "read_file"


async def test_renders_calls_defaults_to_true_without_a_template():
    fmt = ChatFormatter(FakeEngine(tf.load("granite-3.3-8b"), expose_template=False))
    assert fmt.template_source() is None
    assert fmt.renders_calls() is True


# --------------------------------------------------------------------------- #
# Tool results: role "tool" vs wrapped user turns
# --------------------------------------------------------------------------- #


async def test_tool_role_is_used_when_supported():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    await fmt.render(conversation(), TOOLS)
    roles = [m["role"] for m in fmt.engine.applied[-1]["messages"]]
    assert "tool" in roles
    assert "<tool_response>" not in json.dumps(fmt.engine.applied[-1]["messages"])


async def test_tool_results_are_wrapped_when_the_role_is_unknown():
    fmt = formatter_for("gemma-3-4b-it")
    await fmt.render(conversation(), TOOLS)
    sent = fmt.engine.applied[-1]["messages"]
    assert "tool" not in [m["role"] for m in sent]
    wrapped = [m for m in sent if m["role"] == "user" and "<tool_response>" in m["content"]]
    assert wrapped and RESULT in wrapped[0]["content"]


async def test_consecutive_tool_results_are_merged_into_one_turn():
    fmt = formatter_for("gemma-3-4b-it")
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": {"path": "b.py"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "AAA"},
        {"role": "tool", "tool_call_id": "b", "content": "BBB"},
    ]
    await fmt.render(messages, TOOLS)
    sent = fmt.engine.applied[-1]["messages"]
    merged = [m for m in sent if "AAA" in m["content"]]
    assert merged and "BBB" in merged[0]["content"]
    assert merged[0]["content"].count("<tool_response>") == 2


# --------------------------------------------------------------------------- #
# System placement
# --------------------------------------------------------------------------- #

NO_SYSTEM_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] != 'system' %}"
    "<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


async def test_system_text_is_prepended_to_the_first_user_message():
    fmt = ChatFormatter(FakeEngine(NO_SYSTEM_TEMPLATE))
    assert fmt.info.supports_system is False
    prompt = await fmt.render([{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}])
    assert prompt == f"<user>{SYSTEM}\n\n{USER}</user><assistant>"


async def test_system_only_conversation_without_system_support():
    fmt = ChatFormatter(FakeEngine(NO_SYSTEM_TEMPLATE))
    prompt = await fmt.render([{"role": "system", "content": SYSTEM}])
    assert prompt == f"<user>{SYSTEM}</user><assistant>"


async def test_several_system_messages_are_joined():
    fmt = ChatFormatter(FakeEngine(NO_SYSTEM_TEMPLATE))
    prompt = await fmt.render([{"role": "system", "content": "A"}, {"role": "system", "content": "B"},
                               {"role": "user", "content": "C"}])
    assert prompt == "<user>A\n\nB\n\nC</user><assistant>"


async def test_developer_role_is_treated_as_system():
    fmt = ChatFormatter(FakeEngine(NO_SYSTEM_TEMPLATE))
    prompt = await fmt.render([{"role": "developer", "content": SYSTEM}, {"role": "user", "content": USER}])
    assert SYSTEM in prompt


# --------------------------------------------------------------------------- #
# Layout fallbacks
# --------------------------------------------------------------------------- #


# A template with native tools that only accepts JSON-string arguments (what
# DeepSeek V3.x does to templates that *do* render a tools list).
STRING_ARGS_TEMPLATE = (
    "{%- if tools %}[TOOLS]{% for t in tools %}{{ t['function']['name'] }} {% endfor %}[/TOOLS]{% endif -%}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'assistant' and m['tool_calls'] is defined and m['tool_calls'] -%}"
    "{% for c in m['tool_calls'] %}[CALL]{{ c['function']['name'] + '(' + c['function']['arguments'] + ')' }}"
    "[/CALL]{% endfor %}"
    "{%- elif m['role'] == 'tool' -%}[RESULT]{{ m['content'] }}[/RESULT]"
    "{%- else -%}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}<assistant>{% endif -%}"
)


async def test_arguments_are_retried_as_a_json_string():
    fmt = ChatFormatter(FakeEngine(STRING_ARGS_TEMPLATE))
    assert fmt.info.supports_tools and fmt.info.supports_tool_role
    prompt = await fmt.render(conversation(), TOOLS)
    assert '[CALL]read_file({"path": "README.md"})[/CALL]' in prompt
    assert fmt._variant == frozenset({"args_as_string"})
    sent = fmt.engine.applied[-1]["messages"]
    call = next(m for m in sent if m.get("tool_calls"))
    assert isinstance(call["tool_calls"][0]["function"]["arguments"], str)


async def test_deepseek_prompts_tools_because_its_template_cannot_render_them():
    fmt = formatter_for("deepseek-v3.1")
    assert fmt.info.supports_tools is False and fmt.info.supports_tool_role is True
    prompt = await fmt.render(conversation(), TOOLS)
    assert "# Tools" in prompt and '{"path": "README.md"}' in prompt
    assert RESULT in prompt


async def test_gemma_falls_back_to_merged_roles():
    """Gemma insists on strict alternation: two user turns in a row must merge."""
    fmt = formatter_for("gemma-3-4b-it")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {"role": "user", "content": "I jeszcze to."},
    ]
    prompt = await fmt.render(messages, None)
    assert USER in prompt and "I jeszcze to." in prompt and SYSTEM in prompt
    assert fmt._variant and "merge_roles" in fmt._variant
    assert prompt.count("<start_of_turn>user") == 1


async def test_multiple_calls_are_split_for_llama3():
    fmt = formatter_for("llama-3.1-8b-instruct")
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}},
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": {"path": "b.py"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "AAA"},
        {"role": "tool", "tool_call_id": "b", "content": "BBB"},
    ]
    await fmt.render(messages, TOOLS)
    sent = fmt.engine.applied[-1]["messages"]
    calls = [m for m in sent if m.get("tool_calls")]
    assert len(calls) == 2 and all(len(m["tool_calls"]) == 1 for m in calls)
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "assistant", "tool"]


async def test_the_working_layout_is_remembered():
    fmt = formatter_for("deepseek-v3.1")
    await fmt.render(conversation(), TOOLS)
    attempts = len(fmt.engine.applied)
    fmt.clear_caches()
    await fmt.render(conversation(), TOOLS)
    assert len(fmt.engine.applied) == attempts + 1, "the remembered variant must be tried first"


async def test_local_renderer_takes_over_when_the_engine_cannot_render():
    fmt = formatter_for("qwen3-4b-instruct-2507", broken=True)
    prompt = await fmt.render(conversation(), TOOLS)
    assert "read_file" in prompt and USER in prompt
    assert fmt._prefer_local is True
    engine_calls = len(fmt.engine.applied)
    fmt.clear_caches()
    await fmt.render(conversation(), TOOLS)
    assert len(fmt.engine.applied) == engine_calls, "a broken engine is not asked again"


async def test_local_render_strips_a_leading_bos_for_gguf():
    template = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}"
    engine = FakeEngine(template, broken=True)
    engine.bos_token = "<s>"
    prompt = await ChatFormatter(engine).render([{"role": "user", "content": "x"}])
    assert prompt == "x"
    engine2 = FakeEngine(template, broken=True)
    engine2.bos_token = "<s>"
    prompt2 = await ChatFormatter(engine2, strip_leading_bos=False).render([{"role": "user", "content": "x"}])
    assert prompt2 == "<s>x"


async def test_polish_error_when_no_layout_works():
    fmt = ChatFormatter(FakeEngine(None, broken=True))
    with pytest.raises(EngineError) as excinfo:
        await fmt.render([{"role": "user", "content": USER}])
    assert "Nie udało się przygotować promptu" in str(excinfo.value)


async def test_a_template_that_always_raises_is_reported():
    fmt = ChatFormatter(FakeEngine("{{ raise_exception('nope') }}"))
    with pytest.raises(EngineError):
        await fmt.render(conversation(), TOOLS)


# --------------------------------------------------------------------------- #
# Call ids
# --------------------------------------------------------------------------- #


async def test_mistral_call_ids_are_nine_alphanumerics():
    fmt = formatter_for("devstral-small-2507")
    await fmt.render(conversation(), TOOLS)
    sent = fmt.engine.applied[-1]["messages"]
    call = next(m for m in sent if m.get("tool_calls"))["tool_calls"][0]
    assert len(call["id"]) == 9 and call["id"].isalnum()
    result = next(m for m in sent if m["role"] == "tool")
    assert result["tool_call_id"] == call["id"]


async def test_kimi_call_ids_get_the_functions_shape():
    fmt = formatter_for("kimi-k2")
    await fmt.render(conversation(), TOOLS)
    sent = fmt.engine.applied[-1]["messages"]
    call = next(m for m in sent if m.get("tool_calls"))["tool_calls"][0]
    assert call["id"] == "functions.read_file:0"


async def test_missing_tool_names_are_filled_from_the_call():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
        {"role": "tool", "tool_call_id": "x1", "content": "dane"},
    ]
    await fmt.render(messages, TOOLS)
    result = next(m for m in fmt.engine.applied[-1]["messages"] if m["role"] == "tool")
    assert result["name"] == "read_file"


# --------------------------------------------------------------------------- #
# Message normalisation
# --------------------------------------------------------------------------- #


async def test_content_none_becomes_empty_text():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    await fmt.render([{"role": "user", "content": None}])
    assert fmt.engine.applied[-1]["messages"][0]["content"] == ""


async def test_openai_content_parts_are_flattened_for_tool_results():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}}]},
        {"role": "tool", "tool_call_id": "x", "content": [{"type": "text", "text": "część 1"},
                                                          {"type": "text", "text": "część 2"}]},
    ]
    prompt = await fmt.render(messages, TOOLS)
    assert "część 1\nczęść 2" in prompt


async def test_string_arguments_are_normalised_to_dicts():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = [
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
    ]
    await fmt.render(messages, TOOLS)
    call = next(m for m in fmt.engine.applied[-1]["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert call["function"]["arguments"] == {"path": "a.py"}


async def test_unparsable_string_arguments_survive_as_input():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "x", "type": "function", "function": {"name": "bash", "arguments": "ls -la"}}]}]
    await fmt.render(messages, TOOLS)
    call = next(m for m in fmt.engine.applied[-1]["messages"] if m.get("tool_calls"))["tool_calls"][0]
    assert call["function"]["arguments"] == {"input": "ls -la"}


# --------------------------------------------------------------------------- #
# Thinking
# --------------------------------------------------------------------------- #


async def test_thinking_switch_is_passed_to_the_template():
    fmt = formatter_for("local-gguf-tiny")
    assert fmt.info.thinking_kwarg == "enable_thinking"
    on = await fmt.render([{"role": "user", "content": USER}], thinking=True)
    off = await fmt.render([{"role": "user", "content": USER}], thinking=False)
    assert on != off
    assert off.rstrip().endswith("</think>")
    assert fmt.engine.applied[-1]["kwargs"] == {"enable_thinking": False}


async def test_thinking_off_lowers_reasoning_effort_for_harmony():
    fmt = formatter_for("gpt-oss-20b")
    off = await fmt.render([{"role": "user", "content": USER}], thinking=False)
    on = await fmt.render([{"role": "user", "content": USER}], thinking=True)
    assert "Reasoning: low" in off and "Reasoning: low" not in on
    assert fmt.engine.applied[-1]["kwargs"] == {}


async def test_templates_without_a_switch_get_no_kwargs():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    await fmt.render([{"role": "user", "content": USER}], thinking=False)
    assert fmt.engine.applied[-1]["kwargs"] == {}


@pytest.mark.parametrize("slug,thinking,expected", [
    ("local-gguf-tiny", True, False),   # Qwen3 does not pre-open <think>
    ("local-gguf-tiny", False, False),
    ("minimax-m2", True, True),         # MiniMax always opens <think>
    ("deepseek-v3.1", True, True),      # DeepSeek opens <think> when thinking is on
    ("deepseek-v3.1", False, False),    # ... and closes it immediately when off
    ("qwen3-4b-instruct-2507", True, False),
])
async def test_starts_in_reasoning_detection(slug, thinking, expected):
    fmt = formatter_for(slug)
    prompt = await fmt.render([{"role": "user", "content": USER}], thinking=thinking)
    assert fmt.starts_in_reasoning(prompt) is expected


async def test_splitter_and_parse_are_wired_to_the_prompt():
    fmt = formatter_for("minimax-m2")
    prompt = await fmt.render([{"role": "user", "content": USER}], TOOLS)
    splitter = fmt.splitter(prompt, TOOLS)
    pieces = splitter.feed("Trzeba to sprawdzić.</think>Gotowe.") + splitter.finish()
    assert pieces == [("reasoning", "Trzeba to sprawdzić."), ("text", "Gotowe.")]
    parsed = fmt.parse("Myślę.</think>Odpowiedź.", prompt, TOOLS)
    assert parsed.reasoning == "Myślę." and parsed.content == "Odpowiedź."


async def test_parse_uses_the_prompted_format_when_tools_are_prompted():
    fmt = formatter_for("gemma-3-4b-it")
    prompt = await fmt.render([{"role": "user", "content": USER}], TOOLS)
    assert fmt.parse_format(TOOLS) == "hermes"
    parsed = fmt.parse('<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>',
                       prompt, TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [("read_file", {"path": "a.py"})]


@pytest.mark.parametrize("slug,fmt_name", [
    ("qwen3-coder-30b", "qwen3_coder"), ("glm-4.5-air", "glm"), ("gpt-oss-20b", "harmony"),
    ("devstral-small-2507", "mistral"), ("kimi-k2", "kimi_k2"), ("llama-3.1-8b-instruct", "llama3"),
])
def test_tool_format_follows_the_template(slug, fmt_name):
    fmt = formatter_for(slug)
    assert fmt.tool_format == fmt_name
    assert fmt.parse_format(TOOLS) == fmt_name


# --------------------------------------------------------------------------- #
# Stop strings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug,expected", [
    ("qwen3-4b-instruct-2507", ["<|im_end|>"]),
    ("gpt-oss-20b", ["<|return|>", "<|call|>"]),
    ("llama-3.1-8b-instruct", ["<|eot_id|>", "<|eom_id|>"]),
    ("gemma-3-4b-it", ["<end_of_turn>"]),
    ("glm-4.5-air", ["<|user|>", "<|observation|>"]),
])
def test_stop_strings(slug, expected):
    fmt = formatter_for(slug)
    assert fmt.stop_strings() == expected
    fmt.stop_strings().append("X")
    assert fmt.stop_strings() == expected  # callers get a copy


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


async def test_count_is_the_exact_token_count_of_the_prompt():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = conversation()
    prompt = await fmt.render(messages, TOOLS, add_generation_prompt=True)
    expected = len(await fmt.engine.tokenize(prompt))
    assert await fmt.count(messages, TOOLS) == expected


async def test_counts_are_cached_by_content():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = conversation()
    first = await fmt.count(messages, TOOLS)
    calls = fmt.engine.tokenize_calls
    assert await fmt.count(messages, TOOLS) == first
    assert fmt.engine.tokenize_calls == calls, "same prompt must not be tokenized twice"
    assert await fmt.count_text("zupełnie inny tekst") > 0
    assert fmt.engine.tokenize_calls == calls + 1


async def test_renders_are_cached_per_request():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    messages = conversation()
    await fmt.render(messages, TOOLS)
    applied = len(fmt.engine.applied)
    await fmt.render(messages, TOOLS)
    assert len(fmt.engine.applied) == applied
    await fmt.render(messages, TOOLS, thinking=False)     # different request key
    assert len(fmt.engine.applied) > applied


async def test_clear_caches_forgets_everything():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    await fmt.count(conversation(), TOOLS)
    applied, tokenized = len(fmt.engine.applied), fmt.engine.tokenize_calls
    fmt.clear_caches()
    await fmt.count(conversation(), TOOLS)
    assert len(fmt.engine.applied) > applied and fmt.engine.tokenize_calls > tokenized


async def test_growing_history_grows_the_count():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    short = await fmt.count([{"role": "user", "content": USER}], None)
    long = await fmt.count(conversation(), TOOLS)
    assert long > short > 0


async def test_count_cache_survives_many_different_prompts():
    fmt = formatter_for("qwen3-4b-instruct-2507", )
    for i in range(200):
        await fmt.count([{"role": "user", "content": f"wiadomość {i}"}], None)
    assert await fmt.count([{"role": "user", "content": "wiadomość 0"}], None) > 0


def test_count_fast_is_a_cheap_estimate():
    assert ChatFormatter.count_fast("") == 0
    text = "a" * 350
    assert ChatFormatter.count_fast(text) == round(350 / CHARS_PER_TOKEN)
    assert ChatFormatter.count_fast("krótko") >= 1


# --------------------------------------------------------------------------- #
# prepare() is inspectable
# --------------------------------------------------------------------------- #


def test_prepare_returns_what_the_template_gets():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    msgs, tools, kwargs = fmt.prepare(conversation(), TOOLS, thinking=True)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert tools is not None and len(tools) == 2
    assert kwargs == {}
    assert all("reasoning" not in m for m in msgs)


def test_prepare_with_an_explicit_variant():
    fmt = formatter_for("qwen3-4b-instruct-2507")
    msgs, _, _ = fmt.prepare(conversation(), TOOLS, variant=["wrap_results"])
    assert "tool" not in [m["role"] for m in msgs]


# --------------------------------------------------------------------------- #
# Real engines
# --------------------------------------------------------------------------- #

TOOL_PROMPT_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from disk.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Path of the file."}},
                       "required": ["path"]}}},
]
ASK = "Read the file README.md. Call the tool, do not answer in prose."


async def _round_trip(engine: Engine) -> None:
    fmt = ChatFormatter(engine)
    messages = [{"role": "system", "content": "You are a coding agent. Use the tools you are given."},
                {"role": "user", "content": ASK}]
    prompt = await fmt.render(messages, TOOL_PROMPT_TOOLS, thinking=False)
    assert "read_file" in prompt
    assert await fmt.count(messages, TOOL_PROMPT_TOOLS) > 10

    params = SamplingParams(max_tokens=160, temperature=0.0, stop=fmt.stop_strings())
    splitter = fmt.splitter(prompt, TOOL_PROMPT_TOOLS)
    text = ""
    async with contextlib.aclosing(engine.complete(prompt, params)) as stream:
        async for chunk in stream:
            text += chunk.text
            for kind, piece in splitter.feed(chunk.text):
                assert kind in ("text", "reasoning")
    splitter.finish()
    parsed = fmt.parse(text, prompt, TOOL_PROMPT_TOOLS)
    assert parsed.tool_calls, f"model did not call a tool: {text!r}"
    call = parsed.tool_calls[0]
    assert call.name == "read_file"
    assert "README" in json.dumps(call.arguments)


@pytest.fixture
def lc_home(tmp_path, monkeypatch):
    """Keep engine logs out of the user's real ~/.localcode."""
    home = tmp_path / "lc-home"
    home.mkdir()
    monkeypatch.setenv("LOCALCODE_HOME", str(home))
    return home


@pytest.mark.real_model
async def test_round_trip_through_llama_server(lc_home):
    from localcode.runtime.llama_engine import LlamaServerEngine, find_llama_server

    if find_llama_server(Settings()) is None:
        pytest.skip("llama-server not installed")
    path = model_path("gguf_tiny")
    model = LocalModel(key="unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M", format="gguf", path=str(path),
                       display_name="Qwen3 0.6B", size_bytes=path.stat().st_size)
    engine = LlamaServerEngine("fmt-gguf", model, LoadConfig(n_ctx=2048), settings=Settings())
    await engine.start()
    try:
        assert engine.template_info.supports_tools
        await _round_trip(engine)
    finally:
        await engine.stop()


@pytest.mark.real_model
async def test_round_trip_through_mlx(lc_home):
    mlx_engine = pytest.importorskip("localcode.runtime.mlx_engine",
                                     reason="MlxEngine is not written yet")
    path = model_path("mlx_tiny")
    model = LocalModel(key="mlx-community/Qwen3-0.6B-4bit", format="mlx", path=str(path),
                       display_name="Qwen3 0.6B MLX", size_bytes=1)
    engine = mlx_engine.MlxEngine("fmt-mlx", model, LoadConfig(n_ctx=2048), settings=Settings())
    await engine.start()
    try:
        await _round_trip(engine)
    finally:
        await engine.stop()
        await asyncio.sleep(0)
