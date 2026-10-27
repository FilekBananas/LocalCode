"""Tests for ``localcode.runtime.templates`` against real chat templates.

Every template family LocalCode must support is checked in under
``tests/fixtures/templates`` (see ``template_fixtures.py``), so these tests run
offline. For each fixture we assert

* that a realistic conversation renders — system + user + assistant tool call +
  tool result + assistant answer, with and without tool schemas, with and
  without a generation prompt, with the thinking switch on and off;
* the exact ``TemplateInfo`` ``inspect_template`` derives from it;
* which message layouts the template *rejects* — those rejections are the
  reason ``ChatFormatter`` has layout variants, so they are pinned here.

A ``network``-marked test re-downloads the templates and checks that upstream
still behaves the way the fixtures say.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

import template_fixtures as tf
from localcode.runtime.base import TemplateInfo
from localcode.runtime.templates import (
    THINKING_KWARGS,
    TemplateRenderError,
    inspect_template,
    render_template,
    renders_tool_calls,
    stop_strings_for_template,
    template_variables,
    thinking_kwargs,
)

# --------------------------------------------------------------------------- #
# Expectations per fixture
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Expect:
    """What ``inspect_template`` must say about a fixture, plus its quirks."""

    info: TemplateInfo
    args_style: tuple[str, ...] = ("dict", "string")  # tool-call argument shapes the template accepts
    tool_history: bool = True      # the template renders an assistant tool call at all
    strict_alternation: bool = False  # rejects system/assistant/tool sequences (Gemma)


def _info(fmt: str, *, tools: bool = True, tool_role: bool = True, system: bool = True, thinking: bool = False,
          kwarg: str | None = None, stops: tuple[str, ...] = (),
          efforts: tuple[str, ...] | None = None) -> TemplateInfo:
    # A template's reasoning-effort control follows its thinking switch:
    # gpt-oss offers low/medium/high, a boolean switch only off/on.
    # (tests/runtime/test_effort.py pins the levels per family independently.)
    if efforts is None:
        efforts = () if kwarg is None else (("low", "medium", "high") if kwarg == "reasoning_effort"
                                            else ("off", "on"))
    return TemplateInfo(supports_tools=tools, supports_tool_role=tool_role, supports_system=system,
                        supports_thinking=thinking, thinking_kwarg=kwarg, tool_format=fmt,
                        stop_strings=list(stops), effort_levels=list(efforts),
                        effort_kwarg=kwarg if efforts else None)


IM_END = ("<|im_end|>",)

EXPECTED: dict[str, Expect] = {
    "qwen3-4b-instruct-2507": Expect(_info("hermes", stops=IM_END)),
    "qwen3-coder-30b": Expect(_info("qwen3_coder", stops=IM_END), args_style=("dict",)),
    "gpt-oss-20b": Expect(_info("harmony", thinking=True, kwarg="reasoning_effort",
                                stops=("<|return|>", "<|call|>"))),
    "devstral-small-2507": Expect(_info("mistral")),
    "glm-4.5-air": Expect(_info("glm", thinking=True, kwarg="enable_thinking",
                                stops=("<|user|>", "<|observation|>")), args_style=("dict",)),
    # Gemma 3 has no tool support at all: tools are prompted, results are wrapped.
    "gemma-3-4b-it": Expect(_info("hermes", tools=False, tool_role=False, stops=("<end_of_turn>",)),
                            args_style=(), tool_history=False, strict_alternation=True),
    "llama-3.1-8b-instruct": Expect(_info("llama3", stops=("<|eot_id|>", "<|eom_id|>"))),
    # DeepSeek V3.1 never renders a `tools` list (tools go into the system prompt)
    # but it does render tool calls and results — with arguments as a JSON string.
    "deepseek-v3.1": Expect(_info("deepseek", tools=False, thinking=True, kwarg="thinking",
                                  stops=("<｜end▁of▁sentence｜>",)),
                            args_style=("string",)),
    "minimax-m2": Expect(_info("minimax_m2", thinking=True, stops=("[e~[",)), args_style=("dict",)),
    "kimi-k2": Expect(_info("kimi_k2", stops=IM_END)),
    # Granite renders the tool schemas but drops assistant tool_calls from history.
    "granite-3.3-8b": Expect(_info("granite", thinking=True, kwarg="thinking", stops=("<|end_of_text|>",)),
                             tool_history=False),
    "local-mlx-tiny": Expect(_info("hermes", thinking=True, kwarg="enable_thinking", stops=IM_END)),
    "local-mlx-small": Expect(_info("hermes", thinking=True, stops=IM_END)),
    "local-gguf-tiny": Expect(_info("hermes", thinking=True, kwarg="enable_thinking", stops=IM_END)),
    "local-gguf-small": Expect(_info("hermes", thinking=True, stops=IM_END)),
}

SLUGS = [s.slug for s in tf.available()]
TOOL_SLUGS = [s for s in SLUGS if EXPECTED[s].args_style]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file."},
                    "limit": {"type": "integer", "description": "How many lines to read."},
                },
                "required": ["path"],
            },
        },
    }
]

SYS = "You are LocalCode, an assistant."
USER = "Pokaż zawartość pliku źródłowego."
RESULT = "# LocalCode — żółć\n"
ANSWER = "Plik zaczyna się nagłówkiem."
CALL_ID = "abcdefghi"  # 9 alphanumerics: the strictest shape (Mistral)


def _template(slug: str) -> str:
    return tf.load(slug)


def plain_conversation() -> list[dict]:
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "Dzięki!"},
    ]


def tool_conversation(args_style: str) -> list[dict]:
    """A conversation containing a tool call and its result."""
    arguments = {"path": "README.md"} if args_style == "dict" else json.dumps({"path": "README.md"})
    return [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": CALL_ID, "type": "function",
                         "function": {"name": "read_file", "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": CALL_ID, "name": "read_file", "content": RESULT},
        {"role": "assistant", "content": ANSWER},
    ]


def render(slug: str, messages, tools=None, **kwargs) -> str:
    return render_template(_template(slug), messages, tools, bos_token="<s>", eos_token="</s>", **kwargs)


# --------------------------------------------------------------------------- #
# Fixture inventory
# --------------------------------------------------------------------------- #


def test_every_expected_fixture_is_checked_in():
    missing = [s.slug for s in tf.SPECS if s.slug not in SLUGS]
    # only the embedding model legitimately has no chat template
    assert missing == ["local-gguf-embed"], missing
    assert set(SLUGS) == set(EXPECTED)
    assert len(SLUGS) >= 12


def test_fixtures_are_the_families_we_claim_to_support():
    families = {EXPECTED[s].info.tool_format for s in SLUGS}
    assert {"hermes", "qwen3_coder", "harmony", "mistral", "glm", "llama3", "deepseek",
            "minimax_m2", "kimi_k2", "granite"} <= families


@pytest.mark.parametrize("slug", SLUGS)
def test_spec_family_matches_expectation(slug):
    assert tf.SPEC_BY_SLUG[slug].family == EXPECTED[slug].info.tool_format


# --------------------------------------------------------------------------- #
# inspect_template
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_inspect_template(slug):
    assert inspect_template(_template(slug)) == EXPECTED[slug].info


@pytest.mark.parametrize("slug", SLUGS)
def test_inspect_returns_independent_copies(slug):
    a = inspect_template(_template(slug))
    a.stop_strings.append("MUTATED")
    b = inspect_template(_template(slug))
    assert "MUTATED" not in b.stop_strings


def test_inspect_without_template_is_conservative():
    for empty in (None, "", "   "):
        info = inspect_template(empty)
        assert info == TemplateInfo(tool_format="hermes")


def test_inspect_of_a_broken_template_does_not_raise():
    info = inspect_template("{% for m in messages %}{{ m.content }}")  # missing endfor
    assert info.supports_tools is False
    assert info.tool_format == "hermes"


def test_inspect_ignores_a_tools_key_on_messages():
    """Phi-4 style ``message['tools']`` must not count as tool support."""
    tpl = "{% for m in messages %}{{ m['tools'] | default('') }}{{ m['content'] }}{% endfor %}"
    assert inspect_template(tpl).supports_tools is False


def test_inspect_detects_a_dropped_system_message():
    tpl = ("{% for m in messages %}{% if m['role'] != 'system' %}{{ m['role'] }}: {{ m['content'] }}\n"
           "{% endif %}{% endfor %}")
    info = inspect_template(tpl)
    assert info.supports_system is False
    assert info.supports_tool_role is False


def test_inspect_detects_a_template_that_rejects_the_tool_role():
    tpl = ("{% for m in messages %}{% if m['role'] not in ['user', 'assistant'] %}"
           "{{ raise_exception('bad role') }}{% endif %}{{ m['content'] }}{% endfor %}")
    info = inspect_template(tpl)
    assert info.supports_tool_role is False
    assert info.supports_system is False


@pytest.mark.parametrize("slug", SLUGS)
def test_renders_tool_calls_matches_expectation(slug):
    assert renders_tool_calls(_template(slug)) is EXPECTED[slug].tool_history


def test_renders_tool_calls_without_template():
    assert renders_tool_calls(None) is False
    assert renders_tool_calls("{{ messages[0]['content'] }}") is False


# --------------------------------------------------------------------------- #
# Rendering: plain conversations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", SLUGS)
def test_renders_plain_conversation(slug):
    out = render(slug, plain_conversation())
    assert USER in out and ANSWER in out and "Dzięki!" in out
    assert (SYS in out) is EXPECTED[slug].info.supports_system
    assert "żółć" not in out  # sanity: the tool result is not part of this conversation


@pytest.mark.parametrize("slug", SLUGS)
def test_generation_prompt_only_appends(slug):
    with_prompt = render(slug, plain_conversation(), add_generation_prompt=True)
    without = render(slug, plain_conversation(), add_generation_prompt=False)
    assert with_prompt.startswith(without)
    # Mistral's [/INST] already is the generation prompt, so it may add nothing
    if slug != "devstral-small-2507":
        assert len(with_prompt) > len(without)


@pytest.mark.parametrize("slug", SLUGS)
def test_polish_text_is_not_escaped(slug):
    out = render(slug, [{"role": "user", "content": "Zażółć gęślą jaźń <b>&</b>"}])
    assert "Zażółć gęślą jaźń <b>&</b>" in out
    assert "&amp;" not in out and "\\u017c" not in out


@pytest.mark.parametrize("slug", SLUGS)
def test_single_user_message_renders(slug):
    """The cheapest possible prompt must work everywhere (used by title generation)."""
    out = render(slug, [{"role": "user", "content": "Cześć"}])
    assert "Cześć" in out
    assert out.strip()


# --------------------------------------------------------------------------- #
# Rendering: tools
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", TOOL_SLUGS)
def test_renders_tool_conversation(slug):
    expect = EXPECTED[slug]
    style = expect.args_style[0]
    out = render(slug, tool_conversation(style), TOOLS if expect.info.supports_tools else None)
    assert USER in out
    assert RESULT.strip() in out, "tool result must reach the prompt"
    assert ANSWER in out
    if expect.info.supports_tools:
        assert "read_file" in out
        assert "Read a file from disk." in out, "the tool description must reach the prompt"
    if expect.tool_history:
        assert "README.md" in out, "the assistant's tool call must reach the prompt"


@pytest.mark.parametrize("slug", TOOL_SLUGS)
def test_rejected_argument_styles_raise_a_polish_error(slug):
    """Templates that only accept one argument shape must fail loudly on the other
    (this is what makes ChatFormatter try its layout variants)."""
    expect = EXPECTED[slug]
    for style in ("dict", "string"):
        if style in expect.args_style:
            continue
        with pytest.raises(TemplateRenderError) as excinfo:
            render(slug, tool_conversation(style), TOOLS if expect.info.supports_tools else None)
        assert "szablon" in str(excinfo.value).lower()
        assert excinfo.value.detail


@pytest.mark.parametrize("slug", [s for s in SLUGS if EXPECTED[s].info.supports_tools])
def test_tools_are_only_rendered_when_passed(slug):
    without = render(slug, plain_conversation(), None)
    with_tools = render(slug, plain_conversation(), TOOLS)
    assert "read_file" not in without
    assert "read_file" in with_tools
    assert len(with_tools) > len(without)


def test_gemma_rejects_a_tool_conversation():
    """Gemma 3 insists on strictly alternating roles — the formatter must wrap
    tool results and merge roles for it."""
    with pytest.raises(TemplateRenderError):
        render("gemma-3-4b-it", tool_conversation("dict"))
    ok = render("gemma-3-4b-it", [
        {"role": "user", "content": f"{SYS}\n\n{USER}"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "read_file", "arguments": {}}\n</tool_call>'},
        {"role": "user", "content": f"<tool_response>\n{RESULT}\n</tool_response>"},
    ])
    assert "tool_response" in ok and SYS in ok


def test_empty_tool_list_is_the_same_as_no_tools():
    conv = plain_conversation()
    assert render("qwen3-4b-instruct-2507", conv, []) == render("qwen3-4b-instruct-2507", conv, None)


# --------------------------------------------------------------------------- #
# Thinking switches
# --------------------------------------------------------------------------- #

THINKING_SLUGS = [s for s in SLUGS if EXPECTED[s].info.thinking_kwarg]


@pytest.mark.parametrize("slug", THINKING_SLUGS)
def test_thinking_switch_changes_the_prompt(slug):
    kwarg = EXPECTED[slug].info.thinking_kwarg
    assert kwarg is not None
    messages = [{"role": "user", "content": USER}]
    on = render(slug, messages, **thinking_kwargs(kwarg, True))
    off = render(slug, messages, **thinking_kwargs(kwarg, False))
    assert on != off


def test_qwen_thinking_off_closes_the_think_block():
    messages = [{"role": "user", "content": USER}]
    off = render("local-gguf-tiny", messages, **thinking_kwargs("enable_thinking", False))
    on = render("local-gguf-tiny", messages, **thinking_kwargs("enable_thinking", True))
    assert off.rstrip().endswith("</think>")
    assert not on.rstrip().endswith("</think>")


def test_glm_thinking_off_appends_nothink():
    off = render("glm-4.5-air", [{"role": "user", "content": USER}], **thinking_kwargs("enable_thinking", False))
    assert "/nothink" in off and "<think></think>" in off


def test_deepseek_thinking_switch_opens_or_closes_the_block():
    messages = [{"role": "user", "content": USER}]
    on = render("deepseek-v3.1", messages, **thinking_kwargs("thinking", True))
    off = render("deepseek-v3.1", messages, **thinking_kwargs("thinking", False))
    assert on.rstrip().endswith("<think>")
    assert off.rstrip().endswith("</think>")


def test_gpt_oss_reasoning_effort_is_lowered_not_disabled():
    messages = [{"role": "user", "content": USER}]
    default = render("gpt-oss-20b", messages)
    low = render("gpt-oss-20b", messages, **thinking_kwargs("reasoning_effort", False))
    assert "Reasoning: low" in low
    assert "Reasoning: low" not in default
    assert thinking_kwargs("reasoning_effort", True) == {}


def test_granite_thinking_switch_applies_without_a_system_message():
    """Granite only injects its thinking instructions when the caller sends no
    system message of its own — pinned so the behaviour cannot surprise us."""
    user_only = [{"role": "user", "content": USER}]
    assert render("granite-3.3-8b", user_only, thinking=True) != render("granite-3.3-8b", user_only, thinking=False)
    with_system = plain_conversation()
    assert render("granite-3.3-8b", with_system, thinking=True) == render("granite-3.3-8b", with_system, thinking=False)


def test_thinking_kwargs_table():
    assert thinking_kwargs(None, True) == {}
    assert thinking_kwargs("enable_thinking", True) == {"enable_thinking": True}
    assert thinking_kwargs("enable_thinking", False) == {"enable_thinking": False}
    assert thinking_kwargs("thinking", False) == {"thinking": False}
    assert thinking_kwargs("thinking_budget", True) == {}
    assert thinking_kwargs("thinking_budget", False) == {"thinking_budget": 0}
    assert thinking_kwargs("unknown_switch", False) == {"unknown_switch": False}
    for kwarg in THINKING_KWARGS:
        assert isinstance(thinking_kwargs(kwarg, True), dict)


# --------------------------------------------------------------------------- #
# render_template mechanics
# --------------------------------------------------------------------------- #


def test_extra_kwargs_become_template_variables():
    tpl = "{{ custom_var }}|{{ bos_token }}|{{ eos_token }}|{{ add_generation_prompt }}"
    out = render_template(tpl, [], custom_file=None, custom_var="VALUE", bos_token="<s>", eos_token="</s>",
                          add_generation_prompt=False)
    assert out == "VALUE|<s>|</s>|False"


def test_tojson_filter_does_not_escape_and_takes_options():
    tpl = "{{ value | tojson }}|{{ value | tojson(ensure_ascii=True) }}|{{ value | tojson(indent=2) }}"
    out = render_template(tpl, [], value={"k": "ą<&>"})
    plain, ascii_, indented = out.split("|")
    assert plain == '{"k": "ą<&>"}'
    assert ascii_ == '{"k": "\\u0105<&>"}'
    assert indented.startswith("{\n")


def test_tojson_sort_keys_and_separators():
    out = render_template("{{ v | tojson(sort_keys=True, separators=(',', ':')) }}", [], v={"b": 1, "a": 2})
    assert out == '{"a":2,"b":1}'


def test_strftime_now_is_available():
    out = render_template("{{ strftime_now('%Y') }}", [])
    assert out.isdigit() and len(out) == 4


def test_generation_block_is_a_noop():
    out = render_template("A{% generation %}B{% endgeneration %}C", [])
    assert out == "ABC"


def test_loopcontrols_extension_is_enabled():
    out = render_template("{% for i in [1,2,3] %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}", [])
    assert out == "1"


def test_sandbox_blocks_attribute_access():
    with pytest.raises(TemplateRenderError):
        render_template("{{ ''.__class__.__mro__ }}", [])


def test_raise_exception_becomes_a_polish_error():
    with pytest.raises(TemplateRenderError) as excinfo:
        render_template("{{ raise_exception('roles must alternate') }}", [])
    assert "Szablon czatu modelu zgłosił błąd" in str(excinfo.value)
    assert "roles must alternate" in excinfo.value.detail


def test_syntax_error_becomes_a_polish_error_with_a_line_number():
    with pytest.raises(TemplateRenderError) as excinfo:
        render_template("{% for m in messages %}{{ m.content }}", [])
    assert "niepoprawny" in str(excinfo.value)
    assert "line" in excinfo.value.detail


def test_type_error_becomes_a_polish_error():
    with pytest.raises(TemplateRenderError) as excinfo:
        render_template("{{ 'a' + messages[0]['content'] }}", [{"role": "user", "content": {"not": "a string"}}])
    assert str(excinfo.value).startswith("Nie udało się")
    assert "TypeError" in excinfo.value.detail


def test_empty_template_is_rejected():
    for empty in ("", "   ", "\n"):
        with pytest.raises(TemplateRenderError) as excinfo:
            render_template(empty, [{"role": "user", "content": "x"}])
        assert "szablonu czatu" in str(excinfo.value)


def test_undefined_variables_render_as_empty():
    """Templates read many optional variables (``documents``, ``tools`` …)."""
    out = render_template("[{{ documents }}][{{ tools }}][{{ whatever }}]", [])
    assert out == "[None][None][]"


def test_render_is_deterministic_and_does_not_mutate_input():
    messages = tool_conversation("dict")
    snapshot = json.dumps(messages, sort_keys=True)
    first = render("qwen3-4b-instruct-2507", messages, TOOLS)
    second = render("qwen3-4b-instruct-2507", messages, TOOLS)
    assert first == second
    assert json.dumps(messages, sort_keys=True) == snapshot


# --------------------------------------------------------------------------- #
# template_variables / stop_strings_for_template
# --------------------------------------------------------------------------- #


def test_template_variables():
    variables = template_variables(_template("local-gguf-tiny"))
    assert {"messages", "tools", "add_generation_prompt", "enable_thinking"} <= set(variables)
    assert template_variables("{% not jinja %}") == frozenset()


@pytest.mark.parametrize("marker,expected", [
    ("<|im_end|>", ["<|im_end|>"]),
    ("<|eot_id|>", ["<|eot_id|>", "<|eom_id|>"]),
    ("<end_of_turn>", ["<end_of_turn>"]),
    ("<|end|>", ["<|end|>"]),
    ("<｜end▁of▁sentence｜>", ["<｜end▁of▁sentence｜>"]),
    ("[e~[", ["[e~["]),
    ("nothing special", []),
])
def test_stop_strings_for_template(marker, expected):
    assert stop_strings_for_template(f"{{{{ '{marker}' }}}}") == expected


def test_harmony_end_token_is_not_a_stop_string():
    tpl = _template("gpt-oss-20b")
    assert "<|end|>" in tpl
    assert "<|end|>" not in stop_strings_for_template(tpl)
    assert stop_strings_for_template(tpl) == ["<|return|>", "<|call|>"]


@pytest.mark.parametrize("slug", SLUGS)
def test_stop_strings_appear_in_the_template(slug):
    tpl = _template(slug)
    for stop in inspect_template(tpl).stop_strings:
        assert stop in tpl


# --------------------------------------------------------------------------- #
# Upstream check (network)
# --------------------------------------------------------------------------- #


@pytest.mark.network
@pytest.mark.parametrize("slug", [s.slug for s in tf.SPECS if s.repo])
def test_upstream_template_still_matches_the_fixture(slug):
    """Hugging Face still serves a template of the family the fixture claims.

    Run ``python tests/runtime/template_fixtures.py`` to refresh the fixtures.
    """
    spec = tf.SPEC_BY_SLUG[slug]
    template = tf.fetch(spec)
    if not template:
        pytest.skip(f"{spec.repo}: no chat template available")
    info = inspect_template(template)
    assert info.tool_format == spec.family
    expected = EXPECTED[slug].info
    assert replace(info, stop_strings=[]) == replace(expected, stop_strings=[])
