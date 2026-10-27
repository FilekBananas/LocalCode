"""Tests for ``localcode.runtime.toolcalls`` — the tool-call robustness core.

The heart of the file is ``CASES``: a table of outputs local models really
produce, one group per convention (Hermes/Qwen, Qwen3-Coder XML, Llama 3 JSON
and ``<|python_tag|>``, Mistral ``[TOOL_CALLS]``, gpt-oss harmony channels, GLM
``arg_key``/``arg_value``, DeepSeek, Kimi, MiniMax, Gemma, pythonic, fenced
JSON) plus everything that goes wrong in practice: missing braces, trailing
commas, single quotes, raw newlines inside strings, arguments smuggled as a
JSON string, misspelled tool names, calls buried in prose, JSON with braces and
quotes inside its strings, Polish text, truncated generations.

Every case is checked four ways:

1. ``parse_output`` returns the expected calls, content and reasoning;
2. no tool markup leaks into the visible text;
3. ``StreamSplitter`` fed in chunks of 1, 3, 7, 13 characters and at random
   boundaries produces exactly the same visible text and reasoning as the
   one-shot parse (the two documented retroactive repairs excepted);
4. the stream never shows tool markup either.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.toolcalls import (
    REASONING_CLOSERS,
    REASONING_OPENERS,
    TOOL_FORMATS,
    ParsedOutput,
    StreamSplitter,
    ToolCall,
    detect_tool_format,
    format_hint,
    loads_lenient,
    new_call_id,
    parse_output,
    prompted_tools_block,
    render_tool_call,
)

# --------------------------------------------------------------------------- #
# The tools the model is offered in (almost) every case
# --------------------------------------------------------------------------- #

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "limit": {"type": "integer", "description": "How many lines."},
            "offset": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Write a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search files.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}, "regex": {"type": "boolean"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "todo_write", "description": "Update the todo list.",
        "parameters": {"type": "object", "properties": {
            "todos": {"type": "array", "items": {"type": "string"}}},
            "required": ["todos"]}}},
]

_DEFAULT_TOOLS = object()  # sentinel: "use TOOLS"


@dataclass(frozen=True)
class Case:
    """One model output and everything we expect the parser to make of it."""

    id: str
    text: str
    fmt: str | None
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    content: str | None = None          # exact visible text (None = not asserted)
    reasoning: str | None = None
    errors: int = 0                     # how many model-facing error strings
    starts_in_reasoning: bool = False
    tools: Any = _DEFAULT_TOOLS
    stream_equivalent: bool = True      # False only for the documented retro repairs

    @property
    def known_tools(self) -> Any:
        return TOOLS if self.tools is _DEFAULT_TOOLS else self.tools

    def parse(self) -> ParsedOutput:
        return parse_output(self.text, tool_format=self.fmt, known_tools=self.known_tools,
                            starts_in_reasoning=self.starts_in_reasoning)

    def splitter(self) -> StreamSplitter:
        return StreamSplitter(self.fmt, self.starts_in_reasoning, known_tools=self.known_tools)


READ_A = ("read_file", {"path": "a.py"})
READ_B = ("read_file", {"path": "b.py"})

CASES: list[Case] = [
    # ---------------------------------------------------------------- hermes
    Case("hermes.simple", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>',
         "hermes", (READ_A,), content=""),
    Case("hermes.prose_around",
         'Sprawdzę plik.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>\nGotowe.',
         "hermes", (READ_A,), content="Sprawdzę plik.\n\nGotowe."),
    Case("hermes.two_calls",
         '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
         '<tool_call>{"name":"read_file","arguments":{"path":"b.py"}}</tool_call>',
         "hermes", (READ_A, READ_B), content=""),
    Case("hermes.no_closing_tag", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}',
         "hermes", (READ_A,), content=""),
    Case("hermes.missing_closing_brace",
         '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.trailing_comma",
         '<tool_call>{"name": "read_file", "arguments": {"path": "a.py",}}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.single_quotes",
         "<tool_call>{'name': 'read_file', 'arguments': {'path': 'a.py'}}</tool_call>", "hermes", (READ_A,)),
    Case("hermes.python_literals",
         "<tool_call>{'name': 'grep', 'arguments': {'pattern': 'x', 'regex': True}}</tool_call>",
         "hermes", (("grep", {"pattern": "x", "regex": True}),)),
    Case("hermes.unescaped_newline_in_string",
         '<tool_call>{"name": "write_file", "arguments": {"path": "a.py", "content": "line1\nline2"}}</tool_call>',
         "hermes", (("write_file", {"path": "a.py", "content": "line1\nline2"}),)),
    Case("hermes.arguments_as_json_string",
         '<tool_call>{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}</tool_call>',
         "hermes", (READ_A,)),
    Case("hermes.wrong_name_casing",
         '<tool_call>{"name": "Read_File", "arguments": {"path": "a.py"}}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.alias_name_and_argument",
         '<tool_call>{"name": "cat", "arguments": {"file": "a.py"}}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.fuzzy_name",
         '<tool_call>{"name": "read_fil", "arguments": {"path": "a.py"}}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.namespaced_name",
         '<tool_call>{"name": "functions.read_file", "arguments": {"path": "a.py"}}</tool_call>',
         "hermes", (READ_A,)),
    Case("hermes.parameters_instead_of_arguments",
         '<tool_call>{"name": "read_file", "parameters": {"path": "a.py"}}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.braces_inside_string",
         '<tool_call>{"name": "bash", "arguments": {"command": "awk \'{print $1}\' a.txt"}}</tool_call>',
         "hermes", (("bash", {"command": "awk '{print $1}' a.txt"}),)),
    Case("hermes.quotes_inside_string",
         '<tool_call>{"name": "bash", "arguments": {"command": "echo \\"hi\\" > a.txt"}}</tool_call>',
         "hermes", (("bash", {"command": 'echo "hi" > a.txt'}),)),
    Case("hermes.closing_tag_inside_string",
         '<tool_call>{"name": "write_file", "arguments": {"path": "a.py", '
         '"content": "Użyj </tool_call> w tekście"}}</tool_call>',
         "hermes", (("write_file", {"path": "a.py", "content": "Użyj </tool_call> w tekście"}),)),
    Case("hermes.polish_arguments",
         '<tool_call>{"name": "write_file", "arguments": {"path": "żółć.txt", '
         '"content": "Zażółć gęślą jaźń"}}</tool_call>',
         "hermes", (("write_file", {"path": "żółć.txt", "content": "Zażółć gęślą jaźń"}),)),
    Case("hermes.nested_object_argument",
         '<tool_call>{"name": "todo_write", "arguments": {"todos": ["a {b}", "c \\"d\\""]}}</tool_call>',
         "hermes", (("todo_write", {"todos": ["a {b}", 'c "d"']}),)),
    Case("hermes.string_int_coerced",
         '<tool_call>{"name":"read_file","arguments":{"path":"a.py","limit":"20"}}</tool_call>',
         "hermes", (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("hermes.string_bool_coerced",
         '<tool_call>{"name":"grep","arguments":{"pattern":"x","regex":"true"}}</tool_call>',
         "hermes", (("grep", {"pattern": "x", "regex": True}),)),
    Case("hermes.float_int_coerced",
         '<tool_call>{"name":"read_file","arguments":{"path":"a.py","limit":20.0}}</tool_call>',
         "hermes", (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("hermes.array_of_calls",
         '<tool_call>[{"name":"read_file","arguments":{"path":"a.py"}},'
         '{"name":"read_file","arguments":{"path":"b.py"}}]</tool_call>', "hermes", (READ_A, READ_B)),
    Case("hermes.name_then_json", '<tool_call>read_file\n{"path": "a.py"}</tool_call>', "hermes", (READ_A,)),
    Case("hermes.chatter_inside_block",
         '<tool_call>\nSure! {"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>',
         "hermes", (READ_A,)),
    Case("hermes.followed_by_im_end",
         '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call><|im_end|>', "hermes", (READ_A,),
         content=""),
    Case("hermes.hallucinated_tool_response",
         '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>\n'
         "<tool_response>\nzmyślony wynik\n</tool_response>\nPlik ma 3 linie.",
         "hermes", (READ_A,), content=""),
    Case("hermes.fenced_json_call", '```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
         "hermes", (READ_A,), content=""),
    Case("hermes.call_inside_plain_fence",
         'Oto wywołanie:\n```\n<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>\n```\nKoniec.',
         "hermes", (READ_A,), content="Oto wywołanie:\n\nKoniec."),
    Case("hermes.mentioned_in_prose",
         "Aby wywołać narzędzie, użyj <tool_call> z JSON-em w środku. To wszystko.", "hermes",
         content="Aby wywołać narzędzie, użyj <tool_call> z JSON-em w środku. To wszystko."),
    Case("hermes.inline_code_marker", "Napisz `<tool_call>` a potem JSON.", "hermes",
         content="Napisz `<tool_call>` a potem JSON."),
    # ------------------------------------------------------------- reasoning
    Case("think.plain", "<think>Muszę przeczytać plik.</think>Oto odpowiedź.", "hermes",
         content="Oto odpowiedź.", reasoning="Muszę przeczytać plik."),
    Case("think.text_before", "Wstęp.<think>myśl</think>Odpowiedź.", "hermes",
         content="Wstęp.Odpowiedź.", reasoning="myśl"),
    Case("think.unclosed", "<think>Myślę dalej i nie kończę", "hermes",
         content="", reasoning="Myślę dalej i nie kończę"),
    Case("think.call_inside_reasoning",
         '<think>Trzeba to sprawdzić <tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call></think>'
         "Zrobione.", "hermes", (READ_A,), content="Zrobione.", reasoning="Trzeba to sprawdzić"),
    Case("think.call_after_reasoning",
         '<think>Plan</think><tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>',
         "hermes", (READ_A,), content="", reasoning="Plan"),
    Case("think.starts_in_reasoning_flag", "Myślę o tym.</think>Odpowiedź.", "hermes",
         content="Odpowiedź.", reasoning="Myślę o tym.", starts_in_reasoning=True),
    Case("think.call_with_starts_in_reasoning",
         'Plan.</think><tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>',
         "hermes", (READ_A,), content="", reasoning="Plan.", starts_in_reasoning=True),
    # parse_output repairs a stray </think> retroactively; a stream cannot.
    Case("think.orphan_close_without_flag", "Zastanawiam się...</think>Odpowiedź.", "hermes",
         content="Odpowiedź.", reasoning="Zastanawiam się...", stream_equivalent=False),
    Case("think.magistral_brackets", "[THINK]Analiza[/THINK]Odpowiedź.", "mistral",
         content="Odpowiedź.", reasoning="Analiza"),
    Case("think.seed_oss", "<seed:think>Analiza</seed:think>Odpowiedź.", "seed_oss",
         content="Odpowiedź.", reasoning="Analiza"),
    Case("think.fenced_marker_is_literal", "Przykład:\n```xml\n<think>przykład</think>\n```\nKoniec.", "hermes",
         content="Przykład:\n```xml\n<think>przykład</think>\n```\nKoniec."),
    Case("think.inline_code_marker_inside_reasoning",
         "<think>Trzeba napisać `<tool_call>` ręcznie</think>Gotowe.", "hermes",
         content="Gotowe.", reasoning="Trzeba napisać `<tool_call>` ręcznie"),
    Case("think.eos_inside_reasoning", "<think>myśl<|im_end|>", "hermes", content="", reasoning="myśl"),
    Case("think.new_turn_inside_reasoning", "<think>myśl<|im_start|>user\nnowe pytanie", "hermes",
         content="", reasoning="myśl"),
    Case("fence.language_with_a_space_is_prose",
         'Kod:\n```json output\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```',
         "hermes", content='Kod:\n```json output\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```'),
    Case("harmony.pseudo_header_is_text", "<|channel|>" + "x" * 500 + "<|message|>tekst<|return|>", "harmony",
         content="x" * 500 + "tekst"),
    # ----------------------------------------------------------- qwen3-coder
    Case("qwen3_coder.simple",
         "<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
         "qwen3_coder", (READ_A,), content=""),
    Case("qwen3_coder.multiline_value",
         "<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\n"
         "def f():\n    return 1\n</parameter>\n</function>\n</tool_call>",
         "qwen3_coder", (("write_file", {"path": "a.py", "content": "def f():\n    return 1"}),)),
    Case("qwen3_coder.typed_value",
         "<tool_call>\n<function=read_file>\n<parameter=path>a.py</parameter>\n"
         "<parameter=limit>20</parameter>\n</function>\n</tool_call>",
         "qwen3_coder", (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("qwen3_coder.two_calls",
         "<tool_call><function=read_file><parameter=path>a.py</parameter></function></tool_call>\n"
         "<tool_call><function=read_file><parameter=path>b.py</parameter></function></tool_call>",
         "qwen3_coder", (READ_A, READ_B)),
    Case("qwen3_coder.prose_before",
         "Najpierw przeczytam plik.\n<tool_call>\n<function=read_file>\n<parameter=path>a.py</parameter>\n"
         "</function>\n</tool_call>", "qwen3_coder", (READ_A,), content="Najpierw przeczytam plik."),
    Case("qwen3_coder.truncated", "<tool_call>\n<function=read_file>\n<parameter=path>\na.py",
         "qwen3_coder", (), content="", errors=1),
    Case("seed_oss.wrapper",
         "<seed:tool_call>\n<function=read_file>\n<parameter=path>a.py</parameter>\n</function>\n</seed:tool_call>",
         "seed_oss", (READ_A,)),
    # ---------------------------------------------------------------- llama3
    Case("llama3.bare_json", '{"name": "read_file", "parameters": {"path": "a.py"}}', "llama3", (READ_A,),
         content=""),
    Case("llama3.python_tag", '<|python_tag|>{"name": "read_file", "parameters": {"path": "a.py"}}<|eom_id|>',
         "llama3", (READ_A,), content=""),
    Case("llama3.python_tag_builtin", '<|python_tag|>brave_search.call(query="x")<|eom_id|>', "llama3",
         (("brave_search", {"query": "x"}),)),
    Case("llama3.prose_then_json", 'Zaraz sprawdzę.\n{"name": "read_file", "parameters": {"path": "a.py"}}',
         "llama3", (READ_A,), content="Zaraz sprawdzę."),
    Case("llama3.semicolon_separated",
         '{"name": "read_file", "parameters": {"path": "a.py"}};'
         '{"name": "read_file", "parameters": {"path": "b.py"}}', "llama3", (READ_A, READ_B), content=""),
    Case("llama3.newline_separated",
         '{"name": "read_file", "parameters": {"path": "a.py"}}\n'
         '{"name": "read_file", "parameters": {"path": "b.py"}}', "llama3", (READ_A, READ_B), content=""),
    Case("llama3.semicolon_python_tag", '<|python_tag|>read_file(path="a.py");read_file(path="b.py")<|eom_id|>',
         "llama3", (READ_A, READ_B)),
    # --------------------------------------------------------------- mistral
    Case("mistral.array", '[TOOL_CALLS][{"name": "read_file", "arguments": {"path": "a.py"}}]', "mistral",
         (READ_A,), content=""),
    Case("mistral.name_args", '[TOOL_CALLS]read_file[ARGS]{"path": "a.py"}', "mistral", (READ_A,)),
    Case("mistral.call_id", '[TOOL_CALLS]read_file[CALL_ID]abcdefghi[ARGS]{"path": "a.py"}', "mistral", (READ_A,)),
    Case("mistral.two_calls",
         '[TOOL_CALLS][{"name":"read_file","arguments":{"path":"a.py"}},'
         '{"name":"read_file","arguments":{"path":"b.py"}}]', "mistral", (READ_A, READ_B)),
    Case("mistral.prose_before_and_eos",
         'Sprawdzam.[TOOL_CALLS][{"name": "read_file", "arguments": {"path": "a.py"}}]</s>', "mistral",
         (READ_A,), content="Sprawdzam."),
    Case("mistral.unicode_args",
         '[TOOL_CALLS]write_file[ARGS]{"path": "notatki.txt", "content": "Zażółć gęślą jaźń — ćwierć"}',
         "mistral", (("write_file", {"path": "notatki.txt", "content": "Zażółć gęślą jaźń — ćwierć"}),)),
    # --------------------------------------------------------------- harmony
    Case("harmony.analysis_then_call",
         "<|channel|>analysis<|message|>Trzeba odczytać plik.<|end|>"
         "<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>"
         '{"path": "a.py"}<|call|>', "harmony", (READ_A,), content="", reasoning="Trzeba odczytać plik."),
    Case("harmony.analysis_then_final",
         "<|channel|>analysis<|message|>Myślę.<|end|><|start|>assistant<|channel|>final<|message|>"
         "Odpowiedź.<|return|>", "harmony", content="Odpowiedź.", reasoning="Myślę."),
    Case("harmony.two_calls",
         '<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{"path":"a.py"}<|call|>'
         "<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>"
         '{"path":"b.py"}<|call|>', "harmony", (READ_A, READ_B), content=""),
    Case("harmony.commentary_preamble",
         "<|channel|>commentary<|message|>Zaraz to zrobię.<|end|>"
         "<|start|>assistant<|channel|>final<|message|>OK<|return|>", "harmony", content="Zaraz to zrobię.OK"),
    Case("harmony.starts_in_analysis",
         "Analiza.<|end|><|start|>assistant<|channel|>final<|message|>Odp.<|return|>", "harmony",
         content="Odp.", reasoning="Analiza.", starts_in_reasoning=True),
    Case("harmony.python_tool", "<|channel|>commentary to=python<|constrain|>code<|message|>print(1)<|call|>",
         "harmony", (("python", {"code": "print(1)"}),)),
    # ------------------------------------------------------------------- glm
    Case("glm.simple", "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n</tool_call>",
         "glm", (READ_A,), content=""),
    Case("glm.two_args",
         "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n"
         "<arg_key>limit</arg_key>\n<arg_value>20</arg_value>\n</tool_call>",
         "glm", (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("glm.after_thinking",
         "<think>Plan</think>\n<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n"
         "</tool_call>", "glm", (READ_A,), reasoning="Plan"),
    Case("glm.no_arguments", "<tool_call>todo_write\n</tool_call>", "glm", (("todo_write", {}),)),
    Case("glm.unicode",
         "<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>żółć.txt</arg_value>\n"
         "<arg_key>content</arg_key>\n<arg_value>Zażółć gęślą jaźń</arg_value>\n</tool_call>",
         "glm", (("write_file", {"path": "żółć.txt", "content": "Zażółć gęślą jaźń"}),)),
    # -------------------------------------------------------------- deepseek
    Case("deepseek.v31",
         "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file"
         '<｜tool▁sep｜>{"path": "a.py"}<｜tool▁call▁end｜>'
         "<｜tool▁calls▁end｜>", "deepseek", (READ_A,), content=""),
    Case("deepseek.v3_fenced_arguments",
         "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function"
         '<｜tool▁sep｜>read_file\n```json\n{"path": "a.py"}\n```'
         "<｜tool▁call▁end｜><｜tool▁calls▁end｜>", "deepseek", (READ_A,)),
    Case("deepseek.two_calls",
         "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file"
         '<｜tool▁sep｜>{"path": "a.py"}<｜tool▁call▁end｜>'
         "<｜tool▁call▁begin｜>read_file"
         '<｜tool▁sep｜>{"path": "b.py"}<｜tool▁call▁end｜>'
         "<｜tool▁calls▁end｜>", "deepseek", (READ_A, READ_B)),
    Case("deepseek.prose_before",
         "Sprawdzę.\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file"
         '<｜tool▁sep｜>{"path": "a.py"}<｜tool▁call▁end｜>'
         "<｜tool▁calls▁end｜>", "deepseek", (READ_A,), content="Sprawdzę."),
    # -------------------------------------------- kimi / minimax / gemma / …
    Case("kimi.simple",
         "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>"
         '{"path": "a.py"}<|tool_call_end|><|tool_calls_section_end|>', "kimi_k2", (READ_A,), content=""),
    Case("minimax.simple",
         '<minimax:tool_call>\n<invoke name="read_file">\n<parameter name="path">a.py</parameter>\n</invoke>\n'
         "</minimax:tool_call>", "minimax_m2", (READ_A,), content=""),
    Case("minimax.typed_parameter",
         '<minimax:tool_call>\n<invoke name="read_file">\n<parameter name="path">a.py</parameter>\n'
         '<parameter name="limit" string="false">20</parameter>\n</invoke>\n</minimax:tool_call>',
         "minimax_m2", (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("gemma4.simple", '<|tool_call>call:read_file{path:<|"|>a.py<|"|>}<tool_call|>', "gemma4", (READ_A,)),
    Case("function_gemma.simple",
         "<start_function_call>call:read_file{path:<escape>a.py<escape>}<end_function_call>",
         "function_gemma", (READ_A,)),
    Case("gemma3.tool_code_fence", '```tool_code\nread_file(path="a.py")\n```', "hermes", (READ_A,), content=""),
    Case("pythonic.wrapped", '<|tool_call_start|>[read_file(path="a.py")]<|tool_call_end|>', "pythonic", (READ_A,)),
    Case("pythonic.bare_list", '[read_file(path="a.py"), read_file(path="b.py")]', "pythonic", (READ_A, READ_B)),
    Case("pythonic.typed_kwargs", '[read_file(path="a.py", limit=20)]', "pythonic",
         (("read_file", {"path": "a.py", "limit": 20}),)),
    Case("granite.tool_call_list", '<|tool_call|>[{"name": "read_file", "arguments": {"path": "a.py"}}]',
         "granite", (READ_A,)),
    Case("nemotron.toolcall", '<TOOLCALL>[{"name": "read_file", "arguments": {"path": "a.py"}}]</TOOLCALL>',
         "nemotron", (READ_A,)),
    Case("command_r.start_action",
         '<|START_ACTION|>[{"tool_name": "read_file", "parameters": {"path": "a.py"}}]<|END_ACTION|>',
         "command_r", (READ_A,)),
    # ---------------------------------------------------- generic / fallback
    Case("json.openai_tool_calls_shape",
         '{"tool_calls": [{"id": "call_1", "type": "function", '
         '"function": {"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}}]}', None, (READ_A,)),
    Case("json.singular_tool_call_wrapper",
         '{"tool_call": {"name": "read_file", "arguments": {"path": "a.py"}}}', None, (READ_A,)),
    Case("json.bare_array_of_calls",
         '[{"name": "read_file", "arguments": {"path": "a.py"}}, '
         '{"name": "read_file", "arguments": {"path": "b.py"}}]', "llama3", (READ_A, READ_B), content=""),
    Case("json.bare_array_cut_off",
         '[{"name": "read_file", "arguments": {"path": "a.py"}}', "llama3", (), content="", errors=1),
    Case("json.fenced_wrapper",
         '```json\n{"tool_call": {"name": "read_file", "arguments": {"path": "a.py"}}}\n```', None, (READ_A,),
         content=""),
    Case("json.recipient_name", '{"recipient_name": "functions.read_file", "parameters": {"path": "a.py"}}',
         "hermes", (READ_A,)),
    Case("react.action_lines",
         'Thought: muszę odczytać plik.\nAction: read_file\nAction Input: {"path": "a.py"}', None, (READ_A,),
         content="Thought: muszę odczytać plik.", stream_equivalent=False),
    # a model that answers in another family's markup is still understood
    Case("mixed.hermes_from_a_glm_model", '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n'
         "</tool_call>", "glm", (READ_A,)),
    Case("mixed.glm_from_a_hermes_model",
         "<tool_call>read_file\n<arg_key>path</arg_key>\n<arg_value>a.py</arg_value>\n</tool_call>",
         "hermes", (READ_A,)),
    Case("mixed.xml_from_a_hermes_model",
         "<tool_call><function=read_file><parameter=path>a.py</parameter></function></tool_call>",
         "hermes", (READ_A,)),
    # ------------------------------------------------------- plain prose ---
    Case("prose.no_call", "Plik zawiera funkcję `main`, która liczy sumę. Nie potrzebuję narzędzi.", "hermes",
         content="Plik zawiera funkcję `main`, która liczy sumę. Nie potrzebuję narzędzi."),
    Case("prose.json_answer", 'Oto wynik:\n```json\n{"suma": 42, "nazwa": "test"}\n```\nTo wszystko.', "hermes",
         content='Oto wynik:\n```json\n{"suma": 42, "nazwa": "test"}\n```\nTo wszystko.'),
    Case("prose.python_fence", 'Kod:\n```python\ndef f():\n    return {"name": "read_file"}\n```\nGotowe.',
         "hermes", content='Kod:\n```python\ndef f():\n    return {"name": "read_file"}\n```\nGotowe.'),
    Case("prose.braces", "Użyj `{}` aby zadeklarować pusty słownik.", "hermes",
         content="Użyj `{}` aby zadeklarować pusty słownik."),
    Case("prose.json_object_answer", '{"suma": 42, "nazwa": "test"}', "hermes",
         content='{"suma": 42, "nazwa": "test"}'),
    Case("prose.empty", "", "hermes", content=""),
    Case("prose.whitespace_only", "   \n  ", "hermes", content=""),
    Case("prose.partial_marker_at_end", "Odpowiedź<tool_ca", "hermes", content="Odpowiedź<tool_ca"),
    # --------------------------------------------------------------- errors
    Case("error.unknown_tool_is_still_a_call",
         '<tool_call>{"name": "nonexistent_thing", "arguments": {}}</tool_call>', "hermes",
         (("nonexistent_thing", {}),)),
    Case("error.missing_name", '<tool_call>{"arguments": {"path": "a.py"}}</tool_call>', "hermes", (), errors=1),
    Case("error.truncated_string_argument",
         '<tool_call>{"name": "write_file", "arguments": {"path": "a.py", "content": "abc', "hermes", (),
         content="", errors=1),
    Case("error.garbage_block", "<tool_call>totally not json at all</tool_call>", "hermes", (), errors=1),
    Case("error.truncated_mid_json", '<tool_call>{"name": "read_file", "argum', "hermes", (), errors=1),
    # ----------------------------------------------------- no tools offered
    Case("no_tools.markup_stays_visible",
         '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>', "hermes", (),
         content='<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>', tools=[]),
    Case("no_tools.reasoning_still_split", "<think>myśl</think>Odpowiedź.", "hermes", (),
         content="Odpowiedź.", reasoning="myśl", tools=[]),
    Case("unknown_tools.call_parsed_without_schemas",
         '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>', "hermes", (READ_A,),
         tools=None),
]

CASE_IDS = [c.id for c in CASES]
assert len(set(CASE_IDS)) == len(CASE_IDS), "duplicate case ids"

MARKUP_NEEDLES = (
    "<tool_call>", "</tool_call>", "<function=", "<parameter=", "[TOOL_CALLS]", "<|python_tag|>",
    "<|channel|>", "<|message|>", "<|call|>", "<|return|>", "<arg_key>", "<arg_value>",
    "<｜tool▁call▁begin｜>", "<|tool_call_begin|>", "<minimax:tool_call>", "<|tool_call>",
    "<start_function_call>", "<|tool_call_start|>", "<TOOLCALL>", "<|START_ACTION|>", "<|im_end|>", "<|eom_id|>",
    "<tool_response>",
)


def _leaks(text: str, case: Case) -> list[str]:
    """Markup that must never be visible (unless the case is *about* it)."""
    if case.tools == [] or "mentioned_in_prose" in case.id or "inline_code" in case.id:
        return []
    return [n for n in MARKUP_NEEDLES if n in text]


# --------------------------------------------------------------------------- #
# parse_output
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expected_tool_calls(case: Case):
    result = case.parse()
    got = [(c.name, c.arguments) for c in result.tool_calls]
    assert got == [(name, args) for name, args in case.calls]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expected_content_and_reasoning(case: Case):
    result = case.parse()
    if case.content is not None:
        assert result.content == case.content
    if case.reasoning is not None:
        assert result.reasoning == case.reasoning


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_no_markup_leaks_into_visible_text(case: Case):
    result = case.parse()
    assert _leaks(result.content, case) == []
    assert _leaks(result.reasoning, case) == []


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_expected_error_count(case: Case):
    result = case.parse()
    assert len(result.errors) == case.errors, result.errors
    for message in result.errors:
        assert message.strip() and message[0].isupper()


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_call_ids_are_unique_and_well_shaped(case: Case):
    result = case.parse()
    ids = [c.id for c in result.tool_calls]
    assert len(set(ids)) == len(ids)
    for call in result.tool_calls:
        # the model's own id is kept when it sent one (Kimi: "functions.read_file:0")
        assert call.id and not any(ch.isspace() for ch in call.id)
        assert isinstance(call.arguments, dict)
        assert call.raw


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_parsing_is_deterministic(case: Case):
    first, second = case.parse(), case.parse()
    assert (first.content, first.reasoning, first.errors) == (second.content, second.reasoning, second.errors)
    assert [(c.name, c.arguments) for c in first.tool_calls] == [(c.name, c.arguments) for c in second.tool_calls]


# --------------------------------------------------------------------------- #
# StreamSplitter equivalence
# --------------------------------------------------------------------------- #


def _stream(case: Case, chunks) -> tuple[str, str, StreamSplitter]:
    splitter = case.splitter()
    pieces: list[tuple[str, str]] = []
    for chunk in chunks:
        pieces += splitter.feed(chunk)
    pieces += splitter.finish()
    text = "".join(v for k, v in pieces if k == "text")
    reasoning = "".join(v for k, v in pieces if k == "reasoning")
    for kind, _ in pieces:
        assert kind in ("text", "reasoning")
    return text, reasoning, splitter


def _fixed_chunks(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _random_chunks(text: str, seed: int) -> list[str]:
    rnd = random.Random(seed)
    out: list[str] = []
    i = 0
    while i < len(text):
        n = rnd.randint(1, 17)
        out.append(text[i:i + n])
        i += n
    return out or [""]


@pytest.mark.parametrize("size", [1, 3, 7, 13])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_stream_matches_one_shot_parse(case: Case, size: int):
    if not case.stream_equivalent:
        pytest.skip("documented retroactive repair: parse_output only")
    expected = case.parse()
    text, reasoning, _ = _stream(case, _fixed_chunks(case.text, size))
    assert text.strip() == expected.content
    assert reasoning.strip() == expected.reasoning


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_stream_matches_with_random_chunking(case: Case, seed: int):
    if not case.stream_equivalent:
        pytest.skip("documented retroactive repair: parse_output only")
    expected = case.parse()
    text, reasoning, _ = _stream(case, _random_chunks(case.text, seed))
    assert text.strip() == expected.content
    assert reasoning.strip() == expected.reasoning


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_stream_never_leaks_markup(case: Case):
    for size in (1, 2, 5, 11, 997):
        text, reasoning, splitter = _stream(case, _fixed_chunks(case.text, size))
        assert _leaks(text, case) == [], (size, text)
        assert _leaks(reasoning, case) == [], (size, reasoning)
        assert splitter.tool_blocks_seen >= (1 if case.calls and case.stream_equivalent else 0)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_stream_is_finished_after_finish(case: Case):
    splitter = case.splitter()
    splitter.feed(case.text)
    splitter.finish()
    assert splitter.feed("more text") == []
    assert splitter.finish() == []
    assert splitter.in_tool_call is False


def test_in_tool_call_flag_tracks_the_markup():
    splitter = StreamSplitter("hermes", known_tools=TOOLS)
    splitter.feed("Zaraz sprawdzę. ")
    assert splitter.in_tool_call is False
    splitter.feed('<tool_call>{"name": "read_file", ')
    assert splitter.in_tool_call is True
    splitter.feed('"arguments": {"path": "a.py"}}</tool_call>')
    assert splitter.in_tool_call is False
    splitter.finish()
    assert splitter.tool_blocks_seen == 1


def test_stream_emits_reasoning_and_text_in_order():
    splitter = StreamSplitter("hermes", known_tools=TOOLS)
    pieces = splitter.feed("<think>myślę</think>odpowiedź") + splitter.finish()
    assert pieces == [("reasoning", "myślę"), ("text", "odpowiedź")]


def test_stream_holds_back_a_split_marker():
    """A marker arriving one character at a time must never flash as text."""
    splitter = StreamSplitter("hermes", known_tools=TOOLS)
    seen: list[str] = []
    for char in '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>':
        seen += [v for k, v in splitter.feed(char) if k == "text"]
    seen += [v for k, v in splitter.finish() if k == "text"]
    assert "".join(seen).strip() == ""


# --------------------------------------------------------------------------- #
# The two documented retroactive repairs
# --------------------------------------------------------------------------- #


def test_orphan_think_close_is_repaired_only_by_parse_output():
    text = "Zastanawiam się...</think>Odpowiedź."
    parsed = parse_output(text, tool_format="hermes", known_tools=TOOLS)
    assert parsed.reasoning == "Zastanawiam się..."
    assert parsed.content == "Odpowiedź."
    streamed, reasoning, _ = _stream(
        Case("x", text, "hermes"), _fixed_chunks(text, 4))
    assert reasoning == ""          # the stream cannot un-send what it already showed
    assert streamed == "Zastanawiam się...Odpowiedź."


def test_prose_fallback_call_is_only_found_by_parse_output():
    text = 'Action: read_file\nAction Input: {"path": "a.py"}'
    parsed = parse_output(text, tool_format=None, known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]
    streamed, _, splitter = _stream(Case("x", text, None), _fixed_chunks(text, 4))
    assert splitter.tool_blocks_seen == 0
    assert "Action Input" in streamed


def test_fallback_needs_known_tools():
    text = 'Action: read_file\nAction Input: {"path": "a.py"}'
    assert parse_output(text, known_tools=None).tool_calls == []
    assert parse_output(text, known_tools=[]).tool_calls == []


# --------------------------------------------------------------------------- #
# Reasoning details
# --------------------------------------------------------------------------- #


def test_reasoning_blocks_are_joined_with_a_blank_line():
    parsed = parse_output("<think>raz</think>tekst<think>dwa</think>koniec", tool_format="hermes")
    assert parsed.reasoning == "raz\n\ndwa"
    assert parsed.content == "tekstkoniec"


def test_reasoning_openers_and_closers_are_consistent():
    assert "<think>" in REASONING_OPENERS and "</think>" in REASONING_CLOSERS
    assert "<|channel|>analysis<|message|>" in REASONING_OPENERS


@pytest.mark.parametrize("opener,closer", [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<seed:think>", "</seed:think>"),
    ("[THINK]", "[/THINK]"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ("<|START_THINKING|>", "<|END_THINKING|>"),
    ("<longcat_think>", "</longcat_think>"),
    ("◁think▷", "◁/think▷"),
])
def test_every_reasoning_pair_is_split(opener, closer):
    parsed = parse_output(f"{opener}myśl{closer}odpowiedź", tool_format="hermes")
    assert parsed.reasoning == "myśl"
    assert parsed.content == "odpowiedź"


def test_calls_inside_reasoning_are_ignored_when_there_are_calls_outside():
    text = ('<think>Może <tool_call>{"name":"read_file","arguments":{"path":"draft.py"}}</tool_call></think>'
            '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>')
    parsed = parse_output(text, tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]


# --------------------------------------------------------------------------- #
# loads_lenient
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ("{'a': 1}", {"a": 1}),
    ('{"a": 1,}', {"a": 1}),
    ("{a: 1}", {"a": 1}),
    ('{"a": True, "b": False, "c": None}', {"a": True, "b": False, "c": None}),
    ('{"a": "x\ny"}', {"a": "x\ny"}),
    ('{"a": "he said \\"hi\\""}', {"a": 'he said "hi"'}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('{"a": 1} trailing text', {"a": 1}),
    ('{"a": [1, 2, 3,]}', {"a": [1, 2, 3]}),
    ('{"a": {"b": {"c": 1}}}', {"a": {"b": {"c": 1}}}),
    ('{"ą": "żółć"}', {"ą": "żółć"}),
    ('{"a": 1 // komentarz\n}', {"a": 1}),
    ('{"a": "\\u0105"}', {"a": "ą"}),
    ("[1, 2]", [1, 2]),
])
def test_loads_lenient_repairs(text, expected):
    assert loads_lenient(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "nie json"])
def test_loads_lenient_rejects(text):
    with pytest.raises(ValueError):
        loads_lenient(text)


def test_loads_lenient_requires_a_string():
    with pytest.raises(ValueError):
        loads_lenient(42)  # type: ignore[arg-type]


def test_loads_lenient_survives_deep_nesting():
    with pytest.raises(ValueError):
        loads_lenient("[" * 400)


def test_loads_lenient_handles_unicode_escapes():
    assert loads_lenient('{"a": "\\ud83d\\ude00"}') == {"a": "\U0001f600"}      # surrogate pair
    assert loads_lenient('{"a": "\\u0105"}') == {"a": "ą"}


def test_lone_surrogates_are_replaced():
    """A lone surrogate would explode on the first ``.encode("utf-8")``."""
    value = loads_lenient('{"a": "x\\ud800y"}')["a"]
    assert value == "x\ufffdy"
    call = _one_call('<tool_call>{"name":"write_file","arguments":'
                     '{"path":"a.txt","content":"x\\udfffy"}}</tool_call>')
    call.arguments["content"].encode("utf-8")   # must not raise


def test_loads_lenient_reads_odd_string_syntax():
    assert loads_lenient("{'a': 'it\\'s'}") == {"a": "it's"}
    assert loads_lenient('{"a": """potrójny"""}') == {"a": "potrójny"}
    assert loads_lenient('{"a": "C:\\path"}') == {"a": "C:\\path"}   # invalid escape kept


# --------------------------------------------------------------------------- #
# Type coercion details
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arguments,expected", [
    ({"path": "a", "limit": "20"}, {"path": "a", "limit": 20}),
    ({"path": "a", "limit": "20.0"}, {"path": "a", "limit": 20}),
    ({"path": "a", "limit": 20.0}, {"path": "a", "limit": 20}),
    ({"path": 42}, {"path": "42"}),                     # string-typed parameter
    ({"path": True}, {"path": "true"}),
    ({"path": "a", "limit": None}, {"path": "a", "limit": None}),
])
def test_scalar_coercion(arguments, expected):
    text = json.dumps({"name": "read_file", "arguments": arguments})
    assert _one_call(f"<tool_call>{text}</tool_call>").arguments == expected


def test_object_argument_is_stringified_for_a_string_parameter():
    call = _one_call('<tool_call>{"name":"write_file","arguments":'
                     '{"path":"a","content":{"k":1}}}</tool_call>')
    assert json.loads(call.arguments["content"]) == {"k": 1}


def test_scalar_is_wrapped_for_an_array_parameter():
    call = _one_call('<tool_call>{"name":"todo_write","arguments":{"todos":"jedno"}}</tool_call>')
    assert call.arguments == {"todos": ["jedno"]}


# --------------------------------------------------------------------------- #
# Pythonic calls
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text,expected", [
    ('[read_file("a.py")]', {"path": "a.py"}),                       # positional → schema order
    ('[read_file(**{"path": "a.py"})]', {"path": "a.py"}),
    ('[tools.read_file(path="a.py")]', {"path": "a.py"}),
    ('[read_file(path="a.py", limit=20)]', {"path": "a.py", "limit": 20}),
    ("[read_file(path='a.py')]", {"path": "a.py"}),
])
def test_pythonic_call_shapes(text, expected):
    call = _one_call(text, fmt="pythonic")
    assert call.name == "read_file" and call.arguments == expected


def test_pythonic_non_literal_argument_is_kept_as_source():
    call = _one_call("[read_file(path=x + 1)]", fmt="pythonic")
    assert call.arguments == {"path": "x + 1"}


def test_pythonic_statements_on_separate_lines():
    parsed = parse_output('read_file(path="a.py")\nread_file(path="b.py")', tool_format="pythonic",
                          known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A, READ_B]


# --------------------------------------------------------------------------- #
# Fallbacks in prose
# --------------------------------------------------------------------------- #


def test_json_call_buried_in_a_sentence_is_rescued():
    parsed = parse_output('Zrobię tak: {"name": "read_file", "arguments": {"path": "a.py"}} i tyle.',
                          tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]
    assert "read_file" not in parsed.content


def test_a_fence_around_a_rescued_call_is_swallowed():
    parsed = parse_output('Wynik:\n```json\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```\nKoniec.',
                          tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]
    assert "```" not in parsed.content


def test_a_whole_answer_that_is_one_pythonic_call():
    parsed = parse_output('read_file(path="a.py")', tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]
    assert parsed.content == ""


def test_prose_is_not_rescued_when_the_tool_is_unknown():
    text = 'Zrobię tak: {"name": "send_email", "arguments": {"to": "x"}} i tyle.'
    assert parse_output(text, tool_format="hermes", known_tools=TOOLS).tool_calls == []


# --------------------------------------------------------------------------- #
# Harmony details
# --------------------------------------------------------------------------- #


def test_harmony_constrain_without_a_space():
    parsed = parse_output('<|channel|>commentary to=functions.read_file<|constrain|>json<|message|>'
                          '{"path":"a.py"}<|call|>', tool_format="harmony", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]


def test_harmony_unterminated_message_is_still_shown():
    parsed = parse_output("<|channel|>final<|message|>Odpowiedź bez końca", tool_format="harmony")
    assert parsed.content == "Odpowiedź bez końca"


def test_harmony_restarted_header():
    parsed = parse_output("<|start|>assistant<|start|>assistant<|channel|>final<|message|>OK<|return|>",
                          tool_format="harmony")
    assert parsed.content == "OK"


def test_harmony_absurd_header_is_treated_as_text():
    parsed = parse_output("<|channel|>" + "x" * 500 + "<|message|>tekst<|return|>", tool_format="harmony")
    assert "tekst" in parsed.content


def test_harmony_tool_message_cut_off_is_reported():
    parsed = parse_output('<|channel|>commentary to=functions.write_file <|constrain|>json<|message|>'
                          '{"path": "a.py", "content": "abc', tool_format="harmony", known_tools=TOOLS)
    assert parsed.tool_calls == [] and len(parsed.errors) == 1


# --------------------------------------------------------------------------- #
# Argument coercion and name resolution
# --------------------------------------------------------------------------- #


def _one_call(text: str, tools: Any = TOOLS, fmt: str | None = "hermes") -> ToolCall:
    parsed = parse_output(text, tool_format=fmt, known_tools=tools)
    assert len(parsed.tool_calls) == 1, parsed
    return parsed.tool_calls[0]


def test_textual_arguments_are_typed_by_the_schema():
    call = _one_call("<tool_call><function=grep><parameter=pattern>x</parameter>"
                     "<parameter=regex>false</parameter></function></tool_call>", fmt="qwen3_coder")
    assert call.arguments == {"pattern": "x", "regex": False}


def test_textual_arguments_without_a_schema_are_guessed():
    call = _one_call("<tool_call><function=unknown_tool><parameter=n>12</parameter>"
                     "<parameter=flag>true</parameter><parameter=s>tekst</parameter></function></tool_call>",
                     fmt="qwen3_coder")
    assert call.arguments == {"n": 12, "flag": True, "s": "tekst"}


def test_string_typed_parameters_are_not_converted():
    call = _one_call('<tool_call>{"name":"read_file","arguments":{"path":"123"}}</tool_call>')
    assert call.arguments == {"path": "123"}


def test_array_argument_from_a_json_string():
    call = _one_call('<tool_call>{"name":"todo_write","arguments":{"todos":"[\\"a\\",\\"b\\"]"}}</tool_call>')
    assert call.arguments == {"todos": ["a", "b"]}


@pytest.mark.parametrize("name", ["read_file", "readfile", "read-file", "Read_File", "functions.read_file",
                                  "default_api.read_file", "read_file()", "cat", "open_file", "read"])
def test_tool_name_resolution(name):
    call = _one_call('<tool_call>{"name": "%s", "arguments": {"path": "a.py"}}</tool_call>' % name)
    assert call.name == "read_file"


def test_unknown_tool_name_is_kept_as_written():
    call = _one_call('<tool_call>{"name": "send_email", "arguments": {"to": "x"}}</tool_call>')
    assert call.name == "send_email"


def test_argument_aliases_are_renamed():
    call = _one_call('<tool_call>{"name": "write_file", "arguments": {"filepath": "a.py", "text": "x"}}</tool_call>')
    assert call.arguments == {"path": "a.py", "content": "x"}


def test_known_tools_may_be_names_schemas_or_tool_objects():
    from localcode.agent.tools import default_tools, tool_schemas

    text = '<tool_call>{"name": "cat", "arguments": {"file": "a.py"}}</tool_call>'
    by_name = parse_output(text, tool_format="hermes", known_tools=["read_file", "bash"])
    assert [(c.name, c.arguments) for c in by_name.tool_calls] == [("read_file", {"file": "a.py"})]

    tools = [t for t in default_tools() if t.name in ("read_file", "bash")]
    by_object = parse_output(text, tool_format="hermes", known_tools=tools)
    assert by_object.tool_calls[0].name == "read_file"
    assert "path" in by_object.tool_calls[0].arguments, "the schema of a Tool object is used"

    by_schema = parse_output(text, tool_format="hermes", known_tools=tool_schemas(tools))
    assert [(c.name, c.arguments) for c in by_schema.tool_calls] == \
        [(c.name, c.arguments) for c in by_object.tool_calls]

    single = parse_output(text, tool_format="hermes", known_tools="read_file")
    assert single.tool_calls[0].name == "read_file"


def test_to_message_shape():
    call = _one_call('<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>')
    message = call.to_message()
    assert message["type"] == "function"
    assert message["function"] == {"name": "read_file", "arguments": {"path": "a.py"}}
    assert message["id"] == call.id


def test_new_call_id_shape():
    ids = {new_call_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(len(i) == 9 and i.isalnum() for i in ids)


# --------------------------------------------------------------------------- #
# render_tool_call (used when a template drops tool_calls from the history)
# --------------------------------------------------------------------------- #


ROUND_TRIP_ARGS = [
    ("read_file", {"path": "a.py"}),
    ("read_file", {"path": "a.py", "limit": 20}),
    ("grep", {"pattern": "x", "regex": True}),
    ("write_file", {"path": "żółć.txt", "content": "Zażółć\ngęślą"}),
    ("todo_write", {"todos": ["a", "b"]}),
]


@pytest.mark.parametrize("fmt", TOOL_FORMATS)
@pytest.mark.parametrize("name,arguments", ROUND_TRIP_ARGS, ids=[a[0] + str(i) for i, a in
                                                                 enumerate(ROUND_TRIP_ARGS)])
def test_render_tool_call_round_trips(fmt, name, arguments):
    text = render_tool_call(fmt, name, arguments)
    parsed = parse_output(text, tool_format=fmt, known_tools=TOOLS)
    assert parsed.errors == []
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [(name, arguments)]
    assert parsed.content == ""


def test_render_tool_call_defaults_to_hermes():
    assert render_tool_call(None, "read_file", {"path": "a.py"}) == render_tool_call(
        "hermes", "read_file", {"path": "a.py"})
    assert render_tool_call("nieznany_format", "read_file", {}).startswith("<tool_call>")


def test_render_tool_call_ignores_non_dict_arguments():
    assert "read_file" in render_tool_call("hermes", "read_file", None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# detect_tool_format
# --------------------------------------------------------------------------- #


def _model(key: str, *, arch: str | None = None, template: str | None = None) -> LocalModel:
    return LocalModel(key=key, format="gguf", path="/x.gguf", display_name=key, size_bytes=1,
                      meta=ModelMeta(arch=arch, chat_template=template))


@pytest.mark.parametrize("template,expected", [
    ("{% if tools %}<tool_call>{{ tools }}</tool_call>{% endif %}", "hermes"),
    ("{{ tools }}<function=name><parameter=x>", "qwen3_coder"),
    ("{{ tools }}<|channel|>commentary<|start|>", "harmony"),
    ("{{ tools }}[TOOL_CALLS]", "mistral"),
    ("{{ tools }}<arg_key>", "glm"),
    ("{{ tools }}<｜tool▁calls▁begin｜>", "deepseek"),
    ("{{ tools }}<|tool_calls_section_begin|>", "kimi_k2"),
    ("{{ tools }}<minimax:tool_call>", "minimax_m2"),
    ("{{ tools }}<|tool_call>x<tool_call|>", "gemma4"),
    ("{{ tools }}<start_function_call>", "function_gemma"),
    ("{{ tools }}<|tool_call_start|>", "pythonic"),
    ("{{ tools }}<|python_tag|>", "llama3"),
    ("{{ tools }}<TOOLCALL>", "nemotron"),
    ("{{ tools }}<|START_ACTION|>", "command_r"),
    ("{{ tools }}<|tool_call|>", "granite"),
    ("{{ tools }}<seed:tool_call>", "seed_oss"),
    ("{{ tools }}<longcat_tool_call>", "longcat"),
])
def test_detect_tool_format_from_template(template, expected):
    assert detect_tool_format(None, template) == expected


def test_detect_tool_format_without_tool_support_is_hermes():
    """A template that cannot render tools means we will prompt for Hermes JSON."""
    assert detect_tool_format(None, "{% for m in messages %}{{ m['content'] }}{% endfor %}") == "hermes"


@pytest.mark.parametrize("key,expected", [
    ("openai/gpt-oss-20b", "harmony"),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "qwen3_coder"),
    ("mistralai/Devstral-Small-2507", "mistral"),
    ("zai-org/GLM-4.5-Air", "glm"),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek"),
    ("moonshotai/Kimi-K2-Instruct", "kimi_k2"),
    ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama3"),
    ("meta-llama/Llama-4-Scout", "pythonic"),
    ("ibm-granite/granite-3.3-8b-instruct", "granite"),
    ("CohereLabs/c4ai-command-r-08-2024", "command_r"),
    ("Qwen/Qwen3-4B-Instruct-2507", "hermes"),
    ("google/gemma-3-4b-it", "hermes"),
    ("unknown/model", "hermes"),
])
def test_detect_tool_format_from_name(key, expected):
    assert detect_tool_format(_model(key)) == expected


def test_detect_tool_format_prefers_the_template_over_the_name():
    model = _model("Qwen/Qwen3-4B", template="{{ tools }}[TOOL_CALLS]")
    assert detect_tool_format(model) == "mistral"


def test_detect_tool_format_defaults_to_hermes():
    assert detect_tool_format() == "hermes"
    assert detect_tool_format(None, None) == "hermes"
    assert detect_tool_format(None, "") == "hermes"


def test_every_format_has_a_hint():
    for fmt in TOOL_FORMATS:
        hint = format_hint(fmt)
        assert hint and "tool_name" in hint
    assert format_hint(None) == format_hint("hermes")
    assert format_hint("nieznany") == format_hint("hermes")


# --------------------------------------------------------------------------- #
# prompted_tools_block
# --------------------------------------------------------------------------- #


def test_prompted_tools_block_lists_every_tool_compactly():
    block = prompted_tools_block(TOOLS)
    for tool in TOOLS:
        assert tool["function"]["name"] in block
    assert "path: string" in block
    assert "limit?: integer" in block          # optional parameters are marked
    assert "<tool_call>" in block and '{"name": "tool_name", "arguments": {"param": "value"}}' in block
    assert "<tool_response>" in block
    assert len(block) < 1800, "the block is paid for on every request"


def test_prompted_tools_block_survives_odd_schemas():
    block = prompted_tools_block([
        {"type": "function", "function": {"name": "a"}},
        {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}},
        {"name": "c", "parameters": {"properties": {"x": {"enum": ["p", "q"]}}}},
        {"function": {"description": "no name at all"}},
        {"type": "function", "function": {"name": "d", "parameters": {"properties": {
            "x": {"type": ["string", "null"]}, "y": {"type": "array", "items": {"type": "string"}},
            "z": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}},
    ])
    assert "- a()" in block and "- b()" in block
    assert '"p"|"q"' in block
    assert "string[]" in block


def test_a_prompted_block_and_its_reply_round_trip():
    """What the block tells the model to write must be what the parser reads."""
    block = prompted_tools_block(TOOLS)
    example = block.split("To call a tool, reply with exactly:\n")[1].split("\nUse one block")[0]
    reply = example.replace("tool_name", "read_file").replace('"param": "value"', '"path": "a.py"')
    parsed = parse_output(reply, tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [READ_A]


# --------------------------------------------------------------------------- #
# Robustness / fuzzing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_truncating_any_case_never_raises(case: Case):
    """A generation cut at max_tokens must not break the parser."""
    for cut in range(0, len(case.text), max(1, len(case.text) // 12)):
        parsed = parse_output(case.text[:cut], tool_format=case.fmt, known_tools=case.known_tools,
                              starts_in_reasoning=case.starts_in_reasoning)
        assert isinstance(parsed.content, str)
        splitter = case.splitter()
        splitter.feed(case.text[:cut])
        splitter.finish()


def test_random_garbage_never_raises():
    rnd = random.Random(1234)
    alphabet = list('<>{}[]"\'\\/,:;=_ \n\tabcnameargumentstool_call żółć0123456789|')
    for _ in range(300):
        text = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 220)))
        for fmt in (None, "hermes", "harmony", "glm", "qwen3_coder", "llama3"):
            parsed = parse_output(text, tool_format=fmt, known_tools=TOOLS)
            assert isinstance(parsed.content, str)
            assert all(isinstance(c.arguments, dict) for c in parsed.tool_calls)


def test_arguments_as_a_plain_string_are_reported_not_looped():
    """``"arguments": "a.py"`` used to recurse until RecursionError."""
    parsed = parse_output('<tool_call>{"name":"read_file","arguments":"a.py"}</tool_call>',
                          tool_format="hermes", known_tools=TOOLS)
    assert parsed.tool_calls == []
    assert len(parsed.errors) == 1
    assert "read_file" in parsed.errors[0]


def test_non_ascii_text_in_brackets_is_prose():
    """``str.isalpha()`` is unicode-aware, the identifier regex is not."""
    for text in ["[łatka]", "[ćwiczenie(1)]", "{ą: 1}", "[żółć"]:
        parsed = parse_output(text, tool_format="pythonic", known_tools=TOOLS)
        assert parsed.tool_calls == []


def test_a_closed_block_with_a_missing_brace_is_not_reported_as_truncated():
    complete = parse_output('<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}</tool_call>',
                            tool_format="hermes", known_tools=TOOLS)
    assert [(c.name, c.arguments) for c in complete.tool_calls] == [READ_A]
    assert complete.errors == []
    # but a block that was cut off mid-string is still reported
    cut = parse_output('<tool_call>{"name": "read_file", "arguments": {"path": "a.p',
                       tool_format="hermes", known_tools=TOOLS)
    assert cut.tool_calls == []
    assert "cut off" in cut.errors[0]


FUZZ_FRAGMENTS = [
    "Tekst odpowiedzi. ", "Zażółć gęślą jaźń. ", "\n", "\n\n", "Lista:\n- raz\n- dwa\n", "`kod` ",
    "```python\nprint(1)\n```\n", '```json\n{"a": 1}\n```\n', "<think>", "</think>", "myślę ",
    '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>',
    '<tool_call>{"name":"bash","arguments":{"command":"ls -la"}}', "<tool_call>", "</tool_call>",
    "<function=read_file><parameter=path>a.py</parameter></function>",
    '[TOOL_CALLS][{"name":"read_file","arguments":{"path":"a.py"}}]',
    '{"name": "read_file", "parameters": {"path": "a.py"}}', "<|im_end|>", "<|eot_id|>", "<|python_tag|>",
    "<arg_key>path</arg_key><arg_value>a.py</arg_value>", "<seed:think>", "</seed:think>", "[THINK]", "[/THINK]",
    "<|channel|>analysis<|message|>myśl<|end|>", "<|start|>assistant<|channel|>final<|message|>odp<|return|>",
    '<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{"path":"a.py"}<|call|>',
    "{", "}", "[", "]", '"name"', "read_file(path='a.py')", "<tool_response>",
    '<|tool_call_start|>[read_file(path="a.py")]<|tool_call_end|>',
]
FUZZ_FORMATS = [None, "hermes", "qwen3_coder", "harmony", "glm", "llama3", "mistral", "pythonic", "deepseek"]


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_fuzzed_outputs_stream_exactly_like_they_parse(seed):
    """The streaming and one-shot paths must agree on *any* mixture of markup.

    Only ``repaired_after_stream`` answers may differ — that flag exists
    precisely because those repairs need the whole answer.
    """
    rnd = random.Random(seed)
    checked = 0
    for _ in range(500):
        text = "".join(rnd.choice(FUZZ_FRAGMENTS) for _ in range(rnd.randint(1, 7)))
        fmt = rnd.choice(FUZZ_FORMATS)
        in_reasoning = rnd.random() < 0.15
        tools = rnd.choice([TOOLS, TOOLS, None])
        expected = parse_output(text, tool_format=fmt, known_tools=tools, starts_in_reasoning=in_reasoning)
        if expected.repaired_after_stream:
            continue
        checked += 1
        for size in (1, 3, 13):
            splitter = StreamSplitter(fmt, in_reasoning, known_tools=tools)
            pieces: list[tuple[str, str]] = []
            for i in range(0, len(text), size):
                pieces += splitter.feed(text[i:i + size])
            pieces += splitter.finish()
            streamed = "".join(v for k, v in pieces if k == "text").strip()
            reasoned = "".join(v for k, v in pieces if k == "reasoning").strip()
            assert streamed == expected.content, (size, fmt, text)
            assert reasoned == expected.reasoning, (size, fmt, text)
    assert checked > 300


def test_repaired_after_stream_marks_only_the_retro_repairs():
    plain = parse_output("Zwykła odpowiedź.", tool_format="hermes", known_tools=TOOLS)
    assert plain.repaired_after_stream is False
    call = parse_output('<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>',
                        tool_format="hermes", known_tools=TOOLS)
    assert call.repaired_after_stream is False
    orphan = parse_output("Myślę.</think>Odp.", tool_format="hermes", known_tools=TOOLS)
    assert orphan.repaired_after_stream is True
    expected_flag = parse_output("Myślę.</think>Odp.", tool_format="hermes", known_tools=TOOLS,
                                 starts_in_reasoning=True)
    assert expected_flag.repaired_after_stream is False
    prose = parse_output('Action: read_file\nAction Input: {"path": "a.py"}', known_tools=TOOLS)
    assert prose.repaired_after_stream is True


def test_a_very_long_argument_is_parsed():
    content = "linia\n" * 4000
    text = json.dumps({"name": "write_file", "arguments": {"path": "big.txt", "content": content}})
    call = _one_call(f"<tool_call>{text}</tool_call>")
    assert call.arguments["content"] == content


def test_many_calls_in_one_answer():
    text = "".join('<tool_call>{"name":"read_file","arguments":{"path":"f%d.py"}}</tool_call>' % i
                   for i in range(50))
    parsed = parse_output(text, tool_format="hermes", known_tools=TOOLS)
    assert len(parsed.tool_calls) == 50
    assert len({c.id for c in parsed.tool_calls}) == 50
