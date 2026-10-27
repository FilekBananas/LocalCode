"""Tests for localcode.agent.context (budgets, truncation, staged compaction)."""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest

from localcode.agent import context as ctx
from localcode.agent.context import (
    DROPPED_NOTE,
    STRATEGIES,
    SUMMARY_FOOTER,
    SUMMARY_HEADER,
    ContextManager,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
    limits_for,
    structure_problems,
    truncate,
)
from localcode.agent.prompts import EMPTY_RESPONSE_NUDGE, REPEATED_CALL_NUDGE
from localcode.events import Compaction, ContextUpdate

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a command.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]


def count_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    """Deterministic stand-in for ChatFormatter.count (ignores ``reasoning``)."""
    n = 2
    for m in messages:
        visible = {k: v for k, v in m.items() if k != "reasoning"}
        n += len(json.dumps(visible, ensure_ascii=False)) // 4 + 3
    if tools:
        n += len(json.dumps(tools)) // 4
    return n


class FakeCounter:
    """Async counter with call accounting; ``weight(message)`` can make some messages pricier."""

    def __init__(self, weight=None) -> None:
        self.calls = 0
        self.weight = weight

    async def __call__(self, messages, tools=None) -> int:
        self.calls += 1
        await asyncio.sleep(0)
        if self.weight is None:
            return count_tokens(messages, tools)
        n = count_tokens([], tools)
        for m in messages:
            n += int((count_tokens([m]) - 2) * self.weight(m))
        return n


class RecordingSummarizer:
    def __init__(self, text: str | None = None, *, exc: BaseException | None = None, delay: float = 0.0) -> None:
        self.text = text if text is not None else (
            "Goal: fix the parser bug\nDone so far: read the parser and the tests\n"
            "Files touched: src/parser.py: read\nKey facts/decisions: tokenizer is in src/lexer.py\n"
            "Current state/errors: test_parse fails with KeyError\nNext steps: fix KeyError\nTodo: none"
        )
        self.exc = exc
        self.delay = delay
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, prompt_messages):
        self.calls.append(prompt_messages)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.text


def call(cid: str, name: str, **args: Any) -> dict[str, Any]:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def step(cid: str, name: str, output: str, text: str = "", **args: Any) -> list[dict[str, Any]]:
    return [
        {"role": "assistant", "content": text, "tool_calls": [call(cid, name, **args)]},
        {"role": "tool", "tool_call_id": cid, "name": name, "content": output},
    ]


def big_output(n_lines: int, prefix: str = "line") -> str:
    return "\n".join(f"{prefix} {i}: some output text for testing purposes" for i in range(n_lines))


def last_user_content(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m["content"]
    return None


def strategies(events: list[Compaction]) -> list[str]:
    return [e.strategy for e in events]


def kept_verbatim(original: str, messages: list[dict[str, Any]]) -> bool:
    """``original`` (a user message) is present verbatim, outside any summary block."""
    content = last_user_content(messages)
    if not isinstance(content, str):
        return False
    task, summary, rest = ctx._split_head(content)
    if summary is None:
        return content.endswith(original)
    return (task or "").startswith(original) or (rest or "").endswith(original)


SYSTEM = {"role": "system", "content": "You are a coding agent. Be brief."}


# --------------------------------------------------------------------------- #
# estimate_tokens
# --------------------------------------------------------------------------- #


def test_estimate_basic_rules():
    assert estimate_tokens("") == 0
    assert estimate_tokens("1234567890") == 10          # one token per digit
    assert estimate_tokens("ąęłóśżźćń") == 9             # one per non-ASCII char
    assert estimate_tokens("abcdefg") == 2               # 7 / 3.5
    assert estimate_tokens("....") == 2                  # 0.5 per punctuation
    text = "def foo(x):\n    return x * 2  # double"
    assert estimate_tokens(text) >= len(text) / 4


def test_estimate_is_subadditive_and_monotone():
    rng = random.Random(1)
    for _ in range(200):
        a = "".join(rng.choice("abc xyz 012 .,;ąę\n") for _ in range(rng.randint(0, 200)))
        b = "".join(rng.choice("abc xyz 012 .,;ąę\n") for _ in range(rng.randint(0, 200)))
        assert estimate_tokens(a + b) <= estimate_tokens(a) + estimate_tokens(b) + 1
        assert estimate_tokens(a + b) >= max(estimate_tokens(a), estimate_tokens(b)) - 1


def test_estimate_messages_counts_calls_and_ignores_reasoning():
    plain = {"role": "assistant", "content": "hello"}
    with_reasoning = dict(plain, reasoning="x" * 5000)
    assert estimate_message_tokens(with_reasoning) == estimate_message_tokens(plain)
    with_call = dict(plain, tool_calls=[call("1", "read_file", path="src/app.py")])
    assert estimate_message_tokens(with_call) > estimate_message_tokens(plain) + 5
    assert estimate_messages_tokens([plain], TOOLS) > estimate_messages_tokens([plain])
    image = {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]}
    assert estimate_message_tokens(image) >= 256


@pytest.mark.real_model
def test_estimate_is_conservative_against_real_tokenizer():
    from lc_testing import model_path

    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(model_path("mlx_tiny") / "tokenizer.json"))
    repo = Path(__file__).resolve().parents[2]
    samples = [
        (repo / "localcode" / "config.py").read_text(),
        (repo / "localcode" / "runtime" / "base.py").read_text(),
        "The quick brown fox jumps over the lazy dog. " * 30,
        "Zażółć gęślą jaźń. Modele lokalne mają małe okna kontekstu, więc trzeba oszczędzać tokeny. " * 8,
        json.dumps({"path": "src/app.py", "items": [{"content": "Fix bug", "status": "pending"}] * 10}),
        "tests/test_x.py::test_a PASSED\ntests/test_x.py::test_b FAILED\nE   AssertionError: assert 1 == 2\n" * 20,
    ]
    for text in samples:
        actual = len(tok.encode(text, add_special_tokens=False).ids)
        assert estimate_tokens(text) >= 0.9 * actual, text[:60]


# --------------------------------------------------------------------------- #
# truncate
# --------------------------------------------------------------------------- #

MARKER_RE = re.compile(r"\[… (\d+) lines? \((\d+) chars\) omitted(?: — [^…]*)? …\]")


def test_truncate_short_text_unchanged():
    assert truncate("abc\ndef", 100) == "abc\ndef"
    assert truncate("", 0) == ""


def test_truncate_marker_counts_whole_lines():
    lines = [f"line number {i:03d}" for i in range(100)]
    text = "\n".join(lines)
    out = truncate(text, 400)
    assert len(out) <= 400
    m = MARKER_RE.search(out)
    assert m, out
    head, tail = out.split(m.group(0))
    head_lines = [x for x in head.split("\n") if x]
    tail_lines = [x for x in tail.split("\n") if x]
    assert head_lines == lines[: len(head_lines)]
    assert tail_lines == lines[len(lines) - len(tail_lines):]
    assert int(m.group(1)) == 100 - len(head_lines) - len(tail_lines)
    kept = sum(len(x) for x in head_lines + tail_lines) + len(head_lines) - 1 + len(tail_lines) - 1
    assert abs(int(m.group(2)) - (len(text) - kept)) <= 2


def test_truncate_head_ratio_and_hint():
    text = "\n".join(f"row {i}" for i in range(500))
    head_heavy = truncate(text, 300, head_ratio=0.9, hint="use offset/limit")
    tail_heavy = truncate(text, 300, head_ratio=0.1)
    assert "— use offset/limit …]" in head_heavy
    assert head_heavy.index("[…") > tail_heavy.index("[…")
    assert len(head_heavy) <= 300 and len(tail_heavy) <= 300


def test_truncate_single_long_line():
    text = "x" * 5000
    out = truncate(text, 200)
    assert len(out) <= 200
    assert "[… 1 line (" in out
    assert out.startswith("xxx") and out.endswith("xxx")


def test_truncate_tiny_budget_is_plain_cut():
    assert truncate("abcdefghij" * 10, 5) == "abcde"


@pytest.mark.parametrize("seed", range(20))
def test_truncate_never_exceeds_budget(seed):
    rng = random.Random(seed)
    for _ in range(50):
        n_lines = rng.randint(1, 200)
        text = "\n".join("".join(rng.choice("ab cdą1{}") for _ in range(rng.randint(0, 300))) for _ in range(n_lines))
        limit = rng.randint(0, len(text) + 10)
        out = truncate(text, limit, head_ratio=rng.random(), hint=rng.choice([None, "hint text"]))
        assert len(out) <= max(limit, 0)
        if len(text) <= limit:
            assert out == text


# --------------------------------------------------------------------------- #
# Budgets and limits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n_ctx,reserve",
    [(512, 256), (1024, 512), (2048, 512), (4096, 819), (8192, 1638), (32768, 6553), (65536, 8192), (262144, 8192)],
)
def test_default_budgets(n_ctx, reserve):
    cm = ContextManager(n_ctx, count_tokens)
    assert cm.reserve_output == reserve
    assert cm.usable == n_ctx - reserve
    assert cm.soft_limit == int(0.8 * cm.usable)


def test_custom_budget_and_threshold():
    cm = ContextManager(8192, count_tokens, reserve_output=1000, compact_threshold=0.5)
    assert cm.usable == 7192
    assert cm.soft_limit == 3596
    assert cm.needs_compaction(3597) and not cm.needs_compaction(3596)
    assert cm.output_budget(8000) == 192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_ctx": 100},
        {"n_ctx": 4096, "compact_threshold": 1.5},
        {"n_ctx": 4096, "compact_threshold": 0.0},
        {"n_ctx": 4096, "compact_threshold": "abc"},
        {"n_ctx": 4096, "reserve_output": -1},
        {"n_ctx": 4096, "reserve_output": 4096},
    ],
)
def test_invalid_arguments_raise_polish_errors(kwargs):
    n_ctx = kwargs.pop("n_ctx")
    with pytest.raises(ValueError) as err:
        ContextManager(n_ctx, count_tokens, **kwargs)
    assert any(word in str(err.value) for word in ("kontekstu", "Rezerwa", "Próg"))


def test_counter_must_be_callable():
    with pytest.raises(TypeError):
        ContextManager(4096, 42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "n_ctx,chars,lines,matches",
    [(4096, 3000, 60, 20), (8192, 6000, 120, 40), (16384, 10000, 200, 60), (32768, 16000, 300, 100), (131072, 24000, 400, 150)],
)
def test_limits_scale_with_context(n_ctx, chars, lines, matches):
    lim = ContextManager(n_ctx, count_tokens).limits()
    assert (lim.max_output_chars, lim.read_default_lines, lim.grep_max_matches) == (chars, lines, matches)
    assert lim.bash_timeout_s > 0 and lim.list_max_entries > 0


def test_limits_tiny_window_caps_output_share():
    cm = ContextManager(2048, count_tokens)
    assert cm.limits().max_output_chars <= int(cm.usable * 0.35 * 3.5)
    assert limits_for(2048).max_output_chars == 3000
    out = cm.truncate_tool_output(big_output(500))
    assert len(out) <= cm.limits().max_output_chars and "use offset/limit or grep" in out


def test_keep_tool_results_depends_on_window():
    assert ContextManager(8192, count_tokens).keep_tool_results == 2
    assert ContextManager(8193, count_tokens).keep_tool_results == 5


# --------------------------------------------------------------------------- #
# fit: stages
# --------------------------------------------------------------------------- #


async def test_under_limit_returns_same_objects_without_events():
    msgs = [SYSTEM, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    counter = FakeCounter()
    cm = ContextManager(8192, counter)
    out, events = await cm.fit(msgs, TOOLS)
    assert events == []
    assert out == msgs and all(a is b for a, b in zip(out, msgs))
    assert out is not msgs  # a new list, the caller's list is never touched
    assert counter.calls == 1


async def test_mask_stage_only():
    msgs = [SYSTEM, {"role": "user", "content": "Find the bug"}]
    for i in range(8):
        msgs += step(f"c{i}", "read_file", big_output(60, f"file{i}"), path=f"src/f{i}.py")
    cm = ContextManager(8192, FakeCounter())  # K = 2
    before = count_tokens(msgs, TOOLS)
    assert before > cm.soft_limit
    out, events = await cm.fit(msgs, TOOLS)
    assert strategies(events) == ["mask_tool_results"]
    assert events[0].before == before and events[0].after == count_tokens(out, TOOLS) <= cm.soft_limit
    tool_msgs = [m for m in out if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == [f"c{i}" for i in range(8)]
    for m in tool_msgs[:-2]:
        assert m["content"].startswith("[old output of read_file removed to save context — call the tool again if you need it]")
        assert m["content"].endswith(": some output text for testing purposes")  # first line kept
        assert m["name"] == "read_file"
    for orig, m in zip([x for x in msgs if x["role"] == "tool"][-2:], tool_msgs[-2:]):
        assert m is orig
    assert structure_problems(out) == []
    # every assistant message is the very same object (prefix before the first mask is stable)
    assert all(a is b for a, b in zip(out, msgs) if a["role"] != "tool")


async def test_masking_uses_call_name_when_tool_message_has_no_name():
    msgs = [SYSTEM, {"role": "user", "content": "go"}]
    for i in range(8):
        pair = step(f"c{i}", "grep", big_output(80), pattern="foo")
        pair[1].pop("name")
        msgs += pair
    cm = ContextManager(4096, FakeCounter())
    out, events = await cm.fit(msgs)
    assert "mask_tool_results" in strategies(events)
    masked = [m for m in out if m["role"] == "tool" and "removed to save context" in m["content"]]
    assert masked and all("[old output of grep removed" in m["content"] for m in masked)


async def test_masking_is_prefix_stable_when_history_is_adopted():
    msgs = [SYSTEM, {"role": "user", "content": "Find the bug"}]
    for i in range(8):
        msgs += step(f"c{i}", "read_file", big_output(60), path=f"src/f{i}.py")
    cm = ContextManager(8192, FakeCounter())
    out, events = await cm.fit(msgs, TOOLS)
    assert events
    grown = out + step("c9", "bash", "exit code 0\nok", command="ls")
    out2, events2 = await cm.fit(grown, TOOLS)
    assert events2 == []
    assert all(a is b for a, b in zip(out2, grown))


async def test_shrink_stage_for_long_old_assistant_texts_and_arguments():
    long_text = "\n".join(f"Explanation paragraph {i} " + "word " * 30 for i in range(40))
    long_file = "\n".join(f"def function_{i}(x):\n    return x + {i}" for i in range(300))
    msgs = [SYSTEM, {"role": "user", "content": "Write the module"}]
    msgs += step("w1", "write_file", "ok", text=long_text, path="src/mod.py", content=long_file)
    msgs += [{"role": "assistant", "content": long_text}, {"role": "user", "content": "Now explain it"}]
    msgs += [{"role": "assistant", "content": "It adds numbers."}, {"role": "user", "content": "Thanks, and the tests?"}]
    cm = ContextManager(8192, FakeCounter())
    assert count_tokens(msgs) > cm.soft_limit
    out, events = await cm.fit(msgs)
    assert strategies(events) == ["shrink_messages"]
    assert count_tokens(out) <= cm.soft_limit
    shrunk_call = out[2]
    args = shrunk_call["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict) and args["path"] == "src/mod.py"
    assert len(args["content"]) < len(long_file) and "omitted" in args["content"]
    assert len(shrunk_call["content"]) < len(long_text)
    assert msgs[2]["tool_calls"][0]["function"]["arguments"]["content"] == long_file  # input untouched
    assert out[-1] is msgs[-1] and out[-2] is msgs[-2]
    assert structure_problems(out) == []


def many_small_turns(n: int) -> list[dict[str, Any]]:
    msgs = [SYSTEM, {"role": "user", "content": "Fix the parser bug in src/parser.py"}]
    for i in range(n):
        msgs += step(f"r{i}", "read_file", f"{i}: small file content", text=f"Reading part {i}", path=f"src/p{i}.py")
        msgs += step(f"b{i}", "bash", f"exit code {i % 2}\nran", command=f"pytest -k t{i}")
        msgs.append({"role": "assistant", "content": f"Checked part {i}."})
        msgs.append({"role": "user", "content": f"Continue with part {i + 1}"})
    return msgs


async def test_summarize_stage_uses_model_summary():
    msgs = many_small_turns(40)
    summarizer = RecordingSummarizer()
    cm = ContextManager(4096, FakeCounter(), summarizer=summarizer)
    out, events = await cm.fit(msgs, TOOLS)
    assert "summarize" in strategies(events)
    assert len(summarizer.calls) == 1
    prompt = summarizer.calls[0]
    assert prompt[0]["role"] == "system" and "Files touched" in prompt[0]["content"]
    assert "CALL read_file" in prompt[1]["content"] and "RESULT bash" in prompt[1]["content"]
    assert estimate_messages_tokens(prompt) <= cm.usable
    head = out[1]
    assert head["role"] == "user"
    assert head["content"].startswith("Fix the parser bug in src/parser.py\n\n" + SUMMARY_HEADER)
    assert "test_parse fails with KeyError" in head["content"] and SUMMARY_FOOTER in head["content"]
    assert out[0] is SYSTEM
    assert last_user_content(out).endswith("Continue with part 40")
    assert count_tokens(out, TOOLS) <= cm.soft_limit
    assert cm.last_summary_source == "model"
    assert structure_problems(out) == []


async def test_summarize_fallback_when_summarizer_fails():
    msgs = [SYSTEM, {"role": "user", "content": "Make the tests pass"}]
    msgs += step("t1", "todo_write", "ok", todos=[{"content": "read code", "status": "done"},
                                                  {"content": "fix bug", "status": "in_progress"},
                                                  {"content": "run tests", "status": "pending"}])
    for i in range(30):
        msgs += step(f"r{i}", "read_file", big_output(10), path=f"src/mod{i}.py")
        msgs += step(f"e{i}", "edit_file", "Error: old_string not found" if i == 3 else "ok",
                     path=f"src/mod{i}.py", old_string="a", new_string="b")
        msgs += step(f"b{i}", "bash", f"exit code {1 if i < 29 else 0}\n...", command="pytest -q")
        msgs += step(f"g{i}", "grep", "src/x.py:1: foo", pattern="foo", path="src")
    summarizer = RecordingSummarizer(exc=RuntimeError("engine crashed"))
    cm = ContextManager(8192, FakeCounter(), summarizer=summarizer)
    out, events = await cm.fit(msgs, TOOLS)
    assert "summarize" in strategies(events)
    assert cm.last_summary_source == "fallback" and "engine crashed" in cm.last_summary_error
    head = out[1]["content"]
    assert "Files changed:" in head and "(edit_file)" in head and "src/mod3.py (edit_file, failed)" in head
    assert "(+" in head  # older entries are counted, not listed
    assert "`pytest -q` → exit 1 (×" in head
    assert "Errors: edit_file: Error: old_string not found" in head
    assert "Todo: [x] read code; [~] fix bug; [ ] run tests" in head
    assert "Files read:" in head
    _, summary, _ = ctx._split_head(head)
    assert estimate_tokens(summary) <= int(cm.summary_tokens * 1.5)
    assert structure_problems(out) == []
    assert count_tokens(out, TOOLS) <= cm.soft_limit


async def test_fallback_summary_prioritizes_within_tiny_budget():
    msgs = [SYSTEM, {"role": "user", "content": "Do it"}]
    msgs += step("t1", "todo_write", "ok", todos=[{"content": "only task", "status": "pending"}])
    for i in range(60):
        msgs += step(f"r{i}", "read_file", big_output(5), path=f"src/some/deep/directory/module_{i}.py")
        msgs += step(f"e{i}", "edit_file", "ok", path=f"src/some/deep/directory/module_{i}.py")
    cm = ContextManager(2048, FakeCounter())
    out, events = await cm.fit(msgs)
    assert "summarize" in strategies(events)
    _, summary, _ = ctx._split_head(out[1]["content"])
    assert summary.startswith("(Automatic summary")
    assert "Files changed:" in summary and "Todo: [ ] only task" in summary
    assert estimate_tokens(summary) <= int(cm.summary_tokens * 1.5)


async def test_summarize_fallback_without_summarizer():
    cm = ContextManager(4096, FakeCounter())
    out, events = await cm.fit(many_small_turns(40))
    assert "summarize" in strategies(events)
    assert cm.last_summary_source == "fallback"
    assert "User messages:" in out[1]["content"] and "Continue with part" in out[1]["content"]


async def test_summarize_fallback_on_timeout():
    summarizer = RecordingSummarizer(delay=5)
    cm = ContextManager(4096, FakeCounter(), summarizer=summarizer, summarizer_timeout=0.05)
    out, events = await cm.fit(many_small_turns(40))
    assert "summarize" in strategies(events)
    assert cm.last_summary_source == "fallback" and "Timeout" in cm.last_summary_error


@pytest.mark.parametrize("raw", ["", "   ", "<think>I should summarize… let me think" + " more" * 50, None, 42])
async def test_summarize_fallback_on_useless_output(raw):
    summarizer = RecordingSummarizer()
    summarizer.text = raw  # type: ignore[assignment]
    cm = ContextManager(4096, FakeCounter(), summarizer=summarizer)
    out, events = await cm.fit(many_small_turns(40))
    assert "summarize" in strategies(events)
    assert cm.last_summary_source == "fallback"


async def test_summary_is_cleaned_and_capped():
    raw = "<think>internal reasoning here</think>\n```\nGoal: cleaned goal\n" + "Done so far: step\n" * 2000 + "```"
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer(raw))
    out, _ = await cm.fit(many_small_turns(40))
    head = out[1]["content"]
    assert "internal reasoning" not in head and "```" not in head
    assert "Goal: cleaned goal" in head
    _, summary, _ = ctx._split_head(head)
    assert estimate_tokens(summary) <= int(cm.summary_tokens * 1.5)


async def test_sync_summarizer_is_supported():
    def summarizer(prompt):
        return "Goal: sync summary works fine\nTodo: none"

    cm = ContextManager(4096, FakeCounter(), summarizer=summarizer)  # type: ignore[arg-type]
    out, _ = await cm.fit(many_small_turns(40))
    assert "sync summary works fine" in out[1]["content"] and cm.last_summary_source == "model"


async def test_resummarize_replaces_previous_summary():
    summarizer = RecordingSummarizer()
    cm = ContextManager(4096, FakeCounter(), summarizer=summarizer)
    out, _ = await cm.fit(many_small_turns(40))
    grown = list(out)
    for i in range(40, 80):
        grown += step(f"x{i}", "read_file", big_output(8), path=f"src/q{i}.py")
        grown += [{"role": "assistant", "content": f"Checked {i}."}, {"role": "user", "content": f"Next {i}"}]
    summarizer.text = "Goal: second summary\nDone so far: everything\nTodo: none"
    out2, events2 = await cm.fit(grown)
    assert "summarize" in strategies(events2)
    assert len(summarizer.calls) == 2
    assert "Previous summary" in summarizer.calls[1][1]["content"]
    assert "test_parse fails with KeyError" in summarizer.calls[1][1]["content"]
    head = out2[1]["content"]
    assert head.count(SUMMARY_HEADER) == 1 and head.count(SUMMARY_FOOTER) == 1
    assert head.startswith("Fix the parser bug in src/parser.py\n\n")
    assert "second summary" in head and "test_parse fails" not in head
    assert last_user_content(out2).endswith("Next 79")
    assert structure_problems(out2) == []


async def test_long_single_turn_keeps_task_verbatim():
    task = "Refactor the whole project. " + "Requirement detail. " * 400  # long, but it is also the last user message
    msgs = [SYSTEM, {"role": "user", "content": task}]
    for i in range(200):
        msgs += step(f"c{i}", "read_file", big_output(20), path=f"f{i}.py")
    cm = ContextManager(16384, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert "summarize" in strategies(events)
    assert out[1]["content"].startswith(task + "\n\n" + SUMMARY_HEADER)
    assert out[2]["role"] == "assistant"
    assert structure_problems(out) == []


async def test_old_long_task_is_trimmed_when_not_latest():
    task = "Initial task. " + "Pasted log line with details. " * 2000
    msgs = [SYSTEM, {"role": "user", "content": task}, {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Next: fix the parser"}]
    msgs += many_small_turns(30)[2:]
    cm = ContextManager(8192, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert "shrink_messages" in strategies(events)
    assert "truncate" not in strategies(events)
    task_part, _, _ = ctx._split_head(out[1]["content"])
    assert task_part.startswith("Initial task.") and len(task_part) <= cm._task_chars
    assert "omitted" in task_part
    assert count_tokens(out) <= cm.soft_limit
    assert last_user_content(out).endswith("Continue with part 30")


async def test_latest_user_message_in_long_turn_is_pulled_next_to_summary():
    msgs = many_small_turns(5)
    msgs.append({"role": "user", "content": "NEW REQUEST: add logging everywhere"})
    for i in range(40):
        msgs += step(f"n{i}", "read_file", big_output(15), path=f"src/n{i}.py")
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert "summarize" in strategies(events)
    users = [m for m in out if m["role"] == "user"]
    assert len(users) == 1
    assert users[0]["content"].endswith(SUMMARY_FOOTER + "\n\nNEW REQUEST: add logging everywhere")
    assert out[2]["role"] == "assistant"
    assert structure_problems(out) == []
    # the merged latest message survives a second compaction verbatim
    grown = list(out)
    for i in range(40, 80):
        grown += step(f"n{i}", "read_file", big_output(15), path=f"src/n{i}.py")
    out2, _ = await cm.fit(grown)
    users2 = [m for m in out2 if m["role"] == "user"]
    assert len(users2) == 1 and users2[0]["content"].endswith("NEW REQUEST: add logging everywhere")
    assert users2[0]["content"].count(SUMMARY_HEADER) == 1


async def test_nudge_does_not_evict_the_human_request():
    msgs = many_small_turns(3)
    msgs.append({"role": "user", "content": "HUMAN: please rename foo to bar"})
    for i in range(40):
        msgs += step(f"n{i}", "grep", big_output(15), pattern="foo")
    msgs.append({"role": "user", "content": REPEATED_CALL_NUDGE})
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert events
    text = out[1]["content"]
    assert "HUMAN: please rename foo to bar" in text
    assert last_user_content(out).endswith(REPEATED_CALL_NUDGE)
    assert structure_problems(out) == []


async def test_drop_oldest_when_counter_disagrees_with_estimates():
    # The newest tool results cost 10x more than estimated, so the summary plan keeps too many
    # recent units and the drop stage has to finish the job.
    msgs = [SYSTEM]
    for i in range(60):
        msgs += [{"role": "user", "content": f"Question {i}"},
                 {"role": "assistant", "content": f"Answer {i}. " + "Short explanation text. " * 16}]
    msgs.append({"role": "user", "content": "Now investigate the crash"})
    for i in range(4):
        msgs += step(f"c{i}", "read_file", big_output(18) + "\nHEAVY", path=f"f{i}.py")
    counter = FakeCounter(weight=lambda m: 10.0 if "HEAVY" in str(m.get("content")) else 1.0)
    cm = ContextManager(8192, counter, summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    names = strategies(events)
    assert names == sorted(names, key=STRATEGIES.index)
    assert "summarize" in names and "drop_oldest" in names
    assert await counter(out) <= cm.soft_limit
    assert DROPPED_NOTE in out[1]["content"]
    assert kept_verbatim("Now investigate the crash", out)
    assert out[-1] is msgs[-1]
    assert structure_problems(out) == []


async def test_drop_oldest_without_any_user_message():
    msgs = [SYSTEM]
    for i in range(30):
        msgs.append({"role": "assistant", "content": f"Note {i} " + "text " * 200})
        msgs.append({"role": "tool", "tool_call_id": f"orphan{i}", "content": "x"})  # orphan results stay glued
    cm = ContextManager(4096, FakeCounter())
    out, events = await cm.fit(msgs)
    assert "drop_oldest" in strategies(events)
    assert out[0] is SYSTEM and out[-1] is msgs[-1] and out[-2] is msgs[-2]
    assert count_tokens(out) <= cm.soft_limit


async def test_truncate_stage_for_huge_last_tool_output():
    msgs = [SYSTEM, {"role": "user", "content": "cat the log"}] + step("c1", "bash", big_output(5000), command="cat log")
    cm = ContextManager(4096, FakeCounter())
    out, events = await cm.fit(msgs, TOOLS)
    assert strategies(events)[-1] == "truncate"
    assert count_tokens(out, TOOLS) <= cm.usable
    assert "cut to fit the context window" in out[-1]["content"]
    assert out[-1]["tool_call_id"] == "c1" and out[1]["content"] == "cat the log"
    assert structure_problems(out) == []


async def test_truncate_stage_for_huge_last_user_message():
    huge = "Please analyse this log:\n" + big_output(6000)
    msgs = [SYSTEM, {"role": "user", "content": huge}]
    cm = ContextManager(2048, FakeCounter())
    out, events = await cm.fit(msgs, TOOLS)
    assert strategies(events) == ["truncate"]
    assert count_tokens(out, TOOLS) <= cm.usable
    assert out[1]["content"].startswith("Please analyse this log:")
    assert out[0] is SYSTEM


async def test_truncate_stage_huge_system_prompt_as_last_resort():
    msgs = [{"role": "system", "content": "Rules:\n" + big_output(3000)}, {"role": "user", "content": "hi"}]
    cm = ContextManager(4096, FakeCounter())
    out, events = await cm.fit(msgs)
    assert strategies(events) == ["truncate"]
    assert count_tokens(out) <= cm.usable
    assert out[0]["role"] == "system" and out[0]["content"].startswith("Rules:")
    assert out[1] is msgs[1]


async def test_on_overflow_is_aggressive():
    msgs = [SYSTEM, {"role": "user", "content": "Investigate"}]
    for i in range(12):
        msgs += step(f"c{i}", "read_file", big_output(25), path=f"f{i}.py")
    cm = ContextManager(16384, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert events == []  # under the soft limit according to the counter
    out2, events2 = await cm.on_overflow(out)
    assert events2
    assert count_tokens(out2) <= int(cm.usable * 0.6)
    unmasked = [m for m in out2 if m["role"] == "tool" and "removed to save context" not in m["content"]]
    assert len(unmasked) <= 1
    assert structure_problems(out2) == []


async def test_manual_compact_summarizes_under_limit():
    msgs = many_small_turns(6)
    summarizer = RecordingSummarizer()
    cm = ContextManager(32768, FakeCounter(), summarizer=summarizer)
    assert count_tokens(msgs) < cm.soft_limit
    out, events = await cm.compact(msgs, TOOLS)
    assert "summarize" in strategies(events) and len(summarizer.calls) == 1
    assert SUMMARY_HEADER in out[1]["content"] and len(out) < len(msgs)
    assert structure_problems(out) == []


async def test_counter_failure_falls_back_to_estimate():
    async def broken(messages, tools=None):
        raise RuntimeError("engine is gone")

    cm = ContextManager(4096, broken)
    msgs = [SYSTEM, {"role": "user", "content": "go"}]
    for i in range(10):
        msgs += step(f"c{i}", "read_file", big_output(60), path=f"f{i}.py")
    out, events = await cm.fit(msgs)
    assert events
    assert estimate_messages_tokens(out) <= cm.soft_limit
    usage = await cm.usage(out)
    assert usage.used == estimate_messages_tokens(out, [])


async def test_sync_counter_is_supported():
    cm = ContextManager(4096, count_tokens)
    msgs = [SYSTEM, {"role": "user", "content": "go"}]
    for i in range(10):
        msgs += step(f"c{i}", "read_file", big_output(60), path=f"f{i}.py")
    out, events = await cm.fit(msgs, TOOLS)
    assert events and count_tokens(out, TOOLS) <= cm.soft_limit


async def test_multimodal_content_is_tolerated():
    msgs = [SYSTEM, {"role": "user", "content": [{"type": "text", "text": "What is on this screenshot?"},
                                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]
    for i in range(20):
        msgs += step(f"c{i}", "read_file", big_output(40), path=f"f{i}.py")
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert events and structure_problems(out) == []
    head = out[1]["content"]
    assert isinstance(head, list) and head[0]["text"] == "What is on this screenshot?"
    assert any(p.get("type") == "image_url" for p in head)
    assert any(isinstance(p.get("text"), str) and p["text"].startswith(SUMMARY_HEADER) for p in head)


# --------------------------------------------------------------------------- #
# usage
# --------------------------------------------------------------------------- #


async def test_usage_breakdown_and_caching():
    counter = FakeCounter()
    cm = ContextManager(8192, counter)
    msgs = [SYSTEM, {"role": "user", "content": "hello " * 50}, {"role": "assistant", "content": "hi " * 30}]
    upd = await cm.usage(msgs, TOOLS)
    assert isinstance(upd, ContextUpdate)
    assert upd.used == count_tokens(msgs, TOOLS)
    assert upd.limit == 8192 and upd.reserve == cm.reserve_output
    assert upd.system > 0 and upd.tools > 0 and upd.history > 0
    assert upd.system + upd.tools + upd.history == upd.used
    calls = counter.calls
    await cm.usage(msgs + [{"role": "user", "content": "more"}], TOOLS)
    assert counter.calls == calls + 1  # fixed costs are cached
    upd2 = await cm.usage(msgs, TOOLS, system_tokens=7)
    assert upd2.system == 7
    d = upd.to_dict()
    assert d["type"] == "context" and {"used", "limit", "system", "tools", "history", "reserve"} <= set(d)


async def test_usage_without_system_message_or_tools():
    cm = ContextManager(4096, FakeCounter())
    upd = await cm.usage([{"role": "user", "content": "hi"}])
    assert upd.system == 0 and upd.tools == 0 and upd.history == upd.used


# --------------------------------------------------------------------------- #
# structure_problems
# --------------------------------------------------------------------------- #


def test_structure_problems_detects_violations():
    ok = [SYSTEM, {"role": "user", "content": "a"}] + step("1", "bash", "x", command="ls") + [
        {"role": "assistant", "content": "done"}]
    assert structure_problems(ok) == []
    assert structure_problems(ok + [{"role": "assistant", "content": "again"}])
    assert structure_problems([SYSTEM, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert structure_problems([SYSTEM, {"role": "assistant", "content": "a"}])
    assert structure_problems([SYSTEM, {"role": "user", "content": "a"}, {"role": "tool", "tool_call_id": "9", "content": "x"}])
    bad_id = [SYSTEM, {"role": "user", "content": "a"}] + step("1", "bash", "x")
    bad_id[-1] = dict(bad_id[-1], tool_call_id="2")
    assert structure_problems(bad_id)
    missing = [SYSTEM, {"role": "user", "content": "a"},
               {"role": "assistant", "content": "", "tool_calls": [call("1", "bash")]},
               {"role": "user", "content": "b"}]
    assert structure_problems(missing)
    pending = [SYSTEM, {"role": "user", "content": "a"}, {"role": "assistant", "content": "", "tool_calls": [call("1", "bash")]}]
    assert structure_problems(pending) == []
    assert structure_problems([{"role": "user", "content": "a"}, SYSTEM])


# --------------------------------------------------------------------------- #
# Property-style invariants over random conversations
# --------------------------------------------------------------------------- #

WORDS = ("the parser function returns value error file test module class import ąę łódź "
         "config path build 42 1337 {} [] () => :: self.x = y; # TODO").split()
TOOL_NAMES = ("read_file", "bash", "grep", "edit_file", "write_file", "list_dir", "glob", "todo_write")


def rand_text(rng: random.Random, n_chars: int) -> str:
    out: list[str] = []
    size = 0
    while size < n_chars:
        line = " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 12)))
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)[:n_chars]


def rand_args(rng: random.Random, name: str) -> dict[str, Any]:
    if name == "bash":
        return {"command": rng.choice(["pytest -q", "ls -la", "git status", "make build"])}
    if name in ("edit_file", "write_file"):
        return {"path": f"src/m{rng.randint(0, 9)}.py", "content": rand_text(rng, rng.choice([10, 200, 4000]))}
    if name == "todo_write":
        return {"todos": [{"content": f"task {k}", "status": rng.choice(["pending", "in_progress", "done"])}
                          for k in range(rng.randint(1, 4))]}
    if name in ("grep", "glob"):
        return {"pattern": rng.choice(["foo", "*.py", "class .*"]), "path": "src"}
    return {"path": f"src/f{rng.randint(0, 20)}.py"}


def rand_output(rng: random.Random, name: str) -> str:
    size = rng.choice([5, 50, 400, 2000, 8000, 20000])
    prefix = f"exit code {rng.choice([0, 1, 2])}\n" if name == "bash" else ""
    return prefix + rand_text(rng, size)


def rand_conversation(rng: random.Random) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "You are a test agent. " * rng.randint(3, 30)}]
    call_n = 0
    for _turn in range(rng.randint(1, 6)):
        msgs.append({"role": "user", "content": rand_text(rng, rng.choice([10, 100, 600, 3000]))})
        steps = rng.randint(0, 10)
        for _ in range(steps):
            calls = []
            for _ in range(rng.choice([1, 1, 1, 2, 3])):
                call_n += 1
                name = rng.choice(TOOL_NAMES)
                calls.append(call(f"call_{call_n}", name, **rand_args(rng, name)))
            msgs.append({"role": "assistant", "content": rand_text(rng, rng.choice([0, 0, 40, 300, 3000])),
                         "tool_calls": calls, "reasoning": "thinking " * rng.randint(0, 50)})
            for c in calls:
                name = c["function"]["name"]
                msgs.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": rand_output(rng, name)})
        ending = rng.random()
        if steps == 0 or ending < 0.7:
            msgs.append({"role": "assistant", "content": rand_text(rng, rng.choice([5, 200, 2500]))})
        elif ending < 0.8 and msgs[-1]["role"] == "tool":
            msgs.append({"role": "user", "content": rng.choice([REPEATED_CALL_NUDGE, EMPTY_RESPONSE_NUDGE])})
            msgs.append({"role": "assistant", "content": "ok"})
    if rng.random() < 0.5:
        msgs.append({"role": "user", "content": rand_text(rng, rng.choice([10, 300, 2000]))})
    return msgs


def make_summarizer(rng: random.Random):
    kind = rng.choice(["none", "good", "fail", "empty", "huge", "think"])
    if kind == "none":
        return None
    texts = {
        "good": "Goal: g\nDone so far: stuff\nFiles touched: a.py: edited\nNext steps: more\nTodo: none",
        "empty": "",
        "huge": "Done so far: " + "very long summary " * 3000,
        "think": "<think>hmm</think>Goal: after thinking\nTodo: none",
    }
    if kind == "fail":
        return RecordingSummarizer(exc=RuntimeError("boom"))
    return RecordingSummarizer(texts[kind])


def minimal_context(cm: ContextManager, out: list[dict[str, Any]]) -> bool:
    """True when stage 4 has nothing left to drop."""
    body = ctx._body_start(out)
    units = ctx._units(out, body)
    t, pinned = cm._pinned_users(out)
    droppable = [u for u in units[:-1] if u[0] != t and u[0] not in pinned]
    return not droppable


async def check_fit(rng: random.Random, n_ctx: int) -> None:
    msgs = rand_conversation(rng)
    assert structure_problems(msgs) == [], structure_problems(msgs)
    snapshot = copy.deepcopy(msgs)
    tools = TOOLS if rng.random() < 0.5 else None
    counter = FakeCounter()
    cm = ContextManager(n_ctx, counter, summarizer=make_summarizer(rng), summarizer_timeout=5)
    out, events = await cm.fit(msgs, tools)

    assert msgs == snapshot, "input was mutated"
    problems = structure_problems(out)
    assert problems == [], problems
    assert out[0] is msgs[0], "system message changed"

    names = strategies(events)
    assert names == sorted(names, key=STRATEGIES.index)
    assert len(set(names)) == len(names)
    before = count_tokens(msgs, tools)
    final = count_tokens(out, tools)
    if events:
        assert events[0].before == before and events[-1].after == final
        assert all(e.before > e.after for e in events)
        assert all(a.after == b.before for a, b in zip(events, events[1:]))
    else:
        assert before <= cm.soft_limit and all(a is b for a, b in zip(out, msgs)) and len(out) == len(msgs)

    assert final <= cm.usable, (final, cm.usable, names)
    if final > cm.soft_limit:
        assert minimal_context(cm, out), names

    last_in = last_user_content(msgs)
    last_out = last_user_content(out)
    if "truncate" not in names:
        assert isinstance(last_out, str) and kept_verbatim(last_in, out)

    # tool pairing: every result follows its caller, ids and names preserved
    calls_in = {c["id"]: c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []}
    for i, m in enumerate(out):
        if m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            following = [x["tool_call_id"] for x in out[i + 1: i + 1 + len(ids)]]
            assert following == ids
        if m["role"] == "tool":
            assert calls_in[m["tool_call_id"]] == m["name"]

    # fitting the result again is a no-op when it is under the soft limit (stable prefix)
    if final <= cm.soft_limit:
        again, ev2 = await cm.fit(out, tools)
        assert ev2 == [] and all(a is b for a, b in zip(again, out))


@pytest.mark.parametrize("seed", range(80))
async def test_fit_invariants_random_conversations(seed):
    rng = random.Random(seed)
    for n_ctx in (2048, 4096, 8192, 16384, 32768):
        if rng.random() < 0.6:
            await check_fit(rng, n_ctx)


@pytest.mark.parametrize("seed", range(40))
async def test_overflow_and_compact_invariants(seed):
    rng = random.Random(10_000 + seed)
    msgs = rand_conversation(rng)
    n_ctx = rng.choice([4096, 8192, 32768])
    cm = ContextManager(n_ctx, FakeCounter(), summarizer=make_summarizer(rng), summarizer_timeout=5)
    snapshot = copy.deepcopy(msgs)
    for method in (cm.on_overflow, cm.compact):
        out, events = await method(msgs)
        assert msgs == snapshot
        assert structure_problems(out) == []
        assert out[0] is msgs[0]
        names = strategies(events)
        assert names == sorted(names, key=STRATEGIES.index)
        if "truncate" not in names:
            assert kept_verbatim(last_user_content(msgs), out)
    out, _ = await cm.on_overflow(msgs)
    assert count_tokens(out) <= int(cm.usable * 0.85)


# --------------------------------------------------------------------------- #
# Adversarial shapes the random conversations never produce
# --------------------------------------------------------------------------- #


async def test_pending_tool_call_without_results_stays_valid():
    """The turn was interrupted after the call was emitted: no results yet."""
    msgs = [SYSTEM, {"role": "user", "content": "start the audit"}]
    for i in range(40):
        msgs += step(f"c{i}", "read_file", big_output(40), path=f"f{i}.py")
    msgs.append({"role": "assistant", "content": "one more look",
                 "tool_calls": [call("pending", "read_file", path="last.py")]})
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer())
    out, events = await cm.fit(msgs)
    assert events and structure_problems(out) == []
    assert out[-1].get("tool_calls") and out[-1]["tool_calls"][0]["id"] == "pending"
    assert count_tokens(out) <= cm.usable


async def test_multi_call_step_is_never_split_by_compaction():
    """One assistant message with three calls keeps all three results next to it."""
    msgs = [SYSTEM, {"role": "user", "content": "check everything"}]
    for i in range(25):
        calls = [call(f"c{i}_{k}", "read_file", path=f"f{i}_{k}.py") for k in range(3)]
        msgs.append({"role": "assistant", "content": f"step {i}", "tool_calls": calls})
        msgs += [{"role": "tool", "tool_call_id": c["id"], "name": "read_file", "content": big_output(30)}
                 for c in calls]
    msgs.append({"role": "user", "content": "now summarise the findings"})
    cm = ContextManager(4096, FakeCounter(), summarizer=RecordingSummarizer())
    out, _ = await cm.fit(msgs)
    assert structure_problems(out) == []
    assert kept_verbatim("now summarise the findings", out)


async def test_fit_terminates_when_nothing_can_shrink_the_prompt():
    """A counter that never changes must not spin: fit() returns a result anyway."""
    calls = 0

    async def stubborn(messages, tools=None):
        nonlocal calls
        calls += 1
        return 100_000

    msgs = [SYSTEM, {"role": "user", "content": "task"}]
    for i in range(30):
        msgs += step(f"c{i}", "bash", big_output(20), command=f"cmd {i}")
    msgs.append({"role": "user", "content": "final question"})
    cm = ContextManager(4096, stubborn, summarizer=RecordingSummarizer())
    out, events = await asyncio.wait_for(cm.fit(msgs), 10)
    assert events == [] and structure_problems(out) == []
    assert kept_verbatim("final question", out)
    assert calls < 200


@pytest.mark.parametrize("seed", range(40))
async def test_fit_invariants_with_a_pending_tool_call(seed):
    rng = random.Random(50_000 + seed)
    msgs = rand_conversation(rng)
    while msgs and msgs[-1]["role"] in ("tool", "assistant"):
        msgs.pop()
    msgs.append({"role": "assistant", "content": rand_text(rng, rng.choice([0, 200])),
                 "tool_calls": [call("pending", "read_file", path="x.py")]})
    assert structure_problems(msgs) == []
    snapshot = copy.deepcopy(msgs)
    cm = ContextManager(rng.choice([2048, 4096, 8192]), FakeCounter(),
                        summarizer=make_summarizer(rng), summarizer_timeout=5)
    out, events = await cm.fit(msgs)
    assert msgs == snapshot
    assert structure_problems(out) == []
    assert out[-1].get("tool_calls"), "the pending call must survive"
    assert count_tokens(out) <= cm.usable
