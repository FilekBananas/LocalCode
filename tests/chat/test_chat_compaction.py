"""Chatbot context compaction: planning, summarising, persistence and fallbacks.

The chatbot keeps long conversations inside a small window by replacing the
oldest turns with one model-written summary (:mod:`localcode.chat.compaction`),
storing it on the conversation and sending ``system + summary + the messages
after it`` on every later turn (:mod:`localcode.chat.service`).

The tests below cover the whole path with the chat fakes: when compaction is
triggered at all, what it may and may not remove, how the summary is stored,
reused and extended, what happens when the summarising call fails, and the
``compaction`` SSE frame plus ``POST /api/conversations/{id}/compact``. The last
test is a ``real_model`` end-to-end run with the tiny local GGUF model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chat_fakes import (CHAT_KEY, FakeEngine, FakeLibrary, FakeRuntime, collect, make_model, make_service,
                        types_of)
from localcode.chat.compaction import (
    DEFAULT_THRESHOLD,
    MIN_REGION_MESSAGES,
    RECENT_SHARE,
    STRATEGY_SUMMARIZE,
    SUMMARY_HEADER,
    ChatCompactor,
    build_summary_prompt,
    clean_summary,
    pending_messages,
    plan_compaction,
    render_transcript,
)
from localcode.chat.api import ChatApi
from localcode.chat.context import ContextBuilder, TokenCounter
from localcode.chat.db import ChatDB
from localcode.chat.repo import ChatRepo, coerce_settings
from localcode.chat.service import ApiError
from localcode.config import Settings
from localcode.runtime.base import EngineError

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SUMMARY_MARKER = "You compress the beginning of a conversation"
FAKE_SUMMARY = ("Goal: użytkownik planuje wyjazd w góry.\n"
                "Facts: wyjeżdża 12 maja, nazywa się Filip, ma psa Burka.\n"
                "Decisions: jedzie pociągiem.\nOpen questions: nocleg.\nPreferences: odpowiadaj po polsku.")


async def count4(text: str) -> int:
    return (len(text) + 3) // 4


def counter() -> TokenCounter:
    return TokenCounter(count4)


def msg(seq: int, role: str, content: str, *, status: str = "complete",
        attachments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"id": f"m{seq}", "seq": seq, "role": role, "content": content, "status": status,
            "attachments": attachments or []}


def history(pairs: int, *, chars: int = 600, start: int = 1) -> list[dict[str, Any]]:
    """``pairs`` user/assistant turns, each message ``chars`` characters long."""
    out: list[dict[str, Any]] = []
    seq = start
    for i in range(pairs):
        out.append(msg(seq, "user", f"P{i} " + "pytanie o góry " * (chars // 15)))
        out.append(msg(seq + 1, "assistant", f"O{i} " + "odpowiedź o górach " * (chars // 19)))
        seq += 2
    return out


def is_summary_prompt(prompt: str) -> bool:
    return SUMMARY_MARKER in prompt


class Summaries:
    """Counts summarisation calls made through a :class:`FakeEngine` script."""

    def __init__(self, text: str | list[str] = FAKE_SUMMARY, *, answer: str = "Odpowiedź modelu.",
                 fail: BaseException | None = None) -> None:
        self.text = text
        self.answer = answer
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str, params: Any) -> list[str]:
        if not is_summary_prompt(prompt):
            return [self.answer]
        self.calls += 1
        self.prompts.append(prompt)
        if self.fail is not None:
            raise self.fail
        text = self.text if isinstance(self.text, str) else self.text[min(self.calls - 1, len(self.text) - 1)]
        return [text] if text else []


def settings_dict(**over: Any) -> dict[str, Any]:
    base = coerce_settings({"memoryEnabled": True})
    base.update(over)
    return base


def service_with(engine: FakeEngine, repo: ChatRepo, config: Settings | None = None, **kwargs: Any):
    library = FakeLibrary([make_model(CHAT_KEY, name="Qwen3 0.6B", caps=["thinking", "tools"])])
    runtime = FakeRuntime({CHAT_KEY: engine})
    return make_service(repo, runtime, library, config or Settings(), **kwargs)


def seed(repo: ChatRepo, messages: list[tuple[str, str]], *, title: str = "Rozmowa") -> dict[str, Any]:
    conversation = repo.create_conversation({"title": title, "provider": "local", "model": CHAT_KEY})
    for role, content in messages:
        repo.create_message({"conversationId": conversation["id"], "role": role, "content": content,
                             "status": "complete"})
    return repo.get_conversation(conversation["id"])


def long_pairs(n: int, *, chars: int = 700) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i in range(n):
        out.append(("user", f"Pytanie {i}: " + "opowiedz o górach i pociągach " * (chars // 30)))
        out.append(("assistant", f"Odpowiedź {i}: " + "w górach jest pięknie o tej porze " * (chars // 33)))
    return out


async def run_turn(service: Any, conversation_id: str | None, content: str, **over: Any) -> list[dict[str, Any]]:
    body = {"conversationId": conversation_id, "content": content}
    body.update(over)
    frames = await collect(service.send(body))
    await service.drain()
    return frames


def frame(frames: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((f for f in frames if f["type"] == kind), None)


def answer_prompts(engine: FakeEngine) -> list[str]:
    return [p for p in engine.prompts if not is_summary_prompt(p)]


# --------------------------------------------------------------------------- #
# 1. Planning: what is picked and what may never be touched
# --------------------------------------------------------------------------- #


def test_pending_messages_skips_placeholders_current_turn_and_the_summarised_part() -> None:
    items = [msg(1, "user", "a"), msg(2, "assistant", "b"), msg(3, "user", "c"),
             msg(4, "assistant", "błąd", status="error"), msg(5, "user", "teraz"),
             msg(6, "assistant", "", status="streaming")]
    # The last user message and everything after it is the current turn.
    assert [m["seq"] for m in pending_messages(items)] == [1, 2, 3]
    assert [m["seq"] for m in pending_messages(items, after_seq=2)] == [3]
    # force=True (manual /compact) also offers the last turn.
    assert [m["seq"] for m in pending_messages(items, current_turn=False)] == [1, 2, 3, 5]
    assert pending_messages(items, after_seq=99) == []


def test_plan_cut_lands_on_a_user_turn_and_keeps_the_recent_budget() -> None:
    items = [msg(i, "user" if i % 2 == 1 else "assistant", "x") for i in range(1, 9)]
    sizes = [100] * len(items)
    plan = plan_compaction(items, sizes, recent_budget=450)
    assert plan is not None
    # 450 tokens ⇒ two whole turns (4 messages) stay verbatim; the cut is on a user turn.
    assert plan.kept == 4 and len(plan.region) == 4
    assert items[len(plan.region)]["role"] == "user"
    assert plan.region[-1]["role"] == "assistant"      # a pair is never split
    assert plan.upto_seq == plan.region[-1]["seq"]
    assert plan.region_tokens == 400 and bool(plan) is True


def test_plan_declines_when_there_is_nothing_worth_summarising() -> None:
    small = [msg(1, "user", "x"), msg(2, "assistant", "y")]
    assert plan_compaction(small, [10, 10], recent_budget=0) is None          # under MIN_REGION_TOKENS
    assert plan_compaction(small[:1], [500], recent_budget=0) is None         # under MIN_REGION_MESSAGES
    assert MIN_REGION_MESSAGES == 2
    # Everything still fits in the verbatim budget.
    assert plan_compaction(small, [300, 300], recent_budget=5000) is None


def test_render_transcript_names_attachments_and_clips_messages() -> None:
    items = [msg(1, "user", "a" * 5000, attachments=[{"name": "raport.pdf"}]),
             msg(2, "assistant", "b" * 5000),
             msg(3, "assistant", "", status="complete")]
    text = render_transcript(items)
    assert text.startswith("USER: ")
    assert "[attachments: raport.pdf]" in text
    assert "ASSISTANT: " in text
    assert len(text) < 5000                      # both messages were clipped
    assert text.count("\n") == 1                 # the empty assistant message is skipped


@pytest.mark.parametrize("raw,expected", [
    ("<think>myślę…</think>\nGoal: coś tam, dłuższy tekst podsumowania.", "Goal:"),
    ("```\nGoal: coś tam, dłuższy tekst podsumowania.\n```", "Goal:"),
    (f"{SUMMARY_HEADER}\nGoal: coś tam, dłuższy tekst podsumowania.", "Goal:"),
])
def test_clean_summary_strips_reasoning_fences_and_our_header(raw: str, expected: str) -> None:
    out = clean_summary(raw)
    assert out.startswith(expected) and SUMMARY_HEADER not in out and "<think>" not in out


@pytest.mark.parametrize("raw", ["", "krótkie", None, 42, "<think>nigdy nie skończone", "</think>"])
def test_clean_summary_rejects_unusable_answers(raw: Any) -> None:
    assert clean_summary(raw) == ""


def test_build_summary_prompt_folds_in_the_previous_summary() -> None:
    plain = build_summary_prompt("USER: cześć", target_tokens=500)
    assert [m["role"] for m in plain] == ["system", "user"]
    assert "<transcript>" in plain[1]["content"]
    extended = build_summary_prompt("USER: cześć", previous_summary="STARE PODSUMOWANIE", target_tokens=500)
    assert "STARE PODSUMOWANIE" in extended[1]["content"]
    assert "fold it into your answer" in extended[1]["content"]


# --------------------------------------------------------------------------- #
# 2. The compactor
# --------------------------------------------------------------------------- #


def compactor(*, n_ctx: int = 4096, summarizer: Any = None, **kwargs: Any) -> ChatCompactor:
    return ChatCompactor(n_ctx=n_ctx, max_tokens=None, counter=counter(), summarizer=summarizer, **kwargs)


async def ok_summarizer(messages: list[dict[str, Any]], want: int) -> str:
    return FAKE_SUMMARY


async def test_needs_compaction_follows_the_threshold() -> None:
    comp = compactor(threshold=0.5)
    assert comp.threshold == 0.5
    assert not comp.needs_compaction(comp.soft_limit)
    assert comp.needs_compaction(comp.soft_limit + 1)
    assert compactor(threshold="nonsense").threshold == DEFAULT_THRESHOLD    # type: ignore[arg-type]
    assert compactor(threshold=5.0).threshold == DEFAULT_THRESHOLD
    assert 0 < compactor().recent_budget < compactor().prompt_budget * RECENT_SHARE + 1


async def test_summarize_produces_a_result_and_never_touches_the_current_turn() -> None:
    items = history(8)
    result = await compactor(summarizer=ok_summarizer).summarize(items)
    assert result.ok and result.changed and result.summary == FAKE_SUMMARY
    assert result.strategy == STRATEGY_SUMMARIZE and not result.extended
    assert result.tokens > 0 and result.messages >= 2 and result.region_tokens > 0
    last_user = max(m["seq"] for m in items if m["role"] == "user")
    assert result.upto_seq < last_user
    assert result.to_dict()["uptoSeq"] == result.upto_seq


async def test_summarize_extends_an_existing_summary_with_only_the_new_turns() -> None:
    seen: list[list[dict[str, Any]]] = []

    async def spy(messages: list[dict[str, Any]], want: int) -> str:
        seen.append(messages)
        return FAKE_SUMMARY + " rozszerzone"

    items = history(10)
    result = await compactor(summarizer=spy).summarize(items, previous_summary="STARE", summary_upto_seq=6)
    assert result.ok and result.extended
    body = seen[0][1]["content"]
    assert "STARE" in body
    assert "P0 " not in body and "P1 " not in body and "P2 " not in body     # seq ≤ 6 not resent
    assert "P3 " in body
    assert result.upto_seq > 6


@pytest.mark.parametrize("failure", [EngineError("busy"), RuntimeError("boom")])
async def test_summarize_reports_engine_failures_instead_of_raising(failure: BaseException) -> None:
    async def broken(messages: list[dict[str, Any]], want: int) -> str:
        raise failure

    result = await compactor(summarizer=broken).summarize(history(8))
    assert not result.ok and not result.changed and result.error
    assert type(failure).__name__ in result.error


async def test_summarize_reports_a_timeout_and_an_empty_answer() -> None:
    async def slow(messages: list[dict[str, Any]], want: int) -> str:
        await asyncio.sleep(5)
        return FAKE_SUMMARY

    result = await compactor(summarizer=slow, timeout=0.05).summarize(history(8))
    assert not result.ok and "TimeoutError" in (result.error or "")

    async def empty(messages: list[dict[str, Any]], want: int) -> str:
        return "   "

    result = await compactor(summarizer=empty).summarize(history(8))
    assert not result.ok and result.error == "summarizer returned no usable text"


async def test_summarize_without_a_summarizer_or_without_material() -> None:
    assert (await compactor().summarize(history(8))).error == "no summarizer available"
    tiny = [msg(1, "user", "cześć"), msg(2, "assistant", "hej"), msg(3, "user", "co słychać?")]
    assert (await compactor(summarizer=ok_summarizer).summarize(tiny)).error == "nothing to summarize"


async def test_summarize_cancellation_propagates() -> None:
    async def hang(messages: list[dict[str, Any]], want: int) -> str:
        await asyncio.sleep(30)
        return FAKE_SUMMARY

    task = asyncio.ensure_future(compactor(summarizer=hang, timeout=None).summarize(history(8)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_forced_compaction_summarises_the_whole_backlog() -> None:
    items = history(4)
    natural = await compactor(summarizer=ok_summarizer).summarize(items)
    forced = await compactor(summarizer=ok_summarizer).summarize(items, force=True)
    assert forced.messages > natural.messages
    assert forced.upto_seq == items[-1]["seq"]


async def test_summary_is_capped_to_its_share_of_the_prompt() -> None:
    async def rambling(messages: list[dict[str, Any]], want: int) -> str:
        return "Goal: " + "bardzo długie podsumowanie " * 4000

    comp = compactor(n_ctx=4096, summarizer=rambling)
    result = await comp.summarize(history(12))
    assert result.ok and await count4(result.summary) <= comp.summary_cap + 8


async def test_a_single_huge_last_user_message_is_left_alone() -> None:
    items = [msg(1, "user", "x" * 40_000)]
    comp = compactor(summarizer=ok_summarizer)
    assert comp.needs_compaction(await count4(items[0]["content"]))
    result = await comp.summarize(items)
    assert not result.ok and result.error == "nothing to summarize"


async def test_four_k_window_still_compacts() -> None:
    comp = compactor(n_ctx=4096, summarizer=ok_summarizer)
    result = await comp.summarize(history(12))
    assert result.ok and result.messages >= 2
    assert result.tokens <= comp.summary_cap + 8


# --------------------------------------------------------------------------- #
# 3. Invariants of the prompt built from a summary
# --------------------------------------------------------------------------- #


async def build_ctx(items: list[dict[str, Any]], *, summary: str | None, upto: int, n_ctx: int = 4096,
                    retrieved: list[dict[str, Any]] | None = None) -> Any:
    builder = ContextBuilder(
        settings=settings_dict(globalSystemPrompt="GLOBAL-PROMPT"), folder=None,
        conversation={"id": "c1", "systemPrompt": "CONV-PROMPT"}, history=items,
        memories_global=[{"id": "m", "scope": "global", "folderId": None, "kind": "fact",
                          "content": "WAŻNE-WSPOMNIENIE", "source": "user", "pinned": True,
                          "createdAt": 0, "updatedAt": 0}],
        memories_folder=[], retrieved=retrieved or [], query=items[-1]["content"],
        n_ctx=n_ctx, max_tokens=None, counter=counter(), summary=summary, summary_upto_seq=upto)
    return await builder.build()


async def test_prompt_after_compaction_is_system_plus_summary_plus_the_rest() -> None:
    items = history(6, chars=400)
    upto = items[5]["seq"]
    ctx = await build_ctx(items, summary=FAKE_SUMMARY, upto=upto)
    system = ctx.messages[0]["content"]
    assert ctx.messages[0]["role"] == "system"
    assert SUMMARY_HEADER in system and "Burka" in system
    assert "GLOBAL-PROMPT" in system and "CONV-PROMPT" in system and "WAŻNE-WSPOMNIENIE" in system
    body = "\n".join(m["content"] for m in ctx.messages[1:])
    assert "P0 " not in body and "P1 " not in body and "P2 " not in body
    current = [m for m in items if m["role"] == "user"][-1]
    assert current["content"][:20] in body                        # the current question, verbatim
    assert ctx.report.summarized_messages == 6
    assert ctx.report.summary_tokens > 0


async def test_prompt_is_a_valid_chat_template_sequence() -> None:
    items = history(6, chars=400)
    # cut in the middle of a pair: the builder must still hand over user/assistant/user…
    ctx = await build_ctx(items, summary=FAKE_SUMMARY, upto=items[4]["seq"])
    roles = [m["role"] for m in ctx.messages]
    assert roles[0] == "system" and roles[1] == "user" and roles[-1] == "user"
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))
    assert all(isinstance(m["content"], str) for m in ctx.messages)


async def test_citations_and_memories_survive_compaction() -> None:
    chunk = {"id": "k-0", "documentId": "k", "documentName": "notatki.md", "folderId": None,
             "scope": "global", "idx": 0, "content": "CYTAT-Z-BAZY o pociągach", "tokenEstimate": 10,
             "score": 1.0}
    items = history(6, chars=400)
    ctx = await build_ctx(items, summary=FAKE_SUMMARY, upto=items[5]["seq"], retrieved=[chunk])
    assert "CYTAT-Z-BAZY" in ctx.messages[0]["content"]
    assert "WAŻNE-WSPOMNIENIE" in ctx.messages[0]["content"]
    assert [c["documentName"] for c in ctx.citations] == ["notatki.md"]


# --------------------------------------------------------------------------- #
# 4. The service: triggering, persistence, reuse, frames, failure
# --------------------------------------------------------------------------- #


async def test_short_conversation_is_not_compacted(repo: ChatRepo) -> None:
    engine = FakeEngine(Summaries(), n_ctx=8192)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, [("user", "cześć"), ("assistant", "hej")])
        frames = await run_turn(service, conversation["id"], "co u ciebie?")
        assert frame(frames, "compaction") is None
        assert engine.script.calls == 0
        assert repo.get_conversation(conversation["id"])["summary"] is None
        assert frame(frames, "context")["summary"] == 0
    finally:
        await service.aclose()


async def test_long_conversation_is_compacted_stored_and_reported(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(12))
        frames = await run_turn(service, conversation["id"], "podsumuj, proszę")
        compaction = frame(frames, "compaction")
        assert compaction is not None
        assert compaction["strategy"] == STRATEGY_SUMMARIZE
        assert compaction["after"] < compaction["before"]
        assert script.calls == 1
        stored = repo.get_conversation(conversation["id"])
        assert stored["summary"] == FAKE_SUMMARY
        assert stored["summaryUptoSeq"] > 0 and stored["summaryTokens"] > 0
        # the summary reached the answering prompt, the oldest turn did not
        prompt = answer_prompts(engine)[-1]
        assert "Burka" in prompt and SUMMARY_HEADER in prompt
        assert "Pytanie 0:" not in prompt
        assert "podsumuj, proszę" in prompt
        assert frame(frames, "context")["summary"] > 0
        assert types_of(frames)[0] == "start" and types_of(frames)[-1] == "done"
    finally:
        await service.aclose()


async def test_summary_is_reused_and_not_regenerated_when_nothing_new_fell_out(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(12))
        await run_turn(service, conversation["id"], "pierwsze pytanie")
        first_seq = repo.get_conversation(conversation["id"])["summaryUptoSeq"]
        assert script.calls == 1
        frames = await run_turn(service, conversation["id"], "krótkie drugie pytanie")
        assert script.calls == 1, "the stored summary must be reused, not rewritten"
        assert frame(frames, "compaction") is None
        stored = repo.get_conversation(conversation["id"])
        assert stored["summary"] == FAKE_SUMMARY and stored["summaryUptoSeq"] == first_seq
        prompt = answer_prompts(engine)[-1]
        assert SUMMARY_HEADER in prompt and "Pytanie 0:" not in prompt
    finally:
        await service.aclose()


async def test_second_compaction_extends_the_first_summary(repo: ChatRepo) -> None:
    script = Summaries([FAKE_SUMMARY, FAKE_SUMMARY + "\nFacts: doszedł też namiot."])
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(12))
        await run_turn(service, conversation["id"], "pierwsze pytanie")
        first = repo.get_conversation(conversation["id"])
        for _ in range(8):
            repo.create_message({"conversationId": conversation["id"], "role": "user",
                                 "content": "kolejne pytanie o góry " * 60, "status": "complete"})
            repo.create_message({"conversationId": conversation["id"], "role": "assistant",
                                 "content": "kolejna odpowiedź o górach " * 60, "status": "complete"})
        await run_turn(service, conversation["id"], "drugie pytanie")
        assert script.calls == 2
        second_prompt = script.prompts[1]
        assert FAKE_SUMMARY in second_prompt, "the previous summary is folded into the new one"
        assert "Pytanie 0:" not in second_prompt, "already summarised turns are not resent"
        assert "kolejne pytanie o góry" in second_prompt
        stored = repo.get_conversation(conversation["id"])
        assert "namiot" in stored["summary"]
        assert stored["summaryUptoSeq"] > first["summaryUptoSeq"]
    finally:
        await service.aclose()


@pytest.mark.parametrize("script", [
    Summaries(fail=EngineError("model zajęty")),
    Summaries(""),                                   # empty answer
    Summaries("za krótkie"),                         # unusable answer
])
async def test_a_failed_summary_never_breaks_the_turn(repo: ChatRepo, script: Summaries) -> None:
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(12))
        frames = await run_turn(service, conversation["id"], "pytanie mimo awarii")
        assert frame(frames, "error") is None
        assert frame(frames, "compaction") is None
        assert frame(frames, "done") is not None
        assert repo.get_conversation(conversation["id"])["summary"] is None
        stored = frame(frames, "done")["message"]
        assert stored["status"] == "complete" and stored["content"]
        # trimming fallback: the prompt still fits the window
        assert frame(frames, "context")["used"] <= frame(frames, "context")["limit"]
    finally:
        await service.aclose()


async def test_a_summary_timeout_falls_back_to_trimming(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096, delay=0.2)
    service = service_with(engine, repo, compact_timeout=0.05)
    try:
        conversation = seed(repo, long_pairs(12))
        frames = await run_turn(service, conversation["id"], "pytanie mimo zwłoki")
        assert frame(frames, "compaction") is None
        assert frame(frames, "done") is not None
        assert repo.get_conversation(conversation["id"])["summary"] is None
    finally:
        await service.aclose()


async def test_compaction_threshold_comes_from_the_settings(repo: ChatRepo) -> None:
    config = Settings()
    config.chat.auto_compact_threshold = None
    config.agent.auto_compact_threshold = 0.5
    engine = FakeEngine(Summaries(), n_ctx=4096)
    service = service_with(engine, repo, config)
    try:
        assert service.compact_threshold() == 0.5
        config.chat.auto_compact_threshold = 0.9
        assert service.compact_threshold() == 0.9
        config.chat.auto_compact_threshold = 42.0        # invalid → the agent's value
        assert service.compact_threshold() == 0.5
    finally:
        await service.aclose()


# --------------------------------------------------------------------------- #
# 5. Edit / regenerate invalidate the summary
# --------------------------------------------------------------------------- #


async def test_editing_a_message_inside_the_summary_drops_it(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(12))
        await run_turn(service, conversation["id"], "pierwsze pytanie")
        stored = repo.get_conversation(conversation["id"])
        upto = stored["summaryUptoSeq"]
        assert upto > 0
        target = next(m for m in repo.list_messages(conversation["id"])
                      if m["role"] == "user" and m["seq"] <= upto)
        await run_turn(service, conversation["id"], "zmienione pytanie", editMessageId=target["id"])
        after = repo.get_conversation(conversation["id"])
        assert after["summaryUptoSeq"] == 0
        assert after["summary"] is None or after["summaryUptoSeq"] < target["seq"]
    finally:
        await service.aclose()


def test_regenerate_and_delete_never_leave_a_summary_past_the_change(repo: ChatRepo) -> None:
    conversation = seed(repo, [("user", "a"), ("assistant", "b"), ("user", "c"), ("assistant", "d")])
    messages = repo.list_messages(conversation["id"])
    repo.set_summary(conversation["id"], FAKE_SUMMARY, messages[3]["seq"], 100)
    # regenerate → everything after the last user message goes, and so does the summary
    repo.remove_messages_after(conversation["id"], messages[2]["seq"])
    assert repo.get_conversation(conversation["id"])["summaryUptoSeq"] == 0

    repo.set_summary(conversation["id"], FAKE_SUMMARY, messages[0]["seq"], 100)
    assert repo.get_conversation(conversation["id"])["summaryUptoSeq"] == messages[0]["seq"]
    # a change after the summarised region leaves it in place
    assert repo.invalidate_summary_from(conversation["id"], messages[2]["seq"]) is False
    assert repo.get_conversation(conversation["id"])["summary"] == FAKE_SUMMARY
    # deleting a summarised message drops it
    assert repo.delete_message(messages[0]["id"]) is True
    assert repo.get_conversation(conversation["id"])["summary"] is None


def test_set_summary_with_an_empty_text_clears_it(repo: ChatRepo) -> None:
    conversation = seed(repo, [("user", "a"), ("assistant", "b")])
    repo.set_summary(conversation["id"], FAKE_SUMMARY, 2, 50)
    repo.set_summary(conversation["id"], "   ", 2, 50)
    stored = repo.get_conversation(conversation["id"])
    assert stored["summary"] is None and stored["summaryUptoSeq"] == 0 and stored["summaryTokens"] == 0
    assert repo.set_summary("nope", FAKE_SUMMARY, 2, 50) is None


def test_summary_survives_a_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "chat.db"
    repo = ChatRepo(ChatDB(db_path), tmp_path / "uploads")
    try:
        conversation = seed(repo, [("user", "a"), ("assistant", "b")])
        repo.set_summary(conversation["id"], FAKE_SUMMARY, 2, 77)
    finally:
        repo.close()
    again = ChatRepo(ChatDB(db_path), tmp_path / "uploads")
    try:
        stored = again.get_conversation(conversation["id"])
        assert stored["summary"] == FAKE_SUMMARY
        assert stored["summaryUptoSeq"] == 2 and stored["summaryTokens"] == 77
    finally:
        again.close()


# --------------------------------------------------------------------------- #
# 6. Manual compaction (POST /api/conversations/{id}/compact)
# --------------------------------------------------------------------------- #


async def test_manual_compact_summarises_the_whole_backlog(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        api = ChatApi(repo, service)
        conversation = seed(repo, long_pairs(10))
        out = await api.compact_conversation(conversation["id"])
        assert out["changed"] is True and out["strategy"] == STRATEGY_SUMMARIZE
        assert out["after"] < out["before"] and out["messages"] >= 2
        assert out["summary"] == FAKE_SUMMARY
        stored = repo.get_conversation(conversation["id"])
        assert stored["summary"] == FAKE_SUMMARY
        # nothing new to summarise afterwards
        again = await api.compact_conversation(conversation["id"])
        assert again["changed"] is False and again["before"] == again["after"]
        assert script.calls == 1
    finally:
        await service.aclose()


async def test_manual_compact_on_a_short_conversation_is_not_an_error(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=8192)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, [("user", "cześć"), ("assistant", "hej")])
        out = await service.compact_conversation(conversation["id"])
        assert out["changed"] is False and out["messages"] == 0 and out["summary"] == ""
        assert script.calls == 0
    finally:
        await service.aclose()


async def test_manual_compact_reports_a_failed_summary_and_a_missing_conversation(repo: ChatRepo) -> None:
    engine = FakeEngine(Summaries(fail=EngineError("zajęty")), n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(10))
        with pytest.raises(ApiError) as excinfo:
            await service.compact_conversation(conversation["id"])
        assert excinfo.value.status == 409 and excinfo.value.code == "compaction_failed"
        assert repo.get_conversation(conversation["id"])["summary"] is None
        with pytest.raises(ApiError) as missing:
            await service.compact_conversation("nie-ma-takiej")
        assert missing.value.status == 404
    finally:
        await service.aclose()


# --------------------------------------------------------------------------- #
# 7. Edge cases
# --------------------------------------------------------------------------- #


async def test_one_oversized_user_message_still_answers(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, [])
        frames = await run_turn(service, conversation["id"], "opowiedz o tym: " + "bardzo długi tekst " * 3000)
        assert frame(frames, "error") is None and frame(frames, "done") is not None
        assert script.calls == 0                 # nothing but the current turn to summarise
        assert repo.get_conversation(conversation["id"])["summary"] is None
        ctx = frame(frames, "context")
        assert ctx["used"] <= ctx["limit"]
    finally:
        await service.aclose()


async def test_compaction_with_attachments_mentions_them_and_keeps_the_turn(repo: ChatRepo) -> None:
    items = history(8, chars=400)
    items[0]["attachments"] = [{"name": "umowa.pdf", "kind": "text"}]
    seen: list[str] = []

    async def spy(messages: list[dict[str, Any]], want: int) -> str:
        seen.append(messages[1]["content"])
        return FAKE_SUMMARY

    result = await compactor(summarizer=spy).summarize(items)
    assert result.ok
    assert "[attachments: umowa.pdf]" in seen[0]


async def test_tiny_four_k_model_end_to_end(repo: ChatRepo) -> None:
    script = Summaries()
    engine = FakeEngine(script, n_ctx=4096)
    service = service_with(engine, repo)
    try:
        conversation = seed(repo, long_pairs(14))
        frames = await run_turn(service, conversation["id"], "krótkie pytanie na koniec")
        compaction = frame(frames, "compaction")
        assert compaction is not None and compaction["after"] < compaction["before"]
        ctx = frame(frames, "context")
        assert ctx["limit"] == 4096 and ctx["used"] <= 4096 and ctx["summary"] > 0
        assert frame(frames, "error") is None
    finally:
        await service.aclose()


# --------------------------------------------------------------------------- #
# 8. Real model
# --------------------------------------------------------------------------- #


@pytest.mark.real_model
async def test_real_model_compaction_round_trip(tmp_path: Path, isolated_home: Path) -> None:
    """A real ~15-message conversation is summarised, stored, reused and answered.

    The tiny GGUF model runs at ``n_ctx`` 2048, so the backlog cannot fit and the
    compactor must write a real summary. The follow-up question asks about a fact
    that only lives in the summarised part.
    """
    from lc_testing import model_path
    from localcode.hub.metadata import read_gguf_meta
    from localcode.hub.types import LocalModel
    from localcode.runtime.manager import RuntimeManager

    path = model_path("gguf_tiny")
    meta = read_gguf_meta(path)
    model = LocalModel(key=CHAT_KEY, format="gguf", path=str(path), display_name="Qwen3 0.6B",
                       size_bytes=path.stat().st_size, quant=meta.quant, meta=meta, capabilities=["thinking"])

    class OneModelLibrary:
        def list(self) -> list[LocalModel]:
            return [model]

        def get(self, key: str) -> LocalModel | None:
            return model if key == model.key else None

        def resolve(self, query: str) -> LocalModel | None:
            return model

        def load_defaults(self, key: str) -> dict[str, Any] | None:
            return None

    config = Settings()
    config.runtime.default_ctx = 2048
    config.runtime.max_loaded_models = 1
    config.runtime.idle_ttl_minutes = 0
    config.chat.auto_compact_threshold = 0.6

    repo = ChatRepo(ChatDB(tmp_path / "chat.db"), tmp_path / "uploads")
    runtime = RuntimeManager(config, OneModelLibrary())
    service = make_service(repo, runtime, OneModelLibrary(), config, status_delay=0.2, title_timeout=30.0,
                           compact_timeout=300.0)
    try:
        filler = ("Warto pamiętać, że w górach pogoda potrafi zmienić się w kilkanaście minut, "
                  "dlatego zawsze zabieraj kurtkę przeciwdeszczową, zapas wody i naładowany telefon. ")
        facts = [
            ("Mam na imię Filip i mieszkam w Gdańsku.", "Miło Cię poznać, Anno. " + filler * 4),
            ("Mój pies nazywa się Burek i ma 4 lata.", "Burek to świetne imię dla psa. " + filler * 4),
            ("Planuję wyjazd w Tatry 12 maja.", "Maj to dobry termin na Tatry. " + filler * 4),
            ("Pojadę pociągiem, nie samochodem.", "Pociąg do Zakopanego jest wygodny. " + filler * 4),
            ("Mój numer buta to 44.", "Zanotowane, numer buta 44. " + filler * 4),
            ("Lubię herbatę miętową, nie kawę.", "Herbata miętowa dobrze nawadnia. " + filler * 4),
            ("Opowiedz mi coś o pogodzie w górach wiosną.", "Wiosną w górach pogoda zmienia się bardzo "
             "szybko: rano bywa mroźno, w południe ciepło, a po południu często pada. " + filler * 4),
        ]
        conversation = repo.create_conversation({"title": "Wyjazd", "provider": "local", "model": CHAT_KEY})
        for question, answer in facts:
            repo.create_message({"conversationId": conversation["id"], "role": "user", "content": question,
                                 "status": "complete"})
            repo.create_message({"conversationId": conversation["id"], "role": "assistant", "content": answer,
                                 "status": "complete"})
        assert len(repo.list_messages(conversation["id"])) == 14

        frames = await run_turn(service, conversation["id"], "Jak nazywa się mój pies?")
        compaction = frame(frames, "compaction")
        stored = repo.get_conversation(conversation["id"])
        print(f"\n=== context frame: {frame(frames, 'context')} ===")
        print(f"\n=== real compaction: {compaction} ===")
        print(f"summary ({stored['summaryTokens']} tok, upto seq {stored['summaryUptoSeq']}):\n"
              f"{(stored['summary'] or '')[:800]}")
        assert compaction is not None, "a 2048-token window must force compaction"
        assert compaction["after"] < compaction["before"]
        assert stored["summary"] and stored["summaryUptoSeq"] > 0 and stored["summaryTokens"] > 0
        first_seq = stored["summaryUptoSeq"]
        assert frame(frames, "error") is None

        # The stored summary is reused on the next turn instead of being rewritten.
        frames2 = await run_turn(service, conversation["id"], "A w którym miesiącu jadę w Tatry?")
        again = repo.get_conversation(conversation["id"])
        assert again["summary"] == stored["summary"] and again["summaryUptoSeq"] == first_seq
        assert frame(frames2, "error") is None
        answer = (frame(frames2, "done")["message"]["content"] or "").lower()
        print(f"=== answer from the summarised part: {answer[:300]} ===")
        assert answer.strip(), "the model must still answer after compaction"
        # The fact lives only in the summarised part; the summary itself must carry it.
        assert "burek" in (stored["summary"] or "").lower() or "maj" in (stored["summary"] or "").lower()
    finally:
        await service.aclose()
        await runtime.shutdown()
        repo.close()


# --------------------------------------------------------------------------- #
# Language of the summary (small models ignore "use the same language")
# --------------------------------------------------------------------------- #


def test_detect_language_picks_the_conversation_language() -> None:
    from localcode.chat.compaction import detect_language

    polish = "Cześć, mam psa Burka i mieszkam w Gdańsku. Czy to jest dla mnie dobre rozwiązanie?"
    assert detect_language(polish) == "Polish"
    assert detect_language("Hello, I have a dog named Rex and I live in Boston.") == "English"
    # too short to tell → fall back to the configured UI language
    assert detect_language("ok", fallback="pl") == "Polish"


def test_summary_prompt_orders_the_language_explicitly() -> None:
    from localcode.chat.compaction import build_summary_prompt

    messages = build_summary_prompt("rozmowa", language="Polish")
    system, user = messages[0]["content"], messages[1]["content"]
    assert "WRITE THE SUMMARY IN POLISH" in system
    assert "written in Polish" in user
    # without a detected language the generic instruction stays
    generic = build_summary_prompt("x")[0]["content"]
    assert "SAME LANGUAGE" in generic


# --------------------------------------------------------------------------- #
# Regression: a summary must never be bigger than what it replaces
# --------------------------------------------------------------------------- #


async def test_summary_bigger_than_the_region_is_rejected() -> None:
    """A verbose summary of a few short turns would GROW the prompt: refuse it."""
    long_summary = "Goal: " + "bardzo szczegółowe podsumowanie rozmowy o górach " * 60

    async def verbose(messages: list[dict[str, Any]], want: int) -> str:
        return long_summary

    items = history(3, chars=120) + [msg(7, "user", "Nowe pytanie?")]
    comp = compactor(n_ctx=32768, summarizer=verbose)
    plan = await comp.plan(items, force=True)
    assert plan is not None and plan.region_tokens > 0
    result = await comp.summarize(items, force=True)
    assert not result.ok and not result.changed
    assert "larger" in (result.error or "")          # the summary was bigger than the region
    assert result.region_tokens == plan.region_tokens


async def test_extended_summary_may_grow_up_to_the_region_plus_the_old_summary() -> None:
    """Extending replaces the previous summary too, so its size is available."""
    items = history(6, chars=300) + [msg(13, "user", "Nowe pytanie?")]
    previous = "Goal: " + "stare podsumowanie " * 40
    comp = compactor(n_ctx=32768, summarizer=ok_summarizer)
    result = await comp.summarize(items, previous_summary=previous, summary_upto_seq=4, force=True)
    assert result.ok and result.extended


async def test_summary_covering_no_messages_is_not_reported_as_success() -> None:
    """``upto_seq = 0`` means "covers nothing": storing it would re-summarise forever."""
    items = [msg(0, "user", "Pierwsze pytanie " * 40), msg(0, "assistant", "Odpowiedź " * 40),
             msg(0, "user", "Drugie pytanie " * 40), msg(0, "assistant", "Odpowiedź " * 40),
             msg(0, "user", "Aktualne pytanie?")]
    result = await compactor(summarizer=ok_summarizer).summarize(items)
    assert not result.ok and result.upto_seq == 0


async def test_current_question_survives_a_summary_that_covers_it() -> None:
    """Even a summary covering every stored message keeps the current turn verbatim."""
    items = history(5, chars=300)
    last_user = [m for m in items if m["role"] == "user"][-1]
    ctx = await build_ctx(items, summary=FAKE_SUMMARY, upto=items[-1]["seq"] + 100)
    assert ctx.messages[-1]["role"] == "user"
    assert last_user["content"][:30] in ctx.messages[-1]["content"]
    roles = [m["role"] for m in ctx.messages]
    assert roles[0] == "system" and all(a != b for a, b in zip(roles[1:], roles[2:]))


async def test_current_question_survives_a_tiny_window() -> None:
    items = history(20, chars=800)
    counts: dict[str, int] = {"calls": 0}

    async def exact(messages: list[dict[str, Any]]) -> int:
        counts["calls"] += 1
        return sum(len(m["content"]) for m in messages) // 4

    builder = ContextBuilder(
        settings=settings_dict(), folder=None, conversation={"id": "c1"}, history=items,
        n_ctx=1024, max_tokens=None, counter=counter(), exact_count=exact,
        query=items[-2]["content"])
    ctx = await builder.build()
    assert ctx.messages[-1]["role"] == "user"
    assert counts["calls"] < 200
    roles = [m["role"] for m in ctx.messages]
    assert roles[0] == "system" and all(a != b for a, b in zip(roles[1:], roles[2:]))


# --------------------------------------------------------------------------- #
# A tiny window must summarise, not silently forget
# --------------------------------------------------------------------------- #


async def test_trimmed_history_triggers_compaction(repo) -> None:
    """When the window is so small that history is trimmed away, compact instead.

    The prompt is measured after trimming, so it always looks like it "fits";
    without this rule a 2k-context model answers the sixth turn with no memory of
    the first and the user is never told anything was lost.
    """
    # Messages so long that trimming leaves almost nothing: the assembled prompt
    # then sits far BELOW the compaction threshold while most of the conversation
    # has quietly disappeared.
    engine = FakeEngine(n_ctx=2048)
    service = service_with(engine, repo)
    conversation = seed(repo, long_pairs(6, chars=6000))

    frames = await run_turn(service, conversation["id"], "I co dalej?", maxTokens=384)
    kinds = [f["type"] for f in frames]
    assert "compaction" in kinds, f"brak kompaktowania mimo przyciętej historii: {kinds}"

    stored = repo.get_conversation(conversation["id"])
    assert stored["summary"], "streszczenie nie zostało zapisane"
    assert int(stored["summaryUptoSeq"]) > 0
