"""Prompt assembly: ai_ux scope rules (shared vs isolated) and the token budget."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from localcode.chat.context import (
    ISOLATION_NOTE,
    MAX_MEMORIES,
    TRUNCATION_MARKER,
    ContextBuilder,
    ContextScope,
    TokenCounter,
    build_system_prompt,
    build_title_prompt,
    compose_text,
    filter_retrieved,
    normalize_turns,
    order_memories,
    plan_budget,
    resolve_scope,
    truncate_to_tokens,
)
from localcode.chat.repo import coerce_settings

NOW = datetime(2026, 9, 22, 14, 5)


async def count4(text: str) -> int:
    return (len(text) + 3) // 4


def counter() -> TokenCounter:
    return TokenCounter(count4)


def settings(**over: Any) -> dict[str, Any]:
    base = coerce_settings({
        "globalSystemPrompt": "GLOBAL-PROMPT: always be polite.",
        "profile": {"displayName": "Filip", "role": "programista", "about": "", "responseStyle": "krótko",
                    "customInstructions": "", "enabled": True},
        "memoryEnabled": True,
    })
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def memory(content: str, *, scope: str = "global", folder: str | None = None, pinned: bool = False,
           updated: int = 0, kind: str = "fact") -> dict[str, Any]:
    return {"id": content, "scope": scope, "folderId": folder, "kind": kind, "content": content,
            "source": "user", "pinned": pinned, "createdAt": 0, "updatedAt": updated}


def chunk(doc: str, content: str, *, scope: str = "global", folder: str | None = None, idx: int = 0,
          score: float = 1.0) -> dict[str, Any]:
    return {"id": f"{doc}-{idx}", "documentId": doc, "documentName": f"{doc}.md", "folderId": folder,
            "scope": scope, "idx": idx, "content": content, "tokenEstimate": 10, "score": score}


SHARED = {"id": "f-shared", "name": "Wspólny", "description": "Opis projektu", "systemPrompt": "FOLDER-PROMPT",
          "extraInfo": "EXTRA", "memoryMode": "shared"}
ISOLATED = {**SHARED, "id": "f-iso", "name": "Izolowany", "memoryMode": "isolated"}
CONVERSATION = {"id": "c1", "systemPrompt": "CONV-PROMPT"}
GLOBAL_MEM = [memory("GLOBAL-MEMORY fact")]
RETRIEVED = [chunk("g", "GLOBAL-CHUNK text"), chunk("s", "SHARED-CHUNK", scope="folder", folder="f-shared"),
             chunk("i", "ISO-CHUNK", scope="folder", folder="f-iso"),
             chunk("o", "OTHER-CHUNK", scope="folder", folder="f-other"),
             chunk("x", "ORPHAN-CHUNK", scope="folder", folder=None)]


def folder_mem(folder: dict[str, Any]) -> list[dict[str, Any]]:
    return [memory(f"FOLDER-MEMORY {folder['id']}", scope="folder", folder=folder["id"])]


# --------------------------------------------------------------------------- #
# Scope rules (reference resolveScope / filterRetrieved)
# --------------------------------------------------------------------------- #


def test_resolve_scope() -> None:
    assert resolve_scope(None) == ContextScope(True, None)
    assert resolve_scope(SHARED) == ContextScope(True, "f-shared")
    assert resolve_scope(ISOLATED) == ContextScope(False, "f-iso")


def test_filter_retrieved() -> None:
    names = lambda chunks: [c["documentId"] for c in chunks]  # noqa: E731
    assert names(filter_retrieved(RETRIEVED, ContextScope(True, None))) == ["g"]
    assert names(filter_retrieved(RETRIEVED, ContextScope(True, "f-shared"))) == ["g", "s"]
    assert names(filter_retrieved(RETRIEVED, ContextScope(False, "f-iso"))) == ["i"]


def _build(folder: dict[str, Any] | None, **kw: Any):
    return build_system_prompt(settings=kw.pop("settings", settings()), folder=folder, conversation=CONVERSATION,
                               query=kw.pop("query", "pytanie"), memories_global=GLOBAL_MEM,
                               memories_folder=folder_mem(folder) if folder else [], retrieved=RETRIEVED,
                               now=NOW, **kw)


def test_no_folder_sees_global_context_only() -> None:
    built = _build(None)
    s = built.system
    for marker in ("GLOBAL-PROMPT", "## About the user", "Filip", "GLOBAL-MEMORY", "CONV-PROMPT", "GLOBAL-CHUNK"):
        assert marker in s
    for marker in ("SHARED-CHUNK", "ISO-CHUNK", "OTHER-CHUNK", "## Project"):
        assert marker not in s
    assert built.used_global and not built.used_folder
    assert [c["documentId"] for c in built.citations] == ["g"]


def test_shared_folder_sees_global_and_project() -> None:
    built = _build(SHARED)
    s = built.system
    order = [s.index(m) for m in ("You are a helpful", "GLOBAL-PROMPT", "## About the user", "GLOBAL-MEMORY",
                                  "## Project: Wspólny", "FOLDER-PROMPT", "EXTRA", "FOLDER-MEMORY f-shared",
                                  "CONV-PROMPT", "## Source material")]
    assert order == sorted(order), "blocks must follow the reference order"
    assert "GLOBAL-CHUNK" in s and "SHARED-CHUNK" in s
    assert "ISO-CHUNK" not in s and "OTHER-CHUNK" not in s and ISOLATION_NOTE not in s
    assert built.used_global and built.used_folder
    assert [c["documentId"] for c in built.citations] == ["g", "s"]
    assert "[1] g.md (fragment 0)" in s and "[2] s.md (fragment 0)" in s


def test_isolated_folder_sees_nothing_global() -> None:
    built = _build(ISOLATED)
    s = built.system
    for marker in ("GLOBAL-PROMPT", "## About the user", "Filip", "GLOBAL-MEMORY", "Long-term memory",
                   "GLOBAL-CHUNK", "SHARED-CHUNK", "OTHER-CHUNK"):
        assert marker not in s, marker
    for marker in ("## Project: Izolowany", "FOLDER-PROMPT", "FOLDER-MEMORY f-iso", "ISO-CHUNK", ISOLATION_NOTE,
                   "CONV-PROMPT"):
        assert marker in s, marker
    assert s.index("FOLDER-MEMORY") < s.index(ISOLATION_NOTE)
    assert not built.used_global and built.used_folder
    assert [c["documentId"] for c in built.citations] == ["i"]


def test_memory_switch_profile_switch_and_empty_query() -> None:
    off = _build(SHARED, settings=settings(memoryEnabled=False))
    assert "GLOBAL-MEMORY" not in off.system and "FOLDER-MEMORY" not in off.system
    no_profile = _build(None, settings=settings(profile={"enabled": False}))
    assert "## About the user" not in no_profile.system
    no_query = _build(SHARED, query="   ")
    assert no_query.citations == [] and "## Source material" not in no_query.system
    assert "Today's date: Tuesday, 22 September 2026." in off.system


def test_memory_ordering_and_caps() -> None:
    entries = [memory(f"m{i}", updated=i) for i in range(50)] + [memory("pinned", pinned=True, updated=-1),
                                                                  memory("   ")]
    ordered = order_memories(entries)
    assert ordered[0]["content"] == "pinned"
    assert [e["content"] for e in ordered[1:4]] == ["m49", "m48", "m47"]
    assert len(ordered) == MAX_MEMORIES
    long = build_system_prompt(settings=settings(), memories_global=[memory("x" * 1000, kind="preference")], now=NOW)
    line = next(line for line in long.system.split("\n") if line.startswith("- [preference]"))
    assert len(line) <= len("- [preference] ") + 400


def test_citation_shape() -> None:
    built = build_system_prompt(settings=settings(), query="q", now=NOW,
                                retrieved=[chunk("g", "słowo " * 100, idx=3, score=0.25)])
    [c] = built.citations
    assert set(c) == {"documentId", "documentName", "chunkIndex", "snippet", "score"}
    assert c["chunkIndex"] == 3 and c["score"] == 0.25 and len(c["snippet"]) <= 240 and c["snippet"].endswith("…")


def test_title_prompt_and_compose_text() -> None:
    prompt = build_title_prompt("  Jak   zrobić\n pierogi? ")
    assert "at most 6 words" in prompt and prompt.endswith("Jak zrobić pierogi?")
    assert build_title_prompt("").endswith("(empty message)")
    msg = {"content": "Zobacz plik", "attachments": [
        {"name": "a.txt", "kind": "text", "textContent": "zawartość"},
        {"name": "img.png", "kind": "image", "textContent": None}]}
    text = compose_text(msg)
    assert text.startswith("Zobacz plik") and "--- Attachment: a.txt ---\nzawartość" in text
    assert "img.png" not in text


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #


def history(n_turns: int, *, words: int = 60) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    seq = 0
    for i in range(n_turns):
        for role in ("user", "assistant"):
            seq += 1
            msgs.append({"id": f"m{seq}", "role": role, "content": f"{role}-{i} " + "słowo " * words,
                         "status": "complete", "seq": seq, "attachments": [],
                         "reasoning": "SECRET-REASONING" if role == "assistant" else None})
    seq += 1
    msgs.append({"id": "current", "role": "user", "content": "CURRENT QUESTION", "status": "complete",
                 "seq": seq, "attachments": []})
    msgs.append({"id": "placeholder", "role": "assistant", "content": "", "status": "streaming", "seq": seq + 1,
                 "attachments": []})
    return msgs


async def build(n_ctx: int, hist: list[dict[str, Any]], **kw: Any):
    params: dict[str, Any] = dict(settings=kw.pop("settings", settings()), folder=kw.pop("folder", None),
                                  conversation=kw.pop("conversation", None), history=hist,
                                  memories_global=kw.pop("memories_global", GLOBAL_MEM),
                                  memories_folder=kw.pop("memories_folder", []),
                                  retrieved=kw.pop("retrieved", []), query=kw.pop("query", "CURRENT QUESTION"),
                                  n_ctx=n_ctx, max_tokens=kw.pop("max_tokens", 4096),
                                  counter=kw.pop("counter", counter()), now=NOW)
    params.update(kw)
    return await ContextBuilder(**params).build()


def test_plan_budget() -> None:
    b = plan_budget(8192, 4096)
    assert b.reserve == 2048 and b.prompt == 8192 - 2048 - 81
    assert b.system == int(b.prompt * 0.15) and b.rag == int(b.prompt * 0.25)
    assert plan_budget(8192, 300).reserve == 300
    assert plan_budget(8192, None).reserve == 2048
    assert plan_budget(100, 50).n_ctx == 256
    assert plan_budget(8192, 4096, scale=0.5).prompt == int((8192 - 2048 - 81) * 0.5)


async def test_everything_fits_in_a_large_window() -> None:
    ctx = await build(32768, history(3))
    msgs = ctx.messages
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "CURRENT QUESTION"}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant"] * 3 + ["user"]
    assert all("SECRET-REASONING" not in m["content"] for m in msgs)
    assert all("reasoning" not in m for m in msgs)
    assert not ctx.report.trimmed and ctx.report.history_included == 6
    assert ctx.report.summary_pl() is None
    assert ctx.max_output_tokens == 4096


async def test_tiny_window_drops_oldest_history_first() -> None:
    ctx = await build(1024, history(12))
    report = ctx.report
    assert report.dropped_messages > 0 and report.history_included > 0
    assert ctx.prompt_tokens <= 1024 - report.reserve
    contents = [m["content"] for m in ctx.messages[1:]]
    assert contents[-1] == "CURRENT QUESTION"
    assert contents[-2].startswith("assistant-11") and contents[-3].startswith("user-11")   # newest kept
    assert not any(c.startswith("user-0 ") for c in contents)                                # oldest dropped
    assert ctx.messages[1]["role"] == "user"
    assert "pominięto" in report.summary_pl()
    assert ctx.max_output_tokens <= 1024 - ctx.prompt_tokens


async def test_history_limit_setting() -> None:
    ctx = await build(32768, history(5), settings=settings(defaults={"historyLimit": 4}))
    # window = last 4 usable messages incl. the current one → 3 earlier; leading assistant is trimmed
    assert ctx.report.dropped_by_limit == 7
    assert [m["role"] for m in ctx.messages[1:]] == ["user", "assistant", "user"]


async def test_rag_gets_fewer_and_shorter_chunks_when_tight() -> None:
    chunks = [chunk(f"d{i}", f"CHUNK{i} " + "treść " * 200, idx=i, score=1 - i / 10) for i in range(6)]
    big = await build(32768, history(1), retrieved=chunks)
    assert len(big.citations) == 6 and big.report.chunks_shortened == 0
    tight = await build(2048, history(10), retrieved=chunks)
    r = tight.report
    assert 0 < r.chunks_included < 6 and r.chunks_total == 6
    assert [c["documentId"] for c in tight.citations] == [f"d{i}" for i in range(r.chunks_included)]
    system = tight.messages[0]["content"]
    assert f"[{r.chunks_included}] d{r.chunks_included - 1}.md" in system
    assert f"\n[{r.chunks_included + 1}] " not in system
    if r.chunks_shortened:
        assert TRUNCATION_MARKER in system
    assert "fragmentów bazy wiedzy" in r.summary_pl()


async def test_short_history_hands_spare_room_to_rag() -> None:
    chunks = [chunk(f"d{i}", f"CHUNK{i} " + "treść " * 150, idx=i) for i in range(8)]
    lonely = await build(4096, history(0), retrieved=chunks)
    crowded = await build(4096, history(30), retrieved=chunks)
    assert lonely.report.chunks_included > crowded.report.chunks_included
    rag_tokens = await count4(lonely.blocks[-1])
    assert rag_tokens > lonely.report.budget * 0.25     # beyond the soft cap because history is short


async def test_memories_are_trimmed_before_instructions() -> None:
    mems = [memory(f"FAKT-{i} " + "szczegół " * 20, updated=i) for i in range(40)]
    ctx = await build(2048, history(10), memories_global=mems)
    r = ctx.report
    assert r.memories_total == 40 and 0 <= r.memories_included < 40
    system = ctx.messages[0]["content"]
    assert "GLOBAL-PROMPT" in system and "FAKT-39" in system if r.memories_included else True
    assert "wpisów pamięci" in r.summary_pl()


async def test_isolation_note_survives_a_tiny_budget() -> None:
    folder = {**ISOLATED, "systemPrompt": "INSTRUKCJA " * 2000}
    ctx = await build(1024, history(2), folder=folder, memories_folder=folder_mem(folder))
    system = ctx.messages[0]["content"]
    assert ISOLATION_NOTE in system and "GLOBAL-PROMPT" not in system
    assert "folder" in ctx.report.truncated_blocks
    assert ctx.prompt_tokens <= 1024 - ctx.report.reserve


async def test_attachments_and_huge_current_message_are_truncated() -> None:
    hist = history(0)
    hist[0]["content"] = "Przeczytaj plik"
    hist[0]["attachments"] = [
        {"name": "log.txt", "kind": "text", "textContent": "HEAD " + "linia logu\n" * 3000 + " TAIL"},
        {"name": "zdjecie.png", "kind": "image", "textContent": None}]
    ctx = await build(4096, hist)
    current = ctx.messages[-1]["content"]
    assert current.startswith("Przeczytaj plik") and "--- Attachment: log.txt ---" in current
    assert "HEAD" in current and "TAIL" in current and TRUNCATION_MARKER in current      # head + tail kept
    assert "[Image attachment: zdjecie.png" in current
    assert ctx.report.truncated_attachments == ["log.txt"]
    assert ctx.prompt_tokens <= 4096 - ctx.report.reserve

    huge = history(0)
    huge[0]["content"] = "START " + "tekst " * 20000 + " END"
    ctx2 = await build(4096, huge)
    assert ctx2.report.current_truncated
    assert ctx2.messages[-1]["content"].startswith("START") and ctx2.messages[-1]["content"].endswith("END")
    assert ctx2.prompt_tokens <= 4096 - ctx2.report.reserve


async def test_failed_streaming_and_empty_messages_are_skipped() -> None:
    hist = [
        {"id": "a0", "role": "assistant", "content": "orphan greeting", "status": "complete", "attachments": []},
        {"id": "u1", "role": "user", "content": "pierwsze", "status": "complete", "attachments": []},
        {"id": "a1", "role": "assistant", "content": "partial", "status": "error", "attachments": []},
        {"id": "u2", "role": "user", "content": "drugie", "status": "complete", "attachments": []},
        {"id": "a2", "role": "assistant", "content": "", "status": "stopped", "attachments": []},
        {"id": "a3", "role": "assistant", "content": "przerwana", "status": "stopped", "attachments": []},
        {"id": "u3", "role": "user", "content": "trzecie", "status": "complete", "attachments": []},
        {"id": "p", "role": "assistant", "content": "", "status": "streaming", "attachments": []},
    ]
    ctx = await build(8192, hist)
    assert [(m["role"], m["content"]) for m in ctx.messages[1:]] == [
        ("user", "pierwsze\n\ndrugie"), ("assistant", "przerwana"), ("user", "trzecie")]


async def test_exact_counter_enforces_the_hard_limit() -> None:
    calls: list[int] = []

    async def exact(messages: list[dict[str, Any]]) -> int:
        calls.append(len(messages))
        # the template is much more expensive than the piecewise estimate
        return sum(len(m["content"]) // 4 + 60 for m in messages)

    ctx = await build(2048, history(8), exact_count=exact)
    assert ctx.report.exact and calls
    assert ctx.prompt_tokens <= 2048 - ctx.report.reserve
    assert ctx.messages[-1]["content"] == "CURRENT QUESTION"
    assert ctx.messages[1]["role"] == "user"


async def test_exact_counter_failure_falls_back_to_estimate() -> None:
    async def broken(messages: list[dict[str, Any]]) -> int:
        raise RuntimeError("tokenizer down")

    ctx = await build(4096, history(2), exact_count=broken)
    assert not ctx.report.exact and ctx.prompt_tokens > 0


async def test_scale_and_max_output() -> None:
    small = await build(8192, history(40), max_tokens=100)
    assert small.max_output_tokens == 100
    ctx = await ContextBuilder(settings=settings(), folder=None, conversation=None, history=history(40),
                               n_ctx=8192, max_tokens=4096, counter=counter(), now=NOW).build(scale=0.5)
    assert ctx.prompt_tokens <= int((8192 - 2048 - 81) * 0.5) + 16


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def test_token_counter_caches_and_falls_back() -> None:
    calls: list[str] = []

    async def fn(text: str) -> int:
        calls.append(text)
        return 7

    c = TokenCounter(fn)
    assert await c.count("abc") == 7 and await c.count("abc") == 7 and calls == ["abc"]
    assert await c.count("") == 0

    async def failing(text: str) -> int:
        raise ConnectionError

    f = TokenCounter(failing)
    assert await f.count("x" * 30) == 10          # estimate: 3 chars/token
    for _ in range(3):
        await f.count("y")
    assert not f.exact


async def test_truncate_to_tokens_modes() -> None:
    c = counter()
    text = "początek " + "środek " * 500 + "koniec"
    head, cut = await truncate_to_tokens(text, 50, c)
    assert cut and head.startswith("początek") and head.endswith(TRUNCATION_MARKER) and await c.count(head) <= 50
    both, cut2 = await truncate_to_tokens(text, 50, c, mode="head_tail")
    assert cut2 and both.startswith("początek") and both.endswith("koniec") and await c.count(both) <= 50
    same, cut3 = await truncate_to_tokens("krótki", 50, c)
    assert same == "krótki" and not cut3
    assert await truncate_to_tokens("abc", 0, c) == ("", True)


def test_normalize_turns() -> None:
    msgs = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u1"}, {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "b"}]
    assert normalize_turns(msgs) == [{"role": "system", "content": "s"}, {"role": "user", "content": "u1\n\nu2"},
                                     {"role": "assistant", "content": "b"}]
    assert normalize_turns([]) == []


@pytest.mark.parametrize("n_ctx", [512, 1024, 2048, 4096, 8192, 32768])
async def test_budget_invariant_across_sizes(n_ctx: int) -> None:
    chunks = [chunk(f"d{i}", "treść " * 300, idx=i) for i in range(10)]
    mems = [memory("pamięć " * 30, updated=i) for i in range(20)]
    ctx = await build(n_ctx, history(25, words=120), retrieved=chunks, memories_global=mems, folder=SHARED,
                      memories_folder=folder_mem(SHARED), conversation=CONVERSATION)
    assert ctx.prompt_tokens <= max(256, n_ctx) - ctx.report.reserve
    assert ctx.messages[-1]["content"] == "CURRENT QUESTION"
    assert ctx.max_output_tokens >= 16
