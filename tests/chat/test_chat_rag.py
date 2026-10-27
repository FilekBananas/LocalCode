"""Knowledge base: chunking parity with ai_ux, ingestion, hybrid retrieval, RRF."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from chat_fakes import EMBED_KEY, FakeEngine, FakeRuntime
from chunk_parity_data import CASES
from localcode.chat import rag as rag_mod
from localcode.chat.rag import (
    RagIndex,
    RuntimeEmbedder,
    blob_to_vector,
    chunk_text,
    cosine_scores,
    embedding_prefixes,
    open_lease,
    rrf_fuse,
    runtime_lease,
    vector_to_blob,
)
from localcode.chat.repo import ChatRepo

TOPICS = {
    "animals": ["kot", "pies", "chomik", "zwierz"],
    "sea": ["statek", "fala", "plaża", "morze"],
    "code": ["python", "sqlite", "funkcja"],
}


class TopicEmbedder:
    """Deterministic 'semantic' embedder: one dimension per topic."""

    def __init__(self, model_key: str = "fake/nomic-embed-test", fail: bool = False) -> None:
        self.model_key = model_key
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    async def embed(self, texts: list[str], *, kind: str) -> list[list[float]]:
        self.calls.append((kind, list(texts)))
        if self.fail:
            raise RuntimeError("embedding model crashed")
        out = []
        for text in texts:
            low = text.lower()
            out.append([float(sum(low.count(w) for w in words)) + 0.01 for words in TOPICS.values()])
        return out


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_chunk_parity_with_reference(case: dict[str, Any]) -> None:
    opts = case["opts"] or {}
    assert chunk_text(case["text"], opts.get("maxChars"), opts.get("overlap")) == case["expected"]


def test_chunk_properties() -> None:
    text = "\n\n".join(f"Akapit {i}. " + "Zdanie o treści. " * 30 for i in range(10))
    chunks = chunk_text(text)
    assert chunks and all(40 <= len(c) <= 1400 for c in chunks)
    # consecutive chunks overlap (the tail of one starts the next)
    assert any(chunks[i + 1][:30] in chunks[i] for i in range(len(chunks) - 1))
    assert chunk_text("") == [] and chunk_text(None) == [] and chunk_text("za krótko") == []


# --------------------------------------------------------------------------- #
# Fusion / vectors
# --------------------------------------------------------------------------- #


def test_rrf_fuse() -> None:
    scores = rrf_fuse([["a", "b", "c"], ["c", "a"]])
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert sorted(scores, key=scores.get, reverse=True) == ["a", "c", "b"]
    assert rrf_fuse([]) == {} and rrf_fuse([[]]) == {}
    # an item ranked well by both lists beats one ranked first by only one list
    fused = rrf_fuse([["x", "both"], ["y", "both"]])
    assert fused["both"] > fused["x"]


def test_vector_blob_roundtrip_and_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = vector_to_blob([3.0, 4.0])
    assert list(blob_to_vector(blob)) == pytest.approx([0.6, 0.8])
    blobs = [vector_to_blob([1, 0]), vector_to_blob([0, 1]), vector_to_blob([1, 1]), vector_to_blob([1, 2, 3])]
    with_numpy = cosine_scores([1, 0], blobs)
    monkeypatch.setitem(sys.modules, "numpy", None)      # force the pure-Python path
    pure = cosine_scores([1, 0], blobs)
    assert with_numpy == pytest.approx(pure)
    assert pure[:3] == pytest.approx([1.0, 0.0, 2 ** -0.5], abs=1e-6)
    assert pure[3] == -1.0                                # dimension mismatch
    assert vector_to_blob([0, 0]) == vector_to_blob([0.0, 0.0])


def test_embedding_prefixes() -> None:
    assert embedding_prefixes(EMBED_KEY) == ("search_query: ", "search_document: ")
    assert embedding_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert embedding_prefixes("some/unknown-embedder") == ("", "")


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


async def test_ingest_success_and_failures(repo: ChatRepo) -> None:
    index = RagIndex(repo)
    text = ("Koty są zwierzętami domowymi, które lubią spać na parapecie. " * 40).encode()
    doc = await index.ingest(name="dir/koty.md", mime="", data=text, scope="global")
    assert doc["status"] == "ready" and doc["error"] is None and doc["chunkCount"] >= 2
    assert doc["name"] == "koty.md" and doc["mime"] == "text/markdown" and doc["folderId"] is None
    assert repo.document_path(doc["id"]).read_bytes() == text

    short = await index.ingest(name="short.txt", mime="text/plain", data=b"Kr\xc3\xb3tko.", scope="global")
    assert short["status"] == "ready" and short["chunkCount"] == 1    # short docs still get one chunk

    empty = await index.ingest(name="e.txt", mime="text/plain", data=b"", scope="global")
    assert empty["status"] == "error" and empty["error"] == "Plik jest pusty"
    nofolder = await index.ingest(name="a.txt", mime="text/plain", data=b"tekst", scope="folder", folder_id=None)
    assert nofolder["error"] == "Brak wskazanego projektu dla dokumentu"
    image = await index.ingest(name="a.png", mime="image/png", data=b"\x89PNG....", scope="global")
    assert image["error"] == "Nie znaleziono tekstu do zaindeksowania"
    binary = await index.ingest(name="a.bin", mime="application/octet-stream", data=b"\x00\x01", scope="global")
    assert binary["status"] == "error" and binary["chunkCount"] == 0


async def test_ingest_extraction_crash_is_reported(repo: ChatRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any) -> None:
        raise RuntimeError("zepsute")
    monkeypatch.setattr(rag_mod, "chunk_text", boom)
    doc = await RagIndex(repo).ingest(name="a.txt", mime="text/plain", data=b"tekst dokumentu", scope="global")
    assert doc["status"] == "error" and "zepsute" in doc["error"]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


async def _seed(repo: ChatRepo, index: RagIndex) -> dict[str, Any]:
    shared = repo.create_folder({"name": "Wspólny", "memoryMode": "shared"})
    isolated = repo.create_folder({"name": "Izolowany", "memoryMode": "isolated"})
    other = repo.create_folder({"name": "Obcy"})
    docs = {
        "global": await index.ingest(name="global.txt", mime="text/plain", scope="global",
                                     data="Globalna notatka: żółw i kot mieszkają w ogrodzie.".encode()),
        "shared": await index.ingest(name="shared.txt", mime="text/plain", scope="folder", folder_id=shared["id"],
                                     data="Projekt wspólny: kot ma na imię Mruczek i lubi ryby.".encode()),
        "isolated": await index.ingest(name="iso.txt", mime="text/plain", scope="folder",
                                       folder_id=isolated["id"],
                                       data="Projekt izolowany: tajny kot agent o kryptonimie Z.".encode()),
        "other": await index.ingest(name="other.txt", mime="text/plain", scope="folder", folder_id=other["id"],
                                    data="Inny projekt: kot sąsiada, nie powinien być widoczny.".encode()),
    }
    return {"docs": docs, "shared": shared, "isolated": isolated, "other": other}


def _doc_names(chunks: list[dict[str, Any]]) -> set[str]:
    return {c["documentName"] for c in chunks}


async def test_retrieve_scope_rules(repo: ChatRepo) -> None:
    index = RagIndex(repo)
    s = await _seed(repo, index)
    # no project → global only
    assert _doc_names(await index.retrieve("kot")) == {"global.txt"}
    # shared project → its own + global
    assert _doc_names(await index.retrieve("kot", folder_id=s["shared"]["id"])) == {"global.txt", "shared.txt"}
    # isolated project → only its own
    assert _doc_names(await index.retrieve("kot", folder_id=s["isolated"]["id"], include_global=False)) == {"iso.txt"}
    # isolated without a folder id searches nothing
    assert await index.retrieve("kot", folder_id=None, include_global=False) == []
    assert await index.retrieve("kot", top_k=0) == []
    assert await index.retrieve("   ") == []
    hits = await index.retrieve("kot", folder_id=s["shared"]["id"])
    assert all(set(h) == {"id", "documentId", "folderId", "scope", "idx", "content", "tokenEstimate",
                          "documentName", "score"} for h in hits)
    assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)


async def test_retrieve_folds_diacritics_and_skips_unready(repo: ChatRepo) -> None:
    index = RagIndex(repo)
    s = await _seed(repo, index)
    assert _doc_names(await index.retrieve("zolw")) == {"global.txt"}
    assert _doc_names(await index.retrieve("ŻÓŁW")) == {"global.txt"}
    repo.update_document(s["docs"]["global"]["id"], {"status": "processing"})
    assert await index.retrieve("zolw") == []


async def test_hybrid_retrieval_fuses_dense_hits(repo: ChatRepo) -> None:
    embedder = TopicEmbedder()
    index = RagIndex(repo, lambda: embedder)
    a = await index.ingest(name="chomik.txt", mime="text/plain", scope="global",
                           data="Chomik to małe zwierzę, które chomikuje jedzenie w policzkach.".encode())
    b = await index.ingest(name="koty.txt", mime="text/plain", scope="global",
                           data="Kot i pies to najpopularniejsze zwierzęta domowe w Polsce.".encode())
    c = await index.ingest(name="morze.txt", mime="text/plain", scope="global",
                           data="Statek płynie po morzu, a fala uderza o plażę.".encode())
    await index.drain()
    assert all(k == "document" for k, _ in embedder.calls)
    assert all(t.startswith("search_document: ") is False for _, batch in embedder.calls for t in batch)

    lexical_only = await RagIndex(repo).retrieve("chomik", embedder=None)
    assert _doc_names(lexical_only) == {"chomik.txt"}

    hybrid = await index.retrieve("chomik")
    names = [h["documentName"] for h in hybrid]
    assert names[0] == "chomik.txt"            # found by both rankings
    assert "koty.txt" in names                 # dense-only (same topic)
    assert "morze.txt" not in names            # unrelated topic stays out
    assert embedder.calls[-1][0] == "query"
    del a, b, c


async def test_embeddings_backfilled_lazily(repo: ChatRepo) -> None:
    plain = RagIndex(repo)
    await plain.ingest(name="k.txt", mime="text/plain", scope="global",
                       data="Kot śpi na kanapie przez cały dzień, bo jest zmęczony.".encode())
    embedder = TopicEmbedder()
    assert len(repo.chunks_missing_embedding(embedder.model_key)) == 1
    index = RagIndex(repo, lambda: embedder)
    first = await index.retrieve("kot")                   # FTS still answers immediately
    assert _doc_names(first) == {"k.txt"}
    await index.drain()
    assert repo.chunks_missing_embedding(embedder.model_key) == []


async def test_embedding_failures_fall_back_to_fts(repo: ChatRepo) -> None:
    embedder = TopicEmbedder(fail=True)
    index = RagIndex(repo, lambda: embedder)
    doc = await index.ingest(name="k.txt", mime="text/plain", scope="global",
                             data="Kot śpi na kanapie przez cały dzień, bo jest zmęczony.".encode())
    await index.drain()
    assert doc["status"] == "ready"
    assert _doc_names(await index.retrieve("kot")) == {"k.txt"}


async def test_runtime_embedder_uses_prefixes_and_releases(repo: ChatRepo) -> None:
    engine = FakeEngine()
    runtime = FakeRuntime({EMBED_KEY: engine})
    embedder = RuntimeEmbedder(runtime, EMBED_KEY)
    vectors = await embedder.embed([f"tekst {i}" for i in range(20)], kind="document")
    assert len(vectors) == 20
    assert [len(batch) for batch in engine.embed_calls] == [16, 4]
    assert engine.embed_calls[0][0] == "search_document: tekst 0"
    await embedder.embed(["pytanie"], kind="query")
    assert engine.embed_calls[-1] == ["search_query: pytanie"]
    assert runtime.acquired[0] == (EMBED_KEY, "embedding")
    assert runtime.active == 0 and runtime.released == 2


async def test_open_lease_accepts_every_acquire_shape() -> None:
    engine = object()
    released: list[str] = []

    class CtxRuntime:
        @asynccontextmanager
        async def _cm(self):
            yield SimpleNamespace(engine=engine)
            released.append("cm")

        def acquire(self, key: str, *, purpose: str, n_ctx: int | None = None):
            return self._cm()

    class ReleaseRuntime:
        async def acquire(self, key: str, *, purpose: str):
            return SimpleNamespace(engine=engine, release=lambda: released.append("release"))

    async with runtime_lease(CtxRuntime(), "m", purpose="chat") as e1:
        assert e1 is engine
    handle = await open_lease(ReleaseRuntime(), "m", purpose="chat")
    assert handle.engine is engine
    await handle.release()
    await handle.release()                     # idempotent
    assert released == ["cm", "release"]
