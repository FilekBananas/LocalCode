"""Knowledge base: chunking, ingestion and retrieval (port of ai_ux ``src/lib/rag``).

* :func:`chunk_text` is a line-by-line port of ``chunk.ts``: paragraph- and
  sentence-aware packing into windows of 1400 characters with a 200 character
  overlap cut at a word boundary, dropping fragments shorter than 40 chars.
* :meth:`RagIndex.ingest` ports ``ingest.ts``: the document row is created
  first so every later failure lands on it as ``status="error"`` with a Polish
  message; the original file goes to ``uploads/kb/<id><ext>``.
* Retrieval is hybrid. Lexical ranking uses FTS5 ``bm25()`` over the chunk
  index (diacritics folded, prefix OR query — the same query shape as the
  reference). When an embedding model is configured
  (``settings.chat.embedding_model``, a GGUF embedding model run through the
  runtime with ``purpose="embedding"``) chunks also carry an L2-normalised
  float32 vector and the query is ranked by cosine similarity; both rankings
  are merged with **reciprocal rank fusion** (``k = 60``). Embeddings are
  computed in the background after ingestion and back-filled lazily, so an
  upload never waits for (or fails because of) the embedding model.
* Scope rules are identical to ai_ux: an isolated project sees only its own
  chunks, a shared project sees its chunks plus global ones, a chat outside
  any project sees global chunks only.

:func:`runtime_lease` adapts whatever ``RuntimeManager.acquire`` returns (an
awaitable Lease that is an async context manager, a bare async context
manager, or an object with ``release()``) into one async context manager
yielding the engine; the chat service uses it too.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
from array import array
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Literal, Protocol

from localcode.chat.db import new_id
from localcode.chat.extract import display_name, extract_text, resolve_mime, safe_extension
from localcode.chat.repo import ChatRepo, build_fts_query, estimate_tokens

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Chunking (port of chunk.ts)
# --------------------------------------------------------------------------- #

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP = 200
MIN_CHUNK_CHARS = 40

_SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"')\]]*|$)\s*")


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    """Sentence splitter tuned for Polish and English (breaks after . ! ? … and on newlines)."""
    parts: list[str] = []
    for line in paragraph.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        sentences = _SENTENCE_RE.findall(trimmed)
        if not sentences:
            parts.append(trimmed)
            continue
        for sentence in sentences:
            s = sentence.strip()
            if s:
                parts.append(s)
    return parts if parts else [paragraph.strip()]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort split of an over-long sentence, preferring word boundaries."""
    pieces: list[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        break_at = window.rfind(" ")
        cut = break_at if break_at > max_chars * 0.6 else max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def _tail_overlap(chunk: str, overlap: int) -> str:
    """Tail of a chunk reused as the head of the next one, cut at a word break."""
    if overlap <= 0 or len(chunk) <= overlap:
        return chunk.strip()
    piece = chunk[len(chunk) - overlap:]
    match = re.search(r"\s", piece)
    cut = piece[match.start() + 1:] if match else piece
    return cut.strip()


def chunk_text(text: str | None, max_chars: int | None = None, overlap: int | None = None) -> list[str]:
    """Split ``text`` into overlapping chunks; ``[]`` when nothing is worth indexing."""
    max_c = max(MIN_CHUNK_CHARS * 2, int(max_chars if max_chars is not None else DEFAULT_MAX_CHARS))
    ov = max(0, min(int(overlap if overlap is not None else DEFAULT_OVERLAP), max_c // 2))

    normalized = _normalize_whitespace(text or "")
    if not normalized:
        return []

    # 1. Flatten the document into segments that each fit inside the window.
    segments: list[tuple[str, str]] = []   # (text, separator used when appended)
    for paragraph in _split_paragraphs(normalized):
        if len(paragraph) <= max_c:
            segments.append((paragraph, "\n\n"))
            continue
        first = True
        for sentence in _split_sentences(paragraph):
            pieces = [sentence] if len(sentence) <= max_c else _hard_split(sentence, max_c)
            for piece in pieces:
                segments.append((piece, "\n\n" if first else " "))
                first = False

    # 2. Greedily pack segments, carrying an overlap into every new chunk.
    chunks: list[str] = []
    buffer = ""
    for seg_text, sep in segments:
        if not buffer:
            buffer = seg_text
            continue
        if len(buffer) + len(sep) + len(seg_text) <= max_c:
            buffer += sep + seg_text
            continue
        if buffer.strip():
            chunks.append(buffer.strip())
        carry = _tail_overlap(buffer, ov)
        if carry and len(carry) + len(seg_text) + 2 <= max_c:
            buffer = f"{carry}{sep}{seg_text}"
        else:
            buffer = seg_text
    if buffer.strip():
        chunks.append(buffer.strip())

    # 3. Drop noise and duplicates created by the overlap.
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        if result and result[-1] == chunk:
            continue
        result.append(chunk)
    return result


# --------------------------------------------------------------------------- #
# Fusion and vector helpers
# --------------------------------------------------------------------------- #

RRF_K = 60
DEFAULT_TOP_K = 6
MAX_TOP_K = 20
#: Dense-only hits below this cosine similarity are treated as noise.
DENSE_MIN_SIMILARITY = 0.35
#: Characters of a text sent to the embedding model (≈ 600 tokens).
EMBED_MAX_CHARS = 2000
EMBED_BATCH = 16


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion: ``score(d) = Σ 1 / (k + rank_i(d))`` (ranks from 1)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0 or not math.isfinite(norm):
        return [0.0 for _ in values]
    return [v / norm for v in values]


def vector_to_blob(values: list[float]) -> bytes:
    return array("f", normalize_vector(values)).tobytes()


def blob_to_vector(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    return vec


def cosine_scores(query: list[float], blobs: list[bytes]) -> list[float]:
    """Cosine similarity between ``query`` and each stored (normalised) vector.

    Uses numpy when importable, a pure-Python dot product otherwise. Vectors of
    a different dimension score ``-1``.
    """
    q = normalize_vector(query)
    dim = len(q)
    try:
        import numpy as np  # optional speed-up
    except ImportError:  # pragma: no cover - numpy ships with mlx
        np = None  # type: ignore[assignment]
    if np is not None and blobs:
        qa = np.asarray(q, dtype=np.float32)
        good_idx = [i for i, b in enumerate(blobs) if len(b) == dim * 4]
        result = [-1.0] * len(blobs)
        if good_idx:
            mat = np.frombuffer(b"".join(blobs[i] for i in good_idx), dtype=np.float32).reshape(len(good_idx), dim)
            sims = mat @ qa
            for pos, i in enumerate(good_idx):
                result[i] = float(sims[pos])
        return result
    scores: list[float] = []
    for blob in blobs:
        if len(blob) != dim * 4:
            scores.append(-1.0)
            continue
        vec = blob_to_vector(blob)
        scores.append(sum(a * b for a, b in zip(q, vec)))
    return scores


# --------------------------------------------------------------------------- #
# Runtime lease adapter
# --------------------------------------------------------------------------- #


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class LeaseHandle:
    """An acquired runtime slot: ``engine`` plus an idempotent :meth:`release`."""

    def __init__(self, engine: Any, exit_fn: Callable[[], Any] | None) -> None:
        self.engine = engine
        self._exit_fn = exit_fn
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._exit_fn is not None:
            try:
                await _maybe_await(self._exit_fn())
            except Exception:
                log.warning("releasing runtime lease failed", exc_info=True)


async def open_lease(runtime: Any, model_key: str, *, purpose: str, n_ctx: int | None = None) -> LeaseHandle:
    """Acquire a slot of ``model_key`` from the runtime manager.

    Accepts every reasonable shape of ``RuntimeManager.acquire``: a coroutine
    returning a Lease (async context manager with ``.engine``), a bare async
    context manager, or an object with ``engine`` and ``release()``.
    """
    kwargs: dict[str, Any] = {"purpose": purpose}
    if n_ctx is not None:
        kwargs["n_ctx"] = n_ctx
    obj = runtime.acquire(model_key, **kwargs)
    if inspect.isawaitable(obj) and not hasattr(obj, "__aenter__"):
        obj = await obj
    if hasattr(obj, "__aenter__"):
        entered = await obj.__aenter__()
        engine = getattr(entered, "engine", None) or getattr(obj, "engine", None) or entered

        async def _exit() -> None:
            await obj.__aexit__(None, None, None)

        return LeaseHandle(engine, _exit)
    engine = getattr(obj, "engine", None) or obj
    release = getattr(obj, "release", None)
    return LeaseHandle(engine, release if callable(release) else None)


@asynccontextmanager
async def runtime_lease(runtime: Any, model_key: str, *, purpose: str,
                        n_ctx: int | None = None) -> AsyncIterator[Any]:
    """``async with runtime_lease(rt, key, purpose="embedding") as engine: ...``"""
    handle = await open_lease(runtime, model_key, purpose=purpose, n_ctx=n_ctx)
    try:
        yield handle.engine
    finally:
        await handle.release()


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #

EmbedKind = Literal["query", "document"]


class Embedder(Protocol):
    model_key: str

    async def embed(self, texts: list[str], *, kind: EmbedKind) -> list[list[float]]: ...


#: Task prefixes some embedding families were trained with (matched on the model key).
_EMBED_PREFIXES: list[tuple[str, str, str]] = [
    ("nomic-embed", "search_query: ", "search_document: "),
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("multilingual-e5", "query: ", "passage: "),
    ("e5-", "query: ", "passage: "),
    ("bge-", "Represent this sentence for searching relevant passages: ", ""),
    ("mxbai-embed", "Represent this sentence for searching relevant passages: ", ""),
    ("snowflake-arctic-embed", "Represent this sentence for searching relevant passages: ", ""),
    ("qwen3-embedding", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", ""),
]


def embedding_prefixes(model_key: str) -> tuple[str, str]:
    """``(query_prefix, document_prefix)`` for a model key (``("", "")`` if unknown)."""
    key = model_key.lower()
    for needle, query_prefix, doc_prefix in _EMBED_PREFIXES:
        if needle in key:
            return query_prefix, doc_prefix
    return "", ""


class RuntimeEmbedder:
    """Embeds through ``runtime.acquire(model_key, purpose="embedding")``."""

    def __init__(self, runtime: Any, model_key: str) -> None:
        self.runtime = runtime
        self.model_key = model_key
        self._query_prefix, self._doc_prefix = embedding_prefixes(model_key)

    async def embed(self, texts: list[str], *, kind: EmbedKind) -> list[list[float]]:
        prefix = self._query_prefix if kind == "query" else self._doc_prefix
        prepared = [prefix + (t or " ")[:EMBED_MAX_CHARS] for t in texts]
        out: list[list[float]] = []
        async with runtime_lease(self.runtime, self.model_key, purpose="embedding") as engine:
            for start in range(0, len(prepared), EMBED_BATCH):
                batch = prepared[start:start + EMBED_BATCH]
                vectors = await engine.embed(batch)
                if len(vectors) != len(batch):
                    raise RuntimeError("embedding model returned a wrong number of vectors")
                out.extend([list(map(float, v)) for v in vectors])
        return out


# --------------------------------------------------------------------------- #
# Retrieved chunk
# --------------------------------------------------------------------------- #


@dataclass
class RetrieveParams:
    query: str
    folder_id: str | None = None
    include_global: bool = True
    top_k: int = DEFAULT_TOP_K


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #


class RagIndex:
    """Ingestion + hybrid retrieval over the chat database.

    ``embedder_factory`` returns the embedder to use right now (or ``None`` for
    FTS-only), so a change of ``settings.chat.embedding_model`` takes effect
    without rebuilding the index object.
    """

    def __init__(self, repo: ChatRepo, embedder_factory: Callable[[], Embedder | None] | None = None) -> None:
        self.repo = repo
        self.embedder_factory = embedder_factory or (lambda: None)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._backfill_running: set[str] = set()

    # ------------------------------------------------------------ embeddings
    def _embedder(self) -> Embedder | None:
        try:
            return self.embedder_factory()
        except Exception:
            log.warning("embedder factory failed", exc_info=True)
            return None

    def schedule_embedding(self, document_id: str | None = None) -> asyncio.Task[Any] | None:
        """Embed missing chunks in the background (one backfill per model at a time)."""
        embedder = self._embedder()
        if embedder is None:
            return None
        key = f"{embedder.model_key}:{document_id or '*'}"
        if key in self._backfill_running:
            return None
        self._backfill_running.add(key)

        async def _job() -> None:
            try:
                await self.embed_missing(embedder, document_id=document_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("background embedding failed (%s)", embedder.model_key, exc_info=True)
            finally:
                self._backfill_running.discard(key)

        task = asyncio.get_running_loop().create_task(_job())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def embed_missing(self, embedder: Embedder, *, document_id: str | None = None,
                            batch: int = 64, max_chunks: int = 100_000) -> int:
        """Compute embeddings for ready chunks that lack one from ``embedder``."""
        done = 0
        while done < max_chunks:
            pending = await self.repo.run(self.repo.chunks_missing_embedding, embedder.model_key,
                                          document_id=document_id, limit=batch)
            if not pending:
                break
            vectors = await embedder.embed([content for _, content in pending], kind="document")
            items = [(cid, vector_to_blob(vec), len(vec)) for (cid, _), vec in zip(pending, vectors)]
            await self.repo.run(self.repo.set_chunk_embeddings, embedder.model_key, items)
            done += len(items)
            if len(pending) < batch:
                break
        return done

    async def drain(self) -> None:
        """Wait for background embedding jobs (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -------------------------------------------------------------- ingestion
    async def ingest(self, *, name: str, mime: str | None, data: bytes, scope: str,
                     folder_id: str | None = None) -> dict[str, Any]:
        """Store, extract, chunk and index one document. Never raises for content
        problems — check ``status``/``error`` of the returned KbDocument."""
        scope = "global" if scope == "global" else "folder"
        folder_id = folder_id if scope == "folder" else None
        clean_name = display_name(name, "dokument")
        mime_type = resolve_mime(mime, clean_name)
        doc_id = new_id()
        extension = safe_extension(clean_name)
        relative = f"kb/{doc_id}{extension}"

        await self.repo.run(self.repo.create_document, {
            "id": doc_id, "scope": scope, "folderId": folder_id, "name": clean_name,
            "mime": mime_type, "size": len(data or b""), "path": relative, "status": "processing",
        })

        async def fail(message: str) -> dict[str, Any]:
            updated = await self.repo.run(self.repo.update_document, doc_id,
                                          {"status": "error", "error": message, "chunkCount": 0})
            assert updated is not None
            return updated

        if scope == "folder" and not folder_id:
            return await fail("Brak wskazanego projektu dla dokumentu")
        if not data:
            return await fail("Plik jest pusty")
        try:
            await asyncio.to_thread(self.repo.store_upload, "kb", doc_id, extension, data)
        except OSError as exc:
            return await fail(f"Nie udało się zapisać pliku: {exc.strerror or exc}")

        try:
            result = await asyncio.to_thread(extract_text, data, clean_name, mime_type)
            if result.error:
                return await fail(result.error)
            text = result.text.strip()
            if not text:
                return await fail("Nie znaleziono tekstu do zaindeksowania")
            pieces = await asyncio.to_thread(chunk_text, text)
            # Very short documents still deserve one chunk.
            contents = pieces if pieces else [text]
            await self.repo.run(self.repo.insert_chunks, doc_id, contents,
                                token_estimates=[estimate_tokens(c) for c in contents])
            ready = await self.repo.run(self.repo.update_document, doc_id,
                                        {"status": "ready", "error": None, "chunkCount": len(contents)})
        except Exception as exc:
            log.exception("ingestion of %s failed", clean_name)
            return await fail(f"Nie udało się przetworzyć dokumentu: {exc}")
        self.schedule_embedding(doc_id)
        assert ready is not None
        return ready

    # -------------------------------------------------------------- retrieval
    async def retrieve(self, query: str, *, folder_id: str | None = None, include_global: bool = True,
                       top_k: int = DEFAULT_TOP_K, embedder: Embedder | None | Any = ...) -> list[dict[str, Any]]:
        """Best chunks for ``query`` (best first), as RetrievedChunk dicts:
        ``{id, documentId, folderId, scope, idx, content, tokenEstimate,
        documentName, score}`` where ``score`` is the fused RRF score (higher
        is better). ``[]`` when the scope cannot contain anything."""
        top_k = min(MAX_TOP_K, int(top_k or 0))
        if top_k <= 0 or not (query or "").strip():
            return []
        if not include_global and not folder_id:
            return []
        if embedder is ...:
            embedder = self._embedder()
        candidates = max(top_k * 4, 24)

        lexical: list[dict[str, Any]] = []
        match = build_fts_query(query)
        if match:
            lexical = await self.repo.run(self.repo.search_chunks, match, folder_id=folder_id,
                                          include_global=include_global, limit=candidates)

        dense: list[tuple[dict[str, Any], float]] = []
        if embedder is not None:
            try:
                dense = await self._dense(query, embedder, folder_id=folder_id,
                                          include_global=include_global, limit=candidates)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("dense retrieval failed; using FTS only", exc_info=True)
                dense = []

        by_id: dict[str, dict[str, Any]] = {}
        for row in lexical:
            by_id[row["id"]] = row
        for row, sim in dense:
            existing = by_id.setdefault(row["id"], row)
            existing["similarity"] = sim
        rankings = [[r["id"] for r in lexical]]
        if dense:
            rankings.append([r["id"] for r, _ in dense])
        fused = rrf_fuse(rankings)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        out: list[dict[str, Any]] = []
        for chunk_id, score in ordered:
            row = by_id[chunk_id]
            out.append({
                "id": row["id"],
                "documentId": row["documentId"],
                "folderId": row["folderId"],
                "scope": row["scope"],
                "idx": row["idx"],
                "content": row["content"],
                "tokenEstimate": row["tokenEstimate"],
                "documentName": row["documentName"],
                "score": round(score, 6),
            })
        return out

    async def _dense(self, query: str, embedder: Embedder, *, folder_id: str | None, include_global: bool,
                     limit: int) -> list[tuple[dict[str, Any], float]]:
        rows = await self.repo.run(self.repo.embedded_chunks, embedder.model_key,
                                   folder_id=folder_id, include_global=include_global)
        # Chunks without a vector for this model get one in the background.
        missing = await self.repo.run(self.repo.chunks_missing_embedding, embedder.model_key, limit=1)
        if missing:
            self.schedule_embedding(None)
        if not rows:
            return []
        [query_vec] = await embedder.embed([query[:EMBED_MAX_CHARS]], kind="query")
        sims = await asyncio.to_thread(cosine_scores, query_vec, [r["embedding"] for r in rows])
        ranked = sorted(zip(rows, sims), key=lambda item: item[1], reverse=True)
        out: list[tuple[dict[str, Any], float]] = []
        for row, sim in ranked[:limit]:
            if sim < DENSE_MIN_SIMILARITY:
                break
            clean = {k: v for k, v in row.items() if k not in ("embedding", "embeddingDim")}
            out.append((clean, float(sim)))
        return out
