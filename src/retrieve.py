"""Online retrieval: dense (vector store) + sparse (BM25) fused with RRF, optional cross-encoder rerank,
optional tenant/PHI metadata filter.

Why hybrid: clinical text is full of exact tokens (drug names, ICD-10 codes) that pure dense retrieval
blurs — BM25 catches those, dense catches paraphrase. RRF fuses the two rank lists. The dense side runs
against whichever vector store VECTOR_BACKEND selects (Qdrant or Pinecone) via vectorstore.get_store().
"""
from __future__ import annotations

import json
from functools import lru_cache

from rank_bm25 import BM25Okapi

from . import config as C
from .llm import embed_query
from .tracing import set_attrs, span


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    return json.loads(C.CHUNKS_PATH.read_text())


@lru_cache(maxsize=1)
def _bm25() -> BM25Okapi:
    return BM25Okapi([c["text"].lower().split() for c in _chunks()])


@lru_cache(maxsize=1)
def _store():
    from .vectorstore import get_store

    return get_store()


def reset_cache() -> None:
    """Drop cached chunks + BM25 index + store handle. Call after re-ingesting."""
    _chunks.cache_clear()
    _bm25.cache_clear()
    _store.cache_clear()


def dense_search(query: str, k: int, tenant: str | None = None) -> list[tuple[int, float]]:
    with span("retrieve.dense", backend=C.VECTOR_BACKEND, k=k, tenant=tenant):
        hits = _store().search(embed_query(query), k, tenant)
        set_attrs(n_hits=len(hits))
        return hits


def sparse_search(query: str, k: int, tenant: str | None = None) -> list[tuple[int, float]]:
    with span("retrieve.sparse", k=k, tenant=tenant):
        chunks = _chunks()
        scores = _bm25().get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if tenant:  # BM25 has no native filter -> post-filter by payload
            order = [i for i in order if chunks[i].get("tenant") == tenant]
        hits = [(i, float(scores[i])) for i in order[:k]]
        set_attrs(n_hits=len(hits))
        return hits


def rrf(dense: list[tuple[int, float]], sparse: list[tuple[int, float]], k: int = C.RRF_K):
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across both lists."""
    fused: dict[int, float] = {}
    for rank, (cid, _) in enumerate(dense):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _) in enumerate(sparse):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _rerank(query: str, cand_ids: list[int]) -> list[int]:
    """Cross-encoder precision re-rank. Lazy import: pulls sentence-transformers (heavier)."""
    from sentence_transformers import CrossEncoder

    with span("retrieve.rerank", model=C.RERANK_MODEL, n_candidates=len(cand_ids)):
        ce = CrossEncoder(C.RERANK_MODEL)
        chunks = _chunks()
        scores = ce.predict([(query, chunks[i]["text"]) for i in cand_ids])
        return [cid for cid, _ in sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)]


def retrieve(query: str, tenant: str | None = None, final_k: int | None = None) -> list[dict]:
    final_k = final_k or C.FINAL_K
    with span("retrieve", tenant=tenant, final_k=final_k, rerank=C.RERANK):
        dense = dense_search(query, C.TOP_K_DENSE, tenant)
        sparse = sparse_search(query, C.TOP_K_SPARSE, tenant)
        fused = rrf(dense, sparse)
        cand_ids = [cid for cid, _ in fused[: max(final_k * 3, C.FINAL_K)]]
        if C.RERANK and cand_ids:
            cand_ids = _rerank(query, cand_ids)
        chunks = _chunks()
        out = [chunks[i] for i in cand_ids[:final_k]]
        set_attrs(n_fused=len(fused), n_results=len(out))
        return out
