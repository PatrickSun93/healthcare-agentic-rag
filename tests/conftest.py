"""Shared fixtures. Every test runs with EMBED_PROVIDER=hash (deterministic, no Ollama) and, unless it
opts into the pgvector integration marker, against an in-memory vector store — so the unit suite needs
no services at all."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src import config as C
from src import ingest, retrieve, vectorstore
from src.llm import hash_embed

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class FakeStore:
    """Minimal in-memory implementation of the vectorstore interface (cosine, tenant filter)."""

    def __init__(self):
        self.dim = None
        self.rows: dict[int, tuple[list[float], dict]] = {}

    def recreate(self, dim: int) -> None:
        self.dim = dim
        self.rows = {}

    def upsert(self, ids, vectors, payloads) -> None:
        for i, v, p in zip(ids, vectors, payloads):
            self.rows[i] = (v, p)

    def search(self, vector, k, tenant=None):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(x * x for x in b)) or 1.0
            return dot / (na * nb)

        hits = [
            (i, cos(vector, v)) for i, (v, p) in self.rows.items() if tenant is None or p.get("tenant") == tenant
        ]
        return sorted(hits, key=lambda x: x[1], reverse=True)[:k]


@pytest.fixture(autouse=True)
def _hash_embeddings(monkeypatch):
    monkeypatch.setattr(C, "EMBED_PROVIDER", "hash")
    retrieve.reset_cache()
    yield
    retrieve.reset_cache()


@pytest.fixture
def fake_store(monkeypatch):
    """Route both ingest and retrieve at one in-memory store."""
    store = FakeStore()
    monkeypatch.setattr(vectorstore, "get_store", lambda: store)
    monkeypatch.setattr(ingest, "get_store", lambda: store)
    return store


@pytest.fixture
def tmp_index(monkeypatch, tmp_path):
    """Point chunks.json at a temp dir so tests never touch the repo's index/."""
    monkeypatch.setattr(C, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(C, "CHUNKS_PATH", tmp_path / "chunks.json")
    return tmp_path


@pytest.fixture
def indexed_sample_docs(fake_store, tmp_index, monkeypatch):
    """Full offline path on the synthetic corpus: load -> chunk -> hash-embed -> fake store + chunks.json."""
    monkeypatch.setattr(C, "DATA_DIR", SAMPLE_DOCS)
    ingest.main()
    retrieve.reset_cache()
    return retrieve._chunks()


def write_chunks(path: Path, chunks: list[dict]) -> None:
    import json

    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    path.write_text(json.dumps(chunks))


def embed_chunks_into(store, chunks: list[dict]) -> None:
    store.recreate(C.HASH_EMBED_DIM)
    store.upsert(list(range(len(chunks))), [hash_embed(c["text"]) for c in chunks], chunks)
