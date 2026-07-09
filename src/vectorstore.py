"""Pluggable vector store: qdrant (local, default) or pinecone (managed). Switch via VECTOR_BACKEND.

Point ids are integers aligned with chunks.json, so both backends stay interchangeable — the rest of
the pipeline only sees (chunk_id, score) tuples and never knows which store is behind them.
"""
from __future__ import annotations

from . import config as C


def get_store():
    if C.VECTOR_BACKEND == "qdrant":
        return _QdrantStore()
    if C.VECTOR_BACKEND == "pinecone":
        return _PineconeStore()
    raise ValueError(f"Unknown VECTOR_BACKEND: {C.VECTOR_BACKEND}")


class _QdrantStore:
    def __init__(self):
        from qdrant_client import QdrantClient

        self._c = QdrantClient(url=C.QDRANT_URL)

    def recreate(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._c.collection_exists(C.COLLECTION):
            self._c.delete_collection(C.COLLECTION)
        self._c.create_collection(
            C.COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    def upsert(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        self._c.upsert(
            C.COLLECTION,
            points=[PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)],
        )

    def search(self, vector: list[float], k: int, tenant: str | None = None):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        qf = Filter(must=[FieldCondition(key="tenant", match=MatchValue(value=tenant))]) if tenant else None
        resp = self._c.query_points(C.COLLECTION, query=vector, limit=k, query_filter=qf)
        return [(p.id, p.score) for p in resp.points]


class _PineconeStore:
    """Managed serverless Pinecone. Needs PINECONE_API_KEY. Metadata carries source/section/tenant
    for filtering; full text is hydrated from chunks.json, so we keep vectors lean."""

    def __init__(self):
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=C.PINECONE_API_KEY)

    def recreate(self, dim: int) -> None:
        from pinecone import ServerlessSpec

        if C.PINECONE_INDEX in [ix["name"] for ix in self._pc.list_indexes()]:
            self._pc.delete_index(C.PINECONE_INDEX)
        self._pc.create_index(
            name=C.PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=C.PINECONE_CLOUD, region=C.PINECONE_REGION),
        )

    def upsert(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        idx = self._pc.Index(C.PINECONE_INDEX)
        items = [
            {"id": str(i), "values": v,
             "metadata": {k: p[k] for k in ("source", "section", "tenant") if k in p}}
            for i, v, p in zip(ids, vectors, payloads)
        ]
        for start in range(0, len(items), 100):  # Pinecone caps batch size
            idx.upsert(vectors=items[start : start + 100])

    def search(self, vector: list[float], k: int, tenant: str | None = None):
        flt = {"tenant": {"$eq": tenant}} if tenant else None
        res = self._pc.Index(C.PINECONE_INDEX).query(
            vector=vector, top_k=k, include_metadata=False, filter=flt
        )
        return [(int(m["id"]), m["score"]) for m in res["matches"]]
