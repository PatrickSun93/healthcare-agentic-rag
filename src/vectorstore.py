"""Pluggable vector store: qdrant (local, default), pgvector (Postgres) or pinecone (managed).
Switch via VECTOR_BACKEND.

Point ids are integers aligned with chunks.json, so all backends stay interchangeable — the rest of
the pipeline only sees (chunk_id, score) tuples and never knows which store is behind them. Every
backend implements the same three methods: recreate(dim), upsert(ids, vectors, payloads),
search(vector, k, tenant) -> [(chunk_id, score)], score = cosine similarity (higher is better).
"""
from __future__ import annotations

import re

from . import config as C


def get_store():
    if C.VECTOR_BACKEND == "qdrant":
        return _QdrantStore()
    if C.VECTOR_BACKEND == "pgvector":
        return _PgVectorStore()
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


class _PgVectorStore:
    """Postgres + the pgvector extension. Needs PGVECTOR_URL and `pip install "psycopg[binary]"`.

    One table per collection: id (== index in chunks.json), embedding vector(dim), tenant (the PHI /
    permission filter column) and payload jsonb (the whole chunk, so the table is self-describing and
    can be joined with the rest of a clinical schema). Search is cosine distance via `<=>` under an
    HNSW index; the score comes back as cosine similarity (1 - distance) to match the other backends.

    Tenant filtering is a plain WHERE on an indexed column. On a large table with a very selective
    tenant, HNSW post-filtering can return < k rows — partition by tenant or use pgvector >= 0.8
    iterative scans (`SET hnsw.iterative_scan = relaxed_order`) if that ever matters.
    """

    def __init__(self):
        import psycopg

        self._conn = psycopg.connect(C.PGVECTOR_URL, autocommit=True)
        self._table = _sql_ident(C.COLLECTION)

    def recreate(self, dim: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {self._table}")
            cur.execute(
                f"CREATE TABLE {self._table} ("
                " id INTEGER PRIMARY KEY,"
                f" embedding vector({int(dim)}) NOT NULL,"
                " tenant TEXT,"
                " payload JSONB NOT NULL)"
            )
            # HNSW over ivfflat: no training step, so it works on an empty table and stays good as rows
            # are upserted. vector_cosine_ops pairs with the `<=>` operator used in search().
            cur.execute(f"CREATE INDEX ON {self._table} USING hnsw (embedding vector_cosine_ops)")
            cur.execute(f"CREATE INDEX ON {self._table} (tenant)")

    def upsert(self, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        from psycopg.types.json import Jsonb

        rows = [(i, _pg_vec(v), p.get("tenant"), Jsonb(p)) for i, v, p in zip(ids, vectors, payloads)]
        with self._conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._table} (id, embedding, tenant, payload)"
                " VALUES (%s, %s::vector, %s, %s)"
                " ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding,"
                " tenant = EXCLUDED.tenant, payload = EXCLUDED.payload",
                rows,
            )

    def search(self, vector: list[float], k: int, tenant: str | None = None):
        where = "WHERE tenant = %(tenant)s" if tenant else ""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %(q)s::vector) AS score FROM {self._table} {where}"
                " ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                {"q": _pg_vec(vector), "tenant": tenant, "k": k},
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]


def _sql_ident(name: str) -> str:
    """COLLECTION becomes a table name; only allow identifier characters (never string-format SQL blindly)."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"COLLECTION {name!r} is not a valid Postgres identifier")
    return name


def _pg_vec(v: list[float]) -> str:
    """pgvector's text input format: '[0.1,0.2,...]'. Avoids needing the pgvector-python adapter."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


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
