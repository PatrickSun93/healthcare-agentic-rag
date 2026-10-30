"""Integration: the real Postgres+pgvector backend behind the full ingest -> retrieve -> eval path.

Skipped unless PGVECTOR_URL is set. CI starts a pgvector service container; locally:
    docker compose up -d pgvector
    PGVECTOR_URL=postgresql://postgres:postgres@localhost:5432/healthcare_rag pytest -m integration
"""
from __future__ import annotations

import json
import os

import pytest

from src import config as C
from src import ingest
from src import retrieve as R
from src.eval.evaluate import EVAL_SET, evaluate
from src.llm import hash_embed
from src.vectorstore import get_store
from tests.conftest import SAMPLE_DOCS

pytestmark = pytest.mark.integration

if not os.getenv("PGVECTOR_URL"):
    pytest.skip("PGVECTOR_URL not set — no live pgvector to test against", allow_module_level=True)

TABLE = "test_healthcare_rag"


@pytest.fixture
def pg(monkeypatch):
    monkeypatch.setattr(C, "VECTOR_BACKEND", "pgvector")
    monkeypatch.setattr(C, "PGVECTOR_URL", os.environ["PGVECTOR_URL"])
    monkeypatch.setattr(C, "COLLECTION", TABLE)
    store = get_store()
    yield store
    with store._conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")


def _rows(store) -> int:
    with store._conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE}")
        return cur.fetchone()[0]


def test_roundtrip_scores_and_tenant_filter(pg):
    texts = [
        ("metformin extended-release prior authorization eGFR", "payerA"),
        ("metformin immediate-release step therapy", "payerB"),
        ("stage 2 hypertension systolic 140", "payerA"),
        ("sepsis bundle lactate blood cultures", "payerB"),
    ]
    vecs = [hash_embed(t) for t, _ in texts]
    payloads = [{"text": t, "source": f"{i}.md", "section": "s", "tenant": ten, "chunk_id": i}
                for i, (t, ten) in enumerate(texts)]
    pg.recreate(len(vecs[0]))
    pg.upsert(list(range(4)), vecs, payloads)

    hits = pg.search(vecs[0], k=4)
    assert hits[0][0] == 0 and hits[0][1] == pytest.approx(1.0, abs=1e-6)  # self-hit, cosine similarity
    assert [h[0] for h in hits][:2] == [0, 1]  # the other metformin chunk is next
    assert all(-1.0 <= s <= 1.0 for _, s in hits)

    assert {h[0] for h in pg.search(vecs[0], k=10, tenant="payerA")} == {0, 2}
    assert pg.search(vecs[0], k=10, tenant="nobody") == []

    with pg._conn.cursor() as cur:  # payload is real jsonb, not a string
        cur.execute(f"SELECT payload->>'source', tenant FROM {TABLE} WHERE id = 3")
        assert cur.fetchone() == ("3.md", "payerB")


def test_upsert_is_idempotent_and_recreate_wipes(pg):
    v = hash_embed("alpha")
    pg.recreate(len(v))
    pg.upsert([1, 2], [v, v], [{"text": "a"}, {"text": "b"}])
    pg.upsert([1], [v], [{"text": "a2"}])
    assert _rows(pg) == 2
    pg.recreate(len(v))
    assert _rows(pg) == 0


def test_collection_name_must_be_an_identifier(monkeypatch):
    monkeypatch.setattr(C, "VECTOR_BACKEND", "pgvector")
    monkeypatch.setattr(C, "COLLECTION", "rag; drop table users")
    with pytest.raises(ValueError):
        get_store()


def test_ingest_retrieve_eval_on_pgvector(pg, tmp_index, monkeypatch, tmp_path):
    """Whole offline + online path with the real store. Thresholds come from an actual run with the hash
    embedder (dense-only Recall@5 = 1.0, hybrid Recall@5 = 1.0, MRR = 0.92 on the 6-question set); they
    are set a notch below so the test guards the plumbing, not the embedder."""
    monkeypatch.setattr(C, "DATA_DIR", SAMPLE_DOCS)
    ingest.main()
    R.reset_cache()
    assert _rows(pg) == len(R._chunks()) > 0

    cases = json.loads(EVAL_SET.read_text())
    dense_hits = 0
    for case in cases:
        got = [R._chunks()[i]["source"] for i, _ in R.dense_search(case["question"], k=5)]
        dense_hits += case["expected_source"] in got
    assert dense_hits / len(cases) >= 0.8

    m = evaluate(k=5, verbose=False, out_dir=tmp_path)["metrics"]
    assert m["n"] == len(cases)
    assert m["recall_at_k"] >= 0.8
    assert m["mrr"] >= 0.6
