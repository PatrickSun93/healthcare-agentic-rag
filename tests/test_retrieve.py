"""Sparse (BM25), RRF fusion, tenant filtering and the end-to-end retrieve() path — all on the in-memory
store with hash embeddings."""
from src import config as C
from src import retrieve as R
from tests.conftest import embed_chunks_into, write_chunks

TENANT_CHUNKS = [
    {"text": "metformin extended-release prior authorization criteria eGFR", "source": "a.md", "section": "PA",
     "tenant": "payerA"},
    {"text": "metformin immediate-release step therapy", "source": "b.md", "section": "PA", "tenant": "payerB"},
    {"text": "stage 2 hypertension systolic 140 diastolic 90", "source": "c.md", "section": "BP", "tenant": "payerA"},
    {"text": "sepsis bundle lactate blood cultures antibiotics", "source": "d.md", "section": "Bundle",
     "tenant": "payerB"},
]


def _tenant_corpus(fake_store, tmp_index):
    write_chunks(C.CHUNKS_PATH, [dict(c) for c in TENANT_CHUNKS])
    embed_chunks_into(fake_store, R._chunks())
    R.reset_cache()


# ---------------- RRF ----------------
def test_rrf_prefers_items_in_both_lists():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(3, 5.0), (4, 4.0)]
    fused = R.rrf(dense, sparse, k=60)
    order = [cid for cid, _ in fused]
    assert order[0] == 3  # only id in both lists
    assert set(order) == {1, 2, 3, 4}
    assert fused[0][1] == 1 / (60 + 2) + 1 / (60 + 0)


def test_rrf_ignores_raw_scores_only_ranks_matter():
    assert R.rrf([(1, 1000.0)], [(2, 0.001)]) == [(1, 1 / 60), (2, 1 / 60)]


def test_rrf_empty_inputs():
    assert R.rrf([], []) == []


# ---------------- BM25 ----------------
def test_sparse_search_hits_exact_clinical_token(indexed_sample_docs):
    hits = R.sparse_search("lactate", k=3)
    top = indexed_sample_docs[hits[0][0]]
    assert "lactate" in top["text"].lower()
    assert hits[0][1] > 0


def test_sparse_search_tenant_post_filter(fake_store, tmp_index):
    _tenant_corpus(fake_store, tmp_index)
    hits = R.sparse_search("metformin", k=10, tenant="payerB")
    assert hits, "expected a payerB chunk mentioning metformin"
    assert {R._chunks()[i]["tenant"] for i, _ in hits} == {"payerB"}


# ---------------- dense + tenant ----------------
def test_dense_search_respects_tenant(fake_store, tmp_index):
    _tenant_corpus(fake_store, tmp_index)
    hits = R.dense_search("metformin prior authorization", k=10, tenant="payerA")
    assert {R._chunks()[i]["tenant"] for i, _ in hits} == {"payerA"}
    assert R._chunks()[hits[0][0]]["source"] == "a.md"


# ---------------- end to end ----------------
def test_retrieve_returns_hydrated_chunks_in_order(indexed_sample_docs):
    out = R.retrieve("What lactate level triggers the sepsis protocol?", final_k=3)
    assert len(out) == 3
    assert {"text", "source", "section", "chunk_id"} <= set(out[0])
    assert out[0]["source"] == "sepsis_bundle.md"


def test_retrieve_tenant_isolation(fake_store, tmp_index):
    _tenant_corpus(fake_store, tmp_index)
    for tenant in ("payerA", "payerB"):
        out = R.retrieve("metformin", tenant=tenant, final_k=5)
        assert out and {c["tenant"] for c in out} == {tenant}
    assert R.retrieve("metformin", tenant="nobody") == []


def test_retrieve_final_k_caps_results(indexed_sample_docs):
    assert len(R.retrieve("hypertension", final_k=1)) == 1
    assert len(R.retrieve("hypertension", final_k=2)) == 2


def test_reset_cache_picks_up_reingest(fake_store, tmp_index):
    write_chunks(C.CHUNKS_PATH, [{"text": "alpha", "source": "a.md", "section": "s"}])
    embed_chunks_into(fake_store, R._chunks())
    assert len(R._chunks()) == 1
    write_chunks(C.CHUNKS_PATH, [{"text": "alpha", "source": "a.md", "section": "s"},
                                 {"text": "beta", "source": "b.md", "section": "s"}])
    assert len(R._chunks()) == 1  # still cached
    R.reset_cache()
    assert len(R._chunks()) == 2
