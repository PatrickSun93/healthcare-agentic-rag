import math

import pytest

from src import config as C
from src.llm import describe, embed_query, embed_texts, hash_embed


def test_hash_embed_is_deterministic_and_unit_norm():
    a, b = hash_embed("lactate greater than 2 mmol/L"), hash_embed("lactate greater than 2 mmol/L")
    assert a == b
    assert len(a) == C.HASH_EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_shared_tokens_mean_higher_similarity():
    def cos(u, v):
        return sum(x * y for x, y in zip(u, v))

    q = hash_embed("what lactate level triggers the sepsis protocol")
    near = hash_embed("initiate the sepsis protocol when lactate is greater than 2 mmol/L")
    far = hash_embed("metformin extended-release prior authorization eGFR")
    assert cos(q, near) > cos(q, far)


def test_punctuation_does_not_split_tokens_apart():
    assert hash_embed("lactate;") == hash_embed("lactate")


def test_embed_texts_routes_to_hash_without_network():
    vecs = embed_texts(["a b", "c d"])
    assert len(vecs) == 2 and len(vecs[0]) == C.HASH_EMBED_DIM
    assert embed_query("a b") == vecs[0]


def test_describe_reports_provider_and_model(monkeypatch):
    monkeypatch.setattr(C, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(C, "LLM_MODEL", "llama3.1")
    assert describe() == {"llm": "ollama/llama3.1", "embedding": f"hash/feature-hash-{C.HASH_EMBED_DIM}d"}


def test_unknown_embed_provider_raises(monkeypatch):
    monkeypatch.setattr(C, "EMBED_PROVIDER", "nope")
    with pytest.raises(ValueError):
        embed_texts(["x"])
