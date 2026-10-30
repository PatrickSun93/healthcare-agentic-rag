import json

from src import config as C
from src.eval.evaluate import EVAL_SET, evaluate
from src.prompts import PINNED


def test_evaluate_records_versions_models_and_metrics(indexed_sample_docs, tmp_path, monkeypatch):
    monkeypatch.setattr(C, "VECTOR_BACKEND", "fake-mem")
    result = evaluate(k=5, verbose=False, out_dir=None)

    assert result["prompts"] == PINNED
    assert result["models"]["embedding"].startswith("hash/")
    assert result["models"]["llm"].startswith(f"{C.LLM_PROVIDER}/")
    assert result["retrieval"]["vector_backend"] == "fake-mem"
    assert result["retrieval"]["final_k"] == 5
    assert result["chunking"] == {"max_tokens": C.CHUNK_MAX_TOKENS, "overlap": C.CHUNK_OVERLAP}
    assert result["run_at"].endswith("+00:00")

    n = len(json.loads(EVAL_SET.read_text()))
    m = result["metrics"]
    assert m["n"] == n and m["k"] == 5
    assert 0.0 <= m["mrr"] <= m["recall_at_k"] <= 1.0
    assert len(result["cases"]) == n and {"question", "expected", "got", "hit", "rr"} <= set(result["cases"][0])

    out = tmp_path / "results"
    evaluate(k=5, verbose=False, out_dir=out)
    files = sorted(p.name for p in out.glob("*.json"))
    assert "latest.json" in files and len(files) == 2  # <timestamp>.json + latest.json
    assert json.loads((out / "latest.json").read_text())["metrics"] == m


def test_evaluate_out_dir_none_writes_nothing(indexed_sample_docs, tmp_path):
    before = set(tmp_path.iterdir())  # chunks.json from the index fixture
    evaluate(k=3, verbose=False, out_dir=None)
    assert set(tmp_path.iterdir()) == before
