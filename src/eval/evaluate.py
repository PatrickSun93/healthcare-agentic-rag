"""Retrieval evaluation: Recall@k + MRR over a golden set, written as a versioned results record.

This is the piece most candidates skip — "RAG quality as a metric, not a vibe." Every run writes
eval_results/<timestamp>.json (+ latest.json) carrying the metrics AND what produced them: prompt
versions, LLM + embedding model, vector backend, retrieval knobs, chunking. A regression is then a diff
between two records, not a memory. CI runs this against pgvector with the hash embedder (plumbing
check); the real numbers need Ollama — see README. Generation-side metrics (faithfulness) are the
RAGAS layer (optional, `pip install ragas datasets`).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .. import config as C
from .. import llm, prompts
from ..retrieve import retrieve

EVAL_SET = Path(__file__).parent / "eval_set.json"
EVAL_RESULTS_DIR = C.ROOT / "eval_results"


def run_metadata(k: int) -> dict:
    """Everything that can move a metric, pinned alongside it."""
    retrieval = {
        "vector_backend": C.VECTOR_BACKEND,
        "top_k_dense": C.TOP_K_DENSE,
        "top_k_sparse": C.TOP_K_SPARSE,
        "rrf_k": C.RRF_K,
        "final_k": k,
        "rerank": C.RERANK,
    }
    if C.RERANK:
        retrieval["rerank_model"] = C.RERANK_MODEL
    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompts": prompts.versions(),
        "models": llm.describe(),
        "retrieval": retrieval,
        "chunking": {"max_tokens": C.CHUNK_MAX_TOKENS, "overlap": C.CHUNK_OVERLAP},
        "eval_set": EVAL_SET.name,
    }


def evaluate(k: int = 5, verbose: bool = True, out_dir: Path | None = EVAL_RESULTS_DIR) -> dict:
    cases = json.loads(EVAL_SET.read_text())
    recall_hits, rr_sum = 0, 0.0
    per_case: list[dict] = []

    for case in cases:
        got = retrieve(case["question"], final_k=k)
        got_sources = [c["source"] for c in got]
        expected = case["expected_source"]

        hit = expected in got_sources
        recall_hits += int(hit)
        rr = next((1.0 / rank for rank, s in enumerate(got_sources, 1) if s == expected), 0.0)
        rr_sum += rr
        per_case.append({"question": case["question"], "expected": expected, "got": got_sources, "hit": hit, "rr": rr})
        if verbose:
            print(f"{'✅' if hit else '❌'} rr={rr:.2f}  q={case['question'][:52]!r}")
            print(f"     expected={expected}  got={got_sources[:3]}")

    n = len(cases)
    recall, mrr = recall_hits / n, rr_sum / n
    result = {
        **run_metadata(k),
        "metrics": {"recall_at_k": recall, "mrr": mrr, "n": n, "k": k},
        "cases": per_case,
    }
    if verbose:
        print(f"\nRecall@{k}: {recall_hits}/{n} = {recall:.2f}    MRR: {mrr:.3f}")
        print(f"prompts={result['prompts']}  models={result['models']}  backend={C.VECTOR_BACKEND}")
    if out_dir is not None:
        path = write_result(result, Path(out_dir))
        if verbose:
            print(f"saved -> {path} (+ latest.json)")
    return result


def write_result(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result["run_at"].replace(":", "").replace("+0000", "Z")
    path = out_dir / f"{stamp}.json"
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    path.write_text(payload)
    (out_dir / "latest.json").write_text(payload)
    return path


if __name__ == "__main__":
    evaluate()
