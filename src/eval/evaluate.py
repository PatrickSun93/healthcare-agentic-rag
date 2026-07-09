"""Retrieval evaluation: Recall@k + MRR over a golden set.

This is the piece most candidates skip — "RAG quality as a metric, not a vibe." Wire it into CI so
every change is measured. For generation-side metrics (faithfulness / groundedness) see the RAGAS
note in the README (optional, needs `pip install ragas datasets`).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..retrieve import retrieve

EVAL_SET = Path(__file__).parent / "eval_set.json"


def evaluate(k: int = 5, verbose: bool = True) -> dict:
    cases = json.loads(EVAL_SET.read_text())
    recall_hits, rr_sum = 0, 0.0

    for case in cases:
        got = retrieve(case["question"], final_k=k)
        got_sources = [c["source"] for c in got]
        expected = case["expected_source"]

        hit = expected in got_sources
        recall_hits += int(hit)
        rr = next((1.0 / rank for rank, s in enumerate(got_sources, 1) if s == expected), 0.0)
        rr_sum += rr
        if verbose:
            print(f"{'✅' if hit else '❌'} rr={rr:.2f}  q={case['question'][:52]!r}")
            print(f"     expected={expected}  got={got_sources[:3]}")

    n = len(cases)
    recall, mrr = recall_hits / n, rr_sum / n
    print(f"\nRecall@{k}: {recall_hits}/{n} = {recall:.2f}    MRR: {mrr:.3f}")
    return {"recall_at_k": recall, "mrr": mrr, "n": n, "k": k}


if __name__ == "__main__":
    evaluate()
