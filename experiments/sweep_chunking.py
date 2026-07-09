"""Chunking sweep: for each (chunk_size, overlap), re-ingest and measure Recall@k / MRR.

This is the "don't guess chunk size — sweep and measure" experiment. It turns the chunking knobs
into an empirical decision gated by the eval harness.

Prereq: the stack must be up (docker compose up -d; ollama serve; models pulled). Run from repo root:

    python experiments/sweep_chunking.py
    python experiments/sweep_chunking.py --sizes 200 400 600 800 --overlaps 0 0.1 0.2 --k 5

`--overlaps` values < 1 are a fraction of chunk size (0.1 = 10%); values >= 1 are absolute words.

Note: on the tiny synthetic sample set the source-level metric can saturate (Recall ~ 1.0). The point
is the *method*; add chunk-level labels + a larger corpus to make the numbers discriminate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable from repo root

from src import config as C            # noqa: E402
from src import ingest, retrieve       # noqa: E402
from src.eval.evaluate import evaluate  # noqa: E402


def run(sizes: list[int], overlaps: list[float], k: int) -> list[dict]:
    rows: list[dict] = []
    for size in sizes:
        for frac in overlaps:
            overlap = int(size * frac) if frac < 1 else int(frac)
            # config is a shared module singleton; ingest reads these attributes live.
            C.CHUNK_MAX_TOKENS = size
            C.CHUNK_OVERLAP = overlap
            print(f"\n>>> size={size} overlap={overlap}")
            ingest.main()              # rebuilds the Qdrant collection + chunks.json
            retrieve.reset_cache()     # drop cached chunks/BM25 so eval sees the new index
            metrics = evaluate(k=k, verbose=False)
            rows.append({"size": size, "overlap": overlap, **metrics})
    return rows


def to_markdown(rows: list[dict], k: int) -> str:
    best = max(rows, key=lambda r: (r["recall_at_k"], r["mrr"]))
    lines = [f"| chunk_size | overlap | Recall@{k} | MRR |", "|---|---|---|---|"]
    for r in rows:
        star = "  ⭐ best" if r is best else ""
        lines.append(f"| {r['size']} | {r['overlap']} | {r['recall_at_k']:.2f} | {r['mrr']:.3f}{star} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep chunk size x overlap; measure retrieval quality")
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 400, 600, 800])
    ap.add_argument(
        "--overlaps", type=float, nargs="+", default=[0.0, 0.1, 0.2],
        help="fraction of size if <1 (0.1=10%%), else absolute words",
    )
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    rows = run(args.sizes, args.overlaps, args.k)
    table = to_markdown(rows, args.k)
    print("\n=== Chunking sweep results ===\n" + table)

    out_path = Path(__file__).parent / "chunking_results.md"
    out_path.write_text(f"# Chunking sweep (Recall@{args.k} / MRR)\n\n{table}\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
