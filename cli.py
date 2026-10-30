"""Entry point. Run from the project root.

    python cli.py ingest
    python cli.py query "What PA criteria apply to metformin ER?"
    python cli.py query "hi there" --agent          # router decides retrieval is not needed
    python cli.py eval
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _print_result(r: dict) -> None:
    print("\n=== ANSWER ===\n" + r["answer"])
    if r.get("search_query"):
        print(f"\n(agent search query: {r['search_query']!r}; used_retrieval={r.get('used_retrieval')})")
    if r.get("attempts"):
        print(f"(self-RAG: {r['attempts']} attempt(s); grounded={r.get('grounded')})")
    if r.get("citations"):
        print("\n=== CITATIONS ===")
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['source']} § {c['section']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Healthcare Agentic-RAG demo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="chunk + embed sample docs into Qdrant")

    q = sub.add_parser("query", help="ask a grounded, cited question")
    q.add_argument("question")
    q.add_argument("--tenant", default=None, help="restrict retrieval to a tenant (PHI/permission demo)")
    q.add_argument("--agent", action="store_true", help="route through the agentic layer first")
    q.add_argument("--graph", action="store_true", help="run via the LangGraph agent graph")
    q.add_argument("--selfrag", action="store_true", help="LangGraph self-RAG loop (grade + retry if ungrounded)")

    ev = sub.add_parser("eval", help="run Recall@k + MRR over the golden set; writes eval_results/<ts>.json")
    ev.add_argument("--k", type=int, default=5)

    pdf = sub.add_parser("pdf2md", help="convert PDF(s) to .md transcripts (text layer + OCR fallback)")
    pdf.add_argument("path", help="a .pdf file or a directory containing PDFs")
    pdf.add_argument("--out", default=None, help="output dir for transcripts (default: the docs dir)")

    args = ap.parse_args()

    if args.cmd == "ingest":
        from src.ingest import main as run_ingest

        run_ingest()
    elif args.cmd == "query":
        if args.selfrag:
            from src.graph import answer_selfrag

            _print_result(answer_selfrag(args.question, tenant=args.tenant))
        elif args.graph:
            from src.graph import answer as graph_answer

            _print_result(graph_answer(args.question, tenant=args.tenant))
        elif args.agent:
            from src.agent import answer

            _print_result(answer(args.question, tenant=args.tenant))
        else:
            from src import rag

            _print_result(rag.query(args.question, tenant=args.tenant))
    elif args.cmd == "eval":
        from src.eval.evaluate import evaluate

        evaluate(k=args.k)
    elif args.cmd == "pdf2md":
        from src import config as C
        from src.pdf import pdf_to_md_file

        src_path = Path(args.path)
        out_dir = Path(args.out) if args.out else C.DATA_DIR
        pdfs = [src_path] if src_path.suffix.lower() == ".pdf" else sorted(src_path.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDFs found at {src_path}")
        for pdf_path in pdfs:
            print(f"✅ {pdf_path.name} -> {pdf_to_md_file(pdf_path, out_dir)}")


if __name__ == "__main__":
    main()
