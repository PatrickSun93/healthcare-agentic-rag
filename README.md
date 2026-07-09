# Healthcare Agentic-RAG

A small, **runnable** retrieval-augmented-generation system over clinical documents, built to demonstrate the production RAG skills that agentic-AI roles ask for — **hybrid retrieval, reranking, grounded + cited generation, retrieval-as-an-agent-tool, and an evaluation harness** — running **fully local by default** (Ollama + Qdrant), so PHI never leaves the box.

> Sample data is **synthetic** (metformin prior-auth, hypertension, sepsis). Not medical advice.

## What it demonstrates

| Capability | Where |
|---|---|
| Structure-aware **chunking** (by markdown section) + metadata | `src/ingest.py` |
| **Embeddings** (local Ollama `nomic-embed-text`, or OpenAI) | `src/llm.py` |
| **Vector store** with metadata filtering (Qdrant) | `src/retrieve.py` |
| **Hybrid retrieval**: dense + BM25 fused with **RRF** | `src/retrieve.py` |
| **Reranking** (cross-encoder, optional) | `src/retrieve.py` |
| **Grounded generation** with inline `[n]` **citations** + refusal-on-no-context | `src/rag.py` |
| **Agentic layer**: router decides *whether* to retrieve; RAG exposed as a tool | `src/agent.py` |
| **Evaluation**: Recall@k + MRR over a golden set (RAGAS-ready) | `src/eval/` |
| **PHI/tenant isolation** via metadata filter | `--tenant` flag |

## Architecture

```
 Offline:  data/*.md ─▶ chunk (by section) ─▶ embed ─▶ Qdrant  (+ index/chunks.json for BM25)

 Online:   query ─▶ [router: retrieve?] ─▶ dense ┐
                                                  ├─ RRF ─▶ (rerank) ─▶ grounded LLM ─▶ cited answer
                                          BM25 ───┘
                    └────────────── Evaluation (Recall@k / MRR / RAGAS) ──────────────┘
```

## Quickstart

**Prereqs:** Docker, Python 3.10+, and [Ollama](https://ollama.com) (already installed on this machine).

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Start Ollama + pull the local models
ollama serve &                 # or just open the Ollama app
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.1           # generation

# 3. Python env + deps
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env

# 4. Ingest the sample docs, then ask a question
python cli.py ingest
python cli.py query "What prior-authorization criteria apply to metformin ER?"

# 5. Try the agentic router + the eval harness
python cli.py query "hi there" --agent          # router: no retrieval needed
python cli.py eval                               # Recall@k + MRR
```

## Usage

```bash
python cli.py ingest                              # (re)build the index
python cli.py query "<question>"                  # grounded, cited answer
python cli.py query "<question>" --agent          # route through the agent first
python cli.py query "<question>" --tenant payerA  # restrict retrieval (PHI/permission demo)
python cli.py eval                                # retrieval metrics
```

Switch to a cross-encoder reranker: `pip install sentence-transformers` then set `RERANK=true` in `.env`.
Switch to cloud models: set `LLM_PROVIDER=openai` (or `anthropic`) + the key in `.env`.

## PDF ingestion (text + scanned/OCR)

Drop PDFs into `data/sample_docs/` (or any `DATA_DIR`) — `ingest` accepts `.md`, `.txt`, and `.pdf`.
Each page is read from the **text layer**; a page with almost no text (a scan/photo) is rendered and
**OCR'd**. Transcripts are page-delimited (`## Page N`) so answers can cite the **page**.

```bash
python cli.py pdf2md report.pdf                    # -> report.md transcript (inspect before ingesting)
python cli.py pdf2md ./inbox --out data/sample_docs
python cli.py ingest                               # picks up PDFs / transcripts and embeds them
```

OCR backend via `OCR_BACKEND`: **tesseract** (default — `pip install pytesseract pillow` + `brew install tesseract`),
**ollama** (local vision model, no system deps — `ollama pull llama3.2-vision`, keeps PHI on-box), or **none** (text only).

## Configuration

All knobs live in `.env` (see `.env.example`): providers/models, `QDRANT_URL`, retrieval `TOP_K_*` / `FINAL_K` / `RRF_K`, `RERANK`, and chunking sizes.

## Vector backends & LangGraph

- **Pluggable vector store** via `VECTOR_BACKEND`: `qdrant` (local, default) or **`pinecone`** (managed — set `PINECONE_API_KEY`). Same pipeline, swap the store (`src/vectorstore.py`).
- **LangGraph** agent graph (`src/graph.py`) expresses route → retrieve → generate as a stateful `StateGraph` with a conditional edge (the natural home for a human-in-the-loop interrupt before `generate`):

```bash
python cli.py query "What PA criteria apply to metformin ER?" --graph
python cli.py query "..." --selfrag   # self-RAG: grades the answer, loops back to retrieve if ungrounded (a real LangGraph cycle)
```

## Experiments

Chunk size / overlap is **data-dependent — tune it, don't guess**:

```bash
python experiments/sweep_chunking.py                                   # default grid
python experiments/sweep_chunking.py --sizes 200 400 600 --overlaps 0 0.1 0.2 --k 5
```

It re-ingests for each (size × overlap), runs the eval, and writes a Recall@k / MRR table to
`experiments/chunking_results.md`. On the tiny synthetic set the source-level metric can saturate
(Recall ≈ 1.0) — the *method* is the point; add chunk-level labels + a bigger corpus to discriminate.

## Project layout

```
src/
  config.py     env-driven settings
  llm.py        embeddings + chat (ollama | openai | anthropic)
  ingest.py     load -> chunk -> embed -> Qdrant + chunks.json
  retrieve.py   dense + BM25 -> RRF -> (rerank) -> top-k, with tenant filter
  rag.py        grounded, cited generation (refuses when unsupported)
  agent.py      router (retrieve?) + RAG as a callable tool
  eval/         golden set + Recall@k / MRR
cli.py          ingest | query | eval
data/sample_docs/   synthetic clinical docs
docker-compose.yml  Qdrant
```

## How this maps to the job (talking points)

- **"Walk me through a RAG pipeline"** → this repo, end to end.
- **"How do you reduce hallucination?"** → grounding prompt + inline citations + refuse-on-no-context (`rag.py`); measured by an eval set, not vibes.
- **"Dense vs hybrid?"** → clinical text has exact tokens (ICD-10 `E11.9`, `eGFR`, `lactate`) — BM25 + RRF beats pure vector; see `retrieve.py`.
- **"How do you evaluate RAG?"** → `eval/` Recall@k + MRR as regression gates (RAGAS faithfulness is the next layer).
- **"Agentic?"** → retrieval is one tool the router chooses to call — the same "retrieval as a tool + human-in-the-loop" pattern from my multi-agent projects.
- **Healthcare/PHI** → local-first (Ollama, no data egress) + tenant/metadata filtering.

## Roadmap

- [ ] RAGAS faithfulness / context-recall in the eval harness (`pip install ragas datasets`)
- [ ] Contextual Retrieval (Anthropic): prepend chunk context before embedding
- [ ] Qdrant-native hybrid (sparse vectors) instead of in-memory BM25
- [ ] Wrap the RAG tool in a LangGraph node with a human-in-the-loop checkpoint
- [ ] FHIR / structured-record ingestion; chunk-level (not just source-level) eval labels
```
