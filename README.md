# Healthcare Agentic-RAG

[![CI](https://github.com/PatrickSun93/healthcare-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/PatrickSun93/healthcare-agentic-rag/actions/workflows/ci.yml)

A small, **runnable** retrieval-augmented-generation system over clinical documents, built to demonstrate the production RAG skills that agentic-AI roles ask for — **hybrid retrieval, reranking, grounded + cited generation, retrieval-as-an-agent-tool, an evaluation harness with versioned prompts, OpenTelemetry tracing, and a tested CI'd codebase** — running **fully local by default** (Ollama + Qdrant or Postgres/pgvector), so PHI never leaves the box.

> Sample data is **synthetic** (metformin prior-auth, hypertension, sepsis). Not medical advice.

## What it demonstrates

| Capability | Where |
|---|---|
| Structure-aware **chunking** (by markdown section) + metadata | `src/ingest.py` |
| **Embeddings** (local Ollama `nomic-embed-text`, or OpenAI) | `src/llm.py` |
| **Vector store** with metadata filtering — Qdrant, **Postgres/pgvector**, or Pinecone behind one interface | `src/vectorstore.py` |
| **Hybrid retrieval**: dense + BM25 fused with **RRF** | `src/retrieve.py` |
| **Reranking** (cross-encoder, optional) | `src/retrieve.py` |
| **Grounded generation** with inline `[n]` **citations** + refusal-on-no-context | `src/rag.py` |
| **Agentic layer**: router decides *whether* to retrieve; RAG exposed as a tool | `src/agent.py` |
| **Evaluation**: Recall@k + MRR over a golden set, saved as a results record with prompt/model versions | `src/eval/`, `eval_results/` |
| **Prompt versioning**: every system prompt is a pinned `<name>.<version>.txt` | `src/prompts/` |
| **Tracing**: an OpenTelemetry span per stage (route / retrieve / rerank / generate / grade) | `src/tracing.py` |
| **Tests + CI**: service-free unit suite, pgvector integration in GitHub Actions | `tests/`, `.github/workflows/ci.yml` |
| **PHI/tenant isolation** via metadata filter | `--tenant` flag |

## Architecture

```
 Offline:  data/*.md ─▶ chunk (by section) ─▶ embed ─▶ vector store: Qdrant | pgvector | Pinecone
                                                       (+ index/chunks.json for BM25)

 Online:   query ─▶ [router: retrieve?] ─▶ dense ┐
                                                  ├─ RRF ─▶ (rerank) ─▶ grounded LLM ─▶ cited answer
                                          BM25 ───┘
                    └────────────── Evaluation (Recall@k / MRR / RAGAS) ──────────────┘
```

## Quickstart

**Prereqs:** Docker, Python 3.10+, and [Ollama](https://ollama.com) (already installed on this machine).

```bash
# 1. Start a vector store (Qdrant by default — see "Vector backends" for Postgres/pgvector)
docker compose up -d qdrant

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
python cli.py eval                               # Recall@k + MRR -> eval_results/latest.json
```

## Usage

```bash
python cli.py ingest                              # (re)build the index
python cli.py query "<question>"                  # grounded, cited answer
python cli.py query "<question>" --agent          # route through the agent first
python cli.py query "<question>" --tenant payerA  # restrict retrieval (PHI/permission demo)
python cli.py eval [--k 5]                        # retrieval metrics + results record
TRACING=console python cli.py query "..." --graph # same, with one span per stage printed
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

All knobs live in `.env` (see `.env.example`): providers/models, `VECTOR_BACKEND` + `QDRANT_URL` / `PGVECTOR_URL` / `PINECONE_*`, retrieval `TOP_K_*` / `FINAL_K` / `RRF_K`, `RERANK`, chunking sizes, and `TRACING`.

## Vector backends

One interface (`recreate / upsert / search(vector, k, tenant)` in `src/vectorstore.py`), three stores, picked by `VECTOR_BACKEND`. The rest of the pipeline only ever sees `(chunk_id, score)`.

| `VECTOR_BACKEND` | Pick it when | Needs |
|---|---|---|
| `qdrant` (default) | Local dev / a purpose-built ANN engine: payload filters, no schema to manage | `docker compose up -d qdrant` |
| `pgvector` | The records already live in **Postgres** and you want one database — SQL joins (chunks ↔ patients ↔ tenants), transactions, the backups and access controls you already run | `docker compose up -d pgvector` + `pip install "psycopg[binary]"` |
| `pinecone` | Managed/serverless, nothing to operate; note PHI leaves the box | `PINECONE_API_KEY` + `pip install pinecone` |

**pgvector details.** One table per collection — `id` (= index in `chunks.json`), `embedding vector(dim)`, `tenant text`, `payload jsonb` (the whole chunk, so the table is self-describing). Search is cosine distance via `<=>` under an **HNSW** index (`vector_cosine_ops`); the tenant filter is a `WHERE` on an indexed column; scores come back as cosine similarity so all three backends read alike.

```bash
docker compose up -d pgvector
VECTOR_BACKEND=pgvector python cli.py ingest
VECTOR_BACKEND=pgvector python cli.py query "What PA criteria apply to metformin ER?"
VECTOR_BACKEND=pgvector python cli.py eval
```

## LangGraph

The agent graph (`src/graph.py`) expresses route → retrieve → generate as a stateful `StateGraph` with a conditional edge (the natural home for a human-in-the-loop interrupt before `generate`):

```bash
python cli.py query "What PA criteria apply to metformin ER?" --graph
python cli.py query "..." --selfrag   # self-RAG: grades the answer, loops back to retrieve if ungrounded (a real LangGraph cycle)
```

## Evals: how to run, what you get

```bash
python cli.py eval                                  # Recall@5 + MRR over src/eval/eval_set.json
python cli.py eval --k 3
PROMPT_VERSIONS="grounded_answer=v2" python cli.py eval   # A/B a prompt pin without touching code
```

Every run writes `eval_results/<timestamp>.json` and `eval_results/latest.json` (the latter is committed, so a metric change shows up in a diff). The record carries the metrics **and everything that could have moved them** — prompt versions, LLM and embedding model, vector backend, retrieval knobs, chunking, and per-question hits:

```jsonc
// eval_results/latest.json — a real run: Ollama nomic-embed-text + pgvector (abridged)
{
  "run_at": "2026-10-30T22:00:42+00:00",
  "prompts": {"grounded_answer": "v1", "router": "v1", "grader": "v1", "query_rewrite": "v1", "chitchat": "v1"},
  "models": {"llm": "ollama/gemma4:12b", "embedding": "ollama/nomic-embed-text"},
  "retrieval": {"vector_backend": "pgvector", "top_k_dense": 20, "top_k_sparse": 20, "rrf_k": 60, "final_k": 5, "rerank": false},
  "chunking": {"max_tokens": 400, "overlap": 60},
  "eval_set": "eval_set.json",
  "metrics": {"recall_at_k": 1.0, "mrr": 0.833, "n": 6, "k": 5},
  "cases": [
    {"question": "What eGFR is a contraindication for metformin?", "expected": "metformin_prior_auth.md",
     "got": ["metformin_prior_auth.md", "metformin_prior_auth.md", "metformin_prior_auth.md", ...], "hit": true, "rr": 1.0},
    ...
  ]
}
```

Recall saturates on this 6-question synthetic set (see *Experiments*); the point is the **record** — change a prompt, a model or the backend, re-run, and the diff of `latest.json` says exactly what moved and why.

What CI does vs. what needs a model: CI runs the eval against a real **pgvector** service with `EMBED_PROVIDER=hash` (a deterministic feature-hashing embedder — exercises the plumbing, not retrieval quality) and asserts Recall@5 ≥ 0.8 / MRR ≥ 0.6. The real numbers above need Ollama (`nomic-embed-text`) and are a local step: `ollama serve`, `python cli.py ingest`, `python cli.py eval`. Generation-side metrics (faithfulness) are the RAGAS layer — see Roadmap.

## Tracing

Every stage is an OpenTelemetry span with `rag.*` attributes (tenant, k, hit counts, prompt version, model, grounded verdict), so a slow or ungrounded answer can be read off a trace instead of re-run:

```
graph.invoke {graph=selfrag}
├─ graph.route ── agent.route {retrieve=true} ── llm.chat {model=ollama/gemma4:12b}
├─ graph.retrieve ── retrieve {final_k=5, n_fused=14, n_results=5}
│     ├─ retrieve.dense {backend=pgvector, k=20, n_hits=14} ── embed {model=ollama/nomic-embed-text}
│     ├─ retrieve.sparse {k=20, n_hits=14}
│     └─ retrieve.rerank {model=cross-encoder/..}        (only with RERANK=true)
├─ graph.generate {prompt=grounded_answer@v1, n_contexts=5} ── llm.chat
├─ graph.grade {grounded=false, attempt=1} ── llm.chat
├─ graph.rewrite {attempt=2} ── llm.chat
└─ graph.retrieve … graph.generate … graph.grade {grounded=true, attempt=2}
```

Off by default and dependency-free (`TRACING=off` is a no-op even without the SDK). Switch by env, no vendor SDK:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
TRACING=console python cli.py query "..." --selfrag                       # spans printed to stdout
TRACING=otlp OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  python cli.py query "..." --graph                                        # OTLP/HTTP -> Jaeger / Tempo / any collector
# e.g. docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one   then open http://localhost:16686
```

## Prompt versioning

System prompts are files in `src/prompts/` named `<name>.<version>.txt` (`grounded_answer.v1.txt`, `router.v1.txt`, `grader.v1.txt`, `query_rewrite.v1.txt`, `chitchat.v1.txt`) with the active version pinned in `PINNED`. To change a prompt, add `grounded_answer.v2.txt` and move the pin — shipped versions are never edited in place. The pins are stamped into every eval record and every `generate` span, and `PROMPT_VERSIONS="name=vN,..."` overrides them for one run, which is how you A/B a prompt against the golden set.

## Tests & CI

```bash
uv pip install -r requirements-dev.txt
ruff check .
pytest                                            # unit suite: in-memory store + hash embeddings, no services
docker compose up -d pgvector
PGVECTOR_URL=postgresql://postgres:postgres@localhost:5432/healthcare_rag pytest -m integration
```

- **Unit** (`tests/`): section-aware chunking + windowing, BM25, RRF fusion, tenant isolation on dense/sparse/fused paths, the hash embedder, prompt loading, the eval record, span nesting, and the LangGraph graphs (route → retrieve → generate, chitchat, the self-RAG retry cycle) with a scripted `chat()`.
- **Integration** (`tests/test_pgvector.py`, skipped unless `PGVECTOR_URL` is set): the real Postgres+pgvector backend through ingest → retrieve → eval, plus tenant filtering, idempotent upsert, and identifier validation.
- **CI** (`.github/workflows/ci.yml`): `ruff check` + `pytest` on Python 3.12 with a `pgvector/pgvector:pg16` service container, on every push to `main` and every pull request.

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
  config.py       env-driven settings
  llm.py          embeddings + chat (ollama | openai | anthropic; EMBED_PROVIDER=hash for tests)
  vectorstore.py  one interface, three stores: Qdrant | Postgres/pgvector | Pinecone
  ingest.py       load -> chunk -> embed -> vector store + chunks.json
  retrieve.py     dense + BM25 -> RRF -> (rerank) -> top-k, with tenant filter
  rag.py          grounded, cited generation (refuses when unsupported)
  agent.py        router (retrieve?) + RAG as a callable tool
  graph.py        LangGraph graphs: route -> retrieve -> generate, and the self-RAG cycle
  prompts/        versioned system prompts (<name>.<version>.txt + pins)
  tracing.py      OpenTelemetry spans (off | console | otlp)
  eval/           golden set + Recall@k / MRR -> eval_results/
tests/            pytest: unit (no services) + pgvector integration
cli.py            ingest | query | eval | pdf2md
data/sample_docs/   synthetic clinical docs
docker-compose.yml  Qdrant + Postgres/pgvector
.github/workflows/  CI
```

## How this maps to the job (talking points)

- **"Walk me through a RAG pipeline"** → this repo, end to end.
- **"How do you reduce hallucination?"** → grounding prompt + inline citations + refuse-on-no-context (`rag.py`); measured by an eval set, not vibes.
- **"Dense vs hybrid?"** → clinical text has exact tokens (ICD-10 `E11.9`, `eGFR`, `lactate`) — BM25 + RRF beats pure vector; see `retrieve.py`.
- **"How do you evaluate RAG?"** → `eval/` Recall@k + MRR as regression gates, every run saved with the prompt/model versions that produced it (RAGAS faithfulness is the next layer).
- **"Postgres or a vector database?"** → both are implemented behind one interface here; the table under *Vector backends* is the answer, and the integration tests run on pgvector.
- **"How do you debug a bad answer in production?"** → one OpenTelemetry trace per request: which prompt version, how many contexts, whether the grader passed, how long each stage took.
- **"Agentic?"** → retrieval is one tool the router chooses to call — the same "retrieval as a tool + human-in-the-loop" pattern from my multi-agent projects.
- **Healthcare/PHI** → local-first (Ollama, no data egress) + tenant/metadata filtering.

## Roadmap

- [ ] RAGAS faithfulness / context-recall in the eval harness (`pip install ragas datasets`), recorded in the same results file
- [ ] Contextual Retrieval (Anthropic): prepend chunk context before embedding
- [ ] Qdrant-native hybrid (sparse vectors) instead of in-memory BM25
- [ ] Wrap the RAG tool in a LangGraph node with a human-in-the-loop checkpoint
- [ ] FHIR / structured-record ingestion; chunk-level (not just source-level) eval labels
```
