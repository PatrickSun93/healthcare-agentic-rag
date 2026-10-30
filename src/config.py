"""Central config, all overridable via .env. Defaults to LOCAL Ollama so PHI never leaves the box."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Providers: "ollama" (local, PHI-safe) | "openai" | "anthropic"; EMBED_PROVIDER also takes "hash" ---
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "ollama")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")
HASH_EMBED_DIM = int(os.getenv("HASH_EMBED_DIM", "256"))  # "hash" embedder: deterministic, model-free (tests/CI)

# Ollama (local)
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # `ollama pull nomic-embed-text`
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3.1")            # `ollama pull llama3.1`

# API fallbacks (only used if provider set accordingly)
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL   = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
ANTHROPIC_LLM_MODEL = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-5")

# --- Vector store: qdrant (local) | pgvector (Postgres) | pinecone (managed) ---
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "healthcare_rag")           # Qdrant collection / Postgres table name
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/healthcare_rag")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "healthcare-rag")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION  = os.getenv("PINECONE_REGION", "us-east-1")

# --- Retrieval knobs ---
TOP_K_DENSE  = int(os.getenv("TOP_K_DENSE", "20"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "20"))
RRF_K        = int(os.getenv("RRF_K", "60"))
FINAL_K      = int(os.getenv("FINAL_K", "5"))
RERANK       = os.getenv("RERANK", "false").lower() == "true"   # true => cross-encoder (heavier deps)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Chunking (approximate word-based token budget) ---
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "400"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "60"))

# --- PDF ingestion / OCR (scanned pages) ---
OCR_BACKEND      = os.getenv("OCR_BACKEND", "tesseract")        # tesseract | ollama | none
OCR_VISION_MODEL = os.getenv("OCR_VISION_MODEL", "llama3.2-vision")
OCR_MIN_CHARS    = int(os.getenv("OCR_MIN_CHARS", "20"))        # < this on a page => treat as scanned, OCR it
OCR_DPI          = int(os.getenv("OCR_DPI", "200"))

# --- Tracing: off | console | otlp (OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT) ---
TRACING           = os.getenv("TRACING", "off")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "healthcare-agentic-rag")

# --- Paths ---
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = Path(os.getenv("DATA_DIR", ROOT / "data" / "sample_docs"))
INDEX_DIR   = ROOT / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
