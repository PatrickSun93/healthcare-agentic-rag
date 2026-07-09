"""Offline indexing loop: load docs -> structure-aware chunk -> embed -> upsert to Qdrant.

Also persists index/chunks.json (used to rebuild the BM25 sparse index and to hydrate results at
query time). Qdrant point id == chunk index == position in chunks.json, so the three stay aligned.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config as C
from .llm import embed_texts
from .vectorstore import get_store


def _window(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Approximate token windowing by whitespace words (good enough for a demo)."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_tokens - overlap)
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), step)]


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Structure-aware: split on markdown headings, keep the section title as metadata."""
    parts = re.split(r"(?m)^(#{1,6}\s.*)$", text)
    chunks: list[dict] = []

    def flush(section: str, body: str) -> None:
        body = body.strip()
        if not body:
            return
        for piece in _window(body, C.CHUNK_MAX_TOKENS, C.CHUNK_OVERLAP):
            chunks.append({"text": piece, "source": source, "section": section})

    flush("intro", parts[0] if parts else text)
    i = 1
    while i < len(parts):
        heading = parts[i].strip().lstrip("#").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        flush(heading, body)
        i += 2
    return chunks


def _read_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from .pdf import pdf_to_text  # lazy import: only needs PyMuPDF when a PDF is present

        return pdf_to_text(path)
    return path.read_text()


def load_docs() -> list[dict]:
    docs_dir = Path(C.DATA_DIR)
    md_stems = {p.stem for p in docs_dir.glob("*.md")}
    chunks: list[dict] = []
    for p in sorted(docs_dir.iterdir()):
        if p.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        if p.suffix.lower() == ".pdf" and p.stem in md_stems:
            print(f"  skip {p.name} (transcript {p.stem}.md already present)")
            continue
        print(f"  loading {p.name}")
        chunks.extend(chunk_markdown(_read_doc(p), p.name))
    return chunks


def main() -> None:
    chunks = load_docs()
    if not chunks:
        raise SystemExit(f"No .md docs found in {C.DATA_DIR}")

    print(f"Chunked {len(chunks)} pieces; embedding via {C.EMBED_PROVIDER}/{C.EMBED_MODEL} ...")
    vectors = embed_texts([c["text"] for c in chunks])
    dim = len(vectors[0])

    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    store = get_store()
    store.recreate(dim)
    store.upsert(list(range(len(chunks))), vectors, chunks)

    C.INDEX_DIR.mkdir(exist_ok=True)
    C.CHUNKS_PATH.write_text(json.dumps(chunks, indent=2))
    print(f"Ingested {len(chunks)} chunks (dim={dim}) into {C.VECTOR_BACKEND} '{C.COLLECTION}' + {C.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
