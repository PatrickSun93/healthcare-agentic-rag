"""Grounded, cited generation over retrieved context.

Grounding rules live in the versioned `grounded_answer` prompt (src/prompts/): answer only from context,
cite every claim, refuse when unsupported. This is the anti-hallucination core — in healthcare an
unsupported answer must become a refusal.
"""
from __future__ import annotations

from . import prompts
from .llm import chat
from .retrieve import retrieve

GROUNDED = prompts.get("grounded_answer")
SYSTEM = GROUNDED.text

REFUSAL = "I don't have enough grounded information to answer that."  # must match the prompt verbatim


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {c['source']} § {c['section']})\n{c['text']}"
        for n, c in enumerate(chunks, 1)
    )


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    return f"Context:\n{_format_context(chunks)}\n\nQuestion: {question}\n\nGrounded, cited answer:"


def query(question: str, tenant: str | None = None, final_k: int | None = None) -> dict:
    chunks = retrieve(question, tenant=tenant, final_k=final_k)
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "contexts": []}

    answer = chat(SYSTEM, build_user_prompt(question, chunks))
    return {
        "answer": answer,
        "citations": [
            {"n": n, "source": c["source"], "section": c["section"], "chunk_id": c["chunk_id"]}
            for n, c in enumerate(chunks, 1)
        ],
        "contexts": chunks,
    }
