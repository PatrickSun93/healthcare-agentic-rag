"""Grounded, cited generation over retrieved context.

Grounding rules live in SYSTEM: answer only from context, cite every claim, refuse when unsupported.
This is the anti-hallucination core — in healthcare an unsupported answer must become a refusal.
"""
from __future__ import annotations

from .llm import chat
from .retrieve import retrieve

SYSTEM = (
    "You are a clinical knowledge assistant. Answer ONLY using the provided context. "
    "Cite every claim with [n], where n is the numbered context chunk it came from. "
    "If the context does not contain the answer, reply EXACTLY: "
    "'I don't have enough grounded information to answer that.' Never invent facts, drugs, or doses."
)

REFUSAL = "I don't have enough grounded information to answer that."


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {c['source']} § {c['section']})\n{c['text']}"
        for n, c in enumerate(chunks, 1)
    )


def query(question: str, tenant: str | None = None, final_k: int | None = None) -> dict:
    chunks = retrieve(question, tenant=tenant, final_k=final_k)
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "contexts": []}

    user = (
        f"Context:\n{_format_context(chunks)}\n\n"
        f"Question: {question}\n\nGrounded, cited answer:"
    )
    answer = chat(SYSTEM, user)
    return {
        "answer": answer,
        "citations": [
            {"n": n, "source": c["source"], "section": c["section"], "chunk_id": c["chunk_id"]}
            for n, c in enumerate(chunks, 1)
        ],
        "contexts": chunks,
    }
