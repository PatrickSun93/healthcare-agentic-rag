"""Provider abstraction for embeddings + chat. Default = local Ollama (data stays on-box).

Swap providers purely via .env (LLM_PROVIDER / EMBED_PROVIDER) — the rest of the code is unaware.
EMBED_PROVIDER=hash is a model-free stand-in so the whole pipeline (ingest -> store -> retrieve -> eval)
runs in tests/CI without Ollama or network.
"""
from __future__ import annotations

import hashlib
import math
import re

import httpx

from . import config as C


# ---------------- Embeddings ----------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    if C.EMBED_PROVIDER == "hash":
        return [hash_embed(t) for t in texts]

    if C.EMBED_PROVIDER == "ollama":
        out: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for t in texts:
                r = client.post(
                    f"{C.OLLAMA_BASE}/api/embeddings",
                    json={"model": C.EMBED_MODEL, "prompt": t},
                )
                r.raise_for_status()
                out.append(r.json()["embedding"])
        return out

    if C.EMBED_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=C.OPENAI_API_KEY)
        resp = client.embeddings.create(model=C.OPENAI_EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    raise ValueError(f"Unknown EMBED_PROVIDER: {C.EMBED_PROVIDER}")


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def hash_embed(text: str, dim: int | None = None) -> list[float]:
    """Deterministic feature-hashing embedding: each token hashes to one of `dim` buckets with a ±1 sign,
    then the vector is L2-normalised. Texts that share tokens land near each other, so nearest-neighbour
    search behaves sensibly — enough to exercise stores, fusion and the eval harness. It is NOT semantic:
    never report retrieval quality measured with it."""
    dim = dim or C.HASH_EMBED_DIM
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        vec[int.from_bytes(h[:4], "little") % dim] += 1.0 if h[4] & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def describe() -> dict[str, str]:
    """'provider/model' ids for the active LLM + embedder — recorded in eval results so a metric change
    can be traced to a model swap."""
    llm = {
        "ollama": C.LLM_MODEL,
        "openai": C.OPENAI_LLM_MODEL,
        "anthropic": C.ANTHROPIC_LLM_MODEL,
    }.get(C.LLM_PROVIDER, "?")
    emb = {
        "ollama": C.EMBED_MODEL,
        "openai": C.OPENAI_EMBED_MODEL,
        "hash": f"feature-hash-{C.HASH_EMBED_DIM}d",
    }.get(C.EMBED_PROVIDER, "?")
    return {"llm": f"{C.LLM_PROVIDER}/{llm}", "embedding": f"{C.EMBED_PROVIDER}/{emb}"}


# ---------------- Chat ----------------
def chat(system: str, user: str) -> str:
    if C.LLM_PROVIDER == "ollama":
        with httpx.Client(timeout=300) as client:
            r = client.post(
                f"{C.OLLAMA_BASE}/api/chat",
                json={
                    "model": C.LLM_MODEL,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    if C.LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=C.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=C.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    if C.LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=C.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=C.ANTHROPIC_LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    raise ValueError(f"Unknown LLM_PROVIDER: {C.LLM_PROVIDER}")
