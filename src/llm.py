"""Provider abstraction for embeddings + chat. Default = local Ollama (data stays on-box).

Swap providers purely via .env (LLM_PROVIDER / EMBED_PROVIDER) — the rest of the code is unaware.
"""
from __future__ import annotations

import httpx

from . import config as C


# ---------------- Embeddings ----------------
def embed_texts(texts: list[str]) -> list[list[float]]:
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
