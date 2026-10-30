"""Versioned prompt templates.

Every system prompt lives in this folder as `<name>.<version>.txt`; PINNED says which version the
pipeline runs. To change a prompt, add a new file (grounded_answer.v2.txt) and move the pin — never edit
a shipped version in place. The eval harness records the pins in every results file, so a metric change
can be traced to a prompt change. Override pins without code changes for an A/B run:

    PROMPT_VERSIONS="grounded_answer=v2,router=v1" python cli.py eval
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent

PINNED: dict[str, str] = {
    "grounded_answer": "v1",  # rag.py / graph.py generate node
    "router": "v1",           # agent.py: retrieve or not + query rewrite
    "grader": "v1",           # graph.py self-RAG: is the answer grounded?
    "query_rewrite": "v1",    # graph.py self-RAG: broaden the query on retry
    "chitchat": "v1",         # no-retrieval branch
}


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def _overrides() -> dict[str, str]:
    raw = os.getenv("PROMPT_VERSIONS", "")
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {k.strip(): v.strip() for k, v in pairs}


def versions() -> dict[str, str]:
    """Effective pin per prompt (PINNED + PROMPT_VERSIONS overrides) — what the eval record stores."""
    return {**PINNED, **{k: v for k, v in _overrides().items() if k in PINNED}}


def available(name: str) -> list[str]:
    return sorted(p.suffixes[0].lstrip(".") for p in PROMPTS_DIR.glob(f"{name}.v*.txt"))


@cache
def _load(name: str, version: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}.{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt {name}@{version} not found; have {available(name)}")
    return Prompt(name, version, path.read_text().strip())


def get(name: str, version: str | None = None) -> Prompt:
    if name not in PINNED:
        raise KeyError(f"unknown prompt {name!r}; known: {sorted(PINNED)}")
    return _load(name, version or versions()[name])
