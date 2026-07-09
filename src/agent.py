"""Minimal agentic layer: 'retrieval as a tool'.

The router LLM decides whether a question needs the knowledge base and rewrites it into a good
search query, then calls the RAG tool. This mirrors how a retrieval tool plugs into a larger
agent loop (LangGraph, or a NodeOrchestrator-style runtime) with human-in-the-loop gates around it.
"""
from __future__ import annotations

import json

from . import rag
from .llm import chat

ROUTER_SYSTEM = (
    "You are a router for a clinical assistant. Decide if the user's message needs retrieval from a "
    "clinical knowledge base. Reply with ONLY a compact JSON object: "
    '{"retrieve": true|false, "search_query": "..."}. '
    "Use retrieve=false for greetings/chit-chat; otherwise retrieve=true with an optimized search query."
)


def _route(question: str) -> dict:
    raw = chat(ROUTER_SYSTEM, question)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1])
    except Exception:
        return {"retrieve": True, "search_query": question}  # fail safe: retrieve


def answer(question: str, tenant: str | None = None) -> dict:
    decision = _route(question)
    if not decision.get("retrieve", True):
        return {
            "answer": chat("You are a helpful assistant.", question),
            "used_retrieval": False,
            "citations": [],
        }
    search_query = decision.get("search_query") or question
    result = rag.query(search_query, tenant=tenant)
    result["used_retrieval"] = True
    result["search_query"] = search_query
    return result


# Expose RAG as a callable tool spec (drop into a LangGraph/MCP tool registry or an agent loop).
RAG_TOOL = {
    "name": "clinical_kb_search",
    "description": "Search the clinical knowledge base and return a grounded, cited answer.",
    "func": lambda question, tenant=None: rag.query(question, tenant=tenant),
}
