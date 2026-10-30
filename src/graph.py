"""LangGraph agent graph: route -> (retrieve -> generate) | chitchat.

Same behavior as agent.py, but expressed as an explicit LangGraph StateGraph — a stateful workflow
with a conditional edge (retrieve-or-not) and clean nodes for routing, retrieval, and grounded
generation. This is the LangGraph-native form of "retrieval as a tool + a router that decides",
and the natural place to later add a human-in-the-loop interrupt before generate.

    python cli.py query "..." --graph
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import agent, prompts, rag
from .llm import chat
from .retrieve import retrieve
from .tracing import set_attrs, span, traced

GRADER_SYSTEM = prompts.get("grader").text
REWRITE_SYSTEM = prompts.get("query_rewrite").text


class RAGState(TypedDict, total=False):
    question: str
    tenant: str | None
    retrieve: bool
    search_query: str
    contexts: list
    answer: str
    citations: list
    used_retrieval: bool
    grounded: bool
    attempts: int


@traced("graph.route")
def _route(state: RAGState) -> dict:
    decision = agent._route(state["question"])
    return {
        "retrieve": decision.get("retrieve", True),
        "search_query": decision.get("search_query") or state["question"],
    }


@traced("graph.retrieve")
def _retrieve(state: RAGState) -> dict:
    return {"contexts": retrieve(state["search_query"], tenant=state.get("tenant"))}


@traced("graph.generate")
def _generate(state: RAGState) -> dict:
    chunks = state.get("contexts") or []
    set_attrs(prompt=rag.GROUNDED.id, n_contexts=len(chunks), refused=not chunks)
    if not chunks:
        return {"answer": rag.REFUSAL, "citations": [], "used_retrieval": True}
    return {
        "answer": chat(rag.SYSTEM, rag.build_user_prompt(state["question"], chunks)),
        "citations": [
            {"n": n, "source": c["source"], "section": c["section"], "chunk_id": c["chunk_id"]}
            for n, c in enumerate(chunks, 1)
        ],
        "used_retrieval": True,
    }


@traced("graph.chitchat")
def _chitchat(state: RAGState) -> dict:
    return {
        "answer": chat(agent.CHITCHAT_SYSTEM, state["question"]),
        "citations": [],
        "used_retrieval": False,
    }


def _decide(state: RAGState) -> str:
    return "retrieve" if state.get("retrieve", True) else "chitchat"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("route", _route)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("chitchat", _chitchat)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", _decide, {"retrieve": "retrieve", "chitchat": "chitchat"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.add_edge("chitchat", END)
    return g.compile()


_GRAPH = None


def answer(question: str, tenant: str | None = None) -> dict:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    with span("graph.invoke", graph="rag", tenant=tenant):
        return _GRAPH.invoke({"question": question, "tenant": tenant})


# ---- self-RAG: a graph with a CYCLE (grade the answer; if ungrounded, broaden the query + retry) ----

MAX_ATTEMPTS = 2


@traced("graph.grade")
def _grade(state: RAGState) -> dict:
    answer = state.get("answer", "")
    if answer.strip() == rag.REFUSAL:
        set_attrs(grounded=False, attempt=state.get("attempts", 1))
        return {"grounded": False}  # nothing found yet — a broader query might help
    verdict = chat(
        GRADER_SYSTEM,
        f"Context:\n{rag._format_context(state.get('contexts') or [])}\n\n"
        f"Answer:\n{answer}\n\nIs EVERY claim in the answer directly supported by the context? yes or no.",
    )
    grounded = verdict.strip().lower().startswith("y")
    set_attrs(grounded=grounded, attempt=state.get("attempts", 1))
    return {"grounded": grounded}


@traced("graph.rewrite")
def _rewrite(state: RAGState) -> dict:
    current = state.get("search_query") or state["question"]
    broader = chat(
        REWRITE_SYSTEM,
        f"Question: {state['question']}\nCurrent query: {current}\nBroader query:",
    ).strip().strip('"')
    set_attrs(attempt=state.get("attempts", 1) + 1)
    return {"search_query": broader or current, "attempts": state.get("attempts", 1) + 1}


def _after_grade(state: RAGState) -> str:
    if state.get("grounded") or state.get("attempts", 1) >= MAX_ATTEMPTS:
        return "end"
    return "retry"


def build_selfrag_graph():
    g = StateGraph(RAGState)
    g.add_node("route", _route)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("grade", _grade)
    g.add_node("rewrite", _rewrite)
    g.add_node("chitchat", _chitchat)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", _decide, {"retrieve": "retrieve", "chitchat": "chitchat"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "grade")
    g.add_conditional_edges("grade", _after_grade, {"end": END, "retry": "rewrite"})
    g.add_edge("rewrite", "retrieve")  # <-- the cycle: retry retrieval with a broader query
    g.add_edge("chitchat", END)
    return g.compile()


_SELFRAG = None


def answer_selfrag(question: str, tenant: str | None = None) -> dict:
    global _SELFRAG
    if _SELFRAG is None:
        _SELFRAG = build_selfrag_graph()
    with span("graph.invoke", graph="selfrag", tenant=tenant):
        return _SELFRAG.invoke({"question": question, "tenant": tenant, "attempts": 1})
