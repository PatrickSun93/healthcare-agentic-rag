"""LangGraph paths (route -> retrieve -> generate, chitchat, the self-RAG cycle) with a scripted LLM.
The graph, the retriever and the stores are real; only `chat` is faked, keyed by which versioned system
prompt it receives — so a prompt swap that breaks a node's contract shows up here."""
from __future__ import annotations

import json

import pytest

from src import agent, graph, rag


class ScriptedChat:
    def __init__(self, route: dict | None = None, route_raw: str | None = None, grades=("yes",)):
        self.route = route or {"retrieve": True, "search_query": "metformin prior authorization criteria"}
        self.route_raw = route_raw
        self.grades = list(grades)
        self.systems: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.systems.append(system)
        if system == agent.ROUTER_SYSTEM:
            return self.route_raw if self.route_raw is not None else json.dumps(self.route)
        if system == rag.SYSTEM:
            assert "Context:" in user and "Question:" in user
            return "Metformin ER needs eGFR >= 45 [1]."
        if system == graph.GRADER_SYSTEM:
            return self.grades.pop(0) if len(self.grades) > 1 else self.grades[0]
        if system == graph.REWRITE_SYSTEM:
            return '"metformin coverage criteria"'
        if system == agent.CHITCHAT_SYSTEM:
            return "Hello!"
        raise AssertionError(f"unexpected system prompt: {system[:50]!r}")


@pytest.fixture
def scripted(monkeypatch):
    def install(**kw) -> ScriptedChat:
        fake = ScriptedChat(**kw)
        for mod in (graph, agent, rag):
            monkeypatch.setattr(mod, "chat", fake)
        return fake

    return install


def _names(exporter):
    return [s.name for s in exporter.get_finished_spans()]


def test_graph_routes_chitchat_without_retrieval(indexed_sample_docs, scripted, spans):
    scripted(route={"retrieve": False, "search_query": ""})
    r = graph.answer("hi there")
    assert (r["answer"], r["used_retrieval"], r["citations"]) == ("Hello!", False, [])
    names = _names(spans)
    assert "graph.chitchat" in names and "graph.retrieve" not in names and "retrieve" not in names


def test_graph_retrieves_generates_and_cites(indexed_sample_docs, scripted, spans):
    fake = scripted()
    r = graph.answer("What PA criteria apply to metformin ER?")
    assert r["used_retrieval"] is True
    assert r["answer"].endswith("[1].")
    assert r["citations"][0]["source"] == "metformin_prior_auth.md"
    assert {"n", "source", "section", "chunk_id"} <= set(r["citations"][0])
    assert fake.systems == [agent.ROUTER_SYSTEM, rag.SYSTEM]

    by = {s.name: s for s in spans.get_finished_spans()}
    root = by["graph.invoke"]
    assert root.attributes["rag.graph"] == "rag"
    for node in ("graph.route", "graph.retrieve", "graph.generate"):
        assert by[node].parent.span_id == root.context.span_id
    assert by["retrieve"].parent.span_id == by["graph.retrieve"].context.span_id
    assert by["agent.route"].attributes["rag.retrieve"] is True
    assert by["graph.generate"].attributes["rag.prompt"] == "grounded_answer@v1"
    assert by["graph.generate"].attributes["rag.n_contexts"] > 0
    assert "llm.chat" not in by  # chat is scripted here; its span is covered by the real llm.chat wrapper
    assert by["embed"].attributes["rag.model"].startswith("hash/")


def test_selfrag_retries_once_when_grader_says_no(indexed_sample_docs, scripted, spans):
    fake = scripted(grades=("no", "yes"))
    r = graph.answer_selfrag("What PA criteria apply to metformin ER?")
    assert r["attempts"] == 2 and r["grounded"] is True
    assert r["search_query"] == "metformin coverage criteria"  # the rewrite fed the second retrieval
    assert fake.systems.count(graph.REWRITE_SYSTEM) == 1
    names = _names(spans)
    assert names.count("graph.grade") == 2 and names.count("graph.rewrite") == 1 and names.count("retrieve") == 2


def test_selfrag_gives_up_at_max_attempts(indexed_sample_docs, scripted):
    scripted(grades=("no",))
    r = graph.answer_selfrag("What PA criteria apply to metformin ER?")
    assert r["attempts"] == graph.MAX_ATTEMPTS and r["grounded"] is False


def test_router_falls_back_to_retrieval_on_bad_json(scripted, spans):
    scripted(route_raw="sure, let me think about that")
    assert agent._route("anything") == {"retrieve": True, "search_query": "anything"}
    attrs = {s.name: s.attributes for s in spans.get_finished_spans()}["agent.route"]
    assert attrs["rag.parse_failed"] is True and attrs["rag.retrieve"] is True


def test_rag_query_refuses_when_tenant_has_no_data(indexed_sample_docs, scripted, spans):
    fake = scripted()
    r = rag.query("metformin", tenant="nobody")
    assert r["answer"] == rag.REFUSAL and r["citations"] == []
    assert fake.systems == []  # refusal never reaches the LLM
    attrs = {s.name: s.attributes for s in spans.get_finished_spans()}["rag.query"]
    assert attrs["rag.refused"] is True and attrs["rag.n_contexts"] == 0
