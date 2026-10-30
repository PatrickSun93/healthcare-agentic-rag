"""Spans are emitted per pipeline stage, nest correctly, and carry the attributes a trace viewer needs."""
import io

import pytest

from src import retrieve as R
from src import tracing


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_retrieve_emits_nested_stage_spans(indexed_sample_docs, spans):
    R.retrieve("What lactate level triggers the sepsis protocol?", final_k=2)
    names = [s.name for s in spans.get_finished_spans()]
    assert names.count("retrieve") == 1
    assert {"retrieve.dense", "retrieve.sparse", "embed"} <= set(names)
    assert "retrieve.rerank" not in names  # RERANK=false

    by = _by_name(spans)
    root = by["retrieve"]
    assert by["retrieve.dense"].parent.span_id == root.context.span_id
    assert by["retrieve.sparse"].parent.span_id == root.context.span_id
    assert by["embed"].parent.span_id == by["retrieve.dense"].context.span_id  # the query embedding

    assert root.attributes["rag.final_k"] == 2
    assert root.attributes["rag.n_results"] == 2
    assert root.attributes["rag.rerank"] is False
    assert "rag.tenant" not in root.attributes  # None is dropped, not stringified
    assert by["retrieve.dense"].attributes["rag.backend"]
    assert by["retrieve.dense"].attributes["rag.n_hits"] > 0
    assert by["embed"].attributes["rag.model"].startswith("hash/")


def test_tenant_shows_up_as_attribute(indexed_sample_docs, spans):
    R.retrieve("metformin", tenant="payerA", final_k=3)
    assert _by_name(spans)["retrieve"].attributes["rag.tenant"] == "payerA"


def test_clean_namespaces_keys_and_drops_none():
    assert tracing._clean({"a": 1, "b": None, "c": [1, 2], "x.y": "z", "flag": True}) == {
        "rag.a": 1,
        "rag.c": "[1, 2]",
        "x.y": "z",
        "rag.flag": True,
    }


def test_console_provider_prints_spans():
    provider = tracing._build_provider("console")
    (proc,) = provider._active_span_processor._span_processors
    buf = io.StringIO()
    proc.span_exporter.out = buf  # ConsoleSpanExporter binds sys.stdout at import time, so redirect explicitly
    with provider.get_tracer("t").start_as_current_span("probe-span"):
        pass
    assert '"name": "probe-span"' in buf.getvalue()


def test_otlp_provider_builds_from_env(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    provider = tracing._build_provider("otlp")
    exporters = [type(p.span_exporter).__name__ for p in provider._active_span_processor._span_processors]
    assert exporters == ["OTLPSpanExporter"]
    provider.shutdown()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        tracing._build_provider("datadog")


def test_span_is_noop_safe_when_sdk_missing(monkeypatch):
    monkeypatch.setattr(tracing, "_tracer", lambda: None)
    with tracing.span("anything", k=1):
        tracing.set_attrs(x=1)  # must not raise even though no span is recording
