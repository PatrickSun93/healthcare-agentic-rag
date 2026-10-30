"""OpenTelemetry spans around every pipeline stage: route, retrieve (dense / sparse / rerank), generate,
grade, rewrite, plus each embed and chat call underneath them.

Off by default and dependency-free: with TRACING=off, or without the OpenTelemetry SDK installed,
span() is a no-op context manager. Exporters are picked by env, no vendor SDK involved:

    TRACING=console   spans printed to stdout            (pip install opentelemetry-sdk)
    TRACING=otlp      OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4318) —
                      any collector, Jaeger, Tempo, Grafana ... (pip install opentelemetry-exporter-otlp-proto-http)

OTEL_SERVICE_NAME names the service (default healthcare-agentic-rag). Attribute keys are namespaced
`rag.*` so they don't collide with semantic conventions.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from . import config as C

_TRACER_NAME = "healthcare_rag"
_configured = False


def _build_provider(mode: str):
    """A TracerProvider wired for `mode` (console | otlp). Kept separate from install so it's testable."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": C.OTEL_SERVICE_NAME}))
    if mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # endpoint/headers come from OTEL_* env
    else:
        raise ValueError(f"Unknown TRACING={mode!r}; use off | console | otlp")
    return provider


def _tracer():
    """The global tracer, installing an exporter on first use if TRACING asks for one. None = SDK absent."""
    global _configured
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    if not _configured:
        _configured = True
        if C.TRACING != "off":
            trace.set_tracer_provider(_build_provider(C.TRACING))
    return trace.get_tracer(_TRACER_NAME)


def _clean(attrs: dict) -> dict:
    """OTel attributes must be str/bool/int/float (or lists of them); drop None, stringify the rest."""
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        key = k if "." in k else f"rag.{k}"
        out[key] = v if isinstance(v, (str, bool, int, float)) else str(v)
    return out


@contextmanager
def span(name: str, **attrs) -> Iterator[None]:
    tracer = _tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name, attributes=_clean(attrs)):
        yield


def set_attrs(**attrs) -> None:
    """Attach late-known values (result counts, decisions) to the innermost open span."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    trace.get_current_span().set_attributes(_clean(attrs))


def traced(name: str) -> Callable:
    """Decorator form of span() for graph nodes and other whole-function stages."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco
