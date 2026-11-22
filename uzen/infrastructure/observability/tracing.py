"""OpenTelemetry-compatible request and job tracing.

This is a dependency-free tracer that speaks the W3C Trace Context wire format and
exports spans in the OpenTelemetry (OTLP/JSON) shape, so a deployment can forward
them to a collector without UZEN shipping an SDK:

* :func:`parse_traceparent` validates an incoming ``traceparent`` header (version,
  non-zero 32-hex trace id, non-zero 16-hex span id, flags) and is used to continue
  a caller's trace instead of starting a new one.
* :class:`Tracer` keeps a bounded in-process ring of finished spans (served by
  ``GET /v1/traces``) and hands every span to configured exporters.
* :class:`JsonlSpanExporter` writes one OTLP span object per line, which is exactly
  what a file-log collector or ``otelcol`` file receiver consumes.

The tracer never invents parent ids: a span is a child only when a valid remote
context or an active local span supplies one.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_FLAGS_RE = re.compile(r"^[0-9a-f]{2}$")
_VERSION_RE = re.compile(r"^[0-9a-f]{2}$")
_ZERO_TRACE = "0" * 32
_ZERO_SPAN = "0" * 16

SPAN_KIND_INTERNAL = 1
SPAN_KIND_SERVER = 2
SPAN_KIND_CLIENT = 3


@dataclass(frozen=True)
class TraceContext:
    """A validated remote or local trace context."""

    trace_id: str
    span_id: str
    flags: int = 1

    @property
    def sampled(self) -> bool:
        return bool(self.flags & 0x01)


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Parse a W3C ``traceparent`` value; ``None`` when it is absent or invalid.

    An invalid header is treated as absent (a new trace is started) rather than
    rejected: trace propagation must never break a request.
    """
    if not isinstance(value, str):
        return None
    parts = value.strip().split("-")
    if len(parts) != 4:
        return None
    version, trace_id, span_id, flags = parts
    if not _VERSION_RE.match(version) or version == "ff":
        return None
    if not _TRACE_ID_RE.match(trace_id) or trace_id == _ZERO_TRACE:
        return None
    if not _SPAN_ID_RE.match(span_id) or span_id == _ZERO_SPAN:
        return None
    if not _FLAGS_RE.match(flags):
        return None
    return TraceContext(trace_id, span_id, int(flags, 16))


def format_traceparent(trace_id: str, span_id: str, *, sampled: bool = True) -> str:
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    value = secrets.token_hex(8)
    return value if value != _ZERO_SPAN else "0" * 15 + "1"


def _attribute_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": float(value)}
    return {"stringValue": str(value)}


@dataclass
class Span:
    """One finished-or-in-flight span."""

    trace_id: str
    span_id: str
    name: str
    kind: int = SPAN_KIND_INTERNAL
    parent_span_id: str | None = None
    start_time_ns: int = field(default_factory=lambda: time.time_ns())
    end_time_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "UNSET"
    error: str | None = None
    _tracer: Any = field(default=None, repr=False, compare=False)

    # -- mutation ------------------------------------------------------- #

    def set_attribute(self, key: str, value: Any) -> "Span":
        self.attributes[str(key)] = value
        return self

    def add_event(self, name: str, **attributes: Any) -> "Span":
        self.events.append(
            {
                "name": str(name),
                "time_unix_nano": time.time_ns(),
                "attributes": dict(attributes),
            }
        )
        return self

    def record_exception(self, exc: BaseException) -> "Span":
        self.status = "ERROR"
        self.error = f"{type(exc).__name__}: {exc}"
        self.add_event("exception", type=type(exc).__name__, message=str(exc))
        return self

    def end(self, *, status: str | None = None) -> "Span":
        if self.end_time_ns is None:
            self.end_time_ns = time.time_ns()
        if status is not None:
            self.status = status
        if self._tracer is not None:
            self._tracer._finish(self)
        return self

    # -- serialization -------------------------------------------------- #

    @property
    def duration_ms(self) -> float | None:
        if self.end_time_ns is None:
            return None
        return round((self.end_time_ns - self.start_time_ns) / 1e6, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
            "events": [dict(event) for event in self.events],
            "status": self.status,
            "error": self.error,
        }

    def to_otlp(self, *, service_name: str = "uzen") -> dict[str, Any]:
        """The span as an OTLP/JSON ``Span`` message."""
        end = self.end_time_ns if self.end_time_ns is not None else time.time_ns()
        span: dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": str(self.start_time_ns),
            "endTimeUnixNano": str(end),
            "attributes": [
                {"key": str(key), "value": _attribute_value(value)}
                for key, value in sorted(self.attributes.items())
            ],
            "events": [
                {
                    "name": event["name"],
                    "timeUnixNano": str(event["time_unix_nano"]),
                    "attributes": [
                        {"key": str(key), "value": _attribute_value(value)}
                        for key, value in sorted(event.get("attributes", {}).items())
                    ],
                }
                for event in self.events
            ],
            "status": {
                "code": {"ERROR": 2, "OK": 1}.get(self.status, 0),
                **({"message": self.error} if self.error else {}),
            },
        }
        if self.parent_span_id:
            span["parentSpanId"] = self.parent_span_id
        return span


_CURRENT_SPAN: ContextVar[Span | None] = ContextVar("uzen_current_span", default=None)


class Tracer:
    """Bounded span recorder with pluggable exporters."""

    def __init__(self, *, max_spans: int = 2000, service_name: str | None = None) -> None:
        if int(max_spans) <= 0:
            raise ValueError("max_spans must be positive")
        self.service_name = service_name or os.environ.get("UZEN_SERVICE_NAME", "uzen")
        self._spans: deque[Span] = deque(maxlen=int(max_spans))
        self._lock = threading.RLock()
        self._exporters: list[Any] = []
        self._recorded = 0

    # -- exporters ------------------------------------------------------ #

    def add_exporter(self, exporter: Any) -> None:
        """Register any object with ``export(span)``."""
        if not hasattr(exporter, "export"):
            raise TypeError("exporter must provide export(span)")
        with self._lock:
            self._exporters.append(exporter)

    def clear_exporters(self) -> None:
        with self._lock:
            self._exporters.clear()

    # -- spans ---------------------------------------------------------- #

    def start_span(
        self,
        name: str,
        *,
        kind: int = SPAN_KIND_INTERNAL,
        parent: TraceContext | None = None,
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Span:
        parent_span_id = parent_span_id or (parent.span_id if parent else None)
        if parent is not None:
            resolved_trace = parent.trace_id
        else:
            resolved_trace = trace_id or _new_trace_id()
        span = Span(
            trace_id=resolved_trace,
            span_id=_new_span_id(),
            name=str(name),
            kind=kind,
            parent_span_id=parent_span_id,
            attributes={str(k): v for k, v in dict(attributes or {}).items()},
            _tracer=self,
        )
        return span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: int = SPAN_KIND_INTERNAL,
        parent: TraceContext | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Iterator[Span]:
        """Start a span, making it current for nested spans in this context."""
        current = _CURRENT_SPAN.get()
        span = self.start_span(
            name,
            kind=kind,
            parent=parent,
            parent_span_id=current.span_id if current is not None else None,
            attributes=attributes,
            trace_id=current.trace_id if current is not None else None,
        )
        token = _CURRENT_SPAN.set(span)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
        finally:
            _CURRENT_SPAN.reset(token)
            span.end()

    def _finish(self, span: Span) -> None:
        with self._lock:
            self._spans.append(span)
            self._recorded += 1
            exporters = list(self._exporters)
        for exporter in exporters:
            try:
                exporter.export(span)
            except Exception:  # pragma: no cover - an exporter must not break a trace
                continue

    # -- inspection ----------------------------------------------------- #

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            spans = list(self._spans)[-max(1, int(limit)) :]
        return [span.to_dict() for span in reversed(spans)]

    def to_otlp(self) -> dict[str, Any]:
        """All retained spans as an OTLP/JSON ``ExportTraceServiceRequest``."""
        with self._lock:
            spans = list(self._spans)
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "uzen.tracing", "version": "1.0.0"},
                            "spans": [span.to_otlp(service_name=self.service_name) for span in spans],
                        }
                    ],
                }
            ]
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recorded": self._recorded,
                "retained": len(self._spans),
                "capacity": self._spans.maxlen,
                "exporters": len(self._exporters),
                "service_name": self.service_name,
            }

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()
            self._recorded = 0


def current_span() -> Span | None:
    return _CURRENT_SPAN.get()


def current_trace_context() -> TraceContext | None:
    span = _CURRENT_SPAN.get()
    if span is None:
        return None
    return TraceContext(span.trace_id, span.span_id, 1)


class JsonlSpanExporter:
    """Append one OTLP span object per line to a file (a real file exporter)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, span: Span) -> None:
        line = json.dumps(span.to_otlp(), separators=(",", ":"), sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


TRACER = Tracer()
