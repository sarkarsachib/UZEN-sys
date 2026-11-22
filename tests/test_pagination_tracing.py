"""P10: cursor pagination and OpenTelemetry-compatible tracing."""

from __future__ import annotations

import json

import pytest

from uzen.infrastructure.observability.tracing import (
    TRACER,
    JsonlSpanExporter,
    Tracer,
    format_traceparent,
    parse_traceparent,
)
from uzen.infrastructure.pagination import decode_cursor, encode_cursor
from uzen.jobs.service import JobService

fastapi_testclient = pytest.importorskip("fastapi.testclient")


def _seed_jobs(service: JobService, count: int) -> list[str]:
    return [
        service.db.create_job({"task_scope": "inspection", "objective": f"job {i}"}).id
        for i in range(count)
    ]


def test_job_pagination_walks_every_page_exactly_once(service: JobService) -> None:
    created = set(_seed_jobs(service, 7))
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = service.list_page(limit=3, cursor=cursor)
        assert len(page["jobs"]) <= 3
        seen.extend(row["id"] for row in page["jobs"])
        pages += 1
        cursor = page["next_cursor"]
        if not page["has_more"]:
            assert cursor is None
            break
        assert cursor
    assert pages == 3
    assert len(seen) == len(set(seen)) == 7
    assert set(seen) == created


def test_cursor_is_stable_when_new_rows_arrive(service: JobService) -> None:
    _seed_jobs(service, 4)
    first = service.list_page(limit=2)
    service.db.create_job({"task_scope": "inspection", "objective": "later"})
    second = service.list_page(limit=2, cursor=first["next_cursor"])
    first_ids = {row["id"] for row in first["jobs"]}
    assert first_ids.isdisjoint({row["id"] for row in second["jobs"]})


def test_cursor_rejects_tampering() -> None:
    cursor = encode_cursor({"created_at": "2026-01-01T00:00:00Z", "id": "abc"})
    assert decode_cursor(cursor) == {"created_at": "2026-01-01T00:00:00Z", "id": "abc"}
    with pytest.raises(Exception):
        decode_cursor("not-a-cursor")
    with pytest.raises(Exception):
        decode_cursor(encode_cursor({"created_at": "x"}))


def test_api_pagination_and_bad_cursor(service: JobService) -> None:
    from uzen.interfaces.api.app import create_app

    _seed_jobs(service, 5)
    client = fastapi_testclient.TestClient(create_app(service=service, auto_run=False))
    page = client.get("/v1/jobs", params={"limit": 2}).json()
    assert len(page["jobs"]) == 2 and page["has_more"] is True
    assert client.get("/v1/jobs", params={"cursor": page["next_cursor"], "limit": 2}).status_code == 200
    assert client.get("/v1/jobs", params={"cursor": "@@@"}).status_code == 400
    assert client.get("/v1/jobs", params={"limit": 0}).status_code == 400


def test_artifact_and_model_pagination(service: JobService) -> None:
    for index in range(5):
        service.db.index_artifact(
            artifact_id=f"art_{index}",
            path=f"/tmp/art_{index}",
            name=f"a{index}",
            project_id="default",
        )
        service.db.register_model(artifact_id=f"art_{index}", family="mlp", spec={"i": index})
    page = service.artifacts_page(limit=2)
    assert len(page["artifacts"]) == 2 and page["next_cursor"]
    rest = service.artifacts_page(limit=10, cursor=page["next_cursor"])
    assert len(page["artifacts"]) + len(rest["artifacts"]) == 5
    models = service.models_page(limit=2)
    assert len(models["models"]) == 2 and models["has_more"] is True


def test_traceparent_parsing_and_generation() -> None:
    valid = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
    context = parse_traceparent(valid)
    assert context is not None
    assert context.trace_id == "a" * 32 and context.span_id == "b" * 16 and context.sampled
    assert format_traceparent(context.trace_id, context.span_id) == valid
    for bad in (
        None,
        "00-" + "0" * 32 + "-" + "b" * 16 + "-01",
        "00-" + "a" * 32 + "-" + "0" * 16 + "-01",
        "ff-" + "a" * 32 + "-" + "b" * 16 + "-01",
        "00-" + "a" * 32 + "-" + "b" * 16,
        "garbage",
    ):
        assert parse_traceparent(bad) is None


def test_http_request_continues_an_incoming_trace(service: JobService) -> None:
    from uzen.interfaces.api.app import create_app

    TRACER.clear()
    client = fastapi_testclient.TestClient(create_app(service=service, auto_run=False))
    trace_id = "c" * 32
    response = client.get("/v1/stats", headers={"traceparent": f"00-{trace_id}-{'d' * 16}-01"})
    assert response.status_code == 200
    emitted = parse_traceparent(response.headers.get("traceparent"))
    assert emitted is not None and emitted.trace_id == trace_id
    spans = client.get("/v1/traces").json()["spans"]
    server = [span for span in spans if span["name"].startswith("GET /v1/stats")]
    assert server and server[0]["trace_id"] == trace_id
    assert server[0]["parent_span_id"] == "d" * 16
    assert server[0]["attributes"]["http.response.status_code"] == 200


def test_invalid_or_absent_traceparent_starts_a_new_trace(service: JobService) -> None:
    from uzen.interfaces.api.app import create_app

    TRACER.clear()
    client = fastapi_testclient.TestClient(create_app(service=service, auto_run=False))
    client.get("/v1/health", headers={"traceparent": "broken"})
    spans = TRACER.recent()
    assert spans and all(len(span["trace_id"]) == 32 for span in spans)


def test_nested_spans_carry_parent_and_otlp_shape() -> None:
    tracer = Tracer()
    with tracer.span("outer", attributes={"k": 1}) as outer:
        with tracer.span("inner", attributes={"ok": True}) as inner:
            assert inner.parent_span_id == outer.span_id
            assert inner.trace_id == outer.trace_id
    payload = tracer.to_otlp()
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert {span["name"] for span in spans} == {"outer", "inner"}
    by_name = {span["name"]: span for span in spans}
    assert by_name["inner"]["parentSpanId"] == by_name["outer"]["spanId"]
    assert by_name["outer"]["attributes"] == [{"key": "k", "value": {"intValue": "1"}}]
    assert all(len(span["traceId"]) == 32 and len(span["spanId"]) == 16 for span in spans)


def test_error_span_records_exception() -> None:
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        with tracer.span("boom"):
            raise RuntimeError("exploded")
    span = tracer.recent()[0]
    assert span["status"] == "ERROR"
    assert span["error"] == "RuntimeError: exploded"
    assert any(event["name"] == "exception" for event in span["events"])


def test_jsonl_exporter_writes_otlp_spans(tmp_path) -> None:
    tracer = Tracer()
    exporter = JsonlSpanExporter(tmp_path / "traces" / "spans.jsonl")
    tracer.add_exporter(exporter)
    with tracer.span("exported", attributes={"service": "test"}):
        pass
    lines = (tmp_path / "traces" / "spans.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "exported"
    assert record["traceId"] and record["spanId"]
    assert record["attributes"][0]["key"] == "service"


def test_job_execution_is_traced(service: JobService) -> None:
    from uzen.core.nws.families import build_architecture

    definition = build_architecture("mlp", dict(input_dim=6, hidden_dims=[8], output_dim=3))
    package = definition.build_package(seed=9)
    artifact = service.store.put(package, operation="init").artifact_id
    TRACER.clear()
    job = service.submit(
        {
            "task_scope": "weight_modification",
            "inputs": [{"name": "base", "artifact_id": artifact}],
            "operations": [{"op": "w.arithmetic.scale", "params": {"factor": 0.5}}],
        }
    )
    finished = service.run(job["id"])
    assert finished["state"] == "SUCCEEDED"
    spans = [span for span in TRACER.recent() if span["name"] == "uzen.job.execute"]
    assert spans, TRACER.recent()
    assert spans[0]["attributes"]["uzen.job.id"] == job["id"]
    assert spans[0]["attributes"]["uzen.job.outcome"] == "completed"
    assert spans[0]["duration_ms"] is not None
