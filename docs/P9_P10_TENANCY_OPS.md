# P9 — Tenancy and authorization; P10 — Pagination and tracing

These two phases close the operational gaps in the STRICT_PLUS audit: the control
plane was single-tenant and unpaged, and had no trace propagation.

## P9: projects, scoped tokens, artifact ACLs

* **Projects own work.** `jobs`, `artifacts_index` and `models` carry a
  `project_id` (default `default`, added in place to existing databases by
  `Database._migrate`). `JobService.submit` stamps the authenticated project on the
  job; artifact/model rows inherit it from the creating job or the ambient
  principal.
* **Tokens are stored hashed.** `api_tokens` keeps the SHA-256 of a minted token,
  its project and its scopes. `POST /v1/tokens` returns the plaintext exactly once;
  `DELETE /v1/tokens/{id}` revokes it. A project token can only mint tokens inside
  its own project and only with scopes it already holds; only an admin can cross
  projects or grant `admin`.
* **Scopes are enforced per route.** `uzen/infrastructure/security/access.py` holds
  one ordered table mapping `(method, path)` to a scope
  (`jobs:read`, `jobs:submit`, `jobs:cancel`, `weights:read`, `weights:write`,
  `artifacts:read`, `artifacts:write`, `artifacts:share`, `tokens:admin`,
  `admin`). Unknown routes default to requiring `jobs:read` so a new endpoint is
  never silently public.
* **Ownership is enforced before the handler.** A request naming another project's
  job or artifact is a `403 auth.denied` from the middleware, before the route can
  return a body; a missing id still produces the route's normal 404, so ids from
  other tenants are not distinguishable from absent ones.
* **Artifact sharing is explicit.** `artifact_acl` grants another project `read` or
  `write` on one artifact (`POST /v1/artifacts/{id}/share`,
  `GET /v1/artifacts/{id}/acl`). A read grant never authorizes a write.
* **Mandatory mode.** `UZEN_API_TOKEN` (bootstrap admin) or `UZEN_REQUIRE_AUTH=1`
  makes bearer authentication mandatory; `GET /health`, `GET /metrics`, the
  operation catalogue, the policy document and the worker plane (which
  authenticates with its own worker secret) stay reachable. Without either, the
  process is a documented single-tenant deployment where the caller is an anonymous
  admin of `default`.
* **Admin project switching.** The bootstrap admin may set `X-UZEN-Project` to
  attribute work to a named project; project tokens cannot.

Tests: `tests/test_tenancy.py` (mandatory auth, cross-project 403s, scope denial,
token escalation refusal, revocation, artifact ACL read/write separation, ambient
principal attribution, admin project header).

## P10: cursor pagination and OpenTelemetry-compatible tracing

* **Keyset pagination.** Lists are ordered by `(created_at DESC, id DESC)` and paged
  with an opaque base64url cursor carrying that sort key, so pages stay stable while
  jobs finish and no rows are skipped or repeated. `GET /v1/jobs`, `/v1/models` and
  `/v1/weights` return `{items…, next_cursor, has_more, limit}`; `limit` is clamped
  to 500 and a malformed cursor or non-positive limit is a `400
  pagination.invalid_cursor`.
* **W3C trace context.** `parse_traceparent` validates version, non-zero 32-hex trace
  id, non-zero 16-hex span id and flags; an invalid header starts a new trace rather
  than failing the request. Every HTTP response carries the server span's
  `traceparent`, so a caller can continue the trace.
* **Spans.** `Tracer` records a bounded ring of spans (default 2000) and supports
  pluggable exporters. `JsonlSpanExporter` writes one OTLP/JSON span per line.
  `GET /v1/traces` returns recent spans, `GET /v1/traces/otlp` the
  `ExportTraceServiceRequest` document.
* **Job spans.** `JobService._execute` is wrapped in a `uzen.job.execute` span
  carrying the job id, task scope, kind and operation count, parented to the request
  span when a job is executed inside an HTTP request.

Tests: `tests/test_pagination_tracing.py` (page walking, stability under insertion,
cursor tampering, API paging, artifact/model paging, traceparent parse/format,
trace continuation and response header, parent/child spans, OTLP shape, error spans,
JSONL exporter, job execution span).

## Limits

* Keyset pagination is descending-only (`created_at DESC`); an ascending or
  arbitrary-sort cursor is not implemented.
* Spans are process-local: the ring buffer is for inspection, and durable trace
  storage is the collector's job via the exporter.
* There is no per-token rate limiting; scopes and expiry/revocation are the only
  token controls.
