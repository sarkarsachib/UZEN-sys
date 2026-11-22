"""REST control plane for UZEN.

A versioned, OpenAPI-documented HTTP surface over the same engines the SDK and CLI
use. It is strictly a *control plane*: metadata, orchestration and artifact
management. All tensor math happens in the job service / worker execution plane.

Run::

    uzen serve --root ./uzen-workspace --host 0.0.0.0 --port 8080

Authentication: set ``UZEN_API_TOKEN`` to require ``Authorization: Bearer <token>``
on every ``/v1`` route. Worker routes additionally verify the worker's registered
secret hash.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

try:  # FastAPI is an optional extra: the engine never requires it.
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    FastAPI = None  # type: ignore
    FASTAPI_AVAILABLE = False

from uzen.core.nwpc.registry import default_registry
from uzen.core.nws.families import family_catalogue
from uzen.errors import UzenError
from uzen.infrastructure.observability.metrics import METRICS
from uzen.infrastructure.security.tokens import hash_secret
from uzen.infrastructure.storage.db import JOB_STATES
from uzen.jobs.scheduler import JobScheduler
from uzen.jobs.service import JobService
from uzen.policies.strict_plus.engine import default_engine
from uzen.runtimes.pytorch_runtime import PyTorchRuntime, torch_available
from uzen.workers.protocol.messages import detect_capabilities

API_VERSION = "v1"


def create_app(
    *,
    root: str | Path | None = None,
    service: JobService | None = None,
    api_token: str | None = None,
    auto_run: bool = True,
    lease_seconds: float = 120.0,
    max_attempts: int = 3,
    retry_backoff_seconds: float = 0.0,
    retry_backoff_cap_seconds: float = 300.0,
    aging_seconds: float = 0.0,
    max_in_flight_per_worker: int = 0,
) -> Any:
    if not FASTAPI_AVAILABLE:
        from uzen.errors import RuntimeUnavailable

        raise RuntimeUnavailable(
            "FastAPI is not installed; install the API extra with "
            "`pip install 'uzen[api]'` or run the engine via CLI/SDK instead"
        )
    service = service or JobService(root or os.environ.get("UZEN_ROOT", "./uzen-workspace"))
    # A distributed deployment sets auto_run=False: jobs stay QUEUED until a registered
    # worker claims them, instead of being executed inside the API process.
    scheduler = JobScheduler(
        service.db,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        retry_backoff_cap_seconds=retry_backoff_cap_seconds,
        aging_seconds=aging_seconds,
        max_in_flight_per_worker=max_in_flight_per_worker,
    )
    token = api_token if api_token is not None else os.environ.get("UZEN_API_TOKEN")
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="uzen-job")
    lock = threading.Lock()

    app = FastAPI(
        title="UZEN",
        version="0.1.0",
        description=(
            "Universal Neural Weight Engineering & Architecture Research System — "
            "headless control plane. STRICT_PLUS: no training, fine-tuning, "
            "backpropagation or gradient-based parameter updates."
        ),
    )
    app.state.service = service
    app.state.scheduler = scheduler
    app.state.auto_run = auto_run

    @app.middleware("http")
    async def observe_http(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        METRICS.inc(
            "uzen_http_requests_total",
            method=request.method,
            path=path,
            status=str(response.status_code),
        )
        METRICS.observe(
            "uzen_http_request_duration_seconds",
            time.perf_counter() - started,
            method=request.method,
            path=path,
        )
        return response

    # ------------------------------------------------------------------ #
    # auth / errors
    # ------------------------------------------------------------------ #

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail={"code": "auth.missing", "message": "bearer token required"})
        presented = authorization.split(" ", 1)[1].strip()
        if not hmac.compare_digest(hash_secret(presented), hash_secret(token)):
            raise HTTPException(status_code=401, detail={"code": "auth.invalid", "message": "invalid token"})

    @app.exception_handler(UzenError)
    async def uzen_error_handler(request: Request, exc: UzenError) -> JSONResponse:
        detail = exc.to_dict()
        METRICS.inc("uzen_errors_total", code=str(detail.get("code", type(exc).__name__)))
        return JSONResponse(status_code=exc.http_status, content={"detail": detail})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal.error",
                    "message": str(exc),
                    "details": {"type": type(exc).__name__, "path": str(request.url)},
                }
            },
        )

    # ------------------------------------------------------------------ #
    # meta
    # ------------------------------------------------------------------ #

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": "0.1.0", "policy": "STRICT_PLUS"}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        snapshot = service.stats()
        counts = snapshot.get("jobs_by_state", {})
        for state in JOB_STATES:
            METRICS.set("uzen_jobs", counts.get(state, 0), state=state)
        METRICS.set("uzen_artifacts_indexed", snapshot.get("artifacts_indexed", 0))
        METRICS.set("uzen_workers", snapshot.get("workers", 0))
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")

    @app.get(f"/{API_VERSION}/stats")
    def stats(_: None = Depends(require_auth)) -> dict[str, Any]:
        capabilities = detect_capabilities()
        return {
            **service.stats(),
            "engine": capabilities,
            "pytorch_available": torch_available(),
            "strict_plus": default_engine().describe()["statements"],
        }

    @app.get(f"/{API_VERSION}/operations")
    def operations(family: str | None = None, executable_only: bool = False, _: None = Depends(require_auth)):
        registry = default_registry()
        return {
            "operations": registry.catalogue(family=family, executable_only=executable_only),
            "families": registry.families(),
            "count": len(registry.all()),
        }

    @app.get(f"/{API_VERSION}/operations/{{code}}")
    def operation_detail(code: str, _: None = Depends(require_auth)):
        return default_registry().get(code).to_dict()

    @app.get(f"/{API_VERSION}/policies/strict-plus")
    def policy(_: None = Depends(require_auth)):
        return default_engine().describe()

    @app.get(f"/{API_VERSION}/architectures/families")
    def families(_: None = Depends(require_auth)):
        return {"families": family_catalogue()}

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/jobs", status_code=201)
    def submit_job(
        payload: dict[str, Any],
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_auth),
    ):
        job = service.submit(payload, idempotency_key=idempotency_key)
        if auto_run:
            executor.submit(_safe_run, service, job["id"])
        return job

    def _safe_run(svc: JobService, job_id: str) -> None:
        with lock:
            try:
                svc.run(job_id)
            except Exception:
                # run() records failures internally; this guards the thread pool.
                pass

    @app.get(f"/{API_VERSION}/jobs")
    def list_jobs(state: str | None = None, limit: int = 50, _: None = Depends(require_auth)):
        return {"jobs": service.list(state=state, limit=limit)}

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}")
    def get_job(job_id: str, _: None = Depends(require_auth)):
        return service.get(job_id)

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/run")
    def run_job(job_id: str, _: None = Depends(require_auth)):
        return service.run(job_id)

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/cancel")
    def cancel_job(job_id: str, _: None = Depends(require_auth)):
        return service.cancel(job_id)

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}/events")
    def job_events(job_id: str, after: int = 0, _: None = Depends(require_auth)):
        return {"events": service.events(job_id, after=after)}

    # ------------------------------------------------------------------ #
    # artifacts / models / weights
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/models/import")
    def import_model(payload: dict[str, Any], _: None = Depends(require_auth)):
        repo_id = payload.get("repo_id")
        source = payload.get("source") or payload.get("path") or repo_id
        if not source:
            raise ValueError("model import requires 'source', 'path' or 'repo_id'")
        is_hf_source = bool(repo_id) or Path(str(source)).is_dir() or str(source).lower().endswith(".safetensors")
        if is_hf_source:
            record, report = service.import_model(
                source,
                name=payload.get("name"),
                family=payload.get("family"),
                revision=payload.get("revision"),
                repo_id=repo_id,
            )
            return {
                "artifact_id": record.artifact_id,
                "name": record.manifest.name,
                "family": record.manifest.family,
                "summary": record.manifest.summary,
                "checksum": record.manifest.checksum,
                "compatibility": report["compatibility"],
                "report": report,
            }
        record = service.import_artifact(
            source, name=payload.get("name"), family=payload.get("family")
        )
        response = {
            "artifact_id": record.artifact_id,
            "name": record.manifest.name,
            "family": record.manifest.family,
            "summary": record.manifest.summary,
            "checksum": record.manifest.checksum,
        }
        import_report = record.manifest.metadata.get("import_report")
        if isinstance(import_report, Mapping):
            response["format"] = import_report.get("format")
            response["import_report"] = dict(import_report)
        return response

    @app.post(f"/{API_VERSION}/models/{{artifact_id}}/quantize")
    def quantize_model(artifact_id: str, payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.quantize_model(
            artifact_id,
            scheme=str(payload.get("scheme", "int8")),
        )

    @app.post(f"/{API_VERSION}/models/{{artifact_id}}/forward")
    def forward_model(artifact_id: str, payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        if "tokens" not in payload:
            raise HTTPException(400, detail={"code": "request.invalid", "message": "provide tokens"})
        client = Uzen(root=service.root)
        return client.forward_model(
            artifact_id,
            tokens=payload["tokens"],
            positions=payload.get("positions"),
            include_values=bool(payload.get("include_values", False)),
            capture=payload.get("capture"),
            quantize=payload.get("quantize"),
        )

    @app.post(f"/{API_VERSION}/models/{{artifact_id}}/forward-text")
    def forward_text(artifact_id: str, payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        if "text" not in payload or not isinstance(payload["text"], str):
            raise HTTPException(400, detail={"code": "request.invalid", "message": "provide text"})
        client = Uzen(root=service.root)
        return client.forward_text(
            artifact_id,
            text=payload["text"],
            positions=payload.get("positions"),
            include_values=bool(payload.get("include_values", False)),
            capture=payload.get("capture"),
            quantize=payload.get("quantize"),
        )

    @app.post(f"/{API_VERSION}/models/{{artifact_id}}/generate-text")
    def generate_text(artifact_id: str, payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        if "text" not in payload or not isinstance(payload["text"], str):
            raise HTTPException(400, detail={"code": "request.invalid", "message": "provide text"})
        client = Uzen(root=service.root)
        return client.generate_text(
            artifact_id,
            text=payload["text"],
            max_new_tokens=int(payload.get("max_new_tokens", 16)),
            eos_token_id=payload.get("eos_token_id"),
            positions=payload.get("positions"),
            strategy=str(payload.get("strategy", "greedy")),
            temperature=float(payload.get("temperature", 1.0)),
            top_k=payload.get("top_k"),
            top_p=payload.get("top_p"),
            seed=payload.get("seed"),
            num_beams=int(payload.get("num_beams", 1)),
            quantize=payload.get("quantize"),
        )

    @app.post(f"/{API_VERSION}/models/compare-quantized")
    def compare_quantized_model(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        required = ("source_artifact_id", "quantized_artifact_id", "tokens")
        missing = [key for key in required if key not in payload]
        if missing:
            raise HTTPException(
                400,
                detail={"code": "request.invalid", "message": f"provide {', '.join(missing)}"},
            )
        client = Uzen(root=service.root)
        return client.compare_quantized_model(
            payload["source_artifact_id"],
            payload["quantized_artifact_id"],
            tokens=payload["tokens"],
            positions=payload.get("positions"),
            atol=float(payload.get("atol", 1e-5)),
            rtol=float(payload.get("rtol", 1e-5)),
            max_abs_error_threshold=(
                float(payload["max_abs_error_threshold"])
                if payload.get("max_abs_error_threshold") is not None
                else None
            ),
            mean_abs_error_threshold=(
                float(payload["mean_abs_error_threshold"])
                if payload.get("mean_abs_error_threshold") is not None
                else None
            ),
            max_rel_error_threshold=(
                float(payload["max_rel_error_threshold"])
                if payload.get("max_rel_error_threshold") is not None
                else None
            ),
        )

    @app.post(f"/{API_VERSION}/models/compare")
    def compare_models(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        required = ("source_artifact_id", "candidate_artifact_id", "tokens")
        missing = [key for key in required if key not in payload]
        if missing:
            raise HTTPException(
                400,
                detail={"code": "request.invalid", "message": f"provide {', '.join(missing)}"},
            )
        client = Uzen(root=service.root)
        return client.compare_models(
            payload["source_artifact_id"],
            payload["candidate_artifact_id"],
            tokens=payload["tokens"],
            positions=payload.get("positions"),
            atol=float(payload.get("atol", 1e-5)),
            rtol=float(payload.get("rtol", 1e-5)),
            max_abs_error_threshold=(
                float(payload["max_abs_error_threshold"])
                if payload.get("max_abs_error_threshold") is not None
                else None
            ),
            mean_abs_error_threshold=(
                float(payload["mean_abs_error_threshold"])
                if payload.get("mean_abs_error_threshold") is not None
                else None
            ),
            max_rel_error_threshold=(
                float(payload["max_rel_error_threshold"])
                if payload.get("max_rel_error_threshold") is not None
                else None
            ),
        )

    @app.get(f"/{API_VERSION}/models")
    def list_models(limit: int = 100, _: None = Depends(require_auth)):
        return {"models": service.models(limit=limit)}

    @app.get(f"/{API_VERSION}/weights")
    def list_weights(limit: int = 100, _: None = Depends(require_auth)):
        return {"weights": service.artifacts(limit=limit)}

    @app.post(f"/{API_VERSION}/weights/analyze")
    def analyze_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        if "artifact_id" in payload:
            package = service.get_package(payload["artifact_id"])
            return service.wie.inspect_package(package).to_dict()
        if "path" in payload:
            return service.wie.inspect_file(payload["path"]).to_dict()
        raise HTTPException(400, detail={"code": "request.invalid", "message": "provide artifact_id or path"})

    @app.post(f"/{API_VERSION}/weights/generate")
    def generate_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.generate(
            family=payload.get("family"),
            spec=payload.get("spec"),
            template_artifact_id=payload.get("template_artifact_id"),
            method=str(payload.get("method", "orthogonal")),
            params=payload.get("params"),
            seed=payload.get("seed", 0),
        )

    @app.post(f"/{API_VERSION}/weights/modify")
    def modify_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.modify(
            payload["artifact_id"],
            payload["method"],
            params=payload.get("params"),
            others=payload.get("others", []),
            seed=payload.get("seed", 0),
        )

    @app.post(f"/{API_VERSION}/weights/reconstruct")
    def reconstruct_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "weight_reconstruction",
                "objective": payload.get("objective", "reconstruct weights from supplied components"),
                "inputs": payload["inputs"],
                "operations": payload.get("operations", []),
                "evaluation_plan": payload.get("evaluation_plan", {}),
                "seed": payload.get("seed", 0),
                "budget": payload.get("budget", {}),
                "options": payload.get("options", {}),
            }
        )
        return service.run(job["id"])

    @app.post(f"/{API_VERSION}/weights/evaluate")
    def evaluate_weights(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.interfaces.python_sdk.client import Uzen

        client = Uzen(root=service.root)
        return client.evaluate(
            payload["artifact_id"],
            reference_artifact_id=payload.get("reference_artifact_id"),
            family=payload.get("family"),
            spec=payload.get("spec"),
        )

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}")
    def artifact_manifest(artifact_id: str, _: None = Depends(require_auth)):
        return service.artifact_manifest(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/verify")
    def verify_artifact(artifact_id: str, _: None = Depends(require_auth)):
        return service.verify_artifact(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/lineage")
    def artifact_lineage(artifact_id: str, _: None = Depends(require_auth)):
        return service.store.lineage_graph().describe(artifact_id)

    @app.get(f"/{API_VERSION}/artifacts/{{artifact_id}}/export")
    def artifact_export(artifact_id: str, fmt: str = "safetensors", _: None = Depends(require_auth)):
        target = Path(service.root) / "exports" / f"{artifact_id}.{ 'safetensors' if fmt == 'safetensors' else fmt }"
        path = service.store.export(artifact_id, target, fmt=fmt)
        if fmt == "safetensors":
            return FileResponse(path, media_type="application/octet-stream", filename=f"{artifact_id}.safetensors")
        return JSONResponse({"path": str(path)})

    # ------------------------------------------------------------------ #
    # architectures
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/architectures/compose")
    def compose_architectures(payload: dict[str, Any], _: None = Depends(require_auth)):
        artifact_ids = payload.get("artifact_ids")
        if not isinstance(artifact_ids, list) or len(artifact_ids) < 2:
            raise HTTPException(
                400,
                detail={
                    "code": "request.invalid",
                    "message": "provide at least two artifact_ids",
                },
            )
        return service.compose_architectures(
            [str(item) for item in artifact_ids],
            mode=str(payload.get("mode", "sequential")),
            name=payload.get("name"),
            seed=payload.get("seed", 0),
        )

    @app.post(f"/{API_VERSION}/architectures/{{artifact_id}}/edit")
    def edit_architecture(
        artifact_id: str, payload: dict[str, Any], _: None = Depends(require_auth)
    ):
        operation = payload.get("operation")
        if not operation:
            raise HTTPException(
                400,
                detail={"code": "request.invalid", "message": "provide operation"},
            )
        return service.edit_architecture(
            artifact_id,
            str(operation),
            node_id=payload.get("node_id"),
            keep_layers=payload.get("keep_layers"),
            new_op=payload.get("new_op"),
            params=payload.get("params"),
            port=str(payload.get("port", "a")),
        )

    @app.post(f"/{API_VERSION}/architectures/synthesize")
    def synthesize_architecture(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "architecture_synthesis",
                "objective": payload.get("objective", "synthesize an architecture"),
                "inputs": payload.get("inputs", []),
                "options": {"family": payload["family"], **payload.get("spec", {})},
                "seed": payload.get("seed", 0),
            }
        )
        return service.run(job["id"])

    @app.post(f"/{API_VERSION}/architectures/research")
    def research_architecture(payload: dict[str, Any], _: None = Depends(require_auth)):
        job = service.submit(
            {
                "task_scope": "model_architecture_research",
                "objective": payload.get("objective", "research architecture variants"),
                "options": {
                    "research_family": payload.get("family", "transformer_decoder"),
                    "base_spec": payload.get("base_spec", {}),
                    "variants": payload.get("variants"),
                },
                "seed": payload.get("seed", 0),
            }
        )
        return service.run(job["id"])

    # ------------------------------------------------------------------ #
    # research campaigns (N6/N7/N8)
    # ------------------------------------------------------------------ #

    @app.post(f"/{API_VERSION}/campaigns")
    def run_campaign(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine, CampaignSpec

        if "spec" not in payload:
            raise HTTPException(400, detail={"code": "request.invalid", "message": "provide spec"})
        spec = CampaignSpec.from_dict(payload["spec"])
        if bool(payload.get("background", False)):
            from uzen.core.campaign.control import submit_campaign_run

            return submit_campaign_run(
                service,
                spec,
                distributed=bool(payload.get("distributed", False)),
                resume=bool(payload.get("resume", True)),
                max_rounds=payload.get("max_rounds"),
                trial_timeout_seconds=float(payload.get("trial_timeout_seconds", 600.0)),
                poll_interval_seconds=float(payload.get("poll_interval_seconds", 0.1)),
            )
        executor = None
        if bool(payload.get("distributed", False)):
            from uzen.core.campaign import DistributedTrialExecutor

            executor = DistributedTrialExecutor(
                service,
                timeout_seconds=float(payload.get("trial_timeout_seconds", 600.0)),
                poll_interval_seconds=float(payload.get("poll_interval_seconds", 0.1)),
            )
        engine = CampaignEngine(service.root, trial_executor=executor, job_service=service)
        report = engine.run(
            spec,
            resume=bool(payload.get("resume", True)),
            max_rounds=payload.get("max_rounds"),
        )
        return report.to_dict()

    @app.get(f"/{API_VERSION}/campaigns")
    def list_campaigns(limit: int = 50, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return {"campaigns": CampaignEngine(service.root).list_campaigns(limit=limit)}

    @app.post(f"/{API_VERSION}/campaigns/verify")
    def verify_campaign_corpus(payload: dict[str, Any], _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return CampaignEngine(service.root).verify_corpus(payload.get("digests"))

    @app.get(f"/{API_VERSION}/campaigns/{{campaign_id}}")
    def get_campaign(campaign_id: str, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return CampaignEngine(service.root).get(campaign_id).to_dict()

    @app.get(f"/{API_VERSION}/campaigns/{{campaign_id}}/events")
    def campaign_events(campaign_id: str, after: int = 0, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return {"events": CampaignEngine(service.root).events(campaign_id, after=after)}

    @app.get(f"/{API_VERSION}/campaigns/{{campaign_id}}/jobs")
    def campaign_jobs(campaign_id: str, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        engine = CampaignEngine(service.root, job_service=service)
        return {"campaign_id": campaign_id, "jobs": engine.jobs(campaign_id)}

    @app.post(f"/{API_VERSION}/campaigns/{{campaign_id}}/cancel")
    def cancel_campaign(campaign_id: str, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return CampaignEngine(service.root, job_service=service).cancel(campaign_id)

    @app.get(f"/{API_VERSION}/campaigns/{{campaign_id}}/corpus")
    def campaign_corpus(campaign_id: str, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        engine = CampaignEngine(service.root)
        report = engine.get(campaign_id)
        return {"campaign_id": campaign_id, "merkle": report.corpus_digest, "records": engine.corpus.digests()}

    @app.get(f"/{API_VERSION}/benchmarks/{{digest}}")
    def replay_benchmark(digest: str, _: None = Depends(require_auth)):
        from uzen.core.campaign import CampaignEngine

        return CampaignEngine(service.root).replay(digest)

    # ------------------------------------------------------------------ #
    # workers
    # ------------------------------------------------------------------ #

    @app.get(f"/{API_VERSION}/workers")
    def list_workers(_: None = Depends(require_auth)):
        return {"workers": service.db.workers()}

    @app.post(f"/{API_VERSION}/workers/register", status_code=201)
    def register_worker(payload: dict[str, Any]):
        import secrets

        worker_id = str(payload.get("worker_id", "")).strip()
        if not worker_id:
            raise HTTPException(400, detail={"code": "worker.invalid", "message": "worker_id required"})
        secret = secrets.token_urlsafe(48)
        service.db.register_worker(
            worker_id=worker_id,
            protocol_version=str(payload.get("protocol_version", "uzen.worker/1")),
            capabilities=dict(payload.get("capabilities") or {}),
            token_hash=hash_secret(secret),
        )
        return {
            "worker_id": worker_id,
            "secret": secret,
            "protocol_version": "uzen.worker/1",
            "note": "store this secret on the worker; it is not retrievable again",
        }

    def _authenticate_worker(payload: Mapping[str, Any]) -> str:
        worker_id = str(payload.get("worker_id", ""))
        secret = str(payload.get("secret", ""))
        stored = service.db.worker_token_hash(worker_id)
        if not stored or not secret or not hmac.compare_digest(hash_secret(secret), stored):
            raise HTTPException(401, detail={"code": "worker.auth_failed", "message": "invalid worker credentials"})
        return worker_id

    @app.post(f"/{API_VERSION}/workers/heartbeat")
    def worker_heartbeat(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        status = payload.get("status")
        service.db.heartbeat(worker_id, status=str(status) if status is not None else None)
        return {"ok": True, "worker_id": worker_id}

    @app.post(f"/{API_VERSION}/workers/claim")
    def worker_claim(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        capabilities = payload.get("capabilities")
        requested_kinds = payload.get("job_kinds")
        if requested_kinds is not None and not isinstance(requested_kinds, list):
            raise HTTPException(
                400,
                detail={"code": "worker.invalid", "message": "job_kinds must be a list"},
            )
        job = scheduler.claim(
            worker_id,
            capabilities=dict(capabilities) if isinstance(capabilities, Mapping) else None,
            lease_seconds=(
                float(payload["lease_seconds"]) if payload.get("lease_seconds") is not None else None
            ),
            wait_seconds=(
                float(payload["wait_seconds"]) if payload.get("wait_seconds") is not None else 0.0
            ),
            job_kinds=(
                tuple(str(kind) for kind in requested_kinds)
                if requested_kinds is not None
                else None
            ),
        )
        return {"job": job, "scheduler": scheduler.describe()}

    @app.post(f"/{API_VERSION}/workers/renew")
    def worker_renew(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        ok = scheduler.renew(
            str(payload.get("job_id", "")),
            worker_id,
            token=(str(payload["token"]) if payload.get("token") else None),
            lease_seconds=(
                float(payload["lease_seconds"]) if payload.get("lease_seconds") is not None else None
            ),
        )
        return {
            "ok": ok,
            "detail": None if ok else "lease is no longer held by this worker",
        }

    @app.post(f"/{API_VERSION}/workers/report")
    def worker_report(payload: dict[str, Any]):
        worker_id = _authenticate_worker(payload)
        job_id = str(payload.get("job_id", ""))
        token = str(payload.get("token") or "")
        if not job_id or not token:
            raise HTTPException(
                400,
                detail={
                    "code": "worker.invalid",
                    "message": "job_id and the dispatch token from the claim are required",
                },
            )
        body = payload.get("result")
        if not isinstance(body, Mapping):
            raise HTTPException(400, detail={"code": "worker.invalid", "message": "result object required"})
        outcome = scheduler.report(
            job_id,
            worker_id=worker_id,
            token=token,
            state=str(body.get("state", "FAILED")),
            result=body.get("result") if isinstance(body.get("result"), Mapping) else None,
            error=body.get("error") if isinstance(body.get("error"), Mapping) else None,
        )
        return {"ok": outcome["status"] == "accepted", **outcome}

    # ------------------------------------------------------------------ #
    # scheduler
    # ------------------------------------------------------------------ #

    @app.get(f"/{API_VERSION}/scheduler")
    def scheduler_status(_: None = Depends(require_auth)):
        return {"scheduler": scheduler.describe(), **scheduler.status()}

    @app.post(f"/{API_VERSION}/scheduler/reap")
    def scheduler_reap(_: None = Depends(require_auth)):
        return {"reaped": scheduler.reap()}

    @app.post(f"/{API_VERSION}/jobs/{{job_id}}/preempt")
    def preempt_job(job_id: str, payload: dict[str, Any], _: None = Depends(require_auth)):
        return scheduler.preempt(job_id, reason=str(payload.get("reason", "preempted by operator")))

    @app.post(f"/{API_VERSION}/workers/{{worker_id}}/drain")
    def drain_worker(worker_id: str, _: None = Depends(require_auth)):
        return scheduler.drain_worker(worker_id)

    @app.post(f"/{API_VERSION}/workers/{{worker_id}}/resume")
    def resume_worker(worker_id: str, _: None = Depends(require_auth)):
        return scheduler.resume_worker(worker_id)

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}/dispatch")
    def job_dispatch_history(job_id: str, _: None = Depends(require_auth)):
        return scheduler.history(job_id)

    @app.get(f"/{API_VERSION}/jobs/{{job_id}}/eligibility")
    def job_eligibility(job_id: str, _: None = Depends(require_auth)):
        return scheduler.eligible_workers(job_id)

    return app
