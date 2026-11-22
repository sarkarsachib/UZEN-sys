"""Job orchestration service.

Owns the full job lifecycle described in the specification::

    CREATED -> QUEUED -> DISPATCHED -> RUNNING -> VALIDATING -> SUCCEEDED/FAILED
    (cancellation: CANCEL_REQUESTED -> CANCELLED)

The service is the seam between the headless control plane (REST/MCP/CLI/workers) and
the computational plane (ANE + NWPC + VEL). It resolves input artifacts, builds graph
context when the request supplies an architecture spec, runs the engineering job,
persists the result as an immutable content-addressed artifact, and records the full
provenance and verification evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.ane.planner import ANEResult, AutonomousNeuralEngineer, JobRequest
from uzen.core.nase.synth import ArchitectureSynthesizer
from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph, GraphValidationError
from uzen.core.nws.families import build_architecture
from uzen.core.nws.store import ArtifactRecord, ArtifactStore
from uzen.core.nws.tensors import WeightPackage
from uzen.core.vel.verify import verify
from uzen.core.wie.inspector import WeightIntelligenceEngine
from uzen.core.wie.model_execution import (
    ModelExecution,
    ModelGeneration,
    ModelTextExecution,
    compare_executions,
    execute_stored_model,
    execute_stored_text,
    graph_from_manifest,
    greedy_generate,
)
from uzen.core.wie.tokenization import store_tokenizer
from uzen.engines.mar.engine import ModelArchitectureResearch
from uzen.engines.war.engine import WeightArchitectureResearch
from uzen.engines.wge.engine import WeightGenerationEngine
from uzen.engines.wme.engine import WeightModificationEngine
from uzen.errors import (
    ArtifactIntegrityError,
    ArtifactNotFound,
    InvalidOperationParameters,
    JobNotFound,
    JobStateError,
    UnsupportedFormat,
    UzenError,
)
from uzen.infrastructure.storage.db import Database, JobRow, TERMINAL_STATES
from uzen.jobs.scheduling import requirements_from_request
from uzen.policies.strict_plus.engine import StrictPlusEngine, default_engine
from uzen.util.canonical import sha256_file, utc_now_iso


class JobService:
    def __init__(
        self,
        root: str | Path,
        *,
        engine: StrictPlusEngine | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine or default_engine()
        self.store = ArtifactStore(self.root / "artifacts")
        self.db = Database(db_path or (self.root / "uzen.sqlite3"))
        self.ane = AutonomousNeuralEngineer(engine=self.engine)
        self.wie = WeightIntelligenceEngine()
        self.wme = WeightModificationEngine(engine=self.engine)
        self.wge = WeightGenerationEngine(engine=self.engine)
        self.war = WeightArchitectureResearch(engine=self.engine)
        self.mar = ModelArchitectureResearch(engine=self.engine)
        self.nase = ArchitectureSynthesizer()

    # ------------------------------------------------------------------ #
    # artifacts
    # ------------------------------------------------------------------ #

    def import_artifact(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
    ) -> ArtifactRecord:
        p = Path(path)
        if p.is_dir():
            record, _report = self.import_model(p, name=name, family=family)
            return record
        suffix = p.suffix.lower()
        if suffix in (".pt", ".pth", ".bin", ".ckpt"):
            return self._import_torch_checkpoint(p, name=name, family=family)
        if suffix == ".onnx":
            return self._import_onnx_model(p, name=name, family=family)
        graph_doc: dict[str, Any] | None = None
        report = self.wie.inspect_file(p)
        if report.format == "safetensors":
            from uzen.infrastructure.storage.safetensors_io import (
                decode_graph_metadata,
                read_safetensors,
            )

            arrays, header = read_safetensors(p)
            graph_doc = decode_graph_metadata(header.metadata)
        elif report.format == "npz":
            data = np.load(p, allow_pickle=False)
            arrays = {k: np.asarray(data[k]) for k in data.files}
            header = None
        elif report.format == "gguf":
            from uzen.infrastructure.storage.gguf_io import read_gguf

            arrays, _gguf_metadata = read_gguf(p)
            header = None
        else:
            raise InvalidOperationParameters(f"unsupported import format '{report.format}'")
        guessed = report.family_guess.get("family", "generic")
        source_hash = sha256_file(p)
        package = WeightPackage.from_arrays(
            arrays,
            name=name or p.stem,
            family=family or (guessed if guessed != "unknown" else "generic"),
            metadata={
                "imported_from": str(p),
                "source_sha256": report.metadata.get("sha256") or source_hash,
                "family_inference": report.family_guess,
            },
        )
        import_report: dict[str, Any] = {
            "format": report.format,
            "source_kind": f"{report.format}_file",
            "source": str(p),
            "source_hash": source_hash,
            "source_sha256": source_hash,
            "source_bytes": p.stat().st_size,
            "adapter": "legacy_artifact_importer",
            "graph_reconstructed": False,
            "num_tensors": len(package),
            "family_guess": report.family_guess,
        }
        artifact_metadata: dict[str, Any] = {
            "source_path": str(p),
            "family_guess": report.family_guess,
            "import_report": import_report,
        }
        if graph_doc is not None:
            from uzen.core.nir.graph import Graph

            Graph.from_dict(graph_doc).require_valid(package)
            import_report["graph_reconstructed"] = True
            import_report["graph_name"] = graph_doc.get("name")
            artifact_metadata["graph"] = graph_doc
        record = self.store.put(
            package,
            operation="import",
            reuse_existing=True,
            metadata=artifact_metadata,
            note=(
                "imported source artifact with embedded NIR graph"
                if graph_doc is not None
                else "imported source artifact (immutable origin)"
            ),
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation="import",
            numel=package.numel(),
            num_tensors=len(package),
        )
        if report.family_guess.get("family") not in (None, "unknown"):
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={"inferred": report.family_guess},
            )
        return record

    def _import_torch_checkpoint(
        self,
        path: Path,
        *,
        name: str | None,
        family: str | None,
    ) -> ArtifactRecord:
        """Import a PyTorch state dict without allowing arbitrary pickle execution."""
        from uzen.runtimes.pytorch_runtime import PyTorchRuntime

        source_hash = sha256_file(path)
        package = PyTorchRuntime(device="cpu").load(path)
        if sha256_file(path) != source_hash:
            raise ArtifactIntegrityError(
                "PyTorch checkpoint changed while it was being imported", path=str(path)
            )
        source_dtypes = {
            key: meta.source_dtype
            for key, meta in package.metas.items()
            if meta.source_dtype is not None
        }
        package = WeightPackage.from_arrays(
            package.tensors,
            name=name or package.name,
            family=family or package.family,
            metadata={
                **dict(package.metadata),
                "imported_from": str(path),
                "source_sha256": source_hash,
            },
            source_dtypes=source_dtypes,
        )
        import_report: dict[str, Any] = {
            "format": "pytorch",
            "source_kind": "pytorch_checkpoint",
            "source": str(path),
            "source_hash": source_hash,
            "source_sha256": source_hash,
            "source_bytes": path.stat().st_size,
            "adapter": "pytorch",
            "weights_only": True,
            "graph_reconstructed": False,
            "num_tensors": len(package),
            "source_dtypes": source_dtypes,
        }
        return self._store_imported_package(
            package,
            source=path,
            source_hash=source_hash,
            import_report=import_report,
            note="imported PyTorch weights-only state dict (no graph reconstruction)",
        )

    def _import_onnx_model(
        self,
        path: Path,
        *,
        name: str | None,
        family: str | None,
    ) -> ArtifactRecord:
        """Lift a supported ONNX model into NIR and store its graph and weights."""
        import onnx

        from uzen.runtimes.onnx_runtime import import_onnx

        model = onnx.load(str(path), load_external_data=False)
        if any(
            initializer.data_location == onnx.TensorProto.EXTERNAL
            for initializer in model.graph.initializer
        ):
            raise UnsupportedFormat(
                "ONNX models with external tensor data are not imported because the "
                "sidecar files would not be covered by the recorded source hash; "
                "save the model as a single-file ONNX model first"
            )
        source_hash = sha256_file(path)
        graph, package, onnx_metadata = import_onnx(model)
        if sha256_file(path) != source_hash:
            raise ArtifactIntegrityError(
                "ONNX model changed while it was being imported", path=str(path)
            )
        source_dtypes = {
            key: meta.source_dtype
            for key, meta in package.metas.items()
            if meta.source_dtype is not None
        }
        package = WeightPackage.from_arrays(
            package.tensors,
            name=name or package.name,
            family=family or package.family,
            metadata={
                **dict(package.metadata),
                "imported_from": str(path),
                "source_sha256": source_hash,
                "onnx": dict(onnx_metadata),
            },
            source_dtypes=source_dtypes,
        )
        graph.require_valid(package)
        import_report: dict[str, Any] = {
            "format": "onnx",
            "source_kind": "onnx_model",
            "source": str(path),
            "source_hash": source_hash,
            "source_sha256": source_hash,
            "source_bytes": path.stat().st_size,
            "adapter": "onnx",
            "graph_reconstructed": True,
            "graph_name": graph.name,
            "num_nodes": len(graph.nodes),
            "num_tensors": len(package),
            "onnx": dict(onnx_metadata),
        }
        return self._store_imported_package(
            package,
            source=path,
            source_hash=source_hash,
            import_report=import_report,
            graph_doc=graph.to_dict(),
            note="imported ONNX model with validated NIR graph",
        )

    def _store_imported_package(
        self,
        package: WeightPackage,
        *,
        source: Path,
        source_hash: str,
        import_report: Mapping[str, Any],
        graph_doc: Mapping[str, Any] | None = None,
        note: str,
    ) -> ArtifactRecord:
        metadata: dict[str, Any] = {
            "source_path": str(source),
            "import_report": dict(import_report),
        }
        if graph_doc is not None:
            metadata["graph"] = dict(graph_doc)
        record = self.store.put(
            package,
            operation="import",
            reuse_existing=True,
            operations=[
                {
                    "op": "import.source",
                    "format": str(import_report["format"]),
                    "sha256": source_hash,
                }
            ],
            metadata=metadata,
            note=note,
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation="import",
            numel=package.numel(),
            num_tensors=len(package),
        )
        if package.family not in (None, "unknown", "generic"):
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={"import": dict(import_report)},
            )
        return record

    def import_model(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
        revision: str | None = None,
        repo_id: str | None = None,
        on_progress: Any | None = None,
    ) -> tuple[ArtifactRecord, dict[str, Any]]:
        """Import a Hugging Face/safetensors checkpoint and return its compatibility report."""
        from uzen.core.wie.importer import ModelImporter

        result = ModelImporter().import_source(
            source,
            name=name,
            family=family,
            revision=revision,
            repo_id=repo_id,
            on_progress=on_progress,
        )
        package = result.require_ok()
        artifact_package = result.artifact_package or package
        report = result.report.to_dict()
        reconstruction = result.reconstruction.report if result.reconstruction is not None else None
        graph_doc = (
            reconstruction.graph
            if reconstruction is not None and reconstruction.graph_reconstructed
            else artifact_package.metadata.get("graph")
        )
        metadata: dict[str, Any] = {
            "source_path": str(source),
            "import_report": report,
        }
        if graph_doc is not None:
            from uzen.core.nir.graph import Graph

            Graph.from_dict(graph_doc).require_valid(artifact_package)
            metadata["graph"] = graph_doc
            if reconstruction is None:
                report["metadata"]["graph_restored_from_metadata"] = True
        tokenizer_spec = None
        if result.tokenizer_json is not None:
            tokenizer_spec = store_tokenizer(self.root, result.tokenizer_json)
            metadata["tokenizer"] = tokenizer_spec
        # A tokenizer changes what the model can execute, so imports with one are
        # distinct from tokenizer-less imports of identical tensors.
        import_operations = (
            [{"op": "tokenizer.attach", "kind": tokenizer_spec["kind"], "sha256": tokenizer_spec["sha256"]}]
            if tokenizer_spec is not None
            else []
        )
        record = self.store.put(
            artifact_package,
            operation="import",
            reuse_existing=True,
            operations=import_operations,
            metadata=metadata,
            note=(
                "imported HF/safetensors checkpoint with validated NIR reconstruction"
                if graph_doc is not None
                else "imported HF/safetensors checkpoint (immutable source tensor names)"
            ),
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=artifact_package.name,
            family=artifact_package.family,
            operation="import",
            numel=artifact_package.numel(),
            num_tensors=len(artifact_package),
        )
        if result.report.family not in (None, "unknown", "generic"):
            reconstruction_summary = None
            if reconstruction is not None:
                reconstruction_summary = {
                    "status": reconstruction.status,
                    "architecture": reconstruction.architecture,
                    "family": reconstruction.family,
                    "graph_reconstructed": reconstruction.graph_reconstructed,
                    "execution_validated": reconstruction.execution_validated,
                    "unresolved": reconstruction.unresolved,
                }
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=artifact_package.family,
                spec={
                    "import": {
                        "source_kind": result.report.source_kind,
                        "source_hash": result.report.source_hash,
                        "revision": result.report.revision,
                        "compatibility": result.report.compatibility.to_dict(),
                        "reconstruction": reconstruction_summary,
                    }
                },
            )
        return record, report

    def get_package(self, artifact_id: str) -> WeightPackage:
        if not self.store.exists(artifact_id):
            raise ArtifactNotFound(f"artifact '{artifact_id}' is not in the store")
        return self.store.get(artifact_id)

    def forward_model(
        self,
        artifact_id: str,
        *,
        tokens: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        capture: Sequence[str] | str | None = None,
        quantize: str | None = None,
    ) -> ModelExecution:
        """Execute the stored graph, optionally with weight-only int8 quantization."""
        return execute_stored_model(
            self.get_package(artifact_id),
            self.artifact_manifest(artifact_id),
            artifact_id=artifact_id,
            tokens=tokens,
            positions=positions,
            capture=capture,
            quantize=quantize,
        )

    def forward_text(
        self,
        artifact_id: str,
        *,
        text: str,
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        capture: Sequence[str] | str | None = None,
        quantize: str | None = None,
    ) -> ModelTextExecution:
        """Run text through the stored tokenizer and reconstructed model graph."""
        return execute_stored_text(
            self.get_package(artifact_id),
            self.artifact_manifest(artifact_id),
            root=self.root,
            artifact_id=artifact_id,
            text=text,
            positions=positions,
            capture=capture,
            quantize=quantize,
        )

    def generate_text(
        self,
        artifact_id: str,
        *,
        text: str,
        max_new_tokens: int = 16,
        eos_token_id: int | None = None,
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        num_beams: int = 1,
        quantize: str | None = None,
    ) -> ModelGeneration:
        """Generate tokens with greedy, sampled or beam decoding, optionally int8."""
        return greedy_generate(
            self.get_package(artifact_id),
            self.artifact_manifest(artifact_id),
            root=self.root,
            artifact_id=artifact_id,
            text=text,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            positions=positions,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            num_beams=num_beams,
            quantize=quantize,
        )

    def quantize_model(
        self,
        artifact_id: str,
        *,
        scheme: str = "int8",
    ) -> dict[str, Any]:
        """Persist a weight-only int8 variant of a stored model artifact."""
        if scheme != "int8":
            raise InvalidOperationParameters("quantization scheme must be 'int8'")
        source_manifest = self.artifact_manifest(artifact_id)
        graph = graph_from_manifest(source_manifest)
        if graph.metadata.get("quantization") is not None:
            raise InvalidOperationParameters(
                f"artifact '{artifact_id}' already contains a quantized graph"
            )
        package = self.get_package(artifact_id)
        from uzen.core.wie.quantization import quantize_int8

        quantized = quantize_int8(graph, package)
        report = quantized.report.to_dict()
        if report["num_rewritten_nodes"] == 0:
            raise InvalidOperationParameters(
                f"artifact '{artifact_id}' has no quantizable linear weights"
            )
        source_metadata = dict(source_manifest.get("metadata") or {})
        metadata: dict[str, Any] = {
            "graph": quantized.graph.to_dict(),
            "quantization": report,
            "quantized_from": artifact_id,
        }
        if isinstance(source_metadata.get("tokenizer"), Mapping):
            metadata["tokenizer"] = dict(source_metadata["tokenizer"])
        record = self.store.put(
            quantized.package,
            operation="quantize.int8",
            parents=[artifact_id],
            reuse_existing=True,
            operations=[
                {
                    "op": "quantize.weight_only_int8",
                    "scheme": report["scheme"],
                    "bits": report["bits"],
                    "num_rewritten_nodes": report["num_rewritten_nodes"],
                    "source_artifact_id": artifact_id,
                }
            ],
            metadata=metadata,
            note="weight-only int8 execution variant with quantization report",
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=quantized.package.name,
            family=quantized.package.family,
            operation="quantize.int8",
            numel=quantized.package.numel(),
            num_tensors=len(quantized.package),
        )
        if quantized.package.family not in (None, "unknown", "generic"):
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=quantized.package.family,
                spec={"quantization": report, "parent_artifact_id": artifact_id},
            )
        return {
            "artifact_id": record.artifact_id,
            "parent_artifact_id": artifact_id,
            "graph_name": quantized.graph.name,
            "summary": record.manifest.summary,
            "quantization": report,
        }

    def compare_quantized_model(
        self,
        source_artifact_id: str,
        quantized_artifact_id: str,
        *,
        tokens: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        max_abs_error_threshold: float | None = None,
        mean_abs_error_threshold: float | None = None,
        max_rel_error_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Measure output drift between a source model and its quantized variant."""
        if atol < 0 or rtol < 0:
            raise InvalidOperationParameters("atol and rtol must be non-negative")
        source_manifest = self.artifact_manifest(source_artifact_id)
        candidate_manifest = self.artifact_manifest(quantized_artifact_id)
        source_metadata = dict(source_manifest.get("metadata") or {})
        candidate_metadata = dict(candidate_manifest.get("metadata") or {})
        if isinstance(source_metadata.get("quantization"), Mapping):
            raise InvalidOperationParameters(
                "source artifact is already quantized; compare it with a floating-point source"
            )
        candidate_graph = graph_from_manifest(candidate_manifest)
        report = candidate_graph.metadata.get("quantization")
        if not isinstance(report, Mapping):
            raise InvalidOperationParameters(
                "candidate artifact is not a persisted quantized variant"
            )
        if candidate_metadata.get("quantized_from") != source_artifact_id:
            raise InvalidOperationParameters(
                "candidate artifact was not persisted from the supplied source artifact"
            )
        source = execute_stored_model(
            self.get_package(source_artifact_id),
            source_manifest,
            artifact_id=source_artifact_id,
            tokens=tokens,
            positions=positions,
        )
        candidate = execute_stored_model(
            self.get_package(quantized_artifact_id),
            candidate_manifest,
            artifact_id=quantized_artifact_id,
            tokens=tokens,
            positions=positions,
        )
        comparison = compare_executions(
            source,
            candidate,
            atol=atol,
            rtol=rtol,
            max_abs_error_threshold=max_abs_error_threshold,
            mean_abs_error_threshold=mean_abs_error_threshold,
            max_rel_error_threshold=max_rel_error_threshold,
        )
        comparison["source_vs_quantized"] = True
        comparison["quantization"] = dict(report)
        return comparison

    def compare_models(
        self,
        source_artifact_id: str,
        candidate_artifact_id: str,
        *,
        tokens: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        max_abs_error_threshold: float | None = None,
        mean_abs_error_threshold: float | None = None,
        max_rel_error_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Compare two model artifacts on exactly the same inputs and graph."""
        source_manifest = self.artifact_manifest(source_artifact_id)
        candidate_manifest = self.artifact_manifest(candidate_artifact_id)
        source_graph = graph_from_manifest(source_manifest)
        candidate_graph = graph_from_manifest(candidate_manifest)
        if source_graph.to_dict() != candidate_graph.to_dict():
            raise InvalidOperationParameters(
                "source and candidate artifacts have different stored NIR graphs; "
                "only weight-only modifications can be compared as the same model"
            )
        source = execute_stored_model(
            self.get_package(source_artifact_id),
            source_manifest,
            artifact_id=source_artifact_id,
            tokens=tokens,
            positions=positions,
        )
        candidate = execute_stored_model(
            self.get_package(candidate_artifact_id),
            candidate_manifest,
            artifact_id=candidate_artifact_id,
            tokens=tokens,
            positions=positions,
        )
        return compare_executions(
            source,
            candidate,
            atol=atol,
            rtol=rtol,
            max_abs_error_threshold=max_abs_error_threshold,
            mean_abs_error_threshold=mean_abs_error_threshold,
            max_rel_error_threshold=max_rel_error_threshold,
        )

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def submit(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        job_request = JobRequest.from_dict(request)
        if job_request.policy != "STRICT_PLUS":
            raise InvalidOperationParameters("only STRICT_PLUS jobs are accepted")
        # Campaign trial jobs are a read-only task kind, not a registered engineering
        # operation. Validate their closed payload before they enter the queue.
        task_kind = job_request.options.get("task_kind")
        if task_kind == "campaign_trial":
            from uzen.core.campaign.distributed import validate_campaign_trial_request

            validate_campaign_trial_request(request)
        else:
            # Reject unregistered / out-of-scope / malformed operations at submission, so a
            # job never enters the queue carrying work the engine cannot legally perform.
            from uzen.core.nwpc.registry import default_registry
            from uzen.errors import OperationUnavailable

            registry = default_registry()
            for index, entry in enumerate(job_request.operations):
                if not isinstance(entry, Mapping):
                    raise InvalidOperationParameters(f"operation #{index} must be an object")
                code = str(entry.get("op") or entry.get("code") or "")
                if not code:
                    raise InvalidOperationParameters(f"operation #{index} has no 'op' code")
                spec = registry.get(code)
                if not spec.maturity.executable:
                    raise OperationUnavailable(
                        f"operation '{code}' is '{spec.maturity.value}' and cannot be executed",
                        operation=code,
                    )
                if job_request.task_scope not in spec.scopes:
                    raise OperationUnavailable(
                        f"operation '{code}' is not authorized in scope '{job_request.task_scope.value}'",
                        operation=code,
                        scope=job_request.task_scope.value,
                    )
                spec.validate_params(dict(entry.get("params") or {}))
        # The capability requirement set is derived from the request itself and stored
        # with the job, so dispatch can decide eligibility inside the claiming statement
        # without re-parsing the request (and without trusting the worker's word).
        requirements = requirements_from_request(request)
        row = self.db.create_job(
            request,
            idempotency_key=idempotency_key,
            requirements=requirements,
            priority=int(request.get("priority") or 0),
        )
        self.db.append_event(
            row.id,
            "queued",
            {
                "scope": job_request.task_scope.value,
                "requirements": requirements.to_dict(),
                "priority": row.priority,
            },
        )
        return row.to_dict()

    def get(self, job_id: str) -> dict[str, Any]:
        return self.db.get_job(job_id).to_dict()

    def list(self, *, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.db.list_jobs(state=state, limit=limit)]

    def events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        return self.db.events(job_id, after=after)

    def cancel(self, job_id: str) -> dict[str, Any]:
        row = self.db.get_job(job_id)
        if row.state in TERMINAL_STATES:
            return row.to_dict()
        self.db.request_cancel(job_id)
        if row.state in ("QUEUED", "CREATED"):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancelled before execution"})
        return self.db.get_job(job_id).to_dict()

    # ------------------------------------------------------------------ #

    def run(
        self,
        job_id: str,
        *,
        worker_id: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        """Execute a job, either as the control plane or as a worker holding a lease.

        A leased execution (``worker_id`` and ``token`` supplied) settles the job through
        :meth:`Database.finish_from_worker`, so the dispatch record, the attempt number and
        the idempotency guard are the same ones a remote worker goes through. A local
        execution settles it directly. Either way the caller never declares success: the
        state comes from the executed work or from the recorded error.
        """
        row = self.db.get_job(job_id)
        if row.state in TERMINAL_STATES:
            raise JobStateError(f"job '{job_id}' already finished ({row.state})")
        leased = worker_id is not None and token is not None
        if leased and (row.dispatch_token != token or row.worker_id != worker_id):
            raise JobStateError(
                f"job '{job_id}' is not leased to worker '{worker_id}' with that token",
                job_id=job_id,
                current_worker=row.worker_id,
            )
        if self.db.is_cancel_requested(job_id):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancel requested"})
            return self.db.get_job(job_id).to_dict()
        self.db.transition(job_id, "RUNNING", progress=0.05)
        request = JobRequest.from_dict(row.request)
        try:
            result = self._execute(job_id, request)
            if isinstance(result, dict) and "artifact_id" in result:
                result.setdefault("output_artifact_id", result["artifact_id"])
        except UzenError as exc:
            self._settle_failure(job_id, {"error": exc.to_dict(), "timestamp": utc_now_iso()}, worker_id, token)
            return self.db.get_job(job_id).to_dict()
        except Exception as exc:  # unexpected: recorded with type for diagnosis
            self._settle_failure(
                job_id,
                {
                    "error": {
                        "code": "internal.error",
                        "message": str(exc),
                        "details": {"type": type(exc).__name__},
                    },
                    "timestamp": utc_now_iso(),
                },
                worker_id,
                token,
            )
            return self.db.get_job(job_id).to_dict()
        self.db.transition(job_id, "VALIDATING", progress=0.9)
        if leased:
            assert worker_id is not None and token is not None
            self.db.finish_from_worker(
                job_id, worker_id=worker_id, token=token, state="SUCCEEDED", result=result
            )
        else:
            self.db.set_result(job_id, result)
            self.db.release_lease(job_id)
        return self.db.get_job(job_id).to_dict()

    def _settle_failure(
        self,
        job_id: str,
        error: Mapping[str, Any],
        worker_id: str | None,
        token: str | None,
    ) -> None:
        if worker_id is not None and token is not None:
            self.db.finish_from_worker(
                job_id,
                worker_id=worker_id,
                token=token,
                state="FAILED",
                error=dict(error.get("error") or {}),
            )
            return
        self.db.set_error(job_id, error)
        self.db.release_lease(job_id)

    def run_pending(self, limit: int = 1) -> list[dict[str, Any]]:
        out = []
        for row in self.db.list_jobs(state="QUEUED", limit=limit):
            out.append(self.run(row.id))
        return out

    # ------------------------------------------------------------------ #

    def _execute(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        if request.options.get("task_kind") == "campaign_trial":
            return self._execute_campaign_trial(job_id, request)
        scope = request.task_scope
        if scope.value in ("model_architecture_research", "architecture_synthesis", "model_creation"):
            return self._execute_architecture(job_id, request)
        if scope.value == "weight_architecture_research" and not request.operations:
            return self._execute_weight_research(job_id, request)

        packages = self._resolve_inputs(request)
        target, reference, graph, graph_inputs = self._resolve_context(request, packages)
        self.db.append_event(
            job_id,
            "planning",
            {"objective": request.objective, "inputs": list(packages)},
        )
        result: ANEResult = self.ane.execute(
            request,
            packages,
            target=target,
            graph=graph,
            graph_inputs=graph_inputs,
            reference=reference,
            job_id=job_id,
        )
        self.db.set_plan(job_id, result.plan.to_dict())
        self.db.append_event(
            job_id,
            "executed",
            {
                "candidates": len(result.candidates),
                "best_metric": result.best.metric if result.best else None,
                "diagnosis": result.diagnosis[-5:],
            },
        )
        if self.db.is_cancel_requested(job_id):
            self.db.transition(job_id, "CANCELLED", payload={"reason": "cancel observed before persist"})
            return {"cancelled": True}

        package = result.package
        assert package is not None
        validation = result.verification.to_dict() if result.verification else {}
        parent_ids = [p for p in (package.source_artifact_id,) if p]
        for name, pkg in packages.items():
            if pkg.source_artifact_id and pkg.source_artifact_id not in parent_ids:
                parent_ids.append(pkg.source_artifact_id)
        record = self.store.put(
            package,
            operation=result.best.op_code if result.best else "engineering_job",
            reuse_existing=True,
            parents=parent_ids,
            operations=result.trace,
            rng_seeds=result.seeds,
            policy=result.policy,
            validation=validation,
            metadata={
                "job_id": job_id,
                "objective": request.objective,
                "scope": scope.value,
                "notes": result.notes,
                "warnings": result.warnings,
                "diagnosis": result.diagnosis,
            },
            job_id=job_id,
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation=record.manifest.lineage.operation,
            numel=package.numel(),
            num_tensors=len(package),
            job_id=job_id,
        )
        return {
            "artifact_id": record.artifact_id,
            "summary": record.manifest.summary,
            "plan": result.plan.to_dict(),
            "candidates": [c.to_dict() for c in result.candidates],
            "best_candidate": result.best.to_dict() if result.best else None,
            "verification": validation,
            "notes": result.notes,
            "warnings": result.warnings,
            "diagnosis": result.diagnosis,
            "policy": result.policy,
            "trace": result.trace,
            "output_format": request.output_format,
        }

    # ------------------------------------------------------------------ #

    def _resolve_inputs(self, request: JobRequest) -> dict[str, WeightPackage]:
        packages: dict[str, WeightPackage] = {}
        for index, entry in enumerate(request.inputs):
            name = str(entry.get("name", f"input{index}"))
            if "artifact_id" in entry:
                packages[name] = self.get_package(str(entry["artifact_id"]))
            elif "path" in entry:
                record = self.import_artifact(str(entry["path"]))
                packages[name] = self.store.get(record.artifact_id)
            elif "inline" in entry:
                arrays = {k: np.asarray(v) for k, v in entry["inline"].items()}
                packages[name] = WeightPackage.from_arrays(arrays, name=name)
            else:
                raise InvalidOperationParameters(
                    f"input '{name}' must specify artifact_id, path, or inline tensors"
                )
        if not packages and request.task_scope.value not in ("weight_generation",):
            raise InvalidOperationParameters(
                "the job supplies no inputs; weight engineering requires at least one package"
            )
        if not packages:
            raise InvalidOperationParameters(
                "weight generation requires a template artifact or an architecture spec"
            )
        return packages

    def _resolve_context(
        self, request: JobRequest, packages: Mapping[str, WeightPackage]
    ) -> tuple[WeightPackage | None, WeightPackage | None, Any, dict[str, np.ndarray] | None]:
        plan = request.evaluation_plan or {}
        target = None
        reference = None
        if plan.get("target_artifact_id"):
            target = self.get_package(str(plan["target_artifact_id"]))
        if plan.get("reference_artifact_id"):
            reference = self.get_package(str(plan["reference_artifact_id"]))
        graph = None
        graph_inputs: dict[str, np.ndarray] | None = None
        graph_spec = plan.get("graph") or request.options.get("graph")
        if graph_spec:
            definition = build_architecture(
                str(graph_spec.get("family", "mlp")),
                {k: v for k, v in graph_spec.items() if k != "family"},
            )
            basis = reference or next(iter(packages.values()))
            compat = definition.graph.validate(basis)
            if compat.ok:
                graph = definition.graph
                batch = int(plan.get("batch", 2))
                seq = int(plan.get("seq", 8))
                inputs = definition.sample_inputs(batch=batch, seq=seq, seed=0)
                # ensure shapes match the supplied package family
                try:
                    execute(graph, inputs, basis)
                    graph_inputs = inputs
                except Exception:
                    graph = None
                    graph_inputs = None
        if graph is None and plan.get("metric") == "activation_agreement":
            raise InvalidOperationParameters(
                "activation_agreement requires evaluation_plan.graph (a family spec) and a "
                "compatible reference artifact"
            )
        return target, reference, graph, graph_inputs

    # ------------------------------------------------------------------ #

    def _execute_campaign_trial(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        from uzen.core.campaign.distributed import (
            campaign_trial_outcome,
            validate_campaign_trial_request,
        )

        spec, candidate = validate_campaign_trial_request(request.to_dict())
        outcome = campaign_trial_outcome(spec, candidate)
        self.db.append_event(
            job_id,
            "campaign_trial",
            {
                "campaign_id": spec.candidate_id,
                "candidate_digest": candidate.digest,
                "status": outcome["status"],
            },
        )
        return {
            "task_kind": "campaign_trial",
            "campaign_id": spec.candidate_id,
            "candidate_digest": candidate.digest,
            "trial": outcome,
        }

    def _execute_weight_research(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        packages = self._resolve_inputs(request)
        primary = next(iter(packages.values()))
        report = self.war.run(
            primary,
            experiments=request.operations or None,
            seed=request.seed,
            scope=request.task_scope,
        )
        self.db.append_event(job_id, "research_complete", {"experiments": len(report["experiments"])})
        # Research reports are measurements, not modified parameter sets: no artifact is
        # fabricated here. Any concrete candidate must be produced by an explicit
        # subsequent engineering job with its own provenance.
        return {
            "research": report,
            "artifacts": [],
            "notes": [
                "weight architecture research reports representation trade-offs; it does not "
                "claim downstream quality improvements",
            ],
        }

    # ------------------------------------------------------------------ #
    # architecture engineering (NASE)
    # ------------------------------------------------------------------ #

    def _definition_from_artifact(
        self, artifact_id: str
    ) -> tuple[Any, WeightPackage, dict[str, Any]]:
        manifest = self.artifact_manifest(artifact_id)
        graph_doc = (manifest.get("metadata") or {}).get("graph")
        if not graph_doc:
            raise InvalidOperationParameters(
                f"artifact '{artifact_id}' has no stored NIR graph; only synthesized "
                "architectures can be edited"
            )
        package = self.get_package(artifact_id)
        architecture = (manifest.get("metadata") or {}).get("architecture") or {}
        definition = self.nase.from_graph(
            Graph.from_dict(graph_doc),
            parameter_shapes={entry["name"]: entry["shape"] for entry in manifest["tensors"]},
            family=str(manifest.get("family", "custom")),
            spec=architecture.get("spec") if isinstance(architecture, Mapping) else None,
        )
        return definition, package, manifest

    def _architecture_inputs(self, graph: Graph, *, batch: int = 2, seq: int = 8) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        for name, spec in graph.inputs.items():
            shape = tuple(
                batch if dim == "B" else seq if isinstance(dim, str) else int(dim)
                for dim in spec.shape
            )
            inputs[name] = np.zeros(shape, dtype=spec.dtype)
        return inputs

    def _persist_architecture(
        self,
        definition: Any,
        package: WeightPackage,
        *,
        operation: str,
        parents: list[str],
        note: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        inputs = self._architecture_inputs(definition.graph)
        result = execute(definition.graph, inputs, package)
        verification = verify(
            package,
            graph=definition.graph,
            inputs=inputs,
            policy_records=[],
            operations=[],
        )
        record = self.store.put(
            package,
            operation=operation,
            parents=parents,
            reuse_existing=True,
            metadata={
                "architecture": definition.summary(),
                "graph": definition.graph.to_dict(),
                **(dict(extra or {})),
            },
            validation=verification.to_dict(),
            note=note,
        )
        self.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation=operation,
            numel=package.numel(),
            num_tensors=len(package),
        )
        self.db.register_model(
            artifact_id=record.artifact_id,
            family=package.family,
            spec={**definition.spec, "graph": definition.graph.to_dict()},
        )
        return {
            "artifact_id": record.artifact_id,
            "architecture": definition.summary(),
            "forward_outputs": {key: list(value.shape) for key, value in result.outputs.items()},
            "verification": verification.to_dict(),
        }

    def edit_architecture(
        self,
        artifact_id: str,
        operation: str,
        *,
        node_id: str | None = None,
        keep_layers: int | None = None,
        new_op: str | None = None,
        params: Mapping[str, Any] | None = None,
        port: str = "a",
    ) -> dict[str, Any]:
        """Apply NASE graph surgery to a stored architecture and persist the result.

        Bypass and truncation preserve the surviving tensors exactly; substitution is
        persisted only when the replacement operator binds the existing parameter
        names. A structure that would need new random weights is refused rather than
        silently initialising them.
        """
        definition, package, _manifest = self._definition_from_artifact(artifact_id)
        if operation == "bypass":
            if not node_id:
                raise InvalidOperationParameters("bypass requires node_id")
            report = self.nase.bypass(definition, node_id, port=port)
        elif operation == "truncate":
            if keep_layers is None:
                raise InvalidOperationParameters("truncate requires keep_layers")
            report = self.nase.truncate_layers(definition, int(keep_layers))
        elif operation == "substitute":
            if not node_id or not new_op:
                raise InvalidOperationParameters("substitute requires node_id and new_op")
            report = self.nase.substitute(definition, node_id, new_op, params)
        else:
            raise InvalidOperationParameters(
                f"unknown architecture operation '{operation}'; use bypass|truncate|substitute"
            )
        definition = report.definition
        definition.graph.require_valid()
        surviving = package.without(report.invalidated_parameters)
        validation = definition.graph.validate(surviving)
        if not validation.ok:
            raise GraphValidationError(
                "architecture edit does not bind the surviving parameters; regenerate "
                "weights instead of silently replacing them",
                errors=validation.errors[:10],
                warnings=report.warnings,
            )
        persisted = self._persist_architecture(
            definition,
            surviving,
            operation=f"architecture_{operation}",
            parents=[artifact_id],
            note=f"NASE {operation}",
            extra={"changes": report.changes, "warnings": report.warnings},
        )
        return {
            **persisted,
            "changes": report.changes,
            "invalidated_parameters": report.invalidated_parameters,
            "warnings": report.warnings,
        }

    def compose_architectures(
        self,
        artifact_ids: Sequence[str],
        *,
        mode: str = "sequential",
        name: str | None = None,
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        if len(artifact_ids) < 2:
            raise InvalidOperationParameters("composition requires at least two artifacts")
        definitions = []
        for artifact_id in artifact_ids:
            definition, _package, _manifest = self._definition_from_artifact(artifact_id)
            definitions.append(definition)
        composed = self.nase.compose(definitions, mode=mode, name=name)
        package = composed.build_package(seed=seed, name=name or "composed")
        composed.graph.require_valid(package)
        persisted = self._persist_architecture(
            composed,
            package,
            operation="architecture_compose",
            parents=list(artifact_ids),
            note=f"NASE {mode} composition",
            extra={"composition": mode},
        )
        return {
            **persisted,
            "components": list(artifact_ids),
            "composition": mode,
            "notes": [
                "component parameters are not transplanted by composition; the new "
                "graph is initialised with labelled structured random weights",
                "the source artifacts remain unchanged and appear as lineage parents",
            ],
        }

    def _execute_architecture(self, job_id: str, request: JobRequest) -> dict[str, Any]:
        options = request.options
        if "family" in options:
            spec = {k: v for k, v in options.items() if k != "family"}
            definition = self.nase.synthesize({"family": options["family"], **spec})
            package = definition.build_package(seed=request.seed)
            definition.graph.require_valid(package)
            inputs = definition.sample_inputs(seed=0)
            result = execute(definition.graph, inputs, package)
            verification = verify(
                package,
                graph=definition.graph,
                inputs=inputs,
                policy_records=[],
                operations=[],
            )
            record = self.store.put(
                package,
                operation="architecture_synthesis",
                reuse_existing=True,
                metadata={
                    "job_id": job_id,
                    "architecture": definition.summary(),
                    "graph": definition.graph.to_dict(),
                },
                validation=verification.to_dict(),
                job_id=job_id,
            )
            self.db.index_artifact(
                artifact_id=record.artifact_id,
                path=str(record.path),
                name=package.name,
                family=package.family,
                operation="architecture_synthesis",
                numel=package.numel(),
                num_tensors=len(package),
                job_id=job_id,
            )
            self.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={**definition.spec, "graph": definition.graph.to_dict()},
            )
            return {
                "artifact_id": record.artifact_id,
                "architecture": definition.summary(),
                "forward_outputs": {k: list(v.shape) for k, v in result.outputs.items()},
                "verification": verification.to_dict(),
                "notes": [
                    "weights are structured random initializations with per-tensor origin labels",
                    "the NIR graph is stored in the artifact metadata for reconstruction",
                ],
            }
        if "research" in options or request.task_scope.value in ("model_architecture_research", "architecture_synthesis"):
            family = str(options.get("research_family", options.get("family", "transformer_decoder")))
            base_spec = dict(options.get("base_spec") or options.get("spec") or {})
            if not base_spec:
                raise InvalidOperationParameters(
                    "architecture research requires options.base_spec (e.g. {d_model, n_heads, n_layers, ...})"
                )
            report = self.mar.explore(
                family=family,
                base_spec=base_spec,
                variants=options.get("variants"),
                seed=request.seed,
                scope=request.task_scope,
            )
            return {
                "research": report,
                "notes": [
                    "architecture research measures cost/shape properties of random-weight "
                    "candidates; it does not establish task quality"
                ],
            }
        raise InvalidOperationParameters(
            "architecture job requires options.family (synthesis) or options.base_spec (research)"
        )

    # ------------------------------------------------------------------ #
    # read APIs
    # ------------------------------------------------------------------ #

    def artifacts(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.artifacts(limit=limit)

    def artifact_manifest(self, artifact_id: str) -> dict[str, Any]:
        return self.store.manifest(artifact_id).to_dict()

    def verify_artifact(self, artifact_id: str) -> dict[str, Any]:
        return self.store.verify(artifact_id)

    def models(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.models(limit=limit)

    def stats(self) -> dict[str, Any]:
        return {"store": self.store.stats(), "database": self.db.stats()}
