"""Python SDK for UZEN.

Two transports, one surface:

* ``Uzen.local(root)`` — runs everything in-process against a job service and artifact
  store (no HTTP server required).
* ``Uzen.remote(url, token)`` — talks to a running UZEN REST control plane.

The SDK never fabricates results: local calls execute the real engines, and remote
calls surface the server's typed error payloads as exceptions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from uzen.core.nws.tensors import WeightPackage
from uzen.core.wie.model_execution import ModelExecution, ModelGeneration, ModelTextExecution
from uzen.errors import UzenError
from uzen.jobs.service import JobService


class UzenClientError(UzenError):
    code = "sdk.error"


@dataclass
class _RemoteConfig:
    base_url: str
    token: str | None = None
    timeout: float = 120.0


def _execution_payload(execution: ModelExecution, *, include_values: bool) -> dict[str, Any]:
    payload = execution.summary()
    if include_values:
        payload["outputs"] = {
            name: {
                **payload["outputs"][name],
                "values": execution.outputs[name].tolist(),
            }
            for name in execution.outputs
        }
        payload["captured"] = {
            name: {
                **payload["captured"][name],
                "values": execution.intermediates[name].tolist(),
            }
            for name in execution.intermediates
        }
    return payload


def _text_execution_payload(result: ModelTextExecution, *, include_values: bool) -> dict[str, Any]:
    payload = _execution_payload(result.execution, include_values=include_values)
    payload.update(
        {
            "text": result.text,
            "decoded_text": result.decoded_text,
            "token_ids": [list(result.token_ids)],
            "tokens": [list(result.tokens)],
        }
    )
    return payload


class Uzen:
    """Unified SDK entry point."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if (root is None) == (base_url is None):
            raise UzenClientError(
                "provide exactly one of root= (local engine) or base_url= (remote control plane)"
            )
        self._service = JobService(root) if root is not None else None
        self._remote = _RemoteConfig(base_url.rstrip("/"), token, timeout) if base_url else None

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #

    @property
    def local(self) -> bool:
        return self._service is not None

    @property
    def service(self) -> JobService:
        if self._service is None:
            raise UzenClientError("this client is remote; use job endpoints instead")
        return self._service

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        assert self._remote is not None
        url = f"{self._remote.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self._remote.token:
            req.add_header("Authorization", f"Bearer {self._remote.token}")
        try:
            with urllib.request.urlopen(req, timeout=self._remote.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(payload)
                detail = parsed.get("detail") or parsed
            except json.JSONDecodeError:
                detail = {"message": payload[:500]}
            raise UzenClientError(
                f"control plane returned HTTP {exc.code}", detail=detail, path=path
            ) from exc
        except urllib.error.URLError as exc:
            raise UzenClientError(
                f"cannot reach control plane at {self._remote.base_url}: {exc.reason}", path=path
            ) from exc

    # ------------------------------------------------------------------ #
    # catalogue
    # ------------------------------------------------------------------ #

    def operations(self) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.nwpc.registry import default_registry

            return default_registry().catalogue()
        return self._request("GET", "/v1/operations")["operations"]

    def policy(self) -> dict[str, Any]:
        if self.local:
            return self.service.engine.describe()
        return self._request("GET", "/v1/policies/strict-plus")

    def families(self) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.nws.families import family_catalogue

            return family_catalogue()
        return self._request("GET", "/v1/architectures/families")["families"]

    # ------------------------------------------------------------------ #
    # artifacts
    # ------------------------------------------------------------------ #

    def import_weights(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
        revision: str | None = None,
        repo_id: str | None = None,
    ) -> dict[str, Any]:
        if self.local:
            p = Path(path)
            if repo_id or p.is_dir() or p.suffix.lower() == ".safetensors":
                return self.import_model(
                    path, name=name, family=family, revision=revision, repo_id=repo_id
                )
            record = self.service.import_artifact(path, name=name, family=family)
            result = {
                "artifact_id": record.artifact_id,
                "name": record.manifest.name,
                "family": record.manifest.family,
                "num_tensors": record.num_tensors,
                "checksum": record.manifest.checksum,
            }
            import_report = record.manifest.metadata.get("import_report")
            if isinstance(import_report, Mapping):
                result["format"] = import_report.get("format")
                result["import_report"] = dict(import_report)
            return result
        return self._request(
            "POST",
            "/v1/models/import",
            {
                "path": str(path),
                "name": name,
                "family": family,
                "revision": revision,
                "repo_id": repo_id,
            },
        )

    def import_model(
        self,
        source: str | Path,
        *,
        name: str | None = None,
        family: str | None = None,
        revision: str | None = None,
        repo_id: str | None = None,
        on_progress: Any | None = None,
    ) -> dict[str, Any]:
        """Import an HF directory/file/repository and return the compatibility report."""
        if self.local:
            record, report = self.service.import_model(
                source,
                name=name,
                family=family,
                revision=revision,
                repo_id=repo_id,
                on_progress=on_progress,
            )
            return {
                "artifact_id": record.artifact_id,
                "name": record.manifest.name,
                "family": record.manifest.family,
                "num_tensors": record.num_tensors,
                "checksum": record.manifest.checksum,
                "compatibility": report["compatibility"],
                "report": report,
            }
        return self._request(
            "POST",
            "/v1/models/import",
            {
                "source": str(source),
                "name": name,
                "family": family,
                "revision": revision,
                "repo_id": repo_id,
            },
        )

    def inspect(self, source: str | Path) -> dict[str, Any]:
        if self.local:
            return self.service.wie.inspect_file(source).to_dict()
        return self._request("POST", "/v1/weights/analyze", {"path": str(source)})

    def analyze(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            package = self.service.get_package(artifact_id)
            report = self.service.wie.inspect_package(package)
            return {"artifact_id": artifact_id, **report.to_dict()}
        return self._request("POST", "/v1/weights/analyze", {"artifact_id": artifact_id})

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.artifact_manifest(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}")

    def verify_artifact(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.verify_artifact(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}/verify")

    def lineage(self, artifact_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.store.lineage_graph().describe(artifact_id)
        return self._request("GET", f"/v1/artifacts/{artifact_id}/lineage")

    def package(self, artifact_id: str) -> WeightPackage:
        if not self.local:
            raise UzenClientError(
                "raw packages are only available in local mode; use the REST/MCP interfaces "
                "to obtain artifacts as files"
            )
        return self.service.get_package(artifact_id)

    def quantize_model(
        self,
        artifact_id: str,
        *,
        scheme: str = "int8",
    ) -> dict[str, Any]:
        """Persist and return a quantized execution variant of a stored model."""
        if not self.local:
            return self._request(
                "POST",
                f"/v1/models/{artifact_id}/quantize",
                {"scheme": scheme},
            )
        return self.service.quantize_model(artifact_id, scheme=scheme)

    def forward_model(
        self,
        artifact_id: str,
        *,
        tokens: Sequence[Sequence[int]],
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        include_values: bool = False,
        capture: Sequence[str] | str | None = None,
        quantize: str | None = None,
    ) -> dict[str, Any]:
        """Run a stored reconstructed model graph on token IDs.

        ``quantize="int8"`` executes a weight-only int8 copy and includes its
        quantization report under the response's ``quantization`` key.
        """
        if not self.local:
            return self._request(
                "POST",
                f"/v1/models/{artifact_id}/forward",
                {
                    "tokens": [list(row) for row in tokens],
                    "positions": positions,
                    "include_values": include_values,
                    "capture": capture,
                    "quantize": quantize,
                },
            )
        execution = self.service.forward_model(
            artifact_id,
            tokens=tokens,
            positions=positions,
            capture=capture,
            quantize=quantize,
        )
        return _execution_payload(execution, include_values=include_values)

    def forward_text(
        self,
        artifact_id: str,
        *,
        text: str,
        positions: Sequence[Sequence[int]] | Sequence[int] | None = None,
        include_values: bool = False,
        capture: Sequence[str] | str | None = None,
        quantize: str | None = None,
    ) -> dict[str, Any]:
        """Tokenize text with the artifact's stored tokenizer and run its graph."""
        if not self.local:
            return self._request(
                "POST",
                f"/v1/models/{artifact_id}/forward-text",
                {
                    "text": text,
                    "positions": positions,
                    "include_values": include_values,
                    "capture": capture,
                    "quantize": quantize,
                },
            )
        result = self.service.forward_text(
            artifact_id,
            text=text,
            positions=positions,
            capture=capture,
            quantize=quantize,
        )
        return _text_execution_payload(result, include_values=include_values)

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
    ) -> dict[str, Any]:
        """Generate text with the artifact's stored tokenizer and graph.

        ``quantize="int8"`` quantizes the execution copy once and includes its
        report under the response's ``quantization`` key.
        """
        if not self.local:
            return self._request(
                "POST",
                f"/v1/models/{artifact_id}/generate-text",
                {
                    "text": text,
                    "max_new_tokens": max_new_tokens,
                    "eos_token_id": eos_token_id,
                    "positions": positions,
                    "strategy": strategy,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "seed": seed,
                    "num_beams": num_beams,
                    "quantize": quantize,
                },
            )
        result: ModelGeneration = self.service.generate_text(
            artifact_id,
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
        return result.summary()

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
        """Compare a source model with its persisted int8 variant."""
        if not self.local:
            return self._request(
                "POST",
                "/v1/models/compare-quantized",
                {
                    "source_artifact_id": source_artifact_id,
                    "quantized_artifact_id": quantized_artifact_id,
                    "tokens": [list(row) for row in tokens],
                    "positions": positions,
                    "atol": atol,
                    "rtol": rtol,
                    "max_abs_error_threshold": max_abs_error_threshold,
                    "mean_abs_error_threshold": mean_abs_error_threshold,
                    "max_rel_error_threshold": max_rel_error_threshold,
                },
            )
        return self.service.compare_quantized_model(
            source_artifact_id,
            quantized_artifact_id,
            tokens=tokens,
            positions=positions,
            atol=atol,
            rtol=rtol,
            max_abs_error_threshold=max_abs_error_threshold,
            mean_abs_error_threshold=mean_abs_error_threshold,
            max_rel_error_threshold=max_rel_error_threshold,
        )

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
        """Compare source and modified model artifacts on identical inputs."""
        if not self.local:
            return self._request(
                "POST",
                "/v1/models/compare",
                {
                    "source_artifact_id": source_artifact_id,
                    "candidate_artifact_id": candidate_artifact_id,
                    "tokens": [list(row) for row in tokens],
                    "positions": positions,
                    "atol": atol,
                    "rtol": rtol,
                    "max_abs_error_threshold": max_abs_error_threshold,
                    "mean_abs_error_threshold": mean_abs_error_threshold,
                    "max_rel_error_threshold": max_rel_error_threshold,
                },
            )
        return self.service.compare_models(
            source_artifact_id,
            candidate_artifact_id,
            tokens=tokens,
            positions=positions,
            atol=atol,
            rtol=rtol,
            max_abs_error_threshold=max_abs_error_threshold,
            mean_abs_error_threshold=mean_abs_error_threshold,
            max_rel_error_threshold=max_rel_error_threshold,
        )

    # ------------------------------------------------------------------ #
    # direct engineering calls (local mode executes immediately)
    # ------------------------------------------------------------------ #

    def modify(
        self,
        artifact_id: str,
        method: str,
        *,
        params: Mapping[str, Any] | None = None,
        others: Sequence[str] = (),
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/modify",
                {"artifact_id": artifact_id, "method": method, "params": dict(params or {}), "others": list(others), "seed": seed},
            )
        from uzen.policies.strict_plus.effects import TaskScope

        service = self.service
        primary = service.get_package(artifact_id)
        other_pkgs = [service.get_package(a) for a in others]
        result = service.wme.modify(
            primary,
            method,
            params=params or {},
            others=other_pkgs,
            seed=seed,
            scope=TaskScope.WEIGHT_MODIFICATION,
        )
        parent_metadata = service.artifact_manifest(artifact_id).get("metadata") or {}
        metadata: dict[str, Any] = {
            "method": method,
            "architecture_preserved": result.architecture_preserved,
        }
        for key in ("graph", "model_kind", "source_path", "tokenizer"):
            if key in parent_metadata:
                metadata[key] = parent_metadata[key]
        parents = list(dict.fromkeys([artifact_id, *others]))
        record = service.store.put(
            result.package,
            operation=result.record.code,
            parents=parents,
            operations=[result.record.to_dict()],
            rng_seeds={"main": result.record.seed},
            policy=result.record.policy,
            metadata=metadata,
        )
        service.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=result.package.name,
            family=result.package.family,
            operation=result.record.code,
            numel=result.package.numel(),
            num_tensors=len(result.package),
        )
        return {**result.to_dict(), "artifact_id": record.artifact_id}

    def generate(
        self,
        *,
        family: str | None = None,
        spec: Mapping[str, Any] | None = None,
        template_artifact_id: str | None = None,
        method: str = "orthogonal",
        params: Mapping[str, Any] | None = None,
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/generate",
                {
                    "family": family,
                    "spec": dict(spec or {}),
                    "template_artifact_id": template_artifact_id,
                    "method": method,
                    "params": dict(params or {}),
                    "seed": seed,
                },
            )
        from uzen.core.vel.verify import verify
        from uzen.policies.strict_plus.effects import TaskScope

        service = self.service
        if family:
            result = service.wge.from_architecture(family, spec or {}, seed=seed)
            package = result.package
            operation = f"generate:{family}"
            definition = result.definition
        else:
            if not template_artifact_id:
                raise UzenClientError(
                    "generation needs either family= or template_artifact_id="
                )
            template = service.get_package(template_artifact_id)
            result = service.wge.generate(
                template, method=method, params=params or {}, seed=seed, scope=TaskScope.WEIGHT_GENERATION
            )
            package = result.package
            operation = result.record.code
            definition = None
        record = service.store.put(
            package,
            operation=operation,
            parents=[template_artifact_id] if template_artifact_id else [],
            operations=[result.record.to_dict()] if result.record else [],
            rng_seeds={"main": result.seed},
            metadata={
                "method": method,
                "architecture": definition.summary() if definition else None,
                "graph": definition.graph.to_dict() if definition else None,
            },
        )
        service.db.index_artifact(
            artifact_id=record.artifact_id,
            path=str(record.path),
            name=package.name,
            family=package.family,
            operation=operation,
            numel=package.numel(),
            num_tensors=len(package),
        )
        if definition is not None:
            service.db.register_model(
                artifact_id=record.artifact_id,
                family=package.family,
                spec={**definition.spec, "graph": definition.graph.to_dict()},
            )
            # Generation is only complete with evidence: bind the freshly built parameters
            # to the graph and run a real forward pass plus structural checks.
            evidence = verify(
                package,
                graph=definition.graph,
                inputs=definition.sample_inputs(seed=0),
            )
        else:
            evidence = verify(package, measure_performance=False)
        return {
            **result.to_dict(),
            "artifact_id": record.artifact_id,
            "verification": evidence.to_dict(),
        }

    def evaluate(
        self,
        artifact_id: str,
        *,
        reference_artifact_id: str | None = None,
        family: str | None = None,
        spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.local:
            return self._request(
                "POST",
                "/v1/weights/evaluate",
                {"artifact_id": artifact_id, "reference_artifact_id": reference_artifact_id,
                 "family": family, "spec": dict(spec or {})},
            )
        from uzen.core.vel.verify import verify

        service = self.service
        package = service.get_package(artifact_id)
        reference = service.get_package(reference_artifact_id) if reference_artifact_id else None
        graph = None
        inputs = None
        if family:
            from uzen.core.nws.families import build_architecture

            definition = build_architecture(family, spec or {})
            if definition.graph.validate(package).ok:
                graph = definition.graph
                inputs = definition.sample_inputs(seed=0)
        report = verify(package, reference=reference, graph=graph, inputs=inputs)
        return report.to_dict()

    def compose_architectures(
        self,
        artifact_ids: Sequence[str],
        *,
        mode: str = "sequential",
        name: str | None = None,
        seed: int | str | None = 0,
    ) -> dict[str, Any]:
        """Compose stored architectures and persist the validated result."""
        if not self.local:
            return self._request(
                "POST",
                "/v1/architectures/compose",
                {
                    "artifact_ids": list(artifact_ids),
                    "mode": mode,
                    "name": name,
                    "seed": seed,
                },
            )
        return self.service.compose_architectures(
            list(artifact_ids), mode=mode, name=name, seed=seed
        )

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
        """Bypass, truncate or substitute a stored architecture and persist it."""
        if not self.local:
            return self._request(
                "POST",
                f"/v1/architectures/{artifact_id}/edit",
                {
                    "operation": operation,
                    "node_id": node_id,
                    "keep_layers": keep_layers,
                    "new_op": new_op,
                    "params": dict(params or {}),
                    "port": port,
                },
            )
        return self.service.edit_architecture(
            artifact_id,
            operation,
            node_id=node_id,
            keep_layers=keep_layers,
            new_op=new_op,
            params=params,
            port=port,
        )

    # ------------------------------------------------------------------ #
    # research campaigns (N6/N7/N8)
    # ------------------------------------------------------------------ #

    def run_campaign(
        self,
        spec: Mapping[str, Any],
        *,
        resume: bool = True,
        max_rounds: int | None = None,
        distributed: bool = False,
        trial_timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 0.1,
        background: bool = False,
    ) -> dict[str, Any]:
        """Run, resume or enqueue a deterministic architecture/weight campaign.

        ``distributed=True`` submits each trial to the N4 control-plane queue and
        waits for workers; it never silently falls back to local evaluation.
        ``background=True`` returns a durable campaign-run job instead of blocking;
        a ``CampaignWorker`` restricted to ``campaign_run`` must be running.
        """
        if not self.local:
            return self._request(
                "POST",
                "/v1/campaigns",
                {
                    "spec": dict(spec),
                    "resume": resume,
                    "max_rounds": max_rounds,
                    "distributed": distributed,
                    "trial_timeout_seconds": trial_timeout_seconds,
                    "poll_interval_seconds": poll_interval_seconds,
                    "background": background,
                },
            )
        from uzen.core.campaign import CampaignEngine, CampaignSpec, DistributedTrialExecutor

        parsed = CampaignSpec.from_dict(spec)
        if background:
            from uzen.core.campaign.control import submit_campaign_run

            return submit_campaign_run(
                self.service,
                parsed,
                distributed=distributed,
                resume=resume,
                max_rounds=max_rounds,
                trial_timeout_seconds=trial_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        executor = None
        if distributed:
            executor = DistributedTrialExecutor(
                self.service,
                timeout_seconds=trial_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
        engine = CampaignEngine(
            self.service.root,
            trial_executor=executor,
            job_service=self.service,
        )
        return engine.run(parsed, resume=resume, max_rounds=max_rounds).to_dict()

    def campaign(self, campaign_id: str) -> dict[str, Any]:
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(self.service.root).get(campaign_id).to_dict()
        return self._request("GET", f"/v1/campaigns/{campaign_id}")

    def campaigns(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(self.service.root).list_campaigns(limit=limit)
        return self._request("GET", f"/v1/campaigns?limit={limit}")["campaigns"]

    def campaign_events(self, campaign_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(self.service.root).events(campaign_id, after=after)
        return self._request(
            "GET", f"/v1/campaigns/{campaign_id}/events?after={after}"
        )["events"]

    def campaign_jobs(self, campaign_id: str) -> list[dict[str, Any]]:
        """N4 jobs linked to a campaign (trial evaluations and run coordinators)."""
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(
                self.service.root, job_service=self.service
            ).jobs(campaign_id)
        return self._request("GET", f"/v1/campaigns/{campaign_id}/jobs")["jobs"]

    def cancel_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Cancel a campaign and its linked queued or leased jobs."""
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(
                self.service.root, job_service=self.service
            ).cancel(campaign_id)
        return self._request("POST", f"/v1/campaigns/{campaign_id}/cancel", {})

    def verify_campaign_corpus(self, digests: Sequence[str] | None = None) -> dict[str, Any]:
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(self.service.root).verify_corpus(digests)
        return self._request(
            "POST",
            "/v1/campaigns/verify",
            {"digests": list(digests) if digests is not None else None},
        )

    def replay_benchmark(self, digest: str) -> dict[str, Any]:
        if self.local:
            from uzen.core.campaign import CampaignEngine

            return CampaignEngine(self.service.root).replay(digest)
        return self._request("GET", f"/v1/benchmarks/{digest}")

    # ------------------------------------------------------------------ #
    # jobs
    # ------------------------------------------------------------------ #

    def submit_job(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        if self.local:
            job = self.service.submit(request, idempotency_key=idempotency_key)
            return job
        return self._request("POST", "/v1/jobs", dict(request))

    def run_job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.run(job_id)
        return self._request("POST", f"/v1/jobs/{job_id}/run", {})

    def job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.get(job_id)
        return self._request("GET", f"/v1/jobs/{job_id}")

    def jobs(self, *, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if self.local:
            return self.service.list(state=state, limit=limit)
        query = f"?limit={limit}" + (f"&state={state}" if state else "")
        return self._request("GET", f"/v1/jobs{query}")["jobs"]

    def job_events(self, job_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        if self.local:
            return self.service.events(job_id, after=after)
        return self._request("GET", f"/v1/jobs/{job_id}/events?after={after}")["events"]

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        if self.local:
            return self.service.cancel(job_id)
        return self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    def run_local(self, request: Mapping[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        """Submit and immediately execute a job (local mode only)."""
        if not self.local:
            raise UzenClientError("run_local is only available in local mode")
        job = self.submit_job(request, idempotency_key=idempotency_key)
        return self.run_job(job["id"])

    def stats(self) -> dict[str, Any]:
        if self.local:
            return self.service.stats()
        return self._request("GET", "/v1/stats")
