"""Advanced end-to-end UZEN release pipeline.

This is the full engineering loop, executed on real arrays:

1. generate a tiny transformer decoder;
2. derive two task checkpoints from it through real weight transformations;
3. compile and execute a WPL release program (TIES consensus + scaling);
4. verify the release against the source architecture;
5. persist an int8 execution variant and measure logit drift;
6. export safetensors and GGUF, then read the GGUF payload back;
7. align and regression-test an MLP parameter pair.

Nothing here is a fixture: every tensor comes from the operation registry, every
verification runs a forward pass, and the command exits non-zero if a gate fails.
The report is written to ``<root>/reports/quickstart.json``.

Run with::

    python examples/quickstart.py ./uzen-demo
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from uzen import Uzen
from uzen.core.npa.aligner import ParameterAligner
from uzen.core.nws.families import build_architecture
from uzen.core.vel.verify import verify
from uzen.core.wpl.compiler import describe_program, run_program
from uzen.infrastructure.storage.gguf_io import read_gguf
from uzen.policies.strict_plus.effects import TaskScope
from uzen.util.canonical import utc_now_iso

RELEASE_PROGRAM = """
program "release_blend" {
    input base: package
    input tuned_a: package
    input tuned_b: package

    param density: float = 0.6
    param factor: float = 1.0

    step consensus = w.merge.ties(base, tuned_a, tuned_b, density=density)
    step release = w.arithmetic.scale(consensus, factor=factor)

    output weights = release
}
"""

SPEC = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16)
TOKENS = [[1, 2, 3, 4, 5]]


def _store_pipeline_output(
    client: Uzen,
    result,
    *,
    parent_ids: list[str],
    graph: Mapping[str, Any] | None,
    program: Mapping[str, Any],
) -> str:
    package = result.package()
    record = client.service.store.put(
        package,
        operation="wpl.release_blend",
        parents=parent_ids,
        operations=result.operations(),
        rng_seeds={key: int(value) for key, value in result.seeds.items()},
        policy=result.policy,
        metadata={
            "graph": dict(graph) if graph else None,
            "program": dict(program),
        },
        note="WPL release pipeline: TIES consensus over transformed task checkpoints",
    )
    client.service.db.index_artifact(
        artifact_id=record.artifact_id,
        path=str(record.path),
        name=package.name,
        family=package.family,
        operation="wpl.release_blend",
        numel=package.numel(),
        num_tensors=len(package),
    )
    return record.artifact_id


def _finish(report: dict[str, Any], root: Path, started: float) -> int:
    gates = report["gates"]
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report_path = root / "reports" / "quickstart.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("gates:", json.dumps(gates, sort_keys=True))
    print("report:", report_path)
    print("gate:", "PASS" if all(gates.values()) else "FAIL")
    return 0 if all(gates.values()) else 1


def main(root: str = "./uzen-demo") -> int:
    started = time.perf_counter()
    root_path = Path(root)
    client = Uzen(root=root_path)
    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "root": str(root_path),
        "gates": {},
        "measurements": {},
    }

    print("== 1. generate a transformer decoder ==")
    generated = client.generate(family="transformer_decoder", spec=SPEC, seed=0)
    base_id = generated["artifact_id"]
    base_manifest = client.artifact(base_id)
    graph = base_manifest.get("metadata", {}).get("graph")
    report["measurements"]["base"] = {
        "artifact_id": base_id,
        "num_tensors": generated["num_tensors"],
        "verification_ok": generated["verification"]["ok"],
    }
    print(f"base artifact: {base_id[:12]}… tensors={generated['num_tensors']}")

    print("== 2. derive two task checkpoints (real weight transforms) ==")
    tuned_a = client.modify(base_id, "scale", params={"factor": 1.01})
    tuned_b = client.modify(base_id, "scale", params={"factor": 1.03})
    report["measurements"]["task_checkpoints"] = {
        "tuned_a": tuned_a["artifact_id"],
        "tuned_b": tuned_b["artifact_id"],
        "transform": "w.arithmetic.scale",
    }
    print(f"tuned_a: {tuned_a['artifact_id'][:12]}…  tuned_b: {tuned_b['artifact_id'][:12]}…")

    print("== 3. compile and execute the WPL release program ==")
    program_info = describe_program(RELEASE_PROGRAM, scope=TaskScope.WEIGHT_MODIFICATION)
    pipeline_result, compiled = run_program(
        RELEASE_PROGRAM,
        {
            "base": client.package(base_id),
            "tuned_a": client.package(tuned_a["artifact_id"]),
            "tuned_b": client.package(tuned_b["artifact_id"]),
        },
        seed=7,
        scope=TaskScope.WEIGHT_MODIFICATION,
    )
    release_id = _store_pipeline_output(
        client,
        pipeline_result,
        parent_ids=[base_id, tuned_a["artifact_id"], tuned_b["artifact_id"]],
        graph=graph,
        program=program_info,
    )
    release_package = client.package(release_id)
    report["measurements"]["release"] = {
        "artifact_id": release_id,
        "program": program_info["program"],
        "stages": [stage["code"] for stage in program_info["steps"]],
        "num_tensors": len(release_package),
        "numel": release_package.numel(),
        "compiled": compiled.report.to_dict(),
        "operations": pipeline_result.operations(),
    }
    print(f"release artifact: {release_id[:12]}… stages={len(program_info['steps'])}")

    print("== 4. verify the release against the source architecture ==")
    evaluation = client.evaluate(
        release_id, reference_artifact_id=base_id, family="transformer_decoder", spec=SPEC
    )
    regression = evaluation.get("regression", {})
    report["measurements"]["evaluation"] = {
        "ok": evaluation["ok"],
        "architecture_unchanged": regression.get("architecture_unchanged"),
        "worst_relative_l2_diff": regression.get("worst_relative_l2_diff"),
    }
    print(
        "release verification:", evaluation["ok"],
        "| worst relative L2:", regression.get("worst_relative_l2_diff"),
    )

    print("== 5. persist an int8 variant and measure logit drift ==")
    quantized = client.quantize_model(release_id, scheme="int8")
    comparison = client.compare_quantized_model(
        release_id, quantized["artifact_id"], tokens=TOKENS
    )
    drift = comparison.get("drift", {})
    report["measurements"]["quantization"] = {
        "artifact_id": quantized["artifact_id"],
        "num_rewritten_nodes": quantized["quantization"]["num_rewritten_nodes"],
        "scheme": quantized["quantization"]["scheme"],
        "drift": drift,
    }
    print(
        "int8 rewritten nodes:", quantized["quantization"]["num_rewritten_nodes"],
        "| max_abs_error:", drift.get("max_abs_error"),
    )

    print("== 6. export safetensors + GGUF and verify the GGUF payload ==")
    exports = root_path / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    safetensors_path = client.service.store.export(
        release_id, exports / "release.safetensors", fmt="safetensors"
    )
    gguf_path = client.service.store.export(
        release_id, exports / "release.gguf", fmt="gguf"
    )
    gguf_arrays, gguf_metadata = read_gguf(gguf_path)
    gguf_exact = set(gguf_arrays) == set(release_package.names()) and all(
        gguf_arrays[name].shape == release_package.tensors[name].shape
        and bool((gguf_arrays[name] == release_package.tensors[name]).all())
        for name in gguf_arrays
    )
    report["measurements"]["exports"] = {
        "safetensors": {"path": str(safetensors_path), "bytes": safetensors_path.stat().st_size},
        "gguf": {
            "path": str(gguf_path),
            "bytes": gguf_path.stat().st_size,
            "tensors": len(gguf_arrays),
            "architecture": gguf_metadata.get("general.architecture"),
            "exact_match": gguf_exact,
        },
    }
    print(
        f"safetensors: {safetensors_path.name} ({safetensors_path.stat().st_size:,} bytes)"
        f" | gguf: {gguf_path.name} ({gguf_path.stat().st_size:,} bytes) exact={gguf_exact}"
    )

    print("== 7. align and regression-test an MLP parameter pair ==")
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16], output_dim=4))
    mlp_base = definition.build_package(seed=1)
    mlp_variant = mlp_base.with_tensor(
        "head.weight", mlp_base.tensors["head.weight"] * 0.9
    )
    alignment = ParameterAligner(method="procrustes", threshold=0.5).align(
        mlp_base, mlp_variant
    )
    mlp_inputs = definition.sample_inputs(batch=2, seed=0)
    mlp_verification = verify(
        mlp_variant, reference=mlp_base, graph=definition.graph, inputs=mlp_inputs
    )
    report["measurements"]["alignment"] = {
        "ok": alignment.ok,
        "score": alignment.score,
        "worst_relative_l2_diff": mlp_verification.regression.get("worst_relative_l2_diff"),
    }
    print(
        "alignment ok:", alignment.ok,
        "| verification ok:", mlp_verification.ok,
    )

    drift_values = [
        drift.get("max_abs_error"),
        drift.get("mean_abs_error"),
        drift.get("max_rel_error"),
    ]
    gates = {
        "generation_validated": bool(generated["verification"]["ok"]),
        "wpl_compiled": bool(compiled.report.ok),
        "release_verified": bool(evaluation["ok"]),
        "quantization_rewrote_linear_nodes": int(
            quantized["quantization"]["num_rewritten_nodes"]
        ) > 0,
        "int8_drift_is_finite": all(
            value is not None and math.isfinite(float(value)) for value in drift_values
        ),
        "exports_readable": safetensors_path.is_file()
        and gguf_path.is_file()
        and gguf_exact,
        "alignment_valid": bool(alignment.ok) and bool(mlp_verification.ok),
    }
    report["gates"] = gates
    return _finish(report, root_path, started)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "./uzen-demo"))
