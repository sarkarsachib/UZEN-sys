"""Compiled native execution-plan demonstration (phase N2).

This builds the checked-in C++ kernels, compiles NIR graphs into native execution
plans, and then:

1. runs an MLP, a transformer decoder, a CNN, an LSTM, a top-k MoE, a state-space
   model and a GCN through the NumPy reference executor and through the compiled
   native plan, comparing every output;
2. reports the fusions the compiler found (linear+bias+activation, residual
   add+norm), the prepacked weights and the buffer plan (peak bytes and slot
   reuse);
3. shows the compiled plan covering graphs the direct per-node native executor
   refuses (conv2d/recurrent/MoE/SSM/GCN);
4. executes an int8 ``quantized_linear`` graph through the compiled plan;
5. measures steady-state latency of the NumPy executor, the direct native
   executor and the compiled plan on one fixed transformer;
6. proves the compiler refuses stochastic operators and refuses to silently
   drop a captured value that fusion removed.

Weights are structured random initializations, labelled as such. Nothing here is
trained. The latency numbers are what this machine measured; no speedup is
claimed by the gate.

Run with::

    python examples/native_compiler_demo.py --root /tmp/uzen-compiler-demo
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.core.wie.quantization import quantize_int8
from uzen.errors import InvalidOperationParameters, UnsupportedOperator
from uzen.runtimes.native import (
    NativeLibrary,
    build_native_library,
    compile_native_plan,
    execute_native,
    execute_plan,
)

BENCH_SPEC = dict(
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=128,
    vocab_size=128,
    max_seq_len=32,
)

FAMILIES: list[tuple[str, dict[str, Any]]] = [
    ("mlp", dict(input_dim=32, hidden_dims=[64, 48], output_dim=16)),
    ("transformer_decoder", dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16)),
    (
        "cnn",
        dict(
            in_channels=3,
            height=16,
            width=16,
            channels=[8, 12],
            kernel=3,
            padding=1,
            num_classes=5,
            pool=2,
        ),
    ),
    ("recurrent", dict(vocab_size=32, d_model=16, hidden_size=12, cell="lstm", num_classes=4)),
    ("moe", dict(vocab_size=32, d_model=16, n_experts=4, top_k=2, d_ff=24, num_classes=3)),
    ("state_space", dict(vocab_size=32, d_model=16, state_dim=4, num_classes=3)),
    ("gnn", dict(in_features=6, hidden_dim=10, n_layers=2, nodes=5, num_classes=3)),
]


def _timed(function: Callable[[], Any], repeats: int = 5) -> float:
    function()  # warm-up
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def _max_error(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for name, reference in expected.items():
        difference = np.abs(
            actual[name].astype(np.float64) - reference.astype(np.float64)
        )
        worst = max(worst, float(difference.max()))
    return worst


def main(root: str = "/tmp/uzen-compiler-demo") -> int:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "N2",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "families": {},
        "gates": {},
    }

    library_path = build_native_library(directory / "libuzen_native.so", force=True)
    library = NativeLibrary(library_path)
    report["library"] = library.describe()

    # ------------------------------------------------------------------ #
    # 1. parity of compiled plans across families
    # ------------------------------------------------------------------ #
    print("== compiled-plan parity (NumPy reference vs native plan) ==")
    for family, spec in FAMILIES:
        definition = build_architecture(family, spec)
        package = definition.build_package(seed=11)
        inputs = definition.sample_inputs(batch=2, seed=0)
        reference = numpy_execute(definition.graph, inputs, package)
        plan = compile_native_plan(definition.graph, package)
        execution = execute_plan(plan, inputs, package, library=library)
        error = _max_error(execution.outputs, reference.outputs)
        direct: dict[str, Any] = {"supported": True}
        try:
            direct_outputs = execute_native(
                definition.graph, inputs, package, library=library
            ).outputs
            direct["max_abs_error"] = _max_error(direct_outputs, reference.outputs)
        except UnsupportedOperator as exc:
            direct = {"supported": False, "reason": str(exc).split(";")[0][:120]}
        entry = {
            "nodes": len(definition.graph.nodes),
            "plan_steps": execution.steps,
            "fused_steps": execution.fused_steps,
            "max_abs_error": error,
            "operators": plan.describe()["operators"],
            "memory": {
                key: execution.memory[key]
                for key in ("buffers", "slots", "reused_buffers", "peak_bytes", "allocated_bytes")
            },
            "direct_executor": direct,
        }
        report["families"][family] = entry
        print(
            f"  {family:20s} nodes={entry['nodes']:3d} steps={entry['plan_steps']:3d} "
            f"fused={entry['fused_steps']:2d} max_abs_error={error:.3e} "
            f"peak={entry['memory']['peak_bytes'] / 1024:.1f} KiB "
            f"reuse={entry['memory']['reused_buffers']:2d} "
            f"direct={'yes' if direct['supported'] else 'refused'}"
        )

    report["gates"]["plan_parity"] = all(
        entry["max_abs_error"] < 1e-5 for entry in report["families"].values()
    )

    # ------------------------------------------------------------------ #
    # 2. fusions, prepacking and the buffer plan
    # ------------------------------------------------------------------ #
    print("\n== compilation report (transformer decoder) ==")
    definition = build_architecture("transformer_decoder", dict(d_model=64, n_heads=4, n_layers=2, d_ff=128, vocab_size=128, max_seq_len=32))
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=4, seq=16, seed=0)
    plan = compile_native_plan(definition.graph, package)
    execution = execute_plan(plan, inputs, package, library=library)
    reference = numpy_execute(definition.graph, inputs, package)
    described = plan.describe()
    packed = {
        name: weight.describe() for name, weight in plan.packed_weights.items()
    }
    report["compilation"] = {
        "nodes": len(definition.graph.nodes),
        "steps": described["steps"],
        "fusions": described["fusions"],
        "operators": described["operators"],
        "prepacked": packed,
        "memory": execution.memory,
        "max_abs_error": _max_error(execution.outputs, reference.outputs),
    }
    print(
        f"  nodes={described['steps']} plan steps, "
        f"{len(described['fusions'])} fusions, {len(packed)} prepacked weights"
    )
    for fusion in described["fusions"]:
        print(f"    fused {fusion['kind']}: {fusion['nodes']} -> step {fusion['step']}")
    print(
        f"  memory: {execution.memory['buffers']} buffers in "
        f"{execution.memory['slots']} slots, peak "
        f"{execution.memory['peak_bytes'] / 1024:.1f} KiB of "
        f"{execution.memory['allocated_bytes'] / 1024:.1f} KiB allocated, "
        f"{execution.memory['reused_buffers']} reused"
    )
    overheads = [value["padding_overhead"] for value in packed.values()]
    report["gates"]["fusions_applied"] = len(described["fusions"]) >= 4
    report["gates"]["memory_reuse"] = execution.memory["reused_buffers"] > 0
    report["gates"]["weights_prepacked"] = len(packed) > 0
    print(f"  prepack padding overhead: min={min(overheads):.2f}x max={max(overheads):.2f}x")

    # ------------------------------------------------------------------ #
    # 3. wider coverage than the direct executor
    # ------------------------------------------------------------------ #
    print("\n== coverage: compiled plan vs direct per-node executor ==")
    wider = [
        family
        for family, entry in report["families"].items()
        if not entry["direct_executor"]["supported"]
    ]
    report["coverage"] = {"direct_executor_refused": wider}
    report["gates"]["wider_coverage"] = bool(wider)
    for family in wider:
        print(f"  {family}: direct executor refused, compiled plan executed it")

    # ------------------------------------------------------------------ #
    # 4. int8 quantized linear through a compiled plan
    # ------------------------------------------------------------------ #
    print("\n== int8 quantized_linear in a compiled plan ==")
    quantized = quantize_int8(definition.graph, package)
    quantized_inputs = definition.sample_inputs(batch=2, seq=8, seed=1)
    quantized_reference = numpy_execute(
        quantized.graph, quantized_inputs, quantized.package
    )
    quantized_plan = compile_native_plan(quantized.graph, quantized.package)
    quantized_execution = execute_plan(
        quantized_plan, quantized_inputs, quantized.package, library=library
    )
    quantized_error = _max_error(
        quantized_execution.outputs, quantized_reference.outputs
    )
    report["quantized"] = {
        "rewritten_nodes": quantized.report.num_rewritten_nodes,
        "max_abs_error": quantized_error,
        "native_ops": quantized_execution.native_ops,
        "used_quantized_kernel": "quantized_linear" in quantized_execution.native_ops,
    }
    report["gates"]["int8_plan"] = (
        quantized_execution.native_ops.count("quantized_linear") > 0
        and quantized_error < 1e-5
    )
    print(
        f"  quantized nodes={quantized.report.num_rewritten_nodes} "
        f"max_abs_error={quantized_error:.3e} "
        f"native_ops={sorted(set(quantized_execution.native_ops))}"
    )

    # ------------------------------------------------------------------ #
    # 5. throughput comparison on one fixed transformer (measured)
    # ------------------------------------------------------------------ #
    print("\n== steady-state latency (median of 5 runs, no speedup claim) ==")
    bench_definition = build_architecture("transformer_decoder", BENCH_SPEC)
    bench_package = bench_definition.build_package(seed=5)
    bench_inputs = bench_definition.sample_inputs(batch=4, seq=16, seed=0)
    bench_plan = compile_native_plan(bench_definition.graph, bench_package)
    execute_plan(bench_plan, bench_inputs, bench_package, library=library)  # prepack

    numpy_ms = _timed(
        lambda: numpy_execute(bench_definition.graph, bench_inputs, bench_package)
    ) * 1e3
    direct_ms = _timed(
        lambda: execute_native(
            bench_definition.graph, bench_inputs, bench_package, library=library
        )
    ) * 1e3
    compiled_ms = _timed(
        lambda: execute_plan(
            bench_plan, bench_inputs, bench_package, library=library
        )
    ) * 1e3
    report["timing_ms"] = {
        "numpy": numpy_ms,
        "direct_native": direct_ms,
        "compiled_plan": compiled_ms,
        "compiled_vs_direct": compiled_ms / direct_ms if direct_ms else None,
        "compiled_vs_numpy": compiled_ms / numpy_ms if numpy_ms else None,
        "shape": {"batch": 4, "seq": 16, "d_model": 64, "layers": 2},
    }
    print(f"  numpy executor : {numpy_ms:.3f} ms")
    print(f"  direct native  : {direct_ms:.3f} ms")
    print(f"  compiled plan  : {compiled_ms:.3f} ms")
    print(
        f"  compiled/direct = {compiled_ms / direct_ms:.2f}x, "
        f"compiled/numpy = {compiled_ms / numpy_ms:.2f}x"
    )

    # ------------------------------------------------------------------ #
    # 6. refusal behaviour
    # ------------------------------------------------------------------ #
    print("\n== refusals (no silent fallback) ==")
    vae = build_architecture("vae", dict(input_dim=8, latent_dim=4))
    vae_package = vae.build_package(seed=0)
    stochastic_refused = False
    try:
        compile_native_plan(vae.graph, vae_package)
    except UnsupportedOperator as exc:
        stochastic_refused = "sample_normal" in str(exc)
        print(f"  stochastic graph refused: {str(exc).split(';')[0]}")

    fused_linear = next(
        fusion["nodes"][0]
        for fusion in plan.fusions
        if fusion["kind"] == "linear_activation"
    )
    capture_refused = False
    try:
        execute_plan(
            plan,
            inputs,
            package,
            capture=[fused_linear],
            library=library,
        )
    except InvalidOperationParameters as exc:
        capture_refused = "fused" in str(exc)
        print(f"  fused-value capture refused: {exc}")

    report["gates"]["refuses_stochastic_ops"] = stochastic_refused
    report["gates"]["refuses_fused_capture"] = capture_refused

    # ------------------------------------------------------------------ #
    # gate
    # ------------------------------------------------------------------ #
    print("\n== gates ==")
    print(json.dumps(report["gates"], sort_keys=True))
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    passed = all(report["gates"].values())
    print("gate:", "PASS" if passed else "FAIL")
    print("report:", directory / "report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/uzen-compiler-demo")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.root))
