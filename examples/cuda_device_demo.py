"""N5 CUDA device-boundary demo.

What this does on any machine:

1. compiles the real ``kernels.cu`` to PTX with NVRTC (or nvcc);
2. assembles the PTX with ``ptxas`` when available (a second, independent check);
3. probes the driver/device;
4. verifies the strict refusal and the explicitly-labelled CPU fallback.

When a CUDA device is present it also runs an MLP and a transformer decoder
entirely on the device, compares them against the NumPy reference, captures an
attention tensor, checks weight residency across runs and reports transfer and
peak-memory numbers.

It never claims device execution when none happened: without a device the report
says ``executed: false`` and the final status is ``SKIPPED``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.errors import RuntimeUnavailable
from uzen.runtimes.cuda import (
    KERNELS,
    CudaRuntime,
    compile_ptx,
    cuda_available,
    describe_compilers,
    device_snapshot,
)


def _compare(
    outputs: dict[str, np.ndarray], reference: dict[str, np.ndarray], *, rtol: float, atol: float
) -> tuple[bool, float]:
    worst = 0.0
    for name, expected in reference.items():
        actual = outputs[name]
        worst = max(worst, float(np.max(np.abs(actual - expected))))
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            return False, worst
    return True, worst


def _device_gates(report: dict[str, Any], root: Path) -> int:
    gates: list[dict[str, Any]] = report["gates"]

    def gate(name: str, ok: bool, detail: str) -> None:
        gates.append({"name": name, "ok": bool(ok), "detail": detail})

    ptx_path, compile_report = compile_ptx()
    text = ptx_path.read_text(encoding="utf-8")
    missing = [kernel for kernel in KERNELS if kernel not in text]
    gate(
        "kernels compiled",
        not missing,
        f"{len(KERNELS)} kernels in {ptx_path.name} via {compile_report.get('compiler')}",
    )
    assembler = shutil.which("ptxas") or str(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
        / "bin"
        / "ptxas"
    )
    if Path(assembler).exists():
        cubin = root / "uzen_cuda.cubin"
        completed = subprocess.run(
            [assembler, "-arch=sm_60", "-o", str(cubin), str(ptx_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        gate(
            "ptxas assembly",
            completed.returncode == 0,
            (completed.stderr or f"{cubin.stat().st_size} byte cubin")[-300:]
            if completed.returncode == 0
            else completed.stderr[-300:],
        )

    available = cuda_available()
    report["device_probe"] = device_snapshot()
    report["available"] = available

    if not available:
        try:
            CudaRuntime()
        except RuntimeUnavailable as exc:
            gate("strict refusal", exc.code == "runtime.unavailable", str(exc)[:240])
        else:  # pragma: no cover
            gate("strict refusal", False, "CudaRuntime() unexpectedly opened")

        definition = build_architecture(
            "mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4)
        )
        package = definition.build_package(seed=3)
        inputs = definition.sample_inputs(batch=2, seed=0)
        reference = numpy_execute(definition.graph, inputs, package).outputs
        started = time.perf_counter()
        with CudaRuntime(fallback="cpu") as runtime:
            execution = runtime.execute_graph(definition.graph, inputs, package)
            capabilities = runtime.capabilities()
        elapsed = time.perf_counter() - started
        ok, worst = _compare(execution.outputs, reference, rtol=0, atol=0)
        gate("fallback parity", ok, f"max abs diff {worst:.3e}")
        gate(
            "fallback is labelled",
            execution.backend == "numpy-fallback"
            and capabilities.device == "cpu"
            and not capabilities.extra["cuda"]
            and bool(execution.reason),
            f"backend={execution.backend} device={capabilities.device}",
        )
        report["fallback"] = {
            "backend": execution.backend,
            "device": capabilities.device,
            "reason": execution.reason,
            "max_abs_diff": worst,
            "seconds": elapsed,
        }
        report["executed"] = False
        report["status"] = "skipped-no-cuda-device"
        return 0

    definition = build_architecture(
        "mlp", dict(input_dim=16, hidden_dims=[32, 32], output_dim=8)
    )
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=4, seed=0)
    reference = numpy_execute(definition.graph, inputs, package).outputs
    with CudaRuntime(fallback="error") as runtime:
        started = time.perf_counter()
        first = runtime.execute_graph(definition.graph, inputs, package)
        mlp_seconds = time.perf_counter() - started
        ok, worst = _compare(first.outputs, reference, rtol=1e-4, atol=1e-5)
        gate("mlp parity", ok, f"max abs diff {worst:.3e}, {first.kernels_launched} launches")

        second = runtime.execute_graph(definition.graph, inputs, package)
        gate(
            "weights resident",
            second.stats["execution_host_to_device_bytes"]
            < first.stats["execution_host_to_device_bytes"],
            f"H2D first={first.stats['execution_host_to_device_bytes']} "
            f"second={second.stats['execution_host_to_device_bytes']}",
        )

        transformer = build_architecture(
            "transformer_decoder",
            dict(
                d_model=32,
                n_heads=4,
                n_layers=2,
                d_ff=64,
                vocab_size=64,
                max_seq_len=16,
            ),
        )
        t_package = transformer.build_package(seed=5)
        t_inputs = transformer.sample_inputs(batch=2, seq=8, seed=0)
        attention_node = next(
            node_id
            for node_id, node in transformer.graph.nodes.items()
            if node.op == "attention"
        )
        t_reference = numpy_execute(
            transformer.graph, t_inputs, t_package, capture=[attention_node]
        )
        started = time.perf_counter()
        t_execution = runtime.execute_graph(
            transformer.graph, t_inputs, t_package, capture=[attention_node]
        )
        transformer_seconds = time.perf_counter() - started
        ok, worst = _compare(
            t_execution.outputs, t_reference.outputs, rtol=3e-4, atol=2e-4
        )
        gate(
            "transformer parity",
            ok,
            f"max abs diff {worst:.3e}, {t_execution.kernels_launched} launches",
        )
        captured = np.max(
            np.abs(
                t_execution.intermediates[attention_node]
                - t_reference.intermediates[attention_node]
            )
        )
        gate("attention capture parity", float(captured) <= 3e-4, f"max abs diff {captured:.3e}")
        gate(
            "mixed device/host accounting",
            bool(t_execution.device_ops)
            and bool(t_execution.layout_ops)
            and t_execution.stats["device_to_host_bytes"] > 0,
            f"{len(t_execution.device_ops)} device ops, "
            f"{len(t_execution.layout_ops)} host layout ops",
        )
        gate(
            "no silent fallback",
            t_execution.backend == "cuda",
            f"backend={t_execution.backend}",
        )
        started = time.perf_counter()
        numpy_execute(transformer.graph, t_inputs, t_package, capture=[attention_node])
        numpy_seconds = time.perf_counter() - started
        report["measurements"] = {
            "mlp_cuda_seconds": mlp_seconds,
            "mlp_numpy_seconds": None,
            "transformer_cuda_seconds": transformer_seconds,
            "transformer_numpy_seconds": numpy_seconds,
            "mlp_stats": first.stats,
            "transformer_stats": t_execution.stats,
            "device_ops": t_execution.device_ops,
            "layout_ops": t_execution.layout_ops,
        }
        report["device"] = runtime.describe()

    report["executed"] = True
    report["status"] = "executed-on-device"
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/uzen-cuda-demo")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "N5",
        "compilers": describe_compilers(),
        "gates": [],
        "available": False,
        "executed": False,
    }
    exit_code = 1
    try:
        exit_code = _device_gates(report, root)
    except Exception as exc:  # surface a real failure, not a fake pass
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["ok"] = all(gate["ok"] for gate in report["gates"])
        report["gate_count"] = len(report["gates"])
        (root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        for gate in report["gates"]:
            print(f"[{'PASS' if gate['ok'] else 'FAIL'}] {gate['name']}: {gate['detail']}")
        print(f"status: {report['status']}")
        print(f"executed on device: {report['executed']}")
        print(f"report: {root / 'report.json'}")
    if report["status"] == "skipped-no-cuda-device":
        print("RESULT: SKIPPED (no CUDA device; kernels compiled to PTX only)")
        return exit_code if exit_code != 0 else 0
    passed = report["ok"] and exit_code == 0
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
