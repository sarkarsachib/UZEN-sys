"""Real native-CPU backend demonstration.

This builds the checked-in C++ kernels with the local compiler, then:

1. executes a generated MLP and a generated transformer decoder through both the
   NumPy reference executor and the native C++ executor, comparing every output;
2. exercises causal, sliding-window and ALiBi attention kernels;
3. saves the transformer package to GGUF, reloads it through the native runtime,
   and repeats the forward pass;
4. measures native versus NumPy matmul/attention latency on fixed shapes;
5. proves the native executor refuses int8 ``quantized_linear`` instead of
   silently falling back.

The weights are structured random initializations, explicitly labelled as such.
No speedup is claimed: the benchmark simply records what this machine measured.

Run with::

    python examples/native_backend_demo.py --root /tmp/uzen-native-demo
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
from uzen.errors import UnsupportedOperator
from uzen.runtimes.native import (
    NativeLibrary,
    NativeRuntime,
    build_native_library,
    execute_native,
)

MLP_SPEC = dict(input_dim=64, hidden_dims=[128, 96], output_dim=32)
TRANSFORMER_SPEC = dict(
    d_model=32,
    n_heads=4,
    n_layers=2,
    d_ff=64,
    vocab_size=64,
    max_seq_len=16,
)


def _timed(function: Callable[[], Any], repeats: int = 3) -> float:
    function()  # warm-up
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def _reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool = True,
    window: int | None = None,
    alibi: np.ndarray | None = None,
) -> np.ndarray:
    q64, k64, v64 = (array.astype(np.float64) for array in (q, k, v))
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.einsum("bhtd,bhsd->bhts", q64, k64) * scale
    t, s = scores.shape[-2], scores.shape[-1]
    qpos, kpos = np.arange(s - t, s)[:, None], np.arange(s)[None, :]
    if alibi is not None:
        distance = (qpos - kpos) if causal else np.abs(qpos - kpos)
        scores = scores - np.asarray(alibi)[None, :, None, None] * distance[
            None, None, :, :
        ]
    mask = np.zeros((t, s), dtype=bool)
    if causal:
        mask |= kpos > qpos
    if window is not None:
        mask |= np.abs(qpos - kpos) > window
    scores = np.where(mask[None, None, :, :], -np.inf, scores)
    maximum = np.max(scores, axis=-1, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    exponentials = np.exp(scores - maximum)
    denominator = exponentials.sum(axis=-1, keepdims=True)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return np.einsum(
        "bhts,bhsd->bhtd", exponentials / denominator, v64
    ).astype(np.float32)


def _compare(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, Any]:
    worst = 0.0
    for name in reference:
        difference = np.abs(
            reference[name].astype(np.float64) - candidate[name].astype(np.float64)
        )
        worst = max(worst, float(difference.max()) if difference.size else 0.0)
    return {"outputs": sorted(reference), "max_abs_error": worst}


def main(root: str = "/tmp/uzen-native-demo") -> int:
    started = time.perf_counter()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gates": {},
        "measurements": {},
    }

    print("== 1. build the native C++ kernels ==")
    library_path = build_native_library(
        root_path / "native" / "libuzen_native.so", force=True
    )
    library = NativeLibrary(library_path)
    runtime = NativeRuntime(library_path=library_path)
    report["measurements"]["library"] = {
        "path": str(library_path),
        "bytes": library_path.stat().st_size,
        "capabilities": runtime.capabilities().to_dict(),
    }
    print(f"library: {library_path} ({library_path.stat().st_size:,} bytes)")

    print("== 2. generated MLP: NumPy vs native ==")
    mlp = build_architecture("mlp", MLP_SPEC)
    mlp_package = mlp.build_package(seed=3)
    mlp_inputs = mlp.sample_inputs(batch=4, seed=0)
    mlp_reference = numpy_execute(mlp.graph, mlp_inputs, mlp_package)
    mlp_native = execute_native(
        mlp.graph, mlp_inputs, mlp_package, library=library
    )
    mlp_parity = _compare(mlp_reference.outputs, mlp_native.outputs)
    report["measurements"]["mlp"] = {
        **mlp_parity,
        "native_ops": mlp_native.native_ops,
        "layout_ops": mlp_native.layout_ops,
    }
    print(f"mlp outputs={mlp_parity['outputs']} max_abs_error={mlp_parity['max_abs_error']:.3e}")

    print("== 3. generated transformer decoder: NumPy vs native ==")
    transformer = build_architecture("transformer_decoder", TRANSFORMER_SPEC)
    transformer_package = transformer.build_package(seed=7)
    transformer_inputs = transformer.sample_inputs(batch=1, seq=8, seed=0)
    transformer_reference = numpy_execute(
        transformer.graph, transformer_inputs, transformer_package
    )
    transformer_native = execute_native(
        transformer.graph,
        transformer_inputs,
        transformer_package,
        library=library,
    )
    transformer_parity = _compare(
        transformer_reference.outputs, transformer_native.outputs
    )
    report["measurements"]["transformer"] = {
        **transformer_parity,
        "native_ops": transformer_native.native_ops,
        "layout_ops": transformer_native.layout_ops,
    }
    print(
        f"transformer outputs={transformer_parity['outputs']} "
        f"max_abs_error={transformer_parity['max_abs_error']:.3e}"
    )

    print("== 4. attention masks: causal, window, ALiBi ==")
    rng = np.random.default_rng(11)
    q = rng.standard_normal((2, 4, 6, 16), dtype=np.float32)
    k = rng.standard_normal((2, 4, 9, 16), dtype=np.float32)
    v = rng.standard_normal((2, 4, 9, 16), dtype=np.float32)
    mask_cases = {
        "causal": {},
        "bidirectional_window": {"causal": False, "window": 2},
        "alibi": {
            "alibi": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        },
    }
    attention_results: dict[str, Any] = {}
    for name, options in mask_cases.items():
        expected = _reference_attention(q, k, v, **options)
        actual = library.attention(q, k, v, **options)
        difference = float(np.abs(expected.astype(np.float64) - actual.astype(np.float64)).max())
        attention_results[name] = difference
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    report["measurements"]["attention"] = attention_results
    print("attention max_abs_error:", attention_results)

    print("== 5. GGUF round-trip through the native runtime ==")
    export_path = runtime.save(
        transformer_package, root_path / "exports" / "native-transformer.gguf"
    )
    reloaded_package = runtime.load(export_path)
    reloaded_execution = execute_native(
        transformer.graph,
        transformer_inputs,
        reloaded_package,
        library=library,
    )
    gguf_parity = _compare(
        transformer_reference.outputs, reloaded_execution.outputs
    )
    report["measurements"]["gguf_roundtrip"] = {
        "path": export_path,
        "bytes": Path(export_path).stat().st_size,
        **gguf_parity,
    }
    print(f"gguf: {export_path} ({Path(export_path).stat().st_size:,} bytes) "
          f"max_abs_error={gguf_parity['max_abs_error']:.3e}")

    print("== 6. measured latency on fixed shapes ==")
    a = rng.standard_normal((256, 256), dtype=np.float32)
    b = rng.standard_normal((256, 256), dtype=np.float32)
    # Measure both native paths before invoking NumPy's BLAS. On this host a
    # NumPy matmul perturbs the OpenMP thread pool enough to distort the next
    # native attention timings; the native numbers must stand on their own.
    native_matmul = _timed(lambda: library.matmul(a, b), repeats=5)
    attention_native = _timed(
        lambda: library.attention(q, k, v), repeats=5
    )
    numpy_matmul = _timed(lambda: a @ b, repeats=5)
    attention_numpy = _timed(
        lambda: _reference_attention(q, k, v), repeats=5
    )
    report["measurements"]["benchmark"] = {
        "shape": [256, 256, 256],
        "native_matmul_seconds": native_matmul,
        "numpy_matmul_seconds": numpy_matmul,
        "native_attention_seconds": attention_native,
        "numpy_attention_seconds": attention_numpy,
        "note": (
            "single-process wall clock on this machine, native paths measured "
            "before NumPy BLAS; no speedup is claimed"
        ),
    }
    print(
        f"matmul 256^3 native={native_matmul * 1000:.2f}ms "
        f"numpy={numpy_matmul * 1000:.2f}ms"
    )
    print(
        f"attention native={attention_native * 1000:.2f}ms "
        f"numpy={attention_numpy * 1000:.2f}ms"
    )

    print("== 7. quantized operator refusal ==")
    quantized = quantize_int8(transformer.graph, transformer_package)
    refused = False
    try:
        execute_native(
            quantized.graph,
            transformer_inputs,
            quantized.package,
            library=library,
        )
    except UnsupportedOperator as exc:
        refused = "quantized_linear" in str(exc)
    report["measurements"]["quantized_refusal"] = {
        "rewritten_nodes": quantized.report.num_rewritten_nodes,
        "refused": refused,
    }
    print(f"quantized_linear refused: {refused}")

    report["gates"] = {
        "library_built": library_path.is_file(),
        "mlp_parity": mlp_parity["max_abs_error"] <= 1e-4,
        "transformer_parity": transformer_parity["max_abs_error"] <= 1e-4,
        "attention_parity": all(value <= 1e-5 for value in attention_results.values()),
        "gguf_roundtrip": gguf_parity["max_abs_error"] <= 1e-4,
        "benchmark_measured": native_matmul > 0.0 and numpy_matmul > 0.0,
        "unsupported_refused": refused,
    }
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report_path = root_path / "reports" / "native_backend.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("gates:", json.dumps(report["gates"], sort_keys=True))
    print("report:", report_path)
    print("gate:", "PASS" if all(report["gates"].values()) else "FAIL")
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/uzen-native-demo")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.root))
