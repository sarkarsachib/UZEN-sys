"""Native parameter-representation demonstration (phase N3).

This builds the checked-in C++ kernels and then, on real compiled plans:

1. prepares every representation UZEN ships (blocked dense, transposed, group-wise
   int8, group-wise int4, rank-r SVD factors, magnitude-pruned CSR) and checks each
   kernel against the matrix its packing reconstructs;
2. reports what each representation costs in bytes and bits per weight, and how far
   it moved the numbers;
3. runs whole graphs (MLP, transformer decoder, CNN, LSTM) through the compiled plan
   with all-int8 and all-int4 weights, comparing against the float32 NumPy executor;
4. runs one mixed plan where different linear weights use different native layouts,
   and shows the per-weight provenance the plan reports;
5. measures latency and storage of dense versus int8 versus int4 plans;
6. proves the API refuses unknown layouts, missing options and impossible ranks.

Weights are structured random initializations, labelled as such. Nothing here is
trained, and no representation is presented as lossless unless its measured error is
zero. The latency numbers are what this machine measured; the gate does not claim a
speedup.

Run with::

    python examples/native_representation_demo.py --root /tmp/uzen-representation-demo
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
from uzen.errors import InvalidOperationParameters
from uzen.runtimes.native import (
    NativeLibrary,
    build_native_library,
    compile_native_plan,
    execute_plan,
    normalize_representations,
    prepare_representation,
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
    (
        "transformer_decoder",
        dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16),
    ),
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
]

#: Measured tolerances for the plan parity gate, calibrated on this workspace.
PLAN_TOLERANCES = {
    "dense_blocked": 1e-6,
    "transposed": 1e-6,
    "int8_block": 1e-2,
    "int4_block": 5e-2,
    "mixed": 1e-2,
}


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


def _plan_error(
    definition: Any,
    package: Any,
    inputs: dict[str, np.ndarray],
    spec: dict[str, Any],
    library: NativeLibrary,
    reference: dict[str, np.ndarray],
) -> tuple[float, dict[str, Any]]:
    plan = compile_native_plan(definition.graph, package, representations=spec)
    execution = execute_plan(plan, inputs, package, library=library)
    return _max_error(execution.outputs, reference), execution.summary()


def _kernel_parity(
    library: NativeLibrary, prepared: Any, x: np.ndarray, bias: np.ndarray
) -> float:
    dense = prepared.to_dense()
    expected = (x.astype(np.float64) @ dense.astype(np.float64).T + bias).astype(
        np.float32
    )
    if prepared.kind == "dense_blocked":
        actual = library.gemm_packed(x, prepared.packed, bias=bias)
    elif prepared.kind == "transposed":
        actual = library.gemm_transposed(x, prepared.packed, bias=bias)
    elif prepared.kind == "int8_block":
        actual = library.gemm_int8_block(x, prepared.packed, bias=bias)
    elif prepared.kind == "int4_block":
        actual = library.gemm_int4_block(x, prepared.packed, bias=bias)
    elif prepared.kind == "low_rank":
        u, v = prepared.factors
        actual = library.gemm_low_rank(x, v, u, bias=bias)
    elif prepared.kind == "sparse_csr":
        values, indices, row_ptr = prepared.sparse
        actual = library.sparse_gemm(
            x,
            values,
            indices,
            row_ptr,
            prepared.in_features,
            prepared.out_features,
            bias=bias,
        )
    else:  # pragma: no cover - the kinds above are exhaustive
        raise InvalidOperationParameters(prepared.kind)
    return float(np.abs(actual.astype(np.float64) - expected).max())


def main(root: str = "/tmp/uzen-representation-demo") -> int:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "N3",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "layouts": {},
        "families": {},
        "mixed": {},
        "latency": {},
        "refusals": {},
        "gates": {},
    }

    library_path = build_native_library(directory / "libuzen_native.so", force=True)
    library = NativeLibrary(library_path)
    report["library"] = library.describe()

    # ------------------------------------------------------------------ #
    # 1. every layout: cost, error and kernel parity
    # ------------------------------------------------------------------ #
    print("layout parity and cost (weight 70x130, structured random):")
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((70, 130), dtype=np.float32)
    bias = rng.standard_normal(70, dtype=np.float32)
    x = rng.standard_normal((6, 130), dtype=np.float32)
    # A pruned copy exercises CSR at a density where it actually saves memory.
    keep = rng.random(weight.shape) < 0.04
    pruned = np.where(keep, weight, 0.0).astype(np.float32)
    layouts: list[tuple[str, Any, dict[str, Any]]] = [
        ("dense_blocked", weight, {}),
        ("transposed", weight, {}),
        ("int8_block", weight, {}),
        ("int4_block", weight, {}),
        ("low_rank", weight, {"rank": 20}),
        ("sparse_csr", pruned, {"threshold": 0.0}),
    ]
    for kind, source, options in layouts:
        prepared = prepare_representation(
            "w", source, kind, library=library, **options
        )
        parity = _kernel_parity(library, prepared, x, bias)
        described = prepared.report.to_dict()
        described["kernel_max_abs_error"] = parity
        described["reconstruction_max_abs_error"] = float(
            np.abs(prepared.to_dense() - source).max()
        )
        report["layouts"][kind] = described
        print(
            f"  {kind:14s} bits/weight={described['bits_per_weight']:7.3f} "
            f"bytes={described['packed_bytes']:6d} "
            f"reconstruction={described['reconstruction_max_abs_error']:.3e} "
            f"kernel={parity:.3e}"
        )
    report["gates"]["layout_parity"] = all(
        item["kernel_max_abs_error"] <= 1e-4
        for item in report["layouts"].values()
    )
    report["gates"]["exact_layouts"] = all(
        report["layouts"][kind]["reconstruction_max_abs_error"] == 0.0
        for kind in ("dense_blocked", "transposed")
    )
    report["gates"]["int4_smaller_than_int8"] = (
        report["layouts"]["int4_block"]["packed_bytes"]
        < report["layouts"]["int8_block"]["packed_bytes"]
        < report["layouts"]["dense_blocked"]["packed_bytes"]
    )

    # ------------------------------------------------------------------ #
    # 2. whole graphs through int8 and int4 plans
    # ------------------------------------------------------------------ #
    print("\nrepresentations on whole compiled plans (max abs error vs float32 NumPy):")
    family_gate = True
    for family, spec in FAMILIES:
        definition = build_architecture(family, spec)
        package = definition.build_package(seed=11)
        inputs = definition.sample_inputs(batch=2, seed=0)
        reference = numpy_execute(definition.graph, inputs, package).outputs
        names = compile_native_plan(definition.graph, package).prepack_names
        row: dict[str, Any] = {"weights": len(names)}
        variants = {
            "dense_blocked": {name: "dense_blocked" for name in names},
            "transposed": {name: "transposed" for name in names},
            "int8_block": {name: "int8_block" for name in names},
            "int4_block": {name: "int4_block" for name in names},
        }
        for label, spec_rep in variants.items():
            error, summary = _plan_error(
                definition, package, inputs, spec_rep, library, reference
            )
            row[label] = {
                "max_abs_error": error,
                "representation_bytes": summary["representation_bytes"],
                "representations": {
                    name: item["kind"]
                    for name, item in summary["representations"].items()
                },
            }
            family_gate = family_gate and error <= PLAN_TOLERANCES[label]
        report["families"][family] = row
        print(
            f"  {family:20s} weights={len(names):2d} "
            f"dense={row['dense_blocked']['max_abs_error']:.2e} "
            f"transposed={row['transposed']['max_abs_error']:.2e} "
            f"int8={row['int8_block']['max_abs_error']:.2e} "
            f"int4={row['int4_block']['max_abs_error']:.2e} "
            f"| bytes {row['dense_blocked']['representation_bytes']} -> "
            f"{row['int8_block']['representation_bytes']} -> "
            f"{row['int4_block']['representation_bytes']}"
        )
    report["gates"]["family_parity"] = family_gate

    # ------------------------------------------------------------------ #
    # 3. one mixed plan with per-weight layouts
    # ------------------------------------------------------------------ #
    print("\nmixed plan (transformer decoder): different layouts in one graph")
    mixed_definition = build_architecture(
        "transformer_decoder",
        dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, vocab_size=64, max_seq_len=16),
    )
    mixed_package = mixed_definition.build_package(seed=12)
    mixed_inputs = mixed_definition.sample_inputs(batch=2, seed=0)
    mixed_reference = numpy_execute(
        mixed_definition.graph, mixed_inputs, mixed_package
    ).outputs
    mixed_names = compile_native_plan(
        mixed_definition.graph, mixed_package
    ).prepack_names
    mixed_spec: dict[str, Any] = {}
    cycle = ["int8_block", "transposed", "int4_block", "dense_blocked"]
    for index, name in enumerate(mixed_names):
        shape = mixed_package.get(name).shape
        if index % 5 == 4:
            mixed_spec[name] = {"kind": "low_rank", "rank": int(min(shape))}
        else:
            mixed_spec[name] = cycle[index % len(cycle)]
    mixed_error, mixed_summary = _plan_error(
        mixed_definition,
        mixed_package,
        mixed_inputs,
        mixed_spec,
        library,
        mixed_reference,
    )
    report["mixed"] = {
        "weights": {
            name: {
                "requested": mixed_spec[name],
                "prepared": mixed_summary["representations"][name]["kind"],
                "bits_per_weight": mixed_summary["representations"][name][
                    "bits_per_weight"
                ],
                "packed_bytes": mixed_summary["representations"][name]["packed_bytes"],
                "max_abs_error": mixed_summary["representations"][name][
                    "max_abs_error"
                ],
                "digest": mixed_summary["representations"][name]["digest"],
            }
            for name in mixed_names
        },
        "max_abs_error": mixed_error,
        "representation_bytes": mixed_summary["representation_bytes"],
        "plan_representations": mixed_summary["plan"]["representations"],
    }
    for name in mixed_names:
        item = report["mixed"]["weights"][name]
        print(
            f"  {name:24s} {item['prepared']:14s} "
            f"{item['bits_per_weight']:7.3f} bits/weight "
            f"weight_error={item['max_abs_error']:.3e}"
        )
    print(
        f"  mixed plan max_abs_error={mixed_error:.3e} "
        f"stored={mixed_summary['representation_bytes']} bytes"
    )
    report["gates"]["mixed_parity"] = mixed_error <= PLAN_TOLERANCES["mixed"]

    # ------------------------------------------------------------------ #
    # 4. latency and storage on one benchmark transformer
    # ------------------------------------------------------------------ #
    print("\nlatency (median of 5, batch=4 seq=16 d_model=64 2 layers):")
    bench_definition = build_architecture("transformer_decoder", BENCH_SPEC)
    bench_package = bench_definition.build_package(seed=5)
    bench_inputs = bench_definition.sample_inputs(batch=4, seq=16, seed=0)
    bench_reference = numpy_execute(
        bench_definition.graph, bench_inputs, bench_package
    ).outputs
    bench_names = compile_native_plan(
        bench_definition.graph, bench_package
    ).prepack_names
    bench: dict[str, Any] = {}
    for label, spec_rep in (
        ("dense_blocked", {name: "dense_blocked" for name in bench_names}),
        ("int8_block", {name: "int8_block" for name in bench_names}),
        ("int4_block", {name: "int4_block" for name in bench_names}),
    ):
        bench_plan = compile_native_plan(
            bench_definition.graph, bench_package, representations=spec_rep
        )
        execution = execute_plan(
            bench_plan, bench_inputs, bench_package, library=library
        )
        milliseconds = _timed(
            lambda: execute_plan(
                bench_plan, bench_inputs, bench_package, library=library
            )
        ) * 1e3
        bench[label] = {
            "ms": milliseconds,
            "max_abs_error": _max_error(execution.outputs, bench_reference),
            "representation_bytes": execution.summary()["representation_bytes"],
        }
        print(
            f"  {label:14s} {milliseconds:6.3f} ms  "
            f"stored={bench[label]['representation_bytes']:7d} bytes  "
            f"error={bench[label]['max_abs_error']:.2e}"
        )
    report["latency"] = bench
    report["gates"]["latency_reports_storage"] = (
        bench["int4_block"]["representation_bytes"]
        < bench["int8_block"]["representation_bytes"]
        < bench["dense_blocked"]["representation_bytes"]
    )

    # ------------------------------------------------------------------ #
    # 5. explicit refusals
    # ------------------------------------------------------------------ #
    print("\nrefusals:")
    refusals: dict[str, bool] = {}

    def _refused(label: str, call: Callable[[], Any], message: str) -> None:
        try:
            call()
        except InvalidOperationParameters as error:
            refusals[label] = message in str(error)
        else:  # pragma: no cover - a missing refusal is a failed gate
            refusals[label] = False
        print(f"  {label:26s} refused={refusals[label]}")

    _refused(
        "unknown_layout",
        lambda: normalize_representations({"w": "float8_block"}),
        "unknown representation",
    )
    _refused(
        "low_rank_without_rank",
        lambda: normalize_representations({"w": {"kind": "low_rank"}}),
        "explicit 'rank'",
    )
    _refused(
        "sparse_without_threshold",
        lambda: normalize_representations({"w": {"kind": "sparse_csr"}}),
        "explicit 'threshold'",
    )
    _refused(
        "impossible_rank",
        lambda: prepare_representation(
            "w",
            np.zeros((4, 8), dtype=np.float32),
            "low_rank",
            library=library,
            rank=9,
        ),
        "exceeds the maximum",
    )
    _refused(
        "unused_weight",
        lambda: compile_native_plan(
            mixed_definition.graph,
            mixed_package,
            representations={"not_a_weight": "int8_block"},
        ),
        "no linear node",
    )
    _refused(
        "float64_exact_layout",
        lambda: prepare_representation(
            "w", np.zeros((4, 8), dtype=np.float64), "transposed", library=library
        ),
        "float32",
    )
    report["refusals"] = refusals
    report["gates"]["refusals"] = all(refusals.values())

    # ------------------------------------------------------------------ #
    # 6. gate
    # ------------------------------------------------------------------ #
    passed = all(report["gates"].values())
    report["gate"] = "PASS" if passed else "FAIL"
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\ngates:", json.dumps(report["gates"], sort_keys=True))
    print("gate:", report["gate"])
    print("report:", directory / "report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/uzen-representation-demo")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.root))
