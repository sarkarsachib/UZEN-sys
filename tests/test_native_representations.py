"""Native parameter representations: block int8/int4, low-rank, sparse, transposed.

The kernels are checked against the dequantized/reconstructed matrix, which is
produced by an independent NumPy path (unpacking the packed buffers), so a packing
or indexing error cannot cancel out.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.errors import InvalidOperationParameters
from uzen.runtimes.native import (
    NativeLibrary,
    NativeRuntime,
    build_native_library,
    compile_native_plan,
    execute_plan,
    normalize_representations,
    prepare_representation,
)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None and shutil.which("c++") is None,
    reason="a C++ compiler is required for the native representation tests",
)

MLP_SPEC = dict(input_dim=32, hidden_dims=[64, 64], output_dim=16)


@pytest.fixture(scope="session")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> NativeLibrary:
    output = tmp_path_factory.mktemp("native-representations") / "libuzen_native.so"
    build_native_library(output, force=True)
    return NativeLibrary(output)


def _assert_close(actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-5):
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=atol)


def _max_error(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> float:
    return max(
        float(np.abs(actual[name] - reference).max())
        for name, reference in expected.items()
    )


# --------------------------------------------------------------------------- #
# block int8 / int4
# --------------------------------------------------------------------------- #


def test_int8_block_matches_its_dequantized_matrix(native_library: NativeLibrary):
    rng = np.random.default_rng(0)
    # 130 input features span three 64-value groups; 70 outputs span two blocks.
    weight = rng.standard_normal((70, 130), dtype=np.float32)
    bias = rng.standard_normal(70, dtype=np.float32)
    x = rng.standard_normal((5, 130), dtype=np.float32)
    prepared = prepare_representation("w", weight, "int8_block", library=native_library)
    assert prepared.report.groups == 3
    assert prepared.report.group_size == 64
    assert prepared.report.zero_point == 0
    assert prepared.report.bits_per_weight < 32.0
    dense = prepared.to_dense()
    # Every reconstructed weight is within half a quantization step of the source.
    step = prepared.report.scale_max
    assert prepared.report.max_abs_error <= step * 0.51
    _assert_close(
        native_library.gemm_int8_block(x, prepared.packed, bias=bias),
        (x @ dense.T + bias).astype(np.float32),
    )
    fused = native_library.gemm_int8_block(
        x, prepared.packed, bias=bias, activation="gelu"
    )
    assert fused.shape == (5, 70)


def test_int8_group_scales_are_per_output_row_and_group(native_library: NativeLibrary):
    weight = np.zeros((2, 128), dtype=np.float32)
    weight[:, :64] = 1.0  # first group of both rows peaks at 1.0
    weight[:, 64:] = 0.5  # second group peaks at 0.5
    prepared = prepare_representation("w", weight, "int8_block", library=native_library)
    assert prepared.report.groups == 2
    assert prepared.report.scale_max == pytest.approx(1.0 / 127.0, rel=1e-5)
    assert prepared.report.scale_min == pytest.approx(0.5 / 127.0, rel=1e-5)
    # Dequantization reproduces a uniform 1.0 / 0.5 row to rounding error.
    dense = prepared.to_dense()
    np.testing.assert_allclose(dense[:, :64], 1.0, atol=1e-6)
    np.testing.assert_allclose(dense[:, 64:], 0.5, atol=1e-6)


def test_int4_block_round_trips_signed_nibbles(native_library: NativeLibrary):
    # Values that are exactly representable at the group scale, including both
    # range ends, so the reconstruction must be exact.
    weight = np.array(
        [[7.0, -7.0, 0.0, 3.0], [-3.0, 0.0, 7.0, -7.0]],
        dtype=np.float32,
    )
    prepared = prepare_representation("w", weight, "int4_block", library=native_library)
    assert prepared.report.max_abs_error == 0.0
    np.testing.assert_array_equal(prepared.to_dense(), weight)
    # Feeding one-hot rows recovers the weight rows from the packed nibbles.
    identity = np.eye(4, dtype=np.float32)
    _assert_close(native_library.gemm_int4_block(identity, prepared.packed), weight.T)
    rng = np.random.default_rng(1)
    # Block-aligned shape, so the reported cost is the layout's cost and not padding.
    wide = rng.standard_normal((64, 128), dtype=np.float32)
    prepared = prepare_representation("w", wide, "int4_block", library=native_library)
    assert prepared.report.bits_per_weight < 5.0
    x = rng.standard_normal((4, 128), dtype=np.float32)
    _assert_close(
        native_library.gemm_int4_block(x, prepared.packed),
        (x @ prepared.to_dense().T).astype(np.float32),
    )


def test_int4_packing_refuses_values_that_are_not_nibbles(
    native_library: NativeLibrary,
):
    scales = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(InvalidOperationParameters, match="one nibble per element"):
        native_library.pack_weight_int4_block(
            np.full((2, 3), 200, dtype=np.uint8), scales
        )
    with pytest.raises(InvalidOperationParameters, match="uint8"):
        native_library.pack_weight_int4_block(np.zeros((2, 64), dtype=np.int8), scales)


def test_block_kernels_refuse_bad_dtypes_and_scale_shapes(
    native_library: NativeLibrary,
):
    weight = np.zeros((2, 64), dtype=np.int8)
    with pytest.raises(InvalidOperationParameters, match="scales shape"):
        native_library.pack_weight_int8_block(weight, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(InvalidOperationParameters, match="float32"):
        native_library.pack_weight_int8_block(
            weight, np.ones((2, 1), dtype=np.float64)
        )
    with pytest.raises(InvalidOperationParameters, match="finite and positive"):
        native_library.pack_weight_int8_block(
            np.zeros((2, 64), dtype=np.int8), np.zeros((2, 1), dtype=np.float32)
        )
    with pytest.raises(InvalidOperationParameters, match="int8"):
        native_library.pack_weight_int8_block(
            np.zeros((2, 64), dtype=np.float32), np.ones((2, 1), dtype=np.float32)
        )


# --------------------------------------------------------------------------- #
# low rank and sparse
# --------------------------------------------------------------------------- #


def test_low_rank_is_exact_at_full_rank(native_library: NativeLibrary):
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((24, 40), dtype=np.float32)
    prepared = prepare_representation(
        "w", weight, "low_rank", library=native_library, rank=24
    )
    assert prepared.report.rank == 24
    assert prepared.report.max_abs_error < 1e-5
    x = rng.standard_normal((3, 40), dtype=np.float32)
    u, v = prepared.factors
    _assert_close(
        native_library.gemm_low_rank(x, v, u),
        (x @ (u @ v).T).astype(np.float32),
    )


def test_low_rank_reports_the_error_it_introduces(native_library: NativeLibrary):
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((32, 16), dtype=np.float32)
    prepared = prepare_representation(
        "w", weight, "low_rank", library=native_library, rank=4
    )
    approximation = prepared.to_dense()
    measured = float(np.abs(approximation - weight).max())
    assert prepared.report.max_abs_error == pytest.approx(measured, rel=1e-5)
    assert 0.0 < prepared.report.spectral_energy < 1.0
    # The packed size is factor storage, not a matrix.
    assert prepared.report.packed_bytes == 4 * (32 * 4 + 4 * 16)


@pytest.mark.parametrize("rank", [0, 17])
def test_low_rank_refuses_an_impossible_rank(
    native_library: NativeLibrary, rank: int
):
    weight = np.zeros((4, 16), dtype=np.float32)
    with pytest.raises(InvalidOperationParameters, match="rank"):
        prepare_representation(
            "w", weight, "low_rank", library=native_library, rank=rank
        )


def test_sparse_csr_is_exact_at_zero_threshold(native_library: NativeLibrary):
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((6, 9), dtype=np.float32)
    weight[1, :] = 0.0
    weight[3, :] = 0.0
    prepared = prepare_representation(
        "w", weight, "sparse_csr", library=native_library, threshold=0.0
    )
    assert prepared.report.max_abs_error == 0.0
    assert prepared.report.sparsity == pytest.approx(2 / 6)
    np.testing.assert_array_equal(prepared.to_dense(), weight)
    values, indices, row_ptr = prepared.sparse
    assert int(row_ptr[-1]) == values.size
    x = rng.standard_normal((4, 9), dtype=np.float32)
    _assert_close(
        native_library.sparse_gemm(x, values, indices, row_ptr, 9, 6),
        (x @ weight.T).astype(np.float32),
    )


def test_sparse_csr_reports_magnitude_pruning(native_library: NativeLibrary):
    rng = np.random.default_rng(5)
    weight = rng.standard_normal((64, 48), dtype=np.float32)
    threshold = 0.5
    prepared = prepare_representation(
        "w", weight, "sparse_csr", library=native_library, threshold=threshold
    )
    expected_sparsity = float(np.mean(np.abs(weight) <= threshold))
    assert prepared.report.sparsity == pytest.approx(expected_sparsity)
    # Only weights at or below the threshold were dropped.
    assert prepared.report.max_abs_error <= threshold + 1e-7
    x = rng.standard_normal((2, 48), dtype=np.float32)
    values, indices, row_ptr = prepared.sparse
    _assert_close(
        native_library.sparse_gemm(x, values, indices, row_ptr, 48, 64),
        (x @ prepared.to_dense().T).astype(np.float32),
        atol=1e-4,
    )


def test_sparse_gemm_refuses_a_broken_csr(native_library: NativeLibrary):
    values = np.ones(2, dtype=np.float32)
    indices = np.array([0, 3], dtype=np.int32)
    x = np.ones((1, 3), dtype=np.float32)
    with pytest.raises(InvalidOperationParameters, match="row_ptr"):
        native_library.sparse_gemm(
            x, values, indices, np.array([0, 2], dtype=np.int32), 3, 2
        )
    with pytest.raises(InvalidOperationParameters, match=r"\[0, 3\)"):
        native_library.sparse_gemm(
            x, values, indices, np.array([0, 1, 2], dtype=np.int32), 3, 2
        )


# --------------------------------------------------------------------------- #
# transposed dense layout
# --------------------------------------------------------------------------- #


def test_transposed_layout_is_exact_and_unpadded(native_library: NativeLibrary):
    rng = np.random.default_rng(6)
    weight = rng.standard_normal((20, 34), dtype=np.float32)
    prepared = prepare_representation("w", weight, "transposed", library=native_library)
    assert prepared.report.packed_bytes == weight.nbytes
    assert prepared.report.max_abs_error == 0.0
    np.testing.assert_array_equal(prepared.to_dense(), weight)
    x = rng.standard_normal((3, 34), dtype=np.float32)
    _assert_close(
        native_library.gemm_transposed(x, prepared.packed),
        (x @ weight.T).astype(np.float32),
    )


def test_exact_layouts_refuse_non_float32(native_library: NativeLibrary):
    weight = np.zeros((4, 8), dtype=np.float64)
    with pytest.raises(InvalidOperationParameters, match="float32"):
        prepare_representation("w", weight, "transposed", library=native_library)
    # A lossy representation may reduce precision but reports it.
    prepared = prepare_representation("w", weight, "int8_block", library=native_library)
    assert prepared.report.source_dtype == "float64"


# --------------------------------------------------------------------------- #
# representation-aware plans
# --------------------------------------------------------------------------- #


def _plan_error(definition, package, inputs, spec, library) -> tuple[float, dict]:
    plan = compile_native_plan(
        definition.graph, package, representations=spec
    )
    execution = execute_plan(plan, inputs, package, library=library)
    reference = numpy_execute(definition.graph, inputs, package).outputs
    return _max_error(execution.outputs, reference), execution.representations


@pytest.mark.parametrize(
    ("kind", "tolerance"),
    [
        ("transposed", 1e-6),
        ("dense_blocked", 1e-6),
        ("int8_block", 5e-3),
        ("int4_block", 5e-2),
    ],
)
def test_plan_dispatches_each_representation(
    native_library: NativeLibrary, kind: str, tolerance: float
):
    definition = build_architecture("mlp", MLP_SPEC)
    package = definition.build_package(seed=7)
    inputs = definition.sample_inputs(batch=3, seed=0)
    names = compile_native_plan(definition.graph, package).prepack_names
    error, reports = _plan_error(
        definition, package, inputs, {name: kind for name in names}, native_library
    )
    assert error <= tolerance
    assert {report["kind"] for report in reports.values()} == {kind}


def test_mixed_representations_run_in_one_plan(native_library: NativeLibrary):
    definition = build_architecture("mlp", MLP_SPEC)
    package = definition.build_package(seed=8)
    inputs = definition.sample_inputs(batch=2, seed=0)
    names = compile_native_plan(definition.graph, package).prepack_names
    spec = {
        names[0]: "transposed",
        names[1]: {"kind": "low_rank", "rank": int(min(package.get(names[1]).shape))},
        names[2]: {"kind": "sparse_csr", "threshold": 0.0},
    }
    error, reports = _plan_error(definition, package, inputs, spec, native_library)
    assert error <= 5e-5
    assert [reports[name]["kind"] for name in names] == [
        "transposed",
        "low_rank",
        "sparse_csr",
    ]
    execution = execute_plan(
        compile_native_plan(definition.graph, package, representations=spec),
        inputs,
        package,
        library=native_library,
    )
    summary = execution.summary()
    assert summary["representation_bytes"] > 0
    assert summary["representations"][names[1]]["rank"] == int(
        min(package.get(names[1]).shape)
    )


def test_representation_bytes_are_ordered_as_documented(
    native_library: NativeLibrary,
):
    rng = np.random.default_rng(9)
    weight = rng.standard_normal((128, 128), dtype=np.float32)
    sizes = {}
    for kind, options in (
        ("dense_blocked", {}),
        ("transposed", {}),
        ("int8_block", {}),
        ("int4_block", {}),
        ("low_rank", {"rank": 16}),
        ("sparse_csr", {"threshold": 0.0}),
    ):
        # CSR pays 4 bytes of values + 4 bytes of int32 index per nonzero, so it
        # only beats float32 dense below ~50% density and int4 below ~6%.
        # This weight is ~3% dense, which is where CSR is the smallest layout.
        source = weight
        if kind == "sparse_csr":
            keep = rng.random(weight.shape) < 0.03
            source = np.where(keep, weight, 0.0).astype(np.float32)
        prepared = prepare_representation(
            "w", source, kind, library=native_library, **options
        )
        sizes[kind] = prepared.byte_size
    assert sizes["int4_block"] < sizes["int8_block"]
    assert sizes["int8_block"] < sizes["transposed"]
    assert sizes["transposed"] == weight.nbytes
    # 128x128 at rank 16 is 4 * (128*16 + 16*128) bytes of factors.
    assert sizes["low_rank"] == 4 * (128 * 16 + 16 * 128)
    assert sizes["low_rank"] < sizes["transposed"]
    assert sizes["sparse_csr"] < sizes["int4_block"]


def test_representation_digests_are_deterministic(native_library: NativeLibrary):
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((8, 12), dtype=np.float32)
    first = prepare_representation("w", weight, "int4_block", library=native_library)
    second = prepare_representation("w", weight, "int4_block", library=native_library)
    assert first.report.digest == second.report.digest
    other = prepare_representation("w", weight * 2.0, "int4_block", library=native_library)
    assert other.report.digest != first.report.digest


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #


def test_normalize_representations_validates_options():
    assert normalize_representations(None) == {}
    assert normalize_representations({"w": "int8_block"}) == {
        "w": {"kind": "int8_block"}
    }
    with pytest.raises(InvalidOperationParameters, match="unknown representation"):
        normalize_representations({"w": "float8"})
    with pytest.raises(InvalidOperationParameters, match="explicit 'rank'"):
        normalize_representations({"w": {"kind": "low_rank"}})
    with pytest.raises(InvalidOperationParameters, match="explicit 'threshold'"):
        normalize_representations({"w": {"kind": "sparse_csr"}})
    with pytest.raises(InvalidOperationParameters, match="does not accept option"):
        normalize_representations({"w": {"kind": "int8_block", "rank": 4}})
    with pytest.raises(InvalidOperationParameters, match="positive integer"):
        normalize_representations({"w": {"kind": "low_rank", "rank": 0}})
    with pytest.raises(InvalidOperationParameters, match="non-negative"):
        normalize_representations(
            {"w": {"kind": "sparse_csr", "threshold": -1.0}}
        )


def test_plan_refuses_a_representation_for_an_unused_weight(
    native_library: NativeLibrary,
):
    definition = build_architecture("mlp", MLP_SPEC)
    package = definition.build_package(seed=11)
    with pytest.raises(InvalidOperationParameters, match="no linear node"):
        compile_native_plan(
            definition.graph, package, representations={"not_a_weight": "int8_block"}
        )


def test_prepared_weights_are_cached_per_representation(
    native_library: NativeLibrary,
):
    definition = build_architecture("mlp", MLP_SPEC)
    package = definition.build_package(seed=12)
    names = compile_native_plan(definition.graph, package).prepack_names
    plan = compile_native_plan(
        definition.graph, package, representations={names[0]: "int8_block"}
    )
    first = plan.prepare_weights(package, native_library)
    second = plan.prepare_weights(package, native_library)
    assert first[names[0]] is second[names[0]]
    assert first[names[0]].kind == "int8_block"
    assert plan.describe()["representations"] == {names[0]: {"kind": "int8_block"}}
    summary = plan.representation_summary()
    assert summary[names[0]]["kind"] == "int8_block"
    # Exact layouts keep N2's prepack description fields for existing callers.
    dense = compile_native_plan(definition.graph, package).prepare_weights(
        package, native_library
    )
    assert dense[names[1]].describe()["padding_overhead"] >= 1.0
    assert dense[names[1]].describe()["padded_elements"] >= int(
        package.get(names[1]).size
    )


def test_capabilities_report_the_representations(native_library: NativeLibrary):
    described = NativeRuntime(library_path=native_library.path).capabilities().to_dict()
    extra = described["extra"]
    assert extra["representation_aware_gemm"] is True
    assert sorted(extra["representations"]) == sorted(
        [
            "dense_blocked",
            "transposed",
            "int8_block",
            "int4_block",
            "low_rank",
            "sparse_csr",
        ]
    )
    assert native_library.quant_group == 64
    assert "gemm_int4_block" in native_library.describe()["kernels"]
