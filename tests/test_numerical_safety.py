"""Numerical-safety tests: non-finite values, degenerate shapes and extremes."""

from __future__ import annotations

import numpy as np
import pytest

from uzen.core.nir.executor import execute
from uzen.core.nir.graph import Graph
from uzen.core.nir.types import Ref, TensorType
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import ShapeMismatch, TensorInvalid


def _identity_graph(shape: tuple[int, ...]) -> Graph:
    graph = Graph(name="identity", inputs={"x": TensorType.of(shape)})
    graph.add_node("id", "identity", {"x": Ref("input:x")})
    graph.set_output("y", Ref("id"))
    return graph


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_package_validation_rejects_non_finite_values(value):
    package = WeightPackage.from_arrays({"w": np.array([value], dtype=np.float32)})
    with pytest.raises(TensorInvalid):
        package.validate()


def test_zero_rank_scalar_is_supported():
    package = WeightPackage.from_arrays({"scalar": np.array(3.0, dtype=np.float32)})
    assert package.get("scalar").shape == ()
    assert package.numel() == 1
    package.validate()
    package.assert_unmodified()

    result = execute(_identity_graph(()), {"x": np.array(2.0, dtype=np.float32)})
    assert result.outputs["y"].shape == ()


def test_empty_tensor_is_supported():
    package = WeightPackage.from_arrays({"empty": np.zeros((0, 4), dtype=np.float32)})
    assert package.get("empty").shape == (0, 4)
    assert package.numel() == 0
    package.validate()
    package.assert_unmodified()


def test_underflow_subnormal_remains_finite():
    subnormal = np.nextafter(np.float32(0.0), np.float32(1.0))
    package = WeightPackage.from_arrays({"w": np.array([subnormal], dtype=np.float32)})
    package.validate()
    assert np.isfinite(package.get("w")[0])
    assert package.get("w")[0] == subnormal


def test_extreme_dimension_is_preserved_without_overflow():
    package = WeightPackage.from_arrays({"w": np.zeros((1, 1_000_000), dtype=np.float32)})
    assert package.get("w").shape == (1, 1_000_000)
    assert package.numel() == 1_000_000
    assert package.nbytes() == 4_000_000
    package.validate()


def test_executor_rejects_non_finite_output():
    with pytest.raises(TensorInvalid):
        execute(
            _identity_graph((2,)),
            {"x": np.array([np.inf, 0.0], dtype=np.float32)},
        )


def test_executor_rejects_dimension_mismatch():
    with pytest.raises(ShapeMismatch):
        execute(
            _identity_graph((2,)),
            {"x": np.zeros(3, dtype=np.float32)},
        )


def test_float16_extreme_values_stay_finite():
    package = WeightPackage.from_arrays(
        {"w": np.array([np.finfo(np.float16).max, np.finfo(np.float16).min], dtype=np.float16)}
    )
    package.validate()
    assert np.isfinite(package.get("w")).all()
