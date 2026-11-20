"""CUDA kernel compilation, honest refusal, fallback policy and GPU parity.

The tests that need a device are skipped when none is present, but they are real
tests: they build the same graphs the CPU runtimes use and compare outputs. The
compile tests run anywhere NVRTC or nvcc exists and assemble the PTX with ptxas,
so kernel syntax and PTX validity are checked even without a GPU.
"""

from __future__ import annotations

import locale
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nws.families import build_architecture
from uzen.errors import RuntimeUnavailable, UnsupportedOperator
from uzen.runtimes.cuda import (
    CudaRuntime,
    compile_ptx,
    cuda_available,
    describe_compilers,
    device_snapshot,
)
from uzen.runtimes.cuda.executor import (
    BINARY_KERNELS,
    DEVICE_OPS,
    LAYOUT_OPS,
    UNARY_KERNELS,
)
from uzen.runtimes.cuda.nvrtc import KERNELS

requires_device = pytest.mark.skipif(
    not cuda_available(), reason="a CUDA device and driver are required"
)
requires_compiler = pytest.mark.skipif(
    not describe_compilers()["can_compile"],
    reason="NVRTC or nvcc is required to compile the kernels",
)


def _ptxas() -> str | None:
    found = shutil.which("ptxas")
    if found:
        return found
    bundled = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cuda_nvcc" / "bin" / "ptxas"
    return str(bundled) if bundled.exists() else None


@pytest.fixture(scope="module")
def compiled_ptx(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    output = tmp_path_factory.mktemp("cuda") / "uzen_cuda.ptx"
    path, report = compile_ptx(force=True, output=output)
    return path, report


@requires_compiler
def test_kernel_source_compiles_to_ptx(compiled_ptx: tuple[Path, dict]):
    path, report = compiled_ptx
    assert path.is_file() and path.stat().st_size > 0
    assert report["compiler"] in ("nvrtc", "nvcc")
    text = path.read_text(encoding="utf-8")
    assert ".version" in text and ".target" in text
    missing = [kernel for kernel in KERNELS if kernel not in text]
    assert not missing, f"kernels missing from compiled PTX: {missing}"
    assert len(KERNELS) == len(set(KERNELS))


@requires_compiler
def test_compiled_ptx_assembles_with_ptxas(compiled_ptx: tuple[Path, dict], tmp_path: Path):
    assembler = _ptxas()
    if assembler is None:
        pytest.skip("ptxas is not available")
    path, report = compiled_ptx
    arch = str(report.get("arch", "compute_60")).replace("compute_", "sm_")
    cubin = tmp_path / "uzen_cuda.cubin"
    completed = subprocess.run(
        [assembler, f"-arch={arch}", "-o", str(cubin), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert cubin.is_file() and cubin.stat().st_size > 0


@requires_compiler
def test_compilation_does_not_poison_the_thread_locale(tmp_path: Path):
    before = locale.getpreferredencoding(False)
    compile_ptx(force=True, output=tmp_path / "locale.ptx")
    assert locale.getpreferredencoding(False) == before


@requires_compiler
def test_ptx_cache_is_content_addressed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UZEN_CUDA_CACHE", str(tmp_path))
    first, first_report = compile_ptx(arch="compute_60")
    second, second_report = compile_ptx(arch="compute_60")
    assert first == second
    assert first_report["cache_hit"] is False
    assert second_report["cache_hit"] is True
    assert first.stat().st_size == second.stat().st_size


def test_executor_operator_coverage_is_backed_by_kernels():
    assert set(UNARY_KERNELS.values()) <= set(KERNELS)
    assert {"add", "sub", "mul"} == set(BINARY_KERNELS)
    assert {"linear", "matmul", "layernorm", "rmsnorm", "softmax", "attention"} <= DEVICE_OPS
    assert {"reshape", "rope", "split_qkv", "merge_heads"} <= LAYOUT_OPS
    assert "quantized_linear" not in DEVICE_OPS


def test_compilers_are_reported_truthfully():
    report = describe_compilers()
    assert report["can_compile"] is bool(report["nvcc"] or report["nvrtc"])
    assert "nvcc" in report and "nvrtc" in report


def test_device_snapshot_never_raises():
    snapshot = device_snapshot()
    assert set(snapshot) >= {"available", "reason", "driver", "devices"}
    assert isinstance(snapshot["devices"], list)


@pytest.mark.skipif(cuda_available(), reason="this asserts the no-device policy")
def test_strict_runtime_refuses_without_a_device():
    with pytest.raises(RuntimeUnavailable) as captured:
        CudaRuntime()
    assert captured.value.code == "runtime.unavailable"
    with pytest.raises(RuntimeUnavailable):
        CudaRuntime(auto_compile=False)


@pytest.mark.skipif(cuda_available(), reason="this asserts the no-device policy")
def test_explicit_cpu_fallback_reports_that_it_fell_back():
    definition = build_architecture("mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4))
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    with CudaRuntime(fallback="cpu") as runtime:
        capabilities = runtime.capabilities()
        assert capabilities.backend == "numpy-fallback"
        assert capabilities.device == "cpu"
        assert capabilities.extra["cuda"] is False
        assert capabilities.extra["execution_verified"] is False
        assert capabilities.extra["fallback"] == "cpu"
        execution = runtime.execute_graph(definition.graph, inputs, package)
        assert execution.backend == "numpy-fallback"
        assert execution.reason
        for name, expected in reference.outputs.items():
            np.testing.assert_array_equal(execution.outputs[name], expected)
        described = runtime.describe()
        assert described["available"] is False
        assert described["executions"] == 0


def test_fallback_value_is_validated():
    with pytest.raises(ValueError, match="fallback"):
        CudaRuntime(fallback="maybe")


def test_non_float32_arithmetic_is_refused_before_any_device_work():
    definition = build_architecture("mlp", dict(input_dim=4, hidden_dims=[8], output_dim=2))
    package = definition.build_package(seed=0)
    inputs = definition.sample_inputs(batch=1, seed=0)
    name = next(iter(inputs))
    with CudaRuntime(fallback="cpu") as runtime:
        if runtime.available:  # pragma: no cover - only on a GPU machine
            with pytest.raises(UnsupportedOperator, match="float32"):
                runtime.execute(
                    definition.graph,
                    {name: np.asarray(inputs[name], dtype=np.float64)},
                    package,
                )


@requires_device
def test_mlp_matches_the_numpy_executor():
    definition = build_architecture(
        "mlp", dict(input_dim=8, hidden_dims=[16, 12], output_dim=4)
    )
    package = definition.build_package(seed=3)
    inputs = definition.sample_inputs(batch=2, seed=0)
    reference = numpy_execute(definition.graph, inputs, package)
    with CudaRuntime(fallback="error") as runtime:
        execution = runtime.execute_graph(definition.graph, inputs, package)
        assert execution.backend == "cuda"
        assert execution.kernels_launched > 0
        for name, expected in reference.outputs.items():
            np.testing.assert_allclose(
                execution.outputs[name], expected, rtol=1e-4, atol=1e-5
            )
        capabilities = runtime.capabilities()
        assert capabilities.device.startswith("cuda:")
        assert capabilities.extra["execution_verified"] is True
        assert capabilities.extra["kernels"] == list(KERNELS)


@requires_device
def test_transformer_matches_numpy_and_captures_attention():
    definition = build_architecture(
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
    package = definition.build_package(seed=5)
    inputs = definition.sample_inputs(batch=1, seq=6, seed=0)
    attention_node = next(
        node_id
        for node_id, node in definition.graph.nodes.items()
        if node.op == "attention"
    )
    reference = numpy_execute(
        definition.graph, inputs, package, capture=[attention_node]
    )
    with CudaRuntime(fallback="error") as runtime:
        execution = runtime.execute_graph(
            definition.graph, inputs, package, capture=[attention_node]
        )
        for name, expected in reference.outputs.items():
            np.testing.assert_allclose(
                execution.outputs[name], expected, rtol=3e-4, atol=2e-4
            )
        np.testing.assert_allclose(
            execution.intermediates[attention_node],
            reference.intermediates[attention_node],
            rtol=3e-4,
            atol=2e-4,
        )
        assert {"embedding", "attention", "layernorm", "gelu"} <= set(
            execution.device_ops
        )
        assert {"split_qkv", "merge_heads", "pos_embedding"} <= set(
            execution.layout_ops
        )
        assert execution.stats["host_to_device_bytes"] > 0
        assert execution.stats["device_to_host_bytes"] > 0


@requires_device
def test_weights_stay_resident_between_runs():
    definition = build_architecture(
        "mlp", dict(input_dim=16, hidden_dims=[32, 32], output_dim=8)
    )
    package = definition.build_package(seed=1)
    inputs = definition.sample_inputs(batch=4, seed=0)
    with CudaRuntime(fallback="error") as runtime:
        first = runtime.execute_graph(definition.graph, inputs, package)
        second = runtime.execute_graph(definition.graph, inputs, package)
        assert first.stats["execution_host_to_device_bytes"] > second.stats[
            "execution_host_to_device_bytes"
        ]
        for name, expected in first.outputs.items():
            np.testing.assert_allclose(second.outputs[name], expected, rtol=0, atol=0)
        assert runtime.describe()["executions"] == 2


@requires_device
def test_attention_semantics_match_the_reference():
    rng = np.random.default_rng(7)
    from uzen.core.nir.graph import Graph
    from uzen.core.nir.ops import get_operator
    from uzen.core.nir.types import TensorType
    from uzen.core.nws.tensors import WeightPackage
    from uzen.runtimes.cuda.executor import CudaExecutor
    from uzen.runtimes.cuda.driver import CudaDriver
    from uzen.runtimes.cuda.nvrtc import compile_ptx
    from uzen.runtimes.cuda.allocator import DeviceAllocator

    graph = Graph(name="attention_only")
    graph.add_input("q", TensorType.of((2, 4, 3, 8), "float32"))
    graph.add_input("k", TensorType.of((2, 4, 5, 8), "float32"))
    graph.add_input("v", TensorType.of((2, 4, 5, 8), "float32"))
    reference = graph.add_node(
        "attention",
        "attention",
        inputs={"q": "input:q", "k": "input:k", "v": "input:v"},
        params={
            "causal": True,
            "window": 2,
            "alibi_slopes": [0.1, 0.2, 0.3, 0.4],
        },
    )
    output = "attention"
    graph.set_output(output, reference)
    package = WeightPackage(tensors={}, metas={}, metadata={"test": True})
    path, _ = compile_ptx()
    driver = CudaDriver()
    with driver.context(0) as context:
        module = context.load_module(path.read_text(encoding="utf-8"))
        allocator = DeviceAllocator(context)
        executor = CudaExecutor(
            context,
            module,
            allocator,
            weight_cache={},
            package_fingerprint=package.fingerprint(),
        )
        inputs = {
            "q": rng.standard_normal((2, 4, 3, 8), dtype=np.float32),
            "k": rng.standard_normal((2, 4, 5, 8), dtype=np.float32),
            "v": rng.standard_normal((2, 4, 5, 8), dtype=np.float32),
        }
        execution = executor.execute(graph, inputs, package)
        expected = numpy_execute(graph, inputs, package).outputs[output]
        np.testing.assert_allclose(
            execution.outputs[output], expected, rtol=3e-4, atol=2e-4
        )
        allocator.close()
