"""Native CPU runtime adapter.

The adapter uses the real C++ kernels for graph execution and delegates artifact
IO to the NumPy runtime's safe safetensors/NPZ/GGUF codecs. It advertises only
what it can execute: float32 native arithmetic, with a documented operator set.

It is not a GPU backend and makes no CUDA claims. ``backend="native"`` never
silently falls back; unsupported operators raise :class:`UnsupportedOperator`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.runtimes.base import RuntimeCapabilities
from uzen.runtimes.numpy_runtime import NumpyRuntime

from .compiler import PLAN_OPS, NativePlan, compile_native_plan
from .executor import NativeExecution, execute_native
from .library import NativeLibrary, native_available, resolve_library_path
from .plan_executor import NativePlanExecution, execute_plan
from .representations import REPRESENTATION_KINDS, REPRESENTATION_NOTES


class NativeRuntime:
    """Real native-CPU execution with explicit coverage and no fallback."""

    name = "native"

    def __init__(
        self,
        *,
        library_path: str | os.PathLike[str] | None = None,
        auto_build: bool = True,
        num_threads: int | None = None,
    ) -> None:
        self.num_threads = num_threads
        if num_threads is not None:
            if num_threads <= 0:
                raise ValueError("num_threads must be positive")
            # OpenMP reads this before its first parallel region. Set it before
            # loading the library so the explicit runtime option is honored.
            os.environ["OMP_NUM_THREADS"] = str(int(num_threads))
        self.library = NativeLibrary(library_path, auto_build=auto_build)
        self._io = NumpyRuntime()

    # ------------------------------------------------------------------ #
    # capabilities
    # ------------------------------------------------------------------ #

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            name=self.name,
            backend="native-cpp",
            device="cpu",
            dtypes=("float32",),
            supports_autograd=False,
            supports_safetensors=True,
            supports_torch_checkpoints=False,
            supports_onnx=False,
            supports_gguf=True,
            extra={
                "library": str(self.library.path),
                "execution": "C++ kernels through ctypes",
                "native_ops": [
                    "linear",
                    "matmul",
                    "layernorm",
                    "rmsnorm",
                    "softmax",
                    "gelu",
                    "gelu_exact",
                    "silu",
                    "relu",
                    "tanh",
                    "sigmoid",
                    "add",
                    "sub",
                    "mul",
                    "scale",
                    "clip",
                    "embedding",
                    "mean",
                    "sum",
                    "attention",
                ],
                "layout_ops": [
                    "reshape",
                    "transpose",
                    "slice",
                    "flatten",
                    "concat",
                    "split_qkv",
                    "merge_heads",
                    "rope",
                    "repeat_kv",
                    "pos_embedding",
                    "identity",
                    "dropout",
                    "select",
                ],
                "float32_only": True,
                "compiled_plan": True,
                "plan_ops": sorted(PLAN_OPS),
                "plan_fusions": [
                    "linear + bias + activation",
                    "residual add + layernorm/rmsnorm",
                ],
                "prepacked_gemm": True,
                "representations": list(REPRESENTATION_KINDS),
                "representation_notes": dict(REPRESENTATION_NOTES),
                "representation_aware_gemm": True,
                "threaded_attention": "single-pass (flash-style) online softmax",
                "openmp": "optional; enabled when the compiler accepts -fopenmp",
                "cuda": False,
                "note": (
                    "native CPU arithmetic; layout operators use NumPy for byte "
                    "movement, and unsupported operators raise rather than fall back"
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # artifact IO (shared safe codecs)
    # ------------------------------------------------------------------ #

    def load(self, path: str | os.PathLike[str]) -> WeightPackage:
        return self._io.load(path)

    def save(
        self,
        package: WeightPackage,
        path: str | os.PathLike[str],
        *,
        quantize: str | None = None,
    ) -> str:
        return self._io.save(package, path, quantize=quantize)

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #

    def execute(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        return dict(
            execute_native(
                graph,
                inputs,
                package,
                capture=capture,
                library=self.library,
            ).outputs
        )

    def execute_with_intermediates(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return execute_native(
            graph,
            inputs,
            package,
            capture=capture,
            library=self.library,
        ).intermediates

    def execute_graph(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
    ) -> NativeExecution:
        """Return outputs plus the native/layout operator evidence."""
        return execute_native(
            graph, inputs, package, capture=capture, library=self.library
        )

    # ------------------------------------------------------------------ #
    # compiled plans
    # ------------------------------------------------------------------ #

    def compile(
        self,
        graph: Graph,
        package: WeightPackage | None = None,
        *,
        fuse: bool = True,
        representations: Mapping[str, Any] | None = None,
    ) -> NativePlan:
        """Lower a graph into a native execution plan (fusions + representation)."""
        return compile_native_plan(
            graph, package, fuse=fuse, representations=representations
        )

    def execute_compiled_graph(
        self,
        plan: NativePlan,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
    ) -> NativePlanExecution:
        """Execute a previously compiled plan (prepacked weights are cached)."""
        return execute_plan(
            plan,
            inputs,
            package,
            capture=capture,
            library=self.library,
        )

    def execute_compiled(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
        fuse: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compile and run a graph, returning only the outputs."""
        plan = self.compile(graph, package, fuse=fuse)
        return dict(
            self.execute_compiled_graph(
                plan, inputs, package, capture=capture
            ).outputs
        )

    def describe(self) -> dict[str, Any]:
        described = self.capabilities().to_dict()
        described["kernel_library"] = self.library.describe()
        return described


def native_runtime_available() -> bool:
    return native_available()


def native_library_path() -> Path:
    return resolve_library_path()
