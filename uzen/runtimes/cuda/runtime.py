"""CUDA runtime adapter (N5).

The policy is explicit:

* ``CudaRuntime(fallback="error")`` (the default) refuses to run when no CUDA
  device, driver or compiler is available. It never quietly uses the CPU.
* ``CudaRuntime(fallback="cpu")`` runs the NumPy reference instead and records the
  exact reason in every execution summary. That is a documented fallback, not a
  CUDA execution, and capabilities say ``device="cpu"``.

Kernels are compiled from the checked-in ``kernels.cu`` to PTX with NVRTC (or
nvcc), loaded once through the driver API and reused. Weights stay resident for
the life of a package; a different package fingerprint rebuilds the executor and
its weight cache. Execution is measured, not assumed: ``execution_verified`` only
becomes true after a graph has actually run on the device in this process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.executor import execute as numpy_execute
from uzen.core.nir.graph import Graph
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import RuntimeUnavailable
from uzen.runtimes.base import RuntimeCapabilities
from uzen.runtimes.numpy_runtime import NumpyRuntime

from .allocator import DeviceAllocator
from .driver import CudaDriver, CudaDeviceInfo, cuda_available, device_snapshot
from .executor import DEVICE_OPS, LAYOUT_OPS, CudaExecution, CudaExecutor
from .nvrtc import KERNELS, compile_ptx, describe_compilers

__all__ = ["CudaRuntime", "cuda_available", "device_snapshot"]


class CudaRuntime:
    name = "cuda"

    def __init__(
        self,
        *,
        device_index: int = 0,
        fallback: str = "error",
        ptx_path: str | os.PathLike[str] | None = None,
        arch: str | None = None,
        auto_compile: bool = True,
        max_device_bytes: int | None = None,
    ) -> None:
        if fallback not in ("error", "cpu"):
            raise ValueError("fallback must be 'error' or 'cpu'")
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        self.device_index = device_index
        self.fallback = fallback
        self.arch = arch
        self._io = NumpyRuntime()
        self._snapshot = device_snapshot()
        self.available = bool(self._snapshot.get("available"))
        self.reason = self._snapshot.get("reason")
        self._context: Any = None
        self._module: Any = None
        self._allocator: DeviceAllocator | None = None
        self._executor: CudaExecutor | None = None
        self._weights: dict[tuple[str, str, str], Any] = {}
        self._package_fingerprint: str | None = None
        self._executions = 0
        self.last_execution: dict[str, Any] | None = None
        self._ptx_path: Path | None = None
        self._compile_report: dict[str, Any] | None = None
        self._device_info: CudaDeviceInfo | None = None
        self._closed = False

        if not self.available:
            if fallback == "error":
                raise RuntimeUnavailable(
                    "no CUDA device is available for the cuda runtime",
                    reason=self.reason,
                    device_index=device_index,
                    hint="pass fallback='cpu' to run the NumPy reference explicitly",
                )
            return

        if ptx_path is not None:
            path = Path(ptx_path)
            if not path.exists():
                raise RuntimeUnavailable(f"PTX file not found: {path}")
            self._ptx_path = path
            self._compile_report = {
                "compiler": "prebuilt",
                "path": str(path),
                "cache_hit": True,
                "bytes": path.stat().st_size,
            }
        else:
            if not auto_compile:
                raise RuntimeUnavailable(
                    "auto_compile=False but no ptx_path was supplied"
                )
            path, report = compile_ptx(arch=arch)
            self._ptx_path = path
            self._compile_report = report

        driver = CudaDriver()
        self._device_info = driver.device_info(device_index)
        self._context = driver.context(device_index)
        self._module = self._context.load_module(
            self._ptx_path.read_text(encoding="utf-8")
        )
        self._allocator = DeviceAllocator(self._context, max_live_bytes=max_device_bytes)

    # ------------------------------------------------------------------ #
    # capabilities and IO
    # ------------------------------------------------------------------ #

    def capabilities(self) -> RuntimeCapabilities:
        if not self.available:
            return RuntimeCapabilities(
                name=self.name,
                backend="numpy-fallback",
                device="cpu",
                dtypes=("float32", "float64", "float16", "int64"),
                supports_autograd=False,
                extra={
                    "cuda": False,
                    "reason": self.reason,
                    "fallback": "cpu",
                    "devices": [],
                    "driver": self._snapshot.get("driver"),
                    "compilers": describe_compilers(),
                    "kernels": list(KERNELS),
                    "execution_verified": False,
                    "note": (
                        "no CUDA device: this instance runs the NumPy reference, "
                        "not CUDA; no acceleration is claimed"
                    ),
                },
            )
        info = self._device_info
        return RuntimeCapabilities(
            name=self.name,
            backend="cuda-driver",
            device=f"cuda:{self.device_index}",
            dtypes=("float32",),
            supports_autograd=False,
            supports_safetensors=True,
            supports_torch_checkpoints=False,
            supports_onnx=False,
            supports_gguf=True,
            extra={
                "cuda": True,
                "device": info.to_dict() if info else None,
                "devices": self._snapshot.get("devices", []),
                "driver_version": self._snapshot.get("driver_version"),
                "compiler": (self._compile_report or {}).get("compiler"),
                "compiler_version": (self._compile_report or {}).get("compiler_version"),
                "ptx": str(self._ptx_path),
                "ptx_bytes": (self._compile_report or {}).get("bytes"),
                "kernels": list(KERNELS),
                "device_ops": sorted(DEVICE_OPS),
                "layout_ops": sorted(LAYOUT_OPS),
                "float32_only": True,
                "layout_ops_on_host": True,
                "execution_verified": self._executions > 0,
                "executions": self._executions,
                "last_execution": self.last_execution,
                "note": (
                    "real CUDA kernels launched through the driver API; layout "
                    "operators move bytes on the host and are reported separately"
                ),
            },
        )

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

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": self.name,
            "available": self.available,
            "reason": self.reason,
            "fallback": self.fallback,
            "device_index": self.device_index,
            "device": self._device_info.to_dict() if self._device_info else None,
            "devices": self._snapshot.get("devices", []),
            "driver": self._snapshot.get("driver"),
            "driver_version": self._snapshot.get("driver_version"),
            "compilers": describe_compilers(),
            "ptx": str(self._ptx_path) if self._ptx_path else None,
            "compile": self._compile_report,
            "kernels": list(KERNELS),
            "executions": self._executions,
            "execution_verified": self._executions > 0,
            "memory": self._allocator.snapshot() if self._allocator else None,
        }

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
            self.execute_graph(
                graph, inputs, package, capture=capture
            ).outputs
        )

    def execute_graph(
        self,
        graph: Graph,
        inputs: Mapping[str, np.ndarray],
        package: WeightPackage,
        *,
        capture: Sequence[str] | None = None,
        check_finite: bool = True,
    ) -> CudaExecution:
        if not self.available:
            if self.fallback != "cpu":
                raise RuntimeUnavailable(
                    "the cuda runtime has no device and fallback is disabled",
                    reason=self.reason,
                )
            result = numpy_execute(graph, inputs, package, capture=capture)
            return CudaExecution(
                outputs=dict(result.outputs),
                intermediates=dict(result.intermediates),
                order=list(result.order),
                num_ops=result.num_ops,
                backend="numpy-fallback",
                device="cpu",
                reason=self.reason,
                stats={"fallback": True},
            )

        assert self._allocator is not None
        assert self._context is not None
        assert self._module is not None
        fingerprint = package.fingerprint()
        if self._executor is None or self._package_fingerprint != fingerprint:
            if self._executor is not None:
                for buffer in self._weights.values():
                    self._allocator.free(buffer)
                self._weights.clear()
            self._executor = CudaExecutor(
                self._context,
                self._module,
                self._allocator,
                weight_cache=self._weights,
                package_fingerprint=fingerprint,
                device_label=f"cuda:{self.device_index}",
            )
            self._package_fingerprint = fingerprint

        execution = self._executor.execute(
            graph,
            inputs,
            package,
            capture=capture,
            check_finite=check_finite,
        )
        self._executions += 1
        self.last_execution = execution.summary()
        return execution

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._allocator is not None:
            self._allocator.close()
        if self._context is not None:
            self._context.close()
        self._executor = None
        self._allocator = None
        self._context = None
        self._module = None

    def __enter__(self) -> "CudaRuntime":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
