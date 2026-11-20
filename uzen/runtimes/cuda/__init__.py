"""CUDA runtime package (N5).

The kernels are real CUDA source in ``kernels.cu``; they are compiled to PTX with
NVRTC (or nvcc) and launched through the CUDA driver API. Importing this package
does not probe a device; use :func:`cuda_available` or :class:`CudaRuntime`.

``CudaRuntime`` refuses to run without a device unless ``fallback="cpu"`` is
passed explicitly, in which case it runs the NumPy reference and reports that it
did so. Compiling the kernels needs NVRTC or nvcc, not a GPU; executing them needs
both a driver and a device.
"""

from .driver import CudaDeviceInfo, cuda_available, device_snapshot
from .executor import CudaExecution
from .nvrtc import KERNELS, compile_ptx, describe_compilers
from .runtime import CudaRuntime

__all__ = [
    "CudaDeviceInfo",
    "CudaExecution",
    "CudaRuntime",
    "KERNELS",
    "compile_ptx",
    "cuda_available",
    "describe_compilers",
    "device_snapshot",
]
