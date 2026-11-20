"""CUDA driver API binding (library only; no kernels live here).

The runtime talks to ``libcuda`` through ctypes exactly as it talks to the C++
CPU kernels: no generated wrappers, no hidden framework. PTX is loaded as a
module, functions are looked up by name and launched with explicit argument
buffers.

A machine without a driver or device is a normal, typed condition: ``cuInit``
returns an error and :func:`cuda_available` is ``False``. Nothing here pretends a
GPU exists.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from uzen.errors import CudaKernelError, RuntimeUnavailable

CUDA_SUCCESS = 0
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101

# cuDeviceGetAttribute ids.
ATTR_MULTIPROCESSORS = 16
ATTR_COMPUTE_CAPABILITY_MAJOR = 75
ATTR_COMPUTE_CAPABILITY_MINOR = 76


def _candidate_driver_paths() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("UZEN_CUDA_DRIVER")
    if explicit:
        candidates.append(explicit)
    discovered = ctypes.util.find_library("cuda")
    if discovered:
        candidates.append(discovered)
    patterns = (
        "/usr/lib/x86_64-linux-gnu/libcuda.so*",
        "/usr/lib64/libcuda.so*",
        "/usr/local/cuda/lib64/libcuda.so*",
    )
    for pattern in patterns:
        candidates.extend(str(path) for path in sorted(Path("/").glob(pattern.lstrip("/"))))
    return candidates


def find_driver() -> str | None:
    for candidate in _candidate_driver_paths():
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _error_name(status: int) -> str:
    return {
        0: "CUDA_SUCCESS",
        1: "CUDA_ERROR_INVALID_VALUE",
        2: "CUDA_ERROR_OUT_OF_MEMORY",
        100: "CUDA_ERROR_NO_DEVICE",
        101: "CUDA_ERROR_INVALID_DEVICE",
        209: "CUDA_ERROR_NO_BINARY_FOR_GPU",
        218: "CUDA_ERROR_INVALID_PTX",
        400: "CUDA_ERROR_INVALID_HANDLE",
        700: "CUDA_ERROR_ILLEGAL_ADDRESS",
        719: "CUDA_ERROR_LAUNCH_FAILED",
    }.get(status, f"CUDA_ERROR_{status}")


class Arg:
    """One kernel argument with an explicit CUDA C type."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value: Any) -> None:
        self.kind = kind
        self.value = value

    def holder(self) -> Any:
        if self.kind == "f32":
            return ctypes.c_float(float(self.value))
        if self.kind == "i32":
            return ctypes.c_int(int(self.value))
        if self.kind == "i64":
            return ctypes.c_longlong(int(self.value))
        if self.kind == "ptr":
            pointer = self.value.ptr if isinstance(self.value, DeviceBuffer) else int(self.value)
            return ctypes.c_uint64(pointer)
        if self.kind == "ptr_null":
            return ctypes.c_uint64(0)
        raise TypeError(f"unknown CUDA argument kind '{self.kind}'")


def f32(value: float) -> Arg:
    return Arg("f32", value)


def i32(value: int) -> Arg:
    return Arg("i32", value)


def i64(value: int) -> Arg:
    return Arg("i64", value)


def ptr(value: Any) -> Arg:
    return Arg("ptr", value)


def null_ptr() -> Arg:
    return Arg("ptr_null", None)


@dataclass
class DeviceBuffer:
    """An owned device allocation. Freed through its allocator, never by GC."""

    ptr: int
    nbytes: int
    capacity: int = 0

    def __len__(self) -> int:
        return self.nbytes


@dataclass(frozen=True)
class CudaDeviceInfo:
    index: int
    name: str
    compute_capability: tuple[int, int]
    multiprocessors: int
    memory_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "multiprocessors": self.multiprocessors,
            "memory_bytes": self.memory_bytes,
        }


class CudaDriver:
    def __init__(self, library: str | None = None) -> None:
        path = library or find_driver()
        if not path:
            raise RuntimeUnavailable(
                "the CUDA driver library (libcuda) was not found; install a GPU "
                "driver or set UZEN_CUDA_DRIVER",
                searched=[str(item) for item in _candidate_driver_paths()],
            )
        self.path = path
        try:
            self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeUnavailable(f"cannot load the CUDA driver at {path}: {exc}") from exc
        self._bind()
        self._initialized = False
        self.init_error: str | None = None

    def _bind(self) -> None:
        lib = self._lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDriverGetVersion.restype = ctypes.c_int
        lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuDeviceGetCount.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGetName.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cuDeviceGetAttribute.restype = ctypes.c_int
        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        lib.cuDeviceTotalMem_v2.restype = ctypes.c_int
        lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        lib.cuCtxCreate_v2.restype = ctypes.c_int
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuCtxDestroy_v2.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        lib.cuCtxSynchronize.argtypes = []
        lib.cuCtxSynchronize.restype = ctypes.c_int
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuModuleLoadData.restype = ctypes.c_int
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleUnload.restype = ctypes.c_int
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.cuModuleGetFunction.restype = ctypes.c_int
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemAlloc_v2.restype = ctypes.c_int
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemFree_v2.restype = ctypes.c_int
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyHtoD_v2.restype = ctypes.c_int
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuLaunchKernel.restype = ctypes.c_int

    # -- lifecycle ----------------------------------------------------- #

    def initialize(self) -> bool:
        """Call ``cuInit`` once; returns whether a usable driver is present."""
        if self._initialized:
            return self.init_error is None
        self._initialized = True
        status = int(self._lib.cuInit(0))
        if status != CUDA_SUCCESS:
            self.init_error = _error_name(status)
            return False
        return True

    def driver_version(self) -> int:
        version = ctypes.c_int()
        if int(self._lib.cuDriverGetVersion(ctypes.byref(version))) != CUDA_SUCCESS:
            return 0
        return int(version.value)

    def device_count(self) -> int:
        if not self.initialize():
            return 0
        count = ctypes.c_int()
        status = int(self._lib.cuDeviceGetCount(ctypes.byref(count)))
        if status != CUDA_SUCCESS:
            if status == CUDA_ERROR_NO_DEVICE:
                return 0
            raise CudaKernelError(
                "cuDeviceGetCount failed", status=status, name=_error_name(status)
            )
        return int(count.value)

    def devices(self) -> list[CudaDeviceInfo]:
        count = self.device_count()
        return [self.device_info(index) for index in range(count)]

    def device_info(self, index: int = 0) -> CudaDeviceInfo:
        if not self.initialize():
            raise RuntimeUnavailable(
                "the CUDA driver initialized no device", driver=self.path,
                error=self.init_error,
            )
        count = self.device_count()
        if index < 0 or index >= count:
            raise CudaKernelError(
                f"CUDA device {index} is not present (count={count})",
                device_index=index,
                count=count,
            )
        handle = ctypes.c_int()
        self._check(self._lib.cuDeviceGet(ctypes.byref(handle), index), "cuDeviceGet")

        name_buffer = ctypes.create_string_buffer(256)
        self._check(
            self._lib.cuDeviceGetName(name_buffer, 256, handle), "cuDeviceGetName"
        )
        major = ctypes.c_int()
        minor = ctypes.c_int()
        multiprocessors = ctypes.c_int()
        self._check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(major), ATTR_COMPUTE_CAPABILITY_MAJOR, handle
            ),
            "cuDeviceGetAttribute(major)",
        )
        self._check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(minor), ATTR_COMPUTE_CAPABILITY_MINOR, handle
            ),
            "cuDeviceGetAttribute(minor)",
        )
        self._check(
            self._lib.cuDeviceGetAttribute(
                ctypes.byref(multiprocessors), ATTR_MULTIPROCESSORS, handle
            ),
            "cuDeviceGetAttribute(multiprocessors)",
        )
        memory = ctypes.c_size_t()
        self._check(
            self._lib.cuDeviceTotalMem_v2(ctypes.byref(memory), handle),
            "cuDeviceTotalMem",
        )
        return CudaDeviceInfo(
            index=index,
            name=name_buffer.value.decode("utf-8", errors="replace"),
            compute_capability=(int(major.value), int(minor.value)),
            multiprocessors=int(multiprocessors.value),
            memory_bytes=int(memory.value),
        )

    def context(self, index: int = 0) -> "CudaContext":
        self.device_info(index)
        handle = ctypes.c_void_p()
        self._check(
            self._lib.cuCtxCreate_v2(ctypes.byref(handle), 0, index),
            "cuCtxCreate",
        )
        return CudaContext(self, handle, index)

    def _check(self, status: int, operation: str) -> None:
        if int(status) != CUDA_SUCCESS:
            raise CudaKernelError(
                f"{operation} failed", status=int(status), name=_error_name(int(status))
            )


class CudaContext:
    """One driver context, module and allocation lifetime."""

    def __init__(self, driver: CudaDriver, handle: ctypes.c_void_p, index: int) -> None:
        self.driver = driver
        self.handle = handle
        self.index = index
        self._modules: list[ctypes.c_void_p] = []
        self._closed = False

    def __enter__(self) -> "CudaContext":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def make_current(self) -> None:
        self.driver._check(
            self.driver._lib.cuCtxSetCurrent(self.handle), "cuCtxSetCurrent"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for module in self._modules:
            self.driver._lib.cuModuleUnload(module)
        self._modules.clear()
        self.driver._lib.cuCtxDestroy_v2(self.handle)

    def load_module(self, ptx: str) -> "CudaModule":
        data = ptx.encode("utf-8")
        buffer = ctypes.create_string_buffer(data)
        module = ctypes.c_void_p()
        self.driver._check(
            self.driver._lib.cuModuleLoadData(ctypes.byref(module), buffer),
            "cuModuleLoadData",
        )
        self._modules.append(module)
        return CudaModule(self, module)

    def allocate(self, nbytes: int) -> DeviceBuffer:
        if nbytes <= 0:
            raise ValueError("device allocation must be positive")
        pointer = ctypes.c_uint64()
        self.driver._check(
            self.driver._lib.cuMemAlloc_v2(ctypes.byref(pointer), int(nbytes)),
            "cuMemAlloc",
        )
        return DeviceBuffer(ptr=int(pointer.value), nbytes=int(nbytes))

    def free(self, buffer: DeviceBuffer) -> None:
        self.driver._check(
            self.driver._lib.cuMemFree_v2(ctypes.c_uint64(buffer.ptr)), "cuMemFree"
        )

    def upload(self, source: Any) -> DeviceBuffer:
        pointer, nbytes = _as_host_buffer(source)
        buffer = self.allocate(nbytes)
        self.driver._check(
            self.driver._lib.cuMemcpyHtoD_v2(
                ctypes.c_uint64(buffer.ptr), pointer, ctypes.c_size_t(nbytes)
            ),
            "cuMemcpyHtoD",
        )
        return buffer

    def download(self, buffer: DeviceBuffer, target: Any) -> None:
        pointer, nbytes = _as_host_buffer(target)
        if nbytes != buffer.nbytes:
            raise ValueError(
                f"download target holds {nbytes} bytes, device buffer holds {buffer.nbytes}"
            )
        self.driver._check(
            self.driver._lib.cuMemcpyDtoH_v2(
                pointer, ctypes.c_uint64(buffer.ptr), ctypes.c_size_t(nbytes)
            ),
            "cuMemcpyDtoH",
        )

    def synchronize(self) -> None:
        self.driver._check(self.driver._lib.cuCtxSynchronize(), "cuCtxSynchronize")


class CudaModule:
    def __init__(self, context: CudaContext, handle: ctypes.c_void_p) -> None:
        self.context = context
        self.handle = handle

    def function(self, name: str) -> "CudaFunction":
        handle = ctypes.c_void_p()
        self.context.driver._check(
            self.context.driver._lib.cuModuleGetFunction(
                ctypes.byref(handle), self.handle, name.encode("utf-8")
            ),
            f"cuModuleGetFunction({name})",
        )
        return CudaFunction(self.context, handle, name)


class CudaFunction:
    def __init__(self, context: CudaContext, handle: ctypes.c_void_p, name: str) -> None:
        self.context = context
        self.handle = handle
        self.name = name

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        arguments: Iterable[Arg],
        *,
        shared_bytes: int = 0,
    ) -> None:
        holders = [argument.holder() for argument in arguments]
        pointers = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(holder), ctypes.c_void_p) for holder in holders]
        )
        status = self.context.driver._lib.cuLaunchKernel(
            self.handle,
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(block[0]),
            int(block[1]),
            int(block[2]),
            int(shared_bytes),
            None,
            pointers,
            None,
        )
        self.context.driver._check(status, f"cuLaunchKernel({self.name})")


def _as_host_buffer(array: Any) -> tuple[ctypes.c_void_p, int]:
    import numpy as np

    value = np.ascontiguousarray(array)
    return (
        ctypes.c_void_p(value.ctypes.data),
        int(value.nbytes),
    )


def cuda_available() -> bool:
    """True only when a driver initializes and reports at least one device."""
    try:
        driver = CudaDriver()
    except RuntimeUnavailable:
        return False
    try:
        return driver.device_count() > 0
    except CudaKernelError:
        return False


def device_snapshot() -> dict[str, Any]:
    """Safe capability probe: never raises for a missing driver/device."""
    try:
        driver = CudaDriver()
    except RuntimeUnavailable as exc:
        return {
            "available": False,
            "reason": str(exc),
            "driver": None,
            "devices": [],
        }
    available = False
    devices: list[dict[str, Any]] = []
    reason = driver.init_error
    try:
        available = driver.device_count() > 0
        if available:
            devices = [device.to_dict() for device in driver.devices()]
        elif reason is None:
            reason = "the driver initialized but reported no CUDA device"
    except CudaKernelError as exc:
        reason = str(exc)
    return {
        "available": available,
        "reason": reason,
        "driver": driver.path,
        "driver_version": driver.driver_version(),
        "devices": devices,
    }
