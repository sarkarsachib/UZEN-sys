"""Device memory allocation with size-class pooling and transfer accounting.

Repeated activations of the same shape are the common case in a transformer, so
freed blocks are kept in size-class buckets instead of being returned to the driver
immediately. The allocator never hands out a block smaller than requested, never
frees a block twice, and reports live/peak/reserved bytes plus H2D/D2H transfer
volume so a caller can see the real cost of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .driver import CudaContext, DeviceBuffer


def _round_bytes(nbytes: int, alignment: int = 512) -> int:
    if nbytes <= 0:
        raise ValueError("allocation size must be positive")
    return max(alignment, (nbytes + alignment - 1) // alignment * alignment)


@dataclass
class AllocationStats:
    allocations: int = 0
    frees: int = 0
    reused: int = 0
    live_bytes: int = 0
    peak_live_bytes: int = 0
    reserved_bytes: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    uploads: int = 0
    downloads: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": self.allocations,
            "frees": self.frees,
            "reused_blocks": self.reused,
            "live_bytes": self.live_bytes,
            "peak_live_bytes": self.peak_live_bytes,
            "reserved_bytes": self.reserved_bytes,
            "host_to_device_bytes": self.host_to_device_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "uploads": self.uploads,
            "downloads": self.downloads,
        }


class DeviceAllocator:
    def __init__(self, context: CudaContext, *, max_live_bytes: int | None = None) -> None:
        self.context = context
        self.max_live_bytes = max_live_bytes
        self.stats = AllocationStats()
        self._free: dict[int, list[DeviceBuffer]] = {}
        self._live: dict[int, DeviceBuffer] = {}
        self._closed = False

    def allocate(self, nbytes: int) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("device allocator is closed")
        capacity = _round_bytes(nbytes)
        bucket = self._free.get(capacity)
        if bucket:
            buffer = bucket.pop()
            self.stats.reused += 1
        else:
            if (
                self.max_live_bytes is not None
                and self.stats.reserved_bytes + capacity > self.max_live_bytes
            ):
                raise MemoryError(
                    f"device allocation of {capacity} bytes would exceed the "
                    f"configured limit of {self.max_live_bytes} bytes "
                    f"(reserved {self.stats.reserved_bytes})"
                )
            buffer = self.context.allocate(capacity)
            buffer.capacity = capacity
            self.stats.allocations += 1
            self.stats.reserved_bytes += capacity
        buffer.nbytes = int(nbytes)
        self._live[buffer.ptr] = buffer
        self.stats.live_bytes += buffer.capacity
        self.stats.peak_live_bytes = max(self.stats.peak_live_bytes, self.stats.live_bytes)
        return buffer

    def free(self, buffer: DeviceBuffer) -> None:
        if self._closed:
            return
        live = self._live.pop(buffer.ptr, None)
        if live is None:
            raise RuntimeError(f"device buffer 0x{buffer.ptr:x} is not live")
        self.stats.live_bytes -= live.capacity
        self.stats.frees += 1
        self._free.setdefault(live.capacity, []).append(live)

    def upload(self, array: Any) -> DeviceBuffer:
        """Upload a contiguous host array into a pooled device block."""
        value = np.ascontiguousarray(array)
        buffer = self.allocate(int(value.nbytes))
        if value.nbytes:
            self.context.driver._check(
                self.context.driver._lib.cuMemcpyHtoD_v2(
                    buffer.ptr, value.ctypes.data, int(value.nbytes)
                ),
                "cuMemcpyHtoD",
            )
        self.stats.host_to_device_bytes += int(value.nbytes)
        self.stats.uploads += 1
        return buffer

    upload_array = upload

    def download_array(
        self, buffer: DeviceBuffer, shape: tuple[int, ...], dtype: Any = np.float32
    ) -> np.ndarray:
        value = np.empty(shape, dtype=dtype)
        if value.nbytes:
            self.context.driver._check(
                self.context.driver._lib.cuMemcpyDtoH_v2(
                    value.ctypes.data, buffer.ptr, int(value.nbytes)
                ),
                "cuMemcpyDtoH",
            )
        self.stats.device_to_host_bytes += int(value.nbytes)
        self.stats.downloads += 1
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            **self.stats.to_dict(),
            "cached_blocks": sum(len(blocks) for blocks in self._free.values()),
            "live_blocks": len(self._live),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for buffer in list(self._live.values()):
            try:
                self.context.free(buffer)
            except Exception:  # pragma: no cover - driver teardown best effort
                pass
        self._live.clear()
        for bucket in self._free.values():
            for buffer in bucket:
                try:
                    self.context.free(buffer)
                except Exception:  # pragma: no cover
                    pass
        self._free.clear()
        self.stats.live_bytes = 0
        self.stats.reserved_bytes = 0

    def __enter__(self) -> "DeviceAllocator":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
