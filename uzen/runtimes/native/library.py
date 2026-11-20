"""ctypes bridge to UZEN's native C++ kernels.

The library is discovered from, in order:

1. an explicit ``path`` argument,
2. ``UZEN_NATIVE_LIBRARY``,
3. a library shipped next to this module,
4. the per-user build cache,
5. an on-demand build with the local C++ compiler.

Every kernel validates the rank/shape/dtype contract in Python and then calls the
C ABI with C-contiguous float32 (or int64 index) buffers. A non-zero kernel status
becomes :class:`uzen.errors.NativeKernelError`; there is no silent fallback.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uzen.errors import (
    InvalidOperationParameters,
    NativeKernelError,
    RuntimeUnavailable,
)

from .build import (
    build_native_library,
    default_cache_path,
    library_suffix,
    native_compiler,
)

_C_INT = ctypes.c_int
_C_FLOAT = ctypes.c_float
_C_VOID_P = ctypes.c_void_p
_C_INT64 = ctypes.c_longlong
_C_INT8_P = ctypes.c_void_p

# Block sizes of the prepacked GEMM (must match plan_kernels.cpp).
BLOCK_N = 64
BLOCK_K = 64

ACTIVATION_CODES = {
    "none": 0,
    "relu": 1,
    "gelu": 2,
    "silu": 3,
    "tanh": 4,
    "sigmoid": 5,
    "gelu_exact": 6,
}

RECURRENT_GATES = {"rnn": 1, "lstm": 4, "gru": 3}
RECURRENT_KINDS = {"rnn": 0, "lstm": 1, "gru": 2}


def _pair(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return (int(value[0]), int(value[1]))
    raise InvalidOperationParameters(
        f"native {label} must be an int or [h, w], got {value!r}"
    )


@dataclass(frozen=True)
class PackedWeight:
    """A weight matrix in the blocked layout ``uzen_gemm_packed_f32`` consumes.

    Partial blocks are zero-padded, so the buffer is slightly larger than
    ``out_features * in_features``; :attr:`padding_overhead` reports the ratio
    instead of hiding it.
    """

    array: np.ndarray
    in_features: int
    out_features: int

    @property
    def padded_elements(self) -> int:
        return int(self.array.size)

    @property
    def padding_overhead(self) -> float:
        source = self.in_features * self.out_features
        return self.padded_elements / float(source) if source else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "weight_shape": [self.out_features, self.in_features],
            "block_in": BLOCK_K,
            "block_out": BLOCK_N,
            "padded_elements": self.padded_elements,
            "padding_overhead": round(self.padding_overhead, 3),
        }


#: Group size of the block-quantized layouts (must match plan_kernels.cpp).
QUANT_GROUP = BLOCK_K
INT8_BLOCK_RANGE = 127
INT4_BLOCK_RANGE = 7


@dataclass(frozen=True)
class PackedBlockWeight:
    """Group-wise quantized weight in the blocked GEMM panel layout.

    ``weight`` holds the quantized values (int8, or packed int4 nibbles) in
    ``kBlockK x kBlockN`` panels; ``scales`` holds one float32 scale per
    ``(output feature, group)`` pair in the same panel order. Partial panels are
    padded, so the buffers are slightly larger than ``out * in``; the reported
    overhead is measured, never hidden.
    """

    kind: str
    weight: np.ndarray
    scales: np.ndarray
    in_features: int
    out_features: int
    group: int = QUANT_GROUP

    @property
    def panel_count(self) -> int:
        blocks_out = -(-self.out_features // BLOCK_N)
        blocks_in = -(-self.in_features // BLOCK_K)
        return blocks_out * blocks_in

    @property
    def dense_elements(self) -> int:
        return self.in_features * self.out_features

    @property
    def padded_elements(self) -> int:
        """Quantized value slots, including block padding (nibbles count as one)."""
        return self.panel_count * self.group * BLOCK_N

    @property
    def padding_overhead(self) -> float:
        return (
            self.padded_elements / float(self.dense_elements)
            if self.dense_elements
            else 0.0
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "weight_shape": [self.out_features, self.in_features],
            "group": self.group,
            "block_in": BLOCK_K,
            "block_out": BLOCK_N,
            "padded_elements": self.padded_elements,
            "packed_bytes": int(self.weight.nbytes + self.scales.nbytes),
            "padding_overhead": round(self.padding_overhead, 3),
        }


@dataclass(frozen=True)
class PackedTransposedWeight:
    """A (in_features, out_features) dense layout consumed by ``gemm_transposed``."""

    array: np.ndarray
    in_features: int
    out_features: int

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "transposed",
            "weight_shape": [self.out_features, self.in_features],
            "packed_layout": "in_features-major",
            "packed_bytes": int(self.array.nbytes),
            "padding_overhead": 1.0,
        }


def resolve_library_path(
    path: str | Path | None = None, *, auto_build: bool = True
) -> Path:
    """Resolve a native library path, building it on demand if allowed."""
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise RuntimeUnavailable(f"native library '{candidate}' does not exist")
        return candidate
    environment = os.environ.get("UZEN_NATIVE_LIBRARY")
    if environment:
        candidate = Path(environment)
        if not candidate.is_file():
            raise RuntimeUnavailable(
                f"UZEN_NATIVE_LIBRARY points to '{candidate}', which does not exist"
            )
        return candidate
    packaged = Path(__file__).with_name(f"libuzen_native{library_suffix()}")
    if packaged.is_file():
        return packaged
    cached = default_cache_path()
    if cached.is_file():
        return cached
    if not auto_build:
        raise RuntimeUnavailable(
            "the native kernel library is not built; run "
            "`python -m uzen.runtimes.native.build` or pass an explicit path"
        )
    return build_native_library()


def native_available() -> bool:
    """True when a native library or a usable C++ compiler is present."""
    packaged = Path(__file__).with_name(f"libuzen_native{library_suffix()}")
    environment = os.environ.get("UZEN_NATIVE_LIBRARY")
    return bool(
        (environment and Path(environment).is_file())
        or packaged.is_file()
        or default_cache_path().is_file()
        or native_compiler() is not None
    )


def _as_f32(array: Any, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.float32:
        raise InvalidOperationParameters(
            f"native kernel '{label}' requires float32 arrays, got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _as_int8(array: Any, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.int8:
        raise InvalidOperationParameters(
            f"native kernel '{label}' requires int8 arrays, got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _as_uint8(array: Any, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        raise InvalidOperationParameters(
            f"native kernel '{label}' requires uint8 arrays, got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _as_int32(array: Any, label: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.int32:
        raise InvalidOperationParameters(
            f"native kernel '{label}' requires int32 arrays, got {arr.dtype}"
        )
    return np.ascontiguousarray(arr)


def _pointer(array: np.ndarray | None) -> ctypes.c_void_p | None:
    if array is None:
        return None
    return ctypes.c_void_p(array.ctypes.data)


class NativeLibrary:
    """Loaded native kernel library with validated wrapper methods."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        auto_build: bool = True,
    ) -> None:
        self.path = resolve_library_path(path, auto_build=auto_build)
        try:
            self._lib = ctypes.CDLL(str(self.path))
        except OSError as exc:
            raise RuntimeUnavailable(
                f"cannot load native library '{self.path}': {exc}"
            ) from exc
        self._bind()

    # ------------------------------------------------------------------ #
    # binding
    # ------------------------------------------------------------------ #

    def _bind_function(self, name: str, argtypes: Sequence[Any]) -> Any:
        try:
            function = getattr(self._lib, name)
        except AttributeError as exc:
            raise RuntimeUnavailable(
                f"native library '{self.path}' does not export '{name}'; rebuild it "
                "with `python -m uzen.runtimes.native.build --force`"
            ) from exc
        function.argtypes = list(argtypes)
        function.restype = _C_INT
        return function

    def _bind(self) -> None:
        pointer = _C_VOID_P
        self._linear = self._bind_function(
            "uzen_linear_f32",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_INT],
        )
        self._matmul = self._bind_function(
            "uzen_matmul_f32",
            [pointer, pointer, pointer, _C_INT, _C_INT, _C_INT],
        )
        self._layernorm = self._bind_function(
            "uzen_layernorm_f32",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_FLOAT],
        )
        self._rmsnorm = self._bind_function(
            "uzen_rmsnorm_f32",
            [pointer, pointer, pointer, _C_INT, _C_INT, _C_FLOAT],
        )
        self._softmax = self._bind_function(
            "uzen_softmax_f32", [pointer, pointer, _C_INT, _C_INT]
        )
        self._unary = {
            "gelu": self._bind_function(
                "uzen_gelu_f32", [pointer, pointer, _C_INT]
            ),
            "gelu_exact": self._bind_function(
                "uzen_gelu_exact_f32", [pointer, pointer, _C_INT]
            ),
            "silu": self._bind_function(
                "uzen_silu_f32", [pointer, pointer, _C_INT]
            ),
            "relu": self._bind_function(
                "uzen_relu_f32", [pointer, pointer, _C_INT]
            ),
            "tanh": self._bind_function(
                "uzen_tanh_f32", [pointer, pointer, _C_INT]
            ),
            "sigmoid": self._bind_function(
                "uzen_sigmoid_f32", [pointer, pointer, _C_INT]
            ),
        }
        self._binary = {
            "add": self._bind_function(
                "uzen_add_f32", [pointer, pointer, pointer, _C_INT]
            ),
            "sub": self._bind_function(
                "uzen_sub_f32", [pointer, pointer, pointer, _C_INT]
            ),
            "mul": self._bind_function(
                "uzen_mul_f32", [pointer, pointer, pointer, _C_INT]
            ),
        }
        self._scale = self._bind_function(
            "uzen_scale_f32", [pointer, pointer, _C_INT, _C_FLOAT]
        )
        self._clip = self._bind_function(
            "uzen_clip_f32",
            [pointer, pointer, _C_INT, _C_FLOAT, _C_FLOAT],
        )
        self._embedding = self._bind_function(
            "uzen_embedding_f32",
            [pointer, pointer, pointer, _C_INT64, _C_INT],
        )
        self._reduce = self._bind_function(
            "uzen_reduce_last_f32",
            [pointer, pointer, _C_INT, _C_INT, _C_INT],
        )
        self._attention = self._bind_function(
            "uzen_attention_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_FLOAT,
                _C_INT,
                pointer,
                _C_INT,
            ],
        )
        self._attention_flash = self._bind_function(
            "uzen_attention_flash_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_FLOAT,
                _C_INT,
                pointer,
                _C_INT,
            ],
        )
        self._pack_weight = self._bind_function(
            "uzen_pack_weight_f32", [pointer, pointer, _C_INT, _C_INT]
        )
        self._packed_size = getattr(self._lib, "uzen_packed_size")
        self._packed_size.argtypes = [_C_INT, _C_INT]
        self._packed_size.restype = _C_INT
        self._gemm_packed = self._bind_function(
            "uzen_gemm_packed_f32",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_INT, _C_INT],
        )
        self._add_norm = self._bind_function(
            "uzen_add_norm_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_FLOAT,
                _C_INT,
            ],
        )
        self._conv2d = self._bind_function(
            "uzen_conv2d_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._pool2d = self._bind_function(
            "uzen_pool2d_f32",
            [
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._recurrent = self._bind_function(
            "uzen_recurrent_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._ssm = self._bind_function(
            "uzen_ssm_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._gcn = self._bind_function(
            "uzen_gcn_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._moe = self._bind_function(
            "uzen_moe_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._reduce_axis = self._bind_function(
            "uzen_reduce_axis_f32",
            [pointer, pointer, _C_INT, _C_INT, _C_INT, _C_INT],
        )
        self._quantized_linear = self._bind_function(
            "uzen_quantized_linear_f32",
            [
                pointer,
                _C_INT8_P,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        # Phase N3: alternative parameter representations.
        self._int8_block_group = getattr(self._lib, "uzen_int8_block_group")
        self._int8_block_group.argtypes = []
        self._int8_block_group.restype = _C_INT
        self._pack_int8_block = self._bind_function(
            "uzen_pack_weight_int8_block",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT],
        )
        self._gemm_int8_block = self._bind_function(
            "uzen_gemm_int8_block_f32",
            [pointer, pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_INT, _C_INT],
        )
        self._pack_int4_block = self._bind_function(
            "uzen_pack_weight_int4_block",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT],
        )
        self._gemm_int4_block = self._bind_function(
            "uzen_gemm_int4_block_f32",
            [pointer, pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_INT, _C_INT],
        )
        self._gemm_low_rank = self._bind_function(
            "uzen_gemm_low_rank_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._sparse_gemm = self._bind_function(
            "uzen_sparse_gemm_f32",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
                _C_INT,
            ],
        )
        self._pack_weight_transposed = self._bind_function(
            "uzen_pack_weight_transposed_f32", [pointer, pointer, _C_INT, _C_INT]
        )
        self._gemm_transposed = self._bind_function(
            "uzen_gemm_transposed_f32",
            [pointer, pointer, pointer, pointer, _C_INT, _C_INT, _C_INT, _C_INT],
        )

    @property
    def quant_group(self) -> int:
        """Group size of the block-quantized layouts, as compiled into C++."""
        return int(self._int8_block_group())

    def _check(self, status: int, name: str) -> None:
        if status != 0:
            raise NativeKernelError(
                f"native kernel '{name}' rejected its arguments (status {status})",
                kernel=name,
                status=int(status),
            )

    # ------------------------------------------------------------------ #
    # kernels
    # ------------------------------------------------------------------ #

    def linear(
        self,
        x: Any,
        weight: Any,
        bias: Any | None = None,
    ) -> np.ndarray:
        xa = _as_f32(x, "linear.x")
        wa = _as_f32(weight, "linear.weight")
        if xa.ndim < 1 or wa.ndim != 2:
            raise InvalidOperationParameters(
                "native linear requires x of rank >= 1 and a 2-D weight"
            )
        if xa.shape[-1] != wa.shape[1]:
            raise InvalidOperationParameters(
                f"native linear: input last dim {xa.shape[-1]} != weight in_features "
                f"{wa.shape[1]}"
            )
        out_features = int(wa.shape[0])
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        ba = None
        if bias is not None:
            ba = _as_f32(bias, "linear.bias")
            if ba.shape != (out_features,):
                raise InvalidOperationParameters(
                    f"native linear bias shape {ba.shape} != ({out_features},)"
                )
        output = np.empty((rows, out_features), dtype=np.float32)
        if rows:
            self._check(
                self._linear(
                    _pointer(xa),
                    _pointer(wa),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    int(wa.shape[1]),
                    out_features,
                ),
                "linear",
            )
        return output.reshape(tuple(xa.shape[:-1]) + (out_features,))

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        aa = _as_f32(a, "matmul.a")
        ba = _as_f32(b, "matmul.b")
        if aa.ndim != 2 or ba.ndim != 2:
            raise InvalidOperationParameters("native matmul requires two 2-D arrays")
        if aa.shape[1] != ba.shape[0]:
            raise InvalidOperationParameters(
                f"native matmul inner dims disagree: {aa.shape[1]} vs {ba.shape[0]}"
            )
        m, k, n = int(aa.shape[0]), int(aa.shape[1]), int(ba.shape[1])
        output = np.zeros((m, n), dtype=np.float32)
        if k and m and n:
            self._check(
                self._matmul(
                    _pointer(aa), _pointer(ba), _pointer(output), m, k, n
                ),
                "matmul",
            )
        return output

    def layernorm(
        self,
        x: Any,
        *,
        weight: Any | None = None,
        bias: Any | None = None,
        eps: float = 1e-5,
    ) -> np.ndarray:
        xa = _as_f32(x, "layernorm.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native layernorm requires rank >= 1")
        cols = int(xa.shape[-1])
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        wa, ba = self._norm_params(weight, bias, cols, "layernorm")
        output = np.empty(xa.shape, dtype=np.float32)
        if rows:
            self._check(
                self._layernorm(
                    _pointer(xa),
                    _pointer(wa),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    cols,
                    float(eps),
                ),
                "layernorm",
            )
        return output

    def rmsnorm(
        self,
        x: Any,
        *,
        weight: Any | None = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        xa = _as_f32(x, "rmsnorm.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native rmsnorm requires rank >= 1")
        cols = int(xa.shape[-1])
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        wa, _ = self._norm_params(weight, None, cols, "rmsnorm")
        output = np.empty(xa.shape, dtype=np.float32)
        if rows:
            self._check(
                self._rmsnorm(
                    _pointer(xa), _pointer(wa), _pointer(output), rows, cols, float(eps)
                ),
                "rmsnorm",
            )
        return output

    def _norm_params(
        self,
        weight: Any | None,
        bias: Any | None,
        cols: int,
        label: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        wa = None
        ba = None
        if weight is not None:
            wa = _as_f32(weight, f"{label}.weight")
            if wa.shape != (cols,):
                raise InvalidOperationParameters(
                    f"native {label} weight shape {wa.shape} != ({cols},)"
                )
        if bias is not None:
            ba = _as_f32(bias, f"{label}.bias")
            if ba.shape != (cols,):
                raise InvalidOperationParameters(
                    f"native {label} bias shape {ba.shape} != ({cols},)"
                )
        return wa, ba

    def softmax(self, x: Any) -> np.ndarray:
        xa = _as_f32(x, "softmax.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native softmax requires rank >= 1")
        cols = int(xa.shape[-1])
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        output = np.empty(xa.shape, dtype=np.float32)
        if rows:
            self._check(
                self._softmax(
                    _pointer(xa), _pointer(output), rows, cols
                ),
                "softmax",
            )
        return output

    def unary(self, name: str, x: Any) -> np.ndarray:
        function = self._unary.get(name)
        if function is None:
            raise InvalidOperationParameters(
                f"native unary kernel '{name}' is not implemented; available: "
                f"{', '.join(sorted(self._unary))}"
            )
        xa = _as_f32(x, f"{name}.x")
        output = np.empty(xa.shape, dtype=np.float32)
        count = int(xa.size)
        if count:
            self._check(
                function(_pointer(xa), _pointer(output), count),
                name,
            )
        return output

    def binary(self, name: str, a: Any, b: Any) -> np.ndarray:
        function = self._binary.get(name)
        if function is None:
            raise InvalidOperationParameters(
                f"native binary kernel '{name}' is not implemented; available: "
                f"{', '.join(sorted(self._binary))}"
            )
        left, right = np.broadcast_arrays(
            _as_f32(a, f"{name}.a"), _as_f32(b, f"{name}.b")
        )
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        output = np.empty(left.shape, dtype=np.float32)
        count = int(left.size)
        if count:
            self._check(
                function(
                    _pointer(left), _pointer(right), _pointer(output), count
                ),
                name,
            )
        return output

    def scale(self, x: Any, factor: float) -> np.ndarray:
        xa = _as_f32(x, "scale.x")
        output = np.empty(xa.shape, dtype=np.float32)
        count = int(xa.size)
        if count:
            self._check(
                self._scale(
                    _pointer(xa), _pointer(output), count, float(factor)
                ),
                "scale",
            )
        return output

    def clip(self, x: Any, lower: float, upper: float) -> np.ndarray:
        if lower > upper:
            raise InvalidOperationParameters("native clip min must not exceed max")
        xa = _as_f32(x, "clip.x")
        output = np.empty(xa.shape, dtype=np.float32)
        count = int(xa.size)
        if count:
            self._check(
                self._clip(
                    _pointer(xa),
                    _pointer(output),
                    count,
                    float(lower),
                    float(upper),
                ),
                "clip",
            )
        return output

    def embedding(self, indices: Any, weight: Any) -> np.ndarray:
        index_array = np.asarray(indices)
        if index_array.dtype.kind not in "iu":
            raise InvalidOperationParameters(
                f"native embedding requires integer indices, got {index_array.dtype}"
            )
        wa = _as_f32(weight, "embedding.weight")
        if wa.ndim != 2:
            raise InvalidOperationParameters("native embedding weight must be 2-D")
        flat = np.ascontiguousarray(index_array, dtype=np.int64).reshape(-1)
        if flat.size:
            minimum = int(flat.min())
            maximum = int(flat.max())
            if minimum < 0 or maximum >= wa.shape[0]:
                raise InvalidOperationParameters(
                    f"native embedding index out of range [0, {wa.shape[0]}): "
                    f"min={minimum}, max={maximum}"
                )
        output = np.empty(
            tuple(index_array.shape) + (int(wa.shape[1]),), dtype=np.float32
        )
        if flat.size:
            self._check(
                self._embedding(
                    _pointer(flat),
                    _pointer(wa),
                    _pointer(output),
                    int(flat.size),
                    int(wa.shape[1]),
                ),
                "embedding",
            )
        return output

    def reduce_last(self, x: Any, *, mode: str = "mean", keepdims: bool = False) -> np.ndarray:
        if mode not in ("mean", "sum"):
            raise InvalidOperationParameters(
                f"native reduce mode must be 'mean' or 'sum', got '{mode}'"
            )
        xa = _as_f32(x, f"{mode}.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native reduce requires rank >= 1")
        cols = int(xa.shape[-1])
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        output = np.empty((rows,), dtype=np.float32)
        if rows:
            self._check(
                self._reduce(
                    _pointer(xa),
                    _pointer(output),
                    rows,
                    cols,
                    0 if mode == "mean" else 1,
                ),
                mode,
            )
        shape = tuple(xa.shape[:-1]) + ((1,) if keepdims else ())
        return output.reshape(shape)

    def attention(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        scale: float | str | None = None,
        causal: bool = True,
        alibi: Any | None = None,
        window: int | None = None,
    ) -> np.ndarray:
        qa = _as_f32(q, "attention.q")
        ka = _as_f32(k, "attention.k")
        va = _as_f32(v, "attention.v")
        if not (qa.ndim == ka.ndim == va.ndim == 4):
            raise InvalidOperationParameters(
                "native attention requires (batch, heads, seq, head_dim) tensors"
            )
        if qa.shape[0] != ka.shape[0] or qa.shape[0] != va.shape[0]:
            raise InvalidOperationParameters("native attention batch dimensions disagree")
        if qa.shape[1] != ka.shape[1] or qa.shape[1] != va.shape[1]:
            raise InvalidOperationParameters("native attention head dimensions disagree")
        if qa.shape[3] != ka.shape[3] or qa.shape[3] != va.shape[3]:
            raise InvalidOperationParameters("native attention head_dim dimensions disagree")
        if ka.shape[2] != va.shape[2]:
            raise InvalidOperationParameters("native attention k/v sequence lengths disagree")
        if qa.shape[2] > ka.shape[2]:
            raise InvalidOperationParameters(
                "native attention q sequence cannot exceed the k/v sequence"
            )
        dimensions = int(qa.shape[3])
        if scale is None or scale == "auto":
            scale_value = 1.0 / float(np.sqrt(dimensions))
        else:
            scale_value = float(scale)
        alibi_array = None
        if alibi is not None:
            alibi_array = _as_f32(alibi, "attention.alibi")
            if alibi_array.shape != (qa.shape[1],):
                raise InvalidOperationParameters(
                    f"native attention alibi shape {alibi_array.shape} != "
                    f"({qa.shape[1]},)"
                )
        window_value = -1 if window is None else int(window)
        output = np.empty(qa.shape, dtype=np.float32)
        if int(np.prod(qa.shape[:3])):
            self._check(
                self._attention(
                    _pointer(qa),
                    _pointer(ka),
                    _pointer(va),
                    _pointer(output),
                    int(qa.shape[0]),
                    int(qa.shape[1]),
                    int(qa.shape[2]),
                    int(ka.shape[2]),
                    dimensions,
                    scale_value,
                    1 if causal else 0,
                    _pointer(alibi_array),
                    window_value,
                ),
                "attention",
            )
        return output

    # ------------------------------------------------------------------ #
    # compiled-plan kernels
    # ------------------------------------------------------------------ #

    def pack_weight(self, weight: Any) -> PackedWeight:
        """Prepack (out_features, in_features) weights for the blocked GEMM."""
        wa = _as_f32(weight, "pack_weight")
        if wa.ndim != 2:
            raise InvalidOperationParameters(
                "native pack_weight requires a 2-D (out_features, in_features) weight"
            )
        out_features, in_features = (int(value) for value in wa.shape)
        size = int(self._packed_size(out_features, in_features))
        if size <= 0:
            raise InvalidOperationParameters("native pack_weight got an empty weight")
        packed = np.zeros(size, dtype=np.float32)
        self._check(
            self._pack_weight(
                _pointer(wa), _pointer(packed), out_features, in_features
            ),
            "pack_weight",
        )
        return PackedWeight(packed, in_features, out_features)

    def gemm_packed(
        self,
        x: Any,
        packed: PackedWeight,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ W^T + b) with W already in blocked layout."""
        if activation not in ACTIVATION_CODES:
            raise InvalidOperationParameters(
                f"native gemm activation must be one of {sorted(ACTIVATION_CODES)}, "
                f"got '{activation}'"
            )
        xa = _as_f32(x, "gemm_packed.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native gemm_packed requires rank >= 1")
        if xa.shape[-1] != packed.in_features:
            raise InvalidOperationParameters(
                f"native gemm_packed: input last dim {xa.shape[-1]} != packed "
                f"in_features {packed.in_features}"
            )
        ba = None
        if bias is not None:
            ba = _as_f32(bias, "gemm_packed.bias")
            if ba.shape != (packed.out_features,):
                raise InvalidOperationParameters(
                    f"native gemm_packed bias shape {ba.shape} != "
                    f"({packed.out_features},)"
                )
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        output = np.empty((rows, packed.out_features), dtype=np.float32)
        if rows:
            self._check(
                self._gemm_packed(
                    _pointer(xa),
                    _pointer(packed.array),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    packed.in_features,
                    packed.out_features,
                    ACTIVATION_CODES[activation],
                ),
                "gemm_packed",
            )
        return output.reshape(tuple(xa.shape[:-1]) + (packed.out_features,))

    def add_norm(
        self,
        a: Any,
        b: Any,
        *,
        weight: Any | None = None,
        bias: Any | None = None,
        eps: float = 1e-5,
        kind: str = "layernorm",
        residual: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fused ``a + b`` residual add followed by a normalization."""
        if kind not in ("layernorm", "rmsnorm"):
            raise InvalidOperationParameters(
                f"native add_norm kind must be 'layernorm' or 'rmsnorm', got '{kind}'"
            )
        if kind == "rmsnorm" and bias is not None:
            raise InvalidOperationParameters("native rmsnorm has no bias term")
        aa = _as_f32(a, f"add_norm.{kind}.a")
        ba = _as_f32(b, f"add_norm.{kind}.b")
        if aa.shape != ba.shape:
            raise InvalidOperationParameters(
                f"native add_norm needs equal shapes, got {aa.shape} and {ba.shape}; "
                "broadcast before fusing a residual add into a normalization"
            )
        if aa.ndim < 1:
            raise InvalidOperationParameters("native add_norm requires rank >= 1")
        cols = int(aa.shape[-1])
        rows = int(np.prod(aa.shape[:-1])) if aa.ndim > 1 else 1
        wa, bia = self._norm_params(weight, bias, cols, kind)
        output = np.empty(aa.shape, dtype=np.float32)
        residual_array = np.empty(aa.shape, dtype=np.float32) if residual else None
        if rows:
            self._check(
                self._add_norm(
                    _pointer(aa),
                    _pointer(ba),
                    _pointer(wa),
                    _pointer(bia),
                    _pointer(output),
                    _pointer(residual_array),
                    rows,
                    cols,
                    float(eps),
                    0 if kind == "layernorm" else 1,
                ),
                "add_norm",
            )
        return output, residual_array

    def reduce_axis(
        self, x: Any, axis: int, *, mode: str = "mean", keepdims: bool = False
    ) -> np.ndarray:
        """Mean/sum over any single axis (sequence pooling, channel pooling)."""
        if mode not in ("mean", "sum"):
            raise InvalidOperationParameters(
                f"native reduce mode must be 'mean' or 'sum', got '{mode}'"
            )
        xa = _as_f32(x, f"{mode}.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters("native reduce requires rank >= 1")
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise InvalidOperationParameters(
                f"native reduce axis must be a single int, got {axis!r}"
            )
        if axis < -xa.ndim or axis >= xa.ndim:
            raise InvalidOperationParameters(
                f"native reduce axis {axis} out of range for rank {xa.ndim}"
            )
        normalized = axis % xa.ndim
        outer = int(np.prod(xa.shape[:normalized]))
        size = int(xa.shape[normalized])
        inner = int(np.prod(xa.shape[normalized + 1 :]))
        if size <= 0:
            raise InvalidOperationParameters("native reduce cannot reduce an empty axis")
        output = np.empty((max(outer, 1) * max(inner, 1),), dtype=np.float32)
        if outer and inner:
            self._check(
                self._reduce_axis(
                    _pointer(xa),
                    _pointer(output),
                    outer,
                    size,
                    inner,
                    0 if mode == "mean" else 1,
                ),
                f"{mode}_axis",
            )
        shape = list(xa.shape)
        if keepdims:
            shape[normalized] = 1
        else:
            del shape[normalized]
        return output.reshape(tuple(shape))

    def conv2d(
        self,
        x: Any,
        weight: Any,
        bias: Any | None = None,
        *,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
    ) -> np.ndarray:
        xa = _as_f32(x, "conv2d.x")
        wa = _as_f32(weight, "conv2d.weight")
        if xa.ndim != 4 or wa.ndim != 4:
            raise InvalidOperationParameters(
                "native conv2d expects input (B, C, H, W) and weight (O, C, kh, kw)"
            )
        if xa.shape[1] != wa.shape[1]:
            raise InvalidOperationParameters(
                f"native conv2d channel mismatch: input {xa.shape[1]} vs weight "
                f"{wa.shape[1]}"
            )
        sh, sw = _pair(stride, "conv stride")
        ph, pw = _pair(padding, "conv padding")
        if sh <= 0 or sw <= 0 or ph < 0 or pw < 0:
            raise InvalidOperationParameters(
                "native conv2d stride must be positive and padding non-negative"
            )
        batch, in_channels, in_h, in_w = (int(v) for v in xa.shape)
        out_channels, _, kh, kw = (int(v) for v in wa.shape)
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise InvalidOperationParameters(
                f"native conv2d kernel {kh}x{kw} does not fit the padded input "
                f"{in_h + 2 * ph}x{in_w + 2 * pw}"
            )
        ba = None
        if bias is not None:
            ba = _as_f32(bias, "conv2d.bias")
            if ba.shape != (out_channels,):
                raise InvalidOperationParameters(
                    f"native conv2d bias shape {ba.shape} != ({out_channels},)"
                )
        output = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
        self._check(
            self._conv2d(
                _pointer(xa),
                _pointer(wa),
                _pointer(ba),
                _pointer(output),
                batch,
                in_channels,
                in_h,
                in_w,
                out_channels,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                out_h,
                out_w,
            ),
            "conv2d",
        )
        return output

    def maxpool2d(
        self,
        x: Any,
        *,
        kernel: int | Sequence[int] = 2,
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] = 0,
    ) -> np.ndarray:
        xa = _as_f32(x, "maxpool2d.x")
        if xa.ndim != 4:
            raise InvalidOperationParameters("native maxpool2d expects (B, C, H, W)")
        kh, kw = _pair(kernel, "pool kernel")
        sh, sw = _pair(stride if stride is not None else kernel, "pool stride")
        ph, pw = _pair(padding, "pool padding")
        if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0 or ph < 0 or pw < 0:
            raise InvalidOperationParameters(
                "native maxpool2d kernel/stride must be positive and padding non-negative"
            )
        batch, channels, in_h, in_w = (int(v) for v in xa.shape)
        out_h = (in_h + 2 * ph - kh) // sh + 1
        out_w = (in_w + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise InvalidOperationParameters(
                f"native maxpool2d window {kh}x{kw} does not fit the padded input"
            )
        output = np.empty((batch, channels, out_h, out_w), dtype=np.float32)
        self._check(
            self._pool2d(
                _pointer(xa),
                _pointer(output),
                batch,
                channels,
                in_h,
                in_w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                out_h,
                out_w,
                0,
            ),
            "maxpool2d",
        )
        return output

    def recurrent(
        self,
        x: Any,
        weight_ih: Any,
        weight_hh: Any,
        bias: Any | None = None,
        *,
        kind: str = "rnn",
    ) -> np.ndarray:
        if kind not in RECURRENT_KINDS:
            raise InvalidOperationParameters(
                f"native recurrent kind must be one of {sorted(RECURRENT_KINDS)}"
            )
        xa = _as_f32(x, f"{kind}.x")
        if xa.ndim != 3:
            raise InvalidOperationParameters(
                f"native {kind} expects (batch, seq, features), got {xa.shape}"
            )
        wih = _as_f32(weight_ih, f"{kind}.weight_ih")
        whh = _as_f32(weight_hh, f"{kind}.weight_hh")
        if wih.ndim != 2 or whh.ndim != 2:
            raise InvalidOperationParameters(
                f"native {kind} weights must be 2-D (gates*hidden, features|hidden)"
            )
        batch, steps, features = (int(v) for v in xa.shape)
        gates = RECURRENT_GATES[kind]
        if wih.shape != (whh.shape[0], features):
            raise InvalidOperationParameters(
                f"native {kind} weight_ih shape {wih.shape} != "
                f"({whh.shape[0]}, {features})"
            )
        if whh.shape[0] != whh.shape[1] * gates:
            raise InvalidOperationParameters(
                f"native {kind} weight_hh shape {whh.shape} is not "
                f"({gates} * hidden, hidden)"
            )
        hidden = int(whh.shape[1])
        ba = None
        if bias is not None:
            ba = _as_f32(bias, f"{kind}.bias")
            if ba.shape != (gates * hidden,):
                raise InvalidOperationParameters(
                    f"native {kind} bias shape {ba.shape} != ({gates * hidden},)"
                )
        output = np.empty((batch, steps, hidden), dtype=np.float32)
        self._check(
            self._recurrent(
                _pointer(xa),
                _pointer(wih),
                _pointer(whh),
                _pointer(ba),
                _pointer(output),
                batch,
                steps,
                features,
                hidden,
                RECURRENT_KINDS[kind],
            ),
            kind,
        )
        return output

    def ssm(
        self,
        x: Any,
        a: Any,
        b: Any,
        c: Any,
        d: Any | None = None,
    ) -> np.ndarray:
        xa = _as_f32(x, "ssm.x")
        if xa.ndim != 3:
            raise InvalidOperationParameters("native ssm expects (batch, seq, channels)")
        aa = _as_f32(a, "ssm.A")
        ba = _as_f32(b, "ssm.B")
        ca = _as_f32(c, "ssm.C")
        if not (aa.ndim == ba.ndim == ca.ndim == 2):
            raise InvalidOperationParameters(
                "native ssm expects A, B, C of shape (channels, state_dim)"
            )
        if not (aa.shape == ba.shape == ca.shape):
            raise InvalidOperationParameters(
                f"native ssm parameter shapes disagree: {aa.shape}, {ba.shape}, "
                f"{ca.shape}"
            )
        batch, steps, channels = (int(v) for v in xa.shape)
        if aa.shape[0] != channels:
            raise InvalidOperationParameters(
                f"native ssm channels {aa.shape[0]} != input channels {channels}"
            )
        da = None
        if d is not None:
            da = _as_f32(d, "ssm.D")
            if da.shape != (channels,):
                raise InvalidOperationParameters(
                    f"native ssm D shape {da.shape} != ({channels},)"
                )
        output = np.empty_like(xa)
        self._check(
            self._ssm(
                _pointer(xa),
                _pointer(aa),
                _pointer(ba),
                _pointer(ca),
                _pointer(da),
                _pointer(output),
                batch,
                steps,
                channels,
                int(aa.shape[1]),
            ),
            "ssm",
        )
        return output

    def gcn(
        self,
        x: Any,
        adjacency: Any,
        weight: Any,
        bias: Any | None = None,
    ) -> np.ndarray:
        xa = _as_f32(x, "gcn.x")
        adj = _as_f32(adjacency, "gcn.adjacency")
        wa = _as_f32(weight, "gcn.weight")
        if xa.ndim != 3:
            raise InvalidOperationParameters(
                "native gcn node features must be (batch, nodes, features)"
            )
        batch, nodes, in_features = (int(v) for v in xa.shape)
        if adj.shape != (batch, nodes, nodes):
            raise InvalidOperationParameters(
                f"native gcn adjacency {adj.shape} != ({batch}, {nodes}, {nodes})"
            )
        if wa.ndim != 2 or wa.shape[1] != in_features:
            raise InvalidOperationParameters(
                f"native gcn weight must be (out_features, {in_features}), got "
                f"{wa.shape}"
            )
        out_features = int(wa.shape[0])
        ba = None
        if bias is not None:
            ba = _as_f32(bias, "gcn.bias")
            if ba.shape != (out_features,):
                raise InvalidOperationParameters(
                    f"native gcn bias shape {ba.shape} != ({out_features},)"
                )
        output = np.zeros((batch, nodes, out_features), dtype=np.float32)
        self._check(
            self._gcn(
                _pointer(xa),
                _pointer(adj),
                _pointer(wa),
                _pointer(ba),
                _pointer(output),
                batch,
                nodes,
                in_features,
                out_features,
            ),
            "gcn",
        )
        return output

    def moe(
        self,
        x: Any,
        router: Any,
        up_weight: Any,
        down_weight: Any,
        *,
        up_bias: Any | None = None,
        down_bias: Any | None = None,
        top_k: int = 2,
        activation: str = "gelu",
    ) -> np.ndarray:
        xa = _as_f32(x, "moe.x")
        ra = _as_f32(router, "moe.router")
        up = _as_f32(up_weight, "moe.up_weight")
        down = _as_f32(down_weight, "moe.down_weight")
        if xa.ndim < 2:
            raise InvalidOperationParameters("native moe expects (..., features)")
        features = int(xa.shape[-1])
        if ra.ndim != 2 or ra.shape[1] != features:
            raise InvalidOperationParameters(
                f"native moe router must be (experts, {features}), got {ra.shape}"
            )
        experts = int(ra.shape[0])
        if up.ndim != 3 or down.ndim != 3:
            raise InvalidOperationParameters(
                "native moe expert stacks must be 3-D (experts, out, in)"
            )
        hidden = int(up.shape[1])
        if up.shape != (experts, hidden, features) or down.shape != (
            experts,
            features,
            hidden,
        ):
            raise InvalidOperationParameters(
                f"native moe expert stacks must be ({experts}, hidden, {features}) and "
                f"({experts}, {features}, hidden); got {up.shape} and {down.shape}"
            )
        if top_k < 1 or top_k > experts:
            raise InvalidOperationParameters(
                f"native moe top_k must be in [1, {experts}], got {top_k}"
            )
        if activation not in ACTIVATION_CODES or activation == "none":
            raise InvalidOperationParameters(
                f"native moe activation must be one of "
                f"{sorted(set(ACTIVATION_CODES) - {'none'})}, got '{activation}'"
            )
        ub = None
        if up_bias is not None:
            ub = _as_f32(up_bias, "moe.up_bias")
            if ub.shape != (experts, hidden):
                raise InvalidOperationParameters(
                    f"native moe up_bias shape {ub.shape} != ({experts}, {hidden})"
                )
        db = None
        if down_bias is not None:
            db = _as_f32(down_bias, "moe.down_bias")
            if db.shape != (experts, features):
                raise InvalidOperationParameters(
                    f"native moe down_bias shape {db.shape} != ({experts}, {features})"
                )
        tokens = int(np.prod(xa.shape[:-1]))
        output = np.zeros((tokens, features), dtype=np.float32)
        self._check(
            self._moe(
                _pointer(xa),
                _pointer(ra),
                _pointer(up),
                _pointer(ub),
                _pointer(down),
                _pointer(db),
                _pointer(output),
                tokens,
                features,
                experts,
                hidden,
                int(top_k),
                ACTIVATION_CODES[activation],
            ),
            "moe",
        )
        return output.reshape(tuple(xa.shape))

    def quantized_linear(
        self,
        x: Any,
        weight_int8: Any,
        scale: Any,
        bias: Any | None = None,
    ) -> np.ndarray:
        """float activations x int8 weights with per-output-channel scales."""
        xa = _as_f32(x, "quantized_linear.x")
        weights = np.asarray(weight_int8)
        if weights.dtype != np.int8:
            raise InvalidOperationParameters(
                f"native quantized_linear requires int8 weights, got {weights.dtype}"
            )
        weights = np.ascontiguousarray(weights)
        if weights.ndim != 2:
            raise InvalidOperationParameters(
                "native quantized_linear weight must be 2-D (out_features, in_features)"
            )
        out_features, in_features = (int(v) for v in weights.shape)
        if xa.ndim < 1 or xa.shape[-1] != in_features:
            raise InvalidOperationParameters(
                f"native quantized_linear input last dim {xa.shape[-1]} != weight "
                f"in_features {in_features}"
            )
        scales = _as_f32(scale, "quantized_linear.scale")
        if scales.shape != (out_features,):
            raise InvalidOperationParameters(
                f"native quantized_linear scale shape {scales.shape} != "
                f"({out_features},)"
            )
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise InvalidOperationParameters(
                "native quantized_linear scales must be finite and positive"
            )
        ba = None
        if bias is not None:
            ba = _as_f32(bias, "quantized_linear.bias")
            if ba.shape != (out_features,):
                raise InvalidOperationParameters(
                    f"native quantized_linear bias shape {ba.shape} != "
                    f"({out_features},)"
                )
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        output = np.empty((rows, out_features), dtype=np.float32)
        if rows:
            self._check(
                self._quantized_linear(
                    _pointer(xa),
                    _pointer(weights),
                    _pointer(scales),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    in_features,
                    out_features,
                ),
                "quantized_linear",
            )
        return output.reshape(tuple(xa.shape[:-1]) + (out_features,))

    def attention_flash(
        self,
        q: Any,
        k: Any,
        v: Any,
        *,
        scale: float | str | None = None,
        causal: bool = True,
        alibi: Any | None = None,
        window: int | None = None,
    ) -> np.ndarray:
        """Single-pass attention: no score matrix, one pass over the keys."""
        qa = _as_f32(q, "attention_flash.q")
        ka = _as_f32(k, "attention_flash.k")
        va = _as_f32(v, "attention_flash.v")
        if not (qa.ndim == ka.ndim == va.ndim == 4):
            raise InvalidOperationParameters(
                "native attention_flash requires (batch, heads, seq, head_dim) tensors"
            )
        if qa.shape[0] != ka.shape[0] or qa.shape[0] != va.shape[0]:
            raise InvalidOperationParameters(
                "native attention_flash batch dimensions disagree"
            )
        if qa.shape[1] != ka.shape[1] or qa.shape[1] != va.shape[1]:
            raise InvalidOperationParameters(
                "native attention_flash head dimensions disagree"
            )
        if qa.shape[3] != ka.shape[3] or qa.shape[3] != va.shape[3]:
            raise InvalidOperationParameters(
                "native attention_flash head_dim dimensions disagree"
            )
        if ka.shape[2] != va.shape[2]:
            raise InvalidOperationParameters(
                "native attention_flash k/v sequence lengths disagree"
            )
        if qa.shape[2] > ka.shape[2]:
            raise InvalidOperationParameters(
                "native attention_flash q sequence cannot exceed the k/v sequence"
            )
        dimensions = int(qa.shape[3])
        if scale is None or scale == "auto":
            scale_value = 1.0 / float(np.sqrt(dimensions))
        else:
            scale_value = float(scale)
        alibi_array = None
        if alibi is not None:
            alibi_array = _as_f32(alibi, "attention_flash.alibi")
            if alibi_array.shape != (qa.shape[1],):
                raise InvalidOperationParameters(
                    f"native attention_flash alibi shape {alibi_array.shape} != "
                    f"({qa.shape[1]},)"
                )
        window_value = -1 if window is None else int(window)
        output = np.empty(qa.shape, dtype=np.float32)
        if int(np.prod(qa.shape[:3])):
            self._check(
                self._attention_flash(
                    _pointer(qa),
                    _pointer(ka),
                    _pointer(va),
                    _pointer(output),
                    int(qa.shape[0]),
                    int(qa.shape[1]),
                    int(qa.shape[2]),
                    int(ka.shape[2]),
                    dimensions,
                    scale_value,
                    1 if causal else 0,
                    _pointer(alibi_array),
                    window_value,
                ),
                "attention_flash",
            )
        return output

    # ------------------------------------------------------------------ #
    # parameter representations (phase N3)
    # ------------------------------------------------------------------ #

    def _activation_code(self, activation: str) -> int:
        if activation not in ACTIVATION_CODES:
            raise InvalidOperationParameters(
                f"native gemm activation must be one of {sorted(ACTIVATION_CODES)}, "
                f"got '{activation}'"
            )
        return ACTIVATION_CODES[activation]

    def _gemm_bias(
        self, bias: Any | None, out_features: int, label: str
    ) -> np.ndarray | None:
        if bias is None:
            return None
        ba = _as_f32(bias, f"{label}.bias")
        if ba.shape != (out_features,):
            raise InvalidOperationParameters(
                f"native {label} bias shape {ba.shape} != ({out_features},)"
            )
        return ba

    def _gemm_input(
        self, x: Any, in_features: int, label: str
    ) -> tuple[np.ndarray, int, tuple[int, ...]]:
        xa = _as_f32(x, f"{label}.x")
        if xa.ndim < 1:
            raise InvalidOperationParameters(f"native {label} requires rank >= 1")
        if xa.shape[-1] != in_features:
            raise InvalidOperationParameters(
                f"native {label}: input last dim {xa.shape[-1]} != weight "
                f"in_features {in_features}"
            )
        rows = int(np.prod(xa.shape[:-1])) if xa.ndim > 1 else 1
        return xa, rows, tuple(xa.shape[:-1])

    def pack_weight_int8_block(
        self, weight: Any, scales: Any
    ) -> PackedBlockWeight:
        """Pack symmetric int8 weights with one scale per (row, group)."""
        weights = _as_int8(weight, "pack_weight_int8_block.weight")
        if weights.ndim != 2:
            raise InvalidOperationParameters(
                "native pack_weight_int8_block requires a 2-D weight"
            )
        out_features, in_features = (int(v) for v in weights.shape)
        group = self.quant_group
        groups = -(-in_features // group)
        scale_array = _as_f32(scales, "pack_weight_int8_block.scales")
        if scale_array.shape != (out_features, groups):
            raise InvalidOperationParameters(
                f"native pack_weight_int8_block scales shape {scale_array.shape} != "
                f"({out_features}, {groups}) for group size {group}"
            )
        if not np.isfinite(scale_array).all() or (scale_array <= 0).any():
            raise InvalidOperationParameters(
                "native pack_weight_int8_block scales must be finite and positive"
            )
        packed = PackedBlockWeight(
            kind="int8_block",
            weight=np.zeros(
                -(-out_features // BLOCK_N) * groups * group * BLOCK_N, dtype=np.int8
            ),
            scales=np.zeros(-(-out_features // BLOCK_N) * groups * BLOCK_N, dtype=np.float32),
            in_features=in_features,
            out_features=out_features,
            group=group,
        )
        scale_array = np.ascontiguousarray(scale_array)
        self._check(
            self._pack_int8_block(
                _pointer(weights),
                _pointer(scale_array),
                _pointer(packed.weight),
                _pointer(packed.scales),
                out_features,
                in_features,
            ),
            "pack_weight_int8_block",
        )
        return packed

    def gemm_int8_block(
        self,
        x: Any,
        packed: PackedBlockWeight,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ (wq * scale)^T + b) for packed int8 weights."""
        if packed.kind != "int8_block":
            raise InvalidOperationParameters(
                f"native gemm_int8_block needs an int8_block weight, got '{packed.kind}'"
            )
        code = self._activation_code(activation)
        xa, rows, leading = self._gemm_input(x, packed.in_features, "gemm_int8_block")
        ba = self._gemm_bias(bias, packed.out_features, "gemm_int8_block")
        output = np.empty((rows, packed.out_features), dtype=np.float32)
        if rows:
            self._check(
                self._gemm_int8_block(
                    _pointer(xa),
                    _pointer(packed.weight),
                    _pointer(packed.scales),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    packed.in_features,
                    packed.out_features,
                    code,
                ),
                "gemm_int8_block",
            )
        return output.reshape(leading + (packed.out_features,))

    def pack_weight_int4_block(
        self, nibbles: Any, scales: Any
    ) -> PackedBlockWeight:
        """Pack int4 values (one unsigned nibble each, two's complement)."""
        values = _as_uint8(nibbles, "pack_weight_int4_block.nibbles")
        if values.ndim != 2:
            raise InvalidOperationParameters(
                "native pack_weight_int4_block requires a 2-D (out, in) nibble buffer"
            )
        if values.size and int(values.max()) > 0x0F:
            raise InvalidOperationParameters(
                "native pack_weight_int4_block expects one nibble per element "
                f"(0..15), got a maximum of {int(values.max())}"
            )
        out_features, in_features = (int(v) for v in values.shape)
        group = self.quant_group
        groups = -(-in_features // group)
        scale_array = _as_f32(scales, "pack_weight_int4_block.scales")
        if scale_array.shape != (out_features, groups):
            raise InvalidOperationParameters(
                f"native pack_weight_int4_block scales shape {scale_array.shape} != "
                f"({out_features}, {groups}) for group size {group}"
            )
        if not np.isfinite(scale_array).all() or (scale_array <= 0).any():
            raise InvalidOperationParameters(
                "native pack_weight_int4_block scales must be finite and positive"
            )
        blocks_out = -(-out_features // BLOCK_N)
        packed = PackedBlockWeight(
            kind="int4_block",
            weight=np.zeros(blocks_out * groups * (group // 2) * BLOCK_N, dtype=np.uint8),
            scales=np.zeros(blocks_out * groups * BLOCK_N, dtype=np.float32),
            in_features=in_features,
            out_features=out_features,
            group=group,
        )
        scale_array = np.ascontiguousarray(scale_array)
        self._check(
            self._pack_int4_block(
                _pointer(values),
                _pointer(scale_array),
                _pointer(packed.weight),
                _pointer(packed.scales),
                out_features,
                in_features,
            ),
            "pack_weight_int4_block",
        )
        return packed

    def gemm_int4_block(
        self,
        x: Any,
        packed: PackedBlockWeight,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ (wq * scale)^T + b) for packed int4 weights."""
        if packed.kind != "int4_block":
            raise InvalidOperationParameters(
                f"native gemm_int4_block needs an int4_block weight, got '{packed.kind}'"
            )
        code = self._activation_code(activation)
        xa, rows, leading = self._gemm_input(x, packed.in_features, "gemm_int4_block")
        ba = self._gemm_bias(bias, packed.out_features, "gemm_int4_block")
        output = np.empty((rows, packed.out_features), dtype=np.float32)
        if rows:
            self._check(
                self._gemm_int4_block(
                    _pointer(xa),
                    _pointer(packed.weight),
                    _pointer(packed.scales),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    packed.in_features,
                    packed.out_features,
                    code,
                ),
                "gemm_int4_block",
            )
        return output.reshape(leading + (packed.out_features,))

    def gemm_low_rank(
        self,
        x: Any,
        v: Any,
        u: Any,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ (U V)^T + b) from rank factors ``v`` and ``u``."""
        code = self._activation_code(activation)
        va = _as_f32(v, "gemm_low_rank.v")
        ua = _as_f32(u, "gemm_low_rank.u")
        if va.ndim != 2 or ua.ndim != 2:
            raise InvalidOperationParameters(
                "native gemm_low_rank requires 2-D factors v (rank, in) and u (out, rank)"
            )
        rank, in_features = (int(value) for value in va.shape)
        out_features = int(ua.shape[0])
        if rank <= 0 or in_features <= 0 or out_features <= 0:
            raise InvalidOperationParameters("native gemm_low_rank needs non-empty factors")
        if ua.shape[1] != rank:
            raise InvalidOperationParameters(
                f"native gemm_low_rank: u has {ua.shape[1]} columns but v has rank {rank}"
            )
        xa, rows, leading = self._gemm_input(x, in_features, "gemm_low_rank")
        ba = self._gemm_bias(bias, out_features, "gemm_low_rank")
        output = np.empty((rows, out_features), dtype=np.float32)
        if rows:
            self._check(
                self._gemm_low_rank(
                    _pointer(xa),
                    _pointer(va),
                    _pointer(ua),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    in_features,
                    rank,
                    out_features,
                    code,
                ),
                "gemm_low_rank",
            )
        return output.reshape(leading + (out_features,))

    def sparse_gemm(
        self,
        x: Any,
        values: Any,
        indices: Any,
        row_ptr: Any,
        in_features: int,
        out_features: int,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ W^T + b) for W stored as CSR rows over outputs.

        ``in_features`` is explicit: trailing all-zero input columns produce no
        nonzeros, so the CSR indices alone cannot imply the input width.
        """
        code = self._activation_code(activation)
        in_features = int(in_features)
        out_features = int(out_features)
        if in_features <= 0 or out_features <= 0:
            raise InvalidOperationParameters(
                "native sparse_gemm needs positive in_features and out_features"
            )
        va = _as_f32(values, "sparse_gemm.values")
        if va.ndim != 1:
            raise InvalidOperationParameters("native sparse_gemm values must be 1-D")
        ia = _as_int32(indices, "sparse_gemm.indices")
        rp = _as_int32(row_ptr, "sparse_gemm.row_ptr")
        if ia.shape != va.shape:
            raise InvalidOperationParameters(
                f"native sparse_gemm indices {ia.shape} != values {va.shape}"
            )
        if rp.shape != (out_features + 1,):
            raise InvalidOperationParameters(
                f"native sparse_gemm row_ptr {rp.shape} != ({out_features + 1},)"
            )
        if int(rp[0]) != 0 or int(rp[-1]) != int(va.size):
            raise InvalidOperationParameters(
                "native sparse_gemm row_ptr must start at 0 and end at the nonzero count"
            )
        if va.size and (int(ia.min()) < 0 or int(ia.max()) >= in_features):
            raise InvalidOperationParameters(
                f"native sparse_gemm indices must lie in [0, {in_features})"
            )
        xa, rows, leading = self._gemm_input(x, in_features, "sparse_gemm")
        ba = self._gemm_bias(bias, out_features, "sparse_gemm")
        output = np.empty((rows, out_features), dtype=np.float32)
        if rows:
            self._check(
                self._sparse_gemm(
                    _pointer(xa),
                    _pointer(va),
                    _pointer(ia),
                    _pointer(rp),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    in_features,
                    out_features,
                    int(va.size),
                    code,
                ),
                "sparse_gemm",
            )
        return output.reshape(leading + (out_features,))

    def pack_weight_transposed(self, weight: Any) -> PackedTransposedWeight:
        """Store a (out, in) weight as an (in, out) contiguous layout."""
        wa = _as_f32(weight, "pack_weight_transposed")
        if wa.ndim != 2:
            raise InvalidOperationParameters(
                "native pack_weight_transposed requires a 2-D (out, in) weight"
            )
        out_features, in_features = (int(value) for value in wa.shape)
        packed = np.zeros(in_features * out_features, dtype=np.float32)
        self._check(
            self._pack_weight_transposed(
                _pointer(wa), _pointer(packed), out_features, in_features
            ),
            "pack_weight_transposed",
        )
        return PackedTransposedWeight(packed, in_features, out_features)

    def gemm_transposed(
        self,
        x: Any,
        packed: PackedTransposedWeight,
        bias: Any | None = None,
        *,
        activation: str = "none",
    ) -> np.ndarray:
        """y = activation(x @ W^T + b) from the (in, out) dense layout."""
        code = self._activation_code(activation)
        xa, rows, leading = self._gemm_input(x, packed.in_features, "gemm_transposed")
        ba = self._gemm_bias(bias, packed.out_features, "gemm_transposed")
        output = np.empty((rows, packed.out_features), dtype=np.float32)
        if rows:
            self._check(
                self._gemm_transposed(
                    _pointer(xa),
                    _pointer(packed.array),
                    _pointer(ba),
                    _pointer(output),
                    rows,
                    packed.in_features,
                    packed.out_features,
                    code,
                ),
                "gemm_transposed",
            )
        return output.reshape(leading + (packed.out_features,))

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "compiler": native_compiler(),
            "kernels": sorted(
                [
                    "linear",
                    "matmul",
                    "layernorm",
                    "rmsnorm",
                    "softmax",
                    *self._unary,
                    *self._binary,
                    "scale",
                    "clip",
                    "embedding",
                    "reduce_mean",
                    "reduce_sum",
                    "reduce_axis",
                    "attention",
                    "attention_flash",
                    "pack_weight",
                    "gemm_packed",
                    "add_norm",
                    "conv2d",
                    "maxpool2d",
                    "rnn",
                    "lstm",
                    "gru",
                    "ssm",
                    "gcn",
                    "moe",
                    "quantized_linear",
                    "pack_weight_int8_block",
                    "gemm_int8_block",
                    "pack_weight_int4_block",
                    "gemm_int4_block",
                    "gemm_low_rank",
                    "sparse_gemm",
                    "pack_weight_transposed",
                    "gemm_transposed",
                ]
            ),
            "quant_group": self.quant_group,
            "float32_only": True,
        }
