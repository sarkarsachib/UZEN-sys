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
                    "attention",
                ]
            ),
            "float32_only": True,
        }
