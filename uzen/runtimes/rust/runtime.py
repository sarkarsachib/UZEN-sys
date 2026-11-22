"""Rust kernel backend: build the crate and call it through ``ctypes``.

The crate in this directory (``src/lib.rs``) exposes the inference kernels behind a C
ABI. :func:`build_library` runs ``cargo build --release`` for real and reports the
artifact it produced; :class:`RustKernels` loads that shared library and calls the
symbols with validated NumPy buffers; :class:`RustRuntime` executes the subset of NIR
graphs those kernels cover and refuses every other operator instead of falling back
silently.

Nothing here is a stub: if ``cargo`` (or a prebuilt library) is absent, construction
raises :class:`~uzen.errors.RuntimeUnavailable` naming the reason, and the test-suite
skips rather than claiming Rust execution.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from uzen.core.nir.graph import Graph
from uzen.core.nir.executor import resolve_bindings
from uzen.core.nws.tensors import WeightPackage
from uzen.errors import RuntimeUnavailable, ShapeMismatch, TensorInvalid, UnsupportedOperator

PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST = PACKAGE_DIR / "Cargo.toml"
TARGET_DIR = PACKAGE_DIR / "target"

F32 = ctypes.POINTER(ctypes.c_float)
SIZE = ctypes.c_size_t
U32 = ctypes.c_uint32
FLOAT = ctypes.c_float

RUST_OPS = frozenset(
    {
        "linear",
        "layernorm",
        "rmsnorm",
        "softmax",
        "gelu",
        "silu",
        "relu",
        "tanh",
        "sigmoid",
        "scale",
        "add",
        "mul",
        "identity",
    }
)


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libuzen_rust.dylib"
    if sys.platform.startswith("win"):
        return "uzen_rust.dll"
    return "libuzen_rust.so"


def find_cargo() -> str | None:
    found = shutil.which("cargo")
    if found:
        return found
    candidate = Path.home() / ".cargo" / "bin" / "cargo"
    return str(candidate) if candidate.exists() else None


def library_path(*, release: bool = True) -> Path | None:
    profile = "release" if release else "debug"
    candidate = TARGET_DIR / profile / _library_name()
    return candidate if candidate.exists() else None


def _is_stale(artifact: Path) -> bool:
    """A library older than its manifest/sources must be rebuilt."""
    sources = [MANIFEST, *sorted((PACKAGE_DIR / "src").rglob("*.rs"))]
    newest = max((path.stat().st_mtime for path in sources if path.exists()), default=0.0)
    return artifact.stat().st_mtime < newest


def build_library(*, release: bool = True, force: bool = False, timeout: float = 600.0) -> dict[str, Any]:
    """Build the crate and return a report describing the real artifact."""
    cargo = find_cargo()
    if cargo is None:
        raise RuntimeUnavailable(
            "the Rust backend needs a Rust toolchain; install it from https://rustup.rs "
            "or set CARGO to an existing cargo binary"
        )
    if not MANIFEST.exists():  # pragma: no cover - repository layout guarantee
        raise RuntimeUnavailable(f"missing Cargo.toml at {MANIFEST}")
    existing = library_path(release=release)
    if existing is not None and not force and not _is_stale(existing):
        return {
            "built": False,
            "cached": True,
            "library": str(existing),
            "cargo": cargo,
            "profile": "release" if release else "debug",
        }
    command = [cargo, "build"]
    if release:
        command.append("--release")
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=str(PACKAGE_DIR), capture_output=True, text=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise RuntimeUnavailable(
            "cargo build failed for the UZEN Rust kernels",
            reason=(completed.stderr or completed.stdout).strip()[-4000:],
            command=" ".join(command),
        )
    artifact = library_path(release=release)
    if artifact is None:  # pragma: no cover - cargo always writes the cdylib
        raise RuntimeUnavailable("cargo reported success but no shared library was produced")
    return {
        "built": True,
        "cached": False,
        "library": str(artifact),
        "cargo": cargo,
        "profile": "release" if release else "debug",
        "seconds": round(time.monotonic() - started, 3),
    }


def _pointer(array: np.ndarray) -> Any:
    return array.ctypes.data_as(F32)


class RustKernels:
    """ctypes binding for the compiled kernel library."""

    def __init__(self, path: str | Path | None = None, *, build: bool = True) -> None:
        resolved = Path(path) if path is not None else library_path()
        if resolved is None and build:
            resolved = Path(build_library()["library"])
        if resolved is None or not Path(resolved).exists():
            raise RuntimeUnavailable(
                "the Rust kernel library is not built; run "
                "`python -m uzen.runtimes.rust.build` or call build_library()"
            )
        self.path = Path(resolved)
        self._lib = ctypes.CDLL(str(self.path))
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.uzen_rust_version.argtypes = []
        lib.uzen_rust_version.restype = U32
        lib.uzen_rust_linear.argtypes = [F32, F32, F32, F32, SIZE, SIZE, SIZE]
        lib.uzen_rust_linear.restype = None
        lib.uzen_rust_layernorm.argtypes = [F32, F32, F32, F32, SIZE, SIZE, FLOAT]
        lib.uzen_rust_layernorm.restype = None
        lib.uzen_rust_rmsnorm.argtypes = [F32, F32, F32, SIZE, SIZE, FLOAT]
        lib.uzen_rust_rmsnorm.restype = None
        lib.uzen_rust_softmax.argtypes = [F32, F32, SIZE, SIZE]
        lib.uzen_rust_softmax.restype = None
        for name in (
            "uzen_rust_gelu",
            "uzen_rust_silu",
            "uzen_rust_relu",
            "uzen_rust_tanh",
            "uzen_rust_sigmoid",
        ):
            getattr(lib, name).argtypes = [F32, F32, SIZE]
            getattr(lib, name).restype = None
        lib.uzen_rust_scale.argtypes = [F32, F32, SIZE, FLOAT]
        lib.uzen_rust_scale.restype = None
        for name in ("uzen_rust_add", "uzen_rust_mul"):
            getattr(lib, name).argtypes = [F32, F32, F32, SIZE]
            getattr(lib, name).restype = None

    @property
    def version(self) -> int:
        return int(self._lib.uzen_rust_version())

    @staticmethod
    def _array(value: Any, name: str) -> np.ndarray:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
        if array.dtype != np.float32:
            raise TensorInvalid(f"{name} must be float32, got {array.dtype}")
        return array

    def linear(self, x: Any, weight: Any, bias: Any | None = None) -> np.ndarray:
        source = self._array(x, "x")
        matrix = self._array(weight, "weight")
        if source.ndim < 1 or matrix.ndim != 2 or source.shape[-1] != matrix.shape[1]:
            raise ShapeMismatch(
                f"Rust linear expects weight (out, in) matching x's last axis; "
                f"got x {source.shape}, weight {matrix.shape}"
            )
        rows = int(np.prod(source.shape[:-1], dtype=np.int64)) if source.ndim > 1 else 1
        output = np.zeros(tuple(source.shape[:-1]) + (matrix.shape[0],), dtype=np.float32)
        bias_array = None
        if bias is not None:
            bias_array = self._array(bias, "bias")
            if bias_array.shape != (matrix.shape[0],):
                raise ShapeMismatch(f"Rust linear bias {bias_array.shape} != ({matrix.shape[0]},)")
        self._lib.uzen_rust_linear(
            _pointer(source),
            _pointer(matrix),
            _pointer(bias_array) if bias_array is not None else None,
            _pointer(output),
            SIZE(rows),
            SIZE(matrix.shape[1]),
            SIZE(matrix.shape[0]),
        )
        return output

    def layernorm(self, x: Any, weight: Any | None = None, bias: Any | None = None, *, eps: float = 1e-5) -> np.ndarray:
        source = self._array(x, "x")
        if source.ndim < 2:
            raise ShapeMismatch("Rust layernorm requires rank >= 2")
        output = np.zeros_like(source)
        scale = self._array(weight, "weight") if weight is not None else None
        offset = self._array(bias, "bias") if bias is not None else None
        self._lib.uzen_rust_layernorm(
            _pointer(source),
            _pointer(scale) if scale is not None else None,
            _pointer(offset) if offset is not None else None,
            _pointer(output),
            SIZE(int(np.prod(source.shape[:-1], dtype=np.int64))),
            SIZE(source.shape[-1]),
            FLOAT(float(eps)),
        )
        return output

    def rmsnorm(self, x: Any, weight: Any | None = None, *, eps: float = 1e-6) -> np.ndarray:
        source = self._array(x, "x")
        if source.ndim < 2:
            raise ShapeMismatch("Rust rmsnorm requires rank >= 2")
        output = np.zeros_like(source)
        scale = self._array(weight, "weight") if weight is not None else None
        self._lib.uzen_rust_rmsnorm(
            _pointer(source),
            _pointer(scale) if scale is not None else None,
            _pointer(output),
            SIZE(int(np.prod(source.shape[:-1], dtype=np.int64))),
            SIZE(source.shape[-1]),
            FLOAT(float(eps)),
        )
        return output

    def softmax(self, x: Any) -> np.ndarray:
        source = self._array(x, "x")
        if source.ndim < 1:
            raise ShapeMismatch("Rust softmax requires rank >= 1")
        output = np.zeros_like(source)
        rows = int(np.prod(source.shape[:-1], dtype=np.int64)) if source.ndim > 1 else 1
        self._lib.uzen_rust_softmax(_pointer(source), _pointer(output), SIZE(rows), SIZE(source.shape[-1]))
        return output

    def gelu(self, x: Any) -> np.ndarray:
        source = self._array(x, "x")
        output = np.zeros_like(source)
        self._lib.uzen_rust_gelu(_pointer(source), _pointer(output), SIZE(source.size))
        return output

    def silu(self, x: Any) -> np.ndarray:
        source = self._array(x, "x")
        output = np.zeros_like(source)
        self._lib.uzen_rust_silu(_pointer(source), _pointer(output), SIZE(source.size))
        return output

    def _unary(self, symbol: str, x: Any) -> np.ndarray:
        source = self._array(x, "x")
        output = np.zeros_like(source)
        getattr(self._lib, symbol)(_pointer(source), _pointer(output), SIZE(source.size))
        return output

    def relu(self, x: Any) -> np.ndarray:
        return self._unary("uzen_rust_relu", x)

    def tanh(self, x: Any) -> np.ndarray:
        return self._unary("uzen_rust_tanh", x)

    def sigmoid(self, x: Any) -> np.ndarray:
        return self._unary("uzen_rust_sigmoid", x)

    def scale(self, x: Any, factor: float) -> np.ndarray:
        source = self._array(x, "x")
        output = np.zeros_like(source)
        self._lib.uzen_rust_scale(_pointer(source), _pointer(output), SIZE(source.size), FLOAT(float(factor)))
        return output

    def add(self, a: Any, b: Any) -> np.ndarray:
        left = self._array(a, "a")
        right = self._array(b, "b")
        if left.shape != right.shape:
            raise ShapeMismatch(f"Rust add needs identical shapes, got {left.shape} and {right.shape}")
        output = np.zeros_like(left)
        self._lib.uzen_rust_add(_pointer(left), _pointer(right), _pointer(output), SIZE(left.size))
        return output

    def mul(self, a: Any, b: Any) -> np.ndarray:
        left = self._array(a, "a")
        right = self._array(b, "b")
        if left.shape != right.shape:
            raise ShapeMismatch(f"Rust mul needs identical shapes, got {left.shape} and {right.shape}")
        output = np.zeros_like(left)
        self._lib.uzen_rust_mul(_pointer(left), _pointer(right), _pointer(output), SIZE(left.size))
        return output

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "rust",
            "library": str(self.path),
            "abi_version": self.version,
            "operators": sorted(RUST_OPS),
        }


def _value(env: Mapping[str, np.ndarray], ref: Any) -> np.ndarray:
    """Resolve a NIR reference; graph inputs are keyed ``input:<name>``."""
    if str(ref.node).startswith("input:"):
        return env[f"input:{str(ref.node)[len('input:'):]}"]
    return env[f"{ref.node}:{ref.port}"]


def _name(value: Any, default: str) -> str:
    text = str(value if value is not None else default)
    return text[len("param:"):] if text.startswith("param:") else text


class RustRuntime:
    """Execute the Rust-covered NIR operators on a graph, refusing the rest."""

    name = "rust"

    def __init__(self, kernels: RustKernels | None = None) -> None:
        self.kernels = kernels or RustKernels()

    def execute(
        self,
        graph: Graph,
        package: WeightPackage,
        inputs: Mapping[str, np.ndarray],
        *,
        check_finite: bool = True,
    ) -> dict[str, np.ndarray]:
        bound = resolve_bindings(graph, package)
        env: dict[str, Any] = {}
        for name in graph.inputs:
            if name not in inputs:
                raise ShapeMismatch(f"missing graph input '{name}'")
            env[f"input:{name}"] = np.ascontiguousarray(np.asarray(inputs[name], dtype=np.float32))
        for node_id in graph.topological_order():
            node = graph.nodes[node_id]
            op = node.op
            arrays = {port: _value(env, ref) for port, ref in node.inputs.items()}
            params = dict(node.params)
            weights = bound.get(node_id, {})

            def weight(key: str, default: str) -> np.ndarray | None:
                if not params.get(key):
                    return None
                name = _name(params.get(key), default)
                if name not in weights:
                    raise ShapeMismatch(f"node '{node_id}' has no bound tensor '{name}'")
                return weights[name]

            if op == "linear":
                result = self.kernels.linear(arrays["x"], weight("weight", "weight"), weight("bias", "bias"))
            elif op == "layernorm":
                result = self.kernels.layernorm(
                    arrays["x"],
                    weight("weight", "weight"),
                    weight("bias", "bias"),
                    eps=float(params.get("eps", 1e-5)),
                )
            elif op == "rmsnorm":
                result = self.kernels.rmsnorm(
                    arrays["x"], weight("weight", "weight"), eps=float(params.get("eps", 1e-6))
                )
            elif op == "softmax":
                axis = params.get("axis", -1)
                if axis not in (-1, arrays["x"].ndim - 1):
                    raise UnsupportedOperator(
                        f"the Rust softmax supports only the last axis, got {axis}", node=node_id
                    )
                result = self.kernels.softmax(arrays["x"])
            elif op == "gelu":
                result = self.kernels.gelu(arrays["x"])
            elif op == "silu":
                result = self.kernels.silu(arrays["x"])
            elif op == "relu":
                result = self.kernels.relu(arrays["x"])
            elif op == "tanh":
                result = self.kernels.tanh(arrays["x"])
            elif op == "sigmoid":
                result = self.kernels.sigmoid(arrays["x"])
            elif op == "scale":
                result = self.kernels.scale(arrays["x"], float(params.get("factor", 1.0)))
            elif op == "add":
                result = self.kernels.add(arrays["a"], arrays["b"])
            elif op == "mul":
                result = self.kernels.mul(arrays["a"], arrays["b"])
            elif op in ("identity", "dropout"):
                result = arrays["x"]
            else:
                raise UnsupportedOperator(
                    f"the Rust runtime does not implement NIR operator '{op}' (node "
                    f"'{node_id}'); supported: {', '.join(sorted(RUST_OPS))}",
                    node=node_id,
                    operator=op,
                )
            if check_finite and not np.isfinite(result).all():
                raise TensorInvalid(f"node '{node_id}' ({op}) produced non-finite values")
            env[f"{node_id}:0"] = result
        outputs: dict[str, np.ndarray] = {}
        for name, ref in graph.outputs.items():
            try:
                value = _value(env, ref)
            except KeyError as exc:
                raise ShapeMismatch(
                    f"graph output '{name}' references missing value {ref}"
                ) from exc
            outputs[name] = np.asarray(value, dtype=np.float32)
        return outputs


__all__ = [
    "MANIFEST",
    "RUST_OPS",
    "RustKernels",
    "RustRuntime",
    "build_library",
    "find_cargo",
    "library_path",
]
