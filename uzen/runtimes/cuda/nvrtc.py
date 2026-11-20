"""Compile the CUDA kernel source to PTX.

Two real compilers are supported:

* **NVRTC** (``libnvrtc``) — the normal path for a pip-installed package: it
  compiles the checked-in ``kernels.cu`` at runtime, needs no CUDA toolkit, and
  can run on a machine with no GPU (compilation does not create a device).
* **nvcc** — used when present (``UZEN_CUDA_COMPILER`` overrides), which lets the
  operator precompile the same source.

Both paths are cache-by-source-digest. Nothing is downloaded, and no compiler is
simulated: if neither is available the call raises :class:`RuntimeUnavailable`
with the exact reason.

Compiling proves the kernels are valid CUDA and produces loadable PTX. It does
**not** prove execution: that requires a CUDA device and driver.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import locale
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from uzen.errors import RuntimeUnavailable

SOURCE = Path(__file__).with_name("kernels.cu")
DEFAULT_ARCH = "compute_60"

#: Every kernel the runtime is allowed to dispatch. Tests assert that the
#: compiled PTX contains all of them, so a missing kernel is a build failure,
#: not a silent fallback.
KERNELS = (
    "uzen_linear_f32",
    "uzen_matmul_f32",
    "uzen_layernorm_f32",
    "uzen_rmsnorm_f32",
    "uzen_softmax_f32",
    "uzen_gelu_tanh_f32",
    "uzen_gelu_exact_f32",
    "uzen_silu_f32",
    "uzen_relu_f32",
    "uzen_tanh_f32",
    "uzen_sigmoid_f32",
    "uzen_binary_f32",
    "uzen_scale_f32",
    "uzen_clip_f32",
    "uzen_embedding_f32",
    "uzen_reduce_last_f32",
    "uzen_attention_f32",
)


try:
    _libc = ctypes.CDLL(None)
    _libc.uselocale.argtypes = [ctypes.c_void_p]
    _libc.uselocale.restype = ctypes.c_void_p
except (OSError, AttributeError):  # pragma: no cover - non-POSIX platform
    _libc = None


@contextmanager
def _preserved_locale() -> Iterator[None]:
    """NVRTC switches the calling thread to an ASCII locale and leaves it there.

    ``setlocale`` alone does not undo it because the change is thread-local, which
    breaks unrelated host code after a compile (for example writing a non-ASCII
    tokenizer file with the default encoding). Capture both the global and the
    thread locale and restore each in a ``finally`` block.
    """
    try:
        saved = locale.setlocale(locale.LC_ALL, None)
    except locale.Error:  # pragma: no cover - exotic platform
        saved = None
    thread_locale = _libc.uselocale(None) if _libc is not None else None
    try:
        yield
    finally:
        if _libc is not None and thread_locale is not None:
            _libc.uselocale(thread_locale)
        if saved is not None:
            try:
                locale.setlocale(locale.LC_ALL, saved)
            except locale.Error:  # pragma: no cover
                pass


def _candidate_nvrtc_paths() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("UZEN_NVRTC_LIBRARY")
    if explicit:
        candidates.append(explicit)
    discovered = ctypes.util.find_library("nvrtc")
    if discovered:
        candidates.append(discovered)
    roots = {
        Path(sys.prefix) / "lib",
        Path(sys.base_prefix) / "lib",
    }
    for root in roots:
        candidates.extend(str(path) for path in sorted(root.glob("libnvrtc.so*")))
    for site in (Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",):
        candidates.extend(
            str(path)
            for path in sorted(site.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
        )
    return candidates


def find_nvrtc() -> str | None:
    for candidate in _candidate_nvrtc_paths():
        if candidate and Path(candidate).exists():
            return candidate
    return None


def find_nvcc() -> str | None:
    explicit = os.environ.get("UZEN_CUDA_COMPILER")
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    return shutil.which("nvcc")


class Nvrtc:
    """Minimal, typed ctypes binding to the NVRTC compiler API."""

    def __init__(self, library: str | None = None) -> None:
        path = library or find_nvrtc()
        if not path:
            raise RuntimeUnavailable(
                "NVRTC is not available; install the 'nvidia-cuda-nvrtc-cu12' wheel "
                "or a CUDA toolkit, or set UZEN_NVRTC_LIBRARY",
                searched=[str(item) for item in _candidate_nvrtc_paths()],
            )
        self.path = path
        try:
            with _preserved_locale():
                self._lib = ctypes.CDLL(path)
        except OSError as exc:
            raise RuntimeUnavailable(f"cannot load NVRTC from {path}: {exc}") from exc
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.nvrtcVersion.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.nvrtcVersion.restype = ctypes.c_int
        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.nvrtcCreateProgram.restype = ctypes.c_int
        lib.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.nvrtcCompileProgram.restype = ctypes.c_int
        lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
        lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetProgramLog.restype = ctypes.c_int
        lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetPTXSize.restype = ctypes.c_int
        lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvrtcGetPTX.restype = ctypes.c_int
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.nvrtcDestroyProgram.restype = ctypes.c_int

    def version(self) -> tuple[int, int]:
        major = ctypes.c_int()
        minor = ctypes.c_int()
        with _preserved_locale():
            status = self._lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        if status != 0:
            raise RuntimeUnavailable("NVRTC refused nvrtcVersion", status=status)
        return int(major.value), int(minor.value)

    def compile(self, source: str, *, arch: str, options: tuple[str, ...] = ()) -> tuple[str, str]:
        with _preserved_locale():
            return self._compile(source, arch=arch, options=options)

    def _compile(
        self, source: str, *, arch: str, options: tuple[str, ...] = ()
    ) -> tuple[str, str]:
        program = ctypes.c_void_p()
        encoded = source.encode("utf-8")
        status = self._lib.nvrtcCreateProgram(
            ctypes.byref(program), encoded, b"uzen_kernels.cu", 0, None, None
        )
        if status != 0:
            raise RuntimeUnavailable("NVRTC could not create a program", status=status)
        try:
            flags = [f"--gpu-architecture={arch}", "--std=c++14", *options]
            array = (ctypes.c_char_p * len(flags))(*[flag.encode() for flag in flags])
            status = self._lib.nvrtcCompileProgram(program, len(flags), array)
            log = self._program_log(program)
            if status != 0:
                raise RuntimeUnavailable(
                    "NVRTC rejected the CUDA kernel source",
                    status=status,
                    log=log[-4000:],
                )
            size = ctypes.c_size_t()
            if self._lib.nvrtcGetPTXSize(program, ctypes.byref(size)) != 0 or not size.value:
                raise RuntimeUnavailable("NVRTC produced no PTX", log=log[-2000:])
            buffer = ctypes.create_string_buffer(size.value)
            if self._lib.nvrtcGetPTX(program, buffer) != 0:
                raise RuntimeUnavailable("NVRTC failed to return PTX", log=log[-2000:])
            return buffer.value.decode("utf-8"), log
        finally:
            self._lib.nvrtcDestroyProgram(ctypes.byref(program))

    def _program_log(self, program: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        if self._lib.nvrtcGetProgramLogSize(program, ctypes.byref(size)) != 0 or not size.value:
            return ""
        buffer = ctypes.create_string_buffer(size.value)
        if self._lib.nvrtcGetProgramLog(program, buffer) != 0:
            return ""
        return buffer.value.decode("utf-8", errors="replace")


def cache_directory() -> Path:
    override = os.environ.get("UZEN_CUDA_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "uzen" / "cuda"


def source_digest(source: str | None = None, *, arch: str = DEFAULT_ARCH) -> str:
    digest = hashlib.sha256()
    digest.update((source if source is not None else SOURCE.read_text(encoding="utf-8")).encode("utf-8"))
    digest.update(b"|")
    digest.update(arch.encode("utf-8"))
    digest.update(b"|nvrtc-or-nvcc|float32")
    return digest.hexdigest()[:16]


def cached_ptx_path(*, arch: str = DEFAULT_ARCH) -> Path:
    return cache_directory() / f"uzen_cuda-{source_digest(arch=arch)}.ptx"


def compile_ptx(
    *,
    source: str | None = None,
    arch: str | None = None,
    force: bool = False,
    output: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Compile the kernel source and return ``(ptx_path, report)``.

    The cache is keyed by source digest and target architecture. The report names
    the compiler and whether it was a cache hit, so callers cannot confuse a
    re-used artifact with a fresh compile.
    """
    architecture = arch or os.environ.get("UZEN_CUDA_ARCH", DEFAULT_ARCH)
    text = source if source is not None else SOURCE.read_text(encoding="utf-8")
    destination = Path(output) if output is not None else cached_ptx_path(arch=architecture)
    if destination.exists() and not force:
        return destination, {
            "compiler": "cache",
            "arch": architecture,
            "path": str(destination),
            "cache_hit": True,
            "bytes": destination.stat().st_size,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)

    report: dict[str, Any] = {"arch": architecture, "cache_hit": False}
    nvcc = find_nvcc()
    if nvcc:
        source_path = SOURCE if source is None else destination.with_suffix(".cu")
        if source is not None:
            source_path.write_text(text, encoding="utf-8")
        command = [
            nvcc,
            "-ptx",
            f"-arch={architecture}",
            "-O3",
            "--std=c++14",
            "-o",
            str(temporary),
            str(source_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeUnavailable(
                "nvcc rejected the CUDA kernel source",
                compiler=nvcc,
                errors=(completed.stderr or completed.stdout)[-4000:],
            )
        report.update({"compiler": "nvcc", "compiler_path": nvcc})
    else:
        compiler = Nvrtc()
        ptx, log = compiler.compile(text, arch=architecture)
        temporary.write_text(ptx, encoding="utf-8")
        major, minor = compiler.version()
        report.update(
            {
                "compiler": "nvrtc",
                "compiler_path": compiler.path,
                "compiler_version": f"{major}.{minor}",
                "log": log.strip(),
            }
        )
    os.replace(temporary, destination)
    report.update(
        {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "kernels": list(KERNELS),
        }
    )
    return destination, report


def compiler_available() -> bool:
    return find_nvcc() is not None or find_nvrtc() is not None


def describe_compilers() -> dict[str, Any]:
    nvcc = find_nvcc()
    nvrtc = find_nvrtc()
    version: str | None = None
    if nvrtc:
        try:
            major, minor = Nvrtc(nvrtc).version()
            version = f"{major}.{minor}"
        except RuntimeUnavailable:
            version = None
    return {
        "nvcc": nvcc,
        "nvrtc": nvrtc,
        "nvrtc_version": version,
        "can_compile": bool(nvcc or nvrtc),
    }
