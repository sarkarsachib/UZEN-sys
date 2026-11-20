"""Build the native C++ kernel library.

The source ships with the package; this module compiles it with the local C++
toolchain into a platform-specific shared library and caches the result by source
digest. It never downloads anything and never executes repository code: only the
checked-in ``kernels.cpp`` is compiled.

Usage::

    python -m uzen.runtimes.native.build
    python -m uzen.runtimes.native.build --output /tmp/libuzen_native.so --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from uzen.errors import RuntimeUnavailable

SOURCE = Path(__file__).with_name("kernels.cpp")
BASE_FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared", "-Wno-unknown-pragmas")
OPENMP_FLAG = "-fopenmp"


def library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def native_compiler() -> str | None:
    """Return the C++ compiler this environment can use, or None."""
    explicit = os.environ.get("CXX")
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    for candidate in ("g++", "clang++", "c++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def source_digest(source: Path = SOURCE) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update("|".join(BASE_FLAGS).encode("utf-8"))
    digest.update(platform.machine().encode("utf-8"))
    digest.update(sys.platform.encode("utf-8"))
    return digest.hexdigest()[:16]


def cache_directory() -> Path:
    override = os.environ.get("UZEN_NATIVE_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "uzen" / "native"


def default_cache_path(source: Path = SOURCE) -> Path:
    return cache_directory() / f"libuzen_native-{source_digest(source)}{library_suffix()}"


def build_native_library(
    output: str | Path | None = None,
    *,
    compiler: str | None = None,
    extra_flags: tuple[str, ...] = (),
    force: bool = False,
) -> Path:
    """Compile ``kernels.cpp`` into a shared library and return its path."""
    compiler = compiler or native_compiler()
    if compiler is None:
        raise RuntimeUnavailable(
            "no C++ compiler found; install g++ or clang++ (or set CXX) to build "
            "UZEN's native CPU kernels"
        )
    destination = Path(output) if output is not None else default_cache_path()
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)

    base_command = [
        compiler,
        *BASE_FLAGS,
        *extra_flags,
        str(SOURCE),
        "-o",
        str(temporary),
    ]
    attempts = [
        [*base_command[:-3], OPENMP_FLAG, *base_command[-3:]],
        base_command,
    ]
    errors: list[str] = []
    for command in attempts:
        try:
            completed = subprocess.run(
                command,
                cwd=str(SOURCE.parent),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{exc}")
            continue
        if completed.returncode == 0 and temporary.exists():
            os.replace(temporary, destination)
            return destination
        errors.append((completed.stderr or completed.stdout or "").strip()[-2000:])
    temporary.unlink(missing_ok=True)
    raise RuntimeUnavailable(
        "failed to build the native kernel library",
        compiler=compiler,
        errors=[error for error in errors if error],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compiler", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = build_native_library(
        args.output, compiler=args.compiler, force=args.force
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
