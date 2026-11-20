"""Build (compile) the CUDA kernels to PTX.

Usage::

    python -m uzen.runtimes.cuda.build
    python -m uzen.runtimes.cuda.build --arch compute_80 --force
    python -m uzen.runtimes.cuda.build --output /tmp/uzen_cuda.ptx

Compilation needs NVRTC or nvcc, not a GPU. Loading and running the PTX needs a
CUDA driver and device; this command does not claim either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nvrtc import DEFAULT_ARCH, KERNELS, compile_ptx, describe_compilers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    path, report = compile_ptx(arch=args.arch, force=args.force, output=args.output)
    payload = {
        "ptx": str(path),
        "compilers": describe_compilers(),
        "kernels": list(KERNELS),
        **report,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(path)
        print(f"compiler: {report.get('compiler')} ({report.get('compiler_version') or report.get('compiler_path')})")
        print(f"kernels: {len(KERNELS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
