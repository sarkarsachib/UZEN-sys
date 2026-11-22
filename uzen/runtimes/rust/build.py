"""Build the Rust kernel library.

    python -m uzen.runtimes.rust.build [--debug] [--force] [--json]

Compilation needs a Rust toolchain (``cargo``); nothing is downloaded and no
prebuilt binary is bundled. The report is printed as key/value pairs or JSON so a
build script can consume it.
"""

from __future__ import annotations

import argparse
import json
import sys

from uzen.errors import UzenError

from .runtime import build_library, find_cargo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="build the debug profile")
    parser.add_argument("--force", action="store_true", help="rebuild even if a library exists")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    cargo = find_cargo()
    if cargo is None:
        print(
            "cargo not found: install a Rust toolchain from https://rustup.rs to build "
            "the Rust kernels",
            file=sys.stderr,
        )
        return 2
    try:
        report = build_library(release=not args.debug, force=args.force)
    except UzenError as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report["library"])
        print(f"cargo: {report['cargo']}")
        print(f"profile: {report['profile']}")
        print(f"rebuilt: {report['built']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
