#!/usr/bin/env python3
"""Check that the shared library in a freshly built wheel actually loads.

The CUDA wheels cannot be installed on CI (they depend on ``ri_kernels`` at a
version that is not on PyPI yet) and there is no GPU to run kernels on, so
cibuildwheel's test step is disabled for them. Extracting the library and
``dlopen``-ing it is still worth doing: it catches unresolved symbols, a missing
install rule, and a library that ended up in the wrong directory inside the
wheel - the three ways this build has to go wrong silently.

    python ci/check_library.py wheelhouse ri_kernels_cuda12
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import tempfile
import zipfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=pathlib.Path, help="directory holding the wheels")
    parser.add_argument("package", help="top-level package the library must live in")
    args = parser.parse_args(argv)

    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if not wheels:
        print(f"error: no wheels found in {args.wheelhouse}", file=sys.stderr)
        return 1

    failed = False
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            libs = [
                n
                for n in zf.namelist()
                if n.startswith(f"{args.package}/") and n.endswith(".so")
            ]
            if not libs:
                print(f"error: {wheel.name} contains no .so under {args.package}/", file=sys.stderr)
                print("  contents: " + ", ".join(zf.namelist()), file=sys.stderr)
                failed = True
                continue
            with tempfile.TemporaryDirectory() as tmp:
                for name in libs:
                    path = zf.extract(name, tmp)
                    try:
                        ctypes.CDLL(path)
                    except OSError as exc:
                        print(f"error: {wheel.name}: {name} failed to load: {exc}", file=sys.stderr)
                        failed = True
                    else:
                        print(f"ok: {wheel.name}: {name} loads")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
