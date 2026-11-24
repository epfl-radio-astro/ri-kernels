#!/usr/bin/env python3
"""Check that the shared library in a freshly built wheel loads and exports
only the expected FFI entry points.

The CUDA wheels cannot be installed on CI (they depend on ``ri_kernels`` at a
version that is not on PyPI yet) and there is no GPU to run kernels on, so
cibuildwheel's test step is disabled for them. Extracting the library,
``dlopen``-ing it, and checking its dynamic symbol table catches unresolved
symbols, missing FFI handlers, leaked static-archive symbols (Highway / CUDA
runtime), and misplaced wheel files.

    python ci/check_library.py wheelhouse ri_kernels_cuda12
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

_VIS_PASSES = ("fwd", "jvp", "transpose")
_ANALYTIC_PASSES = ("fwd", "jvp", "full_jvp", "full_transpose")
_LEAKED_SUBSTRINGS = (
    "hwy",
    "cudaLaunchKernel",
    "cudaMalloc",
    "cudaMemcpy",
    "cudaFree",
    "cudaStreamSynchronize",
    "cudaGetErrorString",
)


def _expected_symbols(lib_filename: str) -> list[str]:
    platform = "cpu" if lib_filename == "libri_kernels.so" else "gpu"
    syms = []
    for dt in ("f32", "f64"):
        for p in _VIS_PASSES:
            syms.append(f"ri_rfi_vis_{p}_{platform}_{dt}")
        for p in _ANALYTIC_PASSES:
            syms.append(f"ri_rfi_analytic_vis_{p}_{platform}_{dt}")
    return syms


def _check_nm_exports(lib_path: str, expected: list[str]) -> list[str]:
    nm = shutil.which("nm")
    if nm is None or not sys.platform.startswith("linux"):
        return []
    proc = subprocess.run(
        [nm, "-D", "--defined-only", lib_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"nm failed ({proc.returncode}): {proc.stderr.strip()}"]
    defined = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            defined.add(parts[-1])
    errors = []
    missing = [s for s in expected if s not in defined]
    if missing:
        errors.append(f"missing exported FFI symbols: {', '.join(missing)}")
    leaked = sorted(
        s for s in defined
        if not s.startswith("ri_")
        and any(tok in s for tok in _LEAKED_SUBSTRINGS)
    )
    if leaked:
        preview = ", ".join(leaked[:10])
        suffix = f" (+{len(leaked) - 10} more)" if len(leaked) > 10 else ""
        errors.append(f"leaked static-archive symbols: {preview}{suffix}")
    return errors


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
                    expected = _expected_symbols(pathlib.PurePosixPath(name).name)
                    try:
                        lib = ctypes.CDLL(path)
                    except OSError as exc:
                        print(f"error: {wheel.name}: {name} failed to load: {exc}", file=sys.stderr)
                        failed = True
                        continue
                    missing = [s for s in expected if not hasattr(lib, s)]
                    if missing:
                        print(
                            f"error: {wheel.name}: {name} missing FFI symbols: {', '.join(missing)}",
                            file=sys.stderr,
                        )
                        failed = True
                        continue
                    nm_errors = _check_nm_exports(path, expected)
                    if nm_errors:
                        for err in nm_errors:
                            print(f"error: {wheel.name}: {name}: {err}", file=sys.stderr)
                        failed = True
                        continue
                    print(f"ok: {wheel.name}: {name} loads ({len(expected)} FFI symbols verified)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
