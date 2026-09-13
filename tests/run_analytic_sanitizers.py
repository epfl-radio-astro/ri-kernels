"""Run the whole analytic test file against a particular instrumented library.

Preload the matching ASan runtime before starting Python (LD_PRELOAD on Linux,
DYLD_INSERT_LIBRARIES on macOS). The library and Python wrapper are staged in
an isolated temporary package: no shared object is written into the checkout
and an editable installation cannot silently substitute a different library.

Usage: python -u tests/run_analytic_sanitizers.py /path/to/libri_kernels.so
Additional arguments are passed to pytest, for example -k general_degree_bounds.
"""
import importlib.machinery
import os
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    library = Path(sys.argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    os.environ["JAX_PLATFORMS"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="ri-analytic-sanitizers-") as directory:
        package = Path(directory) / "ri_kernels"
        shutil.copytree(root / "ri_kernels", package,
                        ignore=shutil.ignore_patterns("*.so", "__pycache__"))
        target = package / "libri_kernels.so"
        shutil.copy2(library, target)
        sys.path.insert(0, directory)
        sys.meta_path.insert(0, importlib.machinery.PathFinder)
        import jax
        import jaxlib
        from ri_kernels.jax_api.rfi_analytic_vis_op import _TAB_LIB_ANALYTIC
        if _TAB_LIB_ANALYTIC is None or Path(_TAB_LIB_ANALYTIC._name).resolve() != target.resolve():
            raise RuntimeError("Tests did not load the requested analytic library")
        print(f"JAX {jax.__version__}, jaxlib {jaxlib.__version__}", flush=True)
        print(f"Instrumented build: {library}", flush=True)
        print(f"Loaded library: {_TAB_LIB_ANALYTIC._name}", flush=True)
        import pytest
        return pytest.main([str(root / "tests/test_rfi_analytic_vis_op.py"),
                            "-q", "-s", "-x", *sys.argv[2:]])


if __name__ == "__main__":
    sys.exit(main())
