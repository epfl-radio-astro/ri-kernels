#!/usr/bin/env python3
"""Turn the checked-out source tree into one of the GPU add-on distributions.

``ri_kernels`` ships the Python code plus the CPU library. The GPU kernels are
published separately as ``ri_kernels_cuda12`` / ``ri_kernels_cuda13``: wheels
that contain nothing but ``libri_kernels_cuda.so`` in a top-level package
directory of their own, and that depend on ``ri_kernels`` of the same version
for everything else. ``ri_kernels/jax_api/rfi_vis_op.py`` finds the library
there at import time (see ``_PKG_NAMES``).

Rather than maintaining three near-identical ``pyproject.toml`` files, this
script rewrites the one in the tree in place, so the build metadata that is
shared between all three variants (``[build-system]``, ``[tool.cibuildwheel]``,
the version) has exactly one definition. CI runs it on a disposable checkout
immediately before invoking cibuildwheel:

    python ci/make_variant.py --cuda 12

It is a build-time tool only; it is never executed on an end user's machine.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import tomlkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

STUB_INIT = '''"""CUDA {major} kernels for :mod:`ri_kernels`.

This package exists only to give ``libri_kernels_cuda.so`` a location that pip
owns exclusively. Import :mod:`ri_kernels.jax_api` to use the kernels; it picks
the library up from here automatically.
"""
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        required=True,
        choices=("12", "13"),
        help="CUDA major version the wheel is built against",
    )
    args = parser.parse_args(argv)

    major = args.cuda
    name = f"ri_kernels_cuda{major}"

    doc = tomlkit.parse(PYPROJECT.read_text())
    project = doc["project"]
    version = str(project["version"])

    # `ri_kernels[cuda{major}]` pins this add-on at the version it is built
    # alongside, so the version string lives in the extras table as well as in
    # `[project].version`. Catch the two falling out of step here, before the
    # table is replaced below - CI runs this for both CUDA majors on every push.
    extra = f"cuda{major}"
    pin = f"{name}=={version}"
    requirements = [str(req) for req in project["optional-dependencies"][extra]]
    if pin not in requirements:
        print(
            f"error: the '{extra}' extra in pyproject.toml does not pin "
            f"'{pin}'; it lists {requirements}. Update it to match "
            f"[project].version.",
            file=sys.stderr,
        )
        return 1

    project["name"] = name
    project["description"] = f"RI Kernels - CUDA {major} kernels"
    # The add-on carries no Python code of its own; everything comes from the
    # base package, which must be the exact same build.
    project["dependencies"] = [f"ri_kernels=={version}"]
    project["optional-dependencies"] = {"jax": [f"jax[cuda{major}]>=0.6.0"]}

    skb = doc["tool"]["scikit-build"]
    # Written in the file as the dotted key `wheel.packages`, which tomlkit
    # exposes as the nested table it is.
    skb["wheel"]["packages"] = [name]
    # The env-var overrides are how the *base* package opts into a GPU build
    # locally; here the defines are unconditional, so drop them to keep the
    # generated metadata unambiguous.
    skb.pop("overrides", None)
    skb[tomlkit.key(["cmake", "define", "RI_KERNELS_CPU"])] = "OFF"
    skb[tomlkit.key(["cmake", "define", "RI_KERNELS_CUDA"])] = "ON"
    skb[tomlkit.key(["cmake", "define", "RI_KERNELS_GPU_INSTALL_DIR"])] = name

    # CMakeLists.txt scrapes the version straight out of this file; make sure
    # the rewrite did not disturb the line it matches on.
    text = tomlkit.dumps(doc)
    if f'\nversion = "{version}"' not in text:
        print(
            "error: the 'version' line CMakeLists.txt parses is no longer "
            "present in the generated pyproject.toml",
            file=sys.stderr,
        )
        return 1
    PYPROJECT.write_text(text)

    pkg_dir = ROOT / name
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text(STUB_INIT.format(major=major))

    print(f"prepared {name} {version} (package dir: {pkg_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
