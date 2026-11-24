"""Shared-library discovery and XLA FFI target registration.

Libraries for CPU, CUDA, and ROCm are probed independently without calling
``jax.devices()``, so importing the compiled operator modules never initializes
JAX backends prematurely.
"""

import ctypes
import importlib.util
import logging
import os
import site
import sysconfig

import jax

logger = logging.getLogger(__name__)

_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
_PKG_ROOT = os.path.dirname(_DIR_PATH)

# The CPU library ships inside ri_kernels itself, while the GPU libraries come
# from optional add-on distributions (ri_kernels_cuda12, ri_kernels_cuda13,
# ri_kernels_rocm) that install a top-level package of their own containing
# nothing but the shared library. Search all of them. cuda13 is probed before
# cuda12 so that a machine with both installed picks the newer toolkit.
_PKG_NAMES = (
    "ri_kernels",
    "ri_kernels_cuda13",
    "ri_kernels_cuda12",
    "ri_kernels_rocm",
)


def _site_dirs():
    dirs = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        dirs.append(user_site)
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        dirs.append(purelib)
    return dirs


def _candidate_dirs():
    dirs = []
    override = os.environ.get("RI_KERNELS_LIB_DIR")
    if override:
        dirs.append(override)
    dirs.extend([_DIR_PATH, _PKG_ROOT])
    site_dirs = _site_dirs()
    for pkg in _PKG_NAMES:
        try:
            spec = importlib.util.find_spec(pkg)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            dirs.extend(spec.submodule_search_locations)
        for sp in site_dirs:
            dirs.append(os.path.join(sp, pkg))
    return dirs


def _load_library(name):
    seen = set()
    for d in _candidate_dirs():
        lib_path = os.path.join(d, name)
        if lib_path in seen:
            continue
        seen.add(lib_path)
        if os.path.isfile(lib_path):
            try:
                return ctypes.cdll.LoadLibrary(lib_path)
            except OSError as exc:
                logger.debug("Skipping incompatible library %s: %s", lib_path, exc)
    return None


_TAB_LIB = _load_library("libri_kernels.so")
_TAB_LIB_CUDA = _load_library("libri_kernels_cuda.so")
_TAB_LIB_ROCM = _load_library("libri_kernels_hip.so")

# Backwards-compatible alias: CUDA takes precedence when both are installed.
_TAB_LIB_GPU = _TAB_LIB_CUDA if _TAB_LIB_CUDA is not None else _TAB_LIB_ROCM


def _ffi_target(op, pass_, platform, dtype):
    """Name of an FFI handler: both its exported symbol and its XLA target."""
    return f"ri_{op}_{pass_}_{platform}_{dtype}"


def _register_targets(lib, op, passes, platform, xla_platform):
    for dtype in ("f32", "f64"):
        for pass_ in passes:
            name = _ffi_target(op, pass_, platform, dtype)
            jax.ffi.register_ffi_target(
                name, jax.ffi.pycapsule(getattr(lib, name)), platform=xla_platform
            )


def _check_tab_lib():
    if _TAB_LIB is None:
        raise RuntimeError(
            f"FFI selected, but libri_kernels.so not found! "
            f"Install the wheel built via scikit-build-core or build from "
            f"{_PKG_ROOT}"
        )


def _check_tab_lib_gpu(platform=None):
    if platform == "cuda":
        lib, lib_name = _TAB_LIB_CUDA, "libri_kernels_cuda.so"
    elif platform == "rocm":
        lib, lib_name = _TAB_LIB_ROCM, "libri_kernels_hip.so"
    else:
        lib, lib_name = _TAB_LIB_GPU, "libri_kernels_cuda.so / libri_kernels_hip.so"
    if lib is None:
        raise RuntimeError(
            f"FFI selected, but {lib_name} not found! "
            f"Install the matching add-on package (ri_kernels_cuda12 or "
            f"ri_kernels_cuda13), or rebuild the wheel with "
            f"-Ccmake.define.RI_KERNELS_CUDA=ON (or RI_KERNELS_ROCM=ON)"
        )
