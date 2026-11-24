"""RI Kernels: CPU and GPU kernels for radio interferometry, exposed to JAX."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ri_kernels")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__", "jax_api"]
