import os
import sys

import jax

# The f64 FFI kernels are only reachable with x64 enabled, so turn it on before
# any jax computation happens. Everything in the tests is explicitly typed, so
# the change of default dtypes does not affect them.
jax.config.update("jax_enable_x64", True)

# Allow running the tests straight from a checkout (pytest only puts the tests
# directory on sys.path), while still preferring an installed ri_kernels.
try:  # pragma: no cover - trivial import plumbing
    import ri_kernels  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
