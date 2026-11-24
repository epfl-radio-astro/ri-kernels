"""JAX bindings for the RI kernels.

Exports are loaded lazily (PEP 562) so importing pure-JAX references
(``RFIVisOpRef``, ``RFIAnalyticVisOpRef``) does not load shared libraries or
register FFI targets.
"""

import importlib

_EXPORTS = {
    "RFIAnalyticVisOp": ".rfi_analytic_vis_op",
    "RFIAnalyticVisOpRef": ".rfi_analytic_vis_op_ref",
    "RFIVisOp": ".rfi_vis_op",
    "RFIVisOpRef": ".rfi_vis_op_ref",
    "prepare_indices": ".rfi_vis_op",
}

# Submodules the eager __init__ used to import, so ``api.rfi_vis_op`` keeps
# working after a bare ``import ri_kernels.jax_api as api``.
_SUBMODULES = frozenset(
    ("rfi_analytic_vis_op", "rfi_analytic_vis_op_ref", "rfi_vis_op", "rfi_vis_op_ref")
)

__all__ = list(_EXPORTS)


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name, __name__), name)
    elif name in _SUBMODULES:
        # Importing a submodule binds it on this package itself.
        value = importlib.import_module("." + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
