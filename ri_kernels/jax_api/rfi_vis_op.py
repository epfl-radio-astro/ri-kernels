"""
Custom JAX primitives for RFI visibility calculation using FFI.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir, xla

from ._common import _dtype_suffix
from ._lib import (
    _TAB_LIB,
    _TAB_LIB_CUDA,
    _TAB_LIB_GPU as _TAB_LIB_GPU,
    _TAB_LIB_ROCM,
    _check_tab_lib,
    _check_tab_lib_gpu,
    _ffi_target,
    _register_targets,
)


def prepare_indices(n_ant, a1, a2):
    """
    Prepare indices for sorted usage in the C++ kernel.

    Args:
        n_ant: Number of antennas.
        a1: Antenna 1 indices.
        a2: Antenna 2 indices.

    Returns:
        A tuple (a1_sorter, a1_start, a2_sorter, a2_start) suitable for the kernel.
    """
    a1_sorter = jnp.argsort(a1).astype(a1.dtype)
    a2_sorter = jnp.argsort(a2).astype(a1.dtype)

    v = jnp.arange(0, n_ant, dtype=a1.dtype)
    a1_start = jnp.searchsorted(a1, v, sorter=a1_sorter).astype(a1.dtype)
    a2_start = jnp.searchsorted(a2, v, sorter=a2_sorter).astype(a1.dtype)

    return (a1_sorter, a1_start, a2_sorter, a2_start)


class RFIVisOp:
    """
    Operator for computing RFI visibility using JAX FFI with precomputed indices.

    This class encapsulates the antenna baseline indexing logic required for efficient
    RFI visibility calculations. It precomputes sorted indices and search positions
    for both antenna arrays (a1 and a2) to enable fast lookups in the FFI kernel.
    """

    def __init__(self, n_ant, a1, a2):
        """
        Initialize the RFI visibility operator with antenna baseline information.

        Args:
            n_ant: Number of antennas in the array.
            a1: Array of first antenna indices for each baseline (shape: [n_baselines]).
            a2: Array of second antenna indices for each baseline (shape: [n_baselines]).

        The initialization precomputes:
            - a1_sorter, a2_sorter: Indices that would sort the a1 and a2 arrays
            - a1_start, a2_start: Starting positions for each antenna in the sorted arrays
        """
        a1 = jnp.asarray(a1)
        a2 = jnp.asarray(a2)
        if (
            n_ant < 1
            or a1.ndim != 1
            or a2.shape != a1.shape
            or not jnp.issubdtype(a1.dtype, jnp.integer)
            or not jnp.issubdtype(a2.dtype, jnp.integer)
        ):
            raise ValueError(
                "Expected n_ant >= 1 and matching 1D integer antenna-index vectors"
            )
        self.a1 = a1.astype(jnp.int32)
        self.a2 = a2.astype(jnp.int32)
        (
            self.a1_sorter,
            self.a1_start,
            self.a2_sorter,
            self.a2_start,
        ) = prepare_indices(n_ant, self.a1, self.a2)

    def eval(self, rfi_amp_fine, rfi_phase):
        """
        Evaluate the RFI visibility for given RFI amplitudes and phases.

        Args:
            rfi_amp_fine: Fine-grained RFI amplitude array with shape
                         (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time).
            rfi_phase: RFI phase array in radians with shape matching rfi_amp_fine.
                      On CPU, Highway's vectorised sin/cos is accurate for phase
                      differences within [-39000, +39000] radians; reduce larger
                      phases to a bounded range beforehand.

        Returns:
            Array of RFI visibilities with shape (n_baselines, n_freq, n_time).
        """
        return rfi_vis_op.bind(
            self.a1, self.a1_sorter, self.a1_start,
            self.a2, self.a2_sorter, self.a2_start,
            rfi_amp_fine, rfi_phase,
        )


_PASSES = ("fwd", "jvp", "transpose")

if _TAB_LIB:
    _register_targets(_TAB_LIB, "rfi_vis", _PASSES, "cpu", "cpu")

if _TAB_LIB_CUDA:
    _register_targets(_TAB_LIB_CUDA, "rfi_vis", _PASSES, "gpu", "CUDA")

if _TAB_LIB_ROCM:
    _register_targets(_TAB_LIB_ROCM, "rfi_vis", _PASSES, "gpu", "ROCM")


def _validate_rfi_inputs(a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase):
    _dtype_suffix(rfi_amp_fine.dtype, rfi_phase.dtype)
    if rfi_amp_fine.ndim != 6:
        raise ValueError(
            "Expected rfi_amp_fine of shape (n_ant, n_freq, n_time, n_rfi, "
            f"n_int_freq, n_int_time); got {rfi_amp_fine.shape}"
        )
    if rfi_phase.shape != rfi_amp_fine.shape:
        raise ValueError(
            f"Expected rfi_phase of shape {rfi_amp_fine.shape}; got {rfi_phase.shape}"
        )
    if a1.ndim != 1:
        raise ValueError(f"Expected 1D baseline index array a1; got {a1.shape}")
    n_bl = a1.shape[0]
    n_ant = rfi_amp_fine.shape[0]
    for name, arr, expected_shape in (
        ("a1", a1, (n_bl,)),
        ("a1_sorter", a1_sorter, (n_bl,)),
        ("a1_start", a1_start, (n_ant,)),
        ("a2", a2, (n_bl,)),
        ("a2_sorter", a2_sorter, (n_bl,)),
        ("a2_start", a2_start, (n_ant,)),
    ):
        if arr.shape != expected_shape or jnp.dtype(arr.dtype) != jnp.int32:
            raise ValueError(
                f"Expected int32 {name} of shape {expected_shape}; got {arr.shape} {arr.dtype}"
            )


def _validate_like(name, primal, other):
    if other.shape != primal.shape or jnp.dtype(other.dtype) != jnp.dtype(primal.dtype):
        raise ValueError(
            f"Expected {name} shape {primal.shape} and dtype {primal.dtype}; "
            f"got {other.shape} and {other.dtype}"
        )


def _lowering(pass_, platform, phase_idx=7):
    ffi_platform = "cpu" if platform == "cpu" else "gpu"

    def lowering(ctx, *args):
        if platform == "cpu":
            _check_tab_lib()
        else:
            _check_tab_lib_gpu(platform)
        suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[phase_idx].dtype)
        return jax.ffi.ffi_lowering(_ffi_target("rfi_vis", pass_, ffi_platform, suffix))(ctx, *args)

    return lowering


def _register_lowerings(primitive, pass_, phase_idx=7):
    for platform in ("cpu", "cuda", "rocm"):
        mlir.register_lowering(primitive, _lowering(pass_, platform, phase_idx), platform=platform)


# --- Transpose Op ---

rfi_transpose_op = core.Primitive("rfi_transpose_op")
rfi_transpose_op.def_impl(partial(xla.apply_primitive, rfi_transpose_op))
rfi_transpose_op.multiple_results = True


def rfi_transpose_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
):
    """Abstract evaluation for the RFI transpose operation."""
    _validate_rfi_inputs(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
    )
    expected_g = ShapedArray(
        (a1.shape[0], rfi_amp_fine.shape[1], rfi_amp_fine.shape[2]), rfi_amp_fine.dtype
    )
    _validate_like("visibility cotangent", expected_g, g)
    t1 = ShapedArray(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    t2 = ShapedArray(rfi_phase.shape, rfi_phase.dtype)
    return (t1, t2)


rfi_transpose_op.def_abstract_eval(rfi_transpose_abstract)
_register_lowerings(rfi_transpose_op, "transpose", phase_idx=7)


# --- JVP Op ---

rfi_jvp_op = core.Primitive("rfi_jvp_op")
rfi_jvp_op.def_impl(partial(xla.apply_primitive, rfi_jvp_op))


def rfi_jvp_abstract(
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """Abstract evaluation for the RFI JVP operation."""
    _validate_rfi_inputs(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
    )
    _validate_like("rfi_amp_fine tangent", rfi_amp_fine, rfi_amp_fine_grad)
    _validate_like("rfi_phase tangent", rfi_phase, rfi_phase_grad)
    n_freq = rfi_amp_fine.shape[1]
    n_time = rfi_amp_fine.shape[2]
    return ShapedArray((a1.shape[0], n_freq, n_time), rfi_amp_fine.dtype)


rfi_jvp_op.def_abstract_eval(rfi_jvp_abstract)
_register_lowerings(rfi_jvp_op, "jvp", phase_idx=8)


def rfi_jvp_transpose(
    g,
    a1,
    a1_sorter,
    a1_start,
    a2,
    a2_sorter,
    a2_start,
    rfi_amp_fine,
    rfi_amp_fine_grad,
    rfi_phase,
    rfi_phase_grad,
):
    """Transpose rule for the RFI JVP operation."""
    t1, t2 = rfi_transpose_op.bind(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase, g
    )

    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        t1 if ad.is_undefined_primal(rfi_amp_fine_grad) else None,
        None,
        t2 if ad.is_undefined_primal(rfi_phase_grad) else None,
    )


ad.primitive_transposes[rfi_jvp_op] = rfi_jvp_transpose


# --- Vis Op ---

rfi_vis_op = core.Primitive("rfi_vis_op")
rfi_vis_op.def_impl(partial(xla.apply_primitive, rfi_vis_op))


def rfi_vis_abstract(
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
):
    """Abstract evaluation for the RFI visibility operation."""
    _validate_rfi_inputs(
        a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
    )
    n_freq = rfi_amp_fine.shape[1]
    n_time = rfi_amp_fine.shape[2]
    return ShapedArray((a1.shape[0], n_freq, n_time), rfi_amp_fine.dtype)


rfi_vis_op.def_abstract_eval(rfi_vis_abstract)
_register_lowerings(rfi_vis_op, "fwd", phase_idx=7)


def rfi_vis_jvp(args, tangents):
    """JVP rule for the RFI visibility operation."""
    a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase = args
    _, _, _, _, _, _, rfi_amp_fine_dot, rfi_phase_dot = tangents

    if isinstance(rfi_amp_fine_dot, ad.Zero):
        rfi_amp_fine_dot = jnp.zeros(rfi_amp_fine.shape, rfi_amp_fine.dtype)
    if isinstance(rfi_phase_dot, ad.Zero):
        rfi_phase_dot = jnp.zeros(rfi_phase.shape, rfi_phase.dtype)

    grad = rfi_jvp_op.bind(
        a1,
        a1_sorter,
        a1_start,
        a2,
        a2_sorter,
        a2_start,
        rfi_amp_fine,
        rfi_amp_fine_dot,
        rfi_phase,
        rfi_phase_dot,
    )

    return (
        rfi_vis_op.bind(
            a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine, rfi_phase
        ),
        grad,
    )


ad.primitive_jvps[rfi_vis_op] = rfi_vis_jvp
