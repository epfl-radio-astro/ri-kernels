"""Compiled analytic RFI visibility and its amplitude JVP and transpose."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir, xla

from .rfi_interp_vis_op import RFIInterpVisOp, TILE

from .rfi_vis_op import (
    _TAB_LIB,
    _TAB_LIB_GPU,
    _TAB_PLATFORM_NAME,
    _check_tab_lib,
    _check_tab_lib_gpu,
    _dtype_suffix,
)


class RFIAnalyticVisOp(RFIInterpVisOp):
    """RFI visibility from amplitude coefficients and analytic phase moments.

    The baseline staging and input axis order are those of RFIInterpVisOp.
    Only amp is differentiated; the orbit, integration interval and tables
    are constants. The recurrence has no time sample axis or Nyquist floor.
    """

    def __init__(self, n_ant, a1, a2):
        a1, a2 = np.asarray(a1), np.asarray(a2)
        if (a1.ndim != 1 or a2.shape != a1.shape or
                a1.dtype.kind not in "iu" or a2.dtype.kind not in "iu" or
                n_ant < 1 or np.any(a1 < 0) or np.any(a2 < 0) or
                np.any(a1 >= n_ant) or np.any(a2 >= n_ant)):
            raise ValueError("Expected matching antenna-index vectors within [0, n_ant)")
        super().__init__(n_ant, jnp.asarray(a1, jnp.int32), jnp.asarray(a2, jnp.int32))

    def eval(self, amp, phase, delay_us, w_freq, start_freq, g_time,
             start_time, dnu_mhz, int_time, freq_mhz, *, segments=2, terms=6,
             cubic_terms=3):
        """Return complex (n_bl, n_freq, n_time) visibilities.

        amp and phase have shape (n_ant, n_rfi, n_freq, n_time); delay_us
        is (n_ant, n_rfi, n_time, n_path), with delay derivatives in us/s^k.
        phase holds the reduced centre phase, computed in float64 before
        casting. freq_mhz and dnu_mhz are in MHz, not config.freqs' Hz.

        g_time[t,l,m] holds monomial coefficients in x = 2*tau/int_time,
        including shifted edge stencils. w_freq and the stencil starts are
        identical to RFIInterpVisOp's tables. int_time is a positive scalar
        in seconds, with the same real dtype as phase and all other tables.
        Each stencil must lie in its axis and contain its own cell.

        segments splits [-1,1] equally; terms controls the curvature series,
        cubic_terms the residual cubic series (zero drops cubic phase).
        Derivatives above order three are omitted, as in analytic_rfi_vis.
        The defaults are the measured two-piece configuration; 4,16,3 gives
        the reference's more conservative configuration. Coefficient counts
        up to 9, terms up to 32 and cubic_terms up to 8 are supported.
        """
        _validate_options(segments=segments, terms=terms, cubic_terms=cubic_terms)
        stop = jax.lax.stop_gradient
        int_time = jnp.asarray(int_time, dtype=None if hasattr(int_time, "dtype") else phase.dtype)
        return rfi_analytic_vis_op.bind(
            *self.indices, amp, stop(phase), stop(delay_us), stop(w_freq),
            start_freq, stop(g_time), start_time, stop(dnu_mhz), stop(int_time),
            stop(freq_mhz), segments=segments, terms=terms, cubic_terms=cubic_terms,
        )


_TAB_LIB_ANALYTIC = (
    _TAB_LIB if _TAB_LIB and hasattr(_TAB_LIB, "calc_rfi_analytic_cpu_f32") else None
)
_TAB_LIB_ANALYTIC_GPU = (
    _TAB_LIB_GPU
    if _TAB_LIB_GPU and hasattr(_TAB_LIB_GPU, "calc_rfi_analytic_gpu_f32")
    else None
)

if _TAB_LIB_ANALYTIC:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            jax.ffi.register_ffi_target(
                f"calc_rfi_analytic{_kind}_{_suffix}",
                jax.ffi.pycapsule(
                    getattr(_TAB_LIB_ANALYTIC, f"calc_rfi_analytic{_kind}_cpu_{_suffix}")
                ),
                platform="cpu",
            )

if _TAB_LIB_ANALYTIC_GPU:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            jax.ffi.register_ffi_target(
                f"calc_rfi_analytic{_kind}_gpu_{_suffix}",
                jax.ffi.pycapsule(
                    getattr(_TAB_LIB_ANALYTIC_GPU, f"calc_rfi_analytic{_kind}_gpu_{_suffix}")
                ),
                platform=_TAB_PLATFORM_NAME,
            )


def _check_analytic_lib(platform):
    if platform == "cpu":
        if _TAB_LIB_ANALYTIC is None:
            _check_tab_lib()
            raise RuntimeError("Installed libri_kernels.so has no RFI analytic kernels")
    elif _TAB_LIB_ANALYTIC_GPU is None:
        _check_tab_lib_gpu()
        raise RuntimeError("Installed GPU library has no RFI analytic kernels")


#: Index arrays every primitive takes first: a1, a1_sorter, a1_start, a2,
#: a2_sorter, a2_start, pair_index, tile_pairs.
N_IDX = 8

# The positional layout of the primal arguments after the index arrays.
_ARRAY_NAMES = (
    "amp", "phase", "delay_us", "w_freq", "start_freq", "g_time", "start_time",
    "dnu_mhz", "int_time", "freq_mhz",
)


def _validate(amp, phase, delay, w_freq, start_freq, g_time, start_time, dnu, int_time, freqs):
    suffix = _dtype_suffix(amp.dtype, phase.dtype)
    real = jnp.dtype(phase.dtype)
    for name, x in (("delay_us", delay), ("w_freq", w_freq), ("g_time", g_time),
                    ("dnu_mhz", dnu), ("int_time", int_time), ("freq_mhz", freqs)):
        if jnp.dtype(x.dtype) != real:
            raise TypeError(
                f"RFI analytic kernels require every real input to share the phase "
                f"dtype {real}; got {name}={x.dtype}."
            )
    for name, x in (("start_freq", start_freq), ("start_time", start_time)):
        if jnp.dtype(x.dtype) != jnp.int32:
            raise TypeError(f"RFI analytic kernels require int32 {name}; got {x.dtype}.")
    if len(amp.shape) != 4:
        raise ValueError(f"Expected a rank-4 signal (n_ant, n_rfi, n_freq, n_time); got {amp.shape}")
    n_ant, n_rfi, n_freq, n_time = amp.shape
    expected = {
        "phase": amp.shape,
        "delay_us": (n_ant, n_rfi, n_time, delay.shape[-1] if len(delay.shape) == 4 else -1),
        "w_freq": (n_freq, w_freq.shape[1] if len(w_freq.shape) == 3 else -1, len(dnu.shape) == 1 and dnu.shape[0]),
        "start_freq": (n_freq,),
        "g_time": (n_time, g_time.shape[1] if len(g_time.shape) == 3 else -1, g_time.shape[2] if len(g_time.shape) == 3 else -1),
        "start_time": (n_time,),
        "freq_mhz": (n_freq,),
    }
    got = {"phase": phase.shape, "delay_us": delay.shape, "w_freq": w_freq.shape,
           "start_freq": start_freq.shape, "g_time": g_time.shape,
           "start_time": start_time.shape, "freq_mhz": freqs.shape}
    for name, shape in expected.items():
        if tuple(got[name]) != tuple(shape):
            raise ValueError(f"Expected {name} shape {tuple(shape)}; got {tuple(got[name])}")
    if not (1 <= w_freq.shape[1] <= n_freq and 1 <= g_time.shape[1] <= n_time):
        raise ValueError("A stencil cannot be wider than its axis")
    if delay.shape[-1] < 1:
        raise ValueError("delay_us needs at least the delay itself, (..., 1)")
    if int_time.shape != ():
        raise ValueError("int_time must be a scalar in seconds")
    if len(dnu.shape) != 1 or dnu.shape[0] < 1:
        raise ValueError("dnu_mhz must be a nonempty vector")
    if not 1 <= g_time.shape[2] <= 9:
        raise ValueError("g_time supports 1 through 9 monomial coefficients")
    if min(amp.shape) < 1:
        raise ValueError("The signal axes must be nonempty")
    return suffix


def _validate_like(name, primal, other):
    """Require a tangent or cotangent to match the buffer it is paired with.

    The kernels build views over these buffers from the primal extents, so a
    mismatch reads or writes past the end instead of failing.
    """
    if other.shape != primal.shape or jnp.dtype(other.dtype) != jnp.dtype(primal.dtype):
        raise ValueError(
            f"Expected {name} shape {primal.shape} and dtype {primal.dtype}; "
            f"got {other.shape} and {other.dtype}"
        )


def _output_aval(a1, amp):
    return ShapedArray((a1.shape[0], amp.shape[2], amp.shape[3]), amp.dtype)


def _validate_indices(args):
    amp = args[N_IDX]
    n_ant = amp.shape[0]
    if len(args[0].shape) != 1:
        raise ValueError("Expected a rank-1 baseline index array")
    n_bl = args[0].shape[0]
    n_tiles = (n_ant + TILE - 1) // TILE
    shapes = [(n_bl,), (n_bl,), (n_ant,), (n_bl,), (n_bl,), (n_ant,),
              (n_ant, n_ant), (n_tiles * (n_tiles + 1) // 2, TILE * TILE)]
    for x, shape in zip(args[:N_IDX], shapes):
        if x.shape != shape or jnp.dtype(x.dtype) != jnp.int32:
            raise ValueError(f"Expected int32 index array of shape {shape}; got {x.shape} {x.dtype}")


def _validate_options(*, segments, terms, cubic_terms):
    for name, value, lo, hi in (("segments", segments, 1, 1024),
                               ("terms", terms, 1, 32), ("cubic_terms", cubic_terms, 0, 8)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be a static integer in [{lo}, {hi}]")


def _lowering(prefix, platform):
    def lowering(ctx, *args, **options):
        _check_analytic_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[N_IDX].dtype, ctx.avals_in[N_IDX + 1].dtype)
        target = f"{prefix}{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args, **{k: np.int64(v) for k, v in options.items()})

    return lowering


# --- transpose: the cotangent of the signal from the cotangent of the visibilities

rfi_analytic_transpose_op = core.Primitive("rfi_analytic_transpose_op")
rfi_analytic_transpose_op.def_impl(partial(xla.apply_primitive, rfi_analytic_transpose_op))


def _transpose_abstract(*args, **options):
    _validate_options(**options)
    a1, arrays, g = args[0], args[N_IDX:N_IDX + 10], args[N_IDX + 10]
    _validate(*arrays)
    _validate_indices(args)
    _validate_like("visibility cotangent", _output_aval(a1, arrays[0]), g)
    return ShapedArray(arrays[0].shape, arrays[0].dtype)


rfi_analytic_transpose_op.def_abstract_eval(_transpose_abstract)
mlir.register_lowering(rfi_analytic_transpose_op, _lowering("calc_rfi_analytic_transpose", "cpu"), platform="cpu")
mlir.register_lowering(rfi_analytic_transpose_op, _lowering("calc_rfi_analytic_transpose", "gpu"), platform="gpu")

# --- JVP: linear in the signal tangent -----------------------------------------

rfi_analytic_jvp_op = core.Primitive("rfi_analytic_jvp_op")
rfi_analytic_jvp_op.def_impl(partial(xla.apply_primitive, rfi_analytic_jvp_op))


def _jvp_abstract(*args, **options):
    _validate_options(**options)
    a1, amp, amp_dot, rest = args[0], args[N_IDX], args[N_IDX + 1], args[N_IDX + 2:]
    _validate(amp, *rest)
    _validate_indices(args)
    _validate_like("signal tangent", amp, amp_dot)
    return _output_aval(a1, amp)


rfi_analytic_jvp_op.def_abstract_eval(_jvp_abstract)


def _jvp_lowering(platform):
    def lowering(ctx, *args, **options):
        _check_analytic_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[N_IDX].dtype, ctx.avals_in[N_IDX + 2].dtype)
        target = f"calc_rfi_analytic_jvp{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args, **{k: np.int64(v) for k, v in options.items()})

    return lowering


mlir.register_lowering(rfi_analytic_jvp_op, _jvp_lowering("cpu"), platform="cpu")
mlir.register_lowering(rfi_analytic_jvp_op, _jvp_lowering("gpu"), platform="gpu")


def _jvp_transpose(g, *args, **options):
    indices, amp, rest = args[:N_IDX], args[N_IDX], args[N_IDX + 2:]
    amp_bar = rfi_analytic_transpose_op.bind(*indices, amp, *rest, g, **options)
    # One cotangent, for the linear input amp_dot; every other input is a
    # constant of the linear map.
    return (None,) * N_IDX + (None, amp_bar) + (None,) * len(rest)


ad.primitive_transposes[rfi_analytic_jvp_op] = _jvp_transpose


# --- primal ---------------------------------------------------------------------

rfi_analytic_vis_op = core.Primitive("rfi_analytic_vis_op")
rfi_analytic_vis_op.def_impl(partial(xla.apply_primitive, rfi_analytic_vis_op))


def _vis_abstract(*args, **options):
    _validate_options(**options)
    arrays = args[N_IDX:]
    _validate(*arrays)
    _validate_indices(args)
    return _output_aval(args[0], arrays[0])


rfi_analytic_vis_op.def_abstract_eval(_vis_abstract)
mlir.register_lowering(rfi_analytic_vis_op, _lowering("calc_rfi_analytic", "cpu"), platform="cpu")
mlir.register_lowering(rfi_analytic_vis_op, _lowering("calc_rfi_analytic", "gpu"), platform="gpu")


def _vis_jvp(args, tangents, **options):
    indices, arrays = args[:N_IDX], args[N_IDX:]
    for name, dot in zip(_ARRAY_NAMES[1:], tangents[N_IDX + 1:]):
        if not isinstance(dot, ad.Zero):
            raise TypeError(f"rfi_analytic_vis_op differentiates amp only; got a tangent for {name}")
    primal = rfi_analytic_vis_op.bind(*args, **options)
    dot = tangents[N_IDX]
    if isinstance(dot, ad.Zero):
        return primal, ad.Zero.from_primal_value(primal)
    return primal, rfi_analytic_jvp_op.bind(*indices, arrays[0], dot, *arrays[1:], **options)


ad.primitive_jvps[rfi_analytic_vis_op] = _vis_jvp
