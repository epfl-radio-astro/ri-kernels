"""RFI visibility primitive from the data grid: the signal, phase and delay
polynomial per data cell, the fine samples rebuilt inside the kernel.

The operator computes, for one cell ``(f, t)``, fine sample ``(u, v)``,
source ``r`` and antenna ``a``::

    A[a]    = sum_k sum_l w_freq[f, k, u] w_time[t, l, v]
                          amp[a, r, start_freq[f] + k, start_time[t] + l]
    dtau[a] = sum_{k >= 1} delay_us[a, r, t, k] dt[v]^k / k!
    phi[a]  = phase[a, r, f, t]
              + 2 pi ((freq_mhz[f] + dnu_mhz[u]) dtau[a] + dnu_mhz[u] delay_us[a, r, t, 0])
    S[a]   = A[a] exp(i phi[a])
    vis[bl, f, t] = mean_{u, v} sum_r S[a1[bl]] conj(S[a2[bl]])

Only the signal ``amp`` is differentiated; the phase, the delay and the tables
are constants of the run, and :meth:`RFIInterpVisOp.eval` stops their
gradients explicitly. The weights are data: the polynomial through the
stencil, the conditional mean of a Gaussian process, or any other linear
interpolant is a different table through the same kernel.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, mlir, xla

from .rfi_vis_op import (
    _TAB_LIB,
    _TAB_LIB_GPU,
    _TAB_PLATFORM_NAME,
    _check_tab_lib,
    _check_tab_lib_gpu,
    _dtype_suffix,
    prepare_indices,
)


class RFIInterpVisOp:
    """Compute RFI visibilities from the data grid."""

    def __init__(self, n_ant, a1, a2):
        self.a1 = a1
        self.a2 = a2
        (
            self.a1_sorter,
            self.a1_start,
            self.a2_sorter,
            self.a2_start,
        ) = prepare_indices(n_ant, a1, a2)
        # The baseline index of each antenna pair, -1 where the list has none.
        # The staged GPU kernels form every pair of two antenna tiles once and
        # write it to whichever orderings the list holds, so a pair must not
        # appear twice: the second copy would never be written.
        pairs = np.stack([np.asarray(a1), np.asarray(a2)], axis=1)
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ValueError("RFIInterpVisOp needs each (a1, a2) baseline at most once")
        self.pair_index = (
            jnp.full((n_ant, n_ant), -1, dtype=jnp.int32)
            .at[jnp.asarray(a1), jnp.asarray(a2)]
            .set(jnp.arange(len(a1), dtype=jnp.int32))
        )

    @property
    def indices(self):
        """The index arrays every primitive takes before the data arrays."""
        return (
            self.a1, self.a1_sorter, self.a1_start,
            self.a2, self.a2_sorter, self.a2_start, self.pair_index,
        )

    def eval(self, amp, phase, delay_us, w_freq, start_freq, w_time, start_time, dnu_mhz, dt, freq_mhz):
        """Evaluate the visibilities, ``(n_bl, n_freq, n_time)``.

        Args:
            amp: Complex ``(n_ant, n_rfi, n_freq, n_time)``, the signal on the
                data grid. The only differentiated input.
            phase: Real ``(n_ant, n_rfi, n_freq, n_time)``, the phase at the
                channel and cell centre, reduced to one turn. Reduce it in
                float64 before casting: the unreduced phase is ~1e6 turns and
                the kernel never rebuilds it from ``delay_us``.
            delay_us: Real ``(n_ant, n_rfi, n_time, n_path)``, the geometric
                delay in microseconds and its time derivatives (us/s^k) at the
                cell centre, with the sign that makes the phase ``2 pi f tau``,
                relative to the array mean: a term common to every antenna
                cancels in a baseline's phase difference, and the change of the
                full delay across a cell is ~1e4 wavelengths at orbital range
                rates, which float32 cannot hold to a fraction of a turn.
            w_freq, start_freq: Real ``(n_freq, n_sf, n_int_freq)`` and int32
                ``(n_freq,)``, the interpolation weights across each channel
                and the first channel of each channel's stencil.
            w_time, start_time: The same across each cell, ``(n_time, n_st,
                n_int_time)`` and ``(n_time,)``.
            dnu_mhz, dt: Real ``(n_int_freq,)`` and ``(n_int_time,)``, the fine
                offsets from the channel centre (MHz) and the cell centre (s).
            freq_mhz: Real ``(n_freq,)``, the channel centres (MHz). MHz times
                microseconds is cycles, so no scaling constant enters.

        Each cell's stencil must lie inside its axis and contain the cell:
        ``0 <= start[c] <= c < start[c] + n_stencil <= n_cells``. Precision has
        to match: complex64 with float32 or complex128 with float64.
        """
        stop = jax.lax.stop_gradient
        return rfi_interp_vis_op.bind(
            *self.indices,
            amp, stop(phase), stop(delay_us), stop(w_freq), start_freq,
            stop(w_time), start_time, stop(dnu_mhz), stop(dt), stop(freq_mhz),
        )


_TAB_LIB_INTERP = (
    _TAB_LIB if _TAB_LIB and hasattr(_TAB_LIB, "calc_rfi_interp_cpu_f32") else None
)
_TAB_LIB_INTERP_GPU = (
    _TAB_LIB_GPU
    if _TAB_LIB_GPU and hasattr(_TAB_LIB_GPU, "calc_rfi_interp_gpu_f32")
    else None
)

if _TAB_LIB_INTERP:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            jax.ffi.register_ffi_target(
                f"calc_rfi_interp{_kind}_{_suffix}",
                jax.ffi.pycapsule(
                    getattr(_TAB_LIB_INTERP, f"calc_rfi_interp{_kind}_cpu_{_suffix}")
                ),
                platform="cpu",
            )

if _TAB_LIB_INTERP_GPU:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            jax.ffi.register_ffi_target(
                f"calc_rfi_interp{_kind}_gpu_{_suffix}",
                jax.ffi.pycapsule(
                    getattr(_TAB_LIB_INTERP_GPU, f"calc_rfi_interp{_kind}_gpu_{_suffix}")
                ),
                platform=_TAB_PLATFORM_NAME,
            )


def _check_interp_lib(platform):
    if platform == "cpu":
        if _TAB_LIB_INTERP is None:
            _check_tab_lib()
            raise RuntimeError("Installed libri_kernels.so has no RFI interp kernels")
    elif _TAB_LIB_INTERP_GPU is None:
        _check_tab_lib_gpu()
        raise RuntimeError("Installed GPU library has no RFI interp kernels")


#: Index arrays every primitive takes first: a1, a1_sorter, a1_start, a2,
#: a2_sorter, a2_start, pair_index.
N_IDX = 7

# The positional layout of the primal arguments after the index arrays.
_ARRAY_NAMES = (
    "amp", "phase", "delay_us", "w_freq", "start_freq", "w_time", "start_time",
    "dnu_mhz", "dt", "freq_mhz",
)


def _validate(amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs):
    suffix = _dtype_suffix(amp.dtype, phase.dtype)
    real = jnp.dtype(phase.dtype)
    for name, x in (("delay_us", delay), ("w_freq", w_freq), ("w_time", w_time),
                    ("dnu_mhz", dnu), ("dt", dt), ("freq_mhz", freqs)):
        if jnp.dtype(x.dtype) != real:
            raise TypeError(
                f"RFI interp kernels require every real input to share the phase "
                f"dtype {real}; got {name}={x.dtype}."
            )
    for name, x in (("start_freq", start_freq), ("start_time", start_time)):
        if jnp.dtype(x.dtype) != jnp.int32:
            raise TypeError(f"RFI interp kernels require int32 {name}; got {x.dtype}.")
    if len(amp.shape) != 4:
        raise ValueError(f"Expected a rank-4 signal (n_ant, n_rfi, n_freq, n_time); got {amp.shape}")
    n_ant, n_rfi, n_freq, n_time = amp.shape
    expected = {
        "phase": amp.shape,
        "delay_us": (n_ant, n_rfi, n_time, delay.shape[-1] if len(delay.shape) == 4 else -1),
        "w_freq": (n_freq, w_freq.shape[1] if len(w_freq.shape) == 3 else -1, len(dnu.shape) == 1 and dnu.shape[0]),
        "start_freq": (n_freq,),
        "w_time": (n_time, w_time.shape[1] if len(w_time.shape) == 3 else -1, len(dt.shape) == 1 and dt.shape[0]),
        "start_time": (n_time,),
        "freq_mhz": (n_freq,),
    }
    got = {"phase": phase.shape, "delay_us": delay.shape, "w_freq": w_freq.shape,
           "start_freq": start_freq.shape, "w_time": w_time.shape,
           "start_time": start_time.shape, "freq_mhz": freqs.shape}
    for name, shape in expected.items():
        if tuple(got[name]) != tuple(shape):
            raise ValueError(f"Expected {name} shape {tuple(shape)}; got {tuple(got[name])}")
    if not (1 <= w_freq.shape[1] <= n_freq and 1 <= w_time.shape[1] <= n_time):
        raise ValueError("A stencil cannot be wider than its axis")
    if delay.shape[-1] < 1:
        raise ValueError("delay_us needs at least the delay itself, (..., 1)")
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


def _lowering(prefix, platform):
    def lowering(ctx, *args):
        _check_interp_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[N_IDX].dtype, ctx.avals_in[N_IDX + 1].dtype)
        target = f"{prefix}{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args)

    return lowering


# --- transpose: the cotangent of the signal from the cotangent of the visibilities

rfi_interp_transpose_op = core.Primitive("rfi_interp_transpose_op")
rfi_interp_transpose_op.def_impl(partial(xla.apply_primitive, rfi_interp_transpose_op))


def _transpose_abstract(*args):
    a1, arrays, g = args[0], args[N_IDX:N_IDX + 10], args[N_IDX + 10]
    _validate(*arrays)
    _validate_like("visibility cotangent", _output_aval(a1, arrays[0]), g)
    return ShapedArray(arrays[0].shape, arrays[0].dtype)


rfi_interp_transpose_op.def_abstract_eval(_transpose_abstract)
mlir.register_lowering(rfi_interp_transpose_op, _lowering("calc_rfi_interp_transpose", "cpu"), platform="cpu")
mlir.register_lowering(rfi_interp_transpose_op, _lowering("calc_rfi_interp_transpose", "gpu"), platform="gpu")


# --- JVP: linear in the signal tangent -----------------------------------------

rfi_interp_jvp_op = core.Primitive("rfi_interp_jvp_op")
rfi_interp_jvp_op.def_impl(partial(xla.apply_primitive, rfi_interp_jvp_op))


def _jvp_abstract(*args):
    a1, amp, amp_dot, rest = args[0], args[N_IDX], args[N_IDX + 1], args[N_IDX + 2:]
    _validate(amp, *rest)
    _validate_like("signal tangent", amp, amp_dot)
    return _output_aval(a1, amp)


rfi_interp_jvp_op.def_abstract_eval(_jvp_abstract)


def _jvp_lowering(platform):
    def lowering(ctx, *args):
        _check_interp_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[N_IDX].dtype, ctx.avals_in[N_IDX + 2].dtype)
        target = f"calc_rfi_interp_jvp{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args)

    return lowering


mlir.register_lowering(rfi_interp_jvp_op, _jvp_lowering("cpu"), platform="cpu")
mlir.register_lowering(rfi_interp_jvp_op, _jvp_lowering("gpu"), platform="gpu")


def _jvp_transpose(g, *args):
    indices, amp, rest = args[:N_IDX], args[N_IDX], args[N_IDX + 2:]
    amp_bar = rfi_interp_transpose_op.bind(*indices, amp, *rest, g)
    # One cotangent, for the linear input amp_dot; every other input is a
    # constant of the linear map.
    return (None,) * N_IDX + (None, amp_bar) + (None,) * len(rest)


ad.primitive_transposes[rfi_interp_jvp_op] = _jvp_transpose


# --- primal ---------------------------------------------------------------------

rfi_interp_vis_op = core.Primitive("rfi_interp_vis_op")
rfi_interp_vis_op.def_impl(partial(xla.apply_primitive, rfi_interp_vis_op))


def _vis_abstract(*args):
    arrays = args[N_IDX:]
    _validate(*arrays)
    return _output_aval(args[0], arrays[0])


rfi_interp_vis_op.def_abstract_eval(_vis_abstract)
mlir.register_lowering(rfi_interp_vis_op, _lowering("calc_rfi_interp", "cpu"), platform="cpu")
mlir.register_lowering(rfi_interp_vis_op, _lowering("calc_rfi_interp", "gpu"), platform="gpu")


def _vis_jvp(args, tangents):
    indices, arrays = args[:N_IDX], args[N_IDX:]
    amp, rest = arrays[0], arrays[1:]
    amp_dot, rest_dots = tangents[N_IDX], tangents[N_IDX + 1:]
    # eval() stops the gradient on every input but the signal, so a non-zero
    # tangent can only reach here through a direct bind. The kernel has no
    # derivative with respect to them, so refuse rather than drop it.
    for name, dot in zip(_ARRAY_NAMES[1:], rest_dots):
        if not isinstance(dot, ad.Zero):
            raise TypeError(
                f"rfi_interp_vis_op differentiates the signal only and provides "
                f"no derivative with respect to {name}. Use RFIInterpVisOp.eval, "
                "which applies lax.stop_gradient to the other inputs."
            )
    if isinstance(amp_dot, ad.Zero):
        amp_dot = jnp.zeros_like(amp)
    tangent = rfi_interp_jvp_op.bind(*indices, amp, amp_dot, *rest)
    return rfi_interp_vis_op.bind(*args), tangent


ad.primitive_jvps[rfi_interp_vis_op] = _vis_jvp
