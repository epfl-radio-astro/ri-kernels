"""RFI visibility primitive using compact delays and an explicit frequency grid."""

from functools import partial

import jax
import jax.numpy as jnp
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


class RFIDelayVisOp:
    """Compute RFI visibilities without expanding delays across frequency."""

    def __init__(self, n_ant, a1, a2):
        self.a1 = a1
        self.a2 = a2
        (
            self.a1_sorter,
            self.a1_start,
            self.a2_sorter,
            self.a2_start,
        ) = prepare_indices(n_ant, a1, a2)

    def eval(self, rfi_amp_fine, rfi_delay_us, freq_mhz):
        """Evaluate visibilities using centred delays in μs and frequencies in MHz.

        Args:
            rfi_amp_fine: Complex array shaped
                ``(n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)``.
            rfi_delay_us: Centred geometric delays in μs, shaped
                ``(n_ant, n_time, n_rfi, n_int_time)``.
            freq_mhz: Absolute frequencies in MHz, shaped ``(n_freq,
                n_int_freq)``.

        Since MHz × μs is cycles, the kernel evaluates ``2π freq_mhz ×
        (delay_us_1 - delay_us_2)``. Centre delays across antennas in float64
        before converting them to the kernel dtype. The frequency grid is
        treated as fixed by autodiff; derivatives are provided for amplitudes
        and delays.
        """
        return rfi_delay_vis_op.bind(
            self.a1,
            self.a1_sorter,
            self.a1_start,
            self.a2,
            self.a2_sorter,
            self.a2_start,
            rfi_amp_fine,
            rfi_delay_us,
            freq_mhz,
        )


_TAB_LIB_DELAY = (
    _TAB_LIB if _TAB_LIB and hasattr(_TAB_LIB, "calc_rfi_delay_cpu_f32") else None
)
_TAB_LIB_DELAY_GPU = (
    _TAB_LIB_GPU
    if _TAB_LIB_GPU and hasattr(_TAB_LIB_GPU, "calc_rfi_delay_gpu_f32")
    else None
)

if _TAB_LIB_DELAY:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            _target = f"calc_rfi_delay{_kind}_{_suffix}"
            _symbol = f"calc_rfi_delay{_kind}_cpu_{_suffix}"
            jax.ffi.register_ffi_target(
                _target,
                jax.ffi.pycapsule(getattr(_TAB_LIB_DELAY, _symbol)),
                platform="cpu",
            )

# GPU symbols are registered once GPU implementations are present in an add-on
# library. Keeping the platform lowering here makes CPU and GPU APIs identical.
if _TAB_LIB_DELAY_GPU:
    for _suffix in ("f32", "f64"):
        for _kind in ("", "_jvp", "_transpose"):
            _target = f"calc_rfi_delay{_kind}_gpu_{_suffix}"
            _symbol = f"calc_rfi_delay{_kind}_gpu_{_suffix}"
            jax.ffi.register_ffi_target(
                _target,
                jax.ffi.pycapsule(getattr(_TAB_LIB_DELAY_GPU, _symbol)),
                platform=_TAB_PLATFORM_NAME,
            )


def _check_delay_lib(platform):
    if platform == "cpu":
        if _TAB_LIB_DELAY is None:
            _check_tab_lib()
            raise RuntimeError("Installed libri_kernels.so has no RFI delay kernels")
    elif _TAB_LIB_DELAY_GPU is None:
        _check_tab_lib_gpu()
        raise RuntimeError("Installed GPU library has no RFI delay kernels")


def _validate(amp, delay_us, freq_mhz):
    suffix = _dtype_suffix(amp.dtype, delay_us.dtype)
    if jnp.dtype(freq_mhz.dtype) != jnp.dtype(delay_us.dtype):
        raise TypeError(
            "RFI delay kernels require delay and frequency dtypes to match. "
            f"Got delay={delay_us.dtype}, frequency={freq_mhz.dtype}."
        )
    if len(amp.shape) != 6 or len(delay_us.shape) != 4 or len(freq_mhz.shape) != 2:
        raise ValueError(
            "Expected amplitude, delay, and frequency ranks of 6, 4, and 2"
        )
    expected_compact = (amp.shape[0], amp.shape[2], amp.shape[3], amp.shape[5])
    expected_freq = (amp.shape[1], amp.shape[4])
    if delay_us.shape != expected_compact or freq_mhz.shape != expected_freq:
        raise ValueError(
            f"Expected delay shape {expected_compact} and frequency shape "
            f"{expected_freq}; got {delay_us.shape} and {freq_mhz.shape}"
        )
    return suffix


def _output_aval(a1, amp):
    return ShapedArray((a1.shape[0], amp.shape[1], amp.shape[2]), amp.dtype)


rfi_delay_transpose_op = core.Primitive("rfi_delay_transpose_op")
rfi_delay_transpose_op.def_impl(partial(xla.apply_primitive, rfi_delay_transpose_op))
rfi_delay_transpose_op.multiple_results = True


def _transpose_abstract(a1, a1s, a1b, a2, a2s, a2b, amp, delay, freq, g):
    _validate(amp, delay, freq)
    return ShapedArray(amp.shape, amp.dtype), ShapedArray(delay.shape, delay.dtype)


rfi_delay_transpose_op.def_abstract_eval(_transpose_abstract)


def _transpose_lowering(platform):
    def lowering(ctx, *args):
        _check_delay_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
        target = f"calc_rfi_delay_transpose{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args)

    return lowering


mlir.register_lowering(rfi_delay_transpose_op, _transpose_lowering("cpu"), platform="cpu")
mlir.register_lowering(rfi_delay_transpose_op, _transpose_lowering("gpu"), platform="gpu")


rfi_delay_jvp_op = core.Primitive("rfi_delay_jvp_op")
rfi_delay_jvp_op.def_impl(partial(xla.apply_primitive, rfi_delay_jvp_op))


def _jvp_abstract(
    a1, a1s, a1b, a2, a2s, a2b, amp, amp_dot, delay, delay_dot, freq
):
    _validate(amp, delay, freq)
    return _output_aval(a1, amp)


rfi_delay_jvp_op.def_abstract_eval(_jvp_abstract)


def _jvp_lowering(platform):
    def lowering(ctx, *args):
        _check_delay_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[8].dtype)
        target = f"calc_rfi_delay_jvp{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args)

    return lowering


mlir.register_lowering(rfi_delay_jvp_op, _jvp_lowering("cpu"), platform="cpu")
mlir.register_lowering(rfi_delay_jvp_op, _jvp_lowering("gpu"), platform="gpu")


def _jvp_transpose(
    g, a1, a1s, a1b, a2, a2s, a2b, amp, amp_dot, delay, delay_dot, freq
):
    amp_bar, delay_bar = rfi_delay_transpose_op.bind(
        a1, a1s, a1b, a2, a2s, a2b, amp, delay, freq, g
    )
    return (None, None, None, None, None, None, amp_bar, amp_bar,
            delay_bar, delay_bar, None)


ad.primitive_transposes[rfi_delay_jvp_op] = _jvp_transpose


rfi_delay_vis_op = core.Primitive("rfi_delay_vis_op")
rfi_delay_vis_op.def_impl(partial(xla.apply_primitive, rfi_delay_vis_op))


def _vis_abstract(a1, a1s, a1b, a2, a2s, a2b, amp, delay, freq):
    _validate(amp, delay, freq)
    return _output_aval(a1, amp)


rfi_delay_vis_op.def_abstract_eval(_vis_abstract)


def _vis_lowering(platform):
    def lowering(ctx, *args):
        _check_delay_lib(platform)
        suffix = _dtype_suffix(ctx.avals_in[6].dtype, ctx.avals_in[7].dtype)
        target = f"calc_rfi_delay{'_gpu' if platform == 'gpu' else ''}_{suffix}"
        return jax.ffi.ffi_lowering(target)(ctx, *args)

    return lowering


mlir.register_lowering(rfi_delay_vis_op, _vis_lowering("cpu"), platform="cpu")
mlir.register_lowering(rfi_delay_vis_op, _vis_lowering("gpu"), platform="gpu")


def _vis_jvp(args, tangents):
    *indices, amp, delay, freq = args
    *_, amp_dot, delay_dot, _freq_dot = tangents
    if isinstance(amp_dot, ad.Zero):
        amp_dot = jnp.zeros_like(amp)
    if isinstance(delay_dot, ad.Zero):
        delay_dot = jnp.zeros_like(delay)
    tangent = rfi_delay_jvp_op.bind(
        *indices, amp, amp_dot, delay, delay_dot, freq
    )
    return rfi_delay_vis_op.bind(*args), tangent


ad.primitive_jvps[rfi_delay_vis_op] = _vis_jvp
