"""Tests for :class:`ri_kernels.jax_api.RFIVisOp`.

The FFI kernels are checked against :class:`RFIVisOpRef`, the plain JAX
implementation of the same computation: direct evaluation, the JVP (forward
mode) and the VJP (reverse mode, i.e. the transpose kernel). JAX
differentiates the reference automatically, so the same call produces both
the expected value and the expected derivatives.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import (
    BASELINE_LAYOUTS,
    DEFAULT_SHAPE,
    PRECISIONS,
    SHAPES,
    assert_close,
    make_baselines,
    make_cotangent,
    make_signal,
)
from ri_kernels.jax_api import rfi_vis_op as rfi_vis_op_module
from ri_kernels.jax_api.rfi_vis_op import RFIVisOp, prepare_indices
from ri_kernels.jax_api.rfi_vis_op_ref import RFIVisOpRef


def _require_cpu():
    return os.environ.get("RI_KERNELS_REQUIRE_CPU", "").lower() in ("1", "true", "yes", "on")


def _available_devices():
    """One device per platform for which the matching FFI library was found."""
    devices = []
    if rfi_vis_op_module._TAB_LIB is not None:
        devices.extend(jax.devices("cpu")[:1])
    elif _require_cpu():
        pytest.fail("RI_KERNELS_REQUIRE_CPU is set, but libri_kernels.so was not loaded")
    if rfi_vis_op_module._TAB_LIB_CUDA is not None:
        try:
            devices.extend(jax.devices("cuda")[:1])
        except RuntimeError:
            pass
    if rfi_vis_op_module._TAB_LIB_ROCM is not None:
        try:
            devices.extend(jax.devices("rocm")[:1])
        except RuntimeError:
            pass
    return devices


DEVICES = _available_devices()

if not DEVICES:
    pytest.skip(
        "no ri_kernels FFI library found - build/install the package first",
        allow_module_level=True,
    )


@pytest.fixture(params=DEVICES, ids=lambda device: device.platform)
def device(request):
    """Run the test body with the given device as the default one."""
    with jax.default_device(request.param):
        yield request.param


@pytest.fixture(params=PRECISIONS, ids=["f32", "f64"])
def precision(request):
    if request.param.real == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("x64 is disabled, cannot exercise the f64 kernels")
    return request.param


# --- Direct evaluation -------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_eval_matches_reference(device, precision, shape):
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    vis = op.eval(amp, phase)

    assert vis.shape == (len(a1), shape.n_freq, shape.n_time)
    assert vis.dtype == precision.complex
    assert_close(vis, ref.eval(amp, phase), precision)


@BASELINE_LAYOUTS
def test_eval_baseline_layouts(device, precision, auto_corr, shuffle):
    """Auto-correlations and an unsorted baseline order must be handled."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    assert_close(op.eval(amp, phase), ref.eval(amp, phase), precision)


def test_eval_under_jit(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    assert_close(jax.jit(op.eval)(amp, phase), op.eval(amp, phase), precision)


def test_traced_constructor_under_jit(device, precision):
    """RFIVisOp.__init__ must remain compatible with traced a1/a2 inside jit."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    @jax.jit
    def run(idx1, idx2, a, p):
        return RFIVisOp(shape.n_ant, idx1, idx2).eval(a, p)

    assert_close(run(a1, a2, amp, phase), ref.eval(amp, phase), precision)


def test_int64_baseline_indices(device):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, PRECISIONS[0])
    op = RFIVisOp(shape.n_ant, a1.astype(jnp.int64), a2.astype(jnp.int64))
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    assert op.a1.dtype == jnp.int32
    assert op.a2.dtype == jnp.int32
    assert_close(op.eval(amp, phase), ref.eval(amp, phase), PRECISIONS[0])


def test_empty_baselines(device, precision):
    shape = DEFAULT_SHAPE
    empty = jnp.array([], jnp.int32)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, empty, empty)

    vis, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    assert vis.shape == (0, shape.n_freq, shape.n_time)
    assert vis_dot.shape == vis.shape

    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(jnp.zeros_like(vis))
    np.testing.assert_array_equal(amp_bar, jnp.zeros_like(amp))
    np.testing.assert_array_equal(phase_bar, jnp.zeros_like(phase))


def test_indices_match_prepare_indices(device):
    """The operator stores the same indices as the standalone helper."""
    n_ant = DEFAULT_SHAPE.n_ant
    a1, a2 = make_baselines(n_ant, shuffle=True)
    op = RFIVisOp(n_ant, a1, a2)

    a1_sorter, a1_start, a2_sorter, a2_start = prepare_indices(n_ant, a1, a2)

    np.testing.assert_array_equal(op.a1_sorter, a1_sorter)
    np.testing.assert_array_equal(op.a1_start, a1_start)
    np.testing.assert_array_equal(op.a2_sorter, a2_sorter)
    np.testing.assert_array_equal(op.a2_start, a2_start)


def test_mixed_precision_is_rejected(device):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, _ = make_signal(shape, PRECISIONS[0])
    _, phase = make_signal(shape, PRECISIONS[1])
    op = RFIVisOp(shape.n_ant, a1, a2)

    with pytest.raises(TypeError, match="matched precision"):
        op.eval(amp, phase)


def test_invalid_inputs_rejected(device):
    a1, a2 = make_baselines(5)
    with pytest.raises(ValueError, match="antenna-index"):
        RFIVisOp(0, a1, a2)
    with pytest.raises(ValueError, match="antenna-index"):
        RFIVisOp(5, a1[:-1], a2)
    with pytest.raises(ValueError, match="antenna-index"):
        RFIVisOp(5, a1.astype(jnp.float32), a2)

    op = RFIVisOp(5, a1, a2)
    with pytest.raises(ValueError, match="rfi_amp_fine of shape"):
        op.eval(jnp.zeros((5, 3, 2, 3, 2), jnp.complex64), jnp.zeros((5, 3, 2, 3, 2), jnp.float32))
    with pytest.raises(ValueError, match="rfi_phase of shape"):
        op.eval(
            jnp.zeros((5, 3, 2, 3, 2, 3), jnp.complex64),
            jnp.zeros((5, 3, 2, 3, 2, 4), jnp.float32),
        )


@pytest.mark.parametrize("phase_scale", [1e2, 1e4, 1.7e4], ids=["1e2", "1e4", "1.7e4"])
def test_large_phase_accuracy_against_f64_reference(device, precision, phase_scale):
    """Compare large phase differences (up to ~3.4e4 rad, within Highway's
    documented [-39000, +39000] range) against an upcast float64 reference.

    Isolates trig range-reduction accuracy from float32 phase subtraction
    rounding by evaluating the float64 reference on the exact phase differences
    the kernel sees.
    """
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    rng = np.random.default_rng(7)
    dims = tuple(shape)
    amp_np = rng.normal(size=dims) + 1.0j * rng.normal(size=dims)
    phase_np = rng.uniform(-phase_scale, phase_scale, size=dims)
    amp = jnp.asarray(amp_np, dtype=precision.complex)
    phase = jnp.asarray(phase_np, dtype=precision.real)
    amp_dot, phase_dot = make_signal(shape, precision, seed=8)
    cotangent = make_cotangent(shape, len(a1), precision, seed=9)

    op = RFIVisOp(shape.n_ant, a1, a2)
    vis, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(cotangent)

    # Upcast float64 reference evaluated on the exact phase differences in the
    # kernel's precision so float32 subtraction rounding does not mask trig error.
    a1_np, a2_np = np.asarray(a1), np.asarray(a2)
    amp64 = np.asarray(amp, np.complex128)
    amp_dot64 = np.asarray(amp_dot, np.complex128)
    phase_dot64 = np.asarray(phase_dot, np.float64)
    diff_in_prec = np.asarray(phase[a1] - phase[a2], np.float64)
    exp_d = np.exp(1j * diff_in_prec)
    prod = amp64[a1_np] * amp64[a2_np].conj()
    ref_vis = (prod * exp_d).sum(axis=3).mean(axis=(3, 4)).astype(precision.complex)
    ref_jvp = (
        (
            exp_d * (
                amp_dot64[a1_np] * amp64[a2_np].conj()
                + amp64[a1_np] * amp_dot64[a2_np].conj()
            )
            + 1j * exp_d * (phase_dot64[a1_np] - phase_dot64[a2_np]) * prod
        )
        .sum(axis=3)
        .mean(axis=(3, 4))
        .astype(precision.complex)
    )

    assert_close(vis, ref_vis, precision, f"fwd at scale={phase_scale}")
    assert_close(vis_dot, ref_jvp, precision, f"jvp at scale={phase_scale}")

    # Dot-product check verifies the transpose matches JVP at large phase too.
    forward = float(jnp.sum(cotangent * vis_dot).real)
    backward = float((jnp.sum(amp_bar * amp_dot) + jnp.sum(phase_bar * phase_dot)).real)
    scale = float(jnp.sum(jnp.abs(cotangent) * jnp.abs(vis_dot)))
    np.testing.assert_allclose(forward, backward, rtol=0.0, atol=precision.rtol * scale)


# --- Forward mode (JVP) ------------------------------------------------------


def test_jvp_matches_reference(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    vis, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    ref_vis, ref_vis_dot = jax.jvp(ref.eval, (amp, phase), (amp_dot, phase_dot))

    assert_close(vis, ref_vis, precision, "primal")
    assert_close(vis_dot, ref_vis_dot, precision, "tangent")


@pytest.mark.parametrize("wrt", ["amp", "phase"])
def test_jvp_single_argument(device, precision, wrt):
    """A missing tangent on one input takes the ``ad.Zero`` path of the rule."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    if wrt == "amp":
        fun = lambda a: op.eval(a, phase)
        ref_fun = lambda a: ref.eval(a, phase)
        primal, tangent = amp, amp_dot
    else:
        fun = lambda p: op.eval(amp, p)
        ref_fun = lambda p: ref.eval(amp, p)
        primal, tangent = phase, phase_dot

    vis, vis_dot = jax.jvp(fun, (primal,), (tangent,))
    ref_vis, ref_vis_dot = jax.jvp(ref_fun, (primal,), (tangent,))

    assert_close(vis, ref_vis, precision, "primal")
    assert_close(vis_dot, ref_vis_dot, precision, "tangent")


@BASELINE_LAYOUTS
def test_jvp_baseline_layouts(device, precision, auto_corr, shuffle):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    _, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    _, ref_vis_dot = jax.jvp(ref.eval, (amp, phase), (amp_dot, phase_dot))

    assert_close(vis_dot, ref_vis_dot, precision, "tangent")


def test_jvp_of_zero_tangent_is_zero(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    zeros = (jnp.zeros_like(amp), jnp.zeros_like(phase))
    _, vis_dot = jax.jvp(op.eval, (amp, phase), zeros)

    assert_close(vis_dot, jnp.zeros_like(vis_dot), precision)


def test_jvp_under_jit(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, a1, a2)

    def jvp(a, p, da, dp):
        return jax.jvp(op.eval, (a, p), (da, dp))

    vis, vis_dot = jvp(amp, phase, amp_dot, phase_dot)
    jit_vis, jit_vis_dot = jax.jit(jvp)(amp, phase, amp_dot, phase_dot)

    assert_close(jit_vis, vis, precision, "primal")
    assert_close(jit_vis_dot, vis_dot, precision, "tangent")


# --- Reverse mode (VJP) ------------------------------------------------------


def test_vjp_matches_reference(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)
    cotangent = make_cotangent(shape, len(a1), precision, seed=2)

    vis, vjp_fun = jax.vjp(op.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(cotangent)

    ref_vis, ref_vjp_fun = jax.vjp(ref.eval, amp, phase)
    ref_amp_bar, ref_phase_bar = ref_vjp_fun(cotangent)

    assert_close(vis, ref_vis, precision, "primal")
    assert_close(amp_bar, ref_amp_bar, precision, "rfi_amp_fine cotangent")
    assert_close(phase_bar, ref_phase_bar, precision, "rfi_phase cotangent")


@pytest.mark.parametrize("argnum", [0, 1], ids=["amp", "phase"])
def test_grad_single_argument(device, precision, argnum):
    """Reverse-mode differentiation with respect to a single input."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    def loss(fun):
        return lambda a, p: jnp.sum(jnp.abs(fun(a, p)) ** 2)

    grad_op = jax.grad(loss(op.eval), argnums=argnum)
    grad_ref = jax.grad(loss(ref.eval), argnums=argnum)
    expected = grad_ref(amp, phase)

    assert_close(grad_op(amp, phase), expected, precision, "eager")
    assert_close(jax.jit(grad_op)(amp, phase), expected, precision, "jit")


@BASELINE_LAYOUTS
def test_vjp_baseline_layouts(device, precision, auto_corr, shuffle):
    """The transpose kernel walks the baselines through the sorted indices."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)
    cotangent = make_cotangent(shape, len(a1), precision, seed=2)

    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    _, ref_vjp_fun = jax.vjp(ref.eval, amp, phase)

    amp_bar, phase_bar = vjp_fun(cotangent)
    ref_amp_bar, ref_phase_bar = ref_vjp_fun(cotangent)

    assert_close(amp_bar, ref_amp_bar, precision, "rfi_amp_fine cotangent")
    assert_close(phase_bar, ref_phase_bar, precision, "rfi_phase cotangent")


def test_vjp_of_zero_cotangent_is_zero(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    zero_ct = jnp.zeros(
        (len(a1), shape.n_freq, shape.n_time), dtype=precision.complex
    )
    amp_bar, phase_bar = vjp_fun(zero_ct)

    assert_close(amp_bar, jnp.zeros_like(amp), precision)
    assert_close(phase_bar, jnp.zeros_like(phase), precision)


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_grad_of_real_loss(device, precision, shape):
    """Gradient of a real scalar loss, the way the op is used in practice."""
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    def loss(fun):
        return lambda a, p: jnp.sum(jnp.abs(fun(a, p)) ** 2)

    amp_bar, phase_bar = jax.grad(loss(op.eval), argnums=(0, 1))(amp, phase)
    ref_amp_bar, ref_phase_bar = jax.grad(loss(ref.eval), argnums=(0, 1))(amp, phase)

    assert_close(amp_bar, ref_amp_bar, precision, "d(loss)/d(rfi_amp_fine)")
    assert_close(phase_bar, ref_phase_bar, precision, "d(loss)/d(rfi_phase)")


def test_grad_under_jit(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    def loss(a, p):
        return jnp.sum(jnp.abs(op.eval(a, p)) ** 2)

    grad_fun = jax.grad(loss, argnums=(0, 1))
    amp_bar, phase_bar = grad_fun(amp, phase)
    jit_amp_bar, jit_phase_bar = jax.jit(grad_fun)(amp, phase)

    assert_close(jit_amp_bar, amp_bar, precision, "d(loss)/d(rfi_amp_fine)")
    assert_close(jit_phase_bar, phase_bar, precision, "d(loss)/d(rfi_phase)")


# --- Dot-product test --------------------------------------------------------


def test_jvp_vjp_dot_product(device, precision):
    """``Re<w, J v> == Re<J^T w, v>``: the transpose kernel is the transpose."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=3)
    op = RFIVisOp(shape.n_ant, a1, a2)
    cotangent = make_cotangent(shape, len(a1), precision, seed=4)

    _, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(cotangent)

    forward = jnp.sum(cotangent * vis_dot).real
    backward = (jnp.sum(amp_bar * amp_dot) + jnp.sum(phase_bar * phase_dot)).real

    scale = float(jnp.sum(jnp.abs(cotangent) * jnp.abs(vis_dot)))
    np.testing.assert_allclose(
        float(forward), float(backward), rtol=0.0, atol=precision.rtol * scale
    )


# --- Concurrency & import isolation ------------------------------------------


def test_multi_device_cpu_pmap(precision):
    """Exercise async parallel_for Futures across multiple CPU devices."""
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 2 or rfi_vis_op_module._TAB_LIB is None:
        pytest.skip("requires >= 2 CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    n_dev = len(cpu_devices)
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    op = RFIVisOp(shape.n_ant, a1, a2)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)
    amps, phases = zip(*(make_signal(shape, precision, seed=i) for i in range(n_dev)))
    amps = jnp.stack(amps)
    phases = jnp.stack(phases)

    def loss_and_grad(a, p):
        return jax.value_and_grad(
            lambda x, y: jnp.sum(jnp.abs(op.eval(x, y)) ** 2), argnums=(0, 1)
        )(a, p)

    vals, (amp_bars, phase_bars) = jax.pmap(loss_and_grad)(amps, phases)
    for i in range(n_dev):
        ref_val, (ref_ab, ref_pb) = jax.value_and_grad(
            lambda x, y: jnp.sum(jnp.abs(ref.eval(x, y)) ** 2), argnums=(0, 1)
        )(amps[i], phases[i])
        assert_close(vals[i], ref_val, precision)
        assert_close(amp_bars[i], ref_ab, precision)
        assert_close(phase_bars[i], ref_pb, precision)


def test_reference_import_does_not_load_ffi():
    code = (
        "import sys; "
        "from ri_kernels.jax_api import RFIVisOpRef, RFIAnalyticVisOpRef; "
        "assert 'ri_kernels.jax_api.rfi_vis_op' not in sys.modules; "
        "assert 'ri_kernels.jax_api._lib' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_submodules_reachable_as_attributes():
    """``api.rfi_vis_op`` works after a bare import, as with the eager __init__."""
    code = (
        "import ri_kernels.jax_api as api; "
        "assert api.rfi_vis_op.RFIVisOp is api.RFIVisOp; "
        "assert api.rfi_analytic_vis_op_ref.RFIAnalyticVisOpRef is api.RFIAnalyticVisOpRef; "
        "assert 'rfi_vis_op' in dir(api)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_compiled_import_does_not_initialize_jax_backends():
    code = "\n".join([
        "import sys",
        "from ri_kernels.jax_api import RFIVisOp, RFIAnalyticVisOp",
        "from jax._src import xla_bridge",
        "if not hasattr(xla_bridge, 'backends_are_initialized'):",
        "    sys.exit(42)",
        "assert not xla_bridge.backends_are_initialized(), 'JAX backends initialized on import'",
    ])
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode == 42:
        pytest.skip("jax._src.xla_bridge.backends_are_initialized not available")
    assert proc.returncode == 0, proc.stderr
