"""Tests for :class:`ri_kernels.rfi_vis_op.RFIVisOp`.

The FFI kernels are checked against a plain JAX implementation of the same
computation (``ref_rfi_vis_kernel``): direct evaluation, the JVP (forward mode)
and the VJP (reverse mode, i.e. the transpose kernel). JAX differentiates the
reference automatically, so the same call produces both the expected value and
the expected derivatives.
"""

from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ri_kernels.jax_api import rfi_vis_op as rfi_vis_op_module
from ri_kernels.jax_api.rfi_vis_op import RFIVisOp, prepare_indices


def ref_rfi_vis_kernel(rfi_amp_fine, rfi_phase, a1, a2):
    """Reference RFI visibility computation in pure JAX.

    ``rfi_amp_fine`` / ``rfi_phase`` have shape
    ``(n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)``. The RFI sources
    are summed coherently and the sub-integration samples are averaged, giving
    ``(n_bl, n_freq, n_time)``.
    """
    vis_rfi_fine = jnp.sum(
        rfi_amp_fine[a1]
        * jnp.conjugate(rfi_amp_fine[a2])
        * jnp.exp(1.0j * (rfi_phase[a1] - rfi_phase[a2])),
        axis=3,
    )
    # (n_bl, n_freq, n_time, n_int_freq, n_int_time) -> (n_bl, n_freq, n_time)
    return jnp.mean(vis_rfi_fine, axis=(3, 4))


# --- Test parameters ---------------------------------------------------------

Precision = namedtuple("Precision", "real complex rtol atol")

PRECISIONS = [
    Precision(jnp.float32, jnp.complex64, rtol=1e-5, atol=1e-5),
    Precision(jnp.float64, jnp.complex128, rtol=1e-11, atol=1e-11),
]

Shape = namedtuple("Shape", "n_ant n_freq n_time n_rfi n_int_freq n_int_time")

SHAPES = {
    # n_red = n_rfi * n_int_freq * n_int_time = 18, i.e. not a multiple of the
    # SIMD width: exercises both the vectorised body and the scalar tail.
    "tail": Shape(n_ant=5, n_freq=3, n_time=2, n_rfi=3, n_int_freq=2, n_int_time=3),
    # n_red = 32, a whole number of vectors on every supported target.
    "vector": Shape(n_ant=4, n_freq=2, n_time=3, n_rfi=4, n_int_freq=4, n_int_time=2),
    # n_red = 1, everything is handled by the scalar tail.
    "scalar": Shape(n_ant=6, n_freq=1, n_time=2, n_rfi=1, n_int_freq=1, n_int_time=1),
    "large": Shape(n_ant=16, n_freq=4, n_time=10, n_rfi=24, n_int_freq=6, n_int_time=8),
}

# Shape used by the tests that are not specifically about shapes.
DEFAULT_SHAPE = SHAPES["tail"]

BASELINE_LAYOUTS = pytest.mark.parametrize(
    "auto_corr, shuffle",
    [(True, False), (False, True), (True, True)],
    ids=["auto_corr", "shuffled", "auto_corr_shuffled"],
)


def _available_devices():
    """One device per platform for which the matching FFI library was found."""
    devices = []
    devices.extend(jax.devices("cpu")[:1])
    if rfi_vis_op_module._TAB_LIB_GPU is not None:
        for platform in ("cuda", "rocm"):
            try:
                devices.extend(jax.devices(platform)[:1])
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


# --- Helpers -----------------------------------------------------------------


def make_baselines(n_ant, auto_corr=False, shuffle=False, seed=0):
    """Antenna index pairs for every baseline of an ``n_ant`` array."""
    offset = 0 if auto_corr else 1
    pairs = np.asarray(
        [(i, j) for i in range(n_ant) for j in range(i + offset, n_ant)],
        dtype=np.int32,
    )
    if shuffle:
        pairs = pairs[np.random.default_rng(seed).permutation(len(pairs))]
    return jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])


def make_signal(shape, precision, seed=0):
    """Random ``(rfi_amp_fine, rfi_phase)`` pair of the requested precision."""
    rng = np.random.default_rng(seed)
    dims = tuple(shape)  # (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)
    amp = rng.normal(size=dims) + 1.0j * rng.normal(size=dims)
    phase = rng.uniform(-np.pi, np.pi, size=dims)
    return (
        jnp.asarray(amp, dtype=precision.complex),
        jnp.asarray(phase, dtype=precision.real),
    )


def make_cotangent(shape, n_bl, precision, seed=0):
    """Random cotangent matching the ``(n_bl, n_freq, n_time)`` output."""
    rng = np.random.default_rng(seed)
    dims = (n_bl, shape.n_freq, shape.n_time)
    return jnp.asarray(
        rng.normal(size=dims) + 1.0j * rng.normal(size=dims), dtype=precision.complex
    )


def assert_close(actual, expected, precision, what=""):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, what
    assert actual.dtype == expected.dtype, what
    # The absolute tolerance is scaled with the magnitude of the result so that
    # entries close to zero are not held to the full relative precision.
    scale = max(float(np.max(np.abs(expected))), 1.0) if expected.size else 1.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=precision.rtol,
        atol=precision.atol * scale,
        err_msg=what,
    )


# --- Direct evaluation -------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_eval_matches_reference(device, precision, shape):
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    vis = op.eval(amp, phase)

    assert vis.shape == (len(a1), shape.n_freq, shape.n_time)
    assert vis.dtype == precision.complex
    assert_close(vis, ref_rfi_vis_kernel(amp, phase, a1, a2), precision)


@BASELINE_LAYOUTS
def test_eval_baseline_layouts(device, precision, auto_corr, shuffle):
    """Auto-correlations and an unsorted baseline order must be handled."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    assert_close(op.eval(amp, phase), ref_rfi_vis_kernel(amp, phase, a1, a2), precision)


def test_eval_under_jit(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)

    assert_close(jax.jit(op.eval)(amp, phase), op.eval(amp, phase), precision)


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


# --- Forward mode (JVP) ------------------------------------------------------


def test_jvp_matches_reference(device, precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    amp_dot, phase_dot = make_signal(shape, precision, seed=1)
    op = RFIVisOp(shape.n_ant, a1, a2)

    vis, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    ref_vis, ref_vis_dot = jax.jvp(
        lambda a, p: ref_rfi_vis_kernel(a, p, a1, a2),
        (amp, phase),
        (amp_dot, phase_dot),
    )

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

    if wrt == "amp":
        fun = lambda a: op.eval(a, phase)  # noqa: E731
        ref_fun = lambda a: ref_rfi_vis_kernel(a, phase, a1, a2)  # noqa: E731
        primal, tangent = amp, amp_dot
    else:
        fun = lambda p: op.eval(amp, p)  # noqa: E731
        ref_fun = lambda p: ref_rfi_vis_kernel(amp, p, a1, a2)  # noqa: E731
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

    _, vis_dot = jax.jvp(op.eval, (amp, phase), (amp_dot, phase_dot))
    _, ref_vis_dot = jax.jvp(
        lambda a, p: ref_rfi_vis_kernel(a, p, a1, a2),
        (amp, phase),
        (amp_dot, phase_dot),
    )

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
    cotangent = make_cotangent(shape, len(a1), precision, seed=2)

    vis, vjp_fun = jax.vjp(op.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(cotangent)

    ref_vis, ref_vjp_fun = jax.vjp(
        lambda a, p: ref_rfi_vis_kernel(a, p, a1, a2), amp, phase
    )
    ref_amp_bar, ref_phase_bar = ref_vjp_fun(cotangent)

    assert_close(vis, ref_vis, precision, "primal")
    assert_close(amp_bar, ref_amp_bar, precision, "rfi_amp_fine cotangent")
    assert_close(phase_bar, ref_phase_bar, precision, "rfi_phase cotangent")


@BASELINE_LAYOUTS
def test_vjp_baseline_layouts(device, precision, auto_corr, shuffle):
    """The transpose kernel walks the baselines through the sorted indices."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)
    op = RFIVisOp(shape.n_ant, a1, a2)
    cotangent = make_cotangent(shape, len(a1), precision, seed=2)

    _, vjp_fun = jax.vjp(op.eval, amp, phase)
    _, ref_vjp_fun = jax.vjp(lambda a, p: ref_rfi_vis_kernel(a, p, a1, a2), amp, phase)

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

    def loss(fun):
        return lambda a, p: jnp.sum(jnp.abs(fun(a, p)) ** 2)

    amp_bar, phase_bar = jax.grad(loss(op.eval), argnums=(0, 1))(amp, phase)
    ref_amp_bar, ref_phase_bar = jax.grad(
        loss(lambda a, p: ref_rfi_vis_kernel(a, p, a1, a2)), argnums=(0, 1)
    )(amp, phase)

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
    """``Re<w, J v> == Re<J^T w, v>``: the transpose kernel is the transpose.

    The map is only R-linear (the kernel conjugates the second antenna), so the
    pairing is JAX's: the bilinear ``sum(x * y)`` restricted to its real part.
    """
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

    # Both sides cancel heavily, so the tolerance follows the size of the terms
    # that go into the sums rather than the size of the result.
    scale = float(jnp.sum(jnp.abs(cotangent) * jnp.abs(vis_dot)))
    np.testing.assert_allclose(
        float(forward), float(backward), rtol=0.0, atol=precision.rtol * scale
    )
