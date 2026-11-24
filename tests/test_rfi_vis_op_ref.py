"""Tests for :class:`ri_kernels.jax_api.RFIVisOpRef`, the reference the
:class:`RFIVisOp` tests check against.

The reference needs no FFI library. Its value is checked against the
definition summed term by term in NumPy, and the derivatives JAX forms for it
against central differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from helpers import (
    BASELINE_LAYOUTS, DEFAULT_SHAPE, PRECISIONS, SHAPES,
    assert_close, make_baselines, make_cotangent, make_signal,
)
from ri_kernels.jax_api import RFIVisOpRef


@pytest.fixture(params=PRECISIONS, ids=["f32", "f64"])
def precision(request):
    return request.param


def direct_sum(amp, phase, a1, a2):
    """The visibilities from the definition, one baseline at a time, in float64."""
    amp, phase = np.asarray(amp, np.complex128), np.asarray(phase, np.float64)
    vis = np.zeros((len(a1),) + amp.shape[1:3], np.complex128)
    for b, (p, q) in enumerate(zip(np.asarray(a1), np.asarray(a2))):
        # (n_freq, n_time, n_rfi, n_int_freq, n_int_time)
        fine = amp[p] * amp[q].conj() * np.exp(1j * (phase[p] - phase[q]))
        # Sum the sources, average the fine samples.
        vis[b] = fine.sum(axis=2).mean(axis=(2, 3))
    return vis


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_matches_direct_sum(precision, shape):
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert vis.shape == (len(a1), shape.n_freq, shape.n_time)
    assert vis.dtype == precision.complex
    assert_close(vis, direct_sum(amp, phase, a1, a2).astype(precision.complex), precision)


@BASELINE_LAYOUTS
def test_baseline_layouts(precision, auto_corr, shuffle):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert_close(vis, direct_sum(amp, phase, a1, a2).astype(precision.complex), precision)


def test_under_jit(precision):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    assert_close(jax.jit(ref.eval)(amp, phase), ref.eval(amp, phase), precision)


def test_derivatives_match_finite_differences():
    """The kernel tests take the reference's JAX derivatives as the truth."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=True, shuffle=True)
    amp, phase = make_signal(shape, PRECISIONS[1])

    check_grads(RFIVisOpRef(shape.n_ant, a1, a2).eval, (amp, phase), order=1, modes=("fwd", "rev"))


def test_single_precision_is_preserved():
    """x64 is on in the tests; float32 inputs still give complex64 values and
    cotangents of the inputs' dtypes."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, PRECISIONS[0])
    data = make_cotangent(shape, len(a1), PRECISIONS[0])
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    def loss(a, p):
        return jnp.sum(jnp.abs(ref.eval(a, p) - data) ** 2)

    amp_bar, phase_bar = jax.grad(loss, argnums=(0, 1))(amp, phase)

    assert ref.eval(amp, phase).dtype == jnp.complex64
    assert amp_bar.dtype == jnp.complex64
    assert phase_bar.dtype == jnp.float32


def test_reversal_is_conjugate_and_autos_are_positive(precision):
    shape = DEFAULT_SHAPE._replace(n_ant=3)
    a1 = jnp.array([0, 1, 0, 2, 1], jnp.int32)
    a2 = jnp.array([1, 0, 0, 2, 1], jnp.int32)
    amp, phase = make_signal(shape, precision)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert_close(vis[1], vis[0].conj(), precision)
    autos = np.asarray(vis[2:], np.complex128)
    scale = np.max(np.abs(autos))
    np.testing.assert_allclose(autos.imag, 0, atol=precision.atol * scale)
    assert np.min(autos.real) > 0


def test_empty_baselines():
    shape = DEFAULT_SHAPE
    empty = jnp.array([], jnp.int32)
    amp, phase = make_signal(shape, PRECISIONS[1])
    ref = RFIVisOpRef(shape.n_ant, empty, empty)

    value, vjp_fun = jax.vjp(ref.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(jnp.zeros_like(value))

    assert value.shape == (0, shape.n_freq, shape.n_time)
    np.testing.assert_array_equal(amp_bar, jnp.zeros_like(amp))
    np.testing.assert_array_equal(phase_bar, jnp.zeros_like(phase))


def test_mixed_precision_is_rejected():
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, _ = make_signal(shape, PRECISIONS[0])
    _, phase = make_signal(shape, PRECISIONS[1])

    with pytest.raises(TypeError, match="matched precision"):
        RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)


@pytest.mark.parametrize(
    "amp_shape, phase_shape, match",
    [
        ((5, 3, 2, 3, 2), (5, 3, 2, 3, 2), "rfi_amp_fine of shape"),
        ((5, 3, 2, 3, 2, 3), (5, 3, 2, 3, 2, 4), "rfi_phase of shape"),
    ],
    ids=["rank", "phase-mismatch"],
)
def test_bad_shapes(amp_shape, phase_shape, match):
    a1, a2 = make_baselines(5)

    with pytest.raises(ValueError, match=match):
        RFIVisOpRef(5, a1, a2).eval(jnp.zeros(amp_shape, jnp.complex128), jnp.zeros(phase_shape))
