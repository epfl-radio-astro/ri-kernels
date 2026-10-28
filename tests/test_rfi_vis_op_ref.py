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

from ri_kernels.jax_api import RFIVisOpRef
from test_rfi_vis_op import (
    BASELINE_LAYOUTS, DEFAULT_SHAPE, PRECISIONS, SHAPES,
    assert_close, make_baselines, make_cotangent, make_signal,
)


@pytest.fixture(params=PRECISIONS, ids=["f32", "f64"])
def precision(request):
    return request.param


@pytest.fixture(params=[1, 2], ids=lambda p: f"P{p}")
def pol(request):
    return request.param


def direct_sum(amp, phase, a1, a2):
    """The visibilities from the definition, one baseline and entry at a time, in float64."""
    amp, phase = np.asarray(amp, np.complex128), np.asarray(phase, np.float64)
    n_pol = amp.shape[6]
    vis = np.zeros((len(a1),) + amp.shape[1:3] + (n_pol, n_pol), np.complex128)
    for b, (p, q) in enumerate(zip(np.asarray(a1), np.asarray(a2))):
        rotation = np.exp(1j * (phase[p] - phase[q]))
        for i in range(n_pol):
            for j in range(n_pol):
                # (n_freq, n_time, n_rfi, n_int_freq, n_int_time), summed over the columns.
                fine = np.sum(amp[p, ..., i, :] * amp[q, ..., j, :].conj(), axis=-1) * rotation
                # Sum the sources, average the fine samples.
                vis[b, ..., i, j] = fine.sum(axis=2).mean(axis=(2, 3))
    return vis


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_matches_direct_sum(precision, pol, shape):
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision, pol)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert vis.shape == (len(a1), shape.n_freq, shape.n_time, pol, pol)
    assert_close(vis, direct_sum(amp, phase, a1, a2).astype(precision.complex), precision)


@BASELINE_LAYOUTS
def test_baseline_layouts(precision, pol, auto_corr, shuffle):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=auto_corr, shuffle=shuffle)
    amp, phase = make_signal(shape, precision, pol)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert_close(vis, direct_sum(amp, phase, a1, a2).astype(precision.complex), precision)


def test_under_jit(precision, pol):
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, precision, pol)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    assert_close(jax.jit(ref.eval)(amp, phase), ref.eval(amp, phase), precision)


def test_derivatives_match_finite_differences(pol):
    """The kernel tests take the reference's JAX derivatives as the truth."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant, auto_corr=True, shuffle=True)
    amp, phase = make_signal(shape, PRECISIONS[1], pol)

    check_grads(RFIVisOpRef(shape.n_ant, a1, a2).eval, (amp, phase), order=1, modes=("fwd", "rev"))


def test_single_precision_is_preserved(pol):
    """x64 is on in the tests; float32 inputs still give complex64 values and
    cotangents of the inputs' dtypes."""
    shape = DEFAULT_SHAPE
    a1, a2 = make_baselines(shape.n_ant)
    amp, phase = make_signal(shape, PRECISIONS[0], pol)
    data = make_cotangent(shape, len(a1), PRECISIONS[0], pol)
    ref = RFIVisOpRef(shape.n_ant, a1, a2)

    def loss(a, p):
        return jnp.sum(jnp.abs(ref.eval(a, p) - data) ** 2)

    amp_bar, phase_bar = jax.grad(loss, argnums=(0, 1))(amp, phase)

    assert ref.eval(amp, phase).dtype == jnp.complex64
    assert amp_bar.dtype == jnp.complex64
    assert phase_bar.dtype == jnp.float32


def test_reversal_is_conjugate_transpose_and_autos_are_positive(precision):
    shape = DEFAULT_SHAPE._replace(n_ant=3)
    a1 = jnp.array([0, 1, 0, 2, 1], jnp.int32)
    a2 = jnp.array([1, 0, 0, 2, 1], jnp.int32)
    amp, phase = make_signal(shape, precision, pol=2)

    vis = RFIVisOpRef(shape.n_ant, a1, a2).eval(amp, phase)

    assert_close(vis[1], vis[0].conj().swapaxes(-1, -2), precision)
    autos = np.asarray(vis[2:], np.complex128)
    assert_close(autos, autos.conj().swapaxes(-1, -2), precision)
    scale = np.max(np.abs(autos))
    assert np.min(np.linalg.eigvalsh(autos)) > -1e-5 * scale


def test_empty_baselines(pol):
    shape = DEFAULT_SHAPE
    empty = jnp.array([], jnp.int32)
    amp, phase = make_signal(shape, PRECISIONS[1], pol)
    ref = RFIVisOpRef(shape.n_ant, empty, empty)

    value, vjp_fun = jax.vjp(ref.eval, amp, phase)
    amp_bar, phase_bar = vjp_fun(jnp.zeros_like(value))

    assert value.shape == (0, shape.n_freq, shape.n_time, pol, pol)
    np.testing.assert_array_equal(amp_bar, jnp.zeros_like(amp))
    np.testing.assert_array_equal(phase_bar, jnp.zeros_like(phase))


@pytest.mark.parametrize(
    "amp_shape, phase_shape, match",
    [
        ((5, 3, 2, 3, 2, 3), (5, 3, 2, 3, 2, 3), "pad a scalar signal"),
        ((5, 3, 2, 3, 2, 3, 3, 2), (5, 3, 2, 3, 2, 3), "P, 2"),
        ((5, 3, 2, 3, 2, 3, 2, 2), (5, 3, 2, 3, 2, 4), "rfi_phase of shape"),
    ],
    ids=["scalar", "three-receivers", "phase-mismatch"],
)
def test_bad_shapes(amp_shape, phase_shape, match):
    a1, a2 = make_baselines(5)

    with pytest.raises(ValueError, match=match):
        RFIVisOpRef(5, a1, a2).eval(jnp.zeros(amp_shape, jnp.complex128), jnp.zeros(phase_shape))
