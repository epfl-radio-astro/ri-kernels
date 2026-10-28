"""Tests for :class:`ri_kernels.jax_api.RFIAnalyticVisOpRef`, the reference the
:class:`RFIAnalyticVisOp` tests check against.

The reference needs no FFI library. Its closed form is checked against brute
force quadrature of the same integral, and the derivatives JAX forms for it
against central differences.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from ri_kernels.jax_api import RFIAnalyticVisOpRef
from test_rfi_analytic_vis_op import SHAPES, assert_close, make_baselines, make_inputs, upcast


@pytest.fixture(params=[1, 2], ids=lambda p: f"P{p}")
def pol(request):
    return request.param


def quadrature(amp, phase, delay, w_freq, start_freq, g_time, start_time, dnu_mhz, int_time,
               freq_mhz, a1, a2, *, cubic=True, pieces=64, nodes=16):
    """The visibilities by brute force, in float64.

    Each cell is split into ``pieces`` equal parts of x = 2*tau/int_time in
    [-1, 1], each with ``nodes`` Gauss-Legendre nodes. There, the signal is
    interpolated from its tables and the pair's product is averaged against
    the delay phase. The delay holds derivatives in us/s^k, so the path is
    ``sum_k delay[k] tau**k / k!`` through the cubic, or the quadratic without
    ``cubic``; its constant enters only through the fine channel's offset,
    since phase already holds its share at the channel centre.
    """
    amp, phase, delay, w_freq, g_time, dnu_mhz, freq_mhz = (
        np.asarray(x, np.complex128 if np.iscomplexobj(x) else np.float64)
        for x in (amp, phase, delay, w_freq, g_time, dnu_mhz, freq_mhz)
    )
    start_freq, start_time, a1, a2 = map(np.asarray, (start_freq, start_time, a1, a2))
    nodes, weights = np.polynomial.legendre.leggauss(nodes)
    centres = -1 + (2 * np.arange(pieces) + 1) / pieces
    x = (centres[:, None] + nodes / pieces).ravel()
    weights = np.tile(weights, pieces) / (2 * pieces)  # the mean over [-1, 1]
    tau = x * float(int_time) / 2
    n_freq, n_time, n_pol = amp.shape[2], amp.shape[3], amp.shape[4]
    n_sf, n_st = w_freq.shape[1], g_time.shape[1]
    orders = min(delay.shape[-1], 4 if cubic else 3)
    vis = np.zeros((len(a1), n_freq, n_time, n_pol, n_pol), np.complex128)
    for t in range(n_time):
        # The time basis of each stencil cell at the nodes: (n_st, n_x).
        basis = g_time[t] @ np.power.outer(x, np.arange(g_time.shape[2])).T
        d = delay[:, :, t]
        path = sum(d[..., k, None] * tau**k / math.factorial(k) for k in range(1, orders))
        for f in range(n_freq):
            for u, offset in enumerate(dnu_mhz):
                # (n_ant, n_rfi, P, 2, n_x)
                signal = sum(
                    w_freq[f, k, u] * basis[l] * amp[:, :, start_freq[f] + k, start_time[t] + l, :, :, None]
                    for k in range(n_sf) for l in range(n_st)
                )
                angle = phase[:, :, f, t, None] + 2 * np.pi * (offset * d[..., 0, None]
                                                               + (freq_mhz[f] + offset) * path)
                signal = signal * np.exp(1j * angle)[:, :, None, None]
                for b, (p, q) in enumerate(zip(a1, a2)):
                    vis[b, f, t] += np.einsum("ricx,rjcx,x->ij", signal[p], signal[q].conj(),
                                              weights) / len(dnu_mhz)
    return vis


def reference(args, a1, a2, **options):
    """The reference's visibilities for float64 inputs."""
    return RFIAnalyticVisOpRef(args[0].shape[0], a1, a2).eval(*args, **options)


def regime_inputs(a, b, c):
    """One baseline of one cell row, with these winding, curvature and cubic angles."""
    args = make_inputs(jnp.float64, jnp.complex128, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    nu = float(args[-1][0])
    delay = jnp.zeros_like(args[2])
    for k, value in enumerate((a / (2 * np.pi * nu), b / (np.pi * nu), c / (np.pi / 3 * nu)), 1):
        delay = delay.at[0, 0, :, k].set(value)
    args[2] = delay
    return args


CONSERVATIVE = dict(segments=4, terms=16, cubic_terms=3)


@pytest.mark.parametrize("options", [{}, CONSERVATIVE], ids=["default", "conservative"])
@pytest.mark.parametrize("shape", SHAPES.keys())
def test_matches_quadrature(pol, shape, options):
    args = make_inputs(jnp.float64, jnp.complex128, pol=pol, **SHAPES[shape])
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)

    value = reference(args, a1, a2, **options)

    assert value.shape == (len(a1),) + args[0].shape[2:4] + (pol, pol)
    assert_close(value, quadrature(*args, a1, a2), jnp.float64)


@pytest.mark.parametrize("a,b,c", [
    (0., 0., 0.), (0.3, 0.01, 0.), (20.9, 0.5, .002), (21.1, -.5, -.002), (10000., .2, .002),
    (0., 4., .02), (0., -4., -.02), (10., 4., .01), (1., 4., 0.), (0., 40., .02), (0., -40., .02),
    (0., np.pi / 2 * 2.4999**2, 0.), (0., np.pi / 2 * 2.5001**2, 0.),
])
def test_phase_regimes_match_quadrature(a, b, c):
    """Each side of the moments' branch points, with enough pieces that the
    series truncations sit below the tolerance; the large winding needs
    enough quadrature pieces to resolve its fringes."""
    args = regime_inputs(a, b, c)
    a1, a2 = jnp.array([0], jnp.int32), jnp.array([1], jnp.int32)

    value = reference(args, a1, a2, segments=8, terms=16, cubic_terms=3)

    assert_close(value, quadrature(*args, a1, a2, pieces=4096), jnp.float64)


def test_cubic_terms_zero_drops_the_cubic():
    args = make_inputs(jnp.float64, jnp.complex128, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    args[2] = jnp.zeros_like(args[2]).at[0, :, :, 3].set(.0005)
    a1, a2 = jnp.array([0], jnp.int32), jnp.array([1], jnp.int32)

    full = reference(args, a1, a2, **CONSERVATIVE)
    omitted = reference(args, a1, a2, **{**CONSERVATIVE, "cubic_terms": 0})

    assert_close(full, quadrature(*args, a1, a2), jnp.float64)
    assert_close(omitted, quadrature(*args, a1, a2, cubic=False), jnp.float64)
    assert np.max(np.abs(full - omitted)) > 1e-4


def test_derivatives_match_finite_differences(pol):
    """The kernel tests take the reference's JAX derivatives as the truth.

    The delay is differentiated in units that move each order's angle by a
    comparable amount, so that one step size suits them all.
    """
    args = make_inputs(jnp.float64, jnp.complex128, pol=pol)
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    ref = RFIAnalyticVisOpRef(args[0].shape[0], a1, a2)
    scales = jnp.asarray([10., 1e-3, 1e-3, 1e-3])

    def fn(amp, phase, delay):
        return ref.eval(amp, phase, delay * scales, *args[3:])

    check_grads(fn, (args[0], args[1], args[2] / scales), order=1, modes=("fwd", "rev"))


@pytest.mark.parametrize("half_width,options,phase_terms", [
    (0, dict(segments=1, terms=1, cubic_terms=0), (.7, 0., 0.)),
    (1, dict(segments=2, terms=6, cubic_terms=3), (20.9, .5, .002)),
    (2, dict(segments=3, terms=16, cubic_terms=3), (0., 12., .02)),
])
def test_preserves_single_precision(half_width, options, phase_terms):
    # The kernel tests deliberately upcast their reference inputs. Exercise
    # the reference itself in single precision with x64 still enabled: turning
    # x64 off hides promotions from integer loop indices and real constants.
    assert jax.config.x64_enabled
    args = make_inputs(jnp.float32, jnp.complex64, n_ant=2, n_rfi=1,
                       n_freq=1, n_time=5, n_int_f=1, half_width=half_width)
    a, b, c = phase_terms
    nu = float(args[-1][0])
    delay = jnp.zeros_like(args[2])
    for k, value in enumerate((a / (2*np.pi*nu), b / (np.pi*nu), c / (np.pi/3*nu)), 1):
        delay = delay.at[0, 0, :, k].set(value)
    args[2] = delay
    ref = RFIAnalyticVisOpRef(2, np.array([0]), np.array([1]))

    @jax.jit
    def run(args, tangent, cotangent):
        fn = lambda amp: ref.eval(amp, *args[1:], **options)
        value, dot = jax.jvp(fn, (args[0],), (tangent,))
        return value, dot, jax.vjp(fn, args[0])[1](cotangent)[0]

    tangent = jnp.full_like(args[0], .3 + .7j)
    cotangent = jnp.full((1, 1, 5, 2, 2), .7 - .2j, jnp.complex64)
    single = run(args, tangent, cotangent)
    double = run([upcast(x) for x in args], upcast(tangent), upcast(cotangent))
    for actual, expected in zip(single, double):
        assert actual.dtype == jnp.complex64
        assert expected.dtype == jnp.complex128
        assert_close(actual, expected, jnp.float32)


def test_constants_have_zero_gradient():
    """eval() stops the gradient on the interval and the tables; the signal,
    the phase and the delay get cotangents."""
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(args[0].shape[0])
    ref = RFIAnalyticVisOpRef(args[0].shape[0], a1, a2)

    value, pullback = jax.vjp(ref.eval, *args)
    bars = pullback(jnp.ones_like(value))

    for i in [0, 1, 2]:
        assert np.max(np.abs(bars[i])) > 0
    for i in [3, 5, 7, 8, 9]:
        np.testing.assert_array_equal(bars[i], jnp.zeros_like(args[i]))


def test_reversal_is_conjugate_transpose_and_autos_are_positive():
    args = make_inputs(jnp.float64, jnp.complex128, pol=2, n_ant=3)
    a1, a2 = jnp.array([0, 1, 0, 2, 1], jnp.int32), jnp.array([1, 0, 0, 2, 1], jnp.int32)

    value = reference(args, a1, a2)

    assert_close(value[1], value[0].conj().swapaxes(-1, -2), jnp.float64)
    autos = np.asarray(value[2:])
    assert_close(autos, autos.conj().swapaxes(-1, -2), jnp.float64)
    assert np.min(np.linalg.eigvalsh(autos)) > -1e-10 * np.max(np.abs(autos))


def test_empty_baselines(pol):
    args = make_inputs(jnp.float64, jnp.complex128, pol=pol)
    empty = jnp.array([], jnp.int32)
    ref = RFIAnalyticVisOpRef(5, empty, empty)

    value, pullback = jax.vjp(lambda a, d: ref.eval(a, args[1], d, *args[3:]), args[0], args[2])
    amp_bar, delay_bar = pullback(jnp.zeros_like(value))

    assert value.shape == (0, 3, 6, pol, pol)
    np.testing.assert_array_equal(amp_bar, jnp.zeros_like(args[0]))
    np.testing.assert_array_equal(delay_bar, jnp.zeros_like(args[2]))


@pytest.mark.parametrize("n_ant,a1,a2", [
    (5, [0, 1], [1]), (5, [0, 5], [1, 2]), (5, [-1], [0]), (5, [0.], [1.]), (0, [], []),
], ids=["mismatched", "too-large", "negative", "float", "no-antennas"])
def test_bad_baselines(n_ant, a1, a2):
    with pytest.raises(ValueError, match="antenna-index"):
        RFIAnalyticVisOpRef(n_ant, np.asarray(a1), np.asarray(a2))


@pytest.mark.parametrize("options", [dict(segments=0), dict(terms=0), dict(cubic_terms=-1),
                                     dict(terms=2.5), dict(segments=True)])
def test_bad_options(options):
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)

    with pytest.raises(ValueError, match="static integer"):
        RFIAnalyticVisOpRef(5, a1, a2).eval(*args, **options)


@pytest.mark.parametrize("slot,value,match", [
    (0, jnp.zeros((5, 2, 3, 6), jnp.complex128), "pad a scalar signal"),
    (0, jnp.zeros((5, 2, 3, 6, 3, 2), jnp.complex128), "P 1 or 2"),
    (0, jnp.zeros((4, 2, 3, 6, 2, 2), jnp.complex128), "Expected 5 antennas"),
    (1, jnp.zeros((5, 2, 3, 5)), "phase shape"),
    (2, jnp.zeros((5, 2, 5, 4)), "delay_us shape"),
    (2, jnp.zeros((5, 2, 6, 0)), "delay_us shape"),
], ids=["scalar", "three-receivers", "antennas", "phase", "delay", "no-delay-orders"])
def test_bad_shapes(slot, value, match):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[slot] = value
    a1, a2 = make_baselines(5)

    with pytest.raises(ValueError, match=match):
        RFIAnalyticVisOpRef(5, a1, a2).eval(*args)
