"""Analytic CPU/CUDA values and amplitude derivatives against the JAX specification."""
from functools import partial
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from analytic_reference import analytic_rfi_vis
from ri_kernels.jax_api.rfi_analytic_vis_op import (
    RFIAnalyticVisOp, _TAB_LIB_ANALYTIC, _TAB_LIB_ANALYTIC_GPU,
    rfi_analytic_vis_op, rfi_analytic_jvp_op, rfi_analytic_transpose_op,
)


def _devices():
    devices = jax.devices("cpu")[:1] if _TAB_LIB_ANALYTIC else []
    if _TAB_LIB_ANALYTIC_GPU:
        for platform in ("cuda", "rocm"):
            try:
                devices += jax.devices(platform)[:1]
            except RuntimeError:
                pass
    return devices


@pytest.fixture(params=_devices(), ids=lambda d: d.platform)
def device(request):
    with jax.default_device(request.param):
        yield request.param


@pytest.fixture(params=[(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)])
def precision(request):
    return request.param


def lagrange_basis(nodes, x):
    basis = np.ones((len(nodes), len(x)))
    for k, x_k in enumerate(nodes):
        for j, x_j in enumerate(nodes):
            if j != k:
                basis[k] *= (x - x_j) / (x_k - x_j)
    return basis


def interp_tables(n_cells, half_width, offsets):
    half_width = min(half_width, (n_cells - 1) // 2)
    n_stencil = 2 * half_width + 1
    start = np.clip(np.arange(n_cells) - half_width, 0, n_cells - n_stencil)
    nodes = np.arange(n_stencil)
    weights = np.stack([lagrange_basis(nodes, (c - s) + offsets) for c, s in enumerate(start)])
    return weights, start.astype(np.int32)


def fine_offsets(n_int, spacing):
    return (np.arange(n_int) - n_int // 2) * (spacing / n_int)


def _interp_inputs(real, complex_, seed=0, n_ant=5, n_rfi=2, n_freq=3, n_time=6,
                n_int_f=2, n_int_t=3, half_width=1, n_path=4):
    """Random data-grid inputs and the tables for a config of that shape."""
    rng = np.random.default_rng(seed)
    int_time, chan_width = 2.0, 1e4
    dt = fine_offsets(n_int_t, int_time)
    dnu = fine_offsets(n_int_f, chan_width) / 1e6  # MHz
    freqs = (1.5e8 + chan_width * np.arange(n_freq)) / 1e6  # MHz
    w_time, start_time = interp_tables(n_time, half_width, dt / int_time)
    w_freq, start_freq = interp_tables(n_freq, half_width, dnu * 1e6 / chan_width)

    shape = (n_ant, n_rfi, n_freq, n_time)
    amp = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    phase = rng.uniform(-2 * np.pi, 0.0, size=shape)
    # The delay relative to the array mean, in microseconds, as tabascal's
    # FixedOrbitCoarse writes it: a kilometre-scale array against a LEO satellite.
    scales = [1.7, 0.07, 3e-4, 3e-6, 3e-8, 3e-10][:n_path]
    delay = np.stack(
        [rng.normal(0.0, s, (n_ant, n_rfi, n_time)) for s in scales], axis=-1
    )
    arrays = (amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs)
    out = []
    for x in arrays:
        if np.iscomplexobj(x):
            out.append(jnp.asarray(x, dtype=complex_))
        elif x.dtype.kind == "i":
            out.append(jnp.asarray(x, dtype=jnp.int32))
        else:
            out.append(jnp.asarray(x, dtype=real))
    return out


def make_baselines(n_ant, shuffle=False, autocorr=True):
    lo = 0 if autocorr else 1
    pairs = np.asarray(
        [(i, j) for i in range(n_ant) for j in range(i + lo, n_ant)], dtype=np.int32
    )
    if shuffle:
        pairs = pairs[np.random.default_rng(4).permutation(len(pairs))]
    return jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])


def monomial_tables(n_cells: int, half_width: int):
    """Lagrange coefficients in x = 2*tau/T, including the shifted edge stencils.

    These occupy the time-table slot of ``fine_signal``: the identical stencil
    contraction now yields polynomial coefficients rather than sampled values.
    No Vandermonde fit is needed; multiplying each basis's linear factors gives
    its monomials directly on the host in double precision.
    """
    _, starts = interp_tables(n_cells, half_width, np.zeros(1))
    degree = 2 * min(half_width, (n_cells - 1) // 2)
    coefficients = np.zeros((n_cells, degree + 1, degree + 1))
    for cell, start in enumerate(starts):
        nodes = 2 * (np.arange(degree + 1) + start - cell)
        for k, node in enumerate(nodes):
            polynomial = np.polynomial.Polynomial([1.])
            for j, other in enumerate(nodes):
                if j != k:
                    polynomial *= np.polynomial.Polynomial([-other, 1.]) / (node - other)
            coefficients[cell, k, :len(polynomial.coef)] = polynomial.coef
    return coefficients, starts


def make_inputs(real, complex_, **shape):
    args = _interp_inputs(real, complex_, **shape)
    g, starts = monomial_tables(args[0].shape[-1], shape.get("half_width", 1))
    args[5] = jnp.asarray(g, real)
    args[6] = jnp.asarray(starts, jnp.int32)
    args[8] = jnp.asarray(2., real)
    return args


def upcast(x):
    if jnp.iscomplexobj(x):
        return x.astype(jnp.complex128)
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.float64)
    return x


def reference(amp, phase, delay, *tables, a1, a2, **options):
    return analytic_rfi_vis(amp.swapaxes(0, 1), phase.swapaxes(0, 1),
                            delay.swapaxes(0, 1), *tables, a1, a2, **options)


def assert_close(actual, expected, real):
    tolerance = 3e-5 if real == jnp.float32 else 5e-10
    scale = max(float(np.max(np.abs(expected), initial=0)), 1.)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("half_width,options,phase_terms", [
    (0, dict(segments=1, terms=1, cubic_terms=0), (.7, 0., 0.)),
    (1, dict(segments=2, terms=6, cubic_terms=3), (20.9, .5, .002)),
    (2, dict(segments=3, terms=16, cubic_terms=3), (0., 12., .02)),
])
def test_reference_preserves_single_precision(compiled, half_width, options, phase_terms):
    # The kernel tests deliberately upcast their reference inputs. Exercise
    # the reference itself in single precision with x64 still enabled: turning
    # x64 off hides promotions from integer loop indices and real constants.
    assert jax.config.x64_enabled
    with jax.default_device(jax.devices("cpu")[0]):
        args = make_inputs(jnp.float32, jnp.complex64, n_ant=2, n_rfi=1,
                           n_freq=1, n_time=5, n_int_f=1, half_width=half_width)
        a, b, c = phase_terms
        nu = float(args[-1][0])
        delay = jnp.zeros_like(args[2])
        for k, value in enumerate((a / (2*np.pi*nu), b / (np.pi*nu), c / (np.pi/3*nu)), 1):
            delay = delay.at[0, 0, :, k].set(value)
        args[2] = delay
        fn = partial(reference, a1=jnp.array([0], jnp.int32),
                     a2=jnp.array([1], jnp.int32), **options)
        if compiled:
            fn = jax.jit(fn)
        full = [upcast(x) for x in args]
        tangent = jnp.full_like(args[0], .3 + .7j)
        cotangent = jnp.full((1, 1, 5), .7 - .2j, jnp.complex64)
        value, dot = jax.jvp(lambda amp: fn(amp, *args[1:]), (args[0],), (tangent,))
        value64, dot64 = jax.jvp(lambda amp: fn(amp, *full[1:]), (full[0],), (upcast(tangent),))
        bar = jax.vjp(lambda amp: fn(amp, *args[1:]), args[0])[1](cotangent)[0]
        bar64 = jax.vjp(lambda amp: fn(amp, *full[1:]), full[0])[1](upcast(cotangent))[0]
        for actual, expected in ((value, value64), (dot, dot64), (bar, bar64)):
            assert actual.dtype == jnp.complex64
            assert expected.dtype == jnp.complex128
            assert_close(actual, expected, jnp.float32)


SHAPES = {
    "default": {},
    "held": dict(half_width=0),
    "one-cell": dict(n_time=1, n_freq=1, n_int_f=1),
    "wide": dict(half_width=2, n_time=5, n_freq=5),
    "constant-delay": dict(n_path=1),
    "linear-delay": dict(n_path=2),
    "quadratic-delay": dict(n_path=3),
    "higher-delay": dict(n_path=6),
}
OPTIONS = [dict(segments=2, terms=6, cubic_terms=3),
           dict(segments=4, terms=16, cubic_terms=3),
           dict(segments=1, terms=6, cubic_terms=0)]


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("options", OPTIONS)
def test_value_jvp_vjp(precision, device, shape, options):
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    op = RFIAnalyticVisOp(args[0].shape[0], a1, a2)
    fn = partial(op.eval, **options)
    ref_args = [upcast(x) for x in args]
    ref = partial(reference, a1=a1, a2=a2, **options)
    expected = jax.jit(ref)(*ref_args)
    assert_close(fn(*args), expected, real)
    assert_close(jax.jit(fn)(*args), expected, real)
    tangent = make_inputs(real, complex_, seed=1, **shape)[0]
    derivative = jax.jit(lambda a, da: jax.jvp(lambda x: fn(x, *args[1:]), (a,), (da,))[1])
    expected_dot = jax.jvp(lambda a: ref(a, *ref_args[1:]), (ref_args[0],), (upcast(tangent),))[1]
    actual_dot = derivative(args[0], tangent)
    assert_close(actual_dot, expected_dot, real)
    rng = np.random.default_rng(22)
    g = jnp.asarray(rng.normal(size=expected.shape) + 1j * rng.normal(size=expected.shape), complex_)
    bar = jax.jit(lambda a, g: jax.vjp(lambda x: fn(x, *args[1:]), a)[1](g)[0])(args[0], g)
    expected_bar = jax.vjp(lambda a: ref(a, *ref_args[1:]), ref_args[0])[1](upcast(g))[0]
    assert_close(bar, expected_bar, real)
    # The pairing is real and bilinear under JAX's complex cotangent convention.
    assert_close(jnp.real(jnp.sum(g * actual_dot)), jnp.real(jnp.sum(bar * tangent)), real)


@pytest.mark.parametrize("a,b,c", [
    (0., 0., 0.), (0.3, 0.01, 0.), (20.9, 0.5, .002),
    (21.1, -.5, -.002), (10000., .2, .002),
    (0., 4., .02), (0., -4., -.02), (10., 4., .01),
    (10.0001, 4., .01), (1., 4., 0.), (1.0001, 4., 0.), (0., 40., .02), (0., -40., .02),
    (0., np.pi / 2 * 2.4999**2, 0.),
    (0., np.pi / 2 * 2.5001**2, 0.),
])
def test_phase_regimes(precision, device, a, b, c):
    real, complex_ = precision
    args = make_inputs(real, complex_, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    delay = jnp.zeros_like(args[2])
    nu = float(args[-1][0])
    delay = delay.at[0, 0, :, 1].set(a / (2 * np.pi * nu))
    delay = delay.at[0, 0, :, 2].set(b / (np.pi * nu))
    delay = delay.at[0, 0, :, 3].set(c / (np.pi / 3 * nu))
    args[2] = delay
    a1, a2 = jnp.array([0], jnp.int32), jnp.array([1], jnp.int32)
    op = RFIAnalyticVisOp(2, a1, a2)
    options = dict(segments=2, terms=6, cubic_terms=3)
    ref = partial(reference, a1=a1, a2=a2, **options)
    ref_args = [upcast(x) for x in args]
    expected = ref(*ref_args)
    assert_close(op.eval(*args, **options), expected, real)
    g = jnp.full(expected.shape, .3 + .7j, complex_)
    actual_bar = jax.vjp(lambda amp: op.eval(amp, *args[1:], **options), args[0])[1](g)[0]
    expected_bar = jax.vjp(lambda amp: ref(amp, *ref_args[1:]), ref_args[0])[1](upcast(g))[0]
    assert_close(actual_bar, expected_bar, real)


@pytest.mark.parametrize("n_ant", [1, 33, 65])
def test_sparse_reversed_and_auto_baselines(precision, device, n_ant):
    real, complex_ = precision
    pairs = sorted(set([(0, 0), (n_ant - 1, n_ant - 1), (0, n_ant - 1), (n_ant - 1, 0),
                        (min(31, n_ant - 1), min(32, n_ant - 1))]))[::-1]
    a1, a2 = (jnp.asarray(x, jnp.int32) for x in np.asarray(pairs).T)
    args = make_inputs(real, complex_, n_ant=n_ant, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    op = RFIAnalyticVisOp(n_ant, a1, a2)
    ref_args = [upcast(x) for x in args]
    ref = partial(reference, a1=a1, a2=a2, segments=2, terms=6, cubic_terms=3)
    assert_close(op.eval(*args), ref(*ref_args), real)
    g = jnp.full((len(a1), 1, 3), .4 + .7j, complex_)
    actual = jax.vjp(lambda a: op.eval(a, *args[1:]), args[0])[1](g)[0]
    expected = jax.vjp(lambda a: ref(a, *ref_args[1:]), ref_args[0])[1](upcast(g))[0]
    assert_close(actual, expected, real)


def test_constants_and_zero_tangent(precision, device):
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIAnalyticVisOp(args[0].shape[0], a1, a2)
    value, pullback = jax.vjp(op.eval, *args)
    bars = pullback(jnp.ones_like(value))
    assert np.max(np.abs(bars[0])) > 0
    for i in [1, 2, 3, 5, 7, 8, 9]:
        np.testing.assert_array_equal(bars[i], jnp.zeros_like(args[i]))
    _, dot = jax.jvp(lambda p: op.eval(args[0], p, *args[2:]), (args[1],), (jnp.ones_like(args[1]),))
    np.testing.assert_array_equal(dot, jnp.zeros_like(value))


def test_cubic_translation_matters(device):
    args = make_inputs(jnp.float64, jnp.complex128, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    args[2] = jnp.zeros_like(args[2]).at[0, :, :, 3].set(.0005)
    a1, a2 = jnp.array([0], jnp.int32), jnp.array([1], jnp.int32)
    op = RFIAnalyticVisOp(2, a1, a2)
    full = op.eval(*args)
    omitted = op.eval(*args, cubic_terms=0)
    assert np.max(np.abs(full - omitted)) > 1e-4
    expected = reference(*args, a1=a1, a2=a2, segments=2, terms=6, cubic_terms=3)
    assert_close(full, expected, jnp.float64)


@pytest.mark.parametrize("slot,value,match", [
    (5, jnp.zeros((6, 3, 10)), "monomial"),
    (8, jnp.ones(2), "scalar"), (7, jnp.zeros((2, 1)), "dnu|w_freq"),
    (6, jnp.zeros(6, jnp.int64), "int32"),
])
def test_bad_shapes(slot, value, match, device):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[slot] = value
    a1, a2 = make_baselines(5)
    with pytest.raises((TypeError, ValueError), match=match):
        RFIAnalyticVisOp(5, a1, a2).eval(*args)


@pytest.mark.parametrize("options", [dict(segments=0), dict(terms=33), dict(cubic_terms=-1), dict(terms=2.5)])
def test_bad_options(options, device):
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    with pytest.raises(ValueError, match="static integer"):
        RFIAnalyticVisOp(5, a1, a2).eval(*args, **options)


def test_direct_bind_derivatives_and_shapes(device):
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    options = dict(segments=2, terms=6, cubic_terms=3)
    with pytest.raises(TypeError, match="amp only"):
        jax.jvp(lambda phase: rfi_analytic_vis_op.bind(*op.indices, args[0], phase, *args[2:], **options),
                (args[1],), (jnp.ones_like(args[1]),))
    with pytest.raises(ValueError, match="signal tangent"):
        rfi_analytic_jvp_op.bind(*op.indices, args[0], args[0][:-1], *args[1:], **options)
    with pytest.raises(ValueError, match="visibility cotangent"):
        rfi_analytic_transpose_op.bind(*op.indices, *args, jnp.ones((1, 1, 1), jnp.complex128), **options)


def test_invalid_cpu_values(device):
    if device.platform != "cpu":
        pytest.skip("GPU stencil values are caller-validated, as for RFIInterpVisOp")
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    for slot, value, match in [(8, jnp.array(-1.), "positive"),
                               (6, jnp.full((6,), -1, jnp.int32), "stencil"),
                               (4, jnp.full((3,), 3, jnp.int32), "stencil")]:
        bad = list(args); bad[slot] = value
        with pytest.raises(Exception, match=match):
            op.eval(*bad).block_until_ready()


def test_duplicate_baselines():
    with pytest.raises(ValueError, match="at most once"):
        RFIAnalyticVisOp(2, np.array([0, 0]), np.array([1, 1]))


def test_empty_baselines(precision, device):
    real, complex_ = precision
    args = make_inputs(real, complex_)
    empty = jnp.array([], jnp.int32)
    op = RFIAnalyticVisOp(5, empty, empty)
    value, pullback = jax.vjp(lambda a: op.eval(a, *args[1:]), args[0])
    assert value.shape == (0, 3, 6)
    np.testing.assert_array_equal(pullback(jnp.zeros_like(value))[0], jnp.zeros_like(args[0]))


@pytest.mark.parametrize("shape,options", [
    (dict(half_width=4, n_time=9, n_freq=1, n_rfi=1, n_ant=2),
     dict(segments=2, terms=32, cubic_terms=8)),
    (dict(half_width=3, n_time=7, n_freq=1, n_rfi=1, n_ant=2),
     dict(segments=3, terms=1, cubic_terms=1)),
])
def test_general_degree_bounds(precision, device, shape, options):
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    a1, a2 = make_baselines(2)
    op = RFIAnalyticVisOp(2, a1, a2)
    ref_args = [upcast(x) for x in args]
    ref = partial(reference, a1=a1, a2=a2, **options)
    assert_close(op.eval(*args, **options), ref(*ref_args), real)
    g = jnp.full((len(a1), 1, shape["n_time"]), .5 + .2j, complex_)
    bar = jax.vjp(lambda a: op.eval(a, *args[1:], **options), args[0])[1](g)[0]
    expected = jax.vjp(lambda a: ref(a, *ref_args[1:]), ref_args[0])[1](upcast(g))[0]
    assert_close(bar, expected, real)


@pytest.mark.parametrize("slot", [1, 2, 3, 5, 7, 8, 9])
def test_mixed_real_precision(slot, device):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[slot] = args[slot].astype(jnp.float32)
    a1, a2 = make_baselines(5)
    with pytest.raises(TypeError, match="dtype|precision"):
        RFIAnalyticVisOp(5, a1, a2).eval(*args)


@pytest.mark.parametrize("real_name", ["float32", "float64"])
def test_gpu_scratch_chunks(real_name, tmp_path, device):
    if device.platform == "cpu":
        pytest.skip("Scratch chunking is a GPU path")
    # This shape forces a frequency split at 1 MiB even for complex64. The
    # final chunks are short on both axes, and edge stencils cross chunks.
    code = r'''
import sys
sys.meta_path[:] = [f for f in sys.meta_path if "ScikitBuild" not in type(f).__name__]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from test_rfi_analytic_vis_op import make_inputs, reference, upcast, assert_close, RFIAnalyticVisOp
real = getattr(jnp, sys.argv[1])
complex_ = jnp.complex64 if real == jnp.float32 else jnp.complex128
device = next(d for d in jax.devices() if d.platform != "cpu")
with jax.default_device(device):
    args = make_inputs(real, complex_, n_ant=65, n_rfi=2, n_freq=13, n_time=5, n_int_f=9)
    a1 = jnp.array([0,64,31,32,0,32,64], jnp.int32)
    a2 = jnp.array([64,0,32,31,0,32,64], jnp.int32)
    op = RFIAnalyticVisOp(65, a1, a2)
    ref_args = [upcast(x) for x in args]
    ref = partial(reference, a1=a1, a2=a2, segments=2, terms=6, cubic_terms=3)
    da = jnp.full_like(args[0], .3 + .7j)
    g = jnp.full((len(a1), 13, 5), .7 - .2j, complex_)
    fn = lambda a: op.eval(a, *args[1:])
    rf = lambda a: ref(a, *ref_args[1:])
    y, dy = jax.jit(lambda a, d: jax.jvp(fn, (a,), (d,)))(args[0], da)
    bar = jax.jit(lambda a, g: jax.vjp(fn, a)[1](g)[0])(args[0], g)
    ry, rd = jax.jit(lambda a, d: jax.jvp(rf, (a,), (d,)))(ref_args[0], upcast(da))
    rb = jax.jit(lambda a, g: jax.vjp(rf, a)[1](g)[0])(ref_args[0], upcast(g))
    for actual, expected in [(y,ry), (dy,rd), (bar,rb)]: assert_close(actual, expected, real)
    np.savez(sys.argv[2], value=np.asarray(y), jvp=np.asarray(dy), vjp=np.asarray(bar))
'''
    root = Path(__file__).resolve().parents[1]
    results = []
    for budget in (1, 256):
        dest = tmp_path / f"scratch-{budget}.npz"
        env = dict(os.environ, RI_KERNELS_INTERP_SCRATCH_MB=str(budget),
                   PYTHONPATH=os.pathsep.join([str(root), str(root / "tests"), os.environ.get("PYTHONPATH", "")]))
        run = subprocess.run([sys.executable, "-c", code, real_name, str(dest)],
                             env=env, capture_output=True, text=True, timeout=240)
        assert run.returncode == 0, run.stdout + run.stderr
        with np.load(dest) as saved:
            results.append({key: saved[key] for key in saved})
    for key in results[0]:
        assert_close(results[0][key], results[1][key], getattr(jnp, real_name))


@pytest.mark.parametrize("option,value", [
    ("terms", 33), ("cubic_terms", 9), ("segments", 0),
    ("terms", np.iinfo(np.int64).max),
])
def test_ffi_rejects_excess_capacity(option, value, device):
    # Bypass the Python primitive's validation. The extension must reject an
    # excessive configuration before constructing scratch or scheduling work.
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    options = dict(segments=np.int64(2), terms=np.int64(6), cubic_terms=np.int64(3))
    options[option] = np.int64(value)
    target = "calc_rfi_analytic" + ("_gpu" if device.platform != "cpu" else "") + "_f64"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((len(a1), 3, 6), jnp.complex128))
    with pytest.raises(Exception, match="capacity"):
        call(*op.indices, *args, **options).block_until_ready()


def test_ffi_rejects_excess_coefficients(device):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[5] = jnp.zeros((6, 3, 10), jnp.float64)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    target = "calc_rfi_analytic" + ("_gpu" if device.platform != "cpu" else "") + "_f64"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((len(a1), 3, 6), jnp.complex128))
    with pytest.raises(Exception, match="capacity"):
        call(*op.indices, *args, segments=np.int64(2), terms=np.int64(6),
             cubic_terms=np.int64(3)).block_until_ready()


@pytest.mark.parametrize("n_ant", [5, 33])
def test_analytic_given_indices_match_values_and_derivatives(precision, device, n_ant):
    from ri_kernels.jax_api import analytic_eval_with_indices
    real, complex_ = precision
    args = make_inputs(real, complex_, n_ant=n_ant, n_rfi=1, n_freq=1, n_time=3)
    a1, a2 = make_baselines(n_ant, shuffle=True, autocorr=False)
    whole = RFIAnalyticVisOp(n_ant, a1, a2)
    take = np.arange(0, len(a1), 3)
    part = RFIAnalyticVisOp(n_ant, a1[take], a2[take])
    options = dict(segments=3, terms=8, cubic_terms=2)
    reference = lambda amp: whole.eval(amp, *args[1:], **options)[take]
    given = lambda amp: analytic_eval_with_indices(part.indices, amp, *args[1:], **options)
    tangent = jnp.conj(args[0]) * 0.17
    cot = jnp.arange(len(take) * args[0].shape[3], dtype=real).reshape(len(take), 1, -1) / 10
    outputs = []
    for fn in (reference, given):
        value, dot = jax.jit(lambda a, da: jax.jvp(fn, (a,), (da,)))(args[0], tangent)
        grad = jax.jit(jax.grad(lambda a: jnp.real(jnp.sum(fn(a) * cot))))(args[0])
        outputs.append((value, dot, grad))
    for expected, actual in zip(*outputs):
        np.testing.assert_allclose(actual, expected, rtol=3e-5 if real == jnp.float32 else 1e-10,
                                   atol=3e-5 if real == jnp.float32 else 1e-10)
