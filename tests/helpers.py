"""Shared test helpers and input generators (no FFI library imports)."""

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

# --- RFIVisOp shared test definitions ----------------------------------------

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

DEFAULT_SHAPE = SHAPES["tail"]

BASELINE_LAYOUTS = pytest.mark.parametrize(
    "auto_corr, shuffle",
    [(True, False), (False, True), (True, True)],
    ids=["auto_corr", "shuffled", "auto_corr_shuffled"],
)


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
    scale = max(float(np.max(np.abs(expected))), 1.0) if expected.size else 1.0
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=precision.rtol,
        atol=precision.atol * scale,
        err_msg=what,
    )


# --- RFIAnalyticVisOp shared test definitions --------------------------------

ANALYTIC_SHAPES = {
    "default": {},
    "held": dict(half_width=0),
    "one-cell": dict(n_time=1, n_freq=1, n_int_f=1),
    "wide": dict(half_width=2, n_time=5, n_freq=5),
    "constant-delay": dict(n_path=1),
    "linear-delay": dict(n_path=2),
    "quadratic-delay": dict(n_path=3),
    "higher-delay": dict(n_path=6),
}


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


def _interp_inputs(
    real, complex_, seed=0, n_ant=5, n_rfi=2, n_freq=3, n_time=6,
    n_int_f=2, n_int_t=3, half_width=1, n_path=4,
):
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


def make_analytic_baselines(n_ant, shuffle=False, autocorr=True):
    return make_baselines(n_ant, auto_corr=autocorr, shuffle=shuffle, seed=4)


def monomial_tables(n_cells: int, half_width: int):
    """Lagrange coefficients in x = 2*tau/T, including the shifted edge stencils."""
    _, starts = interp_tables(n_cells, half_width, np.zeros(1))
    degree = 2 * min(half_width, (n_cells - 1) // 2)
    coefficients = np.zeros((n_cells, degree + 1, degree + 1))
    for cell, start in enumerate(starts):
        nodes = 2 * (np.arange(degree + 1) + start - cell)
        for k, node in enumerate(nodes):
            polynomial = np.polynomial.Polynomial([1.0])
            for j, other in enumerate(nodes):
                if j != k:
                    polynomial *= np.polynomial.Polynomial([-other, 1.0]) / (node - other)
            coefficients[cell, k, :len(polynomial.coef)] = polynomial.coef
    return coefficients, starts


def make_analytic_inputs(real, complex_, **shape):
    args = _interp_inputs(real, complex_, **shape)
    g, starts = monomial_tables(args[0].shape[-1], shape.get("half_width", 1))
    args[5] = jnp.asarray(g, real)
    args[6] = jnp.asarray(starts, jnp.int32)
    args[8] = jnp.asarray(2.0, real)
    return args


def upcast(x):
    if jnp.iscomplexobj(x):
        return x.astype(jnp.complex128)
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.float64)
    return x


def assert_analytic_close(actual, expected, real):
    tolerance = 3e-5 if real == jnp.float32 else 5e-10
    scale = max(float(np.max(np.abs(expected), initial=0)), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)
