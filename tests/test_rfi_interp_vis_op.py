"""Tests for the data-grid RFI visibility operator, :class:`RFIInterpVisOp`.

The kernels are held to a plain JAX implementation of the same computation
(``reference``), which JAX differentiates for the expected JVP and VJP. The
tables come from the polynomial (Lagrange) interpolant, which is what
tabascal's ``rfi_vis:PolyInterpVisFFI`` feeds the operator, but nothing here
depends on that: the operator takes the tables as data.
"""

from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ri_kernels.jax_api.rfi_interp_vis_op import (
    RFIInterpVisOp,
    _TAB_LIB_INTERP,
    _TAB_LIB_INTERP_GPU,
    rfi_interp_jvp_op,
    rfi_interp_transpose_op,
    rfi_interp_vis_op,
)

if _TAB_LIB_INTERP is None:
    pytest.skip("RFI interp FFI library is not built", allow_module_level=True)

@pytest.fixture(params=[(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)])
def precision(request):
    real, complex_ = request.param
    if real == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("x64 is disabled")
    return real, complex_


def _available_devices():
    devices = jax.devices("cpu")[:1]
    if _TAB_LIB_INTERP_GPU is not None:
        for platform in ("cuda", "rocm"):
            try:
                devices += jax.devices(platform)[:1]
            except RuntimeError:
                pass
    return devices


@pytest.fixture(params=_available_devices(), ids=lambda d: d.platform)
def device(request):
    with jax.default_device(request.param):
        yield request.param


# --- the tables, as tabascal.poly_interp builds them ---------------------------

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


def make_inputs(real, complex_, seed=0, n_ant=5, n_rfi=2, n_freq=3, n_time=6,
                n_int_f=2, n_int_t=3, half_width=1, n_path=4):
    """Random data-grid inputs and the tables for a config of that shape."""
    rng = np.random.default_rng(seed)
    int_time, chan_width = 2.0, 1e4
    dt = fine_offsets(n_int_t, int_time)
    dnu = fine_offsets(n_int_f, chan_width) / 1e6  # MHz
    freqs = (1.5e8 + chan_width * np.arange(n_freq)) / 1e6  # MHz
    w_time, start_time = interp_tables(n_time, half_width, dt / int_time)
    w_freq, start_freq = interp_tables(n_freq, half_width, dnu / chan_width)

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


def make_strides(n_bl, n_int_t, seed=5):
    """A stride per baseline, 1 to the samples per cell."""
    return jnp.asarray(np.random.default_rng(seed).integers(1, n_int_t + 1, size=n_bl), dtype=jnp.int32)


def make_baselines(n_ant, shuffle=False, autocorr=True):
    lo = 0 if autocorr else 1
    pairs = np.asarray(
        [(i, j) for i in range(n_ant) for j in range(i + lo, n_ant)], dtype=np.int32
    )
    if shuffle:
        pairs = pairs[np.random.default_rng(4).permutation(len(pairs))]
    return jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])


# --- the reference -------------------------------------------------------------

def reference(amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, a1, a2, stride=None):
    """The operator in plain JAX, one time cell at a time: delays in
    microseconds, frequencies in MHz, their product in cycles. A baseline of
    stride s averages the time samples s // 2, s // 2 + s, ... only."""
    n_ant, n_rfi, n_freq, n_time = amp.shape
    n_sf, n_st = w_freq.shape[1], w_time.shape[1]
    n_int_t = len(dt)
    idx_freq = start_freq[:, None] + jnp.arange(n_sf)  # (n_freq, n_sf)
    if stride is None:
        stride = jnp.ones(len(a1), dtype=jnp.int32)
    v = jnp.arange(n_int_t)
    takes = (v[None, :] >= stride[:, None] // 2) & ((v[None, :] - stride[:, None] // 2) % stride[:, None] == 0)
    weight = takes.astype(amp.dtype) / (len(dnu) * takes.sum(axis=1, keepdims=True))  # (n_bl, n_int_t)
    cells = []
    for t in range(n_time):
        idx_time = start_time[t] + jnp.arange(n_st)
        stencil = jnp.take(amp, idx_time, axis=3)  # (n_ant, n_rfi, n_freq, n_st)
        stencil = jnp.take(stencil, idx_freq, axis=2)  # (n_ant, n_rfi, n_freq, n_sf, n_st)
        A = jnp.einsum("arfkl,fku,lv->arfuv", stencil, w_freq, w_time[t])
        d_tau = sum(
            delay[:, :, t, k, None] * dt**k / factorial(k) for k in range(1, delay.shape[-1])
        ) if delay.shape[-1] > 1 else jnp.zeros(amp.shape[:2] + dt.shape, dt.dtype)
        nu = freqs[:, None] + dnu[None, :]  # (n_freq, n_int_f), MHz
        phi = phase[..., t][..., None, None] + 2 * jnp.pi * (
            nu[None, None, :, :, None] * d_tau[:, :, None, None, :]
            + dnu[None, None, None, :, None] * delay[:, :, t, 0][:, :, None, None, None]
        )
        S = A * jnp.exp(1j * phi)
        product = S[a1] * jnp.conj(S[a2])  # (n_bl, n_rfi, n_freq, n_int_f, n_int_t)
        summed = jnp.sum(product, axis=1) * weight[:, None, None, :]
        cells.append(jnp.sum(summed, axis=(-2, -1)))
    return jnp.stack(cells, axis=-1)


def upcast(x):
    """The float64 / complex128 twin of an input, indices untouched."""
    if jnp.iscomplexobj(x):
        return x.astype(jnp.complex128)
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.float64)
    return x


def assert_close(actual, expected, real):
    """Hold a kernel to the reference, at the precision the inputs allow.

    In double the two agree to round-off. In single the kernel is held to the
    float64 reference rather than to the float32 one, which carries rounding
    of its own in different places: measured a few 1e-6 relative with the
    path relative to the array mean, as tabascal supplies it. (With the full
    path it would be 1e-3: its change across a cell is ~1e4 rad per antenna
    at orbital range rates, beyond what float32 resolves to a fraction of a
    turn -- which is why the path is relative.)
    """
    tolerance = 3e-5 if real == jnp.float32 else 5e-10
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale)


def expected_args(args, real):
    """The reference's inputs: the kernel's in float64 when the kernel is float32."""
    return [upcast(x) for x in args] if real == jnp.float32 else list(args)


# --- tests ----------------------------------------------------------------------

SHAPES = {
    "default": {},
    "one-sample": dict(n_int_f=1, n_int_t=1),
    "time-only": dict(n_int_f=1, n_int_t=7, n_freq=1, half_width=2),
    "held": dict(half_width=0),
    "wide": dict(n_freq=7, n_time=9, half_width=3, n_int_t=4),
    "large": dict(n_ant=12, n_rfi=5, n_freq=4, n_time=8, n_int_f=3, n_int_t=6),
    "linear-path": dict(n_path=2),
    "constant-path": dict(n_path=1),
}


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("shuffle", [False, True])
def test_eval_and_jit_match_reference(precision, shape, shuffle, device):
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    a1, a2 = make_baselines(args[0].shape[0], shuffle)
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    expected = reference(*expected_args(args, real), a1, a2)
    assert_close(op.eval(*args), expected, real)
    assert_close(jax.jit(op.eval)(*args), expected, real)


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_strided_baselines_match_reference(precision, shape, device):
    """Each baseline integrates every stride-th time sample, and its own mean."""
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    stride = make_strides(len(a1), len(args[8]))
    op = RFIInterpVisOp(args[0].shape[0], a1, a2, stride=stride)
    expected = reference(*expected_args(args, real), a1, a2, stride)
    assert_close(op.eval(*args), expected, real)
    amp_dot = make_inputs(real, complex_, seed=1, **shape)[0]
    ref_args = expected_args(args, real)
    _, tangent = jax.jvp(lambda a: op.eval(a, *args[1:]), (args[0],), (amp_dot,))
    _, exp_t = jax.jvp(lambda a: reference(a, *ref_args[1:], a1, a2, stride), (ref_args[0],),
                       (upcast(amp_dot) if real == jnp.float32 else amp_dot,))
    assert_close(tangent, exp_t, real)
    cot = jnp.ones((len(a1),) + args[0].shape[2:], dtype=complex_)
    _, pullback = jax.vjp(lambda a: op.eval(a, *args[1:]), args[0])
    _, ref_pullback = jax.vjp(lambda a: reference(a, *ref_args[1:], a1, a2, stride), ref_args[0])
    assert_close(pullback(cot)[0], ref_pullback(upcast(cot) if real == jnp.float32 else cot)[0], real)


def test_a_stride_that_leaves_no_sample_is_refused():
    real, complex_ = jnp.float64, jnp.complex128
    if not jax.config.jax_enable_x64:
        pytest.skip("x64 is disabled")
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    with pytest.raises(ValueError, match="at least 1"):
        RFIInterpVisOp(args[0].shape[0], a1, a2, stride=np.zeros(len(a1), dtype=np.int32))
    op = RFIInterpVisOp(args[0].shape[0], a1, a2, stride=np.full(len(a1), len(args[8]) + 1, dtype=np.int32))
    with jax.default_device(jax.devices("cpu")[0]):
        with pytest.raises(Exception, match="between 1 and"):
            jax.block_until_ready(op.eval(*args))


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_jvp_matches_reference(precision, shape, device):
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    amp_dot = make_inputs(real, complex_, seed=1, **shape)[0]
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    _, tangent = jax.jvp(lambda a: op.eval(a, *args[1:]), (args[0],), (amp_dot,))
    ref_args = expected_args(args, real)
    _, expected = jax.jvp(
        lambda a: reference(a, *ref_args[1:], a1, a2), (ref_args[0],), (upcast(amp_dot) if real == jnp.float32 else amp_dot,)
    )
    assert_close(tangent, expected, real)


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_vjp_matches_reference(precision, shape, device):
    real, complex_ = precision
    args = make_inputs(real, complex_, **shape)
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    n_bl, n_freq, n_time = len(a1), args[0].shape[2], args[0].shape[3]
    rng = np.random.default_rng(2)
    cotangent = jnp.asarray(
        rng.normal(size=(n_bl, n_freq, n_time)) + 1j * rng.normal(size=(n_bl, n_freq, n_time)),
        dtype=complex_,
    )
    _, pullback = jax.vjp(lambda a: op.eval(a, *args[1:]), args[0])
    (amp_bar,) = pullback(cotangent)
    ref_args = expected_args(args, real)
    _, ref_pullback = jax.vjp(lambda a: reference(a, *ref_args[1:], a1, a2), ref_args[0])
    (expected,) = ref_pullback(upcast(cotangent) if real == jnp.float32 else cotangent)
    assert_close(amp_bar, expected, real)


def test_the_constants_get_zero_cotangents(precision, device):
    """eval() stops the gradient on everything but the signal."""
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    vis, pullback = jax.vjp(op.eval, *args)
    bars = pullback(jnp.ones_like(vis))
    assert float(jnp.abs(bars[0]).max()) > 0
    for name, bar, x in zip(("phase", "delay_us", "w_freq"), bars[1:4], args[1:4]):
        np.testing.assert_array_equal(bar, jnp.zeros_like(x)), name


def test_the_constants_tangent_is_refused_on_a_direct_bind(precision):
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    indices = op.indices
    with pytest.raises(TypeError, match="no derivative with respect to phase"):
        jax.jvp(
            lambda p: rfi_interp_vis_op.bind(*indices, args[0], p, *args[2:]),
            (args[1],),
            (jnp.ones_like(args[1]),),
        )


def test_baselines_without_autocorrelations(precision, device):
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True, autocorr=False)
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    ref_args = expected_args(args, real)
    assert_close(op.eval(*args), reference(*ref_args, a1, a2), real)
    cotangent = jnp.ones((len(a1),) + args[0].shape[2:], dtype=complex_)
    _, pullback = jax.vjp(lambda a: op.eval(a, *args[1:]), args[0])
    _, ref_pullback = jax.vjp(lambda a: reference(a, *ref_args[1:], a1, a2), ref_args[0])
    ref_cot = upcast(cotangent) if real == jnp.float32 else cotangent
    assert_close(pullback(cotangent)[0], ref_pullback(ref_cot)[0], real)


def test_rejects_incompatible_shapes_and_dtypes(precision):
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs = args
    with pytest.raises(ValueError, match="Expected phase shape"):
        op.eval(amp, phase[..., :-1], delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs)
    with pytest.raises(ValueError, match="Expected w_time shape"):
        op.eval(amp, phase, delay, w_freq, start_freq, w_time[..., :-1], start_time, dnu, dt, freqs)
    with pytest.raises(ValueError, match="Expected delay_us shape"):
        op.eval(amp, phase, delay[:, :, :-1], w_freq, start_freq, w_time, start_time, dnu, dt, freqs)
    wrong_real = jnp.float64 if real == jnp.float32 else jnp.float32
    with pytest.raises(TypeError, match="share the phase dtype"):
        op.eval(amp, phase, delay.astype(wrong_real), w_freq, start_freq, w_time, start_time, dnu, dt, freqs)
    with pytest.raises(TypeError, match="int32 start_time"):
        op.eval(amp, phase, delay, w_freq, start_freq, w_time, start_time.astype(jnp.int64), dnu, dt, freqs)


def test_rejects_a_stencil_that_leaves_its_cell():
    """The CPU handlers check the contract the transpose's gather relies on."""
    real, complex_ = jnp.float64, jnp.complex128
    if not jax.config.jax_enable_x64:
        pytest.skip("x64 is disabled")
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    bad = list(args)
    bad[6] = bad[6].at[0].set(3)  # cell 0's stencil starts at 3: does not contain it
    with jax.default_device(jax.devices("cpu")[0]):
        with pytest.raises(Exception, match="contain the cell"):
            jax.block_until_ready(op.eval(*bad))


def test_rejects_mismatched_tangents_and_cotangents(precision):
    """The jvp and transpose primitives build views from the primal extents."""
    real, complex_ = precision
    args = make_inputs(real, complex_)
    amp_dot = make_inputs(real, complex_, seed=1)[0]
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIInterpVisOp(args[0].shape[0], a1, a2)
    indices = op.indices
    cotangent = jnp.ones((len(a1),) + args[0].shape[2:], dtype=complex_)
    with pytest.raises(ValueError, match="signal tangent"):
        rfi_interp_jvp_op.bind(*indices, args[0], amp_dot[..., :-1], *args[1:])
    with pytest.raises(ValueError, match="visibility cotangent"):
        rfi_interp_transpose_op.bind(*indices, *args, cotangent[:, :, :-1])


def test_rejects_a_duplicated_baseline():
    a1 = jnp.asarray([0, 1, 0], dtype=jnp.int32)
    a2 = jnp.asarray([1, 2, 1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="at most once"):
        RFIInterpVisOp(3, a1, a2)


def test_the_pair_table_indexes_the_baseline_list():
    a1, a2 = make_baselines(4, shuffle=True)
    op = RFIInterpVisOp(4, a1, a2)
    table = np.asarray(op.pair_index)
    for bl, (i, j) in enumerate(zip(np.asarray(a1), np.asarray(a2))):
        assert table[i, j] == bl
    assert (table >= 0).sum() == len(a1)
