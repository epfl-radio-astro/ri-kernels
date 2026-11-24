"""Analytic CPU/CUDA values and amplitude, phase and delay derivatives against :class:`RFIAnalyticVisOpRef`."""
from collections import namedtuple
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import (
    ANALYTIC_SHAPES as SHAPES,
    assert_analytic_close as assert_close,
    make_analytic_baselines as make_baselines,
    make_analytic_inputs as make_inputs,
    upcast,
)
from ri_kernels.jax_api.rfi_analytic_vis_op import (
    RFIAnalyticVisOp,
    _TAB_LIB_ANALYTIC,
    _TAB_LIB_ANALYTIC_CUDA,
    _TAB_LIB_ANALYTIC_ROCM,
    rfi_analytic_vis_op,
    rfi_analytic_jvp_op,
    rfi_analytic_full_jvp_op,
    rfi_analytic_full_transpose_op,
)
from ri_kernels.jax_api.rfi_analytic_vis_op_ref import RFIAnalyticVisOpRef


def _require_cpu():
    return os.environ.get("RI_KERNELS_REQUIRE_CPU", "").lower() in ("1", "true", "yes", "on")


def _devices():
    devices = []
    if _TAB_LIB_ANALYTIC is not None:
        devices.extend(jax.devices("cpu")[:1])
    elif _require_cpu():
        pytest.fail("RI_KERNELS_REQUIRE_CPU is set, but analytic CPU kernels were not loaded")
    if _TAB_LIB_ANALYTIC_CUDA is not None:
        try:
            devices.extend(jax.devices("cuda")[:1])
        except RuntimeError:
            pass
    if _TAB_LIB_ANALYTIC_ROCM is not None:
        try:
            devices.extend(jax.devices("rocm")[:1])
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


# --- the reference and the operator, each compiled once per configuration -----
#
# Compiling the reference and differentiating it dominates the run time. Every
# test therefore goes through the same two jitted entry points, which return
# the value and all derivatives at once. The reference always runs on the CPU
# in float64, so both precisions and every device share one compilation per
# shape, baseline list and option set.

CPU = jax.devices("cpu")[0]
DEFAULT_OPTIONS = dict(segments=2, terms=6, cubic_terms=3)

# The value; the JVPs of the signal, the phase and the delay alone and of all
# three; the signal's cotangent from the signal-only pullback, the signal's and
# the phase's from the pullback of the two, and all three from the full one.
Results = namedtuple(
    "Results",
    "value amp_jvp phase_jvp delay_jvp full_jvp amp_bar phase_bar full_amp_bar full_phase_bar delay_bar",
)


@partial(jax.jit, static_argnames=("a1", "a2", "segments", "terms", "cubic_terms"))
def _expected(args, amp_dot, phase_dot, delay_dot, g, *, a1, a2, segments, terms, cubic_terms):
    # The reference validates its baselines on the host, so they come in static.
    ref = RFIAnalyticVisOpRef(args[0].shape[0], np.asarray(a1, np.int32), np.asarray(a2, np.int32))
    fn = lambda amp, phase, delay: ref.eval(amp, phase, delay, *args[3:], segments=segments,
                                            terms=terms, cubic_terms=cubic_terms)
    primals, zero = tuple(args[:3]), jnp.zeros_like
    value, full_jvp = jax.jvp(fn, primals, (amp_dot, phase_dot, delay_dot))
    amp_jvp = jax.jvp(fn, primals, (amp_dot, zero(phase_dot), zero(delay_dot)))[1]
    delay_jvp = jax.jvp(fn, primals, (zero(amp_dot), zero(phase_dot), delay_dot))[1]
    amp_bar, phase_bar, delay_bar = jax.vjp(fn, *primals)[1](g)
    return Results(value, amp_jvp, full_jvp - amp_jvp - delay_jvp, delay_jvp, full_jvp,
                   amp_bar, phase_bar, amp_bar, phase_bar, delay_bar)


def expected(args, a1, a2, amp_dot, phase_dot, delay_dot, g, **options):
    """The reference's Results, in float64 on the CPU."""
    with jax.default_device(CPU):
        put = lambda x: jax.device_put(upcast(x), CPU)
        return _expected(tuple(map(put, args)), put(amp_dot), put(phase_dot), put(delay_dot), put(g),
                         a1=tuple(np.asarray(a1).tolist()), a2=tuple(np.asarray(a2).tolist()),
                         **{**DEFAULT_OPTIONS, **options})


def kernel_results(op, args, amp_dot, phase_dot, delay_dot, g, **options):
    """The operator's Results: every handler, with the value checked eagerly too."""
    options = {**DEFAULT_OPTIONS, **options}
    rest = args[3:]
    fn = lambda a, p, d: op.eval(a, p, d, *rest, **options)

    @jax.jit
    def run(a, p, d, da, dp, dd, g):
        value, full_jvp = jax.jvp(fn, (a, p, d), (da, dp, dd))
        # Symbolic-zero phase and delay tangents bind the signal-only JVP;
        # anything else the full one, which forms the delay's derivative only
        # for a delay tangent.
        amp_jvp = jax.jvp(lambda x: fn(x, p, d), (a,), (da,))[1]
        phase_jvp = jax.jvp(lambda y: fn(a, y, d), (p,), (dp,))[1]
        delay_jvp = jax.jvp(lambda z: fn(a, p, z), (d,), (dd,))[1]
        amp_bar = jax.vjp(lambda x: fn(x, p, d), a)[1](g)[0]
        amp_phase_bar = jax.vjp(lambda x, y: fn(x, y, d), a, p)[1](g)
        full_bar = jax.vjp(fn, a, p, d)[1](g)
        return Results(value, amp_jvp, phase_jvp, delay_jvp, full_jvp, amp_bar,
                       amp_phase_bar[1], *full_bar)

    results = run(*args[:3], amp_dot, phase_dot, delay_dot, g)
    assert_close(fn(*args[:3]), results.value, args[1].dtype)
    return results


# What a delay tangent of one moves the angles by differs by orders of
# magnitude between the orders; these bring each order's share of the full
# JVP near the phase's.
DELAY_TANGENT_SCALES = np.array([10., 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])


def perturbations(args, n_bl, seed=1):
    """Random signal, phase and delay tangents and a visibility cotangent."""
    rng = np.random.default_rng(seed)
    amp, phase, delay = args[:3]
    out = (n_bl,) + amp.shape[2:]
    amp_dot = rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape)
    g = rng.normal(size=out) + 1j * rng.normal(size=out)
    delay_dot = rng.normal(size=delay.shape) * DELAY_TANGENT_SCALES[:delay.shape[-1]]
    return (jnp.asarray(amp_dot, amp.dtype), jnp.asarray(rng.normal(size=phase.shape), phase.dtype),
            jnp.asarray(delay_dot, delay.dtype), jnp.asarray(g, amp.dtype))


def check(op, args, a1, a2, seed=1, **options):
    """Compare every handler with the reference and return the kernel's Results."""
    real = args[1].dtype
    perturbation = perturbations(args, len(a1), seed)
    actual = kernel_results(op, args, *perturbation, **options)
    reference_results = expected(args, a1, a2, *perturbation, **options)
    for name, value, ref in zip(Results._fields, actual, reference_results):
        try:
            assert_close(value, ref, real)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None
    assert actual.value.dtype == args[0].dtype
    assert actual.phase_bar.dtype == actual.delay_bar.dtype == real
    # The pairing is real and bilinear under JAX's complex cotangent convention.
    amp_dot, phase_dot, delay_dot, g = perturbation
    assert_close(jnp.real(jnp.sum(g * actual.full_jvp)),
                 jnp.real(jnp.sum(actual.full_amp_bar * amp_dot)) +
                 jnp.sum(actual.full_phase_bar * phase_dot) + jnp.sum(actual.delay_bar * delay_dot),
                 real)
    return actual


# Every shape with the default options, which take the specialised kernels
# wherever the stencil has three coefficients. The conservative and the
# cubic-free configurations take the general path on a few shapes each.
CASES = [(name, {}) for name in SHAPES] + [
    ("default", dict(segments=4, terms=16, cubic_terms=3)),
    ("wide", dict(segments=4, terms=16, cubic_terms=3)),
    ("default", dict(segments=1, terms=6, cubic_terms=0)),
    ("held", dict(segments=1, terms=6, cubic_terms=0)),
]


@pytest.mark.parametrize("shape,options", CASES,
                         ids=[f"{name}-{'-'.join(map(str, o.values())) or 'default'}" for name, o in CASES])
def test_value_and_derivatives(precision, device, shape, options):
    real, complex_ = precision
    args = make_inputs(real, complex_, **SHAPES[shape])
    a1, a2 = make_baselines(args[0].shape[0], shuffle=True)
    check(RFIAnalyticVisOp(args[0].shape[0], a1, a2), args, a1, a2, **options)


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
    check(RFIAnalyticVisOp(2, a1, a2), args, a1, a2)


@pytest.mark.parametrize("n_ant", [1, 33, 65])
def test_sparse_reversed_and_auto_baselines(precision, device, n_ant):
    # Tile edges, reversed orderings and autocorrelations, whose phase
    # cotangent contributions cancel.
    real, complex_ = precision
    pairs = sorted(set([(0, 0), (n_ant - 1, n_ant - 1), (0, n_ant - 1), (n_ant - 1, 0),
                        (min(31, n_ant - 1), min(32, n_ant - 1))]))[::-1]
    a1, a2 = (jnp.asarray(x, jnp.int32) for x in np.asarray(pairs).T)
    args = make_inputs(real, complex_, n_ant=n_ant, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    check(RFIAnalyticVisOp(n_ant, a1, a2), args, a1, a2)


def test_constants_and_zero_tangent(precision, device):
    """eval() stops the gradient on the interval and the tables; the signal,
    the phase and the delay get cotangents."""
    real, complex_ = precision
    args = make_inputs(real, complex_)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIAnalyticVisOp(args[0].shape[0], a1, a2)
    value, pullback = jax.vjp(op.eval, *args)
    bars = pullback(jnp.ones_like(value))
    for i in [0, 1, 2]:
        assert np.max(np.abs(bars[i])) > 0
    for i in [3, 5, 7, 8, 9]:
        np.testing.assert_array_equal(bars[i], jnp.zeros_like(args[i]))
    _, dot = jax.jvp(lambda w: op.eval(*args[:3], w, *args[4:]), (args[3],), (jnp.ones_like(args[3]),))
    np.testing.assert_array_equal(dot, jnp.zeros_like(value))


def test_the_signal_only_jvp_serves_a_fixed_phase_and_delay():
    """A fixed phase and delay bind the signal-only JVP, a differentiated
    phase or delay the full JVP; every VJP binds the one transpose, which
    forms the delay's derivative only when the delay is differentiated."""
    args = make_inputs(jnp.float32, jnp.complex64)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIAnalyticVisOp(args[0].shape[0], a1, a2)
    cot = jnp.ones((len(a1),) + args[0].shape[2:], dtype=jnp.complex64)
    names = ("rfi_analytic_jvp_op", "rfi_analytic_full_jvp_op", "rfi_analytic_full_transpose_op")

    def primitives(text):
        return {name for name in names if name in text}, "delay_grad=True" in text

    def backward(fn, *primals):
        _, pullback = jax.vjp(fn, *primals)
        return primitives(str(jax.make_jaxpr(pullback)(cot)))

    def forward(fn, primal):
        return primitives(str(jax.make_jaxpr(lambda x, t: jax.jvp(fn, (x,), (t,)))(primal, primal)))

    transpose = {"rfi_analytic_full_transpose_op"}
    assert backward(lambda a: op.eval(a, *args[1:]), args[0]) == (transpose, False)
    assert backward(lambda a, p: op.eval(a, p, *args[2:]), args[0], args[1]) == (transpose, False)
    assert backward(lambda p: op.eval(args[0], p, *args[2:]), args[1]) == (transpose, False)
    assert backward(lambda d: op.eval(*args[:2], d, *args[3:]), args[2]) == (transpose, True)
    assert backward(op.eval, *args) == (transpose, True)
    assert forward(lambda x: op.eval(x, *args[1:]), args[0]) == ({"rfi_analytic_jvp_op"}, False)
    assert forward(lambda x: op.eval(args[0], x, *args[2:]), args[1]) == ({"rfi_analytic_full_jvp_op"}, False)
    assert forward(lambda x: op.eval(*args[:2], x, *args[3:]), args[2]) == ({"rfi_analytic_full_jvp_op"}, True)


def test_the_delay_cotangent_is_formed_only_when_read():
    """A delay tangent that is a constant of the transposed map, or a delay
    cotangent that jit finds unused, leaves the delay's derivative out."""
    args = make_inputs(jnp.float32, jnp.complex64)
    a1, a2 = make_baselines(args[0].shape[0])
    op = RFIAnalyticVisOp(args[0].shape[0], a1, a2)
    fn = lambda a, p, d: op.eval(a, p, d, *args[3:])

    def lin(t):
        # Concrete zero tangents for the phase and the delay: the full JVP.
        zeros = (jnp.zeros_like(args[1]), jnp.zeros_like(args[2]))
        return jax.jvp(fn, tuple(args[:3]), (t, *zeros))[1]

    assert "delay_grad=True" in str(jax.make_jaxpr(lin)(args[0]))
    cot = jnp.ones(lin(args[0]).shape, dtype=jnp.complex64)
    assert "delay_grad=True" not in str(jax.make_jaxpr(jax.linear_transpose(lin, args[0]))(cot))

    loss = lambda *x: jnp.sum(jnp.abs(fn(*x)) ** 2)
    every = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    kept = jax.jit(lambda *x: jax.grad(loss, argnums=(0, 1, 2))(*x)[:2])
    assert "delay_grad = 1" in every.lower(*args[:3]).as_text()
    assert "delay_grad = 0" in kept.lower(*args[:3]).as_text()
    for x, y in zip(kept(*args[:3]), every(*args[:3])):
        np.testing.assert_array_equal(x, y)


def test_cubic_translation_matters(device):
    args = make_inputs(jnp.float64, jnp.complex128, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    args[2] = jnp.zeros_like(args[2]).at[0, :, :, 3].set(.0005)
    a1, a2 = jnp.array([0], jnp.int32), jnp.array([1], jnp.int32)
    op = RFIAnalyticVisOp(2, a1, a2)
    full = check(op, args, a1, a2).value
    omitted = op.eval(*args, cubic_terms=0)
    assert np.max(np.abs(full - omitted)) > 1e-4



@pytest.mark.parametrize("slot,value,match", [
    (5, jnp.zeros((6, 3, 10)), "monomial"),
    (8, jnp.ones(2), "scalar"), (7, jnp.zeros((2, 1)), "dnu|w_freq"),
    (6, jnp.zeros(6, jnp.int64), "int32"),
])
def test_bad_shapes(slot, value, match):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[slot] = value
    a1, a2 = make_baselines(5)
    with pytest.raises((TypeError, ValueError), match=match):
        RFIAnalyticVisOp(5, a1, a2).eval(*args)


@pytest.mark.parametrize("options", [dict(segments=0), dict(terms=33), dict(cubic_terms=-1), dict(terms=2.5),
                                     dict(scratch_mb=0), dict(scratch_mb=2**43)])
def test_bad_options(options):
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    with pytest.raises(ValueError, match="static integer"):
        RFIAnalyticVisOp(5, a1, a2).eval(*args, **options)


def test_direct_bind_derivatives_and_shapes():
    args = make_inputs(jnp.float64, jnp.complex128)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    options = dict(segments=2, terms=6, cubic_terms=3, scratch_mb=256)
    with pytest.raises(TypeError, match="amp, phase and delay_us only"):
        jax.jvp(lambda w: rfi_analytic_vis_op.bind(*op.indices, *args[:3], w, *args[4:], **options),
                (args[3],), (jnp.ones_like(args[3]),))
    with pytest.raises(ValueError, match="signal tangent"):
        rfi_analytic_jvp_op.bind(*op.indices, args[0], args[0][:-1], *args[1:], **options)
    amp, phase, delay = args[:3]
    with pytest.raises(ValueError, match="phase tangent"):
        rfi_analytic_full_jvp_op.bind(*op.indices, amp, amp, phase, phase[:-1], delay, delay, *args[3:],
                                      delay_grad=False, **options)
    with pytest.raises(ValueError, match="delay tangent"):
        rfi_analytic_full_jvp_op.bind(*op.indices, amp, amp, phase, phase, delay, delay[:-1], *args[3:],
                                      delay_grad=True, **options)
    with pytest.raises(ValueError, match="visibility cotangent"):
        rfi_analytic_full_transpose_op.bind(*op.indices, *args, jnp.ones((1, 1, 1), jnp.complex128),
                                            delay_grad=True, **options)


def test_invalid_cpu_values(device):
    if device.platform != "cpu":
        pytest.skip("GPU stencil values are caller-validated")
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
    value, pullback = jax.vjp(lambda a, d: op.eval(a, args[1], d, *args[3:]), args[0], args[2])
    assert value.shape == (0, 3, 6)
    amp_bar, delay_bar = pullback(jnp.zeros_like(value))
    np.testing.assert_array_equal(amp_bar, jnp.zeros_like(args[0]))
    np.testing.assert_array_equal(delay_bar, jnp.zeros_like(args[2]))


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
    check(RFIAnalyticVisOp(2, a1, a2), args, a1, a2, **options)


@pytest.mark.parametrize("slot", [1, 2, 3, 5, 7, 8, 9])
def test_mixed_real_precision(slot):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[slot] = args[slot].astype(jnp.float32)
    a1, a2 = make_baselines(5)
    with pytest.raises(TypeError, match="dtype|precision"):
        RFIAnalyticVisOp(5, a1, a2).eval(*args)


@pytest.mark.parametrize("real_name", ["float32", "float64"])
def test_gpu_scratch_chunks(real_name, device):
    if device.platform == "cpu":
        pytest.skip("Scratch chunking is a GPU path")
    # This shape forces a frequency split at 1 MiB even for complex64. The
    # final chunks are short on both axes, and edge stencils cross chunks.
    real = getattr(jnp, real_name)
    complex_ = jnp.complex64 if real == jnp.float32 else jnp.complex128
    args = make_inputs(real, complex_, n_ant=65, n_rfi=2, n_freq=13, n_time=5, n_int_f=9)
    a1 = jnp.array([0, 64, 31, 32, 0, 32, 64], jnp.int32)
    a2 = jnp.array([64, 0, 32, 31, 0, 32, 64], jnp.int32)
    op = RFIAnalyticVisOp(65, a1, a2)
    whole = check(op, args, a1, a2)
    split = kernel_results(op, args, *perturbations(args, len(a1)), scratch_mb=1)
    for name, a, b in zip(Results._fields, whole, split):
        try:
            assert_close(a, b, real)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None


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
    options = dict(segments=np.int64(2), terms=np.int64(6), cubic_terms=np.int64(3),
                   scratch_mb=np.int64(256))
    options[option] = np.int64(value)
    target = "ri_rfi_analytic_vis_fwd_" + ("cpu" if device.platform == "cpu" else "gpu") + "_f64"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((len(a1), 3, 6), jnp.complex128))
    with pytest.raises(Exception, match="capacity"):
        call(*op.indices, *args, **options).block_until_ready()


def test_ffi_rejects_excess_coefficients(device):
    args = make_inputs(jnp.float64, jnp.complex128)
    args[5] = jnp.zeros((6, 3, 10), jnp.float64)
    a1, a2 = make_baselines(5)
    op = RFIAnalyticVisOp(5, a1, a2)
    target = "ri_rfi_analytic_vis_fwd_" + ("cpu" if device.platform == "cpu" else "gpu") + "_f64"
    call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct((len(a1), 3, 6), jnp.complex128))
    with pytest.raises(Exception, match="capacity"):
        call(*op.indices, *args, segments=np.int64(2), terms=np.int64(6),
             cubic_terms=np.int64(3), scratch_mb=np.int64(256)).block_until_ready()


def test_subnormal_winding_no_nan(device):
    """A nonzero winding that rounds to 0.0f takes the a = 0 limit, not 0/0.

    A cell's winding is a = 2*pi * nu * (int_time / 2) * delta_rate / segments,
    formed in double. The GPU kernels run the recurrence in the operator's
    precision, so in float32 a winding below the smallest subnormal becomes
    0.0f while a double ``a == 0`` guard still sees it as nonzero. The CPU
    kernels always recurse in double, so there this only checks that nothing
    changes. Both inputs are normal float32 values, so the test does not
    depend on the GPU flushing subnormals to zero.
    """
    tiny = 1e-25
    args = make_inputs(jnp.float32, jnp.complex64, n_ant=2, n_rfi=1, n_freq=1, n_time=3, n_int_f=1)
    args[9] = jnp.full_like(args[9], tiny)  # freq_mhz; n_int_f=1 makes dnu zero
    zero_delay = jnp.zeros_like(args[2])
    still = list(args)
    still[2] = zero_delay
    args[2] = zero_delay.at[0, :, :, 1].set(tiny)  # only antenna 0 has a delay rate

    # More segments only shrink a, so segments=1 bounds it from above.
    winding = 2 * np.pi * tiny * (float(args[8]) / 2) * tiny
    assert winding > 0 and np.float32(winding) == 0

    op = RFIAnalyticVisOp(2, jnp.array([0], jnp.int32), jnp.array([1], jnp.int32))
    vis = op.eval(*args)
    assert np.all(np.isfinite(np.asarray(vis)))
    assert_close(vis, op.eval(*still), jnp.float32)


def test_multi_device_cpu_pmap():
    """Exercise async parallel_for Futures across multiple CPU devices."""
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) < 2 or _TAB_LIB_ANALYTIC is None:
        pytest.skip("requires >= 2 CPU devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    n_dev = len(cpu_devices)
    a1, a2 = make_baselines(5, shuffle=True)
    op = RFIAnalyticVisOp(5, a1, a2)
    base = make_inputs(jnp.float64, jnp.complex128)
    amps = jnp.stack([make_inputs(jnp.float64, jnp.complex128, seed=i)[0] for i in range(n_dev)])
    rest = base[1:]

    def loss_and_grad(a):
        return jax.value_and_grad(lambda x: jnp.sum(jnp.abs(op.eval(x, *rest)) ** 2))(a)

    vals, grads = jax.pmap(loss_and_grad)(amps)
    for i in range(n_dev):
        ref_val, ref_grad = loss_and_grad(amps[i])
        assert_close(vals[i], ref_val, jnp.float64)
        assert_close(grads[i], ref_grad, jnp.float64)
