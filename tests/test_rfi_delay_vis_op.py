"""Tests for the compact delay/frequency RFI visibility operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ri_kernels.jax_api.rfi_delay_vis_op import (
    RFIDelayVisOp,
    _TAB_LIB_DELAY,
    _TAB_LIB_DELAY_GPU,
)


if _TAB_LIB_DELAY is None:
    pytest.skip("RFI delay FFI library is not built", allow_module_level=True)


@pytest.fixture(params=[(jnp.float32, jnp.complex64), (jnp.float64, jnp.complex128)])
def precision(request):
    real, complex_ = request.param
    if real == jnp.float64 and not jax.config.jax_enable_x64:
        pytest.skip("x64 is disabled")
    return real, complex_


def _available_devices():
    devices = jax.devices("cpu")[:1]
    if _TAB_LIB_DELAY_GPU is not None:
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


def make_inputs(real, complex_, seed=0, max_delay_us=16.7):
    rng = np.random.default_rng(seed)
    shape = (5, 3, 2, 2, 3, 2)
    amp = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    delay = rng.uniform(
        -max_delay_us,
        max_delay_us,
        size=(shape[0], shape[2], shape[3], shape[5]),
    )
    delay -= delay.mean(axis=0, keepdims=True)
    channel = np.asarray([100.0, 500.0, 1000.0])[:, None]
    fine = np.asarray([0.0, 0.01, 0.2])[None, :]
    return (
        jnp.asarray(amp, dtype=complex_),
        jnp.asarray(delay, dtype=real),
        jnp.asarray(channel + fine, dtype=real),
    )


def make_baselines(n_ant, shuffle=False):
    pairs = np.asarray(
        [(i, j) for i in range(n_ant) for j in range(i, n_ant)], dtype=np.int32
    )
    if shuffle:
        pairs = pairs[np.random.default_rng(4).permutation(len(pairs))]
    return jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])


def reference(amp, delay_us, freq_mhz, a1, a2):
    angular_frequency = 2.0 * jnp.pi * freq_mhz
    phase_diff = angular_frequency[None, :, None, None, :, None] * (
        delay_us[a1] - delay_us[a2]
    )[:, None, :, :, None, :]
    fine = (
        amp[a1]
        * jnp.conj(amp[a2])
        * jnp.exp(1j * phase_diff)
    ).sum(axis=3)
    return fine.mean(axis=(3, 4))


def assert_close(actual, expected, real):
    tolerance = 3e-5 if real == jnp.float32 else 5e-10
    scale = max(float(np.max(np.abs(expected))), 1.0)
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance * scale
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_eval_and_jit_match_reference(precision, shuffle, device):
    real, complex_ = precision
    amp, delay, freq = make_inputs(real, complex_)
    a1, a2 = make_baselines(amp.shape[0], shuffle)
    op = RFIDelayVisOp(amp.shape[0], a1, a2)
    expected = reference(amp, delay, freq, a1, a2)
    assert_close(op.eval(amp, delay, freq), expected, real)
    assert_close(jax.jit(op.eval)(amp, delay, freq), expected, real)


def test_jvp_matches_reference(precision, device):
    real, complex_ = precision
    amp, delay, freq = make_inputs(real, complex_)
    amp_dot, delay_dot, _ = make_inputs(real, complex_, seed=1)
    a1, a2 = make_baselines(amp.shape[0], shuffle=True)
    op = RFIDelayVisOp(amp.shape[0], a1, a2)
    _, tangent = jax.jvp(
        op.eval, (amp, delay, freq), (amp_dot, delay_dot, jnp.zeros_like(freq))
    )
    _, expected = jax.jvp(
        lambda a, d: reference(a, d, freq, a1, a2),
        (amp, delay),
        (amp_dot, delay_dot),
    )
    assert_close(tangent, expected, real)


def test_vjp_matches_reference_and_frequency_is_fixed(precision, device):
    real, complex_ = precision
    amp, delay, freq = make_inputs(real, complex_)
    a1, a2 = make_baselines(amp.shape[0], shuffle=True)
    op = RFIDelayVisOp(amp.shape[0], a1, a2)
    cotangent = jnp.ones((len(a1), amp.shape[1], amp.shape[2]), dtype=complex_)
    _, pullback = jax.vjp(op.eval, amp, delay, freq)
    amp_bar, delay_bar, freq_bar = pullback(cotangent)
    _, ref_pullback = jax.vjp(
        lambda a, d: reference(a, d, freq, a1, a2), amp, delay
    )
    expected_amp, expected_delay = ref_pullback(cotangent)
    assert_close(amp_bar, expected_amp, real)
    assert_close(delay_bar, expected_delay, real)
    assert_close(freq_bar, jnp.zeros_like(freq), real)


def test_float32_accuracy_for_ten_kilometre_array():
    amp, delay, freq = make_inputs(jnp.float32, jnp.complex64)
    a1, a2 = make_baselines(amp.shape[0], shuffle=True)
    actual = RFIDelayVisOp(amp.shape[0], a1, a2).eval(amp, delay, freq)
    expected = reference(
        amp.astype(jnp.complex128),
        delay.astype(jnp.float64),
        freq.astype(jnp.float64),
        a1,
        a2,
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-2, atol=4e-2)


def test_float64_accuracy_for_exceptional_hundred_kilometre_array():
    amp, delay, freq = make_inputs(
        jnp.float64, jnp.complex128, max_delay_us=167.0
    )
    a1, a2 = make_baselines(amp.shape[0], shuffle=True)
    actual = RFIDelayVisOp(amp.shape[0], a1, a2).eval(amp, delay, freq)
    assert_close(actual, reference(amp, delay, freq, a1, a2), jnp.float64)


def test_centering_absolute_delays_in_float64_stays_within_float32_budget():
    amp, relative_delay, freq = make_inputs(jnp.float32, jnp.complex64)
    absolute_delay = relative_delay.astype(jnp.float64) + 116_747.0
    centered = absolute_delay - absolute_delay.mean(axis=0, keepdims=True)
    centered = centered.astype(jnp.float32)
    a1, a2 = make_baselines(amp.shape[0])
    op = RFIDelayVisOp(amp.shape[0], a1, a2)
    np.testing.assert_allclose(
        op.eval(amp, centered, freq),
        op.eval(amp, relative_delay, freq),
        rtol=4e-2,
        atol=4e-2,
    )


def test_frequency_grid_contains_expected_spacings():
    _, _, freq = make_inputs(jnp.float32, jnp.complex64)
    np.testing.assert_allclose(np.diff(freq[0]), [0.01, 0.19], atol=1e-5)


def test_rejects_incompatible_shapes_and_dtypes(precision):
    real, complex_ = precision
    amp, delay, freq = make_inputs(real, complex_)
    a1, a2 = make_baselines(amp.shape[0])
    op = RFIDelayVisOp(amp.shape[0], a1, a2)
    with pytest.raises(ValueError, match="Expected delay shape"):
        op.eval(amp, delay[..., :-1], freq)
    wrong_real = jnp.float64 if real == jnp.float32 else jnp.float32
    with pytest.raises(TypeError, match="frequency dtypes to match"):
        op.eval(amp, delay, freq.astype(wrong_real))
