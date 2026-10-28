"""Pure JAX reference for :class:`RFIAnalyticVisOp`.

The same closed form as the compiled kernels, antenna first and polarised:
each cell interpolates the signal's two-column factor to the fine channels as
polynomials in x = 2*tau/int_time, multiplies a baseline's pair of
polynomials into the P x P product, and integrates it against the baseline's
delay phase ``phi0 + a x + b x^2 + c x^3`` on equal pieces of [-1, 1]. The
quadratic phase is integrated analytically, the residual cubic as a series.
The phase moments depend on the baseline alone, so every matrix entry shares
them.

JAX differentiates the signal, the phase and the delay directly; the
integration interval and the tables are constants, as in the compiled op. It
needs no FFI library and holds one time cell of every baseline at once, so it
is meant for checking the kernels and for small problems.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import fresnel


class RFIAnalyticVisOpRef:
    """Pure JAX counterpart of :class:`RFIAnalyticVisOp`, with the same interface."""

    def __init__(self, n_ant, a1, a2):
        a1, a2 = np.asarray(a1), np.asarray(a2)
        if (a1.ndim != 1 or a2.shape != a1.shape or
                a1.dtype.kind not in "iu" or a2.dtype.kind not in "iu" or
                n_ant < 1 or np.any(a1 < 0) or np.any(a2 < 0) or
                np.any(a1 >= n_ant) or np.any(a2 >= n_ant)):
            raise ValueError("Expected matching antenna-index vectors within [0, n_ant)")
        self.n_ant = n_ant
        self.a1 = jnp.asarray(a1, jnp.int32)
        self.a2 = jnp.asarray(a2, jnp.int32)

    def eval(self, amp, phase, delay_us, w_freq, start_freq, g_time,
             start_time, dnu_mhz, int_time, freq_mhz, *, segments=2, terms=6,
             cubic_terms=3, scratch_mb=None):
        """Return complex (n_bl, n_freq, n_time, P, P) visibilities.

        The arguments and options are those of :meth:`RFIAnalyticVisOp.eval`;
        scratch_mb is accepted for compatibility and ignored. The kernels'
        limits on coefficient counts and series lengths do not apply here.
        """
        # Like the compiled op, accept NumPy arrays: they cannot be indexed by
        # the traced cell index.
        amp, phase, delay_us, w_freq, start_freq, g_time, start_time, dnu_mhz, freq_mhz = map(
            jnp.asarray, (amp, phase, delay_us, w_freq, start_freq, g_time, start_time, dnu_mhz, freq_mhz)
        )
        for name, value, lo in (("segments", segments, 1), ("terms", terms, 1),
                                ("cubic_terms", cubic_terms, 0)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lo:
                raise ValueError(f"{name} must be a static integer >= {lo}")
        if amp.ndim != 6 or amp.shape[4] not in (1, 2) or amp.shape[5] != 2:
            hint = " (pad a scalar signal to (..., 1, 2) with a zero second column)" if amp.ndim == 4 else ""
            raise ValueError(
                f"Expected a signal of shape (n_ant, n_rfi, n_freq, n_time, P, 2) with P 1 or 2; "
                f"got {amp.shape}{hint}"
            )
        if amp.shape[0] != self.n_ant:
            raise ValueError(f"Expected {self.n_ant} antennas; got a signal of shape {amp.shape}")
        if phase.shape != amp.shape[:4]:
            raise ValueError(f"Expected phase shape {amp.shape[:4]}; got {phase.shape}")
        if delay_us.ndim != 4 or delay_us.shape[:3] != (*amp.shape[:2], amp.shape[3]) or delay_us.shape[3] < 1:
            raise ValueError(
                f"Expected delay_us shape {(*amp.shape[:2], amp.shape[3])} + (n_path,); got {delay_us.shape}"
            )
        stop = jax.lax.stop_gradient
        int_time = jnp.asarray(int_time, dtype=None if hasattr(int_time, "dtype") else phase.dtype)
        return _analytic_vis(
            amp, phase, delay_us, stop(w_freq), start_freq, stop(g_time), start_time,
            stop(dnu_mhz), stop(int_time), stop(freq_mhz), self.a1, self.a2,
            segments=int(segments), terms=int(terms), cubic_terms=int(cubic_terms),
        )


def _fine_signal(amp, w_freq, start_freq, g_time, start_time):
    """The signal of one time cell at every fine channel, as polynomials in x.

    ``g_time`` ``(n_st, n_coef)`` and ``start_time`` (a scalar) are the cell's
    own rows of the time tables; the frequency tables are whole, since every
    channel of the cell is interpolated at once.

    Returns ``(n_ant, n_rfi, n_freq, n_int_freq, P, 2, n_coef)``.
    """
    n_sf, n_st = w_freq.shape[1], g_time.shape[0]
    # The stencil: the n_st cells around this one, then the n_sf channels
    # around each channel.
    idx_time = start_time + jnp.arange(n_st)  # (n_st,)
    idx_freq = start_freq[:, None] + jnp.arange(n_sf)  # (n_freq, n_sf)
    stencil = jnp.take(amp, idx_time, axis=3)  # (n_ant, n_rfi, n_freq, n_st, P, 2)
    stencil = jnp.take(stencil, idx_freq, axis=2)  # (n_ant, n_rfi, n_freq, n_sf, n_st, P, 2)
    # Separable weights: one table per axis.
    return jnp.einsum("arfklic,fku,lm->arfuicm", stencil, w_freq.astype(amp.dtype),
                      g_time.astype(amp.dtype), precision=lax.Precision.HIGHEST)


def _linear_phase_moments(a, degree):
    """The moments ``mean_{[-1,1]} x**m exp(i*a*x)``, through ``degree``.

    Integration by parts divides by the winding, so upward recurrence is
    stable only while m <= |a|. Above that point the same identity runs
    downwards from a zero tail, 64 orders beyond the last requested moment,
    where the unwanted solution contracts by |a|/m at every step. This also
    supplies the zero-frequency limit without a division by zero.
    """
    a = jnp.asarray(a)
    positive, negative = jnp.exp(1j * a), jnp.exp(-1j * a)
    safe_a = jnp.where(jnp.abs(a) >= 1, a, 1)
    zero = jnp.sinc(a / jnp.pi).astype(positive.dtype)

    def upward_step(last, m):
        sign = jnp.where(m % 2 == 0, 1, -1).astype(a.dtype)
        value = ((positive - sign * negative) / 2 - m.astype(a.dtype) * last) / (1j * safe_a)
        value = jnp.where(m <= jnp.abs(a), value, 0)
        return value, value

    _, upward = lax.scan(upward_step, zero, jnp.arange(1, degree + 1, dtype=jnp.int32))
    upward = jnp.concatenate((zero[..., None], jnp.moveaxis(upward, 0, -1)), axis=-1)

    # Large a uses the upward result throughout. Giving the unused downward
    # branch a benign argument keeps masked overflows out of differentiation.
    down_a = jnp.where(jnp.abs(a) <= degree, a, 0)
    ep, em = jnp.exp(1j * down_a), jnp.exp(-1j * down_a)

    def downward_step(last, m):
        sign = jnp.where(m % 2 == 0, 1, -1).astype(a.dtype)
        previous = ((ep - sign * em) / 2 - 1j * down_a * last) / m.astype(a.dtype)
        return previous, previous

    # The scan runs m = degree+64 .. 1; reversed, output k holds order m = k+1
    # and the requested moments are the first degree+1 of them.
    _, downward = lax.scan(
        downward_step, jnp.zeros_like(positive), jnp.arange(degree + 64, 0, -1, dtype=jnp.int32)
    )
    downward = jnp.moveaxis(downward[::-1][:degree + 1], 0, -1)
    return jnp.where(jnp.arange(degree + 1) <= jnp.abs(a)[..., None], upward, downward)


def _quadratic_phase_moments(a, b, degree, terms):
    """Normalised moments on [-1, 1] of ``exp(i*(a*x+b*x*x))``, through ``degree``.

    Away from the stationary point, expand the curvature about linear-phase
    moments in ``terms`` terms. Near it, the Fresnel seed and its upward
    recurrence are stable, except at small b where division by b is itself
    ill-conditioned; there the same convergent series supplies the limit.
    """
    a, b = jnp.broadcast_arrays(a, b)
    stationary = (jnp.abs(a) <= 2.5 * jnp.abs(b)) & (jnp.abs(b) >= 1)
    series_a, series_b = jnp.where(stationary, 0., a), jnp.where(stationary, 0., b)
    linear = _linear_phase_moments(series_a, degree + 2 * (terms - 1))

    def series_step(carry, k):
        series, coefficient = carry
        window = lax.dynamic_slice_in_dim(linear, 2 * k, degree + 1, axis=-1)
        series = series + coefficient[..., None] * window
        coefficient = coefficient * (1j * series_b) / (k + 1).astype(series_a.dtype)
        return (series, coefficient), None

    init = (jnp.zeros_like(linear[..., :degree + 1]), jnp.ones_like(series_a, dtype=linear.dtype))
    (series, _), _ = lax.scan(series_step, init, jnp.arange(terms, dtype=jnp.int32))

    # Complete the square only near the stationary point: doing so at large
    # winding subtracts almost equal Fresnel values with huge phase arguments.
    fa, fb = jnp.where(stationary, a, 0.), jnp.where(stationary, b, 1.)
    scale = jnp.sqrt(2 * jnp.abs(fb) / jnp.pi)
    shift = fa / (2 * fb)
    sp, cp = fresnel(scale * (1 + shift))
    sm, cm = fresnel(scale * (-1 + shift))
    zero = jnp.exp(-1j * fa * shift / 2) * ((cp - cm) + 1j * jnp.sign(fb) * (sp - sm)) / (2 * scale)
    ep, em = jnp.exp(1j * (fa + fb)), jnp.exp(1j * (-fa + fb))

    def moment_step(carry, m):
        before_last, last = carry
        previous = (m - 1).astype(a.dtype) * before_last
        sign = jnp.where((m - 1) % 2 == 0, 1, -1).astype(a.dtype)
        value = ((ep - sign * em) / 2 - previous - 1j * fa * last) / (2j * fb)
        return (last, value), value

    _, moments = lax.scan(moment_step, (jnp.zeros_like(zero), zero), jnp.arange(1, degree + 1, dtype=jnp.int32))
    moments = jnp.concatenate((zero[..., None], jnp.moveaxis(moments, 0, -1)), axis=-1)
    return jnp.where(stationary[..., None], moments, series)


def _analytic_vis(amp, phase, delay, w_freq, start_freq, g_time, start_time,
                  dnu_mhz, int_time, freqs_mhz, a1, a2, *, segments, terms, cubic_terms):
    """The visibilities (n_bl, n_freq, n_time, P, P), one time cell at a time."""
    amp_degree = g_time.shape[-1] - 1
    degree = 2 * amp_degree
    n_cubic = max(1, cubic_terms)
    moment_degree = degree + 3 * (n_cubic - 1)
    # Type every real constant, including the host tables, so that x64 does
    # not widen a single-precision integration.
    real_dtype = amp.real.dtype
    host_radius = np.asarray(1. / segments, dtype=real_dtype)
    radius = jnp.asarray(host_radius)
    int_time = jnp.asarray(int_time, dtype=real_dtype)
    nu = freqs_mhz[:, None] + dnu_mhz[None, :]  # (n_freq, n_int_freq)
    orders = jnp.arange(degree + 1, dtype=jnp.int32)
    binomial = np.asarray(
        [[math.comb(m, j) if j <= m else 0 for j in range(degree + 1)] for m in range(degree + 1)],
        dtype=real_dtype,
    )
    radius_powers = np.power(host_radius, np.arange(degree + 1), dtype=real_dtype)
    n_path = delay.shape[-1]

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def one_cell(t):
        # (n_bl, n_rfi, n_freq, n_int_freq, P, 2, n_coef)
        amplitude = _fine_signal(amp, w_freq, start_freq, g_time[t], start_time[t])
        p, q = amplitude[a1], amplitude[a2].conj()
        # The product's polynomial: each row of shifted_q holds q[m-k], with
        # zeros outside its polynomial, so all output degrees contract at once.
        indices = orders[:, None] - jnp.arange(amp_degree + 1)
        shifted_q = jnp.where(
            (indices >= 0) & (indices <= amp_degree),
            q[..., jnp.clip(indices, 0, amp_degree)], 0,
        )
        # (n_bl, n_rfi, n_freq, n_int_freq, P, P, degree + 1)
        product = jnp.einsum("...ick,...jcmk->...ijm", p, shifted_q,
                             precision=lax.Precision.HIGHEST)

        # The baseline's phase, shared by every matrix entry: (n_bl, n_rfi, n_freq, n_int_freq).
        delay_cell = delay[:, :, t]  # (n_ant, n_rfi, n_path)
        bl_delay = delay_cell[a1] - delay_cell[a2]
        phi = phase[..., t][a1] - phase[..., t][a2]  # (n_bl, n_rfi, n_freq)
        phi0 = phi[..., None] + 2 * jnp.pi * dnu_mhz * bl_delay[..., 0, None, None]
        zero = jnp.zeros_like(bl_delay[..., 0])
        d1 = bl_delay[..., 1] if n_path > 1 else zero
        d2 = bl_delay[..., 2] if n_path > 2 else zero
        d3 = bl_delay[..., 3] if n_path > 3 and cubic_terms else zero
        a = 2 * jnp.pi * nu * d1[..., None, None] * (int_time / 2)
        b = jnp.pi * nu * d2[..., None, None] * (int_time / 2)**2
        c = (jnp.pi / 3) * nu * d3[..., None, None] * (int_time / 2)**3

        def piece(total, i):
            centre = -1 + (2 * i.astype(real_dtype) + 1) * radius
            # Translate the polynomial to x = centre + radius*y, summing each
            # coefficient from left to right in the source degree.
            centre_powers = (centre**orders).astype(real_dtype)
            combinations = jnp.asarray(binomial)
            radii = jnp.asarray(radius_powers)

            def translate(shifted, m):
                power = centre_powers[jnp.maximum(m - orders, 0)]
                value = combinations[m] * power * radii * product[..., m, None]
                return jnp.where(orders <= m, shifted + value, shifted), None

            shifted, _ = lax.scan(translate, jnp.zeros_like(product), orders)
            moments = _quadratic_phase_moments(
                (a + 2 * b * centre + 3 * c * centre**2) * radius,
                (b + 3 * c * centre) * radius**2, moment_degree, terms,
            )
            # Only the residual c*r^3*y^3 of the translated cubic needs
            # expansion; its lower parts are already in the phase and moments.
            coefficient = jnp.ones_like(a, dtype=product.dtype)
            integral = jnp.zeros_like(product[..., 0])

            def cubic_step(carry, k):
                integral, coefficient = carry
                window = lax.dynamic_slice_in_dim(moments, 3 * k, degree + 1, axis=-1)
                weighted = jnp.sum(shifted * window[..., None, None, :], axis=-1)
                integral = integral + coefficient[..., None, None] * weighted
                coefficient = coefficient * (1j * c * radius**3) / (k + 1).astype(real_dtype)
                return (integral, coefficient), None

            (integral, _), _ = lax.scan(
                cubic_step, (integral, coefficient), jnp.arange(n_cubic, dtype=jnp.int32)
            )
            angle = phi0 + a * centre + b * centre**2 + c * centre**3
            return total + jnp.exp(1j * angle)[..., None, None] * integral / segments, None

        result, _ = lax.scan(
            piece, jnp.zeros(product.shape[:-1], dtype=product.dtype),
            jnp.arange(segments, dtype=jnp.int32),
        )
        # Average the fine channels, sum the sources: (n_bl, n_freq, P, P).
        return jnp.sum(jnp.mean(result, axis=3), axis=1)

    return jnp.moveaxis(lax.map(one_cell, jnp.arange(amp.shape[3])), 0, 2)
