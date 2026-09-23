"""Frozen analytic reference from the tabascal operator boundary.

Keep the moment branches, translation and summation order here independent of
compiled helpers: both kernel precisions are checked against this in float64.
The public operator uses antenna first; this reference retains source first.
"""
import functools
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array, lax

def fine_signal(
    rfi_A: Array, w_freq: Array, start_freq: Array, w_time: Array, start_time: Array
) -> Array:
    """The fine samples of one time cell, for every source, antenna and channel.

    ``w_time`` ``(n_st, n_int_time)`` and ``start_time`` (a scalar) are the
    cell's own row of the tables; the frequency tables are whole, since every
    channel of the cell is interpolated at once.

    Returns ``(n_rfi, n_ant, n_freq, n_int_freq, n_int_time)``.
    """
    n_sf, n_st = w_freq.shape[1], w_time.shape[0]

    # The stencil: the n_st cells around this one, then the n_sf channels
    # around each channel.
    idx_time = start_time + jnp.arange(n_st)  # (n_st,)
    idx_freq = start_freq[:, None] + jnp.arange(n_sf)  # (n_freq, n_sf)
    stencil = jnp.take(rfi_A, idx_time, axis=3)  # (n_rfi, n_ant, n_freq, n_st)
    stencil = jnp.take(stencil, idx_freq, axis=2)  # (n_rfi, n_ant, n_freq, n_sf, n_st)

    # Separable weights: one table per axis.
    return jnp.einsum("rafkl,fku,lv->rafuv", stencil, w_freq, w_time)


def linear_phase_moments(a: Array, degree: int) -> Array:
    """The moments ``mean_{[-1,1]} x**m exp(i*a*x)``, through ``degree``.

    Integration by parts divides by the large winding, so upward recurrence
    is stable only while m <= |a|. Above that point we run the same identity
    downwards from a zero tail, 64 orders beyond the last requested moment.
    The unwanted solution then contracts by |a|/m at every step. This also
    supplies the zero-frequency limit without a division by zero or a
    cancellation-prone Taylor series at moderate winding.
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

    _, upward = lax.scan(upward_step, zero, jnp.arange(1, degree + 1, dtype=jnp.int32), unroll=1)
    upward = jnp.concatenate((zero[..., None], jnp.moveaxis(upward, 0, -1)), axis=-1)

    # Large a uses the upward result throughout. Giving the unused downward
    # branch a benign argument keeps masked overflows out of differentiation.
    down_a = jnp.where(jnp.abs(a) <= degree, a, 0)
    ep, em = jnp.exp(1j * down_a), jnp.exp(-1j * down_a)

    def step(last, m):
        # A floating base raised to an integer scan index adopts the default
        # real dtype under x64. Parity and recurrence factors instead follow a.
        sign = jnp.where(m % 2 == 0, 1, -1).astype(a.dtype)
        previous = ((ep - sign * em) / 2 - 1j * down_a * last) / m.astype(a.dtype)
        return previous, previous

    # The scan runs m = degree+64 .. 1; reversed, output k holds order m = k+1
    # and the requested moments are the first degree+1 of them.
    _, downward = lax.scan(
        step, jnp.zeros_like(positive), jnp.arange(degree + 64, 0, -1, dtype=jnp.int32), unroll=1,
    )
    downward = jnp.moveaxis(downward[::-1][:degree + 1], 0, -1)
    return jnp.where(jnp.arange(degree + 1) <= jnp.abs(a)[..., None], upward, downward)


def quadratic_phase_moments(a: Array, b: Array, degree: int, terms: int = 16) -> Array:
    """Normalised moments on [-1, 1] of ``exp(i*(a*x+b*x*x))``.

    Outside or beyond the neighbourhood of the stationary point, expand the
    curvature about linear-phase moments. The caller splits the cell first:
    with |b| <= 1 per piece, 16 terms leave less than 2e-13 absolute remainder.
    Near the stationary point the Fresnel seed and its upward recurrence are
    stable, except at small b where division by b is itself ill-conditioned.
    There the same convergent series supplies the continuous limit instead.
    """
    from jax.scipy.special import fresnel

    a, b = jnp.broadcast_arrays(a, b)
    stationary = (jnp.abs(a) <= 2.5 * jnp.abs(b)) & (jnp.abs(b) >= 1)
    series_a, series_b = jnp.where(stationary, 0., a), jnp.where(stationary, 0., b)
    linear = linear_phase_moments(series_a, degree + 2 * (terms - 1))
    series = jnp.zeros_like(linear[..., :degree + 1])
    coefficient = jnp.ones_like(series_a, dtype=linear.dtype)

    def series_step(carry, k):
        series, coefficient = carry
        window = lax.dynamic_slice_in_dim(linear, 2*k, degree + 1, axis=-1)
        series = series + coefficient[..., None] * window
        coefficient = coefficient * (1j * series_b) / (k + 1).astype(series_a.dtype)
        return (series, coefficient), None

    (series, _), _ = lax.scan(series_step, (series, coefficient), jnp.arange(terms, dtype=jnp.int32), unroll=1)

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

    _, moments = lax.scan(moment_step, (jnp.zeros_like(zero), zero), jnp.arange(1, degree + 1, dtype=jnp.int32), unroll=1)
    moments = jnp.concatenate((zero[..., None], jnp.moveaxis(moments, 0, -1)), axis=-1)
    return jnp.where(stationary[..., None], moments, series)


def analytic_rfi_vis(
    rfi_A: Array, rfi_phase: Array, rfi_delay: Array,
    w_freq: Array, start_freq: Array, g_time: Array, start_time: Array,
    dnu_mhz: Array, int_time: Array, freqs_mhz: Array, a1: Array, a2: Array,
    *, segments: int = 4, terms: int = 16, cubic_terms: int = 3,
) -> Array:
    """Integrate the amplitude polynomial against the quadratic delay phase.

    ``g_time[t,l,m]`` is the Lagrange basis in powers of x = 2*tau/T. The
    frequency contraction is unchanged, so finite channel integration retains
    exactly the reference's frequency offsets. Time integration is analytic:
    multiply the antenna polynomials by convolution, then contract against
    phase moments. The quadratic phase is integrated analytically and residual
    cubic phase is expanded on each piece. Three terms suffice at the measured
    cubic coefficients; zero deliberately drops cubic phase for comparison.
    Derivatives above order three are omitted.

    Equal pieces keep curvature small without imposing a Nyquist sample count.
    Translation of both the amplitude and phase is exact, including the
    constant phase of each piece. The working arrays carry polynomial degree,
    not fringe winding. Only the data-grid amplitude is differentiated.
    """
    amp_degree = g_time.shape[-1] - 1
    degree = 2 * amp_degree
    n_cubic = max(1, cubic_terms)
    moment_degree = degree + 3 * (n_cubic - 1)
    # Enabling x64 permits double inputs; it must not widen a single-precision
    # integration. Type real constants before arithmetic, including the host
    # power table, so every scan's result keeps its carry's input precision.
    real_dtype = rfi_A.real.dtype
    host_radius = np.asarray(1. / segments, dtype=real_dtype)
    radius = jnp.asarray(host_radius)
    int_time = jnp.asarray(int_time, dtype=real_dtype)
    nu = freqs_mhz[:, None] + dnu_mhz[None, :]
    orders = jnp.arange(degree + 1, dtype=jnp.int32)
    binomial = [
        [math.comb(m, j) if j <= m else 0 for j in range(degree + 1)]
        for m in range(degree + 1)
    ]
    radius_powers = np.power(host_radius, np.arange(degree + 1), dtype=real_dtype)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def one_cell(t):
        amplitude = fine_signal(rfi_A, w_freq, start_freq, g_time[t], start_time[t])
        p, q = amplitude[:, a1], amplitude[:, a2].conj()
        if amp_degree == 0:
            product = p * q
        else:
            # Each row holds q[m-j], with zeros outside its polynomial. Contract
            # all output degrees together, retaining full float32 precision on GPU.
            indices = orders[:, None] - jnp.arange(amp_degree + 1)
            shifted_q = jnp.where(
                (indices >= 0) & (indices <= amp_degree),
                q[..., jnp.clip(indices, 0, amp_degree)], 0,
            )
            product = jnp.einsum("...j,...mj->...m", p, shifted_q, precision=lax.Precision.HIGHEST)
        delay_cell, phase_cell = rfi_delay[:, :, t], rfi_phase[..., t]
        delay = delay_cell[:, a1] - delay_cell[:, a2]
        phi = phase_cell[:, a1] - phase_cell[:, a2]
        phi0 = phi[..., None] + 2 * jnp.pi * dnu_mhz * delay[..., 0, None, None]
        d1 = delay[..., 1] if rfi_delay.shape[-1] > 1 else jnp.zeros_like(delay[..., 0])
        d2 = delay[..., 2] if rfi_delay.shape[-1] > 2 else jnp.zeros_like(delay[..., 0])
        a = 2 * jnp.pi * nu * d1[..., None, None] * (int_time / 2)
        b = jnp.pi * nu * d2[..., None, None] * (int_time / 2)**2
        d3 = delay[..., 3] if rfi_delay.shape[-1] > 3 and cubic_terms else jnp.zeros_like(delay[..., 0])
        c = (jnp.pi / 3) * nu * d3[..., None, None] * (int_time / 2)**3

        def piece(total, i):
            centre = -1 + (2 * i.astype(real_dtype) + 1) * radius
            # x = centre + radius*y. Keeping coefficients dimensionless avoids
            # powers of a seconds-valued T in the moment recurrence.
            # Form all powers together, then sum each translated coefficient
            # from left to right in the source degree.
            # These factors share the product's real dtype. Casting only the
            # shifted result would hide wider arithmetic inside the scan.
            centre_powers = (centre**orders).astype(real_dtype)
            combinations = jnp.asarray(binomial, dtype=real_dtype)
            radii = jnp.asarray(radius_powers, dtype=real_dtype)

            def translate(shifted, m):
                power = centre_powers[jnp.maximum(m - orders, 0)]
                value = combinations[m] * power * radii * product[..., m, None]
                return jnp.where(orders <= m, shifted + value, shifted), None

            shifted, _ = lax.scan(translate, jnp.zeros_like(product), orders, unroll=1)
            moments = quadratic_phase_moments(
                (a + 2*b*centre + 3*c*centre**2)*radius,
                (b + 3*c*centre)*radius**2, moment_degree, terms,
            )
            # Translate the cubic exactly too. Only the residual c*r^3*y^3
            # needs expansion; its constant, linear and quadratic parts are
            # already in the phase and moments. The residual coefficient
            # falls with the cube of the piece width.
            coefficient = jnp.ones_like(a, dtype=product.dtype)
            integral = jnp.zeros_like(coefficient)

            def cubic_step(carry, k):
                integral, coefficient = carry
                window = lax.dynamic_slice_in_dim(moments, 3*k, degree + 1, axis=-1)
                integral = integral + coefficient * jnp.sum(shifted * window, axis=-1)
                coefficient = coefficient * (1j * c * radius**3) / (k + 1).astype(real_dtype)
                return (integral, coefficient), None

            (integral, _), _ = lax.scan(
                cubic_step, (integral, coefficient), jnp.arange(n_cubic, dtype=jnp.int32), unroll=1,
            )
            phase = phi0 + a*centre + b*centre**2 + c*centre**3
            return total + jnp.exp(1j * phase) * integral / segments, None

        result, _ = lax.scan(
            piece, jnp.zeros(product.shape[:-1], dtype=product.dtype), jnp.arange(segments, dtype=jnp.int32), unroll=1,
        )
        return jnp.sum(jnp.mean(result, axis=-1), axis=0)

    return jnp.moveaxis(lax.map(one_cell, jnp.arange(rfi_A.shape[-1])), 0, -1)
