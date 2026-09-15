#pragma once

#include "rfi_analytic_math.hpp"

namespace ri_kernels {

template <typename T> struct AnalyticViews {
  Tensor1D<const int *> a1, a2;
  Tensor2D<const int *> pair, tile_pairs;
  Tensor4D<const Cplx<T> *> amp, amp_dot;
  Tensor4D<const T *> phase, delay;
  Tensor3D<const T *> wf, gt;
  Tensor1D<const int *> sf, st;
  Tensor1D<const T *> dnu, freqs;
  const T *int_time;
  AnalyticOptions options;
};

// Coefficients have the interpolation sample buffer's antenna-fast layout.
// There is one frequency-quadrature axis, but no time-quadrature axis.
TAB_H_D inline std::int64_t analytic_index(std::int64_t cell, std::int64_t r,
    std::int64_t u, std::int64_t m, std::int64_t ant, std::int64_t nr,
    std::int64_t nu, std::int64_t nm, std::int64_t na) {
  return ((((cell * nr + r) * nu + u) * nm + m) * na + ant);
}

// What a baseline contributes to its weights beyond the tables: the two
// antennas' delay difference in the four orders the closed form carries, and
// the difference of their centre phases. Named scalars, not an array indexed
// by a run-time path count -- that array lands in local memory, and this is
// the innermost thing the kernels do.
struct AnalyticPair {
  double tau, rate, curvature, jerk, phase;
};

template <typename T>
RI_ANALYTIC_INLINE double analytic_order(Tensor4D<const T *> delay, std::int64_t ant,
    std::int64_t r, std::int64_t t, int k) {
  return k < delay.shape[3] ? double(delay(ant, r, t, k)) : 0.;
}

// Straight from the data grid. A GPU block covers two antenna tiles and one
// cell, so it stages these instead (see the kernel); the CPU kernels and the
// bounds tests read them here.
template <typename T>
RI_ANALYTIC_INLINE AnalyticPair analytic_pair_of(AnalyticViews<T> v, std::int64_t p,
    std::int64_t q, std::int64_t r, std::int64_t f, std::int64_t t) {
  return {analytic_order(v.delay, p, r, t, 0) - analytic_order(v.delay, q, r, t, 0),
          analytic_order(v.delay, p, r, t, 1) - analytic_order(v.delay, q, r, t, 1),
          analytic_order(v.delay, p, r, t, 2) - analytic_order(v.delay, q, r, t, 2),
          analytic_order(v.delay, p, r, t, 3) - analytic_order(v.delay, q, r, t, 3),
          double(v.phase(p, r, f, t)) - double(v.phase(q, r, f, t))};
}

template <bool Default, typename W, typename T>
RI_ANALYTIC_INLINE void analytic_pair_weights(AnalyticViews<T> v, AnalyticPair d,
    std::int64_t f, std::int64_t u, Cplx<W> *h) {
  const double half = double(*v.int_time) / 2;
  const double nu = double(v.freqs(f)) + double(v.dnu(u));
  const double phi = d.phase + two_pi_c<double>() * double(v.dnu(u)) * d.tau;
  const double a = two_pi_c<double>() * nu * d.rate * half;
  const double b = (two_pi_c<double>() / 2) * nu * d.curvature * half * half;
  const double c = v.options.cubic_terms ?
      (two_pi_c<double>() / 6) * nu * d.jerk * half * half * half : 0;
  analytic_weights<Default, W>(int(v.gt.shape[2]), int(v.options.segments),
      int(v.options.terms), int(v.options.cubic_terms), a, b, c, phi, h);
}

template <typename T, typename W>
TAB_H_D inline Cplx<T> analytic_cast(Cplx<W> z) { return {T(z.re), T(z.im)}; }

template <typename T>
TAB_H_D inline Cplx<T> analytic_coefficient(AnalyticViews<T> v,
    Tensor4D<const Cplx<T> *> amp, std::int64_t ant, std::int64_t r,
    std::int64_t f, std::int64_t t, std::int64_t u, std::int64_t m) {
  return interp_amp(amp, v.wf.ptr + f * v.wf.shape[1] * v.wf.shape[2],
      v.gt.ptr + t * v.gt.shape[1] * v.gt.shape[2], std::int64_t(v.sf(f)),
      std::int64_t(v.st(t)), v.wf.shape[1], v.gt.shape[1], v.wf.shape[2],
      v.gt.shape[2], ant, r, u, m);
}

// JAX's complex transpose pairs without conjugating the cotangent. For
// V = H p conj(q), p_bar = g H conj(q), q_bar = conj(g H p).
template <bool Default, typename W, typename T>
RI_ANALYTIC_INLINE Cplx<T> analytic_contract(const Cplx<T> *p, const Cplx<T> *q,
    const Cplx<T> *dp, const Cplx<T> *dq, const Cplx<W> *h,
    int count, std::int64_t stride, bool jvp) {
  const int n = Default ? AnalyticStorage<Default>::coefficients : count;
  assert(n == count && n >= 1 && n <= AnalyticStorage<Default>::coefficients);
  Cplx<T> total{0, 0};
  RI_ANALYTIC_UNROLL
  for (int m = 0; m < 2 * n - 1; ++m) {
    Cplx<T> product{0, 0};
    RI_ANALYTIC_UNROLL
    for (int j = 0; j < n; ++j) {
      const int k = m - j;
      if (k < 0 || k >= n) continue;
      const auto value = jvp ?
          cadd(cmul(dp[j * stride], cconj(q[k * stride])),
               cmul(p[j * stride], cconj(dq[k * stride]))) :
          cmul(p[j * stride], cconj(q[k * stride]));
      product = cadd(product, value);
    }
    total = cadd(total, cmul(product, analytic_cast<T, W>(h[m])));
  }
  return total;
}

template <ffi::DataType A, ffi::DataType R>
ffi::Error analytic_validate(interp_index_t a1, interp_index_t a2,
    ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tiles,
    ffi::Buffer<A, 4> amp, ffi::Buffer<A, 4> dot,
    ffi::Buffer<R, 4> phase, ffi::Buffer<R, 4> delay,
    ffi::Buffer<R, 3> wf, interp_index_t sf, ffi::Buffer<R, 3> gt,
    interp_index_t st, ffi::Buffer<R, 1> dnu, ffi::Buffer<R, 0> duration,
    ffi::Buffer<R, 1> freq, AnalyticOptions opt, bool cpu) {
  const auto a = amp.dimensions(), p = phase.dimensions(), d = delay.dimensions();
  const auto w = wf.dimensions(), g = gt.dimensions();
  const auto na = a[0], nr = a[1], nf = a[2], nt = a[3], nb = a1.dimensions()[0];
  const auto ntiles = (na + 31) / 32;
  if (na < 1 || nr < 1 || nf < 1 || nt < 1 || a2.dimensions()[0] != nb ||
      !interp_same_shape(amp, dot) || !interp_same_shape(amp, phase) ||
      d[0] != na || d[1] != nr || d[2] != nt || d[3] < 1 ||
      w[0] != nf || w[1] < 1 || w[1] > nf || w[2] < 1 ||
      g[0] != nt || g[1] < 1 || g[1] > nt || g[2] < 1 ||
      sf.dimensions()[0] != nf || st.dimensions()[0] != nt ||
      dnu.dimensions()[0] != w[2] || freq.dimensions()[0] != nf ||
      pair.dimensions()[0] != na || pair.dimensions()[1] != na ||
      tiles.dimensions()[0] != ntiles * (ntiles + 1) / 2 || tiles.dimensions()[1] != 1024)
    return ffi::Error::InvalidArgument("Incompatible analytic signal, tangent, table, or baseline shapes");
  if (!analytic_configuration_fits(g[2], opt))
    return ffi::Error::InvalidArgument(
        "Analytic configuration exceeds compiled coefficient or moment capacity, or has invalid integration options");
  if (cpu) {
    if (!std::isfinite(*duration.typed_data()) || *duration.typed_data() <= 0)
      return ffi::Error::InvalidArgument("int_time must be finite and positive");
    for (std::int64_t b = 0; b < nb; ++b)
      if (a1.typed_data()[b] < 0 || a1.typed_data()[b] >= na ||
          a2.typed_data()[b] < 0 || a2.typed_data()[b] >= na)
        return ffi::Error::InvalidArgument("Baseline antenna out of range");
    for (std::int64_t f = 0; f < nf; ++f) {
      const auto s = sf.typed_data()[f];
      if (s < 0 || s > f || s + w[1] <= f || s + w[1] > nf)
        return ffi::Error::InvalidArgument("Frequency stencil must contain its cell and lie inside its axis");
    }
    for (std::int64_t t = 0; t < nt; ++t) {
      const auto s = st.typed_data()[t];
      if (s < 0 || s > t || s + g[1] <= t || s + g[1] > nt)
        return ffi::Error::InvalidArgument("Time stencil must contain its cell and lie inside its axis");
    }
  }
  return ffi::Error::Success();
}

template <typename T, ffi::DataType A, ffi::DataType R>
AnalyticViews<T> analytic_views(interp_index_t a1, interp_index_t a2,
    ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tiles,
    ffi::Buffer<A, 4> amp, ffi::Buffer<A, 4> dot,
    ffi::Buffer<R, 4> phase, ffi::Buffer<R, 4> delay,
    ffi::Buffer<R, 3> wf, interp_index_t sf, ffi::Buffer<R, 3> gt,
    interp_index_t st, ffi::Buffer<R, 1> dnu, ffi::Buffer<R, 0> duration,
    ffi::Buffer<R, 1> freq, AnalyticOptions opt) {
  const auto a = amp.dimensions(), d = delay.dimensions();
  return {
    {a1.typed_data(), a1.dimensions()[0]}, {a2.typed_data(), a2.dimensions()[0]},
    {pair.typed_data(), a[0], a[0]}, {tiles.typed_data(), tiles.dimensions()[0], 1024},
    {reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1], a[2], a[3]},
    {reinterpret_cast<const Cplx<T> *>(dot.typed_data()), a[0], a[1], a[2], a[3]},
    {phase.typed_data(), a[0], a[1], a[2], a[3]},
    {delay.typed_data(), d[0], d[1], d[2], d[3]},
    {wf.typed_data(), wf.dimensions()[0], wf.dimensions()[1], wf.dimensions()[2]},
    {gt.typed_data(), gt.dimensions()[0], gt.dimensions()[1], gt.dimensions()[2]},
    {sf.typed_data(), sf.dimensions()[0]}, {st.typed_data(), st.dimensions()[0]},
    {dnu.typed_data(), dnu.dimensions()[0]}, {freq.typed_data(), freq.dimensions()[0]},
    duration.typed_data(), opt};
}

} // namespace ri_kernels
