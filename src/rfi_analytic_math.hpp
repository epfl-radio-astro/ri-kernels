#pragma once

#include "rfi_interp_common.hpp"
#include "rfi_analytic_limits.hpp"

namespace ri_kernels {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RI_ANALYTIC_INLINE __host__ __device__ __forceinline__
#define RI_ANALYTIC_UNROLL _Pragma("unroll")
#else
#define RI_ANALYTIC_INLINE inline
#define RI_ANALYTIC_UNROLL
#endif

// The reference works in x = 2*tau/T. The recurrence and the translation below
// keep that coordinate throughout; seconds enter only when forming a, b, c.
//
// The pieces split by what they have to carry. Every *angle* is formed and
// reduced in double whatever the working precision W of the recurrences: a
// rounded winding moves a whole piece's phase, and phi, a and b reach
// magnitudes at which a single-precision angle has no fractional turn left.
// The moments and the weights, by contrast, only ever multiply the coefficient
// buffers, so they need no more than those carry. W is therefore the caller's
// choice: the GPU kernels pass the operator's own precision, because a
// consumer card retires one double-precision instruction for every sixty-four
// single-precision ones and the weights are the whole cost of the kernel; the
// CPU kernels, where double is as fast as float, keep passing double.
//
// The two places W is *not* enough are kept in double regardless: the
// argument reduction here, and the Fresnel seed below, whose small-argument
// series passes through terms a hundredfold larger than its sum.
template <typename W> RI_ANALYTIC_INLINE Cplx<W> analytic_exp(double x);

template <> RI_ANALYTIC_INLINE Cplx<double> analytic_exp<double>(double x) {
  Cplx<double> z;
  sincos_t(x, &z.im, &z.re);
  return z;
}

template <> RI_ANALYTIC_INLINE Cplx<float> analytic_exp<float>(double x) {
  // Reduce to a quadrant against a two-word pi/2, in double, then finish in
  // float. Handing the float library a reduced argument would be enough for
  // accuracy but not for cost: its own reduction is a called routine, and
  // this is the one transcendental in the kernel's inner loop. The two words
  // hold the reduction exact past 1e9 quadrants, far beyond the 1e4 turns the
  // steepest delay winds through a cell.
  constexpr double quarter_hi = 1.57079632679489655799898173427;
  constexpr double quarter_lo = 6.12323399573676603586882014729e-17;
  const double quadrants = rint(x * 0.636619772367581343075535053490);
  const float r = float((x - quadrants * quarter_hi) - quadrants * quarter_lo);
  const float z = r * r;
  // Taylor to r^9 and r^8 on |r| <= pi/4, where the first dropped term is
  // three parts in 1e9 -- below what the float result can carry anyway.
  float s = 2.75573192e-6f;
  s = s * z - 1.98412698e-4f;
  s = s * z + 8.33333333e-3f;
  s = s * z - 1.66666667e-1f;
  s = r + r * z * s;
  float c = 2.48015873e-5f;
  c = c * z - 1.38888889e-3f;
  c = c * z + 4.16666667e-2f;
  c = c * z - 5.00000000e-1f;
  c = 1.f + z * c;
  const int n = int(llrint(quadrants) & 3);
  const float sn = (n & 1) ? c : s, cs = (n & 1) ? s : c;
  return {(n == 1 || n == 2) ? -cs : cs, (n == 2 || n == 3) ? -sn : sn};
}

// C(n, k), with the loop bounded by the widest product the storage allows
// rather than by k. The bound is what matters: given a trip count it cannot
// see, the compiler keeps the loop and its divide instead of folding the
// coefficient away, and the translation at the end of a segment reaches this
// once per (m, j) per segment per baseline. Left as a run-time loop it cost
// more than every other line of the weight put together.
RI_ANALYTIC_INLINE constexpr double analytic_binomial(int n, int k) {
  double v = 1;
  RI_ANALYTIC_UNROLL
  for (int i = 1; i <= AnalyticStorage<false>::product; ++i)
    if (i <= k) v *= double(n - i + 1) / i;
  return v;
}

// The power series is well conditioned at the small arguments reached near
// stationarity. Above it, rational auxiliary functions retain the accuracy
// of the reference Fresnel seed. A two-term asymptotic at 2.5 would leave a
// seed error of order 1e-3, which the higher moments would amplify.
//
// This one stays in double whatever the caller's working precision. Its
// series runs through terms a hundredfold larger than the sum they build, so
// single precision would surrender two digits of the seed before the moment
// recurrence has started; and it is reached only where the cell's phase turns
// over inside it, not on the winding-dominated path the kernels spend their
// time on. The caller rounds the result.
TAB_H_D inline Cplx<double> analytic_fresnel(double u) {
  const double x = fabs(u), pi = two_pi_c<double>() / 2;
  double c, s;
  if (x < 2.5) {
    const double z = pi * x * x / 2;
    double tc = x, ts = x * z / 3;
    c = tc; s = ts;
    for (int n = 1; n < 40; ++n) {
      tc *= -z * z * (4 * n - 3) / ((2. * n) * (2 * n - 1) * (4 * n + 1));
      ts *= -z * z * (4 * n - 1) / ((2. * n) * (2 * n + 1) * (4 * n + 3));
      c += tc; s += ts;
      if (fabs(tc) + fabs(ts) < 1e-17) break;
    }
  } else {
    const double t = 1 / (pi * x * x), z = t * t;
    double fn = 4.21543555043677546506e-1;
    fn = fn * z + 1.43407919780758885261e-1;
    fn = fn * z + 1.15220955073585758835e-2;
    fn = fn * z + 3.45017939782574027900e-4;
    fn = fn * z + 4.63613749287867322088e-6;
    fn = fn * z + 3.05568983790257605827e-8;
    fn = fn * z + 1.02304514164907233465e-10;
    fn = fn * z + 1.72010743268161828879e-13;
    fn = fn * z + 1.34283276233062758925e-16;
    fn = fn * z + 3.76329711269987889006e-20;
    double fd = 1.00000000000000000000e0;
    fd = fd * z + 7.51586398353378947175e-1;
    fd = fd * z + 1.16888925859191382142e-1;
    fd = fd * z + 6.44051526508858611005e-3;
    fd = fd * z + 1.55934409164153020873e-4;
    fd = fd * z + 1.84627567348930545870e-6;
    fd = fd * z + 1.12699224763999035261e-8;
    fd = fd * z + 3.60140029589371370404e-11;
    fd = fd * z + 5.88754533621578410010e-14;
    fd = fd * z + 4.52001434074129701496e-17;
    fd = fd * z + 1.25443237090011264384e-20;
    double gn = 5.04442073643383265887e-1;
    gn = gn * z + 1.97102833525523411709e-1;
    gn = gn * z + 1.87648584092575249293e-2;
    gn = gn * z + 6.84079380915393090172e-4;
    gn = gn * z + 1.15138826111884280931e-5;
    gn = gn * z + 9.82852443688422223854e-8;
    gn = gn * z + 4.45344415861750144738e-10;
    gn = gn * z + 1.08268041139020870318e-12;
    gn = gn * z + 1.37555460633261799868e-15;
    gn = gn * z + 8.36354435630677421531e-19;
    gn = gn * z + 1.86958710162783235106e-22;
    double gd = 1.00000000000000000000e0;
    gd = gd * z + 1.47495759925128324529e0;
    gd = gd * z + 3.37748989120019970451e-1;
    gd = gd * z + 2.53603741420338795122e-2;
    gd = gd * z + 8.14679107184306179049e-4;
    gd = gd * z + 1.27545075667729118702e-5;
    gd = gd * z + 1.04314589657571990585e-7;
    gd = gd * z + 4.60680728146520428211e-10;
    gd = gd * z + 1.10273215066240270757e-12;
    gd = gd * z + 1.38796531259578871258e-15;
    gd = gd * z + 8.39158816283118707363e-19;
    gd = gd * z + 1.86958710162783236342e-22;
    const double f = 1 - z * fn / fd, g = t * gn / gd;
    const auto e = analytic_exp<double>(pi * x * x / 2);
    c = 0.5 + (f * e.im - g * e.re) / (pi * x);
    s = 0.5 - (f * e.re + g * e.im) / (pi * x);
  }
  return {copysign(c, u), copysign(s, u)};
}

// The upward linear recurrence divides by the large winding. Once m exceeds
// |a|, use the reference's zero tail 64 orders above the requested degree and
// recurse down. In particular this supplies the exact continuous a=0 limit.
//
// The default path unrolls the requested orders. The 64-order tail carries
// only one complex scalar; after that each stored order has a compile-time
// index, so the CUDA compiler can keep the recurrence in registers.
template <bool Default, typename W>
RI_ANALYTIC_INLINE void analytic_linear(double a, int requested, Cplx<W> *values) {
  using Storage = AnalyticStorage<Default>;
  const int degree = Default ? Storage::linear - 1 : requested;
  assert(requested == degree && degree >= 0 && degree < Storage::linear);
  const double aa = fabs(a);
  const auto e = analytic_exp<W>(a);
  const W aw = W(a);
  if (aa <= degree) {
    Cplx<W> last{0, 0};
    for (int m = degree + 64; m > degree + 1; --m) {
      const Cplx<W> boundary = (m & 1) ? Cplx<W>{e.re, 0} : Cplx<W>{0, e.im};
      last = cscale(W(1) / W(m), cadd(boundary, cscale(-aw, ctimes_i(last))));
    }
    RI_ANALYTIC_UNROLL
    for (int j = 0; j <= degree; ++j) {
      const int m = degree + 1 - j;
      const Cplx<W> boundary = (m & 1) ? Cplx<W>{e.re, 0} : Cplx<W>{0, e.im};
      last = cscale(W(1. / m), cadd(boundary, cscale(-aw, ctimes_i(last))));
      values[m - 1] = last;
    }
  }
  // Compare against an integer bound, not against |a| itself: the unrolled
  // body would otherwise weigh one double compare per order, and those go
  // down the one pipe a consumer card is short of.
  const int ascent = !(aa < degree) ? degree : int(aa);
  Cplx<W> last{a == 0 ? W(1) : e.im / aw, 0};
  values[0] = last;
  RI_ANALYTIC_UNROLL
  for (int m = 1; m <= degree; ++m) {
    if (m <= ascent) {
      const Cplx<W> boundary = (m & 1) ? Cplx<W>{e.re, 0} : Cplx<W>{0, e.im};
      const auto z = cadd(boundary, cscale(-W(m), last));
      last = cscale(W(-1) / aw, ctimes_i(z));
      values[m] = last;
    }
  }
}

template <bool Default, typename W>
RI_ANALYTIC_INLINE void analytic_quadratic(double a, double b, int requested, int nterms,
                                           Cplx<W> *moments) {
  using Storage = AnalyticStorage<Default>;
  const int degree = Default ? Storage::moments - 1 : requested;
  const int terms = Default ? Storage::terms : nterms;
  assert(requested == degree && nterms == terms);
  assert(degree >= 0 && degree < Storage::moments && terms >= 1 && terms <= Storage::terms);
  assert(degree + 2 * (terms - 1) < Storage::linear);
  if (fabs(a) <= 2.5 * fabs(b) && fabs(b) >= 1) {
    const double scale = sqrt(2 * fabs(b) / (two_pi_c<double>() / 2));
    const double shift = a / (2 * b);
    const auto p = analytic_fresnel(scale * (1 + shift));
    const auto q = analytic_fresnel(scale * (-1 + shift));
    moments[0] = cscale(W(1 / (2 * scale)), cmul(analytic_exp<W>(-a * shift / 2),
                     Cplx<W>{W(p.re - q.re), W((b > 0 ? 1 : -1) * (p.im - q.im))}));
    const auto ep = analytic_exp<W>(a + b), em = analytic_exp<W>(-a + b);
    const W aw = W(a), half_over_b = W(-1 / (2 * b));
    RI_ANALYTIC_UNROLL
    for (int m = 1; m <= degree; ++m) {
      auto z = cscale(W(.5), cadd(ep, cscale((m & 1) ? W(-1) : W(1), em)));
      if (m > 1) z = cadd(z, cscale(-W(m - 1), moments[m - 2]));
      z = cadd(z, cscale(-aw, ctimes_i(moments[m - 1])));
      moments[m] = cscale(half_over_b, ctimes_i(z));
    }
  } else {
    Cplx<W> linear[Storage::linear];
    analytic_linear<Default, W>(a, degree + 2 * (terms - 1), linear);
    // The k = 0 term seeds the sum rather than adding to a zeroed array: the
    // coefficient is one, so the whole first pass would be a copy.
    RI_ANALYTIC_UNROLL
    for (int m = 0; m <= degree; ++m) moments[m] = linear[m];
    // b/k, not (W)b/(W)k, would be a double divide per term: k is a constant
    // of the unrolled loop, but b is not, so the quotient is formed at run
    // time and the double pipe is the one this kernel cannot afford.
    const W bw = W(b);
    Cplx<W> coefficient{1, 0};
    RI_ANALYTIC_UNROLL
    for (int k = 1; k < terms; ++k) {
      coefficient = cscale(bw * (W(1) / W(k)), ctimes_i(coefficient));
      RI_ANALYTIC_UNROLL
      for (int m = 0; m <= degree; ++m)
        moments[m] = cadd(moments[m], cmul(coefficient, linear[2 * k + m]));
    }
  }
}

// Return the weights H_m for the original cell's product coefficients P_m.
// Translating these weights instead of the product itself is the same finite
// sum as the reference, and lets the transpose reuse exactly the forward map.
// Default settings have fixed small bounds so CUDA can scalarise the arrays;
// the bounded general path accommodates wider stencils and convergence sweeps.
template <bool Default, typename W>
RI_ANALYTIC_INLINE void analytic_weights(int count, int segments, int terms, int cubic_terms,
                                      double a, double b, double c, double phi,
                                      Cplx<W> *weights) {
  using Storage = AnalyticStorage<Default>;
  assert(analytic_configuration_fits(count, {segments, terms, cubic_terms}));
  assert(!Default || analytic_is_default(count, {segments, terms, cubic_terms}));
  const auto sizes = Default ? analytic_sizes(Storage::coefficients, Storage::terms, Storage::cubic_terms)
                             : analytic_sizes(count, terms, cubic_terms);
  const int degree = sizes.product - 1, nk = sizes.cubic_terms;
  const int md = sizes.moments - 1, nt = Default ? Storage::terms : terms;
  assert(sizes.product <= Storage::product && sizes.moments <= Storage::moments &&
         sizes.linear <= Storage::linear);
  const double radius = 1. / segments;
  RI_ANALYTIC_UNROLL
  for (int m = 0; m <= degree; ++m) weights[m] = {0, 0};
  for (int i = 0; i < segments; ++i) {
    const double centre = -1 + (2 * i + 1) * radius;
    const W cw = W(centre), rw = W(radius), cube = W(c * radius * radius * radius);
    Cplx<W> moments[Storage::moments], effective[Storage::product];
    analytic_quadratic<Default, W>(
        (a + 2 * b * centre + 3 * c * centre * centre) * radius,
        (b + 3 * c * centre) * radius * radius, md, nt, moments);
    // As in the curvature sum above, the zeroth cubic term is the identity.
    RI_ANALYTIC_UNROLL
    for (int j = 0; j <= degree; ++j) effective[j] = moments[j];
    Cplx<W> coefficient{1, 0};
    RI_ANALYTIC_UNROLL
    for (int k = 1; k < nk; ++k) {
      coefficient = cscale(cube * (W(1) / W(k)), ctimes_i(coefficient));
      RI_ANALYTIC_UNROLL
      for (int j = 0; j <= degree; ++j)
        effective[j] = cadd(effective[j], cmul(coefficient, moments[3 * k + j]));
    }
    const auto e = cscale(rw, analytic_exp<W>(phi + a * centre + b * centre * centre +
                                              c * centre * centre * centre));
    // Translate the segment back to the cell's coordinate. The two powers are
    // walked once per segment rather than rebuilt inside the double sum: the
    // radius one belongs to j alone, so it folds into the moment it scales,
    // and the centre one is a running product.
    W shift[Storage::product];
    shift[0] = 1;
    RI_ANALYTIC_UNROLL
    for (int k = 1; k <= degree; ++k) shift[k] = shift[k - 1] * cw;
    W span = 1;
    RI_ANALYTIC_UNROLL
    for (int j = 0; j <= degree; ++j) {
      effective[j] = cscale(span, effective[j]);
      span *= rw;
    }
    RI_ANALYTIC_UNROLL
    for (int m = 0; m <= degree; ++m) {
      Cplx<W> z{0, 0};
      RI_ANALYTIC_UNROLL
      for (int j = 0; j <= m; ++j)
        z = cadd(z, cscale(W(analytic_binomial(m, j)) * shift[m - j], effective[j]));
      weights[m] = cadd(weights[m], cmul(e, z));
    }
  }
}

} // namespace ri_kernels
