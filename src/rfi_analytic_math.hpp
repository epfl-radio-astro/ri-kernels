#pragma once

#include "rfi_interp_common.hpp"
#include "rfi_analytic_limits.hpp"

namespace ri_kernels {

// The reference works in x = 2*tau/T. The recurrence and the translation below
// keep that coordinate throughout; seconds enter only when forming a, b, c.
// Double intermediates also matter for complex64: a small seed error grows in
// the higher stationary moments, and a rounded winding moves every piece's
// phase. The coefficient buffers and visibility arithmetic retain input precision.
TAB_H_D inline Cplx<double> analytic_exp(double x) {
  Cplx<double> z;
  sincos_t(x, &z.im, &z.re);
  return z;
}

TAB_H_D inline double analytic_power(double x, int n) {
  double y = 1;
  for (int i = 0; i < n; ++i) y *= x;
  return y;
}

TAB_H_D inline double analytic_binomial(int n, int k) {
  double v = 1;
  for (int i = 1; i <= k; ++i) v *= double(n - i + 1) / i;
  return v;
}

// The power series is well conditioned at the small arguments reached near
// stationarity. Above it, rational auxiliary functions retain the accuracy
// of the reference Fresnel seed. A two-term asymptotic at 2.5 would leave a
// seed error of order 1e-3, which the higher moments would amplify.
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
    const auto e = analytic_exp(pi * x * x / 2);
    c = 0.5 + (f * e.im - g * e.re) / (pi * x);
    s = 0.5 - (f * e.re + g * e.im) / (pi * x);
  }
  return {copysign(c, u), copysign(s, u)};
}

// The upward linear recurrence divides by the large winding. Once m exceeds
// |a|, use the reference's zero tail 64 orders above the requested degree and
// recurse down. In particular this supplies the exact continuous a=0 limit.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define RI_ANALYTIC_INLINE __host__ __device__ __forceinline__
#define RI_ANALYTIC_UNROLL _Pragma("unroll")
#else
#define RI_ANALYTIC_INLINE inline
#define RI_ANALYTIC_UNROLL
#endif

// The default path unrolls the requested orders. The 64-order tail carries
// only one complex scalar; after that each stored order has a compile-time
// index, so the CUDA compiler can keep the recurrence in registers.
template <bool Default>
RI_ANALYTIC_INLINE void analytic_linear(double a, int requested, Cplx<double> *values) {
  using Storage = AnalyticStorage<Default>;
  const int degree = Default ? Storage::linear - 1 : requested;
  assert(requested == degree && degree >= 0 && degree < Storage::linear);
  const double aa = fabs(a);
  const auto e = analytic_exp(a);
  if (aa <= degree) {
    Cplx<double> last{0, 0};
    for (int m = degree + 64; m > degree + 1; --m) {
      const Cplx<double> boundary = (m & 1) ? Cplx<double>{e.re, 0} : Cplx<double>{0, e.im};
      last = cscale(1. / m, cadd(boundary, cscale(-a, ctimes_i(last))));
    }
    RI_ANALYTIC_UNROLL
    for (int j = 0; j <= degree; ++j) {
      const int m = degree + 1 - j;
      const Cplx<double> boundary = (m & 1) ? Cplx<double>{e.re, 0} : Cplx<double>{0, e.im};
      last = cscale(1. / m, cadd(boundary, cscale(-a, ctimes_i(last))));
      values[m - 1] = last;
    }
  }
  Cplx<double> last{a == 0 ? 1 : e.im / a, 0};
  values[0] = last;
  RI_ANALYTIC_UNROLL
  for (int m = 1; m <= degree; ++m) {
    if (m <= aa) {
      const Cplx<double> boundary = (m & 1) ? Cplx<double>{e.re, 0} : Cplx<double>{0, e.im};
      const auto z = cadd(boundary, cscale(-double(m), last));
      last = cscale(-1 / a, ctimes_i(z));
      values[m] = last;
    }
  }
}

template <bool Default>
RI_ANALYTIC_INLINE void analytic_quadratic(double a, double b, int requested, int nterms,
                                           Cplx<double> *moments) {
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
    moments[0] = cscale(1 / (2 * scale), cmul(analytic_exp(-a * shift / 2),
                     Cplx<double>{p.re - q.re, (b > 0 ? 1 : -1) * (p.im - q.im)}));
    const auto ep = analytic_exp(a + b), em = analytic_exp(-a + b);
    RI_ANALYTIC_UNROLL
    for (int m = 1; m <= degree; ++m) {
      auto z = cscale(0.5, cadd(ep, cscale((m & 1) ? -1. : 1., em)));
      if (m > 1) z = cadd(z, cscale(-double(m - 1), moments[m - 2]));
      z = cadd(z, cscale(-a, ctimes_i(moments[m - 1])));
      moments[m] = cscale(-1 / (2 * b), ctimes_i(z));
    }
  } else {
    Cplx<double> linear[Storage::linear];
    analytic_linear<Default>(a, degree + 2 * (terms - 1), linear);
    RI_ANALYTIC_UNROLL
    for (int m = 0; m <= degree; ++m) moments[m] = {0, 0};
    Cplx<double> coefficient{1, 0};
    RI_ANALYTIC_UNROLL
    for (int k = 0; k < terms; ++k) {
      RI_ANALYTIC_UNROLL
      for (int m = 0; m <= degree; ++m)
        moments[m] = cadd(moments[m], cmul(coefficient, linear[2 * k + m]));
      coefficient = cscale(b / (k + 1), ctimes_i(coefficient));
    }
  }
}

// Return the weights H_m for the original cell's product coefficients P_m.
// Translating these weights instead of the product itself is the same finite
// sum as the reference, and lets the transpose reuse exactly the forward map.
// Default settings have fixed small bounds so CUDA can scalarise the arrays;
// the bounded general path accommodates wider stencils and convergence sweeps.
template <bool Default>
RI_ANALYTIC_INLINE void analytic_weights(int count, int segments, int terms, int cubic_terms,
                                      double a, double b, double c, double phi,
                                      Cplx<double> *weights) {
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
    Cplx<double> moments[Storage::moments], effective[Storage::product];
    analytic_quadratic<Default>(
        (a + 2 * b * centre + 3 * c * centre * centre) * radius,
        (b + 3 * c * centre) * radius * radius, md, nt, moments);
    RI_ANALYTIC_UNROLL
    for (int j = 0; j <= degree; ++j) effective[j] = {0, 0};
    Cplx<double> coefficient{1, 0};
    RI_ANALYTIC_UNROLL
    for (int k = 0; k < nk; ++k) {
      RI_ANALYTIC_UNROLL
      for (int j = 0; j <= degree; ++j)
        effective[j] = cadd(effective[j], cmul(coefficient, moments[3 * k + j]));
      coefficient = cscale(c * radius * radius * radius / (k + 1), ctimes_i(coefficient));
    }
    const auto e = cscale(radius, analytic_exp(phi + a * centre + b * centre * centre +
                                              c * centre * centre * centre));
    RI_ANALYTIC_UNROLL
    for (int m = 0; m <= degree; ++m) {
      Cplx<double> z{0, 0};
      RI_ANALYTIC_UNROLL
      for (int j = 0; j <= m; ++j)
        z = cadd(z, cscale(analytic_binomial(m, j) * analytic_power(centre, m - j) *
                           analytic_power(radius, j), effective[j]));
      weights[m] = cadd(weights[m], cmul(e, z));
    }
  }
}

} // namespace ri_kernels
