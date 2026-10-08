// Each CPU task stages one cell's antenna coefficients and then contracts its
// baselines. The transpose owns a source, so its stencil scatters cannot race
// with another task and no atomics or per-thread data-grid copies are needed.
#include <vector>
#include "parallel_for.hpp"
#include "rfi_analytic_common.hpp"
#include "visibility.h"

namespace ri_kernels {

// Phase carries the phase tangent through the JVP: each entry of the pair's
// P x P product turned by i and scaled by the antennas' tangent difference
// (see analytic_contract). One set of weights serves every entry.
template <bool Default, int P, bool JVP, bool Phase, typename T>
void analytic_cpu_cells(std::int64_t begin, std::int64_t end,
    AnalyticViews<T> v, Tensor4D<Cplx<T> *> out) {
  static_assert(!Phase || JVP, "The phase tangent rides on the JVP");
  constexpr int ne = P * kRfiColumns, npp = P * P;
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2], nb = v.a1.shape[0];
  std::vector<Cplx<T>> coefficients(na * nm * ne), dots(JVP ? na * nm * ne : 0);
  for (auto cell = begin; cell < end; ++cell) {
    const auto f = cell / nt, t = cell % nt;
    for (std::int64_t bl = 0; bl < nb; ++bl)
      for (int k = 0; k < npp; ++k) out(bl, f, t, k) = {0, 0};
    for (std::int64_t r = 0; r < nr; ++r) {
      for (std::int64_t u = 0; u < nu; ++u) {
        for (std::int64_t m = 0; m < nm; ++m) {
          for (int e = 0; e < ne; ++e) {
            for (std::int64_t ant = 0; ant < na; ++ant) {
              const auto o = (m * ne + e) * na + ant;
              coefficients[o] = analytic_coefficient(v, v.amp, ant, r, f, t, u, m, e);
              if constexpr (JVP)
                dots[o] = analytic_coefficient(v, v.amp_dot, ant, r, f, t, u, m, e);
            }
          }
        }
        for (std::int64_t bl = 0; bl < nb; ++bl) {
          const auto p = v.a1(bl), q = v.a2(bl);
          Cplx<double> h[AnalyticStorage<Default>::product];
          analytic_pair_weights<Default, double>(v, analytic_pair_of(v, p, q, r, f, t), f, u, h);
          Cplx<T> z[npp], primal[npp];
          analytic_contract_pol<Default, P, double>(coefficients.data() + p, coefficients.data() + q,
              JVP ? dots.data() + p : nullptr, JVP ? dots.data() + q : nullptr, h, int(nm),
              ne * na, na, JVP, z, Phase ? primal : nullptr);
          const T dphase = Phase ? T(v.phase_dot(p, r, f, t)) - T(v.phase_dot(q, r, f, t)) : T(0);
          for (int k = 0; k < npp; ++k) {
            if constexpr (Phase) z[k] = cadd(z[k], cscale(dphase, ctimes_i(primal[k])));
            out(bl, f, t, k) = cadd(out(bl, f, t, k), cscale(T(1) / T(nu), z[k]));
          }
        }
      }
    }
  }
}

// Phase adds the phase's cotangent: for V = exp(i (phase_p - phase_q)) K the
// pair's product matrix z gives phase_bar_p -= Im(sum_ij g_ij z_ij) and
// phase_bar_q += the same, JAX's real cotangent Re(g dV/dphase). The task owns
// a source, so the per-antenna sums need no atomics either.
template <bool Default, int P, bool Phase, typename T>
void analytic_cpu_transpose(std::int64_t begin, std::int64_t end,
    AnalyticViews<T> v, Tensor4D<const Cplx<T> *> cot, Tensor5D<Cplx<T> *> out,
    Tensor4D<T *> phase_bar) {
  constexpr int ne = P * kRfiColumns, npp = P * P;
  const auto na = v.amp.shape[0], nf = v.amp.shape[2], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2];
  std::vector<Cplx<T>> coefficients(na * nm * ne), bars(na * nm * ne);
  for (auto r = begin; r < end; ++r) {
    for (std::int64_t ant = 0; ant < na; ++ant)
      for (std::int64_t f = 0; f < nf; ++f)
        for (std::int64_t t = 0; t < nt; ++t) {
          for (int e = 0; e < ne; ++e) out(ant, r, f, t, e) = {0, 0};
          if constexpr (Phase) phase_bar(ant, r, f, t) = 0;
        }
    for (std::int64_t f = 0; f < nf; ++f) {
      for (std::int64_t t = 0; t < nt; ++t) {
        for (std::int64_t u = 0; u < nu; ++u) {
          for (std::int64_t m = 0; m < nm; ++m)
            for (int e = 0; e < ne; ++e)
              for (std::int64_t ant = 0; ant < na; ++ant) {
                const auto o = (m * ne + e) * na + ant;
                coefficients[o] = analytic_coefficient(v, v.amp, ant, r, f, t, u, m, e);
                bars[o] = {0, 0};
              }
          for (std::int64_t bl = 0; bl < v.a1.shape[0]; ++bl) {
            const auto p = v.a1(bl), q = v.a2(bl);
            Cplx<double> h[AnalyticStorage<Default>::product];
            analytic_pair_weights<Default, double>(v, analytic_pair_of(v, p, q, r, f, t), f, u, h);
            Cplx<T> g[npp];
            for (int k = 0; k < npp; ++k) g[k] = cscale(T(1) / T(nu), cot(bl, f, t, k));
            if constexpr (Phase) {
              Cplx<T> z[npp];
              analytic_contract_pol<Default, P, double>(coefficients.data() + p,
                  coefficients.data() + q, coefficients.data() + p, coefficients.data() + q, h,
                  int(nm), ne * na, na, false, z);
              Cplx<T> s{0, 0};
              for (int k = 0; k < npp; ++k) s = cadd(s, cmul(g[k], z[k]));
              phase_bar(p, r, f, t) -= s.im;
              phase_bar(q, r, f, t) += s.im;
            }
            analytic_transpose_pol<Default, P, double>(coefficients.data() + p,
                coefficients.data() + q, g, h, int(nm), ne * na, na, bars.data() + p,
                bars.data() + q, [](Cplx<T> *dest, Cplx<T> value) { *dest = cadd(*dest, value); });
          }
          for (std::int64_t k = 0; k < v.wf.shape[1]; ++k)
            for (std::int64_t l = 0; l < v.gt.shape[1]; ++l)
              for (int e = 0; e < ne; ++e)
                for (std::int64_t ant = 0; ant < na; ++ant) {
                  Cplx<T> z{0, 0};
                  for (std::int64_t m = 0; m < nm; ++m)
                    z = cadd(z, cscale(v.wf(f, k, u) * v.gt(t, l, m), bars[(m * ne + e) * na + ant]));
                  auto &dest = out(ant, r, v.sf(f) + k, v.st(t) + l, e);
                  dest = cadd(dest, z);
                }
        }
      }
    }
  }
}

// All buffers become value-type views before work is scheduled; the XLA call
// frame no longer exists when an asynchronous pool task runs.
template <int Mode, typename T, ffi::DataType A, ffi::DataType R>
ffi::Future analytic_cpu_dispatch(ffi::ThreadPool pool,
    analytic_index_t a1, analytic_index_t a2, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles, ffi::Buffer<A, 6> amp, ffi::Buffer<A, 6> dot,
    ffi::Buffer<R, 4> phase, ffi::Buffer<R, 4> phase_dot, ffi::Buffer<R, 4> delay,
    ffi::Buffer<R, 3> wf, analytic_index_t sf, ffi::Buffer<R, 3> gt, analytic_index_t st,
    ffi::Buffer<R, 1> dnu, ffi::Buffer<R, 0> duration, ffi::Buffer<R, 1> freq,
    ffi::Buffer<A, 5> cot, ffi::Result<ffi::Buffer<A, (Mode == 2 || Mode == 4) ? 6 : 5>> out,
    ffi::Result<ffi::Buffer<R, 4>> *phase_bar,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms, std::int64_t) {
  constexpr bool Transpose = Mode == 2 || Mode == 4, Phase = Mode >= 3;
  const AnalyticOptions options{segments, terms, cubic_terms};
  auto status = analytic_validate<Mode>(a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
                                       wf, sf, gt, st, dnu, duration, freq, cot, *out, phase_bar,
                                       options, true);
  if (!status.success()) return completed_future(std::move(status));
  const auto a = amp.dimensions();
  const std::int64_t nb = a1.element_count(), npol = a[4];
  const bool defaults = analytic_is_default(gt.dimensions()[2], options);
  const auto v = analytic_views<T>(a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
                                  wf, sf, gt, st, dnu, duration, freq, options);
  if constexpr (Transpose) {
    Tensor5D<Cplx<T> *> output(reinterpret_cast<Cplx<T> *>(out->typed_data()), a[0], a[1], a[2], a[3], a[4] * a[5]);
    Tensor4D<T *> pb(Phase ? (*phase_bar)->typed_data() : nullptr, a[0], a[1], a[2], a[3]);
    Tensor4D<const Cplx<T> *> g(reinterpret_cast<const Cplx<T> *>(cot.typed_data()), nb, a[2], a[3], npol * npol);
    return parallel_for(pool, a[1], [v, output, pb, g, defaults, npol](auto b, auto e) {
      analytic_specialise(defaults, npol, [&](auto d, auto p) {
        analytic_cpu_transpose<decltype(d)::value, decltype(p)::value, Phase>(b, e, v, g, output, pb);
      });
    });
  } else {
    Tensor4D<Cplx<T> *> output(reinterpret_cast<Cplx<T> *>(out->typed_data()), nb, a[2], a[3], npol * npol);
    constexpr bool JVP = Mode == 1 || Mode == 3;
    return parallel_for(pool, a[2] * a[3], [v, output, defaults, npol](auto b, auto e) {
      analytic_specialise(defaults, npol, [&](auto d, auto p) {
        analytic_cpu_cells<decltype(d)::value, decltype(p)::value, JVP, Phase>(b, e, v, output);
      });
    });
  }
}

#define RI_ANALYTIC_CONTEXT ffi::ThreadPool pool
#define RI_ANALYTIC_CONTEXT_BIND .Ctx<ffi::ThreadPool>()
#define RI_ANALYTIC_CONTEXT_PASS pool
#define RI_ANALYTIC_RETURN ffi::Future
#define RI_ANALYTIC_DISPATCH analytic_cpu_dispatch
#define RI_ANALYTIC_PLATFORM cpu
#include "rfi_analytic_ffi.hpp"

} // namespace ri_kernels
