// Each CPU task stages one cell's antenna coefficients and then contracts its
// baselines. The transpose owns a source, so its stencil scatters cannot race
// with another task and no atomics or per-thread data-grid copies are needed.
#include <vector>
#include "parallel_for.hpp"
#include "rfi_analytic_common.hpp"
#include "visibility.h"

namespace ri_kernels {

template <bool Default, bool JVP, typename T>
void analytic_cpu_cells(std::int64_t begin, std::int64_t end,
    AnalyticViews<T> v, Tensor3D<Cplx<T> *> out) {
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2], nb = v.a1.shape[0];
  std::vector<Cplx<T>> coefficients(na * nm), dots(JVP ? na * nm : 0);
  for (auto cell = begin; cell < end; ++cell) {
    const auto f = cell / nt, t = cell % nt;
    for (std::int64_t bl = 0; bl < nb; ++bl) out(bl, f, t) = {0, 0};
    for (std::int64_t r = 0; r < nr; ++r) {
      for (std::int64_t u = 0; u < nu; ++u) {
        for (std::int64_t m = 0; m < nm; ++m) {
          for (std::int64_t ant = 0; ant < na; ++ant) {
            coefficients[m * na + ant] = analytic_coefficient(v, v.amp, ant, r, f, t, u, m);
            if constexpr (JVP)
              dots[m * na + ant] = analytic_coefficient(v, v.amp_dot, ant, r, f, t, u, m);
          }
        }
        for (std::int64_t bl = 0; bl < nb; ++bl) {
          const auto p = v.a1(bl), q = v.a2(bl);
          Cplx<double> h[AnalyticStorage<Default>::product];
          analytic_pair_weights<Default>(v, p, q, r, f, t, u, h);
          const auto z = analytic_contract<Default>(coefficients.data() + p, coefficients.data() + q,
              JVP ? dots.data() + p : nullptr, JVP ? dots.data() + q : nullptr, h, int(nm), na, JVP);
          out(bl, f, t) = cadd(out(bl, f, t), cscale(T(1) / T(nu), z));
        }
      }
    }
  }
}

template <bool Default, typename T>
void analytic_cpu_transpose(std::int64_t begin, std::int64_t end,
    AnalyticViews<T> v, Tensor3D<const Cplx<T> *> cot, Tensor4D<Cplx<T> *> out) {
  const auto na = v.amp.shape[0], nf = v.amp.shape[2], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2];
  std::vector<Cplx<T>> coefficients(na * nm), bars(na * nm);
  for (auto r = begin; r < end; ++r) {
    for (std::int64_t ant = 0; ant < na; ++ant)
      for (std::int64_t f = 0; f < nf; ++f)
        for (std::int64_t t = 0; t < nt; ++t) out(ant, r, f, t) = {0, 0};
    for (std::int64_t f = 0; f < nf; ++f) {
      for (std::int64_t t = 0; t < nt; ++t) {
        for (std::int64_t u = 0; u < nu; ++u) {
          for (std::int64_t m = 0; m < nm; ++m)
            for (std::int64_t ant = 0; ant < na; ++ant) {
              coefficients[m * na + ant] = analytic_coefficient(v, v.amp, ant, r, f, t, u, m);
              bars[m * na + ant] = {0, 0};
            }
          for (std::int64_t bl = 0; bl < v.a1.shape[0]; ++bl) {
            const auto p = v.a1(bl), q = v.a2(bl);
            Cplx<double> h[AnalyticStorage<Default>::product];
            analytic_pair_weights<Default>(v, p, q, r, f, t, u, h);
            const auto g = cscale(T(1) / T(nu), cot(bl, f, t));
            for (std::int64_t j = 0; j < nm; ++j)
              for (std::int64_t k = 0; k < nm; ++k) {
                const auto w = cmul(g, analytic_cast<T>(h[j + k]));
                bars[j * na + p] = cadd(bars[j * na + p], cmul(w, cconj(coefficients[k * na + q])));
                bars[k * na + q] = cadd(bars[k * na + q], cconj(cmul(w, coefficients[j * na + p])));
              }
          }
          for (std::int64_t k = 0; k < v.wf.shape[1]; ++k)
            for (std::int64_t l = 0; l < v.gt.shape[1]; ++l)
              for (std::int64_t ant = 0; ant < na; ++ant) {
                Cplx<T> z{0, 0};
                for (std::int64_t m = 0; m < nm; ++m)
                  z = cadd(z, cscale(v.wf(f, k, u) * v.gt(t, l, m), bars[m * na + ant]));
                auto &dest = out(ant, r, v.sf(f) + k, v.st(t) + l);
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
    interp_index_t a1, interp_index_t a2, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles, ffi::Buffer<A, 4> amp, ffi::Buffer<A, 4> dot,
    ffi::Buffer<R, 4> phase, ffi::Buffer<R, 4> delay, ffi::Buffer<R, 3> wf,
    interp_index_t sf, ffi::Buffer<R, 3> gt, interp_index_t st,
    ffi::Buffer<R, 1> dnu, ffi::Buffer<R, 0> duration, ffi::Buffer<R, 1> freq,
    ffi::Buffer<A, 3> cot, ffi::Result<ffi::Buffer<A, Mode == 2 ? 4 : 3>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  const AnalyticOptions options{segments, terms, cubic_terms};
  auto status = analytic_validate(a1, a2, pair, tiles, amp, dot, phase, delay,
                                 wf, sf, gt, st, dnu, duration, freq, options, true);
  if (!status.success()) return completed_future(std::move(status));
  const auto a = amp.dimensions();
  const bool defaults = analytic_is_default(gt.dimensions()[2], options);
  const auto v = analytic_views<T>(a1, a2, pair, tiles, amp, dot, phase, delay,
                                  wf, sf, gt, st, dnu, duration, freq, options);
  if constexpr (Mode == 2) {
    if (!interp_same_shape(amp, *out) || cot.dimensions()[0] != a1.element_count() ||
        cot.dimensions()[1] != a[2] || cot.dimensions()[2] != a[3])
      return completed_future(ffi::Error::InvalidArgument("Invalid analytic transpose output or cotangent shape"));
    Tensor4D<Cplx<T> *> output(reinterpret_cast<Cplx<T> *>(out->typed_data()), a[0], a[1], a[2], a[3]);
    Tensor3D<const Cplx<T> *> g(reinterpret_cast<const Cplx<T> *>(cot.typed_data()), a1.element_count(), a[2], a[3]);
    return parallel_for(pool, a[1], [v, output, g, defaults](auto b, auto e) {
      if (defaults) analytic_cpu_transpose<true>(b, e, v, g, output);
      else analytic_cpu_transpose<false>(b, e, v, g, output);
    });
  } else {
    if (out->dimensions()[0] != a1.element_count() || out->dimensions()[1] != a[2] || out->dimensions()[2] != a[3])
      return completed_future(ffi::Error::InvalidArgument("Invalid analytic visibility output shape"));
    Tensor3D<Cplx<T> *> output(reinterpret_cast<Cplx<T> *>(out->typed_data()), a1.element_count(), a[2], a[3]);
    return parallel_for(pool, a[2] * a[3], [v, output, defaults](auto b, auto e) {
      if (defaults) analytic_cpu_cells<true, Mode == 1>(b, e, v, output);
      else analytic_cpu_cells<false, Mode == 1>(b, e, v, output);
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
