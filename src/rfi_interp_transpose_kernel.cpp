// CPU transpose (VJP with respect to the data-grid signal) of the data-grid RFI
// visibility. Parallel over antennas, each of which owns its slab of the
// output, so no two chunks write the same element.
//
// For antenna a, source r, cell (f, t) and fine sample (u, v), with S the fine
// samples of the primal and vbar the visibility cotangent:
//
//   G(u, v) = sum_{bl: a1[bl] = a} vbar[bl, f, t] conj(S[a2[bl]](u, v))
//           + sum_{bl: a2[bl] = a} conj(vbar[bl, f, t]) conj(S[a1[bl]](u, v))
//   Q(u, v) = exp(i phi[a](u, v)) G(u, v)
//   H[r, f, t, k, l] = mean_{u, v} w_freq[f, k, u] w_time[t, l, v] Q(u, v)
//   amp_bar[a, r, start_freq[f] + k, start_time[t] + l] += H[r, f, t, k, l]
//
// which is JAX's transpose convention for a map that is complex-linear in the
// first antenna's samples and conjugate-linear in the second's. H is the
// per-cell result -- the transpose of what the forward reads -- and the last
// line gathers it onto the data grid.

#include <complex>
#include <cstdint>
#include <vector>

#include "parallel_for.hpp"
#include "rfi_interp_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ri_kernels {

namespace ffi = xla::ffi;

template <typename T> struct TransposeViews {
  Tensor1D<const int *> a1, a1_sorter, a1_start, a2, a2_sorter, a2_start;
  Tensor4D<const Cplx<T> *> amp;
  Tensor4D<const T *> phase, path;
  Tensor3D<const T *> w_freq, w_time;
  Tensor1D<const int *> start_freq, start_time;
  Tensor1D<const T *> dnu, dt, freqs;
  Tensor3D<const Cplx<T> *> vis_bar;
  Tensor4D<Cplx<T> *> amp_bar;
};

template <typename T>
void transpose_antennas(T scale, std::int64_t ant_begin, std::int64_t ant_end,
                        TransposeViews<T> v) {
  const std::int64_t n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const std::int64_t n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const std::int64_t n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const std::int64_t n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const std::int64_t n_bl = v.a1.shape[0];
  const std::int64_t n_stencil = n_sf * n_st;

  // H for one antenna: (n_rfi, n_freq, n_time, n_sf * n_st).
  std::vector<Cplx<T>> H(n_rfi * n_freq * n_time * n_stencil);
  std::vector<Cplx<T>> Q(n_int_f * n_int_t);
  auto h_at = [&](std::int64_t r, std::int64_t f, std::int64_t t,
                  std::int64_t kl) -> Cplx<T> & {
    return H[kl + n_stencil * (t + n_time * (f + n_freq * r))];
  };

  for (std::int64_t ant = ant_begin; ant < ant_end; ++ant) {
    const std::int64_t first = v.a1_start(ant);
    const std::int64_t first_end = ant == n_ant - 1 ? n_bl : v.a1_start(ant + 1);
    const std::int64_t second = v.a2_start(ant);
    const std::int64_t second_end = ant == n_ant - 1 ? n_bl : v.a2_start(ant + 1);

    for (std::int64_t r = 0; r < n_rfi; ++r) {
      for (std::int64_t f = 0; f < n_freq; ++f) {
        const T *wf = &v.w_freq(f, 0, 0);
        const std::int64_t sf = v.start_freq(f);
        const T freq_f = v.freqs(f);
        for (std::int64_t t = 0; t < n_time; ++t) {
          const T *wt = &v.w_time(t, 0, 0);
          const std::int64_t st = v.start_time(t);

          // Q(u, v): the cotangent of this antenna's fine samples, times its
          // own phase factor.
          for (std::int64_t u = 0; u < n_int_f; ++u) {
            for (std::int64_t vv = 0; vv < n_int_t; ++vv) {
              const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
              Cplx<T> g{0, 0};
              for (std::int64_t p = first; p < first_end; ++p) {
                const std::int64_t bl = v.a1_sorter(p), other = v.a2(bl);
                const auto s = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, other, r, u, vv),
                                    phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, other, r, f, t));
                g = cadd(g, cmul(v.vis_bar(bl, f, t), cconj(s)));
              }
              for (std::int64_t p = second; p < second_end; ++p) {
                const std::int64_t bl = v.a2_sorter(p), other = v.a1(bl);
                const auto s = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, other, r, u, vv),
                                    phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, other, r, f, t));
                g = cadd(g, cconj(cmul(v.vis_bar(bl, f, t), s)));
              }
              const auto e = phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, ant, r, f, t);
              Q[u * n_int_t + vv] = cmul(e, g);
            }
          }

          // The stencil's cotangents: Q pushed back through the weights.
          for (std::int64_t k = 0; k < n_sf; ++k) {
            for (std::int64_t l = 0; l < n_st; ++l) {
              Cplx<T> h{0, 0};
              for (std::int64_t u = 0; u < n_int_f; ++u) {
                const T wk = wf[k * n_int_f + u];
                for (std::int64_t vv = 0; vv < n_int_t; ++vv) {
                  h = cadd(h, cscale(wk * wt[l * n_int_t + vv], Q[u * n_int_t + vv]));
                }
              }
              h_at(r, f, t, k * n_st + l) = cscale(scale, h);
            }
          }
        }
      }

      // Gather: each data-grid element from the cells whose stencils cover it.
      // The contract start[c] <= c < start[c] + n_stencil bounds those to the
      // neighbouring cells.
      for (std::int64_t fp = 0; fp < n_freq; ++fp) {
        for (std::int64_t tp = 0; tp < n_time; ++tp) {
          Cplx<T> sum{0, 0};
          for (std::int64_t f = std::max<std::int64_t>(0, fp - n_sf + 1);
               f < std::min<std::int64_t>(n_freq, fp + n_sf); ++f) {
            const std::int64_t k = fp - v.start_freq(f);
            if (k < 0 || k >= n_sf) continue;
            for (std::int64_t t = std::max<std::int64_t>(0, tp - n_st + 1);
                 t < std::min<std::int64_t>(n_time, tp + n_st); ++t) {
              const std::int64_t l = tp - v.start_time(t);
              if (l < 0 || l >= n_st) continue;
              sum = cadd(sum, h_at(r, f, t, k * n_st + l));
            }
          }
          v.amp_bar(ant, r, fp, tp) = sum;
        }
      }
    }
  }
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future calc_rfi_interp_transpose_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> path,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, path, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return completed_future(ffi::Error::InvalidArgument(
        "Incompatible signal, phase, path, table, or baseline shapes"));
  if (!stencils_cover_their_cells(start_freq.typed_data(), amp.dimensions()[2],
                                  w_freq.dimensions()[1]) ||
      !stencils_cover_their_cells(start_time.typed_data(), amp.dimensions()[3],
                                  w_time.dimensions()[1]))
    return completed_future(ffi::Error::InvalidArgument(
        "Each cell's stencil must lie inside the axis and contain the cell"));
  if (vis_bar.dimensions()[0] != a1.dimensions()[0] ||
      vis_bar.dimensions()[1] != amp.dimensions()[2] ||
      vis_bar.dimensions()[2] != amp.dimensions()[3])
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents"));
  if (!interp_same_shape(*amp_bar, amp))
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the signal cotangent to match the signal"));

  const auto a = amp.dimensions();
  TransposeViews<T> views{
      Tensor1D<const int *>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *>(a1_sorter.typed_data(), a1_sorter.dimensions()[0]),
      Tensor1D<const int *>(a1_start.typed_data(), a1_start.dimensions()[0]),
      Tensor1D<const int *>(a2.typed_data(), a2.dimensions()[0]),
      Tensor1D<const int *>(a2_sorter.typed_data(), a2_sorter.dimensions()[0]),
      Tensor1D<const int *>(a2_start.typed_data(), a2_start.dimensions()[0]),
      Tensor4D<const Cplx<T> *>(
          reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1],
          a[2], a[3]),
      Tensor4D<const T *>(phase.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(path.typed_data(), path.dimensions()[0],
                          path.dimensions()[1], path.dimensions()[2],
                          path.dimensions()[3]),
      Tensor3D<const T *>(w_freq.typed_data(), w_freq.dimensions()[0],
                          w_freq.dimensions()[1], w_freq.dimensions()[2]),
      Tensor3D<const T *>(w_time.typed_data(), w_time.dimensions()[0],
                          w_time.dimensions()[1], w_time.dimensions()[2]),
      Tensor1D<const int *>(start_freq.typed_data(), start_freq.dimensions()[0]),
      Tensor1D<const int *>(start_time.typed_data(), start_time.dimensions()[0]),
      Tensor1D<const T *>(dnu.typed_data(), dnu.dimensions()[0]),
      Tensor1D<const T *>(dt.typed_data(), dt.dimensions()[0]),
      Tensor1D<const T *>(freqs.typed_data(), freqs.dimensions()[0]),
      Tensor3D<const Cplx<T> *>(
          reinterpret_cast<const Cplx<T> *>(vis_bar.typed_data()),
          vis_bar.dimensions()[0], vis_bar.dimensions()[1],
          vis_bar.dimensions()[2]),
      Tensor4D<Cplx<T> *>(reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()),
                          a[0], a[1], a[2], a[3]),
  };
  const T scale = T(1) / T(w_freq.dimensions()[2] * w_time.dimensions()[2]);

  return parallel_for(thread_pool, a[0],
                      [scale, views](std::int64_t begin, std::int64_t end) mutable {
                        transpose_antennas<T>(scale, begin, end, views);
                      });
}

ffi::Future calc_rfi_interp_transpose_cpu_f32_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, interp_amp_f32_t amp, interp_real4_f32_t phase,
    interp_real4_f32_t path, interp_real3_f32_t w_freq,
    interp_index_t start_freq, interp_real3_f32_t w_time,
    interp_index_t start_time, interp_real1_f32_t dnu, interp_real1_f32_t dt,
    interp_real1_f32_t freqs, ffi::BufferR3<ffi::C64> vis_bar,
    ffi::Result<interp_amp_f32_t> amp_bar) {
  return calc_rfi_interp_transpose_cpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

ffi::Future calc_rfi_interp_transpose_cpu_f64_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, interp_amp_f64_t amp, interp_real4_f64_t phase,
    interp_real4_f64_t path, interp_real3_f64_t w_freq,
    interp_index_t start_freq, interp_real3_f64_t w_time,
    interp_index_t start_time, interp_real1_f64_t dnu, interp_real1_f64_t dt,
    interp_real1_f64_t freqs, ffi::BufferR3<ffi::C128> vis_bar,
    ffi::Result<interp_amp_f64_t> amp_bar) {
  return calc_rfi_interp_transpose_cpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_cpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_cpu_f32, calc_rfi_interp_transpose_cpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>().Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<interp_amp_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_cpu_f64, calc_rfi_interp_transpose_cpu_f64_impl,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

} // namespace ri_kernels
