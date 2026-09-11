// CPU transpose (VJP with respect to the data-grid signal, and in the full
// variant the phase and the delay) of the data-grid RFI visibility, vectorised
// with Highway.
//
// For antenna a, source r, cell (f, t) and fine sample (u, v), with S the fine
// samples of the primal and vbar the visibility cotangent:
//
//   G(u, v) = sum_{bl: a1[bl] = a} vbar[bl, f, t] conj(S[a2[bl]](u, v)) / n_samples
//           + sum_{bl: a2[bl] = a} conj(vbar[bl, f, t]) conj(S[a1[bl]](u, v)) / n_samples
//   Q(u, v) = exp(i phi[a](u, v)) G(u, v)
//   H[r, f, t, k, l] = sum_{u, v} w_freq[f, k, u] w_time[t, l, v] Q(u, v)
//   amp_bar[a, r, f', t'] = sum over the cells (f, t) whose stencils cover (f', t')
//                           of H[r, f, t, f' - start_freq[f], t' - start_time[t]]
//
// and, in the full variant, phi_bar = -Im(S G) per sample, summed over the
// cell for the phase and weighted by d phi / d delay[j] for the delay.
//
// Work is in phases of time cells: a chunk of cells in parallel, one time cell
// per task, each building the fine samples of every antenna once per channel
// and source (see rfi_interp_cell_inl.hpp) and scattering every baseline's
// cotangent onto them; then, in parallel over antennas and sources, the gather
// of the chunk's H onto the data grid, each output element written by one task
// in a fixed order. The chunk's H is bounded (256 MB at most) and the result is
// deterministic.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include "parallel_for.hpp"
#include "rfi_interp_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_interp_transpose_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "rfi_interp_cell_inl.hpp"

// H of a chunk of time cells: (n_ant, n_rfi, n_freq, n_tc, n_slots), the
// n_sf * n_st stencil cotangents and, in the full variant, the phase's and the
// n_path delay cotangents after them.
template <typename T>
HWY_INLINE Cplx<T> &h_at(Cplx<T> *H, std::int64_t a, std::int64_t r, std::int64_t f,
                         std::int64_t t_local, std::int64_t slot, std::int64_t n_rfi,
                         std::int64_t n_freq, std::int64_t n_tc, std::int64_t n_slots) {
  return H[(((a * n_rfi + r) * n_freq + f) * n_tc + t_local) * n_slots + slot];
}

template <typename T, bool FULL>
HWY_ATTR void transpose_cells_impl(std::int64_t begin, std::int64_t end, TransposeViews<T> v,
                                   Cplx<T> *H, std::int64_t t0, std::int64_t n_tc) {
  using D = TagType<T>;
  const D d_tag;
  const std::int64_t lanes = hn::Lanes(d_tag);

  const std::int64_t n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const std::int64_t n_freq = v.amp.shape[2];
  const std::int64_t n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const std::int64_t n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const std::int64_t n_s = n_int_f * n_int_t, n_bl = v.a1.shape[0];
  const std::int64_t n_stencil = n_sf * n_st, n_path = v.delay.shape[3];
  const std::int64_t n_slots = FULL ? n_stencil + 1 + n_path : n_stencil;
  const T inv = T(1) / T(n_s);

  CellTables<T> cell;
  SampleBuf<T> S, E, G;
  std::vector<T> pc;
  for (std::int64_t t_local = begin; t_local < end; ++t_local) {
    const std::int64_t t = t0 + t_local;
    const std::int64_t st = v.start_time(t);
    for (std::int64_t f = 0; f < n_freq; ++f) {
      const std::int64_t sf = v.start_freq(f);
      cell.build(v.w_freq, v.w_time, v.dnu, v.dt, v.freqs(f), f, t, n_path, lanes, FULL);
      S.resize(n_ant, cell.n_s_padded);
      E.resize(n_ant, cell.n_s_padded);
      G.resize(n_ant, cell.n_s_padded);
      if constexpr (FULL) pc.assign(cell.n_s_padded, T(0));

      for (std::int64_t r = 0; r < n_rfi; ++r) {
        CellSamples(d_tag, cell, v.amp, v.phase, v.delay, sf, st, n_st, n_ant, r, f, t, S, &E);
        std::fill(G.re.begin(), G.re.end(), T(0));
        std::fill(G.im.begin(), G.im.end(), T(0));
        // Each baseline's cotangent onto its two antennas; an autocorrelation
        // gets both terms, its cotangent and the conjugate.
        for (std::int64_t bl = 0; bl < n_bl; ++bl) {
          const std::int64_t a1 = v.a1(bl), a2 = v.a2(bl);
          CellScatterCotangent(d_tag, cell.n_s_padded, cscale(inv, v.vis_bar(bl, f, t)),
                               S.re_at(a1), S.im_at(a1), S.re_at(a2), S.im_at(a2),
                               G.re_at(a1), G.im_at(a1), G.re_at(a2), G.im_at(a2));
        }
        // Per antenna: the phase terms from G, then G turned into
        // Q = exp(i phi) G in place and pushed through the weights onto H.
        for (std::int64_t a = 0; a < n_ant; ++a) {
          if constexpr (FULL) {
            CellPhaseCotangent(d_tag, cell.n_s_padded, S.re_at(a), S.im_at(a),
                               G.re_at(a), G.im_at(a), pc.data());
            h_at(H, a, r, f, t_local, n_stencil, n_rfi, n_freq, n_tc, n_slots) =
                Cplx<T>{CellSumReal(d_tag, cell.n_s_padded, pc.data()), T(0)};
            for (std::int64_t j = 0; j < n_path; ++j)
              h_at(H, a, r, f, t_local, n_stencil + 1 + j, n_rfi, n_freq, n_tc, n_slots) =
                  Cplx<T>{CellWeightedSumReal(d_tag, cell.n_s_padded,
                                              cell.dcoef.data() + j * cell.n_s_padded, pc.data()),
                          T(0)};
          }
          CellMulInPlace(d_tag, cell.n_s_padded, E.re_at(a), E.im_at(a), G.re_at(a), G.im_at(a));
          for (std::int64_t kl = 0; kl < n_stencil; ++kl)
            h_at(H, a, r, f, t_local, kl, n_rfi, n_freq, n_tc, n_slots) =
                CellWeightedSum(d_tag, cell.n_s_padded, cell.weight.data() + kl * cell.n_s_padded,
                                G.re_at(a), G.im_at(a));
        }
      }
    }
  }
}

// The gather of a chunk: items are (antenna, source) pairs; each adds the
// chunk's cells onto every data-grid element their stencils cover, and in
// the full variant writes the phase's and the delay's cotangents of the
// chunk's cells.
template <typename T, bool FULL>
HWY_ATTR void transpose_gather_impl(std::int64_t begin, std::int64_t end, TransposeViews<T> v,
                                    Cplx<T> *H, std::int64_t t0, std::int64_t n_tc) {
  const std::int64_t n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const std::int64_t n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const std::int64_t n_stencil = n_sf * n_st, n_path = v.delay.shape[3];
  const std::int64_t n_slots = FULL ? n_stencil + 1 + n_path : n_stencil;
  const std::int64_t tp_lo = std::max<std::int64_t>(0, t0 - n_st + 1);
  const std::int64_t tp_hi = std::min<std::int64_t>(n_time, t0 + n_tc + n_st - 1);
  for (std::int64_t item = begin; item < end; ++item) {
    const std::int64_t a = item / n_rfi, r = item % n_rfi;
    for (std::int64_t fp = 0; fp < n_freq; ++fp) {
      for (std::int64_t tp = tp_lo; tp < tp_hi; ++tp) {
        Cplx<T> sum{0, 0};
        for (std::int64_t f = std::max<std::int64_t>(0, fp - n_sf + 1);
             f < std::min<std::int64_t>(n_freq, fp + n_sf); ++f) {
          const std::int64_t k = fp - v.start_freq(f);
          if (k < 0 || k >= n_sf) continue;
          for (std::int64_t t = std::max<std::int64_t>(t0, tp - n_st + 1);
               t < std::min<std::int64_t>(t0 + n_tc, tp + n_st); ++t) {
            const std::int64_t l = tp - v.start_time(t);
            if (l < 0 || l >= n_st) continue;
            sum = cadd(sum, h_at(H, a, r, f, t - t0, k * n_st + l, n_rfi, n_freq, n_tc, n_slots));
          }
        }
        v.amp_bar(a, r, fp, tp) = cadd(v.amp_bar(a, r, fp, tp), sum);
      }
    }
    if constexpr (FULL) {
      for (std::int64_t t_local = 0; t_local < n_tc; ++t_local) {
        for (std::int64_t j = 0; j < n_path; ++j) v.delay_bar(a, r, t0 + t_local, j) = 0;
        for (std::int64_t f = 0; f < n_freq; ++f) {
          v.phase_bar(a, r, f, t0 + t_local) = h_at(H, a, r, f, t_local, n_stencil, n_rfi, n_freq, n_tc, n_slots).re;
          for (std::int64_t j = 0; j < n_path; ++j)
            v.delay_bar(a, r, t0 + t_local, j) += h_at(H, a, r, f, t_local, n_stencil + 1 + j, n_rfi, n_freq, n_tc, n_slots).re;
        }
      }
    }
  }
}

// Named per-precision entry points, so the per-target dispatch (HWY_EXPORT_*)
// can target each precision separately. `full` is a runtime argument and the
// bodies are specialised on it.
#define RI_INTERP_TRANSPOSE_TARGETS(SUFFIX, T)                                       \
  HWY_ATTR void transpose_cells_##SUFFIX(std::int64_t begin, std::int64_t end,       \
                                         bool full, TransposeViews<T> v, Cplx<T> *H, \
                                         std::int64_t t0, std::int64_t n_tc) {       \
    if (full) transpose_cells_impl<T, true>(begin, end, v, H, t0, n_tc);             \
    else transpose_cells_impl<T, false>(begin, end, v, H, t0, n_tc);                 \
  }                                                                                  \
  HWY_ATTR void transpose_gather_##SUFFIX(std::int64_t begin, std::int64_t end,      \
                                          bool full, TransposeViews<T> v, Cplx<T> *H,\
                                          std::int64_t t0, std::int64_t n_tc) {      \
    if (full) transpose_gather_impl<T, true>(begin, end, v, H, t0, n_tc);            \
    else transpose_gather_impl<T, false>(begin, end, v, H, t0, n_tc);                \
  }

RI_INTERP_TRANSPOSE_TARGETS(f32, float)
RI_INTERP_TRANSPOSE_TARGETS(f64, double)

} // namespace HWY_NAMESPACE

#if HWY_ONCE

template <bool FULL, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future calc_rfi_interp_transpose_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tile_pairs, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> *phase_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> *delay_bar) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return completed_future(ffi::Error::InvalidArgument(
        "Incompatible signal, phase, delay, table, or baseline shapes"));
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
  if (FULL && !(interp_same_shape(**phase_bar, phase) && interp_same_shape(**delay_bar, delay)))
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the phase and delay cotangents to match the phase and delay"));

  const auto a = amp.dimensions();
  const auto dd = delay.dimensions();
  TransposeViews<T> views{
      Tensor1D<const int *>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *>(a2.typed_data(), a2.dimensions()[0]),
      Tensor4D<const Cplx<T> *>(reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(phase.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(delay.typed_data(), dd[0], dd[1], dd[2], dd[3]),
      Tensor3D<const T *>(w_freq.typed_data(), w_freq.dimensions()[0], w_freq.dimensions()[1], w_freq.dimensions()[2]),
      Tensor3D<const T *>(w_time.typed_data(), w_time.dimensions()[0], w_time.dimensions()[1], w_time.dimensions()[2]),
      Tensor1D<const int *>(start_freq.typed_data(), start_freq.dimensions()[0]),
      Tensor1D<const int *>(start_time.typed_data(), start_time.dimensions()[0]),
      Tensor1D<const T *>(dnu.typed_data(), dnu.dimensions()[0]),
      Tensor1D<const T *>(dt.typed_data(), dt.dimensions()[0]),
      Tensor1D<const T *>(freqs.typed_data(), freqs.dimensions()[0]),
      Tensor3D<const Cplx<T> *>(reinterpret_cast<const Cplx<T> *>(vis_bar.typed_data()),
                                vis_bar.dimensions()[0], vis_bar.dimensions()[1], vis_bar.dimensions()[2]),
      Tensor4D<Cplx<T> *>(reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<T *>(FULL ? (*phase_bar)->typed_data() : nullptr, a[0], a[1], a[2], a[3]),
      Tensor4D<T *>(FULL ? (*delay_bar)->typed_data() : nullptr, dd[0], dd[1], dd[2], dd[3]),
  };
  const std::int64_t n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const std::int64_t n_stencil = w_freq.dimensions()[1] * w_time.dimensions()[1];
  const std::int64_t n_slots = FULL ? n_stencil + 1 + dd[3] : n_stencil;
  // Time chunk: as many cells as keep H within 256 MB, at least one.
  const std::size_t per_cell = sizeof(Cplx<T>) * std::size_t(n_ant) * n_rfi * n_freq * n_slots;
  std::int64_t n_tc = std::int64_t((256u * 1024 * 1024) / per_cell);
  n_tc = std::max<std::int64_t>(1, std::min<std::int64_t>(n_tc, n_time));
  auto H = std::make_shared<std::vector<Cplx<T>>>(std::size_t(n_ant) * n_rfi * n_freq * n_tc * n_slots);
  Cplx<T> *H_ptr = H->data();
  std::fill_n(reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), amp.element_count(), Cplx<T>{0, 0});

  using Body = std::function<void(std::int64_t, std::int64_t)>;
  std::vector<std::pair<std::int64_t, Body>> phases;
  for (std::int64_t t0 = 0; t0 < n_time; t0 += n_tc) {
    const std::int64_t cells = std::min<std::int64_t>(n_tc, n_time - t0);
    phases.emplace_back(cells, [views, H, H_ptr, t0, cells](std::int64_t b, std::int64_t e) {
      if constexpr (std::is_same_v<T, float>) {
        RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_cells_f32)(b, e, FULL, views, H_ptr, t0, cells);
      } else {
        RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_cells_f64)(b, e, FULL, views, H_ptr, t0, cells);
      }
    });
    phases.emplace_back(n_ant * n_rfi, [views, H, H_ptr, t0, cells](std::int64_t b, std::int64_t e) {
      if constexpr (std::is_same_v<T, float>) {
        RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_gather_f32)(b, e, FULL, views, H_ptr, t0, cells);
      } else {
        RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_gather_f64)(b, e, FULL, views, H_ptr, t0, cells);
      }
    });
  }
  return parallel_phases(thread_pool, std::move(phases));
}


ffi::Future calc_rfi_interp_transpose_cpu_f32_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f32_t amp, interp_real4_f32_t phase,
    interp_real4_f32_t delay, interp_real3_f32_t w_freq,
    interp_index_t start_freq, interp_real3_f32_t w_time,
    interp_index_t start_time, interp_real1_f32_t dnu, interp_real1_f32_t dt,
    interp_real1_f32_t freqs, ffi::BufferR3<ffi::C64> vis_bar,
    ffi::Result<interp_amp_f32_t> amp_bar) {
  return calc_rfi_interp_transpose_cpu_impl_tmpl<false, ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp,
      phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar, nullptr, nullptr);
}

ffi::Future calc_rfi_interp_transpose_cpu_f64_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f64_t amp, interp_real4_f64_t phase,
    interp_real4_f64_t delay, interp_real3_f64_t w_freq,
    interp_index_t start_freq, interp_real3_f64_t w_time,
    interp_index_t start_time, interp_real1_f64_t dnu, interp_real1_f64_t dt,
    interp_real1_f64_t freqs, ffi::BufferR3<ffi::C128> vis_bar,
    ffi::Result<interp_amp_f64_t> amp_bar) {
  return calc_rfi_interp_transpose_cpu_impl_tmpl<false, ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp,
      phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar, nullptr, nullptr);
}

// ---- full transpose: the signal's, the phase's and the delay's cotangents ----

#define RI_INTERP_FULL_TRANSPOSE_CPU_ENTRY(NAME, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, VBAR_T) \
  ffi::Future NAME##_impl(                                                          \
      ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,      \
      interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,          \
      interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,                        \
      ffi::BufferR2<ffi::S32> tile_pairs, AMP_T amp, R4 phase, R4 delay, R3 w_freq,  \
      interp_index_t start_freq, R3 w_time, interp_index_t start_time, R1 dnu,       \
      R1 dt, R1 freqs, VBAR_T vis_bar, ffi::Result<AMP_T> amp_bar,                   \
      ffi::Result<R4> phase_bar, ffi::Result<R4> delay_bar) {                        \
    return calc_rfi_interp_transpose_cpu_impl_tmpl<true, AMP_DT, REAL_DT, T>(        \
        thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair,         \
        tile_pairs, amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu,  \
        dt, freqs, vis_bar, amp_bar, &phase_bar, &delay_bar);                        \
  }

RI_INTERP_FULL_TRANSPOSE_CPU_ENTRY(calc_rfi_interp_full_transpose_cpu_f32, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, ffi::BufferR3<ffi::C64>)
RI_INTERP_FULL_TRANSPOSE_CPU_ENTRY(calc_rfi_interp_full_transpose_cpu_f64, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, ffi::BufferR3<ffi::C128>)

extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_transpose_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_transpose_cpu_f64(XLA_FFI_CallFrame *call_frame);

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
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
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
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_full_transpose_cpu_f32, calc_rfi_interp_full_transpose_cpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>().Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<interp_amp_f32_t>().Ret<interp_real4_f32_t>().Ret<interp_real4_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_full_transpose_cpu_f64, calc_rfi_interp_full_transpose_cpu_f64_impl,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>().Ret<interp_real4_f64_t>().Ret<interp_real4_f64_t>());

#endif // HWY_ONCE

} // namespace ri_kernels
