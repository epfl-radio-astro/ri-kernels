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
// One task owns one source and one run of output time cells. It builds that
// source's fine samples and stencil cotangents for the cells whose stencils
// reach its output run -- its own cells plus the n_st - 1 on either side,
// recomputed rather than shared, which is a few per cent of a run and is what
// keeps this a single parallel pass -- and then gathers them onto its own slab
// of the data grid. Tasks write disjoint slabs of every output, so there is no
// accumulation across tasks and the result is deterministic. The per-task
// buffer holds one source's cells only, a few MB.

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

// One task's stencil cotangents: (n_ant, n_freq, n_tc, n_stencil), for one
// source and the cells the task builds.
template <typename T>
HWY_INLINE Cplx<T> &h_at(Cplx<T> *H, std::int64_t a, std::int64_t f, std::int64_t t_local,
                         std::int64_t kl, std::int64_t n_freq, std::int64_t n_tc,
                         std::int64_t n_stencil) {
  return H[((a * n_freq + f) * n_tc + t_local) * n_stencil + kl];
}

// The stencil cotangents H of source `r` on cells [t_lo, t_hi), laid out
// (n_ant, n_freq, t_hi - t_lo, n_slots), and, when FULL, the phase's and the
// delay's cotangents written straight out for the cells in [tp_lo, tp_hi).
template <typename T, bool FULL>
HWY_ATTR void transpose_cells_impl(TransposeViews<T> v, Cplx<T> *H, std::int64_t r,
                                   std::int64_t t_lo, std::int64_t t_hi,
                                   std::int64_t tp_lo, std::int64_t tp_hi) {
  using D = TagType<T>;
  const D d_tag;
  const std::int64_t lanes = hn::Lanes(d_tag);

  const std::int64_t n_ant = v.amp.shape[0];
  const std::int64_t n_freq = v.amp.shape[2];
  const std::int64_t n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const std::int64_t n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const std::int64_t n_s = n_int_f * n_int_t, n_bl = v.a1.shape[0];
  const std::int64_t n_stencil = n_sf * n_st, n_path = v.delay.shape[3];
  const T inv = T(1) / T(n_s);

  const std::int64_t n_tc = t_hi - t_lo;
  CellTables<T> cell;
  SampleBuf<T> S, E, G;
  std::vector<T> pc;
  for (std::int64_t t = t_lo; t < t_hi; ++t) {
    const std::int64_t t_local = t - t_lo;
    const bool own = t >= tp_lo && t < tp_hi;  // the cells this task reports
    const std::int64_t st = v.start_time(t);
    for (std::int64_t f = 0; f < n_freq; ++f) {
      const std::int64_t sf = v.start_freq(f);
      cell.build(v.w_freq, v.w_time, v.dnu, v.dt, v.freqs(f), f, t, n_path, lanes, FULL);
      S.resize(n_ant, cell.n_s_padded);
      E.resize(n_ant, cell.n_s_padded);
      G.resize(n_ant, cell.n_s_padded);
      if constexpr (FULL) pc.assign(cell.n_s_padded, T(0));

      {
        // one source per task
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
            if (own) {
              // The phase and delay cotangents need no gather: they are done
              // with this cell, and the cell is this task's to report.
              v.phase_bar(a, r, f, t) = CellSumReal(d_tag, cell.n_s_padded, pc.data());
              if (f == 0)
                for (std::int64_t j = 0; j < n_path; ++j) v.delay_bar(a, r, t, j) = 0;
              for (std::int64_t j = 0; j < n_path; ++j)
                v.delay_bar(a, r, t, j) += CellWeightedSumReal(
                    d_tag, cell.n_s_padded, cell.dcoef.data() + j * cell.n_s_padded, pc.data());
            }
          }
          CellMulInPlace(d_tag, cell.n_s_padded, E.re_at(a), E.im_at(a), G.re_at(a), G.im_at(a));
          for (std::int64_t kl = 0; kl < n_stencil; ++kl)
            h_at(H, a, f, t_local, kl, n_freq, n_tc, n_stencil) =
                CellWeightedSum(d_tag, cell.n_s_padded, cell.weight.data() + kl * cell.n_s_padded,
                                G.re_at(a), G.im_at(a));
        }
      }
    }
  }
}

// The gather of one task's cells onto its own slab of the data grid: for each
// antenna and each output cell in [tp_lo, tp_hi), the cells whose stencils
// cover it, in a fixed order. Every output element belongs to exactly one
// task, so nothing accumulates across tasks.
template <typename T>
HWY_ATTR void transpose_gather_impl(TransposeViews<T> v, const Cplx<T> *H, std::int64_t r,
                                    std::int64_t t_lo, std::int64_t t_hi,
                                    std::int64_t tp_lo, std::int64_t tp_hi) {
  const std::int64_t n_ant = v.amp.shape[0], n_freq = v.amp.shape[2];
  const std::int64_t n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const std::int64_t n_stencil = n_sf * n_st, n_tc = t_hi - t_lo;
  for (std::int64_t a = 0; a < n_ant; ++a) {
    for (std::int64_t fp = 0; fp < n_freq; ++fp) {
      for (std::int64_t tp = tp_lo; tp < tp_hi; ++tp) {
        Cplx<T> sum{0, 0};
        for (std::int64_t f = std::max<std::int64_t>(0, fp - n_sf + 1);
             f < std::min<std::int64_t>(n_freq, fp + n_sf); ++f) {
          const std::int64_t k = fp - v.start_freq(f);
          if (k < 0 || k >= n_sf) continue;
          for (std::int64_t t = std::max<std::int64_t>(t_lo, tp - n_st + 1);
               t < std::min<std::int64_t>(t_hi, tp + n_st); ++t) {
            const std::int64_t l = tp - v.start_time(t);
            if (l < 0 || l >= n_st) continue;
            sum = cadd(sum, h_at(const_cast<Cplx<T> *>(H), a, f, t - t_lo, k * n_st + l,
                                 n_freq, n_tc, n_stencil));
          }
        }
        v.amp_bar(a, r, fp, tp) = sum;
      }
    }
  }
}

// Named per-precision entry points, so the per-target dispatch (HWY_EXPORT_*)
// can target each precision separately. `full` is a runtime argument and the
// bodies are specialised on it.
#define RI_INTERP_TRANSPOSE_TARGETS(SUFFIX, T)                                        \
  HWY_ATTR void transpose_slab_##SUFFIX(bool full, TransposeViews<T> v, Cplx<T> *H,   \
                                        std::int64_t r, std::int64_t t_lo,            \
                                        std::int64_t t_hi, std::int64_t tp_lo,        \
                                        std::int64_t tp_hi) {                         \
    if (full) transpose_cells_impl<T, true>(v, H, r, t_lo, t_hi, tp_lo, tp_hi);       \
    else transpose_cells_impl<T, false>(v, H, r, t_lo, t_hi, tp_lo, tp_hi);           \
    transpose_gather_impl<T>(v, H, r, t_lo, t_hi, tp_lo, tp_hi);                      \
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
  const std::int64_t n_st = w_time.dimensions()[1];
  const std::int64_t n_stencil = w_freq.dimensions()[1] * n_st;

  // Tasks are (source, run of output cells) pairs, each writing a disjoint
  // slab of every output. A run also builds the n_st - 1 cells on either side
  // of itself, the only work two tasks repeat, so runs want to be long; its
  // stencil cotangents are a buffer the worker allocates for itself, so they
  // also want to fit in a budget shared between the workers. Between the two,
  // enough runs to keep every worker busy and no more.
  const std::int64_t workers = std::max<std::int64_t>(thread_pool.num_threads(), 1);
  const std::size_t per_cell = sizeof(Cplx<T>) * std::size_t(n_ant) * n_freq * n_stencil;
  const std::size_t budget = std::max<std::size_t>(4u << 20, (256u << 20) / workers);
  std::int64_t run_len = std::int64_t(budget / per_cell) - 2 * (n_st - 1);
  run_len = std::min<std::int64_t>(std::max<std::int64_t>(1, run_len), n_time);
  // No more runs than the workers need: fewer, longer runs repeat less work.
  const std::int64_t runs_wanted = std::max<std::int64_t>(1, (workers + n_rfi - 1) / n_rfi);
  run_len = std::max<std::int64_t>(run_len, 1);
  run_len = std::min<std::int64_t>(run_len, std::max<std::int64_t>(1, (n_time + runs_wanted - 1) / runs_wanted));
  const std::int64_t n_runs = (n_time + run_len - 1) / run_len;
  const std::size_t per_task = per_cell * std::size_t(run_len + 2 * (n_st - 1));

  return parallel_for(
      thread_pool, n_rfi * n_runs,
      [views, per_task, n_runs, run_len, n_time, n_st](std::int64_t begin,
                                                       std::int64_t end) mutable {
        std::vector<Cplx<T>> H(per_task / sizeof(Cplx<T>));
        for (std::int64_t item = begin; item < end; ++item) {
          const std::int64_t r = item / n_runs, run = item % n_runs;
          const std::int64_t tp_lo = run * run_len;
          const std::int64_t tp_hi = std::min<std::int64_t>(n_time, tp_lo + run_len);
          if (tp_lo >= tp_hi) continue;
          const std::int64_t t_lo = std::max<std::int64_t>(0, tp_lo - n_st + 1);
          const std::int64_t t_hi = std::min<std::int64_t>(n_time, tp_hi + n_st - 1);
          if constexpr (std::is_same_v<T, float>) {
            RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_slab_f32)
            (FULL, views, H.data(), r, t_lo, t_hi, tp_lo, tp_hi);
          } else {
            RI_KERNELS_EXPORT_AND_DISPATCH_T(transpose_slab_f64)
            (FULL, views, H.data(), r, t_lo, t_hi, tp_lo, tp_hi);
          }
        }
      });
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
