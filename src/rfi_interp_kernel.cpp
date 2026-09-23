// CPU forward and JVP of the data-grid RFI visibility, vectorised with
// Highway and parallel over time cells.
//
// A task takes a run of time cells. For each cell, channel and source it
// builds the fine samples of every antenna once -- the interpolation of the
// cell's stencil and the phase factor, a vector of samples at a time (see
// rfi_interp_cell_inl.hpp) -- and then every baseline's visibility is the dot
// product of its two antennas' samples. An antenna's samples are therefore
// built once per cell rather than once per baseline it is on, which for an
// array of n_ant antennas is a factor of n_ant / 2 less of the expensive part,
// and the part that remains is the dot product, which vectorises.

#include <complex>
#include <cstdint>
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
#define HWY_TARGET_INCLUDE "rfi_interp_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "rfi_interp_cell_inl.hpp"

// The forward for MODE = kNone; otherwise the tangent B(dS, S) + B(S, dS),
// with dS interpolated from amp_dot and, for kFull, plus i dphi S from the
// phase and delay tangents.
template <typename T, JvpMode MODE>
HWY_ATTR void interp_cells_impl(std::int64_t t_begin, std::int64_t t_end,
                                InterpViews<T> v, TangentViews<T> d,
                                Tensor3D<Cplx<T> *> out) {
  constexpr bool JVP = MODE != JvpMode::kNone;
  constexpr bool FULL = MODE == JvpMode::kFull;
  using D = TagType<T>;
  const D d_tag;
  const std::int64_t lanes = hn::Lanes(d_tag);

  const std::int64_t n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const std::int64_t n_freq = v.amp.shape[2];
  const std::int64_t n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const std::int64_t n_int_f = v.w_freq.shape[2];
  const std::int64_t n_s = n_int_f * n_int_t, n_bl = v.a1.shape[0];
  const std::int64_t n_path = v.delay.shape[3];
  const T scale = T(1) / T(n_s);

  CellTables<T> cell;
  SampleBuf<T> S, dS, e_tan;
  std::vector<Cplx<T>> acc(n_bl);
  for (std::int64_t t = t_begin; t < t_end; ++t) {
    const std::int64_t st = v.start_time(t);
    for (std::int64_t f = 0; f < n_freq; ++f) {
      const std::int64_t sf = v.start_freq(f);
      cell.build(v.w_freq, v.w_time, v.dnu, v.dt, v.freqs(f), f, t, n_path, lanes, false);
      S.resize(n_ant, cell.n_s_padded);
      if constexpr (JVP) {
        // The tangent needs the phase factors kept beside the samples.
        dS.resize(n_ant, cell.n_s_padded);
        e_tan.resize(n_ant, cell.n_s_padded);
      }
      for (auto &x : acc) x = Cplx<T>{0, 0};

      for (std::int64_t r = 0; r < n_rfi; ++r) {
        CellSamples(d_tag, cell, v.amp, v.phase, v.delay, sf, st, n_st, n_ant, r, f, t,
                    S, JVP ? &e_tan : nullptr);
        if constexpr (JVP) {
          // dS = interpolate(amp_dot) exp(i phi), plus i dphi S when FULL.
          std::vector<T> a_re(cell.n_s_padded), a_im(cell.n_s_padded);
          std::vector<T> p_re(FULL ? cell.n_s_padded : 0), p_im(FULL ? cell.n_s_padded : 0);
          for (std::int64_t a = 0; a < n_ant; ++a) {
            CellInterpAmp(d_tag, cell, d.amp_dot, sf, st, n_st, a, r, a_re.data(), a_im.data());
            const T *er = e_tan.re_at(a), *ei = e_tan.im_at(a);
            if constexpr (FULL) {
              // dphi, the phase formula on the tangents, needs no centre-phase
              // reduction: CellPhaseFactor gives exp(i phi), so take the phase
              // itself here through the same Horner order.
              CellPhaseValue(d_tag, cell, d.phase_dot, d.delay_dot, a, r, f, t, p_re.data());
            }
            T *ds_re = dS.re_at(a), *ds_im = dS.im_at(a);
            const T *s_re = S.re_at(a), *s_im = S.im_at(a);
            for (std::int64_t k = 0; k < cell.n_s_padded; k += lanes) {
              const auto ar = hn::LoadU(d_tag, a_re.data() + k), ai = hn::LoadU(d_tag, a_im.data() + k);
              const auto br = hn::LoadU(d_tag, er + k), bi = hn::LoadU(d_tag, ei + k);
              auto re = hn::NegMulAdd(ai, bi, hn::Mul(ar, br));
              auto im = hn::MulAdd(ar, bi, hn::Mul(ai, br));
              if constexpr (FULL) {
                // i dphi S = dphi (-Im S, Re S)
                const auto dphi = hn::LoadU(d_tag, p_re.data() + k);
                const auto sr = hn::LoadU(d_tag, s_re + k), si = hn::LoadU(d_tag, s_im + k);
                re = hn::NegMulAdd(dphi, si, re);
                im = hn::MulAdd(dphi, sr, im);
              }
              hn::StoreU(re, d_tag, ds_re + k);
              hn::StoreU(im, d_tag, ds_im + k);
            }
          }
        }
        // Every baseline: a dot product over the cell's samples.
        for (std::int64_t bl = 0; bl < n_bl; ++bl) {
          const std::int64_t a1 = v.a1(bl), a2 = v.a2(bl);
          Cplx<T> term;
          if constexpr (JVP) {
            const Cplx<T> x = CellDotConj(d_tag, cell.n_s_padded, dS.re_at(a1), dS.im_at(a1),
                                          S.re_at(a2), S.im_at(a2));
            const Cplx<T> y = CellDotConj(d_tag, cell.n_s_padded, S.re_at(a1), S.im_at(a1),
                                          dS.re_at(a2), dS.im_at(a2));
            term = cadd(x, y);
          } else {
            term = CellDotConj(d_tag, cell.n_s_padded, S.re_at(a1), S.im_at(a1),
                               S.re_at(a2), S.im_at(a2));
          }
          acc[bl] = cadd(acc[bl], term);
        }
      }
      for (std::int64_t bl = 0; bl < n_bl; ++bl) out(bl, f, t) = cscale(scale, acc[bl]);
    }
  }
}

// Named per-precision entry points, so the per-target dispatch (HWY_EXPORT_*)
// can target each precision separately. The mode is a runtime argument and the
// bodies are specialised on it.
HWY_ATTR void interp_cells_f32(std::int64_t t_begin, std::int64_t t_end, JvpMode mode,
                               InterpViews<float> v, TangentViews<float> d,
                               Tensor3D<Cplx<float> *> out) {
  switch (mode) {
    case JvpMode::kNone: return interp_cells_impl<float, JvpMode::kNone>(t_begin, t_end, v, d, out);
    case JvpMode::kSignal: return interp_cells_impl<float, JvpMode::kSignal>(t_begin, t_end, v, d, out);
    case JvpMode::kFull: return interp_cells_impl<float, JvpMode::kFull>(t_begin, t_end, v, d, out);
  }
}

HWY_ATTR void interp_cells_f64(std::int64_t t_begin, std::int64_t t_end, JvpMode mode,
                               InterpViews<double> v, TangentViews<double> d,
                               Tensor3D<Cplx<double> *> out) {
  switch (mode) {
    case JvpMode::kNone: return interp_cells_impl<double, JvpMode::kNone>(t_begin, t_end, v, d, out);
    case JvpMode::kSignal: return interp_cells_impl<double, JvpMode::kSignal>(t_begin, t_end, v, d, out);
    case JvpMode::kFull: return interp_cells_impl<double, JvpMode::kFull>(t_begin, t_end, v, d, out);
  }
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

template <typename T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error check_inputs(interp_index_t a1, interp_index_t a2,
                        ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
                        ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 3> w_freq,
                        interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
                        interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
                        ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, delay, table, or baseline shapes");
  if (!stencils_cover_their_cells(start_freq.typed_data(), amp.dimensions()[2],
                                  w_freq.dimensions()[1]) ||
      !stencils_cover_their_cells(start_time.typed_data(), amp.dimensions()[3],
                                  w_time.dimensions()[1]))
    return ffi::Error::InvalidArgument(
        "Each cell's stencil must lie inside the axis and contain the cell");
  return ffi::Error::Success();
}

template <bool JVP, bool FULL, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future calc_rfi_interp_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tile_pairs, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<AMP_DT, 4> amp_dot, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> phase_dot, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 4> delay_dot, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  auto error = check_inputs<T>(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs);
  if (!error.success()) return completed_future(std::move(error));
  if (JVP && !interp_same_shape(amp_dot, amp))
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the signal tangent to match the signal"));
  if (FULL && !(interp_same_shape(phase_dot, phase) && interp_same_shape(delay_dot, delay)))
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the phase and delay tangents to match the phase and delay"));

  // Snapshot of everything the chunks need. These views own their extents by
  // value, so they stay valid after the handler returns - unlike the
  // ffi::Buffer arguments, which point into the FFI call frame. See
  // parallel_for().
  const auto a = amp.dimensions();
  const auto dd = delay.dimensions();
  const InterpViews<T> views{
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
  };
  const TangentViews<T> tangents{
      Tensor4D<const Cplx<T> *>(reinterpret_cast<const Cplx<T> *>(amp_dot.typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(phase_dot.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(delay_dot.typed_data(), dd[0], dd[1], dd[2], dd[3]),
  };
  Tensor3D<Cplx<T> *> out_view(reinterpret_cast<Cplx<T> *>(out->typed_data()),
                               out->dimensions()[0], out->dimensions()[1],
                               out->dimensions()[2]);
  const std::int64_t n_time = a[3];
  constexpr JvpMode mode = FULL ? JvpMode::kFull : (JVP ? JvpMode::kSignal : JvpMode::kNone);

  return parallel_for(thread_pool, n_time,
                      [views, tangents, out_view](
                          std::int64_t begin, std::int64_t end) mutable {
                        if constexpr (std::is_same_v<T, float>) {
                          RI_KERNELS_EXPORT_AND_DISPATCH_T(interp_cells_f32)
                          (begin, end, mode, views, tangents, out_view);
                        } else {
                          RI_KERNELS_EXPORT_AND_DISPATCH_T(interp_cells_f64)
                          (begin, end, mode, views, tangents, out_view);
                        }
                      });
}

// ---- forward ----------------------------------------------------------------

ffi::Future calc_rfi_interp_cpu_f32_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f32_t amp, interp_real4_f32_t phase,
    interp_real4_f32_t delay, interp_real3_f32_t w_freq,
    interp_index_t start_freq, interp_real3_f32_t w_time,
    interp_index_t start_time, interp_real1_f32_t dnu, interp_real1_f32_t dt,
    interp_real1_f32_t freqs, ffi::Result<ffi::BufferR3<ffi::C64>> vis) {
  return calc_rfi_interp_cpu_impl_tmpl<false, false, ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp, amp,
      phase, phase, delay, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, vis);
}

ffi::Future calc_rfi_interp_cpu_f64_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f64_t amp, interp_real4_f64_t phase,
    interp_real4_f64_t delay, interp_real3_f64_t w_freq,
    interp_index_t start_freq, interp_real3_f64_t w_time,
    interp_index_t start_time, interp_real1_f64_t dnu, interp_real1_f64_t dt,
    interp_real1_f64_t freqs, ffi::Result<ffi::BufferR3<ffi::C128>> vis) {
  return calc_rfi_interp_cpu_impl_tmpl<false, false, ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp, amp,
      phase, phase, delay, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, vis);
}

// ---- JVP --------------------------------------------------------------------

ffi::Future calc_rfi_interp_jvp_cpu_f32_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f32_t amp, interp_amp_f32_t amp_dot,
    interp_real4_f32_t phase, interp_real4_f32_t delay,
    interp_real3_f32_t w_freq, interp_index_t start_freq,
    interp_real3_f32_t w_time, interp_index_t start_time,
    interp_real1_f32_t dnu, interp_real1_f32_t dt, interp_real1_f32_t freqs,
    ffi::Result<ffi::BufferR3<ffi::C64>> out) {
  return calc_rfi_interp_cpu_impl_tmpl<true, false, ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp,
      amp_dot, phase, phase, delay, delay, w_freq, start_freq, w_time, start_time, dnu, dt,
      freqs, out);
}

ffi::Future calc_rfi_interp_jvp_cpu_f64_impl(
    ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, interp_amp_f64_t amp, interp_amp_f64_t amp_dot,
    interp_real4_f64_t phase, interp_real4_f64_t delay,
    interp_real3_f64_t w_freq, interp_index_t start_freq,
    interp_real3_f64_t w_time, interp_index_t start_time,
    interp_real1_f64_t dnu, interp_real1_f64_t dt, interp_real1_f64_t freqs,
    ffi::Result<ffi::BufferR3<ffi::C128>> out) {
  return calc_rfi_interp_cpu_impl_tmpl<true, false, ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, tile_pairs, amp,
      amp_dot, phase, phase, delay, delay, w_freq, start_freq, w_time, start_time, dnu, dt,
      freqs, out);
}

// ---- full JVP: tangents on the signal, the phase and the delay -------------

#define RI_INTERP_FULL_JVP_CPU_ENTRY(NAME, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, OUT_T) \
  ffi::Future NAME##_impl(                                                       \
      ffi::ThreadPool thread_pool, interp_index_t a1, interp_index_t a1_sorter,   \
      interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,       \
      interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,                     \
      ffi::BufferR2<ffi::S32> tile_pairs, AMP_T amp, AMP_T amp_dot, R4 phase,     \
      R4 phase_dot, R4 delay, R4 delay_dot, R3 w_freq, interp_index_t start_freq, \
      R3 w_time, interp_index_t start_time, R1 dnu, R1 dt, R1 freqs,              \
      ffi::Result<OUT_T> out) {                                                   \
    return calc_rfi_interp_cpu_impl_tmpl<true, true, AMP_DT, REAL_DT, T>(         \
        thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair,      \
        tile_pairs, amp, amp_dot, phase, phase_dot, delay, delay_dot, w_freq,     \
        start_freq, w_time, start_time, dnu, dt, freqs, out);                     \
  }

RI_INTERP_FULL_JVP_CPU_ENTRY(calc_rfi_interp_full_jvp_cpu_f32, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, ffi::BufferR3<ffi::C64>)
RI_INTERP_FULL_JVP_CPU_ENTRY(calc_rfi_interp_full_jvp_cpu_f64, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, ffi::BufferR3<ffi::C128>)

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_jvp_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_jvp_cpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_cpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_jvp_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_jvp_cpu_f64(XLA_FFI_CallFrame *call_frame);

#define RI_INTERP_INDEX_ARGS                                                   \
  .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()           \
      .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()       \
      .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
#define RI_INTERP_TABLE_ARGS(P)                                                \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_cpu_f32, calc_rfi_interp_cpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_cpu_f64, calc_rfi_interp_cpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_cpu_f32, calc_rfi_interp_jvp_cpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_cpu_f64, calc_rfi_interp_jvp_cpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());
// The full JVP: amp, amp_dot, phase, phase_dot, delay, delay_dot, then the tables.
#define RI_INTERP_FULL_TABLE_ARGS(P)                                           \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                 \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_full_jvp_cpu_f32, calc_rfi_interp_full_jvp_cpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_FULL_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_full_jvp_cpu_f64, calc_rfi_interp_full_jvp_cpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::ThreadPool>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_FULL_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());

#endif // HWY_ONCE

} // namespace ri_kernels
