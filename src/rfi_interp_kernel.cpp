// CPU forward and JVP of the data-grid RFI visibility. Plain scalar loops,
// parallel over baselines: the prototype of the operator, written to be read
// beside rfi_interp_common.hpp rather than to be fast.

#include <complex>
#include <cstdint>
#include <type_traits>

#include "parallel_for.hpp"
#include "rfi_interp_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ri_kernels {

namespace ffi = xla::ffi;

// Value-type snapshot of the inputs: parallel_for runs the body after the
// handler has returned, when the ffi::Buffer arguments are gone.
template <typename T> struct InterpViews {
  Tensor1D<const int *> a1, a2;
  Tensor4D<const Cplx<T> *> amp;
  Tensor4D<const T *> phase, delay;
  Tensor3D<const T *> w_freq, w_time;
  Tensor1D<const int *> start_freq, start_time;
  Tensor1D<const T *> dnu, dt, freqs;
};

template <typename T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
InterpViews<T> make_views(interp_index_t a1, interp_index_t a2,
                          ffi::Buffer<AMP_DT, 4> amp,
                          ffi::Buffer<REAL_DT, 4> phase,
                          ffi::Buffer<REAL_DT, 4> delay,
                          ffi::Buffer<REAL_DT, 3> w_freq,
                          interp_index_t start_freq,
                          ffi::Buffer<REAL_DT, 3> w_time,
                          interp_index_t start_time,
                          ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
                          ffi::Buffer<REAL_DT, 1> freqs) {
  const auto a = amp.dimensions();
  return InterpViews<T>{
      Tensor1D<const int *>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *>(a2.typed_data(), a2.dimensions()[0]),
      Tensor4D<const Cplx<T> *>(
          reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1],
          a[2], a[3]),
      Tensor4D<const T *>(phase.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(delay.typed_data(), delay.dimensions()[0],
                          delay.dimensions()[1], delay.dimensions()[2],
                          delay.dimensions()[3]),
      Tensor3D<const T *>(w_freq.typed_data(), w_freq.dimensions()[0],
                          w_freq.dimensions()[1], w_freq.dimensions()[2]),
      Tensor3D<const T *>(w_time.typed_data(), w_time.dimensions()[0],
                          w_time.dimensions()[1], w_time.dimensions()[2]),
      Tensor1D<const int *>(start_freq.typed_data(), start_freq.dimensions()[0]),
      Tensor1D<const int *>(start_time.typed_data(), start_time.dimensions()[0]),
      Tensor1D<const T *>(dnu.typed_data(), dnu.dimensions()[0]),
      Tensor1D<const T *>(dt.typed_data(), dt.dimensions()[0]),
      Tensor1D<const T *>(freqs.typed_data(), freqs.dimensions()[0]),
  };
}

// The tangents of the JVP: the signal's, and for the full JVP the phase's
// and the delay's (views over the primal buffers otherwise, never read).
template <typename T> struct TangentViews {
  Tensor4D<const Cplx<T> *> amp_dot;
  Tensor4D<const T *> phase_dot, delay_dot;
};

// The forward for JVP = false; for JVP = true the tangent
// B(dS, S) + B(S, dS), with dS interpolated from amp_dot and, when FULL,
// plus i dphi S for the phase and delay tangents.
template <typename T, bool JVP, bool FULL>
void interp_rows(std::int64_t bl_begin, std::int64_t bl_end,
                 InterpViews<T> v, TangentViews<T> d,
                 Tensor3D<Cplx<T> *> out) {
  const std::int64_t n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2];
  const std::int64_t n_time = v.amp.shape[3];
  const std::int64_t n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const std::int64_t n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];

  for (std::int64_t bl = bl_begin; bl < bl_end; ++bl) {
    const std::int64_t ant1 = v.a1(bl), ant2 = v.a2(bl);
    const T scale = T(1) / T(n_int_f * n_int_t);
    for (std::int64_t f = 0; f < n_freq; ++f) {
      const T *wf = &v.w_freq(f, 0, 0);
      const std::int64_t sf = v.start_freq(f);
      const T freq_f = v.freqs(f);
      for (std::int64_t t = 0; t < n_time; ++t) {
        const T *wt = &v.w_time(t, 0, 0);
        const std::int64_t st = v.start_time(t);
        Cplx<T> sum{0, 0};
        for (std::int64_t r = 0; r < n_rfi; ++r) {
          for (std::int64_t u = 0; u < n_int_f; ++u) {
            for (std::int64_t vv = 0; vv < n_int_t; ++vv) {
              const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
              const auto e1 = phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, ant1, r, f, t);
              const auto e2 = phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, ant2, r, f, t);
              const auto s1 = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant1, r, u, vv), e1);
              const auto s2 = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant2, r, u, vv), e2);
              if constexpr (JVP) {
                auto d1 = cmul(interp_amp(d.amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant1, r, u, vv), e1);
                auto d2 = cmul(interp_amp(d.amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant2, r, u, vv), e2);
                if constexpr (FULL) {
                  d1 = cadd(d1, cscale(phase_value(d.phase_dot, d.delay_dot, freq_f, dnu_u, dt_v, ant1, r, f, t), ctimes_i(s1)));
                  d2 = cadd(d2, cscale(phase_value(d.phase_dot, d.delay_dot, freq_f, dnu_u, dt_v, ant2, r, f, t), ctimes_i(s2)));
                }
                sum = cadd(sum, cadd(cmul(d1, cconj(s2)), cmul(s1, cconj(d2))));
              } else {
                sum = cadd(sum, cmul(s1, cconj(s2)));
              }
            }
          }
        }
        out(bl, f, t) = cscale(scale, sum);
      }
    }
  }
}

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

  auto views = make_views<T>(a1, a2, amp, phase, delay, w_freq, start_freq,
                             w_time, start_time, dnu, dt, freqs);
  const auto a = amp.dimensions();
  const auto dd = delay.dimensions();
  const TangentViews<T> tangents{
      Tensor4D<const Cplx<T> *>(reinterpret_cast<const Cplx<T> *>(amp_dot.typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(phase_dot.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *>(delay_dot.typed_data(), dd[0], dd[1], dd[2], dd[3]),
  };
  Tensor3D<Cplx<T> *> out_view(reinterpret_cast<Cplx<T> *>(out->typed_data()),
                               out->dimensions()[0], out->dimensions()[1],
                               out->dimensions()[2]);
  const std::int64_t n_bl = a1.dimensions()[0];

  return parallel_for(thread_pool, n_bl,
                      [views, tangents, out_view](
                          std::int64_t begin, std::int64_t end) mutable {
                        interp_rows<T, JVP, FULL>(begin, end, views, tangents, out_view);
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

} // namespace ri_kernels
