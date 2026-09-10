// GPU forward and JVP of the data-grid RFI visibility. One block per
// (baseline, channel, cell), its threads over the (source, fine sample)
// terms; the two weight rows of the cell live in shared memory, everything
// else is read from global memory per term. The prototype: each baseline
// rebuilds both of its antennas' fine samples itself, so an antenna's samples
// are recomputed once per baseline it is on. A kernel meant to be fast would
// stage each antenna's samples once per cell and reuse them.

#include <cstdint>
#include <limits>
#include <string>

#include "gpu_compat.h"
#include "rfi_interp_common.hpp"
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

template <typename T, typename INT_T> struct InterpViews {
  Tensor1D<const int *, INT_T> a1, a2;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, path;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
};

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
InterpViews<T, INT_T> make_views(
    interp_index_t a1, interp_index_t a2, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> path,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs) {
  const auto a = amp.dimensions();
  return InterpViews<T, INT_T>{
      Tensor1D<const int *, INT_T>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a2.typed_data(), a2.dimensions()[0]),
      Tensor4D<const Cplx<T> *, INT_T>(
          reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1],
          a[2], a[3]),
      Tensor4D<const T *, INT_T>(phase.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *, INT_T>(path.typed_data(), path.dimensions()[0],
                                 path.dimensions()[1], path.dimensions()[2],
                                 path.dimensions()[3]),
      Tensor3D<const T *, INT_T>(w_freq.typed_data(), w_freq.dimensions()[0],
                                 w_freq.dimensions()[1], w_freq.dimensions()[2]),
      Tensor3D<const T *, INT_T>(w_time.typed_data(), w_time.dimensions()[0],
                                 w_time.dimensions()[1], w_time.dimensions()[2]),
      Tensor1D<const int *, INT_T>(start_freq.typed_data(), start_freq.dimensions()[0]),
      Tensor1D<const int *, INT_T>(start_time.typed_data(), start_time.dimensions()[0]),
      Tensor1D<const T *, INT_T>(dnu.typed_data(), dnu.dimensions()[0]),
      Tensor1D<const T *, INT_T>(dt.typed_data(), dt.dimensions()[0]),
      Tensor1D<const T *, INT_T>(freqs.typed_data(), freqs.dimensions()[0]),
  };
}

template <typename T, int BLOCK_SIZE, bool JVP, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_interp_kernel(
    T scale, InterpViews<T, INT_T> v, Tensor4D<const Cplx<T> *, INT_T> amp_dot,
    Tensor3D<Cplx<T> *, INT_T> out) {
  using reduce_t = cub::BlockReduce<T, BLOCK_SIZE>;
  __shared__ typename reduce_t::TempStorage real_storage;
  __shared__ typename reduce_t::TempStorage imag_storage;
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2];
  const INT_T n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_compact = n_rfi * n_int_f * n_int_t;

  // The cell's two weight rows, shared by every term of the block.
  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;

  for (INT_T bl = blockIdx.y; bl < v.a1.shape[0]; bl += gridDim.y) {
    const INT_T ant1 = v.a1(bl), ant2 = v.a2(bl);
    for (INT_T f = blockIdx.z; f < n_freq; f += gridDim.z) {
      for (INT_T i = threadIdx.x; i < n_sf * n_int_f; i += BLOCK_SIZE)
        wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
      const INT_T sf = v.start_freq(f);
      const T freq_f = v.freqs(f);
      for (INT_T t = blockIdx.x; t < n_time; t += gridDim.x) {
        for (INT_T i = threadIdx.x; i < n_st * n_int_t; i += BLOCK_SIZE)
          wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
        __syncthreads();
        const INT_T st = v.start_time(t);
        Cplx<T> sum{0, 0};
        for (INT_T compact = threadIdx.x; compact < n_compact; compact += BLOCK_SIZE) {
          const INT_T vv = compact % n_int_t;
          const INT_T q = compact / n_int_t;
          const INT_T u = q % n_int_f;
          const INT_T r = q / n_int_f;
          const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
          const auto e1 = phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, ant1, r, f, t);
          const auto e2 = phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, ant2, r, f, t);
          const auto s1 = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant1, r, u, vv), e1);
          const auto s2 = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant2, r, u, vv), e2);
          if constexpr (JVP) {
            const auto d1 = cmul(interp_amp(amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant1, r, u, vv), e1);
            const auto d2 = cmul(interp_amp(amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, ant2, r, u, vv), e2);
            sum = cadd(sum, cadd(cmul(d1, cconj(s2)), cmul(s1, cconj(d2))));
          } else {
            sum = cadd(sum, cmul(s1, cconj(s2)));
          }
        }
        sum.re = reduce_t(real_storage).Sum(sum.re);
        sum.im = reduce_t(imag_storage).Sum(sum.im);
        __syncthreads();
        if (threadIdx.x == 0) out(bl, f, t) = cscale(scale, sum);
        // Before the next cell overwrites the time row.
        __syncthreads();
      }
      __syncthreads();
    }
  }
}

template <bool JVP, typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_gpu_dispatch(
    cudaStream_t stream, interp_index_t a1, interp_index_t a2,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<AMP_DT, 4> amp_dot,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> path,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  auto views = make_views<T, INT_T>(a1, a2, amp, phase, path, w_freq, start_freq,
                                    w_time, start_time, dnu, dt, freqs);
  const auto a = amp.dimensions();
  Tensor4D<const Cplx<T> *, INT_T> amp_dot_view(
      reinterpret_cast<const Cplx<T> *>(amp_dot.typed_data()), a[0], a[1],
      a[2], a[3]);
  Tensor3D<Cplx<T> *, INT_T> out_view(
      reinterpret_cast<Cplx<T> *>(out->typed_data()), out->dimensions()[0],
      out->dimensions()[1], out->dimensions()[2]);
  const INT_T n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const INT_T n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  const INT_T n_compact = a[1] * n_int_f * n_int_t;
  const T scale = T(1) / T(n_int_f * n_int_t);
  const std::size_t shared = sizeof(T) * (n_sf * n_int_f + n_st * n_int_t);
  const auto grid = create_clamped_grid(a[3], a1.dimensions()[0], a[2]);
#define LAUNCH_INTERP(B)                                                       \
  rfi_interp_kernel<T, B, JVP, INT_T><<<grid, B, shared, stream>>>(            \
      scale, views, amp_dot_view, out_view)
  if (n_compact <= 32) LAUNCH_INTERP(32);
  else if (n_compact <= 64) LAUNCH_INTERP(64);
  else if (n_compact <= 256) LAUNCH_INTERP(128);
  else LAUNCH_INTERP(256);
#undef LAUNCH_INTERP
  const auto status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

template <bool JVP, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_gpu_impl_tmpl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<AMP_DT, 4> amp_dot, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> path, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, path, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, path, table, or baseline shapes");
  if (JVP && !interp_same_shape(amp_dot, amp))
    return ffi::Error::InvalidArgument(
        "Expected the signal tangent to match the signal");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (amp.element_count() < limit && out->element_count() < limit)
    return calc_rfi_interp_gpu_dispatch<JVP, T, std::int32_t>(
        stream, a1, a2, amp, amp_dot, phase, path, w_freq, start_freq, w_time,
        start_time, dnu, dt, freqs, out);
  return calc_rfi_interp_gpu_dispatch<JVP, T, std::int64_t>(
      stream, a1, a2, amp, amp_dot, phase, path, w_freq, start_freq, w_time,
      start_time, dnu, dt, freqs, out);
}

ffi::Error calc_rfi_interp_gpu_f32_impl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, interp_amp_f32_t amp, interp_real4_f32_t phase,
    interp_real4_f32_t path, interp_real3_f32_t w_freq,
    interp_index_t start_freq, interp_real3_f32_t w_time,
    interp_index_t start_time, interp_real1_f32_t dnu, interp_real1_f32_t dt,
    interp_real1_f32_t freqs, ffi::Result<ffi::BufferR3<ffi::C64>> vis) {
  return calc_rfi_interp_gpu_impl_tmpl<false, ffi::C64, ffi::F32, float>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, vis);
}

ffi::Error calc_rfi_interp_gpu_f64_impl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, interp_amp_f64_t amp, interp_real4_f64_t phase,
    interp_real4_f64_t path, interp_real3_f64_t w_freq,
    interp_index_t start_freq, interp_real3_f64_t w_time,
    interp_index_t start_time, interp_real1_f64_t dnu, interp_real1_f64_t dt,
    interp_real1_f64_t freqs, ffi::Result<ffi::BufferR3<ffi::C128>> vis) {
  return calc_rfi_interp_gpu_impl_tmpl<false, ffi::C128, ffi::F64, double>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, vis);
}

ffi::Error calc_rfi_interp_jvp_gpu_f32_impl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, interp_amp_f32_t amp, interp_amp_f32_t amp_dot,
    interp_real4_f32_t phase, interp_real4_f32_t path,
    interp_real3_f32_t w_freq, interp_index_t start_freq,
    interp_real3_f32_t w_time, interp_index_t start_time,
    interp_real1_f32_t dnu, interp_real1_f32_t dt, interp_real1_f32_t freqs,
    ffi::Result<ffi::BufferR3<ffi::C64>> out) {
  return calc_rfi_interp_gpu_impl_tmpl<true, ffi::C64, ffi::F32, float>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, amp_dot,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, out);
}

ffi::Error calc_rfi_interp_jvp_gpu_f64_impl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, interp_amp_f64_t amp, interp_amp_f64_t amp_dot,
    interp_real4_f64_t phase, interp_real4_f64_t path,
    interp_real3_f64_t w_freq, interp_index_t start_freq,
    interp_real3_f64_t w_time, interp_index_t start_time,
    interp_real1_f64_t dnu, interp_real1_f64_t dt, interp_real1_f64_t freqs,
    ffi::Result<ffi::BufferR3<ffi::C128>> out) {
  return calc_rfi_interp_gpu_impl_tmpl<true, ffi::C128, ffi::F64, double>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, amp_dot,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, out);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_gpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_jvp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_jvp_gpu_f64(XLA_FFI_CallFrame *call_frame);

#define RI_INTERP_INDEX_ARGS                                                   \
  .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()           \
      .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
#define RI_INTERP_TABLE_ARGS(P)                                                \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f32, calc_rfi_interp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f64, calc_rfi_interp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f32, calc_rfi_interp_jvp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f64, calc_rfi_interp_jvp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace ri_kernels
