// GPU transpose (VJP with respect to the data-grid signal) of the data-grid
// RFI visibility, in two kernels. The first works one (antenna, channel, cell)
// per block, one source at a time: its threads form Q(u, v), the cotangent of
// the antenna's fine samples times its phase factor, in shared memory, then
// contract Q with the cell's weight rows into the stencil's cotangents H, which
// go to scratch memory. The second gathers each data-grid element of the
// output from the cells whose stencils cover it. Every output element is
// written by exactly one thread, so no atomics and a deterministic sum. See
// rfi_interp_transpose_kernel.cpp for the formulas.

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

template <typename T, typename INT_T> struct TransposeViews {
  Tensor1D<const int *, INT_T> a1, a1_sorter, a1_start, a2, a2_sorter, a2_start;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, path;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
  Tensor3D<const Cplx<T> *, INT_T> vis_bar;
  // (n_ant, n_rfi, n_freq, n_time, n_sf * n_st): the per-cell result.
  Tensor5D<Cplx<T> *, INT_T> H;
  Tensor4D<Cplx<T> *, INT_T> amp_bar;
};

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_interp_transpose_cells(T scale, TransposeViews<T, INT_T> v) {
  extern __shared__ unsigned char dynamic_shared[];
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_bl = v.a1.shape[0];
  const INT_T n_samples = n_int_f * n_int_t, n_stencil = n_sf * n_st;

  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  Cplx<T> *Q = reinterpret_cast<Cplx<T> *>(wt + n_st * n_int_t);

  for (INT_T ant = blockIdx.y; ant < n_ant; ant += gridDim.y) {
    const INT_T first = v.a1_start(ant);
    const INT_T first_end = ant == n_ant - 1 ? n_bl : v.a1_start(ant + 1);
    const INT_T second = v.a2_start(ant);
    const INT_T second_end = ant == n_ant - 1 ? n_bl : v.a2_start(ant + 1);
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
        for (INT_T r = 0; r < n_rfi; ++r) {
          for (INT_T i = threadIdx.x; i < n_samples; i += BLOCK_SIZE) {
            const INT_T u = i / n_int_t, vv = i % n_int_t;
            const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
            Cplx<T> g{0, 0};
            for (INT_T p = first; p < first_end; ++p) {
              const INT_T bl = v.a1_sorter(p), other = v.a2(bl);
              const auto s = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, other, r, u, vv),
                                  phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, other, r, f, t));
              g = cadd(g, cmul(v.vis_bar(bl, f, t), cconj(s)));
            }
            for (INT_T p = second; p < second_end; ++p) {
              const INT_T bl = v.a2_sorter(p), other = v.a1(bl);
              const auto s = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, other, r, u, vv),
                                  phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, other, r, f, t));
              g = cadd(g, cconj(cmul(v.vis_bar(bl, f, t), s)));
            }
            Q[i] = cmul(phase_factor(v.phase, v.path, freq_f, dnu_u, dt_v, ant, r, f, t), g);
          }
          __syncthreads();
          for (INT_T kl = threadIdx.x; kl < n_stencil; kl += BLOCK_SIZE) {
            const INT_T k = kl / n_st, l = kl % n_st;
            Cplx<T> h{0, 0};
            for (INT_T u = 0; u < n_int_f; ++u) {
              const T wk = wf[k * n_int_f + u];
              for (INT_T vv = 0; vv < n_int_t; ++vv)
                h = cadd(h, cscale(wk * wt[l * n_int_t + vv], Q[u * n_int_t + vv]));
            }
            v.H(ant, r, f, t, kl) = cscale(scale, h);
          }
          __syncthreads();
        }
      }
      __syncthreads();
    }
  }
}

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_interp_transpose_gather(TransposeViews<T, INT_T> v) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T n_out = n_ant * n_rfi * n_freq * n_time;
  for (INT_T i = blockIdx.x * INT_T(BLOCK_SIZE) + threadIdx.x; i < n_out;
       i += INT_T(gridDim.x) * BLOCK_SIZE) {
    const INT_T tp = i % n_time;
    const INT_T q = i / n_time;
    const INT_T fp = q % n_freq;
    const INT_T q2 = q / n_freq;
    const INT_T r = q2 % n_rfi;
    const INT_T ant = q2 / n_rfi;
    Cplx<T> sum{0, 0};
    const INT_T f_lo = fp - n_sf + 1 > 0 ? fp - n_sf + 1 : 0;
    const INT_T f_hi = fp + n_sf < n_freq ? fp + n_sf : n_freq;
    const INT_T t_lo = tp - n_st + 1 > 0 ? tp - n_st + 1 : 0;
    const INT_T t_hi = tp + n_st < n_time ? tp + n_st : n_time;
    for (INT_T f = f_lo; f < f_hi; ++f) {
      const INT_T k = fp - v.start_freq(f);
      if (k < 0 || k >= n_sf) continue;
      for (INT_T t = t_lo; t < t_hi; ++t) {
        const INT_T l = tp - v.start_time(t);
        if (l < 0 || l >= n_st) continue;
        sum = cadd(sum, v.H(ant, r, f, t, k * n_st + l));
      }
    }
    v.amp_bar(ant, r, fp, tp) = sum;
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> path, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::BufferR3<AMP_DT> vis_bar, ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar) {
  const auto a = amp.dimensions();
  const INT_T n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const INT_T n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  const INT_T n_stencil = n_sf * n_st, n_samples = n_int_f * n_int_t;

  const std::size_t h_bytes =
      sizeof(Cplx<T>) * std::size_t(a[0]) * a[1] * a[2] * a[3] * n_stencil;
  auto h_mem = scratch.Allocate(h_bytes, alignof(Cplx<T>));
  if (!h_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the "
                                "per-cell cotangents");

  TransposeViews<T, INT_T> views{
      Tensor1D<const int *, INT_T>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a1_sorter.typed_data(), a1_sorter.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a1_start.typed_data(), a1_start.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a2.typed_data(), a2.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a2_sorter.typed_data(), a2_sorter.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a2_start.typed_data(), a2_start.dimensions()[0]),
      Tensor4D<const Cplx<T> *, INT_T>(
          reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1], a[2], a[3]),
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
      Tensor3D<const Cplx<T> *, INT_T>(
          reinterpret_cast<const Cplx<T> *>(vis_bar.typed_data()),
          vis_bar.dimensions()[0], vis_bar.dimensions()[1], vis_bar.dimensions()[2]),
      Tensor5D<Cplx<T> *, INT_T>(reinterpret_cast<Cplx<T> *>(*h_mem), a[0],
                                 a[1], a[2], a[3], n_stencil),
      Tensor4D<Cplx<T> *, INT_T>(
          reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), a[0], a[1], a[2], a[3]),
  };
  const T scale = T(1) / T(n_int_f * n_int_t);
  const std::size_t shared = sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) +
                             sizeof(Cplx<T>) * n_samples;
  const auto grid = create_clamped_grid(a[3], a[0], a[2]);
  const INT_T n_work = n_samples > n_stencil ? n_samples : n_stencil;
#define LAUNCH_CELLS(B)                                                        \
  rfi_interp_transpose_cells<T, B, INT_T><<<grid, B, shared, stream>>>(scale, views)
  if (n_work <= 32) LAUNCH_CELLS(32);
  else if (n_work <= 64) LAUNCH_CELLS(64);
  else if (n_work <= 128) LAUNCH_CELLS(128);
  else LAUNCH_CELLS(256);
#undef LAUNCH_CELLS
  auto status = cudaGetLastError();
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));

  constexpr int gather_block = 256;
  const std::int64_t n_out = std::int64_t(a[0]) * a[1] * a[2] * a[3];
  const auto gather_grid =
      create_clamped_grid(int((n_out + gather_block - 1) / gather_block), 1, 1);
  rfi_interp_transpose_gather<T, gather_block, INT_T>
      <<<gather_grid, gather_block, 0, stream>>>(views);
  status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

// The allocator is move-only: the entry points take it by value from the
// binding and lend it down the call chain by reference.
template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_transpose_gpu_impl_tmpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> path, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::BufferR3<AMP_DT> vis_bar, ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, path, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, path, table, or baseline shapes");
  if (vis_bar.dimensions()[0] != a1.dimensions()[0] ||
      vis_bar.dimensions()[1] != amp.dimensions()[2] ||
      vis_bar.dimensions()[2] != amp.dimensions()[3])
    return ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents");
  if (!interp_same_shape(*amp_bar, amp))
    return ffi::Error::InvalidArgument(
        "Expected the signal cotangent to match the signal");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  const std::int64_t h_count = amp.element_count() *
                               w_freq.dimensions()[1] * w_time.dimensions()[1];
  // use 32 bit indexing if possible
  if (h_count < limit && vis_bar.element_count() < limit)
    return calc_rfi_interp_transpose_gpu_dispatch<T, std::int32_t>(
        stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
        phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
        vis_bar, amp_bar);
  return calc_rfi_interp_transpose_gpu_dispatch<T, std::int64_t>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f32_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, interp_amp_f32_t amp,
    interp_real4_f32_t phase, interp_real4_f32_t path,
    interp_real3_f32_t w_freq, interp_index_t start_freq,
    interp_real3_f32_t w_time, interp_index_t start_time,
    interp_real1_f32_t dnu, interp_real1_f32_t dt, interp_real1_f32_t freqs,
    ffi::BufferR3<ffi::C64> vis_bar, ffi::Result<interp_amp_f32_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f64_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, interp_amp_f64_t amp,
    interp_real4_f64_t phase, interp_real4_f64_t path,
    interp_real3_f64_t w_freq, interp_index_t start_freq,
    interp_real3_f64_t w_time, interp_index_t start_time,
    interp_real1_f64_t dnu, interp_real1_f64_t dt, interp_real1_f64_t freqs,
    ffi::BufferR3<ffi::C128> vis_bar, ffi::Result<interp_amp_f64_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      phase, path, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_gpu_f32, calc_rfi_interp_transpose_gpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>().Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<interp_amp_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_gpu_f64, calc_rfi_interp_transpose_gpu_f64_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

} // namespace gpu
} // namespace ri_kernels
