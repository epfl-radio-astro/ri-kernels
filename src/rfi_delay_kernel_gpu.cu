#include <cstdint>
#include <limits>
#include <string>

#include "gpu_compat.h"
#include "rfi_delay_common.hpp"
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_delay_kernel(
    T scale, Tensor1D<const int *, INT_T> a1,
    Tensor1D<const int *, INT_T> a2,
    Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> amp,
    Tensor4D<const T *, INT_T> delay, Tensor2D<const T *, INT_T> freq,
    Tensor3D<typename gpu_complex_traits<T>::complex_t *, INT_T> vis,
    INT_T n_rfi, INT_T n_int_f, INT_T n_int_t) {
  using traits = gpu_complex_traits<T>;
  using complex_t = typename traits::complex_t;
  using reduce_t = cub::BlockReduce<T, BLOCK_SIZE>;
  __shared__ typename reduce_t::TempStorage real_storage;
  __shared__ typename reduce_t::TempStorage imag_storage;
  extern __shared__ unsigned char dynamic_shared[];
  T *angular_frequency = reinterpret_cast<T *>(dynamic_shared);

  const INT_T n_compact = n_rfi * n_int_t;
  for (INT_T bl = blockIdx.y; bl < a1.shape[0]; bl += gridDim.y) {
    const INT_T ant1 = a1(bl);
    const INT_T ant2 = a2(bl);
    for (INT_T f = blockIdx.z; f < amp.shape[1]; f += gridDim.z) {
      for (INT_T fi = threadIdx.x; fi < n_int_f; fi += BLOCK_SIZE)
        angular_frequency[fi] = two_pi<T>() * freq(f, fi);
      __syncthreads();
      for (INT_T t = blockIdx.x; t < amp.shape[2]; t += gridDim.x) {
        complex_t sum{0, 0};
        for (INT_T compact = threadIdx.x; compact < n_compact;
             compact += BLOCK_SIZE) {
          const INT_T ti = compact % n_int_t;
          const INT_T r = compact / n_int_t;
          const T delay_diff = delay(ant1, t, r, ti) -
                               delay(ant2, t, r, ti);
          INT_T red = ti + n_int_t * n_int_f * r;
          for (INT_T fi = 0; fi < n_int_f; ++fi, red += n_int_t) {
            complex_t e;
            traits::sincos_(angular_frequency[fi] * delay_diff, &e.y, &e.x);
            const auto value = traits::mul(
                traits::mul(amp(ant1, f, t, red),
                            traits::conj(amp(ant2, f, t, red))),
                e);
            sum = traits::add(sum, value);
          }
        }
        sum.x = reduce_t(real_storage).Sum(sum.x);
        sum.y = reduce_t(imag_storage).Sum(sum.y);
        __syncthreads();
        if (threadIdx.x == 0) {
          vis(bl, f, t) = complex_t{scale * sum.x, scale * sum.y};
        }
      }
      // The reduction synchronizes between time samples. This additional
      // barrier protects angular_frequency when a clamped grid loops over f.
      __syncthreads();
    }
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error calc_rfi_delay_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a2, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> vis) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;
  Tensor1D<const int *, INT_T> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2v(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const complex_t *, INT_T> av(
      reinterpret_cast<const complex_t *>(amp.typed_data()), amp.dimensions()[0],
      amp.dimensions()[1], amp.dimensions()[2],
      amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *, INT_T> dv(delay.typed_data(), delay.dimensions()[0],
                                delay.dimensions()[1], delay.dimensions()[2],
                                delay.dimensions()[3]);
  Tensor2D<const T *, INT_T> fv(freq.typed_data(), freq.dimensions()[0],
                                freq.dimensions()[1]);
  Tensor3D<complex_t *, INT_T> vv(
      reinterpret_cast<complex_t *>(vis->typed_data()), vis->dimensions()[0],
      vis->dimensions()[1], vis->dimensions()[2]);
  const INT_T nr = amp.dimensions()[3], nf = amp.dimensions()[4];
  const INT_T nt = amp.dimensions()[5], ncompact = nr * nt;
  const T scale = T(1) / T(nf * nt);
  const auto grid = create_clamped_grid(av.shape[2], a1v.shape[0], av.shape[1]);
#define LAUNCH_DELAY_VIS(B)                                                     \
  rfi_delay_kernel<T, B, INT_T><<<grid, B, sizeof(T) * nf, stream>>>(          \
      scale, a1v, a2v, av, dv, fv, vv, nr, nf, nt)
  if (ncompact <= 32) LAUNCH_DELAY_VIS(32);
  else if (ncompact <= 64) LAUNCH_DELAY_VIS(64);
  else if (ncompact <= 256) LAUNCH_DELAY_VIS(128);
  else LAUNCH_DELAY_VIS(256);
#undef LAUNCH_DELAY_VIS
  const auto status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_delay_gpu_impl_tmpl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> vis) {
  if (!shapes_are_valid(a1, a2, amp, delay, freq))
    return ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (amp.element_count() < limit && delay.element_count() < limit &&
      vis->element_count() < limit)
    return calc_rfi_delay_gpu_dispatch<T, std::int32_t>(stream, a1, a2, amp,
                                                        delay, freq, vis);
  return calc_rfi_delay_gpu_dispatch<T, std::int64_t>(stream, a1, a2, amp,
                                                      delay, freq, vis);
}

ffi::Error calc_rfi_delay_gpu_f32_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C64, 6> amp,
    ffi::Buffer<ffi::F32, 4> delay, ffi::Buffer<ffi::F32, 2> freq,
    ffi::Result<ffi::BufferR3<ffi::C64>> vis) {
  return calc_rfi_delay_gpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
      freq, vis);
}

ffi::Error calc_rfi_delay_gpu_f64_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C128, 6> amp,
    ffi::Buffer<ffi::F64, 4> delay, ffi::Buffer<ffi::F64, 2> freq,
    ffi::Result<ffi::BufferR3<ffi::C128>> vis) {
  return calc_rfi_delay_gpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
      freq, vis);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_gpu_f32,
                              calc_rfi_delay_gpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<delay_amp_f32_t>()
                                  .Arg<delay_real4_f32_t>()
                                  .Arg<delay_real2_f32_t>()
                                  .Ret<ffi::BufferR3<ffi::C64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_gpu_f64,
                              calc_rfi_delay_gpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<delay_amp_f64_t>()
                                  .Arg<delay_real4_f64_t>()
                                  .Arg<delay_real2_f64_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace ri_kernels
