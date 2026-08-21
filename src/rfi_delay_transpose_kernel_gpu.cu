#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

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

// Sums across the lanes of one logical thread group, leaving the result in
// lane 0. A warp-level collective keeps the reduction independent of the other
// groups in the block, which matters because the grid-stride compact loop below
// lets groups run different trip counts. For power-of-two group sizes cub picks
// the shuffle implementation, so TempStorage is empty and costs no shared
// memory.
template <int GROUP_SIZE, int BLOCK_SIZE, typename T>
__device__ inline T warp_sum(T value) {
  if constexpr (GROUP_SIZE == 1) {
    return value;
  } else {
    using reduce_t = cub::WarpReduce<T, GROUP_SIZE>;
    __shared__ typename reduce_t::TempStorage storage[BLOCK_SIZE / GROUP_SIZE];
    return reduce_t(storage[threadIdx.x / GROUP_SIZE]).Sum(value);
  }
}

// One logical thread group owns a compact
// (antenna, time, source, integration-time) delay element. Its lanes split the
// coarse/fine frequencies, write distinct amplitude cotangents, and reduce the
// delay cotangent in registers. This retains the atomics-free ownership of the
// compact output without serializing all frequencies in one thread. fp64 uses
// a narrower group to balance frequency parallelism against strided traffic.
template <typename T, int BLOCK_SIZE, int GROUP_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_delay_transpose_kernel(
    T scale, Tensor1D<const int *, INT_T> a1,
    Tensor1D<const int *, INT_T> a1_sorter,
    Tensor1D<const int *, INT_T> a1_start,
    Tensor1D<const int *, INT_T> a2,
    Tensor1D<const int *, INT_T> a2_sorter,
    Tensor1D<const int *, INT_T> a2_start,
    Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> amp,
    Tensor4D<const T *, INT_T> delay, Tensor2D<const T *, INT_T> freq,
    Tensor3D<const typename gpu_complex_traits<T>::complex_t *, INT_T> vis_bar,
    Tensor4D<typename gpu_complex_traits<T>::complex_t *, INT_T> amp_bar,
    Tensor4D<T *, INT_T> delay_bar, INT_T n_rfi, INT_T n_int_f,
    INT_T n_int_t) {
  using traits = gpu_complex_traits<T>;
  using complex_t = typename traits::complex_t;
  constexpr int groups_per_block = BLOCK_SIZE / GROUP_SIZE;

  const INT_T lane = threadIdx.x % GROUP_SIZE;
  const INT_T group = threadIdx.x / GROUP_SIZE;
  const INT_T compact_size = amp.shape[2] * n_rfi * n_int_t;
  const INT_T stride = gridDim.x * groups_per_block;
  const INT_T n_frequency_terms = amp.shape[1] * n_int_f;

  for (INT_T ant = blockIdx.y; ant < amp.shape[0]; ant += gridDim.y) {
    const INT_T first = a1_start(ant);
    const INT_T first_end = ant == amp.shape[0] - 1 ? a1.shape[0]
                                                     : a1_start(ant + 1);
    const INT_T second = a2_start(ant);
    const INT_T second_end = ant == amp.shape[0] - 1 ? a2.shape[0]
                                                      : a2_start(ant + 1);
    for (INT_T compact = blockIdx.x * groups_per_block + group;
         compact < compact_size; compact += stride) {
      const INT_T ti = compact % n_int_t;
      const INT_T q = compact / n_int_t;
      const INT_T r = q % n_rfi;
      const INT_T t = q / n_rfi;
      const T my_delay = delay(ant, t, r, ti);
      T delay_sum = 0;
      INT_T f = 0;
      INT_T fi = lane;
      while (fi >= n_int_f) {
        fi -= n_int_f;
        ++f;
      }
      for (INT_T frequency_term = lane; frequency_term < n_frequency_terms;
           frequency_term += GROUP_SIZE) {
        const INT_T red = ti + n_int_t * (fi + n_int_f * r);
        const auto my_amp = amp(ant, f, t, red);
        const T omega = two_pi<T>() * freq(f, fi);
        complex_t amp_sum{0, 0};
        T phase_sum = 0;
        for (INT_T p = first; p < first_end; ++p) {
          const INT_T bl = a1_sorter(p);
          const INT_T other = a2(bl);
          const T phase =
              omega * (my_delay - delay(other, t, r, ti));
          complex_t e;
          traits::sincos_(phase, &e.y, &e.x);
          const auto v = traits::mul(
              traits::mul(vis_bar(bl, f, t),
                          traits::conj(amp(other, f, t, red))),
              e);
          amp_sum = traits::add(amp_sum, v);
          phase_sum -= v.y * my_amp.x + v.x * my_amp.y;
        }
        for (INT_T p = second; p < second_end; ++p) {
          const INT_T bl = a2_sorter(p);
          const INT_T other = a1(bl);
          const T phase =
              omega * (delay(other, t, r, ti) - my_delay);
          complex_t e;
          traits::sincos_(phase, &e.y, &e.x);
          const auto v = traits::conj(traits::mul(
              traits::mul(vis_bar(bl, f, t), amp(other, f, t, red)), e));
          amp_sum = traits::add(amp_sum, v);
          phase_sum -= v.x * my_amp.y + v.y * my_amp.x;
        }
        amp_bar(ant, f, t, red) =
            complex_t{scale * amp_sum.x, scale * amp_sum.y};
        delay_sum += omega * phase_sum;
        fi += GROUP_SIZE;
        while (fi >= n_int_f) {
          fi -= n_int_f;
          ++f;
        }
      }
      delay_sum = warp_sum<GROUP_SIZE, BLOCK_SIZE>(delay_sum);
      if (lane == 0)
        delay_bar(ant, t, r, ti) = scale * delay_sum;
    }
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error calc_rfi_delay_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 6>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> delay_bar) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;
  Tensor1D<const int *, INT_T> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1sv(a1_sorter.typed_data(), a1_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1bv(a1_start.typed_data(), a1_start.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2v(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2sv(a2_sorter.typed_data(), a2_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2bv(a2_start.typed_data(), a2_start.dimensions()[0]);
  const INT_T nr = amp.dimensions()[3], nf = amp.dimensions()[4];
  const INT_T nt = amp.dimensions()[5], nred = nr * nf * nt;
  Tensor4D<const complex_t *, INT_T> av(
      reinterpret_cast<const complex_t *>(amp.typed_data()), amp.dimensions()[0],
      amp.dimensions()[1], amp.dimensions()[2], nred);
  Tensor4D<const T *, INT_T> dv(delay.typed_data(), delay.dimensions()[0],
                                delay.dimensions()[1], delay.dimensions()[2],
                                delay.dimensions()[3]);
  Tensor2D<const T *, INT_T> fv(freq.typed_data(), freq.dimensions()[0],
                                freq.dimensions()[1]);
  Tensor3D<const complex_t *, INT_T> g(
      reinterpret_cast<const complex_t *>(vis_bar.typed_data()),
      vis_bar.dimensions()[0], vis_bar.dimensions()[1], vis_bar.dimensions()[2]);
  Tensor4D<complex_t *, INT_T> ab(
      reinterpret_cast<complex_t *>(amp_bar->typed_data()), amp.dimensions()[0],
      amp.dimensions()[1], amp.dimensions()[2], nred);
  Tensor4D<T *, INT_T> db(delay_bar->typed_data(), delay.dimensions()[0],
                           delay.dimensions()[1], delay.dimensions()[2],
                           delay.dimensions()[3]);
  constexpr int block_size = 128;
  const INT_T compact = amp.dimensions()[2] * nr * nt;
  auto launch = [&](auto group_size_c) {
    constexpr int group_size = decltype(group_size_c)::value;
    constexpr int groups_per_block = block_size / group_size;
    const INT_T blocks =
        (compact + groups_per_block - 1) / groups_per_block;
    const auto grid = create_clamped_grid(blocks, amp.dimensions()[0], 1);
    rfi_delay_transpose_kernel<T, block_size, group_size, INT_T>
        <<<grid, block_size, 0, stream>>>(
            T(1) / T(nf * nt), a1v, a1sv, a1bv, a2v, a2sv, a2bv, av, dv, fv,
            g, ab, db, nr, nf, nt);
  };

  if constexpr (std::is_same_v<T, double>) {
    // At large fp64 sizes, strided frequency-parallel loads cost more than the
    // extra parallelism saves. Retain one compact output per thread there.
    const std::int64_t problem_size = std::int64_t(amp.dimensions()[0]) *
                                      amp.dimensions()[1] * amp.dimensions()[2];
    if (problem_size >= 8192)
      launch(std::integral_constant<int, 1>{});
    else
      launch(std::integral_constant<int, 16>{});
  } else {
    launch(std::integral_constant<int, 32>{});
  }
  const auto status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_delay_transpose_gpu_impl_tmpl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 6>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> delay_bar) {
  if (!shapes_are_valid(a1, a2, amp, delay, freq))
    return ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes");
  if (vis_bar.dimensions()[0] != a1.dimensions()[0] ||
      vis_bar.dimensions()[1] != amp.dimensions()[1] ||
      vis_bar.dimensions()[2] != amp.dimensions()[2])
    return ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents");
  if (!same_shape(*amp_bar, amp) || !same_shape(*delay_bar, delay))
    return ffi::Error::InvalidArgument(
        "Expected amplitude and delay cotangents to match their primals");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (amp.element_count() < limit && vis_bar.element_count() < limit)
    return calc_rfi_delay_transpose_gpu_dispatch<T, std::int32_t>(
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
        freq, vis_bar, amp_bar, delay_bar);
  return calc_rfi_delay_transpose_gpu_dispatch<T, std::int64_t>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
      freq, vis_bar, amp_bar, delay_bar);
}

ffi::Error calc_rfi_delay_transpose_gpu_f32_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C64, 6> amp,
    ffi::Buffer<ffi::F32, 4> delay, ffi::Buffer<ffi::F32, 2> freq,
    ffi::BufferR3<ffi::C64> vis_bar,
    ffi::Result<ffi::Buffer<ffi::C64, 6>> amp_bar,
    ffi::Result<ffi::Buffer<ffi::F32, 4>> delay_bar) {
  return calc_rfi_delay_transpose_gpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
      freq, vis_bar, amp_bar, delay_bar);
}

ffi::Error calc_rfi_delay_transpose_gpu_f64_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<ffi::C128, 6> amp,
    ffi::Buffer<ffi::F64, 4> delay, ffi::Buffer<ffi::F64, 2> freq,
    ffi::BufferR3<ffi::C128> vis_bar,
    ffi::Result<ffi::Buffer<ffi::C128, 6>> amp_bar,
    ffi::Result<ffi::Buffer<ffi::F64, 4>> delay_bar) {
  return calc_rfi_delay_transpose_gpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp, delay,
      freq, vis_bar, amp_bar, delay_bar);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_transpose_gpu_f32,
                              calc_rfi_delay_transpose_gpu_f32_impl,
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
                                  .Arg<ffi::BufferR3<ffi::C64>>()
                                  .Ret<delay_amp_f32_t>()
                                  .Ret<delay_real4_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_transpose_gpu_f64,
                              calc_rfi_delay_transpose_gpu_f64_impl,
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
                                  .Arg<ffi::BufferR3<ffi::C128>>()
                                  .Ret<delay_amp_f64_t>()
                                  .Ret<delay_real4_f64_t>());

} // namespace gpu
} // namespace ri_kernels
