#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include "gpu_compat.h"
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

template <typename T> __device__ inline T two_pi() {
  return T(6.283185307179586476925286766559005768L);
}

template <int GROUP_SIZE, typename T>
__device__ inline T warp_sum(T value) {
  for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
#ifdef __HIPCC__
    value += __shfl_down(value, offset, GROUP_SIZE);
#else
    unsigned mask = 0xffffffffU;
    if constexpr (GROUP_SIZE < 32) {
      const unsigned group = (threadIdx.x % 32) / GROUP_SIZE;
      mask = ((1U << GROUP_SIZE) - 1U) << (group * GROUP_SIZE);
    }
    value += __shfl_down_sync(mask, value, offset, GROUP_SIZE);
#endif
  }
  return value;
}

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_delay_vis_kernel(
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

template <typename T, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE) rfi_delay_jvp_kernel(
    T scale, Tensor1D<const int *, INT_T> a1,
    Tensor1D<const int *, INT_T> a2,
    Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> amp,
    Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> amp_dot,
    Tensor4D<const T *, INT_T> delay,
    Tensor4D<const T *, INT_T> delay_dot,
    Tensor2D<const T *, INT_T> freq,
    Tensor3D<typename gpu_complex_traits<T>::complex_t *, INT_T> out,
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
          const T delay_dot_diff = delay_dot(ant1, t, r, ti) -
                                   delay_dot(ant2, t, r, ti);
          INT_T red = ti + n_int_t * n_int_f * r;
          for (INT_T fi = 0; fi < n_int_f; ++fi, red += n_int_t) {
            const auto aa = amp(ant1, f, t, red);
            const auto bb = amp(ant2, f, t, red);
            const auto base = traits::mul(aa, traits::conj(bb));
            auto term = traits::add(
                traits::mul(amp_dot(ant1, f, t, red), traits::conj(bb)),
                traits::mul(aa, traits::conj(amp_dot(ant2, f, t, red))));
            const T omega = angular_frequency[fi];
            const T dphase = omega * delay_dot_diff;
            term.x -= dphase * base.y;
            term.y += dphase * base.x;
            complex_t e;
            traits::sincos_(omega * delay_diff, &e.y, &e.x);
            sum = traits::add(sum, traits::mul(e, term));
          }
        }
        sum.x = reduce_t(real_storage).Sum(sum.x);
        sum.y = reduce_t(imag_storage).Sum(sum.y);
        __syncthreads();
        if (threadIdx.x == 0) {
          out(bl, f, t) = complex_t{scale * sum.x, scale * sum.y};
        }
      }
      __syncthreads();
    }
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
      delay_sum = warp_sum<GROUP_SIZE>(delay_sum);
      if (lane == 0)
        delay_bar(ant, t, r, ti) = scale * delay_sum;
    }
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error delay_vis_dispatch(
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
  rfi_delay_vis_kernel<T, B, INT_T><<<grid, B, sizeof(T) * nf, stream>>>(      \
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

template <typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error delay_jvp_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a2, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<AMP_DT, 6> amp_dot, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 4> delay_dot, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;
  Tensor1D<const int *, INT_T> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2v(a2.typed_data(), a2.dimensions()[0]);
  const INT_T nr = amp.dimensions()[3], nf = amp.dimensions()[4];
  const INT_T nt = amp.dimensions()[5], nred = nr * nf * nt;
  const INT_T ncompact = nr * nt;
  Tensor4D<const complex_t *, INT_T> av(
      reinterpret_cast<const complex_t *>(amp.typed_data()), amp.dimensions()[0],
      amp.dimensions()[1], amp.dimensions()[2], nred);
  Tensor4D<const complex_t *, INT_T> adv(
      reinterpret_cast<const complex_t *>(amp_dot.typed_data()), amp.dimensions()[0],
      amp.dimensions()[1], amp.dimensions()[2], nred);
  Tensor4D<const T *, INT_T> dv(delay.typed_data(), delay.dimensions()[0],
                                delay.dimensions()[1], delay.dimensions()[2],
                                delay.dimensions()[3]);
  Tensor4D<const T *, INT_T> ddv(delay_dot.typed_data(), delay.dimensions()[0],
                                 delay.dimensions()[1], delay.dimensions()[2],
                                 delay.dimensions()[3]);
  Tensor2D<const T *, INT_T> fv(freq.typed_data(), freq.dimensions()[0],
                                freq.dimensions()[1]);
  Tensor3D<complex_t *, INT_T> ov(
      reinterpret_cast<complex_t *>(out->typed_data()), out->dimensions()[0],
      out->dimensions()[1], out->dimensions()[2]);
  const T scale = T(1) / T(nf * nt);
  const auto grid = create_clamped_grid(av.shape[2], a1v.shape[0], av.shape[1]);
#define LAUNCH_DELAY_JVP(B)                                                     \
  rfi_delay_jvp_kernel<T, B, INT_T><<<grid, B, sizeof(T) * nf, stream>>>(      \
      scale, a1v, a2v, av, adv, dv, ddv, fv, ov, nr, nf, nt)
  if (ncompact <= 32) LAUNCH_DELAY_JVP(32);
  else if (ncompact <= 64) LAUNCH_DELAY_JVP(64);
  else if (ncompact <= 256) LAUNCH_DELAY_JVP(128);
  else LAUNCH_DELAY_JVP(256);
#undef LAUNCH_DELAY_JVP
  const auto status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

template <typename T, typename INT_T, ffi::DataType AMP_DT,
          ffi::DataType REAL_DT>
ffi::Error delay_transpose_dispatch(
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

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT>
bool shapes_are_valid(ffi::BufferR1<ffi::S32> a1,
                      ffi::BufferR1<ffi::S32> a2,
                      ffi::Buffer<AMP_DT, 6> amp,
                      ffi::Buffer<REAL_DT, 4> delay,
                      ffi::Buffer<REAL_DT, 2> freq) {
  if (a1.dimensions()[0] != a2.dimensions()[0] ||
      delay.dimensions()[0] != amp.dimensions()[0] ||
      delay.dimensions()[1] != amp.dimensions()[2] ||
      delay.dimensions()[2] != amp.dimensions()[3] ||
      delay.dimensions()[3] != amp.dimensions()[5] ||
      freq.dimensions()[0] != amp.dimensions()[1] ||
      freq.dimensions()[1] != amp.dimensions()[4])
    return false;
  return true;
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error delay_vis_impl(cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32>, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32>, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> vis) {
  if (!shapes_are_valid(a1, a2, amp, delay, freq))
    return ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  if (amp.element_count() < limit && delay.element_count() < limit &&
      vis->element_count() < limit)
    return delay_vis_dispatch<T, std::int32_t>(stream, a1, a2, amp, delay, freq, vis);
  return delay_vis_dispatch<T, std::int64_t>(stream, a1, a2, amp, delay, freq, vis);
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error delay_jvp_impl(cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32>, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32>, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<AMP_DT, 6> amp_dot, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 4> delay_dot, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  if (amp.element_count() < limit && out->element_count() < limit)
    return delay_jvp_dispatch<T, std::int32_t>(stream, a1, a2, amp, amp_dot,
                                               delay, delay_dot, freq, out);
  return delay_jvp_dispatch<T, std::int64_t>(stream, a1, a2, amp, amp_dot,
                                             delay, delay_dot, freq, out);
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error delay_transpose_impl(cudaStream_t stream,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1s,
    ffi::BufferR1<ffi::S32> a1b, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2s, ffi::BufferR1<ffi::S32> a2b,
    ffi::Buffer<AMP_DT, 6> amp, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 2> freq, ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 6>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> delay_bar) {
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  if (amp.element_count() < limit && vis_bar.element_count() < limit)
    return delay_transpose_dispatch<T, std::int32_t>(stream, a1, a1s, a1b, a2,
        a2s, a2b, amp, delay, freq, vis_bar, amp_bar, delay_bar);
  return delay_transpose_dispatch<T, std::int64_t>(stream, a1, a1s, a1b, a2,
      a2s, a2b, amp, delay, freq, vis_bar, amp_bar, delay_bar);
}

#define DEFINE_DELAY_GPU_HANDLERS(SUFFIX, AMP_DT, REAL_DT, T)                 \
ffi::Error calc_rfi_delay_gpu_##SUFFIX##_impl(                                \
    cudaStream_t s, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1,  \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                  \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                  \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<REAL_DT, 4> d,                     \
    ffi::Buffer<REAL_DT, 2> f, ffi::Result<ffi::BufferR3<AMP_DT>> o) {        \
  return delay_vis_impl<AMP_DT, REAL_DT, T>(s,a1,s1,b1,a2,s2,b2,a,d,f,o);    \
}                                                                             \
ffi::Error calc_rfi_delay_jvp_gpu_##SUFFIX##_impl(                            \
    cudaStream_t s, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1,  \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                  \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                  \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<AMP_DT, 6> ad,                     \
    ffi::Buffer<REAL_DT, 4> d, ffi::Buffer<REAL_DT, 4> dd,                   \
    ffi::Buffer<REAL_DT, 2> f, ffi::Result<ffi::BufferR3<AMP_DT>> o) {        \
  return delay_jvp_impl<AMP_DT, REAL_DT, T>(s,a1,s1,b1,a2,s2,b2,a,ad,d,dd,f,o);\
}                                                                             \
ffi::Error calc_rfi_delay_transpose_gpu_##SUFFIX##_impl(                      \
    cudaStream_t s, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1,  \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                  \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                  \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<REAL_DT, 4> d,                     \
    ffi::Buffer<REAL_DT, 2> f, ffi::BufferR3<AMP_DT> g,                      \
    ffi::Result<ffi::Buffer<AMP_DT, 6>> ab,                                  \
    ffi::Result<ffi::Buffer<REAL_DT, 4>> db) {                               \
  return delay_transpose_impl<AMP_DT, REAL_DT, T>(s,a1,s1,b1,a2,s2,b2,a,d,f,g,ab,db);\
}

DEFINE_DELAY_GPU_HANDLERS(f32, ffi::C64, ffi::F32, float)
DEFINE_DELAY_GPU_HANDLERS(f64, ffi::C128, ffi::F64, double)

#define DECLARE_DELAY_GPU_SYMBOL(NAME)                                        \
extern "C" RI_KERNELS_API XLA_FFI_Error *NAME(XLA_FFI_CallFrame *call_frame)
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_gpu_f32);
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_gpu_f64);
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_jvp_gpu_f32);
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_jvp_gpu_f64);
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_transpose_gpu_f32);
DECLARE_DELAY_GPU_SYMBOL(calc_rfi_delay_transpose_gpu_f64);

using delay_amp_f32_t = ffi::Buffer<ffi::C64, 6>;
using delay_amp_f64_t = ffi::Buffer<ffi::C128, 6>;
using delay_real4_f32_t = ffi::Buffer<ffi::F32, 4>;
using delay_real4_f64_t = ffi::Buffer<ffi::F64, 4>;
using delay_real2_f32_t = ffi::Buffer<ffi::F32, 2>;
using delay_real2_f64_t = ffi::Buffer<ffi::F64, 2>;

#define BIND_DELAY_GPU_VIS(NAME, IMPL, AMP, REAL4, REAL2, AMP_DT)             \
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                    \
  .Ctx<ffi::PlatformStream<cudaStream_t>>()                                   \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<AMP>().Arg<REAL4>().Arg<REAL2>().Ret<ffi::BufferR3<AMP_DT>>())

#define BIND_DELAY_GPU_JVP(NAME, IMPL, AMP, REAL4, REAL2, AMP_DT)             \
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                    \
  .Ctx<ffi::PlatformStream<cudaStream_t>>()                                   \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<AMP>().Arg<AMP>().Arg<REAL4>().Arg<REAL4>().Arg<REAL2>()               \
  .Ret<ffi::BufferR3<AMP_DT>>())

#define BIND_DELAY_GPU_TRANSPOSE(NAME, IMPL, AMP, REAL4, REAL2, AMP_DT)       \
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                    \
  .Ctx<ffi::PlatformStream<cudaStream_t>>()                                   \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()             \
  .Arg<AMP>().Arg<REAL4>().Arg<REAL2>().Arg<ffi::BufferR3<AMP_DT>>()          \
  .Ret<AMP>().Ret<REAL4>())

BIND_DELAY_GPU_VIS(calc_rfi_delay_gpu_f32, calc_rfi_delay_gpu_f32_impl,
                   delay_amp_f32_t, delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_GPU_VIS(calc_rfi_delay_gpu_f64, calc_rfi_delay_gpu_f64_impl,
                   delay_amp_f64_t, delay_real4_f64_t, delay_real2_f64_t, ffi::C128);
BIND_DELAY_GPU_JVP(calc_rfi_delay_jvp_gpu_f32, calc_rfi_delay_jvp_gpu_f32_impl,
                   delay_amp_f32_t, delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_GPU_JVP(calc_rfi_delay_jvp_gpu_f64, calc_rfi_delay_jvp_gpu_f64_impl,
                   delay_amp_f64_t, delay_real4_f64_t, delay_real2_f64_t, ffi::C128);
BIND_DELAY_GPU_TRANSPOSE(calc_rfi_delay_transpose_gpu_f32,
                         calc_rfi_delay_transpose_gpu_f32_impl, delay_amp_f32_t,
                         delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_GPU_TRANSPOSE(calc_rfi_delay_transpose_gpu_f64,
                         calc_rfi_delay_transpose_gpu_f64_impl, delay_amp_f64_t,
                         delay_real4_f64_t, delay_real2_f64_t, ffi::C128);

} // namespace gpu
} // namespace ri_kernels
