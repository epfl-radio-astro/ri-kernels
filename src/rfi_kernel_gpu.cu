#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unistd.h>

#include "gpu_compat.h"
#include "rfi_vis_common.hpp"
#include "rfi_vis_gpu.cuh"
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

// rfi_vis(b, f, t, i * P + j) = mean over the fine samples of
// sum_c A[a1, i, c] conj(A[a2, j, c]) exp(i (phase[a1] - phase[a2])).
// Each thread computes the phase factor once per fine sample, turns the first
// antenna's components with it and accumulates all P x P entries; one block
// reduction then combines them.
template <typename T, int P, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_kernel(T scale, Tensor1D<const int *, INT_T> a1,
               Tensor1D<const int *, INT_T> a2,
               Tensor5D<const typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_amp_fine,
               Tensor4D<const T *, INT_T> rfi_phase,
               Tensor4D<typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_vis) {

  using traits = gpu_complex_traits<T>;
  using complex_t = typename traits::complex_t;

  constexpr int E = P * kRfiColumns;
  constexpr int C = kRfiColumns;

  using partials_t = RfiPartials<T, 2 * P * P>;
  using BlockReduce_t = cub::BlockReduce<partials_t, BLOCK_SIZE>;

  __shared__ typename BlockReduce_t::TempStorage temp_storage;

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, P * 2, n_rfi * n_int_f * n_int_t)
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_reduce = rfi_amp_fine.shape[4];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis.shape[0]);
  assert(rfi_amp_fine.shape[3] == E);
  assert(rfi_phase.shape[3] == n_reduce);
  assert(rfi_vis.shape[1] == n_freq);
  assert(rfi_vis.shape[2] == n_time);
  assert(rfi_vis.shape[3] == P * P);


  for (INT_T i_bl = blockIdx.y; i_bl < n_bl; i_bl += gridDim.y) {
    INT_T i_a1 = a1(i_bl);
    INT_T i_a2 = a2(i_bl);

    for (INT_T i_f = blockIdx.z; i_f < n_freq; i_f += gridDim.z) {

      for (INT_T i_t = blockIdx.x; i_t < n_time; i_t += gridDim.x) {
        partials_t sum;
#pragma unroll
        for (int k = 0; k < 2 * P * P; ++k)
          sum.v[k] = T(0);

        const auto ptr_rfi_phase_1 = &rfi_phase(i_a1, i_f, i_t, 0);
        const auto ptr_rfi_phase_2 = &rfi_phase(i_a2, i_f, i_t, 0);

        // block reduction
        for (INT_T i_red = threadIdx.x; i_red < n_reduce; i_red += BLOCK_SIZE) {

          const auto val_rfi_phase_1 = ptr_rfi_phase_1[i_red];
          const auto val_rfi_phase_2 = ptr_rfi_phase_2[i_red];

          complex_t rot;
          traits::sincos_(val_rfi_phase_1 - val_rfi_phase_2, &rot.y, &rot.x);

          complex_t x1[E], x2[E];
#pragma unroll
          for (int e = 0; e < E; ++e) {
            x1[e] = traits::mul(rfi_amp_fine(i_a1, i_f, i_t, e, i_red), rot);
            x2[e] = rfi_amp_fine(i_a2, i_f, i_t, e, i_red);
          }

#pragma unroll
          for (int i = 0; i < P; ++i) {
#pragma unroll
            for (int j = 0; j < P; ++j) {
#pragma unroll
              for (int c = 0; c < C; ++c) {
                const auto res =
                    traits::mul(x1[i * C + c], traits::conj(x2[j * C + c]));
                sum.v[2 * (i * P + j)] += res.x;
                sum.v[2 * (i * P + j) + 1] += res.y;
              }
            }
          }
        }

        sum = BlockReduce_t(temp_storage).Sum(sum);
        __syncthreads(); // required for reuse of temp_storage

        if (threadIdx.x == 0) {
#pragma unroll
          for (int k = 0; k < P * P; ++k) {
            rfi_vis(i_bl, i_f, i_t, k) =
                complex_t{sum.v[2 * k] * scale, sum.v[2 * k + 1] * scale};
          }
        }
      }
    }
  }
}

// A wrapper function providing the interface between the XLA FFI call and the
// kernel above: it checks the shapes, builds the views and picks the
// specialisation for P and the block size.
template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType PHASE_DT>
ffi::Error
rfi_vis_fwd_gpu_dispatch(cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
                          ffi::BufferR1<ffi::S32> a2,
                          ffi::Buffer<AMP_DT, 8> rfi_amp_fine,
                          ffi::Buffer<PHASE_DT, 6> rfi_phase,
                          ffi::Result<ffi::Buffer<AMP_DT, 5>> rfi_vis) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;

  if (a1.dimensions()[0] != a2.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected a1 and a2 to have the same size");
  }

  if (auto err = rfi_validate_signal(rfi_amp_fine, rfi_phase); !err.success()) {
    return err;
  }

  if (auto err = rfi_validate_vis(*rfi_vis, a1.dimensions()[0], rfi_amp_fine);
      !err.success()) {
    return err;
  }

  const auto amp_dims = rfi_amp_fine.dimensions();
  const auto n_pol = amp_dims[3];
  const auto n_red = rfi_n_red(rfi_phase);

  Tensor1D<const int *, INT_T> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor5D<const complex_t *, INT_T> rfi_amp_fine_tensor(
      (const complex_t *)rfi_amp_fine.typed_data(), amp_dims[0], amp_dims[1],
      amp_dims[2], n_pol * kRfiColumns, n_red);
  Tensor4D<const T *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), amp_dims[0], amp_dims[1], amp_dims[2], n_red);

  Tensor4D<complex_t *, INT_T> rfi_vis_tensor(
      (complex_t *)rfi_vis->typed_data(), rfi_vis->dimensions()[0],
      rfi_vis->dimensions()[1], rfi_vis->dimensions()[2], n_pol * n_pol);

  const INT_T n_int_f = (INT_T)amp_dims[6];
  const INT_T n_int_t = (INT_T)amp_dims[7];

  const T scale = T(1) / T(n_int_t * n_int_f);

  const auto n_time = rfi_vis_tensor.shape[2];
  const auto n_bl = a1.dimensions()[0];
  const auto n_freq = rfi_vis_tensor.shape[1];

  // Nothing to compute, and a grid with an empty axis cannot be launched.
  if (n_bl == 0) {
    return ffi::Error::Success();
  }

  auto grid = create_clamped_grid(n_time, n_bl, n_freq);

  auto launch = [&](auto pol_c, auto block_size_c) {
    constexpr int pol = decltype(pol_c)::value;
    constexpr int block_size = decltype(block_size_c)::value;
    dim3 block(block_size);
    rfi_kernel<T, pol, block_size, INT_T><<<grid, block, 0, stream>>>(
        scale, a1_tensor, a2_tensor, rfi_amp_fine_tensor, rfi_phase_tensor,
        rfi_vis_tensor);
  };

  auto launch_pol = [&](auto block_size_c) {
    if (n_pol == 1) {
      launch(std::integral_constant<int, 1>{}, block_size_c);
    } else {
      launch(std::integral_constant<int, 2>{}, block_size_c);
    }
  };

  if (rfi_phase_tensor.shape[3] / 2 < 32) {
    launch_pol(std::integral_constant<int, 32>{});
  } else if (rfi_phase_tensor.shape[3] / 2 < 64) {
    launch_pol(std::integral_constant<int, 64>{});
  } else if (rfi_phase_tensor.shape[3] / 2 < 128) {
    launch_pol(std::integral_constant<int, 128>{});
  } else {
    launch_pol(std::integral_constant<int, 256>{});
  }

  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                cudaGetErrorString(status));
  }

  return ffi::Error::Success();
}

template <typename T, ffi::DataType AMP_DT, ffi::DataType PHASE_DT>
ffi::Error rfi_vis_fwd_gpu_impl_tmpl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 8> rfi_amp_fine,
    ffi::Buffer<PHASE_DT, 6> rfi_phase,
    ffi::Result<ffi::Buffer<AMP_DT, 5>> rfi_vis) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 && rfi_vis->element_count() < max32) {
    return rfi_vis_fwd_gpu_dispatch<T, std::int32_t, AMP_DT, PHASE_DT>(
        stream, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis);
  } else {
    return rfi_vis_fwd_gpu_dispatch<T, std::int64_t, AMP_DT, PHASE_DT>(
        stream, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis);
  }
}

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using rfi_amp_f32_t = ffi::Buffer<ffi::C64, 8>;
using rfi_phase_f32_t = ffi::Buffer<ffi::F32, 6>;
using rfi_vis_f32_t = ffi::Buffer<ffi::C64, 5>;
using rfi_amp_f64_t = ffi::Buffer<ffi::C128, 8>;
using rfi_phase_f64_t = ffi::Buffer<ffi::F64, 6>;
using rfi_vis_f64_t = ffi::Buffer<ffi::C128, 5>;

ffi::Error rfi_vis_fwd_gpu_f32_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_f32_t rfi_amp_fine,
    rfi_phase_f32_t rfi_phase, ffi::Result<rfi_vis_f32_t> rfi_vis) {
  return rfi_vis_fwd_gpu_impl_tmpl<float, ffi::C64, ffi::F32>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_phase, rfi_vis);
}

ffi::Error rfi_vis_fwd_gpu_f64_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_f64_t rfi_amp_fine,
    rfi_phase_f64_t rfi_phase, ffi::Result<rfi_vis_f64_t> rfi_vis) {
  return rfi_vis_fwd_gpu_impl_tmpl<double, ffi::C128, ffi::F64>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_phase, rfi_vis);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_fwd_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_fwd_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_fwd_gpu_f32, rfi_vis_fwd_gpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f32_t>()
                                  .Arg<rfi_phase_f32_t>()
                                  .Ret<rfi_vis_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_fwd_gpu_f64, rfi_vis_fwd_gpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f64_t>()
                                  .Arg<rfi_phase_f64_t>()
                                  .Ret<rfi_vis_f64_t>());
} // namespace gpu
} // namespace ri_kernels
