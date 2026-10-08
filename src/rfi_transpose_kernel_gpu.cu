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
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

// Each antenna gathers its cotangents over the baselines it takes part in.
// With g the P x P cotangent of a baseline and e = exp(i (phase[a1] -
// phase[a2])):
//   as a1, component (i, c): t1_ic = sum_j g_ij conj(A2_jc) e,
//          phase += sum_ic Re(i t1_ic A1_ic);
//   as a2, component (j, c): t2_jc = conj(sum_i g_ij A1_ic e),
//          phase -= sum_jc Im(t2_jc A2_jc).
// The amplitude cotangent is the sum of the t terms. Each thread owns one
// fine sample and all its components, so the phase factor is computed once
// per sample and baseline.
template <typename T, int P, int BLOCK_SIZE, typename INT_T>
__global__ void __launch_bounds__(BLOCK_SIZE)
    rfi_transpose_kernel(T n_int_inv,
                         Tensor1D<const int *, INT_T> a1,
                         Tensor1D<const int *, INT_T> a1_sorter,
                         Tensor1D<const int *, INT_T> a1_start,
                         Tensor1D<const int *, INT_T> a2,
                         Tensor1D<const int *, INT_T> a2_sorter,
                         Tensor1D<const int *, INT_T> a2_start,
                         Tensor5D<const typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_amp_fine,
                         Tensor4D<const T *, INT_T> rfi_phase,
                         Tensor4D<const typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_vis_grad,
                         Tensor5D<typename gpu_complex_traits<T>::complex_t *, INT_T> rfi_amp_fine_grad,
                         Tensor4D<T *, INT_T> rfi_phase_grad) {

  using traits = gpu_complex_traits<T>;
  using complex_t = typename traits::complex_t;

  constexpr int E = P * kRfiColumns;
  constexpr int C = kRfiColumns;

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, P * 2, n_rfi * n_int_f * n_int_t)
  const auto n_ant = rfi_amp_fine.shape[0];
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_red = rfi_amp_fine.shape[4];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis_grad.shape[0]);
  assert(rfi_amp_fine.shape[3] == E);
  assert(rfi_phase.shape[3] == n_red);
  assert(rfi_vis_grad.shape[1] == n_freq);
  assert(rfi_vis_grad.shape[2] == n_time);
  assert(rfi_vis_grad.shape[3] == P * P);

  const INT_T n_ft = n_freq * n_time;

  // Grid: (n_freq * n_time, n_ant). Threads in a block split i_red, which
  // covers the collapsed (n_rfi, n_int_f, n_int_t) inner dimension. All warp
  // lanes access consecutive i_red values of the same antenna and component
  // -> coalesced loads on the hot path. No inter-thread reduction is needed
  // because each thread writes distinct output elements.
  for (INT_T i_ft = blockIdx.x; i_ft < n_ft; i_ft += gridDim.x) {
    const INT_T i_t = i_ft % n_time;
    const INT_T i_f = i_ft / n_time;

    for (INT_T i_ant = blockIdx.y; i_ant < n_ant; i_ant += gridDim.y) {

      const INT_T a1_begin = a1_start(i_ant);
      const INT_T a1_end = (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);
      const INT_T a2_begin = a2_start(i_ant);
      const INT_T a2_end = (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

      for (INT_T i_red = threadIdx.x; i_red < n_red; i_red += BLOCK_SIZE) {

        complex_t my_amp[E], amp_sum[E];
#pragma unroll
        for (int e = 0; e < E; ++e) {
          my_amp[e] = rfi_amp_fine(i_ant, i_f, i_t, e, i_red);
          amp_sum[e] = complex_t{0, 0};
        }
        const auto my_phase = rfi_phase(i_ant, i_f, i_t, i_red);

        T phase_sum = 0;

        // a1 loop: i_ant is the "first" antenna of each baseline.
        for (INT_T i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
          const INT_T i_bl = a1_sorter(i_bl_a1);
          const INT_T i_a2 = a2(i_bl);

          const auto other_phase = rfi_phase(i_a2, i_f, i_t, i_red);

          complex_t rot;
          traits::sincos_(my_phase - other_phase, &rot.y, &rot.x);

          // Same address for every lane in the warp -> broadcast load.
          complex_t g[P * P];
#pragma unroll
          for (int k = 0; k < P * P; ++k)
            g[k] = rfi_vis_grad(i_bl, i_f, i_t, k);

          // w_jc = conj(A2_jc) e
          complex_t w[E];
#pragma unroll
          for (int e = 0; e < E; ++e)
            w[e] = traits::mul(traits::conj(rfi_amp_fine(i_a2, i_f, i_t, e, i_red)), rot);

#pragma unroll
          for (int i = 0; i < P; ++i) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
              complex_t t1{0, 0};
#pragma unroll
              for (int j = 0; j < P; ++j)
                t1 = traits::add(t1, traits::mul(g[i * P + j], w[j * C + c]));
              const int ei = i * C + c;
              amp_sum[ei] = traits::add(amp_sum[ei], t1);
              // Re(i * t1 * my_amp) = -t1.y*my_amp.x - t1.x*my_amp.y
              phase_sum += -t1.y * my_amp[ei].x - t1.x * my_amp[ei].y;
            }
          }
        }

        // a2 loop: i_ant is the "second" antenna of each baseline.
        for (INT_T i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
          const INT_T i_bl = a2_sorter(i_bl_a2);
          const INT_T i_a1 = a1(i_bl);

          const auto other_phase = rfi_phase(i_a1, i_f, i_t, i_red);

          complex_t rot;
          traits::sincos_(other_phase - my_phase, &rot.y, &rot.x);

          complex_t g[P * P];
#pragma unroll
          for (int k = 0; k < P * P; ++k)
            g[k] = rfi_vis_grad(i_bl, i_f, i_t, k);

          // w_ic = A1_ic e
          complex_t w[E];
#pragma unroll
          for (int e = 0; e < E; ++e)
            w[e] = traits::mul(rfi_amp_fine(i_a1, i_f, i_t, e, i_red), rot);

#pragma unroll
          for (int j = 0; j < P; ++j) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
              complex_t s{0, 0};
#pragma unroll
              for (int i = 0; i < P; ++i)
                s = traits::add(s, traits::mul(g[i * P + j], w[i * C + c]));
              const auto t2 = traits::conj(s);
              const int ej = j * C + c;
              amp_sum[ej] = traits::add(amp_sum[ej], t2);
              // Im(t2 * my_amp) = t2.x*my_amp.y + t2.y*my_amp.x
              phase_sum -= t2.x * my_amp[ej].y + t2.y * my_amp[ej].x;
            }
          }
        }

#pragma unroll
        for (int e = 0; e < E; ++e) {
          rfi_amp_fine_grad(i_ant, i_f, i_t, e, i_red) =
              complex_t{amp_sum[e].x * n_int_inv, amp_sum[e].y * n_int_inv};
        }
        rfi_phase_grad(i_ant, i_f, i_t, i_red) = phase_sum * n_int_inv;
      }
    }
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType PHASE_DT>
ffi::Error rfi_vis_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 8> rfi_amp_fine,
    ffi::Buffer<PHASE_DT, 6> rfi_phase, ffi::Buffer<AMP_DT, 5> rfi_vis_grad,
    ffi::Result<ffi::Buffer<AMP_DT, 8>> rfi_amp_fine_grad,
    ffi::Result<ffi::Buffer<PHASE_DT, 6>> rfi_phase_grad) {
  using complex_t = typename gpu_complex_traits<T>::complex_t;

  if (a1.dimensions()[0] != a2.dimensions()[0]) {
    return ffi::Error::InvalidArgument(
        "Expected a1 and a2 to have the same size");
  }

  if (auto err = rfi_validate_signal(rfi_amp_fine, rfi_phase); !err.success()) {
    return err;
  }

  if (auto err = rfi_validate_vis(rfi_vis_grad, a1.dimensions()[0], rfi_amp_fine);
      !err.success()) {
    return err;
  }

  if (!rfi_same_shape(rfi_amp_fine, *rfi_amp_fine_grad)) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_amp_fine and rfi_amp_fine_grad to have the same shape");
  }
  if (!rfi_same_shape(rfi_phase, *rfi_phase_grad)) {
    return ffi::Error::InvalidArgument(
        "Expected rfi_phase and rfi_phase_grad to have the same shape");
  }

  const auto amp_dims = rfi_amp_fine.dimensions();
  const auto n_pol = amp_dims[3];

  Tensor1D<const int *, INT_T> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1_sorter_tensor(a1_sorter.typed_data(),
                                                a1_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a1_start_tensor(a1_start.typed_data(),
                                               a1_start.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_sorter_tensor(a2_sorter.typed_data(),
                                                a2_sorter.dimensions()[0]);
  Tensor1D<const int *, INT_T> a2_start_tensor(a2_start.typed_data(),
                                               a2_start.dimensions()[0]);

  const INT_T n_freq = (INT_T)amp_dims[1];
  const INT_T n_time = (INT_T)amp_dims[2];
  const INT_T n_int_f = (INT_T)amp_dims[6];
  const INT_T n_int_t = (INT_T)amp_dims[7];
  const INT_T n_ant = (INT_T)amp_dims[0];
  const INT_T n_red = (INT_T)rfi_n_red(rfi_phase);

  Tensor5D<const complex_t *, INT_T> rfi_amp_fine_tensor(
      (const complex_t *)rfi_amp_fine.typed_data(), n_ant, n_freq, n_time,
      n_pol * kRfiColumns, n_red);
  Tensor5D<complex_t *, INT_T> rfi_amp_fine_grad_tensor(
      (complex_t *)rfi_amp_fine_grad->typed_data(), n_ant, n_freq, n_time,
      n_pol * kRfiColumns, n_red);
  Tensor4D<const T *, INT_T> rfi_phase_tensor(
      rfi_phase.typed_data(), n_ant, n_freq, n_time, n_red);
  Tensor4D<T *, INT_T> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), n_ant, n_freq, n_time, n_red);

  Tensor4D<const complex_t *, INT_T> rfi_grad_tensor(
      (const complex_t *)rfi_vis_grad.typed_data(),
      rfi_vis_grad.dimensions()[0], rfi_vis_grad.dimensions()[1],
      rfi_vis_grad.dimensions()[2], n_pol * n_pol);

  const T n_int_inv = T(1) / T(n_int_t * n_int_f);

  // Grid: (n_freq * n_time, n_ant). Each block covers one (i_ant, i_f, i_t);
  // threads in the block split the collapsed (n_rfi, n_int_f, n_int_t) dim.
  auto grid = create_clamped_grid(n_freq * n_time, n_ant, 1);

  // BLOCK_SIZE chosen to match n_red (the per-block parallel dim). The ladder
  // avoids leaving most of a large block idle when n_red is small.
  auto launch = [&](auto pol_c, auto block_size_c) {
    constexpr int pol = decltype(pol_c)::value;
    constexpr int block_size = decltype(block_size_c)::value;
    dim3 block(block_size);
    rfi_transpose_kernel<T, pol, block_size, INT_T><<<grid, block, 0, stream>>>(
        n_int_inv, a1_tensor, a1_sorter_tensor, a1_start_tensor, a2_tensor,
        a2_sorter_tensor, a2_start_tensor, rfi_amp_fine_tensor,
        rfi_phase_tensor, rfi_grad_tensor, rfi_amp_fine_grad_tensor,
        rfi_phase_grad_tensor);
  };

  auto launch_pol = [&](auto block_size_c) {
    if (n_pol == 1) {
      launch(std::integral_constant<int, 1>{}, block_size_c);
    } else {
      launch(std::integral_constant<int, 2>{}, block_size_c);
    }
  };

  if (n_red <= 32) {
    launch_pol(std::integral_constant<int, 32>{});
  } else if (n_red <= 64) {
    launch_pol(std::integral_constant<int, 64>{});
  } else if (n_red <= 128) {
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
ffi::Error rfi_vis_transpose_gpu_impl_tmpl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 8> rfi_amp_fine,
    ffi::Buffer<PHASE_DT, 6> rfi_phase, ffi::Buffer<AMP_DT, 5> rfi_vis_grad,
    ffi::Result<ffi::Buffer<AMP_DT, 8>> rfi_amp_fine_grad,
    ffi::Result<ffi::Buffer<PHASE_DT, 6>> rfi_phase_grad) {
  constexpr std::int64_t max32 = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (a1.element_count() < max32 && a2.element_count() < max32 &&
      rfi_amp_fine.element_count() < max32 &&
      rfi_phase.element_count() < max32 &&
      rfi_vis_grad.element_count() < max32) {
    return rfi_vis_transpose_gpu_dispatch<T, std::int32_t, AMP_DT, PHASE_DT>(
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
        rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
  } else {
    return rfi_vis_transpose_gpu_dispatch<T, std::int64_t, AMP_DT, PHASE_DT>(
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
        rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
  }
}

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using rfi_amp_f32_t = ffi::Buffer<ffi::C64, 8>;
using rfi_phase_f32_t = ffi::Buffer<ffi::F32, 6>;
using rfi_vis_f32_t = ffi::Buffer<ffi::C64, 5>;
using rfi_amp_f64_t = ffi::Buffer<ffi::C128, 8>;
using rfi_phase_f64_t = ffi::Buffer<ffi::F64, 6>;
using rfi_vis_f64_t = ffi::Buffer<ffi::C128, 5>;

ffi::Error rfi_vis_transpose_gpu_f32_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_f32_t rfi_amp_fine,
    rfi_phase_f32_t rfi_phase, rfi_vis_f32_t rfi_vis_grad,
    ffi::Result<rfi_amp_f32_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_f32_t> rfi_phase_grad) {
  return rfi_vis_transpose_gpu_impl_tmpl<float, ffi::C64, ffi::F32>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
}

ffi::Error rfi_vis_transpose_gpu_f64_impl(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, rfi_amp_f64_t rfi_amp_fine,
    rfi_phase_f64_t rfi_phase, rfi_vis_f64_t rfi_vis_grad,
    ffi::Result<rfi_amp_f64_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_f64_t> rfi_phase_grad) {
  return rfi_vis_transpose_gpu_impl_tmpl<double, ffi::C128, ffi::F64>(
      stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, rfi_amp_fine,
      rfi_phase, rfi_vis_grad, rfi_amp_fine_grad, rfi_phase_grad);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_transpose_gpu_f32,
                              rfi_vis_transpose_gpu_f32_impl,
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
                                  .Arg<rfi_vis_f32_t>()
                                  .Ret<rfi_amp_f32_t>()
                                  .Ret<rfi_phase_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_transpose_gpu_f64,
                              rfi_vis_transpose_gpu_f64_impl,
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
                                  .Arg<rfi_vis_f64_t>()
                                  .Ret<rfi_amp_f64_t>()
                                  .Ret<rfi_phase_f64_t>());
} // namespace gpu
} // namespace ri_kernels
