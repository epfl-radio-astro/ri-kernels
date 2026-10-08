#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#include "parallel_for.hpp"
#include "rfi_vis_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

// rfi_vis(b, f, t, i * P + j) = mean over the fine samples of
// sum_c A[a1, i, c] conj(A[a2, j, c]) exp(i (phase[a1] - phase[a2])).
// Every entry shares the phase factor, so it is computed once per sample and
// turns the first antenna's components before the P x P products.
template <int P, typename T>
HWY_ATTR void
rfi_kernel_opt_tmpl(T n_int_inv,
                    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                    Tensor5D<const std::complex<T> *> rfi_amp_fine,
                    Tensor4D<const T *> rfi_phase,
                    Tensor4D<std::complex<T> *> rfi_vis) {

  using D = TagType<T>;
  constexpr int E = P * kRfiColumns;
  constexpr int C = kRfiColumns;

  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

  // rfi_amp_fine layout: (n_ant, n_freq, n_time, P * 2, n_rfi * n_int_f * n_int_t)
  const auto n_freq = rfi_amp_fine.shape[1];
  const auto n_time = rfi_amp_fine.shape[2];
  const auto n_red = rfi_amp_fine.shape[4];
  const auto n_bl = a1.shape[0];

  assert(a1.shape[0] == a2.shape[0]);
  assert(a1.shape[0] == rfi_vis.shape[0]);
  assert(rfi_amp_fine.shape[3] == E);
  assert(rfi_phase.shape[3] == n_red);
  assert(rfi_vis.shape[1] == n_freq);
  assert(rfi_vis.shape[2] == n_time);
  assert(rfi_vis.shape[3] == P * P);

  for (std::int64_t i_bl = 0; i_bl < n_bl; ++i_bl) {
    std::int64_t i_a1 = a1(i_bl);
    std::int64_t i_a2 = a2(i_bl);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
        const std::complex<T> *ptr_amp_1[E], *ptr_amp_2[E];
        for (int e = 0; e < E; ++e) {
          ptr_amp_1[e] = &rfi_amp_fine(i_a1, i_f, i_t, e, 0);
          ptr_amp_2[e] = &rfi_amp_fine(i_a2, i_f, i_t, e, 0);
        }

        const auto ptr_val_rfi_phase_1 = &rfi_phase(i_a1, i_f, i_t, 0);
        const auto ptr_val_rfi_phase_2 = &rfi_phase(i_a2, i_f, i_t, 0);

        // Aggregate-initialised: ComplexV's implicit constructor lacks the target
        // attributes, so arrays must not default-construct it.
        ComplexV<D> acc[P * P] = {};
        for (int k = 0; k < P * P; ++k)
          acc[k] = ComplexV<D>{hn::Zero(d), hn::Zero(d)};

        std::int64_t i_red = 0;
        for (; i_red + n_lanes <= n_red; i_red += n_lanes) {
          const auto val_rfi_phase_1 =
              hn::LoadU(d, ptr_val_rfi_phase_1 + i_red);
          const auto val_rfi_phase_2 =
              hn::LoadU(d, ptr_val_rfi_phase_2 + i_red);

          const auto phase_diff = hn::Sub(val_rfi_phase_1, val_rfi_phase_2);

          const auto c = hn::Cos(d, phase_diff);
          const auto s = hn::Sin(d, phase_diff);

          const auto rot = ComplexV<D>{c, s};

          ComplexV<D> x1[E] = {}, x2[E] = {};
          for (int e = 0; e < E; ++e) {
            x1[e] = Mul(LoadU(d, ptr_amp_1[e] + i_red), rot);
            x2[e] = LoadU(d, ptr_amp_2[e] + i_red);
          }

          for (int i = 0; i < P; ++i)
            for (int j = 0; j < P; ++j)
              for (int col = 0; col < C; ++col)
                acc[i * P + j] =
                    Add(acc[i * P + j], MulConj(x1[i * C + col], x2[j * C + col]));
        }

        std::complex<T> sum[P * P];
        for (int k = 0; k < P * P; ++k)
          sum[k] = std::complex<T>(hn::ReduceSum(d, acc[k].re),
                                   hn::ReduceSum(d, acc[k].im));

        for (; i_red < n_red; ++i_red) {
          const auto val_rfi_phase_1 = ptr_val_rfi_phase_1[i_red];
          const auto val_rfi_phase_2 = ptr_val_rfi_phase_2[i_red];

          const std::complex<T> rot(
              std::cos(val_rfi_phase_1 - val_rfi_phase_2),
              std::sin(val_rfi_phase_1 - val_rfi_phase_2));

          std::complex<T> x1[E], x2[E];
          for (int e = 0; e < E; ++e) {
            x1[e] = ptr_amp_1[e][i_red] * rot;
            x2[e] = ptr_amp_2[e][i_red];
          }

          for (int i = 0; i < P; ++i)
            for (int j = 0; j < P; ++j)
              for (int col = 0; col < C; ++col)
                sum[i * P + j] += x1[i * C + col] * std::conj(x2[j * C + col]);
        }

        for (int k = 0; k < P * P; ++k)
          rfi_vis(i_bl, i_f, i_t, k) = sum[k] * n_int_inv;
      }
    }
  }
}

// Thin wrappers with distinct (non-templated) precision so that Highway's
// per-target dispatch (HWY_EXPORT_*) can target each precision separately.
// They keep a single template argument, P, which the dispatch macro accepts.
template <int P>
HWY_ATTR void
rfi_kernel_opt_f32(float n_int_inv,
                   Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                   Tensor5D<const std::complex<float> *> rfi_amp_fine,
                   Tensor4D<const float *> rfi_phase,
                   Tensor4D<std::complex<float> *> rfi_vis) {
  rfi_kernel_opt_tmpl<P, float>(n_int_inv, a1, a2, rfi_amp_fine, rfi_phase,
                                rfi_vis);
}

template <int P>
HWY_ATTR void
rfi_kernel_opt_f64(double n_int_inv,
                   Tensor1D<const int *> a1, Tensor1D<const int *> a2,
                   Tensor5D<const std::complex<double> *> rfi_amp_fine,
                   Tensor4D<const double *> rfi_phase,
                   Tensor4D<std::complex<double> *> rfi_vis) {
  rfi_kernel_opt_tmpl<P, double>(n_int_inv, a1, a2, rfi_amp_fine, rfi_phase,
                                 rfi_vis);
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using rfi_amp_f32_t = ffi::Buffer<ffi::C64, 8>;
using rfi_phase_f32_t = ffi::Buffer<ffi::F32, 6>;
using rfi_vis_f32_t = ffi::Buffer<ffi::C64, 5>;
using rfi_amp_f64_t = ffi::Buffer<ffi::C128, 8>;
using rfi_phase_f64_t = ffi::Buffer<ffi::F64, 6>;
using rfi_vis_f64_t = ffi::Buffer<ffi::C128, 5>;

template <int P, typename T>
void rfi_kernel_dispatch(T n_int_inv, Tensor1D<const int *> a1,
                         Tensor1D<const int *> a2,
                         Tensor5D<const std::complex<T> *> rfi_amp_fine,
                         Tensor4D<const T *> rfi_phase,
                         Tensor4D<std::complex<T> *> rfi_vis) {
  if constexpr (std::is_same_v<T, float>) {
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_kernel_opt_f32<P>)
    (n_int_inv, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis);
  } else {
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_kernel_opt_f64<P>)
    (n_int_inv, a1, a2, rfi_amp_fine, rfi_phase, rfi_vis);
  }
}

template <ffi::DataType AMP_DT, ffi::DataType PHASE_DT, typename T>
ffi::Future rfi_vis_fwd_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<AMP_DT, 8> rfi_amp_fine, ffi::Buffer<PHASE_DT, 6> rfi_phase,
    ffi::Result<ffi::Buffer<AMP_DT, 5>> rfi_vis) {
  if (a1.dimensions()[0] != a2.dimensions()[0]) {
    return completed_future(
        ffi::Error::InvalidArgument("Expected a1 and a2 to have the same size"));
  }

  if (auto err = rfi_validate_signal(rfi_amp_fine, rfi_phase); !err.success()) {
    return completed_future(std::move(err));
  }

  if (auto err = rfi_validate_vis(*rfi_vis, a1.dimensions()[0], rfi_amp_fine);
      !err.success()) {
    return completed_future(std::move(err));
  }

  const auto amp_dims = rfi_amp_fine.dimensions();
  const auto n_pol = amp_dims[3];
  const auto n_red = rfi_n_red(rfi_phase);

  // Snapshot of everything the chunks need. These views own their extents by
  // value, so they stay valid after the handler returns - unlike the ffi::Buffer
  // arguments, which point into the FFI call frame. See parallel_for().
  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor5D<const std::complex<T> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), amp_dims[0], amp_dims[1], amp_dims[2],
      n_pol * kRfiColumns, n_red);
  Tensor4D<const T *> rfi_phase_tensor(
      rfi_phase.typed_data(), amp_dims[0], amp_dims[1], amp_dims[2], n_red);

  Tensor4D<std::complex<T> *> rfi_vis_tensor(
      rfi_vis->typed_data(), rfi_vis->dimensions()[0],
      rfi_vis->dimensions()[1], rfi_vis->dimensions()[2], n_pol * n_pol);

  const auto n_int_f = amp_dims[6];
  const auto n_int_t = amp_dims[7];
  const T n_int_inv = T(1) / T(n_int_f * n_int_t);

  const int64_t n_bl = a1.dimensions()[0];

  return parallel_for(
      thread_pool, n_bl,
      [n_pol, n_int_inv, a1_tensor, a2_tensor, rfi_amp_fine_tensor,
       rfi_phase_tensor,
       rfi_vis_tensor](int64_t i_bl_start, int64_t i_bl_end) mutable {
        const int64_t n_bl_this_chunk = i_bl_end - i_bl_start;

        Tensor1D<const int *> a1_tensor_th(a1_tensor.ptr + i_bl_start,
                                           n_bl_this_chunk);
        Tensor1D<const int *> a2_tensor_th(a2_tensor.ptr + i_bl_start,
                                           n_bl_this_chunk);

        Tensor4D<std::complex<T> *> rfi_vis_tensor_th(
            &rfi_vis_tensor(i_bl_start, 0, 0, 0), n_bl_this_chunk,
            rfi_vis_tensor.shape[1], rfi_vis_tensor.shape[2],
            rfi_vis_tensor.shape[3]);

        if (n_pol == 1) {
          rfi_kernel_dispatch<1, T>(n_int_inv, a1_tensor_th, a2_tensor_th,
                                    rfi_amp_fine_tensor, rfi_phase_tensor,
                                    rfi_vis_tensor_th);
        } else {
          rfi_kernel_dispatch<2, T>(n_int_inv, a1_tensor_th, a2_tensor_th,
                                    rfi_amp_fine_tensor, rfi_phase_tensor,
                                    rfi_vis_tensor_th);
        }
      });
}

ffi::Future rfi_vis_fwd_cpu_f32_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_f32_t rfi_amp_fine, rfi_phase_f32_t rfi_phase,
    ffi::Result<rfi_vis_f32_t> rfi_vis) {
  return rfi_vis_fwd_cpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start,
      rfi_amp_fine, rfi_phase, rfi_vis);
}

ffi::Future rfi_vis_fwd_cpu_f64_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_f64_t rfi_amp_fine, rfi_phase_f64_t rfi_phase,
    ffi::Result<rfi_vis_f64_t> rfi_vis) {
  return rfi_vis_fwd_cpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start,
      rfi_amp_fine, rfi_phase, rfi_vis);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_fwd_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_fwd_cpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_fwd_cpu_f32, rfi_vis_fwd_cpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f32_t>()
                                  .Arg<rfi_phase_f32_t>()
                                  .Ret<rfi_vis_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_fwd_cpu_f64, rfi_vis_fwd_cpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<rfi_amp_f64_t>()
                                  .Arg<rfi_phase_f64_t>()
                                  .Ret<rfi_vis_f64_t>());

#endif

} // namespace ri_kernels
