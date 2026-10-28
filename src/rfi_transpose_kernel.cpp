#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <array>
#include <vector>
#include <unistd.h>

#include "parallel_for.hpp"
#include "rfi_vis_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_transpose_kernel.cpp" // this file

#include "hwy_dispatch.hpp"


namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

// Each antenna gathers its cotangents over the baselines it takes part in.
// With g the P x P cotangent of a baseline and e = exp(i (phase[a1] -
// phase[a2])):
//   as a1, component (i, c): t1_ic = sum_j g_ij conj(A2_jc) e,
//          phase += sum_ic Re(i t1_ic A1_ic);
//   as a2, component (j, c): t2_jc = conj(sum_i g_ij A1_ic e),
//          phase -= sum_jc Im(t2_jc A2_jc).
// The amplitude cotangent is the sum of the t terms. The phase factor is
// computed once per sample and baseline, and shared by every component.
template <int P, typename T>
HWY_ATTR void
rfi_transpose_kernel_opt_tmpl(
    T n_int_inv, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor5D<const std::complex<T> *> rfi_amp_fine,
    Tensor4D<const T *> rfi_phase,
    Tensor4D<const std::complex<T> *> rfi_vis_grad,
    Tensor5D<std::complex<T> *> rfi_amp_fine_grad,
    Tensor4D<T *> rfi_phase_grad) {

  using D = TagType<T>;
  constexpr int E = P * kRfiColumns;
  constexpr int C = kRfiColumns;

  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);

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

  const auto inv_v = hn::Set(d, n_int_inv);

  for (std::int64_t i_ant = i_ant_start; i_ant < i_ant_end; ++i_ant) {

    const std::int64_t a1_begin = a1_start(i_ant);
    const std::int64_t a1_end =
        (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);
    const std::int64_t a2_begin = a2_start(i_ant);
    const std::int64_t a2_end =
        (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {

        auto *p_my_phase_out = &rfi_phase_grad(i_ant, i_f, i_t, 0);
        const auto *p_my_phase_in = &rfi_phase(i_ant, i_f, i_t, 0);

        std::int64_t i_red = 0;
        for (; i_red + n_lanes <= n_red; i_red += n_lanes) {
          // Aggregate-initialised: ComplexV's implicit constructor lacks the
          // target attributes, so arrays must not default-construct it.
          ComplexV<D> my_amp[E] = {}, amp_sum[E] = {};
          for (int e = 0; e < E; ++e) {
            my_amp[e] = LoadU(d, &rfi_amp_fine(i_ant, i_f, i_t, e, i_red));
            amp_sum[e] = ComplexV<D>{hn::Zero(d), hn::Zero(d)};
          }
          const auto my_phase = hn::LoadU(d, p_my_phase_in + i_red);

          auto phase_sum = hn::Zero(d);

          // a1 loop: i_ant is the "first" antenna of the baseline.
          for (std::int64_t i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
            const std::int64_t i_bl = a1_sorter(i_bl_a1);
            const std::int64_t i_a2 = a2(i_bl);

            const auto other_phase =
                hn::LoadU(d, &rfi_phase(i_a2, i_f, i_t, i_red));
            const auto phase_diff = hn::Sub(my_phase, other_phase);
            const auto rot =
                ComplexV<D>{hn::Cos(d, phase_diff), hn::Sin(d, phase_diff)};

            // w_jc = conj(A2_jc) e
            ComplexV<D> w[E] = {};
            for (int e = 0; e < E; ++e)
              w[e] = MulConj(rot, LoadU(d, &rfi_amp_fine(i_a2, i_f, i_t, e, i_red)));

            for (int i = 0; i < P; ++i) {
              for (int col = 0; col < C; ++col) {
                ComplexV<D> t1{hn::Zero(d), hn::Zero(d)};
                for (int j = 0; j < P; ++j) {
                  const auto g_scalar = rfi_vis_grad(i_bl, i_f, i_t, i * P + j);
                  const auto g = ComplexV<D>{hn::Set(d, g_scalar.real()),
                                             hn::Set(d, g_scalar.imag())};
                  t1 = Add(t1, Mul(g, w[j * C + col]));
                }
                const int ei = i * C + col;
                amp_sum[ei] = Add(amp_sum[ei], t1);
                // phase_sum += Re(i t1 my_amp) = -t1.im * my_amp.re - t1.re * my_amp.im
                phase_sum = hn::NegMulAdd(t1.im, my_amp[ei].re, phase_sum);
                phase_sum = hn::NegMulAdd(t1.re, my_amp[ei].im, phase_sum);
              }
            }
          }

          // a2 loop: i_ant is the "second" antenna of the baseline.
          for (std::int64_t i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
            const std::int64_t i_bl = a2_sorter(i_bl_a2);
            const std::int64_t i_a1 = a1(i_bl);

            const auto other_phase =
                hn::LoadU(d, &rfi_phase(i_a1, i_f, i_t, i_red));
            const auto phase_diff = hn::Sub(other_phase, my_phase);
            const auto rot =
                ComplexV<D>{hn::Cos(d, phase_diff), hn::Sin(d, phase_diff)};

            // w_ic = A1_ic e
            ComplexV<D> w[E] = {};
            for (int e = 0; e < E; ++e)
              w[e] = Mul(LoadU(d, &rfi_amp_fine(i_a1, i_f, i_t, e, i_red)), rot);

            for (int j = 0; j < P; ++j) {
              for (int col = 0; col < C; ++col) {
                ComplexV<D> s{hn::Zero(d), hn::Zero(d)};
                for (int i = 0; i < P; ++i) {
                  const auto g_scalar = rfi_vis_grad(i_bl, i_f, i_t, i * P + j);
                  const auto g = ComplexV<D>{hn::Set(d, g_scalar.real()),
                                             hn::Set(d, g_scalar.imag())};
                  s = Add(s, Mul(g, w[i * C + col]));
                }
                // t2 = conj(s)
                const auto t2 = ComplexV<D>{s.re, hn::Neg(s.im)};
                const int ej = j * C + col;
                amp_sum[ej] = Add(amp_sum[ej], t2);
                // phase_sum -= t2.re * my_amp.im + t2.im * my_amp.re
                phase_sum = hn::NegMulAdd(t2.re, my_amp[ej].im, phase_sum);
                phase_sum = hn::NegMulAdd(t2.im, my_amp[ej].re, phase_sum);
              }
            }
          }

          for (int e = 0; e < E; ++e) {
            amp_sum[e].re = hn::Mul(amp_sum[e].re, inv_v);
            amp_sum[e].im = hn::Mul(amp_sum[e].im, inv_v);
            StoreU(d, amp_sum[e], &rfi_amp_fine_grad(i_ant, i_f, i_t, e, i_red));
          }
          phase_sum = hn::Mul(phase_sum, inv_v);
          hn::StoreU(phase_sum, d, p_my_phase_out + i_red);
        }

        for (; i_red < n_red; ++i_red) {
          std::complex<T> my_amp[E], amp_sum[E];
          for (int e = 0; e < E; ++e) {
            my_amp[e] = rfi_amp_fine(i_ant, i_f, i_t, e, i_red);
            amp_sum[e] = 0;
          }
          const auto my_phase = p_my_phase_in[i_red];

          T phase_sum = 0;

          for (std::int64_t i_bl_a1 = a1_begin; i_bl_a1 < a1_end; ++i_bl_a1) {
            const std::int64_t i_bl = a1_sorter(i_bl_a1);
            const std::int64_t i_a2 = a2(i_bl);

            const auto other_phase = rfi_phase(i_a2, i_f, i_t, i_red);
            const std::complex<T> rot(std::cos(my_phase - other_phase),
                                      std::sin(my_phase - other_phase));

            std::complex<T> w[E];
            for (int e = 0; e < E; ++e)
              w[e] = std::conj(rfi_amp_fine(i_a2, i_f, i_t, e, i_red)) * rot;

            for (int i = 0; i < P; ++i) {
              for (int col = 0; col < C; ++col) {
                std::complex<T> t1{0, 0};
                for (int j = 0; j < P; ++j)
                  t1 += rfi_vis_grad(i_bl, i_f, i_t, i * P + j) * w[j * C + col];
                const int ei = i * C + col;
                amp_sum[ei] += t1;
                phase_sum += -t1.imag() * my_amp[ei].real() -
                             t1.real() * my_amp[ei].imag();
              }
            }
          }

          for (std::int64_t i_bl_a2 = a2_begin; i_bl_a2 < a2_end; ++i_bl_a2) {
            const std::int64_t i_bl = a2_sorter(i_bl_a2);
            const std::int64_t i_a1 = a1(i_bl);

            const auto other_phase = rfi_phase(i_a1, i_f, i_t, i_red);
            const std::complex<T> rot(std::cos(other_phase - my_phase),
                                      std::sin(other_phase - my_phase));

            std::complex<T> w[E];
            for (int e = 0; e < E; ++e)
              w[e] = rfi_amp_fine(i_a1, i_f, i_t, e, i_red) * rot;

            for (int j = 0; j < P; ++j) {
              for (int col = 0; col < C; ++col) {
                std::complex<T> s{0, 0};
                for (int i = 0; i < P; ++i)
                  s += rfi_vis_grad(i_bl, i_f, i_t, i * P + j) * w[i * C + col];
                const auto t2 = std::conj(s);
                const int ej = j * C + col;
                amp_sum[ej] += t2;
                phase_sum -= t2.real() * my_amp[ej].imag() +
                             t2.imag() * my_amp[ej].real();
              }
            }
          }

          for (int e = 0; e < E; ++e)
            rfi_amp_fine_grad(i_ant, i_f, i_t, e, i_red) = amp_sum[e] * n_int_inv;
          p_my_phase_out[i_red] = phase_sum * n_int_inv;
        }
      }
    }
  }
}

template <int P>
HWY_ATTR void
rfi_transpose_kernel_opt_f32(
    float n_int_inv, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor5D<const std::complex<float> *> rfi_amp_fine,
    Tensor4D<const float *> rfi_phase,
    Tensor4D<const std::complex<float> *> rfi_vis_grad,
    Tensor5D<std::complex<float> *> rfi_amp_fine_grad,
    Tensor4D<float *> rfi_phase_grad) {
  rfi_transpose_kernel_opt_tmpl<P, float>(
      n_int_inv, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
      a2_start, rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
      rfi_phase_grad);
}

template <int P>
HWY_ATTR void
rfi_transpose_kernel_opt_f64(
    double n_int_inv, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor5D<const std::complex<double> *> rfi_amp_fine,
    Tensor4D<const double *> rfi_phase,
    Tensor4D<const std::complex<double> *> rfi_vis_grad,
    Tensor5D<std::complex<double> *> rfi_amp_fine_grad,
    Tensor4D<double *> rfi_phase_grad) {
  rfi_transpose_kernel_opt_tmpl<P, double>(
      n_int_inv, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
      a2_start, rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
      rfi_phase_grad);
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
void rfi_transpose_kernel_dispatch(
    T n_int_inv, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor5D<const std::complex<T> *> rfi_amp_fine,
    Tensor4D<const T *> rfi_phase,
    Tensor4D<const std::complex<T> *> rfi_vis_grad,
    Tensor5D<std::complex<T> *> rfi_amp_fine_grad,
    Tensor4D<T *> rfi_phase_grad) {
  if constexpr (std::is_same_v<T, float>) {
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_transpose_kernel_opt_f32<P>)
    (n_int_inv, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
     a2_start, rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
     rfi_phase_grad);
  } else {
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_transpose_kernel_opt_f64<P>)
    (n_int_inv, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
     a2_start, rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
     rfi_phase_grad);
  }
}

template <ffi::DataType AMP_DT, ffi::DataType PHASE_DT, typename T>
ffi::Future rfi_vis_transpose_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<AMP_DT, 8> rfi_amp_fine, ffi::Buffer<PHASE_DT, 6> rfi_phase,
    ffi::Buffer<AMP_DT, 5> rfi_vis_grad,
    ffi::Result<ffi::Buffer<AMP_DT, 8>> rfi_amp_fine_grad,
    ffi::Result<ffi::Buffer<PHASE_DT, 6>> rfi_phase_grad) {

  if (a1.dimensions()[0] != a2.dimensions()[0]) {
    return completed_future(
        ffi::Error::InvalidArgument("Expected a1 and a2 to have the same size"));
  }

  if (auto err = rfi_validate_signal(rfi_amp_fine, rfi_phase); !err.success()) {
    return completed_future(std::move(err));
  }

  if (auto err = rfi_validate_vis(rfi_vis_grad, a1.dimensions()[0], rfi_amp_fine);
      !err.success()) {
    return completed_future(std::move(err));
  }

  if (!rfi_same_shape(rfi_amp_fine, *rfi_amp_fine_grad)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Expected rfi_amp_fine and rfi_amp_fine_grad to have the same shape"));
  }
  if (!rfi_same_shape(rfi_phase, *rfi_phase_grad)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Expected rfi_phase and rfi_phase_grad to have the same shape"));
  }

  const auto amp_dims = rfi_amp_fine.dimensions();
  const auto n_pol = amp_dims[3];
  const auto n_red = rfi_n_red(rfi_phase);

  // Snapshot of everything the chunks need. These views own their extents by
  // value, so they stay valid after the handler returns - unlike the ffi::Buffer
  // arguments, which point into the FFI call frame. See parallel_for().
  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a1_sorter_tensor(a1_sorter.typed_data(),
                                         a1_sorter.dimensions()[0]);
  Tensor1D<const int *> a1_start_tensor(a1_start.typed_data(),
                                        a1_start.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *> a2_sorter_tensor(a2_sorter.typed_data(),
                                         a2_sorter.dimensions()[0]);
  Tensor1D<const int *> a2_start_tensor(a2_start.typed_data(),
                                        a2_start.dimensions()[0]);

  Tensor5D<const std::complex<T> *> rfi_amp_fine_tensor(
      rfi_amp_fine.typed_data(), amp_dims[0], amp_dims[1], amp_dims[2],
      n_pol * kRfiColumns, n_red);
  Tensor5D<std::complex<T> *> rfi_amp_fine_grad_tensor(
      rfi_amp_fine_grad->typed_data(), amp_dims[0], amp_dims[1], amp_dims[2],
      n_pol * kRfiColumns, n_red);
  Tensor4D<const T *> rfi_phase_tensor(
      rfi_phase.typed_data(), amp_dims[0], amp_dims[1], amp_dims[2], n_red);
  Tensor4D<T *> rfi_phase_grad_tensor(
      rfi_phase_grad->typed_data(), amp_dims[0], amp_dims[1], amp_dims[2],
      n_red);

  Tensor4D<const std::complex<T> *> rfi_grad_tensor(
      rfi_vis_grad.typed_data(), rfi_vis_grad.dimensions()[0],
      rfi_vis_grad.dimensions()[1], rfi_vis_grad.dimensions()[2],
      n_pol * n_pol);

  const auto n_int_f = amp_dims[6];
  const auto n_int_t = amp_dims[7];
  const T n_int_inv = T(1) / T(n_int_f * n_int_t);

  const int64_t n_ant = rfi_amp_fine_tensor.shape[0];

  return parallel_for(
      thread_pool, n_ant,
      [n_pol, n_int_inv, a1_tensor, a1_sorter_tensor, a1_start_tensor,
       a2_tensor, a2_sorter_tensor, a2_start_tensor, rfi_amp_fine_tensor,
       rfi_phase_tensor, rfi_grad_tensor, rfi_amp_fine_grad_tensor,
       rfi_phase_grad_tensor](int64_t i_ant_start, int64_t i_ant_end) mutable {
        if (n_pol == 1) {
          rfi_transpose_kernel_dispatch<1, T>(
              n_int_inv, i_ant_start, i_ant_end, a1_tensor, a1_sorter_tensor,
              a1_start_tensor, a2_tensor, a2_sorter_tensor, a2_start_tensor,
              rfi_amp_fine_tensor, rfi_phase_tensor, rfi_grad_tensor,
              rfi_amp_fine_grad_tensor, rfi_phase_grad_tensor);
        } else {
          rfi_transpose_kernel_dispatch<2, T>(
              n_int_inv, i_ant_start, i_ant_end, a1_tensor, a1_sorter_tensor,
              a1_start_tensor, a2_tensor, a2_sorter_tensor, a2_start_tensor,
              rfi_amp_fine_tensor, rfi_phase_tensor, rfi_grad_tensor,
              rfi_amp_fine_grad_tensor, rfi_phase_grad_tensor);
        }
      });
}

ffi::Future rfi_vis_transpose_cpu_f32_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_f32_t rfi_amp_fine, rfi_phase_f32_t rfi_phase,
    rfi_vis_f32_t rfi_vis_grad,
    ffi::Result<rfi_amp_f32_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_f32_t> rfi_phase_grad) {
  return rfi_vis_transpose_cpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start,
      rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
      rfi_phase_grad);
}

ffi::Future rfi_vis_transpose_cpu_f64_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    rfi_amp_f64_t rfi_amp_fine, rfi_phase_f64_t rfi_phase,
    rfi_vis_f64_t rfi_vis_grad,
    ffi::Result<rfi_amp_f64_t> rfi_amp_fine_grad,
    ffi::Result<rfi_phase_f64_t> rfi_phase_grad) {
  return rfi_vis_transpose_cpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start,
      rfi_amp_fine, rfi_phase, rfi_vis_grad, rfi_amp_fine_grad,
      rfi_phase_grad);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_transpose_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
ri_rfi_vis_transpose_cpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_transpose_cpu_f32,
                              rfi_vis_transpose_cpu_f32_impl,
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
                                  .Arg<rfi_vis_f32_t>()
                                  .Ret<rfi_amp_f32_t>()
                                  .Ret<rfi_phase_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(ri_rfi_vis_transpose_cpu_f64,
                              rfi_vis_transpose_cpu_f64_impl,
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
                                  .Arg<rfi_vis_f64_t>()
                                  .Ret<rfi_amp_f64_t>()
                                  .Ret<rfi_phase_f64_t>());

#endif

} // namespace ri_kernels
