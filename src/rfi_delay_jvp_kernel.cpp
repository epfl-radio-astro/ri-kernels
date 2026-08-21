#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "parallel_for.hpp"
#include "rfi_delay_common.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_delay_jvp_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

template <typename T>
HWY_ATTR void
rfi_delay_jvp_kernel_opt_tmpl(T scale, std::int64_t i_bl_start,
                              std::int64_t i_bl_end,
                              Tensor1D<const int *> a1,
                              Tensor1D<const int *> a2,
                              Tensor4D<const std::complex<T> *> amp,
                              Tensor4D<const std::complex<T> *> amp_dot,
                              Tensor4D<const T *> delay,
                              Tensor4D<const T *> delay_dot,
                              Tensor2D<const T *> freq,
                              Tensor3D<std::complex<T> *> grad,
                              std::int64_t n_rfi, std::int64_t n_int_f,
                              std::int64_t n_int_t) {

  using D = TagType<T>;

  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);
  const auto zero = hn::Zero(d);

  // amp layout: (n_ant, n_freq, n_time, n_rfi * n_int_f * n_int_t)
  const auto n_freq = amp.shape[1];
  const auto n_time = amp.shape[2];

  std::vector<T> angular_frequency(n_freq * n_int_f);
  for (std::int64_t i_f = 0; i_f < n_freq; ++i_f)
    for (std::int64_t fi = 0; fi < n_int_f; ++fi)
      angular_frequency[i_f * n_int_f + fi] = two_pi<T>() * freq(i_f, fi);

  for (std::int64_t i_bl = i_bl_start; i_bl < i_bl_end; ++i_bl) {
    const auto i_a1 = a1(i_bl);
    const auto i_a2 = a2(i_bl);

    for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
      for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
        ComplexV<D> vector_sum{zero, zero};
        std::complex<T> scalar_sum{0, 0};

        for (std::int64_t r = 0; r < n_rfi; ++r) {
          const T *delay_1 = &delay(i_a1, i_t, r, 0);
          const T *delay_2 = &delay(i_a2, i_t, r, 0);
          const T *delay_dot_1 = &delay_dot(i_a1, i_t, r, 0);
          const T *delay_dot_2 = &delay_dot(i_a2, i_t, r, 0);

          for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
            const T omega = angular_frequency[i_f * n_int_f + fi];
            const auto omega_v = hn::Set(d, omega);
            const std::int64_t i_red = n_int_t * (fi + n_int_f * r);
            const auto *amp_1 = &amp(i_a1, i_f, i_t, i_red);
            const auto *amp_2 = &amp(i_a2, i_f, i_t, i_red);
            const auto *amp_dot_1 = &amp_dot(i_a1, i_f, i_t, i_red);
            const auto *amp_dot_2 = &amp_dot(i_a2, i_f, i_t, i_red);

            std::int64_t ti = 0;
            for (; ti + n_lanes <= n_int_t; ti += n_lanes) {
              const auto a = LoadU(d, amp_1 + ti);
              const auto b = LoadU(d, amp_2 + ti);
              const auto base = MulConj(a, b);
              auto term = Add(MulConj(LoadU(d, amp_dot_1 + ti), b),
                              MulConj(a, LoadU(d, amp_dot_2 + ti)));

              const auto delay_diff = hn::Sub(hn::LoadU(d, delay_1 + ti),
                                              hn::LoadU(d, delay_2 + ti));
              const auto delay_dot_diff =
                  hn::Sub(hn::LoadU(d, delay_dot_1 + ti),
                          hn::LoadU(d, delay_dot_2 + ti));
              const auto phase = hn::Mul(omega_v, delay_diff);
              const auto phase_dot = hn::Mul(omega_v, delay_dot_diff);

              // term += i * phase_dot * base
              term.re = hn::NegMulAdd(phase_dot, base.im, term.re);
              term.im = hn::MulAdd(phase_dot, base.re, term.im);

              const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
              vector_sum = Add(vector_sum, Mul(e, term));
            }

            for (; ti < n_int_t; ++ti) {
              const auto a = amp_1[ti];
              const auto b = amp_2[ti];
              const auto base = a * std::conj(b);
              const auto amp_term =
                  amp_dot_1[ti] * std::conj(b) + a * std::conj(amp_dot_2[ti]);
              const T phase = omega * (delay_1[ti] - delay_2[ti]);
              const T phase_dot = omega * (delay_dot_1[ti] - delay_dot_2[ti]);
              const std::complex<T> e(std::cos(phase), std::sin(phase));
              scalar_sum +=
                  e * (amp_term + std::complex<T>(0, phase_dot) * base);
            }
          }
        }

        scalar_sum += std::complex<T>(hn::ReduceSum(d, vector_sum.re),
                                      hn::ReduceSum(d, vector_sum.im));

        grad(i_bl, i_f, i_t) = scale * scalar_sum;
      }
    }
  }
}

// Thin wrappers with distinct (non-templated) symbols so that Highway's
// per-target dispatch (HWY_EXPORT_*) can target each precision separately.
HWY_ATTR void
rfi_delay_jvp_kernel_opt_f32(float scale, std::int64_t i_bl_start,
                             std::int64_t i_bl_end,
                             Tensor1D<const int *> a1,
                             Tensor1D<const int *> a2,
                             Tensor4D<const std::complex<float> *> amp,
                             Tensor4D<const std::complex<float> *> amp_dot,
                             Tensor4D<const float *> delay,
                             Tensor4D<const float *> delay_dot,
                             Tensor2D<const float *> freq,
                             Tensor3D<std::complex<float> *> grad,
                             std::int64_t n_rfi, std::int64_t n_int_f,
                             std::int64_t n_int_t) {
  rfi_delay_jvp_kernel_opt_tmpl<float>(scale, i_bl_start, i_bl_end, a1, a2, amp,
                                       amp_dot, delay, delay_dot, freq, grad,
                                       n_rfi, n_int_f, n_int_t);
}

HWY_ATTR void
rfi_delay_jvp_kernel_opt_f64(double scale, std::int64_t i_bl_start,
                             std::int64_t i_bl_end,
                             Tensor1D<const int *> a1,
                             Tensor1D<const int *> a2,
                             Tensor4D<const std::complex<double> *> amp,
                             Tensor4D<const std::complex<double> *> amp_dot,
                             Tensor4D<const double *> delay,
                             Tensor4D<const double *> delay_dot,
                             Tensor2D<const double *> freq,
                             Tensor3D<std::complex<double> *> grad,
                             std::int64_t n_rfi, std::int64_t n_int_f,
                             std::int64_t n_int_t) {
  rfi_delay_jvp_kernel_opt_tmpl<double>(scale, i_bl_start, i_bl_end, a1, a2,
                                        amp, amp_dot, delay, delay_dot, freq,
                                        grad, n_rfi, n_int_f, n_int_t);
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future calc_rfi_delay_jvp_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<AMP_DT, 6> amp, ffi::Buffer<AMP_DT, 6> amp_dot,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 4> delay_dot,
    ffi::Buffer<REAL_DT, 2> freq, ffi::Result<ffi::BufferR3<AMP_DT>> grad) {

  if (!shapes_are_valid(a1, a2, amp, delay, freq)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes"));
  }

  if (!same_shape(amp_dot, amp) || !same_shape(delay_dot, delay)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Expected amplitude and delay tangents to match their primals"));
  }

  // Snapshot of everything the chunks need. These views own their extents by
  // value, so they stay valid after the handler returns - unlike the ffi::Buffer
  // arguments, which point into the FFI call frame. See parallel_for().
  Tensor1D<const int *> a1_tensor(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2_tensor(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const std::complex<T> *> amp_tensor(
      amp.typed_data(), amp.dimensions()[0], amp.dimensions()[1],
      amp.dimensions()[2],
      amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const std::complex<T> *> amp_dot_tensor(
      amp_dot.typed_data(), amp.dimensions()[0], amp.dimensions()[1],
      amp.dimensions()[2],
      amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *> delay_tensor(
      delay.typed_data(), delay.dimensions()[0], delay.dimensions()[1],
      delay.dimensions()[2], delay.dimensions()[3]);
  Tensor4D<const T *> delay_dot_tensor(
      delay_dot.typed_data(), delay.dimensions()[0], delay.dimensions()[1],
      delay.dimensions()[2], delay.dimensions()[3]);
  Tensor2D<const T *> freq_tensor(freq.typed_data(), freq.dimensions()[0],
                                  freq.dimensions()[1]);
  Tensor3D<std::complex<T> *> grad_tensor(
      grad->typed_data(), grad->dimensions()[0], grad->dimensions()[1],
      grad->dimensions()[2]);

  const auto n_rfi = amp.dimensions()[3];
  const auto n_int_f = amp.dimensions()[4];
  const auto n_int_t = amp.dimensions()[5];
  const T scale = T(1) / T(n_int_f * n_int_t);

  const std::int64_t n_bl = a1.dimensions()[0];

  return parallel_for(
      thread_pool, n_bl,
      [scale, a1_tensor, a2_tensor, amp_tensor, amp_dot_tensor, delay_tensor,
       delay_dot_tensor, freq_tensor, grad_tensor, n_rfi, n_int_f,
       n_int_t](std::int64_t i_bl_start, std::int64_t i_bl_end) mutable {
        if constexpr (std::is_same_v<T, float>) {
          RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_jvp_kernel_opt_f32)
          (scale, i_bl_start, i_bl_end, a1_tensor, a2_tensor, amp_tensor,
           amp_dot_tensor, delay_tensor, delay_dot_tensor, freq_tensor,
           grad_tensor, n_rfi, n_int_f, n_int_t);
        } else {
          RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_jvp_kernel_opt_f64)
          (scale, i_bl_start, i_bl_end, a1_tensor, a2_tensor, amp_tensor,
           amp_dot_tensor, delay_tensor, delay_dot_tensor, freq_tensor,
           grad_tensor, n_rfi, n_int_f, n_int_t);
        }
      });
}

ffi::Future calc_rfi_delay_jvp_cpu_f32_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<ffi::C64, 6> amp, ffi::Buffer<ffi::C64, 6> amp_dot,
    ffi::Buffer<ffi::F32, 4> delay, ffi::Buffer<ffi::F32, 4> delay_dot,
    ffi::Buffer<ffi::F32, 2> freq, ffi::Result<ffi::BufferR3<ffi::C64>> grad) {
  return calc_rfi_delay_jvp_cpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      amp_dot, delay, delay_dot, freq, grad);
}

ffi::Future calc_rfi_delay_jvp_cpu_f64_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<ffi::C128, 6> amp, ffi::Buffer<ffi::C128, 6> amp_dot,
    ffi::Buffer<ffi::F64, 4> delay, ffi::Buffer<ffi::F64, 4> delay_dot,
    ffi::Buffer<ffi::F64, 2> freq,
    ffi::Result<ffi::BufferR3<ffi::C128>> grad) {
  return calc_rfi_delay_jvp_cpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      amp_dot, delay, delay_dot, freq, grad);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_jvp_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_jvp_cpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_jvp_cpu_f32,
                              calc_rfi_delay_jvp_cpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<delay_amp_f32_t>()
                                  .Arg<delay_amp_f32_t>()
                                  .Arg<delay_real4_f32_t>()
                                  .Arg<delay_real4_f32_t>()
                                  .Arg<delay_real2_f32_t>()
                                  .Ret<ffi::BufferR3<ffi::C64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_jvp_cpu_f64,
                              calc_rfi_delay_jvp_cpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<ffi::BufferR1<ffi::S32>>()
                                  .Arg<delay_amp_f64_t>()
                                  .Arg<delay_amp_f64_t>()
                                  .Arg<delay_real4_f64_t>()
                                  .Arg<delay_real4_f64_t>()
                                  .Arg<delay_real2_f64_t>()
                                  .Ret<ffi::BufferR3<ffi::C128>>());

#endif

} // namespace ri_kernels
