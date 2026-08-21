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
#define HWY_TARGET_INCLUDE "rfi_delay_transpose_kernel.cpp" // this file

#include "hwy_dispatch.hpp"

namespace ri_kernels {

namespace ffi = xla::ffi;

namespace HWY_NAMESPACE { // required: unique per target

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

template <typename T>
HWY_ATTR void
rfi_delay_transpose_kernel_opt_tmpl(
    T scale, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq, Tensor3D<const std::complex<T> *> vis_grad,
    Tensor4D<std::complex<T> *> amp_grad, Tensor4D<T *> delay_grad,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {

  using D = TagType<T>;

  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);
  const auto zero = hn::Zero(d);
  const auto scale_v = hn::Set(d, scale);

  // amp layout: (n_ant, n_freq, n_time, n_rfi * n_int_f * n_int_t)
  const auto n_ant = amp.shape[0];
  const auto n_freq = amp.shape[1];
  const auto n_time = amp.shape[2];
  const auto n_bl = a1.shape[0];

  std::vector<T> angular_frequency(n_freq * n_int_f);
  for (std::int64_t i_f = 0; i_f < n_freq; ++i_f)
    for (std::int64_t fi = 0; fi < n_int_f; ++fi)
      angular_frequency[i_f * n_int_f + fi] = two_pi<T>() * freq(i_f, fi);

  for (std::int64_t i_ant = i_ant_start; i_ant < i_ant_end; ++i_ant) {

    const std::int64_t a1_begin = a1_start(i_ant);
    const std::int64_t a1_end =
        (i_ant == n_ant - 1) ? n_bl : a1_start(i_ant + 1);
    const std::int64_t a2_begin = a2_start(i_ant);
    const std::int64_t a2_end =
        (i_ant == n_ant - 1) ? n_bl : a2_start(i_ant + 1);

    for (std::int64_t i_t = 0; i_t < n_time; ++i_t) {
      for (std::int64_t r = 0; r < n_rfi; ++r) {
        const T *my_delay = &delay(i_ant, i_t, r, 0);
        T *my_delay_grad = &delay_grad(i_ant, i_t, r, 0);

        std::int64_t ti = 0;
        for (; ti + n_lanes <= n_int_t; ti += n_lanes) {
          const auto my_delay_v = hn::LoadU(d, my_delay + ti);
          auto delay_sum = zero;

          for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
            for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
              const std::int64_t i_red = ti + n_int_t * (fi + n_int_f * r);
              const auto my_amp = LoadU(d, &amp(i_ant, i_f, i_t, i_red));
              const T omega = angular_frequency[i_f * n_int_f + fi];
              const auto omega_v = hn::Set(d, omega);
              ComplexV<D> amp_sum{zero, zero};
              auto phase_sum = zero;

              // a1 loop: i_ant is the "first" antenna of the baseline.
              for (auto p = a1_begin; p < a1_end; ++p) {
                const auto i_bl = a1_sorter(p);
                const auto other = a2(i_bl);
                const auto other_amp = LoadU(d, &amp(other, i_f, i_t, i_red));
                const auto other_delay =
                    hn::LoadU(d, &delay(other, i_t, r, ti));
                const auto phase =
                    hn::Mul(omega_v, hn::Sub(my_delay_v, other_delay));
                const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
                const auto grad = vis_grad(i_bl, i_f, i_t);
                const ComplexV<D> grad_v{hn::Set(d, grad.real()),
                                         hn::Set(d, grad.imag())};

                // v = vis_grad * conj(other_amp) * e
                const auto v = Mul(MulConj(grad_v, other_amp), e);
                amp_sum = Add(amp_sum, v);
                // phase_sum += -v.im * my_amp.re - v.re * my_amp.im
                phase_sum = hn::NegMulAdd(v.im, my_amp.re, phase_sum);
                phase_sum = hn::NegMulAdd(v.re, my_amp.im, phase_sum);
              }

              // a2 loop: i_ant is the conjugated "second" antenna.
              for (auto p = a2_begin; p < a2_end; ++p) {
                const auto i_bl = a2_sorter(p);
                const auto other = a1(i_bl);
                const auto other_amp = LoadU(d, &amp(other, i_f, i_t, i_red));
                const auto other_delay =
                    hn::LoadU(d, &delay(other, i_t, r, ti));
                const auto phase =
                    hn::Mul(omega_v, hn::Sub(other_delay, my_delay_v));
                const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
                const auto grad = vis_grad(i_bl, i_f, i_t);
                const ComplexV<D> grad_v{hn::Set(d, grad.real()),
                                         hn::Set(d, grad.imag())};

                // v = conj(vis_grad * other_amp * e)
                const auto pre = Mul(Mul(grad_v, other_amp), e);
                const ComplexV<D> v{pre.re, hn::Neg(pre.im)};
                amp_sum = Add(amp_sum, v);
                phase_sum = hn::NegMulAdd(v.re, my_amp.im, phase_sum);
                phase_sum = hn::NegMulAdd(v.im, my_amp.re, phase_sum);
              }

              amp_sum.re = hn::Mul(scale_v, amp_sum.re);
              amp_sum.im = hn::Mul(scale_v, amp_sum.im);
              StoreU(d, amp_sum, &amp_grad(i_ant, i_f, i_t, i_red));
              delay_sum = hn::MulAdd(omega_v, phase_sum, delay_sum);
            }
          }

          hn::StoreU(hn::Mul(scale_v, delay_sum), d, my_delay_grad + ti);
        }

        for (; ti < n_int_t; ++ti) {
          T delay_sum = 0;

          for (std::int64_t i_f = 0; i_f < n_freq; ++i_f) {
            for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
              const std::int64_t i_red = ti + n_int_t * (fi + n_int_f * r);
              const auto my_amp = amp(i_ant, i_f, i_t, i_red);
              const T omega = angular_frequency[i_f * n_int_f + fi];
              std::complex<T> amp_sum{0, 0};
              T phase_sum = 0;

              for (auto p = a1_begin; p < a1_end; ++p) {
                const auto i_bl = a1_sorter(p);
                const auto other = a2(i_bl);
                const T phase =
                    omega * (my_delay[ti] - delay(other, i_t, r, ti));
                const std::complex<T> e(std::cos(phase), std::sin(phase));
                const auto v = vis_grad(i_bl, i_f, i_t) *
                               std::conj(amp(other, i_f, i_t, i_red)) * e;
                amp_sum += v;
                phase_sum -=
                    v.imag() * my_amp.real() + v.real() * my_amp.imag();
              }

              for (auto p = a2_begin; p < a2_end; ++p) {
                const auto i_bl = a2_sorter(p);
                const auto other = a1(i_bl);
                const T phase =
                    omega * (delay(other, i_t, r, ti) - my_delay[ti]);
                const std::complex<T> e(std::cos(phase), std::sin(phase));
                const auto v = std::conj(vis_grad(i_bl, i_f, i_t) *
                                         amp(other, i_f, i_t, i_red) * e);
                amp_sum += v;
                phase_sum -=
                    v.real() * my_amp.imag() + v.imag() * my_amp.real();
              }

              amp_grad(i_ant, i_f, i_t, i_red) = scale * amp_sum;
              delay_sum += omega * phase_sum;
            }
          }

          my_delay_grad[ti] = scale * delay_sum;
        }
      }
    }
  }
}

// Thin wrappers with distinct (non-templated) symbols so that Highway's
// per-target dispatch (HWY_EXPORT_*) can target each precision separately.
HWY_ATTR void
rfi_delay_transpose_kernel_opt_f32(
    float scale, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<float> *> amp, Tensor4D<const float *> delay,
    Tensor2D<const float *> freq,
    Tensor3D<const std::complex<float> *> vis_grad,
    Tensor4D<std::complex<float> *> amp_grad, Tensor4D<float *> delay_grad,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  rfi_delay_transpose_kernel_opt_tmpl<float>(
      scale, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
      a2_start, amp, delay, freq, vis_grad, amp_grad, delay_grad, n_rfi,
      n_int_f, n_int_t);
}

HWY_ATTR void
rfi_delay_transpose_kernel_opt_f64(
    double scale, std::int64_t i_ant_start, std::int64_t i_ant_end,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<double> *> amp, Tensor4D<const double *> delay,
    Tensor2D<const double *> freq,
    Tensor3D<const std::complex<double> *> vis_grad,
    Tensor4D<std::complex<double> *> amp_grad, Tensor4D<double *> delay_grad,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  rfi_delay_transpose_kernel_opt_tmpl<double>(
      scale, i_ant_start, i_ant_end, a1, a1_sorter, a1_start, a2, a2_sorter,
      a2_start, amp, delay, freq, vis_grad, amp_grad, delay_grad, n_rfi,
      n_int_f, n_int_t);
}

} // namespace HWY_NAMESPACE

#if HWY_ONCE

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future calc_rfi_delay_transpose_cpu_impl_tmpl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<AMP_DT, 6> amp, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 2> freq, ffi::BufferR3<AMP_DT> vis_grad,
    ffi::Result<ffi::Buffer<AMP_DT, 6>> amp_grad,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> delay_grad) {

  if (!shapes_are_valid(a1, a2, amp, delay, freq)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes"));
  }

  if (vis_grad.dimensions()[0] != a1.dimensions()[0] ||
      vis_grad.dimensions()[1] != amp.dimensions()[1] ||
      vis_grad.dimensions()[2] != amp.dimensions()[2]) {
    return completed_future(ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents"));
  }

  if (!same_shape(*amp_grad, amp) || !same_shape(*delay_grad, delay)) {
    return completed_future(ffi::Error::InvalidArgument(
        "Expected amplitude and delay cotangents to match their primals"));
  }

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

  Tensor4D<const std::complex<T> *> amp_tensor(
      amp.typed_data(), amp.dimensions()[0], amp.dimensions()[1],
      amp.dimensions()[2],
      amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *> delay_tensor(
      delay.typed_data(), delay.dimensions()[0], delay.dimensions()[1],
      delay.dimensions()[2], delay.dimensions()[3]);
  Tensor2D<const T *> freq_tensor(freq.typed_data(), freq.dimensions()[0],
                                  freq.dimensions()[1]);
  Tensor3D<const std::complex<T> *> vis_grad_tensor(
      vis_grad.typed_data(), vis_grad.dimensions()[0],
      vis_grad.dimensions()[1], vis_grad.dimensions()[2]);
  Tensor4D<std::complex<T> *> amp_grad_tensor(
      amp_grad->typed_data(), amp.dimensions()[0], amp.dimensions()[1],
      amp.dimensions()[2],
      amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<T *> delay_grad_tensor(
      delay_grad->typed_data(), delay.dimensions()[0], delay.dimensions()[1],
      delay.dimensions()[2], delay.dimensions()[3]);

  const auto n_rfi = amp.dimensions()[3];
  const auto n_int_f = amp.dimensions()[4];
  const auto n_int_t = amp.dimensions()[5];
  const T scale = T(1) / T(n_int_f * n_int_t);

  const std::int64_t n_ant = amp.dimensions()[0];

  return parallel_for(
      thread_pool, n_ant,
      [scale, a1_tensor, a1_sorter_tensor, a1_start_tensor, a2_tensor,
       a2_sorter_tensor, a2_start_tensor, amp_tensor, delay_tensor,
       freq_tensor, vis_grad_tensor, amp_grad_tensor, delay_grad_tensor, n_rfi,
       n_int_f,
       n_int_t](std::int64_t i_ant_start, std::int64_t i_ant_end) mutable {
        if constexpr (std::is_same_v<T, float>) {
          RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_transpose_kernel_opt_f32)
          (scale, i_ant_start, i_ant_end, a1_tensor, a1_sorter_tensor,
           a1_start_tensor, a2_tensor, a2_sorter_tensor, a2_start_tensor,
           amp_tensor, delay_tensor, freq_tensor, vis_grad_tensor,
           amp_grad_tensor, delay_grad_tensor, n_rfi, n_int_f, n_int_t);
        } else {
          RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_transpose_kernel_opt_f64)
          (scale, i_ant_start, i_ant_end, a1_tensor, a1_sorter_tensor,
           a1_start_tensor, a2_tensor, a2_sorter_tensor, a2_start_tensor,
           amp_tensor, delay_tensor, freq_tensor, vis_grad_tensor,
           amp_grad_tensor, delay_grad_tensor, n_rfi, n_int_f, n_int_t);
        }
      });
}

ffi::Future calc_rfi_delay_transpose_cpu_f32_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<ffi::C64, 6> amp, ffi::Buffer<ffi::F32, 4> delay,
    ffi::Buffer<ffi::F32, 2> freq, ffi::BufferR3<ffi::C64> vis_grad,
    ffi::Result<ffi::Buffer<ffi::C64, 6>> amp_grad,
    ffi::Result<ffi::Buffer<ffi::F32, 4>> delay_grad) {
  return calc_rfi_delay_transpose_cpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      delay, freq, vis_grad, amp_grad, delay_grad);
}

ffi::Future calc_rfi_delay_transpose_cpu_f64_impl(
    ffi::ThreadPool thread_pool,
    ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a1_sorter,
    ffi::BufferR1<ffi::S32> a1_start, ffi::BufferR1<ffi::S32> a2,
    ffi::BufferR1<ffi::S32> a2_sorter, ffi::BufferR1<ffi::S32> a2_start,
    ffi::Buffer<ffi::C128, 6> amp, ffi::Buffer<ffi::F64, 4> delay,
    ffi::Buffer<ffi::F64, 2> freq, ffi::BufferR3<ffi::C128> vis_grad,
    ffi::Result<ffi::Buffer<ffi::C128, 6>> amp_grad,
    ffi::Result<ffi::Buffer<ffi::F64, 4>> delay_grad) {
  return calc_rfi_delay_transpose_cpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      thread_pool, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, amp,
      delay, freq, vis_grad, amp_grad, delay_grad);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_transpose_cpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_delay_transpose_cpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_transpose_cpu_f32,
                              calc_rfi_delay_transpose_cpu_f32_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_delay_transpose_cpu_f64,
                              calc_rfi_delay_transpose_cpu_f64_impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
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

#endif

} // namespace ri_kernels
