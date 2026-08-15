#include <complex>
#include <cstdint>

#include "parallel_for.hpp"
#include "rfi_delay_kernel_hwy.hpp"
#include "tensor.hpp"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ri_kernels {
namespace ffi = xla::ffi;

template <typename T>
void rfi_delay_vis_range(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> vis,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  rfi_delay_vis_hwy(bl_begin, bl_end, scale, a1, a2, amp, delay, freq, vis,
                     n_rfi, n_int_f, n_int_t);
}

template <typename T>
void rfi_delay_jvp_range(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp,
    Tensor4D<const std::complex<T> *> amp_dot,
    Tensor4D<const T *> delay, Tensor4D<const T *> delay_dot,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> out,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  rfi_delay_jvp_hwy(bl_begin, bl_end, scale, a1, a2, amp, amp_dot, delay,
                     delay_dot, freq, out, n_rfi, n_int_f, n_int_t);
}

template <typename T>
void rfi_delay_transpose_range(
    std::int64_t ant_begin, std::int64_t ant_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq, Tensor3D<const std::complex<T> *> vis_bar,
    Tensor4D<std::complex<T> *> amp_bar, Tensor4D<T *> delay_bar,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  rfi_delay_transpose_hwy(ant_begin, ant_end, scale, a1, a1_sorter, a1_start,
                           a2, a2_sorter, a2_start, amp, delay, freq, vis_bar,
                           amp_bar, delay_bar, n_rfi, n_int_f, n_int_t);
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future delay_vis_impl(
    ffi::ThreadPool pool, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32>, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32>, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> vis) {
  if (a1.dimensions()[0] != a2.dimensions()[0] ||
      delay.dimensions()[0] != amp.dimensions()[0] ||
      delay.dimensions()[1] != amp.dimensions()[2] ||
      delay.dimensions()[2] != amp.dimensions()[3] ||
      delay.dimensions()[3] != amp.dimensions()[5] ||
      freq.dimensions()[0] != amp.dimensions()[1] ||
      freq.dimensions()[1] != amp.dimensions()[4]) {
    return completed_future(ffi::Error::InvalidArgument(
        "Incompatible amplitude, delay, frequency, or baseline shapes"));
  }
  Tensor1D<const int *> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2v(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const std::complex<T> *> av(
      amp.typed_data(), amp.dimensions()[0], amp.dimensions()[1],
      amp.dimensions()[2], amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *> dv(delay.typed_data(), delay.dimensions()[0],
                         delay.dimensions()[1], delay.dimensions()[2], delay.dimensions()[3]);
  Tensor2D<const T *> fv(freq.typed_data(), freq.dimensions()[0], freq.dimensions()[1]);
  Tensor3D<std::complex<T> *> vv(vis->typed_data(), vis->dimensions()[0],
                                 vis->dimensions()[1], vis->dimensions()[2]);
  const auto n_rfi = amp.dimensions()[3];
  const auto n_int_f = amp.dimensions()[4];
  const auto n_int_t = amp.dimensions()[5];
  const T scale = T(1) / T(n_int_f * n_int_t);
  return parallel_for(pool, a1.dimensions()[0],
      [=](auto begin, auto end) mutable {
        rfi_delay_vis_range(begin, end, scale, a1v, a2v, av, dv, fv, vv,
                            n_rfi, n_int_f, n_int_t);
      });
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future delay_jvp_impl(
    ffi::ThreadPool pool, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32>, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32>,
    ffi::BufferR1<ffi::S32>, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<AMP_DT, 6> amp_dot, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 4> delay_dot, ffi::Buffer<REAL_DT, 2> freq,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  Tensor1D<const int *> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a2v(a2.typed_data(), a2.dimensions()[0]);
  Tensor4D<const std::complex<T> *> av(amp.typed_data(), amp.dimensions()[0], amp.dimensions()[1], amp.dimensions()[2], amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const std::complex<T> *> adv(amp_dot.typed_data(), amp.dimensions()[0], amp.dimensions()[1], amp.dimensions()[2], amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *> dv(delay.typed_data(), delay.dimensions()[0], delay.dimensions()[1], delay.dimensions()[2], delay.dimensions()[3]);
  Tensor4D<const T *> ddv(delay_dot.typed_data(), delay.dimensions()[0], delay.dimensions()[1], delay.dimensions()[2], delay.dimensions()[3]);
  Tensor2D<const T *> fv(freq.typed_data(), freq.dimensions()[0], freq.dimensions()[1]);
  Tensor3D<std::complex<T> *> ov(out->typed_data(), out->dimensions()[0], out->dimensions()[1], out->dimensions()[2]);
  const auto nr = amp.dimensions()[3], nf = amp.dimensions()[4], nt = amp.dimensions()[5];
  const T scale = T(1) / T(nf * nt);
  return parallel_for(pool, a1.dimensions()[0], [=](auto begin, auto end) mutable {
    rfi_delay_jvp_range(begin, end, scale, a1v, a2v, av, adv, dv, ddv, fv, ov, nr, nf, nt);
  });
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Future delay_transpose_impl(
    ffi::ThreadPool pool, ffi::BufferR1<ffi::S32> a1,
    ffi::BufferR1<ffi::S32> a1_sorter, ffi::BufferR1<ffi::S32> a1_start,
    ffi::BufferR1<ffi::S32> a2, ffi::BufferR1<ffi::S32> a2_sorter,
    ffi::BufferR1<ffi::S32> a2_start, ffi::Buffer<AMP_DT, 6> amp,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 2> freq,
    ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 6>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> delay_bar) {
  Tensor1D<const int *> a1v(a1.typed_data(), a1.dimensions()[0]);
  Tensor1D<const int *> a1sv(a1_sorter.typed_data(), a1_sorter.dimensions()[0]);
  Tensor1D<const int *> a1st(a1_start.typed_data(), a1_start.dimensions()[0]);
  Tensor1D<const int *> a2v(a2.typed_data(), a2.dimensions()[0]);
  Tensor1D<const int *> a2sv(a2_sorter.typed_data(), a2_sorter.dimensions()[0]);
  Tensor1D<const int *> a2st(a2_start.typed_data(), a2_start.dimensions()[0]);
  Tensor4D<const std::complex<T> *> av(amp.typed_data(), amp.dimensions()[0], amp.dimensions()[1], amp.dimensions()[2], amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<const T *> dv(delay.typed_data(), delay.dimensions()[0], delay.dimensions()[1], delay.dimensions()[2], delay.dimensions()[3]);
  Tensor2D<const T *> fv(freq.typed_data(), freq.dimensions()[0], freq.dimensions()[1]);
  Tensor3D<const std::complex<T> *> gv(vis_bar.typed_data(), vis_bar.dimensions()[0], vis_bar.dimensions()[1], vis_bar.dimensions()[2]);
  Tensor4D<std::complex<T> *> abv(amp_bar->typed_data(), amp.dimensions()[0], amp.dimensions()[1], amp.dimensions()[2], amp.dimensions()[3] * amp.dimensions()[4] * amp.dimensions()[5]);
  Tensor4D<T *> dbv(delay_bar->typed_data(), delay.dimensions()[0], delay.dimensions()[1], delay.dimensions()[2], delay.dimensions()[3]);
  const auto nr = amp.dimensions()[3], nf = amp.dimensions()[4], nt = amp.dimensions()[5];
  const T scale = T(1) / T(nf * nt);
  return parallel_for(pool, amp.dimensions()[0], [=](auto begin, auto end) mutable {
    rfi_delay_transpose_range(begin, end, scale, a1v, a1sv, a1st, a2v, a2sv, a2st,
                              av, dv, fv, gv, abv, dbv, nr, nf, nt);
  });
}

#define DEFINE_DELAY_HANDLERS(SUFFIX, AMP_DT, REAL_DT, T)                       \
ffi::Future calc_rfi_delay_cpu_##SUFFIX##_impl(                                \
    ffi::ThreadPool p, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1, \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                    \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                    \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<REAL_DT, 4> d,                       \
    ffi::Buffer<REAL_DT, 2> f, ffi::Result<ffi::BufferR3<AMP_DT>> o) {          \
  return delay_vis_impl<AMP_DT, REAL_DT, T>(p,a1,s1,b1,a2,s2,b2,a,d,f,o);      \
}                                                                               \
ffi::Future calc_rfi_delay_jvp_cpu_##SUFFIX##_impl(                            \
    ffi::ThreadPool p, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1, \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                    \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                    \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<AMP_DT, 6> ad,                       \
    ffi::Buffer<REAL_DT, 4> d, ffi::Buffer<REAL_DT, 4> dd,                     \
    ffi::Buffer<REAL_DT, 2> f, ffi::Result<ffi::BufferR3<AMP_DT>> o) {          \
  return delay_jvp_impl<AMP_DT, REAL_DT, T>(p,a1,s1,b1,a2,s2,b2,a,ad,d,dd,f,o);\
}                                                                               \
ffi::Future calc_rfi_delay_transpose_cpu_##SUFFIX##_impl(                      \
    ffi::ThreadPool p, ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> s1, \
    ffi::BufferR1<ffi::S32> b1, ffi::BufferR1<ffi::S32> a2,                    \
    ffi::BufferR1<ffi::S32> s2, ffi::BufferR1<ffi::S32> b2,                    \
    ffi::Buffer<AMP_DT, 6> a, ffi::Buffer<REAL_DT, 4> d,                       \
    ffi::Buffer<REAL_DT, 2> f, ffi::BufferR3<AMP_DT> g,                        \
    ffi::Result<ffi::Buffer<AMP_DT, 6>> ab,                                    \
    ffi::Result<ffi::Buffer<REAL_DT, 4>> db) {                                 \
  return delay_transpose_impl<AMP_DT, REAL_DT, T>(p,a1,s1,b1,a2,s2,b2,a,d,f,g,ab,db);\
}

DEFINE_DELAY_HANDLERS(f32, ffi::C64, ffi::F32, float)
DEFINE_DELAY_HANDLERS(f64, ffi::C128, ffi::F64, double)

#define DECLARE_DELAY_SYMBOL(NAME) \
extern "C" RI_KERNELS_API XLA_FFI_Error * NAME(XLA_FFI_CallFrame *call_frame)
DECLARE_DELAY_SYMBOL(calc_rfi_delay_cpu_f32);
DECLARE_DELAY_SYMBOL(calc_rfi_delay_cpu_f64);
DECLARE_DELAY_SYMBOL(calc_rfi_delay_jvp_cpu_f32);
DECLARE_DELAY_SYMBOL(calc_rfi_delay_jvp_cpu_f64);
DECLARE_DELAY_SYMBOL(calc_rfi_delay_transpose_cpu_f32);
DECLARE_DELAY_SYMBOL(calc_rfi_delay_transpose_cpu_f64);

using delay_amp_f32_t = ffi::Buffer<ffi::C64, 6>;
using delay_amp_f64_t = ffi::Buffer<ffi::C128, 6>;
using delay_real4_f32_t = ffi::Buffer<ffi::F32, 4>;
using delay_real4_f64_t = ffi::Buffer<ffi::F64, 4>;
using delay_real2_f32_t = ffi::Buffer<ffi::F32, 2>;
using delay_real2_f64_t = ffi::Buffer<ffi::F64, 2>;

#define BIND_DELAY_VIS(NAME, IMPL, AMP_BUF, REAL4_BUF, REAL2_BUF, AMP_DT)       \
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                     \
  .Ctx<ffi::ThreadPool>()                                                       \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<AMP_BUF>().Arg<REAL4_BUF>().Arg<REAL2_BUF>()                             \
  .Ret<ffi::BufferR3<AMP_DT>>())

#define BIND_DELAY_JVP(NAME, IMPL, AMP_BUF, REAL4_BUF, REAL2_BUF, AMP_DT)       \
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                     \
  .Ctx<ffi::ThreadPool>()                                                       \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<AMP_BUF>().Arg<AMP_BUF>().Arg<REAL4_BUF>().Arg<REAL4_BUF>()              \
  .Arg<REAL2_BUF>().Ret<ffi::BufferR3<AMP_DT>>())

#define BIND_DELAY_TRANSPOSE(NAME, IMPL, AMP_BUF, REAL4_BUF, REAL2_BUF, AMP_DT)\
XLA_FFI_DEFINE_HANDLER_SYMBOL(NAME, IMPL, ffi::Ffi::Bind()                     \
  .Ctx<ffi::ThreadPool>()                                                       \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<ffi::BufferR1<ffi::S32>>().Arg<ffi::BufferR1<ffi::S32>>()               \
  .Arg<AMP_BUF>().Arg<REAL4_BUF>().Arg<REAL2_BUF>()                             \
  .Arg<ffi::BufferR3<AMP_DT>>().Ret<AMP_BUF>().Ret<REAL4_BUF>())

BIND_DELAY_VIS(calc_rfi_delay_cpu_f32, calc_rfi_delay_cpu_f32_impl,
               delay_amp_f32_t, delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_VIS(calc_rfi_delay_cpu_f64, calc_rfi_delay_cpu_f64_impl,
               delay_amp_f64_t, delay_real4_f64_t, delay_real2_f64_t, ffi::C128);
BIND_DELAY_JVP(calc_rfi_delay_jvp_cpu_f32, calc_rfi_delay_jvp_cpu_f32_impl,
               delay_amp_f32_t, delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_JVP(calc_rfi_delay_jvp_cpu_f64, calc_rfi_delay_jvp_cpu_f64_impl,
               delay_amp_f64_t, delay_real4_f64_t, delay_real2_f64_t, ffi::C128);
BIND_DELAY_TRANSPOSE(calc_rfi_delay_transpose_cpu_f32,
                     calc_rfi_delay_transpose_cpu_f32_impl, delay_amp_f32_t,
                     delay_real4_f32_t, delay_real2_f32_t, ffi::C64);
BIND_DELAY_TRANSPOSE(calc_rfi_delay_transpose_cpu_f64,
                     calc_rfi_delay_transpose_cpu_f64_impl, delay_amp_f64_t,
                     delay_real4_f64_t, delay_real2_f64_t, ffi::C128);

} // namespace ri_kernels
