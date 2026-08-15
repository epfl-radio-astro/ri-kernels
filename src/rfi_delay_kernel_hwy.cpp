#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "rfi_delay_kernel_hwy.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rfi_delay_kernel_hwy.cpp"

#include "hwy_dispatch.hpp"

namespace ri_kernels {
namespace HWY_NAMESPACE {

namespace hn = ::hwy::HWY_NAMESPACE;

#include "complex_vector_inl.hpp"

template <typename T> constexpr T two_pi_hwy() {
  return T(6.283185307179586476925286766559005768L);
}

template <typename T>
HWY_ATTR void rfi_delay_vis_opt(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> vis,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  using D = TagType<T>;
  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);
  const auto zero = hn::Zero(d);
  std::vector<T> angular_frequency(amp.shape[1] * n_int_f);
  for (std::int64_t f = 0; f < amp.shape[1]; ++f)
    for (std::int64_t fi = 0; fi < n_int_f; ++fi)
      angular_frequency[f * n_int_f + fi] = two_pi_hwy<T>() * freq(f, fi);

  for (std::int64_t bl = bl_begin; bl < bl_end; ++bl) {
    const auto ant1 = a1(bl);
    const auto ant2 = a2(bl);
    for (std::int64_t f = 0; f < amp.shape[1]; ++f) {
      for (std::int64_t t = 0; t < amp.shape[2]; ++t) {
        ComplexV<D> vector_sum{zero, zero};
        std::complex<T> scalar_sum{0, 0};
        for (std::int64_t r = 0; r < n_rfi; ++r) {
          const T *delay1 = &delay(ant1, t, r, 0);
          const T *delay2 = &delay(ant2, t, r, 0);
          for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
            const T omega = angular_frequency[f * n_int_f + fi];
            const auto omega_v = hn::Set(d, omega);
            const std::int64_t red_base = n_int_t * (fi + n_int_f * r);
            const auto *amp1 = &amp(ant1, f, t, red_base);
            const auto *amp2 = &amp(ant2, f, t, red_base);

            std::int64_t ti = 0;
            for (; ti + n_lanes <= n_int_t; ti += n_lanes) {
              const auto delay_diff = hn::Sub(hn::LoadU(d, delay1 + ti),
                                              hn::LoadU(d, delay2 + ti));
              const auto phase = hn::Mul(omega_v, delay_diff);
              const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
              const auto value = Mul(MulConj(LoadU(d, amp1 + ti),
                                             LoadU(d, amp2 + ti)),
                                     e);
              vector_sum = Add(vector_sum, value);
            }
            for (; ti < n_int_t; ++ti) {
              const T phase = omega * (delay1[ti] - delay2[ti]);
              const std::complex<T> e(std::cos(phase), std::sin(phase));
              scalar_sum += amp1[ti] * std::conj(amp2[ti]) * e;
            }
          }
        }
        scalar_sum += std::complex<T>(hn::ReduceSum(d, vector_sum.re),
                                      hn::ReduceSum(d, vector_sum.im));
        vis(bl, f, t) = scale * scalar_sum;
      }
    }
  }
}

template <typename T>
HWY_ATTR void rfi_delay_jvp_opt(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp,
    Tensor4D<const std::complex<T> *> amp_dot,
    Tensor4D<const T *> delay, Tensor4D<const T *> delay_dot,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> out,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  using D = TagType<T>;
  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);
  const auto zero = hn::Zero(d);
  std::vector<T> angular_frequency(amp.shape[1] * n_int_f);
  for (std::int64_t f = 0; f < amp.shape[1]; ++f)
    for (std::int64_t fi = 0; fi < n_int_f; ++fi)
      angular_frequency[f * n_int_f + fi] = two_pi_hwy<T>() * freq(f, fi);

  for (std::int64_t bl = bl_begin; bl < bl_end; ++bl) {
    const auto ant1 = a1(bl);
    const auto ant2 = a2(bl);
    for (std::int64_t f = 0; f < amp.shape[1]; ++f) {
      for (std::int64_t t = 0; t < amp.shape[2]; ++t) {
        ComplexV<D> vector_sum{zero, zero};
        std::complex<T> scalar_sum{0, 0};
        for (std::int64_t r = 0; r < n_rfi; ++r) {
          const T *delay1 = &delay(ant1, t, r, 0);
          const T *delay2 = &delay(ant2, t, r, 0);
          const T *delay_dot1 = &delay_dot(ant1, t, r, 0);
          const T *delay_dot2 = &delay_dot(ant2, t, r, 0);
          for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
            const T omega = angular_frequency[f * n_int_f + fi];
            const auto omega_v = hn::Set(d, omega);
            const std::int64_t red_base = n_int_t * (fi + n_int_f * r);
            const auto *amp1 = &amp(ant1, f, t, red_base);
            const auto *amp2 = &amp(ant2, f, t, red_base);
            const auto *amp_dot1 = &amp_dot(ant1, f, t, red_base);
            const auto *amp_dot2 = &amp_dot(ant2, f, t, red_base);

            std::int64_t ti = 0;
            for (; ti + n_lanes <= n_int_t; ti += n_lanes) {
              const auto a = LoadU(d, amp1 + ti);
              const auto b = LoadU(d, amp2 + ti);
              const auto base = MulConj(a, b);
              auto term = Add(MulConj(LoadU(d, amp_dot1 + ti), b),
                              MulConj(a, LoadU(d, amp_dot2 + ti)));
              const auto delay_diff = hn::Sub(hn::LoadU(d, delay1 + ti),
                                              hn::LoadU(d, delay2 + ti));
              const auto delay_dot_diff =
                  hn::Sub(hn::LoadU(d, delay_dot1 + ti),
                          hn::LoadU(d, delay_dot2 + ti));
              const auto phase = hn::Mul(omega_v, delay_diff);
              const auto dphase = hn::Mul(omega_v, delay_dot_diff);
              term.re = hn::NegMulAdd(dphase, base.im, term.re);
              term.im = hn::MulAdd(dphase, base.re, term.im);
              const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
              vector_sum = Add(vector_sum, Mul(e, term));
            }
            for (; ti < n_int_t; ++ti) {
              const auto a = amp1[ti];
              const auto b = amp2[ti];
              const auto base = a * std::conj(b);
              const auto amp_term = amp_dot1[ti] * std::conj(b) +
                                    a * std::conj(amp_dot2[ti]);
              const T phase = omega * (delay1[ti] - delay2[ti]);
              const T dphase = omega * (delay_dot1[ti] - delay_dot2[ti]);
              const std::complex<T> e(std::cos(phase), std::sin(phase));
              scalar_sum += e * (amp_term + std::complex<T>(0, dphase) * base);
            }
          }
        }
        scalar_sum += std::complex<T>(hn::ReduceSum(d, vector_sum.re),
                                      hn::ReduceSum(d, vector_sum.im));
        out(bl, f, t) = scale * scalar_sum;
      }
    }
  }
}

template <typename T>
HWY_ATTR void rfi_delay_transpose_opt(
    std::int64_t ant_begin, std::int64_t ant_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq,
    Tensor3D<const std::complex<T> *> vis_bar,
    Tensor4D<std::complex<T> *> amp_bar, Tensor4D<T *> delay_bar,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t) {
  using D = TagType<T>;
  const D d;
  constexpr std::int64_t n_lanes = hn::Lanes(d);
  const auto zero = hn::Zero(d);
  const auto scale_v = hn::Set(d, scale);
  const auto n_ant = amp.shape[0];
  const auto n_bl = a1.shape[0];
  std::vector<T> angular_frequency(amp.shape[1] * n_int_f);
  for (std::int64_t f = 0; f < amp.shape[1]; ++f)
    for (std::int64_t fi = 0; fi < n_int_f; ++fi)
      angular_frequency[f * n_int_f + fi] = two_pi_hwy<T>() * freq(f, fi);

  for (std::int64_t ant = ant_begin; ant < ant_end; ++ant) {
    const auto a1_begin = a1_start(ant);
    const auto a1_end = ant == n_ant - 1 ? n_bl : a1_start(ant + 1);
    const auto a2_begin = a2_start(ant);
    const auto a2_end = ant == n_ant - 1 ? n_bl : a2_start(ant + 1);
    for (std::int64_t t = 0; t < amp.shape[2]; ++t) {
      for (std::int64_t r = 0; r < n_rfi; ++r) {
        const T *my_delay = &delay(ant, t, r, 0);
        T *my_delay_bar = &delay_bar(ant, t, r, 0);
        std::int64_t ti = 0;
        for (; ti + n_lanes <= n_int_t; ti += n_lanes) {
          const auto my_delay_v = hn::LoadU(d, my_delay + ti);
          auto delay_sum = zero;
          for (std::int64_t f = 0; f < amp.shape[1]; ++f) {
            for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
              const std::int64_t red = ti + n_int_t * (fi + n_int_f * r);
              const auto my_amp = LoadU(d, &amp(ant, f, t, red));
              const T omega = angular_frequency[f * n_int_f + fi];
              const auto omega_v = hn::Set(d, omega);
              ComplexV<D> amp_sum{zero, zero};
              auto phase_sum = zero;

              for (auto p = a1_begin; p < a1_end; ++p) {
                const auto bl = a1_sorter(p);
                const auto other = a2(bl);
                const auto other_amp = LoadU(d, &amp(other, f, t, red));
                const auto other_delay = hn::LoadU(d, &delay(other, t, r, ti));
                const auto phase =
                    hn::Mul(omega_v, hn::Sub(my_delay_v, other_delay));
                const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
                const auto grad = vis_bar(bl, f, t);
                const ComplexV<D> grad_v{hn::Set(d, grad.real()),
                                         hn::Set(d, grad.imag())};
                const auto v = Mul(MulConj(grad_v, other_amp), e);
                amp_sum = Add(amp_sum, v);
                phase_sum = hn::NegMulAdd(v.im, my_amp.re, phase_sum);
                phase_sum = hn::NegMulAdd(v.re, my_amp.im, phase_sum);
              }
              for (auto p = a2_begin; p < a2_end; ++p) {
                const auto bl = a2_sorter(p);
                const auto other = a1(bl);
                const auto other_amp = LoadU(d, &amp(other, f, t, red));
                const auto other_delay = hn::LoadU(d, &delay(other, t, r, ti));
                const auto phase =
                    hn::Mul(omega_v, hn::Sub(other_delay, my_delay_v));
                const ComplexV<D> e{hn::Cos(d, phase), hn::Sin(d, phase)};
                const auto grad = vis_bar(bl, f, t);
                const ComplexV<D> grad_v{hn::Set(d, grad.real()),
                                         hn::Set(d, grad.imag())};
                const auto pre = Mul(Mul(grad_v, other_amp), e);
                const ComplexV<D> v{pre.re, hn::Neg(pre.im)};
                amp_sum = Add(amp_sum, v);
                phase_sum = hn::NegMulAdd(v.re, my_amp.im, phase_sum);
                phase_sum = hn::NegMulAdd(v.im, my_amp.re, phase_sum);
              }
              amp_sum.re = hn::Mul(scale_v, amp_sum.re);
              amp_sum.im = hn::Mul(scale_v, amp_sum.im);
              StoreU(d, amp_sum, &amp_bar(ant, f, t, red));
              delay_sum = hn::MulAdd(omega_v, phase_sum, delay_sum);
            }
          }
          hn::StoreU(hn::Mul(scale_v, delay_sum), d, my_delay_bar + ti);
        }

        for (; ti < n_int_t; ++ti) {
          T delay_sum = 0;
          for (std::int64_t f = 0; f < amp.shape[1]; ++f) {
            for (std::int64_t fi = 0; fi < n_int_f; ++fi) {
              const auto red = ti + n_int_t * (fi + n_int_f * r);
              const auto my_amp = amp(ant, f, t, red);
              const T omega = angular_frequency[f * n_int_f + fi];
              std::complex<T> amp_sum{0, 0};
              T phase_sum = 0;
              for (auto p = a1_begin; p < a1_end; ++p) {
                const auto bl = a1_sorter(p);
                const auto other = a2(bl);
                const T phase = omega * (my_delay[ti] - delay(other, t, r, ti));
                const std::complex<T> e(std::cos(phase), std::sin(phase));
                const auto v = vis_bar(bl, f, t) *
                               std::conj(amp(other, f, t, red)) * e;
                amp_sum += v;
                phase_sum -= v.imag() * my_amp.real() +
                             v.real() * my_amp.imag();
              }
              for (auto p = a2_begin; p < a2_end; ++p) {
                const auto bl = a2_sorter(p);
                const auto other = a1(bl);
                const T phase = omega * (delay(other, t, r, ti) - my_delay[ti]);
                const std::complex<T> e(std::cos(phase), std::sin(phase));
                const auto v = std::conj(vis_bar(bl, f, t) *
                                         amp(other, f, t, red) * e);
                amp_sum += v;
                phase_sum -= v.real() * my_amp.imag() +
                             v.imag() * my_amp.real();
              }
              amp_bar(ant, f, t, red) = scale * amp_sum;
              delay_sum += omega * phase_sum;
            }
          }
          my_delay_bar[ti] = scale * delay_sum;
        }
      }
    }
  }
}

#define DEFINE_DELAY_HWY_TARGET(SUFFIX, T)                                    \
  HWY_ATTR void rfi_delay_vis_opt_##SUFFIX(                                   \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a2, Tensor4D<const std::complex<T> *> a,          \
      Tensor4D<const T *> d, Tensor2D<const T *> f,                           \
      Tensor3D<std::complex<T> *> o, std::int64_t nr, std::int64_t nf,        \
      std::int64_t nt) {                                                       \
    rfi_delay_vis_opt<T>(b, e, s, a1, a2, a, d, f, o, nr, nf, nt);           \
  }                                                                            \
  HWY_ATTR void rfi_delay_jvp_opt_##SUFFIX(                                   \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a2, Tensor4D<const std::complex<T> *> a,          \
      Tensor4D<const std::complex<T> *> ad, Tensor4D<const T *> d,            \
      Tensor4D<const T *> dd, Tensor2D<const T *> f,                          \
      Tensor3D<std::complex<T> *> o, std::int64_t nr, std::int64_t nf,        \
      std::int64_t nt) {                                                       \
    rfi_delay_jvp_opt<T>(b, e, s, a1, a2, a, ad, d, dd, f, o, nr, nf, nt);   \
  }                                                                            \
  HWY_ATTR void rfi_delay_transpose_opt_##SUFFIX(                             \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a1s, Tensor1D<const int *> a1b,                   \
      Tensor1D<const int *> a2, Tensor1D<const int *> a2s,                    \
      Tensor1D<const int *> a2b, Tensor4D<const std::complex<T> *> a,         \
      Tensor4D<const T *> d, Tensor2D<const T *> f,                           \
      Tensor3D<const std::complex<T> *> g, Tensor4D<std::complex<T> *> ab,    \
      Tensor4D<T *> db, std::int64_t nr, std::int64_t nf, std::int64_t nt) {  \
    rfi_delay_transpose_opt<T>(b, e, s, a1, a1s, a1b, a2, a2s, a2b, a, d,   \
                               f, g, ab, db, nr, nf, nt);                     \
  }

DEFINE_DELAY_HWY_TARGET(f32, float)
DEFINE_DELAY_HWY_TARGET(f64, double)

} // namespace HWY_NAMESPACE

#if HWY_ONCE

#define DEFINE_DELAY_HWY_DISPATCH(SUFFIX, T)                                  \
  template <>                                                                  \
  void rfi_delay_vis_hwy<T>(                                                   \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a2, Tensor4D<const std::complex<T> *> a,          \
      Tensor4D<const T *> d, Tensor2D<const T *> f,                           \
      Tensor3D<std::complex<T> *> o, std::int64_t nr, std::int64_t nf,        \
      std::int64_t nt) {                                                       \
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_vis_opt_##SUFFIX)              \
    (b, e, s, a1, a2, a, d, f, o, nr, nf, nt);                               \
  }                                                                            \
  template <>                                                                  \
  void rfi_delay_jvp_hwy<T>(                                                   \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a2, Tensor4D<const std::complex<T> *> a,          \
      Tensor4D<const std::complex<T> *> ad, Tensor4D<const T *> d,            \
      Tensor4D<const T *> dd, Tensor2D<const T *> f,                          \
      Tensor3D<std::complex<T> *> o, std::int64_t nr, std::int64_t nf,        \
      std::int64_t nt) {                                                       \
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_jvp_opt_##SUFFIX)              \
    (b, e, s, a1, a2, a, ad, d, dd, f, o, nr, nf, nt);                       \
  }                                                                            \
  template <>                                                                  \
  void rfi_delay_transpose_hwy<T>(                                             \
      std::int64_t b, std::int64_t e, T s, Tensor1D<const int *> a1,          \
      Tensor1D<const int *> a1s, Tensor1D<const int *> a1b,                   \
      Tensor1D<const int *> a2, Tensor1D<const int *> a2s,                    \
      Tensor1D<const int *> a2b, Tensor4D<const std::complex<T> *> a,         \
      Tensor4D<const T *> d, Tensor2D<const T *> f,                           \
      Tensor3D<const std::complex<T> *> g, Tensor4D<std::complex<T> *> ab,    \
      Tensor4D<T *> db, std::int64_t nr, std::int64_t nf, std::int64_t nt) {  \
    RI_KERNELS_EXPORT_AND_DISPATCH_T(rfi_delay_transpose_opt_##SUFFIX)        \
    (b, e, s, a1, a1s, a1b, a2, a2s, a2b, a, d, f, g, ab, db, nr, nf, nt);  \
  }

DEFINE_DELAY_HWY_DISPATCH(f32, float)
DEFINE_DELAY_HWY_DISPATCH(f64, double)

#endif

} // namespace ri_kernels
