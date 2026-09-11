// The fine samples of a chunk of time cells, materialised once per cell so
// that the staged kernels stage them with contiguous loads instead of each
// rebuilding them, per tile pair the antenna sits in, from a dozen scattered
// gathers per sample.
//
// For the cells (f, t0 + t_local) of the chunk, source r, fine sample s of
// the cell (time-major: s = v * n_int_f + u) and antenna a, at
//
//   ((((f * n_tc + t_local) * n_rfi + r) * n_s + s) * n_ant + a)
//
// the buffer holds S = A_interp exp(i phi), the sample the kernels multiply;
// a second buffer, by mode, holds the phase factor exp(i phi) (the transpose
// turns its cotangents by it) or the tangent dS = dA_interp exp(i phi) (the
// JVP). Antenna fastest: a tile's antennas at one sample are one contiguous
// run. The kernel's threads run along time within a block of one (f, r, a),
// so its reads of the signal, phase and delay are contiguous too.
#pragma once

#include <cstdint>

#include "rfi_interp_common.hpp"
#include "tensor.hpp"

namespace ri_kernels {
namespace gpu {

// The tangent modes: dS from the signal tangent alone, or with i dphi S from
// the phase and delay tangents as well (the full JVP).
enum SampleMode { kSamplesOnly = 0, kSamplesAndPhase = 1, kSamplesAndTangent = 2, kSamplesAndFullTangent = 3 };

template <typename T, typename INT_T> struct SampleViews {
  Tensor4D<const Cplx<T> *, INT_T> amp, amp_dot;
  Tensor4D<const T *, INT_T> phase, delay, phase_dot, delay_dot;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
};

template <typename INT_T>
__device__ __host__ inline std::int64_t sample_index(INT_T f, INT_T t_local, INT_T r, INT_T s, INT_T a,
                                                     INT_T n_tc, INT_T n_rfi, INT_T n_s, INT_T n_ant) {
  return (((std::int64_t(f) * n_tc + t_local) * n_rfi + r) * n_s + s) * n_ant + a;
}

constexpr int kSampleBlock = 256;

template <typename T, typename INT_T, int MODE>
__global__ void __launch_bounds__(kSampleBlock) rfi_interp_samples_kernel(
    SampleViews<T, INT_T> v, Cplx<T> *S, Cplx<T> *S2, INT_T t0, INT_T n_tc) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_s = n_int_f * n_int_t;
  const INT_T n_blocks = n_freq * n_rfi * n_ant;
  for (INT_T b = blockIdx.x; b < n_blocks; b += gridDim.x) {
    const INT_T a = b % n_ant, r = (b / n_ant) % n_rfi, f = b / (n_ant * n_rfi);
    const INT_T sf = v.start_freq(f);
    const T freq_f = v.freqs(f);
    for (INT_T idx = threadIdx.x; idx < n_tc * n_s; idx += kSampleBlock) {
      const INT_T t_local = idx % n_tc, s = idx / n_tc;  // time fastest
      const INT_T t = t0 + t_local, vv = s / n_int_f, u = s % n_int_f;
      const INT_T st = v.start_time(t);
      const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
      const auto e = phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, a, r, f, t);
      constexpr bool tangent = MODE == kSamplesAndTangent || MODE == kSamplesAndFullTangent;
      Cplx<T> amp{0, 0}, amp_dot{0, 0};
      for (INT_T k = 0; k < n_sf; ++k) {
        const T wk = v.w_freq(f, k, u);
        for (INT_T l = 0; l < n_st; ++l) {
          const T w = wk * v.w_time(t, l, vv);
          const Cplx<T> c = v.amp(a, r, sf + k, st + l);
          amp.re += w * c.re;
          amp.im += w * c.im;
          if constexpr (tangent) {
            const Cplx<T> d = v.amp_dot(a, r, sf + k, st + l);
            amp_dot.re += w * d.re;
            amp_dot.im += w * d.im;
          }
        }
      }
      const std::int64_t o = sample_index(f, t_local, r, s, a, n_tc, n_rfi, n_s, n_ant);
      const Cplx<T> sample = cmul(amp, e);
      S[o] = sample;
      if constexpr (MODE == kSamplesAndPhase) S2[o] = e;
      if constexpr (MODE == kSamplesAndTangent) S2[o] = cmul(amp_dot, e);
      if constexpr (MODE == kSamplesAndFullTangent) {
        const T dphi = phase_value(v.phase_dot, v.delay_dot, freq_f, dnu_u, dt_v, a, r, f, t);
        S2[o] = cadd(cmul(amp_dot, e), cscale(dphi, ctimes_i(sample)));
      }
    }
  }
}

} // namespace gpu
} // namespace ri_kernels
