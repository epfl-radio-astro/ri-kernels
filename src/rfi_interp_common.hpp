#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tensor.hpp"
#include "xla/ffi/api/ffi.h"

// Shared by the four interp translation units - the forward/JVP and the
// transpose kernels of the CPU and the GPU library.
//
// The operator: the RFI visibility from the *data grid*. The signal, the phase
// and the path derivatives come in per data cell, and the fine samples of each
// cell are rebuilt here, inside the reduction, from the cell's stencil of
// neighbouring cells and the tables the caller supplies. Nothing on the fine
// grid is ever read from or written to memory.
//
//   amp        (n_ant, n_rfi, n_freq, n_time)  complex  the signal on the data grid
//   phase      (n_ant, n_rfi, n_freq, n_time)  real     phase at the channel and cell
//                                                       centre, reduced to one turn
//   path       (n_ant, n_rfi, n_time, n_path)  real     path L (m) and its time
//                                                       derivatives L_k (m/s^k)
//   w_freq     (n_freq, n_sf, n_int_f)         real     interpolation weights across
//   start_freq (n_freq,)                       int32    each channel, and the first
//                                                       channel of each stencil
//   w_time     (n_time, n_st, n_int_t)         real     the same across each cell
//   start_time (n_time,)                       int32
//   dnu        (n_int_f,)                      real     fine offsets from the channel
//                                                       centre (Hz)
//   dt         (n_int_t,)                      real     fine offsets from the cell
//                                                       centre (s)
//   freqs      (n_freq,)                       real     channel centres (Hz)
//   vis        (n_bl, n_freq, n_time)          complex
//
// For one cell (f, t) and fine sample (u, v), per source r and antenna a:
//
//   A[a]   = sum_k sum_l w_freq[f, k, u] w_time[t, l, v]
//                        amp[a, r, start_freq[f] + k, start_time[t] + l]
//   dL[a]  = sum_{k >= 1} path[a, r, t, k] dt[v]^k / k!
//   phi[a] = phase[a, r, f, t]
//            - (2 pi / c) ((freqs[f] + dnu[u]) dL[a] + dnu[u] path[a, r, t, 0])
//   S[a]   = A[a] exp(i phi[a])
//   vis[bl, f, t] = mean_{u, v} sum_r S[a1[bl]] conj(S[a2[bl]])
//
// Only amp is differentiated. The result is bilinear in S and S is linear in
// amp, so the JVP is B(dS, S) + B(S, dS) with dS the tangent pushed through the
// same interpolation and phase factor, and the transpose scatters the
// visibility cotangent back to each antenna's fine samples, multiplies by that
// antenna's phase factor and contracts with the weight tables, one stencil per
// cell (see rfi_interp_transpose_kernel*.cu/cpp).
//
// The stencil of a cell must contain the cell: start[c] <= c < start[c] +
// n_stencil. The transpose gathers each data-grid cotangent from the cells
// whose stencils cover it, and relies on that to bound the search to the
// neighbouring cells. The CPU handlers check it; the GPU ones trust it.

namespace ri_kernels {

namespace ffi = xla::ffi;

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using interp_amp_f32_t = ffi::Buffer<ffi::C64, 4>;
using interp_amp_f64_t = ffi::Buffer<ffi::C128, 4>;
using interp_real4_f32_t = ffi::Buffer<ffi::F32, 4>;
using interp_real4_f64_t = ffi::Buffer<ffi::F64, 4>;
using interp_real3_f32_t = ffi::Buffer<ffi::F32, 3>;
using interp_real3_f64_t = ffi::Buffer<ffi::F64, 3>;
using interp_real1_f32_t = ffi::Buffer<ffi::F32, 1>;
using interp_real1_f64_t = ffi::Buffer<ffi::F64, 1>;
using interp_index_t = ffi::BufferR1<ffi::S32>;

// A complex number with the layout of std::complex<T> and of the CUDA/HIP
// complex types, so the FFI buffers of any of them can be viewed as this.
template <typename T> struct Cplx {
  T re, im;
};

template <typename T> TAB_H_D inline Cplx<T> cadd(Cplx<T> a, Cplx<T> b) {
  return {a.re + b.re, a.im + b.im};
}
template <typename T> TAB_H_D inline Cplx<T> cmul(Cplx<T> a, Cplx<T> b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <typename T> TAB_H_D inline Cplx<T> cconj(Cplx<T> a) {
  return {a.re, -a.im};
}
template <typename T> TAB_H_D inline Cplx<T> cscale(T s, Cplx<T> a) {
  return {s * a.re, s * a.im};
}

// 2 pi / c, c the speed of light in m/s as tabascal.interferometry has it.
template <typename T> TAB_H_D constexpr T two_pi_over_c() {
  return T(2.0958450219516816e-08L);
}

template <typename T> TAB_H_D inline void sincos_t(T x, T *s, T *c) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  if constexpr (std::is_same_v<T, float>) {
    sincosf(x, s, c);
  } else {
    sincos(x, s, c);
  }
#else
  *s = std::sin(x);
  *c = std::cos(x);
#endif
}

// The interpolated signal of antenna `ant`, source `r` at fine sample (u, v) of
// cell (f, t). `wf` and `wt` are the cell's rows of the two tables, laid out
// (n_sf, n_int_f) and (n_st, n_int_t); `sf` and `st` the stencil starts.
template <typename T, typename INT_T>
TAB_H_D inline Cplx<T> interp_amp(Tensor4D<const Cplx<T> *, INT_T> amp,
                                   const T *wf, const T *wt, INT_T sf,
                                   INT_T st, INT_T n_sf, INT_T n_st,
                                   INT_T n_int_f, INT_T n_int_t, INT_T ant,
                                   INT_T r, INT_T u, INT_T v) {
  Cplx<T> a{0, 0};
  for (INT_T k = 0; k < n_sf; ++k) {
    const T wk = wf[k * n_int_f + u];
    for (INT_T l = 0; l < n_st; ++l) {
      const T w = wk * wt[l * n_int_t + v];
      const Cplx<T> c = amp(ant, r, sf + k, st + l);
      a.re += w * c.re;
      a.im += w * c.im;
    }
  }
  return a;
}

// exp(i phi) of antenna `ant`, source `r` at fine sample (u, v) of cell (f, t):
// the reduced centre phase plus the change across the cell (the path's Taylor
// series in dt[v]) and across the channel (linear in dnu[u]). The centre phase
// is never rebuilt here from path[..., 0]: in single precision that is a
// million-turn product with no fraction of a turn left in it.
template <typename T, typename INT_T>
TAB_H_D inline Cplx<T> phase_factor(Tensor4D<const T *, INT_T> phase,
                                     Tensor4D<const T *, INT_T> path,
                                     T freq_f, T dnu_u, T dt_v, INT_T ant,
                                     INT_T r, INT_T f, INT_T t) {
  const INT_T n_path = path.shape[3];
  // Horner from the highest derivative down, each term divided by k!.
  T acc = 0;
  T inv_factorial = 1;
  for (INT_T k = 2; k < n_path; ++k) inv_factorial /= T(k);
  for (INT_T k = n_path - 1; k >= 1; --k) {
    acc = acc * dt_v + path(ant, r, t, k) * inv_factorial;
    inv_factorial *= T(k);
  }
  const T d_path = acc * dt_v;
  const T phi = phase(ant, r, f, t) -
                two_pi_over_c<T>() *
                    ((freq_f + dnu_u) * d_path + dnu_u * path(ant, r, t, 0));
  Cplx<T> e;
  sincos_t(phi, &e.im, &e.re);
  return e;
}

template <ffi::DataType AMP_DT, ffi::DataType REAL_DT>
bool interp_shapes_are_valid(
    interp_index_t a1, interp_index_t a2, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> path,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs) {
  const auto a = amp.dimensions();
  const auto n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const auto p = phase.dimensions();
  const auto l = path.dimensions();
  const auto wf = w_freq.dimensions();
  const auto wt = w_time.dimensions();
  return a1.dimensions()[0] == a2.dimensions()[0] &&
         p[0] == n_ant && p[1] == n_rfi && p[2] == n_freq && p[3] == n_time &&
         l[0] == n_ant && l[1] == n_rfi && l[2] == n_time && l[3] >= 1 &&
         wf[0] == n_freq && wf[1] >= 1 && wf[1] <= n_freq &&
         wt[0] == n_time && wt[1] >= 1 && wt[1] <= n_time &&
         start_freq.dimensions()[0] == n_freq &&
         start_time.dimensions()[0] == n_time &&
         dnu.dimensions()[0] == wf[2] && dt.dimensions()[0] == wt[2] &&
         freqs.dimensions()[0] == n_freq;
}

// Tangents and cotangents are read through views built from the primal
// extents, so a mismatched buffer would run off the end rather than fail.
template <typename LHS, typename RHS>
bool interp_same_shape(const LHS &lhs, const RHS &rhs) {
  const auto a = lhs.dimensions();
  const auto b = rhs.dimensions();
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// Host-side check of the stencil contract (see the top of this file).
inline bool stencils_cover_their_cells(const int *start, std::int64_t n_cells,
                                       std::int64_t n_stencil) {
  for (std::int64_t c = 0; c < n_cells; ++c) {
    if (start[c] < 0 || start[c] + n_stencil > n_cells || start[c] > c ||
        c >= start[c] + n_stencil)
      return false;
  }
  return true;
}

} // namespace ri_kernels
