#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "tensor.hpp"
#include "xla/ffi/api/ffi.h"

// The FFI operand types, the antenna tiling and the complex arithmetic shared
// by the analytic kernels of the CPU and the GPU library.

namespace ri_kernels {

namespace ffi = xla::ffi;

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using analytic_amp_f32_t = ffi::Buffer<ffi::C64, 4>;
using analytic_amp_f64_t = ffi::Buffer<ffi::C128, 4>;
using analytic_real4_f32_t = ffi::Buffer<ffi::F32, 4>;
using analytic_real4_f64_t = ffi::Buffer<ffi::F64, 4>;
using analytic_real3_f32_t = ffi::Buffer<ffi::F32, 3>;
using analytic_real3_f64_t = ffi::Buffer<ffi::F64, 3>;
using analytic_real1_f32_t = ffi::Buffer<ffi::F32, 1>;
using analytic_real1_f64_t = ffi::Buffer<ffi::F64, 1>;
using analytic_index_t = ffi::BufferR1<ffi::S32>;

// Antennas per tile in the staged GPU kernels. A tile pair is a fixed
// kAnalyticTilePairs entries wide because that is how the caller builds it --
// TILE in rfi_analytic_vis_op.py. The two definitions have to move together;
// every handler that takes a tile-pair list validates its width against this
// one.
//
// The value is the warp width on NVIDIA, which is what makes a tile diagonal
// bank-conflict-free, but nothing here depends on the wave size: the kernels
// synchronise through __syncthreads() and shared-memory atomics only, so a
// wave64 AMD part runs the same layout correctly.
constexpr int kAnalyticTile = 32;
constexpr int kAnalyticTilePairs = kAnalyticTile * kAnalyticTile;

// Tiles needed to cover n_ant antennas. The kernels index partial buffers by
// it, so it has to be callable on the device as well as in the handlers'
// shape checks.
TAB_H_D constexpr std::int64_t analytic_tile_count(std::int64_t n_ant) {
  return (n_ant + kAnalyticTile - 1) / kAnalyticTile;
}

// A complex number with the layout of std::complex<T> and of the CUDA/HIP
// complex types, so the FFI buffers of any of them can be viewed as this.
template <typename T> struct Cplx {
  T re, im;
};

// i * a
template <typename T> TAB_H_D inline Cplx<T> ctimes_i(Cplx<T> a) {
  return {-a.im, a.re};
}

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

template <typename T> TAB_H_D constexpr T two_pi_c() {
  return T(6.283185307179586476925286766559005768L);
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

// Tangents and cotangents are read through views built from the primal
// extents, so a mismatched buffer would run off the end rather than fail.
template <typename LHS, typename RHS>
bool analytic_same_shape(const LHS &lhs, const RHS &rhs) {
  const auto a = lhs.dimensions();
  const auto b = rhs.dimensions();
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

} // namespace ri_kernels
