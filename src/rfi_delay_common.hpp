#pragma once

#include <cstddef>

#include "xla/ffi/api/ffi.h"

// Shared by the six delay translation units - the forward, JVP and transpose
// kernels of both the CPU and the GPU library.

namespace ri_kernels {

namespace ffi = xla::ffi;

// Type aliases to avoid commas inside XLA_FFI_DEFINE_HANDLER_SYMBOL macro args.
using delay_amp_f32_t = ffi::Buffer<ffi::C64, 6>;
using delay_amp_f64_t = ffi::Buffer<ffi::C128, 6>;
using delay_real4_f32_t = ffi::Buffer<ffi::F32, 4>;
using delay_real4_f64_t = ffi::Buffer<ffi::F64, 4>;
using delay_real2_f32_t = ffi::Buffer<ffi::F32, 2>;
using delay_real2_f64_t = ffi::Buffer<ffi::F64, 2>;

template <typename T> constexpr T two_pi() {
  return T(6.283185307179586476925286766559005768L);
}

// amp layout:   (n_ant, n_freq, n_time, n_rfi, n_int_f, n_int_t)
// delay layout: (n_ant, n_time, n_rfi, n_int_t)
// freq layout:  (n_freq, n_int_f)
template <ffi::DataType AMP_DT, ffi::DataType REAL_DT>
bool shapes_are_valid(ffi::BufferR1<ffi::S32> a1, ffi::BufferR1<ffi::S32> a2,
                      ffi::Buffer<AMP_DT, 6> amp,
                      ffi::Buffer<REAL_DT, 4> delay,
                      ffi::Buffer<REAL_DT, 2> freq) {
  return a1.dimensions()[0] == a2.dimensions()[0] &&
         delay.dimensions()[0] == amp.dimensions()[0] &&
         delay.dimensions()[1] == amp.dimensions()[2] &&
         delay.dimensions()[2] == amp.dimensions()[3] &&
         delay.dimensions()[3] == amp.dimensions()[5] &&
         freq.dimensions()[0] == amp.dimensions()[1] &&
         freq.dimensions()[1] == amp.dimensions()[4];
}

// Tangents and cotangents are read through views built from the primal
// extents, so a mismatched buffer would run off the end rather than fail.
template <typename LHS, typename RHS>
bool same_shape(const LHS &lhs, const RHS &rhs) {
  const auto a = lhs.dimensions();
  const auto b = rhs.dimensions();
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

} // namespace ri_kernels
