#pragma once

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ri_kernels {

// Both operators, RFIVisOp and RFIAnalyticVisOp, take a signal with P
// receivers per antenna (one or two) and two latent columns, a rank-one signal
// padded with a zero second column. Their kernels flatten (P, 2) into the
// component e = i * kRfiColumns + c, row-major like the buffer, and return
// (n_bl, n_freq, n_time, P, P) visibilities.
constexpr int kRfiColumns = 2;
constexpr int kRfiMaxPol = 2;

// The signal of RFIVisOp, as the kernels see it, is
// (n_ant, n_freq, n_time, P, 2, n_rfi, n_int_freq, n_int_time). The Python
// wrapper moves (P, 2) in front of the reduction axes, so that each component
// is contiguous over n_red = n_rfi * n_int_freq * n_int_time. The phase is
// (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time), shared by every
// component.

// Checks a signal and its phase against each other.
template <typename AMP, typename PHASE>
xla::ffi::Error rfi_validate_signal(const AMP &amp, const PHASE &phase) {
  const auto a = amp.dimensions(), p = phase.dimensions();
  if (a[3] < 1 || a[3] > kRfiMaxPol || a[4] != kRfiColumns) {
    return xla::ffi::Error::InvalidArgument(
        "Expected rfi_amp_fine of shape (n_ant, n_freq, n_time, P, 2, n_rfi, "
        "n_int_freq, n_int_time) with P 1 or 2");
  }
  for (int i = 0; i < 3; ++i) {
    if (a[i] != p[i] || a[5 + i] != p[3 + i]) {
      return xla::ffi::Error::InvalidArgument(
          "Expected rfi_phase to match rfi_amp_fine without its (P, 2) axes");
    }
  }
  return xla::ffi::Error::Success();
}

// Checks a visibility buffer (value, tangent or cotangent) against the
// baselines and the signal: (n_bl, n_freq, n_time, P, P).
template <typename VIS, typename AMP>
xla::ffi::Error rfi_validate_vis(const VIS &vis, std::int64_t n_bl,
                                 const AMP &amp) {
  const auto v = vis.dimensions(), a = amp.dimensions();
  if (v[0] != n_bl || v[1] != a[1] || v[2] != a[2] || v[3] != a[3] ||
      v[4] != a[3]) {
    return xla::ffi::Error::InvalidArgument(
        "Expected visibilities of shape (n_bl, n_freq, n_time, P, P)");
  }
  return xla::ffi::Error::Success();
}

// Tangents and cotangents are read through views built from the primal
// extents, so a mismatched buffer would run off the end rather than fail.
template <typename X, typename Y> bool rfi_same_shape(const X &x, const Y &y) {
  const auto a = x.dimensions(), b = y.dimensions();
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (a[i] != b[i]) return false;
  return true;
}

// The product of the reduction axes, n_rfi * n_int_freq * n_int_time.
template <typename PHASE> std::int64_t rfi_n_red(const PHASE &phase) {
  const auto p = phase.dimensions();
  return p[3] * p[4] * p[5];
}

} // namespace ri_kernels
