// Scratch is reused across time chunks, keeping every channel together when
// one full time cell fits. Only a cell that exceeds the budget is split across
// frequency. The transpose keeps its smaller partials across all channels so
// that both gathers retain their stencil reach and their summation order.
// Allocation capacities are separate from the actual extents of a tail chunk;
// kernels pack the samples using those actual extents.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <string>

#include "rfi_interp_common.hpp"

namespace ri_kernels {
namespace gpu {

#ifndef RI_INTERP_SCRATCH_MB
#define RI_INTERP_SCRATCH_MB 256u
#endif

// The build-time default can be overridden once per process, before the first
// call. Nonpositive environment values leave the build-time default in place.
inline std::int64_t interp_scratch_budget() {
  static const std::int64_t budget = [] {
    std::int64_t mb = RI_INTERP_SCRATCH_MB;
    if (const char *env = std::getenv("RI_KERNELS_INTERP_SCRATCH_MB")) {
      const long value = std::strtol(env, nullptr, 10);
      if (value > 0) mb = value;
    }
    constexpr std::int64_t mib = 1024 * 1024;
    constexpr auto limit = std::numeric_limits<std::int64_t>::max();
    return mb > limit / mib ? limit : mb * mib;
  }();
  return budget;
}

inline bool interp_checked_product(std::initializer_list<std::int64_t> factors,
                                   std::int64_t &result) {
  result = 1;
  for (const auto factor : factors) {
    if (factor < 0 || (factor != 0 && result > std::numeric_limits<std::int64_t>::max() / factor))
      return false;
    result *= factor;
  }
  return true;
}

inline bool interp_checked_sum(std::int64_t a, std::int64_t b, std::int64_t &result) {
  if (a < 0 || b < 0 || a > std::numeric_limits<std::int64_t>::max() - b) return false;
  result = a + b;
  return true;
}

inline std::int64_t interp_ceil_div(std::int64_t n, std::int64_t d) {
  return n / d + (n % d != 0);
}

struct InterpChunkPlan {
  std::int64_t n_tc, n_fc;
  std::int64_t sample_bytes, h_bytes, total_bytes;
};

template <typename INT_T> struct InterpCellChunk {
  INT_T t0, n_tc, f0, n_fc;
};

// h_per_time is frequency-complete; sample_per_freq is one sample buffer for
// one channel and time cell. All byte arithmetic stays checked and in 64 bits.
inline ffi::Error make_interp_chunk_plan(
    std::int64_t n_time, std::int64_t n_freq, std::int64_t sample_per_freq,
    std::int64_t h_per_time, std::int64_t n_sample_buffers, InterpChunkPlan &plan) {
  if (n_time < 1 || n_freq < 1 || sample_per_freq < 1 || h_per_time < 0 || n_sample_buffers < 1)
    return ffi::Error::InvalidArgument("Expected nonempty interpolation cells and samples");
  const auto overflow = [] {
    return ffi::Error::InvalidArgument("Interpolation scratch size exceeds the 64-bit byte range");
  };
  std::int64_t per_freq, minimum, full_samples, per_time;
  if (!interp_checked_product({sample_per_freq, n_sample_buffers}, per_freq) ||
      !interp_checked_sum(h_per_time, per_freq, minimum) ||
      !interp_checked_product({per_freq, n_freq}, full_samples) ||
      !interp_checked_sum(h_per_time, full_samples, per_time)) return overflow();

  const auto budget = interp_scratch_budget();
  if (minimum > budget)
    return ffi::Error::InvalidArgument(
        "Interpolation scratch needs at least " + std::to_string(minimum) +
        " bytes for one frequency and time cell (including frequency-complete partials), "
        "but RI_KERNELS_INTERP_SCRATCH_MB allows " + std::to_string(budget) + " bytes");

  plan.n_tc = 1;
  plan.n_fc = n_freq;
  if (per_time <= budget) {
    plan.n_tc = std::min(n_time, budget / per_time);
  } else {
    const auto w_max = (budget - h_per_time) / per_freq;
    const auto n_chunks = interp_ceil_div(n_freq, w_max);
    plan.n_fc = interp_ceil_div(n_freq, n_chunks);
  }
  std::int64_t sample_total;
  if (!interp_checked_product({sample_per_freq, plan.n_fc, plan.n_tc}, plan.sample_bytes) ||
      !interp_checked_product({h_per_time, plan.n_tc}, plan.h_bytes) ||
      !interp_checked_product({plan.sample_bytes, n_sample_buffers}, sample_total) ||
      !interp_checked_sum(plan.h_bytes, sample_total, plan.total_bytes)) return overflow();
  if (std::uint64_t(plan.total_bytes) > std::numeric_limits<std::size_t>::max()) return overflow();
  return ffi::Error::Success();
}

// Clamp a grid-stride axis while it is still wide; casting a large work count
// to int before create_clamped_grid would wrap instead of clamping it.
inline int interp_grid_extent(std::int64_t count) {
  return int(std::min<std::int64_t>(count, std::numeric_limits<int>::max()));
}

} // namespace gpu
} // namespace ri_kernels
