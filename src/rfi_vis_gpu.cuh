#pragma once

#include "gpu_compat.h"

namespace ri_kernels {
namespace gpu {

// N real partial sums that one cub::BlockReduce combines at once: the real
// and imaginary parts of a cell's P x P visibility entries, k = 2 * (i * P + j)
// and k + 1.
template <typename T, int N> struct RfiPartials {
  T v[N];

  __host__ __device__ RfiPartials operator+(const RfiPartials &other) const {
    RfiPartials result;
#pragma unroll
    for (int k = 0; k < N; ++k)
      result.v[k] = v[k] + other.v[k];
    return result;
  }
};

} // namespace gpu
} // namespace ri_kernels
