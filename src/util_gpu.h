#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include "gpu_compat.h"

namespace ri_kernels {
namespace gpu {

struct DeviceLimits {
  int max_grid[3];
  std::size_t shared_mem_per_block;
};

cudaError_t get_device_limits(DeviceLimits &out);

inline dim3 create_clamped_grid(std::int64_t x, std::int64_t y, std::int64_t z,
                                const DeviceLimits &limits) {
  auto clamp = [](std::int64_t v, int hi) {
    return static_cast<unsigned int>(std::clamp<std::int64_t>(v, 1, hi));
  };
  return dim3(clamp(x, limits.max_grid[0]),
              clamp(y, limits.max_grid[1]),
              clamp(z, limits.max_grid[2]));
}

} // namespace gpu
} // namespace ri_kernels
