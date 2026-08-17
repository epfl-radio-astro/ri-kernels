#pragma once

#include <algorithm>
#include "gpu_compat.h"

namespace ri_kernels {
namespace gpu {

const cudaDeviceProp &get_device_prop();

template <typename T> __device__ inline T two_pi() {
  return T(6.283185307179586476925286766559005768L);
}

inline dim3 create_clamped_grid(int x, int y, int z) {
  const auto &prop = get_device_prop();
  return dim3(std::min<int>(x, prop.maxGridSize[0]),
              std::min<int>(y, prop.maxGridSize[1]),
              std::min<int>(z, prop.maxGridSize[2]));
}

} // namespace gpu
} // namespace ri_kernels
