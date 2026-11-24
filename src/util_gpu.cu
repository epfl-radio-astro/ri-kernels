#include "util_gpu.h"

#include "gpu_compat.h"

namespace ri_kernels {
namespace gpu {

cudaError_t get_device_limits(DeviceLimits &out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  int shared_mem = 0;
  if ((err = cudaDeviceGetAttribute(&out.max_grid[0], cudaDevAttrMaxGridDimX,
                                    device)) != cudaSuccess)
    return err;
  if ((err = cudaDeviceGetAttribute(&out.max_grid[1], cudaDevAttrMaxGridDimY,
                                    device)) != cudaSuccess)
    return err;
  if ((err = cudaDeviceGetAttribute(&out.max_grid[2], cudaDevAttrMaxGridDimZ,
                                    device)) != cudaSuccess)
    return err;
  if ((err = cudaDeviceGetAttribute(&shared_mem,
                                    cudaDevAttrMaxSharedMemoryPerBlock,
                                    device)) != cudaSuccess)
    return err;
  out.shared_mem_per_block =
      shared_mem > 0 ? static_cast<std::size_t>(shared_mem) : 0;
  return cudaSuccess;
}

} // namespace gpu
} // namespace ri_kernels
