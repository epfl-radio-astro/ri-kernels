#include "util_gpu.h"

#include "gpu_compat.h"
#include <mutex>

namespace ri_kernels {
namespace gpu {

namespace {

// Devices whose limits are cached; higher ordinals are still queried, just
// not cached.
constexpr int kMaxCachedDevices = 64;

cudaError_t query_device_limits(int device, DeviceLimits &out) {
  cudaError_t err;
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

} // namespace

cudaError_t get_device_limits(DeviceLimits &out) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device < 0 || device >= kMaxCachedDevices)
    return query_device_limits(device, out);

  // Limits are fixed per device, so each one is queried once. A failed query
  // is not cached and is retried on the next call.
  static std::mutex mutex;
  static DeviceLimits cache[kMaxCachedDevices];
  static bool cached[kMaxCachedDevices] = {};

  std::lock_guard<std::mutex> lock(mutex);
  if (!cached[device]) {
    DeviceLimits limits;
    if ((err = query_device_limits(device, limits)) != cudaSuccess) return err;
    cache[device] = limits;
    cached[device] = true;
  }
  out = cache[device];
  return cudaSuccess;
}

} // namespace gpu
} // namespace ri_kernels
