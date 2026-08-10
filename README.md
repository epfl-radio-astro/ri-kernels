# RI Kernels

CPU and GPU kernels for radio interferometry, exposed to JAX as FFI custom
calls with full support for automatic differentiation.

## Installation

The CPU kernels and all of the Python code live in `ri_kernels`. The GPU kernels
ship as add-on packages, one per CUDA major version, selected with an extra:

```bash
pip install ri_kernels               # CPU kernels only
pip install "ri_kernels[cuda12]"     # + CUDA 12 kernels and jax[cuda12]
pip install "ri_kernels[cuda13]"     # + CUDA 13 kernels and jax[cuda13]
```

## Building from source

Requires CMake >= 3.20, a C++20 compiler, and network access at configure time
(Google Highway is fetched by CMake).

```bash
pip install .                      # CPU only
RI_KERNELS_CUDA=1 pip install .    # CPU + CUDA, needs nvcc
RI_KERNELS_ROCM=1 pip install .    # CPU + ROCm, needs hipcc
```

Both libraries land in the `ri_kernels` package directory in that case, which is
also a location the loader searches. Note that `pip install .[cuda12]` is a
different thing: the extra resolves `ri_kernels_cuda12` from the index, so it
needs a published release. To build the GPU kernels from the checkout, use
`RI_KERNELS_CUDA=1`.

Other CMake options of note: `RI_KERNELS_CPU` (default `ON`),
`RI_KERNELS_MULTI_ARCH` (dynamic SIMD dispatch, default `ON` — turn it off and
set arch flags via `CMAKE_CXX_FLAGS` for a single-target build),
`RI_KERNELS_BUNDLED_HIGHWAY`, and `CMAKE_CUDA_ARCHITECTURES`.

## Tests

Tests can be run with

```bash
python -m pytest tests
```
