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
`RI_KERNELS_BUNDLED_HIGHWAY`, `CMAKE_CUDA_ARCHITECTURES`, and
`RI_KERNELS_STATIC_CUDART` (default `ON`; set it to `OFF` when a package
manager should provide the shared CUDA runtime).

## Tests

Tests can be run with

```bash
python -m pytest tests
```

## Delay-based RFI visibilities

`RFIDelayVisOp` avoids expanding geometric delays over every frequency sample.
The native primal, JVP, and transpose kernels support CPU, CUDA, and ROCm. It
accepts:

- amplitudes shaped
  `(n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time)`;
- delays in μs shaped `(n_ant, n_time, n_rfi, n_int_time)`; and
- absolute frequencies in MHz shaped `(n_freq, n_int_freq)`.

Since MHz × μs is cycles, the baseline phase is `2π f_MHz Δτ_μs`. The delay
input is smaller than an expanded phase array by `n_freq × n_int_freq`; the
frequency input contains only `n_freq × n_int_freq` elements. JVP and VJP rules
differentiate amplitudes and delays. Frequencies are fixed coordinates and
receive a zero cotangent.

Absolute satellite delays are about `116,747 μs` and must not be converted
directly to float32. Before calling the kernel, subtract a common delay across
antennas for each time/source/sub-time sample while still in float64, then cast
the centred result. This is exactly visibility-invariant because only `Δτ`
enters a baseline.

Float32 is intended for ordinary arrays with maximum antenna separations of
about 10 km, corresponding to `|Δτ| ≲ 33.4 μs`. At 1 GHz the worst-case phase
resolution is about `0.025 rad`. Use float64 for exceptional arrays approaching
100 km. Around 1 GHz, float32 frequency resolution is about `61 Hz`, comfortably
below both the expected minimum `10 kHz` spacing (`0.01 MHz`) and the more
typical `0.2 MHz` spacing.
