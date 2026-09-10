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

## RFI visibilities

Two operators compute the same quantity, the per-baseline RFI visibility

```
vis[bl, f, t] = mean over (n_int_freq, n_int_time) of
                sum over n_rfi of A[a1] conj(A[a2]) exp(i (φ[a1] - φ[a2]))
```

for amplitudes `A` shaped `(n_ant, n_freq, n_time, n_rfi, n_int_freq,
n_int_time)`, giving an output of shape `(n_baselines, n_freq, n_time)`. They
differ only in how the phase `φ` is supplied. Both are constructed from the
baseline layout and evaluated through `eval`:

```python
from ri_kernels.jax_api import RFIVisOp, RFIDelayVisOp

vis = RFIVisOp(n_ant, a1, a2).eval(rfi_amp_fine, rfi_phase)
vis = RFIDelayVisOp(n_ant, a1, a2).eval(rfi_amp_fine, rfi_delay_us, freq_mhz)
```

Each operator has native primal, JVP, and transpose kernels for CPU, CUDA, and
ROCm, so both forward- and reverse-mode differentiation stay inside the
kernels. Precision has to match across the inputs: complex64 with float32, or
complex128 with float64.

### Explicit phases

`RFIVisOp` takes the phase in radians as a full array with the same shape as
the amplitudes. Both amplitudes and phases are differentiated.

### Delay-based phases

`RFIDelayVisOp` avoids expanding geometric delays over every frequency sample.
Instead of the phase array it accepts:

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

## RFI visibilities from the data grid

`RFIInterpVisOp` computes the same visibility as the two operators above from
inputs that live on the *data grid* only. The RFI signal comes in per data
cell, the phase as its value at the cell centre plus the time derivatives of
the path there, and the fine samples inside each cell are rebuilt inside the
kernel from the cell's stencil of neighbouring cells and interpolation tables
the caller supplies. Nothing of the fine grid is ever read from or written to
memory.

```python
from ri_kernels.jax_api import RFIInterpVisOp

vis = RFIInterpVisOp(n_ant, a1, a2, stride=None).eval(
    amp, phase, delay_us, w_freq, start_freq, w_time, start_time, dnu_mhz, dt, freq_mhz
)
```

`stride`, int `(n_bl,)`, is every how-many-th time sample of a cell each
baseline integrates, from sample `stride // 2` (1, the default, is all of
them): the variable sampling of tabascal's `RiemannVisVariable`, done inside
the kernel, so the slow baselines of an array cost a fraction of the fast ones.

- `amp`, complex `(n_ant, n_rfi, n_freq, n_time)`: the signal on the data
  grid, and the only differentiated input.
- `phase`, real `(n_ant, n_rfi, n_freq, n_time)`: the phase at the channel and
  cell centre, reduced to one turn (in float64, before casting).
- `delay_us`, real `(n_ant, n_rfi, n_time, n_path)`: the geometric delay in
  microseconds, with the sign that makes the phase `2π f τ` as `RFIDelayVisOp`
  has it, and its first `n_path - 1` time derivatives at the cell centre,
  relative to the array mean (a common term cancels in every baseline; the
  full delay's change across a cell is ~1e4 wavelengths, beyond float32).
- `w_freq`, `start_freq` and `w_time`, `start_time`: per cell, the weights
  that turn its stencil of `n_sf` (`n_st`) neighbouring cells into its
  `n_int_freq` (`n_int_time`) fine samples, and the first cell of the stencil.
  Each stencil must lie inside its axis and contain its cell.
- `dnu_mhz`, `dt`: the fine offsets from the channel centre (MHz) and the
  cell centre (s); `freq_mhz`: the channel centres (MHz). MHz × μs is cycles.

For one cell `(f, t)`, fine sample `(u, v)`, source `r` and antenna `a`:

```
A[a]   = sum_k sum_l w_freq[f, k, u] w_time[t, l, v] amp[a, r, start_freq[f] + k, start_time[t] + l]
dtau[a] = sum_{k >= 1} delay_us[a, r, t, k] dt[v]^k / k!
phi[a]  = phase[a, r, f, t] + 2 pi ((freq_mhz[f] + dnu_mhz[u]) dtau[a] + dnu_mhz[u] delay_us[a, r, t, 0])
vis[bl, f, t] = mean_{u, v in stride[bl]} sum_r A[a1] exp(i phi[a1]) conj(A[a2] exp(i phi[a2]))
```

The weights are data. The polynomial through the stencil, the conditional mean
of a Gaussian process prior, or any other linear interpolant is a different
table through the same kernel. The phase, delay and tables are constants of a
run: `eval` stops their gradients, and the JVP and transpose kernels
differentiate the signal alone. The transpose is deterministic -- every output
element is written by exactly one thread -- at the cost of a scratch buffer of
`n_sf * n_st` times the signal on the GPU.

These kernels are a prototype of the operator rather than a fast
implementation: each baseline rebuilds both of its antennas' fine samples
itself, so an antenna's samples are recomputed once per baseline it is on. A
kernel meant to be fast would stage each antenna's samples once per cell and
reuse them across its baselines.
