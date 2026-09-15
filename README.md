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

vis = RFIInterpVisOp(n_ant, a1, a2).eval(
    amp, phase, delay_us, w_freq, start_freq, w_time, start_time, dnu_mhz, dt, freq_mhz
)
```

The constructor builds, per pair of 32-antenna tiles, the list of the pairs
the baselines cover there (`tile_pair_list`); the staged GPU kernels work a
tile pair per block from the fine samples materialised once per cell.
Variable sampling per baseline group is a matter of calling the operator per
group with the time tables cut to the group's samples.

The operator differentiates the signal, the phase and the delay polynomial,
and carries two kernel pairs for it: a JVP and transpose for the signal
alone, and a full pair that takes the phase and delay tangents as well
(`dS = i dphi S`) and returns their cotangents (per fine sample
`-Im(S G)`, `G` the cotangent factor before the phase; summed over the
cell's samples for the phase, weighted by `d phi / d delay[k]` for the
delay). The JVP rule binds the signal-only pair whenever the phase and delay
tangents are symbolic zeros, a fixed orbit with nothing learnable upstream
of them, so such a run computes no phase derivative; a fitted trajectory
gets the full pair without a switch.

- `amp`, complex `(n_ant, n_rfi, n_freq, n_time)`: the signal on the data
  grid. Differentiated, as are `phase` and `delay_us`.
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
vis[bl, f, t] = mean_{u, v} sum_r A[a1] exp(i phi[a1]) conj(A[a2] exp(i phi[a2]))
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

## Analytic RFI visibilities

`RFIAnalyticVisOp` integrates the time polynomial against phase moments, with
frequency quadrature unchanged. It stages amplitude coefficients per antenna
and cell, then contracts pairs of antenna tiles. The default time scratch axis
has three coefficients, independent of fringe winding.

```python
from ri_kernels.jax_api import RFIAnalyticVisOp

vis = RFIAnalyticVisOp(n_ant, a1, a2).eval(
    amp, phase, delay_us, w_freq, start_freq, g_time, start_time,
    dnu_mhz, int_time, freq_mhz,
    segments=2, terms=6, cubic_terms=3,
)
```

The operands follow `RFIInterpVisOp`'s axis order and precision contract.
`g_time[t,l,m]` contains monomial coefficients in `x = 2*tau/int_time`, as
`tabascal.poly_interp.monomial_tables` produces them, including the shifted edge
stencils. `int_time` is a positive scalar in seconds. Channel centres and
frequency offsets are in **MHz**; divide `config.freqs` in Hz by `1e6`.
Only `amp` is differentiated. The compiled JVP and transpose follow JAX's
complex cotangent convention; `eval` stops gradients on the remaining inputs.

The moment branches, segmentation and cubic translation follow
`tabascal.coarse_rfi_vis.analytic_rfi_vis`. `segments`, `terms`, and
`cubic_terms` are static options: close over them or mark them static when
jitting. The defaults use two pieces, six curvature terms and three cubic
terms. `segments=4, terms=16` selects the reference's conservative settings.
Zero cubic terms deliberately omits cubic phase; delay derivatives above
order three are always omitted. Coefficient counts 1–9, segments 1–1024,
curvature terms 1–32 and cubic terms 0–8 are supported. These limits bound
local storage; convergence still depends on the supplied delay and interval.

Every angle -- the centre phase, the winding and the segment translations --
is formed and reduced in double for both input precisions: a single-precision
angle has no fractional turn left at the magnitudes the delay reaches. The
moment recurrences that follow only multiply the coefficient buffers, so the
CPU kernels keep them in double, where it is free, while the GPU kernels run
them at the operator's own precision: a consumer card retires one
double-precision instruction per sixty-four single-precision ones, and a
complex64 weight costs well over an order of magnitude more in its double form
there. Fresnel seeds stay in double either way, and use a
small-argument series and rational auxiliary functions: a two-term asymptotic
near 2.5 is insufficient for the higher moments. The common three-coefficient,
six-term, three-cubic-term path has unrolled moment orders so its CUDA
recurrence can stay in registers. Wider configurations use a bounded general
implementation.

GPU scratch uses `RI_KERNELS_INTERP_SCRATCH_MB` (256 MiB by default), with time
chunks and, when necessary, frequency chunks. The transpose reduces coefficient
cotangents within each tile in shared memory, writes separate partials for
partner tiles, and gathers through the interpolation stencils. Shared-memory
atomics mean the last bits of the GPU transpose can vary between runs.

After building on the GPU host, run:

```bash
python -m pytest tests/test_rfi_analytic_vis_op.py tests/test_rfi_interp_vis_op.py
python tests/benchmark_rfi_analytic_vis_op.py --antennas 256 512 --iterations 100
```

The analytic tests compare both precisions and both amplitude derivatives
against a frozen float64 JAX reference. They cover zero winding, both recurrence
branches, the Fresnel boundary, cubic translation, wider coefficient counts,
sparse/reversed/autocorrelation baselines, partial antenna tiles and forced
scratch splitting. The benchmark reports synchronised forward, JVP and VJP
times after compilation and warmup, alongside 6571-sample quadrature. It uses
synthetic inputs; repeat the scientific accuracy check and the 100-iteration
optimisation with the production SKA-Low data. Inspect ptxas register/spill
reports and Nsight local-memory traffic for the default specialisation before
interpreting its speedup. CUDA compilation, device tests and performance need
the GPU host; CPU tests do not establish those results.

### Analytic memory diagnostics

Build the CPU extension with both sanitizers and debug assertions, then run
**the whole Python file against that extension**. The native bounds test uses
independent allocations and is useful alongside this run, but it does not
exercise XLA's FFI buffer handling.

```bash
cmake -S . -B /tmp/ri-analytic-asan \
  -DRI_KERNELS_CPU=ON -DRI_KERNELS_CUDA=OFF \
  -DRI_KERNELS_SANITIZE=ON -DRI_KERNELS_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build /tmp/ri-analytic-asan -j 4
ctest --test-dir /tmp/ri-analytic-asan --output-on-failure
```

Preload the ASan runtime belonging to the compiler used for that build before
starting the Python test runner. For a Linux GCC build:

```bash
LD_PRELOAD="$(c++ -print-file-name=libasan.so)" \
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python -u tests/run_analytic_sanitizers.py /tmp/ri-analytic-asan/libri_kernels.so
```

On macOS with Apple Clang, use
`DYLD_INSERT_LIBRARIES="$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)"`
in place of `LD_PRELOAD`. If Command Line Tools cannot find C++ headers such
as `<atomic>`, add
`-DCMAKE_CXX_FLAGS="-nostdinc++ -isystem $(xcrun --show-sdk-path)/usr/include/c++/v1"`
to the configure command. The runner forces the CPU backend, prints the JAX
versions and the loaded library path, and disables pytest output capture. It
stages the wrapper and shared object in a temporary package to prevent an
editable install from substituting another build. No library is placed in the
checkout. Additional pytest arguments, such as `-k general_degree_bounds`,
can narrow a follow-up run after the full sequence has been investigated.

All analytic stack capacities now come from `rfi_analytic_limits.hpp`'s
parameter limits and a single size calculation. Compile-time checks tie the
largest convolution, cubic and curvature indices to those capacities; the FFI
handlers reject unsupported configurations before staging or scheduling work.
The sizes remain exact: bounds protection does not rely on spare elements.
