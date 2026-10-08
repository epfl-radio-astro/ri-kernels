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

`RFIVisOp` computes the per-baseline polarised RFI visibility

```
vis[bl, f, t, i, j] = mean over (n_int_freq, n_int_time) of
                      sum over n_rfi and c of
                      A[a1, ..., i, c] conj(A[a2, ..., j, c]) exp(i (φ[a1] - φ[a2]))
```

for signal factors `A` shaped `(n_ant, n_freq, n_time, n_rfi, n_int_freq,
n_int_time, P, 2)` and phases `φ` in radians shaped `(n_ant, n_freq, n_time,
n_rfi, n_int_freq, n_int_time)`, giving an output of shape `(n_baselines,
n_freq, n_time, P, P)`. It is constructed from the baseline layout and
evaluated through `eval`:

```python
from ri_kernels.jax_api import RFIVisOp

vis = RFIVisOp(n_ant, a1, a2).eval(rfi_amp_fine, rfi_phase)
```

As for `RFIAnalyticVisOp` below, the signal is a two-column factor per
receiver (P is 1 or 2). Pad a rank-one signal with a zero second column; a
zero column still permits X/Y response and leakage. P = 1 gives one
same-receiver correlation such as XX, and P = 2 gives `[[XX, XY], [YX, YY]]`.
The phase is common to every receiver and column, so the kernels compute one
phase factor per fine sample for all P × P entries. A reversed baseline is the
conjugate transpose, which exchanges XY and YX. `eval` moves the `(P, 2)` axes
in front of the fine-sample axes before calling the kernels, which keeps each
component contiguous for SIMD and coalesced loads.

The operator has native primal, JVP, and transpose kernels for CPU, CUDA, and
ROCm, so both forward- and reverse-mode differentiation stay inside the
kernels. Precision has to match across the inputs: complex64 with float32, or
complex128 with float64. Both amplitudes and phases are differentiated.

## Analytic RFI visibilities

`RFIAnalyticVisOp` integrates the time polynomial against phase moments, with
frequency quadrature unchanged. It stages amplitude coefficients per antenna
and cell, then contracts pairs of antenna tiles. The default time scratch axis
has three coefficients, independent of fringe winding. The signal is a
two-column factor per receiver, so one call returns a single correlation
(P = 1) or the full 2 × 2 polarisation matrix (P = 2) of every baseline.

```python
from ri_kernels.jax_api import RFIAnalyticVisOp

vis = RFIAnalyticVisOp(n_ant, a1, a2).eval(
    amp, phase, delay_us, w_freq, start_freq, g_time, start_time,
    dnu_mhz, int_time, freq_mhz,
    segments=2, terms=6, cubic_terms=3,
)
```

- `amp`, complex `(n_ant, n_rfi, n_freq, n_time, P, 2)`: the signal factor on
  the data grid, per receiver (P is 1 or 2) and latent column. There are
  always two columns; pad a rank-one signal with a zero second column (a
  zero column still permits X/Y response and leakage). Factors may be fitted
  directly or built upstream, e.g. as Jones matrices times a source factor;
  the kernel is agnostic.
- `phase`, real `(n_ant, n_rfi, n_freq, n_time)`: the phase at the channel and
  cell centre, reduced to one turn (in float64, before casting). It is common
  to every receiver and column.
- `delay_us`, real `(n_ant, n_rfi, n_time, n_path)`: the geometric delay in
  microseconds, with the sign that makes the phase `2π f τ`, and its first
  `n_path - 1` time derivatives at the cell centre, relative to the array mean
  (a common term cancels in every baseline; the full delay's change across a
  cell is ~1e4 wavelengths, beyond float32).
- `w_freq`, `start_freq`: per channel, the weights that turn its stencil of
  `n_sf` neighbouring channels into its fine channels, and the first channel
  of the stencil.
- `g_time`, `start_time`: per cell, monomial coefficients in
  `x = 2*tau/int_time`, as `tabascal.poly_interp.monomial_tables` produces
  them, including the shifted edge stencils, and the first cell of the
  stencil. Each stencil must lie inside its axis and contain its cell.
- `dnu_mhz`: the fine offsets from the channel centre; `freq_mhz`: the channel
  centres. Both are in **MHz**; divide `config.freqs` in Hz by `1e6`. MHz × μs
  is cycles.
- `int_time`: a positive scalar in seconds.

The result has shape `(n_bl, n_freq, n_time, P, P)`, summed over the sources:

    V[b, ..., i, j] = Σ_rfi ∫ Σ_c amp[a1, ..., i, c] conj(amp[a2, ..., j, c]) e^{iΦ}

P = 1 gives one same-receiver correlation such as XX, and P = 2 gives
`[[XX, XY], [YX, YY]]`. Every matrix entry and both columns share one set of
phase moments per baseline, so the moment cost does not grow with P. A
reversed baseline is the conjugate transpose, which exchanges XY and YX.

Every real operand shares the precision of `phase`; the stencil starts are
int32. The constructor builds, per pair of 32-antenna tiles, the list of the
pairs the baselines cover there (`tile_pair_list`), which the staged GPU
kernels work through a tile pair per block; each `(a1, a2)` baseline may
appear at most once.

`amp` and `phase` are differentiated; `delay_us`, `int_time` and the tables
are constants whose gradients `eval` stops. Two kernel pairs carry the
derivatives: a JVP and transpose for the signal alone,
and a full pair that takes the phase tangent as well and returns its
cotangent. The phase enters every cell's weights as the one factor
`exp(i (phase[a1] - phase[a2]))`, so the phase tangent turns the pair's
product by `i` (`dV = i (dphase[a1] - dphase[a2]) V` per source and fine
channel, for every matrix entry) and the cotangent is `-Im(Σ_ij g_ij V_ij)` on
`a1` and the opposite on `a2`, summed over the baseline's sources and fine
channels; both cost one extra
contraction against weights the kernel has already formed. The JVP rule binds
the signal-only pair whenever the phase tangent is a symbolic zero, so a
fixed orbit computes no phase derivative. The compiled JVPs and transposes
follow JAX's complex cotangent convention.

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

GPU scratch is bounded by the `scratch_mb` argument of `eval` (256 MiB by
default), with time chunks and, when necessary, frequency chunks. A smaller
budget means more chunks, and each chunk of a transpose passes over the whole
signal cotangent. The transpose reduces coefficient cotangents within each tile
in shared memory, writes separate partials for partner tiles, and gathers
through the interpolation stencils. Shared-memory atomics mean the last bits of
the GPU transpose can vary between runs. The staged tiles grow with P; when
they exceed the default 48 KiB (double precision, P = 2, wide stencils) the
kernel opts in to the card's larger shared-memory limit.

After building on the GPU host, run:

```bash
python -m pytest tests/test_rfi_analytic_vis_op.py
python tests/benchmark_rfi_analytic_vis_op.py --antennas 256 512 --iterations 100
```

The analytic tests compare both precisions, P = 1 and 2, and the amplitude
and phase derivatives against a frozen float64 JAX reference; the polarised
matrix is recovered from its scalar form by the polarisation identity. They
cover zero winding, both recurrence branches, the Fresnel boundary, cubic
translation, wider coefficient counts, sparse/reversed/autocorrelation
baselines, partial antenna tiles and forced scratch splitting. The benchmark
reports synchronised forward, JVP and VJP times, signal-only and with the
phase, after compilation and warmup. It uses
synthetic inputs; repeat the scientific accuracy check and the 100-iteration
optimisation with the production SKA-Low data. Inspect ptxas register/spill
reports and Nsight local-memory traffic for the default specialisation (at
`--pol 1` and `--pol 2`: P = 2 holds four accumulators per pair) before
interpreting its speedup. CUDA compilation, device tests and performance need
the GPU host; CPU tests do not establish those results.
