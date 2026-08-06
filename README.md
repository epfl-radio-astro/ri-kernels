# ri_kernels

SIMD and GPU kernels for radio interferometry, exposed to JAX as FFI custom
calls.

## Installation

The CPU kernels and all of the Python code live in `ri_kernels`. The GPU kernels
ship as add-on packages, one per CUDA major version, selected with an extra:

```bash
pip install ri_kernels               # CPU kernels only
pip install "ri_kernels[cuda12]"     # + CUDA 12 kernels and jax[cuda12]
pip install "ri_kernels[cuda13]"     # + CUDA 13 kernels and jax[cuda13]
```

Quote the extra: `zsh` and `fish` both try to glob a bare `[...]`.

The add-on packages can also be installed by name, which brings in `ri_kernels`
itself but leaves your `jaxlib` alone — use this if you already have a CUDA JAX
installed some other way:

```bash
pip install ri_kernels_cuda12   # for jax[cuda12]
pip install ri_kernels_cuda13   # for jax[cuda13]
```

Nothing else changes at the call site — `ri_kernels.jax_api` finds the GPU
library wherever it was installed and registers the `CUDA` FFI targets. If the
library for a platform is not installed, only the lowerings for that platform
raise; the import still succeeds.

There are no ROCm wheels; build from source with `RI_KERNELS_ROCM=1`.

## Building from source

Requires CMake >= 3.20, a C++20 compiler, and network access at configure time
(Google Highway is fetched by `FetchContent`). `jax==0.6.0` is pulled into the
build environment automatically for its XLA FFI headers.

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
`RI_KERNELS_BUNDLED_HIGHWAY`, and `CMAKE_CUDA_ARCHITECTURES` (default
`80;90a-real;90`).

## Tests

```bash
pytest tests
```

The tests skip themselves when no kernel library is importable, so make sure
`ri_kernels` is installed (or built in place) first.

## Releasing

`.github/workflows/wheels.yml` builds all three distributions on every push and
publishes them to TestPyPI when a GitHub Release is published. The release tag
must match `[project].version` in `pyproject.toml`.

The CUDA wheels are generated from the same source tree by `ci/make_variant.py`,
which rewrites `pyproject.toml` into the add-on's metadata. Bumping the version
means editing `[project].version` and the `ri_kernels_cuda1X==` pins in the
`cuda12` / `cuda13` extras; `make_variant.py` refuses to build if they disagree,
so a missed pin fails CI rather than shipping.
