// GPU forward and JVP of the data-grid RFI visibility, staged.
//
// The fine samples of a chunk of time cells are materialised first, once per
// cell and antenna (rfi_interp_samples_gpu.cuh). The antennas are cut into
// tiles of kTile, and one block works one unordered pair of tiles (I, J) on
// one cell (f, t): for each source and each chunk of the cell's fine samples
// it loads the samples of the two tiles, contiguous runs, into shared memory,
// then forms the baseline products of the tile pair from there.
//
// The pairs a block forms are the ones the baseline list holds in its tile
// pair, listed by the caller (tile_pairs, see rfi_interp_vis_op.py). The two
// orderings of a pair are conjugates, so each is formed once and written to
// whichever orderings the list holds, through the pair table.
//
// Shared memory is bounded whatever the problem size: the chunk of samples
// is sized so the staged tiles stay within ~32 KB, which any device with the
// 48 KB default allows without opting in.

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "gpu_compat.h"
#include "rfi_interp_common.hpp"
#include "rfi_interp_samples_gpu.cuh"
#include "tensor.hpp"
#include "util_gpu.h"
#include "visibility.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ri_kernels {
namespace gpu {

// How much scratch one call may take for the chunk it works on. The kernels
// walk the time cells in chunks so that this bounds what they hold whatever
// the problem size; a smaller budget is more chunks and a smaller peak.
// RI_INTERP_SCRATCH_MB sets the default at build time and the environment
// variable overrides it at run time, so a run can be fitted to a card without
// rebuilding.
#ifndef RI_INTERP_SCRATCH_MB
#define RI_INTERP_SCRATCH_MB 256u
#endif

inline std::size_t interp_scratch_budget() {
  static const std::size_t budget = [] {
    std::size_t mb = RI_INTERP_SCRATCH_MB;
    if (const char *env = std::getenv("RI_KERNELS_INTERP_SCRATCH_MB")) {
      const long value = std::strtol(env, nullptr, 10);
      if (value > 0) mb = std::size_t(value);
    }
    return mb * 1024u * 1024u;
  }();

  return budget;
}

constexpr int kTile = 32;      // antennas per tile
constexpr int kBlock = 256;    // threads per block
constexpr int kPairs = kTile * kTile / kBlock;  // list entries per thread

template <typename T, typename INT_T> struct InterpViews {
  Tensor1D<const int *, INT_T> a1, a2;
  Tensor2D<const int *, INT_T> pair, tile_pairs;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, delay;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
};

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
InterpViews<T, INT_T> make_views(
    interp_index_t a1, interp_index_t a2, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs) {
  const auto a = amp.dimensions();
  return InterpViews<T, INT_T>{
      Tensor1D<const int *, INT_T>(a1.typed_data(), a1.dimensions()[0]),
      Tensor1D<const int *, INT_T>(a2.typed_data(), a2.dimensions()[0]),
      Tensor2D<const int *, INT_T>(pair.typed_data(), pair.dimensions()[0], pair.dimensions()[1]),
      Tensor2D<const int *, INT_T>(tile_pairs.typed_data(), tile_pairs.dimensions()[0], tile_pairs.dimensions()[1]),
      Tensor4D<const Cplx<T> *, INT_T>(
          reinterpret_cast<const Cplx<T> *>(amp.typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *, INT_T>(phase.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *, INT_T>(delay.typed_data(), delay.dimensions()[0],
                                 delay.dimensions()[1], delay.dimensions()[2],
                                 delay.dimensions()[3]),
      Tensor3D<const T *, INT_T>(w_freq.typed_data(), w_freq.dimensions()[0],
                                 w_freq.dimensions()[1], w_freq.dimensions()[2]),
      Tensor3D<const T *, INT_T>(w_time.typed_data(), w_time.dimensions()[0],
                                 w_time.dimensions()[1], w_time.dimensions()[2]),
      Tensor1D<const int *, INT_T>(start_freq.typed_data(), start_freq.dimensions()[0]),
      Tensor1D<const int *, INT_T>(start_time.typed_data(), start_time.dimensions()[0]),
      Tensor1D<const T *, INT_T>(dnu.typed_data(), dnu.dimensions()[0]),
      Tensor1D<const T *, INT_T>(dt.typed_data(), dt.dimensions()[0]),
      Tensor1D<const T *, INT_T>(freqs.typed_data(), freqs.dimensions()[0]),
  };
}

// The unordered tile pair (I, J), I <= J, of a linear index.
template <typename INT_T>
__device__ inline void tile_pair_of(INT_T tp, INT_T n_tiles, INT_T &I, INT_T &J) {
  I = 0;
  INT_T rem = tp;
  while (rem >= n_tiles - I) { rem -= n_tiles - I; ++I; }
  J = I + rem;
}

template <typename T, bool JVP, typename INT_T>
__global__ void __launch_bounds__(kBlock) rfi_interp_staged_kernel(
    InterpViews<T, INT_T> v, const Cplx<T> *S, const Cplx<T> *dS,
    Tensor3D<Cplx<T> *, INT_T> out, INT_T c_v, INT_T n_tiles, INT_T n_tile_pairs,
    INT_T t0, INT_T n_tc) {
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2];
  const INT_T n_int_f = v.w_freq.shape[2], n_int_t = v.w_time.shape[2];
  const INT_T n_s = n_int_f * n_int_t;
  const INT_T n_cells = n_freq * n_tc;
  const INT_T chunk = c_v * n_int_f;  // samples per chunk

  // Layout: the staged tiles, sample-major (tile[s * kTile + a]).
  Cplx<T> *tile_I = reinterpret_cast<Cplx<T> *>(dynamic_shared);
  Cplx<T> *tile_J = tile_I + chunk * kTile;
  Cplx<T> *dtile_I = tile_J + chunk * kTile;
  Cplx<T> *dtile_J = dtile_I + chunk * kTile;

  const int tid = threadIdx.x;

  for (INT_T tp = blockIdx.y; tp < n_tile_pairs; tp += gridDim.y) {
    INT_T I, J;
    tile_pair_of(tp, n_tiles, I, J);
    const bool same = I == J;

    // This thread's pairs from the list, kPairs consecutive entries: which
    // antennas. Entries are padded with -1, so the idle threads are the
    // trailing ones.
    INT_T pi[kPairs], pj[kPairs];
    bool has[kPairs], any = false;
    for (int k = 0; k < kPairs; ++k) {
      const INT_T e = v.tile_pairs(tp, tid * kPairs + k);
      has[k] = e >= 0;
      pi[k] = has[k] ? e / kTile : 0;
      pj[k] = has[k] ? e % kTile : 0;
      any = any || has[k];
    }
    const T scale = T(1) / T(n_s);
    // A tile pair with no pair at all (the list's first entry is -1) is skipped
    // by every thread alike: the first entry is thread 0's.
    if (v.tile_pairs(tp, 0) < 0) continue;

    for (INT_T cell = blockIdx.x; cell < n_cells; cell += gridDim.x) {
      const INT_T f = cell / n_tc, t_local = cell % n_tc, t = t0 + t_local;

      Cplx<T> acc[kPairs];
      for (int k = 0; k < kPairs; ++k) acc[k] = Cplx<T>{0, 0};

      for (INT_T r = 0; r < n_rfi; ++r) {
        for (INT_T s0 = 0; s0 < n_s; s0 += chunk) {
          const INT_T cs = n_s - s0 < chunk ? n_s - s0 : chunk;
          __syncthreads();  // the previous chunk's products are done
          // Stage: tile I, and tile J when it is a different tile, from the
          // materialised samples: a tile at one sample is one contiguous run.
          for (INT_T idx = tid; idx < cs * kTile; idx += kBlock) {
            const INT_T s_local = idx / kTile, a_local = idx % kTile;
            const INT_T s = s0 + s_local;
            const INT_T a = I * kTile + a_local;
            const std::int64_t o = sample_index(f, t_local, r, s, a, n_tc, n_rfi, n_s, n_ant);
            tile_I[idx] = a < n_ant ? S[o] : Cplx<T>{0, 0};
            if constexpr (JVP) dtile_I[idx] = a < n_ant ? dS[o] : Cplx<T>{0, 0};
            if (!same) {
              const INT_T a2 = J * kTile + a_local;
              const std::int64_t o2 = sample_index(f, t_local, r, s, a2, n_tc, n_rfi, n_s, n_ant);
              tile_J[idx] = a2 < n_ant ? S[o2] : Cplx<T>{0, 0};
              if constexpr (JVP) dtile_J[idx] = a2 < n_ant ? dS[o2] : Cplx<T>{0, 0};
            }
          }
          __syncthreads();
          if (!any) continue;
          const Cplx<T> *TJ = same ? tile_I : tile_J;
          const Cplx<T> *DJ = same ? dtile_I : dtile_J;
          // Multiply: this thread's pairs over the chunk's samples, the four
          // together so their loads overlap (an absent pair multiplies
          // antenna 0's samples and is never written).
          for (INT_T s_local = 0; s_local < cs; ++s_local) {
            for (int k = 0; k < kPairs; ++k) {
              const Cplx<T> a = tile_I[s_local * kTile + pi[k]];
              const Cplx<T> b = TJ[s_local * kTile + pj[k]];
              if constexpr (JVP) {
                const Cplx<T> da = dtile_I[s_local * kTile + pi[k]];
                const Cplx<T> db = DJ[s_local * kTile + pj[k]];
                acc[k] = cadd(acc[k], cadd(cmul(da, cconj(b)), cmul(a, cconj(db))));
              } else {
                acc[k] = cadd(acc[k], cmul(a, cconj(b)));
              }
            }
          }
        }
      }

      // Write each pair to the orderings the baseline list holds.
      for (int k = 0; k < kPairs; ++k) {
        if (!has[k]) continue;
        const INT_T a1 = I * kTile + pi[k], a2 = J * kTile + pj[k];
        const INT_T bl = v.pair(a1, a2);
        if (bl >= 0) out(bl, f, t) = cscale(scale, acc[k]);
        if (a1 != a2) {
          const INT_T bl2 = v.pair(a2, a1);
          if (bl2 >= 0) out(bl2, f, t) = cscale(scale, cconj(acc[k]));
        }
      }
    }
  }
}

// Whole time samples per chunk: as many as keep the tiles within the budget,
// at least one, at most the cell's count.
template <typename T>
std::int64_t staged_time_samples(std::int64_t n_int_f, std::int64_t n_int_t, int n_tiles_shared, std::size_t budget) {
  const std::size_t per_time_sample = sizeof(Cplx<T>) * kTile * n_tiles_shared * n_int_f;
  std::int64_t c_v = std::int64_t(budget / per_time_sample);
  if (c_v > 32) c_v = 32;
  if (c_v > n_int_t) c_v = n_int_t;
  return c_v < 1 ? 1 : c_v;
}

template <bool JVP, bool FULL, typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_gpu_dispatch(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1, interp_index_t a2,
    ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<AMP_DT, 4> amp_dot,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> phase_dot,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 4> delay_dot,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  auto views = make_views<T, INT_T>(a1, a2, pair, tile_pairs, amp, phase, delay,
                                    w_freq, start_freq, w_time, start_time, dnu, dt, freqs);
  const auto a = amp.dimensions();
  Tensor4D<const Cplx<T> *, INT_T> amp_dot_view(
      reinterpret_cast<const Cplx<T> *>(amp_dot.typed_data()), a[0], a[1], a[2], a[3]);
  Tensor3D<Cplx<T> *, INT_T> out_view(
      reinterpret_cast<Cplx<T> *>(out->typed_data()), out->dimensions()[0],
      out->dimensions()[1], out->dimensions()[2]);
  const INT_T n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const INT_T n_int_f = w_freq.dimensions()[2], n_int_t = w_time.dimensions()[2];
  const INT_T n_s = n_int_f * n_int_t;
  const INT_T n_tiles = (n_ant + kTile - 1) / kTile;
  const INT_T n_tile_pairs = n_tiles * (n_tiles + 1) / 2;
  const int n_tiles_shared = JVP ? 4 : 2;
  const INT_T c_v = staged_time_samples<T>(n_int_f, n_int_t, n_tiles_shared, 32 * 1024);
  const std::size_t shared = sizeof(Cplx<T>) * kTile * c_v * n_int_f * n_tiles_shared;

  // The samples of a chunk of time cells: as many cells as keep them within
  // 256 MB, at least one.
  // A time cell of the chunk spans every channel: n_freq cells of samples.
  const std::size_t per_cell = sizeof(Cplx<T>) * std::size_t(n_ant) * n_rfi * n_freq * n_s * (JVP ? 2 : 1);
  INT_T n_tc = INT_T(interp_scratch_budget() / per_cell);
  if (n_tc < 1) n_tc = 1;
  if (n_tc > n_time) n_tc = n_time;
  auto s_mem = scratch.Allocate(per_cell * n_tc, alignof(Cplx<T>));
  if (!s_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the fine samples");
  Cplx<T> *S = reinterpret_cast<Cplx<T> *>(*s_mem);
  Cplx<T> *dS = JVP ? S + std::size_t(n_ant) * n_rfi * n_freq * n_s * n_tc : nullptr;
  const auto dd = delay.dimensions();
  const SampleViews<T, INT_T> sample_views{
      views.amp, amp_dot_view, views.phase, views.delay,
      Tensor4D<const T *, INT_T>(phase_dot.typed_data(), a[0], a[1], a[2], a[3]),
      Tensor4D<const T *, INT_T>(delay_dot.typed_data(), dd[0], dd[1], dd[2], dd[3]),
      views.w_freq, views.w_time, views.start_freq, views.start_time, views.dnu, views.dt, views.freqs};
  constexpr int mode = JVP ? (FULL ? kSamplesAndFullTangent : kSamplesAndTangent) : kSamplesOnly;

  for (INT_T t0 = 0; t0 < n_time; t0 += n_tc) {
    const INT_T cells = n_time - t0 < n_tc ? n_time - t0 : n_tc;
    const auto sample_grid = create_clamped_grid(int(n_freq * n_rfi * n_ant), 1, 1);
    rfi_interp_samples_kernel<T, INT_T, mode>
        <<<sample_grid, kSampleBlock, 0, stream>>>(sample_views, S, dS, t0, cells);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    const auto grid = create_clamped_grid(n_freq * cells, n_tile_pairs, 1);
    rfi_interp_staged_kernel<T, JVP, INT_T><<<grid, kBlock, shared, stream>>>(
        views, S, dS, out_view, c_v, n_tiles, n_tile_pairs, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
  }
  return ffi::Error::Success();
}

// The allocator is move-only: the entry points take it by value from the
// binding and lend it down the call chain by reference.
template <bool JVP, bool FULL, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_gpu_impl_tmpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<AMP_DT, 4> amp_dot,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> phase_dot,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 4> delay_dot,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, path, table, or baseline shapes");
  if (pair.dimensions()[0] != amp.dimensions()[0] || pair.dimensions()[1] != amp.dimensions()[0])
    return ffi::Error::InvalidArgument("Expected an (n_ant, n_ant) pair table");
  const std::int64_t n_tiles = (amp.dimensions()[0] + kTile - 1) / kTile;
  if (tile_pairs.dimensions()[0] != n_tiles * (n_tiles + 1) / 2 ||
      tile_pairs.dimensions()[1] != kTile * kTile)
    return ffi::Error::InvalidArgument("Expected a (n_tile_pairs, 1024) tile-pair list");
  if (JVP && !interp_same_shape(amp_dot, amp))
    return ffi::Error::InvalidArgument("Expected the signal tangent to match the signal");
  if (FULL && !(interp_same_shape(phase_dot, phase) && interp_same_shape(delay_dot, delay)))
    return ffi::Error::InvalidArgument("Expected the phase and delay tangents to match the phase and delay");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (amp.element_count() < limit && out->element_count() < limit)
    return calc_rfi_interp_gpu_dispatch<JVP, FULL, T, std::int32_t>(
        stream, scratch, a1, a2, pair, tile_pairs, amp, amp_dot, phase, phase_dot, delay,
        delay_dot, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, out);
  return calc_rfi_interp_gpu_dispatch<JVP, FULL, T, std::int64_t>(
      stream, scratch, a1, a2, pair, tile_pairs, amp, amp_dot, phase, phase_dot, delay,
      delay_dot, w_freq, start_freq, w_time, start_time, dnu, dt, freqs, out);
}

#define RI_INTERP_GPU_ENTRY(NAME, JVP, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, DOT_PARAM, DOT_ARG) \
  ffi::Error NAME##_impl(                                                       \
      cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,     \
      interp_index_t a1_sorter,                                                  \
      interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,      \
      interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,                   \
      ffi::BufferR2<ffi::S32> tile_pairs, AMP_T amp DOT_PARAM,                    \
      R4 phase, R4 delay, R3 w_freq, interp_index_t start_freq, R3 w_time,        \
      interp_index_t start_time, R1 dnu, R1 dt, R1 freqs,                        \
      ffi::Result<ffi::BufferR3<AMP_DT>> out) {                                  \
    return calc_rfi_interp_gpu_impl_tmpl<JVP, false, AMP_DT, REAL_DT, T>(        \
        stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, \
        tile_pairs, amp, DOT_ARG, phase, phase, delay, delay, w_freq, start_freq, w_time, \
        start_time, dnu, dt, freqs, out);                                        \
  }

#define RI_COMMA_DOT_F32 , interp_amp_f32_t amp_dot
#define RI_COMMA_DOT_F64 , interp_amp_f64_t amp_dot
#define RI_NO_DOT

RI_INTERP_GPU_ENTRY(calc_rfi_interp_gpu_f32, false, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, RI_NO_DOT, amp)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_gpu_f64, false, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, RI_NO_DOT, amp)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_jvp_gpu_f32, true, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, RI_COMMA_DOT_F32, amp_dot)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_jvp_gpu_f64, true, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, RI_COMMA_DOT_F64, amp_dot)

// The full JVP: tangents on the signal, the phase and the delay.
#define RI_INTERP_FULL_JVP_GPU_ENTRY(NAME, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, OUT_T) \
  ffi::Error NAME##_impl(                                                          \
      cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,        \
      interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,         \
      interp_index_t a2_sorter, interp_index_t a2_start,                            \
      ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs, AMP_T amp,  \
      AMP_T amp_dot, R4 phase, R4 phase_dot, R4 delay, R4 delay_dot, R3 w_freq,      \
      interp_index_t start_freq, R3 w_time, interp_index_t start_time, R1 dnu,       \
      R1 dt, R1 freqs, ffi::Result<OUT_T> out) {                                     \
    return calc_rfi_interp_gpu_impl_tmpl<true, true, AMP_DT, REAL_DT, T>(            \
        stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair,     \
        tile_pairs, amp, amp_dot, phase, phase_dot, delay, delay_dot, w_freq,        \
        start_freq, w_time, start_time, dnu, dt, freqs, out);                        \
  }

RI_INTERP_FULL_JVP_GPU_ENTRY(calc_rfi_interp_full_jvp_gpu_f32, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, ffi::BufferR3<ffi::C64>)
RI_INTERP_FULL_JVP_GPU_ENTRY(calc_rfi_interp_full_jvp_gpu_f64, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, ffi::BufferR3<ffi::C128>)

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_full_jvp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_full_jvp_gpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_gpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_jvp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_jvp_gpu_f64(XLA_FFI_CallFrame *call_frame);

#define RI_INTERP_INDEX_ARGS                                                   \
  .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()           \
      .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()       \
      .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
#define RI_INTERP_TABLE_ARGS(P)                                                \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f32, calc_rfi_interp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f64, calc_rfi_interp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f32, calc_rfi_interp_jvp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f64, calc_rfi_interp_jvp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());

#define RI_INTERP_FULL_TABLE_ARGS(P)                                           \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                 \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_full_jvp_gpu_f32, calc_rfi_interp_full_jvp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_FULL_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_full_jvp_gpu_f64, calc_rfi_interp_full_jvp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_FULL_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace ri_kernels
