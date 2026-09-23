// GPU transpose (VJP with respect to the data-grid signal) of the data-grid
// RFI visibility, staged. See rfi_interp_transpose_kernel.cpp for the formulas.
//
// A block owns one tile of kInterpTile antennas, one cell and one source, and
// walks every partner tile to build its own antennas' cotangent samples:
//
//   W[i, j] = (vbar[pair(i, j)] + conj(vbar[pair(j, i)])) / n_samples,
//   G[i](s) = sum over every partner j of  W[i, j] conj(S[j](s)),
//   Q[i](s) = E_i(s) G[i](s),
//   H[i, kl] = sum_s w_freq[.., s] w_time[.., s] Q[i](s).
//
// The partner tile that owns j does the mirror of this with W(J, I), which is
// the conjugate transpose of W(I, J), so every baseline reaches both of its
// antennas exactly once and no block writes another's antennas.
//
// Owning the output rather than the tile pair is what keeps this from growing
// as the square of the antenna count. The phase factor and the stencil
// contraction are applied once per antenna, not once per tile pair it appears
// in, and the partials H carry an antenna axis rather than a tile-pair one:
// (n_ant, n_rfi, n_freq, n_time_chunk, n_slots) instead of
// (n_tile_pairs, 2, kInterpTile, ...), smaller by a factor of n_tiles + 1 --
// 162 MB to 18 MB at 256 antennas, 32 sources and 32 channels. The gather then
// has no tile-pair axis to sum over either.
//
// The fine samples and phase factors of a chunk of time cells are materialised
// first, once per cell and antenna (rfi_interp_samples_gpu.cuh), so staging is
// contiguous loads. If one full time cell exceeds the scratch budget, the
// samples and phase factors are materialised in balanced frequency chunks.
// H keeps every channel until all those chunks are complete: the amplitude
// gather can cross their stencil boundaries, and the delay gather still sums
// channels in its original order. Time batching is unchanged whenever a full
// time cell fits. Every element is owned by one thread: no atomics, and the
// result is deterministic.
//
// The full variant adds the phase's and the delay's cotangents: per fine
// sample phi_bar = Re(i S G) = -Im(S G), summed over the cell's samples for
// the phase and weighted by d phi / d delay[k] for the delay, both carried as
// extra slots of the partials and gathered the same way (the delay's summed
// over the channels as well).
//
// Shared memory: the cell's weight rows, one partner's cotangent weights, the
// own and partner sample tiles with the cotangent samples, and the cell's
// partials for this source. The sample chunk shrinks until that fits what the
// device will give a block.

#include <algorithm>
#include <cstdint>
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

// kInterpTile, the antennas per tile, comes from rfi_interp_common.hpp: the
// forward kernel and the analytic operator cut the antennas the same way.
constexpr int kBlockT = 256;   // threads per block

// Samples per chunk: the four tiles within 16 KB.
template <typename T> struct ChunkT { static constexpr int value = sizeof(T) == 4 ? 16 : 4; };

template <typename T, typename INT_T> struct TransposeViews {
  Tensor2D<const int *, INT_T> pair, tile_pairs;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, delay;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
  Tensor3D<const Cplx<T> *, INT_T> vis_bar;
  // The chunk's materialised samples and phase factors (rfi_interp_samples_gpu.cuh).
  const Cplx<T> *S, *E;
  // The per-cell cotangents for the time cells of the current chunk, laid out
  // (n_ant, n_rfi, n_freq, n_time_chunk, n_slots): the n_sf * n_st stencil
  // cotangents and, in the full variant, the phase's and the n_path delay
  // cotangents after them.
  Cplx<T> *H;
  Tensor4D<Cplx<T> *, INT_T> amp_bar;
  // The full variant's extra outputs; unset otherwise.
  Tensor4D<T *, INT_T> phase_bar, delay_bar;
};

template <typename INT_T>
__device__ inline std::int64_t h_index(INT_T a, INT_T r, INT_T f, INT_T t_local, INT_T slot,
                                       INT_T n_rfi, INT_T n_freq, INT_T n_tc, INT_T n_slots) {
  return ((((std::int64_t(a) * n_rfi + r) * n_freq + f) * n_tc + t_local) * n_slots) + slot;
}

// SPLIT as in the forward: the unsplit instantiation, which is what a call
// that fits its budget launches, keeps a single channel index in the cell loop.
template <typename T, typename INT_T, bool FULL, bool SPLIT>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_own(
    TransposeViews<T, INT_T> v, INT_T n_tiles, InterpCellChunk<INT_T> cells, INT_T kChunk) {
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2];
  const INT_T t0 = cells.t0, n_tc = cells.n_tc;
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_stencil = n_sf * n_st, n_s = n_int_f * n_int_t;
  const INT_T n_path = v.delay.shape[3];
  const INT_T n_slots = FULL ? n_stencil + 1 + n_path : n_stencil;
  const T inv = T(1) / T(n_s);

  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  Cplx<T> *W = reinterpret_cast<Cplx<T> *>(dynamic_shared + head);  // [j][i], one partner
  Cplx<T> *S_own = W + kInterpTile * kInterpTile;                             // [s][a]
  Cplx<T> *S_par = S_own + kChunk * kInterpTile;
  Cplx<T> *G = S_par + kChunk * kInterpTile;                             // [s][i]
  Cplx<T> *Hacc = G + kChunk * kInterpTile;                              // [slot][i]
  T *P = reinterpret_cast<T *>(Hacc + kInterpTile * n_slots);            // [s][i], full variant

  const int tid = threadIdx.x;
  const INT_T I = blockIdx.y;             // the tile of antennas this block owns
  const INT_T r = blockIdx.z;             // and the source

  for (INT_T cell = blockIdx.x; cell < (SPLIT ? cells.n_fc : n_freq) * n_tc; cell += gridDim.x) {
    const INT_T f_local = cell / n_tc, t_local = cell % n_tc, t = t0 + t_local;
    const INT_T f = SPLIT ? cells.f0 + f_local : f_local;
    const T freq_f = v.freqs(f);
    __syncthreads();  // the previous cell is done with the shared arrays
    for (INT_T i = tid; i < n_sf * n_int_f; i += kBlockT)
      wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
    for (INT_T i = tid; i < n_st * n_int_t; i += kBlockT)
      wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
    for (INT_T idx = tid; idx < kInterpTile * n_slots; idx += kBlockT) Hacc[idx] = Cplx<T>{0, 0};

    for (INT_T s0 = 0; s0 < n_s; s0 += kChunk) {
      const INT_T cs = n_s - s0 < kChunk ? n_s - s0 : kChunk;
      __syncthreads();  // the weights and the previous chunk's contraction
      // The own tile's samples, and a clean G to accumulate the partners into.
      for (INT_T idx = tid; idx < cs * kInterpTile; idx += kBlockT) {
        const INT_T s_local = idx / kInterpTile, i_local = idx % kInterpTile;
        const INT_T a = I * kInterpTile + i_local;
        S_own[idx] = a < n_ant
                         ? v.S[sample_index(f_local, t_local, r, s0 + s_local, a, n_tc, n_rfi, n_s, n_ant)]
                         : Cplx<T>{0, 0};
        G[idx] = Cplx<T>{0, 0};
      }

      for (INT_T J = 0; J < n_tiles; ++J) {
        __syncthreads();  // the previous partner's W and samples are done with
        // This partner's cotangent weights. Cheap beside the product below --
        // one entry against kChunk sample products -- so rebuilding them per
        // source and per chunk costs less than holding every partner's.
        for (INT_T idx = tid; idx < kInterpTile * kInterpTile; idx += kBlockT) {
          const INT_T i_local = idx / kInterpTile, j_local = idx % kInterpTile;
          const INT_T a = I * kInterpTile + i_local, b = J * kInterpTile + j_local;
          Cplx<T> w{0, 0};
          if (a < n_ant && b < n_ant) {
            const INT_T bl = v.pair(a, b);
            if (bl >= 0) w = cadd(w, cscale(inv, v.vis_bar(bl, f, t)));
            const INT_T bl2 = v.pair(b, a);
            if (bl2 >= 0) w = cadd(w, cscale(inv, cconj(v.vis_bar(bl2, f, t))));
          }
          // Stored transposed, [j][i]: the product below has a thread per own
          // antenna i and walks j, so this is what makes a warp's reads
          // consecutive words rather than 32 apart in the same bank.
          W[j_local * kInterpTile + i_local] = w;
        }
        if (J != I) {
          for (INT_T idx = tid; idx < cs * kInterpTile; idx += kBlockT) {
            const INT_T s_local = idx / kInterpTile, j_local = idx % kInterpTile;
            const INT_T b = J * kInterpTile + j_local;
            S_par[idx] = b < n_ant
                             ? v.S[sample_index(f_local, t_local, r, s0 + s_local, b, n_tc, n_rfi, n_s, n_ant)]
                             : Cplx<T>{0, 0};
          }
        }
        __syncthreads();
        const Cplx<T> *SJ = J == I ? S_own : S_par;
        // G[i](s) += sum_j W[i, j] conj(S_J[j](s)).
        for (INT_T idx = tid; idx < cs * kInterpTile; idx += kBlockT) {
          const INT_T s_local = idx / kInterpTile, i_local = idx % kInterpTile;
          Cplx<T> acc{0, 0};
          for (INT_T j = 0; j < kInterpTile; ++j)
            acc = cadd(acc, cmul(W[j * kInterpTile + i_local],
                                 cconj(SJ[s_local * kInterpTile + j])));
          G[idx] = cadd(G[idx], acc);
        }
      }

      __syncthreads();
      // G is complete for this chunk: turn it by the antenna's phase factor and
      // push it through the stencil weights, once, onto the cell's partials.
      for (INT_T idx = tid; idx < cs * kInterpTile; idx += kBlockT) {
        const INT_T s_local = idx / kInterpTile, i_local = idx % kInterpTile;
        const INT_T a = I * kInterpTile + i_local;
        if (a >= n_ant) {
          G[idx] = Cplx<T>{0, 0};
          if constexpr (FULL) P[idx] = T(0);
          continue;
        }
        const Cplx<T> e = v.E[sample_index(f_local, t_local, r, s0 + s_local, a, n_tc, n_rfi, n_s, n_ant)];
        if constexpr (FULL) P[idx] = -cmul(S_own[idx], G[idx]).im;
        G[idx] = cmul(e, G[idx]);
      }
      __syncthreads();
      for (INT_T idx = tid; idx < kInterpTile * n_slots; idx += kBlockT) {
        const INT_T i_local = idx % kInterpTile, slot = idx / kInterpTile;
        Cplx<T> h{0, 0};
        if (slot < n_stencil) {
          const INT_T kf = slot / n_st, kt = slot % n_st;
          for (INT_T s_local = 0; s_local < cs; ++s_local) {
            const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
            h = cadd(h, cscale(wf[kf * n_int_f + u] * wt[kt * n_int_t + vv],
                               G[s_local * kInterpTile + i_local]));
          }
        } else if constexpr (FULL) {
          const INT_T k = slot - n_stencil - 1;  // -1 is the phase, then the delay orders
          for (INT_T s_local = 0; s_local < cs; ++s_local) {
            const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
            const T pc = P[s_local * kInterpTile + i_local];
            h.re += k < 0 ? pc : delay_phase_coeff(k, freq_f, v.dnu(u), v.dt(vv)) * pc;
          }
        }
        Hacc[idx] = cadd(Hacc[idx], h);
      }
    }

    // The cell's partials for this antenna tile and source, written once.
    __syncthreads();
    for (INT_T idx = tid; idx < kInterpTile * n_slots; idx += kBlockT) {
      const INT_T i_local = idx % kInterpTile, slot = idx / kInterpTile;
      const INT_T a = I * kInterpTile + i_local;
      if (a < n_ant)
        v.H[h_index(a, r, f, t_local, slot, n_rfi, n_freq, n_tc, n_slots)] = Hacc[idx];
    }
  }
}

// Sum, per data-grid element, the cells of the chunk whose stencils cover it.
// Every output element is written by one thread per chunk, in a fixed order.
template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_gather(
    TransposeViews<T, INT_T> v, INT_T t0, INT_T n_tc, INT_T n_slots) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T tp_lo = t0 - n_st + 1 > 0 ? t0 - n_st + 1 : 0;
  const INT_T tp_hi = t0 + n_tc + n_st - 1 < n_time ? t0 + n_tc + n_st - 1 : n_time;
  const INT_T n_tp = tp_hi - tp_lo;
  const INT_T n_out = n_ant * n_rfi * n_freq * n_tp;
  for (INT_T i = blockIdx.x * INT_T(kBlockT) + threadIdx.x; i < n_out;
       i += INT_T(gridDim.x) * kBlockT) {
    const INT_T tpt = tp_lo + i % n_tp;
    const INT_T q = i / n_tp;
    const INT_T fp = q % n_freq;
    const INT_T q2 = q / n_freq;
    const INT_T r = q2 % n_rfi;
    const INT_T ant = q2 / n_rfi;
    Cplx<T> sum{0, 0};
    const INT_T f_lo = fp - n_sf + 1 > 0 ? fp - n_sf + 1 : 0;
    const INT_T f_hi = fp + n_sf < n_freq ? fp + n_sf : n_freq;
    const INT_T t_lo = tpt - n_st + 1 > t0 ? tpt - n_st + 1 : t0;
    const INT_T t_hi = tpt + n_st < t0 + n_tc ? tpt + n_st : t0 + n_tc;
    for (INT_T f = f_lo; f < f_hi; ++f) {
      const INT_T k = fp - v.start_freq(f);
      if (k < 0 || k >= n_sf) continue;
      for (INT_T t = t_lo; t < t_hi; ++t) {
        const INT_T l = tpt - v.start_time(t);
        if (l < 0 || l >= n_st) continue;
        sum = cadd(sum, v.H[h_index(ant, r, f, t - t0, k * n_st + l, n_rfi, n_freq, n_tc, n_slots)]);
      }
    }
    v.amp_bar(ant, r, fp, tpt) = cadd(v.amp_bar(ant, r, fp, tpt), sum);
  }
}

// The full variant's phase and delay cotangents: the phase's is the cell's own
// slot, the delay's are summed over the channels.
template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_gather_phase(
    TransposeViews<T, INT_T> v, INT_T t0, INT_T n_tc, INT_T n_slots) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T n_stencil = n_sf * n_st, n_path = v.delay.shape[3];
  const INT_T n_out = n_ant * n_rfi * n_tc;
  for (INT_T i = blockIdx.x * INT_T(kBlockT) + threadIdx.x; i < n_out;
       i += INT_T(gridDim.x) * kBlockT) {
    const INT_T t_local = i % n_tc, q = i / n_tc;
    const INT_T r = q % n_rfi, ant = q / n_rfi;
    for (INT_T k = 0; k < n_path; ++k) v.delay_bar(ant, r, t0 + t_local, k) = 0;
    for (INT_T f = 0; f < n_freq; ++f) {
      v.phase_bar(ant, r, f, t0 + t_local) =
          v.H[h_index(ant, r, f, t_local, n_stencil, n_rfi, n_freq, n_tc, n_slots)].re;
      for (INT_T k = 0; k < n_path; ++k)
        v.delay_bar(ant, r, t0 + t_local, k) +=
            v.H[h_index(ant, r, f, t_local, n_stencil + 1 + k, n_rfi, n_freq, n_tc, n_slots)].re;
    }
  }
}

template <bool FULL, typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::ScratchAllocator &scratch,
    ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> *phase_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> *delay_bar) {
  const auto a = amp.dimensions();
  const auto dd = delay.dimensions();
  const std::int64_t n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const std::int64_t n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const std::int64_t n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  std::int64_t n_stencil, n_s, n_slots, extra_slots = 0;
  if (!interp_checked_product({n_sf, n_st}, n_stencil) ||
      !interp_checked_product({n_int_f, n_int_t}, n_s) ||
      (FULL && !interp_checked_sum(1, dd[3], extra_slots)) ||
      !interp_checked_sum(n_stencil, extra_slots, n_slots))
    return ffi::Error::InvalidArgument("Interpolation stencil or sample count exceeds the 64-bit index range");
  const auto n_tiles = interp_tile_count(n_ant);

  // Shared memory, in the order the kernel lays it out: the cell's weight rows,
  // one partner's cotangent weights and this source's partials, which are there
  // whatever the chunk; then the own and partner sample tiles and the cotangent
  // samples, which scale with it. The chunk halves until the whole fits what the
  // device will give a block -- 48 KB everywhere, more on a kernel that opts in.
  // The weight rows grow with the stencil width and the samples per cell, so a
  // long cell in double precision is what runs this out.
  std::int64_t wf_count, wt_count, weight_count, weight_bytes, padded_weights, partial_bytes, fixed;
  if (!interp_checked_product({n_sf, n_int_f}, wf_count) ||
      !interp_checked_product({n_st, n_int_t}, wt_count) ||
      !interp_checked_sum(wf_count, wt_count, weight_count) ||
      !interp_checked_product({sizeof(T), weight_count}, weight_bytes) ||
      !interp_checked_sum(weight_bytes, 15, padded_weights) ||
      !interp_checked_product({sizeof(Cplx<T>), kInterpTile, n_slots}, partial_bytes))
    return ffi::Error::InvalidArgument("Interpolation shared memory exceeds the 64-bit byte range");
  const std::int64_t head = padded_weights / 16 * 16;
  if (!interp_checked_sum(head, sizeof(Cplx<T>) * kInterpTile * kInterpTile, fixed) ||
      !interp_checked_sum(fixed, partial_bytes, fixed))
    return ffi::Error::InvalidArgument("Interpolation shared memory exceeds the 64-bit byte range");
  const std::int64_t per_sample =
      sizeof(Cplx<T>) * 3 * std::size_t(kInterpTile) +
      (FULL ? sizeof(T) * std::size_t(kInterpTile) : 0);
  const std::int64_t shared_limit = max_dynamic_shared_bytes();
  std::int64_t kChunk = ChunkT<T>::value;
  while (kChunk > 1 && (fixed > shared_limit || per_sample * kChunk > shared_limit - fixed)) kChunk /= 2;
  if (kChunk > std::int64_t(n_s)) kChunk = std::max<std::int64_t>(1, std::int64_t(n_s));
  std::int64_t shared;
  if (!interp_checked_sum(fixed, per_sample * kChunk, shared))
    return ffi::Error::InvalidArgument("Interpolation shared memory exceeds the 64-bit byte range");
  if (shared > shared_limit)
    return ffi::Error::InvalidArgument(
        "The transpose needs " + std::to_string(shared) + " bytes of shared memory and this device "
        "offers " + std::to_string(shared_limit) + ". The stencil rows of one cell are " +
        std::to_string(head) + " of that and grow with the stencil width and the samples per cell, "
        "so a shorter cell, a narrower stencil or single precision will fit.");
  if (shared > 48 * 1024) {
    for (const auto fn : {reinterpret_cast<const void *>(rfi_interp_transpose_own<T, INT_T, FULL, false>),
                          reinterpret_cast<const void *>(rfi_interp_transpose_own<T, INT_T, FULL, true>)}) {
      const auto attr =
          cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared));
      if (attr != cudaSuccess)
        return ffi::Error::Internal(std::string("Could not raise the shared memory limit to ") +
                                    std::to_string(shared) + " bytes: " + cudaGetErrorString(attr));
    }
  }

  // H spans every channel of the time chunk. Only S and E shrink with the
  // frequency width, and their capacity offsets stay fixed for shorter tails.
  std::int64_t per_cell_h, sample_per_freq, amp_bar_bytes;
  if (!interp_checked_product({sizeof(Cplx<T>), n_ant, n_rfi, n_freq, n_slots}, per_cell_h) ||
      !interp_checked_product({sizeof(Cplx<T>), n_ant, n_rfi, n_s}, sample_per_freq) ||
      !interp_checked_product({sizeof(Cplx<T>), n_ant, n_rfi, n_freq, n_time}, amp_bar_bytes))
    return ffi::Error::InvalidArgument("Interpolation scratch or output size exceeds the 64-bit byte range");
  InterpChunkPlan plan;
  auto plan_status = make_interp_chunk_plan(n_time, n_freq, sample_per_freq, per_cell_h, 2, plan);
  if (!plan_status.success()) return plan_status;
  std::int64_t sample_work;
  if (!interp_checked_product({n_s, plan.n_tc}, sample_work) ||
      sample_work > std::numeric_limits<INT_T>::max() ||
      n_slots > std::numeric_limits<INT_T>::max() / kInterpTile ||
      weight_count > std::numeric_limits<INT_T>::max())
    return ffi::Error::InvalidArgument("Interpolation sample or stencil count exceeds the kernel index range");
  auto h_mem = scratch.Allocate(std::size_t(plan.total_bytes), alignof(Cplx<T>));
  if (!h_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the per-cell cotangents and samples");
  Cplx<T> *H = reinterpret_cast<Cplx<T> *>(*h_mem);
  Cplx<T> *S = H + plan.h_bytes / sizeof(Cplx<T>);
  Cplx<T> *E = S + plan.sample_bytes / sizeof(Cplx<T>);

  TransposeViews<T, INT_T> views{
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
      Tensor3D<const Cplx<T> *, INT_T>(
          reinterpret_cast<const Cplx<T> *>(vis_bar.typed_data()),
          vis_bar.dimensions()[0], vis_bar.dimensions()[1], vis_bar.dimensions()[2]),
      S, E, H,
      Tensor4D<Cplx<T> *, INT_T>(
          reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), a[0], a[1], a[2], a[3]),
      Tensor4D<T *, INT_T>(FULL ? (*phase_bar)->typed_data() : nullptr, a[0], a[1], a[2], a[3]),
      Tensor4D<T *, INT_T>(FULL ? (*delay_bar)->typed_data() : nullptr, dd[0], dd[1], dd[2], dd[3]),
  };
  const SampleViews<T, INT_T> sample_views{views.amp, views.amp, views.phase, views.delay,
                                           views.phase, views.delay,
                                           views.w_freq, views.w_time, views.start_freq,
                                           views.start_time, views.dnu, views.dt, views.freqs};

  const bool split = plan.n_fc != n_freq;
  auto status = cudaMemsetAsync(amp_bar->typed_data(), 0, std::size_t(amp_bar_bytes), stream);
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU memset error: ") + cudaGetErrorString(status));

  for (std::int64_t t0 = 0; t0 < n_time;) {
    const auto nt = std::min<std::int64_t>(n_time - t0, plan.n_tc);
    for (std::int64_t f0 = 0; f0 < n_freq;) {
      const auto nf = std::min<std::int64_t>(n_freq - f0, plan.n_fc);
      std::int64_t sample_blocks, cell_blocks;
      if (!interp_checked_product({nf, n_rfi, n_ant}, sample_blocks) ||
          !interp_checked_product({nf, nt}, cell_blocks))
        return ffi::Error::InvalidArgument("Interpolation launch size exceeds the 64-bit index range");
      const InterpCellChunk<INT_T> cells{INT_T(t0), INT_T(nt), INT_T(f0), INT_T(nf)};
      const auto sample_grid = create_clamped_grid(interp_grid_extent(sample_blocks), 1, 1);
      if (split)
        rfi_interp_samples_chunk_kernel<T, INT_T, kSamplesAndPhase>
            <<<sample_grid, kSampleBlock, 0, stream>>>(sample_views, S, E, cells);
      else
        rfi_interp_samples_kernel<T, INT_T, kSamplesAndPhase>
            <<<sample_grid, kSampleBlock, 0, stream>>>(sample_views, S, E, INT_T(t0), INT_T(nt));
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
      // A block per (cell, own tile, source), just as in the unsplit launch.
      const auto grid = create_clamped_grid(interp_grid_extent(cell_blocks),
                                            interp_grid_extent(n_tiles), interp_grid_extent(n_rfi));
      if (grid.y != n_tiles || grid.z != n_rfi)
        return ffi::Error::InvalidArgument("Interpolation antenna tiles or sources exceed the device grid limits");
      if (split)
        rfi_interp_transpose_own<T, INT_T, FULL, true><<<grid, kBlockT, shared, stream>>>(
            views, INT_T(n_tiles), cells, INT_T(kChunk));
      else
        rfi_interp_transpose_own<T, INT_T, FULL, false><<<grid, kBlockT, shared, stream>>>(
            views, INT_T(n_tiles), cells, INT_T(kChunk));
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
      f0 += nf;
    }
    // All frequencies of H are now complete; gather once in the original order.
    std::int64_t halo, reach, n_out, n_cells_out;
    if (!interp_checked_product({2, n_st - 1}, halo) ||
        !interp_checked_sum(nt, halo, reach) ||
        !interp_checked_product({n_ant, n_rfi, n_freq, reach}, n_out) ||
        !interp_checked_product({n_ant, n_rfi, nt}, n_cells_out))
      return ffi::Error::InvalidArgument("Interpolation gather size exceeds the 64-bit index range");
    const auto gather_grid = create_clamped_grid(interp_grid_extent(interp_ceil_div(n_out, kBlockT)), 1, 1);
    rfi_interp_transpose_gather<T, INT_T><<<gather_grid, kBlockT, 0, stream>>>(
        views, INT_T(t0), INT_T(nt), INT_T(n_slots));
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    if constexpr (FULL) {
      const auto phase_grid = create_clamped_grid(interp_grid_extent(interp_ceil_div(n_cells_out, kBlockT)), 1, 1);
      rfi_interp_transpose_gather_phase<T, INT_T><<<phase_grid, kBlockT, 0, stream>>>(
          views, INT_T(t0), INT_T(nt), INT_T(n_slots));
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    }
    t0 += nt;
  }
  return ffi::Error::Success();
}

// The allocator is move-only: the entry points take it by value from the
// binding and lend it down the call chain by reference.
template <bool FULL, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_transpose_gpu_impl_tmpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::BufferR3<AMP_DT> vis_bar, ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar,
    ffi::Result<ffi::Buffer<REAL_DT, 4>> *phase_bar, ffi::Result<ffi::Buffer<REAL_DT, 4>> *delay_bar) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, delay, table, or baseline shapes");
  if (FULL && !(interp_same_shape(**phase_bar, phase) && interp_same_shape(**delay_bar, delay)))
    return ffi::Error::InvalidArgument("Expected the phase and delay cotangents to match the phase and delay");
  const std::int64_t n_ant = amp.dimensions()[0];
  if (pair.dimensions()[0] != n_ant || pair.dimensions()[1] != n_ant)
    return ffi::Error::InvalidArgument("Expected an (n_ant, n_ant) pair table");
  const std::int64_t n_tiles = interp_tile_count(n_ant);
  if (tile_pairs.dimensions()[0] != n_tiles * (n_tiles + 1) / 2 ||
      tile_pairs.dimensions()[1] != kInterpTilePairs)
    return ffi::Error::InvalidArgument(
        "Expected a (n_tile_pairs, " + std::to_string(kInterpTilePairs) + ") tile-pair list");
  if (vis_bar.dimensions()[0] != a1.dimensions()[0] ||
      vis_bar.dimensions()[1] != amp.dimensions()[2] ||
      vis_bar.dimensions()[2] != amp.dimensions()[3])
    return ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents");
  if (!interp_same_shape(*amp_bar, amp))
    return ffi::Error::InvalidArgument("Expected the signal cotangent to match the signal");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible; the scratch is indexed in 64 bits anyway
  if (amp.element_count() < limit && vis_bar.element_count() < limit)
    return calc_rfi_interp_transpose_gpu_dispatch<FULL, T, std::int32_t>(
        stream, scratch, pair, tile_pairs, amp, phase, delay, w_freq, start_freq, w_time,
        start_time, dnu, dt, freqs, vis_bar, amp_bar, phase_bar, delay_bar);
  return calc_rfi_interp_transpose_gpu_dispatch<FULL, T, std::int64_t>(
      stream, scratch, pair, tile_pairs, amp, phase, delay, w_freq, start_freq, w_time,
      start_time, dnu, dt, freqs, vis_bar, amp_bar, phase_bar, delay_bar);
}

#define RI_INTERP_TRANSPOSE_GPU_ENTRY(NAME, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, VBAR_T) \
  ffi::Error NAME##_impl(                                                          \
      cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,       \
      interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,        \
      interp_index_t a2_sorter, interp_index_t a2_start,                           \
      ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs,            \
      AMP_T amp, R4 phase, R4 delay, R3 w_freq, interp_index_t start_freq,          \
      R3 w_time, interp_index_t start_time, R1 dnu, R1 dt, R1 freqs, VBAR_T vis_bar, \
      ffi::Result<AMP_T> amp_bar) {                                                 \
    return calc_rfi_interp_transpose_gpu_impl_tmpl<false, AMP_DT, REAL_DT, T>(      \
        stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair,    \
        tile_pairs, amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, \
        dt, freqs, vis_bar, amp_bar, nullptr, nullptr);                             \
  }

#define RI_INTERP_FULL_TRANSPOSE_GPU_ENTRY(NAME, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, VBAR_T) \
  ffi::Error NAME##_impl(                                                          \
      cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,       \
      interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,        \
      interp_index_t a2_sorter, interp_index_t a2_start,                           \
      ffi::BufferR2<ffi::S32> pair, ffi::BufferR2<ffi::S32> tile_pairs,            \
      AMP_T amp, R4 phase, R4 delay, R3 w_freq, interp_index_t start_freq,          \
      R3 w_time, interp_index_t start_time, R1 dnu, R1 dt, R1 freqs, VBAR_T vis_bar, \
      ffi::Result<AMP_T> amp_bar, ffi::Result<R4> phase_bar, ffi::Result<R4> delay_bar) { \
    return calc_rfi_interp_transpose_gpu_impl_tmpl<true, AMP_DT, REAL_DT, T>(       \
        stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair,    \
        tile_pairs, amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, \
        dt, freqs, vis_bar, amp_bar, &phase_bar, &delay_bar);                       \
  }

RI_INTERP_TRANSPOSE_GPU_ENTRY(calc_rfi_interp_transpose_gpu_f32, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, ffi::BufferR3<ffi::C64>)
RI_INTERP_TRANSPOSE_GPU_ENTRY(calc_rfi_interp_transpose_gpu_f64, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, ffi::BufferR3<ffi::C128>)
RI_INTERP_FULL_TRANSPOSE_GPU_ENTRY(calc_rfi_interp_full_transpose_gpu_f32, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, ffi::BufferR3<ffi::C64>)
RI_INTERP_FULL_TRANSPOSE_GPU_ENTRY(calc_rfi_interp_full_transpose_gpu_f64, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, ffi::BufferR3<ffi::C128>)

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_full_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);

#define RI_INTERP_T_INDEX_ARGS                                                 \
  .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()           \
      .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()       \
      .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_gpu_f32, calc_rfi_interp_transpose_gpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        RI_INTERP_T_INDEX_ARGS
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>().Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<interp_amp_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_gpu_f64, calc_rfi_interp_transpose_gpu_f64_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        RI_INTERP_T_INDEX_ARGS
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_full_transpose_gpu_f32, calc_rfi_interp_full_transpose_gpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        RI_INTERP_T_INDEX_ARGS
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>().Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>().Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<interp_amp_f32_t>().Ret<interp_real4_f32_t>().Ret<interp_real4_f32_t>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_full_transpose_gpu_f64, calc_rfi_interp_full_transpose_gpu_f64_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        RI_INTERP_T_INDEX_ARGS
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>().Ret<interp_real4_f64_t>().Ret<interp_real4_f64_t>());

} // namespace gpu
} // namespace ri_kernels
