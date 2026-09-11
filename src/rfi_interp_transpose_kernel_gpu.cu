// GPU transpose (VJP with respect to the data-grid signal) of the data-grid
// RFI visibility, staged. See rfi_interp_transpose_kernel.cpp for the
// formulas. Here they are arranged as the forward's mirror: for a pair of
// antenna tiles (I, J) on one cell, with W the (kTile x kTile) matrix of the
// tile pair's cotangent weights,
//
//   W[i, j] = (vbar[pair(i, j)] + conj(vbar[pair(j, i)])) / n_samples,
//   Q_I[i](s) = E_i(s) sum_j W[i, j] conj(S_J[j](s)),
//   Q_J[j](s) = E_j(s) sum_i conj(W[i, j]) conj(S_I[i](s)),
//
// two small dense products per source and chunk of samples, from the two
// tiles' samples staged in shared memory and the phase factors E, followed
// by the push of Q through the stencil weights onto a per-tile-pair partial
// of the stencil cotangents H. The fine samples and phase factors of a chunk
// of time cells are materialised first, once per cell and antenna
// (rfi_interp_samples_gpu.cuh), so staging is contiguous loads. Every
// element is owned by one thread: no atomics, and the result is
// deterministic. The partials are chunked over time cells, and a gather
// kernel sums each antenna's tile pairs and the cells whose stencils cover
// a data-grid element, in a fixed order.
//
// The full variant adds the phase's and the delay's cotangents: per fine
// sample phi_bar = Re(i S G) = -Im(S G) with G the cotangent factor before
// the phase (see rfi_interp_transpose_kernel.cpp), summed over the cell's
// samples for the phase and weighted by d phi / d delay[k] for the delay,
// both carried as extra slots of the per-tile-pair partials and gathered
// the same way (the delay's summed over the channels as well).
//
// Shared memory: the cell's weight rows, W in both layouts (16 KB single),
// the two staged tiles and the two Q tiles (the chunk is sized for 16 KB):
// within the 48 KB every device offers, whatever the stencil.

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

constexpr int kTileT = 32;     // antennas per tile
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
  // Per-tile-pair partials of the per-cell cotangents for the time cells of
  // the current chunk, laid out (n_tile_pairs, 2 sides, kTileT, n_rfi,
  // n_freq, n_time_chunk, n_slots): the n_sf * n_st stencil cotangents and,
  // in the full variant, the phase's and the n_path delay cotangents after.
  Cplx<T> *H;
  Tensor4D<Cplx<T> *, INT_T> amp_bar;
  // The full variant's extra outputs; unset otherwise.
  Tensor4D<T *, INT_T> phase_bar, delay_bar;
};

template <typename INT_T>
__device__ inline std::int64_t h_index(INT_T tp, INT_T side, INT_T a_local, INT_T r, INT_T f,
                                       INT_T t_local, INT_T kl, INT_T n_rfi, INT_T n_freq,
                                       INT_T n_tc, INT_T n_stencil) {
  return (((((std::int64_t(tp) * 2 + side) * kTileT + a_local) * n_rfi + r) * n_freq + f) * n_tc + t_local) * n_stencil + kl;
}

// The unordered tile pair (I, J), I <= J, of a linear index, and back.
template <typename INT_T>
__device__ inline void tile_pair_of_t(INT_T tp, INT_T n_tiles, INT_T &I, INT_T &J) {
  I = 0;
  INT_T rem = tp;
  while (rem >= n_tiles - I) { rem -= n_tiles - I; ++I; }
  J = I + rem;
}
template <typename INT_T>
__device__ inline INT_T tile_pair_index(INT_T I, INT_T J, INT_T n_tiles) {
  return I * n_tiles - I * (I - 1) / 2 + (J - I);
}

template <typename T, typename INT_T, bool FULL>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_pairs(
    TransposeViews<T, INT_T> v, INT_T n_tiles, INT_T n_tile_pairs, INT_T t0, INT_T n_tc) {
  constexpr int kChunk = ChunkT<T>::value;
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_stencil = n_sf * n_st, n_s = n_int_f * n_int_t;
  const INT_T n_path = v.delay.shape[3];
  const INT_T n_slots = FULL ? n_stencil + 1 + n_path : n_stencil;
  const T inv = T(1) / T(n_s);

  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  Cplx<T> *Wp = reinterpret_cast<Cplx<T> *>(dynamic_shared + head);  // [i][j]
  Cplx<T> *Wt = Wp + kTileT * kTileT;                                 // [j][i]
  Cplx<T> *S_I = Wt + kTileT * kTileT;                                // [s][a]
  Cplx<T> *S_J = S_I + kChunk * kTileT;
  Cplx<T> *Q_I = S_J + kChunk * kTileT;
  Cplx<T> *Q_J = Q_I + kChunk * kTileT;
  T *P_I = reinterpret_cast<T *>(Q_J + kChunk * kTileT);  // [s][a]: -Im(S G), full variant
  T *P_J = P_I + kChunk * kTileT;

  const int tid = threadIdx.x;

  for (INT_T tp = blockIdx.y; tp < n_tile_pairs; tp += gridDim.y) {
    if (v.tile_pairs(tp, 0) < 0) continue;  // no baseline in this tile pair
    INT_T I, J;
    tile_pair_of_t(tp, n_tiles, I, J);
    const bool same = I == J;
    const INT_T n_sides = same ? 1 : 2;

    for (INT_T cell = blockIdx.x; cell < n_freq * n_tc; cell += gridDim.x) {
      const INT_T f = cell / n_tc, t_local = cell % n_tc, t = t0 + t_local;
      const T freq_f = v.freqs(f);
      __syncthreads();  // the previous cell is done with the shared arrays
      for (INT_T i = tid; i < n_sf * n_int_f; i += kBlockT)
        wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
      for (INT_T i = tid; i < n_st * n_int_t; i += kBlockT)
        wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
      // The tile pair's cotangent weights: W[i, j] gathers the baseline that
      // holds (i, j) in that order and the conjugate of the one holding (j, i);
      // an autocorrelation is both, its cotangent plus the conjugate.
      for (INT_T idx = tid; idx < kTileT * kTileT; idx += kBlockT) {
        const INT_T i = idx / kTileT, j = idx % kTileT;
        const INT_T a1 = I * kTileT + i, a2 = J * kTileT + j;
        Cplx<T> w{0, 0};
        if (a1 < n_ant && a2 < n_ant) {
          const INT_T bl = v.pair(a1, a2);
          if (bl >= 0) w = cadd(w, cscale(inv, v.vis_bar(bl, f, t)));
          const INT_T bl2 = v.pair(a2, a1);
          if (bl2 >= 0) w = cadd(w, cscale(inv, cconj(v.vis_bar(bl2, f, t))));
        }
        Wp[i * kTileT + j] = w;
        Wt[j * kTileT + i] = w;
      }

      for (INT_T r = 0; r < n_rfi; ++r) {
        for (INT_T s0 = 0; s0 < n_s; s0 += kChunk) {
          const INT_T cs = n_s - s0 < kChunk ? n_s - s0 : kChunk;
          __syncthreads();  // weights ready; the previous chunk's contraction done
          for (INT_T idx = tid; idx < cs * kTileT; idx += kBlockT) {
            const INT_T s_local = idx / kTileT, a_local = idx % kTileT;
            const INT_T s = s0 + s_local;
            const INT_T a = I * kTileT + a_local;
            S_I[idx] = a < n_ant ? v.S[sample_index(f, t_local, r, s, a, n_tc, n_rfi, n_s, n_ant)] : Cplx<T>{0, 0};
            if (!same) {
              const INT_T a2 = J * kTileT + a_local;
              S_J[idx] = a2 < n_ant ? v.S[sample_index(f, t_local, r, s, a2, n_tc, n_rfi, n_s, n_ant)] : Cplx<T>{0, 0};
            }
          }
          __syncthreads();
          const Cplx<T> *TJ = same ? S_I : S_J;
          // Q_I: lanes along i read W's transpose and broadcast the partner sample.
          for (INT_T idx = tid; idx < cs * kTileT; idx += kBlockT) {
            const INT_T s_local = idx / kTileT, i = idx % kTileT;
            Cplx<T> acc{0, 0};
            for (INT_T j = 0; j < kTileT; ++j)
              acc = cadd(acc, cmul(Wt[j * kTileT + i], cconj(TJ[s_local * kTileT + j])));
            const INT_T a = I * kTileT + i;
            Q_I[idx] = a < n_ant ? cmul(v.E[sample_index(f, t_local, r, s0 + s_local, a, n_tc, n_rfi, n_s, n_ant)], acc)
                                 : Cplx<T>{0, 0};
            if constexpr (FULL) P_I[idx] = -cmul(S_I[idx], acc).im;
          }
          if (!same) {
            for (INT_T idx = tid; idx < cs * kTileT; idx += kBlockT) {
              const INT_T s_local = idx / kTileT, j = idx % kTileT;
              Cplx<T> acc{0, 0};
              for (INT_T i = 0; i < kTileT; ++i)
                acc = cadd(acc, cmul(cconj(Wp[i * kTileT + j]), cconj(S_I[s_local * kTileT + i])));
              const INT_T a2 = J * kTileT + j;
              Q_J[idx] = a2 < n_ant ? cmul(v.E[sample_index(f, t_local, r, s0 + s_local, a2, n_tc, n_rfi, n_s, n_ant)], acc)
                                    : Cplx<T>{0, 0};
              if constexpr (FULL) P_J[idx] = -cmul(S_J[idx], acc).im;
            }
          }
          __syncthreads();
          // Push the chunk through the stencil weights onto the partial in
          // scratch: one thread per element across the chunks.
          for (INT_T idx = tid; idx < n_sides * kTileT * n_slots; idx += kBlockT) {
            const INT_T a_local = idx % kTileT, rest = idx / kTileT;
            const INT_T slot = rest % n_slots, side = rest / n_slots;
            Cplx<T> h{0, 0};
            if (slot < n_stencil) {
              const INT_T kf = slot / n_st, kt = slot % n_st;
              const Cplx<T> *Q = side ? Q_J : Q_I;
              for (INT_T s_local = 0; s_local < cs; ++s_local) {
                const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
                h = cadd(h, cscale(wf[kf * n_int_f + u] * wt[kt * n_int_t + vv], Q[s_local * kTileT + a_local]));
              }
            } else if constexpr (FULL) {
              // The phase's cotangent, then the delay's per polynomial order.
              const T *P = side ? P_J : P_I;
              const INT_T k = slot - n_stencil - 1;
              for (INT_T s_local = 0; s_local < cs; ++s_local) {
                const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
                const T pc = P[s_local * kTileT + a_local];
                h.re += k < 0 ? pc : delay_phase_coeff(k, freq_f, v.dnu(u), v.dt(vv)) * pc;
              }
            }
            Cplx<T> &target = v.H[h_index(tp, side, a_local, r, f, t_local, slot, n_rfi, n_freq, n_tc, n_slots)];
            target = s0 == 0 ? h : cadd(target, h);
          }
        }
      }
    }
  }
}

// Sum, per data-grid element, the partials of the antenna's tile pairs over
// the chunk's cells whose stencils cover it. Every output element is written
// by one thread per chunk, in a fixed order.
template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_gather(
    TransposeViews<T, INT_T> v, INT_T n_tiles, INT_T t0, INT_T n_tc, INT_T n_slots) {
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
    const INT_T Ia = ant / kTileT, a_local = ant % kTileT;
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
        for (INT_T Jb = 0; Jb < n_tiles; ++Jb) {
          const INT_T tp = Jb >= Ia ? tile_pair_index(Ia, Jb, n_tiles) : tile_pair_index(Jb, Ia, n_tiles);
          if (v.tile_pairs(tp, 0) < 0) continue;  // never written: no baseline there
          const INT_T side = Jb >= Ia ? 0 : 1;
          sum = cadd(sum, v.H[h_index(tp, side, a_local, r, f, t - t0, k * n_st + l, n_rfi, n_freq, n_tc, n_slots)]);
        }
      }
    }
    v.amp_bar(ant, r, fp, tpt) = cadd(v.amp_bar(ant, r, fp, tpt), sum);
  }
}

// The full variant's phase and delay cotangents: per antenna, source and time
// cell of the chunk, the phase's per channel from the cell's own slot, the
// delay's summed over the channels, each over the antenna's tile pairs.
template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_gather_phase(
    TransposeViews<T, INT_T> v, INT_T n_tiles, INT_T t0, INT_T n_tc, INT_T n_slots) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1], n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T n_stencil = n_sf * n_st, n_path = v.delay.shape[3];
  const INT_T n_out = n_ant * n_rfi * n_tc;
  for (INT_T i = blockIdx.x * INT_T(kBlockT) + threadIdx.x; i < n_out;
       i += INT_T(gridDim.x) * kBlockT) {
    const INT_T t_local = i % n_tc, q = i / n_tc;
    const INT_T r = q % n_rfi, ant = q / n_rfi;
    const INT_T Ia = ant / kTileT, a_local = ant % kTileT;
    for (INT_T k = 0; k < n_path; ++k) v.delay_bar(ant, r, t0 + t_local, k) = 0;
    for (INT_T f = 0; f < n_freq; ++f) {
      T phase_sum = 0;
      for (INT_T Jb = 0; Jb < n_tiles; ++Jb) {
        const INT_T tp = Jb >= Ia ? tile_pair_index(Ia, Jb, n_tiles) : tile_pair_index(Jb, Ia, n_tiles);
        if (v.tile_pairs(tp, 0) < 0) continue;
        const INT_T side = Jb >= Ia ? 0 : 1;
        phase_sum += v.H[h_index(tp, side, a_local, r, f, t_local, n_stencil, n_rfi, n_freq, n_tc, n_slots)].re;
        for (INT_T k = 0; k < n_path; ++k)
          v.delay_bar(ant, r, t0 + t_local, k) +=
              v.H[h_index(tp, side, a_local, r, f, t_local, n_stencil + 1 + k, n_rfi, n_freq, n_tc, n_slots)].re;
      }
      v.phase_bar(ant, r, f, t0 + t_local) = phase_sum;
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
  const INT_T n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const INT_T n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const INT_T n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  const INT_T n_stencil = n_sf * n_st, n_s = n_int_f * n_int_t;
  const INT_T n_slots = FULL ? n_stencil + 1 + INT_T(dd[3]) : n_stencil;
  const INT_T n_tiles = (n_ant + kTileT - 1) / kTileT;
  const INT_T n_tile_pairs = n_tiles * (n_tiles + 1) / 2;

  constexpr int kChunk = ChunkT<T>::value;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  const std::size_t shared = head + sizeof(Cplx<T>) * (2 * std::size_t(kTileT) * kTileT + 4 * std::size_t(kChunk) * kTileT)
                             + (FULL ? sizeof(T) * 2 * std::size_t(kChunk) * kTileT : 0);

  // Time chunk: as many cells as keep the scratch (the per-tile-pair partials,
  // the samples and the phase factors) within 256 MB, at least one.
  const std::size_t per_cell_h = sizeof(Cplx<T>) * std::size_t(n_tile_pairs) * 2 * kTileT * n_rfi * n_freq * n_slots;
  const std::size_t per_cell_s = sizeof(Cplx<T>) * std::size_t(n_ant) * n_rfi * n_freq * n_s;
  INT_T n_tc = INT_T((256u * 1024 * 1024) / (per_cell_h + 2 * per_cell_s));
  if (n_tc < 1) n_tc = 1;
  if (n_tc > n_time) n_tc = n_time;
  auto h_mem = scratch.Allocate((per_cell_h + 2 * per_cell_s) * n_tc, alignof(Cplx<T>));
  if (!h_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the per-cell cotangents and samples");
  Cplx<T> *H = reinterpret_cast<Cplx<T> *>(*h_mem);
  Cplx<T> *S = H + per_cell_h * n_tc / sizeof(Cplx<T>);
  Cplx<T> *E = S + per_cell_s * n_tc / sizeof(Cplx<T>);

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

  auto status = cudaMemsetAsync(amp_bar->typed_data(), 0,
                                sizeof(Cplx<T>) * std::size_t(amp.element_count()), stream);
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU memset error: ") + cudaGetErrorString(status));

  for (INT_T t0 = 0; t0 < n_time; t0 += n_tc) {
    const INT_T cells = n_time - t0 < n_tc ? n_time - t0 : n_tc;
    const auto sample_grid = create_clamped_grid(int(n_freq * n_rfi * n_ant), 1, 1);
    rfi_interp_samples_kernel<T, INT_T, kSamplesAndPhase>
        <<<sample_grid, kSampleBlock, 0, stream>>>(sample_views, S, E, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    const auto grid = create_clamped_grid(n_freq * cells, n_tile_pairs, 1);
    rfi_interp_transpose_pairs<T, INT_T, FULL><<<grid, kBlockT, shared, stream>>>(views, n_tiles, n_tile_pairs, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    const std::int64_t reach = std::int64_t(cells) + 2 * (n_st - 1);
    const std::int64_t n_out = std::int64_t(n_ant) * n_rfi * n_freq * reach;
    const auto gather_grid = create_clamped_grid(int((n_out + kBlockT - 1) / kBlockT), 1, 1);
    rfi_interp_transpose_gather<T, INT_T><<<gather_grid, kBlockT, 0, stream>>>(views, n_tiles, t0, cells, n_slots);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    if constexpr (FULL) {
      const std::int64_t n_cells_out = std::int64_t(n_ant) * n_rfi * cells;
      const auto phase_grid = create_clamped_grid(int((n_cells_out + kBlockT - 1) / kBlockT), 1, 1);
      rfi_interp_transpose_gather_phase<T, INT_T><<<phase_grid, kBlockT, 0, stream>>>(views, n_tiles, t0, cells, n_slots);
      status = cudaGetLastError();
      if (status != cudaSuccess)
        return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    }
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
  const std::int64_t n_tiles = (n_ant + kTileT - 1) / kTileT;
  if (tile_pairs.dimensions()[0] != n_tiles * (n_tiles + 1) / 2 ||
      tile_pairs.dimensions()[1] != kTileT * kTileT)
    return ffi::Error::InvalidArgument("Expected a (n_tile_pairs, 1024) tile-pair list");
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
