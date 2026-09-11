// GPU transpose (VJP with respect to the data-grid signal) of the data-grid
// RFI visibility, staged. See rfi_interp_transpose_kernel.cpp for the
// formulas. Here they are arranged as the forward's mirror: one block works
// one unordered pair of antenna tiles (I, J) on one cell, stages the fine
// samples of the two tiles once per source and chunk, and walks the pairs the
// baseline list holds in the tile pair, sorted by stride (tile_pairs, see
// rfi_interp_vis_op.py), each over its own subset of the time samples. A pair
// (i, j) with cotangent weight w (the baseline's cotangent over its sample
// count, conjugated when the list holds the pair as (j, i)) contributes
//
//   G_I[i](s) += w conj(S_J[j](s)),   G_J[j](s) += conj(w) conj(S_I[i](s)),
//
// accumulated into per-tile cotangent tiles in shared memory with atomics
// (the list orders a stride's pairs along the tile diagonals, so a warp's
// lanes add to distinct antennas); each tile is then turned by its own phase
// factors and pushed back through the stencil weights into a per-tile-pair
// partial of the stencil cotangents H, accumulated over the chunks in
// scratch memory by the thread that owns the element. The partials are
// chunked over time cells, and a gather kernel sums each antenna's tile pairs
// and the cells whose stencils cover a data-grid element, in a fixed order.
// The per-pair accumulation order within a block is not fixed, so the result
// is reproducible to rounding rather than to the bit.
//
// Shared memory: the cell's weight rows, the cotangent weights of the tile
// pair (8 KB single, 16 KB double) and the two staged tiles with their two
// cotangent tiles (16 KB, the chunk is sized for it): within the 48 KB every
// device offers, whatever the stencil.

#include <cstdint>
#include <limits>
#include <string>

#include "gpu_compat.h"
#include "rfi_interp_common.hpp"
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
constexpr int kPairsT = kTileT * kTileT / kBlockT;  // list entries per thread

template <typename T, typename INT_T> struct TransposeViews {
  Tensor2D<const int *, INT_T> pair;
  Tensor1D<const int *, INT_T> stride;
  Tensor2D<const int *, INT_T> tile_pairs;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, delay;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
  Tensor3D<const Cplx<T> *, INT_T> vis_bar;
  // Per-tile-pair partials of the per-cell stencil cotangents for the time
  // cells of the current chunk, laid out
  // (n_tile_pairs, 2 sides, kTileT, n_rfi, n_freq, n_time_chunk, n_sf * n_st).
  Cplx<T> *H;
  Tensor4D<Cplx<T> *, INT_T> amp_bar;
};

template <typename T>
__device__ inline std::int64_t h_index(std::int64_t tp, std::int64_t side, std::int64_t a_local,
                                       std::int64_t r, std::int64_t f, std::int64_t t_local,
                                       std::int64_t kl, std::int64_t n_rfi, std::int64_t n_freq,
                                       std::int64_t n_tc, std::int64_t n_stencil) {
  return (((((tp * 2 + side) * kTileT + a_local) * n_rfi + r) * n_freq + f) * n_tc + t_local) * n_stencil + kl;
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

// The first time sample >= v_lo that a baseline of stride `st` integrates.
template <typename INT_T>
__device__ inline INT_T first_taken_t(INT_T v_lo, INT_T st) {
  const INT_T v0 = st / 2;
  if (v_lo <= v0) return v0;
  return v0 + (v_lo - v0 + st - 1) / st * st;
}

template <typename T>
__device__ inline void atomic_cadd(Cplx<T> *target, Cplx<T> x) {
  atomicAdd(&target->re, x.re);
  atomicAdd(&target->im, x.im);
}

template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_pairs(
    TransposeViews<T, INT_T> v, INT_T c_v, INT_T n_tiles, INT_T n_tile_pairs,
    INT_T t0, INT_T n_tc) {
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_stencil = n_sf * n_st;
  const INT_T chunk = c_v * n_int_f;

  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  Cplx<T> *Wp = reinterpret_cast<Cplx<T> *>(dynamic_shared + head);
  Cplx<T> *tile_I = Wp + kTileT * kTileT;
  Cplx<T> *tile_J = tile_I + chunk * kTileT;
  Cplx<T> *G_I = tile_J + chunk * kTileT;
  Cplx<T> *G_J = G_I + chunk * kTileT;

  const int tid = threadIdx.x;

  for (INT_T tp = blockIdx.y; tp < n_tile_pairs; tp += gridDim.y) {
    if (v.tile_pairs(tp, 0) < 0) continue;  // no baseline in this tile pair
    INT_T I, J;
    tile_pair_of_t(tp, n_tiles, I, J);
    const bool same = I == J;
    const INT_T n_sides = same ? 1 : 2;

    INT_T pi[kPairsT], pj[kPairsT], pstride[kPairsT];
    T pinv[kPairsT];
    for (int k = 0; k < kPairsT; ++k) {
      const INT_T e = v.tile_pairs(tp, tid + k * kBlockT);
      pstride[k] = 0;
      if (e < 0) continue;
      pi[k] = (e & (kTileT * kTileT - 1)) / kTileT;
      pj[k] = e & (kTileT - 1);
      // The entry's stride; the orderings of the pair that share it are the
      // ones the entry stands for (an ordering with another stride has its
      // own entry).
      const INT_T bstride = e >> 10;
      pstride[k] = bstride;
      const INT_T count = stride_count(n_int_t, bstride);
      pinv[k] = count > 0 ? T(1) / T(n_int_f * count) : T(0);
    }

    for (INT_T cell = blockIdx.x; cell < n_freq * n_tc; cell += gridDim.x) {
      const INT_T f = cell / n_tc, t_local = cell % n_tc, t = t0 + t_local;
      __syncthreads();  // the previous cell is done with the shared arrays
      for (INT_T i = tid; i < n_sf * n_int_f; i += kBlockT)
        wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
      for (INT_T i = tid; i < n_st * n_int_t; i += kBlockT)
        wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
      // The cotangent weight of each of this thread's pairs on this cell.
      for (int k = 0; k < kPairsT; ++k) {
        if (pstride[k] == 0) continue;
        const INT_T a1 = I * kTileT + pi[k], a2 = J * kTileT + pj[k];
        Cplx<T> w{0, 0};
        const INT_T bl = v.pair(a1, a2);
        if (bl >= 0 && v.stride(bl) == pstride[k]) w = cadd(w, cscale(pinv[k], v.vis_bar(bl, f, t)));
        if (a1 != a2) {
          const INT_T bl2 = v.pair(a2, a1);
          if (bl2 >= 0 && v.stride(bl2) == pstride[k])
            w = cadd(w, cscale(pinv[k], cconj(v.vis_bar(bl2, f, t))));
        }
        Wp[tid + k * kBlockT] = w;
      }
      const INT_T sf = v.start_freq(f), st = v.start_time(t);
      const T freq_f = v.freqs(f);

      for (INT_T r = 0; r < n_rfi; ++r) {
        for (INT_T v_lo = 0; v_lo < n_int_t; v_lo += c_v) {
          const INT_T cv = n_int_t - v_lo < c_v ? n_int_t - v_lo : c_v;
          const INT_T s0 = v_lo * n_int_f, cs = cv * n_int_f;
          __syncthreads();  // weights ready; the previous chunk's contraction done
          // Stage the samples of both tiles and clear their cotangent tiles.
          for (INT_T idx = tid; idx < cs * kTileT; idx += kBlockT) {
            const INT_T s_local = idx / kTileT, a_local = idx % kTileT;
            const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
            const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
            Cplx<T> S{0, 0};
            const INT_T a = I * kTileT + a_local;
            if (a < n_ant)
              S = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, a, r, u, vv),
                       phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, a, r, f, t));
            tile_I[idx] = S;
            G_I[idx] = Cplx<T>{0, 0};
            if (!same) {
              Cplx<T> S2{0, 0};
              const INT_T a2 = J * kTileT + a_local;
              if (a2 < n_ant)
                S2 = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, a2, r, u, vv),
                          phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, a2, r, f, t));
              tile_J[idx] = S2;
              G_J[idx] = Cplx<T>{0, 0};
            }
          }
          __syncthreads();
          const Cplx<T> *TJ = same ? tile_I : tile_J;
          Cplx<T> *GJ = same ? G_I : G_J;
          // Scatter: each pair over its own time samples.
          for (int k = 0; k < kPairsT; ++k) {
            const INT_T bstride = pstride[k];
            if (bstride == 0) continue;
            const Cplx<T> w = Wp[tid + k * kBlockT], wc = cconj(w);
            const INT_T i_local = pi[k], j_local = pj[k];
            for (INT_T vv = first_taken_t(v_lo, bstride); vv < v_lo + cv; vv += bstride) {
              for (INT_T u = 0; u < n_int_f; ++u) {
                const INT_T s_local = (vv - v_lo) * n_int_f + u;
                const Cplx<T> Si = tile_I[s_local * kTileT + i_local];
                const Cplx<T> Sj = TJ[s_local * kTileT + j_local];
                atomic_cadd(&G_I[s_local * kTileT + i_local], cmul(w, cconj(Sj)));
                atomic_cadd(&GJ[s_local * kTileT + j_local], cmul(wc, cconj(Si)));
              }
            }
          }
          __syncthreads();
          // Turn each antenna's cotangent samples by its own phase factor.
          for (INT_T idx = tid; idx < cs * kTileT; idx += kBlockT) {
            const INT_T s_local = idx / kTileT, a_local = idx % kTileT;
            const INT_T s = s0 + s_local, vv = s / n_int_f, u = s % n_int_f;
            const T dnu_u = v.dnu(u), dt_v = v.dt(vv);
            const INT_T a = I * kTileT + a_local;
            if (a < n_ant)
              G_I[idx] = cmul(phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, a, r, f, t), G_I[idx]);
            if (!same) {
              const INT_T a2 = J * kTileT + a_local;
              if (a2 < n_ant)
                G_J[idx] = cmul(phase_factor(v.phase, v.delay, freq_f, dnu_u, dt_v, a2, r, f, t), G_J[idx]);
            }
          }
          __syncthreads();
          // Push the chunk back through the stencil weights, onto the partial
          // in scratch: one thread per element across the chunks.
          for (INT_T idx = tid; idx < n_sides * kTileT * n_stencil; idx += kBlockT) {
            const INT_T side = idx / (kTileT * n_stencil), rem = idx % (kTileT * n_stencil);
            const INT_T a_local = rem / n_stencil, kl = rem % n_stencil;
            const INT_T kf = kl / n_st, kt = kl % n_st;
            const Cplx<T> *G = side ? G_J : G_I;
            Cplx<T> h{0, 0};
            for (INT_T vv = v_lo; vv < v_lo + cv; ++vv) {
              const T wtv = wt[kt * n_int_t + vv];
              for (INT_T u = 0; u < n_int_f; ++u)
                h = cadd(h, cscale(wf[kf * n_int_f + u] * wtv, G[((vv - v_lo) * n_int_f + u) * kTileT + a_local]));
            }
            Cplx<T> &target = v.H[h_index<T>(tp, side, a_local, r, f, t_local, kl, n_rfi, n_freq, n_tc, n_stencil)];
            target = v_lo == 0 ? h : cadd(target, h);
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
    TransposeViews<T, INT_T> v, INT_T n_tiles, INT_T t0, INT_T n_tc) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T n_stencil = n_sf * n_st;
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
          sum = cadd(sum, v.H[h_index<T>(tp, side, a_local, r, f, t - t0, k * n_st + l, n_rfi, n_freq, n_tc, n_stencil)]);
        }
      }
    }
    v.amp_bar(ant, r, fp, tpt) = cadd(v.amp_bar(ant, r, fp, tpt), sum);
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::ScratchAllocator &scratch,
    ffi::BufferR2<ffi::S32> pair, interp_index_t stride, ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::BufferR3<AMP_DT> vis_bar,
    ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar) {
  const auto a = amp.dimensions();
  const INT_T n_ant = a[0], n_rfi = a[1], n_freq = a[2], n_time = a[3];
  const INT_T n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const INT_T n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  const INT_T n_stencil = n_sf * n_st;
  const INT_T n_tiles = (n_ant + kTileT - 1) / kTileT;
  const INT_T n_tile_pairs = n_tiles * (n_tiles + 1) / 2;

  // Whole time samples per chunk: as many as keep the four tiles within 16 KB.
  const std::size_t per_time_sample = sizeof(Cplx<T>) * kTileT * 4 * n_int_f;
  INT_T c_v = INT_T((16u * 1024) / per_time_sample);
  if (c_v > 32) c_v = 32;
  if (c_v > n_int_t) c_v = n_int_t;
  if (c_v < 1) c_v = 1;
  const INT_T chunk = c_v * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  const std::size_t shared =
      head + sizeof(Cplx<T>) * (std::size_t(kTileT) * kTileT + 4 * std::size_t(chunk) * kTileT);

  // Time chunk: as many cells as keep the scratch within 256 MB, at least one.
  const std::size_t per_cell = sizeof(Cplx<T>) * std::size_t(n_tile_pairs) * 2 * kTileT * n_rfi * n_freq * n_stencil;
  INT_T n_tc = INT_T((256u * 1024 * 1024) / per_cell);
  if (n_tc < 1) n_tc = 1;
  if (n_tc > n_time) n_tc = n_time;
  auto h_mem = scratch.Allocate(per_cell * n_tc, alignof(Cplx<T>));
  if (!h_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the per-cell cotangents");

  TransposeViews<T, INT_T> views{
      Tensor2D<const int *, INT_T>(pair.typed_data(), pair.dimensions()[0], pair.dimensions()[1]),
      Tensor1D<const int *, INT_T>(stride.typed_data(), stride.dimensions()[0]),
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
      reinterpret_cast<Cplx<T> *>(*h_mem),
      Tensor4D<Cplx<T> *, INT_T>(
          reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), a[0], a[1], a[2], a[3]),
  };

  auto status = cudaMemsetAsync(amp_bar->typed_data(), 0,
                                sizeof(Cplx<T>) * std::size_t(amp.element_count()), stream);
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU memset error: ") + cudaGetErrorString(status));

  for (INT_T t0 = 0; t0 < n_time; t0 += n_tc) {
    const INT_T cells = n_time - t0 < n_tc ? n_time - t0 : n_tc;
    const auto grid = create_clamped_grid(n_freq * cells, n_tile_pairs, 1);
    rfi_interp_transpose_pairs<T, INT_T><<<grid, kBlockT, shared, stream>>>(
        views, c_v, n_tiles, n_tile_pairs, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    const std::int64_t reach = std::int64_t(cells) + 2 * (n_st - 1);
    const std::int64_t n_out = std::int64_t(n_ant) * n_rfi * n_freq * reach;
    const auto gather_grid = create_clamped_grid(int((n_out + kBlockT - 1) / kBlockT), 1, 1);
    rfi_interp_transpose_gather<T, INT_T><<<gather_grid, kBlockT, 0, stream>>>(views, n_tiles, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
  }
  return ffi::Error::Success();
}

// The allocator is move-only: the entry points take it by value from the
// binding and lend it down the call chain by reference.
template <ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_transpose_gpu_impl_tmpl(
    cudaStream_t stream, ffi::ScratchAllocator &scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    interp_index_t stride, ffi::BufferR2<ffi::S32> tile_pairs,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::BufferR3<AMP_DT> vis_bar, ffi::Result<ffi::Buffer<AMP_DT, 4>> amp_bar) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, delay, table, or baseline shapes");
  if (pair.dimensions()[0] != amp.dimensions()[0] || pair.dimensions()[1] != amp.dimensions()[0])
    return ffi::Error::InvalidArgument("Expected an (n_ant, n_ant) pair table");
  if (stride.dimensions()[0] != a1.dimensions()[0])
    return ffi::Error::InvalidArgument("Expected one stride per baseline");
  const std::int64_t n_tiles = (amp.dimensions()[0] + kTileT - 1) / kTileT;
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
    return calc_rfi_interp_transpose_gpu_dispatch<T, std::int32_t>(
        stream, scratch, pair, stride, tile_pairs, amp, phase, delay, w_freq, start_freq, w_time,
        start_time, dnu, dt, freqs, vis_bar, amp_bar);
  return calc_rfi_interp_transpose_gpu_dispatch<T, std::int64_t>(
      stream, scratch, pair, stride, tile_pairs, amp, phase, delay, w_freq, start_freq, w_time,
      start_time, dnu, dt, freqs, vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f32_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    interp_index_t stride, ffi::BufferR2<ffi::S32> tile_pairs,
    interp_amp_f32_t amp, interp_real4_f32_t phase, interp_real4_f32_t delay,
    interp_real3_f32_t w_freq, interp_index_t start_freq,
    interp_real3_f32_t w_time, interp_index_t start_time,
    interp_real1_f32_t dnu, interp_real1_f32_t dt, interp_real1_f32_t freqs,
    ffi::BufferR3<ffi::C64> vis_bar, ffi::Result<interp_amp_f32_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, stride,
      tile_pairs, amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f64_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    interp_index_t stride, ffi::BufferR2<ffi::S32> tile_pairs,
    interp_amp_f64_t amp, interp_real4_f64_t phase, interp_real4_f64_t delay,
    interp_real3_f64_t w_freq, interp_index_t start_freq,
    interp_real3_f64_t w_time, interp_index_t start_time,
    interp_real1_f64_t dnu, interp_real1_f64_t dt, interp_real1_f64_t freqs,
    ffi::BufferR3<ffi::C128> vis_bar, ffi::Result<interp_amp_f64_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, stride,
      tile_pairs, amp, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *
calc_rfi_interp_transpose_gpu_f64(XLA_FFI_CallFrame *call_frame);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_rfi_interp_transpose_gpu_f32, calc_rfi_interp_transpose_gpu_f32_impl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
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
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

} // namespace gpu
} // namespace ri_kernels
