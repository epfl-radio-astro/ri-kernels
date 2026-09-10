// GPU transpose (VJP with respect to the data-grid signal) of the data-grid
// RFI visibility, staged. See rfi_interp_transpose_kernel.cpp for the
// formulas; here they are arranged as, per source, cell and antenna a,
//
//   G[a](s) = sum_j W[a, j](s) conj(S[j](s)),
//   W[a, j](s) = (vbar[pair(a, j)] + conj(vbar[pair(j, a)])) / count[a, j]  if the
//                baseline integrates time sample v(s), else 0
//
// a product of the (n_ant, n_ant) cotangent matrix W of the cell with the
// conjugated fine samples, which is what a tiled kernel does well: one block
// owns a tile of TA antennas `a` on one cell, holds W's rows for the tile in
// shared memory, and walks the partner antennas in tiles of kPartner whose
// samples it stages once each. An antenna's samples are rebuilt once per own
// tile, n_ant / TA times, instead of once per baseline.
//
// The per-cell stencil cotangents H go to scratch memory a chunk of time
// cells at a time, and a gather kernel accumulates each chunk onto the data
// grid, so the scratch is bounded (a few hundred MB at most) and every output
// element is still written by one thread per chunk, in order: deterministic.

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

template <typename T, typename INT_T> struct TransposeViews {
  Tensor2D<const int *, INT_T> pair;
  Tensor1D<const int *, INT_T> stride;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, delay;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
  Tensor3D<const Cplx<T> *, INT_T> vis_bar;
  // (n_ant, n_rfi, n_freq, n_time_chunk, n_sf * n_st): the per-cell result
  // for the time cells of the current chunk.
  Tensor5D<Cplx<T> *, INT_T> H;
  Tensor4D<Cplx<T> *, INT_T> amp_bar;
};

constexpr int kBlockT = 256;
constexpr int kPartner = 32;   // partner antennas staged per step
constexpr int kOwnMax = 16;    // at most this many own antennas per block
constexpr int kChunkMax = 32;  // at most this many samples per chunk
constexpr int kGPerThread = kOwnMax * kChunkMax / kBlockT;

template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_staged(
    TransposeViews<T, INT_T> v, INT_T tile_own, INT_T chunk, INT_T t0, INT_T n_tc) {
  extern __shared__ unsigned char dynamic_shared[];
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_s = n_int_f * n_int_t, n_stencil = n_sf * n_st;
  const INT_T n_own_tiles = (n_ant + tile_own - 1) / tile_own;
  const INT_T n_partner_tiles = (n_ant + kPartner - 1) / kPartner;

  // Layout: weight rows | W_I (n_ant x tile_own, partner-major) | the mask of
  // the chunk's time samples each (partner, own) baseline integrates | S_J
  // tile (chunk x kPartner, sample-major) | Q (chunk x tile_own) | H_s
  // (tile_own x n_stencil), the complex blocks from 16-byte boundaries.
  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  Cplx<T> *W = reinterpret_cast<Cplx<T> *>(dynamic_shared + head);
  unsigned *Wmask = reinterpret_cast<unsigned *>(W + n_ant * tile_own);
  const std::size_t mask_end = (head + sizeof(Cplx<T>) * n_ant * tile_own +
                                sizeof(unsigned) * n_ant * tile_own + 15) / 16 * 16;
  Cplx<T> *SJ = reinterpret_cast<Cplx<T> *>(dynamic_shared + mask_end);
  Cplx<T> *Q = SJ + chunk * kPartner;
  Cplx<T> *Hs = Q + chunk * tile_own;

  const int tid = threadIdx.x;

  for (INT_T I = blockIdx.y; I < n_own_tiles; I += gridDim.y) {
    for (INT_T cell = blockIdx.x; cell < n_freq * n_tc; cell += gridDim.x) {
      const INT_T f = cell / n_tc, t = t0 + cell % n_tc;
      __syncthreads();  // the previous cell is done with the shared blocks
      for (INT_T i = tid; i < n_sf * n_int_f; i += kBlockT)
        wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
      for (INT_T i = tid; i < n_st * n_int_t; i += kBlockT)
        wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
      const INT_T sf = v.start_freq(f), st = v.start_time(t);
      const T freq_f = v.freqs(f);
      // W[j, a_local] for every partner j: the cotangents of the baselines
      // (a, j) and (j, a), the second conjugated, over the number of samples
      // the baseline integrates.
      for (INT_T idx = tid; idx < n_ant * tile_own; idx += kBlockT) {
        const INT_T j = idx / tile_own, a = I * tile_own + idx % tile_own;
        Cplx<T> w{0, 0};
        INT_T bstride = 1;
        if (a < n_ant) {
          const INT_T b1 = v.pair(a, j);
          if (b1 >= 0) { w = cadd(w, v.vis_bar(b1, f, t)); bstride = v.stride(b1); }
          const INT_T b2 = v.pair(j, a);
          if (b2 >= 0) { w = cadd(w, cconj(v.vis_bar(b2, f, t))); bstride = v.stride(b2); }
        }
        const INT_T count = stride_count(n_int_t, bstride);
        W[idx] = count > 0 ? cscale(T(1) / T(n_int_f * count), w) : Cplx<T>{0, 0};
      }

      for (INT_T r = 0; r < n_rfi; ++r) {
        __syncthreads();
        for (INT_T idx = tid; idx < tile_own * n_stencil; idx += kBlockT) Hs[idx] = Cplx<T>{0, 0};

        for (INT_T s0 = 0; s0 < n_s; s0 += chunk) {
          const INT_T cs = n_s - s0 < chunk ? n_s - s0 : chunk;
          const INT_T v_lo = s0 / n_int_f;
          // Which of the chunk's time samples each (partner, own) baseline
          // integrates, one bit per time sample from v_lo.
          __syncthreads();  // the previous chunk's products are done with the masks
          for (INT_T idx = tid; idx < n_ant * tile_own; idx += kBlockT) {
            const INT_T j = idx / tile_own, a = I * tile_own + idx % tile_own;
            INT_T bstride = 1;
            if (a < n_ant) {
              INT_T bl = v.pair(a, j);
              if (bl < 0) bl = v.pair(j, a);
              if (bl >= 0) bstride = v.stride(bl);
            }
            Wmask[idx] = stride_mask(v_lo, n_int_t, bstride);
          }
          // G for this thread's (a_local, s_local) entries, in registers.
          Cplx<T> g[kGPerThread];
          for (int m = 0; m < kGPerThread; ++m) g[m] = Cplx<T>{0, 0};

          for (INT_T J = 0; J < n_partner_tiles; ++J) {
            __syncthreads();  // the previous partner tile's products are done
            // Skip a partner tile with no baseline to any antenna of this tile:
            // its W block is zero, so it would contribute nothing at the cost
            // of staging its samples (a variable-sampling call covers a
            // fraction of the baselines).
            {
              __shared__ int any_baseline;
              if (tid == 0) any_baseline = 0;
              __syncthreads();
              for (INT_T idx = tid; idx < tile_own * kPartner; idx += kBlockT) {
                const INT_T a = I * tile_own + idx / kPartner, j = J * kPartner + idx % kPartner;
                if (a < n_ant && j < n_ant && (v.pair(a, j) >= 0 || v.pair(j, a) >= 0))
                  any_baseline = 1;
              }
              __syncthreads();
              const bool skip = !any_baseline;
              __syncthreads();  // everyone has read the flag before it is reset again
              if (skip) continue;
            }
            for (INT_T idx = tid; idx < cs * kPartner; idx += kBlockT) {
              const INT_T s_local = idx / kPartner, jl = idx % kPartner;
              const INT_T j = J * kPartner + jl, s = s0 + s_local;
              Cplx<T> S{0, 0};
              if (j < n_ant) {
                const INT_T vv = s / n_int_f, u = s % n_int_f;  // time-major
                const auto e = phase_factor(v.phase, v.delay, freq_f, v.dnu(u), v.dt(vv), j, r, f, t);
                S = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, j, r, u, vv), e);
              }
              SJ[idx] = S;
            }
            __syncthreads();
            const INT_T j_end = n_ant - J * kPartner < kPartner ? n_ant - J * kPartner : kPartner;
            for (int m = 0; m < kGPerThread; ++m) {
              const INT_T idx = tid + m * kBlockT;
              if (idx >= cs * tile_own) break;
              const INT_T s_local = idx / tile_own, a_local = idx % tile_own;
              const unsigned bit = 1u << ((s0 + s_local) / n_int_f - v_lo);
              Cplx<T> acc = g[m];
              for (INT_T jl = 0; jl < j_end; ++jl) {
                const INT_T j = J * kPartner + jl;
                if (!(Wmask[j * tile_own + a_local] & bit)) continue;
                acc = cadd(acc, cmul(W[j * tile_own + a_local], cconj(SJ[s_local * kPartner + jl])));
              }
              g[m] = acc;
            }
          }
          // Q = e^{i phi_a} G, then the stencil contraction of this chunk.
          __syncthreads();
          for (int m = 0; m < kGPerThread; ++m) {
            const INT_T idx = tid + m * kBlockT;
            if (idx >= cs * tile_own) break;
            const INT_T s_local = idx / tile_own, a_local = idx % tile_own;
            const INT_T a = I * tile_own + a_local, s = s0 + s_local;
            Cplx<T> q{0, 0};
            if (a < n_ant) {
              const INT_T vv = s / n_int_f, u = s % n_int_f;  // time-major
              q = cmul(phase_factor(v.phase, v.delay, freq_f, v.dnu(u), v.dt(vv), a, r, f, t), g[m]);
            }
            Q[idx] = q;
          }
          __syncthreads();
          for (INT_T idx = tid; idx < tile_own * n_stencil; idx += kBlockT) {
            const INT_T a_local = idx / n_stencil, kl = idx % n_stencil;
            const INT_T k = kl / n_st, l = kl % n_st;
            Cplx<T> h = Hs[idx];
            for (INT_T s_local = 0; s_local < cs; ++s_local) {
              const INT_T s = s0 + s_local;
              const T w = wf[k * n_int_f + s % n_int_f] * wt[l * n_int_t + s / n_int_f];
              h = cadd(h, cscale(w, Q[s_local * tile_own + a_local]));
            }
            Hs[idx] = h;
          }
        }
        __syncthreads();
        for (INT_T idx = tid; idx < tile_own * n_stencil; idx += kBlockT) {
          const INT_T a = I * tile_own + idx / n_stencil, kl = idx % n_stencil;
          if (a < n_ant) v.H(a, r, f, t - t0, kl) = Hs[idx];
        }
      }
    }
  }
}

// Accumulates the chunk's per-cell cotangents onto the data grid: each output
// element in the range the chunk's stencils can reach, from the cells whose
// stencils cover it. One thread per element, added in chunk order.
template <typename T, typename INT_T>
__global__ void __launch_bounds__(kBlockT) rfi_interp_transpose_gather(
    TransposeViews<T, INT_T> v, INT_T t0, INT_T n_tc) {
  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_st = v.w_time.shape[1];
  const INT_T tp_lo = t0 - n_st + 1 > 0 ? t0 - n_st + 1 : 0;
  const INT_T tp_hi = t0 + n_tc + n_st - 1 < n_time ? t0 + n_tc + n_st - 1 : n_time;
  const INT_T n_tp = tp_hi - tp_lo;
  const INT_T n_out = n_ant * n_rfi * n_freq * n_tp;
  for (INT_T i = blockIdx.x * INT_T(kBlockT) + threadIdx.x; i < n_out;
       i += INT_T(gridDim.x) * kBlockT) {
    const INT_T tp = tp_lo + i % n_tp;
    const INT_T q = i / n_tp;
    const INT_T fp = q % n_freq;
    const INT_T q2 = q / n_freq;
    const INT_T r = q2 % n_rfi;
    const INT_T ant = q2 / n_rfi;
    Cplx<T> sum{0, 0};
    const INT_T f_lo = fp - n_sf + 1 > 0 ? fp - n_sf + 1 : 0;
    const INT_T f_hi = fp + n_sf < n_freq ? fp + n_sf : n_freq;
    const INT_T t_lo = tp - n_st + 1 > t0 ? tp - n_st + 1 : t0;
    const INT_T t_hi = tp + n_st < t0 + n_tc ? tp + n_st : t0 + n_tc;
    for (INT_T f = f_lo; f < f_hi; ++f) {
      const INT_T k = fp - v.start_freq(f);
      if (k < 0 || k >= n_sf) continue;
      for (INT_T t = t_lo; t < t_hi; ++t) {
        const INT_T l = tp - v.start_time(t);
        if (l < 0 || l >= n_st) continue;
        sum = cadd(sum, v.H(ant, r, f, t - t0, k * n_st + l));
      }
    }
    v.amp_bar(ant, r, fp, tp) = cadd(v.amp_bar(ant, r, fp, tp), sum);
  }
}

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_transpose_gpu_dispatch(
    cudaStream_t stream, ffi::ScratchAllocator &scratch,
    ffi::BufferR2<ffi::S32> pair, interp_index_t stride, ffi::Buffer<AMP_DT, 4> amp,
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
  const INT_T n_stencil = n_sf * n_st, n_s = n_int_f * n_int_t;

  // Own tile: as many antennas as keep W's rows and their masks within 24 KB,
  // at most kOwnMax.
  INT_T tile_own = kOwnMax;
  while (tile_own > 1 &&
         std::size_t(n_ant) * tile_own * (sizeof(Cplx<T>) + sizeof(unsigned)) > 24 * 1024)
    tile_own /= 2;
  const INT_T chunk = n_s < kChunkMax ? n_s : kChunkMax;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  const std::size_t mask_end = (head + sizeof(Cplx<T>) * n_ant * tile_own +
                                sizeof(unsigned) * n_ant * tile_own + 15) / 16 * 16;
  const std::size_t shared =
      mask_end + sizeof(Cplx<T>) * (chunk * kPartner + chunk * tile_own + tile_own * n_stencil);

  // Time chunk: as many cells as keep the scratch within 256 MB, at least one.
  const std::size_t per_cell = sizeof(Cplx<T>) * std::size_t(n_ant) * n_rfi * n_freq * n_stencil;
  INT_T n_tc = INT_T((256u * 1024 * 1024) / per_cell);
  if (n_tc < 1) n_tc = 1;
  if (n_tc > n_time) n_tc = n_time;
  auto h_mem = scratch.Allocate(per_cell * n_tc, alignof(Cplx<T>));
  if (!h_mem.has_value())
    return ffi::Error::Internal("Could not allocate scratch memory for the per-cell cotangents");

  TransposeViews<T, INT_T> views{
      Tensor2D<const int *, INT_T>(pair.typed_data(), pair.dimensions()[0], pair.dimensions()[1]),
      Tensor1D<const int *, INT_T>(stride.typed_data(), stride.dimensions()[0]),
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
      Tensor5D<Cplx<T> *, INT_T>(reinterpret_cast<Cplx<T> *>(*h_mem), a[0], a[1], a[2], n_tc, n_stencil),
      Tensor4D<Cplx<T> *, INT_T>(
          reinterpret_cast<Cplx<T> *>(amp_bar->typed_data()), a[0], a[1], a[2], a[3]),
  };
  const INT_T n_own_tiles = (n_ant + tile_own - 1) / tile_own;

  auto status = cudaMemsetAsync(amp_bar->typed_data(), 0,
                                sizeof(Cplx<T>) * std::size_t(amp.element_count()), stream);
  if (status != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU memset error: ") + cudaGetErrorString(status));

  for (INT_T t0 = 0; t0 < n_time; t0 += n_tc) {
    const INT_T cells = n_time - t0 < n_tc ? n_time - t0 : n_tc;
    const auto grid = create_clamped_grid(n_freq * cells, n_own_tiles, 1);
    rfi_interp_transpose_staged<T, INT_T><<<grid, kBlockT, shared, stream>>>(
        views, tile_own, chunk, t0, cells);
    status = cudaGetLastError();
    if (status != cudaSuccess)
      return ffi::Error::Internal(std::string("GPU kernel launch error: ") + cudaGetErrorString(status));
    const std::int64_t reach = std::int64_t(cells) + 2 * (n_st - 1);
    const std::int64_t n_out = std::int64_t(n_ant) * n_rfi * n_freq * reach;
    const auto gather_grid = create_clamped_grid(int((n_out + kBlockT - 1) / kBlockT), 1, 1);
    rfi_interp_transpose_gather<T, INT_T><<<gather_grid, kBlockT, 0, stream>>>(views, t0, cells);
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
    interp_index_t stride, ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<REAL_DT, 4> phase,
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
  if (vis_bar.dimensions()[0] != a1.dimensions()[0] ||
      vis_bar.dimensions()[1] != amp.dimensions()[2] ||
      vis_bar.dimensions()[2] != amp.dimensions()[3])
    return ffi::Error::InvalidArgument(
        "Expected the visibility cotangent to match the baseline, frequency, "
        "and time extents");
  if (!interp_same_shape(*amp_bar, amp))
    return ffi::Error::InvalidArgument("Expected the signal cotangent to match the signal");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  const std::int64_t h_count = amp.element_count() * w_freq.dimensions()[1] * w_time.dimensions()[1];
  // use 32 bit indexing if possible
  if (h_count < limit && vis_bar.element_count() < limit)
    return calc_rfi_interp_transpose_gpu_dispatch<T, std::int32_t>(
        stream, scratch, pair, stride, amp, phase, delay, w_freq, start_freq, w_time,
        start_time, dnu, dt, freqs, vis_bar, amp_bar);
  return calc_rfi_interp_transpose_gpu_dispatch<T, std::int64_t>(
      stream, scratch, pair, stride, amp, phase, delay, w_freq, start_freq, w_time,
      start_time, dnu, dt, freqs, vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f32_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    interp_index_t stride, interp_amp_f32_t amp, interp_real4_f32_t phase, interp_real4_f32_t delay,
    interp_real3_f32_t w_freq, interp_index_t start_freq,
    interp_real3_f32_t w_time, interp_index_t start_time,
    interp_real1_f32_t dnu, interp_real1_f32_t dt, interp_real1_f32_t freqs,
    ffi::BufferR3<ffi::C64> vis_bar, ffi::Result<interp_amp_f32_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C64, ffi::F32, float>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, stride, amp,
      phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
      vis_bar, amp_bar);
}

ffi::Error calc_rfi_interp_transpose_gpu_f64_impl(
    cudaStream_t stream, ffi::ScratchAllocator scratch, interp_index_t a1,
    interp_index_t a1_sorter, interp_index_t a1_start, interp_index_t a2,
    interp_index_t a2_sorter, interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    interp_index_t stride, interp_amp_f64_t amp, interp_real4_f64_t phase, interp_real4_f64_t delay,
    interp_real3_f64_t w_freq, interp_index_t start_freq,
    interp_real3_f64_t w_time, interp_index_t start_time,
    interp_real1_f64_t dnu, interp_real1_f64_t dt, interp_real1_f64_t freqs,
    ffi::BufferR3<ffi::C128> vis_bar, ffi::Result<interp_amp_f64_t> amp_bar) {
  return calc_rfi_interp_transpose_gpu_impl_tmpl<ffi::C128, ffi::F64, double>(
      stream, scratch, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, stride, amp,
      phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt, freqs,
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
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>().Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>().Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>().Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<interp_amp_f64_t>());

} // namespace gpu
} // namespace ri_kernels
