// The scratch axis is polynomial coefficient, with antennas contiguous. A
// block stages two antenna tiles once per source and frequency offset. Each
// pair's moment recurrence stays within its thread; no array of moments
// crosses a kernel boundary.
//
// Those recurrences run at the operator's own precision, not in double: they
// are what the kernel spends its time on, and a GeForce part issues one
// double-precision instruction per sixty-four single-precision ones, so a
// complex64 call that computed its weights in double pays better than an
// order of magnitude over for digits its coefficient buffers cannot hold. The
// angles the recurrences are seeded from stay in double either way; see the
// working precision W in rfi_analytic_math.hpp for what that division rests on.
#include "gpu_compat.h"
#include "rfi_analytic_common.hpp"
#include "rfi_analytic_scratch_gpu.cuh"
#include "util_gpu.h"
#include "visibility.h"

namespace ri_kernels {
namespace gpu {

// kAnalyticTile, the antennas per tile, comes from rfi_analytic_base.hpp: it
// has to match the tile-pair list the Python op builds.
constexpr int kAnalyticBlock = 128;
static_assert(kAnalyticTilePairs % kAnalyticBlock == 0,
              "kAnalyticTilePairs must be a multiple of kAnalyticBlock");
// The delay orders the closed form carries: tau and its first three
// derivatives, whatever longer path the caller hands in.
constexpr int kAnalyticOrders = kAnalyticDelayOrders;

template <bool JVP, typename T>
__global__ void analytic_materialise(AnalyticViews<T> v, Cplx<T> *s, Cplx<T> *ds,
                                     AnalyticCellChunk<std::int64_t> chunk) {
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], nm = v.gt.shape[2], nu = v.wf.shape[2];
  const auto work = chunk.n_fc * chunk.n_tc * nr * nu * nm * na;
  for (std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += std::int64_t(blockDim.x) * gridDim.x) {
    auto z = index;
    const auto ant = z % na; z /= na;
    const auto m = z % nm; z /= nm;
    const auto u = z % nu; z /= nu;
    const auto r = z % nr; z /= nr;
    const auto t = chunk.t0 + z % chunk.n_tc, f = chunk.f0 + z / chunk.n_tc;
    s[index] = analytic_coefficient(v, v.amp, ant, r, f, t, u, m);
    if constexpr (JVP) ds[index] = analytic_coefficient(v, v.amp_dot, ant, r, f, t, u, m);
  }
}

__device__ inline void analytic_add(float *dest, float value) {
  atomicAdd(dest, value);
}

// Double-precision atomicAdd arrived with Pascal, and the x86_64 CUDA 12 wheel
// still carries an SM 5.2 cubin, so the f64 kernel has to be compilable without
// it. The compare-and-swap loop is NVIDIA's documented stand-in. Only the
// pre-Pascal device passes take it; the host pass and every target this kernel
// is actually run on take the instruction.
__device__ inline void analytic_add(double *dest, double value) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
  atomicAdd(dest, value);
#else
  auto *bits = reinterpret_cast<unsigned long long *>(dest);
  unsigned long long old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
        __double_as_longlong(value + __longlong_as_double(assumed)));
    // The NaN-payload comparison is deliberate: a repeat means another thread
    // won, not that the sum is unrepresentable.
  } while (assumed != old);
#endif
}

template <typename T>
__device__ inline void analytic_atomic(Cplx<T> *dest, Cplx<T> value) {
  analytic_add(&dest->re, value.re);
  analytic_add(&dest->im, value.im);
}

// The transpose reduces within each tile in shared memory, then writes one
// partial per antenna and partner tile. Separate blocks never contend for a
// global cotangent. The stencil gather sums partner tiles in order, keeping
// the scratch bounded by the chunk planner in rfi_analytic_scratch_gpu.cuh.
//
// Modes: 0 forward, 1 signal JVP, 2 full JVP (the phase tangent as well), 3
// transpose (the signal and phase cotangents). The phase enters every weight
// as exp(i (phase_p - phase_q)), so its derivative is the pair's own product
// turned by i: the full JVP adds i (dphase_p - dphase_q) z and the
// transpose sums -Im(g z) into p's slot and +Im(g z) into q's,
// per antenna in shared memory like the coefficient cotangents, written as
// one partial per partner tile and gathered in order afterwards.
//
// Delay, in the last two modes, differentiates the delay as well. Its order
// k moves one angle of the closed form, so its derivative is the product z_k
// against that angle's row of weight derivatives (analytic_weights), turned by
// i and scaled by what the order adds to the angle: the full JVP adds
// i s_k (ddelay_p - ddelay_q) z_k, the transpose sums -s_k Im(g z_k) and
// +s_k Im(g z_k) as for the phase, one partial per order besides.
template <bool Default, int Mode, bool Delay, typename T>
__global__ void analytic_tiles(AnalyticViews<T> v, const Cplx<T> *s,
    const Cplx<T> *ds, Cplx<T> *partials, T *phase_partials, T *delay_partials,
    Tensor3D<const Cplx<T> *> cot, Tensor3D<Cplx<T> *> out,
    AnalyticCellChunk<std::int64_t> chunk) {
  constexpr bool JVP = Mode == 1 || Mode == 2, Transpose = Mode == 3;
  constexpr bool Phase = Mode >= 2;
  static_assert(!Delay || Phase, "The delay derivative rides on the full passes");
  constexpr int orders = Delay ? kAnalyticOrders : 1;
  constexpr int pairs_per_thread = kAnalyticTilePairs / kAnalyticBlock;
  extern __shared__ __align__(16) unsigned char shared[];
  const auto nm = v.gt.shape[2], na = v.amp.shape[0], nr = v.amp.shape[1], nu = v.wf.shape[2];
  const auto ntiles = analytic_tile_count(na);
  Cplx<T> *si = reinterpret_cast<Cplx<T> *>(shared), *sj = si + kAnalyticTile * nm;
  Cplx<T> *xi = Mode ? sj + kAnalyticTile * nm : si;
  Cplx<T> *xj = Mode ? xi + kAnalyticTile * nm : sj;
  // The two tiles' own delay orders and centre phases, order-major over the
  // pair's antennas so that a warp walking a tile diagonal reads one bank
  // each. Tile I occupies the first kAnalyticTile slots of a row, tile J the
  // rest; a tile paired with itself simply stages its antennas twice.
  constexpr int span = 2 * kAnalyticTile;
  T *sd = reinterpret_cast<T *>(Mode ? xj + kAnalyticTile * nm : sj + kAnalyticTile * nm);
  T *sp = sd + kAnalyticOrders * span;
  // The phase tangent (full JVP) or the phase cotangent sums (transpose)
  // of the two tiles' antennas, laid out like sp; the delay's like sd.
  T *sq = Phase ? sp + span : nullptr;
  T *se = Delay ? sq + span : nullptr;
  const std::int64_t npath = v.delay.shape[3] < kAnalyticOrders ? v.delay.shape[3] : kAnalyticOrders;
  for (std::int64_t tp = blockIdx.y; tp < v.tile_pairs.shape[0]; tp += gridDim.y) {
    std::int64_t I = 0, rem = tp;
    while (rem >= ntiles - I) { rem -= ntiles - I; ++I; }
    const auto J = I + rem;
    for (std::int64_t cell = blockIdx.x; cell < chunk.n_fc * chunk.n_tc; cell += gridDim.x) {
      const auto t = chunk.t0 + cell % chunk.n_tc, f = chunk.f0 + cell / chunk.n_tc;
      Cplx<T> acc[pairs_per_thread] = {};
      for (std::int64_t r = 0; r < nr; ++r) {
        // Every pair this block forms draws on 64 antennas, and the whole
        // cell reuses them. Read them once here rather than twice per pair:
        // the gather is by antenna, so each warp would otherwise pull a
        // separate sector per lane, and that cost more than the closed form.
        for (int z = threadIdx.x; z < span; z += blockDim.x) {
          const std::int64_t ant = (z < kAnalyticTile ? I : J) * kAnalyticTile +
                                   (z % kAnalyticTile);
          const bool live = ant < na;
          for (int k = 0; k < kAnalyticOrders; ++k)
            sd[k * span + z] = live && k < v.delay.shape[3] ? v.delay(ant, r, t, k) : T(0);
          sp[z] = live ? v.phase(ant, r, f, t) : T(0);
          if constexpr (JVP && Phase) sq[z] = live ? v.phase_dot(ant, r, f, t) : T(0);
          if constexpr (Transpose) sq[z] = T(0);
          if constexpr (Delay)
            for (int k = 0; k < kAnalyticOrders; ++k)
              se[k * span + z] = JVP && live && k < npath ? v.delay_dot(ant, r, t, k) : T(0);
        }
        __syncthreads();
        for (std::int64_t u = 0; u < nu; ++u) {
          for (std::int64_t z = threadIdx.x; z < nm * kAnalyticTile; z += blockDim.x) {
            const auto m = z / kAnalyticTile, il = z % kAnalyticTile;
            const auto p = I * kAnalyticTile + il, q = J * kAnalyticTile + il;
            const auto o = analytic_index(cell, r, u, m, 0, nr, nu, nm, na);
            si[z] = p < na ? s[o + p] : Cplx<T>{0, 0};
            sj[z] = q < na ? s[o + q] : Cplx<T>{0, 0};
            if constexpr (JVP) {
              xi[z] = p < na ? ds[o + p] : Cplx<T>{0, 0};
              xj[z] = q < na ? ds[o + q] : Cplx<T>{0, 0};
            }
            if constexpr (Transpose) { xi[z] = {0, 0}; xj[z] = {0, 0}; }
          }
          __syncthreads();
          for (int k = 0; k < pairs_per_thread; ++k) {
            const auto pair = v.tile_pairs(tp, threadIdx.x + k * kAnalyticBlock);
            if (pair < 0) continue;
            const auto il = pair / kAnalyticTile, jl = pair % kAnalyticTile;
            const std::int64_t p = I * kAnalyticTile + il, q = J * kAnalyticTile + jl;
            const auto ip = il, iq = kAnalyticTile + jl;
            const AnalyticPair d{
                double(sd[ip]) - double(sd[iq]),
                double(sd[span + ip]) - double(sd[span + iq]),
                double(sd[2 * span + ip]) - double(sd[2 * span + iq]),
                double(sd[3 * span + ip]) - double(sd[3 * span + iq]),
                double(sp[ip]) - double(sp[iq])};
            Cplx<T> h[orders * AnalyticStorage<Default>::product];
            analytic_pair_weights<Default, T, Delay>(v, d, f, u, h);
            double scale[Delay ? kAnalyticOrders : 1];
            if constexpr (Delay) analytic_delay_scales(v, f, u, scale);
            if constexpr (!Transpose) {
              Cplx<T> primal[orders] = {};
              auto z = analytic_contract<Default, T, orders>(si + il, sj + jl, xi + il, xj + jl, h,
                                                             int(nm), kAnalyticTile, JVP,
                                                             Phase ? primal : nullptr);
              if constexpr (Phase) z = cadd(z, cscale(sq[ip] - sq[iq], ctimes_i(primal[0])));
              if constexpr (Delay) {
                RI_ANALYTIC_UNROLL
                for (int o = 0; o < kAnalyticOrders; ++o)
                  z = cadd(z, cscale(T(scale[o]) * (se[o * span + ip] - se[o * span + iq]),
                                     ctimes_i(primal[o])));
              }
              acc[k] = cadd(acc[k], z);
            } else {
              Cplx<T> g{0, 0};
              const auto b1 = v.pair(p, q), b2 = v.pair(q, p);
              if (b1 >= 0) g = cot(b1, f, t);
              if (p != q && b2 >= 0) g = cadd(g, cconj(cot(b2, f, t)));
              g = cscale(T(1) / T(nu), g);
              RI_ANALYTIC_UNROLL
              for (int j = 0; j < (Default ? AnalyticStorage<Default>::coefficients : nm); ++j) {
                Cplx<T> gp{0, 0}, gq{0, 0};
                RI_ANALYTIC_UNROLL
                for (int l = 0; l < (Default ? AnalyticStorage<Default>::coefficients : nm); ++l) {
                  const auto w = cmul(g, h[j + l]);
                  gp = cadd(gp, cmul(w, cconj(sj[l * kAnalyticTile + jl])));
                  gq = cadd(gq, cconj(cmul(w, si[l * kAnalyticTile + il])));
                }
                analytic_atomic(xi + j * kAnalyticTile + il, gp);
                analytic_atomic((I == J ? xi : xj) + j * kAnalyticTile + jl, gq);
              }
              Cplx<T> rows[orders];
              const auto z = analytic_contract<Default, T, orders>(si + il, sj + jl, si + il, sj + jl,
                                                                   h, int(nm), kAnalyticTile, false,
                                                                   Delay ? rows : nullptr);
              const auto gz = cmul(g, z);
              analytic_add(sq + ip, -gz.im);
              analytic_add(sq + (I == J ? 0 : kAnalyticTile) + jl, gz.im);
              if constexpr (Delay) {
                RI_ANALYTIC_UNROLL
                for (int o = 0; o < kAnalyticOrders; ++o) {
                  if (o >= npath) break;
                  const T w = T(scale[o]) * cmul(g, rows[o]).im;
                  analytic_add(se + o * span + ip, -w);
                  analytic_add(se + o * span + (I == J ? 0 : kAnalyticTile) + jl, w);
                }
              }
            }
          }
          __syncthreads();
          if constexpr (Transpose) {
            for (std::int64_t z = threadIdx.x; z < nm * kAnalyticTile; z += blockDim.x) {
              const auto m = z / kAnalyticTile, il = z % kAnalyticTile;
              const auto p = I * kAnalyticTile + il, q = J * kAnalyticTile + il;
              const auto o = analytic_index(cell, r, u, m, 0, nr, nu, nm, na);
              if (p < na) partials[(o + p) * ntiles + J] = xi[z];
              if (I != J && q < na) partials[(o + q) * ntiles + I] = xj[z];
            }
            __syncthreads();
          }
        }
        if constexpr (Transpose) {
          // The phase cotangent has no frequency-offset or coefficient axis:
          // one partial per antenna and partner tile for the whole (cell, r).
          for (int z = threadIdx.x; z < span; z += blockDim.x) {
            const std::int64_t ant = (z < kAnalyticTile ? I : J) * kAnalyticTile +
                                     (z % kAnalyticTile);
            if (ant >= na) continue;
            const auto o = ((cell * nr + r) * na + ant) * ntiles;
            const auto tile = z < kAnalyticTile ? J : I;
            if (z >= kAnalyticTile && I == J) continue;
            phase_partials[o + tile] = sq[z];
            if constexpr (Delay)
              for (int k = 0; k < npath; ++k)
                delay_partials[(o * kAnalyticOrders) + k * ntiles + tile] = se[k * span + z];
          }
          __syncthreads();
        }
      }
      if constexpr (!Transpose) {
        for (int k = 0; k < pairs_per_thread; ++k) {
          const auto pair = v.tile_pairs(tp, threadIdx.x + k * kAnalyticBlock);
          if (pair < 0) continue;
          const auto p = I * kAnalyticTile + pair / kAnalyticTile;
          const auto q = J * kAnalyticTile + pair % kAnalyticTile;
          const auto b1 = v.pair(p, q), b2 = v.pair(q, p);
          const auto z = cscale(T(1) / T(nu), acc[k]);
          if (b1 >= 0) out(b1, f, t) = z;
          if (p != q && b2 >= 0) out(b2, f, t) = cconj(z);
        }
      }
    }
  }
}

// One writer per data-grid amplitude. A chunk may cover only part of the
// contributing cells; its gather still reads the full stencil tables, and
// adds that contribution before the next chunk reuses scratch on the stream.
template <typename T>
__global__ void analytic_gather(AnalyticViews<T> v, const Cplx<T> *partials,
    Tensor4D<Cplx<T> *> out, AnalyticCellChunk<std::int64_t> chunk, bool first) {
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], nf = v.amp.shape[2], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2], ntiles = analytic_tile_count(na);
  for (std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < na * nr * nf * nt; index += std::int64_t(blockDim.x) * gridDim.x) {
    auto z = index;
    const auto t = z % nt; z /= nt;
    const auto f = z % nf; z /= nf;
    const auto r = z % nr, ant = z / nr;
    Cplx<T> total{0, 0};
    const auto f_begin = max(chunk.f0, f - v.wf.shape[1] + 1);
    const auto f_end = min(chunk.f0 + chunk.n_fc, f + v.wf.shape[1]);
    const auto t_begin = max(chunk.t0, t - v.gt.shape[1] + 1);
    const auto t_end = min(chunk.t0 + chunk.n_tc, t + v.gt.shape[1]);
    for (auto fc = f_begin; fc < f_end; ++fc) {
      const auto k = f - v.sf(fc);
      if (k < 0 || k >= v.wf.shape[1]) continue;
      for (auto tc = t_begin; tc < t_end; ++tc) {
        const auto l = t - v.st(tc);
        if (l < 0 || l >= v.gt.shape[1]) continue;
        const auto cell = (fc - chunk.f0) * chunk.n_tc + tc - chunk.t0;
        for (std::int64_t u = 0; u < nu; ++u)
          for (std::int64_t m = 0; m < nm; ++m) {
            Cplx<T> bar{0, 0};
            const auto o = analytic_index(cell, r, u, m, ant, nr, nu, nm, na) * ntiles;
            for (std::int64_t tile = 0; tile < ntiles; ++tile) bar = cadd(bar, partials[o + tile]);
            total = cadd(total, cscale(v.wf(fc, k, u) * v.gt(tc, l, m), bar));
          }
      }
    }
    out.ptr[index] = first ? total : cadd(out.ptr[index], total);
  }
}

// The phase cotangent's gather: the partner tiles' partials summed in order.
// Each cell belongs to exactly one chunk, so every element is assigned once.
template <typename T>
__global__ void analytic_gather_phase(AnalyticViews<T> v, const T *partials,
    Tensor4D<T *> out, AnalyticCellChunk<std::int64_t> chunk) {
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], ntiles = analytic_tile_count(na);
  const auto work = chunk.n_fc * chunk.n_tc * nr * na;
  for (std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += std::int64_t(blockDim.x) * gridDim.x) {
    auto z = index;
    const auto ant = z % na; z /= na;
    const auto r = z % nr; z /= nr;
    const auto t = chunk.t0 + z % chunk.n_tc, f = chunk.f0 + z / chunk.n_tc;
    T total = 0;
    for (std::int64_t tile = 0; tile < ntiles; ++tile) total += partials[index * ntiles + tile];
    out(ant, r, f, t) = total;
  }
}

// The delay cotangent's gather: per antenna, source, cell and order, the
// chunk's channels and the partner tiles' partials summed in order. The delay
// has no channel axis, so a time cell split across frequency chunks adds to
// what its first chunk assigned; orders past the closed form's stay zero.
template <typename T>
__global__ void analytic_gather_delay(AnalyticViews<T> v, const T *partials,
    Tensor4D<T *> out, AnalyticCellChunk<std::int64_t> chunk) {
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], npath = v.delay.shape[3];
  const auto ntiles = analytic_tile_count(na);
  const auto work = chunk.n_tc * nr * na * npath;
  for (std::int64_t index = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < work; index += std::int64_t(blockDim.x) * gridDim.x) {
    auto z = index;
    const auto k = z % npath; z /= npath;
    const auto ant = z % na; z /= na;
    const auto r = z % nr;
    const auto tc = z / nr;
    T total = 0;
    if (k < kAnalyticOrders)
      for (std::int64_t fc = 0; fc < chunk.n_fc; ++fc) {
        const auto cell = fc * chunk.n_tc + tc;
        const auto o = ((((cell * nr + r) * na + ant) * kAnalyticOrders) + k) * ntiles;
        for (std::int64_t tile = 0; tile < ntiles; ++tile) total += partials[o + tile];
      }
    auto &dest = out(ant, r, chunk.t0 + tc, k);
    dest = chunk.f0 == 0 ? total : dest + total;
  }
}

template <bool Default, int Mode, bool Delay, typename T>
ffi::Error analytic_gpu_launch(cudaStream_t stream, ffi::ScratchAllocator &scratch,
    std::int64_t scratch_budget, AnalyticViews<T> v, Cplx<T> *output,
    const Cplx<T> *cotangent, T *phase_output, T *delay_output) {
  constexpr bool JVP = Mode == 1 || Mode == 2, Transpose = Mode == 3;
  constexpr bool Phase = Mode >= 2;
  const auto na = v.amp.shape[0], nr = v.amp.shape[1], nf = v.amp.shape[2], nt = v.amp.shape[3];
  const auto nm = v.gt.shape[2], nu = v.wf.shape[2], ntiles = analytic_tile_count(na);
  const auto npath = v.delay.shape[3];
  std::int64_t per_cell, phase_per_time = 0;
  if (!analytic_checked_product({std::int64_t(sizeof(Cplx<T>)), na, nr, nu, nm}, per_cell))
    return ffi::Error::InvalidArgument("Analytic coefficient size exceeds the 64-bit byte range");
  // The phase partials, one per antenna and partner tile, planned as the
  // frequency-complete per-time-cell buffer and indexed by the chunk's cells;
  // with Delay, as many again per delay order after them.
  if (Transpose && !analytic_checked_product({std::int64_t(sizeof(T)), nf, nr, na, ntiles,
                                              1 + (Delay ? kAnalyticOrders : 0)}, phase_per_time))
    return ffi::Error::InvalidArgument("Analytic phase cotangent size exceeds the 64-bit byte range");
  // Without Delay the transpose's delay cotangent is zero.
  if (Transpose && !Delay) {
    std::int64_t delay_bytes = 0;
    if (!analytic_checked_product({std::int64_t(sizeof(T)), na, nr, nt, npath}, delay_bytes) ||
        std::uint64_t(delay_bytes) > std::numeric_limits<std::size_t>::max())
      return ffi::Error::InvalidArgument("Analytic delay cotangent size exceeds the 64-bit byte range");
    auto error = cudaMemsetAsync(delay_output, 0, std::size_t(delay_bytes), stream);
    if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
  }
  const std::int64_t buffers = Transpose ? 1 + ntiles : JVP ? 2 : 1;
  AnalyticChunkPlan plan;
  auto status = make_analytic_chunk_plan(nt, nf, per_cell, phase_per_time, buffers, scratch_budget, plan);
  if (!status.success()) return status;
  auto mem = scratch.Allocate(std::size_t(plan.total_bytes), 16);
  if (!mem.has_value()) return ffi::Error::Internal("Could not allocate analytic coefficient scratch");
  Cplx<T> *s = reinterpret_cast<Cplx<T> *>(*mem);
  Cplx<T> *extra = Mode ? s + plan.sample_bytes / sizeof(Cplx<T>) : nullptr;
  T *phase_partials = Transpose ? reinterpret_cast<T *>(
      reinterpret_cast<unsigned char *>(*mem) + plan.sample_bytes * buffers) : nullptr;
  T *delay_partials = Transpose && Delay ? phase_partials + plan.n_tc * nf * nr * na * ntiles : nullptr;
  const std::size_t shared = sizeof(Cplx<T>) * nm * kAnalyticTile * (Mode ? 4 : 2) +
                            sizeof(T) * 2 * kAnalyticTile *
                                (kAnalyticOrders + 1 + (Phase ? 1 : 0) + (Delay ? kAnalyticOrders : 0));
  DeviceLimits limits;
  if (auto err = get_device_limits(limits); err != cudaSuccess)
    return ffi::Error::Internal(std::string("GPU device query error: ") + cudaGetErrorString(err));
  if (shared > limits.shared_mem_per_block) return ffi::Error::Internal("Analytic tiles exceed shared memory limit");
  Tensor3D<Cplx<T> *> out(output, v.a1.shape[0], nf, nt);
  Tensor3D<const Cplx<T> *> cot(cotangent, v.a1.shape[0], nf, nt);
  Tensor4D<Cplx<T> *> bar(output, na, nr, nf, nt);
  Tensor4D<T *> phase_bar(phase_output, na, nr, nf, nt);
  Tensor4D<T *> delay_bar(delay_output, na, nr, nt, npath);
  for (std::int64_t t0 = 0; t0 < nt; t0 += plan.n_tc) {
    for (std::int64_t f0 = 0; f0 < nf; f0 += plan.n_fc) {
      const AnalyticCellChunk<std::int64_t> chunk{t0, std::min(plan.n_tc, nt - t0), f0, std::min(plan.n_fc, nf - f0)};
      const auto work = chunk.n_tc * chunk.n_fc * per_cell / sizeof(Cplx<T>);
      const auto grid = create_clamped_grid(analytic_ceil_div(work, 128), 1, 1, limits);
      analytic_materialise<JVP><<<grid, 128, 0, stream>>>(v, s, extra, chunk);
      auto error = cudaGetLastError();
      if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
      const auto pair_grid = create_clamped_grid(chunk.n_tc * chunk.n_fc,
                                                 v.tile_pairs.shape[0], 1, limits);
      analytic_tiles<Default, Mode, Delay><<<pair_grid, kAnalyticBlock, shared, stream>>>(
          v, s, extra, extra, phase_partials, delay_partials, cot, out, chunk);
      error = cudaGetLastError();
      if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
      if constexpr (Transpose) {
        const auto gather_grid = create_clamped_grid(analytic_ceil_div(na * nr * nf * nt, 128), 1, 1, limits);
        analytic_gather<<<gather_grid, 128, 0, stream>>>(v, extra, bar, chunk, t0 == 0 && f0 == 0);
        error = cudaGetLastError();
        if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
        const auto cells = chunk.n_tc * chunk.n_fc;
        const auto phase_grid = create_clamped_grid(analytic_ceil_div(cells * nr * na, 128), 1, 1, limits);
        analytic_gather_phase<<<phase_grid, 128, 0, stream>>>(v, phase_partials, phase_bar, chunk);
        error = cudaGetLastError();
        if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
        if constexpr (Delay) {
          const auto delay_grid = create_clamped_grid(
              analytic_ceil_div(chunk.n_tc * nr * na * npath, 128), 1, 1, limits);
          analytic_gather_delay<<<delay_grid, 128, 0, stream>>>(v, delay_partials, delay_bar, chunk);
          error = cudaGetLastError();
          if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
        }
      }
    }
  }
  return ffi::Error::Success();
}

template <int Mode, typename T, ffi::DataType A, ffi::DataType R>
ffi::Error analytic_gpu_dispatch(cudaStream_t stream, ffi::ScratchAllocator &scratch,
    analytic_index_t a1, analytic_index_t a2, ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles, ffi::Buffer<A, 4> amp, ffi::Buffer<A, 4> dot,
    ffi::Buffer<R, 4> phase, ffi::Buffer<R, 4> phase_dot, ffi::Buffer<R, 4> delay,
    ffi::Buffer<R, 4> delay_dot, ffi::Buffer<R, 3> wf, analytic_index_t sf,
    ffi::Buffer<R, 3> gt, analytic_index_t st, ffi::Buffer<R, 1> dnu,
    ffi::Buffer<R, 0> duration, ffi::Buffer<R, 1> freq,
    ffi::Buffer<A, 3> cot, ffi::Result<ffi::Buffer<A, Mode == 3 ? 4 : 3>> out,
    ffi::Result<ffi::Buffer<R, 4>> *phase_bar, ffi::Result<ffi::Buffer<R, 4>> *delay_bar,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms,
    std::int64_t scratch_mb, bool delay_grad) {
  const AnalyticOptions options{segments, terms, cubic_terms};
  auto status = analytic_validate(a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
                                 delay_dot, wf, sf, gt, st, dnu, duration, freq, options, false);
  if (!status.success()) return status;
  std::int64_t scratch_budget;
  if (!analytic_scratch_bytes(scratch_mb, scratch_budget))
    return ffi::Error::InvalidArgument("Expected scratch_mb to be a positive size in MiB");
  const auto a = amp.dimensions();
  if constexpr (Mode == 3) {
    if (!analytic_same_shape(amp, *out) || cot.dimensions()[0] != a1.element_count() ||
        cot.dimensions()[1] != a[2] || cot.dimensions()[2] != a[3])
      return ffi::Error::InvalidArgument("Invalid analytic transpose output or cotangent shape");
    if (!analytic_same_shape(phase, **phase_bar))
      return ffi::Error::InvalidArgument("Expected the phase cotangent to match the phase");
    if (!analytic_same_shape(delay, **delay_bar))
      return ffi::Error::InvalidArgument("Expected the delay cotangent to match the delay");
  } else {
    if (out->dimensions()[0] != a1.element_count() || out->dimensions()[1] != a[2] || out->dimensions()[2] != a[3])
      return ffi::Error::InvalidArgument("Invalid analytic visibility output shape");
  }
  auto v = analytic_views<T>(a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
                             delay_dot, wf, sf, gt, st, dnu, duration, freq, options);
  auto output = reinterpret_cast<Cplx<T> *>(out->typed_data());
  auto cotangent = reinterpret_cast<const Cplx<T> *>(cot.typed_data());
  T *phase_output = Mode == 3 ? (*phase_bar)->typed_data() : nullptr;
  T *delay_output = Mode == 3 ? (*delay_bar)->typed_data() : nullptr;
  // Only the full JVP and the transpose differentiate the delay; the others
  // never instantiate the delay kernels.
  constexpr bool Full = Mode >= 2;
  const bool defaults = analytic_is_default(gt.dimensions()[2], options), with_delay = Full && delay_grad;
  if (defaults && with_delay)
    return analytic_gpu_launch<true, Mode, Full>(stream, scratch, scratch_budget, v, output, cotangent,
                                                 phase_output, delay_output);
  if (defaults)
    return analytic_gpu_launch<true, Mode, false>(stream, scratch, scratch_budget, v, output, cotangent,
                                                  phase_output, delay_output);
  if (with_delay)
    return analytic_gpu_launch<false, Mode, Full>(stream, scratch, scratch_budget, v, output, cotangent,
                                                  phase_output, delay_output);
  return analytic_gpu_launch<false, Mode, false>(stream, scratch, scratch_budget, v, output, cotangent,
                                                 phase_output, delay_output);
}

#define RI_ANALYTIC_CONTEXT cudaStream_t stream, ffi::ScratchAllocator scratch
#define RI_ANALYTIC_CONTEXT_BIND .Ctx<ffi::PlatformStream<cudaStream_t>>().Ctx<ffi::ScratchAllocator>()
#define RI_ANALYTIC_CONTEXT_PASS stream, scratch
#define RI_ANALYTIC_RETURN ffi::Error
#define RI_ANALYTIC_DISPATCH analytic_gpu_dispatch
#define RI_ANALYTIC_PLATFORM gpu
#include "rfi_analytic_ffi.hpp"

} // namespace gpu
} // namespace ri_kernels
