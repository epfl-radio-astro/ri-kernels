// GPU forward and JVP of the data-grid RFI visibility, staged.
//
// The antennas are cut into tiles of TA. One block works one unordered pair
// of tiles (I, J) on one cell (f, t): for each source and each chunk of the
// cell's fine samples it builds the samples of the two tiles once, into
// shared memory, then forms every baseline product of the tile pair from
// there. An antenna's samples are therefore rebuilt once per tile it is
// paired with, n_ant / TA times, instead of once per baseline it is on. The
// products of the two orderings of a pair are conjugates, so each pair is
// formed once and written to whichever of its orderings the baseline list
// holds, through the (n_ant, n_ant) pair table the caller supplies.
//
// Shared memory is bounded whatever the problem size: the chunk of samples
// is sized so the staged tiles stay within ~32 KB, which any device with the
// 48 KB default allows without opting in.

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

template <typename T, typename INT_T> struct InterpViews {
  Tensor1D<const int *, INT_T> a1, a2;
  Tensor2D<const int *, INT_T> pair;
  Tensor4D<const Cplx<T> *, INT_T> amp;
  Tensor4D<const T *, INT_T> phase, delay;
  Tensor3D<const T *, INT_T> w_freq, w_time;
  Tensor1D<const int *, INT_T> start_freq, start_time;
  Tensor1D<const T *, INT_T> dnu, dt, freqs;
};

template <typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
InterpViews<T, INT_T> make_views(
    interp_index_t a1, interp_index_t a2, ffi::BufferR2<ffi::S32> pair,
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

// The fine sample S = A exp(i phi) of antenna `a`, source `r`, sample s of
// cell (f, t); the tangent dS beside it when JVP.
template <typename T, typename INT_T, bool JVP>
__device__ inline void staged_sample(
    InterpViews<T, INT_T> &v, Tensor4D<const Cplx<T> *, INT_T> amp_dot,
    const T *wf, const T *wt, INT_T sf, INT_T st, INT_T n_sf, INT_T n_st,
    INT_T n_int_f, INT_T n_int_t, T freq_f, INT_T a, INT_T r, INT_T f,
    INT_T t, INT_T s, Cplx<T> &S, Cplx<T> &dS) {
  const INT_T u = s / n_int_t, vv = s % n_int_t;
  const auto e = phase_factor(v.phase, v.delay, freq_f, v.dnu(u), v.dt(vv), a, r, f, t);
  S = cmul(interp_amp(v.amp, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, a, r, u, vv), e);
  if constexpr (JVP) {
    dS = cmul(interp_amp(amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, a, r, u, vv), e);
  }
}

constexpr int kTile = 32;      // antennas per tile
constexpr int kBlock = 256;    // threads per block
constexpr int kPairs = kTile * kTile / kBlock;  // baseline pairs per thread

template <typename T, bool JVP, typename INT_T>
__global__ void __launch_bounds__(kBlock) rfi_interp_staged_kernel(
    T scale, InterpViews<T, INT_T> v, Tensor4D<const Cplx<T> *, INT_T> amp_dot,
    Tensor3D<Cplx<T> *, INT_T> out, INT_T chunk, INT_T n_tiles, INT_T n_tile_pairs) {
  extern __shared__ unsigned char dynamic_shared[];

  const INT_T n_ant = v.amp.shape[0], n_rfi = v.amp.shape[1];
  const INT_T n_freq = v.amp.shape[2], n_time = v.amp.shape[3];
  const INT_T n_sf = v.w_freq.shape[1], n_int_f = v.w_freq.shape[2];
  const INT_T n_st = v.w_time.shape[1], n_int_t = v.w_time.shape[2];
  const INT_T n_s = n_int_f * n_int_t;
  const INT_T n_cells = n_freq * n_time;

  // Layout: the cell's weight rows, then the staged tiles, sample-major
  // (tile[s * kTile + a]) so a warp's consecutive antennas read consecutive
  // words. The tiles start at a 16-byte boundary.
  T *wf = reinterpret_cast<T *>(dynamic_shared);
  T *wt = wf + n_sf * n_int_f;
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  Cplx<T> *tile_I = reinterpret_cast<Cplx<T> *>(dynamic_shared + head);
  Cplx<T> *tile_J = tile_I + chunk * kTile;
  Cplx<T> *dtile_I = tile_J + chunk * kTile;
  Cplx<T> *dtile_J = dtile_I + chunk * kTile;

  const int tid = threadIdx.x;
  const int j_local = tid % kTile;  // this thread's partner antenna within tile J

  for (INT_T tp = blockIdx.y; tp < n_tile_pairs; tp += gridDim.y) {
    // The unordered pair (I, J), I <= J, of this linear index.
    INT_T I = 0, rem = tp;
    while (rem >= n_tiles - I) { rem -= n_tiles - I; ++I; }
    const INT_T J = I + rem;
    const bool same = I == J;

    for (INT_T cell = blockIdx.x; cell < n_cells; cell += gridDim.x) {
      const INT_T f = cell / n_time, t = cell % n_time;
      __syncthreads();  // the previous cell's tiles and rows are done with
      for (INT_T i = tid; i < n_sf * n_int_f; i += kBlock)
        wf[i] = v.w_freq(f, i / n_int_f, i % n_int_f);
      for (INT_T i = tid; i < n_st * n_int_t; i += kBlock)
        wt[i] = v.w_time(t, i / n_int_t, i % n_int_t);
      const INT_T sf = v.start_freq(f), st = v.start_time(t);
      const T freq_f = v.freqs(f);

      Cplx<T> acc[kPairs];
      for (int k = 0; k < kPairs; ++k) acc[k] = Cplx<T>{0, 0};

      for (INT_T r = 0; r < n_rfi; ++r) {
        for (INT_T s0 = 0; s0 < n_s; s0 += chunk) {
          const INT_T cs = n_s - s0 < chunk ? n_s - s0 : chunk;
          __syncthreads();  // rows loaded; the previous chunk's products done
          // Stage: tile I, and tile J when it is a different tile.
          for (INT_T idx = tid; idx < cs * kTile; idx += kBlock) {
            const INT_T s_local = idx / kTile, a_local = idx % kTile;
            const INT_T s = s0 + s_local;
            Cplx<T> S{0, 0}, dS{0, 0};
            const INT_T a = I * kTile + a_local;
            if (a < n_ant)
              staged_sample<T, INT_T, JVP>(v, amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, freq_f, a, r, f, t, s, S, dS);
            tile_I[idx] = S;
            if constexpr (JVP) dtile_I[idx] = dS;
            if (!same) {
              Cplx<T> S2{0, 0}, dS2{0, 0};
              const INT_T a2 = J * kTile + a_local;
              if (a2 < n_ant)
                staged_sample<T, INT_T, JVP>(v, amp_dot, wf, wt, sf, st, n_sf, n_st, n_int_f, n_int_t, freq_f, a2, r, f, t, s, S2, dS2);
              tile_J[idx] = S2;
              if constexpr (JVP) dtile_J[idx] = dS2;
            }
          }
          __syncthreads();
          const Cplx<T> *TJ = same ? tile_I : tile_J;
          const Cplx<T> *DJ = same ? dtile_I : dtile_J;
          // Multiply: pair k of this thread is (i = tid / kTile + 8 k, j = tid % kTile).
          for (INT_T s = 0; s < cs; ++s) {
            const Cplx<T> b = TJ[s * kTile + j_local];
            Cplx<T> db{0, 0};
            if constexpr (JVP) db = DJ[s * kTile + j_local];
            for (int k = 0; k < kPairs; ++k) {
              const int i_local = tid / kTile + k * (kBlock / kTile);
              const Cplx<T> a = tile_I[s * kTile + i_local];
              if constexpr (JVP) {
                const Cplx<T> da = dtile_I[s * kTile + i_local];
                acc[k] = cadd(acc[k], cadd(cmul(da, cconj(b)), cmul(a, cconj(db))));
              } else {
                acc[k] = cadd(acc[k], cmul(a, cconj(b)));
              }
            }
          }
        }
      }

      // Write each pair to the orderings the baseline list holds. Within a
      // tile paired with itself the pairs (i, j) and (j, i) are both formed;
      // only i <= j writes, and covers both orderings.
      for (int k = 0; k < kPairs; ++k) {
        const int i_local = tid / kTile + k * (kBlock / kTile);
        if (same && i_local > j_local) continue;
        const INT_T a1 = I * kTile + i_local, a2 = J * kTile + j_local;
        if (a1 >= n_ant || a2 >= n_ant) continue;
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

// The chunk of samples staged at once: as many as keep the tiles within the
// budget, at most the cell's sample count.
template <typename T>
std::int64_t staged_chunk(std::int64_t n_s, int n_tiles_shared, std::size_t budget) {
  const std::size_t per_sample = sizeof(Cplx<T>) * kTile * n_tiles_shared;
  std::int64_t chunk = std::int64_t(budget / per_sample);
  if (chunk > 32) chunk = 32;
  if (chunk > n_s) chunk = n_s;
  return chunk < 1 ? 1 : chunk;
}

template <bool JVP, typename T, typename INT_T, ffi::DataType AMP_DT, ffi::DataType REAL_DT>
ffi::Error calc_rfi_interp_gpu_dispatch(
    cudaStream_t stream, interp_index_t a1, interp_index_t a2,
    ffi::BufferR2<ffi::S32> pair, ffi::Buffer<AMP_DT, 4> amp,
    ffi::Buffer<AMP_DT, 4> amp_dot, ffi::Buffer<REAL_DT, 4> phase,
    ffi::Buffer<REAL_DT, 4> delay, ffi::Buffer<REAL_DT, 3> w_freq,
    interp_index_t start_freq, ffi::Buffer<REAL_DT, 3> w_time,
    interp_index_t start_time, ffi::Buffer<REAL_DT, 1> dnu,
    ffi::Buffer<REAL_DT, 1> dt, ffi::Buffer<REAL_DT, 1> freqs,
    ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  auto views = make_views<T, INT_T>(a1, a2, pair, amp, phase, delay, w_freq, start_freq,
                                    w_time, start_time, dnu, dt, freqs);
  const auto a = amp.dimensions();
  Tensor4D<const Cplx<T> *, INT_T> amp_dot_view(
      reinterpret_cast<const Cplx<T> *>(amp_dot.typed_data()), a[0], a[1], a[2], a[3]);
  Tensor3D<Cplx<T> *, INT_T> out_view(
      reinterpret_cast<Cplx<T> *>(out->typed_data()), out->dimensions()[0],
      out->dimensions()[1], out->dimensions()[2]);
  const INT_T n_sf = w_freq.dimensions()[1], n_int_f = w_freq.dimensions()[2];
  const INT_T n_st = w_time.dimensions()[1], n_int_t = w_time.dimensions()[2];
  const INT_T n_s = n_int_f * n_int_t;
  const T scale = T(1) / T(n_s);
  const INT_T n_tiles = (a[0] + kTile - 1) / kTile;
  const INT_T n_tile_pairs = n_tiles * (n_tiles + 1) / 2;
  const int n_tiles_shared = JVP ? 4 : 2;
  const INT_T chunk = staged_chunk<T>(n_s, n_tiles_shared, 32 * 1024);
  const std::size_t head = (sizeof(T) * (n_sf * n_int_f + n_st * n_int_t) + 15) / 16 * 16;
  const std::size_t shared = head + sizeof(Cplx<T>) * kTile * chunk * n_tiles_shared;
  const auto grid = create_clamped_grid(a[2] * a[3], n_tile_pairs, 1);
  rfi_interp_staged_kernel<T, JVP, INT_T><<<grid, kBlock, shared, stream>>>(
      scale, views, amp_dot_view, out_view, chunk, n_tiles, n_tile_pairs);
  const auto status = cudaGetLastError();
  return status == cudaSuccess
             ? ffi::Error::Success()
             : ffi::Error::Internal(std::string("GPU kernel launch error: ") +
                                    cudaGetErrorString(status));
}

template <bool JVP, ffi::DataType AMP_DT, ffi::DataType REAL_DT, typename T>
ffi::Error calc_rfi_interp_gpu_impl_tmpl(
    cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,
    interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,
    interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair,
    ffi::Buffer<AMP_DT, 4> amp, ffi::Buffer<AMP_DT, 4> amp_dot,
    ffi::Buffer<REAL_DT, 4> phase, ffi::Buffer<REAL_DT, 4> delay,
    ffi::Buffer<REAL_DT, 3> w_freq, interp_index_t start_freq,
    ffi::Buffer<REAL_DT, 3> w_time, interp_index_t start_time,
    ffi::Buffer<REAL_DT, 1> dnu, ffi::Buffer<REAL_DT, 1> dt,
    ffi::Buffer<REAL_DT, 1> freqs, ffi::Result<ffi::BufferR3<AMP_DT>> out) {
  if (!interp_shapes_are_valid(a1, a2, amp, phase, delay, w_freq, start_freq,
                               w_time, start_time, dnu, dt, freqs))
    return ffi::Error::InvalidArgument(
        "Incompatible signal, phase, delay, table, or baseline shapes");
  if (pair.dimensions()[0] != amp.dimensions()[0] || pair.dimensions()[1] != amp.dimensions()[0])
    return ffi::Error::InvalidArgument("Expected an (n_ant, n_ant) pair table");
  if (JVP && !interp_same_shape(amp_dot, amp))
    return ffi::Error::InvalidArgument("Expected the signal tangent to match the signal");
  constexpr std::int64_t limit = std::numeric_limits<std::int32_t>::max();
  // use 32 bit indexing if possible
  if (amp.element_count() < limit && out->element_count() < limit)
    return calc_rfi_interp_gpu_dispatch<JVP, T, std::int32_t>(
        stream, a1, a2, pair, amp, amp_dot, phase, delay, w_freq, start_freq, w_time,
        start_time, dnu, dt, freqs, out);
  return calc_rfi_interp_gpu_dispatch<JVP, T, std::int64_t>(
      stream, a1, a2, pair, amp, amp_dot, phase, delay, w_freq, start_freq, w_time,
      start_time, dnu, dt, freqs, out);
}

#define RI_INTERP_GPU_ENTRY(NAME, JVP, AMP_DT, REAL_DT, T, AMP_T, R4, R3, R1, DOT_PARAM, DOT_ARG) \
  ffi::Error NAME##_impl(                                                       \
      cudaStream_t stream, interp_index_t a1, interp_index_t a1_sorter,          \
      interp_index_t a1_start, interp_index_t a2, interp_index_t a2_sorter,      \
      interp_index_t a2_start, ffi::BufferR2<ffi::S32> pair, AMP_T amp DOT_PARAM, \
      R4 phase, R4 delay, R3 w_freq, interp_index_t start_freq, R3 w_time,        \
      interp_index_t start_time, R1 dnu, R1 dt, R1 freqs,                        \
      ffi::Result<ffi::BufferR3<AMP_DT>> out) {                                  \
    return calc_rfi_interp_gpu_impl_tmpl<JVP, AMP_DT, REAL_DT, T>(               \
        stream, a1, a1_sorter, a1_start, a2, a2_sorter, a2_start, pair, amp,     \
        DOT_ARG, phase, delay, w_freq, start_freq, w_time, start_time, dnu, dt,   \
        freqs, out);                                                             \
  }

#define RI_COMMA_DOT_F32 , interp_amp_f32_t amp_dot
#define RI_COMMA_DOT_F64 , interp_amp_f64_t amp_dot
#define RI_NO_DOT

RI_INTERP_GPU_ENTRY(calc_rfi_interp_gpu_f32, false, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, RI_NO_DOT, amp)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_gpu_f64, false, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, RI_NO_DOT, amp)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_jvp_gpu_f32, true, ffi::C64, ffi::F32, float, interp_amp_f32_t, interp_real4_f32_t, interp_real3_f32_t, interp_real1_f32_t, RI_COMMA_DOT_F32, amp_dot)
RI_INTERP_GPU_ENTRY(calc_rfi_interp_jvp_gpu_f64, true, ffi::C128, ffi::F64, double, interp_amp_f64_t, interp_real4_f64_t, interp_real3_f64_t, interp_real1_f64_t, RI_COMMA_DOT_F64, amp_dot)

// Exported: see visibility.h. The attribute has to sit inside the extern "C".
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_gpu_f64(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_jvp_gpu_f32(XLA_FFI_CallFrame *call_frame);
extern "C" RI_KERNELS_API XLA_FFI_Error *calc_rfi_interp_jvp_gpu_f64(XLA_FFI_CallFrame *call_frame);

#define RI_INTERP_INDEX_ARGS                                                   \
  .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()           \
      .Arg<interp_index_t>().Arg<interp_index_t>().Arg<interp_index_t>()       \
      .Arg<ffi::BufferR2<ffi::S32>>()
#define RI_INTERP_TABLE_ARGS(P)                                                \
  .Arg<interp_real4_##P##_t>().Arg<interp_real4_##P##_t>()                     \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real3_##P##_t>().Arg<interp_index_t>()                       \
      .Arg<interp_real1_##P##_t>().Arg<interp_real1_##P##_t>()                 \
      .Arg<interp_real1_##P##_t>()

XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f32, calc_rfi_interp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_gpu_f64, calc_rfi_interp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f32, calc_rfi_interp_jvp_gpu_f32_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f32_t>()
                                  .Arg<interp_amp_f32_t>()
                                  RI_INTERP_TABLE_ARGS(f32)
                                  .Ret<ffi::BufferR3<ffi::C64>>());
XLA_FFI_DEFINE_HANDLER_SYMBOL(calc_rfi_interp_jvp_gpu_f64, calc_rfi_interp_jvp_gpu_f64_impl,
                              ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  RI_INTERP_INDEX_ARGS.Arg<interp_amp_f64_t>()
                                  .Arg<interp_amp_f64_t>()
                                  RI_INTERP_TABLE_ARGS(f64)
                                  .Ret<ffi::BufferR3<ffi::C128>>());

} // namespace gpu
} // namespace ri_kernels
