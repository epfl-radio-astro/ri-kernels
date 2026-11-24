#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TAB_H_D __host__ __device__
#else
#define TAB_H_D
#endif

namespace ri_kernels {

namespace detail {
template <typename INDEX_T>
TAB_H_D inline bool in_bounds(INDEX_T idx, INDEX_T limit) {
  if constexpr (std::is_signed_v<INDEX_T>) {
    return idx >= 0 && idx < limit;
  } else {
    return idx < limit;
  }
}
} // namespace detail

/**
 * @brief A 1D tensor view wrapper for convenient indexing.
 *
 * @tparam POINTER_T The type of the pointer (e.g., int*, const float*).
 * @tparam INDEX_T The type of the index (default: std::int64_t).
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor1D {
  Tensor1D(POINTER_T p, INDEX_T s0) : ptr(p), shape{s0} {}

  TAB_H_D inline auto &operator()(INDEX_T i0) const {
    assert(detail::in_bounds(i0, shape[0]));
    return ptr[i0];
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[1] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 2D tensor view wrapper for convenient indexing.
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor2D {
  Tensor2D(POINTER_T p, INDEX_T s0, INDEX_T s1) : ptr(p), shape{s0, s1} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1) const {
    assert(detail::in_bounds(i0, shape[0]));
    assert(detail::in_bounds(i1, shape[1]));
    assert(i1 + shape[1] * i0 < shape[0] * shape[1]);
    return ptr[i1 + shape[1] * i0];
  }

  TAB_H_D inline Tensor1D<POINTER_T, INDEX_T> slice(INDEX_T i0) const {
    assert(detail::in_bounds(i0, shape[0]));
    return Tensor1D<POINTER_T, INDEX_T>(ptr + i0 * shape[1], shape[1]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[2] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 3D tensor view wrapper for convenient indexing.
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor3D {
  Tensor3D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2)
      : ptr(p), shape{s0, s1, s2} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2) const {
    assert(detail::in_bounds(i0, shape[0]));
    assert(detail::in_bounds(i1, shape[1]));
    assert(detail::in_bounds(i2, shape[2]));
    assert(i2 + shape[2] * (i1 + i0 * shape[1]) <
           shape[0] * shape[1] * shape[2]);
    return ptr[i2 + shape[2] * (i1 + i0 * shape[1])];
  }

  TAB_H_D inline Tensor2D<POINTER_T, INDEX_T> slice(INDEX_T i0) const {
    assert(detail::in_bounds(i0, shape[0]));
    return Tensor2D<POINTER_T, INDEX_T>(ptr + i0 * shape[1] * shape[2],
                                        shape[1], shape[2]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[3] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 4D tensor view wrapper for convenient indexing.
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor4D {
  Tensor4D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3)
      : ptr(p), shape{s0, s1, s2, s3} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2,
                                  INDEX_T i3) const {
    assert(detail::in_bounds(i0, shape[0]));
    assert(detail::in_bounds(i1, shape[1]));
    assert(detail::in_bounds(i2, shape[2]));
    assert(detail::in_bounds(i3, shape[3]));
    return ptr[i3 + shape[3] * (i2 + shape[2] * (i1 + i0 * shape[1]))];
  }

  TAB_H_D inline Tensor3D<POINTER_T, INDEX_T> slice(INDEX_T i0) const {
    assert(detail::in_bounds(i0, shape[0]));
    return Tensor3D<POINTER_T, INDEX_T>(
        ptr + i0 * shape[1] * shape[2] * shape[3], shape[1], shape[2],
        shape[3]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[4] = {0}; ///< Shape of the tensor.
};

/**
 * @brief A 5D tensor view wrapper for convenient indexing.
 */
template <typename POINTER_T, typename INDEX_T = std::int64_t> struct Tensor5D {
  Tensor5D(POINTER_T p, INDEX_T s0, INDEX_T s1, INDEX_T s2, INDEX_T s3,
           INDEX_T s4)
      : ptr(p), shape{s0, s1, s2, s3, s4} {}

  TAB_H_D inline auto &operator()(INDEX_T i0, INDEX_T i1, INDEX_T i2,
                                  INDEX_T i3, INDEX_T i4) const {
    assert(detail::in_bounds(i0, shape[0]));
    assert(detail::in_bounds(i1, shape[1]));
    assert(detail::in_bounds(i2, shape[2]));
    assert(detail::in_bounds(i3, shape[3]));
    assert(detail::in_bounds(i4, shape[4]));
    return ptr[i4 +
               shape[4] *
                   (i3 + shape[3] * (i2 + shape[2] * (i1 + i0 * shape[1])))];
  }

  TAB_H_D inline Tensor4D<POINTER_T, INDEX_T> slice(INDEX_T i0) const {
    assert(detail::in_bounds(i0, shape[0]));
    return Tensor4D<POINTER_T, INDEX_T>(
        ptr + i0 * shape[1] * shape[2] * shape[3] * shape[4], shape[1],
        shape[2], shape[3], shape[4]);
  }

  POINTER_T ptr = nullptr;      ///< Pointer to the data buffer.
  const INDEX_T shape[5] = {0}; ///< Shape of the tensor.
};

} // namespace ri_kernels
