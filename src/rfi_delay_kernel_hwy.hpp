#pragma once

#include <complex>
#include <cstdint>

#include "tensor.hpp"

namespace ri_kernels {

template <typename T>
void rfi_delay_vis_hwy(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> vis,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t);

template <typename T>
void rfi_delay_jvp_hwy(
    std::int64_t bl_begin, std::int64_t bl_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a2,
    Tensor4D<const std::complex<T> *> amp,
    Tensor4D<const std::complex<T> *> amp_dot,
    Tensor4D<const T *> delay, Tensor4D<const T *> delay_dot,
    Tensor2D<const T *> freq, Tensor3D<std::complex<T> *> out,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t);

template <typename T>
void rfi_delay_transpose_hwy(
    std::int64_t ant_begin, std::int64_t ant_end, T scale,
    Tensor1D<const int *> a1, Tensor1D<const int *> a1_sorter,
    Tensor1D<const int *> a1_start, Tensor1D<const int *> a2,
    Tensor1D<const int *> a2_sorter, Tensor1D<const int *> a2_start,
    Tensor4D<const std::complex<T> *> amp, Tensor4D<const T *> delay,
    Tensor2D<const T *> freq,
    Tensor3D<const std::complex<T> *> vis_bar,
    Tensor4D<std::complex<T> *> amp_bar, Tensor4D<T *> delay_bar,
    std::int64_t n_rfi, std::int64_t n_int_f, std::int64_t n_int_t);

} // namespace ri_kernels
