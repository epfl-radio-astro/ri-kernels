#pragma once

#include <cassert>
#include <cstdint>
#include "tensor.hpp"

namespace ri_kernels {

struct AnalyticOptions {
  std::int64_t segments, terms, cubic_terms;
};

// These are the supported parameters, not independently chosen array sizes.
// Every buffer and dispatch decision below follows from these two settings.
constexpr int kAnalyticDefaultCoefficients = 3;
constexpr AnalyticOptions kAnalyticDefaultOptions{2, 6, 3};
constexpr int kAnalyticMaxCoefficients = 9;
constexpr AnalyticOptions kAnalyticMaxOptions{1024, 32, 8};

struct AnalyticSizes {
  int coefficients, cubic_terms, product, moments, linear;
};

TAB_H_D constexpr AnalyticSizes analytic_sizes(int coefficients, int terms, int cubic_terms) {
  // Convolution doubles the amplitude degree. Each retained cubic term adds
  // three powers, then each curvature term adds two. Disabling cubic phase
  // still retains its zeroth term, so it cannot reduce the product capacity.
  const int nc = cubic_terms > 0 ? cubic_terms : 1;
  const int product = 2 * (coefficients - 1) + 1;
  const int moments = product + 3 * (nc - 1);
  return {coefficients, nc, product, moments, moments + 2 * (terms - 1)};
}

template <bool Default> struct AnalyticStorage {
  static constexpr int coefficients = Default ? kAnalyticDefaultCoefficients : kAnalyticMaxCoefficients;
  static constexpr int terms = Default ? kAnalyticDefaultOptions.terms : kAnalyticMaxOptions.terms;
  static constexpr int cubic_terms = Default ? kAnalyticDefaultOptions.cubic_terms : kAnalyticMaxOptions.cubic_terms;
  static constexpr auto sizes = analytic_sizes(coefficients, terms, cubic_terms);
  static constexpr int product = sizes.product;
  static constexpr int moments = sizes.moments;
  static constexpr int linear = sizes.linear;

  // These are the largest indices used by convolution, cubic contraction,
  // and curvature contraction respectively. The tail of the downward
  // recurrence is a scalar and never extends the stored linear moments.
  static_assert(2 * (coefficients - 1) == product - 1);
  static_assert((product - 1) + 3 * (sizes.cubic_terms - 1) == moments - 1);
  static_assert((moments - 1) + 2 * (terms - 1) == linear - 1);
};

TAB_H_D constexpr bool analytic_is_default(std::int64_t coefficients, AnalyticOptions options) {
  // Segmentation changes how many times the local calculation runs, not its
  // storage. It is deliberately absent from the specialisation predicate.
  return coefficients == kAnalyticDefaultCoefficients &&
         options.terms == kAnalyticDefaultOptions.terms &&
         options.cubic_terms == kAnalyticDefaultOptions.cubic_terms;
}

TAB_H_D constexpr bool analytic_configuration_fits(std::int64_t coefficients, AnalyticOptions options) {
  // Check the public limits before arithmetic or narrowing to int. Direct
  // FFI callers must get the same protection as the Python wrapper.
  if (coefficients < 1 || coefficients > kAnalyticMaxCoefficients ||
      options.segments < 1 || options.segments > kAnalyticMaxOptions.segments ||
      options.terms < 1 || options.terms > kAnalyticMaxOptions.terms ||
      options.cubic_terms < 0 || options.cubic_terms > kAnalyticMaxOptions.cubic_terms)
    return false;
  const auto need = analytic_sizes(int(coefficients), int(options.terms), int(options.cubic_terms));
  using Capacity = AnalyticStorage<false>;
  return need.product <= Capacity::product && need.moments <= Capacity::moments &&
         need.linear <= Capacity::linear;
}

static_assert(analytic_configuration_fits(kAnalyticDefaultCoefficients, kAnalyticDefaultOptions));
static_assert(analytic_configuration_fits(kAnalyticMaxCoefficients, kAnalyticMaxOptions));
static_assert(!analytic_configuration_fits(kAnalyticMaxCoefficients + 1, kAnalyticMaxOptions));
static_assert(!analytic_configuration_fits(kAnalyticMaxCoefficients,
              {2, kAnalyticMaxOptions.terms + 1, kAnalyticMaxOptions.cubic_terms}));
static_assert(!analytic_configuration_fits(kAnalyticMaxCoefficients,
              {2, kAnalyticMaxOptions.terms, kAnalyticMaxOptions.cubic_terms + 1}));

} // namespace ri_kernels
