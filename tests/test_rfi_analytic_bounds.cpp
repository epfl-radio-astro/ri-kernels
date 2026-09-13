// Exact-sized allocations expose each kernel buffer to ASan independently of
// XLA's allocator. The standalone cases also distinguish native kernel writes
// from work performed by the differentiated JAX reference in the Python tests.
#include <iostream>
#include <stdexcept>
#include "rfi_analytic_kernel.cpp"

namespace ri_kernels {

template <typename T>
void check_cells(int count, AnalyticOptions options) {
  const std::int64_t na = 2, nr = 1, nf = 1, nt = count, nu = 2, nb = 3;
  std::vector<int> a1{0, 0, 1}, a2{0, 1, 1}, pair{0, 1, -1, 2}, tiles(1024, -1);
  std::vector<int> sf(nf, 0), st(nt, 0);
  std::vector<Cplx<T>> amp(na * nr * nf * nt, {T(1), T(.2)});
  std::vector<Cplx<T>> dot(amp.size(), {T(.3), T(.7)}), bar(amp.size());
  std::vector<Cplx<T>> vis(nb * nf * nt), tangent(vis.size()), cot(vis.size(), {T(.5), T(.2)});
  std::vector<T> phase(amp.size(), T(.1)), delay(na * nr * nt * 4, T(0));
  std::vector<T> wf(nf * nu, T(1)), gt(nt * count * count, T(0));
  std::vector<T> dnu{T(-.0025), T(0)}, freqs{T(150)};
  T duration = T(2);
  // A constant signal with nonzero coefficients through the last order
  // exercises every convolution and transpose index while remaining bounded.
  for (std::int64_t t = 0; t < nt; ++t) {
    for (int m = 0; m < count; ++m) gt[(t * count + t) * count + m] = T(1. / (m + 1));
    delay[t * 4 + 1] = T(.07);
    delay[t * 4 + 2] = T(.0003);
    delay[t * 4 + 3] = T(.000003);
  }
  AnalyticViews<T> v{
      {a1.data(), nb}, {a2.data(), nb}, {pair.data(), na, na}, {tiles.data(), 1, 1024},
      {amp.data(), na, nr, nf, nt}, {dot.data(), na, nr, nf, nt},
      {phase.data(), na, nr, nf, nt}, {delay.data(), na, nr, nt, 4},
      {wf.data(), nf, 1, nu}, {gt.data(), nt, count, count},
      {sf.data(), nf}, {st.data(), nt}, {dnu.data(), nu}, {freqs.data(), nf}, &duration, options};
  analytic_cpu_cells<false, false>(0, nf * nt, v, {vis.data(), nb, nf, nt});
  analytic_cpu_cells<false, true>(0, nf * nt, v, {tangent.data(), nb, nf, nt});
  analytic_cpu_transpose<false>(0, nr, v, {cot.data(), nb, nf, nt}, {bar.data(), na, nr, nf, nt});
  for (const auto *values : {&vis, &tangent, &bar})
    for (const auto z : *values)
      if (!std::isfinite(z.re) || !std::isfinite(z.im)) throw std::runtime_error("Nonfinite cell result");
}

void check_moments() {
  for (int count = 1; count <= 9; ++count)
    for (int terms : {1, 6, 32})
      for (int cubic_terms : {0, 1, 3, 8}) {
        const int degree = 2 * (count - 1);
        const int md = degree + 3 * (std::max(1, cubic_terms) - 1);
        const int ld = md + 2 * (terms - 1);
        std::vector<Cplx<double>> h(degree + 1), moments(md + 1), linear(ld + 1);
        // Zero, downward/upward transition, fast winding, and stationarity.
        for (double a : {0., .3, double(ld) - .1, double(ld), double(ld) + .1, 10000.}) {
          analytic_linear<false>(a, ld, linear.data());
          for (double b : {0., .01, 1., -1., 40.}) {
            analytic_quadratic<false>(a, b, md, terms, moments.data());
            analytic_weights<false>(count, 2, terms, cubic_terms, a, b, .002, .1, h.data());
          }
        }
      }
}

} // namespace ri_kernels

int main() {
  using namespace ri_kernels;
  check_cells<float>(9, {2, 32, 8});
  check_cells<float>(7, {3, 1, 1});
  check_cells<double>(9, {2, 32, 8});
  check_cells<double>(7, {3, 1, 1});
  check_moments();
  std::cout << "Analytic bounds checks passed\n";
}
