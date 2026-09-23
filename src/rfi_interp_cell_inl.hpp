// The per-cell building blocks of the data-grid RFI kernels, vectorised with
// Highway. This file must be included inside a HWY_NAMESPACE. It must not have
// an include guard.
//
// Every loop here runs over the fine samples of one data cell, enumerated
// time-major as k = v * n_int_f + u, which is the only axis long enough to
// vectorise and the only one whose values are contiguous. The quantities that
// depend on the cell alone (the interpolation weights, the channel offsets,
// the powers of the time offsets) are built once per cell into CellTables; the
// quantities that depend on an antenna and a source are broadcast scalars that
// multiply them. The sample arrays are kept split into real and imaginary
// parts, so a load is a load rather than a load and a shuffle, and padded to a
// whole number of vectors with zeros, so no loop needs a scalar tail: a padded
// lane contributes zero to every product and every sum taken here.

// Real and imaginary parts of one cell's samples for every antenna, each
// antenna's run padded to a whole number of vectors.
template <typename T> struct SampleBuf {
  std::vector<T> re, im;
  std::int64_t stride = 0;  // padded samples per antenna

  void resize(std::int64_t n_ant, std::int64_t n_s_padded) {
    stride = n_s_padded;
    re.assign(std::size_t(n_ant) * n_s_padded, T(0));
    im.assign(std::size_t(n_ant) * n_s_padded, T(0));
  }
  T *re_at(std::int64_t a) { return re.data() + a * stride; }
  T *im_at(std::int64_t a) { return im.data() + a * stride; }
  const T *re_at(std::int64_t a) const { return re.data() + a * stride; }
  const T *im_at(std::int64_t a) const { return im.data() + a * stride; }
};

// Per-cell tables, rebuilt whenever the cell changes.
template <typename T> struct CellTables {
  std::int64_t n_s = 0, n_s_padded = 0, n_stencil = 0, n_path = 0;
  std::vector<T> weight;   // (n_stencil, n_s_padded): w_freq[kf, u] w_time[kt, v]
  std::vector<T> dt_s;     // (n_s_padded,): dt[v]
  std::vector<T> dnu_s;    // (n_s_padded,): dnu[u]
  std::vector<T> nu_s;     // (n_s_padded,): freqs[f] + dnu[u]
  std::vector<T> dcoef;    // (n_path, n_s_padded): d phi / d delay[j]

  // The tables of cell (f, t); `with_dcoef` only for the full transpose.
  template <typename INT_T>
  void build(Tensor3D<const T *, INT_T> w_freq, Tensor3D<const T *, INT_T> w_time,
             Tensor1D<const T *, INT_T> dnu, Tensor1D<const T *, INT_T> dt,
             T freq_f, INT_T f, INT_T t, INT_T n_path_in, std::int64_t lanes,
             bool with_dcoef) {
    const std::int64_t n_sf = w_freq.shape[1], n_int_f = w_freq.shape[2];
    const std::int64_t n_st = w_time.shape[1], n_int_t = w_time.shape[2];
    n_s = n_int_f * n_int_t;
    n_s_padded = (n_s + lanes - 1) / lanes * lanes;
    n_stencil = n_sf * n_st;
    n_path = n_path_in;
    weight.assign(std::size_t(n_stencil) * n_s_padded, T(0));
    dt_s.assign(n_s_padded, T(0));
    dnu_s.assign(n_s_padded, T(0));
    nu_s.assign(n_s_padded, T(0));
    for (std::int64_t v = 0; v < n_int_t; ++v) {
      for (std::int64_t u = 0; u < n_int_f; ++u) {
        const std::int64_t k = v * n_int_f + u;
        dt_s[k] = dt(v);
        dnu_s[k] = dnu(u);
        nu_s[k] = freq_f + dnu(u);
        for (std::int64_t kf = 0; kf < n_sf; ++kf)
          for (std::int64_t kt = 0; kt < n_st; ++kt)
            weight[(kf * n_st + kt) * n_s_padded + k] = w_freq(f, kf, u) * w_time(t, kt, v);
      }
    }
    if (!with_dcoef) {
      dcoef.clear();
      return;
    }
    dcoef.assign(std::size_t(n_path) * n_s_padded, T(0));
    for (std::int64_t k = 0; k < n_s; ++k)
      for (std::int64_t j = 0; j < n_path; ++j)
        dcoef[j * n_s_padded + k] = delay_phase_coeff<T, std::int64_t>(j, freq_f, dnu_s[k], dt_s[k]);
  }
};

// The phase phi of antenna `a`, source `r` at one vector of this cell's
// samples, by the formula of phase_value() with the same Horner order.
template <class D, typename T, typename INT_T>
HWY_INLINE HWY_ATTR hn::Vec<D> CellPhi(D d, const CellTables<T> &c,
                                       Tensor4D<const T *, INT_T> phase,
                                       Tensor4D<const T *, INT_T> delay,
                                       INT_T a, INT_T r, INT_T f, INT_T t, std::int64_t k) {
  const auto dt_v = hn::LoadU(d, c.dt_s.data() + k);
  // Horner from the highest derivative down, each term divided by j!.
  auto acc = hn::Zero(d);
  T inv_factorial = 1;
  for (INT_T j = 2; j < c.n_path; ++j) inv_factorial /= T(j);
  for (INT_T j = INT_T(c.n_path) - 1; j >= 1; --j) {
    acc = hn::MulAdd(acc, dt_v, hn::Set(d, delay(a, r, t, j) * inv_factorial));
    inv_factorial *= T(j);
  }
  const auto d_tau = hn::Mul(acc, dt_v);
  const auto nu = hn::LoadU(d, c.nu_s.data() + k);
  const auto dnu = hn::LoadU(d, c.dnu_s.data() + k);
  return hn::MulAdd(hn::Set(d, two_pi_c<T>()),
                    hn::MulAdd(nu, d_tau, hn::Mul(dnu, hn::Set(d, delay(a, r, t, 0)))),
                    hn::Set(d, phase(a, r, f, t)));
}

// phi over the whole cell. Linear in (phase, delay), so called on their
// tangents it is the phase's tangent dphi.
template <class D, typename T, typename INT_T>
HWY_INLINE HWY_ATTR void CellPhaseValue(D d, const CellTables<T> &c,
                                        Tensor4D<const T *, INT_T> phase,
                                        Tensor4D<const T *, INT_T> delay,
                                        INT_T a, INT_T r, INT_T f, INT_T t,
                                        T *HWY_RESTRICT phi_out) {
  const std::int64_t lanes = hn::Lanes(d);
  for (std::int64_t k = 0; k < c.n_s_padded; k += lanes)
    hn::StoreU(CellPhi(d, c, phase, delay, a, r, f, t, k), d, phi_out + k);
}

// exp(i phi) over the whole cell, into (e_re, e_im).
template <class D, typename T, typename INT_T>
HWY_INLINE HWY_ATTR void CellPhaseFactor(D d, const CellTables<T> &c,
                                         Tensor4D<const T *, INT_T> phase,
                                         Tensor4D<const T *, INT_T> delay,
                                         INT_T a, INT_T r, INT_T f, INT_T t,
                                         T *HWY_RESTRICT e_re, T *HWY_RESTRICT e_im) {
  const std::int64_t lanes = hn::Lanes(d);
  for (std::int64_t k = 0; k < c.n_s_padded; k += lanes) {
    const auto phi = CellPhi(d, c, phase, delay, a, r, f, t, k);
    hn::StoreU(hn::Cos(d, phi), d, e_re + k);
    hn::StoreU(hn::Sin(d, phi), d, e_im + k);
  }
}

// The interpolated data-grid value A of antenna `a`, source `r` on this cell:
// the stencil's coefficients are scalars and the weights are per sample, so
// each stencil entry is one broadcast multiplied into the running sample.
template <class D, typename T, typename INT_T>
HWY_INLINE HWY_ATTR void CellInterpAmp(D d, const CellTables<T> &c,
                                       Tensor4D<const Cplx<T> *, INT_T> amp,
                                       INT_T sf, INT_T st, INT_T n_st, INT_T a, INT_T r,
                                       T *HWY_RESTRICT a_re, T *HWY_RESTRICT a_im) {
  const std::int64_t lanes = hn::Lanes(d);
  for (std::int64_t k = 0; k < c.n_s_padded; k += lanes) {
    auto re = hn::Zero(d);
    auto im = hn::Zero(d);
    for (std::int64_t kl = 0; kl < c.n_stencil; ++kl) {
      const Cplx<T> value = amp(a, r, sf + INT_T(kl / n_st), st + INT_T(kl % n_st));
      const auto w = hn::LoadU(d, c.weight.data() + kl * c.n_s_padded + k);
      re = hn::MulAdd(w, hn::Set(d, value.re), re);
      im = hn::MulAdd(w, hn::Set(d, value.im), im);
    }
    hn::StoreU(re, d, a_re + k);
    hn::StoreU(im, d, a_im + k);
  }
}

// S = A exp(i phi) for every antenna of the cell, and the phase factors beside
// them when `e` is given (the transpose turns its cotangents by them).
template <class D, typename T, typename INT_T>
HWY_INLINE HWY_ATTR void CellSamples(D d, const CellTables<T> &c,
                                     Tensor4D<const Cplx<T> *, INT_T> amp,
                                     Tensor4D<const T *, INT_T> phase,
                                     Tensor4D<const T *, INT_T> delay,
                                     INT_T sf, INT_T st, INT_T n_st, INT_T n_ant,
                                     INT_T r, INT_T f, INT_T t,
                                     SampleBuf<T> &S, SampleBuf<T> *e) {
  const std::int64_t lanes = hn::Lanes(d);
  std::vector<T> e_re(c.n_s_padded), e_im(c.n_s_padded);
  for (INT_T a = 0; a < n_ant; ++a) {
    T *s_re = S.re_at(a), *s_im = S.im_at(a);
    T *er = e ? e->re_at(a) : e_re.data();
    T *ei = e ? e->im_at(a) : e_im.data();
    CellPhaseFactor(d, c, phase, delay, a, r, f, t, er, ei);
    CellInterpAmp(d, c, amp, sf, st, n_st, a, r, s_re, s_im);
    for (std::int64_t k = 0; k < c.n_s_padded; k += lanes) {
      const auto ar = hn::LoadU(d, s_re + k), ai = hn::LoadU(d, s_im + k);
      const auto br = hn::LoadU(d, er + k), bi = hn::LoadU(d, ei + k);
      hn::StoreU(hn::NegMulAdd(ai, bi, hn::Mul(ar, br)), d, s_re + k);
      hn::StoreU(hn::MulAdd(ar, bi, hn::Mul(ai, br)), d, s_im + k);
    }
  }
}

// The cotangent scatter of one baseline onto its two antennas' samples:
// G1 += w conj(S2) and G2 += conj(w) conj(S1), over one cell's samples.
template <class D, typename T>
HWY_INLINE HWY_ATTR void CellScatterCotangent(D d, std::int64_t n_s_padded, Cplx<T> w,
                                              const T *HWY_RESTRICT s1_re, const T *HWY_RESTRICT s1_im,
                                              const T *HWY_RESTRICT s2_re, const T *HWY_RESTRICT s2_im,
                                              T *HWY_RESTRICT g1_re, T *HWY_RESTRICT g1_im,
                                              T *HWY_RESTRICT g2_re, T *HWY_RESTRICT g2_im) {
  const std::int64_t lanes = hn::Lanes(d);
  const auto wr = hn::Set(d, w.re), wi = hn::Set(d, w.im);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) {
    const auto ar = hn::LoadU(d, s1_re + k), ai = hn::LoadU(d, s1_im + k);
    const auto br = hn::LoadU(d, s2_re + k), bi = hn::LoadU(d, s2_im + k);
    hn::StoreU(hn::MulAdd(wi, bi, hn::MulAdd(wr, br, hn::LoadU(d, g1_re + k))), d, g1_re + k);
    hn::StoreU(hn::NegMulAdd(wr, bi, hn::MulAdd(wi, br, hn::LoadU(d, g1_im + k))), d, g1_im + k);
    hn::StoreU(hn::NegMulAdd(wi, ai, hn::MulAdd(wr, ar, hn::LoadU(d, g2_re + k))), d, g2_re + k);
    hn::StoreU(hn::NegMulAdd(wr, ai, hn::NegMulAdd(wi, ar, hn::LoadU(d, g2_im + k))), d, g2_im + k);
  }
}

// x[k] <- y[k] x[k] over one cell's samples, both complex.
template <class D, typename T>
HWY_INLINE HWY_ATTR void CellMulInPlace(D d, std::int64_t n_s_padded,
                                        const T *HWY_RESTRICT y_re, const T *HWY_RESTRICT y_im,
                                        T *HWY_RESTRICT x_re, T *HWY_RESTRICT x_im) {
  const std::int64_t lanes = hn::Lanes(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) {
    const auto ar = hn::LoadU(d, x_re + k), ai = hn::LoadU(d, x_im + k);
    const auto br = hn::LoadU(d, y_re + k), bi = hn::LoadU(d, y_im + k);
    hn::StoreU(hn::NegMulAdd(ai, bi, hn::Mul(ar, br)), d, x_re + k);
    hn::StoreU(hn::MulAdd(ar, bi, hn::Mul(ai, br)), d, x_im + k);
  }
}

// -Im(x[k] y[k]) per sample, the cotangent of the phase there.
template <class D, typename T>
HWY_INLINE HWY_ATTR void CellPhaseCotangent(D d, std::int64_t n_s_padded,
                                            const T *HWY_RESTRICT x_re, const T *HWY_RESTRICT x_im,
                                            const T *HWY_RESTRICT y_re, const T *HWY_RESTRICT y_im,
                                            T *HWY_RESTRICT out) {
  const std::int64_t lanes = hn::Lanes(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) {
    const auto ar = hn::LoadU(d, x_re + k), ai = hn::LoadU(d, x_im + k);
    const auto br = hn::LoadU(d, y_re + k), bi = hn::LoadU(d, y_im + k);
    hn::StoreU(hn::Neg(hn::MulAdd(ar, bi, hn::Mul(ai, br))), d, out + k);
  }
}

// sum_k w[k] x[k], a real weight against a complex value.
template <class D, typename T>
HWY_INLINE HWY_ATTR Cplx<T> CellWeightedSum(D d, std::int64_t n_s_padded,
                                            const T *HWY_RESTRICT w,
                                            const T *HWY_RESTRICT x_re, const T *HWY_RESTRICT x_im) {
  const std::int64_t lanes = hn::Lanes(d);
  auto re = hn::Zero(d);
  auto im = hn::Zero(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) {
    const auto wv = hn::LoadU(d, w + k);
    re = hn::MulAdd(wv, hn::LoadU(d, x_re + k), re);
    im = hn::MulAdd(wv, hn::LoadU(d, x_im + k), im);
  }
  return Cplx<T>{hn::ReduceSum(d, re), hn::ReduceSum(d, im)};
}

// sum_k w[k] x[k], both real.
template <class D, typename T>
HWY_INLINE HWY_ATTR T CellWeightedSumReal(D d, std::int64_t n_s_padded,
                                          const T *HWY_RESTRICT w, const T *HWY_RESTRICT x) {
  const std::int64_t lanes = hn::Lanes(d);
  auto acc = hn::Zero(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes)
    acc = hn::MulAdd(hn::LoadU(d, w + k), hn::LoadU(d, x + k), acc);
  return hn::ReduceSum(d, acc);
}

// sum_k x[k], real.
template <class D, typename T>
HWY_INLINE HWY_ATTR T CellSumReal(D d, std::int64_t n_s_padded, const T *HWY_RESTRICT x) {
  const std::int64_t lanes = hn::Lanes(d);
  auto acc = hn::Zero(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) acc = hn::Add(acc, hn::LoadU(d, x + k));
  return hn::ReduceSum(d, acc);
}

// sum_k x[k] conj(y[k]) over one cell's samples.
template <class D, typename T>
HWY_INLINE HWY_ATTR Cplx<T> CellDotConj(D d, std::int64_t n_s_padded,
                                        const T *HWY_RESTRICT x_re, const T *HWY_RESTRICT x_im,
                                        const T *HWY_RESTRICT y_re, const T *HWY_RESTRICT y_im) {
  const std::int64_t lanes = hn::Lanes(d);
  auto re = hn::Zero(d);
  auto im = hn::Zero(d);
  for (std::int64_t k = 0; k < n_s_padded; k += lanes) {
    const auto xr = hn::LoadU(d, x_re + k), xi = hn::LoadU(d, x_im + k);
    const auto yr = hn::LoadU(d, y_re + k), yi = hn::LoadU(d, y_im + k);
    re = hn::MulAdd(xi, yi, hn::MulAdd(xr, yr, re));
    im = hn::NegMulAdd(xr, yi, hn::MulAdd(xi, yr, im));
  }
  return Cplx<T>{hn::ReduceSum(d, re), hn::ReduceSum(d, im)};
}
