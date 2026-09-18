// The five analytic handlers per precision share the interpolation operand
// order, with a scalar duration in the time-offset slot: the forward, the
// signal JVP and transpose, and the full JVP and transpose that carry the
// phase tangent and cotangent as well. Options are static FFI attributes.
// Included once by each platform with its execution context and dispatcher.
#define RI_ANALYTIC_NAME_I(kind, platform, precision) calc_rfi_analytic##kind##_##platform##_##precision
#define RI_ANALYTIC_NAME_(kind, platform, precision) RI_ANALYTIC_NAME_I(kind, platform, precision)
#define RI_ANALYTIC_NAME(kind, precision) RI_ANALYTIC_NAME_(kind, RI_ANALYTIC_PLATFORM, precision)
#define RI_ANALYTIC_IMPL_I(name) name##_impl
#define RI_ANALYTIC_IMPL_(name) RI_ANALYTIC_IMPL_I(name)
#define RI_ANALYTIC_IMPL(kind, precision) RI_ANALYTIC_IMPL_(RI_ANALYTIC_NAME(kind, precision))

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(, f32)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f32_t amp,
    interp_real4_f32_t phase,
    interp_real4_f32_t delay,
    interp_real3_f32_t wf,
    interp_index_t sf,
    interp_real3_f32_t gt,
    interp_index_t st,
    interp_real1_f32_t dnu,
    ffi::BufferR0<ffi::F32> duration,
    interp_real1_f32_t freq,
    ffi::Result<ffi::BufferR3<ffi::C64>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<0, float, ffi::C64, ffi::F32>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(, f32)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(, f32),
    RI_ANALYTIC_IMPL(, f32), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<interp_real1_f32_t>()
        .Ret<ffi::BufferR3<ffi::C64>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_jvp, f32)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f32_t amp,
    interp_amp_f32_t dot,
    interp_real4_f32_t phase,
    interp_real4_f32_t delay,
    interp_real3_f32_t wf,
    interp_index_t sf,
    interp_real3_f32_t gt,
    interp_index_t st,
    interp_real1_f32_t dnu,
    ffi::BufferR0<ffi::F32> duration,
    interp_real1_f32_t freq,
    ffi::Result<ffi::BufferR3<ffi::C64>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<1, float, ffi::C64, ffi::F32>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, dot, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_jvp, f32)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_jvp, f32),
    RI_ANALYTIC_IMPL(_jvp, f32), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<interp_real1_f32_t>()
        .Ret<ffi::BufferR3<ffi::C64>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_transpose, f32)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f32_t amp,
    interp_real4_f32_t phase,
    interp_real4_f32_t delay,
    interp_real3_f32_t wf,
    interp_index_t sf,
    interp_real3_f32_t gt,
    interp_index_t st,
    interp_real1_f32_t dnu,
    ffi::BufferR0<ffi::F32> duration,
    interp_real1_f32_t freq,
    ffi::BufferR3<ffi::C64> cot,
    ffi::Result<ffi::BufferR4<ffi::C64>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<2, float, ffi::C64, ffi::F32>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, cot, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_transpose, f32)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_transpose, f32),
    RI_ANALYTIC_IMPL(_transpose, f32), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<ffi::BufferR4<ffi::C64>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_full_jvp, f32)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f32_t amp,
    interp_amp_f32_t dot,
    interp_real4_f32_t phase,
    interp_real4_f32_t phase_dot,
    interp_real4_f32_t delay,
    interp_real3_f32_t wf,
    interp_index_t sf,
    interp_real3_f32_t gt,
    interp_index_t st,
    interp_real1_f32_t dnu,
    ffi::BufferR0<ffi::F32> duration,
    interp_real1_f32_t freq,
    ffi::Result<ffi::BufferR3<ffi::C64>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<3, float, ffi::C64, ffi::F32>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_full_jvp, f32)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_full_jvp, f32),
    RI_ANALYTIC_IMPL(_full_jvp, f32), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<interp_real1_f32_t>()
        .Ret<ffi::BufferR3<ffi::C64>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_full_transpose, f32)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f32_t amp,
    interp_real4_f32_t phase,
    interp_real4_f32_t delay,
    interp_real3_f32_t wf,
    interp_index_t sf,
    interp_real3_f32_t gt,
    interp_index_t st,
    interp_real1_f32_t dnu,
    ffi::BufferR0<ffi::F32> duration,
    interp_real1_f32_t freq,
    ffi::BufferR3<ffi::C64> cot,
    ffi::Result<ffi::BufferR4<ffi::C64>> out,
    ffi::Result<ffi::BufferR4<ffi::F32>> phase_bar,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<4, float, ffi::C64, ffi::F32>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, cot, out, &phase_bar, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_full_transpose, f32)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_full_transpose, f32),
    RI_ANALYTIC_IMPL(_full_transpose, f32), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real4_f32_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f32_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<interp_real1_f32_t>()
        .Arg<ffi::BufferR3<ffi::C64>>()
        .Ret<ffi::BufferR4<ffi::C64>>()
        .Ret<ffi::BufferR4<ffi::F32>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(, f64)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f64_t amp,
    interp_real4_f64_t phase,
    interp_real4_f64_t delay,
    interp_real3_f64_t wf,
    interp_index_t sf,
    interp_real3_f64_t gt,
    interp_index_t st,
    interp_real1_f64_t dnu,
    ffi::BufferR0<ffi::F64> duration,
    interp_real1_f64_t freq,
    ffi::Result<ffi::BufferR3<ffi::C128>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<0, double, ffi::C128, ffi::F64>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(, f64)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(, f64),
    RI_ANALYTIC_IMPL(, f64), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<interp_real1_f64_t>()
        .Ret<ffi::BufferR3<ffi::C128>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_jvp, f64)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f64_t amp,
    interp_amp_f64_t dot,
    interp_real4_f64_t phase,
    interp_real4_f64_t delay,
    interp_real3_f64_t wf,
    interp_index_t sf,
    interp_real3_f64_t gt,
    interp_index_t st,
    interp_real1_f64_t dnu,
    ffi::BufferR0<ffi::F64> duration,
    interp_real1_f64_t freq,
    ffi::Result<ffi::BufferR3<ffi::C128>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<1, double, ffi::C128, ffi::F64>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, dot, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_jvp, f64)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_jvp, f64),
    RI_ANALYTIC_IMPL(_jvp, f64), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<interp_real1_f64_t>()
        .Ret<ffi::BufferR3<ffi::C128>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_transpose, f64)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f64_t amp,
    interp_real4_f64_t phase,
    interp_real4_f64_t delay,
    interp_real3_f64_t wf,
    interp_index_t sf,
    interp_real3_f64_t gt,
    interp_index_t st,
    interp_real1_f64_t dnu,
    ffi::BufferR0<ffi::F64> duration,
    interp_real1_f64_t freq,
    ffi::BufferR3<ffi::C128> cot,
    ffi::Result<ffi::BufferR4<ffi::C128>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<2, double, ffi::C128, ffi::F64>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, cot, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_transpose, f64)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_transpose, f64),
    RI_ANALYTIC_IMPL(_transpose, f64), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<ffi::BufferR4<ffi::C128>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_full_jvp, f64)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f64_t amp,
    interp_amp_f64_t dot,
    interp_real4_f64_t phase,
    interp_real4_f64_t phase_dot,
    interp_real4_f64_t delay,
    interp_real3_f64_t wf,
    interp_index_t sf,
    interp_real3_f64_t gt,
    interp_index_t st,
    interp_real1_f64_t dnu,
    ffi::BufferR0<ffi::F64> duration,
    interp_real1_f64_t freq,
    ffi::Result<ffi::BufferR3<ffi::C128>> out,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<3, double, ffi::C128, ffi::F64>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, dot, phase, phase_dot, delay,
      wf, sf, gt, st, dnu, duration, freq, *out, out, nullptr, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_full_jvp, f64)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_full_jvp, f64),
    RI_ANALYTIC_IMPL(_full_jvp, f64), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<interp_real1_f64_t>()
        .Ret<ffi::BufferR3<ffi::C128>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));

RI_ANALYTIC_RETURN RI_ANALYTIC_IMPL(_full_transpose, f64)(
    RI_ANALYTIC_CONTEXT,
    interp_index_t a1,
    interp_index_t a1_sorter,
    interp_index_t a1_start,
    interp_index_t a2,
    interp_index_t a2_sorter,
    interp_index_t a2_start,
    ffi::BufferR2<ffi::S32> pair,
    ffi::BufferR2<ffi::S32> tiles,
    interp_amp_f64_t amp,
    interp_real4_f64_t phase,
    interp_real4_f64_t delay,
    interp_real3_f64_t wf,
    interp_index_t sf,
    interp_real3_f64_t gt,
    interp_index_t st,
    interp_real1_f64_t dnu,
    ffi::BufferR0<ffi::F64> duration,
    interp_real1_f64_t freq,
    ffi::BufferR3<ffi::C128> cot,
    ffi::Result<ffi::BufferR4<ffi::C128>> out,
    ffi::Result<ffi::BufferR4<ffi::F64>> phase_bar,
    std::int64_t segments, std::int64_t terms, std::int64_t cubic_terms) {
  return RI_ANALYTIC_DISPATCH<4, double, ffi::C128, ffi::F64>(
      RI_ANALYTIC_CONTEXT_PASS, a1, a2, pair, tiles, amp, amp, phase, phase, delay,
      wf, sf, gt, st, dnu, duration, freq, cot, out, &phase_bar, segments, terms, cubic_terms);
}
extern "C" RI_KERNELS_API XLA_FFI_Error *RI_ANALYTIC_NAME(_full_transpose, f64)(XLA_FFI_CallFrame *);
XLA_FFI_DEFINE_HANDLER_SYMBOL(RI_ANALYTIC_NAME(_full_transpose, f64),
    RI_ANALYTIC_IMPL(_full_transpose, f64), ffi::Ffi::Bind() RI_ANALYTIC_CONTEXT_BIND
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<interp_index_t>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<interp_amp_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real4_f64_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real3_f64_t>()
        .Arg<interp_index_t>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR0<ffi::F64>>()
        .Arg<interp_real1_f64_t>()
        .Arg<ffi::BufferR3<ffi::C128>>()
        .Ret<ffi::BufferR4<ffi::C128>>()
        .Ret<ffi::BufferR4<ffi::F64>>()
        .Attr<std::int64_t>("segments")
        .Attr<std::int64_t>("terms")
        .Attr<std::int64_t>("cubic_terms"));
