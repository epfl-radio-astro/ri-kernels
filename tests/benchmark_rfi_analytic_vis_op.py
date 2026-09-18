"""Time compiled forward, JVP and VJP, signal-only and full, with synchronised GPU execution.

Run from a built checkout with, for example:
  python tests/benchmark_rfi_analytic_vis_op.py --antennas 256 512 --iterations 100
The quadrature comparison uses the same coarse data and frequency tables.
Compilation and warmup are excluded; production orbit data are still needed
for the scientific accuracy and end-to-end optimisation comparison.
"""
import argparse
import json
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from test_rfi_analytic_vis_op import _interp_inputs, make_inputs, make_baselines, reference
from ri_kernels.jax_api import RFIAnalyticVisOp, RFIInterpVisOp


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--antennas", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=6571)
    parser.add_argument("--sources", type=int, default=2)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--cells", type=int, default=16)
    parser.add_argument("--precision", choices=["float32", "float64"], default="float32")
    parser.add_argument("--analytic-only", action="store_true")
    # The pure-JAX analytic form the kernel is validated against. Worth timing
    # beside it: the kernel exists because expressing the moment recurrence in
    # JAX costs sequential array launches that a kernel does in registers.
    parser.add_argument("--with-reference", action="store_true")
    options = parser.parse_args()
    device = next((d for d in jax.devices() if d.platform != "cpu"), None)
    if device is None:
        parser.error("This benchmark requires a GPU")
    real = getattr(jnp, options.precision)
    complex_ = jnp.complex64 if real == jnp.float32 else jnp.complex128
    with jax.default_device(device):
        for na in options.antennas:
            shape = dict(n_ant=na, n_rfi=options.sources, n_freq=options.channels,
                         n_time=options.cells, n_int_f=2)
            a1, a2 = make_baselines(na, autocorr=False)
            operators = [("analytic", RFIAnalyticVisOp(na, a1, a2), make_inputs(real, complex_, **shape))]
            if not options.analytic_only:
                operators.append(("interp", RFIInterpVisOp(na, a1, a2),
                                  _interp_inputs(real, complex_, n_int_t=options.samples, **shape)))
            if options.with_reference:
                operators.append(("jax-analytic", None, make_inputs(real, complex_, **shape)))
            for name, op, args in operators:
                if op is None:
                    ref = partial(reference, a1=a1, a2=a2, segments=2, terms=6, cubic_terms=3)
                    fn = lambda a: ref(a, *args[1:])
                    fn2 = lambda a, p: ref(a, p, *args[2:])
                else:
                    fn = lambda a: op.eval(a, *args[1:])
                    fn2 = lambda a, p: op.eval(a, p, *args[2:])
                tangent = jnp.full_like(args[0], .3 + .7j)
                phase_tangent = jnp.full_like(args[1], .1)
                cotangent = jnp.full((len(a1), options.channels, options.cells), .7 - .2j, complex_)
                amp, phase = args[0], args[1]
                # The full pair carries the phase derivative as well; the
                # interp operator names them the same way.
                calls = {
                    "forward": jax.jit(fn),
                    "jvp": jax.jit(lambda a: jax.jvp(fn, (a,), (tangent,))[1]),
                    "vjp": jax.jit(lambda a: jax.vjp(fn, a)[1](cotangent)[0]),
                    "full-jvp": jax.jit(lambda a, p: jax.jvp(fn2, (a, p), (tangent, phase_tangent))[1]),
                    "full-vjp": jax.jit(lambda a, p: jax.vjp(fn2, a, p)[1](cotangent)),
                }
                for kind, call in calls.items():
                    inputs = (amp, phase) if kind.startswith("full") else (amp,)
                    # A case that cannot run -- the quadrature transpose needs
                    # more shared memory than a small card offers at this
                    # sampling -- is a result, not a reason to lose the sweep.
                    try:
                        for _ in range(3):
                            jax.block_until_ready(call(*inputs))
                        timings = []
                        for _ in range(options.iterations):
                            start = time.perf_counter()
                            jax.block_until_ready(call(*inputs))
                            timings.append(time.perf_counter() - start)
                    except Exception as exc:
                        print(json.dumps(dict(operator=name, derivative=kind, antennas=na,
                            failed=str(exc).split("[")[0].strip()[:160])), flush=True)
                        continue
                    print(json.dumps(dict(operator=name, derivative=kind, antennas=na,
                        device=device.device_kind, jax=jax.__version__, precision=options.precision,
                        sources=options.sources, channels=options.channels, cells=options.cells,
                        samples=options.samples if name == "interp" else None,
                        iterations=options.iterations, seconds_median=float(np.median(timings)),
                        seconds_min=min(timings))), flush=True)


if __name__ == "__main__":
    main()
