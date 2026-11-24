"""Time compiled RFIVisOp forward, JVP, and VJP on CPU or GPU.

Run with an explicit library directory or installed package, for example:
  RI_KERNELS_LIB_DIR=/tmp/ri_kernels_build python3 tests/benchmark_rfi_vis_op.py
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from ri_kernels.jax_api import RFIVisOp  # noqa: E402


def _make_baselines(n_ant, autocorr=False):
    offset = 0 if autocorr else 1
    pairs = np.asarray(
        [(i, j) for i in range(n_ant) for j in range(i + offset, n_ant)],
        dtype=np.int32,
    )
    return jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--antennas", type=int, default=32)
    parser.add_argument("--freqs", type=int, default=4)
    parser.add_argument("--times", type=int, default=8)
    parser.add_argument("--rfi", type=int, default=4)
    parser.add_argument("--int-freq", type=int, default=8)
    parser.add_argument("--int-time", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--precisions", nargs="+", choices=["float32", "float64"],
                        default=["float32", "float64"])
    args = parser.parse_args()

    if args.device == "cpu":
        dev = jax.devices("cpu")[0]
    else:
        dev = next((d for d in jax.devices() if d.platform != "cpu"), None)
        if dev is None:
            parser.error("No GPU device available")

    rng = np.random.default_rng(0)
    dims = (args.antennas, args.freqs, args.times, args.rfi, args.int_freq, args.int_time)

    with jax.default_device(dev):
        a1, a2 = _make_baselines(args.antennas)
        op = RFIVisOp(args.antennas, a1, a2)
        for prec in args.precisions:
            real = getattr(jnp, prec)
            cplx = jnp.complex64 if real == jnp.float32 else jnp.complex128
            amp = jnp.asarray(rng.normal(size=dims) + 1j * rng.normal(size=dims), cplx)
            phase = jnp.asarray(rng.uniform(-np.pi, np.pi, size=dims), real)
            amp_dot = jnp.asarray(rng.normal(size=dims) + 1j * rng.normal(size=dims), cplx)
            phase_dot = jnp.asarray(rng.normal(size=dims), real)
            cot_dims = (len(a1), args.freqs, args.times)
            cot = jnp.asarray(rng.normal(size=cot_dims) + 1j * rng.normal(size=cot_dims), cplx)

            calls = {
                "fwd": jax.jit(op.eval),
                "jvp": jax.jit(lambda a, p: jax.jvp(op.eval, (a, p), (amp_dot, phase_dot))[1]),
                "vjp": jax.jit(lambda a, p: jax.vjp(op.eval, a, p)[1](cot)),
            }
            for kind, fn in calls.items():
                for _ in range(3):
                    jax.block_until_ready(fn(amp, phase))
                timings = []
                for _ in range(args.iterations):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(amp, phase))
                    timings.append(time.perf_counter() - t0)
                print(json.dumps({
                    "op": "RFIVisOp",
                    "pass": kind,
                    "precision": prec,
                    "device": dev.platform,
                    "n_ant": args.antennas,
                    "n_red": args.rfi * args.int_freq * args.int_time,
                    "iterations": args.iterations,
                    "ms_median": float(np.median(timings)) * 1e3,
                    "ms_min": min(timings) * 1e3,
                }), flush=True)


if __name__ == "__main__":
    main()
