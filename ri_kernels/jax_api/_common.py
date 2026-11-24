"""Pure-Python helpers shared across compiled and reference JAX operators."""

import jax.numpy as jnp


def _dtype_suffix(amp_dtype, phase_dtype):
    """Pick the FFI symbol suffix for a (rfi_amp_fine, rfi_phase) dtype pair.

    The kernels accept matched precision: complex64+float32 or complex128+float64.
    Mixed precision is rejected at the Python boundary.
    """
    amp = jnp.dtype(amp_dtype)
    phase = jnp.dtype(phase_dtype)
    if amp == jnp.complex64 and phase == jnp.float32:
        return "f32"
    if amp == jnp.complex128 and phase == jnp.float64:
        return "f64"
    raise TypeError(
        f"RFI FFI kernels require matched precision: (complex64, float32) "
        f"or (complex128, float64). Got (amp={amp}, phase={phase})."
    )
