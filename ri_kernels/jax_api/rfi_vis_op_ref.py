"""Pure JAX reference for :class:`RFIVisOp`.

The same polarised visibility as the compiled primitives,
``sum_rfi mean_fine sum_c A[a1, ..., i, c] conj(A[a2, ..., j, c]) exp(i (phase[a1] - phase[a2]))``,
written as one contraction over gathered baselines. JAX differentiates it
directly, so it needs no FFI library; it holds every baseline's fine samples
at once and is meant for checking the kernels and for small problems.
"""

import jax
import jax.numpy as jnp

from .rfi_vis_op import _validate


class RFIVisOpRef:
    """Pure JAX counterpart of :class:`RFIVisOp`, with the same interface."""

    def __init__(self, n_ant, a1, a2):
        """
        Args:
            n_ant: Number of antennas in the array.
            a1: Array of first antenna indices for each baseline (shape: [n_baselines]).
            a2: Array of second antenna indices for each baseline (shape: [n_baselines]).
        """
        self.n_ant = n_ant
        self.a1 = jnp.asarray(a1)
        self.a2 = jnp.asarray(a2)

    def eval(self, rfi_amp_fine, rfi_phase):
        """
        Evaluate the polarised RFI visibility for given RFI amplitudes and phases.

        Args:
            rfi_amp_fine: Fine-grained RFI signal factor with shape
                         (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time, P, 2).
            rfi_phase: RFI phase array with shape
                       (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time).

        Returns:
            Array of RFI visibilities with shape (n_baselines, n_freq, n_time, P, P),
            as :meth:`RFIVisOp.eval`.
        """
        _validate(rfi_amp_fine, rfi_phase)
        rotation = jnp.exp(1j * (rfi_phase[self.a1] - rfi_phase[self.a2]))
        vis_fine = jnp.einsum(
            "bftrxyic,bftrxyjc,bftrxy->bftxyij",
            rfi_amp_fine[self.a1],
            jnp.conjugate(rfi_amp_fine[self.a2]),
            rotation.astype(rfi_amp_fine.dtype),
            # Ampere and later GPUs would otherwise contract f32 in TF32.
            precision=jax.lax.Precision.HIGHEST,
        )
        # (n_bl, n_freq, n_time, n_int_freq, n_int_time, P, P) -> (n_bl, n_freq, n_time, P, P)
        return jnp.mean(vis_fine, axis=(3, 4))
