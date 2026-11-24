"""Pure JAX reference for :class:`RFIVisOp`.

The same visibility as the compiled primitives,
``mean_fine sum_rfi A[a1] conj(A[a2]) exp(i (phase[a1] - phase[a2]))``,
evaluated on gathered baselines. JAX differentiates it directly, so it needs
no FFI library; it holds every baseline's fine samples at once and is meant
for checking the kernels and for small problems.
"""

import jax.numpy as jnp

from ._common import _dtype_suffix


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
        Evaluate the RFI visibility for given RFI amplitudes and phases.

        Args:
            rfi_amp_fine: Fine-grained RFI amplitude array with shape
                         (n_ant, n_freq, n_time, n_rfi, n_int_freq, n_int_time).
            rfi_phase: RFI phase array with shape matching rfi_amp_fine.

        Returns:
            Array of RFI visibilities with shape (n_baselines, n_freq, n_time),
            as :meth:`RFIVisOp.eval`.
        """
        # The same matched-precision rule as the kernels.
        _dtype_suffix(rfi_amp_fine.dtype, rfi_phase.dtype)
        if rfi_amp_fine.ndim != 6:
            raise ValueError(
                "Expected rfi_amp_fine of shape (n_ant, n_freq, n_time, n_rfi, "
                f"n_int_freq, n_int_time); got {rfi_amp_fine.shape}"
            )
        if rfi_phase.shape != rfi_amp_fine.shape:
            raise ValueError(
                f"Expected rfi_phase of shape {rfi_amp_fine.shape}; got {rfi_phase.shape}"
            )
        vis_fine = jnp.sum(
            rfi_amp_fine[self.a1]
            * jnp.conjugate(rfi_amp_fine[self.a2])
            * jnp.exp(1j * (rfi_phase[self.a1] - rfi_phase[self.a2])),
            axis=3,
        )
        # (n_bl, n_freq, n_time, n_int_freq, n_int_time) -> (n_bl, n_freq, n_time)
        return jnp.mean(vis_fine, axis=(3, 4))
