#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass

import jax
from mlstm_kernels.jax import get_mlstm_kernel

from .config import mLSTMBackend


@dataclass
class mLSTMBackendTritonConfig:
    autocast_dtype: str | None = None
    """Dtype to use for the kernel computation. If None, uses the query dtype."""
    chunk_size: int = 64
    """Chunk size for the kernel computation."""
    reduce_slicing: bool = True
    """Whether to reduce slicing operations before the kernel computation.
    Speeds up computation during training, but may limit initial states and
    forwarding states during inference."""
    backend_name: str = "chunkwise--triton_xl_chunk"
    """Backend name for the kernel type used"""
    eps: float = 1e-6
    """Epsilon value used in the kernel"""
    norm_val: float = 1.0
    """Normalizer upper bound value - max(norm_val e^-m, |n q|)"""
    stabilize_correctly: bool = True
    """Whether to stabilize correctly, i.e. scale norm_val with the maximizer state - see above"""

    def assign_model_config_params(self, model_config):
        pass


class mLSTMBackendTriton(mLSTMBackend):
    config_class = mLSTMBackendTritonConfig

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        i: jax.Array,
        f: jax.Array,
        c_initial: jax.Array | None = None,
        n_initial: jax.Array | None = None,
        m_initial: jax.Array | None = None,
        return_last_states: bool = False,
    ) -> jax.Array:
        """
        Forward pass of the mLSTM cell using triton kernels.

        Args:
            q: Query tensor of shape (B, NH, S, DHQK).
            k: Key tensor of shape (B, NH, S, DHQK).
            v: Value tensor of shape (B, NH, S, DHV).
            i: Input gate tensor of shape (B, NH, S, 1) or (B, NH, S).
            f: Forget gate tensor of shape (B, NH, S, 1) or (B, NH, S).
            c_initial: Initial cell state tensor of shape (B, NH, DHQK, DHV).
            n_initial: Initial norm state tensor of shape (B, NH, DHQK).
            m_initial: Initial maximizer state tensor of shape (B, NH).
            return_last_states: Whether to return the last states.

        Returns:
            Output tensor of shape (B, NH, S, DH).
        """
        autocast_kernel_dtype = self.config.autocast_dtype
        if autocast_kernel_dtype is None:
            autocast_kernel_dtype = q.dtype
        if isinstance(autocast_kernel_dtype, str):
            autocast_kernel_dtype = getattr(jax.numpy, autocast_kernel_dtype)
        if i.ndim == q.ndim:
            # Squeeze input and forget gate on last axis.
            i = i[..., 0]
            f = f[..., 0]
        kernel_fn = get_mlstm_kernel(self.config.backend_name)
        return kernel_fn(
            q,
            k,
            v,
            i,
            f,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
            return_last_states=return_last_states,
            chunk_size=self.config.chunk_size,
            autocast_kernel_dtype=autocast_kernel_dtype,
        )

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        Triton kernels already handle the head dimension, hence not to be vmaped over.

        Returns:
            bool: False
        """
        return False
