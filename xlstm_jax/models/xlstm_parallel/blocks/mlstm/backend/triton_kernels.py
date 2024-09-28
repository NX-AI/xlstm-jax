from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.kernels import mlstm_chunkwise_max_triton

from .config import mLSTMBackend


@dataclass
class mLSTMBackendTritonConfig:
    autocast_dtype: jnp.dtype | str | None = None
    """Dtype to use for the kernel computation. If None, uses the query dtype."""
    chunk_size: int = 64
    """Chunk size for the kernel computation."""

    def assign_model_config_params(self, *args, **kwargs):
        pass


class mLSTMBackendTriton(mLSTMBackend):
    config_class = mLSTMBackendTritonConfig

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, i: jax.Array, f: jax.Array) -> jax.Array:
        """
        Forward pass of the mLSTM cell using triton kernels.

        Args:
            q: Query tensor of shape (B, NH, S, DH).
            k: Key tensor of shape (B, NH, S, DH).
            v: Value tensor of shape (B, NH, S, DH).
            i: Input gate tensor of shape (B, NH, S, 1) or (B, NH, S).
            f: Forget gate tensor of shape (B, NH, S, 1) or (B, NH, S).

        Returns:
            Output tensor of shape (B, NH, S, DH).
        """
        autocast_kernel_dtype = self.config.autocast_dtype
        if autocast_kernel_dtype is None:
            autocast_kernel_dtype = q.dtype
        if i.ndim == q.ndim:
            # Squeeze input and forget gate on last axis.
            i = i[..., 0]
            f = f[..., 0]
        return mlstm_chunkwise_max_triton(
            q, k, v, i, f, chunk_size=self.config.chunk_size, autocast_kernel_dtype=autocast_kernel_dtype
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
