from dataclasses import dataclass
from typing import Literal

import jax
from flax import linen as nn

from xlstm_jax.kernels import mlstm_chunkwise_max_triton, mlstm_chunkwise_triton_stablef

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
    backend_name: Literal["max_triton", "triton_stablef"] = "max_triton"
    """Backend name for the kernel type used"""
    eps: float = 1e-6
    """Epsilon value used in the kernel"""
    norm_val: float = 1.0
    """Normalizer upper bound value - max(norm_val e^-m, |n q|)"""
    stabilize_correctly: bool = True
    """Whether to stabilize correctly, i.e. scale norm_val with the maximizer state - see above"""

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
        if isinstance(autocast_kernel_dtype, str):
            autocast_kernel_dtype = getattr(jax.numpy, autocast_kernel_dtype)
        if i.ndim == q.ndim:
            # Squeeze input and forget gate on last axis.
            i = i[..., 0]
            f = f[..., 0]
        if self.config.backend_name == "triton_stablef":
            return mlstm_chunkwise_triton_stablef(
                q,
                k,
                v,
                i,
                f,
                chunk_size=self.config.chunk_size,
                autocast_kernel_dtype=autocast_kernel_dtype,
                norm_val=self.config.norm_val,
                eps=self.config.eps,
                stabilize_correctly=self.config.stabilize_correctly,
            )
        elif self.config.backend_name == "max_triton":
            return mlstm_chunkwise_max_triton(
                q,
                k,
                v,
                i,
                f,
                chunk_size=self.config.chunk_size,
                autocast_kernel_dtype=autocast_kernel_dtype,
                reduce_slicing=self.config.reduce_slicing,
            )
        else:
            raise ValueError(f"Bad kernels backend name {self.config.backend_name}")

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        Triton kernels already handle the head dimension, hence not to be vmaped over.

        Returns:
            bool: False
        """
        return False
