from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.ln import LayerNorm
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig


@dataclass
class xLSTMBlockConfig:
    mlstm: mLSTMLayerConfig | None = None
    slstm: None = None

    feedforward: FeedForwardConfig | None = None
    dtype: str = "bfloat16"

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int | None = None
    _block_idx: int | None = None

    def __post_init__(self):
        assert self.mlstm is not None or self.slstm is not None, "Either mlstm or slstm must be provided"
        assert self.mlstm is None or self.slstm is None, "Only one of mlstm or slstm can be provided"
        embedding_dim = self.mlstm.embedding_dim if self.mlstm is not None else self.slstm.embedding_dim
        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx
        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx
        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class xLSTMBlock(nn.Module):
    """
    An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config: xLSTMBlockConfig

    @nn.compact
    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        xlstm_norm = LayerNorm(weight=True, bias=False, dtype=self.config._dtype, name="xlstm_norm")
        if self.config.mlstm is not None:
            xlstm = mLSTMLayer(config=self.config.mlstm, name="xlstm")
        elif self.config.slstm is not None:
            # xlstm = sLSTMLayer(config=self.config.slstm)
            raise NotImplementedError("sLSTM not implemented in JAX yet.")
        else:
            raise ValueError("Either mlstm or slstm must be provided")
        x_norm = xlstm_norm(x)
        x_xlstm = xlstm(x_norm, **kwargs)
        x += x_xlstm

        if self.config.feedforward is not None:
            ffn_norm = LayerNorm(weight=True, bias=False, dtype=self.config._dtype, name="ffn_norm")
            ffn = create_feedforward(config=self.config.feedforward)
            x = x + ffn(ffn_norm(x), **kwargs)
        return x
