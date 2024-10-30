from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import small_init, wang_init


@dataclass
class FeedForwardConfig(SubModelConfig):
    """
    Configuration for the feedforward network.
    """

    multiple_of: int = 64
    """The hidden dimension of the feedforward network will be increased to a multiple of this value. This is useful for
    ensuring an efficient use of the hardware, e.g. for tensor cores."""
    ffn_dim_multiplier: float = 1.0
    """Multiplier for the hidden dimension of the feedforward network. By default, the hidden dimension is up to 8/3 of
    the input dimension. This multiplier is applied to this default size and can be used to increase or decrease the
    hidden dimension. This is in line with the original PyTorch Llama implementation."""
    use_bias: bool = False
    """Whether to use bias in the feedforward network."""
    dropout_rate: float = 0.0
    """Dropout rate for the feedforward network."""
    num_layers: int = 12
    """Number of layers in the whole Llama Transformer model. Used for initialization."""
    dtype: str = "float32"
    """Data type of the activations in the network."""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class FeedForward(nn.Module):
    config: FeedForwardConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        assert self.config.parallel.model_axis_size == 1, "Feedforward network does not support model parallelism yet."
        embed_dim = x.shape[-1]

        # Determine hidden dimension
        hidden_dim = int(8 * embed_dim / 3)
        if self.config.ffn_dim_multiplier:
            hidden_dim = int(hidden_dim * self.config.ffn_dim_multiplier)
        hidden_dim = self.config.multiple_of * ((hidden_dim + self.config.multiple_of - 1) // self.config.multiple_of)

        # Feedforward network
        up_proj = nn.Dense(
            features=hidden_dim,
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config._dtype,
            name="proj_up",
        )(x)
        gate_preact = nn.Dense(
            features=hidden_dim,
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config._dtype,
            name="proj_up_gate",
        )(x)
        feats = up_proj * nn.silu(gate_preact)
        out = nn.Dense(
            features=embed_dim,
            use_bias=self.config.use_bias,
            kernel_init=wang_init(embed_dim, 2 * self.config.num_layers),
            dtype=self.config._dtype,
            name="proj_down",
        )(feats)
        out = nn.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
        return out
