from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn

from .components.init import small_init
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False


class xLSTMLMModel(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, idx: jax.Array, train: bool = False) -> jax.Array:
        x = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim,
            embedding_init=small_init(self.config.embedding_dim),
            dtype=self.config.dtype,
            name="token_embedding",
        )(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        x = xLSTMBlockStack(config=self.config, name="xlstm_block_stack")(x, train=train)
        logits = nn.Dense(
            features=self.config.vocab_size,
            kernel_init=small_init(self.config.embedding_dim),
            use_bias=False,
            dtype=jnp.float32,
            name="lm_head",
        )(x)
        return logits
