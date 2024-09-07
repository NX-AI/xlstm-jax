# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

from flax import linen as nn
import jax
import jax.numpy as jnp

from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .blocks.mlstm.block import mLSTMBlockConfig, mLSTMLayerConfig


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
            dtype=self.config.dtype,
            name="token_embedding",
        )(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        x = xLSTMBlockStack(config=self.config, name="blocks")(x, train=train)
        logits = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=jnp.float32,
            name="lm_head",
        )(x)
        return logits