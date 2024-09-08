# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass

from flax import linen as nn
import jax
import jax.numpy as jnp

from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .blocks.mlstm.block import mLSTMBlockConfig, mLSTMLayerConfig
from .utils import ParallelConfig, prepare_module

@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    parallel: ParallelConfig | None = None


class xLSTMLMModel(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, idx: jax.Array, train: bool = False) -> jax.Array:
        embed_fn = prepare_module(
            nn.Embed,
            "Embed",
            config=self.config.parallel
        )
        x = embed_fn(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim,
            dtype=self.config.dtype,
            name="token_embedding",
        )(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        stack_fn = prepare_module(
            xLSTMBlockStack,
            "BlockStack",
            config=self.config.parallel
        )
        x = stack_fn(config=self.config, name="xlstm_block_stack")(x, train=train)
        pred_fn = prepare_module(
            nn.Dense,
            "LMHead",
            config=self.config.parallel
        )
        logits = pred_fn(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=jnp.float32,
            name="lm_head",
        )(x)
        return logits
