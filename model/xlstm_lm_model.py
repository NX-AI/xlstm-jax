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


def test_xLSTMLMModel():
    config = xLSTMLMModelConfig(
        vocab_size=100,
        embedding_dim=16,
        num_blocks=2,
        context_length=128,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        dtype=jnp.bfloat16,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=16,
                context_length=128,
                dtype=jnp.bfloat16
            )
        )
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng, 2)
    model = xLSTMLMModel(config=config)
    input_tensor = jax.random.randint(inp_rng, (2, 128), 0, 100)
    params = model.init(model_rng, input_tensor)
    logits = model.apply(params, input_tensor)
    assert logits.shape == (2, 128, 100)
    assert logits.dtype == jnp.float32
    print("All tests for xLSTMLMModel passed successfully.")