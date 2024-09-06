# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig:
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


class mLSTMBlock(xLSTMBlock):
    config: mLSTMBlockConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        return xLSTMBlock(config=xLSTMBlockConfig(
                mlstm=self.config.mlstm,
                slstm=None,
                feedforward=None,
                _num_blocks=self.config._num_blocks,
                _block_idx=self.config._block_idx,
            ))(x, train=train, **kwargs)


def test_mLSTMBlock():
    config = mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
            proj_factor=2.0,
            embedding_dim=16,
            bias=True,
            dropout=0.0,
            context_length=128,
            dtype=jnp.bfloat16,
        ),
        _num_blocks=1,
        _block_idx=0,
    )
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng, dp_rng = jax.random.split(rng, 3)
    block = mLSTMBlock(config=config)
    input_tensor = jax.random.normal(inp_rng, (2, 128, 16))
    params = block.init(model_rng, input_tensor)
    output_tensor = block.apply(params, input_tensor, rngs={"dropout": dp_rng}, train=True)
    assert output_tensor.shape == (2, 128, 16)
    print("All tests for mLSTMBlock passed successfully.")
