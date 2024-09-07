# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

import jax
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
