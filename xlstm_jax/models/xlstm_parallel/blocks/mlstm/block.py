from dataclasses import dataclass, field
from functools import partial

import jax
from flax import linen as nn

from xlstm_jax.models.configs import SubModelConfig

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig(SubModelConfig):
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


def get_partial_mLSTMBlock(config: mLSTMBlockConfig, *args, **kwargs) -> callable:
    return partial(
        xLSTMBlock,
        xLSTMBlockConfig(
            mlstm=config.mlstm,
            slstm=None,
            feedforward=None,
            dtype=config.mlstm.dtype,
            parallel=config.parallel,
            _num_blocks=config._num_blocks,
            _block_idx=config._block_idx,
        ),
        *args,
        **kwargs,
    )


def mLSTMBlock(config: mLSTMBlockConfig, *args, **kwargs) -> nn.Module:
    block_fn = get_partial_mLSTMBlock(config, *args, **kwargs)
    return block_fn()
