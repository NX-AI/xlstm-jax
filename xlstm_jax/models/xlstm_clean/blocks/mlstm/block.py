from dataclasses import dataclass, field

from flax import linen as nn

from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig:
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int | None = None
    _block_idx: int | None = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


def mLSTMBlock(config: mLSTMBlockConfig, *args, **kwargs) -> nn.Module:
    return xLSTMBlock(
        config=xLSTMBlockConfig(
            mlstm=config.mlstm,
            slstm=None,
            feedforward=None,
            dtype=config.mlstm.dtype,
            _num_blocks=config._num_blocks,
            _block_idx=config._block_idx,
        ),
        *args,
        **kwargs,
    )
