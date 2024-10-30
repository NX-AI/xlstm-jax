from dataclasses import dataclass, field
from functools import partial

from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig

from ...components.feedforward import FeedForwardConfig
from ..xlstm_block import xLSTMBlock, xLSTMBlockConfig
from .layer import mLSTMLayerConfig


@dataclass
class mLSTMBlockConfig(SubModelConfig):
    mlstm: mLSTMLayerConfig = field(default_factory=mLSTMLayerConfig)
    feedforward: FeedForwardConfig | None = None
    add_post_norm: bool = False
    """If True, adds a normalization layer after the mLSTM layer and the feedforward layer."""
    parallel: ParallelConfig | None = None
    """Parallel configuration for the model."""

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int | None = None
    _block_idx: int | None = None

    def __post_init__(self):
        self.mlstm._num_blocks = self._num_blocks
        self.mlstm.__post_init__()


def get_partial_mLSTMBlock(config: mLSTMBlockConfig, *args, **kwargs) -> callable:
    return partial(
        xLSTMBlock,
        xLSTMBlockConfig(
            mlstm=config.mlstm,
            slstm=None,
            parallel=config.parallel,
            feedforward=config.feedforward,
            dtype=config.mlstm.dtype,
            norm_eps=config.mlstm.mlstm_cell.norm_eps,
            norm_type=config.mlstm.mlstm_cell.norm_type,
            add_post_norm=config.add_post_norm,
            _num_blocks=config._num_blocks,
            _block_idx=config._block_idx,
        ),
        *args,
        **kwargs,
    )


def mLSTMBlock(config: mLSTMBlockConfig, *args, **kwargs) -> nn.Module:
    block_fn = get_partial_mLSTMBlock(config, *args, **kwargs)
    return block_fn()
