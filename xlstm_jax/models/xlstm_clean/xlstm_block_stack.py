#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from .blocks.mlstm.block import mLSTMBlock, mLSTMBlockConfig
from .components.ln import LayerNorm


@dataclass
class xLSTMBlockStackConfig:
    mlstm_block: mLSTMBlockConfig | None = None
    slstm_block: None = None

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0
    dtype: str = "bfloat16"

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: list[int] | Literal["all"] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: str = None

    @property
    def block_map(self) -> list[int]:
        return list(map(int, self._block_map.split(",")))

    def _create_block_map(self) -> str:
        """Creates the block map, that specifies which block is used at which position."""
        block_map = [0] * self.num_blocks

        for slstm_position_idx in self.slstm_at:
            assert slstm_position_idx < self.num_blocks, f"Invalid slstm position {slstm_position_idx}"
            block_map[slstm_position_idx] = 1

        block_map_str = ",".join(map(str, block_map))

        return block_map_str

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)

    def __post_init__(self):
        if self.mlstm_block is None:
            self.slstm_at = "all"
        if self.slstm_at == "all":
            self.slstm_at = list(range(self.num_blocks))

        if self.mlstm_block is not None:
            self.mlstm_block.mlstm.embedding_dim = self.embedding_dim
            self.mlstm_block.mlstm.bias = self.bias
            self.mlstm_block.mlstm.dropout = self.dropout
            self.mlstm_block.mlstm.context_length = self.context_length
            self.mlstm_block.mlstm.dtype = self.dtype

            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()


class xLSTMBlockStack(nn.Module):
    config: xLSTMBlockStackConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        blocks = BlockStack(config=self.config, name="blocks")
        # for block in blocks:
        #     x = block(x, **kwargs)
        x = blocks(x, **kwargs)
        if self.config.add_post_blocks_norm:
            x = LayerNorm(dtype=self.config._dtype, name="post_blocks_norm")(x)
        return x


class BlockStack(nn.Module):
    config: xLSTMBlockStackConfig

    @nn.compact
    def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
        blocks = self._create_blocks(config=self.config)
        for block in blocks:
            x = block(x, *args, **kwargs)
        return x

    def _create_blocks(self, config: xLSTMBlockStackConfig):
        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                blocks.append(mLSTMBlock(config=config, name=f"{block_idx}"))
            elif block_type_int == 1:
                config = deepcopy(self.config.slstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                # blocks.append(sLSTMBlock(config=config))
                raise NotImplementedError("sLSTM not implemented in JAX yet.")
            else:
                raise ValueError(f"Invalid block type {block_type_int}")
        return blocks
