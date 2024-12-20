#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import InitDistribution, prepare_module

from .blocks.mlstm.block import get_partial_mLSTMBlock, mLSTMBlockConfig


@dataclass
class xLSTMBlockStackConfig(SubModelConfig):
    mlstm_block: mLSTMBlockConfig | None = None
    slstm_block: Any | None = (
        None  # TODO: this is temporary for hydra, since a pure None type is not supported and
        #       the sLSTMBlockConfig is not implemented yet
    )

    context_length: int = -1
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    bias: bool = False
    dropout: float = 0.0
    scan_blocks: bool = False
    dtype: str = "bfloat16"

    parallel: ParallelConfig | None = None
    init_distribution_embed: InitDistribution = "normal"
    """Distribution type from which to sample the embeddings."""
    init_distribution_out: InitDistribution = "normal"
    """Distribution type from which to sample the LM output head."""

    # The block indices at which sLSTM blocks are placed.
    # Indexing starts from 0.
    slstm_at: list[int] = field(default_factory=list)

    # _block_map is a string that specifies which block is used at which position
    # 0: use the mLSTM block
    # 1: use the sLSTM block
    _block_map: str | None = None

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
            self.mlstm_block.mlstm.norm_type = self.norm_type
            self.mlstm_block.parallel = self.parallel

            self.mlstm_block._num_blocks = self.num_blocks
            # call post init, for setting inner_embedding_dim
            self.mlstm_block.__post_init__()

        if self.slstm_block is not None:
            self.slstm_block.slstm.dropout = self.dropout
            self.slstm_block.slstm.embedding_dim = self.embedding_dim
            self.slstm_block._num_blocks = self.num_blocks
            self.slstm_block.parallel = self.parallel
            self.slstm_block.__post_init__()

        self._block_map = self._create_block_map()

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class xLSTMBlockStack(nn.Module):
    config: xLSTMBlockStackConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        blocks = BlockStack(config=self.config, name="blocks")
        x = blocks(x, **kwargs)
        # Post-block norms integrated in LM Model
        return x


class BlockStack(nn.Module):
    config: xLSTMBlockStackConfig

    @nn.compact
    def __call__(self, x: jax.Array, *args, document_borders: jax.Array | None = None, **kwargs) -> jax.Array:
        """
        Forward pass of the block stack.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_dim).
            args: Additional arguments to pass to the mLSTM/sLSTM and feedforward layers.
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).
            kwargs: Additional kwargs to pass to the mLSTM/sLSTM and feedforward layers.

        Returns:
            The output tensor of the mLSTM layer, shape (batch_size, context_length, embedding_dim).
        """
        if not self.config.scan_blocks:
            blocks = self._create_blocks(config=self.config)
            # TODO: Add train etc. flags, but need to be static for remat.
            for block in blocks:
                x = block(x, document_borders=document_borders)
        else:
            assert all(v == 0 for v in self.config.block_map), "scan_blocks only supported for pure mLSTM blocks"
            block_fn = prepare_module(
                get_partial_mLSTMBlock(self.config.mlstm_block),
                "mLSTMBlock",
                self.config.parallel,
            )
            block = block_fn(name="block")
            x, _ = nn.scan(
                lambda module, carry, _: (module(carry, document_borders=document_borders), None),
                variable_axes={"params": 0, "intermediates": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.num_blocks,
                metadata_params={
                    "partition_name": None,
                },
            )(block, x, ())
        return x

    def _create_blocks(self, config: xLSTMBlockStackConfig):
        blocks = []
        for block_idx, block_type_int in enumerate(config.block_map):
            if block_type_int == 0:
                config = deepcopy(self.config.mlstm_block)
                if hasattr(config, "_block_idx"):
                    config._block_idx = block_idx
                    config.__post_init__()
                mlstm_layer_fn = prepare_module(
                    get_partial_mLSTMBlock(config=config),
                    "mLSTMBlock",
                    config.parallel,
                )
                blocks.append(mlstm_layer_fn(name=f"{block_idx}"))
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
