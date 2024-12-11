#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import small_init, wang_init
from ...components.linear_headwise import (
    LinearHeadwiseExpand,
    LinearHeadwiseExpandConfig,
)
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0
    vmap_qk: bool = False

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1
    dtype: str = "bfloat16"

    _num_blocks: int = 1
    _inner_embedding_dim: int = None

    mlstm_cell: mLSTMCellConfig = field(default_factory=mLSTMCellConfig)

    def __post_init__(self):
        self._set_proj_up_dim(embedding_dim=self.embedding_dim)
        self._inner_embedding_dim = self._proj_up_dim
        self.mlstm_cell.context_length = self.context_length
        self.mlstm_cell.embedding_dim = self._inner_embedding_dim
        self.mlstm_cell.num_heads = self.num_heads
        self.mlstm_cell.dtype = self.dtype

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class mLSTMLayer(nn.Module):
    config: mLSTMLayerConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        # up-projection
        x_inner = nn.Dense(
            features=2 * self.config._inner_embedding_dim,
            dtype=self.config._dtype,
            kernel_init=small_init(x.shape[-1]),
            use_bias=self.config.bias,
            name="proj_up",
        )(x)
        x_mlstm, z = jnp.split(x_inner, 2, axis=-1)

        # mlstm branch
        x_mlstm_conv = CausalConv1d(
            config=CausalConv1dConfig(
                feature_dim=self.config._inner_embedding_dim,
                kernel_size=self.config.conv1d_kernel_size,
                dtype=self.config.dtype,
            ),
            name="conv1d",
        )(x_mlstm)
        x_mlstm_conv_act = nn.swish(x_mlstm_conv)

        num_proj_heads = round(self.config._inner_embedding_dim // self.config.qkv_proj_blocksize)
        if self.config.vmap_qk:
            qk = nn.vmap(
                LinearHeadwiseExpand,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=None,
                out_axes=0,
                axis_size=2,
            )(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="qk_proj",
            )(x_mlstm_conv_act)
            q, k = qk[0], qk[1]
        else:
            q = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="q_proj",
            )(x_mlstm_conv_act)
            k = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=small_init(self.config.embedding_dim),
                name="k_proj",
            )(x_mlstm_conv_act)
        v = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
                dtype=self.config.dtype,
            ),
            kernel_init=small_init(self.config.embedding_dim),
            name="v_proj",
        )(x_mlstm)

        h_tilde_state = mLSTMCell(config=self.config.mlstm_cell, name="mlstm_cell")(q=q, k=k, v=v)
        learnable_skip = self.param("learnable_skip", nn.initializers.ones, (x_mlstm_conv_act.shape[-1],))
        learnable_skip = jnp.broadcast_to(learnable_skip, x_mlstm_conv_act.shape)
        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * nn.swish(z)

        # down-projection
        y = nn.Dense(
            features=self.config.embedding_dim,
            dtype=self.config._dtype,
            kernel_init=wang_init(x.shape[-1], num_blocks=self.config._num_blocks),
            use_bias=self.config.bias,
            name="proj_down",
        )(h_state)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)
        return y
