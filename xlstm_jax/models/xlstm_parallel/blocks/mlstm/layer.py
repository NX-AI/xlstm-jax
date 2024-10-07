from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import ModelParallelismWrapper, TPAsyncDense, TPDense

from ....configs import ParallelConfig
from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.init import InitDistribution, small_init, wang_init
from ...components.linear_headwise import LinearHeadwiseExpand, LinearHeadwiseExpandConfig
from ...utils import UpProjConfigMixin, prepare_module
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0
    vmap_qk: bool = False
    init_distribution: InitDistribution = "normal"

    # will be set toplevel config
    embedding_dim: int = -1
    bias: bool = False
    dropout: float = 0.0
    context_length: int = -1
    dtype: jnp.dtype = jnp.bfloat16
    parallel: ParallelConfig | None = None

    # For debugging purposes, we allow skipping the mLSTM cell
    # and only have the up and down projection.
    debug_cell: bool = False
    gate_input: Literal["qkv", "x_mlstm", "x_mlstm_conv", "x_mlstm_conv_act"] = "qkv"
    """Which input to use for the mLSTM cell gates. Options are:
    - "qkv": use the query, key and value vectors concatenated as input. Default, as in paper version.
    - "x_mlstm": use the output of the mLSTM up projection layer. These are the same features that go into
        the V projection.
    - "x_mlstm_conv": use the output of the convolution on the mLSTM up projection features.
    - "x_mlstm_conv_act": use the output of the activation function on the convolution on the mLSTM up projection
        features. These are the same features that go into the QK projection.
    """

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


class mLSTMLayer(nn.Module):
    config: mLSTMLayerConfig

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True, **kwargs) -> jax.Array:
        B, S, _ = x.shape
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config.num_heads % tp_size == 0, "num_heads must be divisible by the number of model replicas"
        embedding_dim = x.shape[-1] * tp_size

        tp_dense_fn = (
            partial(TPAsyncDense, use_bidirectional_gather=True, use_bidirectional_scatter=True)
            if self.config.parallel.tp_async_dense
            else TPDense
        )
        # up-projection
        up_proj_init = small_init(dim=embedding_dim, distribution=self.config.init_distribution)
        if self.config.parallel.tp_async_dense:
            x_inner = tp_dense_fn(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config.dtype,
                    use_bias=self.config.bias,
                    features=2 * self.config._inner_embedding_dim // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=up_proj_init,
                name="proj_up",
            )(x)
        else:
            # split projection up into two smaller matrices, as splitting large feature vector costs time.
            x = jax.lax.all_gather(x, self.config.parallel.model_axis_name, axis=-1, tiled=True)
            x_mlstm = TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config.dtype,
                    use_bias=self.config.bias,
                    features=self.config._inner_embedding_dim // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=up_proj_init,
                skip_communication=True,
                name="proj_up_mlstm",
            )(x)
            z = TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config.dtype,
                    use_bias=self.config.bias,
                    features=self.config._inner_embedding_dim // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=up_proj_init,
                skip_communication=True,
                name="proj_up_z",
            )(x)
            x_inner = (x_mlstm, z)
            # x_inner = TPDense(
            #     dense_fn=partial(
            #         nn.Dense,
            #         dtype=self.config.dtype,
            #         use_bias=self.config.bias,
            #         features=2 * self.config._inner_embedding_dim // tp_size,
            #     ),
            #     model_axis_name=self.config.parallel.model_axis_name,
            #     tp_mode="gather",
            #     skip_communication=False,
            #     kernel_init_adjustment=tp_size**-0.5,
            #     name="proj_up",
            # )(x)

        if self.config.debug_cell:
            if isinstance(x_inner, tuple):
                h_state = x_inner[0]
            else:
                h_state = x_inner[..., : x_inner.shape[-1] // 2]
        else:
            # inner cell
            inner_config = deepcopy(self.config)
            inner_config.num_heads = self.config.num_heads // tp_size
            inner_config.embedding_dim = self.config.embedding_dim // tp_size
            inner_config.__post_init__()
            mlstm_inner_layer = prepare_module(
                mLSTMInnerLayer,
                "mLSTMInnerLayer",
                self.config.parallel,
            )
            h_state = ModelParallelismWrapper(
                module_fn=partial(
                    mlstm_inner_layer,
                    config=inner_config,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                name="inner_layer",
            )(x_inner)

        # down-projection
        y = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config.dtype,
                use_bias=self.config.bias,
                features=self.config.embedding_dim // tp_size
                if self.config.parallel.tp_async_dense
                else self.config.embedding_dim,
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="scatter",
            kernel_init=wang_init(embedding_dim, num_blocks=self.config._num_blocks),
            name="proj_down",
        )(h_state)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)
        return y


class mLSTMInnerLayer(nn.Module):
    config: mLSTMLayerConfig

    @nn.compact
    def __call__(self, x_inner: jax.Array | tuple[jax.Array, jax.Array]) -> jax.Array:
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        if isinstance(x_inner, tuple):
            x_mlstm, z = x_inner
        else:
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
        qkv_init = small_init(dim=self.config.embedding_dim * tp_size, distribution=self.config.init_distribution)
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
                kernel_init=qkv_init,
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
                kernel_init=qkv_init,
                name="q_proj",
            )(x_mlstm_conv_act)
            k = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=qkv_init,
                name="k_proj",
            )(x_mlstm_conv_act)
        v = LinearHeadwiseExpand(
            config=LinearHeadwiseExpandConfig(
                in_features=self.config._inner_embedding_dim,
                num_heads=num_proj_heads,
                bias=self.config.bias,
                dtype=self.config.dtype,
            ),
            kernel_init=qkv_init,
            name="v_proj",
        )(x_mlstm)

        mlstm_cell = prepare_module(
            mLSTMCell,
            "mLSTMCell",
            self.config.parallel,
        )
        gate_input = None
        if self.config.gate_input == "qkv":
            gate_input = q, k, v
        elif self.config.gate_input == "x_mlstm":
            gate_input = x_mlstm
        elif self.config.gate_input == "x_mlstm_conv":
            gate_input = x_mlstm_conv
        elif self.config.gate_input == "x_mlstm_conv_act":
            gate_input = x_mlstm_conv_act
        else:
            raise ValueError(f"Invalid gate_inputs: {self.config.gate_input}")
        h_tilde_state = mlstm_cell(config=self.config.mlstm_cell, name="mlstm_cell")(
            q=q, k=k, v=v, gate_input=gate_input
        )
        learnable_skip = self.param("learnable_skip", nn.initializers.ones, (x_mlstm_conv_act.shape[-1],))
        learnable_skip = jnp.broadcast_to(learnable_skip, x_mlstm_conv_act.shape)
        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * nn.swish(z)
        return h_state
