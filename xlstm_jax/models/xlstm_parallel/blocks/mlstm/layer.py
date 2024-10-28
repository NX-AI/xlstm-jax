from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import ModelParallelismWrapper, TPAsyncDense, TPDense
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.shared import InitDistribution, InitFnName, create_common_init_fn, prepare_module, small_init

from ...components.conv import CausalConv1d, CausalConv1dConfig
from ...components.linear_headwise import LinearHeadwiseExpand, LinearHeadwiseExpandConfig
from ...utils import UpProjConfigMixin
from .cell import mLSTMCell, mLSTMCellConfig


@dataclass
class mLSTMLayerConfig(UpProjConfigMixin):
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    num_heads: int = 4
    proj_factor: float = 2.0
    vmap_qk: bool = False
    init_distribution: InitDistribution = "normal"
    """Distribution type from which to sample the weights."""
    output_init_fn: InitFnName = "wang"
    """Initialization function for the output projection layer."""
    layer_type: Literal["mlstm", "mlstm_v1"] = "mlstm"
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    """Type of normalization layer to use."""
    qk_dim_factor: float = 1.0
    """Factor to scale the qk projection dimension by. By default, the qk projection dimension is the same as the
    inner embedding dimension, split into num_heads. This factor is applied to this default size."""
    v_dim_factor: float = 1.0
    """Factor to scale the v projection dimension by. By default, the v projection dimension is the same as the
    inner embedding dimension, split into num_heads. This factor is applied to this default size."""

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
        self.mlstm_cell.norm_type = self.norm_type
        self.mlstm_cell.parallel = self.parallel


class mLSTMLayer(nn.Module):
    """
    The mLSTM layer with Mamba block style.
    """

    config: mLSTMLayerConfig

    @nn.compact
    def __call__(
        self, x: jax.Array, document_borders: jax.Array | None = None, train: bool = True, **kwargs
    ) -> jax.Array:
        """
        Forward pass of the mLSTM layer.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_dim).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).
            train: Whether the model is in training mode. If True, may apply dropout.

        Returns:
            The output tensor of the mLSTM layer, shape (batch_size, context_length, embedding_dim).
        """
        B, S, _ = x.shape
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        assert self.config.num_heads % tp_size == 0, "num_heads must be divisible by the number of model replicas"
        embedding_dim = x.shape[-1] * tp_size
        v_dim = int(self.config._inner_embedding_dim * self.config.v_dim_factor)
        assert embedding_dim % tp_size == 0, "embedding_dim must be divisible by the number of model replicas"
        assert v_dim % tp_size == 0, "value dimension must be divisible by the number of model replicas"
        # qk-factor does not need to be tested, as they are solely applied in the LinearHeadwise.

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
                    features=(v_dim + self.config._inner_embedding_dim) // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=up_proj_init,
                use_bias=self.config.bias,
                name="proj_up",
            )(x)
            x_inner = (x_inner[..., v_dim // tp_size :], x_inner[..., : v_dim // tp_size])
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
                use_bias=self.config.bias,
                skip_communication=True,
                name="proj_up_mlstm",
            )(x)
            z = TPDense(
                dense_fn=partial(
                    nn.Dense,
                    dtype=self.config.dtype,
                    features=v_dim // tp_size,
                ),
                model_axis_name=self.config.parallel.model_axis_name,
                tp_mode="gather",
                kernel_init=up_proj_init,
                use_bias=self.config.bias,
                skip_communication=True,
                name="proj_up_z",
            )(x)
            x_inner = (x_mlstm, z)

        if self.config.debug_cell:
            h_state = x_inner[1]
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
            )(x_inner, document_borders)

        # down-projection
        y = tp_dense_fn(
            dense_fn=partial(
                nn.Dense,
                dtype=self.config.dtype,
                features=self.config.embedding_dim // tp_size
                if self.config.parallel.tp_async_dense
                else self.config.embedding_dim,
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            tp_mode="scatter",
            kernel_init=create_common_init_fn(
                fn_name=self.config.output_init_fn,
                dim=embedding_dim,
                num_blocks=self.config._num_blocks,
                distribution=self.config.init_distribution,
            ),
            use_bias=self.config.bias,
            name="proj_down",
        )(h_state)
        y = nn.Dropout(rate=self.config.dropout, deterministic=not train)(y)
        return y


class mLSTMInnerLayer(nn.Module):
    """
    The inner mLSTM layer with Mamba block style.

    Applies a convolutional layer followed by a mLSTM cell.
    """

    config: mLSTMLayerConfig

    @nn.compact
    def __call__(
        self, x_inner: jax.Array | tuple[jax.Array, jax.Array], document_borders: jax.Array | None = None
    ) -> jax.Array:
        """
        Forward pass of the inner mLSTM layer.

        Args:
            x_inner: input tensor of shape (batch_size, context_length, embedding_dim + v_dim) or a tuple of two
                tensors (x_mlstm, z) with shapes (batch_size, context_length, embedding_dim) and
                (batch_size, context_length, v_dim). Represent the up-projected features that are mapped to query, key
                and value vectors, as well as skip connection z.
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).

        Returns:
            The output tensor of the inner mLSTM layer, shape (batch_size, context_length, v_dim).
        """
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
                    expand_factor_up=self.config.qk_dim_factor,
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
                    expand_factor_up=self.config.qk_dim_factor,
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
                    expand_factor_up=self.config.qk_dim_factor,
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
                expand_factor_up=self.config.v_dim_factor,
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
            q=q, k=k, v=v, gate_input=gate_input, document_borders=document_borders
        )
        if h_tilde_state.shape[-1] != x_mlstm_conv_act.shape[-1]:
            # Add a linear headwise projection to match the dimensions
            assert (
                h_tilde_state.shape[-1] / x_mlstm_conv_act.shape[-1] == self.config.v_dim_factor
            ), f"Invalid dimensions: {h_tilde_state.shape[-1]} != {x_mlstm_conv_act.shape[-1]}"
            x_mlstm_conv_act = LinearHeadwiseExpand(
                config=LinearHeadwiseExpandConfig(
                    in_features=self.config._inner_embedding_dim,
                    num_heads=num_proj_heads,
                    expand_factor_up=self.config.v_dim_factor,
                    bias=self.config.bias,
                    dtype=self.config.dtype,
                ),
                kernel_init=qkv_init,
                name="x_mlstm_conv_scaling",
            )(x_mlstm_conv_act)
        learnable_skip = self.param("learnable_skip", nn.initializers.ones, (x_mlstm_conv_act.shape[-1],))
        learnable_skip = jnp.broadcast_to(learnable_skip, x_mlstm_conv_act.shape)
        h_tilde_state_skip = h_tilde_state + (learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * nn.swish(z)
        return h_state
