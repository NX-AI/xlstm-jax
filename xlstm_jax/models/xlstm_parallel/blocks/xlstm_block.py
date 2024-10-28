from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.shared import prepare_module

from ..components.feedforward import FeedForwardConfig, create_feedforward
from ..components.normalization import NormLayer
from .mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from .mlstm.layer_v1 import mLSTMLayerV1


@dataclass
class xLSTMBlockConfig:
    mlstm: mLSTMLayerConfig | None = None
    slstm: None = None
    parallel: ParallelConfig | None = None

    feedforward: FeedForwardConfig | None = None
    dtype: jnp.dtype = jnp.bfloat16
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in layer norm."""
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    """Type of normalization layer to use."""
    add_post_norm: bool = False
    """If True, adds a normalization layer after the mLSTM/sLSTM layer and the feedforward layer.
    Note that this is not the post-norm on the residual connection, but is applied to the output of the layers
    before the residual connection, following e.g. Gemma-2."""

    # we initialize these with None to catch the case where they are not set
    _num_blocks: int = None
    _block_idx: int = None

    def __post_init__(self):
        assert self.mlstm is not None or self.slstm is not None, "Either mlstm or slstm must be provided"
        assert self.mlstm is None or self.slstm is None, "Only one of mlstm or slstm can be provided"
        embedding_dim = self.mlstm.embedding_dim if self.mlstm is not None else self.slstm.embedding_dim
        if self.mlstm:
            self.mlstm._num_blocks = self._num_blocks
            self.mlstm._block_idx = self._block_idx
            self.mlstm.parallel = self.parallel
            self.mlstm.norm_type = self.norm_type
        if self.slstm:
            self.slstm._num_blocks = self._num_blocks
            self.slstm._block_idx = self._block_idx
            self.slstm.parallel = self.parallel
        if self.feedforward:
            self.feedforward.embedding_dim = embedding_dim
            self.feedforward.parallel = self.parallel
            self.feedforward._num_blocks = self._num_blocks
            self.feedforward.__post_init__()


class ResidualBlock(nn.Module):
    """
    A residual block that applies a list of modules in sequence and adds the input to the output.

    Modules are created within this block to wrap them as children module of this one.
    """

    module_fns: list[Callable[..., nn.Module]]

    @nn.compact
    def __call__(self, x: jax.Array, module_kwargs: list[dict[str, Any] | None] | None = None, **kwargs) -> jax.Array:
        """
        Forward pass of residual block.

        Args:
            x: Input tensor.
            module_kwargs: List of kwargs to pass to each individual module. If None, empty kwargs are used. If shorter
                than the number of modules, the last kwargs are set to empty.
            kwargs: Additional kwargs to pass to *all* modules.

        Returns:
            Output tensor.
        """
        # Prepare module kwargs.
        if module_kwargs is None:
            module_kwargs = [{}] * len(self.module_fns)
        else:
            module_kwargs = module_kwargs + [{}] * max(0, len(self.module_fns) - len(module_kwargs))
            module_kwargs = [{} if kws is None else kws for kws in module_kwargs]

        # Forward pass.
        res = x
        for module_fn, module_kws in zip(self.module_fns, module_kwargs):
            x = module_fn()(x, **module_kws, **kwargs)
        return res + x


class xLSTMBlock(nn.Module):
    """
    An xLSTM block can be either an sLSTM Block or an mLSTM Block.

    It contains the pre-LayerNorms and the skip connections.
    """

    config: xLSTMBlockConfig

    @nn.compact
    def __call__(self, x: jax.Array, document_borders: jax.Array | None = None, **kwargs) -> jax.Array:
        """
        Forward pass of xLSTM block.

        Includes both mLSTM/sLSTM and feedforward layers if specified.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_dim).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).
            kwargs: Additional kwargs to pass to the mLSTM/sLSTM and feedforward layers.

        Returns:
            The output tensor of the xLSTM block, shape (batch_size, context_length, embedding_dim).
        """
        # LayerNorm best to do over model axis, not sync beforehand due to costly embedding size.
        norm_fn = partial(
            NormLayer,
            weight=True,
            bias=False,
            dtype=self.config.dtype,
            axis_name=self.config.parallel.model_axis_name,
            eps=self.config.norm_eps,
            norm_type=self.config.norm_type,
            model_axis_name=self.config.parallel.model_axis_name,
        )

        # mLSTM or sLSTM
        if self.config.mlstm is not None:
            block_class = mLSTMLayer
            if self.config.mlstm.layer_type == "mlstm_v1":
                block_class = mLSTMLayerV1
            xlstm_fn = partial(block_class, config=self.config.mlstm, name="xlstm")
        elif self.config.slstm is not None:
            # xlstm = sLSTMLayer(config=self.config.slstm)
            raise NotImplementedError("sLSTM not implemented in JAX yet.")
        else:
            raise ValueError("Either mlstm or slstm must be provided")

        # Create Residual Block with mLSTM/sLSTM. Separate block needed for best FSDP/Remat support.
        xlstm_res_block = partial(
            ResidualBlock,
            [
                partial(norm_fn, name="xlstm_norm"),
                xlstm_fn,
            ]
            + ([partial(norm_fn, name="xlstm_post_norm")] if self.config.add_post_norm else []),
        )
        xlstm_res_block = prepare_module(
            xlstm_res_block,
            "xLSTMResBlock",
            config=self.config.parallel,
        )
        x = xlstm_res_block(name="xlstm_res_block")(x, [{}, {"document_borders": document_borders}], **kwargs)

        # Feedforward
        if self.config.feedforward is not None:
            ffn_res_block = partial(
                ResidualBlock,
                [
                    partial(norm_fn, name="ffn_norm"),
                    partial(create_feedforward, config=self.config.feedforward),
                ]
                + ([partial(norm_fn, name="ffn_post_norm")] if self.config.add_post_norm else []),
            )
            ffn_res_block = prepare_module(
                ffn_res_block,
                "FFNResBlock",
                config=self.config.parallel,
            )
            x = ffn_res_block(name="ffn_res_block")(x, **kwargs)
        return x
