from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.array_utils import split_array_over_mesh
from xlstm_jax.distributed.data_parallel import shard_module_params
from xlstm_jax.distributed.tensor_parallel import ModelParallelismWrapper

from ..configs import ParallelConfig
from .components.init import small_init
from .components.normalization import resolve_norm
from .utils import prepare_module, soft_cap_logits
from .xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig


@dataclass
class xLSTMLMModelConfig(xLSTMBlockStackConfig):
    vocab_size: int = -1
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in normalization layer."""
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    """Type of normalization layer to use."""
    logits_soft_cap: float | None = None
    """Soft cap for the LM output logits. If None, no cap is applied."""
    lm_head_dtype: str | Any = jnp.float32
    """Data type to perform the LM Head Dense layer in. The output will always be casted to float32 for numerical
    stability."""
    parallel: ParallelConfig | None = None


class xLSTMLMModel(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, idx: jax.Array, document_borders: jax.Array | None = None, train: bool = False) -> jax.Array:
        """
        Forward pass of the xLSTM language model.

        Args:
            idx: input token indices of shape (batch_size, context_length).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). For document border tokens, the mLSTM memory will be reset if selected in
                config (see mlstm_cell). Shape (batch_size, context_length).
            train: Whether the model is in training mode. If True, may apply dropout.

        Returns:
            The predicted logits of the language model, shape (batch_size, context_length, vocab_size).
        """
        # Embedding
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        embed_fn = partial(
            ModelParallelismWrapper,
            module_fn=partial(
                nn.Embed,
                num_embeddings=self.config.vocab_size,
                features=self.config.embedding_dim // tp_size,
                embedding_init=small_init(self.config.embedding_dim, self.config.init_distribution_embed),
                dtype=self.config.dtype,
                name="embed",
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            name="token_embedding",
        )
        embed_fn = prepare_module(embed_fn, "Embed", config=self.config.parallel)
        x = embed_fn()(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not train)
        # BlockStack
        stack_fn = prepare_module(xLSTMBlockStack, "BlockStack", config=self.config.parallel)
        x = stack_fn(config=self.config, name="xlstm_block_stack")(x, document_borders=document_borders, train=train)
        # LMHead
        pred_fn = prepare_module(
            partial(TPOutputLayer, config=self.config, name="lm_head"), "LMHead", config=self.config.parallel
        )
        logits = pred_fn()(x)
        logits = soft_cap_logits(logits, self.config.logits_soft_cap)
        return logits


class TPOutputLayer(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Gather outputs over feature dimension and split over sequence length.
        x = jax.lax.all_gather(x, axis_name=self.config.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.model_axis_name, split_axis=1)
        # Apply norm - Shard parameters over model axis.
        if self.config.add_post_blocks_norm:
            norm_class, norm_kwargs = resolve_norm(
                self.config.norm_type,
                weight=True,
                bias=False,
                eps=self.config.norm_eps,
                dtype=self.config.dtype,
                name="out_norm",
            )
            norm_class = shard_module_params(
                norm_class,
                axis_name=self.config.parallel.model_axis_name,
                min_weight_size=self.config.parallel.fsdp_min_weight_size,
            )
            x = norm_class(**norm_kwargs)(x)
        # Apply output layer - Shard parameters over model axis.
        dense_fn = shard_module_params(
            nn.Dense,
            axis_name=self.config.parallel.model_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )
        x = dense_fn(
            features=self.config.vocab_size,
            kernel_init=small_init(self.config.embedding_dim, self.config.init_distribution_out),
            use_bias=False,
            dtype=self.config.lm_head_dtype,
            name="out_dense",
        )(x)
        # Output will be enforced to be float32.
        x = x.astype(jnp.float32)
        return x
