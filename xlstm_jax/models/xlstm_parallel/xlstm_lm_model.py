from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed.tensor_parallel import ModelParallelismWrapper
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.shared import TPLMHead, prepare_module, small_init

from .components.normalization import resolve_norm
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
    lm_head_dtype: str = "float32"
    """Data type to perform the LM Head Dense layer in. The output will always be casted to float32 for numerical
    stability."""
    parallel: ParallelConfig | None = None

    @property
    def _lm_head_dtype(self) -> jnp.dtype:
        """
        Return the real dtype instead of the str in config.

        Returns:
            Dtype corresponding to the respective str attribute.
        """
        return getattr(jnp, self.lm_head_dtype)


class xLSTMLMModel(nn.Module):
    config: xLSTMLMModelConfig

    @nn.compact
    def __call__(
        self, idx: jax.Array, document_borders: jax.Array | None = None, train: bool = False, **kwargs
    ) -> jax.Array:
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
                dtype=self.config._dtype,
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
        norm_class, norm_kwargs = resolve_norm(
            self.config.norm_type,
            weight=True,
            bias=False,
            eps=self.config.norm_eps,
            dtype=self.config.dtype,
            name="out_norm",
        )
        pred_fn = prepare_module(
            partial(
                TPLMHead,
                parallel=self.config.parallel,
                vocab_size=self.config.vocab_size,
                kernel_init=small_init(self.config.embedding_dim, self.config.init_distribution_out),
                norm_fn=partial(norm_class, **norm_kwargs),
                lm_head_dtype=self.config._lm_head_dtype,
                logits_soft_cap=self.config.logits_soft_cap,
                name="lm_head",
            ),
            "LMHead",
            config=self.config.parallel,
        )
        logits = pred_fn()(x)
        return logits
