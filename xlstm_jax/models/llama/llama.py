#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed import ModelParallelismWrapper
from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import TPLMHead, prepare_module, small_init

from .attention import AttentionBackend, SelfAttention, SelfAttentionConfig, precompute_freqs
from .feedforward import FeedForward, FeedForwardConfig


@dataclass
class LlamaConfig(SubModelConfig):
    """
    Configuration for the LLAMA model.

    For simplicity, all the configuration options are kept in a single class. Sub-configs for the attention and
    feedforward blocks are generated dynamically from this class.
    """

    vocab_size: int
    """Vocabulary size."""
    embedding_dim: int
    """Embedding dimension."""
    num_blocks: int
    """Number of transformer blocks. One block consists of self-attention and feedforward block."""
    head_dim: int = 128
    """Dimension of the attention heads. The number of heads is inferred by `embed_dim // head_dim`."""
    qk_norm: bool = False
    """Whether to apply RMSNorm to Q and K."""
    causal: bool = True
    """Whether to use causal attention masking."""
    theta: float = 1e4
    """Rotary position encoding frequency."""
    ffn_multiple_of: int = 64
    """Multiple of the feedforward hidden dimension to increase to."""
    ffn_dim_multiplier: float = 1.0
    """Multiplier to apply to the feedforward hidden dimension. By default, the hidden dimension is 8/3 of the
    embedding dimension. The multiplier is applied to this hidden dimension."""
    use_bias: bool = False
    """Whether to use bias in the linear layers."""
    scan_blocks: bool = True
    """Whether to scan the transformer blocks. Recommended for larger models to reduce compilation time."""
    attention_backend: AttentionBackend = "xla"
    """Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels are used.
    cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on all backends."""
    mask_across_document_boundaries: bool = False
    """Whether to mask attention across document boundaries. If True, the tokens in a document can only attend to
    tokens within the same document."""
    dtype: str = "float32"
    """Data type to use for the activations."""
    dropout_rate: float = 0.0
    """Dropout rate to apply to the activations."""
    add_embedding_dropout: bool = False
    """Whether to apply dropout to the embeddings."""
    logits_soft_cap: float | None = None
    """Soft cap for the logits. If not None, the logits are soft-clipped to the range
    [-logits_soft_cap, logits_soft_cap]."""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


class SelfAttentionBlock(nn.Module):
    """
    Attention block consisting of self-attention and residual connection.

    Args:
        config: Configuration for the attention block.
        train: Whether to run in training mode or not. If True, applies dropout.
    """

    config: LlamaConfig
    train: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        freqs: tuple[jax.Array, jax.Array] | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """
        Apply self-attention to the input tensor.

        Includes a pre-norm.

        Args:
            x: Input tensor.
            freqs: Precomputed frequency tensors.
            segment_ids: Segment IDs for the input tensor.

        Returns:
            Output tensor of self-attention with residual connection.
        """
        res = x
        x = nn.RMSNorm(name="pre_norm", dtype=self.config._dtype)(x)
        attn_config = SelfAttentionConfig(
            head_dim=self.config.head_dim,
            qk_norm=self.config.qk_norm,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
            num_layers=self.config.num_blocks,
            dropout_rate=self.config.dropout_rate,
            attention_backend=self.config.attention_backend,
            causal=self.config.causal,
            parallel=self.config.parallel,
        )
        x = SelfAttention(config=attn_config, name="attn")(x, freqs, segment_ids=segment_ids, train=self.train)
        self.sow("intermediates", "attn_out_std", x.std(axis=-1).mean())
        self.sow("intermediates", "attn_out_abs_max", jnp.abs(x).max())
        x += res
        return x


class FFNBlock(nn.Module):
    """
    Feedforward block consisting of a feedforward layer and residual connection.

    Args:
        config: Configuration for the feedforward block.
        train: Whether to run in training mode or not. If True, applies dropout.
    """

    config: LlamaConfig
    train: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply feedforward layer to the input tensor.

        Includes a pre-norm.

        Args:
            x: Input tensor.

        Returns:
            Output tensor of feedforward layer with residual connection.
        """
        res = x
        x = nn.RMSNorm(name="pre_norm", dtype=self.config._dtype)(x)
        ffn_config = FeedForwardConfig(
            multiple_of=self.config.ffn_multiple_of,
            ffn_dim_multiplier=self.config.ffn_dim_multiplier,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
            num_layers=self.config.num_blocks,
            dropout_rate=self.config.dropout_rate,
            parallel=self.config.parallel,
        )
        x = FeedForward(config=ffn_config, name="ffn")(x, train=self.train)
        self.sow("intermediates", "ffn_out_std", x.std(axis=-1).mean())
        self.sow("intermediates", "ffn_out_abs_max", jnp.abs(x).max())
        x += res
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of self-attention and feedforward block.

    Args:
        config: Configuration for the transformer block.
        train: Whether to run in training mode or not. If True, applies dropout.
    """

    config: LlamaConfig
    train: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        freqs: tuple[jax.Array, jax.Array] | None = None,
        segment_ids: jax.Array | None = None,
    ) -> jax.Array:
        """
        Apply transformer block to the input tensor.

        Args:
            x: Input tensor.
            freqs: Precomputed frequency tensors.
            segment_ids: Segment IDs for the input tensor

        Returns:
            Output tensor of the transformer block.
        """
        # Self-attention
        attn_block = partial(SelfAttentionBlock, config=self.config, train=self.train)
        attn_block = prepare_module(
            attn_block,
            "AttnBlock",
            self.config.parallel,
        )
        x = attn_block(name="attn")(x, freqs, segment_ids)

        # Feedforward
        ffn_block = partial(FFNBlock, config=self.config, train=self.train)
        ffn_block = prepare_module(
            ffn_block,
            "FFNBlock",
            self.config.parallel,
        )
        x = ffn_block(name="ffn")(x)

        return x


class TransformerBlockStack(nn.Module):
    """
    Stack of transformer blocks.

    Args:
        config: Configuration for the transformer block stack.
    """

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        pos_idx: jax.Array | None = None,
        document_borders: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        """
        Apply stack of transformer blocks to the input tensor.

        Args:
            x: Input tensor.
            pos_idx: Optional input tensor of positional indices. If None, positional indices are generated as an
                increasing sequence from 0 to context_length - 1. Shape (batch_size, context_length).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). If self.config.mask_across_document_boundaries is True, these borders will be
                used to identify which document each token belongs to, and masks out attention of tokens across
                different documents. Shape (batch_size, context_length).
            train: Whether to run in training mode or not. If True, applies dropout.

        Returns:
            Output tensor of the transformer block stack.
        """
        freqs = precompute_freqs(self.config.head_dim, pos_idx, x.shape[1], self.config.theta, dtype=self.config._dtype)
        if document_borders is not None and self.config.mask_across_document_boundaries:
            segment_ids = jnp.cumsum(document_borders.astype(jnp.int32), axis=1)
        else:
            segment_ids = None

        block_fn = partial(TransformerBlock, config=self.config, train=train)
        block_fn = prepare_module(
            block_fn,
            "TransformerBlock",
            self.config.parallel,
        )

        if not self.config.scan_blocks:
            for layer_idx in range(self.config.num_blocks):
                x = block_fn(name=f"block_{layer_idx}")(x, freqs, segment_ids)
        else:
            block = block_fn(name="block")
            x, _ = nn.scan(
                lambda module, carry, _: (module(carry, freqs, segment_ids), None),
                variable_axes={"params": 0, "intermediates": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.num_blocks,
                metadata_params={
                    "partition_name": None,
                },
            )(block, x, ())
        return x


class LlamaTransformer(nn.Module):
    """
    LLAMA transformer model.

    Args:
        config: Configuration for the LLAMA model.
    """

    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        idx: jax.Array,
        pos_idx: jax.Array | None = None,
        document_borders: jax.Array | None = None,
        train: bool = False,
        **kwargs,
    ) -> jax.Array:
        """
        Apply LLAMA transformer model to the input tensor.

        Example inputs for batch size 1 and context length 9:
                     idx = [[1, 8, 3, 2, 1, 4, 5, 5, 3]]
                 pos_idx = [[0, 1, 2, 3, 0, 1, 2, 3, 4]]
        document_borders = [[T, F, F, F, T, F, F, F, F]]
        where 1 is the end-/start-of-document token, and T and F are True and False, respectively.

        Args:
            idx: Input tensor of token indices.
            pos_idx: Optional input tensor of positional indices. If None, positional indices are generated as an
                increasing sequence from 0 to context_length - 1. Shape (batch_size, context_length).
            document_borders: Optional boolean tensor indicating which input tokens represent document borders (True)
                and which don't (False). If self.config.mask_across_document_boundaries is True, these borders will be
                used to identify which document each token belongs to, and masks out attention of tokens across
                different documents. Shape (batch_size, context_length).
            train: Whether to run in training mode or not. If True, applies dropout.

        Returns:
            Output tensor of the LLAMA model.
        """
        # Embedding
        tp_size = jax.lax.psum(1, self.config.parallel.model_axis_name)
        embed_fn = partial(
            ModelParallelismWrapper,
            module_fn=partial(
                nn.Embed,
                num_embeddings=self.config.vocab_size,
                features=self.config.embedding_dim // tp_size,
                embedding_init=small_init(self.config.embedding_dim),
                dtype=self.config._dtype,
                name="embed",
            ),
            model_axis_name=self.config.parallel.model_axis_name,
            name="token_embedding",
        )
        embed_fn = prepare_module(embed_fn, "Embed", config=self.config.parallel)
        x = embed_fn()(idx)
        if self.config.add_embedding_dropout:
            x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)
        # BlockStack
        stack_fn = prepare_module(TransformerBlockStack, "BlockStack", config=self.config.parallel)
        x = stack_fn(config=self.config, name="block_stack")(
            x, pos_idx=pos_idx, document_borders=document_borders, train=train
        )
        # LMHead
        pred_fn = prepare_module(
            partial(
                TPLMHead,
                parallel=self.config.parallel,
                vocab_size=self.config.vocab_size,
                kernel_init=small_init(self.config.embedding_dim),
                norm_fn=partial(nn.RMSNorm, dtype=self.config.dtype, name="out_norm"),
                lm_head_dtype=self.config._dtype,
                logits_soft_cap=self.config.logits_soft_cap,
                name="lm_head",
            ),
            "LMHead",
            config=self.config.parallel,
        )
        logits = pred_fn()(x)
        return logits
