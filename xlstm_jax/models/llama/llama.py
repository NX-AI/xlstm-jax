from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.distributed import ModelParallelismWrapper, shard_module_params, split_array_over_mesh
from xlstm_jax.models.xlstm_parallel.components.init import small_init

from ..configs import ParallelConfig, SubModelConfig
from .attention import SelfAttention, SelfAttentionConfig, create_causal_mask, precompute_freqs
from .feedforward import FeedForward, FeedForwardConfig
from .utils import prepare_module


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
    dtype: jnp.dtype = jnp.float32
    """Data type to use for the activations."""
    dropout_rate: float = 0.0
    """Dropout rate to apply to the activations."""
    add_embedding_dropout: bool = False
    """Whether to apply dropout to the embeddings."""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""


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
        mask: jax.Array | None = None,
        freqs: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        """
        Apply self-attention to the input tensor.

        Includes a pre-norm.

        Args:
            x: Input tensor.
            mask: Attention mask tensor. See `SelfAttention` for details.
            freqs: Precomputed frequency tensors.

        Returns:
            Output tensor of self-attention with residual connection.
        """
        res = x
        x = nn.RMSNorm(name="pre_norm", dtype=self.config.dtype)(x)
        attn_config = SelfAttentionConfig(
            head_dim=self.config.head_dim,
            qk_norm=self.config.qk_norm,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
            num_layers=self.config.num_blocks,
            dropout_rate=self.config.dropout_rate,
        )
        x = SelfAttention(config=attn_config, name="attn")(x, mask, freqs, train=self.train)
        self.sow("intermediates", "attn_out_std", x.std(axis=-1).mean())
        self.sow("intermediates", "attn_out_abs_max", jnp.abs(x).max())
        x = x + res
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
        x = nn.RMSNorm(name="pre_norm", dtype=self.config.dtype)(x)
        ffn_config = FeedForwardConfig(
            multiple_of=self.config.ffn_multiple_of,
            ffn_dim_multiplier=self.config.ffn_dim_multiplier,
            use_bias=self.config.use_bias,
            dtype=self.config.dtype,
            num_layers=self.config.num_blocks,
            dropout_rate=self.config.dropout_rate,
        )
        x = FeedForward(config=ffn_config, name="ffn")(x, train=self.train)
        self.sow("intermediates", "ffn_out_std", x.std(axis=-1).mean())
        self.sow("intermediates", "ffn_out_abs_max", jnp.abs(x).max())
        x = x + res
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
        mask: jax.Array | None = None,
        freqs: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        """
        Apply transformer block to the input tensor.

        Args:
            x: Input tensor.
            mask: Attention mask tensor. See `SelfAttention` for details.
            freqs: Precomputed frequency tensors.

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
        x = attn_block(name="attn")(x, mask, freqs)

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
    def __call__(self, x: jax.Array, train: bool = False) -> jax.Array:
        """
        Apply stack of transformer blocks to the input tensor.

        Args:
            x: Input tensor.
            train: Whether to run in training mode or not. If True, applies dropout.

        Returns:
            Output tensor of the transformer block stack.
        """
        mask = create_causal_mask(x.shape[1], dtype=jnp.bool_)
        freqs = precompute_freqs(self.config.head_dim, x.shape[1], self.config.theta)
        block_fn = partial(TransformerBlock, config=self.config, train=train)
        block_fn = prepare_module(
            block_fn,
            "TransformerBlock",
            self.config.parallel,
        )

        if not self.config.scan_blocks:
            for layer_idx in range(self.config.num_blocks):
                x = block_fn(name=f"block_{layer_idx}")(x, mask, freqs)
        else:
            block = block_fn(name="block")
            x, _ = nn.scan(
                lambda module, carry, _: (module(carry, mask, freqs), None),
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
    def __call__(self, idx: jax.Array, train: bool = False) -> jax.Array:
        """
        Apply LLAMA transformer model to the input tensor.

        Args:
            idx: Input tensor of token indices.
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
                dtype=self.config.dtype,
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
        x = stack_fn(config=self.config, name="block_stack")(x, train=train)
        # LMHead
        pred_fn = prepare_module(
            partial(TPOutputLayer, config=self.config, name="lm_head"), "LMHead", config=self.config.parallel
        )
        logits = pred_fn()(x)
        return logits


class TPOutputLayer(nn.Module):
    """
    Output layer for the LLAMA model.

    Supports Tensor Parallelism.

    Args:
        config: Configuration for the LLAMA model.
    """

    config: LlamaConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply output layer to the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Logits of the output layer.
        """
        # Gather outputs over feature dimension and split over sequence length.
        x = jax.lax.all_gather(x, axis_name=self.config.parallel.model_axis_name, axis=-1, tiled=True)
        x = split_array_over_mesh(x, axis_name=self.config.parallel.model_axis_name, split_axis=1)
        # Apply norm - Shard parameters over model axis.
        norm_fn = shard_module_params(
            nn.RMSNorm,
            axis_name=self.config.parallel.model_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )
        x = norm_fn(dtype=self.config.dtype, name="out_norm")(x)
        # Apply output layer - Shard parameters over model axis.
        dense_fn = shard_module_params(
            nn.Dense,
            axis_name=self.config.parallel.model_axis_name,
            min_weight_size=self.config.parallel.fsdp_min_weight_size,
        )
        # In Llama, the output layer is in bfloat16 and only the logits are casted up to float32.
        x = dense_fn(
            features=self.config.vocab_size,
            kernel_init=small_init(self.config.embedding_dim),
            use_bias=False,
            dtype=jnp.bfloat16,
            name="out_dense",
        )(x)
        x = x.astype(jnp.float32)
        return x
