from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import small_init, wang_init


def create_causal_mask(seqlen: int, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """
    Create a causal mask for the given sequence length.

    Args:
        seqlen: Length of the sequence.
        dtype: Data type of the mask.

    Returns:
        Causal mask of shape (1, 1, seqlen, seqlen).
    """
    seq = jnp.arange(seqlen)
    mask = (seq[:, None] >= seq[None, :]).astype(dtype)
    return mask[None, None, :, :]


def precompute_freqs(feat_dim: int, max_length: int, theta: float = 1e4):
    """
    Compute the sine and cosine frequencies for the rotary embeddings.

    Args:
        feat_dim: Feature dimension of the input.
        max_length: Maximum length of the input sequence.
        theta: Theta parameter for the wave length calculation.

    Returns:
        Tuple of the sine and cosine frequencies.
    """
    freqs = 1.0 / (theta ** (jnp.arange(0.0, float(feat_dim), 2.0)[: (feat_dim // 2)] / feat_dim))
    t = jnp.arange(max_length)
    freqs = jnp.outer(t, freqs)
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)
    return freqs_sin, freqs_cos


def apply_rotary_emb(
    xq: jax.Array,
    xk: jax.Array,
    freqs_sin: jax.Array | None = None,
    freqs_cos: jax.Array | None = None,
    theta: float = 1e4,
) -> tuple[jax.Array, jax.Array]:
    """
    Apply the rotary embeddings to the queries and keys.

    Args:
        xq: Array containing the query features of shape (B, NH, S, DHQK).
        xk: Array containing the key features of shape (B, NH, S, DHQK).
        freqs_sin: Sine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
        freqs_cos: Cosine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
        theta: Theta parameter for calculating the frequencies.

    Returns:
        Tuple of the query and key features with the rotary embeddings applied.
    """
    if freqs_sin is None or freqs_cos is None:
        freqs_sin, freqs_cos = precompute_freqs(xq.shape[-1], xq.shape[-2], theta=theta)

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = jnp.moveaxis(xq.reshape(xq.shape[:-1] + (-1, 2)), -1, 0)
    xk_r, xk_i = jnp.moveaxis(xk.reshape(xk.shape[:-1] + (-1, 2)), -1, 0)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = freqs_cos[None, :, None, :]
    freqs_sin = freqs_sin[None, :, None, :]

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = jax.lax.collapse(jnp.stack([xq_out_r, xq_out_i], axis=-1), 3)
    xk_out = jax.lax.collapse(jnp.stack([xk_out_r, xk_out_i], axis=-1), 3)

    return xq_out, xk_out


@dataclass
class SelfAttentionConfig(SubModelConfig):
    """
    Configuration for the self attention module.
    """

    head_dim: int = 64
    """Dimension of the attention heads. Number of heads is inferred from the head and embedding dimensions."""
    qk_norm: bool = False
    """Whether to apply RMSNorm to the query and key tensors."""
    use_bias: bool = False
    """Whether to use bias in the linear layers of the self attention module."""
    dropout_rate: float = 0.0
    """Dropout rate for the self attention module. Only applied during training."""
    num_layers: int = 12
    """Number of layers in the Llama model. Used for initialization."""
    dtype: jnp.dtype = jnp.float32
    """Data type of the activations in the network."""
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    """Parallel configuration."""


class SelfAttention(nn.Module):
    """
    Self attention module with support for rotary embeddings.

    Args:
        config: Configuration for the self attention module.
    """

    config: SelfAttentionConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        freqs: tuple[jax.Array, jax.Array] | None = None,
        train: bool = False,
    ) -> jax.Array:
        """
        Apply multi-head self attention to the input.

        Args:
            x: Input tensor of shape (B, S, D).
            mask: Mask tensor of shape (B, S, S) with dtype bool. Where False indicates masked positions for which the
                attention logits are overwritten to -inf. If None, no mask is applied.
            freqs: Tuple of sine and cosine frequencies for the rotary embeddings. If None, calculates the frequencies
                based on the input shape.
            train: Whether the model is in training mode or not. If True, applies dropout to the output.

        Returns:
            Output tensor of shape (B, S, D).
        """
        assert self.config.parallel.model_axis_size == 1, "Feedforward network does not support model parallelism yet."
        _, seqlen, embed_dim = x.shape
        head_dim = self.config.head_dim
        assert (
            embed_dim % head_dim == 0
        ), f"Embedding dimension must be divisible by the head dimension, got {embed_dim=} and {head_dim=}."
        num_heads = embed_dim // head_dim

        # QKV Layers
        q = nn.DenseGeneral(
            features=(num_heads, head_dim),
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config.dtype,
            name="dense_q",
        )(x)
        k = nn.DenseGeneral(
            features=(num_heads, head_dim),
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config.dtype,
            name="dense_k",
        )(x)
        v = nn.DenseGeneral(
            features=(num_heads, head_dim),
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config.dtype,
            name="dense_v",
        )(x)

        # Normalize qk
        if self.config.qk_norm:
            q = nn.RMSNorm(name="q_norm", dtype=self.config.dtype)(q)
            k = nn.RMSNorm(name="k_norm", dtype=self.config.dtype)(k)

        # Apply rotary embeddings
        if freqs is None:
            freqs = precompute_freqs(head_dim, seqlen)
        q, k = apply_rotary_emb(q, k, *freqs)

        # Swap head and sequence dimensions
        q = q.swapaxes(1, 2)  # (B, H, S, D)
        k = k.swapaxes(1, 2)  # (B, H, S, D)
        v = v.swapaxes(1, 2)  # (B, H, S, D)

        # Compute attention
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        attn_logits = attn_logits / jnp.sqrt(head_dim)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, jnp.finfo(attn_logits.dtype).min)
        # Upcast softmax precision to float32
        attn_probs = nn.softmax(attn_logits.astype(jnp.float32), axis=-1).astype(self.config.dtype)
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        attn_out = attn_out.swapaxes(1, 2)  # (B, S, H, D)

        # Output projection
        out = nn.DenseGeneral(
            features=embed_dim,
            axis=(-2, -1),
            use_bias=self.config.use_bias,
            kernel_init=wang_init(embed_dim, 2 * self.config.num_layers),
            dtype=self.config.dtype,
            name="dense_out",
        )(attn_out)
        out = nn.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
        return out
