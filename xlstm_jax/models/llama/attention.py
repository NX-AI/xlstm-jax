from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental.pallas.ops.gpu.attention import mha as mha_triton

from xlstm_jax.models.configs import ParallelConfig, SubModelConfig
from xlstm_jax.models.shared import small_init, wang_init

AttentionBackend = Literal["xla", "pallas_triton", "cudnn"]


def precompute_freqs(
    feat_dim: int,
    pos_idx: jax.Array | None = None,
    max_length: int | None = None,
    theta: float = 1e4,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the sine and cosine frequencies for the rotary embeddings.

    Args:
        feat_dim: Feature dimension of the input.
        pos_idx: Positional indices of the tokens in the input sequence. If None, uses an arange up to max_length.
        max_length: Maximum length of the input sequence. Only used if pos is None.
        theta: Theta parameter for the wave length calculation.
        dtype: Data type of the returned frequencies.

    Returns:
        Tuple of the sine and cosine frequencies, shape (B, S, D//2). If pos_idx is None, shape is (1, S, D//2).
    """
    if pos_idx is None:
        assert max_length is not None, "If pos_idx is None, max_length must be specified."
        t = jnp.arange(max_length, dtype=jnp.float32)[None, :]
    else:
        t = pos_idx.astype(jnp.float32)
    freqs = 1.0 / (theta ** (jnp.arange(0.0, float(feat_dim), 2.0, dtype=jnp.float32)[: (feat_dim // 2)] / feat_dim))
    freqs = jax.vmap(jnp.outer, in_axes=(0, None))(t, freqs)
    freqs_cos = jnp.cos(freqs).astype(dtype)
    freqs_sin = jnp.sin(freqs).astype(dtype)
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
        xq: Array containing the query features of shape (B, S, NH, DHQK).
        xk: Array containing the key features of shape (B, S, NH, DHQK).
        freqs_sin: Sine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
        freqs_cos: Cosine frequencies for the rotary embeddings. If None, computes them based on the shape of xq.
        theta: Theta parameter for calculating the frequencies.

    Returns:
        Tuple of the query and key features with the rotary embeddings applied.
    """
    if freqs_sin is None or freqs_cos is None:
        freqs_sin, freqs_cos = precompute_freqs(xq.shape[-1], max_length=xq.shape[-2], theta=theta, dtype=xq.dtype)

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = jnp.moveaxis(xq.reshape(xq.shape[:-1] + (-1, 2)), -1, 0)
    xk_r, xk_i = jnp.moveaxis(xk.reshape(xk.shape[:-1] + (-1, 2)), -1, 0)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = jnp.broadcast_to(freqs_cos[:, :, None, :], xq_r.shape).astype(xq.dtype)
    freqs_sin = jnp.broadcast_to(freqs_sin[:, :, None, :], xq_r.shape).astype(xq.dtype)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = jax.lax.collapse(jnp.stack([xq_out_r, xq_out_i], axis=-1), 3)
    xk_out = jax.lax.collapse(jnp.stack([xk_out_r, xk_out_i], axis=-1), 3)

    return xq_out, xk_out


def segment_mask(segment_ids: jax.Array) -> jax.Array:
    """
    Create a mask for the self attention module based on the segment IDs.

    Args:
        segment_ids: Segment IDs for the input tensor. The attention weights between elements of different segments
            is set to zero. Shape (B, S).

    Returns:
        Boolean tensor of shape (B, 1, S, S).
    """
    mask = jnp.equal(segment_ids[:, None, :], segment_ids[:, :, None]).astype(jnp.bool_)
    mask = jnp.expand_dims(mask, axis=1)
    return mask


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
    dtype: str = "float32"
    """Data type of the activations in the network."""
    attention_backend: AttentionBackend = "xla"
    """Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels are used.
    cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on all backends."""
    causal: bool = True
    """Whether to use causal attention masking for the self attention module."""
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
        freqs: tuple[jax.Array, jax.Array] | None = None,
        segment_ids: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        """
        Apply multi-head self attention to the input.

        Args:
            x: Input tensor of shape (B, S, D).
            freqs: Tuple of sine and cosine frequencies for the rotary embeddings. If None, calculates the frequencies
                based on the input shape.
            segment_ids: Segment IDs for the input tensor. The attention weights between elements of different segments
                is set to zero. If None, all elements are treated as belonging to the same segment, i.e. no masking.
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
            dtype=self.config._dtype,
            name="dense_q",
        )(x)
        k = nn.DenseGeneral(
            features=(num_heads, head_dim),
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config._dtype,
            name="dense_k",
        )(x)
        v = nn.DenseGeneral(
            features=(num_heads, head_dim),
            use_bias=self.config.use_bias,
            kernel_init=small_init(embed_dim),
            dtype=self.config._dtype,
            name="dense_v",
        )(x)

        # Normalize qk
        if self.config.qk_norm:
            q = nn.RMSNorm(name="q_norm", dtype=self.config._dtype)(q)
            k = nn.RMSNorm(name="k_norm", dtype=self.config._dtype)(k)

        # Apply rotary embeddings
        if freqs is None:
            freqs = precompute_freqs(head_dim, max_length=seqlen, dtype=q.dtype)
        q, k = apply_rotary_emb(q, k, *freqs)

        # Compute attention
        attn_out = multihead_attention(
            q,
            k,
            v,
            segment_ids=segment_ids,
            causal=self.config.causal,
            backend=self.config.attention_backend,
        )

        # Output projection
        out = nn.DenseGeneral(
            features=embed_dim,
            axis=(-2, -1),
            use_bias=self.config.use_bias,
            kernel_init=wang_init(embed_dim, 2 * self.config.num_layers),
            dtype=self.config._dtype,
            name="dense_out",
        )(attn_out)
        out = nn.Dropout(rate=self.config.dropout_rate)(out, deterministic=not train)
        return out


def multihead_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array | None = None,
    causal: bool = True,
    qk_scale: float | None = None,
    backend: AttentionBackend = "xla",
) -> jax.Array:
    """
    Compute multi-head self attention.

    Args:
        q: Query tensor of shape (B, S, NH, DHQK).
        k: Key tensor of shape (B, S, NH, DHQK).
        v: Value tensor of shape (B, S, NH, DHV).
        segment_ids: Segment IDs for the input tensor. The attention weights between elements of different segments
            is set to zero. If None, all elements are treated as belonging to the same segment, i.e. no masking.
        causal: Whether to use causal attention masking for the self attention module.
        qk_scale: Scaling factor for the query-key logits. If None, defaults to 1/sqrt(DHQK). The scaling factor is
            applied to the query tensor before the dot product.
        backend: Which backend to use for the attention module. If triton or cudnn, respective Flash Attention kernels
            are used. cudnn is only supported for GPU backends, pallas_triton for both CPU and GPU backends, and xla on
            all backends.

    Returns:
        Output tensor of shape (B, S, NH, DHV).
    """
    if qk_scale is None:
        qk_scale = q.shape[2] ** -0.5

    # Apply scaling. We apply it here to avoid potential overflows in fp16 in the dot product.
    q = q * qk_scale

    if backend == "pallas_triton":
        return mha_triton(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sm_scale=1.0,
            causal=causal,
            interpret=jax.default_backend() == "cpu",
        )
    elif backend in ["xla", "cudnn"]:
        mask = segment_mask(segment_ids) if segment_ids is not None else None
        return jax.nn.dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            scale=1.0,
            is_causal=causal,
            implementation=backend,
        )
    else:
        raise ValueError(f"Unknown attention backend {backend}.")
