import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import mLSTMBackend


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
    freqs_cos = freqs_cos[None, :, :]
    freqs_sin = freqs_sin[None, :, :]

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = jax.lax.collapse(jnp.stack([xq_out_r, xq_out_i], axis=-1), 2)
    xk_out = jax.lax.collapse(jnp.stack([xk_out_r, xk_out_i], axis=-1), 2)

    return xq_out, xk_out


def attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    attention_mask: jax.Array = None,
    qkv_dtype: jnp.dtype | None = None,
    activation_function: Literal["softmax", "sigmoid", "none"] = "softmax",
    qk_pre_activation_function: Literal["silu", "swish", "none"] = "none",
    theta: float = 1e4,
    **kwargs,
) -> jax.Array:
    """
    This is an attention backend that mimics the attention mechanism of the transformer.

    Note that no forget and input gate are applied here.

    Args:
        queries: Array containing the query features of shape (B, NH, S, DHQK).
        keys: Array containing the key features of shape (B, NH, S, DHQK).
        values: Array containing the value features of shape (B, NH, S, DHV).
        attention_mask: Array of shape (S,S) denoting the attention mask. By default, uses a causal mask which is a
            lower triangular matrix. Dtype should be bool, where False denotes masked positions.
        qkv_dtype: Dtype of the queries, keys and values. If None, uses the dtype of queries.
        activation_function: Activation function to apply on the attention logits. Softmax is performed over the key
            sequence as in default transformers. Sigmoid is applied with a bias of -log(S).
        qk_pre_activation_function: Activation function to apply on the queries and keys before computing the attention
            logits.
        theta: Theta parameter for the rotary embeddings.

    Returns:
        The output features of the attention of shape (B, NH, S, DHV).
    """
    _, context_length, key_dim = queries.shape
    if qkv_dtype is None:
        qkv_dtype = queries.dtype
        assert qkv_dtype == keys.dtype == values.dtype, (
            f"queries, keys and values must have the same dtype, got {queries.dtype} for queries,"
            f" {keys.dtype} for keys and {values.dtype} for values."
        )
    else:
        queries = queries.astype(qkv_dtype)
        keys = keys.astype(qkv_dtype)
        values = values.astype(qkv_dtype)
    assert (
        queries.shape == keys.shape and queries.shape[:-1] == values.shape[:-1]
    ), f"queries, keys and values must have fitting shapes: {queries.shape}, {keys.shape}, {values.shape}"
    if attention_mask is not None:
        assert attention_mask.shape == (
            context_length,
            context_length,
        ), f"lower_triangular_matrix must have shape (S, S), got {attention_mask.shape}"

    if qk_pre_activation_function in ("silu", "swish"):
        queries = jax.nn.swish(queries)
        keys = jax.nn.swish(keys)

    queries, keys = apply_rotary_emb(queries, keys, theta=theta)

    keys_scaled = keys / math.sqrt(key_dim)
    qk_matrix = jnp.einsum("bqh,bkh->bqk", queries, keys_scaled)  # (B, S, S)
    if attention_mask is not None:
        qk_matrix = jnp.where(attention_mask, qk_matrix, jnp.full_like(qk_matrix, -jnp.inf))
    if activation_function == "softmax":
        qk_matrix = jax.nn.softmax(qk_matrix, axis=-1)
    elif activation_function == "sigmoid":
        # Bias from https://arxiv.org/pdf/2409.04431
        qk_matrix = jax.nn.sigmoid(qk_matrix - jnp.log(context_length))
    elif activation_function == "none":
        pass
    else:
        raise ValueError(f"Unknown activation function {activation_function}")

    out = jnp.einsum("bqk,bkh->bqh", qk_matrix, values)  # (B, S, DH)
    return out


@dataclass
class mLSTMBackendAttentionConfig:
    context_length: int = -1
    activation_function: Literal["softmax", "sigmoid", "none"] = "softmax"
    """Activation function to apply on the attention logits. Softmax is performed over the key sequence
    as in default transformers. Sigmoid is applied with a bias of -log(context_length)."""
    qk_pre_activation_function: Literal["swish", "none"] = "none"
    """Activation function to apply on the queries and keys before computing the attention logits."""
    theta: float = 1e4
    """Theta parameter for the rotary embeddings."""

    def assign_model_config_params(self, model_config, *args, **kwargs):
        self.context_length = model_config.context_length


class mLSTMBackendAttention(mLSTMBackend):
    config_class = mLSTMBackendAttentionConfig

    @nn.compact
    def __call__(
        self, q: jax.Array, k: jax.Array, v: jax.Array, i: jax.Array | None = None, f: jax.Array | None = None
    ):
        """Forward pass of the attention backend."""
        del i, f
        causal_mask = jnp.tril(jnp.ones((self.config.context_length, self.config.context_length), dtype=jnp.bool_))
        return attention(
            q,
            k,
            v,
            attention_mask=causal_mask,
            activation_function=self.config.activation_function,
            qk_pre_activation_function=self.config.qk_pre_activation_function,
            theta=self.config.theta,
        )

    @property
    def can_vmap_over_heads(self) -> bool:
        """
        Whether the backend can be vmaped over the heads dimension.

        The backend is written independent of the heads dimension, and thus can be vmapped.

        Returns:
            bool: True
        """
        return True
