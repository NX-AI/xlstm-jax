from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from ...components.init import bias_linspace_init_
from ...components.ln import MultiHeadLayerNorm
from .backend import create_mlstm_backend, mLSTMBackendNameAndKwargs


@dataclass
class mLSTMCellConfig:
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1
    backend: mLSTMBackendNameAndKwargs = field(
        default_factory=lambda: mLSTMBackendNameAndKwargs(name="parallel_stabilized")
    )
    dtype: jnp.dtype = jnp.bfloat16


class mLSTMCell(nn.Module):
    config: mLSTMCellConfig

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs):
        B, S, _ = q.shape
        qkv = jnp.concatenate([q, k, v], axis=-1)

        # compute input and forget gate pre-activations  - why taking all heads as input?
        igate_preact = nn.Dense(
            features=self.config.num_heads,
            dtype=self.config.dtype,
            bias_init=nn.initializers.normal(stddev=0.1),
            kernel_init=nn.initializers.zeros,
            name="igate",
        )(qkv)
        fgate_preact = nn.Dense(
            features=self.config.num_heads,
            dtype=self.config.dtype,
            bias_init=bias_linspace_init_(3.0, 6.0),
            kernel_init=nn.initializers.zeros,
            name="fgate",
        )(qkv)

        q = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
        k = k.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
        v = v.transpose(0, 2, 1, 3)  # (B, NH, S, DH)

        igate_preact = igate_preact.transpose(0, 2, 1)[..., None]  # (B, NH, S, 1)
        fgate_preact = fgate_preact.transpose(0, 2, 1)[..., None]  # (B, NH, S, 1)

        backend_fn = create_mlstm_backend(self.config)
        h_state = backend_fn(q, k, v, igate_preact, fgate_preact)

        h_state_norm = MultiHeadLayerNorm(weight=True, bias=False, dtype=self.config.dtype, name="outnorm")(h_state)
        h_state_norm = h_state_norm.transpose(0, 2, 1, 3).reshape(B, S, -1)
        return h_state_norm
