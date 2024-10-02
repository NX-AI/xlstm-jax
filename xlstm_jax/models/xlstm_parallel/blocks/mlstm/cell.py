from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.configs import SubModelConfig

from ...components.init import bias_linspace_init
from ...components.ln import MultiHeadLayerNorm
from ...utils import soft_cap_logits
from .backend import create_mlstm_backend, mLSTMBackend, mLSTMBackendNameAndKwargs


@dataclass
class mLSTMCellConfig(SubModelConfig):
    context_length: int = -1
    embedding_dim: int = -1
    num_heads: int = -1
    backend: mLSTMBackendNameAndKwargs = field(
        default_factory=lambda: mLSTMBackendNameAndKwargs(name="parallel_stabilized")
    )
    norm_eps: float = 1e-6
    """Epsilon value for numerical stability in layer norm."""
    dtype: jnp.dtype = jnp.bfloat16
    gate_dtype: jnp.dtype = jnp.float32
    gate_soft_cap: float | None = None
    """Soft cap for the gate pre-activations. If None, no cap is applied."""


class mLSTMCell(nn.Module):
    config: mLSTMCellConfig

    @nn.compact
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs):
        B, S, _ = q.shape
        qkv = jnp.concatenate([q, k, v], axis=-1)

        # compute input and forget gate pre-activations  - why taking all heads as input?
        with jax.named_scope("mlstm_gates"):
            igate_preact = nn.Dense(
                features=self.config.num_heads,
                dtype=self.config.gate_dtype,
                bias_init=nn.initializers.normal(stddev=0.1),
                kernel_init=nn.initializers.zeros,
                name="igate",
            )(qkv)
            fgate_preact = nn.Dense(
                features=self.config.num_heads,
                dtype=self.config.gate_dtype,
                bias_init=bias_linspace_init(3.0, 6.0),
                kernel_init=nn.initializers.zeros,
                name="fgate",
            )(qkv)
            igate_preact = soft_cap_logits(igate_preact, self.config.gate_soft_cap)
            fgate_preact = soft_cap_logits(fgate_preact, self.config.gate_soft_cap)
            self.sow("intermediates", "max_igate_preact", jnp.max(igate_preact))
            self.sow("intermediates", "max_fgate_preact", jnp.max(fgate_preact))
            self.sow("intermediates", "min_igate_preact", jnp.min(igate_preact))
            self.sow("intermediates", "min_fgate_preact", jnp.min(fgate_preact))
            self.sow("intermediates", "mean_igate_preact", jnp.mean(igate_preact))
            self.sow("intermediates", "mean_fgate_preact", jnp.mean(fgate_preact))

        q = q.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        k = k.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        v = v.reshape(B, S, self.config.num_heads, -1)  # (B, S, NH, DH)
        igate_preact = igate_preact[..., None]  # (B, S, NH, 1)
        fgate_preact = fgate_preact[..., None]  # (B, S, NH, 1)

        backend_fn: mLSTMBackend = create_mlstm_backend(self.config)

        if backend_fn.can_vmap_over_heads:
            # Vmap over the heads dimension without needing to transpose the input tensors.
            backend_fn = jax.vmap(backend_fn, in_axes=(2, 2, 2, 2, 2), out_axes=2)
            with jax.named_scope("mlstm_backend"):
                h_state = backend_fn(q, k, v, igate_preact, fgate_preact)
        else:
            # Manual transpose to work over heads.
            q = q.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            k = k.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            v = v.transpose(0, 2, 1, 3)  # (B, NH, S, DH)
            igate_preact = igate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
            fgate_preact = fgate_preact.transpose(0, 2, 1, 3)  # (B, NH, S, 1)
            with jax.named_scope("mlstm_backend"):
                h_state = backend_fn(q, k, v, igate_preact, fgate_preact)
            h_state = h_state.transpose(0, 2, 1, 3)

        h_state_norm = MultiHeadLayerNorm(
            weight=True, bias=False, dtype=self.config.dtype, name="outnorm", axis=2, eps=self.config.norm_eps
        )(h_state)
        h_state_norm = h_state_norm.reshape(B, S, -1)
        return h_state_norm
