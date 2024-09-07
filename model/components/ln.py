# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import jax
import jax.numpy as jnp
from flax import linen as nn


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    weight: bool = True
    bias: bool = False
    eps: float = 1e-5
    residual_weight: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.LayerNorm(
            epsilon=self.eps,
            use_bias=self.bias,
            use_scale=self.weight,
            dtype=self.dtype,
        )(x)


def MultiHeadLayerNorm(*args, axis: int = 1, **kwargs):
    return nn.vmap(
        LayerNorm,
        variable_axes={"params": None},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": False},
    )(*args, **kwargs)
