# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbinian Pöppel
import jax
import jax.numpy as jnp
from flax import linen as nn


def LayerNorm(weight: bool = True, bias: bool = False, eps: float = 1e-5, residual_weight: bool = True, dtype: jnp.dtype = jnp.bfloat16, **kwargs):
    return nn.LayerNorm(
        epsilon=eps,
        use_bias=bias,
        use_scale=weight,
        dtype=dtype,
        **kwargs
    )


def MultiHeadLayerNorm(weight: bool = True, bias: bool = False, eps: float = 1e-5, residual_weight: bool = True, dtype: jnp.dtype = jnp.bfloat16, axis: int = 1, **kwargs):
    return nn.vmap(
        nn.LayerNorm,
        variable_axes={"params": None},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": False},
    )(epsilon=eps,
        use_bias=bias,
        use_scale=weight,
        dtype=dtype, **kwargs)
