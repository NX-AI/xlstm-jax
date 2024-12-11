#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax.numpy as jnp
from flax import linen as nn


def LayerNorm(
    weight: bool = True,
    bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    **kwargs,
):
    return nn.LayerNorm(epsilon=eps, use_bias=bias, use_scale=weight, dtype=dtype, **kwargs)


def MultiHeadLayerNorm(
    weight: bool = True,
    bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    axis: int = 1,
    **kwargs,
):
    return nn.vmap(
        nn.LayerNorm,
        variable_axes={"params": 0},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": True},
    )(epsilon=eps, use_bias=bias, use_scale=weight, dtype=dtype, **kwargs)
