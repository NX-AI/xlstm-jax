from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from .init import small_init


@dataclass
class LinearHeadwiseExpandConfig:
    in_features: int = 0
    # this is the number of heads that the in_features are split into
    # if num_heads=1, this is a normal linear layer
    # if num_heads>1, the in_features are split into num_heads and each head is projected separately
    # if num_heads=in_features, each feature is projected separately
    num_heads: int = -1
    expand_factor_up: float = 1

    # this is internally computed
    # but can be overwritten if you want to use a different output dimension
    # if > 0 the expand factor is ignored
    _out_features: int = -1

    bias: bool = True
    trainable_weight: bool = True
    trainable_bias: bool = True
    dtype: Any = jnp.float32

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert self.in_features % self.num_heads == 0, "in_features must be a multiple of num_heads"

        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.

    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    config: LinearHeadwiseExpandConfig
    kernel_init: Any = None
    bias_init: callable = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_features = x.shape[-1]
        in_features_per_head = in_features // self.config.num_heads
        out_features_per_head = self.config._out_features // self.config.num_heads

        if self.kernel_init is None:
            # From PyTorch code.
            kernel_init = small_init(in_features_per_head)
        else:
            # In general, all initializers will be overwritten.
            kernel_init = self.kernel_init

        weight = self.param(
            "kernel",
            kernel_init,
            (self.config.num_heads, out_features_per_head, in_features_per_head),
        )
        if not self.config.trainable_weight:
            weight = jax.lax.stop_gradient(weight)
        weight = weight.astype(self.config.dtype)

        x = x.reshape(*x.shape[:-1], self.config.num_heads, in_features_per_head)
        x = jnp.einsum("...hd,hod->...ho", x, weight)
        x = x.reshape(*x.shape[:-2], -1)
        if self.config.bias:
            bias = self.param("bias", self.bias_init, (self.config._out_features,))
            if not self.config.trainable_bias:
                bias = jax.lax.stop_gradient(bias)
            bias = bias.astype(self.config.dtype)
            bias = jnp.broadcast_to(bias, x.shape)
            x = x + bias
        return x

    def extra_repr(self):
        return (
            f"in_features={self.config.in_features}, "
            f"num_heads={self.config.num_heads}, "
            f"expand_factor_up={self.config.expand_factor_up}, "
            f"bias={self.config.bias}, "
            f"trainable_weight={self.config.trainable_weight}, "
            f"trainable_bias={self.config.trainable_bias}, "
        )
