# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck, Korbininan Pöppel
from dataclasses import dataclass
from math import sqrt

import jax
import jax.numpy as jnp
from flax import linen as nn


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

    def __post_init__(self):
        assert self.num_heads > 0, "num_heads must be set"
        assert self.num_heads <= self.in_features, "num_heads must be <= in_features"
        assert (
            self.in_features % self.num_heads == 0
        ), "in_features must be a multiple of num_heads"

        if self._out_features < 0:
            self._out_features = round(self.expand_factor_up * self.in_features)


class LinearHeadwiseExpand(nn.Module):
    """This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    config: LinearHeadwiseExpandConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        in_features = x.shape[-1]
        in_features_per_head = in_features // self.config.num_heads
        out_features_per_head = self.config._out_features // self.config.num_heads

        weight = self.param(
            "weight",
            jax.nn.initializers.normal(stddev=sqrt(2 / 5 / in_features_per_head)),
            (self.config.num_heads, out_features_per_head, in_features_per_head),
        )
        if not self.config.trainable_weight:
            weight = jax.lax.stop_gradient(weight)

        x = x.reshape(*x.shape[:-1], self.config.num_heads, in_features_per_head)
        x = jnp.einsum("...hd,hod->...ho", x, weight)
        x = x.reshape(*x.shape[:-2], -1)
        if self.config.bias:
            bias = self.param(
                "bias", jax.nn.initializers.zeros, (self.config._out_features,)
            )
            if not self.config.trainable_bias:
                bias = jax.lax.stop_gradient(bias)
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


def test_linear_headwise():
    config = LinearHeadwiseExpandConfig(in_features=4, num_heads=2, expand_factor_up=1)
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng)
    input_tensor = jax.random.normal(inp_rng, (2, 5, 4))
    model = LinearHeadwiseExpand(config)
    params = model.init(model_rng, input_tensor)
    output_tensor = model.apply(params, input_tensor)
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    input_tensor = input_tensor.at[0, 0, 2].set(-1.0)
    output_tensor_new = model.apply(params, input_tensor)
    diff = (output_tensor_new - output_tensor) != 0
    assert not jnp.any(diff[1]), "Output tensor changed unexpectedly."
    assert not jnp.any(diff[0, 1:]), "Output tensor changed unexpectedly."
    assert not jnp.any(diff[0, 0, :2]), "Output tensor changed unexpectedly."
    assert jnp.all(diff[0, 0, 3:]), "Output tensor changed unexpectedly."
    print("All tests for LinearHeadwiseExpand passed successfully.")
