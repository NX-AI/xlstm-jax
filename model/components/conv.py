# Copyright (c) NXAI GmbH and its affiliates 2024
# Maximilian Beck
from dataclasses import dataclass, field

from flax import linen as nn
import jax.numpy as jnp
import jax


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)
    dtype: jnp.dtype = jnp.bfloat16

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"


class CausalConv1d(nn.Module):
    config: CausalConv1dConfig
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.config.kernel_size == 0:
            return x
        if self.config.channel_mixing:
            groups = 1
        else:
            groups = x.shape[-1]
        pad = (
            self.config.kernel_size - 1
        )  # padding of this size assures temporal causality.
        x = nn.Conv(
            features=self.config.feature_dim,
            kernel_size=(self.config.kernel_size,),
            feature_group_count=groups,
            padding=[(pad, 0)],
            use_bias=self.config.causal_conv_bias,
            dtype=self.config.dtype,
            **self.config.conv1d_kwargs,
        )(x)
        return x


def test_causal_conv1d():
    config = CausalConv1dConfig(feature_dim=3, kernel_size=4, channel_mixing=False)
    rng = jax.random.PRNGKey(0)
    inp_rng, model_rng = jax.random.split(rng)
    input_tensor = jax.random.normal(inp_rng, (2, 5, 3))
    model = CausalConv1d(config)
    params = model.init(model_rng, input_tensor)
    output_tensor = model.apply(params, input_tensor)
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    input_tensor = input_tensor.at[0, 2, 0].set(-1.0)
    output_tensor_new = model.apply(params, input_tensor)
    diff = (output_tensor_new - output_tensor) != 0
    assert (
        diff.any()
    ), "Expected output to change after changing input, but it remained the same"
    assert diff[
        0, 2:, 0
    ].all(), f"Expected output to change after changing input, but it remained the same: {diff}"
    assert not diff[
        :, :2, :
    ].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"
    assert not diff[
        :, 2:, 1:
    ].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"
    assert not diff[
        1
    ].any(), f"Expected output to remain unchanged after changing input, but it changed: {diff}"
    print("All tests for CausalConv1D passed successfully.")
