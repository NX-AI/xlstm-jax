#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import linen as nn

from xlstm_jax.models.shared import uniform_init


@dataclass
class CausalConv1dConfig:
    feature_dim: int = None  # F
    kernel_size: int = 4
    causal_conv_bias: bool = True
    channel_mixing: bool = False
    conv1d_kwargs: dict = field(default_factory=dict)
    dtype: str = "bfloat16"

    def __post_init__(self):
        assert self.kernel_size >= 0, "kernel_size must be >= 0"

    @property
    def _dtype(self) -> jnp.dtype:
        """
        Returns the real dtype instead of the str from configs.

        Returns:
            The jnp dtype corresponding to the string value.
        """
        return getattr(jnp, self.dtype)


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
        pad = self.config.kernel_size - 1  # padding of this size assures temporal causality.
        fan_in = self.config.kernel_size * (x.shape[-1] // groups)
        x = nn.Conv(
            features=self.config.feature_dim,
            kernel_size=(self.config.kernel_size,),
            kernel_init=jax.nn.initializers.variance_scaling(2.0 / (1 + 5), "fan_in", "uniform"),  # PyTorch default
            bias_init=uniform_init(min_val=-1.0 / jnp.sqrt(fan_in), max_val=1.0 / jnp.sqrt(fan_in)),
            feature_group_count=groups,
            padding=[(pad, 0)],
            use_bias=self.config.causal_conv_bias,
            dtype=self.config._dtype,
            name="conv",
            **self.config.conv1d_kwargs,
        )(x)
        return x
