#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable

import jax
from flax import linen as nn

from xlstm_jax.distributed import shard_module_params
from xlstm_jax.models.configs import ParallelConfig


def prepare_module(
    layer: Callable[..., nn.Module], layer_name: str, config: ParallelConfig | None
) -> Callable[..., nn.Module]:
    """
    Remats and shards layer if needed.

    This function wraps the layer function in a remat and/or sharding function if its layer name is present in the
    remat and fsdp configuration, respectively.

    Args:
        layer: The layer to prepare.
        layer_name: The name of the layer.
        config: The configuration to use.

    Returns:
        The layer with remat and sharding applied if needed.
    """
    if config is None:
        return layer
    # Shard parameters over model axis.
    #  Performed before remat, such that the gathered parameters would not be kept under remat.
    if config.fsdp_modules is not None and layer_name in config.fsdp_modules:
        layer = shard_module_params(
            layer,
            axis_name=config.fsdp_axis_name,
            min_weight_size=config.fsdp_min_weight_size,
            gather_dtype=config.fsdp_gather_dtype,
            grad_scatter_dtype=config.fsdp_grad_scatter_dtype,
        )
    if config.remat is not None and layer_name in config.remat:
        layer = nn.remat(layer, prevent_cse=False)
    return layer


def soft_cap_logits(logits: jax.Array, cap_value: float | jax.Array) -> jax.Array:
    """
    Soft caps logits to a value.

    Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
    and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
    https://arxiv.org/abs/2408.00118

    Args:
        logits: The logits to cap.
        cap_value: The value to cap logits to. If None, no cap is applied.

    Returns:
        The capped logits.
    """
    if cap_value is None:
        return logits
    return cap_value * nn.tanh(logits / cap_value)
