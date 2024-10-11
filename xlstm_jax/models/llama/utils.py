from collections.abc import Callable

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
