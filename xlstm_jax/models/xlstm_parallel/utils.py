import math
from collections.abc import Callable
from dataclasses import dataclass

from xlstm_jax.distributed.data_parallel import shard_module_params

from flax import linen as nn


@dataclass
class UpProjConfigMixin:
    proj_factor: float = None  # will be overridden by subclasses
    round_proj_up_dim_up: bool = True
    round_proj_up_to_multiple_of: int = 64

    # internal
    _proj_up_dim: int = None  # will be computed from embedding_dim and proj_factor

    def _set_proj_up_dim(self, embedding_dim: int) -> None:
        if self.proj_factor is not None and embedding_dim is not None:
            proj_up_dim = self.proj_factor * embedding_dim
            multiple_of_multiplier = proj_up_dim / self.round_proj_up_to_multiple_of
            if self.round_proj_up_dim_up:
                multiple_of_multiplier = math.ceil(multiple_of_multiplier)
            else:
                multiple_of_multiplier = math.floor(multiple_of_multiplier)

            self._proj_up_dim = int(multiple_of_multiplier * self.round_proj_up_to_multiple_of)


@dataclass
class ParallelConfig:
    data_axis_name: str = "dp"
    pipeline_axis_name: str = "pipe"
    model_axis_name: str = "tp"
    remat: list[str] | tuple[str] = ()
    fsdp_modules: list[str] | tuple[str] = ()
    fsdp_min_weight_size: int = 2**18
    tp_async_dense: bool = True


def prepare_module(
    layer: Callable[..., nn.Module], layer_name: str, config: ParallelConfig | None
) -> Callable[..., nn.Module]:
    """Remats and shards layer if needed.

    This function wraps the layer function in a remat and/or sharding function if its layer name is present in the remat and fsdp configuration, respectively.

    Args:
        layer: The layer to prepare.
        layer_name: The name of the layer.
        config: The configuration to use.

    Returns:
        The layer with remat and sharding applied if needed.
    """
    if config is None:
        return layer
    # Shard parameters over model axis. Performed before remat, such that the gathered parameters would not be kept under remat.
    if config.fsdp_modules is not None and layer_name in config.fsdp_modules:
        layer = shard_module_params(
            layer,
            axis_name=config.data_axis_name,
            min_weight_size=config.fsdp_min_weight_size,
        )
    if config.remat is not None and layer_name in config.remat:
        layer = nn.remat(layer, prevent_cse=False)
    return layer
