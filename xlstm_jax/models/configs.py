from collections.abc import Sequence
from dataclasses import dataclass

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=True)
class ParallelConfig(ConfigDict):
    data_axis_size: int = -1
    fsdp_axis_size: int = 1
    pipeline_axis_size: int = 1
    model_axis_size: int = 1
    data_axis_name: str = "dp"
    fsdp_axis_name: str = "fsdp"
    pipeline_axis_name: str = "pipe"
    model_axis_name: str = "tp"
    remat: Sequence[str] = ()
    fsdp_modules: Sequence[str] = ()
    fsdp_min_weight_size: int = 2**18
    tp_async_dense: bool = True


@dataclass(kw_only=True, frozen=True)
class ModelConfig(ConfigDict):
    """Base class for model configurations."""

    model_class: callable
    parallel: ParallelConfig
    model_config: ConfigDict | None = None


@dataclass
class SubModelConfig:
    """Sub-model configuration.

    This class is currently a quick fix to allow for post-init style model configs,
    like the xlstm-clean we ported from the original xlstm codebase. Once the config
    system is more mature, we should remove this and all becomes a subclass of
    ModelConfig.
    """

    def to_dict(self):
        """Converts the config to a dictionary.

        Helpful for saving to disk or logging.
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict) or hasattr(v, "to_dict"):
                d[k] = v.to_dict()
            elif isinstance(v, (tuple, list)):
                d[k] = tuple([x.to_dict() if isinstance(x, ConfigDict) or hasattr(v, "to_dict") else x for x in v])
            else:
                d[k] = v
        return d
