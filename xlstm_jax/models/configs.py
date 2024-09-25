from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=True)
class ParallelConfig(ConfigDict):
    """
    Configuration for parallelism.

    Attributes:
        data_axis_size: Size of the data axis. If -1, it will be inferred by the
            number of available devices.
        fsdp_axis_size: Size of the FSDP axis. If -1, it will be inferred by the
            number of available devices.
        pipeline_axis_size: Size of the pipeline axis. If -1, it will be
            inferred by the number of available devices.
        model_axis_size: Size of the model axis. If -1, it will be inferred by
            the number of available devices.
        data_axis_name: Name of the data axis.
        fsdp_axis_name: Name of the FSDP axis.
        pipeline_axis_name: Name of the pipeline axis.
        model_axis_name: Name of the model axis.
        remat: Module names on which we apply activation checkpointing /
            rematerialization.
        fsdp_modules: Module names on which we apply FSDP sharding.
        fsdp_min_weight_size: Minimum size of a parameter to be sharded with FSDP.
        fsdp_gather_dtype: The dtype to cast the parameters to before gathering
            with FSDP. If None, no casting is performed and parameters are gathered
            in original precision (e.g. float32).
        fsdp_grad_scatter_dtype: The dtype to cast the gradients to before
            scattering. If None, the dtype of the parameters is used.
        tp_async_dense: Whether to use asynchronous tensor parallelism for dense layers.
            Default to False, as on local hardware, ppermute communication introduces
            large overhead.
    """

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
    fsdp_gather_dtype: Literal["float32", "bfloat16", "float16"] | None = None
    fsdp_grad_scatter_dtype: Literal["float32", "bfloat16", "float16"] | None = None
    tp_async_dense: bool = False


@dataclass(kw_only=True, frozen=True)
class ModelConfig(ConfigDict):
    """Base class for model configurations."""

    model_class: callable
    parallel: ParallelConfig
    model_config: ConfigDict | None = None


@dataclass
class SubModelConfig:
    """
    Sub-model configuration.

    This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
    the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
    of ModelConfig.
    """

    def to_dict(self):
        """
        Converts the config to a dictionary.

        Helpful for saving to disk or logging.
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict) or hasattr(v, "to_dict"):
                d[k] = v.to_dict()
            elif isinstance(v, (tuple, list)):
                d[k] = tuple([x.to_dict() if isinstance(x, ConfigDict) or hasattr(v, "to_dict") else x for x in v])
            elif isinstance(v, (int, float, str, bool)):
                d[k] = v
            else:
                d[k] = str(v)
        return d
