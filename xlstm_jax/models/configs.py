import importlib
import inspect
import json
import re
from dataclasses import dataclass, field

from xlstm_jax.configs import ConfigDict


@dataclass(kw_only=True, frozen=False)
class ParallelConfig:
    """Configuration for parallelism."""

    data_axis_size: int = -1
    """Size of the data axis. If -1, it will be inferred by the number of available devices."""
    fsdp_axis_size: int = 1
    """Size of the FSDP axis. If -1, it will be inferred by the number of available devices."""
    pipeline_axis_size: int = 1
    """Size of the pipeline axis. If -1, it will be inferred by the number of available devices."""
    model_axis_size: int = 1
    """Size of the model axis. If -1, it will be inferred by the number of available devices."""
    data_axis_name: str = "dp"
    """Name of the data axis."""
    fsdp_axis_name: str = "fsdp"
    """Name of the FSDP axis."""
    pipeline_axis_name: str = "pipe"
    """Name of the pipeline axis."""
    model_axis_name: str = "tp"
    """Name of the model axis."""
    remat: list[str] = field(default_factory=lambda: [])
    """Module names on which we apply activation checkpointing / rematerialization."""
    fsdp_modules: list[str] = field(default_factory=lambda: [])
    """Module names on which we apply FSDP sharding."""
    fsdp_min_weight_size: int = 2**18
    """Minimum size of a parameter to be sharded with FSDP."""
    fsdp_gather_dtype: str | None = None
    """The dtype to cast the parameters to before gathering with FSDP. If `None`, no casting is performed and parameters
    are gathered in original precision (e.g. `float32`)."""
    fsdp_grad_scatter_dtype: str | None = None
    """The dtype to cast the gradients to before scattering. If `None`, the dtype of the parameters is used."""
    tp_async_dense: bool = False
    """Whether to use asynchronous tensor parallelism for dense layers. Default to `False`, as on local hardware,
    ppermute communication introduces large overhead."""

    def __post_init__(self):
        _allowed_fsdp_dtypes = ["float32", "bfloat16", "float16"]

        if self.fsdp_gather_dtype is not None:
            assert self.fsdp_gather_dtype in _allowed_fsdp_dtypes
        if self.fsdp_grad_scatter_dtype is not None:
            assert self.fsdp_grad_scatter_dtype in _allowed_fsdp_dtypes


@dataclass(kw_only=True, frozen=False)
class ModelConfig(ConfigDict):
    """Base class for model configurations."""

    model_class: callable
    """Model class."""
    parallel: ParallelConfig
    """Parallelism configuration."""
    model_config: ConfigDict | None = None
    """Model configuration."""

    @staticmethod
    def from_metadata(metadata_content: str) -> "ModelConfig":
        """
        Creates a model config from a metadata file content.

        Args:
            metadata_content: Content of the metadata file, currently in JSON format.

        Returns:
            Tuple of the model_class and the model configuration parsed into a nested ModelConfig format.
        """
        cfg_dict = json.loads(metadata_content)
        model_class_path = cfg_dict["model"]["model_class"]
        module_path = ".".join(model_class_path.split(".")[:-1])
        module = importlib.import_module(module_path)
        model_class = getattr(module, model_class_path.split(".")[-1])
        model_class_cfg = inspect.get_annotations(model_class)["config"]
        model_cfg = ConfigDict.from_dict(model_class_cfg, data=cfg_dict["model"]["model_config"])
        if isinstance(cfg_dict["model"]["parallel"], str):
            parallel_args = re.match(r"ParallelConfig\((.*)\)", cfg_dict["model"]["parallel"]).group(1)
            parallel_args_json = (
                "{"
                + re.sub(
                    r"([a-zA-Z_]+)\=",
                    r'"\1": ',
                    parallel_args.replace("'", '"')
                    .replace("(", "[")
                    .replace(")", "]")
                    .replace("None", "null")
                    .replace("True", "true")
                    .replace("False", "false"),
                )
                + "}"
            )
            parallel_cfg = json.loads(parallel_args_json)
        else:
            parallel_cfg = cfg_dict["model"]["parallel"]
        if parallel_cfg["fsdp_gather_dtype"] == "None":
            parallel_cfg["fsdp_gather_dtype"] = None
        if parallel_cfg["fsdp_grad_scatter_dtype"] == "None":
            parallel_cfg["fsdp_grad_scatter_dtype"] = None
        parallel = ParallelConfig(**parallel_cfg)
        return ModelConfig(model_class=model_class, parallel=parallel, model_config=model_cfg)


@dataclass
class SubModelConfig:
    """
    Sub-model configuration.

    This class is currently a quick fix to allow for post-init style model configs, like the xlstm-clean we ported from
    the original xlstm codebase. Once the config system is more mature, we should remove this and all becomes a subclass
    of ModelConfig.
    """

    def to_dict(self) -> dict:
        """
        Converts the config to a dictionary.

        Helpful for saving to disk or logging.
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigDict) or hasattr(v, "to_dict"):
                d[k] = v.to_dict()
            elif isinstance(v, (tuple, list)):
                d[k] = tuple(x.to_dict() if isinstance(x, ConfigDict) or hasattr(v, "to_dict") else x for x in v)
            elif isinstance(v, (int, float, str, bool)):
                d[k] = v
            else:
                d[k] = str(v)
        return d
