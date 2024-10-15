"""Register all config dataclasses in the project to Hydra's ConfigStore and define hydra config schemas."""

import pathlib
from dataclasses import MISSING, dataclass

from hydra.core.config_store import ConfigStore

from xlstm_jax.dataset.configs import DataConfig, HFHubDataConfig, HFLocalDataConfig, SyntheticDataConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.trainer.base.trainer import TrainerConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.callbacks.lr_monitor import LearningRateMonitorConfig
from xlstm_jax.trainer.callbacks.profiler import JaxProfilerConfig
from xlstm_jax.trainer.logger.base_logger import LoggerConfig
from xlstm_jax.trainer.optimizer.optimizer import OptimizerConfig
from xlstm_jax.trainer.optimizer.scheduler import SchedulerConfig


@dataclass
class QuickHackModelConfig:
    """TODO: This class will be deleted once Hydra instantiate or our own Registry is used."""

    name: str
    vocab_size: int
    embedding_dim: int
    num_blocks: int
    context_length: int
    tie_weights: bool
    add_embedding_dropout: bool
    add_post_blocks_norm: bool
    scan_blocks: bool
    norm_eps: float
    norm_type: str
    dtype: str

    # mlstm_block_config
    add_post_norm: bool

    # mlstm_layer_config
    layer_type: str
    num_heads: int

    # mlstm_cell_config
    gate_dtype: str
    backend: str
    igate_bias_init_range: float
    add_qk_norm: bool
    cell_norm_type: str
    cell_norm_eps: float

    # feedforward_config
    proj_factor: float
    act_fn: str
    ff_type: str
    ff_dtype: str

    # Extra Llama params
    head_dim: int

    def __post_init__(self):
        """Once this class is removed due to Hydra instantiate or our own Registry, make sure that the
        allowed backends are checked, similar to this"""
        allowed_backends = ["parallel_stabilized", "fwbw_stabilized", "triton_kernels", "attention"]
        assert self.backend in allowed_backends


@dataclass
class BaseSyntheticDataConfig(SyntheticDataConfig):
    data_config_type: str = "synthetic"


@dataclass
class BaseHFHubDataConfig(HFHubDataConfig):
    data_config_type: str = "huggingface_hub"


@dataclass
class BaseHFLocalDataConfig(HFLocalDataConfig):
    data_config_type: str = "huggingface_local"


@dataclass
class BaseLoggerConfig(LoggerConfig):
    # For now, the parameters for the sub-configs are also defined here.
    # Will very likely be changed once we use Hydra instantiate or our own Registry.
    loggers_to_use: list[str] = MISSING

    # FileLoggerConfig
    file_logger_log_dir: str = MISSING
    file_logger_config_format: str = MISSING

    # TensorBoardLoggerConfig
    tb_log_dir: str = MISSING
    tb_flush_secs: int = MISSING

    # WandBLoggerConfig
    wb_project: str = MISSING
    wb_entity: str = MISSING
    wb_name: str = MISSING
    wb_tags: list[str] = MISSING


@dataclass
class Config:
    """The base config class."""

    parallel: ParallelConfig = MISSING
    data: DataConfig = MISSING
    model: QuickHackModelConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    checkpointing: ModelCheckpointConfig = MISSING
    lr_monitor: LearningRateMonitorConfig = MISSING
    profiling: JaxProfilerConfig = MISSING
    logger: LoggerConfig = MISSING
    trainer: TrainerConfig = MISSING

    device: str = MISSING
    device_count: int = MISSING
    batch_size_per_device: int = MISSING
    global_batch_size: int = MISSING
    lr: float = MISSING
    context_length: int = MISSING
    num_epochs: int = MISSING
    num_train_steps: int
    log_path: pathlib.Path = MISSING
    base_dir: pathlib.Path = MISSING
    task_name: str = MISSING
    logging_name: str = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()
    # Register base config
    cs.store(name="config_schema", node=Config)

    # Register main configs
    cs.store(name="parallel_schema", group="parallel", node=ParallelConfig)
    cs.store(name="synthetic_data_schema", group="data", node=BaseSyntheticDataConfig)
    cs.store(name="huggingface_hub_data_schema", group="data", node=BaseHFHubDataConfig)
    cs.store(name="huggingface_local_data_schema", group="data", node=BaseHFLocalDataConfig)
    cs.store(name="model_schema", group="model", node=QuickHackModelConfig)
    cs.store(name="scheduler_schema", group="scheduler", node=SchedulerConfig)
    cs.store(name="optimizer_schema", group="optimizer", node=OptimizerConfig)

    # callbacks:
    cs.store(name="model_checkpointing_schema", group="checkpointing", node=ModelCheckpointConfig)
    cs.store(name="lr_monitor_config_schema", group="lr_monitor", node=LearningRateMonitorConfig)
    cs.store(name="jax_profiling_schema", group="profiling", node=JaxProfilerConfig)

    # logger
    cs.store(name="logger_schema", group="logger", node=BaseLoggerConfig)

    # trainer
    cs.store(name="trainer_schema", group="trainer", node=TrainerConfig)
