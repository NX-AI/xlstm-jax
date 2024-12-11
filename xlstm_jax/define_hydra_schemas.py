#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""Register all config dataclasses in the project to Hydra's ConfigStore and define hydra config schemas."""

import pathlib
import typing
from dataclasses import MISSING, dataclass

from hydra.core.config_store import ConfigStore

from xlstm_jax.dataset.configs import (
    DataConfig,
    GrainArrayRecordsDataConfig,
    HFHubDataConfig,
    SyntheticDataConfig,
)
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.shared import InitDistribution, InitFnName
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import BackendType
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend.triton_kernels import BackendNameType
from xlstm_jax.models.xlstm_parallel.components.normalization import NormType
from xlstm_jax.trainer.base.trainer import TrainerConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.callbacks.lr_monitor import LearningRateMonitorConfig
from xlstm_jax.trainer.callbacks.profiler import JaxProfilerConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainerConfig
from xlstm_jax.trainer.logger.base_logger import LoggerConfig
from xlstm_jax.trainer.optimizer.optimizer import OptimizerConfig
from xlstm_jax.trainer.optimizer.scheduler import SchedulerConfig


@dataclass
class CombinedModelConfig:
    """This class is a flat config that combines several sub-configs."""

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
    init_distribution_embed: str
    logits_soft_cap: float
    lm_head_dtype: str
    dtype: str

    # mlstm_block_config
    add_post_norm: bool

    # mlstm_layer_config
    layer_type: str
    num_heads: int
    output_init_fn: str
    init_distribution: str
    qk_dim_factor: float
    v_dim_factor: float

    # mlstm_cell_config
    gate_dtype: str
    backend: str
    backend_name: str
    igate_bias_init_range: float
    add_qk_norm: bool
    cell_norm_type: str
    cell_norm_type_v1: str
    cell_norm_eps: float
    gate_soft_cap: float
    reset_at_document_boundaries: bool

    # feedforward_config
    proj_factor: float
    act_fn: str
    ff_type: str
    ff_dtype: str

    # Extra Llama params
    head_dim: int
    attention_backend: str
    theta: float

    def __post_init__(self):
        """make sure that the allowed backends are checked"""

        assert self.backend in typing.get_args(BackendType)
        assert self.backend_name in typing.get_args(BackendNameType)
        assert self.output_init_fn in typing.get_args(InitFnName)
        assert self.init_distribution in typing.get_args(InitDistribution)
        assert self.init_distribution_embed in typing.get_args(InitDistribution)
        assert self.norm_type in typing.get_args(NormType)
        assert self.norm_type_v1 in typing.get_args(NormType)

        # If the soft caps are set to 0.0, set them to None.
        if self.logits_soft_cap <= 0.0:
            self.logits_soft_cap = None
        if self.gate_soft_cap <= 0.0:
            self.gate_soft_cap = None


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
class DataEvalConfig:
    """Supports maximum of 5 data evaluation configurations."""

    ds1: DataConfig | None = None
    ds2: DataConfig | None = None
    ds3: DataConfig | None = None
    ds4: DataConfig | None = None
    ds5: DataConfig | None = None


@dataclass
class DataTrainConfig:
    """Supports maximum of 10 data training configurations."""

    ds1: DataConfig | None = None
    weight1: float = 1.0
    ds2: DataConfig | None = None
    weight2: float = 1.0
    ds3: DataConfig | None = None
    weight3: float = 1.0
    ds4: DataConfig | None = None
    weight4: float = 1.0
    ds5: DataConfig | None = None
    weight5: float = 1.0
    ds6: DataConfig | None = None
    weight6: float = 1.0
    ds7: DataConfig | None = None
    weight7: float = 1.0
    ds8: DataConfig | None = None
    weight8: float = 1.0
    ds9: DataConfig | None = None
    weight9: float = 1.0
    ds10: DataConfig | None = None
    weight10: float = 1.0


@dataclass
class Config:
    """The base config class."""

    parallel: ParallelConfig = MISSING
    model: CombinedModelConfig = MISSING
    scheduler: SchedulerConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    checkpointing: ModelCheckpointConfig = MISSING
    lr_monitor: LearningRateMonitorConfig = MISSING
    profiling: JaxProfilerConfig = MISSING
    logger: LoggerConfig = MISSING
    trainer: TrainerConfig = MISSING

    device: str = MISSING
    device_count: int = MISSING
    n_gpus: int = MISSING
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

    data_train: DataTrainConfig | None = None
    data_eval: DataEvalConfig | None = None


def register_configs() -> None:
    cs = ConfigStore.instance()
    # Register base config
    cs.store(name="config_schema", node=Config)

    # Register main configs
    cs.store(name="parallel_schema", group="parallel", node=ParallelConfig)
    cs.store(name="synthetic_data_schema", group="data", node=SyntheticDataConfig)
    cs.store(name="huggingface_hub_data_schema", group="data", node=HFHubDataConfig)
    cs.store(name="grain_arrayrecord_data_schema", group="data", node=GrainArrayRecordsDataConfig)
    cs.store(name="model_schema", group="model", node=CombinedModelConfig)
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
    cs.store(name="llm_trainer_schema", group="trainer", node=LLMTrainerConfig)
