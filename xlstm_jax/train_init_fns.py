import logging
import math
import os
from typing import Literal

import jax
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.dataset import (
    DataConfig,
    DataIterator,
    GrainArrayRecordsDataConfig,
    HFHubDataConfig,
    LLMBatch,
    SyntheticDataConfig,
    create_data_iterator,
    create_mixed_data_iterator,
    load_tokenizer,
)
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.llama import LlamaConfig, LlamaTransformer
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.backend import mLSTMBackendNameAndKwargs
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.components.feedforward import FeedForwardConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig, WandBLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def init_parallel(cfg: DictConfig) -> ParallelConfig:
    """Initialize configuration for parallelism.

    Args:
        cfg  Config assembled by Hydra.

    Returns:
        Initialized parallel configuration.
    """
    parallel = ParallelConfig(**cfg.parallel)

    return parallel


def init_data_iterator(
    cfg: DictConfig, mesh: Mesh
) -> tuple[DataIterator, DataIterator | dict[str, DataIterator] | None]:
    """Initialize data iterators.

    Args:
        cfg: Config assembled by Hydra.
        mesh: The jax device mesh.

    Returns:
        Training and evaluation data iterators.
    """
    if cfg.data_eval is None and cfg.data_train is None:
        # If config eval is None, we use the config class to split the data into train and eval.
        train_data_iterator = init_single_data_iterator(cfg=cfg.data, mesh=mesh, create_split="train")
        eval_data_iterator = init_single_data_iterator(cfg=cfg.data, mesh=mesh, create_split="eval")
    else:
        # Create train data iterator.
        if cfg.data_train is None:
            train_data_iterator = init_single_data_iterator(cfg=cfg.data, mesh=mesh)
        else:
            train_data_iterator = init_mixed_data_iterator(cfg=cfg.data_train, mesh=mesh)

        # Create evaluation data iterator.
        eval_data_iterator = None
        if cfg.data_eval is not None:
            eval_data_iterator = {}
            for data_config in cfg.data_eval.values():
                if data_config is not None:
                    assert data_config.name is not None, "Evaluation datasets must have a name."
                    assert (
                        data_config.name not in eval_data_iterator
                    ), f"Duplicate evaluation dataset name: {data_config.name}."
                    eval_data_iterator[data_config.name] = init_single_data_iterator(cfg=data_config, mesh=mesh)

    return train_data_iterator, eval_data_iterator


def init_single_data_iterator(
    cfg: DictConfig, mesh: Mesh, create_split: Literal["train", "eval", None] = None
) -> DataIterator:
    """Initialize a single data iterator.

    Args:
        cfg: Data configuration.
        mesh: The jax device mesh.
        create_split: Whether to create a train or eval config from the config class, using the
            `create_train_eval_configs` method. If None, the config is used as is.

    Returns:
        Data iterator.
    """

    # Check data config type.
    config_classes: dict[str, type[DataConfig]] = {
        "synthetic": SyntheticDataConfig,
        "huggingface_hub": HFHubDataConfig,
        "grain_arrayrecord": GrainArrayRecordsDataConfig,
    }
    if cfg.data_config_type not in config_classes:
        raise NotImplementedError(
            "Only synthetic, Huggingface, and ArrayRecord datasets are implemented, "
            f"got {cfg.data_config_type} in config {cfg}."
        )

    # Create data iterator.
    config_class = config_classes[cfg.data_config_type]
    cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True).copy()
    if create_split is None:
        data_config = config_class(**cfg)
        LOGGER.info(f"Data config: {data_config}.")
    else:
        # Remove keys that by default are hard-coded to train.
        for key in ["split", "shuffle_data", "drop_remainder"]:
            if key in cfg:
                cfg.pop(key)
        train_kwargs, eval_kwargs = {}, {}
        # Extract train- and eval-specific keys.
        for train_key in ["grain_packing", "batch_rampup_factors"]:
            if train_key in cfg:
                train_kwargs[train_key] = cfg.pop(train_key)
        for eval_key in ["max_steps_per_epoch"]:
            if eval_key in cfg:
                eval_kwargs[eval_key] = cfg.pop(eval_key)
        # Create train and eval configs.
        train_config, eval_config = config_class.create_train_eval_configs(
            **cfg, train_kwargs=train_kwargs, eval_kwargs=eval_kwargs
        )
        # Select the correct config.
        data_config = train_config if create_split == "train" else eval_config
        LOGGER.info(f"{create_split} data config: {data_config}.")
    data_iterator = create_data_iterator(config=data_config, mesh=mesh)
    return data_iterator


def init_mixed_data_iterator(
    cfg: DictConfig,
    mesh: Mesh,
) -> DataIterator:
    """Initialize a data iterator with mixed data sources.

    Args:
        cfg: Data configuration.
        mesh: The jax device mesh.

    Returns:
        Data iterator.
    """
    config_classes: dict[str, type[DataConfig]] = {
        "huggingface_hub": HFHubDataConfig,
        "grain_arrayrecord": GrainArrayRecordsDataConfig,
    }

    data_configs, data_weights = [], []
    for key in cfg:
        if not key.startswith("val") or cfg[key] is None:
            continue
        data_config = cfg[key]
        if data_config.data_config_type not in config_classes:
            raise NotImplementedError(
                "Only Huggingface and ArrayRecord datasets are supported for mixed datasets, "
                f"got {data_config.data_config_type} in config {data_config}."
            )
        config_class = config_classes[data_config.data_config_type]
        data_config = OmegaConf.to_container(data_config, resolve=True, enum_to_str=True).copy()
        data_config = config_class(**data_config)
        LOGGER.info(f"Data config: {data_config}.")
        data_configs.append(data_config)
        data_weights.append(float(cfg[key.replace("val", "weight")]))
    assert len(data_configs) > 0, "No datasets found in the config."

    data_iterator = create_mixed_data_iterator(configs=data_configs, mesh=mesh, dataset_weights=data_weights)
    return data_iterator


def get_tokenizer_vocab_size(cfg: DictConfig, next_multiple_of: int = 1) -> int:
    """Get the vocabulary size from the tokenizer.

    Args:
        cfg: Config assembled by Hydra.
        next_multiple_of: The vocabulary size will be increased to the next multiple of this number.

    Returns:
        The vocabulary size, increased to the next multiple of `next_multiple_of`.
    """
    assert hasattr(
        cfg.data, "tokenizer_path"
    ), "Tokenizer path is not defined in the config, cannot determine vocab size."
    tokenizer = load_tokenizer(
        cfg.data.tokenizer_path,
        add_bos=cfg.data.get("add_bos", False),
        add_eos=cfg.data.get("add_eos", False),
        hf_access_token=cfg.data.get("hf_access_token", None),
        cache_dir=cfg.data.get("hf_cache_dir", None),
    )
    vocab_size = tokenizer.vocab_size
    log_info(f"Tokenizer {cfg.data.tokenizer_path} has vocabulary size: {vocab_size}.")
    # Round up to the next multiple.
    if next_multiple_of > 1:
        vocab_size = int(math.ceil(vocab_size / next_multiple_of) * next_multiple_of)
        log_info(f"Rounded up to next multiple of {next_multiple_of}: {vocab_size}.")
    return vocab_size


def init_model_config(cfg: DictConfig, parallel: ParallelConfig) -> ModelConfig:
    """Instantiate the model configuration.

    Args:
        cfg: Config assembled by Hydra.
        parallel: Parallel configuration.

    Returns:
        Initialized model configuration.
    """
    # Update the model config with the vocabulary size.
    if cfg.model.vocab_size <= 0:
        log_info("Vocabulary size not set in config. Determining vocabulary size from tokenizer.")
        cfg.model.vocab_size = get_tokenizer_vocab_size(cfg, next_multiple_of=64)
        log_info(f"Vocabulary size: {cfg.model.vocab_size}.")
    # Update the model config with the vocabulary size.
    if cfg.model.vocab_size <= 0:
        log_info("Vocabulary size not set in config. Determining vocabulary size from tokenizer.")
        cfg.model.vocab_size = get_tokenizer_vocab_size(cfg, next_multiple_of=64)
        log_info(f"Vocabulary size: {cfg.model.vocab_size}.")
    # Define model config
    model_name = cfg.model.name.lower()
    if model_name.startswith("llama"):
        model_config = ModelConfig(
            model_class=LlamaTransformer,
            parallel=parallel,
            model_config=LlamaConfig(
                vocab_size=cfg.model.vocab_size,
                embedding_dim=cfg.model.embedding_dim,
                num_blocks=cfg.model.num_blocks,
                head_dim=cfg.model.head_dim,
                qk_norm=cfg.model.add_qk_norm,
                theta=cfg.model.theta,
                add_embedding_dropout=cfg.model.add_embedding_dropout,
                scan_blocks=cfg.model.scan_blocks,
                dtype=cfg.model.dtype,
                attention_backend=cfg.model.attention_backend,
                mask_across_document_boundaries=cfg.model.reset_at_document_boundaries,
                parallel=parallel,
            ),
        )
    elif model_name.startswith("mlstmv1"):
        xlstm_config = xLSTMLMModelConfig(
            vocab_size=cfg.model.vocab_size,
            embedding_dim=cfg.model.embedding_dim,
            num_blocks=cfg.model.num_blocks,
            context_length=cfg.model.context_length,
            tie_weights=cfg.model.tie_weights,
            add_embedding_dropout=cfg.model.add_embedding_dropout,
            add_post_blocks_norm=cfg.model.add_post_blocks_norm,
            parallel=parallel,
            scan_blocks=cfg.model.scan_blocks,
            norm_eps=cfg.model.norm_eps,
            norm_type=cfg.model.norm_type,
            init_distribution_out=cfg.model.init_distribution,
            init_distribution_embed=cfg.model.init_distribution_embed,
            logits_soft_cap=cfg.model.logits_soft_cap,
            lm_head_dtype=cfg.model.lm_head_dtype,
            dtype=cfg.model.dtype,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type=cfg.model.layer_type,
                    num_heads=cfg.model.num_heads,
                    init_distribution=cfg.model.init_distribution,
                    output_init_fn=cfg.model.output_init_fn,
                    qk_dim_factor=cfg.model.qk_dim_factor,
                    v_dim_factor=cfg.model.v_dim_factor,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype=cfg.model.gate_dtype,
                        backend=mLSTMBackendNameAndKwargs(
                            name=cfg.model.backend,
                            kwargs=dict(backend_name=cfg.model.backend_name)
                            if cfg.model.backend == "triton_kernels"
                            else dict(),
                        ),
                        igate_bias_init_range=cfg.model.igate_bias_init_range,
                        add_qk_norm=cfg.model.add_qk_norm,
                        norm_type=cfg.model.cell_norm_type,
                        norm_eps=cfg.model.cell_norm_eps,
                        gate_soft_cap=cfg.model.gate_soft_cap,
                        reset_at_document_boundaries=cfg.model.reset_at_document_boundaries,
                    ),
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=cfg.model.proj_factor,
                    act_fn=cfg.model.act_fn,
                    ff_type=cfg.model.ff_type,
                    dtype=cfg.model.dtype,
                    output_init_fn=cfg.model.output_init_fn,
                    init_distribution=cfg.model.init_distribution,
                ),
                add_post_norm=cfg.model.add_post_norm,
            ),
        )
        model_config = ModelConfig(model_class=xLSTMLMModel, parallel=parallel, model_config=xlstm_config)

    elif model_name.startswith("mlstm"):
        xlstm_config = xLSTMLMModelConfig(
            vocab_size=cfg.model.vocab_size,
            embedding_dim=cfg.model.embedding_dim,
            num_blocks=cfg.model.num_blocks,
            context_length=cfg.model.context_length,
            tie_weights=cfg.model.tie_weights,
            add_embedding_dropout=cfg.model.add_embedding_dropout,
            add_post_blocks_norm=cfg.model.add_post_blocks_norm,
            scan_blocks=cfg.model.scan_blocks,
            dtype=cfg.model.dtype,
            parallel=parallel,
            norm_eps=cfg.model.norm_eps,
            norm_type=cfg.model.norm_type,
            init_distribution_out=cfg.model.init_distribution,
            init_distribution_embed=cfg.model.init_distribution_embed,
            logits_soft_cap=cfg.model.logits_soft_cap,
            lm_head_dtype=cfg.model.lm_head_dtype,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type=cfg.model.layer_type,
                    num_heads=cfg.model.num_heads,
                    init_distribution=cfg.model.init_distribution,
                    output_init_fn=cfg.model.output_init_fn,
                    qk_dim_factor=cfg.model.qk_dim_factor,
                    v_dim_factor=cfg.model.v_dim_factor,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype=cfg.model.gate_dtype,
                        backend=mLSTMBackendNameAndKwargs(
                            name=cfg.model.backend,
                            kwargs=dict(backend_name=cfg.model.backend_name)
                            if cfg.model.backend == "triton_kernels"
                            else dict(),
                        ),
                        igate_bias_init_range=cfg.model.igate_bias_init_range,
                        add_qk_norm=cfg.model.add_qk_norm,
                        norm_type=cfg.model.cell_norm_type,
                        norm_eps=cfg.model.cell_norm_eps,
                        gate_soft_cap=cfg.model.gate_soft_cap,
                        reset_at_document_boundaries=cfg.model.reset_at_document_boundaries,
                    ),
                ),
                add_post_norm=cfg.model.add_post_norm,
            ),
        )
        model_config = ModelConfig(model_class=xLSTMLMModel, parallel=parallel, model_config=xlstm_config)
    else:
        raise ValueError(f'Unknown model, no suitable config found for "{cfg.model.name}".')

    return model_config


def init_logger_config(cfg: DictConfig) -> LoggerConfig:
    """Instantiate logger configuration.

    Args:
        cfg: Config assembled by Hydra.

    Returns:
        Instance of LoggerConfig.
    """
    # TODO: This will be changed with hydra initialization or our own Registry
    log_tools = []

    if "file_logger" in cfg.logger.loggers_to_use:
        file_logger_conig = FileLoggerConfig(
            log_dir=cfg.logger.file_logger_log_dir, config_format=cfg.logger.file_logger_config_format
        )
        log_tools.append(file_logger_conig)

    if "tb_logger" in cfg.logger.loggers_to_use:
        tensorboard_logger_config = TensorBoardLoggerConfig(
            log_dir=cfg.logger.tb_log_dir, tb_flush_secs=cfg.logger.tb_flush_secs
        )
        log_tools.append(tensorboard_logger_config)

    if "wb_logger" in cfg.logger.loggers_to_use:
        tags = cfg.logger.wb_tags
        if tags is None:
            tags = []
        if os.getenv("SLURM_JOB_ID"):
            tags.append(f"slurm_{os.getenv('SLURM_JOB_ID')}")
        wandb_logger_config = WandBLoggerConfig(
            wb_project=cfg.logger.wb_project,
            wb_entity=cfg.logger.wb_entity,
            wb_name=cfg.logger.wb_name,
            wb_tags=tags,
        )
        log_tools.append(wandb_logger_config)

    logger_config = LoggerConfig(
        log_path=cfg.logger.log_path,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        log_tools=log_tools,
        cmd_logging_name=cfg.logger.cmd_logging_name,
    )

    return logger_config


def init_scheduler_config(cfg: DictConfig, data_iterator: DataIterator) -> SchedulerConfig:
    """Instantiate scheduler configuration.

    Args:
        cfg: Config assembled by Hydra.

    Returns:
        Instance of SchedulerConfig following the provided config.
    """
    # Change decay_steps if it was not provided at runtime.
    if cfg.scheduler.decay_steps is None:
        if cfg.get("num_train_steps", 0) > 0:
            cfg.scheduler.decay_steps = cfg.num_train_steps
        elif cfg.get("num_epochs", 0) > 0 and len(data_iterator) > 0:
            cfg.scheduler.decay_steps = len(data_iterator) * cfg.num_epochs
        else:
            raise ValueError(
                "If decay steps are not set, num_train_steps or num_epochs need to be provided in the trainer."
            )

    # Insert the decay_steps into the scheduler config and instantiate it
    scheduler_config = SchedulerConfig(**cfg.scheduler)

    return scheduler_config


def init_optimizer_config(cfg: DictConfig) -> OptimizerConfig:
    """Instantiate optimizer configuration.

    Args:
        cfg: Full Hydra config.

    Returns:
        Instance of OptimizerConfig.
    """
    optimizer_config = OptimizerConfig(**cfg.optimizer)

    return optimizer_config


def init_model_checkpointing(cfg: DictConfig) -> ModelCheckpointConfig:
    """Instantiate model checkpointing configuration.

    Args:
        cfg: Full Hydra config.

    Returns:
        Instance of ModelCheckpointConfig.
    """
    model_checkpointing = ModelCheckpointConfig(**cfg.checkpointing)

    return model_checkpointing


def init_lr_monitor_config(cfg: DictConfig) -> LearningRateMonitorConfig:
    """Instantiate learning rate monitor configuration.

    Args:
        cfg: Full Hydra config.

    Returns:
        Instance of LearningRateMonitorConfig.
    """
    lr_monitor_config = LearningRateMonitorConfig(**cfg.lr_monitor)

    return lr_monitor_config


def init_profiler_config(cfg: DictConfig) -> JaxProfilerConfig:
    """Instantiate profiler configuration.

    Args:
        cfg: Full Hydra config.

    Returns:
        Instance of JaxProfilerConfig.
    """
    profiler_config = JaxProfilerConfig(**cfg.profiling)

    return profiler_config


def init_trainer(cfg: DictConfig, data_iterator: DataIterator, model_config: ModelConfig, mesh: Mesh) -> LLMTrainer:
    """Initializes the LLMTrainer with all sub-configs.

    Args:
        cfg: Full Hydra config.
        data_iterator: A data iterator.
        model_config: A model config.
        mesh: A device mesh.

    Returns:
        Instance of LLM trainer.
    """
    # Instantiate logger config.
    logger_config = init_logger_config(cfg=cfg)

    # Instantiate scheduler config.
    scheduler_config = init_scheduler_config(cfg=cfg, data_iterator=data_iterator)

    # Insert scheduler config into optimizer config
    cfg.optimizer.scheduler = scheduler_config

    # Instantiate optimizer config
    optimizer_config = init_optimizer_config(cfg=cfg)

    # Instantiate model checkpointing config.
    model_checkpointing = init_model_checkpointing(cfg=cfg)

    # Instantiate learning rate monitor config.
    lr_monitor_config = init_lr_monitor_config(cfg=cfg)

    # Instantiate profiler config.
    profiler_config = init_profiler_config(cfg=cfg)

    # Finally, create the trainer with all sub-configs.
    log_info("Creating trainer.")
    trainer_hparams = cfg.trainer.copy()
    del trainer_hparams.logger
    del trainer_hparams.callbacks
    # Convert lists from omegaconf to tuples for the trainer.
    trainer_hparams = OmegaConf.to_container(trainer_hparams, resolve=True, enum_to_str=True)
    trainer = LLMTrainer(
        trainer_config=LLMTrainerConfig(
            callbacks=(model_checkpointing, lr_monitor_config, profiler_config),
            logger=logger_config,
            **trainer_hparams,
        ),
        model_config=model_config,
        optimizer_config=optimizer_config,
        batch=LLMBatch.get_dtype_struct(cfg.global_batch_size, cfg.context_length),
        mesh=mesh,
    )

    return trainer


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)
