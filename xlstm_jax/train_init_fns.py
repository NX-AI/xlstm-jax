import logging
import math
import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from omegaconf import DictConfig, OmegaConf

from xlstm_jax.dataset import (
    DataIterator,
    GrainArrayRecordsDataConfig,
    HFHubDataConfig,
    HFLocalDataConfig,
    LLMBatch,
    SyntheticDataConfig,
    create_data_iterator,
    load_tokenizer,
)
from xlstm_jax.distributed.mesh_utils import initialize_mesh
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


def init_data_iterator(cfg: DictConfig, mesh: Mesh) -> tuple[DataIterator, DataIterator | None]:
    """Initialize data iterators for training and validation.

    Args:
        cfg: Config assembled by Hydra.
        mesh: The jax device mesh.

    Returns:
        Training and validation data iterators.
    """

    # create data iterator TODO: this is not working since we have no instantiated SyntheticDataConfig here.
    # explicit data config here. maybe that will be solved by richard?
    if cfg.data.data_config_type == "synthetic":
        # Delete the data_config_type key from the config since it's not needed anymore
        # and would cause an error when creating the SyntheticDataConfig.
        del cfg.data.data_config_type

        # Create the data config object.
        data_config = SyntheticDataConfig(**cfg.data)

    elif cfg.data.data_config_type == "huggingface_hub":
        # Delete the data_config_type key from the config since it's not needed anymore
        # and would cause an error when instantiating the HFHubDataConfig.
        del cfg.data.data_config_type

        # Create the data config object.
        data_config = HFHubDataConfig(**cfg.data)

    elif cfg.data.data_config_type == "huggingface_local":
        # Delete the data_config_type key from the config since it's not needed anymore
        # and would cause an error when instantiating the HFLocalDataConfig.
        del cfg.data.data_config_type

        # Create the data config object.
        data_config = HFLocalDataConfig(**cfg.data)

    elif cfg.data.data_config_type == "grain_arrayrecord":
        # Delete the data_config_type key from the config since it's not needed anymore
        # and would cause an error when instantiating the GrainArrayRecordsDataConfig.
        del cfg.data.data_config_type

        # Create the data config object.
        data_config = GrainArrayRecordsDataConfig(**cfg.data)

    else:
        raise NotImplementedError("Only synthetic, Huggingface, and ArrayRecord datasets are implemented.")

    # Create data iterators
    data_iterator, eval_data_iterator = create_data_iterator(config=data_config, mesh=mesh)

    return data_iterator, eval_data_iterator


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
        add_bos=cfg.data.add_bos,
        add_eos=cfg.data.add_eos,
        hf_access_token=cfg.data.hf_access_token,
        cache_dir=cfg.data.hf_cache_dir,
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
                add_embedding_dropout=cfg.model.add_embedding_dropout,
                scan_blocks=cfg.model.scan_blocks,
                dtype=getattr(jnp, cfg.model.dtype),
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
            lm_head_dtype=getattr(jnp, cfg.model.lm_head_dtype),
            dtype=getattr(jnp, cfg.model.dtype),
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type=cfg.model.layer_type,
                    num_heads=cfg.model.num_heads,
                    init_distribution=cfg.model.init_distribution,
                    output_init_fn=cfg.model.output_init_fn,
                    qk_dim_factor=cfg.model.qk_dim_factor,
                    v_dim_factor=cfg.model.v_dim_factor,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype=getattr(jnp, cfg.model.gate_dtype),
                        backend=mLSTMBackendNameAndKwargs(name=cfg.model.backend),
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
                    dtype=getattr(jnp, cfg.model.dtype),
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
            parallel=parallel,
            norm_eps=cfg.model.norm_eps,
            norm_type=cfg.model.norm_type,
            init_distribution_out=cfg.model.init_distribution,
            init_distribution_embed=cfg.model.init_distribution_embed,
            logits_soft_cap=cfg.model.logits_soft_cap,
            lm_head_dtype=getattr(jnp, cfg.model.lm_head_dtype),
            dtype=getattr(jnp, cfg.model.dtype),
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    layer_type=cfg.model.layer_type,
                    num_heads=cfg.model.num_heads,
                    init_distribution=cfg.model.init_distribution,
                    output_init_fn=cfg.model.output_init_fn,
                    qk_dim_factor=cfg.model.qk_dim_factor,
                    v_dim_factor=cfg.model.v_dim_factor,
                    mlstm_cell=mLSTMCellConfig(
                        gate_dtype=getattr(jnp, cfg.model.gate_dtype),
                        backend=mLSTMBackendNameAndKwargs(name=cfg.model.backend),
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
        log_path=cfg.logger.log_path, log_every_n_steps=cfg.logger.log_every_n_steps, log_tools=log_tools
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
    trainer = LLMTrainer(
        trainer_config=LLMTrainerConfig(
            callbacks=(model_checkpointing, lr_monitor_config, profiler_config),
            logger=logger_config,
            check_val_every_n_steps=cfg.trainer.check_val_every_n_steps,
            enable_progress_bar=cfg.trainer.enable_progress_bar,
            check_for_nan=cfg.trainer.check_for_nan,
            log_grad_norm=cfg.trainer.log_grad_norm,
            log_grad_norm_per_param=cfg.trainer.log_grad_norm_per_param,
            log_param_norm=cfg.trainer.log_param_norm,
            log_param_norm_per_param=cfg.trainer.log_param_norm_per_param,
            default_train_log_modes=cfg.trainer.default_train_log_modes,
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


def main_train(cfg: DictConfig):
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = init_parallel(cfg=cfg)

    # Initialize device mesh
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")

    log_info(f"Devices: {jax.devices()}.")

    # Compute global batch size.
    global_batch_size = cfg.batch_size_per_device * len(jax.devices())
    cfg.global_batch_size = global_batch_size

    # Create data iterator.
    log_info("Creating data iterator.")
    data_iterator, eval_data_iterator = init_data_iterator(cfg=cfg, mesh=mesh)

    # Instatiate model config.
    model_config = init_model_config(cfg=cfg, parallel=parallel)

    # Instantiate trainer.
    trainer = init_trainer(cfg=cfg, data_iterator=data_iterator, model_config=model_config, mesh=mesh)

    # Save resolved config to output directory
    if jax.process_index() == 0:
        output_dir = cfg.logger.log_path
        with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f, resolve=True)

    # Start training
    log_info("Training model.")
    train_kwargs = {}
    if cfg.get("num_train_steps", None):
        train_kwargs["num_train_steps"] = cfg.num_train_steps
    elif cfg.get("num_epochs", None):
        train_kwargs["num_epochs"] = cfg.num_epochs
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        **train_kwargs,
    )
    log_info(f"Final metrics: {final_metrics}.")

    return final_metrics
