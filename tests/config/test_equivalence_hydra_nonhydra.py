from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize

from xlstm_jax.dataset import LLMBatch, SyntheticDataConfig, create_data_iterator
from xlstm_jax.define_hydra_schemas import register_configs
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.main_train import main_train
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.cell import mLSTMCellConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig


def train_without_hydra(tmpdir):
    """Trains a model without Hydra."""
    # Initialize mesh.
    parallel = ParallelConfig(
        fsdp_min_weight_size=pytest.num_devices,
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
        pipeline_axis_name="pp",
        fsdp_modules=["Embed", "LMHead", "mLSTMBlock"],
    )
    mesh = initialize_mesh(parallel_config=parallel)

    # General hyperparameters.
    batch_size_per_device = 2
    batch_size = batch_size_per_device * pytest.num_devices
    context_length = 32
    log_path = tmpdir
    num_epochs = 1

    # Create data iterator.
    train_config, eval_config = SyntheticDataConfig.create_train_eval_configs(
        train_kwargs=dict(num_batches=252),
        eval_kwargs=dict(num_batches=53),
        global_batch_size=batch_size,
        max_target_length=context_length,
        data_shuffle_seed=42,
    )
    data_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_data_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Define model config - tiny xLSTM.
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=20,
        embedding_dim=128,
        num_blocks=2,
        context_length=context_length,
        tie_weights=False,
        add_embedding_dropout=True,
        add_post_blocks_norm=True,
        parallel=parallel,
        dtype=jnp.float32,
        norm_type="rmsnorm",
        logits_soft_cap=30.0,
        lm_head_dtype=jnp.float32,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=4,
                mlstm_cell=mLSTMCellConfig(gate_dtype=jnp.float32, gate_soft_cap=15.0, igate_bias_init_range=0.0),
            )
        ),
    )

    # Create trainer with sub-configs.
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
                LearningRateMonitorConfig(),
                JaxProfilerConfig(),
            ),
            logger=LoggerConfig(
                log_path=log_path,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                ],
            ),
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        OptimizerConfig(
            name="adamw",
            scheduler=SchedulerConfig(
                name="exponential_decay",
                lr=1e-3,
                decay_steps=len(data_iterator) * num_epochs,
                end_lr_factor=0.1,
                warmup_steps=20,
                cooldown_steps=10,
            ),
            grad_clip_norm=0.5,
            weight_decay=0.1,
            weight_decay_include=[r".*kernel"],
            beta2=0.99,
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    # Train model.
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=num_epochs,
    )

    return final_metrics


def train_with_hydra(tmpdir):
    """Sets up the config via Hydra and calls regular main_train function."""
    register_configs()

    context_length = 32
    batch_size_per_device = 2
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"batch_size_per_device={batch_size_per_device}",
                f"context_length={context_length}",
                f"log_path={tmpdir}",
                "num_epochs=1",
                f"parallel.fsdp_min_weight_size={pytest.num_devices}",
                "parallel.fsdp_axis_size=1",
                "parallel.model_axis_size=1",
                "parallel.data_axis_size=-1",
                f"data.global_batch_size={pytest.num_devices * 2}",
                f"data.max_target_length={context_length}",
                "data.data_shuffle_seed=42",
                "data.num_train_batches=252",
                "data.num_val_batches=53",
                "model.vocab_size=20",
                "model.embedding_dim=128",
                "model.num_blocks=2",
                f"model.context_length={context_length}",
                "model.tie_weights=False",
                "model.add_embedding_dropout=True",
                "model.add_post_blocks_norm=True",
                "model.dtype=float32",
                "model.num_heads=4",
                "model.gate_dtype=float32",
                "model.backend=parallel_stabilized",
                "model.lm_head_dtype=float32",
                "checkpointing.monitor=perplexity",
                "checkpointing.max_to_keep=4",
                "checkpointing.save_optimizer_state=True",
                "checkpointing.enable_async_checkpointing=True",
                "scheduler.lr=1e-3",
                "scheduler.end_lr_factor=0.1",
                "scheduler.warmup_steps=20",
                "scheduler.cooldown_steps=10",
                "optimizer.beta2=0.99",
                "logger.loggers_to_use=[file_logger]",
                "task_name=unit_test",
                "hydra.job.num=0",
            ],
            return_hydra_config=True,
        )

        # Call main train function. We need to set the output_dir and log_path manually because
        # Hydra does not fill the hydra configuration when composing the config compared to using
        # the @hydra.main decorator. Is only relevant for the unit test.
        cfg.hydra.runtime.output_dir = tmpdir
        cfg.logger.log_path = tmpdir
        final_metrics = main_train(cfg=cfg)

        return final_metrics


def test_configs_equivalence(tmpdir):
    """Test if the same config with hydra and without hydra produce the same results."""
    metrics_hydra = train_with_hydra(Path(tmpdir / "hydra"))
    metrics_nonhydra = train_without_hydra(Path(tmpdir / "nonhydra"))
    np.testing.assert_allclose(
        metrics_hydra["val_epoch_1"]["loss"], metrics_nonhydra["val_epoch_1"]["loss"], rtol=1e-3, atol=1e-3
    )
