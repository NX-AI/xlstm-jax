#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
import logging
from pathlib import Path

import jax

from xlstm_jax.dataset import HFHubDataConfig, LLMBatch, create_data_iterator
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig, WandBLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

set_XLA_flags()  # Must be executed before any JAX operation.

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)


def main_train(args: argparse.Namespace):
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=(),
        fsdp_min_weight_size=2**8,
        # remat=("mLSTMBlock"),
        remat=(),
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")
    assert len(jax.devices(backend="gpu")) > 0, "No devices found. This script should be run on GPU support."
    log_info(f"Devices: {jax.devices()}")

    # General hyperparameters.
    batch_size = 16 * len(jax.devices())
    context_length = 2048
    num_epochs = 10
    dtype = "bfloat16"
    lr = 1e-3
    log_path = Path(args.log_dir)

    # Create data iterator.
    log_info("Creating data iterator.")
    train_config, eval_config = HFHubDataConfig.create_train_eval_configs(
        global_batch_size=batch_size,
        max_target_length=context_length,
        hf_path="Salesforce/wikitext",
        hf_data_dir="wikitext-103-raw-v1",
        hf_cache_dir="/nfs-gpu/xlstm/data/hf_cache",
        data_column="text",
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=True,
        add_eos=True,
    )

    train_data_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_data_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Define model config - 120M parameters.
    xlstm_config = xLSTMLMModelConfig(
        vocab_size=50304,
        embedding_dim=768,
        num_blocks=12,
        context_length=2048,
        tie_weights=False,
        add_embedding_dropout=False,
        add_post_blocks_norm=True,
        parallel=parallel,
        scan_blocks=False,
        dtype=dtype,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=4,
            )
        ),
    )

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=[
                ModelCheckpointConfig(
                    every_n_epochs=5,
                    monitor="perplexity",
                    max_to_keep=1,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
                LearningRateMonitorConfig(
                    every_n_steps=20,
                    every_n_epochs=-1,
                    main_process_only=True,
                ),
                JaxProfilerConfig(
                    profile_every_n_minutes=60,
                ),
            ],
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=20,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                    WandBLoggerConfig(
                        wb_project="xlstm_jax",
                        wb_entity="xlstm",
                        wb_name=f"wikitext103_120M_{dtype}_gbs{int(batch_size)}_ctx{context_length}_lr{lr}",
                        wb_tags=["wikitext103", "120M", "reproduction"],
                    ),
                ],
            ),
            check_val_every_n_epoch=1,
            enable_progress_bar=False,
            check_for_nan=True,
            log_grad_norm=True,
            log_grad_norm_per_param=False,
            log_param_norm=True,
            log_param_norm_per_param=False,
            default_train_log_modes=["mean", "std", "max"],
            log_logit_stats=True,
            log_intermediates=True,
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
                lr=lr,
                decay_steps=len(train_data_iterator) * num_epochs,
                end_lr_factor=0.1,
                warmup_steps=1_000,
                cooldown_steps=1_000,
            ),
            grad_clip_norm=1.0,
            weight_decay=0.1,
            weight_decay_include=[r".*kernel"],
            beta2=0.95,
            eps=1e-5,
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    log_info("Training model.")
    final_metrics = trainer.train_model(
        train_loader=train_data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=num_epochs,
    )
    log_info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xLSTM model on WikiText-103 dataset.")
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()
    main_train(args)
