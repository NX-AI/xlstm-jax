import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp

from xlstm_jax.dataset import HFDataConfig, LLMBatch, create_data_iterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer import TrainerConfig
from xlstm_jax.trainer.callbacks.checkpointing import ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer
from xlstm_jax.trainer.logger import LoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

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
        fsdp_axis_size=1,
        model_axis_size=1,
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")
    assert len(jax.devices()) > 0, "No devices found. This script should be run on GPU support."
    log_info(f"Devices: {jax.devices()}")

    # General hyperparameters.
    batch_size = 16 * len(jax.devices())
    context_length = 2048
    num_epochs = 10
    log_path = Path(args.log_dir)

    # Create data iterator.
    log_info("Creating data iterator.")
    data_config = HFDataConfig(
        num_train_epochs=num_epochs,
        global_batch_size=batch_size,
        max_target_length=context_length,
        hf_path="Salesforce/wikitext",
        hf_data_dir="wikitext-103-raw-v1",
        hf_cache_dir="/nfs-gpu/xlstm/data/hf_cache",
        train_data_column="text",
        eval_data_column="text",
        tokenize_train_data=True,
        tokenize_eval_data=True,
        tokenizer_path="gpt2",
        data_shuffle_seed=42,
        add_bos=True,
        add_eos=True,
    )
    data_iterator, eval_data_iterator = create_data_iterator(config=data_config, mesh=mesh)

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
        dtype=jnp.bfloat16,
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=4,
            )
        ),
    )

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        TrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
            ),
            logger=LoggerConfig(log_path=log_path),
            check_val_every_n_steps=5_000,
            check_val_every_n_epoch=1,
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
                warmup_steps=1_000,
                cooldown_steps=1_000,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    log_info("Training model.")
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=num_epochs,
    )
    print("Final metrics", final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xLSTM model on WikiText-103 dataset.")
    parser.add_argument("--log_dir", type=str, default="/nfs-gpu/xlstm/logs/outputs/xlstm-jax/wikitext103")
    args = parser.parse_args()
    main_train(args)
