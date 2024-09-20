import os

from xlstm_jax.distributed.xla_utils import simulate_CPU_devices

os.environ["JAX_PLATFORMS"] = "cpu"  # or "gpu"
if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from xlstm_jax.dataset import LLMBatch, SyntheticDataConfig, create_data_iterator
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


def main_train(args: argparse.Namespace):
    # Initialize mesh.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=2**8,
        fsdp_axis_size=2,
        model_axis_size=2,
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel_config=parallel)

    # General hyperparameters.
    batch_size = 8
    context_length = 32
    num_epochs = 2
    log_path = Path(args.log_dir)

    # Create data iterator.
    data_config = SyntheticDataConfig(
        global_batch_size=32,
        max_target_length=context_length,
        data_shuffle_seed=42,
        num_train_batches=250,
        num_val_batches=50,
    )
    data_iterator, eval_data_iterator = create_data_iterator(config=data_config, mesh=mesh)

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
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                proj_factor=2.0,
                conv1d_kernel_size=4,
                num_heads=4,
                dropout=0.2,
                embedding_dim=128,
                context_length=context_length,
            )
        ),
    )

    # Create trainer with sub-configs.
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
            check_val_every_n_steps=100,
            check_val_every_n_epoch=1,
        ),
        ModelConfig(
            model_class=xLSTMLMModel,
            parallel=parallel,
            model_config=xlstm_config,
        ),
        OptimizerConfig(
            name="adam",
            scheduler=SchedulerConfig(
                name="constant",
                lr=1e-4,
            ),
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    # Train model.
    final_metrics = trainer.train_model(
        train_loader=data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=2,
    )
    print("Final metrics", final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xLSTM model on a synthetic dataset.")
    parser.add_argument("--log_dir", type=str, default="/tmp/train_synthetic")
    args = parser.parse_args()
    main_train(args)
