import argparse
import os
from pathlib import Path

from xlstm_jax.dataset import LLMBatch, SyntheticDataConfig, create_data_iterator
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.distributed.xla_utils import simulate_CPU_devices
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.models.xlstm_parallel.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.models.xlstm_parallel.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

os.environ["JAX_PLATFORMS"] = "cpu"  # or "gpu"
if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


def main_train(args: argparse.Namespace):
    # Initialize mesh.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=("Embed", "LMHead", "mLSTMBlock"),
        fsdp_min_weight_size=NUM_DEVICES,
        fsdp_axis_size=2,
        model_axis_size=2,
        data_axis_size=-1,
    )
    mesh = initialize_mesh(parallel_config=parallel)

    # General hyperparameters.
    batch_size = 8
    context_length = 32
    log_path = Path(args.log_dir)
    num_epochs = 2

    # Create data iterator.
    train_config, eval_config = SyntheticDataConfig.create_train_eval_configs(
        train_kwargs={"num_batches": 252},
        eval_kwargs={"num_batches": 53},
        global_batch_size=32,
        max_target_length=context_length,
        data_shuffle_seed=42,
    )
    train_data_iterator = create_data_iterator(config=train_config, mesh=mesh)
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
        dtype="bfloat16",
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
        LLMTrainerConfig(
            callbacks=[
                ModelCheckpointConfig(
                    monitor="perplexity",
                    max_to_keep=4,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
                LearningRateMonitorConfig(),
                JaxProfilerConfig(),
            ],
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=20,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                ],
            ),
            check_val_every_n_steps=100,
            check_val_every_n_epoch=1,
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
                lr=1e-3,
                decay_steps=len(train_data_iterator) * num_epochs,
                end_lr_factor=0.1,
                warmup_steps=20,
                cooldown_steps=10,
            ),
            grad_clip_norm=1.0,
            weight_decay=0.1,
            weight_decay_include=[r".*kernel"],
            beta2=0.99,
            eps=1e-8,
        ),
        batch=LLMBatch.get_dtype_struct(batch_size, context_length),
        mesh=mesh,
    )

    # Train model.
    final_metrics = trainer.train_model(
        train_loader=train_data_iterator,
        val_loader=eval_data_iterator,
        num_epochs=2,
    )
    print("Final metrics", final_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train xLSTM model on a synthetic dataset.")
    parser.add_argument("--log_dir", type=str, default="/tmp/train_synthetic")
    args = parser.parse_args()
    main_train(args)
