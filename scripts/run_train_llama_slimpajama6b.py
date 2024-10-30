import argparse
import logging
from pathlib import Path

import jax

from xlstm_jax.dataset import HFLocalDataConfig, LLMBatch, create_data_iterator
from xlstm_jax.distributed import set_XLA_flags
from xlstm_jax.distributed.mesh_utils import initialize_mesh
from xlstm_jax.models import ModelConfig
from xlstm_jax.models.configs import ParallelConfig
from xlstm_jax.models.llama import LlamaConfig, LlamaTransformer
from xlstm_jax.trainer.callbacks import JaxProfilerConfig, LearningRateMonitorConfig, ModelCheckpointConfig
from xlstm_jax.trainer.llm.trainer import LLMTrainer, LLMTrainerConfig
from xlstm_jax.trainer.logger import FileLoggerConfig, LoggerConfig, TensorBoardLoggerConfig, WandBLoggerConfig
from xlstm_jax.trainer.optimizer import OptimizerConfig, SchedulerConfig

set_XLA_flags()  # Must be executed before any JAX operation.

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MODEL_CONFIGS = {
    "165M": {
        "model_config": lambda parallel: LlamaConfig(
            vocab_size=50304,
            embedding_dim=768,
            num_blocks=12,
            head_dim=128,
            dtype="bfloat16",
            parallel=parallel,
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("TransformerBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 1e-3,
    },
    "1.3B": {
        "model_config": lambda parallel: LlamaConfig(
            vocab_size=50304,
            embedding_dim=2048,
            num_blocks=24,
            head_dim=128,
            dtype="bfloat16",
            parallel=parallel,
        ),
        "batch_size_per_device": 16,
        "gradient_accumulate_steps": 1,
        "fsdp_modules": (),
        "remat": ("TransformerBlock",),
        "data_axis_size": -1,
        "fsdp_axis_size": 1,
        "lr": 1e-3,
    },
}


def log_info(msg: str):
    if jax.process_index() == 0:
        LOGGER.info(msg)


def main_train(args: argparse.Namespace):
    # Config
    global_model_config = MODEL_CONFIGS[args.model]
    # Create mesh. Needs to be done before any JAX operation due to distribute initialize.
    parallel = ParallelConfig(
        data_axis_name="dp",
        fsdp_axis_name="fsdp",
        model_axis_name="tp",
        pipeline_axis_name="pp",
        fsdp_modules=global_model_config["fsdp_modules"],
        fsdp_gather_dtype="bfloat16",
        fsdp_min_weight_size=2**18,
        remat=global_model_config["remat"],
        fsdp_axis_size=global_model_config["fsdp_axis_size"],
        model_axis_size=global_model_config.get("model_axis_size", 1),
        data_axis_size=global_model_config["data_axis_size"],
        tp_async_dense=False,
    )
    mesh = initialize_mesh(parallel_config=parallel)
    log_info("Mesh initialized.")
    assert len(jax.devices(backend="gpu")) > 0, "No devices found. This script should be run on GPU."
    log_info(f"Devices: {jax.devices()}")

    # General hyperparameters.
    batch_size = global_model_config["batch_size_per_device"] * len(jax.devices())
    context_length = 2048
    num_train_steps = 300_000
    lr = global_model_config.get("lr", 1e-3)
    log_path = Path(args.log_dir)
    base_data_path = Path("/nfs-gpu/xlstm/data/hf_datasets/")

    # Create data iterator.
    log_info("Creating data iterator.")
    data_name = "600B" if args.use_full_dataset else "6B"
    data_path = base_data_path / ("cerebras_SlimPajama-627B" if args.use_full_dataset else "DKYoon_SlimPajama-6B")
    data_path = data_path / f"ctx{context_length}"
    train_config, eval_config = HFLocalDataConfig.create_train_eval_configs(
        global_batch_size=batch_size,
        data_path=data_path,
        max_target_length=context_length,
        data_column="text",
        data_shuffle_seed=123,
    )
    train_data_iterator = create_data_iterator(config=train_config, mesh=mesh)
    eval_data_iterator = create_data_iterator(config=eval_config, mesh=mesh)

    # Define model config.
    llama_config = global_model_config["model_config"](parallel=parallel)
    wb_name = f"llama_slimpajama{data_name}_{args.model}_gbs{int(batch_size)}_ctx{context_length}_lr{lr}"

    # Create trainer with sub-configs.
    log_info("Creating trainer.")
    trainer = LLMTrainer(
        LLMTrainerConfig(
            callbacks=(
                ModelCheckpointConfig(
                    every_n_epochs=1,
                    monitor="perplexity",
                    max_to_keep=1,
                    save_optimizer_state=True,
                    enable_async_checkpointing=True,
                ),
                LearningRateMonitorConfig(
                    every_n_steps=50,
                    every_n_epochs=-1,
                    main_process_only=True,
                ),
                JaxProfilerConfig(
                    profile_every_n_minutes=60,
                ),
            ),
            logger=LoggerConfig(
                log_path=log_path,
                log_every_n_steps=50,
                log_tools=[
                    FileLoggerConfig(log_dir="file_logs", config_format="json"),
                    TensorBoardLoggerConfig(log_dir="tensorboard", tb_flush_secs=10),
                    WandBLoggerConfig(
                        wb_project="xlstm_jax",
                        wb_entity="xlstm",
                        wb_name=wb_name,
                        wb_tags=[f"slimpajama{data_name}", f"llama_{args.model}", "reproduction"],
                    ),
                ],
            ),
            check_val_every_n_steps=2_000,
            enable_progress_bar=False,
            check_for_nan=True,
            log_grad_norm=True,
            log_grad_norm_per_param=True,
            log_param_norm=True,
            log_param_norm_per_param=True,
            default_train_log_modes=("mean", "std", "max"),
            log_logit_stats=True,
            log_intermediates=True,
        ),
        ModelConfig(
            model_class=LlamaTransformer,
            parallel=parallel,
            model_config=llama_config,
        ),
        OptimizerConfig(
            name="adamw",
            scheduler=SchedulerConfig(
                name="cosine_decay",
                lr=lr,
                decay_steps=num_train_steps,
                end_lr_factor=0.1,
                warmup_steps=750,
                cooldown_steps=2_000,
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

    if len(args.load_checkpoint_from) > 0:
        log_info(f"Loading checkpoint from {args.load_checkpoint_from}.")
        trainer.load_pretrained_model(
            Path(args.load_checkpoint_from),
            step_idx=-1,
            load_best=False,
            train_loader=train_data_iterator,
            val_loader=eval_data_iterator,
        )

    log_info("Training model.")
    final_metrics = trainer.train_model(
        train_loader=train_data_iterator,
        val_loader=eval_data_iterator,
        num_train_steps=num_train_steps,
    )
    log_info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama model on SlimPajama6B dataset.")
    parser.add_argument("--model", type=str, choices=MODEL_CONFIGS.keys(), default="165M")
    parser.add_argument("--log_dir", type=str, default="/nfs-gpu/xlstm/logs/outputs/xlstm-jax/llama/")
    parser.add_argument(
        "--use_full_dataset", action="store_true", help="If True, uses the 600B dataset instead of the 6B version."
    )
    parser.add_argument("--load_checkpoint_from", type=str, default="")
    args = parser.parse_args()
    main_train(args)
